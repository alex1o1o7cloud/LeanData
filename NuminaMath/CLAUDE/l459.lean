import Mathlib

namespace negation_of_universal_proposition_l459_45999

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end negation_of_universal_proposition_l459_45999


namespace range_of_a_l459_45954

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

theorem range_of_a (h : ¬(∃ a : ℝ, p a ∨ q a)) :
  ∃ a : ℝ, 1 < a ∧ a < 2 ∧ ∀ b : ℝ, (1 < b ∧ b < 2) → (¬(p b) ∧ ¬(q b)) :=
sorry

end range_of_a_l459_45954


namespace win_probability_comparison_l459_45953

theorem win_probability_comparison :
  let p : ℝ := 1 / 2  -- Probability of winning a single game
  let n₁ : ℕ := 4     -- Total number of games in scenario 1
  let k₁ : ℕ := 3     -- Number of wins needed in scenario 1
  let n₂ : ℕ := 8     -- Total number of games in scenario 2
  let k₂ : ℕ := 5     -- Number of wins needed in scenario 2
  
  -- Probability of winning exactly k₁ out of n₁ games
  let prob₁ : ℝ := (n₁.choose k₁ : ℝ) * p ^ k₁ * (1 - p) ^ (n₁ - k₁)
  
  -- Probability of winning exactly k₂ out of n₂ games
  let prob₂ : ℝ := (n₂.choose k₂ : ℝ) * p ^ k₂ * (1 - p) ^ (n₂ - k₂)
  
  prob₁ > prob₂ := by sorry

end win_probability_comparison_l459_45953


namespace average_tree_height_height_pattern_known_heights_l459_45904

def tree_heights : List ℝ := [8, 4, 16, 8, 32, 16]

theorem average_tree_height : 
  (tree_heights.sum / tree_heights.length : ℝ) = 14 :=
by
  sorry

theorem height_pattern (i : Fin 5) : 
  tree_heights[i] = 2 * tree_heights[i.succ] ∨ 
  tree_heights[i] = tree_heights[i.succ] / 2 :=
by
  sorry

theorem known_heights : 
  tree_heights[0] = 8 ∧ tree_heights[2] = 16 ∧ tree_heights[4] = 32 :=
by
  sorry

end average_tree_height_height_pattern_known_heights_l459_45904


namespace union_condition_intersection_condition_l459_45985

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 2}

-- Theorem for the first part
theorem union_condition (m : ℝ) : A ∪ B m = A → m = 1 := by sorry

-- Theorem for the second part
theorem intersection_condition (m : ℝ) : A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3} → m = 2 := by sorry

end union_condition_intersection_condition_l459_45985


namespace tangent_line_to_parabola_l459_45955

/-- If a line x - y - 1 = 0 is tangent to a parabola y = ax², then a = 1/4 -/
theorem tangent_line_to_parabola (a : ℝ) : 
  (∃ x y : ℝ, x - y - 1 = 0 ∧ y = a * x^2 ∧ 
   ∀ x' y' : ℝ, x' - y' - 1 = 0 → y' ≥ a * x'^2) → 
  a = 1/4 := by
sorry

end tangent_line_to_parabola_l459_45955


namespace sector_area_l459_45927

theorem sector_area (angle : Real) (radius : Real) (area : Real) : 
  angle = 120 * (π / 180) →  -- Convert 120° to radians
  radius = 10 →
  area = (angle / (2 * π)) * π * radius^2 →
  area = 100 * π / 3 := by
sorry

end sector_area_l459_45927


namespace age_difference_proof_l459_45989

theorem age_difference_proof (hurley_age richard_age : ℕ) : 
  hurley_age = 14 →
  hurley_age + 40 + (richard_age + 40) = 128 →
  richard_age - hurley_age = 20 := by
sorry

end age_difference_proof_l459_45989


namespace soccer_match_handshakes_l459_45970

def soccer_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : ℕ :=
  let player_handshakes := team_size * team_size * (num_teams - 1) / 2
  let referee_handshakes := team_size * num_teams * num_referees
  player_handshakes + referee_handshakes

theorem soccer_match_handshakes :
  soccer_handshakes 6 2 3 = 72 := by
  sorry

end soccer_match_handshakes_l459_45970


namespace joels_age_when_dad_twice_as_old_l459_45984

theorem joels_age_when_dad_twice_as_old (joel_current_age dad_current_age : ℕ) 
  (h1 : joel_current_age = 12) 
  (h2 : dad_current_age = 47) : 
  ∃ (years : ℕ), dad_current_age + years = 2 * (joel_current_age + years) ∧ 
                 joel_current_age + years = 35 := by
  sorry

end joels_age_when_dad_twice_as_old_l459_45984


namespace range_of_a_l459_45975

-- Define the propositions P and Q as functions of a
def P (a : ℝ) : Prop := ∀ x > 0, Monotone (fun x => Real.log x / Real.log a)

def Q (a : ℝ) : Prop := ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (h1 : P a ∨ Q a) 
  (h2 : ¬(P a ∧ Q a)) : 
  a > 2 ∨ (-2 < a ∧ a ≤ 1) := by
  sorry

end range_of_a_l459_45975


namespace jeffrey_mailbox_steps_l459_45983

/-- Represents Jeffrey's walking pattern -/
structure WalkingPattern where
  forward : ℕ
  backward : ℕ

/-- Calculates the total steps taken given a walking pattern and distance -/
def totalSteps (pattern : WalkingPattern) (distance : ℕ) : ℕ :=
  distance * (pattern.forward + pattern.backward) / (pattern.forward - pattern.backward)

/-- Theorem: Jeffrey takes 330 steps to reach the mailbox -/
theorem jeffrey_mailbox_steps :
  let pattern : WalkingPattern := { forward := 3, backward := 2 }
  let distance : ℕ := 66
  totalSteps pattern distance = 330 := by
  sorry

#eval totalSteps { forward := 3, backward := 2 } 66

end jeffrey_mailbox_steps_l459_45983


namespace positive_numbers_properties_l459_45978

theorem positive_numbers_properties (a b : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_lt : a < b) (h_sum : a + b = 2) : 
  (1 < b ∧ b < 2) ∧ (Real.sqrt a + Real.sqrt b < 2) := by
  sorry

end positive_numbers_properties_l459_45978


namespace equation_equivalence_l459_45930

theorem equation_equivalence (x : ℝ) : 
  (1 / 2 - (x - 1) / 3 = 1) ↔ (3 - 2 * (x - 1) = 6) := by
  sorry

end equation_equivalence_l459_45930


namespace happy_valley_kennel_arrangement_l459_45949

/-- The number of chickens -/
def num_chickens : ℕ := 4

/-- The number of dogs -/
def num_dogs : ℕ := 3

/-- The number of cats -/
def num_cats : ℕ := 5

/-- The number of empty cages -/
def num_empty_cages : ℕ := 2

/-- The total number of entities (animals + empty cages) -/
def total_entities : ℕ := num_chickens + num_dogs + num_cats + num_empty_cages

/-- The number of animal groups -/
def num_groups : ℕ := 3

/-- The number of possible positions for empty cages -/
def num_positions : ℕ := num_groups + 2

theorem happy_valley_kennel_arrangement :
  (Nat.factorial num_groups) * (Nat.choose num_positions num_empty_cages) *
  (Nat.factorial num_chickens) * (Nat.factorial num_dogs) * (Nat.factorial num_cats) = 1036800 := by
  sorry

end happy_valley_kennel_arrangement_l459_45949


namespace bashers_win_probability_l459_45968

def probability_at_least_4_out_of_5 (p : ℝ) : ℝ :=
  5 * p^4 * (1 - p) + p^5

theorem bashers_win_probability :
  probability_at_least_4_out_of_5 (4/5) = 3072/3125 := by
  sorry

end bashers_win_probability_l459_45968


namespace quadrilateral_diagonal_lengths_l459_45917

/-- A quadrilateral with side lengths 7, 9, 15, and 10 has 10 possible whole number lengths for its diagonal. -/
theorem quadrilateral_diagonal_lengths :
  ∃ (lengths : Finset ℕ),
    (Finset.card lengths = 10) ∧
    (∀ x ∈ lengths,
      (x + 7 > 9) ∧ (x + 9 > 7) ∧ (7 + 9 > x) ∧
      (x + 10 > 15) ∧ (x + 15 > 10) ∧ (10 + 15 > x) ∧
      (x ≥ 6) ∧ (x ≤ 15)) :=
by sorry

end quadrilateral_diagonal_lengths_l459_45917


namespace geometric_sequence_sum_l459_45976

-- Define a geometric sequence with positive terms
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Main theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 10 + a 3 * a 5 = 26 →
  a 5 * a 7 = 5 →
  a 4 + a 8 = 6 := by
sorry

end geometric_sequence_sum_l459_45976


namespace thirtieth_term_value_l459_45912

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 30th term of the specific arithmetic sequence -/
def thirtieth_term : ℝ := arithmetic_sequence 3 4 30

theorem thirtieth_term_value : thirtieth_term = 119 := by
  sorry

end thirtieth_term_value_l459_45912


namespace kittens_from_friends_proof_l459_45928

/-- The number of kittens Joan's cat had initially -/
def initial_kittens : ℕ := 8

/-- The total number of kittens Joan has now -/
def total_kittens : ℕ := 10

/-- The number of kittens Joan got from her friends -/
def kittens_from_friends : ℕ := total_kittens - initial_kittens

theorem kittens_from_friends_proof :
  kittens_from_friends = total_kittens - initial_kittens :=
by sorry

end kittens_from_friends_proof_l459_45928


namespace unique_factorial_solution_l459_45959

theorem unique_factorial_solution : ∃! n : ℕ, n * n.factorial + 2 * n.factorial = 5040 := by
  sorry

end unique_factorial_solution_l459_45959


namespace employee_pay_calculation_l459_45997

def total_pay : ℝ := 550
def a_percentage : ℝ := 1.2

theorem employee_pay_calculation (b_pay : ℝ) (h1 : b_pay > 0) 
  (h2 : b_pay + a_percentage * b_pay = total_pay) : b_pay = 250 := by
  sorry

end employee_pay_calculation_l459_45997


namespace pure_imaginary_iff_m_eq_three_first_quadrant_iff_m_lt_neg_two_or_gt_three_l459_45909

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 2*m - 3) (m^2 + 3*m + 2)

-- Theorem 1: z is a pure imaginary number iff m = 3
theorem pure_imaginary_iff_m_eq_three (m : ℝ) :
  z m = Complex.I * (z m).im ↔ m = 3 := by sorry

-- Theorem 2: z is in the first quadrant iff m < -2 or m > 3
theorem first_quadrant_iff_m_lt_neg_two_or_gt_three (m : ℝ) :
  (z m).re > 0 ∧ (z m).im > 0 ↔ m < -2 ∨ m > 3 := by sorry

end pure_imaginary_iff_m_eq_three_first_quadrant_iff_m_lt_neg_two_or_gt_three_l459_45909


namespace h2o_required_for_reaction_l459_45965

-- Define the chemical reaction
def chemical_reaction (NaH H2O NaOH H2 : ℕ) : Prop :=
  NaH = H2O ∧ NaH = NaOH ∧ NaH = H2

-- Define the problem statement
theorem h2o_required_for_reaction (NaH : ℕ) (h : NaH = 2) :
  ∃ H2O : ℕ, chemical_reaction NaH H2O NaH NaH ∧ H2O = 2 :=
by sorry

end h2o_required_for_reaction_l459_45965


namespace one_third_of_recipe_flour_l459_45962

theorem one_third_of_recipe_flour (original_flour : ℚ) (reduced_flour : ℚ) : 
  original_flour = 17/3 → reduced_flour = original_flour / 3 → reduced_flour = 17/9 := by
  sorry

#check one_third_of_recipe_flour

end one_third_of_recipe_flour_l459_45962


namespace final_fish_count_l459_45980

def fish_count (initial : ℕ) (days : ℕ) : ℕ :=
  let day1 := initial
  let day2 := day1 * 2
  let day3 := day2 * 2 - (day2 * 2) / 3
  let day4 := day3 * 2
  let day5 := day4 * 2 - (day4 * 2) / 4
  let day6 := day5 * 2
  let day7 := day6 * 2 + 15
  day7

theorem final_fish_count :
  fish_count 6 7 = 207 :=
by sorry

end final_fish_count_l459_45980


namespace babysitting_earnings_l459_45974

/-- Calculates the amount earned for a given hour of babysitting -/
def hourlyRate (hour : ℕ) : ℕ :=
  2 * (hour % 6 + 1)

/-- Calculates the total amount earned for a given number of hours -/
def totalEarned (hours : ℕ) : ℕ :=
  (List.range hours).map hourlyRate |>.sum

theorem babysitting_earnings : totalEarned 48 = 288 := by
  sorry

end babysitting_earnings_l459_45974


namespace trapezoid_side_length_l459_45973

/-- Given a trapezoid PQRS with specified dimensions, prove the length of QR -/
theorem trapezoid_side_length (area : ℝ) (altitude PQ RS : ℝ) (h1 : area = 210)
  (h2 : altitude = 10) (h3 : PQ = 12) (h4 : RS = 21) :
  ∃ QR : ℝ, QR = 21 - 0.5 * (Real.sqrt 44 + Real.sqrt 341) :=
by sorry

end trapezoid_side_length_l459_45973


namespace church_attendance_l459_45963

/-- The total number of people in the church is the sum of children, male adults, and female adults. -/
theorem church_attendance (children : ℕ) (male_adults : ℕ) (female_adults : ℕ) 
  (h1 : children = 80) 
  (h2 : male_adults = 60) 
  (h3 : female_adults = 60) : 
  children + male_adults + female_adults = 200 := by
  sorry

#check church_attendance

end church_attendance_l459_45963


namespace inequalities_theorem_l459_45996

theorem inequalities_theorem (a b c d : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > -a) (h4 : c < d) (h5 : d < 0) : 
  (a / b + b / c < 0) ∧ (a - c > b - d) ∧ (a * (d - c) > b * (d - c)) := by
  sorry

end inequalities_theorem_l459_45996


namespace expected_sixes_is_half_l459_45982

-- Define the number of dice
def num_dice : ℕ := 3

-- Define the probability of rolling a 6 on one die
def prob_six : ℚ := 1 / 6

-- Define the expected number of 6's
def expected_sixes : ℚ := num_dice * prob_six

-- Theorem statement
theorem expected_sixes_is_half : expected_sixes = 1 / 2 := by
  sorry

end expected_sixes_is_half_l459_45982


namespace parallel_vectors_k_value_l459_45900

theorem parallel_vectors_k_value (k : ℝ) :
  let a : Fin 2 → ℝ := ![k, Real.sqrt 2]
  let b : Fin 2 → ℝ := ![2, -2]
  (∃ (c : ℝ), a = c • b) →
  k = -Real.sqrt 2 := by
sorry

end parallel_vectors_k_value_l459_45900


namespace jeongyeon_height_is_142_57_l459_45943

/-- Jeongyeon's height in centimeters -/
def jeongyeon_height : ℝ := 1.06 * 134.5

/-- Theorem stating that Jeongyeon's height is 142.57 cm -/
theorem jeongyeon_height_is_142_57 : 
  jeongyeon_height = 142.57 := by sorry

end jeongyeon_height_is_142_57_l459_45943


namespace determinant_problem_l459_45906

theorem determinant_problem (x y z w : ℝ) 
  (h : x * w - y * z = 7) : 
  x * (8 * z + 4 * w) - z * (8 * x + 4 * y) = 28 := by
  sorry

end determinant_problem_l459_45906


namespace series_sum_l459_45957

theorem series_sum : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end series_sum_l459_45957


namespace two_distinct_roots_l459_45942

/-- The equation has exactly two distinct real roots if and only if a > 0 or a = -2 -/
theorem two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧
    x^2 - 6*x + (a-2)*|x-3| + 9 - 2*a = 0 ∧
    y^2 - 6*y + (a-2)*|y-3| + 9 - 2*a = 0 ∧
    (∀ z : ℝ, z^2 - 6*z + (a-2)*|z-3| + 9 - 2*a = 0 → z = x ∨ z = y)) ↔
  (a > 0 ∨ a = -2) :=
sorry

end two_distinct_roots_l459_45942


namespace g_is_zero_l459_45969

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (Real.sin x ^ 4 + 3 * Real.cos x ^ 2) - 
  Real.sqrt (Real.cos x ^ 4 + 3 * Real.sin x ^ 2)

theorem g_is_zero : ∀ x : ℝ, g x = 0 := by sorry

end g_is_zero_l459_45969


namespace walking_time_calculation_walk_two_miles_time_l459_45901

/-- Calculates the time taken to walk a given distance at a constant pace -/
theorem walking_time_calculation (distance : ℝ) (total_distance : ℝ) (total_time : ℝ) :
  (distance > 0) → (total_distance > 0) → (total_time > 0) →
  (total_distance / total_time * total_time = total_distance) →
  (distance / (total_distance / total_time) = distance * total_time / total_distance) := by
  sorry

/-- Proves that walking 2 miles takes 1 hour given the conditions -/
theorem walk_two_miles_time :
  ∃ (pace : ℝ),
    (2 : ℝ) / pace = 1 ∧
    pace * 8 = 16 := by
  sorry

end walking_time_calculation_walk_two_miles_time_l459_45901


namespace bike_speed_calculation_l459_45902

/-- Proves that given the conditions, the bike speed is 15 km/h -/
theorem bike_speed_calculation (distance : ℝ) (car_speed_multiplier : ℝ) (time_difference : ℝ) :
  distance = 15 →
  car_speed_multiplier = 4 →
  time_difference = 45 / 60 →
  ∃ (bike_speed : ℝ), 
    bike_speed > 0 ∧
    distance / bike_speed - distance / (car_speed_multiplier * bike_speed) = time_difference ∧
    bike_speed = 15 :=
by sorry

end bike_speed_calculation_l459_45902


namespace power_mod_37_l459_45951

theorem power_mod_37 (n : ℕ) (h1 : n < 37) (h2 : (6 * n) % 37 = 1) :
  (Nat.pow (Nat.pow 2 n) 4 - 3) % 37 = 35 := by
  sorry

end power_mod_37_l459_45951


namespace cos_double_angle_l459_45903

theorem cos_double_angle (α : ℝ) (a : ℝ × ℝ) :
  a = (Real.cos α, (1 : ℝ) / 2) →
  Real.sqrt ((Real.cos α)^2 + (1 / 4)^2) = Real.sqrt 2 / 2 →
  Real.cos (2 * α) = -(1 : ℝ) / 2 := by
sorry

end cos_double_angle_l459_45903


namespace power_one_third_five_l459_45932

theorem power_one_third_five : (1/3 : ℚ)^5 = 1/243 := by sorry

end power_one_third_five_l459_45932


namespace k_at_negative_eight_l459_45966

-- Define the polynomial h
def h (x : ℝ) : ℝ := x^3 - x - 2

-- Define the property that k is a cubic polynomial with the given conditions
def is_valid_k (k : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x, h x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
    (∀ x, k x = 0 ↔ x = a^2 ∨ x = b^2 ∨ x = c^2) ∧
    k 0 = 2

-- Theorem statement
theorem k_at_negative_eight (k : ℝ → ℝ) (hk : is_valid_k k) : k (-8) = -20 := by
  sorry

end k_at_negative_eight_l459_45966


namespace sin_2A_eq_sin_2B_neither_sufficient_nor_necessary_l459_45977

/-- A triangle ABC -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π

/-- Definition of an isosceles triangle -/
def is_isosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

/-- The condition sin 2A = sin 2B -/
def condition (t : Triangle) : Prop :=
  Real.sin (2 * t.A) = Real.sin (2 * t.B)

/-- The main theorem to prove -/
theorem sin_2A_eq_sin_2B_neither_sufficient_nor_necessary :
  ¬(∀ t : Triangle, condition t → is_isosceles t) ∧
  ¬(∀ t : Triangle, is_isosceles t → condition t) := by
  sorry

end sin_2A_eq_sin_2B_neither_sufficient_nor_necessary_l459_45977


namespace divisibility_of_2_pow_62_plus_1_l459_45921

theorem divisibility_of_2_pow_62_plus_1 :
  ∃ k : ℕ, 2^62 + 1 = k * (2^31 + 2^16 + 1) := by
  sorry

end divisibility_of_2_pow_62_plus_1_l459_45921


namespace bowling_balls_count_l459_45935

/-- The number of red bowling balls -/
def red_balls : ℕ := 30

/-- The difference between green and red bowling balls -/
def green_red_difference : ℕ := 6

/-- The total number of bowling balls -/
def total_balls : ℕ := red_balls + (red_balls + green_red_difference)

theorem bowling_balls_count : total_balls = 66 := by
  sorry

end bowling_balls_count_l459_45935


namespace jessica_rearrangements_time_l459_45916

def name_length : ℕ := 7
def repeated_s : ℕ := 2
def repeated_a : ℕ := 2
def rearrangements_per_minute : ℕ := 15

def total_rearrangements : ℕ := name_length.factorial / (repeated_s.factorial * repeated_a.factorial)

theorem jessica_rearrangements_time :
  (total_rearrangements : ℚ) / rearrangements_per_minute / 60 = 1.4 := by
  sorry

end jessica_rearrangements_time_l459_45916


namespace isosceles_triangle_base_lengths_l459_45994

def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∧ a + b + c = 5 ∧ (a = 2 ∨ b = 2 ∨ c = 2)) ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

theorem isosceles_triangle_base_lengths :
  ∃ (a b c : ℝ), IsoscelesTriangle a b c ∧ (c = 1.5 ∨ c = 2) := by sorry

end isosceles_triangle_base_lengths_l459_45994


namespace abc_value_l459_45919

theorem abc_value (a b c : ℝ) 
  (h1 : a^2 * (b + c) = 2011)
  (h2 : b^2 * (a + c) = 2011)
  (h3 : a ≠ b) : 
  a * b * c = -2011 := by
sorry

end abc_value_l459_45919


namespace prime_power_plus_144_square_l459_45964

theorem prime_power_plus_144_square (p : ℕ) (n : ℕ) (m : ℤ) : 
  p.Prime → n > 0 → (p : ℤ)^n + 144 = m^2 → 
  ((p = 2 ∧ n = 9 ∧ m = 36) ∨ (p = 3 ∧ n = 4 ∧ m = 15)) :=
by sorry

end prime_power_plus_144_square_l459_45964


namespace total_notes_is_133_l459_45972

/-- Calculates the sum of integers from 1 to n -/
def triangleSum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the distribution of notes on the board -/
structure NoteDistribution where
  redRowCount : ℕ
  redPerRow : ℕ
  redScattered : ℕ
  blueRowCount : ℕ
  bluePerRow : ℕ
  blueScattered : ℕ
  greenTriangleBases : List ℕ
  yellowDiagonal1 : ℕ
  yellowDiagonal2 : ℕ
  yellowHexagon : ℕ

/-- Calculates the total number of notes based on the given distribution -/
def totalNotes (dist : NoteDistribution) : ℕ :=
  let redNotes := dist.redRowCount * dist.redPerRow + dist.redScattered
  let blueNotes := dist.blueRowCount * dist.bluePerRow + dist.blueScattered
  let greenNotes := (dist.greenTriangleBases.map triangleSum).sum
  let yellowNotes := dist.yellowDiagonal1 + dist.yellowDiagonal2 + dist.yellowHexagon
  redNotes + blueNotes + greenNotes + yellowNotes

/-- The actual distribution of notes on the board -/
def actualDistribution : NoteDistribution := {
  redRowCount := 5
  redPerRow := 6
  redScattered := 3
  blueRowCount := 4
  bluePerRow := 7
  blueScattered := 12
  greenTriangleBases := [4, 5, 6]
  yellowDiagonal1 := 5
  yellowDiagonal2 := 3
  yellowHexagon := 6
}

/-- Theorem stating that the total number of notes is 133 -/
theorem total_notes_is_133 : totalNotes actualDistribution = 133 := by
  sorry

end total_notes_is_133_l459_45972


namespace cuboid_surface_area_l459_45915

theorem cuboid_surface_area 
  (x y z : ℝ) 
  (edge_sum : 4*x + 4*y + 4*z = 160) 
  (diagonal : x^2 + y^2 + z^2 = 25^2) : 
  2*(x*y + y*z + z*x) = 975 := by
sorry

end cuboid_surface_area_l459_45915


namespace simple_interest_principal_l459_45947

/-- Simple interest calculation -/
theorem simple_interest_principal
  (interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : interest = 25)
  (h2 : rate = 25 / 4)
  (h3 : time = 73 / 365)
  : (interest * 100) / (rate * time) = 2000 :=
by sorry

end simple_interest_principal_l459_45947


namespace cube_square_difference_property_l459_45958

theorem cube_square_difference_property (x : ℝ) : 
  x^3 - x^2 = (x^2 - x)^2 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
  sorry

end cube_square_difference_property_l459_45958


namespace midpoint_complex_plane_l459_45940

theorem midpoint_complex_plane (A B C : ℂ) : 
  A = 6 + 5*I ∧ B = -2 + 3*I ∧ C = (A + B) / 2 → C = 2 + 4*I :=
by sorry

end midpoint_complex_plane_l459_45940


namespace dot_product_bound_l459_45945

theorem dot_product_bound (a b c x y z : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 4) 
  (h2 : x^2 + y^2 + z^2 = 9) : 
  -6 ≤ a * x + b * y + c * z ∧ a * x + b * y + c * z ≤ 6 := by
  sorry

end dot_product_bound_l459_45945


namespace f_2011_is_zero_l459_45926

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2011_is_zero (f : ℝ → ℝ) (h_odd : is_odd f) (h_period : ∀ x, f (x + 1) = -f x) : f 2011 = 0 := by
  sorry

end f_2011_is_zero_l459_45926


namespace grandmother_age_multiple_l459_45979

def milena_age : ℕ := 7

def grandfather_age_difference (grandmother_age : ℕ) : ℕ := grandmother_age + 2

theorem grandmother_age_multiple : ∃ (grandmother_age : ℕ), 
  grandfather_age_difference grandmother_age - milena_age = 58 ∧ 
  grandmother_age = 9 * milena_age := by
  sorry

end grandmother_age_multiple_l459_45979


namespace exists_g_compose_eq_f_l459_45987

noncomputable def f (k ℓ : ℝ) (x : ℝ) : ℝ := k * x + ℓ

theorem exists_g_compose_eq_f (k ℓ : ℝ) (h : k > 0) :
  ∃ (a b : ℝ), ∀ x, f k ℓ x = f a b (f a b x) ∧ a > 0 := by
  sorry

end exists_g_compose_eq_f_l459_45987


namespace five_x_minus_six_greater_than_one_l459_45952

theorem five_x_minus_six_greater_than_one (x : ℝ) :
  (5 * x - 6 > 1) ↔ (5 * x - 6 > 1) :=
by sorry

end five_x_minus_six_greater_than_one_l459_45952


namespace relationship_abc_l459_45971

theorem relationship_abc : 
  let a : ℝ := Real.rpow 0.7 0.6
  let b : ℝ := Real.rpow 0.6 (-0.6)
  let c : ℝ := Real.rpow 0.6 0.7
  b > a ∧ a > c := by sorry

end relationship_abc_l459_45971


namespace sauteTimeRatio_l459_45934

/-- Represents the time spent on various tasks in making calzones -/
structure CalzoneTime where
  total : ℕ
  sauteOnions : ℕ
  kneadDough : ℕ

/-- Calculates the time spent sauteing garlic and peppers -/
def sauteGarlicPeppers (ct : CalzoneTime) : ℕ :=
  ct.total - (ct.sauteOnions + ct.kneadDough + 2 * ct.kneadDough + (ct.kneadDough + 2 * ct.kneadDough) / 10)

/-- Theorem stating the ratio of time spent sauteing garlic and peppers to time spent sauteing onions -/
theorem sauteTimeRatio (ct : CalzoneTime) 
    (h1 : ct.total = 124)
    (h2 : ct.sauteOnions = 20)
    (h3 : ct.kneadDough = 30) :
  4 * (sauteGarlicPeppers ct) = ct.sauteOnions := by
  sorry

#eval sauteGarlicPeppers { total := 124, sauteOnions := 20, kneadDough := 30 }

end sauteTimeRatio_l459_45934


namespace system_solution_l459_45992

theorem system_solution (a b c : ℝ) :
  ∃! (x y z : ℝ),
    (x + a * y + a^2 * z + a^3 = 0) ∧
    (x + b * y + b^2 * z + b^3 = 0) ∧
    (x + c * y + c^2 * z + c^3 = 0) ∧
    (x = -a * b * c) ∧
    (y = a * b + b * c + c * a) ∧
    (z = -(a + b + c)) :=
by sorry

end system_solution_l459_45992


namespace absolute_value_expression_l459_45988

theorem absolute_value_expression : |(|-|-2 + 3| - 2| + 2)| = 5 := by
  sorry

end absolute_value_expression_l459_45988


namespace intersection_of_M_and_N_l459_45960

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end intersection_of_M_and_N_l459_45960


namespace erin_curlers_count_l459_45939

/-- Represents the number of curlers Erin put in her hair -/
def total_curlers : ℕ := 16

/-- Represents the number of small pink curlers -/
def pink_curlers : ℕ := total_curlers / 4

/-- Represents the number of medium blue curlers -/
def blue_curlers : ℕ := 2 * pink_curlers

/-- Represents the number of large green curlers -/
def green_curlers : ℕ := 4

/-- Proves that the total number of curlers is 16 -/
theorem erin_curlers_count :
  total_curlers = pink_curlers + blue_curlers + green_curlers :=
by sorry

end erin_curlers_count_l459_45939


namespace sum_A_B_equals_one_l459_45981

theorem sum_A_B_equals_one (a : ℝ) (ha : a ≠ 1 ∧ a ≠ -1) :
  let x : ℝ := (1 - a) / (1 - 1/a)
  let y : ℝ := 1 - 1/x
  let A : ℝ := 1 / (1 - (1-x)/y)
  let B : ℝ := 1 / (1 - y/(1-x))
  A + B = 1 := by
sorry


end sum_A_B_equals_one_l459_45981


namespace square_perimeter_l459_45931

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (5 * s / 2 = 40) → (4 * s = 64) := by
  sorry

end square_perimeter_l459_45931


namespace polynomial_value_theorem_l459_45998

theorem polynomial_value_theorem (x : ℝ) : 
  x^2 - (5/2)*x = 6 → 2*x^2 - 5*x + 6 = 18 := by
  sorry

end polynomial_value_theorem_l459_45998


namespace largest_multiple_of_8_negation_greater_than_neg_200_l459_45922

theorem largest_multiple_of_8_negation_greater_than_neg_200 :
  ∀ n : ℤ, (n % 8 = 0 ∧ -n > -200) → n ≤ 192 :=
by
  sorry

end largest_multiple_of_8_negation_greater_than_neg_200_l459_45922


namespace largest_integer_with_four_digit_square_l459_45967

theorem largest_integer_with_four_digit_square : ∃ N : ℕ, 
  (∀ n : ℕ, n^2 ≥ 10000 → N ≤ n) ∧ 
  (N^2 < 10000) ∧
  (N^2 ≥ 1000) ∧
  N = 99 := by
sorry

end largest_integer_with_four_digit_square_l459_45967


namespace negation_of_existence_negation_of_exponential_inequality_l459_45925

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x := by sorry

theorem negation_of_exponential_inequality :
  (¬ ∃ x : ℝ, Real.exp x > x) ↔ (∀ x : ℝ, Real.exp x ≤ x) := by sorry

end negation_of_existence_negation_of_exponential_inequality_l459_45925


namespace difference_of_squares_l459_45991

theorem difference_of_squares (m : ℝ) : m^2 - 1 = (m + 1) * (m - 1) := by
  sorry

end difference_of_squares_l459_45991


namespace geometric_sequence_a2_l459_45918

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

theorem geometric_sequence_a2 (a : ℕ → ℚ) 
  (h1 : geometric_sequence a) 
  (h2 : a 1 + a 3 = 4) : 
  a 2 = 8/5 := by
sorry

end geometric_sequence_a2_l459_45918


namespace john_total_weight_l459_45929

/-- The total weight moved by John during his workout -/
def total_weight_moved (weight_per_rep : ℕ) (reps_per_set : ℕ) (num_sets : ℕ) : ℕ :=
  weight_per_rep * reps_per_set * num_sets

/-- Theorem stating that John moves 450 pounds in total -/
theorem john_total_weight :
  total_weight_moved 15 10 3 = 450 := by
  sorry

end john_total_weight_l459_45929


namespace cost_of_two_pans_l459_45913

/-- The cost of a single pot -/
def pot_cost : ℕ := 20

/-- The number of pots purchased -/
def num_pots : ℕ := 3

/-- The number of pans purchased -/
def num_pans : ℕ := 4

/-- The total cost of all items -/
def total_cost : ℕ := 100

/-- The cost of a single pan -/
def pan_cost : ℕ := (total_cost - num_pots * pot_cost) / num_pans

theorem cost_of_two_pans :
  2 * pan_cost = 20 :=
by sorry

end cost_of_two_pans_l459_45913


namespace monkey_climb_proof_l459_45946

/-- The height of the tree in feet -/
def tree_height : ℝ := 19

/-- The number of hours the monkey climbs -/
def climbing_hours : ℕ := 17

/-- The distance the monkey slips back each hour in feet -/
def slip_distance : ℝ := 2

/-- The distance the monkey hops each hour in feet -/
def hop_distance : ℝ := 3

theorem monkey_climb_proof :
  tree_height = (climbing_hours - 1) * (hop_distance - slip_distance) + hop_distance :=
by sorry

end monkey_climb_proof_l459_45946


namespace least_k_cube_divisible_by_2160_l459_45905

theorem least_k_cube_divisible_by_2160 : 
  ∃ k : ℕ+, (k : ℕ)^3 % 2160 = 0 ∧ ∀ m : ℕ+, (m : ℕ)^3 % 2160 = 0 → k ≤ m := by
  sorry

end least_k_cube_divisible_by_2160_l459_45905


namespace not_first_year_percentage_l459_45944

/-- Represents the percentage of associates in each category at a law firm -/
structure LawFirmAssociates where
  secondYear : ℝ
  moreThanTwoYears : ℝ

/-- Theorem stating the percentage of associates who are not first-year associates -/
theorem not_first_year_percentage (firm : LawFirmAssociates) 
  (h1 : firm.secondYear = 25)
  (h2 : firm.moreThanTwoYears = 50) :
  100 - (100 - firm.moreThanTwoYears - firm.secondYear) = 75 := by
  sorry

#check not_first_year_percentage

end not_first_year_percentage_l459_45944


namespace total_amount_is_70_l459_45956

/-- Represents the distribution of money among three people -/
structure Distribution where
  x : ℚ  -- x's share in rupees
  y : ℚ  -- y's share in rupees
  z : ℚ  -- z's share in rupees

/-- Checks if a distribution satisfies the given conditions -/
def is_valid_distribution (d : Distribution) : Prop :=
  d.y = 0.45 * d.x ∧ d.z = 0.3 * d.x ∧ d.y = 18

/-- The theorem to prove -/
theorem total_amount_is_70 (d : Distribution) :
  is_valid_distribution d → d.x + d.y + d.z = 70 := by
  sorry


end total_amount_is_70_l459_45956


namespace inscribed_circle_radius_theorem_l459_45990

noncomputable def inscribed_circle_radius (side_length : ℝ) (A₁ A₂ : ℝ) : ℝ :=
  sorry

theorem inscribed_circle_radius_theorem :
  let side_length : ℝ := 4
  let A₁ : ℝ := 8
  let A₂ : ℝ := 8
  -- Square circumscribes both circles
  side_length ^ 2 = A₁ + A₂ →
  -- Arithmetic progression condition
  A₁ + A₂ / 2 = (A₁ + (A₁ + A₂)) / 2 →
  -- Radius calculation
  inscribed_circle_radius side_length A₁ A₂ = 2 * Real.sqrt (2 / Real.pi)
  := by sorry

end inscribed_circle_radius_theorem_l459_45990


namespace transylvanian_vampire_statement_l459_45937

-- Define the possible species
inductive Species
| Human
| Vampire

-- Define the possible mental states
inductive MentalState
| Sane
| Insane

-- Define a person
structure Person where
  species : Species
  mentalState : MentalState

-- Define the statement made by the person
def madeVampireStatement (p : Person) : Prop :=
  p.mentalState = MentalState.Insane

-- Theorem statement
theorem transylvanian_vampire_statement 
  (p : Person) 
  (h : madeVampireStatement p) : 
  (∃ (s : Species), p.species = s) ∧ 
  (p.mentalState = MentalState.Insane) :=
sorry

end transylvanian_vampire_statement_l459_45937


namespace f_properties_l459_45950

-- Define the function f(x) = x|x - 2|
def f (x : ℝ) : ℝ := x * abs (x - 2)

-- Theorem for the monotonicity intervals and inequality solution
theorem f_properties :
  (∀ x y, x ≤ y ∧ y ≤ 1 → f x ≤ f y) ∧
  (∀ x y, 2 ≤ x ∧ x ≤ y → f x ≤ f y) ∧
  (∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f y ≤ f x) ∧
  (∀ x, f x < 3 ↔ x < 3) :=
sorry

end f_properties_l459_45950


namespace sum_of_three_consecutive_cubes_divisible_by_nine_l459_45907

theorem sum_of_three_consecutive_cubes_divisible_by_nine (k : ℕ) :
  ∃ m : ℤ, k^3 + (k+1)^3 + (k+2)^3 = 9*m := by
  sorry

end sum_of_three_consecutive_cubes_divisible_by_nine_l459_45907


namespace mat_weaving_in_12_days_l459_45924

/-- Represents a group of mat weavers -/
structure WeaverGroup where
  weavers : ℕ
  mats : ℕ
  days : ℕ

/-- Calculates the number of mats a group can weave in a given number of days -/
def mats_in_days (group : WeaverGroup) (target_days : ℕ) : ℕ :=
  (group.mats * target_days) / group.days

/-- Group A of mat weavers -/
def group_A : WeaverGroup :=
  { weavers := 4, mats := 4, days := 4 }

/-- Group B of mat weavers -/
def group_B : WeaverGroup :=
  { weavers := 6, mats := 9, days := 3 }

/-- Group C of mat weavers -/
def group_C : WeaverGroup :=
  { weavers := 8, mats := 16, days := 4 }

theorem mat_weaving_in_12_days :
  mats_in_days group_A 12 = 12 ∧
  mats_in_days group_B 12 = 36 ∧
  mats_in_days group_C 12 = 48 := by
  sorry

end mat_weaving_in_12_days_l459_45924


namespace parallel_lines_iff_a_eq_two_l459_45993

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

theorem parallel_lines_iff_a_eq_two (a : ℝ) :
  parallel (2 / a) ((a - 1) / 1) ↔ a = 2 :=
sorry

end parallel_lines_iff_a_eq_two_l459_45993


namespace smallest_four_digit_divisible_by_smallest_primes_l459_45995

/-- The five smallest prime numbers -/
def smallest_primes : List Nat := [2, 3, 5, 7, 11]

/-- A number is four-digit if it's between 1000 and 9999 inclusive -/
def is_four_digit (n : Nat) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A number is divisible by all numbers in a list if it's divisible by their product -/
def divisible_by_all (n : Nat) (list : List Nat) : Prop :=
  n % (list.prod) = 0

theorem smallest_four_digit_divisible_by_smallest_primes :
  (2310 = (smallest_primes.prod)) ∧
  (is_four_digit 2310) ∧
  (divisible_by_all 2310 smallest_primes) ∧
  (∀ m : Nat, m < 2310 → ¬(is_four_digit m ∧ divisible_by_all m smallest_primes)) :=
by sorry

end smallest_four_digit_divisible_by_smallest_primes_l459_45995


namespace wife_account_percentage_l459_45920

def total_income : ℝ := 200000
def children_count : ℕ := 3
def children_percentage : ℝ := 0.15
def orphan_house_percentage : ℝ := 0.05
def final_amount : ℝ := 40000

theorem wife_account_percentage :
  let children_total := children_count * children_percentage * total_income
  let remaining_after_children := total_income - children_total
  let orphan_house_amount := orphan_house_percentage * remaining_after_children
  let remaining_after_orphan := remaining_after_children - orphan_house_amount
  let wife_amount := remaining_after_orphan - final_amount
  (wife_amount / total_income) * 100 = 32.25 := by sorry

end wife_account_percentage_l459_45920


namespace congruence_solution_l459_45986

theorem congruence_solution (a m : ℕ) (h1 : a < m) (h2 : m ≥ 2) :
  (∃ x : ℕ, (10 * x + 3) % 18 = 7 % 18 ∧ x % m = a) →
  (∃ x : ℕ, x % 9 = 4 ∧ a = 4 ∧ m = 9 ∧ a + m = 13) :=
by sorry

end congruence_solution_l459_45986


namespace art_show_sales_l459_45908

theorem art_show_sales (total : ℕ) (ratio_remaining : ℕ) (ratio_sold : ℕ) (sold : ℕ) : 
  total = 153 →
  ratio_remaining = 9 →
  ratio_sold = 8 →
  (total - sold) * ratio_sold = sold * ratio_remaining →
  sold = 72 := by
sorry

end art_show_sales_l459_45908


namespace largest_package_size_l459_45914

theorem largest_package_size (alex_markers becca_markers charlie_markers : ℕ) 
  (h_alex : alex_markers = 36)
  (h_becca : becca_markers = 45)
  (h_charlie : charlie_markers = 60) :
  Nat.gcd alex_markers (Nat.gcd becca_markers charlie_markers) = 3 := by
  sorry

end largest_package_size_l459_45914


namespace modulus_of_complex_fraction_l459_45923

theorem modulus_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.abs ((1 + i) / i) = Real.sqrt 2 := by sorry

end modulus_of_complex_fraction_l459_45923


namespace angle_measure_proof_l459_45910

theorem angle_measure_proof (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end angle_measure_proof_l459_45910


namespace frog_corner_probability_l459_45938

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents the possible directions of movement -/
inductive Direction
  | Up | Down | Left | Right | UpLeft | UpRight | DownLeft | DownRight

/-- The grid on which Frieda moves -/
def Grid := Fin 4 → Fin 4 → ℝ

/-- Calculates the next position after a hop in a given direction -/
def nextPosition (p : Position) (d : Direction) : Position :=
  sorry

/-- Calculates the probability of reaching a corner from a given position in n hops -/
def cornerProbability (grid : Grid) (p : Position) (n : ℕ) : ℝ :=
  sorry

/-- Theorem: The probability of reaching any corner within 3 hops from (2,2) is 27/64 -/
theorem frog_corner_probability :
  let initialGrid : Grid := λ _ _ => 0
  let startPos : Position := ⟨1, 1⟩  -- (2,2) in 0-based indexing
  cornerProbability initialGrid startPos 3 = 27 / 64 := by
  sorry

end frog_corner_probability_l459_45938


namespace range_of_m_l459_45936

theorem range_of_m (x m : ℝ) : 
  (∀ x, (4 * x - m < 0 → 1 ≤ 3 - x ∧ 3 - x ≤ 4) ∧ 
  ∃ x, (1 ≤ 3 - x ∧ 3 - x ≤ 4 ∧ ¬(4 * x - m < 0))) →
  m > 8 :=
by sorry

end range_of_m_l459_45936


namespace paint_cans_for_house_l459_45941

/-- Calculates the number of paint cans needed for a house painting job. -/
def paint_cans_needed (num_bedrooms : ℕ) (paint_per_room : ℕ) 
  (color_can_size : ℕ) (white_can_size : ℕ) : ℕ :=
  let num_other_rooms := 2 * num_bedrooms
  let total_color_paint := num_bedrooms * paint_per_room
  let total_white_paint := num_other_rooms * paint_per_room
  let color_cans := (total_color_paint + color_can_size - 1) / color_can_size
  let white_cans := (total_white_paint + white_can_size - 1) / white_can_size
  color_cans + white_cans

/-- Theorem stating that the number of paint cans needed for the given conditions is 10. -/
theorem paint_cans_for_house : paint_cans_needed 3 2 1 3 = 10 := by
  sorry

end paint_cans_for_house_l459_45941


namespace lcm_of_16_and_24_l459_45911

theorem lcm_of_16_and_24 :
  let n : ℕ := 16
  let m : ℕ := 24
  let g : ℕ := 8
  Nat.gcd n m = g →
  Nat.lcm n m = 48 :=
by
  sorry

end lcm_of_16_and_24_l459_45911


namespace carpet_shaded_area_l459_45948

/-- Given a square carpet with side length 12 feet, containing one large shaded square
    with side length S and eight smaller congruent shaded squares with side length T,
    where 12:S = S:T = 4, prove that the total shaded area is 13.5 square feet. -/
theorem carpet_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) :
  S^2 + 8 * T^2 = 13.5 := by
  sorry

end carpet_shaded_area_l459_45948


namespace greatest_prime_factor_of_sum_l459_45933

theorem greatest_prime_factor_of_sum (n : ℕ) :
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (5^8 + 10^7) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (5^8 + 10^7) → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (5^8 + 10^7) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (5^8 + 10^7) → q ≤ p ∧ p = 19) :=
by sorry

end greatest_prime_factor_of_sum_l459_45933


namespace monkey_peach_problem_l459_45961

theorem monkey_peach_problem :
  ∀ (num_monkeys num_peaches : ℕ),
    (num_peaches = 14 * num_monkeys + 48) →
    (num_peaches = 18 * num_monkeys - 64) →
    (num_monkeys = 28 ∧ num_peaches = 440) :=
by
  sorry

end monkey_peach_problem_l459_45961
