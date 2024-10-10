import Mathlib

namespace jenny_sold_192_packs_l846_84644

/-- The number of boxes Jenny sold -/
def boxes_sold : ℝ := 24.0

/-- The number of packs per box -/
def packs_per_box : ℝ := 8.0

/-- The total number of packs Jenny sold -/
def total_packs : ℝ := boxes_sold * packs_per_box

theorem jenny_sold_192_packs : total_packs = 192.0 := by
  sorry

end jenny_sold_192_packs_l846_84644


namespace calculations_correctness_l846_84630

theorem calculations_correctness : 
  (-3 - 1 ≠ -2) ∧ 
  ((-3/4) - (3/4) ≠ 0) ∧ 
  (-8 / (-2) ≠ -4) ∧ 
  ((-3)^2 = 9) := by
  sorry

end calculations_correctness_l846_84630


namespace probability_at_least_one_one_is_correct_l846_84610

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The probability of at least one die showing a 1 when two fair 6-sided dice are rolled -/
def probability_at_least_one_one : ℚ := 11 / 36

/-- Theorem stating that the probability of at least one die showing a 1 
    when two fair 6-sided dice are rolled is 11/36 -/
theorem probability_at_least_one_one_is_correct : 
  probability_at_least_one_one = 11 / 36 := by
  sorry

end probability_at_least_one_one_is_correct_l846_84610


namespace fraction_inequality_l846_84615

theorem fraction_inequality (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : (a : ℝ) / b > Real.sqrt 2) :
  (a : ℝ) / b - 1 / (2 * (a : ℝ) * b) > Real.sqrt 2 := by
  sorry

end fraction_inequality_l846_84615


namespace total_soaking_time_with_ink_l846_84624

/-- Represents the soaking time for different types of stains -/
def SoakingTime : Type := Nat → Nat

/-- Calculates the total soaking time for a piece of clothing -/
def totalSoakingTime (stainCounts : List Nat) (soakingTimes : List Nat) : Nat :=
  List.sum (List.zipWith (· * ·) stainCounts soakingTimes)

/-- Calculates the additional time needed for ink stains -/
def additionalInkTime (inkStainCount : Nat) (extraTimePerInkStain : Nat) : Nat :=
  inkStainCount * extraTimePerInkStain

theorem total_soaking_time_with_ink (shirtStainCounts shirtSoakingTimes
                                     pantsStainCounts pantsSoakingTimes
                                     socksStainCounts socksSoakingTimes : List Nat)
                                    (inkStainCount extraTimePerInkStain : Nat) :
  totalSoakingTime shirtStainCounts shirtSoakingTimes +
  totalSoakingTime pantsStainCounts pantsSoakingTimes +
  totalSoakingTime socksStainCounts socksSoakingTimes +
  additionalInkTime inkStainCount extraTimePerInkStain = 54 :=
by
  sorry

#check total_soaking_time_with_ink

end total_soaking_time_with_ink_l846_84624


namespace president_vice_president_selection_l846_84608

def club_members : ℕ := 30
def boys : ℕ := 18
def girls : ℕ := 12

theorem president_vice_president_selection :
  (boys * girls) + (girls * boys) = 432 :=
by sorry

end president_vice_president_selection_l846_84608


namespace range_of_m_l846_84638

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h_eq : 2/x + 1/y = 1) 
  (h_ineq : x + 2*y > m^2 - 2*m) : 
  -2 < m ∧ m < 4 := by
sorry

end range_of_m_l846_84638


namespace parallel_plane_intersection_theorem_l846_84679

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields for a plane

/-- A line in 3D space -/
structure Line where
  -- Add necessary fields for a line

/-- Two planes are parallel -/
def parallel_planes (α β : Plane) : Prop :=
  sorry

/-- A plane intersects another plane along a line -/
def plane_intersect (α γ : Plane) (l : Line) : Prop :=
  sorry

/-- Two lines are parallel -/
def parallel_lines (a b : Line) : Prop :=
  sorry

/-- Theorem: If two parallel planes are intersected by a third plane, 
    the lines of intersection are parallel -/
theorem parallel_plane_intersection_theorem 
  (α β γ : Plane) (a b : Line) 
  (h1 : parallel_planes α β) 
  (h2 : plane_intersect α γ a) 
  (h3 : plane_intersect β γ b) : 
  parallel_lines a b :=
sorry

end parallel_plane_intersection_theorem_l846_84679


namespace arithmetic_calculation_l846_84665

theorem arithmetic_calculation : 12 * 11 + 7 * 8 - 5 * 6 + 10 * 4 = 198 := by
  sorry

end arithmetic_calculation_l846_84665


namespace merged_class_size_is_41_l846_84663

/-- Represents a group of students with a specific student's position --/
structure StudentGroup where
  right_rank : Nat
  left_rank : Nat

/-- Calculates the total number of students in a group --/
def group_size (g : StudentGroup) : Nat :=
  g.right_rank - 1 + g.left_rank

/-- Calculates the total number of students in the merged class --/
def merged_class_size (group_a group_b : StudentGroup) : Nat :=
  group_size group_a + group_size group_b

/-- Theorem stating the total number of students in the merged class --/
theorem merged_class_size_is_41 :
  let group_a : StudentGroup := ⟨13, 8⟩
  let group_b : StudentGroup := ⟨10, 12⟩
  merged_class_size group_a group_b = 41 := by
  sorry

#eval merged_class_size ⟨13, 8⟩ ⟨10, 12⟩

end merged_class_size_is_41_l846_84663


namespace inequality_proof_l846_84609

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end inequality_proof_l846_84609


namespace archie_antibiotic_cost_l846_84635

/-- The total cost of antibiotics for Archie -/
def total_cost (doses_per_day : ℕ) (days : ℕ) (cost_per_dose : ℕ) : ℕ :=
  doses_per_day * days * cost_per_dose

/-- Proof that the total cost of antibiotics for Archie is $63 -/
theorem archie_antibiotic_cost :
  total_cost 3 7 3 = 63 := by
  sorry

end archie_antibiotic_cost_l846_84635


namespace youngest_brother_age_l846_84654

theorem youngest_brother_age (a b c : ℕ) : 
  (a + b + c = 96) → 
  (b = a + 1) → 
  (c = a + 2) → 
  a = 31 := by
sorry

end youngest_brother_age_l846_84654


namespace otimes_four_two_l846_84696

def otimes (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem otimes_four_two : otimes 4 2 = 18 := by
  sorry

end otimes_four_two_l846_84696


namespace shooting_events_contradictory_l846_84634

-- Define the sample space
def Ω : Type := List Bool

-- Define the events
def at_least_one_hit (ω : Ω) : Prop := ω.any id
def three_consecutive_misses (ω : Ω) : Prop := ω = [false, false, false]

-- Define the property of being contradictory events
def contradictory (A B : Ω → Prop) : Prop :=
  (∀ ω : Ω, A ω → ¬B ω) ∧ (∀ ω : Ω, B ω → ¬A ω)

-- Theorem statement
theorem shooting_events_contradictory :
  contradictory at_least_one_hit three_consecutive_misses :=
by sorry

end shooting_events_contradictory_l846_84634


namespace tens_digit_of_19_power_2023_l846_84657

theorem tens_digit_of_19_power_2023 : ∃ n : ℕ, 19^2023 ≡ 50 + n [ZMOD 100] :=
by
  sorry

end tens_digit_of_19_power_2023_l846_84657


namespace museum_visitors_l846_84694

theorem museum_visitors (V T E : ℕ) : 
  V = 6 * T →
  E = 180 →
  E = 3 * T / 5 →
  V = 1800 :=
by sorry

end museum_visitors_l846_84694


namespace tank_water_supply_l846_84692

theorem tank_water_supply (C V : ℝ) 
  (h1 : C = 15 * (V + 10))
  (h2 : C = 12 * (V + 20)) :
  C / V = 20 := by
sorry

end tank_water_supply_l846_84692


namespace henan_population_scientific_notation_l846_84626

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem henan_population_scientific_notation :
  toScientificNotation (98.83 * 1000000) = ScientificNotation.mk 9.883 7 (by norm_num) :=
sorry

end henan_population_scientific_notation_l846_84626


namespace f_monotone_increasing_iff_l846_84617

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1) + 1

theorem f_monotone_increasing_iff (x : ℝ) :
  StrictMono (fun y => f y) ↔ x ∈ Set.Ioo (-1 : ℝ) 1 := by sorry

end f_monotone_increasing_iff_l846_84617


namespace most_probable_occurrences_l846_84645

theorem most_probable_occurrences (p : ℝ) (k₀ : ℕ) (h_p : p = 0.4) (h_k₀ : k₀ = 25) :
  ∃ n : ℕ, 62 ≤ n ∧ n ≤ 64 ∧
  (∀ m : ℕ, (m * p - (1 - p) ≤ k₀ ∧ k₀ < m * p + p) → m = n) :=
by sorry

end most_probable_occurrences_l846_84645


namespace cubic_expansion_sum_l846_84686

theorem cubic_expansion_sum (x a₀ a₁ a₂ a₃ : ℝ) 
  (h : x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) : 
  a₁ + a₂ + a₃ = 19 := by
  sorry

end cubic_expansion_sum_l846_84686


namespace erasers_given_to_doris_l846_84629

def initial_erasers : ℕ := 81
def final_erasers : ℕ := 47

theorem erasers_given_to_doris : initial_erasers - final_erasers = 34 := by
  sorry

end erasers_given_to_doris_l846_84629


namespace no_real_roots_l846_84646

theorem no_real_roots : ∀ x : ℝ, x^2 + 2*x + 4 ≠ 0 := by
  sorry

end no_real_roots_l846_84646


namespace x_intercept_of_line_l846_84653

/-- The x-intercept of the line 6x + 7y = 35 is (35/6, 0) -/
theorem x_intercept_of_line (x y : ℚ) : 
  (6 * x + 7 * y = 35) → (x = 35 / 6 ∧ y = 0) → (6 * (35 / 6) + 7 * 0 = 35) := by
  sorry

#check x_intercept_of_line

end x_intercept_of_line_l846_84653


namespace square_sum_equals_25_l846_84667

theorem square_sum_equals_25 (x y : ℝ) 
  (h1 : y + 6 = (x - 3)^2) 
  (h2 : x + 6 = (y - 3)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 25 := by
  sorry

end square_sum_equals_25_l846_84667


namespace tangent_slope_angle_is_45_degrees_l846_84614

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- The point of interest
def point : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem tangent_slope_angle_is_45_degrees :
  let slope := f' point.1
  let angle := Real.arctan slope
  angle = π / 4 := by sorry

end tangent_slope_angle_is_45_degrees_l846_84614


namespace two_numbers_product_l846_84685

theorem two_numbers_product (ε : ℝ) (h : ε > 0) : 
  ∃ x y : ℝ, x + y = 21 ∧ x^2 + y^2 = 527 ∧ |x * y + 43.05| < ε :=
sorry

end two_numbers_product_l846_84685


namespace particle_probabilities_l846_84618

/-- A particle moves on a line with marked points 0, ±1, ±2, ±3, ... 
    Starting at point 0, it moves to n+1 or n-1 with equal probabilities 1/2 -/
def Particle := ℤ

/-- The probability that the particle will be at point 1 at some time -/
def prob_at_one (p : Particle) : ℝ := sorry

/-- The probability that the particle will be at point -1 at some time -/
def prob_at_neg_one (p : Particle) : ℝ := sorry

/-- The probability that the particle will return to point 0 at some time 
    other than the initial starting point -/
def prob_return_to_zero (p : Particle) : ℝ := sorry

/-- The theorem stating that all three probabilities are equal to 1 -/
theorem particle_probabilities (p : Particle) : 
  prob_at_one p = 1 ∧ prob_at_neg_one p = 1 ∧ prob_return_to_zero p = 1 :=
by sorry

end particle_probabilities_l846_84618


namespace smallest_integer_with_given_remainders_l846_84669

theorem smallest_integer_with_given_remainders : ∃ b : ℕ+, 
  (b : ℕ) % 4 = 3 ∧ 
  (b : ℕ) % 6 = 5 ∧ 
  ∀ k : ℕ+, (k : ℕ) % 4 = 3 ∧ (k : ℕ) % 6 = 5 → k ≥ b :=
by
  use 23
  sorry

end smallest_integer_with_given_remainders_l846_84669


namespace intersection_with_complement_example_l846_84611

open Set

theorem intersection_with_complement_example : 
  let U : Set ℕ := {1, 3, 5, 7, 9}
  let A : Set ℕ := {3, 7, 9}
  let B : Set ℕ := {1, 9}
  A ∩ (U \ B) = {3, 7} := by
sorry

end intersection_with_complement_example_l846_84611


namespace additive_increasing_nonneg_implies_odd_increasing_l846_84698

def is_additive (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂

def is_increasing_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ x₂ → x₂ ≥ 0 → f x₁ ≥ f x₂

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ x₂ → f x₁ ≥ f x₂

theorem additive_increasing_nonneg_implies_odd_increasing
  (f : ℝ → ℝ) (h1 : is_additive f) (h2 : is_increasing_nonneg f) :
  is_odd f ∧ is_increasing f :=
sorry

end additive_increasing_nonneg_implies_odd_increasing_l846_84698


namespace meeting_arrangements_l846_84678

def num_schools : ℕ := 3
def members_per_school : ℕ := 6
def host_representatives : ℕ := 3
def other_representatives : ℕ := 1

def arrange_meeting : ℕ := 
  num_schools * (members_per_school.choose host_representatives) * 
  ((members_per_school.choose other_representatives) ^ (num_schools - 1))

theorem meeting_arrangements :
  arrange_meeting = 2160 := by
  sorry

end meeting_arrangements_l846_84678


namespace simplify_and_evaluate_l846_84662

/-- Given a = -1 and ab = 2, prove that 3(2a²b + ab²) - (3ab² - a²b) evaluates to -14 -/
theorem simplify_and_evaluate (a b : ℤ) (h1 : a = -1) (h2 : a * b = 2) :
  3 * (2 * a^2 * b + a * b^2) - (3 * a * b^2 - a^2 * b) = -14 := by
  sorry

end simplify_and_evaluate_l846_84662


namespace odd_function_zero_value_l846_84619

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_zero_value (f : ℝ → ℝ) (h : OddFunction f) : f 0 = 0 := by
  sorry

end odd_function_zero_value_l846_84619


namespace expression_simplification_l846_84677

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 5) (hz : z ≠ 7) :
  (x - 2) / (6 - z) * (y - 5) / (2 - x) * (z - 7) / (5 - y) = -1 := by
  sorry

end expression_simplification_l846_84677


namespace binomial_sum_of_even_coefficients_l846_84648

theorem binomial_sum_of_even_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₂ + a₄ = 121 := by
sorry

end binomial_sum_of_even_coefficients_l846_84648


namespace dons_walking_speed_l846_84656

/-- Proof of Don's walking speed given the conditions of Cara and Don's walk --/
theorem dons_walking_speed
  (total_distance : ℝ)
  (caras_speed : ℝ)
  (caras_distance : ℝ)
  (don_delay : ℝ)
  (h1 : total_distance = 45)
  (h2 : caras_speed = 6)
  (h3 : caras_distance = 30)
  (h4 : don_delay = 2) :
  ∃ (dons_speed : ℝ), dons_speed = 5 := by
  sorry

end dons_walking_speed_l846_84656


namespace simple_interest_problem_l846_84661

theorem simple_interest_problem (P R T : ℝ) : 
  P = 300 →
  P * (R + 6) / 100 * T = P * R / 100 * T + 90 →
  T = 5 :=
by
  sorry

end simple_interest_problem_l846_84661


namespace pauls_earnings_duration_l846_84627

/-- Calculates how many weeks Paul's earnings will last given his weekly earnings and expenses. -/
def weeks_earnings_last (lawn_mowing : ℚ) (weed_eating : ℚ) (bush_trimming : ℚ) (fence_painting : ℚ)
                        (food_expense : ℚ) (transportation_expense : ℚ) (entertainment_expense : ℚ) : ℚ :=
  (lawn_mowing + weed_eating + bush_trimming + fence_painting) /
  (food_expense + transportation_expense + entertainment_expense)

/-- Theorem stating that Paul's earnings will last 2.5 weeks given his specific earnings and expenses. -/
theorem pauls_earnings_duration :
  weeks_earnings_last 12 8 5 20 10 5 3 = 5/2 := by
  sorry

end pauls_earnings_duration_l846_84627


namespace scaled_circle_area_l846_84640

/-- Given a circle with center P(-5, 3) passing through Q(7, -4), 
    when uniformly scaled by a factor of 2 from its center, 
    the area of the resulting circle is 772π. -/
theorem scaled_circle_area : 
  let P : ℝ × ℝ := (-5, 3)
  let Q : ℝ × ℝ := (7, -4)
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let scale_factor : ℝ := 2
  let scaled_area := π * (scale_factor * r)^2
  scaled_area = 772 * π :=
by sorry

end scaled_circle_area_l846_84640


namespace a_gt_b_neither_sufficient_nor_necessary_for_a_sq_gt_b_sq_l846_84666

theorem a_gt_b_neither_sufficient_nor_necessary_for_a_sq_gt_b_sq :
  ∃ a b : ℝ, (a > b ∧ ¬(a^2 > b^2)) ∧ ∃ c d : ℝ, (c^2 > d^2 ∧ ¬(c > d)) := by
  sorry

end a_gt_b_neither_sufficient_nor_necessary_for_a_sq_gt_b_sq_l846_84666


namespace museum_ticket_cost_l846_84675

theorem museum_ticket_cost (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_price : ℕ) (teacher_ticket_price : ℕ) : 
  num_students = 12 → 
  num_teachers = 4 → 
  student_ticket_price = 1 → 
  teacher_ticket_price = 3 → 
  num_students * student_ticket_price + num_teachers * teacher_ticket_price = 24 := by
sorry

end museum_ticket_cost_l846_84675


namespace three_inverse_mod_191_l846_84672

theorem three_inverse_mod_191 : ∃ x : ℕ, x < 191 ∧ (3 * x) % 191 = 1 ∧ x = 64 := by sorry

end three_inverse_mod_191_l846_84672


namespace complex_expression_evaluation_l846_84668

/-- Evaluates |3-7i| + |3+7i| - arg(3+7i) -/
theorem complex_expression_evaluation :
  let z₁ : ℂ := 3 - 7*I
  let z₂ : ℂ := 3 + 7*I
  Complex.abs z₁ + Complex.abs z₂ - Complex.arg z₂ = 2 * Real.sqrt 58 - Real.arctan (7/3) := by
  sorry

end complex_expression_evaluation_l846_84668


namespace factor_75x_plus_50_l846_84695

theorem factor_75x_plus_50 (x : ℝ) : 75 * x + 50 = 25 * (3 * x + 2) := by
  sorry

end factor_75x_plus_50_l846_84695


namespace interest_rate_is_ten_percent_l846_84602

/-- Given a principal amount and an interest rate satisfying the given conditions,
    prove that the interest rate is 10% --/
theorem interest_rate_is_ten_percent (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 2420)
  (h2 : P * (1 + r)^3 = 2662) :
  r = 0.1 := by
  sorry

end interest_rate_is_ten_percent_l846_84602


namespace puppies_per_cage_l846_84691

def initial_puppies : ℕ := 56
def sold_puppies : ℕ := 24
def num_cages : ℕ := 8

theorem puppies_per_cage :
  (initial_puppies - sold_puppies) / num_cages = 4 :=
by sorry

end puppies_per_cage_l846_84691


namespace midpoint_is_inferior_exists_n_satisfying_conditions_l846_84659

/-- Definition of a superior point in the first quadrant -/
def is_superior_point (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a / b > c / d

/-- Definition of an inferior point in the first quadrant -/
def is_inferior_point (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a / b < c / d

/-- Theorem: The midpoint of a superior point and an inferior point is inferior to the superior point -/
theorem midpoint_is_inferior (a b c d : ℝ) :
  is_superior_point a b c d →
  is_inferior_point ((a + c) / 2) ((b + d) / 2) a b :=
sorry

/-- Definition of the set of integers from 1 to 2021 -/
def S : Set ℤ := {m | 0 < m ∧ m < 2022}

/-- Theorem: There exists an integer n satisfying the given conditions -/
theorem exists_n_satisfying_conditions :
  ∃ n : ℤ, ∀ m ∈ S,
    (is_inferior_point n (2 * m + 1) 2022 m) ∧
    (is_superior_point n (2 * m + 1) 2023 (m + 1)) :=
sorry

end midpoint_is_inferior_exists_n_satisfying_conditions_l846_84659


namespace symmetry_line_of_circles_l846_84655

/-- Given two circles O and C that are symmetric with respect to a line l, 
    prove that l has the equation x - y + 2 = 0 -/
theorem symmetry_line_of_circles (x y : ℝ) : 
  (x^2 + y^2 = 4) →  -- equation of circle O
  (x^2 + y^2 + 4*x - 4*y + 4 = 0) →  -- equation of circle C
  ∃ (l : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ l ↔ p.1 - p.2 + 2 = 0) ∧  -- equation of line l
    (∀ (p : ℝ × ℝ), p ∈ l ↔ 
      ∃ (q r : ℝ × ℝ), 
        (q.1^2 + q.2^2 = 4) ∧  -- q is on circle O
        (r.1^2 + r.2^2 + 4*r.1 - 4*r.2 + 4 = 0) ∧  -- r is on circle C
        (p = ((q.1 + r.1)/2, (q.2 + r.2)/2)) ∧  -- p is midpoint of qr
        ((r.1 - q.1) * (p.1 - q.1) + (r.2 - q.2) * (p.2 - q.2) = 0))  -- qr ⊥ l
  := by sorry

end symmetry_line_of_circles_l846_84655


namespace combined_swim_time_l846_84601

def freestyle_time : ℕ := 48

def backstroke_time : ℕ := freestyle_time + 4

def butterfly_time : ℕ := backstroke_time + 3

def breaststroke_time : ℕ := butterfly_time + 2

def total_time : ℕ := freestyle_time + backstroke_time + butterfly_time + breaststroke_time

theorem combined_swim_time : total_time = 212 := by
  sorry

end combined_swim_time_l846_84601


namespace rectangle_area_increase_l846_84681

theorem rectangle_area_increase (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let new_area := (1.1 * L) * (1.1 * W)
  let original_area := L * W
  (new_area - original_area) / original_area * 100 = 21 := by
sorry


end rectangle_area_increase_l846_84681


namespace molecular_weight_BaCl2_correct_l846_84660

/-- The molecular weight of BaCl2 in g/mol -/
def molecular_weight_BaCl2 : ℝ := 207

/-- The number of moles given in the problem -/
def given_moles : ℝ := 8

/-- The total weight of the given moles of BaCl2 in grams -/
def total_weight : ℝ := 1656

/-- Theorem stating that the molecular weight of BaCl2 is correct -/
theorem molecular_weight_BaCl2_correct :
  molecular_weight_BaCl2 = total_weight / given_moles :=
by sorry

end molecular_weight_BaCl2_correct_l846_84660


namespace root_bounds_l846_84605

theorem root_bounds (b c x : ℝ) (hb : 5.025 ≤ b ∧ b ≤ 5.035) (hc : 1.745 ≤ c ∧ c ≤ 1.755)
  (hx : (3 * x + b) / 4 = (2 * x - 3) / c) :
  7.512 ≤ x ∧ x ≤ 7.618 :=
by sorry

end root_bounds_l846_84605


namespace charity_race_dropouts_l846_84652

/-- The number of people who dropped out of a bicycle charity race --/
def dropouts (initial_racers : ℕ) (joined_racers : ℕ) (finishers : ℕ) : ℕ :=
  (initial_racers + joined_racers) * 2 - finishers

theorem charity_race_dropouts : dropouts 50 30 130 = 30 := by
  sorry

end charity_race_dropouts_l846_84652


namespace circle_chord_intersection_l846_84628

theorem circle_chord_intersection (r : ℝ) (chord_length : ℝ) :
  r = 7 →
  chord_length = 10 →
  let segment_length := r - 2 * Real.sqrt 6
  ∃ (AK KB : ℝ),
    AK = segment_length ∧
    KB = 2 * r - segment_length ∧
    AK + KB = 2 * r ∧
    AK * KB = (chord_length / 2) ^ 2 :=
by sorry

end circle_chord_intersection_l846_84628


namespace intersection_equality_l846_84647

def M : Set ℤ := {-1, 0, 1}
def N (a : ℤ) : Set ℤ := {a, a^2}

theorem intersection_equality (a : ℤ) : M ∩ N a = N a ↔ a = -1 := by
  sorry

end intersection_equality_l846_84647


namespace arithmetic_sequence_property_l846_84607

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  x : ℚ
  first_term : ℚ := 3 * x - 4
  second_term : ℚ := 6 * x - 14
  third_term : ℚ := 4 * x + 3
  is_arithmetic : second_term - first_term = third_term - second_term

/-- The nth term of the sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.first_term + (n - 1) * (seq.second_term - seq.first_term)

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  ∃ n : ℕ, nth_term seq n = 3012 ∧ n = 247 := by
  sorry

end arithmetic_sequence_property_l846_84607


namespace cranberry_juice_cost_l846_84622

/-- The total cost of a can of cranberry juice -/
theorem cranberry_juice_cost (ounces : ℕ) (cost_per_ounce : ℕ) : 
  ounces = 12 → cost_per_ounce = 7 → ounces * cost_per_ounce = 84 := by
  sorry

end cranberry_juice_cost_l846_84622


namespace functional_equation_solution_l846_84687

/-- A monotonic function on ℝ satisfying f(x) · f(y) = f(x + y) is of the form a^x for some a > 0 -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_mono : Monotone f) 
  (h_eq : ∀ x y : ℝ, f x * f y = f (x + y)) :
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f x = a ^ x :=
sorry

end functional_equation_solution_l846_84687


namespace a_7_equals_63_l846_84600

def sequence_a : ℕ → ℚ
  | 0 => 0  -- We define a₀ and a₁ arbitrarily as they are not used
  | 1 => 0
  | 2 => 1
  | 3 => 3
  | (n + 1) => (sequence_a n ^ 2 - sequence_a (n - 1) + 2 * sequence_a n) / (sequence_a (n - 1) + 1)

theorem a_7_equals_63 : sequence_a 7 = 63 := by
  sorry

end a_7_equals_63_l846_84600


namespace indeterminate_roots_l846_84683

/-- Given that the equation mx^2 - 2(m+2)x + m + 5 = 0 has no real roots,
    the number of real roots of (m-5)x^2 - 2(m+2)x + m = 0 cannot be determined
    to be exclusively 0, 1, or 2. -/
theorem indeterminate_roots (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - 2*(m+2)*x + m + 5 ≠ 0) →
  ¬(∀ x : ℝ, (m-5) * x^2 - 2*(m+2)*x + m ≠ 0) ∧
  ¬(∃! x : ℝ, (m-5) * x^2 - 2*(m+2)*x + m = 0) ∧
  ¬(∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2*(m+2)*x + m = 0 ∧ (m-5) * y^2 - 2*(m+2)*y + m = 0) :=
sorry

end indeterminate_roots_l846_84683


namespace cylinder_lateral_surface_area_l846_84632

/-- The lateral surface area of a cylinder with base radius 2 and generatrix length 3 is 12π. -/
theorem cylinder_lateral_surface_area :
  ∀ (r g : ℝ), r = 2 → g = 3 → 2 * π * r * g = 12 * π :=
by
  sorry

end cylinder_lateral_surface_area_l846_84632


namespace f_derivative_f_initial_condition_range_of_x_l846_84693

/-- A function f with the given properties -/
def f : ℝ → ℝ :=
  sorry

theorem f_derivative (x : ℝ) : deriv f x = 5 + Real.cos x :=
  sorry

theorem f_initial_condition : f 0 = 0 :=
  sorry

theorem range_of_x (x : ℝ) :
  f (1 - x) + f (1 - x^2) < 0 ↔ x ∈ Set.Iic (-2) ∪ Set.Ioi 1 :=
  sorry

end f_derivative_f_initial_condition_range_of_x_l846_84693


namespace inequality_solution_set_l846_84699

theorem inequality_solution_set :
  {x : ℝ | (3 / 8 : ℝ) + |x - (1 / 4 : ℝ)| < (7 / 8 : ℝ)} = Set.Ioo (-(1 / 4 : ℝ)) ((3 / 4 : ℝ)) :=
by sorry

end inequality_solution_set_l846_84699


namespace convenience_store_syrup_cost_l846_84620

/-- Calculates the weekly syrup cost for a convenience store. -/
def weekly_syrup_cost (soda_sold : ℕ) (gallons_per_box : ℕ) (cost_per_box : ℕ) : ℕ :=
  (soda_sold / gallons_per_box) * cost_per_box

/-- Theorem stating the weekly syrup cost for the given conditions. -/
theorem convenience_store_syrup_cost :
  weekly_syrup_cost 180 30 40 = 240 := by
  sorry

end convenience_store_syrup_cost_l846_84620


namespace mrs_hilt_spent_74_cents_l846_84649

/-- Calculates the total amount spent by Mrs. Hilt at the school store -/
def school_store_total (notebook_cost ruler_cost pencil_cost : ℕ) (num_pencils : ℕ) : ℕ :=
  notebook_cost + ruler_cost + (pencil_cost * num_pencils)

/-- Proves that Mrs. Hilt spent 74 cents at the school store -/
theorem mrs_hilt_spent_74_cents :
  school_store_total 35 18 7 3 = 74 := by
  sorry

end mrs_hilt_spent_74_cents_l846_84649


namespace water_poured_out_l846_84612

/-- The volume of water poured out from a cylindrical cup -/
theorem water_poured_out (r h : Real) (α β : Real) : 
  r = 4 → 
  h = 8 * Real.sqrt 3 → 
  α = π / 3 → 
  β = π / 6 → 
  let V₁ := π * r^2 * (h - (h - r * Real.tan α))
  let V₂ := π * r^2 * (h / 2)
  V₁ - V₂ = (128 * Real.sqrt 3 * π) / 3 := by sorry

end water_poured_out_l846_84612


namespace product_equals_243_l846_84621

theorem product_equals_243 :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end product_equals_243_l846_84621


namespace sara_movie_tickets_l846_84639

/-- The number of movie theater tickets Sara bought -/
def num_tickets : ℕ := 2

/-- The cost of each movie theater ticket in cents -/
def ticket_cost : ℕ := 1062

/-- The cost of renting a movie in cents -/
def rental_cost : ℕ := 159

/-- The cost of buying a movie in cents -/
def purchase_cost : ℕ := 1395

/-- The total amount Sara spent in cents -/
def total_spent : ℕ := 3678

/-- Theorem stating that the number of tickets Sara bought is correct -/
theorem sara_movie_tickets : 
  num_tickets * ticket_cost + rental_cost + purchase_cost = total_spent :=
by sorry

end sara_movie_tickets_l846_84639


namespace largest_digit_divisible_by_6_l846_84636

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem largest_digit_divisible_by_6 :
  ∀ N : ℕ, N ≤ 9 →
    (is_divisible_by_6 (45670 + N) → N ≤ 8) ∧
    is_divisible_by_6 (45670 + 8) :=
by sorry

end largest_digit_divisible_by_6_l846_84636


namespace coin_authenticity_test_l846_84684

/-- Represents the type of coin -/
inductive CoinType
| Real
| Fake

/-- Represents the weight difference between real and fake coins -/
def weightDifference : ℤ := 1

/-- Represents the total number of coins -/
def totalCoins (n : ℕ) : ℕ := 2 * n + 1

/-- Represents the number of fake coins -/
def fakeCoins (k : ℕ) : ℕ := 2 * k

/-- Represents the scale reading when weighing n coins against n coins -/
def scaleReading (n k₁ k₂ : ℕ) : ℤ := (k₁ : ℤ) - (k₂ : ℤ)

/-- Main theorem: The parity of the scale reading determines the type of the chosen coin -/
theorem coin_authenticity_test (n k : ℕ) (h : k ≤ n) :
  ∀ (chosenCoin : CoinType) (k₁ k₂ : ℕ) (h₁ : k₁ + k₂ = fakeCoins k - 1),
    chosenCoin = CoinType.Fake ↔ scaleReading n k₁ k₂ % 2 ≠ 0 :=
by sorry

end coin_authenticity_test_l846_84684


namespace no_prime_roots_for_quadratic_l846_84625

theorem no_prime_roots_for_quadratic : 
  ¬∃ (k : ℤ), ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    p ≠ q ∧
    (p : ℤ) + q = 72 ∧ 
    (p : ℤ) * q = k ∧
    ∀ (x : ℤ), x^2 - 72*x + k = 0 ↔ x = p ∨ x = q :=
by sorry

end no_prime_roots_for_quadratic_l846_84625


namespace frequency_converges_to_half_l846_84682

/-- A fair coin toss experiment -/
structure CoinTossExperiment where
  n : ℕ  -- number of tosses
  m : ℕ  -- number of heads
  h_m_le_n : m ≤ n  -- m cannot exceed n

/-- The frequency of heads in a coin toss experiment -/
def frequency (e : CoinTossExperiment) : ℚ :=
  e.m / e.n

/-- The theoretical probability of heads for a fair coin -/
def fairCoinProbability : ℚ := 1 / 2

/-- The main theorem: as n approaches infinity, the frequency converges to 1/2 -/
theorem frequency_converges_to_half :
  ∀ ε > 0, ∃ N, ∀ e : CoinTossExperiment, e.n ≥ N →
    |frequency e - fairCoinProbability| < ε :=
sorry

end frequency_converges_to_half_l846_84682


namespace two_distinct_roots_iff_a_in_A_l846_84650

/-- The equation has exactly two distinct roots -/
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (x₁ - a)^2 - 1 = 2 * (x₁ + |x₁|) ∧
  (x₂ - a)^2 - 1 = 2 * (x₂ + |x₂|) ∧
  ∀ x : ℝ, (x - a)^2 - 1 = 2 * (x + |x|) → x = x₁ ∨ x = x₂

/-- The set of values for a -/
def A : Set ℝ := Set.Ioi 1 ∪ Set.Ioo (-1) 1 ∪ Set.Iic (-5/4)

theorem two_distinct_roots_iff_a_in_A :
  ∀ a : ℝ, has_two_distinct_roots a ↔ a ∈ A :=
by sorry

end two_distinct_roots_iff_a_in_A_l846_84650


namespace inequality_proof_l846_84690

theorem inequality_proof (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end inequality_proof_l846_84690


namespace non_shaded_perimeter_l846_84641

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter 
  (total : Rectangle)
  (small : Rectangle)
  (shaded_area : ℝ)
  (h1 : total.width = 12)
  (h2 : total.height = 10)
  (h3 : small.width = 4)
  (h4 : small.height = 3)
  (h5 : shaded_area = 120) :
  perimeter { width := total.width - (total.width - small.width),
              height := total.height - small.height } = 23 := by
sorry

end non_shaded_perimeter_l846_84641


namespace smallest_congruent_integer_l846_84697

theorem smallest_congruent_integer : ∃ n : ℕ+, 
  (n : ℤ) % 3 = 2 ∧ 
  (n : ℤ) % 4 = 3 ∧ 
  (n : ℤ) % 5 = 4 ∧ 
  (n : ℤ) % 6 = 5 ∧ 
  (∀ m : ℕ+, m < n → 
    (m : ℤ) % 3 ≠ 2 ∨ 
    (m : ℤ) % 4 ≠ 3 ∨ 
    (m : ℤ) % 5 ≠ 4 ∨ 
    (m : ℤ) % 6 ≠ 5) :=
by
  sorry

end smallest_congruent_integer_l846_84697


namespace prob_three_red_is_one_fifty_fifth_l846_84633

/-- The number of red balls in the bag -/
def red_balls : ℕ := 4

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 5

/-- The number of green balls in the bag -/
def green_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := red_balls + blue_balls + green_balls

/-- The number of balls to be picked -/
def picked_balls : ℕ := 3

/-- The probability of picking 3 red balls when randomly selecting 3 balls without replacement -/
def prob_three_red : ℚ := (red_balls * (red_balls - 1) * (red_balls - 2)) / 
  (total_balls * (total_balls - 1) * (total_balls - 2))

theorem prob_three_red_is_one_fifty_fifth : prob_three_red = 1 / 55 := by
  sorry

end prob_three_red_is_one_fifty_fifth_l846_84633


namespace sine_function_shifted_symmetric_l846_84643

/-- Given a function f(x) = sin(ωx + φ), prove that under certain conditions, φ = π/6 -/
theorem sine_function_shifted_symmetric (ω φ : Real) : 
  ω > 0 → 
  0 < φ → 
  φ < Real.pi / 2 → 
  (fun x ↦ Real.sin (ω * x + φ)) 0 = -(fun x ↦ Real.sin (ω * x + φ)) (Real.pi / 2) →
  (∀ x, Real.sin (ω * (x + Real.pi / 12) + φ) = -Real.sin (ω * (-x + Real.pi / 12) + φ)) →
  φ = Real.pi / 6 := by
  sorry

end sine_function_shifted_symmetric_l846_84643


namespace derived_sequence_general_term_l846_84658

/-- An arithmetic sequence {a_n} with specific terms and a derived sequence {b_n} -/
def arithmetic_and_derived_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) ∧  -- arithmetic sequence condition
  (a 2 = 8) ∧
  (a 8 = 26) ∧
  (∀ n : ℕ, b n = a (3^n))  -- definition of b_n

/-- The general term of the derived sequence b_n -/
def b_general_term (n : ℕ) : ℝ := 3 * 3^n + 2

/-- Theorem stating that b_n equals the derived general term -/
theorem derived_sequence_general_term (a : ℕ → ℝ) (b : ℕ → ℝ) :
  arithmetic_and_derived_sequence a b →
  ∀ n : ℕ, b n = b_general_term n :=
by
  sorry

end derived_sequence_general_term_l846_84658


namespace g_sum_zero_l846_84671

theorem g_sum_zero (f : ℝ → ℝ) : 
  let g := λ x => f x - f (2010 - x)
  ∀ x, g x + g (2010 - x) = 0 := by
  sorry

end g_sum_zero_l846_84671


namespace count_negative_expressions_l846_84631

theorem count_negative_expressions : 
  let expressions := [-3^2, (-3)^2, -(-3), -|-3|]
  (expressions.filter (· < 0)).length = 2 := by
  sorry

end count_negative_expressions_l846_84631


namespace teacher_distribution_count_l846_84664

/-- The number of ways to distribute teachers to classes --/
def distribute_teachers (n_teachers : ℕ) (n_classes : ℕ) : ℕ :=
  n_classes ^ n_teachers

/-- The number of ways to distribute teachers to classes with at least one empty class --/
def distribute_with_empty (n_teachers : ℕ) (n_classes : ℕ) : ℕ :=
  n_classes * (n_classes - 1) ^ n_teachers

/-- The number of valid distributions of teachers to classes --/
def valid_distributions (n_teachers : ℕ) (n_classes : ℕ) : ℕ :=
  distribute_teachers n_teachers n_classes - distribute_with_empty n_teachers n_classes

theorem teacher_distribution_count :
  valid_distributions 5 3 = 150 := by
  sorry

end teacher_distribution_count_l846_84664


namespace max_arithmetic_progressions_l846_84680

/-- A strictly increasing sequence of 101 real numbers -/
def StrictlyIncreasingSeq (a : Fin 101 → ℝ) : Prop :=
  ∀ i j, i < j → a i < a j

/-- Three terms form an arithmetic progression -/
def IsArithmeticProgression (x y z : ℝ) : Prop :=
  y = (x + z) / 2

/-- Count of arithmetic progressions in a sequence -/
def CountArithmeticProgressions (a : Fin 101 → ℝ) : ℕ :=
  (Finset.range 50).sum (fun i => i + 1) +
  (Finset.range 49).sum (fun i => i + 1)

/-- The main theorem -/
theorem max_arithmetic_progressions (a : Fin 101 → ℝ) 
  (h : StrictlyIncreasingSeq a) :
  CountArithmeticProgressions a = 2500 :=
sorry

end max_arithmetic_progressions_l846_84680


namespace fliers_remaining_l846_84651

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ)
  (h1 : total = 1000)
  (h2 : morning_fraction = 1 / 5)
  (h3 : afternoon_fraction = 1 / 4) :
  total - (morning_fraction * total).num - (afternoon_fraction * (total - (morning_fraction * total).num)).num = 600 :=
by sorry

end fliers_remaining_l846_84651


namespace floor_negative_seven_fourths_l846_84613

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end floor_negative_seven_fourths_l846_84613


namespace banana_permutations_l846_84603

theorem banana_permutations : 
  let total_letters : ℕ := 6
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  let b_count : ℕ := 1
  (total_letters.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
sorry

end banana_permutations_l846_84603


namespace cube_volume_from_diagonal_edge_distance_l846_84674

/-- The volume of a cube, given the distance from its space diagonal to a non-intersecting edge. -/
theorem cube_volume_from_diagonal_edge_distance (d : ℝ) (d_pos : 0 < d) : 
  ∃ (V : ℝ), V = 2 * d^3 * Real.sqrt 2 ∧ 
  (∃ (a : ℝ), a > 0 ∧ a = d * Real.sqrt 2 ∧ V = a^3) :=
by sorry

end cube_volume_from_diagonal_edge_distance_l846_84674


namespace amy_pencils_before_l846_84670

/-- The number of pencils Amy bought at the school store -/
def pencils_bought : ℕ := 7

/-- The total number of pencils Amy has now -/
def total_pencils : ℕ := 10

/-- The number of pencils Amy had before buying more -/
def pencils_before : ℕ := total_pencils - pencils_bought

theorem amy_pencils_before : pencils_before = 3 := by
  sorry

end amy_pencils_before_l846_84670


namespace maize_stolen_l846_84642

def months_in_year : ℕ := 12
def years : ℕ := 2
def maize_per_month : ℕ := 1
def donation : ℕ := 8
def final_amount : ℕ := 27

theorem maize_stolen : 
  (months_in_year * years * maize_per_month + donation) - final_amount = 5 := by
  sorry

end maize_stolen_l846_84642


namespace product_not_exceeding_sum_l846_84604

theorem product_not_exceeding_sum (x y : ℕ) (h : x * y ≤ x + y) :
  (x = 1 ∧ y ≥ 1) ∨ (x = 2 ∧ y = 2) := by
  sorry

end product_not_exceeding_sum_l846_84604


namespace imaginary_part_of_z_l846_84673

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 1 - 3 * Complex.I) : 
  z.im = -2 := by
sorry

end imaginary_part_of_z_l846_84673


namespace perpendicular_bisector_of_intersecting_curves_l846_84623

/-- Given two curves in polar coordinates that intersect, 
    prove the equation of the perpendicular bisector of their intersection points. -/
theorem perpendicular_bisector_of_intersecting_curves 
  (C₁ : ℝ → ℝ → Prop) 
  (C₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ θ ρ, C₁ ρ θ ↔ ρ = 2 * Real.sin θ)
  (h₂ : ∀ θ ρ, C₂ ρ θ ↔ ρ = 2 * Real.cos θ)
  (A B : ℝ × ℝ)
  (hA : C₁ A.1 A.2 ∧ C₂ A.1 A.2)
  (hB : C₁ B.1 B.2 ∧ C₂ B.1 B.2)
  (hAB : A ≠ B) :
  ∃ (ρ θ : ℝ), ρ * Real.sin θ + ρ * Real.cos θ = 1 :=
sorry

end perpendicular_bisector_of_intersecting_curves_l846_84623


namespace isosceles_triangle_base_length_l846_84689

/-- Represents an isosceles triangle with perimeter 16 and one side length 6 -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  perimeter_eq : side1 + side2 + base = 16
  one_side_6 : side1 = 6 ∨ side2 = 6 ∨ base = 6
  isosceles : side1 = side2 ∨ side1 = base ∨ side2 = base

/-- The base of the isosceles triangle is either 4 or 6 -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle) : t.base = 4 ∨ t.base = 6 := by
  sorry

end isosceles_triangle_base_length_l846_84689


namespace bus_journey_speed_l846_84606

theorem bus_journey_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_part_distance : ℝ) 
  (second_part_speed : ℝ) 
  (h1 : total_distance = 250) 
  (h2 : total_time = 5.2) 
  (h3 : first_part_distance = 124) 
  (h4 : second_part_speed = 60) :
  ∃ (first_part_speed : ℝ), 
    first_part_speed = 40 ∧ 
    first_part_distance / first_part_speed + 
    (total_distance - first_part_distance) / second_part_speed = total_time :=
by sorry

end bus_journey_speed_l846_84606


namespace railway_optimization_l846_84616

/-- The number of round trips per day as a function of the number of carriages -/
def t (n : ℕ) : ℤ := -2 * n + 24

/-- The number of passengers per day as a function of the number of carriages -/
def y (n : ℕ) : ℤ := t n * n * 110 * 2

theorem railway_optimization :
  (t 4 = 16 ∧ t 7 = 10) ∧ 
  (∀ n : ℕ, 1 ≤ n → n < 12 → y n ≤ y 6) ∧
  y 6 = 15840 := by
  sorry

#eval t 4  -- Expected: 16
#eval t 7  -- Expected: 10
#eval y 6  -- Expected: 15840

end railway_optimization_l846_84616


namespace total_precious_stones_l846_84637

theorem total_precious_stones (agate olivine sapphire diamond amethyst ruby : ℕ) : 
  agate = 25 →
  olivine = agate + 5 →
  sapphire = 2 * olivine →
  diamond = olivine + 11 →
  amethyst = sapphire + diamond →
  ruby = diamond + 7 →
  agate + olivine + sapphire + diamond + amethyst + ruby = 305 := by
  sorry

end total_precious_stones_l846_84637


namespace smallest_candy_count_l846_84676

theorem smallest_candy_count : 
  ∃ (n : ℕ), 
    100 ≤ n ∧ n < 1000 ∧ 
    (n + 7) % 9 = 0 ∧ 
    (n - 9) % 7 = 0 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < n ∧ (m + 7) % 9 = 0 ∧ (m - 9) % 7 = 0 → false :=
by
  -- The proof goes here
  sorry

end smallest_candy_count_l846_84676


namespace each_student_gets_seven_squares_l846_84688

/-- Calculates the number of chocolate squares each student receives -/
def chocolate_squares_per_student (gerald_bars : ℕ) (squares_per_bar : ℕ) (teacher_multiplier : ℕ) (num_students : ℕ) : ℕ :=
  let total_bars := gerald_bars + gerald_bars * teacher_multiplier
  let total_squares := total_bars * squares_per_bar
  total_squares / num_students

/-- Theorem stating that each student gets 7 squares of chocolate -/
theorem each_student_gets_seven_squares :
  chocolate_squares_per_student 7 8 2 24 = 7 := by
  sorry

end each_student_gets_seven_squares_l846_84688
