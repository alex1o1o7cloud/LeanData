import Mathlib

namespace NUMINAMATH_CALUDE_candidate_X_loses_by_6_percent_l1247_124700

/-- Represents the political parties --/
inductive Party
  | Republican
  | Democrat
  | Independent

/-- Represents the candidates --/
inductive Candidate
  | X
  | Y

/-- The ratio of registered voters for each party --/
def partyRatio : Party → Nat
  | Party.Republican => 3
  | Party.Democrat => 2
  | Party.Independent => 5

/-- The percentage of voters from each party expected to vote for candidate X --/
def votePercentageForX : Party → Rat
  | Party.Republican => 70/100
  | Party.Democrat => 30/100
  | Party.Independent => 40/100

/-- The percentage of registered voters who will not vote --/
def nonVoterPercentage : Rat := 10/100

/-- Theorem stating that candidate X is expected to lose by approximately 6% --/
theorem candidate_X_loses_by_6_percent :
  ∃ (total_voters : Nat),
    total_voters > 0 →
    let votes_for_X := (partyRatio Party.Republican * (votePercentageForX Party.Republican : Rat) +
                        partyRatio Party.Democrat * (votePercentageForX Party.Democrat : Rat) +
                        partyRatio Party.Independent * (votePercentageForX Party.Independent : Rat)) *
                       (1 - nonVoterPercentage) * total_voters
    let votes_for_Y := (partyRatio Party.Republican * (1 - votePercentageForX Party.Republican : Rat) +
                        partyRatio Party.Democrat * (1 - votePercentageForX Party.Democrat : Rat) +
                        partyRatio Party.Independent * (1 - votePercentageForX Party.Independent : Rat)) *
                       (1 - nonVoterPercentage) * total_voters
    let total_votes := votes_for_X + votes_for_Y
    let percentage_difference := (votes_for_Y - votes_for_X) / total_votes * 100
    abs (percentage_difference - 6) < 1 := by
  sorry

end NUMINAMATH_CALUDE_candidate_X_loses_by_6_percent_l1247_124700


namespace NUMINAMATH_CALUDE_cory_patio_set_cost_l1247_124721

def patio_set_cost (table_cost chair_cost : ℕ) (num_chairs : ℕ) : ℕ :=
  table_cost + num_chairs * chair_cost

theorem cory_patio_set_cost : patio_set_cost 55 20 4 = 135 := by
  sorry

end NUMINAMATH_CALUDE_cory_patio_set_cost_l1247_124721


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l1247_124779

theorem difference_of_squares_example : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l1247_124779


namespace NUMINAMATH_CALUDE_g_at_2_l1247_124754

-- Define the function g
def g (d : ℝ) (x : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + d * x - 8

-- State the theorem
theorem g_at_2 (d : ℝ) : g d (-2) = 4 → g d 2 = -84 := by
  sorry

end NUMINAMATH_CALUDE_g_at_2_l1247_124754


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l1247_124767

/-- For the equation (m-1)x^2 - 2x + 1 = 0 to be a quadratic equation, m ≠ 1 -/
theorem quadratic_equation_condition (m : ℝ) : 
  (∀ x, (m - 1) * x^2 - 2*x + 1 = 0 → (m - 1) ≠ 0) → m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l1247_124767


namespace NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l1247_124715

theorem floor_plus_x_eq_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17/4 ∧ x = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l1247_124715


namespace NUMINAMATH_CALUDE_board_longest_piece_length_l1247_124761

/-- Given a board of length 240 cm cut into four pieces, prove that the longest piece is 120 cm -/
theorem board_longest_piece_length :
  ∀ (L M T F : ℝ),
    L + M + T + F = 240 →
    L = M + T + F →
    M = L / 2 - 10 →
    T ^ 2 = L - M →
    L = 120 := by
  sorry

end NUMINAMATH_CALUDE_board_longest_piece_length_l1247_124761


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l1247_124705

/-- Given three positive integers with LCM 45600 and product 109183500000, their HCF is 2393750 -/
theorem hcf_from_lcm_and_product (a b c : ℕ+) 
  (h_lcm : Nat.lcm (a.val) (Nat.lcm (b.val) (c.val)) = 45600)
  (h_product : a * b * c = 109183500000) :
  Nat.gcd (a.val) (Nat.gcd (b.val) (c.val)) = 2393750 := by
  sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l1247_124705


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l1247_124707

theorem consecutive_integers_sum_of_cubes (n : ℕ) : 
  (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 8830 →
  (n - 1)^3 + n^3 + (n + 1)^3 + (n + 2)^3 = 52264 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l1247_124707


namespace NUMINAMATH_CALUDE_orange_harvest_theorem_l1247_124723

-- Define the number of days for the harvest
def harvest_days : ℕ := 4

-- Define the total number of sacks harvested
def total_sacks : ℕ := 56

-- Define the function to calculate sacks per day
def sacks_per_day (total : ℕ) (days : ℕ) : ℕ := total / days

-- Theorem statement
theorem orange_harvest_theorem : 
  sacks_per_day total_sacks harvest_days = 14 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_theorem_l1247_124723


namespace NUMINAMATH_CALUDE_abigail_report_words_l1247_124748

/-- Represents Abigail's report writing scenario -/
structure ReportWriting where
  typing_speed : ℕ  -- words per 30 minutes
  words_written : ℕ
  time_needed : ℕ  -- in minutes

/-- Calculates the total number of words in the report -/
def total_words (r : ReportWriting) : ℕ :=
  r.words_written + r.typing_speed * r.time_needed / 30

/-- Theorem stating that the total words in Abigail's report is 1000 -/
theorem abigail_report_words :
  ∃ (r : ReportWriting), r.typing_speed = 300 ∧ r.words_written = 200 ∧ r.time_needed = 80 ∧ total_words r = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_abigail_report_words_l1247_124748


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_is_correct_l1247_124733

/-- The number of zeros after the decimal point and before the first non-zero digit
    in the terminating decimal representation of 1/(2^7 * 5^3) -/
def zeros_before_first_nonzero : ℕ :=
  4

/-- The fraction we're considering -/
def fraction : ℚ :=
  1 / (2^7 * 5^3)

/-- Theorem stating that the number of zeros before the first non-zero digit
    in the terminating decimal representation of our fraction is correct -/
theorem zeros_before_first_nonzero_is_correct :
  zeros_before_first_nonzero = 4 ∧
  ∃ (n : ℕ), fraction * 10^zeros_before_first_nonzero = n / 10^zeros_before_first_nonzero ∧
             n % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_is_correct_l1247_124733


namespace NUMINAMATH_CALUDE_ten_people_round_table_with_pair_l1247_124784

/-- The number of ways to arrange n people around a round table -/
def roundTableArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange n people around a round table
    when two specific people must sit next to each other -/
def roundTableArrangementsWithPair (n : ℕ) : ℕ :=
  2 * roundTableArrangements (n - 1)

/-- Theorem: There are 80,640 ways to arrange 10 people around a round table
    when two specific people must sit next to each other -/
theorem ten_people_round_table_with_pair :
  roundTableArrangementsWithPair 10 = 80640 := by
  sorry

end NUMINAMATH_CALUDE_ten_people_round_table_with_pair_l1247_124784


namespace NUMINAMATH_CALUDE_john_relatives_money_l1247_124780

theorem john_relatives_money (grandpa : ℕ) : 
  grandpa = 30 → 
  (grandpa + 3 * grandpa + 2 * grandpa + (3 * grandpa) / 2 : ℕ) = 225 := by
  sorry

end NUMINAMATH_CALUDE_john_relatives_money_l1247_124780


namespace NUMINAMATH_CALUDE_nesbitt_inequality_l1247_124742

theorem nesbitt_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 ∧
  (a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_nesbitt_inequality_l1247_124742


namespace NUMINAMATH_CALUDE_association_members_after_four_years_l1247_124725

/-- Represents the number of people in the association after k years -/
def association_members (k : ℕ) : ℕ :=
  match k with
  | 0 => 20
  | n + 1 => 4 * association_members n - 18

/-- The number of people in the association after 4 years is 3590 -/
theorem association_members_after_four_years :
  association_members 4 = 3590 := by
  sorry

end NUMINAMATH_CALUDE_association_members_after_four_years_l1247_124725


namespace NUMINAMATH_CALUDE_mardi_gras_necklaces_l1247_124782

theorem mardi_gras_necklaces 
  (boudreaux_necklaces : ℕ)
  (rhonda_necklaces : ℕ)
  (latch_necklaces : ℕ)
  (h1 : boudreaux_necklaces = 12)
  (h2 : rhonda_necklaces = boudreaux_necklaces / 2)
  (h3 : latch_necklaces = 3 * rhonda_necklaces - 4)
  : latch_necklaces = 14 := by
  sorry

end NUMINAMATH_CALUDE_mardi_gras_necklaces_l1247_124782


namespace NUMINAMATH_CALUDE_summer_work_hours_adjustment_l1247_124717

theorem summer_work_hours_adjustment 
  (initial_weeks : ℕ) 
  (initial_hours_per_week : ℝ) 
  (unavailable_weeks : ℕ) 
  (adjusted_hours_per_week : ℝ) :
  initial_weeks > unavailable_weeks →
  initial_weeks * initial_hours_per_week = 
    (initial_weeks - unavailable_weeks) * adjusted_hours_per_week →
  adjusted_hours_per_week = initial_hours_per_week * (initial_weeks / (initial_weeks - unavailable_weeks)) :=
by
  sorry

#eval (31.25 : Float)

end NUMINAMATH_CALUDE_summer_work_hours_adjustment_l1247_124717


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1247_124713

theorem closest_integer_to_cube_root : ∃ n : ℤ, 
  n = 10 ∧ ∀ m : ℤ, m ≠ n → |n - (7^3 + 9^3 + 3)^(1/3)| < |m - (7^3 + 9^3 + 3)^(1/3)| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1247_124713


namespace NUMINAMATH_CALUDE_ball_probability_l1247_124706

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 30)
  (h_yellow : yellow = 10)
  (h_red : red = 7)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l1247_124706


namespace NUMINAMATH_CALUDE_mashas_measurements_impossible_l1247_124732

/-- A pentagon inscribed in a circle with given interior angles -/
structure InscribedPentagon where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ

/-- The sum of interior angles of a pentagon is 540° -/
axiom pentagon_angle_sum (p : InscribedPentagon) :
  p.angle1 + p.angle2 + p.angle3 + p.angle4 + p.angle5 = 540

/-- Opposite angles in an inscribed quadrilateral sum to 180° -/
axiom inscribed_quadrilateral_opposite_angles (a b : ℝ) :
  a + b = 180 → ∃ (p : InscribedPentagon), p.angle1 = a ∧ p.angle3 = b

/-- Masha's measurements -/
def mashas_pentagon : InscribedPentagon := {
  angle1 := 80,
  angle2 := 90,
  angle3 := 100,
  angle4 := 130,
  angle5 := 140
}

/-- Theorem: Masha's measurements are impossible for a pentagon inscribed in a circle -/
theorem mashas_measurements_impossible : 
  ¬∃ (p : InscribedPentagon), p = mashas_pentagon :=
sorry

end NUMINAMATH_CALUDE_mashas_measurements_impossible_l1247_124732


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1247_124749

theorem quadratic_inequality_solution_set : 
  {x : ℝ | x^2 - 50*x + 500 ≤ 9} = {x : ℝ | 13.42 ≤ x ∧ x ≤ 36.58} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1247_124749


namespace NUMINAMATH_CALUDE_percentage_passed_both_l1247_124768

theorem percentage_passed_both (total : ℕ) (h : total > 0) :
  let failed_hindi := (25 : ℕ) * total / 100
  let failed_english := (50 : ℕ) * total / 100
  let failed_both := (25 : ℕ) * total / 100
  let passed_both := total - (failed_hindi + failed_english - failed_both)
  (passed_both * 100 : ℕ) / total = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_passed_both_l1247_124768


namespace NUMINAMATH_CALUDE_twentieth_term_is_41_l1247_124734

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem twentieth_term_is_41 :
  arithmetic_sequence 3 2 20 = 41 := by sorry

end NUMINAMATH_CALUDE_twentieth_term_is_41_l1247_124734


namespace NUMINAMATH_CALUDE_paper_sheets_calculation_l1247_124789

theorem paper_sheets_calculation (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) :
  num_classes = 4 →
  students_per_class = 20 →
  sheets_per_student = 5 →
  num_classes * students_per_class * sheets_per_student = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_paper_sheets_calculation_l1247_124789


namespace NUMINAMATH_CALUDE_antonios_meatballs_l1247_124724

/-- Antonio's meatball problem -/
theorem antonios_meatballs (recipe_amount : ℚ) (family_members : ℕ) (total_hamburger : ℚ) : 
  recipe_amount = 1/8 →
  family_members = 8 →
  total_hamburger = 4 →
  (total_hamburger / recipe_amount) / family_members = 4 :=
by sorry

end NUMINAMATH_CALUDE_antonios_meatballs_l1247_124724


namespace NUMINAMATH_CALUDE_octal_addition_example_l1247_124744

/-- Represents a digit in the octal number system -/
def OctalDigit : Type := Fin 8

/-- Represents an octal number as a list of octal digits -/
def OctalNumber : Type := List OctalDigit

/-- Addition operation for octal numbers -/
def octal_add : OctalNumber → OctalNumber → OctalNumber :=
  sorry

/-- Conversion from a natural number to an octal number -/
def nat_to_octal : Nat → OctalNumber :=
  sorry

/-- Theorem: 47 + 56 = 125 in the octal number system -/
theorem octal_addition_example :
  octal_add (nat_to_octal 47) (nat_to_octal 56) = nat_to_octal 125 := by
  sorry

end NUMINAMATH_CALUDE_octal_addition_example_l1247_124744


namespace NUMINAMATH_CALUDE_middle_three_sum_is_twelve_l1247_124753

/-- Represents a card with a color and a number -/
inductive Card
  | red (n : Nat)
  | blue (n : Nat)

/-- Checks if a number divides another number -/
def divides (a b : Nat) : Bool :=
  b % a == 0

/-- Checks if a stack of cards satisfies the alternating color and division rules -/
def validStack (stack : List Card) : Bool :=
  match stack with
  | [] => true
  | [_] => true
  | (Card.blue b) :: (Card.red r) :: (Card.blue b') :: rest =>
      divides r b && divides r b' && validStack ((Card.red r) :: (Card.blue b') :: rest)
  | _ => false

/-- Returns the sum of the numbers on the middle three cards -/
def middleThreeSum (stack : List Card) : Nat :=
  let mid := stack.length / 2
  match (stack.get? (mid - 1), stack.get? mid, stack.get? (mid + 1)) with
  | (some (Card.blue b1), some (Card.red r), some (Card.blue b2)) => b1 + r + b2
  | _ => 0

/-- The main theorem -/
theorem middle_three_sum_is_twelve :
  ∃ (stack : List Card),
    stack.length = 9 ∧
    stack.head? = some (Card.blue 2) ∧
    (∀ n, 1 ≤ n ∧ n ≤ 4 → (Card.red n) ∈ stack) ∧
    (∀ n, 2 ≤ n ∧ n ≤ 6 → (Card.blue n) ∈ stack) ∧
    validStack stack ∧
    middleThreeSum stack = 12 :=
  sorry


end NUMINAMATH_CALUDE_middle_three_sum_is_twelve_l1247_124753


namespace NUMINAMATH_CALUDE_ninth_grade_students_l1247_124703

theorem ninth_grade_students (S : ℕ) : 
  (S / 4 : ℚ) + (3 * S / 4 / 3 : ℚ) + 20 + 70 = S → S = 180 := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_students_l1247_124703


namespace NUMINAMATH_CALUDE_kitchen_length_l1247_124756

/-- Calculates the length of a rectangular kitchen given its dimensions and painting information. -/
theorem kitchen_length (width height : ℝ) (total_area_painted : ℝ) : 
  width = 16 ∧ 
  height = 10 ∧ 
  total_area_painted = 1680 → 
  ∃ length : ℝ, length = 12 ∧ 
    total_area_painted / 3 = 2 * (length * height + width * height) :=
by sorry

end NUMINAMATH_CALUDE_kitchen_length_l1247_124756


namespace NUMINAMATH_CALUDE_square_rectangle_overlap_ratio_l1247_124708

theorem square_rectangle_overlap_ratio : 
  ∀ (s x y : ℝ),
  s > 0 → x > 0 → y > 0 →
  (0.25 * s^2 = 0.4 * x * y) →
  (y = s) →
  (x / y = 5 / 8) := by
sorry

end NUMINAMATH_CALUDE_square_rectangle_overlap_ratio_l1247_124708


namespace NUMINAMATH_CALUDE_general_term_formula_l1247_124716

/-- The sequence defined by the problem -/
def a (n : ℕ+) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 3/4
  else if n = 3 then 5/9
  else if n = 4 then 7/16
  else (2*n - 1) / (n^2)

/-- The theorem stating that the general term formula is correct -/
theorem general_term_formula (n : ℕ+) : a n = (2*n - 1) / (n^2) := by
  sorry

end NUMINAMATH_CALUDE_general_term_formula_l1247_124716


namespace NUMINAMATH_CALUDE_committee_meeting_attendance_l1247_124759

theorem committee_meeting_attendance :
  ∀ (associate_profs assistant_profs : ℕ),
    2 * associate_profs + assistant_profs = 11 →
    associate_profs + 2 * assistant_profs = 16 →
    associate_profs + assistant_profs = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_committee_meeting_attendance_l1247_124759


namespace NUMINAMATH_CALUDE_lottery_problem_l1247_124785

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Yellow

/-- Represents the contents of the bag -/
structure Bag :=
  (red : ℕ)
  (yellow : ℕ)

/-- Calculates the probability of drawing a red ball -/
def prob_red (b : Bag) : ℚ :=
  b.red / (b.red + b.yellow)

/-- Calculates the probability of drawing two balls of the same color -/
def prob_same_color (b : Bag) : ℚ :=
  let total := b.red + b.yellow
  (b.red * (b.red - 1) + b.yellow * (b.yellow - 1)) / (total * (total - 1))

theorem lottery_problem :
  let initial_bag : Bag := ⟨1, 3⟩
  let red_added_bag : Bag := ⟨2, 3⟩
  let yellow_added_bag : Bag := ⟨1, 4⟩
  (prob_red initial_bag = 1/4) ∧
  (prob_same_color yellow_added_bag > prob_same_color red_added_bag) := by
  sorry


end NUMINAMATH_CALUDE_lottery_problem_l1247_124785


namespace NUMINAMATH_CALUDE_delegates_with_female_count_l1247_124727

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose delegates with at least one female student. -/
def delegates_with_female (male_count female_count delegate_count : ℕ) : ℕ :=
  (choose female_count 1 * choose male_count (delegate_count - 1)) +
  (choose female_count 2 * choose male_count (delegate_count - 2)) +
  (choose female_count 3 * choose male_count (delegate_count - 3))

theorem delegates_with_female_count :
  delegates_with_female 4 3 3 = 31 := by sorry

end NUMINAMATH_CALUDE_delegates_with_female_count_l1247_124727


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1247_124752

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > 2 → x > 1) ∧ ¬(x > 1 → x > 2) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l1247_124752


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1247_124737

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℝ, 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 36 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1247_124737


namespace NUMINAMATH_CALUDE_largest_mu_inequality_l1247_124769

theorem largest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∀ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ a*b + μ*b*c + 2*c*d) → μ ≤ 3/4) ∧
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a^2 + b^2 + c^2 + d^2 ≥ a*b + 3/4*b*c + 2*c*d) :=
by sorry

end NUMINAMATH_CALUDE_largest_mu_inequality_l1247_124769


namespace NUMINAMATH_CALUDE_equation_solution_l1247_124711

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1247_124711


namespace NUMINAMATH_CALUDE_find_k_value_l1247_124770

theorem find_k_value (k : ℝ) : 
  (∃ x y : ℝ, x = 2 ∧ y = -1 ∧ y - k * x = 7) → k = -4 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l1247_124770


namespace NUMINAMATH_CALUDE_largest_non_sum_of_30multiple_and_composite_l1247_124772

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The statement to be proved -/
theorem largest_non_sum_of_30multiple_and_composite :
  ∀ n : ℕ, n > 93 →
    ∃ (k : ℕ) (c : ℕ), k > 0 ∧ isComposite c ∧ n = 30 * k + c :=
by sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_30multiple_and_composite_l1247_124772


namespace NUMINAMATH_CALUDE_roots_less_than_one_l1247_124743

theorem roots_less_than_one (a b : ℝ) (h : abs a + abs b < 1) :
  ∀ x, x^2 + a*x + b = 0 → abs x < 1 :=
sorry

end NUMINAMATH_CALUDE_roots_less_than_one_l1247_124743


namespace NUMINAMATH_CALUDE_least_sum_with_conditions_l1247_124796

theorem least_sum_with_conditions (m n : ℕ+) 
  (h1 : Nat.gcd (m + n) 210 = 1)
  (h2 : ∃ k : ℕ, m^m.val = k * n^n.val)
  (h3 : ¬∃ k : ℕ, m = k * n) :
  (∀ p q : ℕ+, 
    Nat.gcd (p + q) 210 = 1 → 
    (∃ k : ℕ, p^p.val = k * q^q.val) → 
    (¬∃ k : ℕ, p = k * q) → 
    m + n ≤ p + q) →
  m + n = 407 := by
sorry

end NUMINAMATH_CALUDE_least_sum_with_conditions_l1247_124796


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1247_124738

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 15*x + 6 = 0 → ∃ s t : ℝ, s + t = 15 ∧ s*t = 6 ∧ s^2 + t^2 = 213 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1247_124738


namespace NUMINAMATH_CALUDE_inequality_solution_l1247_124786

theorem inequality_solution (x : ℝ) :
  (x - 4) / (x^2 + 3*x + 10) ≥ 0 ↔ x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1247_124786


namespace NUMINAMATH_CALUDE_solution_mixing_l1247_124701

theorem solution_mixing (x y : Real) :
  x + y = 40 →
  0.30 * x + 0.80 * y = 0.45 * 40 →
  y = 12 →
  x = 28 →
  0.30 * 28 + 0.80 * 12 = 0.45 * 40 :=
by sorry

end NUMINAMATH_CALUDE_solution_mixing_l1247_124701


namespace NUMINAMATH_CALUDE_monkey_apple_problem_l1247_124739

/-- Given a number of monkeys and apples, this function checks if they satisfy the conditions:
    1. If each monkey gets 3 apples, there will be 6 left.
    2. If each monkey gets 4 apples, the last monkey will get less than 4 apples. -/
def satisfies_conditions (monkeys : ℕ) (apples : ℕ) : Prop :=
  (apples = 3 * monkeys + 6) ∧ 
  (apples < 4 * monkeys) ∧ 
  (apples > 4 * (monkeys - 1))

/-- Theorem stating that the only solutions satisfying the conditions are
    (7 monkeys, 27 apples), (8 monkeys, 30 apples), or (9 monkeys, 33 apples) -/
theorem monkey_apple_problem :
  ∀ monkeys apples : ℕ, 
    satisfies_conditions monkeys apples ↔ 
    ((monkeys = 7 ∧ apples = 27) ∨ 
     (monkeys = 8 ∧ apples = 30) ∨ 
     (monkeys = 9 ∧ apples = 33)) :=
by sorry

end NUMINAMATH_CALUDE_monkey_apple_problem_l1247_124739


namespace NUMINAMATH_CALUDE_f_max_value_l1247_124788

/-- The quadratic function f(y) = -9y^2 + 15y + 3 -/
def f (y : ℝ) := -9 * y^2 + 15 * y + 3

/-- The maximum value of f(y) is 6.25 -/
theorem f_max_value : ∃ (y : ℝ), f y = 6.25 ∧ ∀ (z : ℝ), f z ≤ 6.25 := by
  sorry

end NUMINAMATH_CALUDE_f_max_value_l1247_124788


namespace NUMINAMATH_CALUDE_product_sum_relation_l1247_124720

theorem product_sum_relation (a b : ℝ) : 
  (a * b = 2 * (a + b) + 1) → (b = 7) → (b - a = 4) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l1247_124720


namespace NUMINAMATH_CALUDE_parabola_point_comparison_l1247_124771

/-- Parabola function -/
def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

/-- Theorem: For the parabola y = -x^2 + 2x - 2, if (-2, y₁) and (3, y₂) are points on the parabola, then y₁ < y₂ -/
theorem parabola_point_comparison (y₁ y₂ : ℝ) 
  (h₁ : f (-2) = y₁) 
  (h₂ : f 3 = y₂) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_comparison_l1247_124771


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l1247_124760

theorem quadratic_solution_property (k : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 + 5 * x + k = 0 ∧ 3 * y^2 + 5 * y + k = 0 ∧ 
   |x + y| = x^2 + y^2) ↔ k = -10/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l1247_124760


namespace NUMINAMATH_CALUDE_backpack_price_equation_l1247_124731

/-- Represents the price of a backpack after discounts -/
def discounted_price (x : ℝ) : ℝ := 0.8 * x - 10

/-- Theorem stating that the discounted price equals the final selling price -/
theorem backpack_price_equation (x : ℝ) : 
  discounted_price x = 90 ↔ 0.8 * x - 10 = 90 := by sorry

end NUMINAMATH_CALUDE_backpack_price_equation_l1247_124731


namespace NUMINAMATH_CALUDE_tangent_line_problem_l1247_124758

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := (k + 1) * x^3 - 3 * (k + 2) * x^2 - k^2 - 2 * k

-- Define the derivative of f
def f' (k : ℝ) (x : ℝ) : ℝ := 3 * (k + 1) * x^2 - 6 * (k + 2) * x

theorem tangent_line_problem (k : ℝ) (h1 : k > -1) :
  (∀ x ∈ Set.Ioo 0 4, f' k x < 0) →
  (k = 0 ∧ 
   ∃ t : ℝ, t = f' 0 1 ∧ 9 * 1 + (-5) + 4 = 0 ∧ 
   ∀ x y : ℝ, y = t * (x - 1) + (-5) ↔ 9 * x + y + 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l1247_124758


namespace NUMINAMATH_CALUDE_boundaries_hit_l1247_124747

def total_runs : ℕ := 120
def sixes_hit : ℕ := 8
def runs_per_six : ℕ := 6
def runs_per_boundary : ℕ := 4

theorem boundaries_hit :
  let runs_by_running := total_runs / 2
  let runs_by_sixes := sixes_hit * runs_per_six
  let runs_by_boundaries := total_runs - runs_by_running - runs_by_sixes
  runs_by_boundaries / runs_per_boundary = 3 := by sorry

end NUMINAMATH_CALUDE_boundaries_hit_l1247_124747


namespace NUMINAMATH_CALUDE_poly_factorable_iff_l1247_124777

/-- A polynomial in x and y with a parameter k -/
def poly (k : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + x + k*y - k

/-- A linear factor with integer coefficients -/
structure LinearFactor where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Check if a polynomial can be factored into two linear factors -/
def isFactorable (k : ℤ) : Prop :=
  ∃ (f g : LinearFactor), ∀ (x y : ℤ),
    poly k x y = (f.a * x + f.b * y + f.c) * (g.a * x + g.b * y + g.c)

/-- The main theorem: the polynomial is factorable iff k = 0 or k = 16 -/
theorem poly_factorable_iff (k : ℤ) : isFactorable k ↔ k = 0 ∨ k = 16 :=
sorry

end NUMINAMATH_CALUDE_poly_factorable_iff_l1247_124777


namespace NUMINAMATH_CALUDE_bakery_weekly_sales_l1247_124740

/-- Represents the daily sales of cakes for a specific type -/
structure DailySales :=
  (monday : Nat)
  (tuesday : Nat)
  (wednesday : Nat)
  (thursday : Nat)
  (friday : Nat)
  (saturday : Nat)
  (sunday : Nat)

/-- Represents the weekly sales data for all cake types -/
structure WeeklySales :=
  (chocolate : DailySales)
  (vanilla : DailySales)
  (strawberry : DailySales)

def bakery_sales : WeeklySales :=
  { chocolate := { monday := 6, tuesday := 7, wednesday := 4, thursday := 8, friday := 9, saturday := 10, sunday := 5 },
    vanilla := { monday := 4, tuesday := 5, wednesday := 3, thursday := 7, friday := 6, saturday := 8, sunday := 4 },
    strawberry := { monday := 3, tuesday := 2, wednesday := 6, thursday := 4, friday := 5, saturday := 7, sunday := 4 } }

def total_sales (sales : DailySales) : Nat :=
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday + sales.friday + sales.saturday + sales.sunday

theorem bakery_weekly_sales :
  total_sales bakery_sales.chocolate = 49 ∧
  total_sales bakery_sales.vanilla = 37 ∧
  total_sales bakery_sales.strawberry = 31 := by
  sorry

end NUMINAMATH_CALUDE_bakery_weekly_sales_l1247_124740


namespace NUMINAMATH_CALUDE_no_intersection_l1247_124794

theorem no_intersection : ¬∃ x : ℝ, |3*x + 6| = -|4*x - 3| := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_l1247_124794


namespace NUMINAMATH_CALUDE_no_integer_square_root_Q_l1247_124762

/-- The polynomial Q(x) = x^4 + 8x^3 + 18x^2 + 11x + 27 -/
def Q (x : ℤ) : ℤ := x^4 + 8*x^3 + 18*x^2 + 11*x + 27

/-- Theorem stating that there are no integer values of x for which Q(x) is a perfect square -/
theorem no_integer_square_root_Q :
  ∀ x : ℤ, ¬∃ y : ℤ, Q x = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_square_root_Q_l1247_124762


namespace NUMINAMATH_CALUDE_inequality_proof_l1247_124730

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x - (1 + a) * x - b

theorem inequality_proof (a b : ℝ) (h : ∀ x, f a b x ≥ 0) : b * (a + 1) / 2 < 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1247_124730


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l1247_124757

theorem dennis_teaching_years 
  (total_years : ℕ) 
  (virginia_adrienne_diff : ℕ) 
  (dennis_virginia_diff : ℕ) 
  (h1 : total_years = 102)
  (h2 : virginia_adrienne_diff = 9)
  (h3 : dennis_virginia_diff = 9)
  : ∃ (adrienne virginia dennis : ℕ),
    adrienne + virginia + dennis = total_years ∧
    virginia = adrienne + virginia_adrienne_diff ∧
    dennis = virginia + dennis_virginia_diff ∧
    dennis = 43 :=
by sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l1247_124757


namespace NUMINAMATH_CALUDE_false_balance_inequality_l1247_124763

/-- A false balance with two pans A and B -/
structure FalseBalance where
  l : ℝ  -- length of arm A
  l' : ℝ  -- length of arm B
  false_balance : l ≠ l'

/-- The balance condition for the false balance -/
def balances (b : FalseBalance) (w1 w2 : ℝ) (on_a : Bool) : Prop :=
  if on_a then w1 * b.l = w2 * b.l' else w1 * b.l' = w2 * b.l

theorem false_balance_inequality (b : FalseBalance) (p x y : ℝ) 
  (h1 : balances b p x false)
  (h2 : balances b p y true) :
  x + y > 2 * p := by
  sorry

end NUMINAMATH_CALUDE_false_balance_inequality_l1247_124763


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1247_124750

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 = 16*x - 10) → (∃ y : ℝ, y^2 = 16*y - 10 ∧ x + y = 16) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1247_124750


namespace NUMINAMATH_CALUDE_music_tool_cost_proof_l1247_124755

/-- The cost of Joan's purchases at the music store -/
def total_spent : ℚ := 163.28

/-- The cost of the trumpet Joan bought -/
def trumpet_cost : ℚ := 149.16

/-- The cost of the song book Joan bought -/
def song_book_cost : ℚ := 4.14

/-- The cost of the music tool -/
def music_tool_cost : ℚ := total_spent - trumpet_cost - song_book_cost

theorem music_tool_cost_proof : music_tool_cost = 9.98 := by
  sorry

end NUMINAMATH_CALUDE_music_tool_cost_proof_l1247_124755


namespace NUMINAMATH_CALUDE_sportswear_problem_l1247_124798

/-- Sportswear Problem -/
theorem sportswear_problem 
  (first_batch_cost : ℝ) 
  (second_batch_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : first_batch_cost = 12000)
  (h2 : second_batch_cost = 26400)
  (h3 : selling_price = 150) :
  ∃ (first_batch_quantity second_batch_quantity : ℕ),
    (second_batch_quantity = 2 * first_batch_quantity) ∧
    (second_batch_cost / second_batch_quantity = first_batch_cost / first_batch_quantity + 10) ∧
    (second_batch_quantity = 240) ∧
    (first_batch_quantity * (selling_price - first_batch_cost / first_batch_quantity) +
     second_batch_quantity * (selling_price - second_batch_cost / second_batch_quantity) = 15600) := by
  sorry

end NUMINAMATH_CALUDE_sportswear_problem_l1247_124798


namespace NUMINAMATH_CALUDE_list_number_fraction_l1247_124783

theorem list_number_fraction (list : List ℝ) (n : ℝ) :
  list.length = 21 →
  n ∈ list →
  list.Pairwise (·≠·) →
  n = 4 * ((list.sum - n) / 20) →
  n = (1 / 6) * list.sum :=
by sorry

end NUMINAMATH_CALUDE_list_number_fraction_l1247_124783


namespace NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l1247_124775

theorem tan_fifteen_pi_fourths : Real.tan (15 * π / 4) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l1247_124775


namespace NUMINAMATH_CALUDE_mike_baseball_cards_l1247_124773

/-- 
Given that Mike initially has 87 baseball cards and Sam buys 13 of them,
prove that Mike will have 74 baseball cards remaining.
-/
theorem mike_baseball_cards (initial_cards : ℕ) (bought_cards : ℕ) (remaining_cards : ℕ) :
  initial_cards = 87 →
  bought_cards = 13 →
  remaining_cards = initial_cards - bought_cards →
  remaining_cards = 74 := by
sorry

end NUMINAMATH_CALUDE_mike_baseball_cards_l1247_124773


namespace NUMINAMATH_CALUDE_bike_price_l1247_124774

theorem bike_price (P : ℝ) : P + 0.1 * P = 82500 → P = 75000 := by
  sorry

end NUMINAMATH_CALUDE_bike_price_l1247_124774


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l1247_124702

theorem sine_cosine_inequality (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c :=
sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l1247_124702


namespace NUMINAMATH_CALUDE_hexagon_reflected_arcs_area_l1247_124766

-- Define the side length of the hexagon
def side_length : ℝ := 2

-- Define the number of sides in a hexagon
def num_sides : ℕ := 6

-- Theorem statement
theorem hexagon_reflected_arcs_area :
  let r := side_length / Real.sqrt 3
  let hexagon_area := 3 * Real.sqrt 3 / 2 * side_length^2
  let sector_area := π * r^2 / 6
  let triangle_area := Real.sqrt 3 / 4 * side_length^2
  let reflected_arc_area := sector_area - triangle_area
  hexagon_area - num_sides * reflected_arc_area = 12 * Real.sqrt 3 - 8 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_reflected_arcs_area_l1247_124766


namespace NUMINAMATH_CALUDE_probability_adjacent_is_two_thirds_l1247_124735

/-- The number of ways to arrange 3 distinct objects in a row -/
def total_arrangements : ℕ := 6

/-- The number of arrangements where A and B are adjacent -/
def adjacent_arrangements : ℕ := 4

/-- The probability of A and B being adjacent when A, B, and C stand in a row -/
def probability_adjacent : ℚ := adjacent_arrangements / total_arrangements

theorem probability_adjacent_is_two_thirds :
  probability_adjacent = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_adjacent_is_two_thirds_l1247_124735


namespace NUMINAMATH_CALUDE_street_trees_l1247_124793

theorem street_trees (road_length : ℕ) (tree_interval : ℕ) (h1 : road_length = 2575) (h2 : tree_interval = 25) : 
  (road_length / tree_interval) + 1 = 104 := by
  sorry

end NUMINAMATH_CALUDE_street_trees_l1247_124793


namespace NUMINAMATH_CALUDE_product_45_sum_5_l1247_124797

theorem product_45_sum_5 (v w x y z : ℤ) : 
  v ≠ w ∧ v ≠ x ∧ v ≠ y ∧ v ≠ z ∧
  w ≠ x ∧ w ≠ y ∧ w ≠ z ∧
  x ≠ y ∧ x ≠ z ∧
  y ≠ z ∧
  v * w * x * y * z = 45 →
  v + w + x + y + z = 5 := by
sorry

end NUMINAMATH_CALUDE_product_45_sum_5_l1247_124797


namespace NUMINAMATH_CALUDE_only_B_in_third_quadrant_l1247_124790

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of being in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The given points -/
def pointA : Point := ⟨2, 3⟩
def pointB : Point := ⟨-1, -4⟩
def pointC : Point := ⟨-4, 1⟩
def pointD : Point := ⟨5, -3⟩

/-- Theorem stating that only point B is in the third quadrant -/
theorem only_B_in_third_quadrant :
  ¬isInThirdQuadrant pointA ∧
  isInThirdQuadrant pointB ∧
  ¬isInThirdQuadrant pointC ∧
  ¬isInThirdQuadrant pointD :=
sorry

end NUMINAMATH_CALUDE_only_B_in_third_quadrant_l1247_124790


namespace NUMINAMATH_CALUDE_trig_identity_special_case_l1247_124765

theorem trig_identity_special_case : 
  Real.cos (60 * π / 180 + 30 * π / 180) * Real.cos (60 * π / 180 - 30 * π / 180) + 
  Real.sin (60 * π / 180 + 30 * π / 180) * Real.sin (60 * π / 180 - 30 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_special_case_l1247_124765


namespace NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l1247_124718

def lunch_cost : ℝ := 60.50
def total_spent : ℝ := 72.6

theorem tip_percentage_is_twenty_percent :
  (total_spent - lunch_cost) / lunch_cost * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l1247_124718


namespace NUMINAMATH_CALUDE_rectangles_may_not_be_similar_squares_always_similar_equilateral_triangles_always_similar_isosceles_right_triangles_always_similar_l1247_124719

-- Define the shapes
structure Square where
  side : ℝ
  side_positive : side > 0

structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

structure IsoscelesRightTriangle where
  leg : ℝ
  leg_positive : leg > 0

structure Rectangle where
  length : ℝ
  width : ℝ
  length_positive : length > 0
  width_positive : width > 0

-- Define similarity
def similar {α : Type*} (x y : α) : Prop := sorry

-- Theorem stating that rectangles may not always be similar
theorem rectangles_may_not_be_similar :
  ∃ (r1 r2 : Rectangle), ¬ similar r1 r2 :=
sorry

-- Theorems stating that other shapes are always similar
theorem squares_always_similar (s1 s2 : Square) :
  similar s1 s2 :=
sorry

theorem equilateral_triangles_always_similar (t1 t2 : EquilateralTriangle) :
  similar t1 t2 :=
sorry

theorem isosceles_right_triangles_always_similar (t1 t2 : IsoscelesRightTriangle) :
  similar t1 t2 :=
sorry

end NUMINAMATH_CALUDE_rectangles_may_not_be_similar_squares_always_similar_equilateral_triangles_always_similar_isosceles_right_triangles_always_similar_l1247_124719


namespace NUMINAMATH_CALUDE_probability_cos_pi_x_geq_half_over_interval_probability_equals_one_third_l1247_124704

/-- The probability that cos(πx) ≥ 1/2 for x uniformly distributed in [-1, 1] -/
theorem probability_cos_pi_x_geq_half_over_interval (x : ℝ) : 
  ℝ := by sorry

/-- The probability is equal to 1/3 -/
theorem probability_equals_one_third : 
  probability_cos_pi_x_geq_half_over_interval = 1/3 := by sorry

end NUMINAMATH_CALUDE_probability_cos_pi_x_geq_half_over_interval_probability_equals_one_third_l1247_124704


namespace NUMINAMATH_CALUDE_expression_equality_l1247_124710

theorem expression_equality : (2023^2 - 2015^2) / (2030^2 - 2008^2) = 4/11 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1247_124710


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l1247_124746

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a^2 - 1) * Real.exp (a * x)

-- Define the derivative of f
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then 2 * a * x
  else if x < 0 then a * (a^2 - 1) * Real.exp (a * x)
  else if a > 1 then 2 * a else a * (a^2 - 1)

-- Theorem statement
theorem f_increasing_iff_a_in_range (a : ℝ) :
  (∀ x, f' a x > 0) ↔ (1 < a ∧ a ≤ Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l1247_124746


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l1247_124764

/-- The equation has exactly one solution when its discriminant is zero -/
def has_one_solution (a : ℝ) (k : ℝ) : Prop :=
  (a + 1/a + 1)^2 - 4*k = 0

/-- The condition for exactly one positive value of a -/
def unique_positive_a (k : ℝ) : Prop :=
  ∃! a : ℝ, a > 0 ∧ has_one_solution a k

theorem quadratic_equation_unique_solution :
  ∃! k : ℝ, k ≠ 0 ∧ unique_positive_a k ∧ k = 1/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l1247_124764


namespace NUMINAMATH_CALUDE_like_terms_imply_sum_of_squares_l1247_124736

theorem like_terms_imply_sum_of_squares (m n : ℤ) : 
  (m + 10 = 3*n - m) → (7 - n = n - m) → m^2 - 2*m*n + n^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_sum_of_squares_l1247_124736


namespace NUMINAMATH_CALUDE_star_equation_has_two_distinct_real_roots_l1247_124799

-- Define the ☆ operation
def star (m n : ℝ) : ℝ := m * n^2 - m * n - 1

-- Theorem statement
theorem star_equation_has_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ star 1 x₁ = 0 ∧ star 1 x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_star_equation_has_two_distinct_real_roots_l1247_124799


namespace NUMINAMATH_CALUDE_square_side_length_l1247_124729

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * side * 2 = diagonal * diagonal ∧ side = 2 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l1247_124729


namespace NUMINAMATH_CALUDE_allan_bought_three_balloons_l1247_124778

/-- The number of balloons Allan bought at the park -/
def balloons_bought_by_allan (allan_initial : ℕ) (jake_total : ℕ) (jake_difference : ℕ) : ℕ :=
  (jake_total - jake_difference) - allan_initial

/-- Theorem stating that Allan bought 3 balloons at the park -/
theorem allan_bought_three_balloons :
  balloons_bought_by_allan 2 6 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_allan_bought_three_balloons_l1247_124778


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1247_124776

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose n 3 = 7 * Nat.choose n 1) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1247_124776


namespace NUMINAMATH_CALUDE_fourth_angle_measure_l1247_124791

-- Define a quadrilateral type
structure Quadrilateral :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)
  (angle4 : ℝ)

-- Define the property that the sum of angles in a quadrilateral is 360°
def sum_of_angles (q : Quadrilateral) : Prop :=
  q.angle1 + q.angle2 + q.angle3 + q.angle4 = 360

-- Theorem statement
theorem fourth_angle_measure (q : Quadrilateral) 
  (h1 : q.angle1 = 120)
  (h2 : q.angle2 = 85)
  (h3 : q.angle3 = 90)
  (h4 : sum_of_angles q) :
  q.angle4 = 65 := by
  sorry

end NUMINAMATH_CALUDE_fourth_angle_measure_l1247_124791


namespace NUMINAMATH_CALUDE_minor_axis_length_l1247_124722

def ellipse_equation (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

theorem minor_axis_length :
  ∃ (minor_axis_length : ℝ),
    minor_axis_length = 4 ∧
    ∀ (x y : ℝ), ellipse_equation x y →
      ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
        x^2 / a^2 + y^2 / b^2 = 1 ∧
        minor_axis_length = 2 * b :=
by sorry

end NUMINAMATH_CALUDE_minor_axis_length_l1247_124722


namespace NUMINAMATH_CALUDE_samantha_sleep_hours_l1247_124712

/-- Represents the number of hours Samantha sleeps per night -/
def samantha_sleep : ℝ := 8

/-- Represents the number of hours Samantha's baby sister sleeps per night -/
def baby_sister_sleep : ℝ := 2.5 * samantha_sleep

/-- Represents the number of hours Samantha's father sleeps per night -/
def father_sleep : ℝ := 0.5 * baby_sister_sleep

theorem samantha_sleep_hours :
  samantha_sleep = 8 ∧
  baby_sister_sleep = 2.5 * samantha_sleep ∧
  father_sleep = 0.5 * baby_sister_sleep ∧
  7 * father_sleep = 70 := by
  sorry

#check samantha_sleep_hours

end NUMINAMATH_CALUDE_samantha_sleep_hours_l1247_124712


namespace NUMINAMATH_CALUDE_largest_interesting_number_l1247_124795

/-- A real number is interesting if removing one digit from its decimal representation results in 2x -/
def IsInteresting (x : ℝ) : Prop :=
  ∃ (y : ℕ) (z : ℝ), 0 < x ∧ x < 1 ∧ x = y / 10 + z ∧ 2 * x = z

/-- The largest interesting number is 0.375 -/
theorem largest_interesting_number :
  IsInteresting (3 / 8) ∧ ∀ x : ℝ, IsInteresting x → x ≤ 3 / 8 :=
sorry

end NUMINAMATH_CALUDE_largest_interesting_number_l1247_124795


namespace NUMINAMATH_CALUDE_odd_square_mod_eight_l1247_124792

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_square_mod_eight_l1247_124792


namespace NUMINAMATH_CALUDE_function_inequality_l1247_124751

open Real

/-- Given a function f: ℝ → ℝ with derivative f', 
    if x · f'(x) + f(x) < 0 for all x, then 2 · f(2) > 3 · f(3) -/
theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
    (h : ∀ x, HasDerivAt f (f' x) x)
    (h' : ∀ x, x * f' x + f x < 0) :
    2 * f 2 > 3 * f 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1247_124751


namespace NUMINAMATH_CALUDE_sqrt_product_is_eight_l1247_124728

theorem sqrt_product_is_eight :
  Real.sqrt (9 - Real.sqrt 77) * Real.sqrt 2 * (Real.sqrt 11 - Real.sqrt 7) * (9 + Real.sqrt 77) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_is_eight_l1247_124728


namespace NUMINAMATH_CALUDE_ant_walk_theorem_l1247_124781

/-- The length of a cube's side in centimeters -/
def cube_side_length : ℝ := 18

/-- The number of cube edges the ant walks along -/
def number_of_edges : ℕ := 5

/-- The distance the ant walks on the cube's surface -/
def ant_walk_distance : ℝ := cube_side_length * number_of_edges

theorem ant_walk_theorem : ant_walk_distance = 90 := by
  sorry

end NUMINAMATH_CALUDE_ant_walk_theorem_l1247_124781


namespace NUMINAMATH_CALUDE_two_numbers_sum_product_l1247_124714

theorem two_numbers_sum_product (S P : ℝ) (h : S^2 ≥ 4*P) :
  ∃! (x y : ℝ × ℝ), (x.1 + x.2 = S ∧ x.1 * x.2 = P) ∧ (y.1 + y.2 = S ∧ y.1 * y.2 = P) ∧ x ≠ y :=
by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_product_l1247_124714


namespace NUMINAMATH_CALUDE_f_iteration_result_l1247_124741

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^2 + 1 else -z^2 - 1

-- State the theorem
theorem f_iteration_result :
  f (f (f (f (2 + I)))) = 1042434 - 131072 * I :=
by sorry

end NUMINAMATH_CALUDE_f_iteration_result_l1247_124741


namespace NUMINAMATH_CALUDE_simplify_fraction_cube_l1247_124745

theorem simplify_fraction_cube (a b : ℝ) (ha : a ≠ 0) :
  (3 * b / (2 * a^2))^3 = 27 * b^3 / (8 * a^6) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_cube_l1247_124745


namespace NUMINAMATH_CALUDE_abs_neg_three_equals_three_l1247_124726

theorem abs_neg_three_equals_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_equals_three_l1247_124726


namespace NUMINAMATH_CALUDE_gift_spending_calculation_l1247_124787

/-- Given a total amount spent and an amount spent on giftwrapping and other expenses,
    calculate the amount spent on gifts. -/
def amount_spent_on_gifts (total_amount : ℚ) (giftwrapping_amount : ℚ) : ℚ :=
  total_amount - giftwrapping_amount

/-- Prove that the amount spent on gifts is $561.00, given the total amount
    spent is $700.00 and the amount spent on giftwrapping is $139.00. -/
theorem gift_spending_calculation :
  amount_spent_on_gifts 700 139 = 561 := by
  sorry

end NUMINAMATH_CALUDE_gift_spending_calculation_l1247_124787


namespace NUMINAMATH_CALUDE_x_over_u_value_l1247_124709

theorem x_over_u_value (u v w x : ℝ) 
  (h1 : u / v = 5)
  (h2 : w / v = 3)
  (h3 : w / x = 2 / 3) :
  x / u = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_x_over_u_value_l1247_124709
