import Mathlib

namespace geometric_sequence_inequality_l3958_395844

/-- Given a geometric sequence with positive terms and common ratio not equal to 1,
    prove that the arithmetic mean of the 3rd and 9th terms is greater than
    the geometric mean of the 5th and 7th terms. -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  q ≠ 1 →           -- Common ratio is not 1
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence property
  (a 3 + a 9) / 2 > Real.sqrt (a 5 * a 7) := by
  sorry

end geometric_sequence_inequality_l3958_395844


namespace problem_solution_l3958_395836

theorem problem_solution (m n : ℝ) : 
  (∃ k : ℝ, k^2 = m + 3 ∧ (k = 1 ∨ k = -1)) →
  (2*n - 12)^(1/3) = 4 →
  m = -2 ∧ n = 38 ∧ Real.sqrt (m + n) = 6 :=
by sorry

end problem_solution_l3958_395836


namespace sufficient_not_necessary_condition_l3958_395850

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (0 < a ∧ a < b → 1 / a > 1 / b) ∧
  ¬(1 / a > 1 / b → 0 < a ∧ a < b) :=
sorry

end sufficient_not_necessary_condition_l3958_395850


namespace intersection_of_specific_sets_l3958_395823

theorem intersection_of_specific_sets :
  let A : Set ℤ := {1, 2, -3}
  let B : Set ℤ := {1, -4, 5}
  A ∩ B = {1} := by
sorry

end intersection_of_specific_sets_l3958_395823


namespace compute_expression_l3958_395879

theorem compute_expression : 2⁻¹ + |-5| - Real.sin (30 * π / 180) + (π - 1)^0 = 6 := by
  sorry

end compute_expression_l3958_395879


namespace arden_cricket_club_members_l3958_395810

/-- The cost of a pair of gloves in dollars -/
def glove_cost : ℕ := 6

/-- The additional cost of a cap compared to a pair of gloves in dollars -/
def cap_additional_cost : ℕ := 8

/-- The total expenditure of the club in dollars -/
def total_expenditure : ℕ := 4140

/-- The number of gloves and caps each member needs -/
def items_per_member : ℕ := 2

theorem arden_cricket_club_members :
  ∃ (n : ℕ), n * (items_per_member * (glove_cost + (glove_cost + cap_additional_cost))) = total_expenditure ∧
  n = 103 := by
  sorry

end arden_cricket_club_members_l3958_395810


namespace hanyoung_weight_l3958_395807

theorem hanyoung_weight (hanyoung joohyung : ℝ) 
  (h1 : hanyoung = joohyung - 4)
  (h2 : hanyoung + joohyung = 88) : 
  hanyoung = 42 := by
sorry

end hanyoung_weight_l3958_395807


namespace losing_candidate_percentage_approx_33_percent_l3958_395833

/-- Calculates the percentage of votes received by a losing candidate -/
def losingCandidatePercentage (totalVotes : ℕ) (lossMargin : ℕ) : ℚ :=
  let candidateVotes := (totalVotes - lossMargin) / 2
  (candidateVotes : ℚ) / totalVotes * 100

/-- Theorem stating that given the conditions, the losing candidate's vote percentage is approximately 33% -/
theorem losing_candidate_percentage_approx_33_percent 
  (totalVotes : ℕ) (lossMargin : ℕ) 
  (h1 : totalVotes = 2450) 
  (h2 : lossMargin = 833) : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |losingCandidatePercentage totalVotes lossMargin - 33| < ε :=
sorry

end losing_candidate_percentage_approx_33_percent_l3958_395833


namespace double_plus_five_difference_l3958_395813

theorem double_plus_five_difference (x : ℝ) (h : x = 4) : 2 * x + 5 - x / 2 = 11 := by
  sorry

end double_plus_five_difference_l3958_395813


namespace expand_polynomial_l3958_395842

theorem expand_polynomial (x : ℝ) : (x + 3) * (4 * x^2 - 5 * x + 6) = 4 * x^3 + 7 * x^2 - 9 * x + 18 := by
  sorry

end expand_polynomial_l3958_395842


namespace fraction_meaningful_l3958_395859

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x + 5)) ↔ x ≠ -5 := by sorry

end fraction_meaningful_l3958_395859


namespace combined_list_size_l3958_395895

def combined_friends_list (james_friends john_friends shared_friends : ℕ) : ℕ :=
  james_friends + john_friends - shared_friends

theorem combined_list_size :
  let james_friends : ℕ := 75
  let john_friends : ℕ := 3 * james_friends
  let shared_friends : ℕ := 25
  combined_friends_list james_friends john_friends shared_friends = 275 := by
  sorry

end combined_list_size_l3958_395895


namespace damage_proportion_l3958_395882

/-- The proportion of a 3x2 rectangle that can be reached by the midpoint of a 2-unit line segment
    rotating freely within the rectangle -/
theorem damage_proportion (rectangle_length : Real) (rectangle_width : Real) (log_length : Real) :
  rectangle_length = 3 ∧ rectangle_width = 2 ∧ log_length = 2 →
  (rectangle_length * rectangle_width - 4 * (Real.pi / 4 * (log_length / 2)^2)) / (rectangle_length * rectangle_width) = 1 - Real.pi / 6 := by
  sorry

end damage_proportion_l3958_395882


namespace cakes_left_is_two_l3958_395839

def cakes_baked_yesterday : ℕ := 3
def cakes_baked_lunch : ℕ := 5
def cakes_sold_dinner : ℕ := 6

def cakes_left : ℕ := cakes_baked_yesterday + cakes_baked_lunch - cakes_sold_dinner

theorem cakes_left_is_two : cakes_left = 2 := by
  sorry

end cakes_left_is_two_l3958_395839


namespace greatest_product_of_digits_divisible_by_35_l3958_395811

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_single_digit : tens < 10
  units_single_digit : units < 10

/-- Check if a number is divisible by another number -/
def isDivisibleBy (n m : Nat) : Prop := ∃ k, n = m * k

theorem greatest_product_of_digits_divisible_by_35 :
  ∀ n : TwoDigitNumber,
    isDivisibleBy (10 * n.tens + n.units) 35 →
    ∀ m : TwoDigitNumber,
      isDivisibleBy (10 * m.tens + m.units) 35 →
      n.units * n.tens ≤ 40 ∧
      (m.units * m.tens = 40 → n.units * n.tens = 40) :=
sorry

end greatest_product_of_digits_divisible_by_35_l3958_395811


namespace max_pairs_correct_max_pairs_achievable_l3958_395876

/-- The maximum number of pairs that can be chosen from the set {1, 2, ..., 3009}
    such that no two pairs have a common element and all sums of pairs are distinct
    and less than or equal to 3009 -/
def max_pairs : ℕ := 1003

theorem max_pairs_correct : 
  ∀ (k : ℕ) (pairs : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ Finset.range 3009 ∧ p.2 ∈ Finset.range 3009) →
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 3009) →
    pairs.card ≤ max_pairs :=
by sorry

theorem max_pairs_achievable :
  ∃ (pairs : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ∈ Finset.range 3009 ∧ p.2 ∈ Finset.range 3009) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 3009) ∧
    pairs.card = max_pairs :=
by sorry

end max_pairs_correct_max_pairs_achievable_l3958_395876


namespace bricklayer_problem_l3958_395837

theorem bricklayer_problem (x : ℝ) 
  (h1 : (x / 12 + x / 15 - 15) * 6 = x) : x = 900 := by
  sorry

#check bricklayer_problem

end bricklayer_problem_l3958_395837


namespace prob_draw_3_equals_expected_l3958_395870

-- Define the defect rate
def defect_rate : ℝ := 0.03

-- Define the probability of drawing exactly 3 products
def prob_draw_3 (p : ℝ) : ℝ := p^2 * (1 - p) + p^3

-- Theorem statement
theorem prob_draw_3_equals_expected : 
  prob_draw_3 defect_rate = defect_rate^2 * (1 - defect_rate) + defect_rate^3 :=
by sorry

end prob_draw_3_equals_expected_l3958_395870


namespace billboard_dimensions_l3958_395843

/-- Given a rectangular photograph and a billboard, prove the billboard's dimensions -/
theorem billboard_dimensions 
  (photo_width : ℝ) 
  (photo_length : ℝ) 
  (billboard_area : ℝ) 
  (h1 : photo_width = 30) 
  (h2 : photo_length = 40) 
  (h3 : billboard_area = 48) : 
  ∃ (billboard_width billboard_length : ℝ), 
    billboard_width = 6 ∧ 
    billboard_length = 8 ∧ 
    billboard_width * billboard_length = billboard_area :=
by sorry

end billboard_dimensions_l3958_395843


namespace problem_1_problem_2_problem_3_problem_4_l3958_395868

-- Problem 1
theorem problem_1 : -3 + 8 - 7 - 15 = -17 := by sorry

-- Problem 2
theorem problem_2 : 23 - 6 * (-3) + 2 * (-4) = 33 := by sorry

-- Problem 3
theorem problem_3 : -8 / (4/5) * (-2/3) = 20/3 := by sorry

-- Problem 4
theorem problem_4 : -(2^2) - 9 * ((-1/3)^2) + |(-4)| = -1 := by sorry

end problem_1_problem_2_problem_3_problem_4_l3958_395868


namespace heather_emily_weight_difference_l3958_395809

/-- Given the weights of Heather and Emily, prove that Heather is 78 pounds heavier than Emily. -/
theorem heather_emily_weight_difference :
  let heather_weight : ℕ := 87
  let emily_weight : ℕ := 9
  heather_weight - emily_weight = 78 := by sorry

end heather_emily_weight_difference_l3958_395809


namespace jelly_bean_probability_l3958_395887

/-- The probability of selecting either a blue or purple jelly bean from a bag -/
theorem jelly_bean_probability :
  let red : ℕ := 8
  let green : ℕ := 9
  let yellow : ℕ := 10
  let blue : ℕ := 12
  let purple : ℕ := 5
  let total : ℕ := red + green + yellow + blue + purple
  let blue_or_purple : ℕ := blue + purple
  (blue_or_purple : ℚ) / total = 17 / 44 :=
by sorry

end jelly_bean_probability_l3958_395887


namespace min_value_theorem_l3958_395849

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  arithmetic_sequence a →
  (∀ k : ℕ, a k > 0) →
  a 7 = a 6 + 2 * a 5 →
  Real.sqrt (a m * a n) = 2 * Real.sqrt 2 * a 1 →
  (2 : ℝ) / m + 8 / n ≥ 18 / 5 :=
by sorry

end min_value_theorem_l3958_395849


namespace square_cut_parts_l3958_395856

/-- Represents a square grid paper -/
structure GridPaper :=
  (size : ℕ)

/-- Represents a folded square -/
structure FoldedSquare :=
  (original : GridPaper)
  (folded_size : ℕ)

/-- Represents a cut on the folded square -/
inductive Cut
  | Midpoint : Cut

/-- The number of parts resulting from unfolding after the cut -/
def num_parts_after_cut (fs : FoldedSquare) (c : Cut) : ℕ :=
  fs.original.size + 1

theorem square_cut_parts :
  ∀ (gp : GridPaper) (fs : FoldedSquare) (c : Cut),
    gp.size = 8 →
    fs.original = gp →
    fs.folded_size = 1 →
    c = Cut.Midpoint →
    num_parts_after_cut fs c = 9 :=
sorry

end square_cut_parts_l3958_395856


namespace iphone_price_calculation_l3958_395846

def calculate_final_price (initial_price : ℝ) 
                          (discount1 : ℝ) (tax1 : ℝ) 
                          (discount2 : ℝ) (tax2 : ℝ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_tax1 := price_after_discount1 * (1 + tax1)
  let price_after_discount2 := price_after_tax1 * (1 - discount2)
  let final_price := price_after_discount2 * (1 + tax2)
  final_price

theorem iphone_price_calculation :
  let initial_price : ℝ := 1000
  let discount1 : ℝ := 0.1
  let tax1 : ℝ := 0.08
  let discount2 : ℝ := 0.2
  let tax2 : ℝ := 0.06
  let final_price := calculate_final_price initial_price discount1 tax1 discount2 tax2
  ∃ ε > 0, |final_price - 824.26| < ε :=
sorry

end iphone_price_calculation_l3958_395846


namespace choose_two_from_four_eq_six_l3958_395852

def choose_two_from_four : ℕ := sorry

theorem choose_two_from_four_eq_six : choose_two_from_four = 6 := by sorry

end choose_two_from_four_eq_six_l3958_395852


namespace janet_walk_time_l3958_395867

-- Define Janet's walking pattern
def blocks_north : ℕ := 3
def blocks_west : ℕ := 7 * blocks_north
def blocks_south : ℕ := 8
def blocks_east : ℕ := 2 * blocks_south

-- Define Janet's walking speed
def blocks_per_minute : ℕ := 2

-- Calculate net distance from home
def net_south : ℤ := blocks_south - blocks_north
def net_west : ℤ := blocks_west - blocks_east

-- Total distance to walk home
def total_distance : ℕ := (net_south.natAbs + net_west.natAbs : ℕ)

-- Time to walk home
def time_to_home : ℚ := total_distance / blocks_per_minute

-- Theorem to prove
theorem janet_walk_time : time_to_home = 5 := by
  sorry

end janet_walk_time_l3958_395867


namespace highest_score_in_test_l3958_395805

/-- Given a math test with scores, prove the highest score -/
theorem highest_score_in_test (mark_score least_score highest_score : ℕ) : 
  mark_score = 2 * least_score →
  mark_score = 46 →
  highest_score - least_score = 75 →
  highest_score = 98 :=
by sorry

end highest_score_in_test_l3958_395805


namespace graph_single_point_implies_d_eq_21_l3958_395855

/-- The equation of the graph -/
def graph_equation (x y d : ℝ) : Prop :=
  3 * x^2 + y^2 + 12 * x - 6 * y + d = 0

/-- The condition that the graph consists of a single point -/
def is_single_point (d : ℝ) : Prop :=
  ∃! (x y : ℝ), graph_equation x y d

/-- Theorem: If the graph consists of a single point, then d = 21 -/
theorem graph_single_point_implies_d_eq_21 :
  ∀ d : ℝ, is_single_point d → d = 21 := by
  sorry

end graph_single_point_implies_d_eq_21_l3958_395855


namespace smallest_n_congruence_l3958_395841

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 23 * n ≡ 789 [ZMOD 11]) → n ≥ 9 :=
by sorry

end smallest_n_congruence_l3958_395841


namespace largest_two_digit_prime_factor_l3958_395847

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

/-- Checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- Checks if a number is a factor of another number -/
def isFactor (a b : ℕ) : Prop := b % a = 0

theorem largest_two_digit_prime_factor :
  ∃ (p : ℕ), isPrime p ∧ 
             isTwoDigit p ∧ 
             isFactor p (binomial 300 150) ∧
             (∀ (q : ℕ), isPrime q → isTwoDigit q → isFactor q (binomial 300 150) → q ≤ p) ∧
             p = 97 := by sorry

end largest_two_digit_prime_factor_l3958_395847


namespace ticket_distribution_l3958_395866

/-- The number of ways to distribute 5 consecutive tickets to 5 people. -/
def distribute_tickets : ℕ := 5 * 4 * 3 * 2 * 1

/-- The number of ways for A and B to receive consecutive tickets. -/
def consecutive_for_ab : ℕ := 4 * 2

/-- The number of ways to distribute the remaining tickets to 3 people. -/
def distribute_remaining : ℕ := 3 * 2 * 1

/-- 
Theorem: The number of ways to distribute 5 consecutive movie tickets to 5 people, 
including A and B, such that A and B receive consecutive tickets, is equal to 48.
-/
theorem ticket_distribution : 
  consecutive_for_ab * distribute_remaining = 48 := by
  sorry

end ticket_distribution_l3958_395866


namespace host_horse_speed_calculation_l3958_395824

/-- The daily travel distance of the guest's horse in li -/
def guest_horse_speed : ℚ := 300

/-- The fraction of the day that passes before the host realizes the guest left without clothes -/
def realization_time : ℚ := 1/3

/-- The fraction of the day that has passed when the host returns home -/
def return_time : ℚ := 3/4

/-- The daily travel distance of the host's horse in li -/
def host_horse_speed : ℚ := 780

theorem host_horse_speed_calculation :
  let catch_up_time : ℚ := return_time - realization_time
  let guest_travel_time : ℚ := realization_time + catch_up_time
  2 * guest_horse_speed * guest_travel_time = host_horse_speed * catch_up_time :=
by sorry

end host_horse_speed_calculation_l3958_395824


namespace batsman_average_proof_l3958_395872

/-- Calculates the average runs for a batsman given two sets of matches with different averages -/
def calculateAverageRuns (matches1 : ℕ) (average1 : ℚ) (matches2 : ℕ) (average2 : ℚ) : ℚ :=
  ((matches1 : ℚ) * average1 + (matches2 : ℚ) * average2) / ((matches1 + matches2) : ℚ)

/-- Theorem: Given a batsman's performance in two sets of matches, prove the overall average -/
theorem batsman_average_proof (matches1 matches2 : ℕ) (average1 average2 : ℚ) :
  matches1 = 20 ∧ matches2 = 10 ∧ average1 = 30 ∧ average2 = 15 →
  calculateAverageRuns matches1 average1 matches2 average2 = 25 := by
  sorry

#eval calculateAverageRuns 20 30 10 15

end batsman_average_proof_l3958_395872


namespace allocation_schemes_l3958_395861

def doctors : ℕ := 2
def nurses : ℕ := 4
def hospitals : ℕ := 2
def doctors_per_hospital : ℕ := 1
def nurses_per_hospital : ℕ := 2

theorem allocation_schemes :
  (Nat.choose doctors hospitals) * (Nat.choose nurses (nurses_per_hospital * hospitals)) = 12 :=
sorry

end allocation_schemes_l3958_395861


namespace complex_real_condition_l3958_395888

theorem complex_real_condition (m : ℝ) : 
  (Complex.I * Complex.I = -1) →
  ((m^2 + Complex.I) * (1 - m * Complex.I)).im = 0 →
  m = 1 := by sorry

end complex_real_condition_l3958_395888


namespace roots_relation_l3958_395877

/-- The polynomial h(x) -/
def h (x : ℝ) : ℝ := x^3 - 2*x^2 + 4*x - 1

/-- The polynomial j(x) -/
def j (x p q r : ℝ) : ℝ := x^3 + p*x^2 + q*x + r

/-- Theorem stating the relationship between h(x) and j(x) and the values of p, q, and r -/
theorem roots_relation (p q r : ℝ) : 
  (∀ s, h s = 0 → ∃ t, j t p q r = 0 ∧ s = t + 2) → 
  p = 4 ∧ q = 8 ∧ r = 7 := by
  sorry

end roots_relation_l3958_395877


namespace juice_theorem_l3958_395818

def juice_problem (tom_initial jerry_initial : ℚ) 
  (drink_fraction transfer_fraction : ℚ) (final_transfer : ℚ) : Prop :=
  let tom_after_drinking := tom_initial * (1 - drink_fraction)
  let jerry_after_drinking := jerry_initial * (1 - drink_fraction)
  let jerry_transfer := jerry_after_drinking * transfer_fraction
  let tom_before_final := tom_after_drinking + jerry_transfer
  let jerry_before_final := jerry_after_drinking - jerry_transfer
  let tom_final := tom_before_final - final_transfer
  let jerry_final := jerry_before_final + final_transfer
  (jerry_initial = 2 * tom_initial) ∧
  (tom_final = jerry_final + 4) ∧
  (tom_initial + jerry_initial - (tom_final + jerry_final) = 80)

theorem juice_theorem : 
  juice_problem 40 80 (2/3) (1/4) 5 := by sorry

end juice_theorem_l3958_395818


namespace line_relationships_l3958_395828

/-- Two lines in the 2D plane -/
structure TwoLines where
  l1 : ℝ → ℝ → ℝ → ℝ  -- (2a+1)x+(a+2)y+3=0
  l2 : ℝ → ℝ → ℝ → ℝ  -- (a-1)x-2y+2=0

/-- Definition of parallel lines -/
def parallel (lines : TwoLines) (a : ℝ) : Prop :=
  ∀ x y, lines.l1 a x y = 0 ↔ lines.l2 a x y = 0

/-- Definition of perpendicular lines -/
def perpendicular (lines : TwoLines) (a : ℝ) : Prop :=
  ∀ x1 y1 x2 y2, lines.l1 a x1 y1 = 0 ∧ lines.l2 a x2 y2 = 0 →
    (x2 - x1) * ((2 * a + 1) * (x2 - x1) + (a + 2) * (y2 - y1)) +
    (y2 - y1) * ((a - 1) * (x2 - x1) - 2 * (y2 - y1)) = 0

/-- The main theorem -/
theorem line_relationships (lines : TwoLines) :
  (∀ a, parallel lines a ↔ a = 0) ∧
  (∀ a, perpendicular lines a ↔ a = -1 ∨ a = 5/2) := by
  sorry

#check line_relationships

end line_relationships_l3958_395828


namespace annulus_area_l3958_395801

/-- The area of an annulus formed by two concentric circles -/
theorem annulus_area (R r t : ℝ) (h1 : R > r) (h2 : R^2 = r^2 + t^2) : 
  π * R^2 - π * r^2 = π * t^2 := by sorry

end annulus_area_l3958_395801


namespace apple_cost_l3958_395815

/-- The cost of an apple and an orange given two price combinations -/
theorem apple_cost (apple orange : ℝ) 
  (h1 : 6 * apple + 3 * orange = 1.77)
  (h2 : 2 * apple + 5 * orange = 1.27) :
  apple = 0.21 := by
sorry

end apple_cost_l3958_395815


namespace mindmaster_secret_codes_l3958_395812

/-- The number of colors available for the pegs. -/
def num_colors : ℕ := 7

/-- The number of slots in each code. -/
def code_length : ℕ := 5

/-- The total number of possible codes without restrictions. -/
def total_codes : ℕ := num_colors ^ code_length

/-- The number of colors excluding red. -/
def non_red_colors : ℕ := num_colors - 1

/-- The number of codes without any red pegs. -/
def codes_without_red : ℕ := non_red_colors ^ code_length

/-- The number of valid secret codes in Mindmaster. -/
def valid_secret_codes : ℕ := total_codes - codes_without_red

theorem mindmaster_secret_codes : valid_secret_codes = 9031 := by
  sorry

end mindmaster_secret_codes_l3958_395812


namespace largest_n_unique_k_l3958_395816

theorem largest_n_unique_k : ∃ (n : ℕ), n > 0 ∧ n ≤ 27 ∧
  (∃! (k : ℤ), (9:ℚ)/17 < (n:ℚ)/(n+k) ∧ (n:ℚ)/(n+k) < 8/15) ∧
  (∀ (m : ℕ), m > 27 → ¬(∃! (k : ℤ), (9:ℚ)/17 < (m:ℚ)/(m+k) ∧ (m:ℚ)/(m+k) < 8/15)) :=
by sorry

end largest_n_unique_k_l3958_395816


namespace max_value_interval_l3958_395806

open Real

noncomputable def f (a x : ℝ) : ℝ := 3 * log x - x^2 + (a - 1/2) * x

theorem max_value_interval (a : ℝ) :
  (∃ x ∈ Set.Ioo 1 3, ∀ y ∈ Set.Ioo 1 3, f a x ≥ f a y) ↔ a ∈ Set.Ioo (-1/2) (11/2) := by
  sorry

end max_value_interval_l3958_395806


namespace solution_satisfies_conditions_l3958_395831

-- Define the function y(x)
def y (x : ℝ) : ℝ := (x + 1)^2

-- State the theorem
theorem solution_satisfies_conditions :
  (∀ x, (deriv^[2] y) x = 2) ∧ 
  y 0 = 1 ∧ 
  (deriv y) 0 = 2 :=
by
  sorry


end solution_satisfies_conditions_l3958_395831


namespace skittles_pencils_difference_l3958_395854

def number_of_children : ℕ := 17
def pencils_per_child : ℕ := 3
def skittles_per_child : ℕ := 18

theorem skittles_pencils_difference :
  (number_of_children * skittles_per_child) - (number_of_children * pencils_per_child) = 255 := by
  sorry

end skittles_pencils_difference_l3958_395854


namespace article_sale_price_l3958_395889

/-- Given an article with unknown cost price, prove that the selling price
    incurring a loss equal to the profit at $852 is $448, given that the
    selling price for a 50% profit is $975. -/
theorem article_sale_price (cost : ℝ) (loss_price : ℝ) : 
  (852 - cost = cost - loss_price) →  -- Profit at $852 equals loss at loss_price
  (cost + 0.5 * cost = 975) →         -- 50% profit price is $975
  loss_price = 448 := by
  sorry

end article_sale_price_l3958_395889


namespace water_one_fifth_after_three_pourings_l3958_395874

def water_remaining (n : ℕ) : ℚ :=
  1 / (2 * n - 1)

theorem water_one_fifth_after_three_pourings :
  water_remaining 3 = 1 / 5 := by
  sorry

#check water_one_fifth_after_three_pourings

end water_one_fifth_after_three_pourings_l3958_395874


namespace floor_equation_solution_l3958_395808

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3*x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ 4/3 ≤ x ∧ x < 5/3 :=
sorry

end floor_equation_solution_l3958_395808


namespace quadratic_root_reciprocal_l3958_395838

/-- If m is a root of ax² + bx + 1 = 0, then 1/m is a root of x² + bx + a = 0 -/
theorem quadratic_root_reciprocal (a b m : ℝ) (hm : m ≠ 0) 
  (h : a * m^2 + b * m + 1 = 0) : 
  (1/m)^2 + b * (1/m) + a = 0 := by
  sorry

end quadratic_root_reciprocal_l3958_395838


namespace sport_corn_syrup_amount_l3958_395864

/-- Represents the ratios in a flavored drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation ratio -/
def sport_ratio (r : DrinkRatio) : DrinkRatio :=
  { flavoring := r.flavoring,
    corn_syrup := r.corn_syrup / 3,
    water := r.water * 2 }

/-- Theorem stating the amount of corn syrup in the sport formulation -/
theorem sport_corn_syrup_amount (water_amount : ℚ) :
  let sr := sport_ratio standard_ratio
  let flavoring := water_amount / sr.water
  flavoring * sr.corn_syrup = 7 :=
sorry

end sport_corn_syrup_amount_l3958_395864


namespace can_calculate_average_if_complete_info_cannot_calculate_camerons_average_l3958_395845

/-- Represents a tour guide's daily work --/
structure TourGuideDay where
  numTours : Nat
  totalQuestions : Nat
  groupSizes : List Nat

/-- Calculates the average number of questions per tourist --/
def averageQuestionsPerTourist (day : TourGuideDay) : Option ℚ :=
  if day.groupSizes.length = day.numTours ∧ day.groupSizes.sum ≠ 0 then
    some ((day.totalQuestions : ℚ) / (day.groupSizes.sum : ℚ))
  else
    none

/-- Theorem: If we have complete information, we can calculate the average questions per tourist --/
theorem can_calculate_average_if_complete_info (day : TourGuideDay) :
    day.groupSizes.length = day.numTours ∧ day.groupSizes.sum ≠ 0 →
    ∃ avg : ℚ, averageQuestionsPerTourist day = some avg :=
  sorry

/-- Cameron's specific day --/
def cameronsDay : TourGuideDay :=
  { numTours := 4
  , totalQuestions := 68
  , groupSizes := [] }  -- Empty list because we don't know the group sizes

/-- Theorem: We cannot calculate the average for Cameron's day due to missing information --/
theorem cannot_calculate_camerons_average :
    averageQuestionsPerTourist cameronsDay = none :=
  sorry

end can_calculate_average_if_complete_info_cannot_calculate_camerons_average_l3958_395845


namespace system_solution_l3958_395881

theorem system_solution (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : ℝ),
    (z + a*y + a^2*x + a^3 = 0) ∧
    (z + b*y + b^2*x + b^3 = 0) ∧
    (z + c*y + c^2*x + c^3 = 0) ∧
    (x = -(a+b+c)) ∧
    (y = a*b + a*c + b*c) ∧
    (z = -a*b*c) := by
  sorry

end system_solution_l3958_395881


namespace candy_distribution_proof_l3958_395826

def distribute_candies (total_candies : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

theorem candy_distribution_proof :
  distribute_candies 10 5 = 7 :=
by sorry

end candy_distribution_proof_l3958_395826


namespace brown_paint_amount_l3958_395857

def total_paint : ℕ := 69
def white_paint : ℕ := 20
def green_paint : ℕ := 15

theorem brown_paint_amount :
  total_paint - (white_paint + green_paint) = 34 := by
  sorry

end brown_paint_amount_l3958_395857


namespace prudence_sleep_hours_l3958_395862

/-- The number of hours Prudence sleeps per night from Sunday to Thursday -/
def sleepHoursSundayToThursday : ℝ := 6

/-- The number of hours Prudence sleeps on Friday and Saturday nights -/
def sleepHoursFridaySaturday : ℝ := 9

/-- The number of hours Prudence naps on Saturday and Sunday -/
def napHours : ℝ := 1

/-- The total number of hours Prudence sleeps in 4 weeks -/
def totalSleepHours : ℝ := 200

/-- The number of weeks -/
def numWeeks : ℝ := 4

/-- The number of nights from Sunday to Thursday -/
def nightsSundayToThursday : ℝ := 5

/-- The number of nights for Friday and Saturday -/
def nightsFridaySaturday : ℝ := 2

/-- The number of nap days (Saturday and Sunday) -/
def napDays : ℝ := 2

theorem prudence_sleep_hours :
  sleepHoursSundayToThursday * nightsSundayToThursday +
  sleepHoursFridaySaturday * nightsFridaySaturday +
  napHours * napDays * numWeeks = totalSleepHours :=
by sorry

end prudence_sleep_hours_l3958_395862


namespace absolute_value_equals_sqrt_of_square_l3958_395890

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end absolute_value_equals_sqrt_of_square_l3958_395890


namespace least_multiple_of_15_greater_than_520_l3958_395840

theorem least_multiple_of_15_greater_than_520 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n > 520 → n ≥ 525 := by
  sorry

end least_multiple_of_15_greater_than_520_l3958_395840


namespace refrigerator_deposit_l3958_395884

/-- Proves the deposit amount for a refrigerator purchase with installments -/
theorem refrigerator_deposit (cash_price : ℕ) (num_installments : ℕ) (installment_amount : ℕ) (savings : ℕ) : 
  cash_price = 8000 →
  num_installments = 30 →
  installment_amount = 300 →
  savings = 4000 →
  cash_price + savings = num_installments * installment_amount + (cash_price + savings - num_installments * installment_amount) :=
by sorry

end refrigerator_deposit_l3958_395884


namespace boat_speed_l3958_395878

theorem boat_speed (v_s : ℝ) (t_d t_u : ℝ) (h1 : v_s = 8) (h2 : t_u = 2 * t_d) : 
  ∃ v_b : ℝ, v_b > 0 ∧ (v_b - v_s) * t_u = (v_b + v_s) * t_d ∧ v_b = 24 :=
by sorry

end boat_speed_l3958_395878


namespace rectangle_area_l3958_395858

/-- The area of a rectangle with diagonal x and length three times its width -/
theorem rectangle_area (x : ℝ) : 
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w ^ 2 + l ^ 2 = x ^ 2 → w * l = (3 / 10) * x ^ 2 := by
  sorry

end rectangle_area_l3958_395858


namespace soup_can_price_l3958_395817

/-- Calculates the normal price of a can of soup given a "buy 1 get one free" offer -/
theorem soup_can_price (total_cans : ℕ) (total_paid : ℚ) : 
  total_cans > 0 → total_paid > 0 → (total_paid / (total_cans / 2 : ℚ) = 0.60) := by
  sorry

end soup_can_price_l3958_395817


namespace geometric_sequence_common_ratio_l3958_395860

/-- A geometric sequence with positive first term and increasing terms -/
def IncreasingGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 > 0 ∧ q > 1 ∧ ∀ n, a (n + 1) = q * a n

/-- The relation between consecutive terms in the sequence -/
def SequenceRelation (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_increasing : IncreasingGeometricSequence a q)
  (h_relation : SequenceRelation a) :
  q = 2 := by
  sorry

end geometric_sequence_common_ratio_l3958_395860


namespace alex_win_probability_l3958_395853

-- Define the game conditions
def standardDie : ℕ := 6
def evenNumbers : Set ℕ := {2, 4, 6}

-- Define the probability of Kelvin winning on first roll
def kelvinFirstWinProb : ℚ := 1 / 2

-- Define the probability of Alex winning on first roll, given Kelvin didn't win
def alexFirstWinProb : ℚ := 2 / 3

-- Define the probability of Kelvin winning on second roll, given Alex didn't win on first
def kelvinSecondWinProb : ℚ := 5 / 6

-- Define the probability of Alex winning on second roll, given Kelvin didn't win on second
def alexSecondWinProb : ℚ := 2 / 3

-- State the theorem
theorem alex_win_probability :
  let totalAlexWinProb := 
    kelvinFirstWinProb * alexFirstWinProb + 
    (1 - kelvinFirstWinProb) * (1 - alexFirstWinProb) * (1 - kelvinSecondWinProb) * alexSecondWinProb
  totalAlexWinProb = 22 / 27 := by
  sorry

end alex_win_probability_l3958_395853


namespace common_tangents_count_l3958_395829

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 16*y - 48 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 44 = 0

-- Define the function to count common tangents
def count_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents circle1 circle2 = 2 := by sorry

end common_tangents_count_l3958_395829


namespace largest_ball_radius_l3958_395869

/-- The radius of the torus circle -/
def torus_radius : ℝ := 2

/-- The x-coordinate of the torus circle center -/
def torus_center_x : ℝ := 4

/-- The z-coordinate of the torus circle center -/
def torus_center_z : ℝ := 1

/-- The theorem stating that the radius of the largest spherical ball that can sit on top of the center of the torus and touch the horizontal plane is 4 -/
theorem largest_ball_radius : 
  ∃ (r : ℝ), r = 4 ∧ 
  (torus_center_x ^ 2 + (r - torus_center_z) ^ 2 = (r + torus_radius) ^ 2) ∧
  r > 0 :=
sorry

end largest_ball_radius_l3958_395869


namespace max_correct_answers_l3958_395880

theorem max_correct_answers (total_questions : ℕ) (correct_score : ℤ) (incorrect_score : ℤ) (total_score : ℤ) :
  total_questions = 25 →
  correct_score = 5 →
  incorrect_score = -3 →
  total_score = 57 →
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct_score * correct + incorrect_score * incorrect = total_score ∧
    correct ≤ 12 ∧
    ∀ (c i u : ℕ),
      c + i + u = total_questions →
      correct_score * c + incorrect_score * i = total_score →
      c ≤ 12 :=
by sorry

end max_correct_answers_l3958_395880


namespace triangle_probability_theorem_l3958_395804

noncomputable def triangle_probability (XY : ℝ) (angle_XYZ : ℝ) : ℝ :=
  (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3

theorem triangle_probability_theorem (XY : ℝ) (angle_XYZ : ℝ) :
  XY = 12 →
  angle_XYZ = π / 6 →
  triangle_probability XY angle_XYZ = (Real.sqrt 3 - Real.sqrt 2) / Real.sqrt 3 :=
by sorry

end triangle_probability_theorem_l3958_395804


namespace find_y_value_l3958_395819

theorem find_y_value (x y z : ℤ) 
  (eq1 : x + y + z = 355)
  (eq2 : x - y = 200)
  (eq3 : x + z = 500) :
  y = -145 := by
  sorry

end find_y_value_l3958_395819


namespace intersection_A_B_complement_union_A_B_C_subset_B_implies_m_range_l3958_395898

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x ≤ 6}
def B : Set ℝ := {x | x^2 - 4*x < 0}
def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 4} := by sorry

theorem complement_union_A_B : 
  (Set.univ : Set ℝ) \ (A ∪ B) = {x : ℝ | x ≤ 0 ∨ x > 6} := by sorry

theorem C_subset_B_implies_m_range (m : ℝ) : 
  C m ⊆ B → m ≤ 5/2 := by sorry

end intersection_A_B_complement_union_A_B_C_subset_B_implies_m_range_l3958_395898


namespace equal_angle_measure_l3958_395875

-- Define the structure of our shape
structure RectangleTriangleConfig where
  -- Rectangle properties
  rect_length : ℝ
  rect_height : ℝ
  rect_height_lt_length : rect_height < rect_length

  -- Triangle properties
  triangle_base : ℝ
  triangle_leg : ℝ

  -- Shared side property
  shared_side_eq_rect_length : triangle_base = rect_length

  -- Isosceles triangle property
  isosceles_triangle : triangle_base = 2 * triangle_leg

-- Theorem statement
theorem equal_angle_measure (config : RectangleTriangleConfig) :
  let angle := Real.arccos (config.triangle_leg / config.triangle_base) * (180 / Real.pi)
  angle = 45 := by sorry

end equal_angle_measure_l3958_395875


namespace more_sad_players_left_l3958_395886

/-- Represents the state of a player in the game -/
inductive PlayerState
| Sad
| Cheerful

/-- Represents the game with its rules and initial state -/
structure Game where
  initialPlayers : Nat
  remainingPlayers : Nat
  sadPlayers : Nat
  cheerfulPlayers : Nat

/-- Definition of a valid game state -/
def validGameState (g : Game) : Prop :=
  g.initialPlayers = 36 ∧
  g.remainingPlayers + g.sadPlayers + g.cheerfulPlayers = g.initialPlayers ∧
  g.remainingPlayers ≥ 1

/-- The game ends when only one player remains -/
def gameEnded (g : Game) : Prop :=
  g.remainingPlayers = 1

/-- Theorem stating that more sad players have left the game than cheerful players when the game ends -/
theorem more_sad_players_left (g : Game) 
  (h1 : validGameState g) 
  (h2 : gameEnded g) : 
  g.sadPlayers > g.cheerfulPlayers :=
sorry

end more_sad_players_left_l3958_395886


namespace average_of_remaining_numbers_l3958_395892

theorem average_of_remaining_numbers
  (total : ℕ)
  (avg_all : ℚ)
  (avg_first_two : ℚ)
  (avg_next_two : ℚ)
  (h_total : total = 6)
  (h_avg_all : avg_all = 3.95)
  (h_avg_first_two : avg_first_two = 3.6)
  (h_avg_next_two : avg_next_two = 3.85) :
  (total * avg_all - 2 * avg_first_two - 2 * avg_next_two) / 2 = 4.4 := by
  sorry

end average_of_remaining_numbers_l3958_395892


namespace equidistant_point_y_coordinate_l3958_395896

/-- The y-coordinate of the point on the y-axis that is equidistant from A(-3, 0) and B(2, 5) is 2. -/
theorem equidistant_point_y_coordinate : ∃ y : ℝ, 
  ((-3 - 0)^2 + (0 - y)^2 = (2 - 0)^2 + (5 - y)^2) ∧ y = 2 := by
  sorry

end equidistant_point_y_coordinate_l3958_395896


namespace total_amount_l3958_395851

/-- The ratio of money distribution among w, x, y, and z -/
structure MoneyDistribution where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  ratio_condition : x = 0.7 * w ∧ y = 0.5 * w ∧ z = 0.3 * w

/-- The problem statement -/
theorem total_amount (d : MoneyDistribution) (h : d.y = 90) :
  d.w + d.x + d.y + d.z = 450 := by
  sorry


end total_amount_l3958_395851


namespace field_area_calculation_l3958_395873

theorem field_area_calculation (smaller_area larger_area : ℝ) : 
  smaller_area = 315 →
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area + larger_area = 700 := by
  sorry

end field_area_calculation_l3958_395873


namespace apple_orange_ratio_l3958_395865

/-- Represents the number of fruits each child received -/
structure FruitDistribution where
  mike_oranges : ℕ
  matt_apples : ℕ
  mark_bananas : ℕ

/-- The fruit distribution satisfies the problem conditions -/
def valid_distribution (d : FruitDistribution) : Prop :=
  d.mike_oranges = 3 ∧
  d.mark_bananas = d.mike_oranges + d.matt_apples ∧
  d.mike_oranges + d.matt_apples + d.mark_bananas = 18

theorem apple_orange_ratio (d : FruitDistribution) 
  (h : valid_distribution d) : 
  d.matt_apples / d.mike_oranges = 2 := by
  sorry

#check apple_orange_ratio

end apple_orange_ratio_l3958_395865


namespace f_simplification_f_value_when_cos_eq_one_fifth_f_value_at_negative_1860_degrees_l3958_395894

noncomputable section

open Real

def f (α : ℝ) : ℝ := (sin (π - α) * cos (2 * π - α) * tan (-α - π)) / (tan (-α) * sin (-π - α))

theorem f_simplification (α : ℝ) (h : π < α ∧ α < 3 * π / 2) : f α = cos α := by
  sorry

theorem f_value_when_cos_eq_one_fifth (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2) (h2 : cos (α - 3 * π / 2) = 1 / 5) :
  f α = -2 * Real.sqrt 6 / 5 := by
  sorry

theorem f_value_at_negative_1860_degrees :
  f (-1860 * π / 180) = 1 / 2 := by
  sorry

end f_simplification_f_value_when_cos_eq_one_fifth_f_value_at_negative_1860_degrees_l3958_395894


namespace sum_even_implies_diff_even_l3958_395802

theorem sum_even_implies_diff_even (a b : ℤ) : 
  Even (a + b) → Even (a - b) := by
  sorry

end sum_even_implies_diff_even_l3958_395802


namespace largest_number_with_digits_4_1_sum_14_l3958_395803

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 4 ∨ d = 1

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digits_4_1_sum_14 :
  ∀ n : ℕ, is_valid_number n ∧ sum_of_digits n = 14 → n ≤ 4411 :=
by sorry

end largest_number_with_digits_4_1_sum_14_l3958_395803


namespace mia_study_time_l3958_395885

theorem mia_study_time (total_minutes : ℕ) (tv_fraction : ℚ) (study_minutes : ℕ) : 
  total_minutes = 1440 →
  tv_fraction = 1 / 5 →
  study_minutes = 288 →
  (study_minutes : ℚ) / (total_minutes - (tv_fraction * total_minutes : ℚ)) = 1 / 4 := by
  sorry

end mia_study_time_l3958_395885


namespace min_value_expression_min_value_achieved_l3958_395820

theorem min_value_expression (x : ℝ) : (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 :=
sorry

theorem min_value_achieved : ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = 2023 :=
sorry

end min_value_expression_min_value_achieved_l3958_395820


namespace chocolate_distribution_l3958_395834

theorem chocolate_distribution (total : ℕ) (michael_share : ℕ) (paige_share : ℕ) (mandy_share : ℕ) : 
  total = 60 →
  michael_share = total / 2 →
  paige_share = (total - michael_share) / 2 →
  mandy_share = total - michael_share - paige_share →
  mandy_share = 15 := by
sorry

end chocolate_distribution_l3958_395834


namespace total_money_l3958_395835

def jack_money : ℕ := 26
def ben_money : ℕ := jack_money - 9
def eric_money : ℕ := ben_money - 10

theorem total_money : eric_money + ben_money + jack_money = 50 := by
  sorry

end total_money_l3958_395835


namespace find_b_find_perimeter_l3958_395893

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.B = (3 * t.c - t.b) * Real.cos t.A

def side_condition (t : Triangle) : Prop :=
  t.a * Real.sin t.B = 2 * Real.sqrt 2

def area_condition (t : Triangle) : Prop :=
  1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 2

-- Theorem 1
theorem find_b (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : side_condition t) : 
  t.b = 3 :=
sorry

-- Theorem 2
theorem find_perimeter (t : Triangle) 
  (h1 : t.a = 2 * Real.sqrt 2) 
  (h2 : area_condition t) : 
  t.a + t.b + t.c = 4 + 2 * Real.sqrt 2 :=
sorry

end find_b_find_perimeter_l3958_395893


namespace min_sum_polygons_l3958_395821

theorem min_sum_polygons (m n : ℕ) : 
  m ≥ 1 → n ≥ 3 → 
  (180 * m * n - 360 * m) % 8 = 0 → 
  ∀ (m' n' : ℕ), m' ≥ 1 → n' ≥ 3 → (180 * m' * n' - 360 * m') % 8 = 0 → 
  m + n ≤ m' + n' :=
by sorry

end min_sum_polygons_l3958_395821


namespace count_words_beginning_ending_with_A_l3958_395814

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering --/
def word_length : ℕ := 5

/-- The number of variable positions in the word --/
def variable_positions : ℕ := word_length - 2

/-- The number of five-letter words beginning and ending with 'A' --/
def words_beginning_ending_with_A : ℕ := alphabet_size ^ variable_positions

theorem count_words_beginning_ending_with_A :
  words_beginning_ending_with_A = 17576 :=
sorry

end count_words_beginning_ending_with_A_l3958_395814


namespace nina_walking_distance_l3958_395832

/-- Proves that Nina's walking distance to school is 0.4 miles, given John's distance and the difference between their distances. -/
theorem nina_walking_distance
  (john_distance : ℝ)
  (difference : ℝ)
  (h1 : john_distance = 0.7)
  (h2 : difference = 0.3)
  (h3 : john_distance = nina_distance + difference)
  : nina_distance = 0.4 :=
by
  sorry

end nina_walking_distance_l3958_395832


namespace probability_of_either_test_l3958_395899

theorem probability_of_either_test (p_math p_english : ℚ) 
  (h_math : p_math = 5/8)
  (h_english : p_english = 1/4)
  (h_independent : True) -- We don't need to express independence in the theorem statement
  : 1 - (1 - p_math) * (1 - p_english) = 23/32 := by
  sorry

end probability_of_either_test_l3958_395899


namespace jerry_feather_ratio_l3958_395863

def jerryFeatherProblem (hawk_feathers eagle_feathers total_feathers feathers_given feathers_left : ℕ) : Prop :=
  hawk_feathers = 6 ∧
  eagle_feathers = 17 * hawk_feathers ∧
  total_feathers = hawk_feathers + eagle_feathers ∧
  feathers_given = 10 ∧
  feathers_left = 49

theorem jerry_feather_ratio 
  (hawk_feathers eagle_feathers total_feathers feathers_given feathers_left : ℕ) 
  (h : jerryFeatherProblem hawk_feathers eagle_feathers total_feathers feathers_given feathers_left) : 
  ∃ (feathers_after_giving feathers_sold : ℕ),
    feathers_after_giving = total_feathers - feathers_given ∧
    feathers_sold = feathers_after_giving - feathers_left ∧
    2 * feathers_sold = feathers_after_giving :=
sorry

end jerry_feather_ratio_l3958_395863


namespace problem_solution_l3958_395897

theorem problem_solution (a b c : ℚ) 
  (h1 : a + b + c = 72)
  (h2 : a + 4 = b - 8)
  (h3 : a + 4 = 4 * c) : 
  a = 236 / 9 := by
sorry

end problem_solution_l3958_395897


namespace select_representatives_l3958_395825

theorem select_representatives (boys girls total reps : ℕ) 
  (h1 : boys = 6)
  (h2 : girls = 4)
  (h3 : total = boys + girls)
  (h4 : reps = 3) :
  (Nat.choose total reps) - (Nat.choose boys reps) = 100 := by
  sorry

end select_representatives_l3958_395825


namespace unique_kids_count_l3958_395800

/-- The number of unique kids Julia played with across the week -/
def total_unique_kids (monday tuesday wednesday thursday friday : ℕ) 
  (wednesday_from_monday thursday_from_tuesday friday_from_monday friday_from_wednesday : ℕ) : ℕ :=
  monday + tuesday + (wednesday - wednesday_from_monday) + 
  (thursday - thursday_from_tuesday) + 
  (friday - friday_from_monday - (friday_from_wednesday - wednesday_from_monday))

theorem unique_kids_count :
  let monday := 12
  let tuesday := 7
  let wednesday := 15
  let thursday := 10
  let friday := 18
  let wednesday_from_monday := 5
  let thursday_from_tuesday := 7
  let friday_from_monday := 9
  let friday_from_wednesday := 5
  total_unique_kids monday tuesday wednesday thursday friday
    wednesday_from_monday thursday_from_tuesday friday_from_monday friday_from_wednesday = 36 := by
  sorry

end unique_kids_count_l3958_395800


namespace pizza_toppings_l3958_395822

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 24)
  (h2 : pepperoni_slices = 15)
  (h3 : mushroom_slices = 16)
  (h4 : ∀ (slice : ℕ), slice < total_slices → (slice < pepperoni_slices ∨ slice < mushroom_slices)) :
  pepperoni_slices + mushroom_slices - total_slices = 7 :=
by sorry

end pizza_toppings_l3958_395822


namespace W_min_value_l3958_395883

/-- The function W defined on real numbers x and y -/
def W (x y : ℝ) : ℝ := 5 * x^2 - 4 * x * y + y^2 - 2 * y + 8 * x + 3

/-- Theorem stating that W has a minimum value of -2 -/
theorem W_min_value :
  (∀ x y : ℝ, W x y ≥ -2) ∧ (∃ x y : ℝ, W x y = -2) := by
  sorry

end W_min_value_l3958_395883


namespace tourist_meeting_time_l3958_395827

/-- Represents a tourist -/
structure Tourist where
  name : String

/-- Represents a meeting between two tourists -/
structure Meeting where
  tourist1 : Tourist
  tourist2 : Tourist
  time : ℕ  -- Time in hours after noon

/-- The problem setup -/
def tourist_problem (vitya pasha katya masha : Tourist) : Prop :=
  ∃ (vitya_masha vitya_katya pasha_masha pasha_katya : Meeting),
    -- Meetings
    vitya_masha.tourist1 = vitya ∧ vitya_masha.tourist2 = masha ∧ vitya_masha.time = 0 ∧
    vitya_katya.tourist1 = vitya ∧ vitya_katya.tourist2 = katya ∧ vitya_katya.time = 2 ∧
    pasha_masha.tourist1 = pasha ∧ pasha_masha.tourist2 = masha ∧ pasha_masha.time = 3 ∧
    -- Vitya and Pasha travel at the same speed from A to B
    (vitya_masha.time - vitya_katya.time = pasha_masha.time - pasha_katya.time) ∧
    -- Katya and Masha travel at the same speed from B to A
    (vitya_masha.time - pasha_masha.time = vitya_katya.time - pasha_katya.time) →
    pasha_katya.time = 5

theorem tourist_meeting_time (vitya pasha katya masha : Tourist) :
  tourist_problem vitya pasha katya masha := by
  sorry

#check tourist_meeting_time

end tourist_meeting_time_l3958_395827


namespace problem_statement_l3958_395891

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (parallelPlanes : Plane → Plane → Prop)

-- Given two lines a and b, and two planes α and β
variable (a b : Line) (α β : Plane)

-- Given that b is perpendicular to α
variable (h : perpendicular b α)

theorem problem_statement :
  (parallel a α → perpendicularLines a b) ∧
  (perpendicular b β → parallelPlanes α β) := by sorry

end problem_statement_l3958_395891


namespace inequality_range_l3958_395871

theorem inequality_range (m : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ m * x * (x + y)) → 
  -6 ≤ m ∧ m ≤ 2 := by
sorry

end inequality_range_l3958_395871


namespace arithmetic_calculation_l3958_395830

theorem arithmetic_calculation : 8 + (-2)^3 / (-4) * (-7 + 5) = 4 := by
  sorry

end arithmetic_calculation_l3958_395830


namespace interest_rate_problem_l3958_395848

theorem interest_rate_problem (R T : ℝ) : 
  900 * (1 + R * T / 100) = 956 ∧
  900 * (1 + (R + 4) * T / 100) = 1064 →
  T = 3 := by
sorry

end interest_rate_problem_l3958_395848
