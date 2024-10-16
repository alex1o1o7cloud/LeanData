import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l4026_402639

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 5*x + 5/x + 1/x^2 = 42)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l4026_402639


namespace NUMINAMATH_CALUDE_duct_tape_cutting_time_l4026_402610

/-- The time required to cut all strands of duct tape -/
def cutting_time (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ) : ℚ :=
  total_strands / (hannah_rate + son_rate)

/-- Theorem stating the time required to cut all strands -/
theorem duct_tape_cutting_time :
  cutting_time 22 8 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_duct_tape_cutting_time_l4026_402610


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l4026_402625

theorem sum_of_x_and_y_on_circle (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 44) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l4026_402625


namespace NUMINAMATH_CALUDE_cohen_fish_eater_birds_l4026_402696

/-- The number of fish-eater birds Cohen saw on the first day -/
def first_day_birds : ℕ := 300

/-- The number of fish-eater birds Cohen saw on the second day -/
def second_day_birds : ℕ := 2 * first_day_birds

/-- The number of fish-eater birds Cohen saw on the third day -/
def third_day_birds : ℕ := second_day_birds - 200

/-- The total number of fish-eater birds Cohen saw over three days -/
def total_birds : ℕ := 1300

theorem cohen_fish_eater_birds :
  first_day_birds + second_day_birds + third_day_birds = total_birds :=
by sorry

end NUMINAMATH_CALUDE_cohen_fish_eater_birds_l4026_402696


namespace NUMINAMATH_CALUDE_valid_numbers_count_l4026_402670

/-- The number of ways to distribute n identical objects into k distinct boxes --/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of 6-digit positive integers with digits from 1 to 6 in increasing order --/
def total_increasing_numbers : ℕ := stars_and_bars 6 6

/-- The number of 5-digit positive integers with digits from 1 to 5 in increasing order --/
def numbers_starting_with_6 : ℕ := stars_and_bars 5 5

/-- The number of 6-digit positive integers with digits from 1 to 6 in increasing order, not starting with 6 --/
def valid_numbers : ℕ := total_increasing_numbers - numbers_starting_with_6

theorem valid_numbers_count : valid_numbers = 336 := by
  sorry

end NUMINAMATH_CALUDE_valid_numbers_count_l4026_402670


namespace NUMINAMATH_CALUDE_three_card_selection_l4026_402662

/-- The number of cards in the special deck -/
def deck_size : ℕ := 60

/-- The number of cards to be picked -/
def cards_to_pick : ℕ := 3

/-- The number of ways to choose and order 3 different cards from a 60-card deck -/
def ways_to_pick : ℕ := 205320

/-- Theorem stating that the number of ways to choose and order 3 different cards
    from a 60-card deck is equal to 205320 -/
theorem three_card_selection :
  (deck_size * (deck_size - 1) * (deck_size - 2)) = ways_to_pick :=
by sorry

end NUMINAMATH_CALUDE_three_card_selection_l4026_402662


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l4026_402650

theorem sum_of_three_numbers (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xy : x * y = 60) (h_xz : x * z = 90) (h_yz : y * z = 150) : 
  x + y + z = 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l4026_402650


namespace NUMINAMATH_CALUDE_circle_through_points_l4026_402694

theorem circle_through_points : ∃ (A B C D : ℝ), 
  (A * 0^2 + B * 0^2 + C * 0 + D * 0 + 1 = 0) ∧ 
  (A * 4^2 + B * 0^2 + C * 4 + D * 0 + 1 = 0) ∧ 
  (A * (-1)^2 + B * 1^2 + C * (-1) + D * 1 + 1 = 0) ∧ 
  (A = 1 ∧ B = 1 ∧ C = -4 ∧ D = -6) := by
  sorry

end NUMINAMATH_CALUDE_circle_through_points_l4026_402694


namespace NUMINAMATH_CALUDE_sports_participation_l4026_402635

theorem sports_participation (total_students : ℕ) (basketball cricket soccer : ℕ)
  (basketball_cricket basketball_soccer cricket_soccer : ℕ) (all_three : ℕ)
  (h1 : total_students = 50)
  (h2 : basketball = 16)
  (h3 : cricket = 11)
  (h4 : soccer = 10)
  (h5 : basketball_cricket = 5)
  (h6 : basketball_soccer = 4)
  (h7 : cricket_soccer = 3)
  (h8 : all_three = 2) :
  basketball + cricket + soccer - (basketball_cricket + basketball_soccer + cricket_soccer) + all_three = 27 := by
  sorry

end NUMINAMATH_CALUDE_sports_participation_l4026_402635


namespace NUMINAMATH_CALUDE_three_tangent_lines_l4026_402623

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in the 2D plane represented by its equation y^2 = ax -/
structure Parabola where
  a : ℝ

/-- Predicate to check if a line passes through a given point -/
def Line.passesThrough (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Predicate to check if a line has only one common point with a parabola -/
def Line.hasOnlyOneCommonPoint (l : Line) (p : Parabola) : Prop :=
  ∃! x y, l.passesThrough x y ∧ y^2 = p.a * x

/-- The main theorem stating that there are exactly 3 lines passing through (0,6)
    and having only one common point with the parabola y^2 = -12x -/
theorem three_tangent_lines :
  ∃! (lines : Finset Line),
    (∀ l ∈ lines, l.passesThrough 0 6 ∧ l.hasOnlyOneCommonPoint (Parabola.mk (-12))) ∧
    lines.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_tangent_lines_l4026_402623


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l4026_402675

theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, S n = (n / 2) * (a 1 + a n)) →  -- sum formula
  (a 8 / a 7 = 13 / 5) →                -- given condition
  (S 15 / S 13 = 3) :=                  -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l4026_402675


namespace NUMINAMATH_CALUDE_florist_roses_theorem_l4026_402645

/-- Calculates the final number of roses a florist has after selling and picking more roses. -/
def final_roses (initial : ℕ) (sold : ℕ) (picked : ℕ) : ℕ :=
  initial - sold + picked

/-- Proves that the final number of roses is correct given the initial number,
    the number sold, and the number picked. -/
theorem florist_roses_theorem (initial : ℕ) (sold : ℕ) (picked : ℕ) 
    (h1 : initial ≥ sold) : 
  final_roses initial sold picked = initial - sold + picked :=
by
  -- The proof goes here
  sorry

/-- Verifies the specific case from the original problem. -/
example : final_roses 37 16 19 = 40 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_florist_roses_theorem_l4026_402645


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l4026_402646

/-- Given a quadratic function with vertex (6, -2) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 11. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = -2 ↔ x = 6) →  -- vertex condition
  a * 1^2 + b * 1 + c = 0 →                  -- x-intercept condition
  ∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l4026_402646


namespace NUMINAMATH_CALUDE_sequence_matches_given_terms_l4026_402666

/-- Definition of the sequence -/
def a (n : ℕ) : ℚ := (n + 2 : ℚ) / (2 * n + 3 : ℚ)

/-- The theorem stating that the first four terms of the sequence match the given values -/
theorem sequence_matches_given_terms :
  (a 1 = 3 / 5) ∧ (a 2 = 4 / 7) ∧ (a 3 = 5 / 9) ∧ (a 4 = 6 / 11) :=
by sorry

end NUMINAMATH_CALUDE_sequence_matches_given_terms_l4026_402666


namespace NUMINAMATH_CALUDE_base_subtraction_proof_l4026_402613

def to_base_10 (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem base_subtraction_proof :
  let base_7_num := to_base_10 [5, 2, 3] 7
  let base_5_num := to_base_10 [4, 6, 1] 5
  base_7_num - base_5_num = 107 := by sorry

end NUMINAMATH_CALUDE_base_subtraction_proof_l4026_402613


namespace NUMINAMATH_CALUDE_intersection_condition_implies_m_leq_neg_one_l4026_402630

/-- Given sets A and B, prove that if A ∩ B = A, then m ≤ -1 -/
theorem intersection_condition_implies_m_leq_neg_one (m : ℝ) : 
  let A : Set ℝ := {x | |x - 1| < 2}
  let B : Set ℝ := {x | x ≥ m}
  A ∩ B = A → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_implies_m_leq_neg_one_l4026_402630


namespace NUMINAMATH_CALUDE_pentagram_impossible_l4026_402679

/-- Represents a pentagram arrangement of numbers -/
structure PentagramArrangement :=
  (numbers : Fin 10 → ℕ)
  (is_permutation : Function.Injective numbers)
  (valid_range : ∀ i, numbers i ∈ Finset.range 11 \ {0})

/-- Represents a line in the pentagram -/
inductive PentagramLine
  | Line1 | Line2 | Line3 | Line4 | Line5

/-- Get the four positions on a given line -/
def line_positions (l : PentagramLine) : Fin 4 → Fin 10 :=
  sorry  -- Implementation details omitted

/-- The sum of numbers on a given line -/
def line_sum (arr : PentagramArrangement) (l : PentagramLine) : ℕ :=
  (Finset.range 4).sum (λ i => arr.numbers (line_positions l i))

/-- Statement: It's impossible to arrange numbers 1 to 10 in a pentagram
    such that all line sums are equal -/
theorem pentagram_impossible : ¬ ∃ (arr : PentagramArrangement),
  ∀ (l1 l2 : PentagramLine), line_sum arr l1 = line_sum arr l2 :=
sorry

end NUMINAMATH_CALUDE_pentagram_impossible_l4026_402679


namespace NUMINAMATH_CALUDE_sum_coordinates_reflection_l4026_402697

/-- Given a point A with coordinates (x,y) reflected over the y-axis to point B,
    the sum of all coordinates of A and B is equal to 2y. -/
theorem sum_coordinates_reflection (x y : ℝ) : 
  let A := (x, y)
  let B := (-x, y)
  x + y + (-x) + y = 2 * y := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_reflection_l4026_402697


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l4026_402689

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := λ x ↦ (x, 4)

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

-- State the theorem
theorem vector_difference_magnitude :
  ∃ x : ℝ, parallel a (b x) ∧ 
    Real.sqrt ((a.1 - (b x).1)^2 + (a.2 - (b x).2)^2) = 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l4026_402689


namespace NUMINAMATH_CALUDE_root_implies_a_value_l4026_402608

theorem root_implies_a_value (a : ℝ) : 
  ((-2 : ℝ)^2 + 3*(-2) + a = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_value_l4026_402608


namespace NUMINAMATH_CALUDE_daleyza_project_units_l4026_402620

/-- Calculates the total number of units in a three-building construction project -/
def total_units (first_building : ℕ) : ℕ :=
  let second_building := (2 : ℕ) * first_building / 5
  let third_building := (6 : ℕ) * second_building / 5
  first_building + second_building + third_building

/-- Theorem stating that given the specific conditions of Daleyza's project, 
    the total number of units is 7520 -/
theorem daleyza_project_units : total_units 4000 = 7520 := by
  sorry

end NUMINAMATH_CALUDE_daleyza_project_units_l4026_402620


namespace NUMINAMATH_CALUDE_odd_numbers_sum_product_l4026_402663

theorem odd_numbers_sum_product (n : ℕ) (odds : Finset ℕ) (a b c : ℕ) : 
  n = 1997 →
  odds.card = n →
  (∀ x ∈ odds, Odd x) →
  (odds.sum id = odds.prod id) →
  a ∈ odds ∧ b ∈ odds ∧ c ∈ odds →
  a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 →
  a.Prime ∧ b.Prime ∧ c.Prime →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a = 5 ∧ b = 7 ∧ c = 59) ∨ (a = 5 ∧ b = 59 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 5 ∧ c = 59) ∨ (a = 7 ∧ b = 59 ∧ c = 5) ∨ 
  (a = 59 ∧ b = 5 ∧ c = 7) ∨ (a = 59 ∧ b = 7 ∧ c = 5) :=
by sorry


end NUMINAMATH_CALUDE_odd_numbers_sum_product_l4026_402663


namespace NUMINAMATH_CALUDE_wave_number_probability_l4026_402674

/-- A permutation of the digits 1,2,3,4,5 --/
def Permutation := Fin 5 → Fin 5

/-- A permutation is valid if it's bijective --/
def is_valid_permutation (p : Permutation) : Prop :=
  Function.Bijective p

/-- A permutation represents a wave number if it satisfies the wave pattern --/
def is_wave_number (p : Permutation) : Prop :=
  p 0 < p 1 ∧ p 1 > p 2 ∧ p 2 < p 3 ∧ p 3 > p 4

/-- The total number of valid permutations --/
def total_permutations : ℕ := 120

/-- The number of wave numbers --/
def wave_numbers : ℕ := 16

/-- The main theorem: probability of selecting a wave number --/
theorem wave_number_probability :
  (wave_numbers : ℚ) / total_permutations = 2 / 15 := by sorry

end NUMINAMATH_CALUDE_wave_number_probability_l4026_402674


namespace NUMINAMATH_CALUDE_points_form_circle_l4026_402660

theorem points_form_circle :
  ∀ (x y : ℝ), (∃ t : ℝ, x = Real.cos t ∧ y = Real.sin t) → x^2 + y^2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_points_form_circle_l4026_402660


namespace NUMINAMATH_CALUDE_borrowed_sum_l4026_402667

/-- Proves that given the conditions of the problem, the principal amount is 1050 --/
theorem borrowed_sum (P : ℝ) : 
  (P * 0.06 * 6 = P - 672) → P = 1050 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sum_l4026_402667


namespace NUMINAMATH_CALUDE_max_sum_n_l4026_402684

/-- An arithmetic sequence with first term 11 and common difference -2 -/
def arithmeticSequence (n : ℕ) : ℤ :=
  11 - 2 * (n - 1)

/-- The sum of the first n terms of the arithmetic sequence -/
def sumOfTerms (n : ℕ) : ℤ :=
  n * (arithmeticSequence 1 + arithmeticSequence n) / 2

/-- The value of n that maximizes the sum of the first n terms -/
theorem max_sum_n : ∃ (n : ℕ), n = 6 ∧ 
  ∀ (m : ℕ), sumOfTerms m ≤ sumOfTerms n :=
sorry

end NUMINAMATH_CALUDE_max_sum_n_l4026_402684


namespace NUMINAMATH_CALUDE_ufo_convention_attendees_l4026_402671

theorem ufo_convention_attendees 
  (total_attendees : ℕ) 
  (total_presenters : ℕ) 
  (male_presenters female_presenters : ℕ) 
  (male_general female_general : ℕ) :
  total_attendees = 1000 →
  total_presenters = 420 →
  male_presenters = female_presenters + 20 →
  female_general = male_general + 56 →
  total_attendees = total_presenters + male_general + female_general →
  male_general = 262 := by
sorry

end NUMINAMATH_CALUDE_ufo_convention_attendees_l4026_402671


namespace NUMINAMATH_CALUDE_area_between_specific_lines_l4026_402647

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Calculate the area between two lines given their defining points -/
def areaBetweenLines (l1 : Line) (l2 : Line) : ℝ :=
  sorry

/-- The main theorem stating that the area between the given lines is 40.8 -/
theorem area_between_specific_lines :
  let l1 : Line := { p1 := { x := 0, y := 3 }, p2 := { x := 5, y := 2 } }
  let l2 : Line := { p1 := { x := 2, y := 6 }, p2 := { x := 6, y := 1 } }
  areaBetweenLines l1 l2 = 40.8 := by
  sorry

end NUMINAMATH_CALUDE_area_between_specific_lines_l4026_402647


namespace NUMINAMATH_CALUDE_f_composition_l4026_402657

def f (x : ℝ) : ℝ := 2 * x + 1

theorem f_composition : ∀ x : ℝ, f (f x) = 4 * x + 3 := by sorry

end NUMINAMATH_CALUDE_f_composition_l4026_402657


namespace NUMINAMATH_CALUDE_popcorn_probability_l4026_402668

theorem popcorn_probability : 
  let white_ratio : ℚ := 3/4
  let yellow_ratio : ℚ := 1/4
  let white_pop_prob : ℚ := 2/3
  let yellow_pop_prob : ℚ := 3/4
  let fizz_prob : ℚ := 1/4
  
  let white_pop_fizz : ℚ := white_ratio * white_pop_prob * fizz_prob
  let yellow_pop_fizz : ℚ := yellow_ratio * yellow_pop_prob * fizz_prob
  let total_pop_fizz : ℚ := white_pop_fizz + yellow_pop_fizz
  
  white_pop_fizz / total_pop_fizz = 8/11 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_probability_l4026_402668


namespace NUMINAMATH_CALUDE_min_contribution_problem_l4026_402617

/-- Proves that given 12 people contributing a total of $20.00, with a maximum individual contribution of $9.00, the minimum amount each person must have contributed is $1.00. -/
theorem min_contribution_problem (n : ℕ) (total : ℚ) (max_contrib : ℚ) (h1 : n = 12) (h2 : total = 20) (h3 : max_contrib = 9) : 
  ∃ (min_contrib : ℚ), 
    min_contrib = 1 ∧ 
    n * min_contrib ≤ total ∧ 
    ∀ (individual_contrib : ℚ), individual_contrib ≤ max_contrib → 
      (n - 1) * min_contrib + individual_contrib ≤ total :=
by sorry

end NUMINAMATH_CALUDE_min_contribution_problem_l4026_402617


namespace NUMINAMATH_CALUDE_prove_arrangements_l4026_402698

def num_students : ℕ := 7

def adjacent_pair : ℕ := 1
def non_adjacent_pair : ℕ := 1
def remaining_students : ℕ := num_students - 4

def arrangements_theorem : Prop :=
  (num_students = 7) →
  (adjacent_pair = 1) →
  (non_adjacent_pair = 1) →
  (remaining_students = num_students - 4) →
  (Nat.factorial 2 * Nat.factorial 4 * (Nat.factorial 5 / Nat.factorial 3) =
   Nat.factorial 2 * Nat.factorial 4 * Nat.factorial 5 / Nat.factorial 3)

theorem prove_arrangements : arrangements_theorem := by sorry

end NUMINAMATH_CALUDE_prove_arrangements_l4026_402698


namespace NUMINAMATH_CALUDE_set_contains_all_integers_l4026_402681

def is_closed_under_subtraction (A : Set ℤ) : Prop :=
  ∀ x y, x ∈ A → y ∈ A → (x - y) ∈ A

theorem set_contains_all_integers (A : Set ℤ) 
  (h_closed : is_closed_under_subtraction A) 
  (h_four : 4 ∈ A) 
  (h_nine : 9 ∈ A) : 
  A = Set.univ :=
sorry

end NUMINAMATH_CALUDE_set_contains_all_integers_l4026_402681


namespace NUMINAMATH_CALUDE_ribbon_per_box_l4026_402676

/-- Given the total amount of ribbon, the amount remaining, and the number of boxes tied,
    calculate the amount of ribbon used per box. -/
theorem ribbon_per_box 
  (total_ribbon : ℝ) 
  (remaining_ribbon : ℝ) 
  (num_boxes : ℕ) 
  (h1 : total_ribbon = 4.5)
  (h2 : remaining_ribbon = 1)
  (h3 : num_boxes = 5) :
  (total_ribbon - remaining_ribbon) / num_boxes = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_per_box_l4026_402676


namespace NUMINAMATH_CALUDE_chi_square_test_error_probability_l4026_402673

/-- Represents the chi-square statistic -/
def chi_square : ℝ := 15.02

/-- Represents the critical value -/
def critical_value : ℝ := 6.635

/-- Represents the p-value -/
def p_value : ℝ := 0.01

/-- Represents the sample size -/
def sample_size : ℕ := 1000

/-- Represents the probability of error in rejecting the null hypothesis -/
def error_probability : ℝ := p_value

theorem chi_square_test_error_probability :
  error_probability = p_value :=
sorry

end NUMINAMATH_CALUDE_chi_square_test_error_probability_l4026_402673


namespace NUMINAMATH_CALUDE_bookshelf_problem_l4026_402628

/-- Bookshelf purchasing problem -/
theorem bookshelf_problem (price_A price_B : ℕ) 
  (h1 : 3 * price_A + 2 * price_B = 1020)
  (h2 : 4 * price_A + 3 * price_B = 1440)
  (total_bookshelves : ℕ) (h3 : total_bookshelves = 20)
  (max_budget : ℕ) (h4 : max_budget = 4320) :
  (price_A = 180 ∧ price_B = 240) ∧ 
  (∃ (m : ℕ), 
    (m = 8 ∨ m = 9 ∨ m = 10) ∧
    (total_bookshelves - m ≥ m) ∧
    (price_A * m + price_B * (total_bookshelves - m) ≤ max_budget)) := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_problem_l4026_402628


namespace NUMINAMATH_CALUDE_sum_square_diagonals_formula_l4026_402624

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  R : ℝ  -- radius of the circumscribed circle
  OP : ℝ  -- length of segment OP
  h_R_pos : R > 0  -- radius is positive
  h_OP_pos : OP > 0  -- OP is positive
  h_OP_le_2R : OP ≤ 2 * R  -- OP cannot be longer than the diameter

/-- The sum of squares of diagonals of an inscribed quadrilateral -/
def sumSquareDiagonals (q : InscribedQuadrilateral) : ℝ :=
  8 * q.R^2 - 4 * q.OP^2

/-- Theorem: The sum of squares of diagonals of an inscribed quadrilateral
    is equal to 8R^2 - 4OP^2 -/
theorem sum_square_diagonals_formula (q : InscribedQuadrilateral) :
  sumSquareDiagonals q = 8 * q.R^2 - 4 * q.OP^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_square_diagonals_formula_l4026_402624


namespace NUMINAMATH_CALUDE_production_rates_l4026_402644

/-- The rate at which A makes parts per hour -/
def rate_A : ℝ := sorry

/-- The rate at which B makes parts per hour -/
def rate_B : ℝ := sorry

/-- The time it takes for A to make 90 parts equals the time for B to make 120 parts -/
axiom time_ratio : (90 / rate_A) = (120 / rate_B)

/-- A and B together make 35 parts per hour -/
axiom total_rate : rate_A + rate_B = 35

theorem production_rates : rate_A = 15 ∧ rate_B = 20 := by
  sorry

end NUMINAMATH_CALUDE_production_rates_l4026_402644


namespace NUMINAMATH_CALUDE_fish_cost_theorem_l4026_402691

theorem fish_cost_theorem (dog_fish : ℕ) (fish_price : ℕ) :
  dog_fish = 40 →
  fish_price = 4 →
  (dog_fish + dog_fish / 2) * fish_price = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_cost_theorem_l4026_402691


namespace NUMINAMATH_CALUDE_difference_of_numbers_l4026_402658

theorem difference_of_numbers (x y : ℝ) (h_sum : x + y = 36) (h_product : x * y = 105) :
  |x - y| = 6 * Real.sqrt 24.333 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l4026_402658


namespace NUMINAMATH_CALUDE_square_less_than_triple_l4026_402643

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l4026_402643


namespace NUMINAMATH_CALUDE_alice_exceeds_quota_by_655_l4026_402680

/-- Represents the sales information for a shoe brand -/
structure ShoeBrand where
  name : String
  cost : Nat
  maxSales : Nat
  actualSales : Nat

/-- Calculates the total sales for a given shoe brand -/
def calculateSales (brand : ShoeBrand) : Nat :=
  brand.cost * brand.actualSales

/-- Calculates the total sales across all shoe brands -/
def totalSales (brands : List ShoeBrand) : Nat :=
  brands.foldl (fun acc brand => acc + calculateSales brand) 0

/-- The main theorem stating that Alice exceeds her quota by $655 -/
theorem alice_exceeds_quota_by_655 (brands : List ShoeBrand) (quota : Nat) : 
  brands = [
    { name := "Adidas", cost := 45, maxSales := 15, actualSales := 10 },
    { name := "Nike", cost := 60, maxSales := 12, actualSales := 12 },
    { name := "Reeboks", cost := 35, maxSales := 20, actualSales := 15 },
    { name := "Puma", cost := 50, maxSales := 10, actualSales := 8 },
    { name := "Converse", cost := 40, maxSales := 18, actualSales := 14 }
  ] ∧ quota = 2000 →
  totalSales brands - quota = 655 := by
  sorry

end NUMINAMATH_CALUDE_alice_exceeds_quota_by_655_l4026_402680


namespace NUMINAMATH_CALUDE_smallest_valid_number_l4026_402618

def is_valid_divisor (d : ℕ) : Prop :=
  d > 0 ∧ 150 % d = 50 ∧ 55 % d = 5 ∧ 175 % d = 25

def is_greatest_divisor (d : ℕ) : Prop :=
  is_valid_divisor d ∧ ∀ k > d, ¬is_valid_divisor k

theorem smallest_valid_number : ∃ n : ℕ, n > 0 ∧ is_valid_divisor n ∧ 
  ∃ d : ℕ, is_greatest_divisor d ∧ n % d = 5 ∧ 
  ∀ m < n, ¬(is_valid_divisor m ∧ ∃ k : ℕ, is_greatest_divisor k ∧ m % k = 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l4026_402618


namespace NUMINAMATH_CALUDE_boat_journey_time_l4026_402614

/-- Calculates the total journey time for a boat traveling upstream and downstream -/
theorem boat_journey_time 
  (distance : ℝ) 
  (initial_current_speed : ℝ) 
  (upstream_current_speed : ℝ) 
  (boat_still_speed : ℝ) 
  (headwind_speed_reduction : ℝ) : 
  let upstream_time := distance / (boat_still_speed - upstream_current_speed)
  let downstream_speed := (boat_still_speed - headwind_speed_reduction) + initial_current_speed
  let downstream_time := distance / downstream_speed
  upstream_time + downstream_time = 26.67 :=
by
  sorry

#check boat_journey_time 56 2 3 6 1

end NUMINAMATH_CALUDE_boat_journey_time_l4026_402614


namespace NUMINAMATH_CALUDE_root_sum_theorem_l4026_402603

theorem root_sum_theorem (a b c : ℝ) : 
  a^3 - 15*a^2 + 22*a - 8 = 0 →
  b^3 - 15*b^2 + 22*b - 8 = 0 →
  c^3 - 15*c^2 + 22*c - 8 = 0 →
  a / (1/a + b*c) + b / (1/b + c*a) + c / (1/c + a*b) = 181/9 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l4026_402603


namespace NUMINAMATH_CALUDE_unique_solution_implies_equal_absolute_values_l4026_402659

theorem unique_solution_implies_equal_absolute_values (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃! x, a * (x - a)^2 + b * (x - b)^2 = 0) → |a| = |b| :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_equal_absolute_values_l4026_402659


namespace NUMINAMATH_CALUDE_compound_composition_l4026_402686

/-- Atomic weight of hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1

/-- Atomic weight of chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.5

/-- Atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 68

/-- Number of oxygen atoms in the compound -/
def n : ℕ := 2

theorem compound_composition :
  molecular_weight = atomic_weight_H + atomic_weight_Cl + n * atomic_weight_O :=
sorry

end NUMINAMATH_CALUDE_compound_composition_l4026_402686


namespace NUMINAMATH_CALUDE_function_non_negative_implies_bounds_l4026_402688

theorem function_non_negative_implies_bounds 
  (a b A B : ℝ) 
  (h : ∀ θ : ℝ, 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_function_non_negative_implies_bounds_l4026_402688


namespace NUMINAMATH_CALUDE_complement_of_union_equals_set_l4026_402615

-- Define the universal set U
def U : Finset Nat := {1,2,3,4,5}

-- Define set A
def A : Finset Nat := {1,2}

-- Define set B
def B : Finset Nat := {2,3}

-- Theorem statement
theorem complement_of_union_equals_set (h : U = {1,2,3,4,5} ∧ A = {1,2} ∧ B = {2,3}) :
  (U \ (A ∪ B)) = {4,5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_set_l4026_402615


namespace NUMINAMATH_CALUDE_unique_number_outside_range_l4026_402655

theorem unique_number_outside_range 
  (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) 
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = (a * x + b) / (c * x + d))
  (hf19 : f 19 = 19)
  (hf97 : f 97 = 97)
  (hfinv : ∀ x, x ≠ -d/c → f (f x) = x) :
  ∃! y, ∀ x, f x ≠ y ∧ y = 58 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_outside_range_l4026_402655


namespace NUMINAMATH_CALUDE_prime_divides_sum_of_powers_l4026_402665

theorem prime_divides_sum_of_powers (p : ℕ) (hp : Prime p) :
  ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_sum_of_powers_l4026_402665


namespace NUMINAMATH_CALUDE_fred_final_card_count_l4026_402638

/-- Calculates the final number of baseball cards Fred has after a series of transactions. -/
def final_card_count (initial : ℕ) (given_away : ℕ) (traded : ℕ) (received : ℕ) : ℕ :=
  initial - given_away + received

/-- Proves that Fred ends up with 6 baseball cards after the given transactions. -/
theorem fred_final_card_count :
  final_card_count 5 2 1 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_fred_final_card_count_l4026_402638


namespace NUMINAMATH_CALUDE_paper_boat_travel_time_l4026_402656

/-- Represents the problem of calculating the time for a paper boat to travel along an embankment --/
theorem paper_boat_travel_time 
  (embankment_length : ℝ)
  (boat_length : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (h1 : embankment_length = 50)
  (h2 : boat_length = 10)
  (h3 : downstream_time = 5)
  (h4 : upstream_time = 4) :
  let downstream_speed := embankment_length / downstream_time
  let upstream_speed := embankment_length / upstream_time
  let boat_speed := (downstream_speed + upstream_speed) / 2
  let current_speed := (downstream_speed - upstream_speed) / 2
  (embankment_length / current_speed) = 40 := by
  sorry

end NUMINAMATH_CALUDE_paper_boat_travel_time_l4026_402656


namespace NUMINAMATH_CALUDE_periodic_function_value_l4026_402692

def periodic_function (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x : ℝ, f (x + period) = f x

theorem periodic_function_value 
  (f : ℝ → ℝ) 
  (h_periodic : periodic_function f (π / 2))
  (h_value : f (π / 3) = 1) : 
  f (17 * π / 6) = 1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l4026_402692


namespace NUMINAMATH_CALUDE_burger_cost_is_100_l4026_402653

/-- The cost of a burger in cents -/
def burger_cost : ℕ := 100

/-- The cost of a soda in cents -/
def soda_cost : ℕ := 50

/-- Charles' purchase -/
def charles_purchase (b s : ℕ) : Prop := 4 * b + 3 * s = 550

/-- Alice's purchase -/
def alice_purchase (b s : ℕ) : Prop := 3 * b + 2 * s = 400

/-- Bill's purchase -/
def bill_purchase (b s : ℕ) : Prop := 2 * b + s = 250

theorem burger_cost_is_100 :
  charles_purchase burger_cost soda_cost ∧
  alice_purchase burger_cost soda_cost ∧
  bill_purchase burger_cost soda_cost ∧
  burger_cost = 100 :=
sorry

end NUMINAMATH_CALUDE_burger_cost_is_100_l4026_402653


namespace NUMINAMATH_CALUDE_geometric_sequence_value_l4026_402699

theorem geometric_sequence_value (a : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ (2*a + 2) = r * a ∧ (3*a + 3) = r * (2*a + 2)) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_value_l4026_402699


namespace NUMINAMATH_CALUDE_george_and_hannah_win_l4026_402685

-- Define the set of students
inductive Student : Type
  | Elaine : Student
  | Frank : Student
  | George : Student
  | Hannah : Student

-- Define a function to represent winning a prize
def wins_prize (s : Student) : Prop := sorry

-- Define the conditions
axiom only_two_winners :
  ∃ (a b : Student), a ≠ b ∧
    (∀ s : Student, wins_prize s ↔ (s = a ∨ s = b))

axiom elaine_implies_frank :
  wins_prize Student.Elaine → wins_prize Student.Frank

axiom frank_implies_george :
  wins_prize Student.Frank → wins_prize Student.George

axiom george_implies_hannah :
  wins_prize Student.George → wins_prize Student.Hannah

-- Theorem to prove
theorem george_and_hannah_win :
  wins_prize Student.George ∧ wins_prize Student.Hannah ∧
  ¬wins_prize Student.Elaine ∧ ¬wins_prize Student.Frank :=
sorry

end NUMINAMATH_CALUDE_george_and_hannah_win_l4026_402685


namespace NUMINAMATH_CALUDE_expression_evaluation_l4026_402612

theorem expression_evaluation : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4026_402612


namespace NUMINAMATH_CALUDE_fourth_number_11th_row_l4026_402690

/-- The last number in row i of the pattern -/
def lastNumber (i : ℕ) : ℕ := 5 * i

/-- The fourth number in row i of the pattern -/
def fourthNumber (i : ℕ) : ℕ := lastNumber i - 1

/-- Theorem: The fourth number in the 11th row is 54 -/
theorem fourth_number_11th_row :
  fourthNumber 11 = 54 := by sorry

end NUMINAMATH_CALUDE_fourth_number_11th_row_l4026_402690


namespace NUMINAMATH_CALUDE_average_visitors_theorem_l4026_402606

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitors (daysInMonth : Nat) (sundayVisitors : Nat) (otherDayVisitors : Nat) : Nat :=
  let numSundays := (daysInMonth + 6) / 7
  let numOtherDays := daysInMonth - numSundays
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / daysInMonth

theorem average_visitors_theorem :
  averageVisitors 30 660 240 = 296 := by
  sorry

end NUMINAMATH_CALUDE_average_visitors_theorem_l4026_402606


namespace NUMINAMATH_CALUDE_exam_pass_count_l4026_402687

theorem exam_pass_count :
  let total_candidates : ℕ := 120
  let overall_average : ℚ := 35
  let pass_average : ℚ := 39
  let fail_average : ℚ := 15
  ∃ pass_count : ℕ,
    pass_count ≤ total_candidates ∧
    (pass_count : ℚ) * pass_average + (total_candidates - pass_count : ℚ) * fail_average = 
      (total_candidates : ℚ) * overall_average ∧
    pass_count = 100 := by
sorry

end NUMINAMATH_CALUDE_exam_pass_count_l4026_402687


namespace NUMINAMATH_CALUDE_equation_solutions_count_l4026_402602

theorem equation_solutions_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 + x - 12)^2 = 81) ∧ s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l4026_402602


namespace NUMINAMATH_CALUDE_boys_in_class_l4026_402621

theorem boys_in_class (total : ℕ) (girls_ratio boys_ratio others_ratio : ℕ) 
  (h1 : total = 63)
  (h2 : girls_ratio = 4)
  (h3 : boys_ratio = 3)
  (h4 : others_ratio = 2)
  (h5 : ∃ k : ℕ, total = k * (girls_ratio + boys_ratio + others_ratio)) :
  ∃ num_boys : ℕ, num_boys = 21 ∧ num_boys * (girls_ratio + boys_ratio + others_ratio) = boys_ratio * total :=
by
  sorry

#check boys_in_class

end NUMINAMATH_CALUDE_boys_in_class_l4026_402621


namespace NUMINAMATH_CALUDE_rectangle_area_21_implies_y_7_l4026_402652

/-- Represents a rectangle EFGH with vertices E(0, 0), F(0, 3), G(y, 3), and H(y, 0) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := 3 * r.y

theorem rectangle_area_21_implies_y_7 (r : Rectangle) (h : area r = 21) : r.y = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_21_implies_y_7_l4026_402652


namespace NUMINAMATH_CALUDE_otimes_two_three_l4026_402632

-- Define the new operation ⊗
def otimes (a b : ℝ) : ℝ := 4 * a + 5 * b

-- Theorem to prove
theorem otimes_two_three : otimes 2 3 = 23 := by
  sorry

end NUMINAMATH_CALUDE_otimes_two_three_l4026_402632


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l4026_402678

-- Define the cyclic sum function
def cyclicSum (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c + f b c a + f c a b

-- State the theorem
theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  cyclicSum (fun x y z => (y + z) * (x^4 - y^2 * z^2) / (x*y + 2*y*z + z*x)) a b c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l4026_402678


namespace NUMINAMATH_CALUDE_probability_no_more_than_one_complaint_probability_two_complaints_in_two_months_l4026_402636

-- Define the probabilities for complaints in a single month
def p_zero_complaints : ℝ := 0.3
def p_one_complaint : ℝ := 0.5
def p_two_complaints : ℝ := 0.2

-- Theorem for part (I)
theorem probability_no_more_than_one_complaint :
  p_zero_complaints + p_one_complaint = 0.8 := by sorry

-- Theorem for part (II)
theorem probability_two_complaints_in_two_months :
  let p_two_total := p_zero_complaints * p_two_complaints +
                     p_two_complaints * p_zero_complaints +
                     p_one_complaint * p_one_complaint
  p_two_total = 0.37 := by sorry

end NUMINAMATH_CALUDE_probability_no_more_than_one_complaint_probability_two_complaints_in_two_months_l4026_402636


namespace NUMINAMATH_CALUDE_minimum_area_of_rectangle_l4026_402642

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Checks if the actual dimensions are within the reported range --/
def withinReportedRange (reported : Rectangle) (actual : Rectangle) : Prop :=
  (actual.length ≥ reported.length - 0.5) ∧
  (actual.length ≤ reported.length + 0.5) ∧
  (actual.width ≥ reported.width - 0.5) ∧
  (actual.width ≤ reported.width + 0.5)

/-- Checks if the length is at least twice the width --/
def lengthAtLeastTwiceWidth (r : Rectangle) : Prop :=
  r.length ≥ 2 * r.width

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- The reported dimensions of the tile --/
def reportedDimensions : Rectangle :=
  { length := 4, width := 6 }

theorem minimum_area_of_rectangle :
  ∃ (minRect : Rectangle),
    withinReportedRange reportedDimensions minRect ∧
    lengthAtLeastTwiceWidth minRect ∧
    area minRect = 60.5 ∧
    ∀ (r : Rectangle),
      withinReportedRange reportedDimensions r →
      lengthAtLeastTwiceWidth r →
      area r ≥ 60.5 := by
  sorry

end NUMINAMATH_CALUDE_minimum_area_of_rectangle_l4026_402642


namespace NUMINAMATH_CALUDE_no_factorial_with_2021_zeros_l4026_402601

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- There is no natural number n such that n! ends with exactly 2021 zeros -/
theorem no_factorial_with_2021_zeros : ∀ n : ℕ, trailingZeros n ≠ 2021 := by
  sorry

end NUMINAMATH_CALUDE_no_factorial_with_2021_zeros_l4026_402601


namespace NUMINAMATH_CALUDE_simplify_fraction_l4026_402622

theorem simplify_fraction : 18 * (8 / 15) * (1 / 12) = 18 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4026_402622


namespace NUMINAMATH_CALUDE_min_value_on_line_l4026_402661

theorem min_value_on_line (x y : ℝ) (h : x + 2 * y = 3) :
  2^x + 4^y ≥ 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_line_l4026_402661


namespace NUMINAMATH_CALUDE_age_difference_l4026_402605

theorem age_difference (c d : ℕ) (hc : c < 10) (hd : d < 10) 
  (h : 10 * c + d + 10 = 3 * (10 * d + c + 10)) :
  (10 * c + d) - (10 * d + c) = 54 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l4026_402605


namespace NUMINAMATH_CALUDE_lcm_one_to_ten_l4026_402695

theorem lcm_one_to_ten : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := by
  sorry

#eval Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))

end NUMINAMATH_CALUDE_lcm_one_to_ten_l4026_402695


namespace NUMINAMATH_CALUDE_sum_of_squares_l4026_402693

theorem sum_of_squares (a b c d : ℝ) : 
  a + b = -3 →
  a * b + b * c + c * a = -4 →
  a * b * c + b * c * d + c * d * a + d * a * b = 14 →
  a * b * c * d = 30 →
  a^2 + b^2 + c^2 + d^2 = 141 / 4 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l4026_402693


namespace NUMINAMATH_CALUDE_S_seven_two_l4026_402609

def S (a b : ℕ) : ℕ := 3 * a + 5 * b

theorem S_seven_two : S 7 2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_S_seven_two_l4026_402609


namespace NUMINAMATH_CALUDE_problem_solution_l4026_402641

theorem problem_solution (p q : Prop) (hp : 1 > -2) (hq : Even 2) : 
  (p ∨ q) ∧ (p ∧ q) ∧ ¬(¬p) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4026_402641


namespace NUMINAMATH_CALUDE_translated_line_equation_l4026_402631

/-- 
Theorem: The equation of a line with slope 2 passing through the point (2, 5) is y = 2x + 1.
-/
theorem translated_line_equation (x y : ℝ) : 
  (∃ b : ℝ, y = 2 * x + b) ∧ (2 = 2 ∧ 5 = 2 * 2 + y - 2 * x) → y = 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_translated_line_equation_l4026_402631


namespace NUMINAMATH_CALUDE_complex_imaginary_part_eq_two_l4026_402649

theorem complex_imaginary_part_eq_two (a : ℝ) :
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  Complex.im z = 2 → a = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_eq_two_l4026_402649


namespace NUMINAMATH_CALUDE_min_value_of_f_l4026_402634

/-- The function f(x) = (x^2 + ax - 1)e^(x-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 
  ((2*x + a) * Real.exp (x - 1)) + ((x^2 + a*x - 1) * Real.exp (x - 1))

theorem min_value_of_f (a : ℝ) :
  (f_deriv a (-2) = 0) →  -- x = -2 is an extremum point
  (∃ x, f a x = -1) ∧ (∀ y, f a y ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l4026_402634


namespace NUMINAMATH_CALUDE_jane_albert_same_committee_l4026_402640

def total_people : ℕ := 6
def committee_size : ℕ := 3

def probability_same_committee : ℚ := 1 / 5

theorem jane_albert_same_committee :
  let total_combinations := Nat.choose total_people committee_size
  let favorable_combinations := Nat.choose (total_people - 2) (committee_size - 2)
  (favorable_combinations : ℚ) / total_combinations = probability_same_committee :=
sorry

end NUMINAMATH_CALUDE_jane_albert_same_committee_l4026_402640


namespace NUMINAMATH_CALUDE_nonzero_x_equality_l4026_402682

theorem nonzero_x_equality (x : ℝ) (hx : x ≠ 0) (h : (9 * x)^18 = (18 * x)^9) : x = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_x_equality_l4026_402682


namespace NUMINAMATH_CALUDE_kannon_bananas_l4026_402669

/-- Proves that Kannon had 1 banana last night given the conditions of the problem -/
theorem kannon_bananas : 
  ∀ (bananas_last_night : ℕ),
    (3 + bananas_last_night + 4) +  -- fruits last night
    ((3 + 4) + 10 * bananas_last_night + 2 * (3 + 4)) = 39 → -- fruits today
    bananas_last_night = 1 := by
  sorry

end NUMINAMATH_CALUDE_kannon_bananas_l4026_402669


namespace NUMINAMATH_CALUDE_printer_cost_l4026_402683

/-- The cost of a printer given the conditions of the merchant's purchase. -/
theorem printer_cost (total_cost : ℕ) (keyboard_cost : ℕ) (num_keyboards : ℕ) (num_printers : ℕ)
  (h1 : total_cost = 2050)
  (h2 : keyboard_cost = 20)
  (h3 : num_keyboards = 15)
  (h4 : num_printers = 25) :
  (total_cost - num_keyboards * keyboard_cost) / num_printers = 70 := by
  sorry

end NUMINAMATH_CALUDE_printer_cost_l4026_402683


namespace NUMINAMATH_CALUDE_powers_of_two_difference_divisible_by_1987_l4026_402648

theorem powers_of_two_difference_divisible_by_1987 :
  ∃ a b : ℕ, 0 ≤ a ∧ a < b ∧ b ≤ 1987 ∧ (1987 ∣ 2^b - 2^a) :=
sorry

end NUMINAMATH_CALUDE_powers_of_two_difference_divisible_by_1987_l4026_402648


namespace NUMINAMATH_CALUDE_english_score_l4026_402672

theorem english_score (korean math : ℕ) (h1 : (korean + math) / 2 = 88) 
  (h2 : (korean + math + 94) / 3 = 90) : 94 = 94 := by
  sorry

end NUMINAMATH_CALUDE_english_score_l4026_402672


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4026_402626

def A : Set (ℝ × ℝ) := {p | p.2 = p.1^2}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1}

theorem intersection_of_A_and_B : A ∩ B = {(0, 0), (1, 1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4026_402626


namespace NUMINAMATH_CALUDE_gear_teeth_problem_l4026_402607

theorem gear_teeth_problem (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 60) (h4 : 4 * x - 20 = 5 * y) (h5 : 5 * y = 10 * z) : x = 30 ∧ y = 20 ∧ z = 10 := by
  sorry

end NUMINAMATH_CALUDE_gear_teeth_problem_l4026_402607


namespace NUMINAMATH_CALUDE_parallel_to_y_axis_l4026_402600

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the problem
theorem parallel_to_y_axis (m n : ℝ) :
  let A : Point2D := ⟨-3, m⟩
  let B : Point2D := ⟨n, -4⟩
  (A.x = B.x) → -- Condition for line AB to be parallel to y-axis
  (n = -3 ∧ m ≠ -4) := by
  sorry


end NUMINAMATH_CALUDE_parallel_to_y_axis_l4026_402600


namespace NUMINAMATH_CALUDE_factor_x9_minus_512_l4026_402677

theorem factor_x9_minus_512 (x : ℝ) : 
  x^9 - 512 = (x - 2) * (x^2 + 2*x + 4) * (x^6 + 2*x^3 + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_x9_minus_512_l4026_402677


namespace NUMINAMATH_CALUDE_negative_response_proportion_l4026_402633

/-- Given 88 total teams and 49 teams with negative responses,
    prove that P = ⌊10000 * (49/88)⌋ = 5568 -/
theorem negative_response_proportion (total_teams : Nat) (negative_responses : Nat)
    (h1 : total_teams = 88)
    (h2 : negative_responses = 49) :
    ⌊(10000 : ℝ) * ((negative_responses : ℝ) / (total_teams : ℝ))⌋ = 5568 := by
  sorry

#check negative_response_proportion

end NUMINAMATH_CALUDE_negative_response_proportion_l4026_402633


namespace NUMINAMATH_CALUDE_marble_ratio_proof_l4026_402654

def marble_problem (initial_marbles : ℕ) (lost_marbles : ℕ) (final_marbles : ℕ) : Prop :=
  let marbles_after_loss := initial_marbles - lost_marbles
  let marbles_given_away := 2 * lost_marbles
  let marbles_before_dog_ate := marbles_after_loss - marbles_given_away
  let marbles_eaten_by_dog := marbles_before_dog_ate - final_marbles
  (2 * marbles_eaten_by_dog = lost_marbles) ∧ (marbles_eaten_by_dog > 0) ∧ (lost_marbles > 0)

theorem marble_ratio_proof : marble_problem 24 4 10 := by sorry

end NUMINAMATH_CALUDE_marble_ratio_proof_l4026_402654


namespace NUMINAMATH_CALUDE_james_rainwater_profit_l4026_402604

/-- Calculates the money James made from selling rainwater collected over two days -/
theorem james_rainwater_profit : 
  let gallons_per_inch : ℝ := 15
  let monday_rain : ℝ := 4
  let tuesday_rain : ℝ := 3
  let price_per_gallon : ℝ := 1.2
  let total_gallons := gallons_per_inch * (monday_rain + tuesday_rain)
  total_gallons * price_per_gallon = 126 := by
sorry


end NUMINAMATH_CALUDE_james_rainwater_profit_l4026_402604


namespace NUMINAMATH_CALUDE_negation_equivalence_l4026_402637

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 + 2*x + 2 ≤ 0) ↔ 
  (∀ x : ℝ, x > 1 → x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4026_402637


namespace NUMINAMATH_CALUDE_john_walks_point_seven_miles_l4026_402627

/-- The distance Nina walks to school in miles -/
def nina_distance : ℝ := 0.4

/-- The additional distance John walks compared to Nina in miles -/
def john_additional_distance : ℝ := 0.3

/-- John's distance to school in miles -/
def john_distance : ℝ := nina_distance + john_additional_distance

/-- Theorem stating that John walks 0.7 miles to school -/
theorem john_walks_point_seven_miles : john_distance = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_john_walks_point_seven_miles_l4026_402627


namespace NUMINAMATH_CALUDE_min_value_of_f_l4026_402629

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem min_value_of_f (a : ℝ) (h1 : a > 2) 
  (h2 : ∀ x > 2, f x ≥ f a) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l4026_402629


namespace NUMINAMATH_CALUDE_sequence_sum_l4026_402651

theorem sequence_sum (A H M O X : ℕ) : 
  (A + 9 + H = 19) →
  (9 + H + M = 19) →
  (H + M + O = 19) →
  (M + O + X = 19) →
  (O + X + 7 = 19) →
  (A ≠ H) → (A ≠ M) → (A ≠ O) → (A ≠ X) →
  (H ≠ M) → (H ≠ O) → (H ≠ X) →
  (M ≠ O) → (M ≠ X) →
  (O ≠ X) →
  A + H + M + O = 26 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l4026_402651


namespace NUMINAMATH_CALUDE_sunflower_height_l4026_402619

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Converts inches to feet, rounding down -/
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem sunflower_height
  (sister_height_feet : ℕ)
  (sister_height_inches : ℕ)
  (sunflower_diff : ℕ)
  (h1 : sister_height_feet = 4)
  (h2 : sister_height_inches = 3)
  (h3 : sunflower_diff = 21) :
  inches_to_feet (feet_inches_to_inches sister_height_feet sister_height_inches + sunflower_diff) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_height_l4026_402619


namespace NUMINAMATH_CALUDE_wind_velocity_calculation_l4026_402664

/-- The relationship between pressure, area, and velocity -/
def pressure_relationship (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^3

/-- The theorem to prove -/
theorem wind_velocity_calculation (k : ℝ) :
  pressure_relationship k 2 8 = 4 →
  pressure_relationship k 4 12.8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_wind_velocity_calculation_l4026_402664


namespace NUMINAMATH_CALUDE_increasing_function_sum_inequality_l4026_402611

/-- An increasing function on the real line. -/
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem: For an increasing function f and real numbers a and b,
    if a + b ≥ 0, then f(a) + f(b) ≥ f(-a) + f(-b). -/
theorem increasing_function_sum_inequality
  (f : ℝ → ℝ) (hf : IncreasingFunction f) (a b : ℝ) :
  a + b ≥ 0 → f a + f b ≥ f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_sum_inequality_l4026_402611


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l4026_402616

theorem unique_two_digit_number : ∃! n : ℕ,
  10 ≤ n ∧ n < 100 ∧
  (∃ a b : ℕ, a ≠ b ∧ a < 10 ∧ b < 10 ∧ n = 10 * a + b) ∧
  n^2 = (n / 10 + n % 10)^3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l4026_402616
