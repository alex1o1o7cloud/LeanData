import Mathlib

namespace kids_at_camp_l2624_262473

theorem kids_at_camp (total : ℕ) (stay_home : ℕ) (h1 : total = 898051) (h2 : stay_home = 268627) :
  total - stay_home = 629424 := by
  sorry

end kids_at_camp_l2624_262473


namespace expansion_has_four_nonzero_terms_l2624_262456

/-- The polynomial expression to be expanded -/
def polynomial_expression (x : ℝ) : ℝ :=
  (2*x + 5) * (3*x^2 + x + 6) - 4*(x^3 + 3*x^2 - 4*x + 1)

/-- The expanded form of the polynomial expression -/
def expanded_polynomial (x : ℝ) : ℝ :=
  2*x^3 + 5*x^2 + 33*x + 26

/-- Theorem stating that the expansion has exactly 4 nonzero terms -/
theorem expansion_has_four_nonzero_terms :
  ∃ (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0),
  ∀ x, polynomial_expression x = a*x^3 + b*x^2 + c*x + d :=
sorry

end expansion_has_four_nonzero_terms_l2624_262456


namespace proposition_equivalence_l2624_262455

theorem proposition_equivalence (A B : Set α) :
  (∀ x, x ∈ A → x ∈ B) ↔ (∀ x, x ∉ B → x ∉ A) := by
  sorry

end proposition_equivalence_l2624_262455


namespace man_in_well_l2624_262472

/-- The number of days required for a man to climb out of a well -/
def daysToClimbOut (wellDepth : ℕ) (climbUp : ℕ) (slipDown : ℕ) : ℕ :=
  let netClimbPerDay := climbUp - slipDown
  (wellDepth - 1) / netClimbPerDay + 1

/-- Theorem: It takes 30 days for a man to climb out of a 30-meter deep well
    when he climbs 4 meters up and slips 3 meters down each day -/
theorem man_in_well : daysToClimbOut 30 4 3 = 30 := by
  sorry

end man_in_well_l2624_262472


namespace parabola_c_value_l2624_262447

/-- Represents a parabola of the form x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  x = p.a * y^2 + p.b * y + p.c

/-- The vertex of a parabola -/
def Parabola.vertex (p : Parabola) : ℝ × ℝ := sorry

theorem parabola_c_value :
  ∀ p : Parabola,
  p.vertex = (3, -5) →
  p.contains 0 6 →
  p.c = 288 / 121 := by
  sorry

end parabola_c_value_l2624_262447


namespace max_area_rectangular_pen_l2624_262462

theorem max_area_rectangular_pen (fencing : ℝ) (h_fencing : fencing = 60) :
  let width := fencing / 6
  let length := 2 * width
  let area := width * length
  area = 200 := by sorry

end max_area_rectangular_pen_l2624_262462


namespace rectangle_dimensions_l2624_262414

/-- A rectangle with perimeter 60 meters and area 221 square meters has dimensions 17 meters and 13 meters. -/
theorem rectangle_dimensions (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 60) (h_area : l * w = 221) :
  (l = 17 ∧ w = 13) ∨ (l = 13 ∧ w = 17) := by
  sorry

end rectangle_dimensions_l2624_262414


namespace f_at_negative_four_l2624_262492

/-- The polynomial f(x) = 12 + 35x − 8x^2 + 79x^3 + 6x^4 + 5x^5 + 3x^6 -/
def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

/-- Theorem: The value of f(-4) is 3392 -/
theorem f_at_negative_four : f (-4) = 3392 := by
  sorry

end f_at_negative_four_l2624_262492


namespace circle_symmetry_l2624_262427

-- Define the original circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y = 2

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x-3)^2 + (y+1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ x y : ℝ, 
  (∃ x₀ y₀ : ℝ, circle_C x₀ y₀ ∧ 
    (x + x₀)/2 - (y + y₀)/2 = 2 ∧ 
    (y - y₀)/(x - x₀) = -1) →
  symmetric_circle x y :=
sorry

end circle_symmetry_l2624_262427


namespace lowest_unique_score_l2624_262474

/-- Represents the scoring system for the national Mathematics Competition. -/
structure ScoringSystem where
  totalProblems : ℕ
  correctPoints : ℕ
  wrongPoints : ℕ
  baseScore : ℕ

/-- Calculates the score based on the number of correct and wrong answers. -/
def calculateScore (system : ScoringSystem) (correct : ℕ) (wrong : ℕ) : ℕ :=
  system.baseScore + system.correctPoints * correct - system.wrongPoints * wrong

/-- Checks if the number of correct answers can be uniquely determined from the score. -/
def isUniqueDetermination (system : ScoringSystem) (score : ℕ) : Prop :=
  ∃! correct : ℕ, ∃ wrong : ℕ,
    correct + wrong ≤ system.totalProblems ∧
    calculateScore system correct wrong = score

/-- The theorem stating that 105 is the lowest score above 100 for which
    the number of correctly solved problems can be uniquely determined. -/
theorem lowest_unique_score (system : ScoringSystem)
    (h1 : system.totalProblems = 40)
    (h2 : system.correctPoints = 5)
    (h3 : system.wrongPoints = 1)
    (h4 : system.baseScore = 40) :
    (∀ s, 100 < s → s < 105 → ¬ isUniqueDetermination system s) ∧
    isUniqueDetermination system 105 := by
  sorry

end lowest_unique_score_l2624_262474


namespace cos_105_degrees_l2624_262485

theorem cos_105_degrees :
  Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_degrees_l2624_262485


namespace intersection_S_T_l2624_262413

def S : Set ℝ := {x | x^2 + 2*x = 0}
def T : Set ℝ := {x | x^2 - 2*x = 0}

theorem intersection_S_T : S ∩ T = {0} := by sorry

end intersection_S_T_l2624_262413


namespace barrels_for_remaining_road_l2624_262400

/-- Represents the road paving problem -/
structure RoadPaving where
  total_length : ℝ
  truckloads_per_mile : ℝ
  day1_paved : ℝ
  day2_paved : ℝ
  pitch_per_truckload : ℝ

/-- Calculates the barrels of pitch needed for the remaining road -/
def barrels_needed (rp : RoadPaving) : ℝ :=
  (rp.total_length - (rp.day1_paved + rp.day2_paved)) * rp.truckloads_per_mile * rp.pitch_per_truckload

/-- Theorem stating the number of barrels needed for the given scenario -/
theorem barrels_for_remaining_road :
  let rp : RoadPaving := {
    total_length := 16,
    truckloads_per_mile := 3,
    day1_paved := 4,
    day2_paved := 7,
    pitch_per_truckload := 0.4
  }
  barrels_needed rp = 6 := by sorry

end barrels_for_remaining_road_l2624_262400


namespace laurie_kurt_difference_l2624_262416

/-- The number of marbles each person has -/
structure Marbles where
  dennis : ℕ
  kurt : ℕ
  laurie : ℕ

/-- The conditions of the problem -/
def marble_problem (m : Marbles) : Prop :=
  m.dennis = 70 ∧ 
  m.kurt = m.dennis - 45 ∧
  m.laurie = 37

/-- The theorem to prove -/
theorem laurie_kurt_difference (m : Marbles) 
  (h : marble_problem m) : m.laurie - m.kurt = 12 := by
  sorry

end laurie_kurt_difference_l2624_262416


namespace perfect_square_floor_equality_l2624_262424

theorem perfect_square_floor_equality (n : ℕ+) :
  ⌊2 * Real.sqrt n⌋ = ⌊Real.sqrt (n - 1) + Real.sqrt (n + 1)⌋ + 1 ↔ ∃ m : ℕ+, n = m^2 :=
by sorry

end perfect_square_floor_equality_l2624_262424


namespace repeating_decimal_equals_fraction_l2624_262489

/-- Represents the repeating decimal 0.35247̄ -/
def repeating_decimal : ℚ :=
  35247 / 100000 + (247 / 100000) / (1 - 1 / 1000)

/-- The fraction we want to prove equality with -/
def target_fraction : ℚ := 3518950 / 999900

/-- Theorem stating that the repeating decimal equals the target fraction -/
theorem repeating_decimal_equals_fraction :
  repeating_decimal = target_fraction := by sorry

end repeating_decimal_equals_fraction_l2624_262489


namespace bridge_length_calculation_l2624_262436

/-- Proves the length of a bridge given train specifications and crossing times -/
theorem bridge_length_calculation (train_length : ℝ) (signal_post_time : ℝ) (bridge_time : ℝ) :
  train_length = 600 →
  signal_post_time = 40 →
  bridge_time = 600 →
  let train_speed := train_length / signal_post_time
  let bridge_length := train_speed * bridge_time - train_length
  bridge_length = 8400 := by
sorry

end bridge_length_calculation_l2624_262436


namespace impossibleTable_l2624_262423

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- Represents a 10x10 table of digits -/
def Table := Fin 10 → Fin 10 → Digit

/-- Converts a sequence of 10 digits to a natural number -/
def toNumber (seq : Fin 10 → Digit) : ℕ := sorry

/-- The main theorem stating the impossibility of constructing the required table -/
theorem impossibleTable : ¬ ∃ (t : Table),
  (∀ i : Fin 10, toNumber (λ j => t i j) > toNumber (λ k => t k k)) ∧
  (∀ j : Fin 10, toNumber (λ k => t k k) > toNumber (λ i => t i j)) := by
  sorry


end impossibleTable_l2624_262423


namespace hockey_team_size_l2624_262468

/-- Calculates the total number of players on a hockey team given specific conditions -/
theorem hockey_team_size 
  (percent_boys : ℚ)
  (num_junior_girls : ℕ)
  (h1 : percent_boys = 60 / 100)
  (h2 : num_junior_girls = 10) : 
  (2 * num_junior_girls : ℚ) / (1 - percent_boys) = 50 := by
  sorry

end hockey_team_size_l2624_262468


namespace arithmetic_expression_equality_l2624_262429

theorem arithmetic_expression_equality : 9 - 3 / (1 / 3) + 3 = 3 := by
  sorry

end arithmetic_expression_equality_l2624_262429


namespace pi_half_irrational_l2624_262458

theorem pi_half_irrational : Irrational (π / 2) :=
by
  sorry

end pi_half_irrational_l2624_262458


namespace jordana_age_proof_l2624_262437

/-- Jennifer's age in ten years -/
def jennifer_future_age : ℕ := 30

/-- The number of years in the future we're considering -/
def years_ahead : ℕ := 10

/-- Jordana's age relative to Jennifer's in the future -/
def jordana_relative_age : ℕ := 3

/-- Jordana's current age -/
def jordana_current_age : ℕ := 80

theorem jordana_age_proof :
  jordana_current_age = jennifer_future_age * jordana_relative_age - years_ahead :=
by sorry

end jordana_age_proof_l2624_262437


namespace min_value_implies_a_l2624_262430

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + 2 * |x - a|

/-- The theorem stating the relationship between the minimum value of f and the value of a -/
theorem min_value_implies_a (a : ℝ) : 
  (∀ x, f a x ≥ 5) ∧ (∃ x, f a x = 5) → a = -6 ∨ a = 4 :=
sorry

end min_value_implies_a_l2624_262430


namespace latest_start_time_is_10am_l2624_262435

-- Define the number of turkeys
def num_turkeys : ℕ := 2

-- Define the weight of each turkey in pounds
def turkey_weight : ℕ := 16

-- Define the roasting time per pound in minutes
def roasting_time_per_pound : ℕ := 15

-- Define the dinner time (18:00 in 24-hour format)
def dinner_time : ℕ := 18 * 60

-- Define the function to calculate the total roasting time in minutes
def total_roasting_time : ℕ := num_turkeys * turkey_weight * roasting_time_per_pound

-- Define the function to calculate the latest start time in minutes after midnight
def latest_start_time : ℕ := dinner_time - total_roasting_time

-- Theorem stating that the latest start time is 10:00 am (600 minutes after midnight)
theorem latest_start_time_is_10am : latest_start_time = 600 := by
  sorry

end latest_start_time_is_10am_l2624_262435


namespace quadratic_function_solution_set_l2624_262488

/-- Given a quadratic function f(x) = x^2 + bx + 1 where f(-1) = f(3),
    prove that the solution set of f(x) > 0 is {x ∈ ℝ | x ≠ 1} -/
theorem quadratic_function_solution_set
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^2 + b*x + 1)
  (h2 : f (-1) = f 3)
  : {x : ℝ | f x > 0} = {x : ℝ | x ≠ 1} :=
by sorry

end quadratic_function_solution_set_l2624_262488


namespace smallest_common_factor_l2624_262428

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (6*m + 4))) ∧ 
  (∃ k : ℕ, k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (6*n + 4)) → 
  n = 1 := by
sorry

end smallest_common_factor_l2624_262428


namespace lives_difference_l2624_262407

/-- The number of lives of animals in a fictional world --/
def lives_problem (cat_lives dog_lives mouse_lives : ℕ) : Prop :=
  cat_lives = 9 ∧
  dog_lives = cat_lives - 3 ∧
  mouse_lives = 13 ∧
  mouse_lives - dog_lives = 7

theorem lives_difference :
  ∃ (cat_lives dog_lives mouse_lives : ℕ),
    lives_problem cat_lives dog_lives mouse_lives :=
by
  sorry

end lives_difference_l2624_262407


namespace binomial_coefficient_third_term_l2624_262402

theorem binomial_coefficient_third_term (a b : ℝ) : 
  Nat.choose 6 2 = 15 := by
  sorry

end binomial_coefficient_third_term_l2624_262402


namespace bobby_deadlift_difference_l2624_262446

/-- Given Bobby's initial deadlift and yearly increase, prove the difference between his deadlift at 18 and 250% of his deadlift at 13 -/
theorem bobby_deadlift_difference (initial_deadlift : ℕ) (yearly_increase : ℕ) (years : ℕ) : 
  initial_deadlift = 300 →
  yearly_increase = 110 →
  years = 5 →
  (initial_deadlift + years * yearly_increase) - (initial_deadlift * 250 / 100) = 100 := by
sorry

end bobby_deadlift_difference_l2624_262446


namespace triangle_side_length_l2624_262442

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (2 * b = a + c) →  -- a, b, c form an arithmetic sequence
  (B = π / 6) →  -- Angle B = 30° (in radians)
  (1 / 2 * a * c * Real.sin B = 3 / 2) →  -- Area of triangle ABC = 3/2
  -- Conclusion
  b = Real.sqrt 3 + 1 := by
  sorry

end triangle_side_length_l2624_262442


namespace abs_diff_eq_sum_abs_iff_product_nonpositive_l2624_262410

theorem abs_diff_eq_sum_abs_iff_product_nonpositive (a b : ℝ) :
  |a - b| = |a| + |b| ↔ a * b ≤ 0 := by
  sorry

end abs_diff_eq_sum_abs_iff_product_nonpositive_l2624_262410


namespace unique_sums_count_l2624_262452

def X : Finset ℕ := {1, 4, 5, 7}
def Y : Finset ℕ := {3, 4, 6, 8}

theorem unique_sums_count : 
  Finset.card ((X.product Y).image (fun p => p.1 + p.2)) = 10 := by
  sorry

end unique_sums_count_l2624_262452


namespace grid_square_triangle_count_l2624_262433

/-- Represents a square divided into a 4x4 grid with diagonals --/
structure GridSquare :=
  (size : ℕ)
  (has_diagonals : Bool)
  (has_small_square_diagonals : Bool)

/-- Counts the number of triangles in a GridSquare --/
def count_triangles (sq : GridSquare) : ℕ :=
  sorry

/-- The main theorem stating that a 4x4 GridSquare with all diagonals has 42 triangles --/
theorem grid_square_triangle_count :
  ∀ (sq : GridSquare), 
    sq.size = 4 ∧ 
    sq.has_diagonals = true ∧ 
    sq.has_small_square_diagonals = true → 
    count_triangles sq = 42 :=
  sorry

end grid_square_triangle_count_l2624_262433


namespace prob_three_pass_min_students_scheme_A_l2624_262453

/-- Represents the two testing schemes -/
inductive Scheme
| A
| B

/-- Represents a student -/
structure Student where
  name : String
  scheme : Scheme

/-- Probability of passing for each scheme -/
def passProbability (s : Scheme) : ℚ :=
  match s with
  | Scheme.A => 2/3
  | Scheme.B => 1/2

/-- Group of students participating in the test -/
def testGroup : List Student := [
  ⟨"A", Scheme.A⟩, ⟨"B", Scheme.A⟩, ⟨"C", Scheme.A⟩,
  ⟨"D", Scheme.B⟩, ⟨"E", Scheme.B⟩
]

/-- Calculates the probability of exactly k students passing out of n students -/
def probExactlyKPass (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

/-- Theorem: The probability of exactly three students passing the test is 19/54 -/
theorem prob_three_pass :
  (probExactlyKPass 3 3 (passProbability Scheme.A) *
   probExactlyKPass 2 0 (passProbability Scheme.B)) +
  (probExactlyKPass 3 2 (passProbability Scheme.A) *
   probExactlyKPass 2 1 (passProbability Scheme.B)) +
  (probExactlyKPass 3 1 (passProbability Scheme.A) *
   probExactlyKPass 2 2 (passProbability Scheme.B)) = 19/54 := by
  sorry

/-- Expected number of passing students given n students choose scheme A -/
def expectedPass (n : ℕ) : ℚ := n * (passProbability Scheme.A) + (5 - n) * (passProbability Scheme.B)

/-- Theorem: The minimum number of students choosing scheme A for the expected number
    of passing students to be at least 3 is 3 -/
theorem min_students_scheme_A :
  (∀ m : ℕ, m < 3 → expectedPass m < 3) ∧
  expectedPass 3 ≥ 3 := by
  sorry

end prob_three_pass_min_students_scheme_A_l2624_262453


namespace q_satisfies_conditions_l2624_262497

/-- A quadratic polynomial that satisfies specific conditions -/
def q (x : ℝ) : ℝ := x^2 + 1

/-- Theorem stating that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-2) = 5 ∧ q 1 = 2 ∧ q 3 = 10 := by
  sorry

#eval q (-2)
#eval q 1
#eval q 3

end q_satisfies_conditions_l2624_262497


namespace three_digit_palindrome_gcf_and_divisibility_l2624_262495

/-- Represents a three-digit palindrome -/
def ThreeDigitPalindrome : Type := { n : ℕ // ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 102 * a + 10 * b }

/-- The set of all three-digit palindromes -/
def AllThreeDigitPalindromes : Set ℕ :=
  { n | ∃ (p : ThreeDigitPalindrome), n = p.val }

theorem three_digit_palindrome_gcf_and_divisibility :
  (∃ (g : ℕ), g > 0 ∧ 
    (∀ n ∈ AllThreeDigitPalindromes, g ∣ n) ∧
    (∀ d : ℕ, d > 0 → (∀ n ∈ AllThreeDigitPalindromes, d ∣ n) → d ∣ g)) ∧
  (∀ n ∈ AllThreeDigitPalindromes, 3 ∣ n) :=
by sorry

end three_digit_palindrome_gcf_and_divisibility_l2624_262495


namespace expression_evaluation_l2624_262405

theorem expression_evaluation :
  let x : ℤ := -2
  (2 * x + 1) * (x - 2) - (2 - x)^2 = -8 := by sorry

end expression_evaluation_l2624_262405


namespace total_students_count_l2624_262466

def third_grade : ℕ := 19

def fourth_grade : ℕ := 2 * third_grade

def second_grade_boys : ℕ := 10
def second_grade_girls : ℕ := 19

def total_students : ℕ := third_grade + fourth_grade + second_grade_boys + second_grade_girls

theorem total_students_count : total_students = 86 := by sorry

end total_students_count_l2624_262466


namespace circle_area_difference_l2624_262425

/-- The difference between the areas of two circles with given circumferences -/
theorem circle_area_difference (c₁ c₂ : ℝ) (h₁ : c₁ = 660) (h₂ : c₂ = 704) :
  ∃ (diff : ℝ), abs (diff - ((c₂^2 - c₁^2) / (4 * Real.pi))) < 0.001 := by
  sorry

end circle_area_difference_l2624_262425


namespace shoe_cost_theorem_l2624_262487

/-- Calculates the final price after applying discount and tax -/
def calculate_price (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  discounted_price * (1 + tax_rate)

/-- Calculates the total cost of four pairs of shoes -/
def total_cost (price1 price2 price3 : ℝ) : ℝ :=
  let pair1 := calculate_price price1 0.10 0.05
  let pair2 := calculate_price (price1 * 1.5) 0.15 0.07
  let pair3_4 := price3 * (1 + 0.12)
  pair1 + pair2 + pair3_4

theorem shoe_cost_theorem :
  total_cost 22 (22 * 1.5) 40 = 95.60 := by
  sorry

end shoe_cost_theorem_l2624_262487


namespace fixed_point_on_line_l2624_262449

theorem fixed_point_on_line (m : ℝ) : (2*m - 1)*2 + (m + 3)*(-3) - (m - 11) = 0 := by
  sorry

end fixed_point_on_line_l2624_262449


namespace solve_quadratic_equation_l2624_262438

theorem solve_quadratic_equation (x : ℝ) (h1 : 3 * x^2 - 9 * x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end solve_quadratic_equation_l2624_262438


namespace parallel_vectors_l2624_262422

/-- Given two vectors AB and CD in R², if AB is parallel to CD and AB = (6,1) and CD = (x,-3), then x = -18 -/
theorem parallel_vectors (AB CD : ℝ × ℝ) (x : ℝ) 
  (h1 : AB = (6, 1)) 
  (h2 : CD = (x, -3)) 
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ AB = k • CD) : 
  x = -18 := by
  sorry

end parallel_vectors_l2624_262422


namespace circle_intersection_m_range_l2624_262444

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*m*y + m + 6 = 0

-- Define the condition that intersections are on the same side of the origin
def intersections_same_side (m : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, y₁ * y₂ > 0 ∧ circle_equation 0 y₁ m ∧ circle_equation 0 y₂ m

-- Theorem statement
theorem circle_intersection_m_range :
  ∀ m : ℝ, intersections_same_side m → (m > 2 ∨ (-6 < m ∧ m < -2)) :=
sorry

end circle_intersection_m_range_l2624_262444


namespace balance_theorem_l2624_262419

-- Define the weights of balls in terms of blue balls
def green_weight : ℚ := 2
def yellow_weight : ℚ := 5/2
def white_weight : ℚ := 3/2

-- Define the balance conditions
axiom green_balance : 3 * green_weight = 6
axiom yellow_balance : 2 * yellow_weight = 5
axiom white_balance : 6 = 4 * white_weight

-- Theorem to prove
theorem balance_theorem : 
  4 * green_weight + 2 * yellow_weight + 2 * white_weight = 16 := by
  sorry


end balance_theorem_l2624_262419


namespace distributions_five_balls_three_boxes_l2624_262417

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def total_distributions (n k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes
    such that at least one specific box is empty -/
def distributions_with_empty_box (n k : ℕ) : ℕ := (k - 1)^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes
    such that at least two specific boxes are empty -/
def distributions_with_two_empty_boxes (n k : ℕ) : ℕ := 1

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes
    such that no box remains empty -/
def distributions_no_empty_boxes (n k : ℕ) : ℕ :=
  total_distributions n k - 
  (k * distributions_with_empty_box n k) +
  (Nat.choose k 2 * distributions_with_two_empty_boxes n k)

theorem distributions_five_balls_three_boxes :
  distributions_no_empty_boxes 5 3 = 150 := by
  sorry

end distributions_five_balls_three_boxes_l2624_262417


namespace function_value_range_bounds_are_tight_l2624_262491

theorem function_value_range (x : ℝ) : 
  ∃ (y : ℝ), y = Real.sin x - Real.cos (x + π/6) ∧ 
  -Real.sqrt 3 ≤ y ∧ y ≤ Real.sqrt 3 :=
by sorry

theorem bounds_are_tight : 
  (∃ (x : ℝ), Real.sin x - Real.cos (x + π/6) = -Real.sqrt 3) ∧
  (∃ (x : ℝ), Real.sin x - Real.cos (x + π/6) = Real.sqrt 3) :=
by sorry

end function_value_range_bounds_are_tight_l2624_262491


namespace max_teams_advancing_l2624_262464

/-- The number of teams in the tournament -/
def num_teams : ℕ := 7

/-- The minimum number of points required to advance -/
def min_points_to_advance : ℕ := 13

/-- The number of points awarded for a win -/
def win_points : ℕ := 3

/-- The number of points awarded for a draw -/
def draw_points : ℕ := 1

/-- The number of points awarded for a loss -/
def loss_points : ℕ := 0

/-- The total number of games played in the tournament -/
def total_games : ℕ := (num_teams * (num_teams - 1)) / 2

/-- The maximum total points that can be awarded in the tournament -/
def max_total_points : ℕ := total_games * win_points

/-- Theorem stating the maximum number of teams that can advance -/
theorem max_teams_advancing :
  ∀ n : ℕ, (n * min_points_to_advance ≤ max_total_points) →
  (∀ m : ℕ, m > n → m * min_points_to_advance > max_total_points) →
  n = 4 := by sorry

end max_teams_advancing_l2624_262464


namespace washer_dryer_cost_l2624_262409

theorem washer_dryer_cost (dryer_cost washer_cost total_cost : ℕ) : 
  dryer_cost = 150 →
  washer_cost = 3 * dryer_cost →
  total_cost = washer_cost + dryer_cost →
  total_cost = 600 := by
sorry

end washer_dryer_cost_l2624_262409


namespace jonathan_took_45_oranges_l2624_262471

/-- The number of oranges Jonathan took -/
def oranges_taken (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that Jonathan took 45 oranges -/
theorem jonathan_took_45_oranges :
  oranges_taken 96 51 = 45 := by
  sorry

end jonathan_took_45_oranges_l2624_262471


namespace square_sum_given_product_and_sum_l2624_262411

theorem square_sum_given_product_and_sum (a b : ℝ) 
  (h1 : a * b = 16) 
  (h2 : a + b = 10) : 
  a^2 + b^2 = 68 := by
sorry

end square_sum_given_product_and_sum_l2624_262411


namespace james_adoption_payment_l2624_262496

/-- The total amount James pays for adopting a puppy and a kitten -/
def jamesPayment (puppyFee kittenFee : ℚ) (multiPetDiscount : ℚ) 
  (friendPuppyContribution friendKittenContribution : ℚ) : ℚ :=
  let totalFee := puppyFee + kittenFee
  let discountedFee := totalFee * (1 - multiPetDiscount)
  let friendContributions := puppyFee * friendPuppyContribution + kittenFee * friendKittenContribution
  discountedFee - friendContributions

/-- Theorem stating that James pays $242.50 for adopting a puppy and a kitten -/
theorem james_adoption_payment :
  jamesPayment 200 150 (1/10) (1/4) (3/20) = 485/2 :=
by sorry

end james_adoption_payment_l2624_262496


namespace work_completion_time_l2624_262432

/-- Given two people working together can complete a task in 10 days,
    and one person (Prakash) can complete the task in 30 days,
    prove that the other person can complete the task alone in 15 days. -/
theorem work_completion_time (prakash_time : ℕ) (joint_time : ℕ) (x : ℕ) :
  prakash_time = 30 →
  joint_time = 10 →
  (1 : ℚ) / x + (1 : ℚ) / prakash_time = (1 : ℚ) / joint_time →
  x = 15 := by
  sorry

end work_completion_time_l2624_262432


namespace specific_quadrilateral_perimeter_l2624_262426

/-- A convex quadrilateral with an interior point -/
structure ConvexQuadrilateral where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  Q : ℝ × ℝ
  area : ℝ
  wq : ℝ
  xq : ℝ
  yq : ℝ
  zq : ℝ
  convex : Bool
  interior : Bool

/-- The perimeter of a quadrilateral -/
def perimeter (quad : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem: The perimeter of the specific quadrilateral is 230 + 10√41 -/
theorem specific_quadrilateral_perimeter (quad : ConvexQuadrilateral) 
  (h_area : quad.area = 2500)
  (h_wq : quad.wq = 30)
  (h_xq : quad.xq = 40)
  (h_yq : quad.yq = 50)
  (h_zq : quad.zq = 60)
  (h_convex : quad.convex = true)
  (h_interior : quad.interior = true) :
  perimeter quad = 230 + 10 * Real.sqrt 41 :=
sorry

end specific_quadrilateral_perimeter_l2624_262426


namespace min_value_expression_l2624_262481

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + 2 * q) + (5 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) ≥ 151 / 18 := by
  sorry

end min_value_expression_l2624_262481


namespace vector_problem_l2624_262498

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

/-- Dot product of two 2D vectors -/
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

/-- Acute angle between vectors -/
def is_acute_angle (v w : Fin 2 → ℝ) : Prop := 
  dot_product v w > 0 ∧ ¬ ∃ (k : ℝ), v = fun i => k * (w i)

/-- Orthogonality of vectors -/
def is_orthogonal (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

theorem vector_problem (x : ℝ) :
  (is_acute_angle a (b x) ↔ x > -2 ∧ x ≠ 1/2) ∧
  (is_orthogonal (fun i => a i + 2 * (b x i)) (fun i => 2 * a i - b x i) ↔ x = 7/2) := by
  sorry

end vector_problem_l2624_262498


namespace intersection_of_M_and_N_l2624_262454

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 2}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Ici 0 := by
  sorry

end intersection_of_M_and_N_l2624_262454


namespace cross_section_distance_l2624_262404

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  /-- The distance from the apex to the base -/
  height : ℝ
  /-- The side length of the base hexagon -/
  baseSide : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- The distance from the apex to the cross section -/
  distanceFromApex : ℝ
  /-- The area of the cross section -/
  area : ℝ

/-- The theorem to be proved -/
theorem cross_section_distance (pyramid : RightHexagonalPyramid) 
  (section1 section2 : CrossSection) :
  section1.area = 162 * Real.sqrt 3 →
  section2.area = 288 * Real.sqrt 3 →
  |section1.distanceFromApex - section2.distanceFromApex| = 6 →
  max section1.distanceFromApex section2.distanceFromApex = 24 :=
by sorry

end cross_section_distance_l2624_262404


namespace unique_positive_p_for_geometric_progression_l2624_262439

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r

/-- The theorem states that 4 is the only positive real number p such that -p-12, 2√p, and p-5 form a geometric progression. -/
theorem unique_positive_p_for_geometric_progression :
  ∃! p : ℝ, p > 0 ∧ IsGeometricProgression (-p - 12) (2 * Real.sqrt p) (p - 5) :=
by
  sorry

#check unique_positive_p_for_geometric_progression

end unique_positive_p_for_geometric_progression_l2624_262439


namespace system_of_equations_solution_l2624_262406

theorem system_of_equations_solution (x y m : ℝ) : 
  x + 2*y = 5*m →
  x - 2*y = 9*m →
  3*x + 2*y = 19 →
  m = 1 := by
sorry

end system_of_equations_solution_l2624_262406


namespace floor_length_approx_l2624_262493

/-- Represents the dimensions and painting cost of a rectangular floor. -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  paintingCost : ℝ
  paintingRate : ℝ

/-- The length of the floor is 200% more than the breadth. -/
def lengthCondition (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth

/-- The total cost to paint the floor is 624 Rs. -/
def totalCostCondition (floor : RectangularFloor) : Prop :=
  floor.paintingCost = 624

/-- The painting rate is 4 Rs per square meter. -/
def paintingRateCondition (floor : RectangularFloor) : Prop :=
  floor.paintingRate = 4

/-- The theorem stating that the length of the floor is approximately 21.63 meters. -/
theorem floor_length_approx (floor : RectangularFloor) 
  (h1 : lengthCondition floor)
  (h2 : totalCostCondition floor)
  (h3 : paintingRateCondition floor) :
  ∃ ε > 0, |floor.length - 21.63| < ε :=
sorry

end floor_length_approx_l2624_262493


namespace fifty_percent_relation_l2624_262403

theorem fifty_percent_relation (x y : ℝ) : 
  (0.5 * x = y + 20) → (x - 2 * y = 40) := by
  sorry

end fifty_percent_relation_l2624_262403


namespace journey_time_equation_l2624_262480

theorem journey_time_equation (x : ℝ) (h1 : x > 0) : 
  (240 / x - 240 / (1.5 * x) = 1) ↔ 
  (240 / x = 240 / (1.5 * x) + 1) := by
sorry

end journey_time_equation_l2624_262480


namespace factorization_equality_l2624_262486

theorem factorization_equality (x y : ℝ) : 4 * x^2 * y - 12 * x * y = 4 * x * y * (x - 3) := by
  sorry

end factorization_equality_l2624_262486


namespace scale_division_l2624_262479

/-- Given a scale of length 80 inches divided into equal parts of 20 inches each,
    prove that the number of equal parts is 4. -/
theorem scale_division (scale_length : ℕ) (part_length : ℕ) (h1 : scale_length = 80) (h2 : part_length = 20) :
  scale_length / part_length = 4 := by
  sorry

end scale_division_l2624_262479


namespace lamp_arrangements_count_l2624_262463

/-- The number of ways to select k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to turn off 3 lamps in a row of 10 lamps,
    where the end lamps must remain on and no two consecutive lamps can be off. -/
def lamp_arrangements : ℕ := choose 6 3

theorem lamp_arrangements_count : lamp_arrangements = 20 := by sorry

end lamp_arrangements_count_l2624_262463


namespace consecutive_integers_sum_equality_l2624_262475

theorem consecutive_integers_sum_equality (x : ℤ) (h : x = 25) :
  (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5)) =
  ((x + 6) + (x + 7) + (x + 8) + (x + 9) + (x + 10)) := by
  sorry

#check consecutive_integers_sum_equality

end consecutive_integers_sum_equality_l2624_262475


namespace pie_eating_contest_l2624_262484

theorem pie_eating_contest (erik_pie frank_pie : ℚ) : 
  erik_pie = 0.6666666666666666 →
  erik_pie = frank_pie + 0.3333333333333333 →
  frank_pie = 0.3333333333333333 := by
  sorry

end pie_eating_contest_l2624_262484


namespace square_difference_of_quadratic_solutions_l2624_262460

theorem square_difference_of_quadratic_solutions : 
  ∀ Φ φ : ℝ, Φ ≠ φ → Φ^2 = Φ + 2 → φ^2 = φ + 2 → (Φ - φ)^2 = 9 := by
  sorry

end square_difference_of_quadratic_solutions_l2624_262460


namespace at_least_one_less_than_two_l2624_262467

theorem at_least_one_less_than_two (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b > 2) : 
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := by
  sorry

end at_least_one_less_than_two_l2624_262467


namespace apple_price_reduction_l2624_262469

-- Define the given conditions
def reduced_price_per_dozen : ℚ := 3
def total_money : ℚ := 40
def additional_apples : ℕ := 64

-- Define the function to calculate the percentage reduction
def calculate_percentage_reduction (original_price reduced_price : ℚ) : ℚ :=
  ((original_price - reduced_price) / original_price) * 100

-- State the theorem
theorem apple_price_reduction :
  let dozens_at_reduced_price := total_money / reduced_price_per_dozen
  let additional_dozens := additional_apples / 12
  let dozens_at_original_price := dozens_at_reduced_price - additional_dozens
  let original_price_per_dozen := total_money / dozens_at_original_price
  calculate_percentage_reduction original_price_per_dozen reduced_price_per_dozen = 40 := by
  sorry

end apple_price_reduction_l2624_262469


namespace coordinate_point_A_coordinate_point_B_l2624_262408

-- Definition of a coordinate point
def is_coordinate_point (x y : ℝ) : Prop := 2 * x - y = 1

-- Part 1
theorem coordinate_point_A : 
  ∀ a : ℝ, is_coordinate_point 3 a ↔ a = 5 :=
sorry

-- Part 2
theorem coordinate_point_B :
  ∀ b c : ℕ, (is_coordinate_point (b + c) (b + 5) ∧ b > 0 ∧ c > 0) ↔ 
  ((b = 2 ∧ c = 2) ∨ (b = 4 ∧ c = 1)) :=
sorry

end coordinate_point_A_coordinate_point_B_l2624_262408


namespace unique_m_existence_l2624_262465

theorem unique_m_existence : ∃! m : ℤ,
  50 ≤ m ∧ m ≤ 180 ∧
  m % 9 = 0 ∧
  m % 10 = 7 ∧
  m % 7 = 5 ∧
  m = 117 := by
  sorry

end unique_m_existence_l2624_262465


namespace range_of_a_l2624_262483

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) → -2 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l2624_262483


namespace horner_method_v₄_l2624_262476

-- Define the polynomial coefficients
def a₀ : ℝ := 12
def a₁ : ℝ := 35
def a₂ : ℝ := -8
def a₃ : ℝ := 79
def a₄ : ℝ := 6
def a₅ : ℝ := 5
def a₆ : ℝ := 3

-- Define x
def x : ℝ := -4

-- Define v₄ using Horner's method
def v₄ : ℝ := ((((a₆ * x + a₅) * x + a₄) * x + a₃) * x + a₂) * x + a₁

-- Theorem statement
theorem horner_method_v₄ : v₄ = 220 := by
  sorry

end horner_method_v₄_l2624_262476


namespace exhibition_arrangement_l2624_262415

/-- The number of display stands -/
def n : ℕ := 9

/-- The number of exhibits -/
def k : ℕ := 3

/-- The number of ways to arrange k distinct objects in n positions,
    where the objects cannot be placed at the ends or adjacent to each other -/
def arrangement_count (n k : ℕ) : ℕ :=
  if n < 2 * k + 1 then 0
  else (n - k - 1).choose k * k.factorial

theorem exhibition_arrangement :
  arrangement_count n k = 60 := by sorry

end exhibition_arrangement_l2624_262415


namespace crayons_given_to_friends_l2624_262461

theorem crayons_given_to_friends (crayons_lost : ℕ) (total_crayons_lost_or_given : ℕ) 
  (h1 : crayons_lost = 535)
  (h2 : total_crayons_lost_or_given = 587) :
  total_crayons_lost_or_given - crayons_lost = 52 := by
  sorry

end crayons_given_to_friends_l2624_262461


namespace sector_max_area_l2624_262418

/-- Given a sector with circumference 30, its area is maximized when the radius is 15/2 and the central angle is 2. -/
theorem sector_max_area (R α : ℝ) : 
  R + R + (α * R) = 30 →  -- circumference condition
  (∀ R' α' : ℝ, R' + R' + (α' * R') = 30 → 
    (1/2) * α * R^2 ≥ (1/2) * α' * R'^2) →  -- area is maximized
  R = 15/2 ∧ α = 2 := by
sorry


end sector_max_area_l2624_262418


namespace society_coleaders_selection_l2624_262450

theorem society_coleaders_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 2) :
  Nat.choose n k = 190 := by
  sorry

end society_coleaders_selection_l2624_262450


namespace floor_sqrt_sum_l2624_262440

theorem floor_sqrt_sum (a b c : ℝ) : 
  |a| = 4 → b^2 = 9 → c^3 = -8 → a > b → b > c → 
  ⌊Real.sqrt (a + b + c)⌋ = 2 := by sorry

end floor_sqrt_sum_l2624_262440


namespace max_ab_value_l2624_262470

theorem max_ab_value (a b c : ℝ) (h1 : a + b + c = 4) (h2 : 3*a + 2*b - c = 0) :
  ∀ x y : ℝ, x + y + c = 4 → 3*x + 2*y - c = 0 → x*y ≤ a*b ∧ a*b = 1/3 :=
sorry

end max_ab_value_l2624_262470


namespace triangle_t_range_l2624_262445

theorem triangle_t_range (a b c : ℝ) (A B C : ℝ) (t : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a * c = (1/4) * b^2 →
  Real.sin A + Real.sin C = t * Real.sin B →
  0 < B → B < Real.pi/2 →
  ∃ (t_min t_max : ℝ), t_min = Real.sqrt 6 / 2 ∧ t_max = Real.sqrt 2 ∧ t_min < t ∧ t < t_max :=
by sorry

end triangle_t_range_l2624_262445


namespace sum_of_roots_l2624_262441

theorem sum_of_roots (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ∀ x : ℝ, x^2 - a*x + 3*b = 0 ↔ x = a ∨ x = b) : 
  a + b = a :=
sorry

end sum_of_roots_l2624_262441


namespace tileB_smallest_unique_p_l2624_262443

/-- Represents a rectangular tile with four labeled sides -/
structure Tile where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The set of all tiles -/
def tiles : Finset Tile := sorry

/-- Tile A -/
def tileA : Tile := { p := 5, q := 2, r := 8, s := 11 }

/-- Tile B -/
def tileB : Tile := { p := 2, q := 1, r := 4, s := 7 }

/-- Tile C -/
def tileC : Tile := { p := 4, q := 9, r := 6, s := 3 }

/-- Tile D -/
def tileD : Tile := { p := 10, q := 6, r := 5, s := 9 }

/-- Tile E -/
def tileE : Tile := { p := 11, q := 3, r := 7, s := 0 }

/-- Function to check if a value is unique among all tiles -/
def isUnique (t : Tile) (f : Tile → ℤ) : Prop :=
  ∀ t' ∈ tiles, t' ≠ t → f t' ≠ f t

/-- Theorem: Tile B has the smallest unique p value -/
theorem tileB_smallest_unique_p :
  isUnique tileB Tile.p ∧ 
  ∀ t ∈ tiles, isUnique t Tile.p → tileB.p ≤ t.p :=
sorry

end tileB_smallest_unique_p_l2624_262443


namespace sixth_student_stickers_l2624_262482

def sticker_sequence (n : ℕ) : ℕ :=
  29 + 6 * (n - 1)

theorem sixth_student_stickers : sticker_sequence 6 = 59 := by
  sorry

end sixth_student_stickers_l2624_262482


namespace sum_divisibility_l2624_262431

theorem sum_divisibility : 
  let y := 72 + 144 + 216 + 288 + 576 + 720 + 4608
  (∃ k : ℤ, y = 6 * k) ∧ 
  (∃ k : ℤ, y = 12 * k) ∧ 
  (∃ k : ℤ, y = 24 * k) ∧ 
  ¬(∃ k : ℤ, y = 48 * k) := by
  sorry

end sum_divisibility_l2624_262431


namespace planes_parallel_if_perpendicular_to_same_line_l2624_262451

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- Define the non-overlapping property for planes
variable (non_overlapping : Plane → Plane → Prop)

-- Theorem statement
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular m α) 
  (h2 : perpendicular m β) 
  (h3 : non_overlapping α β) : 
  parallel α β :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l2624_262451


namespace x_value_equality_l2624_262434

def F (x y z : ℝ) : ℝ := x * y^3 + z^2

theorem x_value_equality : ∃ x : ℝ, F x 3 2 = F x 2 5 ∧ x = 21/19 := by
  sorry

end x_value_equality_l2624_262434


namespace first_digit_of_5_to_n_l2624_262457

theorem first_digit_of_5_to_n (n : ℕ) : 
  (∃ k : ℕ, 7 * 10^k ≤ 2^n ∧ 2^n < 8 * 10^k) → 
  (∃ m : ℕ, 10^m ≤ 5^n ∧ 5^n < 2 * 10^m) :=
by sorry

end first_digit_of_5_to_n_l2624_262457


namespace rachels_homework_l2624_262490

/-- Rachel's homework problem -/
theorem rachels_homework (reading_homework : ℕ) (math_homework : ℕ) : 
  reading_homework = 2 → math_homework = reading_homework + 7 → math_homework = 9 :=
by
  sorry


end rachels_homework_l2624_262490


namespace total_raisins_l2624_262421

theorem total_raisins (yellow_raisins : ℝ) (black_raisins : ℝ) (red_raisins : ℝ)
  (h1 : yellow_raisins = 0.3)
  (h2 : black_raisins = 0.4)
  (h3 : red_raisins = 0.5) :
  yellow_raisins + black_raisins + red_raisins = 1.2 := by
  sorry

end total_raisins_l2624_262421


namespace inequality_equivalence_l2624_262478

theorem inequality_equivalence (x : ℝ) : 
  (5 / 24 + |x - 11 / 48| < 5 / 16) ↔ (1 / 8 < x ∧ x < 1 / 3) := by
  sorry

end inequality_equivalence_l2624_262478


namespace linear_function_not_in_quadrant_III_l2624_262494

/-- A linear function defined by y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents the four quadrants of a 2D coordinate system -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Checks if a point (x, y) is in Quadrant III -/
def isInQuadrantIII (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

/-- Theorem: The linear function y = -2x + 1 does not pass through Quadrant III -/
theorem linear_function_not_in_quadrant_III :
  ∀ x y : ℝ, y = -2 * x + 1 → ¬(isInQuadrantIII x y) := by
  sorry

end linear_function_not_in_quadrant_III_l2624_262494


namespace gary_stickers_l2624_262448

theorem gary_stickers (initial_stickers : ℕ) : 
  (initial_stickers : ℚ) * (2/3) * (3/4) = 36 → initial_stickers = 72 := by
  sorry

end gary_stickers_l2624_262448


namespace mailing_cost_formula_l2624_262477

/-- The cost function for mailing a package -/
noncomputable def mailing_cost (W : ℝ) : ℝ := 8 * ⌈W / 2⌉

/-- Theorem stating the correct formula for the mailing cost -/
theorem mailing_cost_formula (W : ℝ) : 
  mailing_cost W = 8 * ⌈W / 2⌉ := by
  sorry

#check mailing_cost_formula

end mailing_cost_formula_l2624_262477


namespace no_integer_solutions_l2624_262412

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^2 + 3*x*y - 2*y^2 = 122 := by
  sorry

end no_integer_solutions_l2624_262412


namespace rihannas_initial_money_l2624_262420

/-- Represents the shopping scenario and calculates Rihanna's initial money --/
def rihannas_shopping (mango_price apple_juice_price : ℕ) (mango_quantity apple_juice_quantity : ℕ) (money_left : ℕ) : ℕ :=
  let total_cost := mango_price * mango_quantity + apple_juice_price * apple_juice_quantity
  total_cost + money_left

/-- Theorem stating that Rihanna's initial money was $50 --/
theorem rihannas_initial_money :
  rihannas_shopping 3 3 6 6 14 = 50 := by
  sorry

end rihannas_initial_money_l2624_262420


namespace inequality_theorem_l2624_262401

theorem inequality_theorem (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 := by
  sorry

end inequality_theorem_l2624_262401


namespace polynomial_remainder_l2624_262459

/-- Given a polynomial Q(x) where Q(17) = 41 and Q(93) = 13, 
    the remainder when Q(x) is divided by (x - 17)(x - 93) is -7/19*x + 900/19 -/
theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 17 = 41) (h2 : Q 93 = 13) :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 17) * (x - 93) * R x + (-7/19 * x + 900/19) :=
sorry

end polynomial_remainder_l2624_262459


namespace difference_of_squares_l2624_262499

theorem difference_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) := by
  sorry

end difference_of_squares_l2624_262499
