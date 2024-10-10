import Mathlib

namespace mask_usage_duration_l3697_369774

theorem mask_usage_duration (total_masks : ℕ) (family_members : ℕ) (total_days : ℕ) 
  (h1 : total_masks = 100)
  (h2 : family_members = 5)
  (h3 : total_days = 80) :
  (total_masks : ℚ) / total_days / family_members = 1 / 4 := by
  sorry

#check mask_usage_duration

end mask_usage_duration_l3697_369774


namespace sam_win_probability_l3697_369766

/-- The probability of hitting the target with one shot -/
def hit_prob : ℚ := 2/5

/-- The probability of missing the target with one shot -/
def miss_prob : ℚ := 3/5

/-- The probability that Sam wins the game -/
def win_prob : ℚ := 5/8

theorem sam_win_probability :
  (hit_prob + miss_prob * miss_prob * win_prob = win_prob) →
  win_prob = 5/8 := by sorry

end sam_win_probability_l3697_369766


namespace roots_of_quadratic_l3697_369777

theorem roots_of_quadratic (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end roots_of_quadratic_l3697_369777


namespace double_inequality_solution_l3697_369723

theorem double_inequality_solution (x : ℝ) : 
  -2 < (x^2 - 16*x + 11) / (x^2 - 3*x + 4) ∧ 
  (x^2 - 16*x + 11) / (x^2 - 3*x + 4) < 2 ↔ 
  1 < x ∧ x < 3 :=
sorry

end double_inequality_solution_l3697_369723


namespace mary_age_proof_l3697_369788

/-- Mary's current age -/
def mary_age : ℕ := 2

/-- Jay's current age -/
def jay_age : ℕ := mary_age + 7

theorem mary_age_proof :
  (∃ (j m : ℕ),
    j - 5 = (m - 5) + 7 ∧
    j + 5 = 2 * (m + 5) ∧
    m = mary_age) :=
by sorry

end mary_age_proof_l3697_369788


namespace student_b_score_l3697_369712

-- Define the scoring function
def calculateScore (totalQuestions : ℕ) (correctResponses : ℕ) : ℕ :=
  let incorrectResponses := totalQuestions - correctResponses
  correctResponses - 2 * incorrectResponses

-- Theorem statement
theorem student_b_score :
  calculateScore 100 91 = 73 := by
  sorry

end student_b_score_l3697_369712


namespace inequality_proof_l3697_369775

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_condition : x * y + y * z + z * x = x + y + z) : 
  1 / (x^2 + y + 1) + 1 / (y^2 + z + 1) + 1 / (z^2 + x + 1) ≤ 1 ∧ 
  (1 / (x^2 + y + 1) + 1 / (y^2 + z + 1) + 1 / (z^2 + x + 1) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end inequality_proof_l3697_369775


namespace exam_mean_score_l3697_369754

theorem exam_mean_score (q σ : ℝ) 
  (h1 : 58 = q - 2 * σ) 
  (h2 : 98 = q + 3 * σ) : 
  q = 74 := by sorry

end exam_mean_score_l3697_369754


namespace average_of_numbers_l3697_369797

def numbers : List ℝ := [2, 3, 4, 7, 9]

theorem average_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 5 := by
  sorry

end average_of_numbers_l3697_369797


namespace can_collection_difference_l3697_369785

/-- Theorem: Difference in can collection between two days -/
theorem can_collection_difference
  (sarah_yesterday : ℝ)
  (lara_yesterday : ℝ)
  (alex_yesterday : ℝ)
  (sarah_today : ℝ)
  (lara_today : ℝ)
  (alex_today : ℝ)
  (h1 : sarah_yesterday = 50.5)
  (h2 : lara_yesterday = sarah_yesterday + 30.3)
  (h3 : alex_yesterday = 90.2)
  (h4 : sarah_today = 40.7)
  (h5 : lara_today = 70.5)
  (h6 : alex_today = 55.3) :
  (sarah_yesterday + lara_yesterday + alex_yesterday) -
  (sarah_today + lara_today + alex_today) = 55 := by
  sorry

end can_collection_difference_l3697_369785


namespace table_and_chair_price_l3697_369789

/-- The price of a chair in dollars -/
def chair_price : ℝ := by sorry

/-- The price of a table in dollars -/
def table_price : ℝ := 52.5

/-- The relation between chair and table prices -/
axiom price_relation : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

theorem table_and_chair_price : table_price + chair_price = 60 := by sorry

end table_and_chair_price_l3697_369789


namespace sum_of_roots_l3697_369736

theorem sum_of_roots (x : ℝ) : 
  (∃ y z : ℝ, (3*x + 4)*(x - 5) + (3*x + 4)*(x - 7) = 0 ∧ x = y ∨ x = z) → 
  (∃ y z : ℝ, (3*x + 4)*(x - 5) + (3*x + 4)*(x - 7) = 0 ∧ x = y ∨ x = z ∧ y + z = 14/3) :=
by sorry

end sum_of_roots_l3697_369736


namespace min_marks_group_a_l3697_369732

/-- Represents the number of marks for each question in a group -/
structure GroupMarks where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the number of questions in each group -/
structure GroupQuestions where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The examination setup -/
structure Examination where
  marks : GroupMarks
  questions : GroupQuestions
  total_questions : ℕ
  total_marks : ℕ

/-- Conditions for the examination -/
def valid_examination (e : Examination) : Prop :=
  e.total_questions = 100 ∧
  e.questions.a + e.questions.b + e.questions.c = e.total_questions ∧
  e.questions.b = 23 ∧
  e.questions.c = 1 ∧
  e.marks.b = 2 ∧
  e.marks.c = 3 ∧
  e.total_marks = e.questions.a * e.marks.a + e.questions.b * e.marks.b + e.questions.c * e.marks.c ∧
  e.questions.a * e.marks.a ≥ (60 * e.total_marks) / 100

theorem min_marks_group_a (e : Examination) (h : valid_examination e) :
  e.marks.a ≥ 1 :=
sorry

end min_marks_group_a_l3697_369732


namespace tangent_parallel_range_l3697_369784

open Real

/-- The function f(x) = x(m - e^(-2x)) --/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * (m - Real.exp (-2 * x))

/-- The derivative of f with respect to x --/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := m - (1 - 2*x) * Real.exp (-2 * x)

/-- Theorem stating the range of m for which there exist two distinct points
    on the curve y = f(x) where the tangent lines are parallel to y = x --/
theorem tangent_parallel_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv m x₁ = 1 ∧ f_deriv m x₂ = 1) ↔ 
  (1 - Real.exp (-2) < m ∧ m < 1) :=
sorry

end tangent_parallel_range_l3697_369784


namespace workday_meetings_percentage_l3697_369796

def workday_hours : ℕ := 10
def minutes_per_hour : ℕ := 60
def first_meeting_duration : ℕ := 60
def second_meeting_duration : ℕ := 2 * first_meeting_duration
def third_meeting_duration : ℕ := first_meeting_duration / 2

def total_workday_minutes : ℕ := workday_hours * minutes_per_hour
def total_meeting_minutes : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

theorem workday_meetings_percentage :
  (total_meeting_minutes : ℚ) / (total_workday_minutes : ℚ) * 100 = 35 := by
  sorry

end workday_meetings_percentage_l3697_369796


namespace simplify_fraction_product_l3697_369707

theorem simplify_fraction_product : 5 * (12 / 7) * (49 / -60) = -7 := by sorry

end simplify_fraction_product_l3697_369707


namespace log_problem_l3697_369705

-- Define the logarithm function for base 3
noncomputable def log3 (y : ℝ) : ℝ := Real.log y / Real.log 3

-- Define the logarithm function for base 9
noncomputable def log9 (y : ℝ) : ℝ := Real.log y / Real.log 9

theorem log_problem (x : ℝ) (h : log3 (x + 1) = 4) : log9 x = 2 := by
  sorry

end log_problem_l3697_369705


namespace yellow_yellow_pairs_count_l3697_369761

/-- Represents the student pairing scenario in a math contest --/
structure ContestPairing where
  total_students : ℕ
  blue_students : ℕ
  yellow_students : ℕ
  total_pairs : ℕ
  blue_blue_pairs : ℕ

/-- The specific contest pairing scenario from the problem --/
def mathContest : ContestPairing := {
  total_students := 144
  blue_students := 63
  yellow_students := 81
  total_pairs := 72
  blue_blue_pairs := 27
}

/-- Theorem stating that the number of yellow-yellow pairs is 36 --/
theorem yellow_yellow_pairs_count (contest : ContestPairing) 
  (h1 : contest.total_students = contest.blue_students + contest.yellow_students)
  (h2 : contest.total_pairs * 2 = contest.total_students)
  (h3 : contest = mathContest) : 
  contest.yellow_students - (contest.total_pairs - contest.blue_blue_pairs - 
  (contest.blue_students - 2 * contest.blue_blue_pairs)) = 36 := by
  sorry

#check yellow_yellow_pairs_count

end yellow_yellow_pairs_count_l3697_369761


namespace option_b_more_cost_effective_l3697_369730

/-- Cost function for Option A -/
def cost_a (x : ℝ) : ℝ := 60 + 18 * x

/-- Cost function for Option B -/
def cost_b (x : ℝ) : ℝ := 150 + 15 * x

/-- Theorem stating that Option B is more cost-effective for 40 kg of blueberries -/
theorem option_b_more_cost_effective :
  cost_b 40 < cost_a 40 := by sorry

end option_b_more_cost_effective_l3697_369730


namespace special_quadratic_roots_nonnegative_l3697_369743

/-- A quadratic polynomial with two distinct roots satisfying f(x^2 + y^2) ≥ f(2xy) for all x and y -/
structure SpecialQuadratic where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  has_two_distinct_roots : ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ f r₁ = 0 ∧ f r₂ = 0
  special_property : ∀ x y : ℝ, f (x^2 + y^2) ≥ f (2*x*y)

/-- The roots of a SpecialQuadratic are non-negative -/
theorem special_quadratic_roots_nonnegative (sq : SpecialQuadratic) :
  ∃ r₁ r₂ : ℝ, r₁ ≥ 0 ∧ r₂ ≥ 0 ∧ sq.f r₁ = 0 ∧ sq.f r₂ = 0 := by
  sorry

end special_quadratic_roots_nonnegative_l3697_369743


namespace robin_gum_pieces_l3697_369737

/-- The number of gum packages Robin has -/
def num_packages : ℕ := 135

/-- The number of pieces in each package of gum -/
def pieces_per_package : ℕ := 46

/-- The total number of gum pieces Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_pieces : total_pieces = 6210 := by
  sorry

end robin_gum_pieces_l3697_369737


namespace parabola_translation_theorem_l3697_369770

/-- Represents a parabola in the form y = ax² + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a 2D translation --/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The original parabola y = 8x² --/
def original_parabola : Parabola := { a := 8, b := 0, c := 0 }

/-- The translation of 3 units left and 5 units down --/
def translation : Translation := { dx := -3, dy := -5 }

/-- Applies a translation to a parabola --/
def apply_translation (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a
    b := -2 * p.a * t.dx
    c := p.a * t.dx^2 + p.b * t.dx + p.c + t.dy }

theorem parabola_translation_theorem :
  apply_translation original_parabola translation = { a := 8, b := 48, c := -5 } := by
  sorry

end parabola_translation_theorem_l3697_369770


namespace quadratic_real_roots_l3697_369781

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
sorry

end quadratic_real_roots_l3697_369781


namespace expression_simplification_l3697_369740

theorem expression_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (x + a)^3 / ((a - b) * (a - c)) + (x + b)^3 / ((b - a) * (b - c)) + (x + c)^3 / ((c - a) * (c - b)) = a + b + c - 3*x :=
by sorry

end expression_simplification_l3697_369740


namespace equation_solution_l3697_369708

theorem equation_solution : 
  ∀ x : ℝ, (Real.sqrt (x + 16) - 8 / Real.sqrt (x + 16) = 4) ↔ 
  (x = 20 + 8 * Real.sqrt 3 ∨ x = 20 - 8 * Real.sqrt 3) :=
by sorry

end equation_solution_l3697_369708


namespace max_area_between_parabolas_l3697_369711

/-- The parabola C_a -/
def C_a (a x : ℝ) : ℝ := -2 * x^2 + 4 * a * x - 2 * a^2 + a + 1

/-- The parabola C -/
def C (x : ℝ) : ℝ := x^2 - 2 * x

/-- The difference function between C and C_a -/
def f_a (a x : ℝ) : ℝ := C x - C_a a x

/-- Theorem: The maximum area enclosed by parabolas C_a and C is 27/(4√2) -/
theorem max_area_between_parabolas :
  ∃ (a : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f_a a x₁ = 0 ∧ f_a a x₂ = 0) →
  (∫ (x : ℝ) in Set.Icc (min x₁ x₂) (max x₁ x₂), f_a a x) ≤ 27 / (4 * Real.sqrt 2) :=
sorry

end max_area_between_parabolas_l3697_369711


namespace square_flag_side_length_l3697_369703

theorem square_flag_side_length (total_fabric : ℝ) (square_flags : ℕ) (wide_flags : ℕ) (tall_flags : ℕ) (remaining_fabric : ℝ) :
  total_fabric = 1000 ∧ 
  square_flags = 16 ∧ 
  wide_flags = 20 ∧ 
  tall_flags = 10 ∧ 
  remaining_fabric = 294 →
  ∃ (side_length : ℝ),
    side_length = 4 ∧
    side_length^2 * square_flags + 15 * (wide_flags + tall_flags) = total_fabric - remaining_fabric :=
by sorry

end square_flag_side_length_l3697_369703


namespace min_correct_answers_for_environmental_quiz_l3697_369733

/-- Represents a quiz with scoring rules -/
structure Quiz where
  totalQuestions : ℕ
  correctScore : ℕ
  incorrectDeduction : ℕ

/-- Calculates the score for a given number of correct answers -/
def calculateScore (quiz : Quiz) (correctAnswers : ℕ) : ℤ :=
  (quiz.correctScore * correctAnswers : ℤ) - 
  (quiz.incorrectDeduction * (quiz.totalQuestions - correctAnswers) : ℤ)

/-- The minimum number of correct answers needed to exceed the target score -/
def minCorrectAnswers (quiz : Quiz) (targetScore : ℤ) : ℕ :=
  quiz.totalQuestions.succ

theorem min_correct_answers_for_environmental_quiz :
  let quiz : Quiz := ⟨30, 10, 5⟩
  let targetScore : ℤ := 90
  minCorrectAnswers quiz targetScore = 17 ∧
  ∀ (x : ℕ), x ≥ minCorrectAnswers quiz targetScore → calculateScore quiz x > targetScore :=
by sorry

end min_correct_answers_for_environmental_quiz_l3697_369733


namespace tangent_line_minimum_value_l3697_369715

theorem tangent_line_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : ∃ x y : ℝ, y = x - 2*a ∧ y = Real.log (x + b) ∧ 
    (Real.exp y) * (1 / (x + b)) = 1) :
  (1/a + 2/b) ≥ 8 :=
sorry

end tangent_line_minimum_value_l3697_369715


namespace power_product_evaluation_l3697_369738

theorem power_product_evaluation :
  let a : ℕ := 3
  a^2 * a^5 = 2187 :=
by sorry

end power_product_evaluation_l3697_369738


namespace linear_coefficient_of_example_quadratic_l3697_369752

/-- Given a quadratic equation ax² + bx + c = 0, returns the coefficient of the linear term (b) -/
def linearCoefficient (a b c : ℚ) : ℚ := b

theorem linear_coefficient_of_example_quadratic :
  linearCoefficient 2 3 (-4) = 3 := by sorry

end linear_coefficient_of_example_quadratic_l3697_369752


namespace dilution_proof_l3697_369744

/-- Proves that adding 7 ounces of water to 12 ounces of a 40% alcohol solution results in a 25% alcohol solution -/
theorem dilution_proof (original_volume : ℝ) (original_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : 
  original_volume = 12 →
  original_concentration = 0.4 →
  target_concentration = 0.25 →
  water_added = 7 →
  (original_volume * original_concentration) / (original_volume + water_added) = target_concentration :=
by
  sorry

end dilution_proof_l3697_369744


namespace expression_evaluation_l3697_369709

theorem expression_evaluation (x y : ℝ) (hx : x = -2) (hy : y = 1/3) :
  (2*y + 3*x^2) - (x^2 - y) - x^2 = 5 := by
  sorry

end expression_evaluation_l3697_369709


namespace jose_share_of_profit_l3697_369704

/-- Calculates the share of profit for an investor given the total profit and investment ratios -/
def calculate_share_of_profit (total_profit : ℚ) (investment_ratio : ℚ) (total_investment_ratio : ℚ) : ℚ :=
  (investment_ratio / total_investment_ratio) * total_profit

theorem jose_share_of_profit (tom_investment : ℚ) (jose_investment : ℚ) 
  (tom_duration : ℚ) (jose_duration : ℚ) (total_profit : ℚ) :
  tom_investment = 3000 →
  jose_investment = 4500 →
  tom_duration = 12 →
  jose_duration = 10 →
  total_profit = 5400 →
  let tom_investment_ratio := tom_investment * tom_duration
  let jose_investment_ratio := jose_investment * jose_duration
  let total_investment_ratio := tom_investment_ratio + jose_investment_ratio
  calculate_share_of_profit total_profit jose_investment_ratio total_investment_ratio = 3000 := by
sorry

end jose_share_of_profit_l3697_369704


namespace committee_seating_arrangements_l3697_369780

/-- The number of distinct arrangements of chairs and stools -/
def distinct_arrangements (n_women : ℕ) (n_men : ℕ) : ℕ :=
  Nat.choose (n_women + n_men - 1) (n_men - 1)

/-- Theorem stating the number of distinct arrangements for the given problem -/
theorem committee_seating_arrangements :
  distinct_arrangements 12 3 = 91 := by
  sorry

end committee_seating_arrangements_l3697_369780


namespace modular_inverse_57_mod_59_l3697_369731

theorem modular_inverse_57_mod_59 : ∃ x : ℕ, x < 59 ∧ (57 * x) % 59 = 1 :=
by
  use 29
  sorry

end modular_inverse_57_mod_59_l3697_369731


namespace next_term_correct_l3697_369750

/-- Represents a digit (0-9) -/
inductive Digit : Type
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Represents a sequence of digits -/
def Sequence := List Digit

/-- Generates the next term in the sequence based on the current term -/
def nextTerm (current : Sequence) : Sequence :=
  sorry

/-- The starting term of the sequence -/
def startTerm : Sequence :=
  [Digit.one]

/-- Generates the nth term of the sequence -/
def nthTerm (n : Nat) : Sequence :=
  sorry

/-- Converts a Sequence to a list of natural numbers -/
def sequenceToNatList (s : Sequence) : List Nat :=
  sorry

theorem next_term_correct :
  sequenceToNatList (nextTerm [Digit.one, Digit.one, Digit.four, Digit.two, Digit.one, Digit.three]) =
  [3, 1, 1, 2, 1, 3, 1, 4] :=
sorry

end next_term_correct_l3697_369750


namespace veronica_extra_stairs_l3697_369760

/-- Given that Samir climbed 318 stairs and together with Veronica they climbed 495 stairs,
    prove that Veronica climbed 18 stairs more than half of Samir's amount. -/
theorem veronica_extra_stairs (samir_stairs : ℕ) (total_stairs : ℕ) 
    (h1 : samir_stairs = 318)
    (h2 : total_stairs = 495)
    (h3 : ∃ (veronica_stairs : ℕ), veronica_stairs > samir_stairs / 2 ∧ 
                                    veronica_stairs + samir_stairs = total_stairs) : 
  ∃ (veronica_stairs : ℕ), veronica_stairs = samir_stairs / 2 + 18 := by
  sorry

end veronica_extra_stairs_l3697_369760


namespace shelf_capacity_l3697_369769

/-- The number of CDs that a single rack can hold -/
def cds_per_rack : ℕ := 8

/-- The number of racks that can fit on a shelf -/
def racks_per_shelf : ℕ := 4

/-- The total number of CDs that can fit on a shelf -/
def total_cds : ℕ := cds_per_rack * racks_per_shelf

theorem shelf_capacity : total_cds = 32 := by
  sorry

end shelf_capacity_l3697_369769


namespace square_difference_pattern_l3697_369787

theorem square_difference_pattern (n : ℕ) (h : n ≥ 1) :
  (n + 2)^2 - n^2 = 4 * (n + 1) := by
  sorry

end square_difference_pattern_l3697_369787


namespace sum_product_equality_l3697_369716

theorem sum_product_equality : 3 * 12 + 3 * 13 + 3 * 16 + 11 = 134 := by
  sorry

end sum_product_equality_l3697_369716


namespace function_composition_equality_l3697_369759

theorem function_composition_equality (a b c d : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x + b
  let g : ℝ → ℝ := λ x ↦ c * x^2 + d
  (∃ x : ℝ, f (g x) = g (f x)) ↔ (c = 0 ∨ a * b = 0) ∧ a * d = c * b^2 + d - b :=
by sorry

end function_composition_equality_l3697_369759


namespace mike_remaining_cards_l3697_369793

/-- Calculates the number of baseball cards Mike has after Sam's purchase -/
def remaining_cards (initial : ℕ) (bought : ℕ) : ℕ :=
  initial - bought

/-- Theorem stating that Mike has 74 baseball cards after Sam's purchase -/
theorem mike_remaining_cards :
  remaining_cards 87 13 = 74 := by
  sorry

end mike_remaining_cards_l3697_369793


namespace newton_interpolation_polynomial_l3697_369735

/-- The interpolation polynomial -/
def P (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

/-- The given points -/
def x₀ : ℝ := 2
def x₁ : ℝ := 4
def x₂ : ℝ := 5

/-- The given function values -/
def y₀ : ℝ := 1
def y₁ : ℝ := 15
def y₂ : ℝ := 28

theorem newton_interpolation_polynomial :
  P x₀ = y₀ ∧ P x₁ = y₁ ∧ P x₂ = y₂ ∧
  ∀ Q : ℝ → ℝ, (Q x₀ = y₀ ∧ Q x₁ = y₁ ∧ Q x₂ = y₂) →
  (∃ a b c : ℝ, ∀ x, Q x = a * x^2 + b * x + c) →
  (∀ x, Q x = P x) :=
sorry

end newton_interpolation_polynomial_l3697_369735


namespace intersection_A_complement_B_l3697_369734

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {4, 5}

theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 3} := by sorry

end intersection_A_complement_B_l3697_369734


namespace quadratic_vertex_coordinates_l3697_369700

/-- The vertex coordinates of the quadratic function y = 2x^2 - 4x + 5 are (1, 3) -/
theorem quadratic_vertex_coordinates :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 - 4 * x + 5
  ∃ (h k : ℝ), h = 1 ∧ k = 3 ∧ 
    (∀ x : ℝ, f x = 2 * (x - h)^2 + k) ∧
    (∀ x : ℝ, f x ≥ k) :=
by sorry

end quadratic_vertex_coordinates_l3697_369700


namespace ratio_sum_equality_l3697_369771

theorem ratio_sum_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 16)
  (sum_xyz : x^2 + y^2 + z^2 = 49)
  (sum_prod : a*x + b*y + c*z = 28) :
  (a + b + c) / (x + y + z) = 4/7 := by
sorry

end ratio_sum_equality_l3697_369771


namespace geometric_sequence_common_ratio_range_l3697_369776

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio_range 
  (a : ℕ → ℝ) (q : ℝ) (h1 : is_geometric_sequence a) 
  (h2 : a 1 * (a 2 + a 3) = 6 * a 1 - 9) :
  (-1 - Real.sqrt 5) / 2 ≤ q ∧ q ≤ (-1 + Real.sqrt 5) / 2 ∧ q ≠ 0 :=
sorry

end geometric_sequence_common_ratio_range_l3697_369776


namespace simplify_radical_sum_l3697_369782

theorem simplify_radical_sum : 
  Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2) = 2 * Real.sqrt 6 := by
  sorry

end simplify_radical_sum_l3697_369782


namespace min_ratio_four_digit_number_l3697_369729

/-- A structure representing a four-digit number with distinct digits -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_nonzero : a ≠ 0
  distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  digits_range : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

/-- The value of a four-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- The sum of digits of a four-digit number -/
def digit_sum (n : FourDigitNumber) : Nat :=
  n.a + n.b + n.c + n.d

/-- The ratio of a four-digit number to the sum of its digits -/
def ratio (n : FourDigitNumber) : Rat :=
  (value n : Rat) / (digit_sum n : Rat)

theorem min_ratio_four_digit_number :
  ∃ (n : FourDigitNumber), 
    (∀ (m : FourDigitNumber), ratio n ≤ ratio m) ∧ 
    (ratio n = 60.5) ∧
    (value n = 1089) := by
  sorry

end min_ratio_four_digit_number_l3697_369729


namespace cos_15_cos_45_minus_cos_75_sin_45_l3697_369791

theorem cos_15_cos_45_minus_cos_75_sin_45 :
  Real.cos (15 * π / 180) * Real.cos (45 * π / 180) -
  Real.cos (75 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end cos_15_cos_45_minus_cos_75_sin_45_l3697_369791


namespace quadratic_equation_solution_l3697_369722

theorem quadratic_equation_solution :
  ∃ (a b : ℕ+),
    (∃ (x : ℝ), x^2 + 8*x = 48 ∧ x > 0 ∧ x = Real.sqrt a - b) ∧
    a + b = 68 := by
sorry

end quadratic_equation_solution_l3697_369722


namespace min_toothpicks_for_five_squares_l3697_369701

/-- A square formed by toothpicks -/
structure ToothpickSquare where
  side_length : ℝ
  toothpicks_per_square : ℕ

/-- The arrangement of multiple toothpick squares -/
structure SquareArrangement where
  square : ToothpickSquare
  num_squares : ℕ

/-- The number of toothpicks needed for an arrangement of squares -/
def toothpicks_needed (arrangement : SquareArrangement) : ℕ :=
  sorry

/-- The theorem stating the minimum number of toothpicks needed -/
theorem min_toothpicks_for_five_squares
  (square : ToothpickSquare)
  (arrangement : SquareArrangement)
  (h1 : square.side_length = 6)
  (h2 : square.toothpicks_per_square = 4)
  (h3 : arrangement.square = square)
  (h4 : arrangement.num_squares = 5) :
  toothpicks_needed arrangement = 15 :=
sorry

end min_toothpicks_for_five_squares_l3697_369701


namespace expanded_ohara_triple_solution_l3697_369767

/-- An Expanded O'Hara triple is a tuple of four positive integers (a, b, c, x) 
    such that √a + √b + √c = x -/
def IsExpandedOHaraTriple (a b c x : ℕ) : Prop :=
  Real.sqrt a + Real.sqrt b + Real.sqrt c = x

theorem expanded_ohara_triple_solution :
  IsExpandedOHaraTriple 49 64 16 19 := by sorry

end expanded_ohara_triple_solution_l3697_369767


namespace regular_polygon_sides_l3697_369721

/-- The number of sides of a regular polygon where the difference between 
    the number of diagonals and the number of sides is 7. -/
def polygon_sides : ℕ := 7

/-- The number of diagonals in a polygon with n sides. -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_polygon_sides :
  ∃ (n : ℕ), n > 0 ∧ num_diagonals n - n = 7 → n = polygon_sides :=
by sorry

end regular_polygon_sides_l3697_369721


namespace paint_six_boards_time_l3697_369794

/-- The minimum time required to paint both sides of wooden boards. -/
def paint_time (num_boards : ℕ) (paint_time_per_side : ℕ) (drying_time : ℕ) : ℕ :=
  2 * num_boards * paint_time_per_side

theorem paint_six_boards_time :
  paint_time 6 1 5 = 12 :=
by sorry

end paint_six_boards_time_l3697_369794


namespace roots_problem_l3697_369773

theorem roots_problem :
  (∀ x : ℝ, x > 0 → x^2 = 1/16 → x = 1/4) ∧
  (∀ x : ℝ, x^2 = 9 ↔ x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, x^3 = -8 → x = -2) := by
sorry

end roots_problem_l3697_369773


namespace arithmetic_sequence_iff_c_eq_neg_one_l3697_369763

/-- Definition of the sum of the first n terms of the sequence -/
def S (n : ℕ) (c : ℝ) : ℝ := (n + 1)^2 + c

/-- Definition of the nth term of the sequence -/
def a (n : ℕ) (c : ℝ) : ℝ := S n c - S (n - 1) c

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: The sequence is arithmetic if and only if c = -1 -/
theorem arithmetic_sequence_iff_c_eq_neg_one (c : ℝ) :
  is_arithmetic_sequence (a · c) ↔ c = -1 := by sorry

end arithmetic_sequence_iff_c_eq_neg_one_l3697_369763


namespace hexagon_perimeter_is_42_l3697_369741

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The length of ribbon required for each side of the display board -/
def ribbon_length_per_side : ℝ := 7

/-- The perimeter of a hexagonal display board -/
def hexagon_perimeter : ℝ := hexagon_sides * ribbon_length_per_side

/-- Theorem: The perimeter of the hexagonal display board is 42 cm -/
theorem hexagon_perimeter_is_42 : hexagon_perimeter = 42 := by
  sorry

end hexagon_perimeter_is_42_l3697_369741


namespace beijing_shanghai_train_time_l3697_369717

/-- The function relationship between total travel time and average speed for a train on the Beijing-Shanghai railway line -/
theorem beijing_shanghai_train_time (t : ℝ) (v : ℝ) (h : v ≠ 0) : 
  (t = 1463 / v) ↔ (1463 = t * v) :=
by sorry

end beijing_shanghai_train_time_l3697_369717


namespace tan_theta_value_l3697_369764

theorem tan_theta_value (θ : Real) 
  (h : (Real.sin (π - θ) + Real.cos (θ - 2*π)) / (Real.sin θ + Real.cos (π + θ)) = 1/2) : 
  Real.tan θ = -3 := by
sorry

end tan_theta_value_l3697_369764


namespace intersection_A_complement_B_l3697_369792

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x < 0}

-- Define set B
def B : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_A_complement_B_l3697_369792


namespace isosceles_triangle_ef_length_l3697_369748

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  /-- The length of two equal sides -/
  side_length : ℝ
  /-- The ratio of EG to GF -/
  eg_gf_ratio : ℝ
  /-- The side length is positive -/
  side_length_pos : 0 < side_length
  /-- The ratio of EG to GF is 2 -/
  eg_gf_ratio_is_two : eg_gf_ratio = 2

/-- The theorem stating the length of EF in the isosceles triangle -/
theorem isosceles_triangle_ef_length (t : IsoscelesTriangle) (h : t.side_length = 10) :
  ∃ (ef : ℝ), ef = 6 * Real.sqrt 5 := by
  sorry

end isosceles_triangle_ef_length_l3697_369748


namespace integral_tan_sin_l3697_369706

open Real MeasureTheory

theorem integral_tan_sin : ∫ (x : ℝ) in Real.arcsin (2 / Real.sqrt 5)..Real.arcsin (3 / Real.sqrt 10), 
  (2 * Real.tan x + 5) / ((5 - Real.tan x) * Real.sin (2 * x)) = 2 * Real.log (3 / 2) := by sorry

end integral_tan_sin_l3697_369706


namespace circle_intersections_l3697_369768

-- Define the circle based on the given diameter endpoints
def circle_from_diameter (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  let center := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let radius := ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt / 2
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the given circle
def given_circle : Set (ℝ × ℝ) := circle_from_diameter (2, 10) (14, 2)

-- Theorem statement
theorem circle_intersections :
  -- The x-coordinates of the intersections with the x-axis are 4 and 12
  (∃ (x : ℝ), (x, 0) ∈ given_circle ↔ x = 4 ∨ x = 12) ∧
  -- There are no intersections with the y-axis
  (∀ (y : ℝ), (0, y) ∉ given_circle) := by
  sorry

end circle_intersections_l3697_369768


namespace negative_number_with_abs_two_l3697_369720

theorem negative_number_with_abs_two (a : ℝ) (h1 : a < 0) (h2 : |a| = 2) : a = -2 := by
  sorry

end negative_number_with_abs_two_l3697_369720


namespace unique_solution_floor_equation_l3697_369742

theorem unique_solution_floor_equation :
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 4⌋ - ⌊(n : ℚ) / 2⌋^2 = 5 ∧ n = 11 := by
  sorry

end unique_solution_floor_equation_l3697_369742


namespace nail_polish_count_l3697_369719

theorem nail_polish_count (num_girls : ℕ) (nails_per_girl : ℕ) : 
  num_girls = 5 → nails_per_girl = 20 → num_girls * nails_per_girl = 100 := by
  sorry

end nail_polish_count_l3697_369719


namespace initial_men_count_l3697_369725

/-- The initial number of men in a group where:
  1. The average age increases by 3 years when two women replace two men.
  2. The two men being replaced are 18 and 22 years old.
  3. The average age of the women is 30.5 years. -/
def initial_number_of_men : ℕ := 7

/-- The average age increase when women replace men -/
def age_increase : ℝ := 3

/-- The age of the first man being replaced -/
def first_man_age : ℕ := 18

/-- The age of the second man being replaced -/
def second_man_age : ℕ := 22

/-- The average age of the women -/
def women_average_age : ℝ := 30.5

theorem initial_men_count : 
  ∃ (A : ℝ), 
    (initial_number_of_men : ℝ) * (A + age_increase) = 
    initial_number_of_men * A - (first_man_age + second_man_age : ℝ) + 2 * women_average_age :=
by sorry

end initial_men_count_l3697_369725


namespace quadratic_prime_roots_unique_l3697_369713

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem quadratic_prime_roots_unique :
  ∃! k : ℤ, ∃ p q : ℕ,
    is_prime p ∧ is_prime q ∧
    p ≠ q ∧
    ∀ x : ℤ, x^2 - 74*x + k = 0 ↔ x = p ∨ x = q :=
sorry

end quadratic_prime_roots_unique_l3697_369713


namespace existence_of_infinite_set_with_gcd_property_l3697_369726

theorem existence_of_infinite_set_with_gcd_property :
  ∃ (S : Set ℕ), Set.Infinite S ∧
  (∀ (x y z w : ℕ), x ∈ S → y ∈ S → z ∈ S → w ∈ S →
    x < y → z < w → (x, y) ≠ (z, w) →
    Nat.gcd (x * y + 2022) (z * w + 2022) = 1) :=
sorry

end existence_of_infinite_set_with_gcd_property_l3697_369726


namespace meeting_attendance_l3697_369790

/-- The number of people attending a meeting where each person receives two copies of a contract --/
def number_of_people (pages_per_contract : ℕ) (copies_per_person : ℕ) (total_pages_copied : ℕ) : ℕ :=
  total_pages_copied / (pages_per_contract * copies_per_person)

/-- Theorem stating that the number of people in the meeting is 9 --/
theorem meeting_attendance : number_of_people 20 2 360 = 9 := by
  sorry

end meeting_attendance_l3697_369790


namespace range_of_a_l3697_369751

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ |x^2 - 2*x|}

-- Define set B
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + a ≤ 0}

-- Theorem statement
theorem range_of_a (a : ℝ) : A ∩ B a = B a → a ∈ Set.Icc 0 1 := by
  sorry

end range_of_a_l3697_369751


namespace floor_equation_solution_l3697_369747

-- Define the floor function
def floor (x : ℚ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_equation_solution :
  ∀ x : ℚ, floor (5 * x - 2) = 3 * x.num + x.den → x = 5 / 3 := by
  sorry

end floor_equation_solution_l3697_369747


namespace total_gray_trees_l3697_369746

/-- Represents a rectangle with trees -/
structure TreeRectangle where
  totalTrees : ℕ
  whiteTrees : ℕ
  grayTrees : ℕ
  sum_eq : totalTrees = whiteTrees + grayTrees

/-- The problem setup -/
def dronePhotos (rect1 rect2 rect3 : TreeRectangle) : Prop :=
  rect1.totalTrees = rect2.totalTrees ∧
  rect1.totalTrees = rect3.totalTrees ∧
  rect1.totalTrees = 100 ∧
  rect1.whiteTrees = 82 ∧
  rect2.whiteTrees = 82

/-- The theorem to prove -/
theorem total_gray_trees (rect1 rect2 rect3 : TreeRectangle) 
  (h : dronePhotos rect1 rect2 rect3) : 
  rect1.grayTrees + rect2.grayTrees = 26 :=
by sorry

end total_gray_trees_l3697_369746


namespace trig_expression_equals_four_l3697_369779

theorem trig_expression_equals_four : 
  (1 / Real.sin (10 * π / 180)) - (Real.sqrt 3 / Real.cos (10 * π / 180)) = 4 := by
  sorry

end trig_expression_equals_four_l3697_369779


namespace total_population_proof_l3697_369783

def springfield_population : ℕ := 482653
def population_difference : ℕ := 119666

def greenville_population : ℕ := springfield_population - population_difference
def oakville_population : ℕ := 2 * population_difference

def total_population : ℕ := springfield_population + greenville_population + oakville_population

theorem total_population_proof : total_population = 1084972 := by
  sorry

end total_population_proof_l3697_369783


namespace least_positive_integer_for_zero_sums_l3697_369762

theorem least_positive_integer_for_zero_sums (x₁ x₂ x₃ x₄ x₅ : ℝ) : 
  (∃ (S : Finset (Fin 5 × Fin 5 × Fin 5)), 
    S.card = 7 ∧ 
    (∀ (p q r : Fin 5), (p, q, r) ∈ S → p < q ∧ q < r) ∧
    (∀ (p q r : Fin 5), (p, q, r) ∈ S → x₁ * p.val + x₂ * q.val + x₃ * r.val = 0) →
    x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∧
  (∀ n : ℕ, n < 7 → 
    ∃ (x₁' x₂' x₃' x₄' x₅' : ℝ), 
      ∃ (S : Finset (Fin 5 × Fin 5 × Fin 5)),
        S.card = n ∧
        (∀ (p q r : Fin 5), (p, q, r) ∈ S → p < q ∧ q < r) ∧
        (∀ (p q r : Fin 5), (p, q, r) ∈ S → 
          x₁' * p.val + x₂' * q.val + x₃' * r.val = 0) ∧
        ¬(x₁' = 0 ∧ x₂' = 0 ∧ x₃' = 0 ∧ x₄' = 0 ∧ x₅' = 0)) := by
  sorry


end least_positive_integer_for_zero_sums_l3697_369762


namespace sqrt_of_square_root_three_plus_one_squared_l3697_369795

theorem sqrt_of_square_root_three_plus_one_squared :
  Real.sqrt ((Real.sqrt 3 + 1) ^ 2) = Real.sqrt 3 + 1 := by
  sorry

end sqrt_of_square_root_three_plus_one_squared_l3697_369795


namespace product_of_numbers_with_given_sum_and_difference_l3697_369745

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 90 ∧ x - y = 10 → x * y = 2000 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l3697_369745


namespace circumscribed_circle_radius_right_triangle_l3697_369755

/-- The radius of the circumscribed circle of a right triangle with sides 10, 8, and 6 is 5 -/
theorem circumscribed_circle_radius_right_triangle : 
  ∀ (a b c r : ℝ), 
  a = 10 → b = 8 → c = 6 → 
  a^2 = b^2 + c^2 → 
  r = a / 2 → 
  r = 5 := by sorry

end circumscribed_circle_radius_right_triangle_l3697_369755


namespace min_value_sum_reciprocals_l3697_369714

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 2) : 
  1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 9 / 4 := by
sorry

end min_value_sum_reciprocals_l3697_369714


namespace solution_set_when_a_is_one_range_of_a_when_f_leq_one_l3697_369728

-- Define the function f
def f (x a : ℝ) : ℝ := 5 - |x + a| - |x - 2|

-- Part 1
theorem solution_set_when_a_is_one :
  let a := 1
  {x : ℝ | f x a ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} :=
sorry

-- Part 2
theorem range_of_a_when_f_leq_one :
  {a : ℝ | ∀ x, f x a ≤ 1} = {a : ℝ | -6 ≤ a ∧ a ≤ 2} :=
sorry

end solution_set_when_a_is_one_range_of_a_when_f_leq_one_l3697_369728


namespace complex_equation_sum_l3697_369772

theorem complex_equation_sum (a b : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a + i) * i = b + i) : a + b = 0 := by
  sorry

end complex_equation_sum_l3697_369772


namespace sophie_widget_production_l3697_369756

/-- Sophie's widget production problem -/
theorem sophie_widget_production 
  (w t : ℕ) -- w: widgets per hour, t: hours worked on Wednesday
  (h1 : w = 3 * t) -- condition that w = 3t
  : w * t - (w + 5) * (t - 3) = 4 * t + 15 := by
  sorry

end sophie_widget_production_l3697_369756


namespace diet_soda_count_l3697_369739

/-- The number of diet soda bottles in a grocery store -/
def diet_soda : ℕ := sorry

/-- The number of regular soda bottles in the grocery store -/
def regular_soda : ℕ := 60

/-- The difference between regular and diet soda bottles -/
def difference : ℕ := 41

theorem diet_soda_count : diet_soda = 19 :=
  by
  have h1 : regular_soda = diet_soda + difference := sorry
  sorry

end diet_soda_count_l3697_369739


namespace complex_number_location_l3697_369727

theorem complex_number_location :
  let z : ℂ := 1 / (3 + Complex.I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end complex_number_location_l3697_369727


namespace isosceles_triangle_third_side_l3697_369718

theorem isosceles_triangle_third_side 
  (a b c : ℝ) 
  (h_isosceles : (a = b ∧ c = 5) ∨ (a = c ∧ b = 5) ∨ (b = c ∧ a = 5)) 
  (h_side : a = 2 ∨ b = 2 ∨ c = 2) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a = 5 ∨ b = 5 ∨ c = 5 :=
sorry

end isosceles_triangle_third_side_l3697_369718


namespace noodle_shop_solution_l3697_369778

/-- Represents the prices and sales of noodles in a shop -/
structure NoodleShop where
  dine_in_price : ℚ
  fresh_price : ℚ
  april_dine_in_sales : ℕ
  april_fresh_sales : ℕ
  may_fresh_price_decrease : ℚ
  may_fresh_sales_increase : ℚ
  may_total_sales_increase : ℚ

/-- Theorem stating the solution to the noodle shop problem -/
theorem noodle_shop_solution (shop : NoodleShop) : 
  (3 * shop.dine_in_price + 2 * shop.fresh_price = 31) →
  (4 * shop.dine_in_price + shop.fresh_price = 33) →
  (shop.april_dine_in_sales = 2500) →
  (shop.april_fresh_sales = 1500) →
  (shop.may_fresh_price_decrease = 3/4 * shop.may_total_sales_increase) →
  (shop.may_fresh_sales_increase = 5/2 * shop.may_total_sales_increase) →
  (shop.dine_in_price = 7) ∧ 
  (shop.fresh_price = 5) ∧ 
  (shop.may_total_sales_increase = 40/9) := by
  sorry


end noodle_shop_solution_l3697_369778


namespace jims_out_of_pocket_l3697_369765

/-- The cost of Jim's first wedding ring in dollars -/
def first_ring_cost : ℕ := 10000

/-- The cost of Jim's wife's ring in dollars -/
def second_ring_cost : ℕ := 2 * first_ring_cost

/-- The selling price of Jim's first ring in dollars -/
def first_ring_selling_price : ℕ := first_ring_cost / 2

/-- Jim's total out-of-pocket expense in dollars -/
def total_out_of_pocket : ℕ := second_ring_cost + (first_ring_cost - first_ring_selling_price)

/-- Theorem stating Jim's total out-of-pocket expense -/
theorem jims_out_of_pocket : total_out_of_pocket = 25000 := by
  sorry

end jims_out_of_pocket_l3697_369765


namespace opposite_of_negative_fraction_opposite_of_negative_one_over_2023_l3697_369753

theorem opposite_of_negative_fraction (n : ℕ) (n_pos : n > 0) :
  ((-1 : ℚ) / n) + (1 : ℚ) / n = 0 :=
by sorry

theorem opposite_of_negative_one_over_2023 :
  ((-1 : ℚ) / 2023) + (1 : ℚ) / 2023 = 0 :=
by sorry

end opposite_of_negative_fraction_opposite_of_negative_one_over_2023_l3697_369753


namespace line_equation_proof_l3697_369749

/-- Given a line defined by (-1, 4) · ((x, y) - (3, -5)) = 0, 
    prove that its equation in the form y = mx + b has m = 1/4 and b = -23/4 -/
theorem line_equation_proof (x y : ℝ) : 
  (-1 : ℝ) * (x - 3) + 4 * (y + 5) = 0 → 
  ∃ (m b : ℝ), y = m * x + b ∧ m = (1 : ℝ) / 4 ∧ b = -(23 : ℝ) / 4 := by
  sorry

end line_equation_proof_l3697_369749


namespace smallest_third_term_is_negative_one_l3697_369724

/-- Given an arithmetic progression with first term 7, adding 3 to the second term
    and 15 to the third term results in a geometric progression. This function
    represents the smallest possible value for the third term of the resulting
    geometric progression. -/
def smallest_third_term_geometric : ℝ := sorry

/-- Theorem stating that the smallest possible value for the third term
    of the resulting geometric progression is -1. -/
theorem smallest_third_term_is_negative_one :
  smallest_third_term_geometric = -1 := by sorry

end smallest_third_term_is_negative_one_l3697_369724


namespace eighth_grade_percentage_combined_schools_combined_schools_eighth_grade_percentage_l3697_369798

theorem eighth_grade_percentage_combined_schools : ℝ → Prop :=
  fun p =>
    let pinecrest_total : ℕ := 160
    let mapleridge_total : ℕ := 250
    let pinecrest_eighth_percent : ℝ := 18
    let mapleridge_eighth_percent : ℝ := 22
    let pinecrest_eighth : ℝ := (pinecrest_eighth_percent / 100) * pinecrest_total
    let mapleridge_eighth : ℝ := (mapleridge_eighth_percent / 100) * mapleridge_total
    let total_eighth : ℝ := pinecrest_eighth + mapleridge_eighth
    let total_students : ℝ := pinecrest_total + mapleridge_total
    p = (total_eighth / total_students) * 100 ∧ p = 20

/-- The percentage of 8th grade students in both schools combined is 20%. -/
theorem combined_schools_eighth_grade_percentage :
  ∃ p, eighth_grade_percentage_combined_schools p :=
sorry

end eighth_grade_percentage_combined_schools_combined_schools_eighth_grade_percentage_l3697_369798


namespace money_left_l3697_369758

def initial_amount : ℕ := 43
def total_spent : ℕ := 38

theorem money_left : initial_amount - total_spent = 5 := by
  sorry

end money_left_l3697_369758


namespace parabola_vertex_l3697_369786

/-- The equation of a parabola is y^2 + 8y + 2x + 1 = 0. 
    This theorem proves that the vertex of the parabola is (7.5, -4). -/
theorem parabola_vertex (x y : ℝ) : 
  (y^2 + 8*y + 2*x + 1 = 0) → (x = 7.5 ∧ y = -4) := by
  sorry

end parabola_vertex_l3697_369786


namespace pond_problem_l3697_369702

theorem pond_problem (initial_fish : ℕ) (fish_caught : ℕ) : 
  initial_fish = 50 →
  fish_caught = 7 →
  (initial_fish * 3 / 2) - (initial_fish - fish_caught) = 32 := by
  sorry

end pond_problem_l3697_369702


namespace minimum_matches_theorem_l3697_369710

/-- Represents the number of points for each match result -/
structure PointSystem where
  win : Nat
  draw : Nat
  loss : Nat

/-- Represents the state of a team in the competition -/
structure TeamState where
  gamesPlayed : Nat
  points : Nat

/-- Represents the requirements for the team -/
structure TeamRequirement where
  targetPoints : Nat
  minWinsNeeded : Nat

def minimumTotalMatches (initialState : TeamState) (pointSystem : PointSystem) (requirement : TeamRequirement) : Nat :=
  initialState.gamesPlayed + requirement.minWinsNeeded +
    ((requirement.targetPoints - initialState.points - requirement.minWinsNeeded * pointSystem.win + pointSystem.draw - 1) / pointSystem.draw)

theorem minimum_matches_theorem (initialState : TeamState) (pointSystem : PointSystem) (requirement : TeamRequirement) :
  initialState.gamesPlayed = 5 ∧
  initialState.points = 14 ∧
  pointSystem.win = 3 ∧
  pointSystem.draw = 1 ∧
  pointSystem.loss = 0 ∧
  requirement.targetPoints = 40 ∧
  requirement.minWinsNeeded = 6 →
  minimumTotalMatches initialState pointSystem requirement = 13 := by
  sorry

end minimum_matches_theorem_l3697_369710


namespace square_sum_of_product_and_sum_l3697_369757

theorem square_sum_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 12) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 40 := by
sorry

end square_sum_of_product_and_sum_l3697_369757


namespace workers_count_l3697_369799

/-- Given a group of workers who collectively contribute 300,000 and would contribute 350,000 if each gave 50 more, prove that there are 1000 workers. -/
theorem workers_count (total : ℕ) (extra_total : ℕ) (extra_per_worker : ℕ) : 
  total = 300000 →
  extra_total = 350000 →
  extra_per_worker = 50 →
  ∃ (num_workers : ℕ), num_workers * (total / num_workers + extra_per_worker) = extra_total ∧ 
                        num_workers = 1000 := by
  sorry

end workers_count_l3697_369799
