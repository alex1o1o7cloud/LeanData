import Mathlib

namespace NUMINAMATH_CALUDE_johns_beef_purchase_l1236_123630

/-- Given that John uses all but 1 pound of beef in soup, uses twice as many pounds of vegetables 
    as beef, and uses 6 pounds of vegetables, prove that John bought 4 pounds of beef. -/
theorem johns_beef_purchase (beef_used : ℝ) (vegetables_used : ℝ) (beef_leftover : ℝ) : 
  beef_leftover = 1 →
  vegetables_used = 2 * beef_used →
  vegetables_used = 6 →
  beef_used + beef_leftover = 4 := by
  sorry

end NUMINAMATH_CALUDE_johns_beef_purchase_l1236_123630


namespace NUMINAMATH_CALUDE_inequality_proof_l1236_123603

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) :
  a^2 + b^2 + 1/a^2 + b/a ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1236_123603


namespace NUMINAMATH_CALUDE_power_of_five_l1236_123689

theorem power_of_five (m : ℕ) : 5^m = 5 * 25^2 * 125^3 → m = 14 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_l1236_123689


namespace NUMINAMATH_CALUDE_son_age_problem_l1236_123657

theorem son_age_problem (father_age son_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 := by
sorry

end NUMINAMATH_CALUDE_son_age_problem_l1236_123657


namespace NUMINAMATH_CALUDE_playground_area_l1236_123647

theorem playground_area (w l : ℚ) (h1 : l = 2 * w + 30) (h2 : 2 * (l + w) = 700) : w * l = 233600 / 9 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_l1236_123647


namespace NUMINAMATH_CALUDE_set_equality_implies_m_equals_negative_one_l1236_123675

theorem set_equality_implies_m_equals_negative_one (m : ℝ) :
  let A : Set ℝ := {m, 2}
  let B : Set ℝ := {m^2 - 2, 2}
  A = B → m = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_m_equals_negative_one_l1236_123675


namespace NUMINAMATH_CALUDE_mk_97_check_one_l1236_123638

theorem mk_97_check_one (x : ℝ) : x = 1 ↔ x ≠ 2 * x ∧ ∃! y : ℝ, y ^ 2 + 2 * x * y + x = 0 := by sorry

end NUMINAMATH_CALUDE_mk_97_check_one_l1236_123638


namespace NUMINAMATH_CALUDE_a_in_A_l1236_123619

def A : Set ℝ := {x | x < 2 * Real.sqrt 3}

theorem a_in_A : 2 ∈ A := by
  sorry

end NUMINAMATH_CALUDE_a_in_A_l1236_123619


namespace NUMINAMATH_CALUDE_martin_answered_40_l1236_123674

/-- The number of questions Campbell answered correctly -/
def campbell_correct : ℕ := 35

/-- The number of questions Kelsey answered correctly -/
def kelsey_correct : ℕ := campbell_correct + 8

/-- The number of questions Martin answered correctly -/
def martin_correct : ℕ := kelsey_correct - 3

/-- Theorem stating that Martin answered 40 questions correctly -/
theorem martin_answered_40 : martin_correct = 40 := by
  sorry

end NUMINAMATH_CALUDE_martin_answered_40_l1236_123674


namespace NUMINAMATH_CALUDE_lucas_cycling_speed_l1236_123625

theorem lucas_cycling_speed 
  (philippe_speed : ℝ) 
  (marta_ratio : ℝ) 
  (lucas_ratio : ℝ) 
  (h1 : philippe_speed = 10)
  (h2 : marta_ratio = 3/4)
  (h3 : lucas_ratio = 4/3) : 
  lucas_ratio * (marta_ratio * philippe_speed) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_lucas_cycling_speed_l1236_123625


namespace NUMINAMATH_CALUDE_triangle_inequality_l1236_123660

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  |a/b + b/c + c/a - b/a - c/b - a/c| < 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1236_123660


namespace NUMINAMATH_CALUDE_johns_allowance_l1236_123667

theorem johns_allowance (A : ℝ) : 
  A > 0 →
  (4/15) * A = 0.88 →
  A = 3.30 := by
sorry

end NUMINAMATH_CALUDE_johns_allowance_l1236_123667


namespace NUMINAMATH_CALUDE_fourth_roll_six_prob_l1236_123639

/-- Represents a six-sided die --/
structure Die where
  prob_six : ℚ
  prob_other : ℚ
  sum_probs : prob_six + 5 * prob_other = 1

/-- The fair die --/
def fair_die : Die where
  prob_six := 1/6
  prob_other := 1/6
  sum_probs := by norm_num

/-- The biased die --/
def biased_die : Die where
  prob_six := 3/4
  prob_other := 1/20
  sum_probs := by norm_num

/-- The probability of choosing each die --/
def prob_choose_die : ℚ := 1/2

/-- The number of initial rolls that are sixes --/
def num_initial_sixes : ℕ := 3

/-- Theorem: Given the conditions, the probability of rolling a six on the fourth roll is 2187/982 --/
theorem fourth_roll_six_prob :
  let prob_fair := prob_choose_die * fair_die.prob_six^num_initial_sixes
  let prob_biased := prob_choose_die * biased_die.prob_six^num_initial_sixes
  let total_prob := prob_fair + prob_biased
  let cond_prob_fair := prob_fair / total_prob
  let cond_prob_biased := prob_biased / total_prob
  cond_prob_fair * fair_die.prob_six + cond_prob_biased * biased_die.prob_six = 2187 / 982 := by
  sorry

end NUMINAMATH_CALUDE_fourth_roll_six_prob_l1236_123639


namespace NUMINAMATH_CALUDE_class_test_theorem_l1236_123693

/-- A theorem about a class test where some students didn't take the test -/
theorem class_test_theorem 
  (total_students : ℕ) 
  (answered_q2 : ℕ) 
  (did_not_take : ℕ) 
  (answered_both : ℕ) 
  (h1 : total_students = 30)
  (h2 : answered_q2 = 22)
  (h3 : did_not_take = 5)
  (h4 : answered_both = 22)
  (h5 : answered_both ≤ answered_q2)
  (h6 : did_not_take + answered_q2 ≤ total_students) :
  ∃ (answered_q1 : ℕ), answered_q1 = answered_both ∧ 
    answered_q1 + (answered_q2 - answered_both) + did_not_take ≤ total_students :=
by
  sorry

end NUMINAMATH_CALUDE_class_test_theorem_l1236_123693


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l1236_123698

/-- Given that 45 cows eat 45 bags of husk in 45 days, 
    this theorem proves that 1 cow will eat 1 bag of husk in 45 days. -/
theorem cow_husk_consumption 
  (num_cows : ℕ) (num_bags : ℕ) (num_days : ℕ) 
  (h1 : num_cows = 45) 
  (h2 : num_bags = 45) 
  (h3 : num_days = 45) : 
  1 = 1 → num_days = 45 := by
  sorry

#check cow_husk_consumption

end NUMINAMATH_CALUDE_cow_husk_consumption_l1236_123698


namespace NUMINAMATH_CALUDE_not_all_products_are_effe_l1236_123635

/-- Represents a two-digit number --/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n ≤ 99 }

/-- Represents a number of the form effe where e and f are single digits --/
def EffeNumber := { n : ℕ // ∃ e f : ℕ, e < 10 ∧ f < 10 ∧ n = 1001 * e + 110 * f }

/-- States that not all products of two-digit numbers result in an effe number --/
theorem not_all_products_are_effe : 
  ¬ (∀ a b : TwoDigitNumber, ∃ n : EffeNumber, a.val * b.val = n.val) :=
sorry

end NUMINAMATH_CALUDE_not_all_products_are_effe_l1236_123635


namespace NUMINAMATH_CALUDE_not_always_same_digit_sum_l1236_123624

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem not_always_same_digit_sum :
  ∃ (N M : ℕ), 
    (sum_of_digits (N + M) = sum_of_digits N) ∧
    (∃ (k : ℕ), k > 1 ∧ sum_of_digits (N + k * M) ≠ sum_of_digits N) :=
sorry

end NUMINAMATH_CALUDE_not_always_same_digit_sum_l1236_123624


namespace NUMINAMATH_CALUDE_smallest_b_for_real_root_l1236_123602

theorem smallest_b_for_real_root : 
  ∀ b : ℕ, (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_real_root_l1236_123602


namespace NUMINAMATH_CALUDE_complement_of_union_A_B_l1236_123670

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log x}
def B : Set ℝ := {x | -7 < 2 + 3*x ∧ 2 + 3*x < 5}

-- State the theorem
theorem complement_of_union_A_B :
  (Set.univ : Set ℝ) \ (A ∪ B) = {x | x ≤ -3} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_A_B_l1236_123670


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l1236_123604

/-- Given a cubic function f(x) = (1/3)x³ - x + m with a maximum value of 1,
    prove that its minimum value is -1/3 -/
theorem cubic_function_extrema (f : ℝ → ℝ) (m : ℝ) 
    (h1 : ∀ x, f x = (1/3) * x^3 - x + m) 
    (h2 : ∃ x₀, ∀ x, f x ≤ f x₀ ∧ f x₀ = 1) : 
    ∃ x₁, ∀ x, f x ≥ f x₁ ∧ f x₁ = -(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l1236_123604


namespace NUMINAMATH_CALUDE_sum_properties_l1236_123634

theorem sum_properties (a b : ℤ) (ha : 6 ∣ a) (hb : 9 ∣ b) : 
  (∃ x y : ℤ, a + b = 2*x + 1 ∧ a + b = 2*y) ∧ 
  (∃ z : ℤ, a + b = 6*z + 3) ∧ 
  (∃ w : ℤ, a + b = 9*w + 3) ∧ 
  (∃ v : ℤ, a + b = 9*v) :=
by sorry

end NUMINAMATH_CALUDE_sum_properties_l1236_123634


namespace NUMINAMATH_CALUDE_banana_permutations_l1236_123620

/-- The number of distinct permutations of letters in a word with repeated letters -/
def distinctPermutations (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The number of distinct permutations of the letters in "BANANA" -/
theorem banana_permutations :
  distinctPermutations 6 [3, 2] = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_l1236_123620


namespace NUMINAMATH_CALUDE_max_correct_answers_l1236_123688

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (blank_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) : 
  total_questions = 60 → 
  correct_points = 5 → 
  blank_points = 0 → 
  incorrect_points = -2 → 
  total_score = 150 → 
  (∃ (correct blank incorrect : ℕ), 
    correct + blank + incorrect = total_questions ∧ 
    correct_points * correct + blank_points * blank + incorrect_points * incorrect = total_score ∧ 
    ∀ (other_correct : ℕ), 
      (∃ (other_blank other_incorrect : ℕ), 
        other_correct + other_blank + other_incorrect = total_questions ∧ 
        correct_points * other_correct + blank_points * other_blank + incorrect_points * other_incorrect = total_score) → 
      other_correct ≤ 38) ∧ 
  (∃ (blank incorrect : ℕ), 
    38 + blank + incorrect = total_questions ∧ 
    correct_points * 38 + blank_points * blank + incorrect_points * incorrect = total_score) :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l1236_123688


namespace NUMINAMATH_CALUDE_seventh_power_sum_l1236_123610

theorem seventh_power_sum (α β γ : ℂ)
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 10) :
  α^7 + β^7 + γ^7 = 65.38 := by
  sorry

end NUMINAMATH_CALUDE_seventh_power_sum_l1236_123610


namespace NUMINAMATH_CALUDE_museum_trip_l1236_123616

theorem museum_trip (people_first : ℕ) : 
  (people_first + 
   2 * people_first + 
   (2 * people_first - 6) + 
   (people_first + 9) = 75) → 
  people_first = 12 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_l1236_123616


namespace NUMINAMATH_CALUDE_cos_sin_eq_linear_solution_exists_l1236_123632

theorem cos_sin_eq_linear_solution_exists :
  ∃ x : ℝ, -2/3 ≤ x ∧ x ≤ 2/3 ∧ 
  -3*π/2 ≤ x ∧ x ≤ 3*π/2 ∧
  Real.cos (Real.sin x) = 3*x/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_eq_linear_solution_exists_l1236_123632


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l1236_123611

/-- Given a cyclic quadrilateral ABCD where the ratio of angles A:B:C is 3:4:6, 
    prove that the measure of angle D is 100°. -/
theorem cyclic_quadrilateral_angle (A B C D : ℝ) : 
  A + B + C + D = 360 →  -- Sum of angles in a quadrilateral
  A / 3 = B / 4 →        -- Ratio condition for A and B
  A / 3 = C / 6 →        -- Ratio condition for A and C
  D = 100 := by
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l1236_123611


namespace NUMINAMATH_CALUDE_dannys_physics_marks_l1236_123666

/-- Danny's marks in different subjects and average -/
structure DannyMarks where
  english : ℕ
  mathematics : ℕ
  chemistry : ℕ
  biology : ℕ
  average : ℕ

/-- The theorem stating Danny's marks in Physics -/
theorem dannys_physics_marks (marks : DannyMarks) 
  (h1 : marks.english = 76)
  (h2 : marks.mathematics = 65)
  (h3 : marks.chemistry = 67)
  (h4 : marks.biology = 75)
  (h5 : marks.average = 73)
  (h6 : (marks.english + marks.mathematics + marks.chemistry + marks.biology + marks.average * 5 - (marks.english + marks.mathematics + marks.chemistry + marks.biology)) / 5 = marks.average) :
  marks.average * 5 - (marks.english + marks.mathematics + marks.chemistry + marks.biology) = 82 := by
  sorry


end NUMINAMATH_CALUDE_dannys_physics_marks_l1236_123666


namespace NUMINAMATH_CALUDE_solve_system_l1236_123663

theorem solve_system (a b : ℝ) (h1 : 3 * a + 4 * b = 2) (h2 : a = 2 * b - 3) : 5 * b = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1236_123663


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l1236_123605

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 104) 
  (h2 : x * y = 35) : 
  (x + y ≤ Real.sqrt 174) ∧ (∃ (a b : ℝ), a^2 + b^2 = 104 ∧ a * b = 35 ∧ a + b = Real.sqrt 174) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l1236_123605


namespace NUMINAMATH_CALUDE_fourth_term_is_one_tenth_l1236_123699

def sequence_term (n : ℕ) : ℚ := 2 / (n^2 + n : ℚ)

theorem fourth_term_is_one_tenth : sequence_term 4 = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_one_tenth_l1236_123699


namespace NUMINAMATH_CALUDE_determinant_solution_set_implies_a_value_l1236_123600

-- Define the determinant function
def det (x a : ℝ) : ℝ := a * x + 2

-- Define the inequality
def inequality (x a : ℝ) : Prop := det x a < 6

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | inequality x a}

-- Theorem statement
theorem determinant_solution_set_implies_a_value :
  (∀ x : ℝ, x > -1 ↔ x ∈ solution_set a) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_determinant_solution_set_implies_a_value_l1236_123600


namespace NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l1236_123684

/-- The number of FGH supermarkets in the US -/
def us : ℕ := sorry

/-- The number of FGH supermarkets in Canada -/
def canada : ℕ := sorry

/-- The total number of FGH supermarkets -/
def total : ℕ := 70

/-- All supermarkets are either in the US or Canada -/
axiom sum_equals_total : us + canada = total

/-- There are 14 more FGH supermarkets in the US than in Canada -/
axiom us_minus_canada : us = canada + 14

theorem fgh_supermarkets_in_us : us = 42 := by sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_in_us_l1236_123684


namespace NUMINAMATH_CALUDE_angle_with_double_supplement_l1236_123643

theorem angle_with_double_supplement (α : ℝ) :
  (180 - α = 2 * α) → α = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_double_supplement_l1236_123643


namespace NUMINAMATH_CALUDE_intersection_sum_l1236_123679

/-- Two lines intersect at a point if the point satisfies both line equations -/
def intersect_at (x y a b : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 = (1/3) * p.2 + a ∧ p.2 = (1/3) * p.1 + b

/-- The problem statement -/
theorem intersection_sum (a b : ℝ) :
  intersect_at 3 1 a b (3, 1) → a + b = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l1236_123679


namespace NUMINAMATH_CALUDE_playground_children_count_l1236_123653

theorem playground_children_count (boys girls : ℕ) 
  (h1 : boys = 40) 
  (h2 : girls = 77) : 
  boys + girls = 117 := by
sorry

end NUMINAMATH_CALUDE_playground_children_count_l1236_123653


namespace NUMINAMATH_CALUDE_simplify_expression_l1236_123695

theorem simplify_expression (b c : ℝ) :
  3 * b * (3 * b^3 + 2 * b) - 2 * b^2 + c * (3 * b^2 - c) = 9 * b^4 + 4 * b^2 + 3 * b^2 * c - c^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1236_123695


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1236_123608

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_equation (x y : ℝ) :
  (∃ x₀ y₀, ellipse x₀ y₀) →  -- Ellipse exists
  (∃ x₁ y₁, asymptote x₁ y₁) →  -- Asymptote exists
  (∀ x₂ y₂, hyperbola_C x₂ y₂ ↔ 
    (∃ c : ℝ, c > 0 ∧  -- Foci distance
    (x₂ - c)^2 + y₂^2 = (x₂ + c)^2 + y₂^2 ∧  -- Same foci as ellipse
    (∃ t : ℝ, x₂ = t ∧ y₂ = Real.sqrt 3 * t)))  -- Asymptote condition
  :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1236_123608


namespace NUMINAMATH_CALUDE_ellipse_focus_implies_k_l1236_123683

/-- Represents an ellipse with equation kx^2 + 5y^2 = 5 -/
structure Ellipse (k : ℝ) where
  equation : ∀ x y : ℝ, k * x^2 + 5 * y^2 = 5

/-- A focus of an ellipse -/
def Focus := ℝ × ℝ

/-- Theorem: For the ellipse kx^2 + 5y^2 = 5, if one of its foci is (2, 0), then k = 1 -/
theorem ellipse_focus_implies_k (k : ℝ) (e : Ellipse k) (f : Focus) :
  f = (2, 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_implies_k_l1236_123683


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2023_l1236_123645

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_2023 (seq : ArithmeticSequence)
  (h1 : seq.a 2 + seq.a 7 = seq.a 8 + 1)
  (h2 : ∃ r : ℝ, r ≠ 0 ∧ seq.a 4 = r * seq.a 2 ∧ seq.a 8 = r * seq.a 4) :
  seq.a 2023 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2023_l1236_123645


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l1236_123671

/-- The cubic polynomial whose roots we're interested in -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 5*x + 7

/-- Theorem stating the properties of the cubic polynomial P and its value at 0 -/
theorem cubic_polynomial_property (P : ℝ → ℝ) (a b c : ℝ) 
  (hf : f a = 0 ∧ f b = 0 ∧ f c = 0)
  (hPa : P a = b + c)
  (hPb : P b = c + a)
  (hPc : P c = a + b)
  (hPsum : P (a + b + c) = -16) :
  P 0 = 25 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l1236_123671


namespace NUMINAMATH_CALUDE_circle_area_increase_l1236_123623

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.01 * r
  let old_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - old_area) / old_area = 0.0201 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l1236_123623


namespace NUMINAMATH_CALUDE_triangle_side_range_l1236_123641

theorem triangle_side_range (A B C : Real) (AB AC BC : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Given equation
  (Real.sqrt 3 * Real.sin B - Real.cos B) * (Real.sqrt 3 * Real.sin C - Real.cos C) = 4 * Real.cos B * Real.cos C →
  -- Sum of two sides
  AB + AC = 4 →
  -- Triangle inequality
  AB > 0 ∧ AC > 0 ∧ BC > 0 →
  -- BC satisfies the triangle inequality
  BC < AB + AC ∧ AB < BC + AC ∧ AC < AB + BC →
  -- Conclusion: Range of BC
  2 ≤ BC ∧ BC < 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1236_123641


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1236_123681

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x > Real.sin x) ↔ (∀ x : ℝ, x ≤ Real.sin x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1236_123681


namespace NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1236_123642

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides. -/
def pentagon : ℕ := 5

theorem sum_interior_angles_pentagon :
  sum_interior_angles pentagon = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_pentagon_l1236_123642


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_line_segment_l1236_123672

-- Define the line segment
def line_segment (x y : ℝ) : Prop := x - 2*y + 1 = 0 ∧ -1 ≤ x ∧ x ≤ 3

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_of_line_segment :
  ∀ x y : ℝ, line_segment x y →
  ∃ x' y' : ℝ, perpendicular_bisector x' y' ∧
  (x' = (x + (-1))/2 ∧ y' = (y + 0)/2) ∧
  (2*x' - y' - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_line_segment_l1236_123672


namespace NUMINAMATH_CALUDE_quarter_circle_sum_limit_l1236_123644

theorem quarter_circle_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |n * (π * D / (4 * n)) - (π * D / 4)| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circle_sum_limit_l1236_123644


namespace NUMINAMATH_CALUDE_quiz_max_correct_answers_l1236_123669

theorem quiz_max_correct_answers :
  ∀ (correct blank incorrect : ℕ),
    correct + blank + incorrect = 60 →
    5 * correct - 2 * incorrect = 150 →
    correct ≤ 38 ∧
    ∃ (c b i : ℕ), c + b + i = 60 ∧ 5 * c - 2 * i = 150 ∧ c = 38 := by
  sorry

end NUMINAMATH_CALUDE_quiz_max_correct_answers_l1236_123669


namespace NUMINAMATH_CALUDE_quadratic_ratio_l1236_123658

/-- 
Given a quadratic expression x^2 + 1440x + 1600, which can be written in the form (x + d)^2 + e,
prove that e/d = -718.
-/
theorem quadratic_ratio (d e : ℝ) : 
  (∀ x, x^2 + 1440*x + 1600 = (x + d)^2 + e) → e/d = -718 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l1236_123658


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l1236_123662

theorem geometric_sequence_inequality (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  q ≠ 1 →  -- common ratio is not 1
  a 1 + a 4 > a 2 + a 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l1236_123662


namespace NUMINAMATH_CALUDE_marta_took_ten_books_l1236_123690

/-- The number of books Marta took off the shelf -/
def books_taken (initial_books : ℝ) (remaining_books : ℕ) : ℝ :=
  initial_books - remaining_books

/-- Theorem stating that Marta took 10 books off the shelf -/
theorem marta_took_ten_books : books_taken 38.0 28 = 10 := by
  sorry

end NUMINAMATH_CALUDE_marta_took_ten_books_l1236_123690


namespace NUMINAMATH_CALUDE_fraction_comparison_l1236_123631

theorem fraction_comparison : (2 : ℚ) / 3 - 66666666 / 100000000 = 2 / (3 * 100000000) := by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1236_123631


namespace NUMINAMATH_CALUDE_ladder_length_l1236_123682

theorem ladder_length (initial_distance : ℝ) (pull_distance : ℝ) (slide_distance : ℝ) :
  initial_distance = 15 →
  pull_distance = 9 →
  slide_distance = 13 →
  ∃ (ladder_length : ℝ) (initial_height : ℝ),
    ladder_length ^ 2 = initial_distance ^ 2 + initial_height ^ 2 ∧
    ladder_length ^ 2 = (initial_distance + pull_distance) ^ 2 + (initial_height - slide_distance) ^ 2 ∧
    ladder_length = 25 := by
sorry

end NUMINAMATH_CALUDE_ladder_length_l1236_123682


namespace NUMINAMATH_CALUDE_sum_of_squares_positive_l1236_123637

theorem sum_of_squares_positive (a b c : ℝ) (sum_zero : a + b + c = 0) (prod_neg : a * b * c < 0) :
  a^2 + b^2 > 0 ∧ b^2 + c^2 > 0 ∧ c^2 + a^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_positive_l1236_123637


namespace NUMINAMATH_CALUDE_fish_pond_problem_l1236_123613

/-- Calculates the number of fish in the second catch given the total number of fish in the pond,
    the number of tagged fish, and the number of tagged fish caught in the second catch. -/
def second_catch_size (total_fish : ℕ) (tagged_fish : ℕ) (tagged_caught : ℕ) : ℕ :=
  tagged_fish * total_fish / tagged_caught

/-- Theorem stating that given a pond with approximately 1000 fish, where 40 fish were initially tagged
    and released, and 2 tagged fish were found in a subsequent catch, the number of fish in the
    subsequent catch is 50. -/
theorem fish_pond_problem (total_fish : ℕ) (tagged_fish : ℕ) (tagged_caught : ℕ) :
  total_fish = 1000 → tagged_fish = 40 → tagged_caught = 2 →
  second_catch_size total_fish tagged_fish tagged_caught = 50 := by
  sorry

#eval second_catch_size 1000 40 2

end NUMINAMATH_CALUDE_fish_pond_problem_l1236_123613


namespace NUMINAMATH_CALUDE_sequence_property_l1236_123687

def sequence_sum (n : ℕ) : ℚ := n * (3 * n - 1) / 2

def sequence_term (n : ℕ) : ℚ := 3 * n - 2

theorem sequence_property (m : ℕ) :
  (∀ n, sequence_sum n = n * (3 * n - 1) / 2) →
  (∀ n, sequence_term n = 3 * n - 2) →
  sequence_term 1 * sequence_term m = (sequence_term 4) ^ 2 →
  m = 34 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l1236_123687


namespace NUMINAMATH_CALUDE_tangent_line_minimum_sum_l1236_123626

theorem tangent_line_minimum_sum (m n : ℝ) : 
  m > 0 → 
  n > 0 → 
  (∃ x : ℝ, (1/Real.exp 1) * x + m + 1 = Real.log x - n + 2 ∧ 
             (1/Real.exp 1) = 1/x) → 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 1/a + 1/b ≥ 1/m + 1/n) →
  1/m + 1/n = 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_sum_l1236_123626


namespace NUMINAMATH_CALUDE_range_of_m_l1236_123678

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 2 / x + 1 / y = 1) (h2 : ∀ x y, x > 0 → y > 0 → 2 / x + 1 / y = 1 → x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1236_123678


namespace NUMINAMATH_CALUDE_num_sequences_eq_248832_l1236_123676

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of class sessions per week -/
def sessions_per_week : ℕ := 5

/-- The number of different sequences of students solving problems in one week -/
def num_sequences : ℕ := num_students ^ sessions_per_week

/-- Theorem stating that the number of different sequences of students solving problems in one week is 248,832 -/
theorem num_sequences_eq_248832 : num_sequences = 248832 := by sorry

end NUMINAMATH_CALUDE_num_sequences_eq_248832_l1236_123676


namespace NUMINAMATH_CALUDE_coefficient_equals_k_squared_minus_one_l1236_123677

theorem coefficient_equals_k_squared_minus_one (k : ℝ) (h1 : k > 0) :
  (∃ b : ℝ, (k * b^2 - b)^2 = k^2 * b^4 - 2 * k * b^3 + k^2 * b^2 - b^2) →
  k = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_coefficient_equals_k_squared_minus_one_l1236_123677


namespace NUMINAMATH_CALUDE_obtuse_triangle_count_l1236_123628

/-- A function that checks if a triangle with sides a, b, and c is obtuse -/
def is_obtuse (a b c : ℝ) : Prop :=
  (a^2 > b^2 + c^2) ∨ (b^2 > a^2 + c^2) ∨ (c^2 > a^2 + b^2)

/-- A function that checks if a triangle with sides a, b, and c is valid -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- The main theorem stating that there are exactly 13 positive integer values of k
    for which a triangle with sides 12, 16, and k is obtuse -/
theorem obtuse_triangle_count :
  (∃! (s : Finset ℕ), s.card = 13 ∧ 
    (∀ k, k ∈ s ↔ (is_valid_triangle 12 16 k ∧ is_obtuse 12 16 k))) := by
  sorry

end NUMINAMATH_CALUDE_obtuse_triangle_count_l1236_123628


namespace NUMINAMATH_CALUDE_children_exceed_bridge_capacity_l1236_123686

/-- Proves that the total weight of five children exceeds the bridge capacity by 51.8 kg -/
theorem children_exceed_bridge_capacity 
  (bridge_capacity : ℝ)
  (kelly_weight sam_weight daisy_weight : ℝ)
  (megan_weight : ℝ)
  (mike_weight : ℝ)
  (h1 : bridge_capacity = 130)
  (h2 : kelly_weight = 34)
  (h3 : sam_weight = 40)
  (h4 : daisy_weight = 28)
  (h5 : megan_weight = kelly_weight * 1.1)
  (h6 : mike_weight = megan_weight + 5)
  : kelly_weight + sam_weight + daisy_weight + megan_weight + mike_weight - bridge_capacity = 51.8 :=
by
  sorry


end NUMINAMATH_CALUDE_children_exceed_bridge_capacity_l1236_123686


namespace NUMINAMATH_CALUDE_ratio_problem_l1236_123614

theorem ratio_problem (p q : ℚ) (h : 25 / 7 + (2 * q - p) / (2 * q + p) = 4) : p / q = -1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1236_123614


namespace NUMINAMATH_CALUDE_water_polo_team_selection_l1236_123650

def team_size : ℕ := 15
def starting_players : ℕ := 7
def coach_count : ℕ := 1

theorem water_polo_team_selection :
  (team_size * (team_size - 1) * (Nat.choose (team_size - 2) (starting_players - 2))) = 270270 := by
  sorry

end NUMINAMATH_CALUDE_water_polo_team_selection_l1236_123650


namespace NUMINAMATH_CALUDE_car_travel_time_l1236_123615

/-- Proves that a car traveling at 160 km/h for 800 km takes 5 hours -/
theorem car_travel_time (speed : ℝ) (distance : ℝ) (h1 : speed = 160) (h2 : distance = 800) :
  distance / speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_time_l1236_123615


namespace NUMINAMATH_CALUDE_common_number_in_list_l1236_123696

theorem common_number_in_list (l : List ℝ) 
  (h_length : l.length = 7)
  (h_avg_first : (l.take 4).sum / 4 = 7)
  (h_avg_last : (l.drop 3).sum / 4 = 9)
  (h_avg_all : l.sum / 7 = 8) :
  ∃ x ∈ l.take 4 ∩ l.drop 3, x = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_number_in_list_l1236_123696


namespace NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l1236_123651

-- Define a differentiable function
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for a function to have an extremum at a point
def HasExtremumAt (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- State the theorem
theorem derivative_zero_necessary_not_sufficient :
  (∀ x : ℝ, HasExtremumAt f x → deriv f x = 0) ∧
  ¬(∀ x : ℝ, deriv f x = 0 → HasExtremumAt f x) :=
sorry

end NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l1236_123651


namespace NUMINAMATH_CALUDE_inequality_sqrt_ratios_l1236_123622

theorem inequality_sqrt_ratios (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_sqrt_ratios_l1236_123622


namespace NUMINAMATH_CALUDE_grain_production_theorem_l1236_123636

theorem grain_production_theorem (planned_wheat planned_corn actual_wheat actual_corn : ℝ) :
  planned_wheat + planned_corn = 18 →
  actual_wheat + actual_corn = 20 →
  actual_wheat = planned_wheat * 1.12 →
  actual_corn = planned_corn * 1.10 →
  actual_wheat = 11.2 ∧ actual_corn = 8.8 := by
  sorry

end NUMINAMATH_CALUDE_grain_production_theorem_l1236_123636


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1236_123621

theorem quadratic_roots_sum (α β : ℝ) : 
  (α^2 - 3*α - 1 = 0) → 
  (β^2 - 3*β - 1 = 0) → 
  7 * α^4 + 10 * β^3 = 1093 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1236_123621


namespace NUMINAMATH_CALUDE_identity_function_property_l1236_123627

theorem identity_function_property (f : ℕ → ℕ) : 
  (∀ m n : ℕ, (f m + f n) ∣ (m + n)) → 
  (∀ m : ℕ, f m = m) := by
  sorry

end NUMINAMATH_CALUDE_identity_function_property_l1236_123627


namespace NUMINAMATH_CALUDE_aarons_brothers_l1236_123659

theorem aarons_brothers (bennett_brothers : ℕ) (h1 : bennett_brothers = 6) 
  (h2 : bennett_brothers = 2 * aaron_brothers - 2) : aaron_brothers = 4 := by
  sorry

end NUMINAMATH_CALUDE_aarons_brothers_l1236_123659


namespace NUMINAMATH_CALUDE_number_symmetry_equation_l1236_123691

theorem number_symmetry_equation (a b : ℕ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10 * a + b) * (100 * b + 10 * (a + b) + a) = (100 * a + 10 * (a + b) + b) * (10 * b + a) := by
  sorry

end NUMINAMATH_CALUDE_number_symmetry_equation_l1236_123691


namespace NUMINAMATH_CALUDE_max_set_size_l1236_123656

def is_valid_set (s : Finset Nat) : Prop :=
  s.card > 0 ∧ 10 ∉ s ∧ s.sum (λ x => x^2) = 2500

theorem max_set_size :
  (∃ (s : Finset Nat), is_valid_set s ∧ s.card = 17) ∧
  (∀ (s : Finset Nat), is_valid_set s → s.card ≤ 17) :=
sorry

end NUMINAMATH_CALUDE_max_set_size_l1236_123656


namespace NUMINAMATH_CALUDE_power_last_digit_match_l1236_123661

theorem power_last_digit_match : ∃ (m n : ℕ), 
  100 ≤ 2^m ∧ 2^m < 1000 ∧ 
  100 ≤ 3^n ∧ 3^n < 1000 ∧ 
  2^m % 10 = 3^n % 10 ∧ 
  2^m % 10 = 3 := by
sorry

end NUMINAMATH_CALUDE_power_last_digit_match_l1236_123661


namespace NUMINAMATH_CALUDE_divisibility_by_48_l1236_123655

theorem divisibility_by_48 (a b c : ℕ) 
  (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c)
  (ga : a > 3) (gb : b > 3) (gc : c > 3) : 
  48 ∣ ((a - b) * (b - c) * (c - a)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_48_l1236_123655


namespace NUMINAMATH_CALUDE_complex_fraction_equals_962_l1236_123649

/-- Helper function to represent the factorization of x^4 + 400 --/
def factor (x : ℤ) : ℤ := (x * (x - 10) + 20) * (x * (x + 10) + 20)

/-- The main theorem stating that the given expression equals 962 --/
theorem complex_fraction_equals_962 : 
  (factor 10 * factor 26 * factor 42 * factor 58) / 
  (factor 2 * factor 18 * factor 34 * factor 50) = 962 := by
  sorry


end NUMINAMATH_CALUDE_complex_fraction_equals_962_l1236_123649


namespace NUMINAMATH_CALUDE_johns_notebooks_l1236_123652

theorem johns_notebooks (total_children : Nat) (wife_notebooks_per_child : Nat) (total_notebooks : Nat) :
  total_children = 3 →
  wife_notebooks_per_child = 5 →
  total_notebooks = 21 →
  ∃ (johns_notebooks_per_child : Nat),
    johns_notebooks_per_child * total_children + wife_notebooks_per_child * total_children = total_notebooks ∧
    johns_notebooks_per_child = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_notebooks_l1236_123652


namespace NUMINAMATH_CALUDE_sin_cos_sum_equivalence_l1236_123664

theorem sin_cos_sum_equivalence (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * x + π / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equivalence_l1236_123664


namespace NUMINAMATH_CALUDE_max_unused_cubes_l1236_123654

/-- The side length of the original cube in small cube units -/
def original_side_length : ℕ := 10

/-- The total number of small cubes in the original cube -/
def total_cubes : ℕ := original_side_length ^ 3

/-- The function that calculates the number of small cubes used in a hollow cube of side length x -/
def cubes_used (x : ℕ) : ℕ := 6 * (x - 1) ^ 2 + 2

/-- The side length of the largest possible hollow cube -/
def largest_hollow_side : ℕ := 13

theorem max_unused_cubes :
  ∃ (unused : ℕ), unused = total_cubes - cubes_used largest_hollow_side ∧
  unused = 134 ∧
  ∀ (x : ℕ), x > largest_hollow_side → cubes_used x > total_cubes :=
sorry

end NUMINAMATH_CALUDE_max_unused_cubes_l1236_123654


namespace NUMINAMATH_CALUDE_piecewise_function_sum_l1236_123612

theorem piecewise_function_sum (f : ℝ → ℝ) (a b c : ℤ) : 
  (∀ x > 0, f x = a * x + b) →
  (∀ x < 0, f x = b * x + c) →
  (f 0 = a * b) →
  (f 2 = 7) →
  (f 0 = 1) →
  (f (-2) = -8) →
  a + b + c = 8 := by
sorry

end NUMINAMATH_CALUDE_piecewise_function_sum_l1236_123612


namespace NUMINAMATH_CALUDE_stream_speed_l1236_123617

/-- Given a boat with a speed in still water and its travel time and distance downstream,
    calculate the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  boat_speed = 24 →
  time = 4 →
  distance = 112 →
  distance = (boat_speed + (distance / time - boat_speed)) * time →
  distance / time - boat_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1236_123617


namespace NUMINAMATH_CALUDE_mary_bill_difference_l1236_123692

/-- Represents the candy distribution problem --/
def candy_distribution (total : ℕ) (kate robert mary bill : ℕ) : Prop :=
  total = 20 ∧
  robert = kate + 2 ∧
  mary > bill ∧
  mary = robert + 2 ∧
  kate = bill + 2 ∧
  kate = 4

/-- Theorem stating the difference between Mary's and Bill's candy pieces --/
theorem mary_bill_difference (total kate robert mary bill : ℕ) 
  (h : candy_distribution total kate robert mary bill) : 
  mary - bill = 6 := by sorry

end NUMINAMATH_CALUDE_mary_bill_difference_l1236_123692


namespace NUMINAMATH_CALUDE_unique_function_property_l1236_123640

def last_digit (n : ℕ) : ℕ := n % 10

def is_constant_one (f : ℕ → ℕ) : Prop :=
  ∀ n, f n = 1

theorem unique_function_property (f : ℕ → ℕ) 
  (h1 : ∀ x y, f (x * y) = f x * f y)
  (h2 : f 30 = 1)
  (h3 : ∀ n, last_digit n = 7 → f n = 1) :
  is_constant_one f :=
sorry

end NUMINAMATH_CALUDE_unique_function_property_l1236_123640


namespace NUMINAMATH_CALUDE_valid_numbers_l1236_123680

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (a + 1) * (b + 2) * (c + 3) * (d + 4) = 234

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n → n = 1109 ∨ n = 2009 :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1236_123680


namespace NUMINAMATH_CALUDE_probability_three_even_dice_l1236_123668

def num_dice : ℕ := 6
def sides_per_die : ℕ := 12

theorem probability_three_even_dice :
  let p := (num_dice.choose 3) * (1 / 2) ^ num_dice / 1
  p = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_probability_three_even_dice_l1236_123668


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l1236_123606

/-- Proves the number of cakes sold by a baker given certain conditions -/
theorem baker_cakes_sold (cake_price : ℕ) (pie_price : ℕ) (pies_sold : ℕ) (total_revenue : ℕ) :
  cake_price = 12 →
  pie_price = 7 →
  pies_sold = 126 →
  total_revenue = 6318 →
  ∃ cakes_sold : ℕ, cakes_sold * cake_price + pies_sold * pie_price = total_revenue ∧ cakes_sold = 453 :=
by sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l1236_123606


namespace NUMINAMATH_CALUDE_diamond_six_three_l1236_123685

-- Define the diamond operation
def diamond (a b : ℕ) : ℕ := 4 * a + 2 * b

-- State the theorem
theorem diamond_six_three : diamond 6 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_diamond_six_three_l1236_123685


namespace NUMINAMATH_CALUDE_line_distance_theorem_l1236_123618

/-- The line equation 4x - 3y + c = 0 -/
def line_equation (x y c : ℝ) : Prop := 4 * x - 3 * y + c = 0

/-- The distance function from (1,1) to (a,b) -/
def distance_squared (a b : ℝ) : ℝ := (a - 1)^2 + (b - 1)^2

/-- The theorem stating the relationship between the line and the minimum distance -/
theorem line_distance_theorem (a b c : ℝ) :
  line_equation a b c →
  (∀ x y, line_equation x y c → distance_squared a b ≤ distance_squared x y) →
  distance_squared a b = 4 →
  c = -11 ∨ c = 9 := by sorry

end NUMINAMATH_CALUDE_line_distance_theorem_l1236_123618


namespace NUMINAMATH_CALUDE_omega_sequence_monotone_l1236_123648

def is_omega_sequence (d : ℕ → ℕ) : Prop :=
  (∀ n, (d n + d (n + 2)) / 2 ≤ d (n + 1)) ∧
  (∃ M : ℝ, ∀ n, (d n : ℝ) ≤ M)

theorem omega_sequence_monotone (d : ℕ → ℕ) 
  (h_omega : is_omega_sequence d) :
  ∀ n, d n ≤ d (n + 1) := by
sorry

end NUMINAMATH_CALUDE_omega_sequence_monotone_l1236_123648


namespace NUMINAMATH_CALUDE_exists_71_cubes_l1236_123609

/-- Represents the number of cubes after a series of divisions -/
def num_cubes : ℕ → ℕ
| 0 => 1
| (n + 1) => num_cubes n + 7

/-- Theorem stating that it's possible to obtain 71 cubes through the division process -/
theorem exists_71_cubes : ∃ n : ℕ, num_cubes n = 71 := by
  sorry

end NUMINAMATH_CALUDE_exists_71_cubes_l1236_123609


namespace NUMINAMATH_CALUDE_josh_pencils_left_josh_pencils_left_proof_l1236_123607

/-- Given that Josh initially had 142 pencils and gave away 31 pencils,
    prove that he has 111 pencils left. -/
theorem josh_pencils_left : ℕ → ℕ → ℕ → Prop :=
  fun initial_pencils pencils_given_away pencils_left =>
    initial_pencils = 142 →
    pencils_given_away = 31 →
    pencils_left = initial_pencils - pencils_given_away →
    pencils_left = 111

/-- Proof of the theorem -/
theorem josh_pencils_left_proof : josh_pencils_left 142 31 111 := by
  sorry

end NUMINAMATH_CALUDE_josh_pencils_left_josh_pencils_left_proof_l1236_123607


namespace NUMINAMATH_CALUDE_chinese_multiplication_puzzle_l1236_123646

theorem chinese_multiplication_puzzle : 
  ∃! (a b d e p q r : ℕ), 
    (0 ≤ a ∧ a ≤ 9) ∧ 
    (0 ≤ b ∧ b ≤ 9) ∧ 
    (0 ≤ d ∧ d ≤ 9) ∧ 
    (0 ≤ e ∧ e ≤ 9) ∧ 
    (0 ≤ p ∧ p ≤ 9) ∧ 
    (0 ≤ q ∧ q ≤ 9) ∧ 
    (0 ≤ r ∧ r ≤ 9) ∧ 
    (a ≠ b) ∧ 
    (10 * a + b) * (10 * a + b) = 10000 * d + 1000 * e + 100 * p + 10 * q + r ∧
    (10 * a + b) * (10 * a + b) ≡ (10 * a + b) [MOD 100] ∧
    d = 5 ∧ e = 0 ∧ p = 6 ∧ q = 2 ∧ r = 5 ∧ a = 2 ∧ b = 5 :=
by sorry

end NUMINAMATH_CALUDE_chinese_multiplication_puzzle_l1236_123646


namespace NUMINAMATH_CALUDE_triangle_exradius_theorem_l1236_123629

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  r_a : ℝ
  r_b : ℝ
  r_c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_R : 0 < R
  pos_r_a : 0 < r_a
  pos_r_b : 0 < r_b
  pos_r_c : 0 < r_c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

theorem triangle_exradius_theorem (t : Triangle) (h : 2 * t.R ≤ t.r_a) :
  t.a > t.b ∧ t.a > t.c ∧ 2 * t.R > t.r_b ∧ 2 * t.R > t.r_c := by
  sorry

end NUMINAMATH_CALUDE_triangle_exradius_theorem_l1236_123629


namespace NUMINAMATH_CALUDE_tomato_plants_l1236_123673

theorem tomato_plants (n : ℕ) (sum : ℕ) : 
  n = 12 → sum = 186 → 
  ∃ a d : ℕ, 
    (∀ i : ℕ, i ≤ n → a + (i - 1) * d = sum / n + (2 * i - n - 1) / 2) ∧
    (a + (n - 1) * d = 21) :=
by sorry

end NUMINAMATH_CALUDE_tomato_plants_l1236_123673


namespace NUMINAMATH_CALUDE_some_number_value_l1236_123694

theorem some_number_value (t k some_number : ℝ) 
  (h1 : t = 5 / 9 * (k - some_number))
  (h2 : t = 35)
  (h3 : k = 95) : 
  some_number = 32 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l1236_123694


namespace NUMINAMATH_CALUDE_smallest_factor_l1236_123697

theorem smallest_factor (n : ℕ) : 
  (∀ m : ℕ, m > 0 → 
    (2^4 ∣ 1452 * m) ∧ 
    (3^3 ∣ 1452 * m) ∧ 
    (13^3 ∣ 1452 * m) → 
    m ≥ n) ↔ 
  n = 676 := by
sorry

end NUMINAMATH_CALUDE_smallest_factor_l1236_123697


namespace NUMINAMATH_CALUDE_stimulus_savings_theorem_l1236_123633

def stimulus_distribution (initial_amount : ℚ) : ℚ :=
  let wife_share := initial_amount / 4
  let after_wife := initial_amount - wife_share
  let first_son_share := after_wife * 3 / 8
  let after_first_son := after_wife - first_son_share
  let second_son_share := after_first_son * 25 / 100
  let after_second_son := after_first_son - second_son_share
  let third_son_share := 500
  let after_third_son := after_second_son - third_son_share
  let daughter_share := after_third_son * 15 / 100
  let savings := after_third_son - daughter_share
  savings

theorem stimulus_savings_theorem :
  stimulus_distribution 4000 = 770.3125 := by sorry

end NUMINAMATH_CALUDE_stimulus_savings_theorem_l1236_123633


namespace NUMINAMATH_CALUDE_odd_function_value_l1236_123665

theorem odd_function_value (f : ℝ → ℝ) : 
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x > 0, f x = x^2 + 1/x) →  -- definition of f for x > 0
  f (-1) = -2 := by sorry

end NUMINAMATH_CALUDE_odd_function_value_l1236_123665


namespace NUMINAMATH_CALUDE_blue_notebook_cost_l1236_123601

theorem blue_notebook_cost (total_spent : ℕ) (total_notebooks : ℕ) 
  (red_notebooks : ℕ) (red_price : ℕ) (green_notebooks : ℕ) (green_price : ℕ) :
  total_spent = 37 →
  total_notebooks = 12 →
  red_notebooks = 3 →
  red_price = 4 →
  green_notebooks = 2 →
  green_price = 2 →
  ∃ (blue_notebooks : ℕ) (blue_price : ℕ),
    blue_notebooks = total_notebooks - red_notebooks - green_notebooks ∧
    blue_price = 3 ∧
    total_spent = red_notebooks * red_price + green_notebooks * green_price + blue_notebooks * blue_price :=
by sorry

end NUMINAMATH_CALUDE_blue_notebook_cost_l1236_123601
