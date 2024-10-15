import Mathlib

namespace NUMINAMATH_CALUDE_intersection_implies_x_value_l2558_255852

def A (x : ℝ) : Set ℝ := {9, 2 - x, x^2 + 1}
def B (x : ℝ) : Set ℝ := {1, 2 * x^2}

theorem intersection_implies_x_value :
  ∀ x : ℝ, A x ∩ B x = {2} → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_x_value_l2558_255852


namespace NUMINAMATH_CALUDE_eighth_root_of_3906250000000001_l2558_255855

theorem eighth_root_of_3906250000000001 :
  let n : ℕ := 3906250000000001
  ∃ (m : ℕ), m ^ 8 = n ∧ m = 101 :=
by
  sorry

end NUMINAMATH_CALUDE_eighth_root_of_3906250000000001_l2558_255855


namespace NUMINAMATH_CALUDE_decimal_difference_value_l2558_255898

/-- The value of the repeating decimal 0.727272... -/
def repeating_decimal : ℚ := 8 / 11

/-- The value of the terminating decimal 0.72 -/
def terminating_decimal : ℚ := 72 / 100

/-- The difference between the repeating decimal 0.727272... and the terminating decimal 0.72 -/
def decimal_difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference_value : decimal_difference = 8 / 1100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_value_l2558_255898


namespace NUMINAMATH_CALUDE_total_dress_designs_l2558_255877

/-- The number of available fabric colors -/
def num_colors : ℕ := 5

/-- The number of available patterns -/
def num_patterns : ℕ := 6

/-- The number of available sizes -/
def num_sizes : ℕ := 3

/-- Theorem stating the total number of possible dress designs -/
theorem total_dress_designs : num_colors * num_patterns * num_sizes = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l2558_255877


namespace NUMINAMATH_CALUDE_inequality_solution_l2558_255896

theorem inequality_solution (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 ↔ x = 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2558_255896


namespace NUMINAMATH_CALUDE_apples_in_baskets_l2558_255876

theorem apples_in_baskets (num_baskets : ℕ) (total_apples : ℕ) (apples_per_basket : ℕ) :
  num_baskets = 37 →
  total_apples = 629 →
  num_baskets * apples_per_basket = total_apples →
  apples_per_basket = 17 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_baskets_l2558_255876


namespace NUMINAMATH_CALUDE_expand_polynomial_l2558_255881

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l2558_255881


namespace NUMINAMATH_CALUDE_min_value_and_range_l2558_255807

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x
def g (a x : ℝ) : ℝ := -x^2 + a*x - 3

def e : ℝ := Real.exp 1

theorem min_value_and_range (t : ℝ) (h : t > 0) :
  (∃ (x : ℝ), x ∈ Set.Icc t (t + 2) ∧
    (∀ (y : ℝ), y ∈ Set.Icc t (t + 2) → f x ≤ f y) ∧
    ((0 < t ∧ t < 1/e → f x = -1/e) ∧
     (t ≥ 1/e → f x = t * Real.log t))) ∧
  (∀ (a : ℝ), (∃ (x₀ : ℝ), x₀ ∈ Set.Icc (1/e) e ∧ 2 * f x₀ ≥ g a x₀) →
    a ≤ -2 + 1/e + 3*e) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_range_l2558_255807


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l2558_255802

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 22 % 31 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 22 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l2558_255802


namespace NUMINAMATH_CALUDE_difference_2050th_2060th_term_l2558_255846

def arithmetic_sequence (a₁ a₂ : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * (a₂ - a₁)

theorem difference_2050th_2060th_term : 
  let a₁ := 2
  let a₂ := 9
  |arithmetic_sequence a₁ a₂ 2060 - arithmetic_sequence a₁ a₂ 2050| = 70 := by
  sorry

end NUMINAMATH_CALUDE_difference_2050th_2060th_term_l2558_255846


namespace NUMINAMATH_CALUDE_exam_total_questions_l2558_255840

/-- Represents an exam with given parameters -/
structure Exam where
  totalTime : ℕ
  answeredQuestions : ℕ
  timeUsed : ℕ
  timeLeftWhenFinished : ℕ

/-- Calculates the total number of questions on the exam -/
def totalQuestions (e : Exam) : ℕ :=
  let remainingTime := e.totalTime - e.timeUsed
  let questionRate := e.answeredQuestions / e.timeUsed
  e.answeredQuestions + questionRate * remainingTime

/-- Theorem stating that the total number of questions on the given exam is 80 -/
theorem exam_total_questions :
  let e : Exam := {
    totalTime := 60,
    answeredQuestions := 16,
    timeUsed := 12,
    timeLeftWhenFinished := 0
  }
  totalQuestions e = 80 := by
  sorry


end NUMINAMATH_CALUDE_exam_total_questions_l2558_255840


namespace NUMINAMATH_CALUDE_tetrahedron_distance_sum_l2558_255810

/-- Tetrahedron with face areas, distances, and volume -/
structure Tetrahedron where
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  S₄ : ℝ
  H₁ : ℝ
  H₂ : ℝ
  H₃ : ℝ
  H₄ : ℝ
  V : ℝ

/-- The theorem about the sum of weighted distances in a tetrahedron -/
theorem tetrahedron_distance_sum (t : Tetrahedron) (k : ℝ) 
    (h₁ : t.S₁ / 1 = k)
    (h₂ : t.S₂ / 2 = k)
    (h₃ : t.S₃ / 3 = k)
    (h₄ : t.S₄ / 4 = k) :
  1 * t.H₁ + 2 * t.H₂ + 3 * t.H₃ + 4 * t.H₄ = 3 * t.V / k := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_distance_sum_l2558_255810


namespace NUMINAMATH_CALUDE_sandwich_combinations_l2558_255818

theorem sandwich_combinations (meat : ℕ) (cheese : ℕ) (bread : ℕ) :
  meat = 12 → cheese = 11 → bread = 5 →
  (meat * (cheese.choose 3) * bread) = 9900 :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l2558_255818


namespace NUMINAMATH_CALUDE_smallest_multiple_of_8_no_repeated_digits_remainder_l2558_255849

/-- A function that checks if a natural number has no repeated digits -/
def hasNoRepeatedDigits (n : ℕ) : Prop := sorry

/-- The smallest multiple of 8 with no repeated digits -/
def M : ℕ := sorry

theorem smallest_multiple_of_8_no_repeated_digits_remainder :
  (M % 1000 = 120) ∧
  (∀ k : ℕ, k < M → (k % 8 = 0 → ¬hasNoRepeatedDigits k)) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_8_no_repeated_digits_remainder_l2558_255849


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l2558_255897

def proposition_p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0 ∧ a > 0

def proposition_q (x : ℝ) : Prop := 2 < x ∧ x ≤ 3

theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, proposition_p x 1 ∧ proposition_q x → 2 < x ∧ x < 3 :=
sorry

theorem range_of_a_when_not_p_implies_not_q :
  ∀ a : ℝ, (∀ x : ℝ, ¬(proposition_p x a) → ¬(proposition_q x)) →
  1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_not_p_implies_not_q_l2558_255897


namespace NUMINAMATH_CALUDE_first_day_student_tickets_l2558_255831

/-- The number of student tickets sold on the first day -/
def student_tickets_day1 : ℕ := 3

/-- The price of a student ticket -/
def student_ticket_price : ℕ := 9

/-- The price of a senior citizen ticket -/
def senior_ticket_price : ℕ := 13

theorem first_day_student_tickets :
  student_tickets_day1 = 3 ∧
  4 * senior_ticket_price + student_tickets_day1 * student_ticket_price = 79 ∧
  12 * senior_ticket_price + 10 * student_ticket_price = 246 ∧
  student_ticket_price = 9 := by
sorry

end NUMINAMATH_CALUDE_first_day_student_tickets_l2558_255831


namespace NUMINAMATH_CALUDE_average_movie_length_l2558_255844

def miles_run : ℕ := 15
def minutes_per_mile : ℕ := 12
def number_of_movies : ℕ := 2

theorem average_movie_length :
  (miles_run * minutes_per_mile) / number_of_movies = 90 :=
by sorry

end NUMINAMATH_CALUDE_average_movie_length_l2558_255844


namespace NUMINAMATH_CALUDE_problem_statement_l2558_255828

theorem problem_statement (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xyz : x * y * z = 1)
  (h_x_z : x + 1 / z = 10)
  (h_y_x : y + 1 / x = 5) :
  z + 1 / y = 17 / 49 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2558_255828


namespace NUMINAMATH_CALUDE_second_to_first_ratio_l2558_255814

/-- Represents the amount of food eaten by each guinea pig -/
structure GuineaPigFood where
  first : ℚ
  second : ℚ
  third : ℚ

/-- Calculates the total food eaten by all guinea pigs -/
def totalFood (gpf : GuineaPigFood) : ℚ :=
  gpf.first + gpf.second + gpf.third

/-- Theorem: The ratio of food eaten by the second guinea pig to the first guinea pig is 2:1 -/
theorem second_to_first_ratio (gpf : GuineaPigFood) : 
  gpf.first = 2 → 
  gpf.third = gpf.second + 3 → 
  totalFood gpf = 13 → 
  gpf.second / gpf.first = 2 := by
sorry

end NUMINAMATH_CALUDE_second_to_first_ratio_l2558_255814


namespace NUMINAMATH_CALUDE_flour_remaining_l2558_255862

theorem flour_remaining (initial_amount : ℝ) : 
  let first_removal_percent : ℝ := 60
  let second_removal_percent : ℝ := 25
  let remaining_after_first : ℝ := initial_amount * (100 - first_removal_percent) / 100
  let final_remaining : ℝ := remaining_after_first * (100 - second_removal_percent) / 100
  final_remaining = initial_amount * 30 / 100 := by sorry

end NUMINAMATH_CALUDE_flour_remaining_l2558_255862


namespace NUMINAMATH_CALUDE_number_divided_by_three_equals_number_minus_five_l2558_255848

theorem number_divided_by_three_equals_number_minus_five : 
  ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_number_divided_by_three_equals_number_minus_five_l2558_255848


namespace NUMINAMATH_CALUDE_four_integers_sum_problem_l2558_255801

theorem four_integers_sum_problem :
  ∀ a b c d : ℕ,
    0 < a ∧ a < b ∧ b < c ∧ c < d →
    a + b + c = 6 →
    a + b + d = 7 →
    a + c + d = 8 →
    b + c + d = 9 →
    a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 :=
by sorry

end NUMINAMATH_CALUDE_four_integers_sum_problem_l2558_255801


namespace NUMINAMATH_CALUDE_point_B_coordinates_l2558_255872

def point_A : ℝ × ℝ := (-1, 5)
def vector_a : ℝ × ℝ := (2, 3)

theorem point_B_coordinates :
  ∀ (B : ℝ × ℝ),
  (B.1 - point_A.1, B.2 - point_A.2) = (3 * vector_a.1, 3 * vector_a.2) →
  B = (5, 14) := by
sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l2558_255872


namespace NUMINAMATH_CALUDE_total_cost_is_649_70_l2558_255804

/-- Calculates the total cost of a guitar and amplifier in dollars --/
def total_cost_in_dollars (guitar_price : ℝ) (amplifier_price : ℝ) 
  (guitar_discount : ℝ) (amplifier_discount : ℝ) (vat_rate : ℝ) (exchange_rate : ℝ) : ℝ :=
  let discounted_guitar := guitar_price * (1 - guitar_discount)
  let discounted_amplifier := amplifier_price * (1 - amplifier_discount)
  let total_with_vat := (discounted_guitar + discounted_amplifier) * (1 + vat_rate)
  total_with_vat * exchange_rate

/-- Theorem stating that the total cost is equal to $649.70 --/
theorem total_cost_is_649_70 :
  total_cost_in_dollars 330 220 0.10 0.05 0.07 1.20 = 649.70 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_649_70_l2558_255804


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2558_255835

/-- Given an arithmetic sequence with a non-zero common difference,
    if its second, third, and sixth terms form a geometric sequence,
    then the common ratio of this geometric sequence is 3. -/
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) -- The arithmetic sequence
  (d : ℝ) -- The common difference of the arithmetic sequence
  (h1 : d ≠ 0) -- The common difference is non-zero
  (h2 : ∀ n, a (n + 1) = a n + d) -- Definition of arithmetic sequence
  (h3 : ∃ r, r ≠ 0 ∧ a 3 = r * a 2 ∧ a 6 = r * a 3) -- Second, third, and sixth terms form a geometric sequence
  : ∃ r, r = 3 ∧ a 3 = r * a 2 ∧ a 6 = r * a 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2558_255835


namespace NUMINAMATH_CALUDE_gcd_n_cube_minus_27_and_n_minus_3_l2558_255864

theorem gcd_n_cube_minus_27_and_n_minus_3 (n : ℕ) (h : n > 3^2) :
  Nat.gcd (n^3 - 27) (n - 3) = n - 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cube_minus_27_and_n_minus_3_l2558_255864


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_under_120_l2558_255842

theorem largest_multiple_of_9_under_120 : 
  ∃ n : ℕ, n * 9 = 117 ∧ 117 < 120 ∧ ∀ m : ℕ, m * 9 < 120 → m * 9 ≤ 117 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_under_120_l2558_255842


namespace NUMINAMATH_CALUDE_expression_equals_six_l2558_255834

theorem expression_equals_six :
  2 - Real.sqrt 3 + (2 - Real.sqrt 3)⁻¹ + (Real.sqrt 3 + 2)⁻¹ = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_six_l2558_255834


namespace NUMINAMATH_CALUDE_lindas_tv_cost_l2558_255865

/-- The cost of Linda's TV purchase, given her original savings and the fraction spent on furniture. -/
theorem lindas_tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 1200 → 
  furniture_fraction = 3/4 → 
  tv_cost = savings * (1 - furniture_fraction) → 
  tv_cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_lindas_tv_cost_l2558_255865


namespace NUMINAMATH_CALUDE_range_of_f_l2558_255861

-- Define the function
def f (x : ℝ) : ℝ := |x - 3| - |x + 5|

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-8) 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2558_255861


namespace NUMINAMATH_CALUDE_max_triangle_area_l2558_255813

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 6*x

-- Define points A and B on the parabola
def pointA (x₁ y₁ : ℝ) : Prop := parabola x₁ y₁
def pointB (x₂ y₂ : ℝ) : Prop := parabola x₂ y₂

-- Define the conditions
def conditions (x₁ x₂ : ℝ) : Prop := x₁ ≠ x₂ ∧ x₁ + x₂ = 4

-- Define the perpendicular bisector intersection with x-axis
def pointC (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ := sorry

-- Define the area of triangle ABC
def triangleArea (x₁ y₁ x₂ y₂ : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_triangle_area 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (hA : pointA x₁ y₁) 
  (hB : pointB x₂ y₂) 
  (hC : conditions x₁ x₂) :
  ∃ (max_area : ℝ), 
    (∀ (x₁' y₁' x₂' y₂' : ℝ), 
      pointA x₁' y₁' → pointB x₂' y₂' → conditions x₁' x₂' →
      triangleArea x₁' y₁' x₂' y₂' ≤ max_area) ∧
    max_area = (14/3) * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l2558_255813


namespace NUMINAMATH_CALUDE_triangle_area_rational_l2558_255858

/-- Given a triangle with vertices whose coordinates are integers adjusted by adding 0.5,
    prove that its area is always rational. -/
theorem triangle_area_rational (x₁ x₂ x₃ y₁ y₂ y₃ : ℤ) :
  ∃ (p q : ℤ), q ≠ 0 ∧ 
    (1/2 : ℚ) * |((x₁ + 1/2) * ((y₂ + 1/2) - (y₃ + 1/2)) + 
                  (x₂ + 1/2) * ((y₃ + 1/2) - (y₁ + 1/2)) + 
                  (x₃ + 1/2) * ((y₁ + 1/2) - (y₂ + 1/2)))| = p / q :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_rational_l2558_255858


namespace NUMINAMATH_CALUDE_sequence_sum_l2558_255874

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = n² + 1, prove a₁ + a₉ = 19 -/
theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = n^2 + 1) : 
    a 1 + a 9 = 19 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2558_255874


namespace NUMINAMATH_CALUDE_joan_picked_apples_l2558_255889

/-- The number of apples Joan has now -/
def total_apples : ℕ := 70

/-- The number of apples Melanie gave to Joan -/
def melanie_apples : ℕ := 27

/-- The number of apples Joan picked from the orchard -/
def orchard_apples : ℕ := total_apples - melanie_apples

theorem joan_picked_apples : orchard_apples = 43 := by
  sorry

end NUMINAMATH_CALUDE_joan_picked_apples_l2558_255889


namespace NUMINAMATH_CALUDE_least_value_quadratic_l2558_255860

theorem least_value_quadratic (y : ℝ) : 
  (2 * y^2 + 7 * y + 3 = 5) → y ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_least_value_quadratic_l2558_255860


namespace NUMINAMATH_CALUDE_equal_balls_probability_l2558_255823

/-- Represents the urn state -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents a single draw operation -/
inductive Draw
  | Red
  | Blue

/-- Performs a single draw operation on the urn state -/
def drawOperation (state : UrnState) (draw : Draw) : UrnState :=
  match draw with
  | Draw.Red => UrnState.mk (state.red + 1) state.blue
  | Draw.Blue => UrnState.mk state.red (state.blue + 1)

/-- Performs a sequence of draw operations on the urn state -/
def performOperations (initial : UrnState) (draws : List Draw) : UrnState :=
  draws.foldl drawOperation initial

/-- Calculates the probability of a specific sequence of draws -/
def sequenceProbability (draws : List Draw) : ℚ :=
  sorry

/-- Calculates the number of valid sequences that result in 4 red and 4 blue balls -/
def validSequencesCount : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem equal_balls_probability :
  let initialState := UrnState.mk 2 1
  let finalState := UrnState.mk 4 4
  (validSequencesCount * sequenceProbability (List.replicate 5 Draw.Red)) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_equal_balls_probability_l2558_255823


namespace NUMINAMATH_CALUDE_audiobook_completion_time_l2558_255839

/-- Calculates the time to finish audiobooks given the number of books, length per book, and daily listening time. -/
def timeToFinishAudiobooks (numBooks : ℕ) (hoursPerBook : ℕ) (hoursPerDay : ℕ) : ℕ :=
  numBooks * (hoursPerBook / hoursPerDay)

/-- Proves that under the given conditions, it takes 90 days to finish the audiobooks. -/
theorem audiobook_completion_time :
  timeToFinishAudiobooks 6 30 2 = 90 :=
by
  sorry

#eval timeToFinishAudiobooks 6 30 2

end NUMINAMATH_CALUDE_audiobook_completion_time_l2558_255839


namespace NUMINAMATH_CALUDE_painted_square_ratio_l2558_255890

/-- Given a square with side length s and a brush of width w, 
    if the painted area along the midline and one diagonal is one-third of the square's area, 
    then the ratio s/w equals 2√2 + 1 -/
theorem painted_square_ratio (s w : ℝ) (h_positive_s : 0 < s) (h_positive_w : 0 < w) :
  s * w + 2 * (1/2 * ((s * Real.sqrt 2) / 2 - (w * Real.sqrt 2) / 2)^2) = s^2 / 3 →
  s / w = 2 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_painted_square_ratio_l2558_255890


namespace NUMINAMATH_CALUDE_probability_three_specific_heads_out_of_five_probability_three_specific_heads_out_of_five_proof_l2558_255894

/-- The probability of getting heads on exactly three specific coins out of five coins -/
theorem probability_three_specific_heads_out_of_five : ℝ :=
  let n_coins : ℕ := 5
  let n_specific_coins : ℕ := 3
  let p_head : ℝ := 1 / 2
  1 / 8

/-- Proof of the theorem -/
theorem probability_three_specific_heads_out_of_five_proof :
  probability_three_specific_heads_out_of_five = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_specific_heads_out_of_five_probability_three_specific_heads_out_of_five_proof_l2558_255894


namespace NUMINAMATH_CALUDE_composite_sum_of_prime_powers_l2558_255809

theorem composite_sum_of_prime_powers (p q t : Nat) : 
  Prime p → Prime q → Prime t → p ≠ q → p ≠ t → q ≠ t →
  ∃ n : Nat, n > 1 ∧ n ∣ (2016^p + 2017^q + 2018^t) :=
by sorry

end NUMINAMATH_CALUDE_composite_sum_of_prime_powers_l2558_255809


namespace NUMINAMATH_CALUDE_square_side_length_l2558_255850

theorem square_side_length (area : ℝ) (side_length : ℝ) :
  area = 4 ∧ area = side_length ^ 2 → side_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2558_255850


namespace NUMINAMATH_CALUDE_iris_mall_spending_l2558_255871

/-- The total amount spent by Iris at the mall --/
def total_spent (jacket_price shorts_price pants_price : ℕ) 
                (jacket_count shorts_count pants_count : ℕ) : ℕ :=
  jacket_price * jacket_count + shorts_price * shorts_count + pants_price * pants_count

/-- Theorem stating that Iris spent $90 at the mall --/
theorem iris_mall_spending : 
  total_spent 10 6 12 3 2 4 = 90 := by
  sorry

end NUMINAMATH_CALUDE_iris_mall_spending_l2558_255871


namespace NUMINAMATH_CALUDE_elephant_hole_theorem_l2558_255824

/-- A paper represents a rectangular sheet with a given area -/
structure Paper where
  area : ℝ
  area_pos : area > 0

/-- A series of cuts can be represented as a function that transforms a paper -/
def Cut := Paper → Paper

/-- The theorem states that there exists a cut that can create a hole larger than the original paper -/
theorem elephant_hole_theorem (initial_paper : Paper) (k : ℝ) (h_k : k > 1) :
  ∃ (cut : Cut), (cut initial_paper).area > k * initial_paper.area := by
  sorry

end NUMINAMATH_CALUDE_elephant_hole_theorem_l2558_255824


namespace NUMINAMATH_CALUDE_triangle_side_length_l2558_255869

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S_ABC = 1/4(a^2 + b^2 - c^2), b = 1, and a = √2, then c = 1. -/
theorem triangle_side_length (a b c : ℝ) (h_area : (a^2 + b^2 - c^2) / 4 = a * b * Real.sin (π/4) / 2)
  (h_b : b = 1) (h_a : a = Real.sqrt 2) : c = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2558_255869


namespace NUMINAMATH_CALUDE_commute_distance_is_21_l2558_255838

/-- Represents the carpool scenario with given parameters -/
structure Carpool where
  friends : ℕ := 5
  gas_price : ℚ := 5/2
  car_efficiency : ℚ := 30
  commute_days_per_week : ℕ := 5
  commute_weeks_per_month : ℕ := 4
  individual_payment : ℚ := 14

/-- Calculates the one-way commute distance given a Carpool scenario -/
def calculate_commute_distance (c : Carpool) : ℚ :=
  (c.individual_payment * c.friends * c.car_efficiency) / 
  (2 * c.gas_price * c.commute_days_per_week * c.commute_weeks_per_month)

/-- Theorem stating that the one-way commute distance is 21 miles -/
theorem commute_distance_is_21 (c : Carpool) : 
  calculate_commute_distance c = 21 := by
  sorry

end NUMINAMATH_CALUDE_commute_distance_is_21_l2558_255838


namespace NUMINAMATH_CALUDE_probability_different_with_three_l2558_255800

/-- The number of faces on a fair die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when throwing two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of favorable outcomes (different numbers with one being 3) -/
def favorableOutcomes : ℕ := 2 * (numFaces - 1)

/-- The probability of getting different numbers on two fair dice with one showing 3 -/
def probabilityDifferentWithThree : ℚ := favorableOutcomes / totalOutcomes

theorem probability_different_with_three :
  probabilityDifferentWithThree = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_with_three_l2558_255800


namespace NUMINAMATH_CALUDE_state_fraction_l2558_255827

theorem state_fraction (total_states : ℕ) (period_states : ℕ) 
  (h1 : total_states = 22) (h2 : period_states = 12) : 
  (period_states : ℚ) / total_states = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_state_fraction_l2558_255827


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2558_255837

theorem complex_magnitude_problem (z : ℂ) (h : Complex.I * (2 - z) = 3 + Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2558_255837


namespace NUMINAMATH_CALUDE_sector_arc_length_l2558_255851

/-- Given a sector with radius 2 and area 4, the length of the arc
    corresponding to the central angle is 4. -/
theorem sector_arc_length (r : ℝ) (S : ℝ) (l : ℝ) : 
  r = 2 → S = 4 → S = (1/2) * r * l → l = 4 := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2558_255851


namespace NUMINAMATH_CALUDE_smallest_n_for_red_vertices_symmetry_l2558_255825

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- A set of 5 red vertices in a regular polygon -/
def RedVertices (n : ℕ) := Fin 5 → Fin n

/-- An axis of symmetry of a regular polygon -/
def AxisOfSymmetry (n : ℕ) := ℕ

/-- Checks if a vertex is reflected onto another vertex across an axis -/
def isReflectedOnto (n : ℕ) (p : RegularPolygon n) (v1 v2 : Fin n) (axis : AxisOfSymmetry n) : Prop :=
  sorry

/-- The main theorem -/
theorem smallest_n_for_red_vertices_symmetry :
  (∀ n : ℕ, n ≥ 14 →
    ∀ p : RegularPolygon n,
    ∀ red : RedVertices n,
    ∃ axis : AxisOfSymmetry n,
    ∀ v1 v2 : Fin 5, v1 ≠ v2 → ¬isReflectedOnto n p (red v1) (red v2) axis) ∧
  (∀ n : ℕ, n < 14 →
    ∃ p : RegularPolygon n,
    ∃ red : RedVertices n,
    ∀ axis : AxisOfSymmetry n,
    ∃ v1 v2 : Fin 5, v1 ≠ v2 ∧ isReflectedOnto n p (red v1) (red v2) axis) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_red_vertices_symmetry_l2558_255825


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2558_255878

theorem quadratic_equation_solution (w : ℝ) :
  (w + 15)^2 = (4*w + 9) * (3*w + 6) →
  w^2 = (((-21 + Real.sqrt 7965) / 22)^2) ∨ w^2 = (((-21 - Real.sqrt 7965) / 22)^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2558_255878


namespace NUMINAMATH_CALUDE_fifty_third_number_is_53_l2558_255863

/-- Represents the sequence of numbers spoken in the modified counting game -/
def modifiedCountingSequence : ℕ → ℕ
| 0 => 1  -- Jo starts with 1
| n + 1 => 
  let prevNum := modifiedCountingSequence n
  if prevNum % 3 = 0 then prevNum + 2  -- Skip a number after multiples of 3
  else prevNum + 1  -- Otherwise, increment by 1

/-- The 53rd number in the modified counting sequence is 53 -/
theorem fifty_third_number_is_53 : modifiedCountingSequence 52 = 53 := by
  sorry

#eval modifiedCountingSequence 52  -- Evaluates to 53

end NUMINAMATH_CALUDE_fifty_third_number_is_53_l2558_255863


namespace NUMINAMATH_CALUDE_chord_probability_chord_probability_proof_l2558_255841

/-- The probability that a randomly chosen point on a circle's circumference,
    when connected to a fixed point on the circumference, forms a chord with
    length between R and √3R, where R is the radius of the circle. -/
theorem chord_probability (R : ℝ) (R_pos : R > 0) : ℝ :=
  1 / 3

/-- Proof of the chord probability theorem -/
theorem chord_probability_proof (R : ℝ) (R_pos : R > 0) :
  chord_probability R R_pos = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_probability_chord_probability_proof_l2558_255841


namespace NUMINAMATH_CALUDE_sin_180_degrees_l2558_255836

theorem sin_180_degrees : Real.sin (π) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_180_degrees_l2558_255836


namespace NUMINAMATH_CALUDE_cuboid_volume_example_l2558_255892

/-- The volume of a cuboid with given base area and height -/
def cuboid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  base_area * height

/-- Theorem: The volume of a cuboid with base area 18 m^2 and height 8 m is 144 m^3 -/
theorem cuboid_volume_example : cuboid_volume 18 8 = 144 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_example_l2558_255892


namespace NUMINAMATH_CALUDE_custom_mul_two_neg_three_l2558_255830

-- Define the custom multiplication operation
def custom_mul (a b : ℤ) : ℤ := a^2 + a*b - b^2

-- Theorem statement
theorem custom_mul_two_neg_three :
  custom_mul 2 (-3) = -11 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_two_neg_three_l2558_255830


namespace NUMINAMATH_CALUDE_absolute_curve_sufficient_not_necessary_l2558_255803

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the property of being on the curve y = |x|
def onAbsoluteCurve (p : Point2D) : Prop :=
  p.y = |p.x|

-- Define the property of equal distance to both axes
def equalDistanceToAxes (p : Point2D) : Prop :=
  |p.x| = |p.y|

-- Theorem statement
theorem absolute_curve_sufficient_not_necessary :
  (∀ p : Point2D, onAbsoluteCurve p → equalDistanceToAxes p) ∧
  (∃ p : Point2D, equalDistanceToAxes p ∧ ¬onAbsoluteCurve p) :=
sorry

end NUMINAMATH_CALUDE_absolute_curve_sufficient_not_necessary_l2558_255803


namespace NUMINAMATH_CALUDE_min_total_cost_l2558_255815

/-- Represents a salon with prices for haircut, facial cleaning, and nails -/
structure Salon where
  name : String
  haircut : ℕ
  facial : ℕ
  nails : ℕ

/-- Calculates the total cost of services at a salon -/
def totalCost (s : Salon) : ℕ := s.haircut + s.facial + s.nails

/-- The list of salons with their prices -/
def salonList : List Salon := [
  { name := "Gustran Salon", haircut := 45, facial := 22, nails := 30 },
  { name := "Barbara's Shop", haircut := 30, facial := 28, nails := 40 },
  { name := "The Fancy Salon", haircut := 34, facial := 30, nails := 20 }
]

/-- Theorem: The minimum total cost among the salons is 84 -/
theorem min_total_cost : 
  (salonList.map totalCost).minimum? = some 84 := by
  sorry

end NUMINAMATH_CALUDE_min_total_cost_l2558_255815


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l2558_255893

/-- An arithmetic-geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℝ

/-- Sum of the first n terms of a sequence -/
def SumOfFirstNTerms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem arithmetic_geometric_sequence_sum (a : ArithmeticGeometricSequence) :
  let S := SumOfFirstNTerms a.a
  S 2 = 3 ∧ S 4 = 15 → S 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l2558_255893


namespace NUMINAMATH_CALUDE_square_difference_squared_l2558_255895

theorem square_difference_squared : (7^2 - 3^2)^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_squared_l2558_255895


namespace NUMINAMATH_CALUDE_saras_savings_jar_l2558_255899

theorem saras_savings_jar (total_amount : ℕ) (total_bills : ℕ) 
  (h1 : total_amount = 84)
  (h2 : total_bills = 58) : 
  ∃ (ones twos : ℕ), 
    ones + twos = total_bills ∧ 
    ones + 2 * twos = total_amount ∧
    ones = 32 := by
  sorry

end NUMINAMATH_CALUDE_saras_savings_jar_l2558_255899


namespace NUMINAMATH_CALUDE_problem_statement_l2558_255882

theorem problem_statement (x y : ℝ) (h : |x - 1/2| + Real.sqrt (y^2 - 1) = 0) : 
  |x| + |y| = 3/2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2558_255882


namespace NUMINAMATH_CALUDE_quadratic_root_arithmetic_sequence_l2558_255873

/-- Given real numbers a, b, c forming an arithmetic sequence with c ≥ b ≥ a ≥ 0,
    the single root of the quadratic cx^2 + bx + a = 0 is -1 - (√3)/3 -/
theorem quadratic_root_arithmetic_sequence (a b c : ℝ) : 
  c ≥ b ∧ b ≥ a ∧ a ≥ 0 →
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →
  (∃! x : ℝ, c*x^2 + b*x + a = 0) →
  (∃ x : ℝ, c*x^2 + b*x + a = 0 ∧ x = -1 - Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_arithmetic_sequence_l2558_255873


namespace NUMINAMATH_CALUDE_max_students_is_25_l2558_255870

/-- A graph representing friendships in a class of students. -/
structure FriendshipGraph (n : ℕ) where
  friends : Fin n → Fin n → Prop

/-- The property that among any six students, there are two that are not friends. -/
def hasTwoNonFriends (G : FriendshipGraph n) : Prop :=
  ∀ (s : Finset (Fin n)), s.card = 6 → ∃ (i j : Fin n), i ∈ s ∧ j ∈ s ∧ i ≠ j ∧ ¬G.friends i j

/-- The property that for any pair of non-friends, there is a student among the remaining four who is friends with both. -/
def hasCommonFriend (G : FriendshipGraph n) : Prop :=
  ∀ (i j : Fin n), i ≠ j → ¬G.friends i j →
    ∃ (k : Fin n), k ≠ i ∧ k ≠ j ∧ G.friends i k ∧ G.friends j k

/-- The main theorem: The maximum number of students satisfying the given conditions is 25. -/
theorem max_students_is_25 :
  (∃ (n : ℕ) (G : FriendshipGraph n), hasTwoNonFriends G ∧ hasCommonFriend G) ∧
  (∀ (n : ℕ) (G : FriendshipGraph n), hasTwoNonFriends G → hasCommonFriend G → n ≤ 25) :=
sorry

end NUMINAMATH_CALUDE_max_students_is_25_l2558_255870


namespace NUMINAMATH_CALUDE_overall_loss_percentage_l2558_255806

/-- Calculate the overall loss percentage on three articles with given purchase prices, exchange rates, shipping fees, selling prices, and sales tax. -/
theorem overall_loss_percentage
  (purchase_a purchase_b purchase_c : ℝ)
  (exchange_eur exchange_gbp : ℝ)
  (shipping_fee : ℝ)
  (sell_a sell_b sell_c : ℝ)
  (sales_tax_rate : ℝ)
  (h_purchase_a : purchase_a = 100)
  (h_purchase_b : purchase_b = 200)
  (h_purchase_c : purchase_c = 300)
  (h_exchange_eur : exchange_eur = 1.1)
  (h_exchange_gbp : exchange_gbp = 1.3)
  (h_shipping_fee : shipping_fee = 10)
  (h_sell_a : sell_a = 110)
  (h_sell_b : sell_b = 250)
  (h_sell_c : sell_c = 330)
  (h_sales_tax_rate : sales_tax_rate = 0.05) :
  ∃ (loss_percentage : ℝ), 
    abs (loss_percentage - 0.0209) < 0.0001 ∧
    loss_percentage = 
      (((sell_a + sell_b + sell_c) * (1 + sales_tax_rate) - 
        (purchase_a + purchase_b * exchange_eur + purchase_c * exchange_gbp + 3 * shipping_fee)) / 
       (purchase_a + purchase_b * exchange_eur + purchase_c * exchange_gbp + 3 * shipping_fee)) * (-100) :=
by sorry

end NUMINAMATH_CALUDE_overall_loss_percentage_l2558_255806


namespace NUMINAMATH_CALUDE_sine_cosine_identity_l2558_255821

theorem sine_cosine_identity :
  Real.sin (65 * π / 180) * Real.cos (35 * π / 180) -
  Real.sin (25 * π / 180) * Real.sin (35 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_identity_l2558_255821


namespace NUMINAMATH_CALUDE_abc_inequalities_l2558_255819

theorem abc_inequalities (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1/3) ∧ (1/a + 1/b + 1/c ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequalities_l2558_255819


namespace NUMINAMATH_CALUDE_jen_birds_count_l2558_255816

/-- The number of birds Jen has given the conditions -/
def total_birds (chickens ducks geese : ℕ) : ℕ :=
  chickens + ducks + geese

/-- Theorem stating the total number of birds Jen has -/
theorem jen_birds_count :
  ∀ (chickens ducks geese : ℕ),
    ducks = 150 →
    ducks = 4 * chickens + 10 →
    geese = (ducks + chickens) / 2 →
    total_birds chickens ducks geese = 277 := by
  sorry

#check jen_birds_count

end NUMINAMATH_CALUDE_jen_birds_count_l2558_255816


namespace NUMINAMATH_CALUDE_student_travel_fraction_l2558_255859

theorem student_travel_fraction (total_distance : ℝ) 
  (bus_fraction : ℝ) (car_distance : ℝ) :
  total_distance = 90 ∧ 
  bus_fraction = 2/3 ∧ 
  car_distance = 12 →
  (total_distance - (bus_fraction * total_distance + car_distance)) / total_distance = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_student_travel_fraction_l2558_255859


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_l2558_255820

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*a*c)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 :=
by sorry

theorem equality_condition (a : ℝ) (ha : a > 0) :
  (a / Real.sqrt (a^2 + 8*a*a)) + (a / Real.sqrt (a^2 + 8*a*a)) + (a / Real.sqrt (a^2 + 8*a*a)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_l2558_255820


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2558_255826

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 16) :
  a 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2558_255826


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_system_l2558_255868

theorem smallest_solution_congruence_system (x : ℕ) : x = 1309 ↔ 
  (x > 0) ∧
  (3 * x ≡ 9 [MOD 12]) ∧ 
  (5 * x + 4 ≡ 14 [MOD 7]) ∧ 
  (4 * x - 3 ≡ 2 * x + 5 [MOD 17]) ∧ 
  (x ≡ 4 [MOD 11]) ∧
  (∀ y : ℕ, y > 0 → 
    (3 * y ≡ 9 [MOD 12]) → 
    (5 * y + 4 ≡ 14 [MOD 7]) → 
    (4 * y - 3 ≡ 2 * y + 5 [MOD 17]) → 
    (y ≡ 4 [MOD 11]) → 
    y ≥ x) :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_system_l2558_255868


namespace NUMINAMATH_CALUDE_roots_of_varying_signs_l2558_255884

theorem roots_of_varying_signs :
  (∃ x y : ℝ, x * y < 0 ∧ 4 * x^2 - 8 = 40 ∧ 4 * y^2 - 8 = 40) ∧
  (∃ x y : ℝ, x * y < 0 ∧ (3*x-2)^2 = (x+2)^2 ∧ (3*y-2)^2 = (y+2)^2) ∧
  (∃ x y : ℝ, x * y < 0 ∧ x^3 - 8*x^2 + 13*x + 10 = 0 ∧ y^3 - 8*y^2 + 13*y + 10 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_roots_of_varying_signs_l2558_255884


namespace NUMINAMATH_CALUDE_hotel_rooms_l2558_255885

theorem hotel_rooms (total_floors : Nat) (unavailable_floors : Nat) (available_rooms : Nat) :
  total_floors = 10 →
  unavailable_floors = 1 →
  available_rooms = 90 →
  (total_floors - unavailable_floors) * (available_rooms / (total_floors - unavailable_floors)) = available_rooms :=
by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_l2558_255885


namespace NUMINAMATH_CALUDE_correct_sunset_time_l2558_255822

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the sunset time given sunrise time and daylight duration -/
def calculateSunset (sunrise : Time) (daylight : Duration) : Time :=
  { hours := (sunrise.hours + daylight.hours + (sunrise.minutes + daylight.minutes) / 60) % 24,
    minutes := (sunrise.minutes + daylight.minutes) % 60 }

theorem correct_sunset_time :
  let sunrise : Time := { hours := 7, minutes := 12 }
  let daylight : Duration := { hours := 9, minutes := 45 }
  let calculated_sunset : Time := calculateSunset sunrise daylight
  calculated_sunset = { hours := 16, minutes := 57 } :=
by sorry

end NUMINAMATH_CALUDE_correct_sunset_time_l2558_255822


namespace NUMINAMATH_CALUDE_library_books_theorem_l2558_255857

/-- The number of books taken by the librarian -/
def books_taken : ℕ := 10

/-- The number of books that can fit on each shelf -/
def books_per_shelf : ℕ := 4

/-- The number of shelves needed for the remaining books -/
def shelves_needed : ℕ := 9

/-- The total number of books to put away -/
def total_books : ℕ := 46

theorem library_books_theorem :
  total_books = books_per_shelf * shelves_needed + books_taken := by
  sorry

end NUMINAMATH_CALUDE_library_books_theorem_l2558_255857


namespace NUMINAMATH_CALUDE_geometric_sequence_cosine_l2558_255887

open Real

theorem geometric_sequence_cosine (a : ℝ) : 
  0 < a → a < 2 * π → 
  (∃ r : ℝ, cos a * r = cos (2 * a) ∧ cos (2 * a) * r = cos (3 * a)) → 
  a = π := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_cosine_l2558_255887


namespace NUMINAMATH_CALUDE_psychologist_pricing_l2558_255856

theorem psychologist_pricing (F A : ℝ) 
  (h1 : F + 4 * A = 300)  -- 5 hours of therapy costs $300
  (h2 : F + 2 * A = 188)  -- 3 hours of therapy costs $188
  : F - A = 20 := by
  sorry

end NUMINAMATH_CALUDE_psychologist_pricing_l2558_255856


namespace NUMINAMATH_CALUDE_sum_of_six_smallest_multiples_of_12_l2558_255866

theorem sum_of_six_smallest_multiples_of_12 : 
  (Finset.range 6).sum (λ i => 12 * (i + 1)) = 252 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_six_smallest_multiples_of_12_l2558_255866


namespace NUMINAMATH_CALUDE_gold_coin_percentage_l2558_255832

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  beadPercentage : ℝ
  silverCoinPercentage : ℝ
  goldCoinPercentage : ℝ

/-- The urn composition satisfies the given conditions --/
def validUrnComposition (u : UrnComposition) : Prop :=
  u.beadPercentage = 30 ∧
  u.silverCoinPercentage + u.goldCoinPercentage = 70 ∧
  u.silverCoinPercentage = 35

theorem gold_coin_percentage (u : UrnComposition) 
  (h : validUrnComposition u) : u.goldCoinPercentage = 35 := by
  sorry

#check gold_coin_percentage

end NUMINAMATH_CALUDE_gold_coin_percentage_l2558_255832


namespace NUMINAMATH_CALUDE_parallel_line_plane_false_l2558_255888

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when a line is in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

theorem parallel_line_plane_false : 
  ¬ (∀ (α : Plane3D) (b : Line3D), 
    parallel_line_plane b α → 
    (∀ (a : Line3D), line_in_plane a α → parallel_lines b a)) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_plane_false_l2558_255888


namespace NUMINAMATH_CALUDE_sqrt_two_equals_two_to_one_sixth_l2558_255880

theorem sqrt_two_equals_two_to_one_sixth : ∃ (x : ℝ), x > 0 ∧ x^2 = 2 ∧ x = 2^(1/6) := by sorry

end NUMINAMATH_CALUDE_sqrt_two_equals_two_to_one_sixth_l2558_255880


namespace NUMINAMATH_CALUDE_equal_one_two_digit_prob_l2558_255845

/-- A 20-sided die with numbers from 1 to 20 -/
def twentySidedDie : Finset ℕ := Finset.range 20

/-- The probability of rolling a one-digit number on a 20-sided die -/
def probOneDigit : ℚ := 9 / 20

/-- The probability of rolling a two-digit number on a 20-sided die -/
def probTwoDigit : ℚ := 11 / 20

/-- The number of dice rolled -/
def numDice : ℕ := 6

/-- The probability of rolling an equal number of one-digit and two-digit numbers on 6 20-sided dice -/
theorem equal_one_two_digit_prob : 
  (Nat.choose numDice (numDice / 2) : ℚ) * probOneDigit ^ (numDice / 2) * probTwoDigit ^ (numDice / 2) = 970701 / 3200000 := by
  sorry

end NUMINAMATH_CALUDE_equal_one_two_digit_prob_l2558_255845


namespace NUMINAMATH_CALUDE_number_division_theorem_l2558_255829

theorem number_division_theorem : ∃! n : ℕ, 
  n / (2615 + 3895) = 3 * (3895 - 2615) ∧ 
  n % (2615 + 3895) = 65 := by
sorry

end NUMINAMATH_CALUDE_number_division_theorem_l2558_255829


namespace NUMINAMATH_CALUDE_smallest_student_count_l2558_255808

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ
  twelfth : ℕ

/-- The ratios between 9th grade and other grades --/
def ratios : GradeCount → Prop
  | ⟨n, t, e, w⟩ => 3 * t = 2 * n ∧ 5 * e = 4 * n ∧ 7 * w = 6 * n

/-- The total number of students --/
def total_students (g : GradeCount) : ℕ :=
  g.ninth + g.tenth + g.eleventh + g.twelfth

/-- The theorem stating the smallest possible number of students --/
theorem smallest_student_count :
  ∃ (g : GradeCount), ratios g ∧ total_students g = 349 ∧
  (∀ (h : GradeCount), ratios h → total_students h ≥ 349) :=
sorry

end NUMINAMATH_CALUDE_smallest_student_count_l2558_255808


namespace NUMINAMATH_CALUDE_house_number_painting_cost_l2558_255854

/-- Calculates the sum of digits for numbers in an arithmetic sequence --/
def sumOfDigits (start : ℕ) (diff : ℕ) (count : ℕ) : ℕ :=
  sorry

/-- Calculates the total cost of painting house numbers --/
def totalCost (southStart southDiff northStart northDiff housesPerSide : ℕ) : ℕ :=
  sorry

theorem house_number_painting_cost :
  totalCost 5 7 7 8 25 = 125 :=
sorry

end NUMINAMATH_CALUDE_house_number_painting_cost_l2558_255854


namespace NUMINAMATH_CALUDE_total_adoption_cost_l2558_255867

def cat_cost : ℕ := 50
def adult_dog_cost : ℕ := 100
def puppy_cost : ℕ := 150
def num_cats : ℕ := 2
def num_adult_dogs : ℕ := 3
def num_puppies : ℕ := 2

theorem total_adoption_cost :
  cat_cost * num_cats + adult_dog_cost * num_adult_dogs + puppy_cost * num_puppies = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_adoption_cost_l2558_255867


namespace NUMINAMATH_CALUDE_cubic_factor_implies_c_zero_l2558_255853

/-- Represents a cubic polynomial of the form ax^3 + bx^2 + cx + d -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a quadratic polynomial of the form ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a linear polynomial of the form ax + b -/
structure LinearPolynomial where
  a : ℝ
  b : ℝ

def has_factor (p : CubicPolynomial) (q : QuadraticPolynomial) : Prop :=
  ∃ l : LinearPolynomial, 
    p.a * (q.a * l.a) = p.a ∧
    p.b * (q.a * l.b + q.b * l.a) = p.b ∧
    p.c * (q.b * l.b + q.c * l.a) = p.c ∧
    p.d * (q.c * l.b) = p.d

theorem cubic_factor_implies_c_zero 
  (p : CubicPolynomial) 
  (h : p.a = 3 ∧ p.b = 0 ∧ p.d = 12) 
  (q : QuadraticPolynomial) 
  (hq : q.a = 1 ∧ q.c = 2) 
  (h_factor : has_factor p q) : 
  p.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factor_implies_c_zero_l2558_255853


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_unique_solution_l2558_255879

theorem binomial_coefficient_equation_unique_solution : 
  ∃! n : ℕ, Nat.choose 25 n + Nat.choose 25 12 = Nat.choose 26 13 ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_unique_solution_l2558_255879


namespace NUMINAMATH_CALUDE_crackers_per_box_l2558_255886

theorem crackers_per_box (darren_boxes calvin_boxes total_crackers : ℕ) : 
  darren_boxes = 4 →
  calvin_boxes = 2 * darren_boxes - 1 →
  total_crackers = 264 →
  (darren_boxes + calvin_boxes) * (total_crackers / (darren_boxes + calvin_boxes)) = total_crackers →
  total_crackers / (darren_boxes + calvin_boxes) = 24 := by
sorry

end NUMINAMATH_CALUDE_crackers_per_box_l2558_255886


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l2558_255833

theorem triangle_perimeter_bound : 
  ∀ (a b c : ℝ), 
    a = 7 → 
    b = 21 → 
    a + b > c → 
    a + c > b → 
    b + c > a → 
    a + b + c < 56 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l2558_255833


namespace NUMINAMATH_CALUDE_box_of_balls_l2558_255811

theorem box_of_balls (N : ℕ) : N - 44 = 70 - N → N = 57 := by
  sorry

end NUMINAMATH_CALUDE_box_of_balls_l2558_255811


namespace NUMINAMATH_CALUDE_complex_root_implies_positive_triangle_l2558_255843

theorem complex_root_implies_positive_triangle (a b c α β : ℝ) :
  α > 0 →
  β ≠ 0 →
  Complex.I ^ 2 = -1 →
  (α + β * Complex.I) ^ 2 - (a + b + c) * (α + β * Complex.I) + (a * b + b * c + c * a) = 0 →
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  Real.sqrt a + Real.sqrt b > Real.sqrt c ∧
  Real.sqrt b + Real.sqrt c > Real.sqrt a ∧
  Real.sqrt c + Real.sqrt a > Real.sqrt b :=
by sorry

end NUMINAMATH_CALUDE_complex_root_implies_positive_triangle_l2558_255843


namespace NUMINAMATH_CALUDE_game_of_thrones_percentage_l2558_255891

/-- Represents the vote counts for each book --/
structure VoteCounts where
  gameOfThrones : ℕ
  twilight : ℕ
  artOfTheDeal : ℕ

/-- Calculates the altered vote counts after tampering --/
def alteredVotes (original : VoteCounts) : VoteCounts :=
  { gameOfThrones := original.gameOfThrones,
    twilight := original.twilight / 2,
    artOfTheDeal := original.artOfTheDeal / 5 }

/-- Calculates the total number of altered votes --/
def totalAlteredVotes (altered : VoteCounts) : ℕ :=
  altered.gameOfThrones + altered.twilight + altered.artOfTheDeal

/-- Theorem: The percentage of altered votes for Game of Thrones is 50% --/
theorem game_of_thrones_percentage (original : VoteCounts)
  (h1 : original.gameOfThrones = 10)
  (h2 : original.twilight = 12)
  (h3 : original.artOfTheDeal = 20) :
  (alteredVotes original).gameOfThrones * 100 / (totalAlteredVotes (alteredVotes original)) = 50 := by
  sorry


end NUMINAMATH_CALUDE_game_of_thrones_percentage_l2558_255891


namespace NUMINAMATH_CALUDE_function_non_negative_implies_a_range_l2558_255883

theorem function_non_negative_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, x^2 - 4*x + a ≥ 0) → a ∈ Set.Ici 3 :=
by sorry

end NUMINAMATH_CALUDE_function_non_negative_implies_a_range_l2558_255883


namespace NUMINAMATH_CALUDE_number_is_nine_l2558_255805

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def xiao_qian_statement (n : ℕ) : Prop := is_perfect_square n ∧ n < 5

def xiao_lu_statement (n : ℕ) : Prop := n < 7 ∧ n ≥ 10

def xiao_dai_statement (n : ℕ) : Prop := is_perfect_square n ∧ n ≥ 5

def one_all_true (n : ℕ) : Prop :=
  (xiao_qian_statement n) ∨ (xiao_lu_statement n) ∨ (xiao_dai_statement n)

def one_all_false (n : ℕ) : Prop :=
  (¬xiao_qian_statement n) ∨ (¬xiao_lu_statement n) ∨ (¬xiao_dai_statement n)

def one_true_one_false (n : ℕ) : Prop :=
  (is_perfect_square n ∧ ¬(n < 5)) ∨
  ((n < 7) ∧ ¬(n ≥ 10)) ∨
  ((is_perfect_square n) ∧ ¬(n ≥ 5))

theorem number_is_nine :
  ∃ n : ℕ, n ≥ 1 ∧ n ≤ 99 ∧ one_all_true n ∧ one_all_false n ∧ one_true_one_false n ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_number_is_nine_l2558_255805


namespace NUMINAMATH_CALUDE_relay_team_permutations_l2558_255812

theorem relay_team_permutations (n : ℕ) (k : ℕ) :
  n = 5 → k = 3 → Nat.factorial k = 6 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_permutations_l2558_255812


namespace NUMINAMATH_CALUDE_sum_first_last_number_l2558_255817

theorem sum_first_last_number (a b c d : ℝ) : 
  (a + b + c) / 3 = 6 →
  (b + c + d) / 3 = 5 →
  d = 4 →
  a + d = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_first_last_number_l2558_255817


namespace NUMINAMATH_CALUDE_find_m_l2558_255875

theorem find_m : ∃ m : ℝ, ∀ x y : ℝ, (2*x + y)*(x - 2*y) = 2*x^2 - m*x*y - 2*y^2 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2558_255875


namespace NUMINAMATH_CALUDE_square_inequality_equivalence_l2558_255847

theorem square_inequality_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a > b ↔ a^2 > b^2 := by sorry

end NUMINAMATH_CALUDE_square_inequality_equivalence_l2558_255847
