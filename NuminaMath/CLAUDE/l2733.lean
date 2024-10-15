import Mathlib

namespace NUMINAMATH_CALUDE_min_sum_squares_l2733_273363

def S : Finset Int := {-6, -4, -3, -1, 1, 3, 5, 7}

theorem min_sum_squares (p q r s t u v w : Int)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2733_273363


namespace NUMINAMATH_CALUDE_prob_two_heads_and_three_l2733_273399

-- Define the probability of getting heads on a fair coin
def prob_heads : ℚ := 1/2

-- Define the probability of rolling a 3 on a fair six-sided die
def prob_three : ℚ := 1/6

-- State the theorem
theorem prob_two_heads_and_three (h1 : prob_heads = 1/2) (h2 : prob_three = 1/6) : 
  prob_heads * prob_heads * prob_three = 1/24 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_heads_and_three_l2733_273399


namespace NUMINAMATH_CALUDE_student_marks_l2733_273336

/-- Calculate the total marks secured in an exam given the following conditions:
  * total_questions: The total number of questions in the exam
  * correct_answers: The number of questions answered correctly
  * marks_per_correct: The number of marks awarded for each correct answer
  * marks_lost_per_wrong: The number of marks lost for each wrong answer
-/
def calculate_total_marks (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : ℤ :=
  (correct_answers * marks_per_correct : ℤ) - 
  ((total_questions - correct_answers) * marks_lost_per_wrong)

/-- Theorem stating that under the given conditions, the student secures 160 marks -/
theorem student_marks : 
  calculate_total_marks 60 44 4 1 = 160 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_l2733_273336


namespace NUMINAMATH_CALUDE_production_days_calculation_l2733_273384

theorem production_days_calculation (n : ℕ) 
  (h1 : (n * 50 : ℝ) / n = 50)  -- Average of 50 units for n days
  (h2 : ((n * 50 + 105 : ℝ) / (n + 1) = 55)) -- New average of 55 units including today
  : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_production_days_calculation_l2733_273384


namespace NUMINAMATH_CALUDE_cost_per_metre_l2733_273373

theorem cost_per_metre (total_length : ℝ) (total_cost : ℝ) (h1 : total_length = 9.25) (h2 : total_cost = 434.75) :
  total_cost / total_length = 47 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_metre_l2733_273373


namespace NUMINAMATH_CALUDE_remaining_water_bottles_l2733_273333

/-- Calculates the number of remaining water bottles after a soccer match --/
theorem remaining_water_bottles (initial_bottles : ℕ) 
  (first_break_players : ℕ) (first_break_bottles_per_player : ℕ)
  (second_break_players : ℕ) (second_break_extra_bottles : ℕ)
  (third_break_players : ℕ) : 
  initial_bottles = 5 * 12 →
  first_break_players = 11 →
  first_break_bottles_per_player = 2 →
  second_break_players = 14 →
  second_break_extra_bottles = 4 →
  third_break_players = 12 →
  initial_bottles - 
  (first_break_players * first_break_bottles_per_player +
   second_break_players + second_break_extra_bottles +
   third_break_players) = 8 := by
sorry

end NUMINAMATH_CALUDE_remaining_water_bottles_l2733_273333


namespace NUMINAMATH_CALUDE_exists_functions_satisfying_equations_l2733_273330

/-- A function defined on non-zero real numbers -/
def NonZeroRealFunction := {f : ℝ → ℝ // ∀ x ≠ 0, f x ≠ 0}

/-- The property that f and g satisfy the given equations -/
def SatisfiesEquations (f g : NonZeroRealFunction) : Prop :=
  ∀ x ≠ 0, f.val x + g.val (1/x) = x ∧ g.val x + f.val (1/x) = 1/x

theorem exists_functions_satisfying_equations :
  ∃ f g : NonZeroRealFunction, SatisfiesEquations f g ∧ f.val 1 = 1/2 ∧ g.val 1 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_exists_functions_satisfying_equations_l2733_273330


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2733_273368

/-- Given a geometric sequence {a_n} where a₁ = 3 and a₄ = 24, prove that a₃ + a₄ + a₅ = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →
  a 4 = 24 →
  a 3 + a 4 + a 5 = 84 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2733_273368


namespace NUMINAMATH_CALUDE_fraction_change_l2733_273319

theorem fraction_change (a b k : ℕ+) :
  (a : ℚ) / b < 1 → (a + k : ℚ) / (b + k) > a / b ∧
  (a : ℚ) / b > 1 → (a + k : ℚ) / (b + k) < a / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_change_l2733_273319


namespace NUMINAMATH_CALUDE_final_price_calculation_l2733_273328

/-- Calculate the final price of a shirt and pants after a series of price changes -/
theorem final_price_calculation (S P : ℝ) :
  let shirt_price_1 := S * 1.20
  let pants_price_1 := P * 0.90
  let combined_price_1 := shirt_price_1 + pants_price_1
  let combined_price_2 := combined_price_1 * 1.15
  let final_price := combined_price_2 * 0.95
  final_price = 1.311 * S + 0.98325 * P := by sorry

end NUMINAMATH_CALUDE_final_price_calculation_l2733_273328


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l2733_273346

def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem z_in_third_quadrant :
  let z : ℂ := -1 - 2*I
  is_in_third_quadrant z :=
by sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l2733_273346


namespace NUMINAMATH_CALUDE_dye_mixture_volume_l2733_273321

theorem dye_mixture_volume : 
  let water_volume : ℚ := 20 * (3/5)
  let vinegar_volume : ℚ := 18 * (5/6)
  water_volume + vinegar_volume = 27
  := by sorry

end NUMINAMATH_CALUDE_dye_mixture_volume_l2733_273321


namespace NUMINAMATH_CALUDE_unique_solution_for_quadratic_difference_l2733_273388

theorem unique_solution_for_quadratic_difference (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃! (a b : ℝ), ∀ x : ℝ, (x + m)^2 - (x + n)^2 = (m - n)^2 → x = a * m + b * n ∧ a = 0 ∧ b ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_quadratic_difference_l2733_273388


namespace NUMINAMATH_CALUDE_two_x_minus_y_value_l2733_273345

theorem two_x_minus_y_value (x y : ℝ) 
  (hx : |x| = 2) 
  (hy : |y| = 3) 
  (hxy : x / y < 0) : 
  2 * x - y = 7 ∨ 2 * x - y = -7 := by
sorry

end NUMINAMATH_CALUDE_two_x_minus_y_value_l2733_273345


namespace NUMINAMATH_CALUDE_characterization_of_f_l2733_273352

-- Define the property of being strictly increasing
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the functional equation
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (f x + y) = f (x + y) + f 0

-- Theorem statement
theorem characterization_of_f :
  ∀ f : ℝ → ℝ, StrictlyIncreasing f → SatisfiesEquation f →
  ∃ k : ℝ, ∀ x, f x = x - k :=
sorry

end NUMINAMATH_CALUDE_characterization_of_f_l2733_273352


namespace NUMINAMATH_CALUDE_inverse_tangent_sum_l2733_273315

theorem inverse_tangent_sum : Real.arctan (1/2) + Real.arctan (1/5) + Real.arctan (1/8) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_tangent_sum_l2733_273315


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_l2733_273316

/-- A quadratic function with parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - (k + 1) * x + k + 4

/-- The discriminant of the quadratic function -/
def discriminant (k : ℝ) : ℝ := (k + 1)^2 - 4 * (k + 4)

/-- The function has two distinct zeros iff the discriminant is positive -/
def has_two_distinct_zeros (k : ℝ) : Prop := discriminant k > 0

/-- The range of k for which the function has two distinct zeros -/
def k_range (k : ℝ) : Prop := k < -3 ∨ k > 5

theorem quadratic_two_zeros (k : ℝ) : 
  has_two_distinct_zeros k ↔ k_range k := by sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_l2733_273316


namespace NUMINAMATH_CALUDE_john_chore_time_l2733_273398

/-- Given a ratio of cartoon watching time to chore time and the total cartoon watching time,
    calculate the required chore time. -/
def chore_time (cartoon_ratio : ℕ) (chore_ratio : ℕ) (total_cartoon_time : ℕ) : ℕ :=
  (chore_ratio * total_cartoon_time) / cartoon_ratio

theorem john_chore_time :
  let cartoon_ratio : ℕ := 10
  let chore_ratio : ℕ := 8
  let total_cartoon_time : ℕ := 120
  chore_time cartoon_ratio chore_ratio total_cartoon_time = 96 := by
  sorry

#eval chore_time 10 8 120

end NUMINAMATH_CALUDE_john_chore_time_l2733_273398


namespace NUMINAMATH_CALUDE_min_slope_tangent_line_l2733_273372

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

-- Theorem statement
theorem min_slope_tangent_line :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, f' x ≥ f' (-1)) ∧
    (f' (-1) = 3) ∧
    (f (-1) = -5) ∧
    (a * x + b * y + c = 0) ∧
    (a / b = f' (-1)) ∧
    ((-1) * a + (-5) * b + c = 0) ∧
    (a = 3 ∧ b = -1 ∧ c = -2) :=
by sorry

end NUMINAMATH_CALUDE_min_slope_tangent_line_l2733_273372


namespace NUMINAMATH_CALUDE_vector_expression_equality_l2733_273365

theorem vector_expression_equality : 
  let v1 : Fin 2 → ℝ := ![3, -4]
  let v2 : Fin 2 → ℝ := ![2, -3]
  let v3 : Fin 2 → ℝ := ![1, 6]
  v1 + 5 • v2 - v3 = ![12, -25] := by sorry

end NUMINAMATH_CALUDE_vector_expression_equality_l2733_273365


namespace NUMINAMATH_CALUDE_twenty_percent_of_twenty_l2733_273344

theorem twenty_percent_of_twenty : (20 : ℝ) / 100 * 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_of_twenty_l2733_273344


namespace NUMINAMATH_CALUDE_math_textbooks_in_one_box_l2733_273307

def total_textbooks : ℕ := 15
def math_textbooks : ℕ := 4
def num_boxes : ℕ := 3
def box_capacities : Fin 3 → ℕ := ![4, 5, 6]

def probability_all_math_in_one_box : ℚ := 1 / 143

theorem math_textbooks_in_one_box :
  let total_arrangements := (total_textbooks.choose (box_capacities 0)) * 
                            ((total_textbooks - box_capacities 0).choose (box_capacities 1)) * 
                            ((total_textbooks - box_capacities 0 - box_capacities 1).choose (box_capacities 2))
  let favorable_outcomes := (box_capacities 0).choose math_textbooks * 
                            ((total_textbooks - math_textbooks).choose (box_capacities 0 - math_textbooks)) * 
                            ((total_textbooks - box_capacities 0).choose (box_capacities 1)) +
                            (box_capacities 1).choose math_textbooks * 
                            ((total_textbooks - math_textbooks).choose (box_capacities 1 - math_textbooks)) * 
                            ((total_textbooks - box_capacities 1).choose (box_capacities 0)) +
                            (box_capacities 2).choose math_textbooks * 
                            ((total_textbooks - math_textbooks).choose (box_capacities 2 - math_textbooks)) * 
                            ((total_textbooks - box_capacities 2).choose (box_capacities 0))
  (favorable_outcomes : ℚ) / (total_arrangements : ℚ) = probability_all_math_in_one_box := by
  sorry

end NUMINAMATH_CALUDE_math_textbooks_in_one_box_l2733_273307


namespace NUMINAMATH_CALUDE_problem_proof_l2733_273300

theorem problem_proof : 289 + 2 * 17 * 8 + 64 = 625 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2733_273300


namespace NUMINAMATH_CALUDE_inequality_proof_l2733_273364

theorem inequality_proof (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2733_273364


namespace NUMINAMATH_CALUDE_remainder_3211_103_l2733_273395

theorem remainder_3211_103 : 3211 % 103 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3211_103_l2733_273395


namespace NUMINAMATH_CALUDE_N_is_positive_l2733_273311

theorem N_is_positive (a b : ℝ) : 
  let N := 4*a^2 - 12*a*b + 13*b^2 - 6*a + 4*b + 13
  0 < N := by sorry

end NUMINAMATH_CALUDE_N_is_positive_l2733_273311


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2733_273304

theorem polynomial_factorization (x : ℝ) :
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x - 1)^4 * (x + 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2733_273304


namespace NUMINAMATH_CALUDE_train_length_calculation_l2733_273362

/-- Given a train traveling at a certain speed that crosses a bridge of known length in a specific time, calculate the length of the train. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 215 →
  crossing_time = 30 →
  (train_speed * crossing_time) - bridge_length = 160 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l2733_273362


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2733_273377

/-- Given two perpendicular vectors a and b in ℝ², prove that m = 2 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) 
  (ha : a = (-2, 3)) (hb : b = (3, m)) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2733_273377


namespace NUMINAMATH_CALUDE_one_third_of_seven_point_two_l2733_273382

theorem one_third_of_seven_point_two :
  (7.2 : ℚ) / 3 = 2 + 2 / 5 := by sorry

end NUMINAMATH_CALUDE_one_third_of_seven_point_two_l2733_273382


namespace NUMINAMATH_CALUDE_substitution_result_l2733_273390

theorem substitution_result (x y : ℝ) :
  (y = x - 1) ∧ (x - 2*y = 7) → (x - 2*x + 2 = 7) := by sorry

end NUMINAMATH_CALUDE_substitution_result_l2733_273390


namespace NUMINAMATH_CALUDE_ellipse_equation_l2733_273343

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  c : ℝ  -- Half of focal distance
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_c_less_a : c < a
  h_pythagoras : a^2 = b^2 + c^2

/-- The standard form equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation (e : Ellipse)
  (h_sum : e.a + e.b = 5)  -- Half of the sum of axes lengths
  (h_focal : e.c = 2 * Real.sqrt 5) :
  standard_equation e = fun x y ↦ x^2 / 36 + y^2 / 16 = 1 := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ellipse_equation_l2733_273343


namespace NUMINAMATH_CALUDE_new_cost_percentage_l2733_273334

/-- The cost function -/
def cost (t c a x : ℝ) (n : ℕ) : ℝ := t * c * (a * x) ^ n

/-- Theorem stating the relationship between the original and new cost -/
theorem new_cost_percentage (t c a x : ℝ) (n : ℕ) :
  let O := cost t c a x n
  let E := cost t (2*c) (2*a) x (n+2)
  E = 2^(n+1) * x^2 * O :=
by sorry

end NUMINAMATH_CALUDE_new_cost_percentage_l2733_273334


namespace NUMINAMATH_CALUDE_collinear_points_imply_a_equals_four_l2733_273353

-- Define the points
def A : ℝ × ℝ := (2, 2)
def B : ℝ → ℝ × ℝ := λ a => (a, 0)
def C : ℝ × ℝ := (0, 4)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, q.1 - p.1 = t * (r.1 - p.1) ∧ q.2 - p.2 = t * (r.2 - p.2)

-- Theorem statement
theorem collinear_points_imply_a_equals_four :
  ∀ a : ℝ, collinear A (B a) C → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_imply_a_equals_four_l2733_273353


namespace NUMINAMATH_CALUDE_gasoline_price_increase_l2733_273302

theorem gasoline_price_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (price_increase : ℝ) 
  (budget_increase : ℝ) 
  (quantity_decrease : ℝ) 
  (h1 : budget_increase = 0.15)
  (h2 : quantity_decrease = 0.08000000000000007)
  (h3 : original_price > 0)
  (h4 : original_quantity > 0) :
  original_price * original_quantity * (1 + budget_increase) = 
  (original_price * (1 + price_increase)) * (original_quantity * (1 - quantity_decrease)) →
  price_increase = 0.25 := by
sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_l2733_273302


namespace NUMINAMATH_CALUDE_inequality_proof_l2733_273358

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2733_273358


namespace NUMINAMATH_CALUDE_circle_ratio_l2733_273342

theorem circle_ratio (R r c d : ℝ) (h1 : R > r) (h2 : r > 0) (h3 : c > 0) (h4 : d > 0) :
  π * R^2 = (c / d) * (π * R^2 - π * r^2 + 2 * (2 * r^2)) →
  R / r = Real.sqrt (c * (4 - π) / (d * π - c * π)) :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_l2733_273342


namespace NUMINAMATH_CALUDE_number_equation_solution_l2733_273359

theorem number_equation_solution : 
  ∃ x : ℝ, (10 * x = 2 * x - 36) ∧ (x = -4.5) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2733_273359


namespace NUMINAMATH_CALUDE_equation_solution_range_l2733_273322

-- Define the equation
def equation (x k : ℝ) : Prop :=
  (x^2 + k*x + 3) / (x - 1) = 3*x + k

-- Define the condition for exactly one positive real solution
def has_one_positive_solution (k : ℝ) : Prop :=
  ∃! x : ℝ, x > 0 ∧ equation x k

-- Theorem statement
theorem equation_solution_range :
  ∀ k : ℝ, has_one_positive_solution k ↔ (k = -33/8 ∨ k = -4 ∨ k ≥ -3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2733_273322


namespace NUMINAMATH_CALUDE_shared_divisors_count_l2733_273357

theorem shared_divisors_count (a b : ℕ) (ha : a = 9240) (hb : b = 8820) :
  (Finset.filter (fun d : ℕ => d ∣ a ∧ d ∣ b) (Finset.range (min a b + 1))).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_shared_divisors_count_l2733_273357


namespace NUMINAMATH_CALUDE_proposition_p_or_q_is_true_l2733_273320

open Real

theorem proposition_p_or_q_is_true :
  (∀ x > 0, exp x > 1 + x) ∨
  (∀ f : ℝ → ℝ, (∀ x, f x + 2 = -(f (-x) + 2)) → 
   ∀ x, f (x - 0) + 0 = f (-(x - 0)) + 4) := by sorry

end NUMINAMATH_CALUDE_proposition_p_or_q_is_true_l2733_273320


namespace NUMINAMATH_CALUDE_outside_bookshop_discount_l2733_273341

/-- The price of a math textbook in the school bookshop -/
def school_price : ℝ := 45

/-- The amount Peter saves by buying 3 math textbooks from outside bookshops -/
def savings : ℝ := 27

/-- The number of textbooks Peter buys -/
def num_textbooks : ℕ := 3

/-- The percentage discount offered by outside bookshops -/
def discount_percentage : ℝ := 20

theorem outside_bookshop_discount :
  let outside_price := school_price - (savings / num_textbooks)
  discount_percentage = (school_price - outside_price) / school_price * 100 :=
by sorry

end NUMINAMATH_CALUDE_outside_bookshop_discount_l2733_273341


namespace NUMINAMATH_CALUDE_shortest_distance_between_inscribed_circles_shortest_distance_proof_l2733_273380

/-- The shortest distance between two circles inscribed in two of nine identical squares 
    (each with side length 1) that form a larger square -/
theorem shortest_distance_between_inscribed_circles : ℝ :=
  let large_square_side : ℝ := 3
  let small_square_side : ℝ := 1
  let num_small_squares : ℕ := 9
  let circle_radius : ℝ := small_square_side / 2
  2 * Real.sqrt 2 - 1

/-- Proof of the shortest distance between the inscribed circles -/
theorem shortest_distance_proof :
  shortest_distance_between_inscribed_circles = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_between_inscribed_circles_shortest_distance_proof_l2733_273380


namespace NUMINAMATH_CALUDE_tank_filling_ratio_l2733_273340

/-- Proves that the ratio of initial water to total capacity is 1/2 given specific tank conditions -/
theorem tank_filling_ratio 
  (capacity : ℝ) 
  (inflow_rate : ℝ) 
  (outflow_rate1 : ℝ) 
  (outflow_rate2 : ℝ) 
  (fill_time : ℝ) 
  (h1 : capacity = 10) 
  (h2 : inflow_rate = 0.5) 
  (h3 : outflow_rate1 = 0.25) 
  (h4 : outflow_rate2 = 1/6) 
  (h5 : fill_time = 60) : 
  (capacity - (inflow_rate - outflow_rate1 - outflow_rate2) * fill_time) / capacity = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_ratio_l2733_273340


namespace NUMINAMATH_CALUDE_first_number_is_202_l2733_273367

def numbers : List ℕ := [202, 204, 205, 206, 209, 209, 210, 212]

theorem first_number_is_202 (x : ℕ) 
  (h : (numbers.sum + x) / 9 = 207) : 
  numbers.head? = some 202 := by
  sorry

end NUMINAMATH_CALUDE_first_number_is_202_l2733_273367


namespace NUMINAMATH_CALUDE_race_probability_l2733_273370

theorem race_probability (p_x p_y p_z : ℝ) : 
  p_x = 1/7 →
  p_y = 1/3 →
  p_x + p_y + p_z = 0.6761904761904762 →
  p_z = 0.2 := by
sorry

end NUMINAMATH_CALUDE_race_probability_l2733_273370


namespace NUMINAMATH_CALUDE_shoeing_time_proof_l2733_273331

/-- Calculates the minimum time required for blacksmiths to shoe horses -/
def minimum_shoeing_time (num_blacksmiths : ℕ) (num_horses : ℕ) (time_per_shoe : ℕ) : ℕ :=
  let total_hooves := num_horses * 4
  let total_time := total_hooves * time_per_shoe
  total_time / num_blacksmiths

/-- Proves that the minimum time for 48 blacksmiths to shoe 60 horses is 25 minutes -/
theorem shoeing_time_proof :
  minimum_shoeing_time 48 60 5 = 25 := by
  sorry

#eval minimum_shoeing_time 48 60 5

end NUMINAMATH_CALUDE_shoeing_time_proof_l2733_273331


namespace NUMINAMATH_CALUDE_initial_gum_pieces_l2733_273389

theorem initial_gum_pieces (x : ℕ) : x + 16 + 20 = 61 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_initial_gum_pieces_l2733_273389


namespace NUMINAMATH_CALUDE_cloud_counting_l2733_273360

theorem cloud_counting (carson_clouds : ℕ) (brother_multiplier : ℕ) : 
  carson_clouds = 6 → 
  brother_multiplier = 3 → 
  carson_clouds + carson_clouds * brother_multiplier = 24 :=
by sorry

end NUMINAMATH_CALUDE_cloud_counting_l2733_273360


namespace NUMINAMATH_CALUDE_projection_incircle_inequality_l2733_273326

/-- Represents a right triangle with legs a and b, hypotenuse c, projections p and q, and incircle radii ρ_a and ρ_b -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ
  q : ℝ
  ρ_a : ℝ
  ρ_b : ℝ
  h_right : a^2 + b^2 = c^2
  h_a_lt_b : a < b
  h_p_proj : p * c = a^2
  h_q_proj : q * c = b^2
  h_ρ_a_def : ρ_a * (a + c - b) = a * b
  h_ρ_b_def : ρ_b * (b + c - a) = a * b

/-- Theorem stating the inequalities for projections and incircle radii in a right triangle -/
theorem projection_incircle_inequality (t : RightTriangle) : t.p < t.ρ_a ∧ t.q > t.ρ_b := by
  sorry

end NUMINAMATH_CALUDE_projection_incircle_inequality_l2733_273326


namespace NUMINAMATH_CALUDE_unique_three_digit_perfect_square_product_l2733_273324

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that converts a three-digit number to its cyclic permutations -/
def cyclic_permutations (n : ℕ) : Fin 3 → ℕ
| 0 => n
| 1 => (n % 100) * 10 + n / 100
| 2 => (n % 10) * 100 + n / 10

/-- The main theorem stating that 243 is the only three-digit number satisfying the given conditions -/
theorem unique_three_digit_perfect_square_product :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧
  (n / 100 ≤ n / 10 % 10 ∧ n / 100 ≤ n % 10) ∧
  is_perfect_square (cyclic_permutations n 0 * cyclic_permutations n 1 * cyclic_permutations n 2) ∧
  n = 243 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_perfect_square_product_l2733_273324


namespace NUMINAMATH_CALUDE_least_integer_square_48_more_than_double_l2733_273339

theorem least_integer_square_48_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 48 ∧ ∀ y : ℤ, y^2 = 2*y + 48 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_48_more_than_double_l2733_273339


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_l2733_273366

theorem similar_triangles_perimeter (h_small h_large p_small : ℝ) : 
  h_small / h_large = 3 / 5 →
  p_small = 12 →
  ∃ p_large : ℝ, p_large = 20 ∧ p_small / p_large = h_small / h_large :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_l2733_273366


namespace NUMINAMATH_CALUDE_order_of_logarithms_and_root_l2733_273348

theorem order_of_logarithms_and_root (a b c : ℝ) : 
  a = 2 * Real.log 0.99 → 
  b = Real.log 0.98 → 
  c = Real.sqrt 0.96 - 1 → 
  c < b ∧ b < a := by
sorry

end NUMINAMATH_CALUDE_order_of_logarithms_and_root_l2733_273348


namespace NUMINAMATH_CALUDE_variance_invariant_under_translation_regression_line_minimizes_squared_residuals_sum_residuals_zero_l2733_273379

-- Define a data set as a list of real numbers
def DataSet := List ℝ

-- Define the sample variance
def sampleVariance (data : DataSet) : ℝ := sorry

-- Define a function to subtract a constant from each data point
def subtractConstant (data : DataSet) (c : ℝ) : DataSet := sorry

-- Define a type for a regression line
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

-- Define a function to calculate residuals
def residuals (data : DataSet) (line : RegressionLine) : DataSet := sorry

-- Define a function to calculate the sum of squared residuals
def sumSquaredResiduals (data : DataSet) (line : RegressionLine) : ℝ := sorry

-- Define a function to find the least squares regression line
def leastSquaresRegressionLine (data : DataSet) : RegressionLine := sorry

-- Theorem 1: Subtracting a constant doesn't change the sample variance
theorem variance_invariant_under_translation (data : DataSet) (c : ℝ) :
  sampleVariance (subtractConstant data c) = sampleVariance data := by sorry

-- Theorem 2: The regression line minimizes the sum of squared residuals
theorem regression_line_minimizes_squared_residuals (data : DataSet) :
  ∀ line : RegressionLine,
    sumSquaredResiduals data (leastSquaresRegressionLine data) ≤ sumSquaredResiduals data line := by sorry

-- Theorem 3: The sum of residuals for the least squares regression line is zero
theorem sum_residuals_zero (data : DataSet) :
  (residuals data (leastSquaresRegressionLine data)).sum = 0 := by sorry

end NUMINAMATH_CALUDE_variance_invariant_under_translation_regression_line_minimizes_squared_residuals_sum_residuals_zero_l2733_273379


namespace NUMINAMATH_CALUDE_square_sum_theorem_l2733_273329

theorem square_sum_theorem (a b : ℕ) 
  (h1 : ∃ k : ℕ, a * b = k^2)
  (h2 : ∃ m : ℕ, (2 * a + 1) * (2 * b + 1) = m^2) :
  ∃ n : ℕ, 
    n % 2 = 0 ∧ 
    n > 2 ∧ 
    ∃ p : ℕ, (a + n) * (b + n) = p^2 :=
by sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l2733_273329


namespace NUMINAMATH_CALUDE_tan_30_plus_3sin_30_l2733_273349

theorem tan_30_plus_3sin_30 :
  Real.tan (30 * Real.pi / 180) + 3 * Real.sin (30 * Real.pi / 180) = (2 * Real.sqrt 3 + 9) / 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_plus_3sin_30_l2733_273349


namespace NUMINAMATH_CALUDE_number_of_colors_l2733_273350

/-- A crayon factory produces crayons of different colors. -/
structure CrayonFactory where
  /-- Number of crayons of each color in a box -/
  crayons_per_color_per_box : ℕ
  /-- Number of boxes filled per hour -/
  boxes_per_hour : ℕ
  /-- Total number of crayons produced in 4 hours -/
  total_crayons_4hours : ℕ

/-- Theorem stating the number of colors produced by the factory -/
theorem number_of_colors (factory : CrayonFactory)
    (h1 : factory.crayons_per_color_per_box = 2)
    (h2 : factory.boxes_per_hour = 5)
    (h3 : factory.total_crayons_4hours = 160) :
    (factory.total_crayons_4hours / (4 * factory.boxes_per_hour * factory.crayons_per_color_per_box) : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_colors_l2733_273350


namespace NUMINAMATH_CALUDE_operation_result_is_four_digit_l2733_273355

/-- A nonzero digit is a natural number between 1 and 9, inclusive. -/
def NonzeroDigit : Type :=
  { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

/-- The result of the operation 543C + 721 - DE4 for any nonzero digits C, D, and E. -/
def OperationResult (C D E : NonzeroDigit) : ℕ :=
  5430 + C.val + 721 - (100 * D.val + 10 * E.val + 4)

/-- The theorem stating that the result of the operation is always a 4-digit number. -/
theorem operation_result_is_four_digit (C D E : NonzeroDigit) :
  1000 ≤ OperationResult C D E ∧ OperationResult C D E < 10000 :=
by sorry

end NUMINAMATH_CALUDE_operation_result_is_four_digit_l2733_273355


namespace NUMINAMATH_CALUDE_math_team_selection_count_l2733_273309

-- Define the number of boys and girls in the math club
def num_boys : ℕ := 10
def num_girls : ℕ := 10

-- Define the required number of boys and girls in the team
def required_boys : ℕ := 4
def required_girls : ℕ := 3

-- Define the total team size
def team_size : ℕ := required_boys + required_girls

-- Theorem statement
theorem math_team_selection_count :
  (Nat.choose num_boys required_boys) * (Nat.choose num_girls required_girls) = 25200 := by
  sorry

end NUMINAMATH_CALUDE_math_team_selection_count_l2733_273309


namespace NUMINAMATH_CALUDE_sin_arccos_twelve_thirteenths_l2733_273306

theorem sin_arccos_twelve_thirteenths : 
  Real.sin (Real.arccos (12 / 13)) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_sin_arccos_twelve_thirteenths_l2733_273306


namespace NUMINAMATH_CALUDE_cubic_sum_of_quadratic_roots_l2733_273361

theorem cubic_sum_of_quadratic_roots :
  ∀ x₁ x₂ : ℝ,
  (x₁^2 + 4*x₁ + 2 = 0) →
  (x₂^2 + 4*x₂ + 2 = 0) →
  (x₁ ≠ x₂) →
  x₁^3 + 14*x₂ + 55 = 7 :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_of_quadratic_roots_l2733_273361


namespace NUMINAMATH_CALUDE_vins_bike_distance_l2733_273378

/-- Calculates the total distance ridden in a week given daily distances and number of days -/
def total_distance (to_school : ℕ) (from_school : ℕ) (days : ℕ) : ℕ :=
  (to_school + from_school) * days

/-- Proves that given the specific distances and number of days, the total distance is 65 miles -/
theorem vins_bike_distance : total_distance 6 7 5 = 65 := by
  sorry

end NUMINAMATH_CALUDE_vins_bike_distance_l2733_273378


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l2733_273392

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def reverse_number (h t u : ℕ) : ℕ := u * 100 + t * 10 + h

theorem three_digit_number_problem (h t u : ℕ) :
  is_single_digit h ∧
  is_single_digit t ∧
  u = h + 6 ∧
  u + h = 16 ∧
  (h * 100 + t * 10 + u + reverse_number h t u) % 10 = 6 ∧
  ((h * 100 + t * 10 + u + reverse_number h t u) / 10) % 10 = 9 →
  h = 5 ∧ t = 5 ∧ u = 11 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l2733_273392


namespace NUMINAMATH_CALUDE_paul_dog_food_needed_l2733_273313

/-- The amount of dog food needed per day for a given weight in pounds -/
def dogFoodNeeded (weight : ℕ) : ℕ := weight / 10

/-- The total weight of Paul's dogs in pounds -/
def totalDogWeight : ℕ := 20 + 40 + 10 + 30 + 50

/-- Theorem: Paul needs 15 pounds of dog food per day for his five dogs -/
theorem paul_dog_food_needed : dogFoodNeeded totalDogWeight = 15 := by
  sorry

end NUMINAMATH_CALUDE_paul_dog_food_needed_l2733_273313


namespace NUMINAMATH_CALUDE_intersection_M_N_l2733_273376

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≠ x}

theorem intersection_M_N : M ∩ N = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2733_273376


namespace NUMINAMATH_CALUDE_fitness_club_comparison_l2733_273354

/-- Represents a fitness club with a monthly subscription cost -/
structure FitnessClub where
  name : String
  monthlyCost : ℕ

/-- Calculates the yearly cost for a given number of months -/
def yearlyCost (club : FitnessClub) (months : ℕ) : ℕ :=
  club.monthlyCost * months

/-- Calculates the cost per visit given total cost and number of visits -/
def costPerVisit (totalCost : ℕ) (visits : ℕ) : ℚ :=
  totalCost / visits

/-- Represents the two attendance patterns -/
inductive AttendancePattern
  | Regular
  | MoodBased

/-- Calculates the number of visits per year based on the attendance pattern -/
def visitsPerYear (pattern : AttendancePattern) : ℕ :=
  match pattern with
  | .Regular => 96
  | .MoodBased => 56

theorem fitness_club_comparison (alpha beta : FitnessClub) 
    (h_alpha : alpha.monthlyCost = 999)
    (h_beta : beta.monthlyCost = 1299) :
    (∀ (pattern : AttendancePattern), 
      costPerVisit (yearlyCost alpha 12) (visitsPerYear pattern) < 
      costPerVisit (yearlyCost beta 12) (visitsPerYear pattern)) ∧
    (costPerVisit (yearlyCost alpha 12) (visitsPerYear .MoodBased) > 
     costPerVisit (yearlyCost beta 8) (visitsPerYear .MoodBased)) := by
  sorry

#check fitness_club_comparison

end NUMINAMATH_CALUDE_fitness_club_comparison_l2733_273354


namespace NUMINAMATH_CALUDE_fifth_set_fraction_approx_three_fourths_l2733_273396

-- Define the duration of the whole match in minutes
def whole_match_duration : ℕ := 665

-- Define the duration of the fifth set in minutes
def fifth_set_duration : ℕ := 491

-- Define a function to calculate the fraction
def match_fraction : ℚ := fifth_set_duration / whole_match_duration

-- Define what we consider as "approximately equal" (e.g., within 0.02)
def approximately_equal (x y : ℚ) : Prop := abs (x - y) < 1/50

-- Theorem statement
theorem fifth_set_fraction_approx_three_fourths :
  approximately_equal match_fraction (3/4) :=
sorry

end NUMINAMATH_CALUDE_fifth_set_fraction_approx_three_fourths_l2733_273396


namespace NUMINAMATH_CALUDE_total_rabbits_caught_l2733_273301

/-- Represents the number of rabbits caught on a given day -/
def rabbits_caught (day : ℕ) : ℕ :=
  203 - 3 * day

/-- Represents the number of squirrels caught on a given day -/
def squirrels_caught (day : ℕ) : ℕ :=
  16 + 2 * day

/-- The day when more squirrels are caught than rabbits -/
def crossover_day : ℕ :=
  38

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a : ℤ) (d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem total_rabbits_caught : 
  arithmetic_sum crossover_day 200 (-3) = 5491 := by
  sorry

end NUMINAMATH_CALUDE_total_rabbits_caught_l2733_273301


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2733_273325

/-- Given a hyperbola and related geometric conditions, prove its asymptotes. -/
theorem hyperbola_asymptotes (a b c : ℝ) (E F₁ F₂ D : ℝ × ℝ) :
  a > 0 ∧ b > 0 ∧
  (λ (x y : ℝ) => y^2 / a^2 - x^2 / b^2 = 1) E.1 E.2 ∧  -- E is on the hyperbola
  (λ (x y : ℝ) => 4*x^2 + 4*y^2 = b^2) D.1 D.2 ∧       -- D is on the circle
  (E.1 - F₁.1) * D.1 + (E.2 - F₁.2) * D.2 = 0 ∧       -- EF₁ is tangent to circle at D
  2 * D.1 = E.1 + F₁.1 ∧ 2 * D.2 = E.2 + F₁.2 →       -- D is midpoint of EF₁
  (λ (x y : ℝ) => x + 2*y = 0 ∨ x - 2*y = 0) E.1 E.2  -- Asymptotes equations
  := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2733_273325


namespace NUMINAMATH_CALUDE_max_profit_for_container_l2733_273332

/-- Represents the container and goods properties --/
structure Container :=
  (volume : ℝ)
  (weight_capacity : ℝ)
  (chemical_volume_per_ton : ℝ)
  (paper_volume_per_ton : ℝ)
  (chemical_profit_per_ton : ℝ)
  (paper_profit_per_ton : ℝ)

/-- Calculates the profit for a given allocation of goods --/
def profit (c : Container) (chemical_tons : ℝ) (paper_tons : ℝ) : ℝ :=
  c.chemical_profit_per_ton * chemical_tons + c.paper_profit_per_ton * paper_tons

/-- Checks if the allocation satisfies the container constraints --/
def is_valid_allocation (c : Container) (chemical_tons : ℝ) (paper_tons : ℝ) : Prop :=
  chemical_tons ≥ 0 ∧ paper_tons ≥ 0 ∧
  chemical_tons + paper_tons ≤ c.weight_capacity ∧
  c.chemical_volume_per_ton * chemical_tons + c.paper_volume_per_ton * paper_tons ≤ c.volume

/-- Theorem stating the maximum profit for the given container --/
theorem max_profit_for_container :
  ∃ (c : Container) (chemical_tons paper_tons : ℝ),
    c.volume = 12 ∧
    c.weight_capacity = 5 ∧
    c.chemical_volume_per_ton = 1 ∧
    c.paper_volume_per_ton = 3 ∧
    c.chemical_profit_per_ton = 100000 ∧
    c.paper_profit_per_ton = 200000 ∧
    is_valid_allocation c chemical_tons paper_tons ∧
    profit c chemical_tons paper_tons = 850000 ∧
    ∀ (other_chemical other_paper : ℝ),
      is_valid_allocation c other_chemical other_paper →
      profit c other_chemical other_paper ≤ 850000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_for_container_l2733_273332


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l2733_273310

/-- The number of ways to distribute n indistinguishable items among k distinguishable categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 5 indistinguishable items among 3 distinguishable categories is 21 -/
theorem ice_cream_combinations : distribute 5 3 = 21 := by sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l2733_273310


namespace NUMINAMATH_CALUDE_triangle_incenter_inequality_l2733_273335

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the intersection points of angle bisectors with opposite sides
def angleBisectorIntersection (t : Triangle) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_incenter_inequality (t : Triangle) :
  let I := incenter t
  let (A', B', C') := angleBisectorIntersection t
  let ratio := (distance I t.A * distance I t.B * distance I t.C) /
               (distance A' t.A * distance B' t.B * distance C' t.C)
  (1 / 4 : ℝ) < ratio ∧ ratio ≤ (8 / 27 : ℝ) := by sorry

end NUMINAMATH_CALUDE_triangle_incenter_inequality_l2733_273335


namespace NUMINAMATH_CALUDE_equal_numbers_product_l2733_273383

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 →
  a = 12 →
  b = 22 →
  c = d →
  c * d = 529 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l2733_273383


namespace NUMINAMATH_CALUDE_product_in_fourth_quadrant_l2733_273312

def complex_multiply (a b c d : ℝ) : ℂ :=
  Complex.mk (a * c - b * d) (a * d + b * c)

def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem product_in_fourth_quadrant :
  let z₁ : ℂ := Complex.mk 3 1
  let z₂ : ℂ := Complex.mk 1 (-1)
  let z : ℂ := complex_multiply z₁.re z₁.im z₂.re z₂.im
  fourth_quadrant z := by sorry

end NUMINAMATH_CALUDE_product_in_fourth_quadrant_l2733_273312


namespace NUMINAMATH_CALUDE_geometric_progression_existence_l2733_273375

theorem geometric_progression_existence : ∃ (a : ℕ → ℚ), 
  (∀ n, a (n + 1) = a n * (3/2)) ∧ 
  (a 1 = 2^99) ∧
  (∀ n, a (n + 1) > a n) ∧
  (∀ n ≤ 100, ∃ m : ℕ, a n = m) ∧
  (∀ n > 100, ∀ m : ℕ, a n ≠ m) := by
  sorry

#check geometric_progression_existence

end NUMINAMATH_CALUDE_geometric_progression_existence_l2733_273375


namespace NUMINAMATH_CALUDE_average_stamps_is_25_l2733_273374

/-- Calculates the average number of stamps collected per day -/
def average_stamps_collected (days : ℕ) (initial_stamps : ℕ) (daily_increase : ℕ) : ℚ :=
  let total_stamps := (days : ℚ) / 2 * (2 * initial_stamps + (days - 1) * daily_increase)
  total_stamps / days

/-- Proves that the average number of stamps collected per day is 25 -/
theorem average_stamps_is_25 :
  average_stamps_collected 6 10 6 = 25 := by
sorry

end NUMINAMATH_CALUDE_average_stamps_is_25_l2733_273374


namespace NUMINAMATH_CALUDE_cosine_sum_simplification_l2733_273391

theorem cosine_sum_simplification :
  Real.cos ((2 * Real.pi) / 17) + Real.cos ((6 * Real.pi) / 17) + Real.cos ((8 * Real.pi) / 17) = (Real.sqrt 13 - 1) / 4 :=
by sorry

end NUMINAMATH_CALUDE_cosine_sum_simplification_l2733_273391


namespace NUMINAMATH_CALUDE_min_sum_a1_a2_l2733_273381

/-- The sequence (aᵢ) is defined by aₙ₊₂ = (aₙ + 3007) / (1 + aₙ₊₁) for n ≥ 1, where all aᵢ are positive integers. -/
def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, a (n + 2) = (a n + 3007) / (1 + a (n + 1))

/-- The minimum possible value of a₁ + a₂ is 114. -/
theorem min_sum_a1_a2 :
  ∀ a : ℕ → ℕ, is_valid_sequence a → a 1 + a 2 ≥ 114 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_a1_a2_l2733_273381


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2733_273397

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- A point on the right branch of a hyperbola -/
structure RightBranchPoint (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2/a^2 - y^2/b^2 = 1
  right_branch : x ≥ a

/-- Distance between two points -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := sorry

/-- Left focus of the hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Right focus of the hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The ratio |PF₁|²/|PF₂| for a point P on the hyperbola -/
def focal_ratio (h : Hyperbola a b) (p : RightBranchPoint h) : ℝ := sorry

/-- The minimum value of focal_ratio over all points on the right branch -/
def min_focal_ratio (h : Hyperbola a b) : ℝ := sorry

theorem hyperbola_eccentricity_range (a b : ℝ) (h : Hyperbola a b) :
  min_focal_ratio h = 8 * a → 1 < eccentricity h ∧ eccentricity h ≤ 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2733_273397


namespace NUMINAMATH_CALUDE_valid_placements_count_l2733_273387

/-- Represents a ball -/
inductive Ball : Type
| A : Ball
| B : Ball
| C : Ball
| D : Ball

/-- Represents a box -/
inductive Box : Type
| one : Box
| two : Box
| three : Box

/-- A placement of balls into boxes -/
def Placement := Ball → Box

/-- Checks if a placement is valid -/
def isValidPlacement (p : Placement) : Prop :=
  (∀ b : Box, ∃ ball : Ball, p ball = b) ∧ 
  (p Ball.A ≠ p Ball.B)

/-- The number of valid placements -/
def numValidPlacements : ℕ := sorry

theorem valid_placements_count : numValidPlacements = 30 := by sorry

end NUMINAMATH_CALUDE_valid_placements_count_l2733_273387


namespace NUMINAMATH_CALUDE_expression_simplification_l2733_273338

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 - 2) / x) * ((y^2 - 2) / y) - ((x^2 + 2) / y) * ((y^2 + 2) / x) = -4 * (x / y + y / x) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2733_273338


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2733_273305

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals :
  diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2733_273305


namespace NUMINAMATH_CALUDE_arithmetic_sign_change_geometric_sign_alternation_l2733_273369

-- Define an arithmetic progression
def arithmetic_progression (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Define a geometric progression
def geometric_progression (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

-- Theorem for arithmetic progression sign change
theorem arithmetic_sign_change (a₁ : ℝ) (d : ℝ) :
  ∃ k : ℕ, ∀ n : ℕ, n ≤ k → (arithmetic_progression a₁ d n > 0) ∧
                     ∀ m : ℕ, m > k → (arithmetic_progression a₁ d m < 0) :=
sorry

-- Theorem for geometric progression sign alternation
theorem geometric_sign_alternation (a₁ : ℝ) (r : ℝ) (h : r < 0) :
  ∀ n : ℕ, (geometric_progression a₁ r (2*n) > 0) ∧ 
           (geometric_progression a₁ r (2*n + 1) < 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sign_change_geometric_sign_alternation_l2733_273369


namespace NUMINAMATH_CALUDE_binary_subtraction_theorem_l2733_273394

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of 111001₂ -/
def binary_111001 : List Bool := [true, false, false, true, true, true]

theorem binary_subtraction_theorem :
  binary_to_decimal binary_111001 - 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_binary_subtraction_theorem_l2733_273394


namespace NUMINAMATH_CALUDE_inverse_i_minus_inverse_l2733_273318

/-- Given a complex number i where i^2 = -1, prove that (i - i⁻¹)⁻¹ = -i/2 -/
theorem inverse_i_minus_inverse (i : ℂ) (h : i^2 = -1) : (i - i⁻¹)⁻¹ = -i/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_i_minus_inverse_l2733_273318


namespace NUMINAMATH_CALUDE_graphs_intersection_l2733_273337

/-- 
Given non-zero real numbers k and b, this theorem states that the graphs of 
y = kx + b and y = kb/x can only intersect in the first and third quadrants when kb > 0.
-/
theorem graphs_intersection (k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0) :
  (∀ x y : ℝ, y = k * x + b ∧ y = k * b / x → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) ↔ k * b > 0 :=
by sorry

end NUMINAMATH_CALUDE_graphs_intersection_l2733_273337


namespace NUMINAMATH_CALUDE_choir_members_count_l2733_273323

theorem choir_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 300 ∧ 
  n % 6 = 1 ∧ 
  n % 8 = 3 ∧ 
  n % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l2733_273323


namespace NUMINAMATH_CALUDE_binomial_max_fifth_term_l2733_273347

/-- 
If in the binomial expansion of (√x + 2/x)^n, only the fifth term has the maximum 
binomial coefficient, then n = 8.
-/
theorem binomial_max_fifth_term (n : ℕ) : 
  (∀ k : ℕ, k ≠ 4 → Nat.choose n k ≤ Nat.choose n 4) ∧ 
  (∀ k : ℕ, k ≠ 4 → Nat.choose n k < Nat.choose n 4) → 
  n = 8 := by sorry

end NUMINAMATH_CALUDE_binomial_max_fifth_term_l2733_273347


namespace NUMINAMATH_CALUDE_red_tiles_181_implies_total_2116_l2733_273314

/-- Represents a square floor tiled with congruent square tiles -/
structure SquareFloor :=
  (side : ℕ)

/-- Calculates the number of red tiles on a square floor -/
def red_tiles (floor : SquareFloor) : ℕ :=
  4 * floor.side - 2

/-- Calculates the total number of tiles on a square floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side * floor.side

/-- Theorem stating that a square floor with 181 red tiles has 2116 total tiles -/
theorem red_tiles_181_implies_total_2116 :
  ∀ (floor : SquareFloor), red_tiles floor = 181 → total_tiles floor = 2116 :=
by
  sorry

end NUMINAMATH_CALUDE_red_tiles_181_implies_total_2116_l2733_273314


namespace NUMINAMATH_CALUDE_hundreds_digit_of_factorial_difference_is_zero_l2733_273386

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem hundreds_digit_of_factorial_difference_is_zero :
  ∃ k : ℕ, factorial 25 - factorial 20 = 1000 * k :=
sorry

end NUMINAMATH_CALUDE_hundreds_digit_of_factorial_difference_is_zero_l2733_273386


namespace NUMINAMATH_CALUDE_fraction_simplification_l2733_273393

theorem fraction_simplification : (2 / (3 + Real.sqrt 5)) * (2 / (3 - Real.sqrt 5)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2733_273393


namespace NUMINAMATH_CALUDE_determinant_equation_solution_l2733_273356

/-- Definition of a second-order determinant -/
def secondOrderDet (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem stating that if |x+3 x-3; x-3 x+3| = 12, then x = 1 -/
theorem determinant_equation_solution :
  ∀ x : ℝ, secondOrderDet (x + 3) (x - 3) (x - 3) (x + 3) = 12 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equation_solution_l2733_273356


namespace NUMINAMATH_CALUDE_fruit_sales_theorem_l2733_273371

/-- The standard weight of a batch of fruits in kilograms -/
def standard_weight : ℕ := 30

/-- The weight deviations from the standard weight -/
def weight_deviations : List ℤ := [9, -10, -5, 6, -7, -6, 7, 10]

/-- The price per kilogram on the first day in yuan -/
def price_per_kg : ℕ := 10

/-- The discount rate for the second day as a rational number -/
def discount_rate : ℚ := 1/10

theorem fruit_sales_theorem :
  let total_weight := (List.sum weight_deviations + standard_weight * weight_deviations.length : ℤ)
  let first_day_sales := (price_per_kg * (total_weight / 2) : ℚ)
  let second_day_sales := (price_per_kg * (1 - discount_rate) * (total_weight - total_weight / 2) : ℚ)
  total_weight = 244 ∧ (first_day_sales + second_day_sales : ℚ) = 2318 := by
  sorry

end NUMINAMATH_CALUDE_fruit_sales_theorem_l2733_273371


namespace NUMINAMATH_CALUDE_boots_cost_ratio_l2733_273385

theorem boots_cost_ratio (initial_amount : ℚ) (toilet_paper_cost : ℚ) (additional_money : ℚ) :
  initial_amount = 50 →
  toilet_paper_cost = 12 →
  additional_money = 35 →
  let remaining_after_toilet_paper := initial_amount - toilet_paper_cost
  let groceries_cost := 2 * toilet_paper_cost
  let remaining_after_groceries := remaining_after_toilet_paper - groceries_cost
  let total_boot_cost := remaining_after_groceries + 2 * additional_money
  let single_boot_cost := total_boot_cost / 2
  (single_boot_cost / remaining_after_groceries : ℚ) = 3 := by
sorry

end NUMINAMATH_CALUDE_boots_cost_ratio_l2733_273385


namespace NUMINAMATH_CALUDE_circle_center_condition_l2733_273303

-- Define the equation of the circle
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*y + 3 - a = 0

-- Define the condition for the center to be in the second quadrant
def center_in_second_quadrant (a : ℝ) : Prop :=
  a < 0 ∧ 1 > 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a < -2

-- Theorem statement
theorem circle_center_condition (a : ℝ) :
  (∃ x y : ℝ, circle_equation x y a) ∧
  center_in_second_quadrant a
  ↔ a_range a :=
sorry

end NUMINAMATH_CALUDE_circle_center_condition_l2733_273303


namespace NUMINAMATH_CALUDE_find_integers_with_sum_and_lcm_l2733_273317

theorem find_integers_with_sum_and_lcm :
  ∃ (a b : ℕ+), 
    (a + b : ℕ) = 3972 ∧ 
    Nat.lcm a b = 985928 ∧ 
    a = 1964 ∧ 
    b = 2008 := by
  sorry

end NUMINAMATH_CALUDE_find_integers_with_sum_and_lcm_l2733_273317


namespace NUMINAMATH_CALUDE_two_quadratic_solving_algorithms_l2733_273327

/-- A quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- An algorithm to solve a quadratic equation -/
structure QuadraticSolver where
  solve : QuadraticEquation → Set ℝ

/-- The specific quadratic equation x^2 - 5x + 6 = 0 -/
def specificEquation : QuadraticEquation :=
  { a := 1, b := -5, c := 6 }

theorem two_quadratic_solving_algorithms :
  ∃ (algo1 algo2 : QuadraticSolver), algo1 ≠ algo2 ∧
    algo1.solve specificEquation = algo2.solve specificEquation :=
sorry

end NUMINAMATH_CALUDE_two_quadratic_solving_algorithms_l2733_273327


namespace NUMINAMATH_CALUDE_eggs_per_year_is_3380_l2733_273351

/-- The number of eggs Lisa cooks for her family for breakfast in a year -/
def eggs_per_year : ℕ :=
  let days_per_week : ℕ := 5
  let num_children : ℕ := 4
  let eggs_per_child : ℕ := 2
  let eggs_for_husband : ℕ := 3
  let eggs_for_self : ℕ := 2
  let weeks_per_year : ℕ := 52
  
  let eggs_per_day : ℕ := num_children * eggs_per_child + eggs_for_husband + eggs_for_self
  let eggs_per_week : ℕ := eggs_per_day * days_per_week
  
  eggs_per_week * weeks_per_year

theorem eggs_per_year_is_3380 : eggs_per_year = 3380 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_year_is_3380_l2733_273351


namespace NUMINAMATH_CALUDE_smallest_mu_inequality_l2733_273308

theorem smallest_mu_inequality (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0) :
  (∀ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 + 2*a*d ≥ μ*(a*b + b*c + c*d)) → μ ≥ 2) ∧
  (∀ a b c d : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 + 2*a*d ≥ 2*(a*b + b*c + c*d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_mu_inequality_l2733_273308
