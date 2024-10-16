import Mathlib

namespace NUMINAMATH_CALUDE_max_constant_inequality_l613_61393

theorem max_constant_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x^2 + y^2 = 1) :
  ∃ (c : ℝ), c = 1/2 ∧ x^6 + y^6 ≥ c*x*y ∧ ∀ (c' : ℝ), (∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 + y'^2 = 1 → x'^6 + y'^6 ≥ c'*x'*y') → c' ≤ c :=
sorry

end NUMINAMATH_CALUDE_max_constant_inequality_l613_61393


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l613_61373

-- Define the triangles and their sides
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the similarity relation between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Given values
def ABC : Triangle := { A := 15, B := 0, C := 24 }
def FGH : Triangle := { A := 0, B := 0, C := 18 }

-- Theorem statement
theorem similar_triangles_side_length :
  similar ABC FGH →
  FGH.A = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l613_61373


namespace NUMINAMATH_CALUDE_triangle_arctans_sum_l613_61390

theorem triangle_arctans_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : b^2 + c^2 = a^2) (h5 : Real.arcsin (1/2) + Real.arcsin (1/2) = Real.pi/2) :
  Real.arctan (b/(c+a)) + Real.arctan (c/(b+a)) = Real.pi/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_arctans_sum_l613_61390


namespace NUMINAMATH_CALUDE_total_books_l613_61348

theorem total_books (keith_books jason_books megan_books : ℕ) 
  (h1 : keith_books = 20) 
  (h2 : jason_books = 21) 
  (h3 : megan_books = 15) : 
  keith_books + jason_books + megan_books = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l613_61348


namespace NUMINAMATH_CALUDE_right_triangle_angle_sum_l613_61368

theorem right_triangle_angle_sum (A B C : ℝ) : 
  A = 20 → C = 90 → A + B + C = 180 → B = 70 := by sorry

end NUMINAMATH_CALUDE_right_triangle_angle_sum_l613_61368


namespace NUMINAMATH_CALUDE_anniversary_18_months_ago_proof_l613_61391

/-- The anniversary Bella and Bob celebrated 18 months ago -/
def anniversary_18_months_ago : ℕ := 2

/-- The number of months until their 4th anniversary -/
def months_until_4th_anniversary : ℕ := 6

/-- The current duration of their relationship in months -/
def current_relationship_duration : ℕ := 4 * 12 - months_until_4th_anniversary

/-- The duration of their relationship 18 months ago in months -/
def relationship_duration_18_months_ago : ℕ := current_relationship_duration - 18

theorem anniversary_18_months_ago_proof :
  anniversary_18_months_ago = relationship_duration_18_months_ago / 12 :=
by sorry

end NUMINAMATH_CALUDE_anniversary_18_months_ago_proof_l613_61391


namespace NUMINAMATH_CALUDE_two_numbers_divisible_by_three_l613_61395

def numbers : List Nat := [222, 2222, 22222, 222222]

theorem two_numbers_divisible_by_three : 
  (numbers.filter (fun n => n % 3 = 0)).length = 2 := by sorry

end NUMINAMATH_CALUDE_two_numbers_divisible_by_three_l613_61395


namespace NUMINAMATH_CALUDE_intersection_equality_necessary_not_sufficient_l613_61389

theorem intersection_equality_necessary_not_sufficient :
  (∀ (M N P : Set α), M = N → M ∩ P = N ∩ P) ∧
  (∃ (M N P : Set α), M ∩ P = N ∩ P ∧ M ≠ N) :=
by sorry

end NUMINAMATH_CALUDE_intersection_equality_necessary_not_sufficient_l613_61389


namespace NUMINAMATH_CALUDE_power_three_inverse_exponent_l613_61351

theorem power_three_inverse_exponent (x y : ℕ) : 
  (2^x : ℕ) ∣ 900 ∧ 
  ∀ k > x, ¬((2^k : ℕ) ∣ 900) ∧ 
  (5^y : ℕ) ∣ 900 ∧ 
  ∀ l > y, ¬((5^l : ℕ) ∣ 900) → 
  (1/3 : ℚ)^(2*(y - x)) = 1 := by
sorry

end NUMINAMATH_CALUDE_power_three_inverse_exponent_l613_61351


namespace NUMINAMATH_CALUDE_billy_youtube_suggestions_l613_61353

/-- The number of suggestion sets Billy watches before finding a video he likes -/
def num_sets : ℕ := 5

/-- The number of videos Billy watches from the final set -/
def videos_from_final_set : ℕ := 5

/-- The total number of videos Billy watches -/
def total_videos : ℕ := 65

/-- The number of suggestions generated each time -/
def suggestions_per_set : ℕ := 15

theorem billy_youtube_suggestions :
  (num_sets - 1) * suggestions_per_set + videos_from_final_set = total_videos :=
by sorry

end NUMINAMATH_CALUDE_billy_youtube_suggestions_l613_61353


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l613_61374

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The statement to prove -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 3 ^ 2 - 6 * a 3 + 5 = 0 →
  a 15 ^ 2 - 6 * a 15 + 5 = 0 →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l613_61374


namespace NUMINAMATH_CALUDE_prob_blank_one_shot_prob_blank_three_shots_prob_away_from_vertices_l613_61329

-- Define the number of bullets and the number of blanks
def total_bullets : ℕ := 4
def blank_bullets : ℕ := 1

-- Define the number of shots
def shots : ℕ := 3

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Theorem for the probability of shooting a blank in one shot
theorem prob_blank_one_shot : 
  (blank_bullets : ℝ) / total_bullets = 1 / 4 := by sorry

-- Theorem for the probability of a blank appearing in 3 shots
theorem prob_blank_three_shots : 
  1 - (total_bullets - blank_bullets : ℝ) * (total_bullets - blank_bullets - 1) * (total_bullets - blank_bullets - 2) / 
    (total_bullets * (total_bullets - 1) * (total_bullets - 2)) = 3 / 4 := by sorry

-- Theorem for the probability of all shots being more than 1 unit away from vertices
theorem prob_away_from_vertices (triangle_area : ℝ) (h : triangle_area = side_length^2 * Real.sqrt 3 / 4) :
  1 - (3 * π / 2) / triangle_area = 1 - Real.sqrt 3 * π / 150 := by sorry

end NUMINAMATH_CALUDE_prob_blank_one_shot_prob_blank_three_shots_prob_away_from_vertices_l613_61329


namespace NUMINAMATH_CALUDE_rehab_centers_multiple_l613_61334

/-- The number of rehabilitation centers visited by each person and the total visited --/
structure RehabCenters where
  lisa : ℕ
  jude : ℕ
  han : ℕ
  jane : ℕ
  total : ℕ

/-- The conditions of the problem --/
def problem_conditions (rc : RehabCenters) : Prop :=
  rc.lisa = 6 ∧
  rc.jude = rc.lisa / 2 ∧
  rc.han = 2 * rc.jude - 2 ∧
  rc.total = 27 ∧
  rc.jane = rc.total - (rc.lisa + rc.jude + rc.han)

/-- The theorem to be proved --/
theorem rehab_centers_multiple (rc : RehabCenters) 
  (h : problem_conditions rc) : ∃ x : ℕ, x = 2 ∧ rc.jane = x * rc.han + 6 := by
  sorry

end NUMINAMATH_CALUDE_rehab_centers_multiple_l613_61334


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l613_61370

theorem right_triangle_leg_length
  (hypotenuse : ℝ)
  (leg1 : ℝ)
  (h1 : hypotenuse = 15)
  (h2 : leg1 = 9)
  (h3 : hypotenuse^2 = leg1^2 + leg2^2) :
  leg2 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l613_61370


namespace NUMINAMATH_CALUDE_nested_expression_value_l613_61360

theorem nested_expression_value : (3*(3*(3*(3*(3*(3*(3+2)+2)+2)+2)+2)+2)+2) = 4373 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l613_61360


namespace NUMINAMATH_CALUDE_tangent_problem_l613_61341

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α/2 + β) = 1/2) 
  (h2 : Real.tan (β - α/2) = 1/3) : 
  Real.tan α = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l613_61341


namespace NUMINAMATH_CALUDE_icosahedron_cube_relation_l613_61300

/-- Given a cube with edge length a and an inscribed icosahedron, 
    m is the length of the line segment connecting two vertices 
    of the icosahedron on a face of the cube -/
def icosahedron_in_cube (a m : ℝ) : Prop :=
  a > 0 ∧ m > 0 ∧ a^2 - a*m - m^2 = 0

/-- Theorem stating the relationship between the cube's edge length 
    and the distance between icosahedron vertices on a face -/
theorem icosahedron_cube_relation {a m : ℝ} 
  (h : icosahedron_in_cube a m) : a^2 - a*m - m^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_icosahedron_cube_relation_l613_61300


namespace NUMINAMATH_CALUDE_cube_root_equation_solutions_l613_61379

theorem cube_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (10 * x - 2) ^ (1/3) + (20 * x + 3) ^ (1/3) - 5 * x ^ (1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = -1/25 ∨ x = 1/375 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solutions_l613_61379


namespace NUMINAMATH_CALUDE_winning_pair_probability_l613_61304

-- Define the deck
def deck_size : ℕ := 9
def num_colors : ℕ := 3
def num_letters : ℕ := 3

-- Define a winning pair
def is_winning_pair (card1 card2 : ℕ × ℕ) : Prop :=
  (card1.1 = card2.1) ∨ (card1.2 = card2.2)

-- Define the probability of drawing a winning pair
def prob_winning_pair : ℚ :=
  (num_colors * (num_letters.choose 2) + num_letters * (num_colors.choose 2)) / deck_size.choose 2

-- Theorem statement
theorem winning_pair_probability : prob_winning_pair = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_winning_pair_probability_l613_61304


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l613_61369

/-- Proves that the given trigonometric expression equals 1 --/
theorem trig_expression_equals_one :
  let expr := (Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
               Real.cos (160 * π / 180) * Real.cos (110 * π / 180)) /
              (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) + 
               Real.cos (156 * π / 180) * Real.cos (106 * π / 180))
  expr = 1 := by sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l613_61369


namespace NUMINAMATH_CALUDE_distinct_numbers_probability_l613_61378

def num_sides : ℕ := 5
def num_dice : ℕ := 5

theorem distinct_numbers_probability : 
  (Nat.factorial num_sides : ℚ) / (num_sides ^ num_dice : ℚ) = 24 / 625 := by
  sorry

end NUMINAMATH_CALUDE_distinct_numbers_probability_l613_61378


namespace NUMINAMATH_CALUDE_line_circle_intersection_x_intercept_l613_61355

/-- The x-intercept of a line that intersects a circle --/
theorem line_circle_intersection_x_intercept
  (m : ℝ)  -- Slope of the line
  (h1 : ∀ x y : ℝ, m * x + y + 3 * m - Real.sqrt 3 = 0 → x^2 + y^2 = 12 → 
         ∃ A B : ℝ × ℝ, A ≠ B ∧ 
         m * A.1 + A.2 + 3 * m - Real.sqrt 3 = 0 ∧
         A.1^2 + A.2^2 = 12 ∧
         m * B.1 + B.2 + 3 * m - Real.sqrt 3 = 0 ∧
         B.1^2 + B.2^2 = 12)
  (h2 : ∃ A B : ℝ × ℝ, (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12) :
  ∃ x : ℝ, x = -6 ∧ m * x + 3 * m - Real.sqrt 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_x_intercept_l613_61355


namespace NUMINAMATH_CALUDE_forest_width_is_correct_l613_61367

/-- The width of a forest in miles -/
def forest_width : ℝ := 6

/-- The length of the forest in miles -/
def forest_length : ℝ := 4

/-- The number of trees per square mile -/
def trees_per_square_mile : ℕ := 600

/-- The number of trees one logger can cut per day -/
def trees_per_logger_per_day : ℕ := 6

/-- The number of days in a month -/
def days_per_month : ℕ := 30

/-- The number of loggers working -/
def number_of_loggers : ℕ := 8

/-- The number of months it takes to cut down all trees -/
def months_to_cut_all_trees : ℕ := 10

theorem forest_width_is_correct : 
  forest_width * forest_length * trees_per_square_mile = 
  (trees_per_logger_per_day * days_per_month * number_of_loggers * months_to_cut_all_trees : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_forest_width_is_correct_l613_61367


namespace NUMINAMATH_CALUDE_sunglasses_and_hat_probability_l613_61302

/-- The probability that a randomly selected person wearing sunglasses is also wearing a hat -/
theorem sunglasses_and_hat_probability
  (total_sunglasses : ℕ)
  (total_hats : ℕ)
  (prob_sunglasses_given_hat : ℚ)
  (h1 : total_sunglasses = 60)
  (h2 : total_hats = 45)
  (h3 : prob_sunglasses_given_hat = 3 / 5) :
  (total_hats : ℚ) * prob_sunglasses_given_hat / total_sunglasses = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_and_hat_probability_l613_61302


namespace NUMINAMATH_CALUDE_range_x_when_m_is_one_range_m_for_not_p_sufficient_not_necessary_l613_61359

-- Define propositions p and q
def p (x m : ℝ) : Prop := |2*x - m| ≥ 1
def q (x : ℝ) : Prop := (1 - 3*x) / (x + 2) > 0

-- Theorem for part (I)
theorem range_x_when_m_is_one :
  ∃ a b : ℝ, a = -2 ∧ b = 0 ∧
  ∀ x : ℝ, a < x ∧ x ≤ b ↔ p x 1 ∧ q x :=
sorry

-- Theorem for part (II)
theorem range_m_for_not_p_sufficient_not_necessary :
  ∃ a b : ℝ, a = -3 ∧ b = -1/3 ∧
  ∀ m : ℝ, a ≤ m ∧ m ≤ b ↔
    (∀ x : ℝ, ¬(p x m) → q x) ∧
    ¬(∀ x : ℝ, q x → ¬(p x m)) :=
sorry

end NUMINAMATH_CALUDE_range_x_when_m_is_one_range_m_for_not_p_sufficient_not_necessary_l613_61359


namespace NUMINAMATH_CALUDE_sum_is_monomial_then_m_plus_k_equals_five_l613_61328

/-- If the sum of two terms is a monomial, then the exponents must be equal -/
axiom monomial_sum_exponents_equal {x y : ℕ → ℕ → ℕ} {a b c d m k : ℕ} (h : ∀ n, x n m + y n k = c) :
  m = a ∧ k = b

/-- If the sum of 3x^3y^k and 7x^my^2 is a monomial, then m + k = 5 -/
theorem sum_is_monomial_then_m_plus_k_equals_five (m k : ℕ) 
  (h : ∃ (c : ℕ → ℕ → ℕ), ∀ x y, 3 * (x^3 * y^k) + 7 * (x^m * y^2) = c x y) : 
  m + k = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_monomial_then_m_plus_k_equals_five_l613_61328


namespace NUMINAMATH_CALUDE_constant_function_invariant_l613_61376

-- Define g as a function from ℝ to ℝ
def g : ℝ → ℝ := λ x => 5

-- Theorem statement
theorem constant_function_invariant (x : ℝ) : g (3 * x - 7) = 5 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_invariant_l613_61376


namespace NUMINAMATH_CALUDE_tan_22_5_deg_identity_l613_61397

theorem tan_22_5_deg_identity : 
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_identity_l613_61397


namespace NUMINAMATH_CALUDE_longest_altitudes_sum_is_14_l613_61303

/-- A triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : a = 6
  h_b : b = 8
  h_c : c = 10
  h_right : a^2 + b^2 = c^2

/-- The sum of the lengths of the two longest altitudes in the triangle -/
def longest_altitudes_sum (t : RightTriangle) : ℝ := t.a + t.b

theorem longest_altitudes_sum_is_14 (t : RightTriangle) :
  longest_altitudes_sum t = 14 := by
  sorry

end NUMINAMATH_CALUDE_longest_altitudes_sum_is_14_l613_61303


namespace NUMINAMATH_CALUDE_two_p_plus_q_l613_61308

theorem two_p_plus_q (p q : ℚ) (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_p_plus_q_l613_61308


namespace NUMINAMATH_CALUDE_parabola_equation_l613_61354

/-- A parabola with specified properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  axis_of_symmetry : a ≠ 0 → b = -4 * a
  tangent_line : ∃ x : ℝ, a * x^2 + b * x + c = 2 * x + 1 ∧
                 ∀ y : ℝ, y ≠ x → a * y^2 + b * y + c > 2 * y + 1
  y_intercepts : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
                 a * x₁^2 + b * x₁ + c = 0 ∧
                 a * x₂^2 + b * x₂ + c = 0 ∧
                 (x₁ - x₂)^2 = 8

/-- The parabola equation is one of the two specified forms -/
theorem parabola_equation (p : Parabola) : 
  (p.a = 1 ∧ p.b = 4 ∧ p.c = 2) ∨ (p.a = 1/2 ∧ p.b = 2 ∧ p.c = 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l613_61354


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_two_sqrt_two_l613_61357

theorem sqrt_difference_equals_two_sqrt_two :
  Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_two_sqrt_two_l613_61357


namespace NUMINAMATH_CALUDE_one_positive_real_solution_l613_61343

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^12 + 5*x^11 + 20*x^10 + 1300*x^9 - 1105*x^8

-- Theorem statement
theorem one_positive_real_solution :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_one_positive_real_solution_l613_61343


namespace NUMINAMATH_CALUDE_problem_statement_l613_61325

theorem problem_statement (P Q : ℝ) 
  (h1 : P^2 - P*Q = 1) 
  (h2 : 4*P*Q - 3*Q^2 = 2) : 
  P^2 + 3*P*Q - 3*Q^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l613_61325


namespace NUMINAMATH_CALUDE_find_x_value_l613_61332

theorem find_x_value (x : ℝ) : 
  (max 1 (max 2 (max 3 x)) = 1 + 2 + 3 + x) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l613_61332


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l613_61362

theorem matrix_equation_solution (x : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, x]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2, -2; -1, 1]
  B * A = !![2, 4; -1, -2] → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l613_61362


namespace NUMINAMATH_CALUDE_rehana_age_l613_61346

/-- Represents the ages of Rehana, Phoebe, and Jacob -/
structure Ages where
  rehana : ℕ
  phoebe : ℕ
  jacob : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.jacob = 3 ∧
  ages.jacob = (3 * ages.phoebe) / 5 ∧
  ages.rehana + 5 = 3 * (ages.phoebe + 5)

/-- The theorem stating Rehana's current age -/
theorem rehana_age :
  ∃ (ages : Ages), problem_conditions ages ∧ ages.rehana = 25 := by
  sorry

end NUMINAMATH_CALUDE_rehana_age_l613_61346


namespace NUMINAMATH_CALUDE_a_can_be_any_real_l613_61349

theorem a_can_be_any_real : ∀ (a b c d : ℤ), 
  b > 0 → d < 0 → (a : ℚ) / b > (c : ℚ) / d → 
  (∃ (x : ℝ), x > 0 ∧ (∃ (a : ℤ), (a : ℚ) / b > (c : ℚ) / d ∧ (a : ℝ) = x)) ∧
  (∃ (y : ℝ), y < 0 ∧ (∃ (a : ℤ), (a : ℚ) / b > (c : ℚ) / d ∧ (a : ℝ) = y)) ∧
  (∃ (a : ℤ), (a : ℚ) / b > (c : ℚ) / d ∧ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_a_can_be_any_real_l613_61349


namespace NUMINAMATH_CALUDE_power_of_seven_l613_61392

theorem power_of_seven (k : ℕ) (h : 7^k = 2) : 7^(2*k + 2) = 784 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_l613_61392


namespace NUMINAMATH_CALUDE_largest_lucky_number_is_499_l613_61307

def lucky_number (a b : ℕ) : ℕ := a + b + a * b

def largest_lucky_number_after_three_operations : ℕ :=
  let n1 := lucky_number 1 4
  let n2 := lucky_number 4 n1
  let n3 := lucky_number n1 n2
  max n1 (max n2 n3)

theorem largest_lucky_number_is_499 :
  largest_lucky_number_after_three_operations = 499 := by sorry

end NUMINAMATH_CALUDE_largest_lucky_number_is_499_l613_61307


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l613_61345

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 9| = |x - 3| := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l613_61345


namespace NUMINAMATH_CALUDE_apple_purchase_cost_l613_61322

/-- Represents a purchase option for apples -/
structure AppleOption where
  count : ℕ
  price : ℕ

/-- Calculates the total cost of purchasing apples -/
def totalCost (option1 : AppleOption) (option2 : AppleOption) (count1 : ℕ) (count2 : ℕ) : ℕ :=
  option1.price * count1 + option2.price * count2

/-- Calculates the total number of apples purchased -/
def totalApples (option1 : AppleOption) (option2 : AppleOption) (count1 : ℕ) (count2 : ℕ) : ℕ :=
  option1.count * count1 + option2.count * count2

theorem apple_purchase_cost (option1 : AppleOption) (option2 : AppleOption) :
  option1.count = 4 →
  option1.price = 15 →
  option2.count = 7 →
  option2.price = 25 →
  ∃ (count1 count2 : ℕ),
    count1 = count2 ∧
    totalApples option1 option2 count1 count2 = 28 ∧
    totalCost option1 option2 count1 count2 = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_cost_l613_61322


namespace NUMINAMATH_CALUDE_scientific_notation_317000_l613_61371

theorem scientific_notation_317000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 317000 = a * (10 : ℝ) ^ n ∧ a = 3.17 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_317000_l613_61371


namespace NUMINAMATH_CALUDE_player_arrangement_count_l613_61358

/-- The number of ways to arrange players from three teams -/
def arrange_players (n : ℕ) : ℕ :=
  (n.factorial) * (n.factorial) * (n.factorial) * (n.factorial)

/-- Theorem: The number of ways to arrange 9 players from 3 teams is 1296 -/
theorem player_arrangement_count :
  arrange_players 3 = 1296 := by
  sorry

#eval arrange_players 3

end NUMINAMATH_CALUDE_player_arrangement_count_l613_61358


namespace NUMINAMATH_CALUDE_diet_soda_count_l613_61342

/-- The number of bottles of regular soda -/
def regular_soda : ℕ := 79

/-- The difference between regular and diet soda bottles -/
def difference : ℕ := 26

/-- The number of bottles of diet soda -/
def diet_soda : ℕ := regular_soda - difference

theorem diet_soda_count : diet_soda = 53 := by
  sorry

end NUMINAMATH_CALUDE_diet_soda_count_l613_61342


namespace NUMINAMATH_CALUDE_max_roses_is_317_l613_61383

/-- Represents the price of roses in cents to avoid floating-point issues -/
def individual_price : ℕ := 530
def dozen_price : ℕ := 3600
def two_dozen_price : ℕ := 5000
def budget : ℕ := 68000

/-- Calculates the maximum number of roses that can be purchased with the given budget -/
def max_roses : ℕ :=
  let two_dozen_sets := budget / two_dozen_price
  let remaining_budget := budget - two_dozen_sets * two_dozen_price
  let individual_roses := remaining_budget / individual_price
  two_dozen_sets * 24 + individual_roses

theorem max_roses_is_317 : max_roses = 317 := by sorry

end NUMINAMATH_CALUDE_max_roses_is_317_l613_61383


namespace NUMINAMATH_CALUDE_sum_of_squares_l613_61396

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l613_61396


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocal_product_l613_61317

theorem min_value_sum_reciprocal_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (1 / a + 1 / b) ≥ 4 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y) * (1 / x + 1 / y) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocal_product_l613_61317


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l613_61356

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l613_61356


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l613_61364

theorem triangle_inequality_theorem (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l613_61364


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l613_61399

/-- The average speed of a car given its distances traveled in two hours -/
theorem average_speed_two_hours (d1 d2 : ℝ) : d1 = 80 → d2 = 60 → (d1 + d2) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l613_61399


namespace NUMINAMATH_CALUDE_no_even_three_digit_sum_31_l613_61377

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_even_three_digit_sum_31 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 31 ∧ Even n :=
sorry

end NUMINAMATH_CALUDE_no_even_three_digit_sum_31_l613_61377


namespace NUMINAMATH_CALUDE_max_three_digit_sum_not_factor_l613_61384

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_not_factor_of_product (n : ℕ) : Prop :=
  ¬(2 * Nat.factorial (n - 1)) % (n + 1) = 0

theorem max_three_digit_sum_not_factor :
  ∃ (n : ℕ), is_three_digit n ∧ sum_not_factor_of_product n ∧
  ∀ (m : ℕ), is_three_digit m → sum_not_factor_of_product m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_three_digit_sum_not_factor_l613_61384


namespace NUMINAMATH_CALUDE_danny_found_58_new_caps_l613_61313

/-- Represents the number of bottle caps Danny has at different stages -/
structure BottleCaps where
  initial : ℕ
  thrown_away : ℕ
  final : ℕ

/-- Calculates the number of new bottle caps Danny found -/
def new_bottle_caps (bc : BottleCaps) : ℕ :=
  bc.final - (bc.initial - bc.thrown_away)

/-- Theorem stating that Danny found 58 new bottle caps -/
theorem danny_found_58_new_caps : 
  ∀ (bc : BottleCaps), 
  bc.initial = 69 → bc.thrown_away = 60 → bc.final = 67 → 
  new_bottle_caps bc = 58 := by
  sorry

end NUMINAMATH_CALUDE_danny_found_58_new_caps_l613_61313


namespace NUMINAMATH_CALUDE_sin_300_degrees_l613_61331

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l613_61331


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l613_61339

theorem quadratic_inequality_solution (z : ℝ) :
  z^2 - 40*z + 350 ≤ 6 ↔ 20 - 2*Real.sqrt 14 ≤ z ∧ z ≤ 20 + 2*Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l613_61339


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l613_61301

theorem students_taking_one_subject (both : ℕ) (geometry : ℕ) (only_biology : ℕ)
  (h1 : both = 15)
  (h2 : geometry = 40)
  (h3 : only_biology = 20) :
  geometry - both + only_biology = 45 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l613_61301


namespace NUMINAMATH_CALUDE_f_properties_l613_61363

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -(Real.sin x)^2 + a * Real.sin x - 1

theorem f_properties :
  (∀ x, f 1 x ≥ -3) ∧
  (∀ x, f 1 x = -3 → ∃ y, f 1 y = -3) ∧
  (∀ a, (∀ x, f a x ≤ 1/2) ∧ (∃ y, f a y = 1/2) ↔ a = -5/2 ∨ a = 5/2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l613_61363


namespace NUMINAMATH_CALUDE_trigonometric_identities_l613_61386

theorem trigonometric_identities (α : ℝ) (h : 2 * Real.sin α + Real.cos α = 0) :
  (((2 * Real.cos α - Real.sin α) / (Real.sin α + Real.cos α)) = 5) ∧
  ((Real.sin α / (Real.sin α ^ 3 - Real.cos α ^ 3)) = 5/3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l613_61386


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l613_61375

theorem quadratic_inequality_condition (a : ℝ) : 
  (a ≠ 0 ∧ ∀ x : ℝ, a * x^2 + 2 * a * x - 4 < 0) ↔ (-4 < a ∧ a < 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l613_61375


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l613_61323

theorem repeating_decimal_difference : 
  let x : ℚ := 72 / 99  -- $0.\overline{72}$ as a fraction
  let y : ℚ := 72 / 100 -- $0.72$ as a fraction
  x - y = 2 / 275 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l613_61323


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l613_61333

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 24 < 0 → n ≤ 7 :=
by
  sorry

theorem seven_satisfies_inequality :
  (7 : ℤ)^2 - 11*7 + 24 < 0 :=
by
  sorry

theorem eight_does_not_satisfy_inequality :
  (8 : ℤ)^2 - 11*8 + 24 ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l613_61333


namespace NUMINAMATH_CALUDE_four_x_plus_t_is_odd_l613_61380

theorem four_x_plus_t_is_odd (x t : ℤ) (h : 2 * x - t = 11) : Odd (4 * x + t) := by
  sorry

end NUMINAMATH_CALUDE_four_x_plus_t_is_odd_l613_61380


namespace NUMINAMATH_CALUDE_function_domain_implies_a_range_l613_61340

/-- If the function f(x) = √(2^(x^2 + 2ax - a) - 1) is defined for all real x, then -1 ≤ a ≤ 0 -/
theorem function_domain_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (2^(x^2 + 2*a*x - a) - 1)) → 
  -1 ≤ a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_function_domain_implies_a_range_l613_61340


namespace NUMINAMATH_CALUDE_base_value_l613_61312

/-- A triangle with specific side length properties -/
structure SpecificTriangle where
  left : ℝ
  right : ℝ
  base : ℝ
  sum_of_sides : left + right + base = 50
  right_longer : right = left + 2
  left_value : left = 12

theorem base_value (t : SpecificTriangle) : t.base = 24 := by
  sorry

end NUMINAMATH_CALUDE_base_value_l613_61312


namespace NUMINAMATH_CALUDE_work_completion_l613_61361

theorem work_completion (days1 days2 men2 : ℕ) 
  (h1 : days1 = 80)
  (h2 : days2 = 56)
  (h3 : men2 = 20)
  (h4 : ∀ m d, m * d = men2 * days2) : 
  ∃ men1 : ℕ, men1 = 14 ∧ men1 * days1 = men2 * days2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l613_61361


namespace NUMINAMATH_CALUDE_triangle_on_axes_zero_volume_l613_61330

/-- Given a triangle ABC with sides of length 8, 6, and 10, where each vertex is on a positive axis,
    prove that the volume of tetrahedron OABC (where O is the origin) is 0. -/
theorem triangle_on_axes_zero_volume (a b c : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 →  -- vertices on positive axes
  a^2 + b^2 = 64 →  -- AB = 8
  b^2 + c^2 = 36 →  -- BC = 6
  c^2 + a^2 = 100 →  -- CA = 10
  (1/6 : ℝ) * a * b * c = 0 :=  -- volume of tetrahedron OABC
by sorry


end NUMINAMATH_CALUDE_triangle_on_axes_zero_volume_l613_61330


namespace NUMINAMATH_CALUDE_truck_max_load_l613_61382

/-- The maximum load a truck can carry, given the mass of lemon bags and remaining capacity -/
theorem truck_max_load (mass_per_bag : ℕ) (num_bags : ℕ) (remaining_capacity : ℕ) :
  mass_per_bag = 8 →
  num_bags = 100 →
  remaining_capacity = 100 →
  mass_per_bag * num_bags + remaining_capacity = 900 := by
  sorry

end NUMINAMATH_CALUDE_truck_max_load_l613_61382


namespace NUMINAMATH_CALUDE_part_one_part_two_l613_61306

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides opposite to angles A, B, C respectively

-- Define the conditions
variable (h1 : 0 < A) -- A is acute
variable (h2 : A < π / 2) -- A is acute
variable (h3 : 3 * b = 5 * a * Real.sin B) -- Given condition

-- Part 1
theorem part_one : 
  Real.sin (2 * A) + Real.cos ((B + C) / 2) ^ 2 = 53 / 50 := by sorry

-- Part 2
theorem part_two (h4 : a = Real.sqrt 2) (h5 : 1 / 2 * b * c * Real.sin A = 3 / 2) :
  b = Real.sqrt 5 ∧ c = Real.sqrt 5 := by sorry

end

end NUMINAMATH_CALUDE_part_one_part_two_l613_61306


namespace NUMINAMATH_CALUDE_complex_real_condition_l613_61314

theorem complex_real_condition (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / Complex.I
  z.im = 0 → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l613_61314


namespace NUMINAMATH_CALUDE_smallest_semicircle_area_l613_61335

/-- Given a right-angled triangle with semicircles on each side, prove that the smallest semicircle has area 144 -/
theorem smallest_semicircle_area (x : ℝ) : 
  x > 0 ∧ x^2 < 180 ∧ 3*x < 180 ∧ x^2 + 3*x = 180 → x^2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_semicircle_area_l613_61335


namespace NUMINAMATH_CALUDE_symmetrical_line_intersection_l613_61366

/-- Given points A and B, and a circle, prove that if the line symmetrical to AB about y=a intersects the circle, then a is in the range [1/3, 3/2]. -/
theorem symmetrical_line_intersection (a : ℝ) : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (0, a)
  let circle (x y : ℝ) := (x + 3)^2 + (y + 2)^2 = 1
  let symmetrical_line (x y : ℝ) := (3 - a) * x - 2 * y + 2 * a = 0
  (∃ x y, circle x y ∧ symmetrical_line x y) → a ∈ Set.Icc (1/3) (3/2) :=
by sorry

end NUMINAMATH_CALUDE_symmetrical_line_intersection_l613_61366


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_15_13_l613_61347

theorem half_abs_diff_squares_15_13 : 
  (1/2 : ℝ) * |15^2 - 13^2| = 28 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_15_13_l613_61347


namespace NUMINAMATH_CALUDE_valid_permutations_64420_l613_61337

def digits : List Nat := [6, 4, 4, 2, 0]

/-- The number of permutations of the digits that form a 5-digit number not starting with 0 -/
def valid_permutations (ds : List Nat) : Nat :=
  let non_zero_digits := ds.filter (· ≠ 0)
  let zero_digits := ds.filter (· = 0)
  non_zero_digits.length * (ds.length - 1).factorial / (non_zero_digits.map (λ d => (ds.filter (· = d)).length)).prod

theorem valid_permutations_64420 :
  valid_permutations digits = 48 := by
  sorry

end NUMINAMATH_CALUDE_valid_permutations_64420_l613_61337


namespace NUMINAMATH_CALUDE_vector_dot_product_l613_61385

/-- Given two vectors a and b in ℝ², where a is parallel to (a + b), prove that their dot product is 4. -/
theorem vector_dot_product (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, x]
  let b : Fin 2 → ℝ := ![1, -1]
  (∃ (k : ℝ), a = k • (a + b)) → 
  (a • b = 4) := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_l613_61385


namespace NUMINAMATH_CALUDE_proportional_segments_l613_61394

theorem proportional_segments (a b c d : ℝ) : 
  b = 3 → c = 6 → d = 9 → (a / b = c / d) → a = 2 := by sorry

end NUMINAMATH_CALUDE_proportional_segments_l613_61394


namespace NUMINAMATH_CALUDE_difference_between_results_l613_61344

theorem difference_between_results (x : ℝ) (h : x = 15) : 2 * x - (26 - x) = 19 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_results_l613_61344


namespace NUMINAMATH_CALUDE_fifty_factorial_trailing_zeros_l613_61319

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 50! is 12 -/
theorem fifty_factorial_trailing_zeros :
  trailingZeros 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fifty_factorial_trailing_zeros_l613_61319


namespace NUMINAMATH_CALUDE_gcd_lcm_relation_l613_61321

theorem gcd_lcm_relation (a b c : ℕ+) :
  (Nat.gcd a (Nat.gcd b c))^2 * Nat.lcm a b * Nat.lcm b c * Nat.lcm c a =
  (Nat.lcm a (Nat.lcm b c))^2 * Nat.gcd a b * Nat.gcd b c * Nat.gcd c a :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_relation_l613_61321


namespace NUMINAMATH_CALUDE_initial_pencils_theorem_l613_61320

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 34

/-- The number of pencils Dan took from the drawer -/
def pencils_taken : ℕ := 22

/-- The number of pencils remaining in the drawer -/
def pencils_remaining : ℕ := 12

/-- Theorem: The initial number of pencils equals the sum of pencils taken and pencils remaining -/
theorem initial_pencils_theorem : initial_pencils = pencils_taken + pencils_remaining := by
  sorry

end NUMINAMATH_CALUDE_initial_pencils_theorem_l613_61320


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l613_61387

def cost_price : ℝ := 180
def selling_price : ℝ := 207

theorem profit_percentage_calculation :
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 15 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l613_61387


namespace NUMINAMATH_CALUDE_parabola_directrix_l613_61318

/-- The directrix of the parabola x = -1/4 * y^2 is x = 1 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), x = -(1/4) * y^2 → 
  ∃ (d : ℝ), d = 1 ∧ 
  ∀ (p : ℝ × ℝ), p.1 = -(1/4) * p.2^2 → 
  (p.1 - d)^2 = (p.1 - 0)^2 + p.2^2 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l613_61318


namespace NUMINAMATH_CALUDE_problem_2a_l613_61372

theorem problem_2a (a b x y : ℝ) 
  (eq1 : a * x + b * y = 7)
  (eq2 : a * x^2 + b * y^2 = 49)
  (eq3 : a * x^3 + b * y^3 = 133)
  (eq4 : a * x^4 + b * y^4 = 406) :
  2014 * (x + y - x * y) - 100 * (a + b) = 6889.33 := by
sorry

end NUMINAMATH_CALUDE_problem_2a_l613_61372


namespace NUMINAMATH_CALUDE_train_speed_calculation_l613_61310

/-- Proves that a train with given length, crossing a platform of given length in a specific time, has a specific speed. -/
theorem train_speed_calculation (train_length platform_length : ℝ) (crossing_time : ℝ) :
  train_length = 480 ∧ 
  platform_length = 620 ∧ 
  crossing_time = 71.99424046076314 →
  ∃ (speed : ℝ), abs (speed - 54.964) < 0.001 ∧ 
  speed = (train_length + platform_length) / crossing_time * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l613_61310


namespace NUMINAMATH_CALUDE_painting_time_equation_l613_61336

/-- The time it takes Doug to paint the room alone, in hours -/
def doug_time : ℝ := 5

/-- The time it takes Dave to paint the room alone, in hours -/
def dave_time : ℝ := 7

/-- The number of one-hour breaks taken -/
def breaks : ℝ := 2

/-- The total time it takes Doug and Dave to paint the room together, including breaks -/
noncomputable def total_time : ℝ := sorry

/-- Theorem stating that the equation (1/5 + 1/7)(t - 2) = 1 is satisfied by the total time -/
theorem painting_time_equation : 
  (1 / doug_time + 1 / dave_time) * (total_time - breaks) = 1 := by sorry

end NUMINAMATH_CALUDE_painting_time_equation_l613_61336


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l613_61315

theorem partial_fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 →
    (B * x - 19) / (x^2 - 8*x + 15) = A / (x - 3) + 5 / (x - 5)) →
  A + B = 33/5 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l613_61315


namespace NUMINAMATH_CALUDE_gcd_45045_30030_l613_61326

theorem gcd_45045_30030 : Nat.gcd 45045 30030 = 15015 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45045_30030_l613_61326


namespace NUMINAMATH_CALUDE_closest_fraction_l613_61324

def fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]
def team_gamma_fraction : ℚ := 13/80

theorem closest_fraction :
  ∀ f ∈ fractions, |team_gamma_fraction - 1/6| ≤ |team_gamma_fraction - f| :=
by sorry

end NUMINAMATH_CALUDE_closest_fraction_l613_61324


namespace NUMINAMATH_CALUDE_equation_solution_l613_61338

theorem equation_solution : ∃ x : ℚ, (2 / 5 - 1 / 3 : ℚ) = 1 / x ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l613_61338


namespace NUMINAMATH_CALUDE_triangle_area_l613_61352

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_area (abc : Triangle) (h1 : abc.c = 3) 
  (h2 : abc.a / Real.cos abc.A = abc.b / Real.cos abc.B)
  (h3 : Real.cos abc.C = 1/4) : 
  (1/2 * abc.a * abc.b * Real.sin abc.C) = (3 * Real.sqrt 15) / 4 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l613_61352


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l613_61327

/-- Calculates the average speed of a cyclist given two trips with different distances and speeds -/
theorem cyclist_average_speed (d1 d2 v1 v2 : ℝ) (h1 : d1 = 9) (h2 : d2 = 12) (h3 : v1 = 12) (h4 : v2 = 9) :
  let t1 := d1 / v1
  let t2 := d2 / v2
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  ∃ ε > 0, |average_speed - 10.1| < ε :=
by sorry

end NUMINAMATH_CALUDE_cyclist_average_speed_l613_61327


namespace NUMINAMATH_CALUDE_sum_of_integers_l613_61350

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 10) (h3 : x * y = 80) : x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l613_61350


namespace NUMINAMATH_CALUDE_shekars_english_score_l613_61305

/-- Given Shekar's scores in four subjects and his average score, prove his English score --/
theorem shekars_english_score
  (math_score science_score social_score biology_score : ℕ)
  (average_score : ℚ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : social_score = 82)
  (h4 : biology_score = 85)
  (h5 : average_score = 71)
  (h6 : (math_score + science_score + social_score + biology_score + english_score : ℚ) / 5 = average_score) :
  english_score = 47 :=
by sorry

end NUMINAMATH_CALUDE_shekars_english_score_l613_61305


namespace NUMINAMATH_CALUDE_todd_ate_cupcakes_l613_61388

theorem todd_ate_cupcakes (initial_cupcakes : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (h1 : initial_cupcakes = 20)
  (h2 : packages = 3)
  (h3 : cupcakes_per_package = 3) :
  initial_cupcakes - (packages * cupcakes_per_package) = 11 :=
by sorry

end NUMINAMATH_CALUDE_todd_ate_cupcakes_l613_61388


namespace NUMINAMATH_CALUDE_orange_cost_l613_61316

/-- Given that three dozen oranges cost $18.00, prove that four dozen oranges at the same rate cost $24.00 -/
theorem orange_cost (cost_three_dozen : ℝ) (h1 : cost_three_dozen = 18) :
  let cost_per_dozen := cost_three_dozen / 3
  cost_per_dozen * 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_l613_61316


namespace NUMINAMATH_CALUDE_product_of_seven_consecutive_divisible_by_ten_l613_61311

theorem product_of_seven_consecutive_divisible_by_ten (n : ℕ+) :
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_seven_consecutive_divisible_by_ten_l613_61311


namespace NUMINAMATH_CALUDE_incorrect_statement_proof_l613_61309

/-- Given non-empty sets A and B where A is not a subset of B, 
    prove that the statement "If x ∉ A, then x ∈ B is an impossible event" is false. -/
theorem incorrect_statement_proof 
  {α : Type*} (A B : Set α) (h_nonempty_A : A.Nonempty) (h_nonempty_B : B.Nonempty) 
  (h_not_subset : ¬(A ⊆ B)) :
  ¬(∀ x, x ∉ A → x ∉ B) :=
sorry

end NUMINAMATH_CALUDE_incorrect_statement_proof_l613_61309


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l613_61365

theorem quadratic_inequality_solution (a : ℝ) (x₁ x₂ : ℝ) : 
  a < 0 →
  (∀ x, x^2 - a*x - 6*a^2 > 0 ↔ x < x₁ ∨ x > x₂) →
  x₂ - x₁ = 5 * Real.sqrt 2 →
  a = -Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l613_61365


namespace NUMINAMATH_CALUDE_a_properties_l613_61381

/-- Sequence a_n satisfying the given recurrence relation -/
def a : ℕ → ℚ
  | 0 => 1  -- a_1 = 1
  | 1 => 6  -- a_2 = 6
  | (n+2) => ((n+3) * (a (n+1) - 1)) / (n+2)

/-- Theorem stating the properties of sequence a_n -/
theorem a_properties :
  (∀ n : ℕ, a n = 2 * n^2 - n) ∧
  (∃ p q : ℚ, p ≠ 0 ∧ q ≠ 0 ∧
    (∃ d : ℚ, ∀ n : ℕ, a (n+1) / (p * (n+1) + q) - a n / (p * n + q) = d) ↔
    p + 2*q = 0) := by sorry

end NUMINAMATH_CALUDE_a_properties_l613_61381


namespace NUMINAMATH_CALUDE_teddy_bear_production_solution_l613_61398

/-- Represents the teddy bear production problem -/
structure TeddyBearProduction where
  /-- The number of days originally planned -/
  days : ℕ
  /-- The number of teddy bears ordered -/
  order : ℕ

/-- The conditions of the teddy bear production problem are satisfied -/
def satisfies_conditions (p : TeddyBearProduction) : Prop :=
  20 * p.days + 100 = p.order ∧ 23 * p.days - 20 = p.order

/-- The theorem stating the solution to the teddy bear production problem -/
theorem teddy_bear_production_solution :
  ∃ (p : TeddyBearProduction), satisfies_conditions p ∧ p.days = 40 ∧ p.order = 900 :=
sorry

end NUMINAMATH_CALUDE_teddy_bear_production_solution_l613_61398
