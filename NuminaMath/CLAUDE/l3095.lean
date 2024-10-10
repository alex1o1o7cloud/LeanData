import Mathlib

namespace stability_comparison_l3095_309511

/-- Represents a student's performance in standing long jump --/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Defines stability of performance based on variance --/
def more_stable (a b : StudentPerformance) : Prop :=
  a.average_score = b.average_score ∧ a.variance < b.variance

/-- Theorem: Given two students with the same average score, 
    the one with lower variance has more stable performance --/
theorem stability_comparison 
  (student_A student_B : StudentPerformance)
  (h_same_average : student_A.average_score = student_B.average_score)
  (h_A_variance : student_A.variance = 0.6)
  (h_B_variance : student_B.variance = 0.35) :
  more_stable student_B student_A :=
sorry

end stability_comparison_l3095_309511


namespace exists_a_plus_ω_l3095_309572

noncomputable def f (ω a x : ℝ) : ℝ := Real.sin (ω * x) + a * Real.cos (ω * x)

theorem exists_a_plus_ω : ∃ (a ω : ℝ), 
  ω > 0 ∧ 
  (∀ x, f ω a x = f ω a (2 * Real.pi / 3 - x)) ∧ 
  (∀ x, f ω a (Real.pi / 6) ≤ f ω a x) ∧ 
  0 ≤ a + ω ∧ a + ω ≤ 10 :=
by sorry

end exists_a_plus_ω_l3095_309572


namespace isosceles_right_triangle_expression_l3095_309562

theorem isosceles_right_triangle_expression (a : ℝ) (h : a > 0) :
  (a + 2 * a) / Real.sqrt (a^2 + a^2) = 3 * Real.sqrt 2 / 2 := by
  sorry

end isosceles_right_triangle_expression_l3095_309562


namespace correct_number_of_workers_l3095_309594

/-- The number of workers in the team -/
def n : ℕ := 9

/-- The number of days it takes the full team to complete the task -/
def full_team_days : ℕ := 7

/-- The number of days it takes the team minus two workers to complete the task -/
def team_minus_two_days : ℕ := 14

/-- The number of days it takes the team minus six workers to complete the task -/
def team_minus_six_days : ℕ := 42

/-- The theorem stating that n is the correct number of workers -/
theorem correct_number_of_workers :
  n * full_team_days = (n - 2) * team_minus_two_days ∧
  n * full_team_days = (n - 6) * team_minus_six_days :=
by sorry

end correct_number_of_workers_l3095_309594


namespace constant_term_equals_twenty_implies_n_equals_three_l3095_309528

/-- The constant term in the expansion of (x + 2 + 1/x)^n -/
def constant_term (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- The theorem stating that if the constant term is 20, then n = 3 -/
theorem constant_term_equals_twenty_implies_n_equals_three :
  ∃ n : ℕ, constant_term n = 20 ∧ n = 3 :=
sorry

end constant_term_equals_twenty_implies_n_equals_three_l3095_309528


namespace abigail_report_time_l3095_309535

/-- The time it takes Abigail to finish her report -/
def report_completion_time (total_words : ℕ) (words_per_half_hour : ℕ) (words_written : ℕ) (proofreading_time : ℕ) : ℕ :=
  let words_left := total_words - words_written
  let half_hour_blocks := (words_left + words_per_half_hour - 1) / words_per_half_hour
  let writing_time := half_hour_blocks * 30
  writing_time + proofreading_time

/-- Theorem stating that Abigail will take 225 minutes to finish her report -/
theorem abigail_report_time :
  report_completion_time 1500 250 200 45 = 225 := by
  sorry

end abigail_report_time_l3095_309535


namespace sibling_ages_l3095_309500

theorem sibling_ages (x y z : ℕ) : 
  x - y = 3 →                    -- Age difference between siblings
  z - 1 = 2 * (x + y) →          -- Father's age one year ago
  z + 20 = (x + 20) + (y + 20) → -- Father's age in 20 years
  (x = 11 ∧ y = 8) :=            -- Ages of the siblings
by sorry

end sibling_ages_l3095_309500


namespace negative_cube_equality_l3095_309520

theorem negative_cube_equality : (-3)^3 = -3^3 := by
  sorry

end negative_cube_equality_l3095_309520


namespace nested_average_equals_seven_eighteenths_l3095_309599

/-- Average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- Average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- The main theorem -/
theorem nested_average_equals_seven_eighteenths :
  avg3 (avg3 1 1 0) (avg2 0 1) 0 = 7 / 18 := by
  sorry

end nested_average_equals_seven_eighteenths_l3095_309599


namespace geometric_properties_l3095_309590

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of a line passing through a point
def passes_through (l : Line) (p : Point) : Prop := sorry

-- Define the concept of corresponding angles
def corresponding_angles (a1 a2 : Angle) : Prop := sorry

-- Define vertical angles
def vertical_angles (a1 a2 : Angle) : Prop := sorry

theorem geometric_properties :
  -- Statement 1
  (∀ a b c : Line, parallel a b → parallel b c → parallel a c) ∧
  -- Statement 2
  (∀ a1 a2 : Angle, corresponding_angles a1 a2 → a1 = a2) ∧
  -- Statement 3
  (∀ p : Point, ∀ l : Line, ∃! m : Line, passes_through m p ∧ parallel m l) ∧
  -- Statement 4
  (∀ a1 a2 : Angle, vertical_angles a1 a2 → a1 = a2) :=
by sorry

end geometric_properties_l3095_309590


namespace parallel_line_plane_intersection_false_l3095_309541

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and intersection relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect_planes : Plane → Plane → Line → Prop)

-- Define our specific objects
variable (m n : Line)
variable (α β : Plane)

-- State that m and n are different lines
variable (m_neq_n : m ≠ n)

-- State that α and β are different planes
variable (α_neq_β : α ≠ β)

-- The theorem to be proved
theorem parallel_line_plane_intersection_false :
  ¬(∀ (m n : Line) (α β : Plane), 
    parallel_line_plane m α → intersect_planes α β n → parallel_lines m n) :=
sorry

end parallel_line_plane_intersection_false_l3095_309541


namespace solution_set_of_inequality_l3095_309516

theorem solution_set_of_inequality (x : ℝ) :
  (3 * x^2 + 7 * x ≤ 2) ↔ (-2 ≤ x ∧ x ≤ 1/3) := by
  sorry

end solution_set_of_inequality_l3095_309516


namespace nine_sided_polygon_diagonals_l3095_309553

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end nine_sided_polygon_diagonals_l3095_309553


namespace arithmetic_sequence_of_primes_l3095_309547

theorem arithmetic_sequence_of_primes : ∃ (p q r : ℕ), 
  (Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r) ∧ 
  (p = 127 ∧ q = 3697 ∧ r = 5527) ∧
  (∃ (d : ℕ), 
    q * (q + 1) - p * (p + 1) = d ∧
    r * (r + 1) - q * (q + 1) = d ∧
    p * (p + 1) < q * (q + 1) ∧ q * (q + 1) < r * (r + 1)) :=
by sorry

end arithmetic_sequence_of_primes_l3095_309547


namespace four_digit_divisible_by_50_l3095_309576

theorem four_digit_divisible_by_50 : 
  (Finset.filter 
    (fun n : ℕ => n ≥ 1000 ∧ n < 10000 ∧ n % 100 = 50 ∧ n % 50 = 0) 
    (Finset.range 10000)).card = 90 := by
  sorry

end four_digit_divisible_by_50_l3095_309576


namespace min_value_sqrt_sum_min_value_sqrt_sum_achieved_l3095_309507

theorem min_value_sqrt_sum (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (-x)^2) ≥ 2 * Real.sqrt 2 :=
sorry

theorem min_value_sqrt_sum_achieved : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (-x)^2) = 2 * Real.sqrt 2 :=
sorry

end min_value_sqrt_sum_min_value_sqrt_sum_achieved_l3095_309507


namespace negative_three_times_b_minus_a_l3095_309570

theorem negative_three_times_b_minus_a (a b : ℚ) (h : a - b = 1/2) : -3 * (b - a) = 3/2 := by
  sorry

end negative_three_times_b_minus_a_l3095_309570


namespace shooting_probabilities_l3095_309509

/-- Represents the probability of hitting a specific ring in a shooting event -/
structure ShootingProbability where
  ring10 : ℝ
  ring9 : ℝ
  ring8 : ℝ

/-- Calculates the probability of hitting either the 10-ring or the 9-ring -/
def prob_10_or_9 (p : ShootingProbability) : ℝ :=
  p.ring10 + p.ring9

/-- Calculates the probability of hitting below the 8-ring -/
def prob_below_8 (p : ShootingProbability) : ℝ :=
  1 - (p.ring10 + p.ring9 + p.ring8)

/-- Theorem stating the probabilities for a given shooting event -/
theorem shooting_probabilities (p : ShootingProbability)
  (h1 : p.ring10 = 0.24)
  (h2 : p.ring9 = 0.28)
  (h3 : p.ring8 = 0.19) :
  prob_10_or_9 p = 0.52 ∧ prob_below_8 p = 0.29 := by
  sorry

end shooting_probabilities_l3095_309509


namespace triangle_extension_similarity_l3095_309592

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the similarity of triangles
def SimilarTriangles (t1 t2 : Triangle) : Prop := sorry

-- Define the extension of a line segment
def ExtendSegment (p1 p2 : ℝ × ℝ) (t : ℝ) : ℝ × ℝ := sorry

-- Define the length of a line segment
def SegmentLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem triangle_extension_similarity (ABC : Triangle) (P : ℝ × ℝ) :
  SegmentLength ABC.A ABC.B = 10 →
  SegmentLength ABC.B ABC.C = 9 →
  SegmentLength ABC.C ABC.A = 7 →
  P = ExtendSegment ABC.B ABC.C 1 →
  SimilarTriangles ⟨P, ABC.A, ABC.B⟩ ⟨P, ABC.C, ABC.A⟩ →
  SegmentLength P ABC.C = 31.5 := by
  sorry

end triangle_extension_similarity_l3095_309592


namespace square_of_97_l3095_309597

theorem square_of_97 : 97 * 97 = 9409 := by
  sorry

end square_of_97_l3095_309597


namespace correct_selection_schemes_l3095_309566

/-- Represents the number of translators for each language --/
structure TranslatorCounts where
  total : ℕ
  english : ℕ
  japanese : ℕ
  both : ℕ

/-- Represents the required team sizes --/
structure TeamSizes where
  english : ℕ
  japanese : ℕ

/-- Calculates the number of different selection schemes --/
def selectionSchemes (counts : TranslatorCounts) (sizes : TeamSizes) : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem correct_selection_schemes :
  let counts : TranslatorCounts := ⟨11, 5, 4, 2⟩
  let sizes : TeamSizes := ⟨4, 4⟩
  selectionSchemes counts sizes = 144 := by sorry

end correct_selection_schemes_l3095_309566


namespace prob_at_least_one_karnataka_l3095_309598

/-- The probability of selecting at least one student from Karnataka -/
theorem prob_at_least_one_karnataka (total : ℕ) (karnataka : ℕ) (selected : ℕ)
  (h1 : total = 10)
  (h2 : karnataka = 3)
  (h3 : selected = 4) :
  (1 : ℚ) - (Nat.choose (total - karnataka) selected : ℚ) / (Nat.choose total selected : ℚ) = 5/6 :=
sorry

end prob_at_least_one_karnataka_l3095_309598


namespace gcd_7163_209_l3095_309580

theorem gcd_7163_209 : Nat.gcd 7163 209 = 19 := by
  have h1 : 7163 = 209 * 34 + 57 := by sorry
  have h2 : 209 = 57 * 3 + 38 := by sorry
  have h3 : 57 = 38 * 1 + 19 := by sorry
  have h4 : 38 = 19 * 2 := by sorry
  sorry

end gcd_7163_209_l3095_309580


namespace spaceship_age_conversion_l3095_309519

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The octal representation of the spaceship's age -/
def spaceship_age_octal : List Nat := [3, 4, 7]

theorem spaceship_age_conversion :
  octal_to_decimal spaceship_age_octal = 483 := by
  sorry

#eval octal_to_decimal spaceship_age_octal

end spaceship_age_conversion_l3095_309519


namespace bill_face_value_is_12250_l3095_309502

/-- Calculates the face value of a bill given the true discount, time period, and annual discount rate. -/
def calculate_face_value (true_discount : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  (true_discount * (100 + rate * time)) / (rate * time)

/-- Theorem stating that given the specific conditions, the face value of the bill is 12250. -/
theorem bill_face_value_is_12250 :
  let true_discount : ℚ := 3500
  let time : ℚ := 2
  let rate : ℚ := 20
  calculate_face_value true_discount time rate = 12250 := by
  sorry

#eval calculate_face_value 3500 2 20

end bill_face_value_is_12250_l3095_309502


namespace horner_rule_v2_value_l3095_309530

/-- Horner's Rule evaluation function -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x⁵ + 4x⁴ + x² + 20x + 16 -/
def f : ℝ → ℝ := fun x => x^5 + 4*x^4 + x^2 + 20*x + 16

/-- Coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [1, 4, 0, 1, 20, 16]

theorem horner_rule_v2_value :
  let x := -2
  let v₂ := (horner_eval (f_coeffs.take 3) x)
  v₂ = -4 :=
by sorry

end horner_rule_v2_value_l3095_309530


namespace new_rectangle_area_greater_than_square_l3095_309524

theorem new_rectangle_area_greater_than_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let new_base := 2 * (a + b)
  let new_height := (2 * b + a) / 3
  let square_side := a + b
  new_base * new_height > square_side * square_side :=
by sorry

end new_rectangle_area_greater_than_square_l3095_309524


namespace quadratic_fraction_value_l3095_309561

theorem quadratic_fraction_value (x : ℝ) (h : x^2 - 3*x + 1 = 0) : 
  x^2 / (x^4 + x^2 + 1) = 1/8 := by
  sorry

end quadratic_fraction_value_l3095_309561


namespace sequence_kth_term_value_l3095_309582

/-- Given a sequence {a_n} with sum S_n = n^2 - 9n and 5 < a_k < 8, prove k = 8 -/
theorem sequence_kth_term_value (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ) :
  (∀ n, S n = n^2 - 9*n) →
  (∀ n, a n = 2*n - 10) →
  (5 < a k ∧ a k < 8) →
  k = 8 := by
sorry

end sequence_kth_term_value_l3095_309582


namespace polynomial_division_existence_l3095_309525

theorem polynomial_division_existence :
  ∃ (Q R : Polynomial ℚ),
    4 * X^5 - 7 * X^4 + 3 * X^3 + 9 * X^2 - 23 * X + 8 = (5 * X^2 + 2 * X - 1) * Q + R ∧
    R.degree < (5 * X^2 + 2 * X - 1).degree := by
  sorry

end polynomial_division_existence_l3095_309525


namespace ninety_degrees_to_radians_l3095_309558

theorem ninety_degrees_to_radians :
  let degrees_to_radians (d : ℝ) : ℝ := d * (π / 180)
  degrees_to_radians 90 = π / 2 := by sorry

end ninety_degrees_to_radians_l3095_309558


namespace reciprocal_and_inverse_sum_l3095_309565

theorem reciprocal_and_inverse_sum (a b : ℚ) (ha : a = 1 / a) (hb : b = -b) :
  a^2007 + b^2007 = 1 ∨ a^2007 + b^2007 = -1 := by
  sorry

end reciprocal_and_inverse_sum_l3095_309565


namespace train_to_subway_ratio_l3095_309506

/-- Represents the travel times for Andrew's journey from Manhattan to the Bronx -/
structure TravelTimes where
  total : ℝ
  subway : ℝ
  biking : ℝ
  train : ℝ

/-- Theorem stating the ratio of train time to subway time -/
theorem train_to_subway_ratio (t : TravelTimes) 
  (h1 : t.total = 38)
  (h2 : t.subway = 10)
  (h3 : t.biking = 8)
  (h4 : t.train = t.total - t.subway - t.biking) :
  t.train / t.subway = 2 := by
  sorry

#check train_to_subway_ratio

end train_to_subway_ratio_l3095_309506


namespace password_guess_probability_l3095_309545

/-- The number of digits in the password -/
def password_length : ℕ := 6

/-- The number of possible digits for each position -/
def digit_options : ℕ := 10

/-- The number of attempts allowed -/
def max_attempts : ℕ := 2

/-- The probability of guessing the correct last digit in no more than 2 attempts -/
theorem password_guess_probability :
  (1 : ℚ) / digit_options + (digit_options - 1 : ℚ) / digit_options * (1 : ℚ) / (digit_options - 1) = 1 / 5 := by
  sorry

end password_guess_probability_l3095_309545


namespace min_value_expression_equality_condition_l3095_309579

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 4 / (x + y)^2 ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 4 / (x + y)^2 = 2 * Real.sqrt 2 ↔ x = y ∧ x = (Real.sqrt 2)^(3/4) :=
by sorry

end min_value_expression_equality_condition_l3095_309579


namespace dacid_weighted_average_l3095_309584

/-- Calculates the weighted average grade given marks and weights -/
def weighted_average (marks : List ℝ) (weights : List ℝ) : ℝ :=
  (List.zip marks weights).map (fun (m, w) => m * w) |>.sum

/-- Theorem: Dacid's weighted average grade is 90.8 -/
theorem dacid_weighted_average :
  let marks := [96, 95, 82, 87, 92]
  let weights := [0.20, 0.25, 0.15, 0.25, 0.15]
  weighted_average marks weights = 90.8 := by
sorry

#eval weighted_average [96, 95, 82, 87, 92] [0.20, 0.25, 0.15, 0.25, 0.15]

end dacid_weighted_average_l3095_309584


namespace triangle_third_side_length_l3095_309559

/-- A triangle with two sides of length 3 and 6 can have a third side of length 6 -/
theorem triangle_third_side_length : ∃ (a b c : ℝ), 
  a = 3 ∧ b = 6 ∧ c = 6 ∧ 
  a + b > c ∧ b + c > a ∧ a + c > b ∧
  a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end triangle_third_side_length_l3095_309559


namespace rectangle_area_error_l3095_309543

theorem rectangle_area_error (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let erroneous_area := (1.02 * L) * (1.03 * W)
  let correct_area := L * W
  let percentage_error := (erroneous_area - correct_area) / correct_area * 100
  percentage_error = 5.06 := by
sorry

end rectangle_area_error_l3095_309543


namespace min_max_sum_sqrt_l3095_309529

theorem min_max_sum_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a^2 + b^2 + c^2 = 6) :
  2 + Real.sqrt 2 ≤ Real.sqrt (4 - a^2) + Real.sqrt (4 - b^2) + Real.sqrt (4 - c^2) ∧
  Real.sqrt (4 - a^2) + Real.sqrt (4 - b^2) + Real.sqrt (4 - c^2) ≤ 3 * Real.sqrt 2 :=
by sorry

end min_max_sum_sqrt_l3095_309529


namespace line_translation_l3095_309573

-- Define the original line
def original_line (x : ℝ) : ℝ := -2 * x + 3

-- Define the translation
def translation : ℝ := 2

-- Define the translated line
def translated_line (x : ℝ) : ℝ := -2 * x + 1

-- Theorem statement
theorem line_translation :
  ∀ x : ℝ, translated_line x = original_line x - translation := by
  sorry

end line_translation_l3095_309573


namespace ice_cream_arrangement_count_l3095_309563

theorem ice_cream_arrangement_count : Nat.factorial 5 = 120 := by
  sorry

end ice_cream_arrangement_count_l3095_309563


namespace sandwich_jam_cost_l3095_309585

theorem sandwich_jam_cost :
  ∀ (N B J : ℕ),
  N > 1 →
  B > 0 →
  J > 0 →
  N * (3 * B + 7 * J) = 378 →
  (N * J * 7 : ℚ) / 100 = 2.52 := by
sorry

end sandwich_jam_cost_l3095_309585


namespace carlas_marbles_l3095_309544

/-- Given that Carla had some marbles and bought more, prove how many she has now. -/
theorem carlas_marbles (initial : ℝ) (bought : ℝ) (total : ℝ) 
  (h1 : initial = 187.0) 
  (h2 : bought = 134.0) 
  (h3 : total = initial + bought) : 
  total = 321.0 := by
  sorry


end carlas_marbles_l3095_309544


namespace isosceles_right_triangles_in_quadrilateral_l3095_309515

-- Define the points
variable (A B C D O₁ O₂ O₃ O₄ : Point)

-- Define the quadrilateral
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define isosceles right triangle
def is_isosceles_right_triangle (X Y Z : Point) : Prop := sorry

-- State the theorem
theorem isosceles_right_triangles_in_quadrilateral 
  (h_quad : is_convex_quadrilateral A B C D)
  (h_ABO₁ : is_isosceles_right_triangle A B O₁)
  (h_BCO₂ : is_isosceles_right_triangle B C O₂)
  (h_CDO₃ : is_isosceles_right_triangle C D O₃)
  (h_DAO₄ : is_isosceles_right_triangle D A O₄)
  (h_O₁_O₃ : O₁ = O₃) :
  O₂ = O₄ := by sorry

end isosceles_right_triangles_in_quadrilateral_l3095_309515


namespace line_in_plane_theorem_l3095_309518

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)
variable (passes_through : Line → Point → Prop)
variable (point_in_plane : Point → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_in_plane_theorem 
  (a b : Line) (α : Plane) (M : Point)
  (h1 : parallel_line_plane a α)
  (h2 : parallel_line_line b a)
  (h3 : passes_through b M)
  (h4 : point_in_plane M α) :
  line_in_plane b α :=
sorry

end line_in_plane_theorem_l3095_309518


namespace revenue_change_specific_l3095_309505

/-- Calculates the change in revenue given price increase, sales decrease, loyalty discount, and sales tax -/
def revenueChange (priceIncrease salesDecrease loyaltyDiscount salesTax : ℝ) : ℝ :=
  let newPrice := 1 + priceIncrease
  let newSales := 1 - salesDecrease
  let discountedPrice := newPrice * (1 - loyaltyDiscount)
  let finalPrice := discountedPrice * (1 + salesTax)
  finalPrice * newSales - 1

/-- The revenue change given specific conditions -/
theorem revenue_change_specific : 
  revenueChange 0.9 0.3 0.1 0.15 = 0.37655 := by sorry

end revenue_change_specific_l3095_309505


namespace shirt_price_l3095_309503

theorem shirt_price (total_items : Nat) (dress_count : Nat) (shirt_count : Nat)
  (total_money : ℕ) (dress_price : ℕ) :
  total_items = dress_count + shirt_count →
  total_money = dress_count * dress_price + shirt_count * (total_money - dress_count * dress_price) / shirt_count →
  dress_count = 7 →
  shirt_count = 4 →
  total_money = 69 →
  dress_price = 7 →
  (total_money - dress_count * dress_price) / shirt_count = 5 := by
sorry

end shirt_price_l3095_309503


namespace min_cuts_for_polygons_l3095_309539

theorem min_cuts_for_polygons (n : ℕ) (k : ℕ) (s : ℕ) : 
  n = 73 ∧ s = 30 ∧ k ≥ n - 1 ∧ 
  (n * ((s - 2) * π) + (k + 1 - n) * π ≤ (k + 1) * 2 * π) →
  k ≥ 1970 :=
sorry

end min_cuts_for_polygons_l3095_309539


namespace binomial_1350_2_l3095_309526

theorem binomial_1350_2 : Nat.choose 1350 2 = 910575 := by sorry

end binomial_1350_2_l3095_309526


namespace firewood_collection_l3095_309571

theorem firewood_collection (total kimberley ela : ℕ) (h1 : total = 35) (h2 : kimberley = 10) (h3 : ela = 13) :
  total - kimberley - ela = 12 := by
  sorry

end firewood_collection_l3095_309571


namespace sine_increases_with_angle_not_always_isosceles_right_angle_condition_obtuse_angle_from_ratio_l3095_309536

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  sum_angles : A + B + C = Real.pi
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Statement A
theorem sine_increases_with_angle (t : Triangle) :
  t.A > t.B → Real.sin t.A > Real.sin t.B :=
sorry

-- Statement B
theorem not_always_isosceles (t : Triangle) :
  Real.sin (2 * t.A) = Real.sin (2 * t.B) →
  ¬(t.A = t.B ∧ t.a = t.b) :=
sorry

-- Statement C
theorem right_angle_condition (t : Triangle) :
  t.a * Real.cos t.B - t.b * Real.cos t.A = t.c →
  t.C = Real.pi / 2 :=
sorry

-- Statement D
theorem obtuse_angle_from_ratio (t : Triangle) :
  ∃ (k : ℝ), k > 0 ∧ t.a = 3*k ∧ t.b = 5*k ∧ t.c = 7*k →
  t.C > Real.pi / 2 :=
sorry

end sine_increases_with_angle_not_always_isosceles_right_angle_condition_obtuse_angle_from_ratio_l3095_309536


namespace marias_carrots_l3095_309578

theorem marias_carrots : 
  ∃ (initial_carrots : ℕ), 
    initial_carrots - 11 + 15 = 52 ∧ 
    initial_carrots = 48 := by
  sorry

end marias_carrots_l3095_309578


namespace expression_simplification_l3095_309557

theorem expression_simplification (a b : ℝ) : 
  (8 * a^3 * b) * (4 * a * b^2) * (1 / (2 * a * b)^3) = 4 * a := by
  sorry

end expression_simplification_l3095_309557


namespace files_remaining_l3095_309564

theorem files_remaining (music_files video_files deleted_files : ℕ) :
  music_files = 4 →
  video_files = 21 →
  deleted_files = 23 →
  music_files + video_files - deleted_files = 2 :=
by
  sorry

end files_remaining_l3095_309564


namespace blackbirds_per_tree_l3095_309514

theorem blackbirds_per_tree (num_trees : ℕ) (num_magpies : ℕ) (total_birds : ℕ) 
  (h1 : num_trees = 7)
  (h2 : num_magpies = 13)
  (h3 : total_birds = 34)
  (h4 : ∃ (blackbirds_per_tree : ℕ), num_trees * blackbirds_per_tree + num_magpies = total_birds) :
  ∃ (blackbirds_per_tree : ℕ), blackbirds_per_tree = 3 ∧ 
    num_trees * blackbirds_per_tree + num_magpies = total_birds :=
by
  sorry

end blackbirds_per_tree_l3095_309514


namespace expo_volunteer_selection_l3095_309556

/-- The number of volunteers --/
def total_volunteers : ℕ := 5

/-- The number of volunteers to be selected --/
def selected_volunteers : ℕ := 4

/-- The number of tasks --/
def total_tasks : ℕ := 4

/-- The number of restricted tasks --/
def restricted_tasks : ℕ := 2

/-- The number of volunteers restricted to certain tasks --/
def restricted_volunteers : ℕ := 2

/-- The number of unrestricted volunteers --/
def unrestricted_volunteers : ℕ := total_volunteers - restricted_volunteers

theorem expo_volunteer_selection :
  (Nat.choose restricted_volunteers 1 * Nat.choose restricted_tasks 1 * (unrestricted_volunteers).factorial) +
  (Nat.choose restricted_volunteers 2 * (unrestricted_volunteers).factorial) = 36 := by
  sorry

end expo_volunteer_selection_l3095_309556


namespace hyperbola_asymptote_l3095_309575

theorem hyperbola_asymptote (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (4 - m) + y^2 / (m - 2) = 1 → 
    (y = (1/3) * x ∨ y = -(1/3) * x)) → 
  m = 7/4 := by
sorry

end hyperbola_asymptote_l3095_309575


namespace average_cost_is_two_l3095_309510

/-- The average cost of fruit given the prices and quantities of apples, bananas, and oranges. -/
def average_cost (apple_price banana_price orange_price : ℚ) 
                 (apple_qty banana_qty orange_qty : ℕ) : ℚ :=
  let total_cost := apple_price * apple_qty + banana_price * banana_qty + orange_price * orange_qty
  let total_qty := apple_qty + banana_qty + orange_qty
  total_cost / total_qty

/-- Theorem stating that the average cost of fruit is $2 given the specified prices and quantities. -/
theorem average_cost_is_two :
  average_cost 2 1 3 12 4 4 = 2 := by
  sorry

end average_cost_is_two_l3095_309510


namespace factor_expression_l3095_309517

theorem factor_expression (x y : ℝ) : -x^2*y + 6*y^2*x - 9*y^3 = -y*(x-3*y)^2 := by
  sorry

end factor_expression_l3095_309517


namespace smallest_base_for_xyxy_cube_l3095_309546

/-- Represents a number in the form xyxy in base b -/
def xyxy_form (x y b : ℕ) : ℕ := x * b^3 + y * b^2 + x * b + y

/-- Checks if a number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

/-- The statement to be proved -/
theorem smallest_base_for_xyxy_cube : 
  (∀ x y : ℕ, ¬ is_perfect_cube (xyxy_form x y 10)) →
  (∃ x y : ℕ, is_perfect_cube (xyxy_form x y 7)) ∧
  (∀ b : ℕ, 1 < b → b < 7 → ∀ x y : ℕ, ¬ is_perfect_cube (xyxy_form x y b)) :=
sorry

end smallest_base_for_xyxy_cube_l3095_309546


namespace max_sum_of_factors_l3095_309549

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 2310 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 2310 → 
  A + B + C ≤ X + Y + Z → A + B + C ≤ 390 := by
sorry

end max_sum_of_factors_l3095_309549


namespace g_increasing_on_negative_l3095_309550

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions on f
variable (h1 : ∀ x y, x < y → f x < f y)  -- f is increasing
variable (h2 : ∀ x, f x < 0)  -- f is always negative

-- Define the function g
def g (x : ℝ) : ℝ := x^2 * f x

-- State the theorem
theorem g_increasing_on_negative (x y : ℝ) (hx : x < 0) (hy : y < 0) (hxy : x < y) :
  g f x < g f y := by sorry

end g_increasing_on_negative_l3095_309550


namespace profit_percentage_l3095_309531

theorem profit_percentage (selling_price cost_price : ℝ) (h : cost_price = 0.81 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100/81 - 1) * 100 := by
  sorry

end profit_percentage_l3095_309531


namespace single_point_equation_l3095_309595

/-- If the equation 3x^2 + y^2 + 6x - 12y + d = 0 represents a single point, then d = 39 -/
theorem single_point_equation (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + p.2^2 + 6 * p.1 - 12 * p.2 + d = 0) → d = 39 := by
  sorry

end single_point_equation_l3095_309595


namespace lydia_planting_age_l3095_309574

/-- Represents the age of Lydia when she planted the tree -/
def planting_age : ℕ := sorry

/-- The time it takes for an apple tree to bear fruit -/
def fruit_bearing_time : ℕ := 7

/-- Lydia's current age -/
def current_age : ℕ := 9

/-- Lydia's age when she first eats an apple from her tree -/
def first_apple_age : ℕ := 11

theorem lydia_planting_age : 
  planting_age = first_apple_age - fruit_bearing_time := by sorry

end lydia_planting_age_l3095_309574


namespace largest_A_for_divisibility_by_3_l3095_309521

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem largest_A_for_divisibility_by_3 :
  ∀ A : ℕ, A ≤ 9 →
    is_divisible_by_3 (3 * 100000 + A * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + 2) →
    A ≤ 9 :=
by sorry

end largest_A_for_divisibility_by_3_l3095_309521


namespace sandy_comic_books_l3095_309554

theorem sandy_comic_books :
  ∃ (initial : ℕ), (initial / 2 + 6 : ℕ) = 13 ∧ initial = 14 :=
by sorry

end sandy_comic_books_l3095_309554


namespace sqrt_inequality_abc_inequality_l3095_309588

-- Problem 1
theorem sqrt_inequality : Real.sqrt 7 + Real.sqrt 13 < 3 + Real.sqrt 11 := by sorry

-- Problem 2
theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * (b^2 + c^2) + b * (c^2 + a^2) + c * (a^2 + b^2) ≥ 6 * a * b * c := by sorry

end sqrt_inequality_abc_inequality_l3095_309588


namespace divide_flour_possible_l3095_309527

/-- Represents the result of a weighing operation -/
inductive Weighing
| MeasuredFlour (amount : ℕ)
| CombinedFlour (amount1 amount2 : ℕ)

/-- Represents a weighing operation using the balance scale -/
def weigh (yeast ginger flour : ℕ) : Weighing :=
  sorry

/-- Represents the process of dividing flour using two weighings -/
def divideFlour (totalFlour yeast ginger : ℕ) : Option (ℕ × ℕ) :=
  sorry

/-- Theorem stating that it's possible to divide 500g of flour into 400g and 100g parts
    using only two weighings with 5g of yeast and 30g of ginger -/
theorem divide_flour_possible :
  ∃ (w1 w2 : Weighing),
    let result := divideFlour 500 5 30
    result = some (400, 100) ∧
    (∃ (f1 : ℕ), w1 = Weighing.MeasuredFlour f1) ∧
    (∃ (f2 : ℕ), w2 = Weighing.MeasuredFlour f2) ∧
    f1 + f2 = 100 :=
  sorry

end divide_flour_possible_l3095_309527


namespace slower_pump_fill_time_l3095_309504

/-- Represents a water pump with a constant fill rate -/
structure Pump where
  rate : ℝ
  rate_positive : rate > 0

/-- Represents a swimming pool -/
structure Pool where
  volume : ℝ
  volume_positive : volume > 0

/-- Theorem stating the time taken by the slower pump to fill the pool alone -/
theorem slower_pump_fill_time (pool : Pool) (pump1 pump2 : Pump)
    (h1 : pump2.rate = 1.5 * pump1.rate)
    (h2 : (pump1.rate + pump2.rate) * 5 = pool.volume) :
    pump1.rate * 12.5 = pool.volume := by
  sorry

end slower_pump_fill_time_l3095_309504


namespace shorter_wall_area_l3095_309540

/-- Given a rectangular hall with specified dimensions, calculate the area of the shorter wall. -/
theorem shorter_wall_area (floor_area : ℝ) (longer_wall_area : ℝ) (height : ℝ) :
  floor_area = 20 →
  longer_wall_area = 10 →
  height = 40 →
  let length := longer_wall_area / height
  let width := floor_area / length
  width * height = 3200 := by
  sorry

#check shorter_wall_area

end shorter_wall_area_l3095_309540


namespace eunji_shopping_l3095_309552

theorem eunji_shopping (initial_money : ℝ) : 
  initial_money * (1 - 1/4) * (1 - 1/3) = 1600 → initial_money = 3200 := by
  sorry

end eunji_shopping_l3095_309552


namespace final_highway_length_l3095_309534

def highway_extension (initial_length day1_construction day2_multiplier additional_miles : ℕ) : ℕ := 
  initial_length + day1_construction + day1_construction * day2_multiplier + additional_miles

theorem final_highway_length :
  highway_extension 200 50 3 250 = 650 := by
  sorry

end final_highway_length_l3095_309534


namespace absolute_value_nonnegative_and_two_plus_two_not_zero_l3095_309577

theorem absolute_value_nonnegative_and_two_plus_two_not_zero :
  (∀ x : ℝ, |x| ≥ 0) ∧ ¬(2 + 2 = 0) := by
  sorry

end absolute_value_nonnegative_and_two_plus_two_not_zero_l3095_309577


namespace count_numbers_with_at_most_three_digits_is_900_l3095_309589

/-- Count of positive integers less than 1000 with at most three different digits -/
def count_numbers_with_at_most_three_digits : ℕ :=
  -- Single-digit numbers
  9 +
  -- Two-digit numbers
  (9 + 144 + 9) +
  -- Three-digit numbers
  (9 + 216 + 504)

/-- Theorem stating that the count of positive integers less than 1000
    with at most three different digits is 900 -/
theorem count_numbers_with_at_most_three_digits_is_900 :
  count_numbers_with_at_most_three_digits = 900 := by
  sorry

#eval count_numbers_with_at_most_three_digits

end count_numbers_with_at_most_three_digits_is_900_l3095_309589


namespace photo_size_is_1_5_l3095_309555

/-- The size of a photo in kilobytes -/
def photo_size : ℝ := sorry

/-- The total storage space of the drive in kilobytes -/
def total_storage : ℝ := 2000 * photo_size

/-- The space used by 400 photos in kilobytes -/
def space_400_photos : ℝ := 400 * photo_size

/-- The space used by 12 200-kilobyte videos in kilobytes -/
def space_12_videos : ℝ := 12 * 200

/-- Theorem stating that the size of each photo is 1.5 kilobytes -/
theorem photo_size_is_1_5 : photo_size = 1.5 := by
  have h1 : total_storage = space_400_photos + space_12_videos := sorry
  sorry

end photo_size_is_1_5_l3095_309555


namespace square_area_PS_l3095_309596

-- Define the triangles and their properties
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0

-- Define the configuration
def Configuration (P Q R S : ℝ × ℝ) : Prop :=
  Triangle P Q R ∧ Triangle P R S ∧
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 25 ∧
  (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = 4 ∧
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 49

-- State the theorem
theorem square_area_PS (P Q R S : ℝ × ℝ) 
  (h : Configuration P Q R S) : 
  (S.1 - P.1)^2 + (S.2 - P.2)^2 = 53 := by
  sorry

end square_area_PS_l3095_309596


namespace set_a_forms_triangle_l3095_309533

/-- Triangle Inequality Theorem: A set of three line segments can form a triangle
    if and only if the sum of the lengths of any two sides is strictly greater
    than the length of the remaining side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that the set of line segments (3, 5, 7) can form a triangle. -/
theorem set_a_forms_triangle : can_form_triangle 3 5 7 := by
  sorry

end set_a_forms_triangle_l3095_309533


namespace sum_of_roots_l3095_309586

theorem sum_of_roots (k : ℝ) (a₁ a₂ : ℝ) (h₁ : a₁ ≠ a₂) 
  (h₂ : 5 * a₁^2 + k * a₁ - 2 = 0) (h₃ : 5 * a₂^2 + k * a₂ - 2 = 0) : 
  a₁ + a₂ = -k / 5 := by
sorry

end sum_of_roots_l3095_309586


namespace cuboid_surface_area_example_l3095_309508

/-- The surface area of a cuboid with given dimensions -/
def cuboidSurfaceArea (length breadth height : ℝ) : ℝ :=
  2 * (length * height + length * breadth + breadth * height)

/-- Theorem: The surface area of a cuboid with length 10 cm, breadth 8 cm, and height 6 cm is 376 cm² -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 10 8 6 = 376 := by
  sorry

end cuboid_surface_area_example_l3095_309508


namespace derivative_of_f_l3095_309548

noncomputable def f (x : ℝ) : ℝ := x + 1 / x

theorem derivative_of_f :
  ∀ x : ℝ, x ≠ 0 → deriv f x = 1 - 1 / x^2 := by sorry

end derivative_of_f_l3095_309548


namespace sally_coin_problem_l3095_309591

/-- Represents Sally's coin collection --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ

/-- Represents the changes in Sally's coin collection --/
def update_collection (initial : CoinCollection) (dad_nickels mom_nickels : ℕ) : CoinCollection :=
  { pennies := initial.pennies,
    nickels := initial.nickels + dad_nickels + mom_nickels }

theorem sally_coin_problem (initial : CoinCollection) (dad_nickels mom_nickels : ℕ) 
  (h1 : initial.nickels = 7)
  (h2 : dad_nickels = 9)
  (h3 : mom_nickels = 2)
  (h4 : (update_collection initial dad_nickels mom_nickels).nickels = 18) :
  initial.nickels = 7 ∧ ∀ (p : ℕ), ∃ (initial' : CoinCollection), 
    initial'.pennies = p ∧ 
    initial'.nickels = initial.nickels ∧
    (update_collection initial' dad_nickels mom_nickels).nickels = 18 :=
sorry

end sally_coin_problem_l3095_309591


namespace x_range_theorem_l3095_309560

theorem x_range_theorem (x : ℝ) : 
  (∀ (a b : ℝ), a > 0 → b > 0 → |x + 1| + |x - 2| ≤ (a + 1/b) * (1/a + b)) →
  -3/2 ≤ x ∧ x ≤ 5/2 := by
  sorry

end x_range_theorem_l3095_309560


namespace angle_with_same_terminal_side_as_negative_265_l3095_309593

-- Define the concept of angles having the same terminal side
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

-- State the theorem
theorem angle_with_same_terminal_side_as_negative_265 :
  same_terminal_side (-265) 95 :=
sorry

end angle_with_same_terminal_side_as_negative_265_l3095_309593


namespace friend_team_assignment_l3095_309512

theorem friend_team_assignment (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k^n = 65536 := by
  sorry

end friend_team_assignment_l3095_309512


namespace books_sold_correct_l3095_309581

/-- The number of books Paul sold in a garage sale -/
def books_sold : ℕ := 94

/-- The initial number of books Paul had -/
def initial_books : ℕ := 2

/-- The number of new books Paul bought -/
def new_books : ℕ := 150

/-- The final number of books Paul has -/
def final_books : ℕ := 58

/-- Theorem stating that the number of books Paul sold is correct -/
theorem books_sold_correct : 
  initial_books - books_sold + new_books = final_books :=
by sorry

end books_sold_correct_l3095_309581


namespace trigonometric_equation_solution_l3095_309522

theorem trigonometric_equation_solution 
  (a b c : ℝ) 
  (h : a ≠ 0 ∨ b ≠ 0) :
  ∃ (n : ℤ), 
    ∀ (x : ℝ), 
      a * Real.sin x + b * Real.cos x = c → 
        x = Real.arctan (b / a) + n * Real.pi :=
sorry

end trigonometric_equation_solution_l3095_309522


namespace room_length_calculation_l3095_309537

/-- The length of a room given carpet and room dimensions -/
theorem room_length_calculation (total_cost carpet_width_cm carpet_price_per_m room_breadth : ℝ)
  (h1 : total_cost = 810)
  (h2 : carpet_width_cm = 75)
  (h3 : carpet_price_per_m = 4.5)
  (h4 : room_breadth = 7.5) :
  total_cost / (carpet_price_per_m * room_breadth * (carpet_width_cm / 100)) = 18 := by
  sorry

end room_length_calculation_l3095_309537


namespace octal_addition_47_56_l3095_309513

/-- Represents a digit in the octal system -/
def OctalDigit := Fin 8

/-- Represents a number in the octal system as a list of digits -/
def OctalNumber := List OctalDigit

/-- Addition operation for octal numbers -/
def octal_add : OctalNumber → OctalNumber → OctalNumber := sorry

/-- Conversion from a natural number to an octal number -/
def nat_to_octal : ℕ → OctalNumber := sorry

/-- Conversion from an octal number to a natural number -/
def octal_to_nat : OctalNumber → ℕ := sorry

theorem octal_addition_47_56 :
  octal_add (nat_to_octal 47) (nat_to_octal 56) = nat_to_octal 125 := by sorry

end octal_addition_47_56_l3095_309513


namespace divisible_by_six_l3095_309523

theorem divisible_by_six (a : ℤ) : 6 ∣ a * (a + 1) * (2 * a + 1) := by
  sorry

end divisible_by_six_l3095_309523


namespace johns_final_push_time_l3095_309501

/-- The time of John's final push in a race, given specific conditions. -/
theorem johns_final_push_time (john_speed steve_speed initial_distance final_distance : ℝ) 
  (h1 : john_speed = 4.2)
  (h2 : steve_speed = 3.7)
  (h3 : initial_distance = 14)
  (h4 : final_distance = 2)
  : (initial_distance + final_distance) / john_speed = 16 / 4.2 := by
  sorry

#eval (14 + 2) / 4.2

end johns_final_push_time_l3095_309501


namespace not_always_true_parallel_transitivity_l3095_309567

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel_lines : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem not_always_true_parallel_transitivity 
  (a b : Line) (α : Plane) : 
  ¬(∀ (a b : Line) (α : Plane), 
    parallel_lines a b → 
    parallel_line_plane a α → 
    parallel_line_plane b α) :=
by
  sorry


end not_always_true_parallel_transitivity_l3095_309567


namespace range_of_function_l3095_309538

theorem range_of_function : 
  ∀ y : ℝ, ∃ x : ℝ, |x + 3| - |x - 5| + 3 * x = y := by
sorry

end range_of_function_l3095_309538


namespace expression_evaluation_l3095_309583

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/5
  (2*x - 3)^2 - (x + 2*y)*(x - 2*y) - 3*y^2 + 3 = 1/25 := by
  sorry

end expression_evaluation_l3095_309583


namespace range_of_m_l3095_309551

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) ↔ m ∈ Set.Icc (-3) 5 :=
sorry

end range_of_m_l3095_309551


namespace probability_theorem_l3095_309587

/-- Probability of reaching point (0, n) for a particle with given movement rules -/
def probability_reach_n (n : ℕ) : ℚ :=
  2/3 + 1/12 * (1 - (-1/3)^(n-1))

/-- Movement rules for the particle -/
structure MovementRules where
  prob_move_1 : ℚ := 2/3
  prob_move_2 : ℚ := 1/3
  vector_1 : Fin 2 → ℤ := ![0, 1]
  vector_2 : Fin 2 → ℤ := ![0, 2]

theorem probability_theorem (n : ℕ) (rules : MovementRules) :
  probability_reach_n n =
  2/3 + 1/12 * (1 - (-1/3)^(n-1)) :=
by sorry

end probability_theorem_l3095_309587


namespace local_arts_students_percentage_l3095_309542

/-- Proves that the percentage of local arts students is 50% --/
theorem local_arts_students_percentage
  (total_arts : ℕ)
  (total_science : ℕ)
  (total_commerce : ℕ)
  (science_local_percentage : ℚ)
  (commerce_local_percentage : ℚ)
  (total_local_percentage : ℕ)
  (h_total_arts : total_arts = 400)
  (h_total_science : total_science = 100)
  (h_total_commerce : total_commerce = 120)
  (h_science_local : science_local_percentage = 25/100)
  (h_commerce_local : commerce_local_percentage = 85/100)
  (h_total_local : total_local_percentage = 327)
  : ∃ (arts_local_percentage : ℚ),
    arts_local_percentage = 50/100 ∧
    arts_local_percentage * total_arts +
    science_local_percentage * total_science +
    commerce_local_percentage * total_commerce =
    total_local_percentage := by
  sorry


end local_arts_students_percentage_l3095_309542


namespace functional_equation_solution_l3095_309568

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → x * f y - y * f x = x * y * f (x / y)

/-- Theorem stating that for any function satisfying the functional equation, f(50) = 0 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 50 = 0 := by
  sorry

end functional_equation_solution_l3095_309568


namespace megacorp_fine_l3095_309532

/-- MegaCorp's fine calculation --/
theorem megacorp_fine :
  let daily_mining_profit : ℕ := 3000000
  let daily_oil_profit : ℕ := 5000000
  let monthly_expenses : ℕ := 30000000
  let days_per_year : ℕ := 365
  let months_per_year : ℕ := 12
  let fine_percentage : ℚ := 1 / 100

  let annual_revenue : ℕ := (daily_mining_profit + daily_oil_profit) * days_per_year
  let annual_expenses : ℕ := monthly_expenses * months_per_year
  let annual_profit : ℕ := annual_revenue - annual_expenses
  let fine : ℚ := (annual_profit : ℚ) * fine_percentage

  fine = 25600000 := by sorry

end megacorp_fine_l3095_309532


namespace sum_difference_1500_l3095_309569

/-- The sum of the first n odd counting numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The sum of the first n even counting numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even counting numbers
    and the sum of the first n odd counting numbers -/
def sumDifference (n : ℕ) : ℕ := sumEvenNumbers n - sumOddNumbers n

theorem sum_difference_1500 :
  sumDifference 1500 = 1500 := by
  sorry

end sum_difference_1500_l3095_309569
