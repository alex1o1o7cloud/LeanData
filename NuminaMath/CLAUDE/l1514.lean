import Mathlib

namespace order_of_abc_l1514_151472

theorem order_of_abc (a b c : ℝ) : 
  a = 2 * Real.log 1.01 →
  b = Real.log 1.02 →
  c = Real.sqrt 1.04 - 1 →
  c < a ∧ a < b :=
by sorry

end order_of_abc_l1514_151472


namespace quadratic_inequality_properties_l1514_151475

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x > 0}

theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = Set.Ioo (-3) 1) :
  b < 0 ∧ c > 0 ∧
  {x : ℝ | a * x - b < 0} = Set.Ioi 2 ∧
  {x : ℝ | a * x^2 - b * x + c < 0} = Set.Iic (-1) ∪ Set.Ioi 3 :=
sorry

end quadratic_inequality_properties_l1514_151475


namespace minimum_distance_point_l1514_151427

/-- The point that minimizes the sum of distances to two fixed points lies on the line connecting those points -/
theorem minimum_distance_point (P Q R : ℝ × ℝ) :
  P.1 = -2 ∧ P.2 = -3 ∧
  Q.1 = 5 ∧ Q.2 = 3 ∧
  R.1 = 2 →
  (∀ m : ℝ, (Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + 
              Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)) ≥
             (Real.sqrt ((R.1 - P.1)^2 + ((3/7) - P.2)^2) + 
              Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - (3/7))^2))) →
  R.2 = 3/7 := by
  sorry

end minimum_distance_point_l1514_151427


namespace max_value_and_min_side_l1514_151454

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sin x - m * Real.cos x

theorem max_value_and_min_side (m : ℝ) (A B C : ℝ) (a b c : ℝ) :
  (∀ x, f m x ≤ f m (π/3)) →  -- f achieves maximum at π/3
  f m (A - π/2) = 0 →         -- condition on angle A
  2 * b + c = 3 →             -- condition on sides b and c
  0 < A ∧ A < π →             -- A is a valid angle
  0 < B ∧ B < π →             -- B is a valid angle
  0 < C ∧ C < π →             -- C is a valid angle
  a > 0 ∧ b > 0 ∧ c > 0 →     -- sides are positive
  A + B + C = π →             -- sum of angles in a triangle
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →  -- cosine rule
  m = -Real.sqrt 3 / 3 ∧ a ≥ 3 * Real.sqrt 21 / 14 :=
by sorry

end max_value_and_min_side_l1514_151454


namespace family_age_average_l1514_151479

/-- Given the ages of four family members with specific relationships, 
    prove that their average age is 31.5 years. -/
theorem family_age_average (devin_age eden_age mom_age grandfather_age : ℕ) :
  devin_age = 12 →
  eden_age = 2 * devin_age →
  mom_age = 2 * eden_age →
  grandfather_age = (devin_age + eden_age + mom_age) / 2 →
  (devin_age + eden_age + mom_age + grandfather_age : ℚ) / 4 = 31.5 := by
  sorry

end family_age_average_l1514_151479


namespace train_crossing_time_l1514_151407

/-- Proves that a train with given length and speed takes a specific time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 1500 ∧ 
  train_speed_kmh = 108 →
  crossing_time = 50 :=
by
  sorry

#check train_crossing_time

end train_crossing_time_l1514_151407


namespace sin_sum_less_than_sum_of_sins_l1514_151455

theorem sin_sum_less_than_sum_of_sins (x y : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 (Real.pi / 2) →
  Real.sin (x + y) < Real.sin x + Real.sin y :=
by sorry

end sin_sum_less_than_sum_of_sins_l1514_151455


namespace volume_integrals_l1514_151483

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x^3)

theorem volume_integrals (π : ℝ) (h₁ : π > 0) :
  (∫ (x : ℝ) in Set.Ioi 0, π * (f x)^2) = π / 6 ∧
  (∫ (x : ℝ) in Set.Icc 0 (Real.rpow 3 (1/3)), π * x^2 * (1 - 3*x^3) * Real.exp (-x^3)) = π * (Real.exp (-1/3) - 2/3) :=
sorry

end volume_integrals_l1514_151483


namespace sum_even_coefficients_l1514_151480

theorem sum_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 * (x + 2)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + 
                                    a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₂ + a₄ + a₆ + a₈ = -24 := by
sorry

end sum_even_coefficients_l1514_151480


namespace range_of_a_l1514_151410

open Set

/-- The statement p: √(2x-1) ≤ 1 -/
def p (x : ℝ) : Prop := Real.sqrt (2 * x - 1) ≤ 1

/-- The statement q: (x-a)(x-(a+1)) ≤ 0 -/
def q (x a : ℝ) : Prop := (x - a) * (x - (a + 1)) ≤ 0

/-- The set of x satisfying statement p -/
def P : Set ℝ := {x | p x}

/-- The set of x satisfying statement q -/
def Q (a : ℝ) : Set ℝ := {x | q x a}

/-- p is a sufficient but not necessary condition for q -/
def sufficient_not_necessary (a : ℝ) : Prop := P ⊂ Q a ∧ P ≠ Q a

theorem range_of_a : 
  ∀ a : ℝ, sufficient_not_necessary a ↔ a ∈ Icc 0 (1/2) :=
sorry

end range_of_a_l1514_151410


namespace melissa_bananas_l1514_151417

/-- Calculates the remaining bananas after sharing -/
def remaining_bananas (initial : ℕ) (shared : ℕ) : ℕ :=
  initial - shared

theorem melissa_bananas : remaining_bananas 88 4 = 84 := by
  sorry

end melissa_bananas_l1514_151417


namespace power_of_36_l1514_151422

theorem power_of_36 : (36 : ℝ) ^ (5/2 : ℝ) = 7776 := by
  sorry

end power_of_36_l1514_151422


namespace tangent_line_y_intercept_l1514_151447

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 11

-- Define the point of tangency
def P : ℝ × ℝ := (1, 12)

-- Theorem statement
theorem tangent_line_y_intercept :
  let slope := (3 : ℝ) -- Derivative of f at x = 1
  let tangent_line (x : ℝ) := slope * (x - P.1) + P.2
  (tangent_line 0) = 9 := by sorry

end tangent_line_y_intercept_l1514_151447


namespace geometric_sequence_fourth_term_l1514_151477

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_product : a 3 * a 5 = 64) :
  a 4 = 8 ∨ a 4 = -8 :=
sorry

end geometric_sequence_fourth_term_l1514_151477


namespace dance_team_recruitment_l1514_151453

theorem dance_team_recruitment (total : ℕ) (track : ℕ) (choir : ℕ) (dance : ℕ) : 
  total = 100 ∧ 
  choir = 2 * track ∧ 
  dance = choir + 10 ∧ 
  total = track + choir + dance → 
  dance = 46 := by
sorry

end dance_team_recruitment_l1514_151453


namespace infinite_product_equals_nine_l1514_151425

def infinite_product : ℕ → ℝ
  | 0 => 3^(1/2)
  | n + 1 => infinite_product n * (3^(n+1))^(1 / 2^(n+1))

theorem infinite_product_equals_nine :
  ∃ (limit : ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |infinite_product n - limit| < ε) ∧ limit = 9 := by
  sorry

end infinite_product_equals_nine_l1514_151425


namespace folding_coincidence_implies_rhombus_l1514_151420

/-- A quadrilateral on a piece of paper. -/
structure PaperQuadrilateral where
  /-- The four vertices of the quadrilateral -/
  vertices : Fin 4 → ℝ × ℝ

/-- Represents the result of folding a paper quadrilateral along a diagonal -/
def foldAlongDiagonal (q : PaperQuadrilateral) (d : Fin 2) : Prop :=
  -- This is a placeholder for the actual folding operation
  sorry

/-- A quadrilateral is a rhombus if it satisfies certain properties -/
def isRhombus (q : PaperQuadrilateral) : Prop :=
  -- This is a placeholder for the actual definition of a rhombus
  sorry

/-- 
If folding a quadrilateral along both diagonals results in coinciding parts each time, 
then the quadrilateral is a rhombus.
-/
theorem folding_coincidence_implies_rhombus (q : PaperQuadrilateral) :
  (∀ d : Fin 2, foldAlongDiagonal q d) → isRhombus q :=
by
  sorry

end folding_coincidence_implies_rhombus_l1514_151420


namespace square_triangle_equal_area_l1514_151452

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ) :
  square_perimeter = 48 →
  triangle_height = 48 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_base * triangle_height →
  triangle_base = 6 := by
  sorry

end square_triangle_equal_area_l1514_151452


namespace matrix_equation_proof_l1514_151497

theorem matrix_equation_proof :
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![16/7, -36/7; -12/7, 27/7]
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, 5; 16, -4]
  let C : Matrix (Fin 2) (Fin 2) ℚ := !![4, -1; -4, 1]
  N * A = B + C := by sorry

end matrix_equation_proof_l1514_151497


namespace quadratic_inequality_l1514_151432

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem quadratic_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 1 5, f a x > 3*a*x) ↔ a < 2*Real.sqrt 2 ∧
  (∀ x : ℝ, (a + 1)*x^2 + x > f a x ↔
    (a = 0 ∧ x > 2) ∨
    (a > 0 ∧ (x < -1/a ∨ x > 2)) ∨
    (-1/2 < a ∧ a < 0 ∧ 2 < x ∧ x < -1/a) ∨
    (a < -1/2 ∧ -1/a < x ∧ x < 2)) :=
sorry

end quadratic_inequality_l1514_151432


namespace x_less_neg_one_sufficient_not_necessary_for_abs_x_greater_x_l1514_151416

theorem x_less_neg_one_sufficient_not_necessary_for_abs_x_greater_x :
  (∃ x : ℝ, x < -1 → abs x > x) ∧ 
  (∃ x : ℝ, abs x > x ∧ x ≥ -1) :=
by sorry

end x_less_neg_one_sufficient_not_necessary_for_abs_x_greater_x_l1514_151416


namespace sin_2alpha_value_l1514_151418

theorem sin_2alpha_value (α : Real) 
  (h1 : α > -π/2 ∧ α < 0) 
  (h2 : Real.tan (π/4 - α) = 3 * Real.cos (2 * α)) : 
  Real.sin (2 * α) = -2/3 := by
  sorry

end sin_2alpha_value_l1514_151418


namespace sam_seashells_l1514_151494

def seashells_problem (yesterday_found : ℕ) (given_to_joan : ℕ) (today_found : ℕ) (given_to_tom : ℕ) : ℕ :=
  yesterday_found - given_to_joan + today_found - given_to_tom

theorem sam_seashells : seashells_problem 35 18 20 5 = 32 := by
  sorry

end sam_seashells_l1514_151494


namespace min_value_reciprocal_sum_l1514_151451

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b - 1 = 0) :
  (2/a + 3/b) ≥ 25 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2*a₀ + 3*b₀ - 1 = 0 ∧ 2/a₀ + 3/b₀ = 25 :=
sorry

end min_value_reciprocal_sum_l1514_151451


namespace right_triangles_common_hypotenuse_l1514_151423

theorem right_triangles_common_hypotenuse (AC AD CD : ℝ) (hAC : AC = 16) (hAD : AD = 32) (hCD : CD = 14) :
  let AB := Real.sqrt (AD^2 - (AC + CD)^2)
  let BC := Real.sqrt (AB^2 + AC^2)
  BC = Real.sqrt 380 := by
sorry

end right_triangles_common_hypotenuse_l1514_151423


namespace triangle_problem_l1514_151405

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  -- Law of cosines
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  -- Given conditions
  a = 2 * Real.sqrt 6 →
  b = 3 →
  Real.sin (B + C)^2 + Real.sqrt 2 * Real.sin (2 * A) = 0 →
  -- Conclusion
  c = 3 ∧ Real.cos B = Real.sqrt 6 / 3 := by
sorry

end triangle_problem_l1514_151405


namespace admission_score_calculation_l1514_151499

theorem admission_score_calculation (total_applicants : ℕ) 
  (admitted_ratio : ℚ) 
  (admitted_avg_diff : ℝ) 
  (not_admitted_avg_diff : ℝ) 
  (total_avg_score : ℝ) 
  (h1 : admitted_ratio = 1 / 4)
  (h2 : admitted_avg_diff = 10)
  (h3 : not_admitted_avg_diff = -26)
  (h4 : total_avg_score = 70) :
  ∃ (admission_score : ℝ),
    admission_score = 87 ∧
    (admitted_ratio * (admission_score + admitted_avg_diff) + 
     (1 - admitted_ratio) * (admission_score + not_admitted_avg_diff) = total_avg_score) := by
  sorry

end admission_score_calculation_l1514_151499


namespace fraction_problem_l1514_151430

theorem fraction_problem (n d : ℚ) : 
  n / (d + 1) = 1 / 2 → (n + 1) / d = 1 → n / d = 2 / 3 := by
  sorry

end fraction_problem_l1514_151430


namespace smaller_number_proof_l1514_151433

theorem smaller_number_proof (x y : ℝ) 
  (sum_eq : x + y = 18)
  (diff_eq : x - y = 4)
  (prod_eq : x * y = 77) :
  y = 7 := by
  sorry

end smaller_number_proof_l1514_151433


namespace periodic_function_roots_l1514_151469

theorem periodic_function_roots (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (2 + x) = f (2 - x))
  (h2 : ∀ x : ℝ, f (7 + x) = f (7 - x))
  (h3 : f 0 = 0) :
  ∃ (roots : Finset ℝ), (∀ x ∈ roots, f x = 0 ∧ x ∈ Set.Icc (-1000) 1000) ∧ roots.card ≥ 201 :=
sorry

end periodic_function_roots_l1514_151469


namespace max_value_problem_l1514_151460

theorem max_value_problem (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 3) 
  (h5 : x ≥ y) (h6 : y ≥ z) : 
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 2916/729 :=
sorry

end max_value_problem_l1514_151460


namespace min_values_xy_and_x_plus_2y_l1514_151408

theorem min_values_xy_and_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / x + 9 / y = 1) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / a + 9 / b = 1 → x * y ≤ a * b) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / a + 9 / b = 1 → x + 2 * y ≤ a + 2 * b) ∧
  x * y = 36 ∧ 
  x + 2 * y = 20 + 6 * Real.sqrt 2 := by
sorry

end min_values_xy_and_x_plus_2y_l1514_151408


namespace sarah_finished_problems_l1514_151419

/-- Calculates the number of problems Sarah finished given the initial number of problems,
    remaining pages, and problems per page. -/
def problems_finished (initial_problems : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  initial_problems - (remaining_pages * problems_per_page)

/-- Proves that Sarah finished 20 problems given the initial conditions. -/
theorem sarah_finished_problems :
  problems_finished 60 5 8 = 20 := by
  sorry

end sarah_finished_problems_l1514_151419


namespace tangent_line_at_two_range_of_m_for_three_roots_l1514_151481

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Theorem for the tangent line equation
theorem tangent_line_at_two :
  ∃ (A B C : ℝ), A ≠ 0 ∧ 
  (∀ x y : ℝ, y = f x → (x = 2 → A * x + B * y + C = 0)) ∧
  A = 12 ∧ B = -1 ∧ C = -17 := by sorry

-- Theorem for the range of m
theorem range_of_m_for_three_roots :
  ∀ m : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f x + m = 0 ∧ f y + m = 0 ∧ f z + m = 0) ↔ 
  -3 < m ∧ m < -2 := by sorry

end tangent_line_at_two_range_of_m_for_three_roots_l1514_151481


namespace popped_kernels_problem_l1514_151478

theorem popped_kernels_problem (bag1_popped bag1_total bag2_popped bag2_total bag3_popped : ℕ)
  (h1 : bag1_popped = 60)
  (h2 : bag1_total = 75)
  (h3 : bag2_popped = 42)
  (h4 : bag2_total = 50)
  (h5 : bag3_popped = 82)
  (h6 : (bag1_popped : ℚ) / bag1_total + (bag2_popped : ℚ) / bag2_total + (bag3_popped : ℚ) / bag3_total = 82 * 3 / 100) :
  bag3_total = 100 := by
  sorry

end popped_kernels_problem_l1514_151478


namespace intersection_segment_length_l1514_151442

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (3, 0)

-- Define the line perpendicular to x-axis passing through the right focus
def perpendicular_line (x y : ℝ) : Prop := x = 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ perpendicular_line p.1 p.2}

-- Statement to prove
theorem intersection_segment_length :
  let A := (3, 16/5)
  let B := (3, -16/5)
  (A ∈ intersection_points) ∧ 
  (B ∈ intersection_points) ∧
  (∀ p ∈ intersection_points, p = A ∨ p = B) ∧
  (dist A B = 32/5) := by sorry


end intersection_segment_length_l1514_151442


namespace p_sufficient_but_not_necessary_for_q_l1514_151441

-- Define the conditions
def p (x : ℝ) : Prop := |x - 1| < 2
def q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

-- Define what it means for p to be sufficient but not necessary for q
def sufficient_but_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x)

-- State the theorem
theorem p_sufficient_but_not_necessary_for_q :
  sufficient_but_not_necessary p q := by
  sorry

end p_sufficient_but_not_necessary_for_q_l1514_151441


namespace decimal_to_fraction_l1514_151470

theorem decimal_to_fraction : 
  (2.36 : ℚ) = 59 / 25 := by sorry

end decimal_to_fraction_l1514_151470


namespace triangle_cosine_relation_l1514_151404

/-- Given a triangle ABC with angles A, B, C and sides a, b, c opposite to these angles respectively.
    If 2sin²A + 2sin²B = 2sin²(A+B) + 3sinAsinB, then cos C = 3/4. -/
theorem triangle_cosine_relation (A B C a b c : Real) : 
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  2 * (Real.sin A)^2 + 2 * (Real.sin B)^2 = 2 * (Real.sin (A + B))^2 + 3 * Real.sin A * Real.sin B →
  Real.cos C = 3/4 := by
  sorry

end triangle_cosine_relation_l1514_151404


namespace scientific_notation_of_71300000_l1514_151491

theorem scientific_notation_of_71300000 :
  ∃ (a : ℝ) (n : ℤ), 71300000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 7.13 ∧ n = 7 :=
sorry

end scientific_notation_of_71300000_l1514_151491


namespace perimeter_of_figure_c_l1514_151436

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.height)

/-- Represents the large rectangle composed of small rectangles -/
structure LargeRectangle where
  small_rectangle : Rectangle
  total_count : ℕ

/-- Theorem: Given the conditions, the perimeter of figure C is 40 cm -/
theorem perimeter_of_figure_c (large_rect : LargeRectangle)
    (h1 : large_rect.total_count = 20)
    (h2 : Rectangle.perimeter { width := 6 * large_rect.small_rectangle.width,
                                height := large_rect.small_rectangle.height } = 56)
    (h3 : Rectangle.perimeter { width := 2 * large_rect.small_rectangle.width,
                                height := 3 * large_rect.small_rectangle.height } = 56) :
  Rectangle.perimeter { width := large_rect.small_rectangle.width,
                        height := 3 * large_rect.small_rectangle.height } = 40 := by
  sorry

end perimeter_of_figure_c_l1514_151436


namespace w_squared_value_l1514_151495

theorem w_squared_value (w : ℝ) (h : (w + 15)^2 = (4*w + 9)*(w + 6)) :
  w^2 = 57.5 - 0.5 * Real.sqrt 229 := by
  sorry

end w_squared_value_l1514_151495


namespace absolute_value_equation_solution_l1514_151473

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 9| = |x + 3| :=
by
  -- The proof goes here
  sorry

end absolute_value_equation_solution_l1514_151473


namespace min_squared_distance_to_origin_l1514_151492

/-- The minimum squared distance from a point on the line 2x + y + 5 = 0 to the origin is 5 -/
theorem min_squared_distance_to_origin : 
  ∀ x y : ℝ, 2 * x + y + 5 = 0 → x^2 + y^2 ≥ 5 := by
  sorry

end min_squared_distance_to_origin_l1514_151492


namespace inequality_equivalence_l1514_151445

theorem inequality_equivalence (x : ℝ) : 
  x * Real.log (x^2 + x + 1) / Real.log (1/10) > 0 ↔ x < -1 := by sorry

end inequality_equivalence_l1514_151445


namespace parallelogram_points_l1514_151435

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points form a parallelogram -/
def is_parallelogram (a b c d : Point) : Prop :=
  (b.x - a.x = d.x - c.x ∧ b.y - a.y = d.y - c.y) ∨
  (c.x - a.x = d.x - b.x ∧ c.y - a.y = d.y - b.y) ∨
  (b.x - a.x = c.x - d.x ∧ b.y - a.y = c.y - d.y)

/-- The main theorem -/
theorem parallelogram_points :
  let a : Point := ⟨3, 7⟩
  let b : Point := ⟨4, 6⟩
  let c : Point := ⟨1, -2⟩
  ∀ d : Point, is_parallelogram a b c d ↔ d = ⟨0, -1⟩ ∨ d = ⟨2, -3⟩ ∨ d = ⟨6, 15⟩ :=
by sorry

end parallelogram_points_l1514_151435


namespace jerry_initial_figures_l1514_151421

/-- The number of books on Jerry's shelf -/
def num_books : ℕ := 9

/-- The number of action figures added later -/
def added_figures : ℕ := 7

/-- The difference between action figures and books after adding -/
def difference : ℕ := 3

/-- The initial number of action figures on Jerry's shelf -/
def initial_figures : ℕ := 5

theorem jerry_initial_figures :
  initial_figures + added_figures = num_books + difference := by sorry

end jerry_initial_figures_l1514_151421


namespace largest_possible_a_l1514_151463

theorem largest_possible_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : d < 150) :
  a ≤ 8924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 8924 ∧
    a' < 3 * b' ∧
    b' < 4 * c' ∧
    c' < 5 * d' ∧
    d' < 150 :=
by sorry

end largest_possible_a_l1514_151463


namespace difference_of_squares_123_23_l1514_151412

theorem difference_of_squares_123_23 : 123^2 - 23^2 = 14600 := by
  sorry

end difference_of_squares_123_23_l1514_151412


namespace james_fish_catch_l1514_151465

/-- The total weight of fish James caught -/
def total_fish_weight (trout salmon tuna bass catfish : ℝ) : ℝ :=
  trout + salmon + tuna + bass + catfish

/-- Theorem stating the total weight of fish James caught -/
theorem james_fish_catch :
  ∃ (trout salmon tuna bass catfish : ℝ),
    trout = 200 ∧
    salmon = trout * 1.6 ∧
    tuna = trout * 2 ∧
    bass = salmon * 3 ∧
    catfish = tuna / 3 ∧
    total_fish_weight trout salmon tuna bass catfish = 2013.33 :=
by
  sorry

end james_fish_catch_l1514_151465


namespace equation_solution_l1514_151413

theorem equation_solution : 
  ∃ x : ℝ, (6 * x^2 + 111 * x + 1) / (2 * x + 37) = 3 * x + 1 ∧ x = -18 := by
  sorry

end equation_solution_l1514_151413


namespace greatest_integer_mike_l1514_151461

theorem greatest_integer_mike (n : ℕ) : 
  (∃ k l : ℤ, n = 9 * k - 1 ∧ n = 10 * l - 4) →
  n < 150 →
  (∀ m : ℕ, (∃ k l : ℤ, m = 9 * k - 1 ∧ m = 10 * l - 4) → m < 150 → m ≤ n) →
  n = 86 := by
sorry

end greatest_integer_mike_l1514_151461


namespace prime_power_divisors_l1514_151415

theorem prime_power_divisors (p q : ℕ) (x : ℕ) (hp : Prime p) (hq : Prime q) :
  (∀ d : ℕ, d ∣ p^4 * q^x ↔ d ∈ Finset.range 51) → x = 9 := by
  sorry

end prime_power_divisors_l1514_151415


namespace arithmetic_calculation_l1514_151493

theorem arithmetic_calculation : 72 / (6 / 2) * 3 = 72 := by
  sorry

end arithmetic_calculation_l1514_151493


namespace x_squared_value_l1514_151406

theorem x_squared_value (x : ℝ) (hx : x > 0) (h : Real.sin (Real.arctan x) = 1 / x) : 
  x^2 = (1 + Real.sqrt 5) / 2 := by
sorry

end x_squared_value_l1514_151406


namespace minimize_quadratic_expression_l1514_151437

theorem minimize_quadratic_expression (b : ℝ) :
  let f : ℝ → ℝ := λ x => (1/3) * x^2 + 7*x - 6
  ∀ x, f b ≤ f x ↔ b = -21/2 :=
by sorry

end minimize_quadratic_expression_l1514_151437


namespace measure_8_and_5_cm_l1514_151400

-- Define the marks on the ruler
def ruler_marks : List ℕ := [0, 7, 11]

-- Define a function to check if a length can be measured
def can_measure (length : ℕ) : Prop :=
  ∃ (a b c : ℤ), a * ruler_marks[1] + b * ruler_marks[2] + c * (ruler_marks[2] - ruler_marks[1]) = length

-- Theorem statement
theorem measure_8_and_5_cm :
  can_measure 8 ∧ can_measure 5 :=
by sorry

end measure_8_and_5_cm_l1514_151400


namespace S_is_closed_closed_set_contains_zero_l1514_151488

-- Define a closed set
def ClosedSet (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S ∧ (x - y) ∈ S ∧ (x * y) ∈ S

-- Define the set S
def S : Set ℝ := {x | ∃ a b : ℤ, x = a + b * Real.sqrt 3}

-- Theorem 1: S is a closed set
theorem S_is_closed : ClosedSet S := sorry

-- Theorem 2: Any closed set contains 0
theorem closed_set_contains_zero (T : Set ℝ) (h : ClosedSet T) : (0 : ℝ) ∈ T := sorry

end S_is_closed_closed_set_contains_zero_l1514_151488


namespace shirt_and_sweater_cost_l1514_151450

theorem shirt_and_sweater_cost (shirt_price sweater_price total_cost : ℝ) : 
  shirt_price = 36.46 →
  sweater_price = shirt_price + 7.43 →
  total_cost = shirt_price + sweater_price →
  total_cost = 80.35 := by
sorry

end shirt_and_sweater_cost_l1514_151450


namespace stating_sum_of_intersections_theorem_l1514_151487

/-- The number of lines passing through the origin -/
def num_lines : ℕ := 180

/-- The angle between each line in degrees -/
def angle_between : ℝ := 1

/-- The equation of the line that intersects with all other lines -/
def intersecting_line (x : ℝ) : ℝ := 100 - x

/-- The sum of x-coordinates of intersection points -/
def sum_of_intersections : ℝ := 8950

/-- 
Theorem stating that the sum of x-coordinates of intersections between 
180 lines passing through the origin (forming 1 degree angles) and the 
line y = 100 - x is equal to 8950.
-/
theorem sum_of_intersections_theorem :
  let lines := List.range num_lines
  let intersection_points := lines.map (λ i => 
    let angle := i * angle_between
    let m := Real.tan (angle * π / 180)
    100 / (1 + m))
  intersection_points.sum = sum_of_intersections := by
  sorry


end stating_sum_of_intersections_theorem_l1514_151487


namespace fraction_equality_l1514_151402

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x - 2 * y) / (3 * x + y) = 3) : 
  (5 * x - y) / (2 * x + 4 * y) = -3 := by
  sorry

end fraction_equality_l1514_151402


namespace quadratic_factorization_l1514_151462

theorem quadratic_factorization (c d : ℕ) (hc : c > d) :
  (∀ x, x^2 - 20*x + 96 = (x - c) * (x - d)) →
  4*d - c = 20 := by
  sorry

end quadratic_factorization_l1514_151462


namespace stating_rowing_speed_calculation_l1514_151428

/-- Represents the speed of the river current in km/h -/
def stream_speed : ℝ := 12

/-- Represents the man's rowing speed in still water in km/h -/
def rowing_speed : ℝ := 24

/-- 
Theorem stating that if it takes thrice as long to row up as to row down the river,
given the stream speed, then the rowing speed in still water is 24 km/h
-/
theorem rowing_speed_calculation (distance : ℝ) (h : distance > 0) :
  (distance / (rowing_speed - stream_speed)) = 3 * (distance / (rowing_speed + stream_speed)) →
  rowing_speed = 24 := by
sorry

end stating_rowing_speed_calculation_l1514_151428


namespace stratified_sampling_pine_saplings_l1514_151449

theorem stratified_sampling_pine_saplings 
  (total_saplings : ℕ) 
  (pine_saplings : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_saplings = 30000) 
  (h2 : pine_saplings = 4000) 
  (h3 : sample_size = 150) :
  (pine_saplings : ℚ) / total_saplings * sample_size = 20 := by
sorry


end stratified_sampling_pine_saplings_l1514_151449


namespace final_value_calculation_l1514_151456

theorem final_value_calculation : 
  let initial_number := 16
  let doubled := initial_number * 2
  let added_five := doubled + 5
  let final_value := added_five * 3
  final_value = 111 := by
sorry

end final_value_calculation_l1514_151456


namespace cube_sum_of_symmetric_polynomials_l1514_151457

theorem cube_sum_of_symmetric_polynomials (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = -3) 
  (h3 : a * b * c = 9) : 
  a^3 + b^3 + c^3 = 22 := by sorry

end cube_sum_of_symmetric_polynomials_l1514_151457


namespace bathing_suit_combinations_total_combinations_l1514_151448

theorem bathing_suit_combinations : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | men_styles, men_sizes, men_colors, women_styles, women_sizes, women_colors =>
    (men_styles * men_sizes * men_colors) + (women_styles * women_sizes * women_colors)

theorem total_combinations (men_styles men_sizes men_colors women_styles women_sizes women_colors : ℕ) :
  men_styles = 5 →
  men_sizes = 3 →
  men_colors = 4 →
  women_styles = 4 →
  women_sizes = 4 →
  women_colors = 5 →
  bathing_suit_combinations men_styles men_sizes men_colors women_styles women_sizes women_colors = 140 :=
by
  sorry

end bathing_suit_combinations_total_combinations_l1514_151448


namespace multiply_to_all_ones_l1514_151434

theorem multiply_to_all_ones : 
  ∃ (A : ℕ) (n : ℕ), (10^9 - 1) * A = (10^n - 1) / 9 :=
sorry

end multiply_to_all_ones_l1514_151434


namespace cube_sum_reciprocal_l1514_151471

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^3 = 2) :
  a^4 + 1/a^4 = (Real.rpow 4 (1/3) - 2)^2 - 2 := by
  sorry

end cube_sum_reciprocal_l1514_151471


namespace min_value_of_M_l1514_151490

theorem min_value_of_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  let M := (1/a - 1) * (1/b - 1) * (1/c - 1)
  M ≥ 8 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 1 ∧
    (1/a₀ - 1) * (1/b₀ - 1) * (1/c₀ - 1) = 8 :=
by sorry

end min_value_of_M_l1514_151490


namespace quadratic_properties_l1514_151438

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 2)^2 + 6

-- State the theorem
theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f x > f y → f y > f ((y - x) + y)) ∧ 
  (∀ x : ℝ, f (2 + x) = f (2 - x)) ∧
  (f 0 = 10) :=
sorry

end quadratic_properties_l1514_151438


namespace absolute_value_inequality_solution_l1514_151468

theorem absolute_value_inequality_solution (x : ℝ) :
  (|x + 2| + |x - 2| < x + 7) ↔ (-7/3 < x ∧ x < 7) :=
sorry

end absolute_value_inequality_solution_l1514_151468


namespace min_white_pairs_8x8_20black_l1514_151474

/-- Represents a grid with black and white cells -/
structure Grid :=
  (size : Nat)
  (black_cells : Nat)

/-- Calculates the total number of adjacent cell pairs in the grid -/
def total_pairs (g : Grid) : Nat :=
  2 * g.size * (g.size - 1)

/-- Represents the minimum number of white cell pairs in the grid -/
def min_white_pairs (g : Grid) : Nat :=
  total_pairs g - (g.black_cells + 40)

/-- Theorem stating the minimum number of white cell pairs in an 8x8 grid with 20 black cells -/
theorem min_white_pairs_8x8_20black :
  ∀ (g : Grid), g.size = 8 → g.black_cells = 20 → min_white_pairs g = 34 :=
by sorry

end min_white_pairs_8x8_20black_l1514_151474


namespace success_arrangements_eq_420_l1514_151482

/-- The number of letters in the word "SUCCESS" -/
def word_length : ℕ := 7

/-- The number of occurrences of the letter 'S' in "SUCCESS" -/
def s_count : ℕ := 3

/-- The number of occurrences of the letter 'C' in "SUCCESS" -/
def c_count : ℕ := 2

/-- The number of unique arrangements of the letters in "SUCCESS" -/
def success_arrangements : ℕ := word_length.factorial / (s_count.factorial * c_count.factorial)

theorem success_arrangements_eq_420 : success_arrangements = 420 := by
  sorry

end success_arrangements_eq_420_l1514_151482


namespace greatest_multiple_of_5_and_6_under_1000_l1514_151464

theorem greatest_multiple_of_5_and_6_under_1000 : 
  ∀ n : ℕ, n < 1000 → n % 5 = 0 → n % 6 = 0 → n ≤ 990 :=
by
  sorry

end greatest_multiple_of_5_and_6_under_1000_l1514_151464


namespace exists_a_with_full_domain_and_range_l1514_151426

/-- Given a real number a, f is a function from ℝ to ℝ defined as f(x) = ax^2 + x + 1 -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + x + 1

/-- Theorem stating that there exists a real number a such that f(a) has domain and range ℝ -/
theorem exists_a_with_full_domain_and_range :
  ∃ a : ℝ, Function.Surjective (f a) ∧ Function.Injective (f a) := by
  sorry

end exists_a_with_full_domain_and_range_l1514_151426


namespace fraction_problem_l1514_151466

theorem fraction_problem :
  ∃ (x y : ℚ), x / y = 2 / 3 ∧ (x / y) * 6 + 6 = 10 :=
by
  sorry

end fraction_problem_l1514_151466


namespace fraction_value_implies_x_l1514_151458

theorem fraction_value_implies_x (x : ℝ) : 2 / (x - 3) = 2 → x = 4 := by
  sorry

end fraction_value_implies_x_l1514_151458


namespace prime_sum_seven_power_l1514_151496

theorem prime_sum_seven_power (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 7 → (p^q = 32 ∨ p^q = 25) := by
  sorry

end prime_sum_seven_power_l1514_151496


namespace f_is_increasing_l1514_151403

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*x

-- Theorem statement
theorem f_is_increasing : ∀ x : ℝ, Monotone f := by sorry

end f_is_increasing_l1514_151403


namespace product_of_roots_l1514_151443

/-- The polynomial coefficients -/
def a : ℝ := 2
def b : ℝ := -5
def c : ℝ := -10
def d : ℝ := 22

/-- The polynomial equation -/
def polynomial (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem product_of_roots :
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, polynomial x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ * r₂ * r₃ = -11 := by
  sorry

end product_of_roots_l1514_151443


namespace christine_savings_l1514_151444

/-- Calculates the amount saved by a salesperson given their commission rate, total sales, and personal needs allocation percentage. -/
def amount_saved (commission_rate : ℚ) (total_sales : ℚ) (personal_needs_percent : ℚ) : ℚ :=
  let total_commission := commission_rate * total_sales
  let personal_needs := personal_needs_percent * total_commission
  total_commission - personal_needs

/-- Proves that given the specific conditions, the amount saved is $1152. -/
theorem christine_savings : 
  amount_saved (12/100) 24000 (60/100) = 1152 := by
  sorry

end christine_savings_l1514_151444


namespace quadratic_root_range_l1514_151409

theorem quadratic_root_range (m : ℝ) (α β : ℝ) : 
  (∃ x, x^2 - 2*(m-1)*x + (m-1) = 0) ∧ 
  (α^2 - 2*(m-1)*α + (m-1) = 0) ∧ 
  (β^2 - 2*(m-1)*β + (m-1) = 0) ∧ 
  (0 < α) ∧ (α < 1) ∧ (1 < β) ∧ (β < 2) →
  (2 < m) ∧ (m < 7/3) := by
sorry

end quadratic_root_range_l1514_151409


namespace sequence_theorem_l1514_151429

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, Real.sqrt (a (n + 1)) - Real.sqrt (a n) = d * n + (Real.sqrt (a 1) - Real.sqrt (a 0))

theorem sequence_theorem (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, Real.sqrt (a (n + 1)) - Real.sqrt (a n) = 2 * n - 2) →
  a 1 = 1 →
  a 3 = 9 →
  ∀ n : ℕ, a n = (n^2 - 3*n + 3)^2 :=
by sorry

end sequence_theorem_l1514_151429


namespace inscribed_circle_triangle_sides_l1514_151401

/-- A triangle with an inscribed circle of radius 3, where one side is divided into segments of 4 and 3 by the point of tangency. -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  s₁ : ℝ
  /-- The length of the second segment of the divided side -/
  s₂ : ℝ
  /-- Condition that the radius is 3 -/
  h_r : r = 3
  /-- Condition that the first segment is 4 -/
  h_s₁ : s₁ = 4
  /-- Condition that the second segment is 3 -/
  h_s₂ : s₂ = 3

/-- The lengths of the sides of the triangle -/
def sideLengths (t : InscribedCircleTriangle) : Fin 3 → ℝ
| 0 => 24
| 1 => 25
| 2 => 7

theorem inscribed_circle_triangle_sides (t : InscribedCircleTriangle) :
  ∀ i, sideLengths t i = if i = 0 then 24 else if i = 1 then 25 else 7 := by
  sorry

end inscribed_circle_triangle_sides_l1514_151401


namespace solution_set_when_a_is_2_range_of_a_for_inequality_l1514_151439

-- Define the function f
def f (x a : ℝ) := |3 * x + 3| + |x - a|

-- Theorem 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 > 4} = {x : ℝ | x > -1/2 ∨ x < -5/4} := by sorry

-- Theorem 2
theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x : ℝ, x > -1 → f x a > 3*x + 4) ↔ a ≤ -2 := by sorry

end solution_set_when_a_is_2_range_of_a_for_inequality_l1514_151439


namespace inequality_proof_l1514_151446

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (a * b / Real.sqrt (c^2 + 3)) + (b * c / Real.sqrt (a^2 + 3)) + (c * a / Real.sqrt (b^2 + 3)) ≤ 3/2 := by
  sorry

end inequality_proof_l1514_151446


namespace power_function_sum_l1514_151467

/-- A function f is a power function if it has the form f(x) = cx^n + k, where c ≠ 0 and n is a real number -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (c k n : ℝ), c ≠ 0 ∧ ∀ x, f x = c * x^n + k

/-- Given that f(x) = ax^(2a+1) - b + 1 is a power function, prove that a + b = 2 -/
theorem power_function_sum (a b : ℝ) 
    (h : isPowerFunction (fun x => a * x^(2*a+1) - b + 1)) : 
  a + b = 2 := by
  sorry

end power_function_sum_l1514_151467


namespace pizza_order_l1514_151414

theorem pizza_order (slices_per_pizza : ℕ) (total_slices : ℕ) (num_people : ℕ)
  (h1 : slices_per_pizza = 4)
  (h2 : total_slices = 68)
  (h3 : num_people = 25) :
  total_slices / slices_per_pizza = 17 := by
sorry

end pizza_order_l1514_151414


namespace system_solution_l1514_151498

theorem system_solution : ∃ (x y : ℝ), 
  x = 2 ∧ y = -1 ∧ 2*x + y = 3 ∧ -x - y = -1 := by sorry

end system_solution_l1514_151498


namespace balloon_problem_l1514_151459

theorem balloon_problem (total_people : ℕ) (total_balloons : ℕ) 
  (x₁ x₂ x₃ x₄ : ℕ) 
  (h1 : total_people = 101)
  (h2 : total_balloons = 212)
  (h3 : x₁ + x₂ + x₃ + x₄ = total_people)
  (h4 : x₁ + 2*x₂ + 3*x₃ + 4*x₄ = total_balloons)
  (h5 : x₄ = x₂ + 13) :
  x₁ = 52 := by
  sorry

end balloon_problem_l1514_151459


namespace juliet_supporter_in_capulet_probability_l1514_151440

-- Define the population distribution
def montague_pop : ℚ := 5/8
def capulet_pop : ℚ := 3/16
def verona_pop : ℚ := 1/8
def mercutio_pop : ℚ := 1 - (montague_pop + capulet_pop + verona_pop)

-- Define the support rates
def montague_romeo_rate : ℚ := 4/5
def capulet_juliet_rate : ℚ := 7/10
def verona_romeo_rate : ℚ := 13/20
def mercutio_juliet_rate : ℚ := 11/20

-- Define the total Juliet supporters
def total_juliet_supporters : ℚ := capulet_pop * capulet_juliet_rate + mercutio_pop * mercutio_juliet_rate

-- Define the probability
def prob_juliet_in_capulet : ℚ := (capulet_pop * capulet_juliet_rate) / total_juliet_supporters

-- Theorem statement
theorem juliet_supporter_in_capulet_probability :
  ∃ (ε : ℚ), abs (prob_juliet_in_capulet - 66/100) < ε ∧ ε < 1/100 :=
sorry

end juliet_supporter_in_capulet_probability_l1514_151440


namespace odd_numbers_properties_l1514_151484

theorem odd_numbers_properties (x y : ℤ) (hx : ∃ k : ℤ, x = 2 * k + 1) (hy : ∃ k : ℤ, y = 2 * k + 1) :
  (∃ m : ℤ, x + y = 2 * m) ∧ 
  (∃ n : ℤ, x - y = 2 * n) ∧ 
  (∃ p : ℤ, x * y = 2 * p + 1) := by
  sorry

end odd_numbers_properties_l1514_151484


namespace teaching_team_formation_l1514_151424

def chinese_teachers : ℕ := 2
def math_teachers : ℕ := 2
def english_teachers : ℕ := 4
def team_size : ℕ := 5

def ways_to_form_team : ℕ := 
  Nat.choose english_teachers 1 + 
  (Nat.choose chinese_teachers 1 * Nat.choose english_teachers 2) +
  (Nat.choose math_teachers 1 * Nat.choose english_teachers 2) +
  (Nat.choose chinese_teachers 1 * Nat.choose math_teachers 1 * Nat.choose english_teachers 3)

theorem teaching_team_formation :
  ways_to_form_team = 44 :=
by sorry

end teaching_team_formation_l1514_151424


namespace veena_bill_fraction_l1514_151485

theorem veena_bill_fraction :
  ∀ (L V A : ℚ),
  V = (1/2) * L →
  A = (3/4) * V →
  V / (L + V + A) = 4/15 := by
sorry

end veena_bill_fraction_l1514_151485


namespace invariant_preserved_cannot_transform_l1514_151486

/-- Represents a letter in the English alphabet -/
def Letter := Fin 26

/-- A 4x4 matrix of letters -/
def LetterMatrix := Matrix (Fin 4) (Fin 4) Letter

/-- The operation of incrementing a letter (with wrapping) -/
def nextLetter (l : Letter) : Letter :=
  ⟨(l.val + 1) % 26, by sorry⟩

/-- The invariant property for a 2x2 submatrix -/
def invariant (a b c d : Letter) : ℤ :=
  (a.val + d.val : ℤ) - (b.val + c.val : ℤ)

/-- Theorem: The invariant is preserved under row and column operations -/
theorem invariant_preserved (a b c d : Letter) :
  (invariant a b c d = invariant (nextLetter a) (nextLetter b) c d) ∧
  (invariant a b c d = invariant (nextLetter a) b (nextLetter c) d) :=
sorry

/-- The initial matrix (a) -/
def matrix_a : LetterMatrix := sorry

/-- The target matrix (b) -/
def matrix_b : LetterMatrix := sorry

/-- Theorem: Matrix (a) cannot be transformed into matrix (b) -/
theorem cannot_transform (ops : ℕ) :
  ∀ (m : LetterMatrix), 
    (∃ (i : Fin 4), ∀ (j : Fin 4), m i j = nextLetter (matrix_a i j)) ∨
    (∃ (j : Fin 4), ∀ (i : Fin 4), m i j = nextLetter (matrix_a i j)) →
    m ≠ matrix_b :=
sorry

end invariant_preserved_cannot_transform_l1514_151486


namespace mildred_oranges_proof_l1514_151476

/-- Calculates the remaining oranges after Mildred's father and friend take some. -/
def remaining_oranges (initial : Float) (father_eats : Float) (friend_takes : Float) : Float :=
  initial - father_eats - friend_takes

/-- Proves that Mildred has 71.5 oranges left after her father and friend take some. -/
theorem mildred_oranges_proof (initial : Float) (father_eats : Float) (friend_takes : Float)
    (h1 : initial = 77.5)
    (h2 : father_eats = 2.25)
    (h3 : friend_takes = 3.75) :
    remaining_oranges initial father_eats friend_takes = 71.5 := by
  sorry

end mildred_oranges_proof_l1514_151476


namespace average_marks_combined_classes_l1514_151431

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 22 →
  n2 = 28 →
  avg1 = 40 →
  avg2 = 60 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 51.2 := by
  sorry

end average_marks_combined_classes_l1514_151431


namespace discount_gain_percent_l1514_151411

theorem discount_gain_percent (marked_price : ℝ) (cost_price : ℝ) (discount_rate : ℝ) :
  cost_price = 0.64 * marked_price →
  discount_rate = 0.12 →
  let selling_price := marked_price * (1 - discount_rate)
  let gain_percent := ((selling_price - cost_price) / cost_price) * 100
  gain_percent = 37.5 := by
  sorry

end discount_gain_percent_l1514_151411


namespace prob_A_union_B_l1514_151489

-- Define the sample space for a fair six-sided die
def Ω : Finset Nat := Finset.range 6

-- Define the probability measure
def P (S : Finset Nat) : ℚ := (S.card : ℚ) / (Ω.card : ℚ)

-- Define event A: getting a 3
def A : Finset Nat := {2}

-- Define event B: getting an even number
def B : Finset Nat := {1, 3, 5}

-- Theorem statement
theorem prob_A_union_B : P (A ∪ B) = 2/3 := by
  sorry

end prob_A_union_B_l1514_151489
