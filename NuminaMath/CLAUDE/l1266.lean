import Mathlib

namespace rectangle_perimeter_l1266_126674

theorem rectangle_perimeter (l w : ℝ) :
  l > 0 ∧ w > 0 ∧                             -- Positive dimensions
  2 * (w + l / 6) = 40 ∧                       -- Perimeter of smaller rectangle
  6 * w = l →                                  -- Relationship between l and w
  2 * (l + w) = 280 :=                         -- Perimeter of original rectangle
by sorry

end rectangle_perimeter_l1266_126674


namespace cylinder_volume_ratio_l1266_126637

/-- Theorem: For two cylinders with given properties, the ratio of their volumes is 3/2 -/
theorem cylinder_volume_ratio (S₁ S₂ V₁ V₂ : ℝ) (h₁ : S₁ / S₂ = 9 / 4) (h₂ : S₁ > 0) (h₃ : S₂ > 0) : 
  ∃ (R r H h : ℝ), 
    R > 0 ∧ r > 0 ∧ H > 0 ∧ h > 0 ∧
    S₁ = π * R^2 ∧
    S₂ = π * r^2 ∧
    V₁ = π * R^2 * H ∧
    V₂ = π * r^2 * h ∧
    2 * π * R * H = 2 * π * r * h →
    V₁ / V₂ = 3 / 2 := by
  sorry

end cylinder_volume_ratio_l1266_126637


namespace function_relation_characterization_l1266_126648

theorem function_relation_characterization 
  (f g : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f m - f n = (m - n) * (g m + g n)) :
  ∃ a b c : ℕ, 
    (∀ n : ℕ, f n = a * n^2 + 2 * b * n + c) ∧ 
    (∀ n : ℕ, g n = a * n + b) :=
by sorry

end function_relation_characterization_l1266_126648


namespace six_digit_numbers_with_zero_l1266_126687

/-- The number of digits in the numbers we're considering -/
def num_digits : ℕ := 6

/-- The total number of possible digits (0 to 9) -/
def base : ℕ := 10

/-- The number of non-zero digits (1 to 9) -/
def non_zero_digits : ℕ := 9

/-- The total number of 6-digit numbers -/
def total_numbers : ℕ := non_zero_digits * base ^ (num_digits - 1)

/-- The number of 6-digit numbers with no zeros -/
def numbers_without_zero : ℕ := non_zero_digits ^ num_digits

/-- The number of 6-digit numbers with at least one zero -/
def numbers_with_zero : ℕ := total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero : numbers_with_zero = 368559 := by
  sorry

end six_digit_numbers_with_zero_l1266_126687


namespace sum_divisible_by_11211_l1266_126635

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define a structure for five consecutive digits
structure ConsecutiveDigits where
  a : Digit
  h1 : a.val + 1 < 10  -- Ensure there are 4 consecutive digits after a
  b : Digit
  h2 : b.val = a.val + 1
  c : Digit
  h3 : c.val = a.val + 2
  d : Digit
  h4 : d.val = a.val + 3
  e : Digit
  h5 : e.val = a.val + 4

-- Define the function to create the number abcde
def makeNumber (digits : ConsecutiveDigits) : ℕ :=
  10000 * digits.a.val + 1000 * digits.b.val + 100 * digits.c.val + 10 * digits.d.val + digits.e.val

-- Define the function to create the reversed number edcba
def makeReversedNumber (digits : ConsecutiveDigits) : ℕ :=
  10000 * digits.e.val + 1000 * digits.d.val + 100 * digits.c.val + 10 * digits.b.val + digits.a.val

-- Theorem statement
theorem sum_divisible_by_11211 (digits : ConsecutiveDigits) :
  11211 ∣ (makeNumber digits + makeReversedNumber digits) := by
  sorry

end sum_divisible_by_11211_l1266_126635


namespace computer_profit_percentage_l1266_126601

theorem computer_profit_percentage (C : ℝ) (P : ℝ) : 
  2560 = C + 0.6 * C →
  2240 = C + P / 100 * C →
  P = 40 := by
sorry

end computer_profit_percentage_l1266_126601


namespace composite_divisor_of_product_l1266_126644

def product_up_to (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem composite_divisor_of_product (m : ℕ) :
  m > 1 →
  (m ∣ product_up_to m) ↔ (¬ Nat.Prime m ∧ m ≠ 4) :=
by sorry

end composite_divisor_of_product_l1266_126644


namespace smallest_number_satisfying_congruences_l1266_126619

theorem smallest_number_satisfying_congruences : ∃ n : ℕ, 
  n > 0 ∧
  n % 4 = 1 ∧
  n % 5 = 1 ∧
  n % 6 = 1 ∧
  n % 7 = 0 ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 1 ∧ m % 6 = 1 ∧ m % 7 = 0 → m ≥ n) ∧
  n = 301 :=
by sorry

end smallest_number_satisfying_congruences_l1266_126619


namespace parabola_line_intersection_l1266_126678

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- The point Q -/
def Q : ℝ × ℝ := (10, 6)

/-- The line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - Q.2 = m * (p.1 - Q.1)}

/-- The condition for non-intersection -/
def no_intersection (m : ℝ) : Prop :=
  line_through_Q m ∩ P = ∅

theorem parabola_line_intersection :
  ∃ (r s : ℝ), (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) → r + s = 40 := by
  sorry

end parabola_line_intersection_l1266_126678


namespace find_a_l1266_126611

def U : Finset ℕ := {1, 3, 5, 7}

theorem find_a (a : ℕ) : 
  let M : Finset ℕ := {1, a - 5}
  M ⊆ U ∧ 
  (U \ M : Finset ℕ) = {5, 7} →
  a = 8 := by
sorry

end find_a_l1266_126611


namespace crayon_selection_combinations_l1266_126649

theorem crayon_selection_combinations : Nat.choose 15 5 = 3003 := by
  sorry

end crayon_selection_combinations_l1266_126649


namespace eighth_grade_students_l1266_126660

/-- The number of students in eighth grade -/
def total_students (num_girls : ℕ) (num_boys : ℕ) : ℕ :=
  num_girls + num_boys

/-- The relationship between the number of boys and girls -/
def boys_girls_relation (num_girls : ℕ) (num_boys : ℕ) : Prop :=
  num_boys = 2 * num_girls - 16

theorem eighth_grade_students :
  ∃ (num_boys : ℕ),
    boys_girls_relation 28 num_boys ∧
    total_students 28 num_boys = 68 :=
by sorry

end eighth_grade_students_l1266_126660


namespace six_by_six_grid_squares_l1266_126664

/-- The number of squares of size n×n in a 6×6 grid -/
def squaresOfSize (n : ℕ) : ℕ := (6 - n) * (6 - n)

/-- The total number of squares in a 6×6 grid -/
def totalSquares : ℕ :=
  squaresOfSize 1 + squaresOfSize 2 + squaresOfSize 3 + squaresOfSize 4

theorem six_by_six_grid_squares :
  totalSquares = 54 := by
  sorry

end six_by_six_grid_squares_l1266_126664


namespace rectangular_field_diagonal_l1266_126663

/-- Given a rectangular field with one side of 15 meters and an area of 120 square meters,
    the length of its diagonal is 17 meters. -/
theorem rectangular_field_diagonal (l w d : ℝ) : 
  l = 15 → 
  l * w = 120 → 
  d^2 = l^2 + w^2 → 
  d = 17 := by
sorry

end rectangular_field_diagonal_l1266_126663


namespace cupboard_books_count_l1266_126668

theorem cupboard_books_count :
  ∃! x : ℕ, x ≤ 400 ∧
    x % 4 = 1 ∧
    x % 5 = 1 ∧
    x % 6 = 1 ∧
    x % 7 = 0 ∧
    x = 301 := by
  sorry

end cupboard_books_count_l1266_126668


namespace bryson_shoes_pairs_l1266_126602

/-- Given that Bryson has a total of 4 new shoes and a pair of shoes consists of 2 shoes,
    prove that the number of pairs of shoes he bought is 2. -/
theorem bryson_shoes_pairs : 
  ∀ (total_shoes : ℕ) (shoes_per_pair : ℕ),
    total_shoes = 4 →
    shoes_per_pair = 2 →
    total_shoes / shoes_per_pair = 2 := by
  sorry

end bryson_shoes_pairs_l1266_126602


namespace index_card_problem_l1266_126699

theorem index_card_problem (n : ℕ+) : 
  ((n : ℝ) * (n + 1) * (2 * n + 1) / 6) / ((n : ℝ) * (n + 1) / 2) = 2023 → n = 3034 := by
  sorry

end index_card_problem_l1266_126699


namespace max_expr_value_l1266_126604

def S : Finset ℕ := {1, 2, 3, 4}

def expr (e f g h : ℕ) : ℕ := e * f^g - h

theorem max_expr_value :
  ∃ (e f g h : ℕ), e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ f ≠ g ∧ f ≠ h ∧ g ≠ h ∧
  expr e f g h = 161 ∧
  ∀ (a b c d : ℕ), a ∈ S → b ∈ S → c ∈ S → d ∈ S →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  expr a b c d ≤ 161 :=
by sorry

end max_expr_value_l1266_126604


namespace sequence_ratio_l1266_126653

/-- Given an arithmetic sequence and a geometric sequence with specific properties, 
    prove that (b-a)/d = 1/2 --/
theorem sequence_ratio (a b c d e : ℝ) : 
  ((-1 : ℝ) - a = a - b) ∧ (b - (-4 : ℝ) = a - b) ∧  -- arithmetic sequence condition
  (c = (-1 : ℝ) * d / c) ∧ (d = c * e / d) ∧ (e = d * (-4 : ℝ) / e) →  -- geometric sequence condition
  (b - a) / d = (1 : ℝ) / 2 := by
  sorry

end sequence_ratio_l1266_126653


namespace logarithmic_equation_solution_l1266_126679

/-- Given that 5(log_a x)^2 + 9(log_b x)^2 = (20(log x)^2) / (log a log b) and a, b, x > 1,
    prove that b = a^((20+√220)/10) or b = a^((20-√220)/10) -/
theorem logarithmic_equation_solution (a b x : ℝ) (ha : a > 1) (hb : b > 1) (hx : x > 1)
  (h : 5 * (Real.log x / Real.log a)^2 + 9 * (Real.log x / Real.log b)^2 = 20 * (Real.log x)^2 / (Real.log a * Real.log b)) :
  b = a^((20 + Real.sqrt 220) / 10) ∨ b = a^((20 - Real.sqrt 220) / 10) := by
  sorry

end logarithmic_equation_solution_l1266_126679


namespace solution_of_linear_system_l1266_126661

theorem solution_of_linear_system :
  ∃ (x y : ℝ), x + 3 * y = 7 ∧ y = 2 * x ∧ x = 1 ∧ y = 2 := by
  sorry

end solution_of_linear_system_l1266_126661


namespace arithmetic_sequence_problem_l1266_126693

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 9)
  (h_sum : a 7 + a 8 = 28) :
  a 4 = 7 :=
sorry

end arithmetic_sequence_problem_l1266_126693


namespace sixth_power_sum_l1266_126632

theorem sixth_power_sum (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12098 := by
  sorry

end sixth_power_sum_l1266_126632


namespace extreme_values_of_f_l1266_126629

/-- Given a positive constant a, prove the extreme values of f(x) = sin(2x) + √(sin²(2x) + a cos²(x)) -/
theorem extreme_values_of_f (a : ℝ) (ha : a > 0) :
  let f := fun (x : ℝ) => Real.sin (2 * x) + Real.sqrt ((Real.sin (2 * x))^2 + a * (Real.cos x)^2)
  (∀ x, f x ≥ 0) ∧ 
  (∃ x, f x = 0) ∧
  (∀ x, f x ≤ Real.sqrt (a + 4)) ∧
  (∃ x, f x = Real.sqrt (a + 4)) := by
  sorry

end extreme_values_of_f_l1266_126629


namespace polygon_sides_sum_l1266_126690

/-- The sum of interior angles of a convex polygon with n sides --/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- The number of sides in a triangle --/
def triangle_sides : ℕ := 3

/-- The number of sides in a hexagon --/
def hexagon_sides : ℕ := 6

/-- The sum of interior angles of the given polygons --/
def total_angle_sum : ℝ := 1260

theorem polygon_sides_sum :
  ∃ (n : ℕ), n = triangle_sides + hexagon_sides ∧
  total_angle_sum = interior_angle_sum triangle_sides + interior_angle_sum hexagon_sides + interior_angle_sum (n - triangle_sides - hexagon_sides) :=
sorry

end polygon_sides_sum_l1266_126690


namespace range_of_a_l1266_126676

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then 3^x + 4*a else 2*x + a^2

theorem range_of_a (a : ℝ) (h₁ : a > 0) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ≥ 7 :=
by sorry

end range_of_a_l1266_126676


namespace value_of_a_l1266_126639

/-- Given a function f(x) = ax³ + 3x² - 6 where f'(-1) = 4, prove that a = 10/3 -/
theorem value_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = a * x^3 + 3 * x^2 - 6)
  (h2 : deriv f (-1) = 4) : 
  a = 10/3 := by
  sorry

end value_of_a_l1266_126639


namespace area_of_pentagon_l1266_126641

-- Define the points and lengths
structure Triangle :=
  (A B C : ℝ × ℝ)

def AB : ℝ := 5
def BC : ℝ := 3
def BD : ℝ := 3
def EC : ℝ := 1
def FD : ℝ := 2

-- Define the triangles
def triangleABC : Triangle := sorry
def triangleABD : Triangle := sorry

-- Define that ABC and ABD are right triangles
axiom ABC_right : triangleABC.C.1^2 + triangleABC.C.2^2 = AB^2
axiom ABD_right : triangleABD.C.1^2 + triangleABD.C.2^2 = AB^2

-- Define that C and D are on opposite sides of AB
axiom C_D_opposite : triangleABC.C.2 * triangleABD.C.2 < 0

-- Define points E and F
def E : ℝ × ℝ := sorry
def F : ℝ × ℝ := sorry

-- Define that E is on AC and F is on AD
axiom E_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (t * triangleABC.A.1 + (1 - t) * triangleABC.C.1, t * triangleABC.A.2 + (1 - t) * triangleABC.C.2)
axiom F_on_AD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (t * triangleABD.A.1 + (1 - t) * triangleABD.C.1, t * triangleABD.A.2 + (1 - t) * triangleABD.C.2)

-- Define the area of a polygon
def area (polygon : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem to prove
theorem area_of_pentagon :
  area [E, C, B, D, F] = 303 / 25 := sorry

end area_of_pentagon_l1266_126641


namespace thomas_leftover_money_l1266_126607

theorem thomas_leftover_money (num_books : ℕ) (book_price : ℚ) (record_price : ℚ) (num_records : ℕ) :
  num_books = 200 →
  book_price = 3/2 →
  record_price = 3 →
  num_records = 75 →
  (num_books : ℚ) * book_price - (num_records : ℚ) * record_price = 75 :=
by sorry

end thomas_leftover_money_l1266_126607


namespace similarity_transformation_result_l1266_126666

/-- A similarity transformation in 2D space -/
structure Similarity2D where
  center : ℝ × ℝ
  ratio : ℝ

/-- Apply a similarity transformation to a point -/
def apply_similarity (s : Similarity2D) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {(s.ratio * (p.1 - s.center.1) + s.center.1, s.ratio * (p.2 - s.center.2) + s.center.2),
   (-s.ratio * (p.1 - s.center.1) + s.center.1, -s.ratio * (p.2 - s.center.2) + s.center.2)}

theorem similarity_transformation_result :
  let s : Similarity2D := ⟨(0, 0), 2⟩
  let A : ℝ × ℝ := (2, 2)
  apply_similarity s A = {(4, 4), (-4, -4)} := by
  sorry

end similarity_transformation_result_l1266_126666


namespace rectangular_box_volume_l1266_126675

/-- The volume of a rectangular box with face areas 24, 16, and 6 square inches is 48 cubic inches -/
theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 24)
  (area2 : w * h = 16)
  (area3 : l * h = 6) :
  l * w * h = 48 := by
  sorry

end rectangular_box_volume_l1266_126675


namespace geometric_sequence_fifth_term_l1266_126612

/-- A geometric sequence with first term 1/3 and the property that 2a_2 = a_4 -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/3 ∧
  (∃ q : ℚ, ∀ n : ℕ, a n = 1/3 * q^(n-1)) ∧
  2 * (a 2) = a 4

theorem geometric_sequence_fifth_term
  (a : ℕ → ℚ)
  (h : geometric_sequence a) :
  a 5 = 4/3 :=
sorry

end geometric_sequence_fifth_term_l1266_126612


namespace line_segment_endpoint_l1266_126626

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 →
  Real.sqrt ((x - 2)^2 + (10 - 5)^2) = 13 →
  x = 14 :=
by
  sorry

end line_segment_endpoint_l1266_126626


namespace cosine_angle_in_ellipse_l1266_126673

/-- The cosine of the angle F₁PF₂ in an ellipse with specific properties -/
theorem cosine_angle_in_ellipse (P : ℝ × ℝ) :
  let F₁ : ℝ × ℝ := (-4, 0)
  let F₂ : ℝ × ℝ := (4, 0)
  let on_ellipse (p : ℝ × ℝ) := p.1^2 / 25 + p.2^2 / 9 = 1
  let triangle_area (p : ℝ × ℝ) := abs ((p.1 - F₁.1) * (p.2 - F₂.2) - (p.2 - F₁.2) * (p.1 - F₂.1)) / 2
  on_ellipse P ∧ triangle_area P = 3 * Real.sqrt 3 →
  let PF₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let PF₂ := Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)
  let cos_angle := (PF₁^2 + PF₂^2 - 64) / (2 * PF₁ * PF₂)
  cos_angle = 1/2 := by
sorry

end cosine_angle_in_ellipse_l1266_126673


namespace midpoint_condition_l1266_126682

-- Define the triangle ABC
structure RightTriangle where
  a : ℝ
  b : ℝ
  hypotenuse : ℝ
  hyp_eq : hypotenuse = Real.sqrt (a^2 + b^2)

-- Define point P on the hypotenuse
def PointOnHypotenuse (triangle : RightTriangle) := 
  { x : ℝ // 0 ≤ x ∧ x ≤ triangle.hypotenuse }

-- Define s as AP² + PB²
def s (triangle : RightTriangle) (p : PointOnHypotenuse triangle) : ℝ :=
  p.val^2 + (triangle.hypotenuse - p.val)^2

-- Define CP²
def CP_squared (triangle : RightTriangle) : ℝ := triangle.a^2

-- Theorem statement
theorem midpoint_condition (triangle : RightTriangle) :
  ∀ p : PointOnHypotenuse triangle, 
    s triangle p = 2 * CP_squared triangle ↔ 
    p.val = triangle.hypotenuse / 2 := by sorry

end midpoint_condition_l1266_126682


namespace divisibility_condition_l1266_126608

def six_digit_number (n : ℕ) : ℕ := 850000 + n * 1000 + 475

theorem divisibility_condition (n : ℕ) : 
  n < 10 → (six_digit_number n % 45 = 0 ↔ n = 7) := by sorry

end divisibility_condition_l1266_126608


namespace defective_pencils_count_l1266_126623

/-- The probability of selecting 3 non-defective pencils out of N non-defective pencils from a total of 6 pencils. -/
def probability (N : ℕ) : ℚ :=
  (Nat.choose N 3 : ℚ) / (Nat.choose 6 3 : ℚ)

/-- The number of defective pencils in a box of 6 pencils. -/
def num_defective (N : ℕ) : ℕ := 6 - N

theorem defective_pencils_count :
  ∃ N : ℕ, N ≤ 6 ∧ probability N = 1/5 ∧ num_defective N = 2 := by
  sorry

#check defective_pencils_count

end defective_pencils_count_l1266_126623


namespace constant_function_equals_derivative_l1266_126625

theorem constant_function_equals_derivative :
  ∀ (f : ℝ → ℝ), (∀ x, f x = 0) → ∀ x, f x = deriv f x := by sorry

end constant_function_equals_derivative_l1266_126625


namespace evaluate_expression_l1266_126613

theorem evaluate_expression (x : ℤ) (h : x = -2) : 5 * x + 7 = -3 := by
  sorry

end evaluate_expression_l1266_126613


namespace heartsuit_three_four_l1266_126651

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_four : heartsuit 3 4 = 36 := by
  sorry

end heartsuit_three_four_l1266_126651


namespace max_x_minus_y_l1266_126683

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), a^2 + b^2 - 4*a - 2*b - 4 = 0 ∧ w = a - b) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l1266_126683


namespace min_value_problem_l1266_126647

theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (sum_constraint : a + b + c = 12) (product_constraint : a * b * c = 27) :
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) ≥ 12 := by
  sorry

end min_value_problem_l1266_126647


namespace equation_one_solutions_l1266_126603

theorem equation_one_solutions (x : ℝ) :
  3 * (x - 1)^2 = 12 ↔ x = 3 ∨ x = -1 := by sorry

end equation_one_solutions_l1266_126603


namespace mikes_bills_l1266_126600

theorem mikes_bills (total_amount : ℕ) (bill_value : ℕ) (num_bills : ℕ) :
  total_amount = 45 →
  bill_value = 5 →
  total_amount = bill_value * num_bills →
  num_bills = 9 := by
sorry

end mikes_bills_l1266_126600


namespace sum_abc_equals_16_l1266_126618

theorem sum_abc_equals_16 (a b c : ℕ+) 
  (h1 : a * b + 2 * c + 3 = 47)
  (h2 : b * c + 2 * a + 3 = 47)
  (h3 : a * c + 2 * b + 3 = 47) :
  a + b + c = 16 := by
  sorry

end sum_abc_equals_16_l1266_126618


namespace profit_maximum_l1266_126614

/-- Profit function for a product -/
def profit (m : ℝ) : ℝ := (m - 8) * (900 - 15 * m)

/-- Maximum profit expression -/
def max_profit_expr (m : ℝ) : ℝ := -15 * (m - 34)^2 + 10140

theorem profit_maximum :
  ∃ (m : ℝ), 
    (∀ (x : ℝ), profit x ≤ profit m) ∧
    (profit m = max_profit_expr m) ∧
    (m = 34) :=
sorry

end profit_maximum_l1266_126614


namespace arithmetic_mean_problem_l1266_126659

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 20 + 5 + y) / 5 = 12 → y = 12 := by
sorry

end arithmetic_mean_problem_l1266_126659


namespace constant_sum_of_powers_l1266_126627

theorem constant_sum_of_powers (n : ℕ) : 
  (∀ x y z : ℝ, x + y + z = 0 → x * y * z = 1 → 
    ∃ c : ℝ, ∀ x' y' z' : ℝ, x' + y' + z' = 0 → x' * y' * z' = 1 → 
      x'^n + y'^n + z'^n = c) ↔ 
  n = 1 ∨ n = 3 := by sorry

end constant_sum_of_powers_l1266_126627


namespace water_usage_for_car_cleaning_l1266_126677

/-- Represents the problem of calculating water usage for car cleaning --/
theorem water_usage_for_car_cleaning
  (total_water : ℝ)
  (plant_water_difference : ℝ)
  (plate_clothes_water : ℝ)
  (h1 : total_water = 65)
  (h2 : plant_water_difference = 11)
  (h3 : plate_clothes_water = 24)
  (h4 : plate_clothes_water * 2 = total_water - (2 * car_water + (2 * car_water - plant_water_difference))) :
  ∃ (car_water : ℝ), car_water = 7 :=
by sorry

end water_usage_for_car_cleaning_l1266_126677


namespace value_of_y_l1266_126646

theorem value_of_y (x y : ℝ) (h1 : x^2 - 3*x + 2 = y + 2) (h2 : x = -5) : y = 40 := by
  sorry

end value_of_y_l1266_126646


namespace compute_expression_l1266_126657

theorem compute_expression : 10 + 4 * (5 + 3)^3 = 2058 := by
  sorry

end compute_expression_l1266_126657


namespace function_inequality_condition_l1266_126684

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f x = 4 * x + 3) →
  a > 0 →
  b > 0 →
  (∀ x : ℝ, |x + 3| < b → |f x + 5| < a) ↔
  b ≤ a / 4 := by
  sorry

end function_inequality_condition_l1266_126684


namespace line_intersection_l1266_126616

theorem line_intersection :
  ∃! p : ℝ × ℝ, 
    (p.2 = -3 * p.1 + 1) ∧ 
    (p.2 + 1 = 15 * p.1) ∧ 
    p = (1/9, 2/3) := by
  sorry

end line_intersection_l1266_126616


namespace binomial_divisibility_l1266_126696

theorem binomial_divisibility (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  let p := 2 * k - 1
  Prime p →
  p ∣ (n.choose 2 - k.choose 2) →
  p^2 ∣ (n.choose 2 - k.choose 2) := by
  sorry

end binomial_divisibility_l1266_126696


namespace number_thought_of_l1266_126671

theorem number_thought_of (x : ℝ) : (x / 6 + 5 = 17) → x = 72 := by
  sorry

end number_thought_of_l1266_126671


namespace cousin_payment_l1266_126605

def friend_payment : ℕ := 5
def brother_payment : ℕ := 8
def total_days : ℕ := 7
def total_amount : ℕ := 119

theorem cousin_payment (cousin_pay : ℕ) : 
  (friend_payment * total_days + brother_payment * total_days + cousin_pay * total_days = total_amount) →
  cousin_pay = 4 := by
sorry

end cousin_payment_l1266_126605


namespace danny_collection_difference_l1266_126620

/-- Represents Danny's collection of bottle caps and wrappers --/
structure Collection where
  park_caps : ℕ
  park_wrappers : ℕ
  beach_caps : ℕ
  beach_wrappers : ℕ
  forest_caps : ℕ
  forest_wrappers : ℕ
  previous_caps : ℕ
  previous_wrappers : ℕ

/-- Calculates the total number of bottle caps in the collection --/
def total_caps (c : Collection) : ℕ :=
  c.park_caps + c.beach_caps + c.forest_caps + c.previous_caps

/-- Calculates the total number of wrappers in the collection --/
def total_wrappers (c : Collection) : ℕ :=
  c.park_wrappers + c.beach_wrappers + c.forest_wrappers + c.previous_wrappers

/-- Theorem stating the difference between bottle caps and wrappers in Danny's collection --/
theorem danny_collection_difference :
  ∀ (c : Collection),
  c.park_caps = 58 →
  c.park_wrappers = 25 →
  c.beach_caps = 34 →
  c.beach_wrappers = 15 →
  c.forest_caps = 21 →
  c.forest_wrappers = 32 →
  c.previous_caps = 12 →
  c.previous_wrappers = 11 →
  total_caps c - total_wrappers c = 42 := by
  sorry

end danny_collection_difference_l1266_126620


namespace jons_number_l1266_126688

theorem jons_number : ∃ (x : ℝ), 5 * (3 * x + 6) - 8 = 142 ∧ x = 8 := by sorry

end jons_number_l1266_126688


namespace expression_simplification_l1266_126650

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ((x^3 + 1) / x * (y^3 + 1) / y) - ((x^3 - 1) / y * (y^3 - 1) / x) = 2*x^2 + 2*y^2 := by
  sorry

end expression_simplification_l1266_126650


namespace fruits_remaining_l1266_126655

/-- Calculates the number of fruits remaining after picking from multiple trees -/
theorem fruits_remaining
  (num_trees : ℕ)
  (fruits_per_tree : ℕ)
  (fraction_picked : ℚ)
  (h1 : num_trees = 8)
  (h2 : fruits_per_tree = 200)
  (h3 : fraction_picked = 2/5) :
  num_trees * fruits_per_tree - num_trees * (fruits_per_tree * fraction_picked) = 960 :=
by
  sorry

#check fruits_remaining

end fruits_remaining_l1266_126655


namespace min_value_ab_l1266_126638

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a * b ≥ 9 := by
sorry

end min_value_ab_l1266_126638


namespace parallelogram_perimeter_l1266_126697

/-- A parallelogram with adjacent sides of length 3 and 5 has a perimeter of 16. -/
theorem parallelogram_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 5) :
  2 * (a + b) = 16 := by
  sorry

end parallelogram_perimeter_l1266_126697


namespace ellipse_foci_on_y_axis_iff_l1266_126656

/-- Represents an ellipse with the equation mx^2 + ny^2 = 1 -/
structure Ellipse (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1

/-- Determines if an ellipse has foci on the y-axis -/
def hasFociOnYAxis (e : Ellipse m n) : Prop := sorry

/-- The main theorem stating that m > n > 0 is necessary and sufficient for
    the equation mx^2 + ny^2 = 1 to represent an ellipse with foci on the y-axis -/
theorem ellipse_foci_on_y_axis_iff (m n : ℝ) :
  (∃ e : Ellipse m n, hasFociOnYAxis e) ↔ m > n ∧ n > 0 := by sorry

end ellipse_foci_on_y_axis_iff_l1266_126656


namespace tank_plastering_cost_l1266_126640

/-- Calculates the cost of plastering a rectangular tank's walls and bottom -/
def plasteringCost (length width depth rate : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * rate

/-- Theorem stating the cost of plastering the given tank -/
theorem tank_plastering_cost :
  plasteringCost 25 12 6 0.75 = 558 := by
  sorry

#eval plasteringCost 25 12 6 0.75

end tank_plastering_cost_l1266_126640


namespace triangle_with_sine_sides_l1266_126691

theorem triangle_with_sine_sides 
  (α β γ : Real) 
  (h_triangle : α + β + γ = π) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_less_than_pi : α < π ∧ β < π ∧ γ < π) : 
  ∃ (a b c : Real), 
    a = Real.sin α ∧ 
    b = Real.sin β ∧ 
    c = Real.sin γ ∧ 
    a + b > c ∧ 
    b + c > a ∧ 
    c + a > b := by
  sorry

end triangle_with_sine_sides_l1266_126691


namespace perfect_square_binomial_l1266_126621

/-- If 9x^2 - 18x + a is the square of a binomial, then a = 9 -/
theorem perfect_square_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end perfect_square_binomial_l1266_126621


namespace min_ratio_folded_strings_l1266_126692

theorem min_ratio_folded_strings (m n : ℕ) : 
  (∃ a : ℕ+, (2^m + 1 : ℕ) = a * (2^n + 1) ∧ a > 1) → 
  (∃ a : ℕ+, (2^m + 1 : ℕ) = a * (2^n + 1) ∧ a > 1 ∧ 
    ∀ b : ℕ+, (2^m + 1 : ℕ) = b * (2^n + 1) ∧ b > 1 → a ≤ b) → 
  (∃ m n : ℕ, (2^m + 1 : ℕ) = 3 * (2^n + 1)) :=
by sorry

end min_ratio_folded_strings_l1266_126692


namespace limit_nonexistent_l1266_126694

/-- The limit of (x^2 - y^2) / (x^2 + y^2) as x and y approach 0 does not exist. -/
theorem limit_nonexistent :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    x ≠ 0 ∧ y ≠ 0 ∧ x^2 + y^2 < δ^2 →
    |((x^2 - y^2) / (x^2 + y^2)) - L| < ε :=
by sorry

end limit_nonexistent_l1266_126694


namespace points_in_quadrants_I_and_II_l1266_126606

/-- A point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The set of points satisfying the given inequalities -/
def SatisfyingPoints : Set Point :=
  {p : Point | p.y > 3 * p.x ∧ p.y > 5 - 2 * p.x}

/-- A point is in Quadrant I if both x and y are positive -/
def InQuadrantI (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- A point is in Quadrant II if x is negative and y is positive -/
def InQuadrantII (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: All points satisfying the inequalities are in Quadrants I or II -/
theorem points_in_quadrants_I_and_II :
  ∀ p ∈ SatisfyingPoints, InQuadrantI p ∨ InQuadrantII p :=
by sorry

end points_in_quadrants_I_and_II_l1266_126606


namespace online_store_commission_percentage_l1266_126672

theorem online_store_commission_percentage 
  (cost : ℝ) 
  (online_price : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : cost = 17) 
  (h2 : online_price = 25.5) 
  (h3 : profit_percentage = 0.2) : 
  (online_price - (cost * (1 + profit_percentage))) / online_price = 0.2 := by
sorry

end online_store_commission_percentage_l1266_126672


namespace polynomial_roots_l1266_126681

def p (x : ℝ) : ℝ := x^3 - 3*x^2 - 4*x + 12

theorem polynomial_roots :
  (∀ x : ℝ, p x = 0 ↔ x = 2 ∨ x = -2 ∨ x = 3) :=
by sorry

end polynomial_roots_l1266_126681


namespace complex_number_in_third_quadrant_l1266_126609

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (2 - I) / I
  (z.re < 0) ∧ (z.im < 0) := by sorry

end complex_number_in_third_quadrant_l1266_126609


namespace quadratic_equation_solution_l1266_126698

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 3*x + 2 = 0 ↔ (x = 1 ∨ x = 2) := by sorry

end quadratic_equation_solution_l1266_126698


namespace defective_pens_count_l1266_126686

def total_pens : ℕ := 10

def prob_non_defective : ℚ := 0.6222222222222222

theorem defective_pens_count (defective : ℕ) 
  (h1 : defective ≤ total_pens)
  (h2 : (((total_pens - defective) : ℚ) / total_pens) * 
        (((total_pens - defective - 1) : ℚ) / (total_pens - 1)) = prob_non_defective) :
  defective = 2 := by sorry

end defective_pens_count_l1266_126686


namespace parity_of_D_2024_2025_2026_l1266_126615

def D : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | 3 => 2
  | n + 4 => D (n + 3) + D (n + 1)

theorem parity_of_D_2024_2025_2026 :
  Odd (D 2024) ∧ Even (D 2025) ∧ Even (D 2026) := by
  sorry

end parity_of_D_2024_2025_2026_l1266_126615


namespace wheels_in_garage_is_39_l1266_126622

/-- The number of wheels in a garage with various vehicles and items -/
def total_wheels_in_garage (
  num_cars : ℕ)
  (num_lawnmowers : ℕ)
  (num_bicycles : ℕ)
  (num_tricycles : ℕ)
  (num_unicycles : ℕ)
  (num_skateboards : ℕ)
  (num_wheelbarrows : ℕ)
  (num_four_wheeled_wagons : ℕ)
  (num_two_wheeled_dollies : ℕ)
  (num_four_wheeled_shopping_carts : ℕ)
  (num_two_wheeled_scooters : ℕ) : ℕ :=
  num_cars * 4 +
  num_lawnmowers * 4 +
  num_bicycles * 2 +
  num_tricycles * 3 +
  num_unicycles * 1 +
  num_skateboards * 4 +
  num_wheelbarrows * 1 +
  num_four_wheeled_wagons * 4 +
  num_two_wheeled_dollies * 2 +
  num_four_wheeled_shopping_carts * 4 +
  num_two_wheeled_scooters * 2

/-- Theorem stating that the total number of wheels in the garage is 39 -/
theorem wheels_in_garage_is_39 :
  total_wheels_in_garage 2 1 3 1 1 1 1 1 1 1 1 = 39 := by
  sorry

end wheels_in_garage_is_39_l1266_126622


namespace tom_gave_balloons_to_fred_l1266_126645

/-- The number of balloons Tom gave to Fred -/
def balloons_given (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem tom_gave_balloons_to_fred (initial : ℕ) (remaining : ℕ) 
  (h1 : initial = 30) (h2 : remaining = 14) :
  balloons_given initial remaining = 16 := by
  sorry

end tom_gave_balloons_to_fred_l1266_126645


namespace complex_arithmetic_equality_l1266_126642

theorem complex_arithmetic_equality : 9 - (8 + 7) * 6 + 5^2 - (4 * 3) + 2 - 1 = -67 := by
  sorry

end complex_arithmetic_equality_l1266_126642


namespace distance_ratio_bound_l1266_126643

/-- Manhattan distance between two points -/
def manhattan_distance (p q : ℝ × ℝ) : ℝ :=
  |p.1 - q.1| + |p.2 - q.2|

/-- The theorem to be proved -/
theorem distance_ratio_bound (points : Finset (ℝ × ℝ)) (h : points.card = 2023) :
  let distances := {d | ∃ (p q : ℝ × ℝ), p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ d = manhattan_distance p q}
  (⨆ d ∈ distances, d) / (⨅ d ∈ distances, d) ≥ 44 :=
sorry

end distance_ratio_bound_l1266_126643


namespace brianna_marbles_l1266_126680

/-- Calculates the number of marbles Brianna has remaining after a series of events. -/
def remaining_marbles (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost - 2 * lost - lost / 2

/-- Theorem stating that Brianna has 10 marbles remaining given the initial conditions. -/
theorem brianna_marbles : remaining_marbles 24 4 = 10 := by
  sorry

#eval remaining_marbles 24 4

end brianna_marbles_l1266_126680


namespace river_boat_capacity_l1266_126654

theorem river_boat_capacity (river_width : ℕ) (boat_width : ℕ) (space_required : ℕ) : 
  river_width = 42 ∧ boat_width = 3 ∧ space_required = 2 →
  (river_width / (boat_width + 2 * space_required) : ℕ) = 6 :=
by sorry

end river_boat_capacity_l1266_126654


namespace function_property_l1266_126624

/-- Strictly increasing function from ℕ+ to ℕ+ -/
def StrictlyIncreasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, x < y → f x < f y

/-- The image of a function f : ℕ+ → ℕ+ -/
def Image (f : ℕ+ → ℕ+) : Set ℕ+ :=
  {y : ℕ+ | ∃ x : ℕ+, f x = y}

theorem function_property (f g : ℕ+ → ℕ+) 
  (h1 : StrictlyIncreasing f)
  (h2 : StrictlyIncreasing g)
  (h3 : Image f ∪ Image g = Set.univ)
  (h4 : Image f ∩ Image g = ∅)
  (h5 : ∀ n : ℕ+, g n = f (f n) + 1) :
  f 240 = 388 := by
  sorry

end function_property_l1266_126624


namespace kelly_nintendo_games_l1266_126662

/-- Proves that Kelly's initial number of Nintendo games is 121.0 --/
theorem kelly_nintendo_games :
  ∀ x : ℝ, (x - 99 = 22.0) → x = 121.0 := by
  sorry

end kelly_nintendo_games_l1266_126662


namespace cookout_buns_l1266_126670

/-- The number of packs of burger buns Alex needs to buy for his cookout. -/
def bun_packs_needed (guests : ℕ) (burgers_per_guest : ℕ) (no_meat_guests : ℕ) (no_bread_guests : ℕ) (buns_per_pack : ℕ) : ℕ :=
  let total_guests := guests - no_meat_guests
  let total_burgers := total_guests * burgers_per_guest
  let buns_needed := total_burgers - (no_bread_guests * burgers_per_guest)
  (buns_needed + buns_per_pack - 1) / buns_per_pack

theorem cookout_buns (guests : ℕ) (burgers_per_guest : ℕ) (no_meat_guests : ℕ) (no_bread_guests : ℕ) (buns_per_pack : ℕ)
    (h1 : guests = 10)
    (h2 : burgers_per_guest = 3)
    (h3 : no_meat_guests = 1)
    (h4 : no_bread_guests = 1)
    (h5 : buns_per_pack = 8) :
  bun_packs_needed guests burgers_per_guest no_meat_guests no_bread_guests buns_per_pack = 3 := by
  sorry

end cookout_buns_l1266_126670


namespace journey_equations_correct_l1266_126669

/-- Represents a journey between two locations with uphill and flat sections. -/
structure Journey where
  uphill_speed : ℝ
  flat_speed : ℝ
  downhill_speed : ℝ
  time_ab : ℝ
  time_ba : ℝ

/-- The correct system of equations for the journey. -/
def correct_equations (j : Journey) (x y : ℝ) : Prop :=
  x / j.uphill_speed + y / j.flat_speed = j.time_ab / 60 ∧
  y / j.flat_speed + x / j.downhill_speed = j.time_ba / 60

/-- Theorem stating that the given system of equations is correct for the journey. -/
theorem journey_equations_correct (j : Journey) (x y : ℝ) 
    (h1 : j.uphill_speed = 3)
    (h2 : j.flat_speed = 4)
    (h3 : j.downhill_speed = 5)
    (h4 : j.time_ab = 70)
    (h5 : j.time_ba = 54) :
  correct_equations j x y :=
sorry

end journey_equations_correct_l1266_126669


namespace dot_product_theorem_l1266_126665

def a : ℝ × ℝ := (1, 2)

theorem dot_product_theorem (b : ℝ × ℝ) 
  (h : (2 • a - b) = (3, 1)) : a • b = 5 := by
  sorry

end dot_product_theorem_l1266_126665


namespace equation_two_solutions_l1266_126636

/-- The equation has exactly two distinct solutions when k < -3/8 -/
theorem equation_two_solutions (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁ - 3) / (k * x₁ + 2) = 2 * x₁ ∧ 
    (x₂ - 3) / (k * x₂ + 2) = 2 * x₂ ∧
    (∀ x : ℝ, (x - 3) / (k * x + 2) = 2 * x → x = x₁ ∨ x = x₂)) ↔ 
  k < -3/8 :=
sorry

end equation_two_solutions_l1266_126636


namespace sunday_to_friday_spending_ratio_l1266_126630

def friday_spending : ℝ := 20

theorem sunday_to_friday_spending_ratio :
  ∀ (sunday_multiple : ℝ),
  friday_spending + 2 * friday_spending + sunday_multiple * friday_spending = 120 →
  sunday_multiple * friday_spending / friday_spending = 3 := by
  sorry

end sunday_to_friday_spending_ratio_l1266_126630


namespace game_result_l1266_126667

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12
  else if n % 2 = 0 then 4
  else 0

def allie_rolls : List ℕ := [2, 6, 3, 1, 6]
def betty_rolls : List ℕ := [4, 6, 3, 5]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result : 
  total_points allie_rolls * total_points betty_rolls = 1120 := by
  sorry

end game_result_l1266_126667


namespace inequality_solution_range_l1266_126633

theorem inequality_solution_range (d : ℝ) :
  (∃ x : ℝ, |2017 - x| + |2018 - x| ≤ d) → d ≥ 1 := by
  sorry

end inequality_solution_range_l1266_126633


namespace pie_eating_contest_l1266_126652

theorem pie_eating_contest (first_student second_student : ℚ) :
  first_student = 7/8 ∧ second_student = 5/6 →
  first_student - second_student = 1/24 := by
  sorry

end pie_eating_contest_l1266_126652


namespace product_equals_sqrt_ratio_l1266_126695

theorem product_equals_sqrt_ratio (a b c : ℝ) :
  a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1) →
  6 * 15 * 7 = (3/2 : ℝ) := by
sorry

end product_equals_sqrt_ratio_l1266_126695


namespace milk_pour_problem_l1266_126610

theorem milk_pour_problem (initial_milk : ℚ) (pour_fraction : ℚ) :
  initial_milk = 3/8 →
  pour_fraction = 5/6 →
  pour_fraction * initial_milk = 5/16 := by
sorry

end milk_pour_problem_l1266_126610


namespace f_properties_l1266_126617

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt ((1 - x^2) / (1 + x^2)) + a * Real.sqrt ((1 + x^2) / (1 - x^2))

theorem f_properties (a : ℝ) (h : a > 0) :
  -- Function domain
  ∀ x : ℝ, -1 < x ∧ x < 1 →
  -- 1. Minimum value when a = 1
  (a = 1 → ∀ x : ℝ, -1 < x ∧ x < 1 → f 1 x ≥ 2) ∧
  -- 2. Monotonicity when a = 1
  (a = 1 → ∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y < 1 → f 1 x < f 1 y) ∧
  -- 3. Range of a for triangle formation
  (∀ r s t : ℝ, -2*Real.sqrt 5/5 ≤ r ∧ r ≤ 2*Real.sqrt 5/5 ∧
                -2*Real.sqrt 5/5 ≤ s ∧ s ≤ 2*Real.sqrt 5/5 ∧
                -2*Real.sqrt 5/5 ≤ t ∧ t ≤ 2*Real.sqrt 5/5 →
    f a r + f a s > f a t ∧ f a s + f a t > f a r ∧ f a t + f a r > f a s) ↔
  (1/15 < a ∧ a < 5/3) := by
  sorry

end f_properties_l1266_126617


namespace function_property_l1266_126634

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_property (h : ∀ x : ℝ, f (Real.exp x) = x + 2) :
  (f 1 = 2) ∧ (∀ x : ℝ, x > 0 → f x = Real.log x + 2) := by sorry

end function_property_l1266_126634


namespace movie_ticket_distribution_l1266_126685

/-- The number of ways to distribute distinct objects to distinct recipients --/
def distribute_distinct (n_objects : ℕ) (n_recipients : ℕ) : ℕ :=
  (n_recipients - n_objects + 1).factorial / (n_recipients - n_objects).factorial

/-- The number of ways to distribute 3 different movie tickets among 10 people --/
theorem movie_ticket_distribution :
  distribute_distinct 3 10 = 720 := by
  sorry

end movie_ticket_distribution_l1266_126685


namespace length_BI_isosceles_triangle_l1266_126631

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Isosceles condition
  isIsosceles : ab > 0 ∧ bc > 0

/-- The incenter of a triangle -/
def incenter (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Length of BI in isosceles triangle ABC -/
theorem length_BI_isosceles_triangle (t : IsoscelesTriangle) 
  (h1 : t.ab = 6) 
  (h2 : t.bc = 8) : 
  ∃ (ε : ℝ), abs (distance (0, 0) (incenter t) - 4.4 * Real.sqrt 1.1) < ε :=
sorry

end length_BI_isosceles_triangle_l1266_126631


namespace sum_in_base7_l1266_126628

/-- Converts a list of digits in base 7 to a natural number -/
def fromBase7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a natural number to a list of digits in base 7 -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem sum_in_base7 :
  fromBase7 [2, 4, 5] + fromBase7 [5, 4, 3] = fromBase7 [1, 1, 2, 1] :=
by sorry

end sum_in_base7_l1266_126628


namespace simplified_fraction_sum_l1266_126658

theorem simplified_fraction_sum (a b : ℕ) (h : a = 75 ∧ b = 180) :
  let g := Nat.gcd a b
  (a / g) + (b / g) = 17 := by
  sorry

end simplified_fraction_sum_l1266_126658


namespace student_competition_theorem_l1266_126689

/-- The number of ways students can sign up for competitions -/
def signup_ways (num_students : ℕ) (num_competitions : ℕ) : ℕ :=
  num_competitions ^ num_students

/-- The number of possible outcomes for championship winners -/
def championship_outcomes (num_students : ℕ) (num_competitions : ℕ) : ℕ :=
  num_students ^ num_competitions

/-- Theorem stating the correct number of ways for signup and championship outcomes -/
theorem student_competition_theorem :
  let num_students : ℕ := 5
  let num_competitions : ℕ := 4
  signup_ways num_students num_competitions = 4^5 ∧
  championship_outcomes num_students num_competitions = 5^4 := by
  sorry

end student_competition_theorem_l1266_126689
