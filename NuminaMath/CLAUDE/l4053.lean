import Mathlib

namespace NUMINAMATH_CALUDE_gcd_13m_plus_4_7m_plus_2_max_l4053_405375

theorem gcd_13m_plus_4_7m_plus_2_max (m : ℕ+) : 
  (Nat.gcd (13 * m.val + 4) (7 * m.val + 2) ≤ 2) ∧ 
  (∃ m : ℕ+, Nat.gcd (13 * m.val + 4) (7 * m.val + 2) = 2) :=
by sorry

end NUMINAMATH_CALUDE_gcd_13m_plus_4_7m_plus_2_max_l4053_405375


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l4053_405316

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 100) :
  x^2 + y^2 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l4053_405316


namespace NUMINAMATH_CALUDE_exists_grade_to_move_l4053_405396

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

theorem exists_grade_to_move :
  ∃ g ∈ group1,
    average (group1.filter (· ≠ g)) > average group1 ∧
    average (g :: group2) > average group2 := by
  sorry

end NUMINAMATH_CALUDE_exists_grade_to_move_l4053_405396


namespace NUMINAMATH_CALUDE_minimum_votes_to_win_l4053_405338

theorem minimum_votes_to_win (total_votes remaining_votes : ℕ)
  (a_votes b_votes c_votes : ℕ) (h1 : total_votes = 1500)
  (h2 : remaining_votes = 500) (h3 : a_votes + b_votes + c_votes = 1000)
  (h4 : a_votes = 350) (h5 : b_votes = 370) (h6 : c_votes = 280) :
  (∀ x : ℕ, x < 261 → 
    ∃ y : ℕ, y ≤ remaining_votes - x ∧ 
      a_votes + x ≤ b_votes + y) ∧
  (∃ z : ℕ, z = 261 ∧ 
    ∀ y : ℕ, y ≤ remaining_votes - z → 
      a_votes + z > b_votes + y) :=
by sorry

end NUMINAMATH_CALUDE_minimum_votes_to_win_l4053_405338


namespace NUMINAMATH_CALUDE_parallelogram_with_equilateral_triangles_l4053_405342

-- Define the points
variable (A B C D P Q : ℝ × ℝ)

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : ℝ × ℝ) : Prop :=
  B.1 - A.1 = D.1 - C.1 ∧ B.2 - A.2 = D.2 - C.2 ∧
  A.1 - D.1 = B.1 - C.1 ∧ A.2 - D.2 = B.2 - C.2

-- Define an equilateral triangle
def is_equilateral_triangle (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 ∧
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = (Z.1 - X.1)^2 + (Z.2 - X.2)^2

-- State the theorem
theorem parallelogram_with_equilateral_triangles
  (h1 : is_parallelogram A B C D)
  (h2 : is_equilateral_triangle B C P)
  (h3 : is_equilateral_triangle C D Q) :
  is_equilateral_triangle A P Q :=
sorry

end NUMINAMATH_CALUDE_parallelogram_with_equilateral_triangles_l4053_405342


namespace NUMINAMATH_CALUDE_equivalent_statements_l4053_405329

theorem equivalent_statements :
  (∀ x : ℝ, x ≥ 0 → x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 > 0 → x < 0) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_statements_l4053_405329


namespace NUMINAMATH_CALUDE_f_properties_l4053_405339

noncomputable def f (x : ℝ) := Real.sin x - Real.cos x

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), (f x)^2 = (f (x + p))^2 ∧
    ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), (f x)^2 = (f (x + q))^2) → p ≤ q) ∧
  (∀ (x : ℝ), f (2*x - Real.pi/2) = Real.sqrt 2 * Real.sin (x/2)) ∧
  (∃ (M : ℝ), M = 1 + Real.sqrt 3 / 2 ∧
    ∀ (x : ℝ), (f x + Real.cos x) * (Real.sqrt 3 * Real.sin x + Real.cos x) ≤ M ∧
    ∃ (x₀ : ℝ), (f x₀ + Real.cos x₀) * (Real.sqrt 3 * Real.sin x₀ + Real.cos x₀) = M) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l4053_405339


namespace NUMINAMATH_CALUDE_cost_price_calculation_l4053_405333

theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  marked_price = 65 ∧ 
  discount_rate = 0.05 ∧ 
  profit_rate = 0.30 →
  ∃ (cost_price : ℝ),
    cost_price = 47.50 ∧
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l4053_405333


namespace NUMINAMATH_CALUDE_solve_equation_l4053_405383

theorem solve_equation (x : ℝ) (h : 5*x - 8 = 15*x + 14) : 6*(x + 3) = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4053_405383


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l4053_405331

theorem right_triangle_perimeter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_area : (1/2) * a * b = 150) (h_leg : a = 30) : 
  a + b + c = 40 + 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l4053_405331


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l4053_405397

theorem repeating_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) + (6 / 9999 : ℚ) = 3793 / 9999 := by
  sorry

#check repeating_decimal_sum

end NUMINAMATH_CALUDE_repeating_decimal_sum_l4053_405397


namespace NUMINAMATH_CALUDE_linda_savings_proof_l4053_405337

def linda_savings (total : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : Prop :=
  furniture_fraction = 3/4 ∧ 
  (1 - furniture_fraction) * total = tv_cost ∧
  tv_cost = 450

theorem linda_savings_proof (total : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) :
  linda_savings total furniture_fraction tv_cost → total = 1800 := by
  sorry

end NUMINAMATH_CALUDE_linda_savings_proof_l4053_405337


namespace NUMINAMATH_CALUDE_triangle_median_inequality_l4053_405388

/-- Given a triangle with sides a, b, and c, and s_c as the length of the median to side c,
    this theorem proves the inequality relating these measurements. -/
theorem triangle_median_inequality (a b c s_c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hs_c : 0 < s_c)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
    (h_median : 2 * s_c^2 = (2 * a^2 + 2 * b^2 - c^2) / 4) :
    (c^2 - (a - b)^2) / (2 * (a + b)) ≤ a + b - 2 * s_c ∧ 
    a + b - 2 * s_c < (c^2 + (a - b)^2) / (4 * s_c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_median_inequality_l4053_405388


namespace NUMINAMATH_CALUDE_exactly_two_approve_probability_l4053_405324

def approval_rate : ℝ := 0.6

def num_voters : ℕ := 4

def num_approving : ℕ := 2

def probability_exactly_two_approve (p : ℝ) (n : ℕ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_approve_probability :
  probability_exactly_two_approve approval_rate num_voters num_approving = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_approve_probability_l4053_405324


namespace NUMINAMATH_CALUDE_opposite_of_negative_eleven_l4053_405326

theorem opposite_of_negative_eleven : 
  ∀ x : ℤ, x + (-11) = 0 → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_eleven_l4053_405326


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l4053_405322

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ :=
  (n / 10000) * 3125 + ((n / 1000) % 10) * 625 + ((n / 100) % 10) * 125 + ((n / 10) % 10) * 25 + (n % 10) * 5

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem base_conversion_subtraction :
  base5ToBase10 52143 - base8ToBase10 4310 = 1175 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l4053_405322


namespace NUMINAMATH_CALUDE_function_symmetry_translation_l4053_405377

def symmetric_wrt_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

/-- If the graph of f(x+1) is symmetric to e^x with respect to the y-axis,
    then f(x) = e^(-(x+1)) -/
theorem function_symmetry_translation (f : ℝ → ℝ) :
  symmetric_wrt_y_axis (λ x => f (x + 1)) Real.exp →
  f = λ x => Real.exp (-(x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_symmetry_translation_l4053_405377


namespace NUMINAMATH_CALUDE_dana_saturday_hours_l4053_405317

theorem dana_saturday_hours
  (hourly_rate : ℕ)
  (friday_hours : ℕ)
  (sunday_hours : ℕ)
  (total_earnings : ℕ)
  (h1 : hourly_rate = 13)
  (h2 : friday_hours = 9)
  (h3 : sunday_hours = 3)
  (h4 : total_earnings = 286) :
  (total_earnings - (friday_hours + sunday_hours) * hourly_rate) / hourly_rate = 10 :=
by sorry

end NUMINAMATH_CALUDE_dana_saturday_hours_l4053_405317


namespace NUMINAMATH_CALUDE_equation_solution_l4053_405302

theorem equation_solution (x : ℝ) : 4 / (1 + 3/x) = 1 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4053_405302


namespace NUMINAMATH_CALUDE_kirills_height_l4053_405369

/-- Proves that Kirill's height is 49 cm given the conditions -/
theorem kirills_height (brother_height : ℕ) 
  (h1 : brother_height - 14 + brother_height = 112) : 
  brother_height - 14 = 49 := by
  sorry

#check kirills_height

end NUMINAMATH_CALUDE_kirills_height_l4053_405369


namespace NUMINAMATH_CALUDE_exterior_angle_sum_l4053_405308

theorem exterior_angle_sum (angle1 angle2 angle3 angle4 : ℝ) :
  angle1 = 100 →
  angle2 = 60 →
  angle3 = 90 →
  angle1 + angle2 + angle3 + angle4 = 360 →
  angle4 = 110 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_sum_l4053_405308


namespace NUMINAMATH_CALUDE_cube_root_abs_sqrt_squared_sum_l4053_405330

theorem cube_root_abs_sqrt_squared_sum (π : ℝ) : 
  ((-8 : ℝ) ^ (1/3 : ℝ)) + |1 - π| + (9 : ℝ).sqrt - (-1 : ℝ)^2 = π - 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_abs_sqrt_squared_sum_l4053_405330


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l4053_405321

theorem complex_magnitude_proof : 
  Complex.abs ((1 + Complex.I) / (1 - Complex.I) + Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l4053_405321


namespace NUMINAMATH_CALUDE_exam_students_count_l4053_405359

theorem exam_students_count (N : ℕ) (T : ℕ) : 
  N * 85 = T ∧
  (N - 5) * 90 = T - 300 ∧
  (N - 8) * 95 = T - 465 ∧
  (N - 15) * 100 = T - 955 →
  N = 30 := by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l4053_405359


namespace NUMINAMATH_CALUDE_mildred_blocks_l4053_405364

/-- The number of blocks Mildred found -/
def blocks_found (initial final : ℕ) : ℕ := final - initial

/-- Proof that Mildred found 84 blocks -/
theorem mildred_blocks : blocks_found 2 86 = 84 := by
  sorry

end NUMINAMATH_CALUDE_mildred_blocks_l4053_405364


namespace NUMINAMATH_CALUDE_polynomial_roots_l4053_405306

theorem polynomial_roots (a b c : ℝ) : 
  (∀ x : ℝ, x^5 + 4*x^4 + a*x = b*x^2 + 4*c ↔ x = 2 ∨ x = -2) ↔ 
  (a = -16 ∧ b = 48 ∧ c = -32) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l4053_405306


namespace NUMINAMATH_CALUDE_sum_of_integers_l4053_405353

theorem sum_of_integers (x y : ℤ) 
  (h1 : x^2 + y^2 = 290) 
  (h2 : x * y = 96) : 
  x + y = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l4053_405353


namespace NUMINAMATH_CALUDE_xiao_hong_math_probability_expected_value_X_xiao_hong_more_likely_math_noon_l4053_405312

-- Define the students
inductive Student : Type
| XiaoHong : Student
| XiaoMing : Student

-- Define the subjects
inductive Subject : Type
| Math : Subject
| Physics : Subject

-- Define the time of day
inductive TimeOfDay : Type
| Noon : TimeOfDay
| Evening : TimeOfDay

-- Define the choice of subjects for a day
structure DailyChoice :=
  (noon : Subject)
  (evening : Subject)

-- Define the probabilities for each student's choices
def choice_probability (s : Student) (dc : DailyChoice) : ℚ :=
  match s, dc with
  | Student.XiaoHong, ⟨Subject.Math, Subject.Math⟩ => 1/4
  | Student.XiaoHong, ⟨Subject.Math, Subject.Physics⟩ => 1/5
  | Student.XiaoHong, ⟨Subject.Physics, Subject.Math⟩ => 7/20
  | Student.XiaoHong, ⟨Subject.Physics, Subject.Physics⟩ => 1/10
  | Student.XiaoMing, ⟨Subject.Math, Subject.Math⟩ => 1/5
  | Student.XiaoMing, ⟨Subject.Math, Subject.Physics⟩ => 1/4
  | Student.XiaoMing, ⟨Subject.Physics, Subject.Math⟩ => 3/20
  | Student.XiaoMing, ⟨Subject.Physics, Subject.Physics⟩ => 3/10

-- Define the number of subjects chosen in a day
def subjects_chosen (s : Student) (dc : DailyChoice) : ℕ :=
  match dc with
  | ⟨Subject.Math, Subject.Math⟩ => 2
  | ⟨Subject.Math, Subject.Physics⟩ => 2
  | ⟨Subject.Physics, Subject.Math⟩ => 2
  | ⟨Subject.Physics, Subject.Physics⟩ => 2

-- Theorem 1: Probability of Xiao Hong choosing math for both noon and evening for exactly 3 out of 5 days
theorem xiao_hong_math_probability : 
  (Finset.sum (Finset.range 6) (λ k => if k = 3 then Nat.choose 5 k * (1/4)^k * (3/4)^(5-k) else 0)) = 45/512 :=
sorry

-- Theorem 2: Expected value of X
theorem expected_value_X :
  (1/100 * 0 + 33/200 * 1 + 33/40 * 2) = 363/200 :=
sorry

-- Theorem 3: Xiao Hong is more likely to choose math at noon when doing physics in the evening
theorem xiao_hong_more_likely_math_noon :
  (choice_probability Student.XiaoHong ⟨Subject.Math, Subject.Physics⟩) / 
  (choice_probability Student.XiaoHong ⟨Subject.Physics, Subject.Physics⟩ + 
   choice_probability Student.XiaoHong ⟨Subject.Math, Subject.Physics⟩) >
  (choice_probability Student.XiaoMing ⟨Subject.Math, Subject.Physics⟩) / 
  (choice_probability Student.XiaoMing ⟨Subject.Physics, Subject.Physics⟩ + 
   choice_probability Student.XiaoMing ⟨Subject.Math, Subject.Physics⟩) :=
sorry

end NUMINAMATH_CALUDE_xiao_hong_math_probability_expected_value_X_xiao_hong_more_likely_math_noon_l4053_405312


namespace NUMINAMATH_CALUDE_probability_two_pairs_one_odd_l4053_405328

def total_socks : ℕ := 12
def socks_per_color : ℕ := 3
def num_colors : ℕ := 4
def drawn_socks : ℕ := 5

def favorable_outcomes : ℕ := 324
def total_outcomes : ℕ := Nat.choose total_socks drawn_socks

theorem probability_two_pairs_one_odd (h : total_outcomes = 792) :
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 22 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_pairs_one_odd_l4053_405328


namespace NUMINAMATH_CALUDE_four_lines_theorem_l4053_405362

-- Define the type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define the type for points
def Point : Type := ℝ × ℝ

-- Define a function to check if a point is on a line
def PointOnLine (p : Point) (l : Line) : Prop := l p.1 p.2

-- Define a function to check if a point is on a circle
def PointOnCircle (p : Point) (c : Point → Prop) : Prop := c p

-- Define a function to get the intersection point of two lines
def Intersection (l1 l2 : Line) : Point := sorry

-- Define a function to get the circle passing through three points
def CircleThrough (p1 p2 p3 : Point) : Point → Prop := sorry

-- Define a function to get the point corresponding to a triple of lines
def CorrespondingPoint (l1 l2 l3 : Line) : Point := sorry

-- State the theorem
theorem four_lines_theorem 
  (l1 l2 l3 l4 : Line) 
  (p1 p2 p3 p4 : Point) 
  (c : Point → Prop) 
  (h1 : PointOnLine p1 l1) 
  (h2 : PointOnLine p2 l2) 
  (h3 : PointOnLine p3 l3) 
  (h4 : PointOnLine p4 l4) 
  (hc1 : PointOnCircle p1 c) 
  (hc2 : PointOnCircle p2 c) 
  (hc3 : PointOnCircle p3 c) 
  (hc4 : PointOnCircle p4 c) :
  ∃ (c' : Point → Prop), 
    PointOnCircle (CorrespondingPoint l2 l3 l4) c' ∧ 
    PointOnCircle (CorrespondingPoint l1 l3 l4) c' ∧ 
    PointOnCircle (CorrespondingPoint l1 l2 l4) c' ∧ 
    PointOnCircle (CorrespondingPoint l1 l2 l3) c' :=
sorry

end NUMINAMATH_CALUDE_four_lines_theorem_l4053_405362


namespace NUMINAMATH_CALUDE_problem_solution_l4053_405390

theorem problem_solution (m n : ℝ) 
  (h1 : m * n = 1)
  (h2 : m^2 + n^2 = 3)
  (h3 : m^3 + n^3 = 44 + n^4)
  (h4 : m^5 + 5 = 11) :
  m^9 + n = 38 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4053_405390


namespace NUMINAMATH_CALUDE_diagonals_bisect_in_rhombus_rectangle_square_l4053_405384

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define the property of diagonals bisecting each other
def diagonals_bisect (q : Quadrilateral) : Prop :=
  let d1_mid := ((q.vertices 0 + q.vertices 2) : ℝ × ℝ) / 2
  let d2_mid := ((q.vertices 1 + q.vertices 3) : ℝ × ℝ) / 2
  d1_mid = d2_mid

-- Define rhombus, rectangle, and square as specific types of quadrilaterals
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- State the theorem
theorem diagonals_bisect_in_rhombus_rectangle_square (q : Quadrilateral) :
  (is_rhombus q ∨ is_rectangle q ∨ is_square q) → diagonals_bisect q :=
by sorry

end NUMINAMATH_CALUDE_diagonals_bisect_in_rhombus_rectangle_square_l4053_405384


namespace NUMINAMATH_CALUDE_jiajia_clover_problem_l4053_405301

theorem jiajia_clover_problem :
  ∀ (n : ℕ),
    (3 * n + 4 = 40) →
    (n = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_jiajia_clover_problem_l4053_405301


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l4053_405354

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_proposition : 
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l4053_405354


namespace NUMINAMATH_CALUDE_train_length_l4053_405367

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : speed = 45 → time = 16 → speed * time * (1000 / 3600) = 200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l4053_405367


namespace NUMINAMATH_CALUDE_three_numbers_theorem_l4053_405372

theorem three_numbers_theorem (x y z : ℝ) 
  (h1 : (x + y + z)^2 = x^2 + y^2 + z^2)
  (h2 : x * y = z^2) :
  (x = 0 ∧ z = 0) ∨ (y = 0 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_theorem_l4053_405372


namespace NUMINAMATH_CALUDE_quotient_calculation_l4053_405311

theorem quotient_calculation (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 166)
  (h2 : divisor = 18)
  (h3 : remainder = 4)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 := by
  sorry

end NUMINAMATH_CALUDE_quotient_calculation_l4053_405311


namespace NUMINAMATH_CALUDE_widget_carton_height_l4053_405351

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents the packing configuration -/
structure PackingConfig where
  widgetsPerCarton : ℕ
  widgetsPerShippingBox : ℕ
  cartonDimensions : BoxDimensions
  shippingBoxDimensions : BoxDimensions

/-- The packing configuration for the Widget Factory -/
def widgetFactoryConfig : PackingConfig :=
  { widgetsPerCarton := 3
  , widgetsPerShippingBox := 300
  , cartonDimensions := 
    { width := 4
    , length := 4
    , height := 0  -- Unknown, to be determined
    }
  , shippingBoxDimensions := 
    { width := 20
    , length := 20
    , height := 20
    }
  }

/-- Theorem: The height of each carton in the Widget Factory configuration is 5 inches -/
theorem widget_carton_height (config : PackingConfig := widgetFactoryConfig) : 
  config.cartonDimensions.height = 5 := by
  sorry


end NUMINAMATH_CALUDE_widget_carton_height_l4053_405351


namespace NUMINAMATH_CALUDE_amy_haircut_l4053_405314

/-- Represents the hair growth problem --/
def hair_problem (initial_length : ℝ) (growth_rate : ℝ) (weeks : ℕ) (final_length : ℝ) : Prop :=
  let growth := growth_rate * weeks
  let before_cut := initial_length + growth
  before_cut - final_length = 6

/-- Theorem stating the solution to Amy's haircut problem --/
theorem amy_haircut : hair_problem 11 0.5 4 7 := by
  sorry

end NUMINAMATH_CALUDE_amy_haircut_l4053_405314


namespace NUMINAMATH_CALUDE_race_solution_l4053_405350

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  half_lap_time : ℝ

/-- Represents the race configuration -/
structure RaceConfig where
  track_length : ℝ
  α : Runner
  β : Runner
  initial_distance : ℝ
  symmetry_time : ℝ
  β_to_Q_time : ℝ
  α_to_finish_time : ℝ

/-- The theorem statement -/
theorem race_solution (config : RaceConfig) 
  (h1 : config.initial_distance = 16)
  (h2 : config.β_to_Q_time = 1 + 2/15)
  (h3 : config.α_to_finish_time = 13 + 13/15)
  (h4 : config.α.speed = config.track_length / (2 * config.α.half_lap_time))
  (h5 : config.β.speed = config.track_length / (2 * config.β.half_lap_time))
  (h6 : config.α.half_lap_time + config.symmetry_time + config.β_to_Q_time + config.α_to_finish_time = 2 * config.α.half_lap_time)
  (h7 : config.β.half_lap_time = config.α.half_lap_time + config.symmetry_time + config.β_to_Q_time)
  (h8 : config.track_length / 2 = config.α.speed * config.α.half_lap_time)
  (h9 : config.track_length / 2 = config.β.speed * config.β.half_lap_time)
  (h10 : config.α.speed * (config.β_to_Q_time + config.α_to_finish_time) = config.track_length / 2) :
  config.α.speed = 8.5 ∧ config.β.speed = 7.5 ∧ config.track_length = 272 := by
  sorry


end NUMINAMATH_CALUDE_race_solution_l4053_405350


namespace NUMINAMATH_CALUDE_trig_expression_value_l4053_405343

theorem trig_expression_value (α : Real) (h : Real.tan (α / 2) = 4) :
  (6 * Real.sin α - 7 * Real.cos α + 1) / (8 * Real.sin α + 9 * Real.cos α - 1) = -85/44 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l4053_405343


namespace NUMINAMATH_CALUDE_inverse_of_inverse_fourteen_l4053_405393

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x - 4

-- Define the inverse function of g
noncomputable def g_inv (x : ℝ) : ℝ := (x + 4) / 3

-- Theorem statement
theorem inverse_of_inverse_fourteen (h : ∀ x, g (g_inv x) = x) :
  g_inv (g_inv 14) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_inverse_fourteen_l4053_405393


namespace NUMINAMATH_CALUDE_vertex_y_coordinate_l4053_405313

/-- The y-coordinate of the vertex of the parabola y = 3x^2 - 6x + 2 is -1 -/
theorem vertex_y_coordinate (x y : ℝ) : 
  y = 3 * x^2 - 6 * x + 2 → 
  ∃ x₀, ∀ x', 3 * x'^2 - 6 * x' + 2 ≥ 3 * x₀^2 - 6 * x₀ + 2 ∧ 
            3 * x₀^2 - 6 * x₀ + 2 = -1 :=
by sorry

end NUMINAMATH_CALUDE_vertex_y_coordinate_l4053_405313


namespace NUMINAMATH_CALUDE_sphere_unique_orientation_independent_projections_l4053_405325

-- Define the type for 3D objects
inductive Object3D
  | Cube
  | RegularTetrahedron
  | RightTriangularPyramid
  | Sphere

-- Define a function to check if an object's projections are orientation-independent
def hasOrientationIndependentProjections (obj : Object3D) : Prop :=
  match obj with
  | Object3D.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_unique_orientation_independent_projections :
  ∀ (obj : Object3D), hasOrientationIndependentProjections obj ↔ obj = Object3D.Sphere :=
sorry

end NUMINAMATH_CALUDE_sphere_unique_orientation_independent_projections_l4053_405325


namespace NUMINAMATH_CALUDE_positive_correlation_groups_l4053_405361

structure Variable where
  name : String

structure VariableGroup where
  var1 : Variable
  var2 : Variable

def has_positive_correlation (group : VariableGroup) : Prop :=
  sorry

def selling_price : Variable := ⟨"selling price"⟩
def sales_volume : Variable := ⟨"sales volume"⟩
def id_number : Variable := ⟨"ID number"⟩
def math_score : Variable := ⟨"math score"⟩
def breakfast_eaters : Variable := ⟨"number of people who eat breakfast daily"⟩
def stomach_diseases : Variable := ⟨"number of people with stomach diseases"⟩
def temperature : Variable := ⟨"temperature"⟩
def cold_drink_sales : Variable := ⟨"cold drink sales volume"⟩
def ebike_weight : Variable := ⟨"weight of an electric bicycle"⟩
def electricity_consumption : Variable := ⟨"electricity consumption per kilometer"⟩

def group1 : VariableGroup := ⟨selling_price, sales_volume⟩
def group2 : VariableGroup := ⟨id_number, math_score⟩
def group3 : VariableGroup := ⟨breakfast_eaters, stomach_diseases⟩
def group4 : VariableGroup := ⟨temperature, cold_drink_sales⟩
def group5 : VariableGroup := ⟨ebike_weight, electricity_consumption⟩

theorem positive_correlation_groups :
  has_positive_correlation group4 ∧ 
  has_positive_correlation group5 ∧
  ¬has_positive_correlation group1 ∧
  ¬has_positive_correlation group2 ∧
  ¬has_positive_correlation group3 :=
by sorry

end NUMINAMATH_CALUDE_positive_correlation_groups_l4053_405361


namespace NUMINAMATH_CALUDE_truck_distance_from_start_l4053_405358

-- Define the truck's travel distances
def north_distance1 : ℝ := 20
def east_distance : ℝ := 30
def north_distance2 : ℝ := 20

-- Define the total north distance
def total_north_distance : ℝ := north_distance1 + north_distance2

-- Theorem to prove
theorem truck_distance_from_start : 
  Real.sqrt (total_north_distance ^ 2 + east_distance ^ 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_truck_distance_from_start_l4053_405358


namespace NUMINAMATH_CALUDE_quadratic_trinomial_zero_discriminant_sum_l4053_405368

/-- A quadratic trinomial can be represented as the sum of two quadratic trinomials with zero discriminants -/
theorem quadratic_trinomial_zero_discriminant_sum (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (f g : ℝ → ℝ),
    (∀ x, a * x^2 + b * x + c = f x + g x) ∧
    (∃ (a₁ b₁ c₁ : ℝ), ∀ x, f x = a₁ * x^2 + b₁ * x + c₁) ∧
    (∃ (a₂ b₂ c₂ : ℝ), ∀ x, g x = a₂ * x^2 + b₂ * x + c₂) ∧
    (b₁^2 - 4 * a₁ * c₁ = 0) ∧
    (b₂^2 - 4 * a₂ * c₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_zero_discriminant_sum_l4053_405368


namespace NUMINAMATH_CALUDE_method_of_continuous_subtraction_equiv_euclid_algorithm_l4053_405334

/-- The Method of Continuous Subtraction as used in ancient Chinese mathematics -/
def methodOfContinuousSubtraction (a b : ℕ) : ℕ :=
  sorry

/-- Euclid's algorithm for finding the greatest common divisor -/
def euclidAlgorithm (a b : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the Method of Continuous Subtraction is equivalent to Euclid's algorithm -/
theorem method_of_continuous_subtraction_equiv_euclid_algorithm :
  ∀ a b : ℕ, methodOfContinuousSubtraction a b = euclidAlgorithm a b :=
sorry

end NUMINAMATH_CALUDE_method_of_continuous_subtraction_equiv_euclid_algorithm_l4053_405334


namespace NUMINAMATH_CALUDE_seven_minus_sqrt_five_floor_l4053_405363

-- Define the integer part function
noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

-- State the theorem
theorem seven_minus_sqrt_five_floor : integerPart (7 - Real.sqrt 5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_seven_minus_sqrt_five_floor_l4053_405363


namespace NUMINAMATH_CALUDE_total_lives_calculation_l4053_405344

theorem total_lives_calculation (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) : 
  initial_players = 8 → additional_players = 2 → lives_per_player = 6 →
  (initial_players + additional_players) * lives_per_player = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_calculation_l4053_405344


namespace NUMINAMATH_CALUDE_greater_number_is_eighteen_l4053_405341

theorem greater_number_is_eighteen (x y : ℝ) 
  (sum : x + y = 30)
  (diff : x - y = 6)
  (y_lower_bound : y ≥ 10)
  (x_greater : x > y) :
  x = 18 := by
sorry

end NUMINAMATH_CALUDE_greater_number_is_eighteen_l4053_405341


namespace NUMINAMATH_CALUDE_congruence_problem_l4053_405365

theorem congruence_problem (x : ℤ) : (3 * x + 7) % 16 = 2 → (2 * x + 11) % 16 = 13 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l4053_405365


namespace NUMINAMATH_CALUDE_quadrilateral_to_square_l4053_405307

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a trapezoid -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- Function to cut a quadrilateral into two trapezoids -/
def cutQuadrilateral (q : Quadrilateral) : (Trapezoid × Trapezoid) :=
  sorry

/-- Function to check if two trapezoids can form a square -/
def canFormSquare (t1 t2 : Trapezoid) : Prop :=
  sorry

/-- Theorem stating that the quadrilateral can be cut and rearranged into a square -/
theorem quadrilateral_to_square (q : Quadrilateral) :
  ∃ (t1 t2 : Trapezoid), 
    (t1, t2) = cutQuadrilateral q ∧ 
    canFormSquare t1 t2 ∧
    ∃ (side : ℝ), side = t1.height ∧ side * side = t1.base1 * t1.height + t2.base1 * t2.height :=
  sorry

end NUMINAMATH_CALUDE_quadrilateral_to_square_l4053_405307


namespace NUMINAMATH_CALUDE_ellipse_properties_hyperbola_properties_parabola_properties_l4053_405386

/-- Ellipse properties -/
theorem ellipse_properties (x y : ℝ) :
  x^2 / 4 + y^2 = 1 →
  ∃ (a b : ℝ), a = 2 * b ∧ a > 0 ∧ b > 0 ∧
    x^2 / a^2 + y^2 / b^2 = 1 ∧
    (2 : ℝ)^2 / a^2 + 0^2 / b^2 = 1 :=
sorry

/-- Hyperbola properties -/
theorem hyperbola_properties (x y : ℝ) :
  y^2 / 20 - x^2 / 16 = 1 →
  ∃ (a b : ℝ), a = 2 * Real.sqrt 5 ∧ a > 0 ∧ b > 0 ∧
    y^2 / a^2 - x^2 / b^2 = 1 ∧
    5^2 / a^2 - 2^2 / b^2 = 1 :=
sorry

/-- Parabola properties -/
theorem parabola_properties (x y : ℝ) :
  y^2 = 4 * x →
  ∃ (p : ℝ), p > 0 ∧
    y^2 = 4 * p * x ∧
    (-2)^2 = 4 * p * 1 ∧
    (∀ (x₀ y₀ : ℝ), y₀^2 = 4 * p * x₀ → x₀ = 0 → y₀ = 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_hyperbola_properties_parabola_properties_l4053_405386


namespace NUMINAMATH_CALUDE_wheel_distance_l4053_405379

/-- The distance covered by a wheel with given radius and number of revolutions -/
theorem wheel_distance (radius : ℝ) (revolutions : ℕ) : 
  radius = Real.sqrt 157 → revolutions = 1000 → 
  2 * Real.pi * radius * revolutions = 78740 := by
  sorry

end NUMINAMATH_CALUDE_wheel_distance_l4053_405379


namespace NUMINAMATH_CALUDE_find_A_l4053_405374

theorem find_A : ∃ A : ℝ, (12 + 3) * (12 - A) = 120 ∧ A = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l4053_405374


namespace NUMINAMATH_CALUDE_tan_function_property_l4053_405318

theorem tan_function_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * (x - c)) = a * Real.tan (b * (x - c) + π)) →  -- period is π/4
  (a * Real.tan (b * (π/3 - c)) = -4) →  -- passes through (π/3, -4)
  (b * (π/4 - c) = π/2) →  -- vertical asymptote at x = π/4
  4 * a * b = 64 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_function_property_l4053_405318


namespace NUMINAMATH_CALUDE_remainder_3042_div_98_l4053_405398

theorem remainder_3042_div_98 : 3042 % 98 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3042_div_98_l4053_405398


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l4053_405357

theorem framed_painting_ratio : 
  ∀ (y : ℝ), 
    y > 0 → 
    (15 + 2*y) * (20 + 6*y) = 2 * 15 * 20 → 
    (min (15 + 2*y) (20 + 6*y)) / (max (15 + 2*y) (20 + 6*y)) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l4053_405357


namespace NUMINAMATH_CALUDE_total_new_games_is_92_l4053_405309

/-- The number of new games Katie has -/
def katie_new_games : ℕ := 84

/-- The number of new games Katie's friends have -/
def friends_new_games : ℕ := 8

/-- The total number of new games Katie and her friends have together -/
def total_new_games : ℕ := katie_new_games + friends_new_games

/-- Theorem stating that the total number of new games is 92 -/
theorem total_new_games_is_92 : total_new_games = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_new_games_is_92_l4053_405309


namespace NUMINAMATH_CALUDE_rotate_D_90_clockwise_l4053_405340

-- Define the rotation matrix for 90° clockwise rotation
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

-- Define the original point D
def D : ℝ × ℝ := (-2, 3)

-- Theorem to prove
theorem rotate_D_90_clockwise :
  rotate90Clockwise D = (3, 2) := by
  sorry

end NUMINAMATH_CALUDE_rotate_D_90_clockwise_l4053_405340


namespace NUMINAMATH_CALUDE_valid_numbers_count_and_max_l4053_405399

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b x y : ℕ),
    is_prime x ∧ is_prime y ∧
    n = 4000 + 100 * a + 10 * b + 5 ∧
    n = 5 * x * 11 * y

theorem valid_numbers_count_and_max :
  (∃! (s : Finset ℕ),
    (∀ n ∈ s, is_valid_number n) ∧
    (∀ n, is_valid_number n → n ∈ s) ∧
    s.card = 3) ∧
  (∃ m : ℕ, is_valid_number m ∧ ∀ n, is_valid_number n → n ≤ m) ∧
  (∃ m : ℕ, is_valid_number m ∧ m = 4785) :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_count_and_max_l4053_405399


namespace NUMINAMATH_CALUDE_point_movement_l4053_405332

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Move a point left by a given number of units -/
def moveLeft (p : Point) (units : ℝ) : Point :=
  ⟨p.x - units, p.y⟩

/-- Move a point up by a given number of units -/
def moveUp (p : Point) (units : ℝ) : Point :=
  ⟨p.x, p.y + units⟩

theorem point_movement :
  let A : Point := ⟨2, -1⟩
  let B : Point := moveUp (moveLeft A 3) 4
  B.x = -1 ∧ B.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_l4053_405332


namespace NUMINAMATH_CALUDE_a_divides_iff_k_divides_l4053_405305

/-- Definition of a_n as the integer consisting of n repetitions of the digit 1 in base 10 -/
def a (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Theorem stating that a_k divides a_l if and only if k divides l -/
theorem a_divides_iff_k_divides (k l : ℕ) (h : k ≥ 1) :
  (a k ∣ a l) ↔ k ∣ l :=
by sorry

end NUMINAMATH_CALUDE_a_divides_iff_k_divides_l4053_405305


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l4053_405371

/-- Represents a participant's scores in a two-day math competition -/
structure Participant where
  day1_score : ℕ
  day1_attempted : ℕ
  day2_score : ℕ
  day2_attempted : ℕ

/-- The maximum possible two-day success ratio for Delta given the competition conditions -/
theorem delta_max_success_ratio 
  (gamma : Participant)
  (total_points : ℕ)
  (h_total : gamma.day1_attempted + gamma.day2_attempted = total_points)
  (h_gamma_day1 : gamma.day1_score = 210 ∧ gamma.day1_attempted = 360)
  (h_gamma_day2 : gamma.day2_score = 150 ∧ gamma.day2_attempted = 240)
  (h_gamma_ratio : (gamma.day1_score + gamma.day2_score : ℚ) / total_points = 3/5) :
  ∃ (delta : Participant),
    (delta.day1_attempted + delta.day2_attempted = total_points) ∧
    (delta.day1_attempted ≠ gamma.day1_attempted) ∧
    (delta.day1_score > 0 ∧ delta.day2_score > 0) ∧
    ((delta.day1_score : ℚ) / delta.day1_attempted < (gamma.day1_score : ℚ) / gamma.day1_attempted) ∧
    ((delta.day2_score : ℚ) / delta.day2_attempted < (gamma.day2_score : ℚ) / gamma.day2_attempted) ∧
    ((delta.day1_score + delta.day2_score : ℚ) / total_points ≤ 1/4) ∧
    ∀ (delta' : Participant),
      (delta'.day1_attempted + delta'.day2_attempted = total_points) →
      (delta'.day1_attempted ≠ gamma.day1_attempted) →
      (delta'.day1_score > 0 ∧ delta'.day2_score > 0) →
      ((delta'.day1_score : ℚ) / delta'.day1_attempted < (gamma.day1_score : ℚ) / gamma.day1_attempted) →
      ((delta'.day2_score : ℚ) / delta'.day2_attempted < (gamma.day2_score : ℚ) / gamma.day2_attempted) →
      ((delta'.day1_score + delta'.day2_score : ℚ) / total_points ≤ (delta.day1_score + delta.day2_score : ℚ) / total_points) := by
  sorry


end NUMINAMATH_CALUDE_delta_max_success_ratio_l4053_405371


namespace NUMINAMATH_CALUDE_june_maths_books_l4053_405355

/-- The number of maths books June bought -/
def num_maths_books : ℕ := sorry

/-- The total amount June has for school supplies -/
def total_amount : ℕ := 500

/-- The cost of each maths book -/
def maths_book_cost : ℕ := 20

/-- The cost of each science book -/
def science_book_cost : ℕ := 10

/-- The cost of each art book -/
def art_book_cost : ℕ := 20

/-- The amount spent on music books -/
def music_books_cost : ℕ := 160

/-- The total cost of all books -/
def total_cost : ℕ := 
  maths_book_cost * num_maths_books + 
  science_book_cost * (num_maths_books + 6) + 
  art_book_cost * (2 * num_maths_books) + 
  music_books_cost

theorem june_maths_books : 
  num_maths_books = 4 ∧ total_cost = total_amount :=
sorry

end NUMINAMATH_CALUDE_june_maths_books_l4053_405355


namespace NUMINAMATH_CALUDE_not_prime_3999991_l4053_405320

theorem not_prime_3999991 : ¬ Nat.Prime 3999991 :=
  sorry

end NUMINAMATH_CALUDE_not_prime_3999991_l4053_405320


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4053_405356

theorem complex_equation_solution (x : ℝ) :
  (x - 2 * Complex.I) * Complex.I = 2 + Complex.I → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4053_405356


namespace NUMINAMATH_CALUDE_negation_equivalence_l4053_405335

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4053_405335


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l4053_405319

/-- Proves that the interest rate is 8% per annum given the conditions of the problem -/
theorem interest_rate_calculation (P : ℝ) (t : ℝ) (I : ℝ) (r : ℝ) :
  P = 2500 →
  t = 8 →
  I = P - 900 →
  I = P * r * t / 100 →
  r = 8 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l4053_405319


namespace NUMINAMATH_CALUDE_factorization_sum_l4053_405352

theorem factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 12*x + 35 = (x + a)*(x + b)) → 
  (∀ x : ℝ, x^2 - 15*x + 56 = (x - b)*(x - c)) → 
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l4053_405352


namespace NUMINAMATH_CALUDE_set_intersection_range_l4053_405327

theorem set_intersection_range (m : ℝ) : 
  let A : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
  let B : Set ℝ := {x | x^2 - 7*x + 10 ≤ 0}
  A ∩ B = A → 2 ≤ m ∧ m ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_set_intersection_range_l4053_405327


namespace NUMINAMATH_CALUDE_function_derivative_at_zero_l4053_405394

theorem function_derivative_at_zero 
  (f : ℝ → ℝ) 
  (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f x = x^2 + 2 * (deriv f 2) * x + 3) : 
  deriv f 0 = -8 := by
sorry

end NUMINAMATH_CALUDE_function_derivative_at_zero_l4053_405394


namespace NUMINAMATH_CALUDE_subject_score_proof_l4053_405373

theorem subject_score_proof (physics chemistry mathematics : ℕ) : 
  (physics + chemistry + mathematics) / 3 = 85 →
  (physics + mathematics) / 2 = 90 →
  (physics + chemistry) / 2 = 70 →
  physics = 65 →
  mathematics = 115 := by
sorry

end NUMINAMATH_CALUDE_subject_score_proof_l4053_405373


namespace NUMINAMATH_CALUDE_derivative_f_at_negative_one_l4053_405380

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 1

-- State the theorem
theorem derivative_f_at_negative_one :
  deriv f (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_negative_one_l4053_405380


namespace NUMINAMATH_CALUDE_total_pets_is_415_l4053_405304

/-- The number of dogs at the farm -/
def num_dogs : ℕ := 43

/-- The number of fish at the farm -/
def num_fish : ℕ := 72

/-- The number of cats at the farm -/
def num_cats : ℕ := 34

/-- The number of chickens at the farm -/
def num_chickens : ℕ := 120

/-- The number of rabbits at the farm -/
def num_rabbits : ℕ := 57

/-- The number of parrots at the farm -/
def num_parrots : ℕ := 89

/-- The total number of pets at the farm -/
def total_pets : ℕ := num_dogs + num_fish + num_cats + num_chickens + num_rabbits + num_parrots

theorem total_pets_is_415 : total_pets = 415 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_is_415_l4053_405304


namespace NUMINAMATH_CALUDE_fraction_proof_l4053_405385

theorem fraction_proof (w x y F : ℝ) 
  (h1 : 5 / w + F = 5 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  F = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_proof_l4053_405385


namespace NUMINAMATH_CALUDE_percent_of_percent_equality_l4053_405391

theorem percent_of_percent_equality (y : ℝ) (h : y ≠ 0) :
  (18 / 100) * y = (30 / 100) * ((60 / 100) * y) := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_equality_l4053_405391


namespace NUMINAMATH_CALUDE_line_through_points_with_45_degree_angle_l4053_405381

/-- Given a line passing through points (-1, 3) and (2, a) with an inclination angle of 45°, prove that a = 6 -/
theorem line_through_points_with_45_degree_angle (a : ℝ) : 
  (∃ (line : ℝ → ℝ), 
    line (-1) = 3 ∧ 
    line 2 = a ∧ 
    (∀ x y : ℝ, y = line x → (y - 3) / (x - (-1)) = 1)) → 
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_line_through_points_with_45_degree_angle_l4053_405381


namespace NUMINAMATH_CALUDE_time_after_2500_minutes_l4053_405389

-- Define a custom datetime type
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

def startDateTime : DateTime :=
  { year := 2011, month := 1, day := 1, hour := 0, minute := 0 }

def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry  -- Implementation details omitted

theorem time_after_2500_minutes :
  addMinutes startDateTime 2500 =
    { year := 2011, month := 1, day := 2, hour := 17, minute := 40 } :=
by sorry

end NUMINAMATH_CALUDE_time_after_2500_minutes_l4053_405389


namespace NUMINAMATH_CALUDE_game_lives_distribution_l4053_405345

/-- Given a game with initial players, players who quit, and total lives among remaining players,
    calculates the number of lives each remaining player has. -/
def lives_per_player (initial_players quitters total_lives : ℕ) : ℕ :=
  total_lives / (initial_players - quitters)

/-- Theorem stating that in a game with 13 initial players, 8 quitters, and 30 total lives,
    each remaining player has 6 lives. -/
theorem game_lives_distribution :
  lives_per_player 13 8 30 = 6 := by
  sorry


end NUMINAMATH_CALUDE_game_lives_distribution_l4053_405345


namespace NUMINAMATH_CALUDE_derivative_ln_plus_reciprocal_l4053_405395

theorem derivative_ln_plus_reciprocal (x : ℝ) (hx : x > 0) :
  deriv (λ x => Real.log x + x⁻¹) x = (x - 1) / x^2 := by sorry

end NUMINAMATH_CALUDE_derivative_ln_plus_reciprocal_l4053_405395


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l4053_405387

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l4053_405387


namespace NUMINAMATH_CALUDE_remainder_theorem_remainder_problem_l4053_405370

/-- A polynomial of the form Dx^4 + Ex^2 + Fx - 2 -/
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x - 2

/-- The remainder theorem -/
theorem remainder_theorem {p : ℝ → ℝ} {a r : ℝ} :
  (∃ q : ℝ → ℝ, ∀ x, p x = (x - a) * q x + r) ↔ p a = r :=
sorry

theorem remainder_problem (D E F : ℝ) :
  (∃ r : ℝ, ∀ x, q D E F x = (x - 2) * r + 14) →
  (∃ s : ℝ, ∀ x, q D E F x = (x + 2) * s - 18) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_remainder_problem_l4053_405370


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l4053_405348

theorem unique_two_digit_number (t : ℕ) : 
  (10 ≤ t ∧ t < 100) ∧ (13 * t ≡ 42 [ZMOD 100]) ↔ t = 34 :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l4053_405348


namespace NUMINAMATH_CALUDE_original_laborers_l4053_405323

/-- Given a piece of work that can be completed by x laborers in 15 days,
    if 5 laborers are absent and the remaining laborers complete the work in 20 days,
    then x = 20. -/
theorem original_laborers (x : ℕ) 
  (h1 : x * 15 = (x - 5) * 20) : x = 20 := by
  sorry

#check original_laborers

end NUMINAMATH_CALUDE_original_laborers_l4053_405323


namespace NUMINAMATH_CALUDE_student_count_l4053_405303

theorem student_count : ℕ :=
  let avg_age : ℕ := 20
  let group1_count : ℕ := 5
  let group1_avg : ℕ := 14
  let group2_count : ℕ := 9
  let group2_avg : ℕ := 16
  let last_student_age : ℕ := 186
  let total_students : ℕ := group1_count + group2_count + 1
  let total_age : ℕ := group1_count * group1_avg + group2_count * group2_avg + last_student_age
  have h1 : avg_age * total_students = total_age := by sorry
  20

end NUMINAMATH_CALUDE_student_count_l4053_405303


namespace NUMINAMATH_CALUDE_no_squares_in_sequence_l4053_405300

def a : ℕ → ℤ
  | 0 => 91
  | n + 1 => 10 * a n + (-1) ^ n

theorem no_squares_in_sequence : ∀ n : ℕ, ¬∃ m : ℤ, a n = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_squares_in_sequence_l4053_405300


namespace NUMINAMATH_CALUDE_decagon_diagonals_l4053_405310

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals :
  num_diagonals decagon_sides = 35 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l4053_405310


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l4053_405366

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (5 : ℚ) / 8 ∧ 
  (∀ p' q' : ℕ+, (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (5 : ℚ) / 8 → q' ≥ q) →
  p + q = 21 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l4053_405366


namespace NUMINAMATH_CALUDE_path_length_of_rotating_triangle_l4053_405392

/-- Represents a square with side length 4 inches -/
def Square := {s : ℝ // s = 4}

/-- Represents an equilateral triangle with side length 2 inches -/
def EquilateralTriangle := {t : ℝ // t = 2}

/-- Calculates the path length of vertex P during rotations -/
noncomputable def pathLength (square : Square) (triangle : EquilateralTriangle) : ℝ :=
  sorry

/-- Theorem stating the path length of vertex P -/
theorem path_length_of_rotating_triangle 
  (square : Square) 
  (triangle : EquilateralTriangle) : 
  pathLength square triangle = (40 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_path_length_of_rotating_triangle_l4053_405392


namespace NUMINAMATH_CALUDE_arrangement_equality_l4053_405382

theorem arrangement_equality (n : ℕ) (r₁ r₂ c₁ c₂ : ℕ) 
  (h₁ : n = r₁ * c₁)
  (h₂ : n = r₂ * c₂)
  (h₃ : n = 48)
  (h₄ : r₁ = 6)
  (h₅ : c₁ = 8)
  (h₆ : r₂ = 2)
  (h₇ : c₂ = 24) :
  Nat.factorial n = Nat.factorial n :=
by sorry

end NUMINAMATH_CALUDE_arrangement_equality_l4053_405382


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l4053_405360

/-- The distance between the vertices of a hyperbola with equation x²/144 - y²/49 = 1 is 24 -/
theorem hyperbola_vertex_distance : 
  ∀ (x y : ℝ), x^2/144 - y^2/49 = 1 → ∃ (d : ℝ), d = 24 ∧ d = 2 * (Real.sqrt 144) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l4053_405360


namespace NUMINAMATH_CALUDE_worm_coverage_l4053_405378

/-- A continuous curve in the plane -/
def ContinuousCurve := Set (ℝ × ℝ)

/-- The length of a continuous curve -/
noncomputable def length (γ : ContinuousCurve) : ℝ := sorry

/-- A semicircle in the plane -/
def Semicircle (center : ℝ × ℝ) (diameter : ℝ) : Set (ℝ × ℝ) := sorry

/-- Whether a set covers another set -/
def covers (A B : Set (ℝ × ℝ)) : Prop := B ⊆ A

theorem worm_coverage (γ : ContinuousCurve) (h : length γ = 1) :
  ∃ (center : ℝ × ℝ), covers (Semicircle center 1) γ := by sorry

end NUMINAMATH_CALUDE_worm_coverage_l4053_405378


namespace NUMINAMATH_CALUDE_max_intersections_three_lines_circle_l4053_405376

/-- A line in a 2D plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A circle in a 2D plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- The number of intersection points between a line and a circle -/
def line_circle_intersections (l : Line) (c : Circle) : ℕ := sorry

/-- The number of intersection points between two lines -/
def line_line_intersections (l1 l2 : Line) : ℕ := sorry

/-- Three distinct lines -/
def three_distinct_lines : Prop :=
  ∃ (l1 l2 l3 : Line), l1 ≠ l2 ∧ l1 ≠ l3 ∧ l2 ≠ l3

theorem max_intersections_three_lines_circle :
  ∀ (l1 l2 l3 : Line) (c : Circle),
  three_distinct_lines →
  (line_circle_intersections l1 c +
   line_circle_intersections l2 c +
   line_circle_intersections l3 c +
   line_line_intersections l1 l2 +
   line_line_intersections l1 l3 +
   line_line_intersections l2 l3) ≤ 9 ∧
  ∃ (l1' l2' l3' : Line) (c' : Circle),
    three_distinct_lines →
    (line_circle_intersections l1' c' +
     line_circle_intersections l2' c' +
     line_circle_intersections l3' c' +
     line_line_intersections l1' l2' +
     line_line_intersections l1' l3' +
     line_line_intersections l2' l3') = 9 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_three_lines_circle_l4053_405376


namespace NUMINAMATH_CALUDE_model_car_velocities_l4053_405347

/-- A model car on a closed circuit -/
structure ModelCar where
  circuit_length : ℕ
  uphill_length : ℕ
  flat_length : ℕ
  downhill_length : ℕ
  vs : ℕ  -- uphill velocity
  vp : ℕ  -- flat velocity
  vd : ℕ  -- downhill velocity

/-- The conditions of the problem -/
def satisfies_conditions (car : ModelCar) : Prop :=
  car.circuit_length = 600 ∧
  car.uphill_length = car.downhill_length ∧
  car.uphill_length + car.flat_length + car.downhill_length = car.circuit_length ∧
  car.vs < car.vp ∧ car.vp < car.vd ∧
  (car.uphill_length / car.vs + car.flat_length / car.vp + car.downhill_length / car.vd : ℚ) = 50

/-- The theorem to prove -/
theorem model_car_velocities (car : ModelCar) :
  satisfies_conditions car →
  ((car.vs = 7 ∧ car.vp = 12 ∧ car.vd = 42) ∨
   (car.vs = 8 ∧ car.vp = 12 ∧ car.vd = 24) ∨
   (car.vs = 9 ∧ car.vp = 12 ∧ car.vd = 18) ∨
   (car.vs = 10 ∧ car.vp = 12 ∧ car.vd = 15)) :=
by sorry

end NUMINAMATH_CALUDE_model_car_velocities_l4053_405347


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l4053_405336

def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x - f (x + y) = f (x^2 * f y + x)

theorem functional_equation_solutions :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x ≥ 0) →
  FunctionalEquation f →
  (∀ x, x > 0 → f x = 0) ∨ (∀ x, x > 0 → f x = 1 / x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l4053_405336


namespace NUMINAMATH_CALUDE_inequality_proof_l4053_405346

theorem inequality_proof (x a b : ℝ) (h1 : x < a) (h2 : a < 0) (h3 : b = -a) :
  x^2 > b^2 ∧ b^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4053_405346


namespace NUMINAMATH_CALUDE_mans_usual_time_l4053_405349

/-- 
Given a man whose walking time increases by 24 minutes when his speed is reduced to 50% of his usual speed,
prove that his usual time to cover the distance is 24 minutes.
-/
theorem mans_usual_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) 
  (h2 : usual_time > 0)
  (h3 : usual_speed / (0.5 * usual_speed) = (usual_time + 24) / usual_time) : 
  usual_time = 24 := by
sorry

end NUMINAMATH_CALUDE_mans_usual_time_l4053_405349


namespace NUMINAMATH_CALUDE_combinations_count_l4053_405315

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the total value we're aiming for in cents -/
def total_value : ℕ := 30

/-- 
  Counts the number of non-negative integer solutions (p, n, d) to the equation 
  p * penny_value + n * nickel_value + d * dime_value = total_value
-/
def count_combinations : ℕ := sorry

theorem combinations_count : count_combinations = 20 := by sorry

end NUMINAMATH_CALUDE_combinations_count_l4053_405315
