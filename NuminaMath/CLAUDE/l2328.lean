import Mathlib

namespace NUMINAMATH_CALUDE_test_questions_count_l2328_232806

theorem test_questions_count : ∀ (total : ℕ), 
  (total % 5 = 0) →  -- The test has 5 equal sections
  (32 : ℚ) / total > (70 : ℚ) / 100 →  -- Percentage of correct answers > 70%
  (32 : ℚ) / total < (77 : ℚ) / 100 →  -- Percentage of correct answers < 77%
  total = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_test_questions_count_l2328_232806


namespace NUMINAMATH_CALUDE_meaningful_expression_l2328_232837

theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x = (Real.sqrt (a + 3)) / (a - 1)) ↔ (a ≥ -3 ∧ a ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2328_232837


namespace NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_angle_proof_by_contradiction_uses_correct_assumption_l2328_232873

/-- A triangle has at most one obtuse angle -/
theorem triangle_at_most_one_obtuse_angle : 
  ∀ (T : Type) (is_triangle : T → Prop) (is_obtuse_angle : T → T → Prop),
  (∀ t : T, is_triangle t → 
    ∃! a : T, is_obtuse_angle t a) :=
by
  sorry

/-- The correct assumption for proof by contradiction of the above theorem -/
def contradiction_assumption (T : Type) (is_triangle : T → Prop) (is_obtuse_angle : T → T → Prop) : Prop :=
  ∃ t : T, is_triangle t ∧ ∃ a b : T, a ≠ b ∧ is_obtuse_angle t a ∧ is_obtuse_angle t b

/-- The proof by contradiction uses the correct assumption -/
theorem proof_by_contradiction_uses_correct_assumption :
  ∀ (T : Type) (is_triangle : T → Prop) (is_obtuse_angle : T → T → Prop),
  ¬(contradiction_assumption T is_triangle is_obtuse_angle) →
  (∀ t : T, is_triangle t → 
    ∃! a : T, is_obtuse_angle t a) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_angle_proof_by_contradiction_uses_correct_assumption_l2328_232873


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2328_232812

theorem complex_fraction_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a * b ≠ a^3) :
  let sum := (a^2 - b^2) / (a * b) + (a * b + b^2) / (a * b - a^3)
  sum ≠ 1 ∧ sum ≠ (b^2 + b) / (b - a^2) ∧ sum ≠ 0 ∧ sum ≠ (a^2 + b) / (a^2 - b) :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2328_232812


namespace NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l2328_232899

/-- The polar to Cartesian conversion theorem for a specific curve -/
theorem polar_to_cartesian_conversion (ρ θ x y : ℝ) :
  (ρ * (Real.cos θ)^2 = 2 * Real.sin θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  x^2 = 2*y := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l2328_232899


namespace NUMINAMATH_CALUDE_unique_divisible_triple_l2328_232896

theorem unique_divisible_triple :
  ∃! (x y z : ℕ), 
    0 < x ∧ x < y ∧ y < z ∧
    Nat.gcd x (Nat.gcd y z) = 1 ∧
    (x + y) % z = 0 ∧
    (y + z) % x = 0 ∧
    (z + x) % y = 0 ∧
    x = 1 ∧ y = 2 ∧ z = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_triple_l2328_232896


namespace NUMINAMATH_CALUDE_intersection_points_form_geometric_sequence_l2328_232865

-- Define the curve C
def curve (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x

-- Define the line l
def line (t : ℝ) : ℝ × ℝ := (-2 + t, -4 + t)

-- Define the point P
def P : ℝ × ℝ := (-2, -4)

-- Define the property of geometric sequence for three positive real numbers
def is_geometric_sequence (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ b^2 = a*c

-- Main theorem
theorem intersection_points_form_geometric_sequence (a : ℝ) :
  a > 0 →
  ∃ t₁ t₂ : ℝ,
    let M := line t₁
    let N := line t₂
    curve a M.1 M.2 ∧
    curve a N.1 N.2 ∧
    is_geometric_sequence (Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2))
                          (Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2))
                          (Real.sqrt ((P.1 - N.1)^2 + (P.2 - N.2)^2)) →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_intersection_points_form_geometric_sequence_l2328_232865


namespace NUMINAMATH_CALUDE_a_gt_6_sufficient_not_necessary_for_a_sq_gt_36_l2328_232846

theorem a_gt_6_sufficient_not_necessary_for_a_sq_gt_36 :
  (∀ a : ℝ, a > 6 → a^2 > 36) ∧
  (∃ a : ℝ, a^2 > 36 ∧ a ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_6_sufficient_not_necessary_for_a_sq_gt_36_l2328_232846


namespace NUMINAMATH_CALUDE_smallest_b_value_l2328_232870

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 8) 
  (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  ∀ c : ℕ+, c < b → ¬(∃ d : ℕ+, d - c = 8 ∧ 
    Nat.gcd ((d^3 + c^3) / (d + c)) (d * c) = 16) :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l2328_232870


namespace NUMINAMATH_CALUDE_derivative_of_f_l2328_232862

/-- Given a function f(x) = (x^2 + 2x - 1)e^(2-x), this theorem states its derivative. -/
theorem derivative_of_f (x : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (x^2 + 2*x - 1) * Real.exp (2 - x)
  deriv f x = (3 - x^2) * Real.exp (2 - x) := by
sorry

end NUMINAMATH_CALUDE_derivative_of_f_l2328_232862


namespace NUMINAMATH_CALUDE_max_area_inscribed_isosceles_triangle_l2328_232807

/-- An isosceles triangle inscribed in a circle --/
structure InscribedIsoscelesTriangle where
  /-- The radius of the circle --/
  radius : ℝ
  /-- The height of the triangle to its base --/
  height : ℝ

/-- The area of an inscribed isosceles triangle --/
def area (t : InscribedIsoscelesTriangle) : ℝ := sorry

/-- Theorem: The area of an isosceles triangle inscribed in a circle with radius 6
    is maximized when the height to the base is 9 --/
theorem max_area_inscribed_isosceles_triangle :
  ∀ t : InscribedIsoscelesTriangle,
  t.radius = 6 →
  area t ≤ area { radius := 6, height := 9 } :=
sorry

end NUMINAMATH_CALUDE_max_area_inscribed_isosceles_triangle_l2328_232807


namespace NUMINAMATH_CALUDE_quadratic_solution_set_l2328_232884

/-- A quadratic function f(x) = ax^2 - bx + 1 -/
def f (a b x : ℝ) : ℝ := a * x^2 - b * x + 1

theorem quadratic_solution_set (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 1/4 < x ∧ x < 1/3) →
  a = 12 ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_l2328_232884


namespace NUMINAMATH_CALUDE_bobs_weight_l2328_232842

theorem bobs_weight (jim_weight bob_weight : ℝ) 
  (h1 : jim_weight + bob_weight = 200)
  (h2 : bob_weight - jim_weight = bob_weight / 3) : 
  bob_weight = 120 := by
sorry

end NUMINAMATH_CALUDE_bobs_weight_l2328_232842


namespace NUMINAMATH_CALUDE_function_properties_l2328_232883

-- Define the function f
def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- State the theorem
theorem function_properties
  (a b c : ℝ)
  (h_min : ∀ x : ℝ, f a b c x ≥ 0 ∧ ∃ y : ℝ, f a b c y = 0)
  (h_sym : ∀ x : ℝ, f a b c (x - 1) = f a b c (-x - 1))
  (h_bound : ∀ x ∈ Set.Ioo 0 5, x ≤ f a b c x ∧ f a b c x ≤ 2 * |x - 1| + 1) :
  (f a b c 1 = 1) ∧
  (∀ x : ℝ, f a b c x = (1/4) * (x + 1)^2) ∧
  (∃ m : ℝ, m > 1 ∧ 
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) ∧
    (∀ m' > m, ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x) ∧
    m = 9) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2328_232883


namespace NUMINAMATH_CALUDE_rope_for_first_post_l2328_232855

theorem rope_for_first_post (second_post third_post fourth_post total : ℕ) 
  (h1 : second_post = 20)
  (h2 : third_post = 14)
  (h3 : fourth_post = 12)
  (h4 : total = 70)
  (h5 : ∃ first_post : ℕ, first_post + second_post + third_post + fourth_post = total) :
  ∃ first_post : ℕ, first_post = 24 ∧ first_post + second_post + third_post + fourth_post = total :=
by
  sorry

end NUMINAMATH_CALUDE_rope_for_first_post_l2328_232855


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2328_232810

def A : Set ℤ := {-2, -1}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2328_232810


namespace NUMINAMATH_CALUDE_triangle_inequality_l2328_232825

/-- Theorem: For any triangle ABC, the sum of square roots of specific ratios involving side lengths, altitude, and inradius is less than or equal to 3/4. -/
theorem triangle_inequality (a b c h_a r : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_pos_h_a : 0 < h_a) (h_pos_r : 0 < r) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let f (x y z w v) := Real.sqrt (x * (w - 2 * v) / ((3 * x + y + z) * (w + 2 * v)))
  (f a b c h_a r) + (f b c a h_a r) + (f c a b h_a r) ≤ 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2328_232825


namespace NUMINAMATH_CALUDE_investment_interest_l2328_232844

/-- Calculates the compound interest earned given initial investment, interest rate, and time period. -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- The interest earned on a $2000 investment at 2% annual compound interest after 3 years is $122. -/
theorem investment_interest : 
  ∃ ε > 0, |compound_interest 2000 0.02 3 - 122| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_investment_interest_l2328_232844


namespace NUMINAMATH_CALUDE_perpendicular_to_same_line_are_parallel_l2328_232857

-- Define the concept of a line in a plane
def Line (P : Type) := P → P → Prop

-- Define the concept of a plane
variable {P : Type}

-- Define the perpendicular relation between lines
def Perpendicular (l₁ l₂ : Line P) : Prop := sorry

-- Define the parallel relation between lines
def Parallel (l₁ l₂ : Line P) : Prop := sorry

-- State the theorem
theorem perpendicular_to_same_line_are_parallel 
  (l₁ l₂ l₃ : Line P) 
  (h₁ : Perpendicular l₁ l₃) 
  (h₂ : Perpendicular l₂ l₃) : 
  Parallel l₁ l₂ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_same_line_are_parallel_l2328_232857


namespace NUMINAMATH_CALUDE_distance_between_cities_l2328_232809

/-- Proves that the distance between two cities is 300 miles given specific travel conditions -/
theorem distance_between_cities (speed_david speed_lewis : ℝ) (meeting_point : ℝ) : 
  speed_david = 50 →
  speed_lewis = 70 →
  meeting_point = 250 →
  ∃ (time : ℝ), 
    time * speed_david = meeting_point ∧
    time * speed_lewis = 2 * 300 - meeting_point →
  300 = 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l2328_232809


namespace NUMINAMATH_CALUDE_hat_problem_probabilities_q_div_p_undefined_l2328_232879

/-- The number of slips in the hat -/
def total_slips : ℕ := 42

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 14

/-- The number of slips for each number -/
def slips_per_number : ℕ := 3

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The probability of drawing four slips with the same number -/
def p : ℚ := 0

/-- The number of ways to choose two distinct numbers and two slips for each -/
def favorable_outcomes : ℕ := Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 2

/-- The probability of drawing two pairs of slips with different numbers -/
def q : ℚ := favorable_outcomes / Nat.choose total_slips drawn_slips

theorem hat_problem_probabilities :
  p = 0 ∧ q = 819 / Nat.choose total_slips drawn_slips :=
sorry

theorem q_div_p_undefined : ¬∃ (x : ℚ), q / p = x :=
sorry

end NUMINAMATH_CALUDE_hat_problem_probabilities_q_div_p_undefined_l2328_232879


namespace NUMINAMATH_CALUDE_equal_roots_condition_l2328_232877

theorem equal_roots_condition (m : ℝ) : 
  (∃! x : ℝ, (x * (x - 2) - (m + 2)) / ((x - 2) * (m - 2)) = (x + 1) / (m + 1)) ↔ 
  (m = -1 ∨ m = -5) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l2328_232877


namespace NUMINAMATH_CALUDE_min_q_value_l2328_232843

def q (a : ℕ) : ℚ :=
  (Nat.choose (42 - a) 2 + Nat.choose (a - 1) 2) / Nat.choose 50 2

theorem min_q_value (a : ℕ) :
  (∀ x, 1 ≤ x ∧ x < a → q x < 1/2) ∧ q a ≥ 1/2 → a = 7 :=
sorry

end NUMINAMATH_CALUDE_min_q_value_l2328_232843


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l2328_232867

theorem pure_imaginary_ratio (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4 * Complex.I) * (c + d * Complex.I) = y * Complex.I) : 
  c / d = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l2328_232867


namespace NUMINAMATH_CALUDE_only_solution_l2328_232886

/-- Represents the position of a person in a line of recruits. -/
structure Position :=
  (ahead : ℕ)

/-- Represents the three brothers in the line of recruits. -/
inductive Brother
| Peter
| Nikolay
| Denis

/-- Gets the initial number of people ahead of a given brother. -/
def initial_ahead (b : Brother) : ℕ :=
  match b with
  | Brother.Peter => 50
  | Brother.Nikolay => 100
  | Brother.Denis => 170

/-- Calculates the number of people in front after turning, given the total number of recruits. -/
def after_turn (n : ℕ) (b : Brother) : ℕ :=
  n - (initial_ahead b + 1)

/-- Checks if the condition after turning is satisfied for a given total number of recruits. -/
def satisfies_condition (n : ℕ) : Prop :=
  ∃ (b1 b2 : Brother), b1 ≠ b2 ∧ after_turn n b1 = 4 * after_turn n b2

/-- The theorem stating that 211 is the only solution. -/
theorem only_solution :
  ∀ n : ℕ, satisfies_condition n ↔ n = 211 :=
sorry

end NUMINAMATH_CALUDE_only_solution_l2328_232886


namespace NUMINAMATH_CALUDE_quadratic_standard_form_l2328_232834

theorem quadratic_standard_form : 
  ∃ (a b c : ℝ), ∀ x, 5 * x^2 = 6 * x - 8 ↔ a * x^2 + b * x + c = 0 ∧ a = 5 ∧ b = -6 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_standard_form_l2328_232834


namespace NUMINAMATH_CALUDE_complement_of_A_l2328_232878

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem complement_of_A (x : ℝ) : x ∈ (U \ A) ↔ x < 1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l2328_232878


namespace NUMINAMATH_CALUDE_triple_a_student_distribution_l2328_232836

theorem triple_a_student_distribution (n : ℕ) (k : ℕ) (h : n = 10 ∧ k = 3) :
  (Nat.choose (n - 1) (k - 1) : ℕ) = 36 :=
sorry

end NUMINAMATH_CALUDE_triple_a_student_distribution_l2328_232836


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2328_232817

theorem coefficient_x_squared_in_expansion :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k : ℕ) * 2^k * (if k = 2 then 1 else 0)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2328_232817


namespace NUMINAMATH_CALUDE_polynomial_sum_l2328_232853

theorem polynomial_sum (a b c d : ℝ) : 
  (fun x : ℝ => (4*x^2 - 3*x + 2)*(5 - x)) = 
  (fun x : ℝ => a*x^3 + b*x^2 + c*x + d) → 
  5*a + 3*b + 2*c + d = 25 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2328_232853


namespace NUMINAMATH_CALUDE_martha_and_john_money_l2328_232860

theorem martha_and_john_money : (5 / 8 : ℚ) + (2 / 5 : ℚ) = 1.025 := by sorry

end NUMINAMATH_CALUDE_martha_and_john_money_l2328_232860


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l2328_232851

/-- Given a parabola y² = 2px (p > 0) and a point A(m, 2√2) on it,
    if a circle centered at A with radius |AF| intersects the y-axis
    with a chord of length 2√7, then m = (2√3)/3 -/
theorem parabola_circle_intersection (p m : ℝ) (hp : p > 0) :
  2 * p * m = 8 →
  let f := (2 / m, 0)
  let r := m + 2 / m
  (r^2 - m^2 = 7) →
  m = (2 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l2328_232851


namespace NUMINAMATH_CALUDE_divisor_and_equation_solution_l2328_232864

theorem divisor_and_equation_solution :
  ∃ (k : ℕ) (base : ℕ+),
    (929260 : ℕ) % (base : ℕ)^k = 0 ∧
    3^k - k^3 = 1 ∧
    base = 17 ∧
    k = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisor_and_equation_solution_l2328_232864


namespace NUMINAMATH_CALUDE_k_of_h_10_l2328_232871

def h (x : ℝ) : ℝ := 4 * x - 5

def k (x : ℝ) : ℝ := 2 * x + 6

theorem k_of_h_10 : k (h 10) = 76 := by
  sorry

end NUMINAMATH_CALUDE_k_of_h_10_l2328_232871


namespace NUMINAMATH_CALUDE_tessa_initial_apples_l2328_232820

theorem tessa_initial_apples :
  ∀ (initial_apples : ℕ),
    (initial_apples + 5 = 9) →
    initial_apples = 4 := by
  sorry

end NUMINAMATH_CALUDE_tessa_initial_apples_l2328_232820


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2328_232830

/-- An isosceles triangle with two given side lengths -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  side1_pos : side1 > 0
  side2_pos : side2 > 0

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : Set ℝ :=
  if t.side1 = t.side2 then
    {2 * t.side1 + t.side2}
  else
    {2 * t.side1 + t.side2, t.side1 + 2 * t.side2}

theorem isosceles_triangle_perimeter :
  ∀ (t : IsoscelesTriangle),
    (t.side1 = 4 ∧ t.side2 = 6) ∨ (t.side1 = 6 ∧ t.side2 = 4) →
      perimeter t = {14, 16} ∧
    (t.side1 = 2 ∧ t.side2 = 6) ∨ (t.side1 = 6 ∧ t.side2 = 2) →
      perimeter t = {14} :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2328_232830


namespace NUMINAMATH_CALUDE_triangle_area_l2328_232875

theorem triangle_area (a c : ℝ) (B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : B = Real.pi / 3) :
  (1/2) * a * c * Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2328_232875


namespace NUMINAMATH_CALUDE_evaluate_expression_l2328_232827

theorem evaluate_expression (a b : ℤ) (ha : a = 3) (hb : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2328_232827


namespace NUMINAMATH_CALUDE_class_average_proof_l2328_232808

/-- Given a class with boys and girls, their average scores, and the ratio of boys to girls,
    prove that the overall class average is 94 points. -/
theorem class_average_proof (boys_avg : ℝ) (girls_avg : ℝ) (ratio : ℝ) :
  boys_avg = 90 →
  girls_avg = 96 →
  ratio = 0.5 →
  (ratio * girls_avg + girls_avg) / (ratio + 1) = 94 := by
  sorry

end NUMINAMATH_CALUDE_class_average_proof_l2328_232808


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2328_232819

/-- Given a line y = mx + c, if the reflection of point (-2, 0) across this line is (6, 4), then m + c = 4 -/
theorem reflection_line_sum (m c : ℝ) : 
  (∀ (x y : ℝ), y = m * x + c → 
    (x + 2) * (x - 6) + (y - 4) * (y - 0) = 0 ∧ 
    (x - 2) = m * (y - 2)) → 
  m + c = 4 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l2328_232819


namespace NUMINAMATH_CALUDE_max_xyz_value_l2328_232863

theorem max_xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y + 2 * z = (x + z) * (y + z)) :
  x * y * z ≤ 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_max_xyz_value_l2328_232863


namespace NUMINAMATH_CALUDE_ends_with_two_zeros_l2328_232874

theorem ends_with_two_zeros (x y : ℕ) :
  (x^2 + x*y + y^2) % 10 = 0 → (x^2 + x*y + y^2) % 100 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ends_with_two_zeros_l2328_232874


namespace NUMINAMATH_CALUDE_rectangle_length_l2328_232872

/-- Represents the properties of a rectangle --/
structure Rectangle where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_width_relation : length = width + 15
  perimeter_formula : perimeter = 2 * length + 2 * width

/-- Theorem stating that a rectangle with the given properties has a length of 45 cm --/
theorem rectangle_length (rect : Rectangle) (h : rect.perimeter = 150) : rect.length = 45 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l2328_232872


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2328_232895

theorem cubic_equation_solution (a b c : ℂ) : 
  (∃ (x₁ x₂ x₃ : ℂ), x₁ = 1 ∧ x₂ = 1 - Complex.I ∧ x₃ = 1 + Complex.I ∧
    (∀ (x : ℂ), x^3 + a*x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  a + b - c = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2328_232895


namespace NUMINAMATH_CALUDE_sandwich_availability_l2328_232802

/-- Given an initial number of sandwich kinds and a number of sold-out sandwich kinds,
    prove that the current number of available sandwich kinds is their difference. -/
theorem sandwich_availability (initial : ℕ) (sold_out : ℕ) (h : sold_out ≤ initial) :
  initial - sold_out = initial - sold_out :=
by sorry

end NUMINAMATH_CALUDE_sandwich_availability_l2328_232802


namespace NUMINAMATH_CALUDE_inequality_for_positive_integers_l2328_232892

theorem inequality_for_positive_integers (n : ℕ) (h : n > 0) : 2 * n - 1 < (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_positive_integers_l2328_232892


namespace NUMINAMATH_CALUDE_chocolate_difference_l2328_232801

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℚ := 3 / 7 * 70

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℚ := 120 / 100 * 40

/-- The number of chocolates Penny ate -/
def penny_chocolates : ℚ := 3 / 8 * 80

/-- The number of chocolates Dime ate -/
def dime_chocolates : ℚ := 1 / 2 * 90

/-- The difference between the number of chocolates eaten by Robert and Nickel combined
    and the number of chocolates eaten by Penny and Dime combined -/
theorem chocolate_difference :
  (robert_chocolates + nickel_chocolates) - (penny_chocolates + dime_chocolates) = -3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_difference_l2328_232801


namespace NUMINAMATH_CALUDE_inscribed_semicircle_radius_l2328_232882

/-- Given a right triangle ABC with AC = 12 and BC = 5, the radius of the inscribed semicircle is 10/3 -/
theorem inscribed_semicircle_radius (A B C : ℝ × ℝ) : 
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (d A C = 12) →
  (d B C = 5) →
  (d A B)^2 = (d A C)^2 + (d B C)^2 →
  (∃ r : ℝ, r = 10/3 ∧ 
    ∃ O : ℝ × ℝ, 
      d O A + d O B = d A B ∧
      d O C = r ∧
      ∀ P : ℝ × ℝ, d O P = r → 
        (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2 ∧
        (P.1 - A.1) * (B.2 - A.2) = (P.2 - A.2) * (B.1 - A.1)) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_semicircle_radius_l2328_232882


namespace NUMINAMATH_CALUDE_polynomial_roots_l2328_232890

def P (x : ℝ) : ℝ := x^6 - 3*x^5 - 6*x^3 - x + 8

theorem polynomial_roots :
  (∀ x < 0, P x > 0) ∧ (∃ x > 0, P x = 0) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2328_232890


namespace NUMINAMATH_CALUDE_expression_evaluation_expression_simplification_l2328_232811

-- Part 1
theorem expression_evaluation :
  Real.sqrt 2 + (1 : ℝ)^2014 + 2 * Real.cos (45 * π / 180) + Real.sqrt 16 = 2 * Real.sqrt 2 + 5 := by
  sorry

-- Part 2
theorem expression_simplification (x y : ℝ) (hx : x = Real.sqrt 2 + 1) (hy : y = Real.sqrt 2 - 1) :
  (x^2 + y^2 - 2*x*y) / (x - y) / ((x / y) - (y / x)) = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_expression_simplification_l2328_232811


namespace NUMINAMATH_CALUDE_donation_distribution_l2328_232826

/-- Proves that donating 80% of $2500 to 8 organizations results in each organization receiving $250 --/
theorem donation_distribution (total_amount : ℝ) (donation_percentage : ℝ) (num_organizations : ℕ) :
  total_amount = 2500 →
  donation_percentage = 0.8 →
  num_organizations = 8 →
  (total_amount * donation_percentage) / num_organizations = 250 := by
sorry

end NUMINAMATH_CALUDE_donation_distribution_l2328_232826


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l2328_232805

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (blueberry_fraction raspberry_fraction blackberry_fraction walnut_fraction : ℚ)
  (h_total : total_pies = 30)
  (h_blueberry : blueberry_fraction = 1/3)
  (h_raspberry : raspberry_fraction = 3/5)
  (h_blackberry : blackberry_fraction = 5/6)
  (h_walnut : walnut_fraction = 1/10) :
  ∃ (max_without_ingredients : ℕ), 
    max_without_ingredients ≤ total_pies ∧
    max_without_ingredients = total_pies - (total_pies * blackberry_fraction).floor ∧
    max_without_ingredients = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l2328_232805


namespace NUMINAMATH_CALUDE_factor_x6_minus_x4_minus_x2_plus_1_l2328_232852

theorem factor_x6_minus_x4_minus_x2_plus_1 (x : ℝ) :
  x^6 - x^4 - x^2 + 1 = (x - 1) * (x + 1) * (x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_x6_minus_x4_minus_x2_plus_1_l2328_232852


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l2328_232824

theorem sum_of_numbers_in_ratio (x : ℝ) :
  x > 0 →
  x^2 + (2*x)^2 + (5*x)^2 = 4320 →
  x + 2*x + 5*x = 96 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l2328_232824


namespace NUMINAMATH_CALUDE_f_sum_equals_three_l2328_232814

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_sum_equals_three 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_odd : is_odd_function (fun x ↦ f (x - 1))) 
  (h_f2 : f 2 = 3) : 
  f 5 + f 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_f_sum_equals_three_l2328_232814


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l2328_232840

/-- Represents the original square pattern of tiles -/
structure OriginalPattern :=
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- Represents the extended pattern after adding two layers -/
structure ExtendedPattern :=
  (black_tiles : ℕ)
  (white_tiles : ℕ)

/-- Function to extend the original pattern with two alternating layers -/
def extend_pattern (original : OriginalPattern) : ExtendedPattern :=
  { black_tiles := original.black_tiles + 24,
    white_tiles := original.white_tiles + 32 }

/-- Theorem stating the ratio of black to white tiles in the extended pattern -/
theorem extended_pattern_ratio 
  (original : OriginalPattern) 
  (h1 : original.black_tiles = 9) 
  (h2 : original.white_tiles = 16) :
  let extended := extend_pattern original
  (extended.black_tiles : ℚ) / extended.white_tiles = 33 / 48 := by
  sorry

#check extended_pattern_ratio

end NUMINAMATH_CALUDE_extended_pattern_ratio_l2328_232840


namespace NUMINAMATH_CALUDE_savings_percentage_approx_l2328_232816

def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 5650
def savings : ℕ := 2350

def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous
def total_salary : ℕ := total_expenses + savings

def percentage_saved : ℚ := (savings : ℚ) / (total_salary : ℚ) * 100

theorem savings_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ abs (percentage_saved - 8.87) < ε :=
sorry

end NUMINAMATH_CALUDE_savings_percentage_approx_l2328_232816


namespace NUMINAMATH_CALUDE_value_of_c_l2328_232854

theorem value_of_c (a b c : ℝ) : 
  12 = 0.06 * a → 
  6 = 0.12 * b → 
  c = b / a → 
  c = 0.25 := by
sorry

end NUMINAMATH_CALUDE_value_of_c_l2328_232854


namespace NUMINAMATH_CALUDE_planes_perpendicular_l2328_232848

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Theorem statement
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_parallel_α : line_parallel_plane m α)
  (h_n_perp_β : line_perpendicular_plane n β)
  (h_m_parallel_n : parallel m n) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l2328_232848


namespace NUMINAMATH_CALUDE_complex_number_in_quadrant_IV_l2328_232815

/-- The complex number (1+i)/(1+2i) lies in Quadrant IV of the complex plane -/
theorem complex_number_in_quadrant_IV : 
  let z : ℂ := (1 + Complex.I) / (1 + 2 * Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_quadrant_IV_l2328_232815


namespace NUMINAMATH_CALUDE_probability_even_product_l2328_232858

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def is_even_product (a b : ℕ) : Prop := Even (a * b)

theorem probability_even_product :
  Nat.card {p : S × S | p.1 ≠ p.2 ∧ is_even_product p.1 p.2} / Nat.choose 7 2 = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_product_l2328_232858


namespace NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l2328_232813

theorem coefficient_x5_in_expansion : 
  (Finset.range 61).sum (fun k => Nat.choose 60 k * (1 : ℕ)^(60 - k) * (1 : ℕ)^k) = 2^60 ∧ 
  Nat.choose 60 5 = 446040 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l2328_232813


namespace NUMINAMATH_CALUDE_angle_relation_l2328_232897

-- Define the structure for a point in the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the structure for a triangle
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Define the structure for a quadrilateral
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

-- Define a function to calculate the angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Define a function to check if two triangles are anti-similar
def antiSimilar (t1 t2 : Triangle) : Prop := sorry

-- Define a function to check if a quadrilateral is convex
def isConvex (q : Quadrilateral) : Prop := sorry

-- Define a function to find the intersection of perpendicular bisectors
def perpendicularBisectorIntersection (A B C D : Point) : Point := sorry

-- Main theorem
theorem angle_relation 
  (A B C D X : Point)
  (Y : Point := perpendicularBisectorIntersection A B C D)
  (h1 : antiSimilar ⟨B, X, C⟩ ⟨A, X, D⟩)
  (h2 : isConvex ⟨A, B, C, D⟩)
  (h3 : angle A D X = angle B C X)
  (h4 : angle D A X = angle C B X)
  (h5 : angle A D X < π/2)
  (h6 : angle D A X < π/2)
  (h7 : angle B C X < π/2)
  (h8 : angle C B X < π/2) :
  angle A Y B = 2 * angle A D X := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_l2328_232897


namespace NUMINAMATH_CALUDE_twenty_team_tournament_games_l2328_232881

/-- Calculates the number of games in a single-elimination tournament. -/
def gamesInTournament (n : ℕ) : ℕ := n - 1

/-- Theorem: A single-elimination tournament with 20 teams requires 19 games to determine the winner. -/
theorem twenty_team_tournament_games :
  gamesInTournament 20 = 19 := by
  sorry

end NUMINAMATH_CALUDE_twenty_team_tournament_games_l2328_232881


namespace NUMINAMATH_CALUDE_smallest_value_for_x_between_zero_and_one_l2328_232859

theorem smallest_value_for_x_between_zero_and_one (x : ℝ) (h : 0 < x ∧ x < 1) :
  x^3 < x^2 ∧ x^2 < x ∧ x < 2*x ∧ 2*x < 3*x := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_between_zero_and_one_l2328_232859


namespace NUMINAMATH_CALUDE_stating_speed_ratio_equals_one_plus_head_start_l2328_232829

/-- The ratio of runner A's speed to runner B's speed in a race where A gives B a head start -/
def speed_ratio : ℝ := 1.11764705882352941

/-- The fraction of the race length that runner A gives as a head start to runner B -/
def head_start : ℝ := 0.11764705882352941

/-- 
Theorem stating that the speed ratio of runner A to runner B is equal to 1 plus the head start fraction,
given that the race ends in a dead heat when A gives B the specified head start.
-/
theorem speed_ratio_equals_one_plus_head_start : 
  speed_ratio = 1 + head_start := by sorry

end NUMINAMATH_CALUDE_stating_speed_ratio_equals_one_plus_head_start_l2328_232829


namespace NUMINAMATH_CALUDE_S_a_is_three_rays_with_common_point_l2328_232893

/-- The set S_a for a positive integer a -/
def S_a (a : ℕ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    (a = p.1 + 2 ∧ p.2 - 4 ≤ a) ∨
    (a = p.2 - 4 ∧ p.1 + 2 ≤ a) ∨
    (p.1 + 2 = p.2 - 4 ∧ a ≤ p.1 + 2)}

/-- The common point of the three rays -/
def common_point (a : ℕ) : ℝ × ℝ := (a - 2, a + 4)

/-- The three rays that form S_a -/
def ray1 (a : ℕ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = a - 2 ∧ p.2 ≤ a + 4}
def ray2 (a : ℕ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a + 4 ∧ p.1 ≤ a - 2}
def ray3 (a : ℕ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 6 ∧ p.1 ≥ a - 2}

/-- Theorem stating that S_a is the union of three rays with a common point -/
theorem S_a_is_three_rays_with_common_point (a : ℕ) :
  S_a a = ray1 a ∪ ray2 a ∪ ray3 a ∧
  common_point a ∈ ray1 a ∧
  common_point a ∈ ray2 a ∧
  common_point a ∈ ray3 a :=
sorry

end NUMINAMATH_CALUDE_S_a_is_three_rays_with_common_point_l2328_232893


namespace NUMINAMATH_CALUDE_fusilli_to_penne_ratio_l2328_232887

/-- Given a survey of pasta preferences, prove the ratio of fusilli to penne preferences --/
theorem fusilli_to_penne_ratio :
  ∀ (total students_fusilli students_penne : ℕ),
  total = 800 →
  students_fusilli = 320 →
  students_penne = 160 →
  (students_fusilli : ℚ) / (students_penne : ℚ) = 2 := by
sorry

end NUMINAMATH_CALUDE_fusilli_to_penne_ratio_l2328_232887


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l2328_232856

theorem quadratic_root_theorem (a b c d : ℝ) (h : a ≠ 0) :
  (a * (b - c - d) * 1^2 + b * (c - a + d) * 1 + c * (a - b - d) = 0) →
  ∃ x : ℝ, x ≠ 1 ∧ 
    a * (b - c - d) * x^2 + b * (c - a + d) * x + c * (a - b - d) = 0 ∧
    x = c * (a - b - d) / (a * (b - c - d)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l2328_232856


namespace NUMINAMATH_CALUDE_condition_equivalent_to_inequality_l2328_232823

-- Define an increasing function on ℝ
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem condition_equivalent_to_inequality
  (f : ℝ → ℝ) (h : IncreasingFunction f) :
  (∀ a b : ℝ, a + b > 0 ↔ f a + f b > f (-a) + f (-b)) :=
by sorry

end NUMINAMATH_CALUDE_condition_equivalent_to_inequality_l2328_232823


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2328_232894

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) : 
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2328_232894


namespace NUMINAMATH_CALUDE_triangle_cosine_inequality_l2328_232800

theorem triangle_cosine_inequality (A B C : Real) 
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0)
  (h_triangle : A + B + C = Real.pi) : 
  (Real.cos A)^2 / (Real.cos B)^2 + 
  (Real.cos B)^2 / (Real.cos C)^2 + 
  (Real.cos C)^2 / (Real.cos A)^2 ≥ 
  4 * ((Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_inequality_l2328_232800


namespace NUMINAMATH_CALUDE_inverse_of_21_mod_47_l2328_232832

theorem inverse_of_21_mod_47 (h : (8⁻¹ : ZMod 47) = 6) : (21⁻¹ : ZMod 47) = 38 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_21_mod_47_l2328_232832


namespace NUMINAMATH_CALUDE_bruce_bank_savings_l2328_232888

/-- The amount of money Bruce puts in the bank given his birthday gifts -/
def money_in_bank (aunt_gift : ℕ) (grandfather_gift : ℕ) : ℕ :=
  (aunt_gift + grandfather_gift) / 5

/-- Theorem stating that Bruce puts $45 in the bank -/
theorem bruce_bank_savings : money_in_bank 75 150 = 45 := by
  sorry

end NUMINAMATH_CALUDE_bruce_bank_savings_l2328_232888


namespace NUMINAMATH_CALUDE_sum_product_inequality_cubic_inequality_l2328_232866

-- Part 1
theorem sum_product_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  a * b + b * c + c * a ≤ 1 / 3 := by
sorry

-- Part 2
theorem cubic_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b) :
  2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := by
sorry

end NUMINAMATH_CALUDE_sum_product_inequality_cubic_inequality_l2328_232866


namespace NUMINAMATH_CALUDE_sum_palindromic_primes_l2328_232803

def isPrime (n : Nat) : Bool := sorry

def reverseDigits (n : Nat) : Nat := sorry

def isPalindromicPrime (n : Nat) : Bool :=
  isPrime n ∧ isPrime (reverseDigits n)

def palindromicPrimes : List Nat :=
  (List.range 90).filter (fun n => n ≥ 10 ∧ isPalindromicPrime n)

theorem sum_palindromic_primes :
  palindromicPrimes.sum = 429 := by sorry

end NUMINAMATH_CALUDE_sum_palindromic_primes_l2328_232803


namespace NUMINAMATH_CALUDE_max_value_theorem_l2328_232833

theorem max_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 2) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 → (a + b) / (a * b * c) ≤ (x + y) / (x * y * z)) →
  (x + y) / (x * y * z) = 13.5 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2328_232833


namespace NUMINAMATH_CALUDE_sequence_20th_term_l2328_232845

/-- Given a sequence {aₙ} where a₁ = 1 and aₙ₊₁ = aₙ + 2 for n ∈ ℕ*, prove that a₂₀ = 39 -/
theorem sequence_20th_term (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2) : 
  a 20 = 39 := by
  sorry

end NUMINAMATH_CALUDE_sequence_20th_term_l2328_232845


namespace NUMINAMATH_CALUDE_problem_solution_l2328_232891

noncomputable def f (x : ℝ) := |Real.log x|

theorem problem_solution (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  (a * b = 1) ∧ 
  ((a + b) / 2 > 1) ∧ 
  (∃ b₀ : ℝ, 3 < b₀ ∧ b₀ < 4 ∧ 1 / b₀^2 + b₀^2 + 2 - 4 * b₀ = 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2328_232891


namespace NUMINAMATH_CALUDE_total_sweets_l2328_232841

theorem total_sweets (num_crates : ℕ) (sweets_per_crate : ℕ) 
  (h1 : num_crates = 4) 
  (h2 : sweets_per_crate = 16) : 
  num_crates * sweets_per_crate = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_sweets_l2328_232841


namespace NUMINAMATH_CALUDE_sqrt_x4_minus_x2_l2328_232869

theorem sqrt_x4_minus_x2 (x : ℝ) : Real.sqrt (x^4 - x^2) = |x| * Real.sqrt (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x4_minus_x2_l2328_232869


namespace NUMINAMATH_CALUDE_chairs_to_hall_l2328_232821

theorem chairs_to_hall (num_students : ℕ) (chairs_per_trip : ℕ) (num_trips : ℕ) :
  num_students = 5 → chairs_per_trip = 5 → num_trips = 10 →
  num_students * chairs_per_trip * num_trips = 250 := by
  sorry

end NUMINAMATH_CALUDE_chairs_to_hall_l2328_232821


namespace NUMINAMATH_CALUDE_union_of_intervals_l2328_232876

open Set

theorem union_of_intervals (A B : Set ℝ) :
  A = Ioc (-1) 1 → B = Ioo 0 2 → A ∪ B = Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_union_of_intervals_l2328_232876


namespace NUMINAMATH_CALUDE_chord_addition_theorem_sum_of_squares_theorem_l2328_232804

/-- Represents a circle with chords --/
structure ChordedCircle where
  num_chords : ℕ
  num_regions : ℕ

/-- The result of adding a chord to a circle --/
structure ChordAdditionResult where
  min_regions : ℕ
  max_regions : ℕ

/-- Function to add a chord to a circle --/
def add_chord (circle : ChordedCircle) : ChordAdditionResult :=
  { min_regions := circle.num_regions + 1,
    max_regions := circle.num_regions + circle.num_chords + 1 }

/-- Theorem statement --/
theorem chord_addition_theorem (initial_circle : ChordedCircle) 
  (h1 : initial_circle.num_chords = 4) 
  (h2 : initial_circle.num_regions = 9) : 
  let result := add_chord initial_circle
  result.min_regions = 10 ∧ result.max_regions = 14 := by
  sorry

/-- Corollary: The sum of squares of max and min regions --/
theorem sum_of_squares_theorem (initial_circle : ChordedCircle) 
  (h1 : initial_circle.num_chords = 4) 
  (h2 : initial_circle.num_regions = 9) : 
  let result := add_chord initial_circle
  result.max_regions ^ 2 + result.min_regions ^ 2 = 296 := by
  sorry

end NUMINAMATH_CALUDE_chord_addition_theorem_sum_of_squares_theorem_l2328_232804


namespace NUMINAMATH_CALUDE_marks_percentage_raise_l2328_232898

/-- Calculates the percentage raise Mark received at his job -/
theorem marks_percentage_raise
  (original_hourly_rate : ℚ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (old_weekly_bills : ℚ)
  (new_weekly_expense : ℚ)
  (new_leftover_amount : ℚ)
  (h1 : original_hourly_rate = 40)
  (h2 : hours_per_day = 8)
  (h3 : days_per_week = 5)
  (h4 : old_weekly_bills = 600)
  (h5 : new_weekly_expense = 100)
  (h6 : new_leftover_amount = 980) :
  (new_leftover_amount + old_weekly_bills + new_weekly_expense - 
   (original_hourly_rate * hours_per_day * days_per_week)) / 
  (original_hourly_rate * hours_per_day * days_per_week) = 1/20 :=
by sorry

end NUMINAMATH_CALUDE_marks_percentage_raise_l2328_232898


namespace NUMINAMATH_CALUDE_number_difference_l2328_232861

theorem number_difference (x y : ℝ) (sum_eq : x + y = 42) (prod_eq : x * y = 437) :
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2328_232861


namespace NUMINAMATH_CALUDE_circle_max_sum_squares_l2328_232847

theorem circle_max_sum_squares :
  ∀ x y : ℝ, x^2 - 4*x - 4 + y^2 = 0 →
  x^2 + y^2 ≤ 12 + 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_max_sum_squares_l2328_232847


namespace NUMINAMATH_CALUDE_mitch_max_boat_length_l2328_232889

/-- The maximum length of boat Mitch can buy given his savings and expenses --/
def max_boat_length (savings : ℚ) (cost_per_foot : ℚ) (license_fee : ℚ) : ℚ :=
  let docking_fee := 3 * license_fee
  let total_fees := license_fee + docking_fee
  let remaining_money := savings - total_fees
  remaining_money / cost_per_foot

/-- Theorem stating the maximum length of boat Mitch can buy --/
theorem mitch_max_boat_length :
  max_boat_length 20000 1500 500 = 12 := by
sorry

end NUMINAMATH_CALUDE_mitch_max_boat_length_l2328_232889


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2328_232835

theorem quadratic_roots_property (d e : ℝ) : 
  (3 * d^2 + 4 * d - 7 = 0) →
  (3 * e^2 + 4 * e - 7 = 0) →
  (d - 2) * (e - 2) = 13/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2328_232835


namespace NUMINAMATH_CALUDE_drilled_solid_surface_area_l2328_232822

noncomputable section

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents the drilled solid S -/
structure DrilledSolid where
  cube : Cube
  tunnelStart : Point3D
  tunnelEnd : Point3D

/-- Calculate the surface area of the drilled solid S -/
def surfaceArea (s : DrilledSolid) : ℝ := sorry

/-- The main theorem stating the surface area of the drilled solid -/
theorem drilled_solid_surface_area 
  (e f g h c d b a i j k : Point3D)
  (cube : Cube)
  (s : DrilledSolid)
  (h1 : cube.edgeLength = 10)
  (h2 : e.x = 10 ∧ e.y = 10 ∧ e.z = 10)
  (h3 : i.x = 7 ∧ i.y = 10 ∧ i.z = 10)
  (h4 : j.x = 10 ∧ j.y = 7 ∧ j.z = 10)
  (h5 : k.x = 10 ∧ k.y = 10 ∧ k.z = 7)
  (h6 : s.cube = cube)
  (h7 : s.tunnelStart = i)
  (h8 : s.tunnelEnd = k) :
  surfaceArea s = 582 + 13.5 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_drilled_solid_surface_area_l2328_232822


namespace NUMINAMATH_CALUDE_sheila_mon_wed_fri_hours_l2328_232868

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ
  hourly_rate : ℕ

/-- Theorem stating the number of hours Sheila works on Monday, Wednesday, and Friday --/
theorem sheila_mon_wed_fri_hours (schedule : WorkSchedule) 
  (h1 : schedule.hours_tue_thu = 6 * 2)
  (h2 : schedule.weekly_earnings = 360)
  (h3 : schedule.hourly_rate = 10)
  (h4 : schedule.weekly_earnings = schedule.hourly_rate * (schedule.hours_mon_wed_fri + schedule.hours_tue_thu)) :
  schedule.hours_mon_wed_fri = 24 := by
  sorry

#check sheila_mon_wed_fri_hours

end NUMINAMATH_CALUDE_sheila_mon_wed_fri_hours_l2328_232868


namespace NUMINAMATH_CALUDE_product_evaluation_l2328_232831

theorem product_evaluation (n : ℕ) (h : n = 3) : 
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2328_232831


namespace NUMINAMATH_CALUDE_building_height_l2328_232850

/-- Given a flagpole and a building casting shadows under similar conditions,
    this theorem proves that the height of the building is 22 meters. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 55)
  : (flagpole_height / flagpole_shadow) * building_shadow = 22 :=
by sorry

end NUMINAMATH_CALUDE_building_height_l2328_232850


namespace NUMINAMATH_CALUDE_k_range_theorem_l2328_232849

/-- The range of k given the conditions in the problem -/
def k_range : Set ℝ := Set.Iic 0 ∪ Set.Ioo (1/2) (5/2)

/-- p: the function y=kx+1 is increasing on ℝ -/
def p (k : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ + 1 < k * x₂ + 1

/-- q: the equation x^2+(2k-3)x+1=0 has real solutions -/
def q (k : ℝ) : Prop := ∃ x : ℝ, x^2 + (2*k - 3)*x + 1 = 0

/-- Main theorem stating the range of k -/
theorem k_range_theorem (h1 : ∀ k : ℝ, ¬(p k ∧ q k)) (h2 : ∀ k : ℝ, p k ∨ q k) : 
  ∀ k : ℝ, k ∈ k_range ↔ (p k ∨ q k) :=
sorry

end NUMINAMATH_CALUDE_k_range_theorem_l2328_232849


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_for_all_x_geq_4_l2328_232818

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for part (1)
theorem solution_set_when_a_is_4 :
  {x : ℝ | f 4 x ≥ 5} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 5} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_all_x_geq_4 :
  (∀ x : ℝ, f a x ≥ 4) ↔ (a ≤ -3 ∨ a ≥ 5) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_4_range_of_a_for_all_x_geq_4_l2328_232818


namespace NUMINAMATH_CALUDE_iron_cars_count_l2328_232880

/-- Represents the initial state and rules for a train delivery problem -/
structure TrainProblem where
  coal_cars : ℕ
  wood_cars : ℕ
  station_distance : ℕ
  travel_time : ℕ
  max_coal_deposit : ℕ
  max_iron_deposit : ℕ
  max_wood_deposit : ℕ
  total_delivery_time : ℕ

/-- Calculates the number of iron cars given a TrainProblem -/
def calculate_iron_cars (problem : TrainProblem) : ℕ :=
  let num_stations := problem.total_delivery_time / problem.travel_time
  num_stations * problem.max_iron_deposit

/-- Theorem stating that for the given problem, the number of iron cars is 12 -/
theorem iron_cars_count (problem : TrainProblem) 
  (h1 : problem.coal_cars = 6)
  (h2 : problem.wood_cars = 2)
  (h3 : problem.station_distance = 6)
  (h4 : problem.travel_time = 25)
  (h5 : problem.max_coal_deposit = 2)
  (h6 : problem.max_iron_deposit = 3)
  (h7 : problem.max_wood_deposit = 1)
  (h8 : problem.total_delivery_time = 100) :
  calculate_iron_cars problem = 12 := by
  sorry

end NUMINAMATH_CALUDE_iron_cars_count_l2328_232880


namespace NUMINAMATH_CALUDE_book_cost_calculation_l2328_232885

/-- Calculates the cost of each book given the total customers, return rate, and total sales after returns. -/
theorem book_cost_calculation (total_customers : ℕ) (return_rate : ℚ) (total_sales : ℚ) : 
  total_customers = 1000 → 
  return_rate = 37 / 100 → 
  total_sales = 9450 → 
  (total_sales / (total_customers * (1 - return_rate))) = 15 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_calculation_l2328_232885


namespace NUMINAMATH_CALUDE_fish_remaining_l2328_232839

theorem fish_remaining (initial : Float) (given_away : Float) : 
  initial = 47.0 → given_away = 22.5 → initial - given_away = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_fish_remaining_l2328_232839


namespace NUMINAMATH_CALUDE_sticker_enlargement_l2328_232828

/-- Given a rectangle with original width and height, and a new width,
    calculate the new height when enlarged proportionately -/
def new_height (original_width original_height new_width : ℚ) : ℚ :=
  (new_width / original_width) * original_height

/-- Theorem stating that a 3x2 inch rectangle enlarged to 12 inches wide
    will be 8 inches tall -/
theorem sticker_enlargement :
  new_height 3 2 12 = 8 := by sorry

end NUMINAMATH_CALUDE_sticker_enlargement_l2328_232828


namespace NUMINAMATH_CALUDE_shoes_mode_median_equal_l2328_232838

structure SalesData where
  sizes : List Float
  volumes : List Nat
  total_pairs : Nat

def mode (data : SalesData) : Float :=
  sorry

def median (data : SalesData) : Float :=
  sorry

theorem shoes_mode_median_equal (data : SalesData) :
  data.sizes = [23, 23.5, 24, 24.5, 25] ∧
  data.volumes = [1, 2, 2, 6, 2] ∧
  data.total_pairs = 15 →
  mode data = 24.5 ∧ median data = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_shoes_mode_median_equal_l2328_232838
