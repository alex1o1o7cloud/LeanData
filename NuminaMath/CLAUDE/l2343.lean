import Mathlib

namespace NUMINAMATH_CALUDE_propositions_b_and_c_are_true_l2343_234317

theorem propositions_b_and_c_are_true :
  (∀ a b : ℝ, |a| > |b| → a^2 > b^2) ∧
  (∀ a b c : ℝ, (a - b) * c^2 > 0 → a > b) := by
  sorry

end NUMINAMATH_CALUDE_propositions_b_and_c_are_true_l2343_234317


namespace NUMINAMATH_CALUDE_petya_vasya_meeting_l2343_234384

/-- The number of street lamps along the alley -/
def num_lamps : ℕ := 100

/-- The lamp number where Petya is observed -/
def petya_observed : ℕ := 22

/-- The lamp number where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- Calculates the meeting point of Petya and Vasya -/
def meeting_point : ℕ := 64

/-- Theorem stating that Petya and Vasya meet at the calculated meeting point -/
theorem petya_vasya_meeting :
  ∀ (petya_speed vasya_speed : ℚ),
  petya_speed > 0 ∧ vasya_speed > 0 →
  (petya_speed * (meeting_point - 1) = vasya_speed * (num_lamps - meeting_point)) ∧
  (petya_speed * (petya_observed - 1) = vasya_speed * (num_lamps - vasya_observed)) :=
by sorry

#check petya_vasya_meeting

end NUMINAMATH_CALUDE_petya_vasya_meeting_l2343_234384


namespace NUMINAMATH_CALUDE_subscription_savings_l2343_234386

def category_A_cost : ℝ := 520
def category_B_cost : ℝ := 860
def category_C_cost : ℝ := 620

def category_A_cut_percentage : ℝ := 0.25
def category_B_cut_percentage : ℝ := 0.35
def category_C_cut_percentage : ℝ := 0.30

def total_savings : ℝ :=
  category_A_cost * category_A_cut_percentage +
  category_B_cost * category_B_cut_percentage +
  category_C_cost * category_C_cut_percentage

theorem subscription_savings : total_savings = 617 := by
  sorry

end NUMINAMATH_CALUDE_subscription_savings_l2343_234386


namespace NUMINAMATH_CALUDE_y_coordinate_relationship_l2343_234395

/-- A quadratic function of the form y = -(x-2)² + h -/
def f (h : ℝ) (x : ℝ) : ℝ := -(x - 2)^2 + h

/-- Theorem stating the relationship between y-coordinates of three points on the quadratic function -/
theorem y_coordinate_relationship (h : ℝ) (y₁ y₂ y₃ : ℝ) 
  (hA : f h (-1/2) = y₁)
  (hB : f h 1 = y₂)
  (hC : f h 2 = y₃) :
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_relationship_l2343_234395


namespace NUMINAMATH_CALUDE_first_discount_percentage_l2343_234331

/-- Proves that the first discount percentage is 25% given the original price, final price, and second discount percentage. -/
theorem first_discount_percentage 
  (original_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) 
  (h1 : original_price = 33.78)
  (h2 : final_price = 19)
  (h3 : second_discount = 25) :
  ∃ (first_discount : ℝ), 
    first_discount = 25 ∧ 
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) := by
  sorry


end NUMINAMATH_CALUDE_first_discount_percentage_l2343_234331


namespace NUMINAMATH_CALUDE_reciprocal_equation_solution_l2343_234336

theorem reciprocal_equation_solution (x : ℝ) : 
  (2 - 1 / (3 - 2 * x) = 1 / (3 - 2 * x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_solution_l2343_234336


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l2343_234353

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicularLine : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane
  (α β : Plane) (m : Line)
  (h1 : perpendicular α β)
  (h2 : perpendicularLine m β)
  (h3 : ¬ contains α m) :
  parallel m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l2343_234353


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2343_234330

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2343_234330


namespace NUMINAMATH_CALUDE_eighth_term_of_arithmetic_sequence_l2343_234320

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the second term is 17 and the fifth term is 19,
    the eighth term is 21. -/
theorem eighth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_second_term : a 2 = 17)
  (h_fifth_term : a 5 = 19) :
  a 8 = 21 := by
  sorry


end NUMINAMATH_CALUDE_eighth_term_of_arithmetic_sequence_l2343_234320


namespace NUMINAMATH_CALUDE_sum_of_bn_l2343_234340

theorem sum_of_bn (m : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (∀ n ∈ Finset.range (2 * m + 1), (a n) * (a (n + 1)) = b n) →
  (∀ n ∈ Finset.range (2 * m), (a n) + (a (n + 1)) = -4 * n) →
  a 1 = 0 →
  (Finset.range (2 * m)).sum b = (8 * m / 3) * (4 * m^2 + 3 * m - 1) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_bn_l2343_234340


namespace NUMINAMATH_CALUDE_derivative_of_periodic_is_periodic_l2343_234339

/-- A function f is periodic with period T if f(x + T) = f(x) for all x -/
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- The main theorem: If f is differentiable and periodic with period T,
    then its derivative f' is also periodic with period T -/
theorem derivative_of_periodic_is_periodic
    (f : ℝ → ℝ) (T : ℝ) (hT : T > 0) (hf : Differentiable ℝ f) (hper : IsPeriodic f T) :
    IsPeriodic (deriv f) T := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_periodic_is_periodic_l2343_234339


namespace NUMINAMATH_CALUDE_expedition_time_theorem_l2343_234376

/-- Represents the expedition parameters and calculates the minimum time to circle the mountain. -/
def minimum_expedition_time (total_distance : ℝ) (walking_speed : ℝ) (food_capacity : ℝ) : ℝ :=
  23.5

/-- Theorem stating that the minimum time to circle the mountain under given conditions is 23.5 days. -/
theorem expedition_time_theorem (total_distance : ℝ) (walking_speed : ℝ) (food_capacity : ℝ) 
  (h1 : total_distance = 100)
  (h2 : walking_speed = 20)
  (h3 : food_capacity = 2) :
  minimum_expedition_time total_distance walking_speed food_capacity = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_expedition_time_theorem_l2343_234376


namespace NUMINAMATH_CALUDE_max_slope_product_30deg_l2343_234332

/-- The maximum product of slopes for two lines intersecting at 30° with one slope four times the other -/
theorem max_slope_product_30deg (m₁ m₂ : ℝ) : 
  m₁ ≠ 0 → m₂ ≠ 0 →  -- nonhorizontal and nonvertical lines
  m₂ = 4 * m₁ →  -- one slope is 4 times the other
  |((m₂ - m₁) / (1 + m₁ * m₂))| = 1 / Real.sqrt 3 →  -- 30° angle between lines
  m₁ * m₂ ≤ (3 * Real.sqrt 3 + Real.sqrt 11)^2 / 16 :=
by sorry

end NUMINAMATH_CALUDE_max_slope_product_30deg_l2343_234332


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2343_234304

def U : Set ℤ := {-3, -1, 0, 1, 3}

def A : Set ℤ := {x | x^2 - 2*x - 3 = 0}

theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = {-3, 0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2343_234304


namespace NUMINAMATH_CALUDE_U_value_l2343_234316

theorem U_value : 
  let U := 1 / (4 - Real.sqrt 9) + 1 / (Real.sqrt 9 - Real.sqrt 8) - 
           1 / (Real.sqrt 8 - Real.sqrt 7) + 1 / (Real.sqrt 7 - Real.sqrt 6) - 
           1 / (Real.sqrt 6 - 3)
  U = 1 := by sorry

end NUMINAMATH_CALUDE_U_value_l2343_234316


namespace NUMINAMATH_CALUDE_c_k_value_l2343_234361

/-- Arithmetic sequence with first term 1 and common difference d -/
def arithmetic_seq (d : ℕ) (n : ℕ) : ℕ :=
  1 + (n - 1) * d

/-- Geometric sequence with first term 1 and common ratio r -/
def geometric_seq (r : ℕ) (n : ℕ) : ℕ :=
  r ^ (n - 1)

/-- Sum of arithmetic and geometric sequences -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r k : ℕ) :
  c_seq d r (k - 1) = 50 ∧ c_seq d r (k + 1) = 500 → c_seq d r k = 78 := by
  sorry

end NUMINAMATH_CALUDE_c_k_value_l2343_234361


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l2343_234363

theorem cubic_inequality_solution (x : ℝ) :
  x^3 - 12*x^2 + 36*x > 0 ↔ (x > 0 ∧ x < 6) ∨ x > 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l2343_234363


namespace NUMINAMATH_CALUDE_problem_statement_l2343_234390

theorem problem_statement (n b : ℝ) : 
  n = 2^(7/3) → n^(3*b + 5) = 256 → b = -11/21 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2343_234390


namespace NUMINAMATH_CALUDE_sunglasses_sold_l2343_234374

/-- Proves that the number of pairs of sunglasses sold is 10 -/
theorem sunglasses_sold (selling_price cost_price sign_cost : ℕ) 
  (h1 : selling_price = 30)
  (h2 : cost_price = 26)
  (h3 : sign_cost = 20) :
  (sign_cost * 2) / (selling_price - cost_price) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_sold_l2343_234374


namespace NUMINAMATH_CALUDE_set_operations_l2343_234346

def A : Set ℝ := {x | 1 < 2*x - 1 ∧ 2*x - 1 < 7}
def B : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem set_operations :
  (A ∩ B = {x | 1 < x ∧ x < 3}) ∧
  (Set.compl (A ∪ B) = {x | x ≤ -1 ∨ x ≥ 4}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2343_234346


namespace NUMINAMATH_CALUDE_min_value_expression_l2343_234371

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1/(a*b) + 1/(a*(a-b)) ≥ 4 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > b₀ ∧ b₀ > 0 ∧ a₀^2 + 1/(a₀*b₀) + 1/(a₀*(a₀-b₀)) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2343_234371


namespace NUMINAMATH_CALUDE_quadratic_linear_intersection_l2343_234380

/-- Quadratic function -/
def y1 (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Linear function -/
def y2 (a b x : ℝ) : ℝ := a * x + b

/-- Theorem stating the main results -/
theorem quadratic_linear_intersection 
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : y1 a b c 1 = 0) 
  (t : ℤ) 
  (h4 : t % 2 = 1) 
  (h5 : y1 a b c (t : ℝ) = 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ y1 a b c x1 = y2 a b x1 ∧ y1 a b c x2 = y2 a b x2) ∧ 
  (t = 1 ∨ t = -1) ∧
  (∀ A1 B1 : ℝ, y1 a b c A1 = y2 a b A1 → y1 a b c B1 = y2 a b B1 → 
    3/2 < |A1 - B1| ∧ |A1 - B1| < Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_linear_intersection_l2343_234380


namespace NUMINAMATH_CALUDE_star_computation_l2343_234301

/-- Operation ⭐ defined as (5a + b) / (a - b) -/
def star (a b : ℚ) : ℚ := (5 * a + b) / (a - b)

theorem star_computation :
  star (star 7 (star 2 5)) 3 = -31 := by
  sorry

end NUMINAMATH_CALUDE_star_computation_l2343_234301


namespace NUMINAMATH_CALUDE_probability_two_red_two_blue_l2343_234347

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 4

theorem probability_two_red_two_blue :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2) / Nat.choose total_marbles selected_marbles = 56 / 147 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_two_blue_l2343_234347


namespace NUMINAMATH_CALUDE_f_properties_l2343_234307

noncomputable def f (x : ℝ) := Real.sin x * (Real.sin x + Real.sqrt 3 * Real.cos x)

theorem f_properties :
  ∃ (T : ℝ) (M : ℝ) (S : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (∀ t, 0 < t → t < T → ¬ (∀ x, f (x + t) = f x)) ∧
    (∀ x, f x ≤ M) ∧
    (∃ x, f x = M) ∧
    T = π ∧
    M = 3/2 ∧
    (∀ A B C a b c : ℝ,
      0 < A ∧ A < π/2 ∧
      0 < B ∧ B < π/2 ∧
      0 < C ∧ C < π/2 ∧
      A + B + C = π ∧
      f (A/2) = 1 ∧
      a = 2 * Real.sqrt 3 ∧
      a = b * Real.sin C ∧
      b = c * Real.sin A ∧
      c = a * Real.sin B →
      1/2 * b * c * Real.sin A ≤ S) ∧
    S = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2343_234307


namespace NUMINAMATH_CALUDE_youngest_child_age_l2343_234338

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The set of ages of the six children -/
def childrenAges (x : ℕ) : Finset ℕ :=
  {x, x + 2, x + 6, x + 8, x + 12, x + 14}

/-- Theorem stating that the youngest child's age is 5 -/
theorem youngest_child_age :
  ∃ (x : ℕ), x = 5 ∧ 
    (∀ y ∈ childrenAges x, isPrime y) ∧
    (childrenAges x).card = 6 :=
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_l2343_234338


namespace NUMINAMATH_CALUDE_favorite_fruit_oranges_l2343_234388

theorem favorite_fruit_oranges (total students_pears students_apples students_strawberries : ℕ) 
  (h_total : total = 450)
  (h_pears : students_pears = 120)
  (h_apples : students_apples = 147)
  (h_strawberries : students_strawberries = 113) :
  total - (students_pears + students_apples + students_strawberries) = 70 := by
  sorry

end NUMINAMATH_CALUDE_favorite_fruit_oranges_l2343_234388


namespace NUMINAMATH_CALUDE_tan_cos_expression_equals_negative_one_l2343_234305

theorem tan_cos_expression_equals_negative_one :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_cos_expression_equals_negative_one_l2343_234305


namespace NUMINAMATH_CALUDE_eight_percent_difference_l2343_234321

theorem eight_percent_difference (x y : ℝ) 
  (hx : 8 = 0.25 * x) 
  (hy : 8 = 0.5 * y) : 
  x - y = 16 := by
  sorry

end NUMINAMATH_CALUDE_eight_percent_difference_l2343_234321


namespace NUMINAMATH_CALUDE_unique_prime_pair_with_square_differences_l2343_234399

theorem unique_prime_pair_with_square_differences : 
  ∃! (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    ∃ (a b : ℕ), a^2 = p - q ∧ b^2 = p*q - q :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_with_square_differences_l2343_234399


namespace NUMINAMATH_CALUDE_expression_simplification_l2343_234306

theorem expression_simplification (a b : ℝ) 
  (h : |a - 1| + b^2 - 6*b + 9 = 0) : 
  ((3*a + 2*b)*(3*a - 2*b) + (3*a - b)^2 - b*(2*a - 3*b)) / (2*a) = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2343_234306


namespace NUMINAMATH_CALUDE_parabola_properties_l2343_234312

/-- A parabola with given properties -/
structure Parabola where
  vertex : ℝ × ℝ
  axis_vertical : Bool
  passing_point : ℝ × ℝ

/-- Shift vector -/
def shift_vector : ℝ × ℝ := (2, 3)

/-- Our specific parabola -/
def our_parabola : Parabola := {
  vertex := (3, -2),
  axis_vertical := true,
  passing_point := (5, 6)
}

/-- The equation of our parabola -/
def parabola_equation (x : ℝ) : ℝ := 2 * x^2 - 12 * x + 16

/-- The new vertex after shifting -/
def new_vertex : ℝ × ℝ := (5, 1)

theorem parabola_properties :
  (∀ x, parabola_equation x = 2 * (x - our_parabola.vertex.1)^2 + our_parabola.vertex.2) ∧
  parabola_equation our_parabola.passing_point.1 = our_parabola.passing_point.2 ∧
  new_vertex = (our_parabola.vertex.1 + shift_vector.1, our_parabola.vertex.2 + shift_vector.2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2343_234312


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l2343_234324

theorem triangle_angle_inequality (A B C : Real) 
  (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) 
  (h4 : A + B + C = π) : A * Real.cos B + Real.sin A * Real.sin C > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l2343_234324


namespace NUMINAMATH_CALUDE_geometry_textbook_weight_l2343_234318

/-- The weight of Kelly's chemistry textbook in pounds -/
def chemistry_weight : ℝ := 7.125

/-- The weight difference between the chemistry and geometry textbooks in pounds -/
def weight_difference : ℝ := 6.5

/-- The weight of Kelly's geometry textbook in pounds -/
def geometry_weight : ℝ := chemistry_weight - weight_difference

theorem geometry_textbook_weight :
  geometry_weight = 0.625 := by sorry

end NUMINAMATH_CALUDE_geometry_textbook_weight_l2343_234318


namespace NUMINAMATH_CALUDE_arccos_zero_equals_pi_half_l2343_234334

theorem arccos_zero_equals_pi_half : Real.arccos 0 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_zero_equals_pi_half_l2343_234334


namespace NUMINAMATH_CALUDE_mans_age_puzzle_l2343_234302

theorem mans_age_puzzle (A : ℕ) (h : A = 72) :
  ∃ N : ℕ, (A + 6) * N - (A - 6) * N = A ∧ N = 6 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_puzzle_l2343_234302


namespace NUMINAMATH_CALUDE_box_edge_length_and_capacity_l2343_234368

/-- Given a cubical box that can contain 999.9999999999998 cubes of 10 cm edge length,
    prove that its edge length is 1 meter and it can contain 1000 cubes. -/
theorem box_edge_length_and_capacity (box_capacity : ℝ) 
  (h1 : box_capacity = 999.9999999999998) : ∃ (edge_length : ℝ),
  edge_length = 1 ∧ 
  (edge_length * 100 / 10)^3 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_box_edge_length_and_capacity_l2343_234368


namespace NUMINAMATH_CALUDE_g_of_two_eq_zero_l2343_234378

/-- The function g(x) = x^2 - 4x + 4 -/
def g (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- Theorem: g(2) = 0 -/
theorem g_of_two_eq_zero : g 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_of_two_eq_zero_l2343_234378


namespace NUMINAMATH_CALUDE_leading_coefficient_of_g_l2343_234328

/-- A polynomial g satisfying g(x + 1) - g(x) = 4x + 6 for all x has a leading coefficient of 2 -/
theorem leading_coefficient_of_g (g : ℝ → ℝ) (hg : ∀ x, g (x + 1) - g x = 4 * x + 6) :
  ∃ (a b c : ℝ), (∀ x, g x = 2 * x^2 + a * x + b) ∧ c = 2 ∧ c ≠ 0 ∧ 
  (∀ d, (∀ x, g x = d * x^2 + a * x + b) → d ≤ c) := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_g_l2343_234328


namespace NUMINAMATH_CALUDE_p_and_q_true_p_and_not_q_false_l2343_234325

-- Define proposition p
def p : Prop := ∀ m : ℝ, ∃ x : ℝ, x^2 - m*x - 1 = 0

-- Define proposition q
def q : Prop := ∃ x₀ : ℕ, x₀^2 - 2*x₀ - 1 ≤ 0

-- Theorem stating that p and q are true
theorem p_and_q_true : p ∧ q := by sorry

-- Theorem stating that p ∧ (¬q) is false
theorem p_and_not_q_false : ¬(p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_p_and_q_true_p_and_not_q_false_l2343_234325


namespace NUMINAMATH_CALUDE_int_tan_triangle_values_l2343_234326

-- Define a triangle with integer tangents
structure IntTanTriangle where
  α : Real
  β : Real
  γ : Real
  tan_α : Int
  tan_β : Int
  tan_γ : Int
  sum_angles : α + β + γ = Real.pi
  tan_α_def : Real.tan α = tan_α
  tan_β_def : Real.tan β = tan_β
  tan_γ_def : Real.tan γ = tan_γ

-- Theorem statement
theorem int_tan_triangle_values (t : IntTanTriangle) :
  (t.tan_α = 1 ∧ t.tan_β = 2 ∧ t.tan_γ = 3) ∨
  (t.tan_α = 1 ∧ t.tan_β = 3 ∧ t.tan_γ = 2) ∨
  (t.tan_α = 2 ∧ t.tan_β = 1 ∧ t.tan_γ = 3) ∨
  (t.tan_α = 2 ∧ t.tan_β = 3 ∧ t.tan_γ = 1) ∨
  (t.tan_α = 3 ∧ t.tan_β = 1 ∧ t.tan_γ = 2) ∨
  (t.tan_α = 3 ∧ t.tan_β = 2 ∧ t.tan_γ = 1) :=
by sorry

end NUMINAMATH_CALUDE_int_tan_triangle_values_l2343_234326


namespace NUMINAMATH_CALUDE_binomial_equation_unique_solution_l2343_234366

theorem binomial_equation_unique_solution :
  ∃! n : ℕ, (Nat.choose 15 n + Nat.choose 15 7 = Nat.choose 16 8) ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_equation_unique_solution_l2343_234366


namespace NUMINAMATH_CALUDE_log_one_over_twentyfive_base_five_l2343_234360

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_over_twentyfive_base_five : log 5 (1 / 25) = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_one_over_twentyfive_base_five_l2343_234360


namespace NUMINAMATH_CALUDE_a_formula_l2343_234319

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 5
  | n + 1 => ⌊a n⌋ + 1 / (a n - ⌊a n⌋)

theorem a_formula (n : ℕ) : a n = 4 * n + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_a_formula_l2343_234319


namespace NUMINAMATH_CALUDE_range_of_m_for_trig_equation_l2343_234393

theorem range_of_m_for_trig_equation :
  ∀ α m : ℝ,
  (∃ α, Real.cos α - Real.sqrt 3 * Real.sin α = (4 * m - 6) / (4 - m)) →
  -1 ≤ m ∧ m ≤ 7/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_trig_equation_l2343_234393


namespace NUMINAMATH_CALUDE_extremum_point_implies_a_value_max_min_values_l2343_234389

-- Define the function f(x) = x^3 - ax
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- Theorem 1: If x=1 is an extremum point of f(x), then a = 3
theorem extremum_point_implies_a_value (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) →
  a = 3 :=
sorry

-- Theorem 2: For f(x) = x^3 - 3x and x ∈ [0, 2], the maximum value is 2 and the minimum value is -2
theorem max_min_values :
  (∀ x ∈ Set.Icc 0 2, f 3 x ≤ 2) ∧
  (∀ x ∈ Set.Icc 0 2, f 3 x ≥ -2) ∧
  (∃ x ∈ Set.Icc 0 2, f 3 x = 2) ∧
  (∃ x ∈ Set.Icc 0 2, f 3 x = -2) :=
sorry

end NUMINAMATH_CALUDE_extremum_point_implies_a_value_max_min_values_l2343_234389


namespace NUMINAMATH_CALUDE_system_solution_range_l2343_234359

theorem system_solution_range (x y a : ℝ) :
  (2 * x + y = 3 - a) →
  (x + 2 * y = 4 + 2 * a) →
  (x + y < 1) →
  (a < -4) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_range_l2343_234359


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2343_234337

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  a5_eq_3 : a 5 = 3
  a4_times_a7_eq_45 : a 4 * a 7 = 45

/-- The main theorem about the specific ratio in the geometric sequence -/
theorem geometric_sequence_ratio
  (seq : GeometricSequence) :
  (seq.a 7 - seq.a 9) / (seq.a 5 - seq.a 7) = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2343_234337


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2343_234322

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (5 - I) / (1 - I)
  (z.im : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2343_234322


namespace NUMINAMATH_CALUDE_sallys_onions_l2343_234308

theorem sallys_onions (fred_onions : ℕ) (given_to_sara : ℕ) (remaining_onions : ℕ) : ℕ :=
  sorry

end NUMINAMATH_CALUDE_sallys_onions_l2343_234308


namespace NUMINAMATH_CALUDE_triangle_angle_b_is_pi_third_l2343_234397

theorem triangle_angle_b_is_pi_third 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : b^2 = a*c) 
  (h2 : Real.sin A + Real.sin C = 2 * Real.sin B) 
  (h3 : A + B + C = π) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0) : 
  B = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_b_is_pi_third_l2343_234397


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l2343_234392

/-- Prove that for a hyperbola with given properties, its parameters satisfy specific values -/
theorem hyperbola_parameters (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a^2 + b^2) / a^2 = 4 →  -- eccentricity is 2
  (a * b / Real.sqrt (a^2 + b^2))^2 = 3 →  -- asymptote is tangent to the circle
  a^2 = 4 ∧ b^2 = 12 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l2343_234392


namespace NUMINAMATH_CALUDE_problem_statement_l2343_234311

theorem problem_statement (x y M : ℝ) (h : M / ((x * y + y^2) / (x - y)^2) = (x^2 - y^2) / y) :
  M = (x + y)^2 / (x - y) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2343_234311


namespace NUMINAMATH_CALUDE_games_that_didnt_work_l2343_234315

/-- The number of games that didn't work given Ned's game purchases and good games. -/
theorem games_that_didnt_work (friend_games garage_sale_games good_games : ℕ) : 
  friend_games = 50 → garage_sale_games = 27 → good_games = 3 → 
  friend_games + garage_sale_games - good_games = 74 := by
  sorry

end NUMINAMATH_CALUDE_games_that_didnt_work_l2343_234315


namespace NUMINAMATH_CALUDE_volume_conversion_l2343_234370

-- Define conversion factors
def feet_to_meters : ℝ := 0.3048
def meters_to_yards : ℝ := 1.09361

-- Define the volume in cubic feet
def volume_cubic_feet : ℝ := 216

-- Define the conversion function from cubic feet to cubic meters
def cubic_feet_to_cubic_meters (v : ℝ) : ℝ := v * (feet_to_meters ^ 3)

-- Define the conversion function from cubic meters to cubic yards
def cubic_meters_to_cubic_yards (v : ℝ) : ℝ := v * (meters_to_yards ^ 3)

-- Theorem statement
theorem volume_conversion :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |cubic_meters_to_cubic_yards (cubic_feet_to_cubic_meters volume_cubic_feet) - 8| < ε :=
sorry

end NUMINAMATH_CALUDE_volume_conversion_l2343_234370


namespace NUMINAMATH_CALUDE_p_start_time_correct_l2343_234343

/-- The time when J starts walking (in hours after midnight) -/
def j_start_time : ℝ := 12

/-- J's walking speed in km/h -/
def j_speed : ℝ := 6

/-- P's cycling speed in km/h -/
def p_speed : ℝ := 8

/-- The time when J is 3 km behind P (in hours after midnight) -/
def final_time : ℝ := 19.3

/-- The distance J is behind P at the final time (in km) -/
def distance_behind : ℝ := 3

/-- The time when P starts following J (in hours after midnight) -/
def p_start_time : ℝ := j_start_time + 1.45

theorem p_start_time_correct :
  j_speed * (final_time - j_start_time) + distance_behind =
  p_speed * (final_time - p_start_time) := by sorry

end NUMINAMATH_CALUDE_p_start_time_correct_l2343_234343


namespace NUMINAMATH_CALUDE_triangle_area_is_3_2_l2343_234323

/-- The area of the triangle bounded by the y-axis and two lines -/
def triangle_area : ℝ :=
  let line1 : ℝ → ℝ → Prop := fun x y ↦ y - 2*x = 1
  let line2 : ℝ → ℝ → Prop := fun x y ↦ 2*y + x = 10
  let y_axis : ℝ → ℝ → Prop := fun x _ ↦ x = 0
  3.2

/-- The area of the triangle is 3.2 -/
theorem triangle_area_is_3_2 : triangle_area = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_3_2_l2343_234323


namespace NUMINAMATH_CALUDE_kamals_biology_marks_l2343_234356

-- Define the known marks and average
def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def average_marks : ℕ := 74
def num_subjects : ℕ := 5

-- Define the theorem
theorem kamals_biology_marks :
  let total_marks := average_marks * num_subjects
  let known_marks_sum := english_marks + math_marks + physics_marks + chemistry_marks
  let biology_marks := total_marks - known_marks_sum
  biology_marks = 85 := by sorry

end NUMINAMATH_CALUDE_kamals_biology_marks_l2343_234356


namespace NUMINAMATH_CALUDE_arithmetic_mean_two_digit_multiples_of_8_l2343_234373

/-- The smallest positive two-digit multiple of 8 -/
def smallest_multiple : ℕ := 16

/-- The largest positive two-digit multiple of 8 -/
def largest_multiple : ℕ := 96

/-- The count of positive two-digit multiples of 8 -/
def count_multiples : ℕ := 11

/-- The sum of all positive two-digit multiples of 8 -/
def sum_multiples : ℕ := 616

/-- Theorem stating that the arithmetic mean of all positive two-digit multiples of 8 is 56 -/
theorem arithmetic_mean_two_digit_multiples_of_8 :
  (sum_multiples : ℚ) / count_multiples = 56 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_two_digit_multiples_of_8_l2343_234373


namespace NUMINAMATH_CALUDE_pr_qs_ratio_l2343_234300

/-- Given four points P, Q, R, and S on a number line, prove that the ratio of lengths PR:QS is 7:12 -/
theorem pr_qs_ratio (P Q R S : ℝ) (hP : P = 3) (hQ : Q = 5) (hR : R = 10) (hS : S = 17) :
  (R - P) / (S - Q) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_pr_qs_ratio_l2343_234300


namespace NUMINAMATH_CALUDE_square_perimeter_l2343_234344

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 392) (h2 : side^2 = area) : 
  4 * side = 112 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2343_234344


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2343_234309

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (2 - 3 * z) = 9 :=
by
  -- The unique solution is z = -79/3
  use -79/3
  constructor
  · -- Prove that -79/3 satisfies the equation
    sorry
  · -- Prove that any z satisfying the equation must equal -79/3
    sorry

#check sqrt_equation_solution

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2343_234309


namespace NUMINAMATH_CALUDE_gcd_consecutive_b_terms_bound_l2343_234367

def b (n : ℕ) : ℕ := (2 * n).factorial + n^2

theorem gcd_consecutive_b_terms_bound (n : ℕ) (h : n ≥ 1) :
  Nat.gcd (b n) (b (n + 1)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_consecutive_b_terms_bound_l2343_234367


namespace NUMINAMATH_CALUDE_samuel_coaching_fee_l2343_234381

/-- Calculates the number of days in a month, assuming a non-leap year -/
def daysInMonth (month : Nat) : Nat :=
  match month with
  | 1 | 3 | 5 | 7 | 8 | 10 | 12 => 31
  | 4 | 6 | 9 | 11 => 30
  | 2 => 28
  | _ => 0

/-- Calculates the total number of days from January 1 to a given date -/
def daysFromNewYear (month : Nat) (day : Nat) : Nat :=
  (List.range (month - 1)).foldl (fun acc m => acc + daysInMonth (m + 1)) day

/-- Represents the coaching period and daily fee -/
structure CoachingData where
  startMonth : Nat
  startDay : Nat
  endMonth : Nat
  endDay : Nat
  dailyFee : Nat

/-- Calculates the total coaching fee -/
def totalCoachingFee (data : CoachingData) : Nat :=
  let totalDays := daysFromNewYear data.endMonth data.endDay - daysFromNewYear data.startMonth data.startDay + 1
  totalDays * data.dailyFee

/-- Theorem: The total coaching fee for Samuel is 7084 dollars -/
theorem samuel_coaching_fee :
  let data : CoachingData := {
    startMonth := 1,
    startDay := 1,
    endMonth := 11,
    endDay := 4,
    dailyFee := 23
  }
  totalCoachingFee data = 7084 := by
  sorry


end NUMINAMATH_CALUDE_samuel_coaching_fee_l2343_234381


namespace NUMINAMATH_CALUDE_max_integer_difference_l2343_234329

theorem max_integer_difference (x y : ℤ) (hx : 4 < x ∧ x < 6) (hy : 6 < y ∧ y < 10) :
  (∃ (a b : ℤ), 4 < a ∧ a < 6 ∧ 6 < b ∧ b < 10 ∧ b - a ≤ y - x) ∧ y - x ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_difference_l2343_234329


namespace NUMINAMATH_CALUDE_d_bounds_l2343_234341

-- Define the circle
def Circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function
def d (P : ℝ × ℝ) : ℝ :=
  let (px, py) := P
  (px - A.1)^2 + (py - A.2)^2 + (px - B.1)^2 + (py - B.2)^2

-- Theorem statement
theorem d_bounds :
  ∀ P : ℝ × ℝ, Circle P.1 P.2 → 
  66 - 16 * Real.sqrt 2 ≤ d P ∧ d P ≤ 66 + 16 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_d_bounds_l2343_234341


namespace NUMINAMATH_CALUDE_abs_equation_solution_l2343_234333

theorem abs_equation_solution : ∃! x : ℝ, |x - 3| = 5 - x := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l2343_234333


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l2343_234348

/-- Given a train of length 1200 meters that takes 120 seconds to pass a point,
    the time required for this train to completely pass a platform of length 700 meters is 190 seconds. -/
theorem train_platform_crossing_time
  (train_length : ℝ)
  (point_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 1200)
  (h2 : point_crossing_time = 120)
  (h3 : platform_length = 700) :
  (train_length + platform_length) / (train_length / point_crossing_time) = 190 :=
sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l2343_234348


namespace NUMINAMATH_CALUDE_marble_arrangement_remainder_l2343_234398

/-- Represents the number of green marbles --/
def green_marbles : ℕ := 7

/-- Represents the minimum number of red marbles required --/
def min_red_marbles : ℕ := green_marbles + 1

/-- Represents the maximum number of additional red marbles that can be added --/
def max_additional_reds : ℕ := min_red_marbles

/-- Represents the total number of spaces where additional red marbles can be placed --/
def total_spaces : ℕ := green_marbles + 1

/-- Represents the number of ways to arrange the marbles --/
def arrangement_count : ℕ := Nat.choose (max_additional_reds + total_spaces - 1) (total_spaces - 1)

theorem marble_arrangement_remainder :
  arrangement_count % 1000 = 435 := by sorry

end NUMINAMATH_CALUDE_marble_arrangement_remainder_l2343_234398


namespace NUMINAMATH_CALUDE_magnitude_of_3_minus_4i_l2343_234362

theorem magnitude_of_3_minus_4i : Complex.abs (3 - 4 * Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_3_minus_4i_l2343_234362


namespace NUMINAMATH_CALUDE_smallest_valid_n_l2343_234358

def is_valid_n (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  2 * n = 100 * c + 10 * b + a + 5

theorem smallest_valid_n :
  ∃ (n : ℕ), is_valid_n n ∧ ∀ m, is_valid_n m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l2343_234358


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l2343_234354

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) (h1 : α ≠ β) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l2343_234354


namespace NUMINAMATH_CALUDE_boisjoli_farm_egg_boxes_l2343_234351

/-- The number of boxes filled with eggs in a week -/
def boxes_filled_per_week (num_hens : ℕ) (days_per_week : ℕ) (eggs_per_box : ℕ) : ℕ :=
  (num_hens * days_per_week) / eggs_per_box

/-- Theorem stating that 270 hens laying eggs for 7 days, packed in boxes of 6, results in 315 boxes per week -/
theorem boisjoli_farm_egg_boxes :
  boxes_filled_per_week 270 7 6 = 315 := by
  sorry

end NUMINAMATH_CALUDE_boisjoli_farm_egg_boxes_l2343_234351


namespace NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l2343_234365

/-- For a parallelogram with area 162 sq m and base 9 m, the ratio of altitude to base is 2/1 -/
theorem parallelogram_altitude_base_ratio :
  ∀ (area base altitude : ℝ),
    area = 162 →
    base = 9 →
    area = base * altitude →
    altitude / base = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l2343_234365


namespace NUMINAMATH_CALUDE_doubled_roots_quadratic_l2343_234369

theorem doubled_roots_quadratic (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 5 * x₁ - 8 = 0 ∧ 2 * x₂^2 - 5 * x₂ - 8 = 0) →
  ((2 * x₁)^2 - 5 * (2 * x₁) - 16 = 0 ∧ (2 * x₂)^2 - 5 * (2 * x₂) - 16 = 0) :=
by sorry

end NUMINAMATH_CALUDE_doubled_roots_quadratic_l2343_234369


namespace NUMINAMATH_CALUDE_earnings_difference_l2343_234394

/-- Calculates the difference in earnings between two sets of tasks with different pay rates -/
theorem earnings_difference (low_tasks : ℕ) (low_rate : ℚ) (high_tasks : ℕ) (high_rate : ℚ) :
  low_tasks = 400 →
  low_rate = 1/4 →
  high_tasks = 5 →
  high_rate = 2 →
  (low_tasks : ℚ) * low_rate - (high_tasks : ℚ) * high_rate = 90 :=
by sorry

end NUMINAMATH_CALUDE_earnings_difference_l2343_234394


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2343_234375

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 986 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  17 ∣ n ∧
  ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2343_234375


namespace NUMINAMATH_CALUDE_digits_1198_to_1200_form_473_l2343_234387

/-- A function that generates the list of positive integers with first digit 1 or 2 -/
def firstDigitOneOrTwo : ℕ → Bool := sorry

/-- The number of digits written before reaching a given position in the list -/
def digitCount (n : ℕ) : ℕ := sorry

/-- The number at a given position in the list -/
def numberAtPosition (n : ℕ) : ℕ := sorry

theorem digits_1198_to_1200_form_473 :
  let pos := 1198
  ∃ (n : ℕ), 
    firstDigitOneOrTwo n ∧ 
    digitCount n ≤ pos ∧ 
    digitCount (n + 1) > pos + 2 ∧
    numberAtPosition n = 473 := by sorry

end NUMINAMATH_CALUDE_digits_1198_to_1200_form_473_l2343_234387


namespace NUMINAMATH_CALUDE_green_marbles_count_l2343_234357

theorem green_marbles_count (G : ℕ) : 
  (2 / (2 + G : ℝ)) * (1 / (1 + G : ℝ)) = 0.1 → G = 3 := by
sorry

end NUMINAMATH_CALUDE_green_marbles_count_l2343_234357


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2343_234352

/-- A quadratic function with a non-zero leading coefficient -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The function value at a given point -/
def QuadraticFunction.value (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The derivative of the quadratic function -/
def QuadraticFunction.derivative (f : QuadraticFunction) (x : ℝ) : ℝ :=
  2 * f.a * x + f.b

theorem quadratic_function_properties (f : QuadraticFunction) 
  (h1 : f.derivative 1 = 0)
  (h2 : f.value 1 = 3)
  (h3 : f.value 2 = 8) :
  f.value (-1) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2343_234352


namespace NUMINAMATH_CALUDE_range_of_a_l2343_234327

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 5*x - 6 ≤ 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - 4*a^2 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a ≥ 0) →
  (∀ x, ¬(q x a) → ¬(p x)) →
  (∃ x, ¬(p x) ∧ (q x a)) →
  a ≥ 5/2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2343_234327


namespace NUMINAMATH_CALUDE_midnight_temperature_l2343_234372

/-- Calculates the final temperature given initial temperature and temperature changes --/
def finalTemperature (initial : Int) (noonChange : Int) (midnightChange : Int) : Int :=
  initial + noonChange - midnightChange

/-- Theorem stating that the final temperature at midnight is -4°C --/
theorem midnight_temperature :
  finalTemperature (-2) 6 8 = -4 := by
  sorry

end NUMINAMATH_CALUDE_midnight_temperature_l2343_234372


namespace NUMINAMATH_CALUDE_fred_has_four_dimes_l2343_234377

/-- The number of dimes Fred has after his sister borrowed some -/
def fred_remaining_dimes (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that Fred has 4 dimes after his sister borrowed 3 from his initial 7 -/
theorem fred_has_four_dimes :
  fred_remaining_dimes 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_four_dimes_l2343_234377


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2343_234355

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 + 5x = 0 -/
def f (x : ℝ) : ℝ := x^2 + 5*x

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f :=
sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2343_234355


namespace NUMINAMATH_CALUDE_street_length_calculation_l2343_234349

/-- Proves that the length of a street is 1800 meters, given that a person crosses it in 12 minutes at a speed of 9 km per hour. -/
theorem street_length_calculation (crossing_time : ℝ) (speed_kmh : ℝ) :
  crossing_time = 12 →
  speed_kmh = 9 →
  (speed_kmh * 1000 / 60) * crossing_time = 1800 := by
sorry

end NUMINAMATH_CALUDE_street_length_calculation_l2343_234349


namespace NUMINAMATH_CALUDE_double_roll_probability_l2343_234342

def die_roll : Finset (Nat × Nat) := Finset.product (Finset.range 6) (Finset.range 6)

def favorable_outcomes : Finset (Nat × Nat) :=
  {(0, 1), (1, 3), (2, 5)}

theorem double_roll_probability :
  (favorable_outcomes.card : ℚ) / die_roll.card = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_double_roll_probability_l2343_234342


namespace NUMINAMATH_CALUDE_division_and_addition_l2343_234345

theorem division_and_addition : (-75) / (-25) + (1 / 2) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_l2343_234345


namespace NUMINAMATH_CALUDE_simplify_expression_l2343_234385

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^4 + b^4 = a^2 + b^2) :
  a/b + b/a - 1/(a*b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2343_234385


namespace NUMINAMATH_CALUDE_smallest_m_correct_l2343_234350

/-- The smallest positive integer m for which 10x^2 - mx + 420 = 0 has integral solutions -/
def smallest_m : ℕ := 130

/-- Predicate to check if a quadratic equation has integral solutions -/
def has_integral_solutions (a b c : ℤ) : Prop :=
  ∃ x : ℤ, a * x^2 + b * x + c = 0

theorem smallest_m_correct :
  (∀ m : ℕ, m < smallest_m → ¬ has_integral_solutions 10 (-m) 420) ∧
  has_integral_solutions 10 (-smallest_m) 420 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_correct_l2343_234350


namespace NUMINAMATH_CALUDE_solve_a_b_l2343_234314

def U (a : ℝ) : Set ℝ := {2, 3, a^2 + 2*a - 3}

def A (b : ℝ) : Set ℝ := {b, 2}

def complement_U_A (a b : ℝ) : Set ℝ := U a \ A b

theorem solve_a_b (a b : ℝ) : 
  complement_U_A a b = {5} →
  ((a = 2 ∨ a = -4) ∧ b = 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_a_b_l2343_234314


namespace NUMINAMATH_CALUDE_betty_height_in_feet_betty_is_three_feet_tall_l2343_234364

/-- Given a dog's height, Carter's height relative to the dog, and Betty's height relative to Carter,
    calculate Betty's height in feet. -/
theorem betty_height_in_feet (dog_height : ℕ) (carter_ratio : ℕ) (betty_diff : ℕ) : ℕ :=
  let carter_height := dog_height * carter_ratio
  let betty_height_inches := carter_height - betty_diff
  betty_height_inches / 12

/-- Prove that Betty is 3 feet tall given the specific conditions. -/
theorem betty_is_three_feet_tall :
  betty_height_in_feet 24 2 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_betty_height_in_feet_betty_is_three_feet_tall_l2343_234364


namespace NUMINAMATH_CALUDE_vector_inequality_not_always_holds_l2343_234310

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_inequality_not_always_holds :
  ∃ (a b : V), ‖a - b‖ > |‖a‖ - ‖b‖| := by sorry

end NUMINAMATH_CALUDE_vector_inequality_not_always_holds_l2343_234310


namespace NUMINAMATH_CALUDE_triangle_similarity_l2343_234313

-- Define the types for points and triangles
variable (Point : Type) (Triangle : Type)

-- Define the necessary relations and properties
variable (is_scalene : Triangle → Prop)
variable (point_on_segment : Point → Point → Point → Prop)
variable (similar_triangles : Triangle → Triangle → Prop)
variable (point_on_line : Point → Point → Point → Prop)
variable (equal_distance : Point → Point → Point → Point → Prop)

-- State the theorem
theorem triangle_similarity 
  (A B C A₁ B₁ C₁ A₂ B₂ C₂ : Point) 
  (ABC A₁B₁C₁ A₂B₂C₂ : Triangle) :
  is_scalene ABC →
  point_on_segment A₁ B C →
  point_on_segment B₁ C A →
  point_on_segment C₁ A B →
  similar_triangles A₁B₁C₁ ABC →
  point_on_line A₂ B₁ C₁ →
  equal_distance A A₂ A₁ A₂ →
  point_on_line B₂ C₁ A₁ →
  equal_distance B B₂ B₁ B₂ →
  point_on_line C₂ A₁ B₁ →
  equal_distance C C₂ C₁ C₂ →
  similar_triangles A₂B₂C₂ ABC :=
by sorry

end NUMINAMATH_CALUDE_triangle_similarity_l2343_234313


namespace NUMINAMATH_CALUDE_biotechnology_graduates_l2343_234396

theorem biotechnology_graduates (total : ℕ) (job : ℕ) (second_degree : ℕ) (neither : ℕ) :
  total = 73 →
  job = 32 →
  second_degree = 45 →
  neither = 9 →
  ∃ (both : ℕ), both = 13 ∧ job + second_degree - both = total - neither :=
by sorry

end NUMINAMATH_CALUDE_biotechnology_graduates_l2343_234396


namespace NUMINAMATH_CALUDE_rhombus_count_in_triangle_l2343_234303

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ

/-- Represents a rhombus composed of smaller equilateral triangles -/
structure Rhombus where
  num_triangles : ℕ

/-- The number of rhombuses in one direction -/
def rhombuses_in_one_direction (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The main theorem -/
theorem rhombus_count_in_triangle (big_triangle : EquilateralTriangle) 
  (small_triangle : EquilateralTriangle) (rhombus : Rhombus) : 
  big_triangle.side_length = 10 →
  small_triangle.side_length = 1 →
  rhombus.num_triangles = 8 →
  (rhombuses_in_one_direction 7) * 3 = 84 := by
  sorry

#check rhombus_count_in_triangle

end NUMINAMATH_CALUDE_rhombus_count_in_triangle_l2343_234303


namespace NUMINAMATH_CALUDE_least_months_to_triple_l2343_234379

theorem least_months_to_triple (rate : ℝ) (triple : ℝ) : ∃ (n : ℕ), n > 0 ∧ (1 + rate)^n > triple ∧ ∀ (m : ℕ), m > 0 → m < n → (1 + rate)^m ≤ triple :=
  by
  -- Let rate be 0.06 (6%) and triple be 3
  have h1 : rate = 0.06 := by sorry
  have h2 : triple = 3 := by sorry
  
  -- The answer is 19
  use 19
  
  sorry -- Skip the proof

end NUMINAMATH_CALUDE_least_months_to_triple_l2343_234379


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2343_234382

theorem absolute_value_inequality (x : ℝ) : 
  abs (x - 1) + abs (x + 2) < 5 ↔ -3 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2343_234382


namespace NUMINAMATH_CALUDE_ellipse_inscribed_triangle_uniqueness_l2343_234391

/-- Represents an ellipse with semi-major axis a and semi-minor axis 1 -/
def Ellipse (a : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + p.2^2 = 1}

/-- Represents a right-angled isosceles triangle inscribed in the ellipse -/
def InscribedTriangle (a : ℝ) := 
  {triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) | 
    let (A, B, C) := triangle
    B = (0, 1) ∧ 
    A ∈ Ellipse a ∧ 
    C ∈ Ellipse a ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
    (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0}

/-- The main theorem -/
theorem ellipse_inscribed_triangle_uniqueness (a : ℝ) 
  (h1 : a > 1) 
  (h2 : ∃! triangle, triangle ∈ InscribedTriangle a) : 
  1 < a ∧ a ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_inscribed_triangle_uniqueness_l2343_234391


namespace NUMINAMATH_CALUDE_chess_game_probability_l2343_234383

theorem chess_game_probability (p_not_losing p_draw : ℝ) 
  (h1 : p_not_losing = 0.8)
  (h2 : p_draw = 0.5) :
  p_not_losing - p_draw = 0.3 := by
sorry

end NUMINAMATH_CALUDE_chess_game_probability_l2343_234383


namespace NUMINAMATH_CALUDE_game_points_theorem_l2343_234335

theorem game_points_theorem (eric : ℕ) (mark : ℕ) (samanta : ℕ) : 
  mark = eric + eric / 2 →
  samanta = mark + 8 →
  eric + mark + samanta = 32 →
  eric = 6 := by
sorry

end NUMINAMATH_CALUDE_game_points_theorem_l2343_234335
