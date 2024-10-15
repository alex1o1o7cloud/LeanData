import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_determination_l4067_406735

/-- Given a polynomial Q(x) = x^4 - 2x^3 + 3x^2 + kx + m, where k and m are constants,
    prove that if Q(0) = 16 and Q(1) = 2, then Q(x) = x^4 - 2x^3 + 3x^2 - 16x + 16 -/
theorem polynomial_determination (k m : ℝ) : 
  let Q := fun (x : ℝ) => x^4 - 2*x^3 + 3*x^2 + k*x + m
  (Q 0 = 16) → (Q 1 = 2) → 
  (∀ x, Q x = x^4 - 2*x^3 + 3*x^2 - 16*x + 16) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_determination_l4067_406735


namespace NUMINAMATH_CALUDE_distinct_triangles_in_cube_l4067_406701

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where
  /-- The number of vertices in a cube. -/
  num_vertices : ℕ
  /-- The number of edges in a cube. -/
  num_edges : ℕ
  /-- The number of edges meeting at each vertex of a cube. -/
  edges_per_vertex : ℕ
  /-- Assertion that a cube has 8 vertices. -/
  vertices_axiom : num_vertices = 8
  /-- Assertion that a cube has 12 edges. -/
  edges_axiom : num_edges = 12
  /-- Assertion that 3 edges meet at each vertex of a cube. -/
  edges_per_vertex_axiom : edges_per_vertex = 3

/-- A function that calculates the number of distinct triangles in a cube. -/
def count_distinct_triangles (c : Cube) : ℕ :=
  c.num_vertices * (c.edges_per_vertex.choose 2) / 2

/-- Theorem stating that the number of distinct triangles formed by connecting three different edges of a cube, 
    where each set of edges shares a common vertex, is equal to 12. -/
theorem distinct_triangles_in_cube (c : Cube) : 
  count_distinct_triangles c = 12 := by
  sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_cube_l4067_406701


namespace NUMINAMATH_CALUDE_badminton_partitions_l4067_406784

def number_of_partitions (n : ℕ) : ℕ := (n.choose 2) * ((n - 2).choose 2) / 2

theorem badminton_partitions :
  number_of_partitions 6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_badminton_partitions_l4067_406784


namespace NUMINAMATH_CALUDE_andrew_kept_130_stickers_l4067_406762

def andrew_stickers : ℕ := 750
def daniel_stickers : ℕ := 250
def fred_extra_stickers : ℕ := 120

def fred_stickers : ℕ := daniel_stickers + fred_extra_stickers
def shared_stickers : ℕ := daniel_stickers + fred_stickers
def andrew_kept_stickers : ℕ := andrew_stickers - shared_stickers

theorem andrew_kept_130_stickers : andrew_kept_stickers = 130 := by
  sorry

end NUMINAMATH_CALUDE_andrew_kept_130_stickers_l4067_406762


namespace NUMINAMATH_CALUDE_smallest_possible_a_l4067_406759

theorem smallest_possible_a (P : ℤ → ℤ) (a : ℕ) (h_a_pos : a > 0) 
  (h_poly : ∀ x : ℤ, ∃ k : ℤ, P x = k)
  (h_odd : P 1 = a ∧ P 3 = a ∧ P 5 = a ∧ P 7 = a)
  (h_even : P 2 = -a ∧ P 4 = -a ∧ P 6 = -a ∧ P 8 = -a ∧ P 10 = -a) :
  945 ≤ a ∧ ∃ Q : ℤ → ℤ, 
    (∀ x : ℤ, ∃ k : ℤ, Q x = k) ∧
    (Q 2 = 126 ∧ Q 4 = -210 ∧ Q 6 = 126 ∧ Q 8 = -18 ∧ Q 10 = 126) ∧
    (∀ x : ℤ, P x - a = (x-1)*(x-3)*(x-5)*(x-7)*(Q x)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l4067_406759


namespace NUMINAMATH_CALUDE_binomial_coefficient_seven_three_l4067_406749

theorem binomial_coefficient_seven_three : 
  Nat.choose 7 3 = 35 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_seven_three_l4067_406749


namespace NUMINAMATH_CALUDE_intersection_point_circle_tangent_to_l₃_l4067_406776

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2 * x + y = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0
def l₃ (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0

-- Define the intersection point C
def C : ℝ × ℝ := (-2, 4)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + (y - 4)^2 = 9

-- Theorem 1: Prove that C is the intersection of l₁ and l₂
theorem intersection_point : l₁ C.1 C.2 ∧ l₂ C.1 C.2 := by sorry

-- Theorem 2: Prove that the circle equation represents a circle with center C and tangent to l₃
theorem circle_tangent_to_l₃ : 
  ∃ (r : ℝ), r > 0 ∧ 
  (∀ (x y : ℝ), circle_equation x y ↔ (x - C.1)^2 + (y - C.2)^2 = r^2) ∧
  (∃ (x y : ℝ), l₃ x y ∧ circle_equation x y) ∧
  (∀ (x y : ℝ), l₃ x y → (x - C.1)^2 + (y - C.2)^2 ≥ r^2) := by sorry

end NUMINAMATH_CALUDE_intersection_point_circle_tangent_to_l₃_l4067_406776


namespace NUMINAMATH_CALUDE_twentieth_number_in_base6_l4067_406741

-- Define a function to convert decimal to base 6
def decimalToBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

-- State the theorem
theorem twentieth_number_in_base6 :
  decimalToBase6 20 = [3, 2] :=
sorry

end NUMINAMATH_CALUDE_twentieth_number_in_base6_l4067_406741


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l4067_406702

theorem unique_solution_inequality (x : ℝ) :
  (x > 0 ∧ x * Real.sqrt (18 - x) + Real.sqrt (24 * x - x^3) ≥ 18) ↔ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l4067_406702


namespace NUMINAMATH_CALUDE_books_per_shelf_l4067_406786

def library1_total : ℕ := 24850
def library2_total : ℕ := 55300
def library1_leftover : ℕ := 154
def library2_leftover : ℕ := 175

theorem books_per_shelf :
  Int.gcd (library1_total - library1_leftover) (library2_total - library2_leftover) = 441 :=
by sorry

end NUMINAMATH_CALUDE_books_per_shelf_l4067_406786


namespace NUMINAMATH_CALUDE_correct_answer_l4067_406750

theorem correct_answer (x : ℚ) (h : 2 * x = 80) : x / 3 = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l4067_406750


namespace NUMINAMATH_CALUDE_difference_in_half_dollars_l4067_406724

/-- The number of quarters Alice has -/
def alice_quarters (p : ℚ) : ℚ := 8 * p + 2

/-- The number of quarters Bob has -/
def bob_quarters (p : ℚ) : ℚ := 3 * p + 6

/-- Conversion factor from quarters to half-dollars -/
def quarter_to_half_dollar : ℚ := 1 / 2

theorem difference_in_half_dollars (p : ℚ) :
  (alice_quarters p - bob_quarters p) * quarter_to_half_dollar = 2.5 * p - 2 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_half_dollars_l4067_406724


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_one_l4067_406757

theorem sum_of_squares_equals_one 
  (a b c p q r : ℝ) 
  (h1 : a * b = p) 
  (h2 : b * c = q) 
  (h3 : c * a = r) 
  (hp : p ≠ 0) 
  (hq : q ≠ 0) 
  (hr : r ≠ 0) : 
  a^2 + b^2 + c^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_one_l4067_406757


namespace NUMINAMATH_CALUDE_student_count_l4067_406714

theorem student_count (avg_student_age avg_with_teacher teacher_age : ℝ) 
  (h1 : avg_student_age = 15)
  (h2 : avg_with_teacher = 16)
  (h3 : teacher_age = 46) :
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * avg_student_age + teacher_age = (n + 1 : ℝ) * avg_with_teacher ∧
    n = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l4067_406714


namespace NUMINAMATH_CALUDE_p_less_than_q_l4067_406734

/-- For all real x, if P = (x-2)(x-4) and Q = (x-3)^2, then P < Q. -/
theorem p_less_than_q (x : ℝ) : (x - 2) * (x - 4) < (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_p_less_than_q_l4067_406734


namespace NUMINAMATH_CALUDE_out_of_pocket_calculation_l4067_406763

def out_of_pocket (initial_purchase : ℝ) (tv_return : ℝ) (bike_return : ℝ) (toaster_purchase : ℝ) : ℝ :=
  let total_return := tv_return + bike_return
  let sold_bike_cost := bike_return * 1.2
  let sold_bike_price := sold_bike_cost * 0.8
  initial_purchase - total_return - sold_bike_price + toaster_purchase

theorem out_of_pocket_calculation :
  out_of_pocket 3000 700 500 100 = 1420 := by
  sorry

end NUMINAMATH_CALUDE_out_of_pocket_calculation_l4067_406763


namespace NUMINAMATH_CALUDE_sqrt_7_to_6th_power_l4067_406751

theorem sqrt_7_to_6th_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_7_to_6th_power_l4067_406751


namespace NUMINAMATH_CALUDE_successive_price_reduction_l4067_406717

theorem successive_price_reduction (initial_reduction : ℝ) (subsequent_reduction : ℝ) 
  (initial_reduction_percent : initial_reduction = 0.25) 
  (subsequent_reduction_percent : subsequent_reduction = 0.40) : 
  1 - (1 - initial_reduction) * (1 - subsequent_reduction) = 0.55 := by
sorry

end NUMINAMATH_CALUDE_successive_price_reduction_l4067_406717


namespace NUMINAMATH_CALUDE_functional_equation_solution_l4067_406727

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((x - y)^2) = x^2 - 2*y*(f x) + (f y)^2

/-- The main theorem stating that functions satisfying the equation are either
    the identity function or the identity function plus one. -/
theorem functional_equation_solution (f : ℝ → ℝ) (hf : SatisfiesEquation f) :
  (∀ x, f x = x) ∨ (∀ x, f x = x + 1) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l4067_406727


namespace NUMINAMATH_CALUDE_grants_score_l4067_406748

/-- Given the scores of three students on a math test, prove Grant's score. -/
theorem grants_score (hunter_score john_score grant_score : ℕ) : 
  hunter_score = 45 →
  john_score = 2 * hunter_score →
  grant_score = john_score + 10 →
  grant_score = 100 := by
sorry

end NUMINAMATH_CALUDE_grants_score_l4067_406748


namespace NUMINAMATH_CALUDE_xiao_dong_language_understanding_l4067_406705

-- Define propositions
variable (P : Prop) -- Xiao Dong understands English
variable (Q : Prop) -- Xiao Dong understands French

-- Theorem statement
theorem xiao_dong_language_understanding : 
  ¬(P ∧ Q) → (P → ¬Q) :=
by
  sorry

end NUMINAMATH_CALUDE_xiao_dong_language_understanding_l4067_406705


namespace NUMINAMATH_CALUDE_f_at_neg_one_equals_two_l4067_406770

-- Define the function f(x) = -2x
def f (x : ℝ) : ℝ := -2 * x

-- Theorem stating that f(-1) = 2
theorem f_at_neg_one_equals_two : f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_at_neg_one_equals_two_l4067_406770


namespace NUMINAMATH_CALUDE_triangle_condition_l4067_406771

def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + 4 + k^2

theorem triangle_condition (k : ℝ) : 
  (∀ a b c : ℝ, 0 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 3 → 
    f k a + f k b > f k c ∧ 
    f k b + f k c > f k a ∧ 
    f k c + f k a > f k b) ↔ 
  k > 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_condition_l4067_406771


namespace NUMINAMATH_CALUDE_unique_divisibility_condition_l4067_406739

theorem unique_divisibility_condition : 
  ∃! A : ℕ, A < 10 ∧ 45 % A = 0 ∧ (273100 + A * 10 + 6) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisibility_condition_l4067_406739


namespace NUMINAMATH_CALUDE_a_plus_b_value_l4067_406798

theorem a_plus_b_value (a b : ℝ) 
  (ha : |a| = 2) 
  (hb : |b| = 3) 
  (hab : |a-b| = -(a-b)) : 
  a + b = 5 ∨ a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l4067_406798


namespace NUMINAMATH_CALUDE_polygon_area_bounds_l4067_406799

-- Define the type for polygons
structure Polygon :=
  (vertices : List (Int × Int))
  (convex : Bool)
  (area : ℝ)

-- Define the theorem
theorem polygon_area_bounds :
  ∃ (a b c : ℝ) (α : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ α > 0 ∧
    (∀ n : ℕ, ∃ P : Polygon,
      P.convex = true ∧
      P.vertices.length = n ∧
      P.area < a * (n : ℝ)^3) ∧
    (∀ n : ℕ, ∀ P : Polygon,
      P.vertices.length = n →
      P.area ≥ b * (n : ℝ)^2) ∧
    (∀ n : ℕ, ∀ P : Polygon,
      P.vertices.length = n →
      P.area ≥ c * (n : ℝ)^(2 + α)) :=
sorry

end NUMINAMATH_CALUDE_polygon_area_bounds_l4067_406799


namespace NUMINAMATH_CALUDE_equality_from_sum_of_squares_l4067_406788

theorem equality_from_sum_of_squares (a b c : ℝ) :
  a^2 + b^2 + c^2 = a*b + b*c + c*a → a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equality_from_sum_of_squares_l4067_406788


namespace NUMINAMATH_CALUDE_equation_represents_point_l4067_406781

theorem equation_represents_point (x y a b : ℝ) : 
  (x - a)^2 + (y + b)^2 = 0 ↔ x = a ∧ y = -b := by
sorry

end NUMINAMATH_CALUDE_equation_represents_point_l4067_406781


namespace NUMINAMATH_CALUDE_age_difference_l4067_406792

/-- Given three people A, B, and C, where C is 16 years younger than A,
    prove that the difference between the total age of A and B and
    the total age of B and C is 16 years. -/
theorem age_difference (A B C : ℕ) (h : C = A - 16) :
  (A + B) - (B + C) = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l4067_406792


namespace NUMINAMATH_CALUDE_derivative_of_sqrt_at_one_l4067_406706

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem derivative_of_sqrt_at_one :
  deriv f 1 = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_sqrt_at_one_l4067_406706


namespace NUMINAMATH_CALUDE_age_sum_l4067_406779

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 18 years old
  Prove that the sum of their ages is 47 years. -/
theorem age_sum (a b c : ℕ) : 
  b = 18 → 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 47 := by sorry

end NUMINAMATH_CALUDE_age_sum_l4067_406779


namespace NUMINAMATH_CALUDE_x_plus_y_power_2023_l4067_406758

theorem x_plus_y_power_2023 (x y : ℝ) (h : |x - 2| + (y + 3)^2 = 0) : 
  (x + y)^2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_power_2023_l4067_406758


namespace NUMINAMATH_CALUDE_white_tree_count_l4067_406726

/-- Represents the number of crepe myrtle trees of each color in the park -/
structure TreeCount where
  total : ℕ
  pink : ℕ
  red : ℕ
  white : ℕ

/-- The conditions of the park's tree distribution -/
def park_conditions (t : TreeCount) : Prop :=
  t.total = 42 ∧
  t.pink = t.total / 3 ∧
  t.red = 2 ∧
  t.white = t.total - t.pink - t.red ∧
  t.white > t.pink ∧ t.white > t.red

/-- Theorem stating that under the given conditions, the number of white trees is 26 -/
theorem white_tree_count (t : TreeCount) (h : park_conditions t) : t.white = 26 := by
  sorry

end NUMINAMATH_CALUDE_white_tree_count_l4067_406726


namespace NUMINAMATH_CALUDE_angle_WYZ_measure_l4067_406765

-- Define the angle measures
def angle_XYZ : ℝ := 130
def angle_XYW : ℝ := 100

-- Define the theorem
theorem angle_WYZ_measure :
  let angle_WYZ := angle_XYZ - angle_XYW
  angle_WYZ = 30 := by sorry

end NUMINAMATH_CALUDE_angle_WYZ_measure_l4067_406765


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l4067_406766

theorem perfect_square_binomial : ∃ (r s : ℝ), (r * x + s)^2 = 4 * x^2 + 20 * x + 25 := by sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l4067_406766


namespace NUMINAMATH_CALUDE_number_equation_l4067_406725

theorem number_equation (x : ℝ) : 3 * x - 1 = 2 * x ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_number_equation_l4067_406725


namespace NUMINAMATH_CALUDE_sequence_prime_value_l4067_406754

theorem sequence_prime_value (p : ℕ) (a : ℕ → ℤ) : 
  Prime p →
  a 0 = 0 →
  a 1 = 1 →
  (∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - p * a n) →
  (∃ m : ℕ, a m = -1) →
  p = 5 := by
sorry

end NUMINAMATH_CALUDE_sequence_prime_value_l4067_406754


namespace NUMINAMATH_CALUDE_convention_handshakes_eq_990_l4067_406793

/-- The number of handshakes at the Annual Mischief Convention -/
def convention_handshakes : ℕ :=
  let total_gremlins : ℕ := 30
  let total_imps : ℕ := 20
  let unfriendly_gremlins : ℕ := 10
  let friendly_gremlins : ℕ := total_gremlins - unfriendly_gremlins

  let gremlin_handshakes : ℕ := 
    (friendly_gremlins * (friendly_gremlins - 1)) / 2 + 
    unfriendly_gremlins * friendly_gremlins

  let imp_gremlin_handshakes : ℕ := total_imps * total_gremlins

  gremlin_handshakes + imp_gremlin_handshakes

theorem convention_handshakes_eq_990 : convention_handshakes = 990 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_eq_990_l4067_406793


namespace NUMINAMATH_CALUDE_lizard_adoption_rate_l4067_406738

def initial_dogs : ℕ := 30
def initial_cats : ℕ := 28
def initial_lizards : ℕ := 20
def dog_adoption_rate : ℚ := 1/2
def cat_adoption_rate : ℚ := 1/4
def new_pets : ℕ := 13
def total_pets_after_month : ℕ := 65

theorem lizard_adoption_rate : 
  let dogs_adopted := (initial_dogs : ℚ) * dog_adoption_rate
  let cats_adopted := (initial_cats : ℚ) * cat_adoption_rate
  let remaining_dogs := initial_dogs - dogs_adopted.floor
  let remaining_cats := initial_cats - cats_adopted.floor
  let total_before_lizard_adoption := remaining_dogs + remaining_cats + initial_lizards + new_pets
  let lizards_adopted := total_before_lizard_adoption - total_pets_after_month
  lizards_adopted / initial_lizards = 1/5 := by sorry

end NUMINAMATH_CALUDE_lizard_adoption_rate_l4067_406738


namespace NUMINAMATH_CALUDE_find_m_value_l4067_406787

/-- Given x and y values, prove that m = 3 when y is linearly related to x with equation y = 1.3x + 0.8 -/
theorem find_m_value (x : Fin 5 → ℝ) (y : Fin 5 → ℝ) (m : ℝ) : 
  x 0 = 1 ∧ x 1 = 3 ∧ x 2 = 4 ∧ x 3 = 5 ∧ x 4 = 7 ∧
  y 0 = 1 ∧ y 1 = m ∧ y 2 = 2*m+1 ∧ y 3 = 2*m+3 ∧ y 4 = 10 ∧
  (∀ i : Fin 5, y i = 1.3 * x i + 0.8) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_value_l4067_406787


namespace NUMINAMATH_CALUDE_original_price_of_tv_l4067_406777

/-- The original price of a television given a discount and total paid amount -/
theorem original_price_of_tv (discount_rate : ℚ) (total_paid : ℚ) : 
  discount_rate = 5 / 100 → 
  total_paid = 456 → 
  (1 - discount_rate) * 480 = total_paid :=
by sorry

end NUMINAMATH_CALUDE_original_price_of_tv_l4067_406777


namespace NUMINAMATH_CALUDE_max_value_of_a_l4067_406718

theorem max_value_of_a (a b c d : ℝ) 
  (h1 : b + c + d = 3 - a) 
  (h2 : 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2) : 
  ∃ (max_a : ℝ), max_a = 2 ∧ ∀ a', (∃ b' c' d', b' + c' + d' = 3 - a' ∧ 
    2 * b'^2 + 3 * c'^2 + 6 * d'^2 = 5 - a'^2) → a' ≤ max_a :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l4067_406718


namespace NUMINAMATH_CALUDE_largest_n_l4067_406775

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem stating the largest possible value of n -/
theorem largest_n : ∃ (x y : ℕ),
  x < 10 ∧ 
  y < 10 ∧ 
  x ≠ y ∧
  isPrime x ∧ 
  isPrime y ∧ 
  isPrime (10 * y + x) ∧
  1000 ≤ x * y * (10 * y + x) ∧ 
  x * y * (10 * y + x) < 10000 ∧
  ∀ (a b : ℕ), 
    a < 10 → 
    b < 10 → 
    a ≠ b →
    isPrime a → 
    isPrime b → 
    isPrime (10 * b + a) →
    1000 ≤ a * b * (10 * b + a) →
    a * b * (10 * b + a) < 10000 →
    a * b * (10 * b + a) ≤ x * y * (10 * y + x) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_l4067_406775


namespace NUMINAMATH_CALUDE_attainable_tables_count_l4067_406744

/-- Represents a table with signs -/
def Table (m n : ℕ) := Fin (2*m) → Fin (2*n) → Bool

/-- Determines if a table is attainable after one transformation -/
def IsAttainable (m n : ℕ) (t : Table m n) : Prop := sorry

/-- Counts the number of attainable tables -/
def CountAttainableTables (m n : ℕ) : ℕ := sorry

theorem attainable_tables_count (m n : ℕ) :
  CountAttainableTables m n = if m % 2 = 1 ∧ n % 2 = 1 then 2^(m+n-2) else 2^(m+n-1) := by sorry

end NUMINAMATH_CALUDE_attainable_tables_count_l4067_406744


namespace NUMINAMATH_CALUDE_find_M_l4067_406729

theorem find_M : ∃ M : ℕ, (1001 + 1003 + 1005 + 1007 + 1009 = 5100 - M) → M = 75 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l4067_406729


namespace NUMINAMATH_CALUDE_both_sports_fans_l4067_406730

/-- The number of students who like basketball -/
def basketball_fans : ℕ := 9

/-- The number of students who like cricket -/
def cricket_fans : ℕ := 8

/-- The number of students who like basketball or cricket or both -/
def total_fans : ℕ := 11

/-- The number of students who like both basketball and cricket -/
def both_fans : ℕ := basketball_fans + cricket_fans - total_fans

theorem both_sports_fans : both_fans = 6 := by
  sorry

end NUMINAMATH_CALUDE_both_sports_fans_l4067_406730


namespace NUMINAMATH_CALUDE_triangle_theorem_l4067_406743

/-- Given a triangle ABC with sides a, b, c, inradius r, and exradii r₁, r₂, r₃ opposite vertices A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ

/-- Conditions for the triangle -/
def ValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b ∧
  t.a > t.r₁ ∧ t.b > t.r₂ ∧ t.c > t.r₃

/-- Definition of an acute triangle -/
def IsAcute (t : Triangle) : Prop :=
  t.a^2 + t.b^2 > t.c^2 ∧ t.b^2 + t.c^2 > t.a^2 ∧ t.c^2 + t.a^2 > t.b^2

/-- The main theorem to be proved -/
theorem triangle_theorem (t : Triangle) (h : ValidTriangle t) :
  IsAcute t ∧ t.a + t.b + t.c > t.r + t.r₁ + t.r₂ + t.r₃ := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l4067_406743


namespace NUMINAMATH_CALUDE_multiply_48_52_l4067_406713

theorem multiply_48_52 : 48 * 52 = 2496 := by
  sorry

end NUMINAMATH_CALUDE_multiply_48_52_l4067_406713


namespace NUMINAMATH_CALUDE_museum_artifact_distribution_l4067_406755

theorem museum_artifact_distribution (total_wings : Nat) 
  (painting_wings : Nat) (large_painting_wing : Nat) 
  (small_painting_wings : Nat) (paintings_per_small_wing : Nat) 
  (artifact_ratio : Nat) :
  total_wings = 8 →
  painting_wings = 3 →
  large_painting_wing = 1 →
  small_painting_wings = 2 →
  paintings_per_small_wing = 12 →
  artifact_ratio = 4 →
  (total_wings - painting_wings) * 
    ((large_painting_wing + small_painting_wings * paintings_per_small_wing) * artifact_ratio / (total_wings - painting_wings)) = 
  (total_wings - painting_wings) * 20 := by
sorry

end NUMINAMATH_CALUDE_museum_artifact_distribution_l4067_406755


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l4067_406745

theorem solution_set_of_inequality (x : ℝ) :
  (3*x - 1) / (2 - x) ≥ 1 ↔ 3/4 ≤ x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l4067_406745


namespace NUMINAMATH_CALUDE_power_division_equality_l4067_406791

theorem power_division_equality (a : ℝ) : a^11 / a^2 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l4067_406791


namespace NUMINAMATH_CALUDE_tyrones_dimes_l4067_406752

/-- Given Tyrone's coin collection and total money, prove the number of dimes he has. -/
theorem tyrones_dimes (value_without_dimes : ℚ) (total_value : ℚ) (dime_value : ℚ) :
  value_without_dimes = 11 →
  total_value = 13 →
  dime_value = 1 / 10 →
  (total_value - value_without_dimes) / dime_value = 20 :=
by sorry

end NUMINAMATH_CALUDE_tyrones_dimes_l4067_406752


namespace NUMINAMATH_CALUDE_calculation_proof_l4067_406767

theorem calculation_proof :
  (1 * (-8) - (-6) + (-3) = -5) ∧
  (5 / 13 - 3.7 + 8 / 13 + 1.7 = -1) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l4067_406767


namespace NUMINAMATH_CALUDE_total_subjects_is_41_l4067_406710

/-- The total number of subjects taken by Millie, Monica, and Marius -/
def total_subjects (monica_subjects marius_subjects millie_subjects : ℕ) : ℕ :=
  monica_subjects + marius_subjects + millie_subjects

/-- Theorem stating the total number of subjects taken by all three students -/
theorem total_subjects_is_41 :
  ∃ (monica_subjects marius_subjects millie_subjects : ℕ),
    monica_subjects = 10 ∧
    marius_subjects = monica_subjects + 4 ∧
    millie_subjects = marius_subjects + 3 ∧
    total_subjects monica_subjects marius_subjects millie_subjects = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_total_subjects_is_41_l4067_406710


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4067_406711

theorem inequality_solution_set (m : ℝ) (h : m < -3) :
  {x : ℝ | (m + 3) * x^2 - (2 * m + 3) * x + m > 0} = {x : ℝ | 1 < x ∧ x < m / (m + 3)} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4067_406711


namespace NUMINAMATH_CALUDE_first_day_of_month_l4067_406700

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def day_after (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => next_day (day_after d n)

theorem first_day_of_month (d : DayOfWeek) :
  day_after d 29 = DayOfWeek.Wednesday → d = DayOfWeek.Tuesday :=
by
  sorry


end NUMINAMATH_CALUDE_first_day_of_month_l4067_406700


namespace NUMINAMATH_CALUDE_impossible_all_black_l4067_406746

/-- Represents a cell on the chessboard -/
structure Cell where
  row : Fin 8
  col : Fin 8

/-- Represents the color of a cell -/
inductive Color
  | White
  | Black

/-- Represents the chessboard -/
def Chessboard := Cell → Color

/-- Represents a valid inversion operation -/
inductive InversionOperation
  | Horizontal : Fin 8 → Fin 6 → InversionOperation
  | Vertical : Fin 6 → Fin 8 → InversionOperation

/-- Applies an inversion operation to the chessboard -/
def applyInversion (board : Chessboard) (op : InversionOperation) : Chessboard :=
  sorry

/-- Checks if the entire chessboard is black -/
def isAllBlack (board : Chessboard) : Prop :=
  ∀ cell, board cell = Color.Black

/-- Initial all-white chessboard -/
def initialBoard : Chessboard :=
  fun _ => Color.White

/-- Theorem stating the impossibility of making the entire chessboard black -/
theorem impossible_all_black :
  ¬ ∃ (operations : List InversionOperation),
    isAllBlack (operations.foldl applyInversion initialBoard) :=
  sorry

end NUMINAMATH_CALUDE_impossible_all_black_l4067_406746


namespace NUMINAMATH_CALUDE_absolute_value_problem_l4067_406773

theorem absolute_value_problem (a b : ℝ) 
  (ha : |a| = 5) 
  (hb : |b| = 2) :
  (a > b → a + b = 7 ∨ a + b = 3) ∧
  (|a + b| = |a| - |b| → (a = -5 ∧ b = 2) ∨ (a = 5 ∧ b = -2)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l4067_406773


namespace NUMINAMATH_CALUDE_divisors_of_2_pow_48_minus_1_l4067_406783

theorem divisors_of_2_pow_48_minus_1 :
  ∃! (a b : ℕ), 60 < a ∧ a < b ∧ b < 70 ∧ (2^48 - 1) % a = 0 ∧ (2^48 - 1) % b = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_divisors_of_2_pow_48_minus_1_l4067_406783


namespace NUMINAMATH_CALUDE_locus_of_center_P_l4067_406715

-- Define the circle A
def circle_A (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 100

-- Define point B
def point_B : ℝ × ℝ := (3, 0)

-- Define that B is inside circle A
def B_inside_A : Prop := circle_A (point_B.1) (point_B.2)

-- Define circle P
def circle_P (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  -- P passes through B
  (center.1 - point_B.1)^2 + (center.2 - point_B.2)^2 = radius^2 ∧
  -- P is tangent to A internally
  ((center.1 + 3)^2 + center.2^2)^(1/2) + radius = 10

-- Theorem statement
theorem locus_of_center_P :
  ∀ (x y : ℝ), (∃ (r : ℝ), circle_P (x, y) r) ↔ x^2/25 + y^2/16 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_center_P_l4067_406715


namespace NUMINAMATH_CALUDE_b_value_l4067_406708

theorem b_value (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 3) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l4067_406708


namespace NUMINAMATH_CALUDE_smallest_prime_triangle_perimeter_l4067_406764

/-- A triangle with prime side lengths and prime perimeter -/
structure PrimeTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  ha : Nat.Prime a
  hb : Nat.Prime b
  hc : Nat.Prime c
  hab : a < b
  hbc : b < c
  hmin : 5 ≤ a
  htri1 : a + b > c
  htri2 : a + c > b
  htri3 : b + c > a
  hperi : Nat.Prime (a + b + c)

/-- The theorem stating the smallest perimeter of a PrimeTriangle is 23 -/
theorem smallest_prime_triangle_perimeter :
  ∀ t : PrimeTriangle, 23 ≤ t.a + t.b + t.c ∧
  ∃ t0 : PrimeTriangle, t0.a + t0.b + t0.c = 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_triangle_perimeter_l4067_406764


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l4067_406716

def total_balls : ℕ := 9
def white_balls : ℕ := 5
def black_balls : ℕ := 4

def prob_first_white : ℚ := white_balls / total_balls
def prob_second_white : ℚ := (white_balls - 1) / (total_balls - 1)

def prob_two_white : ℚ := prob_first_white * prob_second_white

theorem two_white_balls_probability :
  prob_two_white = 5 / 18 := by sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l4067_406716


namespace NUMINAMATH_CALUDE_tunnel_length_l4067_406728

/-- Calculates the length of a tunnel given the train's length, speed, and time to pass through. -/
theorem tunnel_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_seconds : ℝ) :
  train_length = 300 →
  train_speed_kmh = 54 →
  time_seconds = 100 →
  (train_speed_kmh * 1000 / 3600 * time_seconds) - train_length = 1200 := by
  sorry

#check tunnel_length

end NUMINAMATH_CALUDE_tunnel_length_l4067_406728


namespace NUMINAMATH_CALUDE_sqrt_12_same_type_as_sqrt_3_l4067_406795

def is_same_type (a b : ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ a = k * b

theorem sqrt_12_same_type_as_sqrt_3 :
  let options := [Real.sqrt 8, Real.sqrt 12, Real.sqrt 18, Real.sqrt 6]
  ∃ (x : ℝ), x ∈ options ∧ is_same_type x (Real.sqrt 3) ∧
    ∀ (y : ℝ), y ∈ options → y ≠ x → ¬(is_same_type y (Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_same_type_as_sqrt_3_l4067_406795


namespace NUMINAMATH_CALUDE_box_volume_correct_l4067_406797

/-- The volume of an open box formed from a rectangular sheet -/
def boxVolume (x : ℝ) : ℝ := 4 * x^3 - 56 * x^2 + 192 * x

/-- The properties of the box construction -/
structure BoxProperties where
  sheet_length : ℝ
  sheet_width : ℝ
  corner_cut : ℝ
  max_height : ℝ
  h_length : sheet_length = 16
  h_width : sheet_width = 12
  h_max_height : max_height = 6
  h_corner_cut_range : 0 < corner_cut ∧ corner_cut ≤ max_height

/-- Theorem stating that the boxVolume function correctly calculates the volume of the box -/
theorem box_volume_correct (props : BoxProperties) (x : ℝ) 
    (h_x : 0 < x ∧ x ≤ props.max_height) : 
  boxVolume x = (props.sheet_length - 2*x) * (props.sheet_width - 2*x) * x := by
  sorry

#check box_volume_correct

end NUMINAMATH_CALUDE_box_volume_correct_l4067_406797


namespace NUMINAMATH_CALUDE_intersection_points_are_correct_l4067_406721

/-- The set of intersection points of the given lines -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | (∃ (x y : ℝ), p = (x, y) ∧
    (3 * x - 2 * y = 12 ∨
     2 * x + 4 * y = 8 ∨
     -5 * x + 15 * y = 30 ∨
     x = -3) ∧
    (3 * x - 2 * y = 12 ∨
     2 * x + 4 * y = 8 ∨
     -5 * x + 15 * y = 30 ∨
     x = -3) ∧
    (3 * x - 2 * y = 12 ∨
     2 * x + 4 * y = 8 ∨
     -5 * x + 15 * y = 30 ∨
     x = -3))}

/-- The theorem stating that the intersection points are (4, 0) and (-3, -10.5) -/
theorem intersection_points_are_correct :
  intersection_points = {(4, 0), (-3, -10.5)} :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_are_correct_l4067_406721


namespace NUMINAMATH_CALUDE_max_triangle_chain_length_l4067_406733

/-- Represents a triangle divided into smaller triangles -/
structure DividedTriangle where
  n : ℕ  -- number of parts each side is divided into
  total_triangles : ℕ  -- total number of smaller triangles

/-- Represents a chain of triangles within a divided triangle -/
structure TriangleChain (dt : DividedTriangle) where
  length : ℕ  -- number of triangles in the chain

/-- The property that the total number of smaller triangles is n^2 -/
def total_triangles_prop (dt : DividedTriangle) : Prop :=
  dt.total_triangles = dt.n^2

/-- The theorem stating the maximum length of a triangle chain -/
theorem max_triangle_chain_length (dt : DividedTriangle) 
  (h : total_triangles_prop dt) : 
  ∃ (chain : TriangleChain dt), 
    ∀ (other_chain : TriangleChain dt), other_chain.length ≤ chain.length ∧ 
    chain.length = dt.n^2 - dt.n + 1 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_chain_length_l4067_406733


namespace NUMINAMATH_CALUDE_tetrahedron_volume_relation_l4067_406769

/-- A tetrahedron with volume V, face areas S_i, and distances H_i from an internal point to each face. -/
structure Tetrahedron where
  V : ℝ
  S : Fin 4 → ℝ
  H : Fin 4 → ℝ
  K : ℝ
  h_positive : V > 0
  S_positive : ∀ i, S i > 0
  H_positive : ∀ i, H i > 0
  K_positive : K > 0
  h_relation : ∀ i : Fin 4, S i / (i.val + 1 : ℝ) = K

theorem tetrahedron_volume_relation (t : Tetrahedron) :
  t.H 0 + 2 * t.H 1 + 3 * t.H 2 + 4 * t.H 3 = 3 * t.V / t.K := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_relation_l4067_406769


namespace NUMINAMATH_CALUDE_bicycle_trip_speed_l4067_406774

/-- The speed of the second part of a bicycle trip satisfies an equation based on given conditions. -/
theorem bicycle_trip_speed (v : ℝ) : v > 0 → 0.7 + 10 / v = 17 / 7.99 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_trip_speed_l4067_406774


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l4067_406782

theorem sphere_volume_increase (r₁ r₂ V₁ V₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ = 2 * r₁) 
  (h₃ : V₁ = (4/3) * π * r₁^3) (h₄ : V₂ = (4/3) * π * r₂^3) : V₂ = 8 * V₁ := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l4067_406782


namespace NUMINAMATH_CALUDE_range_of_m_l4067_406772

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*m*x + 4 = 0
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*(m-2)*x - 3*m + 10 = 0

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∧ ¬(q m)) → (2 ≤ m ∧ m < 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l4067_406772


namespace NUMINAMATH_CALUDE_absolute_difference_of_mn_l4067_406760

theorem absolute_difference_of_mn (m n : ℝ) 
  (h1 : m * n = 2) 
  (h2 : m + n = 6) : 
  |m - n| = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_of_mn_l4067_406760


namespace NUMINAMATH_CALUDE_sharon_in_middle_l4067_406789

-- Define the people
inductive Person : Type
| Aaron : Person
| Darren : Person
| Karen : Person
| Maren : Person
| Sharon : Person

-- Define the positions in the train
inductive Position : Type
| First : Position
| Second : Position
| Third : Position
| Fourth : Position
| Fifth : Position

def is_behind (p1 p2 : Position) : Prop :=
  match p1, p2 with
  | Position.Second, Position.Third => True
  | Position.Third, Position.Fourth => True
  | Position.Fourth, Position.Fifth => True
  | _, _ => False

def is_in_front (p1 p2 : Position) : Prop :=
  match p1, p2 with
  | Position.First, Position.Second => True
  | Position.First, Position.Third => True
  | Position.First, Position.Fourth => True
  | Position.Second, Position.Third => True
  | Position.Second, Position.Fourth => True
  | Position.Third, Position.Fourth => True
  | _, _ => False

def at_least_one_between (p1 p2 p3 : Position) : Prop :=
  match p1, p2, p3 with
  | Position.First, Position.Third, Position.Fifth => True
  | Position.First, Position.Fourth, Position.Fifth => True
  | Position.First, Position.Third, Position.Fourth => True
  | Position.Second, Position.Fourth, Position.Fifth => True
  | _, _, _ => False

-- Define the seating arrangement
def seating_arrangement (seat : Person → Position) : Prop :=
  (seat Person.Maren = Position.Fifth) ∧
  (∃ p : Position, is_behind (seat Person.Aaron) p ∧ seat Person.Sharon = p) ∧
  (∃ p : Position, is_in_front (seat Person.Darren) (seat Person.Aaron)) ∧
  (at_least_one_between (seat Person.Karen) (seat Person.Darren) (seat Person.Karen) ∨
   at_least_one_between (seat Person.Darren) (seat Person.Karen) (seat Person.Darren))

theorem sharon_in_middle (seat : Person → Position) :
  seating_arrangement seat → seat Person.Sharon = Position.Third :=
sorry

end NUMINAMATH_CALUDE_sharon_in_middle_l4067_406789


namespace NUMINAMATH_CALUDE_remaining_amount_after_ten_months_l4067_406742

/-- Represents a loan scenario where a person borrows money and pays it back in monthly installments. -/
structure LoanScenario where
  /-- The total amount borrowed -/
  borrowed_amount : ℝ
  /-- The fixed amount paid back each month -/
  monthly_payment : ℝ
  /-- Assumption that the borrowed amount is positive -/
  borrowed_positive : borrowed_amount > 0
  /-- Assumption that the monthly payment is positive -/
  payment_positive : monthly_payment > 0
  /-- After 6 months, half of the borrowed amount has been paid back -/
  half_paid_after_six_months : 6 * monthly_payment = borrowed_amount / 2

/-- Theorem stating that the remaining amount owed after 10 months is equal to
    the borrowed amount minus 10 times the monthly payment. -/
theorem remaining_amount_after_ten_months (scenario : LoanScenario) :
  scenario.borrowed_amount - 10 * scenario.monthly_payment =
  scenario.borrowed_amount - (6 * scenario.monthly_payment + 4 * scenario.monthly_payment) :=
by sorry

end NUMINAMATH_CALUDE_remaining_amount_after_ten_months_l4067_406742


namespace NUMINAMATH_CALUDE_symmetric_x_axis_coords_symmetric_y_axis_coords_l4067_406756

/-- Given a point M with coordinates (x, y) in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to X-axis -/
def symmetricXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Definition of symmetry with respect to Y-axis -/
def symmetricYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

/-- Theorem: The coordinates of the point symmetric to M(x, y) with respect to the X-axis are (x, -y) -/
theorem symmetric_x_axis_coords (M : Point2D) :
  symmetricXAxis M = { x := M.x, y := -M.y } := by sorry

/-- Theorem: The coordinates of the point symmetric to M(x, y) with respect to the Y-axis are (-x, y) -/
theorem symmetric_y_axis_coords (M : Point2D) :
  symmetricYAxis M = { x := -M.x, y := M.y } := by sorry

end NUMINAMATH_CALUDE_symmetric_x_axis_coords_symmetric_y_axis_coords_l4067_406756


namespace NUMINAMATH_CALUDE_coin_flip_probability_l4067_406712

/-- The probability of a coin landing tails up -/
def ProbTails (coin : Nat) : ℚ :=
  match coin with
  | 1 => 3/4  -- Coin A
  | 2 => 1/2  -- Coin B
  | 3 => 1/4  -- Coin C
  | _ => 0    -- Invalid coin number

/-- The probability of the desired outcome -/
def DesiredOutcome : ℚ :=
  ProbTails 1 * ProbTails 2 * (1 - ProbTails 3)

theorem coin_flip_probability :
  DesiredOutcome = 9/32 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l4067_406712


namespace NUMINAMATH_CALUDE_sin_equality_necessary_not_sufficient_l4067_406747

theorem sin_equality_necessary_not_sufficient :
  (∀ A B : ℝ, A = B → Real.sin A = Real.sin B) ∧
  (∃ A B : ℝ, Real.sin A = Real.sin B ∧ A ≠ B) :=
by sorry

end NUMINAMATH_CALUDE_sin_equality_necessary_not_sufficient_l4067_406747


namespace NUMINAMATH_CALUDE_book_arrangement_ways_l4067_406785

theorem book_arrangement_ways (n m : ℕ) (h : n + m = 9) (hn : n = 4) (hm : m = 5) :
  Nat.choose (n + m) n = 126 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_ways_l4067_406785


namespace NUMINAMATH_CALUDE_cricket_players_l4067_406719

theorem cricket_players (total : ℕ) (basketball : ℕ) (both : ℕ) 
  (h1 : total = 880) 
  (h2 : basketball = 600) 
  (h3 : both = 220) : 
  total = (total - basketball + both) + basketball - both :=
by sorry

end NUMINAMATH_CALUDE_cricket_players_l4067_406719


namespace NUMINAMATH_CALUDE_expression_value_l4067_406720

theorem expression_value (a b c d x : ℝ) 
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : |x| = 3) :
  2 * x^2 - (a * b - c - d) + |a * b + 3| = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4067_406720


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4067_406740

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4067_406740


namespace NUMINAMATH_CALUDE_chess_class_percentage_l4067_406753

/-- Proves that 20% of students attend chess class given the conditions of the problem -/
theorem chess_class_percentage (total_students : ℕ) (swimming_students : ℕ) 
  (h1 : total_students = 1000)
  (h2 : swimming_students = 20)
  (h3 : ∀ (chess_percentage : ℚ), 
    chess_percentage * total_students * (1/10) = swimming_students) :
  ∃ (chess_percentage : ℚ), chess_percentage = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_chess_class_percentage_l4067_406753


namespace NUMINAMATH_CALUDE_max_a_value_l4067_406732

-- Define the function f(x) = |x-2| + |x-8|
def f (x : ℝ) : ℝ := |x - 2| + |x - 8|

-- State the theorem
theorem max_a_value : 
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), f x ≥ b) → b ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l4067_406732


namespace NUMINAMATH_CALUDE_max_new_lines_theorem_l4067_406731

/-- The maximum number of new lines formed by connecting intersection points 
    of n lines in a plane, where any two lines intersect and no three lines 
    pass through the same point. -/
def max_new_lines (n : ℕ) : ℚ :=
  (1 / 8 : ℚ) * n * (n - 1) * (n - 2) * (n - 3)

/-- Theorem stating the maximum number of new lines formed by connecting 
    intersection points of n lines in a plane, where any two lines intersect 
    and no three lines pass through the same point. -/
theorem max_new_lines_theorem (n : ℕ) (h : n ≥ 3) :
  let original_lines := n
  let any_two_intersect := true
  let no_three_at_same_point := true
  max_new_lines n = (1 / 8 : ℚ) * n * (n - 1) * (n - 2) * (n - 3) :=
by sorry

end NUMINAMATH_CALUDE_max_new_lines_theorem_l4067_406731


namespace NUMINAMATH_CALUDE_smallest_n_is_five_l4067_406768

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def satisfies_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ is_divisible (n^2 - n + 1) k ∧
  ∃ m : ℕ, 1 ≤ m ∧ m ≤ n + 1 ∧ ¬is_divisible (n^2 - n + 1) m

theorem smallest_n_is_five :
  satisfies_condition 5 ∧
  ∀ n : ℕ, 0 < n ∧ n < 5 → ¬satisfies_condition n :=
sorry

end NUMINAMATH_CALUDE_smallest_n_is_five_l4067_406768


namespace NUMINAMATH_CALUDE_problem_solution_l4067_406723

/-- Given f(x) = ax^2 + bx where a ≠ 0 and f(2) = 0 -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem problem_solution (a b : ℝ) (ha : a ≠ 0) :
  (f a b 2 = 0) →
  /- Part I -/
  (∃! x, f a b x - x = 0) →
  (∀ x, f a b x = -1/2 * x^2 + x) ∧
  /- Part II -/
  (a = 1 →
    (∀ x ∈ Set.Icc (-1) 2, f 1 b x ≤ 3) ∧
    (∀ x ∈ Set.Icc (-1) 2, f 1 b x ≥ -1) ∧
    (∃ x ∈ Set.Icc (-1) 2, f 1 b x = 3) ∧
    (∃ x ∈ Set.Icc (-1) 2, f 1 b x = -1)) ∧
  /- Part III -/
  ((∀ x ≥ 2, f a b x ≥ 2 - a) → a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4067_406723


namespace NUMINAMATH_CALUDE_gcd_228_1995_l4067_406761

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l4067_406761


namespace NUMINAMATH_CALUDE_cos_2A_value_l4067_406794

theorem cos_2A_value (A : Real) (h1 : 0 < A ∧ A < π / 2) 
  (h2 : 3 * Real.cos A - 8 * Real.tan A = 0) : 
  Real.cos (2 * A) = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_cos_2A_value_l4067_406794


namespace NUMINAMATH_CALUDE_curve_C_and_perpendicular_lines_l4067_406778

-- Define the parabola
def parabola (P : ℝ × ℝ) : Prop := P.1^2 = P.2

-- Define the curve C
def curve_C (M : ℝ × ℝ) : Prop := M.1^2 = 4 * M.2

-- Define the relationship between P, D, and M
def point_relationship (P D M : ℝ × ℝ) : Prop :=
  D.1 = 0 ∧ D.2 = P.2 ∧ M.1 = 2 * P.1 ∧ M.2 = P.2

-- Define the line l
def line_l (y : ℝ) : Prop := y = -1

-- Define point F
def point_F : ℝ × ℝ := (0, 1)

-- Define perpendicular lines
def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem curve_C_and_perpendicular_lines :
  ∀ (P D M A B A1 B1 : ℝ × ℝ),
    parabola P →
    point_relationship P D M →
    curve_C A ∧ curve_C B →
    line_l A1.2 ∧ line_l B1.2 →
    A1.1 = A.1 ∧ B1.1 = B.1 →
    perpendicular (A1.1 - point_F.1, A1.2 - point_F.2) (B1.1 - point_F.1, B1.2 - point_F.2) :=
by sorry

end NUMINAMATH_CALUDE_curve_C_and_perpendicular_lines_l4067_406778


namespace NUMINAMATH_CALUDE_jackson_charity_collection_l4067_406737

-- Define the working days in a week
def working_days : ℕ := 5

-- Define the amount collected on Monday and Tuesday
def monday_collection : ℕ := 300
def tuesday_collection : ℕ := 40

-- Define the average collection per 4 houses
def avg_collection_per_4_houses : ℕ := 10

-- Define the number of houses visited on each remaining day
def houses_per_day : ℕ := 88

-- Define the goal for the week
def weekly_goal : ℕ := 1000

-- Theorem statement
theorem jackson_charity_collection :
  monday_collection + tuesday_collection +
  (working_days - 2) * (houses_per_day / 4 * avg_collection_per_4_houses) =
  weekly_goal := by sorry

end NUMINAMATH_CALUDE_jackson_charity_collection_l4067_406737


namespace NUMINAMATH_CALUDE_special_polynomial_property_l4067_406796

/-- The polynomial type representing (1-z)^b₁ · (1-z²)^b₂ · (1-z³)^b₃ ··· (1-z³²)^b₃₂ -/
def SpecialPolynomial (b : Fin 32 → ℕ+) : Polynomial ℚ := sorry

/-- The property that after multiplying out and removing terms with degree > 32, 
    the polynomial equals 1 - 2z -/
def HasSpecialProperty (p : Polynomial ℚ) : Prop := sorry

theorem special_polynomial_property (b : Fin 32 → ℕ+) :
  HasSpecialProperty (SpecialPolynomial b) → b 31 = 2^27 - 2^11 := by sorry

end NUMINAMATH_CALUDE_special_polynomial_property_l4067_406796


namespace NUMINAMATH_CALUDE_remainder_problem_l4067_406709

theorem remainder_problem (n : ℕ) (a b c d : ℕ) 
  (h1 : n = 102 * a + b) 
  (h2 : b < 102) 
  (h3 : n = 103 * c + d) 
  (h4 : d < 103) 
  (h5 : a + d = 20) : 
  b = 20 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l4067_406709


namespace NUMINAMATH_CALUDE_median_intersection_property_l4067_406704

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point
def Point := ℝ × ℝ

-- Define the median intersection point
def medianIntersection (t : Triangle) : Point :=
  sorry

-- Define points M, N, P on the sides of the triangle
def dividePoint (A B : Point) (p q : ℝ) : Point :=
  sorry

-- Theorem statement
theorem median_intersection_property (ABC : Triangle) (p q : ℝ) :
  let O := medianIntersection ABC
  let M := dividePoint ABC.A ABC.B p q
  let N := dividePoint ABC.B ABC.C p q
  let P := dividePoint ABC.C ABC.A p q
  let MNP : Triangle := ⟨M, N, P⟩
  let ANBPCMTriangle : Triangle := 
    ⟨ABC.A, ABC.B, ABC.C⟩  -- This is a placeholder, as we don't have a way to define the intersection points
  (O = medianIntersection MNP) ∧ 
  (O = medianIntersection ANBPCMTriangle) :=
sorry

end NUMINAMATH_CALUDE_median_intersection_property_l4067_406704


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l4067_406790

theorem geometric_progression_solution :
  ∀ (b₁ q : ℚ),
    b₁ + b₁ * q + b₁ * q^2 = 21 →
    b₁^2 + (b₁ * q)^2 + (b₁ * q^2)^2 = 189 →
    ((b₁ = 12 ∧ q = 1/2) ∨ (b₁ = 3 ∧ q = 2)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l4067_406790


namespace NUMINAMATH_CALUDE_range_of_function_l4067_406780

theorem range_of_function (x : ℝ) (h : -π/2 ≤ x ∧ x ≤ π/2) :
  ∃ y, -Real.sqrt 3 ≤ y ∧ y ≤ 2 ∧ y = Real.sqrt 3 * Real.sin x + Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_range_of_function_l4067_406780


namespace NUMINAMATH_CALUDE_train_crossing_time_l4067_406736

/-- Given a train traveling at 72 kmph that passes a man on a 260-meter platform in 17 seconds,
    the time taken for the train to cross the entire platform is 30 seconds. -/
theorem train_crossing_time (train_speed_kmph : ℝ) (man_crossing_time : ℝ) (platform_length : ℝ) :
  train_speed_kmph = 72 →
  man_crossing_time = 17 →
  platform_length = 260 →
  (platform_length + train_speed_kmph * 1000 / 3600 * man_crossing_time) / (train_speed_kmph * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l4067_406736


namespace NUMINAMATH_CALUDE_isabel_songs_total_l4067_406703

/-- The number of country albums Isabel bought -/
def country_albums : ℕ := 6

/-- The number of pop albums Isabel bought -/
def pop_albums : ℕ := 2

/-- The number of songs in each album -/
def songs_per_album : ℕ := 9

/-- The total number of songs Isabel bought -/
def total_songs : ℕ := (country_albums + pop_albums) * songs_per_album

theorem isabel_songs_total : total_songs = 72 := by
  sorry

end NUMINAMATH_CALUDE_isabel_songs_total_l4067_406703


namespace NUMINAMATH_CALUDE_f_shifted_l4067_406707

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- State the theorem
theorem f_shifted (x : ℝ) :
  (1 ≤ x ∧ x ≤ 3) → (2 ≤ x ∧ x ≤ 4) → f (x - 1) = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_shifted_l4067_406707


namespace NUMINAMATH_CALUDE_tangent_circle_height_difference_l4067_406722

/-- A parabola with equation y = x^2 + x -/
def parabola (x : ℝ) : ℝ := x^2 + x

/-- A circle inside the parabola, tangent at two points -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangentPoint1 : ℝ × ℝ
  tangentPoint2 : ℝ × ℝ
  tangent_to_parabola1 : parabola tangentPoint1.1 = tangentPoint1.2
  tangent_to_parabola2 : parabola tangentPoint2.1 = tangentPoint2.2
  on_circle1 : (tangentPoint1.1 - center.1)^2 + (tangentPoint1.2 - center.2)^2 = radius^2
  on_circle2 : (tangentPoint2.1 - center.1)^2 + (tangentPoint2.2 - center.2)^2 = radius^2

/-- The theorem stating the height difference -/
theorem tangent_circle_height_difference (c : TangentCircle) :
  c.center.2 - c.tangentPoint1.2 = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_height_difference_l4067_406722
