import Mathlib

namespace NUMINAMATH_CALUDE_S_is_line_l3970_397036

-- Define the complex number (2+5i)
def a : ℂ := 2 + 5 * Complex.I

-- Define the set S
def S : Set ℂ := {z : ℂ | ∃ (r : ℝ), a * z = r}

-- Theorem stating that S is a line
theorem S_is_line : ∃ (m b : ℝ), S = {z : ℂ | z.im = m * z.re + b} :=
sorry

end NUMINAMATH_CALUDE_S_is_line_l3970_397036


namespace NUMINAMATH_CALUDE_triangle_cookie_cutters_l3970_397077

theorem triangle_cookie_cutters (total_sides : ℕ) (square_cutters : ℕ) (hexagon_cutters : ℕ) 
  (h1 : total_sides = 46)
  (h2 : square_cutters = 4)
  (h3 : hexagon_cutters = 2) :
  ∃ (triangle_cutters : ℕ), 
    triangle_cutters * 3 + square_cutters * 4 + hexagon_cutters * 6 = total_sides ∧ 
    triangle_cutters = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cookie_cutters_l3970_397077


namespace NUMINAMATH_CALUDE_train_length_l3970_397067

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 21 → ∃ length : ℝ, abs (length - 350.07) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3970_397067


namespace NUMINAMATH_CALUDE_square_root_81_l3970_397072

theorem square_root_81 : ∀ (x : ℝ), x^2 = 81 ↔ x = 9 ∨ x = -9 := by sorry

end NUMINAMATH_CALUDE_square_root_81_l3970_397072


namespace NUMINAMATH_CALUDE_b_51_equals_5151_l3970_397078

def a (n : ℕ) : ℕ := n * (n + 1) / 2

def is_not_even (n : ℕ) : Prop := ¬(2 ∣ n)

def b : ℕ → ℕ := sorry

theorem b_51_equals_5151 : b 51 = 5151 := by sorry

end NUMINAMATH_CALUDE_b_51_equals_5151_l3970_397078


namespace NUMINAMATH_CALUDE_smallest_integer_greater_than_half_ninths_l3970_397060

theorem smallest_integer_greater_than_half_ninths : ∀ n : ℤ, (1/2 : ℚ) < (n : ℚ)/9 ↔ n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_greater_than_half_ninths_l3970_397060


namespace NUMINAMATH_CALUDE_other_bill_denomination_l3970_397089

-- Define the total amount spent
def total_spent : ℕ := 80

-- Define the number of $10 bills used
def num_ten_bills : ℕ := 2

-- Define the function to calculate the number of other bills
def num_other_bills (n : ℕ) : ℕ := n + 1

-- Define the theorem
theorem other_bill_denomination :
  ∃ (x : ℕ), 
    x * num_other_bills num_ten_bills + 10 * num_ten_bills = total_spent ∧
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_other_bill_denomination_l3970_397089


namespace NUMINAMATH_CALUDE_total_amount_is_3200_l3970_397050

/-- Proves that the total amount of money divided into two parts is 3200, given the problem conditions. -/
theorem total_amount_is_3200 
  (total : ℝ) -- Total amount of money
  (part1 : ℝ) -- First part of money (invested at 3%)
  (part2 : ℝ) -- Second part of money (invested at 5%)
  (h1 : part1 = 800) -- First part is Rs 800
  (h2 : part2 = total - part1) -- Second part is the remainder
  (h3 : 0.03 * part1 + 0.05 * part2 = 144) -- Total interest is Rs 144
  : total = 3200 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_is_3200_l3970_397050


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l3970_397049

-- Define the concept of a line in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop :=
  -- Two lines are perpendicular if their direction vectors are orthogonal
  let (_, _, _) := l1.direction
  let (_, _, _) := l2.direction
  sorry

-- Define parallelism for lines
def parallel (l1 l2 : Line3D) : Prop :=
  -- Two lines are parallel if their direction vectors are scalar multiples of each other
  let (_, _, _) := l1.direction
  let (_, _, _) := l2.direction
  sorry

-- Theorem statement
theorem perpendicular_parallel_transitive (l1 l2 l3 : Line3D) :
  perpendicular l1 l2 → parallel l2 l3 → perpendicular l1 l3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l3970_397049


namespace NUMINAMATH_CALUDE_average_first_21_multiples_of_5_l3970_397064

theorem average_first_21_multiples_of_5 : 
  let multiples := (fun i => 5 * i) 
  let sum := (List.range 21).map multiples |>.sum
  sum / 21 = 55 := by
sorry


end NUMINAMATH_CALUDE_average_first_21_multiples_of_5_l3970_397064


namespace NUMINAMATH_CALUDE_set_relationship_l3970_397099

-- Define the sets M, N, and P
def M : Set ℝ := {x | (x + 3) / (x - 1) ≤ 0 ∧ x ≠ 1}
def N : Set ℝ := {x | |x + 1| ≤ 2}
def P : Set ℝ := {x | (1/2 : ℝ)^(x^2 + 2*x - 3) ≥ 1}

-- State the theorem
theorem set_relationship : M ⊆ N ∧ N = P := by sorry

end NUMINAMATH_CALUDE_set_relationship_l3970_397099


namespace NUMINAMATH_CALUDE_steves_return_speed_l3970_397009

/-- Proves that given a round trip with specified conditions, the return speed is 10 km/h -/
theorem steves_return_speed (total_distance : ℝ) (total_time : ℝ) (outbound_distance : ℝ) :
  total_distance = 40 →
  total_time = 6 →
  outbound_distance = 20 →
  let outbound_speed := outbound_distance / (total_time / 2)
  let return_speed := 2 * outbound_speed
  return_speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_steves_return_speed_l3970_397009


namespace NUMINAMATH_CALUDE_units_digit_of_27_times_46_l3970_397080

theorem units_digit_of_27_times_46 : (27 * 46) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_27_times_46_l3970_397080


namespace NUMINAMATH_CALUDE_root_inequality_l3970_397001

theorem root_inequality (x₀ : ℝ) (h : x₀ > 0) (hroot : Real.log x₀ - 1 / x₀ = 0) :
  2^x₀ > x₀^(1/2) ∧ x₀^(1/2) > Real.log x₀ := by
  sorry

end NUMINAMATH_CALUDE_root_inequality_l3970_397001


namespace NUMINAMATH_CALUDE_square_roots_of_25_l3970_397014

theorem square_roots_of_25 : Set ℝ := by
  -- Define the set of square roots of 25
  let roots : Set ℝ := {x : ℝ | x^2 = 25}
  
  -- Prove that this set is equal to {-5, 5}
  have h : roots = {-5, 5} := by sorry
  
  -- Return the set of square roots
  exact roots

end NUMINAMATH_CALUDE_square_roots_of_25_l3970_397014


namespace NUMINAMATH_CALUDE_power_difference_equality_l3970_397042

theorem power_difference_equality (x : ℝ) (h : x - 1/x = Real.sqrt 2) :
  x^1023 - 1/x^1023 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equality_l3970_397042


namespace NUMINAMATH_CALUDE_smallest_cube_square_l3970_397062

theorem smallest_cube_square (x : ℕ) (M : ℤ) : x = 11025 ↔ 
  (∀ y : ℕ, y > 0 ∧ y < x → ¬(∃ N : ℤ, 2520 * y = N^3 ∧ ∃ z : ℕ, y = z^2)) ∧ 
  (∃ N : ℤ, 2520 * x = N^3) ∧ 
  (∃ z : ℕ, x = z^2) :=
sorry

end NUMINAMATH_CALUDE_smallest_cube_square_l3970_397062


namespace NUMINAMATH_CALUDE_m_values_l3970_397091

-- Define the sets A and B
def A : Set ℝ := {x | x^2 ≠ 1}
def B (m : ℝ) : Set ℝ := {x | m * x = 1}

-- State the theorem
theorem m_values (h : ∀ m : ℝ, A ∪ B m = A) :
  {m : ℝ | ∃ x, x ∈ B m} = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_m_values_l3970_397091


namespace NUMINAMATH_CALUDE_quadratic_vertex_l3970_397038

/-- The quadratic function f(x) = 3x^2 - 6x + 5 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := 2

/-- Theorem: The vertex of the quadratic function f(x) = 3x^2 - 6x + 5 is at the point (1, 2) -/
theorem quadratic_vertex :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y := by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l3970_397038


namespace NUMINAMATH_CALUDE_rain_on_tuesday_l3970_397011

theorem rain_on_tuesday (rain_monday : ℝ) (no_rain : ℝ) (rain_both : ℝ)
  (h1 : rain_monday = 0.62)
  (h2 : no_rain = 0.28)
  (h3 : rain_both = 0.44) :
  rain_monday + (1 - no_rain) - rain_both = 0.54 := by
sorry

end NUMINAMATH_CALUDE_rain_on_tuesday_l3970_397011


namespace NUMINAMATH_CALUDE_sum_of_exponential_equality_l3970_397007

theorem sum_of_exponential_equality (a b : ℝ) (h : (2 : ℝ) ^ b = (2 : ℝ) ^ (6 - a)) : a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponential_equality_l3970_397007


namespace NUMINAMATH_CALUDE_binary_three_is_three_l3970_397087

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of the number 3 -/
def binary_three : List Bool := [true, true]

theorem binary_three_is_three :
  binary_to_decimal binary_three = 3 := by
  sorry

end NUMINAMATH_CALUDE_binary_three_is_three_l3970_397087


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l3970_397079

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (k : ℕ), k > 105 → ¬(∀ (m : ℕ), Even m → m > 0 → 
    k ∣ (m+1)*(m+3)*(m+5)*(m+7)*(m+9)*(m+11)) ∧ 
  (∀ (m : ℕ), Even m → m > 0 → 
    105 ∣ (m+1)*(m+3)*(m+5)*(m+7)*(m+9)*(m+11)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l3970_397079


namespace NUMINAMATH_CALUDE_f_divisible_by_36_l3970_397081

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem f_divisible_by_36 : ∀ n : ℕ, 36 ∣ f n := by sorry

end NUMINAMATH_CALUDE_f_divisible_by_36_l3970_397081


namespace NUMINAMATH_CALUDE_cube_edge_length_range_l3970_397048

theorem cube_edge_length_range (volume : ℝ) (h : volume = 100) :
  ∃ (edge : ℝ), edge ^ 3 = volume ∧ 4 < edge ∧ edge < 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_range_l3970_397048


namespace NUMINAMATH_CALUDE_three_digit_congruence_count_l3970_397039

theorem three_digit_congruence_count : 
  let count := Finset.filter (fun y => 100 ≤ y ∧ y ≤ 999 ∧ (4325 * y + 692) % 17 = 1403 % 17) (Finset.range 1000)
  ↑count.card = 53 := by sorry

end NUMINAMATH_CALUDE_three_digit_congruence_count_l3970_397039


namespace NUMINAMATH_CALUDE_expression_value_l3970_397057

theorem expression_value (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3970_397057


namespace NUMINAMATH_CALUDE_unit_conversions_l3970_397092

-- Define conversion factors
def cm_to_dm : ℚ := 10
def cm_to_m : ℚ := 100
def kg_to_ton : ℚ := 1000
def g_to_kg : ℚ := 1000
def min_to_hour : ℚ := 60

-- Define the theorem
theorem unit_conversions :
  (4800 / cm_to_dm = 480 ∧ 4800 / cm_to_m = 48) ∧
  (5080 / kg_to_ton = 5 ∧ 5080 % kg_to_ton = 80) ∧
  (8 * g_to_kg + 60 = 8060) ∧
  (3 * min_to_hour + 20 = 200) := by
  sorry


end NUMINAMATH_CALUDE_unit_conversions_l3970_397092


namespace NUMINAMATH_CALUDE_cubic_equation_roots_range_l3970_397044

theorem cubic_equation_roots_range (k : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x^3 - 3*x = k ∧ y^3 - 3*y = k ∧ z^3 - 3*z = k) → 
  -2 < k ∧ k < 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_range_l3970_397044


namespace NUMINAMATH_CALUDE_quadrilateral_area_in_regular_octagon_l3970_397034

-- Define a regular octagon
structure RegularOctagon :=
  (side_length : ℝ)

-- Define the area of a quadrilateral formed by two adjacent vertices and two diagonal intersections
def quadrilateral_area (octagon : RegularOctagon) : ℝ :=
  sorry

-- Theorem statement
theorem quadrilateral_area_in_regular_octagon 
  (octagon : RegularOctagon) 
  (h : octagon.side_length = 5) : 
  quadrilateral_area octagon = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_in_regular_octagon_l3970_397034


namespace NUMINAMATH_CALUDE_factorization_equality_l3970_397054

theorem factorization_equality (m x : ℝ) : 
  m^3 * (x - 2) - m * (x - 2) = m * (x - 2) * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3970_397054


namespace NUMINAMATH_CALUDE_solve_sock_problem_l3970_397069

def sock_problem (initial_pairs : ℕ) (lost_pairs : ℕ) (purchased_pairs : ℕ) (gifted_pairs : ℕ) (final_pairs : ℕ) : Prop :=
  let remaining_pairs := initial_pairs - lost_pairs
  ∃ (donated_fraction : ℚ),
    0 ≤ donated_fraction ∧
    donated_fraction ≤ 1 ∧
    remaining_pairs * (1 - donated_fraction) + purchased_pairs + gifted_pairs = final_pairs ∧
    donated_fraction = 2/3

theorem solve_sock_problem :
  sock_problem 40 4 10 3 25 :=
sorry

end NUMINAMATH_CALUDE_solve_sock_problem_l3970_397069


namespace NUMINAMATH_CALUDE_unit_vectors_collinear_with_vector_l3970_397032

def vector : ℝ × ℝ × ℝ := (-3, -4, 5)

theorem unit_vectors_collinear_with_vector :
  let norm := Real.sqrt ((-3)^2 + (-4)^2 + 5^2)
  let unit_vector₁ : ℝ × ℝ × ℝ := (3 * Real.sqrt 2 / 10, 2 * Real.sqrt 2 / 5, -Real.sqrt 2 / 2)
  let unit_vector₂ : ℝ × ℝ × ℝ := (-3 * Real.sqrt 2 / 10, -2 * Real.sqrt 2 / 5, Real.sqrt 2 / 2)
  (∃ (k : ℝ), vector = (k • unit_vector₁)) ∧
  (∃ (k : ℝ), vector = (k • unit_vector₂)) ∧
  (norm * norm = (-3)^2 + (-4)^2 + 5^2) ∧
  (Real.sqrt 2 * Real.sqrt 2 = 2) ∧
  (∀ (v : ℝ × ℝ × ℝ), (∃ (k : ℝ), vector = (k • v)) → (v = unit_vector₁ ∨ v = unit_vector₂)) :=
by sorry

end NUMINAMATH_CALUDE_unit_vectors_collinear_with_vector_l3970_397032


namespace NUMINAMATH_CALUDE_quad_pyramid_volume_l3970_397017

noncomputable section

/-- A quadrilateral pyramid with a square base -/
structure QuadPyramid where
  /-- Side length of the square base -/
  a : ℝ
  /-- Dihedral angle at edge SA -/
  α : ℝ
  /-- The side length is positive -/
  a_pos : 0 < a
  /-- The dihedral angle is within the valid range -/
  α_range : π / 2 < α ∧ α ≤ 2 * π / 3
  /-- Angles between opposite lateral faces are right angles -/
  opposite_faces_right : True

/-- Volume of the quadrilateral pyramid -/
def volume (p : QuadPyramid) : ℝ := (p.a ^ 3 * |Real.cos p.α|) / 3

/-- Theorem stating the volume of the quadrilateral pyramid -/
theorem quad_pyramid_volume (p : QuadPyramid) : 
  volume p = (p.a ^ 3 * |Real.cos p.α|) / 3 := by sorry

end

end NUMINAMATH_CALUDE_quad_pyramid_volume_l3970_397017


namespace NUMINAMATH_CALUDE_pythagorean_triple_3_4_5_l3970_397008

theorem pythagorean_triple_3_4_5 : 
  ∃ (x : ℕ), x > 0 ∧ 3^2 + 4^2 = x^2 :=
by
  use 5
  sorry

#check pythagorean_triple_3_4_5

end NUMINAMATH_CALUDE_pythagorean_triple_3_4_5_l3970_397008


namespace NUMINAMATH_CALUDE_order_of_powers_l3970_397084

theorem order_of_powers : 4^9 < 6^7 ∧ 6^7 < 3^13 := by
  sorry

end NUMINAMATH_CALUDE_order_of_powers_l3970_397084


namespace NUMINAMATH_CALUDE_factorization_problem_1_l3970_397073

theorem factorization_problem_1 (x y : ℝ) :
  9 - x^2 + 12*x*y - 36*y^2 = (3 + x - 6*y) * (3 - x + 6*y) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l3970_397073


namespace NUMINAMATH_CALUDE_solve_for_k_l3970_397056

theorem solve_for_k : ∃ k : ℤ, 2^5 - 10 = 5^2 + k ∧ k = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l3970_397056


namespace NUMINAMATH_CALUDE_difference_of_squares_and_product_l3970_397083

theorem difference_of_squares_and_product (a b : ℝ) 
  (h1 : a^2 + b^2 = 150) 
  (h2 : a * b = 25) : 
  |a - b| = 10 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_and_product_l3970_397083


namespace NUMINAMATH_CALUDE_certain_number_proof_l3970_397024

theorem certain_number_proof (n : ℕ) : 
  n % 10 = 6 ∧ 1442 % 10 = 12 → n = 1446 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3970_397024


namespace NUMINAMATH_CALUDE_lab_budget_remaining_l3970_397006

/-- Given a laboratory budget and expenses, calculate the remaining budget. -/
theorem lab_budget_remaining (budget : ℚ) (flask_cost : ℚ) : 
  budget = 325 →
  flask_cost = 150 →
  let test_tube_cost := (2 / 3) * flask_cost
  let safety_gear_cost := (1 / 2) * test_tube_cost
  let total_expense := flask_cost + test_tube_cost + safety_gear_cost
  budget - total_expense = 25 := by sorry

end NUMINAMATH_CALUDE_lab_budget_remaining_l3970_397006


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3970_397097

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (3 * i + 1) / (1 - i)
  Complex.im z = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3970_397097


namespace NUMINAMATH_CALUDE_sqrt_360000_l3970_397022

theorem sqrt_360000 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_l3970_397022


namespace NUMINAMATH_CALUDE_simplify_expression_l3970_397046

theorem simplify_expression (b : ℝ) : 3*b*(3*b^3 + 2*b^2) - 2*b^2 + 5 = 9*b^4 + 6*b^3 - 2*b^2 + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3970_397046


namespace NUMINAMATH_CALUDE_hazel_lemonade_cups_l3970_397053

/-- The number of cups of lemonade Hazel sold to kids on bikes -/
def cups_sold_to_kids : ℕ := 18

/-- The number of cups of lemonade Hazel made -/
def total_cups : ℕ := 56

theorem hazel_lemonade_cups : 
  total_cups = 56 ∧
  (total_cups / 2 : ℕ) + cups_sold_to_kids + (cups_sold_to_kids / 2 : ℕ) + 1 = total_cups :=
by sorry


end NUMINAMATH_CALUDE_hazel_lemonade_cups_l3970_397053


namespace NUMINAMATH_CALUDE_min_value_theorem_l3970_397088

theorem min_value_theorem (a b : ℝ) (ha : a > 0) 
  (h : ∀ x > 0, (a * x - 2) * (-x^2 - b * x + 4) ≤ 0) : 
  ∃ m : ℝ, m = 2 * Real.sqrt 2 ∧ ∀ b, b + 3 / a ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3970_397088


namespace NUMINAMATH_CALUDE_angle_QRS_is_150_degrees_l3970_397033

/-- A quadrilateral PQRS with specific side lengths and angles -/
structure Quadrilateral :=
  (PQ : ℝ)
  (RS : ℝ)
  (PS : ℝ)
  (angle_QPS : ℝ)
  (angle_RSP : ℝ)

/-- The theorem stating the condition for ∠QRS in the given quadrilateral -/
theorem angle_QRS_is_150_degrees (q : Quadrilateral) 
  (h1 : q.PQ = 40)
  (h2 : q.RS = 20)
  (h3 : q.PS = 60)
  (h4 : q.angle_QPS = 60)
  (h5 : q.angle_RSP = 60) :
  ∃ (angle_QRS : ℝ), angle_QRS = 150 := by
  sorry

end NUMINAMATH_CALUDE_angle_QRS_is_150_degrees_l3970_397033


namespace NUMINAMATH_CALUDE_company_kw_price_percentage_l3970_397058

/-- The price of Company KW as a percentage of the combined assets of Companies A and B -/
theorem company_kw_price_percentage (P A B : ℝ) 
  (h1 : P = 1.30 * A) 
  (h2 : P = 2.00 * B) : 
  ∃ (ε : ℝ), abs (P / (A + B) - 0.7879) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_company_kw_price_percentage_l3970_397058


namespace NUMINAMATH_CALUDE_cube_side_length_l3970_397071

/-- Proves that given the cost of paint, coverage, and total cost to paint a cube,
    the side length of the cube is 8 feet. -/
theorem cube_side_length 
  (paint_cost : ℝ) 
  (paint_coverage : ℝ) 
  (total_cost : ℝ) 
  (h1 : paint_cost = 36.50)
  (h2 : paint_coverage = 16)
  (h3 : total_cost = 876) :
  ∃ (s : ℝ), s = 8 ∧ 
  total_cost = (6 * s^2 / paint_coverage) * paint_cost :=
sorry

end NUMINAMATH_CALUDE_cube_side_length_l3970_397071


namespace NUMINAMATH_CALUDE_samantha_born_in_1975_l3970_397059

-- Define the year of the first AMC 8
def first_amc8_year : ℕ := 1983

-- Define Samantha's age when she took the seventh AMC 8
def samantha_age_seventh_amc8 : ℕ := 14

-- Define the number of years between first and seventh AMC 8
def years_between_first_and_seventh : ℕ := 6

-- Define the year Samantha took the seventh AMC 8
def samantha_seventh_amc8_year : ℕ := first_amc8_year + years_between_first_and_seventh

-- Define Samantha's birth year
def samantha_birth_year : ℕ := samantha_seventh_amc8_year - samantha_age_seventh_amc8

-- Theorem to prove
theorem samantha_born_in_1975 : samantha_birth_year = 1975 := by
  sorry

end NUMINAMATH_CALUDE_samantha_born_in_1975_l3970_397059


namespace NUMINAMATH_CALUDE_holiday_savings_l3970_397021

theorem holiday_savings (sam_savings : ℕ) (total_savings : ℕ) (victory_savings : ℕ) : 
  sam_savings = 1000 →
  total_savings = 1900 →
  victory_savings < sam_savings →
  victory_savings = total_savings - sam_savings →
  sam_savings - victory_savings = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_holiday_savings_l3970_397021


namespace NUMINAMATH_CALUDE_total_discount_is_65_percent_l3970_397040

/-- Represents the discount percentage as a real number between 0 and 1 -/
def Discount := { d : ℝ // 0 ≤ d ∧ d ≤ 1 }

/-- The half-price sale discount -/
def half_price_discount : Discount := ⟨0.5, by norm_num⟩

/-- The additional coupon discount -/
def coupon_discount : Discount := ⟨0.3, by norm_num⟩

/-- Calculates the final price after applying two successive discounts -/
def apply_discounts (d1 d2 : Discount) : ℝ := (1 - d1.val) * (1 - d2.val)

/-- The theorem to be proved -/
theorem total_discount_is_65_percent :
  apply_discounts half_price_discount coupon_discount = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_total_discount_is_65_percent_l3970_397040


namespace NUMINAMATH_CALUDE_tim_necklace_profit_l3970_397037

/-- Represents the properties of a necklace type -/
structure NecklaceType where
  charms : ℕ
  charmCost : ℕ
  sellingPrice : ℕ

/-- Calculates the profit for a single necklace -/
def profit (n : NecklaceType) : ℕ :=
  n.sellingPrice - n.charms * n.charmCost

/-- Represents the sales information -/
structure Sales where
  typeA : NecklaceType
  typeB : NecklaceType
  soldA : ℕ
  soldB : ℕ

/-- Calculates the total profit from all sales -/
def totalProfit (s : Sales) : ℕ :=
  s.soldA * profit s.typeA + s.soldB * profit s.typeB

/-- Tim's necklace business theorem -/
theorem tim_necklace_profit :
  let s : Sales := {
    typeA := { charms := 8, charmCost := 10, sellingPrice := 125 },
    typeB := { charms := 12, charmCost := 18, sellingPrice := 280 },
    soldA := 45,
    soldB := 35
  }
  totalProfit s = 4265 := by sorry

end NUMINAMATH_CALUDE_tim_necklace_profit_l3970_397037


namespace NUMINAMATH_CALUDE_josh_marbles_l3970_397094

/-- The number of marbles Josh has after losing some and giving away half of the remainder --/
def final_marbles (initial : ℕ) (lost : ℕ) : ℕ :=
  let remaining := initial - lost
  remaining - (remaining / 2)

/-- Theorem stating that Josh ends up with 103 marbles --/
theorem josh_marbles : final_marbles 320 115 = 103 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_l3970_397094


namespace NUMINAMATH_CALUDE_floor_length_is_20_l3970_397030

/-- Proves that the length of a rectangular floor is 20 meters given the specified conditions -/
theorem floor_length_is_20 (breadth : ℝ) (length : ℝ) (area : ℝ) (total_cost : ℝ) (rate : ℝ) : 
  length = breadth + 2 * breadth →  -- length is 200% more than breadth
  area = length * breadth →         -- area formula
  area = total_cost / rate →        -- area from cost and rate
  total_cost = 400 →                -- given total cost
  rate = 3 →                        -- given rate per square meter
  length = 20 := by
sorry

end NUMINAMATH_CALUDE_floor_length_is_20_l3970_397030


namespace NUMINAMATH_CALUDE_tattoo_ratio_l3970_397004

def jason_arm_tattoos : ℕ := 2
def jason_leg_tattoos : ℕ := 3
def jason_arms : ℕ := 2
def jason_legs : ℕ := 2
def adam_tattoos : ℕ := 23

def jason_total_tattoos : ℕ :=
  jason_arm_tattoos * jason_arms + jason_leg_tattoos * jason_legs

theorem tattoo_ratio :
  ∃ (m : ℕ), adam_tattoos = m * jason_total_tattoos + 3 ∧
  adam_tattoos.gcd jason_total_tattoos = 1 := by
  sorry

end NUMINAMATH_CALUDE_tattoo_ratio_l3970_397004


namespace NUMINAMATH_CALUDE_cannot_form_triangle_l3970_397016

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
axiom triangle_inequality (a b c : ℝ) : a > 0 ∧ b > 0 ∧ c > 0 → a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: The set of line segments (2, 2, 6) cannot form a triangle -/
theorem cannot_form_triangle : ¬ can_form_triangle 2 2 6 := by
  sorry


end NUMINAMATH_CALUDE_cannot_form_triangle_l3970_397016


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3970_397082

/-- Given a point A(2,1) and a line 2x-y+3=0, prove that 2x-y-3=0 is the equation of the line
    passing through A and parallel to 2x-y+3=0 -/
theorem parallel_line_through_point (x y : ℝ) : 
  (2 * x - y + 3 = 0) →  -- Given line equation
  (2 * 2 - 1 = y) →      -- Point A(2,1) satisfies the new line equation
  (2 * x - y - 3 = 0) →  -- New line equation
  (∃ k : ℝ, k ≠ 0 ∧ (2 : ℝ) / 1 = (2 : ℝ) / 1) -- Parallel lines have equal slopes
  := by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3970_397082


namespace NUMINAMATH_CALUDE_third_term_is_nine_l3970_397090

/-- A geometric sequence where the first term is 4, the second term is 6, and the third term is x -/
def geometric_sequence (x : ℝ) : ℕ → ℝ
| 0 => 4
| 1 => 6
| 2 => x
| (n + 3) => sorry

/-- Theorem: In the given geometric sequence, the third term x is equal to 9 -/
theorem third_term_is_nine :
  ∃ x : ℝ, (∀ n : ℕ, geometric_sequence x (n + 1) = (geometric_sequence x n) * (geometric_sequence x 1 / geometric_sequence x 0)) → x = 9 :=
sorry

end NUMINAMATH_CALUDE_third_term_is_nine_l3970_397090


namespace NUMINAMATH_CALUDE_bertha_initial_balls_l3970_397029

def tennis_balls (initial_balls : ℕ) : Prop :=
  let worn_out := 20 / 10
  let lost := 20 / 5
  let bought := (20 / 4) * 3
  initial_balls - worn_out - lost + bought - 1 = 10

theorem bertha_initial_balls :
  ∃ (initial_balls : ℕ), tennis_balls initial_balls ∧ initial_balls = 2 :=
sorry

end NUMINAMATH_CALUDE_bertha_initial_balls_l3970_397029


namespace NUMINAMATH_CALUDE_average_weight_increase_l3970_397035

theorem average_weight_increase (initial_weight : ℝ) : 
  let initial_average := (initial_weight + 65) / 2
  let new_average := (initial_weight + 74) / 2
  new_average - initial_average = 4.5 := by
sorry


end NUMINAMATH_CALUDE_average_weight_increase_l3970_397035


namespace NUMINAMATH_CALUDE_graph_quadrants_l3970_397041

/-- Given a > 1 and b < -1, the graph of f(x) = a^x + b intersects Quadrants I, III, and IV, but not Quadrant II -/
theorem graph_quadrants (a b : ℝ) (ha : a > 1) (hb : b < -1) :
  let f : ℝ → ℝ := λ x ↦ a^x + b
  (∃ x y, x > 0 ∧ y > 0 ∧ f x = y) ∧  -- Quadrant I
  (∃ x y, x < 0 ∧ y < 0 ∧ f x = y) ∧  -- Quadrant III
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) ∧  -- Quadrant IV
  (∀ x y, ¬(x < 0 ∧ y > 0 ∧ f x = y))  -- Not in Quadrant II
  := by sorry

end NUMINAMATH_CALUDE_graph_quadrants_l3970_397041


namespace NUMINAMATH_CALUDE_sum_squares_five_consecutive_integers_l3970_397013

theorem sum_squares_five_consecutive_integers (n : ℤ) :
  ∃ k : ℤ, (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 5 * k :=
by sorry

end NUMINAMATH_CALUDE_sum_squares_five_consecutive_integers_l3970_397013


namespace NUMINAMATH_CALUDE_stratified_sampling_survey_l3970_397047

theorem stratified_sampling_survey (young_population middle_aged_population elderly_population : ℕ)
  (elderly_sampled : ℕ) (young_sampled : ℕ) :
  young_population = 800 →
  middle_aged_population = 1600 →
  elderly_population = 1400 →
  elderly_sampled = 70 →
  (elderly_sampled : ℚ) / elderly_population = (young_sampled : ℚ) / young_population →
  young_sampled = 40 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_survey_l3970_397047


namespace NUMINAMATH_CALUDE_chord_length_theorem_l3970_397019

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 25

-- Theorem statement
theorem chord_length_theorem :
  ∃ (chord_length : ℝ),
    (∀ (x y : ℝ), line_equation x y → circle_equation x y → 
      chord_length = 4 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l3970_397019


namespace NUMINAMATH_CALUDE_megan_removed_two_albums_l3970_397061

/-- Calculates the number of albums removed from a shopping cart. -/
def albums_removed (initial_albums : ℕ) (songs_per_album : ℕ) (total_songs_bought : ℕ) : ℕ :=
  initial_albums - (total_songs_bought / songs_per_album)

/-- Proves that Megan removed 2 albums from her shopping cart. -/
theorem megan_removed_two_albums :
  albums_removed 8 7 42 = 2 := by
  sorry

end NUMINAMATH_CALUDE_megan_removed_two_albums_l3970_397061


namespace NUMINAMATH_CALUDE_discount_price_l3970_397043

theorem discount_price (a : ℝ) :
  let discounted_price := a
  let discount_rate := 0.3
  let original_price := discounted_price / (1 - discount_rate)
  original_price = 10 / 7 * a :=
by sorry

end NUMINAMATH_CALUDE_discount_price_l3970_397043


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3970_397068

theorem diophantine_equation_solutions (k : ℤ) :
  (k > 7 → ∃ (x y : ℕ), 5 * x + 3 * y = k) ∧
  (k > 15 → ∃ (x y : ℕ+), 5 * x + 3 * y = k) ∧
  (∀ N : ℤ, (∀ k > N, ∃ (x y : ℕ+), 5 * x + 3 * y = k) → N ≥ 15) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3970_397068


namespace NUMINAMATH_CALUDE_collinear_with_a_l3970_397074

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

/-- Given vector a = (1, 2), prove that (k, 2k) is collinear with a for any non-zero real k -/
theorem collinear_with_a (k : ℝ) (hk : k ≠ 0) : 
  collinear (1, 2) (k, 2*k) := by
sorry

end NUMINAMATH_CALUDE_collinear_with_a_l3970_397074


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l3970_397045

-- Define the triangle and circle
def triangle_ratio : Vector ℝ 3 := ⟨[6, 8, 10], by simp⟩
def circle_radius : ℝ := 5

-- Theorem statement
theorem triangle_area_in_circle (sides : Vector ℝ 3) (r : ℝ) 
  (h1 : sides = triangle_ratio) 
  (h2 : r = circle_radius) : 
  ∃ (a b c : ℝ), 
    a * sides[0] = b * sides[1] ∧ 
    a * sides[0] = c * sides[2] ∧
    b * sides[1] = c * sides[2] ∧
    (a * sides[0])^2 + (b * sides[1])^2 = (c * sides[2])^2 ∧
    c * sides[2] = 2 * r ∧
    (1/2) * (a * sides[0]) * (b * sides[1]) = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_in_circle_l3970_397045


namespace NUMINAMATH_CALUDE_solution_g_less_than_6_range_of_a_l3970_397025

-- Define the functions f and g
def f (a x : ℝ) : ℝ := -|x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1| + |2*x + 4|

-- Theorem for the solution of g(x) < 6
theorem solution_g_less_than_6 : 
  {x : ℝ | g x < 6} = Set.Ioo (-9/4) (3/4) := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x₁, ∃ x₂, -g x₁ = f a x₂} = Set.Ici (-5) := by sorry

end NUMINAMATH_CALUDE_solution_g_less_than_6_range_of_a_l3970_397025


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3970_397098

theorem polynomial_factorization (x y : ℝ) : x^3 * y - 4 * x * y^3 = x * y * (x + 2 * y) * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3970_397098


namespace NUMINAMATH_CALUDE_factor_sum_l3970_397012

theorem factor_sum (a b : ℝ) : 
  (∃ m n : ℝ, ∀ x : ℝ, x^4 + a*x^2 + b = (x^2 + 2*x + 5) * (x^2 + m*x + n)) →
  a + b = 31 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l3970_397012


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3970_397000

theorem not_sufficient_nor_necessary (x y : ℝ) : 
  (∃ a b : ℝ, a > b ∧ ¬(|a| > |b|)) ∧ 
  (∃ c d : ℝ, |c| > |d| ∧ ¬(c > d)) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3970_397000


namespace NUMINAMATH_CALUDE_cross_product_example_l3970_397027

/-- The cross product of two 3D vectors -/
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1 - u.1 * v.2.2,
   u.1 * v.2.1 - u.2.1 * v.1)

theorem cross_product_example :
  let u : ℝ × ℝ × ℝ := (3, 2, 4)
  let v : ℝ × ℝ × ℝ := (4, 3, -1)
  cross_product u v = (-14, 19, 1) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_example_l3970_397027


namespace NUMINAMATH_CALUDE_second_smallest_divisor_l3970_397052

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem second_smallest_divisor (n : ℕ) : 
  (is_divisible (n + 3) 12 ∧ 
   is_divisible (n + 3) 35 ∧ 
   is_divisible (n + 3) 40) →
  (∀ m : ℕ, m < n → ¬(is_divisible (m + 3) 12 ∧ 
                      is_divisible (m + 3) 35 ∧ 
                      is_divisible (m + 3) 40)) →
  (∃ d : ℕ, d ≠ 1 ∧ is_divisible (n + 3) d ∧ 
   d ≠ 12 ∧ d ≠ 35 ∧ d ≠ 40 ∧
   (∀ k : ℕ, 1 < k → k < d → ¬is_divisible (n + 3) k)) →
  is_divisible (n + 3) 3 :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_divisor_l3970_397052


namespace NUMINAMATH_CALUDE_ice_cream_consumption_l3970_397086

theorem ice_cream_consumption (friday_amount saturday_amount : Real) 
  (h1 : friday_amount = 3.25)
  (h2 : saturday_amount = 0.25) :
  friday_amount + saturday_amount = 3.50 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_consumption_l3970_397086


namespace NUMINAMATH_CALUDE_disjunction_true_l3970_397028

theorem disjunction_true : 
  (∀ x : ℝ, x < 0 → 2^x > x) ∨ (∃ x : ℝ, x^2 + x + 1 < 0) := by sorry

end NUMINAMATH_CALUDE_disjunction_true_l3970_397028


namespace NUMINAMATH_CALUDE_luke_money_in_january_l3970_397051

/-- The amount of money Luke had in January -/
def initial_amount : ℕ := sorry

/-- The amount Luke spent -/
def spent : ℕ := 11

/-- The amount Luke received from his mom -/
def received : ℕ := 21

/-- The amount Luke has now -/
def current_amount : ℕ := 58

theorem luke_money_in_january :
  initial_amount = 48 :=
by
  have h : initial_amount - spent + received = current_amount := by sorry
  sorry

end NUMINAMATH_CALUDE_luke_money_in_january_l3970_397051


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3970_397076

theorem fraction_inequality_solution_set : 
  {x : ℝ | x / (x + 1) < 0} = Set.Ioo (-1) 0 := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l3970_397076


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_l3970_397066

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallelToPlane : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_and_line (a b : Line) (α : Plane) :
  (perpendicularToPlane a α ∧ perpendicular a b) → parallelToPlane b α := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_l3970_397066


namespace NUMINAMATH_CALUDE_journey_speed_proof_l3970_397093

/-- Proves that given a journey of approximately 3 km divided into three equal parts,
    where the first part is traveled at 3 km/hr, the second at 4 km/hr,
    and the total journey takes 47 minutes, the speed of the third part must be 5 km/hr. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 3.000000000000001)
  (h2 : total_time = 47 / 60) -- Convert 47 minutes to hours
  (h3 : ∃ (d : ℝ), d > 0 ∧ 3 * d = total_distance) -- Equal distances for each part
  (h4 : ∃ (v : ℝ), v > 0 ∧ 1 / 3 + 1 / 4 + 1 / v = total_time) -- Time equation
  : ∃ (v : ℝ), v = 5 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l3970_397093


namespace NUMINAMATH_CALUDE_log_division_simplification_l3970_397026

theorem log_division_simplification :
  Real.log 16 / Real.log (1/16) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l3970_397026


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l3970_397085

-- Define the line l: x + y = 0
def line_l (x y : ℝ) : Prop := x + y = 0

-- Define the symmetric point of (-2, 0) with respect to line l
def symmetric_point (a b : ℝ) : Prop :=
  (b - 0) / (a + 2) = -1 ∧ (a - (-2)) / 2 + b / 2 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 2

-- Define the tangency condition
def is_tangent (x y : ℝ) : Prop := line_l x y ∧ circle_equation x y

-- Theorem statement
theorem circle_tangent_to_line :
  ∃ (x y : ℝ), is_tangent x y ∧
  ∃ (a b : ℝ), symmetric_point a b ∧
  (x - a)^2 + (y - b)^2 = ((a + b) / Real.sqrt 2)^2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l3970_397085


namespace NUMINAMATH_CALUDE_equilateral_triangle_from_sequences_l3970_397003

/-- Given a triangle ABC where:
    - The angles A, B, C form an arithmetic sequence
    - The sides a, b, c (opposite to angles A, B, C respectively) form a geometric sequence
    Prove that the triangle is equilateral -/
theorem equilateral_triangle_from_sequences (A B C a b c : ℝ) : 
  (∃ d : ℝ, B - A = d ∧ C - B = d) →  -- Angles form arithmetic sequence
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- Sides form geometric sequence
  A + B + C = π →                     -- Sum of angles in a triangle
  A > 0 ∧ B > 0 ∧ C > 0 →             -- Positive angles
  a > 0 ∧ b > 0 ∧ c > 0 →             -- Positive side lengths
  (A = π/3 ∧ B = π/3 ∧ C = π/3) :=    -- Triangle is equilateral
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_from_sequences_l3970_397003


namespace NUMINAMATH_CALUDE_work_completion_time_l3970_397096

theorem work_completion_time (a b c : ℝ) 
  (h1 : a + b + c = 1 / 4)  -- a, b, and c together finish in 4 days
  (h2 : b = 1 / 18)         -- b alone finishes in 18 days
  (h3 : c = 1 / 9)          -- c alone finishes in 9 days
  : a = 1 / 12 :=           -- a alone finishes in 12 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3970_397096


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3970_397055

def A (a : ℝ) : Set ℝ := {4, 2, a^2}
def B (a : ℝ) : Set ℝ := {1, a}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {1} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3970_397055


namespace NUMINAMATH_CALUDE_cubic_sum_greater_than_mixed_product_l3970_397005

theorem cubic_sum_greater_than_mixed_product (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : 
  a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_greater_than_mixed_product_l3970_397005


namespace NUMINAMATH_CALUDE_problem_statement_l3970_397010

theorem problem_statement (a b : ℝ) : 
  |a + b - 1| + Real.sqrt (2 * a + b - 2) = 0 → (b - a)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3970_397010


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l3970_397070

theorem smallest_n_divisibility : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → (¬(24 ∣ m^2) ∨ ¬(540 ∣ m^3))) ∧ 
  24 ∣ n^2 ∧ 
  540 ∣ n^3 ∧ 
  n = 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l3970_397070


namespace NUMINAMATH_CALUDE_stating_count_five_digit_divisible_by_12_is_72000_l3970_397002

/-- 
A function that counts the number of positive five-digit integers divisible by 12.
-/
def count_five_digit_divisible_by_12 : ℕ :=
  sorry

/-- 
Theorem stating that the count of positive five-digit integers divisible by 12 is 72000.
-/
theorem count_five_digit_divisible_by_12_is_72000 : 
  count_five_digit_divisible_by_12 = 72000 :=
by sorry

end NUMINAMATH_CALUDE_stating_count_five_digit_divisible_by_12_is_72000_l3970_397002


namespace NUMINAMATH_CALUDE_product_of_integers_l3970_397075

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 18)
  (diff_squares_eq : x^2 - y^2 = 36) :
  x * y = 80 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l3970_397075


namespace NUMINAMATH_CALUDE_water_tank_capacity_water_tank_capacity_proof_l3970_397065

/-- Proves that a cylindrical water tank holds 75 liters when full -/
theorem water_tank_capacity : ℝ → Prop :=
  fun c => 
    (∃ w : ℝ, w / c = 1 / 3 ∧ (w + 5) / c = 2 / 5) → c = 75

/-- The proof of the water tank capacity theorem -/
theorem water_tank_capacity_proof : water_tank_capacity 75 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_water_tank_capacity_proof_l3970_397065


namespace NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l3970_397015

theorem cos_squared_alpha_minus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 1 / 3) :
  Real.cos (α - Real.pi / 4) ^ 2 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_alpha_minus_pi_fourth_l3970_397015


namespace NUMINAMATH_CALUDE_income_ratio_is_seven_to_six_l3970_397018

/-- Represents the income and expenditure of a person -/
structure Person where
  income : ℕ
  expenditure : ℕ

/-- Given the conditions of the problem, prove that the ratio of Rajan's income to Balan's income is 7:6 -/
theorem income_ratio_is_seven_to_six 
  (rajan balan : Person)
  (h1 : rajan.expenditure * 5 = balan.expenditure * 6)
  (h2 : rajan.income - rajan.expenditure = 1000)
  (h3 : balan.income - balan.expenditure = 1000)
  (h4 : rajan.income = 7000) :
  7 * balan.income = 6 * rajan.income := by
  sorry

#check income_ratio_is_seven_to_six

end NUMINAMATH_CALUDE_income_ratio_is_seven_to_six_l3970_397018


namespace NUMINAMATH_CALUDE_train_speed_without_stoppages_l3970_397063

/-- The average speed of a train without stoppages, given certain conditions. -/
theorem train_speed_without_stoppages (distance : ℝ) (time_with_stops : ℝ) (time_without_stops : ℝ)
  (h1 : time_without_stops = time_with_stops / 2)
  (h2 : distance / time_with_stops = 125) :
  distance / time_without_stops = 250 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_without_stoppages_l3970_397063


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l3970_397023

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, y < x → ¬(|2 * y + 3| ≤ 12)) ∧ (|2 * x + 3| ≤ 12) → x = -7 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l3970_397023


namespace NUMINAMATH_CALUDE_caterer_order_total_price_l3970_397095

theorem caterer_order_total_price :
  let ice_cream_bars := 125
  let sundaes := 125
  let ice_cream_bar_price := 0.60
  let sundae_price := 1.2
  let total_price := ice_cream_bars * ice_cream_bar_price + sundaes * sundae_price
  total_price = 225 := by sorry

end NUMINAMATH_CALUDE_caterer_order_total_price_l3970_397095


namespace NUMINAMATH_CALUDE_min_value_and_exponential_sum_l3970_397031

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - 2*a| + |x + b|

-- State the theorem
theorem min_value_and_exponential_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f x a b ≥ 2) 
  (hmin_exists : ∃ x, f x a b = 2) : 
  (2*a + b = 2) ∧ (∀ a' b', a' > 0 → b' > 0 → 2*a' + b' = 2 → 9^a' + 3^b' ≥ 6) ∧ 
  (∃ a' b', a' > 0 ∧ b' > 0 ∧ 2*a' + b' = 2 ∧ 9^a' + 3^b' = 6) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_exponential_sum_l3970_397031


namespace NUMINAMATH_CALUDE_domain_of_composition_l3970_397020

def f : Set ℝ → Prop := λ S => ∀ x ∈ S, 0 ≤ x ∧ x ≤ 1

theorem domain_of_composition (f : Set ℝ → Prop) (h : f (Set.Icc 0 1)) :
  f (Set.Icc 0 (1/2)) :=
sorry

end NUMINAMATH_CALUDE_domain_of_composition_l3970_397020
