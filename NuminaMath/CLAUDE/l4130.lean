import Mathlib

namespace NUMINAMATH_CALUDE_cubic_equation_properties_l4130_413034

theorem cubic_equation_properties :
  (∀ x y : ℕ, x^3 + 5*y = y^3 + 5*x → x = y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + 5*y = y^3 + 5*x) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_properties_l4130_413034


namespace NUMINAMATH_CALUDE_matrix_power_four_l4130_413070

/-- Given a 2x2 matrix A, prove that A^4 equals the given result. -/
theorem matrix_power_four (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A = !![3 * Real.sqrt 2, -3; 3, 3 * Real.sqrt 2] →
  A ^ 4 = !![-81, 0; 0, -81] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_four_l4130_413070


namespace NUMINAMATH_CALUDE_circle_C_equation_line_MN_equation_l4130_413026

-- Define the circle C
def circle_C (x y m : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + m = 0

-- Define the line that the circle is tangent to
def tangent_line (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + Real.sqrt 3 - 2 = 0

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop :=
  x + 2*y = 0

-- Theorem for the equation of circle C
theorem circle_C_equation :
  ∃ m, ∀ x y, circle_C x y m ↔ (x+2)^2 + (y-1)^2 = 4 :=
sorry

-- Theorem for the equation of line MN
theorem line_MN_equation :
  ∃ M N : ℝ × ℝ,
    (∀ x y, circle_C x y 0 → (x, y) = M ∨ (x, y) = N) ∧
    (symmetry_line M.1 M.2 ↔ symmetry_line N.1 N.2) ∧
    ((M.1 - N.1)^2 + (M.2 - N.2)^2 = 12) →
    ∃ c, ∀ x y, (2*x - y + c = 0 ∨ 2*x - y + (10 - c) = 0) ∧ c^2 = 30 :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_line_MN_equation_l4130_413026


namespace NUMINAMATH_CALUDE_angle_sum_pi_half_l4130_413041

theorem angle_sum_pi_half (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) : 
  α + β = π/2 := by sorry

end NUMINAMATH_CALUDE_angle_sum_pi_half_l4130_413041


namespace NUMINAMATH_CALUDE_polygon_diagonals_l4130_413019

theorem polygon_diagonals (n : ℕ) : 
  (n ≥ 3) →  -- Ensure it's a valid polygon
  (n - 3 = 5) →  -- At most 5 diagonals can be drawn from any vertex
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l4130_413019


namespace NUMINAMATH_CALUDE_opposite_face_of_y_l4130_413015

-- Define a cube net
structure CubeNet where
  faces : Finset Char
  y_face : Char
  foldable : Bool

-- Define a property for opposite faces in a cube
def opposite_faces (net : CubeNet) (face1 face2 : Char) : Prop :=
  face1 ∈ net.faces ∧ face2 ∈ net.faces ∧ face1 ≠ face2

-- Theorem statement
theorem opposite_face_of_y (net : CubeNet) :
  net.faces = {'W', 'X', 'Y', 'Z', 'V', net.y_face} →
  net.foldable = true →
  net.y_face ≠ 'V' →
  opposite_faces net net.y_face 'V' :=
sorry

end NUMINAMATH_CALUDE_opposite_face_of_y_l4130_413015


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_C_equals_C_l4130_413028

-- Define the sets A, B, and C
def A : Set ℝ := {x | -2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 3 * x - 5 ≥ x - 1}
def C (m : ℝ) : Set ℝ := {x | -x + m > 0}

-- Theorem for part 1
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < 5} := by sorry

-- Theorem for part 2
theorem union_A_C_equals_C (m : ℝ) : A ∪ C m = C m ↔ m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_C_equals_C_l4130_413028


namespace NUMINAMATH_CALUDE_monkey_peaches_l4130_413072

/-- Represents the number of peaches each monkey gets -/
structure MonkeyShares :=
  (eldest : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- The problem statement -/
theorem monkey_peaches (total : ℕ) (shares : MonkeyShares) : shares.second = 20 :=
  sorry

/-- Conditions of the problem -/
axiom divide_ratio (n m : ℕ) : n / (n + m) = 5 / 9
axiom eldest_share (total : ℕ) (shares : MonkeyShares) : shares.eldest = (total * 5) / 9
axiom second_share (total : ℕ) (shares : MonkeyShares) : 
  shares.second = ((total - shares.eldest) * 5) / 9
axiom third_share (total : ℕ) (shares : MonkeyShares) : 
  shares.third = total - shares.eldest - shares.second
axiom eldest_third_difference (shares : MonkeyShares) : shares.eldest - shares.third = 29

end NUMINAMATH_CALUDE_monkey_peaches_l4130_413072


namespace NUMINAMATH_CALUDE_magnitude_a_minus_2b_equals_sqrt_17_l4130_413017

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

theorem magnitude_a_minus_2b_equals_sqrt_17 :
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_a_minus_2b_equals_sqrt_17_l4130_413017


namespace NUMINAMATH_CALUDE_tax_difference_proof_l4130_413057

-- Define the item price and tax rates
def item_price : ℝ := 15
def tax_rate_1 : ℝ := 0.08
def tax_rate_2 : ℝ := 0.072
def discount_rate : ℝ := 0.005

-- Define the effective tax rate after discount
def effective_tax_rate : ℝ := tax_rate_2 - discount_rate

-- Theorem statement
theorem tax_difference_proof :
  (item_price * (1 + tax_rate_1)) - (item_price * (1 + effective_tax_rate)) = 0.195 := by
  sorry

end NUMINAMATH_CALUDE_tax_difference_proof_l4130_413057


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l4130_413003

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + Real.sqrt 2

-- Define the condition for intersection points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ 
    hyperbola_C A.1 A.2 ∧ hyperbola_C B.1 B.2 ∧
    line_l k A.1 A.2 ∧ line_l k B.1 B.2

-- Define the dot product condition
def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 > 2

-- Main theorem
theorem hyperbola_intersection_theorem (k : ℝ) :
  (hyperbola_C 0 0) ∧  -- Center at origin
  (hyperbola_C 2 0) ∧  -- Right focus at (2,0)
  (hyperbola_C (Real.sqrt 3) 0) ∧  -- Right vertex at (√3,0)
  (intersects_at_two_points k) ∧
  (∀ A B : ℝ × ℝ, hyperbola_C A.1 A.2 ∧ hyperbola_C B.1 B.2 ∧
    line_l k A.1 A.2 ∧ line_l k B.1 B.2 → dot_product_condition A B) →
  (-1 < k ∧ k < -Real.sqrt 3 / 3) ∨ (Real.sqrt 3 / 3 < k ∧ k < 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l4130_413003


namespace NUMINAMATH_CALUDE_set_operations_l4130_413054

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 < 0}
def B : Set ℝ := {x | -1 < x ∧ x < 1}

-- Theorem statement
theorem set_operations :
  (A ∩ B = B) ∧
  (B ⊆ A) ∧
  (A \ B = {x | x ≤ -1 ∨ (1 ≤ x ∧ x < 2)}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l4130_413054


namespace NUMINAMATH_CALUDE_f_equals_f_inv_at_zero_l4130_413062

/-- The function f(x) = 3x^2 - 6x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

/-- The inverse function of f -/
noncomputable def f_inv (x : ℝ) : ℝ := 1 + Real.sqrt ((1 + x) / 3)

/-- Theorem stating that f(0) = f⁻¹(0) -/
theorem f_equals_f_inv_at_zero : f 0 = f_inv 0 := by
  sorry

end NUMINAMATH_CALUDE_f_equals_f_inv_at_zero_l4130_413062


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l4130_413035

def diamond (X Y : ℚ) : ℚ := 4 * X + 3 * Y + 7

theorem diamond_equation_solution :
  ∃! X : ℚ, diamond X 5 = 75 ∧ X = 53 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l4130_413035


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l4130_413051

theorem sum_of_roots_equation (x : ℝ) : (x - 1) * (x + 4) = 18 → ∃ y : ℝ, (y - 1) * (y + 4) = 18 ∧ x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l4130_413051


namespace NUMINAMATH_CALUDE_value_of_a_l4130_413058

theorem value_of_a (a : ℝ) : (0.005 * a = 0.75) → a = 150 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l4130_413058


namespace NUMINAMATH_CALUDE_family_weight_ratio_l4130_413027

/-- Given the weights of three generations in a family, prove the ratio of the child's weight to the grandmother's weight -/
theorem family_weight_ratio :
  ∀ (grandmother daughter child : ℝ),
  grandmother + daughter + child = 160 →
  daughter + child = 60 →
  daughter = 40 →
  child / grandmother = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_family_weight_ratio_l4130_413027


namespace NUMINAMATH_CALUDE_equation_solutions_l4130_413014

theorem equation_solutions : 
  ∀ m n : ℕ, m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ 
  (m = 6 ∧ n = 3) ∨ (m = 9 ∧ n = 3) ∨ (m = 9 ∧ n = 5) ∨ (m = 54 ∧ n = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4130_413014


namespace NUMINAMATH_CALUDE_equation_solution_l4130_413067

theorem equation_solution (x : ℝ) : 
  (x = (-1 + Real.sqrt 3) / 2 ∨ x = (-1 - Real.sqrt 3) / 2) ↔ (2*x + 1)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4130_413067


namespace NUMINAMATH_CALUDE_logan_driving_time_l4130_413040

/-- Proves that Logan drove for 5 hours given the conditions of the problem -/
theorem logan_driving_time (tamika_speed : ℝ) (tamika_time : ℝ) (logan_speed : ℝ) (distance_difference : ℝ)
  (h_tamika_speed : tamika_speed = 45)
  (h_tamika_time : tamika_time = 8)
  (h_logan_speed : logan_speed = 55)
  (h_distance_difference : distance_difference = 85) :
  (tamika_speed * tamika_time - distance_difference) / logan_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_logan_driving_time_l4130_413040


namespace NUMINAMATH_CALUDE_star_seven_two_l4130_413087

def star (a b : ℤ) : ℤ := 4 * a - 4 * b

theorem star_seven_two : star 7 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_star_seven_two_l4130_413087


namespace NUMINAMATH_CALUDE_hairdresser_cash_register_l4130_413025

theorem hairdresser_cash_register (x : ℝ) : 
  (8 * x - 70 = 0) → x = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_hairdresser_cash_register_l4130_413025


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4130_413077

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def common_difference (a : ℕ → ℝ) : ℝ :=
  a 2 - a 1

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 105)
  (h_sum2 : a 2 + a 4 + a 6 = 99) :
  common_difference a = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4130_413077


namespace NUMINAMATH_CALUDE_minimum_employees_to_hire_l4130_413009

theorem minimum_employees_to_hire (S H : Finset Nat) 
  (h1 : S.card = 120)
  (h2 : H.card = 90)
  (h3 : (S ∩ H).card = 40) :
  (S ∪ H).card = 170 := by
sorry

end NUMINAMATH_CALUDE_minimum_employees_to_hire_l4130_413009


namespace NUMINAMATH_CALUDE_oldest_child_age_l4130_413089

theorem oldest_child_age (ages : Fin 4 → ℕ) 
  (h_average : (ages 0 + ages 1 + ages 2 + ages 3) / 4 = 9)
  (h_younger : ages 0 = 6 ∧ ages 1 = 8 ∧ ages 2 = 10) :
  ages 3 = 12 := by
sorry

end NUMINAMATH_CALUDE_oldest_child_age_l4130_413089


namespace NUMINAMATH_CALUDE_car_travel_distance_l4130_413095

/-- Proves that a car traveling for 12 hours at 68 km/h covers 816 km -/
theorem car_travel_distance (travel_time : ℝ) (average_speed : ℝ) (h1 : travel_time = 12) (h2 : average_speed = 68) : travel_time * average_speed = 816 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l4130_413095


namespace NUMINAMATH_CALUDE_remainder_71_73_mod_9_l4130_413083

theorem remainder_71_73_mod_9 : (71 * 73) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_71_73_mod_9_l4130_413083


namespace NUMINAMATH_CALUDE_find_m_l4130_413096

def A : Set ℕ := {1, 2, 3}
def B (m : ℕ) : Set ℕ := {2, m, 4}

theorem find_m : ∃ m : ℕ, (A ∩ B m = {2, 3}) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l4130_413096


namespace NUMINAMATH_CALUDE_power_of_product_l4130_413060

theorem power_of_product (a : ℝ) : (2 * a) ^ 3 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l4130_413060


namespace NUMINAMATH_CALUDE_three_digit_integers_with_specific_remainders_l4130_413020

theorem three_digit_integers_with_specific_remainders :
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, 100 ≤ n ∧ n < 1000 ∧ 
              n % 7 = 3 ∧ 
              n % 10 = 4 ∧ 
              n % 12 = 8) ∧
    S.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_integers_with_specific_remainders_l4130_413020


namespace NUMINAMATH_CALUDE_pentagonal_country_routes_fifty_cities_routes_no_forty_six_routes_l4130_413008

/-- Definition of a pentagonal country -/
def PentagonalCountry (n : ℕ) := n > 0

/-- Number of air routes in a pentagonal country -/
def airRoutes (n : ℕ) : ℕ := (n * 5) / 2

theorem pentagonal_country_routes (n : ℕ) (h : PentagonalCountry n) : 
  airRoutes n = (n * 5) / 2 :=
sorry

theorem fifty_cities_routes : 
  airRoutes 50 = 125 :=
sorry

theorem no_forty_six_routes : 
  ¬ ∃ (n : ℕ), PentagonalCountry n ∧ airRoutes n = 46 :=
sorry

end NUMINAMATH_CALUDE_pentagonal_country_routes_fifty_cities_routes_no_forty_six_routes_l4130_413008


namespace NUMINAMATH_CALUDE_no_cube_root_sum_prime_l4130_413004

theorem no_cube_root_sum_prime (x y p : ℕ+) (hp : Nat.Prime p.val) :
  (x.val : ℝ)^(1/3) + (y.val : ℝ)^(1/3) ≠ (p.val : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_no_cube_root_sum_prime_l4130_413004


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4130_413018

theorem quadratic_inequality_solution_set :
  {x : ℝ | 3 * x^2 - 2 * x - 8 < 0} = {x : ℝ | -4/3 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4130_413018


namespace NUMINAMATH_CALUDE_min_tetrahedra_decomposition_l4130_413099

/-- A tetrahedron is a polyhedron with four triangular faces -/
structure Tetrahedron

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube

/-- Represents a decomposition of a cube into tetrahedra -/
structure CubeDecomposition (c : Cube) where
  tetrahedra : Finset Tetrahedron
  is_valid : Bool  -- This would be a complex condition in practice

/-- The number of tetrahedra in a decomposition -/
def num_tetrahedra (d : CubeDecomposition c) : Nat :=
  d.tetrahedra.card

/-- A predicate that checks if a decomposition is minimal -/
def is_minimal_decomposition (d : CubeDecomposition c) : Prop :=
  ∀ d' : CubeDecomposition c, num_tetrahedra d ≤ num_tetrahedra d'

theorem min_tetrahedra_decomposition (c : Cube) :
  ∃ (d : CubeDecomposition c), is_minimal_decomposition d ∧ num_tetrahedra d = 5 :=
sorry

end NUMINAMATH_CALUDE_min_tetrahedra_decomposition_l4130_413099


namespace NUMINAMATH_CALUDE_alcohol_solution_volume_l4130_413050

theorem alcohol_solution_volume 
  (V : ℝ) 
  (h1 : 0.30 * V + 2.4 = 0.50 * (V + 2.4)) : 
  V = 6 := by
sorry

end NUMINAMATH_CALUDE_alcohol_solution_volume_l4130_413050


namespace NUMINAMATH_CALUDE_prob_sum_not_greater_than_4_prob_first_less_than_second_plus_2_l4130_413021

-- Define the sample space for a single die throw
def Die : Type := Fin 6

-- Define the sample space for two dice throws
def TwoDice : Type := Die × Die

-- Define the probability measure on TwoDice
noncomputable def P : Set TwoDice → ℝ := sorry

-- Define the event where the sum of dice is not greater than 4
def SumNotGreaterThan4 : Set TwoDice :=
  {x | x.1.val + x.2.val ≤ 4}

-- Define the event where the first die is less than the second die plus 2
def FirstLessThanSecondPlus2 : Set TwoDice :=
  {x | x.1.val < x.2.val + 2}

-- Theorem 1: Probability that the sum of dice is not greater than 4 is 1/6
theorem prob_sum_not_greater_than_4 :
  P SumNotGreaterThan4 = 1/6 := by sorry

-- Theorem 2: Probability that the first die is less than the second die plus 2 is 13/18
theorem prob_first_less_than_second_plus_2 :
  P FirstLessThanSecondPlus2 = 13/18 := by sorry

end NUMINAMATH_CALUDE_prob_sum_not_greater_than_4_prob_first_less_than_second_plus_2_l4130_413021


namespace NUMINAMATH_CALUDE_division_problem_l4130_413012

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 23 →
  divisor = 5 →
  remainder = 3 →
  dividend = divisor * quotient + remainder →
  quotient = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4130_413012


namespace NUMINAMATH_CALUDE_rhombus_property_l4130_413049

structure Rhombus (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)
  (is_rhombus : (B - A) = (C - B) ∧ (C - B) = (D - C) ∧ (D - C) = (A - D))

theorem rhombus_property {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (ABCD : Rhombus V) (E F P Q : V) :
  (∃ t : ℝ, E = ABCD.A + t • (ABCD.B - ABCD.A)) →
  (∃ s : ℝ, F = ABCD.A + s • (ABCD.D - ABCD.A)) →
  (ABCD.A - E = ABCD.D - F) →
  (∃ u : ℝ, P = ABCD.B + u • (ABCD.C - ABCD.B)) →
  (∃ v : ℝ, P = ABCD.D + v • (E - ABCD.D)) →
  (∃ w : ℝ, Q = ABCD.C + w • (ABCD.D - ABCD.C)) →
  (∃ x : ℝ, Q = ABCD.B + x • (F - ABCD.B)) →
  (∃ y z : ℝ, P - E = y • (P - ABCD.D) ∧ Q - F = z • (Q - ABCD.B) ∧ y + z = 1) ∧
  (∃ a : ℝ, ABCD.A - P = a • (Q - P)) :=
sorry

end NUMINAMATH_CALUDE_rhombus_property_l4130_413049


namespace NUMINAMATH_CALUDE_system_solution_l4130_413023

theorem system_solution : ∃ (x y z : ℝ), 
  (x^2 - 3*y + z = -4) ∧ 
  (x - 3*y + z^2 = -10) ∧ 
  (3*x + y^2 - 3*z = 0) ∧ 
  (x = -2) ∧ (y = 3) ∧ (z = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4130_413023


namespace NUMINAMATH_CALUDE_max_value_of_3a_plus_2_l4130_413022

theorem max_value_of_3a_plus_2 (a : ℝ) (h : 10 * a^2 + 3 * a + 2 = 5) :
  3 * a + 2 ≤ (31 + 3 * Real.sqrt 129) / 20 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_3a_plus_2_l4130_413022


namespace NUMINAMATH_CALUDE_coin_arrangement_coin_arrangement_proof_l4130_413066

theorem coin_arrangement (total_coins : ℕ) (walls : ℕ) (coins_per_wall : ℕ → Prop) : Prop :=
  (total_coins = 12 ∧ walls = 4) →
  (∀ n, coins_per_wall n → n ≥ 2 ∧ n ≤ 6) →
  (∃! n, coins_per_wall n ∧ n * walls = total_coins)

-- The proof goes here
theorem coin_arrangement_proof : coin_arrangement 12 4 (λ n ↦ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_coin_arrangement_coin_arrangement_proof_l4130_413066


namespace NUMINAMATH_CALUDE_a_range_when_f_decreasing_l4130_413056

/-- A piecewise function f(x) defined based on the parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (a - 3) * x + 3 * a else Real.log x / Real.log a

/-- Theorem stating that if f is decreasing on ℝ, then a is in the open interval (3/4, 1) -/
theorem a_range_when_f_decreasing (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) → a ∈ Set.Ioo (3/4) 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_when_f_decreasing_l4130_413056


namespace NUMINAMATH_CALUDE_function_m_minus_n_l4130_413081

def M (m : ℕ) : Set ℕ := {1, 2, 3, m}
def N (n : ℕ) : Set ℕ := {4, 7, n^4, n^2 + 3*n}

def f (x : ℕ) : ℕ := 3*x + 1

theorem function_m_minus_n (m n : ℕ) : 
  (∀ x ∈ M m, f x ∈ N n) → m - n = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_m_minus_n_l4130_413081


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l4130_413046

theorem inequality_system_solutions :
  let S := {x : ℤ | x ≥ 0 ∧ x - 3 * (x - 1) ≥ 1 ∧ (1 + 3 * x) / 2 > x - 1}
  S = {0, 1} := by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l4130_413046


namespace NUMINAMATH_CALUDE_smallest_c_for_unique_solution_l4130_413005

/-- The system of equations -/
def system (x y c : ℝ) : Prop :=
  2 * (x + 7)^2 + (y - 4)^2 = c ∧ (x + 4)^2 + 2 * (y - 7)^2 = c

/-- The system has a unique solution -/
def has_unique_solution (c : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, system p.1 p.2 c

/-- The smallest value of c for which the system has a unique solution is 6.0 -/
theorem smallest_c_for_unique_solution :
  (∀ c < 6, ¬ has_unique_solution c) ∧ has_unique_solution 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_unique_solution_l4130_413005


namespace NUMINAMATH_CALUDE_fraction_equality_l4130_413002

theorem fraction_equality (x : ℝ) (h : x = 5) : (x^4 - 8*x^2 + 16) / (x^2 - 4) = 21 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4130_413002


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4130_413055

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  ¬(∀ a b : ℝ, a^2 > b^2 → a > b ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4130_413055


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l4130_413086

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 3) : 2 * π * r^2 + π * r^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l4130_413086


namespace NUMINAMATH_CALUDE_horizontal_arrangement_possible_l4130_413088

/-- Represents a chessboard with 65 cells -/
structure ExtendedChessboard :=
  (cells : Fin 65 → Bool)

/-- Represents a domino (1x2 rectangle) -/
structure Domino :=
  (start : Fin 65)
  (horizontal : Bool)

/-- Represents the state of the chessboard with dominos -/
structure BoardState :=
  (board : ExtendedChessboard)
  (dominos : Fin 32 → Domino)

/-- Checks if two cells are adjacent on the extended chessboard -/
def are_adjacent (a b : Fin 65) : Bool := sorry

/-- Checks if a domino placement is valid -/
def valid_domino_placement (board : ExtendedChessboard) (domino : Domino) : Prop := sorry

/-- Checks if all dominos are placed horizontally -/
def all_horizontal (state : BoardState) : Prop := sorry

/-- Represents a move of a domino to adjacent empty cells -/
def valid_move (state₁ state₂ : BoardState) : Prop := sorry

/-- Main theorem: It's always possible to arrange all dominos horizontally -/
theorem horizontal_arrangement_possible (initial_state : BoardState) : 
  (∀ d, valid_domino_placement initial_state.board (initial_state.dominos d)) → 
  ∃ final_state, (valid_move initial_state final_state ∧ all_horizontal final_state) := sorry

end NUMINAMATH_CALUDE_horizontal_arrangement_possible_l4130_413088


namespace NUMINAMATH_CALUDE_odd_operations_l4130_413052

theorem odd_operations (a b : ℤ) (h_even : Even a) (h_odd : Odd b) :
  Odd (a + b) ∧ Odd (a - b) ∧ Odd ((a + b)^2) ∧ 
  ¬(∀ (a b : ℤ), Even a → Odd b → Odd (a * b)) ∧
  ¬(∀ (a b : ℤ), Even a → Odd b → Odd ((a + b) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_odd_operations_l4130_413052


namespace NUMINAMATH_CALUDE_percentage_solution_l4130_413010

/-- The percentage that, when applied to 100 and added to 20, results in 100 -/
def percentage_problem (P : ℝ) : Prop :=
  100 * (P / 100) + 20 = 100

/-- The solution to the percentage problem is 80% -/
theorem percentage_solution : ∃ P : ℝ, percentage_problem P ∧ P = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_solution_l4130_413010


namespace NUMINAMATH_CALUDE_system_solution_l4130_413063

theorem system_solution (x y z : ℤ) : 
  (x + y + z = 6 ∧ x + y * z = 7) ↔ 
  ((x, y, z) = (7, 0, -1) ∨ 
   (x, y, z) = (7, -1, 0) ∨ 
   (x, y, z) = (1, 3, 2) ∨ 
   (x, y, z) = (1, 2, 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4130_413063


namespace NUMINAMATH_CALUDE_sqrt_300_simplification_l4130_413007

theorem sqrt_300_simplification : Real.sqrt 300 = 10 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_300_simplification_l4130_413007


namespace NUMINAMATH_CALUDE_problem_statement_l4130_413074

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) : 
  (1 / a^2 + 1 / b^2 ≥ 1 / 2) ∧ 
  (∃ (m : ℝ), m = 2 * Real.sqrt 6 + 3 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = x * y → |2*x - 1| + |3*y - 1| ≥ m) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4130_413074


namespace NUMINAMATH_CALUDE_min_sheets_for_boats_l4130_413048

theorem min_sheets_for_boats (boats_per_sheet : ℕ) (planes_per_sheet : ℕ) (total_toys : ℕ) :
  boats_per_sheet = 8 →
  planes_per_sheet = 6 →
  total_toys = 80 →
  ∃ (sheets : ℕ), 
    sheets * boats_per_sheet = total_toys ∧
    sheets = 10 ∧
    (∀ (s : ℕ), s * boats_per_sheet = total_toys → s ≥ sheets) :=
by sorry

end NUMINAMATH_CALUDE_min_sheets_for_boats_l4130_413048


namespace NUMINAMATH_CALUDE_expression_evaluation_l4130_413029

theorem expression_evaluation : (2 + 6 * 3 - 4) + 2^3 * 4 / 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4130_413029


namespace NUMINAMATH_CALUDE_largest_difference_of_three_digit_numbers_l4130_413031

/-- A function that represents a 3-digit number given its digits -/
def threeDigitNumber (a b c : Nat) : Nat := 100 * a + 10 * b + c

/-- The set of valid digits -/
def validDigits : Finset Nat := Finset.range 9

theorem largest_difference_of_three_digit_numbers :
  ∃ (U V W X Y Z : Nat),
    U ∈ validDigits ∧ V ∈ validDigits ∧ W ∈ validDigits ∧
    X ∈ validDigits ∧ Y ∈ validDigits ∧ Z ∈ validDigits ∧
    U ≠ V ∧ U ≠ W ∧ U ≠ X ∧ U ≠ Y ∧ U ≠ Z ∧
    V ≠ W ∧ V ≠ X ∧ V ≠ Y ∧ V ≠ Z ∧
    W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧
    X ≠ Y ∧ X ≠ Z ∧
    Y ≠ Z ∧
    threeDigitNumber U V W - threeDigitNumber X Y Z = 864 ∧
    ∀ (A B C D E F : Nat),
      A ∈ validDigits → B ∈ validDigits → C ∈ validDigits →
      D ∈ validDigits → E ∈ validDigits → F ∈ validDigits →
      A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
      B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
      C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
      D ≠ E ∧ D ≠ F ∧
      E ≠ F →
      threeDigitNumber A B C - threeDigitNumber D E F ≤ 864 :=
by
  sorry


end NUMINAMATH_CALUDE_largest_difference_of_three_digit_numbers_l4130_413031


namespace NUMINAMATH_CALUDE_purple_ring_weight_l4130_413075

/-- The weight of the purple ring in an experiment, given the weights of other rings and the total weight -/
theorem purple_ring_weight :
  let orange_weight : ℚ := 0.08333333333333333
  let white_weight : ℚ := 0.4166666666666667
  let total_weight : ℚ := 0.8333333333
  let purple_weight : ℚ := total_weight - orange_weight - white_weight
  purple_weight = 0.3333333333 := by
  sorry

end NUMINAMATH_CALUDE_purple_ring_weight_l4130_413075


namespace NUMINAMATH_CALUDE_no_nonneg_int_solutions_l4130_413079

theorem no_nonneg_int_solutions :
  ¬∃ (x₁ x₂ : ℕ), 96 * x₁ + 97 * x₂ = 1000 := by
  sorry

end NUMINAMATH_CALUDE_no_nonneg_int_solutions_l4130_413079


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l4130_413065

variable (a b : ℝ)

theorem problem_1 : (a - b)^2 - (2*a + b)*(b - 2*a) = 5*a^2 - 2*a*b := by sorry

theorem problem_2 : (3 / (a + 1) - a + 1) / ((a^2 + 4*a + 4) / (a + 1)) = (2 - a) / (a + 2) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l4130_413065


namespace NUMINAMATH_CALUDE_jello_bathtub_cost_l4130_413082

/-- Calculate the total cost of filling a bathtub with jello mix -/
theorem jello_bathtub_cost :
  let bathtub_capacity : ℝ := 6  -- cubic feet
  let cubic_foot_to_gallon : ℝ := 7.5
  let gallon_weight : ℝ := 8  -- pounds
  let jello_per_pound : ℝ := 1.5  -- tablespoons
  let red_jello_cost : ℝ := 0.5  -- dollars per tablespoon
  let blue_jello_cost : ℝ := 0.4  -- dollars per tablespoon
  let green_jello_cost : ℝ := 0.6  -- dollars per tablespoon
  let red_jello_ratio : ℝ := 0.6
  let blue_jello_ratio : ℝ := 0.3
  let green_jello_ratio : ℝ := 0.1

  let total_water_weight := bathtub_capacity * cubic_foot_to_gallon * gallon_weight
  let total_jello_needed := total_water_weight * jello_per_pound
  let red_jello_amount := total_jello_needed * red_jello_ratio
  let blue_jello_amount := total_jello_needed * blue_jello_ratio
  let green_jello_amount := total_jello_needed * green_jello_ratio

  let total_cost := red_jello_amount * red_jello_cost +
                    blue_jello_amount * blue_jello_cost +
                    green_jello_amount * green_jello_cost

  total_cost = 259.2 := by sorry

end NUMINAMATH_CALUDE_jello_bathtub_cost_l4130_413082


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l4130_413038

theorem quadratic_inequality_solution_sets (a b c : ℝ) :
  (∀ x, ax^2 - b*x + c ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2) →
  (∀ x, c*x^2 + b*x + a ≤ 0 ↔ x ≤ -1 ∨ x ≥ -1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l4130_413038


namespace NUMINAMATH_CALUDE_stock_investment_change_l4130_413080

theorem stock_investment_change (x : ℝ) (x_pos : x > 0) : 
  x * (1 + 0.75) * (1 - 0.30) = 1.225 * x := by
  sorry

#check stock_investment_change

end NUMINAMATH_CALUDE_stock_investment_change_l4130_413080


namespace NUMINAMATH_CALUDE_unique_solution_proof_l4130_413039

/-- The positive value of k for which the equation 4x^2 + kx + 4 = 0 has exactly one solution -/
def unique_solution_k : ℝ := 8

theorem unique_solution_proof :
  ∃! (k : ℝ), k > 0 ∧
  (∃! (x : ℝ), 4 * x^2 + k * x + 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_proof_l4130_413039


namespace NUMINAMATH_CALUDE_second_number_is_ninety_l4130_413036

theorem second_number_is_ninety (x y z : ℝ) : 
  z = 4 * y →
  y = 2 * x →
  (x + y + z) / 3 = 165 →
  y = 90 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_ninety_l4130_413036


namespace NUMINAMATH_CALUDE_difference_of_squares_81_49_l4130_413045

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_81_49_l4130_413045


namespace NUMINAMATH_CALUDE_order_of_logarithms_and_fraction_l4130_413094

theorem order_of_logarithms_and_fraction :
  let a := Real.log 5 / Real.log 8
  let b := Real.log 3 / Real.log 4
  let c := 2 / 3
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_order_of_logarithms_and_fraction_l4130_413094


namespace NUMINAMATH_CALUDE_seven_hash_three_l4130_413091

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

-- Axioms for the # operation
axiom hash_zero (r : ℝ) : hash r 0 = r + 1
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 2) s = hash r s + s + 2

-- Theorem to prove
theorem seven_hash_three : hash 7 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_seven_hash_three_l4130_413091


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l4130_413076

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (A : ℝ),
    team_size = 11 →
    captain_age = 27 →
    wicket_keeper_age_diff = 3 →
    (team_size : ℝ) * A = 
      (captain_age : ℝ) + (captain_age + wicket_keeper_age_diff : ℝ) + 
      ((team_size - 2 : ℝ) * (A - 1)) →
    A = 24 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l4130_413076


namespace NUMINAMATH_CALUDE_P_iff_Q_l4130_413024

-- Define a triangle ABC with sides a, b, and c
structure Triangle :=
  (a b c : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b)

-- Define the condition P
def condition_P (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2

-- Define the condition Q
def condition_Q (t : Triangle) : Prop :=
  ∃ x : ℝ, (x^2 + 2*t.a*x + t.b^2 = 0) ∧ (x^2 + 2*t.c*x - t.b^2 = 0)

-- State the theorem
theorem P_iff_Q (t : Triangle) : condition_P t ↔ condition_Q t := by
  sorry

end NUMINAMATH_CALUDE_P_iff_Q_l4130_413024


namespace NUMINAMATH_CALUDE_max_squares_covered_l4130_413084

/-- Represents a square card with side length 2 inches -/
structure Card :=
  (side_length : ℝ)
  (h_side_length : side_length = 2)

/-- Represents a checkerboard with squares of side length 1 inch -/
structure Checkerboard :=
  (square_side_length : ℝ)
  (h_square_side_length : square_side_length = 1)

/-- The number of squares covered by the card on the checkerboard -/
def squares_covered (card : Card) (board : Checkerboard) : ℕ := sorry

/-- The theorem stating the maximum number of squares that can be covered -/
theorem max_squares_covered (card : Card) (board : Checkerboard) :
  ∃ (n : ℕ), squares_covered card board ≤ n ∧ n = 9 := by sorry

end NUMINAMATH_CALUDE_max_squares_covered_l4130_413084


namespace NUMINAMATH_CALUDE_marble_count_l4130_413043

theorem marble_count (white purple red blue green total : ℕ) : 
  white + purple + red + blue + green = total →
  2 * purple = 3 * white →
  5 * white = 2 * red →
  2 * blue = white →
  3 * green = white →
  blue = 24 →
  total = 120 := by
sorry

end NUMINAMATH_CALUDE_marble_count_l4130_413043


namespace NUMINAMATH_CALUDE_distance_between_Disney_and_London_l4130_413011

/-- The distance between lake Disney and lake London --/
def distance_Disney_London : ℝ := 60

/-- The number of migrating birds --/
def num_birds : ℕ := 20

/-- The distance between lake Jim and lake Disney --/
def distance_Jim_Disney : ℝ := 50

/-- The combined distance traveled by all birds in two seasons --/
def total_distance : ℝ := 2200

theorem distance_between_Disney_and_London :
  distance_Disney_London = 
    (total_distance - num_birds * distance_Jim_Disney) / num_birds :=
by sorry

end NUMINAMATH_CALUDE_distance_between_Disney_and_London_l4130_413011


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l4130_413001

/-- A linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a linear function -/
def pointOnLinearFunction (p : Point) (f : LinearFunction) : Prop :=
  p.y = f.m * p.x + f.b

/-- The theorem to be proved -/
theorem y1_greater_than_y2
  (f : LinearFunction)
  (A B C : Point)
  (h1 : f.m ≠ 0)
  (h2 : f.b = 4)
  (h3 : A.x = -2)
  (h4 : B.x = 1)
  (h5 : B.y = 3)
  (h6 : C.x = 3)
  (h7 : pointOnLinearFunction A f)
  (h8 : pointOnLinearFunction B f)
  (h9 : pointOnLinearFunction C f) :
  A.y > C.y :=
sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l4130_413001


namespace NUMINAMATH_CALUDE_circle_equation_l4130_413042

/-- Given a circle with center (a, -2a) passing through (2, -1) and tangent to x + y = 1,
    prove its equation is (x-1)^2 + (y+2)^2 = 2 -/
theorem circle_equation (a : ℝ) :
  (∀ x y : ℝ, y = -2 * x → (x - a)^2 + (y + 2*a)^2 = (2 - a)^2 + (-1 + 2*a)^2) →
  (∀ x y : ℝ, x + y = 1 → ((x - a)^2 + (y + 2*a)^2).sqrt = |x - a + y + 2*a| / Real.sqrt 2) →
  (∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l4130_413042


namespace NUMINAMATH_CALUDE_number_of_balls_in_box_l4130_413000

theorem number_of_balls_in_box : ∃ n : ℕ, n - 44 = 70 - n ∧ n = 57 := by sorry

end NUMINAMATH_CALUDE_number_of_balls_in_box_l4130_413000


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_7n_plus_2_l4130_413071

theorem max_gcd_13n_plus_4_7n_plus_2 :
  (∃ n : ℕ+, Nat.gcd (13 * n + 4) (7 * n + 2) = 2) ∧
  (∀ n : ℕ+, Nat.gcd (13 * n + 4) (7 * n + 2) ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_7n_plus_2_l4130_413071


namespace NUMINAMATH_CALUDE_circle_area_right_triangle_circle_area_right_triangle_value_l4130_413090

/-- The area of a circle passing through the vertices of a right triangle with legs of lengths 4 and 3 -/
theorem circle_area_right_triangle (π : ℝ) : ℝ :=
  let a : ℝ := 3  -- Length of one leg
  let b : ℝ := 4  -- Length of the other leg
  let c : ℝ := Real.sqrt (a^2 + b^2)  -- Length of the hypotenuse
  let r : ℝ := c / 2  -- Radius of the circle
  π * r^2

/-- The area of the circle is equal to 25π/4 -/
theorem circle_area_right_triangle_value (π : ℝ) :
  circle_area_right_triangle π = 25 / 4 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_right_triangle_circle_area_right_triangle_value_l4130_413090


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l4130_413078

theorem cubic_roots_sum (p q r : ℝ) : 
  (6 * p^3 + 500 * p + 1234 = 0) → 
  (6 * q^3 + 500 * q + 1234 = 0) → 
  (6 * r^3 + 500 * r + 1234 = 0) → 
  (p + q)^3 + (q + r)^3 + (r + p)^3 + 100 = 717 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l4130_413078


namespace NUMINAMATH_CALUDE_stratified_sample_size_l4130_413085

/-- Represents a stratified sample from a population -/
structure StratifiedSample where
  teachers : ℕ
  male_students : ℕ
  female_students : ℕ
  sample_teachers : ℕ
  sample_male_students : ℕ
  sample_female_students : ℕ

/-- Calculates the total sample size -/
def total_sample_size (s : StratifiedSample) : ℕ :=
  s.sample_teachers + s.sample_male_students + s.sample_female_students

/-- Theorem: If 100 out of 800 male students are selected in a stratified sample
    from a population of 200 teachers, 800 male students, and 600 female students,
    then the total sample size is 200 -/
theorem stratified_sample_size
  (s : StratifiedSample)
  (h1 : s.teachers = 200)
  (h2 : s.male_students = 800)
  (h3 : s.female_students = 600)
  (h4 : s.sample_male_students = 100)
  (h5 : s.sample_teachers = s.teachers / 8)
  (h6 : s.sample_female_students = s.female_students / 8) :
  total_sample_size s = 200 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l4130_413085


namespace NUMINAMATH_CALUDE_parallelogram_network_l4130_413047

theorem parallelogram_network (first_set : ℕ) (total_parallelograms : ℕ) (second_set : ℕ) : 
  first_set = 7 → 
  total_parallelograms = 588 → 
  total_parallelograms = (first_set - 1) * (second_set - 1) → 
  second_set = 99 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_network_l4130_413047


namespace NUMINAMATH_CALUDE_total_age_now_l4130_413044

-- Define Xavier's and Yasmin's ages as natural numbers
variable (xavier_age yasmin_age : ℕ)

-- Define the conditions
axiom xavier_twice_yasmin : xavier_age = 2 * yasmin_age
axiom xavier_future_age : xavier_age + 6 = 30

-- Theorem to prove
theorem total_age_now : xavier_age + yasmin_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_age_now_l4130_413044


namespace NUMINAMATH_CALUDE_unique_representation_theorem_l4130_413064

theorem unique_representation_theorem (n : ℕ) :
  ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
sorry

end NUMINAMATH_CALUDE_unique_representation_theorem_l4130_413064


namespace NUMINAMATH_CALUDE_perpendicular_equal_magnitude_vectors_l4130_413059

/-- Given two vectors m and n in ℝ², prove that if n is obtained by swapping and negating one component of m, then m and n are perpendicular and have equal magnitudes. -/
theorem perpendicular_equal_magnitude_vectors
  (a b : ℝ) :
  let m : ℝ × ℝ := (a, b)
  let n : ℝ × ℝ := (b, -a)
  (m.1 * n.1 + m.2 * n.2 = 0) ∧ 
  (m.1^2 + m.2^2 = n.1^2 + n.2^2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_equal_magnitude_vectors_l4130_413059


namespace NUMINAMATH_CALUDE_g_13_equals_218_l4130_413037

-- Define the function g
def g (n : ℕ) : ℕ := n^2 + 2*n + 23

-- State the theorem
theorem g_13_equals_218 : g 13 = 218 := by
  sorry

end NUMINAMATH_CALUDE_g_13_equals_218_l4130_413037


namespace NUMINAMATH_CALUDE_min_side_length_l4130_413013

/-- Given two triangles PQR and SQR sharing side QR, with PQ = 7 cm, PR = 15 cm, SR = 10 cm, and QS = 25 cm, the least possible integral length of QR is 16 cm. -/
theorem min_side_length (PQ PR SR QS : ℕ) (h1 : PQ = 7) (h2 : PR = 15) (h3 : SR = 10) (h4 : QS = 25) :
  (∃ QR : ℕ, QR > PR - PQ ∧ QR > QS - SR ∧ ∀ x : ℕ, (x > PR - PQ ∧ x > QS - SR) → x ≥ QR) →
  (∃ QR : ℕ, QR = 16 ∧ QR > PR - PQ ∧ QR > QS - SR ∧ ∀ x : ℕ, (x > PR - PQ ∧ x > QS - SR) → x ≥ QR) :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_l4130_413013


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4130_413030

theorem sufficient_not_necessary_condition (x : ℝ) : 
  (x = 1 → x^3 = x) ∧ ¬(x^3 = x → x = 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4130_413030


namespace NUMINAMATH_CALUDE_max_valid_sequence_length_l4130_413098

/-- A sequence of natural numbers satisfying the given conditions -/
def ValidSequence (a : Fin k → ℕ) : Prop :=
  (∀ i j, i < j → a i < a j) ∧ 
  (∀ i, 1 ≤ a i ∧ a i ≤ 50) ∧
  (∀ i j, i ≠ j → ¬(7 ∣ (a i + a j)))

/-- The maximum length of a valid sequence -/
def MaxValidSequenceLength : ℕ := 23

theorem max_valid_sequence_length :
  (∃ (k : ℕ) (a : Fin k → ℕ), ValidSequence a ∧ k = MaxValidSequenceLength) ∧
  (∀ (k : ℕ) (a : Fin k → ℕ), ValidSequence a → k ≤ MaxValidSequenceLength) :=
sorry

end NUMINAMATH_CALUDE_max_valid_sequence_length_l4130_413098


namespace NUMINAMATH_CALUDE_square_land_side_length_l4130_413073

theorem square_land_side_length (area : ℝ) (side : ℝ) : 
  area = 1024 → side * side = area → side = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l4130_413073


namespace NUMINAMATH_CALUDE_original_denominator_problem_l4130_413016

theorem original_denominator_problem (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →  -- Ensure the original fraction is well-defined
  (3 + 4 : ℚ) / (d + 4) = 1 / 3 → 
  d = 17 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l4130_413016


namespace NUMINAMATH_CALUDE_moon_distance_scientific_notation_l4130_413092

def moon_distance : ℝ := 384000

theorem moon_distance_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), moon_distance = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.84 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_moon_distance_scientific_notation_l4130_413092


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4130_413093

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum of first n terms

/-- Properties of the arithmetic sequence -/
def ArithmeticSequenceProperties (seq : ArithmeticSequence) : Prop :=
  (seq.a 2 = 3) ∧ (seq.S 9 = 6 * seq.S 3)

/-- Theorem: The common difference of the arithmetic sequence is 1 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : ArithmeticSequenceProperties seq) :
  seq.d = 1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4130_413093


namespace NUMINAMATH_CALUDE_car_rental_cost_l4130_413006

/-- The maximum daily rental cost for a car, given budget and mileage constraints -/
theorem car_rental_cost (budget : ℝ) (max_miles : ℝ) (cost_per_mile : ℝ) :
  budget = 88 ∧ max_miles = 190 ∧ cost_per_mile = 0.2 →
  ∃ (daily_rental : ℝ), daily_rental ≤ 50 ∧ daily_rental + max_miles * cost_per_mile ≤ budget :=
by sorry

end NUMINAMATH_CALUDE_car_rental_cost_l4130_413006


namespace NUMINAMATH_CALUDE_cubic_root_of_unity_expression_l4130_413068

theorem cubic_root_of_unity_expression : 
  ∀ ω : ℂ, ω ^ 3 = 1 → ω ≠ (1 : ℂ) → (2 - ω + ω^2)^3 + (2 + ω - ω^2)^3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_of_unity_expression_l4130_413068


namespace NUMINAMATH_CALUDE_article_percentage_gain_l4130_413097

/-- Calculates the percentage gain when selling an article --/
def percentage_gain (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating the percentage gain for the given problem --/
theorem article_percentage_gain :
  let cost_price : ℚ := 40
  let selling_price : ℚ := 350
  percentage_gain cost_price selling_price = 775 := by
  sorry

end NUMINAMATH_CALUDE_article_percentage_gain_l4130_413097


namespace NUMINAMATH_CALUDE_min_max_y_sum_l4130_413061

theorem min_max_y_sum (x y z : ℝ) 
  (h1 : x + y + z = 3) 
  (h2 : x^2 + y^2 + z^2 = 5) 
  (h3 : x * z = 1) : 
  ∃ (m M : ℝ), (∀ y', x + y' + z = 3 ∧ x^2 + y'^2 + z^2 = 5 → m ≤ y' ∧ y' ≤ M) ∧ m = 0 ∧ M = 0 ∧ m + M = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_max_y_sum_l4130_413061


namespace NUMINAMATH_CALUDE_horizontal_line_slope_l4130_413033

/-- The slope of a horizontal line y + 3 = 0 is 0 -/
theorem horizontal_line_slope (x y : ℝ) : y + 3 = 0 → (∀ x₁ x₂, x₁ ≠ x₂ → (y - y) / (x₁ - x₂) = 0) := by
  sorry

end NUMINAMATH_CALUDE_horizontal_line_slope_l4130_413033


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4130_413032

theorem quadratic_equation_solution (a : ℝ) : 
  (∀ x : ℝ, x = 1 → a * x^2 - 6 * x + 3 = 0) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4130_413032


namespace NUMINAMATH_CALUDE_sin_120_degrees_l4130_413053

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l4130_413053


namespace NUMINAMATH_CALUDE_product_xy_is_zero_l4130_413069

theorem product_xy_is_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 108) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_is_zero_l4130_413069
