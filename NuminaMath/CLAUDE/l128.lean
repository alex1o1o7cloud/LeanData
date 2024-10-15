import Mathlib

namespace NUMINAMATH_CALUDE_permutation_combination_equality_l128_12888

theorem permutation_combination_equality (n : ℕ) : 
  (n * (n - 1) = (n + 1) * n / 2) → n! = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_equality_l128_12888


namespace NUMINAMATH_CALUDE_compute_expression_l128_12818

theorem compute_expression : 3 * 3^4 + 9^60 / 9^58 = 324 := by sorry

end NUMINAMATH_CALUDE_compute_expression_l128_12818


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l128_12838

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_f_at_one :
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l128_12838


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l128_12842

theorem imaginary_part_of_complex_expression :
  let i : ℂ := Complex.I
  let z : ℂ := (2 + i) / i * i
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l128_12842


namespace NUMINAMATH_CALUDE_smallest_three_digit_square_append_l128_12817

/-- A function that appends two numbers -/
def append (a b : ℕ) : ℕ := a * (10 ^ (Nat.digits 10 b).length) + b

/-- Predicate to check if a number satisfies the given condition -/
def satisfies_condition (n : ℕ) : Prop :=
  ∃ (m : ℕ), append n (n + 1) = m ^ 2

/-- The smallest three-digit number satisfying the condition -/
def smallest_satisfying_number : ℕ := 183

theorem smallest_three_digit_square_append :
  (smallest_satisfying_number ≥ 100) ∧
  (smallest_satisfying_number < 1000) ∧
  satisfies_condition smallest_satisfying_number ∧
  ∀ n, n ≥ 100 ∧ n < smallest_satisfying_number → ¬(satisfies_condition n) :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_square_append_l128_12817


namespace NUMINAMATH_CALUDE_infinitely_many_k_with_Q_3k_geq_Q_3k1_l128_12849

-- Define Q(n) as the sum of the decimal digits of n
def Q (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem infinitely_many_k_with_Q_3k_geq_Q_3k1 :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ Q (3^k) ≥ Q (3^(k+1)) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_k_with_Q_3k_geq_Q_3k1_l128_12849


namespace NUMINAMATH_CALUDE_intersection_points_range_l128_12821

theorem intersection_points_range (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧
    y₁ = Real.sqrt (4 - x₁^2) ∧
    y₂ = Real.sqrt (4 - x₂^2) ∧
    k * x₁ - y₁ - 2 * k + 4 = 0 ∧
    k * x₂ - y₂ - 2 * k + 4 = 0) ↔
  (3/4 < k ∧ k ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_range_l128_12821


namespace NUMINAMATH_CALUDE_rectangular_plot_area_breadth_ratio_l128_12878

theorem rectangular_plot_area_breadth_ratio :
  let breadth : ℕ := 13
  let length : ℕ := breadth + 10
  let area : ℕ := length * breadth
  area / breadth = 23 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_breadth_ratio_l128_12878


namespace NUMINAMATH_CALUDE_set_equality_l128_12807

def S : Set (ℕ × ℕ) := {(x, y) | 2 * x + 3 * y = 16}

theorem set_equality : S = {(2, 4), (5, 2), (8, 0)} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l128_12807


namespace NUMINAMATH_CALUDE_isosceles_triangle_cut_l128_12809

-- Define the triangle PQR
structure Triangle :=
  (area : ℝ)
  (altitude : ℝ)

-- Define the line segment ST and resulting areas
structure Segment :=
  (length : ℝ)
  (trapezoid_area : ℝ)

-- Define the theorem
theorem isosceles_triangle_cut (PQR : Triangle) (ST : Segment) :
  PQR.area = 144 →
  PQR.altitude = 24 →
  ST.trapezoid_area = 108 →
  ST.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_cut_l128_12809


namespace NUMINAMATH_CALUDE_average_height_problem_l128_12805

/-- Proves that in a class of 50 students, if the average height of 10 students is 167 cm
    and the average height of the whole class is 168.6 cm, then the average height of
    the remaining 40 students is 169 cm. -/
theorem average_height_problem (total_students : ℕ) (group1_students : ℕ) 
  (group2_height : ℝ) (class_avg_height : ℝ) :
  total_students = 50 →
  group1_students = 40 →
  group2_height = 167 →
  class_avg_height = 168.6 →
  ∃ (group1_height : ℝ),
    group1_height = 169 ∧
    (group1_students : ℝ) * group1_height + (total_students - group1_students : ℝ) * group2_height =
      (total_students : ℝ) * class_avg_height :=
by sorry

end NUMINAMATH_CALUDE_average_height_problem_l128_12805


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l128_12882

theorem sum_of_coefficients (a : ℝ) : 
  ((1 + a)^5 = -1) → (a = -2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l128_12882


namespace NUMINAMATH_CALUDE_trevors_future_age_l128_12899

/-- Proves Trevor's age when his older brother is three times Trevor's current age -/
theorem trevors_future_age (t b : ℕ) (h1 : t = 11) (h2 : b = 20) :
  ∃ x : ℕ, b + (x - t) = 3 * t ∧ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_trevors_future_age_l128_12899


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l128_12863

theorem sum_reciprocals_bound (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  1 / (1 + a) + 1 / (1 + b) + 1 / (1 + c) + 1 / (1 + d) > 1 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l128_12863


namespace NUMINAMATH_CALUDE_toms_total_coins_l128_12837

/-- Represents the number of coins Tom has -/
structure TomCoins where
  quarters : ℕ
  nickels : ℕ

/-- The total number of coins Tom has -/
def total_coins (c : TomCoins) : ℕ :=
  c.quarters + c.nickels

/-- Tom's actual coin count -/
def toms_coins : TomCoins :=
  { quarters := 4, nickels := 8 }

theorem toms_total_coins :
  total_coins toms_coins = 12 := by
  sorry

end NUMINAMATH_CALUDE_toms_total_coins_l128_12837


namespace NUMINAMATH_CALUDE_fraction_equality_l128_12823

theorem fraction_equality (x y : ℚ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l128_12823


namespace NUMINAMATH_CALUDE_two_coin_toss_probabilities_l128_12887

theorem two_coin_toss_probabilities (P₁ P₂ P₃ : ℝ) 
  (h1 : P₁ ≥ 0) (h2 : P₂ ≥ 0) (h3 : P₃ ≥ 0)
  (h4 : P₁ ≤ 1) (h5 : P₂ ≤ 1) (h6 : P₃ ≤ 1)
  (h7 : P₁ = (1/2)^2) (h8 : P₂ = (1/2)^2) (h9 : P₃ = 2 * (1/2)^2) : 
  (P₁ + P₂ = P₃) ∧ (P₁ + P₂ + P₃ = 1) ∧ (P₃ = 2*P₁ ∧ P₃ = 2*P₂) := by
  sorry

end NUMINAMATH_CALUDE_two_coin_toss_probabilities_l128_12887


namespace NUMINAMATH_CALUDE_initial_channels_l128_12845

theorem initial_channels (x : ℕ) : 
  x - 20 + 12 - 10 + 8 + 7 = 147 → x = 150 := by
  sorry

end NUMINAMATH_CALUDE_initial_channels_l128_12845


namespace NUMINAMATH_CALUDE_f_properties_l128_12890

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1) + (1 - x) / (1 + x)

theorem f_properties (a : ℝ) (h_a : a > 0) :
  -- 1. If f'(1) = 0, then a = 1
  (∀ x : ℝ, x > 0 → (deriv (f a)) x = 0 → x = 1 → a = 1) ∧
  -- 2. For a ≥ 2, f'(x) > 0 for all x > 0
  (a ≥ 2 → ∀ x : ℝ, x > 0 → (deriv (f a)) x > 0) ∧
  -- 3. For 0 < a < 2, f'(x) < 0 for 0 < x < sqrt((2-a)/a) and f'(x) > 0 for x > sqrt((2-a)/a)
  (0 < a ∧ a < 2 → 
    (∀ x : ℝ, 0 < x ∧ x < Real.sqrt ((2 - a) / a) → (deriv (f a)) x < 0) ∧
    (∀ x : ℝ, x > Real.sqrt ((2 - a) / a) → (deriv (f a)) x > 0)) ∧
  -- 4. The minimum value of f(x) is 1 if and only if a ≥ 2
  (∃ x : ℝ, x ≥ 0 ∧ ∀ y : ℝ, y ≥ 0 → f a x ≤ f a y ∧ f a x = 1) ↔ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_f_properties_l128_12890


namespace NUMINAMATH_CALUDE_song_ratio_after_deletion_l128_12839

theorem song_ratio_after_deletion (total : ℕ) (deletion_percentage : ℚ) 
  (h1 : total = 720) 
  (h2 : deletion_percentage = 1/5) : 
  (total - (deletion_percentage * total).floor) / (deletion_percentage * total).floor = 4 := by
  sorry

end NUMINAMATH_CALUDE_song_ratio_after_deletion_l128_12839


namespace NUMINAMATH_CALUDE_bottle_cost_difference_l128_12868

/-- Represents a bottle of capsules -/
structure Bottle where
  capsules : ℕ
  cost : ℚ

/-- Calculates the cost per capsule for a given bottle -/
def costPerCapsule (b : Bottle) : ℚ := b.cost / b.capsules

/-- The difference in cost per capsule between two bottles -/
def costDifference (b1 b2 : Bottle) : ℚ := costPerCapsule b2 - costPerCapsule b1

theorem bottle_cost_difference :
  let bottleR : Bottle := { capsules := 250, cost := 25/4 }
  let bottleT : Bottle := { capsules := 100, cost := 3 }
  costDifference bottleR bottleT = 1/200
  := by sorry

end NUMINAMATH_CALUDE_bottle_cost_difference_l128_12868


namespace NUMINAMATH_CALUDE_shifted_sine_symmetry_l128_12832

open Real

theorem shifted_sine_symmetry (φ : Real) (h1 : 0 < φ) (h2 : φ < π) :
  let f : Real → Real := λ x ↦ sin (3 * x + φ)
  let g : Real → Real := λ x ↦ f (x - π / 12)
  (∀ x, g x = g (-x)) → φ = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_shifted_sine_symmetry_l128_12832


namespace NUMINAMATH_CALUDE_initial_population_is_4144_l128_12855

/-- Represents the population changes in a village --/
def village_population (initial : ℕ) : ℕ :=
  let after_bombardment := initial * 90 / 100
  let after_departure := after_bombardment * 85 / 100
  let after_refugees := after_departure + 50
  let after_births := after_refugees * 105 / 100
  let after_employment := after_births * 92 / 100
  after_employment + 100

/-- Theorem stating that the initial population of 4144 results in a final population of 3213 --/
theorem initial_population_is_4144 : village_population 4144 = 3213 := by
  sorry

end NUMINAMATH_CALUDE_initial_population_is_4144_l128_12855


namespace NUMINAMATH_CALUDE_vidya_age_difference_l128_12841

theorem vidya_age_difference : 
  let vidya_age : ℕ := 13
  let mother_age : ℕ := 44
  mother_age - 3 * vidya_age = 5 := by sorry

end NUMINAMATH_CALUDE_vidya_age_difference_l128_12841


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l128_12871

theorem nested_fraction_evaluation :
  1 + 1 / (2 + 1 / (3 + 1 / (3 + 3))) = 63 / 44 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l128_12871


namespace NUMINAMATH_CALUDE_isosceles_triangle_height_l128_12826

/-- Given an isosceles triangle and a rectangle with the same area,
    where the base of the triangle equals the width of the rectangle,
    prove that the height of the triangle is twice the length of the rectangle. -/
theorem isosceles_triangle_height (l w h : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  (l * w = 1/2 * w * h) →  -- Areas are equal
  (h = 2 * l) := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_height_l128_12826


namespace NUMINAMATH_CALUDE_cube_spheres_diagonal_outside_length_l128_12813

/-- Given a cube with edge length 1 and identical spheres centered at each vertex,
    where each sphere touches three neighboring spheres, the length of the part of
    the space diagonal of the cube that lies outside the spheres is √3 - 1. -/
theorem cube_spheres_diagonal_outside_length :
  let cube_edge_length : ℝ := 1
  let sphere_radius : ℝ := cube_edge_length / 2
  let cube_diagonal : ℝ := Real.sqrt 3
  let diagonal_inside_spheres : ℝ := 2 * sphere_radius
  cube_diagonal - diagonal_inside_spheres = Real.sqrt 3 - 1 := by
  sorry

#check cube_spheres_diagonal_outside_length

end NUMINAMATH_CALUDE_cube_spheres_diagonal_outside_length_l128_12813


namespace NUMINAMATH_CALUDE_tangent_line_max_difference_l128_12865

theorem tangent_line_max_difference (m n : ℝ) :
  ((m + 1)^2 + (n + 1)^2 = 4) →  -- Condition for tangent line
  (∀ x y : ℝ, (m + 1) * x + (n + 1) * y = 2 → x^2 + y^2 ≤ 1) →  -- Line touches or is outside the circle
  (∃ x y : ℝ, (m + 1) * x + (n + 1) * y = 2 ∧ x^2 + y^2 = 1) →  -- Line touches the circle at least at one point
  (m - n ≤ 2 * Real.sqrt 2) ∧ (∃ m₀ n₀ : ℝ, m₀ - n₀ = 2 * Real.sqrt 2 ∧ 
    ((m₀ + 1)^2 + (n₀ + 1)^2 = 4) ∧
    (∀ x y : ℝ, (m₀ + 1) * x + (n₀ + 1) * y = 2 → x^2 + y^2 ≤ 1) ∧
    (∃ x y : ℝ, (m₀ + 1) * x + (n₀ + 1) * y = 2 ∧ x^2 + y^2 = 1)) :=
by sorry


end NUMINAMATH_CALUDE_tangent_line_max_difference_l128_12865


namespace NUMINAMATH_CALUDE_smallest_n_for_2007_l128_12836

theorem smallest_n_for_2007 : 
  (∃ (n : ℕ) (S : Finset ℕ), 
    n > 1 ∧ 
    S.card = n ∧ 
    (∀ x ∈ S, x > 0) ∧ 
    S.prod id = 2007 ∧ 
    S.sum id = 2007 ∧ 
    (∀ m : ℕ, m > 1 → 
      (∃ T : Finset ℕ, 
        T.card = m ∧ 
        (∀ x ∈ T, x > 0) ∧ 
        T.prod id = 2007 ∧ 
        T.sum id = 2007) → 
      n ≤ m)) ∧ 
  (∀ S : Finset ℕ, 
    S.card > 1 ∧ 
    (∀ x ∈ S, x > 0) ∧ 
    S.prod id = 2007 ∧ 
    S.sum id = 2007 → 
    S.card ≥ 1337) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_2007_l128_12836


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l128_12891

def income : List ℝ := [45, 50, 60, 65, 70]

theorem cab_driver_average_income :
  (income.sum / income.length : ℝ) = 58 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l128_12891


namespace NUMINAMATH_CALUDE_moving_circle_locus_l128_12833

-- Define the fixed circles
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

-- Define the locus of the center of the moving circle
def locus (x y : ℝ) : Prop := (x + 2)^2 - (y^2 / 13^2) = 1 ∧ x < -1

-- State the theorem
theorem moving_circle_locus :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), r > 0 ∧
    (∀ (x' y' : ℝ), circle_M x' y' → (x - x')^2 + (y - y')^2 = r^2) ∧
    (∀ (x' y' : ℝ), circle_N x' y' → (x - x')^2 + (y - y')^2 = r^2)) →
  locus x y :=
by sorry

end NUMINAMATH_CALUDE_moving_circle_locus_l128_12833


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_252_l128_12896

/-- The count of numbers between 1000 and 9999 with four different digits 
    in either strictly increasing or strictly decreasing order -/
def count_special_numbers : ℕ := sorry

/-- A number is considered special if it has four different digits 
    in either strictly increasing or strictly decreasing order -/
def is_special (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  (∃ a b c d : ℕ, 
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
    ((a < b ∧ b < c ∧ c < d) ∨ (a > b ∧ b > c ∧ c > d)))

theorem count_special_numbers_eq_252 : 
  count_special_numbers = 252 :=
sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_252_l128_12896


namespace NUMINAMATH_CALUDE_nabla_problem_l128_12869

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem nabla_problem : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_problem_l128_12869


namespace NUMINAMATH_CALUDE_number_pair_theorem_l128_12852

theorem number_pair_theorem (S P x y : ℝ) (h1 : x + y = S) (h2 : x * y = P) :
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) ∧
  S^2 ≥ 4*P := by
  sorry

end NUMINAMATH_CALUDE_number_pair_theorem_l128_12852


namespace NUMINAMATH_CALUDE_dennis_pants_purchase_l128_12893

def pants_price : ℝ := 110
def socks_price : ℝ := 60
def discount_rate : ℝ := 0.3
def num_socks : ℕ := 2
def total_spent : ℝ := 392

def discounted_pants_price : ℝ := pants_price * (1 - discount_rate)
def discounted_socks_price : ℝ := socks_price * (1 - discount_rate)

theorem dennis_pants_purchase :
  ∃ (num_pants : ℕ),
    num_pants * discounted_pants_price + num_socks * discounted_socks_price = total_spent ∧
    num_pants = 4 := by
  sorry

end NUMINAMATH_CALUDE_dennis_pants_purchase_l128_12893


namespace NUMINAMATH_CALUDE_mineral_water_case_price_l128_12812

/-- The price of a case of mineral water -/
def case_price (daily_consumption : ℚ) (days : ℕ) (bottles_per_case : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / ((daily_consumption * days) / bottles_per_case)

/-- Theorem stating the price of a case of mineral water is $12 -/
theorem mineral_water_case_price :
  case_price (1/2) 240 24 60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mineral_water_case_price_l128_12812


namespace NUMINAMATH_CALUDE_exists_hole_for_unit_cube_l128_12822

/-- A hole in a cube is represented by a rectangle on one face of the cube -/
structure Hole :=
  (width : ℝ)
  (height : ℝ)

/-- A cube is represented by its edge length -/
structure Cube :=
  (edge : ℝ)

/-- A proposition that states a cube can pass through a hole -/
def CanPassThrough (c : Cube) (h : Hole) : Prop :=
  c.edge ≤ h.width ∧ c.edge ≤ h.height

/-- The main theorem stating that there exists a hole in a unit cube through which another unit cube can pass -/
theorem exists_hole_for_unit_cube :
  ∃ (h : Hole), CanPassThrough (Cube.mk 1) h ∧ h.width < 1 ∧ h.height < 1 :=
sorry

end NUMINAMATH_CALUDE_exists_hole_for_unit_cube_l128_12822


namespace NUMINAMATH_CALUDE_power_multiplication_l128_12840

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l128_12840


namespace NUMINAMATH_CALUDE_fourth_corner_rectangle_area_l128_12806

/-- Given a large rectangle divided into 9 smaller rectangles, where three corner rectangles
    have areas 9, 15, and 12, and the area ratios are the same between adjacent small rectangles,
    the area of the fourth corner rectangle is 20. -/
theorem fourth_corner_rectangle_area :
  ∀ (A B C D : ℝ),
    A = 9 →
    B = 15 →
    C = 12 →
    A / C = B / D →
    D = 20 := by
  sorry

end NUMINAMATH_CALUDE_fourth_corner_rectangle_area_l128_12806


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l128_12872

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, -6)
  let b : ℝ × ℝ := (-1, m)
  are_parallel a b → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l128_12872


namespace NUMINAMATH_CALUDE_range_of_m_for_quadratic_equation_l128_12800

theorem range_of_m_for_quadratic_equation (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo (-1) 0 ∧ x₂ ∈ Set.Ioi 3 ∧
    x₁^2 - 2*m*x₁ + m - 3 = 0 ∧ x₂^2 - 2*m*x₂ + m - 3 = 0) →
  m ∈ Set.Ioo (6/5) 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_quadratic_equation_l128_12800


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l128_12816

/-- Given a hyperbola and an intersecting line, prove the eccentricity range -/
theorem hyperbola_eccentricity_range 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (m : ℝ) 
  (h_intersect : ∃ x y : ℝ, y = 2*x + m ∧ x^2/a^2 - y^2/b^2 = 1) :
  ∃ e : ℝ, e^2 = (a^2 + b^2) / a^2 ∧ e > Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l128_12816


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_reversed_difference_l128_12820

theorem smallest_prime_factor_of_reversed_difference (A B C : ℕ) 
  (h1 : A ≠ C) 
  (h2 : A ≤ 9) (h3 : B ≤ 9) (h4 : C ≤ 9) 
  (h5 : A ≠ 0) :
  let ABC := 100 * A + 10 * B + C
  let CBA := 100 * C + 10 * B + A
  ∃ (k : ℕ), ABC - CBA = 3 * k ∧ 
  ∀ (p : ℕ), p < 3 → ¬(∃ (m : ℕ), ABC - CBA = p * m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_reversed_difference_l128_12820


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_union_of_A_and_B_l128_12861

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

-- Theorem for intersection
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 7} := by sorry

-- Theorem for union
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x < 10} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_union_of_A_and_B_l128_12861


namespace NUMINAMATH_CALUDE_third_median_length_l128_12828

/-- An isosceles triangle with specific median lengths and area -/
structure SpecialIsoscelesTriangle where
  -- Two sides of equal length
  base : ℝ
  leg : ℝ
  -- Two medians of equal length
  equalMedian : ℝ
  -- The third median
  thirdMedian : ℝ
  -- Constraints
  isIsosceles : base > 0 ∧ leg > 0
  equalMedianLength : equalMedian = 4
  areaConstraint : area = 3 * Real.sqrt 15
  -- Area calculation (placeholder)
  area : ℝ := sorry

/-- The theorem stating the length of the third median -/
theorem third_median_length (t : SpecialIsoscelesTriangle) : 
  t.thirdMedian = 2 * Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_third_median_length_l128_12828


namespace NUMINAMATH_CALUDE_soda_discount_theorem_l128_12867

/-- Calculates the discounted price for purchasing soda cans -/
def discounted_price (regular_price : ℚ) (num_cans : ℕ) : ℚ :=
  let cases := (num_cans + 23) / 24  -- Round up to nearest case
  let total_regular_price := regular_price * num_cans
  let discount_rate := 
    if cases ≤ 2 then 25/100
    else if cases ≤ 4 then 30/100
    else 35/100
  total_regular_price * (1 - discount_rate)

/-- Theorem stating the discounted price for 70 cans of soda -/
theorem soda_discount_theorem :
  discounted_price (55/100) 70 = 2772/100 := by
  sorry

end NUMINAMATH_CALUDE_soda_discount_theorem_l128_12867


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l128_12802

theorem simplify_and_rationalize (x : ℝ) :
  x = 8 / (Real.sqrt 75 + 3 * Real.sqrt 3 + Real.sqrt 48) →
  x = 2 * Real.sqrt 3 / 9 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l128_12802


namespace NUMINAMATH_CALUDE_specific_ellipse_equation_l128_12831

/-- Represents an ellipse with center at the origin and foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis length
  c : ℝ  -- Distance from center to focus

/-- The equation of an ellipse given its parameters -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / (e.a^2 - e.c^2) = 1

/-- Theorem: The equation of an ellipse with specific properties -/
theorem specific_ellipse_equation :
  ∀ (e : Ellipse),
    e.a = 9 →  -- Half of the major axis length (18/2)
    e.c = 3 →  -- One-third of the semi-major axis (trisecting condition)
    ∀ (x y : ℝ),
      ellipse_equation e x y ↔ x^2 / 81 + y^2 / 72 = 1 := by
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_equation_l128_12831


namespace NUMINAMATH_CALUDE_impossible_to_use_all_parts_l128_12856

theorem impossible_to_use_all_parts (p q r : ℕ) : 
  ¬∃ (x y z : ℕ), (2 * x + 2 * z = 2 * p + 2 * r + 2) ∧ 
                   (2 * x + y = 2 * p + q + 1) ∧ 
                   (y + z = q + r) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_use_all_parts_l128_12856


namespace NUMINAMATH_CALUDE_parallel_line_with_y_intercept_l128_12843

/-- Given a line mx + ny + 1 = 0 parallel to 4x + 3y + 5 = 0 with y-intercept 1/3, prove m = -4 and n = -3 -/
theorem parallel_line_with_y_intercept (m n : ℝ) : 
  (∀ x y, m * x + n * y + 1 = 0 ↔ 4 * x + 3 * y + 5 = 0) →  -- parallel condition
  (∃ y, m * 0 + n * y + 1 = 0 ∧ y = 1/3) →                  -- y-intercept condition
  m = -4 ∧ n = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_with_y_intercept_l128_12843


namespace NUMINAMATH_CALUDE_equation_solution_l128_12884

theorem equation_solution : ∃ (x : ℝ), 
  x > 0 ∧ 
  (1/4) * (5*x^2 - 4) = (x^2 - 40*x - 5) * (x^2 + 20*x + 2) ∧
  x = 20 + 10 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l128_12884


namespace NUMINAMATH_CALUDE_angle_U_measure_l128_12847

-- Define the hexagon and its angles
structure Hexagon :=
  (F I G U R E : ℝ)

-- Define the properties of the hexagon
def is_valid_hexagon (h : Hexagon) : Prop :=
  h.F + h.I + h.G + h.U + h.R + h.E = 720

def angles_congruent (h : Hexagon) : Prop :=
  h.F = h.I ∧ h.I = h.U

def angles_supplementary (h : Hexagon) : Prop :=
  h.G + h.R = 180 ∧ h.E + h.U = 180

-- Theorem statement
theorem angle_U_measure (h : Hexagon) 
  (valid : is_valid_hexagon h) 
  (congruent : angles_congruent h)
  (supplementary : angles_supplementary h) : 
  h.U = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_U_measure_l128_12847


namespace NUMINAMATH_CALUDE_tan_315_degrees_l128_12879

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l128_12879


namespace NUMINAMATH_CALUDE_unique_magnitude_for_complex_roots_l128_12827

theorem unique_magnitude_for_complex_roots (z : ℂ) : 
  z^2 - 6*z + 20 = 0 → ∃! m : ℝ, ∃ z : ℂ, z^2 - 6*z + 20 = 0 ∧ Complex.abs z = m :=
by
  sorry

end NUMINAMATH_CALUDE_unique_magnitude_for_complex_roots_l128_12827


namespace NUMINAMATH_CALUDE_product_of_fraction_is_111_l128_12803

/-- The repeating decimal 0.009̄ as a real number -/
def repeating_decimal : ℚ := 1 / 111

/-- The product of numerator and denominator when the repeating decimal is expressed as a fraction in lowest terms -/
def product_of_fraction : ℕ := 111

/-- Theorem stating that the product of the numerator and denominator of the fraction representation of 0.009̄ in lowest terms is 111 -/
theorem product_of_fraction_is_111 : 
  ∃ (n d : ℕ), d ≠ 0 ∧ repeating_decimal = n / d ∧ Nat.gcd n d = 1 ∧ n * d = product_of_fraction :=
by sorry

end NUMINAMATH_CALUDE_product_of_fraction_is_111_l128_12803


namespace NUMINAMATH_CALUDE_percentage_decrease_l128_12830

theorem percentage_decrease (x y z : ℝ) : 
  x = 1.2 * y → x = 0.48 * z → y = 0.4 * z :=
by sorry

end NUMINAMATH_CALUDE_percentage_decrease_l128_12830


namespace NUMINAMATH_CALUDE_expression_evaluation_l128_12860

theorem expression_evaluation :
  let a : ℤ := (-2)^2
  5 * a^2 - (a^2 - (2*a - 5*a^2) - 2*(a^2 - 3*a)) = 32 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l128_12860


namespace NUMINAMATH_CALUDE_solve_equation_l128_12870

theorem solve_equation (x : ℝ) : 3*x - 5*x + 4*x + 6 = 138 → x = 66 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l128_12870


namespace NUMINAMATH_CALUDE_sqrt_14_less_than_4_l128_12829

theorem sqrt_14_less_than_4 : Real.sqrt 14 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_14_less_than_4_l128_12829


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l128_12895

/-- Theorem: Tangent line intersection for two circles
    Given two circles:
    - Circle 1 with radius 3 and center (0, 0)
    - Circle 2 with radius 5 and center (12, 0)
    The x-coordinate of the point where a line tangent to both circles
    intersects the x-axis (to the right of the origin) is 9/2.
-/
theorem tangent_line_intersection (x : ℚ) : 
  (∃ y : ℚ, (x^2 + y^2 = 3^2 ∧ ((x - 12)^2 + y^2 = 5^2))) → x = 9/2 := by
  sorry

#check tangent_line_intersection

end NUMINAMATH_CALUDE_tangent_line_intersection_l128_12895


namespace NUMINAMATH_CALUDE_third_player_wins_probability_l128_12889

/-- Represents a game where players take turns tossing a fair six-sided die. -/
structure DieTossingGame where
  num_players : ℕ
  target_player : ℕ
  prob_six : ℚ

/-- The probability that the target player is the first to toss a six. -/
noncomputable def probability_target_wins (game : DieTossingGame) : ℚ :=
  sorry

/-- Theorem stating the probability of the third player being the first to toss a six
    in a four-player game. -/
theorem third_player_wins_probability :
  let game := DieTossingGame.mk 4 3 (1/6)
  probability_target_wins game = 125/671 := by
  sorry

end NUMINAMATH_CALUDE_third_player_wins_probability_l128_12889


namespace NUMINAMATH_CALUDE_sheela_savings_percentage_l128_12811

/-- Given Sheela's deposit and monthly income, prove the percentage of income deposited -/
theorem sheela_savings_percentage (deposit : ℝ) (monthly_income : ℝ) 
  (h1 : deposit = 5000)
  (h2 : monthly_income = 25000) :
  (deposit / monthly_income) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sheela_savings_percentage_l128_12811


namespace NUMINAMATH_CALUDE_greatest_power_of_three_l128_12877

def p : ℕ := (List.range 34).foldl (· * ·) 1

theorem greatest_power_of_three (k : ℕ) : k ≤ 15 ↔ (3^k : ℕ) ∣ p := by sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_l128_12877


namespace NUMINAMATH_CALUDE_squares_in_unit_square_l128_12846

/-- Two squares with side lengths a and b contained in a unit square without sharing interior points have a + b ≤ 1 -/
theorem squares_in_unit_square (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) 
  (contained : a ≤ 1 ∧ b ≤ 1) 
  (no_overlap : ∃ (x y x' y' : ℝ), 
    0 ≤ x ∧ x + a ≤ 1 ∧ 
    0 ≤ y ∧ y + a ≤ 1 ∧
    0 ≤ x' ∧ x' + b ≤ 1 ∧ 
    0 ≤ y' ∧ y' + b ≤ 1 ∧
    (x + a ≤ x' ∨ x' + b ≤ x ∨ y + a ≤ y' ∨ y' + b ≤ y)) : 
  a + b ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_squares_in_unit_square_l128_12846


namespace NUMINAMATH_CALUDE_sum_of_roots_l128_12874

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.log 3 / Real.log (3 * x) + Real.log (3 * x) / Real.log 27 = -4/3

-- Define the roots
def roots : Set ℝ := {x | equation x}

-- Theorem statement
theorem sum_of_roots :
  ∃ (a b : ℝ), a ∈ roots ∧ b ∈ roots ∧ a + b = 10/81 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l128_12874


namespace NUMINAMATH_CALUDE_survey_analysis_l128_12844

/-- Represents the survey data and population information -/
structure SurveyData where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  surveyed_male : ℕ
  surveyed_female : ℕ
  male_enthusiasts : ℕ
  male_non_enthusiasts : ℕ
  female_enthusiasts : ℕ
  female_non_enthusiasts : ℕ

/-- Calculates the K² value for the chi-square test -/
def calculate_k_squared (data : SurveyData) : ℚ :=
  let n := data.surveyed_male + data.surveyed_female
  let a := data.male_enthusiasts
  let b := data.male_non_enthusiasts
  let c := data.female_enthusiasts
  let d := data.female_non_enthusiasts
  (n : ℚ) * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The main theorem to prove -/
theorem survey_analysis (data : SurveyData) 
    (h1 : data.total_students = 9000)
    (h2 : data.male_students = 4000)
    (h3 : data.female_students = 5000)
    (h4 : data.surveyed_male = 40)
    (h5 : data.surveyed_female = 50)
    (h6 : data.male_enthusiasts = 20)
    (h7 : data.male_non_enthusiasts = 20)
    (h8 : data.female_enthusiasts = 40)
    (h9 : data.female_non_enthusiasts = 10) :
    (data.surveyed_male : ℚ) / data.surveyed_female = data.male_students / data.female_students ∧
    calculate_k_squared data > 6635 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_survey_analysis_l128_12844


namespace NUMINAMATH_CALUDE_paula_tickets_l128_12898

/-- The number of times Paula wants to ride the go-karts -/
def go_kart_rides : ℕ := 1

/-- The number of times Paula wants to ride the bumper cars -/
def bumper_car_rides : ℕ := 4

/-- The number of tickets required for one go-kart ride -/
def go_kart_tickets : ℕ := 4

/-- The number of tickets required for one bumper car ride -/
def bumper_car_tickets : ℕ := 5

/-- The total number of tickets Paula needs -/
def total_tickets : ℕ := go_kart_rides * go_kart_tickets + bumper_car_rides * bumper_car_tickets

theorem paula_tickets : total_tickets = 24 := by
  sorry

end NUMINAMATH_CALUDE_paula_tickets_l128_12898


namespace NUMINAMATH_CALUDE_list_size_theorem_l128_12815

theorem list_size_theorem (L : List ℝ) (n : ℝ) : 
  L.Nodup → 
  n ∈ L → 
  n = 5 * ((L.sum - n) / (L.length - 1)) → 
  n = 0.2 * L.sum → 
  L.length = 21 :=
sorry

end NUMINAMATH_CALUDE_list_size_theorem_l128_12815


namespace NUMINAMATH_CALUDE_arithmetic_mean_midpoint_l128_12835

/-- Given two points on a number line, their arithmetic mean is located halfway between them -/
theorem arithmetic_mean_midpoint (a b : ℝ) : ∃ m : ℝ, m = (a + b) / 2 ∧ m - a = b - m := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_midpoint_l128_12835


namespace NUMINAMATH_CALUDE_equation_solutions_l128_12810

theorem equation_solutions :
  (∀ x : ℝ, (1/2 * x^2 = 5) ↔ (x = Real.sqrt 10 ∨ x = -Real.sqrt 10)) ∧
  (∀ x : ℝ, ((x - 1)^2 = 16) ↔ (x = 5 ∨ x = -3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l128_12810


namespace NUMINAMATH_CALUDE_projectile_height_time_l128_12858

theorem projectile_height_time : ∃ t : ℝ, t > 0 ∧ -5*t^2 + 25*t = 30 ∧ ∀ s : ℝ, s > 0 ∧ -5*s^2 + 25*s = 30 → t ≤ s := by
  sorry

end NUMINAMATH_CALUDE_projectile_height_time_l128_12858


namespace NUMINAMATH_CALUDE_min_value_problem_l128_12894

theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 9*a*b + 9*b^2 + 3*c^2 ≥ 60 ∧
  (a^2 + 9*a*b + 9*b^2 + 3*c^2 = 60 ↔ a = 6 ∧ b = 2 ∧ c = 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l128_12894


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_eleven_twelfths_l128_12801

theorem sum_of_solutions_eq_eleven_twelfths :
  let f : ℝ → ℝ := λ x ↦ (4*x + 7)*(3*x - 8) + 12
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x + y = 11/12) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_eleven_twelfths_l128_12801


namespace NUMINAMATH_CALUDE_five_player_four_stage_tournament_outcomes_l128_12875

/-- Represents a tournament with a fixed number of players and stages. -/
structure Tournament :=
  (num_players : ℕ)
  (num_stages : ℕ)

/-- Calculates the number of possible outcomes in a tournament. -/
def tournament_outcomes (t : Tournament) : ℕ :=
  2^t.num_stages

/-- Theorem stating that a tournament with 5 players and 4 stages has 16 possible outcomes. -/
theorem five_player_four_stage_tournament_outcomes :
  ∀ t : Tournament, t.num_players = 5 → t.num_stages = 4 →
  tournament_outcomes t = 16 :=
by sorry

end NUMINAMATH_CALUDE_five_player_four_stage_tournament_outcomes_l128_12875


namespace NUMINAMATH_CALUDE_trailing_zeros_100_factorial_l128_12897

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailing_zeros_100_factorial : trailingZeros 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_100_factorial_l128_12897


namespace NUMINAMATH_CALUDE_distribute_7_4_l128_12848

/-- The number of ways to distribute n identical objects into k identical containers,
    with each container containing at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 identical apples into 4 identical packages,
    with each package containing at least one apple. -/
theorem distribute_7_4 : distribute 7 4 = 350 := by sorry

end NUMINAMATH_CALUDE_distribute_7_4_l128_12848


namespace NUMINAMATH_CALUDE_jose_pool_charge_ratio_l128_12834

/-- Represents the daily revenue from Jose's swimming pool --/
def daily_revenue (kids_charge : ℚ) (adults_charge : ℚ) : ℚ :=
  8 * kids_charge + 10 * adults_charge

/-- Represents the weekly revenue from Jose's swimming pool --/
def weekly_revenue (kids_charge : ℚ) (adults_charge : ℚ) : ℚ :=
  7 * daily_revenue kids_charge adults_charge

/-- Theorem stating the ratio of adult to kid charge in Jose's swimming pool --/
theorem jose_pool_charge_ratio :
  ∃ (adults_charge : ℚ),
    weekly_revenue 3 adults_charge = 588 ∧
    adults_charge / 3 = 2 := by
  sorry


end NUMINAMATH_CALUDE_jose_pool_charge_ratio_l128_12834


namespace NUMINAMATH_CALUDE_snow_volume_calculation_l128_12873

/-- Calculates the total volume of snow on a rectangular driveway with two distinct layers. -/
theorem snow_volume_calculation (length width depth1 depth2 : ℝ) 
  (h1 : length = 30) 
  (h2 : width = 4) 
  (h3 : depth1 = 0.5) 
  (h4 : depth2 = 0.3) : 
  length * width * depth1 + length * width * depth2 = 96 := by
  sorry

#check snow_volume_calculation

end NUMINAMATH_CALUDE_snow_volume_calculation_l128_12873


namespace NUMINAMATH_CALUDE_sum_of_coordinates_A_l128_12854

/-- Given points A, B, and C in a 2D plane satisfying specific conditions, 
    prove that the sum of the coordinates of A is 24. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (dist A C / dist A B = 1/3) →
  (dist B C / dist A B = 1/3) →
  B = (2, 6) →
  C = (4, 12) →
  A.1 + A.2 = 24 := by
  sorry

#check sum_of_coordinates_A

end NUMINAMATH_CALUDE_sum_of_coordinates_A_l128_12854


namespace NUMINAMATH_CALUDE_calculation_proof_l128_12808

theorem calculation_proof : -50 * 3 - (-2.5) / 0.1 = -125 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l128_12808


namespace NUMINAMATH_CALUDE_f_local_min_at_one_f_no_local_max_l128_12819

/-- The function f(x) = (x^3 - 1)^2 + 1 -/
def f (x : ℝ) : ℝ := (x^3 - 1)^2 + 1

/-- f has a local minimum at x = 1 -/
theorem f_local_min_at_one : 
  ∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1 :=
sorry

/-- f has no local maximum points -/
theorem f_no_local_max : 
  ¬∃ a, ∃ δ > 0, ∀ x, |x - a| < δ → f x ≤ f a :=
sorry

end NUMINAMATH_CALUDE_f_local_min_at_one_f_no_local_max_l128_12819


namespace NUMINAMATH_CALUDE_sum_of_odd_powers_l128_12886

theorem sum_of_odd_powers (x y z : ℝ) (n : ℕ) (h1 : x + y + z = 1) 
  (h2 : Real.arctan x + Real.arctan y + Real.arctan z = π / 4) : 
  x^(2*n + 1) + y^(2*n + 1) + z^(2*n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_powers_l128_12886


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_3_sufficient_necessary_condition_l128_12892

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

theorem intersection_complement_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | 0 ≤ x ∧ x < 2} := by sorry

theorem sufficient_necessary_condition (m : ℝ) :
  (∀ x, x ∈ B m ↔ x ∈ A) ↔ 1 ≤ m ∧ m ≤ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_3_sufficient_necessary_condition_l128_12892


namespace NUMINAMATH_CALUDE_average_age_combined_group_l128_12853

/-- Calculate the average age of a combined group of sixth-graders, parents, and teachers -/
theorem average_age_combined_group
  (n_students : ℕ) (avg_age_students : ℝ)
  (n_parents : ℕ) (avg_age_parents : ℝ)
  (n_teachers : ℕ) (avg_age_teachers : ℝ)
  (h_students : n_students = 40 ∧ avg_age_students = 12)
  (h_parents : n_parents = 50 ∧ avg_age_parents = 35)
  (h_teachers : n_teachers = 10 ∧ avg_age_teachers = 45) :
  let total_people := n_students + n_parents + n_teachers
  let total_age := n_students * avg_age_students + n_parents * avg_age_parents + n_teachers * avg_age_teachers
  total_age / total_people = 26.8 :=
by sorry

end NUMINAMATH_CALUDE_average_age_combined_group_l128_12853


namespace NUMINAMATH_CALUDE_profit_at_4_max_profit_price_l128_12825

noncomputable section

-- Define the sales volume function
def sales_volume (x : ℝ) : ℝ := 10 / (x - 2) + 4 * (x - 6)^2

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 2) * sales_volume x

-- Theorem for part (1)
theorem profit_at_4 : profit 4 = 42 := by sorry

-- Theorem for part (2)
theorem max_profit_price : 
  ∃ (x : ℝ), 2 < x ∧ x < 6 ∧ 
  (∀ (y : ℝ), 2 < y ∧ y < 6 → profit y ≤ profit x) ∧
  x = 10/3 := by sorry

end

end NUMINAMATH_CALUDE_profit_at_4_max_profit_price_l128_12825


namespace NUMINAMATH_CALUDE_questionnaires_15_16_l128_12885

/-- Represents the number of questionnaires collected for each age group -/
structure QuestionnaireData where
  age_8_10 : ℕ
  age_11_12 : ℕ
  age_13_14 : ℕ
  age_15_16 : ℕ

/-- Represents the sampling data -/
structure SamplingData where
  total_sample : ℕ
  sample_11_12 : ℕ

/-- Theorem stating the number of questionnaires drawn from the 15-16 years old group -/
theorem questionnaires_15_16 (data : QuestionnaireData) (sampling : SamplingData) :
  data.age_8_10 = 120 →
  data.age_11_12 = 180 →
  data.age_13_14 = 240 →
  sampling.total_sample = 300 →
  sampling.sample_11_12 = 60 →
  (data.age_8_10 + data.age_11_12 + data.age_13_14 + data.age_15_16) * sampling.sample_11_12 = 
    sampling.total_sample * data.age_11_12 →
  (sampling.total_sample * data.age_15_16) / (data.age_8_10 + data.age_11_12 + data.age_13_14 + data.age_15_16) = 120 :=
by sorry

end NUMINAMATH_CALUDE_questionnaires_15_16_l128_12885


namespace NUMINAMATH_CALUDE_smallest_ccd_value_l128_12859

theorem smallest_ccd_value (C D : ℕ) : 
  (1 ≤ C ∧ C ≤ 9) →
  (1 ≤ D ∧ D ≤ 9) →
  C ≠ D →
  (10 * C + D : ℕ) < 100 →
  (100 * C + 10 * C + D : ℕ) < 1000 →
  (10 * C + D : ℕ) = (100 * C + 10 * C + D : ℕ) / 7 →
  (∀ (C' D' : ℕ), 
    (1 ≤ C' ∧ C' ≤ 9) →
    (1 ≤ D' ∧ D' ≤ 9) →
    C' ≠ D' →
    (10 * C' + D' : ℕ) < 100 →
    (100 * C' + 10 * C' + D' : ℕ) < 1000 →
    (10 * C' + D' : ℕ) = (100 * C' + 10 * C' + D' : ℕ) / 7 →
    (100 * C + 10 * C + D : ℕ) ≤ (100 * C' + 10 * C' + D' : ℕ)) →
  (100 * C + 10 * C + D : ℕ) = 115 :=
by sorry

end NUMINAMATH_CALUDE_smallest_ccd_value_l128_12859


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l128_12824

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem ninth_term_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (h_arith : is_arithmetic_sequence a)
  (h_first : a 1 = 7/11)
  (h_seventeenth : a 17 = 5/6) :
  a 9 = 97/132 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l128_12824


namespace NUMINAMATH_CALUDE_x_value_proof_l128_12814

theorem x_value_proof (x : ℝ) (h : 9 / x^3 = x / 81) : x = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l128_12814


namespace NUMINAMATH_CALUDE_range_of_a_l128_12851

noncomputable def f (a x : ℝ) : ℝ := a / x - Real.exp (-x)

theorem range_of_a (a : ℝ) :
  (∃ p q : ℝ, p < q ∧ 
    (∀ x : ℝ, x > 0 → (f a x ≤ 0 ↔ p ≤ x ∧ x ≤ q))) →
  0 < a ∧ a < 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l128_12851


namespace NUMINAMATH_CALUDE_felipe_build_time_l128_12850

/-- Represents the time taken by each person to build their house, including break time. -/
structure BuildTime where
  felipe : ℝ
  emilio : ℝ
  carlos : ℝ

/-- Represents the break time taken by each person during construction. -/
structure BreakTime where
  felipe : ℝ
  emilio : ℝ
  carlos : ℝ

/-- The theorem stating Felipe's total build time is 27 months given the problem conditions. -/
theorem felipe_build_time (bt : BuildTime) (brt : BreakTime) : bt.felipe = 27 :=
  by
  have h1 : bt.felipe = bt.emilio / 2 := sorry
  have h2 : bt.carlos = bt.felipe + bt.emilio := sorry
  have h3 : bt.felipe + bt.emilio + bt.carlos = 10.5 * 12 := sorry
  have h4 : brt.felipe = 6 := sorry
  have h5 : brt.emilio = 2 * brt.felipe := sorry
  have h6 : brt.carlos = brt.emilio / 2 := sorry
  have h7 : bt.felipe + brt.felipe = 27 := sorry
  sorry

#check felipe_build_time

end NUMINAMATH_CALUDE_felipe_build_time_l128_12850


namespace NUMINAMATH_CALUDE_ball_probability_l128_12804

theorem ball_probability (p_red p_yellow p_blue : ℝ) : 
  p_red = 0.48 → p_yellow = 0.35 → p_red + p_yellow + p_blue = 1 → p_blue = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l128_12804


namespace NUMINAMATH_CALUDE_tv_price_change_l128_12866

theorem tv_price_change (P : ℝ) : 
  P > 0 → (P * 0.9) * 1.3 = P * 1.17 := by
sorry

end NUMINAMATH_CALUDE_tv_price_change_l128_12866


namespace NUMINAMATH_CALUDE_margaret_mean_score_l128_12864

def scores : List ℕ := [85, 88, 90, 92, 94, 96, 100]

def cyprian_score_count : ℕ := 4
def margaret_score_count : ℕ := 3
def cyprian_mean : ℚ := 92

theorem margaret_mean_score (h1 : scores.length = cyprian_score_count + margaret_score_count)
  (h2 : cyprian_mean = (scores.sum - (scores.sum - cyprian_mean * cyprian_score_count)) / cyprian_score_count) :
  (scores.sum - cyprian_mean * cyprian_score_count) / margaret_score_count = 92.33 := by
  sorry

end NUMINAMATH_CALUDE_margaret_mean_score_l128_12864


namespace NUMINAMATH_CALUDE_triangle_side_length_l128_12862

/-- Theorem: In a triangle ABC where side b = 2, angle A = 45°, and angle C = 75°, 
    the length of side a is equal to (2/3)√6. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  b = 2 → 
  A = 45 * π / 180 → 
  C = 75 * π / 180 → 
  a = (2 / 3) * Real.sqrt 6 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l128_12862


namespace NUMINAMATH_CALUDE_ratio_K_L_l128_12857

theorem ratio_K_L : ∃ (K L : ℤ),
  (∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 →
    (K / (x + 3 : ℝ)) + (L / (x^2 - 3*x : ℝ)) = ((x^2 - x + 5) / (x^3 + x^2 - 9*x) : ℝ)) →
  (K : ℚ) / (L : ℚ) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_K_L_l128_12857


namespace NUMINAMATH_CALUDE_second_man_speed_l128_12880

/-- Given two men walking in the same direction for 1 hour, where one walks at 10 kmph
    and they end up 2 km apart, the speed of the second man is 8 kmph. -/
theorem second_man_speed (speed_first : ℝ) (distance_apart : ℝ) (time : ℝ) (speed_second : ℝ) :
  speed_first = 10 →
  distance_apart = 2 →
  time = 1 →
  speed_first - speed_second = distance_apart / time →
  speed_second = 8 := by
sorry

end NUMINAMATH_CALUDE_second_man_speed_l128_12880


namespace NUMINAMATH_CALUDE_t_shirt_cost_l128_12883

/-- The cost of one T-shirt -/
def T : ℝ := sorry

/-- The cost of one pair of pants -/
def pants_cost : ℝ := 80

/-- The cost of one pair of shoes -/
def shoes_cost : ℝ := 150

/-- The discount rate applied to all items -/
def discount_rate : ℝ := 0.9

/-- The total cost Eugene pays after discount -/
def total_cost : ℝ := 558

theorem t_shirt_cost : T = 20 := by
  have h1 : total_cost = discount_rate * (4 * T + 3 * pants_cost + 2 * shoes_cost) := by sorry
  sorry

end NUMINAMATH_CALUDE_t_shirt_cost_l128_12883


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l128_12876

theorem sum_of_roots_quadratic (x : ℝ) : 
  x^2 - 6*x + 8 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = 6 ∧ x^2 - 6*x + 8 = (x - r₁) * (x - r₂) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l128_12876


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l128_12881

theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 30) :
  let a := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  let r := (d1 * d2) / (4 * a)
  r = (105 * Real.sqrt 274) / 274 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l128_12881
