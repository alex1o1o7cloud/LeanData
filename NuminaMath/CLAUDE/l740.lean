import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_d_μ_M_equals_59_22_l740_74022

-- Define the data distribution
def data_distribution : List (Nat × Nat) :=
  (List.range 28).map (fun x => (x + 1, 24)) ++
  [(29, 22), (30, 22), (31, 14)]

-- Define the total number of data points
def total_count : Nat :=
  data_distribution.foldl (fun acc (_, count) => acc + count) 0

-- Define the median of modes
def d : ℝ := 14.5

-- Define the median of the entire dataset
def M : ℝ := 29

-- Define the mean of the entire dataset
noncomputable def μ : ℝ :=
  let sum := data_distribution.foldl (fun acc (value, count) => acc + value * count) 0
  (sum : ℝ) / total_count

-- Theorem statement
theorem sum_of_d_μ_M_equals_59_22 :
  d + μ + M = 59.22 := by sorry

end NUMINAMATH_CALUDE_sum_of_d_μ_M_equals_59_22_l740_74022


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l740_74049

theorem absolute_value_inequality (x : ℝ) :
  (|x + 1| - |x - 3| ≥ 2) ↔ (x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l740_74049


namespace NUMINAMATH_CALUDE_triangle_division_theorem_l740_74086

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Check if a point is inside or on the boundary of a triangle -/
def pointInTriangle (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Represent a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a line -/
def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Calculate the area of a part of the triangle cut by a line -/
def areaPartition (t : Triangle) (l : Line) : ℝ := sorry

theorem triangle_division_theorem (t : Triangle) (P : ℝ × ℝ) (m n : ℝ) 
  (h_point : pointInTriangle P t) (h_positive : m > 0 ∧ n > 0) :
  ∃ (l : Line), 
    pointOnLine P l ∧ 
    areaPartition t l / (triangleArea t - areaPartition t l) = m / n :=
sorry

end NUMINAMATH_CALUDE_triangle_division_theorem_l740_74086


namespace NUMINAMATH_CALUDE_fraction_addition_l740_74084

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l740_74084


namespace NUMINAMATH_CALUDE_triangle_problem_l740_74060

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with side lengths a, b, c opposite to angles A, B, C
  c * Real.cos A - 2 * b * Real.cos B + a * Real.cos C = 0 →
  a + c = 13 →
  c > a →
  a * c * Real.cos B = 20 →
  B = Real.pi / 3 ∧ Real.sin A = 5 * Real.sqrt 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l740_74060


namespace NUMINAMATH_CALUDE_bus_passengers_second_stop_l740_74091

/-- Given a bus with the following properties:
  * 23 rows of 4 seats each
  * 16 people board at the start
  * At the first stop, 15 people board and 3 get off
  * At the second stop, 17 people board
  * There are 57 empty seats after the second stop
  Prove that 10 people got off at the second stop. -/
theorem bus_passengers_second_stop 
  (total_seats : ℕ) 
  (initial_passengers : ℕ) 
  (first_stop_on : ℕ) 
  (first_stop_off : ℕ) 
  (second_stop_on : ℕ) 
  (empty_seats_after_second : ℕ) 
  (h1 : total_seats = 23 * 4)
  (h2 : initial_passengers = 16)
  (h3 : first_stop_on = 15)
  (h4 : first_stop_off = 3)
  (h5 : second_stop_on = 17)
  (h6 : empty_seats_after_second = 57) :
  ∃ (second_stop_off : ℕ), 
    second_stop_off = 10 ∧ 
    empty_seats_after_second = total_seats - (initial_passengers + first_stop_on - first_stop_off + second_stop_on - second_stop_off) :=
by sorry

end NUMINAMATH_CALUDE_bus_passengers_second_stop_l740_74091


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l740_74037

theorem factor_implies_c_value (c : ℚ) : 
  (∀ x : ℚ, (x - 3) ∣ (c * x^3 - 6 * x^2 - c * x + 10)) → c = 11/6 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l740_74037


namespace NUMINAMATH_CALUDE_root_relationship_l740_74006

def P (x : ℝ) : ℝ := x^3 - 2*x + 1

def Q (x : ℝ) : ℝ := x^3 - 4*x^2 + 4*x - 1

theorem root_relationship (r : ℝ) : P r = 0 → Q (r^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_relationship_l740_74006


namespace NUMINAMATH_CALUDE_hiker_count_l740_74025

theorem hiker_count : ∃ (n m : ℕ), n > 13 ∧ n = 23 ∧ m > 0 ∧ 
  2 * m ≡ 1 [MOD n] ∧ 3 * m ≡ 13 [MOD n] := by
  sorry

end NUMINAMATH_CALUDE_hiker_count_l740_74025


namespace NUMINAMATH_CALUDE_estimate_sum_approximately_equal_500_l740_74015

def round_to_nearest_hundred (n : ℕ) : ℕ :=
  (n + 50) / 100 * 100

def approximately_equal (a b : ℕ) : Prop :=
  round_to_nearest_hundred a = round_to_nearest_hundred b

theorem estimate_sum_approximately_equal_500 :
  approximately_equal (208 + 298) 500 := by sorry

end NUMINAMATH_CALUDE_estimate_sum_approximately_equal_500_l740_74015


namespace NUMINAMATH_CALUDE_fast_food_fries_sales_l740_74069

theorem fast_food_fries_sales (S M L XL : ℕ) : 
  S + M + L + XL = 123 →
  Odd (S + M) →
  XL = 2 * M →
  L = S + M + 7 →
  S = 4 ∧ M = 27 ∧ L = 38 ∧ XL = 54 ∧ XL * 41 = 18 * (S + M + L + XL) :=
by sorry

end NUMINAMATH_CALUDE_fast_food_fries_sales_l740_74069


namespace NUMINAMATH_CALUDE_xy_sum_greater_than_two_l740_74021

theorem xy_sum_greater_than_two (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : x + y > 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_greater_than_two_l740_74021


namespace NUMINAMATH_CALUDE_vector_calculation_l740_74051

/-- Given vectors a and b in ℝ², prove that 2a - b equals (5, 7) -/
theorem vector_calculation (a b : ℝ × ℝ) 
  (ha : a = (2, 4)) (hb : b = (-1, 1)) : 
  (2 : ℝ) • a - b = (5, 7) := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l740_74051


namespace NUMINAMATH_CALUDE_possible_distances_andrey_gleb_l740_74017

/-- Represents the position of a house on a straight street. -/
structure HousePosition where
  position : ℝ

/-- Represents the configuration of houses on the street. -/
structure StreetConfiguration where
  andrey : HousePosition
  borya : HousePosition
  vova : HousePosition
  gleb : HousePosition

/-- The distance between two house positions. -/
def distance (a b : HousePosition) : ℝ :=
  |a.position - b.position|

/-- Theorem stating the possible distances between Andrey's and Gleb's houses. -/
theorem possible_distances_andrey_gleb (config : StreetConfiguration) :
  (distance config.andrey config.borya = 600) →
  (distance config.vova config.gleb = 600) →
  (distance config.andrey config.gleb = 3 * distance config.borya config.vova) →
  (distance config.andrey config.gleb = 900 ∨ distance config.andrey config.gleb = 1800) :=
by sorry

end NUMINAMATH_CALUDE_possible_distances_andrey_gleb_l740_74017


namespace NUMINAMATH_CALUDE_wendy_album_pics_l740_74050

def pictures_per_album (phone_pics camera_pics num_albums : ℕ) : ℕ :=
  (phone_pics + camera_pics) / num_albums

theorem wendy_album_pics : pictures_per_album 22 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_wendy_album_pics_l740_74050


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l740_74062

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (9^3 + 8^5 - 4^5) ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ (9^3 + 8^5 - 4^5) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l740_74062


namespace NUMINAMATH_CALUDE_cost_price_calculation_l740_74053

/-- Given a discount, profit percentage, and markup percentage, 
    calculate the cost price of an item. -/
theorem cost_price_calculation 
  (discount : ℝ) 
  (profit_percentage : ℝ) 
  (markup_percentage : ℝ) 
  (h1 : discount = 45)
  (h2 : profit_percentage = 0.20)
  (h3 : markup_percentage = 0.45) :
  ∃ (cost_price : ℝ), 
    cost_price * (1 + markup_percentage) - discount = cost_price * (1 + profit_percentage) ∧ 
    cost_price = 180 := by
  sorry


end NUMINAMATH_CALUDE_cost_price_calculation_l740_74053


namespace NUMINAMATH_CALUDE_inequality_proof_l740_74089

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1 / (2 * b^2)) * (b + 1 / (2 * a^2)) ≥ 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l740_74089


namespace NUMINAMATH_CALUDE_square_of_nilpotent_matrix_is_zero_l740_74092

theorem square_of_nilpotent_matrix_is_zero (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : A ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_of_nilpotent_matrix_is_zero_l740_74092


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l740_74045

def f (x : ℝ) : ℝ := (x + 1)^2 + 1

theorem f_strictly_increasing : 
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l740_74045


namespace NUMINAMATH_CALUDE_multiply_three_point_six_by_half_l740_74036

theorem multiply_three_point_six_by_half : 3.6 * 0.5 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_point_six_by_half_l740_74036


namespace NUMINAMATH_CALUDE_expression_equals_power_of_seven_l740_74011

theorem expression_equals_power_of_seven : 
  6 * (7 + 1) * (7^2 + 1) * (7^4 + 1) * (7^8 + 1) + 1 = 7^16 := by sorry

end NUMINAMATH_CALUDE_expression_equals_power_of_seven_l740_74011


namespace NUMINAMATH_CALUDE_magician_numbers_l740_74012

theorem magician_numbers : ∃! (a b : ℕ), 
  a * b = 2280 ∧ 
  a + b < 100 ∧ 
  a + b > 9 ∧ 
  Odd (a + b) ∧ 
  a = 40 ∧ 
  b = 57 := by sorry

end NUMINAMATH_CALUDE_magician_numbers_l740_74012


namespace NUMINAMATH_CALUDE_bella_win_probability_l740_74099

theorem bella_win_probability (lose_prob : ℚ) (no_tie : Bool) : lose_prob = 5/11 ∧ no_tie = true → 1 - lose_prob = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_bella_win_probability_l740_74099


namespace NUMINAMATH_CALUDE_star_equality_implies_x_equals_nine_l740_74042

/-- Binary operation ⋆ on pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  fun (a, b) (c, d) => (a - c, b + d)

/-- Theorem stating that if (6,5) ⋆ (2,3) = (x,y) ⋆ (5,4), then x = 9 -/
theorem star_equality_implies_x_equals_nine :
  ∀ x y : ℤ, star (6, 5) (2, 3) = star (x, y) (5, 4) → x = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_star_equality_implies_x_equals_nine_l740_74042


namespace NUMINAMATH_CALUDE_tree_height_problem_l740_74028

/-- Given two trees where one is 20 feet taller than the other and their heights
    are in the ratio 2:3, prove that the height of the taller tree is 60 feet. -/
theorem tree_height_problem (h : ℝ) (h_positive : h > 0) : 
  (h - 20) / h = 2 / 3 → h = 60 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_problem_l740_74028


namespace NUMINAMATH_CALUDE_star_equation_solution_l740_74035

def star (a b : ℕ) : ℕ := a^b + a*b

theorem star_equation_solution :
  ∀ a b : ℕ, 
  a ≥ 2 → b ≥ 2 → 
  star a b = 24 → 
  a + b = 6 := by sorry

end NUMINAMATH_CALUDE_star_equation_solution_l740_74035


namespace NUMINAMATH_CALUDE_no_simultaneous_perfect_squares_l740_74041

theorem no_simultaneous_perfect_squares (n : ℕ+) :
  ¬∃ (a b : ℕ+), ((n + 1) * 2^n.val = a^2) ∧ ((n + 3) * 2^(n.val + 2) = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_perfect_squares_l740_74041


namespace NUMINAMATH_CALUDE_inequality_proof_l740_74095

theorem inequality_proof (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l740_74095


namespace NUMINAMATH_CALUDE_second_candidate_percentage_l740_74052

theorem second_candidate_percentage (total_marks : ℝ) (passing_marks : ℝ) 
  (first_candidate_percentage : ℝ) (first_candidate_deficit : ℝ) 
  (second_candidate_excess : ℝ) : 
  passing_marks = 160 ∧ 
  first_candidate_percentage = 0.20 ∧ 
  first_candidate_deficit = 40 ∧ 
  second_candidate_excess = 20 ∧
  first_candidate_percentage * total_marks = passing_marks - first_candidate_deficit →
  (passing_marks + second_candidate_excess) / total_marks = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_second_candidate_percentage_l740_74052


namespace NUMINAMATH_CALUDE_angle_c_measure_l740_74003

/-- Given a triangle ABC where the sum of angles A and B is 110°, prove that the measure of angle C is 70°. -/
theorem angle_c_measure (A B C : ℝ) (h1 : A + B = 110) (h2 : A + B + C = 180) : C = 70 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_measure_l740_74003


namespace NUMINAMATH_CALUDE_tom_reading_speed_increase_l740_74040

/-- The factor by which Tom's reading speed increased -/
def reading_speed_increase_factor (normal_speed : ℕ) (increased_pages : ℕ) (hours : ℕ) : ℚ :=
  (increased_pages : ℚ) / ((normal_speed * hours) : ℚ)

/-- Theorem stating that Tom's reading speed increased by a factor of 3 -/
theorem tom_reading_speed_increase :
  reading_speed_increase_factor 12 72 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_reading_speed_increase_l740_74040


namespace NUMINAMATH_CALUDE_mark_parking_tickets_l740_74032

theorem mark_parking_tickets :
  ∀ (mark_speeding mark_parking sarah_speeding sarah_parking : ℕ),
  mark_speeding + mark_parking + sarah_speeding + sarah_parking = 24 →
  mark_parking = 2 * sarah_parking →
  mark_speeding = sarah_speeding →
  sarah_speeding = 6 →
  mark_parking = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_mark_parking_tickets_l740_74032


namespace NUMINAMATH_CALUDE_horner_method_v4_l740_74038

def f (x : ℝ) : ℝ := 3*x^6 + 5*x^5 + 6*x^4 + 20*x^3 - 8*x^2 + 35*x + 12

def horner_v4 (a₆ a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  let v₀ := a₆
  let v₁ := v₀ * x + a₅
  let v₂ := v₁ * x + a₄
  let v₃ := v₂ * x + a₃
  v₃ * x + a₂

theorem horner_method_v4 :
  horner_v4 3 5 6 20 (-8) 35 12 (-2) = -16 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v4_l740_74038


namespace NUMINAMATH_CALUDE_odd_function_property_l740_74001

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- Main theorem -/
theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : IsOdd f) 
  (h_even : IsEven (fun x ↦ f (x + 2))) 
  (h_f_neg_one : f (-1) = -1) : 
  f 2017 + f 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l740_74001


namespace NUMINAMATH_CALUDE_floor_product_eq_42_l740_74000

theorem floor_product_eq_42 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 42 ↔ 7 ≤ x ∧ x < 43/6 :=
sorry

end NUMINAMATH_CALUDE_floor_product_eq_42_l740_74000


namespace NUMINAMATH_CALUDE_inequality_proof_l740_74065

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b + b / c + c / a)^2 ≥ (3 / 2) * ((a + b) / c + (b + c) / a + (c + a) / b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l740_74065


namespace NUMINAMATH_CALUDE_multiplication_equalities_l740_74013

theorem multiplication_equalities : 
  (50 * 6 = 300) ∧ (5 * 60 = 300) ∧ (4 * 300 = 1200) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equalities_l740_74013


namespace NUMINAMATH_CALUDE_expression_independent_of_a_l740_74007

theorem expression_independent_of_a (a : ℝ) : 7 + a - (8 * a - (a + 5 - (4 - 6 * a))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_independent_of_a_l740_74007


namespace NUMINAMATH_CALUDE_max_x_plus_z_l740_74009

theorem max_x_plus_z (x y z t : ℝ) 
  (h1 : x^2 + y^2 = 4)
  (h2 : z^2 + t^2 = 9)
  (h3 : x*t + y*z = 6) :
  x + z ≤ Real.sqrt 13 ∧ ∃ x y z t, x^2 + y^2 = 4 ∧ z^2 + t^2 = 9 ∧ x*t + y*z = 6 ∧ x + z = Real.sqrt 13 := by
  sorry

#check max_x_plus_z

end NUMINAMATH_CALUDE_max_x_plus_z_l740_74009


namespace NUMINAMATH_CALUDE_closest_to_target_l740_74094

def options : List ℝ := [-4, -3, 0, 3, 4]

def target : ℝ := -3.4

def distance (x y : ℝ) : ℝ := |x - y|

theorem closest_to_target :
  ∃ (closest : ℝ), closest ∈ options ∧
    (∀ x ∈ options, distance target closest ≤ distance target x) ∧
    closest = -3 := by
  sorry

end NUMINAMATH_CALUDE_closest_to_target_l740_74094


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_sum_lower_bound_l740_74023

theorem quadrilateral_diagonal_sum_lower_bound (x y : ℝ) (α : ℝ) :
  x > 0 → y > 0 → 0 < α → α < π →
  x * y * Real.sin α = 2 →
  x + y ≥ 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_sum_lower_bound_l740_74023


namespace NUMINAMATH_CALUDE_value_of_B_l740_74097

/-- Given the value assignments for letters and words, prove the value of B --/
theorem value_of_B (T L A B : ℤ) : 
  T = 15 →
  B + A + L + L = 40 →
  L + A + B = 25 →
  A + L + L = 30 →
  B = 10 := by
sorry

end NUMINAMATH_CALUDE_value_of_B_l740_74097


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_range_of_a_l740_74018

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 4 < x ∧ x ≤ 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorems to be proved
theorem union_A_B : A ∪ B = {x | 3 ≤ x ∧ x ≤ 10} := by sorry

theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = {x | 7 ≤ x ∧ x ≤ 10} := by sorry

theorem range_of_a (a : ℝ) (h : (A ∩ C a).Nonempty) : a > 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_range_of_a_l740_74018


namespace NUMINAMATH_CALUDE_table_runner_coverage_l740_74034

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) (coverage_percentage : ℝ) (two_layer_area : ℝ) :
  total_runner_area = 204 →
  table_area = 175 →
  coverage_percentage = 0.8 →
  two_layer_area = 24 →
  ∃ (one_layer_area three_layer_area : ℝ),
    one_layer_area + two_layer_area + three_layer_area = coverage_percentage * table_area ∧
    one_layer_area + 2 * two_layer_area + 3 * three_layer_area = total_runner_area ∧
    three_layer_area = 20 :=
by sorry

end NUMINAMATH_CALUDE_table_runner_coverage_l740_74034


namespace NUMINAMATH_CALUDE_tan_sum_reciprocal_l740_74008

theorem tan_sum_reciprocal (a b : ℝ) 
  (h1 : (Real.sin a / Real.cos b) + (Real.sin b / Real.cos a) = 2)
  (h2 : (Real.cos a / Real.sin b) + (Real.cos b / Real.sin a) = 4) :
  (Real.tan a / Real.tan b) + (Real.tan b / Real.tan a) = 44/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_reciprocal_l740_74008


namespace NUMINAMATH_CALUDE_points_per_enemy_l740_74005

theorem points_per_enemy (total_enemies : ℕ) (enemies_not_destroyed : ℕ) (total_points : ℕ) : 
  total_enemies = 8 →
  enemies_not_destroyed = 6 →
  total_points = 10 →
  (total_points : ℚ) / (total_enemies - enemies_not_destroyed : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_points_per_enemy_l740_74005


namespace NUMINAMATH_CALUDE_new_girl_weight_l740_74024

/-- Proves that the weight of a new girl is 80 kg given the conditions of the problem -/
theorem new_girl_weight (n : ℕ) (initial_weight total_weight : ℝ) :
  n = 20 →
  initial_weight = 40 →
  (total_weight - initial_weight + 80) / n = total_weight / n + 2 →
  80 = total_weight - initial_weight + 40 :=
by sorry

end NUMINAMATH_CALUDE_new_girl_weight_l740_74024


namespace NUMINAMATH_CALUDE_sum_odd_integers_mod_12_l740_74033

/-- The sum of the first n odd positive integers -/
def sum_odd_integers (n : ℕ) : ℕ := n * n

/-- The theorem stating that the remainder when the sum of the first 10 odd positive integers 
    is divided by 12 is equal to 4 -/
theorem sum_odd_integers_mod_12 : sum_odd_integers 10 % 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_mod_12_l740_74033


namespace NUMINAMATH_CALUDE_symmetric_point_proof_l740_74026

/-- Given a point (0, 2) and a line x + y - 1 = 0, prove that (-1, 1) is the symmetric point --/
theorem symmetric_point_proof (P : ℝ × ℝ) (P' : ℝ × ℝ) (l : ℝ → ℝ → Prop) :
  P = (0, 2) →
  (∀ x y, l x y ↔ x + y - 1 = 0) →
  P' = (-1, 1) →
  (∀ x y, l ((P.1 + x) / 2) ((P.2 + y) / 2) ↔ l x y) →
  (P'.1 - P.1) * (P'.1 - P.1) + (P'.2 - P.2) * (P'.2 - P.2) =
    ((0 : ℝ) - P.1) * ((0 : ℝ) - P.1) + ((0 : ℝ) - P.2) * ((0 : ℝ) - P.2) :=
by sorry


end NUMINAMATH_CALUDE_symmetric_point_proof_l740_74026


namespace NUMINAMATH_CALUDE_triangle_area_l740_74093

theorem triangle_area (base height : ℝ) (h1 : base = 8.4) (h2 : height = 5.8) :
  (base * height) / 2 = 24.36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l740_74093


namespace NUMINAMATH_CALUDE_students_agreement_count_l740_74090

theorem students_agreement_count :
  let third_grade_count : ℕ := 154
  let fourth_grade_count : ℕ := 237
  third_grade_count + fourth_grade_count = 391 :=
by
  sorry

end NUMINAMATH_CALUDE_students_agreement_count_l740_74090


namespace NUMINAMATH_CALUDE_inequality_preservation_l740_74058

theorem inequality_preservation (m n : ℝ) (h1 : m < n) (h2 : n < 0) :
  m + 2 < n + 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l740_74058


namespace NUMINAMATH_CALUDE_egg_processing_plant_l740_74030

theorem egg_processing_plant (E : ℕ) : 
  (∃ A R : ℕ, 
    E = A + R ∧ 
    A = 388 * (R / 12) ∧
    (A + 37) / R = 405 / 3) →
  E = 125763 := by
sorry

end NUMINAMATH_CALUDE_egg_processing_plant_l740_74030


namespace NUMINAMATH_CALUDE_trig_identity_1_trig_identity_2_l740_74083

-- Problem 1
theorem trig_identity_1 (θ : ℝ) : (Real.sin θ - Real.cos θ) / (Real.tan θ - 1) = Real.cos θ := by
  sorry

-- Problem 2
theorem trig_identity_2 (α : ℝ) : Real.sin α ^ 4 - Real.cos α ^ 4 = 2 * Real.sin α ^ 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_1_trig_identity_2_l740_74083


namespace NUMINAMATH_CALUDE_min_selling_price_A_l740_74088

/-- Represents the number of units of model A purchased -/
def units_A : ℕ := 100

/-- Represents the number of units of model B purchased -/
def units_B : ℕ := 160 - units_A

/-- Represents the cost price of model A in yuan -/
def cost_A : ℕ := 150

/-- Represents the cost price of model B in yuan -/
def cost_B : ℕ := 350

/-- Represents the total cost of purchasing both models in yuan -/
def total_cost : ℕ := 36000

/-- Represents the minimum required gross profit in yuan -/
def min_gross_profit : ℕ := 11000

/-- Theorem stating that the minimum selling price of model A is 200 yuan -/
theorem min_selling_price_A : 
  ∃ (selling_price_A : ℕ), 
    selling_price_A = 200 ∧ 
    units_A * cost_A + units_B * cost_B = total_cost ∧
    units_A * (selling_price_A - cost_A) + units_B * (2 * (selling_price_A - cost_A)) ≥ min_gross_profit ∧
    ∀ (price : ℕ), price < selling_price_A → 
      units_A * (price - cost_A) + units_B * (2 * (price - cost_A)) < min_gross_profit :=
by
  sorry


end NUMINAMATH_CALUDE_min_selling_price_A_l740_74088


namespace NUMINAMATH_CALUDE_pet_store_cats_l740_74054

theorem pet_store_cats (siamese : ℝ) (house : ℝ) (added : ℝ) : 
  siamese = 13.0 → house = 5.0 → added = 10.0 → 
  siamese + house + added = 28.0 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_cats_l740_74054


namespace NUMINAMATH_CALUDE_inverse_matrices_solution_l740_74064

def matrix1 (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![5, -9; a, 12]
def matrix2 (b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![12, b; 3, 5]

theorem inverse_matrices_solution (a b : ℝ) :
  (matrix1 a) * (matrix2 b) = 1 → a = -3 ∧ b = 9 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_solution_l740_74064


namespace NUMINAMATH_CALUDE_polynomial_factorization_l740_74044

def polynomial (x y k : ℤ) : ℤ := x^2 + 5*x*y + x + k*y - k

def is_factorable (k : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ (x y : ℤ),
    polynomial x y k = (a*x + b*y + c) * (d*x + e*y + f)

theorem polynomial_factorization (k : ℤ) :
  is_factorable k ↔ k = 0 ∨ k = 15 ∨ k = -15 := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l740_74044


namespace NUMINAMATH_CALUDE_cost_per_item_jings_purchase_l740_74081

/-- Given a total cost and number of identical items, prove that the cost per item
    is equal to the total cost divided by the number of items. -/
theorem cost_per_item (total_cost : ℝ) (num_items : ℕ) (h : num_items > 0) :
  let cost_per_item := total_cost / num_items
  cost_per_item = total_cost / num_items :=
by
  sorry

/-- For Jing's purchase of 8 identical items with a total cost of $26,
    prove that the cost per item is $26 divided by 8. -/
theorem jings_purchase :
  let total_cost : ℝ := 26
  let num_items : ℕ := 8
  let cost_per_item := total_cost / num_items
  cost_per_item = 26 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_per_item_jings_purchase_l740_74081


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l740_74002

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sin x ^ 2 + 1/2

def is_monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (acute : A < π/2 ∧ B < π/2 ∧ C < π/2)
  (side_a : a = Real.sqrt 19)
  (side_b : b = 5)
  (angle_condition : f A = 0)

theorem triangle_area_theorem (t : Triangle) :
  is_monotone_increasing f (π/2) π ∧ 
  (1/2 * t.b * Real.sqrt (19 - t.b^2 + 2*t.b*Real.sqrt 19 * Real.cos t.A)) = 15 * Real.sqrt 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l740_74002


namespace NUMINAMATH_CALUDE_circle_equation_l740_74048

/-- Given a circle with center (1,2) and a point (-2,6) on the circle,
    prove that its standard equation is (x-1)^2 + (y-2)^2 = 25 -/
theorem circle_equation (x y : ℝ) :
  let center := (1, 2)
  let point := (-2, 6)
  let on_circle := (point.1 - center.1)^2 + (point.2 - center.2)^2 = (x - center.1)^2 + (y - center.2)^2
  on_circle → (x - 1)^2 + (y - 2)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l740_74048


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l740_74039

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : Nat
  sample_size : Nat
  start : Nat
  interval : Nat

/-- Generates the sequence of selected student numbers -/
def generate_sequence (s : SystematicSampling) : List Nat :=
  List.range s.sample_size |>.map (fun i => s.start + i * s.interval)

/-- Checks if all numbers in the sequence are valid student numbers -/
def valid_sequence (s : SystematicSampling) (seq : List Nat) : Prop :=
  seq.all (fun n => 1 ≤ n ∧ n ≤ s.total_students)

theorem systematic_sampling_theorem (s : SystematicSampling) :
  s.total_students = 60 →
  s.sample_size = 5 →
  s.start = 6 →
  s.interval = 12 →
  generate_sequence s = [6, 18, 30, 42, 54] ∧
  valid_sequence s (generate_sequence s) := by
  sorry

#eval generate_sequence { total_students := 60, sample_size := 5, start := 6, interval := 12 }

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l740_74039


namespace NUMINAMATH_CALUDE_circle_equation_proof_l740_74079

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := x/4 + y/2 = 1

/-- Point A is where the line intersects the x-axis -/
def point_A : ℝ × ℝ := (4, 0)

/-- Point B is where the line intersects the y-axis -/
def point_B : ℝ × ℝ := (0, 2)

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

/-- Theorem: The equation of the circle with diameter AB is x^2 + y^2 - 4x - 2y = 0 -/
theorem circle_equation_proof :
  ∀ x y : ℝ, line_equation x y →
  (∃ t : ℝ, x = t * (point_B.1 - point_A.1) + point_A.1 ∧
            y = t * (point_B.2 - point_A.2) + point_A.2) →
  circle_equation x y :=
sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l740_74079


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l740_74014

theorem complex_fraction_simplification :
  (5 - 7 * Complex.I) / (2 - 3 * Complex.I) = 31/13 + (1/13) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l740_74014


namespace NUMINAMATH_CALUDE_distinct_values_of_binomial_sum_l740_74080

theorem distinct_values_of_binomial_sum : ∃ (S : Finset ℕ),
  (∀ r : ℕ, r > 0 ∧ r + 1 ≤ 10 ∧ 17 - r ≤ 10 →
    (Nat.choose 10 (r + 1) + Nat.choose 10 (17 - r)) ∈ S) ∧
  Finset.card S = 2 := by
  sorry

end NUMINAMATH_CALUDE_distinct_values_of_binomial_sum_l740_74080


namespace NUMINAMATH_CALUDE_babylonian_square_58_l740_74072

-- Define the pattern function
def babylonian_square (n : Nat) : Nat × Nat :=
  let square := n * n
  let quotient := square / 60
  let remainder := square % 60
  if remainder = 0 then (quotient - 1, 60) else (quotient, remainder)

-- Theorem statement
theorem babylonian_square_58 : babylonian_square 58 = (56, 4) := by
  sorry

end NUMINAMATH_CALUDE_babylonian_square_58_l740_74072


namespace NUMINAMATH_CALUDE_bob_baked_36_more_l740_74082

/-- The number of additional peanut butter cookies Bob baked after the accident -/
def bob_additional_cookies (alice_initial : ℕ) (bob_initial : ℕ) (lost : ℕ) (alice_additional : ℕ) (final_total : ℕ) : ℕ :=
  final_total - ((alice_initial + bob_initial - lost) + alice_additional)

/-- Theorem stating that Bob baked 36 additional cookies given the problem conditions -/
theorem bob_baked_36_more (alice_initial bob_initial lost alice_additional final_total : ℕ) 
  (h1 : alice_initial = 74)
  (h2 : bob_initial = 7)
  (h3 : lost = 29)
  (h4 : alice_additional = 5)
  (h5 : final_total = 93) :
  bob_additional_cookies alice_initial bob_initial lost alice_additional final_total = 36 := by
  sorry

end NUMINAMATH_CALUDE_bob_baked_36_more_l740_74082


namespace NUMINAMATH_CALUDE_sixth_score_for_target_mean_l740_74071

def david_scores : List ℝ := [85, 88, 90, 82, 94]
def target_mean : ℝ := 90

theorem sixth_score_for_target_mean :
  ∃ (x : ℝ), (david_scores.sum + x) / 6 = target_mean ∧ x = 101 := by
sorry

end NUMINAMATH_CALUDE_sixth_score_for_target_mean_l740_74071


namespace NUMINAMATH_CALUDE_tan_fifteen_equals_sqrt_three_l740_74029

theorem tan_fifteen_equals_sqrt_three : (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_equals_sqrt_three_l740_74029


namespace NUMINAMATH_CALUDE_clock_face_partition_l740_74059

noncomputable def clockFaceAreas (r : ℝ) : (ℝ × ℝ × ℝ × ℝ) :=
  let t₁ := (Real.pi + 2 * Real.sqrt 3 - 6) / 12 * r^2
  let t₂ := (Real.pi - Real.sqrt 3) / 6 * r^2
  let t₃ := (7 * Real.pi + 2 * Real.sqrt 3 - 6) / 12 * r^2
  (t₁, t₂, t₂, t₃)

theorem clock_face_partition (r : ℝ) (h : r > 0) :
  let (t₁, t₂, t₂', t₃) := clockFaceAreas r
  t₁ + t₂ + t₂' + t₃ = Real.pi * r^2 ∧
  t₂ = t₂' ∧
  t₁ > 0 ∧ t₂ > 0 ∧ t₃ > 0 :=
by sorry

end NUMINAMATH_CALUDE_clock_face_partition_l740_74059


namespace NUMINAMATH_CALUDE_chips_juice_weight_difference_l740_74004

/-- Given that 2 bags of chips weigh 800 g and 5 bags of chips and 4 bottles of juice
    together weigh 2200 g, prove that a bag of chips is 350 g heavier than a bottle of juice. -/
theorem chips_juice_weight_difference :
  (∀ (chips_weight bottle_weight : ℕ),
    2 * chips_weight = 800 →
    5 * chips_weight + 4 * bottle_weight = 2200 →
    chips_weight - bottle_weight = 350) :=
by sorry

end NUMINAMATH_CALUDE_chips_juice_weight_difference_l740_74004


namespace NUMINAMATH_CALUDE_plums_given_to_sam_l740_74087

/-- Given Melanie's plum picking and sharing scenario, prove the number of plums given to Sam. -/
theorem plums_given_to_sam 
  (original_plums : ℕ) 
  (plums_left : ℕ) 
  (h1 : original_plums = 7)
  (h2 : plums_left = 4)
  : original_plums - plums_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_plums_given_to_sam_l740_74087


namespace NUMINAMATH_CALUDE_arrangement_theorem_l740_74057

/-- The number of ways to arrange 5 people in a row with two specific people not adjacent -/
def arrangement_count : ℕ := 72

/-- The number of people to be arranged -/
def total_people : ℕ := 5

/-- The number of people who can be freely arranged -/
def free_people : ℕ := total_people - 2

/-- The number of positions where the two specific people can be inserted -/
def insertion_positions : ℕ := free_people + 1

theorem arrangement_theorem :
  arrangement_count = (free_people.factorial) * (insertion_positions.factorial / 2) :=
sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l740_74057


namespace NUMINAMATH_CALUDE_problem_solution_l740_74043

def f (m : ℝ) (x : ℝ) : ℝ := |x - 2| - m

theorem problem_solution :
  (∃ m : ℝ, ∀ x : ℝ, f m (x + 2) ≤ 0 ↔ x ∈ Set.Icc (-1) 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a^2 + b^2 + c^2 = 1 →
    a + 2*b + 3*c ≤ Real.sqrt 14) ∧
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 1 ∧
    a + 2*b + 3*c = Real.sqrt 14) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l740_74043


namespace NUMINAMATH_CALUDE_arrangement_count_l740_74027

/-- The number of representatives in unit A -/
def unitA : ℕ := 7

/-- The number of representatives in unit B -/
def unitB : ℕ := 3

/-- The total number of elements to arrange (treating unit B as one element) -/
def totalElements : ℕ := unitA + 1

/-- The number of possible arrangements -/
def numArrangements : ℕ := (Nat.factorial totalElements) * (Nat.factorial unitB)

theorem arrangement_count : numArrangements = 241920 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_l740_74027


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l740_74085

/-- The equation of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

/-- The equation of the line -/
def line (m x y : ℝ) : Prop := (m+2)*x - (m+4)*y + 2-m = 0

/-- Theorem stating that the line always intersects the ellipse -/
theorem line_intersects_ellipse :
  ∀ m : ℝ, ∃ x y : ℝ, ellipse x y ∧ line m x y := by sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l740_74085


namespace NUMINAMATH_CALUDE_certain_number_proof_l740_74077

theorem certain_number_proof : ∃ n : ℕ, n * 40 = 173 * 240 ∧ n = 1038 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l740_74077


namespace NUMINAMATH_CALUDE_age_problem_l740_74067

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 32 → 
  b = 12 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l740_74067


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_normal_distribution_l740_74010

-- Define the arithmetic mean and standard deviation
variable (μ : ℝ) -- arithmetic mean
def σ : ℝ := 1.5 -- standard deviation

-- Define the relationship between the mean, standard deviation, and the given value
def value_two_std_below_mean : ℝ := μ - 2 * σ

-- State the theorem
theorem arithmetic_mean_of_normal_distribution :
  value_two_std_below_mean = 12 → μ = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_normal_distribution_l740_74010


namespace NUMINAMATH_CALUDE_isabel_weekly_run_distance_l740_74019

/-- Calculates the total distance run in a week given a circuit length, 
    morning runs, afternoon runs, and number of days. -/
def total_distance_run (circuit_length : ℕ) (morning_runs : ℕ) (afternoon_runs : ℕ) (days : ℕ) : ℕ :=
  (circuit_length * (morning_runs + afternoon_runs) * days)

/-- Proves that running a 365-meter circuit 7 times in the morning and 3 times 
    in the afternoon for 7 days results in a total distance of 25550 meters. -/
theorem isabel_weekly_run_distance :
  total_distance_run 365 7 3 7 = 25550 := by
  sorry

#eval total_distance_run 365 7 3 7

end NUMINAMATH_CALUDE_isabel_weekly_run_distance_l740_74019


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l740_74056

theorem parallelogram_side_sum (x y : ℝ) : 
  12 = 10 * y - 2 ∧ 15 = 3 * x + 6 → x + y = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l740_74056


namespace NUMINAMATH_CALUDE_triangle_problem_l740_74075

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  A = π / 4 →
  b = Real.sqrt 6 →
  (1 / 2) * b * c * Real.sin A = (3 + Real.sqrt 3) / 2 →
  -- Definitions from cosine rule
  a = Real.sqrt (b^2 + c^2 - 2*b*c*(Real.cos A)) →
  Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) →
  -- Conclusion
  c = 1 + Real.sqrt 3 ∧ B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l740_74075


namespace NUMINAMATH_CALUDE_jordan_empty_boxes_l740_74020

/-- A structure representing the distribution of items in boxes -/
structure BoxDistribution where
  total : ℕ
  pencils : ℕ
  pens : ℕ
  markers : ℕ
  pencils_and_pens : ℕ
  pencils_and_markers : ℕ
  pens_and_markers : ℕ

/-- The number of boxes with no items, given a box distribution -/
def empty_boxes (d : BoxDistribution) : ℕ :=
  d.total - (d.pencils + d.pens + d.markers - d.pencils_and_pens - d.pencils_and_markers - d.pens_and_markers)

/-- The specific box distribution from the problem -/
def jordan_boxes : BoxDistribution :=
  { total := 15
  , pencils := 8
  , pens := 5
  , markers := 3
  , pencils_and_pens := 2
  , pencils_and_markers := 1
  , pens_and_markers := 1 }

/-- Theorem stating that the number of empty boxes in Jordan's distribution is 3 -/
theorem jordan_empty_boxes :
    empty_boxes jordan_boxes = 3 := by
  sorry


end NUMINAMATH_CALUDE_jordan_empty_boxes_l740_74020


namespace NUMINAMATH_CALUDE_linear_equation_condition_l740_74078

theorem linear_equation_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ k m n : ℝ, (a - 2) * x^(|a| - 1) + 3 * y = k * x + m * y + n) → 
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l740_74078


namespace NUMINAMATH_CALUDE_cara_catches_47_l740_74047

/-- The number of animals Martha's cat catches -/
def martha_animals : ℕ := 3 + 7

/-- The number of animals Cara's cat catches -/
def cara_animals : ℕ := 5 * martha_animals - 3

/-- Theorem stating that Cara's cat catches 47 animals -/
theorem cara_catches_47 : cara_animals = 47 := by
  sorry

end NUMINAMATH_CALUDE_cara_catches_47_l740_74047


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l740_74055

theorem inequality_system_solution_range (k : ℝ) : 
  (∃! x : ℤ, (x^2 - 2*x - 8 > 0) ∧ (2*x^2 + (2*k + 7)*x + 7*k < 0)) ↔ 
  (k ∈ Set.Icc (-5 : ℝ) 3 ∪ Set.Ioc 4 5) :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l740_74055


namespace NUMINAMATH_CALUDE_volunteers_2008_l740_74068

/-- The expected number of volunteers after a given number of years, 
    given an initial number and annual increase rate. -/
def expected_volunteers (initial : ℕ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate) ^ years

/-- Theorem: Given 500 initial volunteers in 2005 and a 20% annual increase,
    the expected number of volunteers in 2008 is 864. -/
theorem volunteers_2008 : 
  ⌊expected_volunteers 500 0.2 3⌋ = 864 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_2008_l740_74068


namespace NUMINAMATH_CALUDE_pam_has_ten_bags_l740_74061

/-- Represents the number of apples in each of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- Represents the ratio of apples in Pam's bags to Gerald's bags -/
def pam_to_gerald_ratio : ℕ := 3

/-- Represents the total number of apples Pam has -/
def pam_total_apples : ℕ := 1200

/-- Calculates the number of bags Pam has -/
def pam_bag_count : ℕ := pam_total_apples / (geralds_bag_count * pam_to_gerald_ratio)

theorem pam_has_ten_bags : pam_bag_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_pam_has_ten_bags_l740_74061


namespace NUMINAMATH_CALUDE_probability_seven_white_three_black_l740_74063

/-- The probability of drawing first a black ball and then a white ball from a bag -/
def probability_black_then_white (white_balls black_balls : ℕ) : ℚ :=
  let total_balls := white_balls + black_balls
  let prob_black_first := black_balls / total_balls
  let prob_white_second := white_balls / (total_balls - 1)
  prob_black_first * prob_white_second

/-- Theorem stating the probability of drawing first a black ball and then a white ball
    from a bag containing 7 white balls and 3 black balls is 7/30 -/
theorem probability_seven_white_three_black :
  probability_black_then_white 7 3 = 7 / 30 := by
  sorry

end NUMINAMATH_CALUDE_probability_seven_white_three_black_l740_74063


namespace NUMINAMATH_CALUDE_percentage_problem_l740_74076

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.3 * x = 120 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l740_74076


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l740_74073

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4^7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l740_74073


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l740_74016

def M : Set ℝ := {1, 2}
def N (a : ℝ) : Set ℝ := {a^2}

theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → N a ⊆ M) ∧
  (∃ a : ℝ, a ≠ 1 ∧ N a ⊆ M) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l740_74016


namespace NUMINAMATH_CALUDE_counterexample_necessity_l740_74031

-- Define the concept of a mathematical statement
def MathStatement : Type := String

-- Define the concept of a proof method
inductive ProofMethod
| Direct : ProofMethod
| Counterexample : ProofMethod
| Other : ProofMethod

-- Define a property of mathematical statements
def CanBeProvedDirectly (s : MathStatement) : Prop := sorry

-- Define the theorem to be proved
theorem counterexample_necessity (s : MathStatement) :
  ¬(∀ s, ¬(CanBeProvedDirectly s) → (∀ m : ProofMethod, m = ProofMethod.Counterexample)) :=
sorry

end NUMINAMATH_CALUDE_counterexample_necessity_l740_74031


namespace NUMINAMATH_CALUDE_marching_band_total_weight_l740_74070

def trumpet_weight : ℕ := 5
def clarinet_weight : ℕ := 5
def trombone_weight : ℕ := 10
def tuba_weight : ℕ := 20
def drum_weight : ℕ := 15

def trumpet_count : ℕ := 6
def clarinet_count : ℕ := 9
def trombone_count : ℕ := 8
def tuba_count : ℕ := 3
def drum_count : ℕ := 2

theorem marching_band_total_weight :
  trumpet_weight * trumpet_count +
  clarinet_weight * clarinet_count +
  trombone_weight * trombone_count +
  tuba_weight * tuba_count +
  drum_weight * drum_count = 245 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_total_weight_l740_74070


namespace NUMINAMATH_CALUDE_printer_equation_l740_74096

/-- The equation for determining the time of the second printer to print 1000 flyers -/
theorem printer_equation (x : ℝ) : 
  (1000 : ℝ) > 0 → x > 0 → (
    (1000 / 10 + 1000 / x = 1000 / 4) ↔ 
    (1 / 10 + 1 / x = 1 / 4)
  ) := by sorry

end NUMINAMATH_CALUDE_printer_equation_l740_74096


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l740_74074

-- Problem 1
theorem problem_1 (a : ℝ) (h : Real.sqrt a + 1 / Real.sqrt a = 3) :
  (a^2 + 1/a^2 + 3) / (4*a + 1/(4*a)) = 10 * Real.sqrt 5 := by sorry

-- Problem 2
theorem problem_2 :
  (1 - Real.log 3 / Real.log 6)^2 + (Real.log 2 / Real.log 6) * (Real.log 18 / Real.log 6) * (Real.log 6 / Real.log 4) = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l740_74074


namespace NUMINAMATH_CALUDE_negative_a_sufficient_not_necessary_l740_74046

/-- Represents a quadratic equation ax² + 2x + 1 = 0 -/
structure QuadraticEquation (a : ℝ) where
  eq : ∀ x : ℝ, a * x^2 + 2 * x + 1 = 0 → x ∈ {x | a * x^2 + 2 * x + 1 = 0}

/-- Predicate indicating if an equation has at least one negative root -/
def has_negative_root (eq : QuadraticEquation a) : Prop :=
  ∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

/-- The main theorem to prove -/
theorem negative_a_sufficient_not_necessary (a : ℝ) :
  (a < 0 → ∀ eq : QuadraticEquation a, has_negative_root eq) ∧
  (∃ a : ℝ, a ≥ 0 ∧ ∃ eq : QuadraticEquation a, has_negative_root eq) :=
sorry

end NUMINAMATH_CALUDE_negative_a_sufficient_not_necessary_l740_74046


namespace NUMINAMATH_CALUDE_equation_solution_l740_74098

theorem equation_solution : 
  ∃ x : ℚ, (x + 5) / (x - 3) = (x - 2) / (x + 2) ∧ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l740_74098


namespace NUMINAMATH_CALUDE_fish_population_estimate_l740_74066

/-- Estimate the fish population in a pond given tagging and sampling data --/
theorem fish_population_estimate
  (initial_tagged : ℕ)
  (august_sample : ℕ)
  (august_tagged : ℕ)
  (left_pond_ratio : ℚ)
  (new_fish_ratio : ℚ)
  (h_initial_tagged : initial_tagged = 50)
  (h_august_sample : august_sample = 80)
  (h_august_tagged : august_tagged = 4)
  (h_left_pond : left_pond_ratio = 3/10)
  (h_new_fish : new_fish_ratio = 45/100)
  (h_representative_sample : True)  -- Assuming the sample is representative
  (h_negligible_tag_loss : True)    -- Assuming tag loss is negligible
  : ↑initial_tagged * (august_sample * (1 - new_fish_ratio)) / august_tagged = 550 := by
  sorry


end NUMINAMATH_CALUDE_fish_population_estimate_l740_74066
