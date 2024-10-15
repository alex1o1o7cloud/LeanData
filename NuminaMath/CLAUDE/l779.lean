import Mathlib

namespace NUMINAMATH_CALUDE_triangle_side_length_l779_77902

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = π/3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l779_77902


namespace NUMINAMATH_CALUDE_product_quotient_l779_77969

theorem product_quotient (a b c d e f : ℚ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 750)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_product_quotient_l779_77969


namespace NUMINAMATH_CALUDE_circular_ring_area_l779_77975

/-- Given a regular n-gon with area t, the area of the circular ring formed by
    its inscribed and circumscribed circles is (π * t * tan(180°/n)) / n. -/
theorem circular_ring_area (n : ℕ) (t : ℝ) (h1 : n ≥ 3) (h2 : t > 0) :
  let T := (Real.pi * t * Real.tan (Real.pi / n)) / n
  ∃ (r R : ℝ), r > 0 ∧ R > r ∧
    t = n * r^2 * Real.sin (Real.pi / n) * Real.cos (Real.pi / n) ∧
    R = r / Real.cos (Real.pi / n) ∧
    T = Real.pi * (R^2 - r^2) :=
by sorry

end NUMINAMATH_CALUDE_circular_ring_area_l779_77975


namespace NUMINAMATH_CALUDE_E_parity_l779_77920

def E : ℕ → ℤ
  | 0 => 2
  | 1 => 3
  | 2 => 4
  | n + 3 => E (n + 2) + 2 * E (n + 1) - E n

theorem E_parity : (E 10 % 2 = 1) ∧ (E 11 % 2 = 0) ∧ (E 12 % 2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_E_parity_l779_77920


namespace NUMINAMATH_CALUDE_largest_valid_domain_l779_77904

def is_valid_domain (S : Set ℝ) : Prop :=
  ∃ g : ℝ → ℝ, 
    (∀ x ∈ S, (1 / x) ∈ S) ∧ 
    (∀ x ∈ S, g x + g (1 / x) = x^2)

theorem largest_valid_domain : 
  is_valid_domain {-1, 1} ∧ 
  ∀ S : Set ℝ, is_valid_domain S → S ⊆ {-1, 1} :=
sorry

end NUMINAMATH_CALUDE_largest_valid_domain_l779_77904


namespace NUMINAMATH_CALUDE_probability_three_same_color_l779_77929

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def total_balls : ℕ := white_balls + black_balls
def drawn_balls : ℕ := 3

def probability_same_color : ℚ :=
  (Nat.choose white_balls drawn_balls + Nat.choose black_balls drawn_balls) /
  Nat.choose total_balls drawn_balls

theorem probability_three_same_color :
  probability_same_color = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_same_color_l779_77929


namespace NUMINAMATH_CALUDE_parametric_to_standard_equation_l779_77950

theorem parametric_to_standard_equation (x y α : ℝ) :
  x = Real.sqrt 3 * Real.cos α + 2 ∧ 
  y = Real.sqrt 3 * Real.sin α - 3 →
  (x - 2)^2 + (y + 3)^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_parametric_to_standard_equation_l779_77950


namespace NUMINAMATH_CALUDE_exponent_multiplication_l779_77961

theorem exponent_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l779_77961


namespace NUMINAMATH_CALUDE_cat_path_tiles_l779_77973

def garden_width : ℕ := 12
def garden_length : ℕ := 20
def tile_size : ℕ := 2
def tiles_width : ℕ := garden_width / tile_size
def tiles_length : ℕ := garden_length / tile_size

theorem cat_path_tiles : 
  tiles_width + tiles_length - Nat.gcd tiles_width tiles_length - 1 = 13 := by
  sorry

end NUMINAMATH_CALUDE_cat_path_tiles_l779_77973


namespace NUMINAMATH_CALUDE_point_reflection_x_axis_l779_77989

/-- Given a point P(-1,2) in the Cartesian coordinate system, 
    its coordinates with respect to the x-axis are (-1,-2). -/
theorem point_reflection_x_axis : 
  let P : ℝ × ℝ := (-1, 2)
  let reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  reflect_x P = (-1, -2) := by sorry

end NUMINAMATH_CALUDE_point_reflection_x_axis_l779_77989


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l779_77901

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (2 * x^2 - 8 * x - 10 = 5 * x + 20) → 
  ∃ (y : ℝ), (2 * y^2 - 8 * y - 10 = 5 * y + 20) ∧ (x + y = 13/2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l779_77901


namespace NUMINAMATH_CALUDE_point_on_line_for_all_k_l779_77993

/-- The point P lies on the line (k+2)x + (1-k)y - 4k - 5 = 0 for all values of k. -/
theorem point_on_line_for_all_k :
  ∀ (k : ℝ), (k + 2) * 3 + (1 - k) * (-1) - 4 * k - 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_for_all_k_l779_77993


namespace NUMINAMATH_CALUDE_fifth_month_sale_l779_77962

def sales_first_four : List ℕ := [6435, 6927, 6855, 7230]
def sale_sixth : ℕ := 6191
def average_sale : ℕ := 6700
def num_months : ℕ := 6

theorem fifth_month_sale :
  let total_sales := average_sale * num_months
  let sum_known_sales := sales_first_four.sum + sale_sixth
  total_sales - sum_known_sales = 6562 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l779_77962


namespace NUMINAMATH_CALUDE_margo_walking_distance_l779_77981

/-- Proves that Margo's total walking distance is 2 miles given the specified conditions -/
theorem margo_walking_distance
  (time_to_friend : ℝ)
  (time_to_return : ℝ)
  (average_speed : ℝ)
  (h1 : time_to_friend = 15)
  (h2 : time_to_return = 25)
  (h3 : average_speed = 3)
  : (time_to_friend + time_to_return) / 60 * average_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_margo_walking_distance_l779_77981


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l779_77955

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l779_77955


namespace NUMINAMATH_CALUDE_cylinder_volume_l779_77939

/-- The volume of a cylinder with base radius 3 and lateral area 12π is 18π. -/
theorem cylinder_volume (r h : ℝ) : r = 3 → 2 * π * r * h = 12 * π → π * r^2 * h = 18 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l779_77939


namespace NUMINAMATH_CALUDE_number_problem_l779_77996

theorem number_problem : ∃ x : ℚ, (x / 6) * 12 = 11 ∧ x = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l779_77996


namespace NUMINAMATH_CALUDE_sum_of_digits_82_l779_77916

theorem sum_of_digits_82 :
  ∀ (tens ones : ℕ),
    tens * 10 + ones = 82 →
    tens - ones = 6 →
    tens + ones = 10 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_82_l779_77916


namespace NUMINAMATH_CALUDE_locus_of_centers_l779_77912

-- Define the two circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the property of being externally tangent to C₁ and internally tangent to C₂
def externally_internally_tangent (a b r : ℝ) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ (x - a)^2 + (y - b)^2 = (r + 2)^2 ∧
                C₂ x y ∧ (x - a)^2 + (y - b)^2 = (3 - r)^2

-- State the theorem
theorem locus_of_centers : 
  ∀ (a b : ℝ), (∃ r : ℝ, externally_internally_tangent a b r) ↔ 
  16 * a^2 + 25 * b^2 - 48 * a - 64 = 0 := by sorry

end NUMINAMATH_CALUDE_locus_of_centers_l779_77912


namespace NUMINAMATH_CALUDE_fraction_property_l779_77933

theorem fraction_property (n : ℕ+) : 
  (∃ (a b c d e f : ℕ), 
    (1 : ℚ) / (2*n + 1) = (a*100000 + b*10000 + c*1000 + d*100 + e*10 + f) / 999999 ∧ 
    a + b + c + d + e + f = 999) ↔ 
  (2*n + 1 = 7 ∨ 2*n + 1 = 13) :=
sorry

end NUMINAMATH_CALUDE_fraction_property_l779_77933


namespace NUMINAMATH_CALUDE_trailing_zeros_bound_l779_77928

/-- The number of trailing zeros in the base-b representation of n! -/
def trailing_zeros (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The largest prime factor of a natural number -/
def largest_prime_factor (b : ℕ) : ℕ := sorry

theorem trailing_zeros_bound {b : ℕ} (hb : b ≥ 2) :
  ∀ n : ℕ, trailing_zeros n b < n / (largest_prime_factor b - 1) := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_bound_l779_77928


namespace NUMINAMATH_CALUDE_circle_line_theorem_l779_77983

/-- Two circles passing through a common point -/
structure TwoCircles where
  D1 : ℝ
  E1 : ℝ
  D2 : ℝ
  E2 : ℝ
  h1 : 2^2 + (-1)^2 + D1*2 + E1*(-1) - 3 = 0
  h2 : 2^2 + (-1)^2 + D2*2 + E2*(-1) - 3 = 0

/-- The equation of the line passing through (D1, E1) and (D2, E2) -/
def line_equation (c : TwoCircles) (x y : ℝ) : Prop :=
  2*x - y + 2 = 0

theorem circle_line_theorem (c : TwoCircles) :
  line_equation c c.D1 c.E1 ∧ line_equation c c.D2 c.E2 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_theorem_l779_77983


namespace NUMINAMATH_CALUDE_inequality_problem_l779_77945

theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (∀ x y z : ℝ, x < y ∧ y < z ∧ x * z < 0 → x * y^2 < x * z^2 → False) ∧
  (a * b > a * c) ∧
  (c * (b - a) > 0) ∧
  (a * c * (a - c) < 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_problem_l779_77945


namespace NUMINAMATH_CALUDE_square_playground_area_l779_77922

theorem square_playground_area (w : ℝ) (s : ℝ) : 
  s = 3 * w + 10 →
  4 * s = 480 →
  s * s = 14400 := by
sorry

end NUMINAMATH_CALUDE_square_playground_area_l779_77922


namespace NUMINAMATH_CALUDE_angle_complement_problem_l779_77907

theorem angle_complement_problem (x : ℝ) : 
  x + 2 * (4 * x + 10) = 90 → x = 70 / 9 :=
by sorry

end NUMINAMATH_CALUDE_angle_complement_problem_l779_77907


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l779_77978

/-- Given two vectors a and b in ℝ², where a = (1, 2) and b = (2x, -3),
    and a is parallel to b, prove that x = -3/4 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![2*x, -3]
  (∃ (k : ℝ), k ≠ 0 ∧ b = k • a) →
  x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l779_77978


namespace NUMINAMATH_CALUDE_cubic_expansion_sum_l779_77953

theorem cubic_expansion_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (2 * x + 1)^3 = a₀ * x^3 + a₁ * x^2 + a₂ * x + a₃) →
  a₁ + a₃ = 13 := by
sorry

end NUMINAMATH_CALUDE_cubic_expansion_sum_l779_77953


namespace NUMINAMATH_CALUDE_max_abs_z_quadratic_equation_l779_77932

/-- Given complex numbers a, b, c, z and a real number k satisfying certain conditions,
    the maximum value of |z| is (k^3 + √(k^6 + 4k^3)) / 2. -/
theorem max_abs_z_quadratic_equation (a b c z d : ℂ) (k : ℝ) 
    (h1 : Complex.abs a = Complex.abs d)
    (h2 : Complex.abs d > 0)
    (h3 : b = k • d)
    (h4 : c = k^2 • d)
    (h5 : a * z^2 + b * z + c = 0) :
    Complex.abs z ≤ (k^3 + Real.sqrt (k^6 + 4 * k^3)) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_quadratic_equation_l779_77932


namespace NUMINAMATH_CALUDE_yoga_time_calculation_l779_77910

/-- Calculates the yoga time given exercise ratios and bicycle riding time -/
def yoga_time (gym_bike_ratio : Rat) (yoga_exercise_ratio : Rat) (bike_time : ℕ) : ℕ :=
  let gym_time := (gym_bike_ratio.num * bike_time) / gym_bike_ratio.den
  let total_exercise_time := gym_time + bike_time
  ((yoga_exercise_ratio.num * total_exercise_time) / yoga_exercise_ratio.den).toNat

/-- Proves that given the specified ratios and bicycle riding time, the yoga time is 20 minutes -/
theorem yoga_time_calculation :
  yoga_time (2 / 3) (2 / 3) 18 = 20 := by
  sorry

#eval yoga_time (2 / 3) (2 / 3) 18

end NUMINAMATH_CALUDE_yoga_time_calculation_l779_77910


namespace NUMINAMATH_CALUDE_average_of_sequence_l779_77934

theorem average_of_sequence (z : ℝ) : (0 + 3*z + 6*z + 12*z + 24*z) / 5 = 9*z := by
  sorry

end NUMINAMATH_CALUDE_average_of_sequence_l779_77934


namespace NUMINAMATH_CALUDE_inequality_range_l779_77992

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, |x + 1| - |x - 2| > k) → k < -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l779_77992


namespace NUMINAMATH_CALUDE_range_of_ab_plus_a_plus_b_l779_77914

def f (x : ℝ) := |x^2 + 2*x - 1|

theorem range_of_ab_plus_a_plus_b 
  (a b : ℝ) 
  (h1 : a < b) 
  (h2 : b < -1) 
  (h3 : f a = f b) :
  ∀ y : ℝ, (∃ x : ℝ, a < x ∧ x < b ∧ y = a*b + a + b) → -1 < y ∧ y < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_ab_plus_a_plus_b_l779_77914


namespace NUMINAMATH_CALUDE_only_B_is_quadratic_l779_77970

-- Define the structure of a general function
structure GeneralFunction where
  f : ℝ → ℝ

-- Define what it means for a function to be quadratic
def is_quadratic (f : GeneralFunction) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f.f x = a * x^2 + b * x + c

-- Define the given functions
def function_A : GeneralFunction :=
  { f := λ x => 2 * x + 1 }

def function_B : GeneralFunction :=
  { f := λ x => -5 * x^2 - 3 }

def function_C (a b c : ℝ) : GeneralFunction :=
  { f := λ x => a * x^2 + b * x + c }

def function_D : GeneralFunction :=
  { f := λ x => x^3 + x + 1 }

-- State the theorem
theorem only_B_is_quadratic :
  ¬ is_quadratic function_A ∧
  is_quadratic function_B ∧
  (∃ a b c, ¬ is_quadratic (function_C a b c)) ∧
  ¬ is_quadratic function_D :=
sorry

end NUMINAMATH_CALUDE_only_B_is_quadratic_l779_77970


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l779_77976

theorem least_number_with_remainder (n : Nat) : n = 125 ↔ 
  (n % 12 = 5 ∧ ∀ m : Nat, m % 12 = 5 → m ≥ n) := by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l779_77976


namespace NUMINAMATH_CALUDE_smallest_n_with_forty_percent_leftmost_one_l779_77959

/-- Returns true if the leftmost digit of n is 1 -/
def leftmost_digit_is_one (n : ℕ) : Bool := sorry

/-- Returns the count of numbers from 1 to n (inclusive) with leftmost digit 1 -/
def count_leftmost_one (n : ℕ) : ℕ := sorry

theorem smallest_n_with_forty_percent_leftmost_one :
  ∀ N : ℕ,
    N > 2017 →
    (count_leftmost_one N : ℚ) / N = 2 / 5 →
    N ≥ 1481480 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_forty_percent_leftmost_one_l779_77959


namespace NUMINAMATH_CALUDE_triangle_min_perimeter_l779_77906

theorem triangle_min_perimeter (a b x : ℕ) (ha : a = 36) (hb : b = 50) : 
  (a + b > x ∧ a + x > b ∧ b + x > a) → (a + b + x ≥ 101) := by
  sorry

end NUMINAMATH_CALUDE_triangle_min_perimeter_l779_77906


namespace NUMINAMATH_CALUDE_chicken_crossed_road_l779_77944

/-- The number of cars dodged by a chicken crossing a road, given the initial and final feather counts. -/
def cars_dodged (initial_feathers final_feathers : ℕ) : ℕ :=
  (initial_feathers - final_feathers) / 2

/-- Theorem stating that the chicken dodged 23 cars given the problem conditions. -/
theorem chicken_crossed_road (initial_feathers final_feathers : ℕ) 
  (h1 : initial_feathers = 5263)
  (h2 : final_feathers = 5217) :
  cars_dodged initial_feathers final_feathers = 23 := by
  sorry

#eval cars_dodged 5263 5217

end NUMINAMATH_CALUDE_chicken_crossed_road_l779_77944


namespace NUMINAMATH_CALUDE_lamp_probability_l779_77900

theorem lamp_probability : Real → Prop :=
  fun p =>
    let total_length : Real := 6
    let min_distance : Real := 2
    p = (total_length - 2 * min_distance) / total_length

#check lamp_probability (1/3)

end NUMINAMATH_CALUDE_lamp_probability_l779_77900


namespace NUMINAMATH_CALUDE_pythagorean_triple_5_12_13_l779_77903

theorem pythagorean_triple_5_12_13 : 5^2 + 12^2 = 13^2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_5_12_13_l779_77903


namespace NUMINAMATH_CALUDE_buffet_meal_combinations_l779_77935

def meat_options : ℕ := 4
def vegetable_options : ℕ := 5
def dessert_options : ℕ := 5
def vegetables_to_choose : ℕ := 3

theorem buffet_meal_combinations :
  meat_options * Nat.choose vegetable_options vegetables_to_choose * dessert_options = 200 := by
  sorry

end NUMINAMATH_CALUDE_buffet_meal_combinations_l779_77935


namespace NUMINAMATH_CALUDE_remainder_7n_mod_3_l779_77966

theorem remainder_7n_mod_3 (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_3_l779_77966


namespace NUMINAMATH_CALUDE_total_laundry_time_l779_77940

/-- Represents the time in minutes for washing and drying a load of laundry -/
structure LaundryTime where
  washing : ℕ
  drying : ℕ

/-- Calculates the total time for a single load of laundry -/
def totalTimeForLoad (load : LaundryTime) : ℕ :=
  load.washing + load.drying

/-- The time for the whites load -/
def whites : LaundryTime := ⟨72, 50⟩

/-- The time for the darks load -/
def darks : LaundryTime := ⟨58, 65⟩

/-- The time for the colors load -/
def colors : LaundryTime := ⟨45, 54⟩

/-- Theorem stating that the total time for all three loads is 344 minutes -/
theorem total_laundry_time :
  totalTimeForLoad whites + totalTimeForLoad darks + totalTimeForLoad colors = 344 := by
  sorry

end NUMINAMATH_CALUDE_total_laundry_time_l779_77940


namespace NUMINAMATH_CALUDE_train_length_l779_77942

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 127 → time = 17 → ∃ (length : ℝ), abs (length - 599.76) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l779_77942


namespace NUMINAMATH_CALUDE_largest_unrepresentable_number_l779_77995

theorem largest_unrepresentable_number (a b : ℤ) (ha : a > 1) (hb : b > 1) :
  ∃ n : ℤ, n = 47 ∧ (∀ m : ℤ, m > n → ∃ x y : ℤ, m = 7*a + 5*b + 7*x + 5*y) ∧
  (¬∃ x y : ℤ, n = 7*a + 5*b + 7*x + 5*y) := by
sorry

end NUMINAMATH_CALUDE_largest_unrepresentable_number_l779_77995


namespace NUMINAMATH_CALUDE_simplify_expression_l779_77936

theorem simplify_expression (y : ℝ) : (3 - Real.sqrt (y^2 - 9))^2 = y^2 - 6 * Real.sqrt (y^2 - 9) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l779_77936


namespace NUMINAMATH_CALUDE_sum_of_cubes_l779_77925

theorem sum_of_cubes (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a * b + a * c + b * c = 2) 
  (h3 : a * b * c = 5) : 
  a^3 + b^3 + c^3 = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l779_77925


namespace NUMINAMATH_CALUDE_product_is_112015_l779_77967

/-- Represents a three-digit number with distinct non-zero digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones
  non_zero : hundreds ≠ 0 ∧ tens ≠ 0 ∧ ones ≠ 0
  valid_range : hundreds < 10 ∧ tens < 10 ∧ ones < 10

def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem product_is_112015 (iks ksi : ThreeDigitNumber) 
  (h1 : iks.hundreds = ksi.ones ∧ iks.tens = ksi.hundreds ∧ iks.ones = ksi.tens)
  (h2 : ∃ (c i k : Nat), c ≠ i ∧ c ≠ k ∧ i ≠ k ∧ 
    c = max iks.hundreds (max iks.tens iks.ones) ∧
    c = max ksi.hundreds (max ksi.tens ksi.ones))
  (h3 : ∃ (p : Nat), p = to_nat iks * to_nat ksi ∧ 
    (∃ (d1 d2 d3 d4 d5 d6 : Nat),
      p = 100000 * d1 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6 ∧
      d1 = c ∧ d2 = c ∧ d3 = c ∧
      ((d4 = i ∧ d5 = k ∧ d6 = 0) ∨ 
       (d4 = i ∧ d5 = 0 ∧ d6 = k) ∨ 
       (d4 = k ∧ d5 = i ∧ d6 = 0) ∨ 
       (d4 = k ∧ d5 = 0 ∧ d6 = i) ∨ 
       (d4 = 0 ∧ d5 = i ∧ d6 = k) ∨ 
       (d4 = 0 ∧ d5 = k ∧ d6 = i))))
  : to_nat iks * to_nat ksi = 112015 := by
  sorry

end NUMINAMATH_CALUDE_product_is_112015_l779_77967


namespace NUMINAMATH_CALUDE_january_salary_l779_77913

/-- Represents the monthly salary for a person -/
structure MonthlySalary where
  january : ℕ
  february : ℕ
  march : ℕ
  april : ℕ
  may : ℕ

/-- The average salary calculation is correct -/
def average_salary_correct (s : MonthlySalary) : Prop :=
  (s.january + s.february + s.march + s.april) / 4 = 8000 ∧
  (s.february + s.march + s.april + s.may) / 4 = 9500

/-- The salary for May is 6500 -/
def may_salary_correct (s : MonthlySalary) : Prop :=
  s.may = 6500

/-- The theorem stating that given the conditions, the salary for January is 500 -/
theorem january_salary (s : MonthlySalary) 
  (h1 : average_salary_correct s) 
  (h2 : may_salary_correct s) : 
  s.january = 500 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l779_77913


namespace NUMINAMATH_CALUDE_unique_solution_l779_77918

theorem unique_solution : ∀ a b c : ℕ, 2^a + 9^b = 2 * 5^c + 5 ↔ a = 1 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l779_77918


namespace NUMINAMATH_CALUDE_odd_log_properties_l779_77905

noncomputable section

-- Define the logarithm function with base a
def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := log a x

-- Theorem statement
theorem odd_log_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- Part 1: The value of m
  (∀ x > 0, log a x + log a (-x) = 0) →
  -- Part 2: The derivative of f
  (∀ x ≠ 0, deriv (f a) x = (Real.log a)⁻¹ / x) ∧
  -- Part 3: The value of a given the range condition
  (∀ x ∈ Set.Ioo 1 (a - 2), f a x ∈ Set.Ioi 1) →
  a = 2 + Real.sqrt 5 := by
sorry

end

end NUMINAMATH_CALUDE_odd_log_properties_l779_77905


namespace NUMINAMATH_CALUDE_range_of_a_l779_77956

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a + Real.cos (2 * x) < 5 - 4 * Real.sin x + Real.sqrt (5 * a - 4)) →
  (4/5 ≤ a ∧ a < 8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l779_77956


namespace NUMINAMATH_CALUDE_sum_distinct_prime_factors_of_420_l779_77999

theorem sum_distinct_prime_factors_of_420 : 
  (Finset.sum (Nat.factors 420).toFinset id) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_distinct_prime_factors_of_420_l779_77999


namespace NUMINAMATH_CALUDE_min_value_a_l779_77915

theorem min_value_a (a b c d : ℕ+) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 2004)
  (h3 : a^2 - b^2 + c^2 - d^2 = 2004) :
  ∀ x : ℕ+, (x > b ∧ b > c ∧ c > d ∧ 
             x + b + c + d = 2004 ∧ 
             x^2 - b^2 + c^2 - d^2 = 2004) → 
    x ≥ 503 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l779_77915


namespace NUMINAMATH_CALUDE_magnitude_AB_is_5_l779_77951

def A : ℝ × ℝ := (-1, -6)
def B : ℝ × ℝ := (2, -2)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem magnitude_AB_is_5 : 
  Real.sqrt ((vector_AB.1)^2 + (vector_AB.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_AB_is_5_l779_77951


namespace NUMINAMATH_CALUDE_expected_sixes_is_half_l779_77909

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a standard die -/
def prob_not_six : ℚ := 1 - prob_six

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 6's when rolling three standard dice -/
def expected_sixes : ℚ := 
  (0 : ℚ) * (prob_not_six ^ num_dice) +
  (1 : ℚ) * (num_dice.choose 1 * prob_six * prob_not_six^2) +
  (2 : ℚ) * (num_dice.choose 2 * prob_six^2 * prob_not_six) +
  (3 : ℚ) * (prob_six ^ num_dice)

theorem expected_sixes_is_half : expected_sixes = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_expected_sixes_is_half_l779_77909


namespace NUMINAMATH_CALUDE_calculation_result_l779_77927

theorem calculation_result : (377 / 13) / 29 * (1 / 4) / 2 = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l779_77927


namespace NUMINAMATH_CALUDE_port_distance_l779_77985

/-- Represents a ship traveling between two ports -/
structure Ship where
  speed : ℝ
  trips : ℕ

/-- Represents the problem setup -/
structure PortProblem where
  blue : Ship
  green : Ship
  first_meeting_distance : ℝ
  total_distance : ℝ

/-- The theorem stating the distance between ports -/
theorem port_distance (p : PortProblem) 
  (h1 : p.blue.trips = 4)
  (h2 : p.green.trips = 3)
  (h3 : p.first_meeting_distance = 20)
  (h4 : p.blue.speed / p.green.speed = p.first_meeting_distance / (p.total_distance - p.first_meeting_distance))
  (h5 : p.blue.speed * p.blue.trips = p.green.speed * p.green.trips) :
  p.total_distance = 35 := by
  sorry

end NUMINAMATH_CALUDE_port_distance_l779_77985


namespace NUMINAMATH_CALUDE_prob_C_correct_prob_C_given_A_correct_l779_77974

/-- Represents a box containing red and white balls -/
structure Box where
  red : ℕ
  white : ℕ

/-- The probability of drawing a red ball from a box -/
def prob_red (b : Box) : ℚ :=
  b.red / (b.red + b.white)

/-- The probability of drawing a white ball from a box -/
def prob_white (b : Box) : ℚ :=
  b.white / (b.red + b.white)

/-- Initial state of box A -/
def box_A : Box := ⟨3, 2⟩

/-- Initial state of box B -/
def box_B : Box := ⟨2, 3⟩

/-- State of box B after transferring a ball from box A -/
def box_B_after (red_transferred : Bool) : Box :=
  if red_transferred then ⟨box_B.red + 1, box_B.white⟩
  else ⟨box_B.red, box_B.white + 1⟩

/-- Probability of event C given the initial conditions -/
def prob_C : ℚ :=
  (prob_red box_A * prob_red (box_B_after true)) +
  (prob_white box_A * prob_red (box_B_after false))

/-- Conditional probability of event C given event A -/
def prob_C_given_A : ℚ :=
  prob_red (box_B_after true)

theorem prob_C_correct : prob_C = 13 / 30 := by sorry

theorem prob_C_given_A_correct : prob_C_given_A = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_C_correct_prob_C_given_A_correct_l779_77974


namespace NUMINAMATH_CALUDE_sum_of_consecutive_terms_l779_77979

theorem sum_of_consecutive_terms (n : ℝ) : n + (n + 1) + (n + 2) + (n + 3) = 20 → n = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_terms_l779_77979


namespace NUMINAMATH_CALUDE_john_reading_speed_l779_77986

/-- Calculates the number of pages read per hour given the total pages, reading duration in weeks, and daily reading hours. -/
def pages_per_hour (total_pages : ℕ) (weeks : ℕ) (hours_per_day : ℕ) : ℚ :=
  (total_pages : ℚ) / ((weeks * 7 : ℕ) * hours_per_day)

/-- Theorem stating that under the given conditions, John reads 50 pages per hour. -/
theorem john_reading_speed :
  let total_pages : ℕ := 2800
  let weeks : ℕ := 4
  let hours_per_day : ℕ := 2
  pages_per_hour total_pages weeks hours_per_day = 50 := by
  sorry

end NUMINAMATH_CALUDE_john_reading_speed_l779_77986


namespace NUMINAMATH_CALUDE_derivative_at_one_equals_three_l779_77943

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x - 1)^2 + 3*(x - 1)

-- State the theorem
theorem derivative_at_one_equals_three :
  deriv f 1 = 3 := by
  sorry


end NUMINAMATH_CALUDE_derivative_at_one_equals_three_l779_77943


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l779_77998

def M : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}
def N : Set ℝ := {x | x ≤ -3 ∨ x ≥ 6}

theorem intersection_of_M_and_N :
  M ∩ N = {x | -5 ≤ x ∧ x ≤ -3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l779_77998


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l779_77994

/-- Represents the speed of a swimmer in various conditions -/
structure SwimmerSpeed where
  stillWater : ℝ
  stream : ℝ

/-- Calculates the effective speed of the swimmer -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.stillWater + s.stream else s.stillWater - s.stream

/-- Theorem: Given the conditions, the swimmer's speed in still water is 5 km/h -/
theorem swimmer_speed_in_still_water
  (s : SwimmerSpeed)
  (h1 : effectiveSpeed s true * 3 = 18)  -- Downstream condition
  (h2 : effectiveSpeed s false * 3 = 12) -- Upstream condition
  : s.stillWater = 5 := by
  sorry


end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l779_77994


namespace NUMINAMATH_CALUDE_equal_probability_for_claudia_and_adela_l779_77926

/-- The probability that a single die roll is not a multiple of 3 -/
def p_not_multiple_of_3 : ℚ := 2/3

/-- The probability that a single die roll is a multiple of 3 -/
def p_multiple_of_3 : ℚ := 1/3

/-- The number of dice rolled -/
def n : ℕ := 2

theorem equal_probability_for_claudia_and_adela :
  p_not_multiple_of_3 ^ n = n * p_multiple_of_3 * p_not_multiple_of_3 ^ (n - 1) :=
sorry

end NUMINAMATH_CALUDE_equal_probability_for_claudia_and_adela_l779_77926


namespace NUMINAMATH_CALUDE_greatest_possible_large_chips_l779_77960

/-- Represents the number of chips in the box -/
def total_chips : ℕ := 60

/-- Represents the number of large chips -/
def large_chips : ℕ := 29

/-- Represents the number of small chips -/
def small_chips : ℕ := total_chips - large_chips

/-- Represents the difference between small and large chips -/
def difference : ℕ := small_chips - large_chips

theorem greatest_possible_large_chips :
  (total_chips = small_chips + large_chips) ∧
  (∃ p : ℕ, Nat.Prime p ∧ small_chips = large_chips + p ∧ p ∣ large_chips) ∧
  (∀ l : ℕ, l > large_chips →
    ¬(∃ p : ℕ, Nat.Prime p ∧ (total_chips - l) = l + p ∧ p ∣ l)) :=
by sorry

#eval large_chips -- Should output 29
#eval small_chips -- Should output 31
#eval difference -- Should output 2

end NUMINAMATH_CALUDE_greatest_possible_large_chips_l779_77960


namespace NUMINAMATH_CALUDE_arithmetic_sequence_68th_term_l779_77919

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℕ
  term_21 : ℕ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℕ :=
  seq.first_term + (n - 1) * ((seq.term_21 - seq.first_term) / 20)

/-- Theorem stating that the 68th term of the given arithmetic sequence is 204 -/
theorem arithmetic_sequence_68th_term
  (seq : ArithmeticSequence)
  (h1 : seq.first_term = 3)
  (h2 : seq.term_21 = 63) :
  nth_term seq 68 = 204 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_68th_term_l779_77919


namespace NUMINAMATH_CALUDE_senate_subcommittee_seating_l779_77987

/-- The number of ways to arrange senators around a circular table -/
def arrange_senators (num_democrats : ℕ) (num_republicans : ℕ) : ℕ :=
  -- Arrangements of 2 blocks (Democrats and Republicans) in a circle
  1 *
  -- Permutations of Democrats within their block
  (Nat.factorial num_democrats) *
  -- Permutations of Republicans within their block
  (Nat.factorial num_republicans)

/-- Theorem stating the number of arrangements for 6 Democrats and 6 Republicans -/
theorem senate_subcommittee_seating :
  arrange_senators 6 6 = 518400 :=
by sorry

end NUMINAMATH_CALUDE_senate_subcommittee_seating_l779_77987


namespace NUMINAMATH_CALUDE_not_always_determinable_l779_77923

/-- Represents a weight with a mass -/
structure Weight where
  mass : ℝ

/-- Represents a question about the order of three weights -/
structure Question where
  a : Weight
  b : Weight
  c : Weight

/-- The set of all possible permutations of five weights -/
def AllPermutations : Finset (List Weight) :=
  sorry

/-- The number of questions we can ask -/
def NumQuestions : ℕ := 9

/-- A function that simulates asking a question -/
def askQuestion (q : Question) (perm : List Weight) : Bool :=
  sorry

/-- The main theorem stating that it's not always possible to determine the exact order -/
theorem not_always_determinable (weights : Finset Weight) 
  (h : weights.card = 5) :
  ∃ (perm₁ perm₂ : List Weight),
    perm₁ ∈ AllPermutations ∧ 
    perm₂ ∈ AllPermutations ∧ 
    perm₁ ≠ perm₂ ∧
    ∀ (questions : Finset Question),
      questions.card ≤ NumQuestions →
      ∀ (q : Question),
        q ∈ questions →
        askQuestion q perm₁ = askQuestion q perm₂ :=
  sorry

end NUMINAMATH_CALUDE_not_always_determinable_l779_77923


namespace NUMINAMATH_CALUDE_stadium_attendance_l779_77952

/-- Given a stadium with initial attendees and girls, calculate remaining attendees after some leave --/
def remaining_attendees (total : ℕ) (girls : ℕ) : ℕ :=
  let boys := total - girls
  let boys_left := boys / 4
  let girls_left := girls / 8
  total - (boys_left + girls_left)

/-- Theorem stating that 480 people remain given the initial conditions --/
theorem stadium_attendance : remaining_attendees 600 240 = 480 := by
  sorry

end NUMINAMATH_CALUDE_stadium_attendance_l779_77952


namespace NUMINAMATH_CALUDE_bacteria_exceeds_200_on_day_4_l779_77921

-- Define the bacteria population function
def bacteria_population (initial_population : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_population * growth_factor ^ days

-- Theorem statement
theorem bacteria_exceeds_200_on_day_4 :
  let initial_population := 5
  let growth_factor := 3
  let threshold := 200
  (∀ d : ℕ, d < 4 → bacteria_population initial_population growth_factor d ≤ threshold) ∧
  (bacteria_population initial_population growth_factor 4 > threshold) :=
by sorry

end NUMINAMATH_CALUDE_bacteria_exceeds_200_on_day_4_l779_77921


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l779_77958

/-- An isosceles triangle with two sides of 6cm and 13cm has a perimeter of 32cm. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 6 ∧ b = 13 ∧ c = 13 →  -- Two sides are 13cm (base) and one side is 6cm
  a + b > c ∧ a + c > b ∧ b + c > a →  -- Triangle inequality
  a + b + c = 32 :=  -- Perimeter is 32cm
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l779_77958


namespace NUMINAMATH_CALUDE_total_blankets_is_243_l779_77980

/-- Represents the number of blankets collected over three days --/
def total_blankets : ℕ := 
  let day1_team := 15 * 2
  let day1_online := 5 * 4
  let day1_total := day1_team + day1_online

  let day2_new_members := 5 * 4
  let day2_original_members := 15 * 2 * 3
  let day2_online := 3 * 5
  let day2_total := day2_new_members + day2_original_members + day2_online

  let day3_schools := 22
  let day3_online := 7 * 3
  let day3_business := day2_total / 5
  let day3_total := day3_schools + day3_online + day3_business

  day1_total + day2_total + day3_total

/-- Theorem stating that the total number of blankets collected is 243 --/
theorem total_blankets_is_243 : total_blankets = 243 := by
  sorry

end NUMINAMATH_CALUDE_total_blankets_is_243_l779_77980


namespace NUMINAMATH_CALUDE_suit_price_calculation_l779_77948

theorem suit_price_calculation (original_price : ℝ) (increase_rate : ℝ) (discount_rate : ℝ) : 
  original_price = 150 →
  increase_rate = 0.2 →
  discount_rate = 0.2 →
  let increased_price := original_price * (1 + increase_rate)
  let final_price := increased_price * (1 - discount_rate)
  final_price = 144 := by
sorry

end NUMINAMATH_CALUDE_suit_price_calculation_l779_77948


namespace NUMINAMATH_CALUDE_room_dimension_proof_l779_77908

/-- Proves that given the room dimensions and whitewashing costs, the unknown dimension is 15 feet -/
theorem room_dimension_proof (x : ℝ) : 
  let room_length : ℝ := 25
  let room_height : ℝ := 12
  let door_area : ℝ := 6 * 3
  let window_area : ℝ := 4 * 3
  let num_windows : ℕ := 3
  let whitewash_cost_per_sqft : ℝ := 3
  let total_cost : ℝ := 2718
  let wall_area : ℝ := 2 * (room_length * room_height) + 2 * (x * room_height)
  let non_whitewash_area : ℝ := door_area + num_windows * window_area
  let whitewash_area : ℝ := wall_area - non_whitewash_area
  whitewash_area * whitewash_cost_per_sqft = total_cost → x = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_room_dimension_proof_l779_77908


namespace NUMINAMATH_CALUDE_expression_evaluation_l779_77946

theorem expression_evaluation : 5 + 7 * (2 - 9)^2 = 348 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l779_77946


namespace NUMINAMATH_CALUDE_linear_function_decreases_l779_77977

/-- A linear function with a negative slope decreases as x increases -/
theorem linear_function_decreases (m b : ℝ) (h : m < 0) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m * x₁ + b) > (m * x₂ + b) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreases_l779_77977


namespace NUMINAMATH_CALUDE_triangle_inradius_l779_77941

/-- Given a triangle with perimeter 35 cm and area 78.75 cm², prove its inradius is 4.5 cm -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : p = 35)
  (h_area : A = 78.75)
  (h_inradius : A = r * p / 2) : 
  r = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l779_77941


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l779_77938

theorem quadratic_is_square_of_binomial :
  ∃ (r s : ℚ), (r * X + s)^2 = (81/16 : ℚ) * X^2 + 18 * X + 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l779_77938


namespace NUMINAMATH_CALUDE_min_value_of_f_l779_77982

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x + m

-- State the theorem
theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) ∧ 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -1) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≥ -1) :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_f_l779_77982


namespace NUMINAMATH_CALUDE_part_one_part_two_l779_77997

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

-- Part 1
theorem part_one (m n : ℝ) :
  (∀ x, f m x < 0 ↔ -2 < x ∧ x < n) →
  m = 5/2 ∧ n = 1/2 := by sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∀ x ∈ Set.Icc m (m+1), f m x < 0) →
  m > -Real.sqrt 2 / 2 ∧ m < 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l779_77997


namespace NUMINAMATH_CALUDE_lamp_price_after_discounts_l779_77957

/-- Calculates the final price of a lamp after applying two discounts -/
theorem lamp_price_after_discounts (original_price : ℝ) 
  (first_discount_rate : ℝ) (second_discount_rate : ℝ) : 
  original_price = 120 → 
  first_discount_rate = 0.20 → 
  second_discount_rate = 0.15 → 
  original_price * (1 - first_discount_rate) * (1 - second_discount_rate) = 81.60 := by
sorry

end NUMINAMATH_CALUDE_lamp_price_after_discounts_l779_77957


namespace NUMINAMATH_CALUDE_tina_pens_count_l779_77964

/-- Calculates the total number of pens Tina has given the number of pink pens and the relationships between different colored pens. -/
def total_pens (pink : ℕ) (green_diff : ℕ) (blue_diff : ℕ) : ℕ :=
  pink + (pink - green_diff) + ((pink - green_diff) + blue_diff)

/-- Proves that given the conditions, Tina has 21 pens in total. -/
theorem tina_pens_count : total_pens 12 9 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_tina_pens_count_l779_77964


namespace NUMINAMATH_CALUDE_max_notebooks_is_eleven_l779_77924

/-- Represents the maximum number of notebooks that can be purchased with a given budget. -/
def max_notebooks (single_price : ℕ) (pack4_price : ℕ) (pack7_price : ℕ) (budget : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific pricing and budget, the maximum number of notebooks is 11. -/
theorem max_notebooks_is_eleven :
  max_notebooks 2 6 9 15 = 11 := by
  sorry

end NUMINAMATH_CALUDE_max_notebooks_is_eleven_l779_77924


namespace NUMINAMATH_CALUDE_f_neg_three_equals_six_l779_77968

-- Define the function f with the given property
def f : ℝ → ℝ := sorry

-- State the main theorem
theorem f_neg_three_equals_six :
  (∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y) →
  f 1 = 2 →
  f (-3) = 6 := by sorry

end NUMINAMATH_CALUDE_f_neg_three_equals_six_l779_77968


namespace NUMINAMATH_CALUDE_least_number_divisibility_l779_77988

theorem least_number_divisibility (n : ℕ) (h : n = 59789) : 
  let m := 16142
  (∀ k : ℕ, k < m → ¬((n + k) % 7 = 0 ∧ (n + k) % 11 = 0 ∧ (n + k) % 13 = 0 ∧ (n + k) % 17 = 0)) ∧ 
  ((n + m) % 7 = 0 ∧ (n + m) % 11 = 0 ∧ (n + m) % 13 = 0 ∧ (n + m) % 17 = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l779_77988


namespace NUMINAMATH_CALUDE_embankment_build_time_l779_77911

/-- Represents the time taken to build an embankment given a number of workers -/
def build_time (workers : ℕ) (days : ℚ) : Prop :=
  workers * days = 300

theorem embankment_build_time :
  build_time 75 4 → build_time 50 6 := by
  sorry

end NUMINAMATH_CALUDE_embankment_build_time_l779_77911


namespace NUMINAMATH_CALUDE_discount_calculation_l779_77917

/-- Calculates the discount amount given the cost of a suit, shoes, and the final payment -/
theorem discount_calculation (suit_cost shoes_cost final_payment : ℕ) :
  suit_cost = 430 →
  shoes_cost = 190 →
  final_payment = 520 →
  suit_cost + shoes_cost - final_payment = 100 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l779_77917


namespace NUMINAMATH_CALUDE_circle_radius_from_polar_equation_l779_77990

/-- The radius of a circle given by the polar equation ρ = 2cosθ is 1 -/
theorem circle_radius_from_polar_equation : 
  ∃ (center : ℝ × ℝ) (r : ℝ), 
    (∀ θ : ℝ, (2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ) ∈ 
      {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2}) ∧ 
    r = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_from_polar_equation_l779_77990


namespace NUMINAMATH_CALUDE_exists_special_polynomial_l779_77930

/-- A fifth-degree polynomial with specific properties on [-1,1] -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ (x₁ x₂ : ℝ), -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 ∧
    p x₁ = 1 ∧ p (-x₂) = 1 ∧ p (-x₁) = -1 ∧ p x₂ = -1) ∧
  p (-1) = 0 ∧ p 1 = 0 ∧
  ∀ x, x ∈ Set.Icc (-1) 1 → -1 ≤ p x ∧ p x ≤ 1

/-- There exists a fifth-degree polynomial with the special properties -/
theorem exists_special_polynomial :
  ∃ (p : ℝ → ℝ), ∃ (a b c d e f : ℝ),
    (∀ x, p x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) ∧
    special_polynomial p :=
sorry

end NUMINAMATH_CALUDE_exists_special_polynomial_l779_77930


namespace NUMINAMATH_CALUDE_coin_flip_problem_l779_77972

theorem coin_flip_problem (total_flips : ℕ) (tail_head_difference : ℕ) 
  (h1 : total_flips = 211)
  (h2 : tail_head_difference = 81) : 
  ∃ (heads : ℕ), 
    heads + (heads + tail_head_difference) = total_flips ∧ 
    heads = 65 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_problem_l779_77972


namespace NUMINAMATH_CALUDE_log_216_equals_3_log_2_plus_3_log_3_l779_77963

theorem log_216_equals_3_log_2_plus_3_log_3 :
  Real.log 216 = 3 * (Real.log 2 + Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_log_216_equals_3_log_2_plus_3_log_3_l779_77963


namespace NUMINAMATH_CALUDE_students_on_bus_l779_77965

theorem students_on_bus (first_stop : ℕ) (second_stop : ℕ) 
  (h1 : first_stop = 39) (h2 : second_stop = 29) :
  first_stop + second_stop = 68 := by
  sorry

end NUMINAMATH_CALUDE_students_on_bus_l779_77965


namespace NUMINAMATH_CALUDE_pencil_length_l779_77949

/-- The length of one pencil when two equal-length pencils together measure 24 cubes -/
theorem pencil_length (total_length : ℕ) (pencil_length : ℕ) : 
  total_length = 24 → 2 * pencil_length = total_length → pencil_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l779_77949


namespace NUMINAMATH_CALUDE_james_beat_record_by_116_l779_77931

/-- Represents James's scoring statistics for the football season -/
structure JamesStats where
  touchdownsPerGame : ℕ
  gamesInSeason : ℕ
  twoPointConversions : ℕ
  fieldGoals : ℕ
  extraPointAttempts : ℕ

/-- Calculates the total points scored by James -/
def totalPoints (stats : JamesStats) : ℕ :=
  stats.touchdownsPerGame * 6 * stats.gamesInSeason +
  stats.twoPointConversions * 2 +
  stats.fieldGoals * 3 +
  stats.extraPointAttempts

/-- The old record for points scored in a season -/
def oldRecord : ℕ := 300

/-- Theorem stating that James beat the old record by 116 points -/
theorem james_beat_record_by_116 (stats : JamesStats)
  (h1 : stats.touchdownsPerGame = 4)
  (h2 : stats.gamesInSeason = 15)
  (h3 : stats.twoPointConversions = 6)
  (h4 : stats.fieldGoals = 8)
  (h5 : stats.extraPointAttempts = 20) :
  totalPoints stats - oldRecord = 116 := by
  sorry

#eval totalPoints { touchdownsPerGame := 4, gamesInSeason := 15, twoPointConversions := 6, fieldGoals := 8, extraPointAttempts := 20 } - oldRecord

end NUMINAMATH_CALUDE_james_beat_record_by_116_l779_77931


namespace NUMINAMATH_CALUDE_farm_feet_count_l779_77984

/-- Represents a farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of animals (heads) in the farm -/
def Farm.totalAnimals (f : Farm) : ℕ := f.hens + f.cows

/-- The total number of feet in the farm -/
def Farm.totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- Theorem: In a farm with 44 animals, if there are 24 hens, then the total number of feet is 128 -/
theorem farm_feet_count (f : Farm) (h1 : f.totalAnimals = 44) (h2 : f.hens = 24) : 
  f.totalFeet = 128 := by
  sorry

end NUMINAMATH_CALUDE_farm_feet_count_l779_77984


namespace NUMINAMATH_CALUDE_julieta_total_spend_l779_77954

/-- Calculates the total amount spent by Julieta at the store -/
def total_amount_spent (
  backpack_original_price : ℕ)
  (ringbinder_original_price : ℕ)
  (backpack_price_increase : ℕ)
  (ringbinder_price_reduction : ℕ)
  (num_ringbinders : ℕ) : ℕ :=
  (backpack_original_price + backpack_price_increase) +
  num_ringbinders * (ringbinder_original_price - ringbinder_price_reduction)

/-- Theorem stating that Julieta's total spend is $109 -/
theorem julieta_total_spend :
  total_amount_spent 50 20 5 2 3 = 109 := by
  sorry

end NUMINAMATH_CALUDE_julieta_total_spend_l779_77954


namespace NUMINAMATH_CALUDE_gcd_840_1764_l779_77971

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l779_77971


namespace NUMINAMATH_CALUDE_cos_330_degrees_l779_77991

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l779_77991


namespace NUMINAMATH_CALUDE_soccer_team_losses_l779_77947

theorem soccer_team_losses (total_games : ℕ) (games_won : ℕ) (points_for_win : ℕ) 
  (points_for_draw : ℕ) (points_for_loss : ℕ) (total_points : ℕ) :
  total_games = 20 →
  games_won = 14 →
  points_for_win = 3 →
  points_for_draw = 1 →
  points_for_loss = 0 →
  total_points = 46 →
  ∃ (games_lost : ℕ) (games_drawn : ℕ),
    games_lost = 2 ∧
    games_won + games_drawn + games_lost = total_games ∧
    games_won * points_for_win + games_drawn * points_for_draw + games_lost * points_for_loss = total_points :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_team_losses_l779_77947


namespace NUMINAMATH_CALUDE_quadratic_inequality_l779_77937

theorem quadratic_inequality (x : ℝ) : x^2 - 9*x + 14 ≤ 0 ↔ x ∈ Set.Icc 2 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l779_77937
