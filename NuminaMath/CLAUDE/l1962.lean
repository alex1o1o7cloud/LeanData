import Mathlib

namespace NUMINAMATH_CALUDE_power_product_equality_l1962_196291

theorem power_product_equality (a b : ℕ) (h1 : a = 7^5) (h2 : b = 5^7) : a^7 * b^5 = 35^35 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l1962_196291


namespace NUMINAMATH_CALUDE_count_divisible_by_33_l1962_196286

/-- Represents a 10-digit number of the form a2016b2017 -/
def NumberForm (a b : Nat) : Nat :=
  a * 10^9 + 2 * 10^8 + 0 * 10^7 + 1 * 10^6 + 6 * 10^5 + b * 10^4 + 2 * 10^3 + 0 * 10^2 + 1 * 10 + 7

/-- Predicate to check if a number is a single digit -/
def IsSingleDigit (n : Nat) : Prop := n < 10

/-- Main theorem -/
theorem count_divisible_by_33 :
  ∃! (count : Nat), ∃ (S : Finset (Nat × Nat)),
    (∀ (pair : Nat × Nat), pair ∈ S ↔ 
      IsSingleDigit pair.1 ∧ 
      IsSingleDigit pair.2 ∧ 
      (NumberForm pair.1 pair.2) % 33 = 0) ∧
    S.card = count ∧
    count = 3 := by sorry

end NUMINAMATH_CALUDE_count_divisible_by_33_l1962_196286


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l1962_196270

theorem multiplication_subtraction_equality : (5 * 3) - 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l1962_196270


namespace NUMINAMATH_CALUDE_exponent_expression_equality_l1962_196236

theorem exponent_expression_equality : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_expression_equality_l1962_196236


namespace NUMINAMATH_CALUDE_divisible_by_nine_sequence_l1962_196220

theorem divisible_by_nine_sequence (N : ℕ) : 
  (∃ (seq : List ℕ), 
    seq.length = 1110 ∧ 
    (∀ n ∈ seq, n % 9 = 0) ∧
    (∀ n ∈ seq, N ≤ n ∧ n ≤ 10000) ∧
    (∀ m, N ≤ m ∧ m ≤ 10000 ∧ m % 9 = 0 → m ∈ seq)) →
  N = 27 := by
sorry

end NUMINAMATH_CALUDE_divisible_by_nine_sequence_l1962_196220


namespace NUMINAMATH_CALUDE_no_integer_geometric_progression_angles_l1962_196249

/-- Represents the angles of a triangle in geometric progression -/
structure TriangleAngles where
  a : ℕ+  -- first angle
  r : ℕ+  -- common ratio
  h1 : a < a * r  -- angles are distinct
  h2 : a * r < a * r * r  -- angles are distinct
  h3 : a + a * r + a * r * r = 180  -- sum of angles is 180 degrees

/-- There are no triangles with angles that are distinct positive integers in a geometric progression -/
theorem no_integer_geometric_progression_angles : ¬∃ (t : TriangleAngles), True :=
sorry

end NUMINAMATH_CALUDE_no_integer_geometric_progression_angles_l1962_196249


namespace NUMINAMATH_CALUDE_zeros_of_f_l1962_196295

open Real MeasureTheory Set

noncomputable def f (x : ℝ) := cos x - sin (2 * x)

def I : Set ℝ := Icc 0 (2 * π)

theorem zeros_of_f : 
  (∃ (S : Finset ℝ), S.card = 4 ∧ (∀ x ∈ S, x ∈ I ∧ f x = 0) ∧
  (∀ y ∈ I, f y = 0 → y ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_l1962_196295


namespace NUMINAMATH_CALUDE_percent_decrease_cars_sold_car_sales_decrease_proof_l1962_196265

/-- Calculates the percent decrease in cars sold given the increase in total profit and average profit per car -/
theorem percent_decrease_cars_sold 
  (total_profit_increase : ℝ) 
  (avg_profit_per_car_increase : ℝ) : ℝ :=
  let new_total_profit_ratio := 1 + total_profit_increase
  let new_avg_profit_ratio := 1 + avg_profit_per_car_increase
  let cars_sold_ratio := new_total_profit_ratio / new_avg_profit_ratio
  (1 - cars_sold_ratio) * 100

/-- The percent decrease in cars sold is approximately 30% when total profit increases by 30% and average profit per car increases by 85.71% -/
theorem car_sales_decrease_proof : 
  abs (percent_decrease_cars_sold 0.30 0.8571 - 30) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_cars_sold_car_sales_decrease_proof_l1962_196265


namespace NUMINAMATH_CALUDE_sum_six_to_thousand_l1962_196245

/-- Count of digit 6 occurrences in a number -/
def count_six (n : ℕ) : ℕ := sorry

/-- Sum of digit 6 occurrences from 1 to n -/
def sum_six_occurrences (n : ℕ) : ℕ := sorry

/-- The theorem stating that the sum of digit 6 occurrences from 1 to 1000 is 300 -/
theorem sum_six_to_thousand :
  sum_six_occurrences 1000 = 300 := by sorry

end NUMINAMATH_CALUDE_sum_six_to_thousand_l1962_196245


namespace NUMINAMATH_CALUDE_train_passing_time_specific_train_problem_l1962_196271

/-- The time taken for a faster train to completely pass a slower train -/
theorem train_passing_time (train_length : ℝ) (fast_speed slow_speed : ℝ) : ℝ :=
  let relative_speed := fast_speed - slow_speed
  let relative_speed_mps := relative_speed * (5 / 18)
  let total_distance := 2 * train_length
  total_distance / relative_speed_mps

/-- Proof of the specific train problem -/
theorem specific_train_problem :
  ∃ (t : ℝ), abs (t - train_passing_time 75 46 36) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_train_passing_time_specific_train_problem_l1962_196271


namespace NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l1962_196282

-- Define the custom operation
def otimes (a b : ℝ) : ℝ := a^2 - abs b

-- Theorem statement
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l1962_196282


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l1962_196280

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := (1 - i) * z = 2 + 3 * i

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ second_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l1962_196280


namespace NUMINAMATH_CALUDE_no_intersection_in_S_l1962_196275

-- Define the set S of polynomials
inductive S : (ℝ → ℝ) → Prop
  | base : S (λ x => x)
  | sub {f} : S f → S (λ x => x - f x)
  | add {f} : S f → S (λ x => x + (1 - x) * f x)

-- Define the theorem
theorem no_intersection_in_S :
  ∀ (f g : ℝ → ℝ), S f → S g → f ≠ g →
  ∀ x, 0 < x → x < 1 → f x ≠ g x :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_in_S_l1962_196275


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1962_196233

theorem pure_imaginary_condition (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 - 1) (a - 2)
  (z.re = 0 ∧ z.im ≠ 0) → (a = -1 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1962_196233


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l1962_196231

theorem greatest_three_digit_number : ∃ (n : ℕ), n = 953 ∧
  n ≤ 999 ∧
  ∃ (k : ℕ), n = 9 * k + 2 ∧
  ∃ (m : ℕ), n = 5 * m + 3 ∧
  ∃ (l : ℕ), n = 7 * l + 4 ∧
  ∀ (x : ℕ), x ≤ 999 → 
    (∃ (a b c : ℕ), x = 9 * a + 2 ∧ x = 5 * b + 3 ∧ x = 7 * c + 4) → 
    x ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l1962_196231


namespace NUMINAMATH_CALUDE_geometric_sum_15_l1962_196226

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sum_15 (a : ℕ → ℤ) :
  geometric_sequence a →
  a 1 = 1 →
  (∀ n : ℕ, a (n + 1) = a n * (-2)) →
  a 1 + |a 2| + a 3 + |a 4| = 15 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_15_l1962_196226


namespace NUMINAMATH_CALUDE_evaluate_power_of_power_l1962_196285

theorem evaluate_power_of_power : (3^3)^2 = 729 := by sorry

end NUMINAMATH_CALUDE_evaluate_power_of_power_l1962_196285


namespace NUMINAMATH_CALUDE_part_I_part_II_l1962_196230

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Define set N with parameter a
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}

-- Part I
theorem part_I : 
  M ∩ (U \ N 2) = {x : ℝ | -2 ≤ x ∧ x < 3} := by sorry

-- Part II
theorem part_II :
  ∀ a : ℝ, M ∪ N a = M → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l1962_196230


namespace NUMINAMATH_CALUDE_function_property_center_of_symmetry_range_property_l1962_196229

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1 - a) / (a - x)

theorem function_property (a : ℝ) (x : ℝ) (h : x ≠ a) :
  f a x + f a (2 * a - x) + 2 = 0 := by sorry

theorem center_of_symmetry (a b : ℝ) 
  (h : ∀ x ≠ a, f a x + f a (6 - x) = 2 * b) :
  a + b = -4/7 := by sorry

theorem range_property (a : ℝ) :
  (∀ x ∈ Set.Icc (a + 1/2) (a + 1), f a x ∈ Set.Icc (-3) (-2)) ∧
  (∀ y ∈ Set.Icc (-3) (-2), ∃ x ∈ Set.Icc (a + 1/2) (a + 1), f a x = y) := by sorry

end NUMINAMATH_CALUDE_function_property_center_of_symmetry_range_property_l1962_196229


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l1962_196296

theorem two_digit_reverse_sum (x y n : ℕ) : 
  (x ≠ 0 ∧ y ≠ 0) →
  (10 ≤ x ∧ x < 100) →
  (10 ≤ y ∧ y < 100) →
  (∃ (a b : ℕ), x = 10 * a + b ∧ y = 10 * b + a) →
  x^2 - y^2 = 44 * n →
  x + y + n = 93 := by
sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l1962_196296


namespace NUMINAMATH_CALUDE_box_decoration_combinations_l1962_196204

/-- The number of paint color options available -/
def num_colors : ℕ := 4

/-- The number of decoration options available -/
def num_decorations : ℕ := 3

/-- The total number of combinations for painting and decorating a box -/
def total_combinations : ℕ := num_colors * num_decorations

theorem box_decoration_combinations :
  total_combinations = 12 :=
by sorry

end NUMINAMATH_CALUDE_box_decoration_combinations_l1962_196204


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1962_196260

theorem quadratic_factorization (a b c : ℤ) : 
  (∃ b c : ℤ, ∀ x : ℤ, (x - a) * (x - 6) + 1 = (x + b) * (x + c)) ↔ a = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1962_196260


namespace NUMINAMATH_CALUDE_ellipse_from_hyperbola_l1962_196224

/-- Given a hyperbola with equation x²/4 - y²/5 = 1, prove that the equation of an ellipse
    with foci at the vertices of the hyperbola and vertices at the foci of the hyperbola
    is x²/9 + y²/5 = 1 -/
theorem ellipse_from_hyperbola (x y : ℝ) :
  (x^2 / 4 - y^2 / 5 = 1) →
  ∃ (a b c : ℝ),
    (a^2 = 9 ∧ b^2 = 5 ∧ c^2 = 4) ∧
    (x^2 / a^2 + y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_from_hyperbola_l1962_196224


namespace NUMINAMATH_CALUDE_focus_directrix_distance_l1962_196221

-- Define the ellipse C₁
def ellipse_C1 (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the parabola C₂
def parabola_C2 (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Theorem statement
theorem focus_directrix_distance :
  -- Conditions
  (ellipse_C1 (-2) 0) ∧
  (ellipse_C1 (Real.sqrt 2) ((Real.sqrt 2) / 2)) ∧
  (parabola_C2 3 (-2 * Real.sqrt 3)) ∧
  (parabola_C2 4 (-4)) →
  -- Conclusion
  ∃ (focus_x directrix_x : ℝ),
    -- Left focus of C₁
    focus_x = Real.sqrt 3 ∧
    -- Directrix of C₂
    directrix_x = -1 ∧
    -- Distance between left focus and directrix
    focus_x - directrix_x = Real.sqrt 3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_focus_directrix_distance_l1962_196221


namespace NUMINAMATH_CALUDE_fraction_equality_l1962_196206

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z)/(1/w - 1/z) = 1001) : (w + z)/(w - z) = -501 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1962_196206


namespace NUMINAMATH_CALUDE_find_incorrect_value_l1962_196243

/-- Given a set of observations with an initial mean and a corrected mean after fixing one misrecorded value,
    this theorem proves that the original incorrect value can be determined. -/
theorem find_incorrect_value (n : ℕ) (m1 m2 x : ℚ) (hn : n > 0) :
  let y := n * m1 + x - n * m2
  y = 23 :=
by sorry

end NUMINAMATH_CALUDE_find_incorrect_value_l1962_196243


namespace NUMINAMATH_CALUDE_range_of_a_l1962_196237

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x| ≤ 4) → -4 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1962_196237


namespace NUMINAMATH_CALUDE_smallest_next_divisor_l1962_196267

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_divisor (d m : ℕ) : Prop := ∃ k, m = d * k

theorem smallest_next_divisor 
  (m : ℕ) 
  (h1 : is_even m) 
  (h2 : is_four_digit m) 
  (h3 : is_divisor 437 m) :
  ∃ d : ℕ, 
    is_divisor d m ∧ 
    d > 437 ∧ 
    (∀ d' : ℕ, is_divisor d' m → d' > 437 → d ≤ d') ∧ 
    d = 475 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_l1962_196267


namespace NUMINAMATH_CALUDE_vertex_of_f_l1962_196254

/-- The quadratic function f(x) = -3(x+1)^2 - 2 -/
def f (x : ℝ) : ℝ := -3 * (x + 1)^2 - 2

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-1, -2)

/-- Theorem: The vertex of the quadratic function f is (-1, -2) -/
theorem vertex_of_f : 
  (∀ x : ℝ, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_vertex_of_f_l1962_196254


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l1962_196281

theorem imaginary_part_of_complex_number (z : ℂ) :
  (z.re > 0) →
  (z.im = 2 * z.re) →
  (Complex.abs z = Real.sqrt 5) →
  z.im = 2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l1962_196281


namespace NUMINAMATH_CALUDE_pizza_toppings_count_l1962_196222

theorem pizza_toppings_count (n : ℕ) (h : n = 8) : 
  (n) + (n.choose 2) = 36 := by sorry

end NUMINAMATH_CALUDE_pizza_toppings_count_l1962_196222


namespace NUMINAMATH_CALUDE_count_special_integers_l1962_196238

def is_even_digit (d : Nat) : Bool :=
  d % 2 = 0 ∧ d ≤ 9

def has_only_even_digits (n : Nat) : Bool :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

def is_five_digit (n : Nat) : Bool :=
  10000 ≤ n ∧ n ≤ 99999

theorem count_special_integers :
  (Finset.filter (λ n : Nat => is_five_digit n ∧ has_only_even_digits n ∧ n % 5 = 0)
    (Finset.range 100000)).card = 500 := by
  sorry

end NUMINAMATH_CALUDE_count_special_integers_l1962_196238


namespace NUMINAMATH_CALUDE_train_distance_problem_l1962_196283

/-- The distance between two points A and B, given two trains traveling towards each other --/
theorem train_distance_problem (v1 v2 d : ℝ) (hv1 : v1 = 50) (hv2 : v2 = 60) (hd : d = 100) :
  let x := (v1 * d) / (v2 - v1)
  x + (x + d) = 1100 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l1962_196283


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1962_196261

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1962_196261


namespace NUMINAMATH_CALUDE_simplification_fraction_l1962_196227

theorem simplification_fraction (k : ℤ) : 
  let simplified := (6 * k + 18) / 6
  ∃ (a b : ℤ), simplified = a * k + b ∧ a / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplification_fraction_l1962_196227


namespace NUMINAMATH_CALUDE_at_least_one_not_perfect_square_l1962_196266

theorem at_least_one_not_perfect_square (d : ℕ+) :
  ¬(∃ x y z : ℕ, (2 * d - 1 = x^2) ∧ (5 * d - 1 = y^2) ∧ (13 * d - 1 = z^2)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_not_perfect_square_l1962_196266


namespace NUMINAMATH_CALUDE_system_solution_l1962_196253

theorem system_solution : ∃ (x y : ℝ), 
  (x - 2*y = 3) ∧ (3*x - y = 4) ∧ (x = 1) ∧ (y = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1962_196253


namespace NUMINAMATH_CALUDE_complement_of_P_in_U_l1962_196213

-- Define the universal set U
def U : Set ℝ := {x | x ≥ 0}

-- Define the set P
def P : Set ℝ := {1}

-- Theorem statement
theorem complement_of_P_in_U :
  (U \ P) = {x : ℝ | x ≥ 0 ∧ x ≠ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_in_U_l1962_196213


namespace NUMINAMATH_CALUDE_max_value_x_plus_sqrt_one_minus_x_squared_l1962_196242

theorem max_value_x_plus_sqrt_one_minus_x_squared :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x + Real.sqrt (1 - x^2) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_sqrt_one_minus_x_squared_l1962_196242


namespace NUMINAMATH_CALUDE_malcolm_flat_path_time_l1962_196215

/-- Represents the time in minutes for different parts of Malcolm's routes to school -/
structure RouteTime where
  uphill : ℕ
  path : ℕ
  final : ℕ

/-- Calculates the total time for the first route -/
def first_route_time (r : RouteTime) : ℕ :=
  r.uphill + r.path + r.final

/-- Calculates the total time for the second route -/
def second_route_time (flat_path : ℕ) : ℕ :=
  flat_path + 2 * flat_path

/-- Theorem stating the correct time Malcolm spent on the flat path in the second route -/
theorem malcolm_flat_path_time : ∃ (r : RouteTime) (flat_path : ℕ),
  r.uphill = 6 ∧
  r.path = 2 * r.uphill ∧
  r.final = (r.uphill + r.path) / 3 ∧
  second_route_time flat_path = first_route_time r + 18 ∧
  flat_path = 14 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_flat_path_time_l1962_196215


namespace NUMINAMATH_CALUDE_merchant_discount_percentage_l1962_196292

theorem merchant_discount_percentage 
  (markup_percentage : ℝ) 
  (profit_percentage : ℝ) 
  (discount_percentage : ℝ) : 
  markup_percentage = 75 → 
  profit_percentage = 5 → 
  discount_percentage = 40 → 
  let cost_price := 100
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := cost_price * (1 + profit_percentage / 100)
  let discount := marked_price - selling_price
  discount / marked_price * 100 = discount_percentage :=
by sorry

end NUMINAMATH_CALUDE_merchant_discount_percentage_l1962_196292


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l1962_196297

open Real

theorem tangent_equation_solution :
  ∃! y : ℝ, 0 ≤ y ∧ y < 2 * π ∧
  tan (150 * π / 180 - y) = (sin (150 * π / 180) - sin y) / (cos (150 * π / 180) - cos y) →
  y = 0 ∨ y = 2 * π := by
sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l1962_196297


namespace NUMINAMATH_CALUDE_mountain_climbs_l1962_196210

/-- Proves that Boris needs to climb his mountain 4 times to match Hugo's total climb -/
theorem mountain_climbs (hugo_elevation : ℕ) (boris_difference : ℕ) (hugo_climbs : ℕ) : 
  hugo_elevation = 10000 →
  boris_difference = 2500 →
  hugo_climbs = 3 →
  (hugo_elevation * hugo_climbs) / (hugo_elevation - boris_difference) = 4 := by
sorry

end NUMINAMATH_CALUDE_mountain_climbs_l1962_196210


namespace NUMINAMATH_CALUDE_bucket_weight_l1962_196258

theorem bucket_weight (a b : ℝ) : ℝ :=
  let three_fourths_weight := a
  let one_third_weight := b
  let full_weight := (8 / 5) * a - (3 / 5) * b
  full_weight

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l1962_196258


namespace NUMINAMATH_CALUDE_cora_cookie_expenditure_l1962_196203

def cookies_per_day : ℕ := 3
def cookie_cost : ℕ := 18
def days_in_april : ℕ := 30

theorem cora_cookie_expenditure :
  cookies_per_day * cookie_cost * days_in_april = 1620 := by
  sorry

end NUMINAMATH_CALUDE_cora_cookie_expenditure_l1962_196203


namespace NUMINAMATH_CALUDE_rectangle_problem_l1962_196234

-- Define the rectangle EFGH
structure Rectangle (EFGH : Type) where
  is_rectangle : EFGH → Prop

-- Define point M on FG
def M_on_FG (EFGH : Type) (M : EFGH) : Prop := sorry

-- Define angle EMH as 90°
def angle_EMH_90 (EFGH : Type) (E M H : EFGH) : Prop := sorry

-- Define UV perpendicular to FG
def UV_perp_FG (EFGH : Type) (U V F G : EFGH) : Prop := sorry

-- Define FU = UM
def FU_eq_UM (EFGH : Type) (F U M : EFGH) : Prop := sorry

-- Define MH intersects UV at N
def MH_intersect_UV_at_N (EFGH : Type) (M H U V N : EFGH) : Prop := sorry

-- Define S on GH such that SE passes through N
def S_on_GH_SE_through_N (EFGH : Type) (S G H E N : EFGH) : Prop := sorry

-- Define triangle MNE with given measurements
def triangle_MNE (EFGH : Type) (M N E : EFGH) : Prop :=
  let ME := 25
  let EN := 20
  let MN := 20
  sorry

-- Theorem statement
theorem rectangle_problem (EFGH : Type) 
  (E F G H M U V N S : EFGH) 
  (rect : Rectangle EFGH) 
  (h1 : M_on_FG EFGH M)
  (h2 : angle_EMH_90 EFGH E M H)
  (h3 : UV_perp_FG EFGH U V F G)
  (h4 : FU_eq_UM EFGH F U M)
  (h5 : MH_intersect_UV_at_N EFGH M H U V N)
  (h6 : S_on_GH_SE_through_N EFGH S G H E N)
  (h7 : triangle_MNE EFGH M N E) :
  ∃ (FM NV : ℝ), FM = 15 ∧ NV = 5 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_problem_l1962_196234


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l1962_196250

theorem polynomial_value_theorem (P : Int → Int) (a b c d : Int) :
  (∀ x : Int, ∃ y : Int, P x = y) →  -- P has integer coefficients
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →  -- a, b, c, d are distinct
  (P a = 5 ∧ P b = 5 ∧ P c = 5 ∧ P d = 5) →  -- P(a) = P(b) = P(c) = P(d) = 5
  ¬ ∃ k : Int, P k = 8 :=  -- There does not exist an integer k such that P(k) = 8
by sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l1962_196250


namespace NUMINAMATH_CALUDE_max_equal_distribution_l1962_196248

theorem max_equal_distribution (bags eyeliners scarves hairbands : ℕ) 
  (h1 : bags = 2923)
  (h2 : eyeliners = 3239)
  (h3 : scarves = 1785)
  (h4 : hairbands = 1379) :
  Nat.gcd bags (Nat.gcd eyeliners (Nat.gcd scarves hairbands)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_equal_distribution_l1962_196248


namespace NUMINAMATH_CALUDE_button_sequence_l1962_196262

theorem button_sequence (a : Fin 6 → ℕ) : 
  a 0 = 1 ∧ 
  (∀ i : Fin 5, a (i + 1) = 3 * a i) ∧ 
  a 4 = 81 ∧ 
  a 5 = 243 → 
  a 3 = 27 := by
sorry

end NUMINAMATH_CALUDE_button_sequence_l1962_196262


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1962_196268

theorem cubic_equation_solutions :
  let f : ℂ → ℂ := λ x => x^3 + 4*x^2*Real.sqrt 3 + 12*x + 8*Real.sqrt 3 + x + Real.sqrt 3
  ∃ (z₁ z₂ z₃ : ℂ), 
    z₁ = -Real.sqrt 3 ∧ 
    z₂ = -Real.sqrt 3 + Complex.I ∧ 
    z₃ = -Real.sqrt 3 - Complex.I ∧
    (∀ z : ℂ, f z = 0 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1962_196268


namespace NUMINAMATH_CALUDE_log_equation_implies_p_zero_l1962_196218

theorem log_equation_implies_p_zero (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq2 : q + 2 > 0) :
  Real.log p - Real.log q = Real.log (p / (q + 2)) → p = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_implies_p_zero_l1962_196218


namespace NUMINAMATH_CALUDE_number_problem_l1962_196232

theorem number_problem (n : ℚ) : (1/2 : ℚ) * (3/5 : ℚ) * n = 36 → n = 120 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1962_196232


namespace NUMINAMATH_CALUDE_first_three_decimal_digits_l1962_196290

theorem first_three_decimal_digits (n : ℕ) (x : ℝ) : 
  n = 2005 → x = (10^n + 1)^(11/8) → 
  ∃ (k : ℕ), x = k + 0.375 + r ∧ 0 ≤ r ∧ r < 0.001 :=
sorry

end NUMINAMATH_CALUDE_first_three_decimal_digits_l1962_196290


namespace NUMINAMATH_CALUDE_zoo_feeding_arrangements_l1962_196294

/-- Represents the number of pairs of animals in the zoo -/
def num_pairs : ℕ := 6

/-- Calculates the number of ways to arrange the animals according to the specified pattern -/
def arrangement_count : ℕ :=
  (num_pairs - 1) * -- choices for the second female
  num_pairs * -- choices for the first male
  (Finset.prod (Finset.range (num_pairs - 1)) (λ i => num_pairs - i)) * -- choices for remaining females
  (Finset.prod (Finset.range (num_pairs - 1)) (λ i => num_pairs - i)) -- choices for remaining males

/-- The theorem stating that the number of possible arrangements is 432000 -/
theorem zoo_feeding_arrangements : arrangement_count = 432000 := by
  sorry

end NUMINAMATH_CALUDE_zoo_feeding_arrangements_l1962_196294


namespace NUMINAMATH_CALUDE_smallest_k_divisible_by_24_l1962_196240

-- Define q as the largest prime with 2021 digits
def q : ℕ := sorry

-- Axiom: q is prime
axiom q_prime : Nat.Prime q

-- Axiom: q has 2021 digits
axiom q_digits : 10^2020 ≤ q ∧ q < 10^2021

-- Theorem to prove
theorem smallest_k_divisible_by_24 :
  ∃ k : ℕ, k > 0 ∧ 24 ∣ (q^2 - k) ∧ ∀ m : ℕ, m > 0 → 24 ∣ (q^2 - m) → k ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_k_divisible_by_24_l1962_196240


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l1962_196252

/-- The slope of a chord of an ellipse bisected by a given point -/
theorem ellipse_chord_slope (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 / 36 + y₁^2 / 9 = 1) →  -- P(x₁, y₁) is on the ellipse
  (x₂^2 / 36 + y₂^2 / 9 = 1) →  -- Q(x₂, y₂) is on the ellipse
  ((x₁ + x₂) / 2 = 1) →         -- Midpoint x-coordinate is 1
  ((y₁ + y₂) / 2 = 1) →         -- Midpoint y-coordinate is 1
  (y₂ - y₁) / (x₂ - x₁) = -1/4  -- Slope of PQ is -1/4
:= by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l1962_196252


namespace NUMINAMATH_CALUDE_rainfall_difference_l1962_196257

theorem rainfall_difference (sunday monday tuesday : ℝ) : 
  sunday = 4 ∧ 
  monday > sunday ∧ 
  tuesday = 2 * monday ∧ 
  sunday + monday + tuesday = 25 →
  monday - sunday = 3 := by
sorry

end NUMINAMATH_CALUDE_rainfall_difference_l1962_196257


namespace NUMINAMATH_CALUDE_binary_calculation_l1962_196212

theorem binary_calculation : 
  (0b101101 * 0b10101 + 0b1010 / 0b10) = 0b110111100000 := by
  sorry

end NUMINAMATH_CALUDE_binary_calculation_l1962_196212


namespace NUMINAMATH_CALUDE_triangle_altitude_and_median_l1962_196278

/-- Triangle with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ := (4, 0)
  B : ℝ × ℝ := (6, 7)
  C : ℝ × ℝ := (0, 3)

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The altitude from A to BC -/
def altitude (t : Triangle) : LineEquation :=
  { a := 2, b := 7, c := -21 }

/-- The median from BC -/
def median (t : Triangle) : LineEquation :=
  { a := 5, b := 1, c := -20 }

theorem triangle_altitude_and_median (t : Triangle) :
  (altitude t = { a := 2, b := 7, c := -21 }) ∧
  (median t = { a := 5, b := 1, c := -20 }) := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_median_l1962_196278


namespace NUMINAMATH_CALUDE_equation_factorization_l1962_196207

theorem equation_factorization :
  ∀ x : ℝ, (5*x - 1)^2 = 3*(5*x - 1) ↔ (5*x - 1)*(5*x - 4) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_factorization_l1962_196207


namespace NUMINAMATH_CALUDE_divisibility_property_l1962_196225

theorem divisibility_property (n : ℕ) (h1 : n > 2) (h2 : Even n) :
  (n + 1) ∣ (n + 1)^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1962_196225


namespace NUMINAMATH_CALUDE_cost_per_share_is_50_l1962_196216

/-- Represents the savings and investment scenario of a married couple --/
structure SavingsScenario where
  wife_weekly_savings : ℕ
  husband_monthly_savings : ℕ
  savings_period_months : ℕ
  investment_fraction : ℚ
  num_shares_bought : ℕ

/-- Calculates the cost per share of stock based on the given savings scenario --/
def cost_per_share (scenario : SavingsScenario) : ℚ :=
  let total_savings := (scenario.wife_weekly_savings * 4 * scenario.savings_period_months +
                        scenario.husband_monthly_savings * scenario.savings_period_months)
  let investment_amount := (total_savings : ℚ) * scenario.investment_fraction
  investment_amount / scenario.num_shares_bought

/-- Theorem stating that the cost per share is $50 for the given scenario --/
theorem cost_per_share_is_50 (scenario : SavingsScenario)
  (h1 : scenario.wife_weekly_savings = 100)
  (h2 : scenario.husband_monthly_savings = 225)
  (h3 : scenario.savings_period_months = 4)
  (h4 : scenario.investment_fraction = 1/2)
  (h5 : scenario.num_shares_bought = 25) :
  cost_per_share scenario = 50 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_share_is_50_l1962_196216


namespace NUMINAMATH_CALUDE_division_problem_l1962_196277

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 689 →
  divisor = 36 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  quotient = 19 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1962_196277


namespace NUMINAMATH_CALUDE_trajectory_equation_l1962_196255

/-- The trajectory of points equidistant from A(-1, 1, 0) and B(2, -1, -1) in 3D space -/
theorem trajectory_equation :
  ∀ (x y z : ℝ),
  (x + 1)^2 + (y - 1)^2 + z^2 = (x - 2)^2 + (y + 1)^2 + (z + 1)^2 →
  3*x - 2*y - z = 2 := by
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1962_196255


namespace NUMINAMATH_CALUDE_yellow_red_difference_after_border_l1962_196219

/-- Represents a hexagonal tile figure --/
structure HexFigure where
  red_tiles : ℕ
  yellow_tiles : ℕ

/-- Adds a border of yellow tiles to a hexagonal figure --/
def add_yellow_border (fig : HexFigure) : HexFigure :=
  { red_tiles := fig.red_tiles,
    yellow_tiles := fig.yellow_tiles + 18 }

/-- The initial hexagonal figure --/
def initial_figure : HexFigure :=
  { red_tiles := 15,
    yellow_tiles := 10 }

theorem yellow_red_difference_after_border :
  (add_yellow_border initial_figure).yellow_tiles - (add_yellow_border initial_figure).red_tiles = 13 :=
by sorry

end NUMINAMATH_CALUDE_yellow_red_difference_after_border_l1962_196219


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l1962_196272

def complex_number (a b : ℝ) : ℂ := Complex.mk a b

theorem z_in_third_quadrant :
  let i : ℂ := complex_number 0 1
  let z : ℂ := (1 - 2 * i) / i
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l1962_196272


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1962_196264

def U : Set Nat := {0, 1, 2, 3, 4, 5}
def M : Set Nat := {0, 3, 5}
def N : Set Nat := {1, 4, 5}

theorem intersection_complement_equality : M ∩ (U \ N) = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1962_196264


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1962_196239

theorem x_squared_plus_reciprocal (x : ℝ) (hx : x ≠ 0) :
  x^4 + 1/x^4 = 23 → x^2 + 1/x^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1962_196239


namespace NUMINAMATH_CALUDE_fencing_requirement_l1962_196205

/-- Represents a rectangular field with given dimensions and fencing requirements. -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  uncovered_side : ℝ

/-- Calculates the required fencing for a rectangular field. -/
def required_fencing (field : RectangularField) : ℝ :=
  field.length + 2 * field.width

/-- Theorem stating the required fencing for a specific rectangular field. -/
theorem fencing_requirement (field : RectangularField)
  (h1 : field.area = 650)
  (h2 : field.uncovered_side = 20)
  (h3 : field.area = field.length * field.width)
  (h4 : field.length = field.uncovered_side) :
  required_fencing field = 85 :=
sorry

end NUMINAMATH_CALUDE_fencing_requirement_l1962_196205


namespace NUMINAMATH_CALUDE_percentage_equality_l1962_196251

theorem percentage_equality (x y : ℝ) (hx : x ≠ 0) :
  (0.4 * 0.5 * x = 0.2 * 0.3 * y) → y = (10/3) * x := by
sorry

end NUMINAMATH_CALUDE_percentage_equality_l1962_196251


namespace NUMINAMATH_CALUDE_inequality_not_always_hold_l1962_196298

theorem inequality_not_always_hold (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) :
  ¬ (∀ a b c, a > b ∧ b > 0 ∧ c ≠ 0 → (a - b) / c > 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_hold_l1962_196298


namespace NUMINAMATH_CALUDE_myrtle_dropped_eggs_l1962_196241

/-- Represents the problem of calculating how many eggs Myrtle dropped --/
theorem myrtle_dropped_eggs (hens : ℕ) (eggs_per_hen : ℕ) (days : ℕ) (neighbor_took : ℕ) (myrtle_has : ℕ)
  (h1 : hens = 3)
  (h2 : eggs_per_hen = 3)
  (h3 : days = 7)
  (h4 : neighbor_took = 12)
  (h5 : myrtle_has = 46) :
  hens * eggs_per_hen * days - neighbor_took - myrtle_has = 5 := by
  sorry

#check myrtle_dropped_eggs

end NUMINAMATH_CALUDE_myrtle_dropped_eggs_l1962_196241


namespace NUMINAMATH_CALUDE_age_difference_l1962_196259

/-- Represents the ages of Ramesh, Mahesh, and Suresh -/
structure Ages where
  ramesh : ℕ
  mahesh : ℕ
  suresh : ℕ

/-- The ratio of present ages -/
def presentRatio (a : Ages) : Bool :=
  2 * a.mahesh = 5 * a.ramesh ∧ 5 * a.suresh = 8 * a.mahesh

/-- The ratio of ages after 15 years -/
def futureRatio (a : Ages) : Bool :=
  14 * (a.ramesh + 15) = 9 * (a.mahesh + 15) ∧
  21 * (a.mahesh + 15) = 14 * (a.suresh + 15)

/-- The theorem to be proved -/
theorem age_difference (a : Ages) :
  presentRatio a → futureRatio a → a.suresh - a.mahesh = 45 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1962_196259


namespace NUMINAMATH_CALUDE_soccer_team_lineup_count_l1962_196284

theorem soccer_team_lineup_count :
  let team_size : ℕ := 16
  let positions_to_fill : ℕ := 5
  (team_size.factorial) / ((team_size - positions_to_fill).factorial) = 524160 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_lineup_count_l1962_196284


namespace NUMINAMATH_CALUDE_xyz_value_l1962_196288

theorem xyz_value (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5)
  (eq2 : y + 1/z = 2)
  (eq3 : z + 1/x = 8/3) :
  x * y * z = 8 + 3 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1962_196288


namespace NUMINAMATH_CALUDE_triangle_side_perp_distance_relation_l1962_196244

/-- Represents a triangle with side lengths and perpendicular distances -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ

/-- Theorem stating the relationship between side lengths and perpendicular distances -/
theorem triangle_side_perp_distance_relation (t : Triangle) 
  (h_side : t.a < t.b ∧ t.b < t.c) : 
  t.h_a > t.h_b ∧ t.h_b > t.h_c := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_perp_distance_relation_l1962_196244


namespace NUMINAMATH_CALUDE_toys_produced_daily_l1962_196287

/-- The number of toys produced per week -/
def toys_per_week : ℕ := 3400

/-- The number of working days per week -/
def working_days : ℕ := 5

/-- The number of toys produced each day -/
def toys_per_day : ℕ := toys_per_week / working_days

/-- Theorem stating that the number of toys produced each day is 680 -/
theorem toys_produced_daily : toys_per_day = 680 := by
  sorry

end NUMINAMATH_CALUDE_toys_produced_daily_l1962_196287


namespace NUMINAMATH_CALUDE_final_number_is_odd_l1962_196289

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The process of replacing two numbers with their absolute difference cubed -/
def replace_process (a b : ℤ) : ℤ := |a - b|^3

/-- The theorem stating that the final number after the replace process is odd -/
theorem final_number_is_odd (n : ℕ) (h : n = 2017) :
  ∃ (k : ℕ), Odd (sum_to_n n) ∧
  (∀ (a b : ℤ), Odd (a + b) ↔ Odd (replace_process a b)) →
  Odd k := by sorry

end NUMINAMATH_CALUDE_final_number_is_odd_l1962_196289


namespace NUMINAMATH_CALUDE_green_ball_probability_l1962_196235

/-- Represents a container with red and green balls -/
structure Container where
  red : Nat
  green : Nat

/-- The game setup -/
def game : List Container := [
  { red := 8, green := 4 },
  { red := 7, green := 4 },
  { red := 7, green := 4 }
]

/-- The probability of selecting each container -/
def containerProb : Rat := 1 / 3

/-- Calculates the probability of drawing a green ball from a given container -/
def greenProbFromContainer (c : Container) : Rat :=
  c.green / (c.red + c.green)

/-- Calculates the total probability of drawing a green ball -/
def totalGreenProb : Rat :=
  (game.map greenProbFromContainer).sum / game.length

/-- The main theorem: the probability of drawing a green ball is 35/99 -/
theorem green_ball_probability : totalGreenProb = 35 / 99 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l1962_196235


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l1962_196299

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 160 ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 160 := by sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l1962_196299


namespace NUMINAMATH_CALUDE_smallest_factor_l1962_196276

theorem smallest_factor (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 36 → ¬(2^5 ∣ (936 * m) ∧ 3^3 ∣ (936 * m) ∧ 12^2 ∣ (936 * m))) ∧
  (2^5 ∣ (936 * 36) ∧ 3^3 ∣ (936 * 36) ∧ 12^2 ∣ (936 * 36)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_l1962_196276


namespace NUMINAMATH_CALUDE_min_people_with_hat_and_glove_l1962_196228

theorem min_people_with_hat_and_glove (n : ℕ) (gloves hats both : ℕ) : 
  n > 0 → 
  gloves = (2 * n) / 5 → 
  hats = (3 * n) / 4 → 
  both = gloves + hats - n → 
  both ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_min_people_with_hat_and_glove_l1962_196228


namespace NUMINAMATH_CALUDE_three_over_x_is_fraction_l1962_196223

/-- A fraction is defined as an expression with a variable in the denominator. -/
def is_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (a : ℚ) (b : ℚ → ℚ), ∀ x, f x = a / (b x) ∧ b x ≠ 0

/-- The function f(x) = 3/x is a fraction. -/
theorem three_over_x_is_fraction :
  is_fraction (λ x : ℚ => 3 / x) :=
sorry

end NUMINAMATH_CALUDE_three_over_x_is_fraction_l1962_196223


namespace NUMINAMATH_CALUDE_coefficient_c_positive_l1962_196263

-- Define a quadratic trinomial
def quadratic_trinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for no roots
def no_roots (a b c : ℝ) : Prop := ∀ x, quadratic_trinomial a b c x ≠ 0

-- Theorem statement
theorem coefficient_c_positive
  (a b c : ℝ)
  (h1 : no_roots a b c)
  (h2 : a + b + c > 0) :
  c > 0 :=
sorry

end NUMINAMATH_CALUDE_coefficient_c_positive_l1962_196263


namespace NUMINAMATH_CALUDE_adams_dried_fruits_l1962_196256

/-- The problem of calculating the amount of dried fruits Adam bought --/
theorem adams_dried_fruits :
  ∀ (dried_fruits : ℝ),
  (3 : ℝ) * 12 + dried_fruits * 8 = 56 →
  dried_fruits = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_adams_dried_fruits_l1962_196256


namespace NUMINAMATH_CALUDE_larger_integer_problem_l1962_196293

theorem larger_integer_problem (a b : ℕ+) : 
  (b : ℚ) / (a : ℚ) = 7 / 3 → 
  (a : ℕ) * b = 189 → 
  b = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l1962_196293


namespace NUMINAMATH_CALUDE_hyperbola_tangent_slope_range_l1962_196214

/-- The range of slopes for a line passing through the right focus of a hyperbola
    and intersecting its right branch at exactly one point. -/
theorem hyperbola_tangent_slope_range (x y : ℝ) :
  (x^2 / 4 - y^2 / 12 = 1) →  -- Equation of the hyperbola
  ∃ (m : ℝ), -- Slope of the line
    (m ≥ -Real.sqrt 3 ∧ m ≤ Real.sqrt 3) ∧ -- Range of slopes
    (∃ (x₀ y₀ : ℝ), -- Point of intersection
      x₀^2 / 4 - y₀^2 / 12 = 1 ∧ -- Point lies on the hyperbola
      y₀ = m * (x₀ - (Real.sqrt 5))) ∧ -- Line passes through right focus (√5, 0)
    (∀ (x₁ y₁ : ℝ), -- Uniqueness of intersection
      x₁ ≠ x₀ →
      x₁^2 / 4 - y₁^2 / 12 = 1 →
      y₁ ≠ m * (x₁ - (Real.sqrt 5))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_slope_range_l1962_196214


namespace NUMINAMATH_CALUDE_smallest_number_l1962_196269

theorem smallest_number (a b c d : ℚ) (ha : a = -2) (hb : b = -5/2) (hc : c = 0) (hd : d = 1/5) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1962_196269


namespace NUMINAMATH_CALUDE_exam_papers_count_l1962_196202

theorem exam_papers_count (average : ℝ) (new_average : ℝ) (geography_increase : ℝ) (history_increase : ℝ) :
  average = 63 →
  new_average = 65 →
  geography_increase = 20 →
  history_increase = 2 →
  ∃ n : ℕ, n * average + geography_increase + history_increase = n * new_average ∧ n = 11 :=
by sorry

end NUMINAMATH_CALUDE_exam_papers_count_l1962_196202


namespace NUMINAMATH_CALUDE_ribbon_fraction_l1962_196211

theorem ribbon_fraction (total_fraction : ℚ) (num_packages : ℕ) 
  (h1 : total_fraction = 5 / 12)
  (h2 : num_packages = 5) :
  total_fraction / num_packages = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_fraction_l1962_196211


namespace NUMINAMATH_CALUDE_sum_of_a_values_l1962_196201

theorem sum_of_a_values (a b : ℝ) (h1 : a + 1/b = 8) (h2 : b + 1/a = 3) : 
  ∃ (a₁ a₂ : ℝ), (a₁ + 1/b = 8 ∧ b + 1/a₁ = 3) ∧ 
                 (a₂ + 1/b = 8 ∧ b + 1/a₂ = 3) ∧ 
                 a₁ ≠ a₂ ∧ 
                 a₁ + a₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_values_l1962_196201


namespace NUMINAMATH_CALUDE_fruit_basket_cost_l1962_196273

/-- Represents the composition and prices of a fruit basket -/
structure FruitBasket where
  banana_count : ℕ
  apple_count : ℕ
  strawberry_count : ℕ
  avocado_count : ℕ
  banana_price : ℚ
  apple_price : ℚ
  strawberry_dozen_price : ℚ
  avocado_price : ℚ
  grape_half_bunch_price : ℚ

/-- Calculates the total cost of the fruit basket -/
def total_cost (fb : FruitBasket) : ℚ :=
  fb.banana_count * fb.banana_price +
  fb.apple_count * fb.apple_price +
  (fb.strawberry_count / 12) * fb.strawberry_dozen_price +
  fb.avocado_count * fb.avocado_price +
  2 * fb.grape_half_bunch_price

/-- The given fruit basket -/
def given_basket : FruitBasket := {
  banana_count := 4
  apple_count := 3
  strawberry_count := 24
  avocado_count := 2
  banana_price := 1
  apple_price := 2
  strawberry_dozen_price := 4
  avocado_price := 3
  grape_half_bunch_price := 2
}

/-- Theorem stating that the total cost of the given fruit basket is $28 -/
theorem fruit_basket_cost : total_cost given_basket = 28 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_cost_l1962_196273


namespace NUMINAMATH_CALUDE_kiyana_grapes_l1962_196246

/-- Proves that if Kiyana has 24 grapes and gives away half of them, the number of grapes she gives away is 12. -/
theorem kiyana_grapes : 
  let total_grapes : ℕ := 24
  let grapes_given_away : ℕ := total_grapes / 2
  grapes_given_away = 12 := by
  sorry

end NUMINAMATH_CALUDE_kiyana_grapes_l1962_196246


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1962_196274

theorem complex_fraction_equality : 50 / (8 - 3/7) = 350/53 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1962_196274


namespace NUMINAMATH_CALUDE_inscribed_squares_equal_area_l1962_196247

/-- 
Given an isosceles right triangle with an inscribed square parallel to the legs,
prove that if this square has an area of 625, then a square inscribed with sides
parallel and perpendicular to the hypotenuse also has an area of 625.
-/
theorem inscribed_squares_equal_area (side : ℝ) (h_area : side^2 = 625) :
  let hypotenuse := side * Real.sqrt 2
  let side_hyp_square := hypotenuse / 2
  side_hyp_square^2 = 625 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_equal_area_l1962_196247


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1962_196208

theorem max_value_sqrt_sum (x : ℝ) (h : -36 ≤ x ∧ x ≤ 36) :
  Real.sqrt (36 + x) + Real.sqrt (36 - x) ≤ 12 ∧
  ∃ y, -36 ≤ y ∧ y ≤ 36 ∧ Real.sqrt (36 + y) + Real.sqrt (36 - y) = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1962_196208


namespace NUMINAMATH_CALUDE_at_least_one_blue_multiple_of_three_l1962_196209

/-- Represents a marked point on the circle --/
structure MarkedPoint where
  value : Int

/-- Represents a chord on the circle --/
structure Chord where
  points : List MarkedPoint

/-- The configuration of chords and points on the circle --/
structure CircleConfiguration where
  chords : List Chord
  endpointZeros : Nat
  endpointOnes : Nat

/-- Calculates yellow numbers (sum of endpoint values) for a chord --/
def yellowNumbers (chord : Chord) : List Int :=
  sorry

/-- Calculates blue numbers (absolute difference of endpoint values) for a chord --/
def blueNumbers (chord : Chord) : List Int :=
  sorry

/-- Checks if the yellow numbers are consecutive from 0 to N --/
def isConsecutiveYellow (yellowNums : List Int) : Bool :=
  sorry

theorem at_least_one_blue_multiple_of_three 
  (config : CircleConfiguration) 
  (h1 : config.chords.length = 2019)
  (h2 : config.endpointZeros = 2019)
  (h3 : config.endpointOnes = 2019)
  (h4 : ∀ c ∈ config.chords, c.points.length ≥ 2)
  (h5 : let allYellow := config.chords.map yellowNumbers |>.join
        isConsecutiveYellow allYellow) :
  ∃ (b : Int), b ∈ (config.chords.map blueNumbers |>.join) ∧ b % 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_at_least_one_blue_multiple_of_three_l1962_196209


namespace NUMINAMATH_CALUDE_stock_price_change_l1962_196217

theorem stock_price_change (total_stocks : ℕ) (higher_percentage : ℚ) : 
  total_stocks = 4200 →
  higher_percentage = 35/100 →
  ∃ (higher lower : ℕ),
    higher + lower = total_stocks ∧
    higher = (1 + higher_percentage) * lower ∧
    higher = 2412 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l1962_196217


namespace NUMINAMATH_CALUDE_smallest_four_digit_sum_16_l1962_196279

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem smallest_four_digit_sum_16 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 16 → 1960 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_sum_16_l1962_196279


namespace NUMINAMATH_CALUDE_triple_debt_days_l1962_196200

def loan_amount : ℝ := 20
def daily_interest_rate : ℝ := 0.10

def days_to_triple_debt : ℕ := 20

theorem triple_debt_days :
  ∀ n : ℕ, (n : ℝ) ≥ (days_to_triple_debt : ℝ) ↔
  loan_amount * (1 + n * daily_interest_rate) ≥ 3 * loan_amount :=
by sorry

end NUMINAMATH_CALUDE_triple_debt_days_l1962_196200
