import Mathlib

namespace NUMINAMATH_CALUDE_product_of_four_numbers_l879_87941

theorem product_of_four_numbers (A B C D : ℝ) : 
  A > 0 → B > 0 → C > 0 → D > 0 →
  A + B + C + D = 40 →
  A + 3 = B - 3 ∧ A + 3 = C * 3 ∧ A + 3 = D / 3 →
  A * B * C * D = 2666.25 := by
sorry

end NUMINAMATH_CALUDE_product_of_four_numbers_l879_87941


namespace NUMINAMATH_CALUDE_cubic_odd_and_increasing_l879_87929

def f (x : ℝ) : ℝ := x^3

theorem cubic_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_odd_and_increasing_l879_87929


namespace NUMINAMATH_CALUDE_triangle_properties_l879_87925

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : 3 * (t.b^2 + t.c^2) = 3 * t.a^2 + 2 * t.b * t.c)
  (h2 : t.a = 2)
  (h3 : t.b + t.c = 2 * Real.sqrt 2)
  (h4 : Real.sin t.B = Real.sqrt 2 * Real.cos t.C) :
  (∃ S : ℝ, S = Real.sqrt 2 / 2 ∧ S = 1/2 * t.b * t.c * Real.sin t.A) ∧ 
  Real.cos t.C = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l879_87925


namespace NUMINAMATH_CALUDE_horner_v3_value_l879_87935

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^6 - 5x^5 + 6x^4 + x^2 + 3x + 2 -/
def f : List ℤ := [2, 3, 1, 0, 6, -5, 1]

/-- Theorem: Horner's method for f(x) at x = -2 gives v₃ = -40 -/
theorem horner_v3_value :
  let coeffs := f.take 4
  horner coeffs (-2) = -40 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l879_87935


namespace NUMINAMATH_CALUDE_total_carrots_l879_87937

theorem total_carrots (sally_carrots fred_carrots : ℕ) :
  sally_carrots = 6 → fred_carrots = 4 → sally_carrots + fred_carrots = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_l879_87937


namespace NUMINAMATH_CALUDE_inverse_function_property_l879_87938

theorem inverse_function_property (f : ℝ → ℝ) (h_inv : Function.Injective f) :
  f 1 = 0 → (Function.invFun f 0) + 1 = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_function_property_l879_87938


namespace NUMINAMATH_CALUDE_external_tangent_intercept_l879_87981

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in slope-intercept form --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if a line is tangent to a circle --/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Checks if a line is external tangent to two circles --/
def isExternalTangent (l : Line) (c1 c2 : Circle) : Prop :=
  sorry

theorem external_tangent_intercept :
  let c1 : Circle := { center := (3, -2), radius := 3 }
  let c2 : Circle := { center := (15, 8), radius := 8 }
  ∀ l : Line,
    l.slope > 0 →
    isExternalTangent l c1 c2 →
    l.intercept = 720 / 11 :=
sorry

end NUMINAMATH_CALUDE_external_tangent_intercept_l879_87981


namespace NUMINAMATH_CALUDE_bear_ate_54_pies_l879_87922

/-- Represents the eating scenario of Masha and the Bear -/
structure EatingScenario where
  totalPies : ℕ
  bearRaspberrySpeed : ℕ
  bearPieSpeed : ℕ
  bearRaspberryRatio : ℕ

/-- Calculates the number of pies eaten by the Bear -/
def bearPies (scenario : EatingScenario) : ℕ :=
  sorry

/-- Theorem stating that the Bear ate 54 pies -/
theorem bear_ate_54_pies (scenario : EatingScenario) 
  (h1 : scenario.totalPies = 60)
  (h2 : scenario.bearRaspberrySpeed = 6)
  (h3 : scenario.bearPieSpeed = 3)
  (h4 : scenario.bearRaspberryRatio = 2) :
  bearPies scenario = 54 := by
  sorry

end NUMINAMATH_CALUDE_bear_ate_54_pies_l879_87922


namespace NUMINAMATH_CALUDE_inequality_holds_in_intervals_l879_87998

theorem inequality_holds_in_intervals (a b : ℝ) : 
  (((0 ≤ a ∧ a < b ∧ b ≤ π/2) ∨ (π ≤ a ∧ a < b ∧ b ≤ 3*π/2)) → 
   (a - Real.sin a < b - Real.sin b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_in_intervals_l879_87998


namespace NUMINAMATH_CALUDE_product_equals_442_l879_87910

/-- Converts a list of digits in a given base to its decimal (base 10) representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

/-- The binary representation of the first number -/
def binary_num : List Nat := [1, 0, 1, 1]

/-- The ternary representation of the second number -/
def ternary_num : List Nat := [1, 2, 0, 1]

theorem product_equals_442 :
  (to_decimal binary_num 2) * (to_decimal ternary_num 3) = 442 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_442_l879_87910


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l879_87949

theorem right_triangle_perimeter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_area : (1/2) * a * b = 150) (h_leg : a = 30) : 
  a + b + c = 40 + 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l879_87949


namespace NUMINAMATH_CALUDE_lcm_is_perfect_square_l879_87986

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a*b) % (a*b*(a - b)) = 0) : 
  ∃ k : ℕ, Nat.lcm a b = k^2 := by
sorry

end NUMINAMATH_CALUDE_lcm_is_perfect_square_l879_87986


namespace NUMINAMATH_CALUDE_cryptarithm_solution_exists_l879_87923

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_different_digits (Φ E B P A J : ℕ) : Prop :=
  is_valid_digit Φ ∧ is_valid_digit E ∧ is_valid_digit B ∧ 
  is_valid_digit P ∧ is_valid_digit A ∧ is_valid_digit J ∧
  Φ ≠ E ∧ Φ ≠ B ∧ Φ ≠ P ∧ Φ ≠ A ∧ Φ ≠ J ∧
  E ≠ B ∧ E ≠ P ∧ E ≠ A ∧ E ≠ J ∧
  B ≠ P ∧ B ≠ A ∧ B ≠ J ∧
  P ≠ A ∧ P ≠ J ∧
  A ≠ J

theorem cryptarithm_solution_exists :
  ∃ (Φ E B P A J : ℕ), 
    are_different_digits Φ E B P A J ∧
    (Φ : ℚ) / E + (B * 10 + P : ℚ) / A / J = 1 :=
sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_exists_l879_87923


namespace NUMINAMATH_CALUDE_equation_solution_l879_87973

theorem equation_solution : 
  ∃ x : ℝ, (x / (x - 1) = (x - 3) / (2*x - 2)) ∧ (x = -3) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l879_87973


namespace NUMINAMATH_CALUDE_max_n_for_specific_sequence_l879_87942

/-- Represents an arithmetic sequence with first term a₁, nth term aₙ, and common difference d. -/
structure ArithmeticSequence where
  a₁ : ℤ
  aₙ : ℤ
  d : ℕ+
  n : ℕ
  h_arithmetic : aₙ = a₁ + (n - 1) * d

/-- The maximum value of n for a specific arithmetic sequence. -/
def maxN (seq : ArithmeticSequence) : ℕ :=
  seq.n

/-- Theorem stating the maximum value of n for the given arithmetic sequence. -/
theorem max_n_for_specific_sequence :
  ∀ seq : ArithmeticSequence,
    seq.a₁ = -6 →
    seq.aₙ = 0 →
    seq.n ≥ 3 →
    maxN seq ≤ 7 ∧ ∃ seq' : ArithmeticSequence, seq'.a₁ = -6 ∧ seq'.aₙ = 0 ∧ seq'.n ≥ 3 ∧ maxN seq' = 7 :=
sorry

end NUMINAMATH_CALUDE_max_n_for_specific_sequence_l879_87942


namespace NUMINAMATH_CALUDE_candy_distribution_l879_87962

/-- 
Given a group of students where each student receives a fixed number of candy pieces,
this theorem proves that the total number of candy pieces given away is equal to
the product of the number of students and the number of pieces per student.
-/
theorem candy_distribution (num_students : ℕ) (pieces_per_student : ℕ) 
  (h1 : num_students = 9) 
  (h2 : pieces_per_student = 2) : 
  num_students * pieces_per_student = 18 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l879_87962


namespace NUMINAMATH_CALUDE_integral_evaluation_l879_87985

theorem integral_evaluation :
  ∫ x in (1 : ℝ)..2, (x + 1/x + 1/x^2) = 2 + Real.log 2 := by sorry

end NUMINAMATH_CALUDE_integral_evaluation_l879_87985


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_relation_l879_87979

theorem rectangular_prism_volume_relation (c : ℝ) (hc : c > 0) :
  let a := (4 : ℝ)^(1/3) * c
  let b := (2 : ℝ)^(1/3) * c
  2 * c^3 = a * b * c := by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_relation_l879_87979


namespace NUMINAMATH_CALUDE_sum_minus_seven_tenths_l879_87906

theorem sum_minus_seven_tenths (a b c : ℝ) (ha : a = 34.5) (hb : b = 15.2) (hc : c = 0.7) :
  a + b - c = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_minus_seven_tenths_l879_87906


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l879_87971

theorem min_value_and_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 2*b + 3*c = 8) →
  (∃ (m : ℝ), m = 1/a + 2/b + 3/c ∧ m ≥ 4.5 ∧ ∀ (x : ℝ), x = 1/a + 2/b + 3/c → x ≥ m) ∧
  (∃ (x : ℝ), (x = a + 1/b ∨ x = b + 1/c ∨ x = c + 1/a) ∧ x ≥ 2) :=
by sorry


end NUMINAMATH_CALUDE_min_value_and_inequality_l879_87971


namespace NUMINAMATH_CALUDE_virginia_march_rainfall_l879_87951

/-- Calculates the rainfall in March given the rainfall amounts for April, May, June, July, and the average rainfall for 5 months. -/
def march_rainfall (april may june july average : ℝ) : ℝ :=
  5 * average - (april + may + june + july)

/-- Theorem stating that the rainfall in March was 3.79 inches given the specified conditions. -/
theorem virginia_march_rainfall :
  let april : ℝ := 4.5
  let may : ℝ := 3.95
  let june : ℝ := 3.09
  let july : ℝ := 4.67
  let average : ℝ := 4
  march_rainfall april may june july average = 3.79 := by
  sorry

end NUMINAMATH_CALUDE_virginia_march_rainfall_l879_87951


namespace NUMINAMATH_CALUDE_probability_inequality_l879_87991

theorem probability_inequality (p q : ℝ) (m n : ℕ+) 
  (h1 : p ≥ 0) (h2 : q ≥ 0) (h3 : p + q = 1) :
  (1 - p ^ (m : ℝ)) ^ (n : ℝ) + (1 - q ^ (n : ℝ)) ^ (m : ℝ) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_probability_inequality_l879_87991


namespace NUMINAMATH_CALUDE_transform_sin_function_l879_87976

open Real

theorem transform_sin_function (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π/2) :
  let f : ℝ → ℝ := λ x ↦ 2 * sin (3*x + φ)
  let g : ℝ → ℝ := λ x ↦ 2 * sin (3*x) + 1
  (∀ x, f x = f (2*φ - x)) →  -- (φ, 0) is center of symmetry
  (∃ h : ℝ → ℝ, ∀ x, g x = h (f (x - π/12)) + 1) :=
by sorry

end NUMINAMATH_CALUDE_transform_sin_function_l879_87976


namespace NUMINAMATH_CALUDE_triangle_problem_l879_87978

open Real

theorem triangle_problem (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * sin (A - C) = sin B) :
  sin A = (3 * sqrt 10) / 10 ∧
  (∀ (AB : ℝ), AB = 5 → ∃ (h : ℝ), h = 6 ∧ h * AB / 2 = sin C * (AB * sin A / sin C) * (AB * sin B / sin C) / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l879_87978


namespace NUMINAMATH_CALUDE_trapezoid_solution_l879_87957

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  a : ℝ  -- Length of the shorter parallel side
  h : ℝ  -- Height of the trapezoid
  area : ℝ -- Area of the trapezoid

/-- Properties of the trapezoid -/
def trapezoid_properties (t : Trapezoid) : Prop :=
  t.h = (2 * t.a + 3) / 2 ∧
  t.area = t.a^2 + 3 * t.a + 9 / 4 ∧
  t.area = 2 * t.a^2 - 7.75

/-- Theorem stating the solution to the trapezoid problem -/
theorem trapezoid_solution (t : Trapezoid) (h : trapezoid_properties t) :
  t.a = 5 ∧ t.a + 3 = 8 ∧ t.h = 6.5 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_solution_l879_87957


namespace NUMINAMATH_CALUDE_seating_theorem_l879_87940

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (n : ℕ) (no_adjacent_pair : ℕ) (no_adjacent_triple : ℕ) : ℕ :=
  factorial n - (factorial (n - 1) * factorial 2 + factorial (n - 2) * factorial 3) + 
  factorial (n - 3) * factorial 2 * factorial 3

theorem seating_theorem : seating_arrangements 8 2 3 = 25360 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l879_87940


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l879_87964

theorem cos_sixty_degrees : Real.cos (60 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l879_87964


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l879_87961

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | Real.log (x - 1) ≤ 0}

-- State the theorem
theorem intersection_equals_open_interval :
  A ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l879_87961


namespace NUMINAMATH_CALUDE_city_inhabitants_problem_l879_87958

theorem city_inhabitants_problem :
  ∃ n : ℕ,
    n > 150 ∧
    (∃ x : ℕ, n = x^2) ∧
    (∃ y : ℕ, n + 1000 = y^2 + 1) ∧
    (∃ z : ℕ, n + 2000 = z^2) ∧
    n = 249001 := by
  sorry

end NUMINAMATH_CALUDE_city_inhabitants_problem_l879_87958


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l879_87982

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The value of 'a' for which the given lines are parallel -/
theorem parallel_lines_a_value :
  (∀ x y : ℝ, 3 * y + 6 * a = 9 * x ↔ y - 2 = (2 * a - 3) * x) → a = 3 := by
  sorry

#check parallel_lines_a_value

end NUMINAMATH_CALUDE_parallel_lines_a_value_l879_87982


namespace NUMINAMATH_CALUDE_dave_won_fifteen_tickets_l879_87948

/-- Calculates the number of tickets Dave won later at the arcade -/
def tickets_won_later (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : ℕ :=
  final_tickets - (initial_tickets - spent_tickets)

/-- Theorem stating that Dave won 15 tickets later -/
theorem dave_won_fifteen_tickets :
  tickets_won_later 25 22 18 = 15 := by
  sorry

end NUMINAMATH_CALUDE_dave_won_fifteen_tickets_l879_87948


namespace NUMINAMATH_CALUDE_train_length_proof_l879_87936

-- Define the speed of the train in km/hr
def train_speed_kmh : ℝ := 108

-- Define the time it takes for the train to pass the tree in seconds
def passing_time : ℝ := 8

-- Theorem to prove the length of the train
theorem train_length_proof : 
  train_speed_kmh * 1000 / 3600 * passing_time = 240 := by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l879_87936


namespace NUMINAMATH_CALUDE_monk_problem_l879_87928

theorem monk_problem (total_mantou total_monks : ℕ) 
  (big_monk_consumption small_monk_consumption : ℚ) :
  total_mantou = 100 →
  total_monks = 100 →
  big_monk_consumption = 1 →
  small_monk_consumption = 1/3 →
  ∃ (big_monks small_monks : ℕ),
    big_monks + small_monks = total_monks ∧
    big_monks * big_monk_consumption + small_monks * small_monk_consumption = total_mantou ∧
    big_monks = 25 ∧
    small_monks = 75 := by
  sorry

end NUMINAMATH_CALUDE_monk_problem_l879_87928


namespace NUMINAMATH_CALUDE_inequality_proof_l879_87987

theorem inequality_proof (x : ℝ) (h : x > 0) : Real.log (Real.exp 2 / x) ≤ (1 + x) / x := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l879_87987


namespace NUMINAMATH_CALUDE_sand_collection_total_weight_l879_87920

theorem sand_collection_total_weight (eden_buckets mary_buckets iris_buckets : ℕ) 
  (sand_weight_per_bucket : ℕ) :
  eden_buckets = 4 →
  mary_buckets = eden_buckets + 3 →
  iris_buckets = mary_buckets - 1 →
  sand_weight_per_bucket = 2 →
  (eden_buckets + mary_buckets + iris_buckets) * sand_weight_per_bucket = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_sand_collection_total_weight_l879_87920


namespace NUMINAMATH_CALUDE_different_meal_combinations_l879_87984

theorem different_meal_combinations (n : ℕ) (h : n = 12) : n * (n - 1) = 132 := by
  sorry

end NUMINAMATH_CALUDE_different_meal_combinations_l879_87984


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l879_87952

theorem circle_ratio_after_increase (r : ℝ) : 
  let new_radius : ℝ := r + 1
  let new_circumference : ℝ := 2 * Real.pi * new_radius
  let new_diameter : ℝ := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l879_87952


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l879_87988

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a^2 + b^2 = 31) : 
  a * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l879_87988


namespace NUMINAMATH_CALUDE_quadratic_factorization_l879_87911

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l879_87911


namespace NUMINAMATH_CALUDE_sean_patch_profit_l879_87963

/-- Calculates the net profit for Sean's patch business -/
theorem sean_patch_profit :
  let order_quantity : ℕ := 100
  let cost_per_patch : ℚ := 125/100
  let sell_price_per_patch : ℚ := 12
  let total_cost : ℚ := order_quantity * cost_per_patch
  let total_revenue : ℚ := order_quantity * sell_price_per_patch
  let net_profit : ℚ := total_revenue - total_cost
  net_profit = 1075 := by sorry

end NUMINAMATH_CALUDE_sean_patch_profit_l879_87963


namespace NUMINAMATH_CALUDE_light_ray_reflection_l879_87966

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The starting point A -/
def A : Point := ⟨-3, 4⟩

/-- The final point B -/
def B : Point := ⟨-2, 6⟩

/-- The equation of the light ray after reflecting off the y-axis -/
def final_ray : Line := ⟨2, 1, -2⟩

/-- Theorem stating that the final ray passes through point B and has the correct equation -/
theorem light_ray_reflection :
  (B.on_line final_ray) ∧
  (final_ray.a = 2 ∧ final_ray.b = 1 ∧ final_ray.c = -2) := by sorry

end NUMINAMATH_CALUDE_light_ray_reflection_l879_87966


namespace NUMINAMATH_CALUDE_sin_2x_plus_1_equals_shifted_cos_l879_87912

theorem sin_2x_plus_1_equals_shifted_cos (x : ℝ) : 
  Real.sin (2 * x) + 1 = Real.cos (2 * (x - π / 4)) + 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_plus_1_equals_shifted_cos_l879_87912


namespace NUMINAMATH_CALUDE_max_period_linear_recurrence_l879_87970

/-- The maximum period of a second-order linear recurrence sequence modulo a prime -/
theorem max_period_linear_recurrence (p : Nat) (hp : Prime p) 
  (a b c d : Int) : ∃ (x : Nat → Int), 
  (x 0 = c) ∧ 
  (x 1 = d) ∧ 
  (∀ n, x (n + 2) = a * x (n + 1) + b * x n) ∧ 
  (∃ t, t ≤ p^2 - 1 ∧ 
    ∀ n ≥ p^2, (x (n + t) : ZMod p) = (x n : ZMod p)) ∧
  (∀ t' < p^2 - 1, ∃ n ≥ p^2, (x (n + t') : ZMod p) ≠ (x n : ZMod p)) :=
sorry

end NUMINAMATH_CALUDE_max_period_linear_recurrence_l879_87970


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l879_87944

/-- Given a geometric sequence {a_n} where a_4 = 4, prove that a_2 * a_6 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence property
  a 4 = 4 →                                            -- given condition
  a 2 * a 6 = 16 :=                                    -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l879_87944


namespace NUMINAMATH_CALUDE_min_value_and_reciprocal_sum_l879_87975

noncomputable def f (a b c x : ℝ) : ℝ := |x + a| + |x - b| + c

theorem min_value_and_reciprocal_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x, f a b c x ≥ 5) 
  (hex : ∃ x, f a b c x = 5) :
  (a + b + c = 5) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → 1/a' + 1/b' + 1/c' ≥ 9/5) ∧
  (∃ a' b' c' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 1/a' + 1/b' + 1/c' = 9/5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_reciprocal_sum_l879_87975


namespace NUMINAMATH_CALUDE_range_of_a_l879_87994

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x₀ : ℝ, x₀^2 + 2*x₀ + a > 0)
  (h2 : ∀ x : ℝ, x > 0 → x + 1/x > a) : 
  1 < a ∧ a < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l879_87994


namespace NUMINAMATH_CALUDE_lock_code_difference_l879_87924

def is_valid_code (a b c : Nat) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (b * b) = (a * c * c)

def code_value (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

theorem lock_code_difference : 
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : Nat),
    is_valid_code a₁ b₁ c₁ ∧
    is_valid_code a₂ b₂ c₂ ∧
    (∀ a b c, is_valid_code a b c → 
      code_value a b c ≤ code_value a₁ b₁ c₁ ∧
      code_value a b c ≥ code_value a₂ b₂ c₂) ∧
    code_value a₁ b₁ c₁ - code_value a₂ b₂ c₂ = 541 :=
by sorry

end NUMINAMATH_CALUDE_lock_code_difference_l879_87924


namespace NUMINAMATH_CALUDE_tile_arrangements_l879_87960

def brown_tiles : ℕ := 2
def purple_tiles : ℕ := 1
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 4

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem tile_arrangements :
  (Nat.factorial total_tiles) / 
  (Nat.factorial yellow_tiles * Nat.factorial green_tiles * 
   Nat.factorial brown_tiles * Nat.factorial purple_tiles) = 12600 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l879_87960


namespace NUMINAMATH_CALUDE_guppies_needed_per_day_l879_87917

/-- The number of guppies eaten by a moray eel per day -/
def moray_eel_guppies : ℕ := 20

/-- The number of betta fish -/
def num_betta_fish : ℕ := 5

/-- The number of guppies eaten by each betta fish per day -/
def betta_fish_guppies : ℕ := 7

/-- The total number of guppies needed per day -/
def total_guppies : ℕ := moray_eel_guppies + num_betta_fish * betta_fish_guppies

theorem guppies_needed_per_day : total_guppies = 55 := by
  sorry

end NUMINAMATH_CALUDE_guppies_needed_per_day_l879_87917


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l879_87972

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 = 4 * p.2}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -2}

-- Define a function to get tangent points on C from a point on l
def tangentPoints (E : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ C ∧ ∃ m : ℝ, (p.2 - E.2) = m * (p.1 - E.1) ∧ p.1 = 2 * m}

-- Theorem statement
theorem tangent_line_intersection (E : ℝ × ℝ) (hE : E ∈ l) :
  ∃ A B : ℝ × ℝ, A ∈ tangentPoints E ∧ B ∈ tangentPoints E ∧ A ≠ B ∧
  ∃ t : ℝ, (1 - t) • A + t • B = (0, 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l879_87972


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l879_87933

/-- Proves that the initial average weight of 6 people in an elevator was 156 lbs,
    given that a 7th person weighing 121 lbs entered and increased the average to 151 lbs. -/
theorem elevator_weight_problem (initial_count : Nat) (new_person_weight : Nat) (new_average : Nat) :
  initial_count = 6 →
  new_person_weight = 121 →
  new_average = 151 →
  ∃ (initial_average : Nat),
    initial_average = 156 ∧
    (initial_count * initial_average + new_person_weight) / (initial_count + 1) = new_average :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l879_87933


namespace NUMINAMATH_CALUDE_valid_basis_vectors_l879_87915

def vector_a : Fin 2 → ℝ := ![3, 4]

def vector_e1 : Fin 2 → ℝ := ![-1, 2]
def vector_e2 : Fin 2 → ℝ := ![3, -1]

theorem valid_basis_vectors :
  ∃ (x y : ℝ), vector_a = x • vector_e1 + y • vector_e2 ∧
  ¬(∃ (k : ℝ), vector_e1 = k • vector_e2) :=
by sorry

end NUMINAMATH_CALUDE_valid_basis_vectors_l879_87915


namespace NUMINAMATH_CALUDE_fourth_power_of_cube_root_l879_87993

theorem fourth_power_of_cube_root (x : ℝ) : 
  x = (3 + Real.sqrt (1 + Real.sqrt 5)) ^ (1/3) → x^4 = 9 + 12 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_cube_root_l879_87993


namespace NUMINAMATH_CALUDE_fraction_equality_l879_87955

theorem fraction_equality (a b : ℝ) (h1 : 3 * a = 4 * b) (h2 : a * b ≠ 0) :
  (a + b) / a = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l879_87955


namespace NUMINAMATH_CALUDE_inequality_solution_set_l879_87932

theorem inequality_solution_set (x : ℝ) : (2 * x - 1) / (3 * x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l879_87932


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l879_87921

theorem complex_subtraction_simplification :
  (7 - 3*I) - (2 + 5*I) = 5 - 8*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l879_87921


namespace NUMINAMATH_CALUDE_two_faces_same_edges_l879_87959

/-- A face of a polyhedron -/
structure Face where
  edges : ℕ
  edges_ge_3 : edges ≥ 3

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Set Face
  nonempty : faces.Nonempty

theorem two_faces_same_edges (P : ConvexPolyhedron) : 
  ∃ f₁ f₂ : Face, f₁ ∈ P.faces ∧ f₂ ∈ P.faces ∧ f₁ ≠ f₂ ∧ f₁.edges = f₂.edges :=
sorry

end NUMINAMATH_CALUDE_two_faces_same_edges_l879_87959


namespace NUMINAMATH_CALUDE_rainfall_volume_calculation_l879_87953

-- Define the rainfall in centimeters
def rainfall_cm : ℝ := 5

-- Define the ground area in hectares
def ground_area_hectares : ℝ := 1.5

-- Define the conversion factor from hectares to square meters
def hectares_to_sqm : ℝ := 10000

-- Define the conversion factor from centimeters to meters
def cm_to_m : ℝ := 0.01

-- Theorem statement
theorem rainfall_volume_calculation :
  let rainfall_m := rainfall_cm * cm_to_m
  let ground_area_sqm := ground_area_hectares * hectares_to_sqm
  rainfall_m * ground_area_sqm = 750 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_volume_calculation_l879_87953


namespace NUMINAMATH_CALUDE_rep_for_A_percent_is_20_l879_87969

/-- Represents the voting scenario in a city -/
structure VotingScenario where
  total_voters : ℝ
  dem_percent : ℝ
  rep_percent : ℝ
  dem_for_A_percent : ℝ
  total_for_A_percent : ℝ
  rep_for_A_percent : ℝ

/-- The conditions of the voting scenario -/
def city_voting : VotingScenario :=
  { total_voters := 100, -- Assuming 100 for simplicity
    dem_percent := 60,
    rep_percent := 40,
    dem_for_A_percent := 85,
    total_for_A_percent := 59,
    rep_for_A_percent := 20 }

theorem rep_for_A_percent_is_20 (v : VotingScenario) (h1 : v.dem_percent + v.rep_percent = 100) 
    (h2 : v.dem_percent = 60) (h3 : v.dem_for_A_percent = 85) (h4 : v.total_for_A_percent = 59) :
  v.rep_for_A_percent = 20 := by
  sorry

#check rep_for_A_percent_is_20

end NUMINAMATH_CALUDE_rep_for_A_percent_is_20_l879_87969


namespace NUMINAMATH_CALUDE_football_season_duration_l879_87913

theorem football_season_duration (total_games : ℕ) (games_per_month : ℕ) 
  (h1 : total_games = 323) 
  (h2 : games_per_month = 19) : 
  total_games / games_per_month = 17 := by
  sorry

end NUMINAMATH_CALUDE_football_season_duration_l879_87913


namespace NUMINAMATH_CALUDE_some_frames_are_not_tars_l879_87980

universe u

-- Define the types
variable (Tar Kite Rope Frame : Type u)

-- Define the relations
variable (is_tar : Tar → Prop)
variable (is_kite : Kite → Prop)
variable (is_rope : Rope → Prop)
variable (is_frame : Frame → Prop)

-- Hypotheses
variable (h1 : ∀ t : Tar, ∃ k : Kite, is_kite k)
variable (h2 : ∀ k : Kite, ∀ r : Rope, ¬(is_kite k ∧ is_rope r))
variable (h3 : ∃ r : Rope, ∃ f : Frame, is_rope r ∧ is_frame f)

-- Theorem to prove
theorem some_frames_are_not_tars :
  ∃ f : Frame, ¬∃ t : Tar, is_frame f ∧ is_tar t :=
sorry

end NUMINAMATH_CALUDE_some_frames_are_not_tars_l879_87980


namespace NUMINAMATH_CALUDE_files_per_folder_l879_87945

theorem files_per_folder (initial_files : ℕ) (deleted_files : ℕ) (num_folders : ℕ) :
  initial_files = 93 →
  deleted_files = 21 →
  num_folders = 9 →
  (initial_files - deleted_files) / num_folders = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_files_per_folder_l879_87945


namespace NUMINAMATH_CALUDE_pascal_row15_element4_l879_87919

/-- Pascal's triangle element -/
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

/-- The fourth element in Row 15 of Pascal's triangle -/
def row15_element4 : ℕ := pascal 15 3

/-- Theorem: The fourth element in Row 15 of Pascal's triangle is 455 -/
theorem pascal_row15_element4 : row15_element4 = 455 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row15_element4_l879_87919


namespace NUMINAMATH_CALUDE_perimeter_is_ten_x_l879_87999

/-- The perimeter of a figure composed of rectangular segments -/
def perimeter_of_figure (x : ℝ) (hx : x ≠ 0) : ℝ :=
  let vertical_length1 := 3 * x
  let vertical_length2 := x
  let horizontal_length1 := 2 * x
  let horizontal_length2 := x
  vertical_length1 + vertical_length2 + horizontal_length1 + horizontal_length2 + 
  (3 * x - x) + (2 * x - x)

theorem perimeter_is_ten_x (x : ℝ) (hx : x ≠ 0) :
  perimeter_of_figure x hx = 10 * x := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_ten_x_l879_87999


namespace NUMINAMATH_CALUDE_cheesecakes_sold_l879_87939

theorem cheesecakes_sold (display : ℕ) (fridge : ℕ) (left : ℕ) : 
  display + fridge - left = display - (display + fridge - left - fridge) :=
by sorry

#check cheesecakes_sold 10 15 18

end NUMINAMATH_CALUDE_cheesecakes_sold_l879_87939


namespace NUMINAMATH_CALUDE_a_neg_two_sufficient_not_necessary_l879_87903

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def z (a : ℝ) : ℂ := Complex.mk (a^2 - 4) (a + 1)

theorem a_neg_two_sufficient_not_necessary :
  (∃ (a : ℝ), a ≠ -2 ∧ is_pure_imaginary (z a)) ∧
  (∀ (a : ℝ), a = -2 → is_pure_imaginary (z a)) :=
sorry

end NUMINAMATH_CALUDE_a_neg_two_sufficient_not_necessary_l879_87903


namespace NUMINAMATH_CALUDE_average_visitors_is_276_l879_87983

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitors (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays := 4
  let totalOtherDays := 26
  let totalVisitors := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  totalVisitors / 30

/-- Theorem stating that the average number of visitors is 276 given the specified conditions -/
theorem average_visitors_is_276 :
  averageVisitors 510 240 = 276 := by
  sorry

end NUMINAMATH_CALUDE_average_visitors_is_276_l879_87983


namespace NUMINAMATH_CALUDE_last_ten_seconds_distance_l879_87908

/-- The distance function of a plane's taxiing after landing -/
def distance (t : ℝ) : ℝ := 60 * t - 1.5 * t^2

/-- The time at which the plane stops -/
def stop_time : ℝ := 20

/-- Theorem: The plane travels 150 meters in the last 10 seconds before stopping -/
theorem last_ten_seconds_distance : 
  distance stop_time - distance (stop_time - 10) = 150 := by
  sorry

end NUMINAMATH_CALUDE_last_ten_seconds_distance_l879_87908


namespace NUMINAMATH_CALUDE_students_per_school_l879_87909

theorem students_per_school (total_schools : ℕ) (total_students : ℕ) 
  (h1 : total_schools = 25) (h2 : total_students = 6175) : 
  total_students / total_schools = 247 := by
  sorry

end NUMINAMATH_CALUDE_students_per_school_l879_87909


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l879_87954

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℝ) * (1 - 0.4) = 420 → initial_apples = 700 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l879_87954


namespace NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l879_87968

/-- The maximum y-coordinate of a point on the curve r = sin 3θ is 9/16 -/
theorem max_y_coordinate_sin_3theta : 
  let r : ℝ → ℝ := λ θ => Real.sin (3 * θ)
  let y : ℝ → ℝ := λ θ => r θ * Real.sin θ
  ∃ (θ_max : ℝ), ∀ (θ : ℝ), y θ ≤ y θ_max ∧ y θ_max = 9/16 :=
by
  sorry


end NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l879_87968


namespace NUMINAMATH_CALUDE_x_value_l879_87990

theorem x_value (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l879_87990


namespace NUMINAMATH_CALUDE_fraction_equality_l879_87901

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 4) 
  (h2 : r / t = 8 / 15) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l879_87901


namespace NUMINAMATH_CALUDE_students_in_cars_l879_87927

theorem students_in_cars (total_students : ℕ) (num_buses : ℕ) (students_per_bus : ℕ) :
  total_students = 396 →
  num_buses = 7 →
  students_per_bus = 56 →
  total_students - (num_buses * students_per_bus) = 4 :=
by sorry

end NUMINAMATH_CALUDE_students_in_cars_l879_87927


namespace NUMINAMATH_CALUDE_divisibility_of_difference_l879_87992

theorem divisibility_of_difference : 43^43 - 17^17 ≡ 0 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_difference_l879_87992


namespace NUMINAMATH_CALUDE_inequality_proof_l879_87996

theorem inequality_proof (x : ℝ) (h : x ≠ 1) :
  Real.sqrt (x^2 - 2*x + 2) ≥ -Real.sqrt 5 * x ↔ (-1 ≤ x ∧ x < 1) ∨ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l879_87996


namespace NUMINAMATH_CALUDE_additional_grazing_area_l879_87930

/-- The additional grassy ground area a calf can graze after increasing rope length -/
theorem additional_grazing_area (initial_length new_length obstacle_length obstacle_width : ℝ) 
  (h1 : initial_length = 12)
  (h2 : new_length = 18)
  (h3 : obstacle_length = 4)
  (h4 : obstacle_width = 3) :
  (π * new_length^2 - obstacle_length * obstacle_width) - π * initial_length^2 = 180 * π - 12 :=
by sorry

end NUMINAMATH_CALUDE_additional_grazing_area_l879_87930


namespace NUMINAMATH_CALUDE_existence_of_a_and_b_l879_87916

theorem existence_of_a_and_b : ∃ (a b : ℝ), a = b + 1 ∧ a^4 = b^4 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_and_b_l879_87916


namespace NUMINAMATH_CALUDE_passed_candidates_count_l879_87956

/-- Prove the number of passed candidates given total candidates and average marks -/
theorem passed_candidates_count
  (total_candidates : ℕ)
  (avg_all : ℚ)
  (avg_passed : ℚ)
  (avg_failed : ℚ)
  (h_total : total_candidates = 120)
  (h_avg_all : avg_all = 35)
  (h_avg_passed : avg_passed = 39)
  (h_avg_failed : avg_failed = 15) :
  ∃ (passed_candidates : ℕ), passed_candidates = 100 ∧
    passed_candidates ≤ total_candidates ∧
    (passed_candidates : ℚ) * avg_passed +
    (total_candidates - passed_candidates : ℚ) * avg_failed =
    (total_candidates : ℚ) * avg_all :=
by sorry

end NUMINAMATH_CALUDE_passed_candidates_count_l879_87956


namespace NUMINAMATH_CALUDE_tangent_line_implies_k_value_l879_87926

/-- Given a curve y = 3ln(x) + x + k, where k ∈ ℝ, if there exists a point P(x₀, y₀) on the curve
    such that the tangent line at P has the equation 4x - y - 1 = 0, then k = 2. -/
theorem tangent_line_implies_k_value (k : ℝ) (x₀ y₀ : ℝ) :
  y₀ = 3 * Real.log x₀ + x₀ + k →
  (∀ x y, y = 4 * x - 1 ↔ 4 * x - y - 1 = 0) →
  (∃ m b, ∀ x, 3 / x + 1 = m ∧ y₀ - m * x₀ = b ∧ y₀ = 4 * x₀ - 1) →
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_k_value_l879_87926


namespace NUMINAMATH_CALUDE_smallest_translation_l879_87907

open Real

theorem smallest_translation (φ : ℝ) : φ > 0 ∧ 
  (∀ x : ℝ, sin (2 * (x + φ)) = cos (2 * x - π / 3)) →
  φ = π / 12 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_translation_l879_87907


namespace NUMINAMATH_CALUDE_cylindrical_to_cartesian_l879_87943

/-- Given a point P in cylindrical coordinates (r, θ, z) = (√2, π/4, 1),
    prove that its Cartesian coordinates (x, y, z) are (1, 1, 1). -/
theorem cylindrical_to_cartesian :
  let r : ℝ := Real.sqrt 2
  let θ : ℝ := π / 4
  let z : ℝ := 1
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y, z) = (1, 1, 1) := by sorry

end NUMINAMATH_CALUDE_cylindrical_to_cartesian_l879_87943


namespace NUMINAMATH_CALUDE_count_divisible_by_five_l879_87934

/-- The set of available digits --/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A function to check if a three-digit number is valid (no leading zero) --/
def isValidNumber (n : Nat) : Bool :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 ≠ 0)

/-- A function to check if a number is formed from distinct digits in the given set --/
def isFromDistinctDigits (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

/-- The set of valid three-digit numbers formed from the given digits --/
def validNumbers : Finset Nat :=
  Finset.filter (fun n => isValidNumber n ∧ isFromDistinctDigits n) (Finset.range 1000)

/-- The theorem to be proved --/
theorem count_divisible_by_five :
  (validNumbers.filter (fun n => n % 5 = 0)).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_five_l879_87934


namespace NUMINAMATH_CALUDE_power_multiplication_l879_87977

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l879_87977


namespace NUMINAMATH_CALUDE_probability_consecutive_is_one_eighteenth_l879_87918

/-- A standard six-sided die -/
def Die : Type := Fin 6

/-- The set of all possible outcomes when rolling four dice -/
def AllOutcomes : Finset (Die × Die × Die × Die) := sorry

/-- A function to check if four numbers are consecutive -/
def AreConsecutive (a b c d : ℕ) : Prop := sorry

/-- The set of favorable outcomes (four consecutive numbers in any order) -/
def FavorableOutcomes : Finset (Die × Die × Die × Die) := sorry

/-- The probability of rolling four consecutive numbers in any order -/
def ProbabilityConsecutive : ℚ :=
  (FavorableOutcomes.card : ℚ) / (AllOutcomes.card : ℚ)

/-- The main theorem: the probability is 1/18 -/
theorem probability_consecutive_is_one_eighteenth :
  ProbabilityConsecutive = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_probability_consecutive_is_one_eighteenth_l879_87918


namespace NUMINAMATH_CALUDE_positive_integer_pairs_l879_87967

theorem positive_integer_pairs : 
  ∀ (a b : ℕ+), 
    (∃ (k : ℕ+), k * a = b^4 + 1) → 
    (∃ (l : ℕ+), l * b = a^4 + 1) → 
    (Int.floor (Real.sqrt a.val) = Int.floor (Real.sqrt b.val)) → 
    ((a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1)) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_pairs_l879_87967


namespace NUMINAMATH_CALUDE_mike_red_notebooks_l879_87947

/-- Represents the number of red notebooks Mike bought -/
def red_notebooks : ℕ := sorry

/-- Represents the number of blue notebooks Mike bought -/
def blue_notebooks : ℕ := sorry

/-- The total cost of all notebooks -/
def total_cost : ℕ := 37

/-- The total number of notebooks -/
def total_notebooks : ℕ := 12

/-- The cost of each red notebook -/
def red_cost : ℕ := 4

/-- The number of green notebooks -/
def green_notebooks : ℕ := 2

/-- The cost of each green notebook -/
def green_cost : ℕ := 2

/-- The cost of each blue notebook -/
def blue_cost : ℕ := 3

theorem mike_red_notebooks : 
  red_notebooks = 3 ∧
  red_notebooks + green_notebooks + blue_notebooks = total_notebooks ∧
  red_notebooks * red_cost + green_notebooks * green_cost + blue_notebooks * blue_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_mike_red_notebooks_l879_87947


namespace NUMINAMATH_CALUDE_share_division_l879_87931

theorem share_division (total : ℚ) (a b c : ℚ) : 
  total = 700 →
  a + b + c = total →
  a = b / 2 →
  b = c / 2 →
  c = 400 := by
sorry

end NUMINAMATH_CALUDE_share_division_l879_87931


namespace NUMINAMATH_CALUDE_solve_candy_store_problem_l879_87902

def candy_store_problem (initial_money : ℚ) (gum_packs : ℕ) (gum_price : ℚ) 
  (chocolate_bars : ℕ) (candy_canes : ℕ) (candy_cane_price : ℚ) (money_left : ℚ) : Prop :=
  ∃ (chocolate_bar_price : ℚ),
    initial_money = 
      gum_packs * gum_price + 
      chocolate_bars * chocolate_bar_price + 
      candy_canes * candy_cane_price + 
      money_left ∧
    chocolate_bar_price = 1

theorem solve_candy_store_problem :
  candy_store_problem 10 3 1 5 2 (1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_store_problem_l879_87902


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l879_87974

/-- Given an arithmetic sequence {aₙ}, prove that S₁₃ = 13 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence condition
  (∀ n : ℕ, S n = (n / 2) * (a 1 + a n)) →               -- sum formula
  a 3 + a 5 + 2 * a 10 = 4 →                             -- given condition
  S 13 = 13 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l879_87974


namespace NUMINAMATH_CALUDE_friend_distribution_l879_87989

theorem friend_distribution (F : ℕ) (h1 : F > 0) : 
  (100 / F : ℚ) - (100 / (F + 5) : ℚ) = 1 → F = 20 := by
  sorry

end NUMINAMATH_CALUDE_friend_distribution_l879_87989


namespace NUMINAMATH_CALUDE_expression_simplification_l879_87965

theorem expression_simplification 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) :
  (a^4 * b^4 + a^4 * c^4 + b^4 * c^4) / 
  ((a^2 - b*c)^2 * (b^2 - a*c)^2 * (c^2 - a*b)^2) = 
  1 / (a^2 - b*c)^2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l879_87965


namespace NUMINAMATH_CALUDE_equal_ratios_fraction_l879_87904

theorem equal_ratios_fraction (x y z : ℝ) (h : x/2 = y/3 ∧ y/3 = z/4) :
  (x + y) / (3*y - 2*z) = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_fraction_l879_87904


namespace NUMINAMATH_CALUDE_solution_ratio_l879_87905

theorem solution_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (eq1 : 8 * x - 6 * y = c) (eq2 : 12 * y - 18 * x = d) : c / d = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_ratio_l879_87905


namespace NUMINAMATH_CALUDE_F_negative_sufficient_not_necessary_l879_87900

/-- Represents a general equation of the form x^2 + y^2 + Dx + Ey + F = 0 -/
structure GeneralEquation where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Predicate to check if a GeneralEquation represents a circle -/
def is_circle (eq : GeneralEquation) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ 
    eq.D = -2 * h ∧ 
    eq.E = -2 * k ∧ 
    eq.F = h^2 + k^2 - r^2

/-- Theorem stating that F < 0 is a sufficient but not necessary condition for a circle -/
theorem F_negative_sufficient_not_necessary (eq : GeneralEquation) :
  (eq.F < 0 → is_circle eq) ∧ ¬(is_circle eq → eq.F < 0) :=
sorry

end NUMINAMATH_CALUDE_F_negative_sufficient_not_necessary_l879_87900


namespace NUMINAMATH_CALUDE_decrease_by_percentage_decrease_80_by_150_percent_l879_87914

theorem decrease_by_percentage (n : ℝ) (p : ℝ) : 
  n - (p / 100) * n = n * (1 - p / 100) := by sorry

theorem decrease_80_by_150_percent : 
  80 - (150 / 100) * 80 = -40 := by sorry

end NUMINAMATH_CALUDE_decrease_by_percentage_decrease_80_by_150_percent_l879_87914


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l879_87946

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}
def N : Set ℝ := {x | 2*x - x^2 ≥ 0}

-- Define the open interval (1, 2]
def open_closed_interval : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem intersection_equals_interval : M ∩ N = open_closed_interval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l879_87946


namespace NUMINAMATH_CALUDE_gift_contribution_max_l879_87950

/-- Given a group of people contributing money, calculates the maximum possible contribution by a single person. -/
def max_contribution (n : ℕ) (total : ℚ) (min_contribution : ℚ) : ℚ :=
  total - (n - 1 : ℚ) * min_contribution

/-- Theorem stating the maximum possible contribution in the given scenario. -/
theorem gift_contribution_max (n : ℕ) (total : ℚ) (min_contribution : ℚ)
  (h_n : n = 10)
  (h_total : total = 20)
  (h_min : min_contribution = 1)
  (h_positive : ∀ i, i ≤ n → min_contribution ≤ (max_contribution n total min_contribution)) :
  max_contribution n total min_contribution = 11 :=
by sorry

end NUMINAMATH_CALUDE_gift_contribution_max_l879_87950


namespace NUMINAMATH_CALUDE_angle5_measure_l879_87995

-- Define the angles
variable (angle1 angle2 angle5 : ℝ)

-- Define the conditions
axiom angle1_fraction : angle1 = (1/4) * angle2
axiom supplementary : angle2 + angle5 = 180

-- State the theorem
theorem angle5_measure : angle5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_angle5_measure_l879_87995


namespace NUMINAMATH_CALUDE_remainder_sum_l879_87997

theorem remainder_sum (a b : ℤ) 
  (ha : a % 60 = 53) 
  (hb : b % 45 = 22) : 
  (a + b) % 30 = 15 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l879_87997
