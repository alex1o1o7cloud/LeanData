import Mathlib

namespace NUMINAMATH_CALUDE_students_in_line_l2323_232360

/-- The number of students in a line, given specific positions of two students and the number of students between them. -/
theorem students_in_line
  (yoojung_position : ℕ)  -- Position of Yoojung
  (eunjung_position : ℕ)  -- Position of Eunjung from the back
  (students_between : ℕ)  -- Number of students between Yoojung and Eunjung
  (h1 : yoojung_position = 1)  -- Yoojung is at the front
  (h2 : eunjung_position = 5)  -- Eunjung is 5th from the back
  (h3 : students_between = 30)  -- 30 students between Yoojung and Eunjung
  : ℕ :=
by
  sorry

#check students_in_line

end NUMINAMATH_CALUDE_students_in_line_l2323_232360


namespace NUMINAMATH_CALUDE_max_value_sin_cos_max_value_achievable_l2323_232308

theorem max_value_sin_cos (θ : ℝ) : 
  (1/2) * Real.sin (3 * θ)^2 - (1/2) * Real.cos (2 * θ) ≤ 1 :=
sorry

theorem max_value_achievable : 
  ∃ θ : ℝ, (1/2) * Real.sin (3 * θ)^2 - (1/2) * Real.cos (2 * θ) = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_max_value_achievable_l2323_232308


namespace NUMINAMATH_CALUDE_eighth_power_sum_l2323_232354

theorem eighth_power_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^8 + b^8 = 47 := by
  sorry

end NUMINAMATH_CALUDE_eighth_power_sum_l2323_232354


namespace NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l2323_232345

theorem tagged_fish_in_second_catch 
  (initial_tagged : ℕ) 
  (second_catch : ℕ) 
  (total_fish : ℕ) 
  (h1 : initial_tagged = 80) 
  (h2 : second_catch = 80) 
  (h3 : total_fish = 3200) :
  ∃ (tagged_in_second : ℕ), 
    tagged_in_second = 2 ∧ 
    (tagged_in_second : ℚ) / second_catch = initial_tagged / total_fish :=
by
  sorry

end NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l2323_232345


namespace NUMINAMATH_CALUDE_not_equivalent_fraction_l2323_232321

theorem not_equivalent_fraction (h : 0.000000275 = 2.75 * 10^(-7)) : 
  (11/40) * 10^(-7) ≠ 2.75 * 10^(-7) := by
  sorry

end NUMINAMATH_CALUDE_not_equivalent_fraction_l2323_232321


namespace NUMINAMATH_CALUDE_remainder_not_always_same_l2323_232393

theorem remainder_not_always_same (a b : ℕ) :
  (3 * a + b) % 10 = (3 * b + a) % 10 →
  ¬(a % 10 = b % 10) :=
by sorry

end NUMINAMATH_CALUDE_remainder_not_always_same_l2323_232393


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_negative_two_l2323_232358

theorem sum_of_coefficients_equals_negative_two :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ),
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = 
    a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + 
    a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + 
    a₉*(x+2)^9 + a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_negative_two_l2323_232358


namespace NUMINAMATH_CALUDE_day_relationship_l2323_232376

/-- Represents days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  day : Nat

/-- Function to determine the weekday of a given YearDay -/
def weekday_of_yearday : YearDay → Weekday := sorry

/-- Theorem stating the relationship between the given days and their weekdays -/
theorem day_relationship (N : Int) :
  (weekday_of_yearday ⟨N, 250⟩ = Weekday.Wednesday) →
  (weekday_of_yearday ⟨N + 1, 150⟩ = Weekday.Wednesday) →
  (weekday_of_yearday ⟨N - 1, 50⟩ = Weekday.Saturday) :=
by sorry

end NUMINAMATH_CALUDE_day_relationship_l2323_232376


namespace NUMINAMATH_CALUDE_jelly_cost_l2323_232334

theorem jelly_cost (N B J : ℕ) (h1 : N = 15) 
  (h2 : 6 * B * N + 7 * J * N = 315) 
  (h3 : B > 0) (h4 : J > 0) : 
  7 * J * N / 100 = 315 / 100 := by
  sorry

end NUMINAMATH_CALUDE_jelly_cost_l2323_232334


namespace NUMINAMATH_CALUDE_inequality_range_l2323_232384

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (3 : ℝ)^(a*x - 1) < (1/3 : ℝ)^(a*x^2)) ↔ -4 < a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l2323_232384


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l2323_232361

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l2323_232361


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2323_232391

theorem inequality_solution_set (x : ℝ) : 
  (x^2 + 8*x < 20) ↔ (-10 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2323_232391


namespace NUMINAMATH_CALUDE_ellipse_properties_l2323_232394

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  f1 : ℝ × ℝ  -- Focus 1
  f2 : ℝ × ℝ  -- Focus 2
  p : ℝ × ℝ   -- Point on ellipse
  h1 : a > b
  h2 : b > 0
  h3 : (p.1^2 / a^2) + (p.2^2 / b^2) = 1  -- P is on the ellipse
  h4 : (p.1 - f1.1) * (p.1 - f2.1) + (p.2 - f1.2) * (p.2 - f2.2) = 0  -- PF₁ ⟂ PF₂
  h5 : (f1.1 - f2.1)^2 + (f1.2 - f2.2)^2 = 12  -- |F₁F₂| = 2√3
  h6 : abs ((p.1 - f1.1) * (p.2 - f2.2) - (p.2 - f1.2) * (p.1 - f2.1)) = 2  -- Area of triangle PF₁F₂ is 1

/-- The theorem to be proved -/
theorem ellipse_properties (e : Ellipse) :
  (e.a = 2 ∧ e.b = 1) ∧
  (∀ m : ℝ, ∃ A B : ℝ × ℝ,
    (A.1^2 / 4 + A.2^2 = 1) ∧
    (B.1^2 / 4 + B.2^2 = 1) ∧
    (A.2 + B.2 = A.1 + B.1 + 2*m) ↔
    -3 * Real.sqrt 5 / 5 < m ∧ m < 3 * Real.sqrt 5 / 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2323_232394


namespace NUMINAMATH_CALUDE_sum_of_17th_roots_minus_one_l2323_232316

theorem sum_of_17th_roots_minus_one (ω : ℂ) : 
  ω^17 = 1 → ω ≠ 1 → ω + ω^2 + ω^3 + ω^4 + ω^5 + ω^6 + ω^7 + ω^8 + ω^9 + ω^10 + ω^11 + ω^12 + ω^13 + ω^14 + ω^15 + ω^16 = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_17th_roots_minus_one_l2323_232316


namespace NUMINAMATH_CALUDE_square_tiles_problem_l2323_232396

theorem square_tiles_problem (n : ℕ) : 
  (4 * n - 4 = 52) → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_tiles_problem_l2323_232396


namespace NUMINAMATH_CALUDE_cookie_cutter_sides_l2323_232303

/-- The number of sides on a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides on a square -/
def square_sides : ℕ := 4

/-- The number of sides on a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of triangle-shaped cookie cutters -/
def num_triangles : ℕ := 6

/-- The number of square-shaped cookie cutters -/
def num_squares : ℕ := 4

/-- The number of hexagon-shaped cookie cutters -/
def num_hexagons : ℕ := 2

/-- The total number of sides on all cookie cutters -/
def total_sides : ℕ := num_triangles * triangle_sides + num_squares * square_sides + num_hexagons * hexagon_sides

theorem cookie_cutter_sides : total_sides = 46 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cutter_sides_l2323_232303


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2323_232300

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 1) * x^2 + 4 * x + 1 = 0 ∧ 
   (k - 1) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 5 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2323_232300


namespace NUMINAMATH_CALUDE_value_of_a_l2323_232344

theorem value_of_a (a c : ℝ) (h1 : c / a = 4) (h2 : a + c = 30) : a = 6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2323_232344


namespace NUMINAMATH_CALUDE_optimal_inequality_l2323_232364

theorem optimal_inequality (a b c d : ℝ) 
  (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) (hd : d ≥ -1) :
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3/4) * (a + b + c + d) ∧ 
  ∀ k > 3/4, ∃ x y z w : ℝ, x ≥ -1 ∧ y ≥ -1 ∧ z ≥ -1 ∧ w ≥ -1 ∧ 
    x^3 + y^3 + z^3 + w^3 + 1 < k * (x + y + z + w) :=
by sorry

end NUMINAMATH_CALUDE_optimal_inequality_l2323_232364


namespace NUMINAMATH_CALUDE_conditions_for_a_and_b_l2323_232373

/-- Given a system of equations, prove the conditions for a and b -/
theorem conditions_for_a_and_b (a b x y : ℝ) : 
  (x^2 + x*y + y^2 - y = 0) →
  (a * x^2 + b * x * y + x = 0) →
  ((a + 1)^2 = 4*(b + 1) ∧ b ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_conditions_for_a_and_b_l2323_232373


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2323_232366

theorem arithmetic_calculations :
  ((5 : ℤ) - (-10) + (-32) - 7 = -24) ∧
  ((1/4 + 1/6 - 1/2 : ℚ) * 12 + (-2)^3 / (-4) = 1) ∧
  ((3^2 : ℚ) + (-2-5) / 7 - |-(1/4)| * (-2)^4 + (-1)^2023 = 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2323_232366


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2323_232377

theorem polynomial_factorization (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3) * (8*x^2 + x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2323_232377


namespace NUMINAMATH_CALUDE_max_sum_same_color_as_center_l2323_232332

/-- Represents a 5x5 checkerboard grid with alternating colors -/
def Grid := Fin 5 → Fin 5 → Bool

/-- A valid numbering of the grid satisfies the adjacent consecutive property -/
def ValidNumbering (g : Grid) (n : Fin 5 → Fin 5 → Fin 25) : Prop := sorry

/-- The sum of numbers in squares of the same color as the center square -/
def SumSameColorAsCenter (g : Grid) (n : Fin 5 → Fin 5 → Fin 25) : ℕ := sorry

/-- The maximum sum of numbers in squares of the same color as the center square -/
def MaxSumSameColorAsCenter (g : Grid) : ℕ := sorry

theorem max_sum_same_color_as_center (g : Grid) :
  MaxSumSameColorAsCenter g = 169 := by sorry

end NUMINAMATH_CALUDE_max_sum_same_color_as_center_l2323_232332


namespace NUMINAMATH_CALUDE_binary_decimal_octal_conversion_l2323_232383

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : Nat) : List Nat :=
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_decimal_octal_conversion :
  binary_to_decimal binary_101101 = 45 ∧
  decimal_to_octal 45 = [5, 5] := by
  sorry

end NUMINAMATH_CALUDE_binary_decimal_octal_conversion_l2323_232383


namespace NUMINAMATH_CALUDE_area_triangle_AOB_l2323_232311

/-- Given a sector AOB with area 2π/3 and radius 2, the area of triangle AOB is √3. -/
theorem area_triangle_AOB (S : ℝ) (r : ℝ) (h1 : S = 2 * π / 3) (h2 : r = 2) :
  (1 / 2) * r^2 * Real.sin (S / r^2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_AOB_l2323_232311


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2323_232347

theorem quadratic_root_zero (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 6 * x + k^2 - k = 0) ∧
  ((k - 1) * 0^2 + 6 * 0 + k^2 - k = 0) →
  k = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2323_232347


namespace NUMINAMATH_CALUDE_tub_volume_ratio_l2323_232398

theorem tub_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (2/3 : ℝ) * V₂ →
  V₁ / V₂ = 8/9 := by
sorry

end NUMINAMATH_CALUDE_tub_volume_ratio_l2323_232398


namespace NUMINAMATH_CALUDE_angle_sum_equality_l2323_232355

-- Define the points in 2D space
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (1, 0)
def D : ℝ × ℝ := (2, 0)
def E : ℝ × ℝ := (3, 0)
def F : ℝ × ℝ := (3, 1)

-- Define the angles
def angle_FBE : ℝ := sorry
def angle_FCE : ℝ := sorry
def angle_FDE : ℝ := sorry

-- Theorem statement
theorem angle_sum_equality : angle_FBE + angle_FCE = angle_FDE := by sorry

end NUMINAMATH_CALUDE_angle_sum_equality_l2323_232355


namespace NUMINAMATH_CALUDE_number_order_l2323_232368

theorem number_order : 
  (1 * 4^3) < (8 * 9 + 5) ∧ (8 * 9 + 5) < (2 * 6^2 + 1 * 6 + 0) := by
  sorry

end NUMINAMATH_CALUDE_number_order_l2323_232368


namespace NUMINAMATH_CALUDE_S_equation_holds_iff_specific_pairs_l2323_232337

/-- Given real numbers x, y, z with x + y + z = 0, S_r is defined as x^r + y^r + z^r -/
def S (r : ℕ+) (x y z : ℝ) : ℝ := x^(r:ℕ) + y^(r:ℕ) + z^(r:ℕ)

/-- The theorem states that for positive integers m and n, 
    the equation S_{m+n}/(m+n) = (S_m/m) * (S_n/n) holds if and only if 
    (m, n) is one of the pairs (2, 3), (3, 2), (2, 5), or (5, 2) -/
theorem S_equation_holds_iff_specific_pairs (x y z : ℝ) (h : x + y + z = 0) :
  ∀ m n : ℕ+, 
    (S (m + n) x y z) / (m + n : ℝ) = (S m x y z) / (m : ℝ) * (S n x y z) / (n : ℝ) ↔ 
    ((m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 5) ∨ (m = 5 ∧ n = 2)) :=
by sorry

end NUMINAMATH_CALUDE_S_equation_holds_iff_specific_pairs_l2323_232337


namespace NUMINAMATH_CALUDE_route_down_length_for_given_conditions_l2323_232327

/-- Represents a hiking trip up and down a mountain -/
structure MountainHike where
  rate_up : ℝ
  time : ℝ
  rate_down_factor : ℝ

/-- Calculates the length of the route down the mountain -/
def route_down_length (hike : MountainHike) : ℝ :=
  hike.rate_up * hike.rate_down_factor * hike.time

/-- Theorem stating the length of the route down the mountain for the given conditions -/
theorem route_down_length_for_given_conditions :
  let hike : MountainHike := {
    rate_up := 3,
    time := 2,
    rate_down_factor := 1.5
  }
  route_down_length hike = 9 := by sorry

end NUMINAMATH_CALUDE_route_down_length_for_given_conditions_l2323_232327


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2323_232318

theorem complex_equation_solution (Z : ℂ) : Z = (2 - Z) * Complex.I → Z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2323_232318


namespace NUMINAMATH_CALUDE_direct_square_root_most_suitable_l2323_232395

/-- The quadratic equation to be solved -/
def quadratic_equation (x : ℝ) : Prop := (x - 1)^2 = 4

/-- Possible solution methods for quadratic equations -/
inductive SolutionMethod
  | CompletingSquare
  | QuadraticFormula
  | Factoring
  | DirectSquareRoot

/-- Predicate to determine if a method is the most suitable for solving a given equation -/
def is_most_suitable_method (eq : ℝ → Prop) (method : SolutionMethod) : Prop :=
  ∀ other_method : SolutionMethod, method = other_method ∨ 
    (∃ (complexity_measure : SolutionMethod → ℕ), 
      complexity_measure method < complexity_measure other_method)

/-- Theorem stating that the direct square root method is the most suitable for the given equation -/
theorem direct_square_root_most_suitable :
  is_most_suitable_method quadratic_equation SolutionMethod.DirectSquareRoot :=
sorry

end NUMINAMATH_CALUDE_direct_square_root_most_suitable_l2323_232395


namespace NUMINAMATH_CALUDE_tea_mixture_price_l2323_232323

/-- Given two types of tea with different prices per kg, calculate the price per kg of their mixture when mixed in equal quantities. -/
theorem tea_mixture_price (price_a price_b : ℚ) (h1 : price_a = 65) (h2 : price_b = 70) :
  (price_a + price_b) / 2 = 67.5 := by
  sorry

#check tea_mixture_price

end NUMINAMATH_CALUDE_tea_mixture_price_l2323_232323


namespace NUMINAMATH_CALUDE_function_inequality_implies_unique_a_l2323_232375

theorem function_inequality_implies_unique_a :
  ∀ (a : ℝ),
  (∀ (x : ℝ), Real.exp x + a * (x^2 - x) - Real.cos x ≥ 0) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_unique_a_l2323_232375


namespace NUMINAMATH_CALUDE_smallest_n_with_five_pairs_l2323_232341

/-- The function f(n) returns the number of distinct ordered pairs of positive integers (a, b) such that a² + b² = n -/
def f (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1 * p.1 + p.2 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 125 is the smallest positive integer n for which f(n) = 5 -/
theorem smallest_n_with_five_pairs : (∀ m : ℕ, m > 0 ∧ m < 125 → f m ≠ 5) ∧ f 125 = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_five_pairs_l2323_232341


namespace NUMINAMATH_CALUDE_factor_expression_l2323_232305

theorem factor_expression (x : ℝ) : 60 * x + 45 + 9 * x^2 = 3 * (3 * x + 5) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2323_232305


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l2323_232349

/-- Given two plane vectors a and b with an angle of 120° between them,
    |a| = 1, |b| = 2, and a vector m satisfying m · a = m · b = 1,
    prove that |m| = √21/3 -/
theorem vector_magnitude_problem (a b m : ℝ × ℝ) :
  (∃ θ : ℝ, θ = 2 * π / 3 ∧ a.1 * b.1 + a.2 * b.2 = ‖a‖ * ‖b‖ * Real.cos θ) →
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  m • a = 1 →
  m • b = 1 →
  ‖m‖ = Real.sqrt 21 / 3 := by
  sorry

#check vector_magnitude_problem

end NUMINAMATH_CALUDE_vector_magnitude_problem_l2323_232349


namespace NUMINAMATH_CALUDE_nabla_neg_five_neg_seven_l2323_232313

def nabla (a b : ℝ) : ℝ := a * b + a - b

theorem nabla_neg_five_neg_seven : nabla (-5) (-7) = 37 := by
  sorry

end NUMINAMATH_CALUDE_nabla_neg_five_neg_seven_l2323_232313


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2323_232302

theorem cubic_equation_solution (x : ℝ) : (x + 2)^3 = 64 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2323_232302


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2323_232342

theorem inequality_solution_set (m n : ℝ) (h : m > n) :
  {x : ℝ | (n - m) * x > 0} = {x : ℝ | x < 0} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2323_232342


namespace NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_39_seconds_l2323_232380

/-- The time taken for a train to pass a jogger -/
theorem train_passing_jogger (jogger_speed : ℝ) (train_speed : ℝ) 
  (train_length : ℝ) (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- Proof that the train passes the jogger in 39 seconds -/
theorem train_passes_jogger_in_39_seconds : 
  train_passing_jogger 9 45 120 270 = 39 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_39_seconds_l2323_232380


namespace NUMINAMATH_CALUDE_triangle_rotation_l2323_232324

theorem triangle_rotation (a₁ a₂ a₃ : ℝ) (h1 : 12 * a₁ = 360) (h2 : 6 * a₂ = 360) (h3 : a₁ + a₂ + a₃ = 180) :
  ∃ n : ℕ, n * a₃ ≥ 360 ∧ ∀ m : ℕ, m * a₃ ≥ 360 → n ≤ m ∧ n = 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_rotation_l2323_232324


namespace NUMINAMATH_CALUDE_no_root_intersection_l2323_232346

theorem no_root_intersection : ∀ x : ℝ,
  (∃ y : ℝ, y = Real.sqrt x ∧ y = Real.sqrt (x - 6) + 1) →
  x^2 - 5*x + 6 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_root_intersection_l2323_232346


namespace NUMINAMATH_CALUDE_emmett_jumping_jacks_l2323_232320

/-- The number of jumping jacks Emmett did -/
def jumping_jacks : ℕ := sorry

/-- The number of pushups Emmett did -/
def pushups : ℕ := 8

/-- The number of situps Emmett did -/
def situps : ℕ := 20

/-- The total number of exercises Emmett did -/
def total_exercises : ℕ := jumping_jacks + pushups + situps

/-- The percentage of exercises that were pushups -/
def pushup_percentage : ℚ := 1/5

theorem emmett_jumping_jacks : 
  jumping_jacks = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_emmett_jumping_jacks_l2323_232320


namespace NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l2323_232382

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ := sorry

/-- Theorem: The distance between the foci of the specific ellipse is 6√3 -/
theorem specific_ellipse_foci_distance :
  let e : ParallelAxisEllipse := ⟨(6, 0), (0, 3)⟩
  foci_distance e = 6 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l2323_232382


namespace NUMINAMATH_CALUDE_rubble_initial_money_l2323_232386

/-- The amount of money Rubble had initially -/
def initial_money : ℝ := 15

/-- The cost of a notebook -/
def notebook_cost : ℝ := 4

/-- The cost of a pen -/
def pen_cost : ℝ := 1.5

/-- The number of notebooks Rubble bought -/
def num_notebooks : ℕ := 2

/-- The number of pens Rubble bought -/
def num_pens : ℕ := 2

/-- The amount of money Rubble had left after the purchase -/
def money_left : ℝ := 4

theorem rubble_initial_money :
  initial_money = 
    (num_notebooks : ℝ) * notebook_cost + 
    (num_pens : ℝ) * pen_cost + 
    money_left :=
by
  sorry

end NUMINAMATH_CALUDE_rubble_initial_money_l2323_232386


namespace NUMINAMATH_CALUDE_smallest_number_l2323_232314

theorem smallest_number : ∀ (a b c d : ℝ), a = 0 ∧ b = -1 ∧ c = -Real.sqrt 2 ∧ d = 2 → 
  c ≤ a ∧ c ≤ b ∧ c ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2323_232314


namespace NUMINAMATH_CALUDE_equality_check_l2323_232381

theorem equality_check : 
  (3^2 ≠ 2^3) ∧ 
  ((-2)^3 = -2^3) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  (-(-2) ≠ -|-2|) :=
by sorry

end NUMINAMATH_CALUDE_equality_check_l2323_232381


namespace NUMINAMATH_CALUDE_max_right_angles_in_triangle_l2323_232392

theorem max_right_angles_in_triangle : ℕ :=
  -- Define the sum of angles in a triangle
  let sum_of_angles : ℝ := 180

  -- Define a right angle in degrees
  let right_angle : ℝ := 90

  -- Define the maximum number of right angles
  let max_right_angles : ℕ := 1

  -- Theorem statement
  max_right_angles

end NUMINAMATH_CALUDE_max_right_angles_in_triangle_l2323_232392


namespace NUMINAMATH_CALUDE_simplified_ratio_of_boys_to_girls_l2323_232333

def number_of_boys : ℕ := 12
def number_of_girls : ℕ := 18

theorem simplified_ratio_of_boys_to_girls :
  (number_of_boys : ℚ) / (number_of_girls : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_ratio_of_boys_to_girls_l2323_232333


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l2323_232385

/-- Given an arithmetic sequence with first three terms 2x - 3, 3x + 1, and 5x + k,
    prove that k = 5 - x makes these terms form an arithmetic sequence. -/
theorem arithmetic_sequence_condition (x k : ℝ) : 
  let a₁ := 2*x - 3
  let a₂ := 3*x + 1
  let a₃ := 5*x + k
  (a₂ - a₁ = a₃ - a₂) → k = 5 - x := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l2323_232385


namespace NUMINAMATH_CALUDE_map_distance_conversion_l2323_232353

/-- Given a map scale where 1 inch represents 500 meters, 
    this theorem proves that a line segment of 7.25 inches 
    on the map represents 3625 meters in reality. -/
theorem map_distance_conversion 
  (scale : ℝ) 
  (map_length : ℝ) 
  (h1 : scale = 500) 
  (h2 : map_length = 7.25) : 
  map_length * scale = 3625 := by
sorry

end NUMINAMATH_CALUDE_map_distance_conversion_l2323_232353


namespace NUMINAMATH_CALUDE_math_books_same_box_probability_l2323_232335

def total_textbooks : ℕ := 12
def math_textbooks : ℕ := 3
def box_capacities : List ℕ := [3, 4, 5]

def probability_all_math_in_same_box : ℚ :=
  3 / 44

theorem math_books_same_box_probability :
  probability_all_math_in_same_box = 3 / 44 :=
by sorry

end NUMINAMATH_CALUDE_math_books_same_box_probability_l2323_232335


namespace NUMINAMATH_CALUDE_six_by_six_grid_shaded_percentage_l2323_232339

/-- Represents a square grid --/
structure SquareGrid :=
  (side : ℕ)
  (total_squares : ℕ)
  (shaded_squares : ℕ)

/-- Calculates the percentage of shaded area in a square grid --/
def shaded_percentage (grid : SquareGrid) : ℚ :=
  (grid.shaded_squares : ℚ) / (grid.total_squares : ℚ)

theorem six_by_six_grid_shaded_percentage :
  let grid : SquareGrid := ⟨6, 36, 21⟩
  shaded_percentage grid = 7 / 12 := by
  sorry

#eval (7 : ℚ) / 12 * 100  -- To show the decimal representation

end NUMINAMATH_CALUDE_six_by_six_grid_shaded_percentage_l2323_232339


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2323_232359

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h1 : seq.S 2016 = 2016)
    (h2 : seq.S 2016 / 2016 - seq.S 16 / 16 = 2000) :
  seq.a 1 = -2014 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2323_232359


namespace NUMINAMATH_CALUDE_distance_on_number_line_l2323_232365

theorem distance_on_number_line : 
  let point_a : ℤ := -2006
  let point_b : ℤ := 17
  abs (point_b - point_a) = 2023 := by sorry

end NUMINAMATH_CALUDE_distance_on_number_line_l2323_232365


namespace NUMINAMATH_CALUDE_pascal_row10_sums_l2323_232352

/-- Represents a row in Pascal's Triangle -/
def PascalRow (n : ℕ) := Fin (n + 1) → ℕ

/-- The 10th row of Pascal's Triangle -/
def row10 : PascalRow 10 := sorry

/-- Sum of elements in a Pascal's Triangle row -/
def row_sum (n : ℕ) (row : PascalRow n) : ℕ := sorry

/-- Sum of squares of elements in a Pascal's Triangle row -/
def row_sum_of_squares (n : ℕ) (row : PascalRow n) : ℕ := sorry

theorem pascal_row10_sums :
  (row_sum 10 row10 = 2^10) ∧
  (row_sum_of_squares 10 row10 = 183756) := by sorry

end NUMINAMATH_CALUDE_pascal_row10_sums_l2323_232352


namespace NUMINAMATH_CALUDE_three_digit_ends_in_five_divisible_by_five_l2323_232387

/-- A three-digit positive integer -/
def ThreeDigitInt (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

theorem three_digit_ends_in_five_divisible_by_five :
  ∀ N : ℕ, ThreeDigitInt N → onesDigit N = 5 → 
    (∃ k : ℕ, N = 5 * k) := by sorry

end NUMINAMATH_CALUDE_three_digit_ends_in_five_divisible_by_five_l2323_232387


namespace NUMINAMATH_CALUDE_intersection_and_union_range_of_a_l2323_232312

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Define set C with parameter a
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem for part (1)
theorem intersection_and_union :
  (A ∩ B = {x | 3 ≤ x ∧ x < 6}) ∧
  ((Set.univ \ B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ 9 ≤ x}) := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (C a ⊆ B) ↔ (2 ≤ a ∧ a ≤ 8) := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_range_of_a_l2323_232312


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2323_232307

theorem polynomial_remainder (x : ℝ) : 
  (x^3 - 3*x + 5) % (x - 1) = 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2323_232307


namespace NUMINAMATH_CALUDE_x_divisibility_l2323_232363

def x : ℕ := 36^2 + 48^2 + 64^3 + 81^2

theorem x_divisibility :
  (∃ k : ℕ, x = 3 * k) ∧
  (∃ k : ℕ, x = 4 * k) ∧
  (∃ k : ℕ, x = 9 * k) ∧
  ¬(∃ k : ℕ, x = 16 * k) := by
  sorry

end NUMINAMATH_CALUDE_x_divisibility_l2323_232363


namespace NUMINAMATH_CALUDE_elsa_marbles_l2323_232338

/-- The number of marbles Elsa started with -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Elsa lost at breakfast -/
def lost_at_breakfast : ℕ := 3

/-- The number of marbles Elsa gave to Susie at lunch -/
def given_to_susie : ℕ := 5

/-- The number of new marbles Elsa's mom bought -/
def new_marbles : ℕ := 12

/-- The number of marbles Elsa had at the end of the day -/
def final_marbles : ℕ := 54

theorem elsa_marbles :
  initial_marbles = 40 :=
by
  have h1 : initial_marbles - lost_at_breakfast - given_to_susie + new_marbles + 2 * given_to_susie = final_marbles :=
    sorry
  sorry

end NUMINAMATH_CALUDE_elsa_marbles_l2323_232338


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2323_232343

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 690 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 6 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2323_232343


namespace NUMINAMATH_CALUDE_circle_equation_proof_l2323_232325

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line on which the center of the required circle lies
def centerLine (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

-- Define the equation of the required circle
def requiredCircle (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 13

-- Theorem statement
theorem circle_equation_proof :
  ∀ x y : ℝ,
  (circle1 x y ∧ circle2 x y) →
  ∃ h k : ℝ,
  centerLine h k ∧
  requiredCircle x y ∧
  (x - h)^2 + (y - k)^2 = (x + 1)^2 + (y - 1)^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l2323_232325


namespace NUMINAMATH_CALUDE_complex_number_location_l2323_232304

theorem complex_number_location :
  let z : ℂ := (1 : ℂ) / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l2323_232304


namespace NUMINAMATH_CALUDE_delta_y_value_l2323_232309

def f (x : ℝ) := x^2 + 1

theorem delta_y_value (x : ℝ) (Δx : ℝ) (h1 : x = 2) (h2 : Δx = 0.1) :
  f (x + Δx) - f x = 0.41 := by
  sorry

end NUMINAMATH_CALUDE_delta_y_value_l2323_232309


namespace NUMINAMATH_CALUDE_log_equation_solution_l2323_232388

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log 3 / Real.log x = 2 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2323_232388


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l2323_232330

/-- Given a quadratic function y = x^2 - 2x + 3, prove it can be expressed as y = (x + m)^2 + h
    where m = -1 and h = 2 -/
theorem quadratic_complete_square :
  ∃ (m h : ℝ), ∀ (x y : ℝ),
    y = x^2 - 2*x + 3 → y = (x + m)^2 + h ∧ m = -1 ∧ h = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l2323_232330


namespace NUMINAMATH_CALUDE_cone_lateral_surface_angle_l2323_232310

/-- The angle in the lateral surface unfolding of a cone, given that its lateral surface area is twice the area of its base. -/
theorem cone_lateral_surface_angle (r : ℝ) (h : r > 0) : 
  let l := 2 * r
  let base_area := π * r^2
  let lateral_area := π * r * l
  lateral_area = 2 * base_area →
  (lateral_area / (π * l^2)) * 360 = 180 :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_angle_l2323_232310


namespace NUMINAMATH_CALUDE_baseball_team_groups_l2323_232331

theorem baseball_team_groups (new_players returning_players players_per_group : ℕ) 
  (h1 : new_players = 48)
  (h2 : returning_players = 6)
  (h3 : players_per_group = 6) :
  (new_players + returning_players) / players_per_group = 9 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_groups_l2323_232331


namespace NUMINAMATH_CALUDE_circle_center_point_is_center_l2323_232340

/-- The center of a circle given by the equation x^2 + 4x + y^2 - 6y = 24 is (-2, 3) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + 4*x + y^2 - 6*y = 24) ↔ ((x + 2)^2 + (y - 3)^2 = 37) :=
by sorry

/-- The point (-2, 3) is the center of the circle -/
theorem point_is_center : 
  ∃! (a b : ℝ), ∀ (x y : ℝ), (x^2 + 4*x + y^2 - 6*y = 24) ↔ ((x - a)^2 + (y - b)^2 = 37) ∧ 
  a = -2 ∧ b = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_point_is_center_l2323_232340


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2323_232370

/-- A monotonic function on ℝ satisfying f(x) · f(y) = f(x + y) is of the form a^x for some a > 0 -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_mono : Monotone f) 
  (h_eq : ∀ x y : ℝ, f x * f y = f (x + y)) :
  ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f x = a ^ x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2323_232370


namespace NUMINAMATH_CALUDE_floor_neg_five_thirds_l2323_232329

theorem floor_neg_five_thirds : ⌊(-5/3 : ℚ)⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_five_thirds_l2323_232329


namespace NUMINAMATH_CALUDE_engineering_majors_consecutive_probability_l2323_232328

/-- The number of people sitting at the round table -/
def total_people : ℕ := 11

/-- The number of engineering majors -/
def engineering_majors : ℕ := 5

/-- The number of ways to arrange engineering majors consecutively after fixing one position -/
def consecutive_arrangements : ℕ := 7

/-- The number of ways to choose seats for engineering majors without restriction -/
def total_arrangements : ℕ := Nat.choose (total_people - 1) (engineering_majors - 1)

/-- The probability of engineering majors sitting consecutively -/
def probability : ℚ := consecutive_arrangements / total_arrangements

theorem engineering_majors_consecutive_probability :
  probability = 1 / 30 :=
sorry

end NUMINAMATH_CALUDE_engineering_majors_consecutive_probability_l2323_232328


namespace NUMINAMATH_CALUDE_smallest_T_for_162_l2323_232319

/-- Represents the removal process of tokens in a circle -/
def removeTokens (T : ℕ) : ℕ → ℕ
| 0 => T
| n + 1 => removeTokens (T / 2) n

/-- Checks if a given T results in 162 as the last token -/
def lastTokenIs162 (T : ℕ) : Prop :=
  removeTokens T (Nat.log2 T) = 162

/-- Theorem stating that 209 is the smallest T where the last token is 162 -/
theorem smallest_T_for_162 :
  lastTokenIs162 209 ∧ ∀ k < 209, ¬lastTokenIs162 k :=
sorry

end NUMINAMATH_CALUDE_smallest_T_for_162_l2323_232319


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_equivalence_l2323_232356

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem binomial_coefficient_divisibility_equivalence 
  (p : ℕ) (n : ℕ) 
  (h1 : is_prime p) 
  (h2 : is_prime (11 * 39 * p)) : 
  (∃ k : ℕ, k ≤ n ∧ p ∣ Nat.choose n k) ↔ 
  (∃ s q : ℕ, n = p^s * q - 1 ∧ s ≥ 0 ∧ 0 < q ∧ q < p) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_equivalence_l2323_232356


namespace NUMINAMATH_CALUDE_evaporation_problem_l2323_232389

theorem evaporation_problem (x : ℚ) : 
  (1 - x) * (1 - 1/4) = 1/6 → x = 7/9 := by
sorry

end NUMINAMATH_CALUDE_evaporation_problem_l2323_232389


namespace NUMINAMATH_CALUDE_limit_rational_function_l2323_232317

/-- The limit of (2x^2 - x - 1) / (x^3 + 2x^2 - x - 2) as x approaches 1 is 1/2 -/
theorem limit_rational_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 1| ∧ |x - 1| < δ → 
    |(2*x^2 - x - 1) / (x^3 + 2*x^2 - x - 2) - 1/2| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_rational_function_l2323_232317


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2323_232372

/-- Represents an isosceles triangle with perimeter 16 and one side length 6 -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  perimeter_eq : side1 + side2 + base = 16
  one_side_6 : side1 = 6 ∨ side2 = 6 ∨ base = 6
  isosceles : side1 = side2 ∨ side1 = base ∨ side2 = base

/-- The base of the isosceles triangle is either 4 or 6 -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle) : t.base = 4 ∨ t.base = 6 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2323_232372


namespace NUMINAMATH_CALUDE_each_student_gets_seven_squares_l2323_232371

/-- Calculates the number of chocolate squares each student receives -/
def chocolate_squares_per_student (gerald_bars : ℕ) (squares_per_bar : ℕ) (teacher_multiplier : ℕ) (num_students : ℕ) : ℕ :=
  let total_bars := gerald_bars + gerald_bars * teacher_multiplier
  let total_squares := total_bars * squares_per_bar
  total_squares / num_students

/-- Theorem stating that each student gets 7 squares of chocolate -/
theorem each_student_gets_seven_squares :
  chocolate_squares_per_student 7 8 2 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_each_student_gets_seven_squares_l2323_232371


namespace NUMINAMATH_CALUDE_largest_n_for_product_2304_l2323_232306

theorem largest_n_for_product_2304 :
  ∀ (d_a d_b : ℤ),
  ∃ (n : ℕ),
  (∀ k : ℕ, (1 + (k - 1) * d_a) * (3 + (k - 1) * d_b) = 2304 → k ≤ n) ∧
  (1 + (n - 1) * d_a) * (3 + (n - 1) * d_b) = 2304 ∧
  n = 20 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_product_2304_l2323_232306


namespace NUMINAMATH_CALUDE_exp_monotone_in_interval_l2323_232399

theorem exp_monotone_in_interval (a b : ℝ) (h : -1 < a ∧ a < b ∧ b < 1) : Real.exp a < Real.exp b := by
  sorry

end NUMINAMATH_CALUDE_exp_monotone_in_interval_l2323_232399


namespace NUMINAMATH_CALUDE_product_plus_one_is_square_l2323_232315

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) :
  ∃ n : ℕ, x * y + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_is_square_l2323_232315


namespace NUMINAMATH_CALUDE_equation_roots_l2323_232378

theorem equation_roots : ∀ (x : ℝ), x * (x - 3)^2 * (5 + x) = 0 ↔ x ∈ ({0, 3, -5} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_l2323_232378


namespace NUMINAMATH_CALUDE_carpet_cost_l2323_232322

/-- The cost of carpeting a room with given dimensions and carpet specifications. -/
theorem carpet_cost (room_length room_width carpet_width carpet_cost : ℝ) :
  room_length = 13 ∧
  room_width = 9 ∧
  carpet_width = 0.75 ∧
  carpet_cost = 12 →
  room_length * room_width * carpet_cost = 1404 := by
  sorry

end NUMINAMATH_CALUDE_carpet_cost_l2323_232322


namespace NUMINAMATH_CALUDE_amusement_park_visits_l2323_232362

theorem amusement_park_visits 
  (season_pass_cost : ℕ) 
  (cost_per_trip : ℕ) 
  (youngest_son_visits : ℕ) 
  (oldest_son_visits : ℕ) : 
  season_pass_cost = 100 → 
  cost_per_trip = 4 → 
  youngest_son_visits = 15 → 
  oldest_son_visits * cost_per_trip = season_pass_cost - (youngest_son_visits * cost_per_trip) → 
  oldest_son_visits = 10 := by
sorry

end NUMINAMATH_CALUDE_amusement_park_visits_l2323_232362


namespace NUMINAMATH_CALUDE_circle_equation_from_center_and_chord_l2323_232379

/-- The equation of a circle given its center and a chord. -/
theorem circle_equation_from_center_and_chord 
  (center_x center_y : ℝ) 
  (line1 : ℝ → ℝ → ℝ) (line2 : ℝ → ℝ → ℝ) (line3 : ℝ → ℝ → ℝ)
  (h1 : line1 center_x center_y = 0)
  (h2 : line2 center_x center_y = 0)
  (h3 : ∃ (A B : ℝ × ℝ), line3 A.1 A.2 = 0 ∧ line3 B.1 B.2 = 0)
  (h4 : ∀ (A B : ℝ × ℝ), line3 A.1 A.2 = 0 → line3 B.1 B.2 = 0 → 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36)
  (h5 : line1 x y = x - y - 1)
  (h6 : line2 x y = 2*x - y - 1)
  (h7 : line3 x y = 3*x + 4*y - 11) :
  ∀ (x y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_center_and_chord_l2323_232379


namespace NUMINAMATH_CALUDE_investment_sum_l2323_232351

/-- Given a sum invested at simple interest for two years, 
    if the difference in interest between 15% p.a. and 12% p.a. is 420, 
    then the sum invested is 7000. -/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 420) → P = 7000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l2323_232351


namespace NUMINAMATH_CALUDE_negation_equivalence_l2323_232357

theorem negation_equivalence (m : ℝ) :
  (¬ ∃ x < 0, x^2 + 2*x - m > 0) ↔ (∀ x < 0, x^2 + 2*x - m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2323_232357


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l2323_232348

theorem largest_prime_factor_of_1001 : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ 1001 ∧ ∀ (q : ℕ), q.Prime → q ∣ 1001 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l2323_232348


namespace NUMINAMATH_CALUDE_smallest_sum_of_primes_and_composites_l2323_232397

def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 0 → d < n → n % d ≠ 0

def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def formNumbers (digits : List ℕ) : List ℕ :=
  sorry

theorem smallest_sum_of_primes_and_composites (digits : List ℕ) :
  digits = [1, 2, 3, 4, 5, 6, 7, 8, 9] →
  (∃ nums : List ℕ,
    nums = formNumbers digits ∧
    (∀ n ∈ nums, isPrime n) ∧
    nums.sum = 318 ∧
    (∀ otherNums : List ℕ,
      otherNums = formNumbers digits →
      (∀ n ∈ otherNums, isPrime n) →
      otherNums.sum ≥ 318)) ∧
  (∃ nums : List ℕ,
    nums = formNumbers digits ∧
    (∀ n ∈ nums, isComposite n) ∧
    nums.sum = 127 ∧
    (∀ otherNums : List ℕ,
      otherNums = formNumbers digits →
      (∀ n ∈ otherNums, isComposite n) →
      otherNums.sum ≥ 127)) :=
by sorry


end NUMINAMATH_CALUDE_smallest_sum_of_primes_and_composites_l2323_232397


namespace NUMINAMATH_CALUDE_hillarys_money_after_deposit_l2323_232350

/-- The amount of money Hillary is left with after selling crafts and making a deposit -/
def hillarys_remaining_money (craft_price : ℕ) (crafts_sold : ℕ) (extra_money : ℕ) (deposit : ℕ) : ℕ :=
  craft_price * crafts_sold + extra_money - deposit

/-- Theorem stating that Hillary is left with 25 dollars after selling crafts and making a deposit -/
theorem hillarys_money_after_deposit :
  hillarys_remaining_money 12 3 7 18 = 25 := by
  sorry

end NUMINAMATH_CALUDE_hillarys_money_after_deposit_l2323_232350


namespace NUMINAMATH_CALUDE_quadrilateral_pyramid_ratio_l2323_232369

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a quadrilateral pyramid -/
structure QuadrilateralPyramid where
  P : Point3D
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Checks if two line segments are parallel -/
def areParallel (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Checks if two line segments are perpendicular -/
def arePerpendicular (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Checks if a line is perpendicular to a plane -/
def isPerpendicularToPlane (p1 p2 : Point3D) (plane : Plane3D) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Calculates the sine of the angle between a line and a plane -/
def sineAngleLinePlane (p1 p2 : Point3D) (plane : Plane3D) : ℝ := sorry

/-- Main theorem -/
theorem quadrilateral_pyramid_ratio 
  (pyramid : QuadrilateralPyramid) 
  (Q : Point3D)
  (h1 : areParallel pyramid.A pyramid.B pyramid.C pyramid.D)
  (h2 : arePerpendicular pyramid.A pyramid.B pyramid.A pyramid.D)
  (h3 : distance pyramid.A pyramid.B = 4)
  (h4 : distance pyramid.A pyramid.D = 2 * Real.sqrt 2)
  (h5 : distance pyramid.C pyramid.D = 2)
  (h6 : isPerpendicularToPlane pyramid.P pyramid.A (Plane3D.mk 0 0 1 0))  -- Assuming ABCD is on the xy-plane
  (h7 : distance pyramid.P pyramid.A = 4)
  (h8 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q.x = pyramid.P.x + t * (pyramid.B.x - pyramid.P.x) ∧
                              Q.y = pyramid.P.y + t * (pyramid.B.y - pyramid.P.y) ∧
                              Q.z = pyramid.P.z + t * (pyramid.B.z - pyramid.P.z))
  (h9 : sineAngleLinePlane Q pyramid.C (Plane3D.mk 1 0 0 0) = Real.sqrt 3 / 3)  -- Assuming PAC is on the yz-plane
  : ∃ (t : ℝ), distance pyramid.P Q / distance pyramid.P pyramid.B = 7/12 ∧ 
               Q.x = pyramid.P.x + t * (pyramid.B.x - pyramid.P.x) ∧
               Q.y = pyramid.P.y + t * (pyramid.B.y - pyramid.P.y) ∧
               Q.z = pyramid.P.z + t * (pyramid.B.z - pyramid.P.z) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_pyramid_ratio_l2323_232369


namespace NUMINAMATH_CALUDE_second_person_speed_l2323_232326

/-- Given two persons starting at the same point, walking in opposite directions
    for 3.5 hours, with one person walking at 6 km/hr, and ending up 45.5 km apart,
    the speed of the second person is 7 km/hr. -/
theorem second_person_speed (person1_speed : ℝ) (person2_speed : ℝ) (time : ℝ) (distance : ℝ) :
  person1_speed = 6 →
  time = 3.5 →
  distance = 45.5 →
  distance = (person1_speed + person2_speed) * time →
  person2_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_person_speed_l2323_232326


namespace NUMINAMATH_CALUDE_least_repeating_digits_eight_elevenths_l2323_232336

/-- The least number of digits in a repeating block of the decimal expansion of 8/11 is 2. -/
theorem least_repeating_digits_eight_elevenths : ∃ (n : ℕ), n = 2 ∧ 
  (∀ (m : ℕ), m < n → ¬ (∃ (k : ℕ+), 8 * (10^m - 1) = 11 * k)) ∧
  (∃ (k : ℕ+), 8 * (10^n - 1) = 11 * k) := by
  sorry

end NUMINAMATH_CALUDE_least_repeating_digits_eight_elevenths_l2323_232336


namespace NUMINAMATH_CALUDE_space_division_cube_tetrahedron_l2323_232301

/-- The number of parts into which the space is divided by the facets of a polyhedron -/
def num_parts (V F E : ℕ) : ℕ := 1 + V + F + E

/-- Properties of a cube -/
def cube_vertices : ℕ := 8
def cube_edges : ℕ := 12
def cube_faces : ℕ := 6

/-- Properties of a tetrahedron -/
def tetrahedron_vertices : ℕ := 4
def tetrahedron_edges : ℕ := 6
def tetrahedron_faces : ℕ := 4

theorem space_division_cube_tetrahedron :
  (num_parts cube_vertices cube_faces cube_edges = 27) ∧
  (num_parts tetrahedron_vertices tetrahedron_faces tetrahedron_edges = 15) :=
by sorry

end NUMINAMATH_CALUDE_space_division_cube_tetrahedron_l2323_232301


namespace NUMINAMATH_CALUDE_equation_roots_imply_a_range_l2323_232374

theorem equation_roots_imply_a_range :
  ∀ a : ℝ, (∃ x : ℝ, (2 - 2^(-|x - 3|))^2 = 3 + a) → -2 ≤ a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_imply_a_range_l2323_232374


namespace NUMINAMATH_CALUDE_number_division_problem_l2323_232390

theorem number_division_problem (n : ℕ) : 
  (n / (555 + 445) = 2 * (555 - 445)) ∧ 
  (n % (555 + 445) = 40) → 
  n = 220040 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l2323_232390


namespace NUMINAMATH_CALUDE_fish_catching_ratio_l2323_232367

/-- The number of fish Blaine caught -/
def blaine_fish : ℕ := 5

/-- The total number of fish caught by Keith and Blaine -/
def total_fish : ℕ := 15

/-- The number of fish Keith caught -/
def keith_fish : ℕ := total_fish - blaine_fish

/-- The ratio of fish Keith caught to fish Blaine caught -/
def fish_ratio : ℚ := keith_fish / blaine_fish

theorem fish_catching_ratio :
  fish_ratio = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_fish_catching_ratio_l2323_232367
