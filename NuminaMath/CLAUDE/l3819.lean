import Mathlib

namespace NUMINAMATH_CALUDE_bijection_property_l3819_381927

theorem bijection_property (k : ℕ) (f : ℤ → ℤ) 
  (h_bij : Function.Bijective f)
  (h_prop : ∀ i j : ℤ, |i - j| ≤ k → |f i - f j| ≤ k) :
  ∀ i j : ℤ, |f i - f j| = |i - j| := by
  sorry

end NUMINAMATH_CALUDE_bijection_property_l3819_381927


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3819_381996

/-- The axis of symmetry of a parabola y = a(x+1)(x-3) where a ≠ 0 -/
def axisOfSymmetry (a : ℝ) (h : a ≠ 0) : ℝ := 1

/-- Theorem stating that the axis of symmetry of the parabola y = a(x+1)(x-3) where a ≠ 0 is x = 1 -/
theorem parabola_axis_of_symmetry (a : ℝ) (h : a ≠ 0) :
  axisOfSymmetry a h = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l3819_381996


namespace NUMINAMATH_CALUDE_committee_formation_count_l3819_381989

/-- The number of ways to form a committee of size k from n eligible members. -/
def committee_count (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of members in the club. -/
def total_members : ℕ := 12

/-- The size of the committee to be formed. -/
def committee_size : ℕ := 5

/-- The number of ineligible members (Casey). -/
def ineligible_members : ℕ := 1

/-- The number of eligible members for the committee. -/
def eligible_members : ℕ := total_members - ineligible_members

theorem committee_formation_count :
  committee_count eligible_members committee_size = 462 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l3819_381989


namespace NUMINAMATH_CALUDE_wildcats_score_is_36_l3819_381961

/-- The score of the Panthers -/
def panthers_score : ℕ := 17

/-- The difference between the Wildcats' and Panthers' scores -/
def score_difference : ℕ := 19

/-- The score of the Wildcats -/
def wildcats_score : ℕ := panthers_score + score_difference

theorem wildcats_score_is_36 : wildcats_score = 36 := by
  sorry

end NUMINAMATH_CALUDE_wildcats_score_is_36_l3819_381961


namespace NUMINAMATH_CALUDE_fifth_root_inequality_l3819_381909

theorem fifth_root_inequality (x y : ℝ) : x < y → x^(1/5) > y^(1/5) := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_inequality_l3819_381909


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a1_l3819_381948

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a1 (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 9)
  (h_a3_a2 : 2 * a 3 - a 2 = 6) :
  a 1 = -3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a1_l3819_381948


namespace NUMINAMATH_CALUDE_equation_coefficients_l3819_381980

/-- Given a quadratic equation of the form ax^2 + bx + c = 0,
    this function returns a triple (a, b, c) of the coefficients -/
def quadratic_coefficients (f : ℝ → ℝ) : ℝ × ℝ × ℝ := sorry

theorem equation_coefficients :
  let f : ℝ → ℝ := λ x => -x^2 + 3*x - 1
  quadratic_coefficients f = (-1, 3, -1) := by sorry

end NUMINAMATH_CALUDE_equation_coefficients_l3819_381980


namespace NUMINAMATH_CALUDE_system_solution_l3819_381920

theorem system_solution :
  let x : ℚ := -51/61
  let y : ℚ := 378/61
  let z : ℚ := 728/61
  (4*x - 3*y + z = -10) ∧
  (3*x + 5*y - 2*z = 8) ∧
  (x - 2*y + 7*z = 5) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3819_381920


namespace NUMINAMATH_CALUDE_scaling_circle_not_hyperbola_l3819_381932

-- Define a circle
def Circle := Set (ℝ × ℝ)

-- Define a scaling transformation
def ScalingTransformation := (ℝ × ℝ) → (ℝ × ℝ)

-- Define a hyperbola
def Hyperbola := Set (ℝ × ℝ)

-- Theorem statement
theorem scaling_circle_not_hyperbola (c : Circle) (s : ScalingTransformation) :
  ∀ h : Hyperbola, (s '' c) ≠ h :=
sorry

end NUMINAMATH_CALUDE_scaling_circle_not_hyperbola_l3819_381932


namespace NUMINAMATH_CALUDE_final_sign_is_minus_l3819_381933

/-- Represents the state of the board with plus and minus signs -/
structure BoardState where
  plus_count : Nat
  minus_count : Nat

/-- Represents an operation on the board -/
inductive Operation
  | same_sign
  | different_sign

/-- Applies an operation to the board state -/
def apply_operation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.same_sign => 
      if state.plus_count ≥ 2 then 
        { plus_count := state.plus_count - 1, minus_count := state.minus_count }
      else 
        { plus_count := state.plus_count + 1, minus_count := state.minus_count - 2 }
  | Operation.different_sign => 
      { plus_count := state.plus_count - 1, minus_count := state.minus_count }

/-- Theorem: After 24 operations, the final sign is a minus sign -/
theorem final_sign_is_minus (initial_state : BoardState) 
    (h_initial : initial_state.plus_count = 10 ∧ initial_state.minus_count = 15) 
    (operations : List Operation) 
    (h_operations : operations.length = 24) : 
    (operations.foldl apply_operation initial_state).plus_count = 0 ∧ 
    (operations.foldl apply_operation initial_state).minus_count = 1 := by
  sorry

end NUMINAMATH_CALUDE_final_sign_is_minus_l3819_381933


namespace NUMINAMATH_CALUDE_trajectory_equation_l3819_381974

-- Define the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- Define the distance difference condition
def distance_difference : ℝ := 4

-- Define the trajectory of point C
def trajectory_of_C (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1 ∧ x ≥ 2

-- State the theorem
theorem trajectory_equation :
  ∀ (C : ℝ × ℝ),
    (Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) -
     Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = distance_difference) →
    trajectory_of_C C.1 C.2 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l3819_381974


namespace NUMINAMATH_CALUDE_water_jars_problem_l3819_381956

theorem water_jars_problem (total_volume : ℚ) (x : ℕ) : 
  total_volume = 42 →
  (x : ℚ) * (1/4 + 1/2 + 1) = total_volume →
  3 * x = 72 :=
by sorry

end NUMINAMATH_CALUDE_water_jars_problem_l3819_381956


namespace NUMINAMATH_CALUDE_original_recipe_butter_l3819_381994

/-- Represents a bread recipe with butter and flour quantities -/
structure BreadRecipe where
  butter : ℝ  -- Amount of butter in ounces
  flour : ℝ   -- Amount of flour in cups

/-- The original bread recipe -/
def original_recipe : BreadRecipe := { butter := 0, flour := 5 }

/-- The scaled up recipe -/
def scaled_recipe : BreadRecipe := { butter := 12, flour := 20 }

/-- The scale factor between the original and scaled recipe -/
def scale_factor : ℝ := 4

theorem original_recipe_butter :
  original_recipe.butter = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_original_recipe_butter_l3819_381994


namespace NUMINAMATH_CALUDE_h_more_efficient_l3819_381997

/-- The daily harvest rate of a K combine in hectares -/
def k_rate : ℝ := sorry

/-- The daily harvest rate of an H combine in hectares -/
def h_rate : ℝ := sorry

/-- The total harvest of 4 K combines and 3 H combines in 5 days -/
def harvest1 : ℝ := 5 * (4 * k_rate + 3 * h_rate)

/-- The total harvest of 3 K combines and 5 H combines in 4 days -/
def harvest2 : ℝ := 4 * (3 * k_rate + 5 * h_rate)

/-- The theorem stating that H combines harvest more per day than K combines -/
theorem h_more_efficient : harvest1 = harvest2 → h_rate > k_rate := by
  sorry

end NUMINAMATH_CALUDE_h_more_efficient_l3819_381997


namespace NUMINAMATH_CALUDE_trapezoid_area_l3819_381940

theorem trapezoid_area (outer_area inner_area : ℝ) (h1 : outer_area = 36) (h2 : inner_area = 4) :
  let total_trapezoid_area := outer_area - inner_area
  let num_trapezoids := 4
  let single_trapezoid_area := total_trapezoid_area / num_trapezoids
  single_trapezoid_area = 8 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3819_381940


namespace NUMINAMATH_CALUDE_three_m_minus_n_l3819_381999

theorem three_m_minus_n (m n : ℝ) (h : m + 1 = (n - 2) / 3) : 3 * m - n = -5 := by
  sorry

end NUMINAMATH_CALUDE_three_m_minus_n_l3819_381999


namespace NUMINAMATH_CALUDE_absolute_value_four_l3819_381936

theorem absolute_value_four (x : ℝ) : |x| = 4 → x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_four_l3819_381936


namespace NUMINAMATH_CALUDE_logarithm_equality_implies_golden_ratio_l3819_381937

theorem logarithm_equality_implies_golden_ratio (p q : ℝ) 
  (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 9 = Real.log q / Real.log 12 ∧ 
       Real.log p / Real.log 9 = Real.log (p + q) / Real.log 16) : 
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equality_implies_golden_ratio_l3819_381937


namespace NUMINAMATH_CALUDE_system_solution_l3819_381935

theorem system_solution (x y : ℝ) : 
  (2 * x + y = 5) ∧ (x - 3 * y = 6) ↔ (x = 3 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3819_381935


namespace NUMINAMATH_CALUDE_smallest_angle_in_special_right_triangle_l3819_381972

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The statement of the problem -/
theorem smallest_angle_in_special_right_triangle :
  ∀ a b : ℕ,
    a + b = 90 →
    a > b →
    isPrime a →
    isPrime b →
    isPrime (a - b) →
    b ≥ 17 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_special_right_triangle_l3819_381972


namespace NUMINAMATH_CALUDE_quadratic_binomial_square_l3819_381943

theorem quadratic_binomial_square (a b : ℝ) : 
  (∃ c d : ℝ, ∀ x : ℝ, 6 * x^2 + 18 * x + a = (c * x + d)^2) ∧
  (∃ c d : ℝ, ∀ x : ℝ, 3 * x^2 + b * x + 4 = (c * x + d)^2) →
  a = 13.5 ∧ b = 18 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_binomial_square_l3819_381943


namespace NUMINAMATH_CALUDE_min_perimeter_nine_square_rectangle_l3819_381923

/-- Represents a rectangle divided into nine squares with integer side lengths -/
structure NineSquareRectangle where
  a : ℕ
  b : ℕ

/-- The perimeter of a NineSquareRectangle -/
def perimeter (r : NineSquareRectangle) : ℕ :=
  2 * ((3 * r.a + 8 * r.a) + (2 * r.a + 12 * r.a))

/-- Theorem stating the minimum perimeter of a NineSquareRectangle is 52 -/
theorem min_perimeter_nine_square_rectangle :
  ∃ (r : NineSquareRectangle), perimeter r = 52 ∧ ∀ (s : NineSquareRectangle), perimeter s ≥ 52 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_nine_square_rectangle_l3819_381923


namespace NUMINAMATH_CALUDE_solve_for_q_l3819_381967

theorem solve_for_q (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_q_l3819_381967


namespace NUMINAMATH_CALUDE_power_calculation_l3819_381938

theorem power_calculation : 2^24 / 16^3 * 2^4 = 65536 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l3819_381938


namespace NUMINAMATH_CALUDE_square_sum_nonzero_implies_nonzero_element_l3819_381953

theorem square_sum_nonzero_implies_nonzero_element (a b : ℝ) : 
  a^2 + b^2 ≠ 0 → (a ≠ 0 ∨ b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_nonzero_implies_nonzero_element_l3819_381953


namespace NUMINAMATH_CALUDE_smallest_class_size_l3819_381941

/-- Represents the number of students in a physical education class with the given arrangement. -/
def class_size (n : ℕ) : ℕ := 5 * n + 2

/-- Proves that the smallest possible class size satisfying the given conditions is 42 students. -/
theorem smallest_class_size :
  ∃ (n : ℕ), 
    (class_size n > 40) ∧ 
    (∀ m : ℕ, class_size m > 40 → m ≥ class_size n) ∧
    (class_size n = 42) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_l3819_381941


namespace NUMINAMATH_CALUDE_acute_triangle_perimeter_bound_l3819_381908

/-- Given an acute-angled triangle with circumradius R and perimeter P, prove that P ≥ 4R. -/
theorem acute_triangle_perimeter_bound (R : ℝ) (P : ℝ) (α β γ : ℝ) :
  R > 0 →  -- R is positive (implied by being a radius)
  P > 0 →  -- P is positive (implied by being a perimeter)
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  0 < γ ∧ γ < π/2 →  -- γ is acute
  α + β + γ = π →  -- sum of angles in a triangle
  P = 2 * R * (Real.sin α + Real.sin β + Real.sin γ) →  -- perimeter formula using sine rule
  P ≥ 4 * R :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_perimeter_bound_l3819_381908


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l3819_381951

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p ∣ (36^2 + 49^2) ∧ 
    ∀ (q : ℕ), Nat.Prime q → q ∣ (36^2 + 49^2) → q ≤ p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l3819_381951


namespace NUMINAMATH_CALUDE_apple_problem_l3819_381971

theorem apple_problem (x : ℚ) : 
  (((x / 2 + 10) * 2 / 3 + 2) / 2 + 1 = 12) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_apple_problem_l3819_381971


namespace NUMINAMATH_CALUDE_indefinite_integral_equality_l3819_381990

/-- The derivative of -(8/9) · √((1 + ∜(x³)) / ∜(x³))³ with respect to x
    is equal to (√(1 + ∜(x³))) / (x² · ⁸√x) for x > 0 -/
theorem indefinite_integral_equality (x : ℝ) (h : x > 0) :
  deriv (fun x => -(8/9) * Real.sqrt ((1 + x^(1/4)) / x^(1/4))^3) x =
  (Real.sqrt (1 + x^(3/4))) / (x^2 * x^(1/8)) :=
sorry

end NUMINAMATH_CALUDE_indefinite_integral_equality_l3819_381990


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3819_381915

theorem min_value_quadratic (a b : ℝ) :
  2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a - 4 * b + 2044 ≥ 1976 ∧
  (2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a - 4 * b + 2044 = 1976 ↔ a = 8 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3819_381915


namespace NUMINAMATH_CALUDE_lunch_cost_is_24_l3819_381966

/-- The cost of the Taco Grande Plate -/
def taco_grande_cost : ℕ := sorry

/-- The cost of Mike's additional items -/
def mike_additional_cost : ℕ := 2 + 4 + 2

/-- John's bill is equal to the cost of the Taco Grande Plate -/
def johns_bill : ℕ := taco_grande_cost

/-- Mike's bill is equal to the cost of the Taco Grande Plate plus the additional items -/
def mikes_bill : ℕ := taco_grande_cost + mike_additional_cost

/-- Mike's bill is twice as large as John's bill -/
axiom mikes_bill_twice_johns : mikes_bill = 2 * johns_bill

/-- The combined total cost of Mike and John's lunch -/
def total_cost : ℕ := johns_bill + mikes_bill

theorem lunch_cost_is_24 : total_cost = 24 := by sorry

end NUMINAMATH_CALUDE_lunch_cost_is_24_l3819_381966


namespace NUMINAMATH_CALUDE_negative_integer_equation_solution_l3819_381929

theorem negative_integer_equation_solution :
  ∃! (N : ℤ), N < 0 ∧ N + 2 * N^2 = 12 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_negative_integer_equation_solution_l3819_381929


namespace NUMINAMATH_CALUDE_bmw_length_l3819_381922

theorem bmw_length : 
  let straight_segments : ℕ := 7
  let straight_length : ℝ := 2
  let diagonal_segments : ℕ := 2
  let diagonal_length : ℝ := Real.sqrt 2
  straight_segments * straight_length + diagonal_segments * diagonal_length = 14 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_bmw_length_l3819_381922


namespace NUMINAMATH_CALUDE_circle_center_l3819_381928

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Theorem statement
theorem circle_center :
  ∃ (c : ℝ × ℝ), (c.1 = 1 ∧ c.2 = 0) ∧
  ∀ (x y : ℝ), circle_equation x y ↔ (x - c.1)^2 + (y - c.2)^2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3819_381928


namespace NUMINAMATH_CALUDE_completing_square_result_l3819_381910

theorem completing_square_result (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) → ((x - 3)^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l3819_381910


namespace NUMINAMATH_CALUDE_multiple_properties_l3819_381930

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 5 * k) 
  (hb : ∃ m : ℤ, b = 10 * m) : 
  (∃ n : ℤ, b = 5 * n) ∧ 
  (∃ p : ℤ, a + b = 5 * p) ∧ 
  (∃ q : ℤ, a + b = 2 * q) := by
  sorry

end NUMINAMATH_CALUDE_multiple_properties_l3819_381930


namespace NUMINAMATH_CALUDE_lcm_hcf_relation_l3819_381977

theorem lcm_hcf_relation (x y : ℕ+) (h_lcm : Nat.lcm x y = 1637970) (h_hcf : Nat.gcd x y = 210) (h_x : x = 10780) : y = 31910 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_relation_l3819_381977


namespace NUMINAMATH_CALUDE_complement_of_union_M_N_l3819_381969

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem complement_of_union_M_N : 
  (M ∪ N)ᶜ = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_M_N_l3819_381969


namespace NUMINAMATH_CALUDE_sequence_formula_l3819_381950

def geometric_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ k, k ≥ 1 → a (k + 1) - a k = (a 2 - a 1) * (2 ^ (k - 1))

theorem sequence_formula (a : ℕ → ℝ) (n : ℕ) :
  a 1 = 1 →
  geometric_sequence a n →
  a 2 - a 1 = 2 →
  ∀ k, k ≥ 1 → a k = 2^k - 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_l3819_381950


namespace NUMINAMATH_CALUDE_garrison_size_l3819_381931

/-- The initial number of days the provisions would last -/
def initial_days : ℕ := 28

/-- The number of days that passed before reinforcements arrived -/
def days_before_reinforcement : ℕ := 12

/-- The number of men that arrived as reinforcement -/
def reinforcement : ℕ := 1110

/-- The number of days the provisions would last after reinforcement arrived -/
def remaining_days : ℕ := 10

/-- The initial number of men in the garrison -/
def initial_men : ℕ := 1850

theorem garrison_size :
  ∃ (M : ℕ),
    M * initial_days = 
    (M + reinforcement) * remaining_days + 
    M * days_before_reinforcement ∧
    M = initial_men :=
by sorry

end NUMINAMATH_CALUDE_garrison_size_l3819_381931


namespace NUMINAMATH_CALUDE_square_difference_factorization_l3819_381916

theorem square_difference_factorization (x y : ℝ) : 
  49 * x^2 - 36 * y^2 = (-6*y + 7*x) * (6*y + 7*x) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_factorization_l3819_381916


namespace NUMINAMATH_CALUDE_solve_for_x_l3819_381904

theorem solve_for_x (x y : ℚ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3819_381904


namespace NUMINAMATH_CALUDE_sales_at_540_l3819_381959

/-- Represents the sales model for a product -/
structure SalesModel where
  originalPrice : ℕ
  initialSales : ℕ
  reductionStep : ℕ
  salesIncreasePerStep : ℕ

/-- Calculates the sales volume given a price reduction -/
def salesVolume (model : SalesModel) (priceReduction : ℕ) : ℕ :=
  model.initialSales + (priceReduction / model.reductionStep) * model.salesIncreasePerStep

/-- Theorem stating the sales volume at a specific price point -/
theorem sales_at_540 (model : SalesModel) 
  (h1 : model.originalPrice = 600)
  (h2 : model.initialSales = 750)
  (h3 : model.reductionStep = 5)
  (h4 : model.salesIncreasePerStep = 30) :
  salesVolume model 60 = 1110 := by
  sorry

#eval salesVolume { originalPrice := 600, initialSales := 750, reductionStep := 5, salesIncreasePerStep := 30 } 60

end NUMINAMATH_CALUDE_sales_at_540_l3819_381959


namespace NUMINAMATH_CALUDE_mrs_hilt_shopping_l3819_381903

/-- Mrs. Hilt's shopping problem -/
theorem mrs_hilt_shopping (pencil_cost candy_cost remaining_money : ℕ) 
  (h1 : pencil_cost = 20)
  (h2 : candy_cost = 5)
  (h3 : remaining_money = 18) :
  pencil_cost + candy_cost + remaining_money = 43 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_shopping_l3819_381903


namespace NUMINAMATH_CALUDE_shoeing_time_for_48_blacksmiths_60_horses_l3819_381975

/-- The minimum time required for a group of blacksmiths to shoe a group of horses -/
def minimum_shoeing_time (num_blacksmiths : ℕ) (num_horses : ℕ) (time_per_horseshoe : ℕ) : ℕ :=
  let total_horseshoes := num_horses * 4
  let total_time := total_horseshoes * time_per_horseshoe
  total_time / num_blacksmiths

theorem shoeing_time_for_48_blacksmiths_60_horses : 
  minimum_shoeing_time 48 60 5 = 25 := by
  sorry

#eval minimum_shoeing_time 48 60 5

end NUMINAMATH_CALUDE_shoeing_time_for_48_blacksmiths_60_horses_l3819_381975


namespace NUMINAMATH_CALUDE_candidate_d_votes_l3819_381965

theorem candidate_d_votes (total_votes : ℕ) (invalid_percentage : ℚ)
  (candidate_a_percentage : ℚ) (candidate_b_percentage : ℚ) (candidate_c_percentage : ℚ)
  (h1 : total_votes = 10000)
  (h2 : invalid_percentage = 1/4)
  (h3 : candidate_a_percentage = 2/5)
  (h4 : candidate_b_percentage = 3/10)
  (h5 : candidate_c_percentage = 1/5) :
  ↑total_votes * (1 - invalid_percentage) * (1 - (candidate_a_percentage + candidate_b_percentage + candidate_c_percentage)) = 750 := by
  sorry

#check candidate_d_votes

end NUMINAMATH_CALUDE_candidate_d_votes_l3819_381965


namespace NUMINAMATH_CALUDE_cyclic_fraction_product_l3819_381906

theorem cyclic_fraction_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (x + y) / z = (y + z) / x ∧ (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 :=
by sorry

end NUMINAMATH_CALUDE_cyclic_fraction_product_l3819_381906


namespace NUMINAMATH_CALUDE_triangle_validity_and_perimeter_l3819_381946

/-- A triangle with side lengths a, b, and c is valid if it satisfies the triangle inequality -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- The perimeter of a triangle with side lengths a, b, and c -/
def triangle_perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem triangle_validity_and_perimeter :
  let a := 15
  let b := 6
  let c := 13
  is_valid_triangle a b c ∧ triangle_perimeter a b c = 34 := by
  sorry

end NUMINAMATH_CALUDE_triangle_validity_and_perimeter_l3819_381946


namespace NUMINAMATH_CALUDE_min_value_theorem_l3819_381985

def vector_a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
def vector_b (y : ℝ) : ℝ × ℝ := (4, y)

def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem min_value_theorem (x y : ℝ) :
  perpendicular (vector_a x) (vector_b y) →
  ∃ (min : ℝ), min = 6 ∧ ∀ (z : ℝ), 9^x + 3^y ≥ z := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3819_381985


namespace NUMINAMATH_CALUDE_triangle_inequality_check_l3819_381942

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem stating that among the given sets, only {17, 17, 25} can form a triangle -/
theorem triangle_inequality_check : 
  ¬(can_form_triangle 3 4 8) ∧ 
  ¬(can_form_triangle 5 6 11) ∧ 
  ¬(can_form_triangle 6 8 16) ∧ 
  can_form_triangle 17 17 25 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_check_l3819_381942


namespace NUMINAMATH_CALUDE_all_points_collinear_l3819_381970

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  ∃ l : Line, p.onLine l ∧ q.onLine l ∧ r.onLine l

/-- Main theorem -/
theorem all_points_collinear (M : Set Point) (h_finite : Set.Finite M)
  (h_line : ∀ p q r : Point, p ∈ M → q ∈ M → r ∈ M → p ≠ q → 
    (∃ l : Line, p.onLine l ∧ q.onLine l) → (∃ s : Point, s ∈ M ∧ s ≠ p ∧ s ≠ q ∧ s.onLine l)) :
  ∀ p q r : Point, p ∈ M → q ∈ M → r ∈ M → collinear p q r :=
sorry

end NUMINAMATH_CALUDE_all_points_collinear_l3819_381970


namespace NUMINAMATH_CALUDE_catch_up_distance_l3819_381900

/-- The problem of two people traveling at different speeds --/
theorem catch_up_distance
  (speed_a : ℝ)
  (speed_b : ℝ)
  (delay : ℝ)
  (h1 : speed_a = 10)
  (h2 : speed_b = 20)
  (h3 : delay = 6)
  : speed_b * (speed_a * delay / (speed_b - speed_a)) = 120 :=
by sorry

end NUMINAMATH_CALUDE_catch_up_distance_l3819_381900


namespace NUMINAMATH_CALUDE_system_solution_proof_l3819_381976

theorem system_solution_proof : ∃ (x y : ℝ), 
  (2 * x + 7 * y = -6) ∧ 
  (2 * x - 5 * y = 18) ∧ 
  (x = 4) ∧ 
  (y = -2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l3819_381976


namespace NUMINAMATH_CALUDE_two_solutions_l3819_381912

/-- The number of ordered pairs of integers (x, y) satisfying x^4 + y^2 = 4y -/
def count_solutions : ℕ := 2

/-- Predicate that checks if a pair of integers satisfies the equation -/
def satisfies_equation (x y : ℤ) : Prop :=
  x^4 + y^2 = 4*y

theorem two_solutions :
  (∃! (s : Finset (ℤ × ℤ)), s.card = count_solutions ∧ 
    ∀ (p : ℤ × ℤ), p ∈ s ↔ satisfies_equation p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_l3819_381912


namespace NUMINAMATH_CALUDE_bad_carrots_count_l3819_381921

theorem bad_carrots_count (haley_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : 
  haley_carrots = 39 → mom_carrots = 38 → good_carrots = 64 → 
  haley_carrots + mom_carrots - good_carrots = 13 := by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l3819_381921


namespace NUMINAMATH_CALUDE_unique_valid_number_l3819_381995

def is_valid_product (a b : Nat) : Prop :=
  ∃ (x y : Nat), x < 10 ∧ y < 10 ∧ a * 10 + b = x * y

def is_valid_number (n : Nat) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  (∀ i : Fin 9, (n / 10^i.val % 10) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (∀ i : Fin 9, ∀ j : Fin 9, i ≠ j → (n / 10^i.val % 10) ≠ (n / 10^j.val % 10)) ∧
  (∀ i : Fin 8, is_valid_product (n / 10^(i+1).val % 10) (n / 10^i.val % 10))

theorem unique_valid_number : 
  ∃! n : Nat, is_valid_number n ∧ n = 728163549 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3819_381995


namespace NUMINAMATH_CALUDE_correct_product_after_decimal_error_l3819_381993

theorem correct_product_after_decimal_error (incorrect_product : ℝ) 
  (h1 : incorrect_product = 12.04) : 
  ∃ (factor1 factor2 : ℝ), 
    (0.01 ≤ factor1 ∧ factor1 < 1) ∧ 
    (factor1 * 100 * factor2 = incorrect_product) ∧
    (factor1 * factor2 = 0.1204) := by
  sorry

end NUMINAMATH_CALUDE_correct_product_after_decimal_error_l3819_381993


namespace NUMINAMATH_CALUDE_complex_number_equality_l3819_381905

theorem complex_number_equality (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (2 - Complex.I)
  Complex.re z = Complex.im z → a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3819_381905


namespace NUMINAMATH_CALUDE_horner_method_proof_l3819_381984

def horner_polynomial (x : ℝ) : ℝ := x * (x * (x * (x * (2 * x + 0) + 4) + 3) + 1)

theorem horner_method_proof :
  let f (x : ℝ) := 3 * x^2 + 2 * x^5 + 4 * x^3 + x
  f 3 = horner_polynomial 3 ∧ horner_polynomial 3 = 624 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_proof_l3819_381984


namespace NUMINAMATH_CALUDE_inscribed_triangles_area_relation_l3819_381902

/-- Triangle type -/
structure Triangle where
  area : ℝ

/-- Inscribed triangle relation -/
def inscribed (outer inner : Triangle) : Prop :=
  inner.area < outer.area

/-- Parallel sides relation -/
def parallel_sides (t1 t2 : Triangle) : Prop :=
  true  -- We don't need to define this precisely for the theorem

/-- Theorem statement -/
theorem inscribed_triangles_area_relation (a b c : Triangle)
  (h1 : inscribed a b)
  (h2 : inscribed b c)
  (h3 : parallel_sides a c) :
  b.area = Real.sqrt (a.area * c.area) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangles_area_relation_l3819_381902


namespace NUMINAMATH_CALUDE_factorial_products_perfect_square_l3819_381918

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem factorial_products_perfect_square : 
  is_perfect_square (factorial 99 * factorial 100) ∧
  ¬is_perfect_square (factorial 97 * factorial 98) ∧
  ¬is_perfect_square (factorial 97 * factorial 99) ∧
  ¬is_perfect_square (factorial 98 * factorial 99) ∧
  ¬is_perfect_square (factorial 98 * factorial 100) :=
by sorry

end NUMINAMATH_CALUDE_factorial_products_perfect_square_l3819_381918


namespace NUMINAMATH_CALUDE_equation_solution_l3819_381979

def f (x : ℝ) (b : ℝ) : ℝ := 2 * x - b

theorem equation_solution :
  let b : ℝ := 3
  let x : ℝ := 5
  2 * (f x b) - 11 = f (x - 2) b :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3819_381979


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3819_381958

def is_odd_prime (n : ℕ) : Prop := Nat.Prime n ∧ n % 2 = 1

def is_scalene_triangle (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    is_odd_prime a →
    is_odd_prime b →
    is_odd_prime c →
    is_scalene_triangle a b c →
    Nat.Prime (a + b + c) →
    a + b + c ≥ 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3819_381958


namespace NUMINAMATH_CALUDE_range_of_a_l3819_381944

theorem range_of_a (x a : ℝ) : 
  (∀ x, (|x + 1| > 2 → x > a)) →
  (∀ x, (-3 ≤ x ∧ x ≤ 1 → x ≤ a)) →
  (∃ x, x ≤ a ∧ |x + 1| > 2) →
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3819_381944


namespace NUMINAMATH_CALUDE_stratified_sampling_sophomores_l3819_381957

theorem stratified_sampling_sophomores (total_students : ℕ) (sophomore_students : ℕ) (sample_size : ℕ)
  (h1 : total_students = 4500)
  (h2 : sophomore_students = 1500)
  (h3 : sample_size = 600) :
  (sophomore_students : ℚ) / total_students * sample_size = 200 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sophomores_l3819_381957


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3819_381901

theorem min_value_of_expression (x y z w : ℝ) 
  (hx : -1 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 1) 
  (hz : -1 < z ∧ z < 1) 
  (hw : -2 < w ∧ w < 2) :
  (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w/2)) + 
   1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w/2))) ≥ 2 ∧
  (1 / ((1 - 0) * (1 - 0) * (1 - 0) * (1 - 0/2)) + 
   1 / ((1 + 0) * (1 + 0) * (1 + 0) * (1 + 0/2))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3819_381901


namespace NUMINAMATH_CALUDE_recycling_problem_l3819_381949

/-- Given a total number of cans and a number of bags, calculates the number of cans per bag -/
def cans_per_bag (total_cans : ℕ) (num_bags : ℕ) : ℕ :=
  total_cans / num_bags

theorem recycling_problem (total_cans : ℕ) (num_bags : ℕ) 
  (h1 : total_cans = 122) (h2 : num_bags = 2) : 
  cans_per_bag total_cans num_bags = 61 := by
  sorry

end NUMINAMATH_CALUDE_recycling_problem_l3819_381949


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3819_381925

theorem max_value_on_circle (x y : ℝ) : 
  (x - 3)^2 + (y - 4)^2 = 9 → 
  ∃ (z : ℝ), z = 3*x + 4*y ∧ z ≤ 40 ∧ ∃ (x₀ y₀ : ℝ), (x₀ - 3)^2 + (y₀ - 4)^2 = 9 ∧ 3*x₀ + 4*y₀ = 40 :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3819_381925


namespace NUMINAMATH_CALUDE_consumption_increase_l3819_381982

theorem consumption_increase (original_tax original_consumption : ℝ) 
  (h1 : original_tax > 0) (h2 : original_consumption > 0) : 
  let new_tax := 0.76 * original_tax
  let revenue_decrease := 0.1488
  let new_revenue := (1 - revenue_decrease) * (original_tax * original_consumption)
  ∃ (consumption_increase : ℝ), 
    new_tax * (original_consumption * (1 + consumption_increase)) = new_revenue ∧ 
    consumption_increase = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_consumption_increase_l3819_381982


namespace NUMINAMATH_CALUDE_program_arrangements_l3819_381964

/-- The number of ways to arrange n items in k positions --/
def arrangement (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to insert 3 new programs into a list of 10 existing programs --/
theorem program_arrangements : 
  arrangement 11 3 + arrangement 3 2 * arrangement 11 2 + arrangement 3 3 * arrangement 11 1 = 1716 := by
  sorry

end NUMINAMATH_CALUDE_program_arrangements_l3819_381964


namespace NUMINAMATH_CALUDE_quadratic_equation_a_value_l3819_381914

theorem quadratic_equation_a_value (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c →
    (x = -2 ∧ y = -3) ∨ (x = 1 ∧ y = 0)) →
  (∀ x y : ℝ, y = a * (x + 2)^2 - 3) →
  a = 1/3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_a_value_l3819_381914


namespace NUMINAMATH_CALUDE_product_repeating_third_twelve_l3819_381962

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1/3

/-- The product of 0.333... and 12 is 4 --/
theorem product_repeating_third_twelve : repeating_third * 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_repeating_third_twelve_l3819_381962


namespace NUMINAMATH_CALUDE_binary_1001101_equals_octal_115_l3819_381952

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enumFrom 0 b).foldl (λ acc (i, x) => acc + if x then 2^i else 0) 0

def octal_to_decimal (o : List ℕ) : ℕ :=
  (List.enumFrom 0 o).foldl (λ acc (i, x) => acc + x * 8^i) 0

theorem binary_1001101_equals_octal_115 :
  binary_to_decimal [true, false, true, true, false, false, true] =
  octal_to_decimal [5, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_1001101_equals_octal_115_l3819_381952


namespace NUMINAMATH_CALUDE_system_solution_l3819_381919

/-- Proves that (x = 2, y = -1) is the solution to the system of equations:
    2x - y = 5
    5x + 2y = 8 -/
theorem system_solution : ∃ x y : ℝ, (2 * x - y = 5) ∧ (5 * x + 2 * y = 8) ∧ (x = 2) ∧ (y = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3819_381919


namespace NUMINAMATH_CALUDE_rational_results_l3819_381968

-- Define the natural logarithm (ln) and common logarithm (lg)
noncomputable def ln (x : ℝ) := Real.log x
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the logarithm with an arbitrary base
noncomputable def log (b : ℝ) (x : ℝ) := ln x / ln b

-- State the theorem
theorem rational_results :
  (2 * lg 2 + lg 25 = 2) ∧
  (3^(1 / ln 3) - Real.exp 1 = 0) ∧
  (log 4 3 * log 3 6 * log 6 8 = 3/2) := by sorry

end NUMINAMATH_CALUDE_rational_results_l3819_381968


namespace NUMINAMATH_CALUDE_lice_check_time_proof_l3819_381947

theorem lice_check_time_proof (kindergarteners first_graders second_graders third_graders : ℕ)
  (time_per_check : ℕ) (h1 : kindergarteners = 26) (h2 : first_graders = 19)
  (h3 : second_graders = 20) (h4 : third_graders = 25) (h5 : time_per_check = 2) :
  (kindergarteners + first_graders + second_graders + third_graders) * time_per_check / 60 = 3 :=
by sorry

end NUMINAMATH_CALUDE_lice_check_time_proof_l3819_381947


namespace NUMINAMATH_CALUDE_sam_travel_time_l3819_381907

-- Define the points and distances
def point_A : ℝ := 0
def point_B : ℝ := 1000
def point_C : ℝ := 600

-- Define Sam's speed
def sam_speed : ℝ := 50

-- State the theorem
theorem sam_travel_time :
  let total_distance := point_B - point_A
  let time := total_distance / sam_speed
  (point_C - point_A = 600) ∧ 
  (point_B - point_C = 400) ∧ 
  (sam_speed = 50) →
  time = 20 := by sorry

end NUMINAMATH_CALUDE_sam_travel_time_l3819_381907


namespace NUMINAMATH_CALUDE_relation_between_exponents_l3819_381945

theorem relation_between_exponents 
  {a b c d x y p z : ℝ} 
  (h1 : a^x = b^p) 
  (h2 : a^x = c) 
  (h3 : b^y = a^z) 
  (h4 : b^y = d) 
  (ha : a > 0) 
  (hb : b > 0) : 
  p * y = x * z := by
sorry


end NUMINAMATH_CALUDE_relation_between_exponents_l3819_381945


namespace NUMINAMATH_CALUDE_photograph_perimeter_l3819_381954

theorem photograph_perimeter (w h m : ℝ) 
  (border_1 : (w + 2) * (h + 2) = m)
  (border_3 : (w + 6) * (h + 6) = m + 52) :
  2 * w + 2 * h = 10 := by
  sorry

end NUMINAMATH_CALUDE_photograph_perimeter_l3819_381954


namespace NUMINAMATH_CALUDE_prism_base_side_length_l3819_381917

/-- Given a rectangular prism with a square base, prove that with the given dimensions and properties, the side length of the base is 2 meters. -/
theorem prism_base_side_length (height : ℝ) (density : ℝ) (weight : ℝ) (volume : ℝ) (side : ℝ) :
  height = 8 →
  density = 2700 →
  weight = 86400 →
  volume = weight / density →
  volume = side^2 * height →
  side = 2 := by
  sorry


end NUMINAMATH_CALUDE_prism_base_side_length_l3819_381917


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_l3819_381939

theorem stratified_sampling_proportion (total : ℕ) (first_year : ℕ) (second_year : ℕ) 
  (sample_first : ℕ) (sample_second : ℕ) :
  total = first_year + second_year →
  first_year * sample_second = second_year * sample_first →
  sample_first = 6 →
  first_year = 30 →
  second_year = 40 →
  sample_second = 8 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_l3819_381939


namespace NUMINAMATH_CALUDE_evaluate_nested_expression_l3819_381983

def f (x : ℕ) : ℕ := 3 * (3 * (3 * (3 * (3 * x + 2) + 2) + 2) + 2) + 2

theorem evaluate_nested_expression :
  f 5 = 1457 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_nested_expression_l3819_381983


namespace NUMINAMATH_CALUDE_max_area_right_quadrilateral_in_circle_l3819_381998

/-- 
Given a circle with radius r, prove that the area of a right quadrilateral inscribed in the circle 
with one side tangent to the circle and one side a chord of the circle is maximized when the 
distance from the center of the circle to the midpoint of the chord is r/2.
-/
theorem max_area_right_quadrilateral_in_circle (r : ℝ) (h : r > 0) :
  ∃ (x y : ℝ),
    x^2 + y^2 = r^2 ∧  -- Pythagorean theorem for right triangle OCE
    (∀ (x' y' : ℝ), x'^2 + y'^2 = r^2 → (x + r) * y ≥ (x' + r) * y') ∧  -- Area maximization condition
    x = r / 2  -- The distance that maximizes the area
  := by sorry

end NUMINAMATH_CALUDE_max_area_right_quadrilateral_in_circle_l3819_381998


namespace NUMINAMATH_CALUDE_handshake_theorem_l3819_381992

theorem handshake_theorem (n : ℕ) (h : n = 30) :
  let total_handshakes := n * 3 / 2
  total_handshakes = 45 := by
  sorry

end NUMINAMATH_CALUDE_handshake_theorem_l3819_381992


namespace NUMINAMATH_CALUDE_membership_change_fall_increase_value_l3819_381913

/-- The percentage increase in membership during fall -/
def fall_increase : ℝ := sorry

/-- The percentage decrease in membership during spring -/
def spring_decrease : ℝ := 19

/-- The total percentage increase from original to spring membership -/
def total_increase : ℝ := 12.52

/-- Theorem stating the relationship between fall increase, spring decrease, and total increase -/
theorem membership_change :
  (1 + fall_increase / 100) * (1 - spring_decrease / 100) = 1 + total_increase / 100 :=
sorry

/-- The fall increase is approximately 38.91% -/
theorem fall_increase_value : 
  ∃ ε > 0, |fall_increase - 38.91| < ε :=
sorry

end NUMINAMATH_CALUDE_membership_change_fall_increase_value_l3819_381913


namespace NUMINAMATH_CALUDE_number_equation_l3819_381924

theorem number_equation (x : ℝ) : 43 + 3 * x = 58 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3819_381924


namespace NUMINAMATH_CALUDE_expression_equality_l3819_381926

theorem expression_equality (a b c n : ℝ) 
  (h1 : a + b = c * n) 
  (h2 : b + c = a * n) 
  (h3 : a + c = b * n) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  (a + b) * (b + c) * (a + c) / (a * b * c) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3819_381926


namespace NUMINAMATH_CALUDE_percentage_runs_by_running_is_fifty_percent_l3819_381934

def total_runs : ℕ := 120
def boundaries : ℕ := 3
def sixes : ℕ := 8
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

def runs_from_boundaries_and_sixes : ℕ := boundaries * runs_per_boundary + sixes * runs_per_six

def runs_by_running : ℕ := total_runs - runs_from_boundaries_and_sixes

theorem percentage_runs_by_running_is_fifty_percent :
  (runs_by_running : ℚ) / total_runs * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_percentage_runs_by_running_is_fifty_percent_l3819_381934


namespace NUMINAMATH_CALUDE_shifted_roots_polynomial_l3819_381911

theorem shifted_roots_polynomial (r₁ r₂ : ℝ) (h_sum : r₁ + r₂ = 15) (h_prod : r₁ * r₂ = 36) :
  (X - (r₁ + 3)) * (X - (r₂ + 3)) = X^2 - 21*X + 90 :=
by sorry

end NUMINAMATH_CALUDE_shifted_roots_polynomial_l3819_381911


namespace NUMINAMATH_CALUDE_registered_students_calculation_l3819_381981

/-- The number of students registered for a science course. -/
def registered_students (students_yesterday : ℕ) (students_absent_today : ℕ) : ℕ :=
  let students_today := (2 * students_yesterday) - (2 * students_yesterday / 10)
  students_today + students_absent_today

/-- Theorem stating the number of registered students given the problem conditions. -/
theorem registered_students_calculation :
  registered_students 70 30 = 156 := by
  sorry

#eval registered_students 70 30

end NUMINAMATH_CALUDE_registered_students_calculation_l3819_381981


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3819_381960

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem min_value_of_reciprocal_sum (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 2014 = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 2) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3819_381960


namespace NUMINAMATH_CALUDE_parallelogram_ABCD_area_l3819_381963

-- Define the parallelogram vertices
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (5, 1)
def C : ℝ × ℝ := (7, 4)
def D : ℝ × ℝ := (3, 4)

-- Define a function to calculate the area of a parallelogram given two vectors
def parallelogramArea (v1 v2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  abs (x1 * y2 - x2 * y1)

-- Theorem statement
theorem parallelogram_ABCD_area :
  parallelogramArea (B.1 - A.1, B.2 - A.2) (D.1 - A.1, D.2 - A.2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_ABCD_area_l3819_381963


namespace NUMINAMATH_CALUDE_fettuccine_tortellini_ratio_l3819_381973

/-- The ratio of students preferring fettuccine to those preferring tortellini -/
theorem fettuccine_tortellini_ratio 
  (total_students : ℕ) 
  (fettuccine_preference : ℕ) 
  (tortellini_preference : ℕ) 
  (h1 : total_students = 800)
  (h2 : fettuccine_preference = 200)
  (h3 : tortellini_preference = 160) : 
  (fettuccine_preference : ℚ) / tortellini_preference = 5 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_fettuccine_tortellini_ratio_l3819_381973


namespace NUMINAMATH_CALUDE_book_cost_solution_l3819_381987

def book_cost_problem (p : ℝ) : Prop :=
  7 * p < 15 ∧ 11 * p > 22

theorem book_cost_solution :
  ∃ p : ℝ, book_cost_problem p ∧ p = 2.10 := by
sorry

end NUMINAMATH_CALUDE_book_cost_solution_l3819_381987


namespace NUMINAMATH_CALUDE_expenditure_ratio_proof_l3819_381978

/-- Given two persons P1 and P2 with incomes and expenditures, prove their expenditure ratio --/
theorem expenditure_ratio_proof 
  (income_ratio : ℚ) -- Ratio of incomes P1:P2
  (savings : ℕ) -- Amount saved by each person
  (income_p1 : ℕ) -- Income of P1
  (h1 : income_ratio = 5 / 4) -- Income ratio condition
  (h2 : savings = 1600) -- Savings condition
  (h3 : income_p1 = 4000) -- P1's income condition
  : (income_p1 - savings) / ((income_p1 * 4 / 5) - savings) = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_expenditure_ratio_proof_l3819_381978


namespace NUMINAMATH_CALUDE_line_intersects_parabola_once_l3819_381986

-- Define the point A
def A : ℝ × ℝ := (1, 2)

-- Define the parabola C
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line L
def L (y : ℝ) : Prop := y = 2

-- Theorem statement
theorem line_intersects_parabola_once :
  L (A.2) ∧ 
  (∃! p : ℝ × ℝ, C p.1 p.2 ∧ L p.2) ∧
  (∀ y : ℝ, L y → ∃ x : ℝ, (x, y) = A ∨ C x y) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_once_l3819_381986


namespace NUMINAMATH_CALUDE_first_bouquet_carnations_l3819_381955

/-- The number of carnations in the first bouquet -/
def carnations_in_first_bouquet (total_bouquets : ℕ) 
  (carnations_in_second : ℕ) (carnations_in_third : ℕ) (average : ℕ) : ℕ :=
  total_bouquets * average - carnations_in_second - carnations_in_third

/-- Theorem stating the number of carnations in the first bouquet -/
theorem first_bouquet_carnations :
  carnations_in_first_bouquet 3 14 13 12 = 9 := by
  sorry

#eval carnations_in_first_bouquet 3 14 13 12

end NUMINAMATH_CALUDE_first_bouquet_carnations_l3819_381955


namespace NUMINAMATH_CALUDE_no_four_identical_digits_in_1990_denominator_l3819_381991

theorem no_four_identical_digits_in_1990_denominator :
  ¬ ∃ (A : ℕ) (d : ℕ), 
    A > 0 ∧ A < 1990 ∧ d < 10 ∧
    ∃ (k : ℕ), (A * 10^k) % 1990 = d * 1111 :=
by sorry

end NUMINAMATH_CALUDE_no_four_identical_digits_in_1990_denominator_l3819_381991


namespace NUMINAMATH_CALUDE_ellipse_m_range_l3819_381988

theorem ellipse_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (2 + m) - y^2 / (m + 1) = 1 ∧ 
   (2 + m > 0 ∧ -(m + 1) > 0) ∧ 
   (2 + m ≠ -(m + 1))) ↔ 
  (m > -2 ∧ m < -3/2) ∨ (m > -3/2 ∧ m < -1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l3819_381988
