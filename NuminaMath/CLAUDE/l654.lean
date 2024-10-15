import Mathlib

namespace NUMINAMATH_CALUDE_triangle_side_b_is_4_triangle_area_is_4_sqrt_3_l654_65431

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  side_angle_relation : a = b * Real.cos C + c * Real.cos B

-- Theorem 1
theorem triangle_side_b_is_4 (abc : Triangle) (h : abc.a - 4 * Real.cos abc.C = abc.c * Real.cos abc.B) :
  abc.b = 4 := by sorry

-- Theorem 2
theorem triangle_area_is_4_sqrt_3 (abc : Triangle)
  (h1 : abc.a - 4 * Real.cos abc.C = abc.c * Real.cos abc.B)
  (h2 : abc.a^2 + abc.b^2 + abc.c^2 = 2 * Real.sqrt 3 * abc.a * abc.b * Real.sin abc.C) :
  abc.a * abc.b * Real.sin abc.C / 2 = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_b_is_4_triangle_area_is_4_sqrt_3_l654_65431


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l654_65407

theorem gcd_polynomial_and_multiple (y : ℤ) : 
  18090 ∣ y → 
  Int.gcd ((3*y + 5)*(6*y + 7)*(10*y + 3)*(5*y + 11)*(y + 7)) y = 8085 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_multiple_l654_65407


namespace NUMINAMATH_CALUDE_solve_otimes_equation_l654_65477

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := (a - 2) * (b + 1)

-- Theorem statement
theorem solve_otimes_equation : 
  ∃! x : ℝ, otimes (-4) (x + 3) = 6 ∧ x = -5 := by sorry

end NUMINAMATH_CALUDE_solve_otimes_equation_l654_65477


namespace NUMINAMATH_CALUDE_roses_cut_l654_65401

def initial_roses : ℕ := 6
def final_roses : ℕ := 16

theorem roses_cut (cut_roses : ℕ) : cut_roses = final_roses - initial_roses := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l654_65401


namespace NUMINAMATH_CALUDE_range_of_x_minus_2y_l654_65435

theorem range_of_x_minus_2y (x y : ℝ) 
  (hx : 30 < x ∧ x < 42) 
  (hy : 16 < y ∧ y < 24) : 
  ∀ z, z ∈ Set.Ioo (-18 : ℝ) 10 ↔ ∃ (x' y' : ℝ), 
    30 < x' ∧ x' < 42 ∧ 
    16 < y' ∧ y' < 24 ∧ 
    z = x' - 2*y' :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_minus_2y_l654_65435


namespace NUMINAMATH_CALUDE_matrix_linear_combination_l654_65465

theorem matrix_linear_combination : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, 1; 3, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-3, -2; 2, -4]
  2 • A + 3 • B = !![-1, -4; 12, -2] := by
  sorry

end NUMINAMATH_CALUDE_matrix_linear_combination_l654_65465


namespace NUMINAMATH_CALUDE_egg_laying_hens_l654_65414

/-- Calculates the number of egg-laying hens on Mr. Curtis's farm -/
theorem egg_laying_hens (total_chickens roosters non_laying_hens : ℕ) 
  (h1 : total_chickens = 325)
  (h2 : roosters = 28)
  (h3 : non_laying_hens = 20) :
  total_chickens - roosters - non_laying_hens = 277 := by
  sorry

#check egg_laying_hens

end NUMINAMATH_CALUDE_egg_laying_hens_l654_65414


namespace NUMINAMATH_CALUDE_rectangular_cube_height_l654_65486

-- Define the dimensions of the rectangular cube
def length : ℝ := 3
def width : ℝ := 2

-- Define the side length of the reference cube
def cubeSide : ℝ := 2

-- Define the surface area of the rectangular cube
def surfaceArea (h : ℝ) : ℝ := 2 * length * width + 2 * length * h + 2 * width * h

-- Define the surface area of the reference cube
def cubeSurfaceArea : ℝ := 6 * cubeSide^2

-- Theorem statement
theorem rectangular_cube_height : 
  ∃ h : ℝ, surfaceArea h = cubeSurfaceArea ∧ h = 1.2 := by sorry

end NUMINAMATH_CALUDE_rectangular_cube_height_l654_65486


namespace NUMINAMATH_CALUDE_claire_crafting_hours_l654_65482

def total_hours : ℕ := 24
def cleaning_hours : ℕ := 4
def cooking_hours : ℕ := 2
def sleeping_hours : ℕ := 8

def remaining_hours : ℕ := total_hours - (cleaning_hours + cooking_hours + sleeping_hours)

theorem claire_crafting_hours : remaining_hours / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_claire_crafting_hours_l654_65482


namespace NUMINAMATH_CALUDE_max_z3_value_max_z3_value_tight_l654_65440

theorem max_z3_value (z₁ z₂ z₃ : ℂ) 
  (h₁ : Complex.abs z₁ ≤ 1)
  (h₂ : Complex.abs z₂ ≤ 2)
  (h₃ : Complex.abs (2 * z₃ - z₁ - z₂) ≤ Complex.abs (z₁ - z₂)) :
  Complex.abs z₃ ≤ Real.sqrt 5 :=
by
  sorry

theorem max_z3_value_tight : ∃ (z₁ z₂ z₃ : ℂ),
  Complex.abs z₁ ≤ 1 ∧
  Complex.abs z₂ ≤ 2 ∧
  Complex.abs (2 * z₃ - z₁ - z₂) ≤ Complex.abs (z₁ - z₂) ∧
  Complex.abs z₃ = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_z3_value_max_z3_value_tight_l654_65440


namespace NUMINAMATH_CALUDE_percentage_less_than_l654_65460

theorem percentage_less_than (p t j : ℝ) 
  (ht : t = p * (1 - 0.0625))
  (hj : j = t * (1 - 0.20)) : 
  j = p * (1 - 0.25) := by
sorry

end NUMINAMATH_CALUDE_percentage_less_than_l654_65460


namespace NUMINAMATH_CALUDE_calculate_expression_l654_65444

theorem calculate_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l654_65444


namespace NUMINAMATH_CALUDE_x_squared_plus_x_minus_one_zero_l654_65425

theorem x_squared_plus_x_minus_one_zero (x : ℝ) :
  x^2 + x - 1 = 0 → x^4 + 2*x^3 - 3*x^2 - 4*x + 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_x_minus_one_zero_l654_65425


namespace NUMINAMATH_CALUDE_tunnel_length_proof_l654_65451

/-- Represents the scale of a map -/
structure MapScale where
  ratio : ℚ

/-- Represents a length on a map -/
structure MapLength where
  length : ℚ
  unit : String

/-- Represents an actual length in reality -/
structure ActualLength where
  length : ℚ
  unit : String

/-- Converts a MapLength to an ActualLength based on a given MapScale -/
def convertMapLengthToActual (scale : MapScale) (mapLength : MapLength) : ActualLength :=
  { length := mapLength.length * scale.ratio
    unit := "cm" }

/-- Converts centimeters to kilometers -/
def cmToKm (cm : ℚ) : ℚ :=
  cm / 100000

theorem tunnel_length_proof (scale : MapScale) (mapLength : MapLength) :
  scale.ratio = 38000 →
  mapLength.length = 7 →
  mapLength.unit = "cm" →
  let actualLength := convertMapLengthToActual scale mapLength
  cmToKm actualLength.length = 2.66 := by
    sorry

end NUMINAMATH_CALUDE_tunnel_length_proof_l654_65451


namespace NUMINAMATH_CALUDE_bottle_production_l654_65427

/-- Given that 6 identical machines produce 420 bottles per minute at a constant rate,
    prove that 10 such machines will produce 2800 bottles in 4 minutes. -/
theorem bottle_production
  (machines : ℕ → ℕ) -- Function mapping number of machines to bottles produced per minute
  (h1 : machines 6 = 420) -- 6 machines produce 420 bottles per minute
  (h2 : ∀ n : ℕ, machines n = n * (machines 1)) -- Constant rate production
  : machines 10 * 4 = 2800 := by
  sorry


end NUMINAMATH_CALUDE_bottle_production_l654_65427


namespace NUMINAMATH_CALUDE_indefinite_integral_sin_3x_l654_65402

theorem indefinite_integral_sin_3x (x : ℝ) :
  (deriv (fun x => -1/3 * (x + 5) * Real.cos (3 * x) + 1/9 * Real.sin (3 * x))) x
  = (x + 5) * Real.sin (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_sin_3x_l654_65402


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l654_65449

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 2^k ∣ (10^1000 + 4^500) ∧ 
   ∀ m : ℕ, 2^m ∣ (10^1000 + 4^500) → m ≤ k) → 
  n = 1003 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l654_65449


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l654_65484

/-- A quadratic function satisfying certain conditions -/
def f (x : ℝ) : ℝ := 2*x^2 - 2*x + 1

/-- The main theorem about the quadratic function f -/
theorem quadratic_function_properties :
  (∀ x, f (x + 1) - f x = 2 * x) ∧
  f 0 = 1 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, -1/2 ≤ f x ∧ f x ≤ 1) ∧
  (∀ a : ℝ,
    (a ≤ -1/2 → ∀ x ∈ Set.Icc a (a + 1), 2*a^2 + 2*a + 3 ≤ f x ∧ f x ≤ 2*a^2 - 2*a + 1) ∧
    (-1/2 < a ∧ a ≤ 0 → ∀ x ∈ Set.Icc a (a + 1), -1/2 ≤ f x ∧ f x ≤ 2*a^2 - 2*a + 1) ∧
    (0 ≤ a ∧ a < 1/2 → ∀ x ∈ Set.Icc a (a + 1), -1/2 ≤ f x ∧ f x ≤ 2*a^2 + 2*a + 3) ∧
    (1/2 ≤ a → ∀ x ∈ Set.Icc a (a + 1), 2*a^2 - 2*a + 1 ≤ f x ∧ f x ≤ 2*a^2 + 2*a + 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l654_65484


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l654_65490

theorem least_subtraction_for_divisibility (n m k : ℕ) (h : n - k ≥ 0) : 
  (∃ q : ℕ, n - k = m * q) ∧ 
  (∀ j : ℕ, j < k → ¬(∃ q : ℕ, n - j = m * q)) → 
  k = n % m :=
sorry

#check least_subtraction_for_divisibility 2361 23 15

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l654_65490


namespace NUMINAMATH_CALUDE_pizza_slices_l654_65464

theorem pizza_slices (x : ℚ) 
  (half_eaten : x / 2 = x - x / 2)
  (third_of_remaining_eaten : x / 2 - (x / 2) / 3 = x / 2 - x / 6)
  (four_slices_left : x / 2 - x / 6 = 4) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l654_65464


namespace NUMINAMATH_CALUDE_f_increasing_when_a_1_m_range_l654_65424

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + a*x

-- Define monotonically increasing
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Theorem 1: f is monotonically increasing when a = 1
theorem f_increasing_when_a_1 :
  monotonically_increasing (f 1) := by sorry

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (m : ℝ) : Prop :=
  (∀ a : ℝ, monotonically_increasing (f a) → |a - 1| ≤ m) ∧
  (∃ a : ℝ, |a - 1| ≤ m ∧ ¬monotonically_increasing (f a))

-- Theorem 2: The range of m is [0,1)
theorem m_range :
  ∀ m : ℝ, (m > 0 ∧ necessary_not_sufficient m) ↔ (0 ≤ m ∧ m < 1) := by sorry

end NUMINAMATH_CALUDE_f_increasing_when_a_1_m_range_l654_65424


namespace NUMINAMATH_CALUDE_f_is_even_l654_65419

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_f_is_even_l654_65419


namespace NUMINAMATH_CALUDE_weekly_allowance_calculation_l654_65488

theorem weekly_allowance_calculation (weekly_allowance : ℚ) : 
  (4 * weekly_allowance / 2 * 3 / 4 = 15) → weekly_allowance = 10 := by
  sorry

end NUMINAMATH_CALUDE_weekly_allowance_calculation_l654_65488


namespace NUMINAMATH_CALUDE_longer_string_length_l654_65485

theorem longer_string_length 
  (total_length : ℕ) 
  (difference : ℕ) 
  (h1 : total_length = 348) 
  (h2 : difference = 72) : 
  ∃ (longer shorter : ℕ), 
    longer + shorter = total_length ∧ 
    longer - shorter = difference ∧ 
    longer = 210 := by
  sorry

end NUMINAMATH_CALUDE_longer_string_length_l654_65485


namespace NUMINAMATH_CALUDE_fractional_equation_root_l654_65454

theorem fractional_equation_root (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ m / (x - 2) + 2 * x / (x - 2) = 1) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l654_65454


namespace NUMINAMATH_CALUDE_triangle_side_length_l654_65471

theorem triangle_side_length (a b c : ℝ) (S : ℝ) (hA : a = 4) (hB : b = 5) (hS : S = 5 * Real.sqrt 3) :
  c = Real.sqrt 21 ∨ c = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l654_65471


namespace NUMINAMATH_CALUDE_two_possible_values_for_k_l654_65448

theorem two_possible_values_for_k (a b c k : ℝ) : 
  (a / (b + c) = k ∧ b / (c + a) = k ∧ c / (a + b) = k) → 
  (k = 1/2 ∨ k = -1) ∧ ∀ x : ℝ, (x = 1/2 ∨ x = -1) → ∃ a b c : ℝ, a / (b + c) = x ∧ b / (c + a) = x ∧ c / (a + b) = x :=
by sorry

end NUMINAMATH_CALUDE_two_possible_values_for_k_l654_65448


namespace NUMINAMATH_CALUDE_percentage_of_b_l654_65412

theorem percentage_of_b (a b c : ℝ) (h1 : 12 = 0.04 * a) (h2 : ∃ p, p * b = 4) (h3 : c = b / a) :
  ∃ p, p * b = 4 ∧ p = 4 / (c * 300) := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_b_l654_65412


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_eq_neg_one_l654_65421

/-- Two vectors in R² -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Definition of parallel vectors in R² -/
def parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.x = k * w.x ∧ v.y = k * w.y

/-- The main theorem -/
theorem parallel_vectors_imply_x_eq_neg_one :
  ∀ (x : ℝ),
  let a : Vector2D := ⟨x, 1⟩
  let b : Vector2D := ⟨1, -1⟩
  parallel a b → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_eq_neg_one_l654_65421


namespace NUMINAMATH_CALUDE_function_properties_l654_65408

def f (a c x : ℝ) : ℝ := a * x^2 + 2 * x + c

def g (a c x : ℝ) : ℝ := f a c x - 2 * x - 3 + |x - 1|

theorem function_properties :
  ∀ a c : ℕ+,
  f a c 1 = 5 →
  6 < f a c 2 ∧ f a c 2 < 11 →
  (a = 1 ∧ c = 2) ∧
  (∀ x : ℝ, g a c x ≥ -1/4) ∧
  (∃ x : ℝ, g a c x = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l654_65408


namespace NUMINAMATH_CALUDE_shoes_cost_eleven_l654_65491

/-- The cost of shoes given initial amount, sweater cost, T-shirt cost, and remaining amount -/
def cost_of_shoes (initial_amount sweater_cost tshirt_cost remaining_amount : ℕ) : ℕ :=
  initial_amount - sweater_cost - tshirt_cost - remaining_amount

/-- Theorem stating that the cost of shoes is 11 given the problem conditions -/
theorem shoes_cost_eleven :
  cost_of_shoes 91 24 6 50 = 11 := by
  sorry

end NUMINAMATH_CALUDE_shoes_cost_eleven_l654_65491


namespace NUMINAMATH_CALUDE_square_of_sum_l654_65420

theorem square_of_sum (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 := by sorry

end NUMINAMATH_CALUDE_square_of_sum_l654_65420


namespace NUMINAMATH_CALUDE_warehouse_repacking_l654_65429

/-- The number of books left over after repacking in the warehouse scenario -/
theorem warehouse_repacking (initial_boxes : Nat) (books_per_initial_box : Nat) 
  (damaged_books : Nat) (books_per_new_box : Nat) 
  (h1 : initial_boxes = 1200)
  (h2 : books_per_initial_box = 35)
  (h3 : damaged_books = 100)
  (h4 : books_per_new_box = 45) : 
  (initial_boxes * books_per_initial_box - damaged_books) % books_per_new_box = 5 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_repacking_l654_65429


namespace NUMINAMATH_CALUDE_revenue_fall_percentage_l654_65446

theorem revenue_fall_percentage (R R' P P' : ℝ) 
  (h1 : P = 0.1 * R)
  (h2 : P' = 0.14 * R')
  (h3 : P' = 0.98 * P) :
  R' = 0.7 * R := by
sorry

end NUMINAMATH_CALUDE_revenue_fall_percentage_l654_65446


namespace NUMINAMATH_CALUDE_circle_tangent_slope_l654_65467

/-- The circle with center (2,0) and radius √3 -/
def Circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 3

/-- The vector from origin to point M -/
def OM (x y : ℝ) : ℝ × ℝ := (x, y)

/-- The vector from center C to point M -/
def CM (x y : ℝ) : ℝ × ℝ := (x - 2, y)

/-- The dot product of OM and CM -/
def dotProduct (x y : ℝ) : ℝ := x * (x - 2) + y * y

theorem circle_tangent_slope (x y : ℝ) :
  Circle x y →
  dotProduct x y = 0 →
  (y / x = Real.sqrt 3 ∨ y / x = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_slope_l654_65467


namespace NUMINAMATH_CALUDE_six_hour_rental_cost_l654_65438

/-- Represents the cost structure for kayak and paddle rental --/
structure RentalCost where
  paddleFee : ℕ
  kayakHourlyRate : ℕ

/-- Calculates the total cost for a given number of hours --/
def totalCost (rc : RentalCost) (hours : ℕ) : ℕ :=
  rc.paddleFee + rc.kayakHourlyRate * hours

theorem six_hour_rental_cost 
  (rc : RentalCost)
  (three_hour_cost : totalCost rc 3 = 30)
  (kayak_rate : rc.kayakHourlyRate = 5) :
  totalCost rc 6 = 45 := by
  sorry

#check six_hour_rental_cost

end NUMINAMATH_CALUDE_six_hour_rental_cost_l654_65438


namespace NUMINAMATH_CALUDE_william_wins_l654_65403

theorem william_wins (total_rounds : ℕ) (williams_advantage : ℕ) (williams_wins : ℕ) : 
  total_rounds = 15 → williams_advantage = 5 → williams_wins = 10 → 
  williams_wins = (total_rounds + williams_advantage) / 2 := by
  sorry

end NUMINAMATH_CALUDE_william_wins_l654_65403


namespace NUMINAMATH_CALUDE_mersenne_prime_definition_l654_65411

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, n = 2^p - 1 ∧ Nat.Prime n

def largest_known_prime : ℕ := 2^82589933 - 1

axiom largest_known_prime_is_prime : Nat.Prime largest_known_prime

theorem mersenne_prime_definition :
  ∀ n : ℕ, is_mersenne_prime n → (∃ name : String, name = "Mersenne prime") :=
by sorry

end NUMINAMATH_CALUDE_mersenne_prime_definition_l654_65411


namespace NUMINAMATH_CALUDE_range_of_a_l654_65428

-- Define the sets A, B, and C
def A : Set ℝ := {x | -5 < x ∧ x < 1}
def B : Set ℝ := {x | -2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem range_of_a (a : ℝ) (h : A ∩ B ⊆ C a) : a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l654_65428


namespace NUMINAMATH_CALUDE_no_real_roots_condition_implies_inequality_g_no_intersect_l654_65458

/-- A quadratic function that doesn't intersect with y = x -/
structure NoIntersectQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  no_intersect : ∀ x : ℝ, a * x^2 + b * x + c ≠ x

def f (q : NoIntersectQuadratic) (x : ℝ) : ℝ := q.a * x^2 + q.b * x + q.c

theorem no_real_roots (q : NoIntersectQuadratic) : ∀ x : ℝ, f q (f q x) ≠ x := by sorry

theorem condition_implies_inequality (q : NoIntersectQuadratic) (h : q.a + q.b + q.c = 0) :
  ∀ x : ℝ, f q (f q x) < x := by sorry

def g (q : NoIntersectQuadratic) (x : ℝ) : ℝ := q.a * x^2 - q.b * x + q.c

theorem g_no_intersect (q : NoIntersectQuadratic) : ∀ x : ℝ, g q x ≠ -x := by sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_implies_inequality_g_no_intersect_l654_65458


namespace NUMINAMATH_CALUDE_student_excess_is_105_l654_65492

/-- Represents the composition of a fourth-grade classroom -/
structure Classroom where
  students : Nat
  guinea_pigs : Nat
  teachers : Nat

/-- The number of fourth-grade classrooms -/
def num_classrooms : Nat := 5

/-- A fourth-grade classroom in Big Valley School -/
def big_valley_classroom : Classroom :=
  { students := 25, guinea_pigs := 3, teachers := 1 }

/-- Theorem: The number of students exceeds the total number of guinea pigs and teachers by 105 in all fourth-grade classrooms -/
theorem student_excess_is_105 : 
  (num_classrooms * big_valley_classroom.students) - 
  (num_classrooms * (big_valley_classroom.guinea_pigs + big_valley_classroom.teachers)) = 105 := by
  sorry

end NUMINAMATH_CALUDE_student_excess_is_105_l654_65492


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l654_65473

theorem geometric_series_common_ratio :
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := -8/3
  let a₃ : ℚ := 64/21
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 2 → a₂ = a₁ * r ∧ a₃ = a₂ * r) →
  r = -14/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l654_65473


namespace NUMINAMATH_CALUDE_correct_election_result_l654_65418

/-- Election results with three candidates -/
structure ElectionResult where
  total_votes : ℕ
  votes_a : ℕ
  votes_b : ℕ
  votes_c : ℕ

/-- Conditions of the election -/
def election_conditions (result : ElectionResult) : Prop :=
  (result.votes_a : ℚ) / result.total_votes = 45 / 100 ∧
  (result.votes_b : ℚ) / result.total_votes = 35 / 100 ∧
  (result.votes_c : ℚ) / result.total_votes = 20 / 100 ∧
  result.votes_a - result.votes_b = 2500 ∧
  result.total_votes = result.votes_a + result.votes_b + result.votes_c

/-- Theorem stating the correct election results -/
theorem correct_election_result :
  ∃ (result : ElectionResult),
    election_conditions result ∧
    result.total_votes = 25000 ∧
    result.votes_a = 11250 ∧
    result.votes_b = 8750 ∧
    result.votes_c = 5000 := by
  sorry

end NUMINAMATH_CALUDE_correct_election_result_l654_65418


namespace NUMINAMATH_CALUDE_papi_calot_plants_l654_65443

/-- The number of plants Papi Calot needs to buy for his potato garden. -/
def total_plants (rows : ℕ) (plants_per_row : ℕ) (additional_plants : ℕ) : ℕ :=
  rows * plants_per_row + additional_plants

/-- Theorem stating the total number of plants Papi Calot needs to buy. -/
theorem papi_calot_plants : total_plants 7 18 15 = 141 := by
  sorry

end NUMINAMATH_CALUDE_papi_calot_plants_l654_65443


namespace NUMINAMATH_CALUDE_afternoon_campers_count_l654_65410

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 44

/-- The difference between morning and afternoon campers -/
def difference : ℕ := 5

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := morning_campers - difference

theorem afternoon_campers_count : afternoon_campers = 39 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_campers_count_l654_65410


namespace NUMINAMATH_CALUDE_inequality_system_solution_l654_65422

theorem inequality_system_solution (x : ℝ) :
  (1/3 * x - 1 ≤ 1/2 * x + 1) →
  (3 * x - (x - 2) ≥ 6) →
  (x + 1 > (4 * x - 1) / 3) →
  (2 ≤ x ∧ x < 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l654_65422


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l654_65461

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 48) :
  x^2 + 4*x*y + 4*y^2 + 3*z^2 ≥ 144 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 48 ∧ x^2 + 4*x*y + 4*y^2 + 3*z^2 = 144 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l654_65461


namespace NUMINAMATH_CALUDE_lcm_of_308_and_275_l654_65479

theorem lcm_of_308_and_275 :
  let a := 308
  let b := 275
  let hcf := 11
  let lcm := Nat.lcm a b
  (Nat.gcd a b = hcf) → (lcm = 7700) := by
sorry

end NUMINAMATH_CALUDE_lcm_of_308_and_275_l654_65479


namespace NUMINAMATH_CALUDE_spiral_similarity_composition_l654_65405

open Real

/-- A spiral similarity (also known as a rotational homothety) -/
structure SpiralSimilarity where
  center : ℝ × ℝ
  angle : ℝ
  coefficient : ℝ

/-- Composition of two spiral similarities -/
def compose (P₁ P₂ : SpiralSimilarity) : SpiralSimilarity :=
  sorry

/-- Rotation -/
structure Rotation where
  center : ℝ × ℝ
  angle : ℝ

/-- Check if a spiral similarity is a rotation -/
def isRotation (P : SpiralSimilarity) : Prop :=
  sorry

/-- The angle between two vectors -/
def vectorAngle (v₁ v₂ : ℝ × ℝ) : ℝ :=
  sorry

theorem spiral_similarity_composition
  (P₁ P₂ : SpiralSimilarity)
  (h₁ : P₁.angle = P₂.angle)
  (h₂ : P₁.coefficient * P₂.coefficient = 1)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (hN : N = sorry) -- N = P₁(M)
  : 
  let P := compose P₂ P₁
  ∃ (R : Rotation), 
    isRotation P ∧ 
    P.center = R.center ∧
    R.center.fst = P₁.center.fst ∧ R.center.snd = P₂.center.snd ∧
    R.angle = 2 * vectorAngle (M.fst - P₁.center.fst, M.snd - P₁.center.snd) (N.fst - M.fst, N.snd - M.snd) :=
sorry

end NUMINAMATH_CALUDE_spiral_similarity_composition_l654_65405


namespace NUMINAMATH_CALUDE_initial_pigs_count_l654_65423

theorem initial_pigs_count (initial_cows initial_goats added_cows added_pigs added_goats total_after : ℕ) :
  initial_cows = 2 →
  initial_goats = 6 →
  added_cows = 3 →
  added_pigs = 5 →
  added_goats = 2 →
  total_after = 21 →
  ∃ initial_pigs : ℕ, 
    initial_cows + initial_pigs + initial_goats + added_cows + added_pigs + added_goats = total_after ∧
    initial_pigs = 3 :=
by sorry

end NUMINAMATH_CALUDE_initial_pigs_count_l654_65423


namespace NUMINAMATH_CALUDE_triangle_properties_l654_65487

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Define the property that a^2 + b^2 - c^2 = ab -/
def satisfiesProperty (t : Triangle) : Prop :=
  t.a^2 + t.b^2 - t.c^2 = t.a * t.b

/-- Define the collinearity of vectors (2sin A, 1) and (cos C, 1/2) -/
def vectorsAreCollinear (t : Triangle) : Prop :=
  2 * Real.sin t.A * (1/2) = Real.cos t.C * 1

/-- Theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : satisfiesProperty t) 
  (h2 : vectorsAreCollinear t) : 
  t.C = π/3 ∧ t.B = π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l654_65487


namespace NUMINAMATH_CALUDE_gcd_153_68_l654_65445

theorem gcd_153_68 : Nat.gcd 153 68 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_153_68_l654_65445


namespace NUMINAMATH_CALUDE_bet_is_unfair_a_has_advantage_l654_65442

/-- Represents the outcome of rolling two dice -/
def DiceRoll := Fin 6 × Fin 6

/-- The probability of A winning (sum < 8) -/
def probAWins : ℚ := 7/12

/-- The probability of B winning (sum ≥ 8) -/
def probBWins : ℚ := 5/12

/-- A's bet amount in forints -/
def aBet : ℚ := 10

/-- B's bet amount in forints -/
def bBet : ℚ := 8

/-- Expected gain for A in forints -/
def expectedGainA : ℚ := bBet * probAWins - aBet * probBWins

theorem bet_is_unfair : expectedGainA = 1/2 := by sorry

theorem a_has_advantage : expectedGainA > 0 := by sorry

end NUMINAMATH_CALUDE_bet_is_unfair_a_has_advantage_l654_65442


namespace NUMINAMATH_CALUDE_intersection_point_l654_65452

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The equation of the first line: 2x + y + 2 = 0 -/
def line1 (x y : ℝ) : Prop := 2 * x + y + 2 = 0

/-- The equation of the second line: ax + 4y - 2 = 0 -/
def line2 (a x y : ℝ) : Prop := a * x + 4 * y - 2 = 0

/-- The theorem stating that the intersection point of the two perpendicular lines is (-1, 0) -/
theorem intersection_point :
  ∃ (a : ℝ),
    (∀ x y : ℝ, perpendicular (-2) (-a/4)) →
    (∀ x y : ℝ, line1 x y ∧ line2 a x y → x = -1 ∧ y = 0) :=
by sorry


end NUMINAMATH_CALUDE_intersection_point_l654_65452


namespace NUMINAMATH_CALUDE_dogs_and_movies_percentage_l654_65474

theorem dogs_and_movies_percentage
  (total_students : ℕ)
  (dogs_and_games_percentage : ℚ)
  (dogs_preference : ℕ)
  (h1 : total_students = 30)
  (h2 : dogs_and_games_percentage = 1/2)
  (h3 : dogs_preference = 18) :
  (dogs_preference - (dogs_and_games_percentage * total_students)) / total_students = 1/10 :=
sorry

end NUMINAMATH_CALUDE_dogs_and_movies_percentage_l654_65474


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l654_65413

/-- Given a line L1 with equation 2x-y+3=0 and a point P(1,1), 
    the line L2 passing through P and perpendicular to L1 
    has the equation x+2y-3=0 -/
theorem perpendicular_line_equation : 
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 2*x - y + 3 = 0
  let P : ℝ × ℝ := (1, 1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ x + 2*y - 3 = 0
  (∀ x y, L2 x y ↔ (y - P.2 = -(1/2) * (x - P.1))) ∧ 
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → L2 x₁ y₁ → L2 x₂ y₂ → 
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) = 0) ∧
  L2 P.1 P.2 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l654_65413


namespace NUMINAMATH_CALUDE_evaluate_expression_l654_65400

theorem evaluate_expression : (0.5 ^ 4) / (0.05 ^ 3) = 500 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l654_65400


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l654_65475

/-- A composite figure made of squares and triangles -/
structure CompositeFigure where
  squareSideLength : ℝ
  triangleSideLength : ℝ
  numSquares : ℕ
  numTriangles : ℕ

/-- Calculate the perimeter of the composite figure -/
def perimeter (figure : CompositeFigure) : ℝ :=
  let squareContribution := 2 * figure.squareSideLength * (figure.numSquares + 2)
  let triangleContribution := figure.triangleSideLength * figure.numTriangles
  squareContribution + triangleContribution

/-- Theorem: The perimeter of the specific composite figure is 17 -/
theorem specific_figure_perimeter :
  let figure : CompositeFigure :=
    { squareSideLength := 2
      triangleSideLength := 1
      numSquares := 4
      numTriangles := 3 }
  perimeter figure = 17 := by
  sorry

end NUMINAMATH_CALUDE_specific_figure_perimeter_l654_65475


namespace NUMINAMATH_CALUDE_hat_promotion_savings_l654_65439

/-- Calculates the percentage saved when buying three hats under a promotional offer --/
theorem hat_promotion_savings : 
  let regular_price : ℝ := 60
  let discount_second : ℝ := 0.25
  let discount_third : ℝ := 0.35
  let total_regular : ℝ := 3 * regular_price
  let price_first : ℝ := regular_price
  let price_second : ℝ := regular_price * (1 - discount_second)
  let price_third : ℝ := regular_price * (1 - discount_third)
  let total_discounted : ℝ := price_first + price_second + price_third
  let savings : ℝ := total_regular - total_discounted
  let percentage_saved : ℝ := (savings / total_regular) * 100
  percentage_saved = 20 := by
  sorry


end NUMINAMATH_CALUDE_hat_promotion_savings_l654_65439


namespace NUMINAMATH_CALUDE_complex_subtraction_l654_65466

theorem complex_subtraction (i : ℂ) (h : i * i = -1) :
  let z₁ : ℂ := 3 + 4 * i
  let z₂ : ℂ := 1 + 2 * i
  z₁ - z₂ = 2 + 2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l654_65466


namespace NUMINAMATH_CALUDE_cubic_sum_zero_l654_65483

theorem cubic_sum_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum : (a^2 / (b - c)^2) + (b^2 / (c - a)^2) + (c^2 / (a - b)^2) = 0) :
  (a^3 / (b - c)^3) + (b^3 / (c - a)^3) + (c^3 / (a - b)^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_zero_l654_65483


namespace NUMINAMATH_CALUDE_amusement_park_problem_l654_65496

/-- The number of children who got on the Ferris wheel -/
def ferris_wheel_riders : ℕ := sorry

theorem amusement_park_problem :
  let total_children : ℕ := 5
  let ferris_wheel_cost : ℕ := 5
  let merry_go_round_cost : ℕ := 3
  let ice_cream_cost : ℕ := 8
  let ice_cream_per_child : ℕ := 2
  let total_spent : ℕ := 110
  ferris_wheel_riders * ferris_wheel_cost +
  total_children * merry_go_round_cost +
  total_children * ice_cream_per_child * ice_cream_cost = total_spent ∧
  ferris_wheel_riders = 3 :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_problem_l654_65496


namespace NUMINAMATH_CALUDE_C_power_50_l654_65456

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

theorem C_power_50 : C^50 = !![101, 50; -200, -99] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l654_65456


namespace NUMINAMATH_CALUDE_a_values_theorem_l654_65459

theorem a_values_theorem (a b x : ℝ) (h1 : a - b = x) (h2 : x ≠ 0) (h3 : a^3 - b^3 = 19*x^3) :
  a = 3*x ∨ a = -2*x :=
by sorry

end NUMINAMATH_CALUDE_a_values_theorem_l654_65459


namespace NUMINAMATH_CALUDE_mehki_age_proof_l654_65437

/-- Proves that Mehki's age is 16 years old given the specified conditions -/
theorem mehki_age_proof (zrinka_age jordyn_age mehki_age : ℕ) : 
  zrinka_age = 6 →
  jordyn_age = zrinka_age - 4 →
  mehki_age = 2 * (jordyn_age + zrinka_age) →
  mehki_age = 16 := by
sorry

end NUMINAMATH_CALUDE_mehki_age_proof_l654_65437


namespace NUMINAMATH_CALUDE_line_through_point_with_opposite_intercepts_l654_65455

theorem line_through_point_with_opposite_intercepts :
  ∃ (m c : ℝ), 
    (∀ x y : ℝ, y = m * x + c ↔ 
      (x = 1 ∧ y = 3) ∨ 
      (∃ a : ℝ, (x = a ∧ y = 0) ∨ (x = 0 ∧ y = -a))) →
    m = 1 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_with_opposite_intercepts_l654_65455


namespace NUMINAMATH_CALUDE_mike_changes_64_tires_l654_65457

/-- The number of tires on a motorcycle -/
def motorcycle_tires : ℕ := 2

/-- The number of tires on a car -/
def car_tires : ℕ := 4

/-- The number of motorcycles Mike changes tires on -/
def num_motorcycles : ℕ := 12

/-- The number of cars Mike changes tires on -/
def num_cars : ℕ := 10

/-- The total number of tires Mike changes -/
def total_tires : ℕ := num_motorcycles * motorcycle_tires + num_cars * car_tires

theorem mike_changes_64_tires : total_tires = 64 := by
  sorry

end NUMINAMATH_CALUDE_mike_changes_64_tires_l654_65457


namespace NUMINAMATH_CALUDE_project_selection_count_l654_65434

theorem project_selection_count : ∀ (n_key m_key n_general m_general : ℕ),
  n_key = 4 →
  m_key = 2 →
  n_general = 6 →
  m_general = 2 →
  (Nat.choose n_key m_key * Nat.choose (n_general - 1) (m_general - 1)) +
  (Nat.choose (n_key - 1) (m_key - 1) * Nat.choose n_general m_general) -
  (Nat.choose (n_key - 1) (m_key - 1) * Nat.choose (n_general - 1) (m_general - 1)) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_project_selection_count_l654_65434


namespace NUMINAMATH_CALUDE_complex_points_on_circle_l654_65453

theorem complex_points_on_circle 
  (a₁ a₂ a₃ a₄ a₅ : ℂ) 
  (h_nonzero : a₁ ≠ 0 ∧ a₂ ≠ 0 ∧ a₃ ≠ 0 ∧ a₄ ≠ 0 ∧ a₅ ≠ 0)
  (h_ratio : a₂ / a₁ = a₃ / a₂ ∧ a₃ / a₂ = a₄ / a₃ ∧ a₄ / a₃ = a₅ / a₄)
  (S : ℝ)
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 4 * (1 / a₁ + 1 / a₂ + 1 / a₃ + 1 / a₄ + 1 / a₅))
  (h_S_real : a₁ + a₂ + a₃ + a₄ + a₅ = S)
  (h_S_bound : abs S ≤ 2) :
  ∃ (r : ℝ), r > 0 ∧ Complex.abs a₁ = r ∧ Complex.abs a₂ = r ∧ 
             Complex.abs a₃ = r ∧ Complex.abs a₄ = r ∧ Complex.abs a₅ = r := by
  sorry

end NUMINAMATH_CALUDE_complex_points_on_circle_l654_65453


namespace NUMINAMATH_CALUDE_point_c_transformation_l654_65417

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate (p : ℝ × ℝ) (t : ℝ × ℝ) : ℝ × ℝ := (p.1 + t.1, p.2 + t.2)

theorem point_c_transformation :
  let c : ℝ × ℝ := (3, 3)
  let c' := translate (reflect_x (reflect_y c)) (3, -4)
  c' = (0, -7) := by sorry

end NUMINAMATH_CALUDE_point_c_transformation_l654_65417


namespace NUMINAMATH_CALUDE_max_value_x_minus_2z_l654_65469

theorem max_value_x_minus_2z (x y z : ℝ) :
  x^2 + y^2 + z^2 = 16 →
  ∃ (max : ℝ), max = 4 * Real.sqrt 5 ∧ ∀ (x' y' z' : ℝ), x'^2 + y'^2 + z'^2 = 16 → x' - 2*z' ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_x_minus_2z_l654_65469


namespace NUMINAMATH_CALUDE_allen_reading_speed_l654_65470

/-- The number of pages in Allen's book -/
def total_pages : ℕ := 120

/-- The number of days Allen took to read the book -/
def days_to_read : ℕ := 12

/-- The number of pages Allen read per day -/
def pages_per_day : ℕ := total_pages / days_to_read

/-- Theorem stating that Allen read 10 pages per day -/
theorem allen_reading_speed : pages_per_day = 10 := by
  sorry

end NUMINAMATH_CALUDE_allen_reading_speed_l654_65470


namespace NUMINAMATH_CALUDE_function_equality_implies_a_value_l654_65450

/-- The function f(x) = x -/
def f (x : ℝ) : ℝ := x

/-- The function g(x) = ax^2 - x, parameterized by a -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x

/-- The theorem stating that under given conditions, a = 3/2 -/
theorem function_equality_implies_a_value :
  ∀ (a : ℝ), a > 0 →
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ * f x₂ = g a x₁ * g a x₂) →
  a = 3/2 := by sorry

end NUMINAMATH_CALUDE_function_equality_implies_a_value_l654_65450


namespace NUMINAMATH_CALUDE_stone_bucket_probability_l654_65468

/-- The probability of having exactly k stones in the bucket after n seconds -/
def f (n k : ℕ) : ℚ :=
  (↑(Nat.floor ((n - k : ℤ) / 2)) : ℚ) / 2^n

/-- The main theorem stating the probability of having 1337 stones after 2017 seconds -/
theorem stone_bucket_probability : f 2017 1337 = 340 / 2^2017 := by sorry

end NUMINAMATH_CALUDE_stone_bucket_probability_l654_65468


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l654_65441

/-- Represents the total number of handshakes at the event -/
def total_handshakes : ℕ := 300

/-- Calculates the number of athlete handshakes given the total number of athletes -/
def athlete_handshakes (n : ℕ) : ℕ := (3 * n * n) / 4

/-- Calculates the number of coach handshakes given the total number of athletes -/
def coach_handshakes (n : ℕ) : ℕ := n

/-- Theorem stating the minimum number of coach handshakes -/
theorem min_coach_handshakes :
  ∃ n : ℕ, 
    athlete_handshakes n + coach_handshakes n = total_handshakes ∧
    coach_handshakes n = 20 ∧
    ∀ m : ℕ, 
      athlete_handshakes m + coach_handshakes m = total_handshakes →
      coach_handshakes m ≥ coach_handshakes n :=
by sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l654_65441


namespace NUMINAMATH_CALUDE_shopkeeper_milk_packets_l654_65426

/-- Proves that the shopkeeper bought 150 packets of milk given the conditions -/
theorem shopkeeper_milk_packets 
  (packet_volume : ℕ) 
  (ounce_to_ml : ℕ) 
  (total_ounces : ℕ) 
  (h1 : packet_volume = 250)
  (h2 : ounce_to_ml = 30)
  (h3 : total_ounces = 1250) :
  (total_ounces * ounce_to_ml) / packet_volume = 150 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_milk_packets_l654_65426


namespace NUMINAMATH_CALUDE_research_institute_reward_allocation_l654_65406

theorem research_institute_reward_allocation :
  let n : ℕ := 10
  let a₁ : ℚ := 2
  let r : ℚ := 2
  let S := (a₁ * (1 - r^n)) / (1 - r)
  S = 2046 := by
sorry

end NUMINAMATH_CALUDE_research_institute_reward_allocation_l654_65406


namespace NUMINAMATH_CALUDE_debate_team_boys_l654_65495

theorem debate_team_boys (girls : ℕ) (groups : ℕ) (group_size : ℕ) (boys : ℕ) : 
  girls = 45 →
  groups = 8 →
  group_size = 7 →
  groups * group_size = girls + boys →
  boys = 11 := by
sorry

end NUMINAMATH_CALUDE_debate_team_boys_l654_65495


namespace NUMINAMATH_CALUDE_decagon_sign_change_impossible_l654_65499

/-- Represents a point in the decagon where a number is placed -/
structure Point where
  value : Int
  is_vertex : Bool
  is_intersection : Bool

/-- Represents the decagon configuration -/
structure Decagon where
  points : List Point
  
/-- Represents an operation that can be performed on the decagon -/
inductive Operation
  | FlipSide : Nat → Operation  -- Flip signs along the nth side
  | FlipDiagonal : Nat → Operation  -- Flip signs along the nth diagonal

/-- Applies an operation to the decagon -/
def apply_operation (d : Decagon) (op : Operation) : Decagon :=
  sorry

/-- Checks if all points in the decagon have negative values -/
def all_negative (d : Decagon) : Bool :=
  sorry

/-- Initial setup of the decagon with all +1 values -/
def initial_decagon : Decagon :=
  sorry

theorem decagon_sign_change_impossible :
  ∀ (ops : List Operation),
    ¬(all_negative (ops.foldl apply_operation initial_decagon)) :=
  sorry

end NUMINAMATH_CALUDE_decagon_sign_change_impossible_l654_65499


namespace NUMINAMATH_CALUDE_triangle_on_bottom_l654_65478

/-- Represents the positions of faces on a cube -/
inductive CubeFace
  | Top
  | Bottom
  | East
  | South
  | West
  | North

/-- Represents the flattened cube configuration -/
structure FlattenedCube where
  faces : List CubeFace
  triangle_position : CubeFace

/-- The specific flattened cube configuration from the problem -/
def problem_cube : FlattenedCube := sorry

/-- Theorem stating that the triangle is on the bottom face in the given configuration -/
theorem triangle_on_bottom (c : FlattenedCube) : c.triangle_position = CubeFace.Bottom := by
  sorry

end NUMINAMATH_CALUDE_triangle_on_bottom_l654_65478


namespace NUMINAMATH_CALUDE_like_terms_imply_m_and_n_l654_65476

/-- Two algebraic expressions are like terms if their variables have the same base and exponents -/
def are_like_terms (expr1 expr2 : ℕ → ℕ → ℝ) : Prop :=
  ∀ x y, ∃ c1 c2 : ℝ, expr1 x y = c1 * (x^(expr1 1 0) * y^(expr1 0 1)) ∧
                      expr2 x y = c2 * (x^(expr2 1 0) * y^(expr2 0 1)) ∧
                      expr1 1 0 = expr2 1 0 ∧
                      expr1 0 1 = expr2 0 1

theorem like_terms_imply_m_and_n (m n : ℕ) :
  are_like_terms (λ x y => -3 * x^(m-1) * y^3) (λ x y => 4 * x * y^(m+n)) →
  m = 2 ∧ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_imply_m_and_n_l654_65476


namespace NUMINAMATH_CALUDE_divisor_problem_l654_65430

theorem divisor_problem (n : ℕ) (h : n = 1101) : 
  ∃ (d : ℕ), d > 1 ∧ (n + 3) % d = 0 ∧ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l654_65430


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l654_65498

-- Define the operation ⊗
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem bowtie_equation_solution :
  ∃ y : ℝ, bowtie 5 y = 12 → y = 42 :=
by sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l654_65498


namespace NUMINAMATH_CALUDE_arrangements_theorem_l654_65480

-- Define the number of officers and intersections
def num_officers : ℕ := 5
def num_intersections : ℕ := 3

-- Define the function to calculate the number of arrangements
def arrangements_with_AB_together : ℕ := sorry

-- State the theorem
theorem arrangements_theorem : arrangements_with_AB_together = 36 := by sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l654_65480


namespace NUMINAMATH_CALUDE_difference_of_squares_divisible_by_eight_l654_65447

theorem difference_of_squares_divisible_by_eight (a b : ℤ) (h : a > b) :
  ∃ k : ℤ, 4 * (a - b) * (a + b + 1) = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_divisible_by_eight_l654_65447


namespace NUMINAMATH_CALUDE_hyperbola_equation_l654_65494

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the right focus
def right_focus (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the asymptote
def asymptote (a b : ℝ) (x y : ℝ) : Prop :=
  y = b / a * x ∨ y = -b / a * x

-- Define an equilateral triangle
def equilateral_triangle (A B C : ℝ × ℝ) (side_length : ℝ) : Prop :=
  dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

theorem hyperbola_equation (a b : ℝ) (F A : ℝ × ℝ) :
  a > 0 → b > 0 →
  hyperbola a b F.1 F.2 →
  F = right_focus a →
  asymptote a b A.1 A.2 →
  equilateral_triangle (0, 0) F A 2 →
  ∃ (x y : ℝ), x^2 - y^2 / 3 = 1 := by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l654_65494


namespace NUMINAMATH_CALUDE_work_completion_time_l654_65416

/-- Represents the amount of work one man can do in one day -/
def man_work : ℝ := sorry

/-- Represents the amount of work one boy can do in one day -/
def boy_work : ℝ := sorry

/-- The number of days it takes 6 men and 8 boys to complete the work -/
def x : ℝ := sorry

theorem work_completion_time :
  (6 * man_work + 8 * boy_work) * x = (26 * man_work + 48 * boy_work) * 2 ∧
  (6 * man_work + 8 * boy_work) * x = (15 * man_work + 20 * boy_work) * 4 →
  x = 5 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l654_65416


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l654_65433

-- Define the equations
def equation1 (x : ℝ) : Prop := (x + 8) * (x + 1) = -12
def equation2 (x : ℝ) : Prop := 2 * x^2 + 4 * x - 1 = 0

-- Theorem for equation 1
theorem solution_equation1 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ ∧ equation1 x₂ ∧ x₁ = -4 ∧ x₂ = -5 :=
by sorry

-- Theorem for equation 2
theorem solution_equation2 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation2 x₁ ∧ equation2 x₂ ∧
  x₁ = (-2 + Real.sqrt 6) / 2 ∧ x₂ = (-2 - Real.sqrt 6) / 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l654_65433


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l654_65409

theorem circle_diameter_ratio (D C : ℝ → Prop) (r_D r_C : ℝ) : 
  (∀ x, C x → D x) →  -- C is inside D
  (2 * r_D = 20) →    -- Diameter of D is 20 cm
  (π * r_D^2 - π * r_C^2 = 2 * π * r_C^2) →  -- Ratio of shaded area to area of C is 2:1
  2 * r_C = 20 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l654_65409


namespace NUMINAMATH_CALUDE_range_of_a_l654_65462

-- Define the conditions
def p (x a : ℝ) : Prop := |x - a| < 4
def q (x : ℝ) : Prop := -x^2 + 5*x - 6 > 0

-- Define the theorem
theorem range_of_a :
  ∃ (a_min a_max : ℝ),
    (a_min = -1 ∧ a_max = 6) ∧
    (∀ a : ℝ, (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) ↔ a_min ≤ a ∧ a ≤ a_max) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l654_65462


namespace NUMINAMATH_CALUDE_simple_interest_period_l654_65493

theorem simple_interest_period (P : ℝ) : 
  (P * 4 * 5 / 100 = 1680) → 
  (P * 5 * 4 / 100 = 1680) → 
  ∃ T : ℝ, T = 5 ∧ P * 4 * T / 100 = 1680 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_period_l654_65493


namespace NUMINAMATH_CALUDE_ninas_allowance_l654_65497

theorem ninas_allowance (game_cost : ℝ) (tax_rate : ℝ) (savings_rate : ℝ) (weeks : ℕ) :
  game_cost = 50 →
  tax_rate = 0.1 →
  savings_rate = 0.5 →
  weeks = 11 →
  ∃ (allowance : ℝ),
    allowance * savings_rate * weeks = game_cost * (1 + tax_rate) ∧
    allowance = 10 := by
  sorry

end NUMINAMATH_CALUDE_ninas_allowance_l654_65497


namespace NUMINAMATH_CALUDE_no_reciprocal_sum_equals_sum_reciprocals_l654_65432

theorem no_reciprocal_sum_equals_sum_reciprocals :
  ¬∃ (x y : ℝ), x ≠ -y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (1 / (x + y) = 1 / x + 1 / y) := by
  sorry

end NUMINAMATH_CALUDE_no_reciprocal_sum_equals_sum_reciprocals_l654_65432


namespace NUMINAMATH_CALUDE_garden_border_rocks_l654_65404

theorem garden_border_rocks (rocks_placed : Float) (additional_rocks : Float) : 
  rocks_placed = 125.0 → additional_rocks = 64.0 → rocks_placed + additional_rocks = 189.0 := by
  sorry

end NUMINAMATH_CALUDE_garden_border_rocks_l654_65404


namespace NUMINAMATH_CALUDE_expression_value_l654_65415

theorem expression_value (p q : ℝ) : 
  (∃ x : ℝ, x = 3 ∧ p * x^3 + q * x - 1 = 13) → 
  (∃ y : ℝ, y = -3 ∧ p * y^3 + q * y - 1 = -15) :=
sorry

end NUMINAMATH_CALUDE_expression_value_l654_65415


namespace NUMINAMATH_CALUDE_kramers_packing_rate_l654_65436

/-- Kramer's cigarette packing rate -/
theorem kramers_packing_rate 
  (boxes_per_case : ℕ) 
  (cases_packed : ℕ) 
  (packing_time_hours : ℕ) 
  (h1 : boxes_per_case = 5)
  (h2 : cases_packed = 240)
  (h3 : packing_time_hours = 2) :
  (boxes_per_case * cases_packed) / (packing_time_hours * 60) = 10 := by
  sorry

#check kramers_packing_rate

end NUMINAMATH_CALUDE_kramers_packing_rate_l654_65436


namespace NUMINAMATH_CALUDE_sculpture_cost_in_cny_l654_65472

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℝ := 5

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℝ := 8

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 200

/-- Theorem stating the cost of the sculpture in Chinese yuan -/
theorem sculpture_cost_in_cny :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 320 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_cny_l654_65472


namespace NUMINAMATH_CALUDE_exchange_probability_l654_65481

/-- Represents the colors of balls -/
inductive Color
  | Red | Green | Yellow | Violet | Black | Orange

/-- Represents a bag of balls -/
def Bag := List Color

/-- Initial configuration of Arjun's bag -/
def arjunInitialBag : Bag :=
  [Color.Red, Color.Red, Color.Green, Color.Yellow, Color.Violet]

/-- Initial configuration of Becca's bag -/
def beccaInitialBag : Bag :=
  [Color.Black, Color.Black, Color.Orange]

/-- Represents the exchange process -/
def exchange (bag1 bag2 : Bag) : Bag × Bag :=
  sorry

/-- Checks if a bag has exactly 3 different colors -/
def hasThreeColors (bag : Bag) : Bool :=
  sorry

/-- Calculates the probability of the final configuration -/
def finalProbability (arjunBag beccaBag : Bag) : ℚ :=
  sorry

/-- The main theorem to be proved -/
theorem exchange_probability :
  finalProbability arjunInitialBag beccaInitialBag = 3/10 :=
sorry

end NUMINAMATH_CALUDE_exchange_probability_l654_65481


namespace NUMINAMATH_CALUDE_function_zero_l654_65489

theorem function_zero (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = -f x) 
  (h2 : ∀ x, f (-x) = f x) : 
  ∀ x, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_function_zero_l654_65489


namespace NUMINAMATH_CALUDE_point_movement_l654_65463

/-- The possible final positions of a point that starts 3 units from the origin,
    moves 4 units right, and then 1 unit left. -/
def final_positions : Set ℤ :=
  {0, 6}

/-- The theorem stating the possible final positions of the point. -/
theorem point_movement (A : ℤ) : 
  (abs A = 3) → 
  ((A + 4 - 1) ∈ final_positions) :=
by sorry

end NUMINAMATH_CALUDE_point_movement_l654_65463
