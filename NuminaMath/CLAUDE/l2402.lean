import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l2402_240228

theorem system_solution :
  ∃! (X Y Z : ℝ),
    0.15 * 40 = 0.25 * X + 2 ∧
    0.30 * 60 = 0.20 * Y + 3 ∧
    0.10 * Z = X - Y ∧
    X = 16 ∧ Y = 75 ∧ Z = -590 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2402_240228


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2402_240299

theorem complex_modulus_problem (z : ℂ) : z = (Complex.I - 2) / (1 + Complex.I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2402_240299


namespace NUMINAMATH_CALUDE_f_one_root_m_range_l2402_240206

/-- A cubic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

/-- The theorem stating the range of m for which f has exactly one real root -/
theorem f_one_root_m_range (m : ℝ) :
  (∃! x, f m x = 0) ↔ m < -2 ∨ m > 2 := by
  sorry

end NUMINAMATH_CALUDE_f_one_root_m_range_l2402_240206


namespace NUMINAMATH_CALUDE_tinas_pens_l2402_240208

theorem tinas_pens (pink green blue : ℕ) : 
  pink = 12 ∧ 
  green = pink - 9 ∧ 
  blue = green + 3 → 
  pink + green + blue = 21 := by
  sorry

end NUMINAMATH_CALUDE_tinas_pens_l2402_240208


namespace NUMINAMATH_CALUDE_table_runner_coverage_l2402_240281

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) 
  (coverage_percentage : ℝ) (two_layer_area : ℝ) : 
  total_runner_area = 208 →
  table_area = 175 →
  coverage_percentage = 0.8 →
  two_layer_area = 24 →
  ∃ (three_layer_area : ℝ),
    three_layer_area = 22 ∧
    total_runner_area = (coverage_percentage * table_area - two_layer_area - three_layer_area) +
                        2 * two_layer_area +
                        3 * three_layer_area :=
by sorry

end NUMINAMATH_CALUDE_table_runner_coverage_l2402_240281


namespace NUMINAMATH_CALUDE_min_C_over_D_l2402_240219

theorem min_C_over_D (x C D : ℝ) (hx : x ≠ 0) 
  (hC : x^3 + 1/x^3 = C) (hD : x - 1/x = D) :
  ∀ y : ℝ, y ≠ 0 → y^3 + 1/y^3 / (y - 1/y) ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_C_over_D_l2402_240219


namespace NUMINAMATH_CALUDE_root_existence_l2402_240246

theorem root_existence : ∃ x : ℝ, 3 < x ∧ x < 4 ∧ Real.log x = 8 - 2 * x := by
  sorry

end NUMINAMATH_CALUDE_root_existence_l2402_240246


namespace NUMINAMATH_CALUDE_cube_vertex_configurations_l2402_240293

/-- Represents a vertex of a cube -/
inductive CubeVertex
  | A | B | C | D | A1 | B1 | C1 | D1

/-- Represents a set of 4 vertices from a cube -/
def VertexSet := Finset CubeVertex

/-- Checks if a set of vertices forms a rectangle -/
def is_rectangle (vs : VertexSet) : Prop := sorry

/-- Checks if a set of vertices forms a tetrahedron with all equilateral triangle faces -/
def is_equilateral_tetrahedron (vs : VertexSet) : Prop := sorry

/-- Checks if a set of vertices forms a tetrahedron with all right triangle faces -/
def is_right_tetrahedron (vs : VertexSet) : Prop := sorry

/-- Checks if a set of vertices forms a tetrahedron with three isosceles right triangle faces and one equilateral triangle face -/
def is_mixed_tetrahedron (vs : VertexSet) : Prop := sorry

theorem cube_vertex_configurations :
  ∃ (vs1 vs2 vs3 vs4 : VertexSet),
    is_rectangle vs1 ∧
    is_equilateral_tetrahedron vs2 ∧
    is_right_tetrahedron vs3 ∧
    is_mixed_tetrahedron vs4 :=
  sorry

end NUMINAMATH_CALUDE_cube_vertex_configurations_l2402_240293


namespace NUMINAMATH_CALUDE_power_equality_l2402_240248

theorem power_equality (p : ℕ) (h : (81 : ℕ)^6 = 3^p) : p = 24 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2402_240248


namespace NUMINAMATH_CALUDE_equation_solution_l2402_240210

theorem equation_solution (y : ℝ) : 
  ∃ x : ℝ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 ∧ x = -21/38 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2402_240210


namespace NUMINAMATH_CALUDE_gcd_8321_6489_l2402_240266

theorem gcd_8321_6489 : Nat.gcd 8321 6489 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8321_6489_l2402_240266


namespace NUMINAMATH_CALUDE_ratio_PC_PB_is_zero_l2402_240232

/-- A square with side length 6, where N is the midpoint of AB and P is the intersection of BD and CN -/
structure SquareABCD where
  -- Define the square
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Conditions
  is_square : A = (0, 0) ∧ B = (6, 0) ∧ C = (6, 6) ∧ D = (0, 6)
  -- Define N as midpoint of AB
  N : ℝ × ℝ
  N_is_midpoint : N = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  -- Define P as intersection of BD and CN
  P : ℝ × ℝ
  P_on_BD : (P.2 - D.2) = ((B.2 - D.2) / (B.1 - D.1)) * (P.1 - D.1)
  P_on_CN : (P.2 - N.2) = ((C.2 - N.2) / (C.1 - N.1)) * (P.1 - N.1)

/-- The ratio of PC to PB is 0 -/
theorem ratio_PC_PB_is_zero (square : SquareABCD) : 
  let PC := Real.sqrt ((square.P.1 - square.C.1)^2 + (square.P.2 - square.C.2)^2)
  let PB := Real.sqrt ((square.P.1 - square.B.1)^2 + (square.P.2 - square.B.2)^2)
  PC / PB = 0 := by
  sorry

end NUMINAMATH_CALUDE_ratio_PC_PB_is_zero_l2402_240232


namespace NUMINAMATH_CALUDE_min_value_f_max_value_y_l2402_240226

/-- The minimum value of f(x) = 4/x + x for x > 0 is 4 -/
theorem min_value_f (x : ℝ) (hx : x > 0) :
  (4 / x + x) ≥ 4 ∧ ∃ x₀ > 0, 4 / x₀ + x₀ = 4 := by sorry

/-- The maximum value of y = x(1 - 3x) for 0 < x < 1/3 is 1/12 -/
theorem max_value_y (x : ℝ) (hx1 : x > 0) (hx2 : x < 1/3) :
  x * (1 - 3 * x) ≤ 1/12 ∧ ∃ x₀ ∈ (Set.Ioo 0 (1/3)), x₀ * (1 - 3 * x₀) = 1/12 := by sorry

end NUMINAMATH_CALUDE_min_value_f_max_value_y_l2402_240226


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l2402_240279

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l2402_240279


namespace NUMINAMATH_CALUDE_fewer_spoons_purchased_l2402_240296

/-- The number of types of silverware --/
def numTypes : ℕ := 4

/-- The initially planned number of pieces per type --/
def initialPerType : ℕ := 15

/-- The total number of pieces actually purchased --/
def actualTotal : ℕ := 44

/-- Theorem stating that the number of fewer spoons purchased is 4 --/
theorem fewer_spoons_purchased :
  (numTypes * initialPerType - actualTotal) / numTypes = 4 := by
  sorry

end NUMINAMATH_CALUDE_fewer_spoons_purchased_l2402_240296


namespace NUMINAMATH_CALUDE_triangle_area_l2402_240253

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c^2 = (a - b)^2 + 6 →
  C = π / 3 →
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2402_240253


namespace NUMINAMATH_CALUDE_A_2022_coordinates_l2402_240269

/-- The companion point transformation --/
def companion_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2) + 1, p.1 + 1)

/-- The sequence of points starting from A1 --/
def A : ℕ → ℝ × ℝ
  | 0 => (2, 4)
  | n + 1 => companion_point (A n)

/-- The main theorem --/
theorem A_2022_coordinates :
  A 2021 = (-3, 3) := by
  sorry

end NUMINAMATH_CALUDE_A_2022_coordinates_l2402_240269


namespace NUMINAMATH_CALUDE_A_equals_B_l2402_240297

def A : Set ℤ := {x | ∃ n : ℤ, x = 2*n - 1}
def B : Set ℤ := {x | ∃ n : ℤ, x = 2*n + 1}

theorem A_equals_B : A = B := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l2402_240297


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l2402_240254

/-- Given a square A with perimeter 24 cm and a square B with area equal to one-fourth the area of square A, prove that the perimeter of square B is 12 cm. -/
theorem square_perimeter_relation (A B : ℝ → ℝ → Prop) : 
  (∃ a, ∀ x y, A x y ↔ (x = 0 ∨ x = a) ∧ (y = 0 ∨ y = a) ∧ 4 * a = 24) →
  (∃ b, ∀ x y, B x y ↔ (x = 0 ∨ x = b) ∧ (y = 0 ∨ y = b) ∧ b^2 = (a^2 / 4)) →
  (∃ p, p = 4 * b ∧ p = 12) :=
sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l2402_240254


namespace NUMINAMATH_CALUDE_red_paint_amount_l2402_240209

/-- Given a paint mixture with a ratio of red to white as 5:7, 
    if 21 quarts of white paint are used, then 15 quarts of red paint should be used. -/
theorem red_paint_amount (red white : ℚ) : 
  (red / white = 5 / 7) → (white = 21) → (red = 15) := by
  sorry

end NUMINAMATH_CALUDE_red_paint_amount_l2402_240209


namespace NUMINAMATH_CALUDE_units_digit_of_3_power_2004_l2402_240288

theorem units_digit_of_3_power_2004 : 3^2004 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_3_power_2004_l2402_240288


namespace NUMINAMATH_CALUDE_initial_outlay_is_10000_l2402_240213

/-- Calculates the profit for a horseshoe manufacturing company --/
def horseshoe_profit (initial_outlay : ℝ) (sets_produced : ℕ) : ℝ :=
  let manufacturing_cost := initial_outlay + 20 * sets_produced
  let revenue := 50 * sets_produced
  revenue - manufacturing_cost

/-- Proves that the initial outlay is $10,000 given the conditions --/
theorem initial_outlay_is_10000 :
  ∃ (initial_outlay : ℝ),
    horseshoe_profit initial_outlay 500 = 5000 ∧
    initial_outlay = 10000 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_outlay_is_10000_l2402_240213


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2402_240230

/-- An odd function -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- An even function -/
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

/-- F(x) is increasing on (-∞, 0) -/
def increasing_on_neg (F : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → F x₁ < F x₂

theorem solution_set_equivalence (f g : ℝ → ℝ) 
  (hf : odd_function f) (hg : even_function g)
  (hF : increasing_on_neg (λ x => f x * g x))
  (hg2 : g 2 = 0) :
  {x : ℝ | f x * g x < 0} = {x : ℝ | x < -2 ∨ (0 < x ∧ x < 2)} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2402_240230


namespace NUMINAMATH_CALUDE_students_not_coming_to_class_l2402_240268

theorem students_not_coming_to_class (pieces_per_student : ℕ) 
  (total_pieces_last_monday : ℕ) (total_pieces_this_monday : ℕ) :
  pieces_per_student = 4 →
  total_pieces_last_monday = 40 →
  total_pieces_this_monday = 28 →
  total_pieces_last_monday / pieces_per_student - 
  total_pieces_this_monday / pieces_per_student = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_students_not_coming_to_class_l2402_240268


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l2402_240247

theorem sum_of_two_numbers (x y : ℕ) : y = x + 4 → y = 30 → x + y = 56 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l2402_240247


namespace NUMINAMATH_CALUDE_triangle_theorem_l2402_240251

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  let condition1 := 2 * a * Real.cos B = b + 2 * c
  let condition2 := 2 * Real.sin C + Real.tan A * Real.cos B + Real.sin B = 0
  let condition3 := (a - c) / Real.sin B = (b + c) / (Real.sin A + Real.sin C)
  b = 2 ∧ c = 4 ∧ 
  (condition1 ∨ condition2 ∨ condition3) →
  A = 2 * Real.pi / 3 ∧
  ∃ (D : ℝ × ℝ), 
    let BC := Real.sqrt ((b - c * Real.cos A)^2 + (c * Real.sin A)^2)
    let BD := BC / 4
    let AD := Real.sqrt (((3/4) * b)^2 + ((1/4) * c)^2 + 
               (3/4) * b * (1/4) * c * Real.cos A)
    AD = Real.sqrt 31 / 2

theorem triangle_theorem : 
  ∀ (a b c A B C : ℝ), triangle_problem a b c A B C :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2402_240251


namespace NUMINAMATH_CALUDE_quadratic_greater_than_zero_l2402_240229

theorem quadratic_greater_than_zero (x : ℝ) :
  (x + 2) * (x - 3) - 4 > 0 ↔ x < (1 - Real.sqrt 41) / 2 ∨ x > (1 + Real.sqrt 41) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_greater_than_zero_l2402_240229


namespace NUMINAMATH_CALUDE_new_person_age_l2402_240278

/-- Given a group of people with an initial average age and size, 
    calculate the age of a new person that changes the average to a new value. -/
theorem new_person_age (n : ℕ) (initial_avg new_avg : ℚ) : 
  n = 17 → 
  initial_avg = 14 → 
  new_avg = 15 → 
  (n : ℚ) * initial_avg + (new_avg * ((n : ℚ) + 1) - (n : ℚ) * initial_avg) = 32 := by
  sorry

#check new_person_age

end NUMINAMATH_CALUDE_new_person_age_l2402_240278


namespace NUMINAMATH_CALUDE_triangle_inequality_l2402_240286

theorem triangle_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_sum : a + b + c ≤ 2) : 
  -3 < (a^3/b + b^3/c + c^3/a - a^3/c - b^3/a - c^3/b) ∧
  (a^3/b + b^3/c + c^3/a - a^3/c - b^3/a - c^3/b) < 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2402_240286


namespace NUMINAMATH_CALUDE_total_feed_amount_l2402_240292

/-- The price per pound of the cheaper feed -/
def cheap_price : ℚ := 18 / 100

/-- The price per pound of the expensive feed -/
def expensive_price : ℚ := 53 / 100

/-- The desired price per pound of the mixed feed -/
def mixed_price : ℚ := 36 / 100

/-- The amount of cheaper feed used (in pounds) -/
def cheap_amount : ℚ := 17

/-- The theorem stating that the total amount of feed mixed is 35 pounds -/
theorem total_feed_amount : 
  ∃ (expensive_amount : ℚ),
    cheap_amount + expensive_amount = 35 ∧
    (cheap_amount * cheap_price + expensive_amount * expensive_price) / (cheap_amount + expensive_amount) = mixed_price :=
by sorry

end NUMINAMATH_CALUDE_total_feed_amount_l2402_240292


namespace NUMINAMATH_CALUDE_unique_prime_sum_diff_l2402_240203

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem unique_prime_sum_diff :
  ∃! p : ℕ, is_prime p ∧ 
    (∃ a b : ℕ, is_prime a ∧ is_prime b ∧ p = a + b) ∧
    (∃ c d : ℕ, is_prime c ∧ is_prime d ∧ p = c - d) :=
by
  use 5
  sorry

end NUMINAMATH_CALUDE_unique_prime_sum_diff_l2402_240203


namespace NUMINAMATH_CALUDE_range_of_a_l2402_240264

-- Define the set of real numbers where the expression is meaningful
def MeaningfulSet : Set ℝ :=
  {a : ℝ | a - 2 ≥ 0 ∧ a ≠ 4}

-- Theorem stating the range of values for a
theorem range_of_a : MeaningfulSet = Set.Icc 2 4 ∪ Set.Ioi 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2402_240264


namespace NUMINAMATH_CALUDE_no_natural_square_diff_2014_l2402_240220

theorem no_natural_square_diff_2014 : ¬∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_square_diff_2014_l2402_240220


namespace NUMINAMATH_CALUDE_sandwich_combinations_l2402_240222

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents the number of bread options that can go with a specific meat/cheese combination. -/
def num_bread_options : ℕ := 5

/-- Represents the number of restricted combinations (ham/cheddar and turkey/swiss). -/
def num_restricted_combinations : ℕ := 2

theorem sandwich_combinations :
  (num_breads * num_meats * num_cheeses) - (num_bread_options * num_restricted_combinations) = 200 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l2402_240222


namespace NUMINAMATH_CALUDE_exchange_three_cows_to_chickens_l2402_240212

/-- Exchange rates between animals -/
structure ExchangeRates where
  cows_to_sheep : ℚ      -- Rate of cows to sheep
  sheep_to_rabbits : ℚ   -- Rate of sheep to rabbits
  rabbits_to_chickens : ℚ -- Rate of rabbits to chickens

/-- Given the exchange rates, calculate how many chickens can be exchanged for a given number of cows -/
def cows_to_chickens (rates : ExchangeRates) (num_cows : ℚ) : ℚ :=
  num_cows * rates.cows_to_sheep * rates.sheep_to_rabbits * rates.rabbits_to_chickens

/-- Theorem stating that 3 cows can be exchanged for 819 chickens given the specified exchange rates -/
theorem exchange_three_cows_to_chickens :
  let rates : ExchangeRates := {
    cows_to_sheep := 42 / 2,
    sheep_to_rabbits := 26 / 3,
    rabbits_to_chickens := 3 / 2
  }
  cows_to_chickens rates 3 = 819 := by
  sorry


end NUMINAMATH_CALUDE_exchange_three_cows_to_chickens_l2402_240212


namespace NUMINAMATH_CALUDE_contrapositive_proof_l2402_240245

theorem contrapositive_proof : 
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔ 
  (∀ x : ℝ, x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_contrapositive_proof_l2402_240245


namespace NUMINAMATH_CALUDE_five_digit_square_number_l2402_240274

theorem five_digit_square_number : ∃! n : ℕ, 
  (n * n ≥ 10000) ∧ 
  (n * n < 100000) ∧ 
  (n * n / 10000 = 2) ∧ 
  ((n * n / 10) % 10 = 5) ∧ 
  (∃ m : ℕ, n * n = m * m) :=
by sorry

end NUMINAMATH_CALUDE_five_digit_square_number_l2402_240274


namespace NUMINAMATH_CALUDE_transformed_variance_l2402_240298

variable {n : ℕ}
variable (x : Fin n → ℝ)

def variance (x : Fin n → ℝ) : ℝ := sorry

theorem transformed_variance
  (h : variance x = 3) :
  variance (fun i => 2 * x i + 4) = 12 := by sorry

end NUMINAMATH_CALUDE_transformed_variance_l2402_240298


namespace NUMINAMATH_CALUDE_sum_powers_l2402_240249

theorem sum_powers (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  ¬(∀ (a b c d : ℝ), (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) := by
sorry

end NUMINAMATH_CALUDE_sum_powers_l2402_240249


namespace NUMINAMATH_CALUDE_smallest_angle_in_right_triangle_l2402_240255

theorem smallest_angle_in_right_triangle (α β γ : ℝ) : 
  α = 90 → β = 55 → α + β + γ = 180 → min α (min β γ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_right_triangle_l2402_240255


namespace NUMINAMATH_CALUDE_function_equality_implies_sum_greater_than_four_l2402_240265

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 2 / x

theorem function_equality_implies_sum_greater_than_four
  (x₁ x₂ : ℝ)
  (h₁ : x₁ > 0)
  (h₂ : x₂ > 0)
  (h₃ : x₁ ≠ x₂)
  (h₄ : f x₁ = f x₂) :
  x₁ + x₂ > 4 :=
by sorry

end NUMINAMATH_CALUDE_function_equality_implies_sum_greater_than_four_l2402_240265


namespace NUMINAMATH_CALUDE_positive_number_has_square_root_l2402_240214

theorem positive_number_has_square_root :
  ∀ x : ℝ, x > 0 → ∃ y : ℝ, y * y = x :=
by sorry

end NUMINAMATH_CALUDE_positive_number_has_square_root_l2402_240214


namespace NUMINAMATH_CALUDE_sum_of_roots_l2402_240261

theorem sum_of_roots (k d y₁ y₂ : ℝ) 
  (h₁ : y₁ ≠ y₂) 
  (h₂ : 4 * y₁^2 - k * y₁ = d) 
  (h₃ : 4 * y₂^2 - k * y₂ = d) : 
  y₁ + y₂ = k / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2402_240261


namespace NUMINAMATH_CALUDE_special_equation_result_l2402_240284

theorem special_equation_result (x : ℝ) (h : x + 1/x = Real.sqrt 7) :
  x^12 - 5*x^8 + 2*x^6 = 1944 * Real.sqrt 7 * x - 2494 := by
  sorry

end NUMINAMATH_CALUDE_special_equation_result_l2402_240284


namespace NUMINAMATH_CALUDE_cookie_brownie_difference_l2402_240234

def cookies_remaining (initial_cookies : ℕ) (daily_cookie_consumption : ℕ) (days : ℕ) : ℕ :=
  initial_cookies - daily_cookie_consumption * days

def brownies_remaining (initial_brownies : ℕ) (daily_brownie_consumption : ℕ) (days : ℕ) : ℕ :=
  initial_brownies - daily_brownie_consumption * days

theorem cookie_brownie_difference :
  let initial_cookies : ℕ := 60
  let initial_brownies : ℕ := 10
  let daily_cookie_consumption : ℕ := 3
  let daily_brownie_consumption : ℕ := 1
  let days : ℕ := 7
  cookies_remaining initial_cookies daily_cookie_consumption days -
  brownies_remaining initial_brownies daily_brownie_consumption days = 36 := by
  sorry

end NUMINAMATH_CALUDE_cookie_brownie_difference_l2402_240234


namespace NUMINAMATH_CALUDE_cubic_equation_value_l2402_240285

theorem cubic_equation_value (m : ℝ) (h : m^2 + m - 1 = 0) :
  m^3 + 2*m^2 + 2006 = 2007 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l2402_240285


namespace NUMINAMATH_CALUDE_maddie_friday_episodes_l2402_240260

/-- Represents the TV watching schedule for a week -/
structure TVSchedule where
  total_episodes : ℕ
  episode_duration : ℕ
  monday_minutes : ℕ
  thursday_minutes : ℕ
  weekend_minutes : ℕ

/-- Calculates the number of episodes watched on Friday -/
def episodes_on_friday (schedule : TVSchedule) : ℕ :=
  let total_minutes := schedule.total_episodes * schedule.episode_duration
  let other_days_minutes := schedule.monday_minutes + schedule.thursday_minutes + schedule.weekend_minutes
  let friday_minutes := total_minutes - other_days_minutes
  friday_minutes / schedule.episode_duration

/-- Theorem stating that Maddie watched 2 episodes on Friday -/
theorem maddie_friday_episodes :
  let schedule := TVSchedule.mk 8 44 138 21 105
  episodes_on_friday schedule = 2 := by
  sorry

end NUMINAMATH_CALUDE_maddie_friday_episodes_l2402_240260


namespace NUMINAMATH_CALUDE_staff_discount_price_l2402_240223

/-- Given a dress with original price d, after a 35% discount and an additional 30% staff discount,
    the final price is 0.455 times the original price. -/
theorem staff_discount_price (d : ℝ) : d * (1 - 0.35) * (1 - 0.30) = d * 0.455 := by
  sorry

#check staff_discount_price

end NUMINAMATH_CALUDE_staff_discount_price_l2402_240223


namespace NUMINAMATH_CALUDE_morgans_mean_score_l2402_240291

def scores : List ℝ := [78, 82, 90, 95, 98, 102, 105]
def alex_count : ℕ := 4
def morgan_count : ℕ := 3
def alex_mean : ℝ := 91.5

theorem morgans_mean_score (h1 : scores.length = alex_count + morgan_count)
                            (h2 : alex_count * alex_mean = (scores.take alex_count).sum) :
  (scores.drop alex_count).sum / morgan_count = 94.67 := by
  sorry

end NUMINAMATH_CALUDE_morgans_mean_score_l2402_240291


namespace NUMINAMATH_CALUDE_interval_equivalence_l2402_240290

theorem interval_equivalence (a : ℝ) : -1 < a ∧ a < 1 ↔ |a| < 1 := by
  sorry

end NUMINAMATH_CALUDE_interval_equivalence_l2402_240290


namespace NUMINAMATH_CALUDE_birds_can_gather_l2402_240283

/-- Represents a configuration of birds on trees -/
structure BirdConfiguration (n : ℕ) where
  positions : Fin n → Fin n

/-- The sum of bird positions in a configuration -/
def sum_positions (n : ℕ) (config : BirdConfiguration n) : ℕ :=
  (Finset.univ.sum fun i => (config.positions i).val) + n

/-- A bird movement that preserves the sum of positions -/
def valid_movement (n : ℕ) (config1 config2 : BirdConfiguration n) : Prop :=
  sum_positions n config1 = sum_positions n config2

/-- All birds are on the same tree -/
def all_gathered (n : ℕ) (config : BirdConfiguration n) : Prop :=
  ∃ k : Fin n, ∀ i : Fin n, config.positions i = k

/-- Initial configuration with one bird on each tree -/
def initial_config (n : ℕ) : BirdConfiguration n :=
  ⟨id⟩

/-- Theorem: Birds can gather on one tree iff n is odd and greater than 1 -/
theorem birds_can_gather (n : ℕ) :
  (∃ (config : BirdConfiguration n), valid_movement n (initial_config n) config ∧ all_gathered n config) ↔
  n % 2 = 1 ∧ n > 1 :=
sorry

end NUMINAMATH_CALUDE_birds_can_gather_l2402_240283


namespace NUMINAMATH_CALUDE_statement_equivalence_l2402_240252

theorem statement_equivalence (triangle_red circle_large : Prop) :
  (triangle_red → ¬circle_large) ↔ 
  (circle_large → ¬triangle_red) ∧ 
  (¬triangle_red ∨ ¬circle_large) := by sorry

end NUMINAMATH_CALUDE_statement_equivalence_l2402_240252


namespace NUMINAMATH_CALUDE_octal_135_to_binary_l2402_240275

/-- Converts an octal digit to its binary representation --/
def octal_to_binary_digit (d : Nat) : Nat :=
  match d with
  | 0 => 0
  | 1 => 1
  | 2 => 10
  | 3 => 11
  | 4 => 100
  | 5 => 101
  | 6 => 110
  | 7 => 111
  | _ => 0  -- Default case, should not occur for valid octal digits

/-- Converts an octal number to its binary representation --/
def octal_to_binary (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  octal_to_binary_digit hundreds * 1000000 +
  octal_to_binary_digit tens * 1000 +
  octal_to_binary_digit ones

theorem octal_135_to_binary : octal_to_binary 135 = 1011101 := by
  sorry

end NUMINAMATH_CALUDE_octal_135_to_binary_l2402_240275


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2402_240227

theorem gcd_factorial_eight_and_factorial_six_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2402_240227


namespace NUMINAMATH_CALUDE_square_greater_than_abs_l2402_240259

theorem square_greater_than_abs (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_abs_l2402_240259


namespace NUMINAMATH_CALUDE_pedro_excess_squares_l2402_240240

-- Define the initial number of squares and multipliers for each player
def jesus_initial : ℕ := 60
def jesus_multiplier : ℕ := 2
def linden_initial : ℕ := 75
def linden_multiplier : ℕ := 3
def pedro_initial : ℕ := 200
def pedro_multiplier : ℕ := 4

-- Calculate the final number of squares for each player
def jesus_final : ℕ := jesus_initial * jesus_multiplier
def linden_final : ℕ := linden_initial * linden_multiplier
def pedro_final : ℕ := pedro_initial * pedro_multiplier

-- Define the theorem to be proved
theorem pedro_excess_squares : 
  pedro_final - (jesus_final + linden_final) = 455 := by
  sorry

end NUMINAMATH_CALUDE_pedro_excess_squares_l2402_240240


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l2402_240294

theorem quadratic_real_roots_range (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x^2 - 2 * k * x + (k - 3) = 0) →
  k ≥ 3/4 ∧ k ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l2402_240294


namespace NUMINAMATH_CALUDE_point_coordinates_l2402_240233

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The third quadrant of the 2D plane -/
def ThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The distance of a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point in the third quadrant with distance 3 to the x-axis
    and distance 5 to the y-axis has coordinates (-5, -3) -/
theorem point_coordinates (p : Point) 
  (h1 : ThirdQuadrant p) 
  (h2 : DistanceToXAxis p = 3) 
  (h3 : DistanceToYAxis p = 5) : 
  p = Point.mk (-5) (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2402_240233


namespace NUMINAMATH_CALUDE_b_zero_iff_f_even_l2402_240237

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define what it means for f to be even
def is_even (a b c : ℝ) : Prop :=
  ∀ x, f a b c x = f a b c (-x)

-- State the theorem
theorem b_zero_iff_f_even (a b c : ℝ) :
  b = 0 ↔ is_even a b c :=
sorry

end NUMINAMATH_CALUDE_b_zero_iff_f_even_l2402_240237


namespace NUMINAMATH_CALUDE_jamies_mean_score_l2402_240262

def scores : List ℕ := [80, 85, 90, 95, 100, 105]

theorem jamies_mean_score 
  (h1 : scores.length = 6)
  (h2 : ∃ alex_scores jamie_scores : List ℕ, 
        alex_scores.length = 3 ∧ 
        jamie_scores.length = 3 ∧ 
        scores = alex_scores ++ jamie_scores)
  (h3 : ∃ alex_scores : List ℕ, 
        alex_scores.length = 3 ∧ 
        alex_scores.sum / alex_scores.length = 85)
  : ∃ jamie_scores : List ℕ,
    jamie_scores.length = 3 ∧
    jamie_scores.sum / jamie_scores.length = 100 := by
  sorry

end NUMINAMATH_CALUDE_jamies_mean_score_l2402_240262


namespace NUMINAMATH_CALUDE_min_sum_quadratic_roots_l2402_240256

theorem min_sum_quadratic_roots (a b : ℕ+) (h1 : ∃ x y : ℝ, 
  x ≠ y ∧ -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ 
  a * x^2 + b * x + 1 = 0 ∧ a * y^2 + b * y + 1 = 0) : 
  (∀ a' b' : ℕ+, (∃ x y : ℝ, 
    x ≠ y ∧ -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ 
    a' * x^2 + b' * x + 1 = 0 ∧ a' * y^2 + b' * y + 1 = 0) → 
  (a'.val + b'.val : ℕ) ≥ (a.val + b.val)) ∧ 
  (a.val + b.val : ℕ) = 10 := by sorry

end NUMINAMATH_CALUDE_min_sum_quadratic_roots_l2402_240256


namespace NUMINAMATH_CALUDE_veranda_area_l2402_240242

/-- Calculates the area of a veranda surrounding a rectangular room -/
theorem veranda_area (room_length room_width veranda_width : ℝ) : 
  room_length = 18 ∧ room_width = 12 ∧ veranda_width = 2 →
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 136 := by
  sorry

end NUMINAMATH_CALUDE_veranda_area_l2402_240242


namespace NUMINAMATH_CALUDE_tomato_cucumber_price_difference_l2402_240202

theorem tomato_cucumber_price_difference :
  ∀ (tomato_price cucumber_price : ℝ),
  tomato_price < cucumber_price →
  cucumber_price = 5 →
  2 * tomato_price + 3 * cucumber_price = 23 →
  (cucumber_price - tomato_price) / cucumber_price = 0.2 :=
by
  sorry

end NUMINAMATH_CALUDE_tomato_cucumber_price_difference_l2402_240202


namespace NUMINAMATH_CALUDE_adoption_time_proof_l2402_240243

/-- The number of days required to adopt all puppies -/
def adoption_days (initial_puppies : ℕ) (new_puppies : ℕ) (adopted_per_day : ℕ) : ℕ :=
  (initial_puppies + new_puppies) / adopted_per_day

/-- Theorem stating that it takes 9 days to adopt all puppies under given conditions -/
theorem adoption_time_proof :
  adoption_days 2 34 4 = 9 :=
by sorry

end NUMINAMATH_CALUDE_adoption_time_proof_l2402_240243


namespace NUMINAMATH_CALUDE_triangle_area_l2402_240271

theorem triangle_area (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  C = π / 4 →
  c = 2 →
  -- Area formula
  (1 / 2) * a * c * Real.sin B = (3 + Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2402_240271


namespace NUMINAMATH_CALUDE_outfits_count_l2402_240207

/-- The number of unique outfits that can be made from a given number of shirts, ties, and belts. -/
def uniqueOutfits (shirts ties belts : ℕ) : ℕ := shirts * ties * belts

/-- Theorem stating that with 8 shirts, 6 ties, and 4 belts, the number of unique outfits is 192. -/
theorem outfits_count : uniqueOutfits 8 6 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l2402_240207


namespace NUMINAMATH_CALUDE_election_winner_percentage_l2402_240239

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  (total_votes = winner_votes + (winner_votes - margin)) →
  (winner_votes = 650) →
  (margin = 300) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 13/20 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l2402_240239


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l2402_240218

theorem complex_magnitude_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l2402_240218


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l2402_240277

open Set

def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}

theorem intersection_of_S_and_T : S ∩ T = Ioc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l2402_240277


namespace NUMINAMATH_CALUDE_trigonometric_properties_l2402_240250

theorem trigonometric_properties :
  (∀ x, 2 * Real.sin (2 * x - π / 3) = 2 * Real.sin (2 * (5 * π / 6 - x) - π / 3)) ∧
  (∀ x, Real.tan x = -Real.tan (π - x)) ∧
  (∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ < π / 2 ∧ x₂ < π / 2 ∧ x₁ > x₂ ∧ Real.sin x₁ < Real.sin x₂) ∧
  (∀ x₁ x₂, Real.sin (2 * x₁ - π / 4) = Real.sin (2 * x₂ - π / 4) →
    (∃ k : ℤ, x₁ - x₂ = k * π ∨ x₁ + x₂ = k * π + 3 * π / 4)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_properties_l2402_240250


namespace NUMINAMATH_CALUDE_train_stop_time_l2402_240236

/-- Proves that a train with given speeds stops for 10 minutes per hour -/
theorem train_stop_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 48 → 
  speed_with_stops = 40 → 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 10 := by
  sorry

#check train_stop_time

end NUMINAMATH_CALUDE_train_stop_time_l2402_240236


namespace NUMINAMATH_CALUDE_milk_packets_problem_l2402_240217

theorem milk_packets_problem (n : ℕ) 
  (h1 : n > 2)
  (h2 : n * 20 = (n - 2) * 12 + 2 * 32) : 
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_milk_packets_problem_l2402_240217


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l2402_240287

/-- The cost of plastering a rectangular tank's walls and bottom -/
theorem tank_plastering_cost
  (length width depth : ℝ)
  (cost_per_sq_m_paise : ℝ)
  (h_length : length = 40)
  (h_width : width = 18)
  (h_depth : depth = 10)
  (h_cost : cost_per_sq_m_paise = 125) :
  let bottom_area := length * width
  let perimeter := 2 * (length + width)
  let wall_area := perimeter * depth
  let total_area := bottom_area + wall_area
  let cost_per_sq_m_rupees := cost_per_sq_m_paise / 100
  total_area * cost_per_sq_m_rupees = 2350 :=
by
  sorry


end NUMINAMATH_CALUDE_tank_plastering_cost_l2402_240287


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_theorem_l2402_240231

/-- Given two positive integers m and n with specific HCF, LCM, and sum,
    prove that the sum of their reciprocals equals 2/31.5 -/
theorem sum_of_reciprocals_theorem (m n : ℕ+) : 
  Nat.gcd m.val n.val = 6 →
  Nat.lcm m.val n.val = 210 →
  m + n = 80 →
  (1 : ℚ) / m + (1 : ℚ) / n = 2 / 31.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_theorem_l2402_240231


namespace NUMINAMATH_CALUDE_quadratic_necessary_not_sufficient_l2402_240200

theorem quadratic_necessary_not_sufficient :
  (∀ x : ℝ, (|x - 2| < 1) → (x^2 - 5*x + 4 < 0)) ∧
  (∃ x : ℝ, (x^2 - 5*x + 4 < 0) ∧ ¬(|x - 2| < 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_necessary_not_sufficient_l2402_240200


namespace NUMINAMATH_CALUDE_apartments_with_one_resident_l2402_240276

theorem apartments_with_one_resident (total : ℕ) (at_least_one_percent : ℚ) (at_least_two_percent : ℚ) :
  total = 120 →
  at_least_one_percent = 85 / 100 →
  at_least_two_percent = 60 / 100 →
  (total * at_least_one_percent - total * at_least_two_percent : ℚ) = 30 := by
sorry

end NUMINAMATH_CALUDE_apartments_with_one_resident_l2402_240276


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l2402_240224

theorem algebraic_expression_equality (x : ℝ) (h : x^2 + 2*x - 3 = 7) : 
  2*x^2 + 4*x + 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l2402_240224


namespace NUMINAMATH_CALUDE_liars_on_black_chairs_l2402_240270

def room_scenario (total_people : ℕ) (initial_black_claims : ℕ) (final_white_claims : ℕ) : Prop :=
  -- Total number of people is positive
  total_people > 0 ∧
  -- Initially, all people claim to be on black chairs
  initial_black_claims = total_people ∧
  -- After rearrangement, some people claim to be on white chairs
  final_white_claims > 0 ∧ final_white_claims < total_people

theorem liars_on_black_chairs 
  (total_people : ℕ) 
  (initial_black_claims : ℕ) 
  (final_white_claims : ℕ) 
  (h : room_scenario total_people initial_black_claims final_white_claims) :
  -- The number of liars on black chairs after rearrangement
  (final_white_claims / 2) = 8 :=
sorry

end NUMINAMATH_CALUDE_liars_on_black_chairs_l2402_240270


namespace NUMINAMATH_CALUDE_points_three_units_from_negative_three_l2402_240211

def distance (x y : ℝ) : ℝ := |x - y|

theorem points_three_units_from_negative_three :
  ∀ x : ℝ, distance x (-3) = 3 ↔ x = 0 ∨ x = -6 := by
  sorry

end NUMINAMATH_CALUDE_points_three_units_from_negative_three_l2402_240211


namespace NUMINAMATH_CALUDE_taxi_ride_distance_l2402_240201

/-- Calculates the distance of a taxi ride given the fare structure and total fare -/
theorem taxi_ride_distance
  (initial_fare : ℚ)
  (initial_distance : ℚ)
  (additional_fare : ℚ)
  (additional_distance : ℚ)
  (total_fare : ℚ)
  (h1 : initial_fare = 8)
  (h2 : initial_distance = 1/5)
  (h3 : additional_fare = 4/5)
  (h4 : additional_distance = 1/5)
  (h5 : total_fare = 39.2) :
  ∃ (distance : ℚ), distance = 8 ∧ 
    total_fare = initial_fare + (distance - initial_distance) / additional_distance * additional_fare :=
by sorry

end NUMINAMATH_CALUDE_taxi_ride_distance_l2402_240201


namespace NUMINAMATH_CALUDE_distribute_5_3_l2402_240215

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    with each box containing at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects into 3 distinct boxes,
    with each box containing at least one object, is 150. -/
theorem distribute_5_3 : distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l2402_240215


namespace NUMINAMATH_CALUDE_simplify_expression_l2402_240244

theorem simplify_expression : 
  2 - (2 / (2 + Real.sqrt 5)) + (2 / (2 - Real.sqrt 5)) = 2 - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2402_240244


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_a_range_when_intersection_empty_l2402_240216

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem for part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 2) = {x | 1 < x ∧ x ≤ 2} := by sorry

-- Theorem for part 2
theorem a_range_when_intersection_empty :
  ∀ a : ℝ, A ∩ B a = ∅ → a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_a_range_when_intersection_empty_l2402_240216


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2402_240280

/-- Given points (2, y₁) and (-2, y₂) on the graph of y = ax² + bx + d, where y₁ - y₂ = -8, prove that b = -2 -/
theorem quadratic_coefficient (a d y₁ y₂ : ℝ) : 
  y₁ = 4 * a + 2 * b + d →
  y₂ = 4 * a - 2 * b + d →
  y₁ - y₂ = -8 →
  b = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2402_240280


namespace NUMINAMATH_CALUDE_choir_members_count_l2402_240221

theorem choir_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 300 ∧ 
  n % 6 = 1 ∧ 
  n % 8 = 3 ∧ 
  n % 9 = 5 ∧ 
  n = 193 := by sorry

end NUMINAMATH_CALUDE_choir_members_count_l2402_240221


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2402_240241

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |5 * x - 7| + 2 = 2 ∧ x = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2402_240241


namespace NUMINAMATH_CALUDE_license_plate_increase_l2402_240273

/-- The number of possible characters for letters in the new scheme -/
def new_letter_options : ℕ := 30

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_options : ℕ := 10

/-- The number of letters in the new license plate scheme -/
def new_letter_count : ℕ := 2

/-- The number of digits in the new license plate scheme -/
def new_digit_count : ℕ := 5

/-- The number of letters in the previous license plate scheme -/
def old_letter_count : ℕ := 3

/-- The number of digits in the previous license plate scheme -/
def old_digit_count : ℕ := 3

theorem license_plate_increase :
  (new_letter_options ^ new_letter_count * digit_options ^ new_digit_count) /
  (alphabet_size ^ old_letter_count * digit_options ^ old_digit_count) =
  (900 : ℚ) / 17576 * 100 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_increase_l2402_240273


namespace NUMINAMATH_CALUDE_expression_evaluation_l2402_240205

theorem expression_evaluation (x y : ℝ) (hx : x = -2) (hy : y = 1) :
  2 * x^2 - (2*x*y - 3*y^2) + 2*(x^2 + x*y - 2*y^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2402_240205


namespace NUMINAMATH_CALUDE_water_bottles_profit_l2402_240263

def water_bottles_problem (total_bottles : ℕ) (standard_rate_bottles : ℕ) (standard_rate_price : ℚ)
  (discount_threshold : ℕ) (discount_rate : ℚ) (selling_rate_bottles : ℕ) (selling_rate_price : ℚ) : Prop :=
  let standard_price_per_bottle : ℚ := standard_rate_price / standard_rate_bottles
  let total_cost_without_discount : ℚ := total_bottles * standard_price_per_bottle
  let total_cost_with_discount : ℚ := total_cost_without_discount * (1 - discount_rate)
  let selling_price_per_bottle : ℚ := selling_rate_price / selling_rate_bottles
  let total_revenue : ℚ := total_bottles * selling_price_per_bottle
  let profit : ℚ := total_revenue - total_cost_with_discount
  (total_bottles > discount_threshold) ∧ (profit = 325)

theorem water_bottles_profit :
  water_bottles_problem 1500 6 3 1200 (1/10) 3 2 :=
sorry

end NUMINAMATH_CALUDE_water_bottles_profit_l2402_240263


namespace NUMINAMATH_CALUDE_at_least_one_red_probability_l2402_240267

theorem at_least_one_red_probability
  (prob_red_A prob_red_B : ℚ)
  (h_prob_A : prob_red_A = 1/3)
  (h_prob_B : prob_red_B = 1/2) :
  1 - (1 - prob_red_A) * (1 - prob_red_B) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_red_probability_l2402_240267


namespace NUMINAMATH_CALUDE_infinitely_many_squares_l2402_240238

/-- An arithmetic sequence of positive integers -/
def ArithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n => a + n * d

/-- A number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem infinitely_many_squares
  (a d : ℕ) -- First term and common difference
  (h_pos : ∀ n, 0 < ArithmeticSequence a d n) -- Sequence is positive
  (h_square : ∃ n, IsPerfectSquare (ArithmeticSequence a d n)) -- At least one square exists
  : ∀ m : ℕ, ∃ n > m, IsPerfectSquare (ArithmeticSequence a d n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_squares_l2402_240238


namespace NUMINAMATH_CALUDE_solution_set_a_eq_one_solution_set_is_real_l2402_240235

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + a * x - 2

-- Part 1: Solution set for a = 1
theorem solution_set_a_eq_one :
  {x : ℝ | f 1 x ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Part 2: Conditions for solution set to be ℝ
theorem solution_set_is_real :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≤ 0) ↔ -8 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_eq_one_solution_set_is_real_l2402_240235


namespace NUMINAMATH_CALUDE_ellipse_minimum_area_l2402_240258

/-- An ellipse containing two specific circles has a minimum area of 16π -/
theorem ellipse_minimum_area :
  ∀ a b : ℝ,
  (∀ x y : ℝ, x^2 / (4*a^2) + y^2 / (4*b^2) = 1 →
    ((x - 2)^2 + y^2 ≤ 4 ∧ (x + 2)^2 + y^2 ≤ 4)) →
  4 * π * a * b ≥ 16 * π :=
by sorry

end NUMINAMATH_CALUDE_ellipse_minimum_area_l2402_240258


namespace NUMINAMATH_CALUDE_statue_original_cost_l2402_240204

/-- If a statue is sold for $660 with a 20% profit, then its original cost was $550. -/
theorem statue_original_cost (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 660 → profit_percentage = 0.20 → 
  selling_price = (1 + profit_percentage) * 550 := by
sorry

end NUMINAMATH_CALUDE_statue_original_cost_l2402_240204


namespace NUMINAMATH_CALUDE_salt_solution_problem_l2402_240282

theorem salt_solution_problem (initial_volume : ℝ) (added_water : ℝ) (final_salt_percentage : ℝ) :
  initial_volume = 80 →
  added_water = 20 →
  final_salt_percentage = 8 →
  let final_volume := initial_volume + added_water
  let initial_salt_amount := (initial_volume * final_salt_percentage) / final_volume
  let initial_salt_percentage := (initial_salt_amount / initial_volume) * 100
  initial_salt_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_salt_solution_problem_l2402_240282


namespace NUMINAMATH_CALUDE_bug_positions_l2402_240272

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Set of positions reachable by the bug in at most n steps -/
def reachablePositions (n : ℕ) : Set ℚ :=
  {x | ∃ (k : ℕ), k ≤ n ∧ ∃ (steps : List (ℚ → ℚ)),
    steps.length = k ∧
    (∀ step ∈ steps, step = (· + 2) ∨ step = (· / 2)) ∧
    x = (steps.foldl (λ acc f => f acc) 1)}

/-- The main theorem -/
theorem bug_positions (n : ℕ) :
  (reachablePositions n).ncard = fib (n + 4) - (n + 4) :=
sorry

end NUMINAMATH_CALUDE_bug_positions_l2402_240272


namespace NUMINAMATH_CALUDE_optimal_price_and_profit_l2402_240295

-- Define the purchase price
def purchase_price : ℝ := 50

-- Define the linear relationship between price and sales volume
def sales_volume (x : ℝ) : ℝ := -20 * x + 2600

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - purchase_price) * sales_volume x

-- Define the constraint that selling price is not lower than purchase price
def price_constraint (x : ℝ) : Prop := x ≥ purchase_price

-- Define the constraint that profit per shirt should not exceed 30% of purchase price
def profit_constraint (x : ℝ) : Prop := x - purchase_price ≤ 0.3 * purchase_price

-- Theorem statement
theorem optimal_price_and_profit :
  ∃ (x : ℝ), 
    price_constraint x ∧ 
    profit_constraint x ∧ 
    (∀ y : ℝ, price_constraint y → profit_constraint y → profit x ≥ profit y) ∧
    x = 65 ∧
    profit x = 19500 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_and_profit_l2402_240295


namespace NUMINAMATH_CALUDE_square_grid_perimeter_l2402_240225

/-- The perimeter of a 3x3 grid of congruent squares with a total area of 576 square centimeters is 192 centimeters. -/
theorem square_grid_perimeter (total_area : ℝ) (side_length : ℝ) (perimeter : ℝ) : 
  total_area = 576 →
  side_length * side_length * 9 = total_area →
  perimeter = 4 * 3 * side_length →
  perimeter = 192 := by
sorry

end NUMINAMATH_CALUDE_square_grid_perimeter_l2402_240225


namespace NUMINAMATH_CALUDE_comic_cost_l2402_240257

theorem comic_cost (initial_money : ℕ) (comics_bought : ℕ) (money_left : ℕ) : 
  initial_money = 87 → comics_bought = 8 → money_left = 55 → 
  (initial_money - money_left) / comics_bought = 4 := by
sorry

end NUMINAMATH_CALUDE_comic_cost_l2402_240257


namespace NUMINAMATH_CALUDE_square_twelve_y_minus_five_l2402_240289

theorem square_twelve_y_minus_five (y : ℝ) (h : 6 * y^2 + 7 = 4 * y + 13) : 
  (12 * y - 5)^2 = 161 := by
  sorry

end NUMINAMATH_CALUDE_square_twelve_y_minus_five_l2402_240289
