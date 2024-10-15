import Mathlib

namespace NUMINAMATH_CALUDE_max_b_value_l4092_409212

def is_prime (n : ℕ) : Prop := sorry

theorem max_b_value (a b c : ℕ) : 
  (a * b * c = 720) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (c = 3) →
  is_prime a →
  is_prime b →
  is_prime c →
  (∀ x : ℕ, (1 < x) ∧ (x < b) ∧ is_prime x → x ≤ 3) →
  (b ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l4092_409212


namespace NUMINAMATH_CALUDE_odd_sum_is_odd_l4092_409232

theorem odd_sum_is_odd (a b : ℤ) (ha : Odd a) (hb : Odd b) : Odd (a + 2*b + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_is_odd_l4092_409232


namespace NUMINAMATH_CALUDE_determinant_2x2_matrix_l4092_409239

theorem determinant_2x2_matrix (x : ℝ) :
  Matrix.det !![5, x; -3, 9] = 45 + 3 * x := by sorry

end NUMINAMATH_CALUDE_determinant_2x2_matrix_l4092_409239


namespace NUMINAMATH_CALUDE_rachel_assembly_time_l4092_409251

/-- Calculates the total time taken to assemble furniture -/
def total_assembly_time (num_chairs : ℕ) (num_tables : ℕ) (time_per_piece : ℕ) : ℕ :=
  (num_chairs + num_tables) * time_per_piece

/-- Proves that the total assembly time for Rachel's furniture is 40 minutes -/
theorem rachel_assembly_time :
  total_assembly_time 7 3 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_rachel_assembly_time_l4092_409251


namespace NUMINAMATH_CALUDE_expression_evaluation_l4092_409241

theorem expression_evaluation : 
  let expr := 125 - 25 * 4
  expr = 25 := by sorry

#check expression_evaluation

end NUMINAMATH_CALUDE_expression_evaluation_l4092_409241


namespace NUMINAMATH_CALUDE_square_cutting_existence_l4092_409210

theorem square_cutting_existence : ∃ (a b c S : ℝ), 
  a^2 + 3*b^2 + 5*c^2 = S^2 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_cutting_existence_l4092_409210


namespace NUMINAMATH_CALUDE_smallest_k_for_single_root_l4092_409205

-- Define the functions f and g
def f (x : ℝ) : ℝ := 41 * x^2 - 4 * x + 4
def g (x : ℝ) : ℝ := -2 * x^2 + x

-- Define the combined function h
def h (k : ℝ) (x : ℝ) : ℝ := f x + k * g x

-- Define the discriminant of h
def discriminant (k : ℝ) : ℝ := (k - 4)^2 - 4 * (41 - 2*k) * 4

-- Theorem statement
theorem smallest_k_for_single_root :
  ∃ d : ℝ, d = -40 ∧ 
  (∀ k : ℝ, (∃ x : ℝ, h k x = 0 ∧ (∀ y : ℝ, h k y = 0 → y = x)) → k ≥ d) ∧
  (∃ x : ℝ, h d x = 0 ∧ (∀ y : ℝ, h d y = 0 → y = x)) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_single_root_l4092_409205


namespace NUMINAMATH_CALUDE_fraction_equivalence_l4092_409288

theorem fraction_equivalence (a b k : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hk : k ≠ 0) :
  (k * a) / (k * b) = a / b :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l4092_409288


namespace NUMINAMATH_CALUDE_lizzy_final_amount_l4092_409240

/-- Calculates the total amount Lizzy will have after loans are returned with interest -/
def lizzys_total_amount (initial_amount : ℝ) (alice_loan : ℝ) (bob_loan : ℝ) 
  (alice_interest_rate : ℝ) (bob_interest_rate : ℝ) : ℝ :=
  initial_amount - alice_loan - bob_loan + 
  alice_loan * (1 + alice_interest_rate) + 
  bob_loan * (1 + bob_interest_rate)

/-- Theorem stating that Lizzy will have $52.75 after loans are returned -/
theorem lizzy_final_amount : 
  lizzys_total_amount 50 25 20 0.15 0.20 = 52.75 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_final_amount_l4092_409240


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_and_b_l4092_409258

/-- Given a function f(x) = x^3 - 3ax^2 + b, prove that if the curve y = f(x) is tangent
    to the line y = 8 at the point (2, f(2)), then a = 1 and b = 12. -/
theorem tangent_line_implies_a_and_b (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - 3*a*x^2 + b
  (f 2 = 8) ∧ (deriv f 2 = 0) → a = 1 ∧ b = 12 := by
  sorry

#check tangent_line_implies_a_and_b

end NUMINAMATH_CALUDE_tangent_line_implies_a_and_b_l4092_409258


namespace NUMINAMATH_CALUDE_tangency_condition_l4092_409221

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 + 6

/-- The hyperbola equation -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m*x^2 = 6

/-- The tangency condition -/
def are_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, parabola x y ∧ hyperbola m x y ∧
  ∀ x' y' : ℝ, parabola x' y' ∧ hyperbola m x' y' → (x = x' ∧ y = y')

/-- The theorem stating the condition for tangency -/
theorem tangency_condition :
  ∀ m : ℝ, are_tangent m ↔ (m = 12 + 10 * Real.sqrt 6 ∨ m = 12 - 10 * Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_tangency_condition_l4092_409221


namespace NUMINAMATH_CALUDE_binary_110010_is_50_l4092_409225

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enumFrom 0 b).foldl (fun acc (i, x) => acc + if x then 2^i else 0) 0

def binary_110010 : List Bool := [false, true, false, false, true, true]

theorem binary_110010_is_50 : binary_to_decimal binary_110010 = 50 := by
  sorry

end NUMINAMATH_CALUDE_binary_110010_is_50_l4092_409225


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l4092_409270

/-- Given a line passing through points (1, 4) and (-2, -2), prove that the product of its slope and y-intercept is 4. -/
theorem line_slope_intercept_product (m b : ℝ) : 
  (4 = m * 1 + b) → 
  (-2 = m * (-2) + b) → 
  m * b = 4 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l4092_409270


namespace NUMINAMATH_CALUDE_train_length_l4092_409261

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (h1 : speed_kmh = 72) (h2 : time_s = 5.999520038396929) :
  ∃ (length_m : ℝ), abs (length_m - 119.99) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l4092_409261


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l4092_409222

def a : ℝ × ℝ × ℝ := (1, -1, 4)
def b : ℝ × ℝ × ℝ := (1, 0, 3)
def c : ℝ × ℝ × ℝ := (1, -3, 8)

def scalar_triple_product (v1 v2 v3 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := v1
  let (x2, y2, z2) := v2
  let (x3, y3, z3) := v3
  x1 * (y2 * z3 - y3 * z2) - y1 * (x2 * z3 - x3 * z2) + z1 * (x2 * y3 - x3 * y2)

theorem vectors_not_coplanar : scalar_triple_product a b c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l4092_409222


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l4092_409200

theorem complex_magnitude_theorem (b : ℝ) :
  (Complex.I * Complex.I.re = ((1 + b * Complex.I) * (2 + Complex.I)).re) →
  Complex.abs ((2 * b + 3 * Complex.I) / (1 + b * Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l4092_409200


namespace NUMINAMATH_CALUDE_perfect_squares_l4092_409229

theorem perfect_squares (k : ℕ) (h1 : k > 0) (h2 : ∃ a : ℕ, k * (k + 1) = 3 * a^2) : 
  (∃ m : ℕ, k = 3 * m^2) ∧ (∃ n : ℕ, k + 1 = n^2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_l4092_409229


namespace NUMINAMATH_CALUDE_tan_45_degrees_l4092_409227

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l4092_409227


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4092_409230

theorem partial_fraction_decomposition (M₁ M₂ : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (27 * x - 19) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) →
  M₁ * M₂ = -2170 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4092_409230


namespace NUMINAMATH_CALUDE_range_of_b_minus_a_l4092_409250

theorem range_of_b_minus_a (a b : ℝ) : 
  (a < b) →
  (∀ x : ℝ, (a ≤ x ∧ x ≤ b) → (x^2 + x - 2 ≤ 0)) →
  (∃ x : ℝ, (x^2 + x - 2 ≤ 0) ∧ ¬(a ≤ x ∧ x ≤ b)) →
  (0 < b - a) ∧ (b - a < 3) := by
sorry

end NUMINAMATH_CALUDE_range_of_b_minus_a_l4092_409250


namespace NUMINAMATH_CALUDE_candy_mix_cost_l4092_409282

/-- Prove the cost of candy B given the mixture conditions -/
theorem candy_mix_cost (total_weight : ℝ) (mix_cost_per_pound : ℝ) 
  (candy_a_cost : ℝ) (candy_a_weight : ℝ) :
  total_weight = 5 →
  mix_cost_per_pound = 2 →
  candy_a_cost = 3.2 →
  candy_a_weight = 1 →
  ∃ (candy_b_cost : ℝ),
    candy_b_cost = 1.7 ∧
    total_weight * mix_cost_per_pound = 
      candy_a_weight * candy_a_cost + (total_weight - candy_a_weight) * candy_b_cost :=
by
  sorry


end NUMINAMATH_CALUDE_candy_mix_cost_l4092_409282


namespace NUMINAMATH_CALUDE_equilateral_triangle_properties_l4092_409247

/-- Proves properties of an equilateral triangle with given area and side length -/
theorem equilateral_triangle_properties :
  ∀ (area base altitude perimeter : ℝ),
  area = 450 →
  base = 25 →
  area = (1/2) * base * altitude →
  perimeter = 3 * base →
  altitude = 36 ∧ perimeter = 75 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_properties_l4092_409247


namespace NUMINAMATH_CALUDE_james_total_toys_l4092_409272

/-- The number of toy cars James buys -/
def toy_cars : ℕ := 20

/-- The number of toy soldiers James buys -/
def toy_soldiers : ℕ := 2 * toy_cars

/-- The total number of toys James buys -/
def total_toys : ℕ := toy_cars + toy_soldiers

theorem james_total_toys : total_toys = 60 := by sorry

end NUMINAMATH_CALUDE_james_total_toys_l4092_409272


namespace NUMINAMATH_CALUDE_zero_point_implies_a_range_l4092_409298

/-- Given a function y = x³ - ax where x ∈ ℝ and y has a zero point at (1, 2),
    prove that a ∈ (1, 4) -/
theorem zero_point_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Ioo 1 2, x^3 - a*x = 0) →
  a ∈ Set.Ioo 1 4 := by
  sorry

end NUMINAMATH_CALUDE_zero_point_implies_a_range_l4092_409298


namespace NUMINAMATH_CALUDE_f_bound_iff_m_range_l4092_409295

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x + x^2 / m^2 - x

theorem f_bound_iff_m_range (m : ℝ) (hm : m ≠ 0) :
  (∀ a b : ℝ, a ∈ Set.Icc (-1) 1 → b ∈ Set.Icc (-1) 1 → |f m a - f m b| ≤ Real.exp 1) ↔
  m ∈ Set.Iic (-Real.sqrt 2 / 2) ∪ Set.Ici (Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_f_bound_iff_m_range_l4092_409295


namespace NUMINAMATH_CALUDE_sum_even_integers_102_to_200_l4092_409255

/-- The sum of even integers from 102 to 200 inclusive -/
def sum_even_102_to_200 : ℕ := 7550

/-- The sum of the first 50 positive even integers -/
def sum_first_50_even : ℕ := 2550

/-- The number of even integers from 102 to 200 inclusive -/
def num_even_102_to_200 : ℕ := 50

theorem sum_even_integers_102_to_200 :
  sum_even_102_to_200 = (num_even_102_to_200 / 2) * (102 + 200) :=
by sorry

end NUMINAMATH_CALUDE_sum_even_integers_102_to_200_l4092_409255


namespace NUMINAMATH_CALUDE_a_more_stable_than_b_l4092_409217

/-- Represents a shooter with their shooting variance -/
structure Shooter where
  name : String
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Definition of more stable shooting performance -/
def more_stable (a b : Shooter) : Prop :=
  a.variance < b.variance

/-- Theorem stating that shooter A has more stable performance than B -/
theorem a_more_stable_than_b :
  let a : Shooter := ⟨"A", 0.12, by norm_num⟩
  let b : Shooter := ⟨"B", 0.6, by norm_num⟩
  more_stable a b := by
  sorry


end NUMINAMATH_CALUDE_a_more_stable_than_b_l4092_409217


namespace NUMINAMATH_CALUDE_three_lines_intersection_l4092_409248

/-- Three lines intersect at a single point if and only if m = 22/7 -/
theorem three_lines_intersection (x y m : ℚ) : 
  (y = 3 * x + 2) ∧ 
  (y = -4 * x + 10) ∧ 
  (y = 2 * x + m) → 
  m = 22 / 7 := by
sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l4092_409248


namespace NUMINAMATH_CALUDE_drum_capacity_ratio_l4092_409281

theorem drum_capacity_ratio (c_x c_y : ℝ) : 
  c_x > 0 → c_y > 0 →
  (1/2 * c_x + 1/2 * c_y = 3/4 * c_y) →
  c_y / c_x = 2 := by
sorry

end NUMINAMATH_CALUDE_drum_capacity_ratio_l4092_409281


namespace NUMINAMATH_CALUDE_fourth_month_sale_l4092_409249

/-- Given the sales for 5 out of 6 months and the average sale for 6 months, 
    prove that the sale in the fourth month must be 8230. -/
theorem fourth_month_sale 
  (sale1 : ℕ) (sale2 : ℕ) (sale3 : ℕ) (sale5 : ℕ) (sale6 : ℕ) (avg_sale : ℕ)
  (h1 : sale1 = 7435)
  (h2 : sale2 = 7920)
  (h3 : sale3 = 7855)
  (h5 : sale5 = 7560)
  (h6 : sale6 = 6000)
  (h_avg : avg_sale = 7500)
  : ∃ (sale4 : ℕ), sale4 = 8230 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = avg_sale :=
by sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l4092_409249


namespace NUMINAMATH_CALUDE_expression_equality_l4092_409220

theorem expression_equality : 5 * 401 + 4 * 401 + 3 * 401 + 400 = 5212 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l4092_409220


namespace NUMINAMATH_CALUDE_pet_ownership_l4092_409231

theorem pet_ownership (total_students : ℕ) 
  (dog_owners cat_owners other_pet_owners : ℕ)
  (no_pet_owners : ℕ)
  (only_dog_owners only_cat_owners only_other_pet_owners : ℕ) :
  total_students = 40 →
  dog_owners = total_students / 2 →
  cat_owners = total_students / 4 →
  other_pet_owners = 8 →
  no_pet_owners = 5 →
  only_dog_owners = 15 →
  only_cat_owners = 4 →
  only_other_pet_owners = 5 →
  ∃ (all_three_pets : ℕ),
    all_three_pets = 1 ∧
    dog_owners = only_dog_owners + (other_pet_owners - only_other_pet_owners) + 
                 (cat_owners - only_cat_owners) - all_three_pets + all_three_pets ∧
    cat_owners = only_cat_owners + (other_pet_owners - only_other_pet_owners) + 
                 (dog_owners - only_dog_owners) - all_three_pets + all_three_pets ∧
    other_pet_owners = only_other_pet_owners + (dog_owners - only_dog_owners) + 
                       (cat_owners - only_cat_owners) - all_three_pets + all_three_pets ∧
    total_students = dog_owners + cat_owners + other_pet_owners - 
                     (dog_owners - only_dog_owners) - (cat_owners - only_cat_owners) - 
                     (other_pet_owners - only_other_pet_owners) + all_three_pets + no_pet_owners :=
by
  sorry

end NUMINAMATH_CALUDE_pet_ownership_l4092_409231


namespace NUMINAMATH_CALUDE_admin_teacher_ratio_l4092_409201

/-- The ratio of administrators to teachers at a graduation ceremony -/
theorem admin_teacher_ratio :
  let graduates : ℕ := 50
  let parents_per_graduate : ℕ := 2
  let teachers : ℕ := 20
  let total_chairs : ℕ := 180
  let grad_parent_chairs := graduates * (parents_per_graduate + 1)
  let admin_chairs := total_chairs - (grad_parent_chairs + teachers)
  (admin_chairs : ℚ) / teachers = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_admin_teacher_ratio_l4092_409201


namespace NUMINAMATH_CALUDE_no_natural_solutions_l4092_409208

theorem no_natural_solutions (x y : ℕ) : 
  (1 : ℚ) / (x^2 : ℚ) + (1 : ℚ) / ((x * y) : ℚ) + (1 : ℚ) / (y^2 : ℚ) ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l4092_409208


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4092_409293

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →  -- given condition
  4 * a 2 - 2 * a 3 = 2 * a 3 - a 4 →  -- arithmetic sequence condition
  a 2 + a 3 + a 4 = 14 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4092_409293


namespace NUMINAMATH_CALUDE_composite_sum_of_powers_l4092_409224

theorem composite_sum_of_powers (a b c d m n : ℕ) 
  (h_pos : 0 < d ∧ d < c ∧ c < b ∧ b < a)
  (h_div : (a + b - c + d) ∣ (a * c + b * d))
  (h_m_pos : 0 < m)
  (h_n_odd : n % 2 = 1) :
  ∃ (k : ℕ), k > 1 ∧ k ∣ (a^n * b^m + c^m * d^n) :=
sorry

end NUMINAMATH_CALUDE_composite_sum_of_powers_l4092_409224


namespace NUMINAMATH_CALUDE_polynomial_leading_coefficient_l4092_409274

/-- A polynomial g satisfying g(x + 1) - g(x) = 12x + 2 for all x has leading coefficient 6 -/
theorem polynomial_leading_coefficient (g : ℝ → ℝ) :
  (∀ x, g (x + 1) - g x = 12 * x + 2) →
  ∃ c, ∀ x, g x = 6 * x^2 - 4 * x + c :=
sorry

end NUMINAMATH_CALUDE_polynomial_leading_coefficient_l4092_409274


namespace NUMINAMATH_CALUDE_triangle_perimeter_upper_bound_l4092_409218

theorem triangle_perimeter_upper_bound :
  ∀ a b c : ℝ,
  a = 5 →
  b = 19 →
  a + b > c →
  a + c > b →
  b + c > a →
  a + b + c < 48 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_upper_bound_l4092_409218


namespace NUMINAMATH_CALUDE_circle_area_equality_l4092_409253

theorem circle_area_equality (r₁ r₂ : ℝ) (h₁ : r₁ = 35) (h₂ : r₂ = 25) :
  ∃ r₃ : ℝ, r₃ = 10 * Real.sqrt 6 ∧ π * r₃^2 = π * (r₁^2 - r₂^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equality_l4092_409253


namespace NUMINAMATH_CALUDE_beth_sold_coins_l4092_409297

-- Define the initial number of coins Beth had
def initial_coins : ℕ := 125

-- Define the number of coins Carl gave to Beth
def gifted_coins : ℕ := 35

-- Define the total number of coins Beth had after receiving the gift
def total_coins : ℕ := initial_coins + gifted_coins

-- Define the number of coins Beth sold (half of her total coins)
def sold_coins : ℕ := total_coins / 2

-- Theorem stating that the number of coins Beth sold is equal to 80
theorem beth_sold_coins : sold_coins = 80 := by
  sorry

end NUMINAMATH_CALUDE_beth_sold_coins_l4092_409297


namespace NUMINAMATH_CALUDE_quadratic_value_l4092_409291

-- Define the quadratic function
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- State the theorem
theorem quadratic_value (p q : ℝ) :
  f p q 1 = 3 → f p q (-3) = 7 → f p q (-5) = 21 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_l4092_409291


namespace NUMINAMATH_CALUDE_shelly_money_theorem_l4092_409215

/-- Calculates the total amount of money Shelly has given the number of $10 and $5 bills -/
def total_money (ten_dollar_bills : ℕ) (five_dollar_bills : ℕ) : ℕ :=
  10 * ten_dollar_bills + 5 * five_dollar_bills

/-- Proves that Shelly has $130 in total -/
theorem shelly_money_theorem :
  let ten_dollar_bills : ℕ := 10
  let five_dollar_bills : ℕ := ten_dollar_bills - 4
  total_money ten_dollar_bills five_dollar_bills = 130 := by
  sorry

#check shelly_money_theorem

end NUMINAMATH_CALUDE_shelly_money_theorem_l4092_409215


namespace NUMINAMATH_CALUDE_ten_row_triangle_pieces_l4092_409266

/-- Calculates the number of rods in a triangle with given number of rows -/
def num_rods (n : ℕ) : ℕ := n * (n + 1) * 3

/-- Calculates the number of connectors in a triangle with given number of rows -/
def num_connectors (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- The total number of pieces in a triangle with given number of rows -/
def total_pieces (n : ℕ) : ℕ := num_rods n + num_connectors n

theorem ten_row_triangle_pieces :
  total_pieces 10 = 366 ∧
  num_rods 3 = 18 ∧
  num_connectors 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_row_triangle_pieces_l4092_409266


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l4092_409203

theorem arithmetic_mean_after_removal (S : Finset ℝ) (a b c : ℝ) :
  S.card = 60 →
  a = 48 ∧ b = 52 ∧ c = 56 →
  a ∈ S ∧ b ∈ S ∧ c ∈ S →
  (S.sum id) / S.card = 42 →
  ((S.sum id) - (a + b + c)) / (S.card - 3) = 41.47 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l4092_409203


namespace NUMINAMATH_CALUDE_students_in_class_l4092_409206

/-- Proves that the number of students in Ms. Leech's class is 30 -/
theorem students_in_class (num_boys : ℕ) (num_girls : ℕ) (cups_per_boy : ℕ) (total_cups : ℕ) :
  num_boys = 10 →
  num_girls = 2 * num_boys →
  cups_per_boy = 5 →
  total_cups = 90 →
  num_boys * cups_per_boy + num_girls * ((total_cups - num_boys * cups_per_boy) / num_girls) = total_cups →
  num_boys + num_girls = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_in_class_l4092_409206


namespace NUMINAMATH_CALUDE_right_triangle_with_specific_perimeter_l4092_409269

theorem right_triangle_with_specific_perimeter :
  ∃ (b c : ℤ), 
    b = 7 ∧ 
    c = 5 ∧ 
    (b : ℝ)^2 + (b + c : ℝ)^2 = (b + 2*c : ℝ)^2 ∧ 
    b + (b + c) + (b + 2*c) = 36 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_specific_perimeter_l4092_409269


namespace NUMINAMATH_CALUDE_toy_car_energy_comparison_l4092_409263

theorem toy_car_energy_comparison (m : ℝ) (h : m > 0) :
  let KE (v : ℝ) := (1/2) * m * v^2
  (KE 4 - KE 2) = 3 * (KE 2 - KE 0) :=
by
  sorry

end NUMINAMATH_CALUDE_toy_car_energy_comparison_l4092_409263


namespace NUMINAMATH_CALUDE_cube_root_comparison_l4092_409244

theorem cube_root_comparison : 2 + Real.rpow 7 (1/3) < Real.rpow 60 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_comparison_l4092_409244


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_half_l4092_409223

/-- Represents a cube constructed from smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  total_small_cubes : ℕ
  white_cubes : ℕ
  black_cubes : ℕ

/-- Calculates the fraction of white surface area for a composite cube -/
def white_surface_fraction (c : CompositeCube) : ℚ :=
  sorry

/-- The specific composite cube from the problem -/
def problem_cube : CompositeCube :=
  { edge_length := 4
  , total_small_cubes := 64
  , white_cubes := 48
  , black_cubes := 16 }

theorem white_surface_fraction_is_half :
  white_surface_fraction problem_cube = 1/2 :=
sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_half_l4092_409223


namespace NUMINAMATH_CALUDE_same_color_choices_l4092_409209

theorem same_color_choices (m : ℕ) : 
  let total_objects := 2 * m
  let red_objects := m
  let blue_objects := m
  (number_of_ways_to_choose_same_color : ℕ) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_same_color_choices_l4092_409209


namespace NUMINAMATH_CALUDE_smallest_value_absolute_equation_l4092_409204

theorem smallest_value_absolute_equation :
  ∃ (x : ℝ), x = -5 ∧ |x - 4| = 9 ∧ ∀ (y : ℝ), |y - 4| = 9 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_absolute_equation_l4092_409204


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l4092_409273

theorem opposite_of_negative_five : 
  (-(- 5 : ℤ)) = (5 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l4092_409273


namespace NUMINAMATH_CALUDE_triangle_sum_zero_l4092_409252

theorem triangle_sum_zero (a b c : ℝ) 
  (ha : |a| ≥ |b + c|) 
  (hb : |b| ≥ |c + a|) 
  (hc : |c| ≥ |a + b|) : 
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_sum_zero_l4092_409252


namespace NUMINAMATH_CALUDE_committee_probability_l4092_409286

def total_members : ℕ := 30
def num_boys : ℕ := 12
def num_girls : ℕ := 18
def committee_size : ℕ := 6

theorem committee_probability :
  let total_combinations := Nat.choose total_members committee_size
  let all_boys_combinations := Nat.choose num_boys committee_size
  let all_girls_combinations := Nat.choose num_girls committee_size
  let prob_at_least_one_each := 1 - (all_boys_combinations + all_girls_combinations : ℚ) / total_combinations
  prob_at_least_one_each = 574287 / 593775 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l4092_409286


namespace NUMINAMATH_CALUDE_figure_to_square_approximation_l4092_409207

/-- A figure on a grid of squares -/
structure GridFigure where
  area : ℕ
  is_on_grid : Bool

/-- Represents a division of a figure into parts -/
structure FigureDivision where
  parts : ℕ
  can_rearrange_to_square : Bool

/-- Theorem: A figure with 18 unit squares can be divided into three parts and rearranged to approximate a square -/
theorem figure_to_square_approximation (f : GridFigure) (d : FigureDivision) :
  f.area = 18 ∧ f.is_on_grid = true ∧ d.parts = 3 → d.can_rearrange_to_square = true := by
  sorry

end NUMINAMATH_CALUDE_figure_to_square_approximation_l4092_409207


namespace NUMINAMATH_CALUDE_farm_animals_l4092_409264

theorem farm_animals (total_legs : ℕ) (total_animals : ℕ) (sheep : ℕ) :
  total_legs = 60 →
  total_animals = 20 →
  total_legs = 2 * (total_animals - sheep) + 4 * sheep →
  sheep = 10 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l4092_409264


namespace NUMINAMATH_CALUDE_parabola_intersection_l4092_409235

/-- The parabola y = x^2 - 2x - 3 intersects the x-axis at (-1, 0) and (3, 0) -/
theorem parabola_intersection (x : ℝ) :
  let y := x^2 - 2*x - 3
  (y = 0 ∧ x = -1) ∨ (y = 0 ∧ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l4092_409235


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l4092_409216

def problem (b : ℝ × ℝ) : Prop :=
  let a : ℝ × ℝ := (2, 1)
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = 4^2 ∧ 
  a.1 * b.1 + a.2 * b.2 = 1 →
  b.1^2 + b.2^2 = 3^2

theorem vector_magnitude_problem : ∀ b : ℝ × ℝ, problem b :=
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l4092_409216


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l4092_409233

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l4092_409233


namespace NUMINAMATH_CALUDE_maya_lift_increase_l4092_409296

/-- Given America's peak lift and Maya's relative lift capacities, calculate the increase in Maya's lift capacity. -/
theorem maya_lift_increase (america_peak : ℝ) (maya_initial_ratio : ℝ) (maya_peak_ratio : ℝ) 
  (h1 : america_peak = 300)
  (h2 : maya_initial_ratio = 1/4)
  (h3 : maya_peak_ratio = 1/2) :
  maya_peak_ratio * america_peak - maya_initial_ratio * america_peak = 75 := by
  sorry

end NUMINAMATH_CALUDE_maya_lift_increase_l4092_409296


namespace NUMINAMATH_CALUDE_almond_butter_servings_l4092_409236

/-- Represents a mixed number as a whole number part and a fraction part -/
structure MixedNumber where
  whole : ℕ
  numerator : ℕ
  denominator : ℕ
  denominator_pos : denominator > 0

/-- Converts a mixed number to a rational number -/
def mixedNumberToRational (m : MixedNumber) : ℚ :=
  m.whole + (m.numerator : ℚ) / m.denominator

theorem almond_butter_servings 
  (container_amount : MixedNumber) 
  (serving_size : ℚ) 
  (h1 : container_amount = ⟨37, 2, 3, by norm_num⟩) 
  (h2 : serving_size = 3) : 
  ∃ (result : MixedNumber), 
    mixedNumberToRational result = 
      mixedNumberToRational container_amount / serving_size ∧
    result = ⟨12, 5, 9, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_almond_butter_servings_l4092_409236


namespace NUMINAMATH_CALUDE_ratio_AD_BC_l4092_409284

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
def is_equilateral (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def is_right_triangle (B C D : ℝ × ℝ) : Prop :=
  (C.1 - B.1) * (D.1 - B.1) + (C.2 - B.2) * (D.2 - B.2) = 0

def BC_twice_BD (B C D : ℝ × ℝ) : Prop :=
  dist B C = 2 * dist B D

-- Theorem statement
theorem ratio_AD_BC (A B C D : ℝ × ℝ) 
  (h1 : is_equilateral A B C)
  (h2 : is_right_triangle B C D)
  (h3 : BC_twice_BD B C D) :
  dist A D / dist B C = 3/2 :=
sorry

end NUMINAMATH_CALUDE_ratio_AD_BC_l4092_409284


namespace NUMINAMATH_CALUDE_birdhouse_volume_difference_l4092_409257

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℚ := 12

/-- Volume of a rectangular prism -/
def volume (width height depth : ℚ) : ℚ := width * height * depth

/-- Sara's birdhouse dimensions in feet -/
def sara_width : ℚ := 1
def sara_height : ℚ := 2
def sara_depth : ℚ := 2

/-- Jake's birdhouse dimensions in inches -/
def jake_width : ℚ := 16
def jake_height : ℚ := 20
def jake_depth : ℚ := 18

/-- Theorem stating the difference in volume between Sara's and Jake's birdhouses -/
theorem birdhouse_volume_difference :
  volume (sara_width * feet_to_inches) (sara_height * feet_to_inches) (sara_depth * feet_to_inches) -
  volume jake_width jake_height jake_depth = 1152 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_volume_difference_l4092_409257


namespace NUMINAMATH_CALUDE_factor_divisor_statements_l4092_409202

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

def is_divisor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem factor_divisor_statements : 
  (is_factor 5 25) ∧ 
  (is_divisor 19 209 ∧ ¬is_divisor 19 63) ∧ 
  (is_divisor 20 80) ∧ 
  (is_divisor 14 28 ∧ is_divisor 14 56) ∧ 
  (is_factor 7 140) := by
  sorry

end NUMINAMATH_CALUDE_factor_divisor_statements_l4092_409202


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4092_409265

/-- Given that the solution set of ax^2 + bx + 2 > 0 is {x | -1/2 < x < 1/3}, prove that a - b = -10 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a - b = -10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4092_409265


namespace NUMINAMATH_CALUDE_inequality_implication_l4092_409214

theorem inequality_implication (a b x y : ℝ) (h1 : 1 < a) (h2 : a < b)
  (h3 : a^x + b^y ≤ a^(-x) + b^(-y)) : x + y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l4092_409214


namespace NUMINAMATH_CALUDE_negation_existence_gt_one_l4092_409243

theorem negation_existence_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_existence_gt_one_l4092_409243


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l4092_409254

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 6 = 4 ∧ ∀ m : ℕ, m < 100 ∧ m % 6 = 4 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l4092_409254


namespace NUMINAMATH_CALUDE_bill_difference_is_18_l4092_409260

-- Define the tip percentages and amount
def mike_tip_percent : ℚ := 15 / 100
def joe_tip_percent : ℚ := 25 / 100
def anna_tip_percent : ℚ := 10 / 100
def tip_amount : ℚ := 3

-- Define the bills as functions of the tip percentage
def bill (tip_percent : ℚ) : ℚ := tip_amount / tip_percent

-- Theorem statement
theorem bill_difference_is_18 :
  let mike_bill := bill mike_tip_percent
  let joe_bill := bill joe_tip_percent
  let anna_bill := bill anna_tip_percent
  let max_bill := max mike_bill (max joe_bill anna_bill)
  let min_bill := min mike_bill (min joe_bill anna_bill)
  max_bill - min_bill = 18 := by sorry

end NUMINAMATH_CALUDE_bill_difference_is_18_l4092_409260


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l4092_409289

-- Define the line L that point P moves on
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + 3 = 0}

-- Define the fixed point M
def M : ℝ × ℝ := (-1, 2)

-- Define the property that Q is on the extension of PM and |PM| = |MQ|
def Q_property (P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q = (t • (P - M) + M)

-- State the theorem
theorem trajectory_of_Q :
  ∀ Q : ℝ × ℝ, (∃ P ∈ L, Q_property P Q) → 2 * Q.1 - Q.2 + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l4092_409289


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4092_409287

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) : 
  i * i = -1 → (1 : ℂ) + i = z * ((1 : ℂ) - i) → Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4092_409287


namespace NUMINAMATH_CALUDE_cube_edge_length_in_pyramid_l4092_409277

/-- Represents a pyramid with an equilateral triangular base -/
structure EquilateralPyramid where
  base_side_length : ℝ
  apex_height : ℝ

/-- Represents a cube -/
structure Cube where
  edge_length : ℝ

/-- Theorem stating the edge length of the cube in the given pyramid configuration -/
theorem cube_edge_length_in_pyramid (p : EquilateralPyramid) (c : Cube) 
  (h1 : p.base_side_length = 3)
  (h2 : p.apex_height = 9)
  (h3 : c.edge_length * Real.sqrt 3 = p.apex_height) : 
  c.edge_length = 3 := by
  sorry


end NUMINAMATH_CALUDE_cube_edge_length_in_pyramid_l4092_409277


namespace NUMINAMATH_CALUDE_road_sign_difference_l4092_409271

/-- Represents the number of road signs at each intersection -/
structure RoadSigns where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The conditions of the road sign problem -/
def roadSignProblem (rs : RoadSigns) : Prop :=
  rs.first = 40 ∧
  rs.second = rs.first + rs.first / 4 ∧
  rs.third = 2 * rs.second ∧
  rs.fourth < rs.third ∧
  rs.first + rs.second + rs.third + rs.fourth = 270

theorem road_sign_difference (rs : RoadSigns) 
  (h : roadSignProblem rs) : rs.third - rs.fourth = 20 := by
  sorry

end NUMINAMATH_CALUDE_road_sign_difference_l4092_409271


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l4092_409256

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 24 ∧ ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  ∀ k : ℕ, k ≤ 24 → (k ∣ (n * (n + 1) * (n + 2) * (n + 3))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l4092_409256


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l4092_409226

theorem min_value_theorem (x : ℝ) (h : x > 4) :
  (x + 15) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 19 :=
by sorry

theorem min_value_achieved (x : ℝ) (h : x > 4) :
  ∃ x₀ > 4, (x₀ + 15) / Real.sqrt (x₀ - 4) = 2 * Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l4092_409226


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_k_equals_five_l4092_409283

/-- Given vectors a, b, and c in ℝ², prove that if (a - c) is parallel to b, then k = 5. -/
theorem parallel_vectors_imply_k_equals_five :
  let a : Fin 2 → ℝ := ![3, 1]
  let b : Fin 2 → ℝ := ![1, 3]
  let c : Fin 2 → ℝ := ![k, 7]
  (∃ (t : ℝ), (a - c) = t • b) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_k_equals_five_l4092_409283


namespace NUMINAMATH_CALUDE_locus_of_constant_sum_distances_l4092_409299

-- Define a type for lines in a plane
structure Line where
  -- Add necessary fields to represent a line

-- Define a type for points in a plane
structure Point where
  -- Add necessary fields to represent a point

-- Define a function to calculate the distance between a point and a line
def distance (p : Point) (l : Line) : ℝ :=
  sorry

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

-- Define a type for the locus
inductive Locus
  | Region
  | Parallelogram
  | Octagon

-- State the theorem
theorem locus_of_constant_sum_distances 
  (l1 l2 m1 m2 : Line) 
  (h_parallel1 : are_parallel l1 l2) 
  (h_parallel2 : are_parallel m1 m2) 
  (sum : ℝ) :
  ∃ (locus : Locus),
    ∀ (p : Point),
      distance p l1 + distance p l2 + distance p m1 + distance p m2 = sum →
      (((are_parallel l1 m1) ∧ (locus = Locus.Region)) ∨
       ((¬are_parallel l1 m1) ∧ ((locus = Locus.Parallelogram) ∨ (locus = Locus.Octagon)))) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_constant_sum_distances_l4092_409299


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l4092_409292

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x = x - 1) ↔ (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l4092_409292


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_f_geq_4_l4092_409246

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + 2 * |x - a|

-- Part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≤ x + 4} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 7/2} := by sorry

-- Part 2
theorem range_of_a_when_f_geq_4 :
  {a : ℝ | ∀ x, f a x ≥ 4} = {a : ℝ | a ≤ -5 ∨ a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_f_geq_4_l4092_409246


namespace NUMINAMATH_CALUDE_special_function_property_l4092_409242

/-- A function that is even, has period 2, and is monotonically decreasing on [-3, -2] -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + 2) = f x) ∧
  (∀ x y, -3 ≤ x ∧ x < y ∧ y ≤ -2 → f y < f x)

/-- Acute angle in a triangle -/
def is_acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < Real.pi / 2

theorem special_function_property 
  (f : ℝ → ℝ) 
  (h_f : special_function f) 
  (α β : ℝ) 
  (h_α : is_acute_angle α) 
  (h_β : is_acute_angle β) : 
  f (Real.sin α) > f (Real.cos β) := by
    sorry

end NUMINAMATH_CALUDE_special_function_property_l4092_409242


namespace NUMINAMATH_CALUDE_james_net_income_l4092_409285

def regular_price : ℝ := 20
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def maintenance_fee : ℝ := 35
def insurance_fee : ℝ := 15

def monday_hours : ℝ := 8
def wednesday_hours : ℝ := 8
def friday_hours : ℝ := 6
def sunday_hours : ℝ := 5

def total_hours : ℝ := monday_hours + wednesday_hours + friday_hours + sunday_hours
def rental_days : ℕ := 4

def discounted_rental : Bool := rental_days ≥ 3

theorem james_net_income :
  let total_rental_income := total_hours * regular_price
  let discounted_income := if discounted_rental then total_rental_income * (1 - discount_rate) else total_rental_income
  let income_with_tax := discounted_income * (1 + sales_tax_rate)
  let total_expenses := maintenance_fee + (insurance_fee * rental_days)
  let net_income := income_with_tax - total_expenses
  net_income = 415.30 := by sorry

end NUMINAMATH_CALUDE_james_net_income_l4092_409285


namespace NUMINAMATH_CALUDE_perpendicular_parallel_relations_l4092_409228

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

variable (α : Plane) (a b : Line)

-- State the theorem
theorem perpendicular_parallel_relations :
  (∀ a b : Line, ∀ α : Plane,
    (parallel a b ∧ perpendicular_line_plane a α → perpendicular_line_plane b α)) ∧
  (∀ a b : Line, ∀ α : Plane,
    (perpendicular_line_plane a α ∧ perpendicular_line_plane b α → parallel a b)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_relations_l4092_409228


namespace NUMINAMATH_CALUDE_triangle_height_l4092_409259

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 4 → area = 10 → area = (base * height) / 2 → height = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l4092_409259


namespace NUMINAMATH_CALUDE_optimal_distribution_maximizes_sum_l4092_409211

/-- Represents the distribution of blue balls between two boxes -/
structure Distribution where
  first_box : ℕ
  second_box : ℕ

/-- Calculates the sum of percentages of blue balls in each box -/
def sum_of_percentages (d : Distribution) : ℚ :=
  d.first_box / 24 + d.second_box / 23

/-- Checks if a distribution is valid given the total number of blue balls -/
def is_valid_distribution (d : Distribution) (total_blue : ℕ) : Prop :=
  d.first_box + d.second_box = total_blue ∧ d.first_box ≤ 24 ∧ d.second_box ≤ 23

theorem optimal_distribution_maximizes_sum :
  ∀ d : Distribution,
  is_valid_distribution d 25 →
  sum_of_percentages d ≤ sum_of_percentages { first_box := 2, second_box := 23 } :=
by sorry

end NUMINAMATH_CALUDE_optimal_distribution_maximizes_sum_l4092_409211


namespace NUMINAMATH_CALUDE_intersection_and_subset_l4092_409262

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | Real.sqrt (x - 1) ≥ 1}

theorem intersection_and_subset : 
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 3}) ∧
  (∀ a : ℝ, (A ∩ B) ⊆ {x | x ≥ a} ↔ a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_subset_l4092_409262


namespace NUMINAMATH_CALUDE_square_root_difference_squared_l4092_409267

theorem square_root_difference_squared : 
  (Real.sqrt (16 - 8 * Real.sqrt 3) - Real.sqrt (16 + 8 * Real.sqrt 3))^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_squared_l4092_409267


namespace NUMINAMATH_CALUDE_square_brush_ratio_l4092_409279

/-- A square with side length s and a brush of width w -/
structure SquareAndBrush where
  s : ℝ
  w : ℝ

/-- The painted area is one-third of the square's area -/
def paintedAreaIsOneThird (sb : SquareAndBrush) : Prop :=
  (1/2 * sb.w^2 + (sb.s - sb.w)^2 / 2) = sb.s^2 / 3

/-- The theorem to be proved -/
theorem square_brush_ratio (sb : SquareAndBrush) 
    (h : paintedAreaIsOneThird sb) : 
    sb.s / sb.w = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_brush_ratio_l4092_409279


namespace NUMINAMATH_CALUDE_a_minus_b_value_l4092_409294

theorem a_minus_b_value (a b : ℝ) (ha : |a| = 4) (hb : |b| = 2) (hab : |a + b| = a + b) :
  a - b = 2 ∨ a - b = 6 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l4092_409294


namespace NUMINAMATH_CALUDE_halfway_fraction_l4092_409275

theorem halfway_fraction : 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 7
  let midpoint := (a + b) / 2
  midpoint = (41 : ℚ) / 56 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l4092_409275


namespace NUMINAMATH_CALUDE_unique_solution_for_rational_equation_l4092_409290

theorem unique_solution_for_rational_equation :
  let k : ℚ := -3/4
  let f (x : ℚ) := (x + 3)/(k*x - 2) - x
  ∃! x, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_rational_equation_l4092_409290


namespace NUMINAMATH_CALUDE_sector_area_l4092_409234

/-- Given a sector with central angle 7/(2π) and arc length 7, its area is 7π. -/
theorem sector_area (central_angle : Real) (arc_length : Real) (area : Real) :
  central_angle = 7 / (2 * Real.pi) →
  arc_length = 7 →
  area = 7 * Real.pi :=
by
  sorry

#check sector_area

end NUMINAMATH_CALUDE_sector_area_l4092_409234


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4092_409238

open Set Real

theorem inequality_solution_set : 
  let S := {x : ℝ | (π/2)^((x-1)^2) ≤ (2/π)^(x^2-5*x-5)}
  S = Icc (-1/2) 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4092_409238


namespace NUMINAMATH_CALUDE_candle_equality_l4092_409280

/-- Represents the number of times each candle is used over n Sundays -/
def total_usage (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the number of times each individual candle is used -/
def individual_usage (n : ℕ) : ℚ := (n + 1) / 2

/-- Theorem stating that for all candles to be of equal length after n Sundays,
    n must be a positive odd integer -/
theorem candle_equality (n : ℕ) (h : n > 0) :
  (∀ (i : ℕ), i ≤ n → (individual_usage n).num % (individual_usage n).den = 0) ↔
  n % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_candle_equality_l4092_409280


namespace NUMINAMATH_CALUDE_soda_price_increase_l4092_409213

/-- Proves that the percentage increase in the price of a can of soda is 50% given the specified conditions. -/
theorem soda_price_increase (initial_combined_price new_candy_price new_soda_price candy_increase : ℝ) 
  (h1 : initial_combined_price = 16)
  (h2 : new_candy_price = 15)
  (h3 : new_soda_price = 6)
  (h4 : candy_increase = 25)
  : (new_soda_price - (initial_combined_price - new_candy_price / (1 + candy_increase / 100))) / 
    (initial_combined_price - new_candy_price / (1 + candy_increase / 100)) * 100 = 50 := by
  sorry

#check soda_price_increase

end NUMINAMATH_CALUDE_soda_price_increase_l4092_409213


namespace NUMINAMATH_CALUDE_six_digit_number_puzzle_l4092_409278

theorem six_digit_number_puzzle : ∃! n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧
  n % 10 = 7 ∧
  7 * 100000 + n / 10 = 5 * n ∧
  n = 142857 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_number_puzzle_l4092_409278


namespace NUMINAMATH_CALUDE_kieras_envelopes_l4092_409245

theorem kieras_envelopes :
  ∀ (yellow : ℕ),
  let blue := 14
  let green := 3 * yellow
  yellow < blue →
  blue + yellow + green = 46 →
  blue - yellow = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_kieras_envelopes_l4092_409245


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l4092_409237

theorem sqrt_product_equality : 2 * Real.sqrt 3 * (3 * Real.sqrt 2) = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l4092_409237


namespace NUMINAMATH_CALUDE_median_of_100_numbers_l4092_409268

def is_median (s : Finset ℕ) (m : ℕ) : Prop :=
  2 * (s.filter (· < m)).card ≤ s.card ∧ 2 * (s.filter (· > m)).card ≤ s.card

theorem median_of_100_numbers (s : Finset ℕ) (h_card : s.card = 100) :
  (∃ x ∈ s, is_median (s.erase x) 78) →
  (∃ y ∈ s, y ≠ x → is_median (s.erase y) 66) →
  is_median s 72 :=
sorry

end NUMINAMATH_CALUDE_median_of_100_numbers_l4092_409268


namespace NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l4092_409219

def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

theorem quadratic_function_satisfies_conditions :
  (f 1 = 0) ∧ (f 5 = 0) ∧ (f 3 = 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l4092_409219


namespace NUMINAMATH_CALUDE_speed_ratio_l4092_409276

/-- The speed of object A -/
def v_A : ℝ := sorry

/-- The speed of object B -/
def v_B : ℝ := sorry

/-- The distance B is initially short of O -/
def initial_distance : ℝ := 600

/-- The time when A and B are first equidistant from O -/
def t1 : ℝ := 3

/-- The time when A and B are again equidistant from O -/
def t2 : ℝ := 12

/-- The theorem stating the ratio of speeds -/
theorem speed_ratio : v_A / v_B = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_l4092_409276
