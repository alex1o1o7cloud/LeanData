import Mathlib

namespace NUMINAMATH_CALUDE_sequence_properties_l808_80845

def a (n : ℕ) : ℤ := 2^n - (-1)^n

theorem sequence_properties :
  (∀ k : ℕ, k > 0 →
    (a k + a (k + 2) = 2 * a (k + 1)) ↔ k = 2) ∧
  (∀ r s : ℕ, r > 1 ∧ s > r →
    (a 1 + a s = 2 * a r) → s = r + 1) ∧
  (∀ q r s t : ℕ, 0 < q ∧ q < r ∧ r < s ∧ s < t →
    ¬(a q + a t = a r + a s)) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l808_80845


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l808_80870

theorem coefficient_x_squared_in_expansion : ∃ (a b c d e : ℤ), 
  (2 * X + 1)^2 * (X - 2)^3 = a * X^5 + b * X^4 + c * X^3 + 10 * X^2 + d * X + e :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l808_80870


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l808_80866

theorem partial_fraction_decomposition_product (M₁ M₂ : ℝ) :
  (∀ x : ℝ, x ≠ 1 → x ≠ 3 → (45 * x - 36) / (x^2 - 4*x + 3) = M₁ / (x - 1) + M₂ / (x - 3)) →
  M₁ * M₂ = -222.75 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l808_80866


namespace NUMINAMATH_CALUDE_f_increasing_l808_80848

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x

-- Theorem statement
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_l808_80848


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l808_80847

def total_potatoes : ℕ := 15
def cooked_potatoes : ℕ := 8
def cooking_time_per_potato : ℕ := 9

theorem remaining_cooking_time :
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 63 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l808_80847


namespace NUMINAMATH_CALUDE_max_sum_cubes_l808_80879

theorem max_sum_cubes (x y z w : ℝ) (h : x^2 + y^2 + z^2 + w^2 = 16) :
  ∃ (M : ℝ), (∀ a b c d : ℝ, a^2 + b^2 + c^2 + d^2 = 16 → a^3 + b^3 + c^3 + d^3 ≤ M) ∧
             (∃ p q r s : ℝ, p^2 + q^2 + r^2 + s^2 = 16 ∧ p^3 + q^3 + r^3 + s^3 = M) ∧
             M = 64 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l808_80879


namespace NUMINAMATH_CALUDE_tripled_base_doubled_exponent_l808_80887

theorem tripled_base_doubled_exponent 
  (c y : ℝ) (d : ℝ) (h_d : d ≠ 0) :
  let s := (3 * c) ^ (2 * d)
  s = c^d / y^d →
  y = 1 / (9 * c) := by
sorry

end NUMINAMATH_CALUDE_tripled_base_doubled_exponent_l808_80887


namespace NUMINAMATH_CALUDE_parabola_symmetry_range_l808_80840

theorem parabola_symmetry_range (a : ℝ) : 
  a > 0 → 
  (∀ x y : ℝ, y = a * x^2 - 1 → 
    ∃ x1 y1 x2 y2 : ℝ, 
      y1 = a * x1^2 - 1 ∧ 
      y2 = a * x2^2 - 1 ∧ 
      x1 + y1 = -(x2 + y2) ∧ 
      (x1 ≠ x2 ∨ y1 ≠ y2)) → 
  a > 3/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_symmetry_range_l808_80840


namespace NUMINAMATH_CALUDE_apples_per_case_l808_80824

theorem apples_per_case (total_apples : ℕ) (num_cases : ℕ) (h1 : total_apples = 1080) (h2 : num_cases = 90) :
  total_apples / num_cases = 12 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_case_l808_80824


namespace NUMINAMATH_CALUDE_store_profit_l808_80822

theorem store_profit (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) :
  price = 64 ∧ 
  profit_percent = 60 ∧ 
  loss_percent = 20 →
  let cost1 := price / (1 + profit_percent / 100)
  let cost2 := price / (1 - loss_percent / 100)
  price * 2 - (cost1 + cost2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_store_profit_l808_80822


namespace NUMINAMATH_CALUDE_a_must_be_negative_l808_80813

theorem a_must_be_negative (a b : ℝ) (hb : b > 0) (h : a / b < -2/3) : a < 0 := by
  sorry

end NUMINAMATH_CALUDE_a_must_be_negative_l808_80813


namespace NUMINAMATH_CALUDE_ellipse_focal_coordinates_specific_ellipse_focal_coordinates_l808_80861

/-- The focal coordinates of an ellipse with equation x²/a² + y²/b² = 1 are (±c, 0) where c² = a² - b² -/
theorem ellipse_focal_coordinates (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let c := Real.sqrt (a^2 - b^2)
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  (∃ x : ℝ, x = c ∨ x = -c) ∧ (∀ x : ℝ, x^2 = c^2 → x = c ∨ x = -c) :=
by sorry

/-- The focal coordinates of the ellipse x²/5 + y²/4 = 1 are (±1, 0) -/
theorem specific_ellipse_focal_coordinates :
  let c := Real.sqrt (5 - 4)
  (∀ x y : ℝ, x^2 / 5 + y^2 / 4 = 1) →
  (∃ x : ℝ, x = 1 ∨ x = -1) ∧ (∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_coordinates_specific_ellipse_focal_coordinates_l808_80861


namespace NUMINAMATH_CALUDE_f_properties_l808_80816

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x₁ x₂ : ℝ, -1 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (¬ ∃ x : ℝ, x < 0 ∧ f a x = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l808_80816


namespace NUMINAMATH_CALUDE_fifth_month_sale_l808_80890

def sales_first_four : List ℕ := [6235, 6927, 6855, 7230]
def required_sixth : ℕ := 5191
def desired_average : ℕ := 6500
def num_months : ℕ := 6

theorem fifth_month_sale :
  let total_required : ℕ := desired_average * num_months
  let sum_known : ℕ := (sales_first_four.sum + required_sixth)
  let fifth_month : ℕ := total_required - sum_known
  fifth_month = 6562 := by sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l808_80890


namespace NUMINAMATH_CALUDE_dentist_age_fraction_l808_80899

/-- Given a dentist's current age A and a fraction F, proves that F = 1/10 when A = 32 and (1/6) * (A - 8) = F * (A + 8) -/
theorem dentist_age_fraction (A : ℕ) (F : ℚ) 
  (h1 : A = 32) 
  (h2 : (1/6 : ℚ) * ((A : ℚ) - 8) = F * ((A : ℚ) + 8)) : 
  F = 1/10 := by sorry

end NUMINAMATH_CALUDE_dentist_age_fraction_l808_80899


namespace NUMINAMATH_CALUDE_sin_cos_value_l808_80808

theorem sin_cos_value (x : Real) (h : 2 * Real.sin x = 5 * Real.cos x) : 
  Real.sin x * Real.cos x = 10 / 29 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_value_l808_80808


namespace NUMINAMATH_CALUDE_four_steps_on_number_line_l808_80853

/-- Given a number line with equally spaced markings where the distance from 0 to 25
    is covered in 7 steps, prove that the number reached after 4 steps from 0 is 100/7. -/
theorem four_steps_on_number_line :
  ∀ (step_length : ℚ),
  step_length * 7 = 25 →
  4 * step_length = 100 / 7 := by
sorry

end NUMINAMATH_CALUDE_four_steps_on_number_line_l808_80853


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l808_80839

theorem simultaneous_equations_solution (n : ℝ) :
  n ≠ (1/2 : ℝ) ↔ ∃ (x y : ℝ), y = (3*n + 1)*x + 2 ∧ y = (5*n - 2)*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l808_80839


namespace NUMINAMATH_CALUDE_fruit_sales_problem_l808_80883

/-- Fruit sales problem -/
theorem fruit_sales_problem 
  (cost_price : ℝ) 
  (base_price : ℝ) 
  (base_sales : ℝ) 
  (price_increment : ℝ) 
  (sales_decrement : ℝ) 
  (min_sales : ℝ) 
  (max_price : ℝ) :
  cost_price = 8 →
  base_price = 10 →
  base_sales = 300 →
  price_increment = 1 →
  sales_decrement = 50 →
  min_sales = 250 →
  max_price = 13 →
  ∃ (sales_function : ℝ → ℝ) (max_profit : ℝ) (donation_range : Set ℝ),
    -- 1. Sales function
    (∀ x, sales_function x = -50 * x + 800) ∧
    -- 2. Maximum profit
    max_profit = 750 ∧
    -- 3. Donation range
    donation_range = {a : ℝ | 2 ≤ a ∧ a ≤ 2.5} :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_sales_problem_l808_80883


namespace NUMINAMATH_CALUDE_divisible_by_five_problem_l808_80897

theorem divisible_by_five_problem (n : ℕ) : 
  n % 5 = 0 ∧ n / 5 = 96 → (n + 17) * 69 = 34293 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_problem_l808_80897


namespace NUMINAMATH_CALUDE_partitions_6_3_l808_80805

def partitions (n : ℕ) (k : ℕ) : ℕ := sorry

theorem partitions_6_3 : partitions 6 3 = 7 := by sorry

end NUMINAMATH_CALUDE_partitions_6_3_l808_80805


namespace NUMINAMATH_CALUDE_kite_AC_length_l808_80859

-- Define the kite ABCD
structure Kite :=
  (A B C D : ℝ × ℝ)
  (diagonals_perpendicular : (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0)
  (BD_length : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 10)
  (AB_equals_BC : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 
                  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2))
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13)
  (AD_equals_DC : Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 
                  Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2))

-- Theorem statement
theorem kite_AC_length (k : Kite) : 
  Real.sqrt ((k.A.1 - k.C.1)^2 + (k.A.2 - k.C.2)^2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_kite_AC_length_l808_80859


namespace NUMINAMATH_CALUDE_geometric_squares_existence_and_uniqueness_l808_80825

theorem geometric_squares_existence_and_uniqueness :
  ∃! k : ℤ,
    (∃ a b c : ℤ,
      (49 + k = a^2) ∧
      (441 + k = b^2) ∧
      (961 + k = c^2) ∧
      (∃ r : ℚ, b = r * a ∧ c = r * b)) ∧
    k = 1152 := by
  sorry

end NUMINAMATH_CALUDE_geometric_squares_existence_and_uniqueness_l808_80825


namespace NUMINAMATH_CALUDE_theater_revenue_calculation_l808_80854

/-- Calculates the total revenue of a movie theater for a day --/
def theater_revenue (
  matinee_ticket_price evening_ticket_price opening_night_ticket_price : ℕ)
  (matinee_popcorn_price evening_popcorn_price opening_night_popcorn_price : ℕ)
  (matinee_drink_price evening_drink_price opening_night_drink_price : ℕ)
  (matinee_customers evening_customers opening_night_customers : ℕ)
  (popcorn_ratio drink_ratio : ℚ)
  (discount_groups : ℕ)
  (discount_group_size : ℕ)
  (discount_percentage : ℚ) : ℕ :=
  sorry

theorem theater_revenue_calculation :
  theater_revenue 5 7 10 8 10 12 3 4 5 32 40 58 (1/2) (1/4) 4 5 (1/10) = 1778 := by
  sorry

end NUMINAMATH_CALUDE_theater_revenue_calculation_l808_80854


namespace NUMINAMATH_CALUDE_simplified_fraction_double_l808_80892

theorem simplified_fraction_double (b : ℝ) :
  b = 5 → 2 * ((15 * b^4) / (75 * b^3)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_double_l808_80892


namespace NUMINAMATH_CALUDE_price_increase_proof_l808_80872

theorem price_increase_proof (x : ℝ) : 
  (1 + x)^2 = 1.44 → x = 0.2 := by sorry

end NUMINAMATH_CALUDE_price_increase_proof_l808_80872


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_l808_80844

def small_bottle_capacity : ℝ := 35
def large_bottle_capacity : ℝ := 500

theorem min_bottles_to_fill (small_cap large_cap : ℝ) (h1 : small_cap = small_bottle_capacity) (h2 : large_cap = large_bottle_capacity) :
  ⌈large_cap / small_cap⌉ = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_bottles_to_fill_l808_80844


namespace NUMINAMATH_CALUDE_weight_loss_duration_l808_80820

/-- Calculates the number of months required to reach a target weight given initial weight, weight loss per month, and target weight. -/
def months_to_reach_weight (initial_weight : ℕ) (weight_loss_per_month : ℕ) (target_weight : ℕ) : ℕ :=
  (initial_weight - target_weight) / weight_loss_per_month

/-- Proves that it takes 12 months to reduce weight from 250 pounds to 154 pounds, losing 8 pounds per month. -/
theorem weight_loss_duration :
  months_to_reach_weight 250 8 154 = 12 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_duration_l808_80820


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l808_80878

theorem rectangular_parallelepiped_volume
  (m n p d : ℝ) 
  (h_positive : m > 0 ∧ n > 0 ∧ p > 0 ∧ d > 0) :
  ∃ (V : ℝ), V = (m * n * p * d^3) / (m^2 + n^2 + p^2)^(3/2) ∧
  ∃ (a b c : ℝ), 
    a / m = b / n ∧ 
    b / n = c / p ∧
    V = a * b * c ∧
    d^2 = a^2 + b^2 + c^2 := by
  sorry

#check rectangular_parallelepiped_volume

end NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l808_80878


namespace NUMINAMATH_CALUDE_cricket_team_members_l808_80877

/-- The number of members in a cricket team satisfying specific age conditions. -/
theorem cricket_team_members : ∃ (n : ℕ),
  n > 0 ∧
  let captain_age : ℕ := 26
  let keeper_age : ℕ := captain_age + 5
  let team_avg_age : ℚ := 24
  let remaining_avg_age : ℚ := team_avg_age - 1
  n * team_avg_age = (n - 2) * remaining_avg_age + (captain_age + keeper_age) ∧
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_members_l808_80877


namespace NUMINAMATH_CALUDE_salary_problem_l808_80898

theorem salary_problem (total_salary : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ) 
  (h1 : total_salary = 14000)
  (h2 : a_spend_rate = 0.8)
  (h3 : b_spend_rate = 0.85)
  (h4 : (1 - a_spend_rate) * (total_salary - b_salary) = (1 - b_spend_rate) * b_salary) :
  b_salary = 8000 :=
by sorry

end NUMINAMATH_CALUDE_salary_problem_l808_80898


namespace NUMINAMATH_CALUDE_delivery_cost_fraction_l808_80882

/-- Proves that the fraction of the remaining amount spent on delivery costs is 1/4 -/
theorem delivery_cost_fraction (total_cost : ℝ) (salary_fraction : ℝ) (order_cost : ℝ)
  (h1 : total_cost = 4000)
  (h2 : salary_fraction = 2/5)
  (h3 : order_cost = 1800) :
  let salary_cost := salary_fraction * total_cost
  let remaining_after_salary := total_cost - salary_cost
  let delivery_cost := remaining_after_salary - order_cost
  delivery_cost / remaining_after_salary = 1/4 := by
sorry

end NUMINAMATH_CALUDE_delivery_cost_fraction_l808_80882


namespace NUMINAMATH_CALUDE_inscribed_pentagon_segments_l808_80891

/-- Represents a convex pentagon with an inscribed circle -/
structure InscribedPentagon where
  -- Side lengths
  FG : ℝ
  GH : ℝ
  HI : ℝ
  IJ : ℝ
  JF : ℝ
  -- Segment lengths from vertices to tangent points
  x : ℝ
  y : ℝ
  z : ℝ
  -- Properties
  convex : Bool
  inscribed : Bool
  -- Relationships between segments and sides
  eq1 : x + y = GH
  eq2 : x + z = FG
  eq3 : y + z = JF

/-- Theorem: Given the specific side lengths, the segment lengths are determined -/
theorem inscribed_pentagon_segments
  (p : InscribedPentagon)
  (h1 : p.FG = 7)
  (h2 : p.GH = 8)
  (h3 : p.HI = 8)
  (h4 : p.IJ = 8)
  (h5 : p.JF = 9)
  (h6 : p.convex)
  (h7 : p.inscribed) :
  p.x = 3 ∧ p.y = 5 ∧ p.z = 4 := by
  sorry

#check inscribed_pentagon_segments

end NUMINAMATH_CALUDE_inscribed_pentagon_segments_l808_80891


namespace NUMINAMATH_CALUDE_square_root_of_nine_l808_80874

theorem square_root_of_nine : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l808_80874


namespace NUMINAMATH_CALUDE_sufficient_conditions_for_positive_product_l808_80802

theorem sufficient_conditions_for_positive_product (a b : ℝ) :
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) ∨ (a > 1 ∧ b > 1)) → a * b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_conditions_for_positive_product_l808_80802


namespace NUMINAMATH_CALUDE_division_remainder_problem_l808_80885

theorem division_remainder_problem (smaller : ℕ) : 
  1614 - smaller = 1360 →
  1614 / smaller = 6 →
  1614 % smaller = 90 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l808_80885


namespace NUMINAMATH_CALUDE_min_white_fraction_is_one_eighth_l808_80880

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cubes : ℕ
  red_cubes : ℕ
  white_cubes : ℕ

/-- Calculates the surface area of a cube -/
def surface_area (c : LargeCube) : ℕ := 6 * c.edge_length * c.edge_length

/-- Calculates the minimum number of white cubes needed to have at least one on each face -/
def min_white_cubes_on_surface : ℕ := 4

/-- Calculates the white surface area when white cubes are placed optimally -/
def white_surface_area : ℕ := min_white_cubes_on_surface * 3

/-- The theorem to be proved -/
theorem min_white_fraction_is_one_eighth (c : LargeCube) 
    (h1 : c.edge_length = 4)
    (h2 : c.small_cubes = 64)
    (h3 : c.red_cubes = 56)
    (h4 : c.white_cubes = 8) :
  (white_surface_area : ℚ) / (surface_area c : ℚ) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_min_white_fraction_is_one_eighth_l808_80880


namespace NUMINAMATH_CALUDE_class_mean_calculation_l808_80886

theorem class_mean_calculation (total_students : ℕ) (first_group : ℕ) (second_group : ℕ)
  (first_mean : ℚ) (second_mean : ℚ) :
  total_students = first_group + second_group →
  first_group = 40 →
  second_group = 10 →
  first_mean = 68 / 100 →
  second_mean = 74 / 100 →
  (first_group * first_mean + second_group * second_mean) / total_students = 692 / 1000 := by
sorry

#eval (40 * (68 : ℚ) / 100 + 10 * (74 : ℚ) / 100) / 50

end NUMINAMATH_CALUDE_class_mean_calculation_l808_80886


namespace NUMINAMATH_CALUDE_unique_integer_pair_l808_80869

theorem unique_integer_pair : ∃! (x y : ℕ+), 
  (x.val : ℝ) ^ (y.val : ℝ) + 1 = (y.val : ℝ) ^ (x.val : ℝ) ∧ 
  2 * (x.val : ℝ) ^ (y.val : ℝ) = (y.val : ℝ) ^ (x.val : ℝ) + 13 ∧ 
  x.val = 2 ∧ y.val = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_pair_l808_80869


namespace NUMINAMATH_CALUDE_product_equals_four_l808_80867

theorem product_equals_four (a b c : ℝ) 
  (h : ∀ x y z : ℝ, x * y * z = (Real.sqrt ((x + 2) * (y + 3))) / (z + 1)) : 
  6 * 15 * 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_product_equals_four_l808_80867


namespace NUMINAMATH_CALUDE_quadratic_common_roots_l808_80811

theorem quadratic_common_roots (p : ℚ) (x : ℚ) : 
  (x^2 - (p+1)*x + (p+1) = 0 ∧ 2*x^2 + (p-2)*x - p - 7 = 0) ↔ 
  ((p = 3 ∧ x = 2) ∨ (p = -3/2 ∧ x = -1)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_common_roots_l808_80811


namespace NUMINAMATH_CALUDE_closest_multiple_of_15_to_2021_l808_80851

theorem closest_multiple_of_15_to_2021 : ∃ (n : ℤ), 
  15 * n = 2025 ∧ 
  ∀ (m : ℤ), m ≠ n → 15 * m ≠ 2025 → |2021 - 15 * n| ≤ |2021 - 15 * m| := by
  sorry

end NUMINAMATH_CALUDE_closest_multiple_of_15_to_2021_l808_80851


namespace NUMINAMATH_CALUDE_student_line_arrangements_l808_80855

theorem student_line_arrangements (n : ℕ) (h : n = 5) :
  (n.factorial : ℕ) - (((n - 1).factorial : ℕ) * 2) = 72 := by
  sorry

end NUMINAMATH_CALUDE_student_line_arrangements_l808_80855


namespace NUMINAMATH_CALUDE_equation_solutions_l808_80875

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 10*x - 12) + 1 / (x^2 + 3*x - 12) + 1 / (x^2 - 14*x - 12) = 0

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x = 1 ∨ x = -21 ∨ x = 5 + Real.sqrt 37 ∨ x = 5 - Real.sqrt 37 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l808_80875


namespace NUMINAMATH_CALUDE_polygon_sides_l808_80835

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1080 → ∃ n : ℕ, n = 8 ∧ sum_interior_angles = (n - 2) * 180 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l808_80835


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_circumscribing_cube_l808_80814

theorem sphere_surface_area_from_circumscribing_cube (cube_volume : ℝ) (sphere_surface_area : ℝ) : 
  cube_volume = 8 → sphere_surface_area = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_circumscribing_cube_l808_80814


namespace NUMINAMATH_CALUDE_nested_expression_equals_one_l808_80828

theorem nested_expression_equals_one :
  (3 * (3 * (3 * (3 * (3 * (3 - 2) - 2) - 2) - 2) - 2) - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_equals_one_l808_80828


namespace NUMINAMATH_CALUDE_solution_to_exponential_equation_l808_80810

theorem solution_to_exponential_equation :
  ∃ y : ℝ, (3 : ℝ)^(y - 2) = (9 : ℝ)^(y + 2) ∧ y = -6 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_exponential_equation_l808_80810


namespace NUMINAMATH_CALUDE_l₂_passes_through_fixed_point_l808_80871

/-- A line in 2D space defined by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The point of symmetry -/
def symmetryPoint : ℝ × ℝ := (2, 1)

/-- Line l₁ defined as y = k(x - 4) -/
def l₁ (k : ℝ) : Line :=
  { slope := k, point := (4, 0) }

/-- Reflect a point about the symmetry point -/
def reflect (p : ℝ × ℝ) : ℝ × ℝ :=
  (2 * symmetryPoint.1 - p.1, 2 * symmetryPoint.2 - p.2)

/-- Line l₂ symmetric to l₁ about the symmetry point -/
def l₂ (k : ℝ) : Line :=
  { slope := -k, point := reflect (l₁ k).point }

theorem l₂_passes_through_fixed_point :
  ∀ k : ℝ, (l₂ k).point = (0, 2) := by sorry

end NUMINAMATH_CALUDE_l₂_passes_through_fixed_point_l808_80871


namespace NUMINAMATH_CALUDE_power_inequality_l808_80843

theorem power_inequality : 22^55 > 33^44 ∧ 33^44 > 55^33 ∧ 55^33 > 66^22 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l808_80843


namespace NUMINAMATH_CALUDE_other_number_proof_l808_80865

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 8820)
  (h2 : Nat.gcd a b = 36)
  (h3 : a = 360) :
  b = 882 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l808_80865


namespace NUMINAMATH_CALUDE_area_fold_points_specific_triangle_l808_80857

/-- Represents a right triangle ABC -/
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  angleB : ℝ

/-- Represents the area of fold points -/
def area_fold_points (t : RightTriangle) : ℝ := sorry

/-- Main theorem: Area of fold points for the given right triangle -/
theorem area_fold_points_specific_triangle :
  let t : RightTriangle := { AB := 45, AC := 90, angleB := 90 }
  area_fold_points t = 379 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_area_fold_points_specific_triangle_l808_80857


namespace NUMINAMATH_CALUDE_greatest_power_of_two_l808_80833

theorem greatest_power_of_two (n : ℕ) : 
  ∃ k : ℕ, 2^k ∣ (10^1503 - 4^752) ∧ ∀ m : ℕ, 2^m ∣ (10^1503 - 4^752) → m ≤ k := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_l808_80833


namespace NUMINAMATH_CALUDE_w_expression_l808_80868

theorem w_expression (x y z w : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
  (eq : 1/x + 1/y + 1/z = 1/w) : 
  w = x*y*z / (y*z + x*z + x*y) := by
sorry

end NUMINAMATH_CALUDE_w_expression_l808_80868


namespace NUMINAMATH_CALUDE_parabola_line_intersection_right_angle_l808_80850

/-- Parabola represented by the equation y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  isParabola : equation = fun x y => y^2 = 4*x

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line represented by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Angle between two vectors -/
def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_line_intersection_right_angle 
  (E : Parabola) 
  (M N : Point)
  (MN : Line)
  (A B : Point)
  (h1 : M.x = 1 ∧ M.y = -3)
  (h2 : N.x = 5 ∧ N.y = 1)
  (h3 : MN.p1 = M ∧ MN.p2 = N)
  (h4 : E.equation A.x A.y ∧ E.equation B.x B.y)
  (h5 : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
        A.x = M.x + t * (N.x - M.x) ∧ 
        A.y = M.y + t * (N.y - M.y))
  (h6 : ∃ s : ℝ, 0 < s ∧ s < 1 ∧ 
        B.x = M.x + s * (N.x - M.x) ∧ 
        B.y = M.y + s * (N.y - M.y))
  : angle (A.x, A.y) (B.x, B.y) = π / 2 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_right_angle_l808_80850


namespace NUMINAMATH_CALUDE_tree_count_proof_l808_80896

theorem tree_count_proof (total : ℕ) (pine_fraction : ℚ) (fir_percent : ℚ) 
  (h1 : total = 520)
  (h2 : pine_fraction = 1 / 3)
  (h3 : fir_percent = 25 / 100) :
  ⌊total * pine_fraction⌋ + ⌊total * fir_percent⌋ = 390 := by
  sorry

end NUMINAMATH_CALUDE_tree_count_proof_l808_80896


namespace NUMINAMATH_CALUDE_polynomial_factorization_l808_80812

theorem polynomial_factorization (a b : ℝ) : a^2 + 2*b - b^2 - 1 = (a-b+1)*(a+b-1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l808_80812


namespace NUMINAMATH_CALUDE_cubic_root_form_l808_80876

theorem cubic_root_form : ∃ (x : ℝ), 
  16 * x^3 - 4 * x^2 - 4 * x - 1 = 0 ∧ 
  x = (Real.rpow 256 (1/3 : ℝ) + Real.rpow 16 (1/3 : ℝ) + 1) / 16 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_form_l808_80876


namespace NUMINAMATH_CALUDE_percentage_of_450_is_172_8_l808_80826

theorem percentage_of_450_is_172_8 : 
  ∃ p : ℝ, (p / 100) * 450 = 172.8 ∧ p = 38.4 := by sorry

end NUMINAMATH_CALUDE_percentage_of_450_is_172_8_l808_80826


namespace NUMINAMATH_CALUDE_backpack_cost_relationship_l808_80856

theorem backpack_cost_relationship (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive for division
  (h2 : 810 > 0) -- Cost of type A backpacks is positive
  (h3 : 600 > 0) -- Cost of type B backpacks is positive
  (h4 : x + 20 > 0) -- Ensure denominator is positive
  : 
  810 / (x + 20) = (600 / x) * (1 - 0.1) :=
sorry

end NUMINAMATH_CALUDE_backpack_cost_relationship_l808_80856


namespace NUMINAMATH_CALUDE_cyclic_inequality_l808_80862

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 / (a^2 + a*b + b^2) + b^3 / (b^2 + b*c + c^2) + c^3 / (c^2 + c*a + a^2) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l808_80862


namespace NUMINAMATH_CALUDE_multiple_subtraction_problem_l808_80863

theorem multiple_subtraction_problem (n : ℝ) (m : ℝ) : 
  n = 6 → m * n - 6 = 2 * n → m * n = 18 := by
  sorry

end NUMINAMATH_CALUDE_multiple_subtraction_problem_l808_80863


namespace NUMINAMATH_CALUDE_complement_M_l808_80849

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x^2 - 4 ≤ 0}

theorem complement_M : Set.compl M = {x : ℝ | x > 2 ∨ x < -2} := by sorry

end NUMINAMATH_CALUDE_complement_M_l808_80849


namespace NUMINAMATH_CALUDE_number_problem_l808_80801

theorem number_problem (x : ℚ) : 4 * x + 7 * x = 55 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l808_80801


namespace NUMINAMATH_CALUDE_angle_B_is_60_degrees_side_c_and_area_l808_80873

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a < t.b ∧ t.b < t.c ∧
  Real.sqrt 3 * t.a = 2 * t.b * Real.sin t.A

-- Theorem 1: Prove that angle B is 60 degrees
theorem angle_B_is_60_degrees (t : Triangle) (h : validTriangle t) :
  t.B = Real.pi / 3 := by
  sorry

-- Theorem 2: Prove side c length and area when a = 2 and b = √7
theorem side_c_and_area (t : Triangle) (h : validTriangle t)
  (ha : t.a = 2) (hb : t.b = Real.sqrt 7) :
  t.c = 3 ∧ (1/2 * t.a * t.c * Real.sin t.B) = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_60_degrees_side_c_and_area_l808_80873


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l808_80829

/-- Given an arithmetic sequence of 5 terms, prove that the first term is 1/6 under specific conditions --/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
  (a 0 + a 1 + a 2 + a 3 + a 4 = 10) →  -- sum of all terms is 10
  (a 2 + a 3 + a 4 = (1 / 7) * (a 0 + a 1)) →  -- sum of larger three is 1/7 of sum of smaller two
  a 0 = 1 / 6 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l808_80829


namespace NUMINAMATH_CALUDE_magazine_publication_theorem_l808_80889

/-- Represents a magazine issue -/
structure Issue :=
  (year : ℕ)
  (month : ℕ)
  (exercisePosition : ℕ)
  (problemPosition : ℕ)

/-- The publication schedule of the magazine -/
def publicationSchedule : 
  (exercisesPerIssue : ℕ) → 
  (problemsPerIssue : ℕ) → 
  (issuesPerYear : ℕ) → 
  (startYear : ℕ) → 
  (lastExerciseNumber : ℕ) → 
  (lastProblemNumber : ℕ) → 
  (Prop) :=
  λ exercisesPerIssue problemsPerIssue issuesPerYear startYear lastExerciseNumber lastProblemNumber =>
    ∃ (exerciseIssue problemIssue : Issue),
      -- The exercise issue is in 1979, 3rd month, 2nd exercise
      exerciseIssue.year = 1979 ∧
      exerciseIssue.month = 3 ∧
      exerciseIssue.exercisePosition = 2 ∧
      -- The problem issue is in 1973, 5th month, 5th problem
      problemIssue.year = 1973 ∧
      problemIssue.month = 5 ∧
      problemIssue.problemPosition = 5 ∧
      -- The serial numbers match the respective years
      (lastExerciseNumber + (exerciseIssue.year - startYear) * exercisesPerIssue * issuesPerYear + 
       (exerciseIssue.month - 1) * exercisesPerIssue + exerciseIssue.exercisePosition = exerciseIssue.year) ∧
      (lastProblemNumber + (problemIssue.year - startYear) * problemsPerIssue * issuesPerYear + 
       (problemIssue.month - 1) * problemsPerIssue + problemIssue.problemPosition = problemIssue.year)

theorem magazine_publication_theorem :
  publicationSchedule 8 8 9 1967 1169 1576 :=
by
  sorry


end NUMINAMATH_CALUDE_magazine_publication_theorem_l808_80889


namespace NUMINAMATH_CALUDE_second_number_value_l808_80894

theorem second_number_value (x y z : ℚ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 4 / 7) :
  y = 240 / 7 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l808_80894


namespace NUMINAMATH_CALUDE_bracelets_given_to_school_is_three_l808_80837

/-- The number of bracelets Chantel gave away to her friends at school -/
def bracelets_given_to_school : ℕ :=
  let days_first_period := 5
  let bracelets_per_day_first_period := 2
  let days_second_period := 4
  let bracelets_per_day_second_period := 3
  let bracelets_given_at_soccer := 6
  let bracelets_remaining := 13
  let total_bracelets_made := days_first_period * bracelets_per_day_first_period + 
                              days_second_period * bracelets_per_day_second_period
  let bracelets_after_soccer := total_bracelets_made - bracelets_given_at_soccer
  bracelets_after_soccer - bracelets_remaining

theorem bracelets_given_to_school_is_three : 
  bracelets_given_to_school = 3 := by sorry

end NUMINAMATH_CALUDE_bracelets_given_to_school_is_three_l808_80837


namespace NUMINAMATH_CALUDE_f_even_and_decreasing_l808_80819

def f (x : ℝ) : ℝ := -x^2

theorem f_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_f_even_and_decreasing_l808_80819


namespace NUMINAMATH_CALUDE_annual_turbans_is_one_l808_80864

/-- Represents the salary structure and conditions of Gopi's servant --/
structure SalaryStructure where
  annual_cash : ℕ  -- Annual cash salary in Rs.
  months_worked : ℕ  -- Number of months the servant worked
  cash_received : ℕ  -- Cash received by the servant
  turbans_received : ℕ  -- Number of turbans received by the servant
  turban_price : ℕ  -- Price of one turban in Rs.

/-- Calculates the number of turbans given as part of the annual salary --/
def calculate_annual_turbans (s : SalaryStructure) : ℕ :=
  -- Implementation not provided, use 'sorry'
  sorry

/-- Theorem stating that the number of turbans given annually is 1 --/
theorem annual_turbans_is_one (s : SalaryStructure) 
  (h1 : s.annual_cash = 90)
  (h2 : s.months_worked = 9)
  (h3 : s.cash_received = 45)
  (h4 : s.turbans_received = 1)
  (h5 : s.turban_price = 90) : 
  calculate_annual_turbans s = 1 := by
  sorry

end NUMINAMATH_CALUDE_annual_turbans_is_one_l808_80864


namespace NUMINAMATH_CALUDE_min_value_theorem_l808_80846

-- Define the optimization problem
def optimization_problem (x y : ℝ) : Prop :=
  x - y ≥ 0 ∧ x + y - 2 ≥ 0 ∧ x ≤ 2

-- Define the objective function
def objective_function (x y : ℝ) : ℝ :=
  x^2 + y^2 - 2*x

-- Theorem statement
theorem min_value_theorem :
  ∃ (min_val : ℝ), min_val = -1/2 ∧
  (∀ (x y : ℝ), optimization_problem x y → objective_function x y ≥ min_val) ∧
  (∃ (x y : ℝ), optimization_problem x y ∧ objective_function x y = min_val) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l808_80846


namespace NUMINAMATH_CALUDE_smallest_value_expression_l808_80893

theorem smallest_value_expression (a b : ℤ) (h1 : a = 3 * b) (h2 : b ≠ 0) :
  (((a + b) / (a - b)) ^ 2 + ((a - b) / (a + b)) ^ 2 : ℝ) = 4.25 := by sorry

end NUMINAMATH_CALUDE_smallest_value_expression_l808_80893


namespace NUMINAMATH_CALUDE_cos_x_plus_3y_eq_one_l808_80815

/-- Given x and y in [-π/6, π/6] and a ∈ ℝ satisfying the system of equations,
    prove that cos(x + 3y) = 1 -/
theorem cos_x_plus_3y_eq_one 
  (x y : ℝ) 
  (hx : x ∈ Set.Icc (-π/6) (π/6))
  (hy : y ∈ Set.Icc (-π/6) (π/6))
  (a : ℝ)
  (eq1 : x^3 + Real.sin x - 3*a = 0)
  (eq2 : 9*y^3 + (1/3) * Real.sin (3*y) + a = 0) :
  Real.cos (x + 3*y) = 1 := by sorry

end NUMINAMATH_CALUDE_cos_x_plus_3y_eq_one_l808_80815


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l808_80858

theorem quadratic_is_square_of_binomial (a : ℚ) :
  (∃ r s : ℚ, ∀ x, a * x^2 - 25 * x + 9 = (r * x + s)^2) →
  a = 625 / 36 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l808_80858


namespace NUMINAMATH_CALUDE_discount_percentage_l808_80821

theorem discount_percentage
  (CP : ℝ) -- Cost Price
  (MP : ℝ) -- Marked Price
  (SP : ℝ) -- Selling Price
  (MP_condition : MP = CP * 1.5) -- Marked Price is 50% above Cost Price
  (SP_condition : SP = CP * 0.99) -- Selling Price results in 1% loss on Cost Price
  : (MP - SP) / MP * 100 = 34 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l808_80821


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_l808_80841

theorem triangle_angle_ratio (right_angle top_angle left_angle : ℝ) : 
  right_angle = 60 →
  top_angle = 70 →
  left_angle + right_angle + top_angle = 180 →
  left_angle / right_angle = 5 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_l808_80841


namespace NUMINAMATH_CALUDE_complementary_angles_l808_80809

theorem complementary_angles (A B : ℝ) : 
  A + B = 90 →  -- A and B are complementary
  A = 7 * B →   -- A is 7 times B
  A = 78.75 :=  -- A is 78.75°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_l808_80809


namespace NUMINAMATH_CALUDE_cubic_sum_simplification_l808_80800

theorem cubic_sum_simplification (a b : ℝ) : 
  a^2 = 9/25 → 
  b^2 = (3 + Real.sqrt 3)^2 / 15 → 
  a < 0 → 
  b > 0 → 
  (a + b)^3 = (-5670 * Real.sqrt 3 + 1620 * Real.sqrt 5 + 15 * Real.sqrt 15) / 50625 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_simplification_l808_80800


namespace NUMINAMATH_CALUDE_curve_through_center_l808_80817

-- Define the square
structure Square where
  center : ℝ × ℝ

-- Define the curve
structure Curve where
  -- A function that takes a real number parameter and returns a point on the curve
  pointAt : ℝ → ℝ × ℝ

-- Define the property that the curve divides the square into two equal areas
def divides_equally (s : Square) (γ : Curve) : Prop :=
  -- This is a placeholder for the actual condition
  sorry

-- Define the property that a line segment passes through a point
def passes_through (a b c : ℝ × ℝ) : Prop :=
  -- This is a placeholder for the actual condition
  sorry

-- The main theorem
theorem curve_through_center (s : Square) (γ : Curve) 
  (h : divides_equally s γ) : 
  ∃ (a b : ℝ × ℝ), (∃ (t₁ t₂ : ℝ), γ.pointAt t₁ = a ∧ γ.pointAt t₂ = b) ∧ 
    passes_through a b s.center := by
  sorry

end NUMINAMATH_CALUDE_curve_through_center_l808_80817


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l808_80806

/-- Given that the solution set of x^2 - px - q < 0 is {x | 2 < x < 3}, 
    prove the values of p and q, and the solution set of qx^2 - px - 1 > 0 -/
theorem quadratic_inequalities (p q : ℝ) : 
  (∀ x, x^2 - p*x - q < 0 ↔ 2 < x ∧ x < 3) →
  p = 5 ∧ q = -6 ∧ 
  (∀ x, q*x^2 - p*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l808_80806


namespace NUMINAMATH_CALUDE_min_value_range_l808_80827

def f (x : ℝ) := x^2 - 6*x + 8

theorem min_value_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, f x ≥ f a) →
  a ∈ Set.Ioo 1 3 ∪ {3} :=
by sorry

end NUMINAMATH_CALUDE_min_value_range_l808_80827


namespace NUMINAMATH_CALUDE_x_percent_of_x_squared_l808_80830

theorem x_percent_of_x_squared (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x^2 = 16) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_percent_of_x_squared_l808_80830


namespace NUMINAMATH_CALUDE_no_simultaneous_properties_l808_80884

theorem no_simultaneous_properties : ¬∃ (star : ℤ → ℤ → ℤ),
  (∀ Z : ℤ, ∃ X Y : ℤ, star X Y = Z) ∧
  (∀ A B : ℤ, star A B = -(star B A)) ∧
  (∀ A B C : ℤ, star (star A B) C = star A (star B C)) :=
by sorry

end NUMINAMATH_CALUDE_no_simultaneous_properties_l808_80884


namespace NUMINAMATH_CALUDE_function_property_l808_80860

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) + x = x * f y + f x)
  (h2 : f (-1) = 9) : 
  f (-500) = 1007 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l808_80860


namespace NUMINAMATH_CALUDE_decreasing_function_a_range_l808_80842

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 4*a*x + 2 else Real.log x / Real.log a

-- Define the property of f being decreasing on the entire real line
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- State the theorem
theorem decreasing_function_a_range (a : ℝ) :
  (is_decreasing (f a)) → (1/2 ≤ a ∧ a ≤ 3/4) :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_a_range_l808_80842


namespace NUMINAMATH_CALUDE_transform_f_eq_g_l808_80895

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- The transformation: shift 1 unit left, then 3 units up -/
def transform (g : ℝ → ℝ) : ℝ → ℝ := λ x => g (x + 1) + 3

/-- The expected result function -/
def g (x : ℝ) : ℝ := (x + 1)^2 + 1

/-- Theorem stating that the transformation of f equals g -/
theorem transform_f_eq_g : transform f = g := by sorry

end NUMINAMATH_CALUDE_transform_f_eq_g_l808_80895


namespace NUMINAMATH_CALUDE_f_max_min_implies_a_range_l808_80831

/-- The function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- Theorem: If f(x) = x^2 - 2x + 3 has a maximum of 3 and a minimum of 2 on [0, a], then a ∈ [1, 2] -/
theorem f_max_min_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 a, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 2) →
  a ∈ Set.Icc 1 2 := by
  sorry

#check f_max_min_implies_a_range

end NUMINAMATH_CALUDE_f_max_min_implies_a_range_l808_80831


namespace NUMINAMATH_CALUDE_specific_meal_cost_l808_80852

/-- Calculates the total amount spent on a meal including tip -/
def totalSpent (lunchCost drinkCost tipPercentage : ℚ) : ℚ :=
  let subtotal := lunchCost + drinkCost
  let tipAmount := (tipPercentage / 100) * subtotal
  subtotal + tipAmount

/-- Theorem: Given the specific costs and tip percentage, the total spent is $68.13 -/
theorem specific_meal_cost :
  totalSpent 50.20 4.30 25 = 68.13 := by sorry

end NUMINAMATH_CALUDE_specific_meal_cost_l808_80852


namespace NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l808_80881

-- Define the number of coin tosses
def num_tosses : ℕ := 8

-- Define the number of heads we're looking for
def target_heads : ℕ := 3

-- Define a function to calculate the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := sorry

-- Define a function to calculate the probability of getting exactly k heads in n tosses
def probability_exactly_k_heads (n k : ℕ) : ℚ :=
  (binomial_coefficient n k : ℚ) / (2 ^ n : ℚ)

-- Theorem statement
theorem probability_three_heads_in_eight_tosses :
  probability_exactly_k_heads num_tosses target_heads = 7 / 32 := by sorry

end NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l808_80881


namespace NUMINAMATH_CALUDE_jolene_babysitting_l808_80823

theorem jolene_babysitting (babysitting_rate : ℕ) (car_wash_rate : ℕ) (num_cars : ℕ) (total_raised : ℕ) :
  babysitting_rate = 30 →
  car_wash_rate = 12 →
  num_cars = 5 →
  total_raised = 180 →
  ∃ (num_families : ℕ), num_families * babysitting_rate + num_cars * car_wash_rate = total_raised ∧ num_families = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_jolene_babysitting_l808_80823


namespace NUMINAMATH_CALUDE_lcm_10_14_20_l808_80803

theorem lcm_10_14_20 : Nat.lcm 10 (Nat.lcm 14 20) = 140 := by sorry

end NUMINAMATH_CALUDE_lcm_10_14_20_l808_80803


namespace NUMINAMATH_CALUDE_rotation_90_degrees_l808_80836

def rotate90 (z : ℂ) : ℂ := z * Complex.I

theorem rotation_90_degrees :
  rotate90 (8 - 5 * Complex.I) = 5 + 8 * Complex.I := by sorry

end NUMINAMATH_CALUDE_rotation_90_degrees_l808_80836


namespace NUMINAMATH_CALUDE_newberg_airport_passengers_l808_80818

theorem newberg_airport_passengers (on_time late : ℕ) 
  (h1 : on_time = 14507) 
  (h2 : late = 213) : 
  on_time + late = 14720 := by sorry

end NUMINAMATH_CALUDE_newberg_airport_passengers_l808_80818


namespace NUMINAMATH_CALUDE_remaining_segments_length_l808_80807

/-- Represents the dimensions of the initial polygon --/
structure PolygonDimensions where
  vertical1 : ℝ
  horizontal1 : ℝ
  vertical2 : ℝ
  horizontal2 : ℝ
  vertical3 : ℝ
  horizontal3 : ℝ

/-- Calculates the total length of segments in the polygon --/
def totalLength (d : PolygonDimensions) : ℝ :=
  d.vertical1 + d.horizontal1 + d.vertical2 + d.horizontal2 + d.vertical3 + d.horizontal3

/-- Theorem: The length of remaining segments after removal is 21 units --/
theorem remaining_segments_length
  (d : PolygonDimensions)
  (h1 : d.vertical1 = 10)
  (h2 : d.horizontal1 = 5)
  (h3 : d.vertical2 = 4)
  (h4 : d.horizontal2 = 3)
  (h5 : d.vertical3 = 4)
  (h6 : d.horizontal3 = 2)
  (h7 : totalLength d = 28)
  (h8 : ∃ (removed : ℝ), removed = 7) :
  totalLength d - 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_remaining_segments_length_l808_80807


namespace NUMINAMATH_CALUDE_rosas_phone_calls_l808_80804

/-- Rosa's phone calls over two weeks -/
theorem rosas_phone_calls (last_week : ℝ) (this_week : ℝ) 
  (h1 : last_week = 10.2)
  (h2 : this_week = 8.6) :
  last_week + this_week = 18.8 := by
  sorry

end NUMINAMATH_CALUDE_rosas_phone_calls_l808_80804


namespace NUMINAMATH_CALUDE_count_squarish_numbers_l808_80838

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_squarish (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  is_perfect_square n ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0) ∧
  is_perfect_square (n / 100) ∧
  is_perfect_square (n % 100) ∧
  is_two_digit (n / 100) ∧
  is_two_digit (n % 100)

theorem count_squarish_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_squarish n) ∧ s.card = 2 := by sorry

end NUMINAMATH_CALUDE_count_squarish_numbers_l808_80838


namespace NUMINAMATH_CALUDE_max_points_at_least_sqrt2_max_points_greater_sqrt2_l808_80888

-- Define a point on a unit sphere
def PointOnUnitSphere := ℝ × ℝ × ℝ

-- Distance function between two points on a unit sphere
def sphereDistance (p q : PointOnUnitSphere) : ℝ := sorry

-- Theorem for part a
theorem max_points_at_least_sqrt2 :
  ∀ (n : ℕ) (points : Fin n → PointOnUnitSphere),
    (∀ (i j : Fin n), i ≠ j → sphereDistance (points i) (points j) ≥ Real.sqrt 2) →
    n ≤ 6 :=
sorry

-- Theorem for part b
theorem max_points_greater_sqrt2 :
  ∀ (n : ℕ) (points : Fin n → PointOnUnitSphere),
    (∀ (i j : Fin n), i ≠ j → sphereDistance (points i) (points j) > Real.sqrt 2) →
    n ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_points_at_least_sqrt2_max_points_greater_sqrt2_l808_80888


namespace NUMINAMATH_CALUDE_lucy_deposit_l808_80832

def initial_balance : ℕ := 65
def withdrawal : ℕ := 4
def final_balance : ℕ := 76

theorem lucy_deposit :
  ∃ (deposit : ℕ), initial_balance + deposit - withdrawal = final_balance :=
by
  sorry

end NUMINAMATH_CALUDE_lucy_deposit_l808_80832


namespace NUMINAMATH_CALUDE_no_real_solutions_quadratic_l808_80834

theorem no_real_solutions_quadratic (k : ℝ) :
  (∀ x : ℝ, x^2 - 4*x + k ≠ 0) ↔ k > 4 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_quadratic_l808_80834
