import Mathlib

namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l1261_126198

theorem triangle_cosine_theorem (a b c : ℝ) (h1 : b^2 = a*c) (h2 : c = 2*a) :
  let cos_C := (a^2 + b^2 - c^2) / (2*a*b)
  cos_C = -Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l1261_126198


namespace NUMINAMATH_CALUDE_sibling_ages_sum_l1261_126163

theorem sibling_ages_sum (a b : ℕ+) : 
  a < b → 
  a * b * b * b = 216 → 
  a + b + b + b = 19 := by
sorry

end NUMINAMATH_CALUDE_sibling_ages_sum_l1261_126163


namespace NUMINAMATH_CALUDE_soccer_players_count_l1261_126168

theorem soccer_players_count (total_socks : ℕ) (socks_per_player : ℕ) (h1 : total_socks = 22) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 11 := by
  sorry

end NUMINAMATH_CALUDE_soccer_players_count_l1261_126168


namespace NUMINAMATH_CALUDE_ratio_calculation_l1261_126194

theorem ratio_calculation (A B C : ℚ) (h : A/B = 3/2 ∧ B/C = 2/5) :
  (4*A + 3*B) / (5*C - 2*B) = 15/23 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l1261_126194


namespace NUMINAMATH_CALUDE_pipe_fill_time_l1261_126152

/-- Given pipes P, Q, and R that can fill a tank, this theorem proves the time it takes for pipe P to fill the tank. -/
theorem pipe_fill_time (fill_rate_Q : ℝ) (fill_rate_R : ℝ) (fill_rate_all : ℝ) 
  (hQ : fill_rate_Q = 1 / 9)
  (hR : fill_rate_R = 1 / 18)
  (hAll : fill_rate_all = 1 / 2)
  (h_sum : ∃ (fill_rate_P : ℝ), fill_rate_P + fill_rate_Q + fill_rate_R = fill_rate_all) :
  ∃ (fill_time_P : ℝ), fill_time_P = 3 := by
  sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l1261_126152


namespace NUMINAMATH_CALUDE_magnitude_squared_of_complex_l1261_126165

theorem magnitude_squared_of_complex (z : ℂ) : z = 3 + 4*I → Complex.abs z ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_squared_of_complex_l1261_126165


namespace NUMINAMATH_CALUDE_area_of_triangle_range_of_sum_a_c_l1261_126143

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 * Real.sqrt 3 ∧
  Real.sqrt 3 * Real.cos t.B = t.b * Real.sin t.C

-- Theorem 1: Area of triangle ABC
theorem area_of_triangle (t : Triangle) (h : triangle_conditions t) (ha : t.a = 2) :
  (1/2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3 :=
sorry

-- Theorem 2: Range of a + c
theorem range_of_sum_a_c (t : Triangle) (h : triangle_conditions t) (acute : t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧ t.A + t.B + t.C = π) :
  2 * Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_range_of_sum_a_c_l1261_126143


namespace NUMINAMATH_CALUDE_largest_sum_is_three_fourths_l1261_126112

theorem largest_sum_is_three_fourths : 
  let sums : List ℚ := [1/4 + 1/2, 1/4 + 1/3, 1/4 + 1/9, 1/4 + 1/10, 1/4 + 1/11]
  (∀ x ∈ sums, x ≤ 1/4 + 1/2) ∧ (1/4 + 1/2 = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_is_three_fourths_l1261_126112


namespace NUMINAMATH_CALUDE_modulus_of_complex_reciprocal_l1261_126188

theorem modulus_of_complex_reciprocal (z : ℂ) : 
  z = (Complex.I - 1)⁻¹ → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_reciprocal_l1261_126188


namespace NUMINAMATH_CALUDE_hemisphere_volume_l1261_126186

/-- The volume of a hemisphere with radius 21.002817118114375 cm is 96993.17249452507 cubic centimeters. -/
theorem hemisphere_volume : 
  let r : Real := 21.002817118114375
  let V : Real := (2/3) * Real.pi * r^3
  V = 96993.17249452507 := by sorry

end NUMINAMATH_CALUDE_hemisphere_volume_l1261_126186


namespace NUMINAMATH_CALUDE_office_employees_l1261_126137

/-- The total number of employees in an office -/
def total_employees : ℕ := 1300

/-- The proportion of female employees -/
def female_ratio : ℚ := 3/5

/-- The proportion of computer literate male employees among all male employees -/
def literate_male_ratio : ℚ := 1/2

/-- The proportion of computer literate employees among all employees -/
def literate_total_ratio : ℚ := 31/50

/-- The number of computer literate female employees -/
def literate_female_count : ℕ := 546

theorem office_employees :
  total_employees = 1300 ∧
  (female_ratio : ℚ) * total_employees = 3/5 * 1300 ∧
  literate_male_ratio * ((1 - female_ratio) * total_employees) = 1/2 * (2/5 * 1300) ∧
  literate_total_ratio * total_employees = 31/50 * 1300 ∧
  literate_female_count = 546 ∧
  literate_female_count + literate_male_ratio * ((1 - female_ratio) * total_employees) =
    literate_total_ratio * total_employees :=
by sorry

#check office_employees

end NUMINAMATH_CALUDE_office_employees_l1261_126137


namespace NUMINAMATH_CALUDE_system_to_quadratic_l1261_126190

theorem system_to_quadratic (x y : ℝ) 
  (eq1 : 3 * x^2 + 9 * x + 4 * y + 2 = 0)
  (eq2 : 3 * x + y + 4 = 0) :
  y^2 + 11 * y - 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_system_to_quadratic_l1261_126190


namespace NUMINAMATH_CALUDE_regular_quad_pyramid_angle_relation_l1261_126161

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  /-- Dihedral angle between a lateral face and the base -/
  α : ℝ
  /-- Dihedral angle between two adjacent lateral faces -/
  β : ℝ

/-- Theorem: In a regular quadrilateral pyramid, 2cosβ + cos2α = -1 -/
theorem regular_quad_pyramid_angle_relation (p : RegularQuadPyramid) :
  2 * Real.cos p.β + Real.cos (2 * p.α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_regular_quad_pyramid_angle_relation_l1261_126161


namespace NUMINAMATH_CALUDE_files_deleted_amy_deleted_files_l1261_126144

theorem files_deleted (initial_music : ℕ) (initial_video : ℕ) (remaining : ℕ) : ℕ :=
  (initial_music + initial_video) - remaining

theorem amy_deleted_files : files_deleted 4 21 2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_files_deleted_amy_deleted_files_l1261_126144


namespace NUMINAMATH_CALUDE_amount_with_r_l1261_126131

theorem amount_with_r (total : ℝ) (p q r : ℝ) : 
  total = 4000 →
  p + q + r = total →
  r = (2/3) * (p + q) →
  r = 1600 := by
sorry

end NUMINAMATH_CALUDE_amount_with_r_l1261_126131


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l1261_126158

theorem fraction_product_simplification (a b c : ℝ) 
  (ha : a ≠ 4) (hb : b ≠ 5) (hc : c ≠ 6) : 
  (a - 4) / (6 - c) * (b - 5) / (4 - a) * (c - 6) / (5 - b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l1261_126158


namespace NUMINAMATH_CALUDE_solution_value_l1261_126151

theorem solution_value (a : ℝ) (h : a^2 - 2*a - 1 = 0) : a^2 - 2*a + 2022 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1261_126151


namespace NUMINAMATH_CALUDE_simplify_expression_l1261_126162

theorem simplify_expression (t : ℝ) (h : t ≠ 0) : (t^5 * t^7) / t^3 = t^9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1261_126162


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l1261_126176

/-- Represents a trapezoid with diagonals and sum of bases -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  sum_of_bases : ℝ

/-- Calculates the area of a trapezoid given its diagonals and sum of bases -/
def area (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that a trapezoid with diagonals 12 and 6, and sum of bases 14, has an area of 16√5 -/
theorem trapezoid_area_theorem :
  let t : Trapezoid := { diagonal1 := 12, diagonal2 := 6, sum_of_bases := 14 }
  area t = 16 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_theorem_l1261_126176


namespace NUMINAMATH_CALUDE_hyperbola_from_ellipse_and_asymptote_l1261_126156

/-- Given ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

/-- Given asymptote equation -/
def asymptote_equation (x y : ℝ) : Prop := x - Real.sqrt 2 * y = 0

/-- Hyperbola equation to be proved -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / (8 * Real.sqrt 3 / 3) - y^2 / (4 * Real.sqrt 3 / 3) = 1

theorem hyperbola_from_ellipse_and_asymptote :
  ∀ x y : ℝ,
  (∃ a b : ℝ, ellipse_equation a b ∧
    (∀ c d : ℝ, hyperbola_equation c d → (c - a)^2 + (d - b)^2 = (c + a)^2 + (d + b)^2)) →
  asymptote_equation x y →
  hyperbola_equation x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_from_ellipse_and_asymptote_l1261_126156


namespace NUMINAMATH_CALUDE_cos_alpha_plus_beta_l1261_126196

theorem cos_alpha_plus_beta (α β : Real) 
  (h1 : Real.sin (3 * Real.pi / 4 + α) = 5 / 13)
  (h2 : Real.cos (Real.pi / 4 - β) = 3 / 5)
  (h3 : 0 < α) (h4 : α < Real.pi / 4) (h5 : Real.pi / 4 < β) (h6 : β < 3 * Real.pi / 4) :
  Real.cos (α + β) = -33 / 65 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_beta_l1261_126196


namespace NUMINAMATH_CALUDE_chairs_per_rectangular_table_l1261_126184

theorem chairs_per_rectangular_table :
  let round_tables : ℕ := 2
  let rectangular_tables : ℕ := 2
  let chairs_per_round_table : ℕ := 6
  let total_chairs : ℕ := 26
  (total_chairs - round_tables * chairs_per_round_table) / rectangular_tables = 7 := by
sorry

end NUMINAMATH_CALUDE_chairs_per_rectangular_table_l1261_126184


namespace NUMINAMATH_CALUDE_jellybeans_left_in_jar_l1261_126113

theorem jellybeans_left_in_jar
  (total_jellybeans : ℕ)
  (total_kids : ℕ)
  (absent_kids : ℕ)
  (jellybeans_per_kid : ℕ)
  (h1 : total_jellybeans = 100)
  (h2 : total_kids = 24)
  (h3 : absent_kids = 2)
  (h4 : jellybeans_per_kid = 3) :
  total_jellybeans - (total_kids - absent_kids) * jellybeans_per_kid = 34 :=
by
  sorry


end NUMINAMATH_CALUDE_jellybeans_left_in_jar_l1261_126113


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l1261_126130

theorem quadratic_negative_root (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ m * x^2 + 2 * x + 1 = 0) ↔ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l1261_126130


namespace NUMINAMATH_CALUDE_equation_solution_range_l1261_126174

theorem equation_solution_range (a : ℝ) : 
  (∃ x : ℝ, 9^x + (a+4)*3^x + 4 = 0) ↔ a ≤ -8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l1261_126174


namespace NUMINAMATH_CALUDE_sandys_age_l1261_126155

/-- Given that Molly is 16 years older than Sandy and their ages are in the ratio 7:9, prove that Sandy is 56 years old. -/
theorem sandys_age (sandy_age molly_age : ℕ) : 
  molly_age = sandy_age + 16 →
  sandy_age * 9 = molly_age * 7 →
  sandy_age = 56 := by
  sorry

end NUMINAMATH_CALUDE_sandys_age_l1261_126155


namespace NUMINAMATH_CALUDE_smallest_prime_pair_l1261_126171

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem smallest_prime_pair : 
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ q = 13 * p + 2 ∧ 
  (∀ (p' : ℕ), is_prime p' ∧ p' < p → ¬(is_prime (13 * p' + 2))) ∧
  p = 3 ∧ q = 41 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_pair_l1261_126171


namespace NUMINAMATH_CALUDE_D_72_l1261_126115

/-- D(n) represents the number of ways of expressing the positive integer n 
    as a product of integers greater than 1, where the order matters. -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(72) is equal to 103 -/
theorem D_72 : D 72 = 103 := by sorry

end NUMINAMATH_CALUDE_D_72_l1261_126115


namespace NUMINAMATH_CALUDE_intersection_M_N_l1261_126103

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2 * x ∧ x > 0}
def N : Set ℝ := {x | ∃ y, y = Real.log (2 * x - x^2) ∧ x > 0 ∧ x < 2}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1261_126103


namespace NUMINAMATH_CALUDE_parking_lot_car_ratio_l1261_126177

theorem parking_lot_car_ratio :
  let red_cars : ℕ := 28
  let black_cars : ℕ := 75
  (red_cars : ℚ) / black_cars = 28 / 75 :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_car_ratio_l1261_126177


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1261_126182

/-- The common ratio of a geometric sequence starting with 10, -20, 40, -80 is -2 -/
theorem geometric_sequence_ratio : ∀ (a : ℕ → ℤ), 
  a 0 = 10 ∧ a 1 = -20 ∧ a 2 = 40 ∧ a 3 = -80 → 
  (∀ n : ℕ, a (n + 1) = a n * (-2)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1261_126182


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1261_126121

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1261_126121


namespace NUMINAMATH_CALUDE_evaluate_expression_l1261_126146

theorem evaluate_expression (x : ℕ) (h : x = 3) : x^2 + x * (x^(Nat.factorial x)) = 2196 :=
by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1261_126146


namespace NUMINAMATH_CALUDE_mark_chicken_nuggets_cost_l1261_126154

-- Define the number of chicken nuggets Mark orders
def total_nuggets : ℕ := 100

-- Define the number of nuggets in a box
def nuggets_per_box : ℕ := 20

-- Define the cost of one box
def cost_per_box : ℕ := 4

-- Theorem to prove
theorem mark_chicken_nuggets_cost :
  (total_nuggets / nuggets_per_box) * cost_per_box = 20 := by
  sorry

end NUMINAMATH_CALUDE_mark_chicken_nuggets_cost_l1261_126154


namespace NUMINAMATH_CALUDE_triangle_right_angled_l1261_126106

theorem triangle_right_angled (A B C : ℝ) (h : A - C = B) : A = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l1261_126106


namespace NUMINAMATH_CALUDE_coursework_materials_theorem_l1261_126107

def total_budget : ℝ := 1000

def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

def coursework_materials_spending : ℝ := 
  total_budget * (1 - (food_percentage + accommodation_percentage + entertainment_percentage))

theorem coursework_materials_theorem : 
  coursework_materials_spending = 300 := by sorry

end NUMINAMATH_CALUDE_coursework_materials_theorem_l1261_126107


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l1261_126149

-- Define a function to convert a list of binary digits to a decimal number
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define our specific binary number
def our_binary : List Bool := [true, true, false, false, true, true]

-- State the theorem
theorem binary_110011_equals_51 :
  binary_to_decimal our_binary = 51 := by sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l1261_126149


namespace NUMINAMATH_CALUDE_perpendicular_nonzero_vectors_exist_l1261_126145

theorem perpendicular_nonzero_vectors_exist :
  ∃ (a b : ℝ × ℝ), a ≠ (0, 0) ∧ b ≠ (0, 0) ∧ a.1 * b.1 + a.2 * b.2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_nonzero_vectors_exist_l1261_126145


namespace NUMINAMATH_CALUDE_model4_best_fitting_l1261_126108

-- Define the structure for a regression model
structure RegressionModel where
  name : String
  r_squared : Float

-- Define the principle of better fitting
def better_fitting (m1 m2 : RegressionModel) : Prop :=
  m1.r_squared > m2.r_squared

-- Define the four models
def model1 : RegressionModel := ⟨"Model 1", 0.55⟩
def model2 : RegressionModel := ⟨"Model 2", 0.65⟩
def model3 : RegressionModel := ⟨"Model 3", 0.79⟩
def model4 : RegressionModel := ⟨"Model 4", 0.95⟩

-- Define a list of all models
def all_models : List RegressionModel := [model1, model2, model3, model4]

-- Theorem: Model 4 has the best fitting effect
theorem model4_best_fitting :
  ∀ m ∈ all_models, m ≠ model4 → better_fitting model4 m :=
by sorry

end NUMINAMATH_CALUDE_model4_best_fitting_l1261_126108


namespace NUMINAMATH_CALUDE_boat_current_rate_l1261_126101

/-- Proves that given a boat with a speed of 20 km/hr in still water,
    traveling 9.6 km downstream in 24 minutes, the rate of the current is 4 km/hr. -/
theorem boat_current_rate (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 20 →
  downstream_distance = 9.6 →
  downstream_time = 24 / 60 →
  ∃ (current_rate : ℝ),
    current_rate = 4 ∧
    downstream_distance = (boat_speed + current_rate) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_current_rate_l1261_126101


namespace NUMINAMATH_CALUDE_acid_mixture_percentage_l1261_126180

theorem acid_mixture_percentage : ∀ (a w : ℝ),
  a + w = 6 →
  a / (a + w + 2) = 15 / 100 →
  (a + 2) / (a + w + 4) = 25 / 100 →
  a / (a + w) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_acid_mixture_percentage_l1261_126180


namespace NUMINAMATH_CALUDE_integral_sqrt_plus_linear_l1261_126132

theorem integral_sqrt_plus_linear (f g : ℝ → ℝ) :
  (∫ x in (0 : ℝ)..1, (Real.sqrt (1 - x^2) + 3*x)) = π/4 + 3/2 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_plus_linear_l1261_126132


namespace NUMINAMATH_CALUDE_motorcycle_sales_decrease_l1261_126102

/-- Represents the pricing and sales of motorcycles before and after a price increase --/
structure MotorcycleSales where
  original_price : ℝ
  new_price : ℝ
  original_quantity : ℕ
  new_quantity : ℕ
  original_revenue : ℝ
  new_revenue : ℝ

/-- The theorem stating the decrease in motorcycle sales after the price increase --/
theorem motorcycle_sales_decrease (sales : MotorcycleSales) : 
  sales.new_price = sales.original_price + 1000 →
  sales.new_revenue = sales.original_revenue + 26000 →
  sales.new_revenue = 594000 →
  sales.new_quantity = 63 →
  sales.original_quantity - sales.new_quantity = 4 := by
  sorry

#check motorcycle_sales_decrease

end NUMINAMATH_CALUDE_motorcycle_sales_decrease_l1261_126102


namespace NUMINAMATH_CALUDE_root_sum_powers_l1261_126175

theorem root_sum_powers (α β : ℝ) : 
  α^2 - 5*α + 6 = 0 → β^2 - 5*β + 6 = 0 → 3*α^3 + 10*β^4 = 2305 := by
sorry

end NUMINAMATH_CALUDE_root_sum_powers_l1261_126175


namespace NUMINAMATH_CALUDE_remaining_fabric_is_294_l1261_126159

/-- Represents the flag-making scenario with given dimensions and quantities --/
structure FlagScenario where
  total_fabric : ℕ
  square_side : ℕ
  wide_length : ℕ
  wide_width : ℕ
  tall_length : ℕ
  tall_width : ℕ
  square_count : ℕ
  wide_count : ℕ
  tall_count : ℕ

/-- Calculates the remaining fabric after making flags --/
def remaining_fabric (scenario : FlagScenario) : ℕ :=
  scenario.total_fabric -
  (scenario.square_count * scenario.square_side * scenario.square_side +
   scenario.wide_count * scenario.wide_length * scenario.wide_width +
   scenario.tall_count * scenario.tall_length * scenario.tall_width)

/-- Theorem stating that the remaining fabric in the given scenario is 294 square feet --/
theorem remaining_fabric_is_294 (scenario : FlagScenario)
  (h1 : scenario.total_fabric = 1000)
  (h2 : scenario.square_side = 4)
  (h3 : scenario.wide_length = 5)
  (h4 : scenario.wide_width = 3)
  (h5 : scenario.tall_length = 3)
  (h6 : scenario.tall_width = 5)
  (h7 : scenario.square_count = 16)
  (h8 : scenario.wide_count = 20)
  (h9 : scenario.tall_count = 10) :
  remaining_fabric scenario = 294 := by
  sorry


end NUMINAMATH_CALUDE_remaining_fabric_is_294_l1261_126159


namespace NUMINAMATH_CALUDE_class_gender_ratio_l1261_126170

theorem class_gender_ratio (female_count : ℕ) (male_count : ℕ) 
  (h1 : female_count = 28)
  (h2 : female_count = male_count + 6) :
  let total_count := female_count + male_count
  (female_count : ℚ) / (male_count : ℚ) = 14 / 11 ∧ 
  (male_count : ℚ) / (total_count : ℚ) = 11 / 25 := by
sorry

end NUMINAMATH_CALUDE_class_gender_ratio_l1261_126170


namespace NUMINAMATH_CALUDE_inequality_and_factorial_l1261_126116

theorem inequality_and_factorial (n : ℕ) : 2 ≤ (1 + 1 / n : ℝ) ^ n ∧ (1 + 1 / n : ℝ) ^ n < 3 ∧ (n / 3 : ℝ) ^ n < n! := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_factorial_l1261_126116


namespace NUMINAMATH_CALUDE_sum_equals_350_l1261_126135

theorem sum_equals_350 : 124 + 129 + 106 + 141 + 237 - 500 + 113 = 350 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_350_l1261_126135


namespace NUMINAMATH_CALUDE_subtract_squared_terms_l1261_126119

theorem subtract_squared_terms (x : ℝ) : 3 * x^2 - 4 * x^2 = -x^2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_squared_terms_l1261_126119


namespace NUMINAMATH_CALUDE_journey_remaining_distance_l1261_126126

/-- Given a total journey distance and the distance already driven, 
    calculate the remaining distance to be driven. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Theorem: For a journey of 1200 miles where 768 miles have been driven,
    the remaining distance is 432 miles. -/
theorem journey_remaining_distance :
  remaining_distance 1200 768 = 432 := by
  sorry

end NUMINAMATH_CALUDE_journey_remaining_distance_l1261_126126


namespace NUMINAMATH_CALUDE_cyclist_round_trip_l1261_126105

/-- Cyclist's round trip problem -/
theorem cyclist_round_trip
  (total_distance : ℝ)
  (first_leg_distance : ℝ)
  (second_leg_distance : ℝ)
  (first_leg_speed : ℝ)
  (second_leg_speed : ℝ)
  (total_round_trip_time : ℝ)
  (h1 : total_distance = first_leg_distance + second_leg_distance)
  (h2 : first_leg_distance = 18)
  (h3 : second_leg_distance = 12)
  (h4 : first_leg_speed = 9)
  (h5 : second_leg_speed = 10)
  (h6 : total_round_trip_time = 7.2)
  : (2 * total_distance) / (total_round_trip_time - (first_leg_distance / first_leg_speed + second_leg_distance / second_leg_speed)) = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_cyclist_round_trip_l1261_126105


namespace NUMINAMATH_CALUDE_workshop_output_comparison_l1261_126195

/-- Represents the monthly increase factor for a workshop -/
structure WorkshopGrowth where
  fixed_amount : ℝ
  percentage : ℝ

/-- Theorem statement for workshop output comparison -/
theorem workshop_output_comparison 
  (growth_A growth_B : WorkshopGrowth)
  (h_initial_equal : growth_A.fixed_amount = growth_B.percentage) -- Initial outputs are equal
  (h_equal_after_7 : 1 + 6 * growth_A.fixed_amount = (1 + growth_B.percentage) ^ 6) -- Equal after 7 months
  : 1 + 3 * growth_A.fixed_amount > (1 + growth_B.percentage) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_workshop_output_comparison_l1261_126195


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1261_126191

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : Real.log (x + y) = 0) :
  (1 / x + 1 / y) ≥ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Real.log (a + b) = 0 ∧ 1 / a + 1 / b = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1261_126191


namespace NUMINAMATH_CALUDE_average_song_length_l1261_126111

-- Define the given conditions
def hours_per_month : ℝ := 20
def cost_per_song : ℝ := 0.5
def yearly_cost : ℝ := 2400
def months_per_year : ℕ := 12
def minutes_per_hour : ℕ := 60

-- Define the theorem
theorem average_song_length :
  let songs_per_year : ℝ := yearly_cost / cost_per_song
  let songs_per_month : ℝ := songs_per_year / months_per_year
  let total_minutes_per_month : ℝ := hours_per_month * minutes_per_hour
  total_minutes_per_month / songs_per_month = 3 := by
  sorry


end NUMINAMATH_CALUDE_average_song_length_l1261_126111


namespace NUMINAMATH_CALUDE_range_of_a_l1261_126134

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → x^3 - a*x + 1 ≥ 0) → 
  0 ≤ a ∧ a ≤ 3 * (2 : ℝ)^(1/3) / 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1261_126134


namespace NUMINAMATH_CALUDE_hamburgers_left_over_correct_verify_solution_l1261_126199

/-- Calculates the number of hamburgers left over after lunch service -/
def hamburgers_left_over (initial : ℕ) (served_hour1 : ℕ) (served_hour2 : ℕ) : ℕ :=
  initial - served_hour1 - served_hour2

/-- Theorem stating that the number of hamburgers left over is correct -/
theorem hamburgers_left_over_correct (initial : ℕ) (served_hour1 : ℕ) (served_hour2 : ℕ) :
  hamburgers_left_over initial served_hour1 served_hour2 = initial - (served_hour1 + served_hour2) :=
by sorry

/-- Verifies the solution for the given problem -/
theorem verify_solution :
  hamburgers_left_over 25 12 6 = 7 :=
by sorry

end NUMINAMATH_CALUDE_hamburgers_left_over_correct_verify_solution_l1261_126199


namespace NUMINAMATH_CALUDE_fraction_operation_equivalence_l1261_126164

theorem fraction_operation_equivalence (x : ℚ) :
  x * (5/6) / (2/7) = x * (35/12) := by
sorry

end NUMINAMATH_CALUDE_fraction_operation_equivalence_l1261_126164


namespace NUMINAMATH_CALUDE_second_set_amount_l1261_126167

def total_spent : ℝ := 900
def first_set : ℝ := 325
def last_set : ℝ := 315

theorem second_set_amount :
  total_spent - first_set - last_set = 260 := by sorry

end NUMINAMATH_CALUDE_second_set_amount_l1261_126167


namespace NUMINAMATH_CALUDE_derivative_f_at_3_l1261_126142

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_f_at_3 : 
  deriv f 3 = 6 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_3_l1261_126142


namespace NUMINAMATH_CALUDE_arithmetic_mean_greater_than_geometric_mean_l1261_126187

theorem arithmetic_mean_greater_than_geometric_mean (x y : ℝ) (hx : x = 16) (hy : y = 64) :
  (x + y) / 2 > Real.sqrt (x * y) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_greater_than_geometric_mean_l1261_126187


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1261_126148

/-- The equation of the tangent line to the curve y = x^2 + 1/x at the point (1, 2) is x - y + 1 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = x^2 + 1/x) → -- Curve equation
  (x = 1 ∧ y = 2) → -- Point on the curve
  (x - y + 1 = 0) -- Equation of the tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1261_126148


namespace NUMINAMATH_CALUDE_hill_distance_l1261_126100

theorem hill_distance (speed_up speed_down : ℝ) (total_time : ℝ) 
  (h1 : speed_up = 1.5)
  (h2 : speed_down = 4.5)
  (h3 : total_time = 6) :
  ∃ d : ℝ, d = 6.75 ∧ d / speed_up + d / speed_down = total_time :=
sorry

end NUMINAMATH_CALUDE_hill_distance_l1261_126100


namespace NUMINAMATH_CALUDE_area_segment_proportions_l1261_126179

/-- Given areas and segments, prove proportional relationships -/
theorem area_segment_proportions 
  (S S'' S' : ℝ) 
  (a a' : ℝ) 
  (h : S / S'' = a / a') 
  (h_pos : S > 0 ∧ S'' > 0 ∧ S' > 0 ∧ a > 0 ∧ a' > 0) :
  (S / a = S' / a') ∧ (S * a' = S' * a) := by
  sorry

end NUMINAMATH_CALUDE_area_segment_proportions_l1261_126179


namespace NUMINAMATH_CALUDE_calculation_proof_l1261_126110

theorem calculation_proof : (8.036 / 0.04) * (1.5 / 0.03) = 10045 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1261_126110


namespace NUMINAMATH_CALUDE_cube_root_of_2197_l1261_126173

theorem cube_root_of_2197 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 2197) : x = 13 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_2197_l1261_126173


namespace NUMINAMATH_CALUDE_parallel_condition_distance_condition_l1261_126127

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given point P with coordinates (a+2, 2a-8) -/
def P (a : ℝ) : Point := ⟨a + 2, 2 * a - 8⟩

/-- Point Q with fixed coordinates (1, -2) -/
def Q : Point := ⟨1, -2⟩

/-- Condition 1: Line PQ is parallel to x-axis -/
def parallel_to_x_axis (P Q : Point) : Prop := P.y = Q.y

/-- Condition 2: Distance from P to y-axis is 4 -/
def distance_to_y_axis (P : Point) : ℝ := |P.x|

/-- Theorem for Condition 1 -/
theorem parallel_condition (a : ℝ) : 
  parallel_to_x_axis (P a) Q → P a = ⟨5, -2⟩ := by sorry

/-- Theorem for Condition 2 -/
theorem distance_condition (a : ℝ) : 
  distance_to_y_axis (P a) = 4 → (P a = ⟨4, -4⟩ ∨ P a = ⟨-4, -20⟩) := by sorry

end NUMINAMATH_CALUDE_parallel_condition_distance_condition_l1261_126127


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1261_126123

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4) / (1 - r)) → r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1261_126123


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1261_126109

theorem unique_quadratic_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, b * x^2 + 16 * x + 12 = 0) : 
  ∃ x, b * x^2 + 16 * x + 12 = 0 ∧ x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1261_126109


namespace NUMINAMATH_CALUDE_gcd_10293_29384_l1261_126122

theorem gcd_10293_29384 : Nat.gcd 10293 29384 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10293_29384_l1261_126122


namespace NUMINAMATH_CALUDE_det_2x2_matrix_l1261_126117

theorem det_2x2_matrix : 
  Matrix.det !![4, 3; 2, 1] = -2 := by
  sorry

end NUMINAMATH_CALUDE_det_2x2_matrix_l1261_126117


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l1261_126125

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total duration of the traffic light cycle -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the total duration of intervals where a color change can be observed -/
def changeWindowDuration (viewingTime : ℕ) : ℕ :=
  3 * viewingTime

/-- Theorem: The probability of observing a color change in a 4-second interval is 2/15 -/
theorem traffic_light_change_probability 
  (cycle : TrafficLightCycle) 
  (h1 : cycle.green = 40)
  (h2 : cycle.yellow = 5)
  (h3 : cycle.red = 45)
  (viewingTime : ℕ)
  (h4 : viewingTime = 4) :
  (changeWindowDuration viewingTime : ℚ) / (cycleDuration cycle : ℚ) = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_change_probability_l1261_126125


namespace NUMINAMATH_CALUDE_constant_ratio_locus_l1261_126128

/-- The locus of points with a constant ratio of distances -/
theorem constant_ratio_locus (x y : ℝ) :
  (((x - 4)^2 + y^2) / (x - 3)^2 = 4) →
  (3 * x^2 - y^2 - 16 * x + 20 = 0) :=
by sorry

end NUMINAMATH_CALUDE_constant_ratio_locus_l1261_126128


namespace NUMINAMATH_CALUDE_total_books_count_l1261_126138

-- Define the number of books per shelf
def booksPerShelf : ℕ := 6

-- Define the number of shelves for each category
def mysteryShelvesCount : ℕ := 8
def pictureShelvesCount : ℕ := 5
def sciFiShelvesCount : ℕ := 4
def nonFictionShelvesCount : ℕ := 3

-- Define the total number of books
def totalBooks : ℕ := 
  booksPerShelf * (mysteryShelvesCount + pictureShelvesCount + sciFiShelvesCount + nonFictionShelvesCount)

-- Theorem statement
theorem total_books_count : totalBooks = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_books_count_l1261_126138


namespace NUMINAMATH_CALUDE_medal_distribution_proof_l1261_126178

def distribute_medals (n : ℕ) : ℕ :=
  Nat.choose (n + 2) 2

theorem medal_distribution_proof (n : ℕ) (h : n = 12) : 
  distribute_medals n = 55 := by
  sorry

end NUMINAMATH_CALUDE_medal_distribution_proof_l1261_126178


namespace NUMINAMATH_CALUDE_min_values_theorem_l1261_126139

theorem min_values_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b + 3 = a * b) :
  (∀ x y, x > 0 → y > 0 → x + y + 3 = x * y → a + b ≤ x + y) ∧
  (∀ x y, x > 0 → y > 0 → x + y + 3 = x * y → a^2 + b^2 ≤ x^2 + y^2) ∧
  (∀ x y, x > 0 → y > 0 → x + y + 3 = x * y → 1/a + 1/b ≤ 1/x + 1/y) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x + y + 3 = x * y ∧ a + b = x + y ∧ a^2 + b^2 = x^2 + y^2 ∧ 1/a + 1/b = 1/x + 1/y) :=
by
  sorry

#check min_values_theorem

end NUMINAMATH_CALUDE_min_values_theorem_l1261_126139


namespace NUMINAMATH_CALUDE_computer_price_increase_l1261_126185

theorem computer_price_increase (x : ℝ) (h : 2 * x = 540) : 
  (351 - x) / x * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1261_126185


namespace NUMINAMATH_CALUDE_elevator_height_after_20_seconds_l1261_126183

/-- Calculates the height of a descending elevator after a given time. -/
def elevatorHeight (initialHeight : ℝ) (descentSpeed : ℝ) (time : ℝ) : ℝ :=
  initialHeight - descentSpeed * time

/-- Theorem: An elevator starting at 120 meters above ground and descending
    at 4 meters per second will be at 40 meters after 20 seconds. -/
theorem elevator_height_after_20_seconds :
  elevatorHeight 120 4 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_elevator_height_after_20_seconds_l1261_126183


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l1261_126197

theorem cara_seating_arrangements (n : ℕ) (k : ℕ) : n = 8 ∧ k = 2 → Nat.choose n k = 28 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l1261_126197


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1261_126141

theorem inequality_equivalence (x : Real) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (2 * Real.cos x ≤ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ∧
   Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ≤ Real.sqrt 2) ↔
  x ∈ Set.Icc (Real.pi / 4) (7 * Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1261_126141


namespace NUMINAMATH_CALUDE_shoe_color_probability_l1261_126189

theorem shoe_color_probability (n : ℕ) (k : ℕ) (p : ℕ) :
  n = 2 * p →
  k = 3 →
  p = 6 →
  (1 - (n * (n - 2) * (n - 4) / (k * (k - 1) * (k - 2))) / (n.choose k)) = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_shoe_color_probability_l1261_126189


namespace NUMINAMATH_CALUDE_fishes_per_body_of_water_l1261_126136

theorem fishes_per_body_of_water 
  (total_bodies : ℕ) 
  (total_fishes : ℕ) 
  (h1 : total_bodies = 6) 
  (h2 : total_fishes = 1050) 
  (h3 : total_fishes % total_bodies = 0) -- Ensuring equal distribution
  : total_fishes / total_bodies = 175 := by
  sorry

end NUMINAMATH_CALUDE_fishes_per_body_of_water_l1261_126136


namespace NUMINAMATH_CALUDE_vertical_asymptote_at_five_l1261_126166

/-- The function f(x) = (x^2 - 3x + 10) / (x - 5) has a vertical asymptote at x = 5 -/
theorem vertical_asymptote_at_five :
  let f : ℝ → ℝ := λ x => (x^2 - 3*x + 10) / (x - 5)
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), 0 < δ ∧ δ < ε →
    (∀ x : ℝ, 0 < |x - 5| ∧ |x - 5| < δ → |f x| > 1/δ) :=
by sorry

end NUMINAMATH_CALUDE_vertical_asymptote_at_five_l1261_126166


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_plus_2_l1261_126140

theorem floor_sqrt_50_squared_plus_2 : ⌊Real.sqrt 50⌋^2 + 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_plus_2_l1261_126140


namespace NUMINAMATH_CALUDE_root_equation_solution_l1261_126104

theorem root_equation_solution (y : ℝ) : 
  (y * (y^5)^(1/3))^(1/7) = 4 → y = 2^(21/4) := by
sorry

end NUMINAMATH_CALUDE_root_equation_solution_l1261_126104


namespace NUMINAMATH_CALUDE_polynomial_proofs_l1261_126150

theorem polynomial_proofs (x : ℝ) : 
  (x^2 + 2*x - 3 = (x + 3)*(x - 1)) ∧ 
  (x^2 + 8*x + 7 = (x + 7)*(x + 1)) ∧ 
  (-x^2 + 2/3*x + 1 < 4/3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_proofs_l1261_126150


namespace NUMINAMATH_CALUDE_first_donor_coins_l1261_126124

theorem first_donor_coins (d1 d2 d3 d4 : ℕ) : 
  d2 = 2 * d1 →
  d3 = 3 * d2 →
  d4 = 4 * d3 →
  d1 + d2 + d3 + d4 = 132 →
  d1 = 4 := by
sorry

end NUMINAMATH_CALUDE_first_donor_coins_l1261_126124


namespace NUMINAMATH_CALUDE_correct_seating_arrangements_l1261_126169

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

/-- The number of people to be seated. -/
def totalPeople : ℕ := 8

/-- The number of ways to seat the people under the given conditions. -/
def seatingArrangements : ℕ :=
  factorial totalPeople - 2 * (factorial (totalPeople - 1) * factorial 2)

theorem correct_seating_arrangements :
  seatingArrangements = 20160 := by sorry

end NUMINAMATH_CALUDE_correct_seating_arrangements_l1261_126169


namespace NUMINAMATH_CALUDE_row3_seat6_representation_l1261_126181

/-- Represents a seat in a movie theater -/
structure Seat :=
  (row : ℕ)
  (seat : ℕ)

/-- The representation format for seats in the theater -/
def represent (s : Seat) : ℕ × ℕ := (s.row, s.seat)

/-- Given condition: (5,8) represents row 5, seat 8 -/
axiom example_representation : represent { row := 5, seat := 8 } = (5, 8)

/-- Theorem: The representation of row 3, seat 6 is (3,6) -/
theorem row3_seat6_representation :
  represent { row := 3, seat := 6 } = (3, 6) := by
  sorry

end NUMINAMATH_CALUDE_row3_seat6_representation_l1261_126181


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1261_126193

-- Define the inverse proportionality relationship
def inverse_proportional (x y : ℝ) := ∃ k : ℝ, k ≠ 0 ∧ x * y = k

-- Theorem statement
theorem inverse_proportion_problem (x y : ℝ → ℝ) :
  (∀ a b : ℝ, inverse_proportional (x a) (y a)) →
  x 2 = 4 →
  x (-3) = -8/3 ∧ x 6 = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1261_126193


namespace NUMINAMATH_CALUDE_cube_side_length_l1261_126160

/-- The side length of a cube given paint cost and coverage -/
theorem cube_side_length 
  (paint_cost : ℝ)  -- Cost of paint per kg
  (paint_coverage : ℝ)  -- Area covered by 1 kg of paint in sq. ft
  (total_cost : ℝ)  -- Total cost to paint the cube
  (h1 : paint_cost = 40)  -- Paint costs Rs. 40 per kg
  (h2 : paint_coverage = 20)  -- 1 kg of paint covers 20 sq. ft
  (h3 : total_cost = 10800)  -- Total cost is Rs. 10800
  : ∃ (side_length : ℝ), side_length = 30 ∧ 
    total_cost = 6 * side_length^2 * paint_cost / paint_coverage :=
by sorry

end NUMINAMATH_CALUDE_cube_side_length_l1261_126160


namespace NUMINAMATH_CALUDE_sqrt2_irrational_l1261_126120

theorem sqrt2_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ (p : ℚ) / q = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_irrational_l1261_126120


namespace NUMINAMATH_CALUDE_student_weighted_average_l1261_126157

def weighted_average (courses1 courses2 courses3 : ℕ) (grade1 grade2 grade3 : ℚ) : ℚ :=
  (courses1 * grade1 + courses2 * grade2 + courses3 * grade3) / (courses1 + courses2 + courses3)

theorem student_weighted_average :
  let courses1 := 8
  let courses2 := 6
  let courses3 := 10
  let grade1 := 92
  let grade2 := 88
  let grade3 := 76
  abs (weighted_average courses1 courses2 courses3 grade1 grade2 grade3 - 84.3) < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_student_weighted_average_l1261_126157


namespace NUMINAMATH_CALUDE_pancake_theorem_l1261_126133

/-- The fraction of pancakes that could be flipped -/
def flipped_fraction : ℚ := 4 / 5

/-- The fraction of flipped pancakes that didn't burn -/
def not_burnt_fraction : ℚ := 51 / 100

/-- The fraction of edible pancakes that weren't dropped -/
def not_dropped_fraction : ℚ := 5 / 6

/-- The percentage of pancakes Anya could offer her family -/
def offered_percentage : ℚ := flipped_fraction * not_burnt_fraction * not_dropped_fraction * 100

theorem pancake_theorem : 
  ∃ (ε : ℚ), abs (offered_percentage - 34) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_pancake_theorem_l1261_126133


namespace NUMINAMATH_CALUDE_M_inequality_l1261_126192

/-- The number of h-subsets with property P_k(X) in a set X of size n -/
def M (n k h : ℕ) : ℕ := sorry

/-- Theorem stating the inequality for M(n,k,h) -/
theorem M_inequality (n k h : ℕ) :
  (n.choose h) / (k.choose h) ≤ M n k h ∧ M n k h ≤ (n - k + h).choose h :=
sorry

end NUMINAMATH_CALUDE_M_inequality_l1261_126192


namespace NUMINAMATH_CALUDE_systematic_sample_smallest_element_l1261_126147

/-- Represents a systematic sample -/
structure SystematicSample where
  total : ℕ
  sampleSize : ℕ
  interval : ℕ
  containsElement : ℕ

/-- The smallest element in a systematic sample -/
def smallestElement (s : SystematicSample) : ℕ :=
  s.interval * (s.containsElement / s.interval)

theorem systematic_sample_smallest_element 
  (s : SystematicSample) 
  (h1 : s.total = 360)
  (h2 : s.sampleSize = 30)
  (h3 : s.interval = s.total / s.sampleSize)
  (h4 : s.containsElement = 105)
  (h5 : s.containsElement ≤ s.total)
  : smallestElement s = 96 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_smallest_element_l1261_126147


namespace NUMINAMATH_CALUDE_number_puzzle_l1261_126129

theorem number_puzzle (x : ℤ) : (x + 2)^2 = x^2 - 2016 → x = -505 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1261_126129


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l1261_126118

theorem matrix_equation_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-21, -2; 13, 1]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![71/14, -109/14; -43/14, 67/14]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l1261_126118


namespace NUMINAMATH_CALUDE_parabola_decreasing_range_l1261_126114

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem parabola_decreasing_range :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (f x₁ > f x₂ ↔ x₁ < 1 ∧ x₂ < 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_decreasing_range_l1261_126114


namespace NUMINAMATH_CALUDE_triangle_OAB_area_and_point_C_l1261_126153

-- Define points in 2D space
def O : Fin 2 → ℝ := ![0, 0]
def A : Fin 2 → ℝ := ![2, 4]
def B : Fin 2 → ℝ := ![6, -2]

-- Define the area of a triangle given three points
def triangleArea (p1 p2 p3 : Fin 2 → ℝ) : ℝ := sorry

-- Define a function to check if two line segments are parallel
def isParallel (p1 p2 p3 p4 : Fin 2 → ℝ) : Prop := sorry

-- Define a function to calculate the length of a line segment
def segmentLength (p1 p2 : Fin 2 → ℝ) : ℝ := sorry

theorem triangle_OAB_area_and_point_C :
  (triangleArea O A B = 14) ∧
  (∃ (C : Fin 2 → ℝ), (C = ![4, -6] ∨ C = ![8, 2]) ∧
                      isParallel O A B C ∧
                      segmentLength O A = segmentLength B C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_OAB_area_and_point_C_l1261_126153


namespace NUMINAMATH_CALUDE_min_value_tan_sum_l1261_126172

/-- For any acute-angled triangle ABC, the expression 
    3 tan B tan C + 2 tan A tan C + tan A tan B 
    is always greater than or equal to 6 + 2√3 + 2√2 + 2√6 -/
theorem min_value_tan_sum (A B C : ℝ) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
    (h_triangle : A + B + C = π) : 
  3 * Real.tan B * Real.tan C + 2 * Real.tan A * Real.tan C + Real.tan A * Real.tan B 
    ≥ 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 2 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_tan_sum_l1261_126172
