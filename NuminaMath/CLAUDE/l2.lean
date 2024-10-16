import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l2_293

theorem perpendicular_vectors_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![x, 2]
  (a 0 * b 0 + a 1 * b 1 = 0) →  -- perpendicular condition
  ‖a + 2 • b‖ = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l2_293


namespace NUMINAMATH_CALUDE_product_seven_reciprocal_squares_sum_l2_285

theorem product_seven_reciprocal_squares_sum (a b : ℕ) (h : a * b = 7) :
  (1 : ℚ) / (a ^ 2) + (1 : ℚ) / (b ^ 2) = 50 / 49 := by
  sorry

end NUMINAMATH_CALUDE_product_seven_reciprocal_squares_sum_l2_285


namespace NUMINAMATH_CALUDE_rectangle_uniquely_symmetric_l2_248

-- Define the properties
def axisymmetric (shape : Type) : Prop := sorry

def centrally_symmetric (shape : Type) : Prop := sorry

-- Define the shapes
def equilateral_triangle : Type := sorry
def rectangle : Type := sorry
def parallelogram : Type := sorry
def regular_pentagon : Type := sorry

-- Theorem statement
theorem rectangle_uniquely_symmetric :
  (axisymmetric equilateral_triangle ∧ centrally_symmetric equilateral_triangle) = False ∧
  (axisymmetric rectangle ∧ centrally_symmetric rectangle) = True ∧
  (axisymmetric parallelogram ∧ centrally_symmetric parallelogram) = False ∧
  (axisymmetric regular_pentagon ∧ centrally_symmetric regular_pentagon) = False :=
sorry

end NUMINAMATH_CALUDE_rectangle_uniquely_symmetric_l2_248


namespace NUMINAMATH_CALUDE_franks_total_work_hours_l2_294

/-- Calculates the total hours worked given the number of hours per day and number of days --/
def totalHours (hoursPerDay : ℕ) (numDays : ℕ) : ℕ :=
  hoursPerDay * numDays

/-- Theorem: Frank's total work hours --/
theorem franks_total_work_hours :
  totalHours 8 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_franks_total_work_hours_l2_294


namespace NUMINAMATH_CALUDE_partner_a_capital_l2_230

/-- Represents the partnership structure and profit distribution --/
structure Partnership where
  total_profit : ℝ
  a_share : ℝ
  b_share : ℝ
  c_share : ℝ
  a_share_def : a_share = (2/3) * total_profit
  bc_share_def : b_share = c_share
  bc_share_sum : b_share + c_share = (1/3) * total_profit

/-- Represents the change in profit rate and its effect on partner a's income --/
structure ProfitChange where
  initial_rate : ℝ
  final_rate : ℝ
  a_income_increase : ℝ
  rate_def : final_rate - initial_rate = 0.02
  initial_rate_def : initial_rate = 0.05
  income_increase_def : a_income_increase = 200

/-- The main theorem stating the capital of partner a --/
theorem partner_a_capital 
  (p : Partnership) 
  (pc : ProfitChange) : 
  ∃ (capital_a : ℝ), capital_a = 300000 := by
  sorry

end NUMINAMATH_CALUDE_partner_a_capital_l2_230


namespace NUMINAMATH_CALUDE_sphere_radius_from_hole_l2_249

/-- Given a spherical hole with a width of 30 cm at the top and a depth of 10 cm,
    the radius of the sphere that created this hole is 16.25 cm. -/
theorem sphere_radius_from_hole (hole_width : ℝ) (hole_depth : ℝ) (sphere_radius : ℝ) : 
  hole_width = 30 → 
  hole_depth = 10 → 
  sphere_radius = (hole_width ^ 2 / 4 + hole_depth ^ 2) / (2 * hole_depth) → 
  sphere_radius = 16.25 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_from_hole_l2_249


namespace NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l2_273

theorem largest_angle_in_ratio_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ratio : b = 2 * a ∧ c = 3 * a) (h_sum : a + b + c = 180) :
  c = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l2_273


namespace NUMINAMATH_CALUDE_abs_negative_2023_l2_290

theorem abs_negative_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l2_290


namespace NUMINAMATH_CALUDE_influenza_test_probability_l2_221

theorem influenza_test_probability 
  (P : Set Ω → ℝ) 
  (A C : Set Ω) 
  (h1 : P (A ∩ C) / P C = 0.9)
  (h2 : P ((Cᶜ) ∩ (Aᶜ)) / P (Cᶜ) = 0.9)
  (h3 : P C = 0.005)
  : P (C ∩ A) / P A = 9 / 208 := by
  sorry

end NUMINAMATH_CALUDE_influenza_test_probability_l2_221


namespace NUMINAMATH_CALUDE_bicycle_speed_calculation_l2_205

theorem bicycle_speed_calculation (distance : ℝ) (speed_difference : ℝ) (time_ratio : ℝ) :
  distance = 10 ∧ 
  speed_difference = 45 ∧ 
  time_ratio = 4 →
  ∃ x : ℝ, x = 15 ∧ 
    distance / x = time_ratio * (distance / (x + speed_difference)) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_speed_calculation_l2_205


namespace NUMINAMATH_CALUDE_difference_closure_l2_239

def is_closed_set (A : Set Int) : Prop :=
  (∃ (a b : Int), a ∈ A ∧ a > 0 ∧ b ∈ A ∧ b < 0) ∧
  (∀ a b : Int, a ∈ A → b ∈ A → (2 * a) ∈ A ∧ (a + b) ∈ A)

theorem difference_closure (A : Set Int) (h : is_closed_set A) :
  ∀ x y : Int, x ∈ A → y ∈ A → (x - y) ∈ A :=
by sorry

end NUMINAMATH_CALUDE_difference_closure_l2_239


namespace NUMINAMATH_CALUDE_inequality_proof_l2_251

theorem inequality_proof (a : ℝ) (x : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1/2) (h3 : x ≥ 0) :
  let f : ℝ → ℝ := λ y => Real.exp y
  let g : ℝ → ℝ := λ y => a * y + 1
  1 / f x + x / g x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2_251


namespace NUMINAMATH_CALUDE_sin_beta_value_l2_275

theorem sin_beta_value (α β : ℝ) 
  (h : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 3/5) : 
  Real.sin β = -(3/5) := by
  sorry

end NUMINAMATH_CALUDE_sin_beta_value_l2_275


namespace NUMINAMATH_CALUDE_equation_solution_l2_264

theorem equation_solution : 
  ∃ y : ℝ, (7 * y / (y + 4) - 5 / (y + 4) = 2 / (y + 4)) ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2_264


namespace NUMINAMATH_CALUDE_max_profit_at_eight_days_max_profit_value_l2_231

/-- Profit function for fruit wholesaler --/
def profit (x : ℕ) : ℝ :=
  let initial_amount := 500
  let purchase_price := 40
  let base_selling_price := 60
  let daily_price_increase := 2
  let daily_loss := 10
  let daily_storage_cost := 40
  let selling_price := base_selling_price + daily_price_increase * x
  let remaining_amount := initial_amount - daily_loss * x
  (selling_price * remaining_amount) - (daily_storage_cost * x) - (initial_amount * purchase_price)

/-- Maximum storage time in days --/
def max_storage_time : ℕ := 8

/-- Theorem: Maximum profit is achieved at 8 days of storage --/
theorem max_profit_at_eight_days :
  ∀ x : ℕ, x ≤ max_storage_time → profit x ≤ profit max_storage_time :=
sorry

/-- Theorem: Maximum profit is 11600 yuan --/
theorem max_profit_value :
  profit max_storage_time = 11600 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_eight_days_max_profit_value_l2_231


namespace NUMINAMATH_CALUDE_exists_coverable_parallelepiped_l2_222

/-- Represents a parallelepiped with integer dimensions -/
structure Parallelepiped where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ+

/-- Checks if three squares can cover a parallelepiped with shared edges -/
def can_cover_with_shared_edges (p : Parallelepiped) (s1 s2 s3 : Square) : Prop :=
  -- The squares cover the surface area of the parallelepiped
  2 * (p.length * p.width + p.length * p.height + p.width * p.height) =
    s1.side * s1.side + s2.side * s2.side + s3.side * s3.side ∧
  -- Each pair of squares shares an edge
  (s1.side = p.length ∨ s1.side = p.width ∨ s1.side = p.height) ∧
  (s2.side = p.length ∨ s2.side = p.width ∨ s2.side = p.height) ∧
  (s3.side = p.length ∨ s3.side = p.width ∨ s3.side = p.height)

/-- Theorem stating the existence of a parallelepiped coverable by three squares with shared edges -/
theorem exists_coverable_parallelepiped :
  ∃ (p : Parallelepiped) (s1 s2 s3 : Square),
    can_cover_with_shared_edges p s1 s2 s3 :=
  sorry

end NUMINAMATH_CALUDE_exists_coverable_parallelepiped_l2_222


namespace NUMINAMATH_CALUDE_leftover_books_l2_211

/-- The number of leftover books when repacking from boxes of 45 to boxes of 47 -/
theorem leftover_books (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) :
  initial_boxes = 1500 →
  books_per_initial_box = 45 →
  books_per_new_box = 47 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 13 := by
  sorry

#eval (1500 * 45) % 47  -- This should output 13

end NUMINAMATH_CALUDE_leftover_books_l2_211


namespace NUMINAMATH_CALUDE_value_of_expression_l2_235

theorem value_of_expression (x : ℝ) (h : 3 * x^2 - 2 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2_235


namespace NUMINAMATH_CALUDE_scientific_notation_929000_l2_246

/-- Expresses a number in scientific notation -/
def scientific_notation (n : ℕ) : ℝ × ℤ :=
  sorry

theorem scientific_notation_929000 :
  scientific_notation 929000 = (9.29, 5) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_929000_l2_246


namespace NUMINAMATH_CALUDE_f_evaluation_l2_233

/-- The function f(x) = 3x^2 - 5x + 8 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 8

/-- Theorem stating that 3f(4) + 2f(-4) = 260 -/
theorem f_evaluation : 3 * f 4 + 2 * f (-4) = 260 := by
  sorry

end NUMINAMATH_CALUDE_f_evaluation_l2_233


namespace NUMINAMATH_CALUDE_cost_of_four_birdhouses_l2_229

/-- The cost to build a given number of birdhouses -/
def cost_of_birdhouses (num_birdhouses : ℕ) : ℚ :=
  let planks_per_house : ℕ := 7
  let nails_per_house : ℕ := 20
  let cost_per_plank : ℚ := 3
  let cost_per_nail : ℚ := 1/20
  num_birdhouses * (planks_per_house * cost_per_plank + nails_per_house * cost_per_nail)

/-- Theorem stating that the cost to build 4 birdhouses is $88 -/
theorem cost_of_four_birdhouses :
  cost_of_birdhouses 4 = 88 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_four_birdhouses_l2_229


namespace NUMINAMATH_CALUDE_vector_dot_product_l2_217

theorem vector_dot_product (a b : ℝ × ℝ) (h1 : a = (1, -1)) (h2 : b = (-1, 2)) :
  (2 • a + b) • a = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l2_217


namespace NUMINAMATH_CALUDE_total_rent_calculation_l2_272

/-- Calculates the total rent collected in a year for a rental building --/
theorem total_rent_calculation (total_units : ℕ) (occupancy_rate : ℚ) (rent_per_unit : ℕ) : 
  total_units = 100 → 
  occupancy_rate = 3/4 →
  rent_per_unit = 400 →
  (total_units : ℚ) * occupancy_rate * rent_per_unit * 12 = 360000 := by
  sorry

end NUMINAMATH_CALUDE_total_rent_calculation_l2_272


namespace NUMINAMATH_CALUDE_cutlery_theorem_l2_298

def cutlery_count (initial_knives : ℕ) : ℕ :=
  let initial_teaspoons := 2 * initial_knives
  let additional_knives := initial_knives / 3
  let additional_teaspoons := (2 * initial_teaspoons) / 3
  let total_knives := initial_knives + additional_knives
  let total_teaspoons := initial_teaspoons + additional_teaspoons
  total_knives + total_teaspoons

theorem cutlery_theorem : cutlery_count 24 = 112 := by
  sorry

end NUMINAMATH_CALUDE_cutlery_theorem_l2_298


namespace NUMINAMATH_CALUDE_daniel_earnings_l2_204

-- Define the delivery schedule and prices
def monday_fabric : ℕ := 20
def monday_yarn : ℕ := 15
def tuesday_fabric : ℕ := 2 * monday_fabric
def tuesday_yarn : ℕ := monday_yarn + 10
def wednesday_fabric : ℕ := tuesday_fabric / 4
def wednesday_yarn : ℕ := tuesday_yarn / 2 + 1  -- Rounded up

def fabric_price : ℕ := 2
def yarn_price : ℕ := 3

-- Calculate total yards of fabric and yarn
def total_fabric : ℕ := monday_fabric + tuesday_fabric + wednesday_fabric
def total_yarn : ℕ := monday_yarn + tuesday_yarn + wednesday_yarn

-- Calculate total earnings
def total_earnings : ℕ := fabric_price * total_fabric + yarn_price * total_yarn

-- Theorem to prove
theorem daniel_earnings : total_earnings = 299 := by
  sorry

end NUMINAMATH_CALUDE_daniel_earnings_l2_204


namespace NUMINAMATH_CALUDE_inequality_condition_l2_283

theorem inequality_condition (x y : ℝ) : 
  (x > 0 ∧ y > 0 → y / x + x / y ≥ 2) ∧ 
  ¬(y / x + x / y ≥ 2 → x > 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l2_283


namespace NUMINAMATH_CALUDE_area_of_N_region_l2_257

-- Define the plane region for point M
def plane_region_M (a b : ℝ) : Prop := sorry

-- Define the transformation from M to N
def transform_M_to_N (a b : ℝ) : ℝ × ℝ := (a + b, a - b)

-- Define the plane region for point N
def plane_region_N (x y : ℝ) : Prop := sorry

-- Theorem statement
theorem area_of_N_region : 
  ∀ (a b : ℝ), plane_region_M a b → 
  (∃ (S : Set (ℝ × ℝ)), (∀ (x y : ℝ), (x, y) ∈ S ↔ plane_region_N x y) ∧ 
                         MeasureTheory.volume S = 4) :=
sorry

end NUMINAMATH_CALUDE_area_of_N_region_l2_257


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_8640_l2_270

/-- The number of positive integer factors of 8640 that are perfect squares -/
def num_perfect_square_factors (n : ℕ) : ℕ :=
  (Finset.range 4).card * (Finset.range 2).card * (Finset.range 1).card

/-- The prime factorization of 8640 -/
def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  [(2, 6), (3, 3), (5, 1)]

theorem perfect_square_factors_of_8640 :
  num_perfect_square_factors 8640 = 8 ∧ prime_factorization 8640 = [(2, 6), (3, 3), (5, 1)] := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_8640_l2_270


namespace NUMINAMATH_CALUDE_cube_set_closed_under_multiplication_cube_set_not_closed_under_addition_cube_set_not_closed_under_division_cube_set_not_closed_under_squaring_l2_240

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def cube_set : Set ℕ := {n : ℕ | is_cube n ∧ n > 0}

theorem cube_set_closed_under_multiplication (a b : ℕ) (ha : a ∈ cube_set) (hb : b ∈ cube_set) :
  (a * b) ∈ cube_set :=
sorry

theorem cube_set_not_closed_under_addition :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ (a + b) ∉ cube_set :=
sorry

theorem cube_set_not_closed_under_division :
  ∃ a b : ℕ, a ∈ cube_set ∧ b ∈ cube_set ∧ b ≠ 0 ∧ (a / b) ∉ cube_set :=
sorry

theorem cube_set_not_closed_under_squaring :
  ∃ a : ℕ, a ∈ cube_set ∧ (a^2) ∉ cube_set :=
sorry

end NUMINAMATH_CALUDE_cube_set_closed_under_multiplication_cube_set_not_closed_under_addition_cube_set_not_closed_under_division_cube_set_not_closed_under_squaring_l2_240


namespace NUMINAMATH_CALUDE_hcl_formation_l2_267

/-- Represents a chemical compound with its coefficient in a chemical equation -/
structure Compound where
  name : String
  coefficient : ℚ

/-- Represents a chemical equation with reactants and products -/
structure ChemicalEquation where
  reactants : List Compound
  products : List Compound

/-- Calculates the number of moles of HCl formed given the initial moles of reactants -/
def molesOfHClFormed (h2so4_moles : ℚ) (nacl_moles : ℚ) (equation : ChemicalEquation) : ℚ :=
  sorry

/-- The main theorem stating that 3 moles of HCl are formed -/
theorem hcl_formation :
  let equation : ChemicalEquation := {
    reactants := [
      {name := "H₂SO₄", coefficient := 1},
      {name := "NaCl", coefficient := 2}
    ],
    products := [
      {name := "HCl", coefficient := 2},
      {name := "Na₂SO₄", coefficient := 1}
    ]
  }
  molesOfHClFormed 3 3 equation = 3 :=
by sorry

end NUMINAMATH_CALUDE_hcl_formation_l2_267


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2_201

-- Define the relationship between x and y
def inverse_relation (x y : ℝ) : Prop := ∃ k : ℝ, k > 0 ∧ 3 * x^2 * y = k

-- Theorem statement
theorem inverse_variation_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_pos₃ : y₁ > 0) (h_pos₄ : y₂ > 0)
  (h_inverse : inverse_relation x₁ y₁ ∧ inverse_relation x₂ y₂)
  (h_initial : x₁ = 3 ∧ y₁ = 30)
  (h_final : x₂ = 6) :
  y₂ = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2_201


namespace NUMINAMATH_CALUDE_actual_car_body_mass_l2_243

/-- Represents the scale factor between the model and the actual car body -/
def scaleFactor : ℝ := 10

/-- Represents the mass of the model car body in kilograms -/
def modelMass : ℝ := 1

/-- Calculates the volume ratio between the actual car body and the model -/
def volumeRatio : ℝ := scaleFactor ^ 3

/-- Calculates the mass of the actual car body in kilograms -/
def actualMass : ℝ := modelMass * volumeRatio

/-- Theorem stating that the mass of the actual car body is 1000 kg -/
theorem actual_car_body_mass : actualMass = 1000 := by
  sorry

end NUMINAMATH_CALUDE_actual_car_body_mass_l2_243


namespace NUMINAMATH_CALUDE_two_numbers_ratio_problem_l2_224

theorem two_numbers_ratio_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / y = 3 →
  (x^2 + y^2) / (x + y) = 5 →
  x = 6 ∧ y = 2 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_ratio_problem_l2_224


namespace NUMINAMATH_CALUDE_equality_of_exponents_l2_237

theorem equality_of_exponents (x y z : ℝ) 
  (h : x * (y + z - x) / x = y * (z + x - y) / y ∧ 
       y * (z + x - y) / y = z * (x + y - z) / z) : 
  x^y * y^x = z^y * y^z ∧ z^y * y^z = x^z * z^x := by
  sorry

end NUMINAMATH_CALUDE_equality_of_exponents_l2_237


namespace NUMINAMATH_CALUDE_only_y_eq_0_is_equation_l2_250

-- Define a type for the expressions
inductive Expression
  | Addition : Expression
  | Equation : Expression
  | Inequality : Expression
  | NotEqual : Expression

-- Define a function to check if an expression is an equation
def isEquation (e : Expression) : Prop :=
  match e with
  | Expression.Equation => True
  | _ => False

-- State the theorem
theorem only_y_eq_0_is_equation :
  let x_plus_1_5 := Expression.Addition
  let y_eq_0 := Expression.Equation
  let six_plus_x_lt_5 := Expression.Inequality
  let ab_neq_60 := Expression.NotEqual
  (¬ isEquation x_plus_1_5) ∧
  (isEquation y_eq_0) ∧
  (¬ isEquation six_plus_x_lt_5) ∧
  (¬ isEquation ab_neq_60) :=
by sorry


end NUMINAMATH_CALUDE_only_y_eq_0_is_equation_l2_250


namespace NUMINAMATH_CALUDE_sin_equality_proof_l2_259

theorem sin_equality_proof (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) :
  Real.sin (n * π / 180) = Real.sin (721 * π / 180) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_proof_l2_259


namespace NUMINAMATH_CALUDE_range_of_m_for_exponential_equation_l2_276

/-- The range of m for which the equation 9^(-x^x) = 4 * 3^(-x^x) + m has a real solution for x -/
theorem range_of_m_for_exponential_equation :
  ∀ m : ℝ, (∃ x : ℝ, (9 : ℝ)^(-x^x) = 4 * (3 : ℝ)^(-x^x) + m) ↔ -3 ≤ m ∧ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_exponential_equation_l2_276


namespace NUMINAMATH_CALUDE_three_digit_number_from_sum_l2_206

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h_a : a < 10
  h_b : b < 10
  h_c : c < 10

/-- Calculates the sum of permutations of a three-digit number -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c +
  100 * n.a + 10 * n.c + n.b +
  100 * n.b + 10 * n.a + n.c +
  100 * n.b + 10 * n.c + n.a +
  100 * n.c + 10 * n.a + n.b +
  100 * n.c + 10 * n.b + n.a

theorem three_digit_number_from_sum (N : Nat) (h_N : N = 3194) :
  ∃ (n : ThreeDigitNumber), sumOfPermutations n = N ∧ n.a = 3 ∧ n.b = 5 ∧ n.c = 8 := by
  sorry

#eval sumOfPermutations { a := 3, b := 5, c := 8, h_a := by norm_num, h_b := by norm_num, h_c := by norm_num }

end NUMINAMATH_CALUDE_three_digit_number_from_sum_l2_206


namespace NUMINAMATH_CALUDE_tourism_revenue_scientific_notation_l2_225

theorem tourism_revenue_scientific_notation :
  let revenue_billion : ℝ := 1480.56
  let scientific_notation : ℝ := 1.48056 * (10 ^ 11)
  revenue_billion * (10 ^ 9) = scientific_notation :=
by sorry

end NUMINAMATH_CALUDE_tourism_revenue_scientific_notation_l2_225


namespace NUMINAMATH_CALUDE_fifteenth_even_multiple_of_5_l2_291

/-- A function that returns the nth positive integer that is both even and a multiple of 5 -/
def evenMultipleOf5 (n : ℕ) : ℕ := 10 * n

/-- The 15th positive integer that is both even and a multiple of 5 is 150 -/
theorem fifteenth_even_multiple_of_5 : evenMultipleOf5 15 = 150 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_even_multiple_of_5_l2_291


namespace NUMINAMATH_CALUDE_increase_by_percentage_l2_215

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 80 → percentage = 150 → result = initial * (1 + percentage / 100) → result = 200 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l2_215


namespace NUMINAMATH_CALUDE_no_convex_polygon_from_regular_triangles_l2_256

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields

/-- A regular triangle -/
structure RegularTriangle where
  -- Add necessary fields

/-- Predicate to check if triangles are non-overlapping -/
def non_overlapping (T : List RegularTriangle) : Prop :=
  sorry

/-- Predicate to check if triangles are distinct -/
def distinct (T : List RegularTriangle) : Prop :=
  sorry

/-- Predicate to check if a polygon is composed of given triangles -/
def composed_of (P : ConvexPolygon) (T : List RegularTriangle) : Prop :=
  sorry

theorem no_convex_polygon_from_regular_triangles 
  (P : ConvexPolygon) (T : List RegularTriangle) :
  T.length ≥ 2 → non_overlapping T → distinct T → ¬(composed_of P T) :=
sorry

end NUMINAMATH_CALUDE_no_convex_polygon_from_regular_triangles_l2_256


namespace NUMINAMATH_CALUDE_quadratic_radical_equality_l2_268

theorem quadratic_radical_equality :
  ∃! x : ℝ, x^2 - 2 = 2*x - 2 ∧ x^2 - 2 ≥ 0 ∧ 2*x - 2 ≥ 0 ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_radical_equality_l2_268


namespace NUMINAMATH_CALUDE_ellipse_properties_l2_247

/-- An ellipse with specific properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  max_distance_to_foci : ℝ
  min_distance_to_foci : ℝ

/-- The standard form of an ellipse equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

/-- Theorem about a specific ellipse and its properties -/
theorem ellipse_properties (C : Ellipse) 
    (h1 : C.center = (0, 0))
    (h2 : C.foci_on_x_axis = true)
    (h3 : C.max_distance_to_foci = 3)
    (h4 : C.min_distance_to_foci = 1) :
  (∃ (x y : ℝ), standard_equation 4 3 x y) ∧ 
  (∃ (P F₁ F₂ : ℝ × ℝ), 
    (standard_equation 4 3 P.1 P.2) →
    (F₁ = (-1, 0) ∧ F₂ = (1, 0)) →
    (∀ (Q : ℝ × ℝ), standard_equation 4 3 Q.1 Q.2 → 
      dot_product (P.1 - F₁.1, P.2 - F₁.2) (P.1 - F₂.1, P.2 - F₂.2) ≤ 3 ∧
      dot_product (Q.1 - F₁.1, Q.2 - F₁.2) (Q.1 - F₂.1, Q.2 - F₂.2) ≥ 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_properties_l2_247


namespace NUMINAMATH_CALUDE_permutations_of_four_l2_274

theorem permutations_of_four (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_four_l2_274


namespace NUMINAMATH_CALUDE_barry_larry_reach_l2_207

/-- The maximum height Barry and Larry can reach when Barry stands on Larry's shoulders -/
def max_reach (barry_reach : ℝ) (larry_height : ℝ) (larry_shoulder_ratio : ℝ) : ℝ :=
  barry_reach + larry_height * larry_shoulder_ratio

/-- Theorem stating the maximum reach of Barry and Larry -/
theorem barry_larry_reach :
  let barry_reach : ℝ := 5
  let larry_height : ℝ := 5
  let larry_shoulder_ratio : ℝ := 0.8
  max_reach barry_reach larry_height larry_shoulder_ratio = 9 := by
  sorry

end NUMINAMATH_CALUDE_barry_larry_reach_l2_207


namespace NUMINAMATH_CALUDE_complementary_of_35_is_55_l2_260

/-- The complementary angle of a given angle in degrees -/
def complementaryAngle (angle : ℝ) : ℝ := 90 - angle

/-- Theorem: The complementary angle of 35° is 55° -/
theorem complementary_of_35_is_55 :
  complementaryAngle 35 = 55 := by
  sorry

end NUMINAMATH_CALUDE_complementary_of_35_is_55_l2_260


namespace NUMINAMATH_CALUDE_inequality_lower_bound_l2_295

theorem inequality_lower_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 2*y) * (2/x + 1/y) ≥ 8 ∧
  ∀ ε > 0, ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ (x₀ + 2*y₀) * (2/x₀ + 1/y₀) < 8 + ε :=
sorry

end NUMINAMATH_CALUDE_inequality_lower_bound_l2_295


namespace NUMINAMATH_CALUDE_uniform_purchase_theorem_l2_245

/-- Represents the price per set based on the number of sets purchased -/
def price_per_set (n : ℕ) : ℕ :=
  if n ≤ 50 then 50
  else if n ≤ 90 then 40
  else 30

/-- The total number of students in both classes -/
def total_students : ℕ := 92

/-- The range of students in Class A -/
def class_a_range (n : ℕ) : Prop := 51 < n ∧ n < 55

/-- The total amount paid when classes purchase uniforms separately -/
def separate_purchase_total : ℕ := 4080

/-- Theorem stating the number of students in each class and the most cost-effective plan -/
theorem uniform_purchase_theorem (class_a class_b : ℕ) :
  class_a + class_b = total_students →
  class_a_range class_a →
  price_per_set class_a * class_a + price_per_set class_b * class_b = separate_purchase_total →
  (class_a = 52 ∧ class_b = 40) ∧
  price_per_set 91 * 91 = 2730 ∧
  ∀ n : ℕ, n ≤ total_students - 8 → price_per_set 91 * 91 ≤ price_per_set n * n :=
by sorry

end NUMINAMATH_CALUDE_uniform_purchase_theorem_l2_245


namespace NUMINAMATH_CALUDE_p_cubed_plus_mp_l2_282

theorem p_cubed_plus_mp (p m : ℤ) (h_p_odd : Odd p) : 
  Odd (p^3 + m*p) ↔ Even m := by
  sorry

end NUMINAMATH_CALUDE_p_cubed_plus_mp_l2_282


namespace NUMINAMATH_CALUDE_express_regular_train_speed_ratio_l2_279

/-- The ratio of speeds between an express train and a regular train -/
def speed_ratio : ℝ := 2.5

/-- The time taken by the regular train from Moscow to St. Petersburg -/
def regular_train_time : ℝ := 10

/-- The time difference in arrival between regular and express trains -/
def arrival_time_difference : ℝ := 3

/-- The waiting time for the express train -/
def express_train_wait_time : ℝ := 3

/-- The time after departure when both trains are at the same distance from Moscow -/
def equal_distance_time : ℝ := 2

theorem express_regular_train_speed_ratio :
  ∀ (v_regular v_express : ℝ),
    v_regular > 0 →
    v_express > 0 →
    express_train_wait_time > 2.5 →
    v_express * equal_distance_time = v_regular * (express_train_wait_time + equal_distance_time) →
    v_express * (regular_train_time - arrival_time_difference - express_train_wait_time) = v_regular * regular_train_time →
    v_express / v_regular = speed_ratio := by
  sorry

end NUMINAMATH_CALUDE_express_regular_train_speed_ratio_l2_279


namespace NUMINAMATH_CALUDE_largest_quantity_l2_278

theorem largest_quantity (a b c d : ℝ) 
  (h : a + 1 = b - 2 ∧ a + 1 = c + 3 ∧ a + 1 = d - 4) : 
  d = max a (max b c) ∧ d ≥ a ∧ d ≥ b ∧ d ≥ c := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l2_278


namespace NUMINAMATH_CALUDE_tan_inequality_l2_288

open Real

theorem tan_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < π/2) (h₃ : 0 < x₂) (h₄ : x₂ < π/2) (h₅ : x₁ ≠ x₂) :
  (tan x₁ + tan x₂) / 2 > tan ((x₁ + x₂) / 2) := by
  sorry

#check tan_inequality

end NUMINAMATH_CALUDE_tan_inequality_l2_288


namespace NUMINAMATH_CALUDE_no_finite_algorithm_for_infinite_sum_l2_262

-- Define what an algorithm is
def Algorithm : Type := ℕ → ℕ

-- Define the property of finiteness for algorithms
def IsFinite (a : Algorithm) : Prop := ∃ n : ℕ, ∀ m : ℕ, m ≥ n → a m = a n

-- Define the infinite sum
def InfiniteSum : ℕ → ℕ
  | 0 => 0
  | n + 1 => InfiniteSum n + (n + 1)

-- Theorem: There is no finite algorithm that can calculate the infinite sum
theorem no_finite_algorithm_for_infinite_sum :
  ¬∃ (a : Algorithm), (IsFinite a) ∧ (∀ n : ℕ, a n = InfiniteSum n) :=
sorry

end NUMINAMATH_CALUDE_no_finite_algorithm_for_infinite_sum_l2_262


namespace NUMINAMATH_CALUDE_inequality_solution_l2_252

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 5) ↔ 
  (x < 0 ∨ (1 < x ∧ x < 2) ∨ 2 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2_252


namespace NUMINAMATH_CALUDE_total_spend_l2_277

-- Define the given conditions
def num_tshirts : ℕ := 3
def cost_per_tshirt : ℕ := 20
def cost_pants : ℕ := 50

-- State the theorem
theorem total_spend : 
  num_tshirts * cost_per_tshirt + cost_pants = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_spend_l2_277


namespace NUMINAMATH_CALUDE_factory_production_rate_l2_266

/-- Represents a chocolate factory's production parameters and calculates the hourly production rate. -/
def ChocolateFactory (total_candies : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  total_candies / (days * hours_per_day)

/-- Theorem stating that for the given production parameters, the factory produces 50 candies per hour. -/
theorem factory_production_rate :
  ChocolateFactory 4000 8 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_rate_l2_266


namespace NUMINAMATH_CALUDE_sector_area_l2_241

/-- Given a sector with radius R and perimeter 4R, its area is R^2 -/
theorem sector_area (R : ℝ) (R_pos : R > 0) : 
  let perimeter := 4 * R
  let arc_length := perimeter - 2 * R
  let area := (1 / 2) * R * arc_length
  area = R^2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l2_241


namespace NUMINAMATH_CALUDE_parabola_c_value_l2_253

/-- A parabola with equation y = ax^2 + bx + c, vertex (-1, -2), and passing through (-2, -1) has c = -1 -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ x y : ℝ, y = a*x^2 + b*x + c) →  -- Equation of the parabola
  (-2 = a*(-1)^2 + b*(-1) + c) →      -- Vertex condition
  (-1 = a*(-2)^2 + b*(-2) + c) →      -- Point condition
  c = -1 := by
sorry


end NUMINAMATH_CALUDE_parabola_c_value_l2_253


namespace NUMINAMATH_CALUDE_max_central_rectangle_area_l2_261

/-- Given a square of side length 23 divided into 9 rectangles, with 4 known areas,
    prove that the maximum area of the central rectangle is 180 -/
theorem max_central_rectangle_area :
  ∀ (a b c d e f : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
    a + b + c = 23 →
    d + e + f = 23 →
    a * d = 13 →
    b * f = 111 →
    c * e = 37 →
    a * f = 123 →
    b * e ≤ 180 :=
by sorry

end NUMINAMATH_CALUDE_max_central_rectangle_area_l2_261


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l2_281

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l2_281


namespace NUMINAMATH_CALUDE_terminal_side_point_theorem_l2_232

theorem terminal_side_point_theorem (m : ℝ) (hm : m ≠ 0) :
  let α := Real.arctan (3 * m / (-4 * m))
  (2 * Real.sin α + Real.cos α = 2/5) ∨ (2 * Real.sin α + Real.cos α = -2/5) := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_point_theorem_l2_232


namespace NUMINAMATH_CALUDE_vector_operation_result_l2_296

def a : ℝ × ℝ × ℝ := (3, 5, 1)
def b : ℝ × ℝ × ℝ := (2, 2, 3)
def c : ℝ × ℝ × ℝ := (4, -1, -3)

theorem vector_operation_result : 
  (2 : ℝ) • a - (3 : ℝ) • b + (4 : ℝ) • c = (16, 0, -19) := by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l2_296


namespace NUMINAMATH_CALUDE_fathers_contribution_l2_219

/-- Given the costs of items, savings, and the amount lacking, calculate the father's contribution --/
theorem fathers_contribution 
  (mp3_cost cd_cost savings lacking : ℕ) 
  (h1 : mp3_cost = 120)
  (h2 : cd_cost = 19)
  (h3 : savings = 55)
  (h4 : lacking = 64) :
  mp3_cost + cd_cost - savings + lacking = 148 := by
  sorry

end NUMINAMATH_CALUDE_fathers_contribution_l2_219


namespace NUMINAMATH_CALUDE_interest_rate_difference_l2_254

theorem interest_rate_difference (principal : ℝ) (original_rate higher_rate : ℝ) 
  (h1 : principal = 500)
  (h2 : principal * higher_rate / 100 - principal * original_rate / 100 = 30) :
  higher_rate - original_rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l2_254


namespace NUMINAMATH_CALUDE_tangents_not_necessarily_coincide_at_both_points_l2_226

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define a general circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of having exactly two intersection points
def has_two_intersections (c : Circle) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧
    parabola A.1 = A.2 ∧
    parabola B.1 = B.2 ∧
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2

-- Define the property of tangents coinciding at a point
def tangents_coincide_at (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1) * (2 * p.1) + (p.2 - c.center.2) = c.radius^2

-- Main theorem
theorem tangents_not_necessarily_coincide_at_both_points :
  ∃ c : Circle, has_two_intersections c ∧
    (∃ A B : ℝ × ℝ, A ≠ B ∧
      parabola A.1 = A.2 ∧
      parabola B.1 = B.2 ∧
      tangents_coincide_at c A ∧
      ¬tangents_coincide_at c B) :=
sorry

end NUMINAMATH_CALUDE_tangents_not_necessarily_coincide_at_both_points_l2_226


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2_265

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  (x + 2) * (x - 3) = 2 * x - 6

-- Define the general form of the equation
def general_form (x : ℝ) : Prop :=
  x^2 - 3*x = 0

-- Theorem statement
theorem quadratic_equation_solution :
  (∀ x, quadratic_equation x ↔ general_form x) ∧
  (∃ x₁ x₂, x₁ = 0 ∧ x₂ = 3 ∧ ∀ x, general_form x ↔ (x = x₁ ∨ x = x₂)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2_265


namespace NUMINAMATH_CALUDE_box_dimensions_l2_271

theorem box_dimensions (x : ℝ) 
  (h1 : x > 0)
  (h2 : ∃ (bow_length : ℝ), 6 * x + bow_length = 156)
  (h3 : ∃ (bow_length : ℝ), 7 * x + bow_length = 178) :
  x = 22 := by
sorry

end NUMINAMATH_CALUDE_box_dimensions_l2_271


namespace NUMINAMATH_CALUDE_total_dice_count_l2_220

theorem total_dice_count (ivan_dice : ℕ) (jerry_dice : ℕ) : 
  ivan_dice = 20 → 
  jerry_dice = 2 * ivan_dice → 
  ivan_dice + jerry_dice = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_dice_count_l2_220


namespace NUMINAMATH_CALUDE_min_chord_length_implies_m_l2_287

/-- Given a circle C: x^2 + y^2 = 4 and a line l: y = kx + m, 
    prove that if the minimum chord length cut by l on C is 2, then m = ±√3 -/
theorem min_chord_length_implies_m (k : ℝ) :
  (∀ x y, x^2 + y^2 = 4 → ∃ m, y = k*x + m) →
  (∃ m, ∀ x y, x^2 + y^2 = 4 ∧ y = k*x + m → 
    ∀ x1 y1 x2 y2, x1^2 + y1^2 = 4 ∧ y1 = k*x1 + m ∧ 
                   x2^2 + y2^2 = 4 ∧ y2 = k*x2 + m →
    (x1 - x2)^2 + (y1 - y2)^2 ≥ 4) →
  ∃ m, m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_chord_length_implies_m_l2_287


namespace NUMINAMATH_CALUDE_equation_solution_l2_269

theorem equation_solution : 
  ∃ t : ℝ, 3 * 3^t + Real.sqrt (9 * 9^t) = 18 ∧ t = 1 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2_269


namespace NUMINAMATH_CALUDE_socorro_multiplication_time_l2_292

/-- The time spent on multiplication problems each day, given the total training time,
    number of training days, and daily time spent on division problems. -/
def time_on_multiplication (total_hours : ℕ) (days : ℕ) (division_minutes : ℕ) : ℕ :=
  ((total_hours * 60) - (days * division_minutes)) / days

/-- Theorem stating that Socorro spends 10 minutes each day on multiplication problems. -/
theorem socorro_multiplication_time :
  time_on_multiplication 5 10 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_socorro_multiplication_time_l2_292


namespace NUMINAMATH_CALUDE_interest_calculation_l2_208

/-- Represents the interest calculation problem -/
theorem interest_calculation (x y z : ℝ) 
  (h1 : x * y / 100 * 2 = 800)  -- Simple interest condition
  (h2 : x * ((1 + y / 100)^2 - 1) = 820)  -- Compound interest condition
  : x = 8000 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l2_208


namespace NUMINAMATH_CALUDE_trees_after_typhoon_l2_218

theorem trees_after_typhoon (initial_trees : ℕ) (dead_trees : ℕ) : 
  initial_trees = 20 → dead_trees = 16 → initial_trees - dead_trees = 4 := by
  sorry

end NUMINAMATH_CALUDE_trees_after_typhoon_l2_218


namespace NUMINAMATH_CALUDE_abc_inequalities_l2_202

theorem abc_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_prod : (a + 1) * (b + 1) * (c + 1) = 8) :
  a + b + c ≥ 3 ∧ a * b * c ≤ 1 ∧
  (a + b + c = 3 ∧ a * b * c = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_abc_inequalities_l2_202


namespace NUMINAMATH_CALUDE_original_number_proof_l2_280

theorem original_number_proof (r : ℝ) : 
  (1.20 * r - r) + (1.35 * r - r) - (r - 0.50 * r) = 110 → r = 2200 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2_280


namespace NUMINAMATH_CALUDE_root_properties_l2_214

theorem root_properties : 
  (∃ x : ℝ, x^3 = -9 ∧ x = -3) ∧ 
  (∀ y : ℝ, y^2 = 9 ↔ y = 3 ∨ y = -3) := by
  sorry

end NUMINAMATH_CALUDE_root_properties_l2_214


namespace NUMINAMATH_CALUDE_network_connections_l2_258

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (n * k) / 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_network_connections_l2_258


namespace NUMINAMATH_CALUDE_difference_of_squares_81_49_l2_236

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_81_49_l2_236


namespace NUMINAMATH_CALUDE_base_7_to_10_23456_l2_238

def base_7_to_10 (d₁ d₂ d₃ d₄ d₅ : ℕ) : ℕ :=
  d₁ * 7^4 + d₂ * 7^3 + d₃ * 7^2 + d₄ * 7^1 + d₅ * 7^0

theorem base_7_to_10_23456 :
  base_7_to_10 2 3 4 5 6 = 6068 := by sorry

end NUMINAMATH_CALUDE_base_7_to_10_23456_l2_238


namespace NUMINAMATH_CALUDE_tangent_and_normal_equations_l2_212

/-- The curve equation -/
def curve (x y : ℝ) : Prop := x^2 - 2*x*y + 3*y^2 - 2*y - 16 = 0

/-- The point on the curve -/
def point : ℝ × ℝ := (1, 3)

/-- Tangent line equation -/
def tangent_line (x y : ℝ) : Prop := 2*x - 7*y + 19 = 0

/-- Normal line equation -/
def normal_line (x y : ℝ) : Prop := 7*x + 2*y - 13 = 0

theorem tangent_and_normal_equations :
  curve point.1 point.2 →
  (∀ x y, tangent_line x y ↔ 
    (y - point.2) = (2/7) * (x - point.1)) ∧
  (∀ x y, normal_line x y ↔ 
    (y - point.2) = (-7/2) * (x - point.1)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_normal_equations_l2_212


namespace NUMINAMATH_CALUDE_age_difference_l2_286

/-- Given two people p and q, prove that p was half of q's age 6 years ago,
    given their current age ratio and sum. -/
theorem age_difference (p q : ℕ) : 
  (p : ℚ) / q = 3 / 4 →  -- Current age ratio
  p + q = 21 →           -- Sum of current ages
  ∃ (y : ℕ), p - y = (q - y) / 2 ∧ y = 6 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2_286


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l2_203

/-- A function f(x) = ax^4 + bx^2 + 5 where a and b are real constants -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + 5

/-- Theorem: If f(20) = 3, then f(20) + f(-20) = 6 -/
theorem f_sum_symmetric (a b : ℝ) (h : f a b 20 = 3) : f a b 20 + f a b (-20) = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l2_203


namespace NUMINAMATH_CALUDE_female_rainbow_count_l2_210

/-- Represents the number of trout in a fishery -/
structure Fishery where
  female_speckled : ℕ
  male_speckled : ℕ
  female_rainbow : ℕ
  male_rainbow : ℕ

/-- The conditions of the fishery problem -/
def fishery_conditions (f : Fishery) : Prop :=
  f.female_speckled + f.male_speckled = 645 ∧
  f.male_speckled = 2 * f.female_speckled + 45 ∧
  4 * f.male_rainbow = 3 * f.female_speckled ∧
  3 * (f.female_speckled + f.male_speckled + f.female_rainbow + f.male_rainbow) = 20 * f.male_rainbow

/-- The theorem stating that under the given conditions, there are 205 female rainbow trout -/
theorem female_rainbow_count (f : Fishery) :
  fishery_conditions f → f.female_rainbow = 205 := by
  sorry


end NUMINAMATH_CALUDE_female_rainbow_count_l2_210


namespace NUMINAMATH_CALUDE_mean_value_theorem_for_f_l2_299

-- Define the function f(x) = x² + 3
def f (x : ℝ) : ℝ := x^2 + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

theorem mean_value_theorem_for_f :
  ∃ c ∈ Set.Ioo (-1 : ℝ) 2,
    f 2 - f (-1) = f' c * (2 - (-1)) ∧
    c = 1 / 2 := by
  sorry

#check mean_value_theorem_for_f

end NUMINAMATH_CALUDE_mean_value_theorem_for_f_l2_299


namespace NUMINAMATH_CALUDE_triple_composition_even_l2_242

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem triple_composition_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
by
  sorry

end NUMINAMATH_CALUDE_triple_composition_even_l2_242


namespace NUMINAMATH_CALUDE_neighborhood_cable_cost_l2_284

/-- Calculate the total cost of power cable for a neighborhood --/
theorem neighborhood_cable_cost
  (east_west_streets : ℕ)
  (north_south_streets : ℕ)
  (east_west_length : ℝ)
  (north_south_length : ℝ)
  (cable_per_street_mile : ℝ)
  (cable_cost_per_mile : ℝ)
  (h1 : east_west_streets = 18)
  (h2 : north_south_streets = 10)
  (h3 : east_west_length = 2)
  (h4 : north_south_length = 4)
  (h5 : cable_per_street_mile = 5)
  (h6 : cable_cost_per_mile = 2000) :
  east_west_streets * east_west_length * cable_per_street_mile * cable_cost_per_mile +
  north_south_streets * north_south_length * cable_per_street_mile * cable_cost_per_mile =
  760000 :=
by sorry

end NUMINAMATH_CALUDE_neighborhood_cable_cost_l2_284


namespace NUMINAMATH_CALUDE_triangle_area_l2_244

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b^2 + c^2 = a^2 - b*c →
  (a * b * Real.cos C) = -4 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2_244


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2_209

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 10) : 
  min x y = 4 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2_209


namespace NUMINAMATH_CALUDE_grid_hole_properties_l2_200

/-- Represents a grid with a hole --/
structure GridWithHole where
  rows : ℕ
  cols : ℕ
  holeRows : ℕ
  holeCols : ℕ
  squareSideLength : ℝ

/-- Calculate the number of removed squares in the grid --/
def removedSquares (g : GridWithHole) : ℕ := 36

/-- Calculate the area of the hole in the grid --/
def holeArea (g : GridWithHole) : ℝ := 36

/-- Calculate the perimeter of the hole in the grid --/
def holePerimeter (g : GridWithHole) : ℝ := 42

/-- Theorem stating the properties of the grid with hole --/
theorem grid_hole_properties (g : GridWithHole) 
  (h1 : g.rows = 10) 
  (h2 : g.cols = 20) 
  (h3 : g.holeRows = 6) 
  (h4 : g.holeCols = 15) 
  (h5 : g.squareSideLength = 1) : 
  removedSquares g = 36 ∧ 
  holeArea g = 36 ∧ 
  holePerimeter g = 42 := by
  sorry

end NUMINAMATH_CALUDE_grid_hole_properties_l2_200


namespace NUMINAMATH_CALUDE_calculate_income_l2_213

/-- Given a person's income and expenditure ratio, and their savings, calculate their income. -/
theorem calculate_income (income expenditure savings : ℕ) : 
  income * 4 = expenditure * 5 →  -- income to expenditure ratio is 5:4
  income - expenditure = savings → -- savings definition
  savings = 4000 → -- given savings amount
  income = 20000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_income_l2_213


namespace NUMINAMATH_CALUDE_kitten_weight_l2_255

theorem kitten_weight (kitten smaller_dog larger_dog : ℝ) 
  (total_weight : kitten + smaller_dog + larger_dog = 36)
  (larger_comparison : kitten + larger_dog = 2 * smaller_dog)
  (smaller_comparison : kitten + smaller_dog = larger_dog) :
  kitten = 9 := by
sorry

end NUMINAMATH_CALUDE_kitten_weight_l2_255


namespace NUMINAMATH_CALUDE_will_buttons_count_l2_263

theorem will_buttons_count (mari_buttons : ℕ) (kendra_buttons : ℕ) (sue_buttons : ℕ) (will_buttons : ℕ) : 
  mari_buttons = 8 →
  kendra_buttons = 5 * mari_buttons + 4 →
  sue_buttons = kendra_buttons / 2 →
  will_buttons = 2 * (kendra_buttons + sue_buttons) →
  will_buttons = 132 :=
by
  sorry

end NUMINAMATH_CALUDE_will_buttons_count_l2_263


namespace NUMINAMATH_CALUDE_minimize_y_l2_289

variable (a b : ℝ)
def y (x : ℝ) := 3 * (x - a)^2 + (x - b)^2

theorem minimize_y :
  ∃ (x : ℝ), ∀ (z : ℝ), y a b x ≤ y a b z ∧ x = (3 * a + b) / 4 := by
  sorry

end NUMINAMATH_CALUDE_minimize_y_l2_289


namespace NUMINAMATH_CALUDE_greatest_4digit_base9_divisible_by_7_l2_297

/-- Converts a base 9 number to decimal --/
def base9ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 9 --/
def decimalToBase9 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 4-digit base 9 number --/
def is4DigitBase9 (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 8888

theorem greatest_4digit_base9_divisible_by_7 :
  ∃ (n : ℕ), is4DigitBase9 n ∧ 
             base9ToDecimal n % 7 = 0 ∧
             ∀ (m : ℕ), is4DigitBase9 m ∧ base9ToDecimal m % 7 = 0 → m ≤ n ∧
             n = 9000 :=
sorry

end NUMINAMATH_CALUDE_greatest_4digit_base9_divisible_by_7_l2_297


namespace NUMINAMATH_CALUDE_solution_concentration_l2_228

def concentrate_solution (initial_volume : ℝ) (final_concentration : ℝ) (water_removed : ℝ) : Prop :=
  ∃ (initial_concentration : ℝ),
    0 < initial_concentration ∧
    initial_concentration < final_concentration ∧
    final_concentration < 1 ∧
    initial_volume > water_removed ∧
    -- The actual concentration calculation would go here, but we lack the initial concentration
    True

theorem solution_concentration :
  concentrate_solution 24 0.6 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_concentration_l2_228


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l2_234

def initial_amount : ℚ := 90

def sandwich_fraction : ℚ := 1/5
def museum_fraction : ℚ := 1/6
def book_fraction : ℚ := 1/2

def remaining_amount : ℚ := initial_amount - (sandwich_fraction * initial_amount + museum_fraction * initial_amount + book_fraction * initial_amount)

theorem jennifer_remaining_money :
  remaining_amount = 12 := by sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l2_234


namespace NUMINAMATH_CALUDE_walters_age_calculation_l2_223

/-- Walter's age at the end of 2000 -/
def walters_age_2000 : ℝ := 37.5

/-- Walter's grandmother's age at the end of 2000 -/
def grandmothers_age_2000 : ℝ := 3 * walters_age_2000

/-- The sum of Walter's and his grandmother's birth years -/
def birth_years_sum : ℕ := 3850

/-- Walter's age at the end of 2010 -/
def walters_age_2010 : ℝ := walters_age_2000 + 10

theorem walters_age_calculation :
  (2000 - walters_age_2000) + (2000 - grandmothers_age_2000) = birth_years_sum ∧
  walters_age_2010 = 47.5 := by
  sorry

#eval walters_age_2010

end NUMINAMATH_CALUDE_walters_age_calculation_l2_223


namespace NUMINAMATH_CALUDE_garage_sale_ratio_l2_216

theorem garage_sale_ratio (treadmill_price chest_price tv_price total_sale : ℚ) : 
  treadmill_price = 100 →
  chest_price = treadmill_price / 2 →
  total_sale = 600 →
  total_sale = treadmill_price + chest_price + tv_price →
  tv_price / treadmill_price = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_ratio_l2_216


namespace NUMINAMATH_CALUDE_quadratic_ratio_l2_227

theorem quadratic_ratio (x : ℝ) : 
  ∃ (d e : ℝ), 
    (∀ x, x^2 + 900*x + 1800 = (x + d)^2 + e) ∧ 
    (e / d = -446) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l2_227
