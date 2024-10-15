import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_equality_l2635_263597

theorem polynomial_equality (a : ℝ) (h : 2 * a^2 - 3 * a - 5 = 0) :
  4 * a^4 - 12 * a^3 + 9 * a^2 - 10 = 15 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2635_263597


namespace NUMINAMATH_CALUDE_tangent_sum_difference_l2635_263531

theorem tangent_sum_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  Real.tan (α + π/4) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_difference_l2635_263531


namespace NUMINAMATH_CALUDE_age_difference_l2635_263510

theorem age_difference (son_age man_age : ℕ) : 
  son_age = 30 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2635_263510


namespace NUMINAMATH_CALUDE_min_abs_phi_l2635_263557

/-- Given a function y = 2sin(2x - φ) whose graph is symmetric about the point (4π/3, 0),
    the minimum value of |φ| is π/3 -/
theorem min_abs_phi (φ : ℝ) : 
  (∀ x : ℝ, 2 * Real.sin (2 * x - φ) = 2 * Real.sin (2 * (8 * π / 3 - x) - φ)) →
  ∃ k : ℤ, φ = 8 * π / 3 - k * π →
  |φ| ≥ π / 3 ∧ ∃ φ₀ : ℝ, |φ₀| = π / 3 ∧ 
    (∀ x : ℝ, 2 * Real.sin (2 * x - φ₀) = 2 * Real.sin (2 * (8 * π / 3 - x) - φ₀)) :=
by sorry

end NUMINAMATH_CALUDE_min_abs_phi_l2635_263557


namespace NUMINAMATH_CALUDE_correct_loan_amounts_l2635_263508

/-- Represents the loan amounts and interest rates for a company's two types of loans. -/
structure LoanInfo where
  typeA : ℝ  -- Amount of Type A loan in yuan
  typeB : ℝ  -- Amount of Type B loan in yuan
  rateA : ℝ  -- Annual interest rate for Type A loan
  rateB : ℝ  -- Annual interest rate for Type B loan

/-- Theorem stating the correct loan amounts given the problem conditions. -/
theorem correct_loan_amounts (loan : LoanInfo) : 
  loan.typeA = 200000 ∧ loan.typeB = 300000 ↔ 
  loan.typeA + loan.typeB = 500000 ∧ 
  loan.rateA * loan.typeA + loan.rateB * loan.typeB = 44000 ∧
  loan.rateA = 0.1 ∧ 
  loan.rateB = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_correct_loan_amounts_l2635_263508


namespace NUMINAMATH_CALUDE_negation_of_implication_l2635_263525

theorem negation_of_implication (a b : ℝ) :
  ¬(∀ x : ℝ, x < a → x < b) ↔ (∃ x : ℝ, x ≥ a ∧ x ≥ b) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2635_263525


namespace NUMINAMATH_CALUDE_sphere_radius_touching_cones_l2635_263529

/-- The radius of a sphere touching three cones and a table -/
theorem sphere_radius_touching_cones (r₁ r₂ r₃ : ℝ) (α β γ : ℝ) : 
  r₁ = 1 → 
  r₂ = 12 → 
  r₃ = 12 → 
  α = -4 * Real.arctan (1/3) → 
  β = 4 * Real.arctan (2/3) → 
  γ = 4 * Real.arctan (2/3) → 
  ∃ R : ℝ, R = 40/21 ∧ 
    (∀ x y z : ℝ, 
      x^2 + y^2 + z^2 = R^2 → 
      (∃ t : ℝ, t ≥ 0 ∧ 
        ((x - r₁)^2 + y^2 = (t * Real.tan (α/2))^2 ∧ z = t) ∨
        ((x - (r₁ + r₂))^2 + y^2 = (t * Real.tan (β/2))^2 ∧ z = t) ∨
        (x^2 + (y - (r₂ + r₃))^2 = (t * Real.tan (γ/2))^2 ∧ z = t) ∨
        z = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_touching_cones_l2635_263529


namespace NUMINAMATH_CALUDE_max_value_of_z_l2635_263582

/-- Given real numbers x and y satisfying the conditions,
    prove that the maximum value of z = 2x - y is 5 -/
theorem max_value_of_z (x y : ℝ) 
  (h1 : x - 2*y + 2 ≥ 0) 
  (h2 : x + y ≤ 1) 
  (h3 : y + 1 ≥ 0) : 
  ∃ (z : ℝ), z = 2*x - y ∧ z ≤ 5 ∧ ∀ (w : ℝ), w = 2*x - y → w ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2635_263582


namespace NUMINAMATH_CALUDE_ten_women_circular_reseating_l2635_263561

/-- The number of ways n women can be reseated in a circular arrangement,
    where each woman sits in her original seat or a seat adjacent to it. -/
def C : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => C (n + 1) + C n

theorem ten_women_circular_reseating : C 10 = 89 := by
  sorry

end NUMINAMATH_CALUDE_ten_women_circular_reseating_l2635_263561


namespace NUMINAMATH_CALUDE_alices_journey_time_l2635_263598

/-- Represents the problem of Alice's journey to the library -/
theorem alices_journey_time :
  ∀ (d : ℝ) (r_w : ℝ),
    r_w > 0 →
    (3/4 * d) / r_w = 9 →
    (1/4 * d) / (4 * r_w) + 9 = 9.75 :=
by sorry

end NUMINAMATH_CALUDE_alices_journey_time_l2635_263598


namespace NUMINAMATH_CALUDE_min_value_reciprocal_product_l2635_263594

theorem min_value_reciprocal_product (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_mean : a + 2*b = 6) : 
  (∀ x y, x > 0 → y > 0 → x + 2*y = 6 → (1 / (a * b)) ≤ (1 / (x * y))) → 
  (1 / (a * b)) = 2/9 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_product_l2635_263594


namespace NUMINAMATH_CALUDE_K33_not_planar_l2635_263503

/-- A bipartite graph with two sets of three vertices each --/
structure BipartiteGraph :=
  (left : Finset ℕ)
  (right : Finset ℕ)
  (edges : Set (ℕ × ℕ))

/-- The K₃,₃ graph --/
def K33 : BipartiteGraph :=
  { left := {1, 2, 3},
    right := {4, 5, 6},
    edges := {(1,4), (1,5), (1,6), (2,4), (2,5), (2,6), (3,4), (3,5), (3,6)} }

/-- A graph is planar if it can be drawn on a plane without edge crossings --/
def isPlanar (G : BipartiteGraph) : Prop := sorry

/-- Theorem: K₃,₃ is not planar --/
theorem K33_not_planar : ¬ isPlanar K33 := by
  sorry

end NUMINAMATH_CALUDE_K33_not_planar_l2635_263503


namespace NUMINAMATH_CALUDE_lucy_age_theorem_l2635_263580

/-- Lucy's age at the end of 2000 -/
def lucy_age_2000 : ℝ := 27.5

/-- Lucy's grandfather's age at the end of 2000 -/
def grandfather_age_2000 : ℝ := 3 * lucy_age_2000

/-- The sum of Lucy's and her grandfather's birth years -/
def birth_years_sum : ℝ := 3890

/-- Lucy's age at the end of 2010 -/
def lucy_age_2010 : ℝ := lucy_age_2000 + 10

theorem lucy_age_theorem :
  lucy_age_2000 = (grandfather_age_2000 / 3) ∧
  (2000 - lucy_age_2000) + (2000 - grandfather_age_2000) = birth_years_sum ∧
  lucy_age_2010 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_lucy_age_theorem_l2635_263580


namespace NUMINAMATH_CALUDE_savings_calculation_l2635_263567

/-- Calculates a person's savings given their income and income-to-expenditure ratio -/
def calculate_savings (income : ℚ) (income_ratio : ℚ) (expenditure_ratio : ℚ) : ℚ :=
  income - (income * expenditure_ratio / income_ratio)

/-- Proves that for a given income and income-to-expenditure ratio, the savings are as calculated -/
theorem savings_calculation (income : ℚ) (income_ratio : ℚ) (expenditure_ratio : ℚ) :
  income = 20000 ∧ income_ratio = 4 ∧ expenditure_ratio = 3 →
  calculate_savings income income_ratio expenditure_ratio = 5000 :=
by sorry

end NUMINAMATH_CALUDE_savings_calculation_l2635_263567


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2635_263548

/-- The minimum value of 1/a + 4/b given the conditions -/
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, a * x - b * y + 2 = 0 → x^2 + y^2 + 2*x - 2*y = 0) → 
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (a * A.1 - b * A.2 + 2 = 0) ∧ (A.1^2 + A.2^2 + 2*A.1 - 2*A.2 = 0) ∧
    (a * B.1 - b * B.2 + 2 = 0) ∧ (B.1^2 + B.2^2 + 2*B.1 - 2*B.2 = 0) ∧
    (∀ C D : ℝ × ℝ, C ≠ D → 
      (a * C.1 - b * C.2 + 2 = 0) → (C.1^2 + C.2^2 + 2*C.1 - 2*C.2 = 0) →
      (a * D.1 - b * D.2 + 2 = 0) → (D.1^2 + D.2^2 + 2*D.1 - 2*D.2 = 0) →
      (A.1 - B.1)^2 + (A.2 - B.2)^2 ≥ (C.1 - D.1)^2 + (C.2 - D.2)^2)) →
  (1/a + 4/b) ≥ 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2635_263548


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2635_263516

/-- 
Given an arithmetic sequence where the sum of the third and fifth terms is 10,
prove that the fourth term is 5.
-/
theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℝ) -- a is the arithmetic sequence
  (h : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) -- definition of arithmetic sequence
  (sum_condition : a 3 + a 5 = 10) -- sum of third and fifth terms is 10
  : a 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2635_263516


namespace NUMINAMATH_CALUDE_snail_distance_is_29_l2635_263585

def snail_path : List ℤ := [3, -5, 8, 0]

def distance (a b : ℤ) : ℤ := Int.natAbs (b - a)

def total_distance (path : List ℤ) : ℤ :=
  (List.zip path path.tail).foldl (fun acc (a, b) => acc + distance a b) 0

theorem snail_distance_is_29 : total_distance snail_path = 29 := by
  sorry

end NUMINAMATH_CALUDE_snail_distance_is_29_l2635_263585


namespace NUMINAMATH_CALUDE_red_jelly_beans_l2635_263575

/-- The number of red jelly beans in a bag, given the following conditions:
  1. It takes three bags of jelly beans to fill the fishbowl.
  2. Each bag has a similar distribution of colors.
  3. One bag contains: 13 black, 36 green, 28 purple, 32 yellow, and 18 white jelly beans.
  4. The total number of red and white jelly beans in the fishbowl is 126. -/
theorem red_jelly_beans (black green purple yellow white : ℕ)
  (h1 : black = 13)
  (h2 : green = 36)
  (h3 : purple = 28)
  (h4 : yellow = 32)
  (h5 : white = 18)
  (h6 : (red + white) * 3 = 126) :
  red = 24 :=
sorry

end NUMINAMATH_CALUDE_red_jelly_beans_l2635_263575


namespace NUMINAMATH_CALUDE_gloria_has_23_maple_trees_l2635_263541

/-- Represents the problem of calculating Gloria's maple trees --/
def GloriasMapleTrees (cabin_price cash_on_hand leftover cypress_count pine_count cypress_price pine_price maple_price : ℕ) : Prop :=
  let total_needed := cabin_price - cash_on_hand
  let cypress_income := cypress_count * cypress_price
  let pine_income := pine_count * pine_price
  let maple_income := total_needed - cypress_income - pine_income
  ∃ (maple_count : ℕ), 
    maple_count * maple_price = maple_income ∧
    maple_count * maple_price + cypress_income + pine_income + cash_on_hand = cabin_price + leftover

theorem gloria_has_23_maple_trees : 
  GloriasMapleTrees 129000 150 350 20 600 100 200 300 → 
  ∃ (maple_count : ℕ), maple_count = 23 :=
sorry

end NUMINAMATH_CALUDE_gloria_has_23_maple_trees_l2635_263541


namespace NUMINAMATH_CALUDE_ribbon_fraction_per_box_l2635_263570

theorem ribbon_fraction_per_box 
  (total_fraction : ℚ) 
  (num_boxes : ℕ) 
  (h1 : total_fraction = 5/12) 
  (h2 : num_boxes = 5) : 
  total_fraction / num_boxes = 1/12 := by
sorry

end NUMINAMATH_CALUDE_ribbon_fraction_per_box_l2635_263570


namespace NUMINAMATH_CALUDE_books_from_second_shop_l2635_263536

theorem books_from_second_shop 
  (first_shop_books : ℕ)
  (first_shop_cost : ℕ)
  (second_shop_cost : ℕ)
  (average_price : ℕ)
  (h1 : first_shop_books = 55)
  (h2 : first_shop_cost = 1500)
  (h3 : second_shop_cost = 340)
  (h4 : average_price = 16) :
  ∃ (second_shop_books : ℕ),
    (first_shop_cost + second_shop_cost) = 
    average_price * (first_shop_books + second_shop_books) ∧
    second_shop_books = 60 := by
  sorry

end NUMINAMATH_CALUDE_books_from_second_shop_l2635_263536


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l2635_263588

/-- Represents the number of units produced by each workshop -/
structure WorkshopProduction where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the sampling information -/
structure SamplingInfo where
  total_sample : ℕ
  workshop_b_sample : ℕ

/-- Theorem stating the correct sample size for the given scenario -/
theorem stratified_sampling_size 
  (prod : WorkshopProduction)
  (sample : SamplingInfo)
  (h1 : prod.a = 96)
  (h2 : prod.b = 84)
  (h3 : prod.c = 60)
  (h4 : sample.workshop_b_sample = 7)
  (h5 : sample.workshop_b_sample / sample.total_sample = prod.b / (prod.a + prod.b + prod.c)) :
  sample.total_sample = 70 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_size_l2635_263588


namespace NUMINAMATH_CALUDE_triangle_area_and_angle_B_l2635_263549

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_area_and_angle_B 
  (a b c : ℝ) 
  (h_triangle : Triangle a b c)
  (h_eq : b^2 = c^2 + a^2 - Real.sqrt 2 * a * c)
  (h_a : a = Real.sqrt 2)
  (h_cos_A : Real.cos A = 4/5)
  : Real.cos B = Real.sqrt 2 / 2 ∧ 
    ∃ (S : ℝ), S = 7/6 ∧ S = 1/2 * a * b * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_and_angle_B_l2635_263549


namespace NUMINAMATH_CALUDE_largest_among_four_l2635_263501

theorem largest_among_four : ∀ (a b c d : ℝ), 
  a = 0 → b = -1 → c = -2 → d = Real.sqrt 3 →
  d = max a (max b (max c d)) :=
by sorry

end NUMINAMATH_CALUDE_largest_among_four_l2635_263501


namespace NUMINAMATH_CALUDE_monday_rainfall_calculation_l2635_263514

def total_rainfall : ℝ := 0.67
def tuesday_rainfall : ℝ := 0.42
def wednesday_rainfall : ℝ := 0.08

theorem monday_rainfall_calculation :
  ∃ (monday_rainfall : ℝ),
    monday_rainfall + tuesday_rainfall + wednesday_rainfall = total_rainfall ∧
    monday_rainfall = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_monday_rainfall_calculation_l2635_263514


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2635_263563

theorem polynomial_simplification (x : ℝ) : 
  5 - 3*x - 7*x^2 + 3 + 12*x - 9*x^2 - 8 + 15*x + 21*x^2 = 5*x^2 + 24*x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2635_263563


namespace NUMINAMATH_CALUDE_rebecca_marbles_l2635_263515

theorem rebecca_marbles (group_size : ℕ) (num_groups : ℕ) (total_marbles : ℕ) : 
  group_size = 4 → num_groups = 5 → total_marbles = group_size * num_groups → total_marbles = 20 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_marbles_l2635_263515


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2635_263568

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - 2*x + a + 1 = 0 ∧ y^2 - 2*y + a + 1 = 0) ↔ a < -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2635_263568


namespace NUMINAMATH_CALUDE_meal_cost_proof_l2635_263560

/-- Given the cost of two different meal combinations, 
    prove the cost of a single sandwich, coffee, and pie. -/
theorem meal_cost_proof (sandwich_cost coffee_cost pie_cost : ℚ) : 
  5 * sandwich_cost + 8 * coffee_cost + 2 * pie_cost = (5.40 : ℚ) →
  3 * sandwich_cost + 11 * coffee_cost + 2 * pie_cost = (4.95 : ℚ) →
  sandwich_cost + coffee_cost + pie_cost = (1.55 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_proof_l2635_263560


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_half_l2635_263552

/-- Represents a cube composed of smaller cubes -/
structure CompositeCube where
  smallCubeCount : ℕ
  smallCubeSideLength : ℝ
  whiteCubeCount : ℕ
  blackCubeCount : ℕ
  redCubeCount : ℕ

/-- The fraction of the surface area that is white -/
def whiteSurfaceFraction (c : CompositeCube) : ℚ :=
  sorry

/-- Our specific cube configuration -/
def ourCube : CompositeCube :=
  { smallCubeCount := 64
  , smallCubeSideLength := 1
  , whiteCubeCount := 36
  , blackCubeCount := 8
  , redCubeCount := 20 }

theorem white_surface_fraction_is_half :
  whiteSurfaceFraction ourCube = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_half_l2635_263552


namespace NUMINAMATH_CALUDE_sin_cos_three_eighths_pi_l2635_263512

theorem sin_cos_three_eighths_pi (π : Real) :
  Real.sin (3 * π / 8) * Real.cos (π / 8) = (2 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_three_eighths_pi_l2635_263512


namespace NUMINAMATH_CALUDE_decimal_34_to_binary_binary_to_decimal_34_l2635_263524

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_34_to_binary :
  toBinary 34 = [false, true, false, false, false, true] :=
by sorry

theorem binary_to_decimal_34 :
  fromBinary [false, true, false, false, false, true] = 34 :=
by sorry

end NUMINAMATH_CALUDE_decimal_34_to_binary_binary_to_decimal_34_l2635_263524


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2635_263583

theorem inequality_solution_set : 
  {x : ℝ | (x - 1) * (x + 2) < 0} = Set.Ioo (-2 : ℝ) (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2635_263583


namespace NUMINAMATH_CALUDE_hyperbola_equation_proof_l2635_263554

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- The eccentricity of the hyperbola -/
  e : ℝ
  /-- The distance from the focus to the asymptote -/
  d : ℝ
  /-- The hyperbola is centered at the origin -/
  center_origin : True
  /-- The foci are on the x-axis -/
  foci_on_x_axis : True
  /-- The eccentricity is √6/2 -/
  e_value : e = Real.sqrt 6 / 2
  /-- The distance from the focus to the asymptote is 1 -/
  d_value : d = 1

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 = 1

/-- Theorem stating that the given hyperbola has the specified equation -/
theorem hyperbola_equation_proof (h : Hyperbola) :
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 / 2 - y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_proof_l2635_263554


namespace NUMINAMATH_CALUDE_bird_feeder_theft_ratio_l2635_263571

/-- Given a bird feeder with the following properties:
  - Holds 2 cups of birdseed
  - Each cup of birdseed can feed 14 birds
  - The feeder actually feeds 21 birds weekly
  Prove that the ratio of birdseed stolen to total birdseed is 1:4 -/
theorem bird_feeder_theft_ratio 
  (total_cups : ℚ) 
  (birds_per_cup : ℕ) 
  (birds_fed : ℕ) : 
  total_cups = 2 →
  birds_per_cup = 14 →
  birds_fed = 21 →
  (total_cups - (birds_fed : ℚ) / (birds_per_cup : ℚ)) / total_cups = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_bird_feeder_theft_ratio_l2635_263571


namespace NUMINAMATH_CALUDE_binomial_26_6_l2635_263518

theorem binomial_26_6 (h1 : Nat.choose 24 5 = 42504) 
                      (h2 : Nat.choose 24 6 = 134596) 
                      (h3 : Nat.choose 24 7 = 346104) : 
  Nat.choose 26 6 = 657800 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l2635_263518


namespace NUMINAMATH_CALUDE_inequality_proof_l2635_263547

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c ≥ a * b * c) : 
  (2/a + 3/b + 6/c ≥ 6 ∧ 2/b + 3/c + 6/a ≥ 6) ∨
  (2/b + 3/c + 6/a ≥ 6 ∧ 2/c + 3/a + 6/b ≥ 6) ∨
  (2/c + 3/a + 6/b ≥ 6 ∧ 2/a + 3/b + 6/c ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2635_263547


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2635_263555

-- Define sets A and B
def A : Set ℝ := {y | ∃ x, y = 2^x}
def B : Set ℝ := {y | ∃ x, y = -x^2 + 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {y | 0 < y ∧ y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2635_263555


namespace NUMINAMATH_CALUDE_percentage_of_population_l2635_263577

theorem percentage_of_population (total_population : ℕ) (part_population : ℕ) :
  total_population = 28800 →
  part_population = 23040 →
  (part_population : ℚ) / (total_population : ℚ) * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_population_l2635_263577


namespace NUMINAMATH_CALUDE_det_scale_by_three_l2635_263564

theorem det_scale_by_three (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 7 →
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = 63 := by
  sorry

end NUMINAMATH_CALUDE_det_scale_by_three_l2635_263564


namespace NUMINAMATH_CALUDE_difference_of_squares_l2635_263593

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2635_263593


namespace NUMINAMATH_CALUDE_integer_count_inequality_l2635_263513

theorem integer_count_inequality (x : ℤ) : 
  (Finset.filter (fun i => (i - 2)^2 ≤ 4) (Finset.range 10)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_integer_count_inequality_l2635_263513


namespace NUMINAMATH_CALUDE_sock_pair_count_l2635_263544

def white_socks : ℕ := 5
def brown_socks : ℕ := 3
def blue_socks : ℕ := 2
def black_socks : ℕ := 2

def total_socks : ℕ := white_socks + brown_socks + blue_socks + black_socks

def choose (n k : ℕ) : ℕ := Nat.choose n k

def same_color_pairs : ℕ :=
  choose white_socks 2 + choose brown_socks 2 + choose blue_socks 2 + choose black_socks 2

theorem sock_pair_count : same_color_pairs = 15 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l2635_263544


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruence_l2635_263543

theorem smallest_four_digit_congruence : ∃ (x : ℕ), 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (5 * y ≡ 15 [ZMOD 20] ∧
     3 * y + 7 ≡ 10 [ZMOD 8] ∧
     -3 * y + 2 ≡ 2 * y [ZMOD 35]) →
    x ≤ y) ∧
  (5 * x ≡ 15 [ZMOD 20] ∧
   3 * x + 7 ≡ 10 [ZMOD 8] ∧
   -3 * x + 2 ≡ 2 * x [ZMOD 35]) ∧
  x = 1009 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruence_l2635_263543


namespace NUMINAMATH_CALUDE_part_one_part_two_l2635_263542

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + 3| - m + 1

-- Part I
theorem part_one (m : ℝ) (h1 : m > 0) 
  (h2 : Set.Iic (-2) ∪ Set.Ici 2 = {x | f m (x - 3) ≥ 0}) : 
  m = 3 := by sorry

-- Part II
theorem part_two : 
  {t : ℝ | ∃ x, |x + 3| - 2 ≥ |2*x - 1| - t^2 + 5/2*t} = 
  Set.Iic 1 ∪ Set.Ici (3/2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2635_263542


namespace NUMINAMATH_CALUDE_first_four_digits_1973_l2635_263505

theorem first_four_digits_1973 (n : ℕ) (h : ∀ k : ℕ, n ≠ 10^k) :
  ∃ j k : ℕ, j > 0 ∧ k > 0 ∧ 1973 ≤ (n^j : ℝ) / (10^k : ℝ) ∧ (n^j : ℝ) / (10^k : ℝ) < 1974 :=
sorry

end NUMINAMATH_CALUDE_first_four_digits_1973_l2635_263505


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l2635_263576

def g (x : ℝ) : ℝ := -3 * x^3 + 50 * x^2 - 4 * x + 10

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) := by
sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l2635_263576


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2635_263562

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    center = (-1, 0) ∧
    radius = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l2635_263562


namespace NUMINAMATH_CALUDE_two_identical_squares_exist_l2635_263572

-- Define the type for the table entries
inductive Entry
| Zero
| One

-- Define the 5x5 table
def Table := Fin 5 → Fin 5 → Entry

-- Define the property of having ones in top-left and bottom-right corners, and zeros in the other corners
def CornerCondition (t : Table) : Prop :=
  t 0 0 = Entry.One ∧
  t 4 4 = Entry.One ∧
  t 0 4 = Entry.Zero ∧
  t 4 0 = Entry.Zero

-- Define a 2x2 square in the table
def Square (t : Table) (i j : Fin 4) : Fin 2 → Fin 2 → Entry :=
  fun x y => t (i + x) (j + y)

-- Define when two squares are equal
def SquaresEqual (s1 s2 : Fin 2 → Fin 2 → Entry) : Prop :=
  ∀ (x y : Fin 2), s1 x y = s2 x y

-- The main theorem
theorem two_identical_squares_exist (t : Table) (h : CornerCondition t) :
  ∃ (i1 j1 i2 j2 : Fin 4), (i1, j1) ≠ (i2, j2) ∧
    SquaresEqual (Square t i1 j1) (Square t i2 j2) := by
  sorry

end NUMINAMATH_CALUDE_two_identical_squares_exist_l2635_263572


namespace NUMINAMATH_CALUDE_right_triangle_properties_l2635_263506

-- Define a right-angled triangle ABC
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  h_right_angle : angleA + angleB = π / 2
  h_sides : c^2 = a^2 + b^2
  h_not_equal : a ≠ b

-- State the theorem
theorem right_triangle_properties (t : RightTriangle) :
  (Real.tan t.angleA * Real.tan t.angleB ≠ 1) ∧
  (Real.sin t.angleA = t.a / t.c) ∧
  (t.c^2 - t.a^2 = t.b^2) ∧
  (t.c = t.b / Real.cos t.angleA) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l2635_263506


namespace NUMINAMATH_CALUDE_remainder_h_x10_div_h_x_l2635_263556

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Define the theorem
theorem remainder_h_x10_div_h_x :
  ∃ (q : ℝ → ℝ), h (x^10) = h x * q x + 6 :=
sorry

end NUMINAMATH_CALUDE_remainder_h_x10_div_h_x_l2635_263556


namespace NUMINAMATH_CALUDE_right_triangle_area_l2635_263520

theorem right_triangle_area (a b : ℝ) (h1 : a^2 = 64) (h2 : b^2 = 81) :
  (1/2) * a * b = 36 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2635_263520


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2635_263533

/-- Calculates the length of a train given its speed and the time it takes to cross a bridge of known length. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_ms := train_speed * 1000 / 3600
  let total_distance := speed_ms * crossing_time
  total_distance - bridge_length

/-- Proves that a train traveling at 40 km/h that crosses a 300-meter bridge in 45 seconds has a length of approximately 199.95 meters. -/
theorem train_length_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_length 40 300 45 - 199.95| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2635_263533


namespace NUMINAMATH_CALUDE_inheritance_satisfies_tax_equation_l2635_263558

/-- Represents the inheritance amount in dollars -/
def inheritance : ℝ := sorry

/-- The total tax paid is $15000 -/
def total_tax : ℝ := 15000

/-- Federal tax rate is 25% -/
def federal_tax_rate : ℝ := 0.25

/-- State tax rate is 15% -/
def state_tax_rate : ℝ := 0.15

/-- Theorem stating that the inheritance satisfies the tax equation -/
theorem inheritance_satisfies_tax_equation : 
  federal_tax_rate * inheritance + state_tax_rate * (1 - federal_tax_rate) * inheritance = total_tax := by
  sorry

end NUMINAMATH_CALUDE_inheritance_satisfies_tax_equation_l2635_263558


namespace NUMINAMATH_CALUDE_wall_width_correct_l2635_263519

/-- Represents the dimensions and properties of a wall -/
structure Wall where
  width : ℝ
  height : ℝ
  length : ℝ
  volume : ℝ

/-- The width of the wall given the conditions -/
def wall_width (w : Wall) : ℝ :=
  (384 : ℝ) ^ (1/3)

/-- Theorem stating that the calculated width satisfies the given conditions -/
theorem wall_width_correct (w : Wall) 
  (h_height : w.height = 6 * w.width)
  (h_length : w.length = 7 * w.height)
  (h_volume : w.volume = 16128) : 
  w.width = wall_width w := by
  sorry

#eval wall_width { width := 0, height := 0, length := 0, volume := 16128 }

end NUMINAMATH_CALUDE_wall_width_correct_l2635_263519


namespace NUMINAMATH_CALUDE_opposite_of_one_l2635_263550

theorem opposite_of_one : ∃ x : ℤ, x + 1 = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_l2635_263550


namespace NUMINAMATH_CALUDE_latoya_phone_card_initial_amount_l2635_263569

/-- The initial amount paid for a prepaid phone card -/
def initial_amount (call_cost_per_minute : ℚ) (call_duration : ℕ) (remaining_credit : ℚ) : ℚ :=
  call_cost_per_minute * call_duration + remaining_credit

/-- Theorem: The initial amount paid for Latoya's phone card is $30.00 -/
theorem latoya_phone_card_initial_amount :
  initial_amount (16 / 100) 22 26.48 = 30 :=
by sorry

end NUMINAMATH_CALUDE_latoya_phone_card_initial_amount_l2635_263569


namespace NUMINAMATH_CALUDE_intersection_point_first_quadrant_l2635_263509

-- Define the quadratic and linear functions
def f (x : ℝ) : ℝ := x^2 - x - 5
def g (x : ℝ) : ℝ := 2*x - 1

-- Define the first quadrant
def first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

-- Theorem statement
theorem intersection_point_first_quadrant :
  ∃! p : ℝ × ℝ, first_quadrant p ∧ f p.1 = g p.1 ∧ f p.1 = p.2 ∧ p = (4, 7) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_first_quadrant_l2635_263509


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l2635_263502

/-- Given a train of length 2000 m that crosses a tree in 200 sec,
    the time it takes to pass a platform of length 2500 m is 450 sec. -/
theorem train_platform_crossing_time :
  ∀ (train_length platform_length tree_crossing_time : ℝ),
    train_length = 2000 →
    platform_length = 2500 →
    tree_crossing_time = 200 →
    (train_length + platform_length) / (train_length / tree_crossing_time) = 450 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l2635_263502


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2635_263539

theorem quadratic_real_roots (a b c : ℝ) (ha : a ≠ 0) (hac : a * c < 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2635_263539


namespace NUMINAMATH_CALUDE_bus_students_count_l2635_263579

/-- The number of students on the left side of the bus -/
def left_students : ℕ := 36

/-- The number of students on the right side of the bus -/
def right_students : ℕ := 27

/-- The total number of students on the bus -/
def total_students : ℕ := left_students + right_students

/-- Theorem: The total number of students on the bus is 63 -/
theorem bus_students_count : total_students = 63 := by
  sorry

end NUMINAMATH_CALUDE_bus_students_count_l2635_263579


namespace NUMINAMATH_CALUDE_circle_radius_from_diameter_l2635_263591

theorem circle_radius_from_diameter (diameter : ℝ) (radius : ℝ) :
  diameter = 14 → radius = diameter / 2 → radius = 7 := by sorry

end NUMINAMATH_CALUDE_circle_radius_from_diameter_l2635_263591


namespace NUMINAMATH_CALUDE_rhombus_longest_diagonal_l2635_263511

/-- Given a rhombus with area 150 square units and diagonals in the ratio 4:3,
    prove that the length of the longest diagonal is 20 units. -/
theorem rhombus_longest_diagonal (area : ℝ) (d₁ d₂ : ℝ) : 
  area = 150 →
  d₁ / d₂ = 4 / 3 →
  area = (1 / 2) * d₁ * d₂ →
  d₁ > d₂ →
  d₁ = 20 := by
sorry

end NUMINAMATH_CALUDE_rhombus_longest_diagonal_l2635_263511


namespace NUMINAMATH_CALUDE_beads_per_necklace_l2635_263578

theorem beads_per_necklace (total_beads : ℕ) (num_necklaces : ℕ) 
  (h1 : total_beads = 28) (h2 : num_necklaces = 4) :
  total_beads / num_necklaces = 7 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l2635_263578


namespace NUMINAMATH_CALUDE_toy_price_calculation_l2635_263538

theorem toy_price_calculation (toy_price : ℝ) : 
  (3 * toy_price + 2 * 5 + 5 * 6 = 70) → toy_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_toy_price_calculation_l2635_263538


namespace NUMINAMATH_CALUDE_parabola_directrix_l2635_263527

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := x^2 + 12*y = 0

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop := y = 3

/-- Theorem: The directrix of the parabola x^2 + 12y = 0 is y = 3 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → directrix_equation y :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2635_263527


namespace NUMINAMATH_CALUDE_thomas_needs_2000_more_l2635_263565

/-- Thomas's savings scenario over two years -/
structure SavingsScenario where
  allowance_per_week : ℕ
  weeks_in_year : ℕ
  hourly_wage : ℕ
  hours_per_week : ℕ
  car_cost : ℕ
  weekly_expenses : ℕ

/-- Calculate the amount Thomas needs to save more -/
def amount_needed_more (s : SavingsScenario) : ℕ :=
  let first_year_savings := s.allowance_per_week * s.weeks_in_year
  let second_year_earnings := s.hourly_wage * s.hours_per_week * s.weeks_in_year
  let total_earnings := first_year_savings + second_year_earnings
  let total_expenses := s.weekly_expenses * (2 * s.weeks_in_year)
  let net_savings := total_earnings - total_expenses
  s.car_cost - net_savings

/-- Thomas's specific savings scenario -/
def thomas_scenario : SavingsScenario :=
  { allowance_per_week := 50
  , weeks_in_year := 52
  , hourly_wage := 9
  , hours_per_week := 30
  , car_cost := 15000
  , weekly_expenses := 35 }

/-- Theorem stating that Thomas needs $2000 more to buy the car -/
theorem thomas_needs_2000_more :
  amount_needed_more thomas_scenario = 2000 := by sorry

end NUMINAMATH_CALUDE_thomas_needs_2000_more_l2635_263565


namespace NUMINAMATH_CALUDE_square_plus_self_divisible_by_two_l2635_263566

theorem square_plus_self_divisible_by_two (a : ℤ) : 
  ∃ k : ℤ, a^2 + a = 2 * k :=
by
  sorry

end NUMINAMATH_CALUDE_square_plus_self_divisible_by_two_l2635_263566


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2635_263537

theorem last_two_digits_product (n : ℤ) : 
  (n % 100 % 4 = 0) → 
  ((n % 100) / 10 + n % 10 = 13) → 
  ((n % 100) / 10 * (n % 10) = 42) := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2635_263537


namespace NUMINAMATH_CALUDE_set_problem_l2635_263523

def I (x : ℝ) : Set ℝ := {2, 3, x^2 + 2*x - 3}
def A : Set ℝ := {5}

theorem set_problem (x y : ℝ) (C : Set ℝ) : 
  C ⊆ I x → C \ A = {2, y} → 
  ((x = -4 ∧ y = 3) ∨ (x = 2 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_set_problem_l2635_263523


namespace NUMINAMATH_CALUDE_largest_x_satisfying_equation_l2635_263592

theorem largest_x_satisfying_equation : 
  ∃ (x : ℝ), ∀ (y : ℝ), 
    (|y^2 - 11*y + 24| + |2*y^2 + 6*y - 56| = |y^2 + 17*y - 80|) → 
    y ≤ x ∧ 
    |x^2 - 11*x + 24| + |2*x^2 + 6*x - 56| = |x^2 + 17*x - 80| ∧
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_satisfying_equation_l2635_263592


namespace NUMINAMATH_CALUDE_ivanov_net_worth_calculation_l2635_263586

/-- The net worth of the Ivanov family -/
def ivanov_net_worth : ℕ := by sorry

/-- The value of the Ivanov family's apartment in rubles -/
def apartment_value : ℕ := 3000000

/-- The value of the Ivanov family's car in rubles -/
def car_value : ℕ := 900000

/-- The amount in the Ivanov family's bank deposit in rubles -/
def bank_deposit : ℕ := 300000

/-- The value of the Ivanov family's securities in rubles -/
def securities_value : ℕ := 200000

/-- The amount of liquid cash the Ivanov family has in rubles -/
def liquid_cash : ℕ := 100000

/-- The Ivanov family's mortgage balance in rubles -/
def mortgage_balance : ℕ := 1500000

/-- The Ivanov family's car loan balance in rubles -/
def car_loan_balance : ℕ := 500000

/-- The Ivanov family's debt to relatives in rubles -/
def debt_to_relatives : ℕ := 200000

/-- Theorem stating that the Ivanov family's net worth is 2,300,000 rubles -/
theorem ivanov_net_worth_calculation :
  ivanov_net_worth = 
    (apartment_value + car_value + bank_deposit + securities_value + liquid_cash) -
    (mortgage_balance + car_loan_balance + debt_to_relatives) :=
by sorry

end NUMINAMATH_CALUDE_ivanov_net_worth_calculation_l2635_263586


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2635_263599

theorem sum_of_squares_of_roots (k l m n a b c : ℕ) :
  k ≠ l ∧ k ≠ m ∧ k ≠ n ∧ l ≠ m ∧ l ≠ n ∧ m ≠ n →
  ((a * k^2 - b * k + c = 0 ∨ c * k^2 - 16 * b * k + 256 * a = 0) ∧
   (a * l^2 - b * l + c = 0 ∨ c * l^2 - 16 * b * l + 256 * a = 0) ∧
   (a * m^2 - b * m + c = 0 ∨ c * m^2 - 16 * b * m + 256 * a = 0) ∧
   (a * n^2 - b * n + c = 0 ∨ c * n^2 - 16 * b * n + 256 * a = 0)) →
  k^2 + l^2 + m^2 + n^2 = 325 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2635_263599


namespace NUMINAMATH_CALUDE_pizza_coverage_theorem_l2635_263546

/-- Represents the properties of a pizza with pepperoni -/
structure PepperoniPizza where
  pizza_diameter : ℝ
  pepperoni_across : ℕ
  total_pepperoni : ℕ

/-- Calculates the fraction of pizza covered by pepperoni -/
def pepperoni_coverage (p : PepperoniPizza) : ℚ :=
  sorry

/-- Theorem stating the fraction of pizza covered by pepperoni for the given conditions -/
theorem pizza_coverage_theorem (p : PepperoniPizza) 
  (h1 : p.pizza_diameter = 18)
  (h2 : p.pepperoni_across = 8)
  (h3 : p.total_pepperoni = 36) :
  pepperoni_coverage p = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_pizza_coverage_theorem_l2635_263546


namespace NUMINAMATH_CALUDE_min_value_theorem_l2635_263532

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (min_val : ℝ), min_val = 2 * Real.sqrt 3 ∧
   ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 →
   (2 * x^2 + 1) / (x * y) - 2 ≥ min_val) ∧
  (2 * a^2 + 1) / (a * b) - 2 = 2 * Real.sqrt 3 ↔ a = (Real.sqrt 3 - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2635_263532


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l2635_263573

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 3)
  (h2 : new_price = 5) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 40 := by
sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l2635_263573


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2635_263589

theorem complex_fraction_equality : (1 + 3 * Complex.I) / (1 - Complex.I) = -1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2635_263589


namespace NUMINAMATH_CALUDE_parallelogram_area_l2635_263551

/-- The area of a parallelogram with base 26 cm and height 16 cm is 416 square centimeters. -/
theorem parallelogram_area : 
  let base : ℝ := 26
  let height : ℝ := 16
  let area : ℝ := base * height
  area = 416 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2635_263551


namespace NUMINAMATH_CALUDE_problem_statement_l2635_263534

theorem problem_statement :
  let M := (Real.sqrt (3 + Real.sqrt 8) + Real.sqrt (3 - Real.sqrt 8)) / Real.sqrt (2 * Real.sqrt 2 + 1) - Real.sqrt (4 - 2 * Real.sqrt 3)
  M = 3 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2635_263534


namespace NUMINAMATH_CALUDE_intersection_single_element_l2635_263530

/-- The value of k when the intersection of sets A and B has only one element -/
theorem intersection_single_element (x y : ℝ) :
  let A := {p : ℝ × ℝ | p.1^2 - 3*p.1*p.2 + 4*p.2^2 = 7/2}
  let B := {p : ℝ × ℝ | ∃ (k : ℝ), k > 0 ∧ k*p.1 + p.2 = 2}
  (∃! p, p ∈ A ∩ B) → (∃ (k : ℝ), k = 1/4 ∧ k > 0 ∧ ∀ p, p ∈ A ∩ B → k*p.1 + p.2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_single_element_l2635_263530


namespace NUMINAMATH_CALUDE_range_of_a_l2635_263521

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : 
  (B a ⊆ A) ↔ (a ≤ -1 ∨ a = 1) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2635_263521


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a2_l2635_263596

/-- An arithmetic sequence with specified conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a2 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 3) 
  (h_sum : a 2 + a 3 = 12) : 
  a 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a2_l2635_263596


namespace NUMINAMATH_CALUDE_math_club_team_selection_l2635_263584

/-- The number of ways to select a team from a math club --/
def select_team (total_boys : ℕ) (total_girls : ℕ) (team_size : ℕ) (team_boys : ℕ) (team_girls : ℕ) 
  (experienced_boys : ℕ) (experienced_girls : ℕ) : ℕ :=
  (Nat.choose (total_boys - experienced_boys) (team_boys - experienced_boys)) * 
  (Nat.choose (total_girls - experienced_girls) (team_girls - experienced_girls))

/-- Theorem: The number of ways to select the team is 540 --/
theorem math_club_team_selection :
  select_team 7 10 6 3 3 1 1 = 540 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l2635_263584


namespace NUMINAMATH_CALUDE_complex_magnitude_l2635_263581

theorem complex_magnitude (i z : ℂ) : 
  i * i = -1 → i * z = 1 - i → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2635_263581


namespace NUMINAMATH_CALUDE_hacker_can_achieve_goal_l2635_263553

/-- Represents a user in the social network -/
structure User where
  id : Nat
  followers : Finset Nat
  rating : Nat

/-- Represents the social network -/
structure SocialNetwork where
  users : Finset User
  m : Nat

/-- Represents a hacker's action: increasing a user's rating by 1 or doing nothing -/
inductive HackerAction
  | Increase (userId : Nat)
  | DoNothing

/-- Update ratings based on followers -/
def updateRatings (sn : SocialNetwork) : SocialNetwork :=
  sorry

/-- Apply hacker's action to the social network -/
def applyHackerAction (sn : SocialNetwork) (action : HackerAction) : SocialNetwork :=
  sorry

/-- Check if all ratings are divisible by m -/
def allRatingsDivisible (sn : SocialNetwork) : Prop :=
  sorry

/-- The main theorem -/
theorem hacker_can_achieve_goal (sn : SocialNetwork) :
  ∃ (actions : List HackerAction), allRatingsDivisible (actions.foldl applyHackerAction sn) :=
sorry

end NUMINAMATH_CALUDE_hacker_can_achieve_goal_l2635_263553


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extreme_value_l2635_263587

noncomputable section

def f (x : ℝ) := Real.log x - x

theorem f_monotonicity_and_extreme_value :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x, 0 < x → f x ≤ f 1) ∧
  f 1 = -1 :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extreme_value_l2635_263587


namespace NUMINAMATH_CALUDE_shaded_area_possibilities_l2635_263535

/-- Represents a rectangle on a grid --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the configuration of rectangles in the problem --/
structure Configuration where
  abcd : Rectangle
  pqrs : Rectangle
  qrst : Rectangle
  upper_right : Rectangle

/-- The main theorem statement --/
theorem shaded_area_possibilities (config : Configuration) : 
  (config.abcd.width * config.abcd.height = 33) →
  (config.abcd.width < 7 ∧ config.abcd.height < 7) →
  (config.abcd.width ≥ 1 ∧ config.abcd.height ≥ 1) →
  (config.pqrs.width < 7 ∧ config.pqrs.height < 7) →
  (config.qrst.width < 7 ∧ config.qrst.height < 7) →
  (config.upper_right.width < 7 ∧ config.upper_right.height < 7) →
  (config.qrst.width = config.qrst.height) →
  (config.pqrs.width < config.upper_right.height) →
  (config.pqrs.width ≠ config.pqrs.height) →
  (config.upper_right.width ≠ config.upper_right.height) →
  (∃ (shaded_area : ℕ), 
    shaded_area = config.abcd.width * config.abcd.height - 
      (config.pqrs.width * config.pqrs.height + 
       config.qrst.width * config.qrst.height + 
       config.upper_right.width * config.upper_right.height) ∧
    (shaded_area = 21 ∨ shaded_area = 20 ∨ shaded_area = 17)) :=
by
  sorry


end NUMINAMATH_CALUDE_shaded_area_possibilities_l2635_263535


namespace NUMINAMATH_CALUDE_vector_subtraction_l2635_263500

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (2, 1) → b = (-3, 4) → a - b = (5, -3) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2635_263500


namespace NUMINAMATH_CALUDE_square_diff_value_l2635_263590

theorem square_diff_value (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_value_l2635_263590


namespace NUMINAMATH_CALUDE_ring_payment_possible_l2635_263528

/-- Represents a chain of rings -/
structure RingChain :=
  (size : ℕ)

/-- Represents a cut ring chain -/
structure CutRingChain :=
  (segments : List RingChain)
  (total_size : ℕ)

/-- Represents a daily payment -/
structure DailyPayment :=
  (rings_given : ℕ)
  (rings_taken : ℕ)

def is_valid_payment_sequence (payments : List DailyPayment) : Prop :=
  payments.length = 7 ∧
  ∀ p ∈ payments, p.rings_given - p.rings_taken = 1

def can_make_payments (chain : RingChain) : Prop :=
  ∃ (cut_chain : CutRingChain) (payments : List DailyPayment),
    chain.size = 7 ∧
    cut_chain.total_size = 7 ∧
    cut_chain.segments.length ≤ 3 ∧
    is_valid_payment_sequence payments

theorem ring_payment_possible :
  ∃ (chain : RingChain), can_make_payments chain :=
sorry

end NUMINAMATH_CALUDE_ring_payment_possible_l2635_263528


namespace NUMINAMATH_CALUDE_logarithm_calculation_l2635_263526

theorem logarithm_calculation : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - (5 : ℝ) ^ (Real.log 3 / Real.log 5) = -1 :=
by sorry

-- Note: We cannot include the second part of the problem due to inconsistencies in the problem statement and solution.

end NUMINAMATH_CALUDE_logarithm_calculation_l2635_263526


namespace NUMINAMATH_CALUDE_distance_to_big_rock_l2635_263559

/-- The distance to Big Rock given the rower's speed, river current, and round trip time -/
theorem distance_to_big_rock (v : ℝ) (c : ℝ) (t : ℝ) (h1 : v = 6) (h2 : c = 1) (h3 : t = 1) :
  ∃ d : ℝ, d = 35 / 12 ∧ d / (v - c) + d / (v + c) = t :=
by sorry

end NUMINAMATH_CALUDE_distance_to_big_rock_l2635_263559


namespace NUMINAMATH_CALUDE_carton_length_is_30_inches_l2635_263507

/-- Proves that the length of a carton is 30 inches given specific dimensions and constraints -/
theorem carton_length_is_30_inches 
  (carton_width : ℕ) 
  (carton_height : ℕ)
  (soap_length : ℕ) 
  (soap_width : ℕ) 
  (soap_height : ℕ)
  (max_soap_boxes : ℕ)
  (h1 : carton_width = 42)
  (h2 : carton_height = 60)
  (h3 : soap_length = 7)
  (h4 : soap_width = 6)
  (h5 : soap_height = 5)
  (h6 : max_soap_boxes = 360) :
  ∃ (carton_length : ℕ), carton_length = 30 ∧ 
    carton_length * carton_width * carton_height = 
    max_soap_boxes * soap_length * soap_width * soap_height :=
by
  sorry

end NUMINAMATH_CALUDE_carton_length_is_30_inches_l2635_263507


namespace NUMINAMATH_CALUDE_three_plane_division_l2635_263545

/-- The number of regions that n planes can divide 3-dimensional space into -/
def regions (n : ℕ) : ℕ := sorry

/-- The minimum number of regions that 3 planes can divide 3-dimensional space into -/
def min_regions : ℕ := regions 3

/-- The maximum number of regions that 3 planes can divide 3-dimensional space into -/
def max_regions : ℕ := regions 3

theorem three_plane_division :
  min_regions = 4 ∧ max_regions = 8 := by sorry

end NUMINAMATH_CALUDE_three_plane_division_l2635_263545


namespace NUMINAMATH_CALUDE_inequality_solution_l2635_263574

/-- Theorem: Solutions to the inequality ax^2 - 2 ≥ 2x - ax for a < 0 -/
theorem inequality_solution (a : ℝ) (h : a < 0) :
  (∀ x : ℝ, ¬(a * x^2 - 2 ≥ 2 * x - a * x) ∨ (a = -2 ∧ x = -1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2635_263574


namespace NUMINAMATH_CALUDE_rich_walk_ratio_l2635_263522

theorem rich_walk_ratio : 
  ∀ (x : ℝ), 
    (20 : ℝ) + 200 + 220 * x + ((20 + 200 + 220 * x) / 2) = 990 → 
    (220 * x) / (20 + 200) = 2 := by
  sorry

end NUMINAMATH_CALUDE_rich_walk_ratio_l2635_263522


namespace NUMINAMATH_CALUDE_cartesian_plane_problem_l2635_263504

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, 0)

-- Define vectors
def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B

-- Define the length of OC
def OC_length : ℝ := 1

-- Theorem statement
theorem cartesian_plane_problem :
  -- Part 1: Angle between OA and OB is 45°
  let angle := Real.arccos ((OA.1 * OB.1 + OA.2 * OB.2) / (Real.sqrt (OA.1^2 + OA.2^2) * Real.sqrt (OB.1^2 + OB.2^2)))
  angle = Real.pi / 4 ∧
  -- Part 2: If OC ⊥ OA, then C has coordinates (±√2/2, ±√2/2)
  (∀ C : ℝ × ℝ, (C.1 * OA.1 + C.2 * OA.2 = 0 ∧ C.1^2 + C.2^2 = OC_length^2) →
    (C = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ∨ C = (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2))) ∧
  -- Part 3: Range of |OA + OB + OC|
  (∀ C : ℝ × ℝ, C.1^2 + C.2^2 = OC_length^2 →
    Real.sqrt 10 - 1 ≤ Real.sqrt ((OA.1 + OB.1 + C.1)^2 + (OA.2 + OB.2 + C.2)^2) ∧
    Real.sqrt ((OA.1 + OB.1 + C.1)^2 + (OA.2 + OB.2 + C.2)^2) ≤ Real.sqrt 10 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_cartesian_plane_problem_l2635_263504


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2635_263517

def N : ℕ := 38 * 38 * 91 * 210

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors N) * 14 = sum_even_divisors N := by sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l2635_263517


namespace NUMINAMATH_CALUDE_poly_simplification_poly_evaluation_l2635_263540

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ :=
  (2*x^5 - 3*x^4 + 5*x^3 - 9*x^2 + 8*x - 15) + (5*x^4 - 2*x^3 + 3*x^2 - 4*x + 9)

-- Define the simplified polynomial
def simplified_poly (x : ℝ) : ℝ :=
  2*x^5 + 2*x^4 + 3*x^3 - 6*x^2 + 4*x - 6

-- Theorem stating that the original polynomial equals the simplified polynomial
theorem poly_simplification (x : ℝ) : original_poly x = simplified_poly x := by
  sorry

-- Theorem stating that the simplified polynomial evaluated at x = 2 equals 98
theorem poly_evaluation : simplified_poly 2 = 98 := by
  sorry

end NUMINAMATH_CALUDE_poly_simplification_poly_evaluation_l2635_263540


namespace NUMINAMATH_CALUDE_prob_both_selected_l2635_263595

/-- The probability of brother X being selected -/
def prob_X : ℚ := 1/5

/-- The probability of brother Y being selected -/
def prob_Y : ℚ := 2/3

/-- Theorem: The probability of both brothers X and Y being selected is 2/15 -/
theorem prob_both_selected : prob_X * prob_Y = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_selected_l2635_263595
