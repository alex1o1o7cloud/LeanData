import Mathlib

namespace NUMINAMATH_CALUDE_not_p_or_q_is_false_l1457_145758

-- Define proposition p
def p : Prop := ∀ x : ℝ, (λ x : ℝ => x^3) (-x) = -((λ x : ℝ => x^3) x)

-- Define proposition q
def q : Prop := ∀ a b c : ℝ, b^2 = a*c → ∃ r : ℝ, (a = b/r ∧ b = c*r) ∨ (a = b*r ∧ b = c/r)

-- Theorem to prove
theorem not_p_or_q_is_false : ¬(¬p ∨ q) := by sorry

end NUMINAMATH_CALUDE_not_p_or_q_is_false_l1457_145758


namespace NUMINAMATH_CALUDE_equation_solution_l1457_145745

theorem equation_solution : ∃ x : ℤ, 45 - (x - (37 - (15 - 17))) = 56 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1457_145745


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1457_145739

def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def B : Set ℝ := {-2, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1457_145739


namespace NUMINAMATH_CALUDE_linear_system_existence_l1457_145731

theorem linear_system_existence :
  ∃ m : ℝ, ∀ x y : ℝ, (m - 1) * x - y = 1 ∧ m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_existence_l1457_145731


namespace NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l1457_145727

/-- Given a point (1, 2, 3) in a three-dimensional Cartesian coordinate system,
    its symmetric point with respect to the xoy plane is (1, 2, -3). -/
theorem symmetric_point_xoy_plane :
  let original_point : ℝ × ℝ × ℝ := (1, 2, 3)
  let xoy_plane : Set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}
  let symmetric_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, p.2.1, -p.2.2)
  symmetric_point original_point = (1, 2, -3) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l1457_145727


namespace NUMINAMATH_CALUDE_second_number_is_40_l1457_145726

theorem second_number_is_40 (a b c : ℕ+) 
  (sum_eq : a + b + c = 120)
  (ratio_ab : (a : ℚ) / (b : ℚ) = 3 / 4)
  (ratio_bc : (b : ℚ) / (c : ℚ) = 7 / 9) :
  b = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_40_l1457_145726


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l1457_145755

theorem consecutive_odd_numbers_sum (n1 n2 n3 : ℕ) : 
  (n1 % 2 = 1) →  -- n1 is odd
  (n2 = n1 + 2) →  -- n2 is the next consecutive odd number
  (n3 = n2 + 2) →  -- n3 is the next consecutive odd number after n2
  (n3 = 27) →      -- the largest number is 27
  (n1 + n2 + n3 ≠ 72) :=  -- their sum cannot be 72
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l1457_145755


namespace NUMINAMATH_CALUDE_seven_people_seven_rooms_l1457_145786

/-- The number of ways to assign n people to m rooms with at most k people per room -/
def assignmentCount (n m k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 131460 ways to assign 7 people to 7 rooms with at most 2 people per room -/
theorem seven_people_seven_rooms : assignmentCount 7 7 2 = 131460 := by sorry

end NUMINAMATH_CALUDE_seven_people_seven_rooms_l1457_145786


namespace NUMINAMATH_CALUDE_min_sum_of_product_3920_l1457_145703

theorem min_sum_of_product_3920 (x y z : ℕ+) (h : x * y * z = 3920) :
  ∃ (a b c : ℕ+), a * b * c = 3920 ∧ (∀ x' y' z' : ℕ+, x' * y' * z' = 3920 → a + b + c ≤ x' + y' + z') ∧ a + b + c = 70 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_3920_l1457_145703


namespace NUMINAMATH_CALUDE_water_speed_calculation_l1457_145723

def swim_speed : ℝ := 4
def distance : ℝ := 8
def time : ℝ := 4

theorem water_speed_calculation (v : ℝ) : 
  (swim_speed - v) * time = distance → v = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_speed_calculation_l1457_145723


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1457_145741

theorem inequality_system_integer_solutions :
  let S := {x : ℤ | (5 * x + 1 > 3 * (x - 1)) ∧ ((x - 1) / 2 ≥ 2 * x - 4)}
  S = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l1457_145741


namespace NUMINAMATH_CALUDE_smallest_result_l1457_145705

def S : Finset Nat := {2, 3, 4, 6, 8, 9}

def process (a b c : Nat) : Nat :=
  max (max ((a + b) * c) ((a + c) * b)) ((b + c) * a)

theorem smallest_result :
  ∃ (a b c : Nat), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  process a b c = 14 ∧
  ∀ (x y z : Nat), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  process x y z ≥ 14 :=
sorry

end NUMINAMATH_CALUDE_smallest_result_l1457_145705


namespace NUMINAMATH_CALUDE_coterminal_pi_third_pi_equals_180_degrees_arc_length_pi_third_l1457_145761

-- Define the set of coterminal angles
def coterminalAngles (θ : ℝ) : Set ℝ := {α | ∃ k : ℤ, α = θ + 2 * k * Real.pi}

-- Statement 1: Coterminal angles with π/3
theorem coterminal_pi_third : 
  coterminalAngles (Real.pi / 3) = {α | ∃ k : ℤ, α = Real.pi / 3 + 2 * k * Real.pi} :=
sorry

-- Statement 2: π radians equals 180 degrees
theorem pi_equals_180_degrees : 
  Real.pi = 180 * (Real.pi / 180) :=
sorry

-- Statement 3: Arc length in a circle
theorem arc_length_pi_third : 
  let r : ℝ := 6
  let θ : ℝ := Real.pi / 3
  r * θ = 2 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_coterminal_pi_third_pi_equals_180_degrees_arc_length_pi_third_l1457_145761


namespace NUMINAMATH_CALUDE_max_fences_for_100_houses_prove_max_fences_199_l1457_145760

/-- Represents a village with houses and fences. -/
structure Village where
  num_houses : ℕ
  num_fences : ℕ

/-- Represents the process of combining houses within a fence. -/
def combine_houses (v : Village) : Village :=
  { num_houses := v.num_houses - 1
  , num_fences := v.num_fences - 2 }

/-- The maximum number of fences for a given number of houses. -/
def max_fences (n : ℕ) : ℕ :=
  2 * n - 1

/-- Theorem stating the maximum number of fences for 100 houses. -/
theorem max_fences_for_100_houses :
  ∃ (v : Village), v.num_houses = 100 ∧ v.num_fences = max_fences v.num_houses :=
by
  sorry

/-- Theorem proving that 199 is the maximum number of fences for 100 houses. -/
theorem prove_max_fences_199 :
  max_fences 100 = 199 :=
by
  sorry

end NUMINAMATH_CALUDE_max_fences_for_100_houses_prove_max_fences_199_l1457_145760


namespace NUMINAMATH_CALUDE_transformation_maps_curve_to_ellipse_l1457_145701

/-- The transformation that maps a curve to an ellipse -/
def transformation (x' y' : ℝ) : ℝ × ℝ :=
  (2 * x', y')

/-- The original curve equation -/
def original_curve (x y : ℝ) : Prop :=
  y^2 = 4

/-- The transformed ellipse equation -/
def transformed_ellipse (x' y' : ℝ) : Prop :=
  x'^2 + y'^2 / 4 = 1

/-- Theorem stating that the transformation maps the original curve to the ellipse -/
theorem transformation_maps_curve_to_ellipse :
  ∀ x' y', original_curve (transformation x' y').1 (transformation x' y').2 ↔ transformed_ellipse x' y' :=
sorry

end NUMINAMATH_CALUDE_transformation_maps_curve_to_ellipse_l1457_145701


namespace NUMINAMATH_CALUDE_abc_product_equals_k_absolute_value_l1457_145756

theorem abc_product_equals_k_absolute_value 
  (a b c k : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ k ≠ 0) 
  (h_eq : a + k / b = b + k / c ∧ b + k / c = c + k / a) : 
  |a * b * c| = |k| := by
  sorry

end NUMINAMATH_CALUDE_abc_product_equals_k_absolute_value_l1457_145756


namespace NUMINAMATH_CALUDE_solve_equation_l1457_145711

theorem solve_equation (x : ℝ) : 
  (1 : ℝ) / 7 + 7 / x = 15 / x + (1 : ℝ) / 15 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1457_145711


namespace NUMINAMATH_CALUDE_min_value_fraction_l1457_145747

theorem min_value_fraction (x : ℝ) (h : x > 5) : 
  x^2 / (x - 5) ≥ 20 ∧ ∃ y > 5, y^2 / (y - 5) = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1457_145747


namespace NUMINAMATH_CALUDE_max_red_balls_l1457_145778

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Yellow
  | Green

/-- The number marked on a ball of a given color -/
def ballNumber (c : BallColor) : Nat :=
  match c with
  | BallColor.Red => 4
  | BallColor.Yellow => 5
  | BallColor.Green => 6

/-- The total number of balls drawn -/
def totalBalls : Nat := 8

/-- The sum of numbers on all drawn balls -/
def totalSum : Nat := 39

/-- A configuration of drawn balls -/
structure BallConfiguration where
  red : Nat
  yellow : Nat
  green : Nat
  sum_eq : red + yellow + green = totalBalls
  number_sum_eq : red * ballNumber BallColor.Red + 
                  yellow * ballNumber BallColor.Yellow + 
                  green * ballNumber BallColor.Green = totalSum

/-- The maximum number of red balls in any valid configuration is 4 -/
theorem max_red_balls : 
  ∀ (config : BallConfiguration), config.red ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_red_balls_l1457_145778


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1457_145767

theorem complex_fraction_equality : (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1457_145767


namespace NUMINAMATH_CALUDE_time_to_save_downpayment_l1457_145724

def salary : ℝ := 150000
def savings_rate : ℝ := 0.10
def house_cost : ℝ := 450000
def downpayment_rate : ℝ := 0.20

def yearly_savings : ℝ := salary * savings_rate
def required_downpayment : ℝ := house_cost * downpayment_rate

theorem time_to_save_downpayment :
  required_downpayment / yearly_savings = 6 := by sorry

end NUMINAMATH_CALUDE_time_to_save_downpayment_l1457_145724


namespace NUMINAMATH_CALUDE_negation_of_universal_exponential_exponential_negation_l1457_145744

theorem negation_of_universal_exponential (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬(P x) :=
by sorry

theorem exponential_negation :
  (¬ ∀ x : ℝ, Real.exp x > 0) ↔ (∃ x : ℝ, Real.exp x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_exponential_exponential_negation_l1457_145744


namespace NUMINAMATH_CALUDE_backpack_cost_l1457_145718

/-- Calculates the total cost of personalized backpacks for grandchildren --/
def totalCost (originalPrice taxRates : List ℝ) (discount monogrammingCost coupon : ℝ) : ℝ :=
  let discountedPrice := originalPrice.map (λ p => p * (1 - discount))
  let priceWithMonogram := discountedPrice.map (λ p => p + monogrammingCost)
  let priceWithTax := List.zipWith (λ p r => p * (1 + r)) priceWithMonogram taxRates
  priceWithTax.sum - coupon

/-- Theorem stating the total cost of backpacks for grandchildren --/
theorem backpack_cost :
  let originalPrice := [20, 20, 20, 20, 20]
  let taxRates := [0.06, 0.08, 0.055, 0.0725, 0.04]
  let discount := 0.2
  let monogrammingCost := 12
  let coupon := 5
  totalCost originalPrice taxRates discount monogrammingCost coupon = 143.61 := by
  sorry

#eval totalCost [20, 20, 20, 20, 20] [0.06, 0.08, 0.055, 0.0725, 0.04] 0.2 12 5

end NUMINAMATH_CALUDE_backpack_cost_l1457_145718


namespace NUMINAMATH_CALUDE_container_volume_ratio_l1457_145733

theorem container_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (2/3 : ℝ) * V₂ →
  V₁ / V₂ = 8/9 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l1457_145733


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1457_145757

def A : Set ℝ := {x | x^2 - 2*x = 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1457_145757


namespace NUMINAMATH_CALUDE_fraction_equality_l1457_145750

theorem fraction_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let x := (1/2) * (Real.sqrt (a/b) - Real.sqrt (b/a))
  (2*a * Real.sqrt (1 + x^2)) / (x + Real.sqrt (1 + x^2)) = a + b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1457_145750


namespace NUMINAMATH_CALUDE_simplify_fraction_l1457_145793

theorem simplify_fraction (a : ℝ) (h : a ≠ 1) :
  1 - 1 / (1 + a / (1 - a)) = a := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1457_145793


namespace NUMINAMATH_CALUDE_investment_problem_l1457_145776

theorem investment_problem (T : ℝ) :
  (0.10 * (T - 700) - 0.08 * 700 = 74) →
  T = 2000 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l1457_145776


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1457_145740

/-- Given that α is inversely proportional to β, prove that when α = 5 and β = 20, 
    then α = 10 when β = 10 -/
theorem inverse_proportion_problem (α β : ℝ) (k : ℝ) 
    (h1 : α * β = k)  -- α is inversely proportional to β
    (h2 : 5 * 20 = k) -- α = 5 when β = 20
    : 10 * 10 = k :=  -- α = 10 when β = 10
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1457_145740


namespace NUMINAMATH_CALUDE_eulers_formula_l1457_145713

theorem eulers_formula (x : ℝ) : Complex.exp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l1457_145713


namespace NUMINAMATH_CALUDE_triangle_exists_from_altitudes_l1457_145700

theorem triangle_exists_from_altitudes (h₁ h₂ h₃ : ℝ) 
  (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (h₃_pos : h₃ > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
    h₁ = (2 * (a * b * c) / (a * (a + b + c))) ∧
    h₂ = (2 * (a * b * c) / (b * (a + b + c))) ∧
    h₃ = (2 * (a * b * c) / (c * (a + b + c))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_exists_from_altitudes_l1457_145700


namespace NUMINAMATH_CALUDE_total_points_sum_l1457_145791

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def gina_rolls : List ℕ := [6, 5, 2, 3, 4]
def helen_rolls : List ℕ := [1, 2, 4, 6, 3]

theorem total_points_sum : (gina_rolls.map g).sum + (helen_rolls.map g).sum = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_points_sum_l1457_145791


namespace NUMINAMATH_CALUDE_real_part_of_z_l1457_145775

theorem real_part_of_z (i : ℂ) (h : i^2 = -1) : Complex.re ((1 + 2*i)^2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l1457_145775


namespace NUMINAMATH_CALUDE_fraction_equality_l1457_145769

theorem fraction_equality (x y : ℚ) (hx : x = 4/7) (hy : y = 5/11) :
  (7*x + 11*y) / (63*x*y) = 11/20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1457_145769


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_sum_zero_l1457_145774

theorem sum_and_reciprocal_sum_zero (a b c d : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d)
  (h4 : a + b + c + d = 0)
  (h5 : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_sum_zero_l1457_145774


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l1457_145772

theorem complex_subtraction_simplification :
  (7 - 3*I) - (9 - 5*I) = -2 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l1457_145772


namespace NUMINAMATH_CALUDE_bijection_iteration_fixed_point_l1457_145704

theorem bijection_iteration_fixed_point {n : ℕ} (f : Fin n → Fin n) (h : Function.Bijective f) :
  ∃ M : ℕ+, ∀ i : Fin n, (f^[M.val] i) = f i := by
  sorry

end NUMINAMATH_CALUDE_bijection_iteration_fixed_point_l1457_145704


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1457_145719

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (x - 3) = 5 → x = 28 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1457_145719


namespace NUMINAMATH_CALUDE_sum_of_squares_l1457_145721

theorem sum_of_squares (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1457_145721


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l1457_145782

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l1457_145782


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l1457_145710

theorem smallest_number_divisible (n : ℕ) : n = 2135 ↔ 
  (∀ d ∈ ({5, 10, 15, 20, 25, 30, 35} : Set ℕ), (n - 35) % d = 0) ∧ 
  (∀ m < n, ∃ d ∈ ({5, 10, 15, 20, 25, 30, 35} : Set ℕ), (m - 35) % d ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l1457_145710


namespace NUMINAMATH_CALUDE_min_colors_2016_board_l1457_145738

/-- A color assignment for a square board. -/
def ColorAssignment (n : ℕ) := Fin n → Fin n → ℕ

/-- Predicate for a valid coloring of a square board. -/
def ValidColoring (n k : ℕ) (c : ColorAssignment n) : Prop :=
  -- One diagonal is colored with the first color
  (∀ i, c i i = 0) ∧
  -- Symmetric cells have the same color
  (∀ i j, c i j = c j i) ∧
  -- Cells in the same row on different sides of the diagonal have different colors
  (∀ i j₁ j₂, i < j₁ ∧ j₂ < i → c i j₁ ≠ c i j₂)

/-- Theorem stating the minimum number of colors needed for a 2016 × 2016 board. -/
theorem min_colors_2016_board :
  (∃ (c : ColorAssignment 2016), ValidColoring 2016 11 c) ∧
  (∀ k < 11, ¬ ∃ (c : ColorAssignment 2016), ValidColoring 2016 k c) :=
sorry

end NUMINAMATH_CALUDE_min_colors_2016_board_l1457_145738


namespace NUMINAMATH_CALUDE_soda_price_increase_l1457_145795

theorem soda_price_increase (candy_new : ℝ) (soda_new : ℝ) (candy_increase : ℝ) (total_old : ℝ)
  (h1 : candy_new = 20)
  (h2 : soda_new = 6)
  (h3 : candy_increase = 0.25)
  (h4 : total_old = 20) :
  (soda_new - (total_old - candy_new / (1 + candy_increase))) / (total_old - candy_new / (1 + candy_increase)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_increase_l1457_145795


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l1457_145773

/-- The number of candy pieces -/
def total_candy : ℕ := 108

/-- Predicate to check if a number divides the total candy evenly -/
def divides_candy (n : ℕ) : Prop := total_candy % n = 0

/-- Predicate to check if a number is a valid student count -/
def valid_student_count (n : ℕ) : Prop :=
  n > 1 ∧ divides_candy n

/-- The set of possible student counts -/
def possible_student_counts : Set ℕ := {12, 36, 54}

/-- Theorem stating that the possible student counts are correct -/
theorem candy_distribution_theorem :
  ∀ n : ℕ, n ∈ possible_student_counts ↔ valid_student_count n :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l1457_145773


namespace NUMINAMATH_CALUDE_simplify_expression_l1457_145709

theorem simplify_expression (a : ℝ) (ha : a > 0) :
  a^2 / (a^(1/2) * a^(2/3)) = a^(5/6) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1457_145709


namespace NUMINAMATH_CALUDE_min_value_of_exponential_expression_l1457_145765

theorem min_value_of_exponential_expression :
  ∀ x : ℝ, 16^x - 4^x + 1 ≥ (3:ℝ)/4 ∧ 
  (16^(-(1:ℝ)/2) - 4^(-(1:ℝ)/2) + 1 = (3:ℝ)/4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_expression_l1457_145765


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l1457_145771

theorem unique_six_digit_number : ∃! n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧  -- six-digit number
  n % 10 = 2 ∧                 -- ends in 2
  2000000 + (n / 10) = 3 * n ∧ -- moving 2 to first position triples the number
  n = 857142 := by
sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l1457_145771


namespace NUMINAMATH_CALUDE_eighth_term_value_l1457_145789

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = n², 
    prove that the 8th term a₈ = 15 -/
theorem eighth_term_value (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h : ∀ n, S n = n^2) : 
    a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l1457_145789


namespace NUMINAMATH_CALUDE_new_pages_read_per_week_jim_new_pages_read_l1457_145725

/-- Calculates the new number of pages read per week after changes in reading speed and time --/
theorem new_pages_read_per_week
  (initial_rate : ℝ)
  (initial_pages : ℝ)
  (speed_increase : ℝ)
  (time_decrease : ℝ)
  (h1 : initial_rate = 40)
  (h2 : initial_pages = 600)
  (h3 : speed_increase = 1.5)
  (h4 : time_decrease = 4)
  : ℝ :=
  by
  -- Proof goes here
  sorry

/-- The main theorem stating that Jim now reads 660 pages per week --/
theorem jim_new_pages_read :
  new_pages_read_per_week 40 600 1.5 4 rfl rfl rfl rfl = 660 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_new_pages_read_per_week_jim_new_pages_read_l1457_145725


namespace NUMINAMATH_CALUDE_ticket_price_reduction_l1457_145737

theorem ticket_price_reduction 
  (original_price : ℚ)
  (sold_increase_ratio : ℚ)
  (revenue_increase_ratio : ℚ)
  (price_reduction : ℚ) :
  original_price = 50 →
  sold_increase_ratio = 1/3 →
  revenue_increase_ratio = 1/4 →
  (original_price - price_reduction) * (1 + sold_increase_ratio) = original_price * (1 + revenue_increase_ratio) →
  price_reduction = 25/2 := by
sorry

end NUMINAMATH_CALUDE_ticket_price_reduction_l1457_145737


namespace NUMINAMATH_CALUDE_fourth_person_height_l1457_145766

/-- Proves that the height of the fourth person is 82 inches given the conditions -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- Heights are in increasing order
  h₂ - h₁ = 2 →  -- Difference between 1st and 2nd person
  h₃ - h₂ = 2 →  -- Difference between 2nd and 3rd person
  h₄ - h₃ = 6 →  -- Difference between 3rd and 4th person
  (h₁ + h₂ + h₃ + h₄) / 4 = 76 →  -- Average height
  h₄ = 82 := by
sorry

end NUMINAMATH_CALUDE_fourth_person_height_l1457_145766


namespace NUMINAMATH_CALUDE_area_of_inscribed_circle_rectangle_l1457_145712

/-- A rectangle with an inscribed circle -/
structure InscribedCircleRectangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The width of the rectangle -/
  w : ℝ
  /-- The height of the rectangle -/
  h : ℝ
  /-- The circle is tangent to all sides -/
  tangent_to_sides : w = h
  /-- The circle passes through the midpoint of a diagonal -/
  passes_through_midpoint : w^2 / 4 + h^2 / 4 = r^2

/-- The area of a rectangle with an inscribed circle passing through the midpoint of a diagonal is 2r^2 -/
theorem area_of_inscribed_circle_rectangle (rect : InscribedCircleRectangle) : 
  rect.w * rect.h = 2 * rect.r^2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribed_circle_rectangle_l1457_145712


namespace NUMINAMATH_CALUDE_percentage_problem_l1457_145708

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 40 + (8 / 100) * 24 = 5.92 ↔ P = 10 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l1457_145708


namespace NUMINAMATH_CALUDE_function_f_property_l1457_145788

/-- A function satisfying the given properties -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = 2 - f x) ∧
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

/-- The theorem statement -/
theorem function_f_property (f : ℝ → ℝ) (a : ℝ) 
  (hf : FunctionF f) 
  (h : ∀ x ∈ Set.Icc 1 2, f (a * x + 2) + f 1 ≤ 2) : 
  a ∈ Set.Iic (-3) :=
sorry

end NUMINAMATH_CALUDE_function_f_property_l1457_145788


namespace NUMINAMATH_CALUDE_previous_day_visitor_count_l1457_145787

/-- The number of visitors to Buckingham Palace on the current day -/
def current_day_visitors : ℕ := 661

/-- The difference in visitors between the current day and the previous day -/
def visitor_difference : ℕ := 61

/-- The number of visitors on the previous day -/
def previous_day_visitors : ℕ := current_day_visitors - visitor_difference

theorem previous_day_visitor_count : previous_day_visitors = 600 := by
  sorry

end NUMINAMATH_CALUDE_previous_day_visitor_count_l1457_145787


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1457_145777

theorem polynomial_division_theorem (x : ℚ) : 
  let dividend := 10 * x^4 - 3 * x^3 + 2 * x^2 - x + 6
  let divisor := 3 * x + 4
  let quotient := 10/3 * x^3 - 49/9 * x^2 + 427/27 * x - 287/54
  let remainder := 914/27
  dividend = divisor * quotient + remainder := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1457_145777


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l1457_145702

/-- Calculates the total cost of plastering a rectangular tank. -/
def plastering_cost (length width depth : ℝ) (cost_per_sqm : ℝ) : ℝ :=
  let bottom_area := length * width
  let long_walls_area := 2 * (length * depth)
  let short_walls_area := 2 * (width * depth)
  let total_area := bottom_area + long_walls_area + short_walls_area
  total_area * cost_per_sqm

/-- Theorem stating the cost of plastering a specific tank. -/
theorem tank_plastering_cost :
  plastering_cost 60 25 10 0.9 = 2880 := by
  sorry

#eval plastering_cost 60 25 10 0.9

end NUMINAMATH_CALUDE_tank_plastering_cost_l1457_145702


namespace NUMINAMATH_CALUDE_goal_state_reachable_l1457_145729

/-- Represents the state of the three jugs -/
structure JugState :=
  (jug1 : ℕ)
  (jug2 : ℕ)
  (jug3 : ℕ)

/-- Represents the capacities of the three jugs -/
structure JugCapacities :=
  (cap1 : ℕ)
  (cap2 : ℕ)
  (cap3 : ℕ)

/-- Defines a valid pouring action between two jugs -/
inductive PourAction
  | pour12 : PourAction
  | pour13 : PourAction
  | pour21 : PourAction
  | pour23 : PourAction
  | pour31 : PourAction
  | pour32 : PourAction

/-- Applies a pour action to a given state, respecting jug capacities -/
def applyPour (state : JugState) (action : PourAction) (caps : JugCapacities) : JugState :=
  sorry

/-- Checks if a given state is the goal state (6, 6, 0) -/
def isGoalState (state : JugState) : Prop :=
  state.jug1 = 6 ∧ state.jug2 = 6 ∧ state.jug3 = 0

/-- Theorem stating that the goal state is reachable -/
theorem goal_state_reachable (initialState : JugState) (caps : JugCapacities) : 
  initialState = JugState.mk 12 0 0 →
  caps = JugCapacities.mk 12 8 5 →
  ∃ (actions : List PourAction), isGoalState (actions.foldl (fun s a => applyPour s a caps) initialState) :=
  sorry

end NUMINAMATH_CALUDE_goal_state_reachable_l1457_145729


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l1457_145790

/-- Represents the total number of teachers -/
def total_teachers : ℕ := 150

/-- Represents the number of senior teachers -/
def senior_teachers : ℕ := 15

/-- Represents the number of intermediate teachers -/
def intermediate_teachers : ℕ := 90

/-- Represents the number of teachers sampled -/
def sampled_teachers : ℕ := 30

/-- Represents the number of junior teachers -/
def junior_teachers : ℕ := total_teachers - senior_teachers - intermediate_teachers

/-- Theorem stating the correct numbers of teachers selected in each category -/
theorem stratified_sampling_result :
  (senior_teachers * sampled_teachers / total_teachers = 3) ∧
  (intermediate_teachers * sampled_teachers / total_teachers = 18) ∧
  (junior_teachers * sampled_teachers / total_teachers = 9) :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_l1457_145790


namespace NUMINAMATH_CALUDE_prob_not_all_same_dice_l1457_145722

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability that not all dice show the same number when rolled -/
def prob_not_all_same : ℚ := 1295 / 1296

/-- Theorem stating that the probability of not all dice showing the same number is 1295/1296 -/
theorem prob_not_all_same_dice (h : sides = 6 ∧ num_dice = 5) : 
  prob_not_all_same = 1295 / 1296 := by sorry

end NUMINAMATH_CALUDE_prob_not_all_same_dice_l1457_145722


namespace NUMINAMATH_CALUDE_bottom_right_not_divisible_by_2011_l1457_145706

/-- Represents a cell on the board -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents the board configuration -/
structure Board where
  size : Nat
  markedCells : List Cell

/-- Checks if a cell is on the main diagonal -/
def isOnMainDiagonal (c : Cell) : Prop := c.row + c.col = 2011

/-- Checks if a cell is in a corner -/
def isCorner (c : Cell) (n : Nat) : Prop :=
  (c.row = 0 ∧ c.col = 0) ∨ (c.row = 0 ∧ c.col = n - 1) ∨
  (c.row = n - 1 ∧ c.col = 0) ∨ (c.row = n - 1 ∧ c.col = n - 1)

/-- The value in the bottom-right corner of the board -/
def bottomRightValue (b : Board) : Nat :=
  sorry  -- Implementation not required for the statement

theorem bottom_right_not_divisible_by_2011 (b : Board) :
  b.size = 2012 →
  (∀ c ∈ b.markedCells, isOnMainDiagonal c ∧ ¬isCorner c b.size) →
  bottomRightValue b % 2011 = 2 :=
sorry

end NUMINAMATH_CALUDE_bottom_right_not_divisible_by_2011_l1457_145706


namespace NUMINAMATH_CALUDE_tiffany_lives_gained_l1457_145798

theorem tiffany_lives_gained (initial_lives lost_lives final_lives : ℕ) 
  (h1 : initial_lives = 43)
  (h2 : lost_lives = 14)
  (h3 : final_lives = 56) : 
  final_lives - (initial_lives - lost_lives) = 27 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_lives_gained_l1457_145798


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l1457_145763

theorem geometric_arithmetic_sequence_problem :
  ∀ a b c : ℝ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (∃ r : ℝ, b = a * r ∧ c = b * r) →
  (a * b * c = 512) →
  (2 * b = (a - 2) + (c - 2)) →
  ((a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = 16 ∧ b = 8 ∧ c = 4)) :=
by sorry


end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l1457_145763


namespace NUMINAMATH_CALUDE_equation_equivalence_and_product_l1457_145742

theorem equation_equivalence_and_product (a c x y : ℝ) :
  ∃ (r s t u : ℤ),
    ((a^r * x - a^s) * (a^t * y - a^u) = a^6 * c^3) ↔
    (a^10 * x * y - a^8 * y - a^7 * x = a^6 * (c^3 - 1)) ∧
    r * s * t * u = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_and_product_l1457_145742


namespace NUMINAMATH_CALUDE_cuboid_height_from_cube_l1457_145732

/-- The length of wire needed to make a cube with given edge length -/
def cube_wire_length (edge : ℝ) : ℝ := 12 * edge

/-- The length of wire needed to make a cuboid with given dimensions -/
def cuboid_wire_length (length width height : ℝ) : ℝ :=
  4 * (length + width + height)

theorem cuboid_height_from_cube (cube_edge length width : ℝ) 
  (h_cube_edge : cube_edge = 10)
  (h_length : length = 8)
  (h_width : width = 5) :
  ∃ (height : ℝ), 
    cube_wire_length cube_edge = cuboid_wire_length length width height ∧ 
    height = 17 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_from_cube_l1457_145732


namespace NUMINAMATH_CALUDE_lunch_break_duration_l1457_145779

-- Define the painting rates and lunch break
structure PaintingData where
  paula_rate : ℝ
  helpers_rate : ℝ
  lunch_break : ℝ

-- Define the workday durations
def monday_duration : ℝ := 9
def tuesday_duration : ℝ := 7
def wednesday_duration : ℝ := 12

-- Define the portions painted each day
def monday_portion : ℝ := 0.6
def tuesday_portion : ℝ := 0.3
def wednesday_portion : ℝ := 0.1

-- Theorem statement
theorem lunch_break_duration (d : PaintingData) : 
  (monday_duration - d.lunch_break) * (d.paula_rate + d.helpers_rate) = monday_portion ∧
  (tuesday_duration - d.lunch_break) * d.helpers_rate = tuesday_portion ∧
  (wednesday_duration - d.lunch_break) * d.paula_rate = wednesday_portion →
  d.lunch_break = 1 := by
  sorry

#check lunch_break_duration

end NUMINAMATH_CALUDE_lunch_break_duration_l1457_145779


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1457_145736

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = (1/4 : ℝ)) 
  (h2 : S = 80) 
  (h3 : S = a / (1 - r)) : 
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1457_145736


namespace NUMINAMATH_CALUDE_parking_theorem_l1457_145785

/-- The number of parking spaces in a row -/
def total_spaces : ℕ := 7

/-- The number of cars to be parked -/
def num_cars : ℕ := 4

/-- The number of consecutive empty spaces required -/
def consecutive_empty : ℕ := 3

/-- The number of different parking arrangements -/
def parking_arrangements : ℕ := 120

/-- Theorem stating that the number of ways to arrange 4 cars and 3 consecutive
    empty spaces in a row of 7 parking spaces is equal to 120 -/
theorem parking_theorem :
  (total_spaces = 7) →
  (num_cars = 4) →
  (consecutive_empty = 3) →
  (parking_arrangements = 120) :=
by sorry

end NUMINAMATH_CALUDE_parking_theorem_l1457_145785


namespace NUMINAMATH_CALUDE_coffee_table_price_is_330_l1457_145784

/-- Represents the living room set purchase -/
structure LivingRoomSet where
  sofa_price : ℕ
  armchair_price : ℕ
  num_armchairs : ℕ
  total_invoice : ℕ

/-- Calculates the price of the coffee table -/
def coffee_table_price (set : LivingRoomSet) : ℕ :=
  set.total_invoice - (set.sofa_price + set.armchair_price * set.num_armchairs)

/-- Theorem stating that the coffee table price is 330 -/
theorem coffee_table_price_is_330 (set : LivingRoomSet) 
  (h1 : set.sofa_price = 1250)
  (h2 : set.armchair_price = 425)
  (h3 : set.num_armchairs = 2)
  (h4 : set.total_invoice = 2430) :
  coffee_table_price set = 330 := by
  sorry

#check coffee_table_price_is_330

end NUMINAMATH_CALUDE_coffee_table_price_is_330_l1457_145784


namespace NUMINAMATH_CALUDE_michaels_fish_count_l1457_145707

theorem michaels_fish_count (original_count added_count total_count : ℕ) : 
  added_count = 18 →
  total_count = 49 →
  original_count + added_count = total_count :=
by
  sorry

end NUMINAMATH_CALUDE_michaels_fish_count_l1457_145707


namespace NUMINAMATH_CALUDE_proportion_estimate_correct_l1457_145768

/-- Proportion of households with 3+ housing sets -/
def proportion_with_3plus_housing (total_households : ℕ) 
  (ordinary_households : ℕ) (high_income_households : ℕ)
  (sampled_ordinary : ℕ) (sampled_high_income : ℕ)
  (sampled_ordinary_with_3plus : ℕ) (sampled_high_income_with_3plus : ℕ) : ℚ :=
  let estimated_ordinary_with_3plus := (sampled_ordinary_with_3plus : ℚ) * ordinary_households / sampled_ordinary
  let estimated_high_income_with_3plus := (sampled_high_income_with_3plus : ℚ) * high_income_households / sampled_high_income
  (estimated_ordinary_with_3plus + estimated_high_income_with_3plus) / total_households

theorem proportion_estimate_correct : 
  proportion_with_3plus_housing 100000 99000 1000 990 100 40 80 = 48 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_proportion_estimate_correct_l1457_145768


namespace NUMINAMATH_CALUDE_ideal_complex_condition_l1457_145716

def is_ideal_complex (z : ℂ) : Prop :=
  z.re = -z.im

theorem ideal_complex_condition (a b : ℝ) :
  let z : ℂ := (a / (1 - 2*I)) + b*I
  is_ideal_complex z → 3*a + 5*b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ideal_complex_condition_l1457_145716


namespace NUMINAMATH_CALUDE_orange_cost_calculation_l1457_145794

theorem orange_cost_calculation (cost_three_dozen : ℝ) (dozen_count : ℕ) :
  cost_three_dozen = 22.5 →
  dozen_count = 4 →
  (cost_three_dozen / 3) * dozen_count = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_orange_cost_calculation_l1457_145794


namespace NUMINAMATH_CALUDE_initial_boys_count_l1457_145748

theorem initial_boys_count (total : ℕ) : 
  let initial_boys := (60 * total) / 100
  let final_total := total + 2
  let final_boys := initial_boys - 3
  (2 * final_boys = final_total) → initial_boys = 24 :=
by sorry

end NUMINAMATH_CALUDE_initial_boys_count_l1457_145748


namespace NUMINAMATH_CALUDE_deposit_difference_approximately_219_01_l1457_145753

-- Constants
def initial_deposit : ℝ := 10000
def a_interest_rate : ℝ := 0.0288
def b_interest_rate : ℝ := 0.0225
def tax_rate : ℝ := 0.20
def years : ℕ := 5

-- A's total amount after 5 years
def a_total : ℝ := initial_deposit + initial_deposit * a_interest_rate * (1 - tax_rate) * years

-- B's total amount after 5 years (compound interest)
def b_total : ℝ := initial_deposit * (1 + b_interest_rate * (1 - tax_rate)) ^ years

-- Theorem statement
theorem deposit_difference_approximately_219_01 :
  ∃ ε > 0, ε < 0.005 ∧ |a_total - b_total - 219.01| < ε :=
sorry

end NUMINAMATH_CALUDE_deposit_difference_approximately_219_01_l1457_145753


namespace NUMINAMATH_CALUDE_x_minus_y_equals_two_l1457_145762

theorem x_minus_y_equals_two (x y : ℝ) 
  (sum_eq : x + y = 6) 
  (diff_squares_eq : x^2 - y^2 = 12) : 
  x - y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_two_l1457_145762


namespace NUMINAMATH_CALUDE_rainy_days_pigeonhole_l1457_145799

theorem rainy_days_pigeonhole (n : ℕ) (m : ℕ) (h : n > 2 * m) :
  ∃ (x : ℕ), x ≤ m ∧ (∃ (S : Finset ℕ), S.card ≥ 3 ∧ ∀ i ∈ S, i < n ∧ x = i % (m + 1)) :=
by
  sorry

#check rainy_days_pigeonhole 64 30

end NUMINAMATH_CALUDE_rainy_days_pigeonhole_l1457_145799


namespace NUMINAMATH_CALUDE_expected_value_3X_plus_2_l1457_145751

/-- Probability distribution for random variable X -/
def prob_dist : List (ℝ × ℝ) :=
  [(1, 0.1), (2, 0.3), (3, 0.4), (4, 0.1), (5, 0.1)]

/-- Expected value of X -/
def E (X : List (ℝ × ℝ)) : ℝ :=
  (X.map (fun (x, p) => x * p)).sum

/-- Theorem: Expected value of 3X+2 is 10.4 -/
theorem expected_value_3X_plus_2 :
  E (prob_dist.map (fun (x, p) => (3 * x + 2, p))) = 10.4 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_3X_plus_2_l1457_145751


namespace NUMINAMATH_CALUDE_multiplication_factor_l1457_145743

theorem multiplication_factor (N : ℝ) (h : N ≠ 0) : 
  let X : ℝ := 5
  let incorrect_value := N / 10
  let correct_value := N * X
  let percentage_error := |correct_value - incorrect_value| / correct_value * 100
  percentage_error = 98 := by sorry

end NUMINAMATH_CALUDE_multiplication_factor_l1457_145743


namespace NUMINAMATH_CALUDE_line_point_a_value_l1457_145796

/-- Given a line y = 0.75x + 1 and points (4, b), (a, 5), and (a, b + 1) on this line, prove that a = 16/3 -/
theorem line_point_a_value (b : ℝ) :
  (∃ (a : ℝ), (4 : ℝ) * (3/4) + 1 = b ∧ 
              a * (3/4) + 1 = 5 ∧ 
              a * (3/4) + 1 = b + 1) →
  ∃ (a : ℝ), a = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_line_point_a_value_l1457_145796


namespace NUMINAMATH_CALUDE_main_theorem_l1457_145720

/-- Definition of the function f --/
def f (a b k : ℤ) : ℤ := a * k^3 + b * k

/-- Definition of n-good --/
def is_n_good (a b n : ℤ) : Prop :=
  ∀ m k : ℤ, n ∣ (f a b k - f a b m) → n ∣ (k - m)

/-- Definition of very good --/
def is_very_good (a b : ℤ) : Prop :=
  ∀ n : ℤ, ∃ m : ℤ, m > n ∧ is_n_good a b m

/-- Main theorem --/
theorem main_theorem :
  (is_n_good 1 (-51^2) 51 ∧ ¬ is_very_good 1 (-51^2)) ∧
  (∀ a b : ℤ, is_n_good a b 2013 → is_very_good a b) := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l1457_145720


namespace NUMINAMATH_CALUDE_x_range_for_given_equation_l1457_145781

theorem x_range_for_given_equation (x y : ℝ) :
  x - 4 * Real.sqrt y = 2 * Real.sqrt (x - y) →
  x = 0 ∨ (4 ≤ x ∧ x ≤ 20) := by
  sorry

end NUMINAMATH_CALUDE_x_range_for_given_equation_l1457_145781


namespace NUMINAMATH_CALUDE_swim_club_members_swim_club_members_proof_l1457_145749

theorem swim_club_members : ℕ → Prop :=
  fun total_members =>
    let passed_test := (30 : ℚ) / 100 * total_members
    let not_passed := total_members - passed_test
    let prep_course := 12
    let no_prep_course := 30
    passed_test + not_passed = total_members ∧
    prep_course + no_prep_course = not_passed ∧
    total_members = 60

-- Proof
theorem swim_club_members_proof : ∃ n : ℕ, swim_club_members n :=
  sorry

end NUMINAMATH_CALUDE_swim_club_members_swim_club_members_proof_l1457_145749


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l1457_145780

def polynomial (x : ℝ) : ℝ := 5*(x - x^4) - 4*(x^2 - 2*x^4 + x^6) + 3*(2*x^2 - x^8)

theorem coefficient_of_x_squared (x : ℝ) : 
  ∃ (a b c : ℝ), polynomial x = 2*x^2 + a*x + b*x^3 + c*x^4 + 
    (-5)*x^4 + 8*x^4 + (-4)*x^6 + (-3)*x^8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l1457_145780


namespace NUMINAMATH_CALUDE_bugs_eating_flowers_l1457_145714

theorem bugs_eating_flowers :
  let bug_amounts : List ℝ := [2.5, 3, 1.5, 2, 4, 0.5, 3]
  bug_amounts.sum = 16.5 := by
sorry

end NUMINAMATH_CALUDE_bugs_eating_flowers_l1457_145714


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l1457_145752

/-- The area of the shaded region between two circles -/
theorem shaded_area_between_circles (r R d : ℝ) (h1 : R = 3 * r) (h2 : d = 6) (h3 : 2 ≤ R - r) : π * R^2 - π * r^2 = 72 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l1457_145752


namespace NUMINAMATH_CALUDE_algebra_test_average_l1457_145735

theorem algebra_test_average (total_average : ℝ) (male_average : ℝ) (male_count : ℕ) (female_count : ℕ) 
  (h1 : total_average = 90)
  (h2 : male_average = 87)
  (h3 : male_count = 8)
  (h4 : female_count = 12) :
  let total_count := male_count + female_count
  let total_score := total_average * total_count
  let male_score := male_average * male_count
  let female_score := total_score - male_score
  female_score / female_count = 92 := by
sorry

end NUMINAMATH_CALUDE_algebra_test_average_l1457_145735


namespace NUMINAMATH_CALUDE_quadratic_roots_real_for_pure_imaginary_k_l1457_145746

theorem quadratic_roots_real_for_pure_imaginary_k :
  ∀ (k : ℂ), (∃ (r : ℝ), k = r * I) →
  ∃ (z₁ z₂ : ℝ), (5 : ℂ) * (z₁ : ℂ)^2 + 7 * I * (z₁ : ℂ) - k = 0 ∧
                 (5 : ℂ) * (z₂ : ℂ)^2 + 7 * I * (z₂ : ℂ) - k = 0 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_real_for_pure_imaginary_k_l1457_145746


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1457_145754

theorem sum_of_cubes (x y z : ℝ) 
  (sum_eq : x + y + z = 7)
  (sum_prod_eq : x*y + x*z + y*z = 9)
  (prod_eq : x*y*z = -18) : 
  x^3 + y^3 + z^3 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1457_145754


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1457_145728

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (ax^2 - 1/x)^6
def coefficient (a : ℝ) : ℝ := -a^3 * binomial 6 3

-- Theorem statement
theorem expansion_coefficient (a : ℝ) : coefficient a = 160 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1457_145728


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_points_l1457_145717

theorem smallest_distance_between_complex_points (z w : ℂ) :
  Complex.abs (z - (2 + 4*I)) = 2 →
  Complex.abs (w - (5 + 5*I)) = 4 →
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 10 + 6 ∧
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 + 4*I)) = 2 →
                   Complex.abs (w' - (5 + 5*I)) = 4 →
                   Complex.abs (z' - w') ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_points_l1457_145717


namespace NUMINAMATH_CALUDE_zoom_video_glitch_duration_l1457_145770

theorem zoom_video_glitch_duration :
  let mac_download_time : ℕ := 10
  let windows_download_time : ℕ := 3 * mac_download_time
  let total_download_time : ℕ := mac_download_time + windows_download_time
  let total_time : ℕ := 82
  let call_time : ℕ := total_time - total_download_time
  let audio_glitch_time : ℕ := 2 * 4
  let video_glitch_time : ℕ := call_time - (audio_glitch_time + 2 * (audio_glitch_time + video_glitch_time))
  video_glitch_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_zoom_video_glitch_duration_l1457_145770


namespace NUMINAMATH_CALUDE_margaux_lending_problem_l1457_145783

/-- Margaux's money lending problem -/
theorem margaux_lending_problem (brother_payment cousin_payment total_days total_collection : ℕ) 
  (h1 : brother_payment = 8)
  (h2 : cousin_payment = 4)
  (h3 : total_days = 7)
  (h4 : total_collection = 119) :
  ∃ (friend_payment : ℕ), 
    friend_payment * total_days + brother_payment * total_days + cousin_payment * total_days = total_collection ∧ 
    friend_payment = 5 := by
  sorry

end NUMINAMATH_CALUDE_margaux_lending_problem_l1457_145783


namespace NUMINAMATH_CALUDE_succeeding_number_in_base_3_l1457_145759

def base_3_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (3^i)) 0

def decimal_to_base_3 (n : Nat) : List Nat :=
  sorry  -- Implementation not provided as it's not needed for the statement

def M : List Nat := [0, 2, 0, 1]  -- Representing 1020 in base 3

theorem succeeding_number_in_base_3 :
  decimal_to_base_3 (base_3_to_decimal M + 1) = [1, 2, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_succeeding_number_in_base_3_l1457_145759


namespace NUMINAMATH_CALUDE_orthogonal_vectors_sum_magnitude_l1457_145730

/-- Prove that given planar vectors a and b, where a and b are orthogonal, 
    a = (-1, 1), and |b| = 1, |a + 2b| = √6. -/
theorem orthogonal_vectors_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0)
  (h_a : a = (-1, 1))
  (h_b_norm : Real.sqrt (b.1^2 + b.2^2) = 1) :
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_sum_magnitude_l1457_145730


namespace NUMINAMATH_CALUDE_cosine_identity_problem_l1457_145792

theorem cosine_identity_problem (α : Real) 
  (h : Real.cos (π / 4 + α) = -1 / 3) : 
  (Real.sin (2 * α) - 2 * Real.sin α ^ 2) / Real.sqrt (1 - Real.cos (2 * α)) = 2 / 3 ∨ 
  (Real.sin (2 * α) - 2 * Real.sin α ^ 2) / Real.sqrt (1 - Real.cos (2 * α)) = -2 / 3 :=
sorry

end NUMINAMATH_CALUDE_cosine_identity_problem_l1457_145792


namespace NUMINAMATH_CALUDE_basketball_contest_l1457_145797

/-- Calculates the total points scored in a basketball contest --/
def total_points (layups dunks free_throws three_pointers alley_oops half_court consecutive : ℕ) : ℕ :=
  layups + dunks + 2 * free_throws + 3 * three_pointers + 4 * alley_oops + 5 * half_court + consecutive

/-- Represents the basketball contest between Reggie and his brother --/
theorem basketball_contest :
  let reggie_points := total_points 4 2 3 2 1 1 2
  let brother_points := total_points 3 1 2 5 2 4 3
  brother_points - reggie_points = 25 := by
  sorry

end NUMINAMATH_CALUDE_basketball_contest_l1457_145797


namespace NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l1457_145715

theorem shortest_altitude_right_triangle :
  ∀ (a b c h : ℝ),
  a = 8 ∧ b = 15 ∧ c = 17 →
  a^2 + b^2 = c^2 →
  h = (2 * (a * b) / 2) / c →
  h = 120 / 17 ∧ 
  (∀ h' : ℝ, (h' = a ∨ h' = b ∨ h' = h) → h ≤ h') :=
by sorry

end NUMINAMATH_CALUDE_shortest_altitude_right_triangle_l1457_145715


namespace NUMINAMATH_CALUDE_extreme_value_and_intersection_l1457_145764

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (a - Real.log x) / x

def g (x : ℝ) : ℝ := -1

theorem extreme_value_and_intersection (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ x ≤ Real.exp 1 ∧ f a x = g x) →
  (∀ (x : ℝ), x > 0 → f a x ≥ -Real.exp (-a - 1)) ∧
  (f a (Real.exp (a + 1)) = -Real.exp (-a - 1)) ∧
  (a ≤ -1 ∨ (0 ≤ a ∧ a ≤ Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_and_intersection_l1457_145764


namespace NUMINAMATH_CALUDE_cube_root_function_l1457_145734

/-- Given a function y = kx^(1/3) where y = 4√3 when x = 64, prove that y = 2√3 when x = 8 -/
theorem cube_root_function (k : ℝ) (y : ℝ → ℝ) :
  (∀ x, y x = k * x^(1/3)) →
  y 64 = 4 * Real.sqrt 3 →
  y 8 = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cube_root_function_l1457_145734
