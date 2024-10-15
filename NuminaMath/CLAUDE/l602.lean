import Mathlib

namespace NUMINAMATH_CALUDE_max_sum_constrained_l602_60232

theorem max_sum_constrained (x y z : ℝ) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_constraint : x^2 + y^2 + z^2 + x + 2*y + 3*z = 13/4) :
  x + y + z ≤ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_constrained_l602_60232


namespace NUMINAMATH_CALUDE_max_brownie_pieces_l602_60208

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The pan dimensions -/
def pan : Rectangle := { length := 24, width := 20 }

/-- The brownie piece dimensions -/
def piece : Rectangle := { length := 4, width := 3 }

/-- Theorem: The maximum number of brownie pieces that can be cut from the pan is 40 -/
theorem max_brownie_pieces : (area pan) / (area piece) = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_brownie_pieces_l602_60208


namespace NUMINAMATH_CALUDE_number_relations_l602_60233

theorem number_relations :
  (∃ x : ℤ, x = -2 - 4 ∧ x = -6) ∧
  (∃ y : ℤ, y = -5 + 3 ∧ y = -2) := by
  sorry

end NUMINAMATH_CALUDE_number_relations_l602_60233


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l602_60282

theorem product_of_three_numbers (x y z n : ℝ) 
  (sum_eq : x + y + z = 255)
  (n_eq_8x : n = 8 * x)
  (n_eq_y_minus_11 : n = y - 11)
  (n_eq_z_plus_13 : n = z + 13) :
  x * y * z = 209805 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l602_60282


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l602_60209

theorem least_n_satisfying_inequality : 
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (1 : ℚ) / m - (1 : ℚ) / (m + 1) < (1 : ℚ) / 15 → m ≥ n) ∧
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l602_60209


namespace NUMINAMATH_CALUDE_fraction_sum_simplest_form_fraction_simplest_form_l602_60203

theorem fraction_sum_simplest_form : (7 : ℚ) / 12 + (8 : ℚ) / 15 = (67 : ℚ) / 60 := by
  sorry

theorem fraction_simplest_form : (67 : ℚ) / 60 = (67 : ℚ) / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplest_form_fraction_simplest_form_l602_60203


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l602_60291

theorem sin_40_tan_10_minus_sqrt_3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l602_60291


namespace NUMINAMATH_CALUDE_expression_change_l602_60285

theorem expression_change (x a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ y => y^2 - 5*y
  (f (x + a) - f x = 2*a*x + a^2 - 5*a) ∧ 
  (f (x - a) - f x = -2*a*x + a^2 + 5*a) :=
sorry

end NUMINAMATH_CALUDE_expression_change_l602_60285


namespace NUMINAMATH_CALUDE_trig_identity_l602_60202

theorem trig_identity (α : ℝ) : 
  (2 * (Real.cos (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) / 
  (2 * (Real.sin (2 * α))^2 + Real.sqrt 3 * Real.sin (4 * α) - 1) = 
  Real.sin (4 * α + π/6) / Real.sin (4 * α - π/6) := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l602_60202


namespace NUMINAMATH_CALUDE_function_upper_bound_l602_60200

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)

/-- A function bounded on [0,1] -/
def BoundedOnUnitInterval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x → x ≤ 1 → |f x| ≤ 1997

/-- The main theorem -/
theorem function_upper_bound
  (f : ℝ → ℝ)
  (h1 : SatisfiesInequality f)
  (h2 : BoundedOnUnitInterval f) :
  ∀ x : ℝ, x ≥ 0 → f x ≤ x^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_function_upper_bound_l602_60200


namespace NUMINAMATH_CALUDE_jogging_time_difference_fathers_jogging_time_saved_l602_60275

/-- Calculates the time difference in minutes between jogging at varying speeds and a constant speed -/
theorem jogging_time_difference (distance : ℝ) (constant_speed : ℝ) 
  (speeds : List ℝ) : ℝ :=
  let varying_time := (speeds.map (λ s => distance / s)).sum
  let constant_time := speeds.length * (distance / constant_speed)
  (varying_time - constant_time) * 60

/-- Proves that the time difference for the given scenario is 3 minutes -/
theorem fathers_jogging_time_saved : 
  jogging_time_difference 3 5 [6, 5, 4, 5] = 3 := by
  sorry

end NUMINAMATH_CALUDE_jogging_time_difference_fathers_jogging_time_saved_l602_60275


namespace NUMINAMATH_CALUDE_divisible_by_5040_l602_60274

theorem divisible_by_5040 (n : ℤ) (h : n > 3) : 
  ∃ k : ℤ, n^7 - 14*n^5 + 49*n^3 - 36*n = 5040 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_5040_l602_60274


namespace NUMINAMATH_CALUDE_added_number_problem_l602_60280

theorem added_number_problem (initial_count : ℕ) (initial_avg : ℚ) (new_avg : ℚ) : 
  initial_count = 6 →
  initial_avg = 24 →
  new_avg = 25 →
  ∃ x : ℚ, (initial_count * initial_avg + x) / (initial_count + 1) = new_avg ∧ x = 31 :=
by sorry

end NUMINAMATH_CALUDE_added_number_problem_l602_60280


namespace NUMINAMATH_CALUDE_cow_heart_ratio_l602_60215

/-- The number of hearts on a standard deck of 52 playing cards -/
def hearts_on_deck : ℕ := 13

/-- The cost of each cow in dollars -/
def cost_per_cow : ℕ := 200

/-- The total cost of all cows in dollars -/
def total_cost : ℕ := 83200

/-- The number of cows in Devonshire -/
def num_cows : ℕ := total_cost / cost_per_cow

theorem cow_heart_ratio :
  num_cows / hearts_on_deck = 32 :=
sorry

end NUMINAMATH_CALUDE_cow_heart_ratio_l602_60215


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l602_60237

/-- The number of markers bought by the shopkeeper -/
def total_markers : ℕ := 2000

/-- The cost price of each marker in dollars -/
def cost_price : ℚ := 3/10

/-- The selling price of each marker in dollars -/
def selling_price : ℚ := 11/20

/-- The target profit in dollars -/
def target_profit : ℚ := 150

/-- The number of markers that need to be sold to achieve the target profit -/
def markers_to_sell : ℕ := 1364

theorem shopkeeper_profit :
  (markers_to_sell : ℚ) * selling_price - (total_markers : ℚ) * cost_price = target_profit :=
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l602_60237


namespace NUMINAMATH_CALUDE_lcm_gcd_product_equality_l602_60244

theorem lcm_gcd_product_equality (a b : ℕ) (h : Nat.lcm a b + Nat.gcd a b = a * b) : a = 2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_equality_l602_60244


namespace NUMINAMATH_CALUDE_bicycle_shop_optimal_plan_l602_60269

/-- Represents the purchase plan for bicycles -/
structure BicyclePlan where
  modelA : ℕ
  modelB : ℕ

/-- The bicycle shop problem -/
theorem bicycle_shop_optimal_plan :
  ∀ (plan : BicyclePlan),
  plan.modelA + plan.modelB = 50 →
  plan.modelB ≥ plan.modelA →
  1000 * plan.modelA + 1600 * plan.modelB ≤ 68000 →
  ∃ (optimalPlan : BicyclePlan),
  optimalPlan.modelA = 25 ∧
  optimalPlan.modelB = 25 ∧
  ∀ (p : BicyclePlan),
  p.modelA + p.modelB = 50 →
  p.modelB ≥ p.modelA →
  1000 * p.modelA + 1600 * p.modelB ≤ 68000 →
  500 * p.modelA + 400 * p.modelB ≤ 500 * optimalPlan.modelA + 400 * optimalPlan.modelB ∧
  500 * optimalPlan.modelA + 400 * optimalPlan.modelB = 22500 :=
by
  sorry


end NUMINAMATH_CALUDE_bicycle_shop_optimal_plan_l602_60269


namespace NUMINAMATH_CALUDE_solve_equation_l602_60276

theorem solve_equation (x : ℚ) (h : x / 4 - x - 3 / 6 = 1) : x = -14/9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l602_60276


namespace NUMINAMATH_CALUDE_y_change_when_x_increases_l602_60278

/-- Regression equation: y = 3 - 5x -/
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

/-- Theorem: When x increases by 1, y decreases by 5 -/
theorem y_change_when_x_increases (x : ℝ) :
  regression_equation (x + 1) = regression_equation x - 5 := by
  sorry

end NUMINAMATH_CALUDE_y_change_when_x_increases_l602_60278


namespace NUMINAMATH_CALUDE_derivative_of_odd_is_even_l602_60251

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem derivative_of_odd_is_even
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) (hodd : OddFunction f) :
  OddFunction f → ∀ x, deriv f (-x) = deriv f x :=
sorry

end NUMINAMATH_CALUDE_derivative_of_odd_is_even_l602_60251


namespace NUMINAMATH_CALUDE_big_stack_orders_l602_60245

/-- The number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- The number of pancakes in a short stack -/
def short_stack : ℕ := 3

/-- The number of customers who ordered the short stack -/
def short_stack_orders : ℕ := 9

/-- The total number of pancakes needed -/
def total_pancakes : ℕ := 57

/-- Theorem stating that the number of customers who ordered the big stack is 6 -/
theorem big_stack_orders : ℕ := by
  sorry

end NUMINAMATH_CALUDE_big_stack_orders_l602_60245


namespace NUMINAMATH_CALUDE_one_intersection_point_condition_l602_60277

open Real

noncomputable def f (x : ℝ) : ℝ := x + log x - 2 / Real.exp 1

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m / x

theorem one_intersection_point_condition (m : ℝ) :
  (∃! x, f x = g m x) →
  (m ≥ 0 ∨ m = -(Real.exp 1 + 1) / (Real.exp 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_one_intersection_point_condition_l602_60277


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l602_60252

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l602_60252


namespace NUMINAMATH_CALUDE_coeff_x3_is_30_l602_60223

/-- The coefficient of x^3 in the expansion of (2x-1)(1/x + x)^6 -/
def coeff_x3 : ℤ := 30

/-- The expression (2x-1)(1/x + x)^6 -/
def expression (x : ℚ) : ℚ := (2*x - 1) * (1/x + x)^6

theorem coeff_x3_is_30 : coeff_x3 = 30 := by sorry

end NUMINAMATH_CALUDE_coeff_x3_is_30_l602_60223


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l602_60221

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define symmetry with respect to the origin
def symmetricToOrigin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

-- Theorem statement
theorem symmetric_point_coordinates :
  let A : Point3D := { x := 2, y := 1, z := 0 }
  let B : Point3D := symmetricToOrigin A
  B.x = -2 ∧ B.y = -1 ∧ B.z = 0 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l602_60221


namespace NUMINAMATH_CALUDE_surface_area_after_corner_removal_l602_60227

/-- The surface area of a cube after removing smaller cubes from its corners --/
theorem surface_area_after_corner_removal (edge_length original_cube_edge : ℝ) 
  (h1 : original_cube_edge = 4)
  (h2 : edge_length = 2) :
  6 * original_cube_edge^2 = 
  6 * original_cube_edge^2 - 8 * (3 * edge_length^2 - 3 * edge_length^2) :=
by sorry

end NUMINAMATH_CALUDE_surface_area_after_corner_removal_l602_60227


namespace NUMINAMATH_CALUDE_xyz_value_l602_60254

theorem xyz_value (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xy : x * y = 40 * Real.rpow 4 (1/3))
  (h_xz : x * z = 56 * Real.rpow 4 (1/3))
  (h_yz : y * z = 32 * Real.rpow 4 (1/3))
  (h_sum : x + y = 18) :
  x * y * z = 16 * Real.sqrt 895 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l602_60254


namespace NUMINAMATH_CALUDE_prob_sum_24_four_dice_is_correct_l602_60213

/-- The probability of rolling a sum of 24 with four fair, six-sided dice -/
def prob_sum_24_four_dice : ℚ := 1 / 1296

/-- The number of sides on each die -/
def sides_per_die : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The sum we're looking for -/
def target_sum : ℕ := 24

/-- Theorem: The probability of rolling a sum of 24 with four fair, six-sided dice is 1/1296 -/
theorem prob_sum_24_four_dice_is_correct : 
  prob_sum_24_four_dice = (1 : ℚ) / sides_per_die ^ num_dice ∧
  target_sum = sides_per_die * num_dice :=
sorry

end NUMINAMATH_CALUDE_prob_sum_24_four_dice_is_correct_l602_60213


namespace NUMINAMATH_CALUDE_perpendicular_sequence_limit_l602_60217

/-- An equilateral triangle ABC with a sequence of points Pₙ on AB defined by perpendicular constructions --/
structure PerpendicularSequence where
  /-- The side length of the equilateral triangle --/
  a : ℝ
  /-- The sequence of distances BPₙ --/
  bp : ℕ → ℝ
  /-- The initial point P₁ is on AB --/
  h_initial : 0 ≤ bp 1 ∧ bp 1 ≤ a
  /-- The recurrence relation for the sequence --/
  h_recurrence : ∀ n, bp (n + 1) = 3/4 * a - 1/8 * bp n

/-- The limit of the perpendicular sequence converges to 2/3 of the side length --/
theorem perpendicular_sequence_limit (ps : PerpendicularSequence) :
  ∃ L, L = 2/3 * ps.a ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N, |ps.bp n - L| < ε :=
sorry

end NUMINAMATH_CALUDE_perpendicular_sequence_limit_l602_60217


namespace NUMINAMATH_CALUDE_least_product_consecutive_primes_above_50_l602_60262

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def consecutive_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ ∀ r : ℕ, is_prime r → (p < r → r < q) → r = p ∨ r = q

theorem least_product_consecutive_primes_above_50 :
  ∃ p q : ℕ, consecutive_primes p q ∧ p > 50 ∧ q > 50 ∧
  p * q = 3127 ∧
  ∀ a b : ℕ, consecutive_primes a b → a > 50 → b > 50 → a * b ≥ 3127 :=
sorry

end NUMINAMATH_CALUDE_least_product_consecutive_primes_above_50_l602_60262


namespace NUMINAMATH_CALUDE_tangent_midpoint_parallel_l602_60255

-- Define the ellipses C and T
def ellipse_C (x y : ℝ) : Prop := x^2/18 + y^2/2 = 1
def ellipse_T (x y : ℝ) : Prop := x^2/9 + y^2 = 1

-- Define a point on an ellipse
def point_on_ellipse (E : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  E P.1 P.2

-- Define a tangent line from a point to an ellipse
def is_tangent (P M : ℝ × ℝ) (E : ℝ → ℝ → Prop) : Prop :=
  point_on_ellipse E M ∧ 
  ∀ Q, point_on_ellipse E Q → (Q ≠ M → (Q.2 - P.2) * (M.1 - P.1) ≠ (Q.1 - P.1) * (M.2 - P.2))

-- Define parallel lines
def parallel (P₁ P₂ Q₁ Q₂ : ℝ × ℝ) : Prop :=
  (P₂.2 - P₁.2) * (Q₂.1 - Q₁.1) = (P₂.1 - P₁.1) * (Q₂.2 - Q₁.2)

theorem tangent_midpoint_parallel :
  ∀ P G H M N : ℝ × ℝ,
    point_on_ellipse ellipse_C P →
    point_on_ellipse ellipse_C G →
    point_on_ellipse ellipse_C H →
    is_tangent P M ellipse_T →
    is_tangent P N ellipse_T →
    G ≠ P →
    H ≠ P →
    (G.2 - P.2) * (M.1 - P.1) = (G.1 - P.1) * (M.2 - P.2) →
    (H.2 - P.2) * (N.1 - P.1) = (H.1 - P.1) * (N.2 - P.2) →
    parallel M N G H :=
by sorry

end NUMINAMATH_CALUDE_tangent_midpoint_parallel_l602_60255


namespace NUMINAMATH_CALUDE_grid_path_problem_l602_60281

/-- The number of paths on a grid from (0,0) to (m,n) with exactly k steps -/
def grid_paths (m n k : ℕ) : ℕ := Nat.choose k m

/-- The problem statement -/
theorem grid_path_problem :
  let m : ℕ := 7  -- width of the grid
  let n : ℕ := 5  -- height of the grid
  let k : ℕ := 10 -- total number of steps
  grid_paths m n k = 120 := by sorry

end NUMINAMATH_CALUDE_grid_path_problem_l602_60281


namespace NUMINAMATH_CALUDE_liquid_x_percentage_in_mixed_solution_l602_60247

/-- The percentage of liquid X in the resulting solution after mixing two solutions. -/
theorem liquid_x_percentage_in_mixed_solution
  (percent_x_in_a : ℝ)
  (percent_x_in_b : ℝ)
  (weight_a : ℝ)
  (weight_b : ℝ)
  (h1 : percent_x_in_a = 0.8)
  (h2 : percent_x_in_b = 1.8)
  (h3 : weight_a = 600)
  (h4 : weight_b = 700) :
  let weight_x_in_a := percent_x_in_a / 100 * weight_a
  let weight_x_in_b := percent_x_in_b / 100 * weight_b
  let total_weight_x := weight_x_in_a + weight_x_in_b
  let total_weight := weight_a + weight_b
  let percent_x_in_mixed := total_weight_x / total_weight * 100
  ∃ ε > 0, |percent_x_in_mixed - 1.34| < ε :=
sorry

end NUMINAMATH_CALUDE_liquid_x_percentage_in_mixed_solution_l602_60247


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l602_60257

theorem fraction_zero_implies_x_negative_one (x : ℝ) : 
  (1 - |x|) / (1 - x) = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l602_60257


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l602_60266

/-- Product of digits of a two-digit number -/
def P (n : ℕ) : ℕ := (n / 10) * (n % 10)

/-- Sum of digits of a two-digit number -/
def S (n : ℕ) : ℕ := (n / 10) + (n % 10)

/-- A two-digit number is between 10 and 99, inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem unique_two_digit_number :
  ∃! M : ℕ, is_two_digit M ∧ M = P M + S M + 1 :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l602_60266


namespace NUMINAMATH_CALUDE_sum_and_convert_l602_60219

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Adds two numbers in base 8 -/
def add_base8 (a b : ℕ) : ℕ := sorry

theorem sum_and_convert :
  let a := 1453
  let b := 567
  base8_to_base10 (add_base8 a b) = 1124 := by sorry

end NUMINAMATH_CALUDE_sum_and_convert_l602_60219


namespace NUMINAMATH_CALUDE_problem_solution_l602_60299

def p : Prop := 0 % 2 = 0
def q : Prop := ∃ k : ℤ, 3 = 2 * k

theorem problem_solution : p ∨ q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l602_60299


namespace NUMINAMATH_CALUDE_books_read_ratio_l602_60243

theorem books_read_ratio : 
  let william_last_month : ℕ := 6
  let brad_this_month : ℕ := 8
  let william_this_month : ℕ := 2 * brad_this_month
  let william_total : ℕ := william_last_month + william_this_month
  let brad_total : ℕ := william_total - 4
  let brad_last_month : ℕ := brad_total - brad_this_month
  ∃ (a b : ℕ), a * william_last_month = b * brad_last_month ∧ a = 3 ∧ b = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_books_read_ratio_l602_60243


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l602_60231

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 26 ∧ 
  (99 ∣ (12702 - x)) ∧ 
  (∀ (y : ℕ), y < x → ¬(99 ∣ (12702 - y))) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l602_60231


namespace NUMINAMATH_CALUDE_ladies_walking_distance_l602_60226

/-- The total distance walked by a group of ladies over a period of days. -/
def total_distance (
  group_size : ℕ
  ) (
  group_distance : ℝ
  ) (
  jamie_extra : ℝ
  ) (
  sue_extra : ℝ
  ) (
  days : ℕ
  ) : ℝ :=
  group_size * group_distance * days + jamie_extra * days + sue_extra * days

/-- Proof that the total distance walked by the ladies is 36 miles. -/
theorem ladies_walking_distance :
  let group_size : ℕ := 5
  let group_distance : ℝ := 3
  let jamie_extra : ℝ := 2
  let sue_extra : ℝ := jamie_extra / 2
  let days : ℕ := 6
  total_distance group_size group_distance jamie_extra sue_extra days = 36 := by
  sorry


end NUMINAMATH_CALUDE_ladies_walking_distance_l602_60226


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l602_60273

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N : M ∩ N = {(3, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l602_60273


namespace NUMINAMATH_CALUDE_chocolate_distribution_problem_l602_60295

/-- The number of ways to distribute n chocolates among k people, 
    with each person receiving at least m chocolates -/
def distribute_chocolates (n k m : ℕ) : ℕ := 
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The problem statement -/
theorem chocolate_distribution_problem : 
  distribute_chocolates 30 3 3 = 253 := by sorry

end NUMINAMATH_CALUDE_chocolate_distribution_problem_l602_60295


namespace NUMINAMATH_CALUDE_complex_number_modulus_l602_60241

theorem complex_number_modulus (z : ℂ) : z = -5 + 12 * Complex.I → Complex.abs z = 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l602_60241


namespace NUMINAMATH_CALUDE_sum_a_c_equals_five_l602_60292

theorem sum_a_c_equals_five 
  (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 40) 
  (h2 : b + d = 8) : 
  a + c = 5 := by sorry

end NUMINAMATH_CALUDE_sum_a_c_equals_five_l602_60292


namespace NUMINAMATH_CALUDE_sum_of_threes_place_values_63130_l602_60286

def number : ℕ := 63130

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def sum_of_threes_place_values (n : ℕ) : ℕ :=
  hundreds_digit n * 100 + tens_digit n * 10

theorem sum_of_threes_place_values_63130 :
  sum_of_threes_place_values number = 330 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_threes_place_values_63130_l602_60286


namespace NUMINAMATH_CALUDE_proposition_relations_l602_60261

-- Define the original proposition
def p (a : ℝ) : Prop := a > 0 → a^2 ≠ 0

-- Define the converse
def converse (a : ℝ) : Prop := a^2 ≠ 0 → a > 0

-- Define the inverse
def inverse (a : ℝ) : Prop := ¬(a > 0) → a^2 = 0

-- Define the contrapositive
def contrapositive (a : ℝ) : Prop := a^2 = 0 → ¬(a > 0)

-- Define the negation
def negation : Prop := ∃ a : ℝ, a > 0 ∧ a^2 = 0

-- Theorem stating the truth values of each related proposition
theorem proposition_relations :
  (∃ a : ℝ, ¬(converse a)) ∧
  (∃ a : ℝ, ¬(inverse a)) ∧
  (∀ a : ℝ, contrapositive a) ∧
  ¬negation :=
sorry

end NUMINAMATH_CALUDE_proposition_relations_l602_60261


namespace NUMINAMATH_CALUDE_nearest_integer_to_power_l602_60264

theorem nearest_integer_to_power : 
  ∃ n : ℤ, n = 3936 ∧ ∀ m : ℤ, |((3:ℝ) + Real.sqrt 5)^5 - (n:ℝ)| ≤ |((3:ℝ) + Real.sqrt 5)^5 - (m:ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_power_l602_60264


namespace NUMINAMATH_CALUDE_auditorium_sampling_is_systematic_l602_60288

/-- Represents a sampling method --/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents an auditorium with a given number of rows and seats per row --/
structure Auditorium where
  rows : ℕ
  seatsPerRow : ℕ

/-- Represents a sampling strategy --/
structure SamplingStrategy where
  interval : ℕ
  startingSeat : ℕ

/-- Determines if a sampling strategy is systematic --/
def isSystematicSampling (strategy : SamplingStrategy) : Prop :=
  strategy.interval > 0 ∧ strategy.startingSeat > 0 ∧ strategy.startingSeat ≤ strategy.interval

/-- The theorem to be proved --/
theorem auditorium_sampling_is_systematic 
  (auditorium : Auditorium) 
  (strategy : SamplingStrategy) : 
  auditorium.rows = 25 → 
  auditorium.seatsPerRow = 20 → 
  strategy.interval = auditorium.seatsPerRow → 
  strategy.startingSeat = 15 → 
  isSystematicSampling strategy ∧ 
  SamplingMethod.Systematic = SamplingMethod.Systematic :=
sorry


end NUMINAMATH_CALUDE_auditorium_sampling_is_systematic_l602_60288


namespace NUMINAMATH_CALUDE_cube_diagonal_l602_60205

theorem cube_diagonal (s : ℝ) (h : s > 0) (eq : s^3 + 36*s = 12*s^2) : 
  Real.sqrt (3 * s^2) = 6 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cube_diagonal_l602_60205


namespace NUMINAMATH_CALUDE_min_questions_to_determine_l602_60218

def questions_to_determine (x : ℕ) : ℕ :=
  if x ≥ 10 ∧ x ≤ 19 then
    if x ≤ 14 then
      if x ≤ 12 then
        if x = 11 then 3 else 3
      else
        if x = 13 then 3 else 3
    else
      if x ≤ 17 then
        if x ≤ 16 then
          if x = 15 then 4 else 4
        else 3
      else
        if x = 18 then 3 else 3
  else 0

theorem min_questions_to_determine :
  ∀ x : ℕ, x ≥ 10 ∧ x ≤ 19 → questions_to_determine x ≤ 3 ∧
  (∀ y : ℕ, y ≥ 10 ∧ y ≤ 19 ∧ y ≠ x → ∃ q : ℕ, q < questions_to_determine x ∧
    (∀ z : ℕ, z ≥ 10 ∧ z ≤ 19 → questions_to_determine z < q → z ≠ x ∧ z ≠ y)) :=
sorry

end NUMINAMATH_CALUDE_min_questions_to_determine_l602_60218


namespace NUMINAMATH_CALUDE_circle_configuration_diameter_l602_60207

/-- Given a configuration of circles as described, prove the diameter length --/
theorem circle_configuration_diameter : 
  ∀ (r s : ℝ) (shaded_area circle_c_area : ℝ),
  r > 0 → s > 0 →
  shaded_area = 39 * Real.pi →
  circle_c_area = 9 * Real.pi →
  shaded_area = (Real.pi / 2) * ((r + s)^2 - r^2 - s^2) - circle_c_area →
  2 * (r + s) = 32 := by
  sorry

end NUMINAMATH_CALUDE_circle_configuration_diameter_l602_60207


namespace NUMINAMATH_CALUDE_added_amount_l602_60259

theorem added_amount (x : ℝ) (y : ℝ) : 
  x = 15 → 3 * (2 * x + y) = 105 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_added_amount_l602_60259


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l602_60210

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) : 
  (180 * (n - 2) : ℝ) = 156 * n ↔ n = 15 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l602_60210


namespace NUMINAMATH_CALUDE_field_trip_fraction_l602_60263

theorem field_trip_fraction (b : ℚ) (g : ℚ) : 
  g = 2 * b →  -- There are twice as many girls as boys
  (2 / 3 * g + 3 / 5 * b) ≠ 0 → -- Total students on trip is not zero
  (2 / 3 * g) / (2 / 3 * g + 3 / 5 * b) = 20 / 29 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_fraction_l602_60263


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l602_60216

theorem largest_multiple_of_15_under_500 :
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l602_60216


namespace NUMINAMATH_CALUDE_log_inequality_l602_60222

theorem log_inequality (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) : 
  Real.log (Real.sqrt (x₁ * x₂)) = (Real.log x₁ + Real.log x₂) / 2 ∧
  Real.log (Real.sqrt (x₁ * x₂)) < Real.log ((x₁ + x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l602_60222


namespace NUMINAMATH_CALUDE_constant_value_proof_l602_60211

/-- The coefficient of x in the expansion of (x - a/x)(1 - √x)^6 -/
def coefficient_of_x (a : ℝ) : ℝ := 1 - 15 * a

/-- The theorem stating that a = -2 when the coefficient of x is 31 -/
theorem constant_value_proof (a : ℝ) : coefficient_of_x a = 31 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_proof_l602_60211


namespace NUMINAMATH_CALUDE_system_solution_exists_l602_60284

theorem system_solution_exists (m : ℝ) :
  (∃ x y : ℝ, y = (m + 1) * x + 2 ∧ y = (3 * m - 2) * x + 5 ∧ y > 5) ↔ m ≠ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_l602_60284


namespace NUMINAMATH_CALUDE_halfway_between_one_seventh_and_one_ninth_l602_60240

theorem halfway_between_one_seventh_and_one_ninth :
  (1 / 7 + 1 / 9) / 2 = 8 / 63 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_seventh_and_one_ninth_l602_60240


namespace NUMINAMATH_CALUDE_circle_condition_l602_60258

theorem circle_condition (x y m : ℝ) : 
  (∃ (a b r : ℝ), r > 0 ∧ (x - a)^2 + (y - b)^2 = r^2 ↔ x^2 + y^2 - x + y + m = 0) → 
  m < (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_circle_condition_l602_60258


namespace NUMINAMATH_CALUDE_helmet_safety_analysis_l602_60287

-- Define the data types
structure YearData where
  year_number : ℕ
  not_wearing_helmets : ℕ

-- Define the data for 4 years
def year_data : List YearData := [
  ⟨1, 1250⟩,
  ⟨2, 1050⟩,
  ⟨3, 1000⟩,
  ⟨4, 900⟩
]

-- Define the contingency table
structure ContingencyTable where
  injured_not_wearing : ℕ
  injured_wearing : ℕ
  not_injured_not_wearing : ℕ
  not_injured_wearing : ℕ

def accident_data : ContingencyTable := ⟨7, 3, 13, 27⟩

-- Define the theorem
theorem helmet_safety_analysis :
  -- Regression line equation
  let b : ℚ := -110
  let a : ℚ := 1325
  let regression_line (x : ℚ) := b * x + a

  -- Estimated number of people not wearing helmets in 2022
  let estimate_2022 : ℕ := 775

  -- Chi-square statistic
  let chi_square : ℚ := 4.6875
  let critical_value : ℚ := 3.841

  -- Theorem statements
  (∀ (x : ℚ), regression_line x = b * x + a) ∧
  (regression_line 5 = estimate_2022) ∧
  (chi_square > critical_value) := by
  sorry


end NUMINAMATH_CALUDE_helmet_safety_analysis_l602_60287


namespace NUMINAMATH_CALUDE_tom_hockey_games_attendance_l602_60212

/-- The number of hockey games Tom attended over six years -/
def total_games_attended (year1 year2 year3 year4 year5 year6 : ℕ) : ℕ :=
  year1 + year2 + year3 + year4 + year5 + year6

/-- Theorem stating that Tom attended 41 hockey games over six years -/
theorem tom_hockey_games_attendance :
  total_games_attended 4 9 5 10 6 7 = 41 := by
  sorry

end NUMINAMATH_CALUDE_tom_hockey_games_attendance_l602_60212


namespace NUMINAMATH_CALUDE_subtraction_of_large_numbers_l602_60201

theorem subtraction_of_large_numbers :
  2222222222222 - 1111111111111 = 1111111111111 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_large_numbers_l602_60201


namespace NUMINAMATH_CALUDE_connect_four_shapes_l602_60294

/-- The number of columns in the Connect Four board -/
def num_columns : ℕ := 7

/-- The number of rows in the Connect Four board -/
def num_rows : ℕ := 8

/-- The number of possible states for each column (0 to 8 checkers) -/
def states_per_column : ℕ := num_rows + 1

/-- The total number of shapes before accounting for symmetry -/
def total_shapes : ℕ := states_per_column ^ num_columns

/-- The number of symmetric shapes -/
def symmetric_shapes : ℕ := states_per_column ^ (num_columns / 2 + 1)

/-- The formula for the number of distinct shapes -/
def distinct_shapes (n : ℕ) : ℕ := 9 * (n * (n + 1) / 2)

/-- The theorem stating that the number of distinct shapes is equal to 9(1+2+...+729) -/
theorem connect_four_shapes :
  ∃ n : ℕ, distinct_shapes n = symmetric_shapes + (total_shapes - symmetric_shapes) / 2 ∧ n = 729 := by
  sorry

end NUMINAMATH_CALUDE_connect_four_shapes_l602_60294


namespace NUMINAMATH_CALUDE_abc_problem_l602_60230

def base_6_value (a b : ℕ) : ℕ := a * 6 + b

theorem abc_problem (A B C : ℕ) : 
  (0 < A) → (A ≤ 5) →
  (0 < B) → (B ≤ 5) →
  (0 < C) → (C ≤ 5) →
  (A ≠ B) → (B ≠ C) → (A ≠ C) →
  (base_6_value A B + A = base_6_value B A) →
  (base_6_value A B + B = base_6_value C 1) →
  (A = 5 ∧ B = 5 ∧ C = 1) := by
sorry

end NUMINAMATH_CALUDE_abc_problem_l602_60230


namespace NUMINAMATH_CALUDE_ellipse_k_range_l602_60260

/-- An ellipse represented by the equation x² + ky² = 2 with foci on the y-axis -/
structure Ellipse where
  k : ℝ
  eq : ∀ (x y : ℝ), x^2 + k * y^2 = 2
  foci_on_y_axis : True  -- This is a placeholder for the foci condition

/-- The range of k for an ellipse with the given properties is (0, 1) -/
theorem ellipse_k_range (e : Ellipse) : 0 < e.k ∧ e.k < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l602_60260


namespace NUMINAMATH_CALUDE_quadratic_solution_l602_60242

theorem quadratic_solution (p q : ℝ) :
  let x : ℝ → ℝ := λ y => y - p / 2
  ∀ y, x y * x y + p * x y + q = 0 ↔ y * y = p * p / 4 - q :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l602_60242


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l602_60253

theorem quadratic_equation_condition (m : ℝ) : 
  (m ^ 2 - 7 = 2 ∧ m - 3 ≠ 0) ↔ m = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l602_60253


namespace NUMINAMATH_CALUDE_line_intersection_with_x_axis_l602_60239

/-- Given a line y = kx + b parallel to y = -3x + 1 and passing through (0, -2),
    prove that its intersection with the x-axis is at (-2/3, 0) -/
theorem line_intersection_with_x_axis
  (k b : ℝ) 
  (parallel : k = -3)
  (passes_through : b = -2) :
  let line := λ x : ℝ => k * x + b
  ∃ x : ℝ, line x = 0 ∧ x = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_with_x_axis_l602_60239


namespace NUMINAMATH_CALUDE_A_profit_share_l602_60246

def investment_A : ℕ := 6300
def investment_B : ℕ := 4200
def investment_C : ℕ := 10500

def profit_share_A : ℚ := 45 / 100
def profit_share_B : ℚ := 30 / 100
def profit_share_C : ℚ := 25 / 100

def total_profit : ℕ := 12200

theorem A_profit_share :
  (profit_share_A * total_profit : ℚ) = 5490 := by sorry

end NUMINAMATH_CALUDE_A_profit_share_l602_60246


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l602_60238

theorem quadratic_inequality_range (m : ℝ) (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) →
  (∀ b : ℝ, b > a → m > b) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l602_60238


namespace NUMINAMATH_CALUDE_biggest_number_is_five_l602_60220

def yoongi_number : ℕ := 4
def jungkook_number : ℕ := 6 - 3
def yuna_number : ℕ := 5

theorem biggest_number_is_five :
  max yoongi_number (max jungkook_number yuna_number) = yuna_number :=
by sorry

end NUMINAMATH_CALUDE_biggest_number_is_five_l602_60220


namespace NUMINAMATH_CALUDE_win_sector_area_l602_60283

/-- Given a circular spinner with radius 12 cm and probability of winning 1/4,
    the area of the WIN sector is 36π square centimeters. -/
theorem win_sector_area (r : ℝ) (p : ℝ) (total_area : ℝ) (win_area : ℝ) :
  r = 12 →
  p = 1 / 4 →
  total_area = π * r^2 →
  win_area = p * total_area →
  win_area = 36 * π := by
sorry

end NUMINAMATH_CALUDE_win_sector_area_l602_60283


namespace NUMINAMATH_CALUDE_binomial_expansion_ratio_l602_60248

theorem binomial_expansion_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₃ / a₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_ratio_l602_60248


namespace NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l602_60224

theorem cos_four_arccos_two_fifths :
  Real.cos (4 * Real.arccos (2/5)) = -47/625 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l602_60224


namespace NUMINAMATH_CALUDE_negation_of_both_even_l602_60214

theorem negation_of_both_even (a b : ℤ) :
  ¬(Even a ∧ Even b) ↔ ¬(Even a ∧ Even b) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_both_even_l602_60214


namespace NUMINAMATH_CALUDE_georgia_carnation_cost_l602_60296

/-- The cost of a single carnation in dollars -/
def single_carnation_cost : ℚ := 0.5

/-- The cost of a dozen carnations in dollars -/
def dozen_carnation_cost : ℚ := 4

/-- The number of teachers Georgia wants to send carnations to -/
def number_of_teachers : ℕ := 5

/-- The number of friends Georgia wants to buy carnations for -/
def number_of_friends : ℕ := 14

/-- The total cost of Georgia's carnation purchases -/
def total_cost : ℚ := 
  (number_of_teachers * dozen_carnation_cost) + 
  dozen_carnation_cost + 
  (2 * single_carnation_cost)

/-- Theorem stating that the total cost of Georgia's carnation purchases is $25.00 -/
theorem georgia_carnation_cost : total_cost = 25 := by
  sorry

end NUMINAMATH_CALUDE_georgia_carnation_cost_l602_60296


namespace NUMINAMATH_CALUDE_complement_union_equality_l602_60249

universe u

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {2, 4, 6}

theorem complement_union_equality : 
  (U \ M) ∪ (U \ N) = U := by sorry

end NUMINAMATH_CALUDE_complement_union_equality_l602_60249


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l602_60206

/-- If 49m^2 + km + 1 is a perfect square trinomial, then k = ±14 -/
theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ (a b : ℤ), ∀ m, 49 * m^2 + k * m + 1 = (a * m + b)^2) →
  k = 14 ∨ k = -14 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l602_60206


namespace NUMINAMATH_CALUDE_envelope_equals_cycloid_l602_60272

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a circle rolling along the x-axis -/
structure RollingCircle where
  radius : ℝ
  center : Point2D

/-- Represents a cycloid curve -/
def Cycloid := ℝ → Point2D

/-- Generates the cycloid traced by a point on the circumference of a circle -/
def circumferenceCycloid (radius : ℝ) : Cycloid := sorry

/-- Generates the envelope of a diameter of a rolling circle -/
def diameterEnvelope (radius : ℝ) : Cycloid := sorry

/-- Theorem stating that the envelope of a diameter is identical to the cycloid traced by a point on the circumference -/
theorem envelope_equals_cycloid (a : ℝ) :
  diameterEnvelope a = circumferenceCycloid (a / 2) := by sorry

end NUMINAMATH_CALUDE_envelope_equals_cycloid_l602_60272


namespace NUMINAMATH_CALUDE_natural_pythagorean_triples_real_circle_equation_l602_60271

-- Part 1: Natural numbers
def natural_solutions : Set (ℕ × ℕ) :=
  {(0, 5), (5, 0), (3, 4), (4, 3)}

theorem natural_pythagorean_triples :
  ∀ (x y : ℕ), x^2 + y^2 = 25 ↔ (x, y) ∈ natural_solutions :=
sorry

-- Part 2: Real numbers
def real_solutions : Set (ℝ × ℝ) :=
  {(x, y) | -5 ≤ x ∧ x ≤ 5 ∧ (y = Real.sqrt (25 - x^2) ∨ y = -Real.sqrt (25 - x^2))}

theorem real_circle_equation :
  ∀ (x y : ℝ), x^2 + y^2 = 25 ↔ (x, y) ∈ real_solutions :=
sorry

end NUMINAMATH_CALUDE_natural_pythagorean_triples_real_circle_equation_l602_60271


namespace NUMINAMATH_CALUDE_discount_calculation_l602_60234

/-- The discount received when buying multiple parts with a given original price,
    number of parts, and final price paid. -/
def discount (original_price : ℕ) (num_parts : ℕ) (final_price : ℕ) : ℕ :=
  original_price * num_parts - final_price

/-- Theorem stating that the discount is $121 given the problem conditions. -/
theorem discount_calculation :
  let original_price : ℕ := 80
  let num_parts : ℕ := 7
  let final_price : ℕ := 439
  discount original_price num_parts final_price = 121 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l602_60234


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l602_60228

/-- The number of available colors for glass panes -/
def num_colors : ℕ := 10

/-- The number of panes in the window frame -/
def num_panes : ℕ := 4

/-- A function that calculates the number of valid arrangements -/
def valid_arrangements (colors : ℕ) (panes : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the number of valid arrangements is 3430 -/
theorem valid_arrangements_count :
  valid_arrangements num_colors num_panes = 3430 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l602_60228


namespace NUMINAMATH_CALUDE_twelfth_finger_number_l602_60268

-- Define the function f
def f : ℕ → ℕ
| 4 => 7
| 7 => 8
| 8 => 3
| 3 => 5
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define a function to apply f n times
def apply_f_n_times (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (apply_f_n_times n x)

-- Theorem statement
theorem twelfth_finger_number : apply_f_n_times 11 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_finger_number_l602_60268


namespace NUMINAMATH_CALUDE_brother_sister_age_diff_l602_60204

/-- The age difference between Mandy's brother and sister -/
def age_difference (mandy_age brother_age_factor sister_mandy_diff : ℕ) : ℕ :=
  brother_age_factor * mandy_age - (mandy_age + sister_mandy_diff)

/-- Theorem stating the age difference between Mandy's brother and sister -/
theorem brother_sister_age_diff :
  ∀ (mandy_age brother_age_factor sister_mandy_diff : ℕ),
    mandy_age = 3 →
    brother_age_factor = 4 →
    sister_mandy_diff = 4 →
    age_difference mandy_age brother_age_factor sister_mandy_diff = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_brother_sister_age_diff_l602_60204


namespace NUMINAMATH_CALUDE_trapezium_side_length_l602_60236

theorem trapezium_side_length (a b h : ℝ) (area : ℝ) : 
  a = 20 → h = 15 → area = 285 → area = (a + b) * h / 2 → b = 18 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l602_60236


namespace NUMINAMATH_CALUDE_highlight_film_average_time_l602_60279

/-- The average time each player gets in the highlight film -/
def average_time (durations : List Nat) : Rat :=
  (durations.sum / 60) / durations.length

/-- Theorem: Given the video durations for 5 players, the average time each player gets is 2 minutes -/
theorem highlight_film_average_time :
  let durations := [130, 145, 85, 60, 180]
  average_time durations = 2 := by sorry

end NUMINAMATH_CALUDE_highlight_film_average_time_l602_60279


namespace NUMINAMATH_CALUDE_seminar_attendees_l602_60250

theorem seminar_attendees (total : ℕ) (company_a : ℕ) : 
  total = 185 →
  company_a = 30 →
  20 = total - (company_a + 2 * company_a + (company_a + 10) + (company_a + 5)) :=
by sorry

end NUMINAMATH_CALUDE_seminar_attendees_l602_60250


namespace NUMINAMATH_CALUDE_boat_stream_speed_ratio_l602_60225

/-- Proves that the ratio of boat speed to stream speed is 3:1 given the time relation -/
theorem boat_stream_speed_ratio 
  (D : ℝ) -- Distance rowed
  (B : ℝ) -- Speed of the boat in still water
  (S : ℝ) -- Speed of the stream
  (h_positive : D > 0 ∧ B > 0 ∧ S > 0) -- Positive distances and speeds
  (h_time_ratio : D / (B - S) = 2 * (D / (B + S))) -- Time against stream is twice time with stream
  : B / S = 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_stream_speed_ratio_l602_60225


namespace NUMINAMATH_CALUDE_total_population_l602_60229

/-- The population of New England -/
def new_england_pop : ℕ := 2100000

/-- The population of New York -/
def new_york_pop : ℕ := (2 * new_england_pop) / 3

/-- The population of Pennsylvania -/
def pennsylvania_pop : ℕ := (3 * new_england_pop) / 2

/-- The combined population of Maryland and New Jersey -/
def md_nj_pop : ℕ := new_england_pop + new_england_pop / 5

/-- Theorem stating the total population of all five states -/
theorem total_population : 
  new_york_pop + new_england_pop + pennsylvania_pop + md_nj_pop = 9170000 := by
  sorry

end NUMINAMATH_CALUDE_total_population_l602_60229


namespace NUMINAMATH_CALUDE_solve_for_A_l602_60270

theorem solve_for_A : ∃ A : ℕ, 3 + 68 * A = 691 ∧ 100 ≤ 68 * A ∧ 68 * A < 1000 ∧ A = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l602_60270


namespace NUMINAMATH_CALUDE_constant_value_l602_60290

theorem constant_value (f : ℝ → ℝ) (c : ℝ) 
  (h1 : ∀ x : ℝ, f x + c * f (8 - x) = x) 
  (h2 : f 2 = 2) : 
  c = 3 := by sorry

end NUMINAMATH_CALUDE_constant_value_l602_60290


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l602_60267

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point (P : Point) (l1 l2 : Line) :
  P.liesOn l2 →
  l2.isParallelTo l1 →
  l1 = Line.mk 3 (-4) 6 →
  P = Point.mk 4 (-1) →
  l2 = Line.mk 3 (-4) (-16) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l602_60267


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l602_60265

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 2 ∧ (1 / (x - 2) + (1 - x) / (2 - x) = 3) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l602_60265


namespace NUMINAMATH_CALUDE_cos_300_degrees_l602_60235

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l602_60235


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l602_60289

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = 5 - x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l602_60289


namespace NUMINAMATH_CALUDE_mixed_tea_sale_price_l602_60298

/-- Represents the types of tea in the mixture -/
inductive TeaType
| First
| Second
| Third

/-- Represents the properties of each tea type -/
def tea_properties : TeaType → (Nat × Nat × Nat) :=
  fun t => match t with
  | TeaType.First  => (120, 30, 50)
  | TeaType.Second => (45, 40, 30)
  | TeaType.Third  => (35, 60, 25)

/-- Calculates the selling price for a given tea type -/
def selling_price (t : TeaType) : Nat :=
  let (weight, cost, profit) := tea_properties t
  weight * cost * (100 + profit) / 100

/-- Theorem stating the sale price of the mixed tea per kg -/
theorem mixed_tea_sale_price :
  (selling_price TeaType.First + selling_price TeaType.Second + selling_price TeaType.Third) /
  (120 + 45 + 35 : Nat) = 51825 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_mixed_tea_sale_price_l602_60298


namespace NUMINAMATH_CALUDE_complex_difference_magnitude_l602_60297

theorem complex_difference_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 2)
  (h3 : z₁ + z₂ = Complex.mk (Real.sqrt 3) 1) :
  Complex.abs (z₁ - z₂) = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_complex_difference_magnitude_l602_60297


namespace NUMINAMATH_CALUDE_remainder_theorem_l602_60256

theorem remainder_theorem (n : ℤ) (h : n % 5 = 2) : (n + 2023) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l602_60256


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l602_60293

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b) ∨ (a = c) ∨ (b = c)
  sumAngles : a + b + c = 180

-- Define the condition of angle ratio
def hasAngleRatio (t : IsoscelesTriangle) : Prop :=
  (t.a = 2 * t.b) ∨ (t.a = 2 * t.c) ∨ (t.b = 2 * t.c) ∨
  (2 * t.a = t.b) ∨ (2 * t.a = t.c) ∨ (2 * t.b = t.c)

-- Theorem statement
theorem isosceles_triangle_base_angle 
  (t : IsoscelesTriangle) 
  (h : hasAngleRatio t) : 
  (t.a = 45 ∧ (t.b = 45 ∨ t.c = 45)) ∨
  (t.b = 45 ∧ (t.a = 45 ∨ t.c = 45)) ∨
  (t.c = 45 ∧ (t.a = 45 ∨ t.b = 45)) ∨
  (t.a = 72 ∧ (t.b = 72 ∨ t.c = 72)) ∨
  (t.b = 72 ∧ (t.a = 72 ∨ t.c = 72)) ∨
  (t.c = 72 ∧ (t.a = 72 ∨ t.b = 72)) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l602_60293
