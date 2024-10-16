import Mathlib

namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l3982_398270

-- Define the propositions p and q
def p (x : ℝ) : Prop := -1 < x ∧ x < 3
def q (x : ℝ) : Prop := x > 5

-- Define the relationship between ¬p and q
theorem not_p_sufficient_not_necessary_for_q :
  (∀ x, ¬(p x) → q x) ∧ ¬(∀ x, q x → ¬(p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_q_l3982_398270


namespace NUMINAMATH_CALUDE_derivative_of_f_l3982_398266

noncomputable def f (x : ℝ) := Real.cos (x^2 + x)

theorem derivative_of_f (x : ℝ) :
  deriv f x = -(2 * x + 1) * Real.sin (x^2 + x) := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3982_398266


namespace NUMINAMATH_CALUDE_gcd_f_x_l3982_398258

def f (x : ℤ) : ℤ := (5*x+3)*(8*x+2)*(12*x+7)*(3*x+11)

theorem gcd_f_x (x : ℤ) (h : ∃ k : ℤ, x = 18720 * k) : 
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 462 := by
  sorry

end NUMINAMATH_CALUDE_gcd_f_x_l3982_398258


namespace NUMINAMATH_CALUDE_water_percentage_is_15_l3982_398248

/-- Calculates the percentage of water in a mixture of three liquids -/
def water_percentage_in_mixture (a_percentage : ℚ) (b_percentage : ℚ) (c_percentage : ℚ) 
  (a_parts : ℚ) (b_parts : ℚ) (c_parts : ℚ) : ℚ :=
  ((a_percentage * a_parts + b_percentage * b_parts + c_percentage * c_parts) / 
   (a_parts + b_parts + c_parts)) * 100

/-- Theorem stating that the percentage of water in the given mixture is 15% -/
theorem water_percentage_is_15 : 
  water_percentage_in_mixture (10/100) (15/100) (25/100) 4 3 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_water_percentage_is_15_l3982_398248


namespace NUMINAMATH_CALUDE_pens_purchased_l3982_398291

theorem pens_purchased (total_cost : ℝ) (num_pencils : ℕ) (pencil_price : ℝ) (pen_price : ℝ)
  (h1 : total_cost = 570)
  (h2 : num_pencils = 75)
  (h3 : pencil_price = 2)
  (h4 : pen_price = 14) :
  (total_cost - num_pencils * pencil_price) / pen_price = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_pens_purchased_l3982_398291


namespace NUMINAMATH_CALUDE_mode_invariant_under_single_removal_l3982_398299

def dataset : List ℕ := [5, 6, 8, 8, 8, 1, 4]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_invariant_under_single_removal (d : ℕ) :
  d ∈ dataset → mode (dataset.erase d) = mode dataset := by
  sorry

end NUMINAMATH_CALUDE_mode_invariant_under_single_removal_l3982_398299


namespace NUMINAMATH_CALUDE_loss_percentage_is_15_percent_l3982_398203

def cost_price : ℚ := 1600
def selling_price : ℚ := 1360

theorem loss_percentage_is_15_percent : 
  (cost_price - selling_price) / cost_price * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_loss_percentage_is_15_percent_l3982_398203


namespace NUMINAMATH_CALUDE_christophers_age_l3982_398259

theorem christophers_age (christopher george ford : ℕ) : 
  george = christopher + 8 →
  ford = christopher - 2 →
  christopher + george + ford = 60 →
  christopher = 18 := by
sorry

end NUMINAMATH_CALUDE_christophers_age_l3982_398259


namespace NUMINAMATH_CALUDE_curve_representation_l3982_398246

-- Define the equations
def equation1 (x y : ℝ) : Prop := x * (x^2 + y^2 - 4) = 0
def equation2 (x y : ℝ) : Prop := x^2 + (x^2 + y^2 - 4)^2 = 0

-- Define what it means for an equation to represent a line and a circle
def represents_line_and_circle (f : ℝ → ℝ → Prop) : Prop :=
  (∃ a : ℝ, ∀ y, f a y) ∧ 
  (∃ c r : ℝ, ∀ x y, f x y ↔ (x - c)^2 + y^2 = r^2)

-- Define what it means for an equation to represent two points
def represents_two_points (f : ℝ → ℝ → Prop) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ, x1 ≠ x2 ∨ y1 ≠ y2 ∧ 
    (∀ x y, f x y ↔ (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2))

-- State the theorem
theorem curve_representation :
  represents_line_and_circle equation1 ∧ 
  represents_two_points equation2 := by sorry

end NUMINAMATH_CALUDE_curve_representation_l3982_398246


namespace NUMINAMATH_CALUDE_white_l_shapes_count_l3982_398233

/-- Represents a square in the grid -/
inductive Square
| White
| NonWhite

/-- Represents the grid as a 2D array of squares -/
def Grid := Array (Array Square)

/-- Represents an L-shape as three connected squares -/
structure LShape where
  pos1 : Nat × Nat
  pos2 : Nat × Nat
  pos3 : Nat × Nat

/-- Checks if an L-shape is valid (connected and L-shaped) -/
def isValidLShape (l : LShape) : Bool := sorry

/-- Checks if an L-shape is entirely white in the given grid -/
def isWhiteLShape (grid : Grid) (l : LShape) : Bool := sorry

/-- Counts the number of white L-shapes in the grid -/
def countWhiteLShapes (grid : Grid) : Nat := sorry

theorem white_l_shapes_count (grid : Grid) : 
  countWhiteLShapes grid = 24 := by sorry

end NUMINAMATH_CALUDE_white_l_shapes_count_l3982_398233


namespace NUMINAMATH_CALUDE_birds_to_asia_count_l3982_398242

/-- The number of bird families that flew to Asia -/
def birds_to_asia (initial : ℕ) (to_africa : ℕ) (remaining : ℕ) : ℕ :=
  initial - to_africa - remaining

/-- Theorem stating that 37 bird families flew to Asia -/
theorem birds_to_asia_count : birds_to_asia 85 23 25 = 37 := by
  sorry

end NUMINAMATH_CALUDE_birds_to_asia_count_l3982_398242


namespace NUMINAMATH_CALUDE_magpie_porridge_l3982_398263

/-- Represents the amount of porridge each chick received -/
structure ChickPorridge where
  x1 : ℝ
  x2 : ℝ
  x3 : ℝ
  x4 : ℝ
  x5 : ℝ
  x6 : ℝ

/-- The conditions of porridge distribution -/
def porridge_conditions (p : ChickPorridge) : Prop :=
  p.x3 = p.x1 + p.x2 ∧
  p.x4 = p.x2 + p.x3 ∧
  p.x5 = p.x3 + p.x4 ∧
  p.x6 = p.x4 + p.x5 ∧
  p.x5 = 10

/-- The total amount of porridge cooked by the magpie -/
def total_porridge (p : ChickPorridge) : ℝ :=
  p.x1 + p.x2 + p.x3 + p.x4 + p.x5 + p.x6

/-- Theorem stating that the total amount of porridge is 40 grams -/
theorem magpie_porridge (p : ChickPorridge) :
  porridge_conditions p → total_porridge p = 40 := by
  sorry

end NUMINAMATH_CALUDE_magpie_porridge_l3982_398263


namespace NUMINAMATH_CALUDE_max_x5_value_l3982_398235

theorem max_x5_value (x1 x2 x3 x4 x5 : ℕ+) 
  (h : x1 + x2 + x3 + x4 + x5 ≤ x1 * x2 * x3 * x4 * x5) :
  x5 ≤ 5 ∧ ∃ (a b c d : ℕ+), a + b + c + d + 5 ≤ a * b * c * d * 5 := by
  sorry

end NUMINAMATH_CALUDE_max_x5_value_l3982_398235


namespace NUMINAMATH_CALUDE_fayes_math_problems_l3982_398225

theorem fayes_math_problems :
  ∀ (total_problems math_problems science_problems finished_problems remaining_problems : ℕ),
    science_problems = 9 →
    finished_problems = 40 →
    remaining_problems = 15 →
    total_problems = math_problems + science_problems →
    total_problems = finished_problems + remaining_problems →
    math_problems = 46 := by
  sorry

end NUMINAMATH_CALUDE_fayes_math_problems_l3982_398225


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3982_398262

def p (x : ℝ) : Prop := x - 1 = Real.sqrt (x - 1)
def q (x : ℝ) : Prop := x = 2

theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3982_398262


namespace NUMINAMATH_CALUDE_dodecagon_interior_angles_sum_l3982_398244

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180° --/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A dodecagon is a polygon with 12 sides --/
def is_dodecagon (n : ℕ) : Prop := n = 12

theorem dodecagon_interior_angles_sum :
  ∀ n : ℕ, is_dodecagon n → sum_interior_angles n = 1800 :=
by sorry

end NUMINAMATH_CALUDE_dodecagon_interior_angles_sum_l3982_398244


namespace NUMINAMATH_CALUDE_delta_value_l3982_398268

theorem delta_value : ∀ Δ : ℤ, 4 * 3 = Δ - 6 → Δ = 18 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l3982_398268


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_reciprocals_ge_four_l3982_398296

theorem product_of_sum_and_sum_of_reciprocals_ge_four (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (1/a + 1/b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_reciprocals_ge_four_l3982_398296


namespace NUMINAMATH_CALUDE_max_value_constraint_l3982_398273

theorem max_value_constraint (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 25*c^2 = 1) :
  (8*a + 5*b + 15*c) ≤ Real.sqrt 115 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l3982_398273


namespace NUMINAMATH_CALUDE_problem_solution_l3982_398228

-- Define the set M
def M : Set ℝ := {m | ∃ x ∈ Set.Icc (-1 : ℝ) 1, m = x^2 - x}

-- Define the set N
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x - (2 - a)) < 0}

-- Theorem statement
theorem problem_solution :
  (M = Set.Icc (-1/4 : ℝ) 2) ∧
  (∀ a : ℝ, N a ⊆ M ↔ 0 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3982_398228


namespace NUMINAMATH_CALUDE_prime_divides_square_implies_divides_l3982_398256

theorem prime_divides_square_implies_divides (p n : ℕ) : 
  Prime p → (p ∣ n^2) → (p ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_square_implies_divides_l3982_398256


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3982_398229

theorem inequality_solution_set (x : ℝ) : 
  x ≠ 3 → ((x^2 - 1) / ((x - 3)^2) ≥ 0 ↔ x ∈ Set.Iic (-1) ∪ Set.Ici 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3982_398229


namespace NUMINAMATH_CALUDE_kg_to_tons_conversion_l3982_398218

theorem kg_to_tons_conversion (kg_per_ton : ℕ) (h : kg_per_ton = 1000) :
  (3600 - 600) / kg_per_ton = 3 := by
  sorry

end NUMINAMATH_CALUDE_kg_to_tons_conversion_l3982_398218


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3982_398241

/-- Given a circle C with equation x^2 + 8x + y^2 - 2y = -4, 
    prove that u + v + s = -3 + √13, where (u,v) is the center and s is the radius -/
theorem circle_center_radius_sum (x y : ℝ) : 
  (∃ (u v s : ℝ), x^2 + 8*x + y^2 - 2*y = -4 ∧ 
  (x - u)^2 + (y - v)^2 = s^2) → 
  (∃ (u v s : ℝ), x^2 + 8*x + y^2 - 2*y = -4 ∧ 
  (x - u)^2 + (y - v)^2 = s^2 ∧ 
  u + v + s = -3 + Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l3982_398241


namespace NUMINAMATH_CALUDE_mans_downstream_speed_l3982_398287

theorem mans_downstream_speed 
  (upstream_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : upstream_speed = 8) 
  (h2 : stream_speed = 2.5) : 
  upstream_speed + 2 * stream_speed = 13 := by
  sorry

end NUMINAMATH_CALUDE_mans_downstream_speed_l3982_398287


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3982_398221

theorem quadratic_roots_range (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + (3*a - 1)*x + a + 8 = 0 ↔ x = x₁ ∨ x = x₂) →  -- quadratic equation with roots x₁ and x₂
  x₁ ≠ x₂ →  -- distinct roots
  x₁ < 1 →   -- x₁ < 1
  x₂ > 1 →   -- x₂ > 1
  a < -2 :=  -- range of a
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3982_398221


namespace NUMINAMATH_CALUDE_sum_of_integers_l3982_398294

theorem sum_of_integers (p q r s : ℤ) 
  (eq1 : p - q + r = 7)
  (eq2 : q - r + s = 8)
  (eq3 : r - s + p = 4)
  (eq4 : s - p + q = 3) :
  p + q + r + s = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3982_398294


namespace NUMINAMATH_CALUDE_equation_equivalence_l3982_398210

theorem equation_equivalence (a b x y : ℝ) :
  a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1) ↔ (a*x - 1)*(a^2*y - 1) = a^5*b^5 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3982_398210


namespace NUMINAMATH_CALUDE_flower_pot_cost_l3982_398261

/-- The cost of the largest pot in a set of 6 pots -/
def largest_pot_cost (total_cost : ℚ) (num_pots : ℕ) (price_diff : ℚ) : ℚ :=
  let smallest_pot_cost := (total_cost - (price_diff * (num_pots - 1) * num_pots / 2)) / num_pots
  smallest_pot_cost + price_diff * (num_pots - 1)

/-- Theorem stating the cost of the largest pot given the problem conditions -/
theorem flower_pot_cost :
  largest_pot_cost 8.25 6 0.1 = 1.625 := by
  sorry

end NUMINAMATH_CALUDE_flower_pot_cost_l3982_398261


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3982_398217

theorem rectangle_circle_area_ratio :
  ∀ (b r : ℝ),
  b > 0 →
  r > 0 →
  6 * b = 2 * Real.pi * r →
  (2 * b^2) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3982_398217


namespace NUMINAMATH_CALUDE_ones_digit_of_9_pow_47_l3982_398272

-- Define a function to get the ones digit of an integer
def ones_digit (n : ℕ) : ℕ := n % 10

-- Theorem stating that the ones digit of 9^47 is 9
theorem ones_digit_of_9_pow_47 : ones_digit (9^47) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_9_pow_47_l3982_398272


namespace NUMINAMATH_CALUDE_jacqueline_boxes_l3982_398260

/-- The number of erasers per box -/
def erasers_per_box : ℕ := 10

/-- The total number of erasers Jacqueline has -/
def total_erasers : ℕ := 40

/-- The number of boxes Jacqueline has -/
def num_boxes : ℕ := total_erasers / erasers_per_box

theorem jacqueline_boxes : num_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_jacqueline_boxes_l3982_398260


namespace NUMINAMATH_CALUDE_jacob_coin_problem_l3982_398205

theorem jacob_coin_problem :
  ∃ (p n d : ℕ),
    p + n + d = 50 ∧
    p + 5 * n + 10 * d = 220 ∧
    d = 18 := by
  sorry

end NUMINAMATH_CALUDE_jacob_coin_problem_l3982_398205


namespace NUMINAMATH_CALUDE_max_value_of_f_l3982_398257

def f (x : ℝ) := -x^2 + 6*x - 10

theorem max_value_of_f :
  ∃ (m : ℝ), m = -1 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x ≤ m) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ f x = m) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3982_398257


namespace NUMINAMATH_CALUDE_product_and_reciprocal_sum_l3982_398250

theorem product_and_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a * b = 4) (h2 : 1 / a = 3 / b) : a + b = 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_sum_l3982_398250


namespace NUMINAMATH_CALUDE_constant_difference_l3982_398208

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivative of f and g
variable (f' g' : ℝ → ℝ)

-- Assume f' and g' are the derivatives of f and g respectively
variable (hf : ∀ x, HasDerivAt f (f' x) x)
variable (hg : ∀ x, HasDerivAt g (g' x) x)

-- State the theorem
theorem constant_difference (h : ∀ x, f' x = g' x) :
  ∃ C, ∀ x, f x - g x = C :=
sorry

end NUMINAMATH_CALUDE_constant_difference_l3982_398208


namespace NUMINAMATH_CALUDE_parabola_tangent_and_circle_l3982_398295

/-- Given a parabola y = x^2 and point P (1, -1), this theorem proves:
    1. The x-coordinates of the tangent points M and N, where x₁ < x₂, are x₁ = 1 - √2 and x₂ = 1 + √2.
    2. The area of a circle with center P tangent to line MN is 16π/5. -/
theorem parabola_tangent_and_circle (x₁ x₂ : ℝ) :
  let P : ℝ × ℝ := (1, -1)
  let T₀ : ℝ → ℝ := λ x => x^2
  let is_tangent (x : ℝ) := T₀ x = (x - 1)^2 - 1 ∧ 2*x = (x^2 + 1) / (x - 1)
  x₁ < x₂ ∧ is_tangent x₁ ∧ is_tangent x₂ →
  (x₁ = 1 - Real.sqrt 2 ∧ x₂ = 1 + Real.sqrt 2) ∧
  (π * ((2 * 1 + 1 + 1) / Real.sqrt (4 + 1))^2 = 16 * π / 5) := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_and_circle_l3982_398295


namespace NUMINAMATH_CALUDE_same_color_probability_is_121_450_l3982_398222

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (maroon : ℕ)
  (teal : ℕ)
  (cyan : ℕ)
  (sparkly : ℕ)
  (total_sides : ℕ)
  (side_sum : maroon + teal + cyan + sparkly = total_sides)

/-- The probability of rolling the same color or element on two identical colored dice -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.maroon^2 + d.teal^2 + d.cyan^2 + d.sparkly^2) / d.total_sides^2

/-- The specific die described in the problem -/
def problem_die : ColoredDie :=
  { maroon := 6
  , teal := 9
  , cyan := 10
  , sparkly := 5
  , total_sides := 30
  , side_sum := by rfl }

theorem same_color_probability_is_121_450 :
  same_color_probability problem_die = 121 / 450 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_is_121_450_l3982_398222


namespace NUMINAMATH_CALUDE_luke_coin_piles_l3982_398253

theorem luke_coin_piles (piles_quarters piles_dimes : ℕ) 
  (h1 : piles_quarters = piles_dimes)
  (h2 : 3 * piles_quarters + 3 * piles_dimes = 30) : 
  piles_quarters = 5 := by
  sorry

end NUMINAMATH_CALUDE_luke_coin_piles_l3982_398253


namespace NUMINAMATH_CALUDE_lawn_chair_price_calculation_l3982_398227

/-- Calculates the final price and overall percent decrease of a lawn chair after discounts and tax --/
theorem lawn_chair_price_calculation (original_price : ℝ) 
  (first_discount_rate second_discount_rate tax_rate : ℝ) :
  original_price = 72.95 ∧ 
  first_discount_rate = 0.10 ∧ 
  second_discount_rate = 0.15 ∧ 
  tax_rate = 0.07 →
  ∃ (final_price percent_decrease : ℝ),
    (abs (final_price - 59.71) < 0.01) ∧ 
    (abs (percent_decrease - 23.5) < 0.1) ∧
    final_price = (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) * (1 + tax_rate) ∧
    percent_decrease = (1 - (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) / original_price) * 100 := by
  sorry

end NUMINAMATH_CALUDE_lawn_chair_price_calculation_l3982_398227


namespace NUMINAMATH_CALUDE_cost_price_theorem_l3982_398238

/-- The cost price per bowl given the conditions of the problem -/
def cost_price_per_bowl (total_bowls : ℕ) (sold_bowls : ℕ) (selling_price : ℚ) (percentage_gain : ℚ) : ℚ :=
  1400 / 103

/-- Theorem stating that the cost price per bowl is 1400/103 given the problem conditions -/
theorem cost_price_theorem (total_bowls : ℕ) (sold_bowls : ℕ) (selling_price : ℚ) (percentage_gain : ℚ) 
  (h1 : total_bowls = 110)
  (h2 : sold_bowls = 100)
  (h3 : selling_price = 14)
  (h4 : percentage_gain = 27.27272727272727 / 100) :
  cost_price_per_bowl total_bowls sold_bowls selling_price percentage_gain = 1400 / 103 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_theorem_l3982_398238


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l3982_398275

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 6 * k

-- Define the solution set types
inductive SolutionSet
  | Interval
  | AllReals
  | Empty

-- State the theorem
theorem quadratic_inequality_solutions (k : ℝ) (h : k ≠ 0) :
  (∀ x, f k x < 0 ↔ x < -3 ∨ x > -2) → k = -2/5 ∧
  (∀ x, f k x < 0) → k < -Real.sqrt 6 / 6 ∧
  (∀ x, f k x ≥ 0) → k ≥ Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l3982_398275


namespace NUMINAMATH_CALUDE_no_common_multiple_in_factors_of_600_l3982_398267

theorem no_common_multiple_in_factors_of_600 : 
  ∀ n : ℕ, n ∣ 600 → ¬(30 ∣ n ∧ 42 ∣ n ∧ 56 ∣ n) :=
by
  sorry

end NUMINAMATH_CALUDE_no_common_multiple_in_factors_of_600_l3982_398267


namespace NUMINAMATH_CALUDE_second_marble_yellow_probability_l3982_398202

structure Bag where
  white : ℕ
  black : ℕ
  yellow : ℕ
  blue : ℕ

def bagX : Bag := { white := 5, black := 5, yellow := 0, blue := 0 }
def bagY : Bag := { white := 0, black := 0, yellow := 8, blue := 2 }
def bagZ : Bag := { white := 0, black := 0, yellow := 3, blue := 4 }

def prob_white_X : ℚ := 1/2
def prob_black_X : ℚ := 1/2
def prob_yellow_Y : ℚ := 4/5
def prob_yellow_Z : ℚ := 3/7

theorem second_marble_yellow_probability :
  let prob_yellow_second := prob_white_X * prob_yellow_Y + prob_black_X * prob_yellow_Z
  prob_yellow_second = 43/70 := by sorry

end NUMINAMATH_CALUDE_second_marble_yellow_probability_l3982_398202


namespace NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l3982_398265

theorem power_of_product_equals_product_of_powers (b : ℝ) : (2 * b^2)^3 = 8 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l3982_398265


namespace NUMINAMATH_CALUDE_time_interval_is_20_minutes_l3982_398279

/-- The time interval between cars given total time and number of cars -/
def time_interval (total_time_hours : ℕ) (num_cars : ℕ) : ℚ :=
  (total_time_hours * 60 : ℚ) / num_cars

/-- Theorem: The time interval between cars is 20 minutes -/
theorem time_interval_is_20_minutes :
  time_interval 10 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_time_interval_is_20_minutes_l3982_398279


namespace NUMINAMATH_CALUDE_candy_box_problem_l3982_398292

theorem candy_box_problem (n : ℕ) : n ≤ 200 →
  (n % 2 = 1 ∧ n % 3 = 1 ∧ n % 4 = 1 ∧ n % 6 = 1) →
  n % 11 = 0 →
  n = 121 := by
sorry

end NUMINAMATH_CALUDE_candy_box_problem_l3982_398292


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l3982_398286

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (team_average_age : ℕ) 
  (h1 : team_size = 11)
  (h2 : captain_age = 25)
  (h3 : team_average_age = 23)
  (h4 : ∃ (wicket_keeper_age : ℕ), 
    wicket_keeper_age > captain_age ∧ 
    (team_size : ℝ) * team_average_age = 
      (team_size - 2 : ℝ) * (team_average_age - 1) + captain_age + wicket_keeper_age) :
  ∃ (wicket_keeper_age : ℕ), wicket_keeper_age = captain_age + 5 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l3982_398286


namespace NUMINAMATH_CALUDE_bill_true_discount_l3982_398251

/-- Given a bill with face value and banker's discount, calculate the true discount -/
def true_discount (face_value banker_discount : ℚ) : ℚ :=
  (banker_discount * face_value) / (banker_discount + face_value)

/-- Theorem stating that for a bill with face value 270 and banker's discount 54, 
    the true discount is 45 -/
theorem bill_true_discount : 
  true_discount 270 54 = 45 := by sorry

end NUMINAMATH_CALUDE_bill_true_discount_l3982_398251


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3982_398204

/-- A line is described by the equation y + 3 = -3(x + 5).
    This theorem proves that the sum of its x-intercept and y-intercept is -24. -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y + 3 = -3 * (x + 5)) → 
  (∃ x_int y_int : ℝ, (y_int + 3 = -3 * (x_int + 5)) ∧ 
                      (0 + 3 = -3 * (x_int + 5)) ∧ 
                      (y_int + 3 = -3 * (0 + 5)) ∧ 
                      (x_int + y_int = -24)) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l3982_398204


namespace NUMINAMATH_CALUDE_problem_solution_l3982_398278

theorem problem_solution (y : ℝ) (hy : y ≠ 0) : (9 * y)^18 = (27 * y)^9 → y = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3982_398278


namespace NUMINAMATH_CALUDE_transform_is_shift_l3982_398298

-- Define a general function type
def RealFunction := ℝ → ℝ

-- Define the transformation
def transform (g : RealFunction) : RealFunction :=
  λ x => g (x - 2) + 3

-- State the theorem
theorem transform_is_shift (g : RealFunction) :
  ∀ x y, transform g x = y ↔ g (x - 2) = y - 3 :=
sorry

end NUMINAMATH_CALUDE_transform_is_shift_l3982_398298


namespace NUMINAMATH_CALUDE_parabola_rotation_l3982_398254

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rotates a point (x, y) by 180 degrees around the origin -/
def rotate180 (x y : ℝ) : ℝ × ℝ := (-x, -y)

/-- The original parabola y = x^2 - 6x -/
def original_parabola : Parabola := { a := 1, b := -6, c := 0 }

/-- The rotated parabola y = -(x+3)^2 + 9 -/
def rotated_parabola : Parabola := { a := -1, b := -6, c := 9 }

theorem parabola_rotation :
  ∀ x y : ℝ,
  y = original_parabola.a * x^2 + original_parabola.b * x + original_parabola.c →
  let (x', y') := rotate180 x y
  y' = rotated_parabola.a * x'^2 + rotated_parabola.b * x' + rotated_parabola.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_rotation_l3982_398254


namespace NUMINAMATH_CALUDE_unique_a_value_l3982_398219

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem unique_a_value : ∃! a : ℝ, (9 ∈ (A a ∩ B a)) ∧ ({9} = A a ∩ B a) := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l3982_398219


namespace NUMINAMATH_CALUDE_binomial_expansion_ratio_l3982_398215

theorem binomial_expansion_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61 / 60 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_ratio_l3982_398215


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3982_398255

theorem quadratic_equation_solution (a b : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 4 * x + 20 = 0 ↔ x = (a : ℂ) + b * I ∨ x = (a : ℂ) - b * I) →
  a + b^2 = 394/25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3982_398255


namespace NUMINAMATH_CALUDE_solve_farmer_problem_l3982_398289

def farmer_problem (total_cattle : ℕ) (male_percentage : ℚ) (male_count : ℕ) (total_milk : ℚ) : Prop :=
  let female_percentage : ℚ := 1 - male_percentage
  let female_count : ℕ := total_cattle - male_count
  let milk_per_female : ℚ := total_milk / female_count
  (male_percentage * total_cattle = male_count) ∧
  (female_percentage * total_cattle = female_count) ∧
  (milk_per_female = 2)

theorem solve_farmer_problem :
  ∃ (total_cattle : ℕ),
    farmer_problem total_cattle (2/5) 50 150 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_farmer_problem_l3982_398289


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_1_a_range_for_interval_containment_l3982_398271

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x + 4
def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for part (1)
theorem solution_set_for_a_eq_1 :
  let a := 1
  ∃ (S : Set ℝ), S = {x | f a x ≥ g x} ∧ S = Set.Icc (-1) ((Real.sqrt 17 - 1) / 2) :=
sorry

-- Theorem for part (2)
theorem a_range_for_interval_containment :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (-1) 1, f a x ≥ g x) → a ∈ Set.Icc (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_1_a_range_for_interval_containment_l3982_398271


namespace NUMINAMATH_CALUDE_coat_price_calculation_shopper_pays_112_75_l3982_398209

/-- Calculate the final price of a coat after discounts and tax -/
theorem coat_price_calculation (original_price : ℝ) (initial_discount_percent : ℝ) 
  (additional_discount : ℝ) (sales_tax_percent : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_percent / 100)
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  let final_price := price_after_additional_discount * (1 + sales_tax_percent / 100)
  final_price

/-- Proof that the shopper pays $112.75 for the coat -/
theorem shopper_pays_112_75 :
  coat_price_calculation 150 25 10 10 = 112.75 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_calculation_shopper_pays_112_75_l3982_398209


namespace NUMINAMATH_CALUDE_range_of_a_l3982_398213

def equation1 (a x : ℝ) : Prop := x^2 + 4*a*x - 4*a + 3 = 0

def equation2 (a x : ℝ) : Prop := x^2 + (a-1)*x + a^2 = 0

def equation3 (a x : ℝ) : Prop := x^2 + 2*a*x - 2*a = 0

def has_real_root (a : ℝ) : Prop :=
  ∃ x : ℝ, equation1 a x ∨ equation2 a x ∨ equation3 a x

theorem range_of_a : ∀ a : ℝ, has_real_root a ↔ a ≥ -1 ∨ a ≤ -3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3982_398213


namespace NUMINAMATH_CALUDE_problem_statement_l3982_398297

theorem problem_statement (a b : ℝ) (h : a - 3*b = 3) : 
  (a + 2*b) - (2*a - b) = -3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3982_398297


namespace NUMINAMATH_CALUDE_library_books_theorem_l3982_398212

-- Define the universe of books in the library
variable (Book : Type)

-- Define the property of being a new arrival
variable (is_new_arrival : Book → Prop)

-- Define the theorem
theorem library_books_theorem (h : ¬ (∀ b : Book, is_new_arrival b)) :
  (∃ b : Book, ¬ is_new_arrival b) ∧ (¬ ∀ b : Book, is_new_arrival b) := by
  sorry

end NUMINAMATH_CALUDE_library_books_theorem_l3982_398212


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l3982_398201

theorem unique_x_with_three_prime_factors (x n : ℕ) : 
  x = 5^n - 1 ∧ 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ 
    x = 2^(Nat.log 2 x) * 11 * p * q) →
  x = 3124 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l3982_398201


namespace NUMINAMATH_CALUDE_inequality_proof_l3982_398211

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 3) :
  (x - 1) * (y - 1) * (z - 1) ≤ 1/4 * (x*y*z - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3982_398211


namespace NUMINAMATH_CALUDE_variance_linear_transformation_l3982_398206

def variance (data : List ℝ) : ℝ := sorry

theorem variance_linear_transformation 
  (data : List ℝ) 
  (h : variance data = 1/3) : 
  variance (data.map (λ x => 3*x - 1)) = 3 := by sorry

end NUMINAMATH_CALUDE_variance_linear_transformation_l3982_398206


namespace NUMINAMATH_CALUDE_leap_year_date_statistics_l3982_398226

/-- Represents the data for dates in a leap year -/
structure LeapYearData where
  dates : Fin 31 → ℕ
  sum_of_values : ℕ
  total_count : ℕ

/-- The mean of the data -/
def mean (data : LeapYearData) : ℚ :=
  data.sum_of_values / data.total_count

/-- The median of the data -/
def median (data : LeapYearData) : ℕ := 16

/-- The median of the modes -/
def median_of_modes (data : LeapYearData) : ℕ := 15

theorem leap_year_date_statistics (data : LeapYearData) 
  (h1 : ∀ i : Fin 29, data.dates i = 12)
  (h2 : data.dates 30 = 11)
  (h3 : data.dates 31 = 7)
  (h4 : data.sum_of_values = 5767)
  (h5 : data.total_count = 366) :
  median_of_modes data < mean data ∧ mean data < median data := by
  sorry


end NUMINAMATH_CALUDE_leap_year_date_statistics_l3982_398226


namespace NUMINAMATH_CALUDE_fraction_irreducible_l3982_398243

theorem fraction_irreducible (n : ℕ) : Nat.gcd (12 * n + 1) (30 * n + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l3982_398243


namespace NUMINAMATH_CALUDE_triangle_theorem_l3982_398249

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the main theorem
theorem triangle_theorem (ABC : Triangle) 
  (h1 : (Real.cos ABC.B - 2 * Real.cos ABC.A) / (2 * ABC.a - ABC.b) = Real.cos ABC.C / ABC.c) :
  -- Part 1: a/b = 2
  ABC.a / ABC.b = 2 ∧
  -- Part 2: If angle A is obtuse and c = 3, then 0 < b < 3
  (ABC.A > Real.pi / 2 ∧ ABC.c = 3 → 0 < ABC.b ∧ ABC.b < 3) :=
by sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3982_398249


namespace NUMINAMATH_CALUDE_bus_journey_fraction_l3982_398234

theorem bus_journey_fraction (total_journey : ℝ) (rail_fraction : ℝ) (foot_distance : ℝ) :
  total_journey = 130 →
  rail_fraction = 3/5 →
  foot_distance = 6.5 →
  (total_journey - (rail_fraction * total_journey + foot_distance)) / total_journey = 45.5 / 130 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_fraction_l3982_398234


namespace NUMINAMATH_CALUDE_equidistant_function_property_l3982_398281

/-- Given a function g(z) = (c+di)z where c and d are real numbers,
    if g(z) is equidistant from z and the origin for all complex z,
    and |c+di| = 5, then d^2 = 99/4 -/
theorem equidistant_function_property (c d : ℝ) :
  (∀ z : ℂ, ‖(c + d * Complex.I) * z - z‖ = ‖(c + d * Complex.I) * z‖) →
  Complex.abs (c + d * Complex.I) = 5 →
  d^2 = 99/4 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_function_property_l3982_398281


namespace NUMINAMATH_CALUDE_driveway_wheel_count_inconsistent_l3982_398288

/-- Represents the number of wheels on various vehicles and items in Jordan's driveway --/
structure DrivewayCounts where
  carCount : ℕ
  bikeCount : ℕ
  trashCanCount : ℕ
  tricycleCount : ℕ
  rollerSkatesPairCount : ℕ

/-- Calculates the total number of wheels based on the counts of vehicles and items --/
def totalWheels (counts : DrivewayCounts) : ℕ :=
  4 * counts.carCount +
  2 * counts.bikeCount +
  2 * counts.trashCanCount +
  3 * counts.tricycleCount +
  4 * counts.rollerSkatesPairCount

/-- Theorem stating that given the conditions, it's impossible to have 25 wheels in total --/
theorem driveway_wheel_count_inconsistent :
  ∀ (counts : DrivewayCounts),
    counts.carCount = 2 ∧
    counts.bikeCount = 2 ∧
    counts.trashCanCount = 1 ∧
    counts.tricycleCount = 1 ∧
    counts.rollerSkatesPairCount = 1 →
    totalWheels counts ≠ 25 :=
by
  sorry

end NUMINAMATH_CALUDE_driveway_wheel_count_inconsistent_l3982_398288


namespace NUMINAMATH_CALUDE_rectangle_length_l3982_398223

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area_eq : length * width = 300
  perimeter_eq : 2 * (length + width) = 70

/-- The length of the rectangle is 20 meters -/
theorem rectangle_length (r : Rectangle) : r.length = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l3982_398223


namespace NUMINAMATH_CALUDE_quiz_points_l3982_398277

theorem quiz_points (n : ℕ) (total : ℕ) (r : ℕ) (h1 : n = 12) (h2 : total = 8190) (h3 : r = 2) :
  let first_question_points := total / (r^n - 1)
  let fifth_question_points := first_question_points * r^4
  fifth_question_points = 32 := by
sorry

end NUMINAMATH_CALUDE_quiz_points_l3982_398277


namespace NUMINAMATH_CALUDE_max_intersections_seven_segments_l3982_398230

/-- A closed polyline with a given number of segments. -/
structure ClosedPolyline :=
  (segments : ℕ)

/-- The maximum number of self-intersection points for a closed polyline. -/
def max_self_intersections (p : ClosedPolyline) : ℕ :=
  (p.segments * (p.segments - 3)) / 2

/-- Theorem: The maximum number of self-intersection points in a closed polyline with 7 segments is 14. -/
theorem max_intersections_seven_segments :
  ∃ (p : ClosedPolyline), p.segments = 7 ∧ max_self_intersections p = 14 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_seven_segments_l3982_398230


namespace NUMINAMATH_CALUDE_max_value_of_N_l3982_398280

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def last_two_digits (n : ℕ) : ℕ := n % 100

def remove_last_two_digits (n : ℕ) : ℕ := n / 100

theorem max_value_of_N :
  ∃ N : ℕ,
    is_perfect_square N ∧
    N ≥ 100 ∧
    last_two_digits N ≠ 0 ∧
    is_perfect_square (remove_last_two_digits N) ∧
    (∀ M : ℕ, 
      (is_perfect_square M ∧
       M ≥ 100 ∧
       last_two_digits M ≠ 0 ∧
       is_perfect_square (remove_last_two_digits M)) →
      M ≤ N) ∧
    N = 1681 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_N_l3982_398280


namespace NUMINAMATH_CALUDE_perfect_square_mod_three_l3982_398216

theorem perfect_square_mod_three (n : ℤ) : 
  (∃ k : ℤ, n = k^2) → (n % 3 = 0 ∨ n % 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_mod_three_l3982_398216


namespace NUMINAMATH_CALUDE_cashew_mixture_problem_l3982_398236

/-- Represents the price of peanuts per pound -/
def peanut_price : ℝ := 2.40

/-- Represents the price of cashews per pound -/
def cashew_price : ℝ := 6.00

/-- Represents the total weight of the mixture in pounds -/
def total_weight : ℝ := 60

/-- Represents the selling price of the mixture per pound -/
def mixture_price : ℝ := 3.00

/-- Represents the amount of cashews in pounds -/
def cashew_amount : ℝ := 10

theorem cashew_mixture_problem :
  ∃ (peanut_amount : ℝ),
    peanut_amount + cashew_amount = total_weight ∧
    peanut_price * peanut_amount + cashew_price * cashew_amount = mixture_price * total_weight :=
by
  sorry

end NUMINAMATH_CALUDE_cashew_mixture_problem_l3982_398236


namespace NUMINAMATH_CALUDE_fifth_term_is_two_l3982_398247

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying specific conditions, prove that its fifth term is 2. -/
theorem fifth_term_is_two (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_2 : a 2 = 2 * a 3 + 1)
  (h_4 : a 4 = 2 * a 3 + 7) : 
  a 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_two_l3982_398247


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l3982_398239

theorem cubic_equation_integer_solutions :
  ∀ x y : ℤ, x^3 + y^3 - 3*x^2 + 6*y^2 + 3*x + 12*y + 6 = 0 ↔ (x = 1 ∧ y = -1) ∨ (x = 2 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l3982_398239


namespace NUMINAMATH_CALUDE_gcf_of_60_90_150_l3982_398245

theorem gcf_of_60_90_150 : Nat.gcd 60 (Nat.gcd 90 150) = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_90_150_l3982_398245


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l3982_398274

theorem consecutive_odd_numbers_sum (n₁ n₂ n₃ : ℕ) : 
  n₁ = 9 →
  n₂ = n₁ + 2 →
  n₃ = n₂ + 2 →
  Odd n₁ →
  Odd n₂ →
  Odd n₃ →
  11 * n₁ - (3 * n₃ + 4 * n₂) = 16 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_sum_l3982_398274


namespace NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_a_squared_positive_l3982_398220

theorem a_positive_sufficient_not_necessary_for_a_squared_positive :
  (∃ a : ℝ, a > 0 → a^2 > 0) ∧ 
  (∃ a : ℝ, a^2 > 0 ∧ ¬(a > 0)) := by
  sorry

end NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_a_squared_positive_l3982_398220


namespace NUMINAMATH_CALUDE_median_mode_difference_l3982_398269

/-- Represents the monthly income data of employees --/
structure IncomeData where
  income : List Nat
  frequency : List Nat
  total_employees : Nat

/-- Calculates the mode of the income data --/
def mode (data : IncomeData) : Nat :=
  sorry

/-- Calculates the median of the income data --/
def median (data : IncomeData) : Nat :=
  sorry

/-- The income data for the company --/
def company_data : IncomeData := {
  income := [45000, 18000, 10000, 5500, 5000, 3400, 3000, 2500],
  frequency := [1, 1, 1, 3, 6, 1, 11, 1],
  total_employees := 25
}

/-- Theorem stating that the median is 400 yuan greater than the mode --/
theorem median_mode_difference (data : IncomeData) : 
  median data = mode data + 400 :=
sorry

end NUMINAMATH_CALUDE_median_mode_difference_l3982_398269


namespace NUMINAMATH_CALUDE_raffle_ticket_sales_difference_l3982_398214

/-- Proves that the difference between Saturday's and Sunday's raffle ticket sales is 284 -/
theorem raffle_ticket_sales_difference (friday_sales : ℕ) (sunday_sales : ℕ) 
  (h1 : friday_sales = 181)
  (h2 : sunday_sales = 78) : 
  2 * friday_sales - sunday_sales = 284 := by
  sorry

#check raffle_ticket_sales_difference

end NUMINAMATH_CALUDE_raffle_ticket_sales_difference_l3982_398214


namespace NUMINAMATH_CALUDE_chess_tournament_games_per_pair_l3982_398231

/-- Represents a chess tournament with a given number of players and total games. -/
structure ChessTournament where
  num_players : ℕ
  total_games : ℕ

/-- Calculates the number of times each player plays against each opponent in a chess tournament. -/
def games_per_pair (tournament : ChessTournament) : ℚ :=
  (2 * tournament.total_games : ℚ) / (tournament.num_players * (tournament.num_players - 1))

/-- Theorem stating that in a chess tournament with 18 players and 306 total games,
    each player plays against each opponent exactly 2 times. -/
theorem chess_tournament_games_per_pair :
  let tournament := ChessTournament.mk 18 306
  games_per_pair tournament = 2 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_games_per_pair_l3982_398231


namespace NUMINAMATH_CALUDE_power_sum_equation_l3982_398283

theorem power_sum_equation (p : ℕ) (a : ℤ) (n : ℕ) :
  Nat.Prime p → (2^p : ℤ) + 3^p = a^n → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equation_l3982_398283


namespace NUMINAMATH_CALUDE_dvd_fraction_proof_l3982_398293

def initial_amount : ℚ := 320
def book_fraction : ℚ := 1/4
def book_additional : ℚ := 10
def dvd_additional : ℚ := 8
def final_amount : ℚ := 130

theorem dvd_fraction_proof :
  ∃ f : ℚ, 
    initial_amount - (book_fraction * initial_amount + book_additional) - 
    (f * (initial_amount - (book_fraction * initial_amount + book_additional)) + dvd_additional) = 
    final_amount ∧ f = 46/115 := by
  sorry

end NUMINAMATH_CALUDE_dvd_fraction_proof_l3982_398293


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l3982_398232

theorem abs_inequality_solution_set (x : ℝ) : 
  2 * |x - 1| - 1 < 0 ↔ 1/2 < x ∧ x < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l3982_398232


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3982_398240

/-- Given vectors a and b in ℝ², and c = a + k * b, prove that if a is perpendicular to c, then k = -10/3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) (h1 : a = (3, 1)) (h2 : b = (1, 0)) :
  let c := a + k • b
  (a.1 * c.1 + a.2 * c.2 = 0) → k = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3982_398240


namespace NUMINAMATH_CALUDE_diamond_circle_area_l3982_398207

/-- A diamond is a quadrilateral with four equal sides -/
structure Diamond where
  side_length : ℝ
  angle_alpha : ℝ
  angle_beta : ℝ

/-- The inscribed circle of a diamond -/
structure InscribedCircle (d : Diamond) where
  center : Point

/-- The circle passing through vertices A, O, and C -/
structure CircumscribedCircle (d : Diamond) (ic : InscribedCircle d) where
  area : ℝ

/-- Main theorem: The area of the circle passing through A, O, and C in the specified diamond -/
theorem diamond_circle_area 
  (d : Diamond) 
  (ic : InscribedCircle d) 
  (cc : CircumscribedCircle d ic) 
  (h1 : d.side_length = 8) 
  (h2 : d.angle_alpha = Real.pi / 3)  -- 60 degrees in radians
  (h3 : d.angle_beta = 2 * Real.pi / 3)  -- 120 degrees in radians
  : cc.area = 48 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_diamond_circle_area_l3982_398207


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l3982_398284

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  sample : Finset ℕ
  h_sample_size : sample.card = sample_size
  h_valid_sample : ∀ n ∈ sample, n ≤ population_size

/-- Checks if a given set of numbers forms an arithmetic sequence -/
def is_arithmetic_sequence (s : Finset ℕ) : Prop :=
  ∃ a d : ℤ, ∀ n ∈ s, ∃ k : ℕ, (n : ℤ) = a + k * d

theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_pop : s.population_size = 52)
  (h_sample : s.sample_size = 4)
  (h_elements : {5, 31, 44} ⊆ s.sample)
  (h_arithmetic : is_arithmetic_sequence s.sample) :
  18 ∈ s.sample := by
  sorry


end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l3982_398284


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l3982_398285

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 3) :
  a / c = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l3982_398285


namespace NUMINAMATH_CALUDE_rope_folding_theorem_l3982_398237

def rope_segments (n : ℕ) : ℕ := 2^n + 1

theorem rope_folding_theorem :
  rope_segments 5 = 33 := by sorry

end NUMINAMATH_CALUDE_rope_folding_theorem_l3982_398237


namespace NUMINAMATH_CALUDE_sum_of_consecutive_multiples_of_three_l3982_398224

theorem sum_of_consecutive_multiples_of_three (a b c : ℕ) : 
  (a % 3 = 0) → 
  (b % 3 = 0) → 
  (c % 3 = 0) → 
  (b = a + 3) → 
  (c = b + 3) → 
  (c = 42) → 
  (a + b + c = 117) := by
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_multiples_of_three_l3982_398224


namespace NUMINAMATH_CALUDE_parallelogram_vertex_D_l3982_398282

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if ABCD forms a parallelogram -/
def isParallelogram (A B C D : Point3D) : Prop :=
  (B.x - A.x) + (D.x - C.x) = 0 ∧
  (B.y - A.y) + (D.y - C.y) = 0 ∧
  (B.z - A.z) + (D.z - C.z) = 0

theorem parallelogram_vertex_D :
  let A : Point3D := ⟨2, 0, 3⟩
  let B : Point3D := ⟨0, 3, -5⟩
  let C : Point3D := ⟨0, 0, 3⟩
  let D : Point3D := ⟨2, -3, 11⟩
  isParallelogram A B C D := by sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_D_l3982_398282


namespace NUMINAMATH_CALUDE_nonright_angle_is_45_l3982_398252

/-- A right isosceles triangle with specific properties -/
structure RightIsoscelesTriangle where
  -- The length of the hypotenuse
  h : ℝ
  -- The height from the right angle to the hypotenuse
  a : ℝ
  -- The product of the hypotenuse and the square of the height is 90
  hyp_height_product : h * a^2 = 90
  -- The triangle is right-angled (implied by being right isosceles)
  right_angled : True
  -- The triangle is isosceles
  isosceles : True

/-- The measure of one of the non-right angles in the triangle -/
def nonRightAngle (t : RightIsoscelesTriangle) : ℝ := 45

/-- Theorem: In a right isosceles triangle where the product of the hypotenuse
    and the square of the height is 90, one of the non-right angles is 45° -/
theorem nonright_angle_is_45 (t : RightIsoscelesTriangle) :
  nonRightAngle t = 45 := by sorry

end NUMINAMATH_CALUDE_nonright_angle_is_45_l3982_398252


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l3982_398264

theorem initial_number_of_persons (avg_weight_increase : ℝ) 
  (old_person_weight new_person_weight : ℝ) :
  avg_weight_increase = 2.5 →
  old_person_weight = 75 →
  new_person_weight = 95 →
  (new_person_weight - old_person_weight) / avg_weight_increase = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l3982_398264


namespace NUMINAMATH_CALUDE_smallest_k_for_sum_of_squares_multiple_of_400_l3982_398290

theorem smallest_k_for_sum_of_squares_multiple_of_400 : 
  ∀ k : ℕ+, k < 800 → ¬(∃ m : ℕ, k * (k + 1) * (2 * k + 1) = 6 * 400 * m) ∧ 
  ∃ m : ℕ, 800 * (800 + 1) * (2 * 800 + 1) = 6 * 400 * m :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_sum_of_squares_multiple_of_400_l3982_398290


namespace NUMINAMATH_CALUDE_sin_30_cos_60_plus_cos_30_sin_60_l3982_398276

theorem sin_30_cos_60_plus_cos_30_sin_60 : 
  Real.sin (30 * π / 180) * Real.cos (60 * π / 180) + 
  Real.cos (30 * π / 180) * Real.sin (60 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_cos_60_plus_cos_30_sin_60_l3982_398276


namespace NUMINAMATH_CALUDE_area_triangle_AOB_l3982_398200

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line intersecting a parabola -/
structure IntersectingLine (p : Parabola) where
  pointA : ℝ × ℝ
  pointB : ℝ × ℝ
  passes_through_focus : (pointA.1 - p.focus.1) * (pointB.2 - p.focus.2) = 
                         (pointB.1 - p.focus.1) * (pointA.2 - p.focus.2)

/-- Theorem: Area of triangle AOB for a specific parabola and intersecting line -/
theorem area_triangle_AOB 
  (p : Parabola) 
  (l : IntersectingLine p) 
  (h_parabola : p.equation = fun x y => y^2 = 4*x) 
  (h_focus : p.focus = (1, 0)) 
  (h_AF_length : Real.sqrt ((l.pointA.1 - p.focus.1)^2 + (l.pointA.2 - p.focus.2)^2) = 3) :
  let O : ℝ × ℝ := (0, 0)
  Real.sqrt (
    (l.pointA.1 * l.pointB.2 - l.pointB.1 * l.pointA.2)^2 +
    (l.pointA.1 * O.2 - O.1 * l.pointA.2)^2 +
    (O.1 * l.pointB.2 - l.pointB.1 * O.2)^2
  ) / 2 = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_AOB_l3982_398200
