import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_for_always_nonnegative_quadratic_l1812_181213

theorem range_of_a_for_always_nonnegative_quadratic :
  {a : ℝ | ∀ x : ℝ, x^2 + a*x + a ≥ 0} = Set.Icc 0 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_always_nonnegative_quadratic_l1812_181213


namespace NUMINAMATH_CALUDE_max_value_expr_min_value_sum_reciprocals_l1812_181224

/-- For x > 0, the expression 4 - 2x - 2/x is at most 0 --/
theorem max_value_expr (x : ℝ) (hx : x > 0) : 4 - 2*x - 2/x ≤ 0 := by
  sorry

/-- Given a + 2b = 1 where a and b are positive real numbers, 
    the expression 1/a + 1/b is at least 3 + 2√2 --/
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + 2*b = 1) : 1/a + 1/b ≥ 3 + 2*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expr_min_value_sum_reciprocals_l1812_181224


namespace NUMINAMATH_CALUDE_max_value_7b_plus_5c_l1812_181279

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem max_value_7b_plus_5c :
  ∀ a b c : ℝ,
  (∃ a' : ℝ, a' ∈ Set.Icc 1 2 ∧
    (∀ x : ℝ, x ∈ Set.Icc 1 2 → f a' b c x ≤ 1)) →
  (∀ k : ℝ, 7 * b + 5 * c ≤ k) →
  k = -6 :=
sorry

end NUMINAMATH_CALUDE_max_value_7b_plus_5c_l1812_181279


namespace NUMINAMATH_CALUDE_characterization_of_functions_l1812_181280

/-- A function is completely multiplicative if f(xy) = f(x)f(y) for all x, y -/
def CompletelyMultiplicative (f : ℤ → ℕ) : Prop :=
  ∀ x y, f (x * y) = f x * f y

/-- The p-adic valuation of an integer -/
noncomputable def vp (p : ℕ) (x : ℤ) : ℕ := sorry

/-- The main theorem characterizing the required functions -/
theorem characterization_of_functions (f : ℤ → ℕ) : 
  (CompletelyMultiplicative f ∧ 
   ∀ a b : ℤ, b ≠ 0 → ∃ q r : ℤ, a = b * q + r ∧ f r < f b) ↔ 
  (∃ n s : ℕ, ∃ p0 : ℕ, Nat.Prime p0 ∧ 
   ∀ x : ℤ, f x = (Int.natAbs x)^n * s^(vp p0 x)) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_functions_l1812_181280


namespace NUMINAMATH_CALUDE_total_deduction_is_111_cents_l1812_181238

-- Define the hourly wage in cents
def hourly_wage : ℚ := 2500

-- Define the tax rate
def tax_rate : ℚ := 15 / 1000

-- Define the retirement contribution rate
def retirement_rate : ℚ := 3 / 100

-- Function to calculate the total deduction
def total_deduction (wage : ℚ) (tax : ℚ) (retirement : ℚ) : ℚ :=
  let tax_amount := wage * tax
  let after_tax := wage - tax_amount
  let retirement_amount := after_tax * retirement
  tax_amount + retirement_amount

-- Theorem stating that the total deduction is 111 cents
theorem total_deduction_is_111_cents :
  ⌊total_deduction hourly_wage tax_rate retirement_rate⌋ = 111 :=
sorry

end NUMINAMATH_CALUDE_total_deduction_is_111_cents_l1812_181238


namespace NUMINAMATH_CALUDE_jack_apples_proof_l1812_181268

def initial_apples : ℕ := 150
def jill_percentage : ℚ := 30 / 100
def june_percentage : ℚ := 20 / 100
def gift_apples : ℕ := 2

def remaining_apples : ℕ := 82

theorem jack_apples_proof :
  let after_jill := initial_apples - (initial_apples * jill_percentage).floor
  let after_june := after_jill - (after_jill * june_percentage).floor
  after_june - gift_apples = remaining_apples :=
by sorry

end NUMINAMATH_CALUDE_jack_apples_proof_l1812_181268


namespace NUMINAMATH_CALUDE_robert_ate_more_chocolates_l1812_181223

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 7

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- The difference in chocolates eaten between Robert and Nickel -/
def chocolate_difference : ℕ := robert_chocolates - nickel_chocolates

theorem robert_ate_more_chocolates : chocolate_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_ate_more_chocolates_l1812_181223


namespace NUMINAMATH_CALUDE_barbara_candy_distribution_l1812_181215

/-- Represents the candy distribution problem --/
structure CandyProblem where
  original_candies : Nat
  bought_candies : Nat
  num_friends : Nat

/-- Calculates the number of candies each friend receives --/
def candies_per_friend (problem : CandyProblem) : Nat :=
  (problem.original_candies + problem.bought_candies) / problem.num_friends

/-- Theorem stating that each friend receives 4 candies --/
theorem barbara_candy_distribution :
  ∀ (problem : CandyProblem),
    problem.original_candies = 9 →
    problem.bought_candies = 18 →
    problem.num_friends = 6 →
    candies_per_friend problem = 4 :=
by
  sorry

#eval candies_per_friend { original_candies := 9, bought_candies := 18, num_friends := 6 }

end NUMINAMATH_CALUDE_barbara_candy_distribution_l1812_181215


namespace NUMINAMATH_CALUDE_optimal_purchase_minimizes_cost_l1812_181239

/-- Represents the prices and quantities of soccer balls for two brands. -/
structure SoccerBallPurchase where
  priceA : ℝ  -- Price of brand A soccer ball
  priceB : ℝ  -- Price of brand B soccer ball
  quantityA : ℝ  -- Quantity of brand A soccer balls
  quantityB : ℝ  -- Quantity of brand B soccer balls

/-- The optimal purchase strategy for soccer balls. -/
def optimalPurchase : SoccerBallPurchase := {
  priceA := 50,
  priceB := 80,
  quantityA := 60,
  quantityB := 20
}

/-- The total cost of the purchase. -/
def totalCost (p : SoccerBallPurchase) : ℝ :=
  p.priceA * p.quantityA + p.priceB * p.quantityB

/-- Theorem stating the optimal purchase strategy minimizes cost under given conditions. -/
theorem optimal_purchase_minimizes_cost :
  let p := optimalPurchase
  (p.priceB = p.priceA + 30) ∧  -- Condition 1
  (1000 / p.priceA = 1600 / p.priceB) ∧  -- Condition 2
  (p.quantityA + p.quantityB = 80) ∧  -- Condition 3
  (p.quantityA ≥ 30) ∧  -- Condition 4
  (p.quantityA ≤ 3 * p.quantityB) ∧  -- Condition 5
  (∀ q : SoccerBallPurchase,
    (q.priceB = q.priceA + 30) →
    (1000 / q.priceA = 1600 / q.priceB) →
    (q.quantityA + q.quantityB = 80) →
    (q.quantityA ≥ 30) →
    (q.quantityA ≤ 3 * q.quantityB) →
    totalCost p ≤ totalCost q) :=
by
  sorry  -- Proof omitted

#check optimal_purchase_minimizes_cost

end NUMINAMATH_CALUDE_optimal_purchase_minimizes_cost_l1812_181239


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1812_181282

theorem solution_set_of_inequality (x : ℝ) :
  (2 - x) / (x + 4) > 0 ↔ -4 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1812_181282


namespace NUMINAMATH_CALUDE_quadrilateral_area_72_l1812_181277

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ :=
  (q.C.x - q.A.x) * (q.C.y - q.B.y)

/-- Theorem: The y-coordinate of B in quadrilateral ABCD that makes its area 72 square units -/
theorem quadrilateral_area_72 (q : Quadrilateral) 
    (h1 : q.A = ⟨0, 0⟩) 
    (h2 : q.B = ⟨8, q.B.y⟩)
    (h3 : q.C = ⟨8, 16⟩)
    (h4 : q.D = ⟨0, 16⟩)
    (h5 : area q = 72) : 
    q.B.y = 9 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_72_l1812_181277


namespace NUMINAMATH_CALUDE_bob_ken_situp_difference_l1812_181240

-- Define the number of sit-ups each person can do
def ken_situps : ℕ := 20
def nathan_situps : ℕ := 2 * ken_situps
def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

-- Theorem statement
theorem bob_ken_situp_difference :
  bob_situps - ken_situps = 10 := by
  sorry

end NUMINAMATH_CALUDE_bob_ken_situp_difference_l1812_181240


namespace NUMINAMATH_CALUDE_range_of_f_l1812_181203

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

-- Theorem statement
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y = π / 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1812_181203


namespace NUMINAMATH_CALUDE_max_value_cubic_quartic_sum_l1812_181287

theorem max_value_cubic_quartic_sum (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_eq_one : x + y + z = 1) :
  x + y^3 + z^4 ≤ 1 ∧ ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ a + b^3 + c^4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cubic_quartic_sum_l1812_181287


namespace NUMINAMATH_CALUDE_calculate_expression_l1812_181254

theorem calculate_expression : 5 + 12 / 3 - 2^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1812_181254


namespace NUMINAMATH_CALUDE_min_xy_over_x2_plus_y2_l1812_181262

theorem min_xy_over_x2_plus_y2 (x y : ℝ) (hx : 1/2 ≤ x ∧ x ≤ 1) (hy : 2/5 ≤ y ∧ y ≤ 1/2) :
  x * y / (x^2 + y^2) ≥ 1/2 ∧ ∃ (x₀ y₀ : ℝ), 1/2 ≤ x₀ ∧ x₀ ≤ 1 ∧ 2/5 ≤ y₀ ∧ y₀ ≤ 1/2 ∧ x₀ * y₀ / (x₀^2 + y₀^2) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_over_x2_plus_y2_l1812_181262


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1812_181242

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card
| White : Card

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "A gets the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "D gets the red card"
def D_gets_red (d : Distribution) : Prop := d Person.D = Card.Red

-- Theorem stating that the events are mutually exclusive but not complementary
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(A_gets_red d ∧ D_gets_red d)) ∧
  (∃ d : Distribution, ¬(A_gets_red d ∨ D_gets_red d)) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1812_181242


namespace NUMINAMATH_CALUDE_d₂₀₁₇_equidistant_points_l1812_181266

/-- The set S of integer coordinates (x, y) where 0 ≤ x, y ≤ 2016 -/
def S : Set (ℤ × ℤ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2016 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2016}

/-- The distance function d₂₀₁₇ -/
def d₂₀₁₇ (a b : ℤ × ℤ) : ℤ :=
  ((a.1 - b.1)^2 + (a.2 - b.2)^2) % 2017

/-- The theorem to be proved -/
theorem d₂₀₁₇_equidistant_points :
  ∃ O ∈ S,
  d₂₀₁₇ O (5, 5) = d₂₀₁₇ O (2, 6) ∧
  d₂₀₁₇ O (5, 5) = d₂₀₁₇ O (7, 11) →
  d₂₀₁₇ O (5, 5) = 1021 := by
  sorry

end NUMINAMATH_CALUDE_d₂₀₁₇_equidistant_points_l1812_181266


namespace NUMINAMATH_CALUDE_women_in_first_class_l1812_181271

theorem women_in_first_class (total_passengers : ℕ) 
  (percent_women : ℚ) (percent_women_first_class : ℚ) :
  total_passengers = 180 →
  percent_women = 65 / 100 →
  percent_women_first_class = 15 / 100 →
  ⌈(total_passengers : ℚ) * percent_women * percent_women_first_class⌉ = 18 :=
by sorry

end NUMINAMATH_CALUDE_women_in_first_class_l1812_181271


namespace NUMINAMATH_CALUDE_midpoint_path_area_ratio_l1812_181210

/-- Represents a particle moving along the edges of an equilateral triangle -/
structure Particle where
  position : ℝ × ℝ
  speed : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents the path traced by the midpoint of two particles -/
def MidpointPath (p1 p2 : Particle) : Set (ℝ × ℝ) :=
  sorry

/-- Calculates the area of a set of points in 2D space -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The main theorem statement -/
theorem midpoint_path_area_ratio
  (triangle : EquilateralTriangle)
  (p1 p2 : Particle)
  (h1 : p1.position = triangle.A ∧ p2.position = triangle.B)
  (h2 : p1.speed = p2.speed)
  : (area (MidpointPath p1 p2)) / (area {triangle.A, triangle.B, triangle.C}) = 1/4 :=
sorry

end NUMINAMATH_CALUDE_midpoint_path_area_ratio_l1812_181210


namespace NUMINAMATH_CALUDE_largest_special_number_l1812_181293

/-- A number is a two-digit number if it's between 10 and 99 inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A number ends in 4 if it leaves a remainder of 4 when divided by 10 -/
def ends_in_four (n : ℕ) : Prop := n % 10 = 4

/-- The set of two-digit numbers divisible by 6 and ending in 4 -/
def special_set : Set ℕ := {n | is_two_digit n ∧ n % 6 = 0 ∧ ends_in_four n}

theorem largest_special_number : 
  ∃ (m : ℕ), m ∈ special_set ∧ ∀ (n : ℕ), n ∈ special_set → n ≤ m ∧ m = 84 :=
sorry

end NUMINAMATH_CALUDE_largest_special_number_l1812_181293


namespace NUMINAMATH_CALUDE_equation_solution_l1812_181225

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  1 - 9 / x + 20 / x^2 = 0 → 2 / x = 1 / 2 ∨ 2 / x = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1812_181225


namespace NUMINAMATH_CALUDE_survey_change_bounds_l1812_181281

theorem survey_change_bounds (initial_yes initial_no final_yes final_no : ℚ) 
  (h1 : initial_yes = 1/2)
  (h2 : initial_no = 1/2)
  (h3 : final_yes = 7/10)
  (h4 : final_no = 3/10)
  (h5 : initial_yes + initial_no = 1)
  (h6 : final_yes + final_no = 1) :
  ∃ (x : ℚ), 1/5 ≤ x ∧ x ≤ 4/5 ∧ 
  (∃ (a b c d : ℚ), 
    a + c = initial_yes ∧
    b + d = initial_no ∧
    a + d = final_yes ∧
    b + c = final_no ∧
    c + d = x) :=
by sorry

end NUMINAMATH_CALUDE_survey_change_bounds_l1812_181281


namespace NUMINAMATH_CALUDE_smallest_height_l1812_181246

/-- Represents a rectangular box with square base -/
structure Box where
  x : ℝ  -- side length of the square base
  h : ℝ  -- height of the box
  area : ℝ -- surface area of the box

/-- The height of the box is twice the side length plus one -/
def height_constraint (b : Box) : Prop :=
  b.h = 2 * b.x + 1

/-- The surface area of the box is at least 150 square units -/
def area_constraint (b : Box) : Prop :=
  b.area ≥ 150

/-- The surface area is calculated as 2x^2 + 4x(2x + 1) -/
def surface_area_calc (b : Box) : Prop :=
  b.area = 2 * b.x^2 + 4 * b.x * (2 * b.x + 1)

/-- Main theorem: The smallest possible integer height is 9 units -/
theorem smallest_height (b : Box) 
  (h1 : height_constraint b) 
  (h2 : area_constraint b) 
  (h3 : surface_area_calc b) : 
  ∃ (min_height : ℕ), min_height = 9 ∧ 
    ∀ (h : ℕ), (∃ (b' : Box), height_constraint b' ∧ area_constraint b' ∧ surface_area_calc b' ∧ b'.h = h) → 
      h ≥ min_height :=
sorry

end NUMINAMATH_CALUDE_smallest_height_l1812_181246


namespace NUMINAMATH_CALUDE_sin_cos_sum_17_43_l1812_181299

theorem sin_cos_sum_17_43 :
  Real.sin (17 * π / 180) * Real.cos (43 * π / 180) +
  Real.cos (17 * π / 180) * Real.sin (43 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_17_43_l1812_181299


namespace NUMINAMATH_CALUDE_right_triangle_area_with_incircle_tangency_l1812_181269

/-- 
Given a right triangle with hypotenuse length c, where the incircle's point of tangency 
divides the hypotenuse in the ratio 4:9, the area of the triangle is (36/169) * c^2.
-/
theorem right_triangle_area_with_incircle_tangency (c : ℝ) (h : c > 0) : 
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧
    a^2 + b^2 = c^2 ∧  -- Pythagorean theorem for right triangle
    (4 / 13) * c * (9 / 13) * c = (1 / 2) * a * b ∧  -- Area calculation
    (1 / 2) * a * b = (36 / 169) * c^2  -- The final area formula
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_with_incircle_tangency_l1812_181269


namespace NUMINAMATH_CALUDE_solve_for_y_l1812_181202

theorem solve_for_y (x y : ℤ) (h1 : x^2 - 5*x + 8 = y + 6) (h2 : x = -8) : y = 106 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1812_181202


namespace NUMINAMATH_CALUDE_expansion_equals_fourth_power_l1812_181249

theorem expansion_equals_fourth_power (x : ℝ) : 
  (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*(x - 1) + 1 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equals_fourth_power_l1812_181249


namespace NUMINAMATH_CALUDE_xy_value_l1812_181252

theorem xy_value (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 126) : x * y = -5 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1812_181252


namespace NUMINAMATH_CALUDE_red_toys_after_removal_l1812_181217

/-- Theorem: Number of red toys after removal --/
theorem red_toys_after_removal
  (total : ℕ)
  (h_total : total = 134)
  (red white : ℕ)
  (h_initial : red + white = total)
  (h_after_removal : red - 2 = 2 * white) :
  red - 2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_red_toys_after_removal_l1812_181217


namespace NUMINAMATH_CALUDE_right_triangle_area_l1812_181206

theorem right_triangle_area (h : ℝ) (θ : ℝ) (area : ℝ) : 
  h = 20 →  -- hypotenuse is 20 inches
  θ = π / 6 →  -- one angle is 30° (π/6 radians)
  area = 50 * Real.sqrt 3 →  -- area is 50√3 square inches
  ∃ (a b : ℝ), 
    a^2 + b^2 = h^2 ∧  -- Pythagorean theorem
    a * b / 2 = area ∧  -- area formula for a triangle
    Real.sin θ = a / h  -- trigonometric relation
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1812_181206


namespace NUMINAMATH_CALUDE_blue_parrots_count_l1812_181243

theorem blue_parrots_count (total_parrots : ℕ) (green_fraction : ℚ) (blue_parrots : ℕ) : 
  total_parrots = 92 →
  green_fraction = 3/4 →
  blue_parrots = total_parrots - (green_fraction * total_parrots).num →
  blue_parrots = 23 := by
sorry

end NUMINAMATH_CALUDE_blue_parrots_count_l1812_181243


namespace NUMINAMATH_CALUDE_logan_watch_hours_l1812_181235

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of minutes Logan watched television -/
def logan_watch_time : ℕ := 300

/-- Theorem: Logan watched television for 5 hours -/
theorem logan_watch_hours : logan_watch_time / minutes_per_hour = 5 := by
  sorry

end NUMINAMATH_CALUDE_logan_watch_hours_l1812_181235


namespace NUMINAMATH_CALUDE_hair_color_theorem_l1812_181245

def hair_color_problem (start_age : ℕ) (current_age : ℕ) (future_colors : ℕ) (years_to_future : ℕ) : ℕ :=
  let current_colors := future_colors - years_to_future
  let years_adding_colors := current_age - start_age
  let initial_colors := current_colors - years_adding_colors
  initial_colors + 1

theorem hair_color_theorem :
  hair_color_problem 15 18 8 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hair_color_theorem_l1812_181245


namespace NUMINAMATH_CALUDE_painter_job_completion_six_to_four_painters_l1812_181219

/-- The number of work-days required for a group of painters to complete a job -/
def work_days (painters : ℕ) (days : ℚ) : ℚ := painters * days

theorem painter_job_completion 
  (initial_painters : ℕ) 
  (initial_days : ℚ) 
  (new_painters : ℕ) : 
  initial_painters > 0 → 
  new_painters > 0 → 
  initial_days > 0 → 
  work_days initial_painters initial_days = work_days new_painters ((initial_painters * initial_days) / new_painters) :=
by
  sorry

theorem six_to_four_painters :
  work_days 6 (14/10) = work_days 4 (21/10) :=
by
  sorry

end NUMINAMATH_CALUDE_painter_job_completion_six_to_four_painters_l1812_181219


namespace NUMINAMATH_CALUDE_triangles_in_circle_impossible_l1812_181259

theorem triangles_in_circle_impossible :
  ∀ (A₁ A₂ : ℝ), A₁ > 1 → A₂ > 1 → A₁ + A₂ > π :=
sorry

end NUMINAMATH_CALUDE_triangles_in_circle_impossible_l1812_181259


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1812_181208

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b) ↔ 0 < a ∧ a = b ∧ b < 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1812_181208


namespace NUMINAMATH_CALUDE_circles_internally_tangent_with_one_common_tangent_l1812_181218

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 4 = 0
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 12*y + 4 = 0

-- Define the centers and radii of the circles
def center_M : ℝ × ℝ := (-1, 2)
def center_N : ℝ × ℝ := (2, 6)
def radius_M : ℝ := 1
def radius_N : ℝ := 6

-- Define the distance between centers
def distance_between_centers : ℝ := 5

-- Define the common tangent line equation
def common_tangent (x y : ℝ) : Prop := 3*x + 4*y = 0

theorem circles_internally_tangent_with_one_common_tangent :
  (distance_between_centers = radius_N - radius_M) ∧
  (∃! (l : ℝ × ℝ → Prop), ∀ x y, l (x, y) ↔ common_tangent x y) ∧
  (∀ x y, circle_M x y ∧ circle_N x y → common_tangent x y) :=
sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_with_one_common_tangent_l1812_181218


namespace NUMINAMATH_CALUDE_task_completion_time_l1812_181297

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

-- Define the theorem
theorem task_completion_time 
  (start_time : Time)
  (end_third_task : Time)
  (num_tasks : Nat)
  (h1 : start_time = { hours := 9, minutes := 0 })
  (h2 : end_third_task = { hours := 11, minutes := 30 })
  (h3 : num_tasks = 4) :
  addMinutes end_third_task ((end_third_task.hours * 60 + end_third_task.minutes - 
    start_time.hours * 60 - start_time.minutes) / 3) = { hours := 12, minutes := 20 } :=
by sorry

end NUMINAMATH_CALUDE_task_completion_time_l1812_181297


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l1812_181229

theorem nested_fraction_equality : 
  (1 : ℚ) / (2 - 1 / (2 - 1 / (2 - 1 / 2))) = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l1812_181229


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1812_181270

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - (2*m + 1)*x₁ + m^2 = 0 ∧ x₂^2 - (2*m + 1)*x₂ + m^2 = 0) ↔ 
  m > -1/4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1812_181270


namespace NUMINAMATH_CALUDE_quadratic_function_value_l1812_181256

/-- Given a quadratic function f(x) = x^2 + px + q where f(3) = 0 and f(2) = 0, 
    prove that f(0) = 6. -/
theorem quadratic_function_value (p q : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + p*x + q) 
  (h2 : f 3 = 0) 
  (h3 : f 2 = 0) : 
  f 0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l1812_181256


namespace NUMINAMATH_CALUDE_inequality_proof_l1812_181290

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1812_181290


namespace NUMINAMATH_CALUDE_gcf_of_180_252_315_l1812_181226

theorem gcf_of_180_252_315 : Nat.gcd 180 (Nat.gcd 252 315) = 9 := by sorry

end NUMINAMATH_CALUDE_gcf_of_180_252_315_l1812_181226


namespace NUMINAMATH_CALUDE_opposite_of_seven_l1812_181207

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_seven : opposite 7 = -7 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_seven_l1812_181207


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1812_181286

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_property : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Main theorem about the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : seq.S 1023 - seq.S 1000 = 23) : 
  seq.a 1012 = 1 ∧ seq.S 2023 = 2023 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1812_181286


namespace NUMINAMATH_CALUDE_bus_seat_difference_l1812_181285

theorem bus_seat_difference :
  let left_seats : ℕ := 15
  let seat_capacity : ℕ := 3
  let back_seat_capacity : ℕ := 9
  let total_capacity : ℕ := 90
  let right_seats : ℕ := (total_capacity - (left_seats * seat_capacity + back_seat_capacity)) / seat_capacity
  left_seats - right_seats = 3 := by
  sorry

end NUMINAMATH_CALUDE_bus_seat_difference_l1812_181285


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1812_181233

-- Define the complex numbers
def z1 (y : ℝ) : ℂ := 3 + y * Complex.I
def z2 : ℂ := 2 - Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ y : ℝ, z1 y / z2 = 1 + Complex.I ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1812_181233


namespace NUMINAMATH_CALUDE_intersection_theorem_l1812_181237

-- Define the set A
def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Define the set B
def B : Set ℝ := {x | x^2 + x - 2 > 0}

-- State the theorem
theorem intersection_theorem : A ∩ B = {y | y > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l1812_181237


namespace NUMINAMATH_CALUDE_average_exists_l1812_181216

theorem average_exists : ∃ N : ℝ, 11 < N ∧ N < 19 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_exists_l1812_181216


namespace NUMINAMATH_CALUDE_azar_winning_configurations_l1812_181273

/-- Represents a tic-tac-toe board configuration -/
def TicTacToeBoard := List (Option Bool)

/-- Checks if a given board configuration is valid according to the game rules -/
def is_valid_board (board : TicTacToeBoard) : Bool :=
  board.length = 9 ∧ 
  (board.filter (· = some true)).length = 4 ∧
  (board.filter (· = some false)).length = 3

/-- Checks if Azar (X) has won in the given board configuration -/
def azar_wins (board : TicTacToeBoard) : Bool :=
  sorry

/-- Counts the number of valid winning configurations for Azar -/
def count_winning_configurations : Nat :=
  sorry

theorem azar_winning_configurations : 
  count_winning_configurations = 100 := by sorry

end NUMINAMATH_CALUDE_azar_winning_configurations_l1812_181273


namespace NUMINAMATH_CALUDE_product_repeating_decimal_nine_l1812_181228

theorem product_repeating_decimal_nine (x : ℚ) : x = 1/3 → x * 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_nine_l1812_181228


namespace NUMINAMATH_CALUDE_fraction_equality_l1812_181227

theorem fraction_equality : (1622^2 - 1615^2) / (1629^2 - 1608^2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1812_181227


namespace NUMINAMATH_CALUDE_quadratic_solution_l1812_181220

theorem quadratic_solution (b : ℝ) : 
  ((-2 : ℝ)^2 + b * (-2) - 63 = 0) → b = 33.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1812_181220


namespace NUMINAMATH_CALUDE_smores_graham_crackers_per_smore_l1812_181258

theorem smores_graham_crackers_per_smore (total_graham_crackers : ℕ) 
  (initial_marshmallows : ℕ) (additional_marshmallows : ℕ) :
  total_graham_crackers = 48 →
  initial_marshmallows = 6 →
  additional_marshmallows = 18 →
  (total_graham_crackers / (initial_marshmallows + additional_marshmallows) : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_smores_graham_crackers_per_smore_l1812_181258


namespace NUMINAMATH_CALUDE_distributor_profit_percentage_profit_percentage_is_87_point_5_l1812_181222

/-- Calculates the profit percentage for a distributor given specific conditions --/
theorem distributor_profit_percentage 
  (commission_rate : ℝ) 
  (cost_price : ℝ) 
  (final_price : ℝ) : ℝ :=
  let distributor_price := final_price / (1 - commission_rate)
  let profit := distributor_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

/-- The profit percentage is approximately 87.5% given the specific conditions --/
theorem profit_percentage_is_87_point_5 :
  let result := distributor_profit_percentage 0.2 19 28.5
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |result - 87.5| < ε :=
sorry

end NUMINAMATH_CALUDE_distributor_profit_percentage_profit_percentage_is_87_point_5_l1812_181222


namespace NUMINAMATH_CALUDE_solution_set_equals_union_l1812_181234

-- Define the set of solutions
def solution_set : Set ℝ := {x : ℝ | |x^2 - 2| < 2}

-- State the theorem
theorem solution_set_equals_union : 
  solution_set = Set.union (Set.Ioo (-2) 0) (Set.Ioo 0 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_union_l1812_181234


namespace NUMINAMATH_CALUDE_election_ratio_l1812_181294

theorem election_ratio (Vx Vy : ℝ) 
  (h1 : 0.64 * (Vx + Vy) = 0.76 * Vx + 0.4000000000000002 * Vy)
  (h2 : Vx > 0)
  (h3 : Vy > 0) :
  Vx / Vy = 2 := by
sorry

end NUMINAMATH_CALUDE_election_ratio_l1812_181294


namespace NUMINAMATH_CALUDE_surface_area_five_cube_removal_l1812_181248

/-- The surface area of a cube after removing central columns -/
def surface_area_after_removal (n : ℕ) : ℕ :=
  let original_surface_area := 6 * n^2
  let removed_surface_area := 6 * (n^2 - 1)
  let added_internal_surface := 2 * 3 * 4 * (n - 1)
  removed_surface_area + added_internal_surface

/-- Theorem stating that the surface area of a 5×5×5 cube after removing central columns is 192 -/
theorem surface_area_five_cube_removal :
  surface_area_after_removal 5 = 192 := by
  sorry

#eval surface_area_after_removal 5

end NUMINAMATH_CALUDE_surface_area_five_cube_removal_l1812_181248


namespace NUMINAMATH_CALUDE_sum_of_symmetric_roots_l1812_181212

theorem sum_of_symmetric_roots (f : ℝ → ℝ) 
  (h_sym : ∀ x, f (3 + x) = f (3 - x)) 
  (h_roots : ∃! (roots : Finset ℝ), roots.card = 6 ∧ ∀ x ∈ roots, f x = 0) : 
  ∃ (roots : Finset ℝ), roots.card = 6 ∧ (∀ x ∈ roots, f x = 0) ∧ (roots.sum id = 18) := by
sorry

end NUMINAMATH_CALUDE_sum_of_symmetric_roots_l1812_181212


namespace NUMINAMATH_CALUDE_garden_area_l1812_181236

/-- The area of a garden surrounding a circular ground -/
theorem garden_area (d : ℝ) (w : ℝ) (h1 : d = 34) (h2 : w = 2) :
  let r := d / 2
  let R := r + w
  π * (R^2 - r^2) = π * 72 := by sorry

end NUMINAMATH_CALUDE_garden_area_l1812_181236


namespace NUMINAMATH_CALUDE_fourth_root_81_times_cube_root_27_times_sqrt_9_l1812_181276

theorem fourth_root_81_times_cube_root_27_times_sqrt_9 : 
  (81 : ℝ) ^ (1/4) * (27 : ℝ) ^ (1/3) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_81_times_cube_root_27_times_sqrt_9_l1812_181276


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_8_sqrt_3_l1812_181275

theorem sqrt_sum_equals_8_sqrt_3 : 
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_8_sqrt_3_l1812_181275


namespace NUMINAMATH_CALUDE_cubic_polynomials_relation_l1812_181201

/-- Two monic cubic polynomials with specific roots and a relation between them -/
theorem cubic_polynomials_relation (k : ℝ) 
  (f g : ℝ → ℝ)
  (hf_monic : ∀ x, f x = x^3 + a * x^2 + b * x + c)
  (hg_monic : ∀ x, g x = x^3 + d * x^2 + e * x + i)
  (hf_roots : (k + 2) * (k + 6) * (f (k + 2)) = 0 ∧ (k + 2) * (k + 6) * (f (k + 6)) = 0)
  (hg_roots : (k + 4) * (k + 8) * (g (k + 4)) = 0 ∧ (k + 4) * (k + 8) * (g (k + 8)) = 0)
  (h_diff : ∀ x, f x - g x = x + k) : 
  k = 7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_relation_l1812_181201


namespace NUMINAMATH_CALUDE_polynomial_expansion_problem_l1812_181284

theorem polynomial_expansion_problem (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x + Real.sqrt 2) ^ 4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_problem_l1812_181284


namespace NUMINAMATH_CALUDE_h_value_l1812_181274

/-- The value of h for which the given conditions are satisfied -/
def h : ℝ := 32

/-- The y-coordinate of the first graph -/
def graph1 (x : ℝ) : ℝ := 4 * (x - h)^2 + 4032 - 4 * h^2

/-- The y-coordinate of the second graph -/
def graph2 (x : ℝ) : ℝ := 5 * (x - h)^2 + 5040 - 5 * h^2

theorem h_value :
  (graph1 0 = 4032) ∧
  (graph2 0 = 5040) ∧
  (∃ (x1 x2 : ℕ), x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ graph1 x1 = 0 ∧ graph1 x2 = 0) ∧
  (∃ (x1 x2 : ℕ), x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ graph2 x1 = 0 ∧ graph2 x2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_h_value_l1812_181274


namespace NUMINAMATH_CALUDE_relative_prime_theorem_l1812_181204

theorem relative_prime_theorem (u v w : ℤ) :
  (Nat.gcd u.natAbs v.natAbs = 1 ∧ Nat.gcd v.natAbs w.natAbs = 1 ∧ Nat.gcd u.natAbs w.natAbs = 1) ↔
  Nat.gcd (u * v + v * w + w * u).natAbs (u * v * w).natAbs = 1 := by
  sorry

#check relative_prime_theorem

end NUMINAMATH_CALUDE_relative_prime_theorem_l1812_181204


namespace NUMINAMATH_CALUDE_f_value_at_5pi_over_3_l1812_181288

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_value_at_5pi_over_3 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : has_period f π)
  (h_domain : ∀ x ∈ Set.Icc 0 (π/2), f x = π/2 - x) :
  f (5*π/3) = π/6 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_5pi_over_3_l1812_181288


namespace NUMINAMATH_CALUDE_solve_for_b_l1812_181263

theorem solve_for_b (y : ℝ) (b : ℝ) (h1 : y > 0) 
  (h2 : (70/100) * y = (8*y) / b + (3*y) / 10) : b = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l1812_181263


namespace NUMINAMATH_CALUDE_tangent_slope_squared_l1812_181200

/-- A line with slope m passing through the point (0, 2) -/
def line (m : ℝ) (x : ℝ) : ℝ := m * x + 2

/-- An ellipse centered at the origin with semi-major axis 3 and semi-minor axis 1 -/
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9

/-- The condition for the line to be tangent to the ellipse -/
def is_tangent (m : ℝ) : Prop :=
  ∃! x, ellipse x (line m x)

theorem tangent_slope_squared (m : ℝ) :
  is_tangent m → m^2 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_squared_l1812_181200


namespace NUMINAMATH_CALUDE_bertsGoldenRetrieverWeight_l1812_181244

/-- Calculates the full adult weight of a golden retriever given its growth pattern -/
def goldenRetrieverAdultWeight (initialWeight : ℕ) : ℕ :=
  let weightAtWeek9 := initialWeight * 2
  let weightAt3Months := weightAtWeek9 * 2
  let weightAt5Months := weightAt3Months * 2
  let finalWeightIncrease := 30
  weightAt5Months + finalWeightIncrease

/-- Theorem stating that the full adult weight of Bert's golden retriever is 78 pounds -/
theorem bertsGoldenRetrieverWeight :
  goldenRetrieverAdultWeight 6 = 78 := by
  sorry

end NUMINAMATH_CALUDE_bertsGoldenRetrieverWeight_l1812_181244


namespace NUMINAMATH_CALUDE_distance_after_ten_reflections_l1812_181272

/-- Represents a circular billiard table with a ball's trajectory -/
structure BilliardTable where
  radius : ℝ
  p_distance : ℝ  -- Distance of point P from the center
  reflection_angle : ℝ  -- Angle of reflection

/-- Calculates the distance between P and the ball's position after n reflections -/
noncomputable def distance_after_reflections (table : BilliardTable) (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating the distance after 10 reflections for the given table -/
theorem distance_after_ten_reflections (table : BilliardTable) 
  (h1 : table.radius = 1)
  (h2 : table.p_distance = 0.4)
  (h3 : table.reflection_angle = Real.arcsin ((Real.sqrt 57 - 5) / 8)) :
  ∃ (ε : ℝ), abs (distance_after_reflections table 10 - 0.0425) < ε ∧ ε > 0 ∧ ε < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_distance_after_ten_reflections_l1812_181272


namespace NUMINAMATH_CALUDE_f_neg_a_value_l1812_181253

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 1

theorem f_neg_a_value (a : ℝ) (h : f a = 4) : f (-a) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_a_value_l1812_181253


namespace NUMINAMATH_CALUDE_largest_base5_to_base10_l1812_181292

/-- Converts a base-5 number to base-10 --/
def base5To10 (d2 d1 d0 : Nat) : Nat :=
  d2 * 5^2 + d1 * 5^1 + d0 * 5^0

/-- The largest three-digit base-5 number --/
def largestBase5 : Nat := base5To10 4 4 4

theorem largest_base5_to_base10 :
  largestBase5 = 124 := by sorry

end NUMINAMATH_CALUDE_largest_base5_to_base10_l1812_181292


namespace NUMINAMATH_CALUDE_negation_equivalence_l1812_181232

theorem negation_equivalence (x y : ℤ) :
  ¬(Even (x + y) → Even x ∧ Even y) ↔ (¬Even (x + y) → ¬(Even x ∧ Even y)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1812_181232


namespace NUMINAMATH_CALUDE_batsman_running_percentage_l1812_181283

/-- Calculates the percentage of runs made by running between the wickets -/
def runs_percentage (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : ℚ :=
  let boundary_runs := 4 * boundaries
  let six_runs := 6 * sixes
  let runs_from_shots := boundary_runs + six_runs
  let runs_from_running := total_runs - runs_from_shots
  (runs_from_running : ℚ) / total_runs * 100

theorem batsman_running_percentage :
  runs_percentage 125 5 5 = 60 :=
sorry

end NUMINAMATH_CALUDE_batsman_running_percentage_l1812_181283


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l1812_181247

def f (x : ℝ) : ℝ := x^4 - 9*x^3 + 21*x^2 + x - 18

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := sorry

theorem polynomial_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, f x = (x - 4) * q x + 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l1812_181247


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1812_181255

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 10 = 0 → x^3 - 3*x^2 - 9*x + 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1812_181255


namespace NUMINAMATH_CALUDE_reciprocal_solutions_imply_m_value_l1812_181278

theorem reciprocal_solutions_imply_m_value (m : ℝ) : 
  (∃ x y : ℝ, 6 * x + 3 = 0 ∧ 3 * y + m = 15 ∧ x * y = 1) → m = 21 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_solutions_imply_m_value_l1812_181278


namespace NUMINAMATH_CALUDE_three_digit_powers_intersection_l1812_181214

/-- A number is a three-digit number if it's between 100 and 999, inclusive. -/
def IsThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The hundreds digit of a natural number -/
def HundredsDigit (n : ℕ) : ℕ := (n / 100) % 10

/-- A power of 3 -/
def PowerOf3 (n : ℕ) : Prop := ∃ m : ℕ, n = 3^m

/-- A power of 7 -/
def PowerOf7 (n : ℕ) : Prop := ∃ m : ℕ, n = 7^m

theorem three_digit_powers_intersection :
  ∃ (n m : ℕ),
    IsThreeDigit n ∧ PowerOf3 n ∧
    IsThreeDigit m ∧ PowerOf7 m ∧
    HundredsDigit n = HundredsDigit m ∧
    HundredsDigit n = 3 ∧
    ∀ (k : ℕ),
      (∃ (p q : ℕ),
        IsThreeDigit p ∧ PowerOf3 p ∧
        IsThreeDigit q ∧ PowerOf7 q ∧
        HundredsDigit p = HundredsDigit q ∧
        HundredsDigit p = k) →
      k = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_powers_intersection_l1812_181214


namespace NUMINAMATH_CALUDE_square_root_problem_l1812_181209

theorem square_root_problem (m : ℝ) : (Real.sqrt (m - 1) = 2) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l1812_181209


namespace NUMINAMATH_CALUDE_sum_of_powers_eight_l1812_181241

theorem sum_of_powers_eight (x : ℕ) : 
  x^8 + x^8 + x^8 + x^8 + x^5 = 4 * x^8 + x^5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_eight_l1812_181241


namespace NUMINAMATH_CALUDE_tan_alpha_negative_two_l1812_181298

theorem tan_alpha_negative_two (α : Real) (h : Real.tan α = -2) :
  (3 * Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α) = -4/7 ∧
  3 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_negative_two_l1812_181298


namespace NUMINAMATH_CALUDE_remainder_problem_l1812_181211

theorem remainder_problem (x y : ℕ) (hx : x > 0) (hy : y ≥ 0)
  (h1 : ∃ r, x ≡ r [MOD 11] ∧ 0 ≤ r ∧ r < 11)
  (h2 : 2 * x ≡ 1 [MOD 6])
  (h3 : 3 * y = (2 * x) / 6)
  (h4 : 7 * y - x = 3) :
  ∃ r, x ≡ r [MOD 11] ∧ r = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1812_181211


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_square_integer_l1812_181261

theorem sum_and_reciprocal_square_integer (a : ℝ) (h1 : a ≠ 0) (h2 : ∃ k : ℤ, a + 1 / a = k) :
  ∃ m : ℤ, a^2 + 1 / a^2 = m :=
sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_square_integer_l1812_181261


namespace NUMINAMATH_CALUDE_solve_investment_problem_l1812_181296

def investment_problem (total_investment : ℝ) (first_account_investment : ℝ) 
  (second_account_rate : ℝ) (total_interest : ℝ) : Prop :=
  let second_account_investment := total_investment - first_account_investment
  let first_account_rate := (total_interest - (second_account_investment * second_account_rate)) / first_account_investment
  first_account_rate = 0.08

theorem solve_investment_problem : 
  investment_problem 8000 3000 0.05 490 := by
  sorry

end NUMINAMATH_CALUDE_solve_investment_problem_l1812_181296


namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_not_five_digit_palindrome_product_result_171_l1812_181250

/-- A function to check if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- A function to check if a number is a five-digit palindrome -/
def isFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

/-- The main theorem -/
theorem smallest_three_digit_palindrome_not_five_digit_palindrome_product :
  ∀ n : ℕ, isThreeDigitPalindrome n → n < 171 → isFiveDigitPalindrome (n * 111) :=
by sorry

/-- The result theorem -/
theorem result_171 :
  isThreeDigitPalindrome 171 ∧ ¬ isFiveDigitPalindrome (171 * 111) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_not_five_digit_palindrome_product_result_171_l1812_181250


namespace NUMINAMATH_CALUDE_book_sale_revenue_l1812_181289

theorem book_sale_revenue (total_books : ℕ) (sold_price : ℕ) (remaining_books : ℕ) : 
  (2 * total_books = 3 * remaining_books) →
  (sold_price = 5) →
  (remaining_books = 50) →
  (2 * total_books / 3 * sold_price = 500) :=
by
  sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l1812_181289


namespace NUMINAMATH_CALUDE_area_ratio_GHI_JKL_l1812_181231

/-- Triangle GHI with sides 7, 24, and 25 -/
def triangle_GHI : Set (ℝ × ℝ) := sorry

/-- Triangle JKL with sides 9, 40, and 41 -/
def triangle_JKL : Set (ℝ × ℝ) := sorry

/-- Area of a triangle -/
def area (triangle : Set (ℝ × ℝ)) : ℝ := sorry

/-- The ratio of the areas of triangle GHI to triangle JKL is 7/15 -/
theorem area_ratio_GHI_JKL : 
  (area triangle_GHI) / (area triangle_JKL) = 7 / 15 := by sorry

end NUMINAMATH_CALUDE_area_ratio_GHI_JKL_l1812_181231


namespace NUMINAMATH_CALUDE_distance_point_to_line_l1812_181295

/-- The distance from a point (2√2, 2√2) to the line x + y - √2 = 0 is 3 -/
theorem distance_point_to_line : 
  let point : ℝ × ℝ := (2 * Real.sqrt 2, 2 * Real.sqrt 2)
  let line (x y : ℝ) : Prop := x + y - Real.sqrt 2 = 0
  ∃ (d : ℝ), d = 3 ∧ 
    d = (|point.1 + point.2 - Real.sqrt 2|) / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l1812_181295


namespace NUMINAMATH_CALUDE_sum_difference_equals_result_l1812_181267

theorem sum_difference_equals_result : 12.1212 + 17.0005 - 9.1103 = 20.0114 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equals_result_l1812_181267


namespace NUMINAMATH_CALUDE_clothing_cost_price_l1812_181257

theorem clothing_cost_price (original_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) (cost_price : ℝ) : 
  original_price = 132 →
  discount_rate = 0.1 →
  profit_rate = 0.1 →
  original_price * (1 - discount_rate) = cost_price * (1 + profit_rate) →
  cost_price = 108 := by
sorry

end NUMINAMATH_CALUDE_clothing_cost_price_l1812_181257


namespace NUMINAMATH_CALUDE_prob_our_team_l1812_181260

/-- A sports team with boys, girls, and Alice -/
structure Team where
  total : ℕ
  boys : ℕ
  girls : ℕ
  has_alice : Bool

/-- Definition of our specific team -/
def our_team : Team :=
  { total := 12
  , boys := 7
  , girls := 5
  , has_alice := true
  }

/-- The probability of choosing two girls, one of whom is Alice -/
def prob_two_girls_with_alice (t : Team) : ℚ :=
  if t.has_alice then
    (t.girls - 1 : ℚ) / (t.total.choose 2 : ℚ)
  else
    0

/-- Theorem stating the probability for our specific team -/
theorem prob_our_team :
  prob_two_girls_with_alice our_team = 2 / 33 := by
  sorry


end NUMINAMATH_CALUDE_prob_our_team_l1812_181260


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1812_181265

theorem sin_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) = 4/5 + Complex.I * 3/5 →
  Complex.exp (Complex.I * δ) = -5/13 + Complex.I * 12/13 →
  Real.sin (γ + δ) = 33/65 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1812_181265


namespace NUMINAMATH_CALUDE_range_of_a_l1812_181205

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + 4 > 0) → -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1812_181205


namespace NUMINAMATH_CALUDE_angle_quadrant_l1812_181291

theorem angle_quadrant (θ : Real) : 
  (Real.sin θ * Real.cos θ > 0) → 
  (0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2) ∨
  (Real.pi < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_quadrant_l1812_181291


namespace NUMINAMATH_CALUDE_geometry_theorem_l1812_181264

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (lines_parallel : Line → Line → Prop)

-- Define non-coincidence for lines and planes
variable (non_coincident_lines : Line → Line → Prop)
variable (non_coincident_planes : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_theorem 
  (m n : Line) (α β : Plane)
  (h_non_coincident_lines : non_coincident_lines m n)
  (h_non_coincident_planes : non_coincident_planes α β) :
  (subset m β ∧ parallel α β → line_parallel m α) ∧
  (perpendicular m α ∧ perpendicular n β ∧ parallel α β → lines_parallel m n) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l1812_181264


namespace NUMINAMATH_CALUDE_x_value_when_y_is_five_l1812_181230

/-- A line in the coordinate plane passing through the origin with slope 1/4 -/
structure Line :=
  (slope : ℚ)
  (passes_origin : Bool)

/-- A point in the coordinate plane -/
structure Point :=
  (x : ℚ)
  (y : ℚ)

/-- Checks if a point lies on a given line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x

theorem x_value_when_y_is_five (k : Line) (p1 p2 : Point) :
  k.slope = 1/4 →
  k.passes_origin = true →
  point_on_line p1 k →
  point_on_line p2 k →
  p1.x * p2.y = 160 →
  p1.y = 8 →
  p2.x = 20 →
  p2.y = 5 →
  p1.x = 32 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_five_l1812_181230


namespace NUMINAMATH_CALUDE_union_A_B_when_m_4_B_subset_A_iff_m_range_l1812_181221

-- Define sets A and B
def A : Set ℝ := {x | 2 * x - 8 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * (m + 1) * x + m^2 = 0}

-- Theorem for part (1)
theorem union_A_B_when_m_4 : A ∪ B 4 = {2, 4, 8} := by sorry

-- Theorem for part (2)
theorem B_subset_A_iff_m_range (m : ℝ) : 
  B m ⊆ A ↔ (m = 4 + 2 * Real.sqrt 2 ∨ m = 4 - 2 * Real.sqrt 2 ∨ m < -1/2) := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_m_4_B_subset_A_iff_m_range_l1812_181221


namespace NUMINAMATH_CALUDE_equal_even_odd_probability_l1812_181251

def num_dice : ℕ := 8
def num_sides : ℕ := 8
def prob_even : ℚ := 1/2
def prob_odd : ℚ := 1/2

theorem equal_even_odd_probability :
  (num_dice.choose (num_dice / 2)) * (prob_even ^ num_dice) = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_equal_even_odd_probability_l1812_181251
