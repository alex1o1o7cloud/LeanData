import Mathlib

namespace cupcakes_per_package_l1081_108180

theorem cupcakes_per_package
  (initial_cupcakes : ℕ)
  (eaten_cupcakes : ℕ)
  (num_packages : ℕ)
  (h1 : initial_cupcakes = 18)
  (h2 : eaten_cupcakes = 8)
  (h3 : num_packages = 5)
  : (initial_cupcakes - eaten_cupcakes) / num_packages = 2 :=
by
  sorry

end cupcakes_per_package_l1081_108180


namespace selling_price_calculation_l1081_108167

theorem selling_price_calculation (cost_price : ℝ) (profit_percentage : ℝ) :
  cost_price = 180 →
  profit_percentage = 15 →
  cost_price + (cost_price * (profit_percentage / 100)) = 207 := by
  sorry

end selling_price_calculation_l1081_108167


namespace point_on_number_line_l1081_108183

/-- Given two points A and B on a number line where A represents -3 and B is 7 units to the right of A, 
    prove that B represents 4. -/
theorem point_on_number_line (A B : ℝ) : A = -3 ∧ B = A + 7 → B = 4 := by
  sorry

end point_on_number_line_l1081_108183


namespace fraction_less_than_mode_l1081_108173

def data_list : List ℕ := [3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 10, 11, 15, 21, 23, 26, 27]

def mode (l : List ℕ) : ℕ := 
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

def count_less_than_mode (l : List ℕ) : ℕ :=
  l.filter (· < mode l) |>.length

theorem fraction_less_than_mode :
  (count_less_than_mode data_list : ℚ) / data_list.length = 1 / 9 := by
  sorry

end fraction_less_than_mode_l1081_108173


namespace hcl_required_l1081_108174

-- Define the chemical reaction
structure Reaction where
  hcl : ℕ  -- moles of Hydrochloric acid
  koh : ℕ  -- moles of Potassium hydroxide
  h2o : ℕ  -- moles of Water
  kcl : ℕ  -- moles of Potassium chloride

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.hcl = r.koh ∧ r.hcl = r.h2o ∧ r.hcl = r.kcl

-- Define the given conditions
def given_conditions (r : Reaction) : Prop :=
  r.koh = 2 ∧ r.h2o = 2 ∧ r.kcl = 2

-- Theorem to prove
theorem hcl_required (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : given_conditions r) : 
  r.hcl = 2 := by
  sorry

#check hcl_required

end hcl_required_l1081_108174


namespace potato_sales_total_weight_l1081_108195

theorem potato_sales_total_weight :
  let morning_sales : ℕ := 29
  let afternoon_sales : ℕ := 17
  let bag_weight : ℕ := 7
  let total_bags : ℕ := morning_sales + afternoon_sales
  let total_weight : ℕ := total_bags * bag_weight
  total_weight = 322 := by sorry

end potato_sales_total_weight_l1081_108195


namespace certain_number_proof_l1081_108135

theorem certain_number_proof (h : 16 * 21.3 = 340.8) : 213 * 16 = 3408 := by
  sorry

end certain_number_proof_l1081_108135


namespace car_travel_and_budget_l1081_108124

/-- Represents a car with its fuel-to-distance ratio and fuel usage -/
structure Car where
  fuel_ratio : Rat
  distance_ratio : Rat
  fuel_used : ℚ
  fuel_cost : ℚ

/-- Calculates the distance traveled by a car -/
def distance_traveled (c : Car) : ℚ :=
  c.distance_ratio * c.fuel_used / c.fuel_ratio

/-- Calculates the fuel cost for a car -/
def fuel_cost (c : Car) : ℚ :=
  c.fuel_cost * c.fuel_used

theorem car_travel_and_budget (car_a car_b : Car) (budget : ℚ) :
  car_a.fuel_ratio = 4/7 ∧
  car_a.distance_ratio = 7/4 ∧
  car_a.fuel_used = 44 ∧
  car_a.fuel_cost = 7/2 ∧
  car_b.fuel_ratio = 3/5 ∧
  car_b.distance_ratio = 5/3 ∧
  car_b.fuel_used = 27 ∧
  car_b.fuel_cost = 13/4 ∧
  budget = 200 →
  distance_traveled car_a + distance_traveled car_b = 122 ∧
  fuel_cost car_a + fuel_cost car_b = 967/4 ∧
  fuel_cost car_a + fuel_cost car_b - budget = 167/4 :=
by sorry

end car_travel_and_budget_l1081_108124


namespace f_is_power_function_l1081_108127

/-- Definition of a power function -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

/-- The function y = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: f is a power function -/
theorem f_is_power_function : is_power_function f := by
  sorry

end f_is_power_function_l1081_108127


namespace lcm_of_5_6_9_21_l1081_108139

theorem lcm_of_5_6_9_21 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 9 21)) = 630 := by
  sorry

end lcm_of_5_6_9_21_l1081_108139


namespace arithmetic_geometric_mean_quadratic_equation_l1081_108189

theorem arithmetic_geometric_mean_quadratic_equation 
  (a b : ℝ) 
  (h_arithmetic_mean : (a + b) / 2 = 8) 
  (h_geometric_mean : Real.sqrt (a * b) = 15) : 
  ∀ x : ℝ, x^2 - 16*x + 225 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end arithmetic_geometric_mean_quadratic_equation_l1081_108189


namespace min_blocks_for_specific_wall_l1081_108169

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  length : ℕ
  height : ℕ

/-- Calculates the minimum number of blocks needed to build a wall --/
def minBlocksNeeded (wall : WallDimensions) (blocks : List BlockDimensions) : ℕ :=
  sorry

/-- Theorem: The minimum number of blocks needed for the specified wall is 404 --/
theorem min_blocks_for_specific_wall :
  let wall : WallDimensions := ⟨120, 8⟩
  let blocks : List BlockDimensions := [⟨2, 1⟩, ⟨3, 1⟩, ⟨1, 1⟩]
  minBlocksNeeded wall blocks = 404 :=
by
  sorry

#check min_blocks_for_specific_wall

end min_blocks_for_specific_wall_l1081_108169


namespace smallest_advantageous_discount_l1081_108114

def two_successive_discounts (d : ℝ) : ℝ := (1 - d) * (1 - d)
def three_successive_discounts (d : ℝ) : ℝ := (1 - d) * (1 - d) * (1 - d)
def two_different_discounts (d1 d2 : ℝ) : ℝ := (1 - d1) * (1 - d2)

theorem smallest_advantageous_discount : ∀ n : ℕ, n ≥ 34 →
  (1 - n / 100 < two_successive_discounts 0.18) ∧
  (1 - n / 100 < three_successive_discounts 0.12) ∧
  (1 - n / 100 < two_different_discounts 0.28 0.07) ∧
  (∀ m : ℕ, m < 34 →
    (1 - m / 100 ≥ two_successive_discounts 0.18) ∨
    (1 - m / 100 ≥ three_successive_discounts 0.12) ∨
    (1 - m / 100 ≥ two_different_discounts 0.28 0.07)) :=
by sorry

end smallest_advantageous_discount_l1081_108114


namespace work_completion_time_l1081_108181

theorem work_completion_time (W : ℝ) (W_p W_q W_r : ℝ) :
  W_p = W_q + W_r →                -- p can do the work in the same time as q and r together
  W_p + W_q = W / 10 →             -- p and q together can complete the work in 10 days
  W_r = W / 35 →                   -- r alone needs 35 days to complete the work
  W_q = W / 28                     -- q alone needs 28 days to complete the work
  := by sorry

end work_completion_time_l1081_108181


namespace volunteer_selection_ways_l1081_108131

/-- The number of volunteers --/
def n : ℕ := 5

/-- The number of days for community service --/
def days : ℕ := 2

/-- The number of people selected each day --/
def selected_per_day : ℕ := 2

/-- Function to calculate the number of ways to select volunteers --/
def select_volunteers (n : ℕ) : ℕ :=
  (n) * (n - 1) * (n - 2)

theorem volunteer_selection_ways :
  select_volunteers n = 60 :=
sorry

end volunteer_selection_ways_l1081_108131


namespace tangent_line_to_circleC_l1081_108143

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The circle C: x^2 + y^2 - 6y + 8 = 0 -/
def circleC : Circle := { center := (0, 3), radius := 1 }

/-- A line in the form y = kx -/
structure Line where
  k : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  ∃ p : ℝ × ℝ, p.2 = l.k * p.1 ∧ 
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    ∀ q : ℝ × ℝ, q.2 = l.k * q.1 → 
      (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 ≥ c.radius^2

theorem tangent_line_to_circleC (l : Line) :
  isTangent l circleC ∧ 
  (∃ p : ℝ × ℝ, isTangent l circleC ∧ isInSecondQuadrant p) →
  l.k = -2 * Real.sqrt 2 := by
  sorry

end tangent_line_to_circleC_l1081_108143


namespace point_on_line_l1081_108188

/-- Given a line passing through points (0,4) and (-6,1), prove that s = 6 
    is the unique solution such that (s,7) lies on this line. -/
theorem point_on_line (s : ℝ) : 
  (∃! x : ℝ, (x - 0) / (-6 - 0) = (7 - 4) / (x - 0) ∧ x = s) → s = 6 :=
by sorry

end point_on_line_l1081_108188


namespace average_multiples_of_6_up_to_100_l1081_108157

def multiples_of_6 (n : ℕ) : Finset ℕ :=
  Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))

theorem average_multiples_of_6_up_to_100 :
  let S := multiples_of_6 100
  (S.sum id) / S.card = 51 := by
  sorry

end average_multiples_of_6_up_to_100_l1081_108157


namespace trapezoid_KL_length_l1081_108119

/-- A trapezoid with points K and L on its diagonals -/
structure Trapezoid :=
  (A B C D K L : ℝ × ℝ)
  (is_trapezoid : sorry)
  (BC : ℝ)
  (AD : ℝ)
  (K_on_AC : sorry)
  (L_on_BD : sorry)
  (CK_KA_ratio : sorry)
  (BL_LD_ratio : sorry)

/-- The length of KL in the trapezoid -/
def KL_length (t : Trapezoid) : ℝ := sorry

theorem trapezoid_KL_length (t : Trapezoid) : 
  KL_length t = (1 / 11) * |7 * t.AD - 4 * t.BC| := by
  sorry

end trapezoid_KL_length_l1081_108119


namespace line_mb_product_l1081_108186

/-- Given a line passing through points (0, -1) and (2, -6) with equation y = mx + b,
    prove that the product mb equals 5/2. -/
theorem line_mb_product (m b : ℚ) : 
  (∀ x y : ℚ, y = m * x + b) →
  (-1 : ℚ) = b →
  (-6 : ℚ) = m * 2 + b →
  m * b = 5 / 2 := by
  sorry

end line_mb_product_l1081_108186


namespace curve_is_circle_l1081_108175

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define what a circle is
def is_circle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    S = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Theorem statement
theorem curve_is_circle :
  is_circle {p : ℝ × ℝ | curve p.1 p.2} :=
sorry

end curve_is_circle_l1081_108175


namespace perfect_square_unique_l1081_108111

/-- Checks if a quadratic expression ax^2 + bx + c is a perfect square trinomial -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  b^2 = 4*a*c ∧ a > 0

theorem perfect_square_unique :
  ¬ is_perfect_square_trinomial 1 0 1 ∧     -- x^2 + 1
  ¬ is_perfect_square_trinomial 1 2 (-1) ∧  -- x^2 + 2x - 1
  ¬ is_perfect_square_trinomial 1 1 1 ∧     -- x^2 + x + 1
  is_perfect_square_trinomial 1 4 4         -- x^2 + 4x + 4
  :=
sorry

end perfect_square_unique_l1081_108111


namespace sufficient_condition_for_inequality_l1081_108165

theorem sufficient_condition_for_inequality (x : ℝ) :
  1 < x ∧ x < 2 → (x + 1) / (x - 1) > 2 := by
  sorry

end sufficient_condition_for_inequality_l1081_108165


namespace sum_greater_two_necessary_not_sufficient_l1081_108101

theorem sum_greater_two_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a + b > 2) ∧
  (∃ a b : ℝ, a + b > 2 ∧ (a ≤ 1 ∨ b ≤ 1)) :=
sorry

end sum_greater_two_necessary_not_sufficient_l1081_108101


namespace smallest_integer_above_root_sum_power_l1081_108156

theorem smallest_integer_above_root_sum_power :
  ∃ n : ℕ, (n = 3323 ∧ (∀ m : ℕ, m < n → m ≤ (Real.sqrt 5 + Real.sqrt 3)^6) ∧
            (∀ k : ℕ, k > (Real.sqrt 5 + Real.sqrt 3)^6 → k ≥ n)) := by
  sorry

end smallest_integer_above_root_sum_power_l1081_108156


namespace system_solution_l1081_108164

theorem system_solution :
  let s : Set (ℚ × ℚ) := {(1/2, 5), (1, 3), (3/2, 2), (5/2, 1)}
  ∀ x y : ℚ, (2*x + y + 2*x*y = 11 ∧ 2*x^2*y + x*y^2 = 15) ↔ (x, y) ∈ s := by
  sorry

end system_solution_l1081_108164


namespace no_three_prime_roots_in_geometric_progression_l1081_108129

theorem no_three_prime_roots_in_geometric_progression :
  ¬∃ (p₁ p₂ p₃ : ℕ) (n₁ n₂ n₃ : ℤ) (a r : ℝ),
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
    p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃ ∧
    n₁ ≠ n₂ ∧ n₂ ≠ n₃ ∧ n₁ ≠ n₃ ∧
    a > 0 ∧ r > 0 ∧
    a * r^n₁ = Real.sqrt p₁ ∧
    a * r^n₂ = Real.sqrt p₂ ∧
    a * r^n₃ = Real.sqrt p₃ :=
by sorry

end no_three_prime_roots_in_geometric_progression_l1081_108129


namespace unique_A_value_l1081_108107

theorem unique_A_value (A : ℝ) (x₁ x₂ : ℂ) 
  (h_distinct : x₁ ≠ x₂)
  (h_eq1 : x₁ * (x₁ + 1) = A)
  (h_eq2 : x₂ * (x₂ + 1) = A)
  (h_eq3 : A * x₁^4 + 3 * x₁^3 + 5 * x₁ = x₂^4 + 3 * x₂^3 + 5 * x₂) :
  A = -7 := by
  sorry

end unique_A_value_l1081_108107


namespace second_number_proof_l1081_108168

theorem second_number_proof (d : ℕ) (n₁ n₂ x : ℕ) : 
  d = 16 →
  n₁ = 25 →
  n₂ = 105 →
  x = 41 →
  x > n₁ →
  x % d = n₁ % d →
  x % d = n₂ % d →
  ∀ y : ℕ, n₁ < y ∧ y < x → y % d ≠ n₁ % d ∨ y % d ≠ n₂ % d :=
by sorry

end second_number_proof_l1081_108168


namespace area_of_circumscribed_circle_l1081_108104

/-- An isosceles triangle with two sides of length 4 and base of length 3 -/
structure IsoscelesTriangle where
  sideLength : ℝ
  baseLength : ℝ
  isIsosceles : sideLength = 4 ∧ baseLength = 3

/-- A circle passing through the vertices of the triangle -/
structure CircumscribedCircle (t : IsoscelesTriangle) where
  radius : ℝ
  passesThrough : True  -- This is a placeholder for the property that the circle passes through all vertices

/-- The theorem stating that the area of the circumscribed circle is 16π -/
theorem area_of_circumscribed_circle (t : IsoscelesTriangle) (c : CircumscribedCircle t) :
  π * c.radius^2 = 16 * π := by sorry

end area_of_circumscribed_circle_l1081_108104


namespace partial_fraction_decomposition_l1081_108148

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ -1 ∧ x^2 - x + 2 ≠ 0 →
  (x^2 + 2*x - 8) / (x^3 - x - 2) = 
  (-9/4) / (x + 1) + (13/4 * x - 7/2) / (x^2 - x + 2) := by
sorry

end partial_fraction_decomposition_l1081_108148


namespace circle_area_equality_l1081_108145

theorem circle_area_equality (r₁ r₂ r₃ : ℝ) : 
  r₁ = 13 → r₂ = 23 → π * r₃^2 = π * (r₂^2 - r₁^2) → r₃ = 6 * Real.sqrt 10 :=
by sorry

end circle_area_equality_l1081_108145


namespace milkshake_hours_l1081_108190

/-- Given that Augustus makes 3 milkshakes per hour and Luna makes 7 milkshakes per hour,
    prove that they have been making milkshakes for 8 hours when they have made 80 milkshakes in total. -/
theorem milkshake_hours (augustus_rate : ℕ) (luna_rate : ℕ) (total_milkshakes : ℕ) (hours : ℕ) :
  augustus_rate = 3 →
  luna_rate = 7 →
  total_milkshakes = 80 →
  augustus_rate * hours + luna_rate * hours = total_milkshakes →
  hours = 8 := by
sorry

end milkshake_hours_l1081_108190


namespace notebook_dispatch_l1081_108178

theorem notebook_dispatch (x y : ℕ) 
  (h1 : x * (y + 5) = x * y + 1250) 
  (h2 : (x + 7) * y = x * y + 3150) : 
  x + y = 700 := by
  sorry

end notebook_dispatch_l1081_108178


namespace seven_valid_configurations_l1081_108161

/-- Represents a square piece --/
structure Square :=
  (label : Char)

/-- Represents the T-shaped figure --/
structure TShape

/-- Represents a configuration of squares added to the T-shape --/
structure Configuration :=
  (square1 : Square)
  (square2 : Square)

/-- Checks if a configuration can be folded into a closed cubical box --/
def can_fold_into_cube (config : Configuration) : Prop :=
  sorry

/-- The set of all possible configurations --/
def all_configurations (squares : Finset Square) : Finset Configuration :=
  sorry

/-- The set of valid configurations that can be folded into a cube --/
def valid_configurations (squares : Finset Square) : Finset Configuration :=
  sorry

theorem seven_valid_configurations :
  ∀ (t : TShape) (squares : Finset Square),
    (Finset.card squares = 8) →
    (Finset.card (valid_configurations squares) = 7) :=
  sorry

end seven_valid_configurations_l1081_108161


namespace tangent_points_distance_l1081_108198

theorem tangent_points_distance (r : ℝ) (d : ℝ) (h1 : r = 7) (h2 : d = 25) :
  let tangent_length := Real.sqrt (d^2 - r^2)
  2 * tangent_length = 48 :=
sorry

end tangent_points_distance_l1081_108198


namespace valid_bases_for_346_l1081_108182

def is_valid_base (b : ℕ) : Prop :=
  b > 1 ∧ b^3 ≤ 346 ∧ 346 < b^4 ∧
  ∃ (d₃ d₂ d₁ d₀ : ℕ), 
    d₃ * b^3 + d₂ * b^2 + d₁ * b^1 + d₀ * b^0 = 346 ∧
    d₃ ≠ 0 ∧ d₀ % 2 = 0

theorem valid_bases_for_346 :
  ∀ b : ℕ, is_valid_base b ↔ (b = 6 ∨ b = 7) :=
sorry

end valid_bases_for_346_l1081_108182


namespace sets_equivalence_l1081_108194

-- Define the sets
def A : Set ℝ := {1}
def B : Set ℝ := {y : ℝ | (y - 1)^2 = 0}
def D : Set ℝ := {x : ℝ | x - 1 = 0}

-- C is not defined as a set because it's not a valid set notation

theorem sets_equivalence :
  (A = B) ∧ (A = D) ∧ (B = D) :=
sorry

-- Note: We can't include C in the theorem because it's not a valid set

end sets_equivalence_l1081_108194


namespace visible_shaded_area_coefficient_sum_l1081_108199

/-- Represents the visible shaded area of a grid with circles on top. -/
def visibleShadedArea (gridSize : ℕ) (smallCircleCount : ℕ) (smallCircleDiameter : ℝ) 
  (largeCircleCount : ℕ) (largeCircleDiameter : ℝ) : ℝ := by sorry

/-- The sum of coefficients A and B in the expression A - Bπ for the visible shaded area. -/
def coefficientSum (gridSize : ℕ) (smallCircleCount : ℕ) (smallCircleDiameter : ℝ) 
  (largeCircleCount : ℕ) (largeCircleDiameter : ℝ) : ℝ := by sorry

theorem visible_shaded_area_coefficient_sum :
  coefficientSum 6 5 1 1 4 = 41.25 := by sorry

end visible_shaded_area_coefficient_sum_l1081_108199


namespace pie_eating_contest_l1081_108153

/-- The amount of pie Erik ate -/
def eriks_pie : ℝ := 0.67

/-- The amount of pie Frank ate -/
def franks_pie : ℝ := 0.33

/-- The difference between Erik's and Frank's pie consumption -/
def pie_difference : ℝ := eriks_pie - franks_pie

/-- Theorem stating that the difference between Erik's and Frank's pie consumption is 0.34 -/
theorem pie_eating_contest : pie_difference = 0.34 := by
  sorry

end pie_eating_contest_l1081_108153


namespace one_sixth_of_twelve_x_plus_six_l1081_108187

theorem one_sixth_of_twelve_x_plus_six (x : ℝ) : (1 / 6) * (12 * x + 6) = 2 * x + 1 := by
  sorry

end one_sixth_of_twelve_x_plus_six_l1081_108187


namespace watermelons_with_seeds_l1081_108171

theorem watermelons_with_seeds (ripe : ℕ) (unripe : ℕ) (seedless : ℕ) : 
  ripe = 11 → unripe = 13 → seedless = 15 → 
  ripe + unripe - seedless = 9 := by
  sorry

end watermelons_with_seeds_l1081_108171


namespace milton_zoology_books_l1081_108150

theorem milton_zoology_books :
  ∀ (z b : ℕ),
  z + b = 80 →
  b = 4 * z →
  z = 16 :=
by
  sorry

end milton_zoology_books_l1081_108150


namespace autobiography_to_fiction_ratio_l1081_108138

theorem autobiography_to_fiction_ratio
  (total_books : ℕ)
  (fiction_books : ℕ)
  (nonfiction_books : ℕ)
  (picture_books : ℕ)
  (h_total : total_books = 35)
  (h_fiction : fiction_books = 5)
  (h_nonfiction : nonfiction_books = fiction_books + 4)
  (h_picture : picture_books = 11)
  : (total_books - fiction_books - nonfiction_books - picture_books) / fiction_books = 2 := by
  sorry

end autobiography_to_fiction_ratio_l1081_108138


namespace waiter_income_fraction_l1081_108128

theorem waiter_income_fraction (salary : ℚ) (salary_positive : salary > 0) : 
  let tips := (7 / 3) * salary
  let bonuses := (2 / 5) * salary
  let total_income := salary + tips + bonuses
  tips / total_income = 5 / 8 := by
sorry

end waiter_income_fraction_l1081_108128


namespace sqrt_3_minus_pi_squared_l1081_108163

theorem sqrt_3_minus_pi_squared (π : ℝ) (h : π > 3) : 
  Real.sqrt ((3 - π)^2) = π - 3 := by
sorry

end sqrt_3_minus_pi_squared_l1081_108163


namespace apple_juice_production_l1081_108113

/-- Given the annual U.S. apple production and its distribution, calculate the amount used for juice -/
theorem apple_juice_production (total_production : ℝ) (cider_percentage : ℝ) (juice_percentage : ℝ) :
  total_production = 7 →
  cider_percentage = 0.25 →
  juice_percentage = 0.60 →
  (total_production * (1 - cider_percentage) * juice_percentage) = 3.15 := by
  sorry

end apple_juice_production_l1081_108113


namespace cost_price_per_meter_l1081_108132

/-- Given a cloth sale scenario, prove the cost price per meter. -/
theorem cost_price_per_meter
  (total_length : ℕ)
  (total_selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 78)
  (h2 : total_selling_price = 6788)
  (h3 : profit_per_meter = 29) :
  (total_selling_price - profit_per_meter * total_length) / total_length = 58 := by
  sorry

end cost_price_per_meter_l1081_108132


namespace negative_300_coterminal_with_60_l1081_108109

/-- An angle is coterminal with 60 degrees if it can be expressed as k * 360 + 60, where k is an integer -/
def is_coterminal_with_60 (angle : ℝ) : Prop :=
  ∃ k : ℤ, angle = k * 360 + 60

/-- Theorem stating that -300 degrees is coterminal with 60 degrees -/
theorem negative_300_coterminal_with_60 : is_coterminal_with_60 (-300) := by
  sorry

end negative_300_coterminal_with_60_l1081_108109


namespace total_games_in_season_l1081_108170

/-- Calculate the number of games in a round-robin tournament -/
def num_games (n : ℕ) (r : ℕ) : ℕ :=
  (n * (n - 1) / 2) * r

/-- The number of teams in the league -/
def num_teams : ℕ := 14

/-- The number of times each team plays every other team -/
def num_rounds : ℕ := 5

theorem total_games_in_season :
  num_games num_teams num_rounds = 455 := by sorry

end total_games_in_season_l1081_108170


namespace polynomial_functional_equation_l1081_108160

/-- Given a polynomial P : ℝ × ℝ → ℝ satisfying P(x - 1, y - 2x + 1) = P(x, y) for all x and y,
    there exists a polynomial Φ : ℝ → ℝ such that P(x, y) = Φ(y - x^2) for all x and y. -/
theorem polynomial_functional_equation
  (P : ℝ → ℝ → ℝ)
  (h : ∀ x y : ℝ, P (x - 1) (y - 2*x + 1) = P x y)
  : ∃ Φ : ℝ → ℝ, ∀ x y : ℝ, P x y = Φ (y - x^2) := by
  sorry

end polynomial_functional_equation_l1081_108160


namespace sqrt_450_simplification_l1081_108133

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplification_l1081_108133


namespace p_necessary_not_sufficient_for_q_l1081_108106

-- Define the conditions
def p (x y : ℝ) : Prop := (x - 1) * (y - 2) = 0
def q (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 0

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient_for_q :
  (∀ x y : ℝ, q x y → p x y) ∧ (∃ x y : ℝ, p x y ∧ ¬(q x y)) := by
  sorry

end p_necessary_not_sufficient_for_q_l1081_108106


namespace sin_15_105_product_l1081_108100

theorem sin_15_105_product : 4 * Real.sin (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end sin_15_105_product_l1081_108100


namespace rafael_net_pay_l1081_108142

/-- Calculates the total net pay for Rafael's work week --/
def calculate_net_pay (monday_hours : ℕ) (tuesday_hours : ℕ) (total_week_hours : ℕ) 
  (max_daily_hours : ℕ) (regular_rate : ℚ) (overtime_rate : ℚ) (bonus : ℚ) 
  (tax_rate : ℚ) (tax_credit : ℚ) : ℚ :=
  let remaining_days := 3
  let remaining_hours := total_week_hours - monday_hours - tuesday_hours
  let wednesday_hours := min max_daily_hours remaining_hours
  let thursday_hours := min max_daily_hours (remaining_hours - wednesday_hours)
  let friday_hours := remaining_hours - wednesday_hours - thursday_hours
  
  let monday_pay := regular_rate * min monday_hours max_daily_hours + 
    overtime_rate * max (monday_hours - max_daily_hours) 0
  let tuesday_pay := regular_rate * tuesday_hours
  let wednesday_pay := regular_rate * wednesday_hours
  let thursday_pay := regular_rate * thursday_hours
  let friday_pay := regular_rate * friday_hours
  
  let total_pay := monday_pay + tuesday_pay + wednesday_pay + thursday_pay + friday_pay + bonus
  let taxes := max (total_pay * tax_rate - tax_credit) 0
  
  total_pay - taxes

/-- Theorem stating that Rafael's net pay for the week is $878 --/
theorem rafael_net_pay : 
  calculate_net_pay 10 8 40 8 20 30 100 (1/10) 50 = 878 := by
  sorry

end rafael_net_pay_l1081_108142


namespace odd_function_zero_value_necessary_not_sufficient_l1081_108176

-- Define what it means for a function to be odd
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_zero_value_necessary_not_sufficient :
  (∀ f : ℝ → ℝ, IsOdd f → f 0 = 0) ∧
  (∃ g : ℝ → ℝ, g 0 = 0 ∧ ¬IsOdd g) :=
sorry

end odd_function_zero_value_necessary_not_sufficient_l1081_108176


namespace sport_formulation_water_amount_l1081_108141

/-- Represents the ratio of flavoring to corn syrup to water in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio :=
  ⟨1, 12, 30⟩

/-- The sport formulation ratio -/
def sport_ratio : DrinkRatio :=
  ⟨1, 4, 60⟩

/-- Calculates the amount of water given the amount of corn syrup and the drink ratio -/
def water_amount (corn_syrup_amount : ℚ) (ratio : DrinkRatio) : ℚ :=
  (corn_syrup_amount * ratio.water) / ratio.corn_syrup

theorem sport_formulation_water_amount :
  water_amount 3 sport_ratio = 45 := by
  sorry

end sport_formulation_water_amount_l1081_108141


namespace cuboid_lateral_surface_area_l1081_108159

/-- The lateral surface area of a cuboid with given dimensions -/
def lateralSurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * height + width * height)

/-- Theorem: The lateral surface area of a cuboid with length 10 m, width 14 m, and height 18 m is 864 m² -/
theorem cuboid_lateral_surface_area :
  lateralSurfaceArea 10 14 18 = 864 := by
  sorry

end cuboid_lateral_surface_area_l1081_108159


namespace bag_slips_problem_l1081_108177

theorem bag_slips_problem (total_slips : ℕ) (num1 num2 : ℕ) (expected_value : ℚ) :
  total_slips = 15 →
  num1 = 3 →
  num2 = 8 →
  expected_value = 5 →
  ∃ (slips_with_num1 : ℕ),
    slips_with_num1 ≤ total_slips ∧
    (slips_with_num1 : ℚ) / total_slips * num1 + 
    ((total_slips - slips_with_num1) : ℚ) / total_slips * num2 = expected_value →
    slips_with_num1 = 9 :=
by sorry

end bag_slips_problem_l1081_108177


namespace sum_of_roots_l1081_108184

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x : ℝ, x^2 - 8*a*x - 9*b = 0 ↔ (x = c ∨ x = d)) →
  (∀ x : ℝ, x^2 - 8*c*x - 9*d = 0 ↔ (x = a ∨ x = b)) →
  a + b + c + d = 648 :=
by sorry

end sum_of_roots_l1081_108184


namespace traffic_light_statement_correct_l1081_108155

/-- A traffic light state can be either red or green -/
inductive TrafficLightState
  | Red
  | Green

/-- A traffic light intersection scenario -/
structure TrafficLightIntersection where
  state : TrafficLightState

/-- The statement about traffic light outcomes is correct -/
theorem traffic_light_statement_correct :
  ∀ (intersection : TrafficLightIntersection),
    (intersection.state = TrafficLightState.Red) ∨
    (intersection.state = TrafficLightState.Green) :=
by sorry

end traffic_light_statement_correct_l1081_108155


namespace jungkook_has_fewest_erasers_l1081_108185

-- Define the number of erasers for each person
def jungkook_erasers : ℕ := 6
def jimin_erasers : ℕ := jungkook_erasers + 4
def seokjin_erasers : ℕ := jimin_erasers - 3

-- Theorem to prove Jungkook has the fewest erasers
theorem jungkook_has_fewest_erasers :
  jungkook_erasers < jimin_erasers ∧ jungkook_erasers < seokjin_erasers :=
by sorry

end jungkook_has_fewest_erasers_l1081_108185


namespace max_value_on_circle_l1081_108126

/-- The maximum value of x^2 + y^2 for points on the circle x^2 - 4x - 4 + y^2 = 0 -/
theorem max_value_on_circle : 
  ∀ x y : ℝ, x^2 - 4*x - 4 + y^2 = 0 → x^2 + y^2 ≤ 12 + 8*Real.sqrt 2 :=
by sorry

end max_value_on_circle_l1081_108126


namespace range_of_cosine_squared_minus_two_sine_l1081_108140

theorem range_of_cosine_squared_minus_two_sine :
  ∀ (x : ℝ), -2 ≤ Real.cos x ^ 2 - 2 * Real.sin x ∧ 
             Real.cos x ^ 2 - 2 * Real.sin x ≤ 2 ∧
             ∃ (y z : ℝ), Real.cos y ^ 2 - 2 * Real.sin y = -2 ∧
                          Real.cos z ^ 2 - 2 * Real.sin z = 2 :=
by sorry

end range_of_cosine_squared_minus_two_sine_l1081_108140


namespace wider_can_radius_l1081_108162

/-- Given two cylindrical cans with the same volume, where the height of one can is five times
    the height of the other, and the radius of the narrower can is 10 units,
    prove that the radius of the wider can is 10√5 units. -/
theorem wider_can_radius (h : ℝ) (volume : ℝ) : 
  volume = π * 10^2 * (5 * h) → 
  volume = π * ((10 * Real.sqrt 5)^2) * h := by
  sorry

end wider_can_radius_l1081_108162


namespace concert_attendance_l1081_108179

theorem concert_attendance (adults : ℕ) (children : ℕ) : 
  children = 3 * adults →
  7 * adults + 3 * children = 6000 →
  adults + children = 1500 := by
  sorry

end concert_attendance_l1081_108179


namespace determine_relationship_l1081_108110

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - x - 2 > 0}
def Q : Set ℝ := {x : ℝ | |x - 1| > 1}

-- Define the possible relationships
inductive Relationship
  | sufficient_not_necessary
  | necessary_not_sufficient
  | necessary_and_sufficient
  | neither_sufficient_nor_necessary

-- Theorem to prove
theorem determine_relationship : Relationship :=
  sorry

end determine_relationship_l1081_108110


namespace min_a1_arithmetic_sequence_l1081_108152

def arithmetic_sequence (a : ℕ → ℕ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem min_a1_arithmetic_sequence (a : ℕ → ℕ) 
  (h_arith : arithmetic_sequence a) 
  (h_pos : ∀ n, a n > 0)
  (h_a9 : a 9 = 2023) :
  a 1 ≥ 7 ∧ ∃ b : ℕ → ℕ, arithmetic_sequence b ∧ (∀ n, b n > 0) ∧ b 9 = 2023 ∧ b 1 = 7 :=
sorry

end min_a1_arithmetic_sequence_l1081_108152


namespace shooter_probability_l1081_108115

theorem shooter_probability (p10 p9 p8 : ℝ) 
  (h1 : p10 = 0.2) 
  (h2 : p9 = 0.3) 
  (h3 : p8 = 0.1) : 
  1 - (p10 + p9) = 0.5 := by
  sorry

end shooter_probability_l1081_108115


namespace min_center_value_l1081_108197

def RegularOctagon (vertices : Fin 8 → ℕ) (center : ℕ) :=
  (∀ i j : Fin 8, i ≠ j → vertices i ≠ vertices j) ∧
  (vertices 0 + vertices 1 + vertices 4 + vertices 5 + center =
   vertices 1 + vertices 2 + vertices 5 + vertices 6 + center) ∧
  (vertices 2 + vertices 3 + vertices 6 + vertices 7 + center =
   vertices 3 + vertices 0 + vertices 7 + vertices 4 + center) ∧
  (vertices 0 + vertices 1 + vertices 2 + vertices 3 +
   vertices 4 + vertices 5 + vertices 6 + vertices 7 =
   vertices 0 + vertices 1 + vertices 4 + vertices 5 + center)

theorem min_center_value (vertices : Fin 8 → ℕ) (center : ℕ) 
  (h : RegularOctagon vertices center) :
  center ≥ 14 := by
  sorry

end min_center_value_l1081_108197


namespace playground_fundraiser_correct_l1081_108172

def playground_fundraiser (johnson_amount sutton_amount rollin_amount total_amount : ℝ) : Prop :=
  johnson_amount = 2300 ∧
  johnson_amount = 2 * sutton_amount ∧
  rollin_amount = 8 * sutton_amount ∧
  rollin_amount = total_amount / 3 ∧
  total_amount * 0.98 = 27048

theorem playground_fundraiser_correct :
  ∃ johnson_amount sutton_amount rollin_amount total_amount : ℝ,
    playground_fundraiser johnson_amount sutton_amount rollin_amount total_amount :=
by
  sorry

end playground_fundraiser_correct_l1081_108172


namespace helga_shoe_shopping_l1081_108149

/-- The number of pairs of shoes Helga tried on at the first store -/
def first_store : ℕ := 7

/-- The number of pairs of shoes Helga tried on at the second store -/
def second_store : ℕ := first_store + 2

/-- The number of pairs of shoes Helga tried on at the third store -/
def third_store : ℕ := 0

/-- The number of pairs of shoes Helga tried on at the fourth store -/
def fourth_store : ℕ := 2 * (first_store + second_store + third_store)

/-- The total number of pairs of shoes Helga tried on -/
def total_shoes : ℕ := first_store + second_store + third_store + fourth_store

theorem helga_shoe_shopping :
  total_shoes = 48 := by sorry

end helga_shoe_shopping_l1081_108149


namespace unique_a_value_l1081_108108

def A (a : ℚ) : Set ℚ := {a + 2, 2 * a^2 + a}

theorem unique_a_value : ∃! a : ℚ, 3 ∈ A a ∧ a = -3/2 := by sorry

end unique_a_value_l1081_108108


namespace imaginary_part_of_z_l1081_108158

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - 2 * Complex.I) : 
  z.im = -3/2 := by sorry

end imaginary_part_of_z_l1081_108158


namespace problem_solution_l1081_108117

theorem problem_solution (x y : ℚ) (hx : x = 3) (hy : y = 5) : 
  (x^5 + 2*y^2 - 15) / 7 = 39 + 5/7 := by
  sorry

end problem_solution_l1081_108117


namespace unique_m_value_l1081_108193

theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A → m = 3 := by
  sorry

end unique_m_value_l1081_108193


namespace total_spent_proof_l1081_108118

/-- The total amount spent on gifts and giftwrapping -/
def total_spent (gift_cost giftwrap_cost : ℚ) : ℚ :=
  gift_cost + giftwrap_cost

/-- Theorem: Given the cost of gifts and giftwrapping, prove the total amount spent -/
theorem total_spent_proof (gift_cost giftwrap_cost : ℚ) 
  (h1 : gift_cost = 561)
  (h2 : giftwrap_cost = 139) : 
  total_spent gift_cost giftwrap_cost = 700 := by
  sorry

end total_spent_proof_l1081_108118


namespace point_adding_procedure_l1081_108123

theorem point_adding_procedure (x : ℕ+) : ∃ x, 9 * x - 8 = 82 := by
  sorry

end point_adding_procedure_l1081_108123


namespace bell_interval_problem_l1081_108103

/-- Represents the intervals of the four bells in seconds -/
structure BellIntervals where
  bell1 : ℕ
  bell2 : ℕ
  bell3 : ℕ
  bell4 : ℕ

/-- Checks if the given intervals result in the bells tolling together after the specified time -/
def tollTogether (intervals : BellIntervals) (time : ℕ) : Prop :=
  time % intervals.bell1 = 0 ∧
  time % intervals.bell2 = 0 ∧
  time % intervals.bell3 = 0 ∧
  time % intervals.bell4 = 0

/-- The main theorem to prove -/
theorem bell_interval_problem (intervals : BellIntervals) :
  intervals.bell1 = 9 →
  intervals.bell3 = 14 →
  intervals.bell4 = 18 →
  tollTogether intervals 630 →
  intervals.bell2 = 5 := by
  sorry

end bell_interval_problem_l1081_108103


namespace prob_three_odd_six_dice_value_l1081_108136

/-- The probability of rolling an odd number on a fair 8-sided die -/
def prob_odd_8sided : ℚ := 1/2

/-- The number of ways to choose 3 dice out of 6 -/
def choose_3_from_6 : ℕ := 20

/-- The probability of rolling exactly three odd numbers when rolling six fair 8-sided dice -/
def prob_three_odd_six_dice : ℚ :=
  (choose_3_from_6 : ℚ) * (prob_odd_8sided ^ 3 * (1 - prob_odd_8sided) ^ 3)

theorem prob_three_odd_six_dice_value : prob_three_odd_six_dice = 5/16 := by
  sorry

end prob_three_odd_six_dice_value_l1081_108136


namespace division_result_l1081_108125

theorem division_result : (64 : ℝ) / 0.08 = 800 := by
  sorry

end division_result_l1081_108125


namespace probability_three_yellow_apples_l1081_108120

def total_apples : ℕ := 10
def yellow_apples : ℕ := 4
def selected_apples : ℕ := 3

def probability_all_yellow : ℚ := (yellow_apples.choose selected_apples) / (total_apples.choose selected_apples)

theorem probability_three_yellow_apples :
  probability_all_yellow = 1 / 30 := by
  sorry

end probability_three_yellow_apples_l1081_108120


namespace equation_solution_l1081_108154

theorem equation_solution :
  ∃! x : ℝ, (32 : ℝ) ^ (x - 2) / (8 : ℝ) ^ (x - 2) = (512 : ℝ) ^ (3 * x) ∧ x = -4/25 := by
  sorry

end equation_solution_l1081_108154


namespace hyperbola_b_value_l1081_108137

-- Define the hyperbola
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Theorem statement
theorem hyperbola_b_value (b : ℝ) :
  (b > 0) →
  (∀ x y : ℝ, hyperbola x y b ↔ asymptotes x y) →
  b = 1 :=
by sorry

end hyperbola_b_value_l1081_108137


namespace arithmetic_calculation_l1081_108146

theorem arithmetic_calculation : 1325 + 180 / 60 * 3 - 225 = 1109 := by
  sorry

end arithmetic_calculation_l1081_108146


namespace hyperbola_eccentricity_l1081_108151

/-- The eccentricity of a hyperbola with asymptote tangent to a specific circle is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →
  (∃ (x y : ℝ), (x - Real.sqrt 3)^2 + (y - 1)^2 = 1) →
  (∃ (m c : ℝ), ∀ (x y : ℝ), y = m*x + c → 
    ((x - Real.sqrt 3)^2 + (y - 1)^2 = 1 ∧ 
     (∃ (t : ℝ), x = a*t ∧ y = b*t))) →
  a / Real.sqrt (a^2 - b^2) = 2 := by
sorry

end hyperbola_eccentricity_l1081_108151


namespace dow_decrease_l1081_108196

def initial_dow : ℝ := 8900
def percentage_decrease : ℝ := 0.02

theorem dow_decrease (initial : ℝ) (decrease : ℝ) :
  initial = initial_dow →
  decrease = percentage_decrease →
  initial * (1 - decrease) = 8722 :=
by sorry

end dow_decrease_l1081_108196


namespace fraction_simplification_l1081_108112

theorem fraction_simplification (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ -y) :
  x / (x - y) - y / (x + y) = (x^2 + y^2) / (x^2 - y^2) := by
  sorry


end fraction_simplification_l1081_108112


namespace soda_discount_percentage_l1081_108122

theorem soda_discount_percentage (regular_price : ℝ) (discounted_total : ℝ) (cans : ℕ) :
  regular_price = 0.15 →
  discounted_total = 10.125 →
  cans = 75 →
  ∃ (discount : ℝ), 
    discount = 0.1 ∧
    cans * regular_price * (1 - discount) = discounted_total :=
by sorry

end soda_discount_percentage_l1081_108122


namespace x_plus_y_equals_three_l1081_108147

theorem x_plus_y_equals_three (x y : ℝ) (h : |x - 1| + (y - 2)^2 = 0) : x + y = 3 := by
  sorry

end x_plus_y_equals_three_l1081_108147


namespace hyperbola_asymptotes_l1081_108102

/-- Given a hyperbola and a circle, prove the equations of the hyperbola's asymptotes -/
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ (x y : ℝ), x^2 / 9 - y^2 / m = 1 ∧ x^2 + y^2 - 4*x - 5 = 0) →
  (∃ (k : ℝ), k = 4/3 ∧ 
    (∀ (x y : ℝ), (x^2 / 9 - y^2 / m = 1) → (y = k*x ∨ y = -k*x))) :=
by sorry

end hyperbola_asymptotes_l1081_108102


namespace sum_of_solutions_quadratic_l1081_108130

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let a : ℝ := -48
  let b : ℝ := 96
  let c : ℝ := -72
  let sum_of_roots := -b / a
  (a * x^2 + b * x + c = 0) → sum_of_roots = 2 :=
by
  sorry

end sum_of_solutions_quadratic_l1081_108130


namespace smallest_a_divisibility_l1081_108116

theorem smallest_a_divisibility : 
  ∃ (n : ℕ), 
    n % 2 = 1 ∧ 
    (55^n + 2000 * 32^n) % 2001 = 0 ∧ 
    ∀ (a : ℕ), a > 0 → a < 2000 → 
      ∀ (m : ℕ), m % 2 = 1 → (55^m + a * 32^m) % 2001 ≠ 0 :=
by sorry

end smallest_a_divisibility_l1081_108116


namespace fathers_age_problem_l1081_108121

/-- Father's age problem -/
theorem fathers_age_problem (F C1 C2 : ℕ) : 
  F = 3 * (C1 + C2) →  -- Father's age is three times the sum of children's ages
  F + 5 = 2 * (C1 + 5 + C2 + 5) →  -- After 5 years, father's age will be twice the sum of children's ages
  F = 45 :=  -- Father's current age is 45
by sorry

end fathers_age_problem_l1081_108121


namespace range_of_m_l1081_108105

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define sets A and B
variable (A B : Set ℝ)

-- State the theorem
theorem range_of_m (h : ∃ (x₁ x₂ : ℝ), x₁ ∈ A ∧ x₂ ∈ A ∧ x₁ ≠ x₂ ∧ f x₁ = f x₂ ∧ f x₁ ∈ B) :
  ∀ m ∈ B, m > -1 := by
  sorry

end range_of_m_l1081_108105


namespace distance_AB_l1081_108166

def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 6)

theorem distance_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 := by
  sorry

end distance_AB_l1081_108166


namespace not_all_prime_l1081_108144

theorem not_all_prime (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_div_a : a ∣ b + c + b * c)
  (h_div_b : b ∣ c + a + c * a)
  (h_div_c : c ∣ a + b + a * b) :
  ¬(Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) := by
  sorry

end not_all_prime_l1081_108144


namespace initial_student_count_l1081_108134

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (dropped_score : ℝ) : 
  initial_avg = 60.5 → new_avg = 64.0 → dropped_score = 8 → 
  ∃ n : ℕ, n > 0 ∧ 
    initial_avg * n = new_avg * (n - 1) + dropped_score ∧
    n = 16 := by
  sorry

end initial_student_count_l1081_108134


namespace max_value_of_f_l1081_108192

def f (x : ℕ) : ℤ := 2 * x - 3

def S : Set ℕ := {x : ℕ | 1 ≤ x ∧ x ≤ 10/3}

theorem max_value_of_f :
  ∃ (m : ℤ), m = 3 ∧ ∀ (x : ℕ), x ∈ S → f x ≤ m :=
sorry

end max_value_of_f_l1081_108192


namespace room_width_calculation_l1081_108191

/-- Proves that given a rectangular room with length 5.5 meters, if the total cost of paving the floor at a rate of 1200 Rs per square meter is 24750 Rs, then the width of the room is 3.75 meters. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) : 
  length = 5.5 → 
  cost_per_sqm = 1200 → 
  total_cost = 24750 → 
  width = total_cost / cost_per_sqm / length → 
  width = 3.75 := by
sorry


end room_width_calculation_l1081_108191
