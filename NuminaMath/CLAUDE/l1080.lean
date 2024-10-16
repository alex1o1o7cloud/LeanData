import Mathlib

namespace NUMINAMATH_CALUDE_twenty_team_tournament_games_l1080_108063

/-- Calculates the number of games in a single-elimination tournament. -/
def tournament_games (num_teams : ℕ) : ℕ :=
  num_teams - 1

/-- Theorem: A single-elimination tournament with 20 teams requires 19 games. -/
theorem twenty_team_tournament_games :
  tournament_games 20 = 19 := by
  sorry

#eval tournament_games 20

end NUMINAMATH_CALUDE_twenty_team_tournament_games_l1080_108063


namespace NUMINAMATH_CALUDE_point_movement_and_quadrant_l1080_108071

/-- Given a point P with coordinates (a - 7, 3 - 2a) in the Cartesian coordinate system,
    which is moved 4 units up and 5 units right to obtain point Q (a - 2, 7 - 2a),
    prove that for Q to be in the first quadrant, 2 < a < 3.5,
    and when a is an integer satisfying this condition, P = (-4, -3) and Q = (1, 1). -/
theorem point_movement_and_quadrant (a : ℝ) :
  let P : ℝ × ℝ := (a - 7, 3 - 2*a)
  let Q : ℝ × ℝ := (a - 2, 7 - 2*a)
  (∀ x y, Q = (x, y) → x > 0 ∧ y > 0) ↔ (2 < a ∧ a < 3.5) ∧
  (∃ n : ℤ, ↑n = a ∧ 2 < a ∧ a < 3.5) →
    P = (-4, -3) ∧ Q = (1, 1) :=
by sorry

end NUMINAMATH_CALUDE_point_movement_and_quadrant_l1080_108071


namespace NUMINAMATH_CALUDE_k_value_l1080_108015

/-- The function f(x) = 4x^2 + 3x + 5 -/
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5

/-- The function g(x) = x^2 + kx - 7 with parameter k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 + k * x - 7

/-- Theorem stating that if f(5) - g(5) = 20, then k = 82/5 -/
theorem k_value (k : ℝ) : f 5 - g k 5 = 20 → k = 82 / 5 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l1080_108015


namespace NUMINAMATH_CALUDE_pet_store_dogs_l1080_108062

theorem pet_store_dogs (cat_count : ℕ) (cat_ratio dog_ratio : ℕ) : 
  cat_count = 21 → cat_ratio = 3 → dog_ratio = 4 → 
  (cat_count * dog_ratio) / cat_ratio = 28 := by
sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l1080_108062


namespace NUMINAMATH_CALUDE_intersected_cubes_in_4x4x4_cube_l1080_108079

/-- Represents a cube composed of unit cubes -/
structure UnitCube where
  side : ℕ

/-- Represents a plane in 3D space -/
structure Plane

/-- Predicate to check if a plane is perpendicular to and bisects an internal diagonal of a cube -/
def is_perpendicular_bisector (c : UnitCube) (p : Plane) : Prop :=
  sorry

/-- Counts the number of unit cubes intersected by a plane in a given cube -/
def intersected_cubes (c : UnitCube) (p : Plane) : ℕ :=
  sorry

/-- Theorem stating that a plane perpendicular to and bisecting an internal diagonal
    of a 4x4x4 cube intersects exactly 40 unit cubes -/
theorem intersected_cubes_in_4x4x4_cube (c : UnitCube) (p : Plane) :
  c.side = 4 → is_perpendicular_bisector c p → intersected_cubes c p = 40 :=
by sorry

end NUMINAMATH_CALUDE_intersected_cubes_in_4x4x4_cube_l1080_108079


namespace NUMINAMATH_CALUDE_inequality_proof_l1080_108083

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1080_108083


namespace NUMINAMATH_CALUDE_inequality_implies_a_greater_than_three_l1080_108019

theorem inequality_implies_a_greater_than_three (a : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 →
    Real.sqrt 2 * (2 * a + 3) * Real.cos (θ - π / 4) + 6 / (Real.sin θ + Real.cos θ) - 2 * Real.sin (2 * θ) < 3 * a + 6) →
  a > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_greater_than_three_l1080_108019


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1080_108008

theorem fraction_evaluation : (2 + 3 + 4) / (2 * 3 * 4) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1080_108008


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l1080_108060

-- Define the cylinders
structure Cylinder where
  radius : ℝ
  height : ℝ

-- Define the theorem
theorem cylinder_height_relationship (c1 c2 : Cylinder) : 
  -- Conditions
  (c1.radius * c1.radius * c1.height = c2.radius * c2.radius * c2.height) →  -- Equal volumes
  (c2.radius = 1.2 * c1.radius) →                                            -- Second radius is 20% more
  -- Conclusion
  (c1.height = 1.44 * c2.height) :=                                          -- First height is 44% more
by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l1080_108060


namespace NUMINAMATH_CALUDE_min_value_theorem_l1080_108051

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 2) :
  (2 / x) + (1 / y) ≥ 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1080_108051


namespace NUMINAMATH_CALUDE_jim_gas_spending_l1080_108001

/-- The amount of gas Jim bought in each state, in gallons -/
def gas_amount : ℝ := 10

/-- The price of gas per gallon in North Carolina, in dollars -/
def nc_price : ℝ := 2

/-- The additional price per gallon in Virginia compared to North Carolina, in dollars -/
def price_difference : ℝ := 1

/-- The total amount Jim spent on gas in both states -/
def total_spent : ℝ := gas_amount * nc_price + gas_amount * (nc_price + price_difference)

theorem jim_gas_spending :
  total_spent = 50 := by sorry

end NUMINAMATH_CALUDE_jim_gas_spending_l1080_108001


namespace NUMINAMATH_CALUDE_golden_section_addition_correct_l1080_108072

/-- The 0.618 method for finding the optimal addition amount --/
def golden_section_addition (a b x : ℝ) : ℝ :=
  a + b - x

/-- Theorem stating the correct formula for the addition point in the 0.618 method --/
theorem golden_section_addition_correct (a b x : ℝ) 
  (h_range : a ≤ x ∧ x ≤ b) 
  (h_good_point : x = a + 0.618 * (b - a)) : 
  golden_section_addition a b x = a + b - x :=
by sorry

end NUMINAMATH_CALUDE_golden_section_addition_correct_l1080_108072


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1080_108038

/-- Given two perpendicular lines with direction vectors (4, -5) and (a, 2), prove that a = 5/2 -/
theorem perpendicular_lines_a_value (a : ℝ) : 
  let v1 : Fin 2 → ℝ := ![4, -5]
  let v2 : Fin 2 → ℝ := ![a, 2]
  (∀ i : Fin 2, (v1 i) * (v2 i) = 0) → a = 5/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l1080_108038


namespace NUMINAMATH_CALUDE_vector_relations_l1080_108085

/-- Two vectors in R² -/
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

/-- Perpendicular vectors have dot product zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Parallel vectors have proportional components -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_relations (x : ℝ) :
  (perpendicular (a x) (b x) → x = 3 ∨ x = -1) ∧
  (parallel (a x) (b x) → x = 0 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_relations_l1080_108085


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l1080_108039

/-- Represents the number of marbles of each color in Jamal's bag -/
structure MarbleCounts where
  blue : ℕ
  green : ℕ
  black : ℕ
  yellow : ℕ

/-- The probability of drawing a black marble from the bag -/
def blackMarbleProbability : ℚ := 1 / 28

/-- The total number of marbles in the bag -/
def totalMarbles (counts : MarbleCounts) : ℕ :=
  counts.blue + counts.green + counts.black + counts.yellow

/-- The theorem stating the number of yellow marbles in Jamal's bag -/
theorem yellow_marbles_count (counts : MarbleCounts) 
  (h_blue : counts.blue = 10)
  (h_green : counts.green = 5)
  (h_black : counts.black = 1)
  (h_prob : (counts.black : ℚ) / (totalMarbles counts) = blackMarbleProbability) :
  counts.yellow = 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l1080_108039


namespace NUMINAMATH_CALUDE_fg_properties_l1080_108016

open Real

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 * x - (a + 1) * log x

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a * x + 4

/-- The sum of f(x) and g(x) -/
noncomputable def sum_fg (a : ℝ) (x : ℝ) : ℝ := f a x + g a x

/-- The difference of f(x) and g(x) -/
noncomputable def diff_fg (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

/-- Theorem stating the conditions for monotonicity and tangency -/
theorem fg_properties :
  (∀ a : ℝ, a ≤ -1 → ∀ x > 0, Monotone (sum_fg a)) ∧
  (∃ a : ℝ, 1 < a ∧ a < 3 ∧ ∃ x > 0, diff_fg a x = 0 ∧ HasDerivAt (diff_fg a) 0 x) :=
sorry

end NUMINAMATH_CALUDE_fg_properties_l1080_108016


namespace NUMINAMATH_CALUDE_compare_negative_decimals_l1080_108027

theorem compare_negative_decimals : -4.3 < -3.4 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_decimals_l1080_108027


namespace NUMINAMATH_CALUDE_dividend_percentage_calculation_l1080_108035

/-- Calculates the dividend percentage given investment details and dividend received -/
theorem dividend_percentage_calculation
  (investment : ℝ)
  (share_face_value : ℝ)
  (premium_percentage : ℝ)
  (dividend_received : ℝ)
  (h1 : investment = 14400)
  (h2 : share_face_value = 100)
  (h3 : premium_percentage = 20)
  (h4 : dividend_received = 720) :
  let share_cost := share_face_value * (1 + premium_percentage / 100)
  let num_shares := investment / share_cost
  let dividend_per_share := dividend_received / num_shares
  let dividend_percentage := (dividend_per_share / share_face_value) * 100
  dividend_percentage = 6 := by
sorry

end NUMINAMATH_CALUDE_dividend_percentage_calculation_l1080_108035


namespace NUMINAMATH_CALUDE_max_chocolate_bars_correct_l1080_108005

/-- The maximum number of chocolate bars Henrique could buy -/
def max_chocolate_bars : ℕ :=
  7

/-- The cost of each chocolate bar in dollars -/
def cost_per_bar : ℚ :=
  135/100

/-- The amount Henrique paid in dollars -/
def amount_paid : ℚ :=
  10

/-- Theorem stating that max_chocolate_bars is the maximum number of bars Henrique could buy -/
theorem max_chocolate_bars_correct :
  (max_chocolate_bars : ℚ) * cost_per_bar < amount_paid ∧
  ((max_chocolate_bars + 1 : ℚ) * cost_per_bar > amount_paid ∨
   amount_paid - (max_chocolate_bars : ℚ) * cost_per_bar ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_chocolate_bars_correct_l1080_108005


namespace NUMINAMATH_CALUDE_jiayuan_supermarket_fruit_weight_l1080_108011

theorem jiayuan_supermarket_fruit_weight :
  let apple_baskets : ℕ := 62
  let pear_baskets : ℕ := 38
  let weight_per_basket : ℕ := 25
  apple_baskets * weight_per_basket + pear_baskets * weight_per_basket = 2500 := by
  sorry

end NUMINAMATH_CALUDE_jiayuan_supermarket_fruit_weight_l1080_108011


namespace NUMINAMATH_CALUDE_inequality_proof_l1080_108042

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c ≥ a * b * c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1080_108042


namespace NUMINAMATH_CALUDE_bijective_function_decomposition_l1080_108021

theorem bijective_function_decomposition
  (f : ℤ → ℤ) (hf : Function.Bijective f) :
  ∃ (u v : ℤ → ℤ), Function.Bijective u ∧ Function.Bijective v ∧ (∀ x, f x = u x + v x) := by
  sorry

end NUMINAMATH_CALUDE_bijective_function_decomposition_l1080_108021


namespace NUMINAMATH_CALUDE_second_order_de_solution_l1080_108081

/-- Given a second-order linear homogeneous differential equation with constant coefficients:
    y'' - 5y' - 6y = 0, prove that y = C₁e^(6x) + C₂e^(-x) is the general solution. -/
theorem second_order_de_solution (y : ℝ → ℝ) (C₁ C₂ : ℝ) :
  (∀ x, (deriv^[2] y) x - 5 * (deriv y) x - 6 * y x = 0) ↔
  (∃ C₁ C₂, ∀ x, y x = C₁ * Real.exp (6 * x) + C₂ * Real.exp (-x)) :=
sorry


end NUMINAMATH_CALUDE_second_order_de_solution_l1080_108081


namespace NUMINAMATH_CALUDE_baseball_team_members_l1080_108064

theorem baseball_team_members (
  pouches_per_pack : ℕ)
  (num_coaches : ℕ)
  (num_helpers : ℕ)
  (num_packs : ℕ)
  (h1 : pouches_per_pack = 6)
  (h2 : num_coaches = 3)
  (h3 : num_helpers = 2)
  (h4 : num_packs = 3)
  : ∃ (team_members : ℕ),
    team_members = num_packs * pouches_per_pack - num_coaches - num_helpers ∧
    team_members = 13 :=
by sorry

end NUMINAMATH_CALUDE_baseball_team_members_l1080_108064


namespace NUMINAMATH_CALUDE_initial_goldfish_count_l1080_108074

theorem initial_goldfish_count (died : ℕ) (remaining : ℕ) (h1 : died = 32) (h2 : remaining = 57) :
  died + remaining = 89 := by
  sorry

end NUMINAMATH_CALUDE_initial_goldfish_count_l1080_108074


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l1080_108026

/-- The hyperbola equation -/
def hyperbola_equation (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 5 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (3, 0)

/-- The asymptotes of the hyperbola are tangent to the circle -/
def asymptotes_tangent_to_circle (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y ∧ 
    (y = (a/b) * (x - 3) ∨ y = -(a/b) * (x - 3))

/-- The right focus of the hyperbola is the center of the circle -/
def right_focus_is_circle_center (a : ℝ) : Prop :=
  (a, 0) = circle_center

/-- The main theorem -/
theorem hyperbola_parameters :
  ∀ (a b : ℝ), 
    a > 0 → b > 0 →
    asymptotes_tangent_to_circle a b →
    right_focus_is_circle_center a →
    a^2 = 4 ∧ b^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l1080_108026


namespace NUMINAMATH_CALUDE_shaded_portion_is_four_ninths_l1080_108056

-- Define the square ABCD
def square_side : ℝ := 6

-- Define the shaded areas
def shaded_area_1 : ℝ := 2 * 2
def shaded_area_2 : ℝ := 4 * 4 - 2 * 2
def shaded_area_3 : ℝ := 6 * 6

-- Total square area
def total_area : ℝ := square_side * square_side

-- Total shaded area
def total_shaded_area : ℝ := shaded_area_1 + shaded_area_2

-- Theorem to prove
theorem shaded_portion_is_four_ninths :
  total_shaded_area / total_area = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_shaded_portion_is_four_ninths_l1080_108056


namespace NUMINAMATH_CALUDE_cubic_fifth_power_roots_l1080_108057

/-- The roots of a cubic polynomial x^3 + ax^2 + bx + c = 0 are the fifth powers of the roots of x^3 - 3x + 1 = 0 if and only if a = 15, b = -198, and c = 1 -/
theorem cubic_fifth_power_roots (a b c : ℝ) : 
  (∀ x : ℂ, x^3 - 3*x + 1 = 0 → ∃ y : ℂ, y^3 + a*y^2 + b*y + c = 0 ∧ y = x^5) ↔ 
  (a = 15 ∧ b = -198 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_fifth_power_roots_l1080_108057


namespace NUMINAMATH_CALUDE_fifty_percent_relation_l1080_108028

theorem fifty_percent_relation (x y : ℝ) : 
  (0.5 * x = y + 20) → (x - 2 * y = 40) := by
  sorry

end NUMINAMATH_CALUDE_fifty_percent_relation_l1080_108028


namespace NUMINAMATH_CALUDE_martin_additional_hens_l1080_108054

/-- Represents the farm's hen and egg production scenario --/
structure FarmScenario where
  initial_hens : ℕ
  initial_days : ℕ
  initial_eggs : ℕ
  final_days : ℕ
  final_eggs : ℕ

/-- Calculates the number of additional hens needed --/
def additional_hens_needed (scenario : FarmScenario) : ℕ :=
  let eggs_per_hen := scenario.final_eggs * scenario.initial_days / (scenario.final_days * scenario.initial_eggs)
  let total_hens_needed := scenario.final_eggs / (eggs_per_hen * scenario.final_days / scenario.initial_days)
  total_hens_needed - scenario.initial_hens

/-- The main theorem stating the number of additional hens Martin needs to buy --/
theorem martin_additional_hens :
  let scenario : FarmScenario := {
    initial_hens := 10,
    initial_days := 10,
    initial_eggs := 80,
    final_days := 15,
    final_eggs := 300
  }
  additional_hens_needed scenario = 15 := by
  sorry

end NUMINAMATH_CALUDE_martin_additional_hens_l1080_108054


namespace NUMINAMATH_CALUDE_first_day_over_200_paperclips_l1080_108099

def paperclips (k : ℕ) : ℕ := 3 * 2^k

theorem first_day_over_200_paperclips :
  (∀ j : ℕ, j < 8 → paperclips j ≤ 200) ∧ paperclips 8 > 200 :=
by sorry

end NUMINAMATH_CALUDE_first_day_over_200_paperclips_l1080_108099


namespace NUMINAMATH_CALUDE_f_properties_l1080_108078

-- Define the function f(x) = lg |sin x|
noncomputable def f (x : ℝ) : ℝ := Real.log (|Real.sin x|)

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = f x) ∧                        -- f is even
  (∀ x, f (x + π) = f x) ∧                     -- f has period π
  (∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f x < f y) -- f is monotonically increasing on (0, π/2)
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l1080_108078


namespace NUMINAMATH_CALUDE_cube_sum_gt_mixed_product_l1080_108073

theorem cube_sum_gt_mixed_product (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_gt_mixed_product_l1080_108073


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1080_108009

-- Define the coefficients of the quadratic equation -16x^2 + 72x - 90 = 0
def a : ℝ := -16
def b : ℝ := 72
def c : ℝ := -90

-- Theorem stating the sum of solutions and absence of positive real solutions
theorem quadratic_equation_properties :
  (let sum_of_solutions := -b / a
   sum_of_solutions = 4.5) ∧
  (∀ x : ℝ, -16 * x^2 + 72 * x - 90 ≠ 0 ∨ x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1080_108009


namespace NUMINAMATH_CALUDE_a_10_equals_1000_l1080_108045

def a (n : ℕ) : ℕ :=
  let first_odd := 2 * n - 1
  let last_odd := first_odd + 2 * (n - 1)
  n * (first_odd + last_odd) / 2

theorem a_10_equals_1000 : a 10 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_a_10_equals_1000_l1080_108045


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l1080_108032

def q (x : ℝ) : ℝ := -10 * x^2 + 40 * x - 30

def numerator (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 4 * x + 6

theorem q_satisfies_conditions :
  (∀ x, x ≠ 1 ∧ x ≠ 3 → q x ≠ 0) ∧
  (q 1 = 0 ∧ q 3 = 0) ∧
  (∀ x, x ≠ 1 ∧ x ≠ 3 → ∃ y, y = numerator x / q x) ∧
  (¬ ∃ L, ∀ ε > 0, ∃ δ > 0, ∀ x, |x| > δ → |numerator x / q x - L| < ε) ∧
  q 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l1080_108032


namespace NUMINAMATH_CALUDE_myfavorite_sum_l1080_108006

def letters : Finset Char := {'m', 'y', 'f', 'a', 'v', 'o', 'r', 'i', 't', 'e'}
def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem myfavorite_sum (f : Char → Nat) 
  (h1 : Function.Bijective f)
  (h2 : ∀ c ∈ letters, f c ∈ digits) :
  (letters.sum fun c => f c) = 45 := by
  sorry

end NUMINAMATH_CALUDE_myfavorite_sum_l1080_108006


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_problem_l1080_108044

/-- The length of a bridge given train parameters --/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

/-- Proof of the bridge length problem --/
theorem bridge_length_problem : 
  bridge_length 360 75 24 = 140 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_problem_l1080_108044


namespace NUMINAMATH_CALUDE_multiply_negative_with_absolute_value_l1080_108023

theorem multiply_negative_with_absolute_value : (-3.6 : ℝ) * |(-2 : ℝ)| = -7.2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_negative_with_absolute_value_l1080_108023


namespace NUMINAMATH_CALUDE_complete_square_l1080_108025

theorem complete_square (b : ℝ) : ∀ x : ℝ, x^2 + b*x = (x + b/2)^2 - (b/2)^2 := by sorry

end NUMINAMATH_CALUDE_complete_square_l1080_108025


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l1080_108066

theorem complex_modulus_squared (z : ℂ) (h : z * Complex.abs z = 3 + 12*I) : Complex.abs z ^ 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l1080_108066


namespace NUMINAMATH_CALUDE_expression_evaluation_l1080_108090

theorem expression_evaluation : -2^3 + (18 - (-3)^2) / (-3) = -11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1080_108090


namespace NUMINAMATH_CALUDE_general_formula_minimize_s_l1080_108017

-- Define the sequence and its sum
def s (n : ℕ) : ℤ := 2 * n^2 - 30 * n

-- Define the general term of the sequence
def a (n : ℕ) : ℤ := 4 * n - 32

-- Theorem 1: The general formula for a_n is 4n - 32
theorem general_formula : ∀ n : ℕ, a n = s n - s (n - 1) := by sorry

-- Theorem 2: s_n is minimized when n = 7 or n = 8
theorem minimize_s : ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ ∀ m : ℕ, s n ≤ s m := by sorry

end NUMINAMATH_CALUDE_general_formula_minimize_s_l1080_108017


namespace NUMINAMATH_CALUDE_runner_distance_l1080_108052

/-- Represents the runner's problem --/
def RunnerProblem (speed time distance : ℝ) : Prop :=
  -- Normal condition
  speed * time = distance ∧
  -- Increased speed condition
  (speed + 1) * (2/3 * time) = distance ∧
  -- Decreased speed condition
  (speed - 1) * (time + 3) = distance

/-- Theorem stating the solution to the runner's problem --/
theorem runner_distance : ∃ (speed time : ℝ), RunnerProblem speed time 6 := by
  sorry


end NUMINAMATH_CALUDE_runner_distance_l1080_108052


namespace NUMINAMATH_CALUDE_unique_numbers_satisfying_conditions_l1080_108033

theorem unique_numbers_satisfying_conditions : 
  ∃! (x y : ℕ), 
    10 ≤ x ∧ x < 100 ∧
    100 ≤ y ∧ y < 1000 ∧
    1000 * x + y = 8 * x * y ∧
    x + y = 141 := by sorry

end NUMINAMATH_CALUDE_unique_numbers_satisfying_conditions_l1080_108033


namespace NUMINAMATH_CALUDE_solve_for_x_l1080_108080

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1080_108080


namespace NUMINAMATH_CALUDE_marcus_pies_l1080_108013

/-- The number of pies Marcus can fit in his oven at once -/
def oven_capacity : ℕ := 5

/-- The number of pies Marcus dropped -/
def dropped_pies : ℕ := 8

/-- The number of pies Marcus has left -/
def remaining_pies : ℕ := 27

/-- The number of batches Marcus baked -/
def batches : ℕ := 7

theorem marcus_pies :
  oven_capacity * batches = remaining_pies + dropped_pies :=
sorry

end NUMINAMATH_CALUDE_marcus_pies_l1080_108013


namespace NUMINAMATH_CALUDE_polynomial_rational_difference_l1080_108007

theorem polynomial_rational_difference (f : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (∀ x y : ℝ, ∃ q : ℚ, x - y = q → ∃ r : ℚ, f x - f y = r) →
  ∃ b : ℚ, ∃ c : ℝ, ∀ x, f x = b * x + c :=
sorry

end NUMINAMATH_CALUDE_polynomial_rational_difference_l1080_108007


namespace NUMINAMATH_CALUDE_sqrt_22_greater_than_4_l1080_108097

theorem sqrt_22_greater_than_4 : Real.sqrt 22 > 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_22_greater_than_4_l1080_108097


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1080_108004

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Right-angled triangle condition
  a^2 + b^2 + c^2 = 2500 →  -- Sum of squares condition
  c = 25 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1080_108004


namespace NUMINAMATH_CALUDE_pet_food_inventory_l1080_108000

theorem pet_food_inventory (dog_food : ℕ) (difference : ℕ) (cat_food : ℕ) : 
  dog_food = 600 → 
  dog_food = cat_food + difference → 
  difference = 273 →
  cat_food = 327 := by
  sorry

end NUMINAMATH_CALUDE_pet_food_inventory_l1080_108000


namespace NUMINAMATH_CALUDE_difference_of_squares_l1080_108098

theorem difference_of_squares (a b : ℝ) (h1 : a + b = 75) (h2 : a - b = 15) :
  a^2 - b^2 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1080_108098


namespace NUMINAMATH_CALUDE_right_triangle_area_l1080_108022

/-- The area of a right triangle with base 12 and height 15 is 90 -/
theorem right_triangle_area : ∀ (base height area : ℝ),
  base = 12 →
  height = 15 →
  area = (1/2) * base * height →
  area = 90 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1080_108022


namespace NUMINAMATH_CALUDE_vector_parallelism_l1080_108040

/-- Given two 2D vectors a and b, prove that when k*a + b is parallel to a - 3*b, k = -1/3 --/
theorem vector_parallelism (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : ∃ (t : ℝ), t ≠ 0 ∧ k • a + b = t • (a - 3 • b)) :
  k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallelism_l1080_108040


namespace NUMINAMATH_CALUDE_min_value_implies_a_possible_a_set_l1080_108096

/-- The quadratic function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The theorem stating the possible values of a -/
theorem min_value_implies_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (a - 2) (a + 2), f x ≥ 6) ∧ 
  (∃ x ∈ Set.Icc (a - 2) (a + 2), f x = 6) →
  a = -3 ∨ a = 5 := by
  sorry

/-- The set of possible values for a -/
def possible_a : Set ℝ := {-3, 5}

/-- The theorem stating that the set of possible values for a is {-3, 5} -/
theorem possible_a_set : 
  ∀ a : ℝ, (∀ x ∈ Set.Icc (a - 2) (a + 2), f x ≥ 6) ∧ 
            (∃ x ∈ Set.Icc (a - 2) (a + 2), f x = 6) ↔ 
            a ∈ possible_a := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_possible_a_set_l1080_108096


namespace NUMINAMATH_CALUDE_wendy_camera_pictures_l1080_108030

/-- Represents the number of pictures in Wendy's photo upload scenario -/
structure WendyPictures where
  phone : ℕ
  albums : ℕ
  per_album : ℕ

/-- The number of pictures Wendy uploaded from her camera -/
def camera_pictures (w : WendyPictures) : ℕ :=
  w.albums * w.per_album - w.phone

/-- Theorem stating the number of pictures Wendy uploaded from her camera -/
theorem wendy_camera_pictures :
  ∀ (w : WendyPictures),
    w.phone = 22 →
    w.albums = 4 →
    w.per_album = 6 →
    camera_pictures w = 2 := by
  sorry

end NUMINAMATH_CALUDE_wendy_camera_pictures_l1080_108030


namespace NUMINAMATH_CALUDE_optimal_discount_order_l1080_108031

def original_price : ℝ := 30
def flat_discount : ℝ := 5
def percentage_discount : ℝ := 0.25

theorem optimal_discount_order :
  (original_price * (1 - percentage_discount) - flat_discount) -
  (original_price - flat_discount) * (1 - percentage_discount) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_optimal_discount_order_l1080_108031


namespace NUMINAMATH_CALUDE_max_balances_correct_max_balances_achievable_l1080_108036

/-- Represents a set of unique weights -/
def UniqueWeights (n : ℕ) := Fin n → ℝ

/-- Represents the state of the balance scale -/
structure BalanceState where
  left : List ℝ
  right : List ℝ

/-- Checks if the balance scale is in equilibrium -/
def isBalanced (state : BalanceState) : Prop :=
  state.left.sum = state.right.sum

/-- Represents a sequence of weight placements -/
def WeightPlacement (n : ℕ) := Fin n → Bool × Fin n

/-- Counts the number of times the scale balances during a sequence of weight placements -/
def countBalances (weights : UniqueWeights 2021) (placements : WeightPlacement m) : ℕ :=
  sorry

/-- The maximum number of times the scale can balance -/
def maxBalances : ℕ := 673

theorem max_balances_correct (weights : UniqueWeights 2021) :
  ∀ (m : ℕ) (placements : WeightPlacement m),
    countBalances weights placements ≤ maxBalances :=
  sorry

theorem max_balances_achievable :
  ∃ (weights : UniqueWeights 2021) (m : ℕ) (placements : WeightPlacement m),
    countBalances weights placements = maxBalances :=
  sorry

end NUMINAMATH_CALUDE_max_balances_correct_max_balances_achievable_l1080_108036


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l1080_108087

theorem cubic_root_equation_solution :
  ∃! x : ℝ, 2.61 * (9 - Real.sqrt (x + 1))^(1/3) + (7 + Real.sqrt (x + 1))^(1/3) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l1080_108087


namespace NUMINAMATH_CALUDE_school_event_water_drinkers_l1080_108086

/-- Proves that 60 students chose water given the conditions of the school event -/
theorem school_event_water_drinkers (total : ℕ) (juice_percent soda_percent : ℚ) 
  (soda_count : ℕ) : 
  juice_percent = 1/2 →
  soda_percent = 3/10 →
  soda_count = 90 →
  total = soda_count / soda_percent →
  (1 - juice_percent - soda_percent) * total = 60 :=
by
  sorry

#check school_event_water_drinkers

end NUMINAMATH_CALUDE_school_event_water_drinkers_l1080_108086


namespace NUMINAMATH_CALUDE_modular_congruence_iff_divisibility_l1080_108077

theorem modular_congruence_iff_divisibility (a n k : ℕ) (ha : a ≥ 2) :
  a ^ k ≡ 1 [MOD a ^ n - 1] ↔ n ∣ k :=
sorry

end NUMINAMATH_CALUDE_modular_congruence_iff_divisibility_l1080_108077


namespace NUMINAMATH_CALUDE_distance_to_center_l1080_108075

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 98}

-- Define the properties of the points
def PointProperties (A B C : ℝ × ℝ) : Prop :=
  A ∈ Circle ∧ B ∈ Circle ∧ C ∈ Circle ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64 ∧  -- AB = 8
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 9 ∧   -- BC = 3
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0  -- Angle ABC is right

-- The theorem
theorem distance_to_center (A B C : ℝ × ℝ) :
  PointProperties A B C → B.1^2 + B.2^2 = 50 := by sorry

end NUMINAMATH_CALUDE_distance_to_center_l1080_108075


namespace NUMINAMATH_CALUDE_sin_390_degrees_l1080_108065

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_390_degrees_l1080_108065


namespace NUMINAMATH_CALUDE_nines_in_hundred_l1080_108084

def count_nines (n : ℕ) : ℕ :=
  (n / 10) + (n - n / 10 * 10)

theorem nines_in_hundred : count_nines 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_nines_in_hundred_l1080_108084


namespace NUMINAMATH_CALUDE_solve_system_l1080_108092

theorem solve_system (a b : ℚ) 
  (eq1 : 5 + 2 * a = 6 - 3 * b) 
  (eq2 : 3 + 4 * b = 10 + 2 * a) : 
  5 - 2 * a = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1080_108092


namespace NUMINAMATH_CALUDE_dice_probability_l1080_108068

theorem dice_probability (p_neither : ℚ) (h : p_neither = 4/9) : 
  1 - p_neither = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l1080_108068


namespace NUMINAMATH_CALUDE_certain_number_proof_l1080_108043

theorem certain_number_proof (k : ℕ) (x : ℕ) 
  (h1 : 823435 % (15^k) = 0)
  (h2 : x^k - k^5 = 1) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1080_108043


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1080_108091

/-- Given two quadratic equations and a relationship between their roots, prove the value of k. -/
theorem quadratic_root_relation (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 9 = 0 → ∃ y : ℝ, y^2 - k*y + 9 = 0 ∧ y = x + 3) →
  k = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1080_108091


namespace NUMINAMATH_CALUDE_total_wheels_at_park_l1080_108069

-- Define the number of regular bikes
def regular_bikes : ℕ := 7

-- Define the number of children's bikes
def children_bikes : ℕ := 11

-- Define the number of wheels on a regular bike
def regular_bike_wheels : ℕ := 2

-- Define the number of wheels on a children's bike
def children_bike_wheels : ℕ := 4

-- Theorem: The total number of wheels Naomi saw at the park is 58
theorem total_wheels_at_park : 
  regular_bikes * regular_bike_wheels + children_bikes * children_bike_wheels = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_at_park_l1080_108069


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l1080_108089

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) : 
  (2 * r) ^ 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l1080_108089


namespace NUMINAMATH_CALUDE_line_parameterization_l1080_108061

/-- Given a line y = 2x - 10 parameterized by (x, y) = (g(t), 20t - 8), 
    prove that g(t) = 10t + 1 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t x y, y = 2*x - 10 ∧ x = g t ∧ y = 20*t - 8) → 
  (∀ t, g t = 10*t + 1) :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_l1080_108061


namespace NUMINAMATH_CALUDE_cross_section_distance_l1080_108029

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  /-- The distance from the apex to the base -/
  height : ℝ
  /-- The side length of the base hexagon -/
  baseSide : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- The distance from the apex to the cross section -/
  distanceFromApex : ℝ
  /-- The area of the cross section -/
  area : ℝ

/-- The theorem to be proved -/
theorem cross_section_distance (pyramid : RightHexagonalPyramid) 
  (section1 section2 : CrossSection) :
  section1.area = 162 * Real.sqrt 3 →
  section2.area = 288 * Real.sqrt 3 →
  |section1.distanceFromApex - section2.distanceFromApex| = 6 →
  max section1.distanceFromApex section2.distanceFromApex = 24 :=
by sorry

end NUMINAMATH_CALUDE_cross_section_distance_l1080_108029


namespace NUMINAMATH_CALUDE_samantha_bus_time_l1080_108058

/-- Represents Samantha's daily schedule --/
structure Schedule where
  wakeUpTime : Nat
  busTime : Nat
  classCount : Nat
  classDuration : Nat
  lunchDuration : Nat
  chessClubDuration : Nat
  arrivalTime : Nat

/-- Calculates the total time spent on the bus given a schedule --/
def busTimeDuration (s : Schedule) : Nat :=
  let totalAwayTime := s.arrivalTime - s.busTime
  let totalSchoolTime := s.classCount * s.classDuration + s.lunchDuration + s.chessClubDuration
  totalAwayTime - totalSchoolTime

/-- Samantha's actual schedule --/
def samanthaSchedule : Schedule :=
  { wakeUpTime := 7 * 60
    busTime := 8 * 60
    classCount := 7
    classDuration := 45
    lunchDuration := 45
    chessClubDuration := 90
    arrivalTime := 17 * 60 + 30 }

/-- Theorem stating that Samantha spends 120 minutes on the bus --/
theorem samantha_bus_time :
  busTimeDuration samanthaSchedule = 120 := by
  sorry

end NUMINAMATH_CALUDE_samantha_bus_time_l1080_108058


namespace NUMINAMATH_CALUDE_unique_winning_combination_l1080_108095

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 60 ∧ ∃ (m n : ℕ), n = 2^m * 3^n

def is_valid_combination (combo : Finset ℕ) : Prop :=
  combo.card = 5 ∧ 
  ∀ n ∈ combo, is_valid_number n ∧
  ∃ k : ℕ, (combo.prod id) = 12^k

theorem unique_winning_combination : 
  ∃! combo : Finset ℕ, is_valid_combination combo :=
sorry

end NUMINAMATH_CALUDE_unique_winning_combination_l1080_108095


namespace NUMINAMATH_CALUDE_car_average_speed_l1080_108010

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 100) (h2 : speed2 = 80) :
  (speed1 + speed2) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l1080_108010


namespace NUMINAMATH_CALUDE_dice_sum_theorem_l1080_108082

/-- Represents a single die -/
structure Die where
  opposite_sum : ℕ
  opposite_sum_is_seven : opposite_sum = 7

/-- Represents a set of 7 dice -/
structure DiceSet where
  dice : Fin 7 → Die
  all_dice_have_opposite_sum_seven : ∀ i, (dice i).opposite_sum = 7

/-- The sum of numbers on the upward faces of a set of dice -/
def upward_sum (d : DiceSet) : ℕ := sorry

/-- The sum of numbers on the downward faces of a set of dice -/
def downward_sum (d : DiceSet) : ℕ := sorry

/-- The probability of getting a specific sum on the upward faces -/
noncomputable def prob_upward_sum (d : DiceSet) (sum : ℕ) : ℝ := sorry

/-- The probability of getting a specific sum on the downward faces -/
noncomputable def prob_downward_sum (d : DiceSet) (sum : ℕ) : ℝ := sorry

theorem dice_sum_theorem (d : DiceSet) (a : ℕ) 
  (h1 : a ≠ 10)
  (h2 : prob_upward_sum d 10 = prob_downward_sum d a) :
  a = 39 := by sorry

end NUMINAMATH_CALUDE_dice_sum_theorem_l1080_108082


namespace NUMINAMATH_CALUDE_abc_sum_l1080_108076

theorem abc_sum (a b c : ℕ+) 
  (eq1 : a * b + c + 10 = 51)
  (eq2 : b * c + a + 10 = 51)
  (eq3 : a * c + b + 10 = 51) :
  a + b + c = 41 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_l1080_108076


namespace NUMINAMATH_CALUDE_cos_neg_thirty_degrees_l1080_108012

theorem cos_neg_thirty_degrees : Real.cos (-(30 * π / 180)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_thirty_degrees_l1080_108012


namespace NUMINAMATH_CALUDE_school_colors_percentage_l1080_108055

theorem school_colors_percentage (N : ℝ) (h_pos : N > 0) : 
  let girls := 0.45 * N
  let boys := N - girls
  let girls_in_colors := 0.60 * girls
  let boys_in_colors := 0.80 * boys
  let total_in_colors := girls_in_colors + boys_in_colors
  (total_in_colors / N) = 0.71 := by
  sorry

end NUMINAMATH_CALUDE_school_colors_percentage_l1080_108055


namespace NUMINAMATH_CALUDE_triangle_theorem_l1080_108002

open Real

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * cos (2 * t.C) + 2 * t.c * cos t.A * cos t.C + t.a + t.b = 0)
  (h2 : t.b = 4 * sin t.B) : 
  t.C = 2 * π / 3 ∧ 
  (∀ S : ℝ, S = 1/2 * t.a * t.b * sin t.C → S ≤ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1080_108002


namespace NUMINAMATH_CALUDE_devin_taught_calculus_four_years_l1080_108041

/-- Represents the number of years Devin taught each subject --/
structure TeachingYears where
  calculus : ℕ
  algebra : ℕ
  statistics : ℕ

/-- Defines the conditions of Devin's teaching career --/
def satisfiesConditions (y : TeachingYears) : Prop :=
  y.algebra = 2 * y.calculus ∧
  y.statistics = 5 * y.algebra ∧
  y.calculus + y.algebra + y.statistics = 52

/-- Theorem stating that given the conditions, Devin taught Calculus for 4 years --/
theorem devin_taught_calculus_four_years :
  ∃ y : TeachingYears, satisfiesConditions y ∧ y.calculus = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_devin_taught_calculus_four_years_l1080_108041


namespace NUMINAMATH_CALUDE_evaluate_64_to_5_6_l1080_108050

theorem evaluate_64_to_5_6 : (64 : ℝ) ^ (5/6) = 32 := by sorry

end NUMINAMATH_CALUDE_evaluate_64_to_5_6_l1080_108050


namespace NUMINAMATH_CALUDE_number_order_l1080_108018

/-- Convert a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- Definition of 85 in base 9 --/
def num_85_base9 : Nat := to_decimal [5, 8] 9

/-- Definition of 210 in base 6 --/
def num_210_base6 : Nat := to_decimal [0, 1, 2] 6

/-- Definition of 1000 in base 4 --/
def num_1000_base4 : Nat := to_decimal [0, 0, 0, 1] 4

/-- Definition of 111111 in base 2 --/
def num_111111_base2 : Nat := to_decimal [1, 1, 1, 1, 1, 1] 2

/-- Theorem stating the order of the numbers --/
theorem number_order :
  num_210_base6 > num_85_base9 ∧
  num_85_base9 > num_1000_base4 ∧
  num_1000_base4 > num_111111_base2 :=
by sorry

end NUMINAMATH_CALUDE_number_order_l1080_108018


namespace NUMINAMATH_CALUDE_opposite_sides_of_line_l1080_108059

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in the form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to evaluate the line equation at a point
def evaluateLine (l : Line2D) (p : Point2D) : ℝ :=
  l.a * p.x + l.b * p.y + l.c

-- Define the specific line and points
def line : Line2D := { a := -3, b := 1, c := 2 }
def origin : Point2D := { x := 0, y := 0 }
def point : Point2D := { x := 2, y := 1 }

-- Theorem statement
theorem opposite_sides_of_line :
  evaluateLine line origin * evaluateLine line point < 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_of_line_l1080_108059


namespace NUMINAMATH_CALUDE_factorization_of_cubic_l1080_108034

theorem factorization_of_cubic (b : ℝ) : 2 * b^3 - 4 * b^2 + 2 * b = 2 * b * (b - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_l1080_108034


namespace NUMINAMATH_CALUDE_banana_proportion_after_adding_l1080_108020

/-- Represents a fruit basket with apples and bananas -/
structure FruitBasket where
  apples : ℕ
  bananas : ℕ

/-- Calculates the fraction of bananas in the basket -/
def bananaProportion (basket : FruitBasket) : ℚ :=
  basket.bananas / (basket.apples + basket.bananas)

/-- The initial basket -/
def initialBasket : FruitBasket := ⟨12, 15⟩

/-- The basket after adding 3 bananas -/
def finalBasket : FruitBasket := ⟨initialBasket.apples, initialBasket.bananas + 3⟩

/-- Theorem stating that the proportion of bananas in the final basket is 3/5 -/
theorem banana_proportion_after_adding : bananaProportion finalBasket = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_banana_proportion_after_adding_l1080_108020


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l1080_108047

def p (x : ℝ) : ℝ := x^3 - 8*x^2 + 19*x - 12

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 3 ∨ x = 4) ∧
  p 1 = 0 ∧ p 3 = 0 ∧ p 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l1080_108047


namespace NUMINAMATH_CALUDE_first_day_exceeding_500_l1080_108024

def bacteria_count (initial_count : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_count * growth_factor ^ days

theorem first_day_exceeding_500 :
  let initial_count := 4
  let growth_factor := 3
  let target := 500
  (∀ d : ℕ, d < 6 → bacteria_count initial_count growth_factor d ≤ target) ∧
  (bacteria_count initial_count growth_factor 6 > target) :=
by sorry

end NUMINAMATH_CALUDE_first_day_exceeding_500_l1080_108024


namespace NUMINAMATH_CALUDE_kim_dropped_one_class_l1080_108093

/-- Calculates the number of classes dropped given the initial number of classes,
    hours per class, and remaining total hours of classes. -/
def classes_dropped (initial_classes : ℕ) (hours_per_class : ℕ) (remaining_hours : ℕ) : ℕ :=
  (initial_classes * hours_per_class - remaining_hours) / hours_per_class

theorem kim_dropped_one_class :
  classes_dropped 4 2 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_kim_dropped_one_class_l1080_108093


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l1080_108049

/-- Given two circles and a line tangent to both, proves the x-coordinate of the intersection point --/
theorem tangent_line_intersection (r1 r2 c2_x : ℝ) (h1 : r1 = 2) (h2 : r2 = 7) (h3 : c2_x = 15) :
  ∃ x : ℝ, x > 0 ∧ (r1 / x = r2 / (c2_x - x)) ∧ x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l1080_108049


namespace NUMINAMATH_CALUDE_max_k_inequality_l1080_108070

theorem max_k_inequality (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∃ (k : ℝ), ∀ m, 0 < m → m < 1/2 → 1/m + 2/(1-2*m) ≥ k) ∧ 
  (∀ k, (∀ m, 0 < m → m < 1/2 → 1/m + 2/(1-2*m) ≥ k) → k ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_max_k_inequality_l1080_108070


namespace NUMINAMATH_CALUDE_ginos_bears_l1080_108053

/-- The number of brown bears Gino has -/
def brown_bears : ℕ := 15

/-- The number of white bears Gino has -/
def white_bears : ℕ := 24

/-- The number of black bears Gino has -/
def black_bears : ℕ := 27

/-- The number of polar bears Gino has -/
def polar_bears : ℕ := 12

/-- The number of grizzly bears Gino has -/
def grizzly_bears : ℕ := 18

/-- The total number of bears Gino has -/
def total_bears : ℕ := brown_bears + white_bears + black_bears + polar_bears + grizzly_bears

theorem ginos_bears : total_bears = 96 := by
  sorry

end NUMINAMATH_CALUDE_ginos_bears_l1080_108053


namespace NUMINAMATH_CALUDE_garden_perimeter_l1080_108094

/-- Given a square garden with area q and perimeter p, if q = 2p + 20, then p = 40 -/
theorem garden_perimeter (q p : ℝ) (h1 : q > 0) (h2 : p > 0) (h3 : q = p^2 / 16) (h4 : q = 2*p + 20) : p = 40 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1080_108094


namespace NUMINAMATH_CALUDE_y_coordinate_of_C_l1080_108003

-- Define the pentagon ABCDE
structure Pentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

-- Define the properties of the pentagon
def symmetricPentagon (p : Pentagon) : Prop :=
  p.A.1 = 0 ∧ p.A.2 = 0 ∧
  p.B.1 = 0 ∧ p.B.2 = 5 ∧
  p.D.1 = 5 ∧ p.D.2 = 5 ∧
  p.E.1 = 5 ∧ p.E.2 = 0 ∧
  p.C.1 = 2.5 -- Vertical line of symmetry

-- Define the area of the pentagon
def pentagonArea (p : Pentagon) : ℝ :=
  50 -- Given area

-- Theorem: The y-coordinate of vertex C is 15
theorem y_coordinate_of_C (p : Pentagon) 
  (h1 : symmetricPentagon p) 
  (h2 : pentagonArea p = 50) : 
  p.C.2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_of_C_l1080_108003


namespace NUMINAMATH_CALUDE_fred_grew_38_cantaloupes_l1080_108088

/-- The number of cantaloupes Tim grew -/
def tims_cantaloupes : ℕ := 44

/-- The total number of cantaloupes Fred and Tim grew together -/
def total_cantaloupes : ℕ := 82

/-- The number of cantaloupes Fred grew -/
def freds_cantaloupes : ℕ := total_cantaloupes - tims_cantaloupes

theorem fred_grew_38_cantaloupes : freds_cantaloupes = 38 := by
  sorry

end NUMINAMATH_CALUDE_fred_grew_38_cantaloupes_l1080_108088


namespace NUMINAMATH_CALUDE_rebecca_camping_items_l1080_108067

/-- Represents the number of tent stakes Rebecca bought. -/
def tent_stakes : ℕ := sorry

/-- Represents the number of packets of drink mix Rebecca bought. -/
def drink_mix : ℕ := 3 * tent_stakes

/-- Represents the number of bottles of water Rebecca bought. -/
def water_bottles : ℕ := tent_stakes + 2

/-- The total number of items Rebecca bought. -/
def total_items : ℕ := 22

theorem rebecca_camping_items : tent_stakes = 4 :=
  by sorry

end NUMINAMATH_CALUDE_rebecca_camping_items_l1080_108067


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l1080_108046

/-- Given a function f(x) = x^3 + ax^2 + 3x - 9 with an extreme value at x = -3, prove that a = 5 -/
theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + 3*x - 9
  (∃ (ε : ℝ), ∀ (x : ℝ), x ≠ -3 → |x + 3| < ε → f x ≤ f (-3) ∨ f x ≥ f (-3)) →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l1080_108046


namespace NUMINAMATH_CALUDE_middle_number_proof_l1080_108048

theorem middle_number_proof (x y z : ℕ) 
  (sum_xy : x + y = 22)
  (sum_xz : x + z = 29)
  (sum_yz : y + z = 37)
  (h_order : x < y ∧ y < z) : y = 15 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l1080_108048


namespace NUMINAMATH_CALUDE_toys_per_day_l1080_108014

def total_weekly_production : ℕ := 5505
def working_days_per_week : ℕ := 5

theorem toys_per_day :
  total_weekly_production / working_days_per_week = 1101 :=
by
  sorry

end NUMINAMATH_CALUDE_toys_per_day_l1080_108014


namespace NUMINAMATH_CALUDE_moose_population_canada_l1080_108037

theorem moose_population_canada :
  ∀ (moose beaver human : ℕ),
    beaver = 2 * moose →
    human = 19 * beaver →
    human = 38000000 →
    moose = 1000000 :=
by
  sorry

end NUMINAMATH_CALUDE_moose_population_canada_l1080_108037
