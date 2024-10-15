import Mathlib

namespace NUMINAMATH_CALUDE_line_through_1_0_perpendicular_to_polar_axis_l1911_191106

-- Define the polar coordinate system
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

-- Define a line in polar coordinates
structure PolarLine where
  equation : PolarPoint → Prop

-- Define the polar axis
def polarAxis : PolarLine :=
  { equation := fun p => p.θ = 0 }

-- Define perpendicularity in polar coordinates
def perpendicular (l1 l2 : PolarLine) : Prop :=
  sorry

-- Define the point (1, 0) in polar coordinates
def point_1_0 : PolarPoint :=
  { ρ := 1, θ := 0 }

-- The theorem to be proved
theorem line_through_1_0_perpendicular_to_polar_axis :
  ∃ (l : PolarLine),
    l.equation = fun p => p.ρ * Real.cos p.θ = 1 ∧
    l.equation point_1_0 ∧
    perpendicular l polarAxis :=
  sorry

end NUMINAMATH_CALUDE_line_through_1_0_perpendicular_to_polar_axis_l1911_191106


namespace NUMINAMATH_CALUDE_circle_inequality_l1911_191105

/-- Given a circle with diameter AC = 1, AB tangent to the circle, and BC intersecting the circle again at D,
    prove that if AB = a and CD = b, then 1/(a^2 + 1/2) < b/a < 1/a^2 -/
theorem circle_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  1 / (a^2 + 1/2) < b / a ∧ b / a < 1 / a^2 := by
  sorry

#check circle_inequality

end NUMINAMATH_CALUDE_circle_inequality_l1911_191105


namespace NUMINAMATH_CALUDE_inequality_proof_equality_conditions_l1911_191147

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  x * (1 - 2*x) * (1 - 3*x) + y * (1 - 2*y) * (1 - 3*y) + z * (1 - 2*z) * (1 - 3*z) ≥ 0 :=
sorry

theorem equality_conditions (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  x * (1 - 2*x) * (1 - 3*x) + y * (1 - 2*y) * (1 - 3*y) + z * (1 - 2*z) * (1 - 3*z) = 0 ↔
  ((x = 0 ∧ y = 1/2 ∧ z = 1/2) ∨
   (y = 0 ∧ z = 1/2 ∧ x = 1/2) ∨
   (z = 0 ∧ x = 1/2 ∧ y = 1/2) ∨
   (x = 1/3 ∧ y = 1/3 ∧ z = 1/3)) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_conditions_l1911_191147


namespace NUMINAMATH_CALUDE_chocolates_per_box_l1911_191137

-- Define the problem parameters
def total_boxes : ℕ := 20
def total_chocolates : ℕ := 500

-- Theorem statement
theorem chocolates_per_box :
  total_chocolates / total_boxes = 25 :=
by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_chocolates_per_box_l1911_191137


namespace NUMINAMATH_CALUDE_circle_equation_solution_l1911_191126

theorem circle_equation_solution (a b : ℝ) (h : a^2 + b^2 = 12*a - 4*b + 20) : a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_solution_l1911_191126


namespace NUMINAMATH_CALUDE_system_solution_l1911_191178

theorem system_solution (a : ℝ) (h : a ≠ 0) :
  ∃! (x : ℝ), 3 * x + 2 * x = 15 * a ∧ (1 / a) * x + x = 9 → x = 6 ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1911_191178


namespace NUMINAMATH_CALUDE_selling_price_for_target_profit_impossibility_of_daily_profit_maximum_profit_l1911_191144

-- Define the variables and constants
variable (x : ℝ)  -- price increase
def original_price : ℝ := 40
def cost_price : ℝ := 30
def initial_sales : ℝ := 600
def sales_decrease_rate : ℝ := 10

-- Define the profit function
def profit (x : ℝ) : ℝ :=
  (original_price + x - cost_price) * (initial_sales - sales_decrease_rate * x)

-- Theorem 1: Selling price for 10,000 yuan monthly profit
theorem selling_price_for_target_profit :
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ profit x₁ = 10000 ∧ profit x₂ = 10000 ∧
  (x₁ + original_price = 80 ∨ x₁ + original_price = 50) ∧
  (x₂ + original_price = 80 ∨ x₂ + original_price = 50) :=
sorry

-- Theorem 2: Impossibility of 15,000 yuan daily profit
theorem impossibility_of_daily_profit :
  ¬∃ x, profit x = 15000 * 30 :=
sorry

-- Theorem 3: Price and value for maximum profit
theorem maximum_profit :
  ∃ x_max, ∀ x, profit x ≤ profit x_max ∧
  x_max + original_price = 65 ∧ profit x_max = 12250 :=
sorry

end NUMINAMATH_CALUDE_selling_price_for_target_profit_impossibility_of_daily_profit_maximum_profit_l1911_191144


namespace NUMINAMATH_CALUDE_train_length_l1911_191194

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 126 → time_s = 16 → speed_kmh * (5/18) * time_s = 560 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1911_191194


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l1911_191184

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l1911_191184


namespace NUMINAMATH_CALUDE_exactly_one_true_proposition_l1911_191158

open Real

theorem exactly_one_true_proposition :
  let prop1 := ∀ x : ℝ, x^4 > x^2
  let prop2 := ∀ p q : Prop, (¬(p ∧ q)) → (¬p ∧ ¬q)
  let prop3 := (¬∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0)
  (¬prop1 ∧ ¬prop2 ∧ prop3) :=
by sorry

#check exactly_one_true_proposition

end NUMINAMATH_CALUDE_exactly_one_true_proposition_l1911_191158


namespace NUMINAMATH_CALUDE_base5_digits_of_1234_l1911_191132

/-- The number of digits in the base-5 representation of a positive integer n -/
def base5Digits (n : ℕ+) : ℕ :=
  Nat.log 5 n + 1

/-- Theorem: The number of digits in the base-5 representation of 1234 is 5 -/
theorem base5_digits_of_1234 : base5Digits 1234 = 5 := by
  sorry

end NUMINAMATH_CALUDE_base5_digits_of_1234_l1911_191132


namespace NUMINAMATH_CALUDE_sphere_radius_when_area_equals_volume_l1911_191174

theorem sphere_radius_when_area_equals_volume :
  ∀ R : ℝ,
  R > 0 →
  (4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3) →
  R = 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_when_area_equals_volume_l1911_191174


namespace NUMINAMATH_CALUDE_gregs_ppo_reward_l1911_191107

theorem gregs_ppo_reward (
  ppo_percentage : Real)
  (coinrun_max_reward : Real)
  (procgen_max_reward : Real)
  (h1 : ppo_percentage = 0.9)
  (h2 : coinrun_max_reward = procgen_max_reward / 2)
  (h3 : procgen_max_reward = 240)
  : ppo_percentage * coinrun_max_reward = 108 := by
  sorry

end NUMINAMATH_CALUDE_gregs_ppo_reward_l1911_191107


namespace NUMINAMATH_CALUDE_sector_central_angle_l1911_191181

theorem sector_central_angle (arc_length radius : ℝ) (h1 : arc_length = 2 * Real.pi) (h2 : radius = 2) :
  arc_length / radius = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1911_191181


namespace NUMINAMATH_CALUDE_sqrt_m_minus_n_l1911_191127

theorem sqrt_m_minus_n (m n : ℝ) 
  (h1 : Real.sqrt (m - 3) = 3) 
  (h2 : Real.sqrt (n + 1) = 2) : 
  Real.sqrt (m - n) = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_m_minus_n_l1911_191127


namespace NUMINAMATH_CALUDE_engineer_teams_count_l1911_191162

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of ways to form a team of engineers with given constraints -/
def engineerTeams : ℕ :=
  let totalEngineers := 15
  let phdEngineers := 5
  let msEngineers := 6
  let bsEngineers := 4
  let teamSize := 5
  let minPhd := 2
  let minMs := 2
  let minBs := 1
  (choose phdEngineers minPhd) * (choose msEngineers minMs) * (choose bsEngineers minBs)

theorem engineer_teams_count :
  engineerTeams = 600 := by sorry

end NUMINAMATH_CALUDE_engineer_teams_count_l1911_191162


namespace NUMINAMATH_CALUDE_literature_study_time_l1911_191103

def science_time : ℕ := 60
def math_time : ℕ := 80
def total_time_hours : ℕ := 3

theorem literature_study_time :
  total_time_hours * 60 - science_time - math_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_literature_study_time_l1911_191103


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_count_l1911_191141

theorem arithmetic_sequence_terms_count (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 15 → aₙ = 99 → d = 4 → a₁ + (n - 1) * d = aₙ → n = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_count_l1911_191141


namespace NUMINAMATH_CALUDE_central_cell_value_l1911_191151

/-- A 3x3 table of real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)

/-- The conditions for the table -/
def satisfies_conditions (t : Table) : Prop :=
  t.a * t.b * t.c = 10 ∧
  t.d * t.e * t.f = 10 ∧
  t.g * t.h * t.i = 10 ∧
  t.a * t.d * t.g = 10 ∧
  t.b * t.e * t.h = 10 ∧
  t.c * t.f * t.i = 10 ∧
  t.a * t.b * t.d * t.e = 3 ∧
  t.b * t.c * t.e * t.f = 3 ∧
  t.d * t.e * t.g * t.h = 3 ∧
  t.e * t.f * t.h * t.i = 3

theorem central_cell_value (t : Table) (h : satisfies_conditions t) : t.e = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l1911_191151


namespace NUMINAMATH_CALUDE_bella_score_l1911_191182

theorem bella_score (n : ℕ) (avg_without : ℚ) (avg_with : ℚ) (bella_score : ℚ) : 
  n = 18 →
  avg_without = 75 →
  avg_with = 76 →
  bella_score = (n * avg_with) - ((n - 1) * avg_without) →
  bella_score = 93 := by
  sorry

end NUMINAMATH_CALUDE_bella_score_l1911_191182


namespace NUMINAMATH_CALUDE_one_third_of_recipe_flour_l1911_191175

-- Define the original amount of flour in the recipe
def original_flour : ℚ := 16/3

-- Define the fraction of the recipe we're making
def recipe_fraction : ℚ := 1/3

-- Define the result we want to prove
def result : ℚ := 16/9

-- Theorem statement
theorem one_third_of_recipe_flour :
  recipe_fraction * original_flour = result := by sorry

end NUMINAMATH_CALUDE_one_third_of_recipe_flour_l1911_191175


namespace NUMINAMATH_CALUDE_cubic_function_tangent_and_minimum_l1911_191172

/-- A cubic function with parameters m and n -/
def f (m n : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + n*x + 1

/-- The derivative of f -/
def f' (m n : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + n

theorem cubic_function_tangent_and_minimum (m n : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ f m n x = 1 ∧ f' m n x = 0) →
  (∃ x : ℝ, ∀ y : ℝ, f m n y ≥ f m n x ∧ f m n x = -31) →
  m = 12 ∧ n = 36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_tangent_and_minimum_l1911_191172


namespace NUMINAMATH_CALUDE_billys_coin_piles_l1911_191129

theorem billys_coin_piles (x : ℕ) : 
  (x + 3) * 4 = 20 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_billys_coin_piles_l1911_191129


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l1911_191153

/-- The number of y-intercepts of the parabola x = 3y^2 - 4y + 5 -/
def num_y_intercepts : ℕ := 0

/-- The parabola equation: x = 3y^2 - 4y + 5 -/
def parabola_equation (y : ℝ) : ℝ := 3 * y^2 - 4 * y + 5

theorem parabola_y_intercepts :
  (∀ y : ℝ, parabola_equation y ≠ 0) ∧ num_y_intercepts = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l1911_191153


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1911_191143

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, ∃ n₀ : ℤ, n₀ ≤ x^2) ↔ (∃ x₀ : ℝ, ∀ n : ℤ, n > x₀^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1911_191143


namespace NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l1911_191168

/-- Represents a box with a square base -/
structure Box where
  side : ℝ
  height : ℝ

/-- Represents the wrapping paper -/
structure WrappingPaper where
  side : ℝ

/-- Calculates the area of the wrapping paper needed to wrap the box -/
def wrappingPaperArea (b : Box) (w : WrappingPaper) : ℝ :=
  w.side * w.side

/-- Theorem stating the area of wrapping paper needed -/
theorem wrapping_paper_area_theorem (s : ℝ) (h : s > 0) :
  let b : Box := { side := 2 * s, height := 3 * s }
  let w : WrappingPaper := { side := 4 * s }
  wrappingPaperArea b w = 24 * s^2 := by
  sorry

#check wrapping_paper_area_theorem

end NUMINAMATH_CALUDE_wrapping_paper_area_theorem_l1911_191168


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_leq_neg_two_l1911_191165

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a^2 - 1

-- State the theorem
theorem empty_solution_set_implies_a_leq_neg_two (a : ℝ) :
  (∀ x : ℝ, f a (f a x) ≥ 0) → a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_leq_neg_two_l1911_191165


namespace NUMINAMATH_CALUDE_small_triangle_area_ratio_l1911_191160

/-- Represents a right triangle divided into a square and two smaller right triangles -/
structure DividedRightTriangle where
  /-- Area of the square -/
  square_area : ℝ
  /-- Area of the first small right triangle -/
  small_triangle1_area : ℝ
  /-- Area of the second small right triangle -/
  small_triangle2_area : ℝ
  /-- The first small triangle's area is n times the square's area -/
  small_triangle1_prop : small_triangle1_area = square_area * n
  /-- The square and two small triangles form a right triangle -/
  forms_right_triangle : square_area + small_triangle1_area + small_triangle2_area > 0

/-- 
If one small right triangle has an area n times the square's area, 
then the other small right triangle has an area 1/(4n) times the square's area 
-/
theorem small_triangle_area_ratio 
  (t : DividedRightTriangle) (n : ℝ) (hn : n > 0) :
  t.small_triangle2_area / t.square_area = 1 / (4 * n) := by
  sorry

end NUMINAMATH_CALUDE_small_triangle_area_ratio_l1911_191160


namespace NUMINAMATH_CALUDE_restaurant_menu_fraction_l1911_191108

theorem restaurant_menu_fraction (total_dishes : ℕ) 
  (vegan_dishes : ℕ) 
  (vegan_with_gluten : ℕ) 
  (low_sugar_gluten_free_vegan : ℕ) :
  vegan_dishes = total_dishes / 4 →
  vegan_dishes = 6 →
  vegan_with_gluten = 4 →
  low_sugar_gluten_free_vegan = 1 →
  low_sugar_gluten_free_vegan = total_dishes / 24 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_menu_fraction_l1911_191108


namespace NUMINAMATH_CALUDE_sequence_problem_l1911_191140

/-- Given a sequence {aₙ}, prove that a₁₉ = 1/16 under specific conditions -/
theorem sequence_problem (a : ℕ → ℚ) : 
  (a 4 = 1) → 
  (a 6 = 1/3) → 
  (∃ d : ℚ, ∀ n : ℕ, 1/(a (n+1)) - 1/(a n) = d) → 
  (a 19 = 1/16) := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1911_191140


namespace NUMINAMATH_CALUDE_plant_order_after_365_days_l1911_191166

-- Define the plants
inductive Plant
| Cactus
| Dieffenbachia
| Orchid

-- Define the order of plants
def PlantOrder := List Plant

-- Define the initial order
def initialOrder : PlantOrder := [Plant.Cactus, Plant.Dieffenbachia, Plant.Orchid]

-- Define Luna's swap operation (left and center)
def lunaSwap (order : PlantOrder) : PlantOrder :=
  match order with
  | [a, b, c] => [b, a, c]
  | _ => order

-- Define Sam's swap operation (right and center)
def samSwap (order : PlantOrder) : PlantOrder :=
  match order with
  | [a, b, c] => [a, c, b]
  | _ => order

-- Define a single day's operation (Luna's swap followed by Sam's swap)
def dailyOperation (order : PlantOrder) : PlantOrder :=
  samSwap (lunaSwap order)

-- Define the operation for multiple days
def multiDayOperation (order : PlantOrder) (days : Nat) : PlantOrder :=
  match days with
  | 0 => order
  | n + 1 => multiDayOperation (dailyOperation order) n

-- Theorem to prove
theorem plant_order_after_365_days :
  multiDayOperation initialOrder 365 = [Plant.Orchid, Plant.Cactus, Plant.Dieffenbachia] :=
sorry

end NUMINAMATH_CALUDE_plant_order_after_365_days_l1911_191166


namespace NUMINAMATH_CALUDE_crayons_per_row_l1911_191113

theorem crayons_per_row (rows : ℕ) (pencils_per_row : ℕ) (total_items : ℕ) 
  (h1 : rows = 11)
  (h2 : pencils_per_row = 31)
  (h3 : total_items = 638) :
  (total_items - rows * pencils_per_row) / rows = 27 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_row_l1911_191113


namespace NUMINAMATH_CALUDE_least_m_for_x_bound_l1911_191111

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 6 * x n + 5) / (x n + 7)

theorem least_m_for_x_bound : 
  ∃ m : ℕ, m = 89 ∧ x m ≤ 5 + 1 / (2^10) ∧ ∀ k < m, x k > 5 + 1 / (2^10) :=
sorry

end NUMINAMATH_CALUDE_least_m_for_x_bound_l1911_191111


namespace NUMINAMATH_CALUDE_segment_equality_l1911_191176

/-- Given points A, B, C, D, E, F on a line with certain distance relationships,
    prove that CD = AB = EF. -/
theorem segment_equality 
  (A B C D E F : ℝ) -- Points on a line represented as real numbers
  (h1 : A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F) -- Points are ordered on the line
  (h2 : C - A = E - C) -- AC = CE
  (h3 : D - B = F - D) -- BD = DF
  (h4 : D - A = F - C) -- AD = CF
  : D - C = B - A ∧ D - C = F - E := by
  sorry

end NUMINAMATH_CALUDE_segment_equality_l1911_191176


namespace NUMINAMATH_CALUDE_B_power_66_l1911_191112

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; -1, 0, 0; 0, 0, 1]

theorem B_power_66 : B ^ 66 = !![(-1 : ℝ), 0, 0; 0, -1, 0; 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_B_power_66_l1911_191112


namespace NUMINAMATH_CALUDE_wills_game_cost_l1911_191189

/-- Proves that the cost of Will's new game is $47 --/
theorem wills_game_cost (initial_money : ℕ) (num_toys : ℕ) (toy_price : ℕ) (game_cost : ℕ) : 
  initial_money = 83 →
  num_toys = 9 →
  toy_price = 4 →
  initial_money = game_cost + (num_toys * toy_price) →
  game_cost = 47 := by
sorry

end NUMINAMATH_CALUDE_wills_game_cost_l1911_191189


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l1911_191169

theorem rectangular_solid_volume (x y z : ℝ) 
  (h1 : x * y = 15)  -- Area of side face
  (h2 : y * z = 10)  -- Area of front face
  (h3 : x * z = 6)   -- Area of bottom face
  : x * y * z = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l1911_191169


namespace NUMINAMATH_CALUDE_area_of_trapezoid_psrt_l1911_191118

/-- Represents a triangle in the diagram -/
structure Triangle where
  area : ℝ

/-- Represents the trapezoid PSRT -/
structure Trapezoid where
  area : ℝ

/-- Represents the diagram configuration -/
structure DiagramConfig where
  pqr : Triangle
  smallestTriangles : Finset Triangle
  psrt : Trapezoid

/-- The main theorem statement -/
theorem area_of_trapezoid_psrt (config : DiagramConfig) : config.psrt.area = 53.5 :=
  by
  have h1 : config.pqr.area = 72 := by sorry
  have h2 : config.smallestTriangles.card = 9 := by sorry
  have h3 : ∀ t ∈ config.smallestTriangles, t.area = 2 := by sorry
  have h4 : ∀ t : Triangle, t ∈ config.smallestTriangles → t.area ≤ config.pqr.area := by sorry
  sorry

/-- Auxiliary definition for isosceles triangle -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Auxiliary definition for triangle similarity -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Additional properties of the configuration -/
axiom pqr_is_isosceles (config : DiagramConfig) : isIsosceles config.pqr
axiom all_triangles_similar (config : DiagramConfig) (t : Triangle) : 
  t ∈ config.smallestTriangles → areSimilar t config.pqr

end NUMINAMATH_CALUDE_area_of_trapezoid_psrt_l1911_191118


namespace NUMINAMATH_CALUDE_number_division_problem_l1911_191150

theorem number_division_problem (N : ℝ) (x : ℝ) : 
  ((N - 34) / 10 = 2) → ((N - 5) / x = 7) → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1911_191150


namespace NUMINAMATH_CALUDE_max_books_buyable_l1911_191114

def total_money : ℚ := 24.41
def book_price : ℚ := 2.75

theorem max_books_buyable : 
  ∀ n : ℕ, n * book_price ≤ total_money ∧ 
  (n + 1) * book_price > total_money → n = 8 := by
sorry

end NUMINAMATH_CALUDE_max_books_buyable_l1911_191114


namespace NUMINAMATH_CALUDE_x_y_z_order_l1911_191146

-- Define the constants
noncomputable def x : ℝ := Real.exp (3⁻¹ * Real.log 3)
noncomputable def y : ℝ := Real.exp (6⁻¹ * Real.log 7)
noncomputable def z : ℝ := 7 ^ (1/7 : ℝ)

-- State the theorem
theorem x_y_z_order : z < y ∧ y < x := by
  sorry

end NUMINAMATH_CALUDE_x_y_z_order_l1911_191146


namespace NUMINAMATH_CALUDE_binomial_15_choose_3_l1911_191161

theorem binomial_15_choose_3 : Nat.choose 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_choose_3_l1911_191161


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1911_191156

/-- Represents the shape created by arranging 8 unit cubes -/
structure CubeShape where
  center_cube : Unit
  surrounding_cubes : Fin 6 → Unit
  top_cube : Unit

/-- Calculates the volume of the CubeShape -/
def volume (shape : CubeShape) : ℕ := 8

/-- Calculates the surface area of the CubeShape -/
def surface_area (shape : CubeShape) : ℕ := 28

/-- Theorem stating that the ratio of volume to surface area is 2/7 -/
theorem volume_to_surface_area_ratio (shape : CubeShape) :
  (volume shape : ℚ) / (surface_area shape : ℚ) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_l1911_191156


namespace NUMINAMATH_CALUDE_davids_math_marks_l1911_191155

/-- Calculates the marks in an unknown subject given the marks in other subjects and the average --/
def calculate_unknown_marks (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + physics + chemistry + biology)

theorem davids_math_marks :
  let english := 81
  let physics := 82
  let chemistry := 67
  let biology := 85
  let average := 76
  calculate_unknown_marks english physics chemistry biology average = 65 := by
  sorry

#eval calculate_unknown_marks 81 82 67 85 76

end NUMINAMATH_CALUDE_davids_math_marks_l1911_191155


namespace NUMINAMATH_CALUDE_spongebob_earnings_l1911_191136

/-- Represents the earnings from selling burgers -/
def burger_earnings (num_burgers : ℕ) (price_per_burger : ℚ) : ℚ :=
  num_burgers * price_per_burger

/-- Represents the earnings from selling large fries -/
def fries_earnings (num_fries : ℕ) (price_per_fries : ℚ) : ℚ :=
  num_fries * price_per_fries

/-- Represents the total earnings for the day -/
def total_earnings (burger_earn : ℚ) (fries_earn : ℚ) : ℚ :=
  burger_earn + fries_earn

theorem spongebob_earnings :
  let num_burgers : ℕ := 30
  let price_per_burger : ℚ := 2
  let num_fries : ℕ := 12
  let price_per_fries : ℚ := 3/2
  let burger_earn := burger_earnings num_burgers price_per_burger
  let fries_earn := fries_earnings num_fries price_per_fries
  total_earnings burger_earn fries_earn = 78 := by
sorry

end NUMINAMATH_CALUDE_spongebob_earnings_l1911_191136


namespace NUMINAMATH_CALUDE_complex_to_polar_l1911_191128

open Complex

theorem complex_to_polar (θ : ℝ) : 
  (1 + Real.sin θ + I * Real.cos θ) / (1 + Real.sin θ - I * Real.cos θ) = 
  Complex.exp (I * (π / 2 - θ)) :=
sorry

end NUMINAMATH_CALUDE_complex_to_polar_l1911_191128


namespace NUMINAMATH_CALUDE_max_set_size_with_prime_triple_sums_l1911_191177

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if the sum of any three elements in a list is prime -/
def allTripleSumsPrime (l : List ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ l → b ∈ l → c ∈ l → a ≠ b → b ≠ c → a ≠ c → isPrime (a + b + c)

/-- The main theorem -/
theorem max_set_size_with_prime_triple_sums :
  ∀ (l : List ℕ), (∀ x ∈ l, x > 0) → l.Nodup → allTripleSumsPrime l → l.length ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_set_size_with_prime_triple_sums_l1911_191177


namespace NUMINAMATH_CALUDE_journey_time_proof_l1911_191117

theorem journey_time_proof (highway_distance : ℝ) (mountain_distance : ℝ) 
  (speed_ratio : ℝ) (mountain_time : ℝ) :
  highway_distance = 60 →
  mountain_distance = 20 →
  speed_ratio = 4 →
  mountain_time = 40 →
  highway_distance / (speed_ratio * (mountain_distance / mountain_time)) + mountain_time = 70 :=
by sorry

end NUMINAMATH_CALUDE_journey_time_proof_l1911_191117


namespace NUMINAMATH_CALUDE_abc_product_l1911_191170

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 160) (h2 : b * (c + a) = 168) (h3 : c * (a + b) = 180) :
  a * b * c = 772 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l1911_191170


namespace NUMINAMATH_CALUDE_winter_clothing_boxes_l1911_191197

/-- Given that each box contains 10 pieces of clothing and the total number of pieces is 60,
    prove that the number of boxes is 6. -/
theorem winter_clothing_boxes (pieces_per_box : ℕ) (total_pieces : ℕ) (num_boxes : ℕ) :
  pieces_per_box = 10 →
  total_pieces = 60 →
  num_boxes * pieces_per_box = total_pieces →
  num_boxes = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_winter_clothing_boxes_l1911_191197


namespace NUMINAMATH_CALUDE_polynomial_coefficient_bound_l1911_191130

theorem polynomial_coefficient_bound (a b c d : ℝ) : 
  (∀ x : ℝ, |x| < 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) → 
  |a| + |b| + |c| + |d| ≤ 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_bound_l1911_191130


namespace NUMINAMATH_CALUDE_remainder_problem_l1911_191154

theorem remainder_problem (N : ℤ) (h : N % 221 = 43) : N % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1911_191154


namespace NUMINAMATH_CALUDE_first_method_is_simple_random_second_method_is_systematic_l1911_191192

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a student in the survey --/
structure Student where
  id : Nat
  deriving Repr

/-- Represents the survey setup --/
structure Survey where
  totalStudents : Nat
  selectedStudents : Nat
  selectionCriteria : Student → Bool

/-- Determines the sampling method based on the survey setup --/
def determineSamplingMethod (s : Survey) : SamplingMethod :=
  sorry

/-- The first survey method --/
def firstSurvey : Survey :=
  { totalStudents := 200
  , selectedStudents := 20
  , selectionCriteria := λ _ => true }

/-- The second survey method --/
def secondSurvey : Survey :=
  { totalStudents := 200
  , selectedStudents := 20
  , selectionCriteria := λ student => student.id % 10 = 2 }

/-- Theorem stating that the first method is simple random sampling --/
theorem first_method_is_simple_random :
  determineSamplingMethod firstSurvey = SamplingMethod.SimpleRandom :=
  sorry

/-- Theorem stating that the second method is systematic sampling --/
theorem second_method_is_systematic :
  determineSamplingMethod secondSurvey = SamplingMethod.Systematic :=
  sorry

end NUMINAMATH_CALUDE_first_method_is_simple_random_second_method_is_systematic_l1911_191192


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1911_191152

theorem perfect_square_condition (W L : ℤ) : 
  (1000 < W) → (W < 2000) → (L > 1) → (W = 2 * L^3) → 
  (∃ m : ℤ, W = m^2) → (L = 8) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1911_191152


namespace NUMINAMATH_CALUDE_triangle_problem_l1911_191191

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a / b = 7 / 5 →
  b / c = 5 / 3 →
  (1 / 2) * b * c * Real.sin A = 45 * Real.sqrt 3 →
  (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) →
  (2 * R * Real.sin A = a) →
  Real.cos A = -1 / 2 ∧ R = 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1911_191191


namespace NUMINAMATH_CALUDE_inequality_proof_l1911_191179

theorem inequality_proof (a b : ℝ) (h : (6*a + 9*b)/(a + b) < (4*a - b)/(a - b)) :
  abs b < abs a ∧ abs a < 2 * abs b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1911_191179


namespace NUMINAMATH_CALUDE_multiplication_as_difference_of_squares_l1911_191163

theorem multiplication_as_difference_of_squares :
  ∀ a b : ℚ,
  (19 + 2/3) * (20 + 1/3) = (a - b) * (a + b) →
  a = 20 ∧ b = 1/3 := by
sorry

end NUMINAMATH_CALUDE_multiplication_as_difference_of_squares_l1911_191163


namespace NUMINAMATH_CALUDE_smallest_side_difference_l1911_191104

def is_valid_triangle (pq qr pr : ℕ) : Prop :=
  pq + qr > pr ∧ pq + pr > qr ∧ qr + pr > pq

theorem smallest_side_difference (pq qr pr : ℕ) :
  pq + qr + pr = 3030 →
  pq < qr →
  qr ≤ pr →
  is_valid_triangle pq qr pr →
  (∀ pq' qr' pr' : ℕ, 
    pq' + qr' + pr' = 3030 →
    pq' < qr' →
    qr' ≤ pr' →
    is_valid_triangle pq' qr' pr' →
    qr - pq ≤ qr' - pq') →
  qr - pq = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_side_difference_l1911_191104


namespace NUMINAMATH_CALUDE_mobius_rest_stop_time_l1911_191119

/-- Proves that the rest stop time for each half of the trip is 1 hour given the conditions of Mobius's journey --/
theorem mobius_rest_stop_time 
  (distance : ℝ) 
  (speed_with_load : ℝ) 
  (speed_without_load : ℝ) 
  (total_trip_time : ℝ) 
  (h1 : distance = 143) 
  (h2 : speed_with_load = 11) 
  (h3 : speed_without_load = 13) 
  (h4 : total_trip_time = 26) : 
  (total_trip_time - (distance / speed_with_load + distance / speed_without_load)) / 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_mobius_rest_stop_time_l1911_191119


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1911_191121

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  2 * x^2 * y - 5 * x * y - 4 * x * y^2 + x * y + 4 * x^2 * y - 7 * x * y^2 =
  6 * x^2 * y - 4 * x * y - 11 * x * y^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) :
  (5 * a^2 + 2 * a - 1) - 4 * (2 * a^2 - 3 * a) =
  -3 * a^2 + 14 * a - 1 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1911_191121


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_equation_l1911_191196

theorem unique_integer_satisfying_equation : 
  ∃! (n : ℕ), n > 0 ∧ (n + 1500) / 90 = ⌊Real.sqrt n⌋ ∧ n = 4530 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_equation_l1911_191196


namespace NUMINAMATH_CALUDE_pie_division_l1911_191138

theorem pie_division (total_pie : ℚ) (num_people : ℕ) (individual_share : ℚ) : 
  total_pie = 5/8 →
  num_people = 4 →
  individual_share = total_pie / num_people →
  individual_share = 5/32 := by
sorry

end NUMINAMATH_CALUDE_pie_division_l1911_191138


namespace NUMINAMATH_CALUDE_exists_committees_with_common_members_l1911_191124

/-- Represents a committee system with members and committees. -/
structure CommitteeSystem where
  members : Finset ℕ
  committees : Finset (Finset ℕ)
  member_count : members.card = 1600
  committee_count : committees.card = 16000
  committee_size : ∀ c ∈ committees, c.card = 80

/-- Theorem stating that in a committee system satisfying the given conditions,
    there exist at least two committees sharing at least 4 members. -/
theorem exists_committees_with_common_members (cs : CommitteeSystem) :
  ∃ (c1 c2 : Finset ℕ), c1 ∈ cs.committees ∧ c2 ∈ cs.committees ∧ c1 ≠ c2 ∧
  (c1 ∩ c2).card ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_committees_with_common_members_l1911_191124


namespace NUMINAMATH_CALUDE_joe_not_eating_pizza_probability_l1911_191171

theorem joe_not_eating_pizza_probability 
  (p_eat : ℚ) 
  (h_eat : p_eat = 5/8) : 
  1 - p_eat = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_joe_not_eating_pizza_probability_l1911_191171


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1911_191164

theorem quadratic_equation_solution :
  ∃ (a b : ℝ),
    (∀ x : ℝ, x^2 - 4*x + 9 = 25 ↔ (x = a ∨ x = b)) ∧
    a ≥ b ∧
    3*a + 2*b = 10 + 2*Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1911_191164


namespace NUMINAMATH_CALUDE_class_size_l1911_191135

theorem class_size (total : ℝ) 
  (h1 : 0.25 * total = total - (0.75 * total))
  (h2 : 0.1875 * total = 0.25 * (0.75 * total))
  (h3 : 18 = 0.75 * total - 0.1875 * total) : 
  total = 32 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1911_191135


namespace NUMINAMATH_CALUDE_doughnut_cost_theorem_l1911_191159

def total_cost (chocolate_count : ℕ) (glazed_count : ℕ) (maple_count : ℕ) (strawberry_count : ℕ)
               (chocolate_price : ℚ) (glazed_price : ℚ) (maple_price : ℚ) (strawberry_price : ℚ)
               (chocolate_discount : ℚ) (maple_discount : ℚ) (free_glazed : ℕ) : ℚ :=
  let chocolate_cost := (chocolate_count : ℚ) * chocolate_price * (1 - chocolate_discount)
  let glazed_cost := (glazed_count : ℚ) * glazed_price
  let maple_cost := (maple_count : ℚ) * maple_price * (1 - maple_discount)
  let strawberry_cost := (strawberry_count : ℚ) * strawberry_price
  let free_glazed_savings := (free_glazed : ℚ) * glazed_price
  chocolate_cost + glazed_cost + maple_cost + strawberry_cost - free_glazed_savings

theorem doughnut_cost_theorem :
  total_cost 10 8 5 2 2 1 (3/2) (5/2) (15/100) (1/10) 1 = 143/4 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_cost_theorem_l1911_191159


namespace NUMINAMATH_CALUDE_eat_cereal_together_l1911_191180

/-- The time needed for two people to eat a certain amount of cereal together -/
def time_to_eat_together (fat_rate : ℚ) (thin_rate : ℚ) (total_amount : ℚ) : ℚ :=
  total_amount / (fat_rate + thin_rate)

/-- Theorem stating the time needed for Mr. Fat and Mr. Thin to eat 5 pounds of cereal together -/
theorem eat_cereal_together :
  let fat_rate : ℚ := 1 / 12
  let thin_rate : ℚ := 1 / 40
  let total_amount : ℚ := 5
  time_to_eat_together fat_rate thin_rate total_amount = 600 / 13 := by sorry

end NUMINAMATH_CALUDE_eat_cereal_together_l1911_191180


namespace NUMINAMATH_CALUDE_unique_natural_pair_l1911_191109

theorem unique_natural_pair : ∃! (a b : ℕ), 
  a ≠ b ∧ 
  (∃ (k : ℕ), ∃ (p : ℕ), Prime p ∧ b^2 + a = p^k) ∧
  (∃ (m : ℕ), (a^2 + b) * m = b^2 + a) ∧
  a = 2 ∧ 
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_natural_pair_l1911_191109


namespace NUMINAMATH_CALUDE_simplify_expression_l1911_191139

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : a^3 + b^3 = a + b) (h2 : a^2 + b^2 = 3*a + b) :
  a/b + b/a + 1/(a*b) = (9*a + 3*b + 3)/(3*a + b - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1911_191139


namespace NUMINAMATH_CALUDE_min_z_in_triangle_ABC_l1911_191122

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (1, 0)

-- Define the function z
def z (p : ℝ × ℝ) : ℝ := p.1 - p.2

-- Define the set of points inside or on the boundary of triangle ABC
def triangle_ABC : Set (ℝ × ℝ) :=
  {p | ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
    p = (a * A.1 + b * B.1 + c * C.1, a * A.2 + b * B.2 + c * C.2)}

-- Theorem statement
theorem min_z_in_triangle_ABC :
  ∃ (p : ℝ × ℝ), p ∈ triangle_ABC ∧ ∀ (q : ℝ × ℝ), q ∈ triangle_ABC → z p ≤ z q ∧ z p = -3 :=
sorry

end NUMINAMATH_CALUDE_min_z_in_triangle_ABC_l1911_191122


namespace NUMINAMATH_CALUDE_parabola_intersects_x_axis_twice_l1911_191101

theorem parabola_intersects_x_axis_twice (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ m * x₁^2 + (m - 3) * x₁ - 1 = 0 ∧ m * x₂^2 + (m - 3) * x₂ - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersects_x_axis_twice_l1911_191101


namespace NUMINAMATH_CALUDE_sixth_term_value_l1911_191145

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sixth_term_value (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : arithmetic_sequence a)
  (h3 : ∀ n : ℕ, a (n + 1) - a n = 2) :
  a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l1911_191145


namespace NUMINAMATH_CALUDE_mrs_heine_biscuits_l1911_191183

/-- Given a number of dogs and biscuits per dog, calculates the total number of biscuits needed -/
def total_biscuits (num_dogs : ℕ) (biscuits_per_dog : ℕ) : ℕ :=
  num_dogs * biscuits_per_dog

/-- Theorem: Mrs. Heine needs to buy 6 biscuits for her 2 dogs, given 3 biscuits per dog -/
theorem mrs_heine_biscuits : total_biscuits 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mrs_heine_biscuits_l1911_191183


namespace NUMINAMATH_CALUDE_number_line_is_line_l1911_191102

/-- A number line represents the set of real numbers. -/
def NumberLine : Type := ℝ

/-- A line is an infinite one-dimensional figure extending in both directions. -/
def Line : Type := ℝ

/-- A number line is equivalent to a line. -/
theorem number_line_is_line : NumberLine ≃ Line := by sorry

end NUMINAMATH_CALUDE_number_line_is_line_l1911_191102


namespace NUMINAMATH_CALUDE_bag_original_price_l1911_191185

/-- Given a bag sold for $120 after a 20% discount, prove its original price was $150 -/
theorem bag_original_price (discounted_price : ℝ) (discount_rate : ℝ) : 
  discounted_price = 120 → 
  discount_rate = 0.2 → 
  discounted_price = (1 - discount_rate) * 150 := by
sorry

end NUMINAMATH_CALUDE_bag_original_price_l1911_191185


namespace NUMINAMATH_CALUDE_janes_minnows_l1911_191190

theorem janes_minnows (prize_minnows : ℕ) (total_players : ℕ) (win_percentage : ℚ) (leftover_minnows : ℕ) 
  (h1 : prize_minnows = 3)
  (h2 : total_players = 800)
  (h3 : win_percentage = 15 / 100)
  (h4 : leftover_minnows = 240) :
  prize_minnows * (win_percentage * total_players).floor + leftover_minnows = 600 := by
  sorry

end NUMINAMATH_CALUDE_janes_minnows_l1911_191190


namespace NUMINAMATH_CALUDE_tan_240_degrees_l1911_191142

theorem tan_240_degrees : Real.tan (240 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_240_degrees_l1911_191142


namespace NUMINAMATH_CALUDE_remaining_water_is_one_cup_l1911_191157

/-- Represents Harry's hike and water consumption --/
structure HikeData where
  total_distance : ℝ
  initial_water : ℝ
  duration : ℝ
  leak_rate : ℝ
  last_mile_consumption : ℝ
  first_miles_rate : ℝ

/-- Calculates the remaining water after the hike --/
def remaining_water (data : HikeData) : ℝ :=
  data.initial_water
  - (data.first_miles_rate * (data.total_distance - 1))
  - data.last_mile_consumption
  - (data.leak_rate * data.duration)

/-- Theorem stating that the remaining water is 1 cup --/
theorem remaining_water_is_one_cup (data : HikeData)
  (h1 : data.total_distance = 7)
  (h2 : data.initial_water = 9)
  (h3 : data.duration = 2)
  (h4 : data.leak_rate = 1)
  (h5 : data.last_mile_consumption = 2)
  (h6 : data.first_miles_rate = 0.6666666666666666)
  : remaining_water data = 1 := by
  sorry

end NUMINAMATH_CALUDE_remaining_water_is_one_cup_l1911_191157


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_planes_l1911_191199

-- Define the concept of a plane
structure Plane :=
  (p : Set (ℝ × ℝ × ℝ))

-- Define the concept of a line
structure Line :=
  (l : Set (ℝ × ℝ × ℝ))

-- Define parallel relationship between planes
def parallel_planes (p1 p2 : Plane) : Prop := sorry

-- Define parallel relationship between a line and a plane
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define when a line is within a plane
def line_within_plane (l : Line) (p : Plane) : Prop := sorry

-- Theorem statement
theorem line_parallel_to_parallel_planes 
  (p1 p2 : Plane) (l : Line) 
  (h1 : parallel_planes p1 p2) 
  (h2 : parallel_line_plane l p1) :
  parallel_line_plane l p2 ∨ line_within_plane l p2 := 
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_planes_l1911_191199


namespace NUMINAMATH_CALUDE_plane_distance_proof_l1911_191195

-- Define the plane's speed in still air
def plane_speed : ℝ := 262.5

-- Define the time taken with tail wind
def time_with_wind : ℝ := 3

-- Define the time taken against wind
def time_against_wind : ℝ := 4

-- Define the wind speed (to be solved)
def wind_speed : ℝ := 37.5

-- Define the distance (to be proved)
def distance : ℝ := 900

-- Theorem statement
theorem plane_distance_proof :
  distance = (plane_speed + wind_speed) * time_with_wind ∧
  distance = (plane_speed - wind_speed) * time_against_wind :=
by sorry

end NUMINAMATH_CALUDE_plane_distance_proof_l1911_191195


namespace NUMINAMATH_CALUDE_empty_solution_set_non_empty_solution_set_l1911_191188

-- Define the inequality
def inequality (x a : ℝ) : Prop := |x - 4| + |3 - x| < a

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | inequality x a}

-- Theorem for the empty solution set case
theorem empty_solution_set (a : ℝ) :
  solution_set a = ∅ ↔ a ≤ 1 :=
sorry

-- Theorem for the non-empty solution set case
theorem non_empty_solution_set (a : ℝ) :
  solution_set a ≠ ∅ ↔ a > 1 :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_non_empty_solution_set_l1911_191188


namespace NUMINAMATH_CALUDE_determine_h_of_x_l1911_191125

theorem determine_h_of_x (x : ℝ) (h : ℝ → ℝ) : 
  (4 * x^4 + 5 * x^2 - 2 * x + 1 + h x = 6 * x^3 - 4 * x^2 + 7 * x - 5) → 
  (h x = -4 * x^4 + 6 * x^3 - 9 * x^2 + 9 * x - 6) := by
  sorry

end NUMINAMATH_CALUDE_determine_h_of_x_l1911_191125


namespace NUMINAMATH_CALUDE_power_boat_travel_time_l1911_191187

/-- Represents the scenario of a power boat and raft traveling on a river -/
structure RiverTravel where
  boatSpeed : ℝ  -- Speed of the power boat relative to the river
  riverSpeed : ℝ  -- Speed of the river current
  totalTime : ℝ  -- Total time until the boat meets the raft after returning
  travelTime : ℝ  -- Time taken by the boat to travel from A to B

/-- The conditions of the river travel scenario -/
def riverTravelConditions (rt : RiverTravel) : Prop :=
  rt.riverSpeed = rt.boatSpeed / 2 ∧
  rt.totalTime = 12 ∧
  (rt.boatSpeed + rt.riverSpeed) * rt.travelTime + 
    (rt.boatSpeed - rt.riverSpeed) * (rt.totalTime - rt.travelTime) = 
    rt.riverSpeed * rt.totalTime

/-- The theorem stating that under the given conditions, 
    the travel time from A to B is 6 hours -/
theorem power_boat_travel_time 
  (rt : RiverTravel) 
  (h : riverTravelConditions rt) : 
  rt.travelTime = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_boat_travel_time_l1911_191187


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1911_191131

theorem fraction_evaluation : (5 * 7) / 10 = 3.5 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1911_191131


namespace NUMINAMATH_CALUDE_smallest_n_for_g_nine_l1911_191123

/-- Sum of digits in base 5 representation -/
def f (n : ℕ) : ℕ := sorry

/-- Sum of digits in base 9 representation of f(n) -/
def g (n : ℕ) : ℕ := sorry

/-- The smallest positive integer n such that g(n) = 9 -/
theorem smallest_n_for_g_nine : 
  (∀ m : ℕ, m > 0 ∧ m < 344 → g m ≠ 9) ∧ g 344 = 9 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_g_nine_l1911_191123


namespace NUMINAMATH_CALUDE_ages_sum_l1911_191148

theorem ages_sum (a b c : ℕ) : 
  a = 20 + 2 * (b + c) →
  a^2 = 1980 + 3 * (b + c)^2 →
  a + b + c = 68 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l1911_191148


namespace NUMINAMATH_CALUDE_min_soldiers_to_add_l1911_191193

theorem min_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) :
  (∃ (M : ℕ), (N + M) % 7 = 0 ∧ (N + M) % 12 = 0) ∧
  (∀ (K : ℕ), K < 82 → ¬((N + K) % 7 = 0 ∧ (N + K) % 12 = 0)) ∧
  ((N + 82) % 7 = 0 ∧ (N + 82) % 12 = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_soldiers_to_add_l1911_191193


namespace NUMINAMATH_CALUDE_tracksuit_discount_problem_l1911_191186

/-- Given a tracksuit with an original price, if it is discounted by 20% and the discount amount is 30 yuan, then the actual amount spent is 120 yuan. -/
theorem tracksuit_discount_problem (original_price : ℝ) : 
  (original_price - original_price * 0.8 = 30) → 
  (original_price * 0.8 = 120) := by
sorry

end NUMINAMATH_CALUDE_tracksuit_discount_problem_l1911_191186


namespace NUMINAMATH_CALUDE_inequality_solution_l1911_191173

theorem inequality_solution (x y : ℝ) : 
  Real.sqrt 3 * Real.tan x - (Real.sin y) ^ (1/4) - 
  Real.sqrt ((3 / (Real.cos x)^2) + Real.sqrt (Real.sin y) - 6) ≥ Real.sqrt 3 ↔ 
  ∃ (n k : ℤ), x = π/4 + n*π ∧ y = k*π :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1911_191173


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l1911_191134

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  (5 * (a + b)^2 + 4 * (b - c)^2 + 3 * (c - a)^2) / (2 * b^2) ≥ 24 :=
by sorry

theorem min_value_attainable (ε : ℝ) (hε : ε > 0) :
  ∃ (a b c : ℝ), b > c ∧ c > a ∧ b ≠ 0 ∧
  (5 * (a + b)^2 + 4 * (b - c)^2 + 3 * (c - a)^2) / (2 * b^2) < 24 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l1911_191134


namespace NUMINAMATH_CALUDE_group_size_is_nine_l1911_191133

/-- The number of people in the original group -/
def n : ℕ := sorry

/-- The age of the person joining the group -/
def joining_age : ℕ := 34

/-- The original average age of the group -/
def original_average : ℕ := 14

/-- The new average age after the person joins -/
def new_average : ℕ := 16

/-- The minimum age in the group -/
def min_age : ℕ := 10

/-- There are two sets of twins in the group -/
axiom twin_sets : ∃ (a b : ℕ), (2 * a + 2 * b ≤ n)

/-- All individuals in the group are at least 10 years old -/
axiom all_above_min_age : ∀ (age : ℕ), age ≥ min_age

/-- The sum of ages in the original group -/
def original_sum : ℕ := n * original_average

/-- The sum of ages after the new person joins -/
def new_sum : ℕ := original_sum + joining_age

theorem group_size_is_nine :
  n * original_average + joining_age = new_average * (n + 1) →
  n = 9 := by sorry

end NUMINAMATH_CALUDE_group_size_is_nine_l1911_191133


namespace NUMINAMATH_CALUDE_largest_possible_median_l1911_191115

def number_set (x : ℤ) : Finset ℤ := {x, 2*x, 6, 4, 7}

def is_median (m : ℤ) (s : Finset ℤ) : Prop :=
  2 * (s.filter (λ i => i ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ i => i ≥ m)).card ≥ s.card

theorem largest_possible_median :
  ∃ (x : ℤ), is_median 7 (number_set x) ∧
  ∀ (y : ℤ) (m : ℤ), is_median m (number_set y) → m ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_median_l1911_191115


namespace NUMINAMATH_CALUDE_spinner_sector_areas_l1911_191149

/-- Represents a circular spinner with WIN and BONUS sectors -/
structure Spinner where
  radius : ℝ
  win_prob : ℝ
  bonus_prob : ℝ

/-- Calculates the area of a sector given its probability and the total area -/
def sector_area (prob : ℝ) (total_area : ℝ) : ℝ := prob * total_area

/-- Theorem stating the areas of WIN and BONUS sectors for a specific spinner -/
theorem spinner_sector_areas (s : Spinner) 
  (h_radius : s.radius = 15)
  (h_win_prob : s.win_prob = 1/3)
  (h_bonus_prob : s.bonus_prob = 1/4) :
  let total_area := π * s.radius^2
  sector_area s.win_prob total_area = 75 * π ∧ 
  sector_area s.bonus_prob total_area = 56.25 * π := by
  sorry


end NUMINAMATH_CALUDE_spinner_sector_areas_l1911_191149


namespace NUMINAMATH_CALUDE_point_A_coordinates_l1911_191100

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation to the left -/
def translateLeft (p : Point) (dx : ℝ) : Point :=
  ⟨p.x - dx, p.y⟩

/-- Translation upwards -/
def translateUp (p : Point) (dy : ℝ) : Point :=
  ⟨p.x, p.y + dy⟩

theorem point_A_coordinates :
  ∃ (A : Point),
    ∃ (dx dy : ℝ),
      translateLeft A dx = Point.mk 1 2 ∧
      translateUp A dy = Point.mk 3 4 ∧
      A = Point.mk 3 2 := by
  sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l1911_191100


namespace NUMINAMATH_CALUDE_decimal_arithmetic_l1911_191198

theorem decimal_arithmetic : 3.456 - 1.78 + 0.032 = 1.678 := by
  sorry

end NUMINAMATH_CALUDE_decimal_arithmetic_l1911_191198


namespace NUMINAMATH_CALUDE_soccer_enjoyment_fraction_l1911_191120

theorem soccer_enjoyment_fraction (total : ℝ) (h_total : total > 0) :
  let enjoy_soccer := 0.7 * total
  let dont_enjoy_soccer := 0.3 * total
  let say_enjoy := 0.75 * enjoy_soccer
  let enjoy_but_say_dont := 0.25 * enjoy_soccer
  let say_dont_enjoy := 0.85 * dont_enjoy_soccer
  let total_say_dont := say_dont_enjoy + enjoy_but_say_dont
  enjoy_but_say_dont / total_say_dont = 35 / 86 := by
sorry

end NUMINAMATH_CALUDE_soccer_enjoyment_fraction_l1911_191120


namespace NUMINAMATH_CALUDE_hayden_ride_payment_l1911_191110

def hayden_earnings (hourly_wage : ℝ) (hours_worked : ℝ) (gas_gallons : ℝ) (gas_price : ℝ) 
  (num_reviews : ℕ) (review_bonus : ℝ) (num_rides : ℕ) (total_owed : ℝ) : Prop :=
  let base_earnings := hourly_wage * hours_worked + gas_gallons * gas_price + num_reviews * review_bonus
  let ride_earnings := total_owed - base_earnings
  ride_earnings / num_rides = 5

theorem hayden_ride_payment : 
  hayden_earnings 15 8 17 3 2 20 3 226 := by sorry

end NUMINAMATH_CALUDE_hayden_ride_payment_l1911_191110


namespace NUMINAMATH_CALUDE_tom_bike_miles_per_day_l1911_191167

theorem tom_bike_miles_per_day 
  (total_miles : ℕ) 
  (days_in_year : ℕ) 
  (first_period_days : ℕ) 
  (miles_per_day_first_period : ℕ) 
  (h1 : total_miles = 11860)
  (h2 : days_in_year = 365)
  (h3 : first_period_days = 183)
  (h4 : miles_per_day_first_period = 30) :
  (total_miles - miles_per_day_first_period * first_period_days) / (days_in_year - first_period_days) = 35 :=
by sorry

end NUMINAMATH_CALUDE_tom_bike_miles_per_day_l1911_191167


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1911_191116

/-- Given a function f and a triangle ABC, prove the lengths of sides b and c. -/
theorem triangle_side_lengths 
  (f : ℝ → ℝ) 
  (vec_a vec_b : ℝ → ℝ × ℝ)
  (A B C : ℝ) 
  (a b c : ℝ) :
  (∀ x, f x = (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2) →
  (∀ x, vec_a x = (2 * Real.cos x, -Real.sqrt 3 * Real.sin (2 * x))) →
  (∀ x, vec_b x = (Real.cos x, 1)) →
  f A = -1 →
  a = Real.sqrt 7 / 2 →
  ∃ (k : ℝ), 3 * Real.sin C = 2 * Real.sin B →
  b = 3/2 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l1911_191116
