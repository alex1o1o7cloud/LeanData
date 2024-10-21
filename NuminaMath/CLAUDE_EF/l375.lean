import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koala_fiber_theorem_l375_37528

/-- Represents the absorption rate of fiber for koalas -/
def koala_absorption_rate : ℚ := 2/5

/-- Represents the amount of fiber absorbed by the koala in ounces -/
def fiber_absorbed : ℚ := 12

/-- Calculates the total amount of fiber eaten by the koala -/
noncomputable def total_fiber_eaten : ℚ := fiber_absorbed / koala_absorption_rate

theorem koala_fiber_theorem : total_fiber_eaten = 30 := by
  -- Unfold the definitions
  unfold total_fiber_eaten fiber_absorbed koala_absorption_rate
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_koala_fiber_theorem_l375_37528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_range_inverse_proportion_a_range_l375_37599

/-- An inverse proportion function passing through first and third quadrants -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := (k - 4) / x

theorem inverse_proportion_k_range (k : ℝ) :
  (∀ x : ℝ, x > 0 → inverse_proportion k x > 0) ∧
  (∀ x : ℝ, x < 0 → inverse_proportion k x < 0) →
  k > 4 := by
  sorry

theorem inverse_proportion_a_range (k a : ℝ) :
  a > 0 →
  (∃ y₁ y₂ : ℝ, y₁ < y₂ ∧
    inverse_proportion k (a + 5) = y₁ ∧
    inverse_proportion k (2 * a + 1) = y₂) →
  0 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_range_inverse_proportion_a_range_l375_37599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l375_37521

/-- The function f(x) = a^x + ka^(-x) -/
noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := a^x + k * a^(-x)

/-- The theorem stating the properties of the function f -/
theorem function_properties (a k : ℝ) (h_a_pos : a > 0) (h_a_neq_one : a ≠ 1)
  (h_odd : ∀ x, f a k (-x) = -(f a k x))
  (h_f_one : f a k 1 = 3/2)
  (h_increasing : ∀ x y, x < y → f a k x < f a k y) :
  k = -1 ∧ a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l375_37521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_b_range_l375_37568

/-- A function f(x) = -1/2 * x^2 + b * ln(2x+4) is decreasing on (-2,+∞) if and only if b is in (-∞,-1] -/
theorem decreasing_function_b_range (b : ℝ) :
  (∀ x > -2, HasDerivAt (fun x => -1/2 * x^2 + b * Real.log (2*x+4)) 
    (-x + b / (x + 2)) x) →
  (∀ x > -2, -x + b / (x + 2) ≤ 0) ↔
  b ∈ Set.Iic (-1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_b_range_l375_37568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_equal_magnitude_vectors_l375_37519

/-- Given two 2D vectors m and n, prove that if m = (a, b), n is perpendicular to m,
    and |m| = |n|, then n = (b, -a) -/
theorem perpendicular_equal_magnitude_vectors
  (m n : ℝ × ℝ) (a b : ℝ)
  (h1 : m = (a, b))
  (h2 : m.1 * n.1 + m.2 * n.2 = 0)  -- perpendicularity condition
  (h3 : m.1^2 + m.2^2 = n.1^2 + n.2^2) :  -- equal magnitude condition
  n = (b, -a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_equal_magnitude_vectors_l375_37519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_2010_l375_37593

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2000 then 2 * Real.cos (Real.pi * x / 3) else x - 100

theorem f_composition_2010 : f (f 2010) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_2010_l375_37593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curry_to_draymond_ratio_l375_37595

/-- The Golden State Team's point distribution -/
structure GoldenStateTeam where
  draymond : ℕ
  curry : ℕ
  kelly : ℕ
  durant : ℕ
  klay : ℕ

/-- Conditions for the Golden State Team's point distribution -/
def ValidGoldenStateTeam (team : GoldenStateTeam) : Prop :=
  team.draymond = 12 ∧
  team.kelly = 9 ∧
  team.durant = 2 * team.kelly ∧
  team.klay = team.draymond / 2 ∧
  team.draymond + team.curry + team.kelly + team.durant + team.klay = 69

/-- Theorem: The ratio of Curry's points to Draymond's points is 2:1 -/
theorem curry_to_draymond_ratio (team : GoldenStateTeam) 
  (h : ValidGoldenStateTeam team) : 
  team.curry = 2 * team.draymond := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curry_to_draymond_ratio_l375_37595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_360_meters_l375_37587

noncomputable section

-- Define the train's speed in km/hour
def train_speed : ℝ := 36

-- Define the time taken to pass the bridge in seconds
def passing_time : ℝ := 50

-- Define the length of the bridge in meters
def bridge_length : ℝ := 140

-- Convert km/hour to m/s
noncomputable def speed_in_mps : ℝ := train_speed * 1000 / 3600

-- Calculate the total distance covered
noncomputable def total_distance : ℝ := speed_in_mps * passing_time

-- Define the length of the train
noncomputable def train_length : ℝ := total_distance - bridge_length

-- Theorem statement
theorem train_length_is_360_meters : train_length = 360 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_360_meters_l375_37587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purchase_total_cost_l375_37554

/-- Calculate the total cost of a purchase with discounts and taxes -/
theorem purchase_total_cost
  (sandwich_price : ℚ)
  (soda_price : ℚ)
  (sandwich_quantity : ℕ)
  (soda_quantity : ℕ)
  (sandwich_discount_rate : ℚ)
  (sales_tax_rate : ℚ)
  (h1 : sandwich_price = 349 / 100)
  (h2 : soda_price = 87 / 100)
  (h3 : sandwich_quantity = 2)
  (h4 : soda_quantity = 4)
  (h5 : sandwich_discount_rate = 1 / 10)
  (h6 : sales_tax_rate = 5 / 100) :
  (sandwich_price * sandwich_quantity - sandwich_price * sandwich_quantity * sandwich_discount_rate +
   soda_price * soda_quantity) * (1 + sales_tax_rate) = 1025 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_purchase_total_cost_l375_37554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_wash_gallons_l375_37565

/-- Represents a washing machine with different wash types and water usage. -/
structure WashingMachine where
  heavy_wash : ℚ
  regular_wash : ℚ
  light_wash : ℚ

/-- Represents the laundry loads to be washed. -/
structure LaundryLoads where
  heavy_loads : ℕ
  regular_loads : ℕ
  light_loads : ℕ
  bleached_loads : ℕ

/-- Calculates the total water usage for given laundry loads. -/
def total_water_usage (wm : WashingMachine) (loads : LaundryLoads) : ℚ :=
  wm.heavy_wash * loads.heavy_loads +
  wm.regular_wash * loads.regular_loads +
  wm.light_wash * (loads.light_loads + loads.bleached_loads)

/-- Theorem: Given the specific conditions, the regular wash uses 10 gallons of water. -/
theorem regular_wash_gallons (wm : WashingMachine) (loads : LaundryLoads) :
  wm.heavy_wash = 20 ∧
  wm.light_wash = 2 ∧
  loads.heavy_loads = 2 ∧
  loads.regular_loads = 3 ∧
  loads.light_loads = 1 ∧
  loads.bleached_loads = 2 ∧
  total_water_usage wm loads = 76 →
  wm.regular_wash = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_wash_gallons_l375_37565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_cost_prices_l375_37551

theorem cloth_cost_prices (type_a_meters : ℝ) (type_a_price : ℝ) (type_b_meters : ℝ) (type_b_price : ℝ)
  (type_c_meters : ℝ) (type_c_price : ℝ) (type_a_loss_percent : ℝ) (type_b_profit_percent : ℝ)
  (type_c_profit_per_meter : ℝ) :
  type_a_meters = 300 ∧ type_a_price = 9000 ∧
  type_b_meters = 250 ∧ type_b_price = 7000 ∧
  type_c_meters = 400 ∧ type_c_price = 12000 ∧
  type_a_loss_percent = 10 ∧ type_b_profit_percent = 5 ∧ type_c_profit_per_meter = 8 →
  ∃ (cost_a cost_b cost_c : ℝ),
    (cost_a ≥ 33.32 ∧ cost_a ≤ 33.34) ∧
    (cost_b ≥ 26.66 ∧ cost_b ≤ 26.68) ∧
    (cost_c = 22) ∧
    type_a_price = type_a_meters * cost_a * (1 - type_a_loss_percent / 100) ∧
    type_b_price = type_b_meters * cost_b * (1 + type_b_profit_percent / 100) ∧
    type_c_price = type_c_meters * (cost_c + type_c_profit_per_meter) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_cost_prices_l375_37551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l375_37504

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 - 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 2 ∨ (2 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l375_37504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yuri_broke_glass_l375_37513

-- Define the set of suspects
inductive Suspect : Type
| Andrei : Suspect
| Viktor : Suspect
| Sergei : Suspect
| Yuri : Suspect

-- Define a function to represent whether a suspect is telling the truth
def is_telling_truth (s : Suspect) : Prop := sorry

-- Define a function to represent whether a suspect broke the glass
def broke_glass (s : Suspect) : Prop := sorry

-- State the theorem
theorem yuri_broke_glass :
  -- Only one suspect is telling the truth
  (∃! s : Suspect, is_telling_truth s) →
  -- Andrei's statement
  (is_telling_truth Suspect.Andrei ↔ broke_glass Suspect.Viktor) →
  -- Viktor's statement
  (is_telling_truth Suspect.Viktor ↔ broke_glass Suspect.Sergei) →
  -- Sergei's statement
  (is_telling_truth Suspect.Sergei ↔ ¬is_telling_truth Suspect.Viktor) →
  -- Yuri's statement
  (is_telling_truth Suspect.Yuri ↔ ¬broke_glass Suspect.Yuri) →
  -- Conclusion: Yuri broke the glass
  broke_glass Suspect.Yuri :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_yuri_broke_glass_l375_37513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_margin_after_cost_increase_l375_37547

/-- Profit margin calculation -/
noncomputable def profit_margin (cost : ℝ) (profit : ℝ) : ℝ :=
  profit / (cost + profit)

theorem profit_margin_after_cost_increase
  (m : ℝ) -- profit amount per item
  (h1 : m > 0) -- profit is positive
  (h2 : profit_margin (5 * m) m = 0.2) -- initial profit margin is 20%
  : profit_margin (6.25 * m) m = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_margin_after_cost_increase_l375_37547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_value_l375_37505

noncomputable def S : ℕ → ℝ
  | 0 => 2
  | n + 1 => (n + 3 : ℝ) + (1 / 2) * S n

theorem sum_value : S 2000 = 4002 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_value_l375_37505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bac_value_l375_37572

noncomputable section

open Real EuclideanGeometry

variable (A B C O : EuclideanSpace ℝ (Fin 2))

/-- O is the circumcenter of triangle ABC -/
def is_circumcenter (O A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

/-- Vector equation relating AO, AB, and AC -/
def vector_equation (A B C O : EuclideanSpace ℝ (Fin 2)) : Prop :=
  O - A = (B - A) + 2 • (C - A)

/-- Main theorem: Given the conditions, prove that sin(angle BAC) = sqrt(10) / 4 -/
theorem sin_bac_value 
  (h1 : is_circumcenter O A B C) 
  (h2 : vector_equation A B C O) : 
  Real.sin (angle A B C) = Real.sqrt 10 / 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_bac_value_l375_37572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_running_distance_l375_37503

theorem girls_running_distance :
  let boys_laps : ℕ := 27
  let girls_extra_laps : ℕ := 9
  let lap_distance : ℚ := 3/4
  (boys_laps + girls_extra_laps : ℚ) * lap_distance = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_running_distance_l375_37503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complementN_l375_37502

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set N
def N : Set ℝ := {x | Real.rpow 2 x < 2}

-- Define the complement of N in ℝ
def complementN : Set ℝ := {x | x ∉ N}

-- Theorem statement
theorem intersection_M_complementN :
  M ∩ complementN = Set.Icc 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_complementN_l375_37502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_25pi_over_3_l375_37543

/-- The function f as defined in the problem -/
noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α)) / (Real.cos (-Real.pi - α) * Real.tan (Real.pi - α))

/-- Theorem stating that f(-25π/3) = 1/2 -/
theorem f_value_at_negative_25pi_over_3 : f (-25 * Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_25pi_over_3_l375_37543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_three_heads_ten_coins_l375_37577

theorem probability_at_most_three_heads_ten_coins :
  let n : ℕ := 10
  let k : ℕ := 3
  let total_outcomes : ℕ := 2^n
  let favorable_outcomes : ℕ := (Finset.range (k+1)).sum (λ i ↦ Nat.choose n i)
  (favorable_outcomes : ℚ) / total_outcomes = 176 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_three_heads_ten_coins_l375_37577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l375_37501

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (1/2) * t.b * t.c * Real.sin t.A

/-- Theorem: In a triangle ABC where b = 2, c = √3, and angle A = 30°, 
    the area is √3/2 and the length of side BC (a) is 1 -/
theorem triangle_properties (t : Triangle) 
    (h1 : t.b = 2) 
    (h2 : t.c = Real.sqrt 3) 
    (h3 : t.A = π/6) : 
    triangleArea t = Real.sqrt 3 / 2 ∧ t.a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l375_37501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_lines_intersection_l375_37544

open Real

noncomputable def g (x : ℝ) : ℝ := log x

theorem orthogonal_lines_intersection (e : ℝ) (h_e : e > 0) : 
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo (1/e) (1/sqrt e) ∧ 
  (∀ x : ℝ, ∃ y : ℝ, y = x ∧ x * x₀ + x * (log x₀) = 0) ∧
  log x₀ = g x₀ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_lines_intersection_l375_37544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_at_7_l375_37585

-- Define the polynomial Q(x)
def Q (x : ℂ) (g h i j k l : ℝ) : ℂ :=
  (3*x^4 - 33*x^3 + g*x^2 + h*x + i) * (4*x^4 - 88*x^3 + j*x^2 + k*x + l)

-- Define the set of complex roots
def roots : Set ℂ := {1, 2, 3, 4, 6}

-- Theorem statement
theorem Q_value_at_7 (g h i j k l : ℝ) :
  (∀ z : ℂ, Q z g h i j k l = 0 → z ∈ roots) →
  Q 7 g h i j k l = 12960 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_at_7_l375_37585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_rectangle_condition_l375_37590

open Real

theorem tangent_rectangle_condition (a : ℝ) : 
  (∃ b c d : ℝ, b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
    (Real.cos a * Real.cos b = -1) ∧ (Real.cos b * Real.cos c = -1) ∧ 
    (Real.cos c * Real.cos d = -1) ∧ (Real.cos d * Real.cos a = -1)) ↔ 
  (∃ n : ℤ, a = n * π) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_rectangle_condition_l375_37590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_properties_l375_37533

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Angle between two lines -/
noncomputable def angle (l1 l2 : Line) : ℝ := sorry

/-- Check if a line bisects an angle -/
def bisects_angle (l : Line) (a : ℝ) : Prop := sorry

/-- Check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop := sorry

/-- Create a line passing through two points -/
noncomputable def Line.through_points (p1 p2 : Point) : Line := sorry

/-- Create a circle passing through three points -/
noncomputable def Circle.through_points (p1 p2 p3 : Point) : Circle := sorry

/-- Check if a point is on the y-axis -/
def on_y_axis (p : Point) : Prop := p.x = 0

/-- Check if a point is a focus of a parabola -/
def is_focus (p : Point) (para : Parabola) : Prop := sorry

/-- Check if a point is external to a parabola -/
def is_external_point (p : Point) (para : Parabola) : Prop := sorry

/-- Check if a point is a tangent point on a parabola -/
def is_tangent_point (p : Point) (para : Parabola) : Prop := sorry

/-- Check if a point is the circumcenter of a triangle -/
def is_circumcenter (m p1 p2 p3 : Point) : Prop := sorry

/-- Main theorem -/
theorem parabola_tangent_properties
  (para : Parabola)
  (F : Point)
  (P A B C D M : Point)
  (h1 : P.y ≠ 0)
  (h2 : is_focus F para)
  (h3 : is_external_point P para)
  (h4 : is_tangent_point A para)
  (h5 : is_tangent_point B para)
  (h6 : on_y_axis C)
  (h7 : on_y_axis D)
  (h8 : is_circumcenter M A B P) :
  (bisects_angle (Line.through_points P F) (angle (Line.through_points A F) (Line.through_points B F))) ∧
  (is_tangent (Line.through_points F M) (Circle.through_points F C D)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_properties_l375_37533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dancing_preference_theorem_l375_37548

theorem dancing_preference_theorem (total_students : ℝ) 
  (enjoy_dancing_percent : ℝ) (dont_enjoy_dancing_percent : ℝ)
  (enjoy_accurate_percent : ℝ) (enjoy_false_percent : ℝ)
  (dont_enjoy_accurate_percent : ℝ) (dont_enjoy_false_percent : ℝ)
  (h1 : enjoy_dancing_percent = 0.7)
  (h2 : dont_enjoy_dancing_percent = 0.3)
  (h3 : enjoy_dancing_percent + dont_enjoy_dancing_percent = 1)
  (h4 : enjoy_accurate_percent = 0.75)
  (h5 : enjoy_false_percent = 0.25)
  (h6 : enjoy_accurate_percent + enjoy_false_percent = 1)
  (h7 : dont_enjoy_accurate_percent = 0.85)
  (h8 : dont_enjoy_false_percent = 0.15)
  (h9 : dont_enjoy_accurate_percent + dont_enjoy_false_percent = 1) :
  let enjoy_dancing := enjoy_dancing_percent * total_students
  let dont_enjoy_dancing := dont_enjoy_dancing_percent * total_students
  let say_enjoy := enjoy_accurate_percent * enjoy_dancing + dont_enjoy_false_percent * dont_enjoy_dancing
  let say_dont_enjoy := enjoy_false_percent * enjoy_dancing + dont_enjoy_accurate_percent * dont_enjoy_dancing
  let fraction := (enjoy_false_percent * enjoy_dancing) / say_dont_enjoy
  ∃ ε > 0, |fraction - 0.407| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dancing_preference_theorem_l375_37548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_series_theorem_l375_37567

noncomputable def alternating_series_sum (a₁ : ℝ) (d : ℝ) (last_term : ℝ) : ℝ :=
  let n := (last_term - a₁) / d + 1
  let num_pairs := (n - 1) / 2
  let pair_sum := -3
  let sum_of_pairs := num_pairs * pair_sum
  sum_of_pairs + last_term

theorem alternating_series_theorem :
  alternating_series_sum 3 6 69 = 36 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_series_theorem_l375_37567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_inscribed_in_sphere_l375_37575

/-- Given a cube inscribed in a sphere, this theorem relates the sphere's volume to the cube's edge length. -/
theorem cube_inscribed_in_sphere (V : ℝ) (a : ℝ) (h1 : V = 36 * Real.pi) :
  (V = (4 / 3) * Real.pi * ((Real.sqrt 3 / 2 * a) ^ 3)) → a = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_inscribed_in_sphere_l375_37575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_required_is_540_l375_37514

-- Define the floor dimensions in feet
def floor_length : ℚ := 10
def floor_width : ℚ := 15

-- Define the tile dimensions in inches
def tile_length : ℚ := 5
def tile_width : ℚ := 8

-- Convert inches to feet
def inches_to_feet (inches : ℚ) : ℚ := inches / 12

-- Calculate the area of one tile in square feet
def tile_area : ℚ := inches_to_feet tile_length * inches_to_feet tile_width

-- Calculate the floor area in square feet
def floor_area : ℚ := floor_length * floor_width

-- Calculate the number of tiles needed
noncomputable def tiles_needed : ℕ := Int.toNat (Int.ceil (floor_area / tile_area))

-- Theorem to prove
theorem tiles_required_is_540 : tiles_needed = 540 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_required_is_540_l375_37514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_orthocenter_l375_37562

-- Define the basic geometric objects
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given points
variable (A B C : Point)

-- Define the allowed operations
noncomputable def mark_arbitrary_point : Point :=
  ⟨0, 0⟩ -- Placeholder implementation

noncomputable def mark_point_on_line (l : Line) : Point :=
  ⟨0, 0⟩ -- Placeholder implementation

noncomputable def draw_line (P₁ P₂ : Point) : Line :=
  ⟨0, 0, 0⟩ -- Placeholder implementation

noncomputable def mark_intersection (l₁ l₂ : Line) : Point :=
  ⟨0, 0⟩ -- Placeholder implementation

noncomputable def draw_parallel_line (l : Line) (d : ℝ) : Line :=
  ⟨0, 0, 0⟩ -- Placeholder implementation

-- Define the orthocenter
noncomputable def orthocenter (A B C : Point) : Point :=
  ⟨0, 0⟩ -- Placeholder implementation

-- Define collinearity
def collinear (A B C : Point) : Prop :=
  ∃ (t : ℝ), B.x - A.x = t * (C.x - A.x) ∧ B.y - A.y = t * (C.y - A.y)

-- State the theorem
theorem construct_orthocenter (h : ¬ collinear A B C) :
  ∃ (H : Point), H = orthocenter A B C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_orthocenter_l375_37562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l375_37542

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Define for 0 to match Lean's nat type
  | n + 1 => sequence_a n + 3 * (n + 1)  -- Adjust index to match problem definition

theorem a_100_value : sequence_a 99 = 15001.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l375_37542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_higher_probability_l375_37532

/-- The probability that a ball is tossed into bin k -/
noncomputable def prob_bin (k : ℕ+) : ℝ := 2^(-(k : ℝ))

/-- The probability that the red ball is in a higher-numbered bin than the green ball -/
noncomputable def prob_red_higher : ℝ := 1/3

theorem red_ball_higher_probability :
  (∀ k : ℕ+, prob_bin k = 2^(-(k : ℝ))) →
  prob_red_higher = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_ball_higher_probability_l375_37532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_concentration_probability_less_than_half_l375_37552

/-- Represents the number of large flasks -/
def N : ℕ := sorry

/-- Represents the number of small flasks -/
def n : ℕ := sorry

/-- The total number of flasks -/
axiom total_flasks : N + n = 100

/-- There are at least 3 flasks of each size -/
axiom min_flasks : N ≥ 3 ∧ n ≥ 3

/-- Calculates the probability of selecting three flasks resulting in a salt concentration between 45% and 55% -/
def probability_salt_concentration (N : ℕ) : ℚ :=
  (N^2 - 100*N + 4950 : ℚ) / 4950

/-- The main theorem stating that for N ≥ 46, the probability is less than 1/2 -/
theorem salt_concentration_probability_less_than_half (h : N ≥ 46) :
  probability_salt_concentration N < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_concentration_probability_less_than_half_l375_37552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_l375_37580

theorem min_abs_diff (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : a * b - 4 * a + 7 * b = 679) :
  ∃ (a' b' : ℤ), a' > 0 ∧ b' > 0 ∧ a' * b' - 4 * a' + 7 * b' = 679 ∧
    ∀ (x y : ℤ), x > 0 → y > 0 → x * y - 4 * x + 7 * y = 679 → |x - y| ≥ |a' - b'| ∧ |a' - b'| = 37 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_l375_37580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l375_37549

noncomputable section

variable (f : ℝ → ℝ)

theorem f_properties (h1 : ∀ a b : ℝ, f (a + b) = f a + f b - 1)
                     (h2 : ∀ x : ℝ, x > 0 → f x > 1)
                     (h3 : f 4 = 5) :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (f 2 = 3) ∧
  ({m : ℝ | f (3 * m^2 - m - 2) < 3} = {m : ℝ | -1 < m ∧ m < 4/3}) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l375_37549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_isosceles_division_2021gon_l375_37557

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- An isosceles triangle -/
structure IsoscelesTriangle where

/-- A diagonal of a polygon -/
structure Diagonal where

/-- A division of a polygon into triangles using non-intersecting diagonals -/
structure PolygonDivision (n : ℕ) where
  polygon : RegularPolygon n
  triangles : List IsoscelesTriangle
  diagonals : List Diagonal
  nonintersecting : Prop
  covers_polygon : Prop

/-- Theorem stating the impossibility of dividing a regular 2021-gon into isosceles triangles -/
theorem no_isosceles_division_2021gon :
  ¬ ∃ (d : PolygonDivision 2021), True := by
  sorry

#check no_isosceles_division_2021gon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_isosceles_division_2021gon_l375_37557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_80_cents_l375_37582

def Penny : ℕ := 1
def Nickel : ℕ := 5
def Dime : ℕ := 10

def TotalValue (pennies nickels dimes : ℕ) : ℕ :=
  pennies * Penny + nickels * Nickel + dimes * Dime

theorem impossible_80_cents :
  ∀ (pennies nickels dimes : ℕ),
    pennies + nickels + dimes = 6 →
    TotalValue pennies nickels dimes ≠ 80 :=
by
  intro pennies nickels dimes
  intro h
  intro contra
  -- The proof goes here
  sorry

#eval TotalValue 0 0 6  -- Should output 60
#eval TotalValue 4 2 0  -- Should output 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_80_cents_l375_37582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_to_line_not_necessarily_parallel_l375_37515

-- Define a 3D space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V] [CompleteSpace V]

-- Define planes and lines in the space
variable (Plane : Type*) (Line : Type*)

-- Define the parallel relation between planes and lines
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem planes_parallel_to_line_not_necessarily_parallel :
  ¬ ∀ (P Q : Plane) (L : Line), parallel_line_plane L P → parallel_line_plane L Q → parallel P Q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_parallel_to_line_not_necessarily_parallel_l375_37515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₃_to_C₁_l375_37571

/-- The curve C₃ -/
def C₃ (x y : ℝ) : Prop := x^2/16 + y^2 = 1

/-- The line representing C₁ -/
def C₁ (x y : ℝ) : Prop := x - 2*y - 5 = 0

/-- The distance from a point (x, y) to the line C₁ -/
noncomputable def distance_to_C₁ (x y : ℝ) : ℝ :=
  |x - 2*y - 5| / Real.sqrt 5

/-- The maximum distance from any point on C₃ to C₁ is 2 + √5 -/
theorem max_distance_C₃_to_C₁ :
  ∀ x y : ℝ, C₃ x y → distance_to_C₁ x y ≤ 2 + Real.sqrt 5 := by
  sorry

#check max_distance_C₃_to_C₁

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₃_to_C₁_l375_37571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l375_37555

theorem sin_cos_relation :
  (∀ α β : ℝ, Real.sin α + Real.cos β = 0 → (Real.sin α)^2 + (Real.sin β)^2 = 1) ∧
  ¬(∀ α β : ℝ, (Real.sin α)^2 + (Real.sin β)^2 = 1 → Real.sin α + Real.cos β = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_relation_l375_37555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_products_selected_l375_37529

/-- Represents the number of defective products selected -/
def ξ : ℕ := sorry

/-- The total number of products -/
def total_products : ℕ := 8

/-- The number of defective products -/
def defective_products : ℕ := 2

/-- The number of products randomly selected -/
def selected_products : ℕ := 3

/-- The set of possible values for ξ -/
def possible_values : Set ℕ := {0, 1, 2}

theorem defective_products_selected :
  ξ ∈ possible_values := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_products_selected_l375_37529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_age_l375_37546

-- Define the ages of a, b, c, and d as real numbers
variable (a b c d : ℝ)

-- Define the conditions
def condition1 (a b : ℝ) : Prop := a = b + 4
def condition2 (b c : ℝ) : Prop := b = 2 * c
def condition3 (a b c : ℝ) : Prop := a + b + c = 720 / 12
def condition4 (a d : ℝ) : Prop := a = d + 0.75

-- Theorem statement
theorem b_age :
  ∀ a b c d : ℝ,
  condition1 a b →
  condition2 b c →
  condition3 a b c →
  condition4 a d →
  b = 22.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_age_l375_37546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_negative_half_l375_37581

/-- Given that the terminal side of angle α passes through point P(-3, -√3), prove that sin α = -1/2 -/
theorem sin_alpha_negative_half (α : ℝ) (P : ℝ × ℝ) : 
  P.1 = -3 ∧ P.2 = -Real.sqrt 3 → Real.sin α = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_negative_half_l375_37581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l375_37506

-- Define the train's length in meters
noncomputable def train_length : ℝ := 300

-- Define the train's speed in km/h
noncomputable def train_speed_kmh : ℝ := 72

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the time to cross the pole in seconds
noncomputable def time_to_cross : ℝ := 15

-- Theorem statement
theorem train_crossing_time :
  train_length / (train_speed_kmh * kmh_to_ms) = time_to_cross :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l375_37506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_perimeter_equals_300π_shaded_area_perimeter_approx_l375_37569

/-- The perimeter of the shaded area in a regular octagon with circles --/
noncomputable def shaded_area_perimeter (side_length : ℝ) (h_side : side_length = 100) : ℝ :=
  let central_angle : ℝ := 135 * Real.pi / 180
  let arc_length : ℝ := side_length * central_angle
  let total_perimeter : ℝ := 4 * arc_length
  total_perimeter

/-- The perimeter of the shaded area is equal to 300π --/
theorem shaded_area_perimeter_equals_300π (side_length : ℝ) (h_side : side_length = 100) :
  shaded_area_perimeter side_length h_side = 300 * Real.pi :=
by sorry

/-- The perimeter of the shaded area is equal to 942 cm when π is approximated as 3.14 --/
theorem shaded_area_perimeter_approx (side_length : ℝ) (h_side : side_length = 100) 
  (h_π_approx : Real.pi = 3.14) :
  shaded_area_perimeter side_length h_side = 942 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_perimeter_equals_300π_shaded_area_perimeter_approx_l375_37569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_p_employee_increase_l375_37570

/-- The percentage increase in employees for Company P from January to December -/
noncomputable def percentage_increase (january_employees : ℝ) (december_employees : ℝ) : ℝ :=
  (december_employees - january_employees) / january_employees * 100

theorem company_p_employee_increase :
  let january_employees : ℝ := 417.39
  let december_employees : ℝ := 480
  abs (percentage_increase january_employees december_employees - 14.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_p_employee_increase_l375_37570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l375_37520

theorem sin_2alpha_value (α : Real) 
  (h1 : π/2 < α ∧ α < π) 
  (h2 : Real.cos α = -1/2) : 
  Real.sin (2*α) = -Real.sqrt 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l375_37520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinates_l375_37561

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three vertices -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

/-- Theorem: The third vertex of the obtuse triangle -/
theorem third_vertex_coordinates :
  let p1 : Point := ⟨1, 6⟩
  let p2 : Point := ⟨-3, 0⟩
  ∀ x : ℝ,
    x > 0 →
    let p3 : Point := ⟨x, 0⟩
    triangleArea p1 p2 p3 = 14 →
    x = 5/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinates_l375_37561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_advertising_agency_clients_eq_180_l375_37596

/-- The number of clients an advertising agency has, given the following conditions:
  * 115 clients use television
  * 110 clients use radio
  * 130 clients use magazines
  * 85 clients use television and magazines
  * 75 clients use television and radio
  * 95 clients use radio and magazines
  * 80 clients use all three
-/
def advertising_agency_clients : ℕ :=
  let T := 115 -- clients using television
  let R := 110 -- clients using radio
  let M := 130 -- clients using magazines
  let T_and_R := 75 -- clients using television and radio
  let T_and_M := 85 -- clients using television and magazines
  let R_and_M := 95 -- clients using radio and magazines
  let T_and_R_and_M := 80 -- clients using all three
  T + R + M - T_and_R - T_and_M - R_and_M + T_and_R_and_M

#eval advertising_agency_clients

theorem advertising_agency_clients_eq_180 : advertising_agency_clients = 180 := by
  -- Unfold the definition of advertising_agency_clients
  unfold advertising_agency_clients
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_advertising_agency_clients_eq_180_l375_37596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_volume_and_icing_sum_l375_37516

/-- Represents a rectangular cake with given dimensions -/
structure Cake where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of one piece after diagonal cut -/
noncomputable def volume_of_piece (cake : Cake) : ℝ :=
  cake.length * cake.width * cake.height / 2

/-- Calculates the area of icing on one piece after diagonal cut -/
noncomputable def icing_area (cake : Cake) : ℝ :=
  cake.length * cake.width / 2 + 
  cake.length * cake.height / 2 + 
  cake.width * cake.height / 2 + 
  cake.length * cake.height

/-- Theorem stating that for a 4x3x2 cake, the sum of volume and icing area of one piece is 38 -/
theorem cake_volume_and_icing_sum :
  let cake : Cake := { length := 4, width := 3, height := 2 }
  volume_of_piece cake + icing_area cake = 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_volume_and_icing_sum_l375_37516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l375_37530

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2^x + a*x

-- State the theorem
theorem find_a : ∃ a : ℝ, (f a (f a 1) = 4 * a) ∧ (a = 2) := by
  -- We'll use 2 as our witness for a
  use 2
  
  -- Split the goal into two parts
  constructor
  
  -- Part 1: Prove f 2 (f 2 1) = 4 * 2
  · -- Evaluate f 2 1
    have h1 : f 2 1 = 2 := by
      simp [f]
      norm_num
    
    -- Now evaluate f 2 2
    calc f 2 (f 2 1) = f 2 2 := by rw [h1]
                    _ = 2^2 + 2*2 := by simp [f]
                    _ = 8 := by norm_num
                    _ = 4 * 2 := by norm_num

  -- Part 2: Prove a = 2
  · rfl

-- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l375_37530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_real_and_second_quadrant_l375_37527

noncomputable def Z (m : ℝ) : ℂ := Complex.log (m^2 + 2*m - 14 : ℂ) + (m^2 - m - 6)*Complex.I

theorem Z_real_and_second_quadrant (m : ℝ) :
  (∃ x : ℝ, Z m = x) ∧ (m = 3) ∧
  ((Z m).re > 0 ∧ (Z m).im > 0) ∧ (-5 < m ∧ m < -1 - Real.sqrt 15) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Z_real_and_second_quadrant_l375_37527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_correct_l375_37511

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

/-- The midpoint of the chord -/
def P : ℝ × ℝ := (8, 1)

/-- The equation of the line containing the chord -/
def chord_line (x y : ℝ) : Prop := 2*x - y - 15 = 0

/-- Helper function to define a line segment -/
def in_line_segment (A B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B

/-- Theorem stating that the chord_line is correct for the given hyperbola and midpoint -/
theorem chord_equation_correct :
  ∃ (A B : ℝ × ℝ),
    let (x₁, y₁) := A
    let (x₂, y₂) := B
    (hyperbola x₁ y₁) ∧
    (hyperbola x₂ y₂) ∧
    ((x₁ + x₂) / 2, (y₁ + y₂) / 2) = P ∧
    ∀ (x y : ℝ), in_line_segment A B (x, y) → chord_line x y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_correct_l375_37511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_expansion_17_29_l375_37560

/-- The decimal expansion of 17/29 -/
def decimal_expansion : ℕ → ℕ := sorry

/-- The period of the decimal expansion of 17/29 -/
def period : ℕ := 28

/-- The 215th digit after the decimal point in the decimal expansion of 17/29 -/
def digit_215 : ℕ := decimal_expansion 215

theorem decimal_expansion_17_29 : digit_215 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_expansion_17_29_l375_37560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_solution_correctness_l375_37517

-- Define the system of equations
def system (a x y : ℝ) : Prop :=
  (x - a)^2 = 8 * (2 * y - x + a - 2) ∧ (1 - Real.sqrt y) / (1 - Real.sqrt (x / 2)) = 1

-- Define the solution set
noncomputable def solution_set (a : ℝ) : Set (ℝ × ℝ) :=
  if a = 10 then {(18, 9)}
  else if a > 2 ∧ a ≠ 10 then
    {(x, y) | (x = a + Real.sqrt (8 * a - 16) ∨ x = a - Real.sqrt (8 * a - 16)) ∧
              (y = (a + Real.sqrt (8 * a - 16)) / 2 ∨ y = (a - Real.sqrt (8 * a - 16)) / 2)}
  else ∅

-- Theorem statement
theorem system_solution :
  ∀ a : ℝ, (∃ x y : ℝ, system a x y) ↔ (a > 2 ∧ a ≠ 10) ∨ a = 10 := by
  sorry

-- Theorem for solution correctness
theorem solution_correctness :
  ∀ a : ℝ, (a > 2 ∧ a ≠ 10) ∨ a = 10 →
    ∀ x y : ℝ, (x, y) ∈ solution_set a ↔ system a x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_solution_correctness_l375_37517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sequence_is_valid_correct_sequence_is_unique_l375_37556

/-- Represents the steps in the survey process --/
inductive SurveyStep
  | CollectData
  | OrganizeData
  | DrawPieChart
  | AnalyzeStatistics

/-- Represents a sequence of survey steps --/
def SurveySequence := List SurveyStep

/-- The correct sequence of steps for the survey --/
def correctSequence : SurveySequence :=
  [SurveyStep.CollectData, SurveyStep.OrganizeData, SurveyStep.DrawPieChart, SurveyStep.AnalyzeStatistics]

/-- Checks if a given sequence of steps is valid for the survey process --/
def isValidSequence (sequence : SurveySequence) : Prop :=
  sequence.length = 4 ∧
  sequence.Nodup ∧
  sequence.head? = some SurveyStep.CollectData ∧
  sequence.getLast? = some SurveyStep.AnalyzeStatistics

/-- Theorem stating that the correct sequence is valid --/
theorem correct_sequence_is_valid :
  isValidSequence correctSequence :=
by sorry

/-- Theorem stating that the correct sequence is the only valid sequence --/
theorem correct_sequence_is_unique (sequence : SurveySequence) :
  isValidSequence sequence → sequence = correctSequence :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sequence_is_valid_correct_sequence_is_unique_l375_37556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_quadrilateral_area_l375_37592

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Point A is on the ellipse -/
axiom point_A_on_ellipse : ellipse_C 1 (Real.sqrt 2 / 2)

/-- F is the right focus of the ellipse -/
axiom F_is_right_focus : ∃ c, c > 0 ∧ c^2 = 2 - 1 ∧ ellipse_C (1 + c) 0

/-- Definition of perpendicular lines through F -/
def perpendicular_lines (m₁ m₂ : ℝ) : Prop :=
  m₁ * m₂ = -1 ∧ m₁ ≠ 0 ∧ m₂ ≠ 0

/-- Intersection points of a line with the ellipse -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ellipse_C x y ∧ y = m * (x - 1)}

/-- Area of the quadrilateral formed by intersection points -/
noncomputable def quadrilateral_area (m₁ m₂ : ℝ) : ℝ :=
  let A₁B₁ := (intersection_points m₁).ncard
  let A₂B₂ := (intersection_points m₂).ncard
  Real.sqrt (A₁B₁^2 * A₂B₂^2)

/-- The main theorem -/
theorem min_quadrilateral_area :
  ∃ (min_area : ℝ), min_area = 16 / 9 ∧
  ∀ (m₁ m₂ : ℝ), perpendicular_lines m₁ m₂ →
  quadrilateral_area m₁ m₂ ≥ min_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_quadrilateral_area_l375_37592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l375_37559

/-- The range of values for m given the conditions of the problem -/
def m_range : Set ℝ := Set.Ioc (-3 * Real.sqrt 2) (-3) ∪ Set.Ico 3 (3 * Real.sqrt 2)

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

/-- The line equation -/
def is_on_line (x y m : ℝ) : Prop := x + 3 * y = m

/-- The angle AOB is less than or equal to 90 degrees -/
def angle_condition (A B : ℝ × ℝ) : Prop := 
  (A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2)) ≥ 0

/-- The main theorem -/
theorem m_range_theorem (m : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    is_on_ellipse A.1 A.2 ∧ 
    is_on_ellipse B.1 B.2 ∧ 
    is_on_line A.1 A.2 m ∧ 
    is_on_line B.1 B.2 m ∧ 
    angle_condition A B) 
  ↔ m ∈ m_range := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l375_37559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_60_multiple_of_6_l375_37500

-- Define a function to check if a number is a factor of 60
def is_factor_of_60 (n : ℕ) : Bool := 60 % n = 0

-- Define a function to check if a number is a multiple of 6
def is_multiple_of_6 (n : ℕ) : Bool := n % 6 = 0

-- Theorem statement
theorem factors_of_60_multiple_of_6 :
  (Finset.filter (λ n => is_factor_of_60 n ∧ is_multiple_of_6 n) (Finset.range 61)).card = 4 := by
  -- Enumerate the factors of 60 that are multiples of 6
  have h : Finset.filter (λ n => is_factor_of_60 n ∧ is_multiple_of_6 n) (Finset.range 61) = {6, 12, 30, 60} := by
    sorry
  -- Count the elements in the set
  rw [h]
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_60_multiple_of_6_l375_37500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_is_zero_one_l375_37573

/-- The point of tangency on the curve y = e^x, where the tangent line passes through (-1, 0) -/
def tangent_point : ℝ × ℝ := (0, 1)

/-- The curve y = e^x -/
noncomputable def curve (x : ℝ) : ℝ := Real.exp x

/-- The derivative of the curve y = e^x -/
noncomputable def curve_derivative (x : ℝ) : ℝ := Real.exp x

/-- The tangent line equation -/
noncomputable def tangent_line (t : ℝ) (x : ℝ) : ℝ := curve_derivative t * (x - t) + curve t

/-- The theorem stating that the tangent point has coordinates (0, 1) -/
theorem tangent_point_is_zero_one : 
  tangent_point = (0, 1) ∧ 
  tangent_line (tangent_point.1) (-1) = 0 ∧
  curve tangent_point.1 = tangent_point.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_is_zero_one_l375_37573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_angles_l375_37539

/-- The number of sides in a regular octagon -/
def n : ℕ := 8

/-- The sum of interior angles of a polygon -/
noncomputable def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- The measure of one interior angle in a regular polygon -/
noncomputable def interior_angle (n : ℕ) : ℝ := sum_interior_angles n / n

/-- The sum of exterior angles of any polygon -/
def sum_exterior_angles : ℝ := 360

/-- The measure of one exterior angle in a regular polygon -/
noncomputable def exterior_angle (n : ℕ) : ℝ := sum_exterior_angles / n

theorem regular_octagon_angles :
  interior_angle n = 135 ∧ exterior_angle n = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_angles_l375_37539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l375_37534

/-- Calculates the length of a train given its speed, the speed of a man running in the opposite direction, and the time it takes for the train to pass the man. -/
noncomputable def train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_ms := relative_speed * (1000 / 3600)
  relative_speed_ms * passing_time

/-- Theorem stating that under the given conditions, the train length is approximately 109.98 meters. -/
theorem train_length_approx :
  let train_speed := (40 : ℝ)
  let man_speed := (4 : ℝ)
  let passing_time := (9 : ℝ)
  abs (train_length train_speed man_speed passing_time - 109.98) < 0.01 :=
by
  sorry

-- Remove the #eval statement as it's not necessary for building
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l375_37534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_heights_properties_l375_37508

/-- Represents the relative heights of students in a class. -/
def relative_heights : List ℤ := [-7, 4, 0, 16, 2, -3, 1, -5, -9, 3, -4, 7, 1, -2, 1, 11]

/-- The reference height used for measuring relative heights. -/
def reference_height : ℕ := 160

/-- Theorem stating the properties of the class heights -/
theorem class_heights_properties :
  let n := relative_heights.length
  let tallest := relative_heights.maximum? -- Changed to maximum?
  let shortest := relative_heights.minimum? -- Changed to minimum?
  let sum := relative_heights.sum
  (
    (List.get! relative_heights 15 = 11) ∧ 
    (∀ t s, tallest = some t → shortest = some s → t - s = 25) ∧
    (((sum : ℚ) / n + reference_height) = 161)
  ) := by
  sorry

#eval relative_heights.maximum? -- For debugging
#eval relative_heights.minimum? -- For debugging

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_heights_properties_l375_37508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l375_37578

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n ∧ -3 < n ∧ n ≤ 1}

-- Define set B
def B : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_complement_B_l375_37578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_49_l375_37538

noncomputable def f (x : ℝ) : ℝ := 5 * x^2 - 4

noncomputable def g (y : ℝ) : ℝ := 
  let x := Real.sqrt ((y + 4) / 5)
  x^2 - x + 2

theorem sum_of_g_49 : 
  (g 49 + g 49) = 126/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_g_49_l375_37538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_age_ratio_l375_37564

-- Define Tom's age
def T : ℕ := 45

-- Define the sum of his children's current ages
def children_sum : ℕ := T / 3

-- Define Tom's age 5 years ago
def T_5_years_ago : ℕ := T - 5

-- Define the sum of his children's ages 5 years ago
def children_sum_5_years_ago : ℕ := children_sum - 10

-- State the theorem
theorem toms_age_ratio :
  (T = 3 * children_sum) ∧
  (T_5_years_ago = 3 * children_sum_5_years_ago) →
  (T = 45 ∧ T / 5 = 9) := by
  intro h
  -- The proof goes here
  sorry

#eval T
#eval T / 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_age_ratio_l375_37564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l375_37531

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

theorem equidistant_point :
  let A : Point3D := ⟨0, 0, -13/6⟩
  let B : Point3D := ⟨5, 1, 0⟩
  let C : Point3D := ⟨0, 2, 3⟩
  distance A B = distance A C := by
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l375_37531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_incurs_loss_l375_37509

/-- Represents the financial details of a radio sale --/
structure RadioSale where
  buying_price : ℚ
  overhead_cost : ℚ
  purchase_tax_rate : ℚ
  sales_tax_rate : ℚ
  selling_price : ℚ

/-- Calculates the profit from a radio sale --/
noncomputable def calculate_profit (sale : RadioSale) : ℚ :=
  let total_cost := sale.buying_price + sale.overhead_cost + (sale.purchase_tax_rate * sale.buying_price)
  let selling_price_before_tax := sale.selling_price / (1 + sale.sales_tax_rate)
  selling_price_before_tax - total_cost

/-- Theorem stating that the retailer incurs a loss in the given scenario --/
theorem retailer_incurs_loss : ∃ (sale : RadioSale), 
  sale.buying_price = 225 ∧ 
  sale.overhead_cost = 28 ∧ 
  sale.purchase_tax_rate = 8/100 ∧ 
  sale.sales_tax_rate = 12/100 ∧ 
  sale.selling_price = 300 ∧ 
  calculate_profit sale < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_incurs_loss_l375_37509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximation_l375_37589

/-- Calculates the annual interest rate given principal, time, and final amount -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (final_amount : ℝ) : ℝ :=
  (final_amount - principal) / (principal * time)

theorem interest_rate_approximation (principal time final_amount : ℝ) 
  (h1 : principal = 886.0759493670886)
  (h2 : time = 2.4)
  (h3 : final_amount = 1120) :
  abs (calculate_interest_rate principal time final_amount - 0.11) < 0.0001 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_interest_rate 886.0759493670886 2.4 1120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_approximation_l375_37589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l375_37586

theorem geometric_sequence_first_term (a : ℝ) (r : ℝ) :
  a * r^4 = Nat.factorial 7 ∧ a * r^7 = Nat.factorial 8 → a = 315 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l375_37586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l375_37597

theorem greatest_integer_fraction : 
  ⌊(5^80 + 3^80 : ℝ) / (5^75 + 3^75 : ℝ)⌋ = 3124 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_fraction_l375_37597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_selling_price_l375_37588

/-- Calculates the final selling price of a cycle in USD --/
noncomputable def final_selling_price (initial_price : ℝ) (additional_discount : ℝ) (vat : ℝ) 
                        (loss_on_sale : ℝ) (exchange_offer : ℝ) (sales_tax : ℝ) 
                        (initial_exchange_rate : ℝ) (final_exchange_rate : ℝ) : ℝ :=
  let discounted_price := initial_price * (1 - additional_discount)
  let price_with_vat := discounted_price * (1 + vat)
  let selling_price := initial_price * (1 - loss_on_sale)
  let price_after_exchange_offer := selling_price * (1 - exchange_offer)
  let price_with_sales_tax := price_after_exchange_offer * (1 + sales_tax)
  price_with_sales_tax / final_exchange_rate

/-- Theorem stating the final selling price of the cycle --/
theorem cycle_selling_price : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |final_selling_price 1400 0.05 0.15 0.25 0.10 0.05 70 75 - 13.23| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_selling_price_l375_37588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l375_37550

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 3 then 6 * x + 8 else 3 * x - 9

-- State the theorem
theorem sum_of_solutions :
  ∃ x₁ x₂ : ℝ, f x₁ = 3 ∧ f x₂ = 3 ∧ x₁ + x₂ = 19/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l375_37550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l375_37536

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - 2 * a * x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - 1/2 * x^2

-- Theorem for the tangent line
theorem tangent_line_at_one (a : ℝ) (h : a = 1) :
  let f' := λ x => 1/x + 2*x - 2
  (λ x => f' 1 * (x - 1) + f 1 1) = (λ x => x - 2) := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x > 1, g a x ≤ 0} = Set.Icc (-1/2) (1/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l375_37536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_is_32_over_3_l375_37540

/-- The area of a square inscribed in the ellipse x²/4 + y²/8 = 1, with its sides parallel to the coordinate axes -/
noncomputable def inscribed_square_area : ℝ := 32/3

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2/4 + y^2/8 = 1

/-- Theorem stating that the area of the inscribed square is 32/3 -/
theorem inscribed_square_area_is_32_over_3 :
  ∃ (t : ℝ), t > 0 ∧ 
  ellipse_equation t t ∧
  ellipse_equation (-t) t ∧
  ellipse_equation t (-t) ∧
  ellipse_equation (-t) (-t) ∧
  inscribed_square_area = (2*t)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_square_area_is_32_over_3_l375_37540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_property_l375_37523

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem inverse_function_property (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 9 = 2) :
  Function.invFun (f a) (Real.log 2 / Real.log a) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_property_l375_37523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l375_37566

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the concept of being on the left branch of the hyperbola
def on_left_branch (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2 ∧ P.1 < 0

-- Define the foci
def left_focus : ℝ × ℝ := (-2, 0)
def right_focus : ℝ × ℝ := (2, 0)

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the theorem
theorem hyperbola_property (P : ℝ × ℝ) :
  on_left_branch P →
  distance P left_focus - distance P right_focus = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l375_37566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_students_count_l375_37541

theorem exam_students_count : ∃ (total_count : ℕ),
  (total_count : ℝ) * 80 = 
    (total_count - 5 : ℝ) * 90 + 
    (5 : ℝ) * 20 ∧
  total_count = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_students_count_l375_37541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_study_score_theorem_l375_37576

/-- Represents a study session with its duration and effectiveness factor -/
structure StudySession where
  duration : ℝ
  effectivenessFactor : ℝ

/-- Represents a test score calculation model -/
structure ScoreModel where
  initialScore : ℝ
  initialSession : StudySession
  newSession : StudySession
  maxScore : ℝ

/-- Calculates the uncapped score based on the study model -/
noncomputable def calculateUncappedScore (model : ScoreModel) : ℝ :=
  let proportionalityConstant := model.initialScore / (model.initialSession.duration * model.initialSession.effectivenessFactor)
  proportionalityConstant * model.newSession.duration * model.newSession.effectivenessFactor

/-- Calculates the final capped score based on the study model -/
noncomputable def calculateFinalScore (model : ScoreModel) : ℝ :=
  min (calculateUncappedScore model) model.maxScore

theorem study_score_theorem (model : ScoreModel) 
  (h_initial_score : model.initialScore = 80)
  (h_initial_duration : model.initialSession.duration = 4)
  (h_initial_effectiveness : model.initialSession.effectivenessFactor = 1)
  (h_new_duration : model.newSession.duration = 5)
  (h_new_effectiveness : model.newSession.effectivenessFactor = 1.2)
  (h_max_score : model.maxScore = 100) :
  calculateFinalScore model = 100 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_study_score_theorem_l375_37576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pieces_on_board_l375_37522

/-- Represents a square on the board --/
structure Square where
  x : Nat
  y : Nat
  h1 : x ≥ 1 ∧ x ≤ 9
  h2 : y ≥ 1 ∧ y ≤ 5

/-- The color of a square --/
inductive Color
  | Red
  | Blue
  | Yellow

/-- Assigns a color to a square based on its coordinates --/
def colorSquare (s : Square) : Color :=
  if s.x % 2 = 0 ∧ s.y % 2 = 0 then Color.Red
  else if s.x % 2 = 1 ∧ s.y % 2 = 1 then Color.Blue
  else Color.Yellow

/-- Represents a move direction --/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Checks if a move is valid (alternating between horizontal and vertical) --/
def isValidMove (prev : Direction) (curr : Direction) : Bool :=
  match prev, curr with
  | Direction.Up, Direction.Left => true
  | Direction.Up, Direction.Right => true
  | Direction.Down, Direction.Left => true
  | Direction.Down, Direction.Right => true
  | Direction.Left, Direction.Up => true
  | Direction.Left, Direction.Down => true
  | Direction.Right, Direction.Up => true
  | Direction.Right, Direction.Down => true
  | _, _ => false

/-- The main theorem to prove --/
theorem max_pieces_on_board :
  ∃ (n : Nat), n = 32 ∧
  (∀ m : Nat, m > n →
    ¬∃ (initial_positions : List Square),
      (∀ s₁ s₂, s₁ ∈ initial_positions → s₂ ∈ initial_positions → s₁ ≠ s₂ → (s₁.x ≠ s₂.x ∨ s₁.y ≠ s₂.y)) ∧
      (∀ t : Nat, ∃ (positions : List Square) (moves : List Direction),
        positions.length = m ∧
        (∀ s₁ s₂, s₁ ∈ positions → s₂ ∈ positions → s₁ ≠ s₂ → (s₁.x ≠ s₂.x ∨ s₁.y ≠ s₂.y)) ∧
        (∀ i, i < t → i + 1 < moves.length → isValidMove (moves.get ⟨i, by sorry⟩) (moves.get ⟨i+1, by sorry⟩)))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pieces_on_board_l375_37522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_cost_price_l375_37526

/-- Represents the original cost price of the watch -/
def C : ℝ := sorry

/-- Represents the selling price of the watch -/
def SP : ℝ := 0.64 * C

/-- Represents the selling price before sales tax and discount -/
def SP' : ℝ := sorry

/-- The selling price after discount but before sales tax equals the actual selling price -/
axiom discount_applied : 0.9 * SP' = SP

/-- If sold for an additional 140, there would be a gain of 4% -/
axiom additional_sale : SP + 140 = 1.04 * C

/-- The original cost price of the watch is 350 -/
theorem watch_cost_price : C = 350 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_cost_price_l375_37526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fortieth_term_equals_21_l375_37525

def sequenceValue (n : ℕ) : ℤ := (100 - n + 1) - n

theorem fortieth_term_equals_21 : sequenceValue 40 = 21 := by
  rw [sequenceValue]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fortieth_term_equals_21_l375_37525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_properties_l375_37584

-- Define the parabola
def Γ (m : ℝ) (x y : ℝ) : Prop := y^2 = m*x

-- Define the point P
def P : ℝ × ℝ := (2, 2)

-- Define the points A and B
variable (x₁ y₁ x₂ y₂ : ℝ)
def A : ℝ × ℝ := (x₁, y₁)
def B : ℝ × ℝ := (x₂, y₂)

-- Define the slope of line AB
noncomputable def k : ℝ := (y₂ - y₁) / (x₂ - x₁)

-- State the theorem
theorem parabola_intersection_properties 
  (hP : Γ 2 P.1 P.2)
  (hA : Γ 2 x₁ y₁)
  (hB : Γ 2 x₂ y₂)
  (h_sym : x₁ + x₂ = 4)
  (h_order : x₁ < 2 ∧ 2 < x₂) :
  (k = -1/2) ∧ 
  (∃ (S : ℝ), S = (64 * Real.sqrt 3) / 9 ∧ 
    ∀ (S' : ℝ), S' = (1/2) * |x₂ - x₁| * |y₂ - 2| → S' ≤ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_properties_l375_37584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_problem_l375_37598

/-- Definition of a plane containing three points -/
def plane (p₁ p₂ p₃ : ℝ × ℝ × ℝ) : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | ∃ (a b : ℝ), p = p₁ + a • (p₂ - p₁) + b • (p₃ - p₁)}

/-- The plane equation problem -/
theorem plane_equation_problem (p₁ p₂ p₃ : ℝ × ℝ × ℝ) 
  (h₁ : p₁ = (2, -1, 3))
  (h₂ : p₂ = (5, -1, 1))
  (h₃ : p₃ = (7, 0, 2)) :
  ∃ (A B C D : ℤ), 
    (∀ (x y z : ℝ), (x, y, z) ∈ plane p₁ p₂ p₃ ↔ A * x + B * y + C * z + D = 0) ∧
    A > 0 ∧
    Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 ∧
    A = 2 ∧ B = -7 ∧ C = 3 ∧ D = -20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_problem_l375_37598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_21_hours_l375_37535

/-- Calculates the total journey time for a boat traveling upstream and downstream in a river -/
noncomputable def total_journey_time (river_speed : ℝ) (boat_speed : ℝ) (distance : ℝ) : ℝ :=
  let upstream_speed := boat_speed - river_speed
  let downstream_speed := boat_speed + river_speed
  let upstream_time := distance / upstream_speed
  let downstream_time := distance / downstream_speed
  upstream_time + downstream_time

/-- Theorem stating that under given conditions, the total journey time is 21 hours -/
theorem journey_time_is_21_hours 
  (river_speed : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_speed = 2)
  (h2 : boat_speed = 6)
  (h3 : distance = 56) :
  total_journey_time river_speed boat_speed distance = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_21_hours_l375_37535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_rule_accuracy_l375_37524

-- Define the function to be integrated
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x + 2)

-- Define the derivative of the function
noncomputable def f_prime (x : ℝ) : ℝ := -1 / (2 * (x + 2) ^ (3/2))

theorem midpoint_rule_accuracy (n : ℕ) (ε : ℝ) : 
  n ≥ 4 →
  ε = 0.1 →
  let a : ℝ := 2
  let b : ℝ := 7
  let M : ℝ := |f_prime a|  -- Maximum of |f'(x)| over [2, 7]
  ((b - a)^2 * M) / (2 * ↑n) ≤ ε :=
by
  sorry

#check midpoint_rule_accuracy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_rule_accuracy_l375_37524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_property_l375_37512

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem about the distance property of an ellipse -/
theorem ellipse_distance_property (a b : ℝ) (x1 y1 x2 y2 xf1 yf1 xf2 yf2 : ℝ) :
  is_ellipse a b x1 y1 →
  is_ellipse a b x2 y2 →
  distance xf1 yf1 x1 y1 + distance xf2 yf2 x1 y1 = 2 * a →
  distance xf1 yf1 x2 y2 + distance xf2 yf2 x2 y2 = 2 * a →
  distance xf2 yf2 x1 y1 + distance xf2 yf2 x2 y2 = 30 →
  distance x1 y1 x2 y2 = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_property_l375_37512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l375_37583

/-- Given a function f(x) = m * sin(ω * x) - cos(ω * x) with the following properties:
    1. x₀ = π/3 is an axis of symmetry for f(x)
    2. m > 0
    3. The smallest positive period of f(x) is π
    4. f(B) = 2 where B is an angle in a triangle ABC
    5. b = √3 where b is the side length opposite to angle B in triangle ABC

    This theorem proves:
    1. m = √3
    2. The intervals of monotonic increase for f(x) are [kπ - π/6, kπ + π/3] for k ∈ ℤ
    3. The range of a - c/2 is (-√3/2, √3) where a and c are side lengths in triangle ABC
-/
theorem function_properties (m ω : ℝ) (f : ℝ → ℝ) (A B C : ℝ) (a b c : ℝ) :
  (∀ x, f x = m * Real.sin (ω * x) - Real.cos (ω * x)) →
  (∀ x, f (π / 3 - x) = f (π / 3 + x)) →
  m > 0 →
  (∀ x, f (x + π) = f x) ∧ (∀ T, T > 0 → (∀ x, f (x + T) = f x) → T ≥ π) →
  f B = 2 →
  b = Real.sqrt 3 →
  (m = Real.sqrt 3 ∧
   (∀ k : ℤ, MonotoneOn f (Set.Icc (↑k * π - π / 6) (↑k * π + π / 3))) ∧
   Set.Ioo (-Real.sqrt 3 / 2) (Real.sqrt 3) (a - c / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l375_37583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_one_count_l375_37553

theorem opposite_of_one_count : ∃! (count : ℕ), count = 3 ∧
  count = (if (-1 : ℚ)^2 = -1 then 1 else 0) +
          (if |(-1 : ℚ)| = -1 then 1 else 0) +
          (if (1 : ℚ) / -1 = -1 then 1 else 0) +
          (if (-1 : ℚ)^2023 = -1 then 1 else 0) +
          (if -((-1) : ℚ) = -1 then 1 else 0) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_one_count_l375_37553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_l375_37518

/-- Calculates the average runs for a batsman over all matches -/
noncomputable def average_runs (first_matches : ℕ) (first_average : ℝ) (second_matches : ℕ) (second_average : ℝ) : ℝ :=
  ((first_matches : ℝ) * first_average + (second_matches : ℝ) * second_average) / ((first_matches + second_matches) : ℝ)

theorem batsman_average :
  let first_matches : ℕ := 15
  let first_average : ℝ := 30
  let second_matches : ℕ := 20
  let second_average : ℝ := 15
  average_runs first_matches first_average second_matches second_average = 25 := by
  simp [average_runs]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_l375_37518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leftover_solid_height_is_correct_l375_37594

/-- The height of the leftover solid when a unit cube is cut through the midpoints of edges
    emanating from one vertex and the cut face is placed on a table. -/
noncomputable def leftover_solid_height : ℝ := (Real.sqrt 3 - 1) / Real.sqrt 3

/-- Theorem stating the height of the leftover solid -/
theorem leftover_solid_height_is_correct :
  let cube_edge_length : ℝ := 1
  let cut_through_midpoints : Bool := true
  let cut_face_on_table : Bool := true
  leftover_solid_height = (Real.sqrt 3 - 1) / Real.sqrt 3 :=
by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leftover_solid_height_is_correct_l375_37594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_rule_derivative_l375_37558

-- Define the function f(x) = x^a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^a

-- State the theorem
theorem power_rule_derivative (a : ℝ) (x : ℝ) (h : x > 0) :
  deriv (f a) x = a * x^(a - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_rule_derivative_l375_37558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_rose_price_is_6_l375_37563

/-- The price of an individual rose -/
def individual_rose_price : ℚ := 6

/-- The cost of one dozen roses -/
def dozen_cost : ℚ := 36

/-- The cost of two dozen roses -/
def two_dozen_cost : ℚ := 50

/-- The maximum budget -/
def max_budget : ℚ := 680

/-- The maximum number of roses that can be purchased with the max_budget -/
def max_roses : ℕ := 317

theorem individual_rose_price_is_6 :
  individual_rose_price = 6 ∧
  dozen_cost = 12 * individual_rose_price ∧
  two_dozen_cost = 24 * individual_rose_price ∧
  max_roses = ⌊max_budget / individual_rose_price⌋ :=
by
  sorry

#eval individual_rose_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_rose_price_is_6_l375_37563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_is_two_l375_37579

/-- Given a cone with slant height h and central angle θ of its unfolded lateral surface,
    calculates the radius of its base. -/
noncomputable def cone_base_radius (h : ℝ) (θ : ℝ) : ℝ :=
  h * (θ / (2 * Real.pi))

/-- Theorem stating that a cone with slant height 6 cm and central angle 120° of its
    unfolded lateral surface has a base radius of 2 cm. -/
theorem cone_radius_is_two :
  cone_base_radius 6 (2 * Real.pi / 3) = 2 := by
  -- Unfold the definition of cone_base_radius
  unfold cone_base_radius
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_radius_is_two_l375_37579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BDGF_is_338_l375_37591

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the square BDEC
structure Square where
  B : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  C : ℝ × ℝ

-- Define the points F and G
noncomputable def F : ℝ × ℝ := (312/17, 0)
noncomputable def G : ℝ × ℝ := (312/17, 26)

-- Define the given triangle
noncomputable def givenTriangle : Triangle := {
  A := (0, 24),
  B := (0, 0),
  C := (26, 0)
}

-- Define the square BDEC
noncomputable def squareBDEC : Square := {
  B := (0, 0),
  D := (26, 26),
  E := (0, 26),
  C := (26, 0)
}

-- Function to calculate the area of a quadrilateral
noncomputable def areaQuadrilateral (a b c d : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_BDGF_is_338 (ABC : Triangle) (BDEC : Square) :
  ABC.A = (0, 24) →
  ABC.B = (0, 0) →
  ABC.C = (26, 0) →
  BDEC.B = (0, 0) →
  BDEC.D = (26, 26) →
  BDEC.E = (0, 26) →
  BDEC.C = (26, 0) →
  areaQuadrilateral BDEC.B BDEC.D G F = 338 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_BDGF_is_338_l375_37591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l375_37510

noncomputable def f (x : ℝ) : ℝ := 2^(x + 2) - 3 * 4^x

theorem f_range :
  (∀ y ∈ Set.Ioo (-4 : ℝ) (4/3), ∃ x < 1, f x = y) ∧
  (∀ x < 1, f x ∈ Set.Icc (-4 : ℝ) (4/3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l375_37510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_implies_x_values_l375_37537

-- Define the points M and N
def M : ℝ × ℝ := (-1, 4)
def N (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem distance_implies_x_values :
  ∀ x : ℝ, distance M (N x) = 5 → x = -6 ∨ x = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_implies_x_values_l375_37537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_correct_proposition_l375_37507

-- Define the types for lines and planes
structure Line where

structure Plane where

-- Define the relationships between lines and planes
axiom parallel_line_plane : Line → Plane → Prop
axiom parallel_lines : Line → Line → Prop
axiom line_in_plane : Line → Plane → Prop
axiom perpendicular_lines : Line → Line → Prop
axiom perpendicular_line_plane : Line → Plane → Prop

-- Define the propositions
def proposition1 (a b : Line) (M : Plane) : Prop :=
  (parallel_line_plane a M ∧ parallel_line_plane b M) → parallel_lines a b

def proposition2 (a b : Line) (M : Plane) : Prop :=
  (line_in_plane b M ∧ parallel_lines a b) → parallel_line_plane a M

def proposition3 (a b c : Line) : Prop :=
  (perpendicular_lines a c ∧ perpendicular_lines b c) → parallel_lines a b

def proposition4 (a b : Line) (M : Plane) : Prop :=
  (perpendicular_line_plane a M ∧ perpendicular_line_plane b M) → parallel_lines a b

-- The main theorem
theorem one_correct_proposition (a b c : Line) (M : Plane) :
  ∃! (p : Prop), p = proposition1 a b M ∨ p = proposition2 a b M ∨ 
                 p = proposition3 a b c ∨ p = proposition4 a b M :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_correct_proposition_l375_37507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_theorem_l375_37545

/-- Represents a train with its length and speed. -/
structure Train where
  length : ℝ
  speed : ℝ

/-- Represents a platform with its length. -/
structure Platform where
  length : ℝ

/-- Calculates the time taken for a train to cross a platform. -/
noncomputable def crossingTime (t : Train) (p : Platform) : ℝ :=
  (t.length + p.length) / t.speed

/-- Theorem: A train of length 100 m crossing a 200 m platform in 15 seconds
    will cross a 300 m platform in 20 seconds. -/
theorem train_crossing_time_theorem 
  (t : Train) 
  (p1 p2 : Platform) 
  (h1 : t.length = 100)
  (h2 : p1.length = 200)
  (h3 : p2.length = 300)
  (h4 : crossingTime t p1 = 15) :
  crossingTime t p2 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_theorem_l375_37545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_l375_37574

noncomputable section

-- Define the points
def B : ℝ × ℝ := (0, 0)
def C (x : ℝ) : ℝ × ℝ := (x, 0)
def A (a b : ℝ) : ℝ × ℝ := (a, b)
def D (x : ℝ) : ℝ × ℝ := (2*x, 0)
def E (a b : ℝ) : ℝ × ℝ := (-a/2, -b/2)
def F (a b : ℝ) : ℝ × ℝ := (3*a, 3*b)
def G : ℝ × ℝ := (32, 24)

-- Define the centroid of a triangle
def centroid (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem centroid_distance :
  ∀ x a b : ℝ,
  x + a = 96 →
  b = 72 →
  let K := centroid (D x) (E a b) (F a b)
  distance G K = 48 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_l375_37574
