import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_14_l108_10865

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  let (x₃, y₃) := c
  (1/2) * abs (x₁*(y₂-y₃) + x₂*(y₃-y₁) + x₃*(y₁-y₂))

/-- The intersection point of a Line with the line x + y = 8 -/
noncomputable def intersectionWithSum8 (l : Line) : ℝ × ℝ :=
  sorry

theorem triangle_area_is_14 (l₁ l₂ : Line) :
  l₁.point = (1, 3) ∧ l₁.slope = 1 ∧
  l₂.point = (1, 3) ∧ l₂.slope = -1/2 →
  triangleArea l₁.point l₂.point (intersectionWithSum8 l₁) = 14 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_14_l108_10865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_investment_after_five_years_l108_10824

-- Define the compound interest function
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- Define the parameters for both accounts
def P1 : ℝ := 600
def r1 : ℝ := 0.10
def n1 : ℝ := 12
def P2 : ℝ := 400
def r2 : ℝ := 0.08
def n2 : ℝ := 4
def t : ℝ := 5

-- Define the theorem
theorem total_investment_after_five_years :
  abs ((compound_interest P1 r1 n1 t) + (compound_interest P2 r2 n2 t) - 1554.998) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_investment_after_five_years_l108_10824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l108_10868

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_decreasing : ∀ x y, x ∈ Set.Icc (-2) 2 → y ∈ Set.Icc (-2) 2 → x < y → f x > f y

-- Define the theorem
theorem range_of_m (m : ℝ) : 
  (f (1 - m) + f (1 - m^2) < 0) ↔ (m ∈ Set.Icc (-1) 1 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l108_10868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_pi_twelfths_l108_10820

theorem tan_difference_pi_twelfths : Real.tan (π / 12) - Real.tan (5 * π / 12) = -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_pi_twelfths_l108_10820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l108_10836

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.log x - 2 * x) / x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := (1 - Real.log x) / (x^2)

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -2
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - y - 3 = 0) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l108_10836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_meet_at_height_l108_10832

/-- Represents a point moving along a line segment --/
structure MovingPoint where
  start : ℝ × ℝ  -- Starting coordinates
  direction : ℝ × ℝ  -- Direction vector
  speed : ℝ  -- Speed in m/s

/-- Calculates the position of a moving point after t seconds --/
def position (p : MovingPoint) (t : ℝ) : ℝ × ℝ :=
  (p.start.1 + t * p.speed * p.direction.1, p.start.2 + t * p.speed * p.direction.2)

/-- An equilateral triangle with side length 52 meters --/
def triangle_side : ℝ := 52

/-- Height of the equilateral triangle --/
noncomputable def triangle_height : ℝ := triangle_side * Real.sqrt 3 / 2

/-- Point P moving from A to C --/
def point_p : MovingPoint :=
  { start := (0, 0), direction := (1, 0), speed := 3 }

/-- Point Q moving from B to C --/
noncomputable def point_q : MovingPoint :=
  { start := (triangle_side / 2, triangle_height), direction := (1/2, -Real.sqrt 3 / 2), speed := 4 }

/-- Distance between two points --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem points_meet_at_height : 
  distance (position point_p 2) (position point_q 2) = triangle_height := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_meet_at_height_l108_10832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aggregate_sales_value_l108_10812

/-- Represents the aggregate sales value of the output of the looms -/
def S : ℕ → ℕ := sorry

/-- The number of looms -/
def total_looms : ℕ := 70

/-- Monthly manufacturing expenses in rupees -/
def manufacturing_expenses : ℕ := 150000

/-- Monthly establishment charges in rupees -/
def establishment_charges : ℕ := 75000

/-- Total monthly expenses in rupees -/
def total_expenses : ℕ := manufacturing_expenses + establishment_charges

/-- The decrease in profit when one loom breaks down, in rupees -/
def profit_decrease : ℕ := 5000

theorem aggregate_sales_value :
  ∃ (s : ℕ), S total_looms = s ∧
  s / total_looms = profit_decrease ∧
  s = 350000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aggregate_sales_value_l108_10812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l108_10860

-- Define the complex number z
noncomputable def z (a : ℝ) : ℂ := (a + 2 * Complex.I ^ 3) / (2 - Complex.I)

-- Define the condition for z being in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop := 
  Complex.re z > 0 ∧ Complex.im z < 0

-- Theorem statement
theorem a_range (a : ℝ) : 
  in_fourth_quadrant (z a) ↔ -1 < a ∧ a < 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l108_10860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l108_10816

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℚ),
  N.mulVec ![4, 0] = ![12, 8] ∧
  N.mulVec ![2, -3] = ![6, -10] ∧
  N = !![3, 0; 2, 14/3] := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l108_10816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l108_10837

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- State the theorem
theorem floor_equation_solution :
  ∀ x : ℝ, (floor (2 * x) + floor (3 * x) = 95) ↔ (19 ≤ x ∧ x < 58 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_equation_solution_l108_10837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_filtration_after_six_hours_check_exp_approximation_l108_10864

/-- Represents the filtration process of pollutants over time -/
structure FiltrationProcess where
  N₀ : ℝ  -- Initial amount of pollutants
  k : ℝ   -- Filtration rate constant

/-- Calculates the remaining pollutants after time t -/
noncomputable def remaining_pollutants (fp : FiltrationProcess) (t : ℝ) : ℝ :=
  fp.N₀ * Real.exp (-fp.k * t)

/-- Calculates the percentage of pollutants filtered out after time t -/
noncomputable def filtered_percentage (fp : FiltrationProcess) (t : ℝ) : ℝ :=
  (1 - remaining_pollutants fp t / fp.N₀) * 100

theorem filtration_after_six_hours 
  (fp : FiltrationProcess)
  (h : filtered_percentage fp 2 = 30) :
  ∃ ε > 0, |filtered_percentage fp 6 - 65.7| < ε := by
  sorry

-- This is now a theorem instead of an evaluation
theorem check_exp_approximation :
  ∃ ε > 0, |Real.exp (-Real.log 0.7) - 0.7| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_filtration_after_six_hours_check_exp_approximation_l108_10864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l108_10804

noncomputable def curve_C (θ : Real) : Real := 3 / (2 - Real.cos θ)

def line_l (t : Real) : Real × Real := (3 + t, 2 + 2*t)

def intersection_points (A B : Real × Real) : Prop :=
  ∃ (θ₁ θ₂ t₁ t₂ : Real),
    0 ≤ θ₁ ∧ θ₁ < 2*Real.pi ∧
    0 ≤ θ₂ ∧ θ₂ < 2*Real.pi ∧
    A.1 = curve_C θ₁ * Real.cos θ₁ ∧
    A.2 = curve_C θ₁ * Real.sin θ₁ ∧
    B.1 = curve_C θ₂ * Real.cos θ₂ ∧
    B.2 = curve_C θ₂ * Real.sin θ₂ ∧
    A = line_l t₁ ∧
    B = line_l t₂

theorem intersection_distance (A B : Real × Real) :
  intersection_points A B → Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 60/19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l108_10804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_warehouse_problem_l108_10823

def weights : List ℕ := [150, 160, 180, 190, 200, 310]

theorem warehouse_problem :
  ∀ (first_shipment second_shipment : List ℕ),
    first_shipment.length = 2 →
    second_shipment.length = 3 →
    first_shipment ⊆ weights →
    second_shipment ⊆ weights →
    first_shipment ∩ second_shipment = ∅ →
    (first_shipment.sum : ℚ) = (second_shipment.sum : ℚ) / 2 →
    ∃! w, w ∈ weights ∧ w ∉ first_shipment ∧ w ∉ second_shipment ∧ w = 200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_warehouse_problem_l108_10823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l108_10802

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℚ),
  N.vecMul (![2, 3] : Fin 2 → ℚ) = ![3, -6] ∧
  N.vecMul (![4, -1] : Fin 2 → ℚ) = ![19, -2] ∧
  N = !![30/7, -13/7; -6/7, -10/7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l108_10802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_m_range_l108_10874

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x / (x + 1)

-- Theorem for monotonicity
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Theorem for the range of m
theorem m_range :
  ∀ m : ℝ, (f (2*m - 1) > f (1 - m)) ↔ (2/3 < m ∧ m < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_m_range_l108_10874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_point_sets_l108_10843

-- Definition of a perpendicular point set
def is_perpendicular_point_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (p1 : ℝ × ℝ), p1 ∈ M →
    ∃ (p2 : ℝ × ℝ), p2 ∈ M ∧ p1.1 * p2.1 + p1.2 * p2.2 = 0

-- Define the sets
def M1 : Set (ℝ × ℝ) := {p | p.2 = 1 / (p.1 ^ 2)}
def M2 : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1 / Real.log 2}
def M3 : Set (ℝ × ℝ) := {p | p.2 = 2 ^ p.1 - 2}
def M4 : Set (ℝ × ℝ) := {p | p.2 = Real.sin p.1 + 1}

-- Theorem statement
theorem perpendicular_point_sets :
  is_perpendicular_point_set M1 ∧
  is_perpendicular_point_set M3 ∧
  is_perpendicular_point_set M4 ∧
  ¬is_perpendicular_point_set M2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_point_sets_l108_10843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_time_l108_10801

/-- Calculates compound interest -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Calculates simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem simple_interest_time : 
  let principal_simple := (1750 : ℝ)
  let rate_simple := (0.08 : ℝ)
  let principal_compound := (4000 : ℝ)
  let rate_compound := (0.10 : ℝ)
  let time_compound := (2 : ℝ)
  ∃ t : ℝ, 
    simpleInterest principal_simple rate_simple t = 
    (1/2) * compoundInterest principal_compound rate_compound time_compound ∧
    t = 3 := by
  sorry

#check simple_interest_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_time_l108_10801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_less_than_b_7_l108_10827

def b (α : ℕ → ℕ) : ℕ → ℚ
  | 0 => 1  -- Add a case for 0
  | 1 => 1 + 1 / α 1
  | n + 1 => 1 + 1 / (b α n)

theorem b_4_less_than_b_7 (α : ℕ → ℕ) (h : ∀ k, α k > 0) : b α 4 < b α 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_less_than_b_7_l108_10827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_fixed_point_g_has_two_fixed_points_l108_10825

-- Define a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Function definitions
noncomputable def f (x : ℝ) : ℝ := 1 + Real.log x
noncomputable def g (x : ℝ) : ℝ := |2 - 1/x|

-- Theorem statements
theorem f_has_one_fixed_point :
  ∃! x : ℝ, is_fixed_point f x :=
sorry

theorem g_has_two_fixed_points :
  ∃ x y : ℝ, x ≠ y ∧ is_fixed_point g x ∧ is_fixed_point g y ∧
  ∀ z : ℝ, is_fixed_point g z → z = x ∨ z = y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_fixed_point_g_has_two_fixed_points_l108_10825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elliptical_field_fencing_cost_l108_10852

-- Define the Ramanujan approximation for ellipse perimeter
noncomputable def ramanujanPerimeter (a b : ℝ) : ℝ :=
  Real.pi * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))

-- Define the cost calculation function
def fencingCost (perimeter rate : ℝ) : ℝ :=
  perimeter * rate

-- Theorem statement
theorem elliptical_field_fencing_cost :
  let a : ℝ := 16  -- semi-major axis
  let b : ℝ := 12  -- semi-minor axis
  let rate : ℝ := 3  -- fencing rate per meter
  let perimeter := ramanujanPerimeter a b
  let cost := fencingCost perimeter rate
  ∃ ε > 0, |cost - 265.32| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elliptical_field_fencing_cost_l108_10852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_always_intersect_l108_10880

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Uniform distribution on a line segment -/
def uniformOnSegment (a b : ℝ) : Type := { x : ℝ // a ≤ x ∧ x ≤ b }

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Whether two circles intersect -/
def circlesIntersect (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center ≤ c1.radius + c2.radius

theorem circles_always_intersect :
  ∀ (x : ℝ), 0 ≤ x → x ≤ 5 →
  let circleA : Circle := { center := (x, 0), radius := 2 }
  let circleB : Circle := { center := (3, 2), radius := 3 }
  circlesIntersect circleA circleB := by
    sorry

#check circles_always_intersect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_always_intersect_l108_10880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_bag_price_l108_10862

/-- Given the following conditions:
  * Total weight of potatoes: 6500 kg
  * Damaged weight: 150 kg
  * Bag size: 50 kg
  * Total sales: $9144
  Prove that the price per bag is $72 -/
theorem potato_bag_price (total_weight damaged_weight bag_size total_sales : ℕ) :
  total_weight = 6500 ∧ 
  damaged_weight = 150 ∧ 
  bag_size = 50 ∧ 
  total_sales = 9144 →
  (total_sales : ℚ) / ((total_weight - damaged_weight) / bag_size) = 72 := by
  intro h
  sorry

#check potato_bag_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_potato_bag_price_l108_10862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_coefficient_l108_10883

/-- 
Given a function f(x) = (1/2) * sin(2x) + a * cos(2x) that is symmetric about x = π/12,
prove that a = √3/2.
-/
theorem symmetric_function_coefficient (a : ℝ) : 
  (∀ x : ℝ, (1/2) * Real.sin (2*x) + a * Real.cos (2*x) = 
             (1/2) * Real.sin (2*(π/6 - x)) + a * Real.cos (2*(π/6 - x))) →
  a = Real.sqrt 3 / 2 :=
by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_coefficient_l108_10883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l108_10861

/-- A power function that passes through the point (4, 1/2) -/
noncomputable def f (x : ℝ) : ℝ := x^(-1/2 : ℝ)

theorem power_function_through_point :
  (∀ x > 0, f x = x^(-1/2 : ℝ)) ∧ f 4 = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l108_10861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_product_of_f_l108_10831

open Real

theorem max_min_product_of_f (f : ℝ → ℝ) (M m : ℝ) : 
  (∀ x, f x = Real.sqrt (4 * sin x ^ 4 - sin x ^ 2 * cos x ^ 2 + 4 * cos x ^ 4)) →
  (∀ x, f x ≤ M) →
  (∃ x, f x = M) →
  (∀ x, m ≤ f x) →
  (∃ x, f x = m) →
  M * m = Real.sqrt 42 := by
  sorry

#check max_min_product_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_product_of_f_l108_10831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_coin_types_l108_10849

/-- Represents the types of coins -/
inductive CoinType
| Gold
| Silver
| Copper

/-- Represents a coin with its weight -/
structure Coin where
  type : CoinType
  weight : Nat

/-- Represents a collection of coins -/
structure CoinCollection where
  coins : List Coin
  has_all_types : ∀ t : CoinType, ∃ c ∈ coins, c.type = t

/-- Represents a weighing operation -/
def Weighing := List Coin → List Coin → Ordering

/-- The main theorem to be proved -/
theorem determine_coin_types 
  (n : Nat) 
  (h_n : n ≥ 3) 
  (collection : CoinCollection) 
  (h_collection : collection.coins.length = n) :
  ∃ (strategy : Nat → Weighing), 
    (∀ i, i ≤ n + 1 → ∃ result, strategy i = result) ∧ 
    (∀ c ∈ collection.coins, ∃ i ≤ n + 1, 
      strategy i [c] [Coin.mk CoinType.Gold 3, Coin.mk CoinType.Silver 2, Coin.mk CoinType.Copper 1] = 
        match c.type with
        | CoinType.Gold => Ordering.eq
        | CoinType.Silver => Ordering.lt
        | CoinType.Copper => Ordering.lt
    ) := by sorry

#check determine_coin_types

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_coin_types_l108_10849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_l108_10834

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 3) ℤ := !![7, 3, -1; 0, 2, 4]

theorem matrix_product :
  A * B = !![21, 11, 1; 28, 8, -12] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_l108_10834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_projection_exists_for_14_teeth_complete_projection_not_exists_for_13_teeth_l108_10859

/-- Represents a gear with a given number of teeth and removed teeth positions -/
structure Gear (n : ℕ) where
  teeth : Fin n → Bool
  removed_count : ℕ
  removed_count_eq : removed_count = 4

/-- Predicate to check if two gears can form a complete projection after rotation -/
def can_form_complete_projection (n : ℕ) (gear_a gear_b : Gear n) : Prop :=
  ∃ (rotation : Fin n), ∀ (i : Fin n),
    gear_a.teeth i ∨ gear_b.teeth (Fin.add i rotation) = true

/-- Theorem stating the existence of a complete projection for 14-tooth gears -/
theorem complete_projection_exists_for_14_teeth :
    ∀ (gear_a gear_b : Gear 14),
      can_form_complete_projection 14 gear_a gear_b := by sorry

/-- Theorem stating the non-existence of a complete projection for 13-tooth gears -/
theorem complete_projection_not_exists_for_13_teeth :
    ∀ (gear_a gear_b : Gear 13),
      ¬(can_form_complete_projection 13 gear_a gear_b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_projection_exists_for_14_teeth_complete_projection_not_exists_for_13_teeth_l108_10859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_is_2pi_l108_10806

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.tan x + (Real.tan x)⁻¹ - (Real.cos x)⁻¹

-- State the theorem
theorem f_period_is_2pi :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q :=
by
  -- We'll use 2π as the period
  use 2 * Real.pi
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_is_2pi_l108_10806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_circle_l108_10811

/-- The function f(x) = (x - 2010)(x + 2011) -/
def f (x : ℝ) : ℝ := (x - 2010) * (x + 2011)

/-- The three intersection points of f with the coordinate axes -/
def A : ℝ × ℝ := (-2011, 0)
def B : ℝ × ℝ := (2010, 0)
def C : ℝ × ℝ := (0, -2010 * 2011)

/-- Define a circle passing through a point -/
def CircleThrough (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

theorem intersection_point_of_circle (h : ∃ (center : ℝ × ℝ) (radius : ℝ), 
  CircleThrough center radius A ∧ CircleThrough center radius B ∧ CircleThrough center radius C) :
  ∃ (D : ℝ × ℝ), D = (0, 1) ∧ 
  (∃ (center : ℝ × ℝ) (radius : ℝ), CircleThrough center radius A ∧ 
   CircleThrough center radius B ∧ CircleThrough center radius C ∧ CircleThrough center radius D) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_of_circle_l108_10811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l108_10873

/-- Represents a point on the ellipse -/
structure EllipsePoint where
  x : ℝ
  y : ℝ
  h : 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 25 * y + 55 = 0
  first_quadrant : x > 0 ∧ y > 0

/-- The ratio y/x for a point on the ellipse -/
noncomputable def ratio (p : EllipsePoint) : ℝ := p.y / p.x

/-- The maximum value of the ratio y/x on the ellipse -/
noncomputable def max_ratio : ℝ := sorry

/-- The minimum value of the ratio y/x on the ellipse -/
noncomputable def min_ratio : ℝ := sorry

/-- The theorem to be proved -/
theorem ellipse_ratio_sum :
  max_ratio + min_ratio = 52 / 51 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_sum_l108_10873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l108_10895

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) (F1 F2 A B : Point) (l : Line) :
  -- The line passes through F1 and intersects the hyperbola at A and B
  (F1.x * l.m + l.c = F1.y) →
  (A.x^2 / h.a^2 - A.y^2 / h.b^2 = 1) →
  (B.x^2 / h.a^2 - B.y^2 / h.b^2 = 1) →
  -- AB = AF2
  (distance A B = distance A F2) →
  -- Angle F1AF2 is 90 degrees
  ((F1.x - A.x) * (F2.x - A.x) + (F1.y - A.y) * (F2.y - A.y) = 0) →
  -- Then the eccentricity is sqrt(6) + sqrt(3)
  eccentricity h = Real.sqrt 6 + Real.sqrt 3 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l108_10895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sin_cos_product_l108_10844

/-- Given vectors a and b in a plane, where a is parallel to b, 
    prove that sin α cos α = 12/25 -/
theorem parallel_vectors_sin_cos_product (α : ℝ) : 
  let a : Fin 2 → ℝ := ![4, 3]
  let b : Fin 2 → ℝ := ![Real.sin α, Real.cos α]
  (∃ (k : ℝ), ∀ (i : Fin 2), a i = k * b i) →
  Real.sin α * Real.cos α = 12/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sin_cos_product_l108_10844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l108_10838

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ S → y ∈ S → x < y → f x < f y

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_increasing : increasing_on f (Set.Ici 0))
  (h_f2 : f 2 = 0) :
  {x : ℝ | (f x + f (-x)) / x < 0} = Set.Iio (-2) ∪ Set.Ioo 0 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l108_10838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poetry_books_count_l108_10846

/-- The number of books Nancy shelved from the cart -/
def total_books : ℕ := 46

/-- The number of history books Nancy shelved -/
def history_books : ℕ := 12

/-- The number of romance books Nancy shelved -/
def romance_books : ℕ := 8

/-- The number of Western novels Nancy shelved -/
def western_books : ℕ := 5

/-- The number of biographies Nancy shelved -/
def biography_books : ℕ := 6

/-- The number of mystery books is equal to the number of Western novels and biographies combined -/
def mystery_books_count : ℕ := western_books + biography_books

/-- The theorem stating that Nancy shelved 4 poetry books -/
theorem poetry_books_count : 
  total_books - (history_books + romance_books + western_books + biography_books + mystery_books_count) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poetry_books_count_l108_10846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_determination_l108_10892

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_determination 
  (A ω φ : ℝ) 
  (h1 : A > 0) 
  (h2 : ω > 0) 
  (h3 : |φ| ≤ π) 
  (h4 : f A ω φ (π/6) = -3) 
  (h5 : f A ω φ (2*π/3) = 3) :
  ∀ x, f A ω φ x = 3 * Real.sin (2*x - 5*π/6) := by
  sorry

#check function_determination

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_determination_l108_10892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l108_10869

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := 
  if x > 1 then Real.log (x - 1) / Real.log a
  else -Real.log (1 - x) / Real.log a

theorem f_solution_set (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  (∀ x, x ≠ 1 → f a x = f a (2 - x)) →
  f a 3 = -1 →
  {x : ℝ | f a x > 1} = Set.Ioi (-1) ∪ Set.Icc 1 (3/2) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l108_10869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spot_ratio_lower_bound_spot_ratio_lower_bound_tight_l108_10886

/-- Represents a configuration of dark spots on a square tablecloth -/
structure SpotConfiguration where
  /-- Total area of dark spots -/
  S : ℝ
  /-- Visible area after folding along specified lines -/
  S₁ : ℝ
  /-- S₁ is at most S -/
  h_S₁_le_S : S₁ ≤ S
  /-- S and S₁ are non-negative -/
  h_S_nonneg : 0 ≤ S
  h_S₁_nonneg : 0 ≤ S₁
  /-- S₁ equals S when folded along one diagonal -/
  h_S₁_eq_S_one_diagonal : S₁ = S

/-- The ratio S₁/S is always greater than or equal to 2/3 -/
theorem spot_ratio_lower_bound (c : SpotConfiguration) : 2 / 3 ≤ c.S₁ / c.S := by
  sorry

/-- The lower bound 2/3 can be achieved -/
theorem spot_ratio_lower_bound_tight : 
  ∃ (c : SpotConfiguration), c.S₁ / c.S = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spot_ratio_lower_bound_spot_ratio_lower_bound_tight_l108_10886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fdi_rural_andhra_pradesh_l108_10839

-- Define the total FDI
variable (F : ℝ) 

-- Define the FDI proportions for each state and region
def gujarat_fdi (F : ℝ) : ℝ := 0.3 * F
def gujarat_rural_fdi (F : ℝ) : ℝ := 0.2 * gujarat_fdi F
def gujarat_urban_fdi (F : ℝ) : ℝ := 0.8 * gujarat_fdi F
def andhra_pradesh_fdi (F : ℝ) : ℝ := 0.2 * F
def andhra_pradesh_rural_fdi (F : ℝ) : ℝ := 0.5 * andhra_pradesh_fdi F

-- State the theorem
theorem fdi_rural_andhra_pradesh (F : ℝ) :
  gujarat_urban_fdi F = 72 → andhra_pradesh_rural_fdi F = 30 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fdi_rural_andhra_pradesh_l108_10839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_permutations_count_l108_10854

-- Define a multiset with two 5s and two 7s
def digit_multiset : Multiset ℕ := {5, 5, 7, 7}

-- Define the number of elements in the multiset
def n : ℕ := 4

-- Define the number of occurrences of each repeated element
def r1 : ℕ := 2  -- number of 5s
def r2 : ℕ := 2  -- number of 7s

-- Theorem: The number of unique permutations of the multiset is 6
theorem unique_permutations_count : 
  (Nat.factorial n) / (Nat.factorial r1 * Nat.factorial r2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_permutations_count_l108_10854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l108_10888

-- Define the function f with domain (0, 2)
def f : Set ℝ := Set.Ioo 0 2

-- Define the function g(x) = f(2x)
def g : Set ℝ := { x | 2 * x ∈ f }

-- Theorem statement
theorem domain_of_g : g = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l108_10888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_one_two_zeros_condition_l108_10870

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x - a else 4*(x-a)*(x-2*a)

-- Theorem 1: Minimum value of f when a = 1 is -1
theorem min_value_when_a_is_one :
  ∃ (m : ℝ), (∀ x, f 1 x ≥ m) ∧ (∃ x, f 1 x = m) ∧ m = -1 := by
  sorry

-- Theorem 2: f has exactly 2 zeros iff 1/2 ≤ a < 1 or a ≥ 2
theorem two_zeros_condition (a : ℝ) :
  (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ 
  ((1/2 ≤ a ∧ a < 1) ∨ a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_one_two_zeros_condition_l108_10870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_from_square_l108_10851

/-- The area of a rhombus formed by connecting the midpoints of a square with side length 4 cm is 8 cm². -/
theorem rhombus_area_from_square (s : ℝ) (h : s = 4) : 
  (s * s) / 2 = 8 := by
  -- Substitute s = 4
  rw [h]
  -- Simplify
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_from_square_l108_10851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_cannot_win_l108_10887

-- Define the players
inductive Player : Type
| X : Player  -- First player
| O : Player  -- Second player

-- Define the board as a 3x3 grid
def Board : Type := Fin 3 → Fin 3 → Option Player

-- Define a winning state
def WinningState (board : Board) (player : Player) : Prop :=
  (∃ (row : Fin 3), (∀ (col : Fin 3), board row col = some player)) ∨
  (∃ (col : Fin 3), (∀ (row : Fin 3), board row col = some player)) ∨
  (∀ (i : Fin 3), board i i = some player) ∨
  (∀ (i : Fin 3), board i (2 - i) = some player)

-- Define optimal play for the first player
def OptimalPlay (board : Board) (player : Player) : Prop :=
  -- This is a placeholder for the optimal strategy
  True

-- The main theorem
theorem second_player_cannot_win 
  (initial_board : Board) 
  (game : ℕ → Board) 
  (h_optimal : ∀ n, OptimalPlay (game n) Player.X) :
  ¬ ∃ n, WinningState (game n) Player.O :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_cannot_win_l108_10887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_EFGH_l108_10817

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_EFGH (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- Angle F is 100°
  ∃ (θf : ℝ), θf = 100 * Real.pi / 180 ∧
  -- Angle G is 140°
  ∃ (θg : ℝ), θg = 140 * Real.pi / 180 ∧
  -- EF = 6
  Real.sqrt ((fx - ex)^2 + (fy - ey)^2) = 6 ∧
  -- FG = 5
  Real.sqrt ((gx - fx)^2 + (gy - fy)^2) = 5 ∧
  -- GH = 7
  Real.sqrt ((hx - gx)^2 + (hy - gy)^2) = 7

-- Theorem statement
theorem area_of_EFGH (q : Quadrilateral) (h : is_valid_EFGH q) :
  ∃ (area : ℝ), abs (area - 26.02) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_EFGH_l108_10817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_transformation_l108_10805

theorem fraction_transformation (a b c d : ℝ) :
  let original := (5 * a - 2 * b) / (4 * b + 2 * a)
  let transformed := (20 * c - 20 * d + 13) / 18
  (a = Real.rpow 6 (1/3) ∧ b = Real.rpow 12 (1/3) ∧ c = Real.rpow 4 (1/3) ∧ d = Real.rpow 2 (1/3)) →
  original = transformed :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_transformation_l108_10805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l108_10821

/-- Represents an ellipse with equation x²/5 + y²/(5+m) = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ (x y : ℝ), x^2/5 + y^2/(5+m) = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (m : ℝ) (e : Ellipse m) : ℝ := 1/2

/-- Theorem: For an ellipse with equation x²/5 + y²/(5+m) = 1 and eccentricity 1/2,
    the value of m is either -5/4 or 5/3 -/
theorem ellipse_m_values (m : ℝ) (e : Ellipse m) 
    (h : eccentricity m e = 1/2) : m = -5/4 ∨ m = 5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_values_l108_10821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_radius_3_l108_10808

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- Theorem: The volume of a sphere with radius 3 cm is 36π cm³ -/
theorem sphere_volume_radius_3 :
  sphere_volume 3 = 36 * Real.pi := by
  unfold sphere_volume
  simp [Real.pi]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_radius_3_l108_10808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sequence_smallest_n_is_2016_l108_10863

open Fin

theorem smallest_n_for_sequence : ℕ → Prop :=
  λ N => ∃ (a : Fin 125 → ℕ),
    (∀ i, a i > 0 ∧ a i ≤ N) ∧
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    (∀ i : Fin 123, a (i.succ) > (a i + a (i.succ.succ)) / 2)

theorem smallest_n_is_2016 :
  smallest_n_for_sequence 2016 ∧ ∀ m < 2016, ¬smallest_n_for_sequence m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_sequence_smallest_n_is_2016_l108_10863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l108_10835

-- Define the constants and functions
noncomputable def x : ℝ := (1 + Real.sqrt 2) ^ 500
noncomputable def n : ℤ := ⌊x⌋
noncomputable def f : ℝ := x - n

-- State the theorem
theorem x_times_one_minus_f_equals_one : x * (1 - f) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l108_10835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l108_10840

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {x | ∃ n : Int, x = 2 / (n - 1) ∧ x ∈ U}

theorem complement_of_A_in_U : 
  (U \ A) = {0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l108_10840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_particle_movement_l108_10818

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a particle moving along the edges of a triangle -/
structure Particle where
  position : ℝ → Point
  speed : ℝ

/-- Predicate to ensure the triangle is equilateral -/
def IsEquilateral (t : Triangle) : Prop :=
  sorry

/-- Predicate to ensure the correct movement direction -/
def MovementDirection (p1 p2 : Particle) (t : Triangle) : Prop :=
  sorry

/-- The ratio of areas for the given particle movement scenario -/
noncomputable def areaRatio (t : Triangle) (p1 p2 : Particle) : ℝ :=
  sorry

/-- Main theorem statement -/
theorem equilateral_triangle_particle_movement
  (t : Triangle)
  (p1 p2 : Particle)
  (h_equilateral : IsEquilateral t)
  (h_start_positions : p1.position 0 = t.A ∧ p2.position 0 = t.C)
  (h_speed_ratio : p1.speed = 2 * p2.speed)
  (h_direction : MovementDirection p1 p2 t) :
  areaRatio t p1 p2 = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_particle_movement_l108_10818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_citric_acid_weight_l108_10828

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in C6H8O7 -/
def carbon_count : ℕ := 6

/-- The number of Hydrogen atoms in C6H8O7 -/
def hydrogen_count : ℕ := 8

/-- The number of Oxygen atoms in C6H8O7 -/
def oxygen_count : ℕ := 7

/-- The number of moles of C6H8O7 -/
def moles : ℝ := 7

/-- The molecular weight of C6H8O7 in g/mol -/
def molecular_weight : ℝ :=
  carbon_weight * (carbon_count : ℝ) +
  hydrogen_weight * (hydrogen_count : ℝ) +
  oxygen_weight * (oxygen_count : ℝ)

/-- The total weight of the given moles of C6H8O7 in grams -/
def total_weight : ℝ := molecular_weight * moles

theorem citric_acid_weight : ∃ ε > 0, |total_weight - 1344.868| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_citric_acid_weight_l108_10828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_AB_is_correct_l108_10822

noncomputable def A : ℝ × ℝ := (1, 0)
noncomputable def B : ℝ × ℝ := (0, 1)

noncomputable def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

noncomputable def magnitude_AB : ℝ := Real.sqrt ((vector_AB.1)^2 + (vector_AB.2)^2)

noncomputable def unit_vector_AB : ℝ × ℝ := (vector_AB.1 / magnitude_AB, vector_AB.2 / magnitude_AB)

theorem unit_vector_AB_is_correct : 
  unit_vector_AB = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_AB_is_correct_l108_10822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l108_10866

/-- Given an arithmetic sequence {a_n} with common difference d and S_n as the sum of the first n terms,
    if (S_{2017}/2017) - (S_{17}/17) = 100, then d = 1/10 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + d) →
  (∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2) →
  S 2017 / 2017 - S 17 / 17 = 100 →
  d = 1 / 10 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l108_10866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_year_date_inequality_l108_10803

def leap_year_dates : List ℕ := 
  (List.range 30).bind (λ i => List.replicate 12 (i + 1)) ++ List.replicate 7 31

def d : ℚ := 31 / 2

noncomputable def μ : ℚ := (leap_year_dates.sum : ℚ) / leap_year_dates.length

def M : ℚ := 16

theorem leap_year_date_inequality : d < μ ∧ μ < M := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_year_date_inequality_l108_10803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l108_10830

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A1 B1 C1 A2 B2 C2 : ℝ) : ℝ :=
  |A1 * C2 - A2 * C1| / Real.sqrt ((A1^2 + B1^2) * (A2^2 + B2^2))

/-- Theorem: The distance between the parallel lines x+2y-1=0 and 2x+4y+1=0 is 3√5/10 -/
theorem distance_between_given_lines :
  distance_between_parallel_lines 1 2 (-1) 2 4 1 = 3 * Real.sqrt 5 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l108_10830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l108_10890

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) :
  (∃ α : ℝ, f = power_function α) →  -- f is a power function
  f 2 = 1/4 →                        -- f passes through (2, 1/4)
  f (1/2) = 4 :=                     -- prove that f(1/2) = 4
by
  intro h1 h2
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l108_10890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse1_theorem_ellipse2_theorem_l108_10878

-- Define the ellipse equation type
def EllipseEq := ℝ → ℝ → Prop

-- Define the general ellipse equation
def generalEllipse (a b : ℝ) : EllipseEq :=
  λ x y ↦ x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions for the first ellipse
def ellipse1Conditions (eq : EllipseEq) : Prop :=
  eq 0 0 ∧ eq (Real.sqrt 6) 1 ∧ eq (-Real.sqrt 3) (-Real.sqrt 2)

-- Define the conditions for the second ellipse
def ellipse2Conditions (eq : EllipseEq) : Prop :=
  eq 3 0 ∧ ∃ a b : ℝ, (a = 3 * b ∨ b = 3 * a) ∧
    (∀ x y, eq x y ↔ x^2 / a^2 + y^2 / b^2 = 1)

-- Theorem for the first ellipse
theorem ellipse1_theorem :
  ellipse1Conditions (generalEllipse 3 (Real.sqrt 3)) :=
by sorry

-- Theorem for the second ellipse
theorem ellipse2_theorem :
  ellipse2Conditions (generalEllipse 3 1) ∧
  ellipse2Conditions (generalEllipse 3 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse1_theorem_ellipse2_theorem_l108_10878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_hyperbola_l108_10876

/-- The points of intersection of two parametric lines form a hyperbola -/
theorem intersection_points_form_hyperbola :
  ∃ (A B : ℝ), A > 0 ∧ B > 0 ∧
  ∀ (t : ℝ), 
    (let x := (-5 - 4*t^2) / (2 - 2*t^2)
     let y := (-2*t*(11 + 4*t^2)) / (6*(1 - t^2))
     x^2/A - y^2/B = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_hyperbola_l108_10876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pompeiu_theorem_l108_10879

-- Define the rotation operator
noncomputable def rotate (α : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 * Real.cos α - v.2 * Real.sin α, v.1 * Real.sin α + v.2 * Real.cos α)

-- Define an equilateral triangle
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  equilateral : B = rotate (2 * Real.pi / 3) A ∧ C = rotate (4 * Real.pi / 3) A

-- Define a point
def Point := ℝ × ℝ

-- Define the circumcircle of a triangle
def onCircumcircle (X : Point) (t : EquilateralTriangle) : Prop :=
  ∃ (center : Point) (radius : ℝ),
    (X.1 - center.1)^2 + (X.2 - center.2)^2 = radius^2 ∧
    (t.A.1 - center.1)^2 + (t.A.2 - center.2)^2 = radius^2 ∧
    (t.B.1 - center.1)^2 + (t.B.2 - center.2)^2 = radius^2 ∧
    (t.C.1 - center.1)^2 + (t.C.2 - center.2)^2 = radius^2

-- Define a degenerate triangle
def degenerateTriangle (a b c : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), b = (k * a.1, k * a.2) ∨ c = (k * a.1, k * a.2) ∨ c = (k * b.1, k * b.2)

-- Main theorem
theorem pompeiu_theorem (t : EquilateralTriangle) (X : Point) :
  let XA : ℝ × ℝ := (X.1 - t.A.1, X.2 - t.A.2)
  let XB : ℝ × ℝ := (X.1 - t.B.1, X.2 - t.B.2)
  let XC : ℝ × ℝ := (X.1 - t.C.1, X.2 - t.C.2)
  (∃ (a b c : ℝ), a * XA.1 + b * XB.1 + c * XC.1 = 0 ∧
                   a * XA.2 + b * XB.2 + c * XC.2 = 0 ∧
                   a + b + c = 0) ∧
  (degenerateTriangle XA XB XC ↔ onCircumcircle X t) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pompeiu_theorem_l108_10879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_in_square_l108_10853

theorem rectangle_length_in_square (area : ℝ) (n : ℕ) (h1 : area = 5760) (h2 : n = 8) :
  ∃ (x : ℝ), 
    (n : ℝ) * x^2 = area ∧ 
    Int.floor (x + 0.5) = 27 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_in_square_l108_10853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_proof_l108_10897

-- Define the original line
noncomputable def originalLine (x : ℝ) : ℝ := (1/2) * x + 1

-- Define the perpendicular line
noncomputable def perpendicularLine (x : ℝ) : ℝ := -2 * x + 4

-- Define the point that the perpendicular line passes through
def point : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem perpendicular_line_proof :
  -- The slopes are negative reciprocals of each other
  (∀ x y, y = originalLine x → (y - originalLine x) / (x - x) = 1/2) ∧
  (∀ x y, y = perpendicularLine x → (y - perpendicularLine x) / (x - x) = -2) ∧
  -- The slopes multiply to -1
  (1/2 * -2 = -1) ∧
  -- The perpendicular line passes through the given point
  perpendicularLine point.fst = point.snd :=
by
  sorry

#check perpendicular_line_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_proof_l108_10897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l108_10875

/-- The distance of the race track in yards. -/
noncomputable def d : ℝ := sorry

/-- The speed of racer A. -/
noncomputable def a : ℝ := sorry

/-- The speed of racer B. -/
noncomputable def b : ℝ := sorry

/-- The speed of racer C. -/
noncomputable def c : ℝ := sorry

/-- A can beat B by 20 yards. -/
axiom A_beats_B : d / a = (d - 20) / b

/-- B can beat C by 10 yards. -/
axiom B_beats_C : d / b = (d - 10) / c

/-- A can beat C by 28 yards. -/
axiom A_beats_C : d / a = (d - 28) / c

/-- All speeds are positive. -/
axiom positive_speeds : 0 < a ∧ 0 < b ∧ 0 < c

/-- The distance of the race track is 100 yards. -/
theorem race_distance : d = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l108_10875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_bounds_l108_10894

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.log x - a

theorem zeros_sum_bounds {a : ℝ} {x₁ x₂ : ℝ} 
  (h₁ : f a x₁ = 0) 
  (h₂ : f a x₂ = 0) 
  (h₃ : x₁ ≠ x₂) : 
  1 < x₁ + x₂ ∧ x₁ + x₂ < 2 / Real.sqrt (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_sum_bounds_l108_10894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l108_10856

/-- Represents a conic section (ellipse or hyperbola) -/
structure Conic where
  a : ℝ
  b : ℝ

/-- The eccentricity of a conic section -/
def eccentricity (c : Conic) : ℝ := sorry

/-- Checks if two conics share the same foci -/
def share_foci (c1 c2 : Conic) : Prop := sorry

/-- The equation of a conic section -/
def conic_equation (c : Conic) (x y : ℝ) : ℝ := sorry

theorem hyperbola_equation (e : Conic) (h : Conic) :
  conic_equation e 5 3 = (1 : ℝ) →
  share_foci e h →
  eccentricity e + eccentricity h = 14/5 →
  conic_equation h 2 (2 * Real.sqrt 3) = (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l108_10856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_line_with_x_intercept_line_with_y_intercept_l108_10807

-- Define the slope of line l
noncomputable def slope_l : ℝ := Real.sqrt 3 / 3

-- Theorem for case 1
theorem line_through_point (x y : ℝ) :
  (x = 3 ∧ y = -4) →
  (Real.sqrt 3 * x - 3 * y - 3 * Real.sqrt 3 - 12 = 0) ↔
  (y - (-4) = slope_l * (x - 3)) :=
by sorry

-- Theorem for case 2
theorem line_with_x_intercept (x y : ℝ) :
  (x = -2 ∧ y = 0) →
  (Real.sqrt 3 * x - 3 * y + 2 * Real.sqrt 3 = 0) ↔
  (y = slope_l * (x + 2)) :=
by sorry

-- Theorem for case 3
theorem line_with_y_intercept (x y : ℝ) :
  (Real.sqrt 3 * x - 3 * y + 9 = 0) ↔
  (y = slope_l * x + 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_line_with_x_intercept_line_with_y_intercept_l108_10807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_unique_parallel_plane_l108_10896

/-- Represents a line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines intersect -/
def do_intersect (l1 l2 : Line3D) : Prop :=
  sorry

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ¬ are_parallel l1 l2 ∧ ¬ do_intersect l1 l2

/-- A plane contains a line -/
def plane_contains_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

/-- A plane is parallel to a line -/
def plane_parallel_to_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

/-- Main theorem: For any two skew lines, there exists a unique plane that contains
    one line and is parallel to the other -/
theorem skew_lines_unique_parallel_plane (l1 l2 : Line3D) 
  (h : are_skew l1 l2) : 
  ∃! p : Plane3D, (plane_contains_line p l1 ∧ plane_parallel_to_line p l2) ∨
                  (plane_contains_line p l2 ∧ plane_parallel_to_line p l1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_lines_unique_parallel_plane_l108_10896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_bound_l108_10850

/-- A function f: [0,1] → ℝ⁺ satisfying certain properties -/
structure SpecialFunction where
  f : Set.Icc 0 1 → ℝ
  f_nonneg : ∀ x, 0 ≤ f x
  f_one : f 1 = 1
  f_superadditive : ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x + y ≤ 1 → f (⟨x + y, by sorry⟩) ≥ f ⟨x, by sorry⟩ + f ⟨y, by sorry⟩

/-- The main theorem stating the upper bound and its optimality -/
theorem special_function_bound (f : SpecialFunction) :
  (∀ x : Set.Icc 0 1, f.f x ≤ 2 * x) ∧
  (∀ c : ℝ, c < 2 → ∃ x : Set.Icc 0 1, f.f x > c * x.val) :=
by
  sorry

#check special_function_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_bound_l108_10850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l108_10845

-- Define the family of curves
def curve (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

-- Define the line
def line (x y : ℝ) : Prop :=
  y = 2 * x

-- Define the chord length function
noncomputable def chord_length (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

-- Theorem statement
theorem max_chord_length :
  ∃ (x y : ℝ), curve (Real.pi / 4) x y ∧ line x y ∧
    (∀ (x' y' θ' : ℝ), curve θ' x' y' ∧ line x' y' →
      chord_length x y ≥ chord_length x' y') ∧
    chord_length x y = 8 * Real.sqrt 5 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l108_10845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangent_lines_with_slope_three_l108_10819

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem two_tangent_lines_with_slope_three :
  ∃! (s : Finset ℝ), (∀ x ∈ s, f' x = 3) ∧ (s.card = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangent_lines_with_slope_three_l108_10819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_cylinder_volume_l108_10855

/-- Represents a cylinder with radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Given two cylinders with specific ratios and the volume of the first cylinder,
    prove that the volume of the second cylinder is 900 cc -/
theorem second_cylinder_volume 
  (c1 c2 : Cylinder) 
  (h_radii : c2.radius = 3 * c1.radius) 
  (h_heights : c2.height = 5/2 * c1.height)
  (h_vol1 : cylinderVolume c1 = 40) :
  cylinderVolume c2 = 900 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_cylinder_volume_l108_10855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_min_max_values_l108_10882

-- Part 1
theorem calculate_expression : 
  (2.25 ^ 0.5) - ((-9.6) ^ 0) - ((27/8) ^ (-2/3)) + ((3/2) ^ (-2)) = 1/2 := by sorry

-- Part 2
noncomputable def f (x : ℝ) := (Real.log x / Real.log (1/4))^2 - (Real.log x / Real.log (1/4)) + 5

theorem min_max_values (x : ℝ) (h : x ∈ Set.Icc 2 4) :
  ∃ (min max : ℝ), (∀ y ∈ Set.Icc 2 4, f y ≥ min ∧ f y ≤ max) ∧ 
  min = 23/4 ∧ max = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_min_max_values_l108_10882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_000216_l108_10826

theorem cube_root_of_000216 : Real.rpow 0.000216 (1/3) = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_000216_l108_10826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l108_10871

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - Real.log (1 - x)

-- State the theorem
theorem f_properties :
  -- f is defined on (-1, 1)
  (∀ x : ℝ, -1 < x ∧ x < 1 → f x = Real.log (1 + x) - Real.log (1 - x)) ∧
  -- f is an odd function
  (∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = -f x) ∧
  -- f is increasing on (0, 1)
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1 → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l108_10871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinate_product_l108_10810

/-- Given a line segment with one endpoint at (10, 7) and midpoint at (4, -3),
    the product of the coordinates of the other endpoint is 26. -/
theorem endpoint_coordinate_product :
  ∀ (endpoint1 midpoint endpoint2 : ℝ × ℝ),
    endpoint1 = (10, 7) ∧ 
    midpoint = (4, -3) ∧ 
    (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2) ∧
    (midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 * endpoint2.2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinate_product_l108_10810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l108_10815

-- Define the right triangle
noncomputable def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ 0 < a ∧ 0 < b ∧ 0 < c

-- Define the area of a triangle
noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (1/2) * base * height

-- Theorem statement
theorem right_triangle_area :
  ∀ (a b c : ℝ),
  right_triangle a b c →
  a = 24 →
  c = 30 →
  triangle_area a b = 216 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l108_10815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l108_10877

open Real

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) / (2*x - 2)

-- State the theorem
theorem f_max_value :
  ∃ (c : ℝ), c ∈ Set.Ioo (-4 : ℝ) 1 ∧
  (∀ x ∈ Set.Ioo (-4 : ℝ) 1, f x ≤ f c) ∧
  f c = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l108_10877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l108_10872

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  a : ℝ  -- Length of longer parallel side
  b : ℝ  -- Length of shorter parallel side
  h : ℝ  -- Height (distance between parallel sides)
  θ : ℝ  -- Angle between shorter parallel side and one non-parallel side

/-- Calculates the area of a trapezium -/
noncomputable def area (t : Trapezium) : ℝ :=
  (1/2) * (t.a + t.b) * t.h

/-- Calculates the length of the non-parallel side adjacent to the shorter parallel side -/
noncomputable def nonParallelSideLength (t : Trapezium) : ℝ :=
  t.h / Real.tan (t.θ * Real.pi / 180)

theorem trapezium_properties (t : Trapezium) 
  (ha : t.a = 26) 
  (hb : t.b = 18) 
  (hh : t.h = 15) 
  (hθ : t.θ = 35) : 
  area t = 330 ∧ 
  abs (nonParallelSideLength t - 21.42) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l108_10872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l108_10814

noncomputable def f (x : ℝ) := x + Real.sqrt (x^2 - 9) + 1 / (x - Real.sqrt (x^2 - 9))

noncomputable def g (x : ℝ) := x^3 + Real.sqrt (x^6 - 1) + 1 / (x^3 + Real.sqrt (x^6 - 1))

theorem problem_solution (x : ℝ) (h : f x = 100) : 
  ‖g x - 31507.361338‖ < 1e-6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l108_10814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_triangle_area_theorem_l108_10893

/-- The area of the triangle formed by the intersections of the internal angle bisectors
    and the corresponding sides of a triangle with sides a, b, and c. -/
noncomputable def angle_bisector_triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (2 * a * b * c * Real.sqrt (s * (s - a) * (s - b) * (s - c))) / ((a + b) * (b + c) * (c + a))

/-- Theorem stating that the area of the triangle formed by the intersections of the internal
    angle bisectors and the corresponding sides of a triangle with sides a, b, and c
    is equal to the formula given by angle_bisector_triangle_area. -/
theorem angle_bisector_triangle_area_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let s := (a + b + c) / 2
  let original_triangle_area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let bisector_intersections_area := original_triangle_area * (2 * a * b * c) / ((a + b) * (b + c) * (c + a))
  bisector_intersections_area = angle_bisector_triangle_area a b c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_triangle_area_theorem_l108_10893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dodgeball_cost_l108_10842

/-- The cost of a dodgeball given budget increase and softball purchases -/
theorem dodgeball_cost (original_budget : ℝ) (new_budget : ℝ) (dodgeball_cost : ℝ)
  (h1 : new_budget = 1.2 * original_budget)
  (h2 : original_budget = 15 * dodgeball_cost)
  (h3 : new_budget = 10 * 9) : 
  dodgeball_cost = 5 := by
  sorry

#check dodgeball_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dodgeball_cost_l108_10842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l108_10884

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ := (sin (π * x)) / (x^2)

noncomputable def g (x : ℝ) : ℝ := f x + f (1 - x)

theorem min_value_of_g :
  ∀ x ∈ Set.Ioo 0 1, g x ≥ 8 ∧ ∃ y ∈ Set.Ioo 0 1, g y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l108_10884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l108_10891

theorem similar_triangles_side_length (A₁ A₂ : ℝ) (s : ℝ) :
  A₁ - A₂ = 32 →
  A₁ / A₂ = 9 →
  ∃ n : ℕ, A₂ = n →
  s = 5 →
  ∃ (S : ℝ), S = 15 ∧ (S / s)^2 = A₁ / A₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_side_length_l108_10891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangency_internal_tangency_common_chord_equation_common_chord_length_l108_10833

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 1 = 0
def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + m = 0

-- Theorem for external tangency
theorem external_tangency :
  ∃ m : ℝ, m = 25 + 10 * Real.sqrt 11 ∧
  ∀ x y : ℝ, circle1 x y → circle2 x y m →
  ∃! p : ℝ × ℝ, circle1 p.1 p.2 ∧ circle2 p.1 p.2 m :=
sorry

-- Theorem for internal tangency
theorem internal_tangency :
  ∃ m : ℝ, m = 25 - 10 * Real.sqrt 11 ∧
  ∀ x y : ℝ, circle1 x y → circle2 x y m →
  ∃! p : ℝ × ℝ, circle1 p.1 p.2 ∧ circle2 p.1 p.2 m :=
sorry

-- Theorem for common chord equation when m = 45
theorem common_chord_equation :
  ∀ x y : ℝ, circle1 x y → circle2 x y 45 →
  4*x + 3*y - 23 = 0 :=
sorry

-- Theorem for common chord length when m = 45
noncomputable def common_points : Set (ℝ × ℝ) :=
  {p | circle1 p.1 p.2 ∧ circle2 p.1 p.2 45}

theorem common_chord_length :
  ∃ p₁ p₂ : ℝ × ℝ, p₁ ∈ common_points ∧ p₂ ∈ common_points ∧
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) = 4 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangency_internal_tangency_common_chord_equation_common_chord_length_l108_10833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_ratio_is_one_l108_10813

def admission_problem (adult_fee child_fee total : ℕ) : Prop :=
  ∃ (a c : ℕ), 
    a ≥ 1 ∧ 
    c ≥ 1 ∧ 
    a * adult_fee + c * child_fee = total ∧
    ∀ (a' c' : ℕ), 
      a' ≥ 1 → 
      c' ≥ 1 → 
      a' * adult_fee + c' * child_fee = total → 
      |(a' : ℚ) / c' - 1| ≥ |(a : ℚ) / c - 1|

theorem closest_ratio_is_one : 
  admission_problem 30 15 2250 → 
  ∃ (a c : ℕ), a = c ∧ a * 30 + c * 15 = 2250 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_ratio_is_one_l108_10813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horses_equivalent_to_camel_l108_10829

/-- The cost of one elephant in rupees -/
noncomputable def elephant_cost : ℝ := 110000 / 10

/-- The cost of one ox in rupees -/
noncomputable def ox_cost : ℝ := (4 * elephant_cost) / 6

/-- The cost of one horse in rupees -/
noncomputable def horse_cost : ℝ := (4 * ox_cost) / 16

/-- The cost of one camel in rupees -/
def camel_cost : ℝ := 4400

/-- The number of horses equivalent to the cost of one camel -/
noncomputable def horses_per_camel : ℝ := camel_cost / horse_cost

theorem horses_equivalent_to_camel : 
  ⌊horses_per_camel⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horses_equivalent_to_camel_l108_10829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MN_l108_10898

noncomputable section

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 * (t - 2 * Real.sqrt 2), t)

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := Real.sqrt (4 / (1 + 3 * Real.sin θ ^ 2))

-- Define the angle β
noncomputable def β : ℝ := Real.arctan (-1/2)

-- Define point M on curve C
noncomputable def point_M : ℝ × ℝ := (curve_C β * Real.cos β, curve_C β * Real.sin β)

-- Define point N on line l
noncomputable def point_N : ℝ × ℝ := line_l (2 * Real.sqrt 2 + Real.sqrt 10 / 2)

-- Statement to prove
theorem distance_MN : 
  let M := point_M
  let N := point_N
  Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) = Real.sqrt 10 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MN_l108_10898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_rearrangements_sufficient_l108_10809

/-- Represents a rearrangement operation on a sequence -/
def Rearrangement (α : Type*) := (Fin 100 → α) → (Fin 100 → α)

/-- Checks if a rearrangement only affects 50 consecutive elements -/
def is_valid_rearrangement {α : Type*} (r : Rearrangement α) : Prop :=
  ∃ (start : Fin 100), ∀ (i : Fin 100), i < start ∨ i ≥ start + 50 → 
    ∀ (seq : Fin 100 → α), r seq i = seq i

/-- Checks if a sequence is sorted in descending order -/
def is_sorted_desc {α : Type*} [Preorder α] (seq : Fin 100 → α) : Prop :=
  ∀ (i j : Fin 100), i < j → seq i ≥ seq j

/-- The main theorem stating that 6 rearrangements are sufficient to sort the sequence -/
theorem six_rearrangements_sufficient {α : Type*} [LinearOrder α] :
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : Rearrangement α),
    (is_valid_rearrangement r₁) ∧
    (is_valid_rearrangement r₂) ∧
    (is_valid_rearrangement r₃) ∧
    (is_valid_rearrangement r₄) ∧
    (is_valid_rearrangement r₅) ∧
    (is_valid_rearrangement r₆) ∧
    (∀ (seq : Fin 100 → α), is_sorted_desc (r₆ (r₅ (r₄ (r₃ (r₂ (r₁ seq)))))))
    := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_rearrangements_sufficient_l108_10809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_triangle_area_l108_10848

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Reflect a point about the line x = k -/
def reflect (p : Point) (k : ℝ) : Point :=
  { x := 2 * k - p.x, y := p.y }

/-- Calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let x1 := t.a.x
  let y1 := t.a.y
  let x2 := t.b.x
  let y2 := t.b.y
  let x3 := t.c.x
  let y3 := t.c.y
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Calculate the area of the union of two triangles -/
noncomputable def unionArea (t1 t2 : Triangle) : ℝ :=
  triangleArea t1 + triangleArea t2 - triangleArea { a := t1.a, b := t1.b, c := t2.c }

/-- The main theorem -/
theorem reflected_triangle_area :
  let original := Triangle.mk
    (Point.mk 6 5)
    (Point.mk 8 (-3))
    (Point.mk 9 1)
  let reflected := Triangle.mk
    (reflect (Point.mk 6 5) 8)
    (reflect (Point.mk 8 (-3)) 8)
    (reflect (Point.mk 9 1) 8)
  unionArea original reflected = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_triangle_area_l108_10848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l108_10885

/-- Definition of the function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(-5*m - 3)

/-- A power function has the form a * x^n where a and n are constants -/
def is_power_function (g : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), ∀ x, g x = a * x^n

/-- A direct proportionality function has the form k * x where k is a constant -/
def is_direct_proportional (g : ℝ → ℝ) : Prop :=
  ∃ k, ∀ x, g x = k * x

/-- An inverse proportionality function has the form k / x where k is a constant -/
def is_inverse_proportional (g : ℝ → ℝ) : Prop :=
  ∃ k, ∀ x, g x = k / x

/-- A quadratic function has the form a * x^2 + b * x + c where a, b, c are constants and a ≠ 0 -/
def is_quadratic (g : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, g x = a * x^2 + b * x + c

theorem f_properties :
  (is_power_function (f 2) ∧ is_power_function (f (-1))) ∧
  is_direct_proportional (f (-4/5)) ∧
  is_inverse_proportional (f (-2/5)) ∧
  is_quadratic (f (-1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l108_10885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_always_wins_l108_10867

def P (X a₃ a₂ a₁ : ℤ) : ℤ := X^4 + a₃*X^3 + a₂*X^2 + a₁*X - 1

theorem first_player_always_wins (a₃ a₂ a₁ : ℤ) (h₁ : a₃ ≠ 0) (h₂ : a₂ ≠ 0) (h₃ : a₁ ≠ 0) :
  ¬(∃ x y : ℤ, x ≠ y ∧ P x a₃ a₂ a₁ = 0 ∧ P y a₃ a₂ a₁ = 0) ∧
  ¬(∃ x : ℤ, P x a₃ a₂ a₁ = 0 ∧ (fun X => 4*X^3 + 3*a₃*X^2 + 2*a₂*X + a₁) x = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_always_wins_l108_10867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l108_10881

/-- Conversion factor from kilometers to miles -/
noncomputable def km_to_mile : ℝ := 1 / 1.60934

/-- Total distance of the journey in miles -/
def total_distance : ℝ := 300

/-- Speed for each quarter of the journey -/
def speed_q1 : ℝ := 75  -- mph
noncomputable def speed_q2 : ℝ := 45 * km_to_mile  -- 45 km/h converted to mph
noncomputable def speed_q3 : ℝ := 50 * km_to_mile  -- 50 km/h converted to mph
def speed_q4 : ℝ := 90  -- mph

/-- Time taken for each quarter of the journey -/
noncomputable def time_q1 : ℝ := total_distance / 4 / speed_q1
noncomputable def time_q2 : ℝ := total_distance / 4 / speed_q2
noncomputable def time_q3 : ℝ := total_distance / 4 / speed_q3
noncomputable def time_q4 : ℝ := total_distance / 4 / speed_q4

/-- Total time of the journey -/
noncomputable def total_time : ℝ := time_q1 + time_q2 + time_q3 + time_q4

/-- Average speed of the entire journey -/
noncomputable def average_speed : ℝ := total_distance / total_time

theorem journey_average_speed : 
  ∃ ε > 0, |average_speed - 43.29| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l108_10881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_area_l108_10889

/-- The area of the overlapping region of two squares -/
theorem overlapping_squares_area (β : Real) (h1 : 0 < β) (h2 : β < Real.pi / 2) (h3 : Real.cos β = 3 / 5) :
  let square_side : Real := 2
  let overlap_area : Real := 2 / 3
  overlap_area = (square_side ^ 2 - 2 * (square_side ^ 2 * (1 - Real.cos β) / 2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_area_l108_10889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_value_l108_10899

open Real

noncomputable def f (a x : ℝ) : ℝ := log x - a * x + (1 - a) / x - 1

noncomputable def g (b x : ℝ) : ℝ := x^2 - 2 * b * x - 5/9

theorem min_b_value (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 0 1, f (1/3) x₁ ≥ g b x₂) →
  b ≥ 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_b_value_l108_10899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pqr_area_l108_10841

/-- The area of a triangle given by three points in a 2D plane -/
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  let (px, py) := p
  let (qx, qy) := q
  let (rx, ry) := r
  (1/2) * abs ((qx - px) * (ry - py) - (rx - px) * (qy - py))

/-- Theorem: The area of triangle PQR with given coordinates is 36 square units -/
theorem triangle_pqr_area :
  let p : ℝ × ℝ := (-4, 2)
  let q : ℝ × ℝ := (8, 2)
  let r : ℝ × ℝ := (6, -4)
  triangleArea p q r = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_pqr_area_l108_10841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_120_degrees_when_norms_equal_l108_10857

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

noncomputable def angle_between_vectors (a b : E) : ℝ := Real.arccos ((inner a b) / (‖a‖ * ‖b‖))

theorem angle_120_degrees_when_norms_equal (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_norm : ‖a‖ = ‖b‖ ∧ ‖a‖ = ‖a + b‖) : 
  angle_between_vectors a b = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_120_degrees_when_norms_equal_l108_10857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l108_10858

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 16 / (x + y)^2 ≥ 4 * Real.sqrt 2 ∧
  (x^2 + y^2 + 16 / (x + y)^2 = 4 * Real.sqrt 2 ↔ x = y ∧ x = (Real.sqrt 2)^(1/4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l108_10858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l108_10800

theorem division_problem (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : x % y = 6)
  (h2 : (x : ℝ) / (y : ℝ) = 96.15) : 
  y = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l108_10800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_approx_l108_10847

noncomputable section

-- Define the segments of the car's journey
def segment1_speed : ℝ := 45
def segment1_distance : ℝ := 15
def segment2_speed : ℝ := 55
def segment2_distance : ℝ := 30
def segment3_speed : ℝ := 65
def segment3_time : ℝ := 35 / 60  -- Convert 35 minutes to hours
def segment4_speed : ℝ := 52
def segment4_time : ℝ := 20 / 60  -- Convert 20 minutes to hours

-- Calculate total distance and time
def total_distance : ℝ := segment1_distance + segment2_distance + 
  (segment3_speed * segment3_time) + (segment4_speed * segment4_time)

def total_time : ℝ := (segment1_distance / segment1_speed) + 
  (segment2_distance / segment2_speed) + segment3_time + segment4_time

-- Define the average speed
def average_speed : ℝ := total_distance / total_time

end noncomputable section

-- Theorem statement
theorem car_average_speed_approx : 
  ∃ ε > 0, |average_speed - 55.85| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_approx_l108_10847
