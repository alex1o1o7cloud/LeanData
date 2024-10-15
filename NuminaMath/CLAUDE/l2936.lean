import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l2936_293655

/-- The ratio of the area of a square inscribed in a quarter-circle to the area of a square inscribed in a full circle, both with radius r -/
theorem inscribed_squares_area_ratio (r : ℝ) (hr : r > 0) :
  ∃ (s₁ s₂ : ℝ),
    s₁ > 0 ∧ s₂ > 0 ∧
    s₁^2 + (s₁/2)^2 = r^2 ∧  -- Square inscribed in quarter-circle
    s₂^2 = 2*r^2 ∧           -- Square inscribed in full circle
    s₁^2 / s₂^2 = 2/5 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l2936_293655


namespace NUMINAMATH_CALUDE_farm_field_area_l2936_293672

/-- Represents the farm field ploughing problem -/
structure FarmField where
  plannedDailyArea : ℕ  -- Planned area to plough per day
  actualDailyArea : ℕ   -- Actual area ploughed per day
  extraDays : ℕ         -- Extra days needed
  remainingArea : ℕ     -- Area left to plough

/-- Calculates the total area of the farm field -/
def totalArea (f : FarmField) : ℕ :=
  f.plannedDailyArea * ((f.actualDailyArea * (f.extraDays + 3) + f.remainingArea) / f.plannedDailyArea)

/-- Theorem stating that the total area of the given farm field is 480 hectares -/
theorem farm_field_area (f : FarmField) 
    (h1 : f.plannedDailyArea = 160)
    (h2 : f.actualDailyArea = 85)
    (h3 : f.extraDays = 2)
    (h4 : f.remainingArea = 40) : 
  totalArea f = 480 := by
  sorry

#eval totalArea { plannedDailyArea := 160, actualDailyArea := 85, extraDays := 2, remainingArea := 40 }

end NUMINAMATH_CALUDE_farm_field_area_l2936_293672


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_negative_four_squared_l2936_293631

theorem arithmetic_square_root_of_negative_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_negative_four_squared_l2936_293631


namespace NUMINAMATH_CALUDE_range_of_alpha_minus_beta_l2936_293600

theorem range_of_alpha_minus_beta (α β : Real) 
  (h1 : -π ≤ α) (h2 : α ≤ β) (h3 : β ≤ π/2) :
  ∀ x, x ∈ Set.Icc (-3*π/2) 0 ↔ ∃ α' β', 
    -π ≤ α' ∧ α' ≤ β' ∧ β' ≤ π/2 ∧ x = α' - β' :=
by sorry

end NUMINAMATH_CALUDE_range_of_alpha_minus_beta_l2936_293600


namespace NUMINAMATH_CALUDE_yellow_shirts_calculation_l2936_293664

/-- The number of yellow shirts in each pack -/
def yellow_shirts_per_pack : ℕ :=
  let black_packs : ℕ := 3
  let yellow_packs : ℕ := 3
  let black_shirts_per_pack : ℕ := 5
  let total_shirts : ℕ := 21
  (total_shirts - black_packs * black_shirts_per_pack) / yellow_packs

theorem yellow_shirts_calculation :
  yellow_shirts_per_pack = 2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_shirts_calculation_l2936_293664


namespace NUMINAMATH_CALUDE_planes_parallel_l2936_293619

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (inPlane : Line → Plane → Prop)
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem planes_parallel (α β : Plane) (a b : Line) :
  (α ≠ β) →
  (perpendicular a α ∧ perpendicular a β) ∨
  (inPlane a α ∧ inPlane b β ∧ parallel a β ∧ parallel b α ∧ skew a b) →
  parallel a β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_l2936_293619


namespace NUMINAMATH_CALUDE_range_of_a_given_quadratic_inequality_l2936_293650

theorem range_of_a_given_quadratic_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_given_quadratic_inequality_l2936_293650


namespace NUMINAMATH_CALUDE_thirty_percent_of_two_hundred_l2936_293603

theorem thirty_percent_of_two_hundred : (30 / 100) * 200 = 60 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_of_two_hundred_l2936_293603


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2936_293682

theorem quadratic_inequality (x : ℝ) : -8 * x^2 + 6 * x - 1 < 0 ↔ 0.25 < x ∧ x < 0.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2936_293682


namespace NUMINAMATH_CALUDE_sales_tax_difference_l2936_293696

theorem sales_tax_difference (price : ℝ) (tax_rate1 : ℝ) (tax_rate2 : ℝ) : 
  price = 50 → tax_rate1 = 0.075 → tax_rate2 = 0.065 → 
  price * tax_rate1 - price * tax_rate2 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_difference_l2936_293696


namespace NUMINAMATH_CALUDE_quadrilateral_is_rectangle_l2936_293622

/-- A quadrilateral in the complex plane -/
structure ComplexQuadrilateral where
  z₁ : ℂ
  z₂ : ℂ
  z₃ : ℂ
  z₄ : ℂ

/-- Predicate to check if a complex number has unit modulus -/
def hasUnitModulus (z : ℂ) : Prop := Complex.abs z = 1

/-- Predicate to check if a ComplexQuadrilateral is a rectangle -/
def isRectangle (q : ComplexQuadrilateral) : Prop :=
  -- Define what it means for a quadrilateral to be a rectangle
  -- This is a placeholder and should be properly defined
  True

/-- Main theorem: Under given conditions, the quadrilateral is a rectangle -/
theorem quadrilateral_is_rectangle (q : ComplexQuadrilateral) 
  (h₁ : hasUnitModulus q.z₁)
  (h₂ : hasUnitModulus q.z₂)
  (h₃ : hasUnitModulus q.z₃)
  (h₄ : hasUnitModulus q.z₄)
  (h_sum : q.z₁ + q.z₂ + q.z₃ + q.z₄ = 0) :
  isRectangle q :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_is_rectangle_l2936_293622


namespace NUMINAMATH_CALUDE_flowers_in_vase_l2936_293630

theorem flowers_in_vase (initial_flowers : ℕ) (removed_flowers : ℕ) : 
  initial_flowers = 13 → removed_flowers = 7 → initial_flowers - removed_flowers = 6 := by
  sorry

end NUMINAMATH_CALUDE_flowers_in_vase_l2936_293630


namespace NUMINAMATH_CALUDE_composition_problem_l2936_293670

theorem composition_problem (c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x + c)
  (hg : ∀ x, g x = c * x + 3)
  (h_comp : ∀ x, f (g x) = 15 * x + d) :
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_composition_problem_l2936_293670


namespace NUMINAMATH_CALUDE_total_sodas_sold_l2936_293652

theorem total_sodas_sold (morning_sales afternoon_sales : ℕ) 
  (h1 : morning_sales = 77)
  (h2 : afternoon_sales = 19) :
  morning_sales + afternoon_sales = 96 := by
sorry

end NUMINAMATH_CALUDE_total_sodas_sold_l2936_293652


namespace NUMINAMATH_CALUDE_lg_17_not_uniquely_calculable_l2936_293606

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Given conditions
axiom lg_8 : lg 8 = 0.9031
axiom lg_9 : lg 9 = 0.9542

-- Define a proposition that lg 17 cannot be uniquely calculated
def lg_17_not_calculable : Prop :=
  ∀ f : ℝ → ℝ → ℝ, 
    (∀ x y : ℝ, f (lg x) (lg y) = lg (x + y)) → 
    ¬∃! z : ℝ, f (lg 8) (lg 9) = z ∧ z = lg 17

-- Theorem statement
theorem lg_17_not_uniquely_calculable : lg_17_not_calculable :=
sorry

end NUMINAMATH_CALUDE_lg_17_not_uniquely_calculable_l2936_293606


namespace NUMINAMATH_CALUDE_exists_synchronous_exp_sin_synchronous_log_square_implies_a_gt_2e_l2936_293645

/-- Definition of synchronous functions -/
def Synchronous (f g : ℝ → ℝ) (m n : ℝ) : Prop :=
  f m = g m ∧ f n = g n

/-- Statement for option B -/
theorem exists_synchronous_exp_sin :
  ∃ n : ℝ, 1/2 < n ∧ n < 1 ∧
  Synchronous (fun x ↦ Real.exp x - 1) (fun x ↦ Real.sin (π * x)) 0 n :=
sorry

/-- Statement for option C -/
theorem synchronous_log_square_implies_a_gt_2e (a : ℝ) :
  (∃ m n : ℝ, Synchronous (fun x ↦ a * Real.log x) (fun x ↦ x^2) m n) →
  a > 2 * Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_exists_synchronous_exp_sin_synchronous_log_square_implies_a_gt_2e_l2936_293645


namespace NUMINAMATH_CALUDE_jerry_softball_time_l2936_293613

theorem jerry_softball_time (
  num_daughters : ℕ)
  (games_per_daughter : ℕ)
  (practice_hours_per_game : ℕ)
  (game_duration : ℕ)
  (h1 : num_daughters = 4)
  (h2 : games_per_daughter = 12)
  (h3 : practice_hours_per_game = 6)
  (h4 : game_duration = 3) :
  num_daughters * games_per_daughter * (practice_hours_per_game + game_duration) = 432 :=
by sorry

end NUMINAMATH_CALUDE_jerry_softball_time_l2936_293613


namespace NUMINAMATH_CALUDE_sphere_volume_area_ratio_l2936_293635

theorem sphere_volume_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 8 / 27 →
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_area_ratio_l2936_293635


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l2936_293648

def k : ℕ := 2017^2 + 2^2017

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := k) : (k^2 + 2^k) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l2936_293648


namespace NUMINAMATH_CALUDE_seed_mixture_weights_l2936_293609

/-- Represents a seed mixture with percentages of different grass types -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ
  clover : ℝ
  sum_to_100 : ryegrass + bluegrass + fescue + clover = 100

/-- The final mixture of seeds -/
def FinalMixture (x y z : ℝ) : SeedMixture :=
  { ryegrass := 35
    bluegrass := 30
    fescue := 35
    clover := 0
    sum_to_100 := by norm_num }

/-- The seed mixtures X, Y, and Z -/
def X : SeedMixture :=
  { ryegrass := 40
    bluegrass := 50
    fescue := 0
    clover := 10
    sum_to_100 := by norm_num }

def Y : SeedMixture :=
  { ryegrass := 25
    bluegrass := 0
    fescue := 70
    clover := 5
    sum_to_100 := by norm_num }

def Z : SeedMixture :=
  { ryegrass := 30
    bluegrass := 20
    fescue := 50
    clover := 0
    sum_to_100 := by norm_num }

/-- The theorem stating the weights of seed mixtures X, Y, and Z in the final mixture -/
theorem seed_mixture_weights (x y z : ℝ) 
  (h_total : x + y + z = 8)
  (h_ratio : x / 3 = y / 2 ∧ x / 3 = z / 3)
  (h_final : FinalMixture x y z = 
    { ryegrass := (X.ryegrass * x + Y.ryegrass * y + Z.ryegrass * z) / 8
      bluegrass := (X.bluegrass * x + Y.bluegrass * y + Z.bluegrass * z) / 8
      fescue := (X.fescue * x + Y.fescue * y + Z.fescue * z) / 8
      clover := (X.clover * x + Y.clover * y + Z.clover * z) / 8
      sum_to_100 := sorry }) :
  x = 3 ∧ y = 2 ∧ z = 3 := by
  sorry

end NUMINAMATH_CALUDE_seed_mixture_weights_l2936_293609


namespace NUMINAMATH_CALUDE_expression_simplification_l2936_293602

theorem expression_simplification (a c x z : ℝ) :
  (c * x * (a^3 * x^3 + 3 * a^3 * z^3 + c^3 * z^3) + a * z * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (c * x + a * z) = 
  a^3 * x^3 + 3 * a^3 * z^3 + c^3 * z^3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2936_293602


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2936_293636

theorem necessary_but_not_sufficient (A B C : Set α) (h : ∀ a, a ∈ A ↔ (a ∈ B ∧ a ∈ C)) :
  (∀ a, a ∈ A → a ∈ B) ∧ ¬(∀ a, a ∈ B → a ∈ A) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2936_293636


namespace NUMINAMATH_CALUDE_unicorn_tower_rope_length_l2936_293673

/-- Represents the setup of the unicorn and tower problem -/
structure UnicornTowerSetup where
  towerRadius : ℝ
  ropeLength : ℝ
  ropeAngle : ℝ
  ropeDistanceFromTower : ℝ
  unicornHeight : ℝ

/-- Calculates the length of rope touching the tower given the problem setup -/
def ropeTouchingTowerLength (setup : UnicornTowerSetup) : ℝ :=
  sorry

/-- Theorem stating the length of rope touching the tower in the given setup -/
theorem unicorn_tower_rope_length :
  let setup : UnicornTowerSetup := {
    towerRadius := 5,
    ropeLength := 30,
    ropeAngle := 30 * Real.pi / 180,  -- Convert to radians
    ropeDistanceFromTower := 5,
    unicornHeight := 5
  }
  ∃ (ε : ℝ), abs (ropeTouchingTowerLength setup - 19.06) < ε ∧ ε > 0 :=
sorry

end NUMINAMATH_CALUDE_unicorn_tower_rope_length_l2936_293673


namespace NUMINAMATH_CALUDE_max_money_collectible_l2936_293687

-- Define the structure of the land plot
structure LandPlot where
  circles : Fin 36 → ℕ
  -- circles represents the amount of money in each of the 36 circles

-- Define the concept of a valid path
def ValidPath (plot : LandPlot) (path : List (Fin 36)) : Prop :=
  -- A path is valid if it doesn't pass twice along the same straight line
  -- The actual implementation of this condition is complex and omitted here
  sorry

-- Define the sum of money collected along a path
def PathSum (plot : LandPlot) (path : List (Fin 36)) : ℕ :=
  path.map plot.circles |> List.sum

-- The main theorem
theorem max_money_collectible (plot : LandPlot) : 
  (∃ (path : List (Fin 36)), ValidPath plot path ∧ PathSum plot path = 47) ∧
  (∀ (path : List (Fin 36)), ValidPath plot path → PathSum plot path ≤ 47) := by
  sorry

end NUMINAMATH_CALUDE_max_money_collectible_l2936_293687


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_geq_arithmetic_mean_l2936_293685

theorem sqrt_sum_squares_geq_arithmetic_mean (a b : ℝ) :
  Real.sqrt ((a^2 + b^2) / 2) ≥ (a + b) / 2 ∧
  (Real.sqrt ((a^2 + b^2) / 2) = (a + b) / 2 ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_geq_arithmetic_mean_l2936_293685


namespace NUMINAMATH_CALUDE_tangent_at_negative_one_a_lower_bound_l2936_293667

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 1

-- Define the derivative of g
def g' (x : ℝ) : ℝ := 2*x

-- Define the condition for the tangent line
def tangent_condition (x₁ a : ℝ) : Prop :=
  ∃ x₂, f' x₁ = g' x₂ ∧ f x₁ + f' x₁ * (x₂ - x₁) = g a x₂

-- Theorem 1: When x₁ = -1, a = 3
theorem tangent_at_negative_one :
  tangent_condition (-1) 3 :=
sorry

-- Theorem 2: For all valid x₁, a ≥ -1
theorem a_lower_bound :
  ∀ x₁ a : ℝ, tangent_condition x₁ a → a ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_at_negative_one_a_lower_bound_l2936_293667


namespace NUMINAMATH_CALUDE_zip_code_relationship_l2936_293626

/-- 
Theorem: Given a sequence of five numbers A, B, C, D, and E satisfying certain conditions,
prove that the sum of the first two numbers (A + B) equals 2.
-/
theorem zip_code_relationship (A B C D E : ℕ) 
  (sum_condition : A + B + C + D + E = 10)
  (third_zero : C = 0)
  (fourth_double_first : D = 2 * A)
  (fourth_fifth_sum : D + E = 8) :
  A + B = 2 := by
  sorry

end NUMINAMATH_CALUDE_zip_code_relationship_l2936_293626


namespace NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l2936_293640

-- Problem 1
theorem ellipse_equation (x y : ℝ) :
  let equation := x^2 / 13 + y^2 / (13/9) = 1
  let center_at_origin := ∀ (t : ℝ), t^2 / 13 + 0^2 / (13/9) ≠ 1 ∧ 0^2 / 13 + t^2 / (13/9) ≠ 1
  let foci_on_x_axis := ∃ (c : ℝ), c^2 = 13 - 13/9 ∧ (c^2 / 13 + 0^2 / (13/9) = 1 ∨ (-c)^2 / 13 + 0^2 / (13/9) = 1)
  let major_axis_triple_minor := 13 = 3 * (13/9)
  let passes_through_p := 3^2 / 13 + 2^2 / (13/9) = 1
  center_at_origin ∧ foci_on_x_axis ∧ major_axis_triple_minor ∧ passes_through_p → equation :=
by sorry

-- Problem 2
theorem hyperbola_equation (x y : ℝ) :
  let equation := x^2 / 10 - y^2 / 6 = 1
  let common_asymptote := ∃ (k : ℝ), k^2 = 10/6 ∧ k^2 = 5/3
  let focal_length_8 := ∃ (c : ℝ), c^2 = 10 + 6 ∧ 2*c = 8
  common_asymptote ∧ focal_length_8 → equation :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l2936_293640


namespace NUMINAMATH_CALUDE_olaf_car_collection_l2936_293697

/-- The number of toy cars in Olaf's collection after receiving gifts from his family -/
def total_cars (initial : ℕ) (dad : ℕ) (auntie : ℕ) (uncle : ℕ) : ℕ :=
  let mum := dad + 5
  let grandpa := 2 * uncle
  initial + dad + mum + auntie + uncle + grandpa

/-- Theorem stating the total number of cars in Olaf's collection -/
theorem olaf_car_collection : 
  total_cars 150 10 6 5 = 196 := by
  sorry

end NUMINAMATH_CALUDE_olaf_car_collection_l2936_293697


namespace NUMINAMATH_CALUDE_unique_solution_l2936_293639

/-- The system of equations has a unique solution at (-2, -4) -/
theorem unique_solution : ∃! (x y : ℝ), 
  (x + 3*y + 14 ≤ 0) ∧ 
  (x^4 + 2*x^2*y^2 + y^4 + 64 - 20*x^2 - 20*y^2 = 8*x*y) ∧
  (x = -2) ∧ (y = -4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2936_293639


namespace NUMINAMATH_CALUDE_project_hours_l2936_293662

theorem project_hours (kate mark pat : ℕ) 
  (h1 : pat = 2 * kate)
  (h2 : pat * 3 = mark)
  (h3 : mark = kate + 80) :
  kate + mark + pat = 144 := by
sorry

end NUMINAMATH_CALUDE_project_hours_l2936_293662


namespace NUMINAMATH_CALUDE_existence_of_special_number_l2936_293625

def small_number (n : ℕ) : Prop := n ≤ 150

theorem existence_of_special_number :
  ∃ (N : ℕ) (a b : ℕ), 
    small_number a ∧ 
    small_number b ∧ 
    b = a + 1 ∧
    ¬(N % a = 0) ∧
    ¬(N % b = 0) ∧
    (∀ (k : ℕ), small_number k → k ≠ a → k ≠ b → N % k = 0) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l2936_293625


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2936_293699

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  a = Real.sqrt 6 →
  b = 2 →
  B = 45 * π / 180 →
  Real.tan A * Real.tan C > 1 →
  A + B + C = π →
  (a / Real.sin A = b / Real.sin B) →
  C = 75 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2936_293699


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2936_293643

theorem trigonometric_inequality (x : ℝ) :
  (-1/4 : ℝ) ≤ 5 * (Real.cos x)^2 - 5 * (Real.cos x)^4 + 5 * Real.sin x * Real.cos x + 1 ∧
  5 * (Real.cos x)^2 - 5 * (Real.cos x)^4 + 5 * Real.sin x * Real.cos x + 1 ≤ (19/4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2936_293643


namespace NUMINAMATH_CALUDE_fence_poles_count_l2936_293612

theorem fence_poles_count (total_length bridge_length pole_spacing : ℕ) 
  (h1 : total_length = 900)
  (h2 : bridge_length = 42)
  (h3 : pole_spacing = 6) : 
  (2 * ((total_length - bridge_length) / pole_spacing)) = 286 := by
  sorry

end NUMINAMATH_CALUDE_fence_poles_count_l2936_293612


namespace NUMINAMATH_CALUDE_replaced_person_age_l2936_293684

theorem replaced_person_age 
  (n : ℕ) 
  (original_avg : ℝ) 
  (new_avg : ℝ) 
  (new_person_age : ℝ) 
  (h1 : n = 10)
  (h2 : original_avg = new_avg + 3)
  (h3 : new_person_age = 12) : 
  n * original_avg - (n * new_avg + new_person_age) = 18 := by
sorry

end NUMINAMATH_CALUDE_replaced_person_age_l2936_293684


namespace NUMINAMATH_CALUDE_x_minus_y_value_l2936_293614

theorem x_minus_y_value (x y : ℤ) (h1 : x + y = 4) (h2 : x = 20) : x - y = 36 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l2936_293614


namespace NUMINAMATH_CALUDE_greatest_b_value_l2936_293620

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 7*x - 10 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 7*5 - 10 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l2936_293620


namespace NUMINAMATH_CALUDE_abc_inequality_l2936_293618

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (b + c - a) * (a + c - b) * (a + b - c) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2936_293618


namespace NUMINAMATH_CALUDE_parabola_focus_l2936_293677

/-- A parabola with equation y^2 = 2px (p > 0) and directrix x = -1 has its focus at (1, 0) -/
theorem parabola_focus (p : ℝ) (h : p > 0) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let directrix := {(x, y) : ℝ × ℝ | x = -1}
  let focus := (1, 0)
  (∀ (point : ℝ × ℝ), point ∈ parabola ↔ 
    Real.sqrt ((point.1 - focus.1)^2 + (point.2 - focus.2)^2) = 
    |point.1 - (-1)|) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l2936_293677


namespace NUMINAMATH_CALUDE_final_value_less_than_original_l2936_293665

theorem final_value_less_than_original (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_upper : q < 50) (hM : M > 0) :
  M * (1 - p / 100) * (1 + q / 100) < M ↔ p > (100 * q - q^2) / 100 := by
  sorry

end NUMINAMATH_CALUDE_final_value_less_than_original_l2936_293665


namespace NUMINAMATH_CALUDE_number_puzzle_l2936_293668

theorem number_puzzle (x : ℚ) : 
  (((5 * x - (1/3) * (5 * x)) / 10) + (1/3) * x + (1/2) * x + (1/4) * x) = 68 → x = 48 := by
sorry

end NUMINAMATH_CALUDE_number_puzzle_l2936_293668


namespace NUMINAMATH_CALUDE_white_more_probable_l2936_293678

def yellow_balls : ℕ := 3
def white_balls : ℕ := 5
def total_balls : ℕ := yellow_balls + white_balls

def prob_yellow : ℚ := yellow_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

theorem white_more_probable : prob_white > prob_yellow := by
  sorry

end NUMINAMATH_CALUDE_white_more_probable_l2936_293678


namespace NUMINAMATH_CALUDE_correct_statements_count_l2936_293607

/-- Represents a programming statement --/
inductive Statement
  | Output (cmd : String) (vars : List String)
  | Input (var : String) (value : String)
  | Assignment (lhs : String) (rhs : String)

/-- Checks if a statement is correct --/
def is_correct (s : Statement) : Bool :=
  match s with
  | Statement.Output cmd vars => cmd = "PRINT"
  | Statement.Input var value => true  -- Simplified for this problem
  | Statement.Assignment lhs rhs => true  -- Simplified for this problem

/-- The list of statements to evaluate --/
def statements : List Statement :=
  [ Statement.Output "INPUT" ["a", "b", "c"]
  , Statement.Input "x" "3"
  , Statement.Assignment "3" "A"
  , Statement.Assignment "A" "B=C"
  ]

/-- Counts the number of correct statements --/
def count_correct (stmts : List Statement) : Nat :=
  stmts.filter is_correct |>.length

theorem correct_statements_count :
  count_correct statements = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_count_l2936_293607


namespace NUMINAMATH_CALUDE_middle_card_first_round_l2936_293669

/-- Represents a card with a positive integer value -/
structure Card where
  value : ℕ+
  
/-- Represents a player in the game -/
structure Player where
  totalCounters : ℕ
  lastRoundCard : Card

/-- Represents the game state -/
structure GameState where
  cards : Fin 3 → Card
  players : Fin 3 → Player
  rounds : ℕ

/-- Conditions of the game -/
def gameConditions (g : GameState) : Prop :=
  g.rounds ≥ 2 ∧
  (g.cards 0).value < (g.cards 1).value ∧ (g.cards 1).value < (g.cards 2).value ∧
  (g.players 0).totalCounters + (g.players 1).totalCounters + (g.players 2).totalCounters = 39 ∧
  (∃ i j k : Fin 3, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (g.players i).totalCounters = 20 ∧
    (g.players j).totalCounters = 10 ∧
    (g.players k).totalCounters = 9) ∧
  (∃ i : Fin 3, (g.players i).totalCounters = 10 ∧
    (g.players i).lastRoundCard = g.cards 2)

/-- The theorem to be proved -/
theorem middle_card_first_round (g : GameState) :
  gameConditions g →
  ∃ i : Fin 3, (g.players i).totalCounters = 9 ∧
    (∃ firstRoundCard : Card, firstRoundCard = g.cards 1) :=
sorry

end NUMINAMATH_CALUDE_middle_card_first_round_l2936_293669


namespace NUMINAMATH_CALUDE_school_teachers_count_l2936_293615

theorem school_teachers_count 
  (total : ℕ) 
  (sample_size : ℕ) 
  (sample_students : ℕ) 
  (h1 : total = 2400)
  (h2 : sample_size = 120)
  (h3 : sample_students = 110)
  (h4 : sample_size ≤ total)
  (h5 : sample_students < sample_size) :
  (sample_size - sample_students) * total / sample_size = 200 := by
sorry

end NUMINAMATH_CALUDE_school_teachers_count_l2936_293615


namespace NUMINAMATH_CALUDE_right_triangle_increase_sides_acute_l2936_293629

/-- Given a right-angled triangle, increasing all sides by the same amount results in an acute-angled triangle -/
theorem right_triangle_increase_sides_acute (a b c k : ℝ) 
  (h_right : a^2 + b^2 = c^2) -- Original triangle is right-angled
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ k > 0) -- Sides and increase are positive
  : (a + k)^2 + (b + k)^2 > (c + k)^2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_increase_sides_acute_l2936_293629


namespace NUMINAMATH_CALUDE_number_of_pupils_l2936_293610

theorem number_of_pupils (total_people : ℕ) (parents : ℕ) (pupils : ℕ) : 
  total_people = 676 → parents = 22 → pupils = total_people - parents → pupils = 654 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_l2936_293610


namespace NUMINAMATH_CALUDE_complex_number_with_sqrt3_imaginary_and_modulus_2_l2936_293679

theorem complex_number_with_sqrt3_imaginary_and_modulus_2 :
  ∀ z : ℂ, (z.im = Real.sqrt 3) → (Complex.abs z = 2) →
  (z = Complex.mk 1 (Real.sqrt 3) ∨ z = Complex.mk (-1) (Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_with_sqrt3_imaginary_and_modulus_2_l2936_293679


namespace NUMINAMATH_CALUDE_students_allowance_l2936_293647

theorem students_allowance (allowance : ℚ) : 
  (allowance > 0) →
  (3/5 * allowance + 1/3 * (2/5 * allowance) + 2/5) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_students_allowance_l2936_293647


namespace NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_prime_factors_2310_l2936_293688

def sum_of_digits (n : ℕ) : ℕ := sorry

def prime_factors (n : ℕ) : List ℕ := sorry

theorem sum_of_digits_of_sum_of_prime_factors_2310 : 
  sum_of_digits (List.sum (prime_factors 2310)) = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_prime_factors_2310_l2936_293688


namespace NUMINAMATH_CALUDE_set_relation_proof_l2936_293656

theorem set_relation_proof (M P : Set α) (h_nonempty : M.Nonempty) 
  (h_not_subset : ¬(M ⊆ P)) : 
  (∃ x ∈ M, x ∉ P) ∧ ¬(∀ x ∈ M, x ∈ P) := by
  sorry

end NUMINAMATH_CALUDE_set_relation_proof_l2936_293656


namespace NUMINAMATH_CALUDE_trophy_count_proof_l2936_293621

/-- The number of trophies Michael has right now -/
def michael_trophies : ℕ := 30

/-- The number of trophies Jack will have in three years -/
def jack_future_trophies : ℕ := 10 * michael_trophies

/-- The number of trophies Michael will have in three years -/
def michael_future_trophies : ℕ := michael_trophies + 100

theorem trophy_count_proof :
  michael_trophies = 30 ∧
  jack_future_trophies = 10 * michael_trophies ∧
  michael_future_trophies = michael_trophies + 100 ∧
  jack_future_trophies + michael_future_trophies = 430 :=
by sorry

end NUMINAMATH_CALUDE_trophy_count_proof_l2936_293621


namespace NUMINAMATH_CALUDE_square_stack_sums_l2936_293695

theorem square_stack_sums : 
  (¬ ∃ n : ℕ+, 10 * n = 8016) ∧ 
  (∃ n : ℕ+, 10 * n = 8020) := by
  sorry

end NUMINAMATH_CALUDE_square_stack_sums_l2936_293695


namespace NUMINAMATH_CALUDE_first_podcast_length_l2936_293634

/-- Given a 6-hour drive and podcast lengths, prove the first podcast is 0.75 hours long -/
theorem first_podcast_length (total_time : ℝ) (podcast1 : ℝ) (podcast2 : ℝ) (podcast3 : ℝ) (podcast4 : ℝ) (podcast5 : ℝ) :
  total_time = 6 →
  podcast2 = 2 * podcast1 →
  podcast3 = 1.75 →
  podcast4 = 1 →
  podcast5 = 1 →
  podcast1 + podcast2 + podcast3 + podcast4 + podcast5 = total_time →
  podcast1 = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_first_podcast_length_l2936_293634


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l2936_293693

/-- Given a quadratic equation k^2x^2 + (4k-1)x + 4 = 0 with two distinct real roots,
    the range of values for k is k < 1/8 and k ≠ 0 -/
theorem quadratic_equation_roots_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    k^2 * x₁^2 + (4*k - 1) * x₁ + 4 = 0 ∧
    k^2 * x₂^2 + (4*k - 1) * x₂ + 4 = 0) →
  k < 1/8 ∧ k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_condition_l2936_293693


namespace NUMINAMATH_CALUDE_min_y_squared_l2936_293641

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  EF : ℝ
  GH : ℝ
  y : ℝ
  is_isosceles : EF > GH
  circle_tangent : True  -- Represents the condition about the tangent circle

/-- The theorem stating the minimum value of y^2 -/
theorem min_y_squared (t : IsoscelesTrapezoid) 
  (h1 : t.EF = 72) 
  (h2 : t.GH = 45) : 
  ∃ (n : ℝ), n^2 = 486 ∧ ∀ (y : ℝ), 
    (∃ (t' : IsoscelesTrapezoid), t'.y = y ∧ t'.EF = t.EF ∧ t'.GH = t.GH) → 
    y^2 ≥ n^2 :=
sorry

end NUMINAMATH_CALUDE_min_y_squared_l2936_293641


namespace NUMINAMATH_CALUDE_money_distribution_l2936_293633

theorem money_distribution (total : ℝ) (a b c d : ℝ) :
  a + b + c + d = total →
  a = (5 / 14) * total →
  b = (2 / 14) * total →
  c = (4 / 14) * total →
  d = (3 / 14) * total →
  c = d + 500 →
  d = 1500 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2936_293633


namespace NUMINAMATH_CALUDE_roots_equation_q_value_l2936_293642

theorem roots_equation_q_value (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 16/3 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_q_value_l2936_293642


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_repeated_sixes_and_seven_l2936_293666

def digits_of_ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

def digits_of_fours (n : ℕ) : ℕ :=
  4 * digits_of_ones n * (10^n) + 4 * digits_of_ones n

theorem sqrt_expression_equals_repeated_sixes_and_seven (n : ℕ) :
  let a := digits_of_ones n
  let fours := digits_of_fours n
  let ones := 10 * a + 1
  let sixes := 6 * a
  Real.sqrt (fours / (2 * n * (1/4)) + ones - sixes) = 6 * a + 1 :=
sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_repeated_sixes_and_seven_l2936_293666


namespace NUMINAMATH_CALUDE_jason_fire_frequency_l2936_293683

/-- Given the conditions of Jason's gameplay in Duty for Ashes, prove that he fires his weapon every 15 seconds on average. -/
theorem jason_fire_frequency
  (flame_duration : ℕ)
  (total_flame_time : ℕ)
  (seconds_per_minute : ℕ)
  (h1 : flame_duration = 5)
  (h2 : total_flame_time = 20)
  (h3 : seconds_per_minute = 60) :
  (seconds_per_minute : ℚ) / ((total_flame_time : ℚ) / (flame_duration : ℚ)) = 15 := by
  sorry

#check jason_fire_frequency

end NUMINAMATH_CALUDE_jason_fire_frequency_l2936_293683


namespace NUMINAMATH_CALUDE_kangaroo_hop_distance_l2936_293674

theorem kangaroo_hop_distance :
  let a : ℚ := 1/2  -- first term
  let r : ℚ := 3/4  -- common ratio
  let n : ℕ := 7    -- number of terms
  (a * (1 - r^n) / (1 - r) : ℚ) = 14297/2048 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_hop_distance_l2936_293674


namespace NUMINAMATH_CALUDE_parallel_tangents_intersection_l2936_293658

theorem parallel_tangents_intersection (x₀ : ℝ) : 
  (2 * x₀ = -3 * x₀^2) → (x₀ = 0 ∨ x₀ = -2/3) := by sorry

end NUMINAMATH_CALUDE_parallel_tangents_intersection_l2936_293658


namespace NUMINAMATH_CALUDE_final_savings_calculation_l2936_293659

/-- Calculates the final savings after a given period --/
def calculateFinalSavings (initialSavings : ℕ) (monthlyIncome : ℕ) (monthlyExpenses : ℕ) (months : ℕ) : ℕ :=
  initialSavings + months * monthlyIncome - months * monthlyExpenses

/-- Theorem stating that the final savings will be 1106900 rubles --/
theorem final_savings_calculation :
  let initialSavings : ℕ := 849400
  let monthlyIncome : ℕ := 45000 + 35000 + 7000 + 10000 + 13000
  let monthlyExpenses : ℕ := 30000 + 10000 + 5000 + 4500 + 9000
  let months : ℕ := 5
  calculateFinalSavings initialSavings monthlyIncome monthlyExpenses months = 1106900 := by
  sorry

#eval calculateFinalSavings 849400 (45000 + 35000 + 7000 + 10000 + 13000) (30000 + 10000 + 5000 + 4500 + 9000) 5

end NUMINAMATH_CALUDE_final_savings_calculation_l2936_293659


namespace NUMINAMATH_CALUDE_wages_decrease_percentage_l2936_293691

theorem wages_decrease_percentage (W : ℝ) (x : ℝ) 
  (h1 : W > 0)  -- Wages are positive
  (h2 : 0 ≤ x ∧ x ≤ 100)  -- Percentage decrease is between 0 and 100
  (h3 : 0.30 * (W * (1 - x / 100)) = 1.80 * (0.15 * W)) :  -- Condition from the problem
  x = 10 := by sorry

end NUMINAMATH_CALUDE_wages_decrease_percentage_l2936_293691


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_range_l2936_293632

/-- A triangle with side lengths a, b, and c is obtuse if and only if a² + b² < c² for some permutation of its sides. --/
def IsObtuseTriangle (a b c : ℝ) : Prop :=
  (a^2 + b^2 < c^2) ∨ (a^2 + c^2 < b^2) ∨ (b^2 + c^2 < a^2)

/-- The range of possible values for the third side of an obtuse triangle with two sides of length 3 and 4. --/
theorem obtuse_triangle_side_range :
  ∀ x : ℝ, IsObtuseTriangle 3 4 x ↔ (5 < x ∧ x < 7) ∨ (1 < x ∧ x < Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_range_l2936_293632


namespace NUMINAMATH_CALUDE_vacation_payment_difference_l2936_293623

/-- Represents the vacation expenses and payments for four people. -/
structure VacationExpenses where
  tom_paid : ℕ
  dorothy_paid : ℕ
  sammy_paid : ℕ
  nancy_paid : ℕ
  total_cost : ℕ
  equal_share : ℕ

/-- The given vacation expenses. -/
def given_expenses : VacationExpenses := {
  tom_paid := 150,
  dorothy_paid := 190,
  sammy_paid := 240,
  nancy_paid := 320,
  total_cost := 900,
  equal_share := 225
}

/-- Theorem stating the difference between Tom's and Dorothy's additional payments. -/
theorem vacation_payment_difference (e : VacationExpenses) 
  (h1 : e.total_cost = e.tom_paid + e.dorothy_paid + e.sammy_paid + e.nancy_paid)
  (h2 : e.equal_share = e.total_cost / 4)
  (h3 : e = given_expenses) :
  (e.equal_share - e.tom_paid) - (e.equal_share - e.dorothy_paid) = 40 := by
  sorry

end NUMINAMATH_CALUDE_vacation_payment_difference_l2936_293623


namespace NUMINAMATH_CALUDE_apples_on_tree_l2936_293686

/-- Represents the number of apples in various states -/
structure AppleCount where
  onTree : ℕ
  onGround : ℕ
  eatenByDog : ℕ
  remaining : ℕ

/-- The theorem stating the number of apples on the tree -/
theorem apples_on_tree (a : AppleCount) 
  (h1 : a.onGround = 8)
  (h2 : a.eatenByDog = 3)
  (h3 : a.remaining = 10)
  (h4 : a.onGround = a.remaining + a.eatenByDog) :
  a.onTree = 5 := by
  sorry


end NUMINAMATH_CALUDE_apples_on_tree_l2936_293686


namespace NUMINAMATH_CALUDE_annual_price_decrease_l2936_293646

def price_2001 : ℝ := 1950
def price_2009 : ℝ := 1670
def year_2001 : ℕ := 2001
def year_2009 : ℕ := 2009

theorem annual_price_decrease :
  (price_2001 - price_2009) / (year_2009 - year_2001 : ℝ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_annual_price_decrease_l2936_293646


namespace NUMINAMATH_CALUDE_five_thursdays_in_august_l2936_293676

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the occurrences of a specific day in a month -/
def countDayInMonth (startDay : DayOfWeek) (daysInMonth : Nat) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Theorem: If July has five Tuesdays and both July and August have 31 days, 
    then Thursday must occur five times in August of the same year -/
theorem five_thursdays_in_august 
  (july_start : DayOfWeek) 
  (h1 : countDayInMonth july_start 31 DayOfWeek.Tuesday = 5) 
  : ∃ (august_start : DayOfWeek), 
    countDayInMonth august_start 31 DayOfWeek.Thursday = 5 :=
  sorry

end NUMINAMATH_CALUDE_five_thursdays_in_august_l2936_293676


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2936_293653

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2936_293653


namespace NUMINAMATH_CALUDE_james_tylenol_intake_l2936_293601

/-- Calculates the total milligrams of Tylenol taken in a day -/
def tylenolPerDay (tabletsPerDose : ℕ) (mgPerTablet : ℕ) (hoursPerDose : ℕ) (hoursPerDay : ℕ) : ℕ :=
  let dosesPerDay := hoursPerDay / hoursPerDose
  let mgPerDose := tabletsPerDose * mgPerTablet
  dosesPerDay * mgPerDose

/-- Proves that James takes 3000 mg of Tylenol per day -/
theorem james_tylenol_intake :
  tylenolPerDay 2 375 6 24 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_james_tylenol_intake_l2936_293601


namespace NUMINAMATH_CALUDE_red_balls_count_l2936_293692

theorem red_balls_count (total_balls : ℕ) (red_prob : ℚ) (red_balls : ℕ) : 
  total_balls = 15 → red_prob = 1/3 → red_balls = (red_prob * total_balls).num → red_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2936_293692


namespace NUMINAMATH_CALUDE_total_sneaker_spending_l2936_293628

/-- Geoff's sneaker spending over three days -/
def sneaker_spending (day1_spend : ℝ) : ℝ :=
  let day2_spend := 4 * day1_spend * (1 - 0.1)  -- 4 times day1 with 10% discount
  let day3_spend := 5 * day1_spend * (1 + 0.08) -- 5 times day1 with 8% tax
  day1_spend + day2_spend + day3_spend

/-- Theorem: Geoff's total sneaker spending over three days is $600 -/
theorem total_sneaker_spending :
  sneaker_spending 60 = 600 := by sorry

end NUMINAMATH_CALUDE_total_sneaker_spending_l2936_293628


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2936_293671

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2936_293671


namespace NUMINAMATH_CALUDE_probability_is_75_1024_l2936_293657

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction of movement -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of moving in any direction -/
def directionProbability : ℚ := 1/4

/-- The starting point -/
def start : Point := ⟨0, 0⟩

/-- The target point -/
def target : Point := ⟨3, 3⟩

/-- The maximum number of steps allowed -/
def maxSteps : ℕ := 8

/-- Calculates the probability of reaching the target point from the start point
    in at most maxSteps steps -/
def probabilityToReachTarget (start : Point) (target : Point) (maxSteps : ℕ) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem probability_is_75_1024 :
  probabilityToReachTarget start target maxSteps = 75/1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_75_1024_l2936_293657


namespace NUMINAMATH_CALUDE_parallel_transitivity_l2936_293651

-- Define a type for lines in a plane
variable (Line : Type)

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem parallel_transitivity (l1 l2 l3 : Line) :
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 :=
by
  sorry

#check parallel_transitivity

end NUMINAMATH_CALUDE_parallel_transitivity_l2936_293651


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2936_293605

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2936_293605


namespace NUMINAMATH_CALUDE_rationalize_sum_l2936_293611

/-- Represents a fraction with a cube root in the denominator -/
structure CubeRootFraction where
  numerator : ℚ
  denominator : ℚ
  root : ℕ

/-- Represents a rationalized fraction with a cube root in the numerator -/
structure RationalizedFraction where
  A : ℤ
  B : ℕ
  C : ℕ

/-- Checks if a number is not divisible by the cube of any prime -/
def not_divisible_by_cube_of_prime (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^3 ∣ n)

/-- Rationalizes a fraction with a cube root in the denominator -/
def rationalize (f : CubeRootFraction) : RationalizedFraction :=
  sorry

theorem rationalize_sum (f : CubeRootFraction) 
  (h : f = { numerator := 2, denominator := 3, root := 7 }) :
  let r := rationalize f
  r.A + r.B + r.C = 72 ∧ 
  r.C > 0 ∧
  not_divisible_by_cube_of_prime r.B :=
sorry

end NUMINAMATH_CALUDE_rationalize_sum_l2936_293611


namespace NUMINAMATH_CALUDE_easter_egg_hunt_friends_l2936_293637

/-- Proves the number of friends at Shonda's Easter egg hunt --/
theorem easter_egg_hunt_friends (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_person : ℕ)
  (shonda_kids : ℕ) (shonda : ℕ) (other_adults : ℕ) :
  baskets = 15 →
  eggs_per_basket = 12 →
  eggs_per_person = 9 →
  shonda_kids = 2 →
  shonda = 1 →
  other_adults = 7 →
  baskets * eggs_per_basket / eggs_per_person - (shonda_kids + shonda + other_adults) = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_easter_egg_hunt_friends_l2936_293637


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l2936_293638

noncomputable def f (x : ℝ) : ℝ := x - Real.exp x

theorem tangent_parallel_to_x_axis :
  ∃ (p : ℝ × ℝ), 
    (∀ x : ℝ, (p.2 = f p.1) ∧ 
    (HasDerivAt f 0 p.1)) →
    p = (0, -1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l2936_293638


namespace NUMINAMATH_CALUDE_man_speed_man_speed_proof_l2936_293675

/-- The speed of a man relative to a train, given the train's length, speed, and time to cross the man. -/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  train_speed_ms - relative_speed

/-- Proof that the speed of the man is approximately 0.833 m/s given the specified conditions. -/
theorem man_speed_proof :
  let train_length : ℝ := 500
  let train_speed_kmh : ℝ := 63
  let crossing_time : ℝ := 29.997600191984642
  abs (man_speed train_length train_speed_kmh crossing_time - 0.833) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_man_speed_man_speed_proof_l2936_293675


namespace NUMINAMATH_CALUDE_expand_product_l2936_293604

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2936_293604


namespace NUMINAMATH_CALUDE_sixteen_factorial_digit_sum_l2936_293698

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem sixteen_factorial_digit_sum :
  ∃ (X Y : ℕ),
    X < 10 ∧ Y < 10 ∧
    factorial 16 = 2092200000000 + X * 100000000 + 208960000 + Y * 1000000 ∧
    X + Y = 7 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_factorial_digit_sum_l2936_293698


namespace NUMINAMATH_CALUDE_road_trip_equation_correct_l2936_293660

/-- Represents a road trip with a stop -/
structure RoadTrip where
  totalDistance : ℝ
  totalTime : ℝ
  stopDuration : ℝ
  speedBeforeStop : ℝ
  speedAfterStop : ℝ

/-- The equation for the road trip is correct -/
def correctEquation (trip : RoadTrip) (t : ℝ) : Prop :=
  trip.speedBeforeStop * t + trip.speedAfterStop * (trip.totalTime - trip.stopDuration / 60 - t) = trip.totalDistance

theorem road_trip_equation_correct (trip : RoadTrip) (t : ℝ) :
  trip.totalDistance = 300 ∧
  trip.totalTime = 4 ∧
  trip.stopDuration = 30 ∧
  trip.speedBeforeStop = 70 ∧
  trip.speedAfterStop = 90 →
  correctEquation trip t ↔ 70 * t + 90 * (3.5 - t) = 300 := by
  sorry


end NUMINAMATH_CALUDE_road_trip_equation_correct_l2936_293660


namespace NUMINAMATH_CALUDE_cost_2005_l2936_293680

/-- Represents the number of songs downloaded in 2004 -/
def songs_2004 : ℕ := 200

/-- Represents the number of songs downloaded in 2005 -/
def songs_2005 : ℕ := 360

/-- Represents the difference in cost per song between 2004 and 2005 in cents -/
def cost_difference : ℕ := 32

/-- Theorem stating that the cost of downloading 360 songs in 2005 was $144.00 -/
theorem cost_2005 (c : ℚ) : 
  (songs_2005 : ℚ) * c = (songs_2004 : ℚ) * (c + cost_difference) → 
  songs_2005 * c = 14400 := by
  sorry

end NUMINAMATH_CALUDE_cost_2005_l2936_293680


namespace NUMINAMATH_CALUDE_park_pathway_width_l2936_293608

/-- Represents a rectangular park with pathways -/
structure Park where
  length : ℝ
  width : ℝ
  lawn_area : ℝ

/-- Calculates the total width of all pathways in the park -/
def total_pathway_width (p : Park) : ℝ :=
  -- Define the function here, but don't implement it
  sorry

/-- Theorem stating the total pathway width for the given park specifications -/
theorem park_pathway_width :
  let p : Park := { length := 60, width := 40, lawn_area := 2109 }
  total_pathway_width p = 2.91 := by
  sorry

end NUMINAMATH_CALUDE_park_pathway_width_l2936_293608


namespace NUMINAMATH_CALUDE_jelly_beans_weight_l2936_293624

theorem jelly_beans_weight (initial_weight : ℝ) : 
  initial_weight > 0 →
  2 * (4 * initial_weight) = 16 →
  initial_weight = 2 := by
sorry

end NUMINAMATH_CALUDE_jelly_beans_weight_l2936_293624


namespace NUMINAMATH_CALUDE_peanut_butter_sandwich_days_l2936_293690

/-- Given:
  - There are 5 school days in a week
  - Karen packs ham sandwiches on 3 school days
  - Karen packs cake on one randomly chosen day
  - The probability of packing a ham sandwich and cake on the same day is 12%
  Prove that Karen packs peanut butter sandwiches on 2 days. -/
theorem peanut_butter_sandwich_days :
  ∀ (total_days ham_days cake_days : ℕ) 
    (prob_ham_and_cake : ℚ),
  total_days = 5 →
  ham_days = 3 →
  cake_days = 1 →
  prob_ham_and_cake = 12 / 100 →
  (ham_days : ℚ) / total_days * (cake_days : ℚ) / total_days = prob_ham_and_cake →
  total_days - ham_days = 2 :=
by sorry

end NUMINAMATH_CALUDE_peanut_butter_sandwich_days_l2936_293690


namespace NUMINAMATH_CALUDE_not_all_observed_values_yield_significant_regression_l2936_293616

/-- A set of observed values -/
structure ObservedValues where
  values : Set (ℝ × ℝ)

/-- A regression line equation -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Definition of representative significance for a regression line -/
def has_representative_significance (ov : ObservedValues) (rl : RegressionLine) : Prop :=
  sorry

/-- The theorem stating that not all sets of observed values yield a regression line with representative significance -/
theorem not_all_observed_values_yield_significant_regression :
  ¬ ∀ (ov : ObservedValues), ∃ (rl : RegressionLine), has_representative_significance ov rl :=
sorry

end NUMINAMATH_CALUDE_not_all_observed_values_yield_significant_regression_l2936_293616


namespace NUMINAMATH_CALUDE_maria_water_bottles_l2936_293661

/-- Calculates the final number of water bottles Maria has after a series of actions. -/
def final_bottle_count (initial : ℕ) (drunk : ℕ) (given_away : ℕ) (bought : ℕ) : ℕ :=
  initial - drunk - given_away + bought

/-- Theorem stating that Maria ends up with 71 bottles given the initial conditions and actions. -/
theorem maria_water_bottles : final_bottle_count 23 12 5 65 = 71 := by
  sorry

end NUMINAMATH_CALUDE_maria_water_bottles_l2936_293661


namespace NUMINAMATH_CALUDE_symmetric_about_x_axis_l2936_293689

-- Define the original function
def g (x : ℝ) : ℝ := x^2 - 3*x

-- Define the symmetric function
def f (x : ℝ) : ℝ := -x^2 + 3*x

-- Theorem statement
theorem symmetric_about_x_axis : 
  ∀ x y : ℝ, g x = y ↔ f x = -y :=
by sorry

end NUMINAMATH_CALUDE_symmetric_about_x_axis_l2936_293689


namespace NUMINAMATH_CALUDE_factory_conditional_probability_l2936_293649

/-- Represents the production data for a factory --/
structure FactoryData where
  total_parts : ℕ
  a_parts : ℕ
  a_qualified : ℕ
  b_parts : ℕ
  b_qualified : ℕ

/-- Calculates the conditional probability of a part being qualified given it was produced by A --/
def conditional_probability (data : FactoryData) : ℚ :=
  data.a_qualified / data.a_parts

/-- Theorem stating the conditional probability for the given problem --/
theorem factory_conditional_probability 
  (data : FactoryData)
  (h1 : data.total_parts = 100)
  (h2 : data.a_parts = 40)
  (h3 : data.a_qualified = 35)
  (h4 : data.b_parts = 60)
  (h5 : data.b_qualified = 50)
  (h6 : data.total_parts = data.a_parts + data.b_parts) :
  conditional_probability data = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_factory_conditional_probability_l2936_293649


namespace NUMINAMATH_CALUDE_rectangular_prism_max_volume_l2936_293627

/-- Given a rectangular prism with space diagonal 10 and orthogonal projection 8,
    its maximum volume is 192 -/
theorem rectangular_prism_max_volume :
  ∀ (a b h : ℝ),
  (a > 0) → (b > 0) → (h > 0) →
  (a^2 + b^2 + h^2 = 10^2) →
  (a^2 + b^2 = 8^2) →
  ∀ (V : ℝ), V = a * b * h →
  V ≤ 192 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_max_volume_l2936_293627


namespace NUMINAMATH_CALUDE_circle_line_distances_l2936_293644

/-- The maximum and minimum distances from a point on the circle x^2 + y^2 = 1 to the line x - 2y - 12 = 0 -/
theorem circle_line_distances :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | x - 2*y - 12 = 0}
  ∃ (max_dist min_dist : ℝ),
    (∀ p ∈ circle, ∀ q ∈ line, dist p q ≤ max_dist) ∧
    (∃ p ∈ circle, ∃ q ∈ line, dist p q = max_dist) ∧
    (∀ p ∈ circle, ∀ q ∈ line, dist p q ≥ min_dist) ∧
    (∃ p ∈ circle, ∃ q ∈ line, dist p q = min_dist) ∧
    max_dist = (12 * Real.sqrt 5) / 5 + 1 ∧
    min_dist = (12 * Real.sqrt 5) / 5 - 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distances_l2936_293644


namespace NUMINAMATH_CALUDE_red_ball_probability_l2936_293681

/-- Represents a container with red and green balls -/
structure Container where
  red : Nat
  green : Nat

/-- The probability of selecting a red ball from the given containers -/
def redBallProbability (x y z : Container) : Rat :=
  (1 / 3 : Rat) * (x.red / (x.red + x.green : Rat)) +
  (1 / 3 : Rat) * (y.red / (y.red + y.green : Rat)) +
  (1 / 3 : Rat) * (z.red / (z.red + z.green : Rat))

/-- Theorem stating the probability of selecting a red ball -/
theorem red_ball_probability :
  let x : Container := { red := 3, green := 7 }
  let y : Container := { red := 7, green := 3 }
  let z : Container := { red := 7, green := 3 }
  redBallProbability x y z = 17 / 30 := by
  sorry


end NUMINAMATH_CALUDE_red_ball_probability_l2936_293681


namespace NUMINAMATH_CALUDE_root_product_theorem_l2936_293654

-- Define the polynomial f(x)
def f (x : ℂ) : ℂ := x^6 + 2*x^3 + 1

-- Define the function h(x)
def h (x : ℂ) : ℂ := x^3 - 3*x

-- State the theorem
theorem root_product_theorem (y₁ y₂ y₃ y₄ y₅ y₆ : ℂ) 
  (hf₁ : f y₁ = 0) (hf₂ : f y₂ = 0) (hf₃ : f y₃ = 0)
  (hf₄ : f y₄ = 0) (hf₅ : f y₅ = 0) (hf₆ : f y₆ = 0) :
  (h y₁) * (h y₂) * (h y₃) * (h y₄) * (h y₅) * (h y₆) = 676 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l2936_293654


namespace NUMINAMATH_CALUDE_no_real_solution_nonzero_z_l2936_293617

theorem no_real_solution_nonzero_z (x y z : ℝ) : 
  x - y = 2 → xy + z^2 + 1 = 0 → z = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_nonzero_z_l2936_293617


namespace NUMINAMATH_CALUDE_not_p_and_q_is_true_l2936_293694

theorem not_p_and_q_is_true (h1 : ¬(p ∧ q)) (h2 : ¬¬q) : (¬p) ∧ q := by
  sorry

end NUMINAMATH_CALUDE_not_p_and_q_is_true_l2936_293694


namespace NUMINAMATH_CALUDE_largest_package_size_l2936_293663

theorem largest_package_size (hazel_pencils leo_pencils mia_pencils : ℕ) 
  (h1 : hazel_pencils = 36)
  (h2 : leo_pencils = 54)
  (h3 : mia_pencils = 72) :
  Nat.gcd hazel_pencils (Nat.gcd leo_pencils mia_pencils) = 18 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l2936_293663
