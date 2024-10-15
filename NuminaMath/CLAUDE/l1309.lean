import Mathlib

namespace NUMINAMATH_CALUDE_gcd_of_ones_l1309_130938

theorem gcd_of_ones (m n : ℕ+) :
  Nat.gcd ((10^(m.val) - 1) / 9) ((10^(n.val) - 1) / 9) = (10^(Nat.gcd m.val n.val) - 1) / 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_ones_l1309_130938


namespace NUMINAMATH_CALUDE_sphere_area_is_14pi_l1309_130998

/-- A cuboid with vertices on a sphere's surface -/
structure CuboidOnSphere where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  vertices_on_sphere : Bool

/-- The surface area of a sphere containing a cuboid -/
def sphere_surface_area (c : CuboidOnSphere) : ℝ := sorry

/-- Theorem: The surface area of the sphere is 14π -/
theorem sphere_area_is_14pi (c : CuboidOnSphere) 
  (h1 : c.edge1 = 1) 
  (h2 : c.edge2 = 2) 
  (h3 : c.edge3 = 3) 
  (h4 : c.vertices_on_sphere = true) : 
  sphere_surface_area c = 14 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_area_is_14pi_l1309_130998


namespace NUMINAMATH_CALUDE_inscribed_sphere_polyhedron_volume_l1309_130985

/-- A polyhedron with an inscribed sphere -/
structure InscribedSpherePolyhedron where
  /-- The volume of the polyhedron -/
  volume : ℝ
  /-- The total surface area of the polyhedron -/
  surface_area : ℝ
  /-- The radius of the inscribed sphere -/
  inscribed_sphere_radius : ℝ
  /-- The radius is positive -/
  radius_pos : 0 < inscribed_sphere_radius

/-- 
Theorem: For a polyhedron with an inscribed sphere, 
the volume of the polyhedron is equal to one-third of the product 
of its total surface area and the radius of the inscribed sphere.
-/
theorem inscribed_sphere_polyhedron_volume 
  (p : InscribedSpherePolyhedron) : 
  p.volume = (1 / 3) * p.surface_area * p.inscribed_sphere_radius := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_polyhedron_volume_l1309_130985


namespace NUMINAMATH_CALUDE_train_problem_l1309_130965

/-- The speed of the freight train in km/h given the conditions of the train problem -/
def freight_train_speed : ℝ := by sorry

theorem train_problem (passenger_length freight_length : ℝ) (passing_time : ℝ) (speed_ratio : ℚ) :
  passenger_length = 200 →
  freight_length = 280 →
  passing_time = 18 →
  speed_ratio = 5 / 3 →
  freight_train_speed = 36 := by sorry

end NUMINAMATH_CALUDE_train_problem_l1309_130965


namespace NUMINAMATH_CALUDE_factorization_m_squared_minus_3m_l1309_130955

theorem factorization_m_squared_minus_3m (m : ℝ) : m^2 - 3*m = m*(m-3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_m_squared_minus_3m_l1309_130955


namespace NUMINAMATH_CALUDE_number_ratio_proof_l1309_130926

theorem number_ratio_proof (N P : ℚ) (h1 : N = 280) (h2 : (1/5) * N + 7 = P - 7) :
  (P - 7) / N = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_proof_l1309_130926


namespace NUMINAMATH_CALUDE_carlas_quadruple_batch_cans_l1309_130946

/-- Represents the number of cans of each ingredient in a normal batch of Carla's chili -/
structure ChiliBatch where
  chilis : ℕ
  beans : ℕ
  tomatoes : ℕ

/-- Calculates the total number of cans in a batch -/
def totalCans (batch : ChiliBatch) : ℕ :=
  batch.chilis + batch.beans + batch.tomatoes

/-- Represents Carla's normal chili batch -/
def carlasNormalBatch : ChiliBatch :=
  { chilis := 1
  , beans := 2
  , tomatoes := 3 }  -- 50% more than beans: 2 * 1.5 = 3

/-- The number of times Carla is multiplying her normal batch -/
def batchMultiplier : ℕ := 4

/-- Theorem: The total number of cans for Carla's quadruple batch is 24 -/
theorem carlas_quadruple_batch_cans : 
  totalCans carlasNormalBatch * batchMultiplier = 24 := by
  sorry


end NUMINAMATH_CALUDE_carlas_quadruple_batch_cans_l1309_130946


namespace NUMINAMATH_CALUDE_cost_for_five_point_five_kg_l1309_130903

-- Define the relationship between strawberries picked and cost
def strawberry_cost (x : ℝ) : ℝ := 16 * x + 2.5

-- Theorem stating the cost for 5.5kg of strawberries
theorem cost_for_five_point_five_kg :
  strawberry_cost 5.5 = 90.5 := by
  sorry

end NUMINAMATH_CALUDE_cost_for_five_point_five_kg_l1309_130903


namespace NUMINAMATH_CALUDE_x_squared_coefficient_in_binomial_expansion_l1309_130978

/-- Given a natural number n, returns the binomial coefficient C(n,k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The exponent n for which the 5th term in (x-1/x)^n has the largest coefficient -/
def n : ℕ := 8

/-- The coefficient of x^2 in the expansion of (x-1/x)^n -/
def coefficient_x_squared : ℤ := -56

theorem x_squared_coefficient_in_binomial_expansion :
  coefficient_x_squared = (-1)^3 * binomial n 3 := by sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_in_binomial_expansion_l1309_130978


namespace NUMINAMATH_CALUDE_stock_increase_factor_l1309_130959

def initial_investment : ℝ := 900
def num_stocks : ℕ := 3
def stock_c_loss_factor : ℝ := 0.5
def final_total_value : ℝ := 1350

theorem stock_increase_factor :
  let initial_per_stock := initial_investment / num_stocks
  let stock_c_final_value := initial_per_stock * stock_c_loss_factor
  let stock_ab_final_value := final_total_value - stock_c_final_value
  let stock_ab_initial_value := initial_per_stock * 2
  stock_ab_final_value / stock_ab_initial_value = 2 := by sorry

end NUMINAMATH_CALUDE_stock_increase_factor_l1309_130959


namespace NUMINAMATH_CALUDE_min_value_expression_l1309_130931

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (x - 2*y)^2 = (x*y)^3) : 
  4/x^2 + 4/(x*y) + 1/y^2 ≥ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1309_130931


namespace NUMINAMATH_CALUDE_sin_90_degrees_l1309_130967

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l1309_130967


namespace NUMINAMATH_CALUDE_myrtle_eggs_count_l1309_130995

/-- The number of eggs Myrtle has after her trip -/
def myrtle_eggs : ℕ :=
  let num_hens : ℕ := 3
  let eggs_per_hen_per_day : ℕ := 3
  let days_gone : ℕ := 7
  let neighbor_taken : ℕ := 12
  let dropped : ℕ := 5
  
  let total_laid : ℕ := num_hens * eggs_per_hen_per_day * days_gone
  let remaining_after_neighbor : ℕ := total_laid - neighbor_taken
  remaining_after_neighbor - dropped

theorem myrtle_eggs_count : myrtle_eggs = 46 := by
  sorry

end NUMINAMATH_CALUDE_myrtle_eggs_count_l1309_130995


namespace NUMINAMATH_CALUDE_line_point_sum_l1309_130968

/-- The line equation y = -5/3x + 10 -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 10

/-- Point P is on the x-axis -/
def P_on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

/-- Point Q is on the y-axis -/
def Q_on_y_axis (Q : ℝ × ℝ) : Prop := Q.1 = 0

/-- Point T is on line segment PQ -/
def T_on_PQ (P Q T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ T = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2)

/-- Area of triangle POQ is 4 times the area of triangle TOP -/
def area_ratio (P Q T : ℝ × ℝ) : Prop :=
  abs ((P.1 - 0) * (Q.2 - 0) - (Q.1 - 0) * (P.2 - 0)) =
  4 * abs ((T.1 - 0) * (P.2 - 0) - (P.1 - 0) * (T.2 - 0))

theorem line_point_sum (P Q T : ℝ × ℝ) (r s : ℝ) :
  line_equation P.1 P.2 →
  line_equation Q.1 Q.2 →
  line_equation T.1 T.2 →
  P_on_x_axis P →
  Q_on_y_axis Q →
  T_on_PQ P Q T →
  area_ratio P Q T →
  T = (r, s) →
  r + s = 7 := by sorry

end NUMINAMATH_CALUDE_line_point_sum_l1309_130968


namespace NUMINAMATH_CALUDE_alien_alphabet_l1309_130943

theorem alien_alphabet (total : ℕ) (both : ℕ) (triangle_only : ℕ) 
  (h1 : total = 120)
  (h2 : both = 32)
  (h3 : triangle_only = 72)
  (h4 : total = both + triangle_only + (total - (both + triangle_only))) :
  total - (both + triangle_only) = 16 := by
  sorry

end NUMINAMATH_CALUDE_alien_alphabet_l1309_130943


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1309_130991

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 135 → n * (180 - interior_angle) = 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1309_130991


namespace NUMINAMATH_CALUDE_advantages_of_early_license_l1309_130956

-- Define the type for advantages
inductive Advantage
  | CostSavings
  | RentalFlexibility
  | EmploymentOpportunities

-- Define a function to check if an advantage applies to getting a license at 18
def is_advantage_at_18 (a : Advantage) : Prop :=
  match a with
  | Advantage.CostSavings => true
  | Advantage.RentalFlexibility => true
  | Advantage.EmploymentOpportunities => true

-- Define a function to check if an advantage applies to getting a license at 30
def is_advantage_at_30 (a : Advantage) : Prop :=
  match a with
  | Advantage.CostSavings => false
  | Advantage.RentalFlexibility => false
  | Advantage.EmploymentOpportunities => false

-- Theorem stating that there are at least three distinct advantages
-- of getting a license at 18 compared to 30
theorem advantages_of_early_license :
  ∃ (a b c : Advantage), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  is_advantage_at_18 a ∧ is_advantage_at_18 b ∧ is_advantage_at_18 c ∧
  ¬is_advantage_at_30 a ∧ ¬is_advantage_at_30 b ∧ ¬is_advantage_at_30 c :=
sorry

end NUMINAMATH_CALUDE_advantages_of_early_license_l1309_130956


namespace NUMINAMATH_CALUDE_smooth_flow_probability_l1309_130905

def cable_capacities : List Nat := [1, 1, 2, 2, 3, 4]

def total_combinations : Nat := Nat.choose 6 3

def smooth_flow_combinations : Nat := 5

theorem smooth_flow_probability :
  (smooth_flow_combinations : ℚ) / total_combinations = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_smooth_flow_probability_l1309_130905


namespace NUMINAMATH_CALUDE_bicycles_in_garage_l1309_130949

theorem bicycles_in_garage (tricycles unicycles total_wheels : ℕ) 
  (h1 : tricycles = 4)
  (h2 : unicycles = 7)
  (h3 : total_wheels = 25) : ∃ bicycles : ℕ, 
  bicycles * 2 + tricycles * 3 + unicycles * 1 = total_wheels ∧ bicycles = 3 := by
  sorry

end NUMINAMATH_CALUDE_bicycles_in_garage_l1309_130949


namespace NUMINAMATH_CALUDE_grassland_area_l1309_130996

theorem grassland_area (width1 : ℝ) (length : ℝ) : 
  width1 > 0 → length > 0 →
  (width1 + 10) * length = 1000 →
  (width1 - 4) * length = 650 →
  width1 * length = 750 := by
sorry

end NUMINAMATH_CALUDE_grassland_area_l1309_130996


namespace NUMINAMATH_CALUDE_sqrt_7225_minus_55_cube_l1309_130910

theorem sqrt_7225_minus_55_cube (c d : ℕ) (hc : c > 0) (hd : d > 0) 
  (h : Real.sqrt 7225 - 55 = (Real.sqrt c - d)^3) : c + d = 19 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_7225_minus_55_cube_l1309_130910


namespace NUMINAMATH_CALUDE_neighbor_rolls_count_l1309_130932

/-- The number of gift wrap rolls Nellie needs to sell for the fundraiser -/
def total_rolls : ℕ := 45

/-- The number of rolls Nellie sold to her grandmother -/
def grandmother_rolls : ℕ := 1

/-- The number of rolls Nellie sold to her uncle -/
def uncle_rolls : ℕ := 10

/-- The number of rolls Nellie still needs to sell to reach her goal -/
def remaining_rolls : ℕ := 28

/-- The number of rolls Nellie sold to her neighbor -/
def neighbor_rolls : ℕ := total_rolls - remaining_rolls - grandmother_rolls - uncle_rolls

theorem neighbor_rolls_count : neighbor_rolls = 6 := by
  sorry

end NUMINAMATH_CALUDE_neighbor_rolls_count_l1309_130932


namespace NUMINAMATH_CALUDE_line_through_three_points_l1309_130909

/-- Given that the points (0, 2), (10, m), and (25, -3) lie on the same line, prove that m = 0. -/
theorem line_through_three_points (m : ℝ) : 
  (∀ t : ℝ, ∃ a b : ℝ, t * (10 - 0) + 0 = 10 ∧ 
                       t * (m - 2) + 2 = m ∧ 
                       t * (25 - 0) + 0 = 25 ∧ 
                       t * (-3 - 2) + 2 = -3) → 
  m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_three_points_l1309_130909


namespace NUMINAMATH_CALUDE_play_school_kids_l1309_130941

/-- The number of kids in a play school -/
def total_kids (white : ℕ) (yellow : ℕ) (both : ℕ) : ℕ :=
  white + yellow - both

/-- Theorem: The total number of kids in the play school is 35 -/
theorem play_school_kids : total_kids 26 28 19 = 35 := by
  sorry

end NUMINAMATH_CALUDE_play_school_kids_l1309_130941


namespace NUMINAMATH_CALUDE_smallest_cube_multiplier_l1309_130969

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_cube_multiplier (k : ℕ) : 
  (∀ m : ℕ, m < 4500 → ¬ ∃ n : ℕ, m * y = n^3) ∧ 
  (∃ n : ℕ, 4500 * y = n^3) := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_multiplier_l1309_130969


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1309_130934

/-- A function f : ℝ → ℝ is quadratic if there exist constants a, b, c : ℝ 
    such that f(x) = ax² + bx + c for all x : ℝ, and a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

/-- The function f(x) = x² + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : IsQuadratic f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l1309_130934


namespace NUMINAMATH_CALUDE_product_of_sines_l1309_130920

theorem product_of_sines : 
  (1 - Real.sin (π / 12)) * (1 - Real.sin (5 * π / 12)) * 
  (1 - Real.sin (7 * π / 12)) * (1 - Real.sin (11 * π / 12)) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_l1309_130920


namespace NUMINAMATH_CALUDE_absolute_value_equality_l1309_130950

theorem absolute_value_equality (x : ℝ) : |x - 2| = |x + 3| → x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l1309_130950


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1309_130915

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 where a > 0, b > 0,
    and one of its asymptotes is y = √2 x, prove that the eccentricity of C is √3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 2) : 
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1309_130915


namespace NUMINAMATH_CALUDE_reggie_free_throws_l1309_130983

/-- Represents the number of points for each type of shot --/
structure PointValues where
  layup : ℕ
  freeThrow : ℕ
  longShot : ℕ

/-- Represents the shots made by a player --/
structure ShotsMade where
  layups : ℕ
  freeThrows : ℕ
  longShots : ℕ

/-- Calculates the total points scored by a player --/
def calculatePoints (pv : PointValues) (sm : ShotsMade) : ℕ :=
  pv.layup * sm.layups + pv.freeThrow * sm.freeThrows + pv.longShot * sm.longShots

theorem reggie_free_throws 
  (pointValues : PointValues)
  (reggieShotsMade : ShotsMade)
  (brotherShotsMade : ShotsMade)
  (h1 : pointValues.layup = 1)
  (h2 : pointValues.freeThrow = 2)
  (h3 : pointValues.longShot = 3)
  (h4 : reggieShotsMade.layups = 3)
  (h5 : reggieShotsMade.longShots = 1)
  (h6 : brotherShotsMade.layups = 0)
  (h7 : brotherShotsMade.freeThrows = 0)
  (h8 : brotherShotsMade.longShots = 4)
  (h9 : calculatePoints pointValues brotherShotsMade = calculatePoints pointValues reggieShotsMade + 2) :
  reggieShotsMade.freeThrows = 2 := by
  sorry

#check reggie_free_throws

end NUMINAMATH_CALUDE_reggie_free_throws_l1309_130983


namespace NUMINAMATH_CALUDE_f_is_linear_equation_one_var_l1309_130948

/-- A linear equation with one variable is of the form ax + b = 0, where a and b are real numbers and a ≠ 0 -/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function f(x) = x - 1 -/
def f (x : ℝ) : ℝ := x - 1

theorem f_is_linear_equation_one_var :
  is_linear_equation_one_var f :=
sorry

end NUMINAMATH_CALUDE_f_is_linear_equation_one_var_l1309_130948


namespace NUMINAMATH_CALUDE_divisibility_by_three_l1309_130922

theorem divisibility_by_three (n : ℕ+) : 
  (∃ k : ℤ, n = 6*k + 1 ∨ n = 6*k + 2) ↔ 
  (∃ m : ℤ, n * 2^(n : ℕ) + 1 = 3 * m) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l1309_130922


namespace NUMINAMATH_CALUDE_square_sum_mod_three_solution_l1309_130990

theorem square_sum_mod_three_solution (x y z : ℕ) :
  (x^2 + y^2 + z^2) % 3 = 1 →
  ((x = 3 ∧ y = 3 ∧ z = 2) ∨
   (x = 3 ∧ y = 2 ∧ z = 3) ∨
   (x = 2 ∧ y = 3 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_mod_three_solution_l1309_130990


namespace NUMINAMATH_CALUDE_solution_characterization_l1309_130902

/-- The set of solutions to the equation (n+1)^k = n! + 1 for natural numbers n and k -/
def SolutionSet : Set (ℕ × ℕ) :=
  {(1, 1), (2, 1), (4, 2)}

/-- The equation (n+1)^k = n! + 1 -/
def EquationHolds (n k : ℕ) : Prop :=
  (n + 1) ^ k = Nat.factorial n + 1

theorem solution_characterization :
  ∀ (n k : ℕ), EquationHolds n k ↔ (n, k) ∈ SolutionSet := by
  sorry

#check solution_characterization

end NUMINAMATH_CALUDE_solution_characterization_l1309_130902


namespace NUMINAMATH_CALUDE_intersection_union_eq_l1309_130913

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {1, 3, 4}
def C : Set ℝ := {x : ℝ | x^2 - 3*x + 2 > 0}

theorem intersection_union_eq : (A ∪ B) ∩ C = {0, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_union_eq_l1309_130913


namespace NUMINAMATH_CALUDE_bookstore_inventory_theorem_l1309_130988

theorem bookstore_inventory_theorem (historical_fiction : ℝ) (mystery : ℝ) (science_fiction : ℝ) (romance : ℝ)
  (historical_fiction_new : ℝ) (mystery_new : ℝ) (science_fiction_new : ℝ) (romance_new : ℝ)
  (h1 : historical_fiction = 0.4)
  (h2 : mystery = 0.3)
  (h3 : science_fiction = 0.2)
  (h4 : romance = 0.1)
  (h5 : historical_fiction_new = 0.35 * historical_fiction)
  (h6 : mystery_new = 0.6 * mystery)
  (h7 : science_fiction_new = 0.45 * science_fiction)
  (h8 : romance_new = 0.8 * romance) :
  historical_fiction_new / (historical_fiction_new + mystery_new + science_fiction_new + romance_new) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_inventory_theorem_l1309_130988


namespace NUMINAMATH_CALUDE_order_of_rational_numbers_l1309_130918

theorem order_of_rational_numbers (a b : ℚ) 
  (ha : a > 0) (hb : b < 0) (hab : |a| < |b|) : 
  b < -a ∧ -a < a ∧ a < -b := by
  sorry

end NUMINAMATH_CALUDE_order_of_rational_numbers_l1309_130918


namespace NUMINAMATH_CALUDE_time_after_56_hours_l1309_130999

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Adds hours to a given time -/
def addHours (t : Time) (h : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + h * 60
  { hour := (totalMinutes / 60) % 24, minute := totalMinutes % 60 }

theorem time_after_56_hours (start : Time) (h : Nat) :
  start = { hour := 9, minute := 4 } →
  h = 56 →
  addHours start h = { hour := 17, minute := 4 } := by
  sorry

end NUMINAMATH_CALUDE_time_after_56_hours_l1309_130999


namespace NUMINAMATH_CALUDE_divisible_by_five_l1309_130933

theorem divisible_by_five (a b : ℕ+) (h : 5 ∣ (a * b)) : 5 ∣ a ∨ 5 ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l1309_130933


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l1309_130927

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬p x) :=
by sorry

theorem negation_of_specific_proposition : 
  (¬∀ x : ℝ, x^2 - 2*x > 0) ↔ (∃ x : ℝ, x^2 - 2*x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l1309_130927


namespace NUMINAMATH_CALUDE_unique_a_value_l1309_130976

open Real

theorem unique_a_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 π, f x = (cos (2 * x) + a) / sin x) →
  (∀ x ∈ Set.Ioo 0 π, |f x| ≤ 3) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_unique_a_value_l1309_130976


namespace NUMINAMATH_CALUDE_screen_area_difference_l1309_130984

theorem screen_area_difference :
  let square_area (diagonal : ℝ) := diagonal^2 / 2
  (square_area 19 - square_area 17) = 36 := by sorry

end NUMINAMATH_CALUDE_screen_area_difference_l1309_130984


namespace NUMINAMATH_CALUDE_component_probability_l1309_130930

theorem component_probability (p : ℝ) : 
  p ∈ Set.Icc 0 1 →
  (1 - (1 - p)^3 = 0.999) →
  p = 0.9 := by
sorry

end NUMINAMATH_CALUDE_component_probability_l1309_130930


namespace NUMINAMATH_CALUDE_jackson_williams_money_ratio_l1309_130961

/-- Given that Jackson and Williams have a total of $150 and Jackson has $125,
    prove that the ratio of Jackson's money to Williams' money is 5:1 -/
theorem jackson_williams_money_ratio :
  ∀ (jackson_money williams_money : ℝ),
    jackson_money + williams_money = 150 →
    jackson_money = 125 →
    jackson_money / williams_money = 5 := by
  sorry

end NUMINAMATH_CALUDE_jackson_williams_money_ratio_l1309_130961


namespace NUMINAMATH_CALUDE_final_digit_is_two_l1309_130919

/-- Represents the state of the board with counts of zeros, ones, and twos -/
structure BoardState where
  zeros : ℕ
  ones : ℕ
  twos : ℕ

/-- Represents an operation on the board -/
inductive Operation
  | ZeroOne   -- Erase 0 and 1, write 2
  | ZeroTwo   -- Erase 0 and 2, write 1
  | OneTwo    -- Erase 1 and 2, write 0

/-- Applies an operation to the board state -/
def applyOperation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.ZeroOne => ⟨state.zeros - 1, state.ones - 1, state.twos + 1⟩
  | Operation.ZeroTwo => ⟨state.zeros - 1, state.ones + 1, state.twos - 1⟩
  | Operation.OneTwo => ⟨state.zeros + 1, state.ones - 1, state.twos - 1⟩

/-- Checks if the board state has only one digit remaining -/
def isFinalState (state : BoardState) : Bool :=
  (state.zeros + state.ones + state.twos = 1)

/-- Theorem: The final digit is always 2, regardless of the order of operations -/
theorem final_digit_is_two (initialState : BoardState) (ops : List Operation) :
  isFinalState (ops.foldl applyOperation initialState) →
  (ops.foldl applyOperation initialState).twos = 1 := by
  sorry

end NUMINAMATH_CALUDE_final_digit_is_two_l1309_130919


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1309_130945

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/2
  let n : ℕ := 7
  geometric_sum a r n = 127/256 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1309_130945


namespace NUMINAMATH_CALUDE_inverse_matrices_solution_l1309_130994

/-- Given two 2x2 matrices that are inverses of each other, prove that a = 6 and b = 3/25 -/
theorem inverse_matrices_solution :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 3; 2, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![1/5, -1/5; b, 2/5]
  A * B = 1 → a = 6 ∧ b = 3/25 := by
  sorry

#check inverse_matrices_solution

end NUMINAMATH_CALUDE_inverse_matrices_solution_l1309_130994


namespace NUMINAMATH_CALUDE_tent_production_equation_correct_l1309_130928

/-- Represents the tent production scenario -/
structure TentProduction where
  original_plan : ℕ
  increase_percentage : ℚ
  days_ahead : ℕ
  daily_increase : ℕ

/-- The equation representing the tent production scenario -/
def production_equation (tp : TentProduction) (x : ℚ) : Prop :=
  (tp.original_plan : ℚ) / (x - tp.daily_increase) - 
  (tp.original_plan * (1 + tp.increase_percentage)) / x = tp.days_ahead

/-- Theorem stating that the equation correctly represents the given conditions -/
theorem tent_production_equation_correct (tp : TentProduction) (x : ℚ) 
  (h1 : tp.original_plan = 7200)
  (h2 : tp.increase_percentage = 1/5)
  (h3 : tp.days_ahead = 4)
  (h4 : tp.daily_increase = 720)
  (h5 : x > tp.daily_increase) :
  production_equation tp x := by
  sorry

end NUMINAMATH_CALUDE_tent_production_equation_correct_l1309_130928


namespace NUMINAMATH_CALUDE_f_inequality_l1309_130951

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_def (x : ℝ) : f (Real.tan (2 * x)) = Real.tan x ^ 4 + (1 / Real.tan x) ^ 4

theorem f_inequality : ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) ≥ 196 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1309_130951


namespace NUMINAMATH_CALUDE_age_difference_richard_david_l1309_130925

/-- Represents the ages of the three brothers -/
structure BrothersAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : BrothersAges) : Prop :=
  ages.david > ages.scott ∧
  ages.richard > ages.david ∧
  ages.david = ages.scott + 8 ∧
  ages.david = 14 ∧
  ages.richard + 8 = 2 * (ages.scott + 8)

/-- The theorem to be proved -/
theorem age_difference_richard_david (ages : BrothersAges) :
  problem_conditions ages → ages.richard - ages.david = 6 := by
  sorry


end NUMINAMATH_CALUDE_age_difference_richard_david_l1309_130925


namespace NUMINAMATH_CALUDE_cyclist_speed_l1309_130942

/-- Proves that a cyclist's speed is 24 km/h given specific conditions -/
theorem cyclist_speed (hiker_speed : ℝ) (cyclist_travel_time : ℝ) (hiker_catch_up_time : ℝ) 
  (hiker_speed_positive : 0 < hiker_speed)
  (cyclist_travel_time_positive : 0 < cyclist_travel_time)
  (hiker_catch_up_time_positive : 0 < hiker_catch_up_time)
  (hiker_speed_val : hiker_speed = 4)
  (cyclist_travel_time_val : cyclist_travel_time = 5 / 60)
  (hiker_catch_up_time_val : hiker_catch_up_time = 25 / 60) : 
  ∃ (cyclist_speed : ℝ), cyclist_speed = 24 := by
  sorry


end NUMINAMATH_CALUDE_cyclist_speed_l1309_130942


namespace NUMINAMATH_CALUDE_problem_statement_l1309_130900

theorem problem_statement (C D : ℝ) :
  (∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-2 * x^2 + 8 * x + 28) / (x - 3)) →
  C + D = 20 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1309_130900


namespace NUMINAMATH_CALUDE_correct_time_is_two_five_and_five_elevenths_l1309_130929

/-- Represents a time between 2 and 3 o'clock --/
structure Time where
  hour : ℕ
  minute : ℚ
  h_hour : hour = 2
  h_minute : 0 ≤ minute ∧ minute < 60

/-- Converts a Time to minutes past 2:00 --/
def timeToMinutes (t : Time) : ℚ :=
  60 * (t.hour - 2) + t.minute

/-- Represents the misread time by swapping hour and minute hands --/
def misreadTime (t : Time) : ℚ :=
  60 * (t.minute / 5) + 5 * t.hour

theorem correct_time_is_two_five_and_five_elevenths (t : Time) :
  misreadTime t = timeToMinutes t - 55 →
  t.hour = 2 ∧ t.minute = 5 + 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_correct_time_is_two_five_and_five_elevenths_l1309_130929


namespace NUMINAMATH_CALUDE_floor_inequality_l1309_130958

theorem floor_inequality (x y : ℝ) : 
  ⌊2*x⌋ + ⌊2*y⌋ ≥ ⌊x⌋ + ⌊y⌋ + ⌊x + y⌋ :=
sorry

end NUMINAMATH_CALUDE_floor_inequality_l1309_130958


namespace NUMINAMATH_CALUDE_inequality_solution_l1309_130917

theorem inequality_solution (x : ℝ) : (x + 1) / x > 1 ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1309_130917


namespace NUMINAMATH_CALUDE_polynomial_sum_l1309_130937

def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

theorem polynomial_sum (a b c d : ℝ) : 
  (∃ (x : ℝ), f a b x = g c d x) ∧
  (f a b (-a/2) = g c d (-c/2)) ∧
  (g c d (-a/2) = 0) ∧
  (f a b (-c/2) = 0) ∧
  (f a b 50 = -200) ∧
  (g c d 50 = -200) →
  a + c = -200 := by sorry

end NUMINAMATH_CALUDE_polynomial_sum_l1309_130937


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1309_130957

theorem complex_equation_solution (i : ℂ) (n : ℝ) 
  (h1 : i * i = -1) 
  (h2 : (2 : ℂ) / (1 - i) = 1 + n * i) : 
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1309_130957


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1309_130981

-- Define the points A and B
def A : ℝ × ℝ := (2, -2)
def B : ℝ × ℝ := (4, 3)

-- Define vector a as a function of k
def a (k : ℝ) : ℝ × ℝ := (2*k - 1, 7)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem statement
theorem parallel_vectors_k_value : 
  ∃ (c : ℝ), c ≠ 0 ∧ a (19/10) = (c * AB.1, c * AB.2) :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1309_130981


namespace NUMINAMATH_CALUDE_prove_age_difference_l1309_130901

def age_difference (freyja_age eli_age sarah_age kaylin_age : ℕ) : Prop :=
  freyja_age = 10 ∧
  eli_age = freyja_age + 9 ∧
  sarah_age = 2 * eli_age ∧
  kaylin_age = 33 ∧
  sarah_age - kaylin_age = 5

theorem prove_age_difference :
  ∃ (freyja_age eli_age sarah_age kaylin_age : ℕ),
    age_difference freyja_age eli_age sarah_age kaylin_age :=
by
  sorry

end NUMINAMATH_CALUDE_prove_age_difference_l1309_130901


namespace NUMINAMATH_CALUDE_water_in_tank_after_rain_l1309_130962

/-- Given an initial amount of water, a water flow rate, and a rainstorm duration,
    calculate the final amount of water in the tank. -/
def final_water_amount (initial_amount : ℝ) (flow_rate : ℝ) (duration : ℝ) : ℝ :=
  initial_amount + flow_rate * duration

/-- Theorem stating that given the specific conditions in the problem,
    the final amount of water in the tank is 280 L. -/
theorem water_in_tank_after_rain : final_water_amount 100 2 90 = 280 := by
  sorry

end NUMINAMATH_CALUDE_water_in_tank_after_rain_l1309_130962


namespace NUMINAMATH_CALUDE_equal_non_overlapping_areas_l1309_130908

-- Define two congruent triangles
def Triangle : Type := ℝ × ℝ × ℝ

-- Define a function to calculate the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the hexagon formed by the intersection
def Hexagon : Type := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

-- Define a function to calculate the area of a hexagon
def hexagon_area (h : Hexagon) : ℝ := sorry

-- Define the overlapping triangles and their intersection
def triangles_overlap (t1 t2 : Triangle) (h : Hexagon) : Prop :=
  ∃ (a1 a2 : ℝ), 
    area t1 = hexagon_area h + a1 ∧
    area t2 = hexagon_area h + a2

-- Theorem statement
theorem equal_non_overlapping_areas 
  (t1 t2 : Triangle) 
  (h : Hexagon) 
  (congruent : area t1 = area t2) 
  (overlap : triangles_overlap t1 t2 h) : 
  ∃ (a : ℝ), 
    area t1 = hexagon_area h + a ∧ 
    area t2 = hexagon_area h + a := 
by sorry

end NUMINAMATH_CALUDE_equal_non_overlapping_areas_l1309_130908


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l1309_130964

-- Define the line l
def line_l (x y : ℝ) : Prop := 2 * x + 4 * y + 3 = 0

-- Define point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the relation between O, Q, and P
def relation_OQP (qx qy px py : ℝ) : Prop :=
  2 * (qx - 0, qy - 0) = (px - qx, py - qy)

-- Theorem statement
theorem trajectory_of_Q (qx qy : ℝ) :
  (∃ px py, point_P px py ∧ relation_OQP qx qy px py) →
  2 * qx + 4 * qy + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l1309_130964


namespace NUMINAMATH_CALUDE_movies_watched_correct_l1309_130914

/-- The number of movies watched in the 'crazy silly school' series --/
def moviesWatched (totalMovies : ℕ) (moviesToWatch : ℕ) : ℕ :=
  totalMovies - moviesToWatch

/-- Theorem: The number of movies watched is correct --/
theorem movies_watched_correct (totalMovies moviesToWatch : ℕ) 
  (h1 : totalMovies = 17) 
  (h2 : moviesToWatch = 10) : 
  moviesWatched totalMovies moviesToWatch = 7 := by
  sorry

#eval moviesWatched 17 10

end NUMINAMATH_CALUDE_movies_watched_correct_l1309_130914


namespace NUMINAMATH_CALUDE_apple_harvest_l1309_130954

/-- Proves that the initial number of apples is 569 given the harvesting conditions -/
theorem apple_harvest (new_apples : ℕ) (rotten_apples : ℕ) (current_apples : ℕ)
  (h1 : new_apples = 419)
  (h2 : rotten_apples = 263)
  (h3 : current_apples = 725) :
  current_apples + rotten_apples - new_apples = 569 := by
  sorry

#check apple_harvest

end NUMINAMATH_CALUDE_apple_harvest_l1309_130954


namespace NUMINAMATH_CALUDE_store_visitors_l1309_130993

theorem store_visitors (first_hour_left second_hour_in second_hour_out final_count : ℕ) :
  first_hour_left = 27 →
  second_hour_in = 18 →
  second_hour_out = 9 →
  final_count = 76 →
  ∃ first_hour_in : ℕ, first_hour_in = 94 ∧
    final_count = first_hour_in - first_hour_left + second_hour_in - second_hour_out :=
by sorry

end NUMINAMATH_CALUDE_store_visitors_l1309_130993


namespace NUMINAMATH_CALUDE_infinite_symmetry_centers_l1309_130987

/-- A point in a 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A figure in a 2D space -/
structure Figure :=
  (points : Set Point)

/-- A symmetry transformation with respect to a center point -/
def symmetryTransform (center : Point) (p : Point) : Point :=
  { x := 2 * center.x - p.x,
    y := 2 * center.y - p.y }

/-- A center of symmetry for a figure -/
def isSymmetryCenter (f : Figure) (c : Point) : Prop :=
  ∀ p ∈ f.points, symmetryTransform c p ∈ f.points

/-- The set of all symmetry centers for a figure -/
def symmetryCenters (f : Figure) : Set Point :=
  { c | isSymmetryCenter f c }

/-- Main theorem: If a figure has more than one center of symmetry, 
    it must have infinitely many centers of symmetry -/
theorem infinite_symmetry_centers (f : Figure) :
  (∃ c₁ c₂ : Point, c₁ ≠ c₂ ∧ c₁ ∈ symmetryCenters f ∧ c₂ ∈ symmetryCenters f) →
  ¬ Finite (symmetryCenters f) :=
sorry

end NUMINAMATH_CALUDE_infinite_symmetry_centers_l1309_130987


namespace NUMINAMATH_CALUDE_rectangular_plot_fence_poles_l1309_130966

/-- Calculates the number of fence poles needed for a rectangular plot -/
def fence_poles (length width pole_distance : ℕ) : ℕ :=
  (2 * (length + width)) / pole_distance

/-- Theorem: A 90m by 40m rectangular plot with fence poles 5m apart needs 52 poles -/
theorem rectangular_plot_fence_poles :
  fence_poles 90 40 5 = 52 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_fence_poles_l1309_130966


namespace NUMINAMATH_CALUDE_hilt_bee_count_l1309_130971

/-- The number of bees Mrs. Hilt saw on the first day -/
def first_day_bees : ℕ := 144

/-- The multiplier for the number of bees on the second day -/
def day_two_multiplier : ℕ := 3

/-- The number of bees Mrs. Hilt saw on the second day -/
def second_day_bees : ℕ := first_day_bees * day_two_multiplier

/-- Theorem stating that Mrs. Hilt saw 432 bees on the second day -/
theorem hilt_bee_count : second_day_bees = 432 := by
  sorry

end NUMINAMATH_CALUDE_hilt_bee_count_l1309_130971


namespace NUMINAMATH_CALUDE_trig_identity_l1309_130979

theorem trig_identity :
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (47 * π / 180) * Real.cos (103 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1309_130979


namespace NUMINAMATH_CALUDE_quadrilateral_areas_product_is_square_l1309_130989

/-- Represents a convex quadrilateral divided by its diagonals -/
structure ConvexQuadrilateral where
  /-- Areas of the four triangles formed by the diagonals -/
  areas : Fin 4 → ℕ

/-- Theorem: The product of the areas of the four triangles in a convex quadrilateral
    divided by its diagonals is a perfect square -/
theorem quadrilateral_areas_product_is_square (q : ConvexQuadrilateral) :
  ∃ k : ℕ, (q.areas 0) * (q.areas 1) * (q.areas 2) * (q.areas 3) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_areas_product_is_square_l1309_130989


namespace NUMINAMATH_CALUDE_paper_fold_cut_ratio_l1309_130997

theorem paper_fold_cut_ratio : 
  let square_side : ℝ := 6
  let fold_ratio : ℝ := 1/3
  let cut_ratio : ℝ := 2/3
  let small_width : ℝ := square_side * fold_ratio
  let large_width : ℝ := square_side * (1 - fold_ratio) * (1 - cut_ratio)
  let small_perimeter : ℝ := 2 * (square_side + small_width)
  let large_perimeter : ℝ := 2 * (square_side + large_width)
  small_perimeter / large_perimeter = 12/17 := by
sorry

end NUMINAMATH_CALUDE_paper_fold_cut_ratio_l1309_130997


namespace NUMINAMATH_CALUDE_sequence_problem_l1309_130960

theorem sequence_problem (a : ℕ → ℚ) (m : ℕ) :
  a 1 = 1 →
  (∀ n : ℕ, n ≥ 1 → a n - a (n + 1) = a (n + 1) * a n) →
  8 * a m = 1 →
  m = 8 :=
by sorry

end NUMINAMATH_CALUDE_sequence_problem_l1309_130960


namespace NUMINAMATH_CALUDE_theater_revenue_specific_case_l1309_130972

def theater_revenue (orchestra_price balcony_price : ℕ) 
                    (total_tickets balcony_orchestra_diff : ℕ) : ℕ :=
  let orchestra_tickets := (total_tickets - balcony_orchestra_diff) / 2
  let balcony_tickets := total_tickets - orchestra_tickets
  orchestra_price * orchestra_tickets + balcony_price * balcony_tickets

theorem theater_revenue_specific_case :
  theater_revenue 12 8 360 140 = 3320 := by
  sorry

end NUMINAMATH_CALUDE_theater_revenue_specific_case_l1309_130972


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1309_130947

theorem solve_linear_equation (x : ℝ) : 5 * x + 3 = 10 * x - 22 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1309_130947


namespace NUMINAMATH_CALUDE_tangent_slope_constraint_implies_a_range_l1309_130936

theorem tangent_slope_constraint_implies_a_range
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x, f x = -x^3 + a*x^2 + b)
  (h2 : ∀ x, (deriv f x) < 1) :
  -Real.sqrt 3 < a ∧ a < Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_constraint_implies_a_range_l1309_130936


namespace NUMINAMATH_CALUDE_race_result_theorem_l1309_130953

-- Define the girls
inductive Girl : Type
  | Anna : Girl
  | Bella : Girl
  | Csilla : Girl
  | Dora : Girl

-- Define the positions
inductive Position : Type
  | First : Position
  | Second : Position
  | Third : Position
  | Fourth : Position

def race_result : Girl → Position := sorry

-- Define the statements
def anna_statement : Prop := race_result Girl.Anna ≠ Position.First ∧ race_result Girl.Anna ≠ Position.Fourth
def bella_statement : Prop := race_result Girl.Bella ≠ Position.First
def csilla_statement : Prop := race_result Girl.Csilla = Position.First
def dora_statement : Prop := race_result Girl.Dora = Position.Fourth

-- Define the condition that three statements are true and one is false
def statements_condition : Prop :=
  (anna_statement ∧ bella_statement ∧ csilla_statement ∧ ¬dora_statement) ∨
  (anna_statement ∧ bella_statement ∧ ¬csilla_statement ∧ dora_statement) ∨
  (anna_statement ∧ ¬bella_statement ∧ csilla_statement ∧ dora_statement) ∨
  (¬anna_statement ∧ bella_statement ∧ csilla_statement ∧ dora_statement)

-- Theorem to prove
theorem race_result_theorem :
  statements_condition →
  (¬dora_statement ∧ race_result Girl.Csilla = Position.First) := by
  sorry

end NUMINAMATH_CALUDE_race_result_theorem_l1309_130953


namespace NUMINAMATH_CALUDE_max_value_cosine_sine_l1309_130904

theorem max_value_cosine_sine (θ : Real) (h : 0 < θ ∧ θ < π) :
  ∃ (max : Real), max = (4 * Real.sqrt 3) / 9 ∧
    ∀ x, 0 < x ∧ x < π →
      Real.cos (x / 2) * (1 + Real.sin x) ≤ max ∧
      ∃ y, 0 < y ∧ y < π ∧ Real.cos (y / 2) * (1 + Real.sin y) = max :=
by sorry

end NUMINAMATH_CALUDE_max_value_cosine_sine_l1309_130904


namespace NUMINAMATH_CALUDE_doughnut_cost_calculation_l1309_130970

theorem doughnut_cost_calculation (num_doughnuts : ℕ) (price_per_doughnut : ℚ) (profit : ℚ) :
  let total_revenue := num_doughnuts * price_per_doughnut
  let cost_of_ingredients := total_revenue - profit
  cost_of_ingredients = num_doughnuts * price_per_doughnut - profit :=
by sorry

-- Example usage with the given values
def dorothy_example : ℚ :=
  let num_doughnuts : ℕ := 25
  let price_per_doughnut : ℚ := 3
  let profit : ℚ := 22
  num_doughnuts * price_per_doughnut - profit

#eval dorothy_example -- This should evaluate to 53

end NUMINAMATH_CALUDE_doughnut_cost_calculation_l1309_130970


namespace NUMINAMATH_CALUDE_charge_difference_l1309_130907

/-- The charge for a single color copy at print shop X -/
def charge_X : ℚ := 1.25

/-- The charge for a single color copy at print shop Y -/
def charge_Y : ℚ := 2.75

/-- The number of color copies -/
def num_copies : ℕ := 80

/-- The theorem stating the difference in charges between print shops Y and X for 80 color copies -/
theorem charge_difference : (num_copies : ℚ) * charge_Y - (num_copies : ℚ) * charge_X = 120 := by
  sorry

end NUMINAMATH_CALUDE_charge_difference_l1309_130907


namespace NUMINAMATH_CALUDE_original_number_proof_l1309_130924

theorem original_number_proof : ∃ x : ℝ, 3 * (2 * x + 9) = 81 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1309_130924


namespace NUMINAMATH_CALUDE_may_friday_to_monday_l1309_130992

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a day in May -/
structure DayInMay where
  day : Nat
  dayOfWeek : DayOfWeek

/-- The function that determines the day of the week for a given day in May -/
def dayOfWeekInMay (d : Nat) : DayOfWeek :=
  sorry

theorem may_friday_to_monday (r n : Nat) 
  (h1 : dayOfWeekInMay r = DayOfWeek.Friday)
  (h2 : dayOfWeekInMay n = DayOfWeek.Monday)
  (h3 : 15 < n)
  (h4 : n < 25) :
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_may_friday_to_monday_l1309_130992


namespace NUMINAMATH_CALUDE_lesser_fraction_l1309_130974

theorem lesser_fraction (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_sum : x + y = 13/14) (h_product : x * y = 1/5) : 
  min x y = 87/700 := by
sorry

end NUMINAMATH_CALUDE_lesser_fraction_l1309_130974


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1309_130977

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 - 3 * 2 = 192 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1309_130977


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l1309_130973

theorem binomial_expansion_properties :
  let f := fun x => (2 * x + 1) ^ 4
  ∃ (a b c d e : ℤ),
    f x = a * x^4 + b * x^3 + c * x^2 + d * x + e ∧
    c = 24 ∧
    a + b + c + d + e = 81 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l1309_130973


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l1309_130940

/-- Calculates the ratio of average speed to still water speed for a boat in a river --/
theorem boat_speed_ratio 
  (v : ℝ) -- Boat speed in still water
  (c : ℝ) -- River current speed
  (d : ℝ) -- Distance traveled each way
  (h1 : v > 0)
  (h2 : c ≥ 0)
  (h3 : c < v)
  (h4 : d > 0)
  : (2 * d) / ((d / (v + c)) + (d / (v - c))) / v = 24 / 25 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l1309_130940


namespace NUMINAMATH_CALUDE_midpoint_of_AB_l1309_130980

-- Define the point F
def F : ℝ × ℝ := (0, 1)

-- Define the line y = -5
def line_y_neg5 (x : ℝ) : ℝ := -5

-- Define the line x - 4y + 2 = 0
def line_l (x y : ℝ) : Prop := x - 4*y + 2 = 0

-- Define the distance condition for point P
def distance_condition (P : ℝ × ℝ) : Prop :=
  ∃ (d : ℝ), d > 0 ∧ 
    (Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) + 4 = Real.sqrt ((P.1 - P.1)^2 + (P.2 - line_y_neg5 P.1)^2))

-- Define the trajectory of P (parabola)
def trajectory (x y : ℝ) : Prop := x^2 = 4*y

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  trajectory A.1 A.2 ∧ trajectory B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem midpoint_of_AB :
  ∀ (P A B : ℝ × ℝ),
  distance_condition P →
  intersection_points A B →
  (A.1 + B.1) / 2 = 1/2 ∧ (A.2 + B.2) / 2 = 5/8 :=
sorry

end NUMINAMATH_CALUDE_midpoint_of_AB_l1309_130980


namespace NUMINAMATH_CALUDE_quadratic_root_value_l1309_130944

theorem quadratic_root_value (m : ℝ) : 
  m^2 - m - 2 = 0 → 2*m^2 - 2*m + 2022 = 2026 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l1309_130944


namespace NUMINAMATH_CALUDE_triangle_area_ordering_l1309_130923

/-- The area of the first triangle -/
def m : ℚ := 15/2

/-- The area of the second triangle -/
def n : ℚ := 13/2

/-- The area of the third triangle -/
def p : ℚ := 7

/-- The side length of the square -/
def square_side : ℚ := 4

/-- The area of the square -/
def square_area : ℚ := square_side * square_side

/-- Theorem stating that the areas of the triangles satisfy n < p < m -/
theorem triangle_area_ordering : n < p ∧ p < m := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_ordering_l1309_130923


namespace NUMINAMATH_CALUDE_solution_set_of_system_l1309_130975

theorem solution_set_of_system : ∃! S : Set (ℝ × ℝ),
  S = {(-1, 2), (2, -1), (-2, 7)} ∧
  ∀ (x y : ℝ), (x, y) ∈ S ↔ 
    (y^2 + 2*x*y + x^2 - 6*y - 6*x + 5 = 0) ∧ 
    (y - x + 1 = x^2 - 3*x) ∧ 
    (x ≠ 0) ∧ 
    (x ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_system_l1309_130975


namespace NUMINAMATH_CALUDE_sum_of_xy_l1309_130963

theorem sum_of_xy (x y : ℕ+) 
  (eq1 : 10 * x + y = 75)
  (eq2 : 10 * y + x = 57) : 
  x + y = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xy_l1309_130963


namespace NUMINAMATH_CALUDE_mean_proportional_proof_l1309_130986

theorem mean_proportional_proof :
  let a : ℝ := 7921
  let b : ℝ := 9481
  let m : ℝ := 8665
  m = (a * b).sqrt := by sorry

end NUMINAMATH_CALUDE_mean_proportional_proof_l1309_130986


namespace NUMINAMATH_CALUDE_pyramid_base_edge_length_l1309_130935

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Slant height from apex to midpoint of a base side -/
  slant_height : ℝ
  /-- Radius of the hemisphere -/
  hemisphere_radius : ℝ

/-- Theorem stating the edge-length of the pyramid's base -/
theorem pyramid_base_edge_length (p : PyramidWithHemisphere) 
  (h1 : p.height = 8)
  (h2 : p.slant_height = 10)
  (h3 : p.hemisphere_radius = 3) :
  ∃ (edge_length : ℝ), edge_length = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_base_edge_length_l1309_130935


namespace NUMINAMATH_CALUDE_inequality_proof_l1309_130921

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  a / b + b / c + c / a + b / a + a / c + c / b + 6 ≥ 2 * Real.sqrt 2 * (Real.sqrt ((1 - a) / a) + Real.sqrt ((1 - b) / b) + Real.sqrt ((1 - c) / c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1309_130921


namespace NUMINAMATH_CALUDE_solution_difference_l1309_130982

theorem solution_difference : ∃ (a b : ℝ), 
  (∀ x : ℝ, (3*x - 9) / (x^2 + x - 6) = x + 1 ↔ (x = a ∨ x = b)) ∧ 
  a > b ∧ 
  a - b = 4 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l1309_130982


namespace NUMINAMATH_CALUDE_limit_of_sequence_l1309_130912

def a (n : ℕ) : ℚ := (4 * n - 3) / (2 * n + 1)

theorem limit_of_sequence : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l1309_130912


namespace NUMINAMATH_CALUDE_factors_of_48_l1309_130906

/-- The number of distinct positive factors of 48 is 10. -/
theorem factors_of_48 : Nat.card (Nat.divisors 48) = 10 := by sorry

end NUMINAMATH_CALUDE_factors_of_48_l1309_130906


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l1309_130952

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem: The loss percentage for a radio with cost price 1900 and selling price 1330 is 30% -/
theorem radio_loss_percentage :
  let cost_price : ℚ := 1900
  let selling_price : ℚ := 1330
  loss_percentage cost_price selling_price = 30 := by
sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l1309_130952


namespace NUMINAMATH_CALUDE_divisible_by_11_iff_valid_pair_l1309_130939

def is_valid_pair (a b : Nat) : Prop :=
  (a, b) ∈ [(8, 0), (9, 1), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9)]

def number_from_digits (a b : Nat) : Nat :=
  380000 + a * 1000 + 750 + b

theorem divisible_by_11_iff_valid_pair (a b : Nat) :
  a < 10 ∧ b < 10 →
  (number_from_digits a b) % 11 = 0 ↔ is_valid_pair a b := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_11_iff_valid_pair_l1309_130939


namespace NUMINAMATH_CALUDE_chinese_team_gold_medal_probability_l1309_130911

theorem chinese_team_gold_medal_probability 
  (prob_A prob_B : ℚ)
  (h1 : prob_A = 3 / 7)
  (h2 : prob_B = 1 / 4)
  (h3 : ∀ x y : ℚ, x + y = prob_A + prob_B → x ≤ prob_A ∧ y ≤ prob_B) :
  prob_A + prob_B = 19 / 28 := by
sorry

end NUMINAMATH_CALUDE_chinese_team_gold_medal_probability_l1309_130911


namespace NUMINAMATH_CALUDE_external_tangent_intercept_l1309_130916

/-- Definition of a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Definition of a line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Function to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- Function to check if a line is a common external tangent to two circles -/
def isCommonExternalTangent (l : Line) (c1 c2 : Circle) : Prop :=
  sorry

theorem external_tangent_intercept : 
  let c1 : Circle := { center := (2, 4), radius := 4 }
  let c2 : Circle := { center := (14, 9), radius := 9 }
  ∃ l : Line, l.slope > 0 ∧ isCommonExternalTangent l c1 c2 ∧ l.intercept = 912 / 119 :=
sorry

end NUMINAMATH_CALUDE_external_tangent_intercept_l1309_130916
