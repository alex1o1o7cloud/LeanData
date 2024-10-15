import Mathlib

namespace NUMINAMATH_CALUDE_power_inequality_set_l1065_106589

theorem power_inequality_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | a^(x+3) > a^(2*x)} = {x : ℝ | x > 3} := by sorry

end NUMINAMATH_CALUDE_power_inequality_set_l1065_106589


namespace NUMINAMATH_CALUDE_dress_designs_count_l1065_106527

/-- The number of color choices available for a dress design. -/
def num_colors : ℕ := 5

/-- The number of pattern choices available for a dress design. -/
def num_patterns : ℕ := 4

/-- The number of accessory choices available for a dress design. -/
def num_accessories : ℕ := 2

/-- The total number of possible dress designs. -/
def total_designs : ℕ := num_colors * num_patterns * num_accessories

/-- Theorem stating that the total number of possible dress designs is 40. -/
theorem dress_designs_count : total_designs = 40 := by
  sorry

end NUMINAMATH_CALUDE_dress_designs_count_l1065_106527


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1065_106540

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 26) 
  (h2 : 4 * (a + b + c) = 28) : 
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l1065_106540


namespace NUMINAMATH_CALUDE_equation_with_multiple_solutions_l1065_106576

theorem equation_with_multiple_solutions (a b : ℝ) :
  (∀ x y : ℝ, x ≠ y → a * x + (b - 3) = (5 * a - 1) * x + 3 * b ∧
                     a * y + (b - 3) = (5 * a - 1) * y + 3 * b) →
  100 * a + 4 * b = 31 := by
sorry

end NUMINAMATH_CALUDE_equation_with_multiple_solutions_l1065_106576


namespace NUMINAMATH_CALUDE_counterfeit_coin_determination_l1065_106508

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a group of coins -/
structure CoinGroup where
  size : Nat
  hasFake : Bool

/-- Represents a weighing action -/
structure Weighing where
  left : CoinGroup
  right : CoinGroup

/-- The state of knowledge about the counterfeit coins -/
inductive FakeState
  | Unknown : FakeState
  | Heavier : FakeState
  | Lighter : FakeState

/-- A strategy for determining the state of the counterfeit coins -/
def Strategy := List Weighing

/-- The result of applying a strategy -/
def StrategyResult := FakeState

/-- Axiom: There are 239 coins in total -/
axiom total_coins : Nat
axiom total_coins_eq : total_coins = 239

/-- Axiom: There are exactly two counterfeit coins -/
axiom num_fake_coins : Nat
axiom num_fake_coins_eq : num_fake_coins = 2

/-- Theorem: It is possible to determine whether the counterfeit coins are heavier or lighter in exactly three weighings -/
theorem counterfeit_coin_determination :
  ∃ (s : Strategy),
    (s.length = 3) ∧
    (∀ (fake_heavier : Bool),
      ∃ (result : StrategyResult),
        (result = FakeState.Heavier ∧ fake_heavier = true) ∨
        (result = FakeState.Lighter ∧ fake_heavier = false)) :=
by sorry

end NUMINAMATH_CALUDE_counterfeit_coin_determination_l1065_106508


namespace NUMINAMATH_CALUDE_women_per_table_l1065_106569

theorem women_per_table (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 5 →
  men_per_table = 3 →
  total_customers = 40 →
  ∃ (women_per_table : ℕ),
    women_per_table * num_tables + men_per_table * num_tables = total_customers ∧
    women_per_table = 5 :=
by sorry

end NUMINAMATH_CALUDE_women_per_table_l1065_106569


namespace NUMINAMATH_CALUDE_gcd_50404_40303_l1065_106521

theorem gcd_50404_40303 : Nat.gcd 50404 40303 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_50404_40303_l1065_106521


namespace NUMINAMATH_CALUDE_ellipse_equation_l1065_106558

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    if its eccentricity is √3/2 and the distance from one endpoint of
    the minor axis to the right focus is 2, then its equation is x²/4 + y² = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := (Real.sqrt 3) / 2
  let d := 2
  (e = Real.sqrt (1 - b^2 / a^2) ∧ d = a) →
  a^2 = 4 ∧ b^2 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1065_106558


namespace NUMINAMATH_CALUDE_sum_lower_bound_l1065_106588

theorem sum_lower_bound (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) (h4 : a = 1 / b) :
  a + 2014 * b > 2015 := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l1065_106588


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l1065_106509

/-- Given a person walking at two different speeds, prove the actual distance traveled -/
theorem actual_distance_traveled 
  (initial_speed : ℝ) 
  (increased_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : initial_speed = 10) 
  (h2 : increased_speed = 15) 
  (h3 : additional_distance = 15) 
  (h4 : ∃ t : ℝ, increased_speed * t = initial_speed * t + additional_distance) : 
  ∃ d : ℝ, d = 30 ∧ d = initial_speed * (additional_distance / (increased_speed - initial_speed)) :=
by sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l1065_106509


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l1065_106529

theorem floor_ceiling_sum : ⌊(3.67 : ℝ)⌋ + ⌈(-14.2 : ℝ)⌉ = -11 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l1065_106529


namespace NUMINAMATH_CALUDE_full_time_one_year_count_l1065_106500

/-- Represents the number of employees in different categories at company x -/
structure CompanyEmployees where
  total : ℕ
  fullTime : ℕ
  atLeastOneYear : ℕ
  neitherFullTimeNorOneYear : ℕ

/-- The function to calculate the number of full-time employees who have worked at least one year -/
def fullTimeAndOneYear (e : CompanyEmployees) : ℕ :=
  e.total - (e.fullTime + e.atLeastOneYear - e.neitherFullTimeNorOneYear)

/-- Theorem stating the number of full-time employees who have worked at least one year -/
theorem full_time_one_year_count (e : CompanyEmployees) 
  (h1 : e.total = 130)
  (h2 : e.fullTime = 80)
  (h3 : e.atLeastOneYear = 100)
  (h4 : e.neitherFullTimeNorOneYear = 20) :
  fullTimeAndOneYear e = 90 := by
  sorry

end NUMINAMATH_CALUDE_full_time_one_year_count_l1065_106500


namespace NUMINAMATH_CALUDE_gcd_problem_l1065_106590

theorem gcd_problem (m n : ℕ+) (h : Nat.gcd m n = 10) : Nat.gcd (12 * m) (18 * n) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1065_106590


namespace NUMINAMATH_CALUDE_marys_next_birthday_age_l1065_106544

/-- Proves that Mary's age on her next birthday is 11 years, given the conditions of the problem. -/
theorem marys_next_birthday_age :
  ∀ (m s d : ℝ),
  m = 1.3 * s →  -- Mary is 30% older than Sally
  s = 0.75 * d →  -- Sally is 25% younger than Danielle
  m + s + d = 30 →  -- Sum of their ages is 30 years
  ⌈m⌉ = 11  -- Mary's age on her next birthday (ceiling of her current age)
  := by sorry

end NUMINAMATH_CALUDE_marys_next_birthday_age_l1065_106544


namespace NUMINAMATH_CALUDE_fraction_meaningful_range_l1065_106531

theorem fraction_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y = 2 / (x + 3)) → x ≠ -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_range_l1065_106531


namespace NUMINAMATH_CALUDE_total_distance_12_hours_l1065_106573

def car_distance (initial_speed : ℕ) (speed_increase : ℕ) (hours : ℕ) : ℕ :=
  let speeds := List.range hours |>.map (fun h => initial_speed + h * speed_increase)
  speeds.sum

theorem total_distance_12_hours :
  car_distance 50 2 12 = 782 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_12_hours_l1065_106573


namespace NUMINAMATH_CALUDE_exists_far_reaching_quadrilateral_with_bounded_area_l1065_106543

/-- A point in the 2D plane with integer coordinates. -/
structure Point where
  x : ℤ
  y : ℤ

/-- A rectangle defined by its width and height. -/
structure Rectangle where
  width : ℤ
  height : ℤ

/-- A quadrilateral defined by its four vertices. -/
structure Quadrilateral where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Predicate to check if a point is on or inside a rectangle. -/
def pointInRectangle (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

/-- Predicate to check if a quadrilateral is far-reaching in a rectangle. -/
def isFarReaching (q : Quadrilateral) (r : Rectangle) : Prop :=
  (pointInRectangle q.v1 r ∧ pointInRectangle q.v2 r ∧ pointInRectangle q.v3 r ∧ pointInRectangle q.v4 r) ∧
  (q.v1.x = 0 ∨ q.v2.x = 0 ∨ q.v3.x = 0 ∨ q.v4.x = 0) ∧
  (q.v1.y = 0 ∨ q.v2.y = 0 ∨ q.v3.y = 0 ∨ q.v4.y = 0) ∧
  (q.v1.x = r.width ∨ q.v2.x = r.width ∨ q.v3.x = r.width ∨ q.v4.x = r.width) ∧
  (q.v1.y = r.height ∨ q.v2.y = r.height ∨ q.v3.y = r.height ∨ q.v4.y = r.height)

/-- Calculate the area of a quadrilateral. -/
def quadrilateralArea (q : Quadrilateral) : ℚ :=
  sorry  -- The actual area calculation would go here

/-- The main theorem to be proved. -/
theorem exists_far_reaching_quadrilateral_with_bounded_area
  (n m : ℕ) (hn : n ≤ 10^10) (hm : m ≤ 10^10) :
  ∃ (q : Quadrilateral), isFarReaching q (Rectangle.mk n m) ∧ quadrilateralArea q ≤ 10^6 := by
  sorry

end NUMINAMATH_CALUDE_exists_far_reaching_quadrilateral_with_bounded_area_l1065_106543


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1065_106528

theorem inequality_solution_set :
  {x : ℝ | -1/3 * x + 1 ≤ -5} = {x : ℝ | x ≥ 18} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1065_106528


namespace NUMINAMATH_CALUDE_logarithm_expression_evaluation_l1065_106557

theorem logarithm_expression_evaluation :
  (Real.log 50 / Real.log 4) / (Real.log 4 / Real.log 25) -
  (Real.log 100 / Real.log 4) / (Real.log 4 / Real.log 50) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_evaluation_l1065_106557


namespace NUMINAMATH_CALUDE_quiz_competition_participants_l1065_106541

theorem quiz_competition_participants (initial_participants : ℕ) : 
  (initial_participants : ℝ) * 0.4 * 0.25 = 30 → initial_participants = 300 := by
  sorry

end NUMINAMATH_CALUDE_quiz_competition_participants_l1065_106541


namespace NUMINAMATH_CALUDE_surviving_cells_after_6_hours_l1065_106596

def cell_population (n : ℕ) : ℕ := 2^n + 1

theorem surviving_cells_after_6_hours :
  cell_population 6 = 65 :=
sorry

end NUMINAMATH_CALUDE_surviving_cells_after_6_hours_l1065_106596


namespace NUMINAMATH_CALUDE_abs_not_positive_iff_eq_l1065_106577

theorem abs_not_positive_iff_eq (y : ℚ) : ¬(0 < |5*y - 8|) ↔ y = 8/5 := by sorry

end NUMINAMATH_CALUDE_abs_not_positive_iff_eq_l1065_106577


namespace NUMINAMATH_CALUDE_intersection_condition_l1065_106539

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1^2}
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - a)^2 ≤ 1}

-- State the theorem
theorem intersection_condition (a : ℝ) :
  M ∩ N a = N a ↔ a ≥ 5/4 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l1065_106539


namespace NUMINAMATH_CALUDE_function_value_at_negative_l1065_106534

theorem function_value_at_negative (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x + 1 / x - 1) (h2 : f a = 2) :
  f (-a) = -4 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_l1065_106534


namespace NUMINAMATH_CALUDE_book_purchase_savings_l1065_106592

theorem book_purchase_savings (full_price_book1 full_price_book2 : ℝ) : 
  full_price_book1 = 33 →
  full_price_book2 > 0 →
  let total_paid := full_price_book1 + (full_price_book2 / 2)
  let full_price := full_price_book1 + full_price_book2
  let savings_ratio := (full_price - total_paid) / full_price
  savings_ratio = 1/5 →
  full_price - total_paid = 11 :=
by sorry

end NUMINAMATH_CALUDE_book_purchase_savings_l1065_106592


namespace NUMINAMATH_CALUDE_carls_watermelon_profit_l1065_106561

/-- Calculates the profit of a watermelon seller -/
def watermelon_profit (initial_count : ℕ) (final_count : ℕ) (price_per_melon : ℕ) : ℕ :=
  (initial_count - final_count) * price_per_melon

/-- Theorem: Carl's watermelon profit -/
theorem carls_watermelon_profit :
  watermelon_profit 53 18 3 = 105 := by
  sorry

end NUMINAMATH_CALUDE_carls_watermelon_profit_l1065_106561


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1065_106542

theorem quadratic_solution_sum (m n : ℝ) (h1 : m ≠ 0) :
  m * 1^2 + n * 1 - 1 = 0 → m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1065_106542


namespace NUMINAMATH_CALUDE_cafeteria_milk_cartons_l1065_106552

/-- Given a number of full stacks of milk cartons and the number of cartons per stack,
    calculate the total number of milk cartons. -/
def totalCartons (numStacks : ℕ) (cartonsPerStack : ℕ) : ℕ :=
  numStacks * cartonsPerStack

/-- Theorem stating that 133 full stacks of 6 milk cartons each result in 798 total cartons. -/
theorem cafeteria_milk_cartons :
  totalCartons 133 6 = 798 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_milk_cartons_l1065_106552


namespace NUMINAMATH_CALUDE_contrapositive_example_l1065_106572

theorem contrapositive_example : 
  (∀ x : ℝ, x > 2 → x^2 > 4) ↔ (∀ x : ℝ, x^2 ≤ 4 → x ≤ 2) := by
sorry

end NUMINAMATH_CALUDE_contrapositive_example_l1065_106572


namespace NUMINAMATH_CALUDE_three_classes_five_spots_l1065_106545

/-- The number of ways for classes to choose scenic spots -/
def num_selection_methods (num_classes : ℕ) (num_spots : ℕ) : ℕ :=
  num_spots ^ num_classes

/-- Theorem: Three classes choosing from five scenic spots results in 5^3 selection methods -/
theorem three_classes_five_spots : num_selection_methods 3 5 = 5^3 := by
  sorry

end NUMINAMATH_CALUDE_three_classes_five_spots_l1065_106545


namespace NUMINAMATH_CALUDE_tangent_line_cubic_l1065_106560

/-- Given a cubic function f(x) = ax³ + x + 1, prove that if its tangent line at 
    (1, f(1)) passes through (2, 7), then a = 1. -/
theorem tangent_line_cubic (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + x + 1
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 1
  let tangent_slope : ℝ := f' 1
  let point_on_curve : ℝ := f 1
  (point_on_curve - 7) / (1 - 2) = tangent_slope → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_l1065_106560


namespace NUMINAMATH_CALUDE_annes_distance_is_six_l1065_106593

/-- The distance traveled by Anne given her walking time and speed -/
def annes_distance (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Theorem stating that Anne's distance traveled is 6 miles -/
theorem annes_distance_is_six : annes_distance 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_annes_distance_is_six_l1065_106593


namespace NUMINAMATH_CALUDE_stream_speed_l1065_106501

/-- Given upstream and downstream speeds of a canoe, calculate the speed of the stream. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 9)
  (h_downstream : downstream_speed = 12) :
  (downstream_speed - upstream_speed) / 2 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1065_106501


namespace NUMINAMATH_CALUDE_product_xy_equals_25_l1065_106567

theorem product_xy_equals_25 (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 32)
  (h2 : (16:ℝ)^(x+y) / (4:ℝ)^(5*y) = 1024) :
  x * y = 25 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_equals_25_l1065_106567


namespace NUMINAMATH_CALUDE_roots_are_imaginary_l1065_106512

theorem roots_are_imaginary (k : ℝ) : 
  let quadratic (x : ℝ) := x^2 - 3*k*x + 2*k^2 - 1
  ∀ r₁ r₂ : ℝ, quadratic r₁ = 0 ∧ quadratic r₂ = 0 → r₁ * r₂ = 8 →
  ∃ a b : ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ 
    (∀ x : ℝ, quadratic x = 0 ↔ x = Complex.mk a b ∨ x = Complex.mk a (-b)) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_imaginary_l1065_106512


namespace NUMINAMATH_CALUDE_abs_ratio_sqrt_five_halves_l1065_106598

theorem abs_ratio_sqrt_five_halves (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + b^2 = 18*a*b) : 
  |((a+b)/(a-b))| = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_abs_ratio_sqrt_five_halves_l1065_106598


namespace NUMINAMATH_CALUDE_pinocchio_problem_l1065_106554

theorem pinocchio_problem (x : ℕ) : 
  x ≠ 0 ∧ x < 10 ∧ (x + x + 1) * x = 111 * x → x = 5 :=
by sorry

end NUMINAMATH_CALUDE_pinocchio_problem_l1065_106554


namespace NUMINAMATH_CALUDE_root_implies_m_value_l1065_106503

theorem root_implies_m_value (x m : ℝ) : 
  x = 2 → x^2 - m*x + 6 = 0 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l1065_106503


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l1065_106591

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

/-- Theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions : 
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by
  sorry

#check f_satisfies_conditions

end NUMINAMATH_CALUDE_f_satisfies_conditions_l1065_106591


namespace NUMINAMATH_CALUDE_joe_marshmallow_fraction_l1065_106570

theorem joe_marshmallow_fraction :
  let dad_marshmallows : ℕ := 21
  let joe_marshmallows : ℕ := 4 * dad_marshmallows
  let dad_roasted : ℕ := dad_marshmallows / 3
  let total_roasted : ℕ := 49
  let joe_roasted : ℕ := total_roasted - dad_roasted
  joe_roasted / joe_marshmallows = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_joe_marshmallow_fraction_l1065_106570


namespace NUMINAMATH_CALUDE_hexagon_pillar_height_l1065_106520

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a regular hexagon with pillars -/
structure HexagonWithPillars where
  sideLength : ℝ
  A : Point3D
  B : Point3D
  C : Point3D
  E : Point3D

/-- The theorem to be proved -/
theorem hexagon_pillar_height 
  (h : HexagonWithPillars) 
  (h_side : h.sideLength > 0)
  (h_A : h.A = ⟨0, 0, 12⟩)
  (h_B : h.B = ⟨h.sideLength, 0, 9⟩)
  (h_C : h.C = ⟨h.sideLength / 2, h.sideLength * Real.sqrt 3 / 2, 10⟩)
  (h_E : h.E = ⟨-h.sideLength, 0, h.E.z⟩) :
  h.E.z = 17 := by
  sorry


end NUMINAMATH_CALUDE_hexagon_pillar_height_l1065_106520


namespace NUMINAMATH_CALUDE_symmetric_point_quadrant_l1065_106549

/-- Given that point P(m,m-n) is symmetric to point Q(2,3) with respect to the origin,
    prove that point M(m,n) is in the second quadrant. -/
theorem symmetric_point_quadrant (m n : ℝ) : 
  (m = -2 ∧ m - n = -3) → m < 0 ∧ n > 0 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_quadrant_l1065_106549


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_64_cube_l1065_106510

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  side_count : ℕ
  total_cubes : ℕ
  inner_side_count : ℕ

/-- The number of small cubes with no painted faces in a cut cube -/
def unpainted_cubes (c : CutCube) : ℕ :=
  c.inner_side_count ^ 3

/-- Theorem: In a cube cut into 64 equal smaller cubes, 
    the number of small cubes with no painted faces is 8 -/
theorem unpainted_cubes_in_64_cube :
  ∃ c : CutCube, c.side_count = 4 ∧ c.total_cubes = 64 ∧ c.inner_side_count = 2 ∧ 
  unpainted_cubes c = 8 :=
sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_64_cube_l1065_106510


namespace NUMINAMATH_CALUDE_boys_girls_points_not_equal_l1065_106515

/-- Represents a round-robin chess tournament with boys and girls -/
structure ChessTournament where
  num_boys : Nat
  num_girls : Nat

/-- Calculate the total number of games in a round-robin tournament -/
def total_games (t : ChessTournament) : Nat :=
  (t.num_boys + t.num_girls) * (t.num_boys + t.num_girls - 1) / 2

/-- Calculate the number of games between boys -/
def boys_games (t : ChessTournament) : Nat :=
  t.num_boys * (t.num_boys - 1) / 2

/-- Calculate the number of games between girls -/
def girls_games (t : ChessTournament) : Nat :=
  t.num_girls * (t.num_girls - 1) / 2

/-- Calculate the number of games between boys and girls -/
def mixed_games (t : ChessTournament) : Nat :=
  t.num_boys * t.num_girls

/-- Theorem: In a round-robin chess tournament with 9 boys and 3 girls,
    the total points scored by all boys cannot equal the total points scored by all girls -/
theorem boys_girls_points_not_equal (t : ChessTournament) 
        (h1 : t.num_boys = 9) 
        (h2 : t.num_girls = 3) : 
        ¬ (boys_games t + mixed_games t / 2 = girls_games t + mixed_games t / 2) := by
  sorry

#eval boys_games ⟨9, 3⟩
#eval girls_games ⟨9, 3⟩
#eval mixed_games ⟨9, 3⟩

end NUMINAMATH_CALUDE_boys_girls_points_not_equal_l1065_106515


namespace NUMINAMATH_CALUDE_cos_960_degrees_l1065_106568

theorem cos_960_degrees : Real.cos (960 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_960_degrees_l1065_106568


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l1065_106535

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem largest_two_digit_prime_factor :
  ∃ (p : ℕ), is_prime p ∧ 
             p ≥ 10 ∧ p < 100 ∧
             p ∣ binomial_coefficient 300 150 ∧
             ∀ (q : ℕ), is_prime q → q ≥ 10 → q < 100 → q ∣ binomial_coefficient 300 150 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_l1065_106535


namespace NUMINAMATH_CALUDE_box_area_is_2144_l1065_106564

/-- The surface area of a box formed by removing square corners from a rectangular sheet. -/
def box_surface_area (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- Theorem stating that the surface area of the box is 2144 square units. -/
theorem box_area_is_2144 :
  box_surface_area 60 40 8 = 2144 :=
by sorry

end NUMINAMATH_CALUDE_box_area_is_2144_l1065_106564


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l1065_106586

/-- Given a circle with diameter endpoints (10, -6) and (-6, 2), 
    the sum of the coordinates of its center is 0. -/
theorem circle_center_coordinate_sum : 
  let x1 : ℝ := 10
  let y1 : ℝ := -6
  let x2 : ℝ := -6
  let y2 : ℝ := 2
  let center_x : ℝ := (x1 + x2) / 2
  let center_y : ℝ := (y1 + y2) / 2
  center_x + center_y = 0 := by sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l1065_106586


namespace NUMINAMATH_CALUDE_min_orange_weight_l1065_106580

theorem min_orange_weight (a o : ℝ) 
  (h1 : a ≥ 8 + 3 * o) 
  (h2 : a ≤ 4 * o) : 
  o ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_orange_weight_l1065_106580


namespace NUMINAMATH_CALUDE_no_natural_number_divisible_by_100_l1065_106548

theorem no_natural_number_divisible_by_100 : ∀ n : ℕ, ¬(100 ∣ (n^2 + 6*n + 2019)) := by
  sorry

end NUMINAMATH_CALUDE_no_natural_number_divisible_by_100_l1065_106548


namespace NUMINAMATH_CALUDE_clara_cookie_sales_l1065_106597

/-- Represents the number of cookies in each type of box --/
structure CookieBox where
  type1 : Nat
  type2 : Nat
  type3 : Nat

/-- Represents the number of boxes sold for each type --/
structure BoxesSold where
  type1 : Nat
  type2 : Nat
  type3 : Nat

/-- Calculates the total number of cookies sold --/
def totalCookiesSold (c : CookieBox) (b : BoxesSold) : Nat :=
  c.type1 * b.type1 + c.type2 * b.type2 + c.type3 * b.type3

theorem clara_cookie_sales (c : CookieBox) (b : BoxesSold) 
    (h1 : c.type1 = 12)
    (h2 : c.type2 = 20)
    (h3 : c.type3 = 16)
    (h4 : b.type2 = 80)
    (h5 : b.type3 = 70)
    (h6 : totalCookiesSold c b = 3320) :
    b.type1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_clara_cookie_sales_l1065_106597


namespace NUMINAMATH_CALUDE_modulus_of_z_l1065_106514

theorem modulus_of_z (i : ℂ) (h : i^2 = -1) : 
  let z : ℂ := i / (1 + i)
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1065_106514


namespace NUMINAMATH_CALUDE_susie_earnings_l1065_106574

/-- Calculates the total earnings from selling pizza slices and whole pizzas --/
def calculate_earnings (price_per_slice : ℚ) (price_per_whole : ℚ) (slices_sold : ℕ) (whole_sold : ℕ) : ℚ :=
  price_per_slice * slices_sold + price_per_whole * whole_sold

/-- Proves that Susie's earnings are $117 given the specified prices and sales --/
theorem susie_earnings : 
  let price_per_slice : ℚ := 3
  let price_per_whole : ℚ := 15
  let slices_sold : ℕ := 24
  let whole_sold : ℕ := 3
  calculate_earnings price_per_slice price_per_whole slices_sold whole_sold = 117 := by
  sorry

end NUMINAMATH_CALUDE_susie_earnings_l1065_106574


namespace NUMINAMATH_CALUDE_square_side_length_l1065_106585

/-- Given a square ABCD with specific points and conditions, prove its side length is 10 -/
theorem square_side_length (A B C D P Q R S Z : ℝ × ℝ) : 
  (∃ s : ℝ, 
    -- Square ABCD
    A = (0, 0) ∧ B = (s, 0) ∧ C = (s, s) ∧ D = (0, s) ∧
    -- P on AB, Q on BC, R on CD, S on DA
    (∃ t₁ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ P = (t₁ * s, 0)) ∧
    (∃ t₂ : ℝ, 0 ≤ t₂ ∧ t₂ ≤ 1 ∧ Q = (s, (1 - t₂) * s)) ∧
    (∃ t₃ : ℝ, 0 ≤ t₃ ∧ t₃ ≤ 1 ∧ R = ((1 - t₃) * s, s)) ∧
    (∃ t₄ : ℝ, 0 ≤ t₄ ∧ t₄ ≤ 1 ∧ S = (0, t₄ * s)) ∧
    -- PR parallel to BC, SQ parallel to AB
    (R.1 - P.1) * (C.2 - B.2) = (R.2 - P.2) * (C.1 - B.1) ∧
    (Q.1 - S.1) * (B.2 - A.2) = (Q.2 - S.2) * (B.1 - A.1) ∧
    -- Z is intersection of PR and SQ
    (Z.1 - P.1) * (R.2 - P.2) = (Z.2 - P.2) * (R.1 - P.1) ∧
    (Z.1 - S.1) * (Q.2 - S.2) = (Z.2 - S.2) * (Q.1 - S.1) ∧
    -- Given distances
    ‖B - P‖ = 7 ∧
    ‖B - Q‖ = 6 ∧
    ‖D - Z‖ = 5) →
  s = 10 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l1065_106585


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_l1065_106518

theorem smallest_five_digit_divisible_by_53 : ∀ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) → -- five-digit number condition
  n % 53 = 0 → -- divisibility by 53 condition
  n ≥ 10017 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_l1065_106518


namespace NUMINAMATH_CALUDE_product_pure_imaginary_implies_magnitude_l1065_106571

open Complex

theorem product_pure_imaginary_implies_magnitude (b : ℝ) :
  (((2 : ℂ) + b * I) * ((1 : ℂ) - I)).re = 0 ∧
  (((2 : ℂ) + b * I) * ((1 : ℂ) - I)).im ≠ 0 →
  abs ((1 : ℂ) + b * I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_implies_magnitude_l1065_106571


namespace NUMINAMATH_CALUDE_amount_after_two_years_l1065_106517

/-- The final amount after compound interest --/
def final_amount (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate) ^ years

/-- The problem statement --/
theorem amount_after_two_years :
  let initial := 2880
  let rate := 1 / 8
  let years := 2
  final_amount initial rate years = 3645 := by
sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l1065_106517


namespace NUMINAMATH_CALUDE_floor_length_calculation_l1065_106526

theorem floor_length_calculation (floor_width : ℝ) (strip_width : ℝ) (rug_area : ℝ) :
  floor_width = 20 →
  strip_width = 4 →
  rug_area = 204 →
  (floor_width - 2 * strip_width) * (floor_length - 2 * strip_width) = rug_area →
  floor_length = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_floor_length_calculation_l1065_106526


namespace NUMINAMATH_CALUDE_simultaneous_inequalities_condition_l1065_106536

theorem simultaneous_inequalities_condition (a b : ℝ) :
  (a > b ∧ 1 / a > 1 / b) ↔ (a > 0 ∧ 0 > b) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_inequalities_condition_l1065_106536


namespace NUMINAMATH_CALUDE_previous_day_visitors_l1065_106594

def total_visitors : ℕ := 406
def current_day_visitors : ℕ := 132

theorem previous_day_visitors : 
  total_visitors - current_day_visitors = 274 := by
  sorry

end NUMINAMATH_CALUDE_previous_day_visitors_l1065_106594


namespace NUMINAMATH_CALUDE_expand_product_l1065_106522

theorem expand_product (x a : ℝ) : 2 * (x + (a + 2)) * (x + (a - 3)) = 2 * x^2 + (4 * a - 2) * x + 2 * a^2 - 2 * a - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1065_106522


namespace NUMINAMATH_CALUDE_expanded_figure_perimeter_l1065_106533

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents the figure composed of squares -/
structure ExpandedFigure where
  squares : List Square
  bottomRowCount : ℕ
  topRowCount : ℕ

/-- Calculates the perimeter of the expanded figure -/
def perimeter (figure : ExpandedFigure) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem expanded_figure_perimeter :
  ∀ (figure : ExpandedFigure),
    (∀ s ∈ figure.squares, s.sideLength = 2) →
    figure.bottomRowCount = 3 →
    figure.topRowCount = 1 →
    figure.squares.length = 4 →
    perimeter figure = 20 :=
  sorry

end NUMINAMATH_CALUDE_expanded_figure_perimeter_l1065_106533


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l1065_106587

theorem sin_product_equals_one_sixteenth :
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * Real.sin (54 * π / 180) * Real.sin (72 * π / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l1065_106587


namespace NUMINAMATH_CALUDE_min_value_theorem_l1065_106525

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 1 → 1/(x-1) + 4/(y-1) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1065_106525


namespace NUMINAMATH_CALUDE_divisibility_property_l1065_106559

theorem divisibility_property (a b c : ℤ) (h : a + b + c = 0) :
  (∃ k : ℤ, a^4 + b^4 + c^4 = k * (a^2 + b^2 + c^2)) ∧
  (∃ m : ℤ, a^100 + b^100 + c^100 = m * (a^2 + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1065_106559


namespace NUMINAMATH_CALUDE_donut_hole_count_donut_hole_count_proof_l1065_106579

/-- The number of donut holes Nira will have coated when all three workers finish simultaneously -/
theorem donut_hole_count : ℕ :=
  let nira_radius : ℝ := 5
  let theo_radius : ℝ := 7
  let kaira_side : ℝ := 6
  let nira_surface_area : ℝ := 4 * Real.pi * nira_radius ^ 2
  let theo_surface_area : ℝ := 4 * Real.pi * theo_radius ^ 2
  let kaira_surface_area : ℝ := 6 * kaira_side ^ 2
  5292

/-- Proof that Nira will have coated 5292 donut holes when all three workers finish simultaneously -/
theorem donut_hole_count_proof : donut_hole_count = 5292 := by
  sorry

end NUMINAMATH_CALUDE_donut_hole_count_donut_hole_count_proof_l1065_106579


namespace NUMINAMATH_CALUDE_average_age_proof_l1065_106555

def luke_age : ℕ := 20
def years_future : ℕ := 8

theorem average_age_proof :
  let bernard_future_age := 3 * luke_age
  let bernard_current_age := bernard_future_age - years_future
  let average_age := (luke_age + bernard_current_age) / 2
  average_age = 36 := by sorry

end NUMINAMATH_CALUDE_average_age_proof_l1065_106555


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1065_106523

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (5*x - 4)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1065_106523


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l1065_106599

theorem consecutive_pages_sum (x y : ℕ) : 
  x + y = 125 → y = x + 1 → y = 63 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l1065_106599


namespace NUMINAMATH_CALUDE_company_fund_problem_l1065_106502

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  initial_fund = 60 * n - 10 →  -- The fund initially contained $10 less than needed for $60 bonuses
  initial_fund = 55 * n + 120 → -- Each employee received $55, and $120 remained
  initial_fund = 1550 :=        -- The initial fund amount was $1550
by sorry

end NUMINAMATH_CALUDE_company_fund_problem_l1065_106502


namespace NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l1065_106563

-- Define the approximate value of the upper bound
def upperBound : ℝ := 0.54

-- State the theorem
theorem arccos_gt_arctan_iff (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 1 →
  Real.arccos x > Real.arctan x ↔ x ∈ Set.Icc (-1 : ℝ) upperBound :=
by sorry

end NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l1065_106563


namespace NUMINAMATH_CALUDE_players_joined_equals_two_l1065_106513

/-- The number of players who joined an online game --/
def players_joined (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  (total_lives / lives_per_player) - initial_players

/-- Theorem: The number of players who joined the game is 2 --/
theorem players_joined_equals_two :
  players_joined 2 6 24 = 2 := by
  sorry

end NUMINAMATH_CALUDE_players_joined_equals_two_l1065_106513


namespace NUMINAMATH_CALUDE_log_and_perpendicular_lines_l1065_106551

theorem log_and_perpendicular_lines (S T : ℝ) : 
  (Real.log S / Real.log 9 = 3/2) →
  ((1 : ℝ) * ((-S : ℝ)) + 5 * T = 0) →
  (S = 27 ∧ T = 135) := by sorry

end NUMINAMATH_CALUDE_log_and_perpendicular_lines_l1065_106551


namespace NUMINAMATH_CALUDE_multiples_of_15_between_12_and_202_l1065_106565

theorem multiples_of_15_between_12_and_202 : 
  (Finset.filter (fun n => n % 15 = 0 ∧ n > 12 ∧ n < 202) (Finset.range 202)).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_12_and_202_l1065_106565


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l1065_106511

theorem matrix_multiplication_result :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 2]
  A * B = !![23, -7; 24, -16] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l1065_106511


namespace NUMINAMATH_CALUDE_covered_area_is_56_l1065_106584

/-- Represents a rectangular strip of paper -/
structure Strip where
  length : ℝ
  width : ℝ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℝ := s.length * s.width

/-- Calculates the area of intersection between two perpendicular strips -/
def intersectionArea (s1 s2 : Strip) : ℝ := s1.width * s2.width

/-- Represents the arrangement of strips on the table -/
structure StripArrangement where
  horizontalStrips : Fin 3 → Strip
  verticalStrips : Fin 2 → Strip
  all_strips_same : ∀ (i : Fin 3) (j : Fin 2), 
    (horizontalStrips i).length = 8 ∧ (horizontalStrips i).width = 2 ∧
    (verticalStrips j).length = 8 ∧ (verticalStrips j).width = 2

/-- Calculates the total area covered by the strips -/
def coveredArea (arr : StripArrangement) : ℝ :=
  let totalStripArea := (3 * stripArea (arr.horizontalStrips 0)) + (2 * stripArea (arr.verticalStrips 0))
  let totalOverlapArea := 6 * intersectionArea (arr.horizontalStrips 0) (arr.verticalStrips 0)
  totalStripArea - totalOverlapArea

/-- Theorem stating that the covered area is 56 square units -/
theorem covered_area_is_56 (arr : StripArrangement) : coveredArea arr = 56 := by
  sorry

end NUMINAMATH_CALUDE_covered_area_is_56_l1065_106584


namespace NUMINAMATH_CALUDE_parabola_sum_coefficients_l1065_106562

/-- A parabola with equation x = ay^2 + by + c, vertex at (-3, 2), and passing through (-1, 0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_condition : -3 = a * 2^2 + b * 2 + c
  point_condition : -1 = a * 0^2 + b * 0 + c

/-- The sum of coefficients a, b, and c for the given parabola is -7/2 -/
theorem parabola_sum_coefficients (p : Parabola) : p.a + p.b + p.c = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_coefficients_l1065_106562


namespace NUMINAMATH_CALUDE_garys_money_l1065_106547

/-- Gary's initial amount of money -/
def initial_amount : ℕ := sorry

/-- Amount Gary spent on the snake -/
def spent_amount : ℕ := 55

/-- Amount Gary had left after buying the snake -/
def remaining_amount : ℕ := 18

/-- Theorem stating that Gary's initial amount equals the sum of spent and remaining amounts -/
theorem garys_money : initial_amount = spent_amount + remaining_amount := by sorry

end NUMINAMATH_CALUDE_garys_money_l1065_106547


namespace NUMINAMATH_CALUDE_average_of_last_part_calculation_l1065_106505

def average_of_last_part (total_count : ℕ) (total_average : ℚ) (first_part_count : ℕ) (first_part_average : ℚ) (middle_result : ℚ) : ℚ :=
  let last_part_count := total_count - first_part_count - 1
  let total_sum := total_count * total_average
  let first_part_sum := first_part_count * first_part_average
  (total_sum - first_part_sum - middle_result) / last_part_count

theorem average_of_last_part_calculation :
  average_of_last_part 25 50 12 14 878 = 204 / 13 := by
  sorry

end NUMINAMATH_CALUDE_average_of_last_part_calculation_l1065_106505


namespace NUMINAMATH_CALUDE_combined_fuel_efficiency_l1065_106524

theorem combined_fuel_efficiency
  (m : ℝ) -- distance driven by each car
  (h_pos : m > 0) -- ensure distance is positive
  (efficiency_linda : ℝ := 30) -- Linda's car efficiency
  (efficiency_joe : ℝ := 15) -- Joe's car efficiency
  (efficiency_anne : ℝ := 20) -- Anne's car efficiency
  : (3 * m) / (m / efficiency_linda + m / efficiency_joe + m / efficiency_anne) = 20 :=
by sorry

end NUMINAMATH_CALUDE_combined_fuel_efficiency_l1065_106524


namespace NUMINAMATH_CALUDE_wire_length_proof_l1065_106575

theorem wire_length_proof (total_wires : ℕ) (overall_avg : ℝ) (long_wires : ℕ) (long_avg : ℝ) :
  total_wires = 6 →
  overall_avg = 80 →
  long_wires = 4 →
  long_avg = 85 →
  let short_wires := total_wires - long_wires
  let short_avg := (total_wires * overall_avg - long_wires * long_avg) / short_wires
  short_avg = 70 := by sorry

end NUMINAMATH_CALUDE_wire_length_proof_l1065_106575


namespace NUMINAMATH_CALUDE_value_after_seven_years_l1065_106519

/-- Calculates the value after n years given initial value, annual increase rate, inflation rate, and tax rate -/
def value_after_years (initial_value : ℝ) (increase_rate : ℝ) (inflation_rate : ℝ) (tax_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * ((1 - tax_rate) * (1 - inflation_rate) * (1 + increase_rate)) ^ years

/-- Theorem stating that the value after 7 years is approximately 126469.75 -/
theorem value_after_seven_years :
  let initial_value : ℝ := 59000
  let increase_rate : ℝ := 1/8
  let inflation_rate : ℝ := 0.03
  let tax_rate : ℝ := 0.07
  let years : ℕ := 7
  abs (value_after_years initial_value increase_rate inflation_rate tax_rate years - 126469.75) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_value_after_seven_years_l1065_106519


namespace NUMINAMATH_CALUDE_rectangle_width_equals_square_side_l1065_106546

/-- The width of a rectangle with length 4 cm and area equal to a square with sides 4 cm is 4 cm. -/
theorem rectangle_width_equals_square_side {width : ℝ} (h : width > 0) : 
  4 * width = 4 * 4 → width = 4 := by
  sorry

#check rectangle_width_equals_square_side

end NUMINAMATH_CALUDE_rectangle_width_equals_square_side_l1065_106546


namespace NUMINAMATH_CALUDE_sin_sum_identity_l1065_106537

theorem sin_sum_identity (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 3) :
  Real.sin (x - 5 * π / 6) + Real.sin (π / 3 - x) ^ 2 = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_identity_l1065_106537


namespace NUMINAMATH_CALUDE_characterize_S_l1065_106538

/-- The function f(A, B, C) = A^3 + B^3 + C^3 - 3ABC -/
def f (A B C : ℕ) : ℤ := A^3 + B^3 + C^3 - 3 * A * B * C

/-- The set of all possible values of f(A, B, C) -/
def S : Set ℤ := {n | ∃ (A B C : ℕ), f A B C = n}

/-- The theorem stating the characterization of S -/
theorem characterize_S : S = {n : ℤ | n ≥ 0 ∧ n % 9 ≠ 3 ∧ n % 9 ≠ 6} := by sorry

end NUMINAMATH_CALUDE_characterize_S_l1065_106538


namespace NUMINAMATH_CALUDE_range_of_a_l1065_106507

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a^2 - 4

-- State the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x ∈ Set.Icc (a - 2) (a^2), f a x ∈ Set.Icc (-4) 0) →
  a ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1065_106507


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1065_106583

theorem complex_equation_solution (a b : ℝ) (h : b ≠ 0) :
  (Complex.I : ℂ)^2 = -1 →
  (a + b * Complex.I)^2 = -b * Complex.I →
  (a = -1/2 ∧ (b = 1/2 ∨ b = -1/2)) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1065_106583


namespace NUMINAMATH_CALUDE_single_elimination_512_players_games_l1065_106578

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_players : ℕ
  single_elimination : Bool

/-- Calculates the number of games required to determine a champion in a single-elimination tournament. -/
def games_required (t : Tournament) : ℕ :=
  if t.single_elimination then t.num_players - 1 else 0

/-- Theorem stating that a single-elimination tournament with 512 players requires 511 games. -/
theorem single_elimination_512_players_games (t : Tournament) 
  (h1 : t.num_players = 512) 
  (h2 : t.single_elimination = true) : 
  games_required t = 511 := by
  sorry

#eval games_required ⟨512, true⟩

end NUMINAMATH_CALUDE_single_elimination_512_players_games_l1065_106578


namespace NUMINAMATH_CALUDE_line_relationships_l1065_106516

-- Define a type for lines in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  -- For simplicity, we'll just use an opaque type
  mk :: (dummy : Unit)

-- Define the relationships between lines
def parallel (l1 l2 : Line3D) : Prop := sorry

def intersects (l1 l2 : Line3D) : Prop := sorry

def skew (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem line_relationships (a b c : Line3D) 
  (h1 : parallel a b) 
  (h2 : intersects a c) :
  skew b c ∨ intersects b c := by sorry

end NUMINAMATH_CALUDE_line_relationships_l1065_106516


namespace NUMINAMATH_CALUDE_cookies_milk_proportion_l1065_106532

/-- Given that 24 cookies require 5 quarts of milk and 1 quart equals 4 cups,
    prove that 8 cookies require 20/3 cups of milk. -/
theorem cookies_milk_proportion :
  let cookies_24 : ℕ := 24
  let quarts_24 : ℕ := 5
  let cups_per_quart : ℕ := 4
  let cookies_8 : ℕ := 8
  cookies_24 * (cups_per_quart * quarts_24) / cookies_8 = 20 / 3 := by sorry

end NUMINAMATH_CALUDE_cookies_milk_proportion_l1065_106532


namespace NUMINAMATH_CALUDE_race_cars_alignment_l1065_106506

theorem race_cars_alignment (a b c : ℕ) (ha : a = 28) (hb : b = 24) (hc : c = 32) :
  Nat.lcm (Nat.lcm a b) c = 672 := by
  sorry

end NUMINAMATH_CALUDE_race_cars_alignment_l1065_106506


namespace NUMINAMATH_CALUDE_sibling_pair_probability_l1065_106581

theorem sibling_pair_probability 
  (business_students : ℕ) 
  (law_students : ℕ) 
  (sibling_pairs : ℕ) 
  (h1 : business_students = 500) 
  (h2 : law_students = 800) 
  (h3 : sibling_pairs = 30) : 
  (sibling_pairs : ℚ) / (business_students * law_students) = 30 / 400000 := by
  sorry

end NUMINAMATH_CALUDE_sibling_pair_probability_l1065_106581


namespace NUMINAMATH_CALUDE_least_common_multiple_9_6_l1065_106595

theorem least_common_multiple_9_6 : Nat.lcm 9 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_9_6_l1065_106595


namespace NUMINAMATH_CALUDE_sixteen_equal_parts_l1065_106553

/-- Represents a rectangular frame with a hollow space inside -/
structure RectangularFrame where
  height : ℝ
  width : ℝ
  hollow : Bool

/-- Represents a division of a rectangular frame -/
structure FrameDivision where
  horizontal_cuts : ℕ
  vertical_cuts : ℕ

/-- Calculates the number of parts resulting from a frame division -/
def number_of_parts (d : FrameDivision) : ℕ :=
  (d.horizontal_cuts + 1) * (d.vertical_cuts + 1)

/-- Theorem stating that one horizontal cut and seven vertical cuts result in 16 equal parts -/
theorem sixteen_equal_parts (f : RectangularFrame) :
  let d := FrameDivision.mk 1 7
  number_of_parts d = 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_equal_parts_l1065_106553


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l1065_106550

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 3*x ∧ 6*s^2 = 6*x) → x = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l1065_106550


namespace NUMINAMATH_CALUDE_intersection_theorem_l1065_106566

-- Define the sets A and B
def A : Set ℝ := {x | x < -3 ∨ x > 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 4}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the expected result
def expected_result : Set ℝ := {x | 1 < x ∧ x ≤ 4}

-- State the theorem
theorem intersection_theorem : A_intersect_B = expected_result := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l1065_106566


namespace NUMINAMATH_CALUDE_same_color_probability_l1065_106504

/-- Represents the number of sides on each die -/
def totalSides : ℕ := 20

/-- Represents the number of orange sides on each die -/
def orangeSides : ℕ := 3

/-- Represents the number of purple sides on each die -/
def purpleSides : ℕ := 5

/-- Represents the number of green sides on each die -/
def greenSides : ℕ := 6

/-- Represents the number of blue sides on each die -/
def blueSides : ℕ := 5

/-- Represents the number of sparkly sides on each die -/
def sparklySides : ℕ := 1

/-- Theorem stating the probability of rolling the same color or shade on both dice -/
theorem same_color_probability : 
  (orangeSides^2 + purpleSides^2 + greenSides^2 + blueSides^2 + sparklySides^2) / totalSides^2 = 24 / 100 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1065_106504


namespace NUMINAMATH_CALUDE_simultaneous_arrival_l1065_106582

theorem simultaneous_arrival (total_distance : ℝ) (alyosha_walk_speed alyosha_cycle_speed vitia_walk_speed vitia_cycle_speed : ℝ) 
  (h1 : total_distance = 20)
  (h2 : alyosha_walk_speed = 4)
  (h3 : alyosha_cycle_speed = 15)
  (h4 : vitia_walk_speed = 5)
  (h5 : vitia_cycle_speed = 20)
  : ∃ x : ℝ, 
    x / alyosha_cycle_speed + (total_distance - x) / alyosha_walk_speed = 
    x / vitia_walk_speed + (total_distance - x) / vitia_cycle_speed ∧ 
    x = 12 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_arrival_l1065_106582


namespace NUMINAMATH_CALUDE_existence_of_multiple_2002_l1065_106556

theorem existence_of_multiple_2002 (a : Fin 41 → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ i j m n p q : Fin 41, i ≠ j ∧ m ≠ n ∧ p ≠ q ∧
    i ≠ m ∧ i ≠ n ∧ i ≠ p ∧ i ≠ q ∧
    j ≠ m ∧ j ≠ n ∧ j ≠ p ∧ j ≠ q ∧
    m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧
    (2002 ∣ (a i - a j) * (a m - a n) * (a p - a q)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_multiple_2002_l1065_106556


namespace NUMINAMATH_CALUDE_sqrt_27_plus_sqrt_75_l1065_106530

theorem sqrt_27_plus_sqrt_75 : Real.sqrt 27 + Real.sqrt 75 = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_plus_sqrt_75_l1065_106530
