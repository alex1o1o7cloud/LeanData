import Mathlib

namespace NUMINAMATH_CALUDE_minimum_shoeing_time_l143_14309

/-- The minimum time required for blacksmiths to shoe horses -/
theorem minimum_shoeing_time
  (num_blacksmiths : ℕ)
  (num_horses : ℕ)
  (time_per_hoof : ℕ)
  (hooves_per_horse : ℕ)
  (h1 : num_blacksmiths = 48)
  (h2 : num_horses = 60)
  (h3 : time_per_hoof = 5)
  (h4 : hooves_per_horse = 4) :
  (num_horses * hooves_per_horse * time_per_hoof) / num_blacksmiths = 25 := by
  sorry

#eval (60 * 4 * 5) / 48  -- Should output 25

end NUMINAMATH_CALUDE_minimum_shoeing_time_l143_14309


namespace NUMINAMATH_CALUDE_nine_twelve_fifteen_pythagorean_triple_l143_14312

/-- A Pythagorean triple is a set of three positive integers (a, b, c) such that a² + b² = c² --/
def is_pythagorean_triple (a b c : ℕ) : Prop := a * a + b * b = c * c

/-- Prove that (9, 12, 15) is a Pythagorean triple --/
theorem nine_twelve_fifteen_pythagorean_triple : is_pythagorean_triple 9 12 15 := by
  sorry

end NUMINAMATH_CALUDE_nine_twelve_fifteen_pythagorean_triple_l143_14312


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l143_14354

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + k - 1 ≠ 0) → k > 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l143_14354


namespace NUMINAMATH_CALUDE_some_number_value_l143_14372

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 25 * 315 * 7) : n = 63 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l143_14372


namespace NUMINAMATH_CALUDE_triangle_sine_sum_maximized_equilateral_maximizes_sine_sum_l143_14397

open Real

theorem triangle_sine_sum_maximized (α β γ : ℝ) : 
  0 < α ∧ 0 < β ∧ 0 < γ →
  α + β + γ = π →
  sin α + sin β + sin γ ≤ 3 * sin (π / 3) :=
sorry

theorem equilateral_maximizes_sine_sum (α β γ : ℝ) :
  0 < α ∧ 0 < β ∧ 0 < γ →
  α + β + γ = π →
  sin α + sin β + sin γ = 3 * sin (π / 3) ↔ α = β ∧ β = γ :=
sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_maximized_equilateral_maximizes_sine_sum_l143_14397


namespace NUMINAMATH_CALUDE_shortest_side_of_similar_triangle_l143_14331

/-- Given two similar right triangles, where the first triangle has a side length of 30 inches
    and a hypotenuse of 34 inches, and the second triangle has a hypotenuse of 102 inches,
    the shortest side of the second triangle is 48 inches. -/
theorem shortest_side_of_similar_triangle (a b c : ℝ) : 
  a^2 + b^2 = 34^2 →  -- Pythagorean theorem for the first triangle
  a = 30 →            -- Given side length of the first triangle
  b ≤ a →             -- b is the shortest side of the first triangle
  c^2 + (3*b)^2 = 102^2 →  -- Pythagorean theorem for the second triangle
  3*b = 48 :=         -- The shortest side of the second triangle is 48 inches
by sorry

end NUMINAMATH_CALUDE_shortest_side_of_similar_triangle_l143_14331


namespace NUMINAMATH_CALUDE_parallel_vectors_iff_k_eq_neg_two_l143_14375

-- Define the vectors
def a : Fin 2 → ℝ := ![1, -2]
def b (k : ℝ) : Fin 2 → ℝ := ![k, 4]

-- Define parallel vectors
def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ (∀ i, u i = c * v i)

-- Theorem statement
theorem parallel_vectors_iff_k_eq_neg_two :
  parallel a (b k) ↔ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_iff_k_eq_neg_two_l143_14375


namespace NUMINAMATH_CALUDE_tank_depth_is_six_l143_14368

-- Define the tank dimensions and plastering cost
def tankLength : ℝ := 25
def tankWidth : ℝ := 12
def plasteringCostPerSqM : ℝ := 0.45
def totalPlasteringCost : ℝ := 334.8

-- Define the function to calculate the total surface area to be plastered
def surfaceArea (depth : ℝ) : ℝ :=
  tankLength * tankWidth + 2 * (tankLength * depth) + 2 * (tankWidth * depth)

-- Theorem statement
theorem tank_depth_is_six :
  ∃ (depth : ℝ), plasteringCostPerSqM * surfaceArea depth = totalPlasteringCost ∧ depth = 6 :=
sorry

end NUMINAMATH_CALUDE_tank_depth_is_six_l143_14368


namespace NUMINAMATH_CALUDE_solid_shapes_count_l143_14366

-- Define the set of geometric shapes
inductive GeometricShape
  | Square
  | Cuboid
  | Circle
  | Sphere
  | Cone

-- Define a function to determine if a shape is solid
def isSolid (shape : GeometricShape) : Bool :=
  match shape with
  | GeometricShape.Square => false
  | GeometricShape.Cuboid => true
  | GeometricShape.Circle => false
  | GeometricShape.Sphere => true
  | GeometricShape.Cone => true

-- Define the list of given shapes
def givenShapes : List GeometricShape :=
  [GeometricShape.Square, GeometricShape.Cuboid, GeometricShape.Circle, GeometricShape.Sphere, GeometricShape.Cone]

-- Theorem statement
theorem solid_shapes_count :
  (givenShapes.filter isSolid).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_solid_shapes_count_l143_14366


namespace NUMINAMATH_CALUDE_difference_of_squares_a_l143_14326

theorem difference_of_squares_a (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_a_l143_14326


namespace NUMINAMATH_CALUDE_max_value_at_13_l143_14308

-- Define the function f(x) = x - 5
def f (x : ℝ) : ℝ := x - 5

-- Theorem statement
theorem max_value_at_13 :
  ∃ (x : ℝ), x ≤ 13 ∧ ∀ (y : ℝ), y ≤ 13 → f y ≤ f x ∧ f x = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_at_13_l143_14308


namespace NUMINAMATH_CALUDE_vinegar_mixture_percentage_l143_14347

/-- The percentage of the second vinegar solution -/
def P : ℝ := 40

/-- Volume of each initial solution in milliliters -/
def initial_volume : ℝ := 10

/-- Volume of the final mixture in milliliters -/
def final_volume : ℝ := 50

/-- Percentage of the first solution -/
def first_percentage : ℝ := 5

/-- Percentage of the final mixture -/
def final_percentage : ℝ := 9

theorem vinegar_mixture_percentage :
  initial_volume * (first_percentage / 100) +
  initial_volume * (P / 100) =
  final_volume * (final_percentage / 100) :=
sorry

end NUMINAMATH_CALUDE_vinegar_mixture_percentage_l143_14347


namespace NUMINAMATH_CALUDE_relationship_abc_l143_14344

theorem relationship_abc :
  ∀ (a b c : ℝ), a = 2 → b = 3 → c = 4 → c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l143_14344


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l143_14341

/-- Given two points A and B on a number line, where A represents -3 and B is obtained by moving 7 units to the right from A, prove that B represents 4. -/
theorem point_movement_on_number_line (A B : ℝ) : A = -3 ∧ B = A + 7 → B = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l143_14341


namespace NUMINAMATH_CALUDE_perimeter_is_60_l143_14348

/-- Square with side length 9 inches -/
def square_side_length : ℝ := 9

/-- Equilateral triangle with side length equal to square's side length -/
def triangle_side_length : ℝ := square_side_length

/-- Figure ABFCE formed after translating the triangle -/
structure Figure where
  AB : ℝ := square_side_length
  BF : ℝ := triangle_side_length
  FC : ℝ := triangle_side_length
  CE : ℝ := square_side_length
  EA : ℝ := square_side_length

/-- Perimeter of the figure ABFCE -/
def perimeter (fig : Figure) : ℝ :=
  fig.AB + fig.BF + fig.FC + fig.CE + fig.EA

/-- Theorem: The perimeter of figure ABFCE is 60 inches -/
theorem perimeter_is_60 (fig : Figure) : perimeter fig = 60 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_60_l143_14348


namespace NUMINAMATH_CALUDE_function_properties_l143_14314

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 + 2*b*x

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
  (∀ x, f a b x ≥ f a b 1) ∧ 
  (f a b 1 = -1) ∧
  (a = 1/3) ∧ 
  (b = -1/2) ∧
  (∀ x, x ≤ -1/3 ∨ x ≥ 1 → (deriv (f a b)) x ≥ 0) ∧
  (∀ x, -1/3 ≤ x ∧ x ≤ 1 → (deriv (f a b)) x ≤ 0) ∧
  (∀ α, -1 < α ∧ α < 5/27 → ∃ x y z, x < y ∧ y < z ∧ f a b x = α ∧ f a b y = α ∧ f a b z = α) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l143_14314


namespace NUMINAMATH_CALUDE_conor_weekly_vegetables_l143_14323

/-- Represents Conor's vegetable chopping capacity and work schedule --/
structure VegetableChopper where
  eggplants_per_day : ℕ
  carrots_per_day : ℕ
  potatoes_per_day : ℕ
  work_days_per_week : ℕ

/-- Calculates the total number of vegetables chopped in a week --/
def total_vegetables_per_week (c : VegetableChopper) : ℕ :=
  (c.eggplants_per_day + c.carrots_per_day + c.potatoes_per_day) * c.work_days_per_week

/-- Theorem stating that Conor can chop 116 vegetables in a week --/
theorem conor_weekly_vegetables :
  ∃ c : VegetableChopper,
    c.eggplants_per_day = 12 ∧
    c.carrots_per_day = 9 ∧
    c.potatoes_per_day = 8 ∧
    c.work_days_per_week = 4 ∧
    total_vegetables_per_week c = 116 :=
by
  sorry

end NUMINAMATH_CALUDE_conor_weekly_vegetables_l143_14323


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l143_14355

theorem diophantine_equation_solutions (x y : ℕ+) :
  x^(y : ℕ) = y^(x : ℕ) + 1 ↔ (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l143_14355


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l143_14333

theorem rectangle_perimeter (x y : ℝ) 
  (rachel_sum : 2 * x + y = 44)
  (heather_sum : x + 2 * y = 40) : 
  2 * (x + y) = 56 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l143_14333


namespace NUMINAMATH_CALUDE_garden_internal_boundary_length_l143_14321

/-- Represents a square plot in the garden -/
structure Plot where
  side : ℕ
  deriving Repr

/-- Represents the garden configuration -/
structure Garden where
  width : ℕ
  height : ℕ
  plots : List Plot
  deriving Repr

/-- Calculates the total area of the garden -/
def gardenArea (g : Garden) : ℕ := g.width * g.height

/-- Calculates the area of a single plot -/
def plotArea (p : Plot) : ℕ := p.side * p.side

/-- Calculates the perimeter of a single plot -/
def plotPerimeter (p : Plot) : ℕ := 4 * p.side

/-- Calculates the external boundary of the garden -/
def externalBoundary (g : Garden) : ℕ := 2 * (g.width + g.height)

/-- The main theorem to prove -/
theorem garden_internal_boundary_length 
  (g : Garden) 
  (h1 : g.width = 6) 
  (h2 : g.height = 7) 
  (h3 : g.plots.length = 5) 
  (h4 : ∀ p ∈ g.plots, ∃ n : ℕ, p.side = n) 
  (h5 : (g.plots.map plotArea).sum = gardenArea g) : 
  ((g.plots.map plotPerimeter).sum - externalBoundary g) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_garden_internal_boundary_length_l143_14321


namespace NUMINAMATH_CALUDE_binomial_linear_transform_l143_14361

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  (p_nonneg : 0 ≤ p)
  (p_le_one : p ≤ 1)

/-- Expected value of a binomial random variable -/
def expected_value (X : BinomialRV n p) : ℝ :=
  n * p

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV n p) : ℝ :=
  n * p * (1 - p)

/-- Theorem: Expected value and variance of η = 5ξ, where ξ ~ B(5, 0.5) -/
theorem binomial_linear_transform :
  ∀ (ξ : BinomialRV 5 (1/2)) (η : ℝ),
  η = 5 * (expected_value ξ) →
  expected_value ξ = 5/2 ∧
  variance ξ = 5/4 ∧
  η = 25/2 ∧
  25 * (variance ξ) = 125/4 :=
sorry

end NUMINAMATH_CALUDE_binomial_linear_transform_l143_14361


namespace NUMINAMATH_CALUDE_ed_lost_no_marbles_l143_14367

def marbles_lost (ed_initial : ℕ) (ed_now : ℕ) (doug : ℕ) : ℕ :=
  ed_initial - ed_now

theorem ed_lost_no_marbles 
  (h1 : ∃ ed_initial : ℕ, ed_initial = doug + 12)
  (h2 : ∃ ed_now : ℕ, ed_now = 17)
  (h3 : doug = 5) :
  marbles_lost (doug + 12) 17 doug = 0 := by
  sorry

end NUMINAMATH_CALUDE_ed_lost_no_marbles_l143_14367


namespace NUMINAMATH_CALUDE_f_properties_l143_14340

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

theorem f_properties (a : ℝ) :
  -- Part 1
  (a = -1/2 →
    ∃ (max min : ℝ),
      (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x ≤ max) ∧
      (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x = max) ∧
      (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x ≥ min) ∧
      (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x = min) ∧
      max = 1/2 + (Real.exp 1)^2/4 ∧
      min = 5/4) ∧
  -- Part 2
  (∃ (mono : ℝ → Prop), mono a ↔
    (∀ x y, 0 < x → 0 < y → x < y → (f a x < f a y ∨ f a x > f a y ∨ f a x = f a y))) ∧
  -- Part 3
  (-1 < a → a < 0 →
    (∀ x, x > 0 → f a x > 1 + a/2 * Real.log (-a)) ∧
    (1/Real.exp 1 - 1 < a ∧ a < 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l143_14340


namespace NUMINAMATH_CALUDE_truck_driver_speed_l143_14325

/-- A truck driver's problem -/
theorem truck_driver_speed 
  (gas_cost : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (total_pay : ℝ)
  (total_hours : ℝ)
  (h1 : gas_cost = 2)
  (h2 : fuel_efficiency = 10)
  (h3 : pay_rate = 0.5)
  (h4 : total_pay = 90)
  (h5 : total_hours = 10)
  : (total_pay / pay_rate) / total_hours = 18 := by
  sorry


end NUMINAMATH_CALUDE_truck_driver_speed_l143_14325


namespace NUMINAMATH_CALUDE_triangle_theorem_l143_14345

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the triangle ABC -/
theorem triangle_theorem (t : Triangle) :
  (Real.sin t.B)^2 = (Real.sin t.A)^2 + (Real.sin t.C)^2 - Real.sin t.A * Real.sin t.C →
  t.B = π / 3 ∧
  (t.b = Real.sqrt 3 ∧ t.a * t.c * Real.sin t.B / 2 = Real.sqrt 3 / 2 →
    t.a + t.c = 3 ∧
    -t.a * t.c * Real.cos t.B = -1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l143_14345


namespace NUMINAMATH_CALUDE_gummy_juice_time_proof_l143_14395

/-- Represents the time in hours since starting to mow the lawn -/
def DrinkTime : ℝ := 1.5

theorem gummy_juice_time_proof :
  let total_time : ℝ := 2.5 -- Total time spent mowing (from 10:00 AM to 12:30 PM)
  let normal_rate : ℝ := 1 / 3 -- Rate of mowing without juice (1/3 of lawn per hour)
  let boosted_rate : ℝ := 1 / 2 -- Rate of mowing with juice (1/2 of lawn per hour)
  let normal_portion : ℝ := DrinkTime * normal_rate -- Portion mowed without juice
  let boosted_portion : ℝ := (total_time - DrinkTime) * boosted_rate -- Portion mowed with juice
  normal_portion + boosted_portion = 1 -- Total lawn mowed equals 1
  ∧ DrinkTime = 1.5 -- Time when Bronquinha drank the juice (1.5 hours after 10:00 AM, which is 11:30 AM)
  := by sorry

#check gummy_juice_time_proof

end NUMINAMATH_CALUDE_gummy_juice_time_proof_l143_14395


namespace NUMINAMATH_CALUDE_triangle_side_values_l143_14378

theorem triangle_side_values (y : ℕ+) : 
  (∃ (a b c : ℝ), a = 8 ∧ b = 11 ∧ c = y.val ^ 2 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  (y = 2 ∨ y = 3 ∨ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l143_14378


namespace NUMINAMATH_CALUDE_cube_edge_length_is_15_l143_14335

/-- The edge length of a cube that displaces a specific volume of water -/
def cube_edge_length (base_length base_width water_rise : ℝ) : ℝ :=
  (base_length * base_width * water_rise) ^ (1/3)

/-- Theorem stating that a cube with the given specifications has an edge length of 15 cm -/
theorem cube_edge_length_is_15 :
  cube_edge_length 20 14 12.053571428571429 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_is_15_l143_14335


namespace NUMINAMATH_CALUDE_floor_sum_rationality_l143_14399

theorem floor_sum_rationality (p q r : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h : ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, ⌊p * n⌋ + ⌊q * n⌋ + ⌊r * n⌋ = n) :
  (∃ a b c : ℤ, p = a / b ∧ q = a / c) ∧ p + q + r = 1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_rationality_l143_14399


namespace NUMINAMATH_CALUDE_resort_worker_period_l143_14360

theorem resort_worker_period (average_tips : ℝ) (total_period : ℕ) : 
  (6 * average_tips = (1 / 2) * (6 * average_tips + (total_period - 1) * average_tips)) →
  total_period = 7 := by
  sorry

end NUMINAMATH_CALUDE_resort_worker_period_l143_14360


namespace NUMINAMATH_CALUDE_ramp_installation_cost_l143_14359

/-- Calculates the total cost of installing a ramp given specific conditions --/
theorem ramp_installation_cost :
  let permit_base_cost : ℝ := 250
  let permit_tax_rate : ℝ := 0.1
  let contractor_labor_rate : ℝ := 150
  let raw_materials_rate : ℝ := 50
  let work_days : ℕ := 3
  let work_hours_per_day : ℝ := 5
  let tool_rental_rate : ℝ := 30
  let lunch_break_hours : ℝ := 0.5
  let raw_materials_markup : ℝ := 0.15
  let inspector_rate_discount : ℝ := 0.8
  let inspector_hours_per_day : ℝ := 2

  let permit_cost : ℝ := permit_base_cost * (1 + permit_tax_rate)
  let raw_materials_cost_with_markup : ℝ := raw_materials_rate * (1 + raw_materials_markup)
  let contractor_hourly_cost : ℝ := contractor_labor_rate + raw_materials_cost_with_markup
  let total_work_hours : ℝ := work_days * work_hours_per_day
  let total_lunch_hours : ℝ := work_days * lunch_break_hours
  let tool_rental_cost : ℝ := tool_rental_rate * work_days
  let contractor_cost : ℝ := contractor_hourly_cost * (total_work_hours - total_lunch_hours) + tool_rental_cost
  let inspector_rate : ℝ := contractor_labor_rate * (1 - inspector_rate_discount)
  let inspector_cost : ℝ := inspector_rate * inspector_hours_per_day * work_days

  let total_cost : ℝ := permit_cost + contractor_cost + inspector_cost

  total_cost = 3432.5 := by sorry

end NUMINAMATH_CALUDE_ramp_installation_cost_l143_14359


namespace NUMINAMATH_CALUDE_maple_trees_planted_l143_14301

theorem maple_trees_planted (initial : ℕ) (final : ℕ) (h1 : initial = 53) (h2 : final = 64) :
  final - initial = 11 := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_planted_l143_14301


namespace NUMINAMATH_CALUDE_alpha_integer_and_nonnegative_l143_14320

theorem alpha_integer_and_nonnegative (α : ℝ) 
  (h : ∀ n : ℕ+, ∃ k : ℤ, (n : ℝ) / α = k) : 
  0 ≤ α ∧ ∃ m : ℤ, α = m := by sorry

end NUMINAMATH_CALUDE_alpha_integer_and_nonnegative_l143_14320


namespace NUMINAMATH_CALUDE_mnp_value_l143_14327

theorem mnp_value (a b x y : ℝ) (m n p : ℤ) 
  (h : a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) 
  (h_equiv : (a^m*x - a^n)*(a^p*y - a^3) = a^5*b^5) : 
  m * n * p = 32 := by
  sorry

end NUMINAMATH_CALUDE_mnp_value_l143_14327


namespace NUMINAMATH_CALUDE_mathematics_permutations_l143_14373

def word : String := "MATHEMATICS"

theorem mathematics_permutations :
  let n : ℕ := word.length
  let m_count : ℕ := word.count 'M'
  let a_count : ℕ := word.count 'A'
  let t_count : ℕ := word.count 'T'
  (n = 11 ∧ m_count = 2 ∧ a_count = 2 ∧ t_count = 2) →
  (Nat.factorial n) / (Nat.factorial m_count * Nat.factorial a_count * Nat.factorial t_count) = 4989600 := by
sorry

end NUMINAMATH_CALUDE_mathematics_permutations_l143_14373


namespace NUMINAMATH_CALUDE_function_inequality_l143_14332

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, (x - 2) * deriv f x ≥ 0) : 
  f 1 + f 3 ≥ 2 * f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l143_14332


namespace NUMINAMATH_CALUDE_video_game_lives_l143_14381

theorem video_game_lives (initial_lives lost_lives gained_lives : ℕ) 
  (h1 : initial_lives = 10)
  (h2 : lost_lives = 6)
  (h3 : gained_lives = 37) :
  initial_lives - lost_lives + gained_lives = 41 :=
by sorry

end NUMINAMATH_CALUDE_video_game_lives_l143_14381


namespace NUMINAMATH_CALUDE_square_side_increase_l143_14338

theorem square_side_increase (p : ℝ) : 
  (1 + p / 100)^2 = 1.96 → p = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_side_increase_l143_14338


namespace NUMINAMATH_CALUDE_investment_return_l143_14398

/-- Given an investment scenario, calculate the percentage return -/
theorem investment_return (total_investment annual_income stock_price : ℝ)
  (h1 : total_investment = 6800)
  (h2 : stock_price = 136)
  (h3 : annual_income = 500) :
  (annual_income / total_investment) * 100 = (500 / 6800) * 100 := by
sorry

#eval (500 / 6800) * 100 -- To display the actual percentage

end NUMINAMATH_CALUDE_investment_return_l143_14398


namespace NUMINAMATH_CALUDE_hotel_pricing_theorem_l143_14353

/-- Hotel pricing model -/
structure HotelPricing where
  flatFee : ℝ  -- Flat fee for the first night
  nightlyFee : ℝ  -- Fixed amount for each additional night

/-- Calculate total cost for a stay -/
def totalCost (pricing : HotelPricing) (nights : ℕ) : ℝ :=
  pricing.flatFee + pricing.nightlyFee * (nights - 1)

/-- The hotel pricing theorem -/
theorem hotel_pricing_theorem (pricing : HotelPricing) :
  totalCost pricing 4 = 200 ∧ totalCost pricing 7 = 350 → pricing.flatFee = 50 := by
  sorry

#check hotel_pricing_theorem

end NUMINAMATH_CALUDE_hotel_pricing_theorem_l143_14353


namespace NUMINAMATH_CALUDE_line_length_problem_l143_14384

theorem line_length_problem (L : ℝ) (h : 0.75 * L - 0.4 * L = 28) : L = 80 := by
  sorry

end NUMINAMATH_CALUDE_line_length_problem_l143_14384


namespace NUMINAMATH_CALUDE_line_equation_sum_of_squares_l143_14349

-- Define the line l
def line_l (x y : ℝ) : Prop := y = 4 * x - 7

-- Define the point (2,1) that the line passes through
def point_on_line : Prop := line_l 2 1

-- Define the equation ax = by + c
def line_equation (a b c : ℤ) (x y : ℝ) : Prop := a * x = b * y + c

-- State that a, b, and c are positive integers with gcd 1
def abc_conditions (a b c : ℤ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ Int.gcd a (Int.gcd b c) = 1

-- Theorem statement
theorem line_equation_sum_of_squares :
  ∀ a b c : ℤ,
  (∀ x y : ℝ, line_l x y ↔ line_equation a b c x y) →
  abc_conditions a b c →
  a^2 + b^2 + c^2 = 66 := by sorry

end NUMINAMATH_CALUDE_line_equation_sum_of_squares_l143_14349


namespace NUMINAMATH_CALUDE_total_items_sold_is_727_l143_14379

/-- Represents the data for a single day of James' sales --/
structure DayData where
  houses : ℕ
  successRate : ℚ
  itemsPerHouse : ℕ

/-- Calculates the number of items sold in a day --/
def itemsSoldInDay (data : DayData) : ℚ :=
  data.houses * data.successRate * data.itemsPerHouse

/-- The week's data --/
def weekData : List DayData := [
  { houses := 20, successRate := 1, itemsPerHouse := 2 },
  { houses := 40, successRate := 4/5, itemsPerHouse := 3 },
  { houses := 50, successRate := 9/10, itemsPerHouse := 1 },
  { houses := 60, successRate := 3/4, itemsPerHouse := 4 },
  { houses := 80, successRate := 1/2, itemsPerHouse := 2 },
  { houses := 100, successRate := 7/10, itemsPerHouse := 1 },
  { houses := 120, successRate := 3/5, itemsPerHouse := 3 }
]

/-- Theorem: The total number of items sold during the week is 727 --/
theorem total_items_sold_is_727 : 
  (weekData.map itemsSoldInDay).sum = 727 := by
  sorry

end NUMINAMATH_CALUDE_total_items_sold_is_727_l143_14379


namespace NUMINAMATH_CALUDE_third_person_gets_max_median_l143_14358

/-- Represents the money distribution among three people -/
structure MoneyDistribution where
  person1 : ℕ
  person2 : ℕ
  person3 : ℕ

/-- The initial distribution of money -/
def initial_distribution : MoneyDistribution :=
  { person1 := 28, person2 := 72, person3 := 98 }

/-- The total amount of money -/
def total_money (d : MoneyDistribution) : ℕ :=
  d.person1 + d.person2 + d.person3

/-- Checks if a distribution is valid (sum equals total money) -/
def is_valid_distribution (d : MoneyDistribution) : Prop :=
  total_money d = total_money initial_distribution

/-- Checks if a number is the median of three numbers -/
def is_median (a b c m : ℕ) : Prop :=
  (a ≤ m ∧ m ≤ c) ∨ (c ≤ m ∧ m ≤ a)

/-- The maximum possible median after redistribution -/
def max_median : ℕ := 99

/-- Theorem: After redistribution to maximize the median, the third person ends up with $99 -/
theorem third_person_gets_max_median :
  ∃ (d : MoneyDistribution),
    is_valid_distribution d ∧
    is_median d.person1 d.person2 d.person3 max_median ∧
    d.person3 = max_median :=
  sorry

end NUMINAMATH_CALUDE_third_person_gets_max_median_l143_14358


namespace NUMINAMATH_CALUDE_starting_lineup_count_l143_14380

/-- The number of ways to choose a starting lineup from a basketball team -/
def choose_lineup (team_size : ℕ) (lineup_size : ℕ) (point_guard : ℕ) : ℕ :=
  team_size * (Nat.choose (team_size - 1) (lineup_size - 1))

/-- Theorem stating the number of ways to choose the starting lineup -/
theorem starting_lineup_count :
  choose_lineup 12 5 1 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l143_14380


namespace NUMINAMATH_CALUDE_eighteen_is_counterexample_l143_14318

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_counterexample (n : ℕ) : Prop :=
  ¬(is_prime n) ∧ ¬(is_prime (n - 3))

theorem eighteen_is_counterexample : is_counterexample 18 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_is_counterexample_l143_14318


namespace NUMINAMATH_CALUDE_red_part_length_l143_14306

/-- The length of the red part of a pencil given specific color proportions -/
theorem red_part_length (total_length : ℝ) (green_ratio : ℝ) (gold_ratio : ℝ) (red_ratio : ℝ)
  (h_total : total_length = 15)
  (h_green : green_ratio = 7/10)
  (h_gold : gold_ratio = 3/7)
  (h_red : red_ratio = 2/3) :
  red_ratio * (total_length - green_ratio * total_length - gold_ratio * (total_length - green_ratio * total_length)) =
  2/3 * (15 - 15 * 7/10 - (15 - 15 * 7/10) * 3/7) :=
by sorry

end NUMINAMATH_CALUDE_red_part_length_l143_14306


namespace NUMINAMATH_CALUDE_old_supervisor_salary_l143_14346

def num_workers : ℕ := 8
def initial_total : ℕ := 9
def initial_avg_salary : ℚ := 430
def new_avg_salary : ℚ := 420
def new_supervisor_salary : ℕ := 780

theorem old_supervisor_salary :
  ∃ (workers_total_salary old_supervisor_salary : ℚ),
    (workers_total_salary + old_supervisor_salary) / initial_total = initial_avg_salary ∧
    (workers_total_salary + new_supervisor_salary) / initial_total = new_avg_salary ∧
    old_supervisor_salary = 870 := by
  sorry

end NUMINAMATH_CALUDE_old_supervisor_salary_l143_14346


namespace NUMINAMATH_CALUDE_odd_function_negative_x_l143_14394

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_x
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_pos : ∀ x > 0, f x = 2 * x - 3) :
  ∀ x < 0, f x = 2 * x + 3 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_negative_x_l143_14394


namespace NUMINAMATH_CALUDE_company_total_individuals_l143_14307

/-- Represents the hierarchical structure of a company -/
structure CompanyHierarchy where
  workers_per_team_lead : Nat
  team_leads_per_manager : Nat
  managers_per_supervisor : Nat

/-- Calculates the total number of individuals in the company given the hierarchy and number of supervisors -/
def total_individuals (h : CompanyHierarchy) (supervisors : Nat) : Nat :=
  let managers := supervisors * h.managers_per_supervisor
  let team_leads := managers * h.team_leads_per_manager
  let workers := team_leads * h.workers_per_team_lead
  workers + team_leads + managers + supervisors

/-- Theorem stating that given the specific hierarchy and 10 supervisors, the total number of individuals is 3260 -/
theorem company_total_individuals :
  let h : CompanyHierarchy := {
    workers_per_team_lead := 15,
    team_leads_per_manager := 4,
    managers_per_supervisor := 5
  }
  total_individuals h 10 = 3260 := by
  sorry

end NUMINAMATH_CALUDE_company_total_individuals_l143_14307


namespace NUMINAMATH_CALUDE_solution_set_inequality_l143_14316

theorem solution_set_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_eq : 1/a + 2/b = 1) :
  {x : ℝ | (2 : ℝ)^(|x-1|-|x+2|) < 1} = {x : ℝ | x > -1/2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l143_14316


namespace NUMINAMATH_CALUDE_calculate_sales_professionals_l143_14377

/-- Calculates the number of sales professionals needed to sell a given number of cars
    over a specified period, with each professional selling a fixed number of cars per month. -/
theorem calculate_sales_professionals
  (total_cars : ℕ)
  (cars_per_salesperson_per_month : ℕ)
  (months_to_sell_all : ℕ)
  (h_total_cars : total_cars = 500)
  (h_cars_per_salesperson : cars_per_salesperson_per_month = 10)
  (h_months_to_sell : months_to_sell_all = 5)
  : (total_cars / months_to_sell_all) / cars_per_salesperson_per_month = 10 := by
  sorry

#check calculate_sales_professionals

end NUMINAMATH_CALUDE_calculate_sales_professionals_l143_14377


namespace NUMINAMATH_CALUDE_arctan_sum_three_four_l143_14350

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_four_l143_14350


namespace NUMINAMATH_CALUDE_cone_division_ratio_l143_14304

/-- Represents a right circular cone -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the division of a cone into two parts -/
structure ConeDivision where
  cone : Cone
  ratio : ℝ

/-- Calculates the surface area ratio of the smaller cone to the whole cone -/
def surfaceAreaRatio (d : ConeDivision) : ℝ := 
  d.ratio ^ 2

/-- Calculates the volume ratio of the smaller cone to the whole cone -/
def volumeRatio (d : ConeDivision) : ℝ := 
  d.ratio ^ 3

theorem cone_division_ratio (d : ConeDivision) 
  (h1 : d.cone.height = 4)
  (h2 : d.cone.baseRadius = 3)
  (h3 : surfaceAreaRatio d = volumeRatio d) :
  d.ratio = 125 / 387 := by
  sorry

#eval (125 : Nat) + 387

end NUMINAMATH_CALUDE_cone_division_ratio_l143_14304


namespace NUMINAMATH_CALUDE_inequality_implies_m_upper_bound_l143_14329

theorem inequality_implies_m_upper_bound :
  (∀ x₁ : ℝ, ∃ x₂ ∈ Set.Icc 3 4, x₁^2 + x₁*x₂ + x₂^2 ≥ 2*x₁ + m*x₂ + 3) →
  m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_m_upper_bound_l143_14329


namespace NUMINAMATH_CALUDE_power_fraction_equality_l143_14337

theorem power_fraction_equality : (2^2017 + 2^2013) / (2^2017 - 2^2013) = 17/15 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l143_14337


namespace NUMINAMATH_CALUDE_equation_roots_l143_14385

/-- Given an equation with two real roots, prove the range of m and a specific case. -/
theorem equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*m*x₁ = -m^2 + 2*x₁ ∧ x₂^2 - 2*m*x₂ = -m^2 + 2*x₂ ∧ x₁ ≠ x₂) → 
  (m ≥ -1/2 ∧ 
   (∀ x₁ x₂ : ℝ, x₁^2 - 2*m*x₁ = -m^2 + 2*x₁ → x₂^2 - 2*m*x₂ = -m^2 + 2*x₂ → |x₁| = x₂ → m = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l143_14385


namespace NUMINAMATH_CALUDE_characterize_valid_triples_l143_14328

def is_valid_triple (a b c : ℕ+) : Prop :=
  (1 : ℚ) / a + 2 / b + 3 / c = 1 ∧
  Nat.Prime a ∧
  a ≤ b ∧ b ≤ c

def valid_triples : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(2, 5, 30), (2, 6, 18), (2, 7, 14), (2, 8, 12), (2, 10, 10),
   (3, 4, 18), (3, 6, 9), (5, 4, 10)}

theorem characterize_valid_triples :
  ∀ a b c : ℕ+, is_valid_triple a b c ↔ (a, b, c) ∈ valid_triples := by
  sorry

end NUMINAMATH_CALUDE_characterize_valid_triples_l143_14328


namespace NUMINAMATH_CALUDE_min_value_sum_l143_14388

theorem min_value_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + a*b + a*c + b*c = 4) :
  ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + x*y + x*z + y*z = 4 →
  2*a + b + c ≤ 2*x + y + z ∧ 2*a + b + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l143_14388


namespace NUMINAMATH_CALUDE_solve_equation_l143_14313

theorem solve_equation (x : ℚ) : x / 4 * 5 + 10 - 12 = 48 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l143_14313


namespace NUMINAMATH_CALUDE_continuity_at_one_l143_14302

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) / (x^3 - 1)

theorem continuity_at_one :
  ∃ (L : ℝ), ContinuousAt (fun x => if x = 1 then L else f x) 1 ↔ L = 2/3 :=
sorry

end NUMINAMATH_CALUDE_continuity_at_one_l143_14302


namespace NUMINAMATH_CALUDE_solution_in_first_and_second_quadrants_l143_14352

-- Define the inequalities
def inequality1 (x y : ℝ) : Prop := y > 3 * x
def inequality2 (x y : ℝ) : Prop := y > 6 - 2 * x

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem solution_in_first_and_second_quadrants :
  ∀ x y : ℝ, inequality1 x y ∧ inequality2 x y →
  first_quadrant x y ∨ second_quadrant x y :=
sorry

end NUMINAMATH_CALUDE_solution_in_first_and_second_quadrants_l143_14352


namespace NUMINAMATH_CALUDE_counterexample_exists_l143_14376

theorem counterexample_exists : ∃ n : ℕ, 
  (∃ p : ℕ, Prime p ∧ ∃ k : ℕ, n = p^k) ∧ 
  Prime (n - 2) ∧ 
  n = 25 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l143_14376


namespace NUMINAMATH_CALUDE_common_y_intercept_l143_14390

theorem common_y_intercept (l₁ l₂ l₃ : ℝ → ℝ) (b : ℝ) :
  (∀ x, l₁ x = 1/2 * x + b) →
  (∀ x, l₂ x = 1/3 * x + b) →
  (∀ x, l₃ x = 1/4 * x + b) →
  ((-2*b) + (-3*b) + (-4*b) = 36) →
  b = -4 := by sorry

end NUMINAMATH_CALUDE_common_y_intercept_l143_14390


namespace NUMINAMATH_CALUDE_fish_value_in_honey_l143_14319

/-- Represents the value of one fish in terms of jars of honey -/
def fish_value (fish_to_bread : ℚ) (bread_to_honey : ℚ) : ℚ :=
  (3 / 4) * bread_to_honey

/-- Theorem stating the value of one fish in jars of honey -/
theorem fish_value_in_honey 
  (h1 : fish_to_bread = 3 / 4)  -- 4 fish = 3 loaves of bread
  (h2 : bread_to_honey = 3)     -- 1 loaf of bread = 3 jars of honey
  : fish_value fish_to_bread bread_to_honey = 9 / 4 := by
  sorry

#eval fish_value (3 / 4) 3  -- Should evaluate to 2.25

end NUMINAMATH_CALUDE_fish_value_in_honey_l143_14319


namespace NUMINAMATH_CALUDE_g_domain_all_reals_l143_14322

/-- The function g(x) = 1 / ((x-2)^2 + (x+2)^2 + 1) is defined for all real numbers. -/
theorem g_domain_all_reals :
  ∀ x : ℝ, (((x - 2)^2 + (x + 2)^2 + 1) ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_g_domain_all_reals_l143_14322


namespace NUMINAMATH_CALUDE_expression_evaluation_l143_14364

theorem expression_evaluation (a b c : ℚ) (ha : a = 12) (hb : b = 15) (hc : c = 19) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b) + a) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b) + 1) = a + b + c - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l143_14364


namespace NUMINAMATH_CALUDE_milk_for_cookies_l143_14396

/-- Given that 18 cookies require 3 quarts of milk, and there are 2 pints in a quart,
    prove that 6 cookies require 2 pints of milk. -/
theorem milk_for_cookies (cookies_large : ℕ) (milk_quarts : ℕ) (cookies_small : ℕ) 
  (pints_per_quart : ℕ) (h1 : cookies_large = 18) (h2 : milk_quarts = 3) 
  (h3 : cookies_small = 6) (h4 : pints_per_quart = 2) : 
  (milk_quarts * pints_per_quart * cookies_small) / cookies_large = 2 :=
by sorry

end NUMINAMATH_CALUDE_milk_for_cookies_l143_14396


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l143_14330

/-- 
If 4x² + (m-3)x + 1 is a perfect square trinomial, then m = 7 or m = -1.
-/
theorem perfect_square_trinomial_condition (m : ℝ) : 
  (∃ (a b : ℝ), ∀ (x : ℝ), 4*x^2 + (m-3)*x + 1 = (a*x + b)^2) → 
  (m = 7 ∨ m = -1) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l143_14330


namespace NUMINAMATH_CALUDE_sin_equality_problem_l143_14393

theorem sin_equality_problem (m : ℤ) (h1 : -90 ≤ m) (h2 : m ≤ 90) :
  Real.sin (m * Real.pi / 180) = Real.sin (710 * Real.pi / 180) → m = -10 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_problem_l143_14393


namespace NUMINAMATH_CALUDE_largest_angle_is_right_l143_14357

/-- Given a triangle ABC with side lengths a, b, and c, where c = 5 and 
    sqrt(a-4) + (b-3)^2 = 0, the largest interior angle of the triangle is 90°. -/
theorem largest_angle_is_right (a b c : ℝ) 
  (h1 : c = 5)
  (h2 : Real.sqrt (a - 4) + (b - 3)^2 = 0) :
  ∃ θ : ℝ, θ = Real.pi / 2 ∧ 
    θ = max (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))
            (max (Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)))
                 (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_is_right_l143_14357


namespace NUMINAMATH_CALUDE_river_width_l143_14336

/-- The width of a river given its depth, flow rate, and volume of water per minute. -/
theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) :
  depth = 3 →
  flow_rate_kmph = 2 →
  volume_per_minute = 3200 →
  ∃ (width : ℝ), abs (width - 32) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_river_width_l143_14336


namespace NUMINAMATH_CALUDE_smallest_integer_l143_14315

theorem smallest_integer (n : ℕ+) : 
  (Nat.lcm 36 n.val) / (Nat.gcd 36 n.val) = 24 → n.val ≥ 96 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l143_14315


namespace NUMINAMATH_CALUDE_circle_portion_area_l143_14382

/-- The area of the portion of the circle x^2 - 16x + y^2 = 51 that lies above the x-axis 
    and to the left of the line y = 10 - x is equal to 8π. -/
theorem circle_portion_area : 
  ∃ (A : ℝ), 
    (∀ x y : ℝ, x^2 - 16*x + y^2 = 51 → y ≥ 0 → y ≤ 10 - x → 
      (x, y) ∈ {p : ℝ × ℝ | p.1^2 - 16*p.1 + p.2^2 = 51 ∧ p.2 ≥ 0 ∧ p.2 ≤ 10 - p.1}) ∧
    A = Real.pi * 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_portion_area_l143_14382


namespace NUMINAMATH_CALUDE_max_crates_third_trip_l143_14310

/-- The weight of each crate in kilograms -/
def crate_weight : ℝ := 1250

/-- The maximum weight capacity of the trailer in kilograms -/
def max_weight : ℝ := 6250

/-- The number of crates on the first trip -/
def first_trip_crates : ℕ := 3

/-- The number of crates on the second trip -/
def second_trip_crates : ℕ := 4

/-- Theorem: The maximum number of crates that can be carried on the third trip is 5 -/
theorem max_crates_third_trip :
  ∃ (x : ℕ), x ≤ 5 ∧
  (∀ y : ℕ, y > x → y * crate_weight > max_weight) ∧
  x * crate_weight ≤ max_weight :=
sorry

end NUMINAMATH_CALUDE_max_crates_third_trip_l143_14310


namespace NUMINAMATH_CALUDE_spoon_cost_l143_14387

theorem spoon_cost (num_plates : ℕ) (plate_cost : ℚ) (num_spoons : ℕ) (total_cost : ℚ) : 
  num_plates = 9 → 
  plate_cost = 2 → 
  num_spoons = 4 → 
  total_cost = 24 → 
  (total_cost - (↑num_plates * plate_cost)) / ↑num_spoons = 1.5 := by
sorry

end NUMINAMATH_CALUDE_spoon_cost_l143_14387


namespace NUMINAMATH_CALUDE_half_meter_cut_l143_14300

theorem half_meter_cut (initial_length : ℚ) (cut_length : ℚ) (result_length : ℚ) : 
  initial_length = 8/15 →
  cut_length = 1/30 →
  result_length = initial_length - cut_length →
  result_length = 1/2 :=
by
  sorry

#check half_meter_cut

end NUMINAMATH_CALUDE_half_meter_cut_l143_14300


namespace NUMINAMATH_CALUDE_square_difference_65_55_l143_14371

theorem square_difference_65_55 : 65^2 - 55^2 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_65_55_l143_14371


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_sqrt2_plus_minus_one_l143_14351

theorem arithmetic_mean_of_sqrt2_plus_minus_one :
  (((Real.sqrt 2) + 1) + ((Real.sqrt 2) - 1)) / 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_sqrt2_plus_minus_one_l143_14351


namespace NUMINAMATH_CALUDE_point_on_line_ratio_l143_14324

/-- Given six points O, A, B, C, D, E on a straight line in that order, with P between C and D,
    prove that OP = (ce - ad) / (a - c + e - d) when AP:PE = CP:PD -/
theorem point_on_line_ratio (a b c d e x : ℝ) 
  (h_order : 0 < a ∧ a < b ∧ b < c ∧ c < x ∧ x < d ∧ d < e) 
  (h_ratio : (a - x) / (x - e) = (c - x) / (x - d)) : 
  x = (c * e - a * d) / (a - c + e - d) := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_ratio_l143_14324


namespace NUMINAMATH_CALUDE_inequality_proof_l143_14317

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 1) :
  Real.sqrt (x + y*z) + Real.sqrt (y + z*x) + Real.sqrt (z + x*y) ≥ 
  Real.sqrt (x*y*z) + Real.sqrt x + Real.sqrt y + Real.sqrt z :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l143_14317


namespace NUMINAMATH_CALUDE_probability_second_high_given_first_inferior_is_eight_ninths_l143_14391

/-- Represents the total number of pencils -/
def total_pencils : ℕ := 10

/-- Represents the number of high-quality pencils -/
def high_quality : ℕ := 8

/-- Represents the number of inferior quality pencils -/
def inferior_quality : ℕ := 2

/-- Represents the probability of drawing a high-quality pencil on the second draw,
    given that the first draw was an inferior quality pencil -/
def probability_second_high_given_first_inferior : ℚ :=
  high_quality / (total_pencils - 1)

theorem probability_second_high_given_first_inferior_is_eight_ninths :
  probability_second_high_given_first_inferior = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_high_given_first_inferior_is_eight_ninths_l143_14391


namespace NUMINAMATH_CALUDE_real_sqrt_reciprocal_range_l143_14343

theorem real_sqrt_reciprocal_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (5 - x)) ↔ x < 5 := by sorry

end NUMINAMATH_CALUDE_real_sqrt_reciprocal_range_l143_14343


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_system_l143_14311

theorem unique_solution_quadratic_system (y : ℚ) 
  (h1 : 9 * y^2 + 8 * y - 1 = 0)
  (h2 : 27 * y^2 + 44 * y - 7 = 0) : 
  y = 1/9 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_system_l143_14311


namespace NUMINAMATH_CALUDE_kims_money_l143_14374

/-- Given the money relationships between Kim, Sal, Phil, and Alex, prove Kim's amount. -/
theorem kims_money (sal phil kim alex : ℝ) 
  (h1 : kim = 1.4 * sal)  -- Kim has 40% more money than Sal
  (h2 : sal = 0.8 * phil)  -- Sal has 20% less money than Phil
  (h3 : alex = 1.25 * (sal + kim))  -- Alex has 25% more money than Sal and Kim combined
  (h4 : sal + phil + alex = 3.6)  -- Sal, Phil, and Alex have a combined total of $3.60
  : kim = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_kims_money_l143_14374


namespace NUMINAMATH_CALUDE_total_tissues_used_l143_14365

/-- The number of tissues Carol had initially -/
def initial_tissues : ℕ := 97

/-- The number of tissues Carol had after use -/
def remaining_tissues : ℕ := 58

/-- The total number of tissues used by Carol and her friends -/
def tissues_used : ℕ := initial_tissues - remaining_tissues

theorem total_tissues_used :
  tissues_used = 39 :=
by sorry

end NUMINAMATH_CALUDE_total_tissues_used_l143_14365


namespace NUMINAMATH_CALUDE_complex_number_problem_l143_14362

theorem complex_number_problem : ∃ (z : ℂ), 
  z.im = (3 * Complex.I).re ∧ 
  z.re = (-3 + Complex.I).im ∧ 
  z = 3 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l143_14362


namespace NUMINAMATH_CALUDE_petes_flag_shapes_l143_14392

def us_stars : ℕ := 50
def us_stripes : ℕ := 13

def circles : ℕ := us_stars / 2 - 3
def squares : ℕ := us_stripes * 2 + 6
def triangles : ℕ := (us_stars - us_stripes) * 2
def diamonds : ℕ := (us_stars + us_stripes) / 4

theorem petes_flag_shapes :
  circles + squares + triangles + diamonds = 143 := by
  sorry

end NUMINAMATH_CALUDE_petes_flag_shapes_l143_14392


namespace NUMINAMATH_CALUDE_rain_and_humidity_probability_l143_14369

/-- The probability of rain in a coastal city in Zhejiang -/
def prob_rain : ℝ := 0.4

/-- The probability that the humidity exceeds 70% on rainy days -/
def prob_humidity_given_rain : ℝ := 0.6

/-- The probability that it rains and the humidity exceeds 70% -/
def prob_rain_and_humidity : ℝ := prob_rain * prob_humidity_given_rain

theorem rain_and_humidity_probability :
  prob_rain_and_humidity = 0.24 :=
sorry

end NUMINAMATH_CALUDE_rain_and_humidity_probability_l143_14369


namespace NUMINAMATH_CALUDE_nested_radical_value_l143_14386

/-- The value of the infinite nested radical √(3 - √(3 - √(3 - ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt 3))))))

/-- Theorem stating that the nested radical equals (-1 + √13) / 2 -/
theorem nested_radical_value : nestedRadical = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l143_14386


namespace NUMINAMATH_CALUDE_sum_of_divisors_of_23_l143_14363

theorem sum_of_divisors_of_23 (h : Nat.Prime 23) : (Finset.sum (Nat.divisors 23) id) = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_of_23_l143_14363


namespace NUMINAMATH_CALUDE_total_tree_count_l143_14383

def douglas_fir_count : ℕ := 350
def douglas_fir_cost : ℕ := 300
def ponderosa_pine_cost : ℕ := 225
def total_cost : ℕ := 217500

theorem total_tree_count : 
  ∃ (ponderosa_pine_count : ℕ),
    douglas_fir_count * douglas_fir_cost + 
    ponderosa_pine_count * ponderosa_pine_cost = total_cost ∧
    douglas_fir_count + ponderosa_pine_count = 850 :=
by sorry

end NUMINAMATH_CALUDE_total_tree_count_l143_14383


namespace NUMINAMATH_CALUDE_proposition_relation_l143_14356

theorem proposition_relation :
  (∀ x y : ℝ, x^2 + y^2 ≤ 2*x → x^2 + y^2 ≤ 4) ∧
  (∃ x y : ℝ, x^2 + y^2 ≤ 4 ∧ x^2 + y^2 > 2*x) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relation_l143_14356


namespace NUMINAMATH_CALUDE_decimal_to_base_five_l143_14370

theorem decimal_to_base_five : 
  (2 * 5^3 + 0 * 5^2 + 1 * 5^1 + 1 * 5^0 : ℕ) = 256 := by sorry

end NUMINAMATH_CALUDE_decimal_to_base_five_l143_14370


namespace NUMINAMATH_CALUDE_bruce_payment_l143_14389

/-- Calculates the total amount Bruce paid for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1110 for his purchase -/
theorem bruce_payment : total_amount 8 70 10 55 = 1110 := by
  sorry

end NUMINAMATH_CALUDE_bruce_payment_l143_14389


namespace NUMINAMATH_CALUDE_square_area_l143_14305

theorem square_area (side_length : ℝ) (h : side_length = 19) :
  side_length * side_length = 361 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l143_14305


namespace NUMINAMATH_CALUDE_simplify_expression_l143_14342

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  x⁻¹ - x + 2 = (1 - (x - 1)^2) / x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l143_14342


namespace NUMINAMATH_CALUDE_lose_sector_area_l143_14303

theorem lose_sector_area (radius : ℝ) (win_probability : ℝ) 
  (h1 : radius = 12)
  (h2 : win_probability = 1/3) : 
  (1 - win_probability) * π * radius^2 = 96 * π := by
  sorry

end NUMINAMATH_CALUDE_lose_sector_area_l143_14303


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l143_14334

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l143_14334


namespace NUMINAMATH_CALUDE_special_sequence_bijective_l143_14339

/-- An integer sequence with specific properties -/
def SpecialSequence (a : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, ∃ k > n, a k > 0) ∧  -- Infinitely many positive integers
  (∀ n : ℕ, ∃ k > n, a k < 0) ∧  -- Infinitely many negative integers
  (∀ n : ℕ+, Function.Injective (fun i => a i % n))  -- Distinct remainders

/-- The main theorem -/
theorem special_sequence_bijective (a : ℕ → ℤ) (h : SpecialSequence a) :
  Function.Bijective a :=
sorry

end NUMINAMATH_CALUDE_special_sequence_bijective_l143_14339
