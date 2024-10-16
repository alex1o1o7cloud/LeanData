import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_minimum_l3686_368690

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x + 6

-- Theorem statement
theorem quadratic_minimum :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min) ∧
  (∀ (x : ℝ), f x ≥ 5) ∧
  (f 1 = 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3686_368690


namespace NUMINAMATH_CALUDE_square_of_sum_l3686_368695

theorem square_of_sum (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l3686_368695


namespace NUMINAMATH_CALUDE_shirt_price_theorem_l3686_368644

/-- Represents the problem of determining shirt prices and profits --/
structure ShirtProblem where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  quantity_ratio : ℝ
  price_difference : ℝ
  discount_quantity : ℕ
  discount_rate : ℝ
  min_profit : ℝ

/-- Calculates the unit price of the first batch --/
def first_batch_unit_price (p : ShirtProblem) : ℝ := 80

/-- Calculates the minimum selling price per shirt --/
def min_selling_price (p : ShirtProblem) : ℝ := 120

/-- Theorem stating the correctness of the calculated prices --/
theorem shirt_price_theorem (p : ShirtProblem) 
  (h1 : p.first_batch_cost = 3200)
  (h2 : p.second_batch_cost = 7200)
  (h3 : p.quantity_ratio = 2)
  (h4 : p.price_difference = 10)
  (h5 : p.discount_quantity = 20)
  (h6 : p.discount_rate = 0.2)
  (h7 : p.min_profit = 3520) :
  first_batch_unit_price p = 80 ∧ 
  min_selling_price p = 120 ∧
  min_selling_price p ≥ (p.min_profit + p.first_batch_cost + p.second_batch_cost) / 
    (p.first_batch_cost / first_batch_unit_price p + 
     p.second_batch_cost / (first_batch_unit_price p + p.price_difference) + 
     p.discount_quantity * (1 - p.discount_rate)) := by
  sorry


end NUMINAMATH_CALUDE_shirt_price_theorem_l3686_368644


namespace NUMINAMATH_CALUDE_christopher_karen_difference_l3686_368679

/-- Proves that Christopher has $8.00 more than Karen given their quarter counts -/
theorem christopher_karen_difference :
  let karen_quarters : ℕ := 32
  let christopher_quarters : ℕ := 64
  let quarter_value : ℚ := 1/4
  let karen_money := karen_quarters * quarter_value
  let christopher_money := christopher_quarters * quarter_value
  christopher_money - karen_money = 8 :=
by sorry

end NUMINAMATH_CALUDE_christopher_karen_difference_l3686_368679


namespace NUMINAMATH_CALUDE_luke_paint_area_l3686_368656

/-- Calculates the area to be painted on a wall with a bookshelf -/
def area_to_paint (wall_height wall_length bookshelf_width bookshelf_height : ℝ) : ℝ :=
  wall_height * wall_length - bookshelf_width * bookshelf_height

/-- Proves that Luke needs to paint 135 square feet -/
theorem luke_paint_area :
  area_to_paint 10 15 3 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_luke_paint_area_l3686_368656


namespace NUMINAMATH_CALUDE_parametric_line_point_at_zero_l3686_368638

/-- A parametric line in 2D space -/
structure ParametricLine where
  /-- The position vector for a given parameter t -/
  pos : ℝ → (ℝ × ℝ)

/-- Theorem: Given a parametric line with specific points, find the point at t = 0 -/
theorem parametric_line_point_at_zero
  (line : ParametricLine)
  (h1 : line.pos 1 = (2, 3))
  (h4 : line.pos 4 = (6, -12)) :
  line.pos 0 = (2/3, 8) := by
  sorry

end NUMINAMATH_CALUDE_parametric_line_point_at_zero_l3686_368638


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3686_368681

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_2a_plus_b := Real.sqrt ((2*a.1 + b.1)^2 + (2*a.2 + b.2)^2)
  angle = π/4 ∧ magnitude_a = 1 ∧ magnitude_2a_plus_b = Real.sqrt 10 →
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_vector_magnitude_problem_l3686_368681


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3686_368603

/-- Given a quadratic function f(x) = ax^2 - c, prove that if -4 ≤ f(1) ≤ -1 and -1 ≤ f(2) ≤ 5, then -1 ≤ f(3) ≤ 20. -/
theorem quadratic_function_range (a c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^2 - c
  (-4 ≤ f 1 ∧ f 1 ≤ -1) → (-1 ≤ f 2 ∧ f 2 ≤ 5) → (-1 ≤ f 3 ∧ f 3 ≤ 20) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3686_368603


namespace NUMINAMATH_CALUDE_sum_of_xyz_l3686_368606

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + y^2 + x*y = 1)
  (eq2 : y^2 + z^2 + y*z = 4)
  (eq3 : z^2 + x^2 + z*x = 5) :
  x + y + z = Real.sqrt (5 + 2 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l3686_368606


namespace NUMINAMATH_CALUDE_parabola_directrix_l3686_368645

/-- Given a parabola with equation y = 4x^2, its directrix has equation y = -1/16 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = 4 * x^2) → (∃ p : ℝ, p > 0 ∧ y = (1 / (4 * p)) * x^2 ∧ -1 / (4 * p) = -1/16) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3686_368645


namespace NUMINAMATH_CALUDE_vacation_savings_theorem_l3686_368611

/-- Calculates the number of months needed to reach a savings goal -/
def months_to_goal (goal : ℕ) (current : ℕ) (monthly : ℕ) : ℕ :=
  ((goal - current) + monthly - 1) / monthly

theorem vacation_savings_theorem (goal current monthly : ℕ) 
  (h1 : goal = 5000)
  (h2 : current = 2900)
  (h3 : monthly = 700) :
  months_to_goal goal current monthly = 3 := by
  sorry

end NUMINAMATH_CALUDE_vacation_savings_theorem_l3686_368611


namespace NUMINAMATH_CALUDE_chord_equation_l3686_368608

/-- Given positive real numbers m, n, s, t satisfying certain conditions,
    prove that the equation of the line containing a chord of the hyperbola
    x²/4 - y²/2 = 1 with midpoint (m, n) is x - 2y + 1 = 0. -/
theorem chord_equation (m n s t : ℝ) 
    (hm : m > 0) (hn : n > 0) (hs : s > 0) (ht : t > 0)
    (h1 : m + n = 2)
    (h2 : m / s + n / t = 9)
    (h3 : s + t = 4 / 9)
    (h4 : ∀ s' t' : ℝ, s' > 0 → t' > 0 → m / s' + n / t' = 9 → s' + t' ≥ 4 / 9)
    (h5 : ∃ x₁ y₁ x₂ y₂ : ℝ, 
      x₁^2 / 4 - y₁^2 / 2 = 1 ∧
      x₂^2 / 4 - y₂^2 / 2 = 1 ∧
      (x₁ + x₂) / 2 = m ∧
      (y₁ + y₂) / 2 = n) :
  ∃ a b c : ℝ, a * m + b * n + c = 0 ∧
             ∀ x y : ℝ, x^2 / 4 - y^2 / 2 = 1 →
               (∃ t : ℝ, x = m + t * a ∧ y = n + t * b) →
               a * x + b * y + c = 0 ∧
               a = 1 ∧ b = -2 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l3686_368608


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3686_368617

-- Define the triangle
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.angleA = 2 * t.angleB ∧
  t.angleC > Real.pi / 2 ∧
  t.angleA + t.angleB + t.angleC = Real.pi

-- Define the perimeter
def perimeter (t : Triangle) : ℕ :=
  t.a.val + t.b.val + t.c.val

-- Theorem statement
theorem min_perimeter_triangle :
  ∃ (t : Triangle), validTriangle t ∧
    (∀ (t' : Triangle), validTriangle t' → perimeter t ≤ perimeter t') ∧
    perimeter t = 77 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3686_368617


namespace NUMINAMATH_CALUDE_beryl_radishes_l3686_368631

def radishes_problem (first_basket : ℕ) (difference : ℕ) : Prop :=
  let second_basket := first_basket + difference
  let total := first_basket + second_basket
  total = 88

theorem beryl_radishes : radishes_problem 37 14 := by
  sorry

end NUMINAMATH_CALUDE_beryl_radishes_l3686_368631


namespace NUMINAMATH_CALUDE_bottom_layer_lights_for_specific_tower_l3686_368612

/-- Represents a tower with a geometric progression of lights -/
structure LightTower where
  layers : ℕ
  total_lights : ℕ
  ratio : ℕ

/-- Calculates the number of lights on the bottom layer of a tower -/
def bottom_layer_lights (tower : LightTower) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- The theorem stating the number of lights on the bottom layer of the specific tower -/
theorem bottom_layer_lights_for_specific_tower :
  let tower : LightTower := ⟨5, 242, 3⟩
  bottom_layer_lights tower = 162 := by
  sorry

end NUMINAMATH_CALUDE_bottom_layer_lights_for_specific_tower_l3686_368612


namespace NUMINAMATH_CALUDE_ribbon_length_difference_l3686_368650

/-- The theorem about ribbon lengths difference after cutting and giving -/
theorem ribbon_length_difference 
  (initial_difference : Real) 
  (cut_length : Real) : 
  initial_difference = 8.8 → 
  cut_length = 4.3 → 
  initial_difference + 2 * cut_length = 17.4 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_difference_l3686_368650


namespace NUMINAMATH_CALUDE_max_a_value_l3686_368625

noncomputable def f (x : ℝ) : ℝ := 1 - Real.sqrt (x + 1)

noncomputable def g (a x : ℝ) : ℝ := Real.log (a * x^2 - 3 * x + 1)

theorem max_a_value :
  ∃ (a : ℝ), ∀ (a' : ℝ),
    (∀ (x₁ : ℝ), x₁ ≥ 0 → ∃ (x₂ : ℝ), f x₁ = g a' x₂) →
    a' ≤ a ∧
    (∀ (x₁ : ℝ), x₁ ≥ 0 → ∃ (x₂ : ℝ), f x₁ = g a x₂) ∧
    a = 9/4 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l3686_368625


namespace NUMINAMATH_CALUDE_coronavirus_case_ratio_l3686_368699

/-- Given the number of coronavirus cases in a country during two waves, 
    prove the ratio of average daily cases between the waves. -/
theorem coronavirus_case_ratio 
  (first_wave_daily : ℕ) 
  (second_wave_total : ℕ) 
  (second_wave_days : ℕ) 
  (h1 : first_wave_daily = 300)
  (h2 : second_wave_total = 21000)
  (h3 : second_wave_days = 14) :
  (second_wave_total / second_wave_days) / first_wave_daily = 5 := by
  sorry


end NUMINAMATH_CALUDE_coronavirus_case_ratio_l3686_368699


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3686_368692

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4
  let θ : ℝ := 3 * Real.pi / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = -2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3686_368692


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3686_368697

/-- The value of b for which the line 2bx + (3b - 2)y = 5b + 6 passes through the point (6, -10) -/
theorem line_passes_through_point : 
  ∃ b : ℚ, b = 14/23 ∧ 2*b*6 + (3*b - 2)*(-10) = 5*b + 6 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3686_368697


namespace NUMINAMATH_CALUDE_circle_radius_l3686_368684

theorem circle_radius (x y : ℝ) : (x - 1)^2 + y^2 = 9 → 3 = Real.sqrt ((x - 1)^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3686_368684


namespace NUMINAMATH_CALUDE_regular_octagon_area_l3686_368630

/-- Regular octagon with area A, longest diagonal d_max, and shortest diagonal d_min -/
structure RegularOctagon where
  A : ℝ
  d_max : ℝ
  d_min : ℝ

/-- The area of a regular octagon is equal to the product of its longest and shortest diagonals -/
theorem regular_octagon_area (o : RegularOctagon) : o.A = o.d_max * o.d_min := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_area_l3686_368630


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3686_368682

theorem inequality_solution_set (x : ℝ) : 
  abs (2 * x - 1) + abs (2 * x + 1) ≤ 6 ↔ x ∈ Set.Icc (-3/2) (3/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3686_368682


namespace NUMINAMATH_CALUDE_max_towns_is_four_l3686_368674

/-- Represents the type of connection between two towns -/
inductive Connection
  | Air
  | Bus
  | Train

/-- Represents a town -/
structure Town where
  id : Nat

/-- Represents the network of towns and their connections -/
structure TownNetwork where
  towns : Finset Town
  connections : Town → Town → Option Connection

/-- Checks if the given network satisfies all conditions -/
def satisfiesConditions (network : TownNetwork) : Prop :=
  -- Condition 1: Each pair of towns is directly linked by just one of air, bus, or train
  (∀ t1 t2 : Town, t1 ∈ network.towns → t2 ∈ network.towns → t1 ≠ t2 →
    ∃! c : Connection, network.connections t1 t2 = some c) ∧
  -- Condition 2: At least one pair is linked by each type of connection
  (∃ t1 t2 : Town, t1 ∈ network.towns ∧ t2 ∈ network.towns ∧ network.connections t1 t2 = some Connection.Air) ∧
  (∃ t1 t2 : Town, t1 ∈ network.towns ∧ t2 ∈ network.towns ∧ network.connections t1 t2 = some Connection.Bus) ∧
  (∃ t1 t2 : Town, t1 ∈ network.towns ∧ t2 ∈ network.towns ∧ network.connections t1 t2 = some Connection.Train) ∧
  -- Condition 3: No town has all three types of connections
  (∀ t : Town, t ∈ network.towns →
    ¬(∃ t1 t2 t3 : Town, t1 ∈ network.towns ∧ t2 ∈ network.towns ∧ t3 ∈ network.towns ∧
      network.connections t t1 = some Connection.Air ∧
      network.connections t t2 = some Connection.Bus ∧
      network.connections t t3 = some Connection.Train)) ∧
  -- Condition 4: No three towns have all connections of the same type
  (∀ t1 t2 t3 : Town, t1 ∈ network.towns → t2 ∈ network.towns → t3 ∈ network.towns →
    t1 ≠ t2 → t2 ≠ t3 → t1 ≠ t3 →
    ¬(network.connections t1 t2 = network.connections t2 t3 ∧
      network.connections t2 t3 = network.connections t1 t3))

/-- The main theorem stating that the maximum number of towns satisfying the conditions is 4 -/
theorem max_towns_is_four :
  (∃ (network : TownNetwork), satisfiesConditions network ∧ network.towns.card = 4) ∧
  (∀ (network : TownNetwork), satisfiesConditions network → network.towns.card ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_towns_is_four_l3686_368674


namespace NUMINAMATH_CALUDE_gary_egg_collection_l3686_368673

/-- Represents the egg-laying rates of the initial chickens -/
def initial_rates : List Nat := [6, 5, 7, 4]

/-- Calculates the number of surviving chickens after two years -/
def surviving_chickens (initial : Nat) (growth_factor : Nat) (mortality_rate : Rat) : Nat :=
  Nat.floor ((initial * growth_factor : Rat) * (1 - mortality_rate))

/-- Calculates the average egg-laying rate -/
def average_rate (rates : List Nat) : Rat :=
  (rates.sum : Rat) / rates.length

/-- Calculates the total eggs per week -/
def total_eggs_per_week (chickens : Nat) (avg_rate : Rat) : Nat :=
  Nat.floor (7 * (chickens : Rat) * avg_rate)

/-- Theorem stating the number of eggs Gary collects per week -/
theorem gary_egg_collection :
  total_eggs_per_week
    (surviving_chickens 4 8 (1/5))
    (average_rate initial_rates) = 959 := by
  sorry

end NUMINAMATH_CALUDE_gary_egg_collection_l3686_368673


namespace NUMINAMATH_CALUDE_distribute_teachers_count_l3686_368619

/-- The number of ways to distribute 6 teachers across 4 neighborhoods --/
def distribute_teachers : ℕ :=
  let n_teachers : ℕ := 6
  let n_neighborhoods : ℕ := 4
  let distribution_3111 : ℕ := (Nat.choose n_teachers 3) * (Nat.factorial n_neighborhoods)
  let distribution_2211 : ℕ := 
    (Nat.choose n_teachers 2) * (Nat.choose (n_teachers - 2) 2) * 
    (Nat.factorial n_neighborhoods) / (Nat.factorial 2)
  distribution_3111 + distribution_2211

/-- Theorem stating that the number of distribution schemes is 1560 --/
theorem distribute_teachers_count : distribute_teachers = 1560 := by
  sorry

end NUMINAMATH_CALUDE_distribute_teachers_count_l3686_368619


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l3686_368667

def p (x : ℝ) : Prop := (x^2 + 6*x + 8) * Real.sqrt (x + 3) ≥ 0

def q (x : ℝ) : Prop := x = -3

theorem p_necessary_not_sufficient :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l3686_368667


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3686_368668

theorem solution_set_of_inequality (x : ℝ) :
  (((x - 2) / (x + 3) > 0) ↔ (x ∈ Set.Iio (-3) ∪ Set.Ioi 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3686_368668


namespace NUMINAMATH_CALUDE_bicentric_quadrilateral_theorem_l3686_368629

/-- A bicentric quadrilateral is a quadrilateral that has both an inscribed circle and a circumscribed circle. -/
structure BicentricQuadrilateral where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The radius of the circumscribed circle -/
  ρ : ℝ
  /-- The distance between the centers of the inscribed and circumscribed circles -/
  h : ℝ
  /-- Ensure r, ρ, and h are positive -/
  r_pos : r > 0
  ρ_pos : ρ > 0
  h_pos : h > 0
  /-- Ensure h is less than ρ (as the incenter must be inside the circumcircle) -/
  h_lt_ρ : h < ρ

/-- The main theorem about bicentric quadrilaterals -/
theorem bicentric_quadrilateral_theorem (q : BicentricQuadrilateral) :
  1 / (q.ρ + q.h)^2 + 1 / (q.ρ - q.h)^2 = 1 / q.r^2 := by
  sorry

end NUMINAMATH_CALUDE_bicentric_quadrilateral_theorem_l3686_368629


namespace NUMINAMATH_CALUDE_election_outcome_depends_on_radicals_l3686_368636

/-- Represents a political group in the election -/
inductive PoliticalGroup
| Socialist
| Republican
| Radical
| Other

/-- Represents the election models -/
inductive ElectionModel
| A
| B

/-- Represents the election system with four political groups -/
structure ElectionSystem where
  groups : Fin 4 → PoliticalGroup
  groupSize : ℕ
  socialistsPrefB : ℕ
  republicansPrefA : ℕ
  radicalSupport : PoliticalGroup

/-- The outcome of the election -/
def electionOutcome (system : ElectionSystem) : ElectionModel :=
  match system.radicalSupport with
  | PoliticalGroup.Socialist => ElectionModel.B
  | PoliticalGroup.Republican => ElectionModel.A
  | _ => sorry -- This case should not occur in our scenario

/-- Theorem stating that the election outcome depends on radicals' support -/
theorem election_outcome_depends_on_radicals (system : ElectionSystem) 
  (h1 : system.socialistsPrefB = system.republicansPrefA)
  (h2 : system.socialistsPrefB > 0) :
  (∃ (support : PoliticalGroup), 
    electionOutcome {system with radicalSupport := support} = ElectionModel.A) ∧
  (∃ (support : PoliticalGroup), 
    electionOutcome {system with radicalSupport := support} = ElectionModel.B) :=
  sorry


end NUMINAMATH_CALUDE_election_outcome_depends_on_radicals_l3686_368636


namespace NUMINAMATH_CALUDE_article_price_calculation_l3686_368613

theorem article_price_calculation (p q : ℝ) : 
  let final_price := 1
  let price_after_increase (x : ℝ) := x * (1 + p / 100)
  let price_after_decrease (y : ℝ) := y * (1 - q / 100)
  let original_price := 10000 / (10000 + 100 * (p - q) - p * q)
  price_after_decrease (price_after_increase original_price) = final_price :=
by sorry

end NUMINAMATH_CALUDE_article_price_calculation_l3686_368613


namespace NUMINAMATH_CALUDE_cuboid_area_example_l3686_368662

/-- The surface area of a cuboid -/
def cuboid_surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a cuboid with length 8 cm, breadth 6 cm, and height 9 cm is 348 square centimeters -/
theorem cuboid_area_example : cuboid_surface_area 8 6 9 = 348 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_area_example_l3686_368662


namespace NUMINAMATH_CALUDE_unique_g_2_value_l3686_368649

theorem unique_g_2_value (g : ℤ → ℤ) 
  (h : ∀ m n : ℤ, g (m + n) + g (m * n + 1) = g m * g n + 1) : 
  ∃! x : ℤ, g 2 = x ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_unique_g_2_value_l3686_368649


namespace NUMINAMATH_CALUDE_gary_remaining_money_l3686_368683

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem: Gary's remaining money -/
theorem gary_remaining_money :
  remaining_money 73 55 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gary_remaining_money_l3686_368683


namespace NUMINAMATH_CALUDE_expected_potato_yield_l3686_368610

/-- Calculates the expected potato yield from a rectangular garden --/
theorem expected_potato_yield
  (garden_length_steps : ℕ)
  (garden_width_steps : ℕ)
  (step_length_feet : ℝ)
  (potato_yield_per_sqft : ℝ)
  (h1 : garden_length_steps = 18)
  (h2 : garden_width_steps = 25)
  (h3 : step_length_feet = 3)
  (h4 : potato_yield_per_sqft = 1/3) :
  (garden_length_steps : ℝ) * step_length_feet *
  (garden_width_steps : ℝ) * step_length_feet *
  potato_yield_per_sqft = 1350 := by
  sorry

end NUMINAMATH_CALUDE_expected_potato_yield_l3686_368610


namespace NUMINAMATH_CALUDE_exam_analysis_theorem_l3686_368637

structure StatisticalAnalysis where
  population_size : ℕ
  sample_size : ℕ
  sample_is_subset : sample_size ≤ population_size

def is_population (sa : StatisticalAnalysis) (n : ℕ) : Prop :=
  n = sa.population_size

def is_sample (sa : StatisticalAnalysis) (n : ℕ) : Prop :=
  n = sa.sample_size

def is_sample_size (sa : StatisticalAnalysis) (n : ℕ) : Prop :=
  n = sa.sample_size

-- The statement we want to prove incorrect
def each_examinee_is_individual_unit (sa : StatisticalAnalysis) : Prop :=
  False  -- This is set to False to represent that the statement is incorrect

theorem exam_analysis_theorem (sa : StatisticalAnalysis) 
  (h_pop : sa.population_size = 13000)
  (h_sample : sa.sample_size = 500) :
  is_population sa 13000 ∧ 
  is_sample sa 500 ∧ 
  is_sample_size sa 500 ∧ 
  ¬(each_examinee_is_individual_unit sa) := by
  sorry

#check exam_analysis_theorem

end NUMINAMATH_CALUDE_exam_analysis_theorem_l3686_368637


namespace NUMINAMATH_CALUDE_product_divisible_by_60_l3686_368659

theorem product_divisible_by_60 (a : ℤ) : 
  60 ∣ (a^2 - 1) * a^2 * (a^2 + 1) := by sorry

end NUMINAMATH_CALUDE_product_divisible_by_60_l3686_368659


namespace NUMINAMATH_CALUDE_candy_redistribution_theorem_l3686_368607

/-- Represents the number of candies each friend has at a given stage -/
structure CandyState where
  vasya : ℕ
  petya : ℕ
  kolya : ℕ

/-- Represents a round of candy redistribution -/
def redistribute (state : CandyState) (giver : Fin 3) : CandyState :=
  match giver with
  | 0 => ⟨state.vasya - (state.petya + state.kolya), 2 * state.petya, 2 * state.kolya⟩
  | 1 => ⟨2 * state.vasya, state.petya - (state.vasya + state.kolya), 2 * state.kolya⟩
  | 2 => ⟨2 * state.vasya, 2 * state.petya, state.kolya - (state.vasya + state.petya)⟩

theorem candy_redistribution_theorem (initial : CandyState) :
  initial.kolya = 36 →
  (redistribute (redistribute (redistribute initial 0) 1) 2).kolya = 36 →
  initial.vasya + initial.petya + initial.kolya = 252 := by
  sorry


end NUMINAMATH_CALUDE_candy_redistribution_theorem_l3686_368607


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3686_368669

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (heq : y + 9 * x = x * y) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ b + 9 * a = a * b → x + y ≤ a + b ∧ x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3686_368669


namespace NUMINAMATH_CALUDE_f_has_one_real_root_l3686_368614

-- Define the polynomial
def f (x : ℝ) : ℝ := (x - 4) * (x^2 + 4*x + 5)

-- Theorem statement
theorem f_has_one_real_root : ∃! x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_one_real_root_l3686_368614


namespace NUMINAMATH_CALUDE_matrix_power_four_l3686_368648

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_power_four :
  A^4 = !![(-4), 6; (-6), 5] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l3686_368648


namespace NUMINAMATH_CALUDE_prize_buying_l3686_368680

/-- Given the conditions for prize buying, prove the number of pens and notebooks. -/
theorem prize_buying (x y : ℝ) (h1 : 60 * (x + 2*y) = 50 * (x + 3*y)) 
  (total_budget : ℝ) (h2 : total_budget = 60 * (x + 2*y)) : 
  (total_budget / x = 100) ∧ (total_budget / y = 300) := by
  sorry

end NUMINAMATH_CALUDE_prize_buying_l3686_368680


namespace NUMINAMATH_CALUDE_mrs_lee_june_percentage_l3686_368634

/-- Represents the Lee family's income structure -/
structure LeeIncome where
  total : ℝ
  mrs_lee : ℝ
  mr_lee : ℝ
  jack : ℝ
  rest : ℝ

/-- Calculates the total income for June based on May's income and the given changes -/
def june_total (may : LeeIncome) : ℝ :=
  1.2 * may.mrs_lee + 1.1 * may.mr_lee + 0.85 * may.jack + may.rest

/-- Theorem stating that Mrs. Lee's earnings in June are between 0% and 60% of the total income -/
theorem mrs_lee_june_percentage (may : LeeIncome)
  (h1 : may.mrs_lee = 0.5 * may.total)
  (h2 : may.total = may.mrs_lee + may.mr_lee + may.jack + may.rest)
  (h3 : may.total > 0) :
  0 < (1.2 * may.mrs_lee) / (june_total may) ∧ (1.2 * may.mrs_lee) / (june_total may) < 0.6 := by
  sorry

end NUMINAMATH_CALUDE_mrs_lee_june_percentage_l3686_368634


namespace NUMINAMATH_CALUDE_foci_coordinates_l3686_368675

/-- The curve equation -/
def curve (a x y : ℝ) : Prop :=
  x^2 / (a - 4) + y^2 / (a + 5) = 1

/-- The foci are fixed points -/
def fixed_foci (a : ℝ) : Prop :=
  ∃ x y : ℝ, ∀ b : ℝ, curve b x y → (x, y) = (0, 3) ∨ (x, y) = (0, -3)

/-- Theorem: If the foci of the curve are fixed points, then their coordinates are (0, ±3) -/
theorem foci_coordinates (a : ℝ) :
  fixed_foci a → ∃ x y : ℝ, curve a x y ∧ ((x, y) = (0, 3) ∨ (x, y) = (0, -3)) :=
sorry

end NUMINAMATH_CALUDE_foci_coordinates_l3686_368675


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3686_368678

-- Define the circle equation
def circle_eq (x y a : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y + 2*a = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the intersection of the circle and the line
def intersection (a : ℝ) : Prop := ∃ x y : ℝ, circle_eq x y a ∧ line_eq x y

-- Define the chord length
def chord_length (a : ℝ) : ℝ := 4

-- Theorem statement
theorem circle_line_intersection (a : ℝ) : 
  intersection a ∧ chord_length a = 4 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3686_368678


namespace NUMINAMATH_CALUDE_baseball_card_value_l3686_368642

/-- The value of a baseball card after four years of depreciation --/
def card_value (initial_value : ℝ) (year1_decrease year2_decrease year3_decrease year4_decrease : ℝ) : ℝ :=
  initial_value * (1 - year1_decrease) * (1 - year2_decrease) * (1 - year3_decrease) * (1 - year4_decrease)

/-- Theorem stating the final value of the baseball card after four years of depreciation --/
theorem baseball_card_value : 
  card_value 100 0.10 0.12 0.08 0.05 = 69.2208 := by
  sorry


end NUMINAMATH_CALUDE_baseball_card_value_l3686_368642


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l3686_368698

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l3686_368698


namespace NUMINAMATH_CALUDE_prob_different_colors_is_three_fourths_l3686_368689

/-- The set of colors for shorts -/
inductive ShortsColor
| Red
| Blue
| Green

/-- The set of colors for jerseys -/
inductive JerseyColor
| Red
| Blue
| Green
| Yellow

/-- The probability of choosing different colors for shorts and jersey -/
def prob_different_colors : ℚ := 3/4

/-- Theorem stating that the probability of choosing different colors for shorts and jersey is 3/4 -/
theorem prob_different_colors_is_three_fourths :
  prob_different_colors = 3/4 := by sorry

end NUMINAMATH_CALUDE_prob_different_colors_is_three_fourths_l3686_368689


namespace NUMINAMATH_CALUDE_tan_sin_30_identity_l3686_368666

theorem tan_sin_30_identity : 
  (Real.tan (30 * π / 180))^2 - (Real.sin (30 * π / 180))^2 = 
  (4 / 3) * (Real.tan (30 * π / 180))^2 * (Real.sin (30 * π / 180))^2 := by
sorry

end NUMINAMATH_CALUDE_tan_sin_30_identity_l3686_368666


namespace NUMINAMATH_CALUDE_original_cube_volume_l3686_368660

theorem original_cube_volume (s : ℝ) (h : (2 * s) ^ 3 = 1728) : s ^ 3 = 216 := by
  sorry

#check original_cube_volume

end NUMINAMATH_CALUDE_original_cube_volume_l3686_368660


namespace NUMINAMATH_CALUDE_total_pastries_l3686_368627

/-- Calculates the total number of pastries for four people given specific conditions -/
theorem total_pastries (grace_pastries : ℕ) (calvin_phoebe_diff : ℕ) (frank_diff : ℕ) : 
  grace_pastries = 30 →
  calvin_phoebe_diff = 5 →
  frank_diff = 8 →
  grace_pastries + 
  2 * (grace_pastries - calvin_phoebe_diff) + 
  (grace_pastries - calvin_phoebe_diff - frank_diff) = 97 :=
by sorry

end NUMINAMATH_CALUDE_total_pastries_l3686_368627


namespace NUMINAMATH_CALUDE_distance_center_M_to_line_L_is_zero_l3686_368622

/-- The circle M -/
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + 1 = 0

/-- The line L -/
def line_L (t x y : ℝ) : Prop :=
  x = 4*t + 3 ∧ y = 3*t + 1

/-- The center of circle M -/
def center_M : ℝ × ℝ :=
  (1, 2)

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a^2 + b^2)

theorem distance_center_M_to_line_L_is_zero :
  distance_point_to_line center_M 3 (-4) 5 = 0 := by sorry

end NUMINAMATH_CALUDE_distance_center_M_to_line_L_is_zero_l3686_368622


namespace NUMINAMATH_CALUDE_soup_ratio_l3686_368618

/-- Given the amount of beef bought, unused beef, and vegetables used, 
    calculate the ratio of vegetables to beef used in the soup -/
theorem soup_ratio (beef_bought : ℚ) (unused_beef : ℚ) (vegetables : ℚ) : 
  beef_bought = 4 → unused_beef = 1 → vegetables = 6 →
  vegetables / (beef_bought - unused_beef) = 2 := by sorry

end NUMINAMATH_CALUDE_soup_ratio_l3686_368618


namespace NUMINAMATH_CALUDE_additional_discount_percentage_l3686_368663

/-- Represents the price of duty shoes in cents -/
def full_price : ℕ := 8500

/-- Represents the first discount percentage for officers who served at least a year -/
def first_discount_percent : ℕ := 20

/-- Represents the price paid by officers who served at least three years in cents -/
def price_three_years : ℕ := 5100

/-- Calculates the price after the first discount -/
def price_after_first_discount : ℕ := full_price - (full_price * first_discount_percent / 100)

/-- Represents the additional discount percentage for officers who served at least three years -/
def additional_discount_percent : ℕ := 25

theorem additional_discount_percentage :
  (price_after_first_discount - price_three_years) * 100 / price_after_first_discount = additional_discount_percent := by
  sorry

end NUMINAMATH_CALUDE_additional_discount_percentage_l3686_368663


namespace NUMINAMATH_CALUDE_set_intersection_and_complement_l3686_368671

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -3 ∨ x > 1}

-- State the theorem
theorem set_intersection_and_complement :
  (A ∩ B = {x | 1 < x ∧ x ≤ 2}) ∧
  ((Aᶜ ∩ Bᶜ) = {x | -3 ≤ x ∧ x ≤ 0}) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_complement_l3686_368671


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l3686_368676

/-- Proves that the equation of a line passing through point (2, -3) with an inclination angle of 45° is x - y - 5 = 0 -/
theorem line_equation_through_point_with_inclination 
  (M : ℝ × ℝ) 
  (h_M : M = (2, -3)) 
  (α : ℝ) 
  (h_α : α = π / 4) : 
  ∀ x y : ℝ, (x - M.1) = (y - M.2) → x - y - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l3686_368676


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3686_368694

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S₃ : ℝ) :
  (∀ n, a (n + 1) - a n = 4) →  -- Common difference is 4
  ((a 3 + 2) / 2 = Real.sqrt (2 * S₃)) →  -- Arithmetic mean = Geometric mean condition
  (S₃ = a 1 + a 2 + a 3) →  -- Definition of S₃
  (a 10 = 38) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3686_368694


namespace NUMINAMATH_CALUDE_sector_area_l3686_368616

-- Define the parameters
def arc_length : ℝ := 1
def radius : ℝ := 4

-- Define the theorem
theorem sector_area : 
  let θ := arc_length / radius
  (1/2) * radius^2 * θ = 2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l3686_368616


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3686_368654

theorem smallest_integer_with_given_remainders : ∃! x : ℕ,
  x > 0 ∧
  x % 4 = 1 ∧
  x % 5 = 2 ∧
  x % 6 = 3 ∧
  ∀ y : ℕ, y > 0 → y < x →
    (y % 4 ≠ 1 ∨ y % 5 ≠ 2 ∨ y % 6 ≠ 3) :=
by
  use 117
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l3686_368654


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3686_368640

/-- The equation 3^(3x^3 - 9x^2 + 15x - 5) = 1 has exactly one real solution. -/
theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (3 : ℝ) ^ (3 * x^3 - 9 * x^2 + 15 * x - 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3686_368640


namespace NUMINAMATH_CALUDE_quadratic_one_root_l3686_368686

theorem quadratic_one_root (m : ℝ) : 
  (∀ x : ℝ, x^2 + 6*m*x + 4*m = 0 → (∀ y : ℝ, y^2 + 6*m*y + 4*m = 0 → x = y)) →
  m = 4/9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l3686_368686


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_17_l3686_368657

theorem smallest_k_for_64_power_gt_4_power_17 : 
  (∃ k : ℕ, 64^k > 4^17 ∧ ∀ m : ℕ, m < k → 64^m ≤ 4^17) ∧ 
  (∀ k : ℕ, 64^k > 4^17 → k ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_17_l3686_368657


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l3686_368623

theorem orange_juice_fraction (pitcher_capacity : ℚ) 
  (orange_juice_fraction : ℚ) (apple_juice_fraction : ℚ) : 
  pitcher_capacity > 0 →
  orange_juice_fraction = 1/4 →
  apple_juice_fraction = 3/8 →
  (orange_juice_fraction * pitcher_capacity) / (2 * pitcher_capacity) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_l3686_368623


namespace NUMINAMATH_CALUDE_product_mod_eight_l3686_368641

theorem product_mod_eight : (55 * 57) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_eight_l3686_368641


namespace NUMINAMATH_CALUDE_complex_modulus_l3686_368626

theorem complex_modulus (z : ℂ) : z / (1 + Complex.I) = -3 * Complex.I → Complex.abs z = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3686_368626


namespace NUMINAMATH_CALUDE_solve_equation_l3686_368602

theorem solve_equation (x : ℚ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3686_368602


namespace NUMINAMATH_CALUDE_smallest_number_in_special_triple_l3686_368658

theorem smallest_number_in_special_triple : 
  ∀ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) →  -- Three positive integers
    ((a + b + c) / 3 : ℚ) = 30 →  -- Arithmetic mean is 30
    b = 29 →  -- Median is 29
    max a (max b c) = b + 4 →  -- Median is 4 less than the largest number
    min a (min b c) = 28 :=  -- The smallest number is 28
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_in_special_triple_l3686_368658


namespace NUMINAMATH_CALUDE_sequence_problem_l3686_368647

theorem sequence_problem (a b c : ℤ → ℝ) 
  (h_positive : ∀ n, a n > 0 ∧ b n > 0 ∧ c n > 0)
  (h_a : ∀ n, a n ≥ (b (n+1) + c (n-1)) / 2)
  (h_b : ∀ n, b n ≥ (c (n+1) + a (n-1)) / 2)
  (h_c : ∀ n, c n ≥ (a (n+1) + b (n-1)) / 2)
  (h_init : a 0 = 26 ∧ b 0 = 6 ∧ c 0 = 2004) :
  a 2005 = 2004 ∧ b 2005 = 26 ∧ c 2005 = 6 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l3686_368647


namespace NUMINAMATH_CALUDE_max_distinct_numbers_with_prime_triple_sums_l3686_368646

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if the sum of any three numbers in a list is prime -/
def allTripleSumsPrime (l : List ℕ) : Prop :=
  ∀ a b c : ℕ, a ∈ l → b ∈ l → c ∈ l → a ≠ b → b ≠ c → a ≠ c → isPrime (a + b + c)

/-- The theorem stating that the maximum number of distinct natural numbers
    that can be chosen such that the sum of any three of them is prime is 4 -/
theorem max_distinct_numbers_with_prime_triple_sums :
  (∃ l : List ℕ, l.length = 4 ∧ l.Nodup ∧ allTripleSumsPrime l) ∧
  (∀ l : List ℕ, l.length > 4 → ¬(l.Nodup ∧ allTripleSumsPrime l)) :=
sorry

end NUMINAMATH_CALUDE_max_distinct_numbers_with_prime_triple_sums_l3686_368646


namespace NUMINAMATH_CALUDE_complex_product_equals_negative_25i_l3686_368665

theorem complex_product_equals_negative_25i :
  let Q : ℂ := 3 + 4*Complex.I
  let E : ℂ := -Complex.I
  let D : ℂ := 3 - 4*Complex.I
  Q * E * D = -25 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_negative_25i_l3686_368665


namespace NUMINAMATH_CALUDE_white_ducks_count_l3686_368687

theorem white_ducks_count (fish_per_white : ℕ) (fish_per_black : ℕ) (fish_per_multi : ℕ)
  (black_ducks : ℕ) (multi_ducks : ℕ) (total_fish : ℕ)
  (h1 : fish_per_white = 5)
  (h2 : fish_per_black = 10)
  (h3 : fish_per_multi = 12)
  (h4 : black_ducks = 7)
  (h5 : multi_ducks = 6)
  (h6 : total_fish = 157) :
  ∃ white_ducks : ℕ, white_ducks * fish_per_white + black_ducks * fish_per_black + multi_ducks * fish_per_multi = total_fish ∧ white_ducks = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_white_ducks_count_l3686_368687


namespace NUMINAMATH_CALUDE_inequality_proof_l3686_368691

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : z ≥ 1)
  (h4 : 1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3686_368691


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l3686_368633

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define symmetry about y-axis
def symmetric_about_y_axis (p q : Point) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_y_axis (a, 3) (4, b) →
  (a + b)^2008 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l3686_368633


namespace NUMINAMATH_CALUDE_square_sum_less_than_one_l3686_368653

theorem square_sum_less_than_one (x y : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : x^3 + y^3 = x - y) : 
  x^2 + y^2 < 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_less_than_one_l3686_368653


namespace NUMINAMATH_CALUDE_preceding_number_in_base_three_l3686_368609

/-- Converts a base-3 number to decimal --/
def baseThreeToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Converts a decimal number to base-3 --/
def decimalToBaseThree (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
    aux n []

theorem preceding_number_in_base_three (M : List Nat) (h : M = [2, 1, 0, 2, 1]) :
  decimalToBaseThree (baseThreeToDecimal M - 1) = [2, 1, 0, 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_preceding_number_in_base_three_l3686_368609


namespace NUMINAMATH_CALUDE_tablet_savings_l3686_368643

/-- The amount saved when buying a tablet from the cheaper store --/
theorem tablet_savings (list_price : ℝ) (tech_discount_percent : ℝ) (electro_discount : ℝ) :
  list_price = 120 →
  tech_discount_percent = 15 →
  electro_discount = 20 →
  list_price * (1 - tech_discount_percent / 100) - (list_price - electro_discount) = 2 :=
by sorry

end NUMINAMATH_CALUDE_tablet_savings_l3686_368643


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3686_368600

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 2) :
  (sin α - 4 * cos α) / (5 * sin α + 2 * cos α) = -1/6 ∧
  4 * sin α ^ 2 - 3 * sin α * cos α - 5 * cos α ^ 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3686_368600


namespace NUMINAMATH_CALUDE_train_speed_calculation_l3686_368685

/-- Proves that a train with given length and time to cross a pole has a specific speed in km/hr -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) :
  train_length = 320 →
  crossing_time = 7.999360051195905 →
  ∃ (speed_kmh : Real), 
    abs (speed_kmh - (train_length / crossing_time * 3.6)) < 0.001 ∧ 
    abs (speed_kmh - 144.018) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l3686_368685


namespace NUMINAMATH_CALUDE_sum_and_fraction_relation_l3686_368661

theorem sum_and_fraction_relation (a b : ℝ) 
  (sum_eq : a + b = 507)
  (frac_eq : (a - b) / b = 1 / 7) : 
  b - a = -34.428571 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_fraction_relation_l3686_368661


namespace NUMINAMATH_CALUDE_cube_sum_and_product_l3686_368655

theorem cube_sum_and_product (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  (a^3 + b^3 = 1008) ∧ ((a + b - (a - b)) * (a^3 + b^3) = 4032) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_product_l3686_368655


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3686_368624

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  b * c / a + a * c / b + a * b / c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3686_368624


namespace NUMINAMATH_CALUDE_whitney_whale_books_l3686_368604

/-- The number of whale books Whitney bought -/
def whale_books : ℕ := sorry

/-- The number of fish books Whitney bought -/
def fish_books : ℕ := 7

/-- The number of magazines Whitney bought -/
def magazines : ℕ := 3

/-- The cost of each book in dollars -/
def book_cost : ℕ := 11

/-- The cost of each magazine in dollars -/
def magazine_cost : ℕ := 1

/-- The total amount Whitney spent in dollars -/
def total_spent : ℕ := 179

/-- Theorem stating that Whitney bought 9 whale books -/
theorem whitney_whale_books : 
  whale_books * book_cost + fish_books * book_cost + magazines * magazine_cost = total_spent ∧
  whale_books = 9 := by sorry

end NUMINAMATH_CALUDE_whitney_whale_books_l3686_368604


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l3686_368628

-- Define the types for lines and planes
variable (L : Type) [LinearOrderedField L]
variable (P : Type) [LinearOrderedField P]

-- Define the relations
variable (perpendicular : L → P → Prop)
variable (perpendicular_lines : L → L → Prop)
variable (subset : L → P → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_and_line_in_plane
  (m n : L) (α : P)
  (h1 : perpendicular m α)
  (h2 : subset n α) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l3686_368628


namespace NUMINAMATH_CALUDE_series_sum_equals_three_l3686_368620

theorem series_sum_equals_three (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (5 * n - 1) / k^n = 13/4) : k = 3 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_three_l3686_368620


namespace NUMINAMATH_CALUDE_tshirts_sold_count_l3686_368632

/-- The revenue generated from selling t-shirts -/
def tshirt_revenue : ℕ := 4300

/-- The revenue generated from each t-shirt -/
def revenue_per_tshirt : ℕ := 215

/-- The number of t-shirts sold -/
def num_tshirts : ℕ := tshirt_revenue / revenue_per_tshirt

theorem tshirts_sold_count : num_tshirts = 20 := by
  sorry

end NUMINAMATH_CALUDE_tshirts_sold_count_l3686_368632


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3686_368670

theorem sin_2alpha_value (α : Real) (h : Real.sin α + Real.cos α = 1/3) : 
  Real.sin (2 * α) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3686_368670


namespace NUMINAMATH_CALUDE_sqrt_3_minus_1_power_l3686_368677

theorem sqrt_3_minus_1_power (N : ℕ) : 
  (Real.sqrt 3 - 1) ^ N = 4817152 - 2781184 * Real.sqrt 3 → N = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_minus_1_power_l3686_368677


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3686_368672

def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3686_368672


namespace NUMINAMATH_CALUDE_coplanar_vectors_lambda_l3686_368605

/-- Given vectors a, b, and c in ℝ³, prove that if they are coplanar and have the specified coordinates, then the third component of c equals 65/7. -/
theorem coplanar_vectors_lambda (a b c : ℝ × ℝ × ℝ) : 
  a = (2, -1, 3) →
  b = (-1, 4, -2) →
  c.1 = 7 ∧ c.2.1 = 5 →
  (∃ (x y : ℝ), c = x • a + y • b) →
  c.2.2 = 65/7 := by
  sorry

end NUMINAMATH_CALUDE_coplanar_vectors_lambda_l3686_368605


namespace NUMINAMATH_CALUDE_cos_sin_75_product_l3686_368651

theorem cos_sin_75_product (θ : Real) (h : θ = 75 * π / 180) : 
  (Real.cos θ + Real.sin θ) * (Real.cos θ - Real.sin θ) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_75_product_l3686_368651


namespace NUMINAMATH_CALUDE_ratio_children_to_adults_l3686_368652

def total_people : ℕ := 120
def children : ℕ := 80

theorem ratio_children_to_adults :
  (children : ℚ) / (total_people - children : ℚ) = 2 / 1 := by sorry

end NUMINAMATH_CALUDE_ratio_children_to_adults_l3686_368652


namespace NUMINAMATH_CALUDE_tower_height_calculation_l3686_368601

/-- Given a mountain and a tower, if the angles of depression from the top of the mountain
    to the top and bottom of the tower are as specified, then the height of the tower is 200m. -/
theorem tower_height_calculation (mountain_height : ℝ) (angle_to_top angle_to_bottom : ℝ) :
  mountain_height = 300 →
  angle_to_top = 30 * π / 180 →
  angle_to_bottom = 60 * π / 180 →
  ∃ (tower_height : ℝ), tower_height = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_tower_height_calculation_l3686_368601


namespace NUMINAMATH_CALUDE_boys_without_calculators_l3686_368688

theorem boys_without_calculators (total_boys : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ)
  (h1 : total_boys = 20)
  (h2 : students_with_calculators = 26)
  (h3 : girls_with_calculators = 15) :
  total_boys - (students_with_calculators - girls_with_calculators) = 9 := by
sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l3686_368688


namespace NUMINAMATH_CALUDE_last_two_digits_33_divisible_by_prime_gt_7_l3686_368635

theorem last_two_digits_33_divisible_by_prime_gt_7 (n : ℕ) :
  (∃ k : ℕ, n = 100 * k + 33) →
  ∃ p : ℕ, p > 7 ∧ Prime p ∧ p ∣ n :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_33_divisible_by_prime_gt_7_l3686_368635


namespace NUMINAMATH_CALUDE_lines_not_parallel_l3686_368615

-- Define the types for our objects
variable (Point Line Plane : Type)

-- Define the relationships
variable (contains : Plane → Line → Prop)
variable (not_contains : Plane → Line → Prop)
variable (on_line : Point → Line → Prop)
variable (in_plane : Point → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_not_parallel 
  (m n : Line) (α : Plane) (A : Point)
  (h1 : not_contains α m)
  (h2 : contains α n)
  (h3 : on_line A m)
  (h4 : in_plane A α) :
  ¬(parallel m n) :=
by sorry

end NUMINAMATH_CALUDE_lines_not_parallel_l3686_368615


namespace NUMINAMATH_CALUDE_smallest_multiple_of_4_and_14_l3686_368639

theorem smallest_multiple_of_4_and_14 : ∀ a : ℕ, a > 0 ∧ 4 ∣ a ∧ 14 ∣ a → a ≥ 28 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_4_and_14_l3686_368639


namespace NUMINAMATH_CALUDE_total_campers_count_l3686_368693

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 36

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 13

/-- The number of campers who went rowing in the evening -/
def evening_campers : ℕ := 49

/-- The total number of campers who went rowing -/
def total_campers : ℕ := morning_campers + afternoon_campers + evening_campers

theorem total_campers_count : total_campers = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_count_l3686_368693


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3686_368621

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) → -8 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3686_368621


namespace NUMINAMATH_CALUDE_element_in_set_l3686_368696

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l3686_368696


namespace NUMINAMATH_CALUDE_negation_equivalence_l3686_368664

theorem negation_equivalence :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3686_368664
