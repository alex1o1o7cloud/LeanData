import Mathlib

namespace NUMINAMATH_CALUDE_factorization_equality_l1574_157417

theorem factorization_equality (x : ℝ) : 12 * x^2 + 18 * x - 24 = 6 * (2 * x - 1) * (x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1574_157417


namespace NUMINAMATH_CALUDE_mary_warm_hours_l1574_157476

/-- The number of sticks of wood produced by chopping up a chair -/
def sticks_per_chair : ℕ := 6

/-- The number of sticks of wood produced by chopping up a table -/
def sticks_per_table : ℕ := 9

/-- The number of sticks of wood produced by chopping up a stool -/
def sticks_per_stool : ℕ := 2

/-- The number of sticks of wood Mary needs to burn per hour to stay warm -/
def sticks_per_hour : ℕ := 5

/-- The number of chairs Mary chops up -/
def chairs_chopped : ℕ := 18

/-- The number of tables Mary chops up -/
def tables_chopped : ℕ := 6

/-- The number of stools Mary chops up -/
def stools_chopped : ℕ := 4

/-- Theorem stating how many hours Mary can keep warm -/
theorem mary_warm_hours : 
  (chairs_chopped * sticks_per_chair + 
   tables_chopped * sticks_per_table + 
   stools_chopped * sticks_per_stool) / sticks_per_hour = 34 := by
  sorry

end NUMINAMATH_CALUDE_mary_warm_hours_l1574_157476


namespace NUMINAMATH_CALUDE_equation_solutions_l1574_157477

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -2 + Real.sqrt 5 ∧ x₂ = -2 - Real.sqrt 5 ∧
    x₁^2 + 4*x₁ - 1 = 0 ∧ x₂^2 + 4*x₂ - 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 3 ∧ y₂ = 1 ∧
    (y₁ - 3)^2 + 2*y₁*(y₁ - 3) = 0 ∧ (y₂ - 3)^2 + 2*y₂*(y₂ - 3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1574_157477


namespace NUMINAMATH_CALUDE_ab_value_l1574_157494

theorem ab_value (a b : ℝ) (h : (a - 2)^2 + |b + 3| = 0) : a * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1574_157494


namespace NUMINAMATH_CALUDE_intersection_point_l1574_157469

theorem intersection_point (x y : ℚ) : 
  (x = 40/17 ∧ y = 21/17) ↔ (3*x + 4*y = 12 ∧ 7*x - 2*y = 14) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l1574_157469


namespace NUMINAMATH_CALUDE_total_geckos_sold_l1574_157466

def geckos_sold_last_year : ℕ := 86

theorem total_geckos_sold (geckos_sold_before : ℕ) 
  (h : geckos_sold_before = 2 * geckos_sold_last_year) : 
  geckos_sold_last_year + geckos_sold_before = 258 := by
  sorry

end NUMINAMATH_CALUDE_total_geckos_sold_l1574_157466


namespace NUMINAMATH_CALUDE_max_a_value_l1574_157432

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 4*x + 3 else -x^2 - 2*x + 3

theorem max_a_value (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 1), f (x + a) ≥ f (2*a - x)) →
  a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1574_157432


namespace NUMINAMATH_CALUDE_sequence_general_term_l1574_157478

theorem sequence_general_term (a : ℕ → ℝ) :
  (a 1 = 1) →
  (∀ n : ℕ, n > 1 → a n = 2 * a (n - 1) + 1) →
  (∀ n : ℕ, n > 0 → a n = 2^n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1574_157478


namespace NUMINAMATH_CALUDE_parabola_c_value_l1574_157487

/-- A parabola passing through two points -/
structure Parabola where
  b : ℝ
  c : ℝ
  pass_point_1 : 2 * 1^2 + b * 1 + c = 4
  pass_point_2 : 2 * 3^2 + b * 3 + c = 16

/-- The value of c for the parabola -/
def c_value (p : Parabola) : ℝ := p.c

theorem parabola_c_value (p : Parabola) : c_value p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1574_157487


namespace NUMINAMATH_CALUDE_unique_valid_statement_l1574_157428

theorem unique_valid_statement : ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧
  (Real.sqrt (a^2 + b^2) = |a - b|) ∧
  ¬(Real.sqrt (a^2 + b^2) = a^2 - b^2) ∧
  ¬(Real.sqrt (a^2 + b^2) = a + b) ∧
  ¬(Real.sqrt (a^2 + b^2) = |a| + |b|) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_valid_statement_l1574_157428


namespace NUMINAMATH_CALUDE_sum_45_25_in_base5_l1574_157474

/-- Converts a decimal number to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a base 5 number to decimal -/
def fromBase5 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two base 5 numbers -/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_45_25_in_base5 :
  let a := 45
  let b := 25
  let a_base5 := toBase5 a
  let b_base5 := toBase5 b
  let sum_base5 := addBase5 a_base5 b_base5
  sum_base5 = [2, 3, 0] := by
  sorry

end NUMINAMATH_CALUDE_sum_45_25_in_base5_l1574_157474


namespace NUMINAMATH_CALUDE_triangle_angle_measurement_l1574_157407

theorem triangle_angle_measurement (A B C : ℝ) : 
  A = 70 ∧ 
  B = 2 * C + 30 ∧ 
  A + B + C = 180 →
  C = 80 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measurement_l1574_157407


namespace NUMINAMATH_CALUDE_select_subjects_with_distinct_grades_l1574_157462

/-- Represents a grade for a single subject -/
def Grade : Type := ℕ

/-- Represents the grades of a student for all subjects -/
def StudentGrades : Type := Fin 12 → Grade

/-- The number of students -/
def numStudents : ℕ := 7

/-- The number of subjects -/
def numSubjects : ℕ := 12

/-- The number of subjects to be selected -/
def numSelectedSubjects : ℕ := 6

theorem select_subjects_with_distinct_grades 
  (grades : Fin numStudents → StudentGrades)
  (h : ∀ i j, i ≠ j → ∃ k, grades i k ≠ grades j k) :
  ∃ (selected : Fin numSelectedSubjects → Fin numSubjects),
    (∀ i j, i ≠ j → ∃ k, grades i (selected k) ≠ grades j (selected k)) :=
sorry

end NUMINAMATH_CALUDE_select_subjects_with_distinct_grades_l1574_157462


namespace NUMINAMATH_CALUDE_expression_simplification_l1574_157498

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 + 1 / (x^2 - 1)) / (x^2 / (x^2 + 2*x + 1)) = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1574_157498


namespace NUMINAMATH_CALUDE_rectangle_fold_theorem_l1574_157468

/-- Given a rectangle ABCD with AB = 4 and BC = 10, folded along a line through A
    such that A meets CD at point G where DG = 3, and C touches the extension of AB at point E,
    prove that the length of segment CE is 1. -/
theorem rectangle_fold_theorem (A B C D G E : ℝ × ℝ) : 
  let AB : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC : ℝ := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let DG : ℝ := Real.sqrt ((D.1 - G.1)^2 + (D.2 - G.2)^2)
  let CE : ℝ := Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2)
  AB = 4 →
  BC = 10 →
  DG = 3 →
  (A.1 - B.1) * (C.1 - D.1) + (A.2 - B.2) * (C.2 - D.2) = 0 → -- rectangle condition
  (A.1 = G.1 ∧ A.2 = G.2) → -- A meets CD at G
  (E.1 - A.1) * (B.1 - A.1) + (E.2 - A.2) * (B.2 - A.2) ≥ 0 → -- C touches extension of AB
  CE = 1 := by
sorry

end NUMINAMATH_CALUDE_rectangle_fold_theorem_l1574_157468


namespace NUMINAMATH_CALUDE_initial_men_count_l1574_157473

/-- The number of days it takes the initial group to complete the job -/
def initial_days : ℕ := 15

/-- The number of men in the second group -/
def second_group_men : ℕ := 18

/-- The number of days it takes the second group to complete the job -/
def second_group_days : ℕ := 20

/-- The total amount of work in man-days -/
def total_work : ℕ := second_group_men * second_group_days

/-- The number of men initially working on the job -/
def initial_men : ℕ := total_work / initial_days

theorem initial_men_count : initial_men = 24 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l1574_157473


namespace NUMINAMATH_CALUDE_cube_with_72cm_edges_l1574_157435

/-- Represents a cube with edge length in centimeters -/
structure Cube where
  edgeLength : ℝ
  edgeLength_pos : edgeLength > 0

/-- The sum of all edge lengths of a cube -/
def Cube.sumOfEdges (c : Cube) : ℝ := 12 * c.edgeLength

/-- The volume of a cube -/
def Cube.volume (c : Cube) : ℝ := c.edgeLength ^ 3

/-- The surface area of a cube -/
def Cube.surfaceArea (c : Cube) : ℝ := 6 * c.edgeLength ^ 2

/-- Theorem stating the properties of a cube with sum of edges 72 cm -/
theorem cube_with_72cm_edges (c : Cube) 
  (h : c.sumOfEdges = 72) : 
  c.volume = 216 ∧ c.surfaceArea = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_72cm_edges_l1574_157435


namespace NUMINAMATH_CALUDE_beth_age_proof_l1574_157420

/-- Beth's current age -/
def beth_age : ℕ := 18

/-- Beth's sister's current age -/
def sister_age : ℕ := 5

/-- Years into the future when Beth will be twice her sister's age -/
def future_years : ℕ := 8

theorem beth_age_proof :
  beth_age = 18 ∧
  sister_age = 5 ∧
  beth_age + future_years = 2 * (sister_age + future_years) :=
by sorry

end NUMINAMATH_CALUDE_beth_age_proof_l1574_157420


namespace NUMINAMATH_CALUDE_max_initial_states_l1574_157489

/-- Represents the state of friendships between sheep and the wolf -/
structure SheepcoteState (n : ℕ) :=
  (wolf_friends : Finset (Fin n))
  (sheep_friendships : Finset (Fin n × Fin n))

/-- Represents the process of the wolf eating sheep -/
def eat_sheep (state : SheepcoteState n) : Option (SheepcoteState n) :=
  sorry

/-- Checks if all sheep can be eaten given an initial state -/
def can_eat_all_sheep (initial_state : SheepcoteState n) : Prop :=
  sorry

/-- The number of valid initial states -/
def num_valid_initial_states (n : ℕ) : ℕ :=
  sorry

theorem max_initial_states (n : ℕ) :
  num_valid_initial_states n = 2^(n-1) :=
sorry

end NUMINAMATH_CALUDE_max_initial_states_l1574_157489


namespace NUMINAMATH_CALUDE_line_equation_for_triangle_l1574_157427

/-- Given a line passing through (-a, 0) and cutting a triangle with area T in the second quadrant,
    prove that the equation of the line is 2Tx - a²y + 2aT = 0 --/
theorem line_equation_for_triangle (a T : ℝ) (h_a : a > 0) (h_T : T > 0) :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b → (x = -a ∧ y = 0) ∨ (x ≥ 0 ∧ y ≥ 0)) ∧ 
    (1/2 * a * (b : ℝ) = T) ∧
    (∀ x y : ℝ, y = m * x + b ↔ 2 * T * x - a^2 * y + 2 * a * T = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_for_triangle_l1574_157427


namespace NUMINAMATH_CALUDE_garden_furniture_cost_ratio_l1574_157485

/-- Given a garden table and bench with a combined cost of 750 and the bench costing 250,
    prove that the ratio of the table's cost to the bench's cost is 2:1. -/
theorem garden_furniture_cost_ratio :
  ∀ (table_cost bench_cost : ℝ),
    bench_cost = 250 →
    table_cost + bench_cost = 750 →
    ∃ (n : ℕ), table_cost = n * bench_cost →
    table_cost / bench_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_furniture_cost_ratio_l1574_157485


namespace NUMINAMATH_CALUDE_quadratic_no_roots_positive_c_l1574_157470

/-- A quadratic polynomial with no real roots and positive sum of coefficients has a positive constant term. -/
theorem quadratic_no_roots_positive_c (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c ≠ 0) →  -- no real roots
  a + b + c > 0 →                   -- sum of coefficients is positive
  c > 0 :=                          -- constant term is positive
by sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_positive_c_l1574_157470


namespace NUMINAMATH_CALUDE_delivery_pay_calculation_l1574_157440

/-- The amount paid per delivery for Oula and Tona --/
def amount_per_delivery : ℝ := sorry

/-- The number of deliveries made by Oula --/
def oula_deliveries : ℕ := 96

/-- The number of deliveries made by Tona --/
def tona_deliveries : ℕ := (3 * oula_deliveries) / 4

/-- The difference in pay between Oula and Tona --/
def pay_difference : ℝ := 2400

theorem delivery_pay_calculation :
  amount_per_delivery * (oula_deliveries - tona_deliveries : ℝ) = pay_difference ∧
  amount_per_delivery = 100 := by sorry

end NUMINAMATH_CALUDE_delivery_pay_calculation_l1574_157440


namespace NUMINAMATH_CALUDE_product_equals_zero_l1574_157422

theorem product_equals_zero (a : ℤ) (h : a = 9) : 
  (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l1574_157422


namespace NUMINAMATH_CALUDE_john_taller_than_lena_l1574_157401

/-- Proves that John is 15 cm taller than Lena given the problem conditions -/
theorem john_taller_than_lena (john_height rebeca_height lena_height : ℕ) :
  john_height = 152 →
  john_height = rebeca_height - 6 →
  lena_height + rebeca_height = 295 →
  john_height - lena_height = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_john_taller_than_lena_l1574_157401


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1574_157415

-- Define a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function
def f (x : ℝ) : ℝ := 2 * x^2 - 7

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1574_157415


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_halves_l1574_157444

theorem opposite_of_negative_three_halves :
  -((-3 : ℚ) / 2) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_halves_l1574_157444


namespace NUMINAMATH_CALUDE_total_distance_is_9km_l1574_157483

/-- Represents the travel plans from the city bus station to Tianbo Mountain -/
inductive TravelPlan
| BusOnly
| BikeOnly
| BikeThenBus
| BusThenBike

/-- Represents the journey from the city bus station to Tianbo Mountain -/
structure Journey where
  distance_to_hehua : ℝ
  distance_from_hehua : ℝ
  bus_speed : ℝ
  bike_speed : ℝ
  bus_stop_time : ℝ

/-- The actual journey based on the problem description -/
def actual_journey : Journey where
  distance_to_hehua := 6
  distance_from_hehua := 3
  bus_speed := 24
  bike_speed := 16
  bus_stop_time := 0.5

/-- Theorem stating that the total distance is 9 km -/
theorem total_distance_is_9km (j : Journey) :
  j.distance_to_hehua + j.distance_from_hehua = 9 ∧
  j.distance_to_hehua = 6 ∧
  j.distance_from_hehua = 3 ∧
  j.bus_speed = 24 ∧
  j.bike_speed = 16 ∧
  j.bus_stop_time = 0.5 ∧
  (j.distance_to_hehua + j.distance_from_hehua) / j.bus_speed + j.bus_stop_time =
    (j.distance_to_hehua + j.distance_from_hehua + 1) / j.bike_speed ∧
  j.distance_to_hehua / j.bus_speed = 4 / j.bike_speed ∧
  (j.distance_to_hehua / j.bus_speed + j.bus_stop_time + j.distance_from_hehua / j.bike_speed) =
    ((j.distance_to_hehua + j.distance_from_hehua) / j.bus_speed + j.bus_stop_time - 0.25) :=
by sorry

#check total_distance_is_9km actual_journey

end NUMINAMATH_CALUDE_total_distance_is_9km_l1574_157483


namespace NUMINAMATH_CALUDE_unbounded_function_l1574_157457

def IsUnbounded (f : ℝ → ℝ) : Prop :=
  ∀ M : ℝ, ∃ x : ℝ, f x > M

theorem unbounded_function (f : ℝ → ℝ) 
  (h_pos : ∀ x, 0 < f x) 
  (h_ineq : ∀ x y, 0 < x → 0 < y → (f (x + f y))^2 ≥ f x * (f (x + f y) + f y)) : 
  IsUnbounded f := by
  sorry

end NUMINAMATH_CALUDE_unbounded_function_l1574_157457


namespace NUMINAMATH_CALUDE_binary_110011_is_51_l1574_157429

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_110011_is_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_is_51_l1574_157429


namespace NUMINAMATH_CALUDE_gas_pressure_final_l1574_157464

/-- Given a gas with pressure inversely proportional to volume, prove the final pressure -/
theorem gas_pressure_final (p₀ v₀ v₁ v₂ : ℝ) (h₀ : p₀ > 0) (h₁ : v₀ > 0) (h₂ : v₁ > 0) (h₃ : v₂ > 0)
  (h_initial : p₀ * v₀ = 6 * 3.6)
  (h_v₁ : v₁ = 7.2)
  (h_v₂ : v₂ = 3.6)
  (h_half : v₂ = v₀) :
  ∃ (p₂ : ℝ), p₂ * v₂ = p₀ * v₀ ∧ p₂ = 6 := by
  sorry

#check gas_pressure_final

end NUMINAMATH_CALUDE_gas_pressure_final_l1574_157464


namespace NUMINAMATH_CALUDE_negation_equivalence_l1574_157449

theorem negation_equivalence (a b : ℝ) : 
  ¬(a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a^2 + b^2 ≠ 0 → a ≠ 0 ∨ b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1574_157449


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_slope_l1574_157412

/-- A line that does not pass through the third quadrant has a non-positive slope -/
theorem line_not_in_third_quadrant_slope (k : ℝ) :
  (∀ x y : ℝ, y = k * x + 3 → ¬(x < 0 ∧ y < 0)) →
  k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_slope_l1574_157412


namespace NUMINAMATH_CALUDE_measure_45_minutes_l1574_157453

/-- Represents a cord that can be burned --/
structure Cord :=
  (burn_time : ℝ)
  (burn_rate_uniform : Bool)

/-- Represents the state of burning a cord --/
inductive BurnState
  | Unlit
  | LitOneEnd (time : ℝ)
  | LitBothEnds (time : ℝ)
  | Burned

/-- Represents the measurement setup --/
structure MeasurementSetup :=
  (cord1 : Cord)
  (cord2 : Cord)
  (state1 : BurnState)
  (state2 : BurnState)

/-- The main theorem stating that 45 minutes can be measured --/
theorem measure_45_minutes 
  (c1 c2 : Cord) 
  (h1 : c1.burn_time = 60) 
  (h2 : c2.burn_time = 60) : 
  ∃ (process : List MeasurementSetup), 
    (∃ (t : ℝ), t = 45 ∧ 
      (∃ (final : MeasurementSetup), final ∈ process ∧ 
        final.state1 = BurnState.Burned ∧ 
        final.state2 = BurnState.Burned)) :=
sorry

end NUMINAMATH_CALUDE_measure_45_minutes_l1574_157453


namespace NUMINAMATH_CALUDE_characterization_of_M_l1574_157455

/-- S(n) represents the sum of digits of n -/
def S (n : ℕ) : ℕ := sorry

/-- M satisfies the given property -/
def satisfies_property (M : ℕ) : Prop :=
  M > 0 ∧ ∀ k : ℕ, 0 < k ∧ k ≤ M → S (M * k) = S M

/-- Main theorem -/
theorem characterization_of_M :
  ∀ M : ℕ, satisfies_property M ↔ ∃ n : ℕ, n > 0 ∧ M = 10^n - 1 :=
sorry

end NUMINAMATH_CALUDE_characterization_of_M_l1574_157455


namespace NUMINAMATH_CALUDE_cube_root_of_four_l1574_157461

theorem cube_root_of_four (x : ℝ) : x^3 = 4 → x = 4^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_of_four_l1574_157461


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l1574_157439

theorem triangle_side_inequality (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a / (1 + a)) + (b / (1 + b)) ≥ c / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_inequality_l1574_157439


namespace NUMINAMATH_CALUDE_intersection_distance_l1574_157431

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  directrix : ℝ  -- x-coordinate of the vertical directrix

/-- Check if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- Check if a point is on the parabola -/
def isOnParabola (p : Point) (pa : Parabola) : Prop :=
  p.x = 2 * (5 + 2 * Real.sqrt 3) * p.y^2 + (5 + 2 * Real.sqrt 3) / 2

/-- The main theorem -/
theorem intersection_distance (e : Ellipse) (pa : Parabola) 
    (p1 p2 : Point) : 
    e.center = Point.mk 0 0 →
    e.a = 4 →
    e.b = 2 →
    pa.directrix = 5 →
    pa.focus = Point.mk (2 * Real.sqrt 3) 0 →
    isOnEllipse p1 e →
    isOnEllipse p2 e →
    isOnParabola p1 pa →
    isOnParabola p2 pa →
    p1 ≠ p2 →
    ∃ (d : ℝ), d = 2 * |p1.y - p2.y| ∧ 
               d = Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2) :=
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l1574_157431


namespace NUMINAMATH_CALUDE_chocolate_division_l1574_157442

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_for_shaina : ℕ) : 
  total_chocolate = 70 / 7 →
  num_piles = 5 →
  piles_for_shaina = 2 →
  (total_chocolate / num_piles) * piles_for_shaina = 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l1574_157442


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1574_157465

theorem simplify_fraction_product (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 5) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) * (x^2 - 6*x + 8) / (x^2 - 8*x + 15) =
  ((x - 1) * (x - 2) * (x - 4)) / ((x - 3) * (x - 5)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1574_157465


namespace NUMINAMATH_CALUDE_equation_solution_l1574_157452

theorem equation_solution : 
  ∃! x : ℝ, (1 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) ∧ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1574_157452


namespace NUMINAMATH_CALUDE_successive_discounts_equivalent_to_single_discount_l1574_157436

-- Define the original price and discount rates
def original_price : ℝ := 50
def discount1 : ℝ := 0.30
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.10

-- Define the equivalent single discount
def equivalent_discount : ℝ := 0.4645

-- Theorem statement
theorem successive_discounts_equivalent_to_single_discount :
  (1 - discount1) * (1 - discount2) * (1 - discount3) = 1 - equivalent_discount :=
by sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalent_to_single_discount_l1574_157436


namespace NUMINAMATH_CALUDE_max_consecutive_semi_primes_l1574_157408

/-- A natural number greater than 25 is semi-prime if it is the sum of two different prime numbers -/
def IsSemiPrime (n : ℕ) : Prop :=
  n > 25 ∧ ∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ n = p + q

/-- The maximum number of consecutive natural numbers that are semi-prime is 5 -/
theorem max_consecutive_semi_primes :
  ∀ k : ℕ, (∀ n : ℕ, ∀ i : ℕ, i < k → IsSemiPrime (n + i)) → k ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_semi_primes_l1574_157408


namespace NUMINAMATH_CALUDE_sequence_formula_l1574_157404

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem sequence_formula :
  let a₁ : ℝ := 20
  let d : ℝ := -9
  ∀ n : ℕ, arithmetic_sequence a₁ d n = -9 * n + 29 := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l1574_157404


namespace NUMINAMATH_CALUDE_total_spent_is_413_06_l1574_157491

/-- Calculates the total amount spent including sales tax -/
def total_spent (speakers_cost cd_player_cost tires_cost tax_rate : ℝ) : ℝ :=
  let subtotal := speakers_cost + cd_player_cost + tires_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Theorem stating that the total amount spent is $413.06 -/
theorem total_spent_is_413_06 :
  total_spent 136.01 139.38 112.46 0.065 = 413.06 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_413_06_l1574_157491


namespace NUMINAMATH_CALUDE_sum_zero_iff_fractions_sum_neg_two_l1574_157447

theorem sum_zero_iff_fractions_sum_neg_two (x y : ℝ) (h : x * y ≠ 0) :
  x + y = 0 ↔ x / y + y / x = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_iff_fractions_sum_neg_two_l1574_157447


namespace NUMINAMATH_CALUDE_total_running_time_l1574_157454

/-- Represents the running data for a single day -/
structure DailyRun where
  distance : ℕ
  basePace : ℕ
  additionalTime : ℕ

/-- Calculates the total time for a single run -/
def runTime (run : DailyRun) : ℕ :=
  run.distance * (run.basePace + run.additionalTime)

/-- The running data for each day of the week -/
def weeklyRuns : List DailyRun :=
  [
    { distance := 3, basePace := 10, additionalTime := 1 },  -- Monday
    { distance := 4, basePace := 9,  additionalTime := 1 },  -- Tuesday
    { distance := 6, basePace := 12, additionalTime := 0 },  -- Wednesday
    { distance := 8, basePace := 8,  additionalTime := 2 },  -- Thursday
    { distance := 3, basePace := 10, additionalTime := 0 }   -- Friday
  ]

/-- The theorem stating that the total running time for the week is 255 minutes -/
theorem total_running_time :
  (weeklyRuns.map runTime).sum = 255 := by
  sorry


end NUMINAMATH_CALUDE_total_running_time_l1574_157454


namespace NUMINAMATH_CALUDE_footprint_calculation_l1574_157437

/-- Calculates the total number of footprints left by Pogo and Grimzi -/
def total_footprints (pogo_rate : ℚ) (grimzi_rate : ℚ) (distance : ℚ) : ℚ :=
  pogo_rate * distance + grimzi_rate * distance

/-- Theorem stating the total number of footprints left by Pogo and Grimzi -/
theorem footprint_calculation :
  let pogo_rate : ℚ := 4
  let grimzi_rate : ℚ := 1/2
  let distance : ℚ := 6000
  total_footprints pogo_rate grimzi_rate distance = 27000 := by
sorry

#eval total_footprints 4 (1/2) 6000

end NUMINAMATH_CALUDE_footprint_calculation_l1574_157437


namespace NUMINAMATH_CALUDE_adams_apples_l1574_157490

/-- 
Given:
- Jackie has 10 apples
- Jackie has 2 more apples than Adam
Prove that Adam has 8 apples
-/
theorem adams_apples (jackie_apples : ℕ) (adam_apples : ℕ) 
  (h1 : jackie_apples = 10)
  (h2 : jackie_apples = adam_apples + 2) : 
  adam_apples = 8 := by
  sorry

end NUMINAMATH_CALUDE_adams_apples_l1574_157490


namespace NUMINAMATH_CALUDE_range_of_x_l1574_157486

theorem range_of_x (x : ℝ) : 
  (1 / x < 3) → (1 / x > -2) → (2 * x - 5 > 0) → (x > 5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1574_157486


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l1574_157430

/-- Tetrahedron PQRS with given properties -/
structure Tetrahedron where
  /-- Angle between faces PQR and QRS -/
  angle : ℝ
  /-- Area of face PQR -/
  area_PQR : ℝ
  /-- Area of face QRS -/
  area_QRS : ℝ
  /-- Length of edge QR -/
  length_QR : ℝ

/-- The volume of a tetrahedron with the given properties -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the specific tetrahedron -/
theorem tetrahedron_volume (t : Tetrahedron) 
  (h1 : t.angle = 45 * π / 180)
  (h2 : t.area_PQR = 150)
  (h3 : t.area_QRS = 50)
  (h4 : t.length_QR = 10) : 
  volume t = 250 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l1574_157430


namespace NUMINAMATH_CALUDE_triangle_inequality_l1574_157402

theorem triangle_inequality (R r p : ℝ) (a b c : ℝ) (m_a m_b m_c : ℝ) : 
  R > 0 → r > 0 → p > 0 → a > 0 → b > 0 → c > 0 →
  R * r = a * b * c / (4 * p) →
  a * b * c ≤ 8 * p^3 →
  (a + b + c)^2 ≤ 3 * (a^2 + b^2 + c^2) →
  p^2 ≤ (m_a^2 + m_b^2 + m_c^2) / 4 →
  m_a^2 + m_b^2 + m_c^2 ≤ 27 * R^2 / 4 →
  27 * R * r ≤ 2 * p^2 ∧ 2 * p^2 ≤ 27 * R^2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1574_157402


namespace NUMINAMATH_CALUDE_line_through_AB_l1574_157409

-- Define the lines and points
def line1 (a₁ b₁ x y : ℝ) : Prop := a₁ * x + b₁ * y + 1 = 0
def line2 (a₂ b₂ x y : ℝ) : Prop := a₂ * x + b₂ * y + 1 = 0
def point_P : ℝ × ℝ := (2, 3)
def point_A (a₁ b₁ : ℝ) : ℝ × ℝ := (a₁, b₁)
def point_B (a₂ b₂ : ℝ) : ℝ × ℝ := (a₂, b₂)

-- Define the theorem
theorem line_through_AB (a₁ b₁ a₂ b₂ : ℝ) 
  (h1 : line1 a₁ b₁ (point_P.1) (point_P.2))
  (h2 : line2 a₂ b₂ (point_P.1) (point_P.2))
  (h3 : a₁ ≠ a₂) :
  ∃ (x y : ℝ), 2 * x + 3 * y + 1 = 0 ↔ 
    (y - b₁) / (x - a₁) = (b₂ - b₁) / (a₂ - a₁) :=
by sorry

end NUMINAMATH_CALUDE_line_through_AB_l1574_157409


namespace NUMINAMATH_CALUDE_exists_indivisible_treasure_l1574_157413

/-- Represents a treasure of gold bars -/
structure Treasure where
  num_bars : ℕ
  total_value : ℕ
  bar_values : Fin num_bars → ℕ
  sum_constraint : (Finset.univ.sum bar_values) = total_value

/-- Represents an even division of a treasure among pirates -/
def EvenDivision (t : Treasure) (num_pirates : ℕ) : Prop :=
  ∃ (division : Fin t.num_bars → Fin num_pirates),
    ∀ p : Fin num_pirates,
      (Finset.univ.filter (λ i => division i = p)).sum t.bar_values =
        t.total_value / num_pirates

/-- The main theorem stating that there exists a treasure that cannot be evenly divided -/
theorem exists_indivisible_treasure :
  ∃ (t : Treasure),
    t.num_bars = 240 ∧
    t.total_value = 360 ∧
    (∀ i : Fin t.num_bars, t.bar_values i > 0) ∧
    ¬(EvenDivision t 3) := by
  sorry

end NUMINAMATH_CALUDE_exists_indivisible_treasure_l1574_157413


namespace NUMINAMATH_CALUDE_count_special_pairs_count_special_pairs_is_192_l1574_157403

/-- The number of pairs (i, j) of integers where 0 ≤ i < j ≤ 50 and 3003 divides 10^j - 10^i -/
theorem count_special_pairs : Nat := by
  sorry

/-- The count of special pairs is 192 -/
theorem count_special_pairs_is_192 : count_special_pairs = 192 := by
  sorry

end NUMINAMATH_CALUDE_count_special_pairs_count_special_pairs_is_192_l1574_157403


namespace NUMINAMATH_CALUDE_smallest_angle_for_trig_equation_l1574_157484

theorem smallest_angle_for_trig_equation : 
  ∃ y : ℝ, y > 0 ∧ 
  (∀ z : ℝ, z > 0 → 6 * Real.sin z * Real.cos z ^ 3 - 6 * Real.sin z ^ 3 * Real.cos z = 3/2 → y ≤ z) ∧
  6 * Real.sin y * Real.cos y ^ 3 - 6 * Real.sin y ^ 3 * Real.cos y = 3/2 ∧
  y = 7.5 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_for_trig_equation_l1574_157484


namespace NUMINAMATH_CALUDE_f_not_in_quadrant_II_l1574_157482

-- Define the linear function
def f (x : ℝ) : ℝ := 3 * x - 4

-- Define Quadrant II
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem: The function f does not pass through Quadrant II
theorem f_not_in_quadrant_II :
  ∀ x : ℝ, ¬(in_quadrant_II x (f x)) :=
by
  sorry

end NUMINAMATH_CALUDE_f_not_in_quadrant_II_l1574_157482


namespace NUMINAMATH_CALUDE_g_zero_values_l1574_157426

theorem g_zero_values (g : ℝ → ℝ) (h : ∀ x : ℝ, g (2 * x) = g x ^ 2) :
  g 0 = 0 ∨ g 0 = 1 := by
sorry

end NUMINAMATH_CALUDE_g_zero_values_l1574_157426


namespace NUMINAMATH_CALUDE_test_probabilities_l1574_157425

def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.6
def prob_C : ℝ := 0.5

theorem test_probabilities :
  let prob_all := prob_A * prob_B * prob_C
  let prob_none := (1 - prob_A) * (1 - prob_B) * (1 - prob_C)
  let prob_at_least_one := 1 - prob_none
  prob_all = 0.24 ∧ prob_at_least_one = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_test_probabilities_l1574_157425


namespace NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1574_157450

theorem geometric_progression_fourth_term 
  (a : ℝ) (r : ℝ) 
  (h1 : a = 4^(1/2 : ℝ)) 
  (h2 : a * r = 4^(1/3 : ℝ)) 
  (h3 : a * r^2 = 4^(1/6 : ℝ)) : 
  a * r^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_fourth_term_l1574_157450


namespace NUMINAMATH_CALUDE_min_sum_arithmetic_sequence_l1574_157405

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem min_sum_arithmetic_sequence :
  let a₁ : ℤ := -28
  let d : ℤ := 4
  ∀ k : ℕ, k ≥ 1 →
    (sum_arithmetic_sequence a₁ d k ≥ sum_arithmetic_sequence a₁ d 7 ∧
     sum_arithmetic_sequence a₁ d k ≥ sum_arithmetic_sequence a₁ d 8) ∧
    (sum_arithmetic_sequence a₁ d 7 = sum_arithmetic_sequence a₁ d 8) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_arithmetic_sequence_l1574_157405


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1574_157446

theorem absolute_value_inequality (x : ℝ) : 
  |x^2 - 5*x + 6| < x^2 - 4 ↔ x > 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1574_157446


namespace NUMINAMATH_CALUDE_base6_greater_than_base8_l1574_157416

/-- Convert a base-6 number to base-10 --/
def base6_to_decimal (n : ℕ) : ℕ :=
  (n % 10) + 6 * ((n / 10) % 10) + 36 * (n / 100)

/-- Convert a base-8 number to base-10 --/
def base8_to_decimal (n : ℕ) : ℕ :=
  (n % 10) + 8 * ((n / 10) % 10) + 64 * (n / 100)

theorem base6_greater_than_base8 : base6_to_decimal 403 > base8_to_decimal 217 := by
  sorry

end NUMINAMATH_CALUDE_base6_greater_than_base8_l1574_157416


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l1574_157497

def U : Set Nat := {2, 3, 4, 5, 6}
def A : Set Nat := {2, 5, 6}
def B : Set Nat := {3, 5}

theorem complement_intersection_problem : (U \ A) ∩ B = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l1574_157497


namespace NUMINAMATH_CALUDE_f_positive_iff_f_plus_abs_lower_bound_f_plus_abs_equality_exists_l1574_157400

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for part (1)
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x > 1 ∨ x < -5 := by sorry

-- Theorem for part (2)
theorem f_plus_abs_lower_bound (x : ℝ) : f x + 3*|x - 4| ≥ 9 := by sorry

-- Theorem for the existence of equality in part (2)
theorem f_plus_abs_equality_exists : ∃ x : ℝ, f x + 3*|x - 4| = 9 := by sorry

end NUMINAMATH_CALUDE_f_positive_iff_f_plus_abs_lower_bound_f_plus_abs_equality_exists_l1574_157400


namespace NUMINAMATH_CALUDE_line_slope_is_two_l1574_157495

/-- Given a line with y-intercept 2 and passing through the point (269, 540),
    prove that its slope is 2. -/
theorem line_slope_is_two (line : Set (ℝ × ℝ)) 
    (y_intercept : (0, 2) ∈ line)
    (point_on_line : (269, 540) ∈ line) :
    let slope := (540 - 2) / (269 - 0)
    slope = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l1574_157495


namespace NUMINAMATH_CALUDE_binomial_coefficient_17_8_l1574_157499

theorem binomial_coefficient_17_8 :
  (Nat.choose 15 6 = 5005) →
  (Nat.choose 15 7 = 6435) →
  (Nat.choose 15 8 = 6435) →
  Nat.choose 17 8 = 24310 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_17_8_l1574_157499


namespace NUMINAMATH_CALUDE_sector_area_l1574_157418

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 8) (h2 : central_angle = 2) :
  let radius := (perimeter - central_angle * (perimeter / (2 + central_angle))) / 2
  let arc_length := central_angle * radius
  (1 / 2) * radius * arc_length = 4 := by sorry

end NUMINAMATH_CALUDE_sector_area_l1574_157418


namespace NUMINAMATH_CALUDE_previous_weekend_earnings_l1574_157463

-- Define the given amounts
def saturday_earnings : ℕ := 18
def sunday_earnings : ℕ := saturday_earnings / 2
def pogo_stick_cost : ℕ := 60
def additional_needed : ℕ := 13

-- Define the total earnings for this weekend
def this_weekend_earnings : ℕ := saturday_earnings + sunday_earnings

-- Define the theorem
theorem previous_weekend_earnings :
  pogo_stick_cost - additional_needed - this_weekend_earnings = 20 := by
  sorry

end NUMINAMATH_CALUDE_previous_weekend_earnings_l1574_157463


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1574_157458

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 ≠ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- State the theorem
theorem possible_values_of_a (a : ℝ) : 
  (B a ⊆ A) ↔ (a = 0 ∨ a = 1 ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1574_157458


namespace NUMINAMATH_CALUDE_sarahs_book_pages_l1574_157423

/-- Calculates the number of pages in each book given Sarah's reading parameters --/
theorem sarahs_book_pages
  (reading_speed : ℕ)  -- words per minute
  (reading_time : ℕ)   -- hours
  (num_books : ℕ)      -- number of books
  (words_per_page : ℕ) -- words per page
  (h1 : reading_speed = 40)
  (h2 : reading_time = 20)
  (h3 : num_books = 6)
  (h4 : words_per_page = 100)
  : (reading_speed * reading_time * 60) / (num_books * words_per_page) = 80 :=
by sorry

end NUMINAMATH_CALUDE_sarahs_book_pages_l1574_157423


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1574_157443

theorem quadratic_rewrite (k : ℝ) :
  let f := fun k : ℝ => 8 * k^2 - 6 * k + 16
  ∃ c r s : ℝ, (∀ k, f k = c * (k + r)^2 + s) ∧ s / r = -119 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1574_157443


namespace NUMINAMATH_CALUDE_part_one_part_two_l1574_157421

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := |2 * x + b|

-- Part I
theorem part_one (b : ℝ) : 
  (∀ x, f b x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 2) → b = -1 := by sorry

-- Part II
theorem part_two :
  ∃ m : ℝ, ∀ x : ℝ, f (-1) (x + 3) + f (-1) (x + 1) ≥ m ∧ 
  ¬∃ m' : ℝ, (m' < m ∧ ∀ x : ℝ, f (-1) (x + 3) + f (-1) (x + 1) ≥ m') := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1574_157421


namespace NUMINAMATH_CALUDE_smallest_base_for_82_five_satisfies_condition_five_is_smallest_base_l1574_157456

theorem smallest_base_for_82 : 
  ∀ b : ℕ, b > 0 → (b^2 ≤ 82 ∧ 82 < b^3) → b ≥ 5 :=
by
  sorry

theorem five_satisfies_condition : 
  5^2 ≤ 82 ∧ 82 < 5^3 :=
by
  sorry

theorem five_is_smallest_base : 
  ∀ b : ℕ, b > 0 → b^2 ≤ 82 ∧ 82 < b^3 → b = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_82_five_satisfies_condition_five_is_smallest_base_l1574_157456


namespace NUMINAMATH_CALUDE_route_down_is_twelve_miles_l1574_157424

/-- Represents the hiking trip up and down a mountain -/
structure MountainHike where
  rate_up : ℝ
  time_up : ℝ
  rate_down_factor : ℝ

/-- The length of the route down the mountain -/
def route_down_length (hike : MountainHike) : ℝ :=
  hike.rate_up * hike.rate_down_factor * hike.time_up

/-- Theorem stating that the length of the route down is 12 miles -/
theorem route_down_is_twelve_miles (hike : MountainHike) 
  (h1 : hike.rate_up = 4)
  (h2 : hike.time_up = 2)
  (h3 : hike.rate_down_factor = 1.5) : 
  route_down_length hike = 12 := by
  sorry

#eval route_down_length ⟨4, 2, 1.5⟩

end NUMINAMATH_CALUDE_route_down_is_twelve_miles_l1574_157424


namespace NUMINAMATH_CALUDE_tissue_count_after_use_l1574_157493

def initial_tissue_count : ℕ := 97
def used_tissue_count : ℕ := 4

theorem tissue_count_after_use :
  initial_tissue_count - used_tissue_count = 93 := by
  sorry

end NUMINAMATH_CALUDE_tissue_count_after_use_l1574_157493


namespace NUMINAMATH_CALUDE_c_equals_zero_l1574_157471

theorem c_equals_zero (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a*b + b - 9) : c = 0 := by
  sorry

end NUMINAMATH_CALUDE_c_equals_zero_l1574_157471


namespace NUMINAMATH_CALUDE_integral_x_minus_one_l1574_157451

theorem integral_x_minus_one : ∫ x in (0 : ℝ)..2, (x - 1) = 0 := by sorry

end NUMINAMATH_CALUDE_integral_x_minus_one_l1574_157451


namespace NUMINAMATH_CALUDE_expand_product_l1574_157480

theorem expand_product (x : ℝ) (h : x ≠ 0) :
  2/5 * (5/x + 10*x^2) = 2/x + 4*x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1574_157480


namespace NUMINAMATH_CALUDE_largest_number_on_board_l1574_157492

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_four (n : ℕ) : Prop := n % 10 = 4

def set_of_interest : Set ℕ := {n | is_two_digit n ∧ n % 6 = 0 ∧ ends_in_four n}

theorem largest_number_on_board : 
  ∃ (m : ℕ), m ∈ set_of_interest ∧ ∀ (n : ℕ), n ∈ set_of_interest → n ≤ m ∧ m = 84 :=
sorry

end NUMINAMATH_CALUDE_largest_number_on_board_l1574_157492


namespace NUMINAMATH_CALUDE_greatest_7_power_divisor_l1574_157414

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- n is divisible by 7^k -/
def divides_by_7_pow (n : ℕ+) (k : ℕ) : Prop := sorry

theorem greatest_7_power_divisor (n : ℕ+) (h1 : num_divisors n = 30) (h2 : num_divisors (7 * n) = 42) :
  ∃ k : ℕ, divides_by_7_pow n k ∧ k = 1 ∧ ∀ m : ℕ, divides_by_7_pow n m → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_greatest_7_power_divisor_l1574_157414


namespace NUMINAMATH_CALUDE_encryption_correct_l1574_157475

def encrypt_digit (d : Nat) : Nat :=
  (d^3 + 1) % 10

def encrypt_number (n : List Nat) : List Nat :=
  n.map encrypt_digit

theorem encryption_correct : 
  encrypt_number [2, 5, 6, 8] = [9, 6, 7, 3] := by sorry

end NUMINAMATH_CALUDE_encryption_correct_l1574_157475


namespace NUMINAMATH_CALUDE_amaya_total_marks_l1574_157434

def total_marks (music maths arts social_studies : ℕ) : ℕ :=
  music + maths + arts + social_studies

theorem amaya_total_marks :
  ∀ (music maths arts social_studies : ℕ),
    music = 70 →
    maths = music - music / 10 →
    arts = maths + 20 →
    social_studies = music + 10 →
    total_marks music maths arts social_studies = 296 :=
by
  sorry

end NUMINAMATH_CALUDE_amaya_total_marks_l1574_157434


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1574_157479

/-- Given a line segment CD with midpoint M(5,4) and one endpoint C(7,-2),
    the sum of the coordinates of the other endpoint D is 13. -/
theorem midpoint_coordinate_sum :
  ∀ (D : ℝ × ℝ),
  (5, 4) = ((7, -2) + D) / 2 →
  D.1 + D.2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1574_157479


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l1574_157410

theorem complex_sum_magnitude (a b c : ℂ) :
  Complex.abs a = 1 →
  Complex.abs b = 1 →
  Complex.abs c = 1 →
  a^3 / (b^2 * c) + b^3 / (a^2 * c) + c^3 / (a^2 * b) = 1 →
  Complex.abs (a + b + c) = 1 ∨ Complex.abs (a + b + c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l1574_157410


namespace NUMINAMATH_CALUDE_triangle_side_sum_l1574_157419

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  b * Real.cos C + c * Real.cos B = 3 * a * Real.cos B →
  b = 2 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 2) / 2 →
  a + c = 4 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_sum_l1574_157419


namespace NUMINAMATH_CALUDE_lines_concurrent_iff_det_zero_l1574_157433

/-- Three lines pass through the same point if and only if the determinant of their coefficients is zero -/
theorem lines_concurrent_iff_det_zero 
  (A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ : ℝ) : 
  (∃ (x y : ℝ), A₁*x + B₁*y + C₁ = 0 ∧ A₂*x + B₂*y + C₂ = 0 ∧ A₃*x + B₃*y + C₃ = 0) ↔ 
  Matrix.det !![A₁, B₁, C₁; A₂, B₂, C₂; A₃, B₃, C₃] = 0 :=
by sorry

end NUMINAMATH_CALUDE_lines_concurrent_iff_det_zero_l1574_157433


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_one_l1574_157459

theorem sum_of_x_and_y_is_one (x y : ℝ) 
  (eq1 : 2021 * x + 2025 * y = 2029)
  (eq2 : 2023 * x + 2027 * y = 2031) : 
  x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_one_l1574_157459


namespace NUMINAMATH_CALUDE_distinct_roots_sum_squares_l1574_157445

theorem distinct_roots_sum_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  x₁^2 + 2*x₁ - k = 0 → 
  x₂^2 + 2*x₂ - k = 0 → 
  x₁^2 + x₂^2 - 2 > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_distinct_roots_sum_squares_l1574_157445


namespace NUMINAMATH_CALUDE_may_savings_l1574_157448

def savings (month : Nat) : Nat :=
  match month with
  | 0 => 10  -- January (0-indexed)
  | n + 1 => 2 * savings n

theorem may_savings : savings 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_may_savings_l1574_157448


namespace NUMINAMATH_CALUDE_f_composition_three_roots_l1574_157460

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

-- State the theorem
theorem f_composition_three_roots (c : ℝ) :
  (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    (∀ x : ℝ, f c (f c x) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃)) ↔
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end NUMINAMATH_CALUDE_f_composition_three_roots_l1574_157460


namespace NUMINAMATH_CALUDE_namjoon_candies_l1574_157488

/-- The number of candies Namjoon gave to Yoongi -/
def candies_given : ℕ := 18

/-- The number of candies left over -/
def candies_left : ℕ := 16

/-- The total number of candies Namjoon had in the beginning -/
def total_candies : ℕ := candies_given + candies_left

theorem namjoon_candies : total_candies = 34 := by
  sorry

end NUMINAMATH_CALUDE_namjoon_candies_l1574_157488


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1574_157406

-- Define the complex number -1-2i
def z : ℂ := -1 - 2 * Complex.I

-- Theorem stating that the imaginary part of z is -2
theorem imaginary_part_of_z :
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1574_157406


namespace NUMINAMATH_CALUDE_negative_one_power_equality_l1574_157438

theorem negative_one_power_equality : (-1 : ℤ)^3 = (-1 : ℤ)^2023 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_power_equality_l1574_157438


namespace NUMINAMATH_CALUDE_lateral_side_is_18_l1574_157411

/-- An isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithIncircle where
  /-- The length of the base of the isosceles triangle -/
  base : ℝ
  /-- The length of the lateral side of the isosceles triangle -/
  lateralSide : ℝ
  /-- The sum of the perimeters of the three smaller triangles formed by the tangents -/
  smallTrianglesPerimeterSum : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : lateralSide ≥ base / 2
  /-- The base is positive -/
  basePositive : base > 0
  /-- The lateral side is positive -/
  lateralSidePositive : lateralSide > 0
  /-- The sum of perimeters is positive -/
  sumPositive : smallTrianglesPerimeterSum > 0

/-- Theorem: If the base is 12 and the sum of perimeters is 48, then the lateral side is 18 -/
theorem lateral_side_is_18 (t : IsoscelesTriangleWithIncircle) 
  (h1 : t.base = 12) 
  (h2 : t.smallTrianglesPerimeterSum = 48) : 
  t.lateralSide = 18 := by
  sorry


end NUMINAMATH_CALUDE_lateral_side_is_18_l1574_157411


namespace NUMINAMATH_CALUDE_bullet_problem_l1574_157496

theorem bullet_problem (n : ℕ) (h1 : n > 4) :
  (5 * (n - 4) = n) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_bullet_problem_l1574_157496


namespace NUMINAMATH_CALUDE_joyce_basketball_shots_l1574_157481

theorem joyce_basketball_shots (initial_shots initial_made next_shots : ℕ) 
  (initial_average new_average : ℚ) : 
  initial_shots = 40 →
  initial_made = 15 →
  next_shots = 15 →
  initial_average = 375/1000 →
  new_average = 45/100 →
  ∃ (next_made : ℕ), 
    next_made = 10 ∧ 
    (initial_made + next_made : ℚ) / (initial_shots + next_shots) = new_average :=
by sorry

end NUMINAMATH_CALUDE_joyce_basketball_shots_l1574_157481


namespace NUMINAMATH_CALUDE_cube_volume_percentage_l1574_157467

def box_length : ℕ := 8
def box_width : ℕ := 6
def box_height : ℕ := 12
def cube_side : ℕ := 4

def cubes_per_length : ℕ := box_length / cube_side
def cubes_per_width : ℕ := box_width / cube_side
def cubes_per_height : ℕ := box_height / cube_side

def total_cubes : ℕ := cubes_per_length * cubes_per_width * cubes_per_height

def cube_volume : ℕ := cube_side ^ 3
def total_cube_volume : ℕ := total_cubes * cube_volume

def box_volume : ℕ := box_length * box_width * box_height

theorem cube_volume_percentage :
  (total_cube_volume : ℚ) / (box_volume : ℚ) * 100 = 200 / 3 := by sorry

end NUMINAMATH_CALUDE_cube_volume_percentage_l1574_157467


namespace NUMINAMATH_CALUDE_mini_van_tank_capacity_l1574_157441

/-- Represents the problem of determining the capacity of a mini-van's tank. -/
theorem mini_van_tank_capacity 
  (service_cost : ℝ) 
  (fuel_cost : ℝ) 
  (num_mini_vans : ℕ) 
  (num_trucks : ℕ) 
  (total_cost : ℝ) 
  (truck_tank_ratio : ℝ) 
  (h1 : service_cost = 2.30)
  (h2 : fuel_cost = 0.70)
  (h3 : num_mini_vans = 4)
  (h4 : num_trucks = 2)
  (h5 : total_cost = 396)
  (h6 : truck_tank_ratio = 2.20) : 
  ∃ (mini_van_capacity : ℝ),
    mini_van_capacity = 65 ∧
    total_cost = 
      (num_mini_vans + num_trucks) * service_cost + 
      (num_mini_vans * mini_van_capacity + num_trucks * (truck_tank_ratio * mini_van_capacity)) * fuel_cost :=
by sorry

end NUMINAMATH_CALUDE_mini_van_tank_capacity_l1574_157441


namespace NUMINAMATH_CALUDE_bcd4_hex_to_dec_l1574_157472

def hex_to_dec (digit : Char) : ℕ :=
  match digit with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | d   => d.toString.toNat!

def hex_string_to_dec (s : String) : ℕ :=
  s.foldr (fun d acc => hex_to_dec d + 16 * acc) 0

theorem bcd4_hex_to_dec :
  hex_string_to_dec "BCD4" = 31444 := by
  sorry

end NUMINAMATH_CALUDE_bcd4_hex_to_dec_l1574_157472
