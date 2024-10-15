import Mathlib

namespace NUMINAMATH_CALUDE_max_value_on_interval_l160_16095

-- Define the function
def f (x : ℝ) : ℝ := x^4 - 8*x^2 + 2

-- State the theorem
theorem max_value_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1) 3 ∧ 
  (∀ x, x ∈ Set.Icc (-1) 3 → f x ≤ f c) ∧
  f c = 11 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l160_16095


namespace NUMINAMATH_CALUDE_tangent_function_property_l160_16026

theorem tangent_function_property (c d : ℝ) (h1 : c > 0) (h2 : d > 0) : 
  (∀ x, c * Real.tan (d * x) = c * Real.tan (d * (x + 3 * π / 4))) →
  c * Real.tan (d * π / 8) = 3 →
  c * d = 4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_function_property_l160_16026


namespace NUMINAMATH_CALUDE_expression_factorization_l160_16018

theorem expression_factorization (x : ℝ) :
  (10 * x^3 + 50 * x^2 - 4) - (3 * x^3 - 5 * x^2 + 2) = 7 * x^3 + 55 * x^2 - 6 :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l160_16018


namespace NUMINAMATH_CALUDE_train_platform_length_equality_l160_16000

/-- Proves that the length of a train equals the length of a platform given specific conditions --/
theorem train_platform_length_equality (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_min : ℝ) :
  train_length = 750 →
  train_speed_kmh = 90 →
  crossing_time_min = 1 →
  ∃ (platform_length : ℝ),
    platform_length = train_length ∧
    platform_length + train_length = train_speed_kmh * (1000 / 3600) * (crossing_time_min * 60) :=
by sorry


end NUMINAMATH_CALUDE_train_platform_length_equality_l160_16000


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_problem_solution_l160_16052

theorem fraction_of_fraction_of_fraction (n : ℕ) : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * n = n / 36 := by
  sorry

theorem problem_solution : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 180 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_problem_solution_l160_16052


namespace NUMINAMATH_CALUDE_geometric_mean_sqrt2_plus_minus_one_l160_16064

theorem geometric_mean_sqrt2_plus_minus_one : 
  ∃ x : ℝ, x^2 = (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) ∧ (x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_sqrt2_plus_minus_one_l160_16064


namespace NUMINAMATH_CALUDE_parabola_equation_dot_product_focus_fixed_point_l160_16074

-- Define the parabola
def Parabola := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the dot product of two points
def dot_product (p q : ℝ × ℝ) : ℝ := p.1 * q.1 + p.2 * q.2

-- Theorem 1: Standard equation of the parabola
theorem parabola_equation (p : ℝ × ℝ) : p ∈ Parabola ↔ p.2^2 = 4 * p.1 := by sorry

-- Theorem 2: Dot product of OA and OB when line passes through focus
theorem dot_product_focus (A B : ℝ × ℝ) (hA : A ∈ Parabola) (hB : B ∈ Parabola) 
  (h_line : ∃ (m : ℝ), A.2 = m * (A.1 - 1) ∧ B.2 = m * (B.1 - 1) ∧ focus.2 = m * (focus.1 - 1)) :
  dot_product A B = -3 := by sorry

-- Theorem 3: Fixed point when dot product is -4
theorem fixed_point (A B : ℝ × ℝ) (hA : A ∈ Parabola) (hB : B ∈ Parabola) 
  (h_dot : dot_product A B = -4) :
  ∃ (m : ℝ), A.2 = m * (A.1 - 2) ∧ B.2 = m * (B.1 - 2) := by sorry

end NUMINAMATH_CALUDE_parabola_equation_dot_product_focus_fixed_point_l160_16074


namespace NUMINAMATH_CALUDE_biography_percentage_before_purchase_l160_16035

theorem biography_percentage_before_purchase
  (increase_rate : ℝ)
  (final_percentage : ℝ)
  (h_increase : increase_rate = 0.8823529411764707)
  (h_final : final_percentage = 0.32)
  : ∃ (initial_percentage : ℝ),
    initial_percentage * (1 + increase_rate) = final_percentage ∧
    initial_percentage = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_biography_percentage_before_purchase_l160_16035


namespace NUMINAMATH_CALUDE_product_range_check_l160_16077

theorem product_range_check : 
  (1200 < 31 * 53 ∧ 31 * 53 < 2400) ∧ 
  (32 * 84 > 2400) ∧ 
  (63 * 54 > 2400) ∧ 
  (1200 < 72 * 24 ∧ 72 * 24 < 2400) := by
  sorry

end NUMINAMATH_CALUDE_product_range_check_l160_16077


namespace NUMINAMATH_CALUDE_exist_three_quadratic_polynomials_l160_16034

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial := ℝ → ℝ

/-- A quadratic polynomial has two distinct real roots -/
def HasTwoDistinctRealRoots (p : QuadraticPolynomial) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ p r₁ = 0 ∧ p r₂ = 0

/-- A quadratic polynomial has no real roots -/
def HasNoRealRoots (p : QuadraticPolynomial) : Prop :=
  ∀ x : ℝ, p x ≠ 0

/-- Theorem: There exist three quadratic polynomials satisfying the given conditions -/
theorem exist_three_quadratic_polynomials :
  ∃ (P₁ P₂ P₃ : QuadraticPolynomial),
    HasTwoDistinctRealRoots P₁ ∧
    HasTwoDistinctRealRoots P₂ ∧
    HasTwoDistinctRealRoots P₃ ∧
    HasNoRealRoots (λ x => P₁ x + P₂ x) ∧
    HasNoRealRoots (λ x => P₁ x + P₃ x) ∧
    HasNoRealRoots (λ x => P₂ x + P₃ x) := by
  sorry

end NUMINAMATH_CALUDE_exist_three_quadratic_polynomials_l160_16034


namespace NUMINAMATH_CALUDE_triangle_radii_relations_l160_16093

/-- Triangle properties -/
class Triangle (α : Type*) [LinearOrderedField α] :=
  (a b c : α)
  (t : α)
  (s : α)
  (ρ : α)
  (ρa ρb ρc : α)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b)
  (semiperimeter : s = (a + b + c) / 2)
  (area_positive : 0 < t)

/-- Theorem about relationships between inradius, exradii, semiperimeter, and area of a triangle -/
theorem triangle_radii_relations {α : Type*} [LinearOrderedField α] (T : Triangle α) :
  T.ρa * T.ρb + T.ρb * T.ρc + T.ρc * T.ρa = T.s^2 ∧
  1 / T.ρ = 1 / T.ρa + 1 / T.ρb + 1 / T.ρc ∧
  T.ρ * T.ρa * T.ρb * T.ρc = T.t^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_radii_relations_l160_16093


namespace NUMINAMATH_CALUDE_f_3_equals_11_l160_16019

-- Define the function f
def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 + b * x + 2

-- State the theorem
theorem f_3_equals_11 (a b : ℝ) :
  f 1 a b = 5 →
  f 2 a b = 8 →
  f 3 a b = 11 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_11_l160_16019


namespace NUMINAMATH_CALUDE_centroid_positions_count_l160_16002

/-- A point on the perimeter of the square -/
structure PerimeterPoint where
  x : Fin 21
  y : Fin 21
  on_perimeter : (x = 0 ∨ x = 20) ∨ (y = 0 ∨ y = 20)

/-- The centroid of a triangle -/
def centroid (p q r : PerimeterPoint) : ℚ × ℚ :=
  ((p.x + q.x + r.x : ℚ) / 3, (p.y + q.y + r.y : ℚ) / 3)

/-- Predicate for valid centroid positions -/
def is_valid_centroid (c : ℚ × ℚ) : Prop :=
  0 < c.1 ∧ c.1 < 20 ∧ 0 < c.2 ∧ c.2 < 20

/-- The main theorem -/
theorem centroid_positions_count :
  ∃ (valid_centroids : Finset (ℚ × ℚ)),
    (∀ c ∈ valid_centroids, is_valid_centroid c) ∧
    (∀ p q r : PerimeterPoint, p ≠ q ∧ q ≠ r ∧ p ≠ r →
      centroid p q r ∈ valid_centroids) ∧
    valid_centroids.card = 3481 :=
  sorry

end NUMINAMATH_CALUDE_centroid_positions_count_l160_16002


namespace NUMINAMATH_CALUDE_divisibility_implication_l160_16024

theorem divisibility_implication (a b : ℕ+) :
  (∀ n : ℕ, a^n ∣ b^(n+1)) → a ∣ b := by sorry

end NUMINAMATH_CALUDE_divisibility_implication_l160_16024


namespace NUMINAMATH_CALUDE_horseshoe_profit_is_22000_l160_16015

/-- Represents the profit calculation for Redo's Horseshoe Company --/
def horseshoe_profit : ℝ :=
  let type_a_initial_outlay : ℝ := 10000
  let type_a_cost_per_set : ℝ := 20
  let type_a_price_high : ℝ := 60
  let type_a_price_low : ℝ := 50
  let type_a_sets_high : ℝ := 300
  let type_a_sets_low : ℝ := 200
  let type_b_initial_outlay : ℝ := 6000
  let type_b_cost_per_set : ℝ := 15
  let type_b_price : ℝ := 40
  let type_b_sets : ℝ := 800

  let type_a_revenue := type_a_price_high * type_a_sets_high + type_a_price_low * type_a_sets_low
  let type_a_cost := type_a_initial_outlay + type_a_cost_per_set * (type_a_sets_high + type_a_sets_low)
  let type_a_profit := type_a_revenue - type_a_cost

  let type_b_revenue := type_b_price * type_b_sets
  let type_b_cost := type_b_initial_outlay + type_b_cost_per_set * type_b_sets
  let type_b_profit := type_b_revenue - type_b_cost

  type_a_profit + type_b_profit

/-- The total profit for Redo's Horseshoe Company is $22,000 --/
theorem horseshoe_profit_is_22000 : horseshoe_profit = 22000 := by
  sorry

end NUMINAMATH_CALUDE_horseshoe_profit_is_22000_l160_16015


namespace NUMINAMATH_CALUDE_frog_food_theorem_l160_16056

/-- The number of flies a frog eats per day -/
def flies_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of flies Betty has caught -/
def flies_caught : ℕ := 10

/-- The number of additional flies Betty needs for a week's food -/
def additional_flies_needed : ℕ := 4

theorem frog_food_theorem :
  flies_per_day * days_in_week - flies_caught = additional_flies_needed :=
by sorry

end NUMINAMATH_CALUDE_frog_food_theorem_l160_16056


namespace NUMINAMATH_CALUDE_regular_icosahedron_has_12_vertices_l160_16070

/-- A regular icosahedron is a polyhedron with equilateral triangles as faces -/
structure RegularIcosahedron where
  /-- The number of faces in the icosahedron -/
  faces : ℕ
  /-- The number of vertices in the icosahedron -/
  vertices : ℕ
  /-- The number of edges in the icosahedron -/
  edges : ℕ
  /-- All faces are equilateral triangles -/
  all_faces_equilateral : True
  /-- Euler's formula for polyhedra: V - E + F = 2 -/
  euler_formula : vertices - edges + faces = 2
  /-- Each face is a triangle, so 3F = 2E -/
  face_edge_relation : 3 * faces = 2 * edges
  /-- Each vertex has degree 5 in an icosahedron -/
  vertex_degree_five : 5 * vertices = 2 * edges

/-- Theorem: A regular icosahedron has 12 vertices -/
theorem regular_icosahedron_has_12_vertices (i : RegularIcosahedron) : i.vertices = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_icosahedron_has_12_vertices_l160_16070


namespace NUMINAMATH_CALUDE_expected_cereal_difference_l160_16047

/-- Represents the outcome of rolling a fair six-sided die -/
inductive DieRoll
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the type of cereal Bob eats based on his die roll -/
inductive CerealType
  | sweetened
  | unsweetened
  | healthy

/-- Maps a die roll to the corresponding cereal type -/
def rollToCereal (roll : DieRoll) : CerealType :=
  match roll with
  | DieRoll.one => CerealType.healthy
  | DieRoll.two => CerealType.unsweetened
  | DieRoll.three => CerealType.unsweetened
  | DieRoll.four => CerealType.sweetened
  | DieRoll.five => CerealType.unsweetened
  | DieRoll.six => CerealType.sweetened

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- The probability of rolling any specific number on a fair six-sided die -/
def probSingle : ℚ := 1 / 6

/-- Theorem stating the expected difference between unsweetened and sweetened cereal days -/
theorem expected_cereal_difference :
  ∃ (diff : ℚ), abs (diff - 60.83) < 0.01 ∧
  diff = daysInYear * (3 * probSingle - 2 * probSingle) := by
  sorry

end NUMINAMATH_CALUDE_expected_cereal_difference_l160_16047


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l160_16006

theorem binomial_coefficient_divisibility (n : ℕ) : (n + 1) ∣ Nat.choose (2 * n) n := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l160_16006


namespace NUMINAMATH_CALUDE_sum_of_d_and_f_is_zero_l160_16079

/-- Given three complex numbers a + bi, c + di, and 3e + fi, prove that d + f = 0 
    under the following conditions:
    1) b = 2
    2) c = -a - 2e
    3) The sum of the three complex numbers is 2i
-/
theorem sum_of_d_and_f_is_zero 
  (a b c d e f : ℂ) 
  (h1 : b = 2)
  (h2 : c = -a - 2*e)
  (h3 : a + b*Complex.I + c + d*Complex.I + 3*e + f*Complex.I = 2*Complex.I) :
  d + f = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_d_and_f_is_zero_l160_16079


namespace NUMINAMATH_CALUDE_bus_seat_capacity_l160_16051

/-- Represents the capacity of a bus with specific seating arrangements. -/
structure BusCapacity where
  left_seats : ℕ
  right_seats : ℕ
  back_seat_capacity : ℕ
  total_capacity : ℕ

/-- Calculates the number of people each regular seat can hold. -/
def seats_capacity (bus : BusCapacity) : ℚ :=
  (bus.total_capacity - bus.back_seat_capacity) / (bus.left_seats + bus.right_seats)

/-- Theorem stating that for the given bus configuration, each seat can hold 3 people. -/
theorem bus_seat_capacity :
  let bus := BusCapacity.mk 15 12 9 90
  seats_capacity bus = 3 := by sorry

end NUMINAMATH_CALUDE_bus_seat_capacity_l160_16051


namespace NUMINAMATH_CALUDE_solve_equation_for_x_l160_16007

theorem solve_equation_for_x (x y : ℤ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : x + y + x * y = 101) : 
  x = 50 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_for_x_l160_16007


namespace NUMINAMATH_CALUDE_tank_capacity_l160_16065

/-- Represents a cylindrical water tank -/
structure WaterTank where
  capacity : ℝ
  currentVolume : ℝ
  fillPercentage : ℝ

/-- The tank contains 120 liters when it is 24% full -/
def partiallyFilledTank : WaterTank :=
  { capacity := 500,
    currentVolume := 120,
    fillPercentage := 0.24 }

/-- Theorem stating that the tank's capacity is 500 liters -/
theorem tank_capacity :
  partiallyFilledTank.capacity = 500 ∧
  partiallyFilledTank.currentVolume = 120 ∧
  partiallyFilledTank.fillPercentage = 0.24 ∧
  partiallyFilledTank.currentVolume = partiallyFilledTank.capacity * partiallyFilledTank.fillPercentage :=
by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l160_16065


namespace NUMINAMATH_CALUDE_min_trips_to_fill_tank_l160_16080

/-- The minimum number of trips required to fill a cylindrical tank using a hemispherical bucket -/
theorem min_trips_to_fill_tank (tank_radius tank_height bucket_radius : ℝ) 
  (hr : tank_radius = 8) 
  (hh : tank_height = 20) 
  (hb : bucket_radius = 6) : 
  ∃ n : ℕ, (n : ℝ) * ((2/3) * Real.pi * bucket_radius^3) ≥ Real.pi * tank_radius^2 * tank_height ∧ 
  ∀ m : ℕ, m < n → (m : ℝ) * ((2/3) * Real.pi * bucket_radius^3) < Real.pi * tank_radius^2 * tank_height :=
by
  sorry

end NUMINAMATH_CALUDE_min_trips_to_fill_tank_l160_16080


namespace NUMINAMATH_CALUDE_camp_wonka_ratio_l160_16014

theorem camp_wonka_ratio : 
  ∀ (total_campers : ℕ) (boys : ℕ),
    total_campers = 96 →
    boys = (2 * total_campers) / 3 →
    (total_campers - boys) * 3 = total_campers :=
by
  sorry

end NUMINAMATH_CALUDE_camp_wonka_ratio_l160_16014


namespace NUMINAMATH_CALUDE_intersection_M_N_l160_16099

def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l160_16099


namespace NUMINAMATH_CALUDE_part_one_part_two_part_two_range_l160_16033

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 3| - |x - a|

-- Part I
theorem part_one :
  let a := 2
  {x : ℝ | f a x ≤ -1/2} = {x : ℝ | x ≥ 11/4} := by sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∃ x : ℝ, f a x ≥ a) → a ∈ Set.Iic (3/2) := by sorry

-- Additional theorem to show the full range of a
theorem part_two_range :
  {a : ℝ | ∃ x : ℝ, f a x ≥ a} = Set.Iic (3/2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_two_range_l160_16033


namespace NUMINAMATH_CALUDE_square_root_equal_self_l160_16066

theorem square_root_equal_self : ∀ x : ℝ, x = Real.sqrt x ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_square_root_equal_self_l160_16066


namespace NUMINAMATH_CALUDE_pulley_centers_distance_l160_16057

theorem pulley_centers_distance 
  (r1 r2 contact_distance : ℝ) 
  (h1 : r1 = 18)
  (h2 : r2 = 16)
  (h3 : contact_distance = 40) :
  let center_distance := Real.sqrt (contact_distance^2 + (r1 - r2)^2)
  center_distance = Real.sqrt 1604 := by
  sorry

end NUMINAMATH_CALUDE_pulley_centers_distance_l160_16057


namespace NUMINAMATH_CALUDE_quadratic_function_statements_l160_16048

/-- A quadratic function y = ax^2 + bx + c where a ≠ 0 and x ∈ M (M is a non-empty set) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  M : Set ℝ
  a_nonzero : a ≠ 0
  M_nonempty : M.Nonempty

/-- Statement 1: When a > 0, the function always has a minimum value of (4ac - b^2) / (4a) -/
def statement1 (f : QuadraticFunction) : Prop :=
  f.a > 0 → ∃ (min : ℝ), ∀ (x : ℝ), x ∈ f.M → f.a * x^2 + f.b * x + f.c ≥ min

/-- Statement 2: The existence of max/min depends on the range of x, and both can exist with values not necessarily (4ac - b^2) / (4a) -/
def statement2 (f : QuadraticFunction) : Prop :=
  ∃ (M1 M2 : Set ℝ), M1.Nonempty ∧ M2.Nonempty ∧
    (∃ (max min : ℝ), (∀ (x : ℝ), x ∈ M1 → f.a * x^2 + f.b * x + f.c ≤ max) ∧
                      (∀ (x : ℝ), x ∈ M2 → f.a * x^2 + f.b * x + f.c ≥ min) ∧
                      (max ≠ (4 * f.a * f.c - f.b^2) / (4 * f.a) ∨
                       min ≠ (4 * f.a * f.c - f.b^2) / (4 * f.a)))

/-- Statement 3: The method to find max/min involves finding the axis of symmetry and analyzing the graph -/
def statement3 (f : QuadraticFunction) : Prop :=
  ∃ (x : ℝ), x = -f.b / (2 * f.a) ∧
    ∀ (y : ℝ), y ∈ f.M → (f.a * y^2 + f.b * y + f.c = f.a * x^2 + f.b * x + f.c ↔ y = x)

theorem quadratic_function_statements (f : QuadraticFunction) :
  (statement1 f ∧ ¬statement2 f ∧ ¬statement3 f) ∨
  (¬statement1 f ∧ statement2 f ∧ ¬statement3 f) ∨
  (¬statement1 f ∧ ¬statement2 f ∧ statement3 f) ∨
  (¬statement1 f ∧ statement2 f ∧ statement3 f) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_statements_l160_16048


namespace NUMINAMATH_CALUDE_santinos_fruits_l160_16091

/-- The number of papaya trees Santino has -/
def papaya_trees : ℕ := 2

/-- The number of mango trees Santino has -/
def mango_trees : ℕ := 3

/-- The number of papayas each papaya tree produces -/
def papayas_per_tree : ℕ := 10

/-- The number of mangos each mango tree produces -/
def mangos_per_tree : ℕ := 20

/-- The total number of fruits Santino has -/
def total_fruits : ℕ := papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree

theorem santinos_fruits : total_fruits = 80 := by
  sorry

end NUMINAMATH_CALUDE_santinos_fruits_l160_16091


namespace NUMINAMATH_CALUDE_linear_function_properties_l160_16032

/-- Linear function passing through (1, 0) and (0, 2) -/
def linear_function (x : ℝ) : ℝ := -2 * x + 2

theorem linear_function_properties :
  let f := linear_function
  -- The range of y is -4 ≤ y < 6 when -2 < x ≤ 3
  (∀ x : ℝ, -2 < x ∧ x ≤ 3 → -4 ≤ f x ∧ f x < 6) ∧
  -- The point P(m, n) satisfying m - n = 4 has coordinates (2, -2)
  (∃ m n : ℝ, f m = n ∧ m - n = 4 ∧ m = 2 ∧ n = -2) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l160_16032


namespace NUMINAMATH_CALUDE_endpoint_sum_l160_16023

/-- Given a line segment with one endpoint (1, 2) and midpoint (5, 6),
    the sum of coordinates of the other endpoint is 19. -/
theorem endpoint_sum (x y : ℝ) : 
  (1 + x) / 2 = 5 ∧ (2 + y) / 2 = 6 → x + y = 19 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_l160_16023


namespace NUMINAMATH_CALUDE_x_positive_iff_reciprocal_positive_l160_16055

theorem x_positive_iff_reciprocal_positive (x : ℝ) :
  x > 0 ↔ 1 / x > 0 := by
  sorry

end NUMINAMATH_CALUDE_x_positive_iff_reciprocal_positive_l160_16055


namespace NUMINAMATH_CALUDE_cousins_distribution_l160_16060

/-- The number of ways to distribute n indistinguishable objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 cousins into 4 rooms -/
theorem cousins_distribution : distribute 5 4 = 66 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_l160_16060


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l160_16017

noncomputable def z : ℂ := (Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I))

theorem imaginary_part_of_z : Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l160_16017


namespace NUMINAMATH_CALUDE_expression_equality_l160_16098

theorem expression_equality : 
  (753^2 + 247^2 - 753 * 247) / (753^3 + 247^3) = 1 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l160_16098


namespace NUMINAMATH_CALUDE_game_sale_problem_l160_16040

theorem game_sale_problem (initial_games : ℕ) (sold_games : ℕ) 
  (sold_at_15 : ℕ) (sold_at_10 : ℕ) (sold_at_8 : ℕ) (games_per_box : ℕ) :
  initial_games = 76 →
  sold_games = 46 →
  sold_at_15 = 20 →
  sold_at_10 = 15 →
  sold_at_8 = 11 →
  games_per_box = 5 →
  sold_games = sold_at_15 + sold_at_10 + sold_at_8 →
  (initial_games - sold_games) % games_per_box = 0 →
  ((initial_games - sold_games) / games_per_box = 6 ∧ 
   sold_at_15 * 15 + sold_at_10 * 10 + sold_at_8 * 8 = 538) := by
  sorry


end NUMINAMATH_CALUDE_game_sale_problem_l160_16040


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l160_16041

theorem negation_of_universal_proposition (f : ℕ+ → ℝ) :
  (¬ ∀ n : ℕ+, f n ≤ n) ↔ (∃ n : ℕ+, f n > n) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l160_16041


namespace NUMINAMATH_CALUDE_honeys_earnings_l160_16075

/-- Honey's earnings problem -/
theorem honeys_earnings (days : ℕ) (spent : ℕ) (saved : ℕ) (daily_earnings : ℕ) : 
  days = 20 → spent = 1360 → saved = 240 → daily_earnings = 80 → 
  days * daily_earnings = spent + saved :=
by
  sorry

#check honeys_earnings

end NUMINAMATH_CALUDE_honeys_earnings_l160_16075


namespace NUMINAMATH_CALUDE_christines_second_dog_weight_l160_16029

/-- The weight of Christine's second dog -/
def weight_second_dog (cat_weights : List ℕ) (additional_weight : ℕ) : ℕ :=
  let total_cat_weight := cat_weights.sum
  let first_dog_weight := total_cat_weight + additional_weight
  2 * (first_dog_weight - total_cat_weight)

/-- Theorem: Christine's second dog weighs 16 pounds -/
theorem christines_second_dog_weight :
  weight_second_dog [7, 10, 13] 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_christines_second_dog_weight_l160_16029


namespace NUMINAMATH_CALUDE_twenty_sixth_card_is_red_l160_16028

-- Define the color type
inductive Color
  | Black
  | Red

-- Define the card sequence type
def CardSequence := Nat → Color

-- Define the property of no two consecutive cards being the same color
def AlternatingColors (seq : CardSequence) : Prop :=
  ∀ n : Nat, seq n ≠ seq (n + 1)

-- Define the problem conditions
def ProblemConditions (seq : CardSequence) : Prop :=
  AlternatingColors seq ∧
  seq 10 = Color.Red ∧
  seq 11 = Color.Red ∧
  seq 25 = Color.Black

-- State the theorem
theorem twenty_sixth_card_is_red (seq : CardSequence) 
  (h : ProblemConditions seq) : seq 26 = Color.Red := by
  sorry


end NUMINAMATH_CALUDE_twenty_sixth_card_is_red_l160_16028


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l160_16049

theorem quadratic_equation_roots : ∃! (r₁ r₂ : ℝ),
  (r₁ ≠ r₂) ∧ 
  (r₁^2 - 6*r₁ + 8 = 0) ∧ 
  (r₂^2 - 6*r₂ + 8 = 0) ∧
  (r₁ = 2 ∨ r₁ = 4) ∧
  (r₂ = 2 ∨ r₂ = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l160_16049


namespace NUMINAMATH_CALUDE_quadratic_roots_counterexample_l160_16092

theorem quadratic_roots_counterexample : 
  ∃ (a b c : ℝ), b - c > a ∧ a ≠ 0 ∧ 
  ¬(∃ (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_counterexample_l160_16092


namespace NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l160_16013

theorem vector_addition_and_scalar_multiplication :
  let v1 : Fin 2 → ℝ := ![3, -8]
  let v2 : Fin 2 → ℝ := ![2, -6]
  let scalar : ℝ := 5
  v1 + scalar • v2 = ![13, -38] := by sorry

end NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l160_16013


namespace NUMINAMATH_CALUDE_complex_equation_solution_l160_16038

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 2 * Complex.I → z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l160_16038


namespace NUMINAMATH_CALUDE_train_speed_calculation_train_speed_is_36_l160_16046

/-- Calculates the speed of a train given the following conditions:
  * A jogger is running at 9 kmph
  * The jogger is 270 meters ahead of the train's engine
  * The train is 120 meters long
  * The train takes 39 seconds to pass the jogger
-/
theorem train_speed_calculation (jogger_speed : ℝ) (initial_distance : ℝ) 
  (train_length : ℝ) (passing_time : ℝ) : ℝ :=
  let total_distance := initial_distance + train_length
  let train_speed := (total_distance / 1000) / (passing_time / 3600)
  by
    sorry

/-- The main theorem stating that under the given conditions, 
    the train's speed is 36 kmph -/
theorem train_speed_is_36 : 
  train_speed_calculation 9 270 120 39 = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_train_speed_is_36_l160_16046


namespace NUMINAMATH_CALUDE_divisibility_by_thirty_l160_16039

theorem divisibility_by_thirty (p : ℕ) (h_prime : Nat.Prime p) (h_ge_seven : p ≥ 7) :
  ∃ k : ℕ, p^2 - 1 = 30 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_thirty_l160_16039


namespace NUMINAMATH_CALUDE_expansion_coefficient_l160_16068

theorem expansion_coefficient (m : ℤ) : 
  (Nat.choose 6 3 : ℤ) * m^3 = -160 → m = -2 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l160_16068


namespace NUMINAMATH_CALUDE_inequality_proof_l160_16025

theorem inequality_proof (a b c : ℝ) (h1 : a < b) (h2 : b < 0) : 
  a * (c^2 + 1) < b * (c^2 + 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l160_16025


namespace NUMINAMATH_CALUDE_unfair_coin_flip_probability_l160_16085

def coin_flip_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem unfair_coin_flip_probability :
  let n : ℕ := 8  -- Total number of flips
  let k : ℕ := 3  -- Number of tails
  let p : ℚ := 2/3  -- Probability of tails
  coin_flip_probability n k p = 448/6561 := by
sorry

end NUMINAMATH_CALUDE_unfair_coin_flip_probability_l160_16085


namespace NUMINAMATH_CALUDE_cyclist_hill_time_l160_16078

/-- Calculates the total time for a cyclist to climb and descend a hill. -/
theorem cyclist_hill_time (hill_length : Real) (climbing_speed_kmh : Real) : 
  hill_length = 400 ∧ 
  climbing_speed_kmh = 7.2 →
  (let climbing_speed_ms := climbing_speed_kmh * (1000 / 3600)
   let descending_speed_ms := 2 * climbing_speed_ms
   let time_climbing := hill_length / climbing_speed_ms
   let time_descending := hill_length / descending_speed_ms
   time_climbing + time_descending) = 300 := by
  sorry


end NUMINAMATH_CALUDE_cyclist_hill_time_l160_16078


namespace NUMINAMATH_CALUDE_inequality_proof_l160_16071

theorem inequality_proof (a b : ℝ) (h : a + b ≠ 0) :
  (a + b) / (a^2 - a*b + b^2) ≤ 4 / |a + b| ∧
  ((a + b) / (a^2 - a*b + b^2) = 4 / |a + b| ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l160_16071


namespace NUMINAMATH_CALUDE_probability_of_n_in_polynomial_l160_16076

def word : String := "polynomial"

def count_letter (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem probability_of_n_in_polynomial :
  (count_letter word 'n' : ℚ) / word.length = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_n_in_polynomial_l160_16076


namespace NUMINAMATH_CALUDE_total_frogs_is_48_l160_16021

/-- The number of frogs in Pond A -/
def frogs_in_pond_a : ℕ := 32

/-- The number of frogs in Pond B -/
def frogs_in_pond_b : ℕ := frogs_in_pond_a / 2

/-- The total number of frogs in both ponds -/
def total_frogs : ℕ := frogs_in_pond_a + frogs_in_pond_b

theorem total_frogs_is_48 : total_frogs = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_frogs_is_48_l160_16021


namespace NUMINAMATH_CALUDE_prime_relation_l160_16082

theorem prime_relation (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → q = 11 * p + 1 → q = 23 := by
  sorry

end NUMINAMATH_CALUDE_prime_relation_l160_16082


namespace NUMINAMATH_CALUDE_simplify_expression_l160_16042

theorem simplify_expression (x y : ℝ) : (5 - 4*x) - (7 + 5*x) + 2*y = -2 - 9*x + 2*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l160_16042


namespace NUMINAMATH_CALUDE_inequality_holds_l160_16088

theorem inequality_holds (a b c : ℝ) (h : a > b) : a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l160_16088


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_l160_16086

theorem triangle_cosine_sum (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = 2 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  b * Real.cos C + c * Real.cos B = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_l160_16086


namespace NUMINAMATH_CALUDE_largest_odd_factor_sum_difference_l160_16089

/-- f(n) represents the largest odd factor of a positive integer n -/
def f (n : ℕ+) : ℕ+ :=
  sorry

/-- The sum of f(i) from a to b -/
def sum_f (a b : ℕ+) : ℕ :=
  sorry

theorem largest_odd_factor_sum_difference :
  sum_f 51 100 - sum_f 1 50 = 1656 :=
sorry

end NUMINAMATH_CALUDE_largest_odd_factor_sum_difference_l160_16089


namespace NUMINAMATH_CALUDE_doctors_lawyers_ratio_l160_16063

theorem doctors_lawyers_ratio (d l : ℕ) (h_total : d + l > 0) :
  (38 * d + 55 * l) / (d + l) = 45 →
  d / l = 10 / 7 := by
sorry

end NUMINAMATH_CALUDE_doctors_lawyers_ratio_l160_16063


namespace NUMINAMATH_CALUDE_initial_money_calculation_l160_16044

theorem initial_money_calculation (remaining_money : ℝ) (spent_percentage : ℝ) (initial_money : ℝ) : 
  remaining_money = 2800 →
  spent_percentage = 0.3 →
  initial_money * (1 - spent_percentage) = remaining_money →
  initial_money = 4000 := by
sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l160_16044


namespace NUMINAMATH_CALUDE_inverse_64_mod_97_l160_16073

theorem inverse_64_mod_97 (h : (8⁻¹ : ZMod 97) = 85) : (64⁻¹ : ZMod 97) = 47 := by
  sorry

end NUMINAMATH_CALUDE_inverse_64_mod_97_l160_16073


namespace NUMINAMATH_CALUDE_positive_solution_between_one_and_two_l160_16022

def f (x : ℝ) := x^2 + 3*x - 5

theorem positive_solution_between_one_and_two :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  have h1 : f 1 < 0 := by sorry
  have h2 : f 2 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_positive_solution_between_one_and_two_l160_16022


namespace NUMINAMATH_CALUDE_tank_capacity_l160_16083

theorem tank_capacity (y : ℝ) 
  (h1 : (7/8) * y - 20 = (1/4) * y) : y = 32 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l160_16083


namespace NUMINAMATH_CALUDE_sale_price_calculation_l160_16012

theorem sale_price_calculation (ticket_price : ℝ) (discount_percentage : ℝ) 
  (h1 : ticket_price = 25)
  (h2 : discount_percentage = 25) :
  ticket_price * (1 - discount_percentage / 100) = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_calculation_l160_16012


namespace NUMINAMATH_CALUDE_inequality_proof_l160_16087

theorem inequality_proof (a : ℝ) : 3 * (1 + a^2 + a^4) ≥ (1 + a + a^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l160_16087


namespace NUMINAMATH_CALUDE_range_of_a_l160_16001

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, |2^x - a| < |5 - 2^x|) → 
  3 < a ∧ a < 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l160_16001


namespace NUMINAMATH_CALUDE_lastDigitOf8To19_eq_2_l160_16036

/-- The last digit of 2^n for n > 0 -/
def lastDigitOfPowerOf2 (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => 6

/-- The last digit of 8^19 -/
def lastDigitOf8To19 : ℕ :=
  lastDigitOfPowerOf2 57

theorem lastDigitOf8To19_eq_2 : lastDigitOf8To19 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lastDigitOf8To19_eq_2_l160_16036


namespace NUMINAMATH_CALUDE_john_computer_cost_l160_16097

/-- Calculates the total cost of a computer after replacing a video card -/
def totalComputerCost (initialCost oldCardSale newCardCost : ℕ) : ℕ :=
  initialCost - oldCardSale + newCardCost

/-- Proves that the total cost of John's computer is $1400 -/
theorem john_computer_cost :
  totalComputerCost 1200 300 500 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_john_computer_cost_l160_16097


namespace NUMINAMATH_CALUDE_sum_of_fractions_l160_16043

theorem sum_of_fractions : (3 : ℚ) / 8 + (7 : ℚ) / 9 = (83 : ℚ) / 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l160_16043


namespace NUMINAMATH_CALUDE_tower_blocks_l160_16009

theorem tower_blocks (A R : ℕ) (h : A - R = 30) : 35 + A - R = 65 := by
  sorry

end NUMINAMATH_CALUDE_tower_blocks_l160_16009


namespace NUMINAMATH_CALUDE_unique_solution_condition_l160_16004

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) * (x - 2) = -26 + k * x) ↔ 
  (k = -2 + 6 * Real.sqrt 6 ∨ k = -2 - 6 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l160_16004


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l160_16037

/-- Given a point P with coordinates (x, -4), prove that if the distance from the x-axis to P
    is half the distance from the y-axis to P, then the distance from the y-axis to P is 8 units. -/
theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -4)
  let dist_to_x_axis := |P.2|
  let dist_to_y_axis := |P.1|
  dist_to_x_axis = (1/2) * dist_to_y_axis →
  dist_to_y_axis = 8 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l160_16037


namespace NUMINAMATH_CALUDE_bill_split_correct_l160_16067

-- Define the given values
def total_bill : ℚ := 139
def num_people : ℕ := 3
def tip_percentage : ℚ := 1 / 10

-- Define the function to calculate the amount each person should pay
def amount_per_person (bill : ℚ) (people : ℕ) (tip : ℚ) : ℚ :=
  (bill * (1 + tip)) / people

-- Theorem statement
theorem bill_split_correct :
  amount_per_person total_bill num_people tip_percentage = 5097 / 100 := by
  sorry

end NUMINAMATH_CALUDE_bill_split_correct_l160_16067


namespace NUMINAMATH_CALUDE_time_expression_l160_16059

/-- Represents the motion of a particle under combined constant accelerations -/
structure ParticleMotion where
  g : ℝ  -- Constant acceleration g
  a : ℝ  -- Additional constant acceleration a
  V₀ : ℝ  -- Initial velocity
  t : ℝ  -- Time
  V : ℝ  -- Final velocity
  S : ℝ  -- Displacement

/-- The final velocity equation for the particle motion -/
def velocity_equation (p : ParticleMotion) : Prop :=
  p.V = (p.g + p.a) * p.t + p.V₀

/-- The displacement equation for the particle motion -/
def displacement_equation (p : ParticleMotion) : Prop :=
  p.S = (1/2) * (p.g + p.a) * p.t^2 + p.V₀ * p.t

/-- Theorem stating that the time can be expressed in terms of S, V, and V₀ -/
theorem time_expression (p : ParticleMotion) 
  (h1 : velocity_equation p) 
  (h2 : displacement_equation p) : 
  p.t = (2 * p.S) / (p.V + p.V₀) := by
  sorry

end NUMINAMATH_CALUDE_time_expression_l160_16059


namespace NUMINAMATH_CALUDE_largest_square_area_l160_16096

theorem largest_square_area (x y z : ℝ) (h1 : x^2 + y^2 = z^2) (h2 : x^2 + y^2 + z^2 = 450) :
  z^2 = 225 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l160_16096


namespace NUMINAMATH_CALUDE_max_b_value_l160_16094

def is_lattice_point (x y : ℤ) : Prop := True

def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ x : ℤ, 1 ≤ x → x ≤ 150 → ¬(is_lattice_point x (line_equation m x).num)

theorem max_b_value :
  ∃ b : ℚ, b = 50/149 ∧
    (∀ m : ℚ, 1/3 < m → m < b → no_lattice_points m) ∧
    (∀ b' : ℚ, b < b' → ∃ m : ℚ, 1/3 < m ∧ m < b' ∧ ¬(no_lattice_points m)) :=
sorry

end NUMINAMATH_CALUDE_max_b_value_l160_16094


namespace NUMINAMATH_CALUDE_philip_paintings_per_day_l160_16090

/-- The number of paintings Philip makes per day -/
def paintings_per_day (initial_paintings : ℕ) (total_paintings : ℕ) (days : ℕ) : ℚ :=
  (total_paintings - initial_paintings : ℚ) / days

/-- Theorem: Philip makes 2 paintings per day -/
theorem philip_paintings_per_day :
  paintings_per_day 20 80 30 = 2 := by
  sorry

end NUMINAMATH_CALUDE_philip_paintings_per_day_l160_16090


namespace NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l160_16081

/-- Given a pyramid with volume V and surface area S, prove that when V = 2 and S = 3,
    the surface area of the inscribed sphere is 16π. -/
theorem inscribed_sphere_surface_area (V S : ℝ) (h1 : V = 2) (h2 : S = 3) :
  let r := 3 * V / S
  4 * Real.pi * r^2 = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l160_16081


namespace NUMINAMATH_CALUDE_abs_less_sufficient_not_necessary_for_decreasing_l160_16062

def is_decreasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) ≤ a n

theorem abs_less_sufficient_not_necessary_for_decreasing :
  (∃ a : ℕ → ℝ, (∀ n, |a (n + 1)| < a n) → is_decreasing a) ∧
  (∃ a : ℕ → ℝ, is_decreasing a ∧ ¬(∀ n, |a (n + 1)| < a n)) :=
sorry

end NUMINAMATH_CALUDE_abs_less_sufficient_not_necessary_for_decreasing_l160_16062


namespace NUMINAMATH_CALUDE_floor_product_equation_l160_16016

theorem floor_product_equation (x : ℝ) : 
  (⌊x * ⌊x⌋⌋ = 48) ↔ (x = -48/7) := by sorry

end NUMINAMATH_CALUDE_floor_product_equation_l160_16016


namespace NUMINAMATH_CALUDE_other_side_heads_probability_l160_16011

/-- Represents the three types of coins -/
inductive Coin
  | Normal
  | DoubleHeads
  | DoubleTails

/-- Represents the possible sides of a coin -/
inductive Side
  | Heads
  | Tails

/-- The probability of selecting each coin -/
def coinSelectionProbability : ℚ := 1/3

/-- The probability of getting heads for each coin type -/
def probabilityOfHeads (c : Coin) : ℚ :=
  match c with
  | Coin.Normal => 1/2
  | Coin.DoubleHeads => 1
  | Coin.DoubleTails => 0

/-- The probability that the other side is heads given that heads was observed -/
def probabilityOtherSideHeads : ℚ := 2/3

theorem other_side_heads_probability :
  probabilityOtherSideHeads = 2/3 := by sorry


end NUMINAMATH_CALUDE_other_side_heads_probability_l160_16011


namespace NUMINAMATH_CALUDE_root_product_theorem_l160_16058

theorem root_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 2 = 0) →
  (b^2 - m*b + 2 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 9/2 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l160_16058


namespace NUMINAMATH_CALUDE_car_trip_average_speed_l160_16010

/-- Calculates the average speed of a car trip given the following conditions:
    - The trip lasts for 8 hours
    - The car averages 50 mph for the first 4 hours
    - The car averages 80 mph for the remaining 4 hours
-/
theorem car_trip_average_speed :
  let total_time : ℝ := 8
  let first_segment_time : ℝ := 4
  let second_segment_time : ℝ := total_time - first_segment_time
  let first_segment_speed : ℝ := 50
  let second_segment_speed : ℝ := 80
  let total_distance : ℝ := first_segment_speed * first_segment_time + second_segment_speed * second_segment_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 65 := by sorry

end NUMINAMATH_CALUDE_car_trip_average_speed_l160_16010


namespace NUMINAMATH_CALUDE_system_solution_negative_implies_m_range_l160_16069

theorem system_solution_negative_implies_m_range (m x y : ℝ) : 
  x - y = 2 * m + 7 →
  x + y = 4 * m - 3 →
  x < 0 →
  y < 0 →
  m < -2/3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_negative_implies_m_range_l160_16069


namespace NUMINAMATH_CALUDE_expression_simplification_l160_16084

theorem expression_simplification (x y : ℝ) : 
  3 * x + 5 * x^2 - 4 * y - (6 - 3 * x - 5 * x^2 + 2 * y) - (4 * y^2 - 8 + 2 * x^2 - y) = 
  8 * x^2 - 4 * y^2 + 6 * x - 5 * y + 2 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l160_16084


namespace NUMINAMATH_CALUDE_power_of_seven_mod_thousand_l160_16005

theorem power_of_seven_mod_thousand : 7^2023 % 1000 = 637 := by sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_thousand_l160_16005


namespace NUMINAMATH_CALUDE_windows_installed_correct_l160_16054

/-- Calculates the number of windows already installed given the total number of windows,
    the time to install each window, and the time left to install remaining windows. -/
def windows_installed (total_windows : ℕ) (time_per_window : ℕ) (time_left : ℕ) : ℕ :=
  total_windows - (time_left / time_per_window)

/-- Proves that the number of windows already installed is correct for the given problem. -/
theorem windows_installed_correct :
  windows_installed 14 4 36 = 5 := by
  sorry

end NUMINAMATH_CALUDE_windows_installed_correct_l160_16054


namespace NUMINAMATH_CALUDE_molds_cost_three_l160_16061

/-- Calculates the cost of popsicle molds given the total budget, cost of popsicle sticks,
    cost and yield of juice bottles, and the number of remaining popsicle sticks. -/
def cost_of_molds (total_budget : ℕ) (stick_pack_cost : ℕ) (stick_pack_size : ℕ)
                  (juice_bottle_cost : ℕ) (popsicles_per_bottle : ℕ)
                  (remaining_sticks : ℕ) : ℕ :=
  let used_sticks := stick_pack_size - remaining_sticks
  let bottles_used := used_sticks / popsicles_per_bottle
  let juice_cost := bottles_used * juice_bottle_cost
  let total_spent := stick_pack_cost + juice_cost
  total_budget - total_spent

/-- The cost of the molds is $3 given the specified conditions. -/
theorem molds_cost_three :
  cost_of_molds 10 1 100 2 20 40 = 3 := by
  sorry

end NUMINAMATH_CALUDE_molds_cost_three_l160_16061


namespace NUMINAMATH_CALUDE_marion_score_l160_16020

/-- Given a 40-item exam, prove Marion's score based on Ella's performance -/
theorem marion_score (total_items : ℕ) (ella_incorrect : ℕ) (marion_bonus : ℕ) :
  total_items = 40 →
  ella_incorrect = 4 →
  marion_bonus = 6 →
  (total_items - ella_incorrect) / 2 + marion_bonus = 24 := by
  sorry

#check marion_score

end NUMINAMATH_CALUDE_marion_score_l160_16020


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l160_16045

theorem sqrt_product_simplification (x : ℝ) :
  Real.sqrt (50 * x^2) * Real.sqrt (18 * x^3) * Real.sqrt (98 * x) = 210 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l160_16045


namespace NUMINAMATH_CALUDE_goods_train_length_l160_16003

/-- The length of a goods train given relative speeds and passing time -/
theorem goods_train_length (v_passenger : ℝ) (v_goods : ℝ) (t_pass : ℝ) : 
  v_passenger = 80 → 
  v_goods = 32 → 
  t_pass = 9 →
  ∃ (length : ℝ), abs (length - 280) < 1 ∧ 
    length = (v_passenger + v_goods) * 1000 / 3600 * t_pass :=
by sorry

end NUMINAMATH_CALUDE_goods_train_length_l160_16003


namespace NUMINAMATH_CALUDE_tile_arrangements_l160_16027

def brown_tiles : ℕ := 2
def purple_tiles : ℕ := 1
def green_tiles : ℕ := 2
def yellow_tiles : ℕ := 2

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem tile_arrangements :
  (Nat.factorial total_tiles) / (Nat.factorial brown_tiles * Nat.factorial purple_tiles * Nat.factorial green_tiles * Nat.factorial yellow_tiles) = 630 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l160_16027


namespace NUMINAMATH_CALUDE_cube_root_neg_eight_plus_sqrt_nine_equals_one_l160_16053

theorem cube_root_neg_eight_plus_sqrt_nine_equals_one :
  ((-8 : ℝ) ^ (1/3 : ℝ)) + (9 : ℝ).sqrt = 1 := by sorry

end NUMINAMATH_CALUDE_cube_root_neg_eight_plus_sqrt_nine_equals_one_l160_16053


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l160_16008

theorem arithmetic_expression_evaluation :
  1 / 2 + ((2 / 3 * 3 / 8) + 4) - 8 / 16 = 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l160_16008


namespace NUMINAMATH_CALUDE_second_transfer_amount_l160_16030

/-- Calculates the amount of a bank transfer given the initial balance, 
    first transfer amount, final balance, and service charge rate. -/
def calculate_second_transfer (initial_balance : ℚ) (first_transfer : ℚ) 
  (final_balance : ℚ) (service_charge_rate : ℚ) : ℚ :=
  let first_transfer_with_charge := first_transfer * (1 + service_charge_rate)
  let total_deduction := initial_balance - final_balance
  (total_deduction - first_transfer_with_charge) / (service_charge_rate)

/-- Theorem stating that given the problem conditions, 
    the second transfer amount is $60. -/
theorem second_transfer_amount 
  (initial_balance : ℚ) 
  (first_transfer : ℚ)
  (final_balance : ℚ)
  (service_charge_rate : ℚ)
  (h1 : initial_balance = 400)
  (h2 : first_transfer = 90)
  (h3 : final_balance = 307)
  (h4 : service_charge_rate = 2/100) :
  calculate_second_transfer initial_balance first_transfer final_balance service_charge_rate = 60 := by
  sorry

#eval calculate_second_transfer 400 90 307 (2/100)

end NUMINAMATH_CALUDE_second_transfer_amount_l160_16030


namespace NUMINAMATH_CALUDE_gcf_of_210_and_294_l160_16072

theorem gcf_of_210_and_294 : Nat.gcd 210 294 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_210_and_294_l160_16072


namespace NUMINAMATH_CALUDE_exists_a_satisfying_conditions_l160_16031

def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

theorem exists_a_satisfying_conditions :
  ∃ a : ℝ, a = -2 ∧ 
    (A a ∩ C = ∅) ∧ 
    (∅ ⊂ A a ∩ B) :=
by sorry

end NUMINAMATH_CALUDE_exists_a_satisfying_conditions_l160_16031


namespace NUMINAMATH_CALUDE_hawks_score_l160_16050

theorem hawks_score (eagles hawks ravens : ℕ) : 
  eagles + hawks + ravens = 120 →
  eagles = hawks + 20 →
  ravens = 2 * hawks →
  hawks = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l160_16050
