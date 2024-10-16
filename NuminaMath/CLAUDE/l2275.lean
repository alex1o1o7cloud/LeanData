import Mathlib

namespace NUMINAMATH_CALUDE_boys_in_class_l2275_227526

theorem boys_in_class (total_students : ℕ) (total_cost : ℕ) (boys_cost : ℕ) (girls_cost : ℕ)
  (h1 : total_students = 43)
  (h2 : total_cost = 1101)
  (h3 : boys_cost = 24)
  (h4 : girls_cost = 27) :
  ∃ (boys girls : ℕ),
    boys + girls = total_students ∧
    boys * boys_cost + girls * girls_cost = total_cost ∧
    boys = 20 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l2275_227526


namespace NUMINAMATH_CALUDE_same_sign_l2275_227575

theorem same_sign (a b c : ℝ) (h1 : (b/a) * (c/a) > 1) (h2 : (b/a) + (c/a) ≥ -2) :
  (a > 0 ∧ b > 0 ∧ c > 0) ∨ (a < 0 ∧ b < 0 ∧ c < 0) :=
sorry

end NUMINAMATH_CALUDE_same_sign_l2275_227575


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_divisibility_l2275_227561

theorem consecutive_numbers_product_divisibility (n : ℕ) (h : n > 1) :
  ∃ k : ℕ, ∀ p : ℕ,
    Prime p →
    (p ≤ 2*n + 1 ↔ ∃ i : ℕ, i < n ∧ p ∣ (k + i)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_divisibility_l2275_227561


namespace NUMINAMATH_CALUDE_bus_network_routes_count_l2275_227557

/-- A bus network in a city. -/
structure BusNetwork where
  /-- The set of bus stops. -/
  stops : Type
  /-- The set of bus routes. -/
  routes : Type
  /-- Predicate indicating if a stop is on a route. -/
  on_route : stops → routes → Prop

/-- Properties of a valid bus network. -/
class ValidBusNetwork (bn : BusNetwork) where
  /-- From any stop to any other stop, you can get there without a transfer. -/
  no_transfer : ∀ (s₁ s₂ : bn.stops), ∃ (r : bn.routes), bn.on_route s₁ r ∧ bn.on_route s₂ r
  /-- For any pair of routes, there is exactly one stop where you can transfer from one route to the other. -/
  unique_transfer : ∀ (r₁ r₂ : bn.routes), ∃! (s : bn.stops), bn.on_route s r₁ ∧ bn.on_route s r₂
  /-- Each route has exactly three stops. -/
  three_stops : ∀ (r : bn.routes), ∃! (s₁ s₂ s₃ : bn.stops), 
    bn.on_route s₁ r ∧ bn.on_route s₂ r ∧ bn.on_route s₃ r ∧ 
    s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃

/-- The theorem stating the relationship between the number of routes and stops. -/
theorem bus_network_routes_count {bn : BusNetwork} [ValidBusNetwork bn] [Fintype bn.stops] [Fintype bn.routes] : 
  Fintype.card bn.routes = Fintype.card bn.stops * (Fintype.card bn.stops - 1) + 1 :=
sorry

end NUMINAMATH_CALUDE_bus_network_routes_count_l2275_227557


namespace NUMINAMATH_CALUDE_odd_integers_sum_product_l2275_227578

theorem odd_integers_sum_product (p q : ℕ) : 
  (p < 16 ∧ q < 16 ∧ Odd p ∧ Odd q) →
  (∃ (S : Finset ℕ), S = {n | ∃ (a b : ℕ), a < 16 ∧ b < 16 ∧ Odd a ∧ Odd b ∧ n = a * b + a + b} ∧ 
   Finset.card S = 36) :=
by sorry

end NUMINAMATH_CALUDE_odd_integers_sum_product_l2275_227578


namespace NUMINAMATH_CALUDE_modulus_of_complex_reciprocal_l2275_227597

theorem modulus_of_complex_reciprocal (i : ℂ) (h : i^2 = -1) :
  Complex.abs (1 / (i - 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_reciprocal_l2275_227597


namespace NUMINAMATH_CALUDE_question_mark_value_l2275_227514

theorem question_mark_value : ∃ x : ℤ, 27474 + 3699 + x - 2047 = 31111 ∧ x = 1985 := by
  sorry

end NUMINAMATH_CALUDE_question_mark_value_l2275_227514


namespace NUMINAMATH_CALUDE_equal_time_travel_ratio_l2275_227541

/-- The ratio of distances when travel times are equal --/
theorem equal_time_travel_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  y / 1 = x / 1 + (x + y) / 10 → x / y = 9 / 11 := by
  sorry

#check equal_time_travel_ratio

end NUMINAMATH_CALUDE_equal_time_travel_ratio_l2275_227541


namespace NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_eq_two_l2275_227536

theorem sqrt_eight_div_sqrt_two_eq_two :
  Real.sqrt 8 / Real.sqrt 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_eq_two_l2275_227536


namespace NUMINAMATH_CALUDE_rational_expression_equality_inequality_system_solution_l2275_227550

-- Part 1
theorem rational_expression_equality (x : ℝ) (h : x ≠ 3) :
  (2*x + 4) / (x^2 - 6*x + 9) / ((2*x - 1) / (x - 3) - 1) = 2 / (x - 3) :=
sorry

-- Part 2
theorem inequality_system_solution (x : ℝ) :
  (5*x - 2 > 3*(x + 1) ∧ (1/2)*x - 1 ≥ 7 - (3/2)*x) ↔ x ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_rational_expression_equality_inequality_system_solution_l2275_227550


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_36_l2275_227579

theorem sum_of_roots_equals_36 : ∃ (x₁ x₂ x₃ : ℝ),
  (∀ x, (11 - x)^3 + (13 - x)^3 = (24 - 2*x)^3 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  x₁ + x₂ + x₃ = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_36_l2275_227579


namespace NUMINAMATH_CALUDE_distance_difference_l2275_227517

def time : ℝ := 6
def carlos_distance : ℝ := 108
def daniel_distance : ℝ := 90

theorem distance_difference : carlos_distance - daniel_distance = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l2275_227517


namespace NUMINAMATH_CALUDE_min_value_problem_l2275_227595

theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  ∃ (m : ℝ), m = (1 : ℝ)/5184 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → 
    1/x + 1/y + 1/z = 9 → x^4 * y^3 * z^2 ≥ m ∧
    a^4 * b^3 * c^2 = m :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2275_227595


namespace NUMINAMATH_CALUDE_parabola_c_value_l2275_227510

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord (-1) = 4 →   -- Condition: parabola passes through (4, -1)
  p.x_coord (-5) = 1 →   -- Condition: vertex is at (1, -5)
  (∀ y, p.x_coord y ≥ p.x_coord (-5)) →  -- Condition: (1, -5) is the vertex
  p.c = 145/12 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2275_227510


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2275_227501

-- Define the universe of discourse
variable (U : Type)

-- Define the predicate for being a domestic mobile phone
variable (D : U → Prop)

-- Define the predicate for having trap consumption
variable (T : U → Prop)

-- State the theorem
theorem negation_of_universal_statement :
  (¬ ∀ x, D x → T x) ↔ (∃ x, D x ∧ ¬ T x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2275_227501


namespace NUMINAMATH_CALUDE_shadow_point_theorem_l2275_227560

-- Define shadow point
def isShadowPoint (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ y > x, f y > f x

-- State the theorem
theorem shadow_point_theorem (f : ℝ → ℝ) (a b : ℝ) 
  (hf : Continuous f) 
  (hab : a < b)
  (h_shadow : ∀ x ∈ Set.Ioo a b, isShadowPoint f x)
  (ha_not_shadow : ¬ isShadowPoint f a)
  (hb_not_shadow : ¬ isShadowPoint f b) :
  (∀ x ∈ Set.Ioo a b, f x ≤ f b) ∧ f a = f b :=
sorry

end NUMINAMATH_CALUDE_shadow_point_theorem_l2275_227560


namespace NUMINAMATH_CALUDE_sucrose_solution_volume_l2275_227594

/-- Given a sucrose solution where 60 cubic centimeters contain 6 grams of sucrose,
    prove that 100 cubic centimeters contain 10 grams of sucrose. -/
theorem sucrose_solution_volume (solution_volume : ℝ) (sucrose_mass : ℝ) :
  (60 : ℝ) / solution_volume = 6 / sucrose_mass →
  (100 : ℝ) / solution_volume = 10 / sucrose_mass :=
by
  sorry

end NUMINAMATH_CALUDE_sucrose_solution_volume_l2275_227594


namespace NUMINAMATH_CALUDE_apple_pear_box_difference_l2275_227569

theorem apple_pear_box_difference :
  ∀ (initial_apples initial_pears additional : ℕ),
    initial_apples = 25 →
    initial_pears = 12 →
    additional = 8 →
    (initial_apples + additional) - (initial_pears + additional) = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_pear_box_difference_l2275_227569


namespace NUMINAMATH_CALUDE_fathers_digging_time_l2275_227524

/-- The father's digging rate in feet per hour -/
def fathersRate : ℝ := 4

/-- The depth difference between Michael's hole and twice his father's hole depth in feet -/
def depthDifference : ℝ := 400

/-- Michael's digging time in hours -/
def michaelsTime : ℝ := 700

/-- Father's digging time in hours -/
def fathersTime : ℝ := 400

theorem fathers_digging_time :
  ∀ (fathersDepth michaelsDepth : ℝ),
  michaelsDepth = 2 * fathersDepth - depthDifference →
  michaelsDepth = fathersRate * michaelsTime →
  fathersDepth = fathersRate * fathersTime :=
by sorry

end NUMINAMATH_CALUDE_fathers_digging_time_l2275_227524


namespace NUMINAMATH_CALUDE_complete_square_equivalence_l2275_227520

/-- Given a quadratic equation x^2 - 4x = 5, prove that it is equivalent to (x-2)^2 = 9 when completed square. -/
theorem complete_square_equivalence :
  ∀ x : ℝ, x^2 - 4*x = 5 ↔ (x-2)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_equivalence_l2275_227520


namespace NUMINAMATH_CALUDE_factor_expression_l2275_227562

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x - 5) * (x + 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2275_227562


namespace NUMINAMATH_CALUDE_prime_divisors_of_29_pow_p_plus_1_l2275_227576

theorem prime_divisors_of_29_pow_p_plus_1 (p : Nat) :
  Nat.Prime p ∧ p ∣ 29^p + 1 ↔ p = 2 ∨ p = 3 ∨ p = 5 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisors_of_29_pow_p_plus_1_l2275_227576


namespace NUMINAMATH_CALUDE_min_participants_is_eleven_l2275_227588

/-- Represents the number of participants in each grade --/
structure Participants where
  fifth : Nat
  sixth : Nat
  seventh : Nat

/-- Checks if the given number of participants satisfies all conditions --/
def satisfiesConditions (n : Nat) (p : Participants) : Prop :=
  p.fifth + p.sixth + p.seventh = n ∧
  (25 * n < 100 * p.fifth) ∧ (100 * p.fifth < 35 * n) ∧
  (30 * n < 100 * p.sixth) ∧ (100 * p.sixth < 40 * n) ∧
  (35 * n < 100 * p.seventh) ∧ (100 * p.seventh < 45 * n)

/-- States that 11 is the minimum number of participants satisfying all conditions --/
theorem min_participants_is_eleven :
  ∃ (p : Participants), satisfiesConditions 11 p ∧
  ∀ (m : Nat) (q : Participants), m < 11 → ¬satisfiesConditions m q :=
by sorry

end NUMINAMATH_CALUDE_min_participants_is_eleven_l2275_227588


namespace NUMINAMATH_CALUDE_rhombus_area_l2275_227522

/-- The area of a rhombus with side length 20 cm and an angle of 60 degrees between two adjacent sides is 200√3 cm². -/
theorem rhombus_area (side : ℝ) (angle : ℝ) (h1 : side = 20) (h2 : angle = π / 3) :
  side * side * Real.sin angle = 200 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l2275_227522


namespace NUMINAMATH_CALUDE_fraction_equality_implies_equality_l2275_227515

theorem fraction_equality_implies_equality (x y m : ℝ) (h : m ≠ 0) :
  x / m = y / m → x = y := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_equality_l2275_227515


namespace NUMINAMATH_CALUDE_drought_pond_fill_time_l2275_227567

/-- Proves the time required to fill a pond under drought conditions -/
theorem drought_pond_fill_time 
  (pond_capacity : ℝ) 
  (normal_rate : ℝ) 
  (drought_factor : ℝ) 
  (h1 : pond_capacity = 200) 
  (h2 : normal_rate = 6) 
  (h3 : drought_factor = 2/3) : 
  pond_capacity / (normal_rate * drought_factor) = 50 := by
  sorry

#check drought_pond_fill_time

end NUMINAMATH_CALUDE_drought_pond_fill_time_l2275_227567


namespace NUMINAMATH_CALUDE_scientific_notation_570_million_l2275_227574

theorem scientific_notation_570_million :
  (570000000 : ℝ) = 5.7 * (10 : ℝ)^8 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_570_million_l2275_227574


namespace NUMINAMATH_CALUDE_sum_of_zeros_transformed_parabola_l2275_227535

/-- The sum of zeros of a transformed parabola -/
theorem sum_of_zeros_transformed_parabola : 
  let f (x : ℝ) := (x - 3)^2 + 4
  let g (x : ℝ) := -(x - 7)^2 + 7
  ∃ a b : ℝ, g a = 0 ∧ g b = 0 ∧ a + b = 14 := by
sorry

end NUMINAMATH_CALUDE_sum_of_zeros_transformed_parabola_l2275_227535


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2275_227558

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℕ, ∃ m : ℕ,
    (120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
    (∀ k : ℕ, k > 120 → ∃ p : ℕ, ¬(k ∣ (p * (p + 1) * (p + 2) * (p + 3) * (p + 4)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l2275_227558


namespace NUMINAMATH_CALUDE_inequality_solution_l2275_227598

-- Define the inequality function
def f (x : ℝ) := x * (x - 2)

-- Define the solution set
def solution_set : Set ℝ := {x | x < 0 ∨ x > 2}

-- Theorem statement
theorem inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2275_227598


namespace NUMINAMATH_CALUDE_stadium_attendance_l2275_227543

theorem stadium_attendance (total : ℕ) (girls : ℕ) 
  (h1 : total = 600) 
  (h2 : girls = 240) : 
  total - ((total - girls) / 4 + girls / 8) = 480 := by
  sorry

end NUMINAMATH_CALUDE_stadium_attendance_l2275_227543


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2275_227521

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, -3) and (-4, 15) is 8 -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := -3
  let x₂ : ℝ := -4
  let y₂ : ℝ := 15
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 8 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l2275_227521


namespace NUMINAMATH_CALUDE_complex_product_modulus_l2275_227506

theorem complex_product_modulus (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = t - 3 * Complex.I →
  t = 6 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_complex_product_modulus_l2275_227506


namespace NUMINAMATH_CALUDE_max_triplets_coordinate_plane_l2275_227518

/-- Given 100 points on a coordinate plane, prove that the maximum number of triplets (A, B, C) 
    where A and B have the same y-coordinate and B and C have the same x-coordinate is 8100. -/
theorem max_triplets_coordinate_plane (points : Finset (ℝ × ℝ)) 
    (h : points.card = 100) : 
  (Finset.sum points (fun B => 
    (points.filter (fun A => A.2 = B.2)).card * 
    (points.filter (fun C => C.1 = B.1)).card
  )) ≤ 8100 := by
  sorry

end NUMINAMATH_CALUDE_max_triplets_coordinate_plane_l2275_227518


namespace NUMINAMATH_CALUDE_regression_line_y_change_l2275_227533

/-- Represents a linear regression equation of the form ŷ = a + bx -/
structure RegressionLine where
  a : ℝ
  b : ℝ

/-- The change in y when x increases by 1 unit -/
def yChange (line : RegressionLine) : ℝ := line.b

theorem regression_line_y_change 
  (line : RegressionLine) 
  (h : line = { a := 2, b := -1.5 }) : 
  yChange line = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_y_change_l2275_227533


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l2275_227577

theorem arithmetic_simplification :
  (30 - (2030 - 30 * 2)) + (2030 - (30 * 2 - 30)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l2275_227577


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2275_227539

def set_A (x y : ℝ) : Prop := abs x ≤ 1 ∧ abs y ≤ 1 ∧ x * y ≤ 0

def set_B (x y : ℝ) : Prop := abs x ≤ 1 ∧ abs y ≤ 1 ∧ x^2 + y^2 ≤ 1

theorem inequality_equivalence (x y : ℝ) :
  Real.sqrt (1 - x^2) * Real.sqrt (1 - y^2) ≥ x * y ↔ set_A x y ∨ set_B x y :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2275_227539


namespace NUMINAMATH_CALUDE_diophantine_equation_solvable_l2275_227591

theorem diophantine_equation_solvable (p : ℕ) (hp : Nat.Prime p) : 
  ∃ (x y z : ℤ), x^2 + y^2 + p * z = 2003 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solvable_l2275_227591


namespace NUMINAMATH_CALUDE_whisker_ratio_proof_l2275_227544

/-- The number of whiskers Princess Puff has -/
def princess_puff_whiskers : ℕ := 14

/-- The number of whiskers Catman Do has -/
def catman_do_whiskers : ℕ := 22

/-- The number of whiskers Catman Do is missing compared to the ratio -/
def missing_whiskers : ℕ := 6

/-- The ratio of Catman Do's whiskers to Princess Puff's whiskers -/
def whisker_ratio : ℚ := 2 / 1

theorem whisker_ratio_proof :
  (catman_do_whiskers + missing_whiskers : ℚ) / princess_puff_whiskers = whisker_ratio :=
sorry

end NUMINAMATH_CALUDE_whisker_ratio_proof_l2275_227544


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l2275_227573

/-- Geometric sequence with given properties -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 3/2) ∧
  (∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q) ∧
  (a 1 + a 2 + a 3 = 9/2)

/-- The general term of the geometric sequence -/
theorem geometric_sequence_general_term (a : ℕ → ℚ) (h : geometric_sequence a) :
  (∀ n : ℕ, a n = 3/2 * (-2)^(n - 1)) ∨ (∀ n : ℕ, a n = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l2275_227573


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2275_227507

/-- Given a hyperbola C with equation x²/(m²+3) - y²/m² = 1 where m > 0,
    and asymptote equation y = ±(1/2)x, prove the following properties --/
theorem hyperbola_properties (m : ℝ) (h1 : m > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / (m^2 + 3) - y^2 / m^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | y = (1/2) * x ∨ y = -(1/2) * x}
  (∀ (x y : ℝ), (x, y) ∈ C → (x, y) ∈ asymptote → x ≠ 0 → y / x = 1/2 ∨ y / x = -1/2) →
  (m = 1) ∧
  (∃ (x y : ℝ), (x, y) ∈ C ∧ y = Real.log (x - 1) ∧ 
    (x^2 / (m^2 + 3) = 1 ∨ y = 0)) ∧
  (∀ (x y : ℝ), y^2 - x^2 / 4 = 1 ↔ (x, y) ∈ asymptote) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2275_227507


namespace NUMINAMATH_CALUDE_sarah_coin_collection_l2275_227552

theorem sarah_coin_collection :
  ∀ (n d q : ℕ),
  n + d + q = 30 →
  5 * n + 10 * d + 50 * q = 600 →
  d = n + 4 →
  q = n + 2 :=
by sorry

end NUMINAMATH_CALUDE_sarah_coin_collection_l2275_227552


namespace NUMINAMATH_CALUDE_characterize_S_l2275_227551

-- Define the set of possible values for 1/x + 1/y
def S : Set ℝ := { z | ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ z = 1/x + 1/y }

-- Theorem statement
theorem characterize_S : S = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_characterize_S_l2275_227551


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2275_227548

/-- The minimum distance from the origin to a point on the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {(x, y) : ℝ × ℝ | x + y = 4}
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧ 
    ∀ (P : ℝ × ℝ), P ∈ line → 
      Real.sqrt (P.1^2 + P.2^2) ≥ d ∧ 
      ∃ (Q : ℝ × ℝ), Q ∈ line ∧ Real.sqrt (Q.1^2 + Q.2^2) = d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2275_227548


namespace NUMINAMATH_CALUDE_dodecagon_hexagon_area_ratio_l2275_227525

/-- Given a regular dodecagon with area n and a hexagon ACEGIK formed by
    connecting every second vertex with area m, prove that m/n = √3 - 3/2 -/
theorem dodecagon_hexagon_area_ratio (n m : ℝ) : 
  n > 0 → -- Assuming positive area for the dodecagon
  (∃ (a : ℝ), a > 0 ∧ n = 3 * a^2 * (2 + Real.sqrt 3)) → -- Area formula for dodecagon
  (∃ (s : ℝ), s > 0 ∧ m = (3 * Real.sqrt 3 / 2) * s^2) → -- Area formula for hexagon
  m / n = Real.sqrt 3 - 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_hexagon_area_ratio_l2275_227525


namespace NUMINAMATH_CALUDE_power_of_product_with_exponent_l2275_227532

theorem power_of_product_with_exponent (x y : ℝ) : (-x * y^3)^2 = x^2 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_with_exponent_l2275_227532


namespace NUMINAMATH_CALUDE_ellipse_symmetric_points_range_l2275_227513

/-- Given an ellipse and two symmetric points on it, prove the range of m -/
theorem ellipse_symmetric_points_range (x₁ y₁ x₂ y₂ m : ℝ) : 
  (x₁^2 / 4 + y₁^2 / 3 = 1) →  -- Point A on ellipse
  (x₂^2 / 4 + y₂^2 / 3 = 1) →  -- Point B on ellipse
  ((y₁ + y₂) / 2 = 4 * ((x₁ + x₂) / 2) + m) →  -- A and B symmetric about y = 4x + m
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →  -- A and B are distinct
  (-2 * Real.sqrt 13 / 13 < m ∧ m < 2 * Real.sqrt 13 / 13) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_symmetric_points_range_l2275_227513


namespace NUMINAMATH_CALUDE_hoseok_number_subtraction_l2275_227559

theorem hoseok_number_subtraction (n : ℕ) : n / 10 = 6 → n - 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_number_subtraction_l2275_227559


namespace NUMINAMATH_CALUDE_vector_properties_l2275_227529

def a : ℝ × ℝ := (2, 4)
def b : ℝ × ℝ := (-2, 1)

theorem vector_properties : 
  (a.1 * b.1 + a.2 * b.2 = 0) ∧ 
  (((a.1 + b.1)^2 + (a.2 + b.2)^2).sqrt = 5) ∧
  (((a.1 - b.1)^2 + (a.2 - b.2)^2).sqrt = 5) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l2275_227529


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2275_227570

theorem regular_polygon_sides (n : ℕ) (central_angle : ℝ) : 
  n > 0 ∧ central_angle = 72 → (360 : ℝ) / n = central_angle → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2275_227570


namespace NUMINAMATH_CALUDE_special_line_equation_l2275_227519

/-- A line passing through point (3, -1) with equal absolute values of intercepts on both axes -/
structure SpecialLine where
  -- The equation of the line in the form ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through (3, -1)
  passes_through : a * 3 + b * (-1) + c = 0
  -- The line has equal absolute values of intercepts on both axes
  equal_intercepts : |a / b| = |b / a| ∨ (a = 0 ∧ b ≠ 0) ∨ (b = 0 ∧ a ≠ 0)

/-- The possible equations for the special line -/
def possible_equations (l : SpecialLine) : Prop :=
  (l.a = 1 ∧ l.b = 1 ∧ l.c = -2) ∨
  (l.a = 1 ∧ l.b = -1 ∧ l.c = -4) ∨
  (l.a = 3 ∧ l.b = 1 ∧ l.c = 0)

/-- Theorem stating that any SpecialLine must have one of the possible equations -/
theorem special_line_equation (l : SpecialLine) : possible_equations l := by
  sorry

end NUMINAMATH_CALUDE_special_line_equation_l2275_227519


namespace NUMINAMATH_CALUDE_equation_solutions_l2275_227587

theorem equation_solutions :
  (∃ x : ℝ, 4.8 - 3 * x = 1.8 ∧ x = 1) ∧
  (∃ x : ℝ, (1/8) / (1/5) = x / 24 ∧ x = 15) ∧
  (∃ x : ℝ, 7.5 * x + 6.5 * x = 2.8 ∧ x = 0.2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2275_227587


namespace NUMINAMATH_CALUDE_marbleChoices_eq_56_l2275_227508

/-- A function that returns the number of ways to choose one marble from a set of 15 
    and two ordered marbles from a set of 8 such that the sum of the two chosen marbles 
    equals the number on the single chosen marble -/
def marbleChoices : ℕ :=
  let jessicaMarbles := Finset.range 15
  let myMarbles := Finset.range 8
  Finset.sum jessicaMarbles (λ j => 
    Finset.sum myMarbles (λ m1 => 
      Finset.sum myMarbles (λ m2 => 
        if m1 + m2 + 2 = j + 1 then 1 else 0)))

theorem marbleChoices_eq_56 : marbleChoices = 56 := by
  sorry

end NUMINAMATH_CALUDE_marbleChoices_eq_56_l2275_227508


namespace NUMINAMATH_CALUDE_cupcake_business_loan_payment_l2275_227596

/-- Calculates the monthly payment for a loan given the total loan amount, down payment, and loan term in years. -/
def calculate_monthly_payment (total_loan : ℕ) (down_payment : ℕ) (years : ℕ) : ℕ :=
  let amount_to_finance := total_loan - down_payment
  let months := years * 12
  amount_to_finance / months

/-- Proves that for a loan of $46,000 with a $10,000 down payment to be paid over 5 years, the monthly payment is $600. -/
theorem cupcake_business_loan_payment :
  calculate_monthly_payment 46000 10000 5 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_business_loan_payment_l2275_227596


namespace NUMINAMATH_CALUDE_trig_identity_l2275_227502

theorem trig_identity (α : ℝ) : 
  (Real.sin (α - π/6))^2 + (Real.sin (α + π/6))^2 - (Real.sin α)^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2275_227502


namespace NUMINAMATH_CALUDE_alternating_draw_probability_l2275_227505

/-- Represents the number of white balls in the box -/
def white_balls : ℕ := 4

/-- Represents the number of black balls in the box -/
def black_balls : ℕ := 4

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls

/-- Represents the probability of drawing all balls in alternating colors -/
def alternating_probability : ℚ := 1 / 35

/-- Theorem stating that the probability of drawing all balls in alternating colors is 1/35 -/
theorem alternating_draw_probability :
  alternating_probability = 1 / 35 := by sorry

end NUMINAMATH_CALUDE_alternating_draw_probability_l2275_227505


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l2275_227585

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_l2275_227585


namespace NUMINAMATH_CALUDE_no_solutions_for_absolute_value_equation_l2275_227547

theorem no_solutions_for_absolute_value_equation :
  ¬ ∃ (x : ℝ), |x - 3| = x^2 + 2*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_absolute_value_equation_l2275_227547


namespace NUMINAMATH_CALUDE_sin_n_squared_not_converge_to_zero_l2275_227542

/-- The sequence x_n = sin(n^2) does not converge to zero. -/
theorem sin_n_squared_not_converge_to_zero :
  ¬ (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |Real.sin (n^2)| < ε) := by
  sorry

end NUMINAMATH_CALUDE_sin_n_squared_not_converge_to_zero_l2275_227542


namespace NUMINAMATH_CALUDE_point_symmetry_l2275_227538

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two points are symmetric with respect to a line -/
def isSymmetric (p1 p2 : Point) (l : Line) : Prop :=
  -- The midpoint of the two points lies on the line
  l.a * ((p1.x + p2.x) / 2) + l.b * ((p1.y + p2.y) / 2) + l.c = 0 ∧
  -- The line connecting the two points is perpendicular to the given line
  (p2.x - p1.x) * l.a + (p2.y - p1.y) * l.b = 0

theorem point_symmetry :
  let a : Point := ⟨-1, 2⟩
  let b : Point := ⟨1, 4⟩
  let l : Line := ⟨1, 1, -3⟩
  isSymmetric a b l := by sorry

end NUMINAMATH_CALUDE_point_symmetry_l2275_227538


namespace NUMINAMATH_CALUDE_sum_equation_solution_l2275_227523

/-- Given a real number k > 1 satisfying the infinite sum equation,
    prove that k equals the given expression. -/
theorem sum_equation_solution (k : ℝ) 
  (h1 : k > 1)
  (h2 : ∑' n, (7 * n - 2) / k^n = 3) :
  k = (21 + Real.sqrt 477) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_equation_solution_l2275_227523


namespace NUMINAMATH_CALUDE_circus_tickets_l2275_227568

theorem circus_tickets (ticket_cost : ℕ) (total_spent : ℕ) (h1 : ticket_cost = 44) (h2 : total_spent = 308) :
  total_spent / ticket_cost = 7 :=
sorry

end NUMINAMATH_CALUDE_circus_tickets_l2275_227568


namespace NUMINAMATH_CALUDE_product_selection_probabilities_l2275_227504

/-- A box containing products -/
structure Box where
  total : ℕ
  good : ℕ
  defective : ℕ
  h_total : total = good + defective

/-- The probability of an event when selecting two products from a box -/
def probability (box : Box) (favorable : ℕ) : ℚ :=
  favorable / (box.total.choose 2)

theorem product_selection_probabilities (box : Box) 
  (h_total : box.total = 6)
  (h_good : box.good = 4)
  (h_defective : box.defective = 2) :
  probability box (box.good * box.defective) = 8 / 15 ∧
  probability box (box.good.choose 2) = 2 / 5 ∧
  1 - probability box (box.good.choose 2) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_product_selection_probabilities_l2275_227504


namespace NUMINAMATH_CALUDE_seashells_given_l2275_227531

theorem seashells_given (initial : ℕ) (left : ℕ) (given : ℕ) : 
  initial ≥ left → given = initial - left → given = 62 - 13 :=
by sorry

end NUMINAMATH_CALUDE_seashells_given_l2275_227531


namespace NUMINAMATH_CALUDE_half_hexagon_perimeter_l2275_227554

/-- A polygon that forms one half of a regular hexagon by symmetrically splitting it -/
structure HalfHexagonPolygon where
  side_length : ℝ
  is_positive : side_length > 0

/-- The perimeter of a HalfHexagonPolygon -/
def perimeter (p : HalfHexagonPolygon) : ℝ :=
  3 * p.side_length

/-- Theorem: The perimeter of a HalfHexagonPolygon is equal to 3 times its side length -/
theorem half_hexagon_perimeter (p : HalfHexagonPolygon) :
  perimeter p = 3 * p.side_length := by
  sorry

end NUMINAMATH_CALUDE_half_hexagon_perimeter_l2275_227554


namespace NUMINAMATH_CALUDE_stratified_sampling_survey_l2275_227546

/-- Given a stratified sampling survey with a total population of 2400 (including 1000 female students),
    if 80 female students are included in a sample of size n, and the sampling fraction is consistent
    across all groups, then n = 192. -/
theorem stratified_sampling_survey (total_population : ℕ) (female_students : ℕ) (sample_size : ℕ) 
    (sampled_females : ℕ) (h1 : total_population = 2400) (h2 : female_students = 1000) 
    (h3 : sampled_females = 80) (h4 : sampled_females * total_population = sample_size * female_students) : 
    sample_size = 192 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_survey_l2275_227546


namespace NUMINAMATH_CALUDE_smallest_multiple_of_eleven_l2275_227528

theorem smallest_multiple_of_eleven (x y : ℤ) 
  (h1 : ∃ k : ℤ, x + 2 = 11 * k) 
  (h2 : ∃ m : ℤ, y - 1 = 11 * m) : 
  (∃ n : ℕ+, ∃ p : ℤ, x^2 + x*y + y^2 + n = 11 * p) ∧ 
  (∀ n : ℕ+, n < 8 → ¬∃ p : ℤ, x^2 + x*y + y^2 + n = 11 * p) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_eleven_l2275_227528


namespace NUMINAMATH_CALUDE_machine_net_worth_l2275_227593

/-- Calculate the total net worth of a machine after 2 years given depreciation and maintenance costs -/
theorem machine_net_worth 
  (initial_value : ℝ)
  (depreciation_rate : ℝ)
  (initial_maintenance_cost : ℝ)
  (maintenance_increase_rate : ℝ)
  (h1 : initial_value = 40000)
  (h2 : depreciation_rate = 0.05)
  (h3 : initial_maintenance_cost = 2000)
  (h4 : maintenance_increase_rate = 0.03) :
  let value_after_year_1 := initial_value * (1 - depreciation_rate)
  let value_after_year_2 := value_after_year_1 * (1 - depreciation_rate)
  let maintenance_cost_year_1 := initial_maintenance_cost
  let maintenance_cost_year_2 := initial_maintenance_cost * (1 + maintenance_increase_rate)
  let total_maintenance_cost := maintenance_cost_year_1 + maintenance_cost_year_2
  let net_worth := value_after_year_2 - total_maintenance_cost
  net_worth = 32040 := by
  sorry


end NUMINAMATH_CALUDE_machine_net_worth_l2275_227593


namespace NUMINAMATH_CALUDE_wall_area_l2275_227580

theorem wall_area (small_tile_area small_tile_proportion total_wall_area : ℝ) 
  (h1 : small_tile_proportion = 1 / 2)
  (h2 : small_tile_area = 80)
  (h3 : small_tile_area = small_tile_proportion * total_wall_area) :
  total_wall_area = 160 := by
  sorry

end NUMINAMATH_CALUDE_wall_area_l2275_227580


namespace NUMINAMATH_CALUDE_fraction_decimal_digits_l2275_227590

/-- The number of digits to the right of the decimal point when a positive rational number is expressed as a decimal. -/
def decimal_digits (q : ℚ) : ℕ :=
  sorry

/-- The fraction in question -/
def fraction : ℚ := (4^7) / (8^5 * 1250)

/-- Theorem stating that the number of digits to the right of the decimal point
    in the decimal representation of the given fraction is 3 -/
theorem fraction_decimal_digits :
  decimal_digits fraction = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_decimal_digits_l2275_227590


namespace NUMINAMATH_CALUDE_prob_at_least_two_of_six_l2275_227583

/-- The number of questions randomly guessed -/
def n : ℕ := 6

/-- The number of choices for each question -/
def k : ℕ := 5

/-- The probability of getting a single question correct -/
def p : ℚ := 1 / k

/-- The probability of getting a single question incorrect -/
def q : ℚ := 1 - p

/-- The probability of getting at least two questions correct out of n questions -/
def prob_at_least_two (n : ℕ) (p : ℚ) : ℚ :=
  1 - (q ^ n + n * p * q ^ (n - 1))

theorem prob_at_least_two_of_six :
  prob_at_least_two n p = 5385 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_of_six_l2275_227583


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l2275_227512

theorem unique_two_digit_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  2 ∣ n ∧ 
  3 ∣ (n + 1) ∧ 
  4 ∣ (n + 2) ∧ 
  5 ∣ (n + 3) ∧ 
  n = 62 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l2275_227512


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l2275_227581

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), C = Real.sqrt (8 / 3) ∧
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z)) ∧
  (∀ (C' : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 2 ≥ C' * (x + y + z)) → C' ≤ C) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l2275_227581


namespace NUMINAMATH_CALUDE_escalator_time_l2275_227566

/-- The time taken for a person to cover the entire length of an escalator -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ)
  (h1 : escalator_speed = 12)
  (h2 : person_speed = 2)
  (h3 : escalator_length = 196) :
  escalator_length / (escalator_speed + person_speed) = 14 :=
by sorry

end NUMINAMATH_CALUDE_escalator_time_l2275_227566


namespace NUMINAMATH_CALUDE_average_weight_problem_l2275_227592

theorem average_weight_problem (d e f : ℝ) 
  (h1 : (d + e) / 2 = 35)
  (h2 : (e + f) / 2 = 41)
  (h3 : e = 26) :
  (d + e + f) / 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l2275_227592


namespace NUMINAMATH_CALUDE_roots_product_value_l2275_227563

theorem roots_product_value (x₁ x₂ : ℝ) : 
  (3 * x₁^2 - 9 * x₁ - 21 = 0) → 
  (3 * x₂^2 - 9 * x₂ - 21 = 0) → 
  (3 * x₁ - 4) * (6 * x₂ - 8) = -202 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_value_l2275_227563


namespace NUMINAMATH_CALUDE_average_score_is_42_l2275_227586

/-- Intelligence contest game setup and results -/
structure ContestData where
  q1_points : ℕ := 20
  q2_points : ℕ := 25
  q3_points : ℕ := 25
  q1_correct : ℕ
  q2_correct : ℕ
  q3_correct : ℕ
  all_correct : ℕ := 1
  two_correct : ℕ := 15
  q1q2_sum : ℕ := 29
  q2q3_sum : ℕ := 20
  q1q3_sum : ℕ := 25

/-- Calculate the average score of the contest -/
def average_score (data : ContestData) : ℚ :=
  let total_participants := data.q1_correct + data.q2_correct + data.q3_correct - 2 * data.all_correct - data.two_correct
  let total_score := data.q1_correct * data.q1_points + (data.q2_correct + data.q3_correct) * data.q2_points
  (total_score : ℚ) / total_participants

/-- Theorem stating that the average score is 42 points -/
theorem average_score_is_42 (data : ContestData) 
  (h1 : data.q1_correct + data.q2_correct = data.q1q2_sum)
  (h2 : data.q2_correct + data.q3_correct = data.q2q3_sum)
  (h3 : data.q1_correct + data.q3_correct = data.q1q3_sum) :
  average_score data = 42 := by
  sorry


end NUMINAMATH_CALUDE_average_score_is_42_l2275_227586


namespace NUMINAMATH_CALUDE_hcl_mixing_theorem_l2275_227511

/-- Represents a solution with a given volume and concentration -/
structure Solution where
  volume : ℝ
  concentration : ℝ

/-- Calculates the volume of pure HCL in a solution -/
def pureHCL (s : Solution) : ℝ := s.volume * s.concentration

/-- Theorem stating the correctness of the mixing process -/
theorem hcl_mixing_theorem (sol1 sol2 final : Solution) : 
  sol1.volume = 30.0 →
  sol1.concentration = 0.1 →
  sol2.volume = 20.0 →
  sol2.concentration = 0.6 →
  final.volume = sol1.volume + sol2.volume →
  final.concentration = 0.3 →
  pureHCL sol1 + pureHCL sol2 = pureHCL final ∧
  final.volume = 50.0 := by
  sorry

#check hcl_mixing_theorem

end NUMINAMATH_CALUDE_hcl_mixing_theorem_l2275_227511


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l2275_227599

theorem students_taking_one_subject (both : ℕ) (math : ℕ) (only_science : ℕ)
  (h1 : both = 15)
  (h2 : math = 30)
  (h3 : only_science = 18) :
  math - both + only_science = 33 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l2275_227599


namespace NUMINAMATH_CALUDE_symmetric_points_fourth_quadrant_l2275_227553

/-- Given two points A(a, 3) and B(2, b) symmetric with respect to the x-axis,
    prove that point M(a, b) is in the fourth quadrant. -/
theorem symmetric_points_fourth_quadrant (a b : ℝ) :
  (a = 2 ∧ b = -3) →  -- Conditions derived from symmetry
  (a > 0 ∧ b < 0)     -- Definition of fourth quadrant
  := by sorry

end NUMINAMATH_CALUDE_symmetric_points_fourth_quadrant_l2275_227553


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2275_227527

/-- The sum of the infinite geometric series 4/3 - 5/12 + 25/144 - 125/1728 + ... -/
theorem geometric_series_sum : 
  let a : ℚ := 4/3
  let r : ℚ := -5/16
  let series_sum : ℚ := a / (1 - r)
  series_sum = 64/63 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2275_227527


namespace NUMINAMATH_CALUDE_count_multiples_of_30_l2275_227530

def smallest_square_multiple_of_30 : ℕ := 900
def smallest_cube_multiple_of_30 : ℕ := 27000

theorem count_multiples_of_30 :
  (Finset.range ((smallest_cube_multiple_of_30 - smallest_square_multiple_of_30) / 30 + 1)).card = 871 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_30_l2275_227530


namespace NUMINAMATH_CALUDE_chessboard_square_selection_l2275_227589

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Represents the number of ways to choose squares from a chessboard -/
def choose_squares (board : Chessboard) (num_squares : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to choose 60 squares from an 11x11 chessboard
    with no adjacent squares is 62 -/
theorem chessboard_square_selection :
  let board : Chessboard := ⟨11⟩
  choose_squares board 60 = 62 := by sorry

end NUMINAMATH_CALUDE_chessboard_square_selection_l2275_227589


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2275_227509

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 20 + a 21 = 10) →
  (a 22 + a 23 = 20) →
  (a 24 + a 25 = 40) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2275_227509


namespace NUMINAMATH_CALUDE_problem_solution_l2275_227555

theorem problem_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 20) : x - y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2275_227555


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2275_227564

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 55053 →
  divisor = 456 →
  quotient = 120 →
  dividend = divisor * quotient + remainder →
  remainder = 333 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2275_227564


namespace NUMINAMATH_CALUDE_all_stones_equal_weight_l2275_227540

/-- A type representing a stone with an integer weight -/
structure Stone where
  weight : ℤ

/-- A function that checks if a list of 12 stones can be split into two groups of 6 with equal weight -/
def canBalanceAny12 (stones : List Stone) : Prop :=
  stones.length = 13 ∧
  ∀ (subset : List Stone), subset.length = 12 ∧ subset.Sublist stones →
    ∃ (group1 group2 : List Stone),
      group1.length = 6 ∧ group2.length = 6 ∧
      group1.Sublist subset ∧ group2.Sublist subset ∧
      (group1.map Stone.weight).sum = (group2.map Stone.weight).sum

/-- The main theorem -/
theorem all_stones_equal_weight (stones : List Stone) :
  canBalanceAny12 stones →
  ∀ (s1 s2 : Stone), s1 ∈ stones → s2 ∈ stones → s1.weight = s2.weight :=
by sorry

end NUMINAMATH_CALUDE_all_stones_equal_weight_l2275_227540


namespace NUMINAMATH_CALUDE_quilt_block_shaded_fraction_l2275_227549

/-- Represents a quilt block -/
structure QuiltBlock where
  size : ℕ
  totalSquares : ℕ
  dividedSquares : ℕ
  shadedTrianglesPerSquare : ℕ

/-- The fraction of a quilt block that is shaded -/
def shadedFraction (q : QuiltBlock) : ℚ :=
  (q.dividedSquares * q.shadedTrianglesPerSquare : ℚ) / (2 * q.totalSquares : ℚ)

/-- Theorem: The shaded fraction of the specified quilt block is 1/8 -/
theorem quilt_block_shaded_fraction :
  let q : QuiltBlock := ⟨4, 16, 4, 1⟩
  shadedFraction q = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_quilt_block_shaded_fraction_l2275_227549


namespace NUMINAMATH_CALUDE_power_seven_mod_eight_l2275_227534

theorem power_seven_mod_eight : 7^202 % 8 = 1 := by sorry

end NUMINAMATH_CALUDE_power_seven_mod_eight_l2275_227534


namespace NUMINAMATH_CALUDE_boxes_theorem_l2275_227572

/-- Represents the operation of adding or removing balls from three consecutive boxes. -/
inductive Operation
  | Add
  | Remove

/-- Represents the state of the boxes after operations. -/
def BoxState (n : ℕ) := Fin n → ℕ

/-- Defines the initial state of the boxes. -/
def initial_state (n : ℕ) : BoxState n :=
  fun i => i.val + 1

/-- Applies an operation to three consecutive boxes. -/
def apply_operation (state : BoxState n) (start : Fin n) (op : Operation) : BoxState n :=
  sorry

/-- Checks if all boxes have exactly k balls. -/
def all_equal (state : BoxState n) (k : ℕ) : Prop :=
  ∀ i : Fin n, state i = k

/-- Main theorem: Characterizes when it's possible to achieve k balls in each box. -/
theorem boxes_theorem (n : ℕ) (h : n ≥ 3) :
  ∀ k : ℕ, k > 0 →
    (∃ (final : BoxState n),
      ∃ (ops : List (Fin n × Operation)),
        all_equal final k ∧
        final = (ops.foldl (fun st (i, op) => apply_operation st i op) (initial_state n))) ↔
    ((n % 3 = 1 ∧ k % 3 = 1) ∨ (n % 3 = 2 ∧ k % 3 = 0)) :=
  sorry

end NUMINAMATH_CALUDE_boxes_theorem_l2275_227572


namespace NUMINAMATH_CALUDE_fraction_cube_equality_l2275_227565

theorem fraction_cube_equality : (81000 ^ 3) / (27000 ^ 3) = 27 := by sorry

end NUMINAMATH_CALUDE_fraction_cube_equality_l2275_227565


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2275_227571

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x > 0, Real.exp x - a * x < 1

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(a - 1)^x) > (-(a - 1)^y)

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ a : ℝ, q a → p a) ∧ (∃ a : ℝ, p a ∧ ¬(q a)) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2275_227571


namespace NUMINAMATH_CALUDE_total_commission_is_4200_l2275_227503

def coupe_price : ℝ := 30000
def suv_price : ℝ := 2 * coupe_price
def luxury_sedan_price : ℝ := 80000
def commission_rate_coupe_suv : ℝ := 0.02
def commission_rate_luxury : ℝ := 0.03

def total_commission : ℝ :=
  coupe_price * commission_rate_coupe_suv +
  suv_price * commission_rate_coupe_suv +
  luxury_sedan_price * commission_rate_luxury

theorem total_commission_is_4200 :
  total_commission = 4200 := by sorry

end NUMINAMATH_CALUDE_total_commission_is_4200_l2275_227503


namespace NUMINAMATH_CALUDE_amp_2_neg1_4_l2275_227537

-- Define the operation &
def amp (a b c : ℝ) : ℝ := b^3 - 3*a*b*c - 4*a*c^2

-- Theorem statement
theorem amp_2_neg1_4 : amp 2 (-1) 4 = -105 := by
  sorry

end NUMINAMATH_CALUDE_amp_2_neg1_4_l2275_227537


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2275_227500

/-- Two lines are parallel if and only if their slopes are equal. -/
def parallel_lines (m1 a1 b1 m2 a2 b2 : ℝ) : Prop :=
  m1 * a2 = m2 * a1

/-- Given that the line 2x + ay + 1 = 0 is parallel to x - 4y - 1 = 0, prove that a = -8 -/
theorem parallel_lines_a_value (a : ℝ) :
  parallel_lines 2 a 1 1 (-4) (-1) → a = -8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2275_227500


namespace NUMINAMATH_CALUDE_cookie_jar_spending_l2275_227516

theorem cookie_jar_spending (initial_amount : ℝ) (final_amount : ℝ) (doris_spent : ℝ) :
  initial_amount = 24 →
  final_amount = 15 →
  initial_amount - (doris_spent + doris_spent / 2) = final_amount →
  doris_spent = 6 := by
sorry

end NUMINAMATH_CALUDE_cookie_jar_spending_l2275_227516


namespace NUMINAMATH_CALUDE_tangent_coincidence_implies_a_range_l2275_227556

/-- Piecewise function f(x) defined as x^2 + x + a for x < 0, and -1/x for x > 0 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + x + a else -1/x

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2*x + 1 else 1/x^2

theorem tangent_coincidence_implies_a_range :
  ∀ a : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ 
   f_derivative a x₁ = f_derivative a x₂ ∧
   f a x₁ - (f_derivative a x₁ * x₁) = f a x₂ - (f_derivative a x₂ * x₂)) →
  -2 < a ∧ a < 1/4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_coincidence_implies_a_range_l2275_227556


namespace NUMINAMATH_CALUDE_gcd_168_54_264_l2275_227584

theorem gcd_168_54_264 : Nat.gcd 168 (Nat.gcd 54 264) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_168_54_264_l2275_227584


namespace NUMINAMATH_CALUDE_square_not_always_positive_l2275_227545

theorem square_not_always_positive : ¬(∀ a : ℝ, a^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l2275_227545


namespace NUMINAMATH_CALUDE_vasya_has_more_placements_l2275_227582

/-- Represents a chessboard --/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a king placement on a board --/
def KingPlacement (b : Board) := Fin b.rows → Fin b.cols

/-- Predicate to check if a king placement is valid (no kings attack each other) --/
def IsValidPlacement (b : Board) (p : KingPlacement b) : Prop := sorry

/-- Number of valid king placements on a board --/
def NumValidPlacements (b : Board) (n : ℕ) : ℕ := sorry

/-- Petya's board --/
def PetyaBoard : Board := ⟨100, 50⟩

/-- Vasya's board (only white cells of a 100 × 100 checkerboard) --/
def VasyaBoard : Board := ⟨100, 50⟩

theorem vasya_has_more_placements :
  NumValidPlacements VasyaBoard 500 > NumValidPlacements PetyaBoard 500 := by
  sorry

end NUMINAMATH_CALUDE_vasya_has_more_placements_l2275_227582
