import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2284_228492

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + 2*m = 0) → m ≤ 9/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2284_228492


namespace NUMINAMATH_CALUDE_equation_solutions_l2284_228464

theorem equation_solutions :
  (∀ x y : ℤ, y^4 + 2*x^4 + 1 = 4*x^2*y ↔ (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = 1)) ∧
  (∀ x y z : ℕ+, 5*(x*y + y*z + z*x) = 4*x*y*z ↔
    ((x = 5 ∧ y = 10 ∧ z = 2) ∨ (x = 5 ∧ y = 2 ∧ z = 10) ∨
     (x = 10 ∧ y = 5 ∧ z = 2) ∨ (x = 10 ∧ y = 2 ∧ z = 5) ∨
     (x = 2 ∧ y = 10 ∧ z = 5) ∨ (x = 2 ∧ y = 5 ∧ z = 10) ∨
     (x = 4 ∧ y = 20 ∧ z = 2) ∨ (x = 4 ∧ y = 2 ∧ z = 20) ∨
     (x = 2 ∧ y = 4 ∧ z = 20) ∨ (x = 2 ∧ y = 20 ∧ z = 4) ∨
     (x = 20 ∧ y = 2 ∧ z = 4) ∨ (x = 20 ∧ y = 4 ∧ z = 2))) := by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l2284_228464


namespace NUMINAMATH_CALUDE_intramural_teams_l2284_228418

theorem intramural_teams (num_boys : ℕ) (num_girls : ℕ) (max_teams : ℕ) :
  num_boys = 32 →
  max_teams = 8 →
  (∃ (boys_per_team : ℕ), num_boys = max_teams * boys_per_team) →
  (∃ (girls_per_team : ℕ), num_girls = max_teams * girls_per_team) →
  ∃ (k : ℕ), num_girls = 8 * k :=
by sorry

end NUMINAMATH_CALUDE_intramural_teams_l2284_228418


namespace NUMINAMATH_CALUDE_exchange_rates_problem_l2284_228484

theorem exchange_rates_problem (drum wife leopard_skin : ℕ) : 
  (2 * drum + 3 * wife + leopard_skin = 111) →
  (3 * drum + 4 * wife = 2 * leopard_skin + 8) →
  (leopard_skin % 2 = 0) →
  (drum = 20 ∧ wife = 9 ∧ leopard_skin = 44) := by
  sorry

end NUMINAMATH_CALUDE_exchange_rates_problem_l2284_228484


namespace NUMINAMATH_CALUDE_max_subset_with_distinct_sums_l2284_228493

def S (A : Finset ℕ) : Finset ℕ :=
  Finset.powerset A \ {∅} |>.image (λ B => B.sum id)

theorem max_subset_with_distinct_sums :
  (∃ (A : Finset ℕ), A ⊆ Finset.range 16 ∧ A.card = 5 ∧ S A.toSet.toFinset = S A) ∧
  ¬(∃ (A : Finset ℕ), A ⊆ Finset.range 16 ∧ A.card = 6 ∧ S A.toSet.toFinset = S A) :=
sorry

end NUMINAMATH_CALUDE_max_subset_with_distinct_sums_l2284_228493


namespace NUMINAMATH_CALUDE_car_journey_distance_l2284_228423

theorem car_journey_distance :
  ∀ (v : ℝ),
  v > 0 →
  v * 7 = (v + 12) * 5 →
  v * 7 = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_car_journey_distance_l2284_228423


namespace NUMINAMATH_CALUDE_max_value_of_f_l2284_228456

noncomputable def f (t : ℝ) : ℝ := (3^t - 4*t)*t / 9^t

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1/16 ∧ ∀ (t : ℝ), f t ≤ M ∧ ∃ (t₀ : ℝ), f t₀ = M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2284_228456


namespace NUMINAMATH_CALUDE_negation_equivalence_l2284_228495

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define a function to get the unit digit of a number
def unitDigit (n : ℕ) : ℕ := n % 10

-- The original proposition
def originalProposition : Prop :=
  ∀ x ∈ PositiveIntegers, unitDigit (x^2) ≠ 2

-- The negation of the original proposition
def negationProposition : Prop :=
  ∃ x ∈ PositiveIntegers, unitDigit (x^2) = 2

-- The theorem to prove
theorem negation_equivalence : ¬originalProposition ↔ negationProposition :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2284_228495


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2284_228473

theorem complex_number_in_second_quadrant : 
  let z : ℂ := 2 * I / (1 - I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2284_228473


namespace NUMINAMATH_CALUDE_picture_area_l2284_228490

theorem picture_area (x y : ℤ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : (3*x + 2)*(y + 4) - x*y = 62) : 
  x * y = 10 := by
sorry

end NUMINAMATH_CALUDE_picture_area_l2284_228490


namespace NUMINAMATH_CALUDE_num_parallelepipeds_is_29_l2284_228409

/-- A set of 4 points in 3D space -/
structure PointSet :=
  (points : Fin 4 → ℝ × ℝ × ℝ)
  (not_coplanar : ∀ (p : ℝ × ℝ × ℝ → ℝ), ¬(∀ i, p (points i) = 0))

/-- A parallelepiped formed by 4 vertices -/
structure Parallelepiped :=
  (vertices : Fin 8 → ℝ × ℝ × ℝ)

/-- The number of distinct parallelepipeds that can be formed from a set of 4 points -/
def num_parallelepipeds (ps : PointSet) : ℕ :=
  -- Definition here (not implemented)
  0

/-- Theorem: The number of distinct parallelepipeds formed by 4 non-coplanar points is 29 -/
theorem num_parallelepipeds_is_29 (ps : PointSet) : num_parallelepipeds ps = 29 := by
  sorry

end NUMINAMATH_CALUDE_num_parallelepipeds_is_29_l2284_228409


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l2284_228431

def purchase_price : ℕ := 12000
def transportation_charges : ℕ := 1000
def selling_price : ℕ := 27000
def profit_percentage : ℚ := 1/2

theorem repair_cost_calculation :
  ∃ (repair_cost : ℕ),
    selling_price = (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage) ∧
    repair_cost = 5000 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l2284_228431


namespace NUMINAMATH_CALUDE_furniture_cost_price_sum_l2284_228476

theorem furniture_cost_price_sum (sp1 sp2 sp3 sp4 : ℕ) 
  (h1 : sp1 = 3000) (h2 : sp2 = 2400) (h3 : sp3 = 12000) (h4 : sp4 = 18000) : 
  (sp1 / 120 * 100 + sp2 / 120 * 100 + sp3 / 120 * 100 + sp4 / 120 * 100 : ℕ) = 29500 := by
  sorry

#check furniture_cost_price_sum

end NUMINAMATH_CALUDE_furniture_cost_price_sum_l2284_228476


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2284_228429

theorem pure_imaginary_condition (a : ℝ) : 
  let z : ℂ := (1 + Complex.I) / (1 + a * Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2284_228429


namespace NUMINAMATH_CALUDE_abc_fraction_value_l2284_228485

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 1 / 3)
  (h2 : b * c / (b + c) = 1 / 4)
  (h3 : a * c / (c + a) = 1 / 5) :
  24 * a * b * c / (a * b + b * c + c * a) = 4 := by
sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l2284_228485


namespace NUMINAMATH_CALUDE_total_population_of_two_villages_l2284_228404

/-- Given two villages A and B with the following properties:
    - 90% of Village A's population is 23040
    - 80% of Village B's population is 17280
    - Village A has three times as many children as Village B
    - The adult population is equally distributed between the two villages
    Prove that the total population of both villages combined is 47,200 -/
theorem total_population_of_two_villages :
  ∀ (population_A population_B children_A children_B : ℕ),
    (population_A : ℚ) * (9 / 10) = 23040 →
    (population_B : ℚ) * (4 / 5) = 17280 →
    children_A = 3 * children_B →
    population_A - children_A = population_B - children_B →
    population_A + population_B = 47200 := by
  sorry

#eval 47200

end NUMINAMATH_CALUDE_total_population_of_two_villages_l2284_228404


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2284_228498

/-- The first term of the geometric series -/
def a₁ : ℚ := 7/8

/-- The second term of the geometric series -/
def a₂ : ℚ := -21/32

/-- The third term of the geometric series -/
def a₃ : ℚ := 63/128

/-- The common ratio of the geometric series -/
def r : ℚ := -3/4

theorem geometric_series_common_ratio :
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2284_228498


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l2284_228424

def digit_product (n : ℕ) : ℕ := 
  if n = 0 then 1 else (n % 10) * digit_product (n / 10)

def digit_sum (n : ℕ) : ℕ := 
  if n = 0 then 0 else (n % 10) + digit_sum (n / 10)

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n

theorem unique_number_satisfying_conditions : 
  ∃! x : ℕ, digit_product x = 44 * x - 86868 ∧ is_cube (digit_sum x) ∧ x = 1989 :=
sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l2284_228424


namespace NUMINAMATH_CALUDE_fitness_center_membership_ratio_l2284_228470

theorem fitness_center_membership_ratio :
  ∀ (f m c : ℕ), 
  (f > 0) → (m > 0) → (c > 0) →
  (35 * f + 30 * m + 10 * c : ℝ) / (f + m + c : ℝ) = 25 →
  ∃ (k : ℕ), f = 3 * k ∧ m = 6 * k ∧ c = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_fitness_center_membership_ratio_l2284_228470


namespace NUMINAMATH_CALUDE_max_expected_expenditure_l2284_228443

-- Define the parameters of the linear regression equation
def b : ℝ := 0.8
def a : ℝ := 2

-- Define the revenue
def revenue : ℝ := 10

-- Define the error bound
def error_bound : ℝ := 0.5

-- Theorem statement
theorem max_expected_expenditure :
  ∀ e : ℝ, |e| < error_bound →
  ∃ y : ℝ, y = b * revenue + a + e ∧ y ≤ 10.5 ∧
  ∀ y' : ℝ, (∃ e' : ℝ, |e'| < error_bound ∧ y' = b * revenue + a + e') → y' ≤ y :=
by sorry

end NUMINAMATH_CALUDE_max_expected_expenditure_l2284_228443


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2284_228496

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2284_228496


namespace NUMINAMATH_CALUDE_same_color_probability_l2284_228469

/-- The probability of drawing 2 balls of the same color from a bag -/
theorem same_color_probability (total : ℕ) (red : ℕ) (yellow : ℕ) (blue : ℕ) :
  total = red + yellow + blue →
  red = 3 →
  yellow = 2 →
  blue = 1 →
  (Nat.choose red 2 + Nat.choose yellow 2) / Nat.choose total 2 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2284_228469


namespace NUMINAMATH_CALUDE_circle_diameter_endpoints_l2284_228406

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- Checks if two points are endpoints of a diameter in a given circle -/
def areDiameterEndpoints (c : Circle) (p1 p2 : Point2D) : Prop :=
  (p1.x - c.center.x)^2 + (p1.y - c.center.y)^2 = c.radius^2 ∧
  (p2.x - c.center.x)^2 + (p2.y - c.center.y)^2 = c.radius^2 ∧
  (p1.x + p2.x) / 2 = c.center.x ∧
  (p1.y + p2.y) / 2 = c.center.y

theorem circle_diameter_endpoints :
  let c : Circle := { center := { x := 1, y := 2 }, radius := Real.sqrt 13 }
  let p1 : Point2D := { x := 3, y := -1 }
  let p2 : Point2D := { x := -1, y := 5 }
  areDiameterEndpoints c p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoints_l2284_228406


namespace NUMINAMATH_CALUDE_project_rotation_lcm_l2284_228489

theorem project_rotation_lcm : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 12)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_project_rotation_lcm_l2284_228489


namespace NUMINAMATH_CALUDE_intersection_properties_l2284_228494

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (2, -1)

-- Define the intersection points M and N (existence assumed)
axiom exists_intersection_points : ∃ (M N : ℝ × ℝ), 
  curve_C M.1 M.2 ∧ line_l M.1 M.2 ∧
  curve_C N.1 N.2 ∧ line_l N.1 N.2 ∧
  M ≠ N

-- State the theorem
theorem intersection_properties :
  ∃ (M N : ℝ × ℝ), 
    curve_C M.1 M.2 ∧ line_l M.1 M.2 ∧
    curve_C N.1 N.2 ∧ line_l N.1 N.2 ∧
    M ≠ N ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 8 ∧
    Real.sqrt ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2) *
    Real.sqrt ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2) = 14 := by
  sorry


end NUMINAMATH_CALUDE_intersection_properties_l2284_228494


namespace NUMINAMATH_CALUDE_isabella_currency_exchange_l2284_228446

theorem isabella_currency_exchange :
  ∃ d : ℕ+, 
    (8 * d.val : ℚ) / 5 - 80 = d.val ∧ 
    (d.val / 100 + (d.val % 100) / 10 + d.val % 10 = 9) := by
  sorry

end NUMINAMATH_CALUDE_isabella_currency_exchange_l2284_228446


namespace NUMINAMATH_CALUDE_fraction_equality_l2284_228401

theorem fraction_equality (a b : ℝ) (h : a / b = 1 / 2) : a / (a + b) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2284_228401


namespace NUMINAMATH_CALUDE_choir_group_calculation_l2284_228481

theorem choir_group_calculation (total_members : ℕ) (group1 group2 group3 : ℕ) (absent : ℕ) :
  total_members = 162 →
  group1 = 22 →
  group2 = 33 →
  group3 = 36 →
  absent = 7 →
  ∃ (group4 group5 : ℕ),
    group4 = group2 - 3 ∧
    group5 = total_members - absent - (group1 + group2 + group3 + group4) ∧
    group5 = 34 :=
by sorry

end NUMINAMATH_CALUDE_choir_group_calculation_l2284_228481


namespace NUMINAMATH_CALUDE_difference_of_squares_l2284_228488

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2284_228488


namespace NUMINAMATH_CALUDE_number_of_factors_of_n_l2284_228412

def n : ℕ := 2^5 * 3^4 * 5^6 * 6^3

theorem number_of_factors_of_n : (Finset.card (Nat.divisors n)) = 504 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_n_l2284_228412


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l2284_228467

theorem lcm_factor_problem (A B : ℕ+) : 
  Nat.gcd A B = 10 →
  A = 150 →
  11 ∣ Nat.lcm A B →
  Nat.lcm A B = 10 * 11 * 15 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l2284_228467


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l2284_228416

theorem right_triangle_segment_ratio 
  (a b c r s : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_perp : r + s = c) 
  (h_r : r * c = a^2) 
  (h_s : s * c = b^2) 
  (h_ratio : a / b = 2 / 5) : 
  r / s = 4 / 25 := by 
sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l2284_228416


namespace NUMINAMATH_CALUDE_complement_B_equals_M_intersection_A_B_l2284_228437

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A (a : ℝ) : Set ℝ := {x | x^2 + (a-1)*x - a > 0}

-- Define set B
def B (a b : ℝ) : Set ℝ := {x | (x+a)*(x+b) > 0}

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem 1
theorem complement_B_equals_M (a b : ℝ) (h : a ≠ b) :
  (U \ B a b = M) ↔ ((a = -3 ∧ b = 1) ∨ (a = 1 ∧ b = -3)) :=
sorry

-- Theorem 2
theorem intersection_A_B (a b : ℝ) (h : -1 < b ∧ b < a ∧ a < 1) :
  A a ∩ B a b = {x | x < -a ∨ x > 1} :=
sorry

end NUMINAMATH_CALUDE_complement_B_equals_M_intersection_A_B_l2284_228437


namespace NUMINAMATH_CALUDE_interest_difference_l2284_228435

/-- Calculate the difference between compound interest and simple interest -/
theorem interest_difference (principal : ℝ) (rate : ℝ) (time : ℝ) (compounding_frequency : ℕ) :
  principal = 1200 →
  rate = 0.1 →
  time = 1 →
  compounding_frequency = 2 →
  let simple_interest := principal * rate * time
  let compound_interest := principal * ((1 + rate / compounding_frequency) ^ (compounding_frequency * time) - 1)
  compound_interest - simple_interest = 3 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l2284_228435


namespace NUMINAMATH_CALUDE_sandwich_jam_cost_l2284_228413

theorem sandwich_jam_cost (N B J : ℕ) (h1 : N > 1) (h2 : N * (3 * B + 7 * J) = 276) : 
  (N * J * 7 : ℚ) / 100 = 0.14 * J := by
  sorry

end NUMINAMATH_CALUDE_sandwich_jam_cost_l2284_228413


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l2284_228422

theorem not_necessarily_right_triangle (a b c : ℝ) (h1 : a = 3^2) (h2 : b = 4^2) (h3 : c = 5^2) :
  ¬ (a^2 + b^2 = c^2) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l2284_228422


namespace NUMINAMATH_CALUDE_clown_mobile_distribution_l2284_228479

theorem clown_mobile_distribution (total_clowns : ℕ) (num_mobiles : ℕ) (clowns_per_mobile : ℕ) :
  total_clowns = 140 →
  num_mobiles = 5 →
  total_clowns = num_mobiles * clowns_per_mobile →
  clowns_per_mobile = 28 := by
  sorry

end NUMINAMATH_CALUDE_clown_mobile_distribution_l2284_228479


namespace NUMINAMATH_CALUDE_amc10_participation_increase_l2284_228415

def participation : Fin 6 → ℕ
  | 0 => 50  -- 2010
  | 1 => 56  -- 2011
  | 2 => 62  -- 2012
  | 3 => 68  -- 2013
  | 4 => 77  -- 2014
  | 5 => 81  -- 2015

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreaseBetween2013And2014 : Prop :=
  ∀ i : Fin 5, percentageIncrease (participation i) (participation (i + 1)) ≤
    percentageIncrease (participation 3) (participation 4)

theorem amc10_participation_increase : largestIncreaseBetween2013And2014 := by
  sorry

end NUMINAMATH_CALUDE_amc10_participation_increase_l2284_228415


namespace NUMINAMATH_CALUDE_mathematics_letter_probability_l2284_228471

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters : ℕ := 8
  let probability : ℚ := unique_letters / alphabet_size
  probability = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_mathematics_letter_probability_l2284_228471


namespace NUMINAMATH_CALUDE_sequence_end_point_sequence_end_point_proof_l2284_228405

theorem sequence_end_point : ℕ → Prop :=
  fun n =>
    (∃ k : ℕ, 9 * k ≥ 10 ∧ 9 * (k + 11109) = n) →
    n = 99999

-- The proof is omitted
theorem sequence_end_point_proof : sequence_end_point 99999 := by
  sorry

end NUMINAMATH_CALUDE_sequence_end_point_sequence_end_point_proof_l2284_228405


namespace NUMINAMATH_CALUDE_quadratic_other_x_intercept_l2284_228434

/-- Given a quadratic function f(x) = ax^2 + bx + c with vertex (5, 10) and
    one x-intercept at (1, 0), the x-coordinate of the other x-intercept is 9. -/
theorem quadratic_other_x_intercept
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h_quad : ∀ x, f x = a * x^2 + b * x + c)
  (h_vertex : f 5 = 10 ∧ ∀ x, f x ≤ f 5)
  (h_intercept : f 1 = 0) :
  ∃ x, x ≠ 1 ∧ f x = 0 ∧ x = 9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_other_x_intercept_l2284_228434


namespace NUMINAMATH_CALUDE_range_of_a_l2284_228459

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a * x + 2 - 3 * a else 2^x - 1

theorem range_of_a :
  ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = f a x₂) ↔ a < 2/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2284_228459


namespace NUMINAMATH_CALUDE_celenes_borrowed_books_l2284_228427

/-- Represents the problem of determining the number of books Celine borrowed -/
theorem celenes_borrowed_books :
  let daily_charge : ℚ := 0.5
  let days_for_first_book : ℕ := 20
  let days_in_may : ℕ := 31
  let total_paid : ℚ := 41
  let num_books_whole_month : ℕ := 2

  let charge_first_book : ℚ := daily_charge * days_for_first_book
  let charge_per_whole_month_book : ℚ := daily_charge * days_in_may
  let charge_whole_month_books : ℚ := num_books_whole_month * charge_per_whole_month_book
  let total_charge : ℚ := charge_first_book + charge_whole_month_books

  total_charge = total_paid →
  num_books_whole_month + 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_celenes_borrowed_books_l2284_228427


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2284_228441

theorem complex_power_magnitude : 
  Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2284_228441


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2284_228411

theorem right_triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a^2 + c^2 = b^2 →
  Real.arctan (a / (c + b)) + Real.arctan (c / (a + b)) = π / 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l2284_228411


namespace NUMINAMATH_CALUDE_complex_roots_problem_l2284_228475

theorem complex_roots_problem (p q r : ℂ) : 
  p + q + r = 1 → p * q * r = 1 → p * q + p * r + q * r = 0 →
  (∃ (σ : Equiv.Perm (Fin 3)), 
    σ.1 0 = p ∧ σ.1 1 = q ∧ σ.1 2 = r ∧
    (∀ x, x^3 - x^2 - 1 = 0 ↔ (x = 2 ∨ x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2))) :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_problem_l2284_228475


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2284_228440

theorem necessary_not_sufficient (a b : ℝ) : 
  (∃ b, a ≠ 0 ∧ a * b = 0) ∧ (a * b ≠ 0 → a ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2284_228440


namespace NUMINAMATH_CALUDE_inclined_plane_friction_l2284_228478

/-- The coefficient of friction between a block and an inclined plane -/
theorem inclined_plane_friction (P F_up F_down : ℝ) (α : ℝ) (μ : ℝ) :
  F_up = 3 * F_down →
  F_up + F_down = P →
  F_up = P * Real.sin α + μ * P * Real.cos α →
  F_down = P * Real.sin α - μ * P * Real.cos α →
  μ = Real.sqrt 3 / 6 := by
sorry

end NUMINAMATH_CALUDE_inclined_plane_friction_l2284_228478


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2284_228450

theorem quadratic_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + 2*x + a = 0 ∧ y^2 + 2*y + a = 0) → a < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2284_228450


namespace NUMINAMATH_CALUDE_algebraic_identity_l2284_228491

theorem algebraic_identity (x y : ℝ) : -y^2 * x + x * y^2 = 0 := by sorry

end NUMINAMATH_CALUDE_algebraic_identity_l2284_228491


namespace NUMINAMATH_CALUDE_sum_of_mixed_numbers_l2284_228402

theorem sum_of_mixed_numbers : 
  (3 + 1/6 : ℚ) + (4 + 2/3 : ℚ) + (6 + 1/18 : ℚ) = 13 + 8/9 := by sorry

end NUMINAMATH_CALUDE_sum_of_mixed_numbers_l2284_228402


namespace NUMINAMATH_CALUDE_equation_solution_l2284_228420

theorem equation_solution (x : ℝ) : 
  (x / 6) / 3 = 6 / (x / 3) → x = 18 ∨ x = -18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2284_228420


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2284_228463

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2284_228463


namespace NUMINAMATH_CALUDE_proportional_expression_l2284_228474

/-- Given that y is directly proportional to x-2 and y = -4 when x = 3,
    prove that the analytical expression of y with respect to x is y = -4x + 8 -/
theorem proportional_expression (x y : ℝ) :
  (∃ k : ℝ, ∀ x, y = k * (x - 2)) →  -- y is directly proportional to x-2
  (3 : ℝ) = x → (-4 : ℝ) = y →       -- when x = 3, y = -4
  y = -4 * x + 8 :=                   -- the analytical expression
by sorry

end NUMINAMATH_CALUDE_proportional_expression_l2284_228474


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2284_228419

/-- The axis of symmetry of a parabola y=(x-h)^2 is x=h -/
theorem parabola_axis_of_symmetry (h : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (x - h)^2
  ∃! a : ℝ, ∀ x y : ℝ, f x = f y → (x + y) / 2 = a :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2284_228419


namespace NUMINAMATH_CALUDE_siyeon_distance_l2284_228425

-- Define the given constants
def giljun_distance : ℝ := 1.05
def difference : ℝ := 0.46

-- Define the theorem
theorem siyeon_distance :
  let giljun_km := giljun_distance
  let diff_km := difference
  let siyeon_km := giljun_km - diff_km
  siyeon_km = 0.59 := by sorry

end NUMINAMATH_CALUDE_siyeon_distance_l2284_228425


namespace NUMINAMATH_CALUDE_cube_volume_increase_l2284_228414

theorem cube_volume_increase (s : ℝ) (h : s > 0) : 
  let original_volume := s^3
  let new_edge_length := 1.4 * s
  let new_volume := new_edge_length^3
  (new_volume - original_volume) / original_volume * 100 = 174.4 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_increase_l2284_228414


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l2284_228452

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_five : opposite (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l2284_228452


namespace NUMINAMATH_CALUDE_circle_plus_92_composed_thrice_l2284_228458

def circle_plus (N : ℝ) : ℝ := 0.75 * N + 2

theorem circle_plus_92_composed_thrice :
  circle_plus (circle_plus (circle_plus 92)) = 43.4375 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_92_composed_thrice_l2284_228458


namespace NUMINAMATH_CALUDE_flagpole_break_height_l2284_228457

theorem flagpole_break_height :
  let initial_height : ℝ := 10
  let horizontal_distance : ℝ := 3
  let break_height : ℝ := Real.sqrt 109 / 2
  (break_height^2 + horizontal_distance^2 = (initial_height - break_height)^2) ∧
  (2 * break_height = Real.sqrt (horizontal_distance^2 + initial_height^2)) :=
by sorry

end NUMINAMATH_CALUDE_flagpole_break_height_l2284_228457


namespace NUMINAMATH_CALUDE_cosine_function_vertical_shift_l2284_228461

theorem cosine_function_vertical_shift 
  (a b c d : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : d > 0) 
  (h5 : ∀ x, 2 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 4) 
  (h6 : Real.cos c = 1) : 
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_cosine_function_vertical_shift_l2284_228461


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l2284_228433

-- Define the concept of opposite for real numbers
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_negative_five : opposite (-5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l2284_228433


namespace NUMINAMATH_CALUDE_makeup_set_cost_l2284_228468

theorem makeup_set_cost (initial_money mom_contribution additional_needed : ℕ) :
  initial_money = 35 →
  mom_contribution = 20 →
  additional_needed = 10 →
  initial_money + mom_contribution + additional_needed = 65 := by
  sorry

end NUMINAMATH_CALUDE_makeup_set_cost_l2284_228468


namespace NUMINAMATH_CALUDE_system_solution_l2284_228465

theorem system_solution :
  ∀ (x y : ℝ),
    (x * y^2 = 15 * x^2 + 17 * x * y + 15 * y^2) ∧
    (x^2 * y = 20 * x^2 + 3 * y^2) →
    ((x = 0 ∧ y = 0) ∨ (x = -19 ∧ y = -2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2284_228465


namespace NUMINAMATH_CALUDE_parabola_directrix_l2284_228462

/-- Given a parabola with equation y² = 6x, its directrix equation is x = -3/2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y^2 = 6*x) → (∃ (k : ℝ), k = -3/2 ∧ x = k) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2284_228462


namespace NUMINAMATH_CALUDE_books_left_to_read_l2284_228428

theorem books_left_to_read 
  (total_books : ℕ) 
  (mcgregor_books : ℕ) 
  (floyd_books : ℕ) 
  (h1 : total_books = 89) 
  (h2 : mcgregor_books = 34) 
  (h3 : floyd_books = 32) : 
  total_books - (mcgregor_books + floyd_books) = 23 := by
sorry

end NUMINAMATH_CALUDE_books_left_to_read_l2284_228428


namespace NUMINAMATH_CALUDE_person_age_in_1930_l2284_228439

theorem person_age_in_1930 (birth_year : ℕ) (death_year : ℕ) (age_at_death : ℕ) :
  (birth_year ≤ 1930) →
  (death_year > 1930) →
  (age_at_death = death_year - birth_year) →
  (age_at_death = birth_year / 31) →
  (1930 - birth_year = 39) :=
by sorry

end NUMINAMATH_CALUDE_person_age_in_1930_l2284_228439


namespace NUMINAMATH_CALUDE_denny_initial_followers_l2284_228417

/-- Calculates the initial number of followers given the daily increase, total unfollows, 
    final follower count, and number of days in a year. -/
def initial_followers (daily_increase : ℕ) (total_unfollows : ℕ) (final_count : ℕ) (days_in_year : ℕ) : ℕ :=
  final_count - (daily_increase * days_in_year) + total_unfollows

/-- Proves that given the specified conditions, the initial number of followers is 100000. -/
theorem denny_initial_followers : 
  initial_followers 1000 20000 445000 365 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_denny_initial_followers_l2284_228417


namespace NUMINAMATH_CALUDE_donut_distribution_l2284_228483

/-- The number of ways to distribute n items among k categories,
    with at least one item in each category. -/
def distribute_with_minimum (n k : ℕ) : ℕ :=
  Nat.choose (n - k + (k - 1)) (k - 1)

/-- Theorem: There are 35 ways to distribute 8 donuts among 4 types,
    with at least one donut of each type. -/
theorem donut_distribution : distribute_with_minimum 8 4 = 35 := by
  sorry

#eval distribute_with_minimum 8 4

end NUMINAMATH_CALUDE_donut_distribution_l2284_228483


namespace NUMINAMATH_CALUDE_calculation_proof_l2284_228480

theorem calculation_proof : 6 * (-1/2) + Real.sqrt 3 * Real.sqrt 8 + (-15)^0 = 2 * Real.sqrt 6 - 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2284_228480


namespace NUMINAMATH_CALUDE_world_expo_arrangements_l2284_228430

theorem world_expo_arrangements (n : ℕ) (k : ℕ) :
  n = 7 → k = 3 → (n.choose k) * ((n - k).choose k) = 140 := by
  sorry

end NUMINAMATH_CALUDE_world_expo_arrangements_l2284_228430


namespace NUMINAMATH_CALUDE_equation_solution_l2284_228447

theorem equation_solution (x y z : ℚ) 
  (eq1 : 3 * x - 2 * y - 3 * z = 0)
  (eq2 : x + 5 * y - 12 * z = 0)
  (z_neq_0 : z ≠ 0) :
  (x^2 - 2*x*y) / (y^2 + 2*z^2) = -1053/1547 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2284_228447


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l2284_228487

/-- A coloring function for points in ℚ × ℚ -/
def Coloring := ℚ × ℚ → Fin 2

/-- The distance between two points in ℚ × ℚ -/
def distance (p q : ℚ × ℚ) : ℚ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt

/-- A valid coloring function assigns different colors to points with distance 1 -/
def is_valid_coloring (f : Coloring) : Prop :=
  ∀ p q : ℚ × ℚ, distance p q = 1 → f p ≠ f q

theorem exists_valid_coloring : ∃ f : Coloring, is_valid_coloring f := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l2284_228487


namespace NUMINAMATH_CALUDE_benny_cards_l2284_228442

theorem benny_cards (added_cards : ℕ) (remaining_cards : ℕ) : 
  added_cards = 4 →
  remaining_cards = 34 →
  ∃ (initial_cards : ℕ),
    initial_cards + added_cards = 2 * remaining_cards ∧
    initial_cards = 64 := by
  sorry

end NUMINAMATH_CALUDE_benny_cards_l2284_228442


namespace NUMINAMATH_CALUDE_magnets_ratio_l2284_228408

theorem magnets_ratio (adam_initial : ℕ) (peter : ℕ) (adam_final : ℕ) : 
  adam_initial = 18 →
  peter = 24 →
  adam_final = peter / 2 →
  adam_final = adam_initial - (adam_initial - adam_final) →
  (adam_initial - adam_final : ℚ) / adam_initial = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_magnets_ratio_l2284_228408


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l2284_228477

/-- Represents an arithmetic sequence of three real numbers. -/
structure ArithmeticSequence (α : Type*) [LinearOrderedField α] where
  p : α
  q : α
  r : α
  is_arithmetic : q - r = p - q
  decreasing : p ≥ q ∧ q ≥ r
  nonnegative : r ≥ 0

/-- The theorem stating the properties of the quadratic equation and its root. -/
theorem quadratic_root_theorem (α : Type*) [LinearOrderedField α] 
  (seq : ArithmeticSequence α) : 
  (∃ x y : α, x = 2 * y ∧ 
   seq.p * x^2 + seq.q * x + seq.r = 0 ∧ 
   seq.p * y^2 + seq.q * y + seq.r = 0) → 
  (∃ y : α, y = -1/6 ∧ seq.p * y^2 + seq.q * y + seq.r = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l2284_228477


namespace NUMINAMATH_CALUDE_town_population_l2284_228436

theorem town_population (growth_rate : ℝ) (future_population : ℕ) (present_population : ℕ) :
  growth_rate = 0.1 →
  future_population = 264 →
  present_population * (1 + growth_rate) = future_population →
  present_population = 240 := by
sorry

end NUMINAMATH_CALUDE_town_population_l2284_228436


namespace NUMINAMATH_CALUDE_advertisement_revenue_l2284_228448

/-- Calculates the revenue from advertisements for a college football program -/
theorem advertisement_revenue
  (production_cost : ℚ)
  (num_programs : ℕ)
  (selling_price : ℚ)
  (desired_profit : ℚ)
  (h1 : production_cost = 70/100)
  (h2 : num_programs = 35000)
  (h3 : selling_price = 50/100)
  (h4 : desired_profit = 8000) :
  production_cost * num_programs + desired_profit - selling_price * num_programs = 15000 :=
by sorry

end NUMINAMATH_CALUDE_advertisement_revenue_l2284_228448


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2284_228497

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2284_228497


namespace NUMINAMATH_CALUDE_class_a_win_probability_class_b_score_expectation_l2284_228410

/-- Represents the result of a single event --/
inductive EventResult
| Win
| Lose

/-- Represents the outcome of the three events for a class --/
structure ClassOutcome :=
  (event1 : EventResult)
  (event2 : EventResult)
  (event3 : EventResult)

/-- Calculates the score for a given ClassOutcome --/
def score (outcome : ClassOutcome) : Int :=
  let e1 := match outcome.event1 with
    | EventResult.Win => 2
    | EventResult.Lose => -1
  let e2 := match outcome.event2 with
    | EventResult.Win => 2
    | EventResult.Lose => -1
  let e3 := match outcome.event3 with
    | EventResult.Win => 2
    | EventResult.Lose => -1
  e1 + e2 + e3

/-- Probabilities of Class A winning each event --/
def probA1 : Float := 0.4
def probA2 : Float := 0.5
def probA3 : Float := 0.8

/-- Theorem stating the probability of Class A winning the championship --/
theorem class_a_win_probability :
  let p := probA1 * probA2 * probA3 +
           (1 - probA1) * probA2 * probA3 +
           probA1 * (1 - probA2) * probA3 +
           probA1 * probA2 * (1 - probA3)
  p = 0.6 := by sorry

/-- Theorem stating the expectation of Class B's total score --/
theorem class_b_score_expectation :
  let p_neg3 := probA1 * probA2 * probA3
  let p_0 := (1 - probA1) * probA2 * probA3 + probA1 * (1 - probA2) * probA3 + probA1 * probA2 * (1 - probA3)
  let p_3 := (1 - probA1) * (1 - probA2) * probA3 + (1 - probA1) * probA2 * (1 - probA3) + probA1 * (1 - probA2) * (1 - probA3)
  let p_6 := (1 - probA1) * (1 - probA2) * (1 - probA3)
  let expectation := -3 * p_neg3 + 0 * p_0 + 3 * p_3 + 6 * p_6
  expectation = 0.9 := by sorry

end NUMINAMATH_CALUDE_class_a_win_probability_class_b_score_expectation_l2284_228410


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2284_228407

theorem largest_prime_factor_of_expression : 
  let n : ℤ := 20^3 + 15^4 - 10^5 + 5^6
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n.natAbs ∧ p = 103 ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ n.natAbs → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2284_228407


namespace NUMINAMATH_CALUDE_selection_theorem_l2284_228421

/-- The number of ways to select 3 people from 11, with at least one of A or B selected and C not selected -/
def selection_ways (n : ℕ) (k : ℕ) (total : ℕ) : ℕ :=
  (2 * Nat.choose (total - 3) (k - 1)) + Nat.choose (total - 3) (k - 2)

/-- Theorem stating that the number of selection ways is 64 -/
theorem selection_theorem : selection_ways 3 3 11 = 64 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l2284_228421


namespace NUMINAMATH_CALUDE_sqrt_neg_two_squared_l2284_228432

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_two_squared_l2284_228432


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l2284_228400

theorem sum_of_fractions_inequality (x y z : ℝ) 
  (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
  (1 + x^2) / (1 + y + z^2) + 
  (1 + y^2) / (1 + z + x^2) + 
  (1 + z^2) / (1 + x + y^2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l2284_228400


namespace NUMINAMATH_CALUDE_even_rows_in_pascal_triangle_l2284_228444

/-- Pascal's triangle row -/
def pascal_row (n : ℕ) : List ℕ := sorry

/-- Check if a row (excluding endpoints) consists of only even numbers -/
def is_even_row (row : List ℕ) : Bool := sorry

/-- Count of even rows in first n rows of Pascal's triangle (excluding row 0 and 1) -/
def count_even_rows (n : ℕ) : ℕ := sorry

/-- Theorem: There are exactly 4 even rows in the first 30 rows of Pascal's triangle (excluding row 0 and 1) -/
theorem even_rows_in_pascal_triangle : count_even_rows 30 = 4 := by sorry

end NUMINAMATH_CALUDE_even_rows_in_pascal_triangle_l2284_228444


namespace NUMINAMATH_CALUDE_outfits_count_l2284_228499

/-- The number of different outfits that can be made with a given number of shirts, pants, and ties. -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (ties : ℕ) : ℕ :=
  shirts * pants * (ties + 1)

/-- Theorem stating that the number of outfits with 7 shirts, 5 pants, and 4 ties (plus the option of no tie) is 175. -/
theorem outfits_count : number_of_outfits 7 5 4 = 175 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l2284_228499


namespace NUMINAMATH_CALUDE_expression_equality_l2284_228454

theorem expression_equality : (2 + Real.sqrt 3) ^ 0 + 3 * Real.tan (π / 6) - |Real.sqrt 3 - 2| + (1 / 2)⁻¹ = 1 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2284_228454


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l2284_228403

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_f_at_one : 
  deriv f 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l2284_228403


namespace NUMINAMATH_CALUDE_restaurant_revenue_l2284_228445

/-- Calculates the total revenue from meals sold at a restaurant --/
theorem restaurant_revenue 
  (x y z : ℝ) -- Costs of kids, adult, and seniors' meals respectively
  (ratio_kids : ℕ) (ratio_adult : ℕ) (ratio_senior : ℕ) -- Ratio of meals sold
  (kids_meals_sold : ℕ) -- Number of kids meals sold
  (h_ratio : ratio_kids = 3 ∧ ratio_adult = 2 ∧ ratio_senior = 1) -- Given ratio
  (h_kids_sold : kids_meals_sold = 12) -- Given number of kids meals sold
  : 
  ∃ (total_revenue : ℝ),
    total_revenue = kids_meals_sold * x + 
      (kids_meals_sold * ratio_adult / ratio_kids) * y + 
      (kids_meals_sold * ratio_senior / ratio_kids) * z ∧
    total_revenue = 12 * x + 8 * y + 4 * z :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_revenue_l2284_228445


namespace NUMINAMATH_CALUDE_lcm_gcf_product_24_36_l2284_228455

theorem lcm_gcf_product_24_36 : Nat.lcm 24 36 * Nat.gcd 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_product_24_36_l2284_228455


namespace NUMINAMATH_CALUDE_tablecloth_radius_l2284_228466

/-- Given a round tablecloth with a diameter of 10 feet, its radius is 5 feet. -/
theorem tablecloth_radius (diameter : ℝ) (h : diameter = 10) : diameter / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tablecloth_radius_l2284_228466


namespace NUMINAMATH_CALUDE_train_length_calculation_l2284_228453

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length_calculation (train_speed : ℝ) (cross_time : ℝ) (bridge_length : ℝ) :
  train_speed = 65 * (1000 / 3600) →
  cross_time = 13.568145317605362 →
  bridge_length = 145 →
  ∃ (train_length : ℝ), abs (train_length - 100) < 0.1 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l2284_228453


namespace NUMINAMATH_CALUDE_loan_amounts_correct_l2284_228426

-- Define the total loan amount in tens of thousands of yuan
def total_loan : ℝ := 68

-- Define the total annual interest in tens of thousands of yuan
def total_interest : ℝ := 8.42

-- Define the annual interest rate for Type A loan
def rate_A : ℝ := 0.12

-- Define the annual interest rate for Type B loan
def rate_B : ℝ := 0.13

-- Define the amount of Type A loan in tens of thousands of yuan
def loan_A : ℝ := 42

-- Define the amount of Type B loan in tens of thousands of yuan
def loan_B : ℝ := 26

theorem loan_amounts_correct : 
  loan_A + loan_B = total_loan ∧ 
  rate_A * loan_A + rate_B * loan_B = total_interest := by
  sorry

end NUMINAMATH_CALUDE_loan_amounts_correct_l2284_228426


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2284_228482

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : ℝ := x^2 + 2*k*x + k

-- Theorem statement
theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, quadratic_equation x k = 0) ↔ (k ≤ 0 ∨ k ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2284_228482


namespace NUMINAMATH_CALUDE_product_as_sum_of_tens_l2284_228486

theorem product_as_sum_of_tens :
  ∃ n : ℕ, n * 10 = 100 * 100 ∧ n = 1000 := by
  sorry

end NUMINAMATH_CALUDE_product_as_sum_of_tens_l2284_228486


namespace NUMINAMATH_CALUDE_smallest_number_l2284_228438

def number_set : Set ℤ := {1, 0, -2, -6}

theorem smallest_number :
  ∀ x ∈ number_set, -6 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2284_228438


namespace NUMINAMATH_CALUDE_eunjis_rank_l2284_228451

/-- Given that Minyoung arrived 33rd in a race and Eunji arrived 11 places after Minyoung,
    prove that Eunji's rank is 44th. -/
theorem eunjis_rank (minyoungs_rank : ℕ) (places_after : ℕ) 
  (h1 : minyoungs_rank = 33) 
  (h2 : places_after = 11) : 
  minyoungs_rank + places_after = 44 := by
  sorry

end NUMINAMATH_CALUDE_eunjis_rank_l2284_228451


namespace NUMINAMATH_CALUDE_inequality_proof_l2284_228449

theorem inequality_proof (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4 > x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2284_228449


namespace NUMINAMATH_CALUDE_angle_CRT_is_72_degrees_l2284_228460

-- Define the triangle CAT
structure Triangle (C A T : Type) where
  angle_ACT : ℝ
  angle_ATC : ℝ
  angle_CAT : ℝ

-- Define the theorem
theorem angle_CRT_is_72_degrees 
  (CAT : Triangle C A T) 
  (h1 : CAT.angle_ACT = CAT.angle_ATC) 
  (h2 : CAT.angle_CAT = 36) 
  (h3 : ∃ (R : Type), (angle_CTR : ℝ) = CAT.angle_ATC / 2) : 
  (angle_CRT : ℝ) = 72 := by
  sorry

end NUMINAMATH_CALUDE_angle_CRT_is_72_degrees_l2284_228460


namespace NUMINAMATH_CALUDE_multiply_and_add_l2284_228472

theorem multiply_and_add : 12 * 25 + 16 * 15 = 540 := by sorry

end NUMINAMATH_CALUDE_multiply_and_add_l2284_228472
