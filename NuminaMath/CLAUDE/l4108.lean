import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4108_410830

theorem quadratic_inequality_solution_set (x : ℝ) : 
  {x : ℝ | x^2 - 4*x + 3 < 0} = Set.Ioo 1 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4108_410830


namespace NUMINAMATH_CALUDE_stratified_sampling_sum_l4108_410806

theorem stratified_sampling_sum (total_population : ℕ) (sample_size : ℕ) 
  (type_a_count : ℕ) (type_b_count : ℕ) :
  total_population = 100 →
  sample_size = 20 →
  type_a_count = 10 →
  type_b_count = 20 →
  (type_a_count * sample_size / total_population + 
   type_b_count * sample_size / total_population : ℚ) = 6 := by
  sorry

#check stratified_sampling_sum

end NUMINAMATH_CALUDE_stratified_sampling_sum_l4108_410806


namespace NUMINAMATH_CALUDE_min_sum_squares_l4108_410895

theorem min_sum_squares (x y : ℝ) (h : x * y - x - y = 1) :
  ∃ (min : ℝ), min = 6 - 4 * Real.sqrt 2 ∧ 
  ∀ (a b : ℝ), a * b - a - b = 1 → a^2 + b^2 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l4108_410895


namespace NUMINAMATH_CALUDE_water_addition_changes_ratio_l4108_410802

/-- Given a mixture of alcohol and water, prove that adding 10 liters of water
    changes the ratio from 4:3 to 4:5 when the initial amount of alcohol is 20 liters. -/
theorem water_addition_changes_ratio :
  let initial_alcohol : ℝ := 20
  let initial_ratio : ℝ := 4 / 3
  let final_ratio : ℝ := 4 / 5
  let water_added : ℝ := 10
  let initial_water : ℝ := initial_alcohol / initial_ratio
  let final_water : ℝ := initial_water + water_added
  initial_alcohol / initial_water = initial_ratio ∧
  initial_alcohol / final_water = final_ratio :=
by sorry

end NUMINAMATH_CALUDE_water_addition_changes_ratio_l4108_410802


namespace NUMINAMATH_CALUDE_find_number_l4108_410825

theorem find_number : ∃ x : ℕ, x * 9999 = 183868020 ∧ x = 18387 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l4108_410825


namespace NUMINAMATH_CALUDE_walter_seal_time_l4108_410808

/-- Given Walter's zoo visit, prove he spent 13 minutes looking at seals. -/
theorem walter_seal_time : ∀ (S : ℕ), 
  S + 8 * S + 13 = 130 → S = 13 := by
  sorry

end NUMINAMATH_CALUDE_walter_seal_time_l4108_410808


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l4108_410875

def f (x : ℝ) := x^2

theorem f_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l4108_410875


namespace NUMINAMATH_CALUDE_min_difference_for_always_larger_l4108_410816

/-- Pratyya's daily number transformation -/
def pratyya_transform (n : ℤ) : ℤ := 2 * n - 2

/-- Payel's daily number transformation -/
def payel_transform (m : ℤ) : ℤ := 2 * m + 2

/-- The difference between Pratyya's and Payel's numbers after t days -/
def difference (n m : ℤ) (t : ℕ) : ℤ :=
  pratyya_transform (n + t) - payel_transform (m + t)

/-- The theorem stating the minimum difference for Pratyya's number to always be larger -/
theorem min_difference_for_always_larger (n m : ℤ) (h : n > m) :
  (∀ t : ℕ, difference n m t > 0) ↔ n - m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_min_difference_for_always_larger_l4108_410816


namespace NUMINAMATH_CALUDE_restricted_arrangements_eq_78_l4108_410891

/-- The number of ways to arrange n elements. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 5 contestants with restrictions. -/
def restrictedArrangements : ℕ :=
  arrangements 4 + 3 * 3 * arrangements 3

/-- Theorem stating that the number of restricted arrangements is 78. -/
theorem restricted_arrangements_eq_78 :
  restrictedArrangements = 78 := by sorry

end NUMINAMATH_CALUDE_restricted_arrangements_eq_78_l4108_410891


namespace NUMINAMATH_CALUDE_ellipse_tangent_existence_l4108_410865

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is outside the ellipse -/
def isOutside (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) > 1

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Checks if a line is tangent to the ellipse -/
def isTangent (e : Ellipse) (l : Line) : Prop :=
  ∃ θ : ℝ, l.p2.x = e.a * Real.cos θ ∧ l.p2.y = e.b * Real.sin θ ∧
    (l.p1.x * l.p2.x / e.a^2) + (l.p1.y * l.p2.y / e.b^2) = 1

/-- Main theorem: For any ellipse and point outside it, there exist two tangent lines -/
theorem ellipse_tangent_existence (e : Ellipse) (p : Point) (h : isOutside e p) :
  ∃ l1 l2 : Line, l1 ≠ l2 ∧ l1.p1 = p ∧ l2.p1 = p ∧ isTangent e l1 ∧ isTangent e l2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_tangent_existence_l4108_410865


namespace NUMINAMATH_CALUDE_m_in_open_interval_l4108_410801

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem m_in_open_interval
  (f : ℝ → ℝ)
  (h_decreasing : monotonically_decreasing f)
  (h_inequality : f (m^2) > f m)
  : m ∈ Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_m_in_open_interval_l4108_410801


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4108_410813

-- Define the triangle
def triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the equation for the third side
def third_side_equation (x : ℝ) : Prop :=
  x^2 - 8*x + 12 = 0

-- Theorem statement
theorem triangle_perimeter : 
  ∃ (x : ℝ), 
    third_side_equation x ∧ 
    triangle 4 7 x ∧ 
    4 + 7 + x = 17 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l4108_410813


namespace NUMINAMATH_CALUDE_x_value_l4108_410817

theorem x_value (x y : ℚ) (h1 : x / y = 15 / 5) (h2 : y = 10) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l4108_410817


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4108_410873

theorem inequality_solution_set : 
  {x : ℝ | x * (2 - x) ≤ 0} = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4108_410873


namespace NUMINAMATH_CALUDE_total_cable_cost_neighborhood_cable_cost_l4108_410804

/-- The total cost of cable for a neighborhood with the given street configuration and cable requirements. -/
theorem total_cable_cost (ew_streets : ℕ) (ew_length : ℝ) (ns_streets : ℕ) (ns_length : ℝ) 
  (cable_per_mile : ℝ) (cost_per_mile : ℝ) : ℝ :=
  let total_street_length := ew_streets * ew_length + ns_streets * ns_length
  let total_cable_length := total_street_length * cable_per_mile
  total_cable_length * cost_per_mile

/-- The total cost of cable for the specific neighborhood described in the problem. -/
theorem neighborhood_cable_cost : total_cable_cost 18 2 10 4 5 2000 = 760000 := by
  sorry

end NUMINAMATH_CALUDE_total_cable_cost_neighborhood_cable_cost_l4108_410804


namespace NUMINAMATH_CALUDE_negate_difference_l4108_410843

theorem negate_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end NUMINAMATH_CALUDE_negate_difference_l4108_410843


namespace NUMINAMATH_CALUDE_unique_m_value_l4108_410809

-- Define the set A
def A (m : ℚ) : Set ℚ := {m + 2, 2 * m^2 + m}

-- Theorem statement
theorem unique_m_value : ∃! m : ℚ, 3 ∈ A m ∧ (∀ x ∈ A m, x = m + 2 ∨ x = 2 * m^2 + m) :=
by sorry

end NUMINAMATH_CALUDE_unique_m_value_l4108_410809


namespace NUMINAMATH_CALUDE_intersection_distance_l4108_410819

-- Define the circle centers and radius
structure CircleConfig where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  r : ℝ

-- Define the conditions
def validConfig (config : CircleConfig) : Prop :=
  1 < config.r ∧ config.r < 2 ∧
  dist config.A config.B = 2 ∧
  dist config.B config.C = 2 ∧
  dist config.A config.C = 2

-- Define the intersection points
def B' (config : CircleConfig) : ℝ × ℝ := sorry
def C' (config : CircleConfig) : ℝ × ℝ := sorry

-- State the theorem
theorem intersection_distance (config : CircleConfig) 
  (h : validConfig config) :
  dist (B' config) (C' config) = 1 + Real.sqrt (3 * (config.r^2 - 1)) :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l4108_410819


namespace NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l4108_410823

theorem pigeonhole_on_permutation_sums (n : ℕ) : 
  ∀ (p : Fin (2*n) → Fin (2*n)), 
  ∃ (i j : Fin (2*n)), i ≠ j ∧ 
  ((p i).val + i.val + 1) % (2*n) = ((p j).val + j.val + 1) % (2*n) :=
sorry

end NUMINAMATH_CALUDE_pigeonhole_on_permutation_sums_l4108_410823


namespace NUMINAMATH_CALUDE_sparrows_among_non_pigeons_l4108_410803

theorem sparrows_among_non_pigeons (sparrows : ℝ) (pigeons : ℝ) (parrots : ℝ) (crows : ℝ)
  (h1 : sparrows = 0.4)
  (h2 : pigeons = 0.2)
  (h3 : parrots = 0.15)
  (h4 : crows = 0.25)
  (h5 : sparrows + pigeons + parrots + crows = 1) :
  sparrows / (1 - pigeons) = 0.5 := by
sorry

end NUMINAMATH_CALUDE_sparrows_among_non_pigeons_l4108_410803


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l4108_410878

theorem reciprocal_of_negative_fraction (n : ℤ) (n_nonzero : n ≠ 0) :
  ((-1 : ℚ) / n)⁻¹ = -n := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l4108_410878


namespace NUMINAMATH_CALUDE_weight_four_moles_CaBr2_l4108_410882

/-- The atomic weight of calcium in g/mol -/
def calcium_weight : ℝ := 40.08

/-- The atomic weight of bromine in g/mol -/
def bromine_weight : ℝ := 79.904

/-- The number of calcium atoms in a molecule of CaBr2 -/
def calcium_atoms : ℕ := 1

/-- The number of bromine atoms in a molecule of CaBr2 -/
def bromine_atoms : ℕ := 2

/-- The number of moles of CaBr2 -/
def moles_CaBr2 : ℝ := 4

/-- The weight of a given number of moles of CaBr2 -/
def weight_CaBr2 (moles : ℝ) : ℝ :=
  moles * (calcium_atoms * calcium_weight + bromine_atoms * bromine_weight)

/-- Theorem stating that the weight of 4 moles of CaBr2 is 799.552 grams -/
theorem weight_four_moles_CaBr2 :
  weight_CaBr2 moles_CaBr2 = 799.552 := by sorry

end NUMINAMATH_CALUDE_weight_four_moles_CaBr2_l4108_410882


namespace NUMINAMATH_CALUDE_m_greater_than_n_l4108_410883

theorem m_greater_than_n (a : ℝ) : 5 * a^2 - a + 1 > 4 * a^2 + a - 1 := by
  sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l4108_410883


namespace NUMINAMATH_CALUDE_josh_marbles_count_l4108_410850

/-- The number of marbles Josh has after receiving some from Jack -/
def total_marbles (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Josh's final marble count is the sum of his initial count and received marbles -/
theorem josh_marbles_count (initial : ℕ) (received : ℕ) :
  total_marbles initial received = initial + received := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_count_l4108_410850


namespace NUMINAMATH_CALUDE_value_of_a_l4108_410854

theorem value_of_a (A B : Set ℕ) (a : ℕ) 
  (hA : A = {1, 2})
  (hB : B = {2, a})
  (hUnion : A ∪ B = {1, 2, 4}) :
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l4108_410854


namespace NUMINAMATH_CALUDE_miranda_goose_feathers_l4108_410864

/-- The number of feathers needed for one pillow -/
def feathers_per_pillow : ℕ := 2 * 300

/-- The number of pillows Miranda can stuff -/
def pillows_stuffed : ℕ := 6

/-- The number of feathers on Miranda's goose -/
def goose_feathers : ℕ := feathers_per_pillow * pillows_stuffed

theorem miranda_goose_feathers : goose_feathers = 3600 := by
  sorry

end NUMINAMATH_CALUDE_miranda_goose_feathers_l4108_410864


namespace NUMINAMATH_CALUDE_arithmetic_progression_roots_l4108_410899

/-- A polynomial of the form x^4 + px^2 + q has 4 real roots in arithmetic progression
    if and only if p ≤ 0 and q = 0.09p^2 -/
theorem arithmetic_progression_roots (p q : ℝ) :
  (∃ (a d : ℝ), ∀ (x : ℝ), x^4 + p*x^2 + q = 0 ↔ 
    x = a - 3*d ∨ x = a - d ∨ x = a + d ∨ x = a + 3*d) ↔ 
  (p ≤ 0 ∧ q = 0.09 * p^2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_roots_l4108_410899


namespace NUMINAMATH_CALUDE_sandy_current_fingernail_length_l4108_410842

/-- Sandy's current age in years -/
def current_age : ℕ := 12

/-- Sandy's age when she achieves the world record in years -/
def record_age : ℕ := 32

/-- The world record for longest fingernails in inches -/
def world_record : ℝ := 26

/-- Sandy's fingernail growth rate in inches per month -/
def growth_rate : ℝ := 0.1

/-- The number of months in a year -/
def months_per_year : ℕ := 12

theorem sandy_current_fingernail_length :
  world_record - (growth_rate * months_per_year * (record_age - current_age : ℝ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandy_current_fingernail_length_l4108_410842


namespace NUMINAMATH_CALUDE_cost_price_calculation_l4108_410822

theorem cost_price_calculation (C : ℝ) : 
  (0.9 * C = C - 0.1 * C) →
  (1.1 * C = C + 0.1 * C) →
  (1.1 * C - 0.9 * C = 50) →
  C = 250 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l4108_410822


namespace NUMINAMATH_CALUDE_exactly_one_positive_integer_satisfies_condition_l4108_410870

theorem exactly_one_positive_integer_satisfies_condition :
  ∃! (n : ℕ), n > 0 ∧ 20 - 5 * n > 12 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_positive_integer_satisfies_condition_l4108_410870


namespace NUMINAMATH_CALUDE_variance_mean_preserved_l4108_410885

def initial_set : List Int := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

def mean (xs : List Int) : ℚ := (xs.sum : ℚ) / xs.length

def variance (xs : List Int) : ℚ :=
  let m := mean xs
  ((xs.map (λ x => ((x : ℚ) - m) ^ 2)).sum) / xs.length

def replacement_set1 : List Int := [-5, -4, -3, -2, -1, 0, 1, 2, 3, -1, 5]
def replacement_set2 : List Int := [-5, 1, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5]

theorem variance_mean_preserved :
  (mean initial_set = mean replacement_set1 ∧
   variance initial_set = variance replacement_set1) ∨
  (mean initial_set = mean replacement_set2 ∧
   variance initial_set = variance replacement_set2) :=
by sorry

end NUMINAMATH_CALUDE_variance_mean_preserved_l4108_410885


namespace NUMINAMATH_CALUDE_exists_minimal_period_greater_than_l4108_410869

/-- Definition of the sequence family F(x) -/
def F (x : ℝ) : (ℕ → ℝ) → Prop :=
  λ a => ∀ n, a (n + 1) = x - 1 / a n

/-- Definition of periodicity for a sequence -/
def IsPeriodic (a : ℕ → ℝ) (p : ℕ) : Prop :=
  ∀ n, a (n + p) = a n

/-- Definition of minimal period for the family F(x) -/
def IsMinimalPeriod (x : ℝ) (p : ℕ) : Prop :=
  (∀ a, F x a → IsPeriodic a p) ∧
  (∀ q, 0 < q → q < p → ∃ a, F x a ∧ ¬IsPeriodic a q)

/-- Main theorem statement -/
theorem exists_minimal_period_greater_than (P : ℕ) :
  ∃ x : ℝ, ∃ p : ℕ, p > P ∧ IsMinimalPeriod x p :=
sorry

end NUMINAMATH_CALUDE_exists_minimal_period_greater_than_l4108_410869


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4108_410807

/-- The number of terms in a geometric sequence with first term 1 and common ratio 1/4 
    that sum to 85/64 -/
theorem geometric_sequence_sum (n : ℕ) : 
  (1 - (1/4)^n) / (1 - 1/4) = 85/64 → n = 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4108_410807


namespace NUMINAMATH_CALUDE_unique_digit_property_l4108_410835

theorem unique_digit_property : ∃! x : ℕ, x < 10 ∧ ∀ a : ℕ, 10 * a + x = a + x + a * x := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_property_l4108_410835


namespace NUMINAMATH_CALUDE_intersection_is_ellipse_l4108_410812

-- Define the plane
def plane (z : ℝ) : Prop := z = 2

-- Define the ellipsoid
def ellipsoid (x y z : ℝ) : Prop := x^2/12 + y^2/4 + z^2/16 = 1

-- Define the intersection curve
def intersection_curve (x y : ℝ) : Prop := x^2/9 + y^2/3 = 1

-- Theorem statement
theorem intersection_is_ellipse :
  ∀ x y z : ℝ,
  plane z ∧ ellipsoid x y z →
  intersection_curve x y ∧
  ∃ a b : ℝ, a = 3 ∧ b = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_is_ellipse_l4108_410812


namespace NUMINAMATH_CALUDE_welders_who_left_l4108_410855

/-- Represents the problem of welders working on an order -/
structure WelderProblem where
  initial_welders : ℕ
  initial_days : ℕ
  remaining_days : ℕ
  welders_left : ℕ

/-- The specific problem instance -/
def problem : WelderProblem :=
  { initial_welders := 12
  , initial_days := 8
  , remaining_days := 28
  , welders_left := 3 }

/-- Theorem stating the number of welders who left for another project -/
theorem welders_who_left (p : WelderProblem) : 
  p.initial_welders - p.welders_left = 9 :=
by sorry

#check welders_who_left problem

end NUMINAMATH_CALUDE_welders_who_left_l4108_410855


namespace NUMINAMATH_CALUDE_hallway_tiling_l4108_410877

theorem hallway_tiling (hallway_length hallway_width : ℕ) 
  (border_tile_size interior_tile_size : ℕ) : 
  hallway_length = 20 → 
  hallway_width = 14 → 
  border_tile_size = 2 → 
  interior_tile_size = 3 → 
  (2 * (hallway_length - 2 * border_tile_size) / border_tile_size + 
   2 * (hallway_width - 2 * border_tile_size) / border_tile_size + 4) + 
  ((hallway_length - 2 * border_tile_size) * 
   (hallway_width - 2 * border_tile_size)) / (interior_tile_size^2) = 48 := by
  sorry

end NUMINAMATH_CALUDE_hallway_tiling_l4108_410877


namespace NUMINAMATH_CALUDE_power_division_equals_729_l4108_410893

theorem power_division_equals_729 : (3 ^ 12) / (27 ^ 2) = 729 := by sorry

end NUMINAMATH_CALUDE_power_division_equals_729_l4108_410893


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l4108_410836

theorem greatest_integer_b_for_quadratic_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 15 ≠ -6) ↔ b ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l4108_410836


namespace NUMINAMATH_CALUDE_sugar_cube_weight_l4108_410868

theorem sugar_cube_weight
  (ants1 : ℕ) (cubes1 : ℕ) (weight1 : ℝ) (hours1 : ℝ)
  (ants2 : ℕ) (cubes2 : ℕ) (hours2 : ℝ)
  (h1 : ants1 = 15)
  (h2 : cubes1 = 600)
  (h3 : weight1 = 10)
  (h4 : hours1 = 5)
  (h5 : ants2 = 20)
  (h6 : cubes2 = 960)
  (h7 : hours2 = 3)
  : ∃ weight2 : ℝ,
    weight2 = 5 ∧
    (ants1 : ℝ) * (cubes1 : ℝ) * weight1 / hours1 =
    (ants2 : ℝ) * (cubes2 : ℝ) * weight2 / hours2 :=
by
  sorry

end NUMINAMATH_CALUDE_sugar_cube_weight_l4108_410868


namespace NUMINAMATH_CALUDE_probability_theorem_l4108_410827

def total_balls : ℕ := 6
def new_balls : ℕ := 4
def old_balls : ℕ := 2

def probability_one_new_one_old : ℚ :=
  (new_balls * old_balls) / (total_balls * (total_balls - 1) / 2)

theorem probability_theorem :
  probability_one_new_one_old = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l4108_410827


namespace NUMINAMATH_CALUDE_max_angle_APB_l4108_410879

/-- An ellipse with focus F and directrix l -/
structure Ellipse where
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- The focus of the ellipse -/
  F : ℝ × ℝ
  /-- The point where the directrix intersects the axis of symmetry -/
  P : ℝ × ℝ

/-- A chord of the ellipse passing through the focus -/
structure Chord (E : Ellipse) where
  /-- One endpoint of the chord -/
  A : ℝ × ℝ
  /-- The other endpoint of the chord -/
  B : ℝ × ℝ
  /-- The chord passes through the focus -/
  passes_through_focus : A.1 < E.F.1 ∧ E.F.1 < B.1

/-- The angle APB formed by a chord AB and the point P -/
def angle_APB (E : Ellipse) (C : Chord E) : ℝ :=
  sorry

/-- The theorem stating that the maximum value of angle APB is 2 arctan e -/
theorem max_angle_APB (E : Ellipse) :
  ∀ C : Chord E, angle_APB E C ≤ 2 * Real.arctan E.e ∧
  ∃ C : Chord E, angle_APB E C = 2 * Real.arctan E.e :=
sorry

end NUMINAMATH_CALUDE_max_angle_APB_l4108_410879


namespace NUMINAMATH_CALUDE_total_practice_time_is_307_5_l4108_410832

/-- Represents Daniel's weekly practice schedule -/
structure PracticeSchedule where
  basketball_school_day : ℝ  -- Minutes of basketball practice on school days
  basketball_weekend_day : ℝ  -- Minutes of basketball practice on weekend days
  soccer_weekday : ℝ  -- Minutes of soccer practice on weekdays
  gymnastics : ℝ  -- Minutes of gymnastics practice
  soccer_saturday : ℝ  -- Minutes of soccer practice on Saturday (averaged)
  swimming_saturday : ℝ  -- Minutes of swimming practice on Saturday (averaged)

/-- Calculates the total practice time for one week -/
def total_practice_time (schedule : PracticeSchedule) : ℝ :=
  schedule.basketball_school_day * 5 +
  schedule.basketball_weekend_day * 2 +
  schedule.soccer_weekday * 3 +
  schedule.gymnastics * 2 +
  schedule.soccer_saturday +
  schedule.swimming_saturday

/-- Daniel's actual practice schedule -/
def daniel_schedule : PracticeSchedule :=
  { basketball_school_day := 15
  , basketball_weekend_day := 30
  , soccer_weekday := 20
  , gymnastics := 30
  , soccer_saturday := 22.5
  , swimming_saturday := 30 }

theorem total_practice_time_is_307_5 :
  total_practice_time daniel_schedule = 307.5 := by
  sorry

end NUMINAMATH_CALUDE_total_practice_time_is_307_5_l4108_410832


namespace NUMINAMATH_CALUDE_fifteen_factorial_sum_TMH_l4108_410888

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def base_ten_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 10) :: aux (m / 10)
    (aux n).reverse

theorem fifteen_factorial_sum_TMH :
  ∃ (T M H : ℕ),
    T < 10 ∧ M < 10 ∧ H < 10 ∧
    base_ten_repr (factorial 15) = [1, 3, 0, 7, M, 7, T, 2, 0, 0, H, 0, 0] ∧
    T + M + H = 2 :=
by sorry

end NUMINAMATH_CALUDE_fifteen_factorial_sum_TMH_l4108_410888


namespace NUMINAMATH_CALUDE_line_AB_parallel_to_xOz_plane_l4108_410800

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
def Vector3D : Type := ℝ × ℝ × ℝ

/-- Calculate the vector from point A to point B -/
def vectorBetweenPoints (A B : Point3D) : Vector3D :=
  (B.x - A.x, B.y - A.y, B.z - A.z)

/-- Check if a vector is parallel to the xOz plane -/
def isParallelToXOZ (v : Vector3D) : Prop :=
  v.2 = 0

/-- The main theorem: Line AB is parallel to xOz plane -/
theorem line_AB_parallel_to_xOz_plane :
  let A : Point3D := ⟨1, 3, 0⟩
  let B : Point3D := ⟨0, 3, -1⟩
  let AB : Vector3D := vectorBetweenPoints A B
  isParallelToXOZ AB := by sorry

end NUMINAMATH_CALUDE_line_AB_parallel_to_xOz_plane_l4108_410800


namespace NUMINAMATH_CALUDE_max_lateral_surface_area_l4108_410859

theorem max_lateral_surface_area (x y : ℝ) : 
  x > 0 → y > 0 → x + y = 10 → 2 * π * x * y ≤ 50 * π :=
by sorry

end NUMINAMATH_CALUDE_max_lateral_surface_area_l4108_410859


namespace NUMINAMATH_CALUDE_range_of_a_l4108_410876

-- Define the inequality and its solution set
def inequality (a x : ℝ) : Prop := (a - 1) * x > a - 1
def solution_set (x : ℝ) : Prop := x < 1

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, inequality a x ↔ solution_set x) → a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4108_410876


namespace NUMINAMATH_CALUDE_new_members_average_weight_l4108_410814

theorem new_members_average_weight 
  (initial_count : ℕ) 
  (initial_average : ℝ) 
  (new_count : ℕ) 
  (new_average : ℝ) 
  (double_counted_weight : ℝ) :
  initial_count = 10 →
  initial_average = 75 →
  new_count = 3 →
  new_average = 77 →
  double_counted_weight = 65 →
  let corrected_total := initial_count * initial_average - double_counted_weight
  let new_total := (initial_count + new_count - 1) * new_average
  let new_members_total := new_total - corrected_total
  (new_members_total / new_count) = 79.67 := by
sorry

end NUMINAMATH_CALUDE_new_members_average_weight_l4108_410814


namespace NUMINAMATH_CALUDE_perpendicular_planes_parallel_l4108_410853

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_parallel 
  (m : Line) (α β : Plane) 
  (h1 : perpendicular m α) 
  (h2 : perpendicular m β) : 
  parallel α β := by sorry

end NUMINAMATH_CALUDE_perpendicular_planes_parallel_l4108_410853


namespace NUMINAMATH_CALUDE_inscribed_squares_side_length_difference_l4108_410880

/-- Given a circle with radius R and a chord at distance h from the center,
    prove that the difference in side lengths of two squares inscribed in the
    segments formed by the chord is 8h/5. Each square has two adjacent vertices
    on the chord and two on the circle arc. -/
theorem inscribed_squares_side_length_difference
  (R h : ℝ) (h_pos : 0 < h) (h_lt_R : h < R) :
  ∃ x y : ℝ,
    (0 < x) ∧ (0 < y) ∧
    ((2 * x - h)^2 + x^2 = R^2) ∧
    ((2 * y + h)^2 + y^2 = R^2) ∧
    (2 * x - 2 * y = 8 * h / 5) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_side_length_difference_l4108_410880


namespace NUMINAMATH_CALUDE_even_perfect_square_factors_l4108_410811

/-- The number of even perfect square factors of 2^6 * 7^10 * 3^2 -/
theorem even_perfect_square_factors : 
  (Finset.filter (fun a => a % 2 = 0 ∧ 2 ≤ a) (Finset.range 7)).card *
  (Finset.filter (fun b => b % 2 = 0) (Finset.range 11)).card *
  (Finset.filter (fun c => c % 2 = 0) (Finset.range 3)).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_even_perfect_square_factors_l4108_410811


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l4108_410886

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l4108_410886


namespace NUMINAMATH_CALUDE_min_slope_tangent_line_l4108_410852

noncomputable def f (x b a : ℝ) : ℝ := Real.log x + x^2 - b*x + a

theorem min_slope_tangent_line (b a : ℝ) (hb : b > 0) :
  ∃ m : ℝ, m = 2 ∧ ∀ x, x > 0 → (1/x + x) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_slope_tangent_line_l4108_410852


namespace NUMINAMATH_CALUDE_sophia_reading_progress_l4108_410820

theorem sophia_reading_progress (total_pages : ℕ) (pages_read : ℕ) : 
  total_pages = 270 →
  pages_read = (total_pages - pages_read) + 90 →
  pages_read / total_pages = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_sophia_reading_progress_l4108_410820


namespace NUMINAMATH_CALUDE_point_same_side_condition_l4108_410860

/-- A point on a line is on the same side as the origin with respect to another line -/
def same_side_as_origin (k b : ℝ) : Prop :=
  ∀ x : ℝ, (x - (k * x + b) + 2) * 2 > 0

/-- Theorem: If a point on y = kx + b is on the same side as the origin
    with respect to x - y + 2 = 0, then k = 1 and b < 2 -/
theorem point_same_side_condition (k b : ℝ) :
  same_side_as_origin k b → k = 1 ∧ b < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_same_side_condition_l4108_410860


namespace NUMINAMATH_CALUDE_cos_equation_solution_l4108_410849

theorem cos_equation_solution (x : ℝ) : 
  (Real.cos (2 * x) - 3 * Real.cos (4 * x))^2 = 16 + (Real.cos (5 * x))^2 → 
  ∃ k : ℤ, x = π / 2 + k * π :=
by sorry

end NUMINAMATH_CALUDE_cos_equation_solution_l4108_410849


namespace NUMINAMATH_CALUDE_perfect_square_condition_l4108_410874

theorem perfect_square_condition (n : ℕ+) : 
  (∃ (a : ℕ), 5^(n : ℕ) + 4 = a^2) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l4108_410874


namespace NUMINAMATH_CALUDE_at_least_one_quadratic_has_root_l4108_410831

theorem at_least_one_quadratic_has_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ x : ℝ, (a * x^2 + 2 * b * x + c = 0) ∨ (b * x^2 + 2 * c * x + a = 0) ∨ (c * x^2 + 2 * a * x + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_quadratic_has_root_l4108_410831


namespace NUMINAMATH_CALUDE_remainder_371073_div_6_l4108_410844

theorem remainder_371073_div_6 : 371073 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_371073_div_6_l4108_410844


namespace NUMINAMATH_CALUDE_min_value_theorem_l4108_410881

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  x^2 + 3*y ≥ 20 + 16 * Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧
    x₀^2 + 3*y₀ = 20 + 16 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4108_410881


namespace NUMINAMATH_CALUDE_equation_solutions_l4108_410841

theorem equation_solutions : 
  {x : ℝ | x^2 - 3 * |x| - 4 = 0} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4108_410841


namespace NUMINAMATH_CALUDE_abs_x_minus_one_necessary_not_sufficient_l4108_410846

theorem abs_x_minus_one_necessary_not_sufficient :
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x - 3) < 0)) ∧
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_necessary_not_sufficient_l4108_410846


namespace NUMINAMATH_CALUDE_right_triangle_trig_identity_l4108_410896

theorem right_triangle_trig_identity 
  (A B C : Real) 
  (right_angle : C = Real.pi / 2)
  (condition1 : Real.cos A ^ 2 + Real.cos B ^ 2 + 2 * Real.sin A * Real.sin B * Real.cos C = 3/2)
  (condition2 : Real.cos B ^ 2 + 2 * Real.sin B * Real.cos A = 5/3) :
  Real.cos A ^ 2 + 2 * Real.sin A * Real.cos B = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_identity_l4108_410896


namespace NUMINAMATH_CALUDE_prob_other_is_one_given_one_is_one_l4108_410851

/-- Represents the number of balls with each label -/
def ballCounts : Fin 3 → Nat
  | 0 => 1  -- number of balls labeled 0
  | 1 => 2  -- number of balls labeled 1
  | 2 => 2  -- number of balls labeled 2

/-- The total number of balls -/
def totalBalls : Nat := (ballCounts 0) + (ballCounts 1) + (ballCounts 2)

/-- The probability of drawing two balls, one of which is labeled 1 -/
def probOneIsOne : ℚ := (ballCounts 1 * (totalBalls - ballCounts 1)) / (totalBalls.choose 2)

/-- The probability of drawing two balls, both labeled 1 -/
def probBothAreOne : ℚ := ((ballCounts 1).choose 2) / (totalBalls.choose 2)

/-- The main theorem to prove -/
theorem prob_other_is_one_given_one_is_one :
  probBothAreOne / probOneIsOne = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_prob_other_is_one_given_one_is_one_l4108_410851


namespace NUMINAMATH_CALUDE_quadratic_expression_equals_39_l4108_410826

theorem quadratic_expression_equals_39 (x : ℝ) :
  (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 + 3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_equals_39_l4108_410826


namespace NUMINAMATH_CALUDE_train_travel_time_l4108_410818

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  hlt24 : hours < 24
  mlt60 : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  let totalMinutes1 := t1.hours * 60 + t1.minutes
  let totalMinutes2 := t2.hours * 60 + t2.minutes
  totalMinutes2 - totalMinutes1

/-- The train travel time theorem -/
theorem train_travel_time :
  let departureTime : Time := ⟨7, 5, by norm_num, by norm_num⟩
  let arrivalTime : Time := ⟨7, 59, by norm_num, by norm_num⟩
  timeDifference departureTime arrivalTime = 54 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_time_l4108_410818


namespace NUMINAMATH_CALUDE_multiples_imply_lower_bound_l4108_410834

theorem multiples_imply_lower_bound (n : ℕ) (a : ℕ) (h1 : n > 1) (h2 : a > n^2) :
  (∀ i ∈ Finset.range n, ∃ k ∈ Finset.range n, (a + k + 1) % (n^2 + i + 1) = 0) →
  a > n^4 - n^3 := by
  sorry

end NUMINAMATH_CALUDE_multiples_imply_lower_bound_l4108_410834


namespace NUMINAMATH_CALUDE_assignment_b_is_valid_l4108_410833

-- Define what a valid assignment statement is
def is_valid_assignment (stmt : String) : Prop :=
  ∃ (var : String) (expr : String), stmt = var ++ "=" ++ expr ∧ var.length > 0

-- Define the specific statement we're checking
def statement_to_check : String := "a=a+1"

-- Theorem to prove
theorem assignment_b_is_valid : is_valid_assignment statement_to_check := by
  sorry

end NUMINAMATH_CALUDE_assignment_b_is_valid_l4108_410833


namespace NUMINAMATH_CALUDE_remaining_episodes_l4108_410845

theorem remaining_episodes (series1_seasons series2_seasons episodes_per_season episodes_lost_per_season : ℕ) 
  (h1 : series1_seasons = 12)
  (h2 : series2_seasons = 14)
  (h3 : episodes_per_season = 16)
  (h4 : episodes_lost_per_season = 2) :
  (series1_seasons * episodes_per_season + series2_seasons * episodes_per_season) -
  (series1_seasons * episodes_lost_per_season + series2_seasons * episodes_lost_per_season) = 364 := by
sorry

end NUMINAMATH_CALUDE_remaining_episodes_l4108_410845


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l4108_410840

theorem solve_quadratic_equation (s t : ℝ) (h1 : t = 8 * s^2) (h2 : t = 4.8) :
  s = Real.sqrt 0.6 ∨ s = -Real.sqrt 0.6 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l4108_410840


namespace NUMINAMATH_CALUDE_work_completion_time_l4108_410862

-- Define the work rates
def work_rate_A : ℚ := 1 / 9
def work_rate_B : ℚ := 1 / 18
def work_rate_combined : ℚ := 1 / 6

-- Define the completion times
def time_A : ℚ := 9
def time_B : ℚ := 18
def time_combined : ℚ := 6

-- Theorem statement
theorem work_completion_time :
  (work_rate_A + work_rate_B = work_rate_combined) →
  (1 / work_rate_A = time_A) →
  (1 / work_rate_B = time_B) →
  (1 / work_rate_combined = time_combined) →
  time_B = 18 := by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l4108_410862


namespace NUMINAMATH_CALUDE_arithmetic_equation_l4108_410892

theorem arithmetic_equation : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l4108_410892


namespace NUMINAMATH_CALUDE_f_max_min_difference_l4108_410838

noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.sin x + Real.cos x) - (1/2) * Real.sin (2 * x)

theorem f_max_min_difference :
  (⨆ (x : ℝ), f x) - (⨅ (x : ℝ), f x) = Real.exp (Real.sqrt 2) - Real.exp (-Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_difference_l4108_410838


namespace NUMINAMATH_CALUDE_probability_all_red_at_fourth_l4108_410821

/-- The number of white balls initially in the bag -/
def initial_white_balls : ℕ := 8

/-- The number of red balls initially in the bag -/
def initial_red_balls : ℕ := 2

/-- The total number of balls initially in the bag -/
def total_balls : ℕ := initial_white_balls + initial_red_balls

/-- The probability of drawing a specific sequence of balls -/
def sequence_probability (red_indices : List ℕ) : ℚ :=
  sorry

/-- The probability of drawing all red balls exactly at the 4th draw -/
def all_red_at_fourth_draw : ℚ :=
  sequence_probability [1, 4] + sequence_probability [2, 4] + sequence_probability [3, 4]

theorem probability_all_red_at_fourth : all_red_at_fourth_draw = 434/10000 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_red_at_fourth_l4108_410821


namespace NUMINAMATH_CALUDE_oxygen_atoms_in_compound_l4108_410829

def atomic_weight_carbon : ℕ := 12
def atomic_weight_hydrogen : ℕ := 1
def atomic_weight_oxygen : ℕ := 16

def num_carbon_atoms : ℕ := 3
def num_hydrogen_atoms : ℕ := 6
def total_molecular_weight : ℕ := 58

theorem oxygen_atoms_in_compound :
  let weight_carbon_hydrogen := num_carbon_atoms * atomic_weight_carbon + num_hydrogen_atoms * atomic_weight_hydrogen
  let weight_oxygen := total_molecular_weight - weight_carbon_hydrogen
  weight_oxygen / atomic_weight_oxygen = 1 := by sorry

end NUMINAMATH_CALUDE_oxygen_atoms_in_compound_l4108_410829


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4108_410890

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + a₃ + a₅ = 123 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4108_410890


namespace NUMINAMATH_CALUDE_min_abs_z_with_constraint_l4108_410858

theorem min_abs_z_with_constraint (z : ℂ) (h : Complex.abs (z - 2*Complex.I) + Complex.abs (z - 5) = 7) :
  Complex.abs z ≥ 20 * Real.sqrt 29 / 29 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_z_with_constraint_l4108_410858


namespace NUMINAMATH_CALUDE_sarahs_bread_shop_profit_l4108_410805

/-- Sarah's bread shop profit calculation --/
theorem sarahs_bread_shop_profit :
  ∀ (total_loaves : ℕ) 
    (cost_per_loaf morning_price afternoon_price evening_price : ℚ)
    (morning_fraction afternoon_fraction : ℚ),
  total_loaves = 60 →
  cost_per_loaf = 1 →
  morning_price = 3 →
  afternoon_price = 3/2 →
  evening_price = 1 →
  morning_fraction = 1/3 →
  afternoon_fraction = 3/4 →
  let morning_sales := (total_loaves : ℚ) * morning_fraction * morning_price
  let remaining_after_morning := total_loaves - (total_loaves : ℚ) * morning_fraction
  let afternoon_sales := remaining_after_morning * afternoon_fraction * afternoon_price
  let remaining_after_afternoon := remaining_after_morning - remaining_after_morning * afternoon_fraction
  let evening_sales := remaining_after_afternoon * evening_price
  let total_revenue := morning_sales + afternoon_sales + evening_sales
  let total_cost := (total_loaves : ℚ) * cost_per_loaf
  let profit := total_revenue - total_cost
  profit = 55 := by
sorry


end NUMINAMATH_CALUDE_sarahs_bread_shop_profit_l4108_410805


namespace NUMINAMATH_CALUDE_system_solutions_l4108_410867

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  (x + y - 2018 = (x - 2019) * y) ∧
  (x + z - 2014 = (x - 2019) * z) ∧
  (y + z + 2 = y * z)

-- State the theorem
theorem system_solutions :
  (∃ (x y z : ℝ), system x y z ∧ x = 2022 ∧ y = 2 ∧ z = 4) ∧
  (∃ (x y z : ℝ), system x y z ∧ x = 2017 ∧ y = 0 ∧ z = -2) ∧
  (∀ (x y z : ℝ), system x y z → (x = 2022 ∧ y = 2 ∧ z = 4) ∨ (x = 2017 ∧ y = 0 ∧ z = -2)) :=
by sorry


end NUMINAMATH_CALUDE_system_solutions_l4108_410867


namespace NUMINAMATH_CALUDE_minimum_contribution_l4108_410872

theorem minimum_contribution 
  (n : ℕ) 
  (total : ℝ) 
  (max_individual : ℝ) 
  (h1 : n = 15) 
  (h2 : total = 30) 
  (h3 : max_individual = 16) : 
  ∃ (min_contribution : ℝ), 
    (∀ (i : ℕ), i ≤ n → min_contribution ≤ max_individual) ∧ 
    (n * min_contribution ≤ total) ∧ 
    (∀ (x : ℝ), (∀ (i : ℕ), i ≤ n → x ≤ max_individual) ∧ (n * x ≤ total) → x ≤ min_contribution) ∧
    min_contribution = 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_contribution_l4108_410872


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l4108_410898

-- Define the group
def total_boys : ℕ := 5
def total_girls : ℕ := 3

-- Define the events
def exactly_one_boy (selected_boys : ℕ) : Prop := selected_boys = 1
def exactly_two_girls (selected_girls : ℕ) : Prop := selected_girls = 2

-- Define the sample space
def sample_space : Set (ℕ × ℕ) :=
  {pair | pair.1 + pair.2 = 2 ∧ pair.1 ≤ total_boys ∧ pair.2 ≤ total_girls}

-- Theorem to prove
theorem events_mutually_exclusive_not_contradictory :
  (∃ (pair : ℕ × ℕ), pair ∈ sample_space ∧ exactly_one_boy pair.1 ∧ ¬exactly_two_girls pair.2) ∧
  (∃ (pair : ℕ × ℕ), pair ∈ sample_space ∧ ¬exactly_one_boy pair.1 ∧ exactly_two_girls pair.2) ∧
  (¬∃ (pair : ℕ × ℕ), pair ∈ sample_space ∧ exactly_one_boy pair.1 ∧ exactly_two_girls pair.2) :=
by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l4108_410898


namespace NUMINAMATH_CALUDE_coconut_grove_theorem_l4108_410828

theorem coconut_grove_theorem (x : ℝ) : 
  ((x + 4) * 60 + x * 120 + (x - 4) * 180) / (3 * x) = 100 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_theorem_l4108_410828


namespace NUMINAMATH_CALUDE_function_properties_l4108_410897

/-- The given function f(x) -/
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ) + 1

/-- Theorem stating the properties of the function f -/
theorem function_properties (A ω φ : ℝ) (h1 : A > 0) (h2 : ω > 0) (h3 : -π/2 ≤ φ ∧ φ ≤ π/2) :
  (∀ x, f A ω φ x = f A ω φ (2*π/3 - x)) → -- Symmetry about x = π/3
  (∃ x, f A ω φ x = 3) → -- Maximum value is 3
  (∀ x, f A ω φ x = f A ω φ (x + π)) → -- Distance between highest points is π
  (∃ θ, f A ω φ (θ/2 + π/3) = 7/5) →
  (∀ x, f A ω φ x = f A ω φ (x + π)) ∧ -- Smallest positive period is π
  (∀ x, f A ω φ x = 2 * Real.sin (2*x - π/6) + 1) ∧ -- Analytical expression
  (∀ θ, f A ω φ (θ/2 + π/3) = 7/5 → Real.sin θ = 2*Real.sqrt 6/5 ∨ Real.sin θ = -2*Real.sqrt 6/5) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4108_410897


namespace NUMINAMATH_CALUDE_spring_festival_gala_arrangements_l4108_410837

/-- The number of ways to insert 3 distinct objects into 11 spaces,
    such that no two inserted objects are adjacent. -/
def insert_non_adjacent (n m : ℕ) : ℕ :=
  Nat.descFactorial (n + 1) m

theorem spring_festival_gala_arrangements : insert_non_adjacent 10 3 = 990 := by
  sorry

end NUMINAMATH_CALUDE_spring_festival_gala_arrangements_l4108_410837


namespace NUMINAMATH_CALUDE_finn_bought_12_boxes_l4108_410824

/-- The cost of one package of index cards -/
def index_card_cost : ℚ := (55.40 - 15 * 1.85) / 7

/-- The number of boxes of paper clips Finn bought -/
def finn_paper_clips : ℚ := (61.70 - 10 * index_card_cost) / 1.85

theorem finn_bought_12_boxes :
  finn_paper_clips = 12 := by sorry

end NUMINAMATH_CALUDE_finn_bought_12_boxes_l4108_410824


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4108_410815

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 12 → b = 5 → c^2 = a^2 + b^2 → c = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l4108_410815


namespace NUMINAMATH_CALUDE_profit_margin_increase_l4108_410857

theorem profit_margin_increase (initial_margin : ℝ) (final_margin : ℝ) : 
  initial_margin = 0.25 →
  final_margin = 0.40 →
  let initial_price := 1 + initial_margin
  let final_price := 1 + final_margin
  (final_price / initial_price - 1) * 100 = 12 := by
sorry

end NUMINAMATH_CALUDE_profit_margin_increase_l4108_410857


namespace NUMINAMATH_CALUDE_larger_triangle_perimeter_l4108_410863

-- Define the original right triangle
def original_triangle (a b c : ℝ) : Prop :=
  a = 8 ∧ b = 15 ∧ c^2 = a^2 + b^2

-- Define the similarity ratio
def similarity_ratio (k : ℝ) (a : ℝ) : Prop :=
  k * a = 20 ∧ k > 0

-- Define the larger similar triangle
def larger_triangle (a b c k : ℝ) : Prop :=
  original_triangle a b c ∧ similarity_ratio k a

-- Theorem statement
theorem larger_triangle_perimeter 
  (a b c k : ℝ) 
  (h : larger_triangle a b c k) : 
  k * (a + b + c) = 100 := by
    sorry


end NUMINAMATH_CALUDE_larger_triangle_perimeter_l4108_410863


namespace NUMINAMATH_CALUDE_oil_leak_total_l4108_410887

/-- The total amount of oil leaked from three pipes -/
def total_oil_leaked (pipe1_before pipe1_during pipe2_before pipe2_during pipe3_before pipe3_rate pipe3_hours : ℕ) : ℕ :=
  pipe1_before + pipe1_during + pipe2_before + pipe2_during + pipe3_before + pipe3_rate * pipe3_hours

/-- Theorem stating that the total amount of oil leaked is 32,975 liters -/
theorem oil_leak_total :
  total_oil_leaked 6522 2443 8712 3894 9654 250 7 = 32975 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_total_l4108_410887


namespace NUMINAMATH_CALUDE_cubic_function_property_l4108_410848

/-- Given a cubic function f(x) with certain properties, prove that f(1) has specific values -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => (1/3) * x^3 + a^2 * x^2 + a * x + b
  (f (-1) = -7/12 ∧ (λ x => x^2 + 2*a^2*x + a) (-1) = 0) → 
  (f 1 = 25/12 ∨ f 1 = 1/12) := by
sorry


end NUMINAMATH_CALUDE_cubic_function_property_l4108_410848


namespace NUMINAMATH_CALUDE_probability_of_quarter_l4108_410856

def quarter_value : ℚ := 25 / 100
def nickel_value : ℚ := 5 / 100
def penny_value : ℚ := 1 / 100

def total_quarter_value : ℚ := 10
def total_nickel_value : ℚ := 5
def total_penny_value : ℚ := 15

def num_quarters : ℕ := (total_quarter_value / quarter_value).num.toNat
def num_nickels : ℕ := (total_nickel_value / nickel_value).num.toNat
def num_pennies : ℕ := (total_penny_value / penny_value).num.toNat

def total_coins : ℕ := num_quarters + num_nickels + num_pennies

theorem probability_of_quarter : 
  (num_quarters : ℚ) / total_coins = 1 / 41 := by sorry

end NUMINAMATH_CALUDE_probability_of_quarter_l4108_410856


namespace NUMINAMATH_CALUDE_problem_solution_l4108_410861

theorem problem_solution (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4108_410861


namespace NUMINAMATH_CALUDE_polar_equivalence_l4108_410839

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Checks if two polar points are equivalent -/
def polar_equivalent (p1 p2 : PolarPoint) : Prop :=
  p1.r * (Real.cos p1.θ) = p2.r * (Real.cos p2.θ) ∧
  p1.r * (Real.sin p1.θ) = p2.r * (Real.sin p2.θ)

theorem polar_equivalence :
  let p1 : PolarPoint := ⟨6, 4*Real.pi/3⟩
  let p2 : PolarPoint := ⟨-6, Real.pi/3⟩
  polar_equivalent p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_polar_equivalence_l4108_410839


namespace NUMINAMATH_CALUDE_negative_three_squared_times_negative_one_third_cubed_l4108_410884

theorem negative_three_squared_times_negative_one_third_cubed :
  -3^2 * (-1/3)^3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_times_negative_one_third_cubed_l4108_410884


namespace NUMINAMATH_CALUDE_inequality_proof_l4108_410871

theorem inequality_proof : (-abs (abs (-20 : ℝ))) / 2 > -4.5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4108_410871


namespace NUMINAMATH_CALUDE_x_minus_y_value_l4108_410847

theorem x_minus_y_value (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 20) : x - y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l4108_410847


namespace NUMINAMATH_CALUDE_complement_union_theorem_l4108_410866

def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x : ℝ | x < 6}

theorem complement_union_theorem :
  (Set.univ \ B) ∪ A = {x : ℝ | x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l4108_410866


namespace NUMINAMATH_CALUDE_exists_special_multiple_l4108_410810

/-- A function that returns true if all digits of a natural number are in the set {0, 1, 8, 9} -/
def valid_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ ({0, 1, 8, 9} : Set ℕ)

/-- The main theorem stating the existence of a number with the required properties -/
theorem exists_special_multiple : ∃ n : ℕ, 
  2003 ∣ n ∧ n < 10^11 ∧ valid_digits n :=
sorry

end NUMINAMATH_CALUDE_exists_special_multiple_l4108_410810


namespace NUMINAMATH_CALUDE_concrete_slab_height_l4108_410889

/-- Proves that the height of each concrete slab is 0.5 feet given the specified conditions --/
theorem concrete_slab_height :
  let num_homes : ℕ := 3
  let slab_length : ℝ := 100
  let slab_width : ℝ := 100
  let concrete_density : ℝ := 150
  let concrete_cost_per_pound : ℝ := 0.02
  let total_foundation_cost : ℝ := 45000

  let total_weight : ℝ := total_foundation_cost / concrete_cost_per_pound
  let total_volume : ℝ := total_weight / concrete_density
  let volume_per_home : ℝ := total_volume / num_homes
  let slab_area : ℝ := slab_length * slab_width
  let slab_height : ℝ := volume_per_home / slab_area

  slab_height = 0.5 := by sorry

end NUMINAMATH_CALUDE_concrete_slab_height_l4108_410889


namespace NUMINAMATH_CALUDE_sports_club_membership_l4108_410894

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 28 →
  badminton = 17 →
  tennis = 19 →
  both = 10 →
  total - (badminton + tennis - both) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_club_membership_l4108_410894
