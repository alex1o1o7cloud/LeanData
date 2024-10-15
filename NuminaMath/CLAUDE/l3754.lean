import Mathlib

namespace NUMINAMATH_CALUDE_power_inequality_l3754_375402

theorem power_inequality (a b t x : ℝ) 
  (h1 : b > a) (h2 : a > 1) (h3 : t > 0) (h4 : a^x = a + t) : b^x > b + t := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3754_375402


namespace NUMINAMATH_CALUDE_newer_truck_travels_195_miles_l3754_375459

/-- The distance traveled by the older truck in miles -/
def older_truck_distance : ℝ := 150

/-- The percentage increase in distance for the newer truck -/
def newer_truck_percentage : ℝ := 0.30

/-- The distance traveled by the newer truck in miles -/
def newer_truck_distance : ℝ := older_truck_distance * (1 + newer_truck_percentage)

/-- Theorem stating that the newer truck travels 195 miles -/
theorem newer_truck_travels_195_miles :
  newer_truck_distance = 195 := by sorry

end NUMINAMATH_CALUDE_newer_truck_travels_195_miles_l3754_375459


namespace NUMINAMATH_CALUDE_min_races_for_top_five_l3754_375485

/-- Represents a horse in the race. -/
structure Horse :=
  (id : Nat)

/-- Represents a race with up to 4 horses. -/
structure Race :=
  (participants : Finset Horse)
  (hLimited : participants.card ≤ 4)

/-- Represents the outcome of a series of races. -/
structure RaceOutcome :=
  (races : List Race)
  (topFive : Finset Horse)
  (hTopFiveSize : topFive.card = 5)

/-- The main theorem stating the minimum number of races required. -/
theorem min_races_for_top_five (horses : Finset Horse) 
  (hSize : horses.card = 30) :
  ∃ (outcome : RaceOutcome), 
    outcome.topFive ⊆ horses ∧ 
    outcome.races.length = 8 ∧ 
    (∀ (alt_outcome : RaceOutcome), 
      alt_outcome.topFive ⊆ horses → 
      alt_outcome.races.length ≥ 8) :=
sorry

end NUMINAMATH_CALUDE_min_races_for_top_five_l3754_375485


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l3754_375487

theorem units_digit_of_expression : 
  ((5 * 21 * 1933) + 5^4 - (6 * 2 * 1944)) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l3754_375487


namespace NUMINAMATH_CALUDE_subcommittee_count_l3754_375430

theorem subcommittee_count (n m k t : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 5) (h4 : t = 5) :
  (Nat.choose n k) - (Nat.choose (n - t) k) = 771 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_count_l3754_375430


namespace NUMINAMATH_CALUDE_evenly_spaced_poles_l3754_375467

/-- Given five evenly spaced poles along a straight road, 
    if the distance between the second and fifth poles is 90 feet, 
    then the distance between the first and fifth poles is 120 feet. -/
theorem evenly_spaced_poles (n : ℕ) (d : ℝ) (h1 : n = 5) (h2 : d = 90) :
  let pole_distance (i j : ℕ) := d * (j - i) / 3
  pole_distance 1 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_evenly_spaced_poles_l3754_375467


namespace NUMINAMATH_CALUDE_cyclic_inequality_l3754_375453

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / (a + b)) + (b^2 / (b + c)) + (c^2 / (c + a)) ≥ (a + b + c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l3754_375453


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l3754_375446

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 160) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 975 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l3754_375446


namespace NUMINAMATH_CALUDE_negation_equivalence_l3754_375489

theorem negation_equivalence (a : ℝ) : 
  (¬∃ x ∈ Set.Icc 1 2, x^2 - a < 0) ↔ (∀ x ∈ Set.Icc 1 2, x^2 ≥ a) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3754_375489


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3754_375435

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 3 ↔ (x : ℚ) / 4 + 3 / 7 < 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3754_375435


namespace NUMINAMATH_CALUDE_carlos_summer_reading_l3754_375424

/-- The number of books Carlos read in summer vacation --/
def total_books : ℕ := 100

/-- The number of books Carlos read in July --/
def july_books : ℕ := 28

/-- The number of books Carlos read in August --/
def august_books : ℕ := 30

/-- The number of books Carlos read in June --/
def june_books : ℕ := total_books - (july_books + august_books)

theorem carlos_summer_reading : june_books = 42 := by
  sorry

end NUMINAMATH_CALUDE_carlos_summer_reading_l3754_375424


namespace NUMINAMATH_CALUDE_recipe_ratio_change_l3754_375401

-- Define the original recipe ratios
def original_flour : ℚ := 8
def original_water : ℚ := 4
def original_sugar : ℚ := 3

-- Define the new recipe quantities
def new_water : ℚ := 2
def new_sugar : ℚ := 6

-- Theorem statement
theorem recipe_ratio_change :
  let original_flour_sugar_ratio := original_flour / original_sugar
  let new_flour := (original_flour / original_water) * 2 * new_water
  let new_flour_sugar_ratio := new_flour / new_sugar
  original_flour_sugar_ratio - new_flour_sugar_ratio = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_recipe_ratio_change_l3754_375401


namespace NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l3754_375445

theorem parallelogram_altitude_base_ratio 
  (area : ℝ) 
  (base : ℝ) 
  (altitude : ℝ) 
  (h1 : area = 450) 
  (h2 : base = 15) 
  (h3 : area = base * altitude) 
  (h4 : ∃ k : ℝ, altitude = k * base) : 
  altitude / base = 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l3754_375445


namespace NUMINAMATH_CALUDE_solution_set_f_leq_5_smallest_a_for_inequality_l3754_375437

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 3| + |x + 2|

-- Theorem 1: The solution set of f(x) ≤ 5 is [0, 2]
theorem solution_set_f_leq_5 : 
  {x : ℝ | f x ≤ 5} = Set.Icc 0 2 := by sorry

-- Theorem 2: The smallest value of a such that f(x) ≤ a - |x| for all x in [-1, 2] is 7
theorem smallest_a_for_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), x ∈ Set.Icc (-1) 2 → f x ≤ a - |x|) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x ∈ Set.Icc (-1) 2 → f x ≤ a - |x|) → a ≥ 7) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_5_smallest_a_for_inequality_l3754_375437


namespace NUMINAMATH_CALUDE_complete_factorization_l3754_375408

theorem complete_factorization (x : ℝ) : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_complete_factorization_l3754_375408


namespace NUMINAMATH_CALUDE_a_66_mod_55_l3754_375491

/-- Definition of a_n as the integer obtained by writing all integers from 1 to n from left to right -/
def a (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that a_66 is congruent to 51 modulo 55 -/
theorem a_66_mod_55 : a 66 ≡ 51 [ZMOD 55] := by
  sorry

end NUMINAMATH_CALUDE_a_66_mod_55_l3754_375491


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3754_375418

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a, if a₁ + 3a₈ + a₁₅ = 120, then a₈ = 24 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a) 
    (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3754_375418


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3754_375426

def silverware_cost : ℝ := 20
def dinner_plates_cost_percentage : ℝ := 0.5

theorem total_cost_calculation :
  let dinner_plates_cost := silverware_cost * dinner_plates_cost_percentage
  let total_cost := silverware_cost + dinner_plates_cost
  total_cost = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3754_375426


namespace NUMINAMATH_CALUDE_sum_of_smallest_multiples_l3754_375494

/-- The smallest positive two-digit multiple of 5 -/
def c : ℕ := sorry

/-- The smallest positive three-digit multiple of 7 -/
def d : ℕ := sorry

/-- c is a two-digit number -/
axiom c_two_digit : 10 ≤ c ∧ c ≤ 99

/-- d is a three-digit number -/
axiom d_three_digit : 100 ≤ d ∧ d ≤ 999

/-- c is a multiple of 5 -/
axiom c_multiple_of_5 : ∃ k : ℕ, c = 5 * k

/-- d is a multiple of 7 -/
axiom d_multiple_of_7 : ∃ k : ℕ, d = 7 * k

/-- c is the smallest two-digit multiple of 5 -/
axiom c_smallest : ∀ x : ℕ, (10 ≤ x ∧ x ≤ 99 ∧ ∃ k : ℕ, x = 5 * k) → c ≤ x

/-- d is the smallest three-digit multiple of 7 -/
axiom d_smallest : ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 ∧ ∃ k : ℕ, x = 7 * k) → d ≤ x

theorem sum_of_smallest_multiples : c + d = 115 := by sorry

end NUMINAMATH_CALUDE_sum_of_smallest_multiples_l3754_375494


namespace NUMINAMATH_CALUDE_machine_work_time_l3754_375490

theorem machine_work_time (time_A time_B time_ABC : ℚ) (time_C : ℚ) : 
  time_A = 4 → time_B = 2 → time_ABC = 12/11 → 
  1/time_A + 1/time_B + 1/time_C = 1/time_ABC → 
  time_C = 6 := by sorry

end NUMINAMATH_CALUDE_machine_work_time_l3754_375490


namespace NUMINAMATH_CALUDE_complex_inequality_l3754_375419

theorem complex_inequality (x y a b : ℝ) 
  (h1 : x^2 + y^2 ≤ 1) 
  (h2 : a^2 + b^2 ≤ 2) : 
  |b * (x^2 - y^2) + 2 * a * x * y| ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l3754_375419


namespace NUMINAMATH_CALUDE_complex_calculation_l3754_375464

theorem complex_calculation (a b : ℂ) (ha : a = 3 + 2*I) (hb : b = -2 + I) :
  4*a - 2*b = 16 + 6*I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l3754_375464


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3754_375482

theorem simplify_and_evaluate (m n : ℤ) (h1 : m = 2) (h2 : n = -1^2023) :
  (2*m + n) * (2*m - n) - (2*m - n)^2 + 2*n*(m + n) = -12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3754_375482


namespace NUMINAMATH_CALUDE_inequality_proof_l3754_375427

theorem inequality_proof (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  x / (1 + y) + y / (1 + x) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3754_375427


namespace NUMINAMATH_CALUDE_binary_sum_exp_eq_four_l3754_375492

/-- B(n) is the number of ones in the base two expression for the positive integer n -/
def B (n : ℕ+) : ℕ := sorry

/-- The infinite sum of B(n)/(n(n+1)) for n from 1 to infinity -/
noncomputable def infiniteSum : ℝ := sorry

theorem binary_sum_exp_eq_four :
  Real.exp infiniteSum = 4 := by sorry

end NUMINAMATH_CALUDE_binary_sum_exp_eq_four_l3754_375492


namespace NUMINAMATH_CALUDE_exactly_one_pair_probability_l3754_375475

def number_of_pairs : ℕ := 8
def shoes_drawn : ℕ := 4

def total_outcomes : ℕ := (Nat.choose (2 * number_of_pairs) shoes_drawn)

def favorable_outcomes : ℕ :=
  (Nat.choose number_of_pairs 1) *
  (Nat.choose (number_of_pairs - 1) 2) *
  (Nat.choose 2 1) *
  (Nat.choose 2 1)

theorem exactly_one_pair_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 24 / 65 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_pair_probability_l3754_375475


namespace NUMINAMATH_CALUDE_goals_scored_theorem_l3754_375413

/-- The number of goals scored by Bruce and Michael -/
def total_goals (bruce_goals : ℕ) (michael_multiplier : ℕ) : ℕ :=
  bruce_goals + michael_multiplier * bruce_goals

/-- Theorem stating that Bruce and Michael scored 16 goals in total -/
theorem goals_scored_theorem :
  total_goals 4 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_goals_scored_theorem_l3754_375413


namespace NUMINAMATH_CALUDE_grain_spilled_calculation_l3754_375484

def original_grain : ℕ := 50870
def remaining_grain : ℕ := 918

theorem grain_spilled_calculation : original_grain - remaining_grain = 49952 := by
  sorry

end NUMINAMATH_CALUDE_grain_spilled_calculation_l3754_375484


namespace NUMINAMATH_CALUDE_equation_solutions_l3754_375498

theorem equation_solutions :
  (∃ x : ℝ, x - 4 = -5 ∧ x = -1) ∧
  (∃ x : ℝ, (1/2) * x + 2 = 6 ∧ x = 8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3754_375498


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l3754_375496

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 4) (hh : h = 10) :
  Real.sqrt ((2 * r)^2 + h^2) = Real.sqrt 164 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l3754_375496


namespace NUMINAMATH_CALUDE_prime_counterexample_l3754_375406

theorem prime_counterexample : ∃ n : ℕ, 
  (Nat.Prime n ∧ ¬Nat.Prime (n + 2)) ∨ (¬Nat.Prime n ∧ Nat.Prime (n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_counterexample_l3754_375406


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3754_375441

theorem polynomial_divisibility (C D : ℂ) : 
  (∀ x : ℂ, x^2 - x + 1 = 0 → x^103 + C*x^2 + D*x + 1 = 0) →
  C = -1 ∧ D = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3754_375441


namespace NUMINAMATH_CALUDE_prob_diff_is_one_third_l3754_375450

/-- The number of marbles of each color in the box -/
def marbles_per_color : ℕ := 1500

/-- The total number of marbles in the box -/
def total_marbles : ℕ := 3 * marbles_per_color

/-- The probability of drawing two marbles of the same color -/
def prob_same_color : ℚ :=
  (3 * (marbles_per_color.choose 2)) / (total_marbles.choose 2)

/-- The probability of drawing two marbles of different colors -/
def prob_diff_color : ℚ :=
  (3 * marbles_per_color * marbles_per_color) / (total_marbles.choose 2)

/-- The theorem stating that the absolute difference between the probabilities is 1/3 -/
theorem prob_diff_is_one_third :
  |prob_same_color - prob_diff_color| = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_diff_is_one_third_l3754_375450


namespace NUMINAMATH_CALUDE_negative_64_to_four_thirds_l3754_375499

theorem negative_64_to_four_thirds : (-64 : ℝ) ^ (4/3) = 256 := by sorry

end NUMINAMATH_CALUDE_negative_64_to_four_thirds_l3754_375499


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3754_375471

/-- Given a line segment with one endpoint (5, -2) and midpoint (3, 4),
    the sum of coordinates of the other endpoint is 11. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (5 + x) / 2 = 3 → 
    (-2 + y) / 2 = 4 → 
    x + y = 11 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3754_375471


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3754_375414

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 + Complex.I → z = 1 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3754_375414


namespace NUMINAMATH_CALUDE_complex_arithmetic_l3754_375409

theorem complex_arithmetic : ((2 : ℂ) + 5*I + (3 : ℂ) - 2*I) - ((1 : ℂ) - 3*I) = (4 : ℂ) + 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l3754_375409


namespace NUMINAMATH_CALUDE_log_equation_solution_l3754_375411

theorem log_equation_solution (x : ℝ) : Real.log (256 : ℝ) / Real.log (3 * x) = x → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3754_375411


namespace NUMINAMATH_CALUDE_sixth_power_sum_l3754_375466

theorem sixth_power_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l3754_375466


namespace NUMINAMATH_CALUDE_keystone_arch_angle_theorem_l3754_375440

/-- Represents a keystone arch made of congruent isosceles trapezoids -/
structure KeystoneArch where
  num_trapezoids : ℕ
  trapezoids_congruent : Bool
  trapezoids_isosceles : Bool
  end_trapezoids_horizontal : Bool

/-- Calculate the larger interior angle of a trapezoid in a keystone arch -/
def larger_interior_angle (arch : KeystoneArch) : ℝ :=
  if arch.num_trapezoids = 9 ∧ 
     arch.trapezoids_congruent ∧ 
     arch.trapezoids_isosceles ∧ 
     arch.end_trapezoids_horizontal
  then 100
  else 0

/-- Theorem: The larger interior angle of each trapezoid in a keystone arch 
    with 9 congruent isosceles trapezoids is 100 degrees -/
theorem keystone_arch_angle_theorem (arch : KeystoneArch) :
  arch.num_trapezoids = 9 ∧ 
  arch.trapezoids_congruent ∧ 
  arch.trapezoids_isosceles ∧ 
  arch.end_trapezoids_horizontal →
  larger_interior_angle arch = 100 := by
  sorry

end NUMINAMATH_CALUDE_keystone_arch_angle_theorem_l3754_375440


namespace NUMINAMATH_CALUDE_lizzy_shipment_cost_l3754_375422

/-- Calculates the total shipment cost for Lizzy's fish shipment --/
def total_shipment_cost (total_weight type_a_capacity type_b_capacity : ℕ)
  (type_a_cost type_b_cost surcharge flat_fee : ℚ)
  (num_type_a : ℕ) : ℚ :=
  let type_a_total_weight := num_type_a * type_a_capacity
  let type_b_total_weight := total_weight - type_a_total_weight
  let num_type_b := (type_b_total_weight + type_b_capacity - 1) / type_b_capacity
  let type_a_total_cost := num_type_a * (type_a_cost + surcharge)
  let type_b_total_cost := num_type_b * (type_b_cost + surcharge)
  type_a_total_cost + type_b_total_cost + flat_fee

theorem lizzy_shipment_cost :
  total_shipment_cost 540 30 50 (3/2) (5/2) (1/2) 10 6 = 46 :=
by sorry

end NUMINAMATH_CALUDE_lizzy_shipment_cost_l3754_375422


namespace NUMINAMATH_CALUDE_direction_vector_b_value_l3754_375463

/-- Given a line passing through two points, prove that its direction vector
    in the form (b, -1) has b = 1. -/
theorem direction_vector_b_value 
  (p1 p2 : ℝ × ℝ) 
  (h1 : p1 = (-3, 2)) 
  (h2 : p2 = (2, -3)) 
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ (p2.1 - p1.1, p2.2 - p1.2) = (k * b, k * (-1))) : 
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_b_value_l3754_375463


namespace NUMINAMATH_CALUDE_quadratic_sequence_exists_smallest_n_for_specific_sequence_l3754_375473

/-- A sequence is quadratic if the absolute difference between consecutive terms is the square of their index. -/
def IsQuadraticSequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i ≥ 1 ∧ i ≤ n → |a i - a (i-1)| = i^2

theorem quadratic_sequence_exists (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ IsQuadraticSequence a n :=
sorry

theorem smallest_n_for_specific_sequence :
  (∃ (a : ℕ → ℤ), a 0 = 0 ∧ a 19 = 1996 ∧ IsQuadraticSequence a 19) ∧
  (∀ n : ℕ, n < 19 → ¬∃ (a : ℕ → ℤ), a 0 = 0 ∧ a n = 1996 ∧ IsQuadraticSequence a n) :=
sorry

end NUMINAMATH_CALUDE_quadratic_sequence_exists_smallest_n_for_specific_sequence_l3754_375473


namespace NUMINAMATH_CALUDE_tetrahedron_vertices_prove_tetrahedron_vertices_l3754_375460

/-- A tetrahedron is a three-dimensional polyhedron with four triangular faces. -/
structure Tetrahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of vertices in a tetrahedron is 4. -/
theorem tetrahedron_vertices (t : Tetrahedron) : Nat :=
  4

#check tetrahedron_vertices

/-- Prove that a tetrahedron has 4 vertices. -/
theorem prove_tetrahedron_vertices (t : Tetrahedron) : tetrahedron_vertices t = 4 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_vertices_prove_tetrahedron_vertices_l3754_375460


namespace NUMINAMATH_CALUDE_salmon_migration_l3754_375465

theorem salmon_migration (male_salmon female_salmon : ℕ) 
  (h1 : male_salmon = 712261)
  (h2 : female_salmon = 259378) :
  male_salmon + female_salmon = 971639 := by
  sorry

end NUMINAMATH_CALUDE_salmon_migration_l3754_375465


namespace NUMINAMATH_CALUDE_quadratic_equation_1_quadratic_equation_2_quadratic_equation_3_quadratic_equation_4_l3754_375457

-- Problem 1
theorem quadratic_equation_1 (x : ℝ) : 
  (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) → x^2 - 2*x - 2 = 0 := by sorry

-- Problem 2
theorem quadratic_equation_2 (x : ℝ) :
  (x = -4 ∨ x = 1) → (x + 4)^2 = 5*(x + 4) := by sorry

-- Problem 3
theorem quadratic_equation_3 (x : ℝ) :
  (x = (-3 + 2*Real.sqrt 6) / 3 ∨ x = (-3 - 2*Real.sqrt 6) / 3) → 3*x^2 + 6*x - 5 = 0 := by sorry

-- Problem 4
theorem quadratic_equation_4 (x : ℝ) :
  (x = (-1 + Real.sqrt 5) / 4 ∨ x = (-1 - Real.sqrt 5) / 4) → 4*x^2 + 2*x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_quadratic_equation_2_quadratic_equation_3_quadratic_equation_4_l3754_375457


namespace NUMINAMATH_CALUDE_modular_inverse_45_mod_47_l3754_375493

theorem modular_inverse_45_mod_47 :
  ∃ x : ℕ, x ≤ 46 ∧ (45 * x) % 47 = 1 ∧ x = 23 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_45_mod_47_l3754_375493


namespace NUMINAMATH_CALUDE_locus_and_tangent_lines_l3754_375472

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point M on the ellipse
def M : ℝ × ℝ := sorry

-- Define point N as the projection of M on x = 3
def N : ℝ × ℝ := (3, M.2)

-- Define point P
def P : ℝ × ℝ := sorry

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define vector addition
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector from O to a point
def vector_to (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - O.1, p.2 - O.2)

-- Define the locus E
def E (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define point A
def A : ℝ × ℝ := (1, 4)

-- Define the tangent line equations
def tangent_line_1 (x y : ℝ) : Prop := x = 1
def tangent_line_2 (x y : ℝ) : Prop := 3*x + 4*y - 19 = 0

theorem locus_and_tangent_lines :
  ellipse M.1 M.2 ∧
  N = (3, M.2) ∧
  vector_to P = vector_add (vector_to M) (vector_to N) →
  (∀ x y, E x y ↔ (∃ m n, ellipse m n ∧ x = m + 3 ∧ y = 2*n)) ∧
  (∀ x y, (tangent_line_1 x y ∨ tangent_line_2 x y) ↔
    (E x y ∧ (x - A.1)^2 + (y - A.2)^2 = ((x - 3)^2 + y^2))) :=
sorry

end NUMINAMATH_CALUDE_locus_and_tangent_lines_l3754_375472


namespace NUMINAMATH_CALUDE_least_integer_square_triple_plus_80_l3754_375405

theorem least_integer_square_triple_plus_80 :
  ∃ x : ℤ, (∀ y : ℤ, y^2 = 3*y + 80 → x ≤ y) ∧ x^2 = 3*x + 80 :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_square_triple_plus_80_l3754_375405


namespace NUMINAMATH_CALUDE_opposite_reciprocal_sum_l3754_375458

theorem opposite_reciprocal_sum (a b c d : ℝ) (m : ℕ) : 
  b ≠ 0 →
  a = -b →
  c * d = 1 →
  m < 2 →
  (m : ℝ) - c * d + (a + b) / 2023 + a / b = -2 ∨ 
  (m : ℝ) - c * d + (a + b) / 2023 + a / b = -1 :=
by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_sum_l3754_375458


namespace NUMINAMATH_CALUDE_tower_remainder_l3754_375434

/-- Represents the number of towers that can be built with cubes up to size n -/
def T : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 1 => if n ≥ 2 then 3 * T n else 2 * T n

/-- The main theorem stating the remainder when T(9) is divided by 500 -/
theorem tower_remainder : T 9 % 500 = 374 := by
  sorry

end NUMINAMATH_CALUDE_tower_remainder_l3754_375434


namespace NUMINAMATH_CALUDE_parabola_symmetric_axis_given_parabola_symmetric_axis_l3754_375436

/-- The symmetric axis of a parabola y = (x - h)^2 + k is x = h -/
theorem parabola_symmetric_axis (h k : ℝ) :
  let f : ℝ → ℝ := λ x => (x - h)^2 + k
  ∃! a : ℝ, ∀ x y : ℝ, f x = f y → |x - a| = |y - a| :=
by sorry

/-- The symmetric axis of the parabola y = (x - 2)^2 + 1 is x = 2 -/
theorem given_parabola_symmetric_axis :
  let f : ℝ → ℝ := λ x => (x - 2)^2 + 1
  ∃! a : ℝ, ∀ x y : ℝ, f x = f y → |x - a| = |y - a| ∧ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_axis_given_parabola_symmetric_axis_l3754_375436


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l3754_375448

/-- Proves that a boat with given characteristics travels 500 km downstream in 5 hours -/
theorem boat_downstream_distance
  (boat_speed : ℝ)
  (upstream_distance : ℝ)
  (upstream_time : ℝ)
  (downstream_time : ℝ)
  (h_boat_speed : boat_speed = 70)
  (h_upstream_distance : upstream_distance = 240)
  (h_upstream_time : upstream_time = 6)
  (h_downstream_time : downstream_time = 5)
  : ∃ (stream_speed : ℝ),
    stream_speed > 0 ∧
    upstream_distance / upstream_time = boat_speed - stream_speed ∧
    downstream_time * (boat_speed + stream_speed) = 500 :=
by sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l3754_375448


namespace NUMINAMATH_CALUDE_sum_and_equality_problem_l3754_375410

theorem sum_and_equality_problem (x y z : ℚ) : 
  x + y + z = 150 ∧ x + 10 = y - 10 ∧ y - 10 = 6 * z → y = 1030 / 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_equality_problem_l3754_375410


namespace NUMINAMATH_CALUDE_dinner_cakes_count_l3754_375449

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := 5

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served over two days -/
def total_cakes : ℕ := 14

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := total_cakes - lunch_cakes - yesterday_cakes

theorem dinner_cakes_count : dinner_cakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cakes_count_l3754_375449


namespace NUMINAMATH_CALUDE_statement_relationship_l3754_375468

theorem statement_relationship :
  (∀ x : ℝ, x^2 - 5*x < 0 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ x^2 - 5*x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_statement_relationship_l3754_375468


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3754_375451

theorem geometric_sequence_condition (a b c : ℝ) : 
  (b^2 ≠ a*c → ¬(∃ r : ℝ, b = a*r ∧ c = b*r)) ∧ 
  (∃ a b c : ℝ, ¬(∃ r : ℝ, b = a*r ∧ c = b*r) ∧ b^2 = a*c) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l3754_375451


namespace NUMINAMATH_CALUDE_cars_meeting_time_l3754_375474

theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) (h1 : highway_length = 333)
  (h2 : speed1 = 54) (h3 : speed2 = 57) : 
  (highway_length / (speed1 + speed2)) = 3 := by
sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l3754_375474


namespace NUMINAMATH_CALUDE_book_pages_theorem_l3754_375469

def pages_read_day1 (total : ℕ) : ℕ :=
  total / 4 + 20

def pages_left_day1 (total : ℕ) : ℕ :=
  total - pages_read_day1 total

def pages_read_day2 (total : ℕ) : ℕ :=
  (pages_left_day1 total) / 3 + 25

def pages_left_day2 (total : ℕ) : ℕ :=
  pages_left_day1 total - pages_read_day2 total

def pages_read_day3 (total : ℕ) : ℕ :=
  (pages_left_day2 total) / 2 + 30

def pages_left_day3 (total : ℕ) : ℕ :=
  pages_left_day2 total - pages_read_day3 total

theorem book_pages_theorem :
  pages_left_day3 480 = 70 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l3754_375469


namespace NUMINAMATH_CALUDE_minimum_condition_range_l3754_375470

/-- Given a function f with derivative f'(x) = a(x+1)(x-a), 
    if f attains its minimum at x = a, then a < -1 or a > 0 -/
theorem minimum_condition_range (f : ℝ → ℝ) (a : ℝ) 
  (h_deriv : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h_min : IsLocalMin f a) :
  a < -1 ∨ a > 0 := by
sorry

end NUMINAMATH_CALUDE_minimum_condition_range_l3754_375470


namespace NUMINAMATH_CALUDE_inequality_proof_l3754_375478

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_condition : |Real.sqrt (a * d) - Real.sqrt (b * c)| ≤ 1) : 
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3754_375478


namespace NUMINAMATH_CALUDE_linear_search_average_comparisons_linear_search_most_efficient_l3754_375423

/-- Represents an array with a specific size and a search function. -/
structure SearchArray (α : Type) where
  size : Nat
  elements : Fin size → α
  search : α → Option (Fin size)

/-- Calculates the average number of comparisons for a linear search. -/
def averageLinearSearchComparisons (n : Nat) : ℚ :=
  (1 + n) / 2

/-- Theorem: The average number of comparisons for a linear search
    on an array of 10,000 elements is 5,000.5 when the element is not present. -/
theorem linear_search_average_comparisons :
  averageLinearSearchComparisons 10000 = 5000.5 := by
  sorry

/-- Theorem: Linear search is the most efficient algorithm for an array
    with partial ordering that doesn't allow for more efficient searches. -/
theorem linear_search_most_efficient (α : Type) (arr : SearchArray α) :
  arr.size = 10000 →
  (∃ (p : α → Prop), ∀ (i j : Fin arr.size), i < j → p (arr.elements i) → p (arr.elements j)) →
  (∀ (search : α → Option (Fin arr.size)), 
    (∀ x, search x = arr.search x) →
    ∃ c, ∀ x, (search x).isNone → c ≥ averageLinearSearchComparisons arr.size) := by
  sorry

end NUMINAMATH_CALUDE_linear_search_average_comparisons_linear_search_most_efficient_l3754_375423


namespace NUMINAMATH_CALUDE_garrett_granola_purchase_l3754_375417

/-- Represents the cost of Garrett's granola bar purchase -/
def total_cost (oatmeal_count : ℕ) (oatmeal_price : ℚ) (peanut_count : ℕ) (peanut_price : ℚ) : ℚ :=
  oatmeal_count * oatmeal_price + peanut_count * peanut_price

/-- Proves that Garrett's total granola bar purchase cost is $19.50 -/
theorem garrett_granola_purchase :
  total_cost 6 1.25 8 1.50 = 19.50 := by
  sorry

end NUMINAMATH_CALUDE_garrett_granola_purchase_l3754_375417


namespace NUMINAMATH_CALUDE_system_solution_l3754_375439

theorem system_solution : ∃ (x y : ℝ), 2 * x + y = 7 ∧ 4 * x + 5 * y = 11 :=
by
  use 4, -1
  sorry

end NUMINAMATH_CALUDE_system_solution_l3754_375439


namespace NUMINAMATH_CALUDE_simplify_expression_l3754_375461

theorem simplify_expression (m n : ℝ) (h : m^2 + 3*m*n = 5) :
  5*m^2 - 3*m*n - (-9*m*n + 3*m^2) = 10 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3754_375461


namespace NUMINAMATH_CALUDE_solve_equation_solve_system_l3754_375488

-- Problem 1
theorem solve_equation (x : ℝ) : (x + 1) / 3 - 1 = (x - 1) / 2 → x = -1 := by sorry

-- Problem 2
theorem solve_system (x y : ℝ) : x - y = 1 ∧ 3 * x + y = 7 → x = 2 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_solve_equation_solve_system_l3754_375488


namespace NUMINAMATH_CALUDE_number_problem_l3754_375431

theorem number_problem (x : ℚ) :
  (35 / 100) * x = (25 / 100) * 40 → x = 200 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3754_375431


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l3754_375497

theorem divisibility_implies_equality (a b : ℕ) :
  (4 * a * b - 1) ∣ (4 * a^2 - 1)^2 → a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l3754_375497


namespace NUMINAMATH_CALUDE_two_dice_probabilities_l3754_375412

/-- A fair die roll, represented as a number from 1 to 6 -/
def DieRoll : Type := Fin 6

/-- The probability space of rolling a fair die twice -/
def TwoDiceRolls : Type := DieRoll × DieRoll

/-- The probability measure for rolling a fair die twice -/
noncomputable def P : Set TwoDiceRolls → ℝ := sorry

/-- Event A: The dot product of (m, n) and (2, -2) is positive -/
def eventA : Set TwoDiceRolls :=
  {roll | (roll.1.val + 1) * 2 - (roll.2.val + 1) * 2 > 0}

/-- The region where x^2 + y^2 ≤ 16 -/
def region16 : Set TwoDiceRolls :=
  {roll | (roll.1.val + 1)^2 + (roll.2.val + 1)^2 ≤ 16}

theorem two_dice_probabilities :
  P eventA = 5/12 ∧ P region16 = 2/9 := by sorry

end NUMINAMATH_CALUDE_two_dice_probabilities_l3754_375412


namespace NUMINAMATH_CALUDE_equivalence_complex_inequality_l3754_375428

theorem equivalence_complex_inequality (a b c d : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  (∀ z : ℂ, Complex.abs (z - a) + Complex.abs (z - b) ≥ 
    Complex.abs (z - c) + Complex.abs (z - d)) ↔
  (∃ t : ℝ, t ∈ Set.Ioo 0 1 ∧ 
    c = t • a + (1 - t) • b ∧ 
    d = (1 - t) • a + t • b) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_complex_inequality_l3754_375428


namespace NUMINAMATH_CALUDE_inequality_proof_l3754_375444

theorem inequality_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (a + 1/b)^2 + (b + 1/a)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3754_375444


namespace NUMINAMATH_CALUDE_prime_divisibility_special_primes_characterization_l3754_375429

theorem prime_divisibility (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p ≠ q → 
  (p + q^2) ∣ (p^2 + q) → (p + q^2) ∣ (p*q - 1) := by
  sorry

-- Part b
def special_primes : Set ℕ := {p | Nat.Prime p ∧ (p + 121) ∣ (p^2 + 11)}

-- The theorem states that the set of special primes is equal to {101, 323, 1211}
theorem special_primes_characterization : 
  special_primes = {101, 323, 1211} := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_special_primes_characterization_l3754_375429


namespace NUMINAMATH_CALUDE_lisa_baby_spoons_l3754_375420

/-- Given the total number of spoons, number of children, number of decorative spoons,
    and number of spoons in the new cutlery set, calculate the number of baby spoons per child. -/
def baby_spoons_per_child (total_spoons : ℕ) (num_children : ℕ) (decorative_spoons : ℕ) (new_cutlery_spoons : ℕ) : ℕ :=
  (total_spoons - decorative_spoons - new_cutlery_spoons) / num_children

/-- Prove that given Lisa's specific situation, each child had 3 baby spoons. -/
theorem lisa_baby_spoons : baby_spoons_per_child 39 4 2 25 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lisa_baby_spoons_l3754_375420


namespace NUMINAMATH_CALUDE_range_of_k_for_special_function_l3754_375479

theorem range_of_k_for_special_function (f : ℝ → ℝ) (k a b : ℝ) :
  (∀ x, f x = Real.sqrt (x + 2) + k) →
  a < b →
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y) →
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) →
  k ∈ Set.Ioo (-9/4) (-2) := by
sorry

end NUMINAMATH_CALUDE_range_of_k_for_special_function_l3754_375479


namespace NUMINAMATH_CALUDE_yoongi_has_smallest_number_l3754_375455

def yoongi_number : ℕ := 4
def jungkook_number : ℕ := 6 * 3
def yuna_number : ℕ := 5

theorem yoongi_has_smallest_number : 
  yoongi_number < jungkook_number ∧ yoongi_number < yuna_number :=
sorry

end NUMINAMATH_CALUDE_yoongi_has_smallest_number_l3754_375455


namespace NUMINAMATH_CALUDE_library_fiction_percentage_l3754_375486

theorem library_fiction_percentage 
  (total_volumes : ℕ) 
  (fiction_percentage : ℚ)
  (transfer_fraction : ℚ)
  (fiction_transfer_fraction : ℚ)
  (h1 : total_volumes = 18360)
  (h2 : fiction_percentage = 30 / 100)
  (h3 : transfer_fraction = 1 / 3)
  (h4 : fiction_transfer_fraction = 1 / 5) :
  let original_fiction := (fiction_percentage * total_volumes : ℚ)
  let transferred_volumes := (transfer_fraction * total_volumes : ℚ)
  let transferred_fiction := (fiction_transfer_fraction * transferred_volumes : ℚ)
  let remaining_fiction := original_fiction - transferred_fiction
  let remaining_volumes := total_volumes - transferred_volumes
  (remaining_fiction / remaining_volumes) * 100 = 35 := by
sorry


end NUMINAMATH_CALUDE_library_fiction_percentage_l3754_375486


namespace NUMINAMATH_CALUDE_tom_worked_eight_hours_l3754_375456

/-- Represents the number of hours Tom worked on Monday -/
def hours : ℝ := 8

/-- Represents the number of customers Tom served per hour -/
def customers_per_hour : ℝ := 10

/-- Represents the bonus point percentage (20% = 0.20) -/
def bonus_percentage : ℝ := 0.20

/-- Represents the total bonus points Tom earned on Monday -/
def total_bonus_points : ℝ := 16

/-- Proves that Tom worked 8 hours on Monday given the conditions -/
theorem tom_worked_eight_hours :
  hours * customers_per_hour * bonus_percentage = total_bonus_points :=
sorry

end NUMINAMATH_CALUDE_tom_worked_eight_hours_l3754_375456


namespace NUMINAMATH_CALUDE_tan_addition_subtraction_formulas_l3754_375415

noncomputable section

open Real

def tan_add (a b : ℝ) : ℝ := (tan a + tan b) / (1 - tan a * tan b)
def tan_sub (a b : ℝ) : ℝ := (tan a - tan b) / (1 + tan a * tan b)

theorem tan_addition_subtraction_formulas (a b : ℝ) :
  (tan (a + b) = tan_add a b) ∧ (tan (a - b) = tan_sub a b) :=
sorry

end

end NUMINAMATH_CALUDE_tan_addition_subtraction_formulas_l3754_375415


namespace NUMINAMATH_CALUDE_equation_solution_l3754_375425

theorem equation_solution : ∃ x : ℚ, (5 + 3.2 * x = 4.4 * x - 30) ∧ (x = 175 / 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3754_375425


namespace NUMINAMATH_CALUDE_complex_calculation_l3754_375407

theorem complex_calculation (z : ℂ) (h : z = 1 + I) : z - 2 / z^2 = 1 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_calculation_l3754_375407


namespace NUMINAMATH_CALUDE_no_quadratic_term_in_polynomial_difference_l3754_375454

theorem no_quadratic_term_in_polynomial_difference (x : ℝ) :
  let p₁ := 2 * x^3 - 8 * x^2 + x - 1
  let p₂ := 3 * x^3 + 2 * m * x^2 - 5 * x + 3
  (∃ m : ℝ, ∀ a b c d : ℝ, p₁ - p₂ = a * x^3 + c * x + d) → m = -4 :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_term_in_polynomial_difference_l3754_375454


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_l3754_375480

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two planes are parallel -/
def are_parallel (p1 p2 : Plane3D) : Prop :=
  ∃ k : ℝ, p1.normal = k • p2.normal

/-- A plane is perpendicular to a line -/
def is_perpendicular_to (p : Plane3D) (l : Line3D) : Prop :=
  ∃ k : ℝ, p.normal = k • l.direction

/-- Theorem: Two planes perpendicular to the same line are parallel -/
theorem planes_perpendicular_to_line_are_parallel (p1 p2 : Plane3D) (l : Line3D)
  (h1 : is_perpendicular_to p1 l) (h2 : is_perpendicular_to p2 l) :
  are_parallel p1 p2 :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_l3754_375480


namespace NUMINAMATH_CALUDE_glass_volume_proof_l3754_375416

theorem glass_volume_proof (V : ℝ) 
  (h1 : 0.4 * V = V - (0.6 * V))  -- pessimist's glass is 60% empty
  (h2 : 0.6 * V - 0.4 * V = 46)   -- difference in water volume
  : V = 230 := by
sorry

end NUMINAMATH_CALUDE_glass_volume_proof_l3754_375416


namespace NUMINAMATH_CALUDE_winnie_lollipops_l3754_375483

/-- The number of lollipops Winnie keeps for herself -/
def lollipops_kept (total_lollipops friends : ℕ) : ℕ :=
  total_lollipops % friends

theorem winnie_lollipops :
  let cherry := 32
  let wintergreen := 105
  let grape := 7
  let shrimp := 198
  let friends := 12
  let total := cherry + wintergreen + grape + shrimp
  lollipops_kept total friends = 6 := by sorry

end NUMINAMATH_CALUDE_winnie_lollipops_l3754_375483


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3754_375477

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 - 2*i) / (2 + i)) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3754_375477


namespace NUMINAMATH_CALUDE_polygon_side_containment_l3754_375495

/-- A polygon is a set of points in the plane. -/
def Polygon : Type := Set (ℝ × ℝ)

/-- A line in the plane. -/
def Line : Type := Set (ℝ × ℝ)

/-- The number of sides of a polygon. -/
def numSides (p : Polygon) : ℕ := sorry

/-- A line contains a side of a polygon. -/
def containsSide (l : Line) (p : Polygon) : Prop := sorry

/-- A line contains exactly one side of a polygon. -/
def containsExactlyOneSide (l : Line) (p : Polygon) : Prop := sorry

/-- Main theorem about 13-sided polygons and polygons with more than 13 sides. -/
theorem polygon_side_containment :
  (∀ p : Polygon, numSides p = 13 → ∃ l : Line, containsExactlyOneSide l p) ∧
  (∀ n : ℕ, n > 13 → ∃ p : Polygon, numSides p = n ∧ 
    ∀ l : Line, containsSide l p → ¬containsExactlyOneSide l p) :=
sorry

end NUMINAMATH_CALUDE_polygon_side_containment_l3754_375495


namespace NUMINAMATH_CALUDE_french_only_students_l3754_375462

/-- Given a group of students with the following properties:
  * There are 28 students in total
  * Some students take French
  * 10 students take Spanish
  * 4 students take both French and Spanish
  * 13 students take neither French nor Spanish
  * Students taking both languages are not counted with those taking only French or only Spanish
This theorem proves that exactly 1 student is taking only French. -/
theorem french_only_students (total : ℕ) (spanish : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 28)
  (h_spanish : spanish = 10)
  (h_both : both = 4)
  (h_neither : neither = 13) :
  total - spanish - both - neither = 1 := by
sorry

end NUMINAMATH_CALUDE_french_only_students_l3754_375462


namespace NUMINAMATH_CALUDE_intersection_when_a_is_3_subset_condition_l3754_375403

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def N (a : ℝ) : Set ℝ := {x | x ≤ 2*a - 5}

-- Theorem for part 1
theorem intersection_when_a_is_3 : 
  M ∩ N 3 = {x | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part 2
theorem subset_condition : 
  ∀ a : ℝ, M ⊆ N a ↔ a ≥ 9/2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_3_subset_condition_l3754_375403


namespace NUMINAMATH_CALUDE_work_completion_time_l3754_375452

theorem work_completion_time (a b c : ℝ) (h1 : a = 2 * b) (h2 : c = 3 * b) 
  (h3 : 1 / a + 1 / b + 1 / c = 1 / 18) : b = 33 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3754_375452


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l3754_375400

theorem girls_to_boys_ratio (total : ℕ) (girls boys : ℕ) 
  (h1 : total = 36)
  (h2 : girls + boys = total)
  (h3 : girls = boys + 6) : 
  girls * 5 = boys * 7 := by
sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l3754_375400


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_fourth_l3754_375432

theorem opposite_of_negative_one_fourth : 
  (-(-(1/4 : ℚ))) = (1/4 : ℚ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_fourth_l3754_375432


namespace NUMINAMATH_CALUDE_last_three_digits_of_8_105_l3754_375404

theorem last_three_digits_of_8_105 : 8^105 ≡ 992 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_8_105_l3754_375404


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3754_375442

theorem polynomial_evaluation (x : ℝ) (h : x = 3) : x^6 - 3*x = 720 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3754_375442


namespace NUMINAMATH_CALUDE_min_value_theorem_l3754_375443

theorem min_value_theorem (m : ℝ) (hm : m > 0)
  (h : ∀ x : ℝ, |x + 1| + |2*x - 1| ≥ m)
  (a b c : ℝ) (heq : a^2 + 2*b^2 + 3*c^2 = m) :
  ∀ a' b' c' : ℝ, a'^2 + 2*b'^2 + 3*c'^2 = m → a + 2*b + 3*c ≥ a' + 2*b' + 3*c' → a + 2*b + 3*c ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3754_375443


namespace NUMINAMATH_CALUDE_weighted_average_problem_l3754_375433

/-- Given numbers 4, 6, 8, p, q with a weighted average of 20,
    where p and q each have twice the weight of 4, 6, 8,
    prove that the average of p and q is 30.5 -/
theorem weighted_average_problem (p q : ℝ) : 
  (4 + 6 + 8 + 2*p + 2*q) / 7 = 20 →
  (p + q) / 2 = 30.5 := by
sorry

end NUMINAMATH_CALUDE_weighted_average_problem_l3754_375433


namespace NUMINAMATH_CALUDE_least_distance_is_one_thirtyfifth_l3754_375481

-- Define the unit segment [0, 1]
def unit_segment : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Define the division points for fifths
def fifth_points : Set ℝ := {x : ℝ | ∃ n : ℕ, 0 ≤ n ∧ n ≤ 5 ∧ x = n / 5}

-- Define the division points for sevenths
def seventh_points : Set ℝ := {x : ℝ | ∃ n : ℕ, 0 ≤ n ∧ n ≤ 7 ∧ x = n / 7}

-- Define all division points
def all_points : Set ℝ := fifth_points ∪ seventh_points

-- Define the distance between two points
def distance (x y : ℝ) : ℝ := |x - y|

-- Theorem statement
theorem least_distance_is_one_thirtyfifth :
  ∃ x y : ℝ, x ∈ all_points ∧ y ∈ all_points ∧ x ≠ y ∧
  distance x y = 1 / 35 ∧
  ∀ a b : ℝ, a ∈ all_points → b ∈ all_points → a ≠ b →
  distance a b ≥ 1 / 35 :=
sorry

end NUMINAMATH_CALUDE_least_distance_is_one_thirtyfifth_l3754_375481


namespace NUMINAMATH_CALUDE_blocks_per_group_l3754_375447

theorem blocks_per_group (total_blocks : ℕ) (num_groups : ℕ) (blocks_per_group : ℕ) :
  total_blocks = 820 →
  num_groups = 82 →
  total_blocks = num_groups * blocks_per_group →
  blocks_per_group = 10 := by
  sorry

end NUMINAMATH_CALUDE_blocks_per_group_l3754_375447


namespace NUMINAMATH_CALUDE_odd_function_a_indeterminate_l3754_375421

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_a_indeterminate (f : ℝ → ℝ) (h : OddFunction f) :
  ¬ ∃ a : ℝ, ∀ g : ℝ → ℝ, OddFunction g → g = f :=
sorry

end NUMINAMATH_CALUDE_odd_function_a_indeterminate_l3754_375421


namespace NUMINAMATH_CALUDE_merchant_bought_15_keyboards_l3754_375438

/-- The number of keyboards bought by a merchant -/
def num_keyboards : ℕ := 15

/-- The number of printers bought by the merchant -/
def num_printers : ℕ := 25

/-- The cost of one keyboard in dollars -/
def cost_keyboard : ℕ := 20

/-- The cost of one printer in dollars -/
def cost_printer : ℕ := 70

/-- The total cost of all items bought by the merchant in dollars -/
def total_cost : ℕ := 2050

/-- Theorem stating that the number of keyboards bought is 15 -/
theorem merchant_bought_15_keyboards :
  num_keyboards * cost_keyboard + num_printers * cost_printer = total_cost :=
sorry

end NUMINAMATH_CALUDE_merchant_bought_15_keyboards_l3754_375438


namespace NUMINAMATH_CALUDE_platform_length_calculation_l3754_375476

/-- Given a train of length 300 meters that crosses a platform in 39 seconds
    and a signal pole in 14 seconds, prove that the length of the platform
    is approximately 535.77 meters. -/
theorem platform_length_calculation (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_time = 39)
  (h3 : pole_time = 14) :
  ∃ platform_length : ℝ, abs (platform_length - 535.77) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_platform_length_calculation_l3754_375476
