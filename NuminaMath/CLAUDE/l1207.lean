import Mathlib

namespace NUMINAMATH_CALUDE_total_ribbons_used_l1207_120751

def dresses_per_day_first_week : ℕ := 2
def days_first_week : ℕ := 7
def dresses_per_day_second_week : ℕ := 3
def days_second_week : ℕ := 2
def ribbons_per_dress : ℕ := 2

theorem total_ribbons_used :
  (dresses_per_day_first_week * days_first_week + 
   dresses_per_day_second_week * days_second_week) * 
  ribbons_per_dress = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_ribbons_used_l1207_120751


namespace NUMINAMATH_CALUDE_marching_band_weight_theorem_l1207_120792

/-- Represents the weight carried by each instrument player in the marching band --/
structure BandWeights where
  trumpet_clarinet : ℕ
  trombone : ℕ
  tuba : ℕ
  drum : ℕ

/-- Represents the number of players for each instrument in the marching band --/
structure BandComposition where
  trumpets : ℕ
  clarinets : ℕ
  trombones : ℕ
  tubas : ℕ
  drummers : ℕ

/-- Calculates the total weight carried by the marching band --/
def total_weight (weights : BandWeights) (composition : BandComposition) : ℕ :=
  (weights.trumpet_clarinet * (composition.trumpets + composition.clarinets)) +
  (weights.trombone * composition.trombones) +
  (weights.tuba * composition.tubas) +
  (weights.drum * composition.drummers)

theorem marching_band_weight_theorem (weights : BandWeights) (composition : BandComposition) :
  weights.trombone = 10 →
  weights.tuba = 20 →
  weights.drum = 15 →
  composition.trumpets = 6 →
  composition.clarinets = 9 →
  composition.trombones = 8 →
  composition.tubas = 3 →
  composition.drummers = 2 →
  total_weight weights composition = 245 →
  weights.trumpet_clarinet = 5 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_weight_theorem_l1207_120792


namespace NUMINAMATH_CALUDE_piggy_bank_pennies_l1207_120704

theorem piggy_bank_pennies (compartments initial_per_compartment final_total : ℕ) 
  (h1 : compartments = 12)
  (h2 : initial_per_compartment = 2)
  (h3 : final_total = 96)
  : (final_total - compartments * initial_per_compartment) / compartments = 6 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_pennies_l1207_120704


namespace NUMINAMATH_CALUDE_perfect_linear_correlation_l1207_120755

/-- A scatter plot where all points fall on a straight line -/
structure PerfectLinearScatterPlot where
  /-- The slope of the line (non-zero real number) -/
  slope : ℝ
  /-- Assumption that the slope is non-zero -/
  slope_nonzero : slope ≠ 0

/-- The correlation coefficient R^2 for a scatter plot -/
def correlation_coefficient (plot : PerfectLinearScatterPlot) : ℝ := sorry

/-- Theorem: The correlation coefficient R^2 is 1 for a perfect linear scatter plot -/
theorem perfect_linear_correlation 
  (plot : PerfectLinearScatterPlot) : 
  correlation_coefficient plot = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_linear_correlation_l1207_120755


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1207_120714

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = 0 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 16 → 
  a + b = -16 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l1207_120714


namespace NUMINAMATH_CALUDE_eq1_roots_eq2_roots_l1207_120772

-- Define the quadratic equations
def eq1 (x : ℝ) : Prop := x^2 + 10*x + 16 = 0
def eq2 (x : ℝ) : Prop := x*(x+4) = 8*x + 12

-- Theorem for the first equation
theorem eq1_roots : 
  (∃ x : ℝ, eq1 x) ↔ (eq1 (-2) ∧ eq1 (-8)) :=
sorry

-- Theorem for the second equation
theorem eq2_roots : 
  (∃ x : ℝ, eq2 x) ↔ (eq2 (-2) ∧ eq2 6) :=
sorry

end NUMINAMATH_CALUDE_eq1_roots_eq2_roots_l1207_120772


namespace NUMINAMATH_CALUDE_davids_math_marks_l1207_120762

def englishMarks : ℝ := 70
def physicsMarks : ℝ := 78
def chemistryMarks : ℝ := 60
def biologyMarks : ℝ := 65
def averageMarks : ℝ := 66.6
def totalSubjects : ℕ := 5

theorem davids_math_marks :
  let totalMarks := averageMarks * totalSubjects
  let knownSubjectsMarks := englishMarks + physicsMarks + chemistryMarks + biologyMarks
  let mathMarks := totalMarks - knownSubjectsMarks
  mathMarks = 60 := by sorry

end NUMINAMATH_CALUDE_davids_math_marks_l1207_120762


namespace NUMINAMATH_CALUDE_fraction_pattern_l1207_120764

theorem fraction_pattern (n m k : ℕ) (h1 : m ≠ 0) (h2 : k ≠ 0) 
  (h3 : n / m = k * n / (k * m)) : 
  (n + m) / m = (k * n + k * m) / (k * m) := by
  sorry

end NUMINAMATH_CALUDE_fraction_pattern_l1207_120764


namespace NUMINAMATH_CALUDE_angle_measure_problem_l1207_120703

theorem angle_measure_problem (C D E F G : Real) : 
  C = 120 →
  C + D = 180 →
  E = 50 →
  F = D →
  E + F + G = 180 →
  G = 70 := by sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l1207_120703


namespace NUMINAMATH_CALUDE_typing_service_problem_l1207_120782

/-- Typing service problem -/
theorem typing_service_problem
  (total_pages : ℕ)
  (first_time_cost : ℕ)
  (revision_cost : ℕ)
  (pages_revised_once : ℕ)
  (total_cost : ℕ)
  (h1 : total_pages = 100)
  (h2 : first_time_cost = 5)
  (h3 : revision_cost = 3)
  (h4 : pages_revised_once = 30)
  (h5 : total_cost = 710)
  : ∃ (pages_revised_twice : ℕ),
    pages_revised_twice = 20 ∧
    total_cost = total_pages * first_time_cost +
                 pages_revised_once * revision_cost +
                 pages_revised_twice * revision_cost * 2 :=
by sorry

end NUMINAMATH_CALUDE_typing_service_problem_l1207_120782


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1207_120750

/-- An arithmetic sequence of 5 terms starting with 3 and ending with 15 -/
def ArithmeticSequence (a b c : ℝ) : Prop :=
  ∃ d : ℝ, (a = 3 + d) ∧ (b = 3 + 2*d) ∧ (c = 3 + 3*d) ∧ (15 = 3 + 4*d)

/-- The sum of the middle three terms of the arithmetic sequence is 27 -/
theorem arithmetic_sequence_sum (a b c : ℝ) 
  (h : ArithmeticSequence a b c) : a + b + c = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1207_120750


namespace NUMINAMATH_CALUDE_floor_problem_solution_exists_and_unique_floor_length_satisfies_conditions_l1207_120748

/-- Represents the dimensions and costs of a rectangular floor with a painted border. -/
structure FloorProblem where
  breadth : ℝ
  length_ratio : ℝ
  floor_paint_rate : ℝ
  floor_paint_cost : ℝ
  border_paint_rate : ℝ
  total_paint_cost : ℝ

/-- The main theorem stating the existence and uniqueness of a solution to the floor problem. -/
theorem floor_problem_solution_exists_and_unique :
  ∃! (fp : FloorProblem),
    fp.length_ratio = 3 ∧
    fp.floor_paint_rate = 3.00001 ∧
    fp.floor_paint_cost = 361 ∧
    fp.border_paint_rate = 15 ∧
    fp.total_paint_cost = 500 ∧
    fp.floor_paint_rate * (fp.length_ratio * fp.breadth * fp.breadth) = fp.floor_paint_cost ∧
    fp.border_paint_rate * (2 * (fp.length_ratio * fp.breadth + fp.breadth)) + fp.floor_paint_cost = fp.total_paint_cost :=
  sorry

/-- Function to calculate the length of the floor given a FloorProblem instance. -/
def calculate_floor_length (fp : FloorProblem) : ℝ :=
  fp.length_ratio * fp.breadth

/-- Theorem stating that the calculated floor length satisfies the problem conditions. -/
theorem floor_length_satisfies_conditions (fp : FloorProblem) :
  fp.length_ratio = 3 →
  fp.floor_paint_rate = 3.00001 →
  fp.floor_paint_cost = 361 →
  fp.border_paint_rate = 15 →
  fp.total_paint_cost = 500 →
  fp.floor_paint_rate * (fp.length_ratio * fp.breadth * fp.breadth) = fp.floor_paint_cost →
  fp.border_paint_rate * (2 * (fp.length_ratio * fp.breadth + fp.breadth)) + fp.floor_paint_cost = fp.total_paint_cost →
  ∃ (length : ℝ), length = calculate_floor_length fp :=
  sorry

end NUMINAMATH_CALUDE_floor_problem_solution_exists_and_unique_floor_length_satisfies_conditions_l1207_120748


namespace NUMINAMATH_CALUDE_compound_composition_l1207_120793

/-- Prove that a compound with 2 I atoms and a molecular weight of 294 g/mol contains 1 Ca atom -/
theorem compound_composition (atomic_weight_Ca atomic_weight_I : ℝ) 
  (h1 : atomic_weight_Ca = 40.08)
  (h2 : atomic_weight_I = 126.90)
  (h3 : 2 * atomic_weight_I + atomic_weight_Ca = 294) : 
  ∃ (n : ℕ), n = 1 ∧ n * atomic_weight_Ca = 294 - 2 * atomic_weight_I :=
by sorry

end NUMINAMATH_CALUDE_compound_composition_l1207_120793


namespace NUMINAMATH_CALUDE_equivalent_operation_l1207_120773

theorem equivalent_operation (x : ℝ) : (x / (5/4)) * (4/3) = x * (16/15) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operation_l1207_120773


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1207_120758

theorem batsman_average_increase (total_runs_before : ℕ) : 
  let innings_before : ℕ := 10
  let new_score : ℕ := 80
  let new_average : ℝ := 30
  let old_average : ℝ := total_runs_before / innings_before
  let increase : ℝ := new_average - old_average
  (total_runs_before + new_score) / (innings_before + 1) = new_average →
  increase = 5 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l1207_120758


namespace NUMINAMATH_CALUDE_otimes_neg_two_four_otimes_equation_l1207_120725

/-- Define the ⊗ operation for rational numbers -/
def otimes (a b : ℚ) : ℚ := a * b^2 + 2 * a * b + a

/-- Theorem 1: (-2) ⊗ 4 = -50 -/
theorem otimes_neg_two_four : otimes (-2) 4 = -50 := by sorry

/-- Theorem 2: If x ⊗ 3 = y ⊗ (-3), then 8x - 2y + 5 = 5 -/
theorem otimes_equation (x y : ℚ) (h : otimes x 3 = otimes y (-3)) : 
  8 * x - 2 * y + 5 = 5 := by sorry

end NUMINAMATH_CALUDE_otimes_neg_two_four_otimes_equation_l1207_120725


namespace NUMINAMATH_CALUDE_goose_survival_l1207_120788

/-- The fraction of goose eggs that hatch -/
def hatch_rate : ℚ := 1/4

/-- The fraction of hatched geese that survive the first month -/
def first_month_survival_rate : ℚ := 4/5

/-- The fraction of geese that survived the first month but did not survive the first year -/
def first_year_mortality_rate : ℚ := 3/5

/-- The fraction of original eggs that result in geese surviving the first year -/
def first_year_survival_fraction : ℚ := 
  hatch_rate * first_month_survival_rate * (1 - first_year_mortality_rate)

theorem goose_survival (n : ℕ) : 
  ⌊(n : ℚ) * first_year_survival_fraction⌋ = ⌊(n : ℚ) * (8/100)⌋ :=
sorry

end NUMINAMATH_CALUDE_goose_survival_l1207_120788


namespace NUMINAMATH_CALUDE_optimal_circle_radii_equilateral_triangle_l1207_120726

/-- Given an equilateral triangle with side length 1, this theorem states that
    the maximum area covered by three circles centered at the vertices,
    not intersecting each other or the opposite sides, is achieved when
    the radii are R_a = √3/2 and R_b = R_c = 1 - √3/2. -/
theorem optimal_circle_radii_equilateral_triangle :
  let side_length : ℝ := 1
  let height : ℝ := Real.sqrt 3 / 2
  let R_a : ℝ := height
  let R_b : ℝ := 1 - height
  let R_c : ℝ := 1 - height
  let area_covered (r_a r_b r_c : ℝ) : ℝ := π / 6 * (r_a^2 + r_b^2 + r_c^2)
  let is_valid_radii (r_a r_b r_c : ℝ) : Prop :=
    r_a ≤ height ∧ r_a ≥ 1/2 ∧
    r_b ≤ 1 - r_a ∧ r_c ≤ 1 - r_a
  ∀ r_a r_b r_c : ℝ,
    is_valid_radii r_a r_b r_c →
    area_covered r_a r_b r_c ≤ area_covered R_a R_b R_c :=
by sorry

end NUMINAMATH_CALUDE_optimal_circle_radii_equilateral_triangle_l1207_120726


namespace NUMINAMATH_CALUDE_brittany_age_theorem_l1207_120723

/-- Brittany's age when she returns from vacation -/
def brittany_age_after_vacation (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ) : ℕ :=
  rebecca_age + age_difference + vacation_duration

/-- Theorem stating Brittany's age after vacation -/
theorem brittany_age_theorem (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ)
  (h1 : rebecca_age = 25)
  (h2 : age_difference = 3)
  (h3 : vacation_duration = 4) :
  brittany_age_after_vacation rebecca_age age_difference vacation_duration = 32 := by
  sorry

#check brittany_age_theorem

end NUMINAMATH_CALUDE_brittany_age_theorem_l1207_120723


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1207_120774

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | 2 * x - 4 ≤ 0 ∧ -x + 1 < 0}
  S = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1207_120774


namespace NUMINAMATH_CALUDE_num_algebraic_expressions_is_five_l1207_120716

/-- An expression is algebraic if it consists of numbers, variables, and arithmetic operations, without equality or inequality symbols. -/
def is_algebraic_expression (e : String) : Bool :=
  match e with
  | "2x^2" => true
  | "1-2x=0" => false
  | "ab" => true
  | "a>0" => false
  | "0" => true
  | "1/a" => true
  | "π" => true
  | _ => false

/-- The list of expressions to be checked -/
def expressions : List String :=
  ["2x^2", "1-2x=0", "ab", "a>0", "0", "1/a", "π"]

/-- The number of algebraic expressions in the list -/
def num_algebraic_expressions : Nat :=
  (expressions.filter is_algebraic_expression).length

theorem num_algebraic_expressions_is_five :
  num_algebraic_expressions = 5 := by
  sorry

end NUMINAMATH_CALUDE_num_algebraic_expressions_is_five_l1207_120716


namespace NUMINAMATH_CALUDE_vector_operation_l1207_120797

def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-3, -4)

theorem vector_operation : 
  (2 : ℝ) • a - b = (-1, 6) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l1207_120797


namespace NUMINAMATH_CALUDE_voter_percentage_for_candidate_A_l1207_120769

theorem voter_percentage_for_candidate_A
  (total_voters : ℝ)
  (democrat_percentage : ℝ)
  (democrat_support_A : ℝ)
  (republican_support_A : ℝ)
  (h1 : democrat_percentage = 0.6)
  (h2 : democrat_support_A = 0.7)
  (h3 : republican_support_A = 0.2)
  (h4 : total_voters > 0) :
  let republican_percentage := 1 - democrat_percentage
  let voters_for_A := total_voters * (democrat_percentage * democrat_support_A + republican_percentage * republican_support_A)
  voters_for_A / total_voters = 0.5 := by
sorry

end NUMINAMATH_CALUDE_voter_percentage_for_candidate_A_l1207_120769


namespace NUMINAMATH_CALUDE_zoo_count_difference_l1207_120708

/-- Proves that the difference between the number of monkeys and giraffes is 22 -/
theorem zoo_count_difference : 
  let zebras : ℕ := 12
  let camels : ℕ := zebras / 2
  let monkeys : ℕ := 4 * camels
  let giraffes : ℕ := 2
  monkeys - giraffes = 22 := by sorry

end NUMINAMATH_CALUDE_zoo_count_difference_l1207_120708


namespace NUMINAMATH_CALUDE_inscribed_circle_area_l1207_120766

theorem inscribed_circle_area (large_square_area : ℝ) (h : large_square_area = 80) :
  let large_side := Real.sqrt large_square_area
  let small_side := large_side / Real.sqrt 2
  let circle_radius := small_side / 2
  circle_radius ^ 2 * Real.pi = 10 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_l1207_120766


namespace NUMINAMATH_CALUDE_total_beneficial_insects_l1207_120736

theorem total_beneficial_insects (ladybugs_with_spots : Nat) (ladybugs_without_spots : Nat) (green_lacewings : Nat) (trichogramma_wasps : Nat)
  (h1 : ladybugs_with_spots = 12170)
  (h2 : ladybugs_without_spots = 54912)
  (h3 : green_lacewings = 67923)
  (h4 : trichogramma_wasps = 45872) :
  ladybugs_with_spots + ladybugs_without_spots + green_lacewings + trichogramma_wasps = 180877 := by
  sorry

end NUMINAMATH_CALUDE_total_beneficial_insects_l1207_120736


namespace NUMINAMATH_CALUDE_negation_equivalence_l1207_120718

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (planet : U → Prop)
variable (orbits_sun : U → Prop)

-- Define the original statement
def every_planet_orbits_sun : Prop := ∀ x, planet x → orbits_sun x

-- Define the negation we want to prove
def some_planets_dont_orbit_sun : Prop := ∃ x, planet x ∧ ¬(orbits_sun x)

-- Theorem statement
theorem negation_equivalence : 
  ¬(every_planet_orbits_sun U planet orbits_sun) ↔ some_planets_dont_orbit_sun U planet orbits_sun :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1207_120718


namespace NUMINAMATH_CALUDE_xyz_value_l1207_120747

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b - c) * (x + 2))
  (eq2 : b = (a - c) * (y + 2))
  (eq3 : c = (a - b) * (z + 2))
  (eq4 : x * y + x * z + y * z = 12)
  (eq5 : x + y + z = 6) :
  x * y * z = 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1207_120747


namespace NUMINAMATH_CALUDE_value_of_expression_l1207_120729

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem value_of_expression (a b c d : ℝ) :
  f a b c d (-2) = -3 →
  8*a - 4*b + 2*c - d = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1207_120729


namespace NUMINAMATH_CALUDE_arithmetic_equality_l1207_120761

theorem arithmetic_equality : 12.05 * 5.4 + 0.6 = 65.67 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l1207_120761


namespace NUMINAMATH_CALUDE_integral_f_cos_nonnegative_l1207_120759

open MeasureTheory Interval RealInnerProductSpace Set

theorem integral_f_cos_nonnegative 
  (f : ℝ → ℝ) 
  (hf_continuous : ContinuousOn f (Icc 0 (2 * Real.pi)))
  (hf'_continuous : ContinuousOn (deriv f) (Icc 0 (2 * Real.pi)))
  (hf''_continuous : ContinuousOn (deriv^[2] f) (Icc 0 (2 * Real.pi)))
  (hf''_nonneg : ∀ x ∈ Icc 0 (2 * Real.pi), deriv^[2] f x ≥ 0) :
  ∫ x in Icc 0 (2 * Real.pi), f x * Real.cos x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_integral_f_cos_nonnegative_l1207_120759


namespace NUMINAMATH_CALUDE_not_in_range_of_g_l1207_120786

/-- The function g(x) defined as x^3 + x^2 + bx + 2 -/
def g (b : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + b*x + 2

/-- Theorem stating that for all real b ≠ 6, -2 is not in the range of g(x) -/
theorem not_in_range_of_g (b : ℝ) (h : b ≠ 6) :
  ¬∃ x, g b x = -2 := by sorry

end NUMINAMATH_CALUDE_not_in_range_of_g_l1207_120786


namespace NUMINAMATH_CALUDE_circle_line_distance_range_l1207_120739

/-- Given a circle x^2 + y^2 = 9 and a line y = x + b, if there are exactly two points
    on the circle that have a distance of 1 to the line, then b is in the range
    (-4√2, -2√2) ∪ (2√2, 4√2) -/
theorem circle_line_distance_range (b : ℝ) : 
  (∃! (p q : ℝ × ℝ), 
    p.1^2 + p.2^2 = 9 ∧ 
    q.1^2 + q.2^2 = 9 ∧ 
    p ≠ q ∧
    (abs (p.2 - p.1 - b) / Real.sqrt 2 = 1) ∧
    (abs (q.2 - q.1 - b) / Real.sqrt 2 = 1)) →
  (b > 2 * Real.sqrt 2 ∧ b < 4 * Real.sqrt 2) ∨ 
  (b < -2 * Real.sqrt 2 ∧ b > -4 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_range_l1207_120739


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1023rd_term_l1207_120740

def arithmetic_sequence (a₁ a₂ a₃ a₄ : ℚ) : Prop :=
  ∃ (d : ℚ), a₂ - a₁ = d ∧ a₃ - a₂ = d ∧ a₄ - a₃ = d

def nth_term (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_1023rd_term (p r : ℚ) :
  arithmetic_sequence (2*p) 15 (4*p+r) (4*p-r) →
  nth_term (2*p) ((4*p-r) - (4*p+r)) 1023 = 61215 / 14 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_1023rd_term_l1207_120740


namespace NUMINAMATH_CALUDE_least_trees_for_rows_trees_168_divisible_least_trees_is_168_l1207_120775

theorem least_trees_for_rows (n : ℕ) : n > 0 ∧ 6 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n → n ≥ 168 := by
  sorry

theorem trees_168_divisible : 6 ∣ 168 ∧ 7 ∣ 168 ∧ 8 ∣ 168 := by
  sorry

theorem least_trees_is_168 : ∃ (n : ℕ), n > 0 ∧ 6 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n ∧ ∀ (m : ℕ), (m > 0 ∧ 6 ∣ m ∧ 7 ∣ m ∧ 8 ∣ m) → m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_least_trees_for_rows_trees_168_divisible_least_trees_is_168_l1207_120775


namespace NUMINAMATH_CALUDE_right_triangle_existence_and_uniqueness_l1207_120722

theorem right_triangle_existence_and_uniqueness 
  (c d : ℝ) (hc : c > 0) (hd : d > 0) (hcd : c > d) :
  ∃! (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    a^2 + b^2 = c^2 ∧ 
    a - b = d := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_existence_and_uniqueness_l1207_120722


namespace NUMINAMATH_CALUDE_shortest_distance_to_circle_l1207_120712

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 - 8*y + 9 = 0

/-- The shortest distance from the origin to the circle -/
def shortest_distance : ℝ := 1

/-- Theorem stating that the shortest distance from the origin to the circle is 1 -/
theorem shortest_distance_to_circle :
  ∃ (p : ℝ × ℝ), circle_equation p.1 p.2 ∧
  ∀ (q : ℝ × ℝ), circle_equation q.1 q.2 →
  Real.sqrt (p.1^2 + p.2^2) ≤ Real.sqrt (q.1^2 + q.2^2) ∧
  Real.sqrt (p.1^2 + p.2^2) = shortest_distance :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_to_circle_l1207_120712


namespace NUMINAMATH_CALUDE_trig_identity_l1207_120732

theorem trig_identity (θ : ℝ) (h : 2 * Real.sin θ + Real.cos θ = 0) :
  Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = - 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1207_120732


namespace NUMINAMATH_CALUDE_complex_equation_first_quadrant_l1207_120794

theorem complex_equation_first_quadrant (z : ℂ) (a : ℝ) : 
  (1 - I) * z = a * I + 1 → 
  (z.re > 0 ∧ z.im > 0) → 
  a = 0 := by sorry

end NUMINAMATH_CALUDE_complex_equation_first_quadrant_l1207_120794


namespace NUMINAMATH_CALUDE_quadratic_minimum_minimum_at_three_l1207_120778

theorem quadratic_minimum (x : ℝ) : x^2 - 6*x + 1 ≥ -8 ∧ ∃ x₀ : ℝ, x₀^2 - 6*x₀ + 1 = -8 := by
  sorry

theorem minimum_at_three : (3 : ℝ)^2 - 6*3 + 1 = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_minimum_at_three_l1207_120778


namespace NUMINAMATH_CALUDE_four_boxes_volume_l1207_120724

/-- The volume of a cube with edge length s -/
def cube_volume (s : ℝ) : ℝ := s ^ 3

/-- The total volume of n identical cubes with edge length s -/
def total_volume (n : ℕ) (s : ℝ) : ℝ := n * cube_volume s

/-- Theorem: The total volume of four cubic boxes, each with an edge length of 5 meters, is 500 cubic meters -/
theorem four_boxes_volume : total_volume 4 5 = 500 := by
  sorry

end NUMINAMATH_CALUDE_four_boxes_volume_l1207_120724


namespace NUMINAMATH_CALUDE_partner_investment_period_l1207_120799

/-- Given two partners p and q with investment ratio 7:5 and profit ratio 7:10,
    where q invests for 16 months, this theorem proves that p invests for 8 months. -/
theorem partner_investment_period (x : ℝ) (t : ℝ) : 
  (7 * x * t) / (5 * x * 16) = 7 / 10 → t = 8 := by
  sorry

end NUMINAMATH_CALUDE_partner_investment_period_l1207_120799


namespace NUMINAMATH_CALUDE_perpendicular_tangents_exist_and_unique_l1207_120727

/-- The line on which we search for the point. -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 2

/-- The parabola to which we find tangents. -/
def parabola (x y : ℝ) : Prop := y = x^2

/-- A point is on a tangent line to the parabola. -/
def is_on_tangent (x y x₀ y₀ : ℝ) : Prop :=
  y = y₀ + 2 * x₀ * (x - x₀)

/-- Two lines are perpendicular. -/
def are_perpendicular (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -1

/-- The theorem stating the existence and uniqueness of the point and its tangents. -/
theorem perpendicular_tangents_exist_and_unique :
  ∃! x₀ y₀ k₁ k₂,
    line x₀ y₀ ∧
    parabola x₀ y₀ ∧
    are_perpendicular (2 * x₀) (2 * x₀) ∧
    (∀ x y, is_on_tangent x y x₀ y₀ → (y = -1/4 + k₁ * (x - 1) ∨ y = -1/4 + k₂ * (x - 1))) ∧
    k₁ = 2 + Real.sqrt 5 ∧
    k₂ = 2 - Real.sqrt 5 :=
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_exist_and_unique_l1207_120727


namespace NUMINAMATH_CALUDE_snakes_count_l1207_120733

theorem snakes_count (breeding_balls : ℕ) (snakes_per_ball : ℕ) (snake_pairs : ℕ) : 
  breeding_balls * snakes_per_ball + 2 * snake_pairs = 36 :=
by
  sorry

#check snakes_count 3 8 6

end NUMINAMATH_CALUDE_snakes_count_l1207_120733


namespace NUMINAMATH_CALUDE_bird_percentage_l1207_120744

/-- The percentage of birds that are not hawks, paddyfield-warblers, kingfishers, or blackbirds in Goshawk-Eurasian Nature Reserve -/
theorem bird_percentage (total : ℝ) (hawks paddyfield_warblers kingfishers blackbirds : ℝ)
  (h1 : hawks = 0.3 * total)
  (h2 : paddyfield_warblers = 0.4 * (total - hawks))
  (h3 : kingfishers = 0.25 * paddyfield_warblers)
  (h4 : blackbirds = 0.15 * (hawks + paddyfield_warblers))
  (h5 : total > 0) :
  (total - (hawks + paddyfield_warblers + kingfishers + blackbirds)) / total = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_bird_percentage_l1207_120744


namespace NUMINAMATH_CALUDE_parallelogram_area_l1207_120780

def v : ℝ × ℝ := (7, 4)
def w : ℝ × ℝ := (2, -9)

theorem parallelogram_area : 
  let v2w := (2 * w.1, 2 * w.2)
  abs (v.1 * v2w.2 - v.2 * v2w.1) = 142 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1207_120780


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1207_120745

-- Define the function f(x) = -x^2 + 2x + 1
def f (x : ℝ) : ℝ := -x^2 + 2*x + 1

-- Define the inequality
def has_solution (m : ℝ) : Prop :=
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ x^2 - 2*x - 1 + m ≤ 0

-- Theorem statement
theorem inequality_solution_range :
  ∀ m : ℝ, has_solution m ↔ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1207_120745


namespace NUMINAMATH_CALUDE_no_solution_exists_l1207_120771

theorem no_solution_exists : ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ y - x = 5 ∧ x * y = 132 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1207_120771


namespace NUMINAMATH_CALUDE_unique_solution_l1207_120760

theorem unique_solution :
  ∃! (A B C D : ℕ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧
    1000 * A + 100 * A + 10 * B + C - (1000 * B + 100 * A + 10 * C + B) = 1000 * A + 100 * B + 10 * C + D ∧
    A = 9 ∧ B = 6 ∧ C = 8 ∧ D = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1207_120760


namespace NUMINAMATH_CALUDE_power_of_negative_product_l1207_120789

theorem power_of_negative_product (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l1207_120789


namespace NUMINAMATH_CALUDE_toy_sales_profit_maximization_l1207_120711

def weekly_sales (x : ℤ) (k : ℚ) (b : ℚ) : ℚ := k * x + b

theorem toy_sales_profit_maximization 
  (k : ℚ) (b : ℚ) 
  (h1 : weekly_sales 120 k b = 80) 
  (h2 : weekly_sales 140 k b = 40) 
  (h3 : ∀ x : ℤ, 100 ≤ x ∧ x ≤ 160) :
  (k = -2 ∧ b = 320) ∧
  (∀ x : ℤ, (x - 100) * (weekly_sales x k b) ≤ 1800) ∧
  ((130 - 100) * (weekly_sales 130 k b) = 1800) :=
sorry

end NUMINAMATH_CALUDE_toy_sales_profit_maximization_l1207_120711


namespace NUMINAMATH_CALUDE_girls_examined_l1207_120777

theorem girls_examined (boys : ℕ) (girls : ℕ) 
  (h1 : boys = 50)
  (h2 : (25 : ℝ) + 0.6 * girls = 0.5667 * (boys + girls)) :
  girls = 100 := by
  sorry

end NUMINAMATH_CALUDE_girls_examined_l1207_120777


namespace NUMINAMATH_CALUDE_orchard_solution_l1207_120715

/-- Represents the number of trees in an orchard -/
structure Orchard where
  peach : ℕ
  apple : ℕ

/-- Conditions for the orchard problem -/
def OrchardConditions (o : Orchard) : Prop :=
  (o.apple = o.peach + 1700) ∧ (o.apple = 3 * o.peach + 200)

/-- Theorem stating the solution to the orchard problem -/
theorem orchard_solution : 
  ∃ o : Orchard, OrchardConditions o ∧ o.peach = 750 ∧ o.apple = 2450 := by
  sorry

end NUMINAMATH_CALUDE_orchard_solution_l1207_120715


namespace NUMINAMATH_CALUDE_socks_difference_l1207_120787

/-- Proves that after losing half of the white socks, the person still has 6 more white socks than black socks -/
theorem socks_difference (black_socks : ℕ) (white_socks : ℕ) : 
  black_socks = 6 →
  white_socks = 4 * black_socks →
  (white_socks / 2) - black_socks = 6 := by
sorry

end NUMINAMATH_CALUDE_socks_difference_l1207_120787


namespace NUMINAMATH_CALUDE_total_songs_bought_l1207_120721

theorem total_songs_bought (country_albums pop_albums rock_albums : ℕ)
  (country_songs_per_album pop_songs_per_album rock_songs_per_album : ℕ) :
  country_albums = 2 ∧
  pop_albums = 8 ∧
  rock_albums = 5 ∧
  country_songs_per_album = 7 ∧
  pop_songs_per_album = 10 ∧
  rock_songs_per_album = 12 →
  country_albums * country_songs_per_album +
  pop_albums * pop_songs_per_album +
  rock_albums * rock_songs_per_album = 154 :=
by sorry

end NUMINAMATH_CALUDE_total_songs_bought_l1207_120721


namespace NUMINAMATH_CALUDE_no_valid_numbers_l1207_120784

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧  -- 3-digit number
  (n / 100 + (n / 10) % 10 + n % 10 = 27) ∧  -- digit-sum is 27
  n % 2 = 0 ∧  -- even number
  n % 10 = 4  -- ends in 4

theorem no_valid_numbers : ¬∃ (n : ℕ), is_valid_number n := by
  sorry

end NUMINAMATH_CALUDE_no_valid_numbers_l1207_120784


namespace NUMINAMATH_CALUDE_sum_of_specific_polynomials_l1207_120765

/-- A linear polynomial -/
def LinearPolynomial (α : Type*) [Field α] := α → α

/-- A cubic polynomial -/
def CubicPolynomial (α : Type*) [Field α] := α → α

/-- The theorem statement -/
theorem sum_of_specific_polynomials 
  (p : LinearPolynomial ℝ) (q : CubicPolynomial ℝ)
  (h1 : p 1 = 1)
  (h2 : q (-1) = -3)
  (h3 : ∃ r : ℝ → ℝ, ∀ x, q x = r x * (x - 2)^2)
  (h4 : ∃ s t : ℝ → ℝ, (∀ x, p x = s x * (x + 1)) ∧ (∀ x, q x = t x * (x + 1)))
  : ∀ x, p x + q x = -1/3 * x^3 + 4/3 * x^2 + 1/3 * x + 13/6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_polynomials_l1207_120765


namespace NUMINAMATH_CALUDE_igor_lied_l1207_120730

-- Define the set of boys
inductive Boy : Type
| andrey : Boy
| maxim : Boy
| igor : Boy
| kolya : Boy

-- Define the possible positions in the race
inductive Position : Type
| first : Position
| second : Position
| third : Position
| fourth : Position

-- Define a function to represent the actual position of each boy
def actual_position : Boy → Position := sorry

-- Define a function to represent whether a boy is telling the truth
def is_truthful : Boy → Prop := sorry

-- State the conditions of the problem
axiom three_truthful : ∃ (a b c : Boy), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  is_truthful a ∧ is_truthful b ∧ is_truthful c ∧ 
  ∀ (d : Boy), d ≠ a ∧ d ≠ b ∧ d ≠ c → ¬is_truthful d

axiom andrey_claim : is_truthful Boy.andrey ↔ 
  actual_position Boy.andrey ≠ Position.first ∧ 
  actual_position Boy.andrey ≠ Position.fourth

axiom maxim_claim : is_truthful Boy.maxim ↔ 
  actual_position Boy.maxim ≠ Position.fourth

axiom igor_claim : is_truthful Boy.igor ↔ 
  actual_position Boy.igor = Position.first

axiom kolya_claim : is_truthful Boy.kolya ↔ 
  actual_position Boy.kolya = Position.fourth

-- Theorem to prove
theorem igor_lied : ¬is_truthful Boy.igor := by sorry

end NUMINAMATH_CALUDE_igor_lied_l1207_120730


namespace NUMINAMATH_CALUDE_solution_interval_l1207_120785

theorem solution_interval (x₀ : ℝ) (k : ℤ) : 
  (Real.log x₀ + x₀ = 4) → 
  (x₀ > k ∧ x₀ < k + 1) → 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_interval_l1207_120785


namespace NUMINAMATH_CALUDE_tamara_cracker_count_l1207_120707

/-- The number of crackers each person has -/
structure CrackerCount where
  tamara : ℕ
  nicholas : ℕ
  marcus : ℕ
  mona : ℕ

/-- The conditions of the cracker problem -/
def CrackerProblem (c : CrackerCount) : Prop :=
  c.tamara = 2 * c.nicholas ∧
  c.marcus = 3 * c.mona ∧
  c.nicholas = c.mona + 6 ∧
  c.marcus = 27

theorem tamara_cracker_count (c : CrackerCount) (h : CrackerProblem c) : c.tamara = 30 := by
  sorry

end NUMINAMATH_CALUDE_tamara_cracker_count_l1207_120707


namespace NUMINAMATH_CALUDE_product_max_min_two_digit_l1207_120783

def max_two_digit : ℕ := 99
def min_two_digit : ℕ := 10

theorem product_max_min_two_digit : max_two_digit * min_two_digit = 990 := by
  sorry

end NUMINAMATH_CALUDE_product_max_min_two_digit_l1207_120783


namespace NUMINAMATH_CALUDE_hexagon_sequence_theorem_l1207_120738

/-- Represents the number of dots in the nth hexagon of the sequence -/
def hexagon_dots (n : ℕ) : ℕ :=
  if n = 0 then 0
  else 1 + 3 * n * (n - 1)

/-- The theorem stating the number of dots in the first four hexagons -/
theorem hexagon_sequence_theorem :
  hexagon_dots 1 = 1 ∧
  hexagon_dots 2 = 7 ∧
  hexagon_dots 3 = 19 ∧
  hexagon_dots 4 = 37 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_sequence_theorem_l1207_120738


namespace NUMINAMATH_CALUDE_K_factorization_l1207_120746

theorem K_factorization (x y z : ℝ) : 
  (x + 2*y + 3*z) * (2*x - y - z) * (y + 2*z + 3*x) +
  (y + 2*z + 3*x) * (2*y - z - x) * (z + 2*x + 3*y) +
  (z + 2*x + 3*y) * (2*z - x - y) * (x + 2*y + 3*z) =
  (y + z - 2*x) * (z + x - 2*y) * (x + y - 2*z) := by
sorry

end NUMINAMATH_CALUDE_K_factorization_l1207_120746


namespace NUMINAMATH_CALUDE_candy_fundraiser_profit_l1207_120763

def candy_fundraiser (boxes_total : ℕ) (boxes_discounted : ℕ) (bars_per_box : ℕ) 
  (selling_price : ℚ) (regular_price : ℚ) (discounted_price : ℚ) : ℚ :=
  let boxes_regular := boxes_total - boxes_discounted
  let total_revenue := boxes_total * bars_per_box * selling_price
  let cost_regular := boxes_regular * bars_per_box * regular_price
  let cost_discounted := boxes_discounted * bars_per_box * discounted_price
  let total_cost := cost_regular + cost_discounted
  total_revenue - total_cost

theorem candy_fundraiser_profit :
  candy_fundraiser 5 3 10 (3/2) 1 (4/5) = 31 := by
  sorry

end NUMINAMATH_CALUDE_candy_fundraiser_profit_l1207_120763


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l1207_120790

theorem quadratic_equation_result (a : ℝ) (h : a^2 - 4*a - 12 = 0) : 2*a^2 - 8*a - 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l1207_120790


namespace NUMINAMATH_CALUDE_boat_stream_speed_ratio_l1207_120734

/-- Proves that if rowing against the stream takes twice as long as rowing with the stream,
    then the ratio of boat speed to stream speed is 3:1 -/
theorem boat_stream_speed_ratio
  (D : ℝ) -- Distance rowed
  (B : ℝ) -- Speed of the boat in still water
  (S : ℝ) -- Speed of the stream
  (hD : D > 0) -- Distance is positive
  (hB : B > 0) -- Boat speed is positive
  (hS : S > 0) -- Stream speed is positive
  (hBS : B > S) -- Boat is faster than the stream
  (h_time : D / (B - S) = 2 * (D / (B + S))) -- Time against stream is twice time with stream
  : B / S = 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_stream_speed_ratio_l1207_120734


namespace NUMINAMATH_CALUDE_notched_circle_distance_l1207_120767

theorem notched_circle_distance (r AB BC : ℝ) (h_r : r = Real.sqrt 75) 
  (h_AB : AB = 8) (h_BC : BC = 3) : ∃ (x y : ℝ), x^2 + y^2 = 65 ∧ 
  (x + AB)^2 + y^2 = r^2 ∧ x^2 + (y + BC)^2 = r^2 := by
  sorry

end NUMINAMATH_CALUDE_notched_circle_distance_l1207_120767


namespace NUMINAMATH_CALUDE_water_level_rise_l1207_120728

/-- Calculates the rise in water level when a cube is immersed in a rectangular vessel. -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length vessel_width : ℝ) : 
  cube_edge = 5 →
  vessel_length = 10 →
  vessel_width = 5 →
  (cube_edge^3) / (vessel_length * vessel_width) = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_water_level_rise_l1207_120728


namespace NUMINAMATH_CALUDE_ramesh_discount_percentage_l1207_120743

/-- The discount percentage Ramesh received on the refrigerator --/
def discount_percentage (purchase_price transport_cost installation_cost no_discount_sale_price : ℚ) : ℚ :=
  let labelled_price := no_discount_sale_price / 1.1
  let discount := labelled_price - purchase_price
  (discount / labelled_price) * 100

/-- Theorem stating the discount percentage Ramesh received --/
theorem ramesh_discount_percentage :
  let purchase_price : ℚ := 14500
  let transport_cost : ℚ := 125
  let installation_cost : ℚ := 250
  let no_discount_sale_price : ℚ := 20350
  abs (discount_percentage purchase_price transport_cost installation_cost no_discount_sale_price - 21.62) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ramesh_discount_percentage_l1207_120743


namespace NUMINAMATH_CALUDE_min_value_of_f_in_interval_l1207_120753

-- Define the function f(x) = x^3 - 12x
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the interval [-4, 4]
def interval : Set ℝ := Set.Icc (-4) 4

-- Theorem statement
theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), x ∈ interval ∧ f x = -16 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_in_interval_l1207_120753


namespace NUMINAMATH_CALUDE_find_divisor_l1207_120757

theorem find_divisor (dividend : Nat) (quotient : Nat) (h1 : dividend = 62976) (h2 : quotient = 123) :
  dividend / quotient = 512 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1207_120757


namespace NUMINAMATH_CALUDE_jellybean_count_l1207_120768

theorem jellybean_count (initial_count : ℕ) : 
  (initial_count : ℝ) * (0.7 ^ 3) = 28 → initial_count = 82 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l1207_120768


namespace NUMINAMATH_CALUDE_triangle_rotation_theorem_l1207_120795

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  O : Point
  P : Point
  Q : Point

/-- Rotates a point -90° around the origin -/
def rotate90Clockwise (p : Point) : Point :=
  { x := p.y, y := -p.x }

theorem triangle_rotation_theorem (t : Triangle) :
  t.O = { x := 0, y := 0 } →
  t.Q = { x := 3, y := 0 } →
  t.P.x > 0 →
  t.P.y > 0 →
  (t.P.y - t.O.y) / (t.P.x - t.O.x) = 1 →
  (t.Q.x - t.O.x) * (t.P.x - t.Q.x) + (t.Q.y - t.O.y) * (t.P.y - t.Q.y) = 0 →
  rotate90Clockwise t.P = { x := 3, y := -3 } := by
  sorry

#check triangle_rotation_theorem

end NUMINAMATH_CALUDE_triangle_rotation_theorem_l1207_120795


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l1207_120779

theorem least_positive_integer_with_remainder_one : ∃ n : ℕ,
  n > 1 ∧
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 10 → n % k = 1) ∧
  (∀ m : ℕ, m > 1 ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ 10 → m % k = 1) → n ≤ m) ∧
  n = 2521 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l1207_120779


namespace NUMINAMATH_CALUDE_smallest_number_l1207_120702

/-- Converts a number from base b to decimal --/
def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- The decimal representation of 85₍₉₎ --/
def num1 : Nat := toDecimal [5, 8] 9

/-- The decimal representation of 210₍₆₎ --/
def num2 : Nat := toDecimal [0, 1, 2] 6

/-- The decimal representation of 1000₍₄₎ --/
def num3 : Nat := toDecimal [0, 0, 0, 1] 4

/-- The decimal representation of 111111₍₂₎ --/
def num4 : Nat := toDecimal [1, 1, 1, 1, 1, 1] 2

/-- Theorem stating that 111111₍₂₎ is the smallest among the given numbers --/
theorem smallest_number : num4 ≤ num1 ∧ num4 ≤ num2 ∧ num4 ≤ num3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1207_120702


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1207_120713

theorem right_triangle_inequality (a b c x : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a^2 = b^2 + c^2 → 
  a ≥ b ∧ a ≥ c → 
  (a^x > b^x + c^x ↔ x > 2) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1207_120713


namespace NUMINAMATH_CALUDE_sum_of_u_and_v_l1207_120731

theorem sum_of_u_and_v (u v : ℚ) 
  (eq1 : 3 * u - 4 * v = 17) 
  (eq2 : 5 * u - 2 * v = 1) : 
  u + v = -8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_u_and_v_l1207_120731


namespace NUMINAMATH_CALUDE_journey_equations_l1207_120719

theorem journey_equations (total_time bike_speed walk_speed total_distance : ℝ)
  (h_total_time : total_time = 20)
  (h_bike_speed : bike_speed = 200)
  (h_walk_speed : walk_speed = 70)
  (h_total_distance : total_distance = 3350) :
  ∃ x y : ℝ,
    x + y = total_time ∧
    bike_speed * x + walk_speed * y = total_distance :=
by sorry

end NUMINAMATH_CALUDE_journey_equations_l1207_120719


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_five_primes_l1207_120710

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem arithmetic_mean_reciprocals_first_five_primes :
  let reciprocals := first_five_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length) = 2927 / 11550 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocals_first_five_primes_l1207_120710


namespace NUMINAMATH_CALUDE_greatest_x_implies_n_l1207_120706

theorem greatest_x_implies_n (x : ℤ) (n : ℝ) : 
  (∀ y : ℤ, 2.13 * (10 : ℝ) ^ y < n → y ≤ 2) →
  (2.13 * (10 : ℝ) ^ 2 < n) ∧
  (∀ m : ℝ, m < n → m ≤ 213) ∧
  (n ≥ 214) :=
sorry

end NUMINAMATH_CALUDE_greatest_x_implies_n_l1207_120706


namespace NUMINAMATH_CALUDE_sequence_general_term_l1207_120717

theorem sequence_general_term (a : ℕ+ → ℚ) :
  a 1 = 1 ∧
  (∀ n : ℕ+, a (n + 1) = (2 * a n) / (2 + a n)) →
  ∀ n : ℕ+, a n = 2 / (n + 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1207_120717


namespace NUMINAMATH_CALUDE_only_setC_is_right_triangle_l1207_120737

-- Define a function to check if three numbers satisfy the Pythagorean theorem
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

-- Define the sets of line segments
def setA : List ℕ := [1, 2, 3]
def setB : List ℕ := [5, 11, 12]
def setC : List ℕ := [5, 12, 13]
def setD : List ℕ := [6, 8, 9]

-- Theorem stating that only setC forms a right triangle
theorem only_setC_is_right_triangle :
  (¬ isPythagoreanTriple setA[0]! setA[1]! setA[2]!) ∧
  (¬ isPythagoreanTriple setB[0]! setB[1]! setB[2]!) ∧
  (isPythagoreanTriple setC[0]! setC[1]! setC[2]!) ∧
  (¬ isPythagoreanTriple setD[0]! setD[1]! setD[2]!) :=
by sorry

end NUMINAMATH_CALUDE_only_setC_is_right_triangle_l1207_120737


namespace NUMINAMATH_CALUDE_marble_jar_problem_l1207_120796

theorem marble_jar_problem (M : ℕ) : 
  (∀ (x : ℕ), x = M / 16 → x - 1 = M / 18) → M = 144 := by
  sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l1207_120796


namespace NUMINAMATH_CALUDE_no_solution_cosine_sine_equation_l1207_120742

theorem no_solution_cosine_sine_equation :
  ∀ x : ℝ, Real.cos (Real.cos (Real.cos (Real.cos x))) > Real.sin (Real.sin (Real.sin (Real.sin x))) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_cosine_sine_equation_l1207_120742


namespace NUMINAMATH_CALUDE_one_sixth_of_x_l1207_120752

theorem one_sixth_of_x (x : ℝ) (h : x / 3 = 4) : x / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_one_sixth_of_x_l1207_120752


namespace NUMINAMATH_CALUDE_truncated_pyramid_ratio_l1207_120776

/-- Given a right prism with a square base of side length L₁ and height H, 
    and a truncated pyramid extracted from it with square bases of side lengths 
    L₁ (bottom) and L₂ (top) and height H, if the volume of the truncated pyramid 
    is 2/3 of the total volume of the prism, then L₁/L₂ = (1 + √5) / 2. -/
theorem truncated_pyramid_ratio (L₁ L₂ H : ℝ) (h₁ : L₁ > 0) (h₂ : L₂ > 0) (h₃ : H > 0) :
  (H / 3 * (L₁^2 + L₁*L₂ + L₂^2) = 2/3 * H * L₁^2) → L₁ / L₂ = (1 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_ratio_l1207_120776


namespace NUMINAMATH_CALUDE_shopping_expense_l1207_120756

theorem shopping_expense (initial_amount : ℝ) (amount_left : ℝ) : 
  initial_amount = 158 →
  amount_left = 78 →
  ∃ (shoes_price bag_price lunch_price : ℝ),
    bag_price = shoes_price - 17 ∧
    lunch_price = bag_price / 4 ∧
    initial_amount = shoes_price + bag_price + lunch_price + amount_left ∧
    shoes_price = 45 := by
  sorry

end NUMINAMATH_CALUDE_shopping_expense_l1207_120756


namespace NUMINAMATH_CALUDE_A_inverse_proof_l1207_120735

def A : Matrix (Fin 3) (Fin 3) ℚ := !![2, 5, 6; 1, 2, 5; 1, 2, 3]

def A_inv : Matrix (Fin 3) (Fin 3) ℚ := !![-2, 3/2, 13/2; 1, 0, 2; 0, -1/2, -1/2]

theorem A_inverse_proof : A⁻¹ = A_inv := by sorry

end NUMINAMATH_CALUDE_A_inverse_proof_l1207_120735


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l1207_120741

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The problem statement -/
theorem base_conversion_subtraction :
  let base_7_num := to_base_10 [0, 3, 4, 2, 5] 7
  let base_8_num := to_base_10 [0, 2, 3, 4] 8
  base_7_num - base_8_num = 10652 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l1207_120741


namespace NUMINAMATH_CALUDE_total_triangles_in_4_layer_grid_l1207_120781

/-- Represents a triangular grid with a given number of layers -/
def TriangularGrid (layers : ℕ) : Type := Unit

/-- Counts the number of small triangles in a triangular grid -/
def countSmallTriangles (grid : TriangularGrid 4) : ℕ := 10

/-- Counts the number of medium triangles (made of 4 small triangles) in a triangular grid -/
def countMediumTriangles (grid : TriangularGrid 4) : ℕ := 6

/-- Counts the number of large triangles (made of 9 small triangles) in a triangular grid -/
def countLargeTriangles (grid : TriangularGrid 4) : ℕ := 1

/-- The total number of triangles in a 4-layer triangular grid is 17 -/
theorem total_triangles_in_4_layer_grid (grid : TriangularGrid 4) :
  countSmallTriangles grid + countMediumTriangles grid + countLargeTriangles grid = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_in_4_layer_grid_l1207_120781


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1207_120791

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - m*x₁ - 1 = 0 ∧ x₂^2 - m*x₂ - 1 = 0 ∧ x₁ > 2 ∧ x₂ < 2) ↔ 
  m > 3/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1207_120791


namespace NUMINAMATH_CALUDE_zero_point_existence_l1207_120749

def f (x : ℝ) := x^3 + 2*x - 5

theorem zero_point_existence :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  have h1 : Continuous f := sorry
  have h2 : f 1 < 0 := sorry
  have h3 : f 2 > 0 := sorry
  sorry

end NUMINAMATH_CALUDE_zero_point_existence_l1207_120749


namespace NUMINAMATH_CALUDE_perpendicular_condition_l1207_120709

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the "contained in" relation
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_condition 
  (a : Line) (α β : Plane) 
  (h_contained : contained_in a α) :
  (∀ β, perpendicular a β → perpendicular_planes α β) ∧ 
  (∃ β, perpendicular_planes α β ∧ ¬perpendicular a β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l1207_120709


namespace NUMINAMATH_CALUDE_special_sequence_1000th_term_l1207_120798

def special_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1010 ∧ 
  a 2 = 1015 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = 2 * n + 1

theorem special_sequence_1000th_term (a : ℕ → ℕ) 
  (h : special_sequence a) : a 1000 = 1676 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_1000th_term_l1207_120798


namespace NUMINAMATH_CALUDE_first_year_growth_rate_l1207_120720

/-- Proves that given the initial and final populations, and the second year's growth rate,
    the first year's growth rate is 22%. -/
theorem first_year_growth_rate (initial_pop : ℕ) (final_pop : ℕ) (second_year_rate : ℚ) :
  initial_pop = 800 →
  final_pop = 1220 →
  second_year_rate = 25 / 100 →
  ∃ (first_year_rate : ℚ),
    first_year_rate = 22 / 100 ∧
    final_pop = initial_pop * (1 + first_year_rate) * (1 + second_year_rate) :=
by sorry

end NUMINAMATH_CALUDE_first_year_growth_rate_l1207_120720


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l1207_120770

/-- The 15th term of an arithmetic sequence with first term -3 and common difference 4 is 53. -/
theorem arithmetic_sequence_15th_term :
  let a : ℕ → ℤ := fun n => -3 + (n - 1) * 4
  a 15 = 53 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l1207_120770


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l1207_120701

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ y : ℤ, x^4 + x^3 + x^2 + x + 1 = y^2) ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l1207_120701


namespace NUMINAMATH_CALUDE_number_problem_l1207_120700

theorem number_problem (x : ℝ) : x - (3/5) * x = 64 → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1207_120700


namespace NUMINAMATH_CALUDE_julio_has_seven_grape_bottles_l1207_120705

-- Define the number of bottles and liters
def julio_orange_bottles : ℕ := 4
def mateo_orange_bottles : ℕ := 1
def mateo_grape_bottles : ℕ := 3
def liters_per_bottle : ℕ := 2
def julio_extra_liters : ℕ := 14

-- Define the function to calculate the number of grape bottles Julio has
def julio_grape_bottles : ℕ :=
  let mateo_total_liters := (mateo_orange_bottles + mateo_grape_bottles) * liters_per_bottle
  let julio_total_liters := mateo_total_liters + julio_extra_liters
  let julio_grape_liters := julio_total_liters - (julio_orange_bottles * liters_per_bottle)
  julio_grape_liters / liters_per_bottle

-- State the theorem
theorem julio_has_seven_grape_bottles :
  julio_grape_bottles = 7 := by sorry

end NUMINAMATH_CALUDE_julio_has_seven_grape_bottles_l1207_120705


namespace NUMINAMATH_CALUDE_paul_work_days_l1207_120754

/-- The number of days it takes Rose to complete the work -/
def rose_days : ℝ := 120

/-- The number of days it takes Paul and Rose together to complete the work -/
def combined_days : ℝ := 48

/-- The number of days it takes Paul to complete the work alone -/
def paul_days : ℝ := 80

/-- Theorem stating that given Rose's and combined work rates, Paul's individual work rate can be determined -/
theorem paul_work_days (rose : ℝ) (combined : ℝ) (paul : ℝ) 
  (h_rose : rose = rose_days) 
  (h_combined : combined = combined_days) :
  paul = paul_days :=
by sorry

end NUMINAMATH_CALUDE_paul_work_days_l1207_120754
