import Mathlib

namespace NUMINAMATH_CALUDE_food_distribution_l3154_315444

/-- The initial number of men for whom the food lasts 50 days -/
def initial_men : ℕ := sorry

/-- The number of days the food lasts for the initial group -/
def initial_days : ℕ := 50

/-- The number of additional men who join -/
def additional_men : ℕ := 20

/-- The number of days the food lasts after additional men join -/
def new_days : ℕ := 25

/-- Theorem stating that the initial number of men is 20 -/
theorem food_distribution :
  initial_men * initial_days = (initial_men + additional_men) * new_days ∧
  initial_men = 20 := by sorry

end NUMINAMATH_CALUDE_food_distribution_l3154_315444


namespace NUMINAMATH_CALUDE_expression_behavior_l3154_315433

theorem expression_behavior (x : ℝ) (h : -3 < x ∧ x < 2) :
  (x^2 + 4*x + 5) / (2*x + 6) ≥ 3/4 ∧
  ((x^2 + 4*x + 5) / (2*x + 6) = 3/4 ↔ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_expression_behavior_l3154_315433


namespace NUMINAMATH_CALUDE_only_first_statement_true_l3154_315411

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (plane_intersection : Plane → Plane → Line)

-- Axioms for parallel and perpendicular relations
axiom parallel_transitive {l1 l2 l3 : Line} : parallel l1 l2 → parallel l2 l3 → parallel l1 l3
axiom perpendicular_not_parallel {l1 l2 : Line} : perpendicular l1 l2 → ¬ parallel l1 l2
axiom plane_perpendicular_not_parallel {p1 p2 : Plane} : plane_perpendicular p1 p2 → ¬ plane_parallel p1 p2

-- The main theorem
theorem only_first_statement_true 
  (a b c : Line) (α β γ : Plane) 
  (h_distinct_lines : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (parallel a b ∧ parallel b c → parallel a c) ∧
  ¬(perpendicular a b ∧ perpendicular b c → parallel a c) ∧
  ¬(plane_perpendicular α β ∧ plane_perpendicular β γ → plane_parallel α γ) ∧
  ¬(plane_perpendicular α β ∧ plane_intersection α β = a ∧ perpendicular b a → line_perpendicular_to_plane b β) :=
sorry

end NUMINAMATH_CALUDE_only_first_statement_true_l3154_315411


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3154_315450

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℝ) (d S : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  (∀ n, a n > 0) →  -- positivity condition
  (∀ n, S * n = (n / 2) * (2 * a 1 + (n - 1) * d)) →  -- sum formula
  (a 2) * (a 2 + S * 5) = (S * 3) ^ 2 →  -- geometric sequence condition
  d / a 1 = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l3154_315450


namespace NUMINAMATH_CALUDE_incorrect_fraction_transformation_l3154_315471

theorem incorrect_fraction_transformation (a b : ℝ) (hb : b ≠ 0) :
  ¬(∀ (a b : ℝ), b ≠ 0 → |(-a)| / b = a / (-b)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_fraction_transformation_l3154_315471


namespace NUMINAMATH_CALUDE_spinner_probability_l3154_315497

theorem spinner_probability (p_A p_B p_C : ℚ) : 
  p_A = 1/3 → p_B = 1/2 → p_A + p_B + p_C = 1 → p_C = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3154_315497


namespace NUMINAMATH_CALUDE_count_special_numbers_l3154_315468

/-- Counts the number of four-digit numbers with digit sum 12 that are divisible by 9 -/
def countSpecialNumbers : ℕ :=
  (Finset.range 9).sum fun a =>
    Nat.choose (14 - (a + 1)) 2

/-- The count of four-digit numbers with digit sum 12 that are divisible by 9 is 354 -/
theorem count_special_numbers : countSpecialNumbers = 354 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_l3154_315468


namespace NUMINAMATH_CALUDE_sum_of_odd_integers_13_to_41_l3154_315495

theorem sum_of_odd_integers_13_to_41 :
  let first_term : ℕ := 13
  let last_term : ℕ := 41
  let common_difference : ℕ := 2
  let n : ℕ := (last_term - first_term) / common_difference + 1
  (n : ℝ) / 2 * (first_term + last_term) = 405 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_odd_integers_13_to_41_l3154_315495


namespace NUMINAMATH_CALUDE_product_of_reals_l3154_315481

theorem product_of_reals (a b : ℝ) (sum_eq : a + b = 10) (sum_cubes_eq : a^3 + b^3 = 172) :
  a * b = 27.6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reals_l3154_315481


namespace NUMINAMATH_CALUDE_class_mean_score_l3154_315470

theorem class_mean_score (total_students : ℕ) (first_day_students : ℕ) (second_day_students : ℕ)
  (first_day_mean : ℚ) (second_day_mean : ℚ) :
  total_students = 50 →
  first_day_students = 40 →
  second_day_students = 10 →
  first_day_mean = 80 / 100 →
  second_day_mean = 90 / 100 →
  let overall_mean := (first_day_students * first_day_mean + second_day_students * second_day_mean) / total_students
  overall_mean = 82 / 100 := by
sorry

end NUMINAMATH_CALUDE_class_mean_score_l3154_315470


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3154_315480

-- Define the universal set U
def U : Finset ℕ := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset ℕ := {1, 4}

-- Define set N
def N : Finset ℕ := {1, 3, 5}

-- Theorem statement
theorem intersection_complement_equality :
  N ∩ (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3154_315480


namespace NUMINAMATH_CALUDE_initial_distance_between_trains_l3154_315482

def train_length_1 : ℝ := 120
def train_length_2 : ℝ := 210
def speed_1 : ℝ := 69
def speed_2 : ℝ := 82
def meeting_time : ℝ := 1.9071321976361095

theorem initial_distance_between_trains : 
  let relative_speed := (speed_1 + speed_2) * 1000 / 3600
  let distance_covered := relative_speed * (meeting_time * 3600)
  distance_covered - (train_length_1 + train_length_2) = 287670 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_between_trains_l3154_315482


namespace NUMINAMATH_CALUDE_set_intersection_range_l3154_315443

theorem set_intersection_range (a : ℝ) : 
  let A : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
  let B : Set ℝ := {x | x < -1 ∨ x > 16}
  A ∩ B = A → a < 6 ∨ a > 7.5 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_range_l3154_315443


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l3154_315459

theorem geometric_progression_ratio (x y z r : ℂ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z →
  ∃ (a : ℂ), x * (y - z) = a ∧ y * (z - x) = a * r ∧ z * (x - y) = a * r^2 →
  x + y + z = 0 →
  r^2 + r + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l3154_315459


namespace NUMINAMATH_CALUDE_all_statements_false_l3154_315429

theorem all_statements_false :
  (¬ ∀ a b : ℝ, a > b → a^2 > b^2) ∧
  (¬ ∀ a b : ℝ, a^2 > b^2 → a > b) ∧
  (¬ ∀ a b c : ℝ, a > b → a*c^2 > b*c^2) ∧
  (¬ ∀ a b : ℝ, (a > b ↔ |a| > |b|)) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_false_l3154_315429


namespace NUMINAMATH_CALUDE_longest_tape_l3154_315489

theorem longest_tape (minji seungyeon hyesu : ℝ) 
  (h_minji : minji = 0.74)
  (h_seungyeon : seungyeon = 13/20)
  (h_hyesu : hyesu = 4/5) :
  hyesu > minji ∧ hyesu > seungyeon :=
by sorry

end NUMINAMATH_CALUDE_longest_tape_l3154_315489


namespace NUMINAMATH_CALUDE_halloween_candy_l3154_315476

theorem halloween_candy (debby_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : 
  debby_candy = 32 → sister_candy = 42 → eaten_candy = 35 →
  debby_candy + sister_candy - eaten_candy = 39 := by
sorry

end NUMINAMATH_CALUDE_halloween_candy_l3154_315476


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3154_315456

theorem complex_fraction_simplification :
  (3 + 8 * Complex.I) / (1 - 4 * Complex.I) = -29/17 + 20/17 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3154_315456


namespace NUMINAMATH_CALUDE_one_way_cost_proof_l3154_315493

/-- Represents the cost of one-way travel from home to office -/
def one_way_cost : ℝ := 16

/-- Represents the total number of trips in 9 working days -/
def total_trips : ℕ := 18

/-- Represents the total cost of travel for 9 working days -/
def total_cost : ℝ := 288

/-- Theorem stating that the one-way cost multiplied by the total number of trips
    equals the total cost for 9 working days -/
theorem one_way_cost_proof :
  one_way_cost * (total_trips : ℝ) = total_cost := by sorry

end NUMINAMATH_CALUDE_one_way_cost_proof_l3154_315493


namespace NUMINAMATH_CALUDE_arbitrarily_large_special_numbers_l3154_315401

/-- A function that checks if all digits of a natural number are 2 or more -/
def all_digits_two_or_more (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≥ 2

/-- A function that checks if the product of any four digits of a number divides the number -/
def product_of_four_divides (n : ℕ) : Prop :=
  ∀ a b c d, a ∈ n.digits 10 → b ∈ n.digits 10 → c ∈ n.digits 10 → d ∈ n.digits 10 →
    (a * b * c * d) ∣ n

/-- The main theorem stating that for any k, there exists a number n > k satisfying the conditions -/
theorem arbitrarily_large_special_numbers :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ all_digits_two_or_more n ∧ product_of_four_divides n :=
sorry

end NUMINAMATH_CALUDE_arbitrarily_large_special_numbers_l3154_315401


namespace NUMINAMATH_CALUDE_max_intersections_is_19_l3154_315461

/-- The maximum number of intersection points between three circles -/
def max_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between a line and three circles -/
def max_line_circle_intersections : ℕ := 6

/-- The number of lines -/
def num_lines : ℕ := 2

/-- The number of intersection points between two lines -/
def line_line_intersections : ℕ := 1

/-- The maximum number of intersection points between 3 circles and 2 straight lines on a plane -/
def max_total_intersections : ℕ :=
  max_circle_intersections + 
  (num_lines * max_line_circle_intersections) + 
  line_line_intersections

theorem max_intersections_is_19 : 
  max_total_intersections = 19 := by sorry

end NUMINAMATH_CALUDE_max_intersections_is_19_l3154_315461


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l3154_315499

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (2*a - 1)*x + a + 1

-- Define monotonicity in an interval
def monotonic_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ ∀ z, a < z ∧ z < b → f z = f x)

-- State the theorem
theorem quadratic_monotonicity (a : ℝ) :
  monotonic_in (f a) 1 2 → (a ≥ 5/2 ∨ a ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l3154_315499


namespace NUMINAMATH_CALUDE_domain_of_function_l3154_315405

/-- The domain of the function f(x) = √(2x-1) / (x^2 + x - 2) -/
theorem domain_of_function (x : ℝ) : 
  x ∈ {y : ℝ | y ≥ (1/2 : ℝ) ∧ y ≠ 1} ↔ 
    (2*x - 1 ≥ 0 ∧ x^2 + x - 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_domain_of_function_l3154_315405


namespace NUMINAMATH_CALUDE_melany_candy_l3154_315425

theorem melany_candy (hugh tommy melany : ℕ) (total_after : ℕ) :
  hugh = 8 →
  tommy = 6 →
  total_after = 7 * 3 →
  hugh + tommy + melany = total_after →
  melany = 7 :=
by sorry

end NUMINAMATH_CALUDE_melany_candy_l3154_315425


namespace NUMINAMATH_CALUDE_tangent_line_problem_l3154_315457

/-- Given two functions f and g, where f is the natural logarithm and g is a quadratic function with parameter m,
    and a line l tangent to both f and g at the point (1, 0), prove that m = -2. -/
theorem tangent_line_problem (m : ℝ) :
  (m < 0) →
  let f : ℝ → ℝ := λ x ↦ Real.log x
  let g : ℝ → ℝ := λ x ↦ (1/2) * x^2 + m * x + 7/2
  let l : ℝ → ℝ := λ x ↦ x - 1
  (∀ x, deriv f x = 1/x) →
  (∀ x, deriv g x = x + m) →
  (f 1 = 0) →
  (g 1 = l 1) →
  (deriv f 1 = deriv l 1) →
  (∃ x, g x = l x ∧ deriv g x = deriv l x) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l3154_315457


namespace NUMINAMATH_CALUDE_triangle_area_with_cosine_root_l3154_315417

/-- The area of a triangle with two sides of length 3 and 5, where the cosine of the angle between them is a root of 5x^2 - 7x - 6 = 0, is equal to 6. -/
theorem triangle_area_with_cosine_root : ∃ (θ : ℝ), 
  (5 * (Real.cos θ)^2 - 7 * (Real.cos θ) - 6 = 0) →
  (1/2 * 3 * 5 * Real.sin θ = 6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_cosine_root_l3154_315417


namespace NUMINAMATH_CALUDE_unique_distribution_function_decomposition_l3154_315442

/-- A distribution function -/
class DistributionFunction (F : ℝ → ℝ) : Prop where
  -- Add necessary axioms for a distribution function

/-- A discrete distribution function -/
class DiscreteDistributionFunction (F : ℝ → ℝ) extends DistributionFunction F : Prop where
  -- Add necessary axioms for a discrete distribution function

/-- An absolutely continuous distribution function -/
class AbsContinuousDistributionFunction (F : ℝ → ℝ) extends DistributionFunction F : Prop where
  -- Add necessary axioms for an absolutely continuous distribution function

/-- A singular distribution function -/
class SingularDistributionFunction (F : ℝ → ℝ) extends DistributionFunction F : Prop where
  -- Add necessary axioms for a singular distribution function

/-- The uniqueness of distribution function decomposition -/
theorem unique_distribution_function_decomposition
  (F : ℝ → ℝ) [DistributionFunction F] :
  ∃! (α₁ α₂ α₃ : ℝ) (Fₐ Fₐbc Fsc : ℝ → ℝ),
    α₁ ≥ 0 ∧ α₂ ≥ 0 ∧ α₃ ≥ 0 ∧
    α₁ + α₂ + α₃ = 1 ∧
    DiscreteDistributionFunction Fₐ ∧
    AbsContinuousDistributionFunction Fₐbc ∧
    SingularDistributionFunction Fsc ∧
    F = λ x => α₁ * Fₐ x + α₂ * Fₐbc x + α₃ * Fsc x :=
by sorry

end NUMINAMATH_CALUDE_unique_distribution_function_decomposition_l3154_315442


namespace NUMINAMATH_CALUDE_concatenated_numbers_divisible_by_45_l3154_315438

def concatenate_numbers (n : ℕ) : ℕ :=
  -- Definition of concatenating numbers from 1 to n
  sorry

theorem concatenated_numbers_divisible_by_45 :
  ∃ k : ℕ, concatenate_numbers 50 = 45 * k := by
  sorry

end NUMINAMATH_CALUDE_concatenated_numbers_divisible_by_45_l3154_315438


namespace NUMINAMATH_CALUDE_particle_probability_l3154_315488

/-- Probability of reaching (0, 0) from (x, y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3

/-- The probability of reaching (0, 0) from (6, 6) is 855/3^12 -/
theorem particle_probability : P 6 6 = 855 / 3^12 := by
  sorry

end NUMINAMATH_CALUDE_particle_probability_l3154_315488


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3154_315478

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  (a = b ∧ (a = 2 ∧ c = 5 ∨ a = 5 ∧ c = 2) ∨
   a = c ∧ (a = 2 ∧ b = 5 ∨ a = 5 ∧ b = 2) ∨
   b = c ∧ (b = 2 ∧ a = 5 ∨ b = 5 ∧ a = 2)) →  -- isosceles with sides 2 and 5
  a + b + c = 12  -- perimeter is 12
:= by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3154_315478


namespace NUMINAMATH_CALUDE_optimal_price_reduction_maximizes_profit_l3154_315490

/-- Profit function for shirt sales based on price reduction -/
def profit (x : ℝ) : ℝ := (2 * x + 20) * (40 - x)

/-- The price reduction that maximizes profit -/
def optimal_reduction : ℝ := 15

theorem optimal_price_reduction_maximizes_profit :
  ∀ x : ℝ, 0 ≤ x → x ≤ 40 → profit x ≤ profit optimal_reduction := by
  sorry

#check optimal_price_reduction_maximizes_profit

end NUMINAMATH_CALUDE_optimal_price_reduction_maximizes_profit_l3154_315490


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3154_315421

/-- The minimum distance from the origin (0, 0) to the line 2x + y + 5 = 0 is √5 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | 2 * p.1 + p.2 + 5 = 0}
  ∃ d : ℝ, d = Real.sqrt 5 ∧ ∀ p ∈ line, d ≤ Real.sqrt (p.1^2 + p.2^2) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3154_315421


namespace NUMINAMATH_CALUDE_min_value_expression_l3154_315492

open Real

theorem min_value_expression (α β : ℝ) (h : α + β = π / 2) :
  (∀ x y : ℝ, (3 * cos α + 4 * sin β - 10)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 65) ∧
  (∃ x y : ℝ, (3 * cos α + 4 * sin β - 10)^2 + (3 * sin α + 4 * cos β - 12)^2 = 65) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3154_315492


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l3154_315430

/-- A rectangular prism with its properties -/
structure RectangularPrism where
  vertices : Nat
  edges : Nat
  dimensions : Nat
  has_face_diagonals : Bool
  has_space_diagonals : Bool

/-- The total number of diagonals in a rectangular prism -/
def total_diagonals (prism : RectangularPrism) : Nat :=
  sorry

/-- Theorem stating that the total number of diagonals in a rectangular prism is 16 -/
theorem rectangular_prism_diagonals :
  ∀ (prism : RectangularPrism),
    prism.vertices = 8 ∧
    prism.edges = 12 ∧
    prism.dimensions = 3 ∧
    prism.has_face_diagonals = true ∧
    prism.has_space_diagonals = true →
    total_diagonals prism = 16 :=
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l3154_315430


namespace NUMINAMATH_CALUDE_stevens_grapes_l3154_315475

def apple_seeds : ℕ := 6
def pear_seeds : ℕ := 2
def grape_seeds : ℕ := 3
def total_seeds_needed : ℕ := 60
def apples_set_aside : ℕ := 4
def pears_set_aside : ℕ := 3
def additional_seeds_needed : ℕ := 3

theorem stevens_grapes (grapes_set_aside : ℕ) : grapes_set_aside = 9 := by
  sorry

#check stevens_grapes

end NUMINAMATH_CALUDE_stevens_grapes_l3154_315475


namespace NUMINAMATH_CALUDE_decreasing_function_implies_a_bound_l3154_315474

/-- A function f: ℝ → ℝ is decreasing if for all x, y ∈ ℝ, x < y implies f(x) > f(y) -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

/-- The function f(x) = -x³ + x² + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + x^2 + a*x

theorem decreasing_function_implies_a_bound :
  ∀ a : ℝ, DecreasingFunction (f a) → a ≤ -1/3 := by sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_a_bound_l3154_315474


namespace NUMINAMATH_CALUDE_square_of_integer_l3154_315487

theorem square_of_integer (n : ℕ+) (h : ∃ (m : ℤ), m = 2 + 2 * Int.sqrt (28 * n.val^2 + 1)) :
  ∃ (k : ℤ), (2 + 2 * Int.sqrt (28 * n.val^2 + 1)) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_integer_l3154_315487


namespace NUMINAMATH_CALUDE_sin_increases_with_angle_sum_of_cosines_positive_l3154_315408

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  -- Ensure the angles form a triangle
  angle_sum : A + B + C = π
  -- Ensure all sides and angles are positive
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0
  positive_angles : A > 0 ∧ B > 0 ∧ C > 0

-- Theorem 1: If angle A is greater than angle B, then sin A is greater than sin B
theorem sin_increases_with_angle (t : Triangle) :
  t.A > t.B → Real.sin t.A > Real.sin t.B := by
  sorry

-- Theorem 2: The sum of cosines of all three angles is always positive
theorem sum_of_cosines_positive (t : Triangle) :
  Real.cos t.A + Real.cos t.B + Real.cos t.C > 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_increases_with_angle_sum_of_cosines_positive_l3154_315408


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3154_315485

noncomputable def line_through_origin (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1}

def vertical_line (x : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = x}

def sloped_line (m : ℝ) (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 + b}

def is_equilateral_triangle (t : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ × ℝ), a ∈ t ∧ b ∈ t ∧ c ∈ t ∧
    (a.1 - b.1)^2 + (a.2 - b.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∧
    (b.1 - c.1)^2 + (b.2 - c.2)^2 = (c.1 - a.1)^2 + (c.2 - a.2)^2

def perimeter (t : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem triangle_perimeter :
  ∃ (m : ℝ),
    let l1 := line_through_origin m
    let l2 := vertical_line 1
    let l3 := sloped_line (Real.sqrt 3 / 3) 1
    let t := l1 ∪ l2 ∪ l3
    is_equilateral_triangle t ∧ perimeter t = 3 + 2 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3154_315485


namespace NUMINAMATH_CALUDE_green_ball_probability_l3154_315413

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers X, Y, and Z -/
def X : Container := ⟨3, 7⟩
def Y : Container := ⟨8, 2⟩
def Z : Container := ⟨5, 5⟩

/-- The list of all containers -/
def containers : List Container := [X, Y, Z]

/-- The probability of selecting a green ball -/
def probabilityGreenBall : ℚ :=
  (List.sum (containers.map greenProbability)) / containers.length

theorem green_ball_probability :
  probabilityGreenBall = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l3154_315413


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3154_315455

theorem quadratic_equation_solutions : ∀ x : ℝ,
  (2 * x^2 + 7 * x - 1 = 4 * x + 1 ↔ x = -2 ∨ x = 1/2) ∧
  (2 * x^2 + 7 * x - 1 = -(x^2 - 19) ↔ x = -4 ∨ x = 5/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3154_315455


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3154_315419

/-- The set P -/
def P : Set ℝ := {x : ℝ | -5 < x ∧ x < 5}

/-- The set Q -/
def Q : Set ℝ := {x : ℝ | |x - 5| < 3}

/-- The open interval (2, 5) -/
def open_interval_2_5 : Set ℝ := {x : ℝ | 2 < x ∧ x < 5}

theorem intersection_P_Q : P ∩ Q = open_interval_2_5 := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3154_315419


namespace NUMINAMATH_CALUDE_mrs_hilt_apples_l3154_315436

/-- Calculates the total number of apples eaten given a rate and time period. -/
def applesEaten (rate : ℕ) (hours : ℕ) : ℕ := rate * hours

/-- Theorem stating that eating 5 apples per hour for 3 hours results in 15 apples eaten. -/
theorem mrs_hilt_apples : applesEaten 5 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_apples_l3154_315436


namespace NUMINAMATH_CALUDE_log_inequality_l3154_315414

theorem log_inequality (k : ℝ) (h : k ≥ 3) :
  Real.log k / Real.log (k - 1) > Real.log (k + 1) / Real.log k := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3154_315414


namespace NUMINAMATH_CALUDE_intersection_of_M_and_S_l3154_315406

def M : Set ℕ := {x | 0 < x ∧ x < 4}
def S : Set ℕ := {2, 3, 5}

theorem intersection_of_M_and_S : M ∩ S = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_S_l3154_315406


namespace NUMINAMATH_CALUDE_bowl_glass_pairings_l3154_315477

/-- The number of bowls -/
def num_bowls : ℕ := 5

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The total number of possible pairings -/
def total_pairings : ℕ := num_bowls * num_glasses

/-- Theorem: The number of possible pairings of bowls and glasses is 25 -/
theorem bowl_glass_pairings :
  total_pairings = 25 := by sorry

end NUMINAMATH_CALUDE_bowl_glass_pairings_l3154_315477


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3154_315458

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def P : Set Nat := {1, 3, 5}
def Q : Set Nat := {1, 2, 4}

theorem complement_union_theorem :
  (U \ P) ∪ Q = {1, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3154_315458


namespace NUMINAMATH_CALUDE_video_recorder_markup_percentage_l3154_315402

/-- Proves that the percentage markup on a video recorder's wholesale cost is 20%,
    given the wholesale cost, employee discount, and final price paid by the employee. -/
theorem video_recorder_markup_percentage
  (wholesale_cost : ℝ)
  (employee_discount_percent : ℝ)
  (employee_paid_price : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : employee_discount_percent = 10)
  (h3 : employee_paid_price = 216) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discounted_price := retail_price * (1 - employee_discount_percent / 100)
  markup_percentage = 20 :=
sorry

end NUMINAMATH_CALUDE_video_recorder_markup_percentage_l3154_315402


namespace NUMINAMATH_CALUDE_rect_to_spherical_conversion_l3154_315422

/-- Conversion from rectangular to spherical coordinates -/
theorem rect_to_spherical_conversion
  (x y z : ℝ)
  (ρ θ φ : ℝ)
  (h_ρ : ρ > 0)
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (h_φ : 0 ≤ φ ∧ φ ≤ Real.pi)
  (h_x : x = 0)
  (h_y : y = -3 * Real.sqrt 3)
  (h_z : z = 3)
  (h_ρ_val : ρ = 6)
  (h_θ_val : θ = 3 * Real.pi / 2)
  (h_φ_val : φ = Real.pi / 3) :
  x = ρ * Real.sin φ * Real.cos θ ∧
  y = ρ * Real.sin φ * Real.sin θ ∧
  z = ρ * Real.cos φ :=
by
  sorry

#check rect_to_spherical_conversion

end NUMINAMATH_CALUDE_rect_to_spherical_conversion_l3154_315422


namespace NUMINAMATH_CALUDE_cab_journey_time_l3154_315469

/-- Given a cab walking at 5/6 of its usual speed and arriving 15 minutes late,
    prove that its usual time to cover the journey is 1.25 hours. -/
theorem cab_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (usual_speed * usual_time = (5/6 * usual_speed) * (usual_time + 1/4)) → 
  usual_time = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_cab_journey_time_l3154_315469


namespace NUMINAMATH_CALUDE_grant_baseball_gear_sale_l3154_315447

/-- The total money Grant made from selling his baseball gear -/
def total_money (card_price bat_price glove_original_price glove_discount cleats_price cleats_count : ℝ) : ℝ :=
  card_price + bat_price + (glove_original_price * (1 - glove_discount)) + (cleats_price * cleats_count)

/-- Theorem stating that Grant made $79 from selling his baseball gear -/
theorem grant_baseball_gear_sale :
  total_money 25 10 30 0.2 10 2 = 79 := by
  sorry

end NUMINAMATH_CALUDE_grant_baseball_gear_sale_l3154_315447


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3154_315473

theorem negation_of_proposition :
  (¬ (∀ x y : ℝ, x^2 + y^2 ≥ 0)) ↔ (∃ x y : ℝ, x^2 + y^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3154_315473


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_and_difference_l3154_315440

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_first_term_and_difference
  (a : ℕ → ℚ) (d : ℚ)
  (h_arith : ArithmeticSequence a d)
  (h_fifth : a 5 = 10)
  (h_sum : a 1 + a 2 + a 3 = 3) :
  a 1 = -2 ∧ d = 3 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_and_difference_l3154_315440


namespace NUMINAMATH_CALUDE_cuboid_volume_l3154_315486

/-- The volume of a cuboid with dimensions 4 cm, 6 cm, and 15 cm is 360 cubic centimeters. -/
theorem cuboid_volume : 
  let length : ℝ := 4
  let width : ℝ := 6
  let height : ℝ := 15
  length * width * height = 360 := by sorry

end NUMINAMATH_CALUDE_cuboid_volume_l3154_315486


namespace NUMINAMATH_CALUDE_sum_odd_500_to_800_l3154_315465

def first_odd_after (n : ℕ) : ℕ :=
  if n % 2 = 0 then n + 1 else n + 2

def last_odd_before (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 1 else n - 2

def sum_odd_between (a b : ℕ) : ℕ :=
  let first := first_odd_after a
  let last := last_odd_before b
  let count := (last - first) / 2 + 1
  count * (first + last) / 2

theorem sum_odd_500_to_800 :
  sum_odd_between 500 800 = 97500 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_500_to_800_l3154_315465


namespace NUMINAMATH_CALUDE_initial_cookies_l3154_315472

/-- The number of basketball team members -/
def team_members : ℕ := 8

/-- The number of cookies Andy ate -/
def andy_ate : ℕ := 3

/-- The number of cookies Andy gave to his brother -/
def brother_got : ℕ := 5

/-- The number of cookies the first player took -/
def first_player_cookies : ℕ := 1

/-- The increase in cookies taken by each subsequent player -/
def cookie_increase : ℕ := 2

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a₁ aₙ : ℕ) : ℕ :=
  n * (a₁ + aₙ) / 2

/-- The total number of cookies taken by the basketball team -/
def team_cookies : ℕ :=
  arithmetic_sum team_members first_player_cookies (first_player_cookies + cookie_increase * (team_members - 1))

/-- The theorem stating the initial number of cookies -/
theorem initial_cookies : 
  andy_ate + brother_got + team_cookies = 72 := by sorry

end NUMINAMATH_CALUDE_initial_cookies_l3154_315472


namespace NUMINAMATH_CALUDE_line_b_production_l3154_315452

/-- Represents the production of a factory with three production lines -/
structure FactoryProduction where
  total : ℕ
  lineA : ℕ
  lineB : ℕ
  lineC : ℕ

/-- 
Given a factory production with three lines where:
1. The total production is 24,000 units
2. The number of units sampled from each line forms an arithmetic sequence
3. The sum of production from all lines equals the total production

Then the production of line B is 8,000 units
-/
theorem line_b_production (prod : FactoryProduction) 
  (h_total : prod.total = 24000)
  (h_arithmetic : prod.lineB * 2 = prod.lineA + prod.lineC)
  (h_sum : prod.lineA + prod.lineB + prod.lineC = prod.total) :
  prod.lineB = 8000 := by
  sorry

end NUMINAMATH_CALUDE_line_b_production_l3154_315452


namespace NUMINAMATH_CALUDE_line_through_points_l3154_315407

/-- Given a line with slope 3 passing through points (3, 4) and (x, 7), prove that x = 4 -/
theorem line_through_points (x : ℝ) : 
  (7 - 4) / (x - 3) = 3 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l3154_315407


namespace NUMINAMATH_CALUDE_triangle_ratio_l3154_315403

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  A = 2 * π / 3 →  -- 120° in radians
  b = 1 →
  (1 / 2) * c * b * Real.sin A = Real.sqrt 3 →
  (b + c) / (Real.sin B + Real.sin C) = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3154_315403


namespace NUMINAMATH_CALUDE_newton_albert_game_l3154_315445

theorem newton_albert_game (a n : ℂ) : 
  a * n = 40 - 24 * I ∧ a = 8 - 4 * I → n = 2.8 - 0.4 * I :=
by sorry

end NUMINAMATH_CALUDE_newton_albert_game_l3154_315445


namespace NUMINAMATH_CALUDE_games_lost_l3154_315418

theorem games_lost (total_games won_games : ℕ) 
  (h1 : total_games = 18) 
  (h2 : won_games = 15) : 
  total_games - won_games = 3 := by
sorry

end NUMINAMATH_CALUDE_games_lost_l3154_315418


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l3154_315416

def double_factorial (n : ℕ) : ℕ :=
  if n ≤ 1 then 1 else n * double_factorial (n - 2)

def braced_notation (x : ℕ) : ℕ := double_factorial x

theorem greatest_prime_factor_of_sum (n : ℕ) (h : n = 22) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (braced_notation n + braced_notation (n - 2)) ∧
    ∀ q : ℕ, Nat.Prime q → q ∣ (braced_notation n + braced_notation (n - 2)) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_l3154_315416


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3154_315428

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {1, 4}

theorem intersection_complement_theorem : A ∩ (U \ B) = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3154_315428


namespace NUMINAMATH_CALUDE_pear_distribution_count_l3154_315463

def family_size : ℕ := 7
def elder_count : ℕ := 4

theorem pear_distribution_count : 
  (elder_count : ℕ) * (Nat.factorial (family_size - 2)) = 480 :=
sorry

end NUMINAMATH_CALUDE_pear_distribution_count_l3154_315463


namespace NUMINAMATH_CALUDE_deck_size_l3154_315437

theorem deck_size (r b : ℕ) : 
  r > 0 → b > 0 →
  r / (r + b) = 1 / 4 →
  r / (r + b + 6) = 1 / 6 →
  r + b = 12 := by
sorry

end NUMINAMATH_CALUDE_deck_size_l3154_315437


namespace NUMINAMATH_CALUDE_gp_common_ratio_l3154_315412

theorem gp_common_ratio (a : ℝ) (r : ℝ) (h : r ≠ 1) :
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 28 →
  r = 3 := by
sorry

end NUMINAMATH_CALUDE_gp_common_ratio_l3154_315412


namespace NUMINAMATH_CALUDE_symmetry_implies_m_equals_one_l3154_315484

/-- Two points are symmetric about the origin if their coordinates are negations of each other -/
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

/-- The theorem stating that if P(2, -1) and Q(-2, m) are symmetric about the origin, then m = 1 -/
theorem symmetry_implies_m_equals_one :
  ∀ m : ℝ, symmetric_about_origin (2, -1) (-2, m) → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_m_equals_one_l3154_315484


namespace NUMINAMATH_CALUDE_not_p_false_sufficient_not_necessary_for_p_or_q_true_l3154_315466

theorem not_p_false_sufficient_not_necessary_for_p_or_q_true (p q : Prop) :
  (¬¬p → p ∨ q) ∧ ∃ (p q : Prop), (p ∨ q) ∧ ¬(¬¬p) :=
sorry

end NUMINAMATH_CALUDE_not_p_false_sufficient_not_necessary_for_p_or_q_true_l3154_315466


namespace NUMINAMATH_CALUDE_min_value_theorem_l3154_315431

theorem min_value_theorem (x : ℝ) (h : x > 2) :
  (x + 4) / Real.sqrt (x - 2) ≥ 2 * Real.sqrt 6 ∧
  ∃ y : ℝ, y > 2 ∧ (y + 4) / Real.sqrt (y - 2) = 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3154_315431


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3154_315426

theorem solution_set_inequality (a b : ℝ) :
  ({x : ℝ | a * x^2 - 5 * x + b > 0} = {x : ℝ | -3 < x ∧ x < 2}) →
  ({x : ℝ | b * x^2 - 5 * x + a > 0} = {x : ℝ | x < -3 ∨ x > 2}) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3154_315426


namespace NUMINAMATH_CALUDE_sum_consecutive_products_l3154_315460

/-- The sum of products of three consecutive integers from 19 to 2001 -/
def S : ℕ → ℕ
  | 0 => 0
  | n + 1 => (18 + n) * (19 + n) * (20 + n) + S n

/-- The main theorem stating the closed form of the sum -/
theorem sum_consecutive_products (n : ℕ) :
  S (1981) = 6 * (Nat.choose 2002 4 - Nat.choose 21 4) :=
by sorry

end NUMINAMATH_CALUDE_sum_consecutive_products_l3154_315460


namespace NUMINAMATH_CALUDE_library_book_count_l3154_315479

/-- The number of books in the library after taking out and bringing back some books -/
def final_book_count (initial : ℕ) (taken_out : ℕ) (brought_back : ℕ) : ℕ :=
  initial - taken_out + brought_back

/-- Theorem: Given 336 initial books, 124 taken out, and 22 brought back, there are 234 books now -/
theorem library_book_count : final_book_count 336 124 22 = 234 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l3154_315479


namespace NUMINAMATH_CALUDE_total_children_l3154_315454

/-- Given a group of children where:
    k children are initially selected and given an apple,
    m children are selected later,
    n of the m children had previously received an apple,
    prove that the total number of children is k * (m/n) -/
theorem total_children (k m n : ℕ) (h : n ≤ m) (h' : n > 0) :
  ∃ (total : ℚ), total = k * (m / n) := by
  sorry

end NUMINAMATH_CALUDE_total_children_l3154_315454


namespace NUMINAMATH_CALUDE_solution_set_f_gt_7_minus_x_range_of_m_for_solution_existence_l3154_315427

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| * |x - 3|

-- Part 1: Solution set of f(x) > 7-x
theorem solution_set_f_gt_7_minus_x :
  {x : ℝ | f x > 7 - x} = {x : ℝ | x < -6 ∨ x > 2} := by sorry

-- Part 2: Range of m for which f(x) ≤ |3m-2| has a solution
theorem range_of_m_for_solution_existence :
  {m : ℝ | ∃ x, f x ≤ |3*m - 2|} = {m : ℝ | m ≤ -1 ∨ m ≥ 7/3} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_7_minus_x_range_of_m_for_solution_existence_l3154_315427


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l3154_315498

theorem ticket_price_possibilities : ∃ (divisors : Finset ℕ), 
  (∀ x ∈ divisors, x ∣ 60 ∧ x ∣ 90) ∧ 
  (∀ x : ℕ, x ∣ 60 ∧ x ∣ 90 → x ∈ divisors) ∧
  Finset.card divisors = 8 :=
sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l3154_315498


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l3154_315451

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^4 + 4*x^2 + 16 : ℝ) = (x^2 + 4) * q x := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l3154_315451


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l3154_315441

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l3154_315441


namespace NUMINAMATH_CALUDE_y_coordinate_range_l3154_315400

-- Define the circle C
def CircleC (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the condition MA^2 + MO^2 ≤ 10
def Condition (x y : ℝ) : Prop := (x - 2)^2 + y^2 + x^2 + y^2 ≤ 10

-- Theorem statement
theorem y_coordinate_range :
  ∀ x y : ℝ, CircleC x y → Condition x y →
  -Real.sqrt 7 / 2 ≤ y ∧ y ≤ Real.sqrt 7 / 2 :=
by sorry

end NUMINAMATH_CALUDE_y_coordinate_range_l3154_315400


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3154_315432

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 - a*b + b^2 = 0) : 
  (a^7 + b^7) / (a - b)^7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3154_315432


namespace NUMINAMATH_CALUDE_fraction_inequality_l3154_315424

theorem fraction_inequality (a b c : ℝ) (h : a > b) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3154_315424


namespace NUMINAMATH_CALUDE_y_relationship_l3154_315409

/-- A quadratic function of the form y = -x² + 2x + c -/
def quadratic (c : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + c

/-- Theorem stating the relationship between y-coordinates of three points on the quadratic function -/
theorem y_relationship (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = quadratic c (-1))
  (h₂ : y₂ = quadratic c 2)
  (h₃ : y₃ = quadratic c 5) :
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry


end NUMINAMATH_CALUDE_y_relationship_l3154_315409


namespace NUMINAMATH_CALUDE_samantha_routes_count_l3154_315449

/-- Represents a location on a grid -/
structure Location :=
  (x : Int) (y : Int)

/-- Represents Central Park -/
structure CentralPark :=
  (sw : Location) (ne : Location)

/-- Calculates the number of shortest paths between two locations on a grid -/
def gridPaths (start finish : Location) : Nat :=
  let dx := (finish.x - start.x).natAbs
  let dy := (finish.y - start.y).natAbs
  Nat.choose (dx + dy) dx

/-- The number of diagonal paths through Central Park -/
def parkPaths : Nat := 2

/-- Theorem stating the number of shortest routes from Samantha's house to her school -/
theorem samantha_routes_count (park : CentralPark) 
  (home : Location) 
  (school : Location) 
  (home_to_sw : home.x = park.sw.x - 3 ∧ home.y = park.sw.y - 2)
  (school_to_ne : school.x = park.ne.x + 3 ∧ school.y = park.ne.y + 3) :
  gridPaths home park.sw * parkPaths * gridPaths park.ne school = 400 := by
  sorry

end NUMINAMATH_CALUDE_samantha_routes_count_l3154_315449


namespace NUMINAMATH_CALUDE_soda_ratio_l3154_315446

/-- Proves that the ratio of regular sodas to diet sodas is 9:7 -/
theorem soda_ratio (total_sodas : ℕ) (diet_sodas : ℕ) : 
  total_sodas = 64 → diet_sodas = 28 → 
  (total_sodas - diet_sodas : ℚ) / diet_sodas = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_soda_ratio_l3154_315446


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l3154_315423

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The theorem stating that the reflection of (1, 6) across the y-axis is (-1, 6) -/
theorem reflection_across_y_axis :
  let original := Point.mk 1 6
  reflect_y original = Point.mk (-1) 6 := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l3154_315423


namespace NUMINAMATH_CALUDE_f_plus_g_is_non_horizontal_line_l3154_315467

/-- Represents a parabola in vertex form -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ
  a_nonzero : a ≠ 0

/-- The function resulting from translating the original parabola 7 units right -/
def f (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h + 7)^2 + p.k

/-- The function resulting from reflecting the parabola and translating 7 units left -/
def g (p : Parabola) (x : ℝ) : ℝ :=
  -p.a * (x - p.h - 7)^2 - p.k

/-- The sum of f and g -/
def f_plus_g (p : Parabola) (x : ℝ) : ℝ :=
  f p x + g p x

/-- Theorem stating that f_plus_g is a non-horizontal line -/
theorem f_plus_g_is_non_horizontal_line (p : Parabola) :
  ∃ m b, m ≠ 0 ∧ ∀ x, f_plus_g p x = m * x + b := by
  sorry

end NUMINAMATH_CALUDE_f_plus_g_is_non_horizontal_line_l3154_315467


namespace NUMINAMATH_CALUDE_greatest_value_2q_minus_r_l3154_315434

theorem greatest_value_2q_minus_r : 
  ∃ (q r : ℕ+), 
    1027 = 21 * q + r ∧ 
    ∀ (q' r' : ℕ+), 1027 = 21 * q' + r' → 2 * q - r ≥ 2 * q' - r' ∧
    2 * q - r = 77 := by
  sorry

end NUMINAMATH_CALUDE_greatest_value_2q_minus_r_l3154_315434


namespace NUMINAMATH_CALUDE_problem_shape_surface_area_l3154_315453

/-- Represents a solid shape made of unit cubes -/
structure CubeShape where
  base_length : ℕ
  base_width : ℕ
  top_length : ℕ
  top_width : ℕ
  total_cubes : ℕ

/-- Calculates the surface area of the CubeShape -/
def surface_area (shape : CubeShape) : ℕ :=
  sorry

/-- The specific cube shape described in the problem -/
def problem_shape : CubeShape :=
  { base_length := 4
  , base_width := 3
  , top_length := 3
  , top_width := 1
  , total_cubes := 15
  }

/-- Theorem stating that the surface area of the problem_shape is 36 square units -/
theorem problem_shape_surface_area :
  surface_area problem_shape = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_shape_surface_area_l3154_315453


namespace NUMINAMATH_CALUDE_fiona_id_is_17_l3154_315435

/-- A structure representing a math club member with an ID number -/
structure MathClubMember where
  name : String
  id : Nat

/-- A predicate to check if a number is prime -/
def isPrime (n : Nat) : Prop := sorry

/-- A predicate to check if a number is a two-digit number -/
def isTwoDigit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

theorem fiona_id_is_17 
  (dan emily fiona : MathClubMember)
  (h1 : isPrime dan.id ∧ isPrime emily.id ∧ isPrime fiona.id)
  (h2 : isTwoDigit dan.id ∧ isTwoDigit emily.id ∧ isTwoDigit fiona.id)
  (h3 : ∃ p q : Nat, dan.id < p ∧ p < q ∧ 
    (emily.id = p ∨ emily.id = q) ∧ 
    (fiona.id = p ∨ fiona.id = q) ∧
    isPrime p ∧ isPrime q)
  (h4 : ∃ today : Nat, emily.id + fiona.id = today ∧ today ≤ 31)
  (h5 : ∃ emilys_birthday : Nat, dan.id + fiona.id = emilys_birthday - 1 ∧ emilys_birthday ≤ 31)
  (h6 : dan.id + emily.id = (emily.id + fiona.id) + 1)
  : fiona.id = 17 := by
  sorry

end NUMINAMATH_CALUDE_fiona_id_is_17_l3154_315435


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3154_315491

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- Theorem statement
theorem diamond_equation_solution :
  ∀ x : ℝ, diamond 5 x = 12 → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l3154_315491


namespace NUMINAMATH_CALUDE_scooter_cost_calculation_l3154_315415

theorem scooter_cost_calculation (original_cost : ℝ) 
  (repair1_percent repair2_percent repair3_percent tax_percent : ℝ)
  (discount_percent profit_percent : ℝ) (profit : ℝ) :
  repair1_percent = 0.05 →
  repair2_percent = 0.10 →
  repair3_percent = 0.07 →
  tax_percent = 0.12 →
  discount_percent = 0.15 →
  profit_percent = 0.30 →
  profit = 2000 →
  profit = profit_percent * original_cost →
  let total_spent := original_cost * (1 + repair1_percent + repair2_percent + repair3_percent + tax_percent)
  total_spent = 1.34 * original_cost :=
by sorry

end NUMINAMATH_CALUDE_scooter_cost_calculation_l3154_315415


namespace NUMINAMATH_CALUDE_sum_of_ages_l3154_315410

-- Define the ages of George, Christopher, and Ford
def christopher_age : ℕ := 18
def george_age : ℕ := christopher_age + 8
def ford_age : ℕ := christopher_age - 2

-- Theorem to prove
theorem sum_of_ages : george_age + christopher_age + ford_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3154_315410


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l3154_315462

theorem polynomial_identity_sum (d₁ d₂ d₃ e₁ e₂ e₃ : ℝ) : 
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + d₁*x + e₁)*(x^2 + d₂*x + e₂)*(x^2 + d₃*x + e₃)*(x^2 + 1)) →
  d₁*e₁ + d₂*e₂ + d₃*e₃ = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l3154_315462


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l3154_315439

theorem smallest_n_multiple_of_seven (x y : ℤ) 
  (hx : (x + 2) % 7 = 0) 
  (hy : (y - 2) % 7 = 0) : 
  ∃ n : ℕ+, (∀ m : ℕ+, (x^2 + x*y + y^2 + m) % 7 = 0 → n ≤ m) ∧ 
            (x^2 + x*y + y^2 + n) % 7 = 0 ∧ 
            n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l3154_315439


namespace NUMINAMATH_CALUDE_last_digit_to_appear_is_six_l3154_315494

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppearedBy (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ unitsDigit (fib k) = d

-- Main theorem
theorem last_digit_to_appear_is_six :
  ∃ n : ℕ, (∀ d : ℕ, d < 10 → digitAppearedBy d n) ∧
  (∀ m : ℕ, m < n → ¬(∀ d : ℕ, d < 10 → digitAppearedBy d m)) ∧
  unitsDigit (fib n) = 6 :=
sorry

end NUMINAMATH_CALUDE_last_digit_to_appear_is_six_l3154_315494


namespace NUMINAMATH_CALUDE_wanda_walking_distance_l3154_315496

/-- The distance Wanda walks to school (in miles) -/
def distance_to_school : ℝ := 0.5

/-- The number of times Wanda walks to and from school per day -/
def trips_per_day : ℕ := 2

/-- The number of school days per week -/
def school_days_per_week : ℕ := 5

/-- The number of weeks -/
def num_weeks : ℕ := 4

/-- The total distance Wanda walks in miles after the given number of weeks -/
def total_distance : ℝ := 
  distance_to_school * 2 * trips_per_day * school_days_per_week * num_weeks

theorem wanda_walking_distance : total_distance = 40 := by
  sorry

end NUMINAMATH_CALUDE_wanda_walking_distance_l3154_315496


namespace NUMINAMATH_CALUDE_ending_number_of_range_l3154_315464

theorem ending_number_of_range : ∃ n : ℕ, 
  (n ≥ 100) ∧ 
  ((200 + 400) / 2 = ((100 + n) / 2) + 150) ∧ 
  (n = 200) := by
  sorry

end NUMINAMATH_CALUDE_ending_number_of_range_l3154_315464


namespace NUMINAMATH_CALUDE_books_loaned_out_l3154_315448

theorem books_loaned_out (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : return_rate = 65 / 100)
  (h3 : final_books = 68) : 
  ∃ (loaned_books : ℕ), loaned_books = 20 ∧ 
    final_books = initial_books - (1 - return_rate) * loaned_books :=
by sorry

end NUMINAMATH_CALUDE_books_loaned_out_l3154_315448


namespace NUMINAMATH_CALUDE_species_x_count_l3154_315483

def ant_farm (x y : ℕ) : Prop :=
  -- Initial total number of ants
  x + y = 50 ∧
  -- Total number of ants on Day 4
  81 * x + 16 * y = 2914

theorem species_x_count : ∃ x y : ℕ, ant_farm x y ∧ 81 * x = 2754 := by
  sorry

end NUMINAMATH_CALUDE_species_x_count_l3154_315483


namespace NUMINAMATH_CALUDE_remainder_equality_l3154_315404

theorem remainder_equality (P P' Q D R R' s s' : ℕ) : 
  P > P' → 
  Q > 0 → 
  P < D → P' < D → Q < D →
  R = P % D →
  R' = P' % D →
  s = (P + P') % D →
  s' = (R + R') % D →
  s = s' :=
by sorry

end NUMINAMATH_CALUDE_remainder_equality_l3154_315404


namespace NUMINAMATH_CALUDE_juice_distribution_l3154_315420

theorem juice_distribution (container_capacity : ℝ) : 
  container_capacity > 0 → 
  let total_juice := (3 / 4) * container_capacity
  let num_cups := 5
  let juice_per_cup := total_juice / num_cups
  (juice_per_cup / container_capacity) * 100 = 15 := by sorry

end NUMINAMATH_CALUDE_juice_distribution_l3154_315420
