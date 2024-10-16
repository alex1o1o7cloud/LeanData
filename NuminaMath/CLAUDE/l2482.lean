import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2482_248217

theorem equation_solution : ∃! x : ℝ, (9 - x)^2 = x^2 ∧ x = (9 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2482_248217


namespace NUMINAMATH_CALUDE_unique_age_pair_l2482_248283

/-- The set of possible ages for X's sons -/
def AgeSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

/-- Predicate for pairs of ages that satisfy the product condition -/
def ProductCondition (a b : ℕ) : Prop :=
  a ∈ AgeSet ∧ b ∈ AgeSet ∧ ∃ (c d : ℕ), c ≠ a ∧ d ≠ b ∧ c * d = a * b ∧ c ∈ AgeSet ∧ d ∈ AgeSet

/-- Predicate for pairs of ages that satisfy the ratio condition -/
def RatioCondition (a b : ℕ) : Prop :=
  a ∈ AgeSet ∧ b ∈ AgeSet ∧ ∃ (c d : ℕ), c ≠ a ∧ d ≠ b ∧ c * b = a * d ∧ c ∈ AgeSet ∧ d ∈ AgeSet

/-- Predicate for pairs of ages that satisfy the difference condition -/
def DifferenceCondition (a b : ℕ) : Prop :=
  a ∈ AgeSet ∧ b ∈ AgeSet ∧ ∀ (c d : ℕ), c ∈ AgeSet → d ∈ AgeSet → c - d = a - b → (c = a ∧ d = b) ∨ (c = b ∧ d = a)

/-- Theorem stating that (8, 2) is the only pair satisfying all conditions -/
theorem unique_age_pair : ∀ (a b : ℕ), 
  ProductCondition a b ∧ RatioCondition a b ∧ DifferenceCondition a b ↔ (a = 8 ∧ b = 2) ∨ (a = 2 ∧ b = 8) :=
sorry

end NUMINAMATH_CALUDE_unique_age_pair_l2482_248283


namespace NUMINAMATH_CALUDE_divisible_by_six_count_and_percentage_l2482_248200

theorem divisible_by_six_count_and_percentage :
  let n : ℕ := 120
  let divisible_count : ℕ := (n / 6 : ℕ)
  divisible_count = 20 ∧ 
  (divisible_count : ℚ) / n * 100 = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_count_and_percentage_l2482_248200


namespace NUMINAMATH_CALUDE_uphill_distance_l2482_248207

/-- Proves that given specific conditions, the uphill distance traveled by a car is 100 km. -/
theorem uphill_distance (uphill_speed downhill_speed downhill_distance average_speed : ℝ) 
  (h1 : uphill_speed = 30)
  (h2 : downhill_speed = 60)
  (h3 : downhill_distance = 50)
  (h4 : average_speed = 36) : 
  ∃ uphill_distance : ℝ, 
    uphill_distance = 100 ∧ 
    average_speed = (uphill_distance + downhill_distance) / (uphill_distance / uphill_speed + downhill_distance / downhill_speed) := by
  sorry

end NUMINAMATH_CALUDE_uphill_distance_l2482_248207


namespace NUMINAMATH_CALUDE_length_of_AD_rhombus_condition_l2482_248251

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℝ := x^2 - (a - 4) * x + a - 1

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (AB AD : ℝ)
  (a : ℝ)
  (eq_AB : quadratic_equation a AB = 0)
  (eq_AD : quadratic_equation a AD = 0)

-- Theorem 1: Length of AD
theorem length_of_AD (ABCD : Quadrilateral) (h : ABCD.AB = 2) : ABCD.AD = 5 := by
  sorry

-- Theorem 2: Condition for rhombus
theorem rhombus_condition (ABCD : Quadrilateral) : ABCD.AB = ABCD.AD ↔ ABCD.a = 10 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AD_rhombus_condition_l2482_248251


namespace NUMINAMATH_CALUDE_couscous_dishes_proof_l2482_248206

/-- Calculates the number of dishes that can be made from couscous shipments -/
def couscous_dishes (shipment1 shipment2 shipment3 pounds_per_dish : ℕ) : ℕ :=
  (shipment1 + shipment2 + shipment3) / pounds_per_dish

/-- Proves that given the specified shipments and dish requirement, 13 dishes can be made -/
theorem couscous_dishes_proof :
  couscous_dishes 7 13 45 5 = 13 := by
  sorry

end NUMINAMATH_CALUDE_couscous_dishes_proof_l2482_248206


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_f_triangle_area_l2482_248224

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sqrt 3 * Real.sin x * Real.cos x + 1/2

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem smallest_positive_period_of_f :
  ∃ T > 0, is_periodic f T ∧ ∀ S, 0 < S ∧ S < T → ¬ is_periodic f S :=
sorry

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  f (B + C) = 3/2 →
  a = Real.sqrt 3 →
  b + c = 3 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_f_triangle_area_l2482_248224


namespace NUMINAMATH_CALUDE_egg_roll_ratio_l2482_248239

-- Define the number of egg rolls each person ate
def matthew_egg_rolls : ℕ := 6
def alvin_egg_rolls : ℕ := 4

-- Define Patrick's egg rolls based on the condition
def patrick_egg_rolls : ℕ := matthew_egg_rolls / 3

-- Theorem to prove the ratio
theorem egg_roll_ratio :
  patrick_egg_rolls / alvin_egg_rolls = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_egg_roll_ratio_l2482_248239


namespace NUMINAMATH_CALUDE_blue_green_difference_l2482_248249

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexagonalFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Adds a border of tiles to a hexagonal figure -/
def add_border (figure : HexagonalFigure) : HexagonalFigure :=
  { blue_tiles := figure.blue_tiles + 18,
    green_tiles := figure.green_tiles + 18 }

/-- The initial hexagonal figure -/
def initial_figure : HexagonalFigure :=
  { blue_tiles := 15, green_tiles := 9 }

/-- The new figure after adding both borders -/
def new_figure : HexagonalFigure :=
  add_border (add_border initial_figure)

theorem blue_green_difference :
  new_figure.blue_tiles - new_figure.green_tiles = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_green_difference_l2482_248249


namespace NUMINAMATH_CALUDE_relationship_between_a_and_b_l2482_248222

theorem relationship_between_a_and_b (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (eq1 : a^5 = a + 1) (eq2 : b^10 = b + 3*a) : 
  a > b ∧ b > 1 := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_a_and_b_l2482_248222


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l2482_248276

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_and_line_in_plane
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : contains α n) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l2482_248276


namespace NUMINAMATH_CALUDE_partition_sum_exists_l2482_248298

theorem partition_sum_exists : ∃ (A B : Finset ℕ),
  A ∪ B = Finset.range 14 \ {0} ∧
  A ∩ B = ∅ ∧
  A.card = 5 ∧
  B.card = 8 ∧
  3 * (A.sum id) + 7 * (B.sum id) = 433 := by
  sorry

end NUMINAMATH_CALUDE_partition_sum_exists_l2482_248298


namespace NUMINAMATH_CALUDE_sqrt_sum_rationalization_l2482_248299

theorem sqrt_sum_rationalization : ∃ (a b c : ℕ+), 
  (Real.sqrt 8 + (1 / Real.sqrt 8) + Real.sqrt 9 + (1 / Real.sqrt 9) = (a * Real.sqrt 8 + b * Real.sqrt 9) / c) ∧
  (∀ (a' b' c' : ℕ+), 
    (Real.sqrt 8 + (1 / Real.sqrt 8) + Real.sqrt 9 + (1 / Real.sqrt 9) = (a' * Real.sqrt 8 + b' * Real.sqrt 9) / c') →
    c ≤ c') ∧
  (a + b + c = 31) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_rationalization_l2482_248299


namespace NUMINAMATH_CALUDE_floor_inequality_l2482_248294

theorem floor_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ⌊5 * x⌋ + ⌊5 * y⌋ ≥ ⌊3 * x + y⌋ + ⌊3 * y + x⌋ := by
  sorry

#check floor_inequality

end NUMINAMATH_CALUDE_floor_inequality_l2482_248294


namespace NUMINAMATH_CALUDE_min_value_problem_l2482_248288

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2 * a + 3 * b = 1) :
  (2 / a) + (3 / b) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2482_248288


namespace NUMINAMATH_CALUDE_sum_100_to_120_l2482_248204

def sum_inclusive_range (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

theorem sum_100_to_120 : sum_inclusive_range 100 120 = 2310 := by sorry

end NUMINAMATH_CALUDE_sum_100_to_120_l2482_248204


namespace NUMINAMATH_CALUDE_nursery_school_fraction_l2482_248277

theorem nursery_school_fraction (total_students : ℕ) 
  (under_three : ℕ) (not_between_three_and_four : ℕ) :
  total_students = 50 →
  under_three = 20 →
  not_between_three_and_four = 25 →
  (total_students - not_between_three_and_four : ℚ) / total_students = 9 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_nursery_school_fraction_l2482_248277


namespace NUMINAMATH_CALUDE_sqrt_450_simplified_l2482_248234

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplified_l2482_248234


namespace NUMINAMATH_CALUDE_line_equation_through_points_l2482_248271

theorem line_equation_through_points (x y : ℝ) : 
  (2 * x - y - 2 = 0) ↔ 
  (∃ t : ℝ, x = 1 - t ∧ y = -2 * t) :=
sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l2482_248271


namespace NUMINAMATH_CALUDE_inequality_proof_l2482_248240

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * c ≥ (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ∧
  (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ≥ (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2482_248240


namespace NUMINAMATH_CALUDE_correct_average_l2482_248297

theorem correct_average (n : ℕ) (initial_avg : ℚ) (correction1 correction2 : ℚ) :
  n = 10 ∧ 
  initial_avg = 40.2 ∧ 
  correction1 = -19 ∧ 
  correction2 = 18 →
  (n * initial_avg + correction1 + correction2) / n = 40.1 := by
sorry

end NUMINAMATH_CALUDE_correct_average_l2482_248297


namespace NUMINAMATH_CALUDE_equation_solutions_l2482_248275

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 16 = 0 ↔ x = 4 ∨ x = -4) ∧
  (∀ x : ℝ, (x + 10)^3 + 27 = 0 ↔ x = -13) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2482_248275


namespace NUMINAMATH_CALUDE_complex_argument_of_two_plus_two_i_sqrt_three_l2482_248291

/-- For the complex number z = 2 + 2i√3, when expressed in the form re^(iθ), θ = π/3 -/
theorem complex_argument_of_two_plus_two_i_sqrt_three :
  let z : ℂ := 2 + 2 * Complex.I * Real.sqrt 3
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_complex_argument_of_two_plus_two_i_sqrt_three_l2482_248291


namespace NUMINAMATH_CALUDE_circle_probabilities_l2482_248264

/-- A type representing the 10 equally spaced points on a circle -/
inductive CirclePoint
  | one | two | three | four | five | six | seven | eight | nine | ten

/-- Function to check if two points form a diameter -/
def is_diameter (p1 p2 : CirclePoint) : Prop := sorry

/-- Function to check if three points form a right triangle -/
def is_right_triangle (p1 p2 p3 : CirclePoint) : Prop := sorry

/-- Function to check if four points form a rectangle -/
def is_rectangle (p1 p2 p3 p4 : CirclePoint) : Prop := sorry

/-- The number of ways to choose n items from a set of 10 -/
def choose_10 (n : Nat) : Nat := sorry

theorem circle_probabilities :
  (∃ (num_diameters : Nat), 
    num_diameters / choose_10 2 = 1 / 9 ∧
    (∀ p1 p2 : CirclePoint, p1 ≠ p2 → 
      (Nat.card {pair | pair = (p1, p2) ∧ is_diameter p1 p2} = num_diameters))) ∧
  (∃ (num_right_triangles : Nat),
    num_right_triangles / choose_10 3 = 1 / 3 ∧
    (∀ p1 p2 p3 : CirclePoint, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
      (Nat.card {triple | triple = (p1, p2, p3) ∧ is_right_triangle p1 p2 p3} = num_right_triangles))) ∧
  (∃ (num_rectangles : Nat),
    num_rectangles / choose_10 4 = 1 / 21 ∧
    (∀ p1 p2 p3 p4 : CirclePoint, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p4 →
      (Nat.card {quad | quad = (p1, p2, p3, p4) ∧ is_rectangle p1 p2 p3 p4} = num_rectangles))) :=
by sorry

end NUMINAMATH_CALUDE_circle_probabilities_l2482_248264


namespace NUMINAMATH_CALUDE_multiply_72519_9999_l2482_248238

theorem multiply_72519_9999 : 72519 * 9999 = 724817481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72519_9999_l2482_248238


namespace NUMINAMATH_CALUDE_sqrt_difference_squared_l2482_248244

theorem sqrt_difference_squared : (Real.sqrt 169 - Real.sqrt 25)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_squared_l2482_248244


namespace NUMINAMATH_CALUDE_students_without_scholarships_l2482_248282

def total_students : ℕ := 300

def full_merit_percent : ℚ := 5 / 100
def half_merit_percent : ℚ := 10 / 100
def sports_percent : ℚ := 3 / 100
def need_based_percent : ℚ := 7 / 100

def full_merit_and_sports_percent : ℚ := 1 / 100
def half_merit_and_need_based_percent : ℚ := 2 / 100
def sports_and_need_based_percent : ℚ := 1 / 200

theorem students_without_scholarships :
  (total_students : ℚ) - 
  (((full_merit_percent + half_merit_percent + sports_percent + need_based_percent) * total_students) -
   ((full_merit_and_sports_percent + half_merit_and_need_based_percent + sports_and_need_based_percent) * total_students)) = 236 := by
  sorry

end NUMINAMATH_CALUDE_students_without_scholarships_l2482_248282


namespace NUMINAMATH_CALUDE_cos_negative_23pi_over_4_l2482_248203

theorem cos_negative_23pi_over_4 : Real.cos (-23 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_23pi_over_4_l2482_248203


namespace NUMINAMATH_CALUDE_symbol_equation_solution_l2482_248261

theorem symbol_equation_solution :
  ∀ (star square circle : ℕ),
    star + square = 24 →
    square + circle = 30 →
    circle + star = 36 →
    square = 9 ∧ circle = 21 ∧ star = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_symbol_equation_solution_l2482_248261


namespace NUMINAMATH_CALUDE_complement_union_A_B_l2482_248257

open Set Real

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 ≥ 2}

-- State the theorem
theorem complement_union_A_B : 
  (Aᶜ ∩ Bᶜ : Set ℝ) = Icc (-sqrt 2) 1 := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l2482_248257


namespace NUMINAMATH_CALUDE_age_sum_in_six_years_l2482_248250

/-- Melanie's current age -/
def melanie_age : ℕ := sorry

/-- Phil's current age -/
def phil_age : ℕ := sorry

/-- The sum of Melanie's and Phil's ages 6 years from now is 42, 
    given that in 10 years, the product of their ages will be 400 more than it is now. -/
theorem age_sum_in_six_years : 
  (melanie_age + 10) * (phil_age + 10) = melanie_age * phil_age + 400 →
  (melanie_age + 6) + (phil_age + 6) = 42 := by sorry

end NUMINAMATH_CALUDE_age_sum_in_six_years_l2482_248250


namespace NUMINAMATH_CALUDE_units_digit_of_p_l2482_248273

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_p (p : ℤ) : 
  p > 0 → 
  units_digit p > 0 →
  units_digit (p^3) - units_digit (p^2) = 0 →
  units_digit (p + 4) = 0 →
  units_digit p = 6 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_p_l2482_248273


namespace NUMINAMATH_CALUDE_milk_production_theorem_l2482_248255

/-- Represents the milk production scenario -/
structure MilkProduction where
  initial_cows : ℕ
  initial_days : ℕ
  initial_gallons : ℕ
  max_daily_per_cow : ℕ
  available_cows : ℕ
  target_days : ℕ

/-- Calculates the total milk production given the scenario -/
def total_milk_production (mp : MilkProduction) : ℕ :=
  let daily_rate_per_cow := mp.initial_gallons / (mp.initial_cows * mp.initial_days)
  let actual_rate := min daily_rate_per_cow mp.max_daily_per_cow
  mp.available_cows * actual_rate * mp.target_days

/-- Theorem stating that the total milk production is 96 gallons -/
theorem milk_production_theorem (mp : MilkProduction) 
  (h1 : mp.initial_cows = 10)
  (h2 : mp.initial_days = 5)
  (h3 : mp.initial_gallons = 40)
  (h4 : mp.max_daily_per_cow = 2)
  (h5 : mp.available_cows = 15)
  (h6 : mp.target_days = 8) :
  total_milk_production mp = 96 := by
  sorry

end NUMINAMATH_CALUDE_milk_production_theorem_l2482_248255


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_l2482_248226

open Real

theorem isosceles_right_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  log a - log c = log (sin B) →
  log (sin B) = -log (sqrt 2) →
  B < π / 2 →
  a = b ∧ C = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_l2482_248226


namespace NUMINAMATH_CALUDE_primary_school_ages_l2482_248242

theorem primary_school_ages (x y : ℕ) : 
  7 ≤ x ∧ x ≤ 13 ∧ 7 ≤ y ∧ y ≤ 13 →
  (x + y) * (x - y) = 63 →
  x = 12 ∧ y = 9 := by
sorry

end NUMINAMATH_CALUDE_primary_school_ages_l2482_248242


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l2482_248262

/-- The function f(x) -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 3*(x - 1)

/-- Theorem stating that the derivative of f(x) at x = 1 is 3 -/
theorem derivative_f_at_one : 
  deriv f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l2482_248262


namespace NUMINAMATH_CALUDE_no_triangle_with_special_angles_l2482_248208

theorem no_triangle_with_special_angles : 
  ¬ ∃ (α β γ : Real), 
    α + β + γ = Real.pi ∧ 
    ((3 * Real.cos α - 2) * (14 * Real.sin α ^ 2 + Real.sin (2 * α) - 12) = 0) ∧
    ((3 * Real.cos β - 2) * (14 * Real.sin β ^ 2 + Real.sin (2 * β) - 12) = 0) ∧
    ((3 * Real.cos γ - 2) * (14 * Real.sin γ ^ 2 + Real.sin (2 * γ) - 12) = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_with_special_angles_l2482_248208


namespace NUMINAMATH_CALUDE_factorial_simplification_l2482_248279

theorem factorial_simplification (N : ℕ) :
  (Nat.factorial (N + 2)) / (Nat.factorial N * (N + 3)) = ((N + 2) * (N + 1)) / (N + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l2482_248279


namespace NUMINAMATH_CALUDE_function_value_comparison_l2482_248212

def f (x : ℝ) : ℝ := 3 * (x - 2)^2 + 5

theorem function_value_comparison (x₁ x₂ : ℝ) 
  (h : |x₁ - 2| > |x₂ - 2|) : f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_function_value_comparison_l2482_248212


namespace NUMINAMATH_CALUDE_pythago_competition_l2482_248209

theorem pythago_competition (n : ℕ) : 
  (∀ s : ℕ, s ≤ n → ∃! (team : Fin 4 → ℕ), ∀ i j : Fin 4, i ≠ j → team i ≠ team j) →
  (∃ daniel : ℕ, daniel < 50 ∧ 
    (∃ eliza fiona greg : ℕ, 
      eliza = 50 ∧ fiona = 81 ∧ greg = 97 ∧
      daniel < eliza ∧ daniel < fiona ∧ daniel < greg ∧
      (∀ x : ℕ, x ≤ 4*n → (x ≤ daniel ↔ 2*x ≤ 4*n + 1)))) →
  n = 25 := by sorry

end NUMINAMATH_CALUDE_pythago_competition_l2482_248209


namespace NUMINAMATH_CALUDE_shaded_area_of_square_shaded_percentage_l2482_248267

/-- The shaded area of a square with side length 6 units -/
theorem shaded_area_of_square (side_length : ℝ) (shaded_square : ℝ) (shaded_region : ℝ) (shaded_strip : ℝ) : 
  side_length = 6 →
  shaded_square = 2^2 →
  shaded_region = 5^2 - 3^2 →
  shaded_strip = 6 * 1 →
  shaded_square + shaded_region + shaded_strip = 26 := by
sorry

/-- The percentage of the square that is shaded -/
theorem shaded_percentage (total_area : ℝ) (shaded_area : ℝ) :
  total_area = 6^2 →
  shaded_area = 26 →
  (shaded_area / total_area) * 100 = 72.22 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_shaded_percentage_l2482_248267


namespace NUMINAMATH_CALUDE_committee_formation_theorem_l2482_248295

/-- The number of ways to form a committee with leaders --/
def committee_formation_ways (n m k : ℕ) : ℕ :=
  (Nat.choose n m) * (2^m - 2)

/-- Theorem stating the number of ways to form the committee --/
theorem committee_formation_theorem :
  committee_formation_ways 10 5 4 = 7560 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_theorem_l2482_248295


namespace NUMINAMATH_CALUDE_unique_intersection_l2482_248289

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|4 * x - 3|

-- State the theorem
theorem unique_intersection :
  ∃! p : ℝ × ℝ, f p.1 = p.2 ∧ g p.1 = p.2 :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l2482_248289


namespace NUMINAMATH_CALUDE_polynomial_without_cubic_and_linear_terms_l2482_248284

theorem polynomial_without_cubic_and_linear_terms 
  (a b : ℝ) 
  (h1 : a - 3 = 0)  -- Coefficient of x^3 is zero
  (h2 : 4 - b = 0)  -- Coefficient of x is zero
  : (a - b) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_without_cubic_and_linear_terms_l2482_248284


namespace NUMINAMATH_CALUDE_tshirt_cost_l2482_248260

theorem tshirt_cost (initial_amount : ℕ) (sweater_cost : ℕ) (shoes_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 91 →
  sweater_cost = 24 →
  shoes_cost = 11 →
  remaining_amount = 50 →
  initial_amount - remaining_amount - sweater_cost - shoes_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_cost_l2482_248260


namespace NUMINAMATH_CALUDE_min_value_fraction_l2482_248263

theorem min_value_fraction (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_one : x + y + z + w = 1) :
  (x + y) / (x * y * z * w) ≥ 108 ∧ 
  ∃ x y z w, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧ 
    x + y + z + w = 1 ∧ (x + y) / (x * y * z * w) = 108 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2482_248263


namespace NUMINAMATH_CALUDE_lg_sum_equals_two_l2482_248268

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem statement
theorem lg_sum_equals_two : lg 4 + lg 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_sum_equals_two_l2482_248268


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l2482_248214

theorem ratio_of_sum_to_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l2482_248214


namespace NUMINAMATH_CALUDE_gcd_of_840_and_1764_l2482_248287

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_840_and_1764_l2482_248287


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2482_248278

theorem absolute_value_inequality (x : ℝ) :
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ ((1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2482_248278


namespace NUMINAMATH_CALUDE_lindas_savings_l2482_248259

/-- Given that Linda spent 3/4 of her savings on furniture and the rest on a TV costing $500,
    prove that her original savings were $2000. -/
theorem lindas_savings (savings : ℝ) : 
  (3/4 : ℝ) * savings + 500 = savings → savings = 2000 := by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l2482_248259


namespace NUMINAMATH_CALUDE_coin_problem_l2482_248233

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin type in cents --/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10
  | CoinType.Quarter => 25
  | CoinType.HalfDollar => 50

/-- A collection of coins --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat
  halfDollars : Nat

/-- The total number of coins in a collection --/
def CoinCollection.totalCoins (c : CoinCollection) : Nat :=
  c.pennies + c.nickels + c.dimes + c.quarters + c.halfDollars

/-- The total value of a coin collection in cents --/
def CoinCollection.totalValue (c : CoinCollection) : Nat :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime +
  c.quarters * coinValue CoinType.Quarter +
  c.halfDollars * coinValue CoinType.HalfDollar

/-- The main theorem to prove --/
theorem coin_problem :
  ∀ (c : CoinCollection),
    c.totalCoins = 12 ∧
    c.totalValue = 166 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    c.quarters ≥ 1 ∧
    c.halfDollars ≥ 1
    →
    c.quarters = 3 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l2482_248233


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_absolute_value_function_l2482_248227

def f (x : ℝ) := |x + 2| - 6

theorem minimum_point_of_translated_absolute_value_function :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x ∧ x₀ = -2 ∧ f x₀ = -6 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_absolute_value_function_l2482_248227


namespace NUMINAMATH_CALUDE_fifth_selected_is_one_l2482_248266

def is_valid (n : ℕ) : Bool :=
  0 < n ∧ n ≤ 20

def unique_valid_numbers (seq : List ℕ) : List ℕ :=
  seq.filter is_valid |>.eraseDups

theorem fifth_selected_is_one (seq : List ℕ) 
  (h : seq = [65, 72, 8, 2, 63, 14, 7, 2, 43, 69, 97, 8, 1]) : 
  (unique_valid_numbers seq).nthLe 4 (by sorry) = 1 := by sorry

end NUMINAMATH_CALUDE_fifth_selected_is_one_l2482_248266


namespace NUMINAMATH_CALUDE_magnitude_of_z_l2482_248223

open Complex

theorem magnitude_of_z (z : ℂ) (h : (1 + 2*I) / z = 2 - I) : abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l2482_248223


namespace NUMINAMATH_CALUDE_zealand_has_one_fifth_l2482_248241

/-- Represents the amount of money each person has -/
structure Money where
  wanda : ℚ
  xander : ℚ
  yusuf : ℚ
  zealand : ℚ

/-- The initial state of money distribution -/
def initial_money : Money :=
  { wanda := 6, xander := 5, yusuf := 4, zealand := 0 }

/-- The state of money after Zealand receives money from others -/
def final_money : Money :=
  { wanda := 5, xander := 4, yusuf := 3, zealand := 3 }

/-- The fraction of money Zealand has at the end -/
def zealand_fraction (m : Money) : ℚ :=
  m.zealand / (m.wanda + m.xander + m.yusuf + m.zealand)

/-- Theorem stating that Zealand ends up with 1/5 of the total money -/
theorem zealand_has_one_fifth :
  zealand_fraction final_money = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_zealand_has_one_fifth_l2482_248241


namespace NUMINAMATH_CALUDE_haley_cupcakes_l2482_248218

theorem haley_cupcakes (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (h1 : todd_ate = 11)
  (h2 : packages = 3)
  (h3 : cupcakes_per_package = 3) :
  todd_ate + packages * cupcakes_per_package = 20 := by
  sorry

end NUMINAMATH_CALUDE_haley_cupcakes_l2482_248218


namespace NUMINAMATH_CALUDE_symmetric_intersection_theorem_l2482_248219

/-- A line that intersects a circle at two points symmetric about another line -/
structure SymmetricIntersection where
  /-- The coefficient of x in the line equation ax + 2y - 2 = 0 -/
  a : ℝ
  /-- The first intersection point -/
  A : ℝ × ℝ
  /-- The second intersection point -/
  B : ℝ × ℝ

/-- The line ax + 2y - 2 = 0 intersects the circle (x-1)² + (y+1)² = 6 -/
def intersects_circle (si : SymmetricIntersection) : Prop :=
  let (x₁, y₁) := si.A
  let (x₂, y₂) := si.B
  si.a * x₁ + 2 * y₁ - 2 = 0 ∧
  si.a * x₂ + 2 * y₂ - 2 = 0 ∧
  (x₁ - 1)^2 + (y₁ + 1)^2 = 6 ∧
  (x₂ - 1)^2 + (y₂ + 1)^2 = 6

/-- A and B are symmetric with respect to the line x + y = 0 -/
def symmetric_about_line (si : SymmetricIntersection) : Prop :=
  let (x₁, y₁) := si.A
  let (x₂, y₂) := si.B
  x₁ + y₁ = -(x₂ + y₂)

/-- The main theorem: if the conditions are met, then a = -2 -/
theorem symmetric_intersection_theorem (si : SymmetricIntersection) :
  intersects_circle si → symmetric_about_line si → si.a = -2 :=
sorry

end NUMINAMATH_CALUDE_symmetric_intersection_theorem_l2482_248219


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l2482_248243

theorem quadratic_always_positive_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l2482_248243


namespace NUMINAMATH_CALUDE_semicircle_area_ratio_l2482_248229

/-- Proves that for a rectangle with sides 8 meters and 12 meters, with semicircles
    drawn on each side (diameters coinciding with the sides), the ratio of the area
    of the large semicircles to the area of the small semicircles is 2.25. -/
theorem semicircle_area_ratio (π : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ)
  (h1 : rectangle_width = 8)
  (h2 : rectangle_length = 12)
  (h3 : π > 0) :
  (2 * π * (rectangle_length / 2)^2 / 2) / (2 * π * (rectangle_width / 2)^2 / 2) = 2.25 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_area_ratio_l2482_248229


namespace NUMINAMATH_CALUDE_nth_equation_holds_l2482_248290

theorem nth_equation_holds (n : ℕ) :
  1 - 1 / ((n + 1: ℚ) ^ 2) = (n / (n + 1 : ℚ)) * ((n + 2) / (n + 1 : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_holds_l2482_248290


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_greater_than_three_l2482_248216

open Real

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

-- State the theorem
theorem monotone_decreasing_implies_a_greater_than_three :
  ∀ a : ℝ, (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 2 → f a x > f a y) → a > 3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_greater_than_three_l2482_248216


namespace NUMINAMATH_CALUDE_quadrilateral_inscribed_circle_l2482_248202

-- Define the types for points and circles
variable (Point : Type) (Circle : Type)

-- Define the necessary geometric predicates
variable (is_convex_quadrilateral : Point → Point → Point → Point → Prop)
variable (on_segment : Point → Point → Point → Prop)
variable (intersection : Point → Point → Point → Point → Point → Prop)
variable (has_inscribed_circle : Point → Point → Point → Point → Prop)

-- State the theorem
theorem quadrilateral_inscribed_circle 
  (A B C D E F G H P : Point) :
  is_convex_quadrilateral A B C D →
  on_segment A B E →
  on_segment B C F →
  on_segment C D G →
  on_segment D A H →
  intersection E G F H P →
  has_inscribed_circle H A E P →
  has_inscribed_circle E B F P →
  has_inscribed_circle F C G P →
  has_inscribed_circle G D H P →
  has_inscribed_circle A B C D :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_inscribed_circle_l2482_248202


namespace NUMINAMATH_CALUDE_find_a_l2482_248236

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ 3}

-- State the theorem
theorem find_a :
  ∀ a : ℝ, solution_set a = {x : ℝ | -2 ≤ x ∧ x ≤ 1} → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_find_a_l2482_248236


namespace NUMINAMATH_CALUDE_wicket_keeper_age_difference_l2482_248248

theorem wicket_keeper_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℕ) 
  (h1 : team_size = 11)
  (h2 : captain_age = 24)
  (h3 : team_avg_age = 21)
  (h4 : ∃ (remaining_avg_age : ℕ), remaining_avg_age = team_avg_age - 1 ∧ 
    (team_size - 2) * remaining_avg_age + captain_age + (captain_age + x) = team_size * team_avg_age) :
  x = 3 :=
sorry

end NUMINAMATH_CALUDE_wicket_keeper_age_difference_l2482_248248


namespace NUMINAMATH_CALUDE_calories_burned_walking_james_walking_calories_l2482_248270

/-- Calculates the calories burned per hour while walking based on dancing data -/
theorem calories_burned_walking (dancing_calories_per_hour : ℝ) 
  (dancing_sessions_per_day : ℕ) (dancing_hours_per_session : ℝ) 
  (dancing_days_per_week : ℕ) (total_calories_per_week : ℝ) : ℝ :=
  let dancing_calories_ratio := 2
  let dancing_hours_per_week := dancing_sessions_per_day * dancing_hours_per_session * dancing_days_per_week
  let walking_calories_per_hour := total_calories_per_week / dancing_hours_per_week / dancing_calories_ratio
  by
    -- Proof goes here
    sorry

/-- Verifies that James burns 300 calories per hour while walking -/
theorem james_walking_calories : 
  calories_burned_walking 600 2 0.5 4 2400 = 300 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_calories_burned_walking_james_walking_calories_l2482_248270


namespace NUMINAMATH_CALUDE_fourth_minus_third_tiles_l2482_248228

/-- The side length of the n-th square in the sequence -/
def side_length (n : ℕ) : ℕ := n^2

/-- The number of tiles in the n-th square -/
def tiles (n : ℕ) : ℕ := (side_length n)^2

theorem fourth_minus_third_tiles : tiles 4 - tiles 3 = 175 := by
  sorry

end NUMINAMATH_CALUDE_fourth_minus_third_tiles_l2482_248228


namespace NUMINAMATH_CALUDE_isosceles_triangle_l2482_248231

/-- Given a triangle ABC where sin A = 2 sin C cos B, prove that B = C -/
theorem isosceles_triangle (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_sin : Real.sin A = 2 * Real.sin C * Real.cos B) : B = C := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l2482_248231


namespace NUMINAMATH_CALUDE_ball_color_probability_l2482_248254

/-- The number of balls -/
def n : ℕ := 8

/-- The probability of a ball being painted black or white -/
def p : ℚ := 1/2

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of exactly k successes in n independent trials with probability p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial n k : ℚ) * p^k * (1 - p)^(n - k)

theorem ball_color_probability :
  binomial_probability n (n/2) p = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_ball_color_probability_l2482_248254


namespace NUMINAMATH_CALUDE_unique_hyperbolas_l2482_248256

/-- Binomial coefficient function -/
def binomial (m n : ℕ) : ℕ := Nat.choose m n

/-- The set of binomial coefficients for 1 ≤ n ≤ m ≤ 5 -/
def binomial_set : Finset ℕ :=
  Finset.filter (λ x => x > 1) $
    Finset.image (λ (m, n) => binomial m n) $
      Finset.filter (λ (m, n) => 1 ≤ n ∧ n ≤ m ∧ m ≤ 5) $
        Finset.product (Finset.range 6) (Finset.range 6)

theorem unique_hyperbolas : Finset.card binomial_set = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_hyperbolas_l2482_248256


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2482_248245

theorem binomial_expansion_coefficient (x : ℝ) (x_ne_zero : x ≠ 0) :
  let expansion := (x^2 - 1/x)^5
  let second_term_coefficient := Finset.sum (Finset.range 6) (fun k => 
    if k = 1 then (-1)^k * (Nat.choose 5 k) * x^(10 - 3*k)
    else 0)
  second_term_coefficient = -5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2482_248245


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l2482_248225

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_nonzero : a ≠ 0

/-- Evaluation of a cubic polynomial at a point -/
def CubicPolynomial.eval (P : CubicPolynomial) (x : ℤ) : ℤ :=
  P.a * x^3 + P.b * x^2 + P.c * x + P.d

/-- Property of having infinitely many pairs of distinct integers (x, y) such that xP(x) = yP(y) -/
def has_infinitely_many_equal_products (P : CubicPolynomial) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ x * P.eval x = y * P.eval y ∧ (abs x > n ∨ abs y > n)

/-- Main theorem: If a cubic polynomial with integer coefficients has infinitely many pairs of 
    distinct integers (x, y) such that xP(x) = yP(y), then it has an integer root -/
theorem cubic_polynomial_integer_root (P : CubicPolynomial) 
    (h : has_infinitely_many_equal_products P) : 
    ∃ k : ℤ, P.eval k = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l2482_248225


namespace NUMINAMATH_CALUDE_larger_number_problem_l2482_248280

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 20775)
  (h2 : L = 23 * S + 143) :
  L = 21713 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2482_248280


namespace NUMINAMATH_CALUDE_sandwich_combinations_l2482_248205

def num_ingredients : ℕ := 7

theorem sandwich_combinations :
  (Nat.choose num_ingredients 3 = 35) ∧ (Nat.choose num_ingredients 4 = 35) := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l2482_248205


namespace NUMINAMATH_CALUDE_probability_both_bins_contain_items_l2482_248211

theorem probability_both_bins_contain_items (p : ℝ) (h1 : 0.5 < p) (h2 : p ≤ 1) :
  let prob_both := 1 - 2 * p^5 + p^10
  prob_both = (1 - p^5)^2 + p^10 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_bins_contain_items_l2482_248211


namespace NUMINAMATH_CALUDE_simplify_expression_l2482_248253

theorem simplify_expression (z : ℝ) : z - 2 + 4*z + 3 - 6*z + 5 - 8*z + 7 = -9*z + 13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2482_248253


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l2482_248213

/-- Proves that Bobby ate 9 pieces of candy at the start -/
theorem bobby_candy_problem (initial : ℕ) (eaten_start : ℕ) (eaten_more : ℕ) (left : ℕ)
  (h1 : initial = 22)
  (h2 : eaten_more = 5)
  (h3 : left = 8)
  (h4 : initial = eaten_start + eaten_more + left) :
  eaten_start = 9 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l2482_248213


namespace NUMINAMATH_CALUDE_distance_between_trees_l2482_248201

/-- Proves that in a yard of given length with a given number of trees planted at equal distances,
    including one at each end, the distance between two consecutive trees is as calculated. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 250)
  (h2 : num_trees = 51)
  (h3 : num_trees ≥ 2) :
  yard_length / (num_trees - 1) = 5 :=
sorry

end NUMINAMATH_CALUDE_distance_between_trees_l2482_248201


namespace NUMINAMATH_CALUDE_infinite_pairs_exist_l2482_248232

/-- C(n) is the number of distinct prime divisors of n -/
def C (n : ℕ) : ℕ := sorry

/-- There exist infinitely many pairs of natural numbers (a,b) satisfying the given conditions -/
theorem infinite_pairs_exist : ∀ k : ℕ, ∃ a b : ℕ, a ≠ b ∧ a > k ∧ b > k ∧ C (a + b) = C a + C b := by
  sorry

end NUMINAMATH_CALUDE_infinite_pairs_exist_l2482_248232


namespace NUMINAMATH_CALUDE_max_e_value_l2482_248265

def b (n : ℕ) : ℕ := 120 + n^2

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem max_e_value :
  ∃ (N : ℕ), e N = 5 ∧ ∀ (n : ℕ), n > 0 → e n ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_e_value_l2482_248265


namespace NUMINAMATH_CALUDE_grazing_area_expansion_l2482_248220

/-- Given a circular grazing area with an initial radius of 9 meters,
    if the area is increased by 1408 square meters,
    the new radius will be 23 meters. -/
theorem grazing_area_expansion (π : ℝ) (h : π > 0) :
  let r₁ : ℝ := 9
  let additional_area : ℝ := 1408
  let r₂ : ℝ := Real.sqrt (r₁^2 + additional_area / π)
  r₂ = 23 := by sorry

end NUMINAMATH_CALUDE_grazing_area_expansion_l2482_248220


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_all_solutions_are_general_l2482_248272

/-- The differential equation -/
def diff_eq (x y : ℝ) : Prop :=
  ∃ (dx dy : ℝ), (y^3 - 2*x*y) * dx + (3*x*y^2 - x^2) * dy = 0

/-- The general solution -/
def general_solution (x y C : ℝ) : Prop :=
  y^3 * x - x^2 * y = C

/-- Theorem stating that the general solution satisfies the differential equation -/
theorem solution_satisfies_equation :
  ∀ (x y C : ℝ), general_solution x y C → diff_eq x y :=
by sorry

/-- Theorem stating that any solution to the differential equation is of the form of the general solution -/
theorem all_solutions_are_general :
  ∀ (x y : ℝ), diff_eq x y → ∃ (C : ℝ), general_solution x y C :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_all_solutions_are_general_l2482_248272


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_two_thirds_l2482_248247

/-- Represents a cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  white_cube_count : ℕ
  black_cube_count : ℕ

/-- Calculates the fraction of white surface area for a given LargeCube -/
def white_surface_fraction (c : LargeCube) : ℚ :=
  -- The actual calculation is not implemented here
  0

/-- The specific cube described in the problem -/
def problem_cube : LargeCube :=
  { edge_length := 4
  , small_cube_count := 64
  , white_cube_count := 30
  , black_cube_count := 34 }

theorem white_surface_fraction_is_two_thirds :
  white_surface_fraction problem_cube = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_two_thirds_l2482_248247


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l2482_248285

/-- 
Proves that the number of years it takes for a man's age to be twice his son's age is 2,
given the initial conditions.
-/
theorem mans_age_twice_sons (
  son_age : ℕ) -- Present age of the son
  (age_diff : ℕ) -- Age difference between man and son
  (h1 : son_age = 27) -- The son's present age is 27
  (h2 : age_diff = 29) -- The man is 29 years older than his son
  : ∃ (years : ℕ), years = 2 ∧ (son_age + years + age_diff = 2 * (son_age + years)) :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l2482_248285


namespace NUMINAMATH_CALUDE_subtract_repeating_decimal_l2482_248258

theorem subtract_repeating_decimal : 
  2 - (8 : ℚ) / 9 = 10 / 9 := by sorry

end NUMINAMATH_CALUDE_subtract_repeating_decimal_l2482_248258


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2482_248293

theorem simplify_sqrt_expression (x : ℝ) (h : x ≠ 0) :
  Real.sqrt (1 + ((x^4 - 1) / (2 * x^2))^2) = x^2 / 2 + 1 / (2 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2482_248293


namespace NUMINAMATH_CALUDE_smallest_number_with_digit_sum_1981_l2482_248286

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that constructs a number with 1 followed by n nines -/
def oneFollowedByNines (n : ℕ) : ℕ := sorry

/-- The theorem stating that the smallest natural number whose digits sum to 1981
    is 1 followed by 220 nines -/
theorem smallest_number_with_digit_sum_1981 :
  ∀ n : ℕ, sumOfDigits n = 1981 → n ≥ oneFollowedByNines 220 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_digit_sum_1981_l2482_248286


namespace NUMINAMATH_CALUDE_smallest_number_in_system_l2482_248235

theorem smallest_number_in_system (x y z : ℝ) 
  (eq1 : 3 * x - y = 20)
  (eq2 : 2 * z = 3 * y)
  (eq3 : x + y + z = 48) :
  x < y ∧ x < z :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_in_system_l2482_248235


namespace NUMINAMATH_CALUDE_race_finish_orders_l2482_248281

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of racers -/
def num_racers : ℕ := 3

/-- Theorem: The number of different possible orders for three distinct individuals 
    to finish a race without ties is equal to 6 -/
theorem race_finish_orders : permutations num_racers = 6 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_orders_l2482_248281


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l2482_248221

def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem quadratic_coefficients :
  ∃ (a b c : ℝ), (∀ x, f x = a*x^2 + b*x + c) ∧ a = 1 ∧ b = -4 ∧ c = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l2482_248221


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2482_248215

/-- Represents a type of cloth with its sales information -/
structure ClothType where
  quantity : ℕ      -- Quantity sold in meters
  totalPrice : ℕ    -- Total selling price in Rs.
  profitPerMeter : ℕ -- Profit per meter in Rs.

/-- Calculates the cost price per meter for a given cloth type -/
def costPricePerMeter (cloth : ClothType) : ℕ :=
  cloth.totalPrice / cloth.quantity - cloth.profitPerMeter

/-- The trader's cloth inventory -/
def traderInventory : List ClothType :=
  [
    { quantity := 85, totalPrice := 8500, profitPerMeter := 15 },  -- Type A
    { quantity := 120, totalPrice := 10200, profitPerMeter := 12 }, -- Type B
    { quantity := 60, totalPrice := 4200, profitPerMeter := 10 }   -- Type C
  ]

theorem cost_price_calculation (inventory : List ClothType) :
  ∀ cloth ∈ inventory,
    costPricePerMeter cloth =
      cloth.totalPrice / cloth.quantity - cloth.profitPerMeter :=
by
  sorry

#eval traderInventory.map costPricePerMeter

end NUMINAMATH_CALUDE_cost_price_calculation_l2482_248215


namespace NUMINAMATH_CALUDE_impossible_to_change_all_signs_l2482_248237

/-- Represents a point in the decagon configuration -/
structure Point where
  value : Int
  mk_point : value = 1 ∨ value = -1

/-- Represents the decagon configuration -/
structure DecagonConfig where
  points : Finset Point
  mk_config : points.card = 220

/-- Represents an operation on the decagon -/
inductive Operation
  | side : Operation
  | diagonal : Operation

/-- Applies an operation to the decagon configuration -/
def apply_operation (config : DecagonConfig) (op : Operation) : DecagonConfig :=
  sorry

/-- Checks if all points in the configuration are -1 -/
def all_negative (config : DecagonConfig) : Prop :=
  ∀ p ∈ config.points, p.value = -1

/-- Main theorem: It's impossible to change all signs to their opposites -/
theorem impossible_to_change_all_signs (initial_config : DecagonConfig) :
  ¬∃ (ops : List Operation), all_negative (ops.foldl apply_operation initial_config) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_change_all_signs_l2482_248237


namespace NUMINAMATH_CALUDE_f_of_one_f_of_a_f_of_f_of_a_l2482_248210

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 3

-- Theorem statements
theorem f_of_one : f 1 = 5 := by sorry

theorem f_of_a (a : ℝ) : f a = 2 * a + 3 := by sorry

theorem f_of_f_of_a (a : ℝ) : f (f a) = 4 * a + 9 := by sorry

end NUMINAMATH_CALUDE_f_of_one_f_of_a_f_of_f_of_a_l2482_248210


namespace NUMINAMATH_CALUDE_line_relationships_l2482_248269

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_relationships (a b c : Line) 
  (h1 : skew a b) (h2 : parallel c a) : 
  ¬ parallel c b := by sorry

end NUMINAMATH_CALUDE_line_relationships_l2482_248269


namespace NUMINAMATH_CALUDE_harry_lost_nineteen_pencils_l2482_248296

/-- The number of pencils Anna has -/
def anna_pencils : ℕ := 50

/-- The number of pencils Harry initially had -/
def harry_initial_pencils : ℕ := 2 * anna_pencils

/-- The number of pencils Harry has left -/
def harry_remaining_pencils : ℕ := 81

/-- The number of pencils Harry lost -/
def harry_lost_pencils : ℕ := harry_initial_pencils - harry_remaining_pencils

theorem harry_lost_nineteen_pencils : harry_lost_pencils = 19 := by
  sorry

end NUMINAMATH_CALUDE_harry_lost_nineteen_pencils_l2482_248296


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l2482_248274

theorem cow_husk_consumption 
  (cows bags days : ℕ) 
  (h : cows = 45 ∧ bags = 45 ∧ days = 45) : 
  (1 : ℕ) * days = 45 := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l2482_248274


namespace NUMINAMATH_CALUDE_function_characterization_l2482_248246

/-- A function from positive integers to non-negative integers -/
def PositiveToNonNegative := ℕ+ → ℕ

/-- The p-adic valuation of a positive integer -/
noncomputable def vp (p : ℕ+) (n : ℕ+) : ℕ := sorry

theorem function_characterization 
  (f : PositiveToNonNegative) 
  (h1 : ∃ n, f n ≠ 0)
  (h2 : ∀ x y, f (x * y) = f x + f y)
  (h3 : ∃ S : Set ℕ+, Set.Infinite S ∧ ∀ n ∈ S, ∀ k < n, f k = f (n - k)) :
  ∃ (N : ℕ+) (p : ℕ+), Nat.Prime p ∧ ∀ n, f n = N * vp p n :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l2482_248246


namespace NUMINAMATH_CALUDE_diagonal_cubes_120_270_300_l2482_248252

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def internal_diagonal_cubes (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd x z) + Nat.gcd x (Nat.gcd y z)

/-- The number of cubes a face diagonal passes through in a rectangular solid -/
def face_diagonal_cubes (x y : ℕ) : ℕ :=
  x + y - Nat.gcd x y

/-- Theorem about the number of cubes diagonals pass through in a 120 × 270 × 300 rectangular solid -/
theorem diagonal_cubes_120_270_300 :
  internal_diagonal_cubes 120 270 300 = 600 ∧
  face_diagonal_cubes 120 270 = 360 := by
  sorry


end NUMINAMATH_CALUDE_diagonal_cubes_120_270_300_l2482_248252


namespace NUMINAMATH_CALUDE_sequence_problem_l2482_248230

-- Define arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom : is_geometric_sequence b)
  (h_a_sum : a 1000 + a 1018 = 2 * Real.pi)
  (h_b_prod : b 6 * b 2012 = 2) :
  Real.tan ((a 2 + a 2016) / (1 + b 3 * b 2015)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l2482_248230


namespace NUMINAMATH_CALUDE_ice_cream_cup_cost_l2482_248292

/-- Given Alok's order and payment, prove the cost of each ice-cream cup --/
theorem ice_cream_cup_cost
  (chapati_count : ℕ)
  (rice_count : ℕ)
  (vegetable_count : ℕ)
  (ice_cream_count : ℕ)
  (chapati_cost : ℕ)
  (rice_cost : ℕ)
  (vegetable_cost : ℕ)
  (total_paid : ℕ)
  (h1 : chapati_count = 16)
  (h2 : rice_count = 5)
  (h3 : vegetable_count = 7)
  (h4 : ice_cream_count = 6)
  (h5 : chapati_cost = 6)
  (h6 : rice_cost = 45)
  (h7 : vegetable_cost = 70)
  (h8 : total_paid = 1021) :
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / ice_cream_count = 35 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cup_cost_l2482_248292
