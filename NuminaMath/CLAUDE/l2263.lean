import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_equivalence_l2263_226382

/-- Given a triangle with angles α, β, γ, side length a opposite to angle α,
    and circumradius R, prove that the two expressions for the area S are equivalent. -/
theorem triangle_area_equivalence (α β γ a R : ℝ) (h_angles : α + β + γ = π)
    (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < a ∧ 0 < R) :
  (a^2 * Real.sin β * Real.sin γ) / (2 * Real.sin α) =
  2 * R^2 * Real.sin α * Real.sin β * Real.sin γ := by
sorry

end NUMINAMATH_CALUDE_triangle_area_equivalence_l2263_226382


namespace NUMINAMATH_CALUDE_fans_per_bleacher_set_l2263_226381

theorem fans_per_bleacher_set (total_fans : ℕ) (num_bleacher_sets : ℕ) 
  (h1 : total_fans = 2436) 
  (h2 : num_bleacher_sets = 3) 
  (h3 : total_fans % num_bleacher_sets = 0) : 
  total_fans / num_bleacher_sets = 812 := by
  sorry

end NUMINAMATH_CALUDE_fans_per_bleacher_set_l2263_226381


namespace NUMINAMATH_CALUDE_simplify_power_expression_l2263_226395

theorem simplify_power_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l2263_226395


namespace NUMINAMATH_CALUDE_committee_probability_l2263_226390

/-- The probability of selecting exactly 2 boys in a 6-person committee 
    randomly chosen from a group of 30 members (12 boys and 18 girls) -/
theorem committee_probability (total_members : ℕ) (boys : ℕ) (girls : ℕ) 
  (committee_size : ℕ) (h1 : total_members = 30) (h2 : boys = 12) 
  (h3 : girls = 18) (h4 : committee_size = 6) (h5 : total_members = boys + girls) :
  (Nat.choose boys 2 * Nat.choose girls 4) / Nat.choose total_members committee_size = 8078 / 23751 := by
sorry

end NUMINAMATH_CALUDE_committee_probability_l2263_226390


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l2263_226310

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) 
  (h1 : total_students = 466)
  (h2 : boys = 127)
  (h3 : boys < total_students - boys) :
  total_students - boys - boys = 212 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l2263_226310


namespace NUMINAMATH_CALUDE_certain_number_proof_l2263_226376

theorem certain_number_proof (h : 2994 / 14.5 = 177) : ∃ x : ℝ, x / 1.45 = 17.7 ∧ x = 25.665 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2263_226376


namespace NUMINAMATH_CALUDE_max_gcd_of_product_7200_l2263_226345

theorem max_gcd_of_product_7200 :
  ∃ (a b : ℕ), a * b = 7200 ∧
  ∀ (x y : ℕ), x * y = 7200 → Nat.gcd x y ≤ Nat.gcd a b ∧
  Nat.gcd a b = 60 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_of_product_7200_l2263_226345


namespace NUMINAMATH_CALUDE_prime_square_difference_divisibility_l2263_226371

theorem prime_square_difference_divisibility (p : ℕ) (hp : Prime p) :
  ∃ (n m : ℕ), n ≠ 0 ∧ m ≠ 0 ∧ n ≠ m ∧
  p - n^2 ≠ 1 ∧
  p - n^2 ≠ p - m^2 ∧
  (p - n^2) ∣ (p - m^2) :=
sorry

end NUMINAMATH_CALUDE_prime_square_difference_divisibility_l2263_226371


namespace NUMINAMATH_CALUDE_system_solution_l2263_226399

theorem system_solution :
  let x : ℚ := -49/23
  let y : ℚ := 136/69
  (7 * x = -9 - 3 * y) ∧ (4 * x = 5 * y - 34) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2263_226399


namespace NUMINAMATH_CALUDE_consistent_grade_theorem_l2263_226364

/-- Represents the grades a student can receive -/
inductive Grade
| A
| B
| C
| D

/-- Represents the grade distribution for a test -/
structure GradeDistribution where
  a_count : Nat
  b_count : Nat
  c_count : Nat
  d_count : Nat

/-- The problem setup -/
structure TestConsistency where
  total_students : Nat
  num_tests : Nat
  grade_distribution : GradeDistribution

/-- Calculate the percentage of students with consistent grades -/
def consistent_grade_percentage (tc : TestConsistency) : Rat :=
  let consistent_count := tc.grade_distribution.a_count + tc.grade_distribution.b_count + 
                          tc.grade_distribution.c_count + tc.grade_distribution.d_count
  (consistent_count : Rat) / (tc.total_students : Rat) * 100

/-- The main theorem to prove -/
theorem consistent_grade_theorem (tc : TestConsistency) 
  (h1 : tc.total_students = 40)
  (h2 : tc.num_tests = 3)
  (h3 : tc.grade_distribution = { a_count := 3, b_count := 6, c_count := 7, d_count := 2 }) :
  consistent_grade_percentage tc = 45 := by
  sorry


end NUMINAMATH_CALUDE_consistent_grade_theorem_l2263_226364


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l2263_226377

theorem quadratic_inequality_empty_solution_set (m : ℝ) :
  (∀ x : ℝ, (m + 1) * x^2 - m * x + m - 1 ≥ 0) ↔ m ≥ 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l2263_226377


namespace NUMINAMATH_CALUDE_root_of_multiplicity_l2263_226302

theorem root_of_multiplicity (k : ℝ) : 
  (∃ x : ℝ, (x - 1) / (x - 3) = k / (x - 3) ∧ 
   ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, |y - x| < δ → 
   |((y - 1) / (y - 3) - k / (y - 3))| < ε * |y - x|) ↔ 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_root_of_multiplicity_l2263_226302


namespace NUMINAMATH_CALUDE_xy_value_l2263_226354

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^2 + y^2 = 4) (h2 : x^4 + y^4 = 7) : 
  x * y = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2263_226354


namespace NUMINAMATH_CALUDE_nesbitt_inequality_l2263_226378

theorem nesbitt_inequality {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 ∧
  (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_nesbitt_inequality_l2263_226378


namespace NUMINAMATH_CALUDE_connection_duration_l2263_226336

/-- Calculates the number of days a client can be connected to the internet given the specified parameters. -/
def days_connected (initial_balance : ℚ) (payment : ℚ) (daily_cost : ℚ) (discontinuation_threshold : ℚ) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the client will be connected for 14 days. -/
theorem connection_duration :
  days_connected 0 7 (1/2) 5 = 14 :=
by sorry

end NUMINAMATH_CALUDE_connection_duration_l2263_226336


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l2263_226331

theorem wire_ratio_proof (total_length : ℝ) (shorter_length : ℝ) :
  total_length = 50 →
  shorter_length = 14.285714285714285 →
  let longer_length := total_length - shorter_length
  shorter_length / longer_length = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l2263_226331


namespace NUMINAMATH_CALUDE_math_and_lang_not_science_l2263_226394

def students : ℕ := 120
def math_students : ℕ := 80
def lang_students : ℕ := 70
def science_students : ℕ := 50
def all_three_students : ℕ := 20

theorem math_and_lang_not_science :
  ∃ (math_and_lang math_and_science lang_and_science : ℕ),
    math_and_lang + math_and_science + lang_and_science = 
      math_students + lang_students + science_students - students + all_three_students ∧
    math_and_lang - all_three_students = 30 := by
  sorry

end NUMINAMATH_CALUDE_math_and_lang_not_science_l2263_226394


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l2263_226306

theorem cylinder_radius_problem (r : ℝ) (y : ℝ) : 
  r > 0 →
  (π * ((r + 4)^2 * 4 - r^2 * 4) = y) →
  (π * (r^2 * 8 - r^2 * 4) = y) →
  r = 2 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l2263_226306


namespace NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l2263_226309

theorem max_value_of_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x * y = 5) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ ∀ (z : ℝ), 2 * x + y ≤ z → z ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l2263_226309


namespace NUMINAMATH_CALUDE_sqrt_difference_comparison_l2263_226365

theorem sqrt_difference_comparison (x : ℝ) (h : x ≥ 1) :
  Real.sqrt x - Real.sqrt (x - 1) > Real.sqrt (x + 1) - Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_comparison_l2263_226365


namespace NUMINAMATH_CALUDE_wire_circle_to_rectangle_area_l2263_226367

/-- Given a wire initially in the form of a circle with radius 3.5 m,
    when bent into a rectangle with length to breadth ratio of 6:5,
    the area of the resulting rectangle is (735 * π^2) / 242 square meters. -/
theorem wire_circle_to_rectangle_area :
  let r : ℝ := 3.5
  let circle_circumference := 2 * Real.pi * r
  let length_to_breadth_ratio : ℚ := 6 / 5
  let rectangle_perimeter := circle_circumference
  let length : ℝ := (21 * Real.pi) / 11
  let breadth : ℝ := (35 * Real.pi) / 22
  rectangle_perimeter = 2 * (length + breadth) →
  length / breadth = length_to_breadth_ratio →
  length * breadth = (735 * Real.pi^2) / 242 := by
  sorry

end NUMINAMATH_CALUDE_wire_circle_to_rectangle_area_l2263_226367


namespace NUMINAMATH_CALUDE_annie_initial_money_l2263_226398

/-- The amount of money Annie had initially -/
def initial_money : ℕ := 132

/-- The price of a hamburger -/
def hamburger_price : ℕ := 4

/-- The price of a milkshake -/
def milkshake_price : ℕ := 5

/-- The number of hamburgers Annie bought -/
def hamburgers_bought : ℕ := 8

/-- The number of milkshakes Annie bought -/
def milkshakes_bought : ℕ := 6

/-- The amount of money Annie had left -/
def money_left : ℕ := 70

theorem annie_initial_money :
  initial_money = 
    hamburger_price * hamburgers_bought + 
    milkshake_price * milkshakes_bought + 
    money_left :=
by sorry

end NUMINAMATH_CALUDE_annie_initial_money_l2263_226398


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l2263_226375

theorem shaded_fraction_of_rectangle (length width : ℝ) (h1 : length = 10) (h2 : width = 15) :
  let total_area := length * width
  let third_area := total_area / 3
  let shaded_area := third_area / 2
  shaded_area / total_area = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l2263_226375


namespace NUMINAMATH_CALUDE_assembly_rate_after_transformation_l2263_226369

/-- Represents the factory's car assembly rate before and after transformation -/
structure AssemblyRate where
  before : ℝ
  after : ℝ

/-- The conditions of the problem -/
def problem_conditions (r : AssemblyRate) : Prop :=
  r.after = (5/3) * r.before ∧
  (40 / r.after) = (30 / r.before) - 2

/-- The theorem to prove -/
theorem assembly_rate_after_transformation (r : AssemblyRate) :
  problem_conditions r → r.after = 5 := by sorry

end NUMINAMATH_CALUDE_assembly_rate_after_transformation_l2263_226369


namespace NUMINAMATH_CALUDE_circle_polar_equation_l2263_226355

/-- A circle in the polar coordinate system with center at (1,0) and passing through the pole -/
structure PolarCircle where
  /-- The radius of the circle as a function of the angle θ -/
  ρ : ℝ → ℝ

/-- The polar coordinate equation of the circle -/
def polar_equation (c : PolarCircle) : Prop :=
  ∀ θ : ℝ, c.ρ θ = 2 * Real.cos θ

/-- Theorem stating that the polar coordinate equation of a circle with center at (1,0) 
    and passing through the pole is ρ = 2cos θ -/
theorem circle_polar_equation :
  ∀ c : PolarCircle, polar_equation c :=
sorry

end NUMINAMATH_CALUDE_circle_polar_equation_l2263_226355


namespace NUMINAMATH_CALUDE_inequality_solution_l2263_226300

theorem inequality_solution (x : ℝ) : 
  (2 < x / (3 * x - 7) ∧ x / (3 * x - 7) ≤ 6) ↔ (7 / 3 < x ∧ x ≤ 14 / 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2263_226300


namespace NUMINAMATH_CALUDE_solution_theorem_l2263_226343

/-- A function satisfying the given condition for all non-zero real numbers -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 4 * x

theorem solution_theorem :
  ∀ f : ℝ → ℝ, SatisfiesCondition f →
    ∀ x : ℝ, x ≠ 0 →
      (f x = f (-x) ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_theorem_l2263_226343


namespace NUMINAMATH_CALUDE_costume_ball_same_gender_dance_l2263_226391

/-- Represents a person at the costume ball -/
structure Person :=
  (partners : Nat)

/-- Represents the costume ball -/
structure CostumeBall :=
  (people : Finset Person)
  (total_people : Nat)
  (total_dances : Nat)

/-- The costume ball satisfies the given conditions -/
def valid_costume_ball (ball : CostumeBall) : Prop :=
  ball.total_people = 20 ∧
  (ball.people.filter (λ p => p.partners = 3)).card = 11 ∧
  (ball.people.filter (λ p => p.partners = 5)).card = 1 ∧
  (ball.people.filter (λ p => p.partners = 6)).card = 8 ∧
  ball.total_dances = (11 * 3 + 1 * 5 + 8 * 6) / 2

theorem costume_ball_same_gender_dance (ball : CostumeBall) 
  (h : valid_costume_ball ball) : 
  ¬ (∀ (dance : Nat), dance < ball.total_dances → 
    ∃ (p1 p2 : Person), p1 ∈ ball.people ∧ p2 ∈ ball.people ∧ p1 ≠ p2) :=
by sorry

end NUMINAMATH_CALUDE_costume_ball_same_gender_dance_l2263_226391


namespace NUMINAMATH_CALUDE_some_number_value_l2263_226348

theorem some_number_value (some_number : ℝ) : 
  (3 * 10^2) * (4 * some_number) = 12 → some_number = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l2263_226348


namespace NUMINAMATH_CALUDE_kate_needs_57_more_l2263_226319

/-- The amount of additional money Kate needs to buy all items -/
def additional_money_needed (pen_price notebook_price art_set_price : ℚ)
  (pen_money_ratio : ℚ) (notebook_discount : ℚ) (art_set_money : ℚ) (art_set_discount : ℚ) : ℚ :=
  (pen_price - pen_price * pen_money_ratio) +
  (notebook_price * (1 - notebook_discount)) +
  (art_set_price * (1 - art_set_discount) - art_set_money)

/-- Theorem stating that Kate needs $57 more to buy all items -/
theorem kate_needs_57_more :
  additional_money_needed 30 20 50 (1/3) 0.15 10 0.4 = 57 := by
  sorry

end NUMINAMATH_CALUDE_kate_needs_57_more_l2263_226319


namespace NUMINAMATH_CALUDE_abc_inequality_l2263_226330

theorem abc_inequality (a b c : ℝ) 
  (ha : a = (1/3)^(2/3))
  (hb : b = (1/5)^(2/3))
  (hc : c = (4/9)^(1/3)) :
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l2263_226330


namespace NUMINAMATH_CALUDE_angle_inclination_range_l2263_226344

-- Define the slope k
def k : ℝ := sorry

-- Define the angle of inclination α in radians
def α : ℝ := sorry

-- Define the relationship between k and α
axiom slope_angle_relation : k = Real.tan α

-- Define the range of k
axiom k_range : -1 ≤ k ∧ k < 1

-- Define the range of α (0 to π)
axiom α_range : 0 ≤ α ∧ α < Real.pi

-- Theorem to prove
theorem angle_inclination_range :
  (0 ≤ α ∧ α < Real.pi / 4) ∨ (3 * Real.pi / 4 ≤ α ∧ α < Real.pi) :=
sorry

end NUMINAMATH_CALUDE_angle_inclination_range_l2263_226344


namespace NUMINAMATH_CALUDE_nala_seashells_l2263_226385

/-- The number of seashells Nala found on the first day -/
def first_day : ℕ := 5

/-- The number of seashells Nala found on the second day -/
def second_day : ℕ := 7

/-- The number of seashells Nala found on the third day is twice the sum of the first two days -/
def third_day : ℕ := 2 * (first_day + second_day)

/-- The total number of seashells Nala has -/
def total_seashells : ℕ := first_day + second_day + third_day

theorem nala_seashells : total_seashells = 36 := by
  sorry

end NUMINAMATH_CALUDE_nala_seashells_l2263_226385


namespace NUMINAMATH_CALUDE_figure_area_is_79_l2263_226363

/-- Calculates the area of a rectangle -/
def rectangleArea (width : ℕ) (height : ℕ) : ℕ := width * height

/-- Represents the dimensions of the figure -/
structure FigureDimensions where
  leftWidth : ℕ
  leftHeight : ℕ
  middleWidth : ℕ
  middleHeight : ℕ
  rightWidth : ℕ
  rightHeight : ℕ

/-- Calculates the total area of the figure -/
def totalArea (d : FigureDimensions) : ℕ :=
  rectangleArea d.leftWidth d.leftHeight +
  rectangleArea d.middleWidth d.middleHeight +
  rectangleArea d.rightWidth d.rightHeight

/-- Theorem: The total area of the figure is 79 square units -/
theorem figure_area_is_79 (d : FigureDimensions) 
  (h1 : d.leftWidth = 6 ∧ d.leftHeight = 7)
  (h2 : d.middleWidth = 4 ∧ d.middleHeight = 3)
  (h3 : d.rightWidth = 5 ∧ d.rightHeight = 5) :
  totalArea d = 79 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_is_79_l2263_226363


namespace NUMINAMATH_CALUDE_amanda_candy_problem_l2263_226326

/-- The number of candy bars Amanda gave to her sister the first time -/
def first_given : ℕ := sorry

/-- The initial number of candy bars Amanda had -/
def initial_candy : ℕ := 7

/-- The number of candy bars Amanda bought -/
def bought_candy : ℕ := 30

/-- The number of candy bars Amanda kept for herself -/
def kept_candy : ℕ := 22

theorem amanda_candy_problem :
  first_given = 3 ∧
  initial_candy - first_given + bought_candy - 4 * first_given = kept_candy :=
sorry

end NUMINAMATH_CALUDE_amanda_candy_problem_l2263_226326


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l2263_226361

/-- Represents a small die in the 4x4x4 cube -/
structure SmallDie where
  /-- The value on each face of the die -/
  faces : Fin 6 → ℕ
  /-- The property that opposite sides sum to 7 -/
  opposite_sum_seven : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents the 4x4x4 cube made of small dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → SmallDie

/-- Calculates the sum of visible values on the large cube -/
def visible_sum (cube : LargeCube) : ℕ := sorry

/-- Theorem stating the smallest possible sum of visible values -/
theorem smallest_visible_sum (cube : LargeCube) :
  visible_sum cube ≥ 144 ∧ ∃ (optimal_cube : LargeCube), visible_sum optimal_cube = 144 := by sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l2263_226361


namespace NUMINAMATH_CALUDE_triangle_ABC_perimeter_l2263_226316

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) : Prop :=
  -- Condition for sides a and b
  (|a - 2| + (b - 5)^2 = 0) ∧
  -- Conditions for side c
  (c = 4) ∧
  (∀ x : ℤ, (x - 3 > 3*(x - 4) ∧ (4*x - 1) / 6 < x + 1) → x ≤ 4)

-- Theorem statement
theorem triangle_ABC_perimeter (a b c : ℝ) :
  triangle_ABC a b c → a + b + c = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_perimeter_l2263_226316


namespace NUMINAMATH_CALUDE_lens_price_proof_l2263_226383

theorem lens_price_proof (price_no_discount : ℝ) (discount_rate : ℝ) (cheaper_lens_price : ℝ) :
  price_no_discount = 300 ∧
  discount_rate = 0.2 ∧
  cheaper_lens_price = 220 ∧
  price_no_discount * (1 - discount_rate) = cheaper_lens_price + 20 :=
by sorry

end NUMINAMATH_CALUDE_lens_price_proof_l2263_226383


namespace NUMINAMATH_CALUDE_smallest_positive_integer_l2263_226320

theorem smallest_positive_integer (x : ℕ+) : (x^3 : ℚ) / (x^2 : ℚ) < 15 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_l2263_226320


namespace NUMINAMATH_CALUDE_x_equals_negative_x_and_abs_x_equals_two_l2263_226350

theorem x_equals_negative_x_and_abs_x_equals_two (x : ℝ) :
  (x = -x → x = 0) ∧ (|x| = 2 → x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_x_equals_negative_x_and_abs_x_equals_two_l2263_226350


namespace NUMINAMATH_CALUDE_range_of_a_l2263_226352

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := (x - t) * abs x

-- State the theorem
theorem range_of_a (t : ℝ) (h_t : t ∈ Set.Ioo 0 2) :
  (∀ x ∈ Set.Icc (-1) 2, ∃ a : ℝ, f t x > x + a) →
  (∃ a : ℝ, ∀ x ∈ Set.Icc (-1) 2, f t x > x + a ∧ a ≤ -1/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2263_226352


namespace NUMINAMATH_CALUDE_log_865_between_consecutive_integers_l2263_226389

theorem log_865_between_consecutive_integers :
  ∃ c d : ℤ, c + 1 = d ∧ (c : ℝ) < Real.log 865 / Real.log 10 ∧ Real.log 865 / Real.log 10 < (d : ℝ) ∧ c + d = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_865_between_consecutive_integers_l2263_226389


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_l2263_226339

def has_real_roots (m : ℕ) : Prop :=
  ∃ x : ℝ, x^2 + x - m = 0

theorem contrapositive_real_roots (m : ℕ) :
  (m > 0 → has_real_roots m) ↔ (¬(has_real_roots m) → m ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_l2263_226339


namespace NUMINAMATH_CALUDE_ivans_bird_feeder_feeds_21_l2263_226362

/-- Calculates the number of birds fed weekly by a bird feeder --/
def birds_fed_weekly (feeder_capacity : ℝ) (birds_per_cup : ℝ) (stolen_amount : ℝ) : ℝ :=
  (feeder_capacity - stolen_amount) * birds_per_cup

/-- Theorem: Ivan's bird feeder feeds 21 birds weekly --/
theorem ivans_bird_feeder_feeds_21 :
  birds_fed_weekly 2 14 0.5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ivans_bird_feeder_feeds_21_l2263_226362


namespace NUMINAMATH_CALUDE_complex_multiplication_l2263_226359

theorem complex_multiplication (i : ℂ) (h : i * i = -1) : 
  (-1 + i) * (2 - i) = -1 + 3*i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l2263_226359


namespace NUMINAMATH_CALUDE_teacher_age_survey_is_comprehensive_l2263_226333

-- Define the survey types
inductive SurveyType
  | TelevisionLifespan
  | CityIncome
  | StudentMyopia
  | TeacherAge

-- Define a function to determine if a survey is suitable for comprehensive method
def isSuitableForComprehensiveSurvey (survey : SurveyType) : Prop :=
  match survey with
  | .TelevisionLifespan => false  -- Involves destructiveness, must be sampled
  | .CityIncome => false          -- Large number of people, suitable for sampling
  | .StudentMyopia => false       -- Large number of people, suitable for sampling
  | .TeacherAge => true           -- Small number of people, easy to survey comprehensively

-- Theorem statement
theorem teacher_age_survey_is_comprehensive :
  isSuitableForComprehensiveSurvey SurveyType.TeacherAge = true := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_survey_is_comprehensive_l2263_226333


namespace NUMINAMATH_CALUDE_square_division_theorem_l2263_226324

theorem square_division_theorem (S : ℝ) (h : S > 0) :
  ∀ (squares : Finset (ℝ × ℝ)),
  (∀ (s : ℝ × ℝ), s ∈ squares → s.1 = s.2) →
  squares.card = 9 →
  (∀ (s : ℝ × ℝ), s ∈ squares → s.1 ≤ S ∧ s.2 ≤ S) →
  (∀ (s₁ s₂ : ℝ × ℝ), s₁ ∈ squares → s₂ ∈ squares → s₁ ≠ s₂ → 
    (s₁.1 ≠ s₂.1 ∨ s₁.2 ≠ s₂.2)) →
  ∃ (s₁ s₂ : ℝ × ℝ), s₁ ∈ squares ∧ s₂ ∈ squares ∧ s₁ ≠ s₂ ∧ s₁.1 = s₂.1 := by
sorry


end NUMINAMATH_CALUDE_square_division_theorem_l2263_226324


namespace NUMINAMATH_CALUDE_sum_equals_negative_seven_and_half_l2263_226334

/-- Given that p + 2 = q + 3 = r + 4 = s + 5 = t + 6 = p + q + r + s + t + 10,
    prove that p + q + r + s + t = -7.5 -/
theorem sum_equals_negative_seven_and_half
  (p q r s t : ℚ)
  (h : p + 2 = q + 3 ∧ 
       q + 3 = r + 4 ∧ 
       r + 4 = s + 5 ∧ 
       s + 5 = t + 6 ∧ 
       t + 6 = p + q + r + s + t + 10) :
  p + q + r + s + t = -7.5 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_negative_seven_and_half_l2263_226334


namespace NUMINAMATH_CALUDE_expansion_properties_l2263_226357

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The expansion of (x^4 + 1/x)^n -/
def expansion (n : ℕ) (x : ℝ) : ℝ := sorry

theorem expansion_properties (n : ℕ) :
  (binomial n 2 - binomial n 1 = 35) →
  (n = 10 ∧ 
   ∃ (c : ℝ), c = (expansion n 1) ∧ c = 45) := by sorry

end NUMINAMATH_CALUDE_expansion_properties_l2263_226357


namespace NUMINAMATH_CALUDE_no_valid_sequence_for_certain_n_l2263_226388

/-- A sequence where each number from 1 to n appears twice, 
    and the second occurrence of each number r is r positions after its first occurrence -/
def ValidSequence (n : ℕ) (seq : List ℕ) : Prop :=
  (seq.length = 2 * n) ∧
  (∀ r ∈ Finset.range n, 
    ∃ i j, seq.nthLe i (by sorry) = r + 1 ∧ 
           seq.nthLe j (by sorry) = r + 1 ∧ 
           j = i + (r + 1))

theorem no_valid_sequence_for_certain_n (n : ℕ) :
  (∃ seq : List ℕ, ValidSequence n seq) → 
  (n % 4 ≠ 2 ∧ n % 4 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_sequence_for_certain_n_l2263_226388


namespace NUMINAMATH_CALUDE_xy_equals_twelve_l2263_226396

theorem xy_equals_twelve (x y : ℝ) 
  (h1 : x * (x + y) = x^2 + 12) 
  (h2 : x - y = 3) : 
  x * y = 12 := by
sorry

end NUMINAMATH_CALUDE_xy_equals_twelve_l2263_226396


namespace NUMINAMATH_CALUDE_complex_coordinate_l2263_226349

theorem complex_coordinate (z : ℂ) (h : Complex.I * z = 2 + 4 * Complex.I) : 
  z = 4 - 2 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_coordinate_l2263_226349


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2263_226374

theorem other_root_of_quadratic (c : ℝ) : 
  (3^2 - 5*3 + c = 0) → 
  (∃ x : ℝ, x ≠ 3 ∧ x^2 - 5*x + c = 0 ∧ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2263_226374


namespace NUMINAMATH_CALUDE_horner_v4_value_l2263_226313

/-- The polynomial f(x) = 12 + 35x - 8x^2 + 79x^3 + 6x^4 + 5x^5 + 3x^6 -/
def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

/-- The fourth intermediate value in Horner's method for polynomial f -/
def v4 (x : ℝ) : ℝ := (((3*x + 5)*x + 6)*x + 79)*x - 8

/-- Theorem: The value of v4 for f(x) at x = -4 is 220 -/
theorem horner_v4_value : v4 (-4) = 220 := by sorry

end NUMINAMATH_CALUDE_horner_v4_value_l2263_226313


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2263_226358

open Set Real

theorem intersection_of_M_and_N :
  let M : Set ℝ := {x | x^2 < 3*x}
  let N : Set ℝ := {x | log x < 0}
  M ∩ N = Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2263_226358


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_75_factorial_l2263_226322

theorem last_two_nonzero_digits_75_factorial (n : ℕ) : n = 75 → 
  ∃ k : ℕ, n.factorial = 100 * k + 76 ∧ k % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_75_factorial_l2263_226322


namespace NUMINAMATH_CALUDE_central_angle_for_arc_length_l2263_226340

/-- For a circle with radius r and an arc length of 3/2r, the corresponding central angle is 3/2 radians. -/
theorem central_angle_for_arc_length (r : ℝ) (h : r > 0) : 
  let arc_length := (3/2) * r
  let central_angle := arc_length / r
  central_angle = 3/2 := by sorry

end NUMINAMATH_CALUDE_central_angle_for_arc_length_l2263_226340


namespace NUMINAMATH_CALUDE_rowing_speed_in_still_water_l2263_226337

/-- The speed of a man rowing a boat in still water, given his downstream performance and current speed -/
theorem rowing_speed_in_still_water 
  (distance : Real) 
  (time : Real) 
  (current_speed : Real) : 
  (distance / 1000) / (time / 3600) - current_speed = 22 :=
by
  -- Assuming:
  -- distance = 80 (meters)
  -- time = 11.519078473722104 (seconds)
  -- current_speed = 3 (km/h)
  sorry

#check rowing_speed_in_still_water


end NUMINAMATH_CALUDE_rowing_speed_in_still_water_l2263_226337


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l2263_226360

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

theorem even_function_implies_a_zero :
  (∀ x, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l2263_226360


namespace NUMINAMATH_CALUDE_trig_fraction_equality_l2263_226392

theorem trig_fraction_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ x : ℝ, (Real.sin x)^4 / a + (Real.cos x)^4 / b = 1 / (a + b)) :
  ∀ x : ℝ, (Real.sin x)^8 / a^3 + (Real.cos x)^8 / b^3 = 1 / (a + b)^3 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_equality_l2263_226392


namespace NUMINAMATH_CALUDE_friend_p_distance_at_meeting_l2263_226397

-- Define the trail length
def trail_length : ℝ := 22

-- Define the speed ratio between Friend P and Friend Q
def speed_ratio : ℝ := 1.2

-- Theorem statement
theorem friend_p_distance_at_meeting :
  let v : ℝ := trail_length / (speed_ratio + 1)  -- Friend Q's speed
  let t : ℝ := trail_length / (v * (speed_ratio + 1))  -- Time to meet
  speed_ratio * v * t = 12 := by
  sorry

end NUMINAMATH_CALUDE_friend_p_distance_at_meeting_l2263_226397


namespace NUMINAMATH_CALUDE_no_common_divisor_l2263_226308

theorem no_common_divisor (a b n : ℕ) 
  (ha : a > 1) 
  (hb : b > 1)
  (hn : n > 0)
  (div_a : a ∣ (2^n - 1))
  (div_b : b ∣ (2^n + 1)) :
  ¬∃ k : ℕ, (a ∣ (2^k + 1)) ∧ (b ∣ (2^k - 1)) :=
sorry

end NUMINAMATH_CALUDE_no_common_divisor_l2263_226308


namespace NUMINAMATH_CALUDE_q_round_time_l2263_226301

/-- The time it takes for two runners to meet at the starting point again -/
def meeting_time : ℕ := 2772

/-- The time it takes for runner P to complete one round -/
def p_round_time : ℕ := 252

/-- Theorem stating that under given conditions, runner Q takes 2772 seconds to complete a round -/
theorem q_round_time : ∀ (q_round_time : ℕ), 
  (meeting_time % p_round_time = 0) →
  (meeting_time % q_round_time = 0) →
  (meeting_time / p_round_time ≠ meeting_time / q_round_time) →
  q_round_time = meeting_time :=
by sorry

end NUMINAMATH_CALUDE_q_round_time_l2263_226301


namespace NUMINAMATH_CALUDE_division_with_remainder_l2263_226342

theorem division_with_remainder (dividend quotient divisor remainder : ℕ) : 
  dividend = 76 → 
  quotient = 4 → 
  divisor = 17 → 
  dividend = divisor * quotient + remainder → 
  remainder = 8 := by
sorry

end NUMINAMATH_CALUDE_division_with_remainder_l2263_226342


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_approximate_roots_l2263_226307

/-- The quadratic equation √3x² + √17x - √6 = 0 has two real roots -/
theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ),
  Real.sqrt 3 * x₁^2 + Real.sqrt 17 * x₁ - Real.sqrt 6 = 0 ∧
  Real.sqrt 3 * x₂^2 + Real.sqrt 17 * x₂ - Real.sqrt 6 = 0 ∧
  x₁ ≠ x₂ :=
by sorry

/-- The roots of the equation √3x² + √17x - √6 = 0 are approximately 0.492 and -2.873 -/
theorem approximate_roots : ∃ (x₁ x₂ : ℝ),
  Real.sqrt 3 * x₁^2 + Real.sqrt 17 * x₁ - Real.sqrt 6 = 0 ∧
  Real.sqrt 3 * x₂^2 + Real.sqrt 17 * x₂ - Real.sqrt 6 = 0 ∧
  abs (x₁ - 0.492) < 0.0005 ∧
  abs (x₂ + 2.873) < 0.0005 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_approximate_roots_l2263_226307


namespace NUMINAMATH_CALUDE_max_sum_of_sides_l2263_226321

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  (2 * t.c - t.b) / t.a = (Real.cos t.B) / (Real.cos t.A)

def side_a_condition (t : Triangle) : Prop :=
  t.a = 2 * Real.sqrt 5

-- Theorem statement
theorem max_sum_of_sides (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : side_a_condition t) : 
  ∃ (max : Real), ∀ (t' : Triangle), 
    satisfies_condition t' → side_a_condition t' → 
    t'.b + t'.c ≤ max ∧ 
    ∃ (t'' : Triangle), satisfies_condition t'' ∧ side_a_condition t'' ∧ t''.b + t''.c = max ∧
    max = 4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_sides_l2263_226321


namespace NUMINAMATH_CALUDE_sin_160_equals_sin_20_l2263_226325

theorem sin_160_equals_sin_20 : Real.sin (160 * π / 180) = Real.sin (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_160_equals_sin_20_l2263_226325


namespace NUMINAMATH_CALUDE_jerrys_debt_problem_jerrys_total_debt_l2263_226387

/-- Jerry's debt payment problem -/
theorem jerrys_debt_problem (payment_two_months_ago : ℕ) 
                            (payment_increase : ℕ) 
                            (remaining_debt : ℕ) : ℕ :=
  let payment_last_month := payment_two_months_ago + payment_increase
  let total_paid := payment_two_months_ago + payment_last_month
  let total_debt := total_paid + remaining_debt
  total_debt

/-- Proof of Jerry's total debt -/
theorem jerrys_total_debt : jerrys_debt_problem 12 3 23 = 50 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_debt_problem_jerrys_total_debt_l2263_226387


namespace NUMINAMATH_CALUDE_cos_graph_transformation_l2263_226384

theorem cos_graph_transformation (x : ℝ) :
  let original_point := (x, Real.cos x)
  let transformed_point := (4 * x, Real.cos (x / 4))
  transformed_point.2 = original_point.2 := by
sorry

end NUMINAMATH_CALUDE_cos_graph_transformation_l2263_226384


namespace NUMINAMATH_CALUDE_right_triangle_area_l2263_226368

theorem right_triangle_area (a b c : ℝ) (ha : a = 15) (hb : b = 36) (hc : c = 39) :
  a^2 + b^2 = c^2 ∧ (1/2 * a * b = 270) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2263_226368


namespace NUMINAMATH_CALUDE_angle_bisector_median_inequality_l2263_226332

variable (a b c : ℝ)
variable (s : ℝ)
variable (f₁ f₂ s₃ : ℝ)

/-- Given a triangle with sides a, b, c, semiperimeter s, 
    angle bisectors f₁ and f₂, and median s₃, 
    prove that f₁ + f₂ + s₃ ≤ √3 * s -/
theorem angle_bisector_median_inequality 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : s = (a + b + c) / 2)
  (h_f₁ : f₁^2 = (b * c * ((b + c)^2 - a^2)) / (b + c)^2)
  (h_f₂ : f₂^2 = (a * b * ((a + b)^2 - c^2)) / (a + b)^2)
  (h_s₃ : (2 * s₃)^2 = 2 * a^2 + 2 * c^2 - b^2) :
  f₁ + f₂ + s₃ ≤ Real.sqrt 3 * s :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_median_inequality_l2263_226332


namespace NUMINAMATH_CALUDE_bus_speed_increase_l2263_226329

/-- The speed increase of a bus per hour, given initial speed and total distance traveled. -/
theorem bus_speed_increase 
  (S₀ : ℝ) 
  (total_distance : ℝ) 
  (x : ℝ) 
  (h1 : S₀ = 35) 
  (h2 : total_distance = 552) 
  (h3 : total_distance = S₀ * 12 + x * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11)) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_bus_speed_increase_l2263_226329


namespace NUMINAMATH_CALUDE_school_population_l2263_226351

/-- Given a school with boys, girls, and teachers, prove that the total number
    of people is 41b/32 when there are 4 times as many boys as girls and 8 times
    as many girls as teachers. -/
theorem school_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 8 * t) :
  b + g + t = (41 * b) / 32 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l2263_226351


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2263_226327

theorem ratio_x_to_y (x y : ℚ) (h : (12 * x - 5 * y) / (17 * x - 3 * y) = 5 / 7) :
  x / y = -20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2263_226327


namespace NUMINAMATH_CALUDE_apple_pear_basket_weights_l2263_226314

/-- Given the conditions of the apple and pear basket problem, prove the weights of individual baskets. -/
theorem apple_pear_basket_weights :
  ∀ (apple_weight pear_weight : ℕ),
  -- Total weight of all baskets is 692 kg
  12 * apple_weight + 14 * pear_weight = 692 →
  -- Weight of pear basket is 10 kg less than apple basket
  pear_weight = apple_weight - 10 →
  -- Prove that apple_weight is 32 kg and pear_weight is 22 kg
  apple_weight = 32 ∧ pear_weight = 22 := by
  sorry

end NUMINAMATH_CALUDE_apple_pear_basket_weights_l2263_226314


namespace NUMINAMATH_CALUDE_total_ninja_stars_l2263_226353

/-- The number of ninja throwing stars each person has -/
structure NinjaStars where
  eric : ℕ
  chad : ℕ
  mark : ℕ
  jennifer : ℕ

/-- The initial distribution of ninja throwing stars -/
def initial_distribution : NinjaStars where
  eric := 4
  chad := 4 * 2
  mark := 0
  jennifer := 0

/-- The final distribution of ninja throwing stars after all transactions -/
def final_distribution : NinjaStars :=
  let chad_after_selling := initial_distribution.chad - 2
  let chad_final := chad_after_selling + 3
  let mark_jennifer_share := 10 / 2
  { eric := initial_distribution.eric,
    chad := chad_final,
    mark := mark_jennifer_share,
    jennifer := mark_jennifer_share }

/-- The theorem stating the total number of ninja throwing stars -/
theorem total_ninja_stars :
  final_distribution.eric +
  final_distribution.chad +
  final_distribution.mark +
  final_distribution.jennifer = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_total_ninja_stars_l2263_226353


namespace NUMINAMATH_CALUDE_factorization_equality_l2263_226366

theorem factorization_equality (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2263_226366


namespace NUMINAMATH_CALUDE_sum_of_digits_of_X_squared_l2263_226346

-- Define the number with 8 repeated ones
def X : ℕ := 11111111

-- Define a function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_X_squared : sum_of_digits (X^2) = 64 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_X_squared_l2263_226346


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l2263_226379

/-- The line equation mx + y - m - 1 = 0 passes through the point (1, 1) for all real m -/
theorem fixed_point_of_line (m : ℝ) : m * 1 + 1 - m - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l2263_226379


namespace NUMINAMATH_CALUDE_function_properties_l2263_226338

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) :
  (∀ x, f (1 + x) = f (x - 1)) →
  (∀ x, f (1 - x) = -f (x - 1)) →
  (is_periodic f 2 ∧ is_odd f) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2263_226338


namespace NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l2263_226373

theorem smallest_divisible_by_one_to_ten : ∃ n : ℕ,
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧
  (∀ m : ℕ, m < n → ∃ j : ℕ, 1 ≤ j ∧ j ≤ 10 ∧ ¬(j ∣ m)) ∧
  n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_one_to_ten_l2263_226373


namespace NUMINAMATH_CALUDE_binary_multiplication_division_equality_l2263_226341

/-- Represents a binary number as a list of booleans, with the least significant bit first. -/
def BinaryNumber := List Bool

/-- Converts a natural number to its binary representation. -/
def toBinary (n : Nat) : BinaryNumber :=
  if n = 0 then [] else (n % 2 = 1) :: toBinary (n / 2)

/-- Converts a binary number to its decimal representation. -/
def toDecimal (b : BinaryNumber) : Nat :=
  b.foldl (fun acc digit => 2 * acc + if digit then 1 else 0) 0

/-- Multiplies two binary numbers. -/
def binaryMultiply (a b : BinaryNumber) : BinaryNumber :=
  toBinary (toDecimal a * toDecimal b)

/-- Divides a binary number by another binary number. -/
def binaryDivide (a b : BinaryNumber) : BinaryNumber :=
  toBinary (toDecimal a / toDecimal b)

theorem binary_multiplication_division_equality :
  let a := [false, true, false, true, true, false, true]  -- 1011010₂
  let b := [false, false, true, false, true, false, true] -- 1010100₂
  let c := [false, true, false, true]                     -- 1010₂
  binaryDivide (binaryMultiply a b) c = 
    [false, false, true, false, false, true, true, true, false, true] -- 1011100100₂
  := by sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_equality_l2263_226341


namespace NUMINAMATH_CALUDE_count_divisors_of_twenty_divisible_by_five_l2263_226318

theorem count_divisors_of_twenty_divisible_by_five : 
  let a : ℕ → Prop := λ n => 
    n > 0 ∧ 5 ∣ n ∧ n ∣ 20
  (Finset.filter a (Finset.range 21)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_divisors_of_twenty_divisible_by_five_l2263_226318


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2263_226304

theorem smallest_solution_of_equation (x : ℝ) :
  (((3 * x) / (x - 3)) + ((3 * x^2 - 27) / x) = 15) →
  (x ≥ -1 ∧ (∀ y : ℝ, y < -1 → ((3 * y) / (y - 3)) + ((3 * y^2 - 27) / y) ≠ 15)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2263_226304


namespace NUMINAMATH_CALUDE_tina_remaining_money_l2263_226386

def monthly_income : ℝ := 1000

def june_bonus_rate : ℝ := 0.1
def investment_return_rate : ℝ := 0.05
def tax_rate : ℝ := 0.1

def june_savings_rate : ℝ := 0.25
def july_savings_rate : ℝ := 0.2
def august_savings_rate : ℝ := 0.3

def june_rent : ℝ := 200
def june_groceries : ℝ := 100
def june_book_rate : ℝ := 0.05

def july_rent : ℝ := 250
def july_groceries : ℝ := 150
def july_shoes_rate : ℝ := 0.15

def august_rent : ℝ := 300
def august_groceries : ℝ := 175
def august_misc_rate : ℝ := 0.1

theorem tina_remaining_money :
  let june_income := monthly_income * (1 + june_bonus_rate)
  let june_expenses := june_rent + june_groceries + (june_income * june_book_rate)
  let june_savings := june_income * june_savings_rate
  let june_remaining := june_income - june_savings - june_expenses

  let july_investment_return := june_savings * investment_return_rate
  let july_income := monthly_income + july_investment_return
  let july_expenses := july_rent + july_groceries + (monthly_income * july_shoes_rate)
  let july_savings := july_income * july_savings_rate
  let july_remaining := july_income - july_savings - july_expenses

  let august_investment_return := july_savings * investment_return_rate
  let august_income := monthly_income + august_investment_return
  let august_expenses := august_rent + august_groceries + (monthly_income * august_misc_rate)
  let august_savings := august_income * august_savings_rate
  let august_remaining := august_income - august_savings - august_expenses

  let total_investment_return := july_investment_return + august_investment_return
  let total_tax := total_investment_return * tax_rate
  let total_remaining := june_remaining + july_remaining + august_remaining - total_tax

  total_remaining = 860.7075 := by sorry

end NUMINAMATH_CALUDE_tina_remaining_money_l2263_226386


namespace NUMINAMATH_CALUDE_equal_selection_probability_l2263_226323

/-- Represents the probability of a student being selected -/
def probability_of_selection (n : ℕ) (total : ℕ) : ℚ := n / total

theorem equal_selection_probability 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (eliminated_students : ℕ) 
  (h1 : total_students = 54) 
  (h2 : selected_students = 5) 
  (h3 : eliminated_students = 4) :
  ∀ (student : ℕ), student ≤ total_students → 
    probability_of_selection selected_students total_students = 5 / 54 :=
by sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l2263_226323


namespace NUMINAMATH_CALUDE_average_weight_problem_l2263_226372

/-- The average weight problem -/
theorem average_weight_problem 
  (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84)
  (h2 : (A + B + C + D) / 4 = 80)
  (h3 : E = D + 8)
  (h4 : A = 80) :
  (B + C + D + E) / 4 = 79 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l2263_226372


namespace NUMINAMATH_CALUDE_worker_C_post_tax_income_l2263_226380

-- Define worker types
inductive Worker : Type
| A
| B
| C

-- Define survey types
inductive SurveyType : Type
| Basic
| Lifestyle
| Technology

-- Define payment rates for each worker and survey type
def baseRate (w : Worker) : ℚ :=
  match w with
  | Worker.A => 30
  | Worker.B => 25
  | Worker.C => 35

def rateMultiplier (w : Worker) (s : SurveyType) : ℚ :=
  match w, s with
  | Worker.A, SurveyType.Basic => 1
  | Worker.A, SurveyType.Lifestyle => 1.2
  | Worker.A, SurveyType.Technology => 1.5
  | Worker.B, SurveyType.Basic => 1
  | Worker.B, SurveyType.Lifestyle => 1.25
  | Worker.B, SurveyType.Technology => 1.45
  | Worker.C, SurveyType.Basic => 1
  | Worker.C, SurveyType.Lifestyle => 1.15
  | Worker.C, SurveyType.Technology => 1.6

-- Define commission rate for technology surveys
def commissionRate : ℚ := 0.05

-- Define tax rates for each worker
def taxRate (w : Worker) : ℚ :=
  match w with
  | Worker.A => 0.15
  | Worker.B => 0.18
  | Worker.C => 0.20

-- Define number of surveys completed by each worker
def surveysCompleted (w : Worker) (s : SurveyType) : ℕ :=
  match w, s with
  | Worker.A, SurveyType.Basic => 80
  | Worker.A, SurveyType.Lifestyle => 50
  | Worker.A, SurveyType.Technology => 35
  | Worker.B, SurveyType.Basic => 90
  | Worker.B, SurveyType.Lifestyle => 45
  | Worker.B, SurveyType.Technology => 40
  | Worker.C, SurveyType.Basic => 70
  | Worker.C, SurveyType.Lifestyle => 40
  | Worker.C, SurveyType.Technology => 60

-- Define health insurance deductions for each worker
def healthInsurance (w : Worker) : ℚ :=
  match w with
  | Worker.A => 200
  | Worker.B => 250
  | Worker.C => 300

-- Calculate earnings for a worker
def earnings (w : Worker) : ℚ :=
  let basicEarnings := (baseRate w) * (surveysCompleted w SurveyType.Basic)
  let lifestyleEarnings := (baseRate w) * (rateMultiplier w SurveyType.Lifestyle) * (surveysCompleted w SurveyType.Lifestyle)
  let techEarnings := (baseRate w) * (rateMultiplier w SurveyType.Technology) * (surveysCompleted w SurveyType.Technology)
  let techCommission := techEarnings * commissionRate
  basicEarnings + lifestyleEarnings + techEarnings + techCommission

-- Calculate post-tax income for a worker
def postTaxIncome (w : Worker) : ℚ :=
  let grossEarnings := earnings w
  let tax := grossEarnings * (taxRate w)
  grossEarnings - tax - (healthInsurance w)

-- Theorem to prove
theorem worker_C_post_tax_income :
  postTaxIncome Worker.C = 5770.40 :=
by sorry

end NUMINAMATH_CALUDE_worker_C_post_tax_income_l2263_226380


namespace NUMINAMATH_CALUDE_board_cutting_l2263_226328

theorem board_cutting (total_length : ℝ) (difference : ℝ) (shorter_piece : ℝ) : 
  total_length = 120 →
  shorter_piece + (2 * shorter_piece + difference) = total_length →
  difference = 15 →
  shorter_piece = 35 := by
sorry

end NUMINAMATH_CALUDE_board_cutting_l2263_226328


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2263_226335

theorem quadratic_equation_roots : 
  let equation := fun (x : ℂ) => x^2 + x + 2
  ∃ (r₁ r₂ : ℂ), r₁ = (-1 + Complex.I * Real.sqrt 7) / 2 ∧ 
                  r₂ = (-1 - Complex.I * Real.sqrt 7) / 2 ∧ 
                  equation r₁ = 0 ∧ 
                  equation r₂ = 0 ∧
                  ∀ (x : ℂ), equation x = 0 → x = r₁ ∨ x = r₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2263_226335


namespace NUMINAMATH_CALUDE_inverse_function_theorem_l2263_226347

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2 - 1

theorem inverse_function_theorem (x : ℝ) (h : x > 0) :
  f (f_inv x) = x ∧ f_inv (f x) = x :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_theorem_l2263_226347


namespace NUMINAMATH_CALUDE_book_pages_count_l2263_226356

def days_in_week : ℕ := 7
def first_period : ℕ := 4
def second_period : ℕ := 2
def last_day : ℕ := 1
def pages_per_day_first_period : ℕ := 42
def pages_per_day_second_period : ℕ := 50
def pages_last_day : ℕ := 30

theorem book_pages_count :
  first_period * pages_per_day_first_period +
  second_period * pages_per_day_second_period +
  pages_last_day = 298 :=
by sorry

end NUMINAMATH_CALUDE_book_pages_count_l2263_226356


namespace NUMINAMATH_CALUDE_value_of_b_l2263_226317

theorem value_of_b : ∀ b : ℕ, (5 ^ 5 * b = 3 * 15 ^ 5) ∧ (b = 9 ^ 3) → b = 729 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l2263_226317


namespace NUMINAMATH_CALUDE_prop_2_prop_4_l2263_226370

-- Define real numbers a, b, and c
variable (a b c : ℝ)

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Proposition 2
theorem prop_2 : IsIrrational (a + 5) ↔ IsIrrational a := by sorry

-- Proposition 4
theorem prop_4 : a < 3 → a < 5 := by sorry

end NUMINAMATH_CALUDE_prop_2_prop_4_l2263_226370


namespace NUMINAMATH_CALUDE_no_divisor_of_form_24k_plus_20_l2263_226393

theorem no_divisor_of_form_24k_plus_20 (n : ℕ) : ¬ ∃ (k : ℕ), (24 * k + 20) ∣ (3^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_divisor_of_form_24k_plus_20_l2263_226393


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l2263_226305

/-- Given two parallel vectors p and q, prove that their sum has magnitude √13 -/
theorem parallel_vectors_sum_magnitude (p q : ℝ × ℝ) (h_parallel : p.1 * q.2 = p.2 * q.1) :
  p = (2, -3) → q.2 = 6 → ‖p + q‖ = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l2263_226305


namespace NUMINAMATH_CALUDE_curve_properties_l2263_226303

-- Define the curve equation
def curve_equation (x y t : ℝ) : Prop :=
  x^2 / (4 - t) + y^2 / (t - 1) = 1

-- Define what it means for the curve to be a hyperbola
def is_hyperbola (t : ℝ) : Prop :=
  t < 1 ∨ t > 4

-- Define what it means for the curve to be an ellipse with foci on the x-axis
def is_ellipse_x_foci (t : ℝ) : Prop :=
  1 < t ∧ t < 5/2

-- State the theorem
theorem curve_properties :
  ∀ t : ℝ,
    (∀ x y : ℝ, curve_equation x y t → is_hyperbola t) ∧
    (∀ x y : ℝ, curve_equation x y t → is_ellipse_x_foci t) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l2263_226303


namespace NUMINAMATH_CALUDE_passing_percentage_is_33_percent_l2263_226312

def total_marks : ℕ := 400
def obtained_marks : ℕ := 92
def failing_margin : ℕ := 40

theorem passing_percentage_is_33_percent :
  (obtained_marks + failing_margin) / total_marks * 100 = 33 := by sorry

end NUMINAMATH_CALUDE_passing_percentage_is_33_percent_l2263_226312


namespace NUMINAMATH_CALUDE_smallest_mu_l2263_226315

theorem smallest_mu : 
  ∃ μ : ℝ, (∀ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≤ a*b + μ*b*c + c*d) ∧ 
  (∀ μ' : ℝ, (∀ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≤ a*b + μ'*b*c + c*d) → μ' ≥ μ) ∧
  μ = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_mu_l2263_226315


namespace NUMINAMATH_CALUDE_max_area_height_l2263_226311

/-- A right trapezoid with an acute angle of 30° and perimeter 6 -/
structure RightTrapezoid where
  height : ℝ
  sumOfBases : ℝ
  acuteAngle : ℝ
  perimeter : ℝ
  area : ℝ
  acuteAngle_eq : acuteAngle = π / 6
  perimeter_eq : perimeter = 6
  area_eq : area = (3 * sumOfBases * height) / 2
  perimeter_constraint : sumOfBases + 3 * height = 6

/-- The height that maximizes the area of the right trapezoid is 1 -/
theorem max_area_height (t : RightTrapezoid) : 
  t.area ≤ (3 : ℝ) / 2 ∧ (t.area = (3 : ℝ) / 2 ↔ t.height = 1) :=
sorry

end NUMINAMATH_CALUDE_max_area_height_l2263_226311
