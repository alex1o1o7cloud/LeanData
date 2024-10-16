import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l4019_401975

theorem quadratic_equation_solutions (x : ℝ) :
  (x^2 + 2*x + 1 = 4) ↔ (x = 1 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l4019_401975


namespace NUMINAMATH_CALUDE_nested_bracket_value_l4019_401983

-- Define the operation [a,b,c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- State the theorem
theorem nested_bracket_value :
  bracket (bracket 80 40 120) (bracket 4 2 6) (bracket 50 25 75) = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_bracket_value_l4019_401983


namespace NUMINAMATH_CALUDE_eighteen_mangoes_yield_fortyeight_lassis_l4019_401955

/-- Given that 3 mangoes make 8 lassis, this function calculates
    the number of lassis that can be made from a given number of mangoes. -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  (mangoes * 8) / 3

/-- Theorem stating that 18 mangoes will yield 48 lassis, 
    given the ratio of 8 lassis to 3 mangoes. -/
theorem eighteen_mangoes_yield_fortyeight_lassis :
  lassis_from_mangoes 18 = 48 := by
  sorry

#eval lassis_from_mangoes 18

end NUMINAMATH_CALUDE_eighteen_mangoes_yield_fortyeight_lassis_l4019_401955


namespace NUMINAMATH_CALUDE_intersection_M_N_l4019_401917

def M : Set ℝ := {-4, -3, -2, -1, 0, 1}
def N : Set ℝ := {x : ℝ | x^2 + 3*x < 0}

theorem intersection_M_N : M ∩ N = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4019_401917


namespace NUMINAMATH_CALUDE_D_300_l4019_401977

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where the order matters. -/
def D (n : ℕ+) : ℕ := sorry

/-- The prime factorization of 300 -/
def primeFactor300 : List ℕ+ := [2, 2, 3, 5, 5]

/-- Theorem: The number of ways to write 300 as a product of integers greater than 1, where the order matters, is 35. -/
theorem D_300 : D 300 = 35 := by sorry

end NUMINAMATH_CALUDE_D_300_l4019_401977


namespace NUMINAMATH_CALUDE_ellipse_theorem_l4019_401968

-- Define the ellipse M
def ellipse_M (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1 ∧ a > 0

-- Define the right focus F
def right_focus (a c : ℝ) : Prop :=
  c > 0 ∧ a^2 = 3 + c^2

-- Define the symmetric property
def symmetric_property (c : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse_M (2*c) (-x+2*c) y ∧ x^2 + y^2 = 0

-- Main theorem
theorem ellipse_theorem (a c : ℝ) 
  (h1 : ellipse_M a 0 0)
  (h2 : right_focus a c)
  (h3 : symmetric_property c) :
  a^2 = 4 ∧ c = 1 ∧
  ∀ (k x₁ y₁ x₂ y₂ : ℝ),
    (ellipse_M a x₁ y₁ ∧ ellipse_M a x₂ y₂ ∧
     y₁ = k*(x₁ - 4) ∧ y₂ = k*(x₂ - 4) ∧ k ≠ 0) →
    ∃ (t : ℝ), t*(y₁ + y₂) + x₁ = 1 ∧ t*(x₁ - x₂) + y₁ = 0 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l4019_401968


namespace NUMINAMATH_CALUDE_sqrt_ab_max_value_l4019_401991

theorem sqrt_ab_max_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  ∃ (m : ℝ), m = 1/2 ∧ ∀ x, x = Real.sqrt (a * b) → x ≤ m :=
sorry

end NUMINAMATH_CALUDE_sqrt_ab_max_value_l4019_401991


namespace NUMINAMATH_CALUDE_jakes_weight_l4019_401929

theorem jakes_weight (jake sister brother : ℝ) : 
  (0.8 * jake = 2 * sister) →
  (jake + sister = 168) →
  (brother = 1.25 * (jake + sister)) →
  (jake + sister + brother = 221) →
  jake = 120 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l4019_401929


namespace NUMINAMATH_CALUDE_sally_next_birthday_l4019_401970

theorem sally_next_birthday (adam mary sally danielle : ℝ) 
  (h1 : adam = 1.3 * mary)
  (h2 : mary = 0.75 * sally)
  (h3 : sally = 0.8 * danielle)
  (h4 : adam + mary + sally + danielle = 60) :
  ⌈sally⌉ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sally_next_birthday_l4019_401970


namespace NUMINAMATH_CALUDE_max_students_social_practice_l4019_401927

theorem max_students_social_practice (max_fund car_rental per_student_cost : ℕ) 
  (h1 : max_fund = 800)
  (h2 : car_rental = 300)
  (h3 : per_student_cost = 15) :
  ∃ (max_students : ℕ), 
    max_students = 33 ∧ 
    max_students * per_student_cost + car_rental ≤ max_fund ∧
    ∀ (n : ℕ), n * per_student_cost + car_rental ≤ max_fund → n ≤ max_students :=
sorry

end NUMINAMATH_CALUDE_max_students_social_practice_l4019_401927


namespace NUMINAMATH_CALUDE_exists_circle_with_n_lattice_points_l4019_401994

/-- A point with integer coordinates in the plane -/
def LatticePoint := ℤ × ℤ

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of lattice points on the circumference of a circle -/
def latticePointsOnCircle (c : Circle) : ℕ :=
  sorry

/-- For every natural number n, there exists a circle with exactly n lattice points on its circumference -/
theorem exists_circle_with_n_lattice_points (n : ℕ) :
  ∃ c : Circle, latticePointsOnCircle c = n :=
sorry

end NUMINAMATH_CALUDE_exists_circle_with_n_lattice_points_l4019_401994


namespace NUMINAMATH_CALUDE_bacteria_growth_30_minutes_l4019_401976

/-- The number of bacteria after a given number of 2-minute intervals, 
    given an initial population and a tripling growth rate every 2 minutes. -/
def bacteria_population (initial_population : ℕ) (intervals : ℕ) : ℕ :=
  initial_population * (3 ^ intervals)

/-- Theorem stating that after 15 intervals (30 minutes), 
    an initial population of 30 bacteria will grow to 430467210. -/
theorem bacteria_growth_30_minutes :
  bacteria_population 30 15 = 430467210 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_30_minutes_l4019_401976


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4019_401964

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < -1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4019_401964


namespace NUMINAMATH_CALUDE_dave_spent_102_l4019_401967

/-- The amount Dave spent on books -/
def dave_spent (animal_books outer_space_books train_books cost_per_book : ℕ) : ℕ :=
  (animal_books + outer_space_books + train_books) * cost_per_book

/-- Theorem stating that Dave spent $102 on books -/
theorem dave_spent_102 :
  dave_spent 8 6 3 6 = 102 := by
  sorry

end NUMINAMATH_CALUDE_dave_spent_102_l4019_401967


namespace NUMINAMATH_CALUDE_work_completion_time_l4019_401978

/-- The number of days it takes for A to complete the work alone -/
def days_A : ℝ := 6

/-- The number of days it takes for B to complete the work alone -/
def days_B : ℝ := 12

/-- The number of days it takes for A and B to complete the work together -/
def days_AB : ℝ := 4

/-- Theorem stating that given the time for B and the time for A and B together, 
    we can determine the time for A alone -/
theorem work_completion_time : 
  (1 / days_A + 1 / days_B = 1 / days_AB) → days_A = 6 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l4019_401978


namespace NUMINAMATH_CALUDE_smallest_covering_radius_l4019_401990

theorem smallest_covering_radius :
  let r : ℝ := Real.sqrt 3 / 2
  ∀ s : ℝ, s < r → ¬(∃ (c₁ c₂ c₃ : ℝ × ℝ),
    (∀ p : ℝ × ℝ, Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≤ 1 →
      Real.sqrt ((p.1 - c₁.1)^2 + (p.2 - c₁.2)^2) ≤ s ∨
      Real.sqrt ((p.1 - c₂.1)^2 + (p.2 - c₂.2)^2) ≤ s ∨
      Real.sqrt ((p.1 - c₃.1)^2 + (p.2 - c₃.2)^2) ≤ s)) ∧
  ∃ (c₁ c₂ c₃ : ℝ × ℝ),
    (∀ p : ℝ × ℝ, Real.sqrt ((p.1 - 0)^2 + (p.2 - 0)^2) ≤ 1 →
      Real.sqrt ((p.1 - c₁.1)^2 + (p.2 - c₁.2)^2) ≤ r ∨
      Real.sqrt ((p.1 - c₂.1)^2 + (p.2 - c₂.2)^2) ≤ r ∨
      Real.sqrt ((p.1 - c₃.1)^2 + (p.2 - c₃.2)^2) ≤ r) :=
by sorry

end NUMINAMATH_CALUDE_smallest_covering_radius_l4019_401990


namespace NUMINAMATH_CALUDE_least_positive_integer_t_l4019_401924

theorem least_positive_integer_t : ∃ (t : ℕ+), 
  (∀ (x y : ℕ+), (x^2 + y^2)^2 + 2*t*x*(x^2 + y^2) = t^2*y^2 → t ≥ 25) ∧ 
  (∃ (x y : ℕ+), (x^2 + y^2)^2 + 2*25*x*(x^2 + y^2) = 25^2*y^2) := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_t_l4019_401924


namespace NUMINAMATH_CALUDE_inequality_solution_l4019_401938

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 - a*x - 6*a^2 > 0

-- Define the solution set
def solution_set (x₁ x₂ : ℝ) : Set ℝ := {x | x < x₁ ∨ x > x₂}

theorem inequality_solution (a : ℝ) (x₁ x₂ : ℝ) :
  a < 0 →
  (∀ x, inequality x a ↔ x ∈ solution_set x₁ x₂) →
  x₂ - x₁ = 5 * Real.sqrt 2 →
  a = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l4019_401938


namespace NUMINAMATH_CALUDE_points_three_units_from_origin_l4019_401936

theorem points_three_units_from_origin (a : ℝ) : 
  |a| = 3 → (a = 3 ∨ a = -3) := by
sorry

end NUMINAMATH_CALUDE_points_three_units_from_origin_l4019_401936


namespace NUMINAMATH_CALUDE_lattice_points_on_curve_l4019_401985

theorem lattice_points_on_curve : 
  ∃! (points : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ points ↔ x^2 - y^2 = 15) ∧ 
    points.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_curve_l4019_401985


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l4019_401986

/-- Given a cone with base radius 6 and volume 30π, its lateral surface area is 39π. -/
theorem cone_lateral_surface_area (r h l : ℝ) : 
  r = 6 → 
  (1 / 3) * π * r^2 * h = 30 * π → 
  l^2 = r^2 + h^2 → 
  π * r * l = 39 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l4019_401986


namespace NUMINAMATH_CALUDE_arithmetic_progression_product_divisible_l4019_401911

/-- An arithmetic progression of natural numbers -/
def arithmeticProgression (a : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a + i * d)

/-- The product of all elements in a list -/
def listProduct (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

theorem arithmetic_progression_product_divisible (a : ℕ) :
  (listProduct (arithmeticProgression a 11 10)) % (Nat.factorial 10) = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_product_divisible_l4019_401911


namespace NUMINAMATH_CALUDE_servant_worked_months_l4019_401982

def yearly_salary : ℚ := 90
def turban_value : ℚ := 50
def received_cash : ℚ := 55

def total_yearly_salary : ℚ := yearly_salary + turban_value
def monthly_salary : ℚ := total_yearly_salary / 12
def total_received : ℚ := received_cash + turban_value

theorem servant_worked_months : 
  ∃ (months : ℚ), months * monthly_salary = total_received ∧ months = 9 := by
  sorry

end NUMINAMATH_CALUDE_servant_worked_months_l4019_401982


namespace NUMINAMATH_CALUDE_expression_one_proof_l4019_401918

theorem expression_one_proof : 
  ((-5/8) / (14/3) * (-16/5) / (-6/7)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_expression_one_proof_l4019_401918


namespace NUMINAMATH_CALUDE_simplify_expression_l4019_401954

theorem simplify_expression (a b : ℝ) : (1:ℝ)*(2*b)*(3*a)*(4*a^2)*(5*b^2)*(6*a^3) = 720*a^6*b^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4019_401954


namespace NUMINAMATH_CALUDE_clock_angle_theorem_l4019_401946

/-- The angle in radians through which the minute hand of a clock turns from 1:00 to 3:20 -/
def clock_angle_radians : ℝ := sorry

/-- The angle in degrees that the minute hand turns per minute -/
def minute_hand_degrees_per_minute : ℝ := 6

/-- The time difference in minutes from 1:00 to 3:20 -/
def time_difference_minutes : ℕ := 2 * 60 + 20

theorem clock_angle_theorem : 
  clock_angle_radians = -(minute_hand_degrees_per_minute * time_difference_minutes * (π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_theorem_l4019_401946


namespace NUMINAMATH_CALUDE_cytoplasm_distribution_in_cell_division_l4019_401948

/-- Represents a cell in a diploid organism -/
structure DiploidCell where
  cytoplasm : Set ℝ
  deriving Inhabited

/-- Represents the process of cell division -/
def cell_division (parent : DiploidCell) : DiploidCell × DiploidCell :=
  sorry

/-- The distribution of cytoplasm during cell division is random -/
def is_random_distribution (division : DiploidCell → DiploidCell × DiploidCell) : Prop :=
  sorry

/-- The distribution of cytoplasm during cell division is unequal -/
def is_unequal_distribution (division : DiploidCell → DiploidCell × DiploidCell) : Prop :=
  sorry

/-- Theorem: In diploid organism cells, the distribution of cytoplasm during cell division is random and unequal -/
theorem cytoplasm_distribution_in_cell_division :
  is_random_distribution cell_division ∧ is_unequal_distribution cell_division :=
sorry

end NUMINAMATH_CALUDE_cytoplasm_distribution_in_cell_division_l4019_401948


namespace NUMINAMATH_CALUDE_impossible_lake_system_dima_is_mistaken_l4019_401923

/-- Represents a lake system with a given number of lakes, outgoing rivers per lake, and incoming rivers per lake. -/
structure LakeSystem where
  num_lakes : ℕ
  outgoing_rivers_per_lake : ℕ
  incoming_rivers_per_lake : ℕ

/-- Theorem stating that a non-empty lake system with 3 outgoing and 4 incoming rivers per lake is impossible. -/
theorem impossible_lake_system : ¬∃ (ls : LakeSystem), ls.num_lakes > 0 ∧ ls.outgoing_rivers_per_lake = 3 ∧ ls.incoming_rivers_per_lake = 4 := by
  sorry

/-- Corollary stating that Dima's claim about the lake system in Vrunlandia is incorrect. -/
theorem dima_is_mistaken : ¬∃ (ls : LakeSystem), ls.num_lakes > 0 ∧ ls.outgoing_rivers_per_lake = 3 ∧ ls.incoming_rivers_per_lake = 4 := by
  exact impossible_lake_system

end NUMINAMATH_CALUDE_impossible_lake_system_dima_is_mistaken_l4019_401923


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l4019_401962

-- Define the function f
def f (x : ℝ) (k : ℝ) : ℝ := x^3 - k*x^2 + 3*x - 5

-- State the theorem
theorem f_derivative_at_2 : 
  ∃ k : ℝ, (∀ x, deriv (f · k) x = 3*x^2 - 2*k*x + 3) ∧ deriv (f · k) 2 = k ∧ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l4019_401962


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l4019_401998

theorem modulus_of_complex_number : 
  let z : ℂ := (1 + 3 * Complex.I) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l4019_401998


namespace NUMINAMATH_CALUDE_council_composition_l4019_401963

/-- Represents a member of the council -/
inductive Member
| Knight
| Liar

/-- The total number of council members -/
def total_members : Nat := 101

/-- Proposition that if any member is removed, the majority of remaining members would be liars -/
def majority_liars_if_removed (knights : Nat) (liars : Nat) : Prop :=
  ∀ (m : Member), 
    (m = Member.Knight → liars > (knights + liars - 1) / 2) ∧
    (m = Member.Liar → knights ≤ (knights + liars - 1) / 2)

theorem council_composition :
  ∃ (knights liars : Nat),
    knights + liars = total_members ∧
    majority_liars_if_removed knights liars ∧
    knights = 50 ∧ liars = 51 := by
  sorry

end NUMINAMATH_CALUDE_council_composition_l4019_401963


namespace NUMINAMATH_CALUDE_locus_of_vertex_A_l4019_401984

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define the median CM
def Median (C M : ℝ × ℝ) : Prop := True

-- Define the constant length of CM
def ConstantLength (CM : ℝ) : Prop := True

-- Define the midpoint of BC
def Midpoint (K B C : ℝ × ℝ) : Prop := 
  K.1 = (B.1 + C.1) / 2 ∧ K.2 = (B.2 + C.2) / 2

-- Define a circle
def Circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Theorem statement
theorem locus_of_vertex_A (A B C : ℝ × ℝ) (K : ℝ × ℝ) (CM : ℝ) :
  Triangle A B C →
  Midpoint K B C →
  ConstantLength CM →
  ∀ M, Median C M →
  ∃ center radius, Circle center radius A ∧ 
    center = K ∧ 
    radius = 2 * CM ∧
    ¬(A.1 = B.1 ∧ A.2 = B.2) ∧ 
    ¬(A.1 = C.1 ∧ A.2 = C.2) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_vertex_A_l4019_401984


namespace NUMINAMATH_CALUDE_symmetry_implies_m_and_n_l4019_401945

/-- Two points are symmetric about the x-axis if their x-coordinates are equal and their y-coordinates are opposites -/
def symmetric_about_x_axis (a b : ℝ × ℝ) : Prop :=
  a.1 = b.1 ∧ a.2 = -b.2

/-- The theorem stating that if A(-4, m-3) and B(2n, 1) are symmetric about the x-axis, then m = 2 and n = -2 -/
theorem symmetry_implies_m_and_n (m n : ℝ) :
  symmetric_about_x_axis (-4, m - 3) (2*n, 1) → m = 2 ∧ n = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_m_and_n_l4019_401945


namespace NUMINAMATH_CALUDE_intersection_M_N_l4019_401901

def M : Set ℝ := {-1, 0, 1, 2, 3}
def N : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4019_401901


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_eq_five_l4019_401952

/-- The function f(x) = x³ + ax² + 3x - 9 has an extreme value at x = -3 -/
def has_extreme_value_at_neg_three (a : ℝ) : Prop :=
  let f := fun (x : ℝ) => x^3 + a*x^2 + 3*x - 9
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x - (-3)| < ε → f x ≤ f (-3) ∨ f x ≥ f (-3)

/-- If f(x) = x³ + ax² + 3x - 9 has an extreme value at x = -3, then a = 5 -/
theorem extreme_value_implies_a_eq_five :
  ∀ (a : ℝ), has_extreme_value_at_neg_three a → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_eq_five_l4019_401952


namespace NUMINAMATH_CALUDE_bus_driver_rate_l4019_401931

theorem bus_driver_rate (hours_worked : ℕ) (total_compensation : ℚ) : 
  hours_worked = 50 →
  total_compensation = 920 →
  ∃ (regular_rate : ℚ),
    (40 * regular_rate + (hours_worked - 40) * (1.75 * regular_rate) = total_compensation) ∧
    regular_rate = 16 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_rate_l4019_401931


namespace NUMINAMATH_CALUDE_probability_both_science_questions_l4019_401919

def total_questions : ℕ := 5
def science_questions : ℕ := 3
def humanities_questions : ℕ := 2

theorem probability_both_science_questions :
  let total_outcomes := total_questions * (total_questions - 1)
  let favorable_outcomes := science_questions * (science_questions - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_both_science_questions_l4019_401919


namespace NUMINAMATH_CALUDE_inscribed_box_sphere_radius_l4019_401906

/-- 
Given a rectangular box Q inscribed in a sphere of radius s,
prove that if the surface area of Q is 312 and the sum of the
lengths of its 12 edges is 96, then s = √66.
-/
theorem inscribed_box_sphere_radius (a b c s : ℝ) : 
  a > 0 → b > 0 → c > 0 → s > 0 →
  4 * (a + b + c) = 96 →
  2 * (a * b + b * c + a * c) = 312 →
  (2 * s)^2 = a^2 + b^2 + c^2 →
  s = Real.sqrt 66 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_sphere_radius_l4019_401906


namespace NUMINAMATH_CALUDE_library_book_purchase_l4019_401939

/-- The library's book purchase problem -/
theorem library_book_purchase :
  let total_spent : ℕ := 4500
  let total_books : ℕ := 300
  let price_zhuangzi : ℕ := 10
  let price_confucius : ℕ := 20
  let price_mencius : ℕ := 15
  let price_laozi : ℕ := 28
  let price_sunzi : ℕ := 12
  ∀ (num_zhuangzi num_confucius num_mencius num_laozi num_sunzi : ℕ),
    num_zhuangzi + num_confucius + num_mencius + num_laozi + num_sunzi = total_books →
    num_zhuangzi * price_zhuangzi + num_confucius * price_confucius + 
    num_mencius * price_mencius + num_laozi * price_laozi + 
    num_sunzi * price_sunzi = total_spent →
    num_zhuangzi = num_confucius →
    num_sunzi = 4 * num_laozi + 15 →
    num_sunzi = 195 :=
by sorry

end NUMINAMATH_CALUDE_library_book_purchase_l4019_401939


namespace NUMINAMATH_CALUDE_chemistry_books_count_l4019_401926

/-- The number of ways to choose 2 items from n items -/
def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The problem statement -/
theorem chemistry_books_count :
  ∃ (C : ℕ), C > 0 ∧ choose_2 15 * choose_2 C = 2940 ∧ C = 8 := by sorry

end NUMINAMATH_CALUDE_chemistry_books_count_l4019_401926


namespace NUMINAMATH_CALUDE_range_of_f_l4019_401980

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 7

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ -2} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l4019_401980


namespace NUMINAMATH_CALUDE_sin_difference_range_l4019_401951

theorem sin_difference_range (a : ℝ) : 
  (∃ x : ℝ, Real.sin (x + π/4) - Real.sin (2*x) = a) → 
  -2 ≤ a ∧ a ≤ 9/8 := by
sorry

end NUMINAMATH_CALUDE_sin_difference_range_l4019_401951


namespace NUMINAMATH_CALUDE_right_triangle_conditions_l4019_401999

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides

-- Define what it means for a triangle to be right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Define the conditions
def condition1 (t : Triangle) : Prop := t.A + t.B = t.C
def condition2 (t : Triangle) : Prop := ∃ (k : Real), t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k
def condition3 (t : Triangle) : Prop := t.A = 90 - t.B

-- Theorem statement
theorem right_triangle_conditions (t : Triangle) :
  (condition1 t ∨ condition2 t ∨ condition3 t) → isRightTriangle t :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_conditions_l4019_401999


namespace NUMINAMATH_CALUDE_farmer_percentage_gain_l4019_401915

/-- Represents the farmer's cow transaction -/
structure CowTransaction where
  total_cows : ℕ
  sold_first : ℕ
  sold_second : ℕ
  price_increase : ℚ

/-- Calculates the percentage gain for a given cow transaction -/
def percentageGain (t : CowTransaction) : ℚ :=
  let first_sale := t.total_cows
  let second_sale := (t.sold_second : ℚ) * (1 + t.price_increase) * (t.total_cows : ℚ) / (t.sold_first : ℚ)
  let total_revenue := first_sale + second_sale
  let profit := total_revenue - (t.total_cows : ℚ)
  (profit / (t.total_cows : ℚ)) * 100

/-- The specific transaction described in the problem -/
def farmerTransaction : CowTransaction :=
  { total_cows := 600
  , sold_first := 500
  , sold_second := 100
  , price_increase := 1/10 }

/-- Theorem stating that the percentage gain for the farmer's transaction is 22% -/
theorem farmer_percentage_gain :
  percentageGain farmerTransaction = 22 := by
  sorry

#eval percentageGain farmerTransaction

end NUMINAMATH_CALUDE_farmer_percentage_gain_l4019_401915


namespace NUMINAMATH_CALUDE_floor_length_calculation_l4019_401932

/-- Given a rectangular floor with width 8 m, covered by a square carpet with 4 m sides,
    leaving 64 square meters uncovered, the length of the floor is 10 m. -/
theorem floor_length_calculation (floor_width : ℝ) (carpet_side : ℝ) (uncovered_area : ℝ) :
  floor_width = 8 →
  carpet_side = 4 →
  uncovered_area = 64 →
  (floor_width * (carpet_side ^ 2 + uncovered_area) / floor_width) = 10 :=
by
  sorry

#check floor_length_calculation

end NUMINAMATH_CALUDE_floor_length_calculation_l4019_401932


namespace NUMINAMATH_CALUDE_daniels_animals_legs_l4019_401907

/-- Calculates the total number of legs for Daniel's animals -/
def totalAnimalLegs (horses dogs cats turtles goats : ℕ) : ℕ :=
  4 * (horses + dogs + cats + turtles + goats)

/-- Theorem: Daniel's animals have 72 legs in total -/
theorem daniels_animals_legs :
  totalAnimalLegs 2 5 7 3 1 = 72 := by
  sorry

end NUMINAMATH_CALUDE_daniels_animals_legs_l4019_401907


namespace NUMINAMATH_CALUDE_problem_statement_l4019_401974

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4019_401974


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l4019_401960

/-- A linear function passing through the first, second, and fourth quadrants -/
def passes_through_124_quadrants (m : ℝ) : Prop :=
  (∀ x y : ℝ, y = (m - 2) * x + m + 1 → 
    (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))

/-- The main theorem stating the condition for m -/
theorem linear_function_quadrants (m : ℝ) : 
  passes_through_124_quadrants m ↔ -1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l4019_401960


namespace NUMINAMATH_CALUDE_intersection_M_N_l4019_401908

def M : Set ℝ := {-1, 1, 2, 4}
def N : Set ℝ := {x | x > 2}

theorem intersection_M_N : M ∩ N = {4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4019_401908


namespace NUMINAMATH_CALUDE_river_width_theorem_l4019_401904

/-- Represents a ferry crossing a river -/
structure Ferry where
  speed : ℝ
  initial_position : ℝ

/-- Represents a river crossing scenario -/
structure RiverCrossing where
  width : ℝ
  ferry1 : Ferry
  ferry2 : Ferry
  first_meeting_distance : ℝ
  second_meeting_distance : ℝ

/-- The theorem stating the width of the river given the crossing conditions -/
theorem river_width_theorem (rc : RiverCrossing) 
  (h1 : rc.ferry1.speed ≠ rc.ferry2.speed)
  (h2 : rc.ferry1.initial_position = 0)
  (h3 : rc.ferry2.initial_position = rc.width)
  (h4 : rc.first_meeting_distance = 720)
  (h5 : rc.second_meeting_distance = 400)
  : rc.width = 1760 := by
  sorry

end NUMINAMATH_CALUDE_river_width_theorem_l4019_401904


namespace NUMINAMATH_CALUDE_factor_polynomial_l4019_401996

theorem factor_polynomial (x : ℝ) : 75 * x^7 - 50 * x^10 = 25 * x^7 * (3 - 2 * x^3) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l4019_401996


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l4019_401920

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  let angle := angle_between_vectors a b
  let a_magnitude := Real.sqrt (a.1^2 + a.2^2)
  let b_magnitude := Real.sqrt (b.1^2 + b.2^2)
  angle = π/3 ∧ a = (2, 0) ∧ b_magnitude = 1 →
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l4019_401920


namespace NUMINAMATH_CALUDE_tethered_dog_area_l4019_401928

/-- The area outside a regular hexagon reachable by a tethered point -/
theorem tethered_dog_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 1 →
  rope_length = 3 →
  let outside_area := (rope_length^2 * (5/6) + 2 * (rope_length - side_length)^2 * (1/6)) * π
  outside_area = (49/6) * π :=
by sorry

end NUMINAMATH_CALUDE_tethered_dog_area_l4019_401928


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4019_401913

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4019_401913


namespace NUMINAMATH_CALUDE_trigonometric_relations_l4019_401905

theorem trigonometric_relations (x : Real) 
  (h1 : -π/2 < x) (h2 : x < 0) (h3 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x * Real.cos x = -12/25) ∧ 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  (Real.tan x = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_relations_l4019_401905


namespace NUMINAMATH_CALUDE_music_students_l4019_401935

/-- Proves that the number of students taking music is 50 given the conditions of the problem -/
theorem music_students (total : ℕ) (art : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : art = 20)
  (h3 : both = 10)
  (h4 : neither = 440) :
  ∃ music : ℕ, music = 50 ∧ total = music + art - both + neither :=
by sorry

end NUMINAMATH_CALUDE_music_students_l4019_401935


namespace NUMINAMATH_CALUDE_final_position_is_correct_l4019_401950

/-- Movement pattern A: 1 unit up, 2 units right -/
def pattern_a : ℤ × ℤ := (2, 1)

/-- Movement pattern B: 3 units left, 2 units down -/
def pattern_b : ℤ × ℤ := (-3, -2)

/-- Calculate the position after n movements -/
def position_after_n_movements (n : ℕ) : ℤ × ℤ :=
  let a_count := (n + 1) / 2
  let b_count := n / 2
  (a_count * pattern_a.1 + b_count * pattern_b.1,
   a_count * pattern_a.2 + b_count * pattern_b.2)

/-- The final position after 15 movements -/
def final_position : ℤ × ℤ := position_after_n_movements 15

theorem final_position_is_correct : final_position = (-5, -6) := by
  sorry

end NUMINAMATH_CALUDE_final_position_is_correct_l4019_401950


namespace NUMINAMATH_CALUDE_limit_example_l4019_401933

/-- The limit of (2x^2 - 5x + 2) / (x - 1/2) as x approaches 1/2 is -3 -/
theorem limit_example (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, x ≠ 1/2 → |x - 1/2| < δ →
    |((2 * x^2 - 5 * x + 2) / (x - 1/2)) + 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_example_l4019_401933


namespace NUMINAMATH_CALUDE_game_ends_in_36_rounds_l4019_401941

/-- Represents the state of a player in the game -/
structure PlayerState :=
  (tokens : ℕ)

/-- Represents the state of the game -/
structure GameState :=
  (a : PlayerState)
  (b : PlayerState)
  (c : PlayerState)
  (round : ℕ)

/-- Updates the game state for a single round -/
def updateRound (state : GameState) : GameState :=
  sorry

/-- Updates the game state for the extra discard every 5 rounds -/
def extraDiscard (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (any player has 0 tokens) -/
def gameEnded (state : GameState) : Bool :=
  sorry

/-- The main theorem stating that the game ends after exactly 36 rounds -/
theorem game_ends_in_36_rounds :
  let initialState : GameState := {
    a := { tokens := 17 },
    b := { tokens := 16 },
    c := { tokens := 15 },
    round := 0
  }
  ∃ (finalState : GameState),
    (finalState.round = 36) ∧
    (gameEnded finalState = true) ∧
    (∀ (intermediateState : GameState),
      intermediateState.round < 36 →
      gameEnded intermediateState = false) :=
sorry

end NUMINAMATH_CALUDE_game_ends_in_36_rounds_l4019_401941


namespace NUMINAMATH_CALUDE_pythagorean_triple_check_l4019_401942

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_triple_check :
  ¬ is_pythagorean_triple 12 15 18 ∧
  is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 6 9 15 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_check_l4019_401942


namespace NUMINAMATH_CALUDE_prob_product_not_zero_l4019_401979

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The probability of getting a number other than 1 on a single die -/
def probNotOne : ℚ := (numSides - 1) / numSides

/-- The number of dice tossed -/
def numDice : ℕ := 3

/-- The probability that (a-1)(b-1)(c-1) ≠ 0 when tossing three eight-sided dice -/
theorem prob_product_not_zero : 
  (probNotOne ^ numDice : ℚ) = 343 / 512 := by sorry

end NUMINAMATH_CALUDE_prob_product_not_zero_l4019_401979


namespace NUMINAMATH_CALUDE_specific_grid_toothpicks_l4019_401910

/-- Represents a rectangular grid of toothpicks with reinforcements -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ
  horizontalReinforcementInterval : ℕ
  verticalReinforcementInterval : ℕ

/-- Calculates the total number of toothpicks in the grid -/
def totalToothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontalLines := grid.height + 1
  let verticalLines := grid.width + 1
  let baseHorizontal := horizontalLines * grid.width
  let baseVertical := verticalLines * grid.height
  let reinforcedHorizontal := (horizontalLines / grid.horizontalReinforcementInterval) * grid.width
  let reinforcedVertical := (verticalLines / grid.verticalReinforcementInterval) * grid.height
  baseHorizontal + baseVertical + reinforcedHorizontal + reinforcedVertical

/-- Theorem stating that the specific grid configuration results in 990 toothpicks -/
theorem specific_grid_toothpicks :
  totalToothpicks { height := 25, width := 15, horizontalReinforcementInterval := 5, verticalReinforcementInterval := 3 } = 990 := by
  sorry

end NUMINAMATH_CALUDE_specific_grid_toothpicks_l4019_401910


namespace NUMINAMATH_CALUDE_consecutive_even_sequence_unique_l4019_401914

/-- A sequence of four consecutive even integers -/
def ConsecutiveEvenSequence (a b c d : ℤ) : Prop :=
  (b = a + 2) ∧ (c = b + 2) ∧ (d = c + 2) ∧ Even a ∧ Even b ∧ Even c ∧ Even d

theorem consecutive_even_sequence_unique :
  ∀ a b c d : ℤ,
  ConsecutiveEvenSequence a b c d →
  c = 14 →
  a + b + c + d = 52 →
  a = 10 ∧ b = 12 ∧ c = 14 ∧ d = 16 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_sequence_unique_l4019_401914


namespace NUMINAMATH_CALUDE_complex_number_simplification_l4019_401959

theorem complex_number_simplification :
  (7 - 3*Complex.I) - 3*(2 - 5*Complex.I) + 4*Complex.I = 1 + 16*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l4019_401959


namespace NUMINAMATH_CALUDE_ellipse_equation_l4019_401989

/-- An ellipse passing through (-√15, 5/2) with the same foci as 9x^2 + 4y^2 = 36 -/
theorem ellipse_equation (x y : ℝ) : 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 = b^2 + 5 ∧ 
   x^2 / b^2 + y^2 / a^2 = 1 ∧
   (-Real.sqrt 15)^2 / b^2 + (5/2)^2 / a^2 = 1) →
  x^2 / 20 + y^2 / 25 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4019_401989


namespace NUMINAMATH_CALUDE_base_conversion_sum_equality_l4019_401969

def base_to_decimal (digits : List Nat) (base : Nat) : Rat :=
  (digits.reverse.enum.map (λ (i, d) => d * base ^ i)).sum

theorem base_conversion_sum_equality : 
  let a := base_to_decimal [2, 5, 4] 8
  let b := base_to_decimal [1, 2] 4
  let c := base_to_decimal [1, 3, 2] 5
  let d := base_to_decimal [2, 2] 3
  a / b + c / d = 33.9167 := by sorry

end NUMINAMATH_CALUDE_base_conversion_sum_equality_l4019_401969


namespace NUMINAMATH_CALUDE_doubled_cost_percentage_doubled_cost_percentage_1600_l4019_401973

-- Define the cost function
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

-- Theorem statement
theorem doubled_cost_percentage (t : ℝ) (b : ℝ) (h : t > 0) (h2 : b > 0) :
  cost t (2 * b) = 16 * cost t b := by
  sorry

-- Corollary to express the result as a percentage
theorem doubled_cost_percentage_1600 (t : ℝ) (b : ℝ) (h : t > 0) (h2 : b > 0) :
  cost t (2 * b) / cost t b = 16 := by
  sorry

end NUMINAMATH_CALUDE_doubled_cost_percentage_doubled_cost_percentage_1600_l4019_401973


namespace NUMINAMATH_CALUDE_monday_rain_inches_l4019_401934

/-- Proves that the number of inches of rain collected on Monday is 4 -/
theorem monday_rain_inches (
  gallons_per_inch : ℝ)
  (tuesday_rain : ℝ)
  (price_per_gallon : ℝ)
  (total_revenue : ℝ)
  (h1 : gallons_per_inch = 15)
  (h2 : tuesday_rain = 3)
  (h3 : price_per_gallon = 1.2)
  (h4 : total_revenue = 126)
  : ∃ (monday_rain : ℝ), monday_rain = 4 ∧
    gallons_per_inch * (monday_rain + tuesday_rain) * price_per_gallon = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_monday_rain_inches_l4019_401934


namespace NUMINAMATH_CALUDE_average_net_income_is_399_50_l4019_401953

/-- Represents the daily income and expense for a cab driver --/
structure DailyFinance where
  income : ℝ
  expense : ℝ

/-- Calculates the net income for a single day --/
def netIncome (df : DailyFinance) : ℝ := df.income - df.expense

/-- The cab driver's finances for 10 days --/
def tenDaysFinances : List DailyFinance := [
  ⟨600, 50⟩,
  ⟨250, 70⟩,
  ⟨450, 100⟩,
  ⟨400, 30⟩,
  ⟨800, 60⟩,
  ⟨450, 40⟩,
  ⟨350, 0⟩,
  ⟨600, 55⟩,
  ⟨270, 80⟩,
  ⟨500, 90⟩
]

/-- Theorem: The average daily net income for the cab driver over 10 days is $399.50 --/
theorem average_net_income_is_399_50 :
  (tenDaysFinances.map netIncome).sum / 10 = 399.50 := by
  sorry


end NUMINAMATH_CALUDE_average_net_income_is_399_50_l4019_401953


namespace NUMINAMATH_CALUDE_xiao_ming_age_l4019_401944

theorem xiao_ming_age : ∃ (xiao_ming_age : ℕ), 
  (∃ (dad_age : ℕ), 
    dad_age - xiao_ming_age = 28 ∧ 
    dad_age = 3 * xiao_ming_age) → 
  xiao_ming_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_age_l4019_401944


namespace NUMINAMATH_CALUDE_sqrt_one_minus_sqrt_three_squared_l4019_401956

theorem sqrt_one_minus_sqrt_three_squared : 
  Real.sqrt ((1 - Real.sqrt 3) ^ 2) = Real.sqrt 3 - 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_one_minus_sqrt_three_squared_l4019_401956


namespace NUMINAMATH_CALUDE_locus_of_perpendicular_foot_l4019_401995

/-- Given a plane P (z = 0), points A on P and O not on P, prove that the locus of points H,
    where H is the foot of the perpendicular from O to any line in P through A,
    forms a circle with the given equation. -/
theorem locus_of_perpendicular_foot (a b d e f : ℝ) :
  let P : Set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}
  let A : ℝ × ℝ × ℝ := (a, b, 0)
  let O : ℝ × ℝ × ℝ := (d, e, f)
  let H := {h : ℝ × ℝ × ℝ | ∃ (u v : ℝ),
    h = ((a * (u^2 + v^2) + (d*u + e*v - a*u - b*v)*u) / (u^2 + v^2),
         (b * (u^2 + v^2) + (d*u + e*v - a*u - b*v)*v) / (u^2 + v^2),
         0)}
  ∀ (x y : ℝ), (x, y, 0) ∈ H ↔ x^2 + y^2 - (a+d)*x - (b+e)*y + a*d + b*e = 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_perpendicular_foot_l4019_401995


namespace NUMINAMATH_CALUDE_hypotenuse_square_l4019_401987

/-- Given complex numbers a, b, and c that are zeros of a polynomial P(z) = z^3 + pz + q,
    and satisfy |a|^2 + |b|^2 + |c|^2 = 360, if the points corresponding to a, b, and c
    form a right triangle with the right angle at a, then the square of the length of
    the hypotenuse is 432. -/
theorem hypotenuse_square (a b c : ℂ) (p q : ℂ) :
  (a^3 + p*a + q = 0) →
  (b^3 + p*b + q = 0) →
  (c^3 + p*c + q = 0) →
  (Complex.abs a)^2 + (Complex.abs b)^2 + (Complex.abs c)^2 = 360 →
  (b - a) • (c - a) = 0 →  -- Right angle at a
  (Complex.abs (b - c))^2 = 432 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_square_l4019_401987


namespace NUMINAMATH_CALUDE_dagger_example_l4019_401988

-- Define the ternary operation ⋄
def dagger (a b c d e f : ℚ) : ℚ := (a * c * e) * ((d * f) / b)

-- Theorem statement
theorem dagger_example : dagger 5 9 7 2 11 5 = 3850 / 9 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l4019_401988


namespace NUMINAMATH_CALUDE_tree_planting_speeds_l4019_401902

-- Define the given constants
def distance : ℝ := 10
def time_difference : ℝ := 1.5
def speed_ratio : ℝ := 2.5

-- Define the walking speed and cycling speed
def walking_speed : ℝ := 4
def cycling_speed : ℝ := 10

-- Define the increased cycling speed
def increased_cycling_speed : ℝ := 12

-- Theorem statement
theorem tree_planting_speeds :
  (distance / walking_speed - distance / cycling_speed = time_difference) ∧
  (cycling_speed = speed_ratio * walking_speed) ∧
  (distance / increased_cycling_speed = distance / cycling_speed - 1/6) :=
sorry

end NUMINAMATH_CALUDE_tree_planting_speeds_l4019_401902


namespace NUMINAMATH_CALUDE_buddy_fraction_l4019_401916

theorem buddy_fraction (s n : ℕ) (h1 : n > 0) (h2 : s > 0) : 
  (n / 3 : ℚ) = (2 * s / 5 : ℚ) → 
  ((n / 3 + 2 * s / 5) / (n + s) : ℚ) = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_buddy_fraction_l4019_401916


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l4019_401961

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ (2 : ℂ) / (i * (3 - i)) = (1 - 3*i) / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l4019_401961


namespace NUMINAMATH_CALUDE_students_without_favorite_l4019_401912

theorem students_without_favorite (total : ℕ) (math_frac english_frac history_frac science_frac : ℚ) : 
  total = 120 →
  math_frac = 3 / 10 →
  english_frac = 5 / 12 →
  history_frac = 1 / 8 →
  science_frac = 3 / 20 →
  total - (↑total * math_frac).floor - (↑total * english_frac).floor - 
  (↑total * history_frac).floor - (↑total * science_frac).floor = 1 := by
  sorry

end NUMINAMATH_CALUDE_students_without_favorite_l4019_401912


namespace NUMINAMATH_CALUDE_symmetric_function_theorem_l4019_401903

/-- A function f: ℝ → ℝ is symmetric about the origin -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem symmetric_function_theorem (f : ℝ → ℝ) 
  (h_sym : SymmetricAboutOrigin f)
  (h_nonneg : ∀ x ≥ 0, f x = x * (1 - x)) :
  ∀ x ≤ 0, f x = x * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_theorem_l4019_401903


namespace NUMINAMATH_CALUDE_pet_shop_dogs_count_l4019_401947

/-- Represents the number of animals of each type in the pet shop -/
structure PetShop where
  dogs : ℕ
  cats : ℕ
  bunnies : ℕ

/-- The ratio of dogs to cats to bunnies -/
def ratio : Fin 3 → ℕ
| 0 => 3  -- dogs
| 1 => 7  -- cats
| 2 => 12 -- bunnies

theorem pet_shop_dogs_count (shop : PetShop) :
  (ratio 0 : ℚ) / shop.dogs = (ratio 1 : ℚ) / shop.cats ∧
  (ratio 0 : ℚ) / shop.dogs = (ratio 2 : ℚ) / shop.bunnies ∧
  shop.dogs + shop.bunnies = 375 →
  shop.dogs = 75 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_dogs_count_l4019_401947


namespace NUMINAMATH_CALUDE_complex_fraction_real_implies_a_equals_two_l4019_401993

theorem complex_fraction_real_implies_a_equals_two (a : ℝ) :
  (((a : ℂ) + 2 * Complex.I) / (1 + Complex.I)).im = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_implies_a_equals_two_l4019_401993


namespace NUMINAMATH_CALUDE_petrov_insurance_cost_l4019_401992

/-- Calculate the total insurance cost for the Petrov family's mortgage --/
def calculate_insurance_cost (apartment_cost loan_amount interest_rate property_rate
                              woman_rate man_rate title_rate maria_share vasily_share : ℝ) : ℝ :=
  let total_loan := loan_amount * (1 + interest_rate)
  let property_cost := total_loan * property_rate
  let title_cost := total_loan * title_rate
  let maria_cost := total_loan * maria_share * woman_rate
  let vasily_cost := total_loan * vasily_share * man_rate
  property_cost + title_cost + maria_cost + vasily_cost

/-- The total insurance cost for the Petrov family's mortgage is 47481.2 rubles --/
theorem petrov_insurance_cost :
  calculate_insurance_cost 13000000 8000000 0.095 0.0009 0.0017 0.0019 0.0027 0.4 0.6 = 47481.2 := by
  sorry

end NUMINAMATH_CALUDE_petrov_insurance_cost_l4019_401992


namespace NUMINAMATH_CALUDE_celeste_opod_probability_l4019_401981

/-- Represents the duration of a song in seconds -/
def SongDuration := ℕ

/-- Represents the o-Pod with its songs -/
structure OPod where
  songs : List SongDuration
  favorite_song : SongDuration

/-- Creates an o-Pod with 10 songs, where each song is 30 seconds longer than the previous one -/
def create_opod (favorite_duration : ℕ) : OPod :=
  { songs := List.range 10 |>.map (fun i => 30 * (i + 1)),
    favorite_song := favorite_duration }

/-- Calculates the probability of not hearing the entire favorite song within a given time -/
noncomputable def prob_not_hear_favorite (opod : OPod) (total_time : ℕ) : ℚ :=
  sorry

theorem celeste_opod_probability :
  let celeste_opod := create_opod 210
  prob_not_hear_favorite celeste_opod 270 = 79 / 90 := by
  sorry

end NUMINAMATH_CALUDE_celeste_opod_probability_l4019_401981


namespace NUMINAMATH_CALUDE_three_and_one_fifth_cubed_l4019_401921

theorem three_and_one_fifth_cubed : (3 + 1/5) ^ 3 = 32.768 := by sorry

end NUMINAMATH_CALUDE_three_and_one_fifth_cubed_l4019_401921


namespace NUMINAMATH_CALUDE_race_speed_calculation_l4019_401958

/-- Given two teams racing on a 300-mile course, where one team finishes 3 hours earlier
    and has an average speed 5 mph greater than the other, prove that the slower team's
    average speed is 20 mph. -/
theorem race_speed_calculation (distance : ℝ) (time_diff : ℝ) (speed_diff : ℝ) :
  distance = 300 ∧ time_diff = 3 ∧ speed_diff = 5 →
  ∃ (speed_e : ℝ) (time_e : ℝ),
    speed_e > 0 ∧
    time_e > 0 ∧
    distance = speed_e * time_e ∧
    distance = (speed_e + speed_diff) * (time_e - time_diff) ∧
    speed_e = 20 :=
by sorry

end NUMINAMATH_CALUDE_race_speed_calculation_l4019_401958


namespace NUMINAMATH_CALUDE_systematic_sampling_example_l4019_401971

def isSystematicSample (sample : List Nat) (totalItems : Nat) (sampleSize : Nat) : Prop :=
  sample.length = sampleSize ∧
  ∀ i, i ∈ sample → i ≤ totalItems ∧
  ∀ i j, i < j → i ∈ sample → j ∈ sample → ∃ k, j - i = k * ((totalItems - 1) / (sampleSize - 1))

theorem systematic_sampling_example :
  isSystematicSample [3, 13, 23, 33, 43] 50 5 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_example_l4019_401971


namespace NUMINAMATH_CALUDE_decagon_perimeter_l4019_401949

/-- A decagon is a polygon with 10 sides -/
def Decagon := Nat

/-- The number of sides in a decagon -/
def decagon_sides : Nat := 10

/-- The length of each side of our specific decagon -/
def side_length : ℝ := 3

/-- The perimeter of a polygon is the sum of the lengths of all its sides -/
def perimeter (n : Nat) (s : ℝ) : ℝ := n * s

/-- Theorem: The perimeter of a decagon with sides of length 3 units is 30 units -/
theorem decagon_perimeter : perimeter decagon_sides side_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_decagon_perimeter_l4019_401949


namespace NUMINAMATH_CALUDE_geese_survival_l4019_401957

/-- Given the following conditions:
  1. 500 goose eggs were laid
  2. 2/3 of eggs hatched
  3. 3/4 of hatched geese survived the first month
  4. 2/5 of geese that survived the first month survived the first year
Prove that 100 geese survived the first year -/
theorem geese_survival (total_eggs : ℕ) (hatch_rate first_month_rate first_year_rate : ℚ) :
  total_eggs = 500 →
  hatch_rate = 2/3 →
  first_month_rate = 3/4 →
  first_year_rate = 2/5 →
  (total_eggs : ℚ) * hatch_rate * first_month_rate * first_year_rate = 100 := by
  sorry

#eval (500 : ℚ) * (2/3) * (3/4) * (2/5)

end NUMINAMATH_CALUDE_geese_survival_l4019_401957


namespace NUMINAMATH_CALUDE_five_objects_three_categories_l4019_401965

/-- The number of ways to distribute n distinguishable objects into k distinct categories -/
def distributionWays (n k : ℕ) : ℕ := k ^ n

/-- Theorem: There are 243 ways to distribute 5 distinguishable objects into 3 distinct categories -/
theorem five_objects_three_categories : distributionWays 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_objects_three_categories_l4019_401965


namespace NUMINAMATH_CALUDE_abc_sum_reciprocal_l4019_401966

theorem abc_sum_reciprocal (a b c : ℝ) (ha : a ≠ 1) (hb : b ≠ 1) (hc : c ≠ 1)
  (h1 : a * b * c = 1)
  (h2 : a^2 + b^2 + c^2 - (1/a^2 + 1/b^2 + 1/c^2) = 8*(a+b+c) - 8*(a*b+b*c+c*a)) :
  1/(a-1) + 1/(b-1) + 1/(c-1) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_reciprocal_l4019_401966


namespace NUMINAMATH_CALUDE_soccer_team_average_goals_l4019_401930

theorem soccer_team_average_goals (pizzas : ℕ) (slices_per_pizza : ℕ) (games : ℕ)
  (h1 : pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : games = 8) :
  (pizzas * slices_per_pizza) / games = 9 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_average_goals_l4019_401930


namespace NUMINAMATH_CALUDE_cos_20_minus_cos_40_l4019_401922

theorem cos_20_minus_cos_40 : Real.cos (20 * π / 180) - Real.cos (40 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_20_minus_cos_40_l4019_401922


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l4019_401909

theorem contrapositive_equivalence (a b : ℝ) :
  (¬ (-Real.sqrt b < a ∧ a < Real.sqrt b) → ¬ (a^2 < b)) ↔
  ((a ≥ Real.sqrt b ∨ a ≤ -Real.sqrt b) → a^2 ≥ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l4019_401909


namespace NUMINAMATH_CALUDE_expected_remaining_people_l4019_401997

/-- The expected number of people remaining in a line of 100 people after a removal process -/
theorem expected_remaining_people (n : Nat) (h : n = 100) :
  let people := n
  let facing_right := n / 2
  let facing_left := n - facing_right
  let expected_remaining := (2^n : ℝ) / (Nat.choose n facing_right) - 1
  expected_remaining = (2^100 : ℝ) / (Nat.choose 100 50) - 1 := by
  sorry

end NUMINAMATH_CALUDE_expected_remaining_people_l4019_401997


namespace NUMINAMATH_CALUDE_total_spent_is_correct_l4019_401900

-- Define the value of a penny in dollars
def penny_value : ℚ := 1 / 100

-- Define the value of a dime in dollars
def dime_value : ℚ := 1 / 10

-- Define the number of pennies spent on ice cream
def ice_cream_pennies : ℕ := 2

-- Define the number of dimes spent on baseball cards
def baseball_cards_dimes : ℕ := 12

-- Theorem statement
theorem total_spent_is_correct :
  (ice_cream_pennies : ℚ) * penny_value + (baseball_cards_dimes : ℚ) * dime_value = 122 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_correct_l4019_401900


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l4019_401972

theorem regular_polygon_sides (interior_angle : ℝ) (sum_except_one : ℝ) : 
  interior_angle = 160 → sum_except_one = 3600 → 
  (sum_except_one + interior_angle) / interior_angle = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l4019_401972


namespace NUMINAMATH_CALUDE_tangent_is_simson_line_l4019_401940

/-- A parabola in a 2D plane. -/
structure Parabola where
  -- Add necessary fields to define a parabola

/-- A triangle in a 2D plane. -/
structure Triangle where
  -- Add necessary fields to define a triangle

/-- The Simson line of a triangle with respect to a point. -/
def SimsonLine (t : Triangle) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- The tangent line to a parabola at a given point. -/
def TangentLine (p : Parabola) (point : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- The vertex of a parabola. -/
def Vertex (p : Parabola) : ℝ × ℝ :=
  sorry

/-- Given three tangent lines to a parabola, find their intersection points forming a triangle. -/
def TriangleFromTangents (p : Parabola) (t1 t2 t3 : Set (ℝ × ℝ)) : Triangle :=
  sorry

theorem tangent_is_simson_line (p : Parabola) (t1 t2 t3 : Set (ℝ × ℝ)) :
  TangentLine p (Vertex p) = SimsonLine (TriangleFromTangents p t1 t2 t3) (Vertex p) :=
sorry

end NUMINAMATH_CALUDE_tangent_is_simson_line_l4019_401940


namespace NUMINAMATH_CALUDE_perpendicular_line_exists_l4019_401925

-- Define a line in a 2D plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in a 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for a point to not be on a line
def Point.notOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c ≠ 0

-- Define what it means for a line to be perpendicular to another line
def Line.perpendicularTo (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- The theorem statement
theorem perpendicular_line_exists (P : Point) (L : Line) (h : P.notOnLine L) :
  ∃ (M : Line), M.perpendicularTo L ∧ P.notOnLine M := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_exists_l4019_401925


namespace NUMINAMATH_CALUDE_prime_power_sum_perfect_square_l4019_401937

theorem prime_power_sum_perfect_square (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r →
  (∃ n : ℕ, p^q + p^r = n^2) ↔ 
  ((p = 2 ∧ q = r ∧ ∃ k : ℕ, Prime k ∧ k > 2 ∧ q = k) ∨
   (p = 3 ∧ ((q = 3 ∧ r = 2) ∨ (q = 2 ∧ r = 3)))) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_sum_perfect_square_l4019_401937


namespace NUMINAMATH_CALUDE_sphere_in_parabolic_glass_l4019_401943

/-- The distance from the highest point of a sphere to the bottom of a parabolic wine glass --/
theorem sphere_in_parabolic_glass (x y : ℝ) (b : ℝ) : 
  (∀ y, 0 ≤ y → y < 15 → x^2 = 2*y) →  -- Parabola equation
  (x^2 + (y - b)^2 = 9) →               -- Sphere equation
  ((2 - 2*b)^2 - 4*(b^2 - 9) = 0) →     -- Tangency condition
  (b + 3 = 8) :=                        -- Distance from highest point to bottom
by sorry

end NUMINAMATH_CALUDE_sphere_in_parabolic_glass_l4019_401943
