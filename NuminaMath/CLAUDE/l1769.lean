import Mathlib

namespace NUMINAMATH_CALUDE_scaled_cylinder_volume_l1769_176998

/-- Theorem: Scaling a cylindrical container -/
theorem scaled_cylinder_volume (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  π * r^2 * h = 3 →
  π * (2*r)^2 * (4*h) = 48 := by
  sorry

end NUMINAMATH_CALUDE_scaled_cylinder_volume_l1769_176998


namespace NUMINAMATH_CALUDE_event_probability_l1769_176971

theorem event_probability (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^3 = 63/64) →
  3 * p * (1 - p)^2 = 9/64 := by
sorry

end NUMINAMATH_CALUDE_event_probability_l1769_176971


namespace NUMINAMATH_CALUDE_intersection_integer_points_l1769_176996

theorem intersection_integer_points (k : ℝ) : 
  (∃! (n : ℕ), n = 3 ∧ 
    (∀ (x y : ℤ), 
      (y = 4*k*x - 1/k ∧ y = (1/k)*x + 2) → 
      (∃ (k₁ k₂ k₃ : ℝ), k = k₁ ∨ k = k₂ ∨ k = k₃))) :=
by sorry

end NUMINAMATH_CALUDE_intersection_integer_points_l1769_176996


namespace NUMINAMATH_CALUDE_total_wheels_l1769_176925

theorem total_wheels (bicycles tricycles : ℕ) 
  (bicycle_wheels tricycle_wheels : ℕ) : 
  bicycles = 24 → 
  tricycles = 14 → 
  bicycle_wheels = 2 → 
  tricycle_wheels = 3 → 
  bicycles * bicycle_wheels + tricycles * tricycle_wheels = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_l1769_176925


namespace NUMINAMATH_CALUDE_jayden_half_ernesto_age_l1769_176986

/-- 
Given:
- Ernesto is currently 11 years old
- Jayden is currently 4 years old

Prove that in 3 years, Jayden will be half of Ernesto's age
-/
theorem jayden_half_ernesto_age :
  let ernesto_age : ℕ := 11
  let jayden_age : ℕ := 4
  let years_until_half : ℕ := 3
  (jayden_age + years_until_half : ℚ) = (1/2 : ℚ) * (ernesto_age + years_until_half : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_jayden_half_ernesto_age_l1769_176986


namespace NUMINAMATH_CALUDE_x_fifth_minus_ten_x_l1769_176969

theorem x_fifth_minus_ten_x (x : ℝ) : x = 5 → x^5 - 10*x = 3075 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_minus_ten_x_l1769_176969


namespace NUMINAMATH_CALUDE_student_team_repetition_l1769_176992

theorem student_team_repetition (n : ℕ) (h : n > 0) :
  ∀ (arrangement : ℕ → Fin (n^2) → Fin n),
  ∃ (week1 week2 : ℕ) (student1 student2 : Fin (n^2)),
    week1 < week2 ∧ week2 ≤ n + 2 ∧
    student1 ≠ student2 ∧
    arrangement week1 student1 = arrangement week1 student2 ∧
    arrangement week2 student1 = arrangement week2 student2 :=
by sorry

end NUMINAMATH_CALUDE_student_team_repetition_l1769_176992


namespace NUMINAMATH_CALUDE_binomial_20_4_l1769_176972

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_4_l1769_176972


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l1769_176941

theorem root_exists_in_interval :
  ∃ x : ℝ, 3/2 < x ∧ x < 2 ∧ 2^x = x^2 + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l1769_176941


namespace NUMINAMATH_CALUDE_mod_17_graph_intercepts_sum_l1769_176933

theorem mod_17_graph_intercepts_sum :
  ∀ x_0 y_0 : ℕ,
  x_0 < 17 →
  y_0 < 17 →
  (5 * x_0) % 17 = 2 →
  (3 * y_0) % 17 = 15 →
  x_0 + y_0 = 19 := by
sorry

end NUMINAMATH_CALUDE_mod_17_graph_intercepts_sum_l1769_176933


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1769_176935

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- Theorem: If S_20 = S_40 for an arithmetic sequence, then S_60 = 0 -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
  (h : seq.S 20 = seq.S 40) : seq.S 60 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1769_176935


namespace NUMINAMATH_CALUDE_triangle_side_expression_l1769_176977

/-- Given a triangle with sides a, b, and c, prove that |a-b+c|-|c-a-b| = 2c-2b -/
theorem triangle_side_expression (a b c : ℝ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  |a - b + c| - |c - a - b| = 2 * c - 2 * b := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_expression_l1769_176977


namespace NUMINAMATH_CALUDE_statement_2_statement_3_l1769_176918

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Statement 2
theorem statement_2 (m n : Line) (α : Plane) :
  parallel m α → perpendicular n α → perpendicular_lines n m :=
sorry

-- Statement 3
theorem statement_3 (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_statement_2_statement_3_l1769_176918


namespace NUMINAMATH_CALUDE_total_students_is_150_l1769_176920

/-- Represents the number of students in a school with age distribution. -/
structure School where
  total : ℕ
  below_8 : ℕ
  age_8 : ℕ
  above_8 : ℕ

/-- Conditions for the school problem. -/
def school_conditions (s : School) : Prop :=
  s.below_8 = (s.total * 20) / 100 ∧
  s.above_8 = (s.age_8 * 2) / 3 ∧
  s.age_8 = 72 ∧
  s.total = s.below_8 + s.age_8 + s.above_8

/-- Theorem stating that the total number of students is 150. -/
theorem total_students_is_150 :
  ∃ s : School, school_conditions s ∧ s.total = 150 := by
  sorry

#check total_students_is_150

end NUMINAMATH_CALUDE_total_students_is_150_l1769_176920


namespace NUMINAMATH_CALUDE_box_max_volume_l1769_176978

/-- Volume function for the box -/
def V (x : ℝ) : ℝ := (16 - 2*x) * (10 - 2*x) * x

/-- The theorem stating the maximum volume and corresponding height -/
theorem box_max_volume :
  ∃ (max_vol : ℝ) (max_height : ℝ),
    (∀ x, 0 < x → x < 5 → V x ≤ max_vol) ∧
    (0 < max_height ∧ max_height < 5) ∧
    (V max_height = max_vol) ∧
    (max_height = 2) ∧
    (max_vol = 144) := by
  sorry

end NUMINAMATH_CALUDE_box_max_volume_l1769_176978


namespace NUMINAMATH_CALUDE_point_relationship_l1769_176900

/-- Given points A(-2,a), B(-1,b), C(3,c) on the graph of y = 4/x, prove that b < a < c -/
theorem point_relationship (a b c : ℝ) : 
  (a = 4 / (-2)) → (b = 4 / (-1)) → (c = 4 / 3) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_point_relationship_l1769_176900


namespace NUMINAMATH_CALUDE_choose_four_from_fifteen_l1769_176985

theorem choose_four_from_fifteen (n : ℕ) (k : ℕ) : n = 15 ∧ k = 4 → Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_fifteen_l1769_176985


namespace NUMINAMATH_CALUDE_beginner_trig_probability_probability_calculation_l1769_176951

/-- Represents the number of students in each course -/
structure CourseEnrollment where
  BC : ℕ  -- Beginner Calculus
  AC : ℕ  -- Advanced Calculus
  IC : ℕ  -- Intermediate Calculus
  BT : ℕ  -- Beginner Trigonometry
  AT : ℕ  -- Advanced Trigonometry
  IT : ℕ  -- Intermediate Trigonometry

/-- Represents the enrollment conditions for the math department -/
def EnrollmentConditions (e : CourseEnrollment) (total : ℕ) : Prop :=
  e.BC + e.AC + e.IC = (60 * total) / 100 ∧
  e.BT + e.AT + e.IT = (40 * total) / 100 ∧
  e.BC + e.BT = (45 * total) / 100 ∧
  e.AC + e.AT = (35 * total) / 100 ∧
  e.IC + e.IT = (20 * total) / 100 ∧
  e.BC = (125 * e.BT) / 100 ∧
  e.IC + e.AC = (120 * (e.IT + e.AT)) / 100

theorem beginner_trig_probability (e : CourseEnrollment) (total : ℕ) :
  EnrollmentConditions e total → total = 5000 → e.BT = 1000 :=
by sorry

theorem probability_calculation (e : CourseEnrollment) (total : ℕ) :
  EnrollmentConditions e total → total = 5000 → e.BT = 1000 →
  (e.BT : ℚ) / total = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_beginner_trig_probability_probability_calculation_l1769_176951


namespace NUMINAMATH_CALUDE_divisible_by_45_sum_of_digits_l1769_176966

theorem divisible_by_45_sum_of_digits (a b : ℕ) : 
  a < 10 → b < 10 → (60000 + 1000 * a + 780 + b) % 45 = 0 → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_45_sum_of_digits_l1769_176966


namespace NUMINAMATH_CALUDE_paint_per_statue_l1769_176956

-- Define the total amount of paint
def total_paint : ℚ := 1/2

-- Define the number of statues that can be painted
def num_statues : ℕ := 2

-- Theorem: Each statue requires 1/4 gallon of paint
theorem paint_per_statue : total_paint / num_statues = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_paint_per_statue_l1769_176956


namespace NUMINAMATH_CALUDE_number_problem_l1769_176916

theorem number_problem : ∃! x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1769_176916


namespace NUMINAMATH_CALUDE_jenna_stamps_problem_l1769_176901

theorem jenna_stamps_problem :
  Nat.gcd 1260 1470 = 210 := by
  sorry

end NUMINAMATH_CALUDE_jenna_stamps_problem_l1769_176901


namespace NUMINAMATH_CALUDE_overall_average_score_l1769_176909

theorem overall_average_score 
  (morning_avg : ℝ) 
  (evening_avg : ℝ) 
  (student_ratio : ℚ) 
  (h_morning_avg : morning_avg = 82) 
  (h_evening_avg : evening_avg = 75) 
  (h_student_ratio : student_ratio = 5 / 3) :
  let m := (student_ratio * evening_students : ℝ)
  let e := evening_students
  let total_students := m + e
  let total_score := morning_avg * m + evening_avg * e
  total_score / total_students = 79.375 :=
by
  sorry

#check overall_average_score

end NUMINAMATH_CALUDE_overall_average_score_l1769_176909


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l1769_176944

theorem angle_in_second_quadrant : ∃ θ : Real, 
  θ = -10 * Real.pi / 3 ∧ 
  π / 2 < θ % (2 * π) ∧ 
  θ % (2 * π) < π :=
sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l1769_176944


namespace NUMINAMATH_CALUDE_angle_property_equivalence_l1769_176997

theorem angle_property_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + 2 * (1 - x)^2 * Real.sin θ > 0) ↔
  (π / 24 < θ ∧ θ < 11 * π / 24) ∨ (25 * π / 24 < θ ∧ θ < 47 * π / 24) :=
by sorry

end NUMINAMATH_CALUDE_angle_property_equivalence_l1769_176997


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1769_176948

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (3*x - 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₂ + a₄ = 136 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1769_176948


namespace NUMINAMATH_CALUDE_girls_fraction_is_half_l1769_176930

/-- Represents a school with a given number of students and boy-to-girl ratio -/
structure School where
  total_students : ℕ
  boy_ratio : ℕ
  girl_ratio : ℕ

/-- Calculates the number of girls in a school -/
def girls_count (s : School) : ℕ :=
  s.total_students * s.girl_ratio /(s.boy_ratio + s.girl_ratio)

/-- The fraction of girls at a dance attended by students from two schools -/
def girls_fraction (school_a : School) (school_b : School) : ℚ :=
  (girls_count school_a + girls_count school_b : ℚ) /
  (school_a.total_students + school_b.total_students)

theorem girls_fraction_is_half :
  let school_a : School := ⟨300, 3, 2⟩
  let school_b : School := ⟨240, 3, 5⟩
  girls_fraction school_a school_b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_girls_fraction_is_half_l1769_176930


namespace NUMINAMATH_CALUDE_g_uniqueness_l1769_176917

/-- The functional equation for g -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y

theorem g_uniqueness (g : ℝ → ℝ) (h1 : g 1 = 1) (h2 : FunctionalEquation g) :
    ∀ x : ℝ, g x = 4^x - 3^x := by
  sorry

end NUMINAMATH_CALUDE_g_uniqueness_l1769_176917


namespace NUMINAMATH_CALUDE_cubes_not_touching_foil_l1769_176982

/-- Represents the dimensions of a rectangular prism --/
structure PrismDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of cubes in a rectangular prism given its dimensions --/
def cubesInPrism (d : PrismDimensions) : ℕ := d.width * d.length * d.height

/-- Theorem: The number of cubes not touching tin foil in the given prism is 128 --/
theorem cubes_not_touching_foil (prism_width : ℕ) (inner_prism : PrismDimensions) : 
  prism_width = 10 →
  inner_prism.width = 2 * inner_prism.length →
  inner_prism.width = 2 * inner_prism.height →
  inner_prism.width ≤ prism_width - 2 →
  cubesInPrism inner_prism = 128 := by
  sorry

#check cubes_not_touching_foil

end NUMINAMATH_CALUDE_cubes_not_touching_foil_l1769_176982


namespace NUMINAMATH_CALUDE_monotonic_decreasing_range_l1769_176911

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := (2 * k - 1) * x + 1

-- State the theorem
theorem monotonic_decreasing_range (k : ℝ) :
  (∀ x y : ℝ, x < y → f k x > f k y) →
  k < (1 / 2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_range_l1769_176911


namespace NUMINAMATH_CALUDE_subtraction_value_problem_l1769_176902

theorem subtraction_value_problem (x y : ℝ) : 
  ((x - 5) / 7 = 7) → ((x - y) / 10 = 3) → y = 24 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_value_problem_l1769_176902


namespace NUMINAMATH_CALUDE_jose_lemons_needed_l1769_176947

/-- The number of lemons needed for a given number of dozen cupcakes -/
def lemons_needed (dozen_cupcakes : ℕ) : ℕ :=
  let tablespoons_per_dozen := 12
  let tablespoons_per_lemon := 4
  (dozen_cupcakes * tablespoons_per_dozen) / tablespoons_per_lemon

theorem jose_lemons_needed : lemons_needed 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jose_lemons_needed_l1769_176947


namespace NUMINAMATH_CALUDE_f_value_at_107_5_l1769_176910

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_value_at_107_5 (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_periodic : ∀ x, f (x + 3) = -1 / f x)
  (h_interval : ∀ x ∈ Set.Icc (-3) (-2), f x = 4 * x) :
  f 107.5 = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_107_5_l1769_176910


namespace NUMINAMATH_CALUDE_units_digit_of_sum_l1769_176949

theorem units_digit_of_sum (a b : ℕ) : (24^4 + 42^4) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_l1769_176949


namespace NUMINAMATH_CALUDE_perimeter_to_hypotenuse_ratio_l1769_176923

/-- Right triangle ABC with altitude CD to hypotenuse AB and circle ω with CD as diameter -/
structure RightTriangleWithCircle where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point D on hypotenuse AB -/
  D : ℝ × ℝ
  /-- Center of circle ω -/
  O : ℝ × ℝ
  /-- Point I outside the triangle -/
  I : ℝ × ℝ
  /-- ABC is a right triangle with right angle at C -/
  is_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  /-- AC = 15 -/
  ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 15
  /-- BC = 20 -/
  bc_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 20
  /-- CD is perpendicular to AB -/
  cd_perpendicular : (D.1 - C.1) * (B.1 - A.1) + (D.2 - C.2) * (B.2 - A.2) = 0
  /-- D is on AB -/
  d_on_ab : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  /-- O is the midpoint of CD -/
  o_midpoint : O = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  /-- AI is tangent to circle ω -/
  ai_tangent : Real.sqrt ((I.1 - A.1)^2 + (I.2 - A.2)^2) * Real.sqrt ((I.1 - O.1)^2 + (I.2 - O.2)^2) = (I.1 - A.1) * (I.1 - O.1) + (I.2 - A.2) * (I.2 - O.2)
  /-- BI is tangent to circle ω -/
  bi_tangent : Real.sqrt ((I.1 - B.1)^2 + (I.2 - B.2)^2) * Real.sqrt ((I.1 - O.1)^2 + (I.2 - O.2)^2) = (I.1 - B.1) * (I.1 - O.1) + (I.2 - B.2) * (I.2 - O.2)

/-- The ratio of the perimeter of triangle ABI to the length of AB is 5/2 -/
theorem perimeter_to_hypotenuse_ratio (t : RightTriangleWithCircle) :
  let ab_length := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let ai_length := Real.sqrt ((t.I.1 - t.A.1)^2 + (t.I.2 - t.A.2)^2)
  let bi_length := Real.sqrt ((t.I.1 - t.B.1)^2 + (t.I.2 - t.B.2)^2)
  (ai_length + bi_length + ab_length) / ab_length = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_to_hypotenuse_ratio_l1769_176923


namespace NUMINAMATH_CALUDE_point_in_intersection_l1769_176967

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set A
def A (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + m > 0}

-- Define set B
def B (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - n > 0}

-- Define the complement of B with respect to U
def C_U_B (n : ℝ) : Set (ℝ × ℝ) := U \ B n

-- Define point P
def P : ℝ × ℝ := (2, 3)

-- State the theorem
theorem point_in_intersection (m n : ℝ) :
  P ∈ A m ∩ C_U_B n ↔ m > -1 ∧ n ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_point_in_intersection_l1769_176967


namespace NUMINAMATH_CALUDE_tank_water_supply_l1769_176995

theorem tank_water_supply (C V : ℝ) 
  (h1 : C = 15 * (V + 10))
  (h2 : C = 12 * (V + 20)) :
  C / V = 20 := by
sorry

end NUMINAMATH_CALUDE_tank_water_supply_l1769_176995


namespace NUMINAMATH_CALUDE_optimal_tax_theorem_l1769_176955

/-- Market model with linear demand and supply functions, and a per-unit tax --/
structure MarketModel where
  -- Demand function: Qd = a - bP
  a : ℝ
  b : ℝ
  -- Supply function: Qs = cP + d
  c : ℝ
  d : ℝ
  -- Elasticity ratio at equilibrium
  elasticity_ratio : ℝ
  -- Tax amount
  tax : ℝ
  -- Producer price after tax
  producer_price : ℝ

/-- Finds the optimal tax rate and resulting revenue for a given market model --/
def optimal_tax_and_revenue (model : MarketModel) : ℝ × ℝ :=
  sorry

/-- The main theorem stating the optimal tax and revenue for the given market conditions --/
theorem optimal_tax_theorem (model : MarketModel) :
  model.a = 688 ∧
  model.b = 4 ∧
  model.elasticity_ratio = 1.5 ∧
  model.tax = 90 ∧
  model.producer_price = 64 →
  optimal_tax_and_revenue model = (54, 10800) :=
sorry

end NUMINAMATH_CALUDE_optimal_tax_theorem_l1769_176955


namespace NUMINAMATH_CALUDE_complex_number_location_l1769_176960

theorem complex_number_location (z : ℂ) (h : (3 - 2*I)*z = 4 + 3*I) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end NUMINAMATH_CALUDE_complex_number_location_l1769_176960


namespace NUMINAMATH_CALUDE_locus_and_circle_existence_l1769_176927

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 18
def C₂ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 2

-- Define the locus of the center of circle M
def locus (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the circle centered at the origin
def origin_circle (x y : ℝ) : Prop := x^2 + y^2 = 8 / 3

-- Define the tangency and orthogonality conditions
def tangent_intersects_locus (t m n : ℝ × ℝ) : Prop :=
  locus m.1 m.2 ∧ locus n.1 n.2 ∧ 
  (∃ k b : ℝ, t.2 = k * t.1 + b ∧ origin_circle t.1 t.2)

def orthogonal (o m n : ℝ × ℝ) : Prop :=
  (m.1 - o.1) * (n.1 - o.1) + (m.2 - o.2) * (n.2 - o.2) = 0

-- Main theorem
theorem locus_and_circle_existence :
  (∀ x y : ℝ, C₁ x y ∨ C₂ x y → 
    ∃ m : ℝ × ℝ, locus m.1 m.2 ∧
    (∀ t : ℝ × ℝ, origin_circle t.1 t.2 → 
      ∃ n : ℝ × ℝ, tangent_intersects_locus t m n ∧ 
        orthogonal (0, 0) m n)) :=
sorry

end NUMINAMATH_CALUDE_locus_and_circle_existence_l1769_176927


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l1769_176924

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  QS : ℝ
  PS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem: The volume of the specific tetrahedron is 24/√737 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    QR := 5,
    QS := 5,
    PS := 4,
    RS := 15/4 * Real.sqrt 2
  }
  tetrahedronVolume t = 24 / Real.sqrt 737 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l1769_176924


namespace NUMINAMATH_CALUDE_cos_angle_with_z_axis_l1769_176990

/-- Given a point Q in the first octant of 3D space, prove that if the cosine of the angle between OQ
    and the x-axis is 2/5, and the cosine of the angle between OQ and the y-axis is 1/4, then the
    cosine of the angle between OQ and the z-axis is √(311) / 20. -/
theorem cos_angle_with_z_axis (Q : ℝ × ℝ × ℝ) 
    (h_pos : Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0)
    (h_cos_alpha : Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = 2/5)
    (h_cos_beta : Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = 1/4) :
  Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2) = Real.sqrt 311 / 20 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_with_z_axis_l1769_176990


namespace NUMINAMATH_CALUDE_jake_initial_balloons_l1769_176926

/-- The number of balloons Jake initially brought to the park -/
def jake_initial : ℕ := 2

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 6

/-- The number of additional balloons Jake bought at the park -/
def jake_additional : ℕ := 3

theorem jake_initial_balloons :
  jake_initial = 2 :=
by
  have h1 : allan_balloons = jake_initial + jake_additional + 1 :=
    sorry
  sorry

end NUMINAMATH_CALUDE_jake_initial_balloons_l1769_176926


namespace NUMINAMATH_CALUDE_students_not_participating_l1769_176921

/-- Given a class with the following properties:
  * There are 15 students in total
  * 7 students participate in mathematical modeling
  * 9 students participate in computer programming
  * 3 students participate in both activities
  This theorem proves that 2 students do not participate in either activity. -/
theorem students_not_participating (total : ℕ) (modeling : ℕ) (programming : ℕ) (both : ℕ) :
  total = 15 →
  modeling = 7 →
  programming = 9 →
  both = 3 →
  total - (modeling + programming - both) = 2 := by
  sorry

end NUMINAMATH_CALUDE_students_not_participating_l1769_176921


namespace NUMINAMATH_CALUDE_right_triangle_sin_a_l1769_176928

theorem right_triangle_sin_a (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : C = Real.pi / 2) (h3 : Real.cos B = 1 / 2) : Real.sin A = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_a_l1769_176928


namespace NUMINAMATH_CALUDE_intersection_slope_problem_l1769_176929

/-- Given two lines intersecting at (40, 30), where one line has a slope of 6
    and the distance between their x-intercepts is 10,
    prove that the slope of the other line is 2. -/
theorem intersection_slope_problem (m : ℝ) : 
  let line1 : ℝ → ℝ := λ x => m * x - 40 * m + 30
  let line2 : ℝ → ℝ := λ x => 6 * x - 210
  let x_intercept1 : ℝ := (40 * m - 30) / m
  let x_intercept2 : ℝ := 35
  (∃ x y, line1 x = line2 x ∧ x = 40 ∧ y = 30) →  -- Lines intersect at (40, 30)
  |x_intercept1 - x_intercept2| = 10 →           -- Distance between x-intercepts is 10
  m = 2                                          -- Slope of the first line is 2
  := by sorry

end NUMINAMATH_CALUDE_intersection_slope_problem_l1769_176929


namespace NUMINAMATH_CALUDE_solve_walnuts_problem_l1769_176976

def walnuts_problem (initial_walnuts boy_gathered girl_gathered girl_ate final_walnuts : ℕ) : Prop :=
  ∃ (dropped : ℕ),
    initial_walnuts + boy_gathered - dropped + girl_gathered - girl_ate = final_walnuts

theorem solve_walnuts_problem :
  walnuts_problem 12 6 5 2 20 → 
  ∃ (dropped : ℕ), dropped = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_walnuts_problem_l1769_176976


namespace NUMINAMATH_CALUDE_fish_feeding_cost_l1769_176952

/-- Calculates the total cost to feed fish for 30 days given the specified conditions --/
theorem fish_feeding_cost :
  let goldfish_count : ℕ := 50
  let koi_count : ℕ := 30
  let guppies_count : ℕ := 20
  let goldfish_food : ℚ := 1.5
  let koi_food : ℚ := 2.5
  let guppies_food : ℚ := 0.75
  let goldfish_special_food_ratio : ℚ := 0.25
  let koi_special_food_ratio : ℚ := 0.4
  let guppies_special_food_ratio : ℚ := 0.1
  let special_food_cost_goldfish : ℚ := 3
  let special_food_cost_others : ℚ := 4
  let regular_food_cost : ℚ := 2
  let days : ℕ := 30

  (goldfish_count * goldfish_food * (goldfish_special_food_ratio * special_food_cost_goldfish +
    (1 - goldfish_special_food_ratio) * regular_food_cost) +
   koi_count * koi_food * (koi_special_food_ratio * special_food_cost_others +
    (1 - koi_special_food_ratio) * regular_food_cost) +
   guppies_count * guppies_food * (guppies_special_food_ratio * special_food_cost_others +
    (1 - guppies_special_food_ratio) * regular_food_cost)) * days = 12375 :=
by sorry


end NUMINAMATH_CALUDE_fish_feeding_cost_l1769_176952


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_8_is_correct_l1769_176981

/-- The probability of getting a sum greater than 8 when tossing two dice -/
def prob_sum_greater_than_8 : ℚ := 5/18

/-- The total number of possible outcomes when tossing two dice -/
def total_outcomes : ℕ := 6 * 6

/-- The number of ways to get a sum of 8 or less when tossing two dice -/
def ways_sum_8_or_less : ℕ := 26

theorem prob_sum_greater_than_8_is_correct :
  prob_sum_greater_than_8 = 1 - (ways_sum_8_or_less : ℚ) / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_8_is_correct_l1769_176981


namespace NUMINAMATH_CALUDE_g_of_five_l1769_176991

/-- Given a function g : ℝ → ℝ satisfying 3g(x) + 4g(1 - x) = 6x^2 for all real x, prove that g(5) = -66/7 -/
theorem g_of_five (g : ℝ → ℝ) (h : ∀ x : ℝ, 3 * g x + 4 * g (1 - x) = 6 * x^2) : g 5 = -66/7 := by
  sorry

end NUMINAMATH_CALUDE_g_of_five_l1769_176991


namespace NUMINAMATH_CALUDE_franks_problems_per_type_l1769_176973

/-- The number of math problems composed by Bill. -/
def bills_problems : ℕ := 20

/-- The number of math problems composed by Ryan. -/
def ryans_problems : ℕ := 2 * bills_problems

/-- The number of math problems composed by Frank. -/
def franks_problems : ℕ := 3 * ryans_problems

/-- The number of different types of math problems each person composes. -/
def problem_types : ℕ := 4

/-- Theorem stating that Frank composes 30 problems of each type. -/
theorem franks_problems_per_type :
  franks_problems / problem_types = 30 := by sorry

end NUMINAMATH_CALUDE_franks_problems_per_type_l1769_176973


namespace NUMINAMATH_CALUDE_no_base_with_final_digit_four_l1769_176964

theorem no_base_with_final_digit_four : 
  ∀ b : ℕ, 3 ≤ b ∧ b ≤ 10 → ¬(981 % b = 4) :=
by sorry

end NUMINAMATH_CALUDE_no_base_with_final_digit_four_l1769_176964


namespace NUMINAMATH_CALUDE_divisible_by_three_l1769_176954

def five_digit_number (n : Nat) : Nat :=
  52000 + n * 100 + 48

theorem divisible_by_three (n : Nat) : 
  n < 10 → (five_digit_number n % 3 = 0 ↔ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_l1769_176954


namespace NUMINAMATH_CALUDE_max_value_theorem_l1769_176959

theorem max_value_theorem (t x1 x2 : ℝ) : 
  t > 2 → x2 > x1 → x1 > 0 → 
  (Real.exp x1 - x1 = t) → (x2 - Real.log x2 = t) →
  (∃ (c : ℝ), c = Real.log t / (x2 - x1) ∧ c ≤ 1 / Real.exp 1 ∧ 
   ∀ (y1 y2 : ℝ), y2 > y1 → y1 > 0 → 
   (Real.exp y1 - y1 = t) → (y2 - Real.log y2 = t) →
   Real.log t / (y2 - y1) ≤ c) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1769_176959


namespace NUMINAMATH_CALUDE_inequalities_proof_l1769_176968

theorem inequalities_proof (a b c d : ℝ) 
  (h1 : b > a) (h2 : a > 1) (h3 : c < d) (h4 : d < -1) : 
  (1/b < 1/a ∧ 1/a < 1) ∧ 
  (1/c > 1/d ∧ 1/d > -1) ∧ 
  (a*d > b*c) := by
sorry


end NUMINAMATH_CALUDE_inequalities_proof_l1769_176968


namespace NUMINAMATH_CALUDE_problem_statement_l1769_176945

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : (12 : ℝ) ^ x = (18 : ℝ) ^ y) (h2 : (12 : ℝ) ^ x = 6 ^ (x * y)) :
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1769_176945


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1769_176980

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1769_176980


namespace NUMINAMATH_CALUDE_matrix_value_proof_l1769_176963

def matrix_operation (a b c d : ℤ) : ℤ := a * c - b * d

theorem matrix_value_proof : matrix_operation 2 3 4 5 = -7 := by
  sorry

end NUMINAMATH_CALUDE_matrix_value_proof_l1769_176963


namespace NUMINAMATH_CALUDE_no_solution_exists_l1769_176970

theorem no_solution_exists (x y z : ℝ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  x * (y + z) + y * (z + x) = y * (z + x) + z * (x + y) → 
  False :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1769_176970


namespace NUMINAMATH_CALUDE_puppies_per_cage_l1769_176994

def initial_puppies : ℕ := 56
def sold_puppies : ℕ := 24
def num_cages : ℕ := 8

theorem puppies_per_cage :
  (initial_puppies - sold_puppies) / num_cages = 4 :=
by sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l1769_176994


namespace NUMINAMATH_CALUDE_number_divided_by_005_equals_900_l1769_176961

theorem number_divided_by_005_equals_900 (x : ℝ) : x / 0.05 = 900 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_005_equals_900_l1769_176961


namespace NUMINAMATH_CALUDE_expression_simplification_l1769_176937

def x : ℚ := -2
def y : ℚ := 1/2

theorem expression_simplification :
  (x + 4*y) * (x - 4*y) + (x - 4*y)^2 - (4*x^2 - x*y) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1769_176937


namespace NUMINAMATH_CALUDE_cows_equivalent_to_buffaloes_or_oxen_l1769_176939

-- Define the variables
variable (B : ℕ) -- Daily fodder consumption of a buffalo
variable (C : ℕ) -- Daily fodder consumption of a cow
variable (O : ℕ) -- Daily fodder consumption of an ox
variable (F : ℕ) -- Total available fodder

-- Define the conditions
axiom buffalo_ox_equiv : 3 * B = 2 * O
axiom initial_fodder : F = (15 * B + 8 * O + 24 * C) * 48
axiom additional_cattle : F = (30 * B + 64 * C) * 24

-- The theorem to prove
theorem cows_equivalent_to_buffaloes_or_oxen : ∃ x : ℕ, x = 2 ∧ 3 * B = x * C := by
  sorry

end NUMINAMATH_CALUDE_cows_equivalent_to_buffaloes_or_oxen_l1769_176939


namespace NUMINAMATH_CALUDE_pig_to_cow_ratio_l1769_176940

/-- Represents the farm scenario with cows and pigs -/
structure Farm where
  numCows : ℕ
  revenueCow : ℕ
  revenuePig : ℕ
  totalRevenue : ℕ

/-- Calculates the number of pigs based on the farm data -/
def calculatePigs (farm : Farm) : ℕ :=
  (farm.totalRevenue - farm.numCows * farm.revenueCow) / farm.revenuePig

/-- Theorem stating that the ratio of pigs to cows is 4:1 -/
theorem pig_to_cow_ratio (farm : Farm)
  (h1 : farm.numCows = 20)
  (h2 : farm.revenueCow = 800)
  (h3 : farm.revenuePig = 400)
  (h4 : farm.totalRevenue = 48000) :
  (calculatePigs farm) / farm.numCows = 4 := by
  sorry

#check pig_to_cow_ratio

end NUMINAMATH_CALUDE_pig_to_cow_ratio_l1769_176940


namespace NUMINAMATH_CALUDE_car_distance_problem_l1769_176965

/-- Calculates the distance between two cars given their speeds and overtake time -/
def distance_between_cars (red_speed black_speed overtake_time : ℝ) : ℝ :=
  (black_speed - red_speed) * overtake_time

/-- Theorem stating that the distance between the cars is 30 miles -/
theorem car_distance_problem :
  let red_speed : ℝ := 40
  let black_speed : ℝ := 50
  let overtake_time : ℝ := 3
  distance_between_cars red_speed black_speed overtake_time = 30 := by
sorry


end NUMINAMATH_CALUDE_car_distance_problem_l1769_176965


namespace NUMINAMATH_CALUDE_sara_movie_rental_l1769_176934

def movie_problem (theater_ticket_price : ℚ) (num_tickets : ℕ) (bought_movie_price : ℚ) (total_spent : ℚ) : Prop :=
  let theater_total : ℚ := theater_ticket_price * num_tickets
  let rental_price : ℚ := total_spent - theater_total - bought_movie_price
  rental_price = 159/100

theorem sara_movie_rental :
  movie_problem (1062/100) 2 (1395/100) (3678/100) :=
by
  sorry

end NUMINAMATH_CALUDE_sara_movie_rental_l1769_176934


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1769_176936

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem smallest_n_satisfying_conditions : 
  ∃ (n : ℕ), is_three_digit n ∧ 
             (9 ∣ (n + 6)) ∧ 
             (6 ∣ (n - 4)) ∧ 
             (∀ m, is_three_digit m → (9 ∣ (m + 6)) → (6 ∣ (m - 4)) → n ≤ m) ∧
             n = 112 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l1769_176936


namespace NUMINAMATH_CALUDE_correct_oranges_to_put_back_l1769_176906

/-- Represents the fruit selection problem -/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  desired_avg_price : ℚ

/-- Calculates the number of oranges to put back -/
def oranges_to_put_back (fs : FruitSelection) : ℕ :=
  sorry

/-- Theorem stating the correct number of oranges to put back -/
theorem correct_oranges_to_put_back (fs : FruitSelection) 
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 10)
  (h4 : fs.initial_avg_price = 56/100)
  (h5 : fs.desired_avg_price = 50/100) :
  oranges_to_put_back fs = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_to_put_back_l1769_176906


namespace NUMINAMATH_CALUDE_at_home_workforce_trend_l1769_176975

/-- Represents the percentage of working adults in Parkertown working at home for a given year -/
def AtHomeWorkforce : ℕ → ℚ
  | 1990 => 12/100
  | 1995 => 15/100
  | 2000 => 14/100
  | 2005 => 28/100
  | _ => 0

/-- The trend of the at-home workforce in Parkertown from 1990 to 2005 -/
theorem at_home_workforce_trend :
  AtHomeWorkforce 1995 > AtHomeWorkforce 1990 ∧
  AtHomeWorkforce 2000 < AtHomeWorkforce 1995 ∧
  AtHomeWorkforce 2005 > AtHomeWorkforce 2000 ∧
  (AtHomeWorkforce 2005 - AtHomeWorkforce 2000) > (AtHomeWorkforce 1995 - AtHomeWorkforce 1990) :=
by sorry

end NUMINAMATH_CALUDE_at_home_workforce_trend_l1769_176975


namespace NUMINAMATH_CALUDE_cube_of_sqrt_three_l1769_176903

theorem cube_of_sqrt_three (x : ℝ) : 
  Real.sqrt (x + 3) = 3 → (x + 3)^3 = 729 := by
sorry

end NUMINAMATH_CALUDE_cube_of_sqrt_three_l1769_176903


namespace NUMINAMATH_CALUDE_prob_different_colors_is_two_thirds_l1769_176931

/-- Represents the possible colors for socks -/
inductive SockColor
| Red
| Blue

/-- Represents the possible colors for headbands -/
inductive HeadbandColor
| Red
| Blue
| Green

/-- The probability of choosing different colors for socks and headbands -/
def prob_different_colors : ℚ :=
  2 / 3

theorem prob_different_colors_is_two_thirds :
  prob_different_colors = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_prob_different_colors_is_two_thirds_l1769_176931


namespace NUMINAMATH_CALUDE_susie_investment_l1769_176907

/-- Proves that Susie's investment at Safe Savings Bank is 0 --/
theorem susie_investment (total_investment : ℝ) (safe_rate : ℝ) (risky_rate : ℝ) (total_after_year : ℝ) 
  (h1 : total_investment = 2000)
  (h2 : safe_rate = 0.04)
  (h3 : risky_rate = 0.06)
  (h4 : total_after_year = 2120)
  (h5 : ∀ x : ℝ, x * (1 + safe_rate) + (total_investment - x) * (1 + risky_rate) = total_after_year) :
  ∃ x : ℝ, x = 0 ∧ x * (1 + safe_rate) + (total_investment - x) * (1 + risky_rate) = total_after_year :=
sorry

end NUMINAMATH_CALUDE_susie_investment_l1769_176907


namespace NUMINAMATH_CALUDE_fraction_calculation_l1769_176938

theorem fraction_calculation (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  ((x + 1) / (y - 1)) / ((y + 2) / (x - 2)) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1769_176938


namespace NUMINAMATH_CALUDE_minimal_ratio_S₁_S₂_l1769_176983

noncomputable def S₁ (α : Real) : Real :=
  4 - (2 * Real.sqrt 2 / Real.cos α)

noncomputable def S₂ (α : Real) : Real :=
  ((Real.sqrt 2 * (Real.sin α + Real.cos α) - 1)^2) / (2 * Real.sin α * Real.cos α)

theorem minimal_ratio_S₁_S₂ :
  ∃ (α₁ α₂ : Real), 
    0 ≤ α₁ ∧ α₁ ≤ Real.pi/12 ∧
    Real.pi/12 ≤ α₂ ∧ α₂ ≤ 5*Real.pi/12 ∧
    S₁ α₁ / (8 - S₁ α₁) = 1/7 ∧
    S₂ α₂ / (8 - S₂ α₂) = 1/7 ∧
    ∀ (β γ : Real), 
      (0 ≤ β ∧ β ≤ Real.pi/12 → S₁ β / (8 - S₁ β) ≥ 1/7) ∧
      (Real.pi/12 ≤ γ ∧ γ ≤ 5*Real.pi/12 → S₂ γ / (8 - S₂ γ) ≥ 1/7) :=
by sorry

end NUMINAMATH_CALUDE_minimal_ratio_S₁_S₂_l1769_176983


namespace NUMINAMATH_CALUDE_largest_integer_in_sequence_l1769_176912

theorem largest_integer_in_sequence (n : ℕ) (start : ℤ) (h1 : n = 40) (h2 : start = -11) :
  (start + (n - 1) : ℤ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_in_sequence_l1769_176912


namespace NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_for_ln_x_plus_one_negative_l1769_176943

theorem x_negative_necessary_not_sufficient_for_ln_x_plus_one_negative :
  (∀ x : ℝ, Real.log (x + 1) < 0 → x < 0) ∧
  (∃ x : ℝ, x < 0 ∧ Real.log (x + 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_for_ln_x_plus_one_negative_l1769_176943


namespace NUMINAMATH_CALUDE_furniture_shop_cost_price_l1769_176913

theorem furniture_shop_cost_price (selling_price : ℕ) (markup_percentage : ℕ) : 
  selling_price = 1000 → markup_percentage = 100 → 
  ∃ (cost_price : ℕ), cost_price * (100 + markup_percentage) / 100 = selling_price ∧ cost_price = 500 := by
sorry

end NUMINAMATH_CALUDE_furniture_shop_cost_price_l1769_176913


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1769_176979

theorem geometric_sequence_fourth_term (a b c x : ℝ) : 
  a ≠ 0 → b / a = c / b → c / b * c = x → a = 0.001 → b = 0.02 → c = 0.4 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1769_176979


namespace NUMINAMATH_CALUDE_lucy_apples_per_week_l1769_176950

/-- Given the following conditions:
  - Chandler eats 23 apples per week
  - They order 168 apples per month
  - There are 4 weeks in a month
  Prove that Lucy can eat 19 apples per week. -/
theorem lucy_apples_per_week :
  ∀ (chandler_apples_per_week : ℕ) 
    (total_apples_per_month : ℕ) 
    (weeks_per_month : ℕ),
  chandler_apples_per_week = 23 →
  total_apples_per_month = 168 →
  weeks_per_month = 4 →
  ∃ (lucy_apples_per_week : ℕ),
    lucy_apples_per_week = 19 ∧
    lucy_apples_per_week * weeks_per_month + 
    chandler_apples_per_week * weeks_per_month = 
    total_apples_per_month :=
by sorry

end NUMINAMATH_CALUDE_lucy_apples_per_week_l1769_176950


namespace NUMINAMATH_CALUDE_same_solution_implies_m_half_l1769_176974

theorem same_solution_implies_m_half 
  (h1 : ∃ x, 4*x + 2*m = 3*x + 1)
  (h2 : ∃ x, 3*x + 2*m = 6*x + 1)
  (h3 : ∃ x, (4*x + 2*m = 3*x + 1) ∧ (3*x + 2*m = 6*x + 1)) :
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_m_half_l1769_176974


namespace NUMINAMATH_CALUDE_sin_double_alpha_l1769_176984

/-- Given that the terminal side of angle α intersects the unit circle at point P(-√3/2, 1/2),
    prove that sin 2α = -√3/2 -/
theorem sin_double_alpha (α : Real) 
  (h : ∃ P : Real × Real, P.1 = -Real.sqrt 3 / 2 ∧ P.2 = 1 / 2 ∧ 
       P.1^2 + P.2^2 = 1 ∧ P.1 = Real.cos α ∧ P.2 = Real.sin α) : 
  Real.sin (2 * α) = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_double_alpha_l1769_176984


namespace NUMINAMATH_CALUDE_milk_packs_per_set_l1769_176914

/-- The number of packs in each set of milk -/
def packs_per_set : ℕ := sorry

/-- The cost of a set of milk packs in dollars -/
def cost_per_set : ℚ := 2.5

/-- The cost of an individual milk pack in dollars -/
def cost_per_pack : ℚ := 1.3

/-- The total savings from buying ten sets in dollars -/
def total_savings : ℚ := 1

theorem milk_packs_per_set :
  packs_per_set = 2 ∧
  10 * cost_per_set + total_savings = 10 * packs_per_set * cost_per_pack :=
sorry

end NUMINAMATH_CALUDE_milk_packs_per_set_l1769_176914


namespace NUMINAMATH_CALUDE_sodium_hydride_requirement_l1769_176942

-- Define the chemical reaction
structure ChemicalReaction where
  naH : ℚ  -- moles of Sodium hydride
  h2o : ℚ  -- moles of Water
  naOH : ℚ -- moles of Sodium hydroxide
  h2 : ℚ   -- moles of Hydrogen

-- Define the balanced equation
def balancedReaction (r : ChemicalReaction) : Prop :=
  r.naH = r.h2o ∧ r.naH = r.naOH ∧ r.naH = r.h2

-- Theorem statement
theorem sodium_hydride_requirement 
  (r : ChemicalReaction) 
  (h1 : r.naOH = 2) 
  (h2 : r.h2 = 2) 
  (h3 : r.h2o = 2) 
  (h4 : balancedReaction r) : 
  r.naH = 2 := by
  sorry

end NUMINAMATH_CALUDE_sodium_hydride_requirement_l1769_176942


namespace NUMINAMATH_CALUDE_tens_digit_of_subtraction_l1769_176904

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hun_less_than_tens : hundreds = tens - 3
  tens_double_units : tens = 2 * units
  is_three_digit : hundreds ≥ 1 ∧ hundreds ≤ 9

def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

def ThreeDigitNumber.reversed (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

theorem tens_digit_of_subtraction (n : ThreeDigitNumber) :
  (n.toNat - n.reversed) / 10 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_subtraction_l1769_176904


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l1769_176908

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  base_diff : ℝ
  midpoint_ratio : ℝ × ℝ
  equal_area_segment : ℝ

/-- The trapezoid satisfying the problem conditions -/
def problem_trapezoid : Trapezoid where
  base_diff := 120
  midpoint_ratio := (3, 4)
  equal_area_segment := x
  where x : ℝ := sorry  -- The actual value of x will be determined in the proof

/-- The theorem to be proved -/
theorem trapezoid_segment_length (t : Trapezoid) : 
  t = problem_trapezoid → ⌊(t.equal_area_segment^2) / 120⌋ = 270 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_length_l1769_176908


namespace NUMINAMATH_CALUDE_cone_height_l1769_176987

theorem cone_height (r : ℝ) (lateral_area : ℝ) (h : ℝ) : 
  r = 1 → lateral_area = 2 * Real.pi → h = Real.sqrt 3 → 
  lateral_area = Real.pi * r * Real.sqrt (h^2 + r^2) :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l1769_176987


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1769_176958

theorem arithmetic_computation : 2 + 5 * 3 - 4 + 6 * 2 / 3 = 17 := by sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1769_176958


namespace NUMINAMATH_CALUDE_geometric_sequence_a1_l1769_176962

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (q : ℚ), ∀ (n : ℕ), a (n + 1) = a n * q

theorem geometric_sequence_a1 (a : ℕ → ℚ) :
  geometric_sequence a →
  a 2 * a 5 = 2 * a 3 →
  (a 4 + a 6) / 2 = 5/4 →
  a 1 = 16 ∨ a 1 = -16 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a1_l1769_176962


namespace NUMINAMATH_CALUDE_min_printers_equal_expenditure_l1769_176988

def printer_costs : List Nat := [400, 350, 500, 200]

theorem min_printers_equal_expenditure :
  let total_cost := Nat.lcm (Nat.lcm (Nat.lcm 400 350) 500) 200
  let num_printers := List.sum (List.map (λ cost => total_cost / cost) printer_costs)
  num_printers = 173 ∧
  ∀ (n : Nat), n < num_printers →
    ∃ (cost : Nat), cost ∈ printer_costs ∧ (n * cost) % total_cost ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_printers_equal_expenditure_l1769_176988


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l1769_176957

theorem least_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * |x| - 2 > 13) → x ≥ -6 :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l1769_176957


namespace NUMINAMATH_CALUDE_instantaneous_speed_at_t_1_l1769_176989

/-- The displacement function for the particle's motion --/
def s (t : ℝ) : ℝ := 2 * t^3

/-- The velocity function (derivative of displacement) --/
def v (t : ℝ) : ℝ := 6 * t^2

theorem instantaneous_speed_at_t_1 :
  v 1 = 6 := by sorry

end NUMINAMATH_CALUDE_instantaneous_speed_at_t_1_l1769_176989


namespace NUMINAMATH_CALUDE_set_01_proper_subset_N_l1769_176905

-- Define the set of natural numbers
def N : Set ℕ := Set.univ

-- Define the set {0,1}
def set_01 : Set ℕ := {0, 1}

-- Theorem to prove
theorem set_01_proper_subset_N : set_01 ⊂ N := by sorry

end NUMINAMATH_CALUDE_set_01_proper_subset_N_l1769_176905


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l1769_176919

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 4}

theorem complement_of_N_in_M : M \ N = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l1769_176919


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l1769_176993

theorem arithmetic_sequence_count (start end_ diff : ℕ) (h1 : start = 24) (h2 : end_ = 162) (h3 : diff = 6) :
  (end_ - start) / diff + 1 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l1769_176993


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1769_176922

/-- The function f(x) = 1 + 2a^(x-1) has a fixed point at (1, 3), where a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 1 + 2 * a^(x - 1)
  f 1 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1769_176922


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1769_176999

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  (1 / (a + 2*b)) + (1 / (b + 2*c)) + (1 / (c + 2*a)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1769_176999


namespace NUMINAMATH_CALUDE_second_candidate_votes_l1769_176946

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℚ) :
  total_votes = 2400 →
  first_candidate_percentage = 80 / 100 →
  (1 - first_candidate_percentage) * total_votes = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_second_candidate_votes_l1769_176946


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_neg_one_range_of_a_for_inequality_l1769_176932

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x + 2|

-- Part 1
theorem solution_set_for_a_eq_neg_one (x : ℝ) :
  (f (-1) x ≥ x + 5) ↔ (x ≤ -2 ∨ x ≥ 4) := by sorry

-- Part 2
theorem range_of_a_for_inequality (a : ℝ) (h : a < 2) :
  (∀ x ∈ Set.Ioo (-5) (-3), f a x > x^2 + 2*x - 5) ↔ (a ≤ -2) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_neg_one_range_of_a_for_inequality_l1769_176932


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1769_176953

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x y : ℝ, x^2 - (a^2 - 2*a - 15)*x + (a - 1) = 0 ∧ 
               y^2 - (a^2 - 2*a - 15)*y + (a - 1) = 0 ∧ 
               x = -y) → 
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1769_176953


namespace NUMINAMATH_CALUDE_saved_fraction_is_one_seventh_l1769_176915

/-- Represents the worker's financial situation over a year -/
structure WorkerFinances where
  P : ℝ  -- Monthly take-home pay
  S : ℝ  -- Fraction of take-home pay saved each month
  E : ℝ  -- Fraction of take-home pay for expenses each month
  T : ℝ  -- Fixed monthly tax amount

/-- Axioms representing the problem conditions -/
axiom monthly_savings (w : WorkerFinances) : w.S * w.P = w.P - w.E * w.P - w.T

/-- The total yearly savings is twice the monthly non-saved amount -/
axiom yearly_savings_condition (w : WorkerFinances) : 12 * w.S * w.P = 2 * (w.P - w.S * w.P)

/-- The main theorem stating that the saved fraction is 1/7 -/
theorem saved_fraction_is_one_seventh (w : WorkerFinances) : w.S = 1/7 := by
  sorry


end NUMINAMATH_CALUDE_saved_fraction_is_one_seventh_l1769_176915
