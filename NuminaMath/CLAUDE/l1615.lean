import Mathlib

namespace NUMINAMATH_CALUDE_dance_troupe_members_l1615_161503

theorem dance_troupe_members : ∃! n : ℕ, 
  150 < n ∧ n < 300 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 3 ∧ 
  n % 9 = 4 ∧ 
  n = 176 := by
  sorry

end NUMINAMATH_CALUDE_dance_troupe_members_l1615_161503


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l1615_161573

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its cutting --/
structure Plywood where
  original : Rectangle
  pieces : Fin 8 → Rectangle

/-- The original plywood dimensions --/
def original_plywood : Rectangle := { length := 16, width := 4 }

theorem plywood_cut_perimeter_difference :
  ∃ (max_cut min_cut : Plywood),
    (∀ i : Fin 8, perimeter (max_cut.pieces i) ≥ perimeter (min_cut.pieces i)) ∧
    (∀ cut : Plywood, ∀ i : Fin 8, 
      perimeter (cut.pieces i) ≤ perimeter (max_cut.pieces i) ∧
      perimeter (cut.pieces i) ≥ perimeter (min_cut.pieces i)) ∧
    (∀ i j : Fin 8, max_cut.pieces i = max_cut.pieces j) ∧
    (∀ i j : Fin 8, min_cut.pieces i = min_cut.pieces j) ∧
    (max_cut.original = original_plywood) ∧
    (min_cut.original = original_plywood) ∧
    (perimeter (max_cut.pieces 0) - perimeter (min_cut.pieces 0) = 21) :=
by sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l1615_161573


namespace NUMINAMATH_CALUDE_billy_sam_money_multiple_l1615_161568

/-- Given that Sam has $75 and Billy has $25 less than a multiple of Sam's money,
    and together they have $200, prove that the multiple is 2. -/
theorem billy_sam_money_multiple : 
  ∀ (sam_money : ℕ) (total_money : ℕ) (multiple : ℚ),
    sam_money = 75 →
    total_money = 200 →
    total_money = sam_money + (multiple * sam_money - 25) →
    multiple = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_sam_money_multiple_l1615_161568


namespace NUMINAMATH_CALUDE_platform_length_l1615_161542

/-- Given a train of length 900 m that takes 39 sec to cross a platform and 18 sec to cross a signal pole, the length of the platform is 1050 m. -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 900)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 18) :
  train_length + (train_length / time_cross_pole * time_cross_platform) - train_length = 1050 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l1615_161542


namespace NUMINAMATH_CALUDE_divisibility_problem_l1615_161579

theorem divisibility_problem (a : ℝ) : 
  (∃ k : ℤ, 2 * 10^10 + a = 11 * k) → 
  0 ≤ a → 
  a < 11 → 
  a = 9 := by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1615_161579


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l1615_161526

/-- Given a circle tangent to lines 3x + 4y = 24 and 3x + 4y = 0,
    with its center on the line x - 2y = 0,
    prove that the center (x, y) satisfies the given equations. -/
theorem circle_center_coordinates (x y : ℚ) : 
  (∃ (r : ℚ), r > 0 ∧ 
    (∀ (x' y' : ℚ), (x' - x)^2 + (y' - y)^2 = r^2 → 
      (3*x' + 4*y' = 24 ∨ 3*x' + 4*y' = 0))) →
  x - 2*y = 0 →
  3*x + 4*y = 12 := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l1615_161526


namespace NUMINAMATH_CALUDE_tank_capacity_l1615_161574

theorem tank_capacity (initial_fullness final_fullness : ℚ) (added_water : ℕ) : 
  initial_fullness = 1/4 →
  final_fullness = 3/4 →
  added_water = 208 →
  (final_fullness - initial_fullness) * (added_water / (final_fullness - initial_fullness)) = 416 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l1615_161574


namespace NUMINAMATH_CALUDE_flour_with_weevils_l1615_161593

theorem flour_with_weevils 
  (p_good_milk : ℝ) 
  (p_good_egg : ℝ) 
  (p_all_good : ℝ) 
  (h1 : p_good_milk = 0.8) 
  (h2 : p_good_egg = 0.4) 
  (h3 : p_all_good = 0.24) : 
  ∃ (p_good_flour : ℝ), 
    p_good_milk * p_good_egg * p_good_flour = p_all_good ∧ 
    1 - p_good_flour = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_flour_with_weevils_l1615_161593


namespace NUMINAMATH_CALUDE_potato_cost_proof_l1615_161583

/-- The initial cost of one bag of potatoes in rubles -/
def initial_cost : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase factor -/
def andrey_increase : ℝ := 2

/-- Boris's first price increase factor -/
def boris_first_increase : ℝ := 1.6

/-- Boris's second price increase factor -/
def boris_second_increase : ℝ := 1.4

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The difference in earnings between Boris and Andrey -/
def earnings_difference : ℝ := 1200

theorem potato_cost_proof :
  (bags_bought * initial_cost * andrey_increase) +
  earnings_difference =
  (boris_first_sale * initial_cost * boris_first_increase) +
  (boris_second_sale * initial_cost * boris_first_increase * boris_second_increase) :=
by sorry

end NUMINAMATH_CALUDE_potato_cost_proof_l1615_161583


namespace NUMINAMATH_CALUDE_leftover_value_is_fifteen_l1615_161550

def quarters_per_roll : ℕ := 50
def dimes_per_roll : ℕ := 60
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

def james_quarters : ℕ := 97
def james_dimes : ℕ := 178
def lindsay_quarters : ℕ := 143
def lindsay_dimes : ℕ := 292

def total_quarters : ℕ := james_quarters + lindsay_quarters
def total_dimes : ℕ := james_dimes + lindsay_dimes

def leftover_quarters : ℕ := total_quarters % quarters_per_roll
def leftover_dimes : ℕ := total_dimes % dimes_per_roll

def leftover_value : ℚ := leftover_quarters * quarter_value + leftover_dimes * dime_value

theorem leftover_value_is_fifteen :
  leftover_value = 15 := by sorry

end NUMINAMATH_CALUDE_leftover_value_is_fifteen_l1615_161550


namespace NUMINAMATH_CALUDE_tangent_line_curve_intersection_l1615_161524

/-- Given a line y = kx + 1 tangent to the curve y = x^3 + ax + b at the point (1, 3), 
    prove that b = -3 -/
theorem tangent_line_curve_intersection (k a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 
    (k * x + 1 = x^3 + a * x + b → x = 1) ∧ 
    (k * 1 + 1 = 1^3 + a * 1 + b) ∧
    (k = 3 * 1^2 + a)) → 
  (∃ b : ℝ, k * 1 + 1 = 1^3 + a * 1 + b ∧ b = -3) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_curve_intersection_l1615_161524


namespace NUMINAMATH_CALUDE_min_value_expression_l1615_161535

theorem min_value_expression (x : ℝ) (hx : x > 0) :
  4 * Real.sqrt x + 4 / x + 1 / (x^2) ≥ 9 ∧
  ∃ y : ℝ, y > 0 ∧ 4 * Real.sqrt y + 4 / y + 1 / (y^2) = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1615_161535


namespace NUMINAMATH_CALUDE_initial_weight_proof_l1615_161523

theorem initial_weight_proof (W : ℝ) : 
  (W > 0) →
  (0.8 * (0.9 * W) = 36000) →
  (W = 50000) := by
sorry

end NUMINAMATH_CALUDE_initial_weight_proof_l1615_161523


namespace NUMINAMATH_CALUDE_seashells_after_month_l1615_161567

/-- Calculates the number of seashells after a given number of weeks -/
def seashells_after_weeks (initial : ℕ) (weekly_increase : ℕ) (weeks : ℕ) : ℕ :=
  initial + weekly_increase * weeks

/-- Theorem stating that starting with 50 seashells and adding 20 per week for 4 weeks results in 130 seashells -/
theorem seashells_after_month (initial : ℕ) (weekly_increase : ℕ) (weeks : ℕ) 
    (h1 : initial = 50) 
    (h2 : weekly_increase = 20) 
    (h3 : weeks = 4) : 
  seashells_after_weeks initial weekly_increase weeks = 130 := by
  sorry

#eval seashells_after_weeks 50 20 4

end NUMINAMATH_CALUDE_seashells_after_month_l1615_161567


namespace NUMINAMATH_CALUDE_two_thousand_plus_sqrt_two_thousand_one_in_A_l1615_161552

-- Define the set A
variable (A : Set ℝ)

-- Define the conditions
axiom one_in_A : 1 ∈ A
axiom square_in_A : ∀ x : ℝ, x ∈ A → x^2 ∈ A
axiom inverse_square_in_A : ∀ x : ℝ, (x^2 - 4*x + 4) ∈ A → x ∈ A

-- State the theorem
theorem two_thousand_plus_sqrt_two_thousand_one_in_A :
  (2000 + Real.sqrt 2001) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_two_thousand_plus_sqrt_two_thousand_one_in_A_l1615_161552


namespace NUMINAMATH_CALUDE_odot_equation_solution_l1615_161537

-- Define the operation ⊙
noncomputable def odot (a b : ℝ) : ℝ :=
  a^2 + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem odot_equation_solution (g : ℝ) (h1 : g ≥ 0) (h2 : odot 4 g = 20) : g = 12 := by
  sorry

end NUMINAMATH_CALUDE_odot_equation_solution_l1615_161537


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1615_161529

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 60 → x - y = 10 → x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l1615_161529


namespace NUMINAMATH_CALUDE_final_student_count_l1615_161596

/-- Calculates the number of students on a bus after three stops -/
def studentsOnBus (initial : ℝ) (stop1On stop1Off stop2On stop2Off stop3On stop3Off : ℝ) : ℝ :=
  initial + (stop1On - stop1Off) + (stop2On - stop2Off) + (stop3On - stop3Off)

/-- Theorem stating the final number of students on the bus -/
theorem final_student_count :
  studentsOnBus 21 7.5 2 1.2 5.3 11 4.8 = 28.6 := by
  sorry

end NUMINAMATH_CALUDE_final_student_count_l1615_161596


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l1615_161577

theorem profit_percent_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.82 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100 / 82 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l1615_161577


namespace NUMINAMATH_CALUDE_number_line_expressions_l1615_161555

theorem number_line_expressions (P Q R S T : ℝ) 
  (hP : P > 3 ∧ P < 4)
  (hQ : Q > 1 ∧ Q < 1.2)
  (hR : R > -0.2 ∧ R < 0)
  (hS : S > 0.8 ∧ S < 1)
  (hT : T > 1.4 ∧ T < 1.6) :
  R / (P * Q) < 0 ∧ (S + T) / R < 0 ∧ P - Q ≥ 0 ∧ P * Q ≥ 0 ∧ (S / Q) * P ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_number_line_expressions_l1615_161555


namespace NUMINAMATH_CALUDE_smallest_positive_difference_l1615_161599

/-- Vovochka's addition method for three-digit numbers -/
def vovochka_sum (a b c d e f : ℕ) : ℕ :=
  (a + d) * 1000 + (b + e) * 100 + (c + f)

/-- Correct addition method for three-digit numbers -/
def correct_sum (a b c d e f : ℕ) : ℕ :=
  (a + d) * 100 + (b + e) * 10 + (c + f)

/-- The difference between Vovochka's sum and the correct sum -/
def sum_difference (a b c d e f : ℕ) : ℤ :=
  (vovochka_sum a b c d e f : ℤ) - (correct_sum a b c d e f : ℤ)

theorem smallest_positive_difference :
  ∃ (a b c d e f : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
    sum_difference a b c d e f > 0 ∧
    ∀ (x y z u v w : ℕ),
      x < 10 → y < 10 → z < 10 → u < 10 → v < 10 → w < 10 →
      sum_difference x y z u v w > 0 →
      sum_difference a b c d e f ≤ sum_difference x y z u v w ∧
    sum_difference a b c d e f = 1800 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_difference_l1615_161599


namespace NUMINAMATH_CALUDE_max_abs_quadratic_function_l1615_161536

theorem max_abs_quadratic_function (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (|f 0| ≤ 2) → (|f 2| ≤ 2) → (|f (-2)| ≤ 2) →
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, |f x| ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_quadratic_function_l1615_161536


namespace NUMINAMATH_CALUDE_original_number_proof_l1615_161551

theorem original_number_proof (w : ℝ) : 
  (w + 0.125 * w) - (w - 0.25 * w) = 30 → w = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1615_161551


namespace NUMINAMATH_CALUDE_square_division_l1615_161553

/-- A square can be divided into two equal parts in at least four different ways. -/
theorem square_division (s : ℝ) (h : s > 0) :
  ∃ (rect1 rect2 : ℝ × ℝ) (tri1 tri2 : ℝ × ℝ × ℝ),
    -- Vertical division
    rect1 = (s, s/2) ∧
    -- Horizontal division
    rect2 = (s/2, s) ∧
    -- Diagonal division (top-left to bottom-right)
    tri1 = (s, s, Real.sqrt 2 * s) ∧
    -- Diagonal division (top-right to bottom-left)
    tri2 = (s, s, Real.sqrt 2 * s) ∧
    -- All divisions result in equal areas
    s * (s/2) = (s/2) * s ∧
    s * (s/2) = (1/2) * s * s ∧
    -- All divisions are valid (non-negative dimensions)
    s > 0 ∧ s/2 > 0 ∧ Real.sqrt 2 * s > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_square_division_l1615_161553


namespace NUMINAMATH_CALUDE_axiom_1_l1615_161513

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the membership relations
variable (pointOnLine : Point → Line → Prop)
variable (pointInPlane : Point → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem axiom_1 (A B : Point) (l : Line) (α : Plane) :
  pointOnLine A l → pointOnLine B l → pointInPlane A α → pointInPlane B α →
  lineInPlane l α := by
  sorry

end NUMINAMATH_CALUDE_axiom_1_l1615_161513


namespace NUMINAMATH_CALUDE_complementary_angle_of_60_13_25_l1615_161517

/-- Represents an angle in degrees, minutes, and seconds -/
structure DMS where
  degrees : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the complementary angle of a given angle in DMS format -/
def complementaryAngle (angle : DMS) : DMS :=
  sorry

/-- Theorem stating that the complementary angle of 60°13'25" is 29°46'35" -/
theorem complementary_angle_of_60_13_25 :
  let givenAngle : DMS := ⟨60, 13, 25⟩
  complementaryAngle givenAngle = ⟨29, 46, 35⟩ := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_of_60_13_25_l1615_161517


namespace NUMINAMATH_CALUDE_digit_swap_difference_l1615_161538

theorem digit_swap_difference (a b c : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) :
  ∃ k : ℤ, (100 * a + 10 * b + c) - (10 * a + 100 * b + c) = 90 * k :=
sorry

end NUMINAMATH_CALUDE_digit_swap_difference_l1615_161538


namespace NUMINAMATH_CALUDE_point_coordinates_l1615_161589

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: If a point P is in the second quadrant, its distance to the x-axis is 3,
    and its distance to the y-axis is 10, then its coordinates are (-10, 3) -/
theorem point_coordinates (P : Point)
  (h1 : SecondQuadrant P)
  (h2 : DistanceToXAxis P = 3)
  (h3 : DistanceToYAxis P = 10) :
  P.x = -10 ∧ P.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1615_161589


namespace NUMINAMATH_CALUDE_brenda_age_is_three_l1615_161560

/-- Represents the ages of family members -/
structure FamilyAges where
  addison : ℕ
  brenda : ℕ
  janet : ℕ

/-- The conditions given in the problem -/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.addison = 4 * ages.brenda ∧
  ages.janet = ages.brenda + 9 ∧
  ages.addison = ages.janet

/-- Theorem stating that if the family ages are valid, Brenda's age is 3 -/
theorem brenda_age_is_three (ages : FamilyAges) 
  (h : validFamilyAges ages) : ages.brenda = 3 := by
  sorry

#check brenda_age_is_three

end NUMINAMATH_CALUDE_brenda_age_is_three_l1615_161560


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1615_161587

theorem complex_equation_solution (z : ℂ) : z * (2 - Complex.I) = 3 + Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1615_161587


namespace NUMINAMATH_CALUDE_expression_factorization_l1615_161525

theorem expression_factorization (a : ℝ) :
  (6 * a^3 + 92 * a^2 - 7) - (-7 * a^3 + a^2 - 7) = 13 * a^2 * (a + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1615_161525


namespace NUMINAMATH_CALUDE_frequency_not_necessarily_equal_probability_l1615_161528

/-- Represents the outcome of a single trial in the random simulation -/
inductive Outcome
  | selected
  | notSelected

/-- Represents the result of multiple trials in the random simulation -/
structure SimulationResult :=
  (trials : ℕ)
  (selections : ℕ)

/-- The theoretical probability of selecting a specific student from 6 students -/
def theoretical_probability : ℚ := 1 / 6

/-- The frequency of selecting a specific student in a simulation -/
def frequency (result : SimulationResult) : ℚ :=
  result.selections / result.trials

/-- Statement: The frequency in a random simulation is not necessarily equal to the theoretical probability -/
theorem frequency_not_necessarily_equal_probability :
  ∃ (result : SimulationResult), frequency result ≠ theoretical_probability :=
sorry

end NUMINAMATH_CALUDE_frequency_not_necessarily_equal_probability_l1615_161528


namespace NUMINAMATH_CALUDE_second_project_breadth_l1615_161546

/-- Represents a digging project with depth, length, and breadth dimensions -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of a digging project -/
def volume (project : DiggingProject) : ℝ :=
  project.depth * project.length * project.breadth

/-- Theorem: Given two digging projects with equal volumes, 
    the breadth of the second project is 50 meters -/
theorem second_project_breadth
  (project1 : DiggingProject)
  (project2 : DiggingProject)
  (h1 : project1.depth = 100)
  (h2 : project1.length = 25)
  (h3 : project1.breadth = 30)
  (h4 : project2.depth = 75)
  (h5 : project2.length = 20)
  (h_equal_volume : volume project1 = volume project2) :
  project2.breadth = 50 := by
  sorry

#check second_project_breadth

end NUMINAMATH_CALUDE_second_project_breadth_l1615_161546


namespace NUMINAMATH_CALUDE_suresh_job_completion_time_l1615_161549

/-- The time it takes Suresh to complete the job alone -/
def suresh_time : ℝ := 15

/-- The time it takes Ashutosh to complete the job alone -/
def ashutosh_time : ℝ := 20

/-- The time Suresh works on the job -/
def suresh_work_time : ℝ := 9

/-- The time Ashutosh works to complete the remaining job -/
def ashutosh_completion_time : ℝ := 8

theorem suresh_job_completion_time : 
  (suresh_work_time / suresh_time) + (ashutosh_completion_time / ashutosh_time) = 1 ∧ 
  suresh_time = 15 := by
  sorry


end NUMINAMATH_CALUDE_suresh_job_completion_time_l1615_161549


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l1615_161598

theorem sin_plus_cos_value (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : Real.tan (θ + π/4) = 1/7) : 
  Real.sin θ + Real.cos θ = -1/5 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l1615_161598


namespace NUMINAMATH_CALUDE_min_values_constraint_l1615_161559

theorem min_values_constraint (x y z : ℝ) (h : x - 2*y + z = 4) :
  (∀ a b c : ℝ, a - 2*b + c = 4 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧
  (∀ a b c : ℝ, a - 2*b + c = 4 → x^2 + (y - 1)^2 + z^2 ≤ a^2 + (b - 1)^2 + c^2) ∧
  (∃ a b c : ℝ, a - 2*b + c = 4 ∧ a^2 + b^2 + c^2 = 8/3) ∧
  (∃ a b c : ℝ, a - 2*b + c = 4 ∧ a^2 + (b - 1)^2 + c^2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_min_values_constraint_l1615_161559


namespace NUMINAMATH_CALUDE_exponential_function_determination_l1615_161562

theorem exponential_function_determination (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = a^x)
  (h2 : a > 0)
  (h3 : a ≠ 1)
  (h4 : f 2 = 4) :
  ∀ x, f x = 2^x :=
by sorry

end NUMINAMATH_CALUDE_exponential_function_determination_l1615_161562


namespace NUMINAMATH_CALUDE_inequality_proof_l1615_161554

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + 2*c)) + (b / (c + 2*a)) + (c / (a + 2*b)) > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1615_161554


namespace NUMINAMATH_CALUDE_bridge_length_l1615_161581

/-- The length of a bridge given specific train and crossing conditions -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 125 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 250 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1615_161581


namespace NUMINAMATH_CALUDE_absolute_value_integral_l1615_161597

theorem absolute_value_integral : ∫ x in (0)..(4), |x - 2| = 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_integral_l1615_161597


namespace NUMINAMATH_CALUDE_zebra_population_last_year_l1615_161518

/-- The number of zebras in a national park over two consecutive years. -/
structure ZebraPopulation where
  current : ℕ
  born : ℕ
  died : ℕ
  last_year : ℕ

/-- Theorem stating the relationship between the zebra population this year and last year. -/
theorem zebra_population_last_year (zp : ZebraPopulation)
    (h1 : zp.current = 725)
    (h2 : zp.born = 419)
    (h3 : zp.died = 263)
    : zp.last_year = 569 := by
  sorry

end NUMINAMATH_CALUDE_zebra_population_last_year_l1615_161518


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1615_161569

/-- A geometric sequence where the sum of every two adjacent terms forms a geometric sequence --/
def GeometricSequenceWithAdjacentSums (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 2) + a (n + 3) = r * (a n + a (n + 1))

/-- The theorem stating the sum of specific terms in the geometric sequence --/
theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequenceWithAdjacentSums a)
  (h_sum1 : a 1 + a 2 = 1/2)
  (h_sum2 : a 3 + a 4 = 1) :
  a 7 + a 8 + a 9 + a 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1615_161569


namespace NUMINAMATH_CALUDE_tan_value_from_sin_cos_equation_l1615_161504

theorem tan_value_from_sin_cos_equation (x : ℝ) 
  (h1 : 0 < x ∧ x < Real.pi / 2) 
  (h2 : Real.sin x ^ 4 / 42 + Real.cos x ^ 4 / 75 = 1 / 117) : 
  Real.tan x = Real.sqrt 14 / 5 := by
sorry

end NUMINAMATH_CALUDE_tan_value_from_sin_cos_equation_l1615_161504


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l1615_161594

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l1615_161594


namespace NUMINAMATH_CALUDE_no_base_for_perfect_square_l1615_161547

theorem no_base_for_perfect_square : ¬ ∃ (b : ℕ), ∃ (n : ℕ), b^2 + 3*b + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_perfect_square_l1615_161547


namespace NUMINAMATH_CALUDE_tissue_pallet_ratio_l1615_161595

theorem tissue_pallet_ratio (total_pallets : ℕ) 
  (paper_towel_pallets : ℕ) (paper_plate_pallets : ℕ) (paper_cup_pallets : ℕ) :
  total_pallets = 20 →
  paper_towel_pallets = total_pallets / 2 →
  paper_plate_pallets = total_pallets / 5 →
  paper_cup_pallets = 1 →
  let tissue_pallets := total_pallets - (paper_towel_pallets + paper_plate_pallets + paper_cup_pallets)
  (tissue_pallets : ℚ) / (total_pallets : ℚ) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_tissue_pallet_ratio_l1615_161595


namespace NUMINAMATH_CALUDE_hot_air_balloon_balloons_l1615_161585

theorem hot_air_balloon_balloons (initial_balloons : ℕ) : 
  (initial_balloons : ℚ) * (2 / 5) = 80 → initial_balloons = 200 :=
by
  sorry

#check hot_air_balloon_balloons

end NUMINAMATH_CALUDE_hot_air_balloon_balloons_l1615_161585


namespace NUMINAMATH_CALUDE_descending_order_proof_l1615_161500

theorem descending_order_proof (a b c d : ℝ) : 
  a = Real.sin (33 * π / 180) →
  b = Real.cos (55 * π / 180) →
  c = Real.tan (35 * π / 180) →
  d = Real.log 5 →
  d > c ∧ c > b ∧ b > a :=
by sorry

end NUMINAMATH_CALUDE_descending_order_proof_l1615_161500


namespace NUMINAMATH_CALUDE_system_demonstrates_transformational_thinking_l1615_161576

/-- A system of two linear equations in two variables -/
structure LinearSystem :=
  (eq1 : ℝ → ℝ → ℝ)
  (eq2 : ℝ → ℝ → ℝ)

/-- The process of substituting one equation into another -/
def substitute (sys : LinearSystem) : ℝ → ℝ :=
  λ y => sys.eq1 (sys.eq2 y y) y

/-- Transformational thinking in the context of solving linear systems -/
def transformational_thinking (sys : LinearSystem) : Prop :=
  ∃ (simplified_eq : ℝ → ℝ), substitute sys = simplified_eq

/-- The given system of linear equations -/
def given_system : LinearSystem :=
  { eq1 := λ x y => 2*x + y
  , eq2 := λ x y => x - 2*y }

/-- Theorem stating that the given system demonstrates transformational thinking -/
theorem system_demonstrates_transformational_thinking :
  transformational_thinking given_system :=
sorry


end NUMINAMATH_CALUDE_system_demonstrates_transformational_thinking_l1615_161576


namespace NUMINAMATH_CALUDE_pool_and_deck_area_l1615_161516

/-- Given a rectangular pool with dimensions 10 feet by 12 feet and a surrounding deck
    with a uniform width of 4 feet, the total area of the pool and deck is 360 square feet. -/
theorem pool_and_deck_area :
  let pool_length : ℕ := 12
  let pool_width : ℕ := 10
  let deck_width : ℕ := 4
  let total_length : ℕ := pool_length + 2 * deck_width
  let total_width : ℕ := pool_width + 2 * deck_width
  total_length * total_width = 360 := by
  sorry

end NUMINAMATH_CALUDE_pool_and_deck_area_l1615_161516


namespace NUMINAMATH_CALUDE_min_fencing_length_proof_l1615_161531

/-- The minimum length of bamboo fencing needed to enclose a rectangular flower bed -/
def min_fencing_length : ℝ := 20

/-- The area of the rectangular flower bed -/
def flower_bed_area : ℝ := 50

theorem min_fencing_length_proof :
  ∀ (length width : ℝ),
  length > 0 →
  width > 0 →
  length * width = flower_bed_area →
  length + 2 * width ≥ min_fencing_length :=
by
  sorry

#check min_fencing_length_proof

end NUMINAMATH_CALUDE_min_fencing_length_proof_l1615_161531


namespace NUMINAMATH_CALUDE_initial_travel_time_l1615_161511

theorem initial_travel_time (distance : ℝ) (new_speed : ℝ) :
  distance = 540 ∧ new_speed = 60 →
  ∃ initial_time : ℝ,
    distance = new_speed * (3/4 * initial_time) ∧
    initial_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_travel_time_l1615_161511


namespace NUMINAMATH_CALUDE_surface_area_after_removing_corners_l1615_161561

/-- Represents the dimensions of a cube in centimeters -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube given its dimensions -/
def surfaceArea (d : CubeDimensions) : ℝ :=
  6 * d.length * d.width

/-- Represents the problem setup -/
structure CubeWithCornersRemoved where
  originalCube : CubeDimensions
  cornerCube : CubeDimensions

/-- The main theorem to be proved -/
theorem surface_area_after_removing_corners
  (c : CubeWithCornersRemoved)
  (h1 : c.originalCube.length = 4)
  (h2 : c.originalCube.width = 4)
  (h3 : c.originalCube.height = 4)
  (h4 : c.cornerCube.length = 2)
  (h5 : c.cornerCube.width = 2)
  (h6 : c.cornerCube.height = 2) :
  surfaceArea c.originalCube = 96 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_after_removing_corners_l1615_161561


namespace NUMINAMATH_CALUDE_perpendicular_vectors_lambda_l1615_161557

/-- Given two vectors a and b in ℝ², where a is perpendicular to (a - b), prove that the second component of b equals 4. -/
theorem perpendicular_vectors_lambda (a b : ℝ × ℝ) : 
  a = (-1, 3) → 
  b.1 = 2 → 
  a • (a - b) = 0 → 
  b.2 = 4 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_lambda_l1615_161557


namespace NUMINAMATH_CALUDE_car_distance_l1615_161502

/-- Given a car that travels 180 miles in 4 hours, prove that it will travel 135 miles in the next 3 hours at the same rate. -/
theorem car_distance (initial_distance : ℝ) (initial_time : ℝ) (next_time : ℝ) :
  initial_distance = 180 ∧ initial_time = 4 ∧ next_time = 3 →
  (initial_distance / initial_time) * next_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l1615_161502


namespace NUMINAMATH_CALUDE_committee_formation_count_l1615_161510

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of Republicans in the city council -/
def num_republicans : ℕ := 10

/-- The number of Democrats in the city council -/
def num_democrats : ℕ := 7

/-- The number of Republicans needed in the committee -/
def committee_republicans : ℕ := 4

/-- The number of Democrats needed in the committee -/
def committee_democrats : ℕ := 3

/-- The total number of ways to form the committee -/
def total_committee_formations : ℕ := 
  binomial num_republicans committee_republicans * binomial num_democrats committee_democrats

theorem committee_formation_count : total_committee_formations = 7350 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l1615_161510


namespace NUMINAMATH_CALUDE_nested_radical_eighteen_l1615_161584

theorem nested_radical_eighteen (x : ℝ) : x = Real.sqrt (18 + x) → x = (1 + Real.sqrt 73) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_eighteen_l1615_161584


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l1615_161544

theorem quadratic_form_ratio (x : ℝ) : 
  ∃ (d e : ℝ), x^2 + 2023*x + 2023 = (x + d)^2 + e ∧ e/d = -1009.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l1615_161544


namespace NUMINAMATH_CALUDE_no_four_distinct_real_roots_l1615_161527

theorem no_four_distinct_real_roots (a b : ℝ) : 
  ¬ ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄) ∧
    (∀ x : ℝ, x^4 - 4*x^3 + 6*x^2 + a*x + b = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄)) :=
sorry

end NUMINAMATH_CALUDE_no_four_distinct_real_roots_l1615_161527


namespace NUMINAMATH_CALUDE_gianna_savings_period_l1615_161509

/-- Proves that Gianna saved money for 365 days given the conditions -/
theorem gianna_savings_period (daily_savings : ℕ) (total_savings : ℕ) 
  (h1 : daily_savings = 39)
  (h2 : total_savings = 14235) :
  total_savings / daily_savings = 365 := by
  sorry

end NUMINAMATH_CALUDE_gianna_savings_period_l1615_161509


namespace NUMINAMATH_CALUDE_spaceship_age_conversion_l1615_161515

/-- Converts a three-digit number in base 9 to base 10 --/
def base9_to_base10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 9^2 + tens * 9^1 + ones * 9^0

/-- The age of the alien spaceship --/
def spaceship_age : Nat := 362

theorem spaceship_age_conversion :
  base9_to_base10 3 6 2 = 299 :=
by sorry

end NUMINAMATH_CALUDE_spaceship_age_conversion_l1615_161515


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1615_161507

theorem cube_sum_reciprocal (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1615_161507


namespace NUMINAMATH_CALUDE_possible_zero_point_l1615_161565

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem possible_zero_point (hf : Continuous f) 
  (h2007 : f 2007 < 0) (h2008 : f 2008 < 0) (h2009 : f 2009 > 0) :
  ∃ x ∈ Set.Ioo 2007 2008, f x = 0 ∨ ∃ y ∈ Set.Ioo 2008 2009, f y = 0 :=
by sorry


end NUMINAMATH_CALUDE_possible_zero_point_l1615_161565


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l1615_161592

theorem sqrt_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l1615_161592


namespace NUMINAMATH_CALUDE_vann_teeth_cleaning_l1615_161563

/-- The number of teeth a dog has -/
def dog_teeth : ℕ := 42

/-- The number of teeth a cat has -/
def cat_teeth : ℕ := 30

/-- The number of teeth a pig has -/
def pig_teeth : ℕ := 44

/-- The number of teeth a horse has -/
def horse_teeth : ℕ := 40

/-- The number of teeth a rabbit has -/
def rabbit_teeth : ℕ := 28

/-- The number of dogs Vann will clean -/
def num_dogs : ℕ := 7

/-- The number of cats Vann will clean -/
def num_cats : ℕ := 12

/-- The number of pigs Vann will clean -/
def num_pigs : ℕ := 9

/-- The number of horses Vann will clean -/
def num_horses : ℕ := 4

/-- The number of rabbits Vann will clean -/
def num_rabbits : ℕ := 15

/-- The total number of teeth Vann will clean -/
def total_teeth : ℕ := 
  num_dogs * dog_teeth + 
  num_cats * cat_teeth + 
  num_pigs * pig_teeth + 
  num_horses * horse_teeth + 
  num_rabbits * rabbit_teeth

theorem vann_teeth_cleaning : total_teeth = 1630 := by
  sorry

end NUMINAMATH_CALUDE_vann_teeth_cleaning_l1615_161563


namespace NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l1615_161522

theorem sphere_cylinder_equal_area (h : ℝ) (d : ℝ) (r : ℝ) :
  h = 16 →
  d = 16 →
  4 * Real.pi * r^2 = 2 * Real.pi * (d / 2) * h →
  r = 8 :=
by sorry

end NUMINAMATH_CALUDE_sphere_cylinder_equal_area_l1615_161522


namespace NUMINAMATH_CALUDE_estimate_three_plus_sqrt_ten_l1615_161543

theorem estimate_three_plus_sqrt_ten : 6 < 3 + Real.sqrt 10 ∧ 3 + Real.sqrt 10 < 7 := by
  sorry

end NUMINAMATH_CALUDE_estimate_three_plus_sqrt_ten_l1615_161543


namespace NUMINAMATH_CALUDE_simplify_86_with_95_base_l1615_161533

/-- Simplifies a score based on a given base score. -/
def simplify_score (score : Int) (base : Int) : Int :=
  score - base

/-- The base score considered as excellent. -/
def excellent_score : Int := 95

/-- Theorem: Simplifying a score of 86 with 95 as the base results in -9. -/
theorem simplify_86_with_95_base :
  simplify_score 86 excellent_score = -9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_86_with_95_base_l1615_161533


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1615_161586

theorem unique_solution_condition (k : ℚ) : 
  (∃! x : ℝ, (x + 3) * (x + 2) = k + 3 * x) ↔ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1615_161586


namespace NUMINAMATH_CALUDE_mika_gave_six_stickers_l1615_161566

/-- Represents the number of stickers Mika had, bought, received, used, and gave away --/
structure StickerCount where
  initial : Nat
  bought : Nat
  birthday : Nat
  usedForCard : Nat
  leftOver : Nat

/-- Calculates the number of stickers Mika gave to her sister --/
def stickersGivenToSister (s : StickerCount) : Nat :=
  s.initial + s.bought + s.birthday - (s.usedForCard + s.leftOver)

/-- Theorem stating that Mika gave 6 stickers to her sister --/
theorem mika_gave_six_stickers (s : StickerCount) 
  (h1 : s.initial = 20)
  (h2 : s.bought = 26)
  (h3 : s.birthday = 20)
  (h4 : s.usedForCard = 58)
  (h5 : s.leftOver = 2) : 
  stickersGivenToSister s = 6 := by
  sorry

end NUMINAMATH_CALUDE_mika_gave_six_stickers_l1615_161566


namespace NUMINAMATH_CALUDE_triangle_median_theorem_l1615_161521

/-- Triangle XYZ with given side lengths and median --/
structure Triangle where
  XY : ℝ
  XZ : ℝ
  XM : ℝ
  YZ : ℝ

/-- The theorem stating the relationship between sides and median in the given triangle --/
theorem triangle_median_theorem (t : Triangle) (h1 : t.XY = 6) (h2 : t.XZ = 9) (h3 : t.XM = 4) :
  t.YZ = Real.sqrt 170 := by
  sorry

#check triangle_median_theorem

end NUMINAMATH_CALUDE_triangle_median_theorem_l1615_161521


namespace NUMINAMATH_CALUDE_eat_porridge_together_l1615_161501

/-- Masha's time to eat one bowl of porridge in minutes -/
def mashaTime : ℝ := 12

/-- The Bear's eating speed relative to Masha's -/
def bearRelativeSpeed : ℝ := 2

/-- Number of bowls to eat together -/
def totalBowls : ℝ := 6

/-- Time for Masha and the Bear to eat all bowls together -/
def totalTime : ℝ := 24

theorem eat_porridge_together :
  (totalBowls * mashaTime) / (1 + bearRelativeSpeed) = totalTime := by
  sorry

end NUMINAMATH_CALUDE_eat_porridge_together_l1615_161501


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l1615_161571

/-- The number of available squares -/
def n : ℕ := 9

/-- The side length of each square in cm -/
def side_length : ℝ := 1

/-- The set of possible widths for rectangles -/
def possible_widths : Finset ℕ := Finset.range n

/-- The set of possible heights for rectangles -/
def possible_heights : Finset ℕ := Finset.range n

/-- The area of a rectangle with given width and height -/
def rectangle_area (w h : ℕ) : ℝ := (w : ℝ) * (h : ℝ) * side_length ^ 2

/-- The set of all valid rectangles (width, height) that can be formed -/
def valid_rectangles : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 * p.2 ≤ n) (possible_widths.product possible_heights)

/-- The sum of areas of all distinct rectangles -/
def sum_of_areas : ℝ := Finset.sum valid_rectangles (fun p => rectangle_area p.1 p.2)

theorem sum_of_rectangle_areas :
  sum_of_areas = 72 := by sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l1615_161571


namespace NUMINAMATH_CALUDE_specific_ellipse_area_l1615_161540

/-- An ellipse with given properties --/
structure Ellipse where
  major_axis_endpoint1 : ℝ × ℝ
  major_axis_endpoint2 : ℝ × ℝ
  point_on_ellipse : ℝ × ℝ

/-- The area of an ellipse --/
def ellipse_area (e : Ellipse) : ℝ := sorry

/-- Theorem stating the area of the specific ellipse --/
theorem specific_ellipse_area :
  let e : Ellipse := {
    major_axis_endpoint1 := (-8, 3),
    major_axis_endpoint2 := (12, 3),
    point_on_ellipse := (10, 6)
  }
  ellipse_area e = 50 * Real.pi := by sorry

end NUMINAMATH_CALUDE_specific_ellipse_area_l1615_161540


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1615_161570

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 8*x + 12

/-- The square inscribed in the region bound by the parabola and the x-axis -/
structure InscribedSquare where
  center : ℝ  -- x-coordinate of the square's center
  sideLength : ℝ  -- side length of the square
  top_on_parabola : parabola (center + sideLength/2) = sideLength
  bottom_on_xaxis : center - sideLength/2 ≥ 0

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area :
  ∀ (s : InscribedSquare), s.sideLength^2 = 24 - 8*Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1615_161570


namespace NUMINAMATH_CALUDE_gcd_cube_plus_square_and_linear_l1615_161506

theorem gcd_cube_plus_square_and_linear (n m : ℤ) (hn : n > 2^3) : 
  Int.gcd (n^3 + m^2) (n + 2) = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_square_and_linear_l1615_161506


namespace NUMINAMATH_CALUDE_greatest_b_for_quadratic_range_l1615_161590

theorem greatest_b_for_quadratic_range : ∃ (b : ℤ), 
  (∀ x : ℝ, x^2 + b*x + 20 ≠ -4) ∧ 
  (∀ c : ℤ, c > b → ∃ x : ℝ, x^2 + c*x + 20 = -4) ∧
  b = 9 := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_for_quadratic_range_l1615_161590


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_real_line_l1615_161534

/-- The solution set of a quadratic inequality is the entire real line -/
theorem quadratic_inequality_solution_set_real_line 
  (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_real_line_l1615_161534


namespace NUMINAMATH_CALUDE_joshuas_skittles_l1615_161532

/-- Given that Joshua gave 40.0 Skittles to each of his 5.0 friends,
    prove that the total number of Skittles his friends have is 200.0. -/
theorem joshuas_skittles (skittles_per_friend : ℝ) (num_friends : ℝ) 
    (h1 : skittles_per_friend = 40.0)
    (h2 : num_friends = 5.0) : 
  skittles_per_friend * num_friends = 200.0 := by
  sorry

end NUMINAMATH_CALUDE_joshuas_skittles_l1615_161532


namespace NUMINAMATH_CALUDE_probability_square_or_circle_l1615_161508

theorem probability_square_or_circle (total : ℕ) (squares : ℕ) (circles : ℕ) 
  (h1 : total = 10) 
  (h2 : squares = 4) 
  (h3 : circles = 3) :
  (squares + circles : ℚ) / total = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_square_or_circle_l1615_161508


namespace NUMINAMATH_CALUDE_a_14_equals_41_l1615_161545

/-- An arithmetic sequence with a_2 = 5 and a_6 = 17 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 5 ∧ a 6 = 17 ∧ ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In the given arithmetic sequence, a_14 = 41 -/
theorem a_14_equals_41 (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 14 = 41 := by
  sorry

end NUMINAMATH_CALUDE_a_14_equals_41_l1615_161545


namespace NUMINAMATH_CALUDE_mean_calculation_l1615_161588

theorem mean_calculation (x : ℝ) : 
  (28 + x + 50 + 78 + 104) / 5 = 62 → 
  (48 + 62 + 98 + 124 + x) / 5 = 76.4 := by
sorry

end NUMINAMATH_CALUDE_mean_calculation_l1615_161588


namespace NUMINAMATH_CALUDE_total_revenue_is_1168_l1615_161578

/-- Calculates the total revenue from apple and orange sales given the following conditions:
  * 50 boxes of apples and 30 boxes of oranges on Saturday
  * 25 boxes of apples and 15 boxes of oranges on Sunday
  * 10 apples in each apple box
  * 8 oranges in each orange box
  * Each apple sold for $1.20
  * Each orange sold for $0.80
  * Total of 720 apples and 380 oranges sold on Saturday and Sunday -/
def total_revenue : ℝ :=
  let apple_boxes_saturday : ℕ := 50
  let orange_boxes_saturday : ℕ := 30
  let apple_boxes_sunday : ℕ := 25
  let orange_boxes_sunday : ℕ := 15
  let apples_per_box : ℕ := 10
  let oranges_per_box : ℕ := 8
  let apple_price : ℝ := 1.20
  let orange_price : ℝ := 0.80
  let total_apples_sold : ℕ := 720
  let total_oranges_sold : ℕ := 380
  let apple_revenue : ℝ := (total_apples_sold : ℝ) * apple_price
  let orange_revenue : ℝ := (total_oranges_sold : ℝ) * orange_price
  apple_revenue + orange_revenue

/-- Theorem stating that the total revenue is $1168 -/
theorem total_revenue_is_1168 : total_revenue = 1168 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_1168_l1615_161578


namespace NUMINAMATH_CALUDE_cubic_equation_root_b_value_l1615_161519

theorem cubic_equation_root_b_value (a b : ℚ) : 
  (∃ x : ℝ, x = 3 + Real.sqrt 5 ∧ x^3 + a*x^2 + b*x + 12 = 0) → b = -14 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_b_value_l1615_161519


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1615_161541

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8/15) (h2 : x - y = 2/15) : x^2 - y^2 = 16/225 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1615_161541


namespace NUMINAMATH_CALUDE_increasing_quadratic_max_value_function_inequality_positive_reals_l1615_161591

-- Statement 1
theorem increasing_quadratic (x : ℝ) (h : x > 0) :
  Monotone (fun x => 2 * x^2 + x + 1) := by sorry

-- Statement 2
theorem max_value_function (x : ℝ) (h : x > 0) :
  (2 - 3*x - 4/x) ≤ (2 - 4*Real.sqrt 3) := by sorry

-- Statement 3
theorem inequality_positive_reals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) * (y + z) * (z + x) ≥ 8 * x * y * z := by sorry

end NUMINAMATH_CALUDE_increasing_quadratic_max_value_function_inequality_positive_reals_l1615_161591


namespace NUMINAMATH_CALUDE_product_of_good_sequences_is_good_l1615_161556

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define the first derivative of a sequence
def FirstDerivative (a : Sequence) : Sequence :=
  λ n => a (n + 1) - a n

-- Define the k-th derivative of a sequence
def KthDerivative (a : Sequence) : ℕ → Sequence
  | 0 => a
  | k + 1 => FirstDerivative (KthDerivative a k)

-- Define a good sequence
def IsGoodSequence (a : Sequence) : Prop :=
  ∀ k n, KthDerivative a k n > 0

-- Theorem statement
theorem product_of_good_sequences_is_good
  (a b : Sequence)
  (ha : IsGoodSequence a)
  (hb : IsGoodSequence b) :
  IsGoodSequence (λ n => a n * b n) :=
by sorry

end NUMINAMATH_CALUDE_product_of_good_sequences_is_good_l1615_161556


namespace NUMINAMATH_CALUDE_parallel_segments_length_l1615_161514

/-- Represents a line segment with a length -/
structure Segment where
  length : ℝ

/-- Represents three parallel line segments -/
structure ParallelSegments where
  ef : Segment
  gh : Segment
  ij : Segment

/-- Theorem: Given three parallel line segments EF, GH, and IJ,
    where IJ = 120 cm and EF = 180 cm, the length of GH is 72 cm -/
theorem parallel_segments_length 
  (segments : ParallelSegments) 
  (h1 : segments.ij.length = 120) 
  (h2 : segments.ef.length = 180) : 
  segments.gh.length = 72 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_length_l1615_161514


namespace NUMINAMATH_CALUDE_total_voters_l1615_161564

/-- The number of voters in each district --/
structure VoterCount where
  district1 : ℕ
  district2 : ℕ
  district3 : ℕ
  district4 : ℕ
  district5 : ℕ
  district6 : ℕ
  district7 : ℕ

/-- The conditions for voter counts in each district --/
def validVoterCount (v : VoterCount) : Prop :=
  v.district1 = 322 ∧
  v.district2 = v.district1 / 2 - 19 ∧
  v.district3 = 2 * v.district1 ∧
  v.district4 = v.district2 + 45 ∧
  v.district5 = 3 * v.district3 - 150 ∧
  v.district6 = (v.district1 + v.district4) + (v.district1 + v.district4) / 5 ∧
  v.district7 = v.district2 + (v.district5 - v.district2) / 2

/-- The theorem stating that the sum of voters in all districts is 4650 --/
theorem total_voters (v : VoterCount) (h : validVoterCount v) :
  v.district1 + v.district2 + v.district3 + v.district4 + v.district5 + v.district6 + v.district7 = 4650 := by
  sorry

end NUMINAMATH_CALUDE_total_voters_l1615_161564


namespace NUMINAMATH_CALUDE_sum_becomes_27_l1615_161572

def numbers : List ℝ := [1.05, 1.15, 1.25, 1.4, 1.5, 1.6, 1.75, 1.85, 1.95]

def sum_with_error (nums : List ℝ) (error_index : Nat) : ℝ :=
  let error_value := nums[error_index]! * 10
  (nums.sum - nums[error_index]!) + error_value

theorem sum_becomes_27 :
  ∃ (i : Nat), i < numbers.length ∧ sum_with_error numbers i = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_becomes_27_l1615_161572


namespace NUMINAMATH_CALUDE_survey_optimism_l1615_161539

theorem survey_optimism (a b c : ℕ) (m n : ℤ) : 
  a + b + c = 100 →
  m = a + b / 2 →
  n = a - c →
  m = 40 →
  n = -20 :=
by sorry

end NUMINAMATH_CALUDE_survey_optimism_l1615_161539


namespace NUMINAMATH_CALUDE_orange_banana_ratio_l1615_161582

/-- Proves that the ratio of oranges to bananas is 2:1 given the problem conditions --/
theorem orange_banana_ratio :
  ∀ (orange_price pear_price banana_price : ℚ),
  pear_price - orange_price = banana_price →
  orange_price + pear_price = 120 →
  pear_price = 90 →
  200 * banana_price + (24000 - 200 * banana_price) / orange_price = 400 →
  (24000 - 200 * banana_price) / orange_price / 200 = 2 :=
by
  sorry

#check orange_banana_ratio

end NUMINAMATH_CALUDE_orange_banana_ratio_l1615_161582


namespace NUMINAMATH_CALUDE_lucy_lovely_age_problem_l1615_161530

theorem lucy_lovely_age_problem (lucy_age : ℕ) (lovely_age : ℕ) (years_until_twice : ℕ) : 
  lucy_age = 50 →
  lucy_age - 5 = 3 * (lovely_age - 5) →
  lucy_age + years_until_twice = 2 * (lovely_age + years_until_twice) →
  years_until_twice = 10 := by
sorry

end NUMINAMATH_CALUDE_lucy_lovely_age_problem_l1615_161530


namespace NUMINAMATH_CALUDE_parabola_equation_from_distances_l1615_161580

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = 2 * C.p * y

/-- Theorem: If a point on the parabola is 8 units from the focus and 6 units from the x-axis,
    then the parabola's equation is x^2 = 8y -/
theorem parabola_equation_from_distances (C : Parabola) (P : PointOnParabola C)
    (h_focus : Real.sqrt ((P.x)^2 + (P.y - C.p/2)^2) = 8)
    (h_xaxis : P.y = 6) :
    C.p = 4 ∧ ∀ (x y : ℝ), x^2 = 2 * C.p * y ↔ x^2 = 8 * y := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_from_distances_l1615_161580


namespace NUMINAMATH_CALUDE_root_and_c_value_l1615_161548

theorem root_and_c_value (x : ℝ) (c : ℝ) : 
  (2 + Real.sqrt 3)^2 - 4*(2 + Real.sqrt 3) + c = 0 →
  (∃ y : ℝ, y ≠ 2 + Real.sqrt 3 ∧ y^2 - 4*y + c = 0 ∧ y = 2 - Real.sqrt 3) ∧
  c = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_and_c_value_l1615_161548


namespace NUMINAMATH_CALUDE_distance_is_150_l1615_161512

/-- The distance between point A and point B in kilometers. -/
def distance : ℝ := 150

/-- The original speed of the car in kilometers per hour. -/
def original_speed : ℝ := sorry

/-- The original travel time in hours. -/
def original_time : ℝ := sorry

/-- Condition 1: If the car's speed is increased by 20%, the car can arrive 25 minutes earlier. -/
axiom condition1 : distance / (original_speed * 1.2) = original_time - 25 / 60

/-- Condition 2: If the car travels 100 kilometers at the original speed and then increases its speed by 25%, the car can arrive 10 minutes earlier. -/
axiom condition2 : 100 / original_speed + (distance - 100) / (original_speed * 1.25) = original_time - 10 / 60

/-- The theorem stating that the distance between point A and point B is 150 kilometers. -/
theorem distance_is_150 : distance = 150 := by sorry

end NUMINAMATH_CALUDE_distance_is_150_l1615_161512


namespace NUMINAMATH_CALUDE_ali_sold_ten_books_tuesday_l1615_161575

/-- The number of books Ali sold on Tuesday -/
def books_sold_tuesday (initial_stock : ℕ) (sold_monday : ℕ) (sold_wednesday : ℕ)
  (sold_thursday : ℕ) (sold_friday : ℕ) (not_sold : ℕ) : ℕ :=
  initial_stock - not_sold - (sold_monday + sold_wednesday + sold_thursday + sold_friday)

/-- Theorem stating that Ali sold 10 books on Tuesday -/
theorem ali_sold_ten_books_tuesday :
  books_sold_tuesday 800 60 20 44 66 600 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ali_sold_ten_books_tuesday_l1615_161575


namespace NUMINAMATH_CALUDE_custom_op_two_five_l1615_161558

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 4 * a + 3 * b

-- State the theorem
theorem custom_op_two_five : custom_op 2 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_two_five_l1615_161558


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1615_161505

/-- An arithmetic sequence {a_n} with its partial sums S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- Theorem: For an arithmetic sequence, if S_4 = 25 and S_8 = 100, then S_12 = 225 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.S 4 = 25) (h2 : seq.S 8 = 100) : seq.S 12 = 225 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1615_161505


namespace NUMINAMATH_CALUDE_even_function_period_2_equivalence_l1615_161520

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

theorem even_function_period_2_equivalence (f : ℝ → ℝ) (h : is_even f) :
  (∀ x, f (1 - x) = f (1 + x)) ↔ has_period_2 f :=
sorry

end NUMINAMATH_CALUDE_even_function_period_2_equivalence_l1615_161520
