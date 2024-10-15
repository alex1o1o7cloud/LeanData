import Mathlib

namespace NUMINAMATH_CALUDE_sector_central_angle_l287_28715

theorem sector_central_angle (r : ℝ) (A : ℝ) (θ : ℝ) : 
  r = 2 → A = 4 → A = (1/2) * r^2 * θ → θ = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l287_28715


namespace NUMINAMATH_CALUDE_process_flowchart_is_most_appropriate_l287_28716

/-- Represents a tool for describing production steps --/
structure ProductionDescriptionTool where
  name : String
  divides_into_processes : Bool
  uses_rectangular_boxes : Bool
  notes_process_info : Bool
  uses_flow_lines : Bool
  can_note_time : Bool

/-- Defines the properties of a Process Flowchart --/
def process_flowchart : ProductionDescriptionTool :=
  { name := "Process Flowchart",
    divides_into_processes := true,
    uses_rectangular_boxes := true,
    notes_process_info := true,
    uses_flow_lines := true,
    can_note_time := true }

/-- Theorem stating that a Process Flowchart is the most appropriate tool for describing production steps --/
theorem process_flowchart_is_most_appropriate :
  ∀ (tool : ProductionDescriptionTool),
    tool.divides_into_processes ∧
    tool.uses_rectangular_boxes ∧
    tool.notes_process_info ∧
    tool.uses_flow_lines ∧
    tool.can_note_time →
    tool = process_flowchart :=
by sorry

end NUMINAMATH_CALUDE_process_flowchart_is_most_appropriate_l287_28716


namespace NUMINAMATH_CALUDE_birds_in_dozens_l287_28717

def total_birds : ℕ := 96

theorem birds_in_dozens : (total_birds / 12 : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_dozens_l287_28717


namespace NUMINAMATH_CALUDE_remainder_problem_l287_28774

theorem remainder_problem (n : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 → n % 39 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l287_28774


namespace NUMINAMATH_CALUDE_condition_A_implies_A_eq_pi_third_condition_D_implies_A_eq_pi_third_l287_28705

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Theorem for condition A
theorem condition_A_implies_A_eq_pi_third (t : Triangle) 
  (h1 : t.a = 7) (h2 : t.b = 8) (h3 : t.c = 5) : 
  t.A = π / 3 := by sorry

-- Theorem for condition D
theorem condition_D_implies_A_eq_pi_third (t : Triangle) 
  (h : 2 * Real.sin (t.B / 2 + t.C / 2) ^ 2 + Real.cos (2 * t.A) = 1) : 
  t.A = π / 3 := by sorry

end NUMINAMATH_CALUDE_condition_A_implies_A_eq_pi_third_condition_D_implies_A_eq_pi_third_l287_28705


namespace NUMINAMATH_CALUDE_function_property_l287_28748

theorem function_property (f : ℝ → ℝ) (h : ∀ x, f (Real.sin x) = Real.sin (2011 * x)) :
  ∀ x, f (Real.cos x) = Real.cos (2011 * x) := by
  sorry

end NUMINAMATH_CALUDE_function_property_l287_28748


namespace NUMINAMATH_CALUDE_solve_equation_l287_28713

theorem solve_equation (k l x : ℚ) 
  (eq1 : 3/4 = k/88)
  (eq2 : 3/4 = (k+l)/120)
  (eq3 : 3/4 = (x-l)/160) : 
  x = 144 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l287_28713


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l287_28781

/-- Given a cube with surface area 150 square units, its volume is 125 cubic units. -/
theorem cube_volume_from_surface_area :
  ∀ s : ℝ, s > 0 → 6 * s^2 = 150 → s^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l287_28781


namespace NUMINAMATH_CALUDE_max_food_per_guest_l287_28775

theorem max_food_per_guest (total_food : ℝ) (min_guests : ℕ) 
  (h1 : total_food = 411)
  (h2 : min_guests = 165) :
  total_food / min_guests = 411 / 165 := by
  sorry

end NUMINAMATH_CALUDE_max_food_per_guest_l287_28775


namespace NUMINAMATH_CALUDE_exercise_books_count_l287_28754

/-- Given a shop with pencils, pens, and exercise books in the ratio 14 : 4 : 3,
    and 140 pencils, prove that there are 30 exercise books. -/
theorem exercise_books_count (pencils : ℕ) (pens : ℕ) (books : ℕ) : 
  pencils = 140 →
  pencils / 14 = pens / 4 →
  pencils / 14 = books / 3 →
  books = 30 := by
sorry

end NUMINAMATH_CALUDE_exercise_books_count_l287_28754


namespace NUMINAMATH_CALUDE_max_radius_third_jar_l287_28759

theorem max_radius_third_jar (pot_diameter : ℝ) (jar1_radius : ℝ) (jar2_radius : ℝ) :
  pot_diameter = 36 →
  jar1_radius = 6 →
  jar2_radius = 12 →
  ∃ (max_radius : ℝ),
    max_radius = 36 / 7 ∧
    ∀ (r : ℝ), r > max_radius →
      ¬ (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
        (x1^2 + y1^2 ≤ (pot_diameter/2)^2) ∧
        (x2^2 + y2^2 ≤ (pot_diameter/2)^2) ∧
        (x3^2 + y3^2 ≤ (pot_diameter/2)^2) ∧
        ((x1 - x2)^2 + (y1 - y2)^2 ≥ (jar1_radius + jar2_radius)^2) ∧
        ((x1 - x3)^2 + (y1 - y3)^2 ≥ (jar1_radius + r)^2) ∧
        ((x2 - x3)^2 + (y2 - y3)^2 ≥ (jar2_radius + r)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_max_radius_third_jar_l287_28759


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_count_l287_28761

/-- Number of elements in a row of Pascal's Triangle -/
def pascal_row_count (n : ℕ) : ℕ := n + 1

/-- Sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of elements in the first 30 rows of Pascal's Triangle is 465 -/
theorem pascal_triangle_30_rows_count : sum_first_n 30 = 465 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_count_l287_28761


namespace NUMINAMATH_CALUDE_perpendicular_line_through_circle_center_l287_28752

/-- Given a circle with equation x^2 + 2x + y^2 = 0 and a line x + y = 0,
    prove that x - y + 1 = 0 is the equation of the line passing through
    the center of the circle and perpendicular to the given line. -/
theorem perpendicular_line_through_circle_center :
  let circle : ℝ × ℝ → Prop := λ p => p.1^2 + 2*p.1 + p.2^2 = 0
  let given_line : ℝ × ℝ → Prop := λ p => p.1 + p.2 = 0
  let perpendicular_line : ℝ × ℝ → Prop := λ p => p.1 - p.2 + 1 = 0
  let center : ℝ × ℝ := (-1, 0)
  (∀ p, circle p ↔ (p.1 + 1)^2 + p.2^2 = 1) →
  perpendicular_line center ∧
  (∀ p q : ℝ × ℝ, p ≠ q →
    given_line p ∧ given_line q →
    perpendicular_line p ∧ perpendicular_line q →
    (p.1 - q.1) * (p.1 - q.1 + q.2 - p.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_circle_center_l287_28752


namespace NUMINAMATH_CALUDE_triangle_theorem_l287_28731

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sqrt 3 * t.b * sin t.C = t.c * cos t.B + t.c) :
  t.B = π / 3 ∧ 
  (t.b ^ 2 = t.a * t.c → 1 / tan t.A + 1 / tan t.C = 2 * Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l287_28731


namespace NUMINAMATH_CALUDE_cylinder_sphere_min_volume_l287_28742

/-- Given a cylinder with lateral surface area 4π and an external tangent sphere,
    prove that the total surface area of the cylinder is 6π when the volume of the sphere is minimum -/
theorem cylinder_sphere_min_volume (r h : ℝ) : 
  r > 0 → h > 0 →
  2 * Real.pi * r * h = 4 * Real.pi →
  (∀ R : ℝ, R > 0 → R^2 ≥ r^2 + (h/2)^2) →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 6 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_sphere_min_volume_l287_28742


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l287_28784

theorem tangent_equation_solution :
  ∃! x : Real, 0 ≤ x ∧ x ≤ 180 ∧
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l287_28784


namespace NUMINAMATH_CALUDE_one_root_of_sum_equation_l287_28735

/-- A reduced quadratic trinomial with two distinct roots -/
structure ReducedQuadraticTrinomial where
  b : ℝ
  c : ℝ
  has_distinct_roots : b^2 - 4*c > 0

/-- The discriminant of a quadratic trinomial -/
def discriminant (f : ReducedQuadraticTrinomial) : ℝ := f.b^2 - 4*f.c

/-- The quadratic function corresponding to a ReducedQuadraticTrinomial -/
def quad_function (f : ReducedQuadraticTrinomial) (x : ℝ) : ℝ := x^2 + f.b * x + f.c

/-- The theorem stating that f(x) + f(x - √D) = 0 has exactly one root -/
theorem one_root_of_sum_equation (f : ReducedQuadraticTrinomial) :
  ∃! x : ℝ, quad_function f x + quad_function f (x - Real.sqrt (discriminant f)) = 0 :=
sorry

end NUMINAMATH_CALUDE_one_root_of_sum_equation_l287_28735


namespace NUMINAMATH_CALUDE_kyle_age_l287_28792

def age_problem (casey shelley kyle julian frederick tyson : ℕ) : Prop :=
  (shelley + 3 = kyle) ∧
  (shelley = julian + 4) ∧
  (julian + 20 = frederick) ∧
  (frederick = 2 * tyson) ∧
  (tyson = 2 * casey) ∧
  (casey = 15)

theorem kyle_age :
  ∀ casey shelley kyle julian frederick tyson : ℕ,
  age_problem casey shelley kyle julian frederick tyson →
  kyle = 47 :=
by
  sorry

end NUMINAMATH_CALUDE_kyle_age_l287_28792


namespace NUMINAMATH_CALUDE_min_value_x_l287_28737

theorem min_value_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ 2 * Real.log 3 - (1/3) * Real.log x) : x ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_l287_28737


namespace NUMINAMATH_CALUDE_choose_and_assign_roles_l287_28778

/-- The number of members in the group -/
def group_size : ℕ := 4

/-- The number of roles to be assigned -/
def roles_count : ℕ := 3

/-- The number of ways to choose and assign roles -/
def ways_to_choose_and_assign : ℕ := group_size * (group_size - 1) * (group_size - 2)

theorem choose_and_assign_roles :
  ways_to_choose_and_assign = 24 :=
sorry

end NUMINAMATH_CALUDE_choose_and_assign_roles_l287_28778


namespace NUMINAMATH_CALUDE_work_time_relation_l287_28796

/-- Given three workers A, B, and C, where:
    - A takes m times as long to do a piece of work as B and C together
    - B takes n times as long as C and A together
    - C takes x times as long as A and B together
    This theorem proves that x = (m + n + 2) / (mn - 1) -/
theorem work_time_relation (m n x : ℝ) (hm : m > 0) (hn : n > 0) (hx : x > 0)
  (hA : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/a = m * (1/(b+c)))
  (hB : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/b = n * (1/(a+c)))
  (hC : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/c = x * (1/(a+b))) :
  x = (m + n + 2) / (m * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_work_time_relation_l287_28796


namespace NUMINAMATH_CALUDE_arithmetic_sum_specific_l287_28720

/-- The sum of an arithmetic sequence with given parameters -/
def arithmetic_sum (a₁ aₙ : Int) (d : Int) : Int :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic sequence from -39 to -1 with common difference 2 is -400 -/
theorem arithmetic_sum_specific : arithmetic_sum (-39) (-1) 2 = -400 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_specific_l287_28720


namespace NUMINAMATH_CALUDE_advertising_department_size_l287_28762

theorem advertising_department_size 
  (total_employees : ℕ) 
  (sample_size : ℕ) 
  (selected_from_ad : ℕ) 
  (h1 : total_employees = 1000)
  (h2 : sample_size = 80)
  (h3 : selected_from_ad = 4) :
  (selected_from_ad : ℚ) / (sample_size : ℚ) = (50 : ℚ) / (total_employees : ℚ) :=
by
  sorry

#check advertising_department_size

end NUMINAMATH_CALUDE_advertising_department_size_l287_28762


namespace NUMINAMATH_CALUDE_simplify_expression_l287_28772

theorem simplify_expression : 
  2 - (2 / (1 + Real.sqrt 2)) - (2 / (1 - Real.sqrt 2)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l287_28772


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l287_28709

/-- Given a line L1 with equation 2x + y - 5 = 0 and a point A(1, 2),
    the line L2 passing through A and perpendicular to L1 has equation x - 2y + 3 = 0 -/
theorem perpendicular_line_equation :
  let L1 : ℝ → ℝ → Prop := fun x y ↦ 2 * x + y - 5 = 0
  let A : ℝ × ℝ := (1, 2)
  let L2 : ℝ → ℝ → Prop := fun x y ↦ x - 2 * y + 3 = 0
  (∀ x y, L2 x y ↔ (y - A.2 = -(1 / (2 : ℝ)) * (x - A.1))) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → L2 x₁ y₁ → L2 x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (2 : ℝ) + (y₂ - y₁) * (1 : ℝ)) * ((x₂ - x₁) * (1 : ℝ) + (y₂ - y₁) * (-2 : ℝ)) = 0) ∧
  L2 A.1 A.2 :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l287_28709


namespace NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l287_28726

theorem solution_set_absolute_value_inequality :
  {x : ℝ | |x - 1| < 4} = Set.Ioo (-3) 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l287_28726


namespace NUMINAMATH_CALUDE_second_train_start_time_l287_28704

/-- The time when the trains meet, in hours after midnight -/
def meeting_time : ℝ := 12

/-- The time when the first train starts, in hours after midnight -/
def train1_start_time : ℝ := 7

/-- The speed of the first train in km/h -/
def train1_speed : ℝ := 20

/-- The speed of the second train in km/h -/
def train2_speed : ℝ := 25

/-- The distance between stations A and B in km -/
def total_distance : ℝ := 200

/-- The theorem stating that the second train must have started at 8 a.m. -/
theorem second_train_start_time :
  ∃ (train2_start_time : ℝ),
    train2_start_time = 8 ∧
    (meeting_time - train1_start_time) * train1_speed +
    (meeting_time - train2_start_time) * train2_speed = total_distance :=
by sorry

end NUMINAMATH_CALUDE_second_train_start_time_l287_28704


namespace NUMINAMATH_CALUDE_married_couple_survival_probability_l287_28741

/-- The probability problem for a married couple's survival over 10 years -/
theorem married_couple_survival_probability 
  (p_man : ℝ) 
  (p_neither : ℝ) 
  (h_man : p_man = 1/4) 
  (h_neither : p_neither = 1/2) : 
  ∃ p_wife : ℝ, 
    p_wife = 1/3 ∧ 
    p_neither = 1 - (p_man + p_wife - p_man * p_wife) := by
  sorry

end NUMINAMATH_CALUDE_married_couple_survival_probability_l287_28741


namespace NUMINAMATH_CALUDE_set_union_problem_l287_28738

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem set_union_problem (x : ℝ) :
  (A x ∩ B x = {9}) → (A x ∪ B x = {-8, -4, 4, -7, 9}) :=
by sorry

end NUMINAMATH_CALUDE_set_union_problem_l287_28738


namespace NUMINAMATH_CALUDE_max_sum_of_logs_l287_28770

-- Define the logarithm function (base 2)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem max_sum_of_logs (x y : ℝ) (h1 : x + y = 4) (h2 : x > 0) (h3 : y > 0) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 4 → lg a + lg b ≤ lg 4) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 4 ∧ lg a + lg b = lg 4) :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_logs_l287_28770


namespace NUMINAMATH_CALUDE_radical_product_simplification_l287_28765

theorem radical_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 63 * q * Real.sqrt (2 * q) := by
  sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l287_28765


namespace NUMINAMATH_CALUDE_min_average_books_borrowed_l287_28751

theorem min_average_books_borrowed (total_students : ℕ) 
  (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 25)
  (h2 : zero_books = 2)
  (h3 : one_book = 12)
  (h4 : two_books = 4)
  (h5 : zero_books + one_book + two_books < total_students) :
  let remaining_students := total_students - (zero_books + one_book + two_books)
  let min_total_books := one_book * 1 + two_books * 2 + remaining_students * 3
  (min_total_books : ℚ) / total_students ≥ 1.64 := by
  sorry

end NUMINAMATH_CALUDE_min_average_books_borrowed_l287_28751


namespace NUMINAMATH_CALUDE_polynomial_simplification_l287_28780

theorem polynomial_simplification (x : ℝ) :
  (3 * x^2 + 5 * x + 9) * (x + 2) - (x + 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x + 2) * (x + 4) =
  6 * x^3 - 28 * x^2 - 59 * x + 42 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l287_28780


namespace NUMINAMATH_CALUDE_unique_m_satisfying_conditions_l287_28789

theorem unique_m_satisfying_conditions : ∃! m : ℤ,
  (∃ x : ℤ, (m * x - 1) / (x - 1) = 2 + 1 / (1 - x)) ∧
  (4 - 2 * (m - 1) * (1 / 2) ≥ 0) ∧
  m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_satisfying_conditions_l287_28789


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l287_28764

def C : Set Nat := {54, 56, 59, 63, 65}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ (∃ (p : Nat), Nat.Prime p ∧ p ∣ n ∧
    ∀ (m : Nat) (q : Nat), m ∈ C → Nat.Prime q → q ∣ m → p ≤ q) ∧
  (∀ (m : Nat) (q : Nat), m ∈ C → Nat.Prime q → q ∣ m → 2 ≤ q) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l287_28764


namespace NUMINAMATH_CALUDE_solution_to_equation_l287_28724

theorem solution_to_equation : 
  {x : ℝ | x = (1/x) + (-x)^2 + 3} = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l287_28724


namespace NUMINAMATH_CALUDE_fish_lost_calculation_l287_28787

/-- The number of fish Alex lost back to the lake in the fishing tournament -/
def fish_lost : ℕ := by sorry

theorem fish_lost_calculation (jacob_initial : ℕ) (alex_multiplier : ℕ) (jacob_additional : ℕ) :
  jacob_initial = 8 →
  alex_multiplier = 7 →
  jacob_additional = 26 →
  fish_lost = alex_multiplier * jacob_initial - (jacob_initial + jacob_additional + 1) := by sorry

end NUMINAMATH_CALUDE_fish_lost_calculation_l287_28787


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l287_28768

theorem arithmetic_geometric_mean_inequality 
  (a b c d x y : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_arithmetic : b - a = c - b ∧ c - b = d - c) 
  (h_x : x = (a + d) / 2) 
  (h_y : y = Real.sqrt (b * c)) : 
  x ≥ y := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l287_28768


namespace NUMINAMATH_CALUDE_expression_simplification_l287_28718

theorem expression_simplification (x y : ℝ) : (x - 2*y) * (x + 2*y) - x * (x - y) = -4*y^2 + x*y := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l287_28718


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l287_28730

/-- Given a parallelogram with height 6 meters and area 72 square meters, its base is 12 meters. -/
theorem parallelogram_base_length (height : ℝ) (area : ℝ) (base : ℝ) : 
  height = 6 → area = 72 → area = base * height → base = 12 := by sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l287_28730


namespace NUMINAMATH_CALUDE_average_weight_decrease_l287_28734

/-- Calculates the decrease in average weight when a new person is added to a group --/
theorem average_weight_decrease (initial_count : ℕ) (initial_average : ℝ) (new_weight : ℝ) :
  initial_count = 20 →
  initial_average = 60 →
  new_weight = 45 →
  let total_weight := initial_count * initial_average
  let new_total_weight := total_weight + new_weight
  let new_count := initial_count + 1
  let new_average := new_total_weight / new_count
  abs (initial_average - new_average - 0.71) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l287_28734


namespace NUMINAMATH_CALUDE_oldest_child_age_l287_28722

theorem oldest_child_age (age1 age2 age3 : ℕ) : 
  age1 = 6 → age2 = 8 → (age1 + age2 + age3) / 3 = 10 → age3 = 16 := by
sorry

end NUMINAMATH_CALUDE_oldest_child_age_l287_28722


namespace NUMINAMATH_CALUDE_function_properties_l287_28776

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem function_properties (a : ℝ) :
  (∀ x, x < -1 ∨ x > 3 → ∀ y, y > x → f y a < f x a) ∧
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f y a ≤ f x a) ∧
  (f 2 a = 20) →
  a = -2 ∧
  (∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f x a ≤ f y a ∧ f x a = -7) :=
by sorry

#check function_properties

end NUMINAMATH_CALUDE_function_properties_l287_28776


namespace NUMINAMATH_CALUDE_f_is_quadratic_l287_28727

/-- A quadratic equation in x is an equation of the form ax^2 + bx + c = 0,
    where a, b, and c are constants and a ≠ 0. -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x^2 + 3x - 5 -/
def f (x : ℝ) : ℝ := x^2 + 3*x - 5

/-- Theorem: f(x) = x^2 + 3x - 5 is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l287_28727


namespace NUMINAMATH_CALUDE_amelia_win_probability_l287_28723

/-- Probability of Amelia winning the coin toss game -/
theorem amelia_win_probability (amelia_heads_prob : ℚ) (blaine_heads_prob : ℚ)
  (h_amelia : amelia_heads_prob = 1/4)
  (h_blaine : blaine_heads_prob = 3/7) :
  let p := amelia_heads_prob + (1 - amelia_heads_prob) * (1 - blaine_heads_prob) * p
  p = 7/16 := by sorry

end NUMINAMATH_CALUDE_amelia_win_probability_l287_28723


namespace NUMINAMATH_CALUDE_football_lineup_combinations_l287_28733

theorem football_lineup_combinations (total_players : ℕ) 
  (offensive_linemen : ℕ) (running_backs : ℕ) : 
  total_players = 12 → offensive_linemen = 3 → running_backs = 4 →
  (offensive_linemen * running_backs * (total_players - 2) * (total_players - 3) = 1080) := by
  sorry

end NUMINAMATH_CALUDE_football_lineup_combinations_l287_28733


namespace NUMINAMATH_CALUDE_probability_two_black_marbles_l287_28763

/-- The probability of drawing two black marbles without replacement from a jar -/
theorem probability_two_black_marbles (blue yellow black : ℕ) 
  (h_blue : blue = 4)
  (h_yellow : yellow = 5)
  (h_black : black = 12) : 
  (black / (blue + yellow + black)) * ((black - 1) / (blue + yellow + black - 1)) = 11 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_black_marbles_l287_28763


namespace NUMINAMATH_CALUDE_max_value_of_expression_l287_28721

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 9 ∧ (x - y)^2 + (y - z)^2 + (z - x)^2 ≥ (a - b)^2 + (b - c)^2 + (c - a)^2) ∧
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l287_28721


namespace NUMINAMATH_CALUDE_beth_sold_coins_l287_28769

theorem beth_sold_coins (initial_coins carl_gift kept_coins : ℕ) 
  (h1 : initial_coins = 250)
  (h2 : carl_gift = 75)
  (h3 : kept_coins = 135) :
  initial_coins + carl_gift - kept_coins = 190 :=
by
  sorry

end NUMINAMATH_CALUDE_beth_sold_coins_l287_28769


namespace NUMINAMATH_CALUDE_sum_of_intercepts_l287_28702

-- Define the parabola function
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

-- Define the x-intercept
def a : ℝ := parabola 0

-- Define the y-intercepts as roots of the equation 0 = 3y^2 - 9y + 4
def y_intercepts : Set ℝ := {y : ℝ | parabola y = 0}

-- Theorem statement
theorem sum_of_intercepts :
  ∃ (b c : ℝ), y_intercepts = {b, c} ∧ a + b + c = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_intercepts_l287_28702


namespace NUMINAMATH_CALUDE_equal_selection_probability_all_students_equal_probability_l287_28743

/-- Represents the probability of a student being selected -/
def selection_probability (total_students : ℕ) (eliminated : ℕ) (selected : ℕ) : ℚ :=
  selected / (total_students - eliminated)

/-- The selection method results in equal probability for all students -/
theorem equal_selection_probability 
  (total_students : ℕ) 
  (eliminated : ℕ) 
  (selected : ℕ) 
  (h1 : total_students = 2004) 
  (h2 : eliminated = 4) 
  (h3 : selected = 50) :
  selection_probability total_students eliminated selected = 1 / 40 :=
sorry

/-- The probability of selection is the same for all students -/
theorem all_students_equal_probability 
  (student1 student2 : ℕ) 
  (h_student1 : student1 ≤ 2004) 
  (h_student2 : student2 ≤ 2004) :
  selection_probability 2004 4 50 = selection_probability 2004 4 50 :=
sorry

end NUMINAMATH_CALUDE_equal_selection_probability_all_students_equal_probability_l287_28743


namespace NUMINAMATH_CALUDE_area_of_specific_trapezoid_l287_28729

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- One of the base angles in radians -/
  baseAngle : ℝ
  /-- Condition: The trapezoid is isosceles -/
  isIsosceles : True
  /-- Condition: The trapezoid is circumscribed around a circle -/
  isCircumscribed : True

/-- Calculate the area of the isosceles trapezoid -/
def areaOfTrapezoid (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The area of the specific isosceles trapezoid is 180 -/
theorem area_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := {
    longerBase := 20,
    baseAngle := Real.arctan 1.5,
    isIsosceles := True.intro,
    isCircumscribed := True.intro
  }
  areaOfTrapezoid t = 180 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_trapezoid_l287_28729


namespace NUMINAMATH_CALUDE_positive_solution_of_equation_l287_28747

theorem positive_solution_of_equation : ∃ (x : ℝ), 
  x > 0 ∧ 
  (1/3) * (4*x^2 - 1) = (x^2 - 60*x - 12) * (x^2 + 30*x + 6) ∧ 
  x = 30 + 2 * Real.sqrt 231 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_of_equation_l287_28747


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l287_28732

theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let plain_pizza_cost : ℚ := 12
  let mushroom_cost : ℚ := 3
  let bob_mushroom_slices : ℕ := total_slices / 3
  let bob_plain_slices : ℕ := 3
  let alice_slices : ℕ := total_slices - (bob_mushroom_slices + bob_plain_slices)
  let total_cost : ℚ := plain_pizza_cost + mushroom_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let bob_payment : ℚ := (bob_mushroom_slices + bob_plain_slices) * cost_per_slice
  let alice_payment : ℚ := alice_slices * (plain_pizza_cost / total_slices)
  bob_payment - alice_payment = 3.75 := by sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l287_28732


namespace NUMINAMATH_CALUDE_net_gain_calculation_l287_28771

def initial_value : ℝ := 500000

def first_sale_profit : ℝ := 0.15
def first_buyback_loss : ℝ := 0.05
def second_sale_profit : ℝ := 0.10
def final_buyback_loss : ℝ := 0.10

def first_sale (value : ℝ) : ℝ := value * (1 + first_sale_profit)
def first_buyback (value : ℝ) : ℝ := value * (1 - first_buyback_loss)
def second_sale (value : ℝ) : ℝ := value * (1 + second_sale_profit)
def final_buyback (value : ℝ) : ℝ := value * (1 - final_buyback_loss)

def total_sales (v : ℝ) : ℝ := first_sale v + second_sale (first_buyback (first_sale v))
def total_purchases (v : ℝ) : ℝ := first_buyback (first_sale v) + final_buyback (second_sale (first_buyback (first_sale v)))

theorem net_gain_calculation (v : ℝ) : 
  total_sales v - total_purchases v = 88837.50 :=
by sorry

end NUMINAMATH_CALUDE_net_gain_calculation_l287_28771


namespace NUMINAMATH_CALUDE_problem_solution_l287_28706

theorem problem_solution (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) : 
  (∃ k : ℕ, m + 1 = 4 * k - 1 ∧ Nat.Prime (m + 1)) →
  (∃ p a : ℕ, Nat.Prime p ∧ (m^(2^n - 1) - 1) / (m - 1) = m^n + p^a) →
  (∃ p : ℕ, Nat.Prime p ∧ p = 4 * (p / 4) - 1 ∧ m = p - 1 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l287_28706


namespace NUMINAMATH_CALUDE_income_percentage_difference_l287_28746

/-- Given the monthly incomes of A, B, and C, prove that B's income is 12% more than C's. -/
theorem income_percentage_difference :
  ∀ (A_annual B_monthly C_monthly : ℝ),
  A_annual = 436800.0000000001 →
  C_monthly = 13000 →
  A_annual / 12 / B_monthly = 5 / 2 →
  (B_monthly - C_monthly) / C_monthly = 0.12 :=
by
  sorry

end NUMINAMATH_CALUDE_income_percentage_difference_l287_28746


namespace NUMINAMATH_CALUDE_divisible_by_sum_of_digits_l287_28703

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem divisible_by_sum_of_digits :
  ∀ n : ℕ, n ≤ 1988 →
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 17 ∧ k % (sum_of_digits k) = 0 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_sum_of_digits_l287_28703


namespace NUMINAMATH_CALUDE_output_for_three_l287_28760

def f (a : ℤ) : ℤ :=
  if a < 10 then 2 * a else a + 1

theorem output_for_three :
  f 3 = 6 :=
by sorry

end NUMINAMATH_CALUDE_output_for_three_l287_28760


namespace NUMINAMATH_CALUDE_john_age_is_thirteen_l287_28700

/-- Represents John's work and earnings over a six-month period --/
structure JohnWork where
  hoursPerDay : ℕ
  hourlyRatePerAge : ℚ
  weeklyBonusThreshold : ℕ
  weeklyBonus : ℚ
  totalDaysWorked : ℕ
  totalEarned : ℚ

/-- Calculates John's age based on his work and earnings --/
def calculateAge (work : JohnWork) : ℕ :=
  sorry

/-- Theorem stating that John's calculated age is 13 --/
theorem john_age_is_thirteen (work : JohnWork) 
  (h1 : work.hoursPerDay = 3)
  (h2 : work.hourlyRatePerAge = 1/2)
  (h3 : work.weeklyBonusThreshold = 3)
  (h4 : work.weeklyBonus = 5)
  (h5 : work.totalDaysWorked = 75)
  (h6 : work.totalEarned = 900) :
  calculateAge work = 13 :=
sorry

end NUMINAMATH_CALUDE_john_age_is_thirteen_l287_28700


namespace NUMINAMATH_CALUDE_gcd_count_l287_28798

def count_gcd_values (a b : ℕ) : Prop :=
  (Nat.gcd a b * Nat.lcm a b = 360) →
  (∃ (S : Finset ℕ), (∀ x ∈ S, ∃ (c d : ℕ), Nat.gcd c d = x ∧ Nat.gcd c d * Nat.lcm c d = 360) ∧
                     (∀ y, (∃ (e f : ℕ), Nat.gcd e f = y ∧ Nat.gcd e f * Nat.lcm e f = 360) → y ∈ S) ∧
                     S.card = 14)

theorem gcd_count : ∀ a b : ℕ, count_gcd_values a b :=
sorry

end NUMINAMATH_CALUDE_gcd_count_l287_28798


namespace NUMINAMATH_CALUDE_problem_trip_mpg_l287_28790

/-- Represents a car trip with odometer readings and gas fill amounts -/
structure CarTrip where
  initial_odometer : ℕ
  final_odometer : ℕ
  gas_fills : List ℕ

/-- Calculates the average miles per gallon for a car trip -/
def averageMPG (trip : CarTrip) : ℚ :=
  let total_distance := trip.final_odometer - trip.initial_odometer
  let total_gas := trip.gas_fills.sum
  (total_distance : ℚ) / total_gas

/-- The specific car trip from the problem -/
def problemTrip : CarTrip := {
  initial_odometer := 68300
  final_odometer := 69600
  gas_fills := [15, 25]
}

/-- Theorem stating that the average MPG for the problem trip is 32.5 -/
theorem problem_trip_mpg : averageMPG problemTrip = 32.5 := by
  sorry


end NUMINAMATH_CALUDE_problem_trip_mpg_l287_28790


namespace NUMINAMATH_CALUDE_pen_buyers_difference_l287_28708

theorem pen_buyers_difference (pen_cost : ℕ+) 
  (h1 : 178 % pen_cost.val = 0)
  (h2 : 252 % pen_cost.val = 0)
  (h3 : 35 * pen_cost.val ≤ 252) :
  252 / pen_cost.val - 178 / pen_cost.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_pen_buyers_difference_l287_28708


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l287_28795

theorem smallest_undefined_inverse (b : ℕ) : 
  (b > 0) → 
  (¬ ∃ x, x * b ≡ 1 [ZMOD 75]) → 
  (¬ ∃ x, x * b ≡ 1 [ZMOD 90]) → 
  (∀ a < b, a > 0 → (∃ x, x * a ≡ 1 [ZMOD 75]) ∨ (∃ x, x * a ≡ 1 [ZMOD 90])) → 
  b = 15 := by
sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l287_28795


namespace NUMINAMATH_CALUDE_R_C_S_collinear_l287_28745

-- Define the ellipse Γ
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 16 + p.2^2 / 9 = 1}

-- Define points A and B
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (4, 0)

-- Define point C outside the ellipse
variable (C : ℝ × ℝ)
axiom C_outside : C ∉ Γ

-- Define points P and Q as intersections of CA and CB with Γ
variable (P Q : ℝ × ℝ)
axiom P_on_Γ : P ∈ Γ
axiom Q_on_Γ : Q ∈ Γ
axiom P_on_CA : ∃ t : ℝ, P = (1 - t) • A + t • C
axiom Q_on_CB : ∃ t : ℝ, Q = (1 - t) • B + t • C

-- Define points R and S as intersections of tangents with Γ
variable (R S : ℝ × ℝ)
axiom R_on_Γ : R ∈ Γ
axiom S_on_Γ : S ∈ Γ

-- Define the tangent condition
def is_tangent (p q : ℝ × ℝ) : Prop := sorry

axiom AQ_tangent : is_tangent A Q
axiom BP_tangent : is_tangent B P

-- Theorem to prove
theorem R_C_S_collinear : ∃ (m b : ℝ), R.2 = m * R.1 + b ∧ C.2 = m * C.1 + b ∧ S.2 = m * S.1 + b :=
sorry

end NUMINAMATH_CALUDE_R_C_S_collinear_l287_28745


namespace NUMINAMATH_CALUDE_metallic_sheet_length_proof_l287_28797

/-- The length of a rectangular metallic sheet that forms a box of volume 24000 m³ when 10 m squares are cut from each corner. -/
def metallic_sheet_length : ℝ := 820

/-- The width of the rectangular metallic sheet. -/
def sheet_width : ℝ := 50

/-- The side length of the square cut from each corner. -/
def corner_cut : ℝ := 10

/-- The volume of the resulting box. -/
def box_volume : ℝ := 24000

theorem metallic_sheet_length_proof :
  (metallic_sheet_length - 2 * corner_cut) * (sheet_width - 2 * corner_cut) * corner_cut = box_volume :=
by sorry

end NUMINAMATH_CALUDE_metallic_sheet_length_proof_l287_28797


namespace NUMINAMATH_CALUDE_janet_weekly_income_l287_28756

/-- Calculates the total income for Janet's exterminator work and sculpture sales --/
def janet_total_income (small_res_rate : ℕ) (large_res_rate : ℕ) (commercial_rate : ℕ)
                       (small_res_hours : ℕ) (large_res_hours : ℕ) (commercial_hours : ℕ)
                       (small_sculpture_price : ℕ) (medium_sculpture_price : ℕ) (large_sculpture_price : ℕ)
                       (small_sculpture_weight : ℕ) (small_sculpture_count : ℕ)
                       (medium_sculpture_weight : ℕ) (medium_sculpture_count : ℕ)
                       (large_sculpture_weight : ℕ) (large_sculpture_count : ℕ) : ℕ :=
  let exterminator_income := small_res_rate * small_res_hours + 
                             large_res_rate * large_res_hours + 
                             commercial_rate * commercial_hours
  let sculpture_income := small_sculpture_price * small_sculpture_weight * small_sculpture_count +
                          medium_sculpture_price * medium_sculpture_weight * medium_sculpture_count +
                          large_sculpture_price * large_sculpture_weight * large_sculpture_count
  exterminator_income + sculpture_income

/-- Theorem stating that Janet's total income for the week is $2320 --/
theorem janet_weekly_income : 
  janet_total_income 70 85 100 10 5 5 20 25 30 4 2 7 1 12 1 = 2320 := by
  sorry

end NUMINAMATH_CALUDE_janet_weekly_income_l287_28756


namespace NUMINAMATH_CALUDE_ellipse_m_value_l287_28767

-- Define the ellipse equation
def ellipse_equation (x y m : ℝ) : Prop := x^2 / m + y^2 / 16 = 1

-- Define the distances from a point to the foci
def distance_to_foci (d1 d2 : ℝ) : Prop := d1 = 3 ∧ d2 = 7

-- Theorem statement
theorem ellipse_m_value (x y m : ℝ) :
  ellipse_equation x y m →
  ∃ (d1 d2 : ℝ), distance_to_foci d1 d2 →
  m = 25 := by
sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l287_28767


namespace NUMINAMATH_CALUDE_alice_and_bob_money_l287_28782

theorem alice_and_bob_money : (5 : ℚ) / 8 + (3 : ℚ) / 5 = 1.225 := by sorry

end NUMINAMATH_CALUDE_alice_and_bob_money_l287_28782


namespace NUMINAMATH_CALUDE_dogs_barking_l287_28777

theorem dogs_barking (initial_dogs : ℕ) (additional_dogs : ℕ) :
  initial_dogs = 30 →
  additional_dogs = 10 →
  initial_dogs + additional_dogs = 40 := by
sorry

end NUMINAMATH_CALUDE_dogs_barking_l287_28777


namespace NUMINAMATH_CALUDE_constant_s_value_l287_28786

theorem constant_s_value : ∃ (s : ℝ), ∀ (x : ℝ),
  (3 * x^3 - 2 * x^2 + x + 6) * (2 * x^3 + s * x^2 + 3 * x + 5) =
  6 * x^6 + s * x^5 + 5 * x^4 + 17 * x^3 + 10 * x^2 + 33 * x + 30 ∧ s = 4 := by
  sorry

end NUMINAMATH_CALUDE_constant_s_value_l287_28786


namespace NUMINAMATH_CALUDE_exponential_inequality_l287_28758

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a < b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l287_28758


namespace NUMINAMATH_CALUDE_unique_solution_cubic_l287_28793

theorem unique_solution_cubic (b : ℝ) : 
  (∃! x : ℝ, x^3 - b*x^2 - 3*b*x + b^2 - 2 = 0) ↔ b = 7/4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_l287_28793


namespace NUMINAMATH_CALUDE_not_perfect_cube_l287_28707

theorem not_perfect_cube (n : ℕ+) (h : ∃ p : ℕ, n^5 + n^3 + 2*n^2 + 2*n + 2 = p^3) :
  ¬∃ q : ℕ, 2*n^2 + n + 2 = q^3 := by
sorry

end NUMINAMATH_CALUDE_not_perfect_cube_l287_28707


namespace NUMINAMATH_CALUDE_simona_treatment_cost_l287_28712

/-- Represents the number of complexes after each treatment -/
def complexes_after_treatment (initial : ℕ) : ℕ → ℕ
| 0 => initial
| (n + 1) => (complexes_after_treatment initial n / 2) + ((complexes_after_treatment initial n + 1) / 2)

/-- The cost of treatment for a given number of cured complexes -/
def treatment_cost (cured_complexes : ℕ) : ℕ := 197 * cured_complexes

theorem simona_treatment_cost :
  ∃ (initial : ℕ),
    initial > 0 ∧
    complexes_after_treatment initial 3 = 1 ∧
    treatment_cost (initial - 1) = 1379 :=
by sorry

end NUMINAMATH_CALUDE_simona_treatment_cost_l287_28712


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_one_l287_28740

/-- Given two perpendicular lines and a quadratic function, prove that the axis of symmetry is x=1 -/
theorem axis_of_symmetry_is_one 
  (a b : ℝ) 
  (h1 : ∀ x y : ℝ, b * x + a * y = 0 → x - 2 * y + 2 = 0 → (b * 1 + a * 0) * (1 * 1 + 2 * 0) = -1) 
  (f : ℝ → ℝ) 
  (h2 : ∀ x : ℝ, f x = a * x^2 - b * x + a) : 
  ∃ p : ℝ, p = 1 ∧ ∀ x : ℝ, f (p + x) = f (p - x) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_is_one_l287_28740


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l287_28750

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  angleB : ℝ

-- Define the theorem
theorem triangle_area_theorem (t : Triangle) (h1 : t.a = Real.sqrt 3) (h2 : t.b = 1) (h3 : t.angleB = π / 6) :
  ∃ (S : ℝ), (S = Real.sqrt 3 / 2 ∨ S = Real.sqrt 3 / 4) ∧ 
  (∃ (angleA angleC : ℝ), 
    angleA + t.angleB + angleC = π ∧
    S = 1/2 * t.a * t.b * Real.sin angleC) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l287_28750


namespace NUMINAMATH_CALUDE_logo_shaded_area_l287_28749

/-- Represents a logo with specific geometric properties. -/
structure Logo where
  /-- Length of vertical straight edges and diameters of small semicircles -/
  edge_length : ℝ
  /-- Rotational symmetry property -/
  has_rotational_symmetry : Prop

/-- Calculates the shaded area of the logo -/
def shaded_area (logo : Logo) : ℝ :=
  sorry

/-- Theorem stating that the shaded area of a logo with specific properties is 4 + π -/
theorem logo_shaded_area (logo : Logo) 
  (h1 : logo.edge_length = 2)
  (h2 : logo.has_rotational_symmetry) : 
  shaded_area logo = 4 + π := by
  sorry

end NUMINAMATH_CALUDE_logo_shaded_area_l287_28749


namespace NUMINAMATH_CALUDE_painting_theorem_l287_28714

/-- The time required for two people to paint a room together, including a break -/
def paint_time (karl_time leo_time break_time : ℝ) : ℝ → Prop :=
  λ t : ℝ => (1 / karl_time + 1 / leo_time) * (t - break_time) = 1

theorem painting_theorem :
  ∃ t : ℝ, paint_time 6 8 0.5 t :=
by
  sorry

end NUMINAMATH_CALUDE_painting_theorem_l287_28714


namespace NUMINAMATH_CALUDE_number_equals_sixteen_l287_28755

theorem number_equals_sixteen : ∃ x : ℝ, 0.0025 * x = 0.04 ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_sixteen_l287_28755


namespace NUMINAMATH_CALUDE_candy_jar_problem_l287_28757

theorem candy_jar_problem (total : ℕ) (blue : ℕ) (red : ℕ) : 
  total = 3409 → 
  blue = 3264 → 
  total = red + blue → 
  red = 145 := by
sorry

end NUMINAMATH_CALUDE_candy_jar_problem_l287_28757


namespace NUMINAMATH_CALUDE_determinant_problems_l287_28736

def matrix1 : Matrix (Fin 3) (Fin 3) ℤ := !![3, 2, 1; 2, 5, 3; 3, 4, 3]

def matrix2 (a b c : ℤ) : Matrix (Fin 3) (Fin 3) ℤ := !![a, b, c; b, c, a; c, a, b]

theorem determinant_problems :
  (Matrix.det matrix1 = 8) ∧
  (∀ a b c : ℤ, Matrix.det (matrix2 a b c) = 3 * a * b * c - a^3 - b^3 - c^3) := by
  sorry

end NUMINAMATH_CALUDE_determinant_problems_l287_28736


namespace NUMINAMATH_CALUDE_min_room_dimensions_l287_28783

/-- The minimum dimensions of a rectangular room that can accommodate a 9' × 12' table --/
theorem min_room_dimensions (table_width : ℝ) (table_length : ℝ) 
  (hw : table_width = 9) (hl : table_length = 12) :
  ∃ (S T : ℝ), 
    S > T ∧ 
    S ≥ Real.sqrt (table_width^2 + table_length^2) ∧
    T ≥ max table_width table_length ∧
    ∀ (S' T' : ℝ), (S' > T' ∧ 
                    S' ≥ Real.sqrt (table_width^2 + table_length^2) ∧ 
                    T' ≥ max table_width table_length) → 
                    (S ≤ S' ∧ T ≤ T') ∧
    S = 15 ∧ T = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_room_dimensions_l287_28783


namespace NUMINAMATH_CALUDE_green_beans_count_l287_28739

def total_beans : ℕ := 572

def red_beans : ℕ := total_beans / 4

def remaining_after_red : ℕ := total_beans - red_beans

def white_beans : ℕ := remaining_after_red / 3

def remaining_after_white : ℕ := remaining_after_red - white_beans

def blue_beans : ℕ := remaining_after_white / 5

def remaining_after_blue : ℕ := remaining_after_white - blue_beans

def yellow_beans : ℕ := remaining_after_blue / 6

def remaining_after_yellow : ℕ := remaining_after_blue - yellow_beans

def green_beans : ℕ := remaining_after_yellow / 2

theorem green_beans_count : green_beans = 95 := by
  sorry

end NUMINAMATH_CALUDE_green_beans_count_l287_28739


namespace NUMINAMATH_CALUDE_gcd_n_pow_13_minus_n_l287_28791

theorem gcd_n_pow_13_minus_n : ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ℤ), (d : ℤ) ∣ (n^13 - n)) ∧ 
  (∀ (k : ℕ), k > 0 → (∀ (n : ℤ), (k : ℤ) ∣ (n^13 - n)) → k ∣ d) ∧
  d = 2730 := by
sorry

end NUMINAMATH_CALUDE_gcd_n_pow_13_minus_n_l287_28791


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l287_28788

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 ∧ b = 36 ∧ c^2 = a^2 + b^2 → c = 39 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l287_28788


namespace NUMINAMATH_CALUDE_intersection_union_sets_l287_28753

theorem intersection_union_sets : 
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {1, 2, 3, 4}
  let P : Set ℕ := {2, 3, 4, 5}
  (M ∩ N) ∪ P = {1, 2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_union_sets_l287_28753


namespace NUMINAMATH_CALUDE_clothing_purchase_properties_l287_28773

/-- Represents the clothing purchase problem for a recitation competition. -/
structure ClothingPurchase where
  total_students : Nat
  combined_cost : Nat
  cost_ratio : Nat → Nat → Prop
  boy_girl_ratio : Nat → Nat → Prop
  max_total_cost : Nat

/-- Calculates the unit prices of men's and women's clothing. -/
def calculate_unit_prices (cp : ClothingPurchase) : Nat × Nat :=
  sorry

/-- Counts the number of valid purchasing plans. -/
def count_valid_plans (cp : ClothingPurchase) : Nat :=
  sorry

/-- Calculates the minimum cost of clothing purchase. -/
def minimum_cost (cp : ClothingPurchase) : Nat :=
  sorry

/-- Main theorem proving the properties of the clothing purchase problem. -/
theorem clothing_purchase_properties (cp : ClothingPurchase) 
  (h1 : cp.total_students = 150)
  (h2 : cp.combined_cost = 220)
  (h3 : cp.cost_ratio = λ m w => 6 * m = 5 * w)
  (h4 : cp.boy_girl_ratio = λ b g => b ≤ 2 * g / 3)
  (h5 : cp.max_total_cost = 17000) :
  let (men_price, women_price) := calculate_unit_prices cp
  men_price = 100 ∧ 
  women_price = 120 ∧ 
  count_valid_plans cp = 11 ∧ 
  minimum_cost cp = 16800 :=
sorry

end NUMINAMATH_CALUDE_clothing_purchase_properties_l287_28773


namespace NUMINAMATH_CALUDE_bus_distance_calculation_l287_28725

/-- Represents a round trip journey with walking and bus ride components. -/
structure Journey where
  total_distance : ℕ
  walking_distance : ℕ
  bus_distance : ℕ

/-- 
Theorem: If a person travels a total of 50 blocks in a round trip, 
where they walk 5 blocks at the beginning and end of each leg of the trip, 
then the distance traveled by bus in one direction is 20 blocks.
-/
theorem bus_distance_calculation (j : Journey) 
  (h1 : j.total_distance = 50)
  (h2 : j.walking_distance = 5) : 
  j.bus_distance = 20 := by
  sorry

#check bus_distance_calculation

end NUMINAMATH_CALUDE_bus_distance_calculation_l287_28725


namespace NUMINAMATH_CALUDE_writer_average_speed_l287_28766

/-- Calculates the average writing speed given the words and hours for two writing sessions -/
def average_writing_speed (words1 : ℕ) (hours1 : ℕ) (words2 : ℕ) (hours2 : ℕ) : ℚ :=
  (words1 + words2 : ℚ) / (hours1 + hours2 : ℚ)

/-- Theorem stating that the average writing speed for the given sessions is 500 words per hour -/
theorem writer_average_speed :
  average_writing_speed 30000 60 50000 100 = 500 := by
  sorry

end NUMINAMATH_CALUDE_writer_average_speed_l287_28766


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l287_28744

theorem partial_fraction_decomposition :
  ∃ (P Q : ℚ), P = 22/9 ∧ Q = -4/9 ∧
  ∀ (x : ℚ), x ≠ 7 ∧ x ≠ -2 →
    (2*x + 8) / (x^2 - 5*x - 14) = P / (x - 7) + Q / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l287_28744


namespace NUMINAMATH_CALUDE_total_profit_calculation_l287_28701

-- Define the profit for 3 shirts
def profit_3_shirts : ℚ := 21

-- Define the profit for 2 pairs of sandals
def profit_2_sandals : ℚ := 4 * profit_3_shirts

-- Define the number of shirts and sandals sold
def shirts_sold : ℕ := 7
def sandals_sold : ℕ := 3

-- Theorem statement
theorem total_profit_calculation :
  (shirts_sold * (profit_3_shirts / 3) + sandals_sold * (profit_2_sandals / 2)) = 175 := by
  sorry


end NUMINAMATH_CALUDE_total_profit_calculation_l287_28701


namespace NUMINAMATH_CALUDE_right_triangle_check_l287_28779

/-- Check if three numbers form a right-angled triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_check :
  ¬ is_right_triangle (1/3) (1/4) (1/5) ∧
  is_right_triangle 3 4 5 ∧
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle 1 (Real.sqrt 3) 4 :=
by sorry

#check right_triangle_check

end NUMINAMATH_CALUDE_right_triangle_check_l287_28779


namespace NUMINAMATH_CALUDE_second_train_speed_l287_28785

/-- Calculates the speed of the second train given the parameters of two trains meeting. -/
theorem second_train_speed
  (length1 : ℝ) (length2 : ℝ) (speed1 : ℝ) (clear_time : ℝ)
  (h1 : length1 = 120) -- Length of first train in meters
  (h2 : length2 = 280) -- Length of second train in meters
  (h3 : speed1 = 42) -- Speed of first train in kmph
  (h4 : clear_time = 20 / 3600) -- Time to clear in hours
  : ∃ (speed2 : ℝ), speed2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_l287_28785


namespace NUMINAMATH_CALUDE_circle_center_trajectory_l287_28719

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m+3)*x + 2*(1-4*m^2) + 16*m^4 + 9 = 0

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  y = 4*(x-3)^2 - 1

-- Theorem statement
theorem circle_center_trajectory :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) →
  ∃ x y : ℝ, trajectory_equation x y ∧ 20/7 < x ∧ x < 4 :=
sorry

end NUMINAMATH_CALUDE_circle_center_trajectory_l287_28719


namespace NUMINAMATH_CALUDE_three_consecutive_not_divisible_by_three_l287_28711

def digit_sum (n : ℕ) : ℕ := sorry

def board_sequence (initial : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => board_sequence initial n + digit_sum (board_sequence initial n)

theorem three_consecutive_not_divisible_by_three (initial : ℕ) :
  ∃ k : ℕ, ¬(board_sequence initial k % 3 = 0) ∧
           ¬(board_sequence initial (k + 1) % 3 = 0) ∧
           ¬(board_sequence initial (k + 2) % 3 = 0) :=
sorry

end NUMINAMATH_CALUDE_three_consecutive_not_divisible_by_three_l287_28711


namespace NUMINAMATH_CALUDE_simplify_square_roots_l287_28794

theorem simplify_square_roots : (Real.sqrt 300 / Real.sqrt 75) - (Real.sqrt 200 / Real.sqrt 50) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l287_28794


namespace NUMINAMATH_CALUDE_fifteenth_term_of_geometric_sequence_l287_28728

/-- Given a geometric sequence where the first term is 12 and the common ratio is 1/3,
    the 15th term is equal to 4/531441. -/
theorem fifteenth_term_of_geometric_sequence (a : ℕ → ℚ) :
  a 1 = 12 →
  (∀ n : ℕ, a (n + 1) = a n * (1/3)) →
  a 15 = 4/531441 := by
sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_geometric_sequence_l287_28728


namespace NUMINAMATH_CALUDE_simplify_expression_1_evaluate_expression_2_l287_28710

-- Expression 1
theorem simplify_expression_1 (a : ℝ) : 
  -2 * a^2 + 3 - (3 * a^2 - 6 * a + 1) + 3 = -5 * a^2 + 6 * a + 2 := by sorry

-- Expression 2
theorem evaluate_expression_2 (x y : ℝ) (hx : x = -2) (hy : y = -3) :
  (1/2) * x - 2 * (x - (1/3) * y^2) + (-3/2 * x + (1/3) * y^2) = 15 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_evaluate_expression_2_l287_28710


namespace NUMINAMATH_CALUDE_remainder_problem_l287_28799

theorem remainder_problem (N : ℤ) : 
  ∃ k : ℤ, N = 35 * k + 25 → ∃ m : ℤ, N = 15 * m + 10 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l287_28799
