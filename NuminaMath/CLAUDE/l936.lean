import Mathlib

namespace NUMINAMATH_CALUDE_intersection_equality_l936_93651

def A : Set ℝ := {x | |x - 4| < 2 * x}
def B (a : ℝ) : Set ℝ := {x | x * (x - a) ≥ (a + 6) * (x - a)}

theorem intersection_equality (a : ℝ) : A ∩ B a = A ↔ a ≤ -14/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l936_93651


namespace NUMINAMATH_CALUDE_circles_intersect_l936_93617

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are intersecting -/
def are_intersecting (c1 c2 : Circle) : Prop :=
  let d := Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2)
  abs (c1.radius - c2.radius) < d ∧ d < c1.radius + c2.radius

theorem circles_intersect : 
  let circle1 : Circle := { center := (0, 0), radius := 2 }
  let circle2 : Circle := { center := (2, 0), radius := 3 }
  are_intersecting circle1 circle2 := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l936_93617


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l936_93670

/-- A quadratic function is a function of the form f(x) = ax² + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_theorem (f : ℝ → ℝ) 
  (h1 : IsQuadratic f) 
  (h2 : f 0 = 0) 
  (h3 : ∀ x, f (x + 1) = f x + x + 1) : 
  ∀ x, f x = (1/2) * x^2 + (1/2) * x := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l936_93670


namespace NUMINAMATH_CALUDE_intersection_equals_positive_l936_93606

-- Define sets A and B
def A : Set ℝ := {x | 2 * x^2 + x > 0}
def B : Set ℝ := {x | 2 * x + 1 > 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_positive : A_intersect_B = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_positive_l936_93606


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l936_93659

theorem tangent_line_to_circle (θ : Real) (h1 : 0 < θ ∧ θ < π) :
  (∃ t : Real, ∀ x y : Real,
    (x = t * Real.cos θ ∧ y = t * Real.sin θ) →
    (∃ α : Real, x = 4 + 2 * Real.cos α ∧ y = 2 * Real.sin α) →
    (∀ x' y' : Real, (x' - 4)^2 + y'^2 = 4 →
      (y' - y) * Real.cos θ = (x' - x) * Real.sin θ)) →
  θ = π/6 ∨ θ = 5*π/6 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l936_93659


namespace NUMINAMATH_CALUDE_x_value_proof_l936_93614

theorem x_value_proof (x : ℝ) : -(-(-(-x))) = -4 → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l936_93614


namespace NUMINAMATH_CALUDE_case_cost_is_nine_l936_93671

/-- The cost of a case of paper towels -/
def case_cost (num_rolls : ℕ) (individual_roll_cost : ℚ) (savings_percent : ℚ) : ℚ :=
  num_rolls * (individual_roll_cost * (1 - savings_percent / 100))

/-- Theorem stating the cost of a case of 12 rolls is $9 -/
theorem case_cost_is_nine :
  case_cost 12 1 25 = 9 := by
  sorry

end NUMINAMATH_CALUDE_case_cost_is_nine_l936_93671


namespace NUMINAMATH_CALUDE_sandy_shirt_cost_l936_93689

/-- The amount Sandy spent on clothes -/
def total_spent : ℝ := 33.56

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℝ := 13.99

/-- The amount Sandy spent on a jacket -/
def jacket_cost : ℝ := 7.43

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℝ := total_spent - shorts_cost - jacket_cost

theorem sandy_shirt_cost : shirt_cost = 12.14 := by
  sorry

end NUMINAMATH_CALUDE_sandy_shirt_cost_l936_93689


namespace NUMINAMATH_CALUDE_eve_last_student_l936_93685

/-- Represents the students in the circle -/
inductive Student
| Alan
| Bob
| Cara
| Dan
| Eve

/-- The order of students in the circle -/
def initialOrder : List Student := [Student.Alan, Student.Bob, Student.Cara, Student.Dan, Student.Eve]

/-- Checks if a number is a multiple of 7 or contains the digit 6 -/
def isEliminationNumber (n : Nat) : Bool :=
  n % 7 == 0 || n.repr.contains '6'

/-- Simulates the elimination process and returns the last student remaining -/
def lastStudent (order : List Student) : Student :=
  sorry

/-- Theorem stating that Eve is the last student remaining -/
theorem eve_last_student : lastStudent initialOrder = Student.Eve :=
  sorry

end NUMINAMATH_CALUDE_eve_last_student_l936_93685


namespace NUMINAMATH_CALUDE_colored_graph_color_bound_l936_93658

/-- A graph with colored edges satisfying certain properties -/
structure ColoredGraph where
  n : ℕ  -- number of vertices
  c : ℕ  -- number of colors
  edge_count : ℕ  -- number of edges
  edge_count_lower_bound : edge_count ≥ n^2 / 10
  no_incident_same_color : Bool  -- property that no two incident edges have the same color
  no_same_color_10_cycle : Bool  -- property that no cycles of size 10 have the same set of colors

/-- Main theorem: There exists a constant k such that c ≥ k * n^(8/5) for any colored graph satisfying the given properties -/
theorem colored_graph_color_bound (G : ColoredGraph) :
  ∃ (k : ℝ), G.c ≥ k * G.n^(8/5) := by
  sorry

end NUMINAMATH_CALUDE_colored_graph_color_bound_l936_93658


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_non_positive_monotonicity_positive_l936_93627

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - Real.log x - 2

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * x - 1/x

theorem tangent_line_at_one (x : ℝ) :
  f 1 1 = -(3/2) ∧ f_deriv 1 1 = 0 :=
sorry

theorem monotonicity_non_positive (a : ℝ) (x : ℝ) (ha : a ≤ 0) (hx : x > 0) :
  f_deriv a x < 0 :=
sorry

theorem monotonicity_positive (a : ℝ) (x : ℝ) (ha : a > 0) (hx : x > 0) :
  (x < Real.sqrt a / a → f_deriv a x < 0) ∧
  (x > Real.sqrt a / a → f_deriv a x > 0) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_non_positive_monotonicity_positive_l936_93627


namespace NUMINAMATH_CALUDE_log_equation_solution_l936_93653

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  4 * Real.log x / Real.log 3 = Real.log (4 * x) / Real.log 3 → x = (4 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l936_93653


namespace NUMINAMATH_CALUDE_bottle_caps_per_box_l936_93657

theorem bottle_caps_per_box (total_caps : ℕ) (num_boxes : ℕ) 
  (h1 : total_caps = 316) (h2 : num_boxes = 79) : 
  total_caps / num_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_per_box_l936_93657


namespace NUMINAMATH_CALUDE_P_consecutive_coprime_l936_93697

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define P(n) as given in the problem
def P : ℕ → ℕ
  | 0 => 0  -- Undefined in the original problem, added for completeness
  | 1 => 0  -- Undefined in the original problem, added for completeness
  | (n + 2) => 
    if n % 2 = 0 then
      (fib ((n / 2) + 1) + fib ((n / 2) - 1)) ^ 2
    else
      fib (n + 2) + fib ((n - 1) / 2)

-- State the theorem
theorem P_consecutive_coprime (k : ℕ) (h : k ≥ 3) : 
  Nat.gcd (P k) (P (k + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_P_consecutive_coprime_l936_93697


namespace NUMINAMATH_CALUDE_bowknot_equation_solution_l936_93665

-- Define the bowknot operation
noncomputable def bowknot (c d : ℝ) : ℝ :=
  c + Real.sqrt (d + Real.sqrt (d + Real.sqrt (d + Real.sqrt d)))

-- Theorem statement
theorem bowknot_equation_solution :
  ∃ x : ℝ, bowknot 3 x = 12 → x = 72 := by sorry

end NUMINAMATH_CALUDE_bowknot_equation_solution_l936_93665


namespace NUMINAMATH_CALUDE_perfect_square_condition_l936_93690

theorem perfect_square_condition (n : ℕ+) : 
  (∃ m : ℕ, n.val^2 + 5*n.val + 13 = m^2) → n.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l936_93690


namespace NUMINAMATH_CALUDE_triangle_area_l936_93645

/-- The area of a triangle with vertices A(2, 2), B(8, 2), and C(4, 10) is 24 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, 2)
  let B : ℝ × ℝ := (8, 2)
  let C : ℝ × ℝ := (4, 10)
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area = 24 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l936_93645


namespace NUMINAMATH_CALUDE_train_passing_time_l936_93625

/-- Prove that a train with given length and speed will pass a fixed point in the calculated time -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 275 →
  train_speed_kmh = 90 →
  passing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  passing_time = 11 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l936_93625


namespace NUMINAMATH_CALUDE_trucks_given_to_jeff_l936_93623

-- Define the variables
def initial_trucks : ℕ := 51
def remaining_trucks : ℕ := 38

-- Define the theorem
theorem trucks_given_to_jeff : 
  initial_trucks - remaining_trucks = 13 := by
  sorry

end NUMINAMATH_CALUDE_trucks_given_to_jeff_l936_93623


namespace NUMINAMATH_CALUDE_line_equation_correct_l936_93674

/-- The y-intercept of the line 2x + y + 2 = 0 -/
def y_intercept : ℝ := -2

/-- The point A through which line l passes -/
def point_A : ℝ × ℝ := (2, 0)

/-- The equation of line l -/
def line_equation (x y : ℝ) : Prop := x - y - 2 = 0

theorem line_equation_correct :
  (line_equation point_A.1 point_A.2) ∧
  (line_equation 0 y_intercept) ∧
  (∀ x y : ℝ, line_equation x y → (2 * x + y + 2 = 0 → y = y_intercept)) :=
sorry

end NUMINAMATH_CALUDE_line_equation_correct_l936_93674


namespace NUMINAMATH_CALUDE_triangle_side_length_l936_93683

theorem triangle_side_length (b c : ℝ) (A : ℝ) (S : ℝ) : 
  b = 2 → 
  A = 2 * π / 3 → 
  S = 2 * Real.sqrt 3 → 
  S = 1/2 * b * c * Real.sin A →
  b^2 + c^2 - 2*b*c*Real.cos A = (2 * Real.sqrt 7)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l936_93683


namespace NUMINAMATH_CALUDE_cards_left_after_distribution_l936_93624

/-- Given the initial number of cards, number of cards given to each student,
    and number of students, prove that the number of cards left is 12. -/
theorem cards_left_after_distribution (initial_cards : ℕ) (cards_per_student : ℕ) (num_students : ℕ)
    (h1 : initial_cards = 357)
    (h2 : cards_per_student = 23)
    (h3 : num_students = 15) :
  initial_cards - (cards_per_student * num_students) = 12 := by
  sorry

#check cards_left_after_distribution

end NUMINAMATH_CALUDE_cards_left_after_distribution_l936_93624


namespace NUMINAMATH_CALUDE_even_sum_condition_l936_93667

theorem even_sum_condition (m n : ℤ) : 
  (∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) → (∃ p : ℤ, m + n = 2 * p) ∧
  ¬(∀ q : ℤ, m + n = 2 * q → ∃ r s : ℤ, m = 2 * r ∧ n = 2 * s) :=
by sorry

end NUMINAMATH_CALUDE_even_sum_condition_l936_93667


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l936_93694

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-3 : ℝ) (1/2) = {x : ℝ | c * x^2 + b * x + a < 0}) :
  {x : ℝ | a * x^2 + b * x + c ≥ 0} = Set.Icc (-1/3 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l936_93694


namespace NUMINAMATH_CALUDE_honey_jar_problem_l936_93669

/-- The proportion of honey remaining after each extraction -/
def remaining_proportion : ℚ := 75 / 100

/-- The number of times the extraction process is repeated -/
def num_extractions : ℕ := 6

/-- The amount of honey remaining after all extractions (in grams) -/
def final_honey : ℚ := 420

/-- Calculates the initial amount of honey given the final amount and extraction process -/
def initial_honey : ℚ := final_honey / remaining_proportion ^ num_extractions

theorem honey_jar_problem :
  initial_honey * remaining_proportion ^ num_extractions = final_honey :=
sorry

end NUMINAMATH_CALUDE_honey_jar_problem_l936_93669


namespace NUMINAMATH_CALUDE_angle_D_is_60_l936_93686

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Real)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  q.A + q.B + q.C + q.D = 360

-- Define the specific conditions of our quadrilateral
def special_quadrilateral (q : Quadrilateral) : Prop :=
  q.A + q.B = 180 ∧ q.C = 2 * q.D

-- Theorem statement
theorem angle_D_is_60 (q : Quadrilateral) 
  (h1 : is_valid_quadrilateral q) 
  (h2 : special_quadrilateral q) : 
  q.D = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_is_60_l936_93686


namespace NUMINAMATH_CALUDE_adjacent_diff_at_least_five_l936_93696

/-- Represents a cell in the 8x8 grid -/
structure Cell :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the 8x8 grid filled with integers from 1 to 64 -/
def Grid := Cell → Fin 64

/-- Two cells are adjacent if they share a common edge -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col)

/-- Main theorem: In any 8x8 grid filled with integers from 1 to 64,
    there exist two adjacent cells whose values differ by at least 5 -/
theorem adjacent_diff_at_least_five (g : Grid) : 
  ∃ c1 c2 : Cell, adjacent c1 c2 ∧ (g c1).val + 5 ≤ (g c2).val ∨ (g c2).val + 5 ≤ (g c1).val :=
sorry

end NUMINAMATH_CALUDE_adjacent_diff_at_least_five_l936_93696


namespace NUMINAMATH_CALUDE_watch_cost_price_l936_93682

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (C : ℝ), 
  (C > 0) ∧ 
  (0.64 * C + 140 = 1.04 * C) ∧ 
  (C = 350) := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l936_93682


namespace NUMINAMATH_CALUDE_square_difference_49_16_l936_93613

theorem square_difference_49_16 : 49^2 - 16^2 = 2145 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_49_16_l936_93613


namespace NUMINAMATH_CALUDE_real_part_reciprocal_l936_93684

theorem real_part_reciprocal (z : ℂ) (h : z = 1 - 2*I) : 
  (1 / z).re = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_real_part_reciprocal_l936_93684


namespace NUMINAMATH_CALUDE_donation_sum_l936_93603

theorem donation_sum : 
  let donation1 : ℝ := 245.00
  let donation2 : ℝ := 225.00
  let donation3 : ℝ := 230.00
  donation1 + donation2 + donation3 = 700.00 := by
  sorry

end NUMINAMATH_CALUDE_donation_sum_l936_93603


namespace NUMINAMATH_CALUDE_system_of_equations_proof_l936_93687

theorem system_of_equations_proof (a b c d : ℂ) 
  (eq1 : a - b - c + d = 12)
  (eq2 : a + b - c - d = 6)
  (eq3 : 2*a + c - d = 15) :
  (b - d)^2 = 9 := by sorry

end NUMINAMATH_CALUDE_system_of_equations_proof_l936_93687


namespace NUMINAMATH_CALUDE_quadratic_inequality_l936_93681

theorem quadratic_inequality (x : ℝ) : x^2 - 40*x + 400 ≤ 10 ↔ 20 - Real.sqrt 10 ≤ x ∧ x ≤ 20 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l936_93681


namespace NUMINAMATH_CALUDE_person_height_from_shadows_l936_93695

/-- Given a tree and a person casting shadows under the same lighting conditions,
    calculate the person's height based on the tree's height and shadow lengths. -/
theorem person_height_from_shadows 
  (tree_height : ℝ) (tree_shadow : ℝ) (person_shadow : ℝ) 
  (tree_height_pos : tree_height > 0)
  (tree_shadow_pos : tree_shadow > 0)
  (person_shadow_pos : person_shadow > 0)
  (h_tree : tree_height = 40 ∧ tree_shadow = 10)
  (h_person_shadow : person_shadow = 15 / 12) -- Convert 15 inches to feet
  : (tree_height / tree_shadow) * person_shadow = 5 := by
  sorry

#check person_height_from_shadows

end NUMINAMATH_CALUDE_person_height_from_shadows_l936_93695


namespace NUMINAMATH_CALUDE_john_ate_three_cookies_l936_93660

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens John bought -/
def dozens_bought : ℕ := 2

/-- The number of cookies John has left -/
def cookies_left : ℕ := 21

/-- The number of cookies John ate -/
def cookies_eaten : ℕ := dozens_bought * dozen - cookies_left

theorem john_ate_three_cookies : cookies_eaten = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_ate_three_cookies_l936_93660


namespace NUMINAMATH_CALUDE_subtraction_division_equality_l936_93680

theorem subtraction_division_equality : 5020 - (502 / 100.4) = 5014.998 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_division_equality_l936_93680


namespace NUMINAMATH_CALUDE_tv_price_increase_l936_93616

theorem tv_price_increase (P : ℝ) (x : ℝ) : 
  (1.30 * P) * (1 + x / 100) = 1.82 * P ↔ x = 40 :=
sorry

end NUMINAMATH_CALUDE_tv_price_increase_l936_93616


namespace NUMINAMATH_CALUDE_problem_solution_l936_93615

/-- The graph of y = x + m - 2 does not pass through the second quadrant -/
def p (m : ℝ) : Prop := ∀ x y : ℝ, y = x + m - 2 → ¬(x < 0 ∧ y > 0)

/-- The equation x^2 + y^2 / (1-m) = 1 represents an ellipse with its focus on the x-axis -/
def q (m : ℝ) : Prop := 0 < 1 - m ∧ 1 - m < 1

theorem problem_solution (m : ℝ) :
  (∀ m, q m → p m) ∧ ¬(∀ m, p m → q m) ∧
  (¬(p m ∧ q m) ∧ (p m ∨ q m) ↔ m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l936_93615


namespace NUMINAMATH_CALUDE_number_comparisons_l936_93656

theorem number_comparisons : 
  (97430 < 100076) ∧ 
  (67500000 > 65700000) ∧ 
  (2648050 > 2648005) ∧ 
  (45000000 = 45000000) := by
  sorry

end NUMINAMATH_CALUDE_number_comparisons_l936_93656


namespace NUMINAMATH_CALUDE_max_three_layer_structures_l936_93610

theorem max_three_layer_structures :
  ∃ (a b c : ℕ),
    1 ≤ a ∧ a ≤ b - 2 ∧ b - 2 ≤ c - 4 ∧
    a^2 + b^2 + c^2 ≤ 1988 ∧
    ∀ (x y z : ℕ),
      1 ≤ x ∧ x ≤ y - 2 ∧ y - 2 ≤ z - 4 ∧
      x^2 + y^2 + z^2 ≤ 1988 →
      (b - a - 1)^2 * (c - b - 1)^2 ≥ (y - x - 1)^2 * (z - y - 1)^2 ∧
    (b - a - 1)^2 * (c - b - 1)^2 = 345 :=
by sorry

end NUMINAMATH_CALUDE_max_three_layer_structures_l936_93610


namespace NUMINAMATH_CALUDE_positive_y_solution_l936_93698

theorem positive_y_solution (x y z : ℝ) 
  (eq1 : x * y = 8 - 3 * x - 2 * y)
  (eq2 : y * z = 15 - 5 * y - 3 * z)
  (eq3 : x * z = 40 - 5 * x - 4 * z)
  (y_pos : y > 0) :
  y = 4 := by
sorry

end NUMINAMATH_CALUDE_positive_y_solution_l936_93698


namespace NUMINAMATH_CALUDE_smallest_difference_is_one_l936_93604

/-- Represents a triangle with integer side lengths -/
structure IntegerTriangle where
  de : ℕ
  ef : ℕ
  df : ℕ

/-- Checks if the given side lengths form a valid triangle -/
def is_valid_triangle (t : IntegerTriangle) : Prop :=
  t.de + t.ef > t.df ∧ t.de + t.df > t.ef ∧ t.ef + t.df > t.de

/-- Theorem: The smallest possible difference between EF and DE in the given conditions is 1 -/
theorem smallest_difference_is_one :
  ∃ (t : IntegerTriangle),
    t.de + t.ef + t.df = 3005 ∧
    t.de < t.ef ∧
    t.ef ≤ t.df ∧
    is_valid_triangle t ∧
    (∀ (u : IntegerTriangle),
      u.de + u.ef + u.df = 3005 →
      u.de < u.ef →
      u.ef ≤ u.df →
      is_valid_triangle u →
      u.ef - u.de ≥ t.ef - t.de) ∧
    t.ef - t.de = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_difference_is_one_l936_93604


namespace NUMINAMATH_CALUDE_product_sum_ratio_l936_93638

theorem product_sum_ratio : (1 * 2 * 3 * 4 * 5 * 6) / (1 + 2 + 3 + 4 + 5 + 6) = 240 / 7 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_ratio_l936_93638


namespace NUMINAMATH_CALUDE_sin_135_degrees_l936_93673

theorem sin_135_degrees : Real.sin (135 * π / 180) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l936_93673


namespace NUMINAMATH_CALUDE_starting_player_wins_l936_93635

/-- A game state representing the cards held by each player -/
structure GameState :=
  (player_cards : List Nat)
  (opponent_cards : List Nat)

/-- Check if a list of digits can form a number divisible by 17 -/
def can_form_divisible_by_17 (digits : List Nat) : Bool :=
  sorry

/-- The optimal strategy for the starting player -/
def optimal_strategy (state : GameState) : Option Nat :=
  sorry

/-- Theorem stating that the starting player wins with optimal play -/
theorem starting_player_wins :
  ∀ (initial_cards : List Nat),
    initial_cards.length = 7 ∧
    (∀ n, n ∈ initial_cards → n ≥ 0 ∧ n ≤ 6) →
    ∃ (final_state : GameState),
      final_state.player_cards ⊆ initial_cards ∧
      final_state.opponent_cards ⊆ initial_cards ∧
      final_state.player_cards.length + final_state.opponent_cards.length = 7 ∧
      can_form_divisible_by_17 final_state.player_cards ∧
      ¬can_form_divisible_by_17 final_state.opponent_cards :=
  sorry

end NUMINAMATH_CALUDE_starting_player_wins_l936_93635


namespace NUMINAMATH_CALUDE_families_increase_l936_93666

theorem families_increase (F : ℝ) (h1 : F > 0) : 
  let families_with_computers_1992 := 0.3 * F
  let families_with_computers_1999 := 1.5 * families_with_computers_1992
  let total_families_1999 := families_with_computers_1999 / (3/7)
  total_families_1999 = 1.05 * F :=
by sorry

end NUMINAMATH_CALUDE_families_increase_l936_93666


namespace NUMINAMATH_CALUDE_coefficient_x_squared_proof_l936_93632

/-- The coefficient of x^2 in the expansion of (x - 2/x)^4 * (x - 2) -/
def coefficient_x_squared : ℤ := 16

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_x_squared_proof :
  coefficient_x_squared = 
    (-(binomial 4 1 : ℤ) * 2) * (-2 : ℤ) := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_proof_l936_93632


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l936_93648

theorem sum_of_three_numbers : 2.12 + 0.004 + 0.345 = 2.469 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l936_93648


namespace NUMINAMATH_CALUDE_triangle_properties_l936_93634

open Real

/-- Given a triangle ABC with angle C = 2π/3 and c² = 5a² + ab, prove the following:
    1. sin B / sin A = 2
    2. The maximum value of sin A * sin B is 1/4 -/
theorem triangle_properties (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angle : angle_C = 2 * π / 3)
  (h_side : c^2 = 5 * a^2 + a * b) :
  (sin angle_B / sin angle_A = 2) ∧
  (∀ x y : ℝ, 0 < x ∧ x < π / 3 → sin x * sin y ≤ 1 / 4) :=
by sorry


end NUMINAMATH_CALUDE_triangle_properties_l936_93634


namespace NUMINAMATH_CALUDE_shop_profit_percentage_l936_93622

/-- Calculates the total profit percentage for a shop selling two types of items -/
theorem shop_profit_percentage
  (cost_price_ratio_A : ℝ)
  (cost_price_ratio_B : ℝ)
  (quantity_A : ℕ)
  (quantity_B : ℕ)
  (price_A : ℝ)
  (price_B : ℝ)
  (h1 : cost_price_ratio_A = 0.95)
  (h2 : cost_price_ratio_B = 0.90)
  (h3 : quantity_A = 100)
  (h4 : quantity_B = 150)
  (h5 : price_A = 50)
  (h6 : price_B = 60) :
  let profit_A := quantity_A * price_A * (1 - cost_price_ratio_A)
  let profit_B := quantity_B * price_B * (1 - cost_price_ratio_B)
  let total_profit := profit_A + profit_B
  let total_cost := quantity_A * price_A * cost_price_ratio_A + quantity_B * price_B * cost_price_ratio_B
  let profit_percentage := (total_profit / total_cost) * 100
  ∃ ε > 0, |profit_percentage - 8.95| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_shop_profit_percentage_l936_93622


namespace NUMINAMATH_CALUDE_product_of_3_6_and_0_5_l936_93678

theorem product_of_3_6_and_0_5 : 3.6 * 0.5 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_3_6_and_0_5_l936_93678


namespace NUMINAMATH_CALUDE_farm_equation_correct_l936_93647

/-- Represents the farm problem with chickens and pigs --/
structure FarmProblem where
  total_heads : ℕ
  total_legs : ℕ
  chicken_count : ℕ
  pig_count : ℕ

/-- The equation correctly represents the farm problem --/
theorem farm_equation_correct (farm : FarmProblem)
  (head_sum : farm.chicken_count + farm.pig_count = farm.total_heads)
  (head_count : farm.total_heads = 70)
  (leg_count : farm.total_legs = 196) :
  2 * farm.chicken_count + 4 * (70 - farm.chicken_count) = 196 := by
  sorry

#check farm_equation_correct

end NUMINAMATH_CALUDE_farm_equation_correct_l936_93647


namespace NUMINAMATH_CALUDE_lcm_of_primes_l936_93677

theorem lcm_of_primes : 
  let p₁ : Nat := 1223
  let p₂ : Nat := 1399
  let p₃ : Nat := 2687
  Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ →
  Nat.lcm p₁ (Nat.lcm p₂ p₃) = 4583641741 :=
by sorry

end NUMINAMATH_CALUDE_lcm_of_primes_l936_93677


namespace NUMINAMATH_CALUDE_min_z_value_l936_93605

theorem min_z_value (x y z : ℤ) (sum_eq : x + y + z = 100) (ineq : x < y ∧ y < 2*z) : 
  ∀ w : ℤ, (∃ a b : ℤ, a + b + w = 100 ∧ a < b ∧ b < 2*w) → w ≥ 21 := by
  sorry

#check min_z_value

end NUMINAMATH_CALUDE_min_z_value_l936_93605


namespace NUMINAMATH_CALUDE_product_comparison_l936_93675

theorem product_comparison (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1.1 * a) * (1.13 * b) * (0.8 * c) < a * b * c :=
by sorry

end NUMINAMATH_CALUDE_product_comparison_l936_93675


namespace NUMINAMATH_CALUDE_square_divided_into_triangles_even_count_l936_93631

theorem square_divided_into_triangles_even_count (a : ℕ) (h : a > 0) :
  let triangle_area : ℚ := 3 * 4 / 2
  let square_area : ℚ := a^2
  let num_triangles : ℚ := square_area / triangle_area
  (∃ k : ℕ, num_triangles = k ∧ k % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_square_divided_into_triangles_even_count_l936_93631


namespace NUMINAMATH_CALUDE_number_relationships_l936_93652

theorem number_relationships : 
  (10 * 10000 = 100000) ∧
  (10 * 1000000 = 10000000) ∧
  (10 * 10000000 = 100000000) ∧
  (100000000 / 10000 = 10000) := by
  sorry

end NUMINAMATH_CALUDE_number_relationships_l936_93652


namespace NUMINAMATH_CALUDE_min_value_expression_l936_93618

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  ((a + 2*b)^2 + (b - 2*c)^2 + (c - 2*a)^2) / b^2 ≥ 25/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l936_93618


namespace NUMINAMATH_CALUDE_fraction_equality_l936_93662

theorem fraction_equality : (900^2 : ℝ) / (264^2 - 256^2) = 194.711 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l936_93662


namespace NUMINAMATH_CALUDE_min_value_theorem_l936_93636

theorem min_value_theorem (x y : ℝ) (h : x - 2*y - 4 = 0) :
  ∃ (min : ℝ), min = 8 ∧ ∀ z, z = 2^x + 1/(4^y) → z ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l936_93636


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l936_93646

def total_group_size : ℕ := 15
def men_count : ℕ := 9
def women_count : ℕ := 6
def selection_size : ℕ := 4

theorem probability_at_least_one_woman :
  let total_combinations := Nat.choose total_group_size selection_size
  let all_men_combinations := Nat.choose men_count selection_size
  (total_combinations - all_men_combinations : ℚ) / total_combinations = 137 / 151 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l936_93646


namespace NUMINAMATH_CALUDE_number_difference_l936_93619

theorem number_difference (a b : ℕ) (h1 : a + b = 22904) (h2 : b % 5 = 0) (h3 : b = 7 * a) : 
  b - a = 17178 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l936_93619


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l936_93691

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 12 * x^2 - 58 * x + 70
  ∃ x : ℝ, f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_equation_l936_93691


namespace NUMINAMATH_CALUDE_planes_formed_by_three_lines_through_point_l936_93621

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in three-dimensional space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents the number of planes formed by three lines -/
inductive NumPlanes
  | one
  | three

/-- Given a point and three lines through it, determines the number of planes formed -/
def planesFormedByThreeLines (p : Point3D) (l1 l2 l3 : Line3D) : NumPlanes :=
  sorry

theorem planes_formed_by_three_lines_through_point 
  (p : Point3D) (l1 l2 l3 : Line3D) 
  (h1 : l1.point = p) (h2 : l2.point = p) (h3 : l3.point = p) :
  planesFormedByThreeLines p l1 l2 l3 = NumPlanes.one ∨ 
  planesFormedByThreeLines p l1 l2 l3 = NumPlanes.three :=
sorry

end NUMINAMATH_CALUDE_planes_formed_by_three_lines_through_point_l936_93621


namespace NUMINAMATH_CALUDE_qualified_products_l936_93688

theorem qualified_products (defect_rate : ℝ) (total_items : ℕ) : 
  defect_rate = 0.005 →
  total_items = 18000 →
  ⌊(1 - defect_rate) * total_items⌋ = 17910 := by
sorry

end NUMINAMATH_CALUDE_qualified_products_l936_93688


namespace NUMINAMATH_CALUDE_kayla_apples_l936_93642

theorem kayla_apples (total : ℕ) (kylie : ℕ) (kayla : ℕ) : 
  total = 340 →
  kayla = 4 * kylie + 10 →
  total = kylie + kayla →
  kayla = 274 := by
sorry

end NUMINAMATH_CALUDE_kayla_apples_l936_93642


namespace NUMINAMATH_CALUDE_ana_number_puzzle_l936_93661

theorem ana_number_puzzle (x : ℝ) : ((x + 3) * 3 - 4) / 2 = 10 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ana_number_puzzle_l936_93661


namespace NUMINAMATH_CALUDE_angle_reflection_l936_93630

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 180 + 360 * (k : Real) < α ∧ α < 270 + 360 * (k : Real)

def is_in_fourth_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 270 + 360 * (k : Real) < α ∧ α < 360 + 360 * (k : Real)

theorem angle_reflection (α : Real) :
  is_in_third_quadrant α → is_in_fourth_quadrant (180 - α) := by
  sorry

end NUMINAMATH_CALUDE_angle_reflection_l936_93630


namespace NUMINAMATH_CALUDE_min_value_theorem_l936_93629

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (min : ℝ), min = 9 ∧ (1/x + 4/y ≥ min) ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ 1/x₀ + 4/y₀ = min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l936_93629


namespace NUMINAMATH_CALUDE_sequence_c_increasing_l936_93607

theorem sequence_c_increasing (n : ℕ) : 
  let a : ℕ → ℤ := λ n => 2 * n^2 - 5 * n + 1
  a (n + 1) > a n :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_c_increasing_l936_93607


namespace NUMINAMATH_CALUDE_max_a_value_l936_93668

def is_lattice_point (x y : ℤ) : Prop := True

def line_passes_through_lattice_point (m : ℚ) : Prop :=
  ∃ x y : ℤ, 0 < x ∧ x ≤ 50 ∧ is_lattice_point x y ∧ y = m * x + 5

theorem max_a_value :
  ∀ a : ℚ, (∀ m : ℚ, 2/3 < m → m < a → ¬line_passes_through_lattice_point m) →
    a ≤ 35/51 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l936_93668


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l936_93654

theorem pure_imaginary_solutions (x : ℂ) :
  (x^4 - 5*x^3 + 10*x^2 - 50*x - 75 = 0) ∧ (∃ k : ℝ, x = k * I) ↔
  (x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l936_93654


namespace NUMINAMATH_CALUDE_cricket_team_throwers_l936_93664

/-- Represents a cricket team with throwers and non-throwers -/
structure CricketTeam where
  total_players : ℕ
  throwers : ℕ
  right_handed : ℕ
  left_handed : ℕ

/-- Conditions for the cricket team problem -/
def valid_cricket_team (team : CricketTeam) : Prop :=
  team.total_players = 58 ∧
  team.throwers + team.right_handed + team.left_handed = team.total_players ∧
  team.throwers + team.right_handed = 51 ∧
  team.left_handed = (team.total_players - team.throwers) / 3

theorem cricket_team_throwers :
  ∀ team : CricketTeam, valid_cricket_team team → team.throwers = 37 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_throwers_l936_93664


namespace NUMINAMATH_CALUDE_surface_area_of_sawed_cube_l936_93679

/-- The total surface area of rectangular blocks obtained by sawing a unit cube -/
def total_surface_area (length_cuts width_cuts height_cuts : ℕ) : ℝ :=
  let original_surface := 6
  let new_surface := (length_cuts + 1) * (width_cuts + 1) * 2 +
                     (length_cuts + 1) * (height_cuts + 1) * 2 +
                     (width_cuts + 1) * (height_cuts + 1) * 2
  original_surface + new_surface - 6

/-- Theorem: The total surface area of 24 rectangular blocks obtained by sawing a unit cube
    1 time along length, 2 times along width, and 3 times along height is 18 square meters -/
theorem surface_area_of_sawed_cube : total_surface_area 1 2 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_sawed_cube_l936_93679


namespace NUMINAMATH_CALUDE_stock_price_increase_l936_93663

theorem stock_price_increase (initial_price : ℝ) (first_year_increase : ℝ) : 
  initial_price > 0 →
  first_year_increase > 0 →
  initial_price * (1 + first_year_increase / 100) * 0.75 * 1.2 = initial_price * 1.08 →
  first_year_increase = 20 := by
sorry

end NUMINAMATH_CALUDE_stock_price_increase_l936_93663


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l936_93639

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^4 + 16 * X^3 + 5 * X^2 - 36 * X + 58 = 
  (X^2 + 5 * X + 3) * q + (-28 * X + 55) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l936_93639


namespace NUMINAMATH_CALUDE_quadratic_factorization_l936_93655

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b)) →
  a - b = -11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l936_93655


namespace NUMINAMATH_CALUDE_circle_centers_line_l936_93608

/-- The set of circles C_k defined by (x-k+1)^2 + (y-3k)^2 = 2k^4 where k is a positive integer -/
def CircleSet (k : ℕ+) (x y : ℝ) : Prop :=
  (x - k + 1)^2 + (y - 3*k)^2 = 2 * k^4

/-- The center of circle C_k -/
def CircleCenter (k : ℕ+) : ℝ × ℝ := (k - 1, 3*k)

/-- The line on which the centers lie -/
def CenterLine (x y : ℝ) : Prop := y = 3*(x + 1) ∧ x ≠ -1

/-- Theorem: If the centers of the circles C_k lie on a fixed line,
    then that line is y = 3(x+1) where x ≠ -1 -/
theorem circle_centers_line :
  (∀ k : ℕ+, ∃ x y : ℝ, CircleCenter k = (x, y) ∧ CenterLine x y) →
  ∀ x y : ℝ, (∃ k : ℕ+, CircleCenter k = (x, y)) → CenterLine x y :=
sorry

end NUMINAMATH_CALUDE_circle_centers_line_l936_93608


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l936_93600

-- Problem 1
theorem problem_1 : (-1)^2023 + 2 * Real.cos (π / 4) - |Real.sqrt 2 - 2| - (1 / 2)⁻¹ = 2 * Real.sqrt 2 - 5 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx : x ≠ 0) : 
  (1 - 1 / (x + 1)) / ((x^2) / (x^2 + 2*x + 1)) = (x + 1) / x := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l936_93600


namespace NUMINAMATH_CALUDE_no_extreme_points_l936_93640

/-- The function f(x) = x^3 - 3x^2 + 3x has no extreme points. -/
theorem no_extreme_points (x : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^3 - 3*x^2 + 3*x
  (∀ a b, a < b → f a < f b) :=
by
  sorry

end NUMINAMATH_CALUDE_no_extreme_points_l936_93640


namespace NUMINAMATH_CALUDE_cosine_sum_lower_bound_l936_93644

theorem cosine_sum_lower_bound (a b c : ℝ) :
  Real.cos (a - b) + Real.cos (b - c) + Real.cos (c - a) ≥ -3/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_lower_bound_l936_93644


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l936_93637

/-- A line defined by y = (m-2)x + m, where 0 < m < 2, does not pass through the third quadrant. -/
theorem line_not_in_third_quadrant (m : ℝ) (h : 0 < m ∧ m < 2) :
  ∃ (x y : ℝ), y = (m - 2) * x + m → ¬(x < 0 ∧ y < 0) :=
by sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l936_93637


namespace NUMINAMATH_CALUDE_units_digit_of_sum_is_seven_l936_93672

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_less_than_10 : hundreds < 10
  tens_less_than_10 : tens < 10
  units_less_than_10 : units < 10
  hundreds_not_zero : hundreds ≠ 0

/-- The condition that the hundreds digit is 3 less than twice the units digit -/
def hundreds_units_relation (n : ThreeDigitNumber) : Prop :=
  n.hundreds = 2 * n.units - 3

/-- The value of the three-digit number -/
def number_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed number -/
def reversed_number (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

/-- The theorem to be proved -/
theorem units_digit_of_sum_is_seven (n : ThreeDigitNumber) 
  (h : hundreds_units_relation n) : 
  (number_value n + reversed_number n) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_is_seven_l936_93672


namespace NUMINAMATH_CALUDE_inverse_negation_implies_contrapositive_l936_93626

-- Define propositions as boolean variables
variable (p q r : Prop)

-- Define the inverse relation
def is_inverse (a b : Prop) : Prop :=
  (a ↔ b) ∧ (¬a ↔ ¬b)

-- Define the negation relation
def is_negation (a b : Prop) : Prop :=
  a ↔ ¬b

-- Define the contrapositive relation
def is_contrapositive (a b : Prop) : Prop :=
  (a ↔ ¬b) ∧ (b ↔ ¬a)

-- State the theorem
theorem inverse_negation_implies_contrapositive
  (h1 : is_inverse p q)
  (h2 : is_negation q r) :
  is_contrapositive p r := by
sorry

end NUMINAMATH_CALUDE_inverse_negation_implies_contrapositive_l936_93626


namespace NUMINAMATH_CALUDE_open_box_volume_proof_l936_93611

/-- Calculates the volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
def open_box_volume (sheet_length sheet_width cut_size : ℝ) : ℝ :=
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size

/-- Proves that the volume of the open box formed from a 48 m x 36 m sheet with 6 m x 6 m corner cuts is 5184 m³. -/
theorem open_box_volume_proof :
  open_box_volume 48 36 6 = 5184 := by
  sorry

#eval open_box_volume 48 36 6

end NUMINAMATH_CALUDE_open_box_volume_proof_l936_93611


namespace NUMINAMATH_CALUDE_second_to_first_ratio_l936_93692

/-- Represents the guesses of four students for the number of jellybeans in a jar. -/
structure JellybeanGuesses where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Defines the conditions for the jellybean guessing problem. -/
def valid_guesses (g : JellybeanGuesses) : Prop :=
  g.first = 100 ∧
  g.third = g.second - 200 ∧
  g.fourth = (g.first + g.second + g.third) / 3 + 25 ∧
  g.fourth = 525

/-- Theorem stating that for valid guesses, the ratio of the second to the first guess is 8:1. -/
theorem second_to_first_ratio (g : JellybeanGuesses) (h : valid_guesses g) :
  g.second / g.first = 8 := by
  sorry

#check second_to_first_ratio

end NUMINAMATH_CALUDE_second_to_first_ratio_l936_93692


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_9_l936_93643

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def five_digit_number (d : ℕ) : ℕ := 56780 + d

theorem five_digit_multiple_of_9 (d : ℕ) : 
  d < 10 → (is_multiple_of_9 (five_digit_number d) ↔ d = 1) := by
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_9_l936_93643


namespace NUMINAMATH_CALUDE_tilde_result_bounds_l936_93650

def tilde (a b : ℚ) : ℚ := |a - b|

def consecutive_integers (n : ℕ) : List ℚ := List.range n

def perform_tilde (l : List ℚ) : ℚ :=
  l.foldl tilde (l.head!)

def max_tilde_result (n : ℕ) : ℚ :=
  if n % 4 == 1 then n - 1 else n

def min_tilde_result (n : ℕ) : ℚ :=
  if n % 4 == 2 || n % 4 == 3 then 1 else 0

theorem tilde_result_bounds (n : ℕ) (l : List ℚ) :
  l.length = n ∧ l.toFinset = (consecutive_integers n).toFinset →
  perform_tilde l ≤ max_tilde_result n ∧
  perform_tilde l ≥ min_tilde_result n :=
sorry

end NUMINAMATH_CALUDE_tilde_result_bounds_l936_93650


namespace NUMINAMATH_CALUDE_weed_spread_incomplete_weeds_cannot_fill_grid_l936_93620

/-- Represents a grid with weeds -/
structure WeedGrid :=
  (size : Nat)
  (initial_weeds : Nat)

/-- Calculates the maximum possible boundary length of a grid -/
def max_boundary (g : WeedGrid) : Nat :=
  4 * g.size

/-- Calculates the maximum initial boundary length of weed-filled cells -/
def initial_boundary (g : WeedGrid) : Nat :=
  4 * g.initial_weeds

/-- The weed spread theorem -/
theorem weed_spread_incomplete (g : WeedGrid) 
  (h_size : g.size = 10) 
  (h_initial : g.initial_weeds = 9) :
  initial_boundary g < max_boundary g := by
  sorry

/-- The main theorem: weeds cannot spread to all cells -/
theorem weeds_cannot_fill_grid (g : WeedGrid) 
  (h_size : g.size = 10) 
  (h_initial : g.initial_weeds = 9) :
  ¬ (∃ (final_weeds : Nat), final_weeds = g.size * g.size) := by
  sorry

end NUMINAMATH_CALUDE_weed_spread_incomplete_weeds_cannot_fill_grid_l936_93620


namespace NUMINAMATH_CALUDE_min_value_condition_l936_93641

def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 3

theorem min_value_condition (b : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f b x ≥ 1) ∧ 
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, f b x = 1) ↔ 
  b = Real.sqrt 2 ∨ b = -3/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_condition_l936_93641


namespace NUMINAMATH_CALUDE_ali_flower_sales_l936_93628

/-- Calculates the total number of flowers sold by Ali -/
def total_flowers_sold (monday : ℕ) (tuesday : ℕ) : ℕ :=
  monday + tuesday + 2 * monday

theorem ali_flower_sales : total_flowers_sold 4 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ali_flower_sales_l936_93628


namespace NUMINAMATH_CALUDE_ahmed_has_13_goats_l936_93649

def adam_goats : ℕ := 7

def andrew_goats : ℕ := 2 * adam_goats + 5

def ahmed_goats : ℕ := andrew_goats - 6

theorem ahmed_has_13_goats : ahmed_goats = 13 := by
  sorry

end NUMINAMATH_CALUDE_ahmed_has_13_goats_l936_93649


namespace NUMINAMATH_CALUDE_susan_bob_cat_difference_l936_93602

/-- Proves that Susan has 8 more cats than Bob after all exchanges -/
theorem susan_bob_cat_difference :
  let susan_initial : ℕ := 21
  let bob_initial : ℕ := 3
  let susan_received : ℕ := 5
  let bob_received : ℕ := 7
  let susan_gave : ℕ := 4
  let susan_final := susan_initial + susan_received - susan_gave
  let bob_final := bob_initial + bob_received + susan_gave
  susan_final - bob_final = 8 := by
  sorry

end NUMINAMATH_CALUDE_susan_bob_cat_difference_l936_93602


namespace NUMINAMATH_CALUDE_equation_solution_l936_93601

theorem equation_solution : ∃ x : ℝ, (6 + 1.5 * x = 2.5 * x - 30 + Real.sqrt 100) ∧ x = 26 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l936_93601


namespace NUMINAMATH_CALUDE_video_game_lives_l936_93633

/-- Given an initial number of players, additional players, and total lives,
    calculate the number of lives per player. -/
def lives_per_player (initial_players : ℕ) (additional_players : ℕ) (total_lives : ℕ) : ℕ :=
  total_lives / (initial_players + additional_players)

/-- Theorem: In the video game scenario, each player has 6 lives. -/
theorem video_game_lives : lives_per_player 2 2 24 = 6 := by
  sorry

#eval lives_per_player 2 2 24

end NUMINAMATH_CALUDE_video_game_lives_l936_93633


namespace NUMINAMATH_CALUDE_may_salary_is_6500_l936_93693

/-- Calculates the salary for May given the average salaries and January's salary -/
def salary_may (avg_jan_to_apr avg_feb_to_may jan_salary : ℚ) : ℚ :=
  4 * avg_feb_to_may - (4 * avg_jan_to_apr - jan_salary)

/-- Proves that the salary for May is 6500 given the conditions -/
theorem may_salary_is_6500 :
  let avg_jan_to_apr : ℚ := 8000
  let avg_feb_to_may : ℚ := 8200
  let jan_salary : ℚ := 5700
  salary_may avg_jan_to_apr avg_feb_to_may jan_salary = 6500 :=
by
  sorry

#eval salary_may 8000 8200 5700

end NUMINAMATH_CALUDE_may_salary_is_6500_l936_93693


namespace NUMINAMATH_CALUDE_probability_problem_l936_93699

theorem probability_problem (p_biology : ℚ) (p_no_chemistry : ℚ)
  (h1 : p_biology = 5/8)
  (h2 : p_no_chemistry = 1/2) :
  let p_no_biology := 1 - p_biology
  let p_neither := p_no_biology * p_no_chemistry
  (p_no_biology = 3/8) ∧ (p_neither = 3/16) := by
  sorry

end NUMINAMATH_CALUDE_probability_problem_l936_93699


namespace NUMINAMATH_CALUDE_arctan_identity_l936_93676

theorem arctan_identity (x : Real) : 
  Real.arctan (Real.tan (70 * π / 180) - 2 * Real.tan (35 * π / 180)) = 20 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_identity_l936_93676


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l936_93609

theorem cheryl_material_usage
  (material1 : ℚ) (material2 : ℚ) (leftover : ℚ)
  (h1 : material1 = 4 / 9)
  (h2 : material2 = 2 / 3)
  (h3 : leftover = 8 / 18) :
  material1 + material2 - leftover = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l936_93609


namespace NUMINAMATH_CALUDE_binary_remainder_by_eight_l936_93612

/-- The remainder when 110111100101₂ is divided by 8 is 5 -/
theorem binary_remainder_by_eight : Nat.mod 0b110111100101 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_remainder_by_eight_l936_93612
