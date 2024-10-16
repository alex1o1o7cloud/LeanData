import Mathlib

namespace NUMINAMATH_CALUDE_circles_intersect_l3185_318599

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop := x^2 - 12*x + y^2 - 8*y - 12 = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop := x^2 + 10*x + y^2 - 10*y + 34 = 0

/-- The shortest distance between the two circles -/
def shortest_distance : ℝ := 0

/-- Theorem stating that the shortest distance between the two circles is 0 -/
theorem circles_intersect : 
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y ∧ shortest_distance = 0 :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l3185_318599


namespace NUMINAMATH_CALUDE_invertible_function_fixed_point_l3185_318555

/-- Given an invertible function f: ℝ → ℝ, if f(a) = 3 and f(3) = a, then a - 3 = 0 -/
theorem invertible_function_fixed_point 
  (f : ℝ → ℝ) (hf : Function.Bijective f) (a : ℝ) 
  (h1 : f a = 3) (h2 : f 3 = a) : a - 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_invertible_function_fixed_point_l3185_318555


namespace NUMINAMATH_CALUDE_quartic_root_product_l3185_318581

theorem quartic_root_product (k : ℝ) : 
  (∃ a b c d : ℝ, 
    (a^4 - 18*a^3 + k*a^2 + 200*a - 1984 = 0) ∧
    (b^4 - 18*b^3 + k*b^2 + 200*b - 1984 = 0) ∧
    (c^4 - 18*c^3 + k*c^2 + 200*c - 1984 = 0) ∧
    (d^4 - 18*d^3 + k*d^2 + 200*d - 1984 = 0) ∧
    (a * b = -32 ∨ a * c = -32 ∨ a * d = -32 ∨ b * c = -32 ∨ b * d = -32 ∨ c * d = -32)) →
  k = 86 := by
sorry

end NUMINAMATH_CALUDE_quartic_root_product_l3185_318581


namespace NUMINAMATH_CALUDE_locus_of_rectangle_vertex_l3185_318579

/-- Given a circle centered at the origin with radius r and a point M(a,b) inside the circle,
    prove that the locus of point T for all rectangles MKTP where K and P lie on the circle
    is a circle centered at the origin with radius √(2r² - (a² + b²)). -/
theorem locus_of_rectangle_vertex (r a b : ℝ) (hr : r > 0) (hab : a^2 + b^2 < r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = 2 * r^2 - (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_rectangle_vertex_l3185_318579


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l3185_318549

def a : ℕ := 999
def b : ℕ := 666

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_9ab : sum_of_digits (9 * a * b) = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l3185_318549


namespace NUMINAMATH_CALUDE_quadratic_equation_negative_root_l3185_318516

theorem quadratic_equation_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ x^2 - 6*a*x - 2 + 2*a + 9*a^2 = 0) ↔ a < (-1 + Real.sqrt 19) / 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_negative_root_l3185_318516


namespace NUMINAMATH_CALUDE_counterfeiters_payment_range_l3185_318525

/-- Represents a counterfeiter who can pay amounts between 1 and 25 rubles --/
structure Counterfeiter where
  pay : ℕ → ℕ
  pay_range : ∀ n, 1 ≤ pay n ∧ pay n ≤ 25

/-- The theorem states that three counterfeiters can collectively pay any amount from 100 to 200 rubles --/
theorem counterfeiters_payment_range (c1 c2 c3 : Counterfeiter) :
  ∀ n, 100 ≤ n ∧ n ≤ 200 → ∃ (x y z : ℕ), x + y + z = n ∧ 
    (∃ (a b c : ℕ), c1.pay a + c2.pay b + c3.pay c = x) ∧
    (∃ (d e f : ℕ), c1.pay d + c2.pay e + c3.pay f = y) ∧
    (∃ (g h i : ℕ), c1.pay g + c2.pay h + c3.pay i = z) :=
  sorry

end NUMINAMATH_CALUDE_counterfeiters_payment_range_l3185_318525


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_values_l3185_318582

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = 3 ∧ p.1 ≠ 2}
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + 2 * p.2 + a = 0}

-- State the theorem
theorem intersection_empty_implies_a_values :
  ∀ a : ℝ, (M ∩ N a = ∅) → (a = -6 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_values_l3185_318582


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3185_318533

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : parallel a α) 
  (h4 : perpendicular b β) 
  (h5 : plane_parallel α β) : 
  line_perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3185_318533


namespace NUMINAMATH_CALUDE_emily_chairs_l3185_318522

/-- The number of chairs Emily bought -/
def num_chairs : ℕ := sorry

/-- The number of tables Emily bought -/
def num_tables : ℕ := 2

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_furniture : ℕ := 8

/-- The total time spent (in minutes) -/
def total_time : ℕ := 48

theorem emily_chairs : 
  num_chairs = 4 ∧ 
  time_per_furniture * (num_chairs + num_tables) = total_time :=
sorry

end NUMINAMATH_CALUDE_emily_chairs_l3185_318522


namespace NUMINAMATH_CALUDE_inequality_proof_l3185_318540

theorem inequality_proof (a b c : ℝ) : 
  a = 0.1 * Real.exp 0.1 → 
  b = 1 / 9 → 
  c = -Real.log 0.9 → 
  c < a ∧ a < b := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3185_318540


namespace NUMINAMATH_CALUDE_inequality_proof_l3185_318500

theorem inequality_proof (a b c : ℝ) : 
  a = (3/4) * Real.exp (2/5) →
  b = 2/5 →
  c = (2/5) * Real.exp (3/4) →
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3185_318500


namespace NUMINAMATH_CALUDE_larger_number_problem_l3185_318558

theorem larger_number_problem (x y : ℝ) : 4 * y = 5 * x → x + y = 54 → y = 30 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3185_318558


namespace NUMINAMATH_CALUDE_complex_fraction_problem_l3185_318504

theorem complex_fraction_problem (x y : ℂ) 
  (h : (x - y) / (x + y) - (x + y) / (x - y) = 2) :
  ∃ (result : ℂ), (x^6 + y^6) / (x^6 - y^6) - (x^6 - y^6) / (x^6 + y^6) = result :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_problem_l3185_318504


namespace NUMINAMATH_CALUDE_student_D_most_stable_l3185_318590

-- Define the set of students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define a function to get the variance for each student
def variance : Student → ℝ
| Student.A => 2.1
| Student.B => 3.5
| Student.C => 9.0
| Student.D => 0.7

-- Define a predicate for most stable performance
def most_stable (s : Student) : Prop :=
  ∀ t : Student, variance s ≤ variance t

-- Theorem: Student D has the most stable performance
theorem student_D_most_stable : most_stable Student.D := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_student_D_most_stable_l3185_318590


namespace NUMINAMATH_CALUDE_triangle_square_apothem_equality_l3185_318539

/-- Theorem: Value of k for a specific right triangle and square configuration -/
theorem triangle_square_apothem_equality (x : ℝ) (k : ℝ) : 
  x > 0 →  -- Ensure positive side lengths
  (3*x)^2 + (4*x)^2 = (5*x)^2 →  -- Pythagorean theorem for right triangle
  12*x = k * (6*x^2) →  -- Perimeter = k * Area for triangle
  4*x = 5 →  -- Apothem equality
  100 = 3 * 40 →  -- Square area = 3 * Square perimeter
  k = 8/5 := by sorry

end NUMINAMATH_CALUDE_triangle_square_apothem_equality_l3185_318539


namespace NUMINAMATH_CALUDE_range_of_ln_b_over_a_l3185_318502

open Real

theorem range_of_ln_b_over_a (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 1 / ℯ ≤ c / a) (h5 : c / a ≤ 2) (h6 : c * log b = a + c * log c) :
  ∃ x : ℝ, x = log (b / a) ∧ 1 ≤ x ∧ x ≤ ℯ - 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_ln_b_over_a_l3185_318502


namespace NUMINAMATH_CALUDE_new_game_cost_new_game_cost_is_8_l3185_318544

def initial_money : ℕ := 57
def toy_cost : ℕ := 4
def num_toys : ℕ := 2

theorem new_game_cost : ℕ :=
  initial_money - (toy_cost * num_toys)

#check new_game_cost

theorem new_game_cost_is_8 : new_game_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_new_game_cost_new_game_cost_is_8_l3185_318544


namespace NUMINAMATH_CALUDE_unique_square_with_special_property_l3185_318535

/-- Checks if a number uses exactly 5 different non-zero digits in base 6 --/
def hasFiveDifferentNonZeroDigitsBase6 (n : ℕ) : Prop := sorry

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : List ℕ := sorry

/-- Moves the last digit of a number to the front --/
def moveLastToFront (n : ℕ) : ℕ := sorry

/-- Reverses the digits of a number --/
def reverseDigits (n : ℕ) : ℕ := sorry

theorem unique_square_with_special_property :
  ∃! n : ℕ,
    n ^ 2 ≤ 54321 ∧
    n ^ 2 ≥ 12345 ∧
    hasFiveDifferentNonZeroDigitsBase6 (n ^ 2) ∧
    (∃ m : ℕ, m ^ 2 = moveLastToFront (n ^ 2) ∧
              m = reverseDigits n) ∧
    n = 221 := by sorry

end NUMINAMATH_CALUDE_unique_square_with_special_property_l3185_318535


namespace NUMINAMATH_CALUDE_five_twelve_thirteen_right_triangle_l3185_318561

/-- A triple of positive integers representing the sides of a triangle -/
structure TripleSides where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Checks if a triple of sides satisfies the Pythagorean theorem -/
def is_right_triangle (sides : TripleSides) : Prop :=
  (sides.a.val ^ 2 : ℕ) + (sides.b.val ^ 2 : ℕ) = (sides.c.val ^ 2 : ℕ)

/-- The triple (5, 12, 13) forms a right triangle -/
theorem five_twelve_thirteen_right_triangle :
  is_right_triangle ⟨5, 12, 13⟩ := by sorry

end NUMINAMATH_CALUDE_five_twelve_thirteen_right_triangle_l3185_318561


namespace NUMINAMATH_CALUDE_sum_of_squares_first_10_base6_l3185_318514

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-6 --/
def base10ToBase6 (n : ℕ) : ℕ := sorry

/-- Computes the sum of squares of the first n base-6 numbers --/
def sumOfSquaresBase6 (n : ℕ) : ℕ := sorry

theorem sum_of_squares_first_10_base6 :
  base10ToBase6 (sumOfSquaresBase6 10) = 231 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_first_10_base6_l3185_318514


namespace NUMINAMATH_CALUDE_number_of_boxes_l3185_318536

def total_oranges : ℕ := 45
def oranges_per_box : ℕ := 5

theorem number_of_boxes : total_oranges / oranges_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boxes_l3185_318536


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3185_318517

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 2) : x^3 + y^3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3185_318517


namespace NUMINAMATH_CALUDE_work_problem_solution_l3185_318577

def work_problem (a_rate b_rate : ℝ) (combined_days : ℝ) : Prop :=
  a_rate = 2 * b_rate →
  combined_days = 6 →
  b_rate * (a_rate + b_rate)⁻¹ * combined_days = 18

theorem work_problem_solution :
  ∀ (a_rate b_rate combined_days : ℝ),
    work_problem a_rate b_rate combined_days :=
by
  sorry

end NUMINAMATH_CALUDE_work_problem_solution_l3185_318577


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3185_318566

/-- The trajectory of the midpoint between a moving point on the unit circle and the fixed point (3, 0) -/
theorem midpoint_trajectory :
  ∀ (a b x y : ℝ),
  a^2 + b^2 = 1 →  -- point (a, b) is on the unit circle
  x = (a + 3) / 2 →  -- x-coordinate of midpoint
  y = b / 2 →  -- y-coordinate of midpoint
  x^2 + y^2 - 3*x + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l3185_318566


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3185_318501

-- Define the set P
def P : Set ℝ := {1, 2, 3, 4}

-- Define the set Q
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem stating that "x ∈ P" is a sufficient but not necessary condition for "x ∈ Q"
theorem p_sufficient_not_necessary_for_q :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3185_318501


namespace NUMINAMATH_CALUDE_nickel_count_l3185_318591

/-- Given a purchase of 150 cents paid with 50 coins consisting of only pennies and nickels,
    prove that the number of nickels used is 25. -/
theorem nickel_count (p n : ℕ) : 
  p + n = 50 →  -- Total number of coins
  p + 5 * n = 150 →  -- Total value in cents
  n = 25 := by sorry

end NUMINAMATH_CALUDE_nickel_count_l3185_318591


namespace NUMINAMATH_CALUDE_mans_speed_with_stream_l3185_318565

/-- Given a man's rowing rate in still water and his speed against the stream,
    prove that his speed with the stream is equal to twice his rate in still water
    minus his speed against the stream. -/
theorem mans_speed_with_stream
  (rate_still_water : ℝ)
  (speed_against_stream : ℝ)
  (h1 : rate_still_water = 7)
  (h2 : speed_against_stream = 4) :
  rate_still_water + (rate_still_water - speed_against_stream) = 2 * rate_still_water - speed_against_stream :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_with_stream_l3185_318565


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l3185_318524

-- Define the set of cards
inductive Card : Type
| Black : Card
| Red : Card
| White : Card

-- Define the set of individuals
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the event "Individual A gets the red card"
def EventA (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Individual B gets the red card"
def EventB (d : Distribution) : Prop := d Person.B = Card.Red

-- Theorem statement
theorem events_mutually_exclusive_but_not_opposite :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventA d ∧ EventB d)) ∧
  -- The events are not opposite
  (∃ d : Distribution, ¬EventA d ∧ ¬EventB d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l3185_318524


namespace NUMINAMATH_CALUDE_star_properties_l3185_318572

noncomputable def star (x y : ℝ) : ℝ := Real.log (10^x + 10^y) / Real.log 10

theorem star_properties :
  (∀ a b : ℝ, star a b = star b a) ∧
  (∀ a b c : ℝ, star (star a b) c = star a (star b c)) ∧
  (∀ a b c : ℝ, star a b + c = star (a + c) (b + c)) ∧
  (∃ a b c : ℝ, star a b * c ≠ star (a * c) (b * c)) :=
by sorry

end NUMINAMATH_CALUDE_star_properties_l3185_318572


namespace NUMINAMATH_CALUDE_traci_flour_l3185_318560

/-- The amount of flour Harris has in his house -/
def harris_flour : ℕ := 400

/-- The amount of flour needed for each cake -/
def flour_per_cake : ℕ := 100

/-- The number of cakes Traci and Harris created each -/
def cakes_per_person : ℕ := 9

/-- The total number of cakes created -/
def total_cakes : ℕ := 2 * cakes_per_person

/-- The theorem stating the amount of flour Traci brought from her own house -/
theorem traci_flour : 
  harris_flour + (total_cakes * flour_per_cake - harris_flour) = 1400 := by
sorry

end NUMINAMATH_CALUDE_traci_flour_l3185_318560


namespace NUMINAMATH_CALUDE_reflection_of_C_l3185_318559

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point C -/
def C : ℝ × ℝ := (3, 1)

theorem reflection_of_C :
  (reflect_x ∘ reflect_y) C = (-3, -1) := by sorry

end NUMINAMATH_CALUDE_reflection_of_C_l3185_318559


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l3185_318584

theorem tic_tac_toe_tie_probability (amy_win : ℚ) (lily_win : ℚ) (john_win : ℚ)
  (h_amy : amy_win = 4/9)
  (h_lily : lily_win = 1/3)
  (h_john : john_win = 1/6) :
  1 - (amy_win + lily_win + john_win) = 1/18 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l3185_318584


namespace NUMINAMATH_CALUDE_monica_second_third_classes_l3185_318596

/-- Represents the number of students in Monica's classes -/
structure MonicasClasses where
  total_classes : Nat
  first_class : Nat
  fourth_class : Nat
  fifth_sixth_classes : Nat
  total_students : Nat

/-- The number of students in Monica's second and third classes combined -/
def students_in_second_third_classes (m : MonicasClasses) : Nat :=
  m.total_students - (m.first_class + m.fourth_class + m.fifth_sixth_classes)

/-- Theorem stating the number of students in Monica's second and third classes -/
theorem monica_second_third_classes :
  ∀ (m : MonicasClasses),
  m.total_classes = 6 →
  m.first_class = 20 →
  m.fourth_class = m.first_class / 2 →
  m.fifth_sixth_classes = 28 * 2 →
  m.total_students = 136 →
  students_in_second_third_classes m = 50 := by
  sorry

end NUMINAMATH_CALUDE_monica_second_third_classes_l3185_318596


namespace NUMINAMATH_CALUDE_polynomial_equality_l3185_318573

theorem polynomial_equality (x : ℝ) (h : ℝ → ℝ) :
  (8 * x^4 - 4 * x^2 + 2 + h x = 2 * x^3 - 6 * x + 4) →
  (h x = -8 * x^4 + 2 * x^3 + 4 * x^2 - 6 * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3185_318573


namespace NUMINAMATH_CALUDE_lcm_gcd_relation_l3185_318578

theorem lcm_gcd_relation (m n : ℕ) (h1 : m > n) (h2 : m > 0) (h3 : n > 0) 
  (h4 : Nat.lcm m n = 30 * Nat.gcd m n) 
  (h5 : (m - n) ∣ Nat.lcm m n) : 
  (m + n) / Nat.gcd m n = 11 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_relation_l3185_318578


namespace NUMINAMATH_CALUDE_shortest_ribbon_length_l3185_318552

theorem shortest_ribbon_length (ribbon_length : ℕ) : 
  (ribbon_length % 2 = 0 ∧ ribbon_length % 5 = 0) → 
  ribbon_length ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_shortest_ribbon_length_l3185_318552


namespace NUMINAMATH_CALUDE_cycling_average_speed_l3185_318551

theorem cycling_average_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (rest_duration : ℝ) 
  (num_rests : ℕ) 
  (h1 : total_distance = 56) 
  (h2 : total_time = 8) 
  (h3 : rest_duration = 0.5) 
  (h4 : num_rests = 2) : 
  total_distance / (total_time - num_rests * rest_duration) = 8 := by
sorry

end NUMINAMATH_CALUDE_cycling_average_speed_l3185_318551


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3185_318575

/-- The coefficient of x^2 in the original function -/
def α : ℝ := 3

/-- The coefficient of x in the original function -/
def β : ℝ := -2

/-- The constant term in the original function -/
def γ : ℝ := 4

/-- The horizontal shift of the graph (to the left) -/
def h : ℝ := 2

/-- The vertical shift of the graph (upwards) -/
def k : ℝ := 5

/-- The coefficient of x^2 in the transformed function -/
def a : ℝ := α

/-- The coefficient of x in the transformed function -/
def b : ℝ := 2 * α * h - β

/-- The constant term in the transformed function -/
def c : ℝ := α * h^2 - β * h + γ + k

/-- Theorem stating that the sum of coefficients in the transformed function equals 30 -/
theorem sum_of_coefficients : a + b + c = 30 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3185_318575


namespace NUMINAMATH_CALUDE_rectangle_formations_l3185_318513

theorem rectangle_formations (h : ℕ) (v : ℕ) (h_val : h = 5) (v_val : v = 4) :
  (Nat.choose h 2) * (Nat.choose v 2) = 60 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_formations_l3185_318513


namespace NUMINAMATH_CALUDE_total_pies_count_l3185_318571

/-- The number of miniature pumpkin pies made by Pinky -/
def pinky_pies : ℕ := 147

/-- The number of miniature pumpkin pies made by Helen -/
def helen_pies : ℕ := 56

/-- The total number of miniature pumpkin pies -/
def total_pies : ℕ := pinky_pies + helen_pies

theorem total_pies_count : total_pies = 203 := by
  sorry

end NUMINAMATH_CALUDE_total_pies_count_l3185_318571


namespace NUMINAMATH_CALUDE_cubic_root_increasing_l3185_318508

theorem cubic_root_increasing : 
  ∀ (x y : ℝ), x < y → (x ^ (1/3 : ℝ)) < (y ^ (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_increasing_l3185_318508


namespace NUMINAMATH_CALUDE_prime_divisor_of_mersenne_number_l3185_318595

theorem prime_divisor_of_mersenne_number (p q : ℕ) : 
  Prime p → Prime q → q ∣ (2^p - 1) → p ∣ (q - 1) := by sorry

end NUMINAMATH_CALUDE_prime_divisor_of_mersenne_number_l3185_318595


namespace NUMINAMATH_CALUDE_evaluate_expression_l3185_318585

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/2) (hz : z = 8) :
  x^3 * y^4 * z = 1/128 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3185_318585


namespace NUMINAMATH_CALUDE_combined_value_of_a_and_b_l3185_318570

-- Define the conversion rate from paise to rupees
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

-- Define the value of a in rupees
def a : ℚ := (paise_to_rupees 95) / 0.005

-- Define the value of b in rupees
def b : ℚ := 3 * a - 50

-- Theorem statement
theorem combined_value_of_a_and_b : a + b = 710 := by sorry

end NUMINAMATH_CALUDE_combined_value_of_a_and_b_l3185_318570


namespace NUMINAMATH_CALUDE_adjacent_sum_6_l3185_318576

/-- Represents a 3x3 table filled with numbers from 1 to 9 --/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if a table is valid according to the given conditions --/
def is_valid_table (t : Table) : Prop :=
  (∀ i j, t i j ≠ 0) ∧  -- All cells are filled
  (∀ x, ∃! i j, t i j = x) ∧  -- Each number appears exactly once
  t 0 0 = 1 ∧ t 2 0 = 2 ∧ t 0 2 = 3 ∧ t 2 2 = 4 ∧  -- Given positions
  (∃ i j, t i j = 5 ∧ 
    (t (i-1) j + t (i+1) j + t i (j-1) + t i (j+1) : ℕ) = 9)  -- Sum around 5 is 9

/-- Sum of adjacent numbers to a given position --/
def adjacent_sum (t : Table) (i j : Fin 3) : ℕ :=
  (t (i-1) j + t (i+1) j + t i (j-1) + t i (j+1) : ℕ)

/-- The main theorem --/
theorem adjacent_sum_6 (t : Table) (h : is_valid_table t) :
  ∃ i j, t i j = 6 ∧ adjacent_sum t i j = 29 :=
sorry

end NUMINAMATH_CALUDE_adjacent_sum_6_l3185_318576


namespace NUMINAMATH_CALUDE_min_digits_removal_l3185_318589

def original_number : ℕ := 20162016

def is_valid_removal (n : ℕ) : Prop :=
  ∃ (removed : ℕ),
    removed > 0 ∧
    removed < original_number ∧
    (original_number - removed) % 2016 = 0 ∧
    (String.length (toString removed) + String.length (toString (original_number - removed)) = 8)

theorem min_digits_removal :
  (∀ n : ℕ, n < 3 → ¬(is_valid_removal n)) ∧
  (∃ n : ℕ, n = 3 ∧ is_valid_removal n) :=
sorry

end NUMINAMATH_CALUDE_min_digits_removal_l3185_318589


namespace NUMINAMATH_CALUDE_min_value_sqrt_expression_l3185_318598

theorem min_value_sqrt_expression (x : ℝ) :
  Real.sqrt (x^2 - Real.sqrt 3 * |x| + 1) + Real.sqrt (x^2 + Real.sqrt 3 * |x| + 3) ≥ Real.sqrt 7 ∧
  (Real.sqrt (x^2 - Real.sqrt 3 * |x| + 1) + Real.sqrt (x^2 + Real.sqrt 3 * |x| + 3) = Real.sqrt 7 ↔ x = Real.sqrt 3 / 4 ∨ x = -Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_expression_l3185_318598


namespace NUMINAMATH_CALUDE_starting_team_combinations_count_l3185_318538

def team_size : ℕ := 18
def starting_team_size : ℕ := 8
def other_players_size : ℕ := starting_team_size - 2  -- 6 players excluding goalie and captain

def number_of_starting_team_combinations : ℕ :=
  team_size *  -- ways to choose goalie
  (team_size - 1) *  -- ways to choose captain (excluding goalie)
  (Nat.choose (team_size - 2) other_players_size)  -- ways to choose remaining 6 players

theorem starting_team_combinations_count :
  number_of_starting_team_combinations = 2455344 :=
by sorry

end NUMINAMATH_CALUDE_starting_team_combinations_count_l3185_318538


namespace NUMINAMATH_CALUDE_sqrt_7200_minus_61_cube_l3185_318557

theorem sqrt_7200_minus_61_cube (a b : ℕ+) :
  (Real.sqrt 7200 - 61 : ℝ) = (Real.sqrt a.val - b.val)^3 →
  a.val + b.val = 21 := by
sorry

end NUMINAMATH_CALUDE_sqrt_7200_minus_61_cube_l3185_318557


namespace NUMINAMATH_CALUDE_ones_divisibility_l3185_318503

theorem ones_divisibility (p : ℕ) (hp : Prime p) :
  (p ≠ 3 → ¬(∃ k : ℤ, (10^p - 1) / 9 = k * p)) ∧
  (p > 5 → ∃ k : ℤ, (10^(p-1) - 1) / 9 = k * p) := by
  sorry

end NUMINAMATH_CALUDE_ones_divisibility_l3185_318503


namespace NUMINAMATH_CALUDE_no_perfect_cubes_l3185_318532

theorem no_perfect_cubes (a b : ℤ) : ¬(∃ x y : ℤ, a^5*b + 3 = x^3 ∧ a*b^5 + 3 = y^3) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_cubes_l3185_318532


namespace NUMINAMATH_CALUDE_citrus_grove_orchards_l3185_318526

theorem citrus_grove_orchards (total : ℕ) (lemons : ℕ) (oranges : ℕ) (limes : ℕ) (grapefruits : ℕ) :
  total = 16 →
  lemons = 8 →
  oranges = lemons / 2 →
  limes + grapefruits = total - lemons - oranges →
  limes = grapefruits →
  grapefruits = 2 := by
sorry

end NUMINAMATH_CALUDE_citrus_grove_orchards_l3185_318526


namespace NUMINAMATH_CALUDE_row_1007_sum_equals_2013_squared_l3185_318543

/-- The sum of numbers in the nth row of the given pattern -/
def row_sum (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The theorem stating that the 1007th row sum equals 2013² -/
theorem row_1007_sum_equals_2013_squared :
  row_sum 1007 = 2013 ^ 2 := by sorry

end NUMINAMATH_CALUDE_row_1007_sum_equals_2013_squared_l3185_318543


namespace NUMINAMATH_CALUDE_car_speed_problem_l3185_318597

/-- Proves that car R's speed is 30 mph given the conditions of the problem -/
theorem car_speed_problem (distance : ℝ) (time_diff : ℝ) (speed_diff : ℝ)
  (h1 : distance = 300)
  (h2 : time_diff = 2)
  (h3 : speed_diff = 10)
  (h4 : distance / (car_r_speed + speed_diff) + time_diff = distance / car_r_speed)
  : car_r_speed = 30 :=
by
  sorry

#check car_speed_problem

end NUMINAMATH_CALUDE_car_speed_problem_l3185_318597


namespace NUMINAMATH_CALUDE_rectangle_area_l3185_318569

theorem rectangle_area (ratio_long : ℕ) (ratio_short : ℕ) (perimeter : ℕ) :
  ratio_long = 4 →
  ratio_short = 3 →
  perimeter = 126 →
  ∃ (length width : ℕ),
    length * ratio_short = width * ratio_long ∧
    2 * (length + width) = perimeter ∧
    length * width = 972 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3185_318569


namespace NUMINAMATH_CALUDE_problem_statement_l3185_318505

def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | (x - 1) * (x - a) ≤ 0}

theorem problem_statement :
  (∀ a : ℝ, B a ⊆ A → a ∈ Set.Icc 1 2) ∧
  (∀ a : ℝ, A ∩ B a = {1} → a ∈ Set.Iic 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3185_318505


namespace NUMINAMATH_CALUDE_stamp_arrangement_count_l3185_318563

/-- Represents a stamp with its value in cents -/
structure Stamp where
  value : Nat
  deriving Repr

/-- Represents an arrangement of stamps -/
def Arrangement := List Stamp

/-- Checks if an arrangement is valid (sums to 15 cents) -/
def isValidArrangement (arr : Arrangement) : Bool :=
  (arr.map (·.value)).sum = 15

/-- Checks if two arrangements are considered the same -/
def isSameArrangement (arr1 arr2 : Arrangement) : Bool :=
  sorry  -- Implementation details omitted

/-- Generates all possible stamp arrangements -/
def generateArrangements (stamps : List (Nat × Nat)) : List Arrangement :=
  sorry  -- Implementation details omitted

/-- Counts unique arrangements -/
def countUniqueArrangements (arrangements : List Arrangement) : Nat :=
  sorry  -- Implementation details omitted

/-- The main theorem to prove -/
theorem stamp_arrangement_count :
  let stamps := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
  let arrangements := generateArrangements stamps
  let validArrangements := arrangements.filter isValidArrangement
  countUniqueArrangements validArrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_stamp_arrangement_count_l3185_318563


namespace NUMINAMATH_CALUDE_c_share_is_40_l3185_318509

/-- Represents the share distribution among three parties -/
structure ShareDistribution where
  total : ℝ
  b_share : ℝ
  c_share : ℝ
  d_share : ℝ

/-- The condition for the share distribution -/
def valid_distribution (s : ShareDistribution) : Prop :=
  s.total = 80 ∧
  s.c_share = 1.5 * s.b_share ∧
  s.d_share = 0.5 * s.b_share ∧
  s.total = s.b_share + s.c_share + s.d_share

/-- Theorem stating that under the given conditions, c's share is 40 rupees -/
theorem c_share_is_40 (s : ShareDistribution) (h : valid_distribution s) : s.c_share = 40 := by
  sorry

end NUMINAMATH_CALUDE_c_share_is_40_l3185_318509


namespace NUMINAMATH_CALUDE_intersection_implies_k_geq_two_l3185_318594

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x : ℝ | x - k ≤ 0}

-- State the theorem
theorem intersection_implies_k_geq_two (k : ℝ) : M ∩ N k = M → k ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_k_geq_two_l3185_318594


namespace NUMINAMATH_CALUDE_julies_savings_l3185_318564

-- Define the initial savings amount
variable (S : ℝ)

-- Define the interest rate
variable (r : ℝ)

-- Define the time period
def t : ℝ := 2

-- Define the simple interest earned
def simple_interest : ℝ := 120

-- Define the compound interest earned
def compound_interest : ℝ := 126

-- Theorem statement
theorem julies_savings :
  (simple_interest = (S / 2) * r * t) ∧
  (compound_interest = (S / 2) * ((1 + r)^t - 1)) →
  S = 1200 := by
sorry

end NUMINAMATH_CALUDE_julies_savings_l3185_318564


namespace NUMINAMATH_CALUDE_f_eval_at_one_l3185_318534

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 2*x + 15
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + 2*x^3 + b*x^2 + 150*x + c

-- State the theorem
theorem f_eval_at_one (a b c : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    g a x = 0 ∧ g a y = 0 ∧ g a z = 0 ∧
    f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0) →
  f b c 1 = -15640 :=
by sorry

end NUMINAMATH_CALUDE_f_eval_at_one_l3185_318534


namespace NUMINAMATH_CALUDE_students_walking_home_fraction_l3185_318562

theorem students_walking_home_fraction :
  let bus_fraction : ℚ := 1/3
  let auto_fraction : ℚ := 1/6
  let bike_fraction : ℚ := 1/15
  let total_fraction : ℚ := 1
  let other_transport_fraction : ℚ := bus_fraction + auto_fraction + bike_fraction
  let walking_fraction : ℚ := total_fraction - other_transport_fraction
  walking_fraction = 13/30 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_fraction_l3185_318562


namespace NUMINAMATH_CALUDE_wall_width_l3185_318547

/-- Given a rectangular wall with specific proportions and volume, prove its width is 4 meters. -/
theorem wall_width (w h l : ℝ) (h_height : h = 6 * w) (h_length : l = 7 * h) 
  (h_volume : w * h * l = 16128) : w = 4 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_l3185_318547


namespace NUMINAMATH_CALUDE_regular_polygon_with_740_diagonals_has_40_sides_l3185_318527

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 740 diagonals has 40 sides -/
theorem regular_polygon_with_740_diagonals_has_40_sides :
  ∃ (n : ℕ), n > 3 ∧ num_diagonals n = 740 ∧ n = 40 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_740_diagonals_has_40_sides_l3185_318527


namespace NUMINAMATH_CALUDE_binomial_distribution_not_equivalent_to_expansion_l3185_318580

-- Define the binomial distribution formula
def binomial_distribution (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- Define the general term of binomial expansion
def binomial_expansion_term (n k : ℕ) (a b : ℝ) : ℝ :=
  (n.choose k) * a^k * b^(n-k)

-- Theorem statement
theorem binomial_distribution_not_equivalent_to_expansion :
  ∃ n k : ℕ, ∃ p : ℝ, 
    binomial_distribution n k p ≠ binomial_expansion_term n k p (1-p) :=
sorry

end NUMINAMATH_CALUDE_binomial_distribution_not_equivalent_to_expansion_l3185_318580


namespace NUMINAMATH_CALUDE_merry_sunday_boxes_l3185_318511

/-- Represents the number of apples in each box -/
def apples_per_box : ℕ := 10

/-- Represents the number of boxes Merry had on Saturday -/
def saturday_boxes : ℕ := 50

/-- Represents the total number of apples sold on Saturday and Sunday -/
def total_apples_sold : ℕ := 720

/-- Represents the number of boxes left after selling -/
def boxes_left : ℕ := 3

/-- Represents the number of boxes Merry had on Sunday -/
def sunday_boxes : ℕ := 25

theorem merry_sunday_boxes :
  sunday_boxes = 25 :=
by sorry

end NUMINAMATH_CALUDE_merry_sunday_boxes_l3185_318511


namespace NUMINAMATH_CALUDE_cube_root_27_fourth_root_81_sixth_root_64_eq_18_l3185_318531

theorem cube_root_27_fourth_root_81_sixth_root_64_eq_18 :
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/6) = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_fourth_root_81_sixth_root_64_eq_18_l3185_318531


namespace NUMINAMATH_CALUDE_projection_matrix_condition_l3185_318523

/-- A 2x2 matrix is a projection matrix if and only if its square equals itself. -/
def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific matrix we're working with -/
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 7/17; c, 10/17]

/-- The main theorem: P is a projection matrix if and only if a = 9/17 and c = 10/17 -/
theorem projection_matrix_condition (a c : ℚ) :
  is_projection_matrix (P a c) ↔ a = 9/17 ∧ c = 10/17 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_condition_l3185_318523


namespace NUMINAMATH_CALUDE_coefficient_properties_l3185_318537

/-- A polynomial with roots at 0, 1, -1, 2, and -2 -/
def Q (a b c d e : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- The theorem stating the properties of the coefficients -/
theorem coefficient_properties (a b c d e : ℝ) :
  (∀ x : ℝ, x = 0 ∨ x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2 → Q a b c d e x = 0) →
  a = 0 ∧ b = 0 ∧ e = 0 ∧ c ≠ 0 ∧ d ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_properties_l3185_318537


namespace NUMINAMATH_CALUDE_exists_coprime_sequence_l3185_318553

theorem exists_coprime_sequence : ∃ (a : ℕ → ℕ), 
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ i j p q r, i ≠ j ∧ i ≠ p ∧ i ≠ q ∧ i ≠ r ∧ j ≠ p ∧ j ≠ q ∧ j ≠ r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r → 
    Nat.gcd (a i + a j) (a p + a q + a r) = 1) :=
by sorry

end NUMINAMATH_CALUDE_exists_coprime_sequence_l3185_318553


namespace NUMINAMATH_CALUDE_sum_21_implies_n_6_l3185_318567

/-- Represents a sequence where a₁ = 1 and aₙ₊₁ = aₙ + 1 -/
def ArithmeticSequence (n : ℕ) : ℕ :=
  n

/-- Sum of the first n terms of the arithmetic sequence -/
def Sn (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Theorem: If Sn = 21, then n = 6 -/
theorem sum_21_implies_n_6 : Sn 6 = 21 :=
  by sorry

end NUMINAMATH_CALUDE_sum_21_implies_n_6_l3185_318567


namespace NUMINAMATH_CALUDE_sequence_sum_l3185_318542

theorem sequence_sum : ∀ (a b c d : ℕ), 
  (b - a = c - b) →  -- arithmetic progression
  (c * c = b * d) →  -- geometric progression
  (d = a + 50) →     -- difference between first and fourth terms
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) →  -- positive integers
  a + b + c + d = 215 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_l3185_318542


namespace NUMINAMATH_CALUDE_exponent_equation_solution_l3185_318512

theorem exponent_equation_solution :
  ∃ x : ℤ, (5 : ℝ)^7 * (5 : ℝ)^x = 125 ∧ x = -4 :=
by sorry

end NUMINAMATH_CALUDE_exponent_equation_solution_l3185_318512


namespace NUMINAMATH_CALUDE_rogers_bike_ride_ratio_l3185_318518

/-- Given that Roger rode his bike for 2 miles in the morning and a total of 12 miles,
    prove that the ratio of evening miles to morning miles is 5:1. -/
theorem rogers_bike_ride_ratio :
  let morning_miles : ℝ := 2
  let total_miles : ℝ := 12
  let evening_miles : ℝ := total_miles - morning_miles
  evening_miles / morning_miles = 5 := by
sorry

end NUMINAMATH_CALUDE_rogers_bike_ride_ratio_l3185_318518


namespace NUMINAMATH_CALUDE_find_FC_l3185_318548

/-- Given a triangle ABC with point D on AC and point E on AD, prove the length of FC. -/
theorem find_FC (DC CB : ℝ) (h1 : DC = 10) (h2 : CB = 12)
  (AB AD ED : ℝ) (h3 : AB = 1/3 * AD) (h4 : ED = 2/3 * AD) : 
  ∃ (FC : ℝ), FC = 506/33 := by
  sorry

end NUMINAMATH_CALUDE_find_FC_l3185_318548


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3185_318545

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (1/x + 1/y) ≥ 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3185_318545


namespace NUMINAMATH_CALUDE_imaginary_part_of_1_plus_2i_l3185_318583

theorem imaginary_part_of_1_plus_2i :
  Complex.im (1 + 2*I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_1_plus_2i_l3185_318583


namespace NUMINAMATH_CALUDE_quiche_theorem_l3185_318506

/-- Quiche ingredients and their properties --/
structure QuicheIngredients where
  spinach_initial : ℝ
  mushrooms_initial : ℝ
  onions_initial : ℝ
  spinach_reduction : ℝ
  mushrooms_reduction : ℝ
  onions_reduction : ℝ
  cream_cheese_volume : ℝ
  cream_cheese_calories : ℝ
  eggs_volume : ℝ
  eggs_calories : ℝ
  oz_to_cup_conversion : ℝ

/-- Calculate the total volume and calorie content of the quiche --/
def quiche_properties (ingredients : QuicheIngredients) : ℝ × ℝ :=
  let cooked_spinach := ingredients.spinach_initial * ingredients.spinach_reduction
  let cooked_mushrooms := ingredients.mushrooms_initial * ingredients.mushrooms_reduction
  let cooked_onions := ingredients.onions_initial * ingredients.onions_reduction
  let total_volume_oz := cooked_spinach + cooked_mushrooms + cooked_onions + 
                         ingredients.cream_cheese_volume + ingredients.eggs_volume
  let total_volume_cups := total_volume_oz * ingredients.oz_to_cup_conversion
  let total_calories := ingredients.cream_cheese_volume * ingredients.cream_cheese_calories + 
                        ingredients.eggs_volume * ingredients.eggs_calories
  (total_volume_cups, total_calories)

/-- Theorem stating the properties of the quiche --/
theorem quiche_theorem (ingredients : QuicheIngredients) 
  (h1 : ingredients.spinach_initial = 40)
  (h2 : ingredients.mushrooms_initial = 25)
  (h3 : ingredients.onions_initial = 15)
  (h4 : ingredients.spinach_reduction = 0.2)
  (h5 : ingredients.mushrooms_reduction = 0.65)
  (h6 : ingredients.onions_reduction = 0.5)
  (h7 : ingredients.cream_cheese_volume = 6)
  (h8 : ingredients.cream_cheese_calories = 80)
  (h9 : ingredients.eggs_volume = 4)
  (h10 : ingredients.eggs_calories = 70)
  (h11 : ingredients.oz_to_cup_conversion = 0.125) :
  quiche_properties ingredients = (5.21875, 760) := by
  sorry

#eval quiche_properties {
  spinach_initial := 40,
  mushrooms_initial := 25,
  onions_initial := 15,
  spinach_reduction := 0.2,
  mushrooms_reduction := 0.65,
  onions_reduction := 0.5,
  cream_cheese_volume := 6,
  cream_cheese_calories := 80,
  eggs_volume := 4,
  eggs_calories := 70,
  oz_to_cup_conversion := 0.125
}

end NUMINAMATH_CALUDE_quiche_theorem_l3185_318506


namespace NUMINAMATH_CALUDE_simplify_fraction_l3185_318586

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  (x^2 + 1) / (x - 1) - 2*x / (x - 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3185_318586


namespace NUMINAMATH_CALUDE_kayla_total_is_15_l3185_318568

def theresa_chocolate : ℕ := 12
def theresa_soda : ℕ := 18

def kayla_chocolate : ℕ := theresa_chocolate / 2
def kayla_soda : ℕ := theresa_soda / 2

def kayla_total : ℕ := kayla_chocolate + kayla_soda

theorem kayla_total_is_15 : kayla_total = 15 := by
  sorry

end NUMINAMATH_CALUDE_kayla_total_is_15_l3185_318568


namespace NUMINAMATH_CALUDE_percentage_problem_l3185_318554

theorem percentage_problem : ∃ p : ℚ, p = 55/100 ∧ p * 40 = 4/5 * 25 + 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3185_318554


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l3185_318520

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ y : ℤ, x^4 + x^3 + x^2 + x + 1 = y^2) ↔ x = -1 ∨ x = 0 ∨ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l3185_318520


namespace NUMINAMATH_CALUDE_B_power_6_l3185_318574

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, -3; 4, 5]

theorem B_power_6 : 
  B^6 = 1715 • B - 16184 • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_6_l3185_318574


namespace NUMINAMATH_CALUDE_multiples_of_12_between_15_and_250_l3185_318507

theorem multiples_of_12_between_15_and_250 : 
  (Finset.filter (λ x => x > 15 ∧ x < 250 ∧ x % 12 = 0) (Finset.range 251)).card = 19 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_12_between_15_and_250_l3185_318507


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3185_318546

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3185_318546


namespace NUMINAMATH_CALUDE_quadrupled_container_volume_l3185_318510

/-- A container with an initial volume and a scale factor for its dimensions. -/
structure Container :=
  (initial_volume : ℝ)
  (scale_factor : ℝ)

/-- The new volume of a container after scaling its dimensions. -/
def new_volume (c : Container) : ℝ :=
  c.initial_volume * c.scale_factor^3

/-- Theorem stating that a container with 5 gallons initial volume and dimensions quadrupled results in 320 gallons. -/
theorem quadrupled_container_volume :
  let c := Container.mk 5 4
  new_volume c = 320 := by
  sorry

end NUMINAMATH_CALUDE_quadrupled_container_volume_l3185_318510


namespace NUMINAMATH_CALUDE_negation_equivalence_l3185_318515

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3185_318515


namespace NUMINAMATH_CALUDE_max_increase_two_letters_l3185_318530

/-- Represents the sets of letters for each position in the license plate --/
structure LetterSets :=
  (first : Finset Char)
  (second : Finset Char)
  (third : Finset Char)

/-- Calculates the total number of possible license plates --/
def totalPlates (sets : LetterSets) : ℕ :=
  sets.first.card * sets.second.card * sets.third.card

/-- The initial configuration of letter sets --/
def initialSets : LetterSets :=
  { first := {'C', 'H', 'L', 'P', 'R'},
    second := {'A', 'I', 'O'},
    third := {'D', 'M', 'N', 'T'} }

/-- Theorem stating the maximum increase in license plates after adding two letters --/
theorem max_increase_two_letters :
  ∃ (newSets : LetterSets), 
    (newSets.first.card + newSets.second.card + newSets.third.card = 
     initialSets.first.card + initialSets.second.card + initialSets.third.card + 2) ∧
    (totalPlates newSets - totalPlates initialSets = 40) ∧
    ∀ (otherSets : LetterSets), 
      (otherSets.first.card + otherSets.second.card + otherSets.third.card = 
       initialSets.first.card + initialSets.second.card + initialSets.third.card + 2) →
      (totalPlates otherSets - totalPlates initialSets ≤ 40) :=
by sorry


end NUMINAMATH_CALUDE_max_increase_two_letters_l3185_318530


namespace NUMINAMATH_CALUDE_max_min_value_sqrt_three_l3185_318521

theorem max_min_value_sqrt_three : 
  ∃ (M : ℝ), M > 0 ∧ 
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 
    min (min (min (1/a) (1/(b^2))) (1/(c^3))) (a + b^2 + c^3) ≤ M) ∧
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    min (min (min (1/a) (1/(b^2))) (1/(c^3))) (a + b^2 + c^3) = M) ∧
  M = Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_max_min_value_sqrt_three_l3185_318521


namespace NUMINAMATH_CALUDE_lanas_boxes_l3185_318541

/-- Given that each box contains 7 pieces of clothing and the total number of pieces is 21,
    prove that the number of boxes is 3. -/
theorem lanas_boxes (pieces_per_box : ℕ) (total_pieces : ℕ) (h1 : pieces_per_box = 7) (h2 : total_pieces = 21) :
  total_pieces / pieces_per_box = 3 := by
  sorry

end NUMINAMATH_CALUDE_lanas_boxes_l3185_318541


namespace NUMINAMATH_CALUDE_mask_count_l3185_318519

theorem mask_count (num_boxes : ℕ) (capacity : ℕ) (lacking : ℕ) (total_masks : ℕ) : 
  num_boxes = 18 → 
  capacity = 15 → 
  lacking = 3 → 
  total_masks = num_boxes * (capacity - lacking) → 
  total_masks = 216 := by
sorry

end NUMINAMATH_CALUDE_mask_count_l3185_318519


namespace NUMINAMATH_CALUDE_special_sequence_sum_l3185_318529

/-- A sequence with specific initial conditions -/
def special_sequence : ℕ → ℚ := sorry

/-- The sum of the first n terms of the special sequence -/
def sum_n (n : ℕ) : ℚ := sorry

theorem special_sequence_sum :
  (special_sequence 1 = 2) →
  (sum_n 2 = 8) →
  (sum_n 3 = 20) →
  ∀ n : ℕ, sum_n n = n * (n + 1) * (2 * n + 4) / 3 := by sorry

end NUMINAMATH_CALUDE_special_sequence_sum_l3185_318529


namespace NUMINAMATH_CALUDE_beef_weight_calculation_l3185_318528

theorem beef_weight_calculation (weight_after : ℝ) (percent_lost : ℝ) 
  (h1 : weight_after = 640)
  (h2 : percent_lost = 20) : 
  weight_after / (1 - percent_lost / 100) = 800 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_calculation_l3185_318528


namespace NUMINAMATH_CALUDE_system_solution_l3185_318550

theorem system_solution :
  ∃ a b c d e : ℤ,
    (ab + a + 2*b = 78 ∧
     bc + 3*b + c = 101 ∧
     cd + 5*c + 3*d = 232 ∧
     de + 4*d + 5*e = 360 ∧
     ea + 2*e + 4*a = 192) →
    ((a = 8 ∧ b = 7 ∧ c = 10 ∧ d = 14 ∧ e = 16) ∨
     (a = -12 ∧ b = -9 ∧ c = -16 ∧ d = -24 ∧ e = -24)) :=
by sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l3185_318550


namespace NUMINAMATH_CALUDE_oil_in_peanut_butter_l3185_318593

/-- Given a ratio of oil to peanuts and the total weight of peanut butter,
    calculate the amount of oil used. -/
def oil_amount (oil_ratio : ℚ) (peanut_ratio : ℚ) (total_weight : ℚ) : ℚ :=
  (oil_ratio / (oil_ratio + peanut_ratio)) * total_weight

/-- Theorem stating that for the given ratios and total weight,
    the amount of oil used is 4 ounces. -/
theorem oil_in_peanut_butter :
  oil_amount 2 8 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oil_in_peanut_butter_l3185_318593


namespace NUMINAMATH_CALUDE_row_sum_1008_equals_2015_squared_l3185_318592

/-- Represents the sum of numbers in a row of the given pattern. -/
def row_sum (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The theorem stating that the sum of numbers in the 1008th row equals 2015². -/
theorem row_sum_1008_equals_2015_squared : row_sum 1008 = 2015 ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_row_sum_1008_equals_2015_squared_l3185_318592


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l3185_318556

/-- Given two rectangles of equal area, where one rectangle has dimensions 8 inches by 45 inches,
    and the other has a length of 15 inches, prove that the width of the second rectangle is 24 inches. -/
theorem equal_area_rectangles_width (area jordan_length jordan_width carol_length : ℝ)
    (h1 : area = jordan_length * jordan_width)
    (h2 : area = carol_length * (area / carol_length))
    (h3 : jordan_length = 8)
    (h4 : jordan_width = 45)
    (h5 : carol_length = 15) :
    area / carol_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l3185_318556


namespace NUMINAMATH_CALUDE_round_trip_speed_l3185_318587

/-- Proves that given a round trip where the return journey takes twice as long as the outward journey,
    and the average speed of the entire trip is 32 miles per hour, the speed of the outward journey is 21⅓ miles per hour. -/
theorem round_trip_speed (d : ℝ) (v : ℝ) (h1 : v > 0) (h2 : d > 0) : 
  (2 * d) / (d / v + 2 * d / v) = 32 → v = 64 / 3 := by
  sorry

#eval (64 : ℚ) / 3  -- To show that 64/3 is indeed equal to 21⅓

end NUMINAMATH_CALUDE_round_trip_speed_l3185_318587


namespace NUMINAMATH_CALUDE_distance_between_trees_l3185_318588

/-- Given a yard of length 441 meters with 22 equally spaced trees (including one at each end),
    the distance between two consecutive trees is 21 meters. -/
theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (distance : ℕ) :
  yard_length = 441 →
  num_trees = 22 →
  distance * (num_trees - 1) = yard_length →
  distance = 21 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_trees_l3185_318588
