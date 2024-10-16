import Mathlib

namespace NUMINAMATH_CALUDE_a_plus_b_equals_34_l3946_394685

theorem a_plus_b_equals_34 (A B : ℝ) :
  (∀ x : ℝ, x ≠ 3 → A / (x - 3) + B * (x + 2) = (-5 * x^2 + 18 * x + 30) / (x - 3)) →
  A + B = 34 := by
sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_34_l3946_394685


namespace NUMINAMATH_CALUDE_quadratic_positivity_set_l3946_394699

/-- Given a quadratic function with zeros at -2 and 3, prove its positivity set -/
theorem quadratic_positivity_set 
  (y : ℝ → ℝ) 
  (h1 : ∀ x, y x = x^2 + b*x + c) 
  (h2 : y (-2) = 0) 
  (h3 : y 3 = 0) :
  {x : ℝ | y x > 0} = {x | x < -2 ∨ x > 3} :=
sorry

end NUMINAMATH_CALUDE_quadratic_positivity_set_l3946_394699


namespace NUMINAMATH_CALUDE_binomial_20_10_l3946_394661

theorem binomial_20_10 (h1 : Nat.choose 18 8 = 43758) 
                       (h2 : Nat.choose 18 9 = 48620) 
                       (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 184756 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_10_l3946_394661


namespace NUMINAMATH_CALUDE_pencil_count_l3946_394693

theorem pencil_count (initial_pencils additional_pencils : ℕ) 
  (h1 : initial_pencils = 27)
  (h2 : additional_pencils = 45) :
  initial_pencils + additional_pencils = 72 :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_l3946_394693


namespace NUMINAMATH_CALUDE_circle_radius_theorem_l3946_394655

theorem circle_radius_theorem (r : ℝ) (h : r > 0) :
  3 * (2 * Real.pi * r) = Real.pi * r^2 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_theorem_l3946_394655


namespace NUMINAMATH_CALUDE_olivias_wallet_problem_l3946_394666

/-- The initial amount of money in Olivia's wallet -/
def initial_money : ℕ := 100

/-- The amount of money Olivia collected from the ATM -/
def atm_money : ℕ := 148

/-- The amount of money Olivia spent at the supermarket -/
def spent_money : ℕ := 89

/-- The amount of money left after visiting the supermarket -/
def remaining_money : ℕ := 159

theorem olivias_wallet_problem :
  initial_money + atm_money = remaining_money + spent_money :=
by sorry

end NUMINAMATH_CALUDE_olivias_wallet_problem_l3946_394666


namespace NUMINAMATH_CALUDE_fourth_root_of_207360000_l3946_394614

theorem fourth_root_of_207360000 : (207360000 : ℝ) ^ (1/4 : ℝ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_207360000_l3946_394614


namespace NUMINAMATH_CALUDE_hexagon_side_sum_l3946_394630

/-- A polygon with six vertices -/
structure Hexagon :=
  (P Q R S T U : ℝ × ℝ)

/-- The area of a polygon -/
def area (h : Hexagon) : ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: For a hexagon PQRSTU with area 40, PQ = 6, QR = 7, and TU = 4, ST + TU = 7 -/
theorem hexagon_side_sum (h : Hexagon) 
  (h_area : area h = 40)
  (h_PQ : distance h.P h.Q = 6)
  (h_QR : distance h.Q h.R = 7)
  (h_TU : distance h.T h.U = 4) :
  distance h.S h.T + distance h.T h.U = 7 := by sorry

end NUMINAMATH_CALUDE_hexagon_side_sum_l3946_394630


namespace NUMINAMATH_CALUDE_special_line_equation_l3946_394622

/-- A line passing through two points -/
structure Line where
  p : ℝ × ℝ
  q : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfiesEquation (point : ℝ × ℝ) (eq : LineEquation) : Prop :=
  eq.a * point.1 + eq.b * point.2 + eq.c = 0

/-- The line l passing through P(x, y) and Q(4x + 2y, x + 3y) -/
def specialLine (x y : ℝ) : Line :=
  { p := (x, y)
    q := (4*x + 2*y, x + 3*y) }

/-- The possible equations for the special line -/
def possibleEquations : List LineEquation :=
  [{ a := 1, b := -1, c := 0 },  -- x - y = 0
   { a := 1, b := -2, c := 0 }]  -- x - 2y = 0

theorem special_line_equation (x y : ℝ) :
  ∃ (eq : LineEquation), eq ∈ possibleEquations ∧
    satisfiesEquation (specialLine x y).p eq ∧
    satisfiesEquation (specialLine x y).q eq :=
  sorry


end NUMINAMATH_CALUDE_special_line_equation_l3946_394622


namespace NUMINAMATH_CALUDE_constant_quantity_l3946_394601

/-- A sequence of real numbers satisfying the recurrence relation a_{n+2} = a_{n+1} + a_n -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = a (n + 1) + a n

/-- The theorem stating that |a_n^2 - a_{n-1} a_{n+1}| is constant for n ≥ 2 -/
theorem constant_quantity (a : ℕ → ℝ) (h : RecurrenceSequence a) :
  ∃ c : ℝ, ∀ n : ℕ, n ≥ 2 → |a n ^ 2 - a (n - 1) * a (n + 1)| = c :=
sorry

end NUMINAMATH_CALUDE_constant_quantity_l3946_394601


namespace NUMINAMATH_CALUDE_popsicle_stick_difference_l3946_394602

theorem popsicle_stick_difference :
  let num_boys : ℕ := 10
  let num_girls : ℕ := 12
  let sticks_per_boy : ℕ := 15
  let sticks_per_girl : ℕ := 12
  let total_boys_sticks := num_boys * sticks_per_boy
  let total_girls_sticks := num_girls * sticks_per_girl
  total_boys_sticks - total_girls_sticks = 6 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_stick_difference_l3946_394602


namespace NUMINAMATH_CALUDE_matrix_product_is_zero_l3946_394659

-- Define the matrices
def matrix1 (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, d, -e],
    ![-d, 0, f],
    ![e, -f, 0]]

def matrix2 (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a^3, a^2*b, a^2*c],
    ![a*b^2, b^3, b^2*c],
    ![a*c^2, b*c^2, c^3]]

-- Theorem statement
theorem matrix_product_is_zero (a b c d e f : ℝ) :
  matrix1 d e f * matrix2 a b c = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_is_zero_l3946_394659


namespace NUMINAMATH_CALUDE_convex_polygon_with_equal_diagonals_l3946_394639

/-- A convex polygon with n sides and all diagonals equal -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 4
  all_diagonals_equal : Bool

/-- Theorem: If a convex n-gon (n ≥ 4) has all diagonals equal, then n is either 4 or 5 -/
theorem convex_polygon_with_equal_diagonals 
  {n : ℕ} (F : ConvexPolygon n) (h : F.all_diagonals_equal = true) : 
  n = 4 ∨ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_with_equal_diagonals_l3946_394639


namespace NUMINAMATH_CALUDE_equation_solutions_l3946_394671

theorem equation_solutions :
  (∀ x, x * (x - 6) = 2 * (x - 8) ↔ x = 4) ∧
  (∀ x, (2 * x - 1)^2 + 3 * (2 * x - 1) + 2 = 0 ↔ x = 0 ∨ x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3946_394671


namespace NUMINAMATH_CALUDE_shelter_adoption_rate_l3946_394687

def puppies_adopted_per_day (initial_puppies : ℕ) (additional_puppies : ℕ) (total_days : ℕ) : ℕ :=
  (initial_puppies + additional_puppies) / total_days

theorem shelter_adoption_rate :
  puppies_adopted_per_day 9 12 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_shelter_adoption_rate_l3946_394687


namespace NUMINAMATH_CALUDE_unique_angle_with_same_tangent_l3946_394638

theorem unique_angle_with_same_tangent :
  ∃! (n : ℤ), -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) ∧ n = 150 := by
  sorry

end NUMINAMATH_CALUDE_unique_angle_with_same_tangent_l3946_394638


namespace NUMINAMATH_CALUDE_cookies_left_l3946_394697

def cookies_problem (days : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) 
  (frank_eats_per_day : ℕ) (ted_eats : ℕ) : ℕ :=
  days * trays_per_day * cookies_per_tray - days * frank_eats_per_day - ted_eats

theorem cookies_left : 
  cookies_problem 6 2 12 1 4 = 134 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l3946_394697


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3946_394644

theorem max_value_on_circle (x y : ℝ) :
  x^2 + y^2 = 25 →
  ∃ (t_max : ℝ), t_max = 6 * Real.sqrt 10 ∧
  ∀ t, t = Real.sqrt (18 * y - 6 * x + 50) + Real.sqrt (8 * y + 6 * x + 50) →
  t ≤ t_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3946_394644


namespace NUMINAMATH_CALUDE_volume_of_rotated_circle_l3946_394618

/-- The volume of the solid generated by rotating a circle around a line -/
theorem volume_of_rotated_circle (k : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y + 1)^2 = 3}
  let line := {(x, y) : ℝ × ℝ | y = k * x - 1}
  let volume := Real.pi * 4 * Real.sqrt 3
  (∀ (x y : ℝ), (x, y) ∈ circle → (0, -1) ∈ line) →
  (∃ (r : ℝ), r^2 = 3 ∧ volume = (4/3) * Real.pi * r^3) :=
by sorry

end NUMINAMATH_CALUDE_volume_of_rotated_circle_l3946_394618


namespace NUMINAMATH_CALUDE_larger_integer_proof_l3946_394621

theorem larger_integer_proof (a b : ℕ+) : 
  (a : ℕ) + 3 = (b : ℕ) → a * b = 88 → b = 11 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l3946_394621


namespace NUMINAMATH_CALUDE_square_properties_l3946_394604

/-- Given a square and a rectangle with the same perimeter, where the rectangle
    has sides of 10 cm and 7 cm, this theorem proves the side length and area of the square. -/
theorem square_properties (square_perimeter rectangle_perimeter : ℝ)
                          (rectangle_side1 rectangle_side2 : ℝ)
                          (h1 : square_perimeter = rectangle_perimeter)
                          (h2 : rectangle_side1 = 10)
                          (h3 : rectangle_side2 = 7)
                          (h4 : rectangle_perimeter = 2 * (rectangle_side1 + rectangle_side2)) :
  ∃ (square_side : ℝ),
    square_side = 8.5 ∧
    square_perimeter = 4 * square_side ∧
    square_side ^ 2 = 72.25 := by
  sorry

end NUMINAMATH_CALUDE_square_properties_l3946_394604


namespace NUMINAMATH_CALUDE_triangle_problem_l3946_394651

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (S₁ S₂ S₃ : ℝ) 
  (h₁ : S₁ - S₂ + S₃ = Real.sqrt 3 / 2)
  (h₂ : Real.sin B = 1 / 3)
  (h₃ : S₁ = Real.sqrt 3 / 4 * a^2)
  (h₄ : S₂ = Real.sqrt 3 / 4 * b^2)
  (h₅ : S₃ = Real.sqrt 3 / 4 * c^2)
  (h₆ : a > 0 ∧ b > 0 ∧ c > 0)
  (h₇ : 0 < A ∧ A < π)
  (h₈ : 0 < B ∧ B < π)
  (h₉ : 0 < C ∧ C < π)
  (h₁₀ : A + B + C = π) :
  (∃ (S : ℝ), S = Real.sqrt 2 / 8 ∧ S = 1/2 * a * c * Real.sin B) ∧
  (Real.sin A * Real.sin C = Real.sqrt 2 / 3 → b = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3946_394651


namespace NUMINAMATH_CALUDE_vaishali_saree_stripes_l3946_394665

theorem vaishali_saree_stripes :
  ∀ (brown gold blue : ℕ),
    gold = 3 * brown →
    blue = 5 * gold →
    blue = 60 →
    brown = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_vaishali_saree_stripes_l3946_394665


namespace NUMINAMATH_CALUDE_percentage_of_female_guests_l3946_394611

theorem percentage_of_female_guests 
  (total_guests : ℕ) 
  (jays_family_females : ℕ) 
  (h1 : total_guests = 240)
  (h2 : jays_family_females = 72)
  (h3 : jays_family_females * 2 = total_guests * (percentage_female_guests / 100)) :
  percentage_female_guests = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_female_guests_l3946_394611


namespace NUMINAMATH_CALUDE_statement_B_only_incorrect_l3946_394636

-- Define the structure for a statistical statement
structure StatisticalStatement where
  label : Char
  content : String
  isCorrect : Bool

-- Define the four statements
def statementA : StatisticalStatement := {
  label := 'A',
  content := "The absolute value of the correlation coefficient approaches 1 as the linear correlation between two random variables strengthens.",
  isCorrect := true
}

def statementB : StatisticalStatement := {
  label := 'B',
  content := "In a three-shot target shooting scenario, \"at least two hits\" and \"exactly one hit\" are complementary events.",
  isCorrect := false
}

def statementC : StatisticalStatement := {
  label := 'C',
  content := "The accuracy of a model fit increases as the band of residual points in a residual plot narrows.",
  isCorrect := true
}

def statementD : StatisticalStatement := {
  label := 'B',
  content := "The variance of a dataset remains unchanged when a constant is added to each data point.",
  isCorrect := true
}

-- Define the list of all statements
def allStatements : List StatisticalStatement := [statementA, statementB, statementC, statementD]

-- Theorem: Statement B is the only incorrect statement
theorem statement_B_only_incorrect :
  ∃! s : StatisticalStatement, s ∈ allStatements ∧ ¬s.isCorrect :=
sorry

end NUMINAMATH_CALUDE_statement_B_only_incorrect_l3946_394636


namespace NUMINAMATH_CALUDE_triangle_side_length_l3946_394634

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ

-- Define the theorem
theorem triangle_side_length (t : Triangle) 
  (ha : t.a = 4) 
  (hb : t.b = 5) 
  (hS : t.S = 5 * Real.sqrt 3) :
  t.c = Real.sqrt 21 ∨ t.c = Real.sqrt 61 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l3946_394634


namespace NUMINAMATH_CALUDE_professor_arrangement_count_l3946_394615

def num_chairs : ℕ := 13
def num_students : ℕ := 9
def num_professors : ℕ := 2

def valid_professor_position (p : ℕ) : Prop :=
  2 ≤ p ∧ p ≤ 12

def valid_professor_pair (p1 p2 : ℕ) : Prop :=
  valid_professor_position p1 ∧ 
  valid_professor_position p2 ∧ 
  p1 + 1 < p2

def count_valid_arrangements : ℕ :=
  (Finset.range 9).sum (λ k => 11 - k)

theorem professor_arrangement_count :
  count_valid_arrangements = 45 :=
sorry

end NUMINAMATH_CALUDE_professor_arrangement_count_l3946_394615


namespace NUMINAMATH_CALUDE_cost_per_square_foot_calculation_l3946_394670

/-- Calculates the cost per square foot of a rented house. -/
theorem cost_per_square_foot_calculation 
  (master_bedroom_bath_area : ℝ)
  (guest_bedroom_area : ℝ)
  (num_guest_bedrooms : ℕ)
  (kitchen_bath_living_area : ℝ)
  (monthly_rent : ℝ)
  (h1 : master_bedroom_bath_area = 500)
  (h2 : guest_bedroom_area = 200)
  (h3 : num_guest_bedrooms = 2)
  (h4 : kitchen_bath_living_area = 600)
  (h5 : monthly_rent = 3000) :
  monthly_rent / (master_bedroom_bath_area + num_guest_bedrooms * guest_bedroom_area + kitchen_bath_living_area) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_per_square_foot_calculation_l3946_394670


namespace NUMINAMATH_CALUDE_min_max_values_l3946_394684

theorem min_max_values (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) 
  (h : x^2 + 4*y^2 + 4*x*y + 4*x^2*y^2 = 32) : 
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a^2 + 4*b^2 + 4*a*b + 4*a^2*b^2 = 32 → x + 2*y ≤ a + 2*b) ∧ 
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a^2 + 4*b^2 + 4*a*b + 4*a^2*b^2 = 32 → 
    Real.sqrt 7 * (x + 2*y) + 2*x*y ≥ Real.sqrt 7 * (a + 2*b) + 2*a*b) ∧
  x + 2*y = 4 ∧ 
  Real.sqrt 7 * (x + 2*y) + 2*x*y = 4 * Real.sqrt 7 + 4 := by
  sorry

end NUMINAMATH_CALUDE_min_max_values_l3946_394684


namespace NUMINAMATH_CALUDE_contrapositive_diagonals_parallelogram_l3946_394646

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define what it means for diagonals to bisect each other
def diagonals_bisect (q : Quadrilateral) : Prop :=
  let mid1 := (q.vertices 0 + q.vertices 2) / 2
  let mid2 := (q.vertices 1 + q.vertices 3) / 2
  mid1 = mid2

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.vertices 0 - q.vertices 1) = (q.vertices 3 - q.vertices 2) ∧
  (q.vertices 0 - q.vertices 3) = (q.vertices 1 - q.vertices 2)

-- The theorem to prove
theorem contrapositive_diagonals_parallelogram :
  ∀ q : Quadrilateral, ¬(is_parallelogram q) → ¬(diagonals_bisect q) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_diagonals_parallelogram_l3946_394646


namespace NUMINAMATH_CALUDE_oranges_sold_l3946_394600

def total_bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def rotten_oranges : ℕ := 50
def oranges_for_juice : ℕ := 30

theorem oranges_sold : 
  (total_bags * oranges_per_bag - rotten_oranges - oranges_for_juice) = 220 := by
  sorry

end NUMINAMATH_CALUDE_oranges_sold_l3946_394600


namespace NUMINAMATH_CALUDE_light_reflection_l3946_394650

-- Define the points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, -1)

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Define a line passing through two points
def line_through_points (p q : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p.2) * (q.1 - p.1) = (x - p.1) * (q.2 - p.2)

-- Define reflection of a point across the y-axis
def reflect_across_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- State the theorem
theorem light_reflection :
  ∃ (C : ℝ × ℝ),
    y_axis C.1 ∧
    line_through_points A C 1 1 ∧
    line_through_points (reflect_across_y_axis A) B C.1 C.2 ∧
    (∀ x y, line_through_points A C x y ↔ x - y + 1 = 0) ∧
    (∀ x y, line_through_points (reflect_across_y_axis A) B x y ↔ x + y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_light_reflection_l3946_394650


namespace NUMINAMATH_CALUDE_bakers_remaining_cakes_l3946_394647

/-- Calculates the number of remaining cakes for a baker --/
def remaining_cakes (initial : ℕ) (additional : ℕ) (sold : ℕ) : ℕ :=
  initial + additional - sold

/-- Theorem: The baker's remaining cakes is 67 --/
theorem bakers_remaining_cakes :
  remaining_cakes 62 149 144 = 67 := by
  sorry

end NUMINAMATH_CALUDE_bakers_remaining_cakes_l3946_394647


namespace NUMINAMATH_CALUDE_points_below_line_l3946_394648

theorem points_below_line 
  (a b : ℝ) 
  (n : ℕ) 
  (h_ab : 0 < a ∧ a < b) 
  (x y : ℕ → ℝ) 
  (h_x : ∀ k, x k = a + k * (b - a) / (n + 1))
  (h_y : ∀ k, y k = a * (b / a) ^ (k / (n + 1)))
  (h_k : ∀ k, k ≤ n) :
  ∀ k, y k < x k :=
sorry

end NUMINAMATH_CALUDE_points_below_line_l3946_394648


namespace NUMINAMATH_CALUDE_exactly_one_positive_integer_satisfies_condition_l3946_394612

theorem exactly_one_positive_integer_satisfies_condition : 
  ∃! (n : ℕ), n > 0 ∧ 30 - 6 * n > 18 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_positive_integer_satisfies_condition_l3946_394612


namespace NUMINAMATH_CALUDE_factorization_equality_l3946_394678

theorem factorization_equality (x y : ℝ) :
  (1 - x^2) * (1 - y^2) - 4*x*y = (x*y - 1 + x + y) * (x*y - 1 - x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3946_394678


namespace NUMINAMATH_CALUDE_books_shop1_is_65_l3946_394673

-- Define the problem parameters
def total_spent_shop1 : ℕ := 6500
def books_shop2 : ℕ := 35
def total_spent_shop2 : ℕ := 2000
def avg_price : ℕ := 85

-- Define the function to calculate the number of books from the first shop
def books_shop1 : ℕ := 
  (total_spent_shop1 + total_spent_shop2) / avg_price - books_shop2

-- Theorem to prove
theorem books_shop1_is_65 : books_shop1 = 65 := by
  sorry

end NUMINAMATH_CALUDE_books_shop1_is_65_l3946_394673


namespace NUMINAMATH_CALUDE_inequality_proof_l3946_394628

theorem inequality_proof (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 1) :
  2 * Real.sqrt ((1 - x^2) * (1 - y^2)) ≤ 2 * (1 - x) * (1 - y) + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3946_394628


namespace NUMINAMATH_CALUDE_min_red_chips_l3946_394640

/-- Represents the number of chips of each color -/
structure ChipCount where
  red : Nat
  blue : Nat

/-- Checks if a number is prime -/
def isPrime (n : Nat) : Prop := sorry

theorem min_red_chips :
  ∀ (chips : ChipCount),
  chips.red + chips.blue = 70 →
  isPrime (chips.red + 2 * chips.blue) →
  chips.red ≥ 69 :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_l3946_394640


namespace NUMINAMATH_CALUDE_shortest_track_length_l3946_394626

theorem shortest_track_length (melanie_piece_length martin_piece_length : ℕ) 
  (h1 : melanie_piece_length = 8)
  (h2 : martin_piece_length = 20) :
  Nat.lcm melanie_piece_length martin_piece_length = 40 := by
sorry

end NUMINAMATH_CALUDE_shortest_track_length_l3946_394626


namespace NUMINAMATH_CALUDE_inequality_and_range_proof_fraction_comparison_l3946_394617

theorem inequality_and_range_proof :
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → ∀ (x : ℝ), x ∈ Set.Icc (-2) 2 →
    |3*a + b| + |a - b| ≥ |a| * (|x + 1| + |x - 1|)) ∧
  (∀ (x : ℝ), (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
    |3*a + b| + |a - b| ≥ |a| * (|x + 1| + |x - 1|)) →
    x ∈ Set.Icc (-2) 2) :=
sorry

theorem fraction_comparison :
  ∀ (a b : ℝ), a ∈ Set.Ioo 0 1 → b ∈ Set.Ioo 0 1 →
    1 / (a * b) + 1 > 1 / a + 1 / b :=
sorry

end NUMINAMATH_CALUDE_inequality_and_range_proof_fraction_comparison_l3946_394617


namespace NUMINAMATH_CALUDE_tourism_revenue_scientific_notation_l3946_394645

/-- Represents the tourism revenue in yuan -/
def tourism_revenue : ℝ := 12.41e9

/-- Represents the scientific notation of the tourism revenue -/
def scientific_notation : ℝ := 1.241e9

/-- Theorem stating that the tourism revenue is equal to its scientific notation representation -/
theorem tourism_revenue_scientific_notation : tourism_revenue = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_tourism_revenue_scientific_notation_l3946_394645


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l3946_394683

theorem gcd_lcm_product (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l3946_394683


namespace NUMINAMATH_CALUDE_min_radius_for_area_l3946_394689

/-- The minimum radius of a circle with an area of at least 314 square feet is 10 feet. -/
theorem min_radius_for_area (π : ℝ) (h : π > 0) : 
  (∀ r : ℝ, π * r^2 ≥ 314 → r ≥ 10) ∧ (∃ r : ℝ, π * r^2 = 314 ∧ r = 10) := by
  sorry

end NUMINAMATH_CALUDE_min_radius_for_area_l3946_394689


namespace NUMINAMATH_CALUDE_paths_from_A_to_D_l3946_394605

/-- The number of paths between two adjacent points -/
def paths_between_adjacent : ℕ := 2

/-- The number of direct paths from A to D -/
def direct_paths : ℕ := 1

/-- The total number of paths from A to D -/
def total_paths : ℕ := paths_between_adjacent^3 + direct_paths

/-- Theorem stating that the total number of paths from A to D is 9 -/
theorem paths_from_A_to_D : total_paths = 9 := by sorry

end NUMINAMATH_CALUDE_paths_from_A_to_D_l3946_394605


namespace NUMINAMATH_CALUDE_unique_solution_cos_tan_cos_l3946_394624

open Real

theorem unique_solution_cos_tan_cos :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ arccos 0.5 ∧ cos x = tan (cos x) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cos_tan_cos_l3946_394624


namespace NUMINAMATH_CALUDE_solve_percentage_equation_l3946_394609

theorem solve_percentage_equation : ∃ x : ℝ, 0.65 * x = 0.20 * 487.50 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_solve_percentage_equation_l3946_394609


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3946_394653

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^3 + a^2*b + a*b^2 + b^3 = 0) : 
  (a^12 + b^12) / (a + b)^12 = 1/32 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3946_394653


namespace NUMINAMATH_CALUDE_perfect_square_powers_of_two_l3946_394635

theorem perfect_square_powers_of_two :
  (∃! (n : ℕ), ∃ (k : ℕ), 2^n + 3 = k^2) ∧
  (∃! (n : ℕ), ∃ (k : ℕ), 2^n + 1 = k^2) :=
by
  constructor
  · -- Proof for 2^n + 3
    sorry
  · -- Proof for 2^n + 1
    sorry

#check perfect_square_powers_of_two

end NUMINAMATH_CALUDE_perfect_square_powers_of_two_l3946_394635


namespace NUMINAMATH_CALUDE_hypergeometric_prob_and_max_likelihood_l3946_394694

/-- Hypergeometric probability distribution -/
def hypergeometric_prob (N M n m : ℕ) : ℚ :=
  (Nat.choose M m * Nat.choose (N - M) (n - m)) / Nat.choose N n

/-- Maximum likelihood estimate for population size -/
def max_likelihood_estimate (M n m : ℕ) : ℕ :=
  (M * n) / m

theorem hypergeometric_prob_and_max_likelihood 
  (N M n m : ℕ) (h1 : M ≤ N) (h2 : n ≤ N) (h3 : m ≤ M) (h4 : m ≤ n) :
  (∀ N', hypergeometric_prob N' M n m ≤ hypergeometric_prob N M n m) →
  N = max_likelihood_estimate M n m := by
  sorry


end NUMINAMATH_CALUDE_hypergeometric_prob_and_max_likelihood_l3946_394694


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l3946_394677

theorem largest_three_digit_congruence :
  ∀ n : ℕ,
  100 ≤ n ∧ n ≤ 999 ∧ (75 * n) % 300 = 225 →
  n ≤ 999 ∧
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (75 * m) % 300 = 225 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l3946_394677


namespace NUMINAMATH_CALUDE_rationalize_sqrt3_plus_1_l3946_394625

theorem rationalize_sqrt3_plus_1 :
  (1 : ℝ) / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt3_plus_1_l3946_394625


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3946_394607

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^14 - 1) % (x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3946_394607


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3946_394667

theorem polynomial_division_theorem (x : ℝ) :
  ∃ (q r : ℝ), 5*x^3 - 4*x^2 + 6*x - 9 = (x - 1) * (5*x^2 + x + 7) + r ∧ r = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3946_394667


namespace NUMINAMATH_CALUDE_parallelogram_area_l3946_394616

/-- The area of a parallelogram with base 22 cm and height 21 cm is 462 square centimeters. -/
theorem parallelogram_area : 
  let base : ℝ := 22
  let height : ℝ := 21
  let area := base * height
  area = 462 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3946_394616


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_slope_l3946_394698

/-- Given a quadrilateral ABCD inscribed in an ellipse, with three sides AB, BC, CD parallel to fixed directions,
    the slope of the fourth side DA is determined by the slopes of the other three sides and the ellipse parameters. -/
theorem inscribed_quadrilateral_slope (a b : ℝ) (m₁ m₂ m₃ : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ m : ℝ, m = (b^2 * (m₁ + m₃ - m₂) + a^2 * m₁ * m₂ * m₃) / (b^2 + a^2 * (m₁ * m₂ + m₂ * m₃ - m₁ * m₃)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_slope_l3946_394698


namespace NUMINAMATH_CALUDE_inequality_condition_l3946_394652

theorem inequality_condition (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π/2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_inequality_condition_l3946_394652


namespace NUMINAMATH_CALUDE_cab_journey_delay_l3946_394654

theorem cab_journey_delay (S : ℝ) (h : S > 0) : 
  let usual_time := 40
  let reduced_speed := (5/6) * S
  let new_time := usual_time * S / reduced_speed
  new_time - usual_time = 8 := by
sorry

end NUMINAMATH_CALUDE_cab_journey_delay_l3946_394654


namespace NUMINAMATH_CALUDE_correlation_coefficient_comparison_l3946_394663

def x : List ℝ := [1, 2, 3, 4, 5]
def y : List ℝ := [3, 5.3, 6.9, 9.1, 10.8]
def U : List ℝ := [1, 2, 3, 4, 5]
def V : List ℝ := [12.7, 10.2, 7, 3.6, 1]

def r1 : ℝ := sorry
def r2 : ℝ := sorry

theorem correlation_coefficient_comparison : r2 < 0 ∧ 0 < r1 := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_comparison_l3946_394663


namespace NUMINAMATH_CALUDE_intersection_sum_l3946_394662

theorem intersection_sum (a b : ℚ) : 
  (3 = (1/3) * 4 + a) → 
  (4 = (1/2) * 3 + b) → 
  (a + b = 25/6) := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l3946_394662


namespace NUMINAMATH_CALUDE_a_greater_than_b_l3946_394637

theorem a_greater_than_b (x y : ℝ) (h1 : x < y) (h2 : y < 0) :
  (x^2 + y^2) * (x - y) > (x^2 - y^2) * (x + y) := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l3946_394637


namespace NUMINAMATH_CALUDE_original_average_problem_l3946_394681

theorem original_average_problem (n : ℕ) (original_avg new_avg added : ℝ) : 
  n = 15 → 
  new_avg = 51 → 
  added = 11 → 
  (n : ℝ) * new_avg = (n : ℝ) * (original_avg + added) → 
  original_avg = 40 := by
sorry

end NUMINAMATH_CALUDE_original_average_problem_l3946_394681


namespace NUMINAMATH_CALUDE_focal_radius_circle_tangent_y_axis_l3946_394692

/-- Represents a parabola y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a circle with diameter equal to the focal radius of a parabola -/
structure FocalRadiusCircle (para : Parabola) where
  center : ℝ × ℝ
  radius : ℝ

/-- The circle with diameter equal to the focal radius of the parabola y^2 = 2px (p > 0) is tangent to the y-axis -/
theorem focal_radius_circle_tangent_y_axis (para : Parabola) :
  ∃ (c : FocalRadiusCircle para), c.center.1 = c.radius := by
  sorry

end NUMINAMATH_CALUDE_focal_radius_circle_tangent_y_axis_l3946_394692


namespace NUMINAMATH_CALUDE_triangle_angles_from_bisector_ratio_l3946_394668

theorem triangle_angles_from_bisector_ratio :
  ∀ (α β γ : ℝ),
  (α > 0) → (β > 0) → (γ > 0) →
  (α + β + γ = 180) →
  (∃ (k : ℝ), k > 0 ∧
    (α/2 + β/2 = 37*k) ∧
    (β/2 + γ/2 = 41*k) ∧
    (γ/2 + α/2 = 42*k)) →
  (α = 72 ∧ β = 66 ∧ γ = 42) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_from_bisector_ratio_l3946_394668


namespace NUMINAMATH_CALUDE_min_value_inequality_l3946_394676

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- Theorem statement
theorem min_value_inequality (k a b c : ℝ) : 
  (∀ x, f x ≥ k) → -- k is the minimum value of f
  (a > 0 ∧ b > 0 ∧ c > 0) → -- a, b, c are positive
  (3 / (k * a) + 3 / (2 * k * b) + 1 / (k * c) = 1) → -- given equation
  a + 2 * b + 3 * c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3946_394676


namespace NUMINAMATH_CALUDE_circle_in_triangle_l3946_394629

/-- The distance traveled by the center of a circle rolling inside a right triangle -/
def distanceTraveled (a b c r : ℝ) : ℝ :=
  (a - 2*r) + (b - 2*r) + (c - 2*r)

theorem circle_in_triangle (a b c r : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_a : a = 9) (h_b : b = 12) (h_c : c = 15) (h_r : r = 2) :
  distanceTraveled a b c r = 24 := by
sorry

end NUMINAMATH_CALUDE_circle_in_triangle_l3946_394629


namespace NUMINAMATH_CALUDE_complex_magnitude_l3946_394633

theorem complex_magnitude (w : ℂ) (h : w^2 = 48 - 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3946_394633


namespace NUMINAMATH_CALUDE_circle_center_transformation_l3946_394664

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

theorem circle_center_transformation :
  let initial_center : ℝ × ℝ := (-2, 6)
  let reflected := reflect_x initial_center
  let final_position := translate_right reflected 5
  final_position = (3, -6) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l3946_394664


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l3946_394623

theorem rectangle_area_problem (x y : ℝ) 
  (h1 : (x + 3) * (y - 1) = x * y) 
  (h2 : (x - 3) * (y + 1.5) = x * y) : 
  x * y = 90 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l3946_394623


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3946_394627

/-- The repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 36 / 99

theorem repeating_decimal_equals_fraction : repeating_decimal = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l3946_394627


namespace NUMINAMATH_CALUDE_train_distance_difference_l3946_394672

theorem train_distance_difference (v1 v2 total_distance : ℝ) 
  (h1 : v1 = 20)
  (h2 : v2 = 25)
  (h3 : total_distance = 675) :
  let t := total_distance / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  |d2 - d1| = 75 := by sorry

end NUMINAMATH_CALUDE_train_distance_difference_l3946_394672


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_is_eight_l3946_394603

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of n consecutive integers starting from a -/
def sum_consecutive (a n : ℕ) : ℕ := n * a + sum_first_n (n - 1)

/-- The proposition that n is the largest number of positive consecutive integers summing to 36 -/
def is_largest_consecutive_sum (n : ℕ) : Prop :=
  (∃ a : ℕ, a > 0 ∧ sum_consecutive a n = 36) ∧
  (∀ m : ℕ, m > n → ∀ a : ℕ, a > 0 → sum_consecutive a m ≠ 36)

theorem largest_consecutive_sum_is_eight :
  is_largest_consecutive_sum 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_is_eight_l3946_394603


namespace NUMINAMATH_CALUDE_exists_special_subset_l3946_394686

theorem exists_special_subset : ∃ (X : Set ℤ), ∀ (n : ℤ), ∃! (p : ℤ × ℤ), p.1 ∈ X ∧ p.2 ∈ X ∧ p.1 + 2 * p.2 = n := by
  sorry

end NUMINAMATH_CALUDE_exists_special_subset_l3946_394686


namespace NUMINAMATH_CALUDE_min_sphere_surface_area_l3946_394641

/-- Represents a cuboid with vertices on a sphere -/
structure CuboidOnSphere where
  -- The length of edge AB
  ab : ℝ
  -- The length of edge AD
  ad : ℝ
  -- The length of edge AA'
  aa' : ℝ
  -- The radius of the sphere
  r : ℝ
  -- All vertices are on the sphere
  vertices_on_sphere : ab ^ 2 + ad ^ 2 + aa' ^ 2 = (2 * r) ^ 2
  -- AB = 2
  ab_equals_two : ab = 2
  -- Volume of pyramid O-A'B'C'D' is 2
  pyramid_volume : (1 / 3) * ad * aa' = 2

/-- The minimum surface area of the sphere is 16π -/
theorem min_sphere_surface_area (c : CuboidOnSphere) : 
  ∃ (min_area : ℝ), min_area = 16 * π ∧ 
  ∀ (area : ℝ), area = 4 * π * c.r ^ 2 → area ≥ min_area := by
  sorry

end NUMINAMATH_CALUDE_min_sphere_surface_area_l3946_394641


namespace NUMINAMATH_CALUDE_specific_coin_flip_probability_l3946_394642

/-- The probability of getting a specific sequence of heads and tails
    when flipping a fair coin multiple times. -/
def coin_flip_probability (n : ℕ) (k : ℕ) : ℚ :=
  (1 / 2) ^ n

theorem specific_coin_flip_probability :
  coin_flip_probability 5 2 = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_specific_coin_flip_probability_l3946_394642


namespace NUMINAMATH_CALUDE_solution_of_equations_l3946_394620

theorem solution_of_equations (x : ℝ) : 
  (|x|^2 - 5*|x| + 6 = 0 ∧ x^2 - 4 = 0) ↔ (x = 2 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equations_l3946_394620


namespace NUMINAMATH_CALUDE_intersection_empty_implies_m_less_than_negative_one_l3946_394675

theorem intersection_empty_implies_m_less_than_negative_one (m : ℝ) : 
  let M := {x : ℝ | x - m ≤ 0}
  let N := {y : ℝ | ∃ x : ℝ, y = (x - 1)^2 - 1}
  M ∩ N = ∅ → m < -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_m_less_than_negative_one_l3946_394675


namespace NUMINAMATH_CALUDE_jason_picked_two_pears_l3946_394696

/-- The number of pears Keith picked -/
def keith_pears : ℕ := 3

/-- The total number of pears picked -/
def total_pears : ℕ := 5

/-- The number of pears Jason picked -/
def jason_pears : ℕ := total_pears - keith_pears

theorem jason_picked_two_pears : jason_pears = 2 := by
  sorry

end NUMINAMATH_CALUDE_jason_picked_two_pears_l3946_394696


namespace NUMINAMATH_CALUDE_a_upper_bound_l3946_394657

/-- Given a real number a, we define a function f and its derivative f' --/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2

def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

/-- We define g as the sum of f and f' --/
def g (a : ℝ) (x : ℝ) : ℝ := f a x + f' a x

/-- Main theorem: If there exists x in [1, 3] such that g(x) ≤ 0, then a ≤ 9/4 --/
theorem a_upper_bound (a : ℝ) (h : ∃ x ∈ Set.Icc 1 3, g a x ≤ 0) : a ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_a_upper_bound_l3946_394657


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_l3946_394660

/-- Represents the profit sharing ratio between two investors -/
structure ProfitRatio where
  praveen : ℕ
  hari : ℕ

/-- Calculates the profit sharing ratio based on investments and durations -/
def calculate_profit_ratio (praveen_investment : ℕ) (praveen_duration : ℕ) 
                           (hari_investment : ℕ) (hari_duration : ℕ) : ProfitRatio :=
  let praveen_contribution := praveen_investment * praveen_duration
  let hari_contribution := hari_investment * hari_duration
  let gcd := Nat.gcd praveen_contribution hari_contribution
  { praveen := praveen_contribution / gcd
  , hari := hari_contribution / gcd }

/-- Theorem stating the profit sharing ratio for the given problem -/
theorem profit_sharing_ratio : 
  calculate_profit_ratio 3220 12 8280 7 = ProfitRatio.mk 2 3 := by
  sorry

end NUMINAMATH_CALUDE_profit_sharing_ratio_l3946_394660


namespace NUMINAMATH_CALUDE_calculate_expression_l3946_394691

theorem calculate_expression : 150 * (150 - 5) + (150 * 150 + 5) = 44255 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3946_394691


namespace NUMINAMATH_CALUDE_two_students_all_pets_l3946_394656

/-- Represents the number of students in each section of the Venn diagram --/
structure PetOwnership where
  total : ℕ
  dogs : ℕ
  cats : ℕ
  other : ℕ
  no_pets : ℕ
  dogs_only : ℕ
  cats_only : ℕ
  other_only : ℕ
  dogs_and_cats : ℕ
  cats_and_other : ℕ
  dogs_and_other : ℕ
  all_three : ℕ

/-- Theorem stating that 2 students have all three types of pets --/
theorem two_students_all_pets (po : PetOwnership) : po.all_three = 2 :=
  by
  have h1 : po.total = 40 := sorry
  have h2 : po.dogs = po.total / 2 := sorry
  have h3 : po.cats = po.total * 5 / 16 := sorry
  have h4 : po.other = 8 := sorry
  have h5 : po.no_pets = 7 := sorry
  have h6 : po.dogs_only = 12 := sorry
  have h7 : po.cats_only = 3 := sorry
  have h8 : po.other_only = 2 := sorry

  have total_pet_owners : po.dogs_only + po.cats_only + po.other_only + 
    po.dogs_and_cats + po.cats_and_other + po.dogs_and_other + po.all_three = 
    po.total - po.no_pets := sorry

  have dog_owners : po.dogs_only + po.dogs_and_cats + po.dogs_and_other + po.all_three = 
    po.dogs := sorry

  have cat_owners : po.cats_only + po.dogs_and_cats + po.cats_and_other + po.all_three = 
    po.cats := sorry

  have other_pet_owners : po.other_only + po.cats_and_other + po.dogs_and_other + 
    po.all_three = po.other := sorry

  sorry


end NUMINAMATH_CALUDE_two_students_all_pets_l3946_394656


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3946_394606

/-- Given that x and y are inversely proportional, prove that y = -49 when x = -8,
    given the conditions that x + y = 42 and x = 2y for some values of x and y. -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) 
    (h1 : x * y = k)  -- x and y are inversely proportional
    (h2 : ∃ (a b : ℝ), a + b = 42 ∧ a = 2 * b ∧ a * b = k) : 
  (-8 : ℝ) * y = k → y = -49 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3946_394606


namespace NUMINAMATH_CALUDE_max_grandchildren_l3946_394608

/-- The number of grandchildren for a person with given children and grandchildren distribution -/
def grandchildren_count (num_children : ℕ) (num_children_with_same : ℕ) (num_grandchildren_same : ℕ) (num_children_different : ℕ) (num_grandchildren_different : ℕ) : ℕ :=
  (num_children_with_same * num_grandchildren_same) + (num_children_different * num_grandchildren_different)

/-- Theorem stating that Max has 58 grandchildren -/
theorem max_grandchildren :
  grandchildren_count 8 6 8 2 5 = 58 := by
  sorry

end NUMINAMATH_CALUDE_max_grandchildren_l3946_394608


namespace NUMINAMATH_CALUDE_ten_mile_taxi_cost_l3946_394610

/-- Calculates the cost of a taxi ride given the fixed cost, per-mile cost, and distance traveled. -/
def taxi_cost (fixed_cost : ℝ) (per_mile_cost : ℝ) (distance : ℝ) : ℝ :=
  fixed_cost + per_mile_cost * distance

/-- Theorem: The cost of a 10-mile taxi ride with a $2.00 fixed cost and $0.30 per-mile cost is $5.00. -/
theorem ten_mile_taxi_cost : 
  taxi_cost 2 0.3 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ten_mile_taxi_cost_l3946_394610


namespace NUMINAMATH_CALUDE_tan_300_degrees_l3946_394649

theorem tan_300_degrees : Real.tan (300 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_300_degrees_l3946_394649


namespace NUMINAMATH_CALUDE_hundred_day_previous_year_is_thursday_l3946_394619

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ

/-- Returns the day of the week for a given day in a year -/
def dayOfWeek (year : Year) (day : ℕ) : DayOfWeek :=
  sorry

/-- Checks if a year is a leap year -/
def isLeapYear (year : Year) : Bool :=
  sorry

theorem hundred_day_previous_year_is_thursday 
  (N : Year)
  (h1 : dayOfWeek N 300 = DayOfWeek.Tuesday)
  (h2 : dayOfWeek (Year.mk (N.value + 1)) 200 = DayOfWeek.Tuesday) :
  dayOfWeek (Year.mk (N.value - 1)) 100 = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_hundred_day_previous_year_is_thursday_l3946_394619


namespace NUMINAMATH_CALUDE_age_problem_l3946_394674

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 27 → 
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3946_394674


namespace NUMINAMATH_CALUDE_rational_function_uniqueness_l3946_394613

/-- A function from rational numbers to rational numbers -/
def RationalFunction := ℚ → ℚ

/-- The property that f(1) = 2 -/
def HasPropertyOne (f : RationalFunction) : Prop :=
  f 1 = 2

/-- The property that f(xy) = f(x)f(y) - f(x + y) + 1 for all x, y ∈ ℚ -/
def HasPropertyTwo (f : RationalFunction) : Prop :=
  ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1

/-- The theorem stating that any function satisfying both properties must be f(x) = x + 1 -/
theorem rational_function_uniqueness (f : RationalFunction)
  (h1 : HasPropertyOne f) (h2 : HasPropertyTwo f) :
  ∀ x : ℚ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_uniqueness_l3946_394613


namespace NUMINAMATH_CALUDE_log_stack_count_15_5_l3946_394690

/-- The number of logs in a stack with a given bottom and top row count -/
def logStackCount (bottom top : ℕ) : ℕ :=
  let n := bottom - top + 1
  n * (bottom + top) / 2

/-- Theorem: A stack of logs with 15 on the bottom row and 5 on the top row has 110 logs -/
theorem log_stack_count_15_5 : logStackCount 15 5 = 110 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_count_15_5_l3946_394690


namespace NUMINAMATH_CALUDE_investment_time_q_is_thirteen_l3946_394643

/-- Represents the investment and profit data for two partners -/
structure PartnershipData where
  investment_ratio_p : ℚ
  investment_ratio_q : ℚ
  profit_ratio_p : ℚ
  profit_ratio_q : ℚ
  investment_time_p : ℚ

/-- Calculates the investment time for partner Q given the partnership data -/
def calculate_investment_time_q (data : PartnershipData) : ℚ :=
  (data.profit_ratio_q * data.investment_ratio_p * data.investment_time_p) / 
  (data.profit_ratio_p * data.investment_ratio_q)

/-- Theorem stating that given the specified partnership data, Q's investment time is 13 months -/
theorem investment_time_q_is_thirteen : 
  let data : PartnershipData := {
    investment_ratio_p := 7,
    investment_ratio_q := 5,
    profit_ratio_p := 7,
    profit_ratio_q := 13,
    investment_time_p := 5
  }
  calculate_investment_time_q data = 13 := by sorry

end NUMINAMATH_CALUDE_investment_time_q_is_thirteen_l3946_394643


namespace NUMINAMATH_CALUDE_subtract_point_six_from_forty_five_point_nine_l3946_394669

theorem subtract_point_six_from_forty_five_point_nine : 45.9 - 0.6 = 45.3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_point_six_from_forty_five_point_nine_l3946_394669


namespace NUMINAMATH_CALUDE_total_ingredients_l3946_394679

def strawberries : ℚ := 0.2
def yogurt : ℚ := 0.1
def orange_juice : ℚ := 0.2

theorem total_ingredients : strawberries + yogurt + orange_juice = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_total_ingredients_l3946_394679


namespace NUMINAMATH_CALUDE_parallel_to_y_axis_coordinates_l3946_394631

/-- Given two points M(a-3, a+4) and N(√5, 9) in a Cartesian coordinate system,
    if the line MN is parallel to the y-axis, then M has coordinates (√5, 7 + √5) -/
theorem parallel_to_y_axis_coordinates (a : ℝ) :
  let M : ℝ × ℝ := (a - 3, a + 4)
  let N : ℝ × ℝ := (Real.sqrt 5, 9)
  (M.1 = N.1) →  -- MN is parallel to y-axis iff x-coordinates are equal
  M = (Real.sqrt 5, 7 + Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_parallel_to_y_axis_coordinates_l3946_394631


namespace NUMINAMATH_CALUDE_m_eq_one_sufficient_not_necessary_l3946_394695

/-- Two lines in the plane -/
structure Line :=
  (a b c : ℝ)

/-- The condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The lines l1 and l2 from the problem -/
def l1 (m : ℝ) : Line := ⟨m, 1, -1⟩
def l2 (m : ℝ) : Line := ⟨m-2, m, -1⟩

/-- The statement to be proved -/
theorem m_eq_one_sufficient_not_necessary :
  (∀ m : ℝ, m = 1 → perpendicular (l1 m) (l2 m)) ∧
  ¬(∀ m : ℝ, perpendicular (l1 m) (l2 m) → m = 1) :=
sorry

end NUMINAMATH_CALUDE_m_eq_one_sufficient_not_necessary_l3946_394695


namespace NUMINAMATH_CALUDE_remainder_17_power_100_mod_7_l3946_394658

theorem remainder_17_power_100_mod_7 : 17^100 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_power_100_mod_7_l3946_394658


namespace NUMINAMATH_CALUDE_coefficient_of_x_l3946_394682

theorem coefficient_of_x (some_number : ℝ) : 
  (2 * (1/2)^2 + some_number * (1/2) - 5 = 0) → some_number = 9 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l3946_394682


namespace NUMINAMATH_CALUDE_factor_and_multiple_of_thirteen_l3946_394632

theorem factor_and_multiple_of_thirteen (n : ℕ) : 
  (∃ k : ℕ, 13 = n * k) ∧ (∃ m : ℕ, n = 13 * m) → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_factor_and_multiple_of_thirteen_l3946_394632


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3946_394680

theorem sqrt_inequality (x : ℝ) (h : x ≥ 4) :
  Real.sqrt (x - 3) - Real.sqrt (x - 1) > Real.sqrt (x - 4) - Real.sqrt (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3946_394680


namespace NUMINAMATH_CALUDE_bulk_warehouse_case_size_l3946_394688

/-- Proves the number of cans in a bulk warehouse case given pricing information -/
theorem bulk_warehouse_case_size (bulk_case_price : ℚ) (grocery_price : ℚ) (grocery_cans : ℕ) (price_difference : ℚ) : 
  bulk_case_price = 12 →
  grocery_price = 6 →
  grocery_cans = 12 →
  price_difference = 1/4 →
  (bulk_case_price / ((grocery_price / grocery_cans) - price_difference) : ℚ) = 48 :=
by sorry

end NUMINAMATH_CALUDE_bulk_warehouse_case_size_l3946_394688
