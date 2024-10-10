import Mathlib

namespace range_of_a_l103_10374

-- Define the function representing |x-2|+|x+3|
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Define the condition that the solution set is ℝ
def solution_set_is_reals (a : ℝ) : Prop :=
  ∀ x : ℝ, f x ≥ a

-- Theorem statement
theorem range_of_a (a : ℝ) :
  solution_set_is_reals a ↔ a ∈ Set.Iic 5 :=
sorry

end range_of_a_l103_10374


namespace function_f_at_zero_l103_10387

/-- A function f: ℝ → ℝ satisfying f(x+y) = f(x) + f(y) + 1/2 for all real x and y -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + (1/2 : ℝ)

/-- Theorem: For a function f satisfying the given property, f(0) = -1/2 -/
theorem function_f_at_zero (f : ℝ → ℝ) (h : FunctionF f) : f 0 = -1/2 := by
  sorry

end function_f_at_zero_l103_10387


namespace statement_D_no_related_factor_l103_10321

-- Define a type for statements
inductive Statement
| A : Statement  -- A timely snow promises a good harvest
| B : Statement  -- If the upper beam is not straight, the lower beam will be crooked
| C : Statement  -- Smoking is harmful to health
| D : Statement  -- Magpies signify joy, crows signify mourning

-- Define what it means for a statement to have a related factor
def has_related_factor (s : Statement) : Prop :=
  ∃ (x y : Prop), (x → y) ∧ (s = Statement.A ∨ s = Statement.B ∨ s = Statement.C)

-- Theorem: Statement D does not have a related factor
theorem statement_D_no_related_factor :
  ¬ has_related_factor Statement.D :=
by
  sorry


end statement_D_no_related_factor_l103_10321


namespace complete_square_sum_l103_10371

theorem complete_square_sum (x : ℝ) : 
  (x^2 - 10*x + 15 = 0) → 
  ∃ (a b : ℤ), ((x + a : ℝ)^2 = b) ∧ (a + b = 5) :=
by sorry

end complete_square_sum_l103_10371


namespace sum_interior_angles_pentagon_l103_10303

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A pentagon is a polygon with 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The sum of the interior angles of a pentagon is 540° -/
theorem sum_interior_angles_pentagon :
  sum_interior_angles pentagon_sides = 540 := by sorry

end sum_interior_angles_pentagon_l103_10303


namespace triangle_side_length_l103_10304

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  A = 30 * π / 180 →
  B = 45 * π / 180 →
  b = 8 →
  a / Real.sin A = b / Real.sin B →
  a = 4 * Real.sqrt 2 := by
sorry

end triangle_side_length_l103_10304


namespace cubic_sum_l103_10366

theorem cubic_sum (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (h : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 := by
sorry

end cubic_sum_l103_10366


namespace dollar_operation_theorem_l103_10342

/-- Define the dollar operation -/
def dollar (k : ℝ) (a b : ℝ) : ℝ := k * (a - b)^2

/-- Theorem stating that (2x - 3y)² $₃ (3y - 2x)² = 0 for any real x and y -/
theorem dollar_operation_theorem (x y : ℝ) : 
  dollar 3 ((2*x - 3*y)^2) ((3*y - 2*x)^2) = 0 := by
  sorry

#check dollar_operation_theorem

end dollar_operation_theorem_l103_10342


namespace greek_cross_dissection_l103_10331

/-- Represents a symmetric Greek cross -/
structure SymmetricGreekCross where
  -- Add necessary properties to define a symmetric Greek cross

/-- Represents a square -/
structure Square where
  -- Add necessary properties to define a square

/-- Represents a part of the dissected cross -/
inductive CrossPart
  | SmallCross : SymmetricGreekCross → CrossPart
  | OtherPart : CrossPart

/-- Theorem stating that a symmetric Greek cross can be dissected as described -/
theorem greek_cross_dissection (cross : SymmetricGreekCross) :
  ∃ (parts : Finset CrossPart) (square : Square),
    parts.card = 5 ∧
    (∃ small_cross : SymmetricGreekCross, CrossPart.SmallCross small_cross ∈ parts) ∧
    (∃ other_parts : Finset CrossPart,
      other_parts.card = 4 ∧
      (∀ p ∈ other_parts, p ∈ parts ∧ p ≠ CrossPart.SmallCross small_cross) ∧
      -- Here we would need to define how the other parts form the square
      True) := by
  sorry

end greek_cross_dissection_l103_10331


namespace line_properties_l103_10363

/-- Point type representing a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing a parametric line -/
structure Line where
  p : Point
  α : ℝ

/-- Function to calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Function to get the x-intercept of a line -/
def xIntercept (l : Line) : Point := sorry

/-- Function to get the y-intercept of a line -/
def yIntercept (l : Line) : Point := sorry

/-- Function to convert a line to its polar form -/
def toPolarForm (l : Line) : ℝ → ℝ := sorry

theorem line_properties (P : Point) (l : Line) (h1 : P.x = 2 ∧ P.y = 1) 
    (h2 : l.p = P) 
    (h3 : ∀ t : ℝ, ∃ x y : ℝ, x = 2 + t * Real.cos l.α ∧ y = 1 + t * Real.sin l.α)
    (h4 : distance P (xIntercept l) * distance P (yIntercept l) = 4) :
  l.α = 3 * Real.pi / 4 ∧ 
  ∀ θ : ℝ, toPolarForm l θ * (Real.cos θ + Real.sin θ) = 3 := by
  sorry

end line_properties_l103_10363


namespace group_50_properties_l103_10301

def last_number (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2)

def first_number (n : ℕ) : ℕ := last_number n - 2 * (n - 1)

def sum_of_group (n : ℕ) : ℕ := n * (first_number n + last_number n) / 2

theorem group_50_properties :
  last_number 50 = 2550 ∧
  first_number 50 = 2452 ∧
  sum_of_group 50 = 50 * 2501 := by
  sorry

end group_50_properties_l103_10301


namespace hexagon_diagonal_theorem_l103_10367

/-- A convex hexagon in a 2D plane -/
structure ConvexHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_convex : sorry

/-- The area of a convex hexagon -/
def area (h : ConvexHexagon) : ℝ := sorry

/-- A diagonal of a convex hexagon -/
def diagonal (h : ConvexHexagon) (i j : Fin 6) : ℝ × ℝ → ℝ × ℝ := sorry

/-- The area of a triangle formed by a diagonal and two adjacent vertices -/
def triangle_area (h : ConvexHexagon) (i j : Fin 6) : ℝ := sorry

/-- Main theorem: In any convex hexagon, there exists a diagonal that separates
    a triangle with area no more than 1/6 of the hexagon's area -/
theorem hexagon_diagonal_theorem (h : ConvexHexagon) :
  ∃ (i j : Fin 6), triangle_area h i j ≤ (1 / 6) * area h := by sorry

end hexagon_diagonal_theorem_l103_10367


namespace prime_triplet_existence_l103_10318

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem prime_triplet_existence :
  (∃ n : ℕ, isPrime (n - 96) ∧ isPrime n ∧ isPrime (n + 96)) ∧
  (¬∃ n : ℕ, isPrime (n - 1996) ∧ isPrime n ∧ isPrime (n + 1996)) :=
sorry

end prime_triplet_existence_l103_10318


namespace function_not_in_first_quadrant_l103_10346

theorem function_not_in_first_quadrant
  (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : b < -1) :
  ∀ x y : ℝ, y = a^x + b → ¬(x > 0 ∧ y > 0) :=
by sorry

end function_not_in_first_quadrant_l103_10346


namespace min_lcm_a_c_l103_10307

theorem min_lcm_a_c (a b c : ℕ+) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 21) :
  ∃ (a' c' : ℕ+), Nat.lcm a' c' = 420 ∧ ∀ (x y : ℕ+), Nat.lcm x b = 20 → Nat.lcm b y = 21 → Nat.lcm a' c' ≤ Nat.lcm x y :=
sorry

end min_lcm_a_c_l103_10307


namespace no_solution_for_four_l103_10379

theorem no_solution_for_four : 
  ∀ X : ℕ, X < 10 →
  (∀ Y : ℕ, Y < 10 → ¬(100 * X + 30 + Y) % 11 = 0) ↔ X = 4 := by
  sorry

end no_solution_for_four_l103_10379


namespace kolya_can_break_rods_to_form_triangles_l103_10348

/-- Represents a rod broken into three parts -/
structure BrokenRod :=
  (part1 : ℝ)
  (part2 : ℝ)
  (part3 : ℝ)
  (sum_to_one : part1 + part2 + part3 = 1)
  (all_positive : part1 > 0 ∧ part2 > 0 ∧ part3 > 0)

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Checks if it's possible to form three triangles from three broken rods -/
def can_form_three_triangles (rod1 rod2 rod3 : BrokenRod) : Prop :=
  ∃ (perm1 perm2 perm3 : Fin 3 → Fin 3),
    can_form_triangle (rod1.part1) (rod2.part1) (rod3.part1) ∧
    can_form_triangle (rod1.part2) (rod2.part2) (rod3.part2) ∧
    can_form_triangle (rod1.part3) (rod2.part3) (rod3.part3)

/-- The main theorem stating that Kolya can break the rods to always form three triangles -/
theorem kolya_can_break_rods_to_form_triangles :
  ∃ (kolya_rod1 kolya_rod2 : BrokenRod),
    ∀ (vasya_rod : BrokenRod),
      can_form_three_triangles kolya_rod1 vasya_rod kolya_rod2 :=
sorry

end kolya_can_break_rods_to_form_triangles_l103_10348


namespace radical_sum_product_l103_10300

theorem radical_sum_product (x y : ℝ) : 
  (x + Real.sqrt y) + (x - Real.sqrt y) = 6 →
  (x + Real.sqrt y) * (x - Real.sqrt y) = 4 →
  x + y = 8 := by
  sorry

end radical_sum_product_l103_10300


namespace sequence_existence_condition_l103_10355

def is_valid_sequence (x : ℕ → Fin 2) (n m : ℕ) : Prop :=
  (∀ i, x i = 0 → x (i + m) = 1) ∧ (∀ i, x i = 1 → x (i + n) = 0)

theorem sequence_existence_condition (n m : ℕ) :
  (∃ x : ℕ → Fin 2, is_valid_sequence x n m) ↔
  (∃ (d p q : ℕ), n = 2^d * p ∧ m = 2^d * q ∧ Odd p ∧ Odd q) :=
sorry

end sequence_existence_condition_l103_10355


namespace distance_between_points_l103_10306

def point1 : ℝ × ℝ := (0, 3)
def point2 : ℝ × ℝ := (4, -5)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = 4 * Real.sqrt 5 := by
  sorry

end distance_between_points_l103_10306


namespace enrollment_difference_l103_10340

/-- Represents the enrollment of a school --/
structure School where
  name : String
  enrollment : Nat

/-- Theorem: The positive difference between the maximum and minimum enrollments is 700 --/
theorem enrollment_difference (schools : List School) 
    (h1 : schools = [
      ⟨"Varsity", 1150⟩, 
      ⟨"Northwest", 1530⟩, 
      ⟨"Central", 1850⟩, 
      ⟨"Greenbriar", 1680⟩, 
      ⟨"Riverside", 1320⟩
    ]) : 
    (List.maximum (schools.map School.enrollment)).getD 0 - 
    (List.minimum (schools.map School.enrollment)).getD 0 = 700 := by
  sorry


end enrollment_difference_l103_10340


namespace circle_radius_l103_10347

/-- A circle with center (0, k) where k > 10, tangent to y = x, y = -x, y = 10, and x-axis has radius 20 -/
theorem circle_radius (k : ℝ) (h1 : k > 10) : 
  let circle := { (x, y) | x^2 + (y - k)^2 = (k - 10)^2 }
  (∀ (x y : ℝ), (x = y ∨ x = -y ∨ y = 10 ∨ y = 0) → 
    (x, y) ∈ circle → x^2 + (y - k)^2 = (k - 10)^2) →
  k - 10 = 20 := by
sorry

end circle_radius_l103_10347


namespace min_max_abs_quadratic_l103_10354

theorem min_max_abs_quadratic :
  ∃ y : ℝ, ∀ z : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x^2 - x*y + x| ≤ (⨆ x ∈ {x : ℝ | 0 ≤ x ∧ x ≤ 2}, |x^2 - x*z + x|)) ∧
  (⨆ x ∈ {x : ℝ | 0 ≤ x ∧ x ≤ 2}, |x^2 - x*y + x|) = 0 :=
by sorry

end min_max_abs_quadratic_l103_10354


namespace quadratic_completion_l103_10329

theorem quadratic_completion (b : ℝ) (m : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 36 = (x + m)^2 + 20) → 
  b = 8 := by
sorry

end quadratic_completion_l103_10329


namespace solve_equation_l103_10368

theorem solve_equation (x y : ℝ) : y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end solve_equation_l103_10368


namespace tangent_lines_to_circle_l103_10302

/-- A line in 2D space, represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space, represented by the equation (x-h)^2 + (y-k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  (c.h * l.a + c.k * l.b + l.c)^2 = (l.a^2 + l.b^2) * c.r^2

/-- The main theorem -/
theorem tangent_lines_to_circle (l : Line) (c : Circle) :
  (l.a = 2 ∧ l.b = -1 ∧ c.h = 0 ∧ c.k = 0 ∧ c.r^2 = 5) →
  (∃ l1 l2 : Line,
    (are_parallel l l1 ∧ is_tangent l1 c) ∧
    (are_parallel l l2 ∧ is_tangent l2 c) ∧
    (l1.c = 5 ∨ l1.c = -5) ∧
    (l2.c = 5 ∨ l2.c = -5) ∧
    (l1.c + l2.c = 0)) :=
by sorry

end tangent_lines_to_circle_l103_10302


namespace divisibility_problem_l103_10349

theorem divisibility_problem (x y : ℕ) : 
  (∀ z : ℕ, z < x → ¬((1056 + z) % 28 = 0 ∧ (1056 + z) % 42 = 0)) ∧
  ((1056 + x) % 28 = 0 ∧ (1056 + x) % 42 = 0) ∧
  (∀ w : ℕ, w > y → ¬((1056 - w) % 28 = 0 ∧ (1056 - w) % 42 = 0)) ∧
  ((1056 - y) % 28 = 0 ∧ (1056 - y) % 42 = 0) →
  x = 36 ∧ y = 48 := by
sorry

end divisibility_problem_l103_10349


namespace two_digit_perfect_square_conditions_l103_10327

theorem two_digit_perfect_square_conditions : ∃! n : ℕ, 
  10 ≤ n ∧ n ≤ 99 ∧ 
  (∃ m : ℕ, 2 * n + 1 = m * m) ∧ 
  (∃ k : ℕ, 3 * n + 1 = k * k) ∧ 
  n = 40 := by
sorry

end two_digit_perfect_square_conditions_l103_10327


namespace trigonometric_identity_l103_10373

/-- 
Given x = γ sin((θ - α)/2) and y = γ sin((θ + α)/2), 
prove that x^2 - 2xy cos α + y^2 = γ^2 sin^2 α
-/
theorem trigonometric_identity 
  (γ θ α x y : ℝ) 
  (hx : x = γ * Real.sin ((θ - α) / 2))
  (hy : y = γ * Real.sin ((θ + α) / 2)) :
  x^2 - 2*x*y*Real.cos α + y^2 = γ^2 * Real.sin α^2 := by
  sorry


end trigonometric_identity_l103_10373


namespace algebra_test_female_students_l103_10370

theorem algebra_test_female_students 
  (total_average : ℝ) 
  (num_male : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 88) 
  (h2 : num_male = 15) 
  (h3 : male_average = 80) 
  (h4 : female_average = 94) : 
  ∃ (num_female : ℕ), 
    (num_male * male_average + num_female * female_average) / (num_male + num_female) = total_average ∧ 
    num_female = 20 := by
sorry


end algebra_test_female_students_l103_10370


namespace sum_of_fractions_geq_three_l103_10390

theorem sum_of_fractions_geq_three (a b c : ℝ) (h : a * b * c = 1) :
  (1 + a + a * b) / (1 + b + a * b) +
  (1 + b + b * c) / (1 + c + b * c) +
  (1 + c + a * c) / (1 + a + a * c) ≥ 3 := by
  sorry

end sum_of_fractions_geq_three_l103_10390


namespace quadratic_equation_roots_l103_10338

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x - 3 = 0 ∧ y^2 + m*y - 3 = 0) ∧
  (3^2 + m*3 - 3 = 0 → (-1)^2 + m*(-1) - 3 = 0) := by
sorry

end quadratic_equation_roots_l103_10338


namespace geometric_arithmetic_sequence_ratio_l103_10328

/-- Given a geometric sequence {a_n} with positive terms and common ratio q,
    if 3a_1, (1/2)a_3, and 2a_2 form an arithmetic sequence, then q = 3. -/
theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence with ratio q
  (3 * a 1 - (1/2) * a 3) = ((1/2) * a 3 - 2 * a 2) →  -- Arithmetic sequence condition
  q = 3 :=
by sorry

end geometric_arithmetic_sequence_ratio_l103_10328


namespace sophists_count_l103_10384

/-- Represents the types of inhabitants on the Isle of Logic -/
inductive Inhabitant
  | Knight
  | Liar
  | Sophist

/-- The Isle of Logic and its inhabitants -/
structure IsleOfLogic where
  knights : Nat
  liars : Nat
  sophists : Nat

/-- Predicate to check if a statement is valid for a sophist -/
def isSophistStatement (isle : IsleOfLogic) (statementLiars : Nat) : Prop :=
  statementLiars ≠ isle.liars ∧ 
  ¬(statementLiars = isle.liars + 1 ∧ isle.sophists > isle.liars)

/-- Theorem: The number of sophists on the Isle of Logic -/
theorem sophists_count (isle : IsleOfLogic) : 
  isle.knights = 40 →
  isle.liars = 25 →
  isSophistStatement isle 26 →
  isle.sophists ≤ 26 →
  isle.sophists = 27 := by
  sorry

end sophists_count_l103_10384


namespace inverse_proportionality_l103_10378

/-- Two real numbers are inversely proportional if their product is constant. -/
theorem inverse_proportionality (x y k : ℝ) (h : x * y = k) :
  ∃ (c : ℝ), ∀ (x' y' : ℝ), x' * y' = k → y' = c / x' :=
by sorry

end inverse_proportionality_l103_10378


namespace derrick_has_34_pictures_l103_10350

/-- The number of wild animal pictures Ralph has -/
def ralph_pictures : ℕ := 26

/-- The additional number of pictures Derrick has compared to Ralph -/
def additional_pictures : ℕ := 8

/-- The number of wild animal pictures Derrick has -/
def derrick_pictures : ℕ := ralph_pictures + additional_pictures

/-- Theorem stating that Derrick has 34 wild animal pictures -/
theorem derrick_has_34_pictures : derrick_pictures = 34 := by sorry

end derrick_has_34_pictures_l103_10350


namespace cannot_be_equation_l103_10341

-- Define the linear function
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the condition that the function passes through (-1, -3)
def passes_through_point (k b : ℝ) : Prop :=
  linear_function k b (-1) = -3

-- Define the condition that the distances from intercepts to origin are equal
def equal_intercept_distances (k b : ℝ) : Prop :=
  abs (b / k) = abs b

-- Theorem statement
theorem cannot_be_equation (k b : ℝ) 
  (h1 : passes_through_point k b) 
  (h2 : equal_intercept_distances k b) :
  ¬(k = -3 ∧ b = -6) :=
sorry

end cannot_be_equation_l103_10341


namespace lawn_mowing_time_l103_10323

/-- Time required to mow a rectangular lawn -/
theorem lawn_mowing_time : 
  ∀ (length width swath_width overlap speed : ℝ),
  length = 90 →
  width = 150 →
  swath_width = 28 / 12 →
  overlap = 4 / 12 →
  speed = 5000 →
  (width / (swath_width - overlap) * length) / speed = 1.35 := by
  sorry

end lawn_mowing_time_l103_10323


namespace sum_of_roots_l103_10377

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 15*a^2 + 20*a - 50 = 0)
  (hb : 8*b^3 - 60*b^2 - 290*b + 2575 = 0) : 
  a + b = 15/2 := by
sorry

end sum_of_roots_l103_10377


namespace negation_of_existence_square_plus_one_positive_negation_l103_10399

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem square_plus_one_positive_negation :
  (¬∃ x : ℝ, x^2 + 1 > 0) ↔ (∀ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end negation_of_existence_square_plus_one_positive_negation_l103_10399


namespace post_circumference_l103_10397

/-- Given a cylindrical post and a squirrel's spiral path, calculate the post's circumference -/
theorem post_circumference (post_height : ℝ) (spiral_rise_per_circuit : ℝ) (squirrel_travel : ℝ) 
  (h1 : post_height = 25)
  (h2 : spiral_rise_per_circuit = 5)
  (h3 : squirrel_travel = 15) :
  squirrel_travel / (post_height / spiral_rise_per_circuit) = 5 := by
  sorry

end post_circumference_l103_10397


namespace restaurant_period_days_l103_10334

def pies_per_day : ℕ := 8
def total_pies : ℕ := 56

theorem restaurant_period_days : 
  total_pies / pies_per_day = 7 := by sorry

end restaurant_period_days_l103_10334


namespace other_piece_price_is_96_l103_10313

/-- The price of one of the other pieces of clothing --/
def other_piece_price (total_spent : ℕ) (num_pieces : ℕ) (price1 : ℕ) (price2 : ℕ) : ℕ :=
  (total_spent - price1 - price2) / (num_pieces - 2)

/-- Theorem stating that the price of one of the other pieces is 96 --/
theorem other_piece_price_is_96 :
  other_piece_price 610 7 49 81 = 96 := by
  sorry

end other_piece_price_is_96_l103_10313


namespace parallel_vectors_m_value_l103_10317

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (6, 3)
  let b : ℝ → ℝ × ℝ := fun m ↦ (m, 2)
  ∀ m : ℝ, are_parallel a (b m) → m = 4 := by
  sorry

end parallel_vectors_m_value_l103_10317


namespace largest_constant_inequality_l103_10380

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 3 ≥ C*(x + y + z)) ↔ C ≤ 2 :=
by sorry

end largest_constant_inequality_l103_10380


namespace first_platform_length_l103_10324

/-- Given a train and two platforms, calculate the length of the first platform. -/
theorem first_platform_length
  (train_length : ℝ)
  (first_crossing_time : ℝ)
  (second_platform_length : ℝ)
  (second_crossing_time : ℝ)
  (h1 : train_length = 310)
  (h2 : first_crossing_time = 15)
  (h3 : second_platform_length = 250)
  (h4 : second_crossing_time = 20) :
  ∃ (first_platform_length : ℝ),
    first_platform_length = 110 ∧
    (train_length + first_platform_length) / first_crossing_time =
    (train_length + second_platform_length) / second_crossing_time :=
by sorry

end first_platform_length_l103_10324


namespace sandy_nickels_theorem_sandy_specific_case_l103_10336

/-- The number of nickels Sandy has after her dad borrows some -/
def nickels_remaining (initial_nickels borrowed_nickels : ℕ) : ℕ :=
  initial_nickels - borrowed_nickels

/-- Theorem stating that Sandy's remaining nickels is the difference between initial and borrowed -/
theorem sandy_nickels_theorem (initial_nickels borrowed_nickels : ℕ) 
  (h : borrowed_nickels ≤ initial_nickels) :
  nickels_remaining initial_nickels borrowed_nickels = initial_nickels - borrowed_nickels :=
by
  sorry

/-- Sandy's specific case -/
theorem sandy_specific_case :
  nickels_remaining 31 20 = 11 :=
by
  sorry

end sandy_nickels_theorem_sandy_specific_case_l103_10336


namespace no_double_application_function_l103_10386

theorem no_double_application_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 2017 := by
  sorry

end no_double_application_function_l103_10386


namespace find_y_l103_10351

theorem find_y : ∃ y : ℚ, (12 : ℚ)^2 * (6 : ℚ)^3 / y = 72 → y = 432 := by sorry

end find_y_l103_10351


namespace nine_power_comparison_l103_10360

theorem nine_power_comparison : 9^(10^10) > 9^20 := by
  sorry

end nine_power_comparison_l103_10360


namespace quilt_shaded_area_is_40_percent_l103_10312

/-- Represents a square quilt with shaded areas -/
structure Quilt where
  total_squares : ℕ
  fully_shaded : ℕ
  half_shaded : ℕ
  quarter_shaded : ℕ

/-- Calculates the percentage of shaded area in the quilt -/
def shaded_percentage (q : Quilt) : ℚ :=
  let shaded_area := q.fully_shaded + q.half_shaded / 2 + q.quarter_shaded / 2
  (shaded_area / q.total_squares) * 100

/-- Theorem stating that the given quilt has 40% shaded area -/
theorem quilt_shaded_area_is_40_percent :
  let q := Quilt.mk 25 4 8 4
  shaded_percentage q = 40 := by sorry

end quilt_shaded_area_is_40_percent_l103_10312


namespace joey_age_l103_10311

def brothers_ages : List ℕ := [4, 6, 8, 10, 12]

def movies_condition (a b : ℕ) : Prop := a + b = 18

def park_condition (a b : ℕ) : Prop := a < 9 ∧ b < 9

theorem joey_age : 
  ∃ (a b c d : ℕ),
    a ∈ brothers_ages ∧
    b ∈ brothers_ages ∧
    c ∈ brothers_ages ∧
    d ∈ brothers_ages ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    movies_condition a b ∧
    park_condition c d ∧
    6 ∉ [a, b, c, d] →
    10 ∈ brothers_ages \ [a, b, c, d, 6] :=
by
  sorry

end joey_age_l103_10311


namespace log_equation_sum_l103_10395

theorem log_equation_sum (A B C : ℕ+) (h_coprime : Nat.Coprime A B ∧ Nat.Coprime A C ∧ Nat.Coprime B C)
  (h_eq : A * Real.log 5 / Real.log 180 + B * Real.log 3 / Real.log 180 + C * Real.log 2 / Real.log 180 = 1) :
  A + B + C = 5 := by
  sorry

end log_equation_sum_l103_10395


namespace five_solutions_l103_10319

/-- The number of distinct ordered pairs of positive integers satisfying the equation -/
def count_solutions : ℕ := 5

/-- The equation that the ordered pairs must satisfy -/
def satisfies_equation (x y : ℕ+) : Prop :=
  (x.val ^ 4 * y.val ^ 4) - (20 * x.val ^ 2 * y.val ^ 2) + 64 = 0

/-- The theorem stating that there are exactly 5 distinct ordered pairs satisfying the equation -/
theorem five_solutions :
  (∃! (s : Finset (ℕ+ × ℕ+)), s.card = count_solutions ∧
    ∀ p ∈ s, satisfies_equation p.1 p.2 ∧
    ∀ p : ℕ+ × ℕ+, satisfies_equation p.1 p.2 → p ∈ s) :=
  sorry

end five_solutions_l103_10319


namespace f_expression_and_g_monotonicity_l103_10345

/-- A linear function f that is increasing on ℝ and satisfies f(f(x)) = 16x + 5 -/
def f : ℝ → ℝ :=
  sorry

/-- g is defined as g(x) = f(x)(x+m) -/
def g (m : ℝ) : ℝ → ℝ :=
  λ x ↦ f x * (x + m)

theorem f_expression_and_g_monotonicity :
  (∀ x y, x < y → f x < f y) ∧  -- f is increasing
  (∀ x, f (f x) = 16 * x + 5) →  -- f(f(x)) = 16x + 5
  (f = λ x ↦ 4 * x + 1) ∧  -- f(x) = 4x + 1
  (∀ m, (∀ x y, 1 < x ∧ x < y → g m x < g m y) → -9/4 ≤ m)  -- If g is increasing on (1,+∞), then m ≥ -9/4
  := by sorry

end f_expression_and_g_monotonicity_l103_10345


namespace geometric_sequence_sum_l103_10372

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 + a 3 = 4) →
  (a 4 + a 5 = 16) →
  (a 8 + a 9 = 256) :=
by
  sorry

end geometric_sequence_sum_l103_10372


namespace negation_equivalence_l103_10391

theorem negation_equivalence :
  (¬ ∃ x : ℝ, 2 * x^2 < Real.cos x) ↔ (∀ x : ℝ, 2 * x^2 ≥ Real.cos x) :=
by sorry

end negation_equivalence_l103_10391


namespace set_inclusion_implies_a_range_l103_10322

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x * (x - a) < 0}
def B : Set ℝ := {x | x^2 - 7*x - 18 < 0}

-- State the theorem
theorem set_inclusion_implies_a_range (a : ℝ) : A a ⊆ B → a ∈ Set.Icc (-2) 9 := by
  sorry

end set_inclusion_implies_a_range_l103_10322


namespace paint_mixture_theorem_l103_10352

/-- Proves that mixing 5 gallons of 20% yellow paint with 5/3 gallons of 40% yellow paint 
    results in a mixture that is 25% yellow -/
theorem paint_mixture_theorem (x : ℝ) :
  let light_green_volume : ℝ := 5
  let light_green_yellow_percent : ℝ := 0.2
  let dark_green_yellow_percent : ℝ := 0.4
  let target_yellow_percent : ℝ := 0.25
  x = 5/3 →
  (light_green_volume * light_green_yellow_percent + x * dark_green_yellow_percent) / 
  (light_green_volume + x) = target_yellow_percent :=
by sorry

end paint_mixture_theorem_l103_10352


namespace beads_per_bracelet_l103_10394

/-- The number of beaded necklaces made on Monday -/
def monday_necklaces : ℕ := 10

/-- The number of beaded necklaces made on Tuesday -/
def tuesday_necklaces : ℕ := 2

/-- The number of beaded bracelets made on Wednesday -/
def wednesday_bracelets : ℕ := 5

/-- The number of beaded earrings made on Wednesday -/
def wednesday_earrings : ℕ := 7

/-- The number of beads needed to make one beaded necklace -/
def beads_per_necklace : ℕ := 20

/-- The number of beads needed to make one beaded earring -/
def beads_per_earring : ℕ := 5

/-- The total number of beads used by Kylie -/
def total_beads : ℕ := 325

/-- Theorem stating that 10 beads are needed to make one beaded bracelet -/
theorem beads_per_bracelet : 
  (total_beads - 
   (monday_necklaces + tuesday_necklaces) * beads_per_necklace - 
   wednesday_earrings * beads_per_earring) / wednesday_bracelets = 10 := by
  sorry

end beads_per_bracelet_l103_10394


namespace system_of_equations_solution_l103_10376

theorem system_of_equations_solution (a b : ℝ) 
  (eq1 : 2 * a + b = 7) 
  (eq2 : a - b = 2) : 
  3 * a = 9 := by sorry

end system_of_equations_solution_l103_10376


namespace cauchy_not_dense_implies_linear_l103_10369

/-- A function satisfying the Cauchy functional equation -/
def CauchyFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- The graph of a function is not dense in the plane -/
def NotDenseGraph (f : ℝ → ℝ) : Prop :=
  ∃ U : Set (ℝ × ℝ), IsOpen U ∧ U.Nonempty ∧ ∀ x : ℝ, (x, f x) ∉ U

/-- A function is linear -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

theorem cauchy_not_dense_implies_linear (f : ℝ → ℝ) 
  (h_cauchy : CauchyFunction f) (h_not_dense : NotDenseGraph f) : 
  LinearFunction f := by
  sorry

end cauchy_not_dense_implies_linear_l103_10369


namespace box_length_l103_10388

/-- The length of a rectangular box given specific conditions --/
theorem box_length (width : ℝ) (volume_gallons : ℝ) (height_inches : ℝ) (conversion_factor : ℝ) : 
  width = 25 →
  volume_gallons = 4687.5 →
  height_inches = 6 →
  conversion_factor = 7.5 →
  ∃ (length : ℝ), length = 50 := by
sorry

end box_length_l103_10388


namespace base_conversion_568_to_octal_l103_10315

theorem base_conversion_568_to_octal :
  (1 * 8^3 + 0 * 8^2 + 7 * 8^1 + 0 * 8^0 : ℕ) = 568 := by
  sorry

end base_conversion_568_to_octal_l103_10315


namespace max_surface_area_30_cubes_l103_10389

/-- Represents a configuration of connected unit cubes -/
structure CubeConfiguration where
  num_cubes : ℕ
  surface_area : ℕ

/-- The number of cubes in our problem -/
def total_cubes : ℕ := 30

/-- Function to calculate the surface area of a linear arrangement of cubes -/
def linear_arrangement_surface_area (n : ℕ) : ℕ :=
  if n ≤ 1 then 6 * n else 2 + 4 * n

/-- Theorem stating that the maximum surface area for 30 connected unit cubes is 122 -/
theorem max_surface_area_30_cubes :
  (∀ c : CubeConfiguration, c.num_cubes = total_cubes → c.surface_area ≤ 122) ∧
  (∃ c : CubeConfiguration, c.num_cubes = total_cubes ∧ c.surface_area = 122) := by
  sorry

#eval linear_arrangement_surface_area total_cubes

end max_surface_area_30_cubes_l103_10389


namespace x_lt_2_necessary_not_sufficient_for_x_lt_0_l103_10362

theorem x_lt_2_necessary_not_sufficient_for_x_lt_0 :
  (∀ x : ℝ, x < 0 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ x ≥ 0) :=
by sorry

end x_lt_2_necessary_not_sufficient_for_x_lt_0_l103_10362


namespace negation_equivalence_l103_10326

theorem negation_equivalence (a b x : ℝ) :
  ¬(x ≠ a ∧ x ≠ b → x^2 - (a+b)*x + a*b ≠ 0) ↔ (x = a ∨ x = b → x^2 - (a+b)*x + a*b = 0) :=
sorry

end negation_equivalence_l103_10326


namespace sum_zero_iff_squared_sum_equal_l103_10333

theorem sum_zero_iff_squared_sum_equal {a b c : ℝ} (h : ¬(a = b ∧ b = c)) :
  a + b + c = 0 ↔ a^2 + a*b + b^2 = b^2 + b*c + c^2 ∧ b^2 + b*c + c^2 = c^2 + c*a + a^2 :=
sorry

end sum_zero_iff_squared_sum_equal_l103_10333


namespace cloth_cost_price_l103_10393

/-- Given a shopkeeper selling cloth with the following conditions:
  * The shopkeeper sells 200 metres of cloth
  * The selling price is Rs. 12000
  * The shopkeeper incurs a loss of Rs. 6 per metre
  Prove that the cost price for one metre of cloth is Rs. 66 -/
theorem cloth_cost_price 
  (total_metres : ℕ) 
  (selling_price : ℕ) 
  (loss_per_metre : ℕ) 
  (h1 : total_metres = 200)
  (h2 : selling_price = 12000)
  (h3 : loss_per_metre = 6) :
  (selling_price + total_metres * loss_per_metre) / total_metres = 66 := by
  sorry

end cloth_cost_price_l103_10393


namespace stamp_exchange_theorem_l103_10356

/-- Represents the number of stamp collectors and countries -/
def n : ℕ := 26

/-- The minimum number of letters needed to exchange stamps -/
def min_letters (n : ℕ) : ℕ := 2 * (n - 1)

/-- Theorem stating the minimum number of letters needed for stamp exchange -/
theorem stamp_exchange_theorem :
  min_letters n = 50 :=
by sorry

end stamp_exchange_theorem_l103_10356


namespace hexagon_walk_distance_l103_10344

def regular_hexagon_side_length : ℝ := 3
def walk_distance : ℝ := 10

theorem hexagon_walk_distance (start_point end_point : ℝ × ℝ) : 
  start_point = (0, 0) →
  end_point = (0.5, -Real.sqrt 3 / 2) →
  Real.sqrt ((end_point.1 - start_point.1)^2 + (end_point.2 - start_point.2)^2) = 1 :=
by sorry

end hexagon_walk_distance_l103_10344


namespace smallest_value_l103_10305

theorem smallest_value (y : ℝ) (h1 : 0 < y) (h2 : y < 1) :
  y^3 < 3*y ∧ y^3 < y^(1/2) ∧ y^3 < 1/y ∧ y^3 < Real.exp y := by
  sorry

#check smallest_value

end smallest_value_l103_10305


namespace cyclic_sum_inequality_l103_10309

/-- For positive real numbers a, b, c ≤ √2 with abc = 2, 
    prove √2 ∑(ab + 3c)/(3ab + c) ≥ a + b + c -/
theorem cyclic_sum_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha2 : a ≤ Real.sqrt 2) (hb2 : b ≤ Real.sqrt 2) (hc2 : c ≤ Real.sqrt 2)
  (habc : a * b * c = 2) :
  Real.sqrt 2 * (((a * b + 3 * c) / (3 * a * b + c)) +
                 ((b * c + 3 * a) / (3 * b * c + a)) +
                 ((c * a + 3 * b) / (3 * c * a + b))) ≥ a + b + c := by
  sorry

end cyclic_sum_inequality_l103_10309


namespace ice_cream_cones_l103_10332

theorem ice_cream_cones (cost_per_cone total_spent : ℕ) (h1 : cost_per_cone = 99) (h2 : total_spent = 198) :
  total_spent / cost_per_cone = 2 :=
by sorry

end ice_cream_cones_l103_10332


namespace coefficient_x6_in_expansion_l103_10358

theorem coefficient_x6_in_expansion : ∃ c : ℤ, c = -10 ∧ 
  (Polynomial.coeff ((1 + X + X^2) * (1 - X)^6) 6 = c) := by
  sorry

end coefficient_x6_in_expansion_l103_10358


namespace roots_squared_relation_l103_10381

def f (x : ℝ) : ℝ := 2 * x^3 - x^2 + 4 * x - 3

def g (b c d x : ℝ) : ℝ := x^3 + b * x^2 + c * x + d

theorem roots_squared_relation (b c d : ℝ) :
  (∀ r : ℝ, f r = 0 → g b c d (r^2) = 0) →
  b = 15/4 ∧ c = 5/2 ∧ d = -9/4 := by
  sorry

end roots_squared_relation_l103_10381


namespace negation_of_all_cats_not_pets_l103_10335

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for "is a cat" and "is a pet"
variable (Cat : U → Prop)
variable (Pet : U → Prop)

-- Define the original statement "All cats are not pets"
def all_cats_not_pets : Prop := ∀ x, Cat x → ¬(Pet x)

-- Define the negation "Some cats are pets"
def some_cats_are_pets : Prop := ∃ x, Cat x ∧ Pet x

-- Theorem statement
theorem negation_of_all_cats_not_pets :
  ¬(all_cats_not_pets U Cat Pet) ↔ some_cats_are_pets U Cat Pet :=
sorry

end negation_of_all_cats_not_pets_l103_10335


namespace sixth_piggy_bank_coins_l103_10343

def coin_sequence (n : ℕ) : ℕ := 72 + 9 * (n - 1)

theorem sixth_piggy_bank_coins :
  coin_sequence 6 = 117 := by
  sorry

end sixth_piggy_bank_coins_l103_10343


namespace rectangle_opposite_vertex_l103_10337

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Predicate to check if four points form a rectangle --/
def is_rectangle (r : Rectangle) : Prop :=
  let midpoint1 := ((r.v1.1 + r.v3.1) / 2, (r.v1.2 + r.v3.2) / 2)
  let midpoint2 := ((r.v2.1 + r.v4.1) / 2, (r.v2.2 + r.v4.2) / 2)
  midpoint1 = midpoint2

/-- The theorem to be proved --/
theorem rectangle_opposite_vertex 
  (r : Rectangle)
  (h1 : r.v1 = (5, 10))
  (h2 : r.v3 = (15, -6))
  (h3 : r.v2 = (11, 2))
  (h4 : is_rectangle r) :
  r.v4 = (9, 2) := by
  sorry


end rectangle_opposite_vertex_l103_10337


namespace focal_length_of_specific_conic_l103_10357

/-- A conic section centered at the origin with coordinate axes as its axes of symmetry -/
structure ConicSection where
  /-- The eccentricity of the conic section -/
  eccentricity : ℝ
  /-- A point that the conic section passes through -/
  point : ℝ × ℝ

/-- The focal length of a conic section -/
def focalLength (c : ConicSection) : ℝ := sorry

/-- Theorem: The focal length of the specified conic section is 6√2 -/
theorem focal_length_of_specific_conic :
  let c : ConicSection := { eccentricity := Real.sqrt 2, point := (5, 4) }
  focalLength c = 6 * Real.sqrt 2 := by sorry

end focal_length_of_specific_conic_l103_10357


namespace smallest_four_digit_multiple_l103_10325

theorem smallest_four_digit_multiple : ∃ (n : ℕ), 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  (2 ∣ n) ∧ (3 ∣ n) ∧ (8 ∣ n) ∧ (9 ∣ n) ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ (2 ∣ m) ∧ (3 ∣ m) ∧ (8 ∣ m) ∧ (9 ∣ m) → m ≥ n) ∧
  n = 1008 :=
by sorry

end smallest_four_digit_multiple_l103_10325


namespace altitude_length_l103_10314

/-- Given a rectangle with length l and width w, and a triangle constructed on its diagonal
    with an area equal to the rectangle's area, the length of the altitude drawn from the
    opposite vertex of the triangle to the diagonal is (2lw) / √(l^2 + w^2). -/
theorem altitude_length (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let diagonal := Real.sqrt (l^2 + w^2)
  let rectangle_area := l * w
  let triangle_area := (1/2) * diagonal * altitude
  altitude = (2 * l * w) / diagonal →
  triangle_area = rectangle_area :=
by
  sorry


end altitude_length_l103_10314


namespace smallest_whole_number_gt_100_odd_factors_l103_10383

theorem smallest_whole_number_gt_100_odd_factors : ∀ n : ℕ, n > 100 → (∃ k : ℕ, n = k^2) → ∀ m : ℕ, m > 100 → (∃ j : ℕ, m = j^2) → n ≤ m → n = 121 := by
  sorry

end smallest_whole_number_gt_100_odd_factors_l103_10383


namespace M_remainder_l103_10365

/-- Number of red flags -/
def red_flags : ℕ := 13

/-- Number of yellow flags -/
def yellow_flags : ℕ := 12

/-- Total number of flags -/
def total_flags : ℕ := red_flags + yellow_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Function to calculate the number of distinguishable arrangements -/
noncomputable def M : ℕ := sorry

/-- Theorem stating the remainder when M is divided by 1000 -/
theorem M_remainder : M % 1000 = 188 := by sorry

end M_remainder_l103_10365


namespace original_light_wattage_l103_10396

theorem original_light_wattage (new_wattage : ℝ) (increase_percentage : ℝ) :
  new_wattage = 67.2 ∧ 
  increase_percentage = 0.12 →
  ∃ original_wattage : ℝ,
    new_wattage = original_wattage * (1 + increase_percentage) ∧
    original_wattage = 60 := by
  sorry

end original_light_wattage_l103_10396


namespace manager_percentage_reduction_l103_10375

/-- Calculates the percentage of managers after some leave the room. -/
def target_percentage (total_employees : ℕ) (initial_percentage : ℚ) (managers_leaving : ℚ) : ℚ :=
  let initial_managers : ℚ := (initial_percentage / 100) * total_employees
  let remaining_managers : ℚ := initial_managers - managers_leaving
  (remaining_managers / total_employees) * 100

/-- The target percentage of managers is approximately 49% -/
theorem manager_percentage_reduction :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (target_percentage 100 99 49.99999999999996 - 49) < ε :=
sorry

end manager_percentage_reduction_l103_10375


namespace smallest_number_of_nuts_l103_10353

theorem smallest_number_of_nuts (N : ℕ) : N = 320 ↔ 
  N > 0 ∧
  N % 11 = 1 ∧
  N % 13 = 8 ∧
  N % 17 = 3 ∧
  N > 41 ∧
  (∀ M : ℕ, M > 0 ∧ M % 11 = 1 ∧ M % 13 = 8 ∧ M % 17 = 3 ∧ M > 41 → N ≤ M) :=
by sorry

end smallest_number_of_nuts_l103_10353


namespace cos_squared_derivative_l103_10316

theorem cos_squared_derivative (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = (Real.cos (2 * x))^2) →
  (deriv f) x = -2 * Real.sin (4 * x) := by
sorry

end cos_squared_derivative_l103_10316


namespace plain_croissant_price_l103_10320

/-- The price of Sean's Sunday pastry purchase --/
def sean_pastry_purchase : ℝ → Prop :=
  fun plain_croissant_price =>
    let almond_croissant_price : ℝ := 4.50
    let salami_cheese_croissant_price : ℝ := 4.50
    let focaccia_price : ℝ := 4.00
    let latte_price : ℝ := 2.50
    let total_spent : ℝ := 21.00
    
    almond_croissant_price +
    salami_cheese_croissant_price +
    plain_croissant_price +
    focaccia_price +
    2 * latte_price = total_spent

theorem plain_croissant_price : ∃ (price : ℝ), sean_pastry_purchase price ∧ price = 3.00 := by
  sorry

end plain_croissant_price_l103_10320


namespace cubic_equation_roots_l103_10330

theorem cubic_equation_roots : ∃ (z : ℂ), z^3 + z^2 - z = 7 + 7*I :=
by
  -- Prove that 4 + i and -3 - i are roots of the equation
  have h1 : (4 + I)^3 + (4 + I)^2 - (4 + I) = 7 + 7*I := by sorry
  have h2 : (-3 - I)^3 + (-3 - I)^2 - (-3 - I) = 7 + 7*I := by sorry
  
  -- Show that at least one of these roots satisfies the equation
  exact ⟨4 + I, h1⟩

-- Note: This theorem only proves the existence of one root,
-- but we know there are at least two roots satisfying the equation.

end cubic_equation_roots_l103_10330


namespace trapezoid_median_length_l103_10310

/-- Given a trapezoid and an equilateral triangle with specific properties,
    prove that the median of the trapezoid has length 24. -/
theorem trapezoid_median_length
  (trapezoid_area : ℝ) 
  (triangle_area : ℝ) 
  (trapezoid_height : ℝ) 
  (triangle_height : ℝ) 
  (h1 : trapezoid_area = 3 * triangle_area)
  (h2 : trapezoid_height = 8 * Real.sqrt 3)
  (h3 : triangle_height = 8 * Real.sqrt 3) :
  trapezoid_area / trapezoid_height = 24 := by
  sorry


end trapezoid_median_length_l103_10310


namespace cubic_difference_l103_10382

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 65) : 
  a^3 - b^3 = 511 := by
  sorry

end cubic_difference_l103_10382


namespace bathroom_size_is_150_l103_10364

/-- Represents the size of a bathroom module in square feet -/
def bathroom_size : ℝ := sorry

/-- Represents the total size of the home in square feet -/
def total_home_size : ℝ := 2000

/-- Represents the size of the kitchen module in square feet -/
def kitchen_size : ℝ := 400

/-- Represents the cost of the kitchen module in dollars -/
def kitchen_cost : ℝ := 20000

/-- Represents the cost of a bathroom module in dollars -/
def bathroom_cost : ℝ := 12000

/-- Represents the cost per square foot of other modules in dollars -/
def other_module_cost_per_sqft : ℝ := 100

/-- Represents the total cost of the home in dollars -/
def total_home_cost : ℝ := 174000

/-- Represents the number of bathrooms in the home -/
def num_bathrooms : ℕ := 2

/-- Theorem stating that the bathroom size is 150 square feet -/
theorem bathroom_size_is_150 : bathroom_size = 150 := by
  sorry

end bathroom_size_is_150_l103_10364


namespace cannot_make_55_cents_l103_10398

def coin_values : List Nat := [5, 10, 25, 50]

theorem cannot_make_55_cents (coins : List Nat) : 
  (coins.length = 6 ∧ 
   ∀ c ∈ coins, c ∈ coin_values) → 
  coins.sum ≠ 55 := by
  sorry

end cannot_make_55_cents_l103_10398


namespace oliver_seashell_collection_l103_10359

-- Define the number of seashells collected on each day
def monday_shells : ℕ := 2
def tuesday_shells : ℕ := 2

-- Define the total number of seashells
def total_shells : ℕ := monday_shells + tuesday_shells

-- Theorem statement
theorem oliver_seashell_collection : total_shells = 4 := by
  sorry

end oliver_seashell_collection_l103_10359


namespace sheridan_fish_problem_l103_10339

/-- The number of fish Mrs. Sheridan's sister gave her -/
def fish_given (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem sheridan_fish_problem : fish_given 22 69 = 47 := by sorry

end sheridan_fish_problem_l103_10339


namespace article_pricing_theorem_l103_10361

/-- Represents the price and profit relationship for an article -/
structure ArticlePricing where
  cost_price : ℝ
  profit_price : ℝ
  loss_price : ℝ
  desired_profit_price : ℝ

/-- The main theorem about the article pricing -/
theorem article_pricing_theorem (a : ArticlePricing) 
  (h1 : a.profit_price - a.cost_price = a.cost_price - a.loss_price)
  (h2 : a.profit_price = 832)
  (h3 : a.desired_profit_price = 896) :
  a.cost_price * 1.4 = a.desired_profit_price :=
sorry

#check article_pricing_theorem

end article_pricing_theorem_l103_10361


namespace true_discount_calculation_l103_10392

/-- Calculates the true discount given the banker's discount and present value -/
def true_discount (bankers_discount : ℚ) (present_value : ℚ) : ℚ :=
  bankers_discount / (1 + bankers_discount / present_value)

/-- Theorem stating that given a banker's discount of 36 and a present value of 180, 
    the true discount is 30 -/
theorem true_discount_calculation :
  true_discount 36 180 = 30 := by
  sorry

#eval true_discount 36 180

end true_discount_calculation_l103_10392


namespace smallest_satisfying_number_l103_10385

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m ^ 3 = n

def prime_factors (n : ℕ) : List ℕ := sorry

def satisfies_conditions (n : ℕ) : Prop :=
  n > 0 ∧
  ¬ is_prime n ∧
  ¬ is_cube n ∧
  (prime_factors n).length % 2 = 0 ∧
  ∀ p ∈ prime_factors n, p > 60

theorem smallest_satisfying_number : 
  satisfies_conditions 3721 ∧ 
  ∀ m : ℕ, m < 3721 → ¬ satisfies_conditions m :=
sorry

end smallest_satisfying_number_l103_10385


namespace fraction_equation_solution_l103_10308

theorem fraction_equation_solution : 
  let x : ℚ := 24
  (1 : ℚ) / 3 + (1 : ℚ) / 2 + (1 : ℚ) / x = (7 : ℚ) / 8 := by
  sorry

end fraction_equation_solution_l103_10308
