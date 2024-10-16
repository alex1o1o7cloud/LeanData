import Mathlib

namespace NUMINAMATH_CALUDE_candidate_percentage_l880_88027

theorem candidate_percentage (passing_marks total_marks : ℕ) 
  (first_candidate_marks second_candidate_marks : ℕ) : 
  passing_marks = 160 →
  first_candidate_marks = passing_marks - 40 →
  second_candidate_marks = passing_marks + 20 →
  second_candidate_marks = total_marks * 30 / 100 →
  first_candidate_marks * 100 / total_marks = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_candidate_percentage_l880_88027


namespace NUMINAMATH_CALUDE_modulus_of_complex_product_l880_88017

theorem modulus_of_complex_product : ∃ (z : ℂ), z = (Complex.I - 2) * (2 * Complex.I + 1) ∧ Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_product_l880_88017


namespace NUMINAMATH_CALUDE_yellow_better_for_fine_gift_l880_88022

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Yellow

/-- Represents the contents of the bag -/
structure Bag :=
  (red : Nat)
  (yellow : Nat)

/-- Calculates the probability of drawing two balls of the same color -/
def probSameColor (b : Bag) : Rat :=
  let total := b.red + b.yellow
  let sameRed := (b.red * (b.red - 1)) / 2
  let sameYellow := (b.yellow * (b.yellow - 1)) / 2
  (sameRed + sameYellow) / ((total * (total - 1)) / 2)

/-- The initial bag configuration -/
def initialBag : Bag := ⟨1, 3⟩

/-- Theorem: Adding a yellow ball gives a higher probability of drawing two balls of the same color -/
theorem yellow_better_for_fine_gift :
  probSameColor ⟨initialBag.red, initialBag.yellow + 1⟩ > 
  probSameColor ⟨initialBag.red + 1, initialBag.yellow⟩ :=
sorry

end NUMINAMATH_CALUDE_yellow_better_for_fine_gift_l880_88022


namespace NUMINAMATH_CALUDE_train_platform_ratio_l880_88053

/-- Given a train passing a pole and a platform, prove the ratio of platform length to train length -/
theorem train_platform_ratio (l t v : ℝ) (h1 : l > 0) (h2 : t > 0) (h3 : v > 0) :
  let pole_time := t
  let platform_time := 3.5 * t
  let train_length := l
  let platform_length := v * platform_time - train_length
  platform_length / train_length = 2.5 := by sorry

end NUMINAMATH_CALUDE_train_platform_ratio_l880_88053


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l880_88074

theorem largest_four_digit_divisible_by_five : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0 → n ≤ 9995 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l880_88074


namespace NUMINAMATH_CALUDE_bees_after_seven_days_l880_88031

/-- Calculates the total number of bees in a hive after a given number of days -/
def total_bees_in_hive (initial_bees : ℕ) (hatch_rate : ℕ) (loss_rate : ℕ) (days : ℕ) : ℕ :=
  initial_bees + days * (hatch_rate - loss_rate) + 1

/-- Theorem stating the total number of bees in the hive after 7 days -/
theorem bees_after_seven_days :
  total_bees_in_hive 12500 3000 900 7 = 27201 := by
  sorry

#eval total_bees_in_hive 12500 3000 900 7

end NUMINAMATH_CALUDE_bees_after_seven_days_l880_88031


namespace NUMINAMATH_CALUDE_volume_of_112_ounces_l880_88096

/-- A substance with volume directly proportional to weight -/
structure Substance where
  /-- Constant of proportionality between volume and weight -/
  k : ℚ
  /-- Assumption: k is positive -/
  k_pos : k > 0

/-- Volume of the substance given its weight -/
def volume (s : Substance) (weight : ℚ) : ℚ :=
  s.k * weight

theorem volume_of_112_ounces (s : Substance) 
  (h : volume s 63 = 27) : volume s 112 = 48 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_112_ounces_l880_88096


namespace NUMINAMATH_CALUDE_call_center_problem_l880_88015

theorem call_center_problem (team_a_agents : ℚ) (team_b_agents : ℚ) 
  (team_a_calls_per_agent : ℚ) (team_b_calls_per_agent : ℚ) :
  team_a_agents = (5 / 8) * team_b_agents →
  team_a_calls_per_agent = (2 / 5) * team_b_calls_per_agent →
  let total_calls := team_a_agents * team_a_calls_per_agent + team_b_agents * team_b_calls_per_agent
  (team_b_agents * team_b_calls_per_agent) / total_calls = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_call_center_problem_l880_88015


namespace NUMINAMATH_CALUDE_system_solution_sum_reciprocals_l880_88082

theorem system_solution_sum_reciprocals (x₀ y₀ : ℚ) :
  x₀ / 3 + y₀ / 5 = 1 ∧ x₀ / 5 + y₀ / 3 = 1 →
  1 / x₀ + 1 / y₀ = 16 / 15 := by
sorry

end NUMINAMATH_CALUDE_system_solution_sum_reciprocals_l880_88082


namespace NUMINAMATH_CALUDE_problem_statement_l880_88007

theorem problem_statement (a : ℝ) 
  (A : Set ℝ) (hA : A = {0, 2, a^2})
  (B : Set ℝ) (hB : B = {1, a})
  (hUnion : A ∪ B = {0, 1, 2, 4}) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l880_88007


namespace NUMINAMATH_CALUDE_square_side_length_l880_88071

theorem square_side_length (area : ℝ) (side : ℝ) :
  area = 625 →
  side * side = area →
  side = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l880_88071


namespace NUMINAMATH_CALUDE_monthly_income_proof_l880_88065

/-- Given the average monthly incomes of three people, prove the income of one person. -/
theorem monthly_income_proof (P Q R : ℝ) 
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (Q + R) / 2 = 5250)
  (h3 : (P + R) / 2 = 6200) :
  P = 3000 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_proof_l880_88065


namespace NUMINAMATH_CALUDE_total_cost_is_2250_l880_88028

def apple_quantity : ℕ := 8
def apple_price : ℕ := 70
def mango_quantity : ℕ := 9
def mango_price : ℕ := 55
def orange_quantity : ℕ := 5
def orange_price : ℕ := 40
def banana_quantity : ℕ := 12
def banana_price : ℕ := 30
def grape_quantity : ℕ := 7
def grape_price : ℕ := 45
def cherry_quantity : ℕ := 4
def cherry_price : ℕ := 80

def total_cost : ℕ := 
  apple_quantity * apple_price + 
  mango_quantity * mango_price + 
  orange_quantity * orange_price + 
  banana_quantity * banana_price + 
  grape_quantity * grape_price + 
  cherry_quantity * cherry_price

theorem total_cost_is_2250 : total_cost = 2250 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_2250_l880_88028


namespace NUMINAMATH_CALUDE_find_c_l880_88042

theorem find_c (p q : ℝ → ℝ) (c : ℝ) 
  (hp : ∀ x, p x = 3 * x - 8)
  (hq : ∀ x, q x = 4 * x - c)
  (h_pq3 : p (q 3) = 14) :
  c = 14 / 3 := by
sorry

end NUMINAMATH_CALUDE_find_c_l880_88042


namespace NUMINAMATH_CALUDE_expression_value_l880_88098

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : a ≠ 0) 
  (h3 : c * d = 1) 
  (h4 : |m| = 3) : 
  m^2 - (-1) + |a + b| - c * d * m = 7 ∨ m^2 - (-1) + |a + b| - c * d * m = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l880_88098


namespace NUMINAMATH_CALUDE_part_one_part_two_l880_88036

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |3 * x + m|

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := f m x - 2 * |x - 1|

-- Part I
theorem part_one (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 3 ↔ f m x - m ≤ 9) → m = -3 := by sorry

-- Part II
theorem part_two (m : ℝ) (h_m_pos : m > 0) :
  (∃ A B C : ℝ × ℝ, 
    A.2 = 0 ∧ B.2 = 0 ∧ C.2 = g m C.1 ∧
    C.1 ∈ Set.Ioo A.1 B.1 ∧
    (1/2) * |B.1 - A.1| * |C.2| > 60) →
  m > 12 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l880_88036


namespace NUMINAMATH_CALUDE_select_and_swap_count_l880_88023

def num_people : ℕ := 8
def num_selected : ℕ := 3

def ways_to_select_and_swap : ℕ := Nat.choose num_people num_selected * (Nat.factorial 2)

theorem select_and_swap_count :
  ways_to_select_and_swap = Nat.choose num_people num_selected * (Nat.factorial 2) :=
by sorry

end NUMINAMATH_CALUDE_select_and_swap_count_l880_88023


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l880_88019

/-- Given an ellipse with equation x²/m + y²/4 = 1 and focal length 4,
    prove that the length of its major axis is 4√2. -/
theorem ellipse_major_axis_length (m : ℝ) :
  (∀ x y : ℝ, x^2 / m + y^2 / 4 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c = 4 ∧ c^2 = m - 4) →     -- Focal length is 4
  (∃ a : ℝ, a = 2 * Real.sqrt 2 ∧ 2 * a = 4 * Real.sqrt 2) := -- Major axis length is 4√2
by sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l880_88019


namespace NUMINAMATH_CALUDE_equation_solution_l880_88032

theorem equation_solution (x y : ℚ) : 
  (4 * x + 2 * y = 12) → 
  (2 * x + 4 * y = 16) → 
  (20 * x^2 + 24 * x * y + 20 * y^2 = 3280 / 9) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l880_88032


namespace NUMINAMATH_CALUDE_max_value_sum_l880_88087

/-- Given positive real numbers x, y, and z satisfying 4x^2 + 9y^2 + 16z^2 = 144,
    the maximum value N of the expression 3xz + 5yz + 8xy plus the sum of x, y, and z
    that produce this maximum is equal to 319. -/
theorem max_value_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 4*x^2 + 9*y^2 + 16*z^2 = 144) :
  ∃ (N x_N y_N z_N : ℝ),
    (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → 4*x'^2 + 9*y'^2 + 16*z'^2 = 144 →
      3*x'*z' + 5*y'*z' + 8*x'*y' ≤ N) ∧
    3*x_N*z_N + 5*y_N*z_N + 8*x_N*y_N = N ∧
    4*x_N^2 + 9*y_N^2 + 16*z_N^2 = 144 ∧
    N + x_N + y_N + z_N = 319 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_l880_88087


namespace NUMINAMATH_CALUDE_monotonic_decreasing_implies_t_bound_l880_88099

-- Define the function f(x)
def f (t : ℝ) (x : ℝ) : ℝ := x^3 - t*x^2 + 3*x

-- Define the derivative of f(x)
def f_derivative (t : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*t*x + 3

-- State the theorem
theorem monotonic_decreasing_implies_t_bound :
  ∀ t : ℝ, (∀ x ∈ Set.Icc 1 4, f_derivative t x ≤ 0) →
  t ≥ 51/8 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_implies_t_bound_l880_88099


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_l880_88089

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci
def are_foci (F₁ F₂ : ℝ × ℝ) : Prop := 
  let (x₁, y₁) := F₁
  let (x₂, y₂) := F₂
  x₁^2 + y₁^2 = 5 ∧ x₂^2 + y₂^2 = 5 ∧ x₁ = -x₂ ∧ y₁ = -y₂

-- Define the distance ratio condition
def distance_ratio (P F₁ F₂ : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  d₁ / d₂ = 2

-- Theorem statement
theorem ellipse_triangle_area 
  (P F₁ F₂ : ℝ × ℝ) 
  (h₁ : is_on_ellipse P.1 P.2) 
  (h₂ : are_foci F₁ F₂) 
  (h₃ : distance_ratio P F₁ F₂) : 
  let area := Real.sqrt (
    (F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 +
    (F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 +
    (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2
  ) / 4
  area = 4 := by sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_l880_88089


namespace NUMINAMATH_CALUDE_circle_equation_l880_88005

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop := y = -4 * x

-- Define the tangent point
def tangent_point : ℝ × ℝ := (3, -2)

-- State the theorem
theorem circle_equation 
  (C : Circle) 
  (h1 : line_l (tangent_point.1) (tangent_point.2))
  (h2 : center_line C.center.1 C.center.2)
  (h3 : ∃ (t : ℝ), C.center.1 + t * (tangent_point.1 - C.center.1) = tangent_point.1 ∧
                   C.center.2 + t * (tangent_point.2 - C.center.2) = tangent_point.2 ∧
                   t = 1) :
  ∀ (x y : ℝ), (x - 1)^2 + (y + 4)^2 = 8 ↔ 
    (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l880_88005


namespace NUMINAMATH_CALUDE_campers_fed_l880_88081

/-- Represents the types of fish caught -/
inductive FishType
  | Trout
  | Bass
  | Salmon

/-- Represents the catch of fish -/
structure Catch where
  troutWeight : ℕ
  bassCount : ℕ
  bassWeight : ℕ
  salmonCount : ℕ
  salmonWeight : ℕ

/-- Calculates the total weight of fish caught -/
def totalWeight (c : Catch) : ℕ :=
  c.troutWeight + c.bassCount * c.bassWeight + c.salmonCount * c.salmonWeight

/-- Calculates the number of campers that can be fed -/
def campersCanFeed (c : Catch) (poundsPerPerson : ℕ) : ℕ :=
  totalWeight c / poundsPerPerson

/-- Theorem stating the number of campers that can be fed -/
theorem campers_fed (c : Catch) (poundsPerPerson : ℕ) :
  c.troutWeight = 8 ∧ 
  c.bassCount = 6 ∧ 
  c.bassWeight = 2 ∧ 
  c.salmonCount = 2 ∧ 
  c.salmonWeight = 12 ∧ 
  poundsPerPerson = 2 → 
  campersCanFeed c poundsPerPerson = 22 := by
  sorry

end NUMINAMATH_CALUDE_campers_fed_l880_88081


namespace NUMINAMATH_CALUDE_intersection_points_count_l880_88034

/-- A line in a 2D plane, represented by coefficients a, b, and c in the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determine if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- Determine if two lines intersect -/
def intersect (l1 l2 : Line) : Prop :=
  ¬(parallel l1 l2)

/-- The three lines given in the problem -/
def line1 : Line := ⟨2, -3, 4⟩
def line2 : Line := ⟨3, 4, 6⟩
def line3 : Line := ⟨6, -9, 8⟩

/-- The theorem to be proved -/
theorem intersection_points_count :
  (intersect line1 line2 ∧ intersect line2 line3 ∧ parallel line1 line3) ∧
  (∃! p : ℝ × ℝ, p.1 * line1.a + p.2 * line1.b = line1.c ∧ p.1 * line2.a + p.2 * line2.b = line2.c) ∧
  (∃! p : ℝ × ℝ, p.1 * line2.a + p.2 * line2.b = line2.c ∧ p.1 * line3.a + p.2 * line3.b = line3.c) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_count_l880_88034


namespace NUMINAMATH_CALUDE_correct_yeast_counting_operation_l880_88001

/-- Represents an experimental operation -/
inductive ExperimentalOperation
  | YeastCounting
  | PigmentSeparation
  | AuxinRooting
  | Plasmolysis

/-- Determines if an experimental operation is correct -/
def is_correct_operation (op : ExperimentalOperation) : Prop :=
  match op with
  | ExperimentalOperation.YeastCounting => true
  | _ => false

/-- Theorem stating that shaking the culture solution before yeast counting is the correct operation -/
theorem correct_yeast_counting_operation :
  is_correct_operation ExperimentalOperation.YeastCounting := by
  sorry

end NUMINAMATH_CALUDE_correct_yeast_counting_operation_l880_88001


namespace NUMINAMATH_CALUDE_function_minimum_and_tangent_line_l880_88014

/-- The function f(x) = (x-1)(x-a)^2 -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x - a)^2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := (x - a) * (3*x - a - 2)

theorem function_minimum_and_tangent_line 
  (h₁ : ∃ δ > 0, ∀ x ∈ Set.Ioo (-δ) δ, f (-2) 0 ≤ f (-2) x) :
  (a = -2) ∧ 
  (∃ xp : ℝ, xp ≠ 1 ∧ f' (-2) xp = f' (-2) 1 ∧ 
    (9 : ℝ) * xp - f (-2) xp + 23 = 0) := by
  sorry


end NUMINAMATH_CALUDE_function_minimum_and_tangent_line_l880_88014


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l880_88058

/-- Given vectors a, b, and c in ℝ², prove that if a - 2b is perpendicular to c, 
    then the k-coordinate of c is -3. -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) : 
  a = (Real.sqrt 3, 1) → 
  b = (0, -1) → 
  c.1 = k → 
  c.2 = Real.sqrt 3 → 
  (a.1 - 2 * b.1, a.2 - 2 * b.2) • c = 0 → 
  k = -3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l880_88058


namespace NUMINAMATH_CALUDE_tan_alpha_value_l880_88009

theorem tan_alpha_value (α : Real) (h : 3 * Real.sin α + 4 * Real.cos α = 5) : 
  Real.tan α = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l880_88009


namespace NUMINAMATH_CALUDE_reciprocal_opposite_sum_l880_88041

theorem reciprocal_opposite_sum (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  : 2*c + 2*d - 3*a*b = -3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_opposite_sum_l880_88041


namespace NUMINAMATH_CALUDE_solve_equation_l880_88013

theorem solve_equation : ∃ (A : ℕ), A < 10 ∧ A * 100 + 72 - 23 = 549 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_solve_equation_l880_88013


namespace NUMINAMATH_CALUDE_cubes_volume_ratio_l880_88054

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of cubes that can fit along each dimension -/
def cubesFit (boxDim : ℕ) (cubeDim : ℕ) : ℕ :=
  boxDim / cubeDim

/-- Calculates the volume occupied by cubes in the box -/
def cubesVolume (box : BoxDimensions) (cubeDim : ℕ) : ℕ :=
  let l := cubesFit box.length cubeDim
  let w := cubesFit box.width cubeDim
  let h := cubesFit box.height cubeDim
  l * w * h * (cubeDim ^ 3)

/-- The main theorem to be proved -/
theorem cubes_volume_ratio (box : BoxDimensions) (cubeDim : ℕ) : 
  box.length = 8 → box.width = 6 → box.height = 12 → cubeDim = 4 →
  (cubesVolume box cubeDim : ℚ) / (boxVolume box : ℚ) = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_cubes_volume_ratio_l880_88054


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l880_88084

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 35 →
  n2 = 45 →
  avg1 = 40 →
  avg2 = 60 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 = ((n1 + n2) : ℚ) * (51.25 : ℚ) :=
by
  sorry

#eval (35 * 40 + 45 * 60) / (35 + 45) -- Should evaluate to 51.25

end NUMINAMATH_CALUDE_average_marks_combined_classes_l880_88084


namespace NUMINAMATH_CALUDE_harold_bought_four_coffees_l880_88029

/-- The cost of items bought on two different days --/
structure PurchaseData where
  doughnut_cost : ℚ
  harold_total : ℚ
  harold_doughnuts : ℕ
  melinda_total : ℚ
  melinda_doughnuts : ℕ
  melinda_coffees : ℕ

/-- Calculate the number of coffees Harold bought --/
def calculate_harold_coffees (data : PurchaseData) : ℕ :=
  sorry

/-- Theorem stating that Harold bought 4 coffees --/
theorem harold_bought_four_coffees (data : PurchaseData) 
  (h1 : data.doughnut_cost = 45/100)
  (h2 : data.harold_total = 491/100)
  (h3 : data.harold_doughnuts = 3)
  (h4 : data.melinda_total = 759/100)
  (h5 : data.melinda_doughnuts = 5)
  (h6 : data.melinda_coffees = 6) :
  calculate_harold_coffees data = 4 := by
    sorry

end NUMINAMATH_CALUDE_harold_bought_four_coffees_l880_88029


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l880_88030

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 - 5*x + 1 - 16
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l880_88030


namespace NUMINAMATH_CALUDE_triangle_height_l880_88050

/-- Given a triangle with base 6 and area 24, prove its height is 8 -/
theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 6 → 
  area = 24 → 
  area = 1/2 * base * height → 
  height = 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l880_88050


namespace NUMINAMATH_CALUDE_range_of_a_l880_88051

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) → a > -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l880_88051


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l880_88025

theorem complex_magnitude_problem : 
  let z : ℂ := (1 + 3*I) / (3 - I) - 3*I
  Complex.abs z = 2 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l880_88025


namespace NUMINAMATH_CALUDE_flower_arrangement_count_l880_88057

/-- The number of different pots of flowers --/
def total_pots : ℕ := 7

/-- The number of pots to be selected --/
def selected_pots : ℕ := 5

/-- The number of pots not allowed in the center --/
def restricted_pots : ℕ := 2

/-- The function to calculate the number of arrangements --/
def flower_arrangements (n m k : ℕ) : ℕ := sorry

theorem flower_arrangement_count :
  flower_arrangements total_pots selected_pots restricted_pots = 1800 := by sorry

end NUMINAMATH_CALUDE_flower_arrangement_count_l880_88057


namespace NUMINAMATH_CALUDE_cashier_payment_problem_l880_88080

theorem cashier_payment_problem :
  (¬ ∃ x y : ℤ, 72 * x + 105 * y = 1) ∧
  (∃ x y : ℤ, 72 * x + 105 * y = 3) := by
  sorry

end NUMINAMATH_CALUDE_cashier_payment_problem_l880_88080


namespace NUMINAMATH_CALUDE_kelly_snacks_total_weight_l880_88059

theorem kelly_snacks_total_weight 
  (peanuts_weight : ℝ) 
  (raisins_weight : ℝ) 
  (h1 : peanuts_weight = 0.1)
  (h2 : raisins_weight = 0.4) : 
  peanuts_weight + raisins_weight = 0.5 := by
sorry

end NUMINAMATH_CALUDE_kelly_snacks_total_weight_l880_88059


namespace NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l880_88077

-- Define propositions P and Q
def P (a b : ℝ) : Prop := a^2 + b^2 > 2*a*b
def Q (a b : ℝ) : Prop := |a + b| < |a| + |b|

-- Theorem statement
theorem P_necessary_not_sufficient_for_Q :
  (∀ a b : ℝ, Q a b → P a b) ∧
  (∃ a b : ℝ, P a b ∧ ¬(Q a b)) :=
sorry

end NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l880_88077


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l880_88039

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n, a n > 0) →
  geometric_sequence a q →
  (a 2 - a 1 = a 1 - (1/2) * a 3) →
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l880_88039


namespace NUMINAMATH_CALUDE_valid_triples_l880_88092

def is_valid_triple (p x y : ℕ) : Prop :=
  Nat.Prime p ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  ∃ a : ℕ, x^(p-1) + y = p^a ∧ 
  ∃ b : ℕ, x + y^(p-1) = p^b

def is_valid_triple_for_two (n i : ℕ) : Prop :=
  n > 0 ∧ n < 2^i

theorem valid_triples :
  ∀ p x y : ℕ, is_valid_triple p x y →
    ((p = 3 ∧ ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2))) ∨
     (p = 2 ∧ ∃ n i : ℕ, is_valid_triple_for_two n i ∧ x = n ∧ y = 2^i - n)) :=
sorry

end NUMINAMATH_CALUDE_valid_triples_l880_88092


namespace NUMINAMATH_CALUDE_M_mod_1000_l880_88094

/-- Number of blue flags -/
def blue_flags : ℕ := 12

/-- Number of green flags -/
def green_flags : ℕ := 9

/-- Total number of flags -/
def total_flags : ℕ := blue_flags + green_flags

/-- Number of flagpoles -/
def flagpoles : ℕ := 2

/-- Function to calculate the number of distinguishable arrangements -/
noncomputable def M : ℕ := sorry

/-- Theorem stating the remainder when M is divided by 1000 -/
theorem M_mod_1000 : M % 1000 = 596 := by sorry

end NUMINAMATH_CALUDE_M_mod_1000_l880_88094


namespace NUMINAMATH_CALUDE_least_months_to_triple_l880_88020

def initial_amount : ℝ := 1500
def monthly_interest_rate : ℝ := 0.06

def compound_factor (t : ℕ) : ℝ := (1 + monthly_interest_rate) ^ t

theorem least_months_to_triple :
  ∀ n : ℕ, n < 20 → compound_factor n ≤ 3 ∧
  compound_factor 20 > 3 :=
by sorry

end NUMINAMATH_CALUDE_least_months_to_triple_l880_88020


namespace NUMINAMATH_CALUDE_horner_method_f_neg_two_l880_88037

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem horner_method_f_neg_two :
  f (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_horner_method_f_neg_two_l880_88037


namespace NUMINAMATH_CALUDE_subset_condition_l880_88091

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

-- State the theorem
theorem subset_condition (a : ℝ) : B a ⊆ A ↔ a = 0 ∨ a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l880_88091


namespace NUMINAMATH_CALUDE_second_machine_rate_l880_88076

/-- Represents a copy machine with a constant rate of copies per minute -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Represents a system of two copy machines -/
structure TwoMachineSystem where
  machine1 : CopyMachine
  machine2 : CopyMachine

/-- Calculates the total copies made by a system in a given time -/
def total_copies (system : TwoMachineSystem) (minutes : ℕ) : ℕ :=
  (system.machine1.copies_per_minute + system.machine2.copies_per_minute) * minutes

/-- Theorem: Given the conditions, the second machine makes 65 copies per minute -/
theorem second_machine_rate (system : TwoMachineSystem) :
  system.machine1.copies_per_minute = 35 →
  total_copies system 30 = 3000 →
  system.machine2.copies_per_minute = 65 := by
  sorry

#check second_machine_rate

end NUMINAMATH_CALUDE_second_machine_rate_l880_88076


namespace NUMINAMATH_CALUDE_arrange_objects_count_l880_88075

/-- The number of ways to arrange 7 indistinguishable objects of one type
    and 3 indistinguishable objects of another type in a row of 10 positions -/
def arrangeObjects : ℕ := Nat.choose 10 3

/-- Theorem stating that the number of arrangements is equal to binomial coefficient (10 choose 3) -/
theorem arrange_objects_count : arrangeObjects = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrange_objects_count_l880_88075


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_five_elevenths_l880_88079

theorem sum_of_fractions_equals_five_elevenths :
  (1 / (2^2 - 1) + 1 / (4^2 - 1) + 1 / (6^2 - 1) + 1 / (8^2 - 1) + 1 / (10^2 - 1) : ℚ) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_five_elevenths_l880_88079


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l880_88016

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference
  (a₁ d : ℝ) (h1 : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 7 = 22)
  (h2 : arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 10 = 40) :
  d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l880_88016


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l880_88086

-- Problem 1
theorem problem_1 (x y : ℝ) :
  (x + y) * (x - 2*y) + (x - y)^2 + 3*x * 2*y = 2*x^2 + 3*x*y - y^2 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 1) :
  (x^2 - 4*x + 4) / (x^2 - x) / (x + 1 - 3 / (x - 1)) = (x - 2) / (x * (x + 2)) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l880_88086


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l880_88002

/-- A circle inscribed in a quadrilateral EFGH -/
structure InscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The circle is tangent to EF at R -/
  ER : ℝ
  /-- The circle is tangent to EF at R -/
  RF : ℝ
  /-- The circle is tangent to GH at S -/
  GS : ℝ
  /-- The circle is tangent to GH at S -/
  SH : ℝ

/-- The theorem stating that the square of the radius of the inscribed circle is 1357 -/
theorem inscribed_circle_radius_squared (c : InscribedCircle)
  (h1 : c.ER = 15)
  (h2 : c.RF = 31)
  (h3 : c.GS = 47)
  (h4 : c.SH = 29) :
  c.r ^ 2 = 1357 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l880_88002


namespace NUMINAMATH_CALUDE_fireflies_remaining_joined_fireflies_l880_88012

/-- The number of fireflies remaining after some join and some leave --/
def remaining_fireflies (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Proof that 9 fireflies remain given the initial conditions --/
theorem fireflies_remaining : remaining_fireflies 3 8 2 = 9 := by
  sorry

/-- The number of fireflies that joined is 4 less than a dozen --/
theorem joined_fireflies : (12 : ℕ) - 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fireflies_remaining_joined_fireflies_l880_88012


namespace NUMINAMATH_CALUDE_f_min_value_f_at_3_l880_88021

-- Define the function f
def f (x : ℝ) : ℝ := 7 * x^2 - 28 * x + 2003

-- Theorem for the minimum value of f
theorem f_min_value : ∃ (min : ℝ), min = 1975 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

-- Theorem for the value of f(3)
theorem f_at_3 : f 3 = 1982 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_f_at_3_l880_88021


namespace NUMINAMATH_CALUDE_max_subway_riders_l880_88056

theorem max_subway_riders (total : ℕ) (part_time full_time : ℕ → ℕ) : 
  total = 251 →
  (∀ p f, part_time p + full_time f = total) →
  (∀ p, part_time p ≤ total) →
  (∀ f, full_time f ≤ total) →
  (∀ p, (part_time p) % 11 = 0) →
  (∀ f, (full_time f) % 13 = 0) →
  (∃ max : ℕ, ∀ p f, 
    part_time p + full_time f = total → 
    (part_time p) / 11 + (full_time f) / 13 ≤ max ∧
    (∃ p' f', part_time p' + full_time f' = total ∧ 
              (part_time p') / 11 + (full_time f') / 13 = max)) →
  (∃ p f, part_time p + full_time f = total ∧ 
          (part_time p) / 11 + (full_time f) / 13 = 22) :=
sorry

end NUMINAMATH_CALUDE_max_subway_riders_l880_88056


namespace NUMINAMATH_CALUDE_closed_path_theorem_l880_88038

/-- A closed path on an m×n table satisfying specific conditions -/
structure ClosedPath (m n : ℕ) where
  -- Ensure m and n are at least 4
  m_ge_four : m ≥ 4
  n_ge_four : n ≥ 4
  -- A is the number of straight-forward vertices
  A : ℕ
  -- B is the number of squares with two opposite sides used
  B : ℕ
  -- C is the number of unused squares
  C : ℕ
  -- The path doesn't intersect itself
  no_self_intersection : True
  -- The path passes through all interior vertices
  passes_all_interior : True
  -- The path doesn't pass through outer vertices
  no_outer_vertices : True

/-- Theorem: For a closed path on an m×n table satisfying the given conditions,
    A = B - C + m + n - 1 -/
theorem closed_path_theorem (m n : ℕ) (path : ClosedPath m n) :
  path.A = path.B - path.C + m + n - 1 := by
  sorry

end NUMINAMATH_CALUDE_closed_path_theorem_l880_88038


namespace NUMINAMATH_CALUDE_fraction_equality_l880_88040

theorem fraction_equality : (10^9 : ℚ) / (2 * 5^2 * 10^3) = 20000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l880_88040


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l880_88003

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l880_88003


namespace NUMINAMATH_CALUDE_angle_difference_equality_l880_88064

/-- Represents a triangle with angles A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

/-- Represents the bisection of angle C into C1 and C2 -/
structure BisectedC (t : Triangle) where
  C1 : ℝ
  C2 : ℝ
  sum_C : C1 + C2 = t.C
  positive : 0 < C1 ∧ 0 < C2

theorem angle_difference_equality (t : Triangle) (bc : BisectedC t) 
    (h_A_B : t.A = t.B - 15) 
    (h_C2_adjacent : True) -- This is just a placeholder for the condition that C2 is adjacent to the side opposite B
    : bc.C1 - bc.C2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_angle_difference_equality_l880_88064


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l880_88049

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the property that a_1 and a_19 are roots of x^2 - 10x + 16 = 0
def roots_property (a : ℕ → ℝ) : Prop :=
  a 1 ^ 2 - 10 * a 1 + 16 = 0 ∧ a 19 ^ 2 - 10 * a 19 + 16 = 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  roots_property a →
  a 8 * a 10 * a 12 = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l880_88049


namespace NUMINAMATH_CALUDE_sequence_product_l880_88093

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that b₂(a₂-a₁) = -8 --/
theorem sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-9 : ℝ) < a₁ ∧ a₁ < a₂ ∧ a₂ < (-1 : ℝ)) →  -- arithmetic sequence condition
  (∃ d : ℝ, a₁ = -9 + d ∧ a₂ = a₁ + d ∧ -1 = a₂ + d) →  -- arithmetic sequence definition
  ((-9 : ℝ) < b₁ ∧ b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < (-1 : ℝ)) →  -- geometric sequence condition
  (∃ q : ℝ, b₁ = -9 * q ∧ b₂ = b₁ * q ∧ b₃ = b₂ * q ∧ -1 = b₃ * q) →  -- geometric sequence definition
  b₂ * (a₂ - a₁) = -8 := by
sorry

end NUMINAMATH_CALUDE_sequence_product_l880_88093


namespace NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l880_88048

/-- The time it takes for Maxwell and Brad to meet, given their speeds and the distance between their homes. -/
theorem maxwell_brad_meeting_time 
  (distance : ℝ) 
  (maxwell_speed : ℝ) 
  (brad_speed : ℝ) 
  (head_start : ℝ) 
  (h1 : distance = 54) 
  (h2 : maxwell_speed = 4) 
  (h3 : brad_speed = 6) 
  (h4 : head_start = 1) :
  ∃ (t : ℝ), t + head_start = 6 ∧ 
  maxwell_speed * (t + head_start) + brad_speed * t = distance :=
sorry

end NUMINAMATH_CALUDE_maxwell_brad_meeting_time_l880_88048


namespace NUMINAMATH_CALUDE_unique_valid_n_l880_88066

def is_valid (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 210 ∧ 
  (∀ k ∈ Finset.range 2013, (n + (k + 1).factorial) % 210 = 0)

theorem unique_valid_n : ∃! n : ℕ, is_valid n := by
  sorry

end NUMINAMATH_CALUDE_unique_valid_n_l880_88066


namespace NUMINAMATH_CALUDE_first_interest_rate_is_five_percent_l880_88062

/-- Proves that the first interest rate is 5% given the problem conditions -/
theorem first_interest_rate_is_five_percent 
  (total_amount : ℝ)
  (first_amount : ℝ)
  (second_rate : ℝ)
  (total_income : ℝ)
  (h1 : total_amount = 2600)
  (h2 : first_amount = 1600)
  (h3 : second_rate = 6)
  (h4 : total_income = 140)
  (h5 : ∃ r, (r * first_amount / 100) + (second_rate * (total_amount - first_amount) / 100) = total_income) :
  ∃ r, r = 5 ∧ (r * first_amount / 100) + (second_rate * (total_amount - first_amount) / 100) = total_income :=
by sorry

end NUMINAMATH_CALUDE_first_interest_rate_is_five_percent_l880_88062


namespace NUMINAMATH_CALUDE_quadratic_function_with_equal_roots_l880_88067

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_with_equal_roots 
  (f : ℝ → ℝ) 
  (h1 : QuadraticFunction f)
  (h2 : ∀ x, deriv f x = 2 * x + 2)
  (h3 : ∃! r : ℝ, f r = 0 ∧ (deriv f r = 0)) :
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_with_equal_roots_l880_88067


namespace NUMINAMATH_CALUDE_new_boy_age_l880_88088

theorem new_boy_age (initial_size : Nat) (initial_avg : Nat) (time_passed : Nat) (new_size : Nat) :
  initial_size = 6 →
  initial_avg = 19 →
  time_passed = 3 →
  new_size = 7 →
  (initial_size * initial_avg + initial_size * time_passed + 1) / new_size = initial_avg →
  1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_new_boy_age_l880_88088


namespace NUMINAMATH_CALUDE_simple_interest_difference_l880_88061

/-- Simple interest calculation and comparison with principal -/
theorem simple_interest_difference (principal : ℕ) (rate : ℚ) (time : ℕ) :
  principal = 2500 →
  rate = 4 / 100 →
  time = 5 →
  principal - (principal * rate * time) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_difference_l880_88061


namespace NUMINAMATH_CALUDE_min_radios_sold_l880_88045

/-- Proves the minimum value of n given the radio sales problem conditions -/
theorem min_radios_sold (n d₁ : ℕ) : 
  0 < n → 
  0 < d₁ → 
  d₁ % n = 0 → 
  10 * n - 30 = 80 → 
  ∀ m : ℕ, 0 < m → 10 * m - 30 = 80 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_min_radios_sold_l880_88045


namespace NUMINAMATH_CALUDE_baseball_card_purchase_l880_88043

/-- The cost of the rare baseball card -/
def card_cost : ℕ := 100

/-- Patricia's money -/
def patricia_money : ℕ := 6

/-- Lisa's money in terms of Patricia's -/
def lisa_money : ℕ := 5 * patricia_money

/-- Charlotte's money in terms of Lisa's -/
def charlotte_money : ℕ := lisa_money / 2

/-- The total money they have -/
def total_money : ℕ := patricia_money + lisa_money + charlotte_money

/-- The additional amount needed to buy the card -/
def additional_money_needed : ℕ := card_cost - total_money

theorem baseball_card_purchase :
  additional_money_needed = 49 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_purchase_l880_88043


namespace NUMINAMATH_CALUDE_area_triangle_BXD_l880_88047

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  base_AB : ℝ
  base_CD : ℝ
  area : ℝ

/-- Theorem about the area of triangle BXD in a trapezoid -/
theorem area_triangle_BXD (ABCD : Trapezoid) (h1 : ABCD.base_AB = 24)
    (h2 : ABCD.base_CD = 36) (h3 : ABCD.area = 360) : ℝ := by
  -- The area of triangle BXD is 57.6 square units
  sorry

#check area_triangle_BXD

end NUMINAMATH_CALUDE_area_triangle_BXD_l880_88047


namespace NUMINAMATH_CALUDE_king_crown_payment_l880_88083

/-- Calculates the total amount paid for a crown, including tip -/
def totalAmountPaid (crownCost : ℝ) (tipRate : ℝ) : ℝ :=
  crownCost + (crownCost * tipRate)

/-- Theorem: The king pays $22,000 for a $20,000 crown with a 10% tip -/
theorem king_crown_payment :
  totalAmountPaid 20000 0.1 = 22000 := by
  sorry

end NUMINAMATH_CALUDE_king_crown_payment_l880_88083


namespace NUMINAMATH_CALUDE_astrophysics_budget_decrease_l880_88026

def current_year_allocations : List (String × Rat) :=
  [("Microphotonics", 14/100),
   ("Home Electronics", 24/100),
   ("Food Additives", 15/100),
   ("Genetically Modified Microorganisms", 19/100),
   ("Industrial Lubricants", 8/100)]

def previous_year_allocations : List (String × Rat) :=
  [("Microphotonics", 12/100),
   ("Home Electronics", 22/100),
   ("Food Additives", 13/100),
   ("Genetically Modified Microorganisms", 18/100),
   ("Industrial Lubricants", 7/100)]

def calculate_astrophysics_allocation (allocations : List (String × Rat)) : Rat :=
  1 - (allocations.map (fun x => x.2)).sum

def calculate_percentage_change (old_value : Rat) (new_value : Rat) : Rat :=
  (new_value - old_value) / old_value * 100

theorem astrophysics_budget_decrease :
  let current_astrophysics := calculate_astrophysics_allocation current_year_allocations
  let previous_astrophysics := calculate_astrophysics_allocation previous_year_allocations
  let percentage_change := calculate_percentage_change previous_astrophysics current_astrophysics
  percentage_change = -2857/100 := by sorry

end NUMINAMATH_CALUDE_astrophysics_budget_decrease_l880_88026


namespace NUMINAMATH_CALUDE_arithmetic_correctness_l880_88068

theorem arithmetic_correctness : 
  ((-4) + (-5) = -9) ∧ 
  (4 / (-2) = -2) ∧ 
  (-5 - (-6) ≠ 11) ∧ 
  (-2 * (-10) ≠ -20) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_correctness_l880_88068


namespace NUMINAMATH_CALUDE_closest_to_fraction_l880_88063

def fraction : ℚ := 805 / 0.410

def options : List ℚ := [0.4, 4, 40, 400, 4000]

theorem closest_to_fraction :
  ∃ (x : ℚ), x ∈ options ∧ 
  ∀ (y : ℚ), y ∈ options → |fraction - x| ≤ |fraction - y| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l880_88063


namespace NUMINAMATH_CALUDE_rationalize_and_product_l880_88044

theorem rationalize_and_product : ∃ (A B C : ℤ),
  (((2 : ℝ) + Real.sqrt 5) / ((3 : ℝ) - 2 * Real.sqrt 5) = A + B * Real.sqrt C) ∧
  A * B * C = -560 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_product_l880_88044


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l880_88078

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (X^4 - 1) * (X^2 - 1) = (X^2 + X + 1) * q + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l880_88078


namespace NUMINAMATH_CALUDE_two_x_eq_zero_is_linear_l880_88004

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x = 0 -/
def f (x : ℝ) : ℝ := 2 * x

/-- Theorem: The equation 2x = 0 is a linear equation -/
theorem two_x_eq_zero_is_linear : is_linear_equation f := by
  sorry


end NUMINAMATH_CALUDE_two_x_eq_zero_is_linear_l880_88004


namespace NUMINAMATH_CALUDE_tan_product_thirty_degrees_l880_88018

theorem tan_product_thirty_degrees :
  let A : Real := 30 * π / 180
  let B : Real := 30 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_thirty_degrees_l880_88018


namespace NUMINAMATH_CALUDE_product_sum_fractions_l880_88033

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5 + 1/6) = 57 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l880_88033


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l880_88070

theorem rope_cutting_problem (a b c : ℕ) (ha : a = 63) (hb : b = 84) (hc : c = 105) :
  Nat.gcd a (Nat.gcd b c) = 21 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l880_88070


namespace NUMINAMATH_CALUDE_clara_hughes_book_purchase_l880_88072

theorem clara_hughes_book_purchase :
  let total_volumes : ℕ := 12
  let paperback_cost : ℕ := 18
  let hardcover_cost : ℕ := 28
  let total_spent : ℕ := 276
  ∃ (hardcover_count : ℕ) (paperback_count : ℕ),
    hardcover_count + paperback_count = total_volumes ∧
    hardcover_cost * hardcover_count + paperback_cost * paperback_count = total_spent ∧
    hardcover_count = 6 :=
by sorry

end NUMINAMATH_CALUDE_clara_hughes_book_purchase_l880_88072


namespace NUMINAMATH_CALUDE_digit_sum_divisible_by_11_l880_88073

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Theorem: In any 39 successive natural numbers, at least one has a digit sum divisible by 11 -/
theorem digit_sum_divisible_by_11 (n : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (digitSum (n + k) % 11 = 0) := by sorry

end NUMINAMATH_CALUDE_digit_sum_divisible_by_11_l880_88073


namespace NUMINAMATH_CALUDE_ryegrass_percentage_in_mixture_l880_88055

/-- Theorem: Percentage of ryegrass in a mixture of seed mixtures X and Y -/
theorem ryegrass_percentage_in_mixture (x_ryegrass : ℝ) (y_ryegrass : ℝ) (x_weight : ℝ) :
  x_ryegrass = 0.40 →
  y_ryegrass = 0.25 →
  x_weight = 0.8667 →
  x_ryegrass * x_weight + y_ryegrass * (1 - x_weight) = 0.380005 :=
by sorry

end NUMINAMATH_CALUDE_ryegrass_percentage_in_mixture_l880_88055


namespace NUMINAMATH_CALUDE_factorial_trailing_zeros_l880_88085

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_trailing_zeros : 
  trailing_zeros 238 = 57 ∧ trailing_zeros 238 - trailing_zeros 236 = 0 :=
by sorry

end NUMINAMATH_CALUDE_factorial_trailing_zeros_l880_88085


namespace NUMINAMATH_CALUDE_missing_figure_proof_l880_88060

theorem missing_figure_proof (x : ℝ) : (1.2 / 100) * x = 0.6 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_missing_figure_proof_l880_88060


namespace NUMINAMATH_CALUDE_total_collected_is_4336_5_l880_88011

/-- Represents the total amount collected by Mark during the week in US dollars -/
def total_collected : ℝ :=
  let households_per_day : ℕ := 60
  let days : ℕ := 7
  let total_households : ℕ := households_per_day * days
  let usd_20_percent : ℝ := 0.25
  let eur_15_percent : ℝ := 0.15
  let gbp_10_percent : ℝ := 0.10
  let both_percent : ℝ := 0.05
  let no_donation_percent : ℝ := 0.30
  let usd_20_amount : ℝ := 20
  let eur_15_amount : ℝ := 15
  let gbp_10_amount : ℝ := 10
  let eur_to_usd : ℝ := 1.1
  let gbp_to_usd : ℝ := 1.3

  let usd_20_donation := (usd_20_percent * total_households) * usd_20_amount
  let eur_15_donation := (eur_15_percent * total_households) * eur_15_amount * eur_to_usd
  let gbp_10_donation := (gbp_10_percent * total_households) * gbp_10_amount * gbp_to_usd
  let both_donation := (both_percent * total_households) * (usd_20_amount + eur_15_amount * eur_to_usd)

  usd_20_donation + eur_15_donation + gbp_10_donation + both_donation

theorem total_collected_is_4336_5 :
  total_collected = 4336.5 := by
  sorry

end NUMINAMATH_CALUDE_total_collected_is_4336_5_l880_88011


namespace NUMINAMATH_CALUDE_factorial_15_l880_88095

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_15 : factorial 15 = 1307674368000 := by sorry

end NUMINAMATH_CALUDE_factorial_15_l880_88095


namespace NUMINAMATH_CALUDE_sin_cos_seven_eighths_pi_l880_88024

theorem sin_cos_seven_eighths_pi : 
  Real.sin (7 * π / 8) * Real.cos (7 * π / 8) = - (Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_seven_eighths_pi_l880_88024


namespace NUMINAMATH_CALUDE_curve_line_tangent_l880_88035

/-- The curve y = √(4 - x²) and the line y = m have exactly one common point if and only if m = 2 -/
theorem curve_line_tangent (m : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = Real.sqrt (4 - p.1^2) ∧ p.2 = m) ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_curve_line_tangent_l880_88035


namespace NUMINAMATH_CALUDE_complement_union_theorem_l880_88090

universe u

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 4}

theorem complement_union_theorem :
  (Aᶜ ∩ U) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l880_88090


namespace NUMINAMATH_CALUDE_certain_number_is_36_l880_88069

theorem certain_number_is_36 : ∃ x : ℝ, 
  ((((x + 10) * 2) / 2) - 2) = 88 / 2 ∧ x = 36 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_36_l880_88069


namespace NUMINAMATH_CALUDE_hay_in_final_mixture_l880_88000

/-- Represents the composition of a feed mixture -/
structure FeedMixture where
  oats : ℝ  -- Percentage of oats
  corn : ℝ  -- Percentage of corn
  hay : ℝ   -- Percentage of hay
  mass : ℝ  -- Mass of the mixture in kg

/-- Theorem stating the amount of hay in the final mixture -/
theorem hay_in_final_mixture
  (stepan : FeedMixture)
  (pavel : FeedMixture)
  (final : FeedMixture)
  (h1 : stepan.hay = 40)
  (h2 : pavel.oats = 26)
  (h3 : stepan.corn = pavel.corn)
  (h4 : stepan.mass = 150)
  (h5 : pavel.mass = 250)
  (h6 : final.corn = 30)
  (h7 : final.mass = stepan.mass + pavel.mass)
  (h8 : final.corn * final.mass = stepan.corn * stepan.mass + pavel.corn * pavel.mass) :
  final.hay * final.mass = 170 := by
  sorry

end NUMINAMATH_CALUDE_hay_in_final_mixture_l880_88000


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l880_88006

theorem difference_of_squares_factorization (a b p q : ℝ) : 
  (∃ x y, -a^2 + 9 = (x + y) * (x - y)) ∧ 
  (¬∃ x y, -a^2 - b^2 = (x + y) * (x - y)) ∧ 
  (¬∃ x y, p^2 - (-q^2) = (x + y) * (x - y)) ∧ 
  (¬∃ x y, a^2 - b^3 = (x + y) * (x - y)) :=
by sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l880_88006


namespace NUMINAMATH_CALUDE_mary_travel_time_l880_88097

/-- Represents the time in minutes for Mary's travel process -/
def travel_process (uber_to_house bag_check waiting_for_boarding : ℕ) : ℕ :=
  let uber_to_airport := 5 * uber_to_house
  let security := 3 * bag_check
  let waiting_for_takeoff := 2 * waiting_for_boarding
  uber_to_house + uber_to_airport + bag_check + security + waiting_for_boarding + waiting_for_takeoff

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℕ) : ℚ :=
  minutes / 60

theorem mary_travel_time :
  minutes_to_hours (travel_process 10 15 20) = 3 := by
  sorry

end NUMINAMATH_CALUDE_mary_travel_time_l880_88097


namespace NUMINAMATH_CALUDE_unique_element_implies_a_value_l880_88010

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

-- State the theorem
theorem unique_element_implies_a_value (a : ℝ) :
  (∃! x, x ∈ A a) → a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_element_implies_a_value_l880_88010


namespace NUMINAMATH_CALUDE_min_total_cost_l880_88046

/-- Represents the number of book corners of each size -/
structure BookCorners where
  medium : ℕ
  small : ℕ

/-- Calculates the total cost for a given configuration of book corners -/
def total_cost (corners : BookCorners) : ℕ :=
  860 * corners.medium + 570 * corners.small

/-- Checks if a configuration of book corners is valid according to the given constraints -/
def is_valid_configuration (corners : BookCorners) : Prop :=
  corners.medium + corners.small = 30 ∧
  80 * corners.medium + 30 * corners.small ≤ 1900 ∧
  50 * corners.medium + 60 * corners.small ≤ 1620

/-- Theorem stating that the minimum total cost is 22320 yuan -/
theorem min_total_cost :
  ∃ (corners : BookCorners),
    is_valid_configuration corners ∧
    total_cost corners = 22320 ∧
    ∀ (other : BookCorners), is_valid_configuration other → total_cost other ≥ 22320 := by
  sorry

end NUMINAMATH_CALUDE_min_total_cost_l880_88046


namespace NUMINAMATH_CALUDE_dans_cards_l880_88008

theorem dans_cards (initial : ℕ) (bought : ℕ) (total : ℕ) : 
  initial = 27 → bought = 20 → total = 88 → total - bought - initial = 41 := by
  sorry

end NUMINAMATH_CALUDE_dans_cards_l880_88008


namespace NUMINAMATH_CALUDE_integral_proof_l880_88052

open Real

noncomputable def f (x : ℝ) := (1/2) * log (abs (x^2 + 2 * sin x))

theorem integral_proof (x : ℝ) :
  deriv f x = (x + cos x) / (x^2 + 2 * sin x) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l880_88052
