import Mathlib

namespace obtuse_triangle_third_side_range_l2041_204103

/-- A triangle with side lengths a, b, and c is obtuse if and only if
    one of its squared side lengths is greater than the sum of the squares of the other two side lengths. -/
def IsObtuse (a b c : ℝ) : Prop :=
  a^2 > b^2 + c^2 ∨ b^2 > a^2 + c^2 ∨ c^2 > a^2 + b^2

/-- The range of the third side length x in an obtuse triangle with side lengths 2, 3, and x. -/
theorem obtuse_triangle_third_side_range :
  ∀ x : ℝ, x > 0 →
    (IsObtuse 2 3 x ↔ (1 < x ∧ x < Real.sqrt 5) ∨ (Real.sqrt 13 < x ∧ x < 5)) :=
by sorry

end obtuse_triangle_third_side_range_l2041_204103


namespace smallest_sum_abc_l2041_204157

theorem smallest_sum_abc (a b c : ℕ+) : 
  (∃ x : ℝ, (Real.sin x)^2 + (Real.sin (3*x))^2 + (Real.sin (5*x))^2 + (Real.sin (7*x))^2 = 2.5 ∧
             Real.cos (a.val * x) * Real.cos (b.val * x) * Real.cos (c.val * x) = 0) →
  (∀ a' b' c' : ℕ+, 
    (∃ x : ℝ, (Real.sin x)^2 + (Real.sin (3*x))^2 + (Real.sin (5*x))^2 + (Real.sin (7*x))^2 = 2.5 ∧
               Real.cos (a'.val * x) * Real.cos (b'.val * x) * Real.cos (c'.val * x) = 0) →
    a'.val + b'.val + c'.val ≥ a.val + b.val + c.val) →
  a.val + b.val + c.val = 14 :=
by sorry

end smallest_sum_abc_l2041_204157


namespace units_digit_of_42_pow_5_plus_27_pow_5_l2041_204176

theorem units_digit_of_42_pow_5_plus_27_pow_5 : (42^5 + 27^5) % 10 = 9 := by
  sorry

end units_digit_of_42_pow_5_plus_27_pow_5_l2041_204176


namespace half_angle_quadrant_l2041_204115

-- Define the concept of quadrant
def in_first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2
def in_third_quadrant (θ : ℝ) : Prop := Real.pi < θ ∧ θ < 3 * Real.pi / 2

theorem half_angle_quadrant (α : ℝ) :
  in_first_quadrant α → in_first_quadrant (α / 2) ∨ in_third_quadrant (α / 2) := by
  sorry

end half_angle_quadrant_l2041_204115


namespace sector_central_angle_l2041_204185

/-- Given a sector with radius 6 and area 6π, its central angle measure in degrees is 60. -/
theorem sector_central_angle (radius : ℝ) (area : ℝ) (angle : ℝ) : 
  radius = 6 → area = 6 * Real.pi → angle = (area * 360) / (Real.pi * radius ^ 2) → angle = 60 := by
  sorry

end sector_central_angle_l2041_204185


namespace annulus_area_l2041_204100

theorem annulus_area (R r s : ℝ) (h1 : R > r) (h2 : R^2 - r^2 = s^2) :
  π * s^2 = π * R^2 - π * r^2 := by
  sorry

end annulus_area_l2041_204100


namespace unique_valid_number_l2041_204192

def is_valid_number (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  100 ≤ n ∧ n < 1000 ∧
  tens = hundreds + 3 ∧
  units = tens - 4 ∧
  (hundreds + tens + units) / 2 = tens

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 473 :=
sorry

end unique_valid_number_l2041_204192


namespace choose_3_from_15_l2041_204112

theorem choose_3_from_15 : Nat.choose 15 3 = 455 := by sorry

end choose_3_from_15_l2041_204112


namespace triangle_area_with_sides_17_17_16_prove_triangle_area_with_sides_17_17_16_l2041_204134

/-- The area of a triangle with two sides of length 17 and one side of length 16 is 120 -/
theorem triangle_area_with_sides_17_17_16 : ℝ → Prop :=
  fun area =>
    ∀ (D E F : ℝ × ℝ),
      let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
      let ef := Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
      let df := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
      de = 17 ∧ ef = 17 ∧ df = 16 →
      area = 120

/-- Proof of the theorem -/
theorem prove_triangle_area_with_sides_17_17_16 : triangle_area_with_sides_17_17_16 120 := by
  sorry

end triangle_area_with_sides_17_17_16_prove_triangle_area_with_sides_17_17_16_l2041_204134


namespace f_value_at_one_l2041_204113

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function g : ℝ → ℝ is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem f_value_at_one
  (f g : ℝ → ℝ)
  (h_even : IsEven f)
  (h_odd : IsOdd g)
  (h_eq : ∀ x, f x - g x = x^2 - x + 1) :
  f 1 = 2 := by
  sorry

end f_value_at_one_l2041_204113


namespace quadratic_three_axis_intersections_l2041_204145

/-- A quadratic function f(x) = kx² - 4x - 3 has three common points with the coordinate axes if and only if k > -4/3 and k ≠ 0 -/
theorem quadratic_three_axis_intersections (k : ℝ) :
  (∃ x₁ x₂ y : ℝ, x₁ ≠ x₂ ∧ 
    (k * x₁^2 - 4 * x₁ - 3 = 0) ∧ 
    (k * x₂^2 - 4 * x₂ - 3 = 0) ∧ 
    (k * 0^2 - 4 * 0 - 3 = y)) ↔ 
  (k > -4/3 ∧ k ≠ 0) :=
sorry

end quadratic_three_axis_intersections_l2041_204145


namespace chef_potato_count_l2041_204155

/-- The number of potatoes a chef needs to cook -/
def total_potatoes (cooked : ℕ) (cooking_time_per_potato : ℕ) (remaining_cooking_time : ℕ) : ℕ :=
  cooked + remaining_cooking_time / cooking_time_per_potato

/-- Proof that the chef needs to cook 13 potatoes in total -/
theorem chef_potato_count : total_potatoes 5 6 48 = 13 := by
  sorry

#eval total_potatoes 5 6 48

end chef_potato_count_l2041_204155


namespace range_of_f_l2041_204149

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 4*x - 1

-- Define the domain
def Domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem range_of_f :
  { y | ∃ x ∈ Domain, f x = y } = { y | -6 ≤ y ∧ y ≤ 3 } :=
sorry

end range_of_f_l2041_204149


namespace fraction_of_powers_equals_3125_l2041_204167

theorem fraction_of_powers_equals_3125 : (125000 ^ 5) / (25000 ^ 5) = 3125 := by
  sorry

end fraction_of_powers_equals_3125_l2041_204167


namespace minimum_value_problem_l2041_204191

theorem minimum_value_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧
  (∀ (c d : ℝ), c ≠ 0 → d ≠ 0 →
    c^2 + d^2 + 2 / c^2 + d / c + 1 / d^2 ≥ x^2 + y^2 + 2 / x^2 + y / x + 1 / y^2) ∧
  x^2 + y^2 + 2 / x^2 + y / x + 1 / y^2 = Real.sqrt 7 :=
sorry

end minimum_value_problem_l2041_204191


namespace inequality_solution_set_l2041_204129

theorem inequality_solution_set :
  {x : ℝ | 2 * x^2 + 2 * x - 3 > 7 - x} = {x : ℝ | x < -2 ∨ x > 5/2} := by
  sorry

end inequality_solution_set_l2041_204129


namespace means_inequality_l2041_204172

theorem means_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (a + b + c) / 3 > (a * b * c) ^ (1/3) ∧ 
  (a * b * c) ^ (1/3) > 3 / ((1/a) + (1/b) + (1/c)) := by
  sorry

#check means_inequality

end means_inequality_l2041_204172


namespace negative_integer_square_plus_self_equals_twelve_l2041_204114

theorem negative_integer_square_plus_self_equals_twelve (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N = -4 := by
  sorry

end negative_integer_square_plus_self_equals_twelve_l2041_204114


namespace l₁_passes_through_point_distance_when_parallel_l2041_204126

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a + 2) * x + y + a + 1 = 0
def l₂ (a x y : ℝ) : Prop := 3 * x + a * y - 2 * a = 0

-- Statement 1: l₁ always passes through (-1, 1)
theorem l₁_passes_through_point (a : ℝ) : l₁ a (-1) 1 := by sorry

-- Helper function to check if lines are parallel
def parallel (a : ℝ) : Prop := a + 2 = 3 / a

-- Statement 2: When l₁ and l₂ are parallel, their distance is 2√10/5
theorem distance_when_parallel (a : ℝ) (h : parallel a) :
  ∃ d : ℝ, d = (2 * Real.sqrt 10) / 5 ∧ 
  (∀ x y : ℝ, l₁ a x y ↔ l₂ a (x + d * 3 / 5) (y - d * 4 / 5)) := by sorry

end l₁_passes_through_point_distance_when_parallel_l2041_204126


namespace section_area_theorem_l2041_204146

/-- Represents a regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  lateral_area : ℝ

/-- Represents a plane intersecting the pyramid -/
structure IntersectingPlane where
  opposite_face_area : ℝ

/-- Calculates the lateral surface area of the section cut off by the plane -/
def section_lateral_area (pyramid : RegularQuadPyramid) (plane : IntersectingPlane) : ℝ :=
  sorry

/-- Theorem statement -/
theorem section_area_theorem (pyramid : RegularQuadPyramid) (plane : IntersectingPlane) :
  pyramid.lateral_area = 25 ∧ plane.opposite_face_area = 4 →
  section_lateral_area pyramid plane = 20.25 :=
sorry

end section_area_theorem_l2041_204146


namespace casper_candy_problem_l2041_204106

def candy_sequence (initial : ℕ) : List ℕ :=
  let day1 := initial / 2 - 3
  let day2 := day1 / 2 - 5
  let day3 := day2 / 2 - 2
  let day4 := day3 / 2
  [initial, day1, day2, day3, day4]

theorem casper_candy_problem (initial : ℕ) :
  candy_sequence initial = [initial, initial / 2 - 3, (initial / 2 - 3) / 2 - 5, ((initial / 2 - 3) / 2 - 5) / 2 - 2, 10] →
  initial = 122 := by
  sorry

end casper_candy_problem_l2041_204106


namespace possible_square_values_l2041_204139

/-- Represents a tiling of a 9x7 rectangle using L-trominoes and 2x2 squares. -/
structure Tiling :=
  (num_squares : ℕ)
  (num_trominoes : ℕ)

/-- The area of the rectangle is 63. -/
axiom rectangle_area : 63 = 9 * 7

/-- The area of a 2x2 square is 4. -/
axiom square_area : 4 = 2 * 2

/-- The area of an L-tromino is 3. -/
axiom tromino_area : 3 = 3

/-- The total area covered by tiles equals the rectangle area. -/
axiom area_equation (t : Tiling) : 4 * t.num_squares + 3 * t.num_trominoes = 63

/-- The number of 2x2 squares is a multiple of 3. -/
axiom squares_multiple_of_three (t : Tiling) : ∃ k : ℕ, t.num_squares = 3 * k

/-- The number of 2x2 squares is at most 3. -/
axiom max_squares (t : Tiling) : t.num_squares ≤ 3

/-- The possible values for the number of 2x2 squares are 0 and 3. -/
theorem possible_square_values (t : Tiling) : t.num_squares = 0 ∨ t.num_squares = 3 :=
sorry

end possible_square_values_l2041_204139


namespace infinite_power_tower_four_l2041_204193

/-- The limit of the sequence defined by a_0 = x, a_(n+1) = x^(a_n) --/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := sorry

theorem infinite_power_tower_four (x : ℝ) :
  x > 0 → infinitePowerTower x = 4 → x = Real.sqrt 2 := by sorry

end infinite_power_tower_four_l2041_204193


namespace line_segment_both_symmetric_l2041_204133

-- Define the shapes
inductive Shape
  | EquilateralTriangle
  | IsoscelesTriangle
  | Parallelogram
  | LineSegment

-- Define symmetry properties
def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => False
  | Shape.IsoscelesTriangle => False
  | Shape.Parallelogram => True
  | Shape.LineSegment => True

def isAxiallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => True
  | Shape.IsoscelesTriangle => True
  | Shape.Parallelogram => False
  | Shape.LineSegment => True

-- Theorem statement
theorem line_segment_both_symmetric :
  ∀ s : Shape, (isCentrallySymmetric s ∧ isAxiallySymmetric s) ↔ s = Shape.LineSegment :=
by sorry

end line_segment_both_symmetric_l2041_204133


namespace fibonacci_type_sequence_count_l2041_204104

/-- A Fibonacci-type sequence is an infinite sequence of integers where each term is the sum of the two preceding ones. -/
def FibonacciTypeSequence (a : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, a n = a (n - 1) + a (n - 2)

/-- Count of Fibonacci-type sequences with two consecutive terms strictly positive and ≤ N -/
def countFibonacciTypeSequences (N : ℕ) : ℕ :=
  if N % 2 = 0 then
    (N / 2) * (N / 2 + 1)
  else
    ((N + 1) / 2) ^ 2

theorem fibonacci_type_sequence_count (N : ℕ) :
  (∃ a : ℤ → ℤ, FibonacciTypeSequence a ∧
    ∃ n : ℤ, 0 < a n ∧ 0 < a (n + 1) ∧ a n ≤ N ∧ a (n + 1) ≤ N) →
  countFibonacciTypeSequences N = 
    if N % 2 = 0 then
      (N / 2) * (N / 2 + 1)
    else
      ((N + 1) / 2) ^ 2 :=
by sorry

#check fibonacci_type_sequence_count

end fibonacci_type_sequence_count_l2041_204104


namespace central_cell_value_l2041_204158

def table_sum (a : ℝ) : ℝ :=
  a + 4*a + 16*a + 3*a + 12*a + 48*a + 9*a + 36*a + 144*a

theorem central_cell_value (a : ℝ) (h : table_sum a = 546) : 12 * a = 24 := by
  sorry

end central_cell_value_l2041_204158


namespace num_paths_correct_l2041_204180

/-- The number of paths from (0,0) to (m,n) on Z^2 using only steps of +(1,0) or +(0,1) -/
def num_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem stating that num_paths gives the correct number of paths -/
theorem num_paths_correct (m n : ℕ) : 
  num_paths m n = Nat.choose (m + n) m := by
  sorry

end num_paths_correct_l2041_204180


namespace rectangle_ratio_square_l2041_204111

theorem rectangle_ratio_square (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ a < b) :
  let d := Real.sqrt (a^2 + b^2)
  (a / b = b / d) → (a / b)^2 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end rectangle_ratio_square_l2041_204111


namespace arithmetic_sequence_common_difference_l2041_204173

/-- An arithmetic sequence with given terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_a2 : a 2 = 9)
  (h_a5 : a 5 = 33) :
  ∃ d : ℝ, d = 8 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l2041_204173


namespace two_digit_divisors_of_723_with_remainder_30_l2041_204181

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divides_with_remainder (d q r : ℕ) : Prop := ∃ k, d * k + r = q

theorem two_digit_divisors_of_723_with_remainder_30 :
  ∃! (S : Finset ℕ),
    (∀ n ∈ S, is_two_digit n ∧ divides_with_remainder n 723 30) ∧
    S.card = 4 ∧
    S = {33, 63, 77, 99} :=
by sorry

end two_digit_divisors_of_723_with_remainder_30_l2041_204181


namespace parking_savings_l2041_204163

/-- Calculates the yearly savings when renting a parking space monthly instead of weekly. -/
theorem parking_savings (weekly_rate : ℕ) (monthly_rate : ℕ) : 
  weekly_rate = 10 → monthly_rate = 24 → (52 * weekly_rate) - (12 * monthly_rate) = 232 := by
  sorry

end parking_savings_l2041_204163


namespace triangle_side_difference_range_l2041_204143

theorem triangle_side_difference_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  a = 1 ∧  -- Given condition
  C - B = π / 2 ∧  -- Given condition
  a / Real.sin A = b / Real.sin B ∧  -- Law of Sines
  b / Real.sin B = c / Real.sin C  -- Law of Sines
  → Real.sqrt 2 / 2 < c - b ∧ c - b < 1 := by sorry

end triangle_side_difference_range_l2041_204143


namespace f_properties_l2041_204182

noncomputable def f (x : ℝ) := x * Real.log x

theorem f_properties :
  (∀ x > 0, f x ≥ -1 / Real.exp 1) ∧
  (∀ t > 0, (∀ x ∈ Set.Icc t (t + 2), f x ≥ min (-1 / Real.exp 1) (f t))) ∧
  (∀ x > 0, Real.log x > 1 / (Real.exp x) - 2 / (Real.exp 1 * x)) := by
  sorry

end f_properties_l2041_204182


namespace symmetric_angles_theorem_l2041_204196

-- Define the property of terminal sides being symmetric with respect to x + y = 0
def symmetric_terminal_sides (α β : Real) : Prop := sorry

-- Define the set of angles β
def angle_set : Set Real := {β | ∃ k : Int, β = 2 * k * Real.pi - Real.pi / 6}

-- State the theorem
theorem symmetric_angles_theorem (α β : Real) 
  (h_symmetric : symmetric_terminal_sides α β) 
  (h_alpha : α = -Real.pi / 3) : 
  β ∈ angle_set := by sorry

end symmetric_angles_theorem_l2041_204196


namespace box_office_scientific_notation_l2041_204178

theorem box_office_scientific_notation :
  let billion : ℝ := 10^9
  let box_office : ℝ := 40.25 * billion
  box_office = 4.025 * 10^9 := by sorry

end box_office_scientific_notation_l2041_204178


namespace min_toothpicks_removal_l2041_204107

/-- Represents a geometric figure made of toothpicks forming triangles -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ := sorry

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_removal (figure : ToothpickFigure) 
  (h1 : figure.total_toothpicks = 40)
  (h2 : figure.upward_triangles = 15)
  (h3 : figure.downward_triangles = 10) :
  min_toothpicks_to_remove figure = 15 := by sorry

end min_toothpicks_removal_l2041_204107


namespace ninety_six_configurations_l2041_204177

/-- Represents a configuration of numbers in the grid -/
def Configuration := Fin 6 → Fin 6

/-- Checks if two positions in the grid are adjacent -/
def adjacent (p1 p2 : Fin 6) : Prop :=
  sorry

/-- Checks if a configuration is valid according to the rules -/
def valid_configuration (c : Configuration) : Prop :=
  ∀ p1 p2 : Fin 6, adjacent p1 p2 → abs (c p1 - c p2) ≠ 3

/-- The total number of valid configurations -/
def total_valid_configurations : ℕ :=
  sorry

/-- Main theorem: There are 96 valid configurations -/
theorem ninety_six_configurations : total_valid_configurations = 96 :=
  sorry

end ninety_six_configurations_l2041_204177


namespace equation_solutions_l2041_204159

theorem equation_solutions :
  (∃ (s1 s2 : Set ℝ),
    (s1 = {x : ℝ | (x - 1)^2 - 25 = 0} ∧ s1 = {6, -4}) ∧
    (s2 = {x : ℝ | 3*x*(x - 2) = x - 2} ∧ s2 = {2, 1/3})) := by
  sorry

end equation_solutions_l2041_204159


namespace root_of_f_l2041_204165

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is indeed the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Given condition: f⁻¹(0) = 2
axiom inverse_intersect_y : f_inv 0 = 2

-- Theorem to prove
theorem root_of_f (h : f_inv 0 = 2) : f 2 = 0 := by
  sorry

end root_of_f_l2041_204165


namespace max_visible_faces_sum_l2041_204153

/-- Represents a single die -/
structure Die :=
  (top : ℕ)
  (bottom : ℕ)
  (left : ℕ)
  (right : ℕ)
  (front : ℕ)
  (back : ℕ)

/-- The grid of dice -/
def DiceGrid := Matrix (Fin 10) (Fin 10) Die

/-- Condition: sum of dots on opposite faces is 7 -/
def oppositeFacesSum7 (d : Die) : Prop :=
  d.top + d.bottom = 7 ∧ d.left + d.right = 7 ∧ d.front + d.back = 7

/-- All dice in the grid satisfy the opposite faces sum condition -/
def allDiceSatisfyCondition (grid : DiceGrid) : Prop :=
  ∀ i j, oppositeFacesSum7 (grid i j)

/-- Count of visible faces -/
def visibleFacesCount : ℕ := 240

/-- Sum of dots on visible faces -/
def visibleFacesSum (grid : DiceGrid) : ℕ :=
  sorry  -- Definition would involve summing specific faces based on visibility

/-- Main theorem -/
theorem max_visible_faces_sum (grid : DiceGrid) 
  (h1 : allDiceSatisfyCondition grid) : 
  visibleFacesSum grid ≤ 920 :=
sorry

end max_visible_faces_sum_l2041_204153


namespace fraction_always_nonnegative_l2041_204187

theorem fraction_always_nonnegative (x : ℝ) : (x^2 + 2*x + 1) / (x^2 + 4*x + 8) ≥ 0 := by
  sorry

end fraction_always_nonnegative_l2041_204187


namespace rotated_P_coordinates_l2041_204109

/-- Square with side length 25 -/
def square_side_length : ℝ := 25

/-- Point Q coordinates -/
def Q : ℝ × ℝ := (0, 7)

/-- Point R is on x-axis -/
def R_on_x_axis (R : ℝ × ℝ) : Prop := R.2 = 0

/-- Line equation where S lies after rotation -/
def S_line_equation (x : ℝ) : Prop := x = 39

/-- Rotation of square about R -/
def rotated_square (P R S : ℝ × ℝ) : Prop :=
  R_on_x_axis R ∧ S_line_equation S.1 ∧ S.2 > 0

/-- Theorem: New coordinates of P after rotation -/
theorem rotated_P_coordinates (P R S : ℝ × ℝ) :
  square_side_length = 25 →
  Q = (0, 7) →
  rotated_square P R S →
  P = (19, 35) := by sorry

end rotated_P_coordinates_l2041_204109


namespace sixteen_squares_covered_l2041_204116

/-- Represents a square on the checkerboard -/
structure Square where
  x : Int
  y : Int

/-- Represents the circular disc -/
structure Disc where
  diameter : ℝ
  center : Square

/-- Represents the checkerboard -/
structure Checkerboard where
  size : Nat
  squares : List Square

/-- Checks if a square is completely covered by the disc -/
def is_covered (s : Square) (d : Disc) : Bool :=
  sorry

/-- Counts the number of squares completely covered by the disc -/
def count_covered_squares (cb : Checkerboard) (d : Disc) : Nat :=
  sorry

/-- Main theorem: 16 squares are completely covered -/
theorem sixteen_squares_covered (cb : Checkerboard) (d : Disc) : 
  cb.size = 6 → d.diameter = 2 → count_covered_squares cb d = 16 := by
  sorry

end sixteen_squares_covered_l2041_204116


namespace celias_savings_l2041_204141

def weeks : ℕ := 4
def food_budget_per_week : ℕ := 100
def rent : ℕ := 1500
def streaming_cost : ℕ := 30
def phone_cost : ℕ := 50
def savings_rate : ℚ := 1 / 10

def total_spending : ℕ := weeks * food_budget_per_week + rent + streaming_cost + phone_cost

def savings_amount : ℚ := (total_spending : ℚ) * savings_rate

theorem celias_savings : savings_amount = 198 := by
  sorry

end celias_savings_l2041_204141


namespace choose_14_3_l2041_204117

theorem choose_14_3 : Nat.choose 14 3 = 364 := by
  sorry

end choose_14_3_l2041_204117


namespace cubic_sum_from_elementary_symmetric_polynomials_l2041_204195

theorem cubic_sum_from_elementary_symmetric_polynomials (p q r : ℝ) 
  (h1 : p + q + r = 7)
  (h2 : p * q + p * r + q * r = 8)
  (h3 : p * q * r = -15) :
  p^3 + q^3 + r^3 = 151 := by
sorry

end cubic_sum_from_elementary_symmetric_polynomials_l2041_204195


namespace hyperbola_center_is_3_5_l2041_204121

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 - 24 * x - 25 * y^2 + 250 * y - 489 = 0

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 5)

/-- Theorem: The center of the hyperbola is (3, 5) -/
theorem hyperbola_center_is_3_5 :
  ∀ (x y : ℝ), hyperbola_equation x y →
  ∃ (a b : ℝ), (x - hyperbola_center.1)^2 / a^2 - (y - hyperbola_center.2)^2 / b^2 = 1 :=
by sorry

end hyperbola_center_is_3_5_l2041_204121


namespace initial_salary_correct_l2041_204119

/-- Kirt's initial monthly salary -/
def initial_salary : ℝ := 6000

/-- Kirt's salary increase rate after one year -/
def salary_increase_rate : ℝ := 0.30

/-- Kirt's total earnings after 3 years -/
def total_earnings : ℝ := 259200

/-- Theorem stating that the initial salary satisfies the given conditions -/
theorem initial_salary_correct : 
  12 * initial_salary + 24 * (initial_salary * (1 + salary_increase_rate)) = total_earnings := by
  sorry

#eval initial_salary

end initial_salary_correct_l2041_204119


namespace louise_pencil_boxes_l2041_204179

def pencil_problem (box_capacity : ℕ) (red_pencils : ℕ) (yellow_pencils : ℕ) : Prop :=
  let blue_pencils := 2 * red_pencils
  let green_pencils := red_pencils + blue_pencils
  let total_boxes := 
    (red_pencils + blue_pencils + yellow_pencils + green_pencils) / box_capacity
  total_boxes = 8

theorem louise_pencil_boxes : 
  pencil_problem 20 20 40 :=
sorry

end louise_pencil_boxes_l2041_204179


namespace matrix_commutation_result_l2041_204186

theorem matrix_commutation_result (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A * B = B * A) → (4 * b ≠ c) → ((a - d) / (c - 4 * b) = -3) := by
  sorry

end matrix_commutation_result_l2041_204186


namespace construct_one_degree_angle_l2041_204166

-- Define the given angle
def given_angle : ℕ := 19

-- Define the target angle
def target_angle : ℕ := 1

-- Theorem stating that it's possible to construct the target angle from the given angle
theorem construct_one_degree_angle :
  ∃ n : ℕ, (n * given_angle) % 360 = target_angle :=
sorry

end construct_one_degree_angle_l2041_204166


namespace twenty_one_billion_scientific_notation_l2041_204183

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem twenty_one_billion_scientific_notation :
  toScientificNotation (21000000000 : ℝ) = ScientificNotation.mk 2.1 10 sorry := by
  sorry

end twenty_one_billion_scientific_notation_l2041_204183


namespace hawking_implications_l2041_204124

-- Define the philosophical implications
def unity_of_world_materiality : Prop := true
def thought_existence_identical : Prop := true

-- Define Hawking's statement
def hawking_statement : Prop := true

-- Theorem to prove
theorem hawking_implications :
  hawking_statement → unity_of_world_materiality ∧ thought_existence_identical :=
by
  sorry

end hawking_implications_l2041_204124


namespace right_triangle_shorter_leg_l2041_204125

theorem right_triangle_shorter_leg : 
  ∀ (a b c : ℕ), 
    a^2 + b^2 = c^2 →  -- Pythagorean theorem
    c = 65 →  -- hypotenuse length
    a ≤ b →  -- a is the shorter leg
    a = 16 := by
  sorry

end right_triangle_shorter_leg_l2041_204125


namespace earthworm_investment_theorem_l2041_204144

/-- Represents the earthworm investment scenario -/
structure EarthwormInvestment where
  okeydokey_apples : ℕ
  okeydokey_worms : ℕ
  artichokey_apples : ℕ
  total_worms : ℕ

/-- The earthworm investment theorem -/
theorem earthworm_investment_theorem (e : EarthwormInvestment) 
  (h1 : e.okeydokey_apples = 5)
  (h2 : e.okeydokey_worms = 25)
  (h3 : e.artichokey_apples = 7)
  (h4 : e.okeydokey_worms * e.artichokey_apples = e.okeydokey_apples * (e.total_worms - e.okeydokey_worms)) :
  e.total_worms = 60 := by
  sorry

#check earthworm_investment_theorem

end earthworm_investment_theorem_l2041_204144


namespace f_of_1_plus_g_of_3_l2041_204198

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x - 5
def g (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem f_of_1_plus_g_of_3 : f (1 + g 3) = 10 := by
  sorry

end f_of_1_plus_g_of_3_l2041_204198


namespace radius_of_circle_B_l2041_204136

/-- Given two circles A and B, prove that the radius of B is 10 cm -/
theorem radius_of_circle_B (diameter_A radius_A radius_B : ℝ) : 
  diameter_A = 80 → radius_A = diameter_A / 2 → radius_A = 4 * radius_B → radius_B = 10 := by
  sorry

end radius_of_circle_B_l2041_204136


namespace min_value_theorem_l2041_204135

theorem min_value_theorem (C D x : ℝ) (hC : C > 0) (hD : D > 0) (hx : x > 0)
  (h1 : x^2 + 1/x^2 = C) (h2 : x + 1/x = D) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 + 3/2 ∧ ∀ y, y = C/(D-2) → y ≥ m :=
sorry

end min_value_theorem_l2041_204135


namespace train_length_l2041_204188

theorem train_length (t_platform : ℝ) (t_pole : ℝ) (l_platform : ℝ) 
  (h1 : t_platform = 36)
  (h2 : t_pole = 18)
  (h3 : l_platform = 300) :
  ∃ l_train : ℝ, l_train = 300 ∧ l_train / t_pole = (l_train + l_platform) / t_platform :=
by
  sorry

end train_length_l2041_204188


namespace largest_integer_solution_l2041_204130

theorem largest_integer_solution (x : ℤ) : (3 - 2 * x > 0) → x ≤ 1 ∧ (∀ y : ℤ, 3 - 2 * y > 0 → y ≤ x) :=
by sorry

end largest_integer_solution_l2041_204130


namespace f_at_five_l2041_204137

def f (x : ℝ) : ℝ := 3 * x^4 - 22 * x^3 + 51 * x^2 - 58 * x + 24

theorem f_at_five : f 5 = 134 := by sorry

end f_at_five_l2041_204137


namespace planet_coloring_theorem_specific_planet_coloring_case_l2041_204171

/-- The number of colors needed for planet coloring -/
def colors_needed (num_planets : ℕ) (num_people : ℕ) : ℕ :=
  num_planets * num_people

/-- Theorem: In the planet coloring scenario, the number of colors needed
    is equal to the number of planets multiplied by the number of people coloring. -/
theorem planet_coloring_theorem (num_planets : ℕ) (num_people : ℕ) :
  colors_needed num_planets num_people = num_planets * num_people :=
by
  sorry

/-- The specific case mentioned in the problem -/
theorem specific_planet_coloring_case :
  colors_needed 8 3 = 24 :=
by
  sorry

end planet_coloring_theorem_specific_planet_coloring_case_l2041_204171


namespace work_completion_time_l2041_204140

theorem work_completion_time (b : ℝ) (c : ℝ) (d : ℝ) (h1 : b = 14) (h2 : c = 2) (h3 : d = 5.000000000000001) : 
  ∃ a : ℝ, a = 4 ∧ 
  (c * (1 / a + 1 / b) + d * (1 / b) = 1) :=
sorry

end work_completion_time_l2041_204140


namespace room_width_l2041_204102

/-- 
Given a rectangular room with length 20 feet and 1 foot longer than its width,
prove that the width of the room is 19 feet.
-/
theorem room_width (length width : ℕ) : 
  length = 20 ∧ length = width + 1 → width = 19 := by
  sorry

end room_width_l2041_204102


namespace arithmetic_sequence_second_term_l2041_204108

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  first : ℤ
  /-- The common difference between consecutive terms -/
  diff : ℤ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + (n - 1) * seq.diff

theorem arithmetic_sequence_second_term
  (seq : ArithmeticSequence)
  (h16 : seq.nthTerm 16 = 8)
  (h17 : seq.nthTerm 17 = 10) :
  seq.nthTerm 2 = -20 := by
  sorry

#check arithmetic_sequence_second_term

end arithmetic_sequence_second_term_l2041_204108


namespace base_8_calculation_l2041_204142

/-- Addition in base 8 -/
def add_base_8 (a b : ℕ) : ℕ := sorry

/-- Subtraction in base 8 -/
def sub_base_8 (a b : ℕ) : ℕ := sorry

/-- Convert a natural number to its base 8 representation -/
def to_base_8 (n : ℕ) : ℕ := sorry

/-- Convert a base 8 number to its decimal representation -/
def from_base_8 (n : ℕ) : ℕ := sorry

theorem base_8_calculation : 
  sub_base_8 (add_base_8 (from_base_8 452) (from_base_8 167)) (from_base_8 53) = from_base_8 570 := by
  sorry

end base_8_calculation_l2041_204142


namespace pencil_theorem_l2041_204174

def pencil_problem (jayden marcus dana ella : ℕ) : Prop :=
  jayden = 20 ∧
  dana = jayden + 15 ∧
  jayden = 2 * marcus ∧
  ella = 3 * marcus - 5 ∧
  dana = marcus + ella

theorem pencil_theorem :
  ∀ jayden marcus dana ella : ℕ,
  pencil_problem jayden marcus dana ella →
  dana = marcus + ella :=
by
  sorry

end pencil_theorem_l2041_204174


namespace distribute_five_into_three_l2041_204151

/-- The number of ways to distribute n distinct objects into k distinct bins,
    where each bin must contain at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  (Nat.choose n k + (Nat.choose n 2 * Nat.choose 3 2) / 2) * Nat.factorial k

/-- The theorem stating that distributing 5 distinct objects into 3 distinct bins,
    where each bin must contain at least one object, results in 150 ways. -/
theorem distribute_five_into_three :
  distribute 5 3 = 150 := by sorry

end distribute_five_into_three_l2041_204151


namespace men_absent_l2041_204170

/-- Proves that 15 men became absent given the original group size, planned completion time, and actual completion time. -/
theorem men_absent (total_men : ℕ) (planned_days : ℕ) (actual_days : ℕ) 
  (h1 : total_men = 180) 
  (h2 : planned_days = 55)
  (h3 : actual_days = 60) :
  ∃ (absent_men : ℕ), 
    absent_men = 15 ∧ 
    (total_men * planned_days = (total_men - absent_men) * actual_days) :=
by
  sorry

#check men_absent

end men_absent_l2041_204170


namespace prop_p_true_prop_q_false_prop_2_true_prop_3_true_l2041_204162

-- Define propositions p and q
def p : Prop := ∀ x y : ℝ, x > y → -x < -y
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Theorem statements
theorem prop_p_true : p := by sorry

theorem prop_q_false : ¬q := by sorry

theorem prop_2_true : p ∨ q := by sorry

theorem prop_3_true : p ∧ ¬q := by sorry

end prop_p_true_prop_q_false_prop_2_true_prop_3_true_l2041_204162


namespace intersection_point_with_median_line_l2041_204190

open Complex

/-- Given complex numbers and a curve, prove the intersection point with the median line -/
theorem intersection_point_with_median_line 
  (a b c : ℝ) 
  (z₁₁ : ℂ) 
  (z₁ : ℂ) 
  (z₂ : ℂ) 
  (h_z₁₁ : z₁₁ = Complex.I * a) 
  (h_z₁ : z₁ = (1/2 : ℝ) + Complex.I * b) 
  (h_z₂ : z₂ = 1 + Complex.I * c) 
  (h_non_collinear : a + c ≠ 2 * b) 
  (z : ℝ → ℂ) 
  (h_z : ∀ t, z t = z₁ * (cos t)^4 + 2 * z₁ * (cos t)^2 * (sin t)^2 + z₂ * (sin t)^4) :
  ∃! p : ℂ, p ∈ Set.range z ∧ 
    p.re = (1/2 : ℝ) ∧ 
    p.im = (a + c + 2*b) / 4 := by
  sorry

end intersection_point_with_median_line_l2041_204190


namespace max_c_value_l2041_204169

noncomputable section

def is_valid_solution (a b c x y z : ℝ) : Prop :=
  a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  a^x + b^y + c^z = 4 ∧
  x * a^x + y * b^y + z * c^z = 6 ∧
  x^2 * a^x + y^2 * b^y + z^2 * c^z = 9

theorem max_c_value (a b c x y z : ℝ) (h : is_valid_solution a b c x y z) :
  c ≤ Real.rpow 4 (1/3) :=
sorry

end

end max_c_value_l2041_204169


namespace regular_decagon_interior_angle_regular_decagon_interior_angle_proof_l2041_204164

/-- The measure of an interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle : ℝ :=
  let n : ℕ := 10  -- number of sides in a decagon
  let total_interior_angle_sum : ℝ := (n - 2) * 180
  let interior_angle : ℝ := total_interior_angle_sum / n
  144

/-- Proof of the theorem -/
theorem regular_decagon_interior_angle_proof :
  regular_decagon_interior_angle = 144 := by
  sorry

end regular_decagon_interior_angle_regular_decagon_interior_angle_proof_l2041_204164


namespace ball_selection_ways_l2041_204175

/-- Represents the number of ways to select balls from a bag -/
def select_balls (total white red black : ℕ) (select : ℕ) 
  (white_min white_max red_min black_max : ℕ) : ℕ := sorry

/-- Theorem stating the number of ways to select balls under given conditions -/
theorem ball_selection_ways : 
  select_balls 20 9 5 6 10 2 8 2 3 = 16 := by sorry

end ball_selection_ways_l2041_204175


namespace cow_count_is_24_l2041_204154

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (a : AnimalCount) : ℕ := 2 * a.ducks + 4 * a.cows

/-- The total number of heads in the group -/
def totalHeads (a : AnimalCount) : ℕ := a.ducks + a.cows

/-- The condition given in the problem -/
def satisfiesCondition (a : AnimalCount) : Prop :=
  totalLegs a = 2 * totalHeads a + 48

theorem cow_count_is_24 (a : AnimalCount) (h : satisfiesCondition a) : a.cows = 24 := by
  sorry

end cow_count_is_24_l2041_204154


namespace midpoint_coordinates_l2041_204150

/-- Given a segment with endpoints A(x₁, y₁) and B(x₂, y₂), and its midpoint M(x₀, y₀),
    prove that the coordinates of the midpoint are the averages of the endpoints' coordinates. -/
theorem midpoint_coordinates (x₀ x₁ x₂ y₀ y₁ y₂ : ℝ) :
  (∀ t : ℝ, t ∈ (Set.Icc 0 1) → 
    (x₀ = (1 - t) * x₁ + t * x₂ ∧ 
     y₀ = (1 - t) * y₁ + t * y₂)) →
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2 := by
  sorry


end midpoint_coordinates_l2041_204150


namespace base_eight_solution_l2041_204147

/-- Converts a list of digits in base h to its decimal representation -/
def to_decimal (digits : List Nat) (h : Nat) : Nat :=
  digits.foldl (fun acc d => acc * h + d) 0

/-- Checks if the equation holds for a given base h -/
def equation_holds (h : Nat) : Prop :=
  to_decimal [9, 8, 7, 6, 5, 4] h + to_decimal [6, 9, 8, 5, 5, 5] h = to_decimal [1, 7, 9, 6, 2, 2, 9] h

theorem base_eight_solution :
  ∃ (h : Nat), h > 0 ∧ equation_holds h ∧ ∀ (k : Nat), k > 0 ∧ equation_holds k → k = h :=
by
  sorry

end base_eight_solution_l2041_204147


namespace car_profit_percent_l2041_204156

/-- Calculate the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent (purchase_price repair_cost selling_price : ℚ) : 
  purchase_price = 48000 →
  repair_cost = 14000 →
  selling_price = 72900 →
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = 1758/100 := by
sorry

end car_profit_percent_l2041_204156


namespace correct_average_weight_l2041_204127

theorem correct_average_weight 
  (num_boys : ℕ) 
  (initial_avg : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) : 
  num_boys = 20 → 
  initial_avg = 58.4 → 
  misread_weight = 56 → 
  correct_weight = 65 → 
  (num_boys * initial_avg + correct_weight - misread_weight) / num_boys = 58.85 := by
  sorry

end correct_average_weight_l2041_204127


namespace number_wall_x_value_l2041_204161

/-- Represents a simplified number wall with given conditions --/
structure NumberWall where
  x : ℤ
  y : ℤ
  -- Define the wall structure based on given conditions
  bottom_row : Vector ℤ 5 := ⟨[x, 7, y, 14, 9], rfl⟩
  second_row_right : Vector ℤ 2 := ⟨[y + 14, 23], rfl⟩
  third_row_right : ℤ := 37
  top : ℤ := 80

/-- The main theorem stating that x must be 12 in the given number wall --/
theorem number_wall_x_value (wall : NumberWall) : wall.x = 12 := by
  sorry

end number_wall_x_value_l2041_204161


namespace cylinder_volume_equals_cube_surface_l2041_204105

/-- The volume of a cylinder with surface area equal to a cube of side length 4 and height equal to its diameter --/
theorem cylinder_volume_equals_cube_surface (π : ℝ) (h : π > 0) : 
  ∃ (r : ℝ), r > 0 ∧ 
  6 * π * r^2 = 96 ∧ 
  π * r^2 * (2 * r) = 128 * Real.sqrt 2 / π :=
sorry

end cylinder_volume_equals_cube_surface_l2041_204105


namespace line_perp_parallel_implies_planes_perp_l2041_204122

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity between a line and a plane -/
def Line3D.perp (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between a line and a plane -/
def Line3D.parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicularity between two planes -/
def Plane3D.perp (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is perpendicular to one plane and parallel to another, 
    then the two planes are perpendicular to each other -/
theorem line_perp_parallel_implies_planes_perp 
  (l : Line3D) (α β : Plane3D) (h1 : l.perp α) (h2 : l.parallel β) : 
  α.perp β :=
sorry

end line_perp_parallel_implies_planes_perp_l2041_204122


namespace min_fraction_sum_l2041_204110

def ValidDigits : Finset Nat := {1, 3, 4, 5, 6, 8, 9}

theorem min_fraction_sum (A B C D : Nat) 
  (hA : A ∈ ValidDigits) (hB : B ∈ ValidDigits) 
  (hC : C ∈ ValidDigits) (hD : D ∈ ValidDigits)
  (hAB : A ≠ B) (hAC : A ≠ C) (hAD : A ≠ D) 
  (hBC : B ≠ C) (hBD : B ≠ D) (hCD : C ≠ D)
  (hB_pos : B > 0) (hD_pos : D > 0) :
  (A : ℚ) / B + (C : ℚ) / D ≥ 11 / 24 := by
  sorry

end min_fraction_sum_l2041_204110


namespace simplify_expression_l2041_204123

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : a^4 + b^4 = a + b) (h2 : a^2 + b^2 = 2) :
  a^2 / b^2 + b^2 / a^2 - 1 / (a^2 * b^2) = 1 := by
  sorry

end simplify_expression_l2041_204123


namespace laura_minimum_score_l2041_204132

def minimum_score (score1 score2 score3 : ℝ) (required_average : ℝ) : ℝ :=
  4 * required_average - (score1 + score2 + score3)

theorem laura_minimum_score :
  minimum_score 80 78 76 85 = 106 := by sorry

end laura_minimum_score_l2041_204132


namespace estimate_wildlife_population_l2041_204131

/-- Estimate the total number of animals in a wildlife reserve using the mark-recapture method. -/
theorem estimate_wildlife_population
  (initial_catch : ℕ)
  (second_catch : ℕ)
  (marked_in_second : ℕ)
  (h1 : initial_catch = 1200)
  (h2 : second_catch = 1000)
  (h3 : marked_in_second = 100) :
  (initial_catch * second_catch) / marked_in_second = 12000 :=
by sorry

end estimate_wildlife_population_l2041_204131


namespace largest_n_for_equation_l2041_204184

theorem largest_n_for_equation : 
  (∃ (n : ℕ+), ∀ (m : ℕ+), 
    (∃ (x y z : ℕ+), m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 8) → 
    m ≤ n) ∧ 
  (∃ (x y z : ℕ+), (10 : ℕ+)^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 8) :=
sorry

end largest_n_for_equation_l2041_204184


namespace corn_profit_problem_l2041_204194

theorem corn_profit_problem (seeds_per_ear : ℕ) (ear_price : ℚ) (bag_price : ℚ) (seeds_per_bag : ℕ) (total_profit : ℚ) :
  seeds_per_ear = 4 →
  ear_price = 1/10 →
  bag_price = 1/2 →
  seeds_per_bag = 100 →
  total_profit = 40 →
  (total_profit / (ear_price - (bag_price / seeds_per_bag) * seeds_per_ear) : ℚ) = 500 := by
  sorry

end corn_profit_problem_l2041_204194


namespace unique_prime_perfect_square_l2041_204168

theorem unique_prime_perfect_square :
  ∀ p : ℕ, Prime p → (∃ q : ℕ, 5^p + 4*p^4 = q^2) → p = 5 :=
by sorry

end unique_prime_perfect_square_l2041_204168


namespace unknowns_and_variables_l2041_204189

-- Define a type for equations
structure Equation where
  f : ℝ → ℝ → ℝ
  c : ℝ

-- Define a type for systems of equations
structure SystemOfEquations where
  eq1 : Equation
  eq2 : Equation

-- Define a type for single equations
structure SingleEquation where
  eq : Equation

-- Define a property for being an unknown
def isUnknown (x : ℝ) (y : ℝ) (system : SystemOfEquations) : Prop :=
  ∀ (sol_x sol_y : ℝ), system.eq1.f sol_x sol_y = system.eq1.c ∧ 
                        system.eq2.f sol_x sol_y = system.eq2.c →
                        x = sol_x ∧ y = sol_y

-- Define a property for being a variable
def isVariable (x : ℝ) (y : ℝ) (single : SingleEquation) : Prop :=
  ∀ (val_x : ℝ), ∃ (val_y : ℝ), single.eq.f val_x val_y = single.eq.c

-- Theorem statement
theorem unknowns_and_variables 
  (x y : ℝ) (system : SystemOfEquations) (single : SingleEquation) : 
  (isUnknown x y system) ∧ (isVariable x y single) := by
  sorry

end unknowns_and_variables_l2041_204189


namespace wildlife_population_estimate_l2041_204160

theorem wildlife_population_estimate 
  (tagged_released : ℕ) 
  (later_captured : ℕ) 
  (tagged_in_sample : ℕ) 
  (h1 : tagged_released = 1200)
  (h2 : later_captured = 1000)
  (h3 : tagged_in_sample = 100) :
  (tagged_released * later_captured) / tagged_in_sample = 12000 :=
by sorry

end wildlife_population_estimate_l2041_204160


namespace exists_n_for_digit_sum_ratio_l2041_204138

/-- S(a) denotes the sum of the digits of the natural number a -/
def digit_sum (a : ℕ) : ℕ := sorry

/-- Theorem stating that for any natural number R, there exists a natural number n 
    such that the ratio of the digit sum of n^2 to the digit sum of n equals R -/
theorem exists_n_for_digit_sum_ratio (R : ℕ) : 
  ∃ n : ℕ, (digit_sum (n^2) : ℚ) / (digit_sum n : ℚ) = R := by sorry

end exists_n_for_digit_sum_ratio_l2041_204138


namespace product_after_digit_reversal_mistake_l2041_204128

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Checks if a number is prime -/
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, 1 < m → m < p → ¬(p % m = 0)

theorem product_after_digit_reversal_mistake (a b : ℕ) :
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  isPrime b →           -- b is prime
  reverseDigits a * b = 280 →  -- product after mistake is 280
  a * b = 28 :=         -- correct product is 28
by sorry

end product_after_digit_reversal_mistake_l2041_204128


namespace simplified_expression_evaluation_l2041_204120

theorem simplified_expression_evaluation (x y : ℝ) 
  (hx : x = -1) (hy : y = 1/2) : 
  2 * (3 * x^2 + x * y^2) - 3 * (2 * x * y^2 - x^2) - 10 * x^2 = 0 := by
  sorry

end simplified_expression_evaluation_l2041_204120


namespace six_six_six_triangle_l2041_204148

/-- Triangle Inequality Theorem: A set of three positive real numbers can form a triangle
    if and only if the sum of any two is greater than the third. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set (6, 6, 6) can form a triangle. -/
theorem six_six_six_triangle : can_form_triangle 6 6 6 := by
  sorry


end six_six_six_triangle_l2041_204148


namespace well_problem_solution_l2041_204152

/-- The depth of the well and the rope lengths of five families -/
def well_problem (e : ℚ) : Prop :=
  ∃ (x a b c d : ℚ),
    -- Depth equations
    x = 2*a + b ∧
    x = 3*b + c ∧
    x = 4*c + d ∧
    x = 5*d + e ∧
    x = 6*e + a ∧
    -- Solutions
    x = (721/76)*e ∧
    a = (265/76)*e ∧
    b = (191/76)*e ∧
    c = (37/19)*e ∧
    d = (129/76)*e

/-- The well depth and rope lengths satisfy the given conditions -/
theorem well_problem_solution :
  ∀ e : ℚ, well_problem e :=
by sorry

end well_problem_solution_l2041_204152


namespace expand_product_l2041_204197

theorem expand_product (x : ℝ) : (2 * x + 3) * (x - 4) = 2 * x^2 - 5 * x - 12 := by
  sorry

end expand_product_l2041_204197


namespace units_digit_of_seven_power_l2041_204101

theorem units_digit_of_seven_power (n : ℕ) : 7^(6^5) ≡ 1 [ZMOD 10] := by
  sorry

end units_digit_of_seven_power_l2041_204101


namespace isosceles_triangle_base_length_l2041_204199

/-- Given an isosceles triangle with base angle α and difference b between
    the radii of its circumscribed and inscribed circles, 
    the length of its base side is (2b * sin(2α)) / (1 - tan²(α/2)) -/
theorem isosceles_triangle_base_length 
  (α : ℝ) 
  (b : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < b) : 
  ∃ (x : ℝ), x = (2 * b * Real.sin (2 * α)) / (1 - Real.tan (α / 2) ^ 2) ∧ 
  x > 0 ∧ 
  ∃ (R r : ℝ), R > 0 ∧ r > 0 ∧ R - r = b ∧
  R = x / (2 * Real.sin (2 * α)) ∧
  r = x / 2 * Real.tan (α / 2) :=
sorry

end isosceles_triangle_base_length_l2041_204199


namespace tooth_fairy_payment_l2041_204118

theorem tooth_fairy_payment (total_money : ℕ) (teeth_count : ℕ) (h1 : total_money = 54) (h2 : teeth_count = 18) :
  total_money / teeth_count = 3 := by
sorry

end tooth_fairy_payment_l2041_204118
