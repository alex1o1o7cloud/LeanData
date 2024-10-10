import Mathlib

namespace unique_number_with_specific_divisors_l1906_190679

theorem unique_number_with_specific_divisors : ∃! (N : ℕ),
  (5 ∣ N) ∧ (49 ∣ N) ∧ (Finset.card (Nat.divisors N) = 10) :=
by
  -- The proof goes here
  sorry

end unique_number_with_specific_divisors_l1906_190679


namespace unique_divisor_with_remainder_l1906_190698

theorem unique_divisor_with_remainder (n : ℕ) (r : ℕ) : 
  (∃! u : ℕ, u > r ∧ n % u = r) ↔ n = 11 ∧ r = 4 := by
  sorry

end unique_divisor_with_remainder_l1906_190698


namespace shortest_altitude_right_triangle_l1906_190602

theorem shortest_altitude_right_triangle (a b c h : ℝ) : 
  a = 8 ∧ b = 15 ∧ c = 17 →
  a^2 + b^2 = c^2 →
  h * c = 2 * (1/2 * a * b) →
  h = 120/17 := by
sorry

end shortest_altitude_right_triangle_l1906_190602


namespace three_planes_divide_space_l1906_190688

-- Define a plane in 3D space
def Plane : Type := ℝ × ℝ × ℝ → Prop

-- Define a function to check if three planes intersect pairwise
def intersect_pairwise (p1 p2 p3 : Plane) : Prop := sorry

-- Define a function to check if three lines are mutually parallel
def mutually_parallel_intersections (p1 p2 p3 : Plane) : Prop := sorry

-- Define a function to count the number of parts the space is divided into
def count_parts (p1 p2 p3 : Plane) : ℕ := sorry

-- Theorem statement
theorem three_planes_divide_space :
  ∀ (p1 p2 p3 : Plane),
    intersect_pairwise p1 p2 p3 →
    mutually_parallel_intersections p1 p2 p3 →
    count_parts p1 p2 p3 = 7 :=
by sorry

end three_planes_divide_space_l1906_190688


namespace flour_calculation_l1906_190677

/-- Given a sugar to flour ratio and an amount of sugar, calculate the required amount of flour -/
def flour_amount (sugar_flour_ratio : ℚ) (sugar_amount : ℚ) : ℚ :=
  sugar_amount / sugar_flour_ratio

theorem flour_calculation (sugar_amount : ℚ) :
  sugar_amount = 50 →
  flour_amount (10 / 1) sugar_amount = 5 := by
sorry

end flour_calculation_l1906_190677


namespace parabola_translation_theorem_l1906_190656

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c - v }

theorem parabola_translation_theorem :
  let original := Parabola.mk 1 0 (-2)
  let translated := translate_parabola original 3 1
  translated = Parabola.mk 1 (-6) (-3) := by sorry

end parabola_translation_theorem_l1906_190656


namespace min_distance_line_circle_min_distance_specific_line_circle_l1906_190627

/-- Given a line and a circle in a 2D plane, this theorem states that 
    the minimum distance between any point on the line and any point on the circle 
    is equal to the difference between the distance from the circle's center 
    to the line and the radius of the circle. -/
theorem min_distance_line_circle (a b c d e f : ℝ) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let circle := {p : ℝ × ℝ | (p.1 - d)^2 + (p.2 - e)^2 = f^2}
  let center := (d, e)
  let radius := f
  let dist_center_to_line := |a * d + b * e + c| / Real.sqrt (a^2 + b^2)
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧
    ∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ line → q' ∈ circle →
      dist_center_to_line - radius ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) :=
by sorry

/-- The minimum distance between a point on the line 2x + y - 6 = 0
    and a point on the circle (x-1)² + (y+2)² = 5 is √5/5. -/
theorem min_distance_specific_line_circle :
  let line := {p : ℝ × ℝ | 2 * p.1 + p.2 - 6 = 0}
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 2)^2 = 5}
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧
    ∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ line → q' ∈ circle →
      Real.sqrt 5 / 5 ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) :=
by sorry

end min_distance_line_circle_min_distance_specific_line_circle_l1906_190627


namespace y_is_odd_square_l1906_190606

def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * x (n + 1) - 2 * x n

def y (n : ℕ) : ℤ := x n ^ 2 + 2 ^ (n + 2)

theorem y_is_odd_square (n : ℕ) (h : n > 0) :
  ∃ k : ℤ, Odd k ∧ y n = k ^ 2 := by
  sorry

end y_is_odd_square_l1906_190606


namespace sum_odd_numbers_eq_square_last_term_eq_2n_minus_1_sum_odd_numbers_40_times_3_eq_4800_l1906_190617

/-- The sum of the first n odd numbers -/
def sum_odd_numbers (n : ℕ) : ℕ := (Finset.range n).sum (fun i => 2 * i + 1)

theorem sum_odd_numbers_eq_square (n : ℕ) : sum_odd_numbers n = n^2 :=
  by sorry

theorem last_term_eq_2n_minus_1 (n : ℕ) : 2 * n - 1 = sum_odd_numbers n - sum_odd_numbers (n - 1) :=
  by sorry

theorem sum_odd_numbers_40_times_3_eq_4800 : 3 * sum_odd_numbers 40 = 4800 :=
  by sorry

end sum_odd_numbers_eq_square_last_term_eq_2n_minus_1_sum_odd_numbers_40_times_3_eq_4800_l1906_190617


namespace parabola_and_line_l1906_190675

/-- A parabola with focus F and point A on it -/
structure Parabola where
  p : ℝ
  y₀ : ℝ
  h_p_pos : p > 0
  h_on_parabola : y₀^2 = 2 * p * 2
  h_focus_dist : (2 - p/2)^2 + y₀^2 = 4^2

/-- A line intersecting the parabola -/
structure IntersectingLine (par : Parabola) where
  m : ℝ
  h_not_origin : m ≠ 0
  h_two_points : ∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    (x₁^2 + (2*m - 8)*x₁ + m^2 = 0) ∧ 
    (x₂^2 + (2*m - 8)*x₂ + m^2 = 0)
  h_perpendicular : ∃ x₁ x₂ y₁ y₂, 
    x₁ ≠ x₂ ∧ 
    y₁ = x₁ + m ∧ 
    y₂ = x₂ + m ∧ 
    x₁*x₂ + y₁*y₂ = 0

/-- The main theorem -/
theorem parabola_and_line (par : Parabola) (l : IntersectingLine par) :
  par.p = 4 ∧ l.m = -8 := by sorry

end parabola_and_line_l1906_190675


namespace sum_of_fractions_l1906_190697

theorem sum_of_fractions : 2 / 5 + 3 / 8 = 31 / 40 := by
  sorry

end sum_of_fractions_l1906_190697


namespace situps_problem_l1906_190604

def total_situps (diana_rate : ℕ) (diana_total : ℕ) (hani_extra : ℕ) : ℕ :=
  let diana_time := diana_total / diana_rate
  let hani_rate := diana_rate + hani_extra
  let hani_total := hani_rate * diana_time
  diana_total + hani_total

theorem situps_problem :
  total_situps 4 40 3 = 110 :=
by sorry

end situps_problem_l1906_190604


namespace division_problem_l1906_190614

theorem division_problem (divisor : ℕ) : 
  22 = divisor * 7 + 1 → divisor = 3 := by
  sorry

end division_problem_l1906_190614


namespace paint_house_time_l1906_190620

/-- Represents the time taken to paint a house given the number of workers and their efficiency -/
def paintTime (workers : ℕ) (efficiency : ℚ) (time : ℚ) : Prop :=
  (workers : ℚ) * efficiency * time = 40

theorem paint_house_time :
  paintTime 5 (4/5) 8 → paintTime 4 (4/5) 10 := by sorry

end paint_house_time_l1906_190620


namespace milk_cost_proof_l1906_190671

def total_cost : ℕ := 42
def banana_cost : ℕ := 12
def bread_cost : ℕ := 9
def apple_cost : ℕ := 14

theorem milk_cost_proof :
  total_cost - (banana_cost + bread_cost + apple_cost) = 7 := by
sorry

end milk_cost_proof_l1906_190671


namespace product_difference_equality_l1906_190685

theorem product_difference_equality : 2012.25 * 2013.75 - 2010.25 * 2015.75 = 7 := by
  sorry

end product_difference_equality_l1906_190685


namespace expression_equality_l1906_190622

theorem expression_equality : 
  Real.sqrt 25 - Real.sqrt 3 + |Real.sqrt 3 - 2| + ((-8 : ℝ) ^ (1/3)) = 5 - 2 * Real.sqrt 3 := by
  sorry

end expression_equality_l1906_190622


namespace octagonal_pyramid_sum_l1906_190642

-- Define the structure of an octagonal pyramid
structure OctagonalPyramid where
  base_vertices : Nat
  base_edges : Nat
  triangular_faces : Nat
  apex_vertex : Nat
  apex_edges : Nat

-- Define the properties of an octagonal pyramid
def is_octagonal_pyramid (p : OctagonalPyramid) : Prop :=
  p.base_vertices = 8 ∧
  p.base_edges = 8 ∧
  p.triangular_faces = 8 ∧
  p.apex_vertex = 1 ∧
  p.apex_edges = 8

-- Calculate the total number of faces
def total_faces (p : OctagonalPyramid) : Nat :=
  1 + p.triangular_faces

-- Calculate the total number of edges
def total_edges (p : OctagonalPyramid) : Nat :=
  p.base_edges + p.apex_edges

-- Calculate the total number of vertices
def total_vertices (p : OctagonalPyramid) : Nat :=
  p.base_vertices + p.apex_vertex

-- Theorem: The sum of faces, edges, and vertices of an octagonal pyramid is 34
theorem octagonal_pyramid_sum (p : OctagonalPyramid) 
  (h : is_octagonal_pyramid p) : 
  total_faces p + total_edges p + total_vertices p = 34 := by
  sorry

end octagonal_pyramid_sum_l1906_190642


namespace carwash_problem_l1906_190695

/-- Represents the number of vehicles of each type washed --/
structure VehicleCounts where
  cars : ℕ
  trucks : ℕ
  suvs : ℕ

/-- Represents the prices for washing each type of vehicle --/
structure WashPrices where
  car : ℕ
  truck : ℕ
  suv : ℕ

/-- Calculates the total amount raised from a car wash --/
def totalRaised (counts : VehicleCounts) (prices : WashPrices) : ℕ :=
  counts.cars * prices.car + counts.trucks * prices.truck + counts.suvs * prices.suv

/-- The main theorem to prove --/
theorem carwash_problem (prices : WashPrices) 
    (h_car_price : prices.car = 5)
    (h_truck_price : prices.truck = 6)
    (h_suv_price : prices.suv = 7)
    (h_total : totalRaised { cars := 7, trucks := 5, suvs := 5 } prices = 100) :
  ∃ (n : ℕ), n = 7 ∧ 
    totalRaised { cars := n, trucks := 5, suvs := 5 } prices = 100 :=
by
  sorry


end carwash_problem_l1906_190695


namespace course_selection_ways_l1906_190699

theorem course_selection_ways (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 6 → k = 3 → m = 1 →
  (n.choose m) * ((n - m).choose (k - m)) * ((k).choose m) = 180 :=
by sorry

end course_selection_ways_l1906_190699


namespace simplify_expression_l1906_190603

theorem simplify_expression :
  ∃ (d e f : ℕ+), 
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ f.val)) ∧
    (((Real.sqrt 3 - 1) ^ (2 - Real.sqrt 2)) / ((Real.sqrt 3 + 1) ^ (2 + Real.sqrt 2)) = d.val - e.val * Real.sqrt f.val) ∧
    (d.val = 14 ∧ e.val = 8 ∧ f.val = 3) :=
by sorry

end simplify_expression_l1906_190603


namespace functional_equation_solution_l1906_190635

/-- A function satisfying g(xy) = xg(y) for all real numbers x and y -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) = x * g y

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h1 : FunctionalEquation g) (h2 : g 1 = 30) : 
  g 50 = 1500 ∧ g 0.5 = 15 := by
  sorry

end functional_equation_solution_l1906_190635


namespace product_equality_l1906_190610

theorem product_equality (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end product_equality_l1906_190610


namespace equation_roots_imply_m_range_l1906_190628

theorem equation_roots_imply_m_range (m : ℝ) : 
  (∃ x : ℝ, x^2 + 4*m*x + 4*m^2 + 2*m + 3 = 0 ∨ x^2 + (2*m + 1)*x + m^2 = 0) → 
  (m ≤ -3/2 ∨ m ≥ -1/4) := by
sorry

end equation_roots_imply_m_range_l1906_190628


namespace line_equation_proof_l1906_190643

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point (x, y) lies on the line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The line we're interested in -/
def ourLine : Line :=
  { slope := 2, yIntercept := 5 }

theorem line_equation_proof :
  (∀ x y : ℝ, ourLine.containsPoint x y ↔ -2 * x + y = 1) ∧
  ourLine.containsPoint (-2) 3 :=
sorry

end line_equation_proof_l1906_190643


namespace equation_equivalence_l1906_190672

theorem equation_equivalence (x : ℚ) : (x - 1) / 2 - x / 5 = 1 ↔ 5 * (x - 1) - 2 * x = 10 := by
  sorry

end equation_equivalence_l1906_190672


namespace cos_shift_equals_sin_shift_l1906_190615

theorem cos_shift_equals_sin_shift (x : ℝ) : 
  Real.cos (x + π/3) = Real.sin (x + 5*π/6) := by
  sorry

end cos_shift_equals_sin_shift_l1906_190615


namespace solution_comparison_l1906_190629

theorem solution_comparison (a a' b b' : ℝ) 
  (ha : a > 0) (ha' : a' > 0) 
  (heq1 : ∃ x, 2 * a * x + b = 0) 
  (heq2 : ∃ x', 2 * a' * x' + b' = 0) 
  (hineq : (- b / (2 * a))^2 > (- b' / (2 * a'))^2) : 
  b^2 / a^2 > b'^2 / a'^2 := by
  sorry

end solution_comparison_l1906_190629


namespace smallest_integer_y_smallest_integer_solution_l1906_190660

theorem smallest_integer_y (y : ℤ) : (8 - 3 * y < 26) ↔ (-5 ≤ y) := by sorry

theorem smallest_integer_solution : ∃ (y : ℤ), (8 - 3 * y < 26) ∧ (∀ (z : ℤ), z < y → 8 - 3 * z ≥ 26) := by sorry

end smallest_integer_y_smallest_integer_solution_l1906_190660


namespace books_per_shelf_l1906_190650

theorem books_per_shelf (total_books : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : total_books = 72)
  (h2 : mystery_shelves = 3)
  (h3 : picture_shelves = 5)
  (h4 : ∃ x : ℕ, total_books = x * (mystery_shelves + picture_shelves)) :
  ∃ x : ℕ, x = 9 ∧ total_books = x * (mystery_shelves + picture_shelves) :=
by sorry

end books_per_shelf_l1906_190650


namespace two_digit_number_divisible_by_8_12_18_l1906_190600

theorem two_digit_number_divisible_by_8_12_18 :
  ∃! n : ℕ, 60 ≤ n ∧ n ≤ 79 ∧ 8 ∣ n ∧ 12 ∣ n ∧ 18 ∣ n := by
  sorry

end two_digit_number_divisible_by_8_12_18_l1906_190600


namespace exists_bijection_sum_inverse_neg_l1906_190690

theorem exists_bijection_sum_inverse_neg : 
  ∃ (f : ℝ → ℝ), Function.Bijective f ∧ ∀ x : ℝ, f x + (Function.invFun f) x = -x := by
  sorry

end exists_bijection_sum_inverse_neg_l1906_190690


namespace circle_equation_correct_l1906_190662

/-- The standard equation of a circle with center (a, b) and radius r -/
def CircleEquation (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

/-- Theorem: The equation (x - 2)^2 + (y + 1)^2 = 4 represents a circle with center (2, -1) and radius 2 -/
theorem circle_equation_correct :
  ∀ x y : ℝ, CircleEquation x y 2 (-1) 2 ↔ (x - 2)^2 + (y + 1)^2 = 4 := by
  sorry

end circle_equation_correct_l1906_190662


namespace sqrt_simplification_l1906_190644

theorem sqrt_simplification :
  (Real.sqrt 18 - Real.sqrt 2) / Real.sqrt 2 = 2 := by
  sorry

end sqrt_simplification_l1906_190644


namespace largest_three_digit_number_satisfying_condition_l1906_190682

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Calculates the sum of digits of a ThreeDigitNumber -/
def ThreeDigitNumber.digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- Checks if a ThreeDigitNumber satisfies the given condition -/
def ThreeDigitNumber.satisfiesCondition (n : ThreeDigitNumber) : Prop :=
  n.toNat = n.digitSum + (2 * n.digitSum)^2

theorem largest_three_digit_number_satisfying_condition :
  ∃ (n : ThreeDigitNumber), n.toNat = 915 ∧
    n.satisfiesCondition ∧
    ∀ (m : ThreeDigitNumber), m.satisfiesCondition → m.toNat ≤ n.toNat :=
  sorry

end largest_three_digit_number_satisfying_condition_l1906_190682


namespace triangle_theorem_l1906_190619

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  t.c * Real.sin t.B + 2 * Real.cos t.A = t.b * Real.sin t.C + 1

def condition2 (t : Triangle) : Prop :=
  Real.cos (2 * t.A) - 3 * Real.cos (t.B + t.C) - 1 = 0

def condition3 (t : Triangle) : Prop :=
  ∃ k : ℝ, k * Real.sqrt 3 * t.b = t.a * Real.sin t.B ∧ k * t.a = Real.cos t.A

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h : condition1 t ∨ condition2 t ∨ condition3 t) : 
  t.A = Real.pi / 3 ∧ 
  (t.a * t.b * Real.sin t.C / 2 = Real.sqrt 3 / 2 → t.a ≥ Real.sqrt 2) :=
sorry

end triangle_theorem_l1906_190619


namespace sum_of_y_values_l1906_190611

/-- Given 5 sets of data points, prove the sum of y values -/
theorem sum_of_y_values 
  (x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (h_sum_x : x₁ + x₂ + x₃ + x₄ + x₅ = 150) 
  (h_regression : ∀ x, (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = x → 
    (y₁ + y₂ + y₃ + y₄ + y₅) / 5 = 0.67 * x + 24.9) : 
  y₁ + y₂ + y₃ + y₄ + y₅ = 225 := by
  sorry

end sum_of_y_values_l1906_190611


namespace mauras_remaining_seashells_l1906_190664

/-- The number of seashells Maura found -/
def total_seashells : ℕ := 75

/-- The number of seashells Maura gave to her sister -/
def given_seashells : ℕ := 18

/-- The number of days Maura's family stays at the beach house -/
def beach_days : ℕ := 21

/-- Theorem stating that Maura has 57 seashells left -/
theorem mauras_remaining_seashells :
  total_seashells - given_seashells = 57 := by sorry

end mauras_remaining_seashells_l1906_190664


namespace exam_score_proof_l1906_190609

theorem exam_score_proof (score1 score2 score3 score4 : ℕ) : 
  score1 = 70 → score2 = 80 → score3 = 90 → 
  (score1 + score2 + score3 + score4) / 4 = 70 → 
  score4 = 40 := by
sorry

end exam_score_proof_l1906_190609


namespace triangle_side_area_relation_l1906_190605

/-- Given a triangle with altitudes m₁, m₂, m₃ to sides a, b, c respectively,
    prove the relation between sides and area. -/
theorem triangle_side_area_relation (m₁ m₂ m₃ a b c S : ℝ) 
  (h₁ : m₁ = 20)
  (h₂ : m₂ = 24)
  (h₃ : m₃ = 30)
  (ha : S = a * m₁ / 2)
  (hb : S = b * m₂ / 2)
  (hc : S = c * m₃ / 2) :
  (a / b = 6 / 5 ∧ b / c = 5 / 4) ∧ S = 10 * a ∧ S = 12 * b ∧ S = 15 * c := by
  sorry

end triangle_side_area_relation_l1906_190605


namespace max_value_of_squares_l1906_190625

theorem max_value_of_squares (x y z : ℤ) 
  (eq1 : x * y + x + y = 20)
  (eq2 : y * z + y + z = 6)
  (eq3 : x * z + x + z = 2) :
  ∃ (a b c : ℤ), a * b + a + b = 20 ∧ b * c + b + c = 6 ∧ a * c + a + c = 2 ∧
    ∀ (x y z : ℤ), x * y + x + y = 20 → y * z + y + z = 6 → x * z + x + z = 2 →
      x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 = 84 :=
sorry

end max_value_of_squares_l1906_190625


namespace sum_of_complex_numbers_l1906_190634

theorem sum_of_complex_numbers :
  let z₁ : ℂ := Complex.mk 2 5
  let z₂ : ℂ := Complex.mk 3 (-7)
  z₁ + z₂ = Complex.mk 5 (-2) :=
by sorry

end sum_of_complex_numbers_l1906_190634


namespace floor_equation_solution_l1906_190636

theorem floor_equation_solution (a : ℝ) : 
  (∀ n : ℕ, 4 * ⌊a * n⌋ = n + ⌊a * ⌊a * n⌋⌋) ↔ a = 2 + Real.sqrt 3 :=
sorry

end floor_equation_solution_l1906_190636


namespace certain_number_exists_l1906_190601

theorem certain_number_exists : ∃ (n : ℕ), n > 0 ∧ 49 % n = 4 ∧ 66 % n = 6 ∧ n = 15 := by
  sorry

end certain_number_exists_l1906_190601


namespace f_2015_equals_2_l1906_190646

/-- A function satisfying the given conditions -/
def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 6) + f x = 0) ∧
  (∀ x : ℝ, f (x - 1) = f (3 - x)) ∧
  (f 1 = -2)

/-- Theorem stating that any function satisfying the conditions has f(2015) = 2 -/
theorem f_2015_equals_2 (f : ℝ → ℝ) (hf : f_conditions f) : f 2015 = 2 := by
  sorry

end f_2015_equals_2_l1906_190646


namespace algebra_class_size_l1906_190624

/-- Given an Algebra 1 class where there are 11 girls and 5 fewer girls than boys,
    prove that the total number of students in the class is 27. -/
theorem algebra_class_size :
  ∀ (num_girls num_boys : ℕ),
    num_girls = 11 →
    num_girls + 5 = num_boys →
    num_girls + num_boys = 27 :=
by
  sorry

end algebra_class_size_l1906_190624


namespace hall_covering_expenditure_l1906_190694

/-- Calculates the total expenditure for covering the interior of a rectangular hall with a mat -/
def calculate_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  let floor_area := length * width
  let wall_area := 2 * (length * height + width * height)
  let total_area := 2 * floor_area + wall_area
  total_area * cost_per_sqm

/-- Proves that the expenditure for covering a specific hall with a mat is 19000 -/
theorem hall_covering_expenditure :
  calculate_expenditure 20 15 5 20 = 19000 := by
  sorry

end hall_covering_expenditure_l1906_190694


namespace second_x_intercept_of_quadratic_l1906_190696

/-- Given a quadratic function with vertex at (5, -3) and one x-intercept at (1, 0),
    the x-coordinate of the second x-intercept is 9. -/
theorem second_x_intercept_of_quadratic 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 1 = 0) 
  (h3 : ∃ y, f 5 = y ∧ y = -3) : 
  ∃ x, f x = 0 ∧ x = 9 := by
sorry

end second_x_intercept_of_quadratic_l1906_190696


namespace f_has_root_in_interval_l1906_190653

-- Define the function f(x) = x^3 + 4x - 3
def f (x : ℝ) := x^3 + 4*x - 3

-- State the theorem
theorem f_has_root_in_interval :
  ∃ c ∈ Set.Icc 0 1, f c = 0 :=
by
  -- Proof goes here
  sorry

end f_has_root_in_interval_l1906_190653


namespace hash_property_l1906_190667

/-- Definition of operation # for non-negative integers -/
def hash (a b : ℕ) : ℕ := 100 + 4 * b^2 + 8 * a * b

/-- Theorem stating the properties of the hash operation -/
theorem hash_property (a b : ℕ) : 
  hash a b = 100 ∧ a + b = 5 → hash a b = 100 + 4 * b^2 + 8 * a * b := by
  sorry

end hash_property_l1906_190667


namespace cheryls_mms_l1906_190689

/-- Cheryl's M&M's Problem -/
theorem cheryls_mms (initial : ℕ) (after_lunch : ℕ) (after_dinner : ℕ) (given_to_sister : ℕ) :
  initial = 25 →
  after_lunch = 7 →
  after_dinner = 5 →
  given_to_sister = initial - (after_lunch + after_dinner) →
  given_to_sister = 13 := by
sorry

end cheryls_mms_l1906_190689


namespace correct_initial_amounts_l1906_190631

/-- Represents the initial amounts of money John and Richard had --/
structure InitialMoney where
  john : ℚ
  richard : ℚ

/-- Represents the final amounts of money John and Richard had after transactions --/
structure FinalMoney where
  john : ℚ
  richard : ℚ

/-- Calculates the final money based on initial money and described transactions --/
def calculateFinalMoney (initial : InitialMoney) : FinalMoney :=
  { john := initial.john - (initial.richard + initial.john),
    richard := 2 * initial.richard + 2 * initial.john }

/-- Theorem stating the correct initial amounts given the final amounts --/
theorem correct_initial_amounts :
  ∃ (initial : InitialMoney),
    let final := calculateFinalMoney initial
    final.john = 7/2 ∧ final.richard = 3 ∧
    initial.john = 5/2 ∧ initial.richard = 3/2 := by
  sorry

end correct_initial_amounts_l1906_190631


namespace M_subset_N_l1906_190659

def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

theorem M_subset_N : M ⊆ N := by sorry

end M_subset_N_l1906_190659


namespace koschei_coins_theorem_l1906_190612

theorem koschei_coins_theorem :
  ∃! n : ℕ, 300 ≤ n ∧ n ≤ 400 ∧ n % 10 = 7 ∧ n % 12 = 9 :=
by sorry

end koschei_coins_theorem_l1906_190612


namespace concert_probability_at_least_seven_concert_probability_is_one_ninth_l1906_190683

/-- The probability that at least 7 out of 8 people stay for an entire concert,
    given that 4 are certain to stay and 4 have a 1/3 probability of staying. -/
theorem concert_probability_at_least_seven (total_people : Nat) (certain_people : Nat)
    (uncertain_people : Nat) (stay_prob : ℚ) : ℚ :=
  let total_people := 8
  let certain_people := 4
  let uncertain_people := 4
  let stay_prob := 1 / 3
  1 / 9

theorem concert_probability_is_one_ninth :
    concert_probability_at_least_seven 8 4 4 (1 / 3) = 1 / 9 := by
  sorry

end concert_probability_at_least_seven_concert_probability_is_one_ninth_l1906_190683


namespace mango_price_reduction_l1906_190665

/-- Represents the price reduction problem for mangoes --/
theorem mango_price_reduction 
  (original_price : ℝ) 
  (original_quantity : ℕ) 
  (total_spent : ℝ) 
  (additional_mangoes : ℕ) :
  original_price = 416.67 →
  original_quantity = 125 →
  total_spent = 360 →
  additional_mangoes = 12 →
  let original_price_per_mango := original_price / original_quantity
  let original_bought_quantity := total_spent / original_price_per_mango
  let new_quantity := original_bought_quantity + additional_mangoes
  let new_price_per_mango := total_spent / new_quantity
  let price_reduction_percentage := (original_price_per_mango - new_price_per_mango) / original_price_per_mango * 100
  price_reduction_percentage = 10 := by
sorry


end mango_price_reduction_l1906_190665


namespace copper_in_mixture_l1906_190684

theorem copper_in_mixture (lead_percentage : Real) (copper_percentage : Real) (lead_mass : Real) (copper_mass : Real) : 
  lead_percentage = 0.25 →
  copper_percentage = 0.60 →
  lead_mass = 5 →
  copper_mass = 12 →
  copper_mass = (copper_percentage / lead_percentage) * lead_mass :=
by
  sorry

#check copper_in_mixture

end copper_in_mixture_l1906_190684


namespace reciprocal_of_negative_sin_60_l1906_190674

theorem reciprocal_of_negative_sin_60 :
  ((-Real.sin (π / 3))⁻¹) = -(2 * Real.sqrt 3) / 3 := by sorry

end reciprocal_of_negative_sin_60_l1906_190674


namespace plywood_cut_theorem_l1906_190607

theorem plywood_cut_theorem :
  ∃ (a b c d : Set (ℝ × ℝ)),
    -- The original square has area 625 cm²
    (∀ (x y : ℝ), (x, y) ∈ a ∪ b ∪ c ∪ d → 0 ≤ x ∧ x ≤ 25 ∧ 0 ≤ y ∧ y ≤ 25) ∧
    -- The four parts are disjoint
    (a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ a ∩ d = ∅ ∧ b ∩ c = ∅ ∧ b ∩ d = ∅ ∧ c ∩ d = ∅) ∧
    -- The four parts cover the entire original square
    (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 25 ∧ 0 ≤ y ∧ y ≤ 25 → (x, y) ∈ a ∪ b ∪ c ∪ d) ∧
    -- The parts can be rearranged into two squares
    (∃ (s₁ s₂ : Set (ℝ × ℝ)),
      -- First square has side length 24 cm
      (∀ (x y : ℝ), (x, y) ∈ s₁ → 0 ≤ x ∧ x ≤ 24 ∧ 0 ≤ y ∧ y ≤ 24) ∧
      -- Second square has side length 7 cm
      (∀ (x y : ℝ), (x, y) ∈ s₂ → 0 ≤ x ∧ x ≤ 7 ∧ 0 ≤ y ∧ y ≤ 7) ∧
      -- The rearranged squares cover the same area as the original parts
      (∀ (x y : ℝ), (x, y) ∈ s₁ ∪ s₂ ↔ (x, y) ∈ a ∪ b ∪ c ∪ d)) :=
by
  sorry


end plywood_cut_theorem_l1906_190607


namespace prime_iff_divisibility_condition_l1906_190676

theorem prime_iff_divisibility_condition (n : ℕ) (h : n ≥ 2) :
  Prime n ↔ ∀ d : ℕ, d > 1 → d ∣ n → (d^2 + n) ∣ (n^2 + d) :=
sorry

end prime_iff_divisibility_condition_l1906_190676


namespace problem_statement_l1906_190666

theorem problem_statement (x y z a b c : ℝ) 
  (h1 : x * y = 2 * a) 
  (h2 : x * z = 3 * b) 
  (h3 : y * z = 4 * c) 
  (h4 : x ≠ 0) 
  (h5 : y ≠ 0) 
  (h6 : z ≠ 0) 
  (h7 : a ≠ 0) 
  (h8 : b ≠ 0) 
  (h9 : c ≠ 0) : 
  2 * x^2 + 3 * y^2 + 4 * z^2 = 12 * b / a + 8 * c / b + 6 * a / c := by
  sorry

end problem_statement_l1906_190666


namespace billion_to_scientific_notation_l1906_190618

theorem billion_to_scientific_notation :
  let billion : ℝ := 10^9
  8.26 * billion = 8.26 * 10^9 := by
  sorry

end billion_to_scientific_notation_l1906_190618


namespace pants_cost_l1906_190680

theorem pants_cost (total_cost : ℕ) (tshirt_cost : ℕ) (num_tshirts : ℕ) (num_pants : ℕ) :
  total_cost = 1500 →
  tshirt_cost = 100 →
  num_tshirts = 5 →
  num_pants = 4 →
  (total_cost - num_tshirts * tshirt_cost) / num_pants = 250 := by
  sorry

end pants_cost_l1906_190680


namespace sum_of_exponents_outside_radical_l1906_190637

-- Define the expression
def original_expression (a b c : ℝ) : ℝ := (24 * a^4 * b^6 * c^11) ^ (1/3)

-- Define the simplified expression
def simplified_expression (a b c : ℝ) : ℝ := 2 * a * b^2 * c^3 * ((3 * a * c^2) ^ (1/3))

-- State the theorem
theorem sum_of_exponents_outside_radical :
  ∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 →
  original_expression a b c = simplified_expression a b c ∧
  (1 + 2 + 3 = 6) := by sorry

end sum_of_exponents_outside_radical_l1906_190637


namespace regular_polygon_with_120_degree_interior_angle_l1906_190673

theorem regular_polygon_with_120_degree_interior_angle :
  ∀ n : ℕ, n ≥ 3 →
  (180 * (n - 2) / n : ℚ) = 120 →
  n = 6 :=
by
  sorry

end regular_polygon_with_120_degree_interior_angle_l1906_190673


namespace line_tangent_to_parabola_l1906_190649

/-- The line 2x + 4y + m = 0 is tangent to the parabola y^2 = 16x if and only if m = 32 -/
theorem line_tangent_to_parabola (m : ℝ) :
  (∀ x y : ℝ, 2 * x + 4 * y + m = 0 → y^2 = 16 * x) ∧
  (∃! p : ℝ × ℝ, 2 * p.1 + 4 * p.2 + m = 0 ∧ p.2^2 = 16 * p.1) ↔
  m = 32 := by sorry

end line_tangent_to_parabola_l1906_190649


namespace shortest_halving_segment_345_triangle_l1906_190657

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The shortest segment that halves the area of a triangle -/
def shortestHalvingSegment (t : Triangle) : ℝ :=
  sorry

/-- Theorem: The shortest segment that halves the area of a 3-4-5 triangle has length 2 -/
theorem shortest_halving_segment_345_triangle :
  let t : Triangle := { a := 3, b := 4, c := 5 }
  shortestHalvingSegment t = 2 := by
  sorry

end shortest_halving_segment_345_triangle_l1906_190657


namespace seven_by_six_grid_paths_l1906_190632

/-- The number of paths on a grid from bottom-left to top-right -/
def grid_paths (width height : ℕ) : ℕ :=
  Nat.choose (width + height) height

theorem seven_by_six_grid_paths :
  grid_paths 7 6 = 1716 := by
  sorry

end seven_by_six_grid_paths_l1906_190632


namespace jerrys_action_figures_l1906_190687

theorem jerrys_action_figures (initial_figures : ℕ) : 
  initial_figures + 4 - 1 = 6 → initial_figures = 3 :=
by
  sorry

end jerrys_action_figures_l1906_190687


namespace cowbell_coloring_l1906_190681

theorem cowbell_coloring (n : ℕ) (hn : n ≥ 3) :
  ∃ (m : ℕ), m = n + 1 ∧
  (∀ (k : ℕ), k > m → 
    ∃ (f : ℕ → Fin n), 
      ∀ (i : ℕ), (∀ (c : Fin n), ∃ (j : ℕ), j < n + 1 ∧ f ((i + j) % k) = c)) ∧
  (¬ ∃ (f : ℕ → Fin n), 
    ∀ (i : ℕ), (∀ (c : Fin n), ∃ (j : ℕ), j < n + 1 ∧ f ((i + j) % m) = c)) :=
by sorry

end cowbell_coloring_l1906_190681


namespace kangaroo_problem_l1906_190670

/-- Represents the number of exchanges required to sort kangaroos -/
def kangaroo_exchanges (total : ℕ) (right_facing : ℕ) (left_facing : ℕ) : ℕ :=
  (right_facing.min 3) * left_facing + (right_facing - 3).max 0 * (left_facing - 2).max 0

/-- Theorem stating that for 10 kangaroos with 6 facing right and 4 facing left, 
    the number of exchanges is 18 -/
theorem kangaroo_problem : 
  kangaroo_exchanges 10 6 4 = 18 := by sorry

end kangaroo_problem_l1906_190670


namespace increased_amount_proof_l1906_190693

theorem increased_amount_proof (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : x = 3) :
  x + y = 60 * (1 / x) → y = 17 := by
  sorry

end increased_amount_proof_l1906_190693


namespace product_of_numbers_l1906_190678

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 205) : x * y = 42 := by
  sorry

end product_of_numbers_l1906_190678


namespace artist_profit_calculation_l1906_190647

/-- Calculates the total profit for an artist given contest winnings, painting sales, and expenses. -/
theorem artist_profit_calculation 
  (contest_prize : ℕ) 
  (num_paintings_sold : ℕ) 
  (price_per_painting : ℕ) 
  (art_supplies_cost : ℕ) 
  (exhibition_fee : ℕ) 
  (h1 : contest_prize = 150)
  (h2 : num_paintings_sold = 3)
  (h3 : price_per_painting = 50)
  (h4 : art_supplies_cost = 40)
  (h5 : exhibition_fee = 20) :
  contest_prize + num_paintings_sold * price_per_painting - (art_supplies_cost + exhibition_fee) = 240 :=
by sorry

end artist_profit_calculation_l1906_190647


namespace park_length_l1906_190686

/-- Given a rectangular park with width 9 km and perimeter 46 km, its length is 14 km. -/
theorem park_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 9 → perimeter = 46 → perimeter = 2 * (length + width) → length = 14 := by
  sorry

end park_length_l1906_190686


namespace polynomial_coefficient_sum_l1906_190633

theorem polynomial_coefficient_sum (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₁ - 2*a₂ + 3*a₃ - 4*a₄ = -216 := by
sorry

end polynomial_coefficient_sum_l1906_190633


namespace circle_area_ratio_l1906_190692

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) : 
  (30 / 360 : ℝ) * (2 * π * r₁) = (24 / 360 : ℝ) * (2 * π * r₂) →
  (π * r₁^2) / (π * r₂^2) = 16 / 25 := by
sorry

end circle_area_ratio_l1906_190692


namespace tournament_sequences_l1906_190616

theorem tournament_sequences (n : ℕ) : (2 * n).choose n = 3432 :=
by
  -- Proof goes here
  sorry

end tournament_sequences_l1906_190616


namespace water_remaining_l1906_190661

theorem water_remaining (initial_water : ℚ) (used_water : ℚ) : 
  initial_water = 7/2 ∧ used_water = 7/3 → initial_water - used_water = 7/6 := by
  sorry

#check water_remaining

end water_remaining_l1906_190661


namespace difference_exists_l1906_190640

def is_valid_sequence (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧ ∀ n : ℕ, n ≥ 1 → x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n

theorem difference_exists (x : ℕ → ℕ) (h : is_valid_sequence x) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
sorry

end difference_exists_l1906_190640


namespace quadratic_function_properties_l1906_190654

/- Define the quadratic function f(x) -/
def f (x : ℝ) := 2 * x^2 - 10 * x

/- Theorem stating the properties of f(x) and the solution sets -/
theorem quadratic_function_properties :
  (∀ x, f x < 0 ↔ 0 < x ∧ x < 5) ∧
  (∀ x ∈ Set.Icc (-1) 4, f x ≤ 12) ∧
  (∃ x ∈ Set.Icc (-1) 4, f x = 12) ∧
  (∀ a < 0,
    (∀ x, (2 * x^2 + (a - 10) * x + 5) / f x > 1 ↔
      ((-1 < a ∧ a < 0 ∧ (x < 0 ∨ (5 < x ∧ x < -5/a))) ∨
       (a = -1 ∧ x < 0) ∨
       (a < -1 ∧ (x < 0 ∨ (-5/a < x ∧ x < 5))))))
  := by sorry

end quadratic_function_properties_l1906_190654


namespace rod_triangle_theorem_l1906_190652

/-- A triple of natural numbers representing the side lengths of a triangle --/
structure TriangleSides where
  a : ℕ
  b : ℕ
  c : ℕ
  a_le_b : a ≤ b
  b_le_c : b ≤ c

/-- Checks if a natural number is prime --/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- Checks if a TriangleSides forms an isosceles triangle --/
def isIsosceles (t : TriangleSides) : Prop :=
  t.a = t.b ∨ t.b = t.c

/-- The main theorem --/
theorem rod_triangle_theorem :
  ∃! (sol : Finset TriangleSides),
    (∀ t ∈ sol, 
      t.a + t.b + t.c = 25 ∧ 
      isPrime t.a ∧ isPrime t.b ∧ isPrime t.c) ∧
    sol.card = 2 ∧
    (∀ t ∈ sol, isIsosceles t) := by sorry

end rod_triangle_theorem_l1906_190652


namespace derivative_f_at_2_l1906_190621

def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

theorem derivative_f_at_2 : 
  deriv f 2 = 15 := by sorry

end derivative_f_at_2_l1906_190621


namespace characterization_of_S_l1906_190641

open Set

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 4 = 0
def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

-- Define the set of a values that satisfy both p and q
def S : Set ℝ := {a : ℝ | p a ∧ q a}

-- State the theorem
theorem characterization_of_S : S = Iic (-2) := by sorry

end characterization_of_S_l1906_190641


namespace consecutive_product_prime_factors_l1906_190626

theorem consecutive_product_prime_factors (n : ℕ) (hn : n ≥ 1) :
  ∃ x : ℕ+, ∃ p : Fin n → ℕ, 
    (∀ i : Fin n, Prime (p i)) ∧ 
    (∀ i j : Fin n, i ≠ j → p i ≠ p j) ∧
    (∀ i : Fin n, (p i) ∣ (x * (x + 1) + 1)) :=
sorry

end consecutive_product_prime_factors_l1906_190626


namespace trigonometric_product_upper_bound_l1906_190630

theorem trigonometric_product_upper_bound :
  ∀ x y z : ℝ,
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 4.5 ∧
  ∃ x y z : ℝ,
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) = 4.5 :=
by sorry

end trigonometric_product_upper_bound_l1906_190630


namespace baking_powder_yesterday_l1906_190648

def baking_powder_today : ℝ := 0.3
def difference_yesterday : ℝ := 0.1

theorem baking_powder_yesterday : baking_powder_today + difference_yesterday = 0.4 := by
  sorry

end baking_powder_yesterday_l1906_190648


namespace sarah_ate_36_candies_l1906_190669

/-- The number of candy pieces Sarah ate -/
def candyEaten (initialCandy : ℕ) (piles : ℕ) (piecesPerPile : ℕ) : ℕ :=
  initialCandy - (piles * piecesPerPile)

/-- Proof that Sarah ate 36 pieces of candy -/
theorem sarah_ate_36_candies :
  candyEaten 108 8 9 = 36 := by
  sorry

end sarah_ate_36_candies_l1906_190669


namespace canoe_trip_average_distance_l1906_190645

/-- Proves that given a 6-day canoe trip with a total distance of 168 km, 
    where 3/7 of the distance is completed in 3 days, 
    the average distance per day for the remaining days is 32 km. -/
theorem canoe_trip_average_distance 
  (total_distance : ℝ) 
  (total_days : ℕ) 
  (completed_fraction : ℚ) 
  (completed_days : ℕ) 
  (h1 : total_distance = 168)
  (h2 : total_days = 6)
  (h3 : completed_fraction = 3/7)
  (h4 : completed_days = 3) : 
  (total_distance * (1 - completed_fraction)) / (total_days - completed_days) = 32 := by
  sorry

end canoe_trip_average_distance_l1906_190645


namespace geometric_sequence_from_arithmetic_l1906_190651

/-- Given a geometric sequence {b_n} where b_1 = 3, and whose 7th, 10th, and 15th terms
    form consecutive terms of an arithmetic sequence with non-zero common difference,
    prove that the general form of b_n is 3 * (5/3)^(n-1). -/
theorem geometric_sequence_from_arithmetic (b : ℕ → ℚ) (d : ℚ) :
  b 1 = 3 →
  d ≠ 0 →
  (∃ a : ℚ, b 7 = a + 6 * d ∧ b 10 = a + 9 * d ∧ b 15 = a + 14 * d) →
  (∀ n : ℕ, b n = 3 * (5/3)^(n-1)) :=
by sorry

end geometric_sequence_from_arithmetic_l1906_190651


namespace partnership_profit_l1906_190639

/-- Calculates the total profit of a partnership given the investments and one partner's profit share -/
theorem partnership_profit
  (a_investment b_investment c_investment : ℕ)
  (c_profit_share : ℕ)
  (h1 : a_investment = 30000)
  (h2 : b_investment = 45000)
  (h3 : c_investment = 50000)
  (h4 : c_profit_share = 36000) :
  ∃ (total_profit : ℕ), total_profit = 90000 ∧
    total_profit * c_investment = (a_investment + b_investment + c_investment) * c_profit_share :=
by sorry

end partnership_profit_l1906_190639


namespace tank_A_height_l1906_190655

-- Define the tanks
structure Tank where
  circumference : ℝ
  height : ℝ

-- Define the problem parameters
def tank_A : Tank := { circumference := 8, height := 10 }
def tank_B : Tank := { circumference := 10, height := 8 }

-- Define the capacity ratio
def capacity_ratio : ℝ := 0.8000000000000001

-- Theorem statement
theorem tank_A_height :
  tank_A.height = 10 ∧
  tank_A.circumference = 8 ∧
  tank_B.circumference = 10 ∧
  tank_B.height = 8 ∧
  (tank_A.circumference * tank_A.height) / (tank_B.circumference * tank_B.height) = capacity_ratio :=
by sorry

end tank_A_height_l1906_190655


namespace function_periodicity_l1906_190638

/-- A function satisfying the given functional equation is periodic with period 4k -/
theorem function_periodicity (f : ℝ → ℝ) (k : ℝ) (hk : k ≠ 0) 
  (h : ∀ x, f (x + k) * (1 - f x) = 1 + f x) : 
  ∀ x, f (x + 4 * k) = f x := by
  sorry

end function_periodicity_l1906_190638


namespace savings_difference_l1906_190613

/-- The original price of the office supplies -/
def original_price : ℝ := 15000

/-- The first discount rate in the successive discounts option -/
def discount1 : ℝ := 0.30

/-- The second discount rate in the successive discounts option -/
def discount2 : ℝ := 0.15

/-- The single discount rate in the alternative option -/
def single_discount : ℝ := 0.40

/-- The price after applying two successive discounts -/
def price_after_successive_discounts : ℝ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- The price after applying a single discount -/
def price_after_single_discount : ℝ :=
  original_price * (1 - single_discount)

/-- Theorem stating the difference in savings between the two discount options -/
theorem savings_difference :
  price_after_single_discount - price_after_successive_discounts = 75 := by
  sorry

end savings_difference_l1906_190613


namespace stratified_sampling_male_athletes_l1906_190608

theorem stratified_sampling_male_athletes :
  let total_athletes : ℕ := 28 + 21
  let male_athletes : ℕ := 28
  let sample_size : ℕ := 14
  let selected_male_athletes : ℕ := (male_athletes * sample_size) / total_athletes
  selected_male_athletes = 8 := by
  sorry

end stratified_sampling_male_athletes_l1906_190608


namespace distribute_6_4_l1906_190691

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 182 ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem distribute_6_4 : distribute 6 4 = 182 := by
  sorry

end distribute_6_4_l1906_190691


namespace unique_N_leads_to_five_l1906_190668

def machine_rule (N : ℕ) : ℕ :=
  if N % 2 = 1 then 2 * N + 2 else N / 2 + 1

def apply_rule_n_times (N : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => N
  | m + 1 => machine_rule (apply_rule_n_times N m)

theorem unique_N_leads_to_five : ∃! N : ℕ, N > 0 ∧ apply_rule_n_times N 6 = 5 ∧ N = 66 := by
  sorry

end unique_N_leads_to_five_l1906_190668


namespace quadratic_expression_value_l1906_190658

theorem quadratic_expression_value : (3^2 : ℝ) - 3*3 + 2 = 2 := by
  sorry

end quadratic_expression_value_l1906_190658


namespace boys_in_school_l1906_190623

theorem boys_in_school (total_students : ℕ) (boy_ratio girl_ratio : ℕ) : 
  total_students = 48 → 
  boy_ratio = 7 →
  girl_ratio = 1 →
  (boy_ratio : ℚ) / girl_ratio = (number_of_boys : ℚ) / (total_students - number_of_boys) →
  number_of_boys = 42 :=
by
  sorry

end boys_in_school_l1906_190623


namespace completing_square_quadratic_l1906_190663

theorem completing_square_quadratic (x : ℝ) :
  x^2 - 2*x = 9 ↔ (x - 1)^2 = 10 := by sorry

end completing_square_quadratic_l1906_190663
