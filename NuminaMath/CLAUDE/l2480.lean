import Mathlib

namespace quadratic_function_properties_l2480_248048

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > 0
  h2 : a + b + c = -a/2
  h3 : 3 * a > 2 * c
  h4 : 2 * c > 2 * b

/-- The main theorem about the properties of the quadratic function -/
theorem quadratic_function_properties (f : QuadraticFunction) :
  (-3 < f.b / f.a ∧ f.b / f.a < -3/4) ∧
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f.a * x^2 + f.b * x + f.c = 0) ∧
  (∀ x₁ x₂ : ℝ, f.a * x₁^2 + f.b * x₁ + f.c = 0 → f.a * x₂^2 + f.b * x₂ + f.c = 0 →
    Real.sqrt 2 ≤ |x₁ - x₂| ∧ |x₁ - x₂| < Real.sqrt 57 / 4) := by
  sorry

end quadratic_function_properties_l2480_248048


namespace book_reading_time_l2480_248093

theorem book_reading_time (total_books : ℕ) (books_per_week : ℕ) (weeks : ℕ) : 
  total_books = 30 → books_per_week = 6 → weeks * books_per_week = total_books → weeks = 5 := by
  sorry

end book_reading_time_l2480_248093


namespace sqrt_360000_l2480_248078

theorem sqrt_360000 : Real.sqrt 360000 = 600 := by
  sorry

end sqrt_360000_l2480_248078


namespace fraction_sum_equals_three_halves_l2480_248038

theorem fraction_sum_equals_three_halves (a b : ℕ+) :
  ∃ x y : ℕ+, (x : ℚ) / ((y : ℚ) + (a : ℚ)) + (y : ℚ) / ((x : ℚ) + (b : ℚ)) = 3 / 2 := by
  sorry

end fraction_sum_equals_three_halves_l2480_248038


namespace max_value_of_f_on_interval_l2480_248008

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2) 2 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-2) 2 → f x ≤ f c ∧
  f c = 11 :=
sorry

end max_value_of_f_on_interval_l2480_248008


namespace min_apples_collected_l2480_248045

theorem min_apples_collected (n : ℕ) 
  (h1 : n > 0)
  (h2 : ∃ (p1 p2 p3 p4 p5 : ℕ), 
    p1 + p2 + p3 + p4 + p5 = 100 ∧ 
    0 < p1 ∧ p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 < p5 ∧
    (∀ i ∈ [p1, p2, p3, p4], (i * (n * 7 / 10) % 100 = 0)))
  (h3 : ∀ m : ℕ, m < n → 
    ¬(∃ (q1 q2 q3 q4 q5 : ℕ), 
      q1 + q2 + q3 + q4 + q5 = 100 ∧ 
      0 < q1 ∧ q1 < q2 ∧ q2 < q3 ∧ q3 < q4 ∧ q4 < q5 ∧
      (∀ i ∈ [q1, q2, q3, q4], (i * (m * 7 / 10) % 100 = 0))))
  : n = 20 :=
sorry

end min_apples_collected_l2480_248045


namespace work_completion_time_l2480_248064

/-- Given two workers a and b, where:
    1. a and b can finish the work together in 30 days
    2. a alone can finish the work in 60 days
    3. a and b worked together for 20 days before b left
    This theorem proves that a finishes the remaining work in 20 days after b left. -/
theorem work_completion_time 
  (total_work : ℝ) 
  (a_rate : ℝ) 
  (b_rate : ℝ) 
  (h1 : a_rate + b_rate = total_work / 30)
  (h2 : a_rate = total_work / 60)
  (h3 : (a_rate + b_rate) * 20 = 2 * total_work / 3) :
  (total_work / 3) / a_rate = 20 := by
  sorry

#check work_completion_time

end work_completion_time_l2480_248064


namespace alternating_color_probability_alternating_color_probability_proof_l2480_248076

/-- The probability of drawing 10 balls with alternating colors from a box containing 5 white and 5 black balls -/
theorem alternating_color_probability : ℚ :=
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let total_arrangements : ℕ := Nat.choose total_balls white_balls
  let successful_arrangements : ℕ := 2
  1 / 126

/-- Proof that the probability of drawing 10 balls with alternating colors from a box containing 5 white and 5 black balls is 1/126 -/
theorem alternating_color_probability_proof :
  alternating_color_probability = 1 / 126 := by
  sorry

end alternating_color_probability_alternating_color_probability_proof_l2480_248076


namespace taller_tree_is_84_feet_l2480_248000

def taller_tree_height (h1 h2 : ℝ) : Prop :=
  h1 > h2 ∧ h1 - h2 = 24 ∧ h2 / h1 = 5 / 7

theorem taller_tree_is_84_feet :
  ∃ (h1 h2 : ℝ), taller_tree_height h1 h2 ∧ h1 = 84 :=
by
  sorry

end taller_tree_is_84_feet_l2480_248000


namespace james_stickers_l2480_248069

theorem james_stickers (x : ℕ) : x + 22 = 61 → x = 39 := by
  sorry

end james_stickers_l2480_248069


namespace tan_45_degrees_equals_one_l2480_248084

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by sorry

end tan_45_degrees_equals_one_l2480_248084


namespace distance_to_axes_l2480_248053

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ := |p.y|

/-- The distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- Theorem stating the distances from point P(3,5) to the x-axis and y-axis -/
theorem distance_to_axes :
  let P : Point := ⟨3, 5⟩
  distanceToXAxis P = 5 ∧ distanceToYAxis P = 3 := by
  sorry

end distance_to_axes_l2480_248053


namespace french_students_count_l2480_248099

theorem french_students_count (total : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) (french : ℕ) : 
  total = 60 →
  german = 22 →
  both = 9 →
  neither = 6 →
  french + german - both = total - neither →
  french = 41 :=
by sorry

end french_students_count_l2480_248099


namespace bryan_stones_sale_l2480_248057

/-- The total money Bryan received from selling his precious stones collection -/
def total_money (num_emeralds num_rubies num_sapphires : ℕ) 
  (price_emerald price_ruby price_sapphire : ℕ) : ℕ :=
  num_emeralds * price_emerald + num_rubies * price_ruby + num_sapphires * price_sapphire

/-- Theorem stating that Bryan received $17555 for his precious stones collection -/
theorem bryan_stones_sale : 
  total_money 3 2 3 1785 2650 2300 = 17555 := by
  sorry

#eval total_money 3 2 3 1785 2650 2300

end bryan_stones_sale_l2480_248057


namespace pizza_slices_l2480_248034

theorem pizza_slices : ∃ S : ℕ,
  S > 0 ∧
  (3 * S / 4 : ℚ) > 0 ∧
  (9 * S / 16 : ℚ) > 4 ∧
  (9 * S / 16 : ℚ) - 4 = 5 ∧
  S = 16 := by
  sorry

end pizza_slices_l2480_248034


namespace tom_remaining_pieces_l2480_248079

/-- The number of boxes Tom initially bought -/
def initial_boxes : ℕ := 12

/-- The number of boxes Tom gave to his little brother -/
def boxes_given : ℕ := 7

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 6

/-- Theorem: Tom still has 30 pieces of candy -/
theorem tom_remaining_pieces : 
  (initial_boxes - boxes_given) * pieces_per_box = 30 := by
  sorry

end tom_remaining_pieces_l2480_248079


namespace ice_melting_problem_l2480_248036

theorem ice_melting_problem (original_volume : ℝ) : 
  (original_volume > 0) →
  (original_volume * (1/4) * (1/4) = 0.2) → 
  (original_volume = 3.2) :=
by
  sorry

end ice_melting_problem_l2480_248036


namespace icosidodecahedron_vertices_icosidodecahedron_vertices_proof_l2480_248007

/-- An icosidodecahedron is a convex polyhedron with 20 triangular faces and 12 pentagonal faces. -/
structure Icosidodecahedron where
  /-- The number of triangular faces -/
  triangular_faces : ℕ
  /-- The number of pentagonal faces -/
  pentagonal_faces : ℕ
  /-- The icosidodecahedron is a convex polyhedron -/
  is_convex : Bool
  /-- The number of triangular faces is 20 -/
  triangular_faces_eq : triangular_faces = 20
  /-- The number of pentagonal faces is 12 -/
  pentagonal_faces_eq : pentagonal_faces = 12

/-- The number of vertices in an icosidodecahedron is 30 -/
theorem icosidodecahedron_vertices (i : Icosidodecahedron) : ℕ := 30

/-- The number of vertices in an icosidodecahedron is 30 -/
theorem icosidodecahedron_vertices_proof (i : Icosidodecahedron) : 
  icosidodecahedron_vertices i = 30 := by
  sorry

end icosidodecahedron_vertices_icosidodecahedron_vertices_proof_l2480_248007


namespace largest_four_digit_divisible_by_4_with_digit_sum_20_l2480_248037

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_four_digit_divisible_by_4_with_digit_sum_20 :
  ∃ (n : ℕ), is_four_digit n ∧ n % 4 = 0 ∧ digit_sum n = 20 ∧
  ∀ (m : ℕ), is_four_digit m → m % 4 = 0 → digit_sum m = 20 → m ≤ n :=
by sorry

end largest_four_digit_divisible_by_4_with_digit_sum_20_l2480_248037


namespace expression_evaluation_l2480_248055

theorem expression_evaluation : 5 * 12 + 2 * 15 - (3 * 7 + 4 * 6) = 45 := by
  sorry

end expression_evaluation_l2480_248055


namespace volume_of_inscribed_sphere_l2480_248004

/-- The volume of a sphere inscribed in a cube with edge length 6 inches is 36π cubic inches. -/
theorem volume_of_inscribed_sphere (π : ℝ) : ℝ := by
  -- Define the edge length of the cube
  let cube_edge : ℝ := 6

  -- Define the radius of the inscribed sphere
  let sphere_radius : ℝ := cube_edge / 2

  -- Define the volume of the sphere
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3

  -- Prove that the volume equals 36π
  sorry

#check volume_of_inscribed_sphere

end volume_of_inscribed_sphere_l2480_248004


namespace distribute_six_books_l2480_248058

/-- The number of ways to distribute n different books among two people, 
    with each person getting one book. -/
def distribute_books (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem stating that distributing 6 different books among two people, 
    with each person getting one book, can be done in 30 different ways. -/
theorem distribute_six_books : distribute_books 6 = 30 := by
  sorry

end distribute_six_books_l2480_248058


namespace problem_1_problem_2a_problem_2b_problem_2c_l2480_248028

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 2

-- State the theorems
theorem problem_1 : 27^(2/3) + Real.log 5 / Real.log 10 - 2 * Real.log 3 / Real.log 2 + Real.log 2 / Real.log 10 + Real.log 9 / Real.log 2 = 10 := by sorry

theorem problem_2a : f (-Real.sqrt 2) = 8 + 5 * Real.sqrt 2 := by sorry

theorem problem_2b (a : ℝ) : f (-a) = 3 * a^2 + 5 * a + 2 := by sorry

theorem problem_2c (a : ℝ) : f (a + 3) = 3 * a^2 + 13 * a + 14 := by sorry

end problem_1_problem_2a_problem_2b_problem_2c_l2480_248028


namespace pattern_B_cannot_fold_into_tetrahedron_l2480_248009

-- Define the structure of a pattern
structure Pattern :=
  (squares : ℕ)
  (foldLines : ℕ)

-- Define the properties of a regular tetrahedron
structure RegularTetrahedron :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)
  (edgesPerVertex : ℕ)

-- Define the folding function (noncomputable as it's conceptual)
noncomputable def canFoldIntoTetrahedron (p : Pattern) : Prop := sorry

-- Define the specific patterns
def patternA : Pattern := ⟨4, 3⟩
def patternB : Pattern := ⟨4, 3⟩
def patternC : Pattern := ⟨4, 3⟩
def patternD : Pattern := ⟨4, 3⟩

-- Define the properties of a regular tetrahedron
def tetrahedron : RegularTetrahedron := ⟨4, 6, 4, 3⟩

-- State the theorem
theorem pattern_B_cannot_fold_into_tetrahedron :
  ¬(canFoldIntoTetrahedron patternB) :=
sorry

end pattern_B_cannot_fold_into_tetrahedron_l2480_248009


namespace smallest_n_multiple_of_five_l2480_248054

theorem smallest_n_multiple_of_five (x y : ℤ) 
  (hx : ∃ k : ℤ, x + 2 = 5 * k) 
  (hy : ∃ k : ℤ, y - 2 = 5 * k) : 
  (∃ n : ℕ+, ∃ m : ℤ, x^2 + x*y + y^2 + n = 5 * m ∧ 
   ∀ k : ℕ+, k < n → ¬∃ m : ℤ, x^2 + x*y + y^2 + k = 5 * m) → 
  (∃ n : ℕ+, n = 1 ∧ ∃ m : ℤ, x^2 + x*y + y^2 + n = 5 * m ∧ 
   ∀ k : ℕ+, k < n → ¬∃ m : ℤ, x^2 + x*y + y^2 + k = 5 * m) :=
by sorry

end smallest_n_multiple_of_five_l2480_248054


namespace division_multiplication_equality_l2480_248046

theorem division_multiplication_equality : -3 / (1/2) * 2 = -12 := by
  sorry

end division_multiplication_equality_l2480_248046


namespace principal_calculation_l2480_248092

/-- Given a principal amount, prove that it equals 2600 if the simple interest
    at 4% for 5 years is 2080 less than the principal. -/
theorem principal_calculation (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2080 → P = 2600 := by
  sorry

end principal_calculation_l2480_248092


namespace zeros_of_f_l2480_248096

open Real MeasureTheory Set

noncomputable def f (x : ℝ) := cos x - sin (2 * x)

def I : Set ℝ := Icc 0 (2 * π)

theorem zeros_of_f : 
  (∃ (S : Finset ℝ), S.card = 4 ∧ (∀ x ∈ S, x ∈ I ∧ f x = 0) ∧
  (∀ y ∈ I, f y = 0 → y ∈ S)) :=
sorry

end zeros_of_f_l2480_248096


namespace power_product_equality_l2480_248081

theorem power_product_equality (a b : ℕ) (h1 : a = 7^5) (h2 : b = 5^7) : a^7 * b^5 = 35^35 := by
  sorry

end power_product_equality_l2480_248081


namespace walkway_time_proof_l2480_248027

theorem walkway_time_proof (walkway_length : ℝ) (time_against : ℝ) (time_stationary : ℝ)
  (h1 : walkway_length = 80)
  (h2 : time_against = 120)
  (h3 : time_stationary = 60) :
  let person_speed := walkway_length / time_stationary
  let walkway_speed := person_speed - walkway_length / time_against
  walkway_length / (person_speed + walkway_speed) = 40 := by
sorry

end walkway_time_proof_l2480_248027


namespace parabola_equation_l2480_248068

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form x^2 = 2py -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line passing through a point M and tangent to a parabola at two points A and B -/
structure TangentLine where
  M : Point
  A : Point
  B : Point
  parabola : Parabola
  h1 : M.x = 2
  h2 : M.y = -2 * parabola.p
  h3 : A.x^2 = 2 * parabola.p * A.y
  h4 : B.x^2 = 2 * parabola.p * B.y

/-- The main theorem to prove -/
theorem parabola_equation (t : TangentLine) 
  (h : (t.A.y + t.B.y) / 2 = 6) : 
  t.parabola.p = 1 ∨ t.parabola.p = 2 := by
  sorry

end parabola_equation_l2480_248068


namespace quadratic_function_problem_l2480_248031

/-- A quadratic function f(x) = ax^2 + bx + c with integer coefficients -/
def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

/-- The value of j that satisfies the given conditions -/
def j : ℤ := 36

theorem quadratic_function_problem (a b c : ℤ) :
  f a b c 2 = 0 ∧
  200 < f a b c 10 ∧ f a b c 10 < 300 ∧
  400 < f a b c 9 ∧ f a b c 9 < 500 ∧
  1000 * j < f a b c 100 ∧ f a b c 100 < 1000 * (j + 1) →
  j = 36 := by sorry

end quadratic_function_problem_l2480_248031


namespace distance_from_negative_three_point_two_l2480_248023

theorem distance_from_negative_three_point_two (x : ℝ) : 
  (|x + 3.2| = 4) ↔ (x = 0.8 ∨ x = -7.2) := by
  sorry

end distance_from_negative_three_point_two_l2480_248023


namespace min_extracted_tablets_l2480_248089

/-- Represents the contents of a medicine box -/
structure MedicineBox where
  tabletA : Nat
  tabletB : Nat

/-- Represents the minimum number of tablets extracted -/
structure ExtractedTablets where
  minA : Nat
  minB : Nat

/-- Given a medicine box with 10 tablets of each kind and a minimum extraction of 12 tablets,
    proves that the minimum number of tablets of each kind among the extracted is 2 for A and 1 for B -/
theorem min_extracted_tablets (box : MedicineBox) (min_extraction : Nat) :
  box.tabletA = 10 → box.tabletB = 10 → min_extraction = 12 →
  ∃ (extracted : ExtractedTablets),
    extracted.minA = 2 ∧ extracted.minB = 1 ∧
    extracted.minA + extracted.minB ≤ min_extraction ∧
    extracted.minA ≤ box.tabletA ∧ extracted.minB ≤ box.tabletB := by
  sorry

end min_extracted_tablets_l2480_248089


namespace unique_triple_solution_l2480_248022

theorem unique_triple_solution :
  ∃! (s : Set (ℝ × ℝ × ℝ)), 
    (∀ (x y z : ℝ), (x, y, z) ∈ s ↔ 
      (1 + x^4 ≤ 2*(y - z)^2 ∧
       1 + y^4 ≤ 2*(z - x)^2 ∧
       1 + z^4 ≤ 2*(x - y)^2)) ∧
    (s = {(1, 0, -1), (1, -1, 0), (0, 1, -1), (0, -1, 1), (-1, 1, 0), (-1, 0, 1)}) :=
by sorry

end unique_triple_solution_l2480_248022


namespace multiple_of_six_l2480_248006

theorem multiple_of_six (n : ℤ) 
  (h : ∃ k : ℤ, (n^5 / 120) + (n^3 / 24) + (n / 30) = k) : 
  ∃ m : ℤ, n = 6 * m :=
by
  sorry

end multiple_of_six_l2480_248006


namespace polynomial_division_theorem_l2480_248098

theorem polynomial_division_theorem (z : ℂ) : 
  ∃ (r : ℂ), 4*z^5 - 3*z^4 + 2*z^3 - 5*z^2 + 9*z - 4 = 
  (z + 3) * (4*z^4 - 15*z^3 + 47*z^2 - 146*z + 447) + r ∧ 
  Complex.abs r < Complex.abs (z + 3) := by
sorry

end polynomial_division_theorem_l2480_248098


namespace T_2023_mod_10_l2480_248019

/-- Represents a sequence of C's and D's -/
inductive Sequence : Type
| C : Sequence
| D : Sequence
| cons : Sequence → Sequence → Sequence

/-- Checks if a sequence is valid (no more than two consecutive C's or D's) -/
def isValid : Sequence → Bool
| Sequence.C => true
| Sequence.D => true
| Sequence.cons s₁ s₂ => sorry  -- Implementation details omitted

/-- Counts the number of valid sequences of length n -/
def T (n : ℕ+) : ℕ :=
  (List.map (fun s => if isValid s then 1 else 0) (sorry : List Sequence)).sum
  -- Implementation details omitted

/-- Main theorem: T(2023) is congruent to 6 modulo 10 -/
theorem T_2023_mod_10 : T 2023 % 10 = 6 := by sorry

end T_2023_mod_10_l2480_248019


namespace negative_quartic_count_l2480_248032

theorem negative_quartic_count : 
  (∃ (S : Finset Int), (∀ x : Int, x ∈ S ↔ x^4 - 63*x^2 + 126 < 0) ∧ Finset.card S = 12) :=
by sorry

end negative_quartic_count_l2480_248032


namespace problem_solution_l2480_248066

theorem problem_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^3 / y = 2)
  (h2 : y^3 / z = 4)
  (h3 : z^3 / x = 8) : 
  x = 2 := by
sorry

end problem_solution_l2480_248066


namespace range_of_a_l2480_248017

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 5*x + 4 < 0 ↔ a - 1 < x ∧ x < a + 1) → 
  (2 ≤ a ∧ a ≤ 3) :=
sorry

end range_of_a_l2480_248017


namespace intersection_of_A_and_B_l2480_248041

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l2480_248041


namespace factorial_fraction_equality_l2480_248013

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end factorial_fraction_equality_l2480_248013


namespace probability_both_divisible_by_4_l2480_248042

/-- A fair 8-sided die -/
def EightSidedDie : Finset ℕ := Finset.range 8 

/-- The probability of an event occurring when tossing a fair 8-sided die -/
def prob (event : Finset ℕ) : ℚ :=
  event.card / EightSidedDie.card

/-- The set of outcomes divisible by 4 on an 8-sided die -/
def divisibleBy4 : Finset ℕ := Finset.filter (·.mod 4 = 0) EightSidedDie

/-- The probability of getting a number divisible by 4 on one 8-sided die -/
def probDivisibleBy4 : ℚ := prob divisibleBy4

theorem probability_both_divisible_by_4 :
  probDivisibleBy4 * probDivisibleBy4 = 1 / 16 := by sorry

end probability_both_divisible_by_4_l2480_248042


namespace max_perimeter_is_nine_l2480_248001

/-- Represents a configuration of three regular polygons meeting at a point -/
structure PolygonConfiguration where
  p : ℕ
  q : ℕ
  r : ℕ
  p_gt_two : p > 2
  q_gt_two : q > 2
  r_gt_two : r > 2
  distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r
  angle_sum : (p - 2) / p + (q - 2) / q + (r - 2) / r = 2

/-- The perimeter of the resulting polygon -/
def perimeter (config : PolygonConfiguration) : ℕ :=
  config.p + config.q + config.r - 6

/-- Theorem stating that the maximum perimeter is 9 -/
theorem max_perimeter_is_nine :
  ∀ config : PolygonConfiguration, perimeter config ≤ 9 ∧ ∃ config : PolygonConfiguration, perimeter config = 9 :=
sorry

end max_perimeter_is_nine_l2480_248001


namespace daniel_elsa_distance_diff_l2480_248085

/-- Calculates the difference in distance traveled between two cyclists given their speeds and times on different tracks. -/
def distance_difference (daniel_plain_speed elsa_plain_speed : ℝ)
                        (plain_time : ℝ)
                        (daniel_hilly_speed elsa_hilly_speed : ℝ)
                        (hilly_time : ℝ) : ℝ :=
  let daniel_total := daniel_plain_speed * plain_time + daniel_hilly_speed * hilly_time
  let elsa_total := elsa_plain_speed * plain_time + elsa_hilly_speed * hilly_time
  daniel_total - elsa_total

/-- The difference in distance traveled between Daniel and Elsa is 7 miles. -/
theorem daniel_elsa_distance_diff :
  distance_difference 20 18 3 16 15 1 = 7 := by
  sorry

end daniel_elsa_distance_diff_l2480_248085


namespace intersection_and_union_subset_condition_l2480_248040

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Theorem for part (I)
theorem intersection_and_union :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) := by sorry

-- Theorem for part (II)
theorem subset_condition (k : ℝ) :
  {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1} ⊆ A ↔ k > 1 ∨ k < -5/2 := by sorry

end intersection_and_union_subset_condition_l2480_248040


namespace price_drop_percentage_l2480_248002

/-- Proves that a 50% increase in quantity sold and a 20.000000000000014% increase in gross revenue
    implies a 20% decrease in price -/
theorem price_drop_percentage (P N : ℝ) (P' N' : ℝ) 
    (h_quantity_increase : N' = 1.5 * N)
    (h_revenue_increase : P' * N' = 1.20000000000000014 * (P * N)) : 
    P' = 0.8 * P := by
  sorry

end price_drop_percentage_l2480_248002


namespace symmetric_curves_l2480_248011

/-- The original curve E -/
def E (x y : ℝ) : Prop :=
  5 * x^2 + 12 * x * y - 22 * x - 12 * y - 19 = 0

/-- The line of symmetry l -/
def l (x y : ℝ) : Prop :=
  x - y + 2 = 0

/-- The symmetric curve E' -/
def E' (x y : ℝ) : Prop :=
  12 * x * y + 5 * y^2 - 78 * y + 45 = 0

/-- Theorem stating that E' is symmetric to E with respect to l -/
theorem symmetric_curves : ∀ (x y x' y' : ℝ),
  l ((x + x') / 2) ((y + y') / 2) →
  E x y ↔ E' x' y' :=
sorry

end symmetric_curves_l2480_248011


namespace inner_square_area_l2480_248030

/-- Represents a square with side length and area -/
structure Square where
  side_length : ℝ
  area : ℝ

/-- Represents the configuration of two squares -/
structure SquareConfiguration where
  outer : Square
  inner : Square
  wi_length : ℝ

/-- Checks if the configuration is valid -/
def is_valid_configuration (config : SquareConfiguration) : Prop :=
  config.outer.side_length = 10 ∧
  config.wi_length = 3 ∧
  config.inner.area = config.inner.side_length ^ 2 ∧
  config.outer.area = config.outer.side_length ^ 2 ∧
  config.inner.side_length < config.outer.side_length

/-- The main theorem -/
theorem inner_square_area (config : SquareConfiguration) :
  is_valid_configuration config →
  config.inner.area = 21.16 := by
  sorry

end inner_square_area_l2480_248030


namespace regular_square_pyramid_volume_l2480_248070

/-- A regular square pyramid with side edge length 2√3 and angle 60° between side edge and base has volume 6 -/
theorem regular_square_pyramid_volume (side_edge : ℝ) (angle : ℝ) : 
  side_edge = 2 * Real.sqrt 3 →
  angle = π / 3 →
  let height := side_edge * Real.sin angle
  let base_area := (side_edge^2) / 2
  let volume := (1/3) * base_area * height
  volume = 6 := by sorry

end regular_square_pyramid_volume_l2480_248070


namespace add_decimals_l2480_248003

theorem add_decimals : (7.45 : ℝ) + 2.56 = 10.01 := by
  sorry

end add_decimals_l2480_248003


namespace inequality_proof_l2480_248059

theorem inequality_proof (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x^4 + y^4 + z^4 = 1) :
  x^3 / (1 - x^8) + y^3 / (1 - y^8) + z^3 / (1 - z^8) ≥ 9/8 * Real.rpow 3 (1/4) := by
  sorry

end inequality_proof_l2480_248059


namespace tower_height_difference_l2480_248015

theorem tower_height_difference : 
  ∀ (h_clyde h_grace : ℕ), 
  h_grace = 8 * h_clyde → 
  h_grace = 40 → 
  h_grace - h_clyde = 35 :=
by
  sorry

end tower_height_difference_l2480_248015


namespace division_problem_l2480_248091

theorem division_problem (dividend quotient remainder : ℕ) (h1 : dividend = 686) (h2 : quotient = 19) (h3 : remainder = 2) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 36 := by
sorry

end division_problem_l2480_248091


namespace valentines_given_to_children_l2480_248043

theorem valentines_given_to_children (initial : ℕ) (remaining : ℕ) :
  initial = 30 → remaining = 22 → initial - remaining = 8 := by
  sorry

end valentines_given_to_children_l2480_248043


namespace merchant_discount_percentage_l2480_248082

theorem merchant_discount_percentage 
  (markup_percentage : ℝ) 
  (profit_percentage : ℝ) 
  (discount_percentage : ℝ) : 
  markup_percentage = 75 → 
  profit_percentage = 5 → 
  discount_percentage = 40 → 
  let cost_price := 100
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := cost_price * (1 + profit_percentage / 100)
  let discount := marked_price - selling_price
  discount / marked_price * 100 = discount_percentage :=
by sorry

end merchant_discount_percentage_l2480_248082


namespace reggie_long_shots_l2480_248014

/-- Represents the number of points for each type of shot --/
inductive ShotType
  | layup : ShotType
  | freeThrow : ShotType
  | longShot : ShotType

def shotValue : ShotType → ℕ
  | ShotType.layup => 1
  | ShotType.freeThrow => 2
  | ShotType.longShot => 3

/-- Represents the number of shots made by each player --/
structure ShotsMade where
  layups : ℕ
  freeThrows : ℕ
  longShots : ℕ

def totalPoints (shots : ShotsMade) : ℕ :=
  shots.layups * shotValue ShotType.layup +
  shots.freeThrows * shotValue ShotType.freeThrow +
  shots.longShots * shotValue ShotType.longShot

theorem reggie_long_shots
  (reggie : ShotsMade)
  (reggie_brother : ShotsMade)
  (h1 : reggie.layups = 3)
  (h2 : reggie.freeThrows = 2)
  (h3 : reggie_brother.layups = 0)
  (h4 : reggie_brother.freeThrows = 0)
  (h5 : reggie_brother.longShots = 4)
  (h6 : totalPoints reggie + 2 = totalPoints reggie_brother) :
  reggie.longShots = 1 := by
  sorry

end reggie_long_shots_l2480_248014


namespace stone_150_is_6_l2480_248074

/-- Represents the counting pattern described in the problem -/
def stone_count (n : ℕ) : ℕ := 
  if n ≤ 12 then n 
  else if n ≤ 23 then 24 - n 
  else stone_count ((n - 1) % 22 + 1)

/-- The total number of stones -/
def total_stones : ℕ := 12

/-- The number at which we want to find the corresponding stone -/
def target_count : ℕ := 150

/-- Theorem stating that the stone counted as 150 is originally stone number 6 -/
theorem stone_150_is_6 : 
  ∃ (k : ℕ), k ≤ total_stones ∧ stone_count target_count = stone_count k ∧ k = 6 := by
  sorry

end stone_150_is_6_l2480_248074


namespace workshop_average_salary_l2480_248049

/-- Represents the average salary of all workers in a workshop -/
def average_salary (total_workers : ℕ) (technicians : ℕ) (technician_salary : ℕ) (other_salary : ℕ) : ℚ :=
  ((technicians * technician_salary + (total_workers - technicians) * other_salary) : ℚ) / total_workers

/-- Theorem stating the average salary of all workers in the workshop -/
theorem workshop_average_salary :
  average_salary 24 8 12000 6000 = 8000 := by
  sorry

end workshop_average_salary_l2480_248049


namespace partial_fraction_decomposition_l2480_248051

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 6 ∧ x ≠ -3 →
  (4 * x + 7) / (x^2 - 3*x - 18) = (31/9) / (x - 6) + (5/9) / (x + 3) := by
sorry

end partial_fraction_decomposition_l2480_248051


namespace book_recipient_sequences_l2480_248071

theorem book_recipient_sequences (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 3) :
  (n * (n - 1) * (n - 2)) = 2730 :=
by sorry

end book_recipient_sequences_l2480_248071


namespace line_parallel_plane_condition_l2480_248073

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (parallel : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)

-- The main theorem
theorem line_parallel_plane_condition :
  -- If a line is parallel to a plane, then it's not contained in the plane
  (∀ (l : Line) (p : Plane), parallel l p → ¬(contained_in l p)) ∧
  -- There exists a line and a plane such that the line is not contained in the plane
  -- but also not parallel to it (i.e., it intersects the plane)
  (∃ (l : Line) (p : Plane), ¬(contained_in l p) ∧ ¬(parallel l p) ∧ intersects l p) :=
sorry

end line_parallel_plane_condition_l2480_248073


namespace initial_customers_l2480_248012

theorem initial_customers (remaining : ℕ) (left : ℕ) : 
  remaining = 5 → left = 3 → remaining + left = 8 :=
by sorry

end initial_customers_l2480_248012


namespace cubic_equation_roots_l2480_248029

theorem cubic_equation_roots (p q : ℝ) :
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
   ∀ x : ℝ, x^3 - 11*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  p + q = 78 := by
sorry

end cubic_equation_roots_l2480_248029


namespace mityas_age_l2480_248088

theorem mityas_age (shura_age mitya_age : ℝ) : 
  (mitya_age = shura_age + 11) →
  (mitya_age - shura_age = 2 * (shura_age - (mitya_age - shura_age))) →
  mitya_age = 27.5 := by
sorry

end mityas_age_l2480_248088


namespace melted_ice_cream_height_l2480_248044

/-- The height of a cylinder with radius 12 inches that has the same volume as a sphere with radius 3 inches is 1/4 inch. -/
theorem melted_ice_cream_height : 
  ∀ (r_sphere r_cylinder : ℝ) (h : ℝ),
  r_sphere = 3 →
  r_cylinder = 12 →
  (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h →
  h = 1 / 4 := by
sorry


end melted_ice_cream_height_l2480_248044


namespace rectangle_area_l2480_248025

theorem rectangle_area (d : ℝ) (h : d > 0) : ∃ (w l : ℝ),
  w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w^2 + l^2 = d^2 ∧ w * l = (3 / 10) * d^2 := by
  sorry

end rectangle_area_l2480_248025


namespace inverse_g_equals_two_l2480_248086

/-- Given nonzero constants a and b, and a function g defined as g(x) = 1 / (2ax + b),
    prove that the inverse of g evaluated at 1 / (4a + b) is equal to 2. -/
theorem inverse_g_equals_two (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let g := fun x => 1 / (2 * a * x + b)
  Function.invFun g (1 / (4 * a + b)) = 2 := by
  sorry

end inverse_g_equals_two_l2480_248086


namespace owen_final_count_l2480_248047

/-- The number of turtles Owen has after all transformations and donations -/
def final_owen_turtles (initial_owen : ℕ) (johanna_difference : ℕ) : ℕ :=
  let initial_johanna := initial_owen - johanna_difference
  let owen_after_month := initial_owen * 2
  let johanna_after_month := initial_johanna / 2
  owen_after_month + johanna_after_month

/-- Theorem stating that Owen ends up with 50 turtles -/
theorem owen_final_count :
  final_owen_turtles 21 5 = 50 := by
  sorry

end owen_final_count_l2480_248047


namespace algebraic_expression_value_l2480_248090

theorem algebraic_expression_value (a : ℝ) (h : a^2 + a - 4 = 0) : 
  (a^2 - 3) * (a + 2) = -2 := by
  sorry

end algebraic_expression_value_l2480_248090


namespace jennifer_sweets_distribution_l2480_248067

theorem jennifer_sweets_distribution (green blue yellow : ℕ) 
  (h1 : green = 212)
  (h2 : blue = 310)
  (h3 : yellow = 502)
  (friends : ℕ)
  (h4 : friends = 3) :
  (green + blue + yellow) / (friends + 1) = 256 := by
sorry

end jennifer_sweets_distribution_l2480_248067


namespace injective_properties_l2480_248039

variable {A B : Type}
variable (f : A → B)

theorem injective_properties (h : Function.Injective f) :
  (∀ (x₁ x₂ : A), x₁ ≠ x₂ → f x₁ ≠ f x₂) ∧
  (∀ (b : B), ∃! (a : A), f a = b) :=
by sorry

end injective_properties_l2480_248039


namespace correct_articles_for_problem_l2480_248035

/-- Represents the possible articles that can be used before a noun -/
inductive Article
  | A
  | An
  | The
  | None

/-- Represents a noun with its properties -/
structure Noun where
  word : String
  startsWithSilentH : Bool
  isCountable : Bool

/-- Represents a fixed phrase -/
structure FixedPhrase where
  phrase : String
  meaning : String

/-- Function to determine the correct article for a noun -/
def correctArticle (n : Noun) : Article := sorry

/-- Function to determine the correct article for a fixed phrase -/
def correctPhraseArticle (fp : FixedPhrase) : Article := sorry

/-- Theorem stating the correct articles for the given problem -/
theorem correct_articles_for_problem 
  (hour : Noun)
  (out_of_question : FixedPhrase)
  (h1 : hour.word = "hour")
  (h2 : hour.startsWithSilentH = true)
  (h3 : hour.isCountable = true)
  (h4 : out_of_question.phrase = "out of __ question")
  (h5 : out_of_question.meaning = "impossible") :
  correctArticle hour = Article.An ∧ correctPhraseArticle out_of_question = Article.The := by
  sorry

end correct_articles_for_problem_l2480_248035


namespace zoo_feeding_arrangements_l2480_248095

/-- Represents the number of pairs of animals in the zoo -/
def num_pairs : ℕ := 6

/-- Calculates the number of ways to arrange the animals according to the specified pattern -/
def arrangement_count : ℕ :=
  (num_pairs - 1) * -- choices for the second female
  num_pairs * -- choices for the first male
  (Finset.prod (Finset.range (num_pairs - 1)) (λ i => num_pairs - i)) * -- choices for remaining females
  (Finset.prod (Finset.range (num_pairs - 1)) (λ i => num_pairs - i)) -- choices for remaining males

/-- The theorem stating that the number of possible arrangements is 432000 -/
theorem zoo_feeding_arrangements : arrangement_count = 432000 := by
  sorry

end zoo_feeding_arrangements_l2480_248095


namespace c_gains_thousand_l2480_248056

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  house_value : Option Int

/-- Represents a house transaction -/
inductive Transaction
  | Buy (price : Int)
  | Sell (price : Int)

def initial_c : FinancialState := { cash := 15000, house_value := some 12000 }
def initial_d : FinancialState := { cash := 16000, house_value := none }

def house_appreciation : Int := 13000
def house_depreciation : Int := 11000

def apply_transaction (state : FinancialState) (t : Transaction) : FinancialState :=
  match t with
  | Transaction.Buy price => { cash := state.cash - price, house_value := some price }
  | Transaction.Sell price => { cash := state.cash + price, house_value := none }

def net_worth (state : FinancialState) : Int :=
  state.cash + state.house_value.getD 0

theorem c_gains_thousand (c d : FinancialState → FinancialState) :
  c = (λ s => apply_transaction s (Transaction.Sell house_appreciation)) ∘
      (λ s => apply_transaction s (Transaction.Buy house_depreciation)) ∘
      (λ s => { s with house_value := some house_appreciation }) →
  d = (λ s => apply_transaction s (Transaction.Buy house_appreciation)) ∘
      (λ s => apply_transaction s (Transaction.Sell house_depreciation)) →
  net_worth (c initial_c) - net_worth initial_c = 1000 :=
sorry

end c_gains_thousand_l2480_248056


namespace sum_a_b_equals_negative_two_l2480_248075

theorem sum_a_b_equals_negative_two (a b : ℝ) :
  |a - 1| + (b + 3)^2 = 0 → a + b = -2 := by
sorry

end sum_a_b_equals_negative_two_l2480_248075


namespace square_value_l2480_248094

theorem square_value : ∃ (square : ℤ), 9210 - 9124 = 210 - square ∧ square = 124 := by
  sorry

end square_value_l2480_248094


namespace f_monotonically_decreasing_l2480_248005

noncomputable def f (x : ℝ) : ℝ := 3 * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) + (Real.cos x)^2 - 2

def monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem f_monotonically_decreasing (k : ℤ) :
  monotonically_decreasing f (π/3 + k*π) (5*π/3 + k*π) := by sorry

end f_monotonically_decreasing_l2480_248005


namespace smallest_multiple_l2480_248033

theorem smallest_multiple (n : ℕ) : n = 2349 ↔ 
  n > 0 ∧ 
  29 ∣ n ∧ 
  n % 97 = 7 ∧ 
  ∀ m : ℕ, m > 0 → 29 ∣ m → m % 97 = 7 → n ≤ m :=
by sorry

end smallest_multiple_l2480_248033


namespace rico_justin_dog_difference_l2480_248072

theorem rico_justin_dog_difference (justin_dogs : ℕ) (camden_dog_legs : ℕ) (camden_rico_ratio : ℚ) :
  justin_dogs = 14 →
  camden_dog_legs = 72 →
  camden_rico_ratio = 3/4 →
  ∃ (rico_dogs : ℕ), rico_dogs - justin_dogs = 10 :=
by
  sorry

end rico_justin_dog_difference_l2480_248072


namespace base_equality_implies_three_l2480_248018

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 10) * 6 + (n % 10)

/-- Converts a number from an arbitrary base to base 10 -/
def baseNToBase10 (n b : ℕ) : ℕ :=
  (n / 100) * b^2 + ((n / 10) % 10) * b + (n % 10)

theorem base_equality_implies_three :
  ∃! (b : ℕ), b > 0 ∧ base6ToBase10 35 = baseNToBase10 132 b :=
by
  sorry

end base_equality_implies_three_l2480_248018


namespace box_filled_with_large_cubes_l2480_248052

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (b : BoxDimensions) : ℕ :=
  b.length * b.width * b.depth

/-- Calculates the volume of a cube given its side length -/
def cubeVolume (c : Cube) : ℕ :=
  c.sideLength * c.sideLength * c.sideLength

/-- Theorem: A box with dimensions 50 × 60 × 43 inches can be filled completely with 1032 cubes of size 5 × 5 × 5 inches -/
theorem box_filled_with_large_cubes :
  let box := BoxDimensions.mk 50 60 43
  let largeCube := Cube.mk 5
  boxVolume box = 1032 * cubeVolume largeCube := by
  sorry


end box_filled_with_large_cubes_l2480_248052


namespace triangle_side_length_l2480_248062

-- Define a triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real
  x : Real
  y : Real
  z : Real

-- State the theorem
theorem triangle_side_length (t : Triangle) : 
  t.y = 7 → 
  t.z = 3 → 
  Real.cos (t.Y - t.Z) = 17/32 → 
  t.x = Real.sqrt 41 := by
  sorry

end triangle_side_length_l2480_248062


namespace no_404_games_tournament_l2480_248020

theorem no_404_games_tournament : ¬ ∃ (n : ℕ), n > 0 ∧ n * (n - 4) / 2 = 404 := by sorry

end no_404_games_tournament_l2480_248020


namespace rectangle_area_equals_perimeter_l2480_248087

theorem rectangle_area_equals_perimeter (x : ℝ) :
  let length : ℝ := 4 * x
  let width : ℝ := x + 7
  let area : ℝ := length * width
  let perimeter : ℝ := 2 * (length + width)
  (length > 0 ∧ width > 0 ∧ area = perimeter) → x = 0.5 := by
sorry

end rectangle_area_equals_perimeter_l2480_248087


namespace lines_intersect_at_point_l2480_248061

-- Define the two lines in parametric form
def line1 (t : ℝ) : ℝ × ℝ := (1 - 2*t, 2 + 6*t)
def line2 (u : ℝ) : ℝ × ℝ := (3 + u, 8 + 3*u)

-- Define the intersection point
def intersection_point : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem lines_intersect_at_point :
  ∃! p : ℝ × ℝ, (∃ t : ℝ, line1 t = p) ∧ (∃ u : ℝ, line2 u = p) ∧ p = intersection_point :=
sorry

end lines_intersect_at_point_l2480_248061


namespace not_enough_ribbons_l2480_248080

/-- Represents the number of ribbons needed for a gift --/
structure RibbonRequirement where
  typeA : ℕ
  typeB : ℕ

/-- Represents the available ribbon supply --/
structure RibbonSupply where
  typeA : ℤ
  typeB : ℤ

def gift_count : ℕ := 8

def initial_supply : RibbonSupply := ⟨10, 12⟩

def requirement_gifts_1_to_4 : RibbonRequirement := ⟨2, 1⟩
def requirement_gifts_5_to_8 : RibbonRequirement := ⟨1, 3⟩

def total_ribbons_needed (req1 req2 : RibbonRequirement) : RibbonRequirement :=
  ⟨req1.typeA * 4 + req2.typeA * 4, req1.typeB * 4 + req2.typeB * 4⟩

def remaining_ribbons (supply : RibbonSupply) (needed : RibbonRequirement) : RibbonSupply :=
  ⟨supply.typeA - needed.typeA, supply.typeB - needed.typeB⟩

theorem not_enough_ribbons :
  let total_needed := total_ribbons_needed requirement_gifts_1_to_4 requirement_gifts_5_to_8
  let remaining := remaining_ribbons initial_supply total_needed
  remaining.typeA < 0 ∧ remaining.typeB < 0 ∧
  remaining.typeA = -2 ∧ remaining.typeB = -4 :=
by sorry

#check not_enough_ribbons

end not_enough_ribbons_l2480_248080


namespace union_of_sets_l2480_248063

theorem union_of_sets (a b : ℕ) (M N : Set ℕ) : 
  M = {3, 2^a} → N = {a, b} → M ∩ N = {2} → M ∪ N = {1, 2, 3} := by
  sorry

end union_of_sets_l2480_248063


namespace Q_equals_G_l2480_248050

-- Define the sets Q and G
def Q : Set ℝ := {y | ∃ x, y = x^2 + 1}
def G : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem Q_equals_G : Q = G := by sorry

end Q_equals_G_l2480_248050


namespace pure_imaginary_complex_number_l2480_248026

theorem pure_imaginary_complex_number (m : ℝ) : 
  (m * (m + 2)) / (m - 1) = 0 → m = 0 := by sorry

end pure_imaginary_complex_number_l2480_248026


namespace largest_quantity_l2480_248021

theorem largest_quantity (a b c d e : ℝ) 
  (h : a - 2 = b + 3 ∧ a - 2 = c - 4 ∧ a - 2 = d + 5 ∧ a - 2 = e + 1) : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e :=
sorry

end largest_quantity_l2480_248021


namespace perpendicular_lines_k_values_l2480_248010

theorem perpendicular_lines_k_values (k : ℝ) :
  let l1 : ℝ → ℝ → ℝ := λ x y => (k - 3) * x + (k + 4) * y + 1
  let l2 : ℝ → ℝ → ℝ := λ x y => (k + 1) * x + 2 * (k - 3) * y + 3
  (∀ x y, l1 x y = 0 → l2 x y = 0 → (k - 3) * (k + 1) + 2 * (k + 4) * (k - 3) = 0) →
  k = 3 ∨ k = -3 :=
by sorry

end perpendicular_lines_k_values_l2480_248010


namespace two_digit_reverse_sum_l2480_248097

theorem two_digit_reverse_sum (x y n : ℕ) : 
  (x ≠ 0 ∧ y ≠ 0) →
  (10 ≤ x ∧ x < 100) →
  (10 ≤ y ∧ y < 100) →
  (∃ (a b : ℕ), x = 10 * a + b ∧ y = 10 * b + a) →
  x^2 - y^2 = 44 * n →
  x + y + n = 93 := by
sorry

end two_digit_reverse_sum_l2480_248097


namespace equal_utility_days_l2480_248024

/-- Daniel's utility function -/
def utility (reading : ℚ) (soccer : ℚ) : ℚ := reading * soccer

/-- Time spent on Wednesday -/
def wednesday (t : ℚ) : ℚ × ℚ := (10 - t, t)

/-- Time spent on Thursday -/
def thursday (t : ℚ) : ℚ × ℚ := (t + 4, 4 - t)

/-- The theorem stating that t = 8/5 makes the utility equal on both days -/
theorem equal_utility_days (t : ℚ) : 
  t = 8/5 ↔ 
  utility (wednesday t).1 (wednesday t).2 = utility (thursday t).1 (thursday t).2 := by
sorry

end equal_utility_days_l2480_248024


namespace arctan_equation_solution_l2480_248077

theorem arctan_equation_solution (x : ℝ) :
  2 * Real.arctan (1/3) + Real.arctan (1/15) + Real.arctan (1/x) = π/4 → x = 53/4 := by
  sorry

end arctan_equation_solution_l2480_248077


namespace highway_length_l2480_248060

/-- The length of a highway given two cars starting from opposite ends -/
theorem highway_length (speed1 speed2 : ℝ) (time : ℝ) (h1 : speed1 = 54)
  (h2 : speed2 = 57) (h3 : time = 3) :
  speed1 * time + speed2 * time = 333 := by
  sorry

end highway_length_l2480_248060


namespace larger_integer_problem_l2480_248083

theorem larger_integer_problem (a b : ℕ+) : 
  (b : ℚ) / (a : ℚ) = 7 / 3 → 
  (a : ℕ) * b = 189 → 
  b = 21 := by
sorry

end larger_integer_problem_l2480_248083


namespace heptagon_diagonals_l2480_248016

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon has 7 sides -/
def heptagon_sides : ℕ := 7

/-- Theorem: A convex heptagon has 14 diagonals -/
theorem heptagon_diagonals : num_diagonals heptagon_sides = 14 := by
  sorry

end heptagon_diagonals_l2480_248016


namespace final_red_probability_zero_l2480_248065

/-- Represents the color of a marble -/
inductive Color
| Red
| Blue

/-- Represents the state of the jar -/
structure JarState :=
  (red : Nat)
  (blue : Nat)

/-- Represents the result of drawing two marbles -/
inductive DrawResult
| SameColor (c : Color)
| DifferentColors

/-- Simulates drawing two marbles from the jar -/
def draw (state : JarState) : DrawResult := sorry

/-- Updates the jar state based on the draw result -/
def updateJar (state : JarState) (result : DrawResult) : JarState := sorry

/-- Simulates the entire process of drawing and updating three times -/
def process (initialState : JarState) : JarState := sorry

/-- The probability of the final marble being red -/
def finalRedProbability (initialState : JarState) : Real := sorry

/-- Theorem stating that the probability of the final marble being red is 0 -/
theorem final_red_probability_zero :
  finalRedProbability ⟨2, 2⟩ = 0 := by sorry

end final_red_probability_zero_l2480_248065
