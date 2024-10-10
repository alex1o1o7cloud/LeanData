import Mathlib

namespace trig_identity_l3230_323074

theorem trig_identity (α : ℝ) : 
  Real.sin α ^ 2 + Real.cos (π/6 - α) ^ 2 - Real.sin α * Real.cos (π/6 - α) = 3/4 := by
  sorry

end trig_identity_l3230_323074


namespace intersection_of_M_and_N_l3230_323064

def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {-2, 0}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end intersection_of_M_and_N_l3230_323064


namespace hawks_score_l3230_323065

/-- The number of touchdowns scored by the Hawks -/
def num_touchdowns : ℕ := 3

/-- The number of points for each touchdown -/
def points_per_touchdown : ℕ := 7

/-- The total number of points scored by the Hawks -/
def total_points : ℕ := num_touchdowns * points_per_touchdown

/-- Theorem stating that the total points scored by the Hawks is 21 -/
theorem hawks_score :
  total_points = 21 := by sorry

end hawks_score_l3230_323065


namespace parabola_increasing_condition_l3230_323004

/-- The parabola y = (a-1)x^2 + 1 increases as x increases when x ≥ 0 if and only if a > 1 -/
theorem parabola_increasing_condition (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → (∀ h : ℝ, h > 0 → ((a - 1) * (x + h)^2 + 1) > ((a - 1) * x^2 + 1))) ↔ 
  a > 1 := by sorry

end parabola_increasing_condition_l3230_323004


namespace cameron_total_questions_l3230_323099

def usual_questions_per_tourist : ℕ := 2

def group_size_1 : ℕ := 6
def group_size_2 : ℕ := 11
def group_size_3 : ℕ := 8
def group_size_4 : ℕ := 7

def inquisitive_tourist_multiplier : ℕ := 3

theorem cameron_total_questions :
  let group_1_questions := group_size_1 * usual_questions_per_tourist
  let group_2_questions := group_size_2 * usual_questions_per_tourist
  let group_3_questions := (group_size_3 - 1) * usual_questions_per_tourist +
                           usual_questions_per_tourist * inquisitive_tourist_multiplier
  let group_4_questions := group_size_4 * usual_questions_per_tourist
  group_1_questions + group_2_questions + group_3_questions + group_4_questions = 68 := by
  sorry

end cameron_total_questions_l3230_323099


namespace triangle_properties_l3230_323043

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle --/
def TriangleABC (t : Triangle) : Prop :=
  t.c = 2 * Real.sqrt 3 ∧
  t.c * Real.cos t.B + (t.b - 2 * t.a) * Real.cos t.C = 0

theorem triangle_properties (t : Triangle) (h : TriangleABC t) :
  t.C = π / 3 ∧ 
  (∃ (max_area : ℝ), max_area = 3 * Real.sqrt 3 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area) :=
sorry

end triangle_properties_l3230_323043


namespace line_plane_relationship_l3230_323021

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using two points or a point and a direction vector
  -- This is a simplified representation
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define a plane using a point and a normal vector
  -- This is a simplified representation
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two lines are parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  -- Definition of parallel lines
  sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Definition of a line parallel to a plane
  sorry

/-- A line is a subset of a plane -/
def line_subset_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Definition of a line being a subset of a plane
  sorry

theorem line_plane_relationship (m n : Line3D) (α : Plane3D) 
  (h1 : parallel_lines m n) (h2 : line_parallel_plane m α) :
  line_parallel_plane n α ∨ line_subset_plane n α := by
  sorry

end line_plane_relationship_l3230_323021


namespace greatest_piece_length_l3230_323015

theorem greatest_piece_length (rope1 rope2 rope3 : ℕ) 
  (h1 : rope1 = 28) (h2 : rope2 = 45) (h3 : rope3 = 63) : 
  Nat.gcd rope1 (Nat.gcd rope2 rope3) = 7 := by
  sorry

end greatest_piece_length_l3230_323015


namespace tangent_line_intersection_l3230_323011

/-- Given two circles on a 2D plane, prove that the x-coordinate of the point 
where a line tangent to both circles intersects the x-axis (to the right of the origin) 
is equal to 9/2. -/
theorem tangent_line_intersection 
  (r₁ r₂ c : ℝ) 
  (h₁ : r₁ = 3) 
  (h₂ : r₂ = 5) 
  (h₃ : c = 12) : 
  ∃ x : ℝ, x > 0 ∧ x = 9/2 ∧ 
  (∃ y : ℝ, (x - 0)^2 + y^2 = r₁^2 ∧ (x - c)^2 + y^2 = r₂^2) :=
sorry

end tangent_line_intersection_l3230_323011


namespace outfits_count_l3230_323083

/-- The number of different outfits that can be created given a set of clothing items. -/
def number_of_outfits (shirts : Nat) (pants : Nat) (ties : Nat) (shoes : Nat) : Nat :=
  shirts * pants * (ties + 1) * shoes

/-- Theorem stating that the number of outfits is 240 given the specific clothing items. -/
theorem outfits_count :
  number_of_outfits 5 4 5 2 = 240 := by
  sorry

end outfits_count_l3230_323083


namespace sqrt_equation_solution_l3230_323095

theorem sqrt_equation_solution :
  ∀ x : ℚ, (x > 2) → (Real.sqrt (7 * x) / Real.sqrt (2 * (x - 2)) = 3) → x = 36 / 11 := by
  sorry

end sqrt_equation_solution_l3230_323095


namespace winning_percentage_calculation_l3230_323027

def total_votes : ℕ := 430
def winning_margin : ℕ := 172

theorem winning_percentage_calculation :
  ∀ (winning_percentage : ℚ),
  (winning_percentage * total_votes / 100 - (100 - winning_percentage) * total_votes / 100 = winning_margin) →
  winning_percentage = 70 := by
sorry

end winning_percentage_calculation_l3230_323027


namespace system_solution_existence_l3230_323091

theorem system_solution_existence (a b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + x*y = a ∧ x^2 - y^2 = b) ↔ 
  -2*a ≤ Real.sqrt 3 * b ∧ Real.sqrt 3 * b ≤ 2*a :=
by sorry

end system_solution_existence_l3230_323091


namespace quadratic_roots_property_l3230_323005

theorem quadratic_roots_property (x₁ x₂ c : ℝ) : 
  (x₁^2 + x₁ + c = 0) →
  (x₂^2 + x₂ + c = 0) →
  (x₁^2 * x₂ + x₂^2 * x₁ = 3) →
  c = -3 := by
sorry

end quadratic_roots_property_l3230_323005


namespace expected_black_pairs_modified_deck_l3230_323073

/-- A deck of cards -/
structure Deck :=
  (total : ℕ)
  (black : ℕ)
  (red : ℕ)
  (h_total : total = black + red)

/-- The expected number of pairs of adjacent black cards in a circular deal -/
def expected_black_pairs (d : Deck) : ℚ :=
  (d.black : ℚ) * (d.black - 1) / (d.total - 1)

/-- The main theorem -/
theorem expected_black_pairs_modified_deck :
  ∃ (d : Deck), d.total = 60 ∧ d.black = 30 ∧ d.red = 30 ∧ expected_black_pairs d = 870 / 59 := by
  sorry

end expected_black_pairs_modified_deck_l3230_323073


namespace machine_value_after_two_years_l3230_323024

/-- Calculates the value of a machine after a given number of years, 
    given its initial value and yearly depreciation rate. -/
def machine_value (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

/-- Theorem stating that a machine purchased for $8,000 with a 10% yearly depreciation rate
    will have a value of $6,480 after two years. -/
theorem machine_value_after_two_years :
  machine_value 8000 0.1 2 = 6480 := by
  sorry

#eval machine_value 8000 0.1 2

end machine_value_after_two_years_l3230_323024


namespace unique_solution_condition_l3230_323000

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 := by
  sorry

end unique_solution_condition_l3230_323000


namespace fish_problem_l3230_323085

/-- The number of fish Ken and Kendra brought home -/
def total_fish_brought_home (ken_caught : ℕ) (ken_released : ℕ) (kendra_caught : ℕ) : ℕ :=
  (ken_caught - ken_released) + kendra_caught

/-- Theorem stating the total number of fish brought home by Ken and Kendra -/
theorem fish_problem :
  ∀ (ken_caught : ℕ) (kendra_caught : ℕ),
    ken_caught = 2 * kendra_caught →
    kendra_caught = 30 →
    total_fish_brought_home ken_caught 3 kendra_caught = 87 := by
  sorry


end fish_problem_l3230_323085


namespace distance_to_point_distance_from_origin_to_point_l3230_323075

theorem distance_to_point : ℝ × ℝ → ℝ
  | (x, y) => Real.sqrt (x^2 + y^2)

theorem distance_from_origin_to_point :
  distance_to_point (12, -5) = 13 := by
  sorry

end distance_to_point_distance_from_origin_to_point_l3230_323075


namespace necessary_but_not_sufficient_arithmetic_sequence_ratio_l3230_323013

-- Definition of a sequence
def Sequence (α : Type) := ℕ → α

-- Definition of a geometric sequence with common ratio 2
def IsGeometricSequenceWithRatio2 (a : Sequence ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n

-- Definition of the condition a_n = 2a_{n-1} for n ≥ 2
def SatisfiesCondition (a : Sequence ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

-- Definition of an arithmetic sequence
def IsArithmeticSequence (a : Sequence ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + d

-- Definition of the sum of first n terms of a sequence
def SumOfFirstNTerms (a : Sequence ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => SumOfFirstNTerms a n + a (n + 1)

-- Theorem 1
theorem necessary_but_not_sufficient :
  (∀ a : Sequence ℝ, IsGeometricSequenceWithRatio2 a → SatisfiesCondition a) ∧
  (∃ a : Sequence ℝ, SatisfiesCondition a ∧ ¬IsGeometricSequenceWithRatio2 a) := by sorry

-- Theorem 2
theorem arithmetic_sequence_ratio :
  ∀ a b : Sequence ℝ,
    IsArithmeticSequence a →
    IsArithmeticSequence b →
    (SumOfFirstNTerms a 5) / (SumOfFirstNTerms b 7) = 15 / 13 →
    a 3 / b 4 = 21 / 13 := by sorry

end necessary_but_not_sufficient_arithmetic_sequence_ratio_l3230_323013


namespace negative_x_squared_times_x_cubed_l3230_323026

theorem negative_x_squared_times_x_cubed (x : ℝ) : (-x^2) * x^3 = -x^5 := by
  sorry

end negative_x_squared_times_x_cubed_l3230_323026


namespace total_crates_sold_l3230_323044

/-- Calculates the total number of crates sold over four days given specific sales conditions --/
theorem total_crates_sold (monday : ℕ) : monday = 5 → 28 = monday + (2 * monday) + (2 * monday - 2) + (monday) := by
  sorry

end total_crates_sold_l3230_323044


namespace rectangular_prism_sum_l3230_323068

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  -- We don't need to define specific dimensions, as they don't affect the result

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (rp : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- Theorem: The sum of edges, vertices, and faces of any rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) :
  num_edges rp + num_vertices rp + num_faces rp = 26 := by
  sorry


end rectangular_prism_sum_l3230_323068


namespace max_value_of_f_l3230_323055

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x : ℝ), f x = M) ∧ M = 2 := by
  sorry

end max_value_of_f_l3230_323055


namespace smallest_k_for_divisibility_by_10_l3230_323058

-- Define a prime number with 2009 digits
def largest_prime_2009_digits : Nat :=
  sorry

-- Define the property of being the largest prime with 2009 digits
def is_largest_prime_2009_digits (p : Nat) : Prop :=
  Nat.Prime p ∧ 
  (Nat.digits 10 p).length = 2009 ∧
  ∀ q, Nat.Prime q → (Nat.digits 10 q).length = 2009 → q ≤ p

-- Theorem statement
theorem smallest_k_for_divisibility_by_10 (p : Nat) 
  (h_p : is_largest_prime_2009_digits p) : 
  (∃ k : Nat, k > 0 ∧ (p^2 - k) % 10 = 0) ∧
  (∀ k : Nat, k > 0 → (p^2 - k) % 10 = 0 → k ≥ 1) :=
by sorry

end smallest_k_for_divisibility_by_10_l3230_323058


namespace tangent_line_at_point_one_neg_one_l3230_323007

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x^2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

-- Theorem statement
theorem tangent_line_at_point_one_neg_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -x := by
sorry

end tangent_line_at_point_one_neg_one_l3230_323007


namespace simplify_expression_l3230_323090

theorem simplify_expression (x : ℝ) (h : x ≠ 2) :
  2 - (2 * (1 - (3 - (2 / (2 - x))))) = 6 - 4 / (2 - x) := by
  sorry

end simplify_expression_l3230_323090


namespace cone_height_equal_cylinder_l3230_323094

/-- Given a cylinder M with base radius 2 and height 6, and a cone N with base diameter
    equal to its slant height, if their volumes are equal, then the height of cone N is 6. -/
theorem cone_height_equal_cylinder (r : ℝ) :
  let cylinder_volume := π * 2^2 * 6
  let cone_base_radius := r
  let cone_height := Real.sqrt 3 * r
  let cone_volume := (1/3) * π * cone_base_radius^2 * cone_height
  cylinder_volume = cone_volume →
  cone_height = 6 := by
sorry

end cone_height_equal_cylinder_l3230_323094


namespace symmetry_implies_congruence_l3230_323034

/-- Two shapes in a plane -/
structure Shape : Type :=
  -- Define necessary properties of a shape

/-- Line of symmetry between two shapes -/
structure SymmetryLine : Type :=
  -- Define necessary properties of a symmetry line

/-- Symmetry relation between two shapes about a line -/
def symmetrical (s1 s2 : Shape) (l : SymmetryLine) : Prop :=
  sorry

/-- Congruence relation between two shapes -/
def congruent (s1 s2 : Shape) : Prop :=
  sorry

/-- Theorem: If two shapes are symmetrical about a line, they are congruent -/
theorem symmetry_implies_congruence (s1 s2 : Shape) (l : SymmetryLine) :
  symmetrical s1 s2 l → congruent s1 s2 :=
by sorry

end symmetry_implies_congruence_l3230_323034


namespace right_triangle_sin_A_l3230_323072

/-- In a right triangle ABC with ∠C = 90°, where sides opposite to angles A, B, and C are a, b, and c respectively, sin A = a/c -/
theorem right_triangle_sin_A (A B C : ℝ) (a b c : ℝ) 
  (h_right : A + B + C = Real.pi)
  (h_C : C = Real.pi / 2)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_pythagorean : a^2 + b^2 = c^2) :
  Real.sin A = a / c := by sorry

end right_triangle_sin_A_l3230_323072


namespace range_of_a_l3230_323035

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x^2 - x - a - 2
def g (a x : ℝ) : ℝ := x^2 - (a+1)*x - 2

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : f a x₁ = 0)
  (h₂ : f a x₂ = 0)
  (h₃ : g a x₃ = 0)
  (h₄ : g a x₄ = 0)
  (h₅ : x₃ < x₁ ∧ x₁ < x₄ ∧ x₄ < x₂) :
  -2 < a ∧ a < 0 := by
sorry

end range_of_a_l3230_323035


namespace emily_coloring_books_l3230_323009

/-- The number of coloring books Emily gave away -/
def books_given_away : ℕ := 2

/-- The initial number of coloring books Emily had -/
def initial_books : ℕ := 7

/-- The number of coloring books Emily bought -/
def books_bought : ℕ := 14

/-- The final number of coloring books Emily has -/
def final_books : ℕ := 19

theorem emily_coloring_books : 
  initial_books - books_given_away + books_bought = final_books :=
sorry

end emily_coloring_books_l3230_323009


namespace least_positive_integer_multiple_53_l3230_323049

theorem least_positive_integer_multiple_53 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), 0 < y ∧ y < x → ¬(53 ∣ (2*y)^2 + 2*47*(2*y) + 47^2)) ∧
  (53 ∣ (2*x)^2 + 2*47*(2*x) + 47^2) ∧
  x = 6 := by
sorry

end least_positive_integer_multiple_53_l3230_323049


namespace total_squares_16x16_board_l3230_323089

/-- The size of the chess board -/
def boardSize : Nat := 16

/-- The total number of squares on a square chess board of given size -/
def totalSquares (n : Nat) : Nat :=
  (n * (n + 1) * (2 * n + 1)) / 6

/-- An irregular shape on the chess board -/
structure IrregularShape where
  size : Nat
  isNonRectangular : Bool

/-- Theorem stating the total number of squares on a 16x16 chess board -/
theorem total_squares_16x16_board (shapes : List IrregularShape) 
  (h1 : ∀ s ∈ shapes, s.size ≥ 4)
  (h2 : ∀ s ∈ shapes, s.isNonRectangular = true) :
  totalSquares boardSize = 1496 := by
  sorry

#eval totalSquares boardSize

end total_squares_16x16_board_l3230_323089


namespace tiger_catch_distance_l3230_323014

/-- Calculates the distance a tiger travels from a zoo given specific conditions --/
def tiger_distance (initial_speed : ℝ) (initial_time : ℝ) (slow_speed : ℝ) (slow_time : ℝ) (chase_speed : ℝ) (chase_time : ℝ) : ℝ :=
  initial_speed * initial_time + slow_speed * slow_time + chase_speed * chase_time

/-- Proves that the tiger is caught 140 miles away from the zoo --/
theorem tiger_catch_distance :
  let initial_speed : ℝ := 25
  let initial_time : ℝ := 7
  let slow_speed : ℝ := 10
  let slow_time : ℝ := 4
  let chase_speed : ℝ := 50
  let chase_time : ℝ := 0.5
  tiger_distance initial_speed initial_time slow_speed slow_time chase_speed chase_time = 140 := by
  sorry

#eval tiger_distance 25 7 10 4 50 0.5

end tiger_catch_distance_l3230_323014


namespace valid_colorings_3x10_l3230_323098

/-- Represents the number of ways to color a 3 × 2n grid with black and white,
    such that no five squares in an 'X' configuration are all the same color. -/
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 8
  | n+2 => 7 * a n + 4 * a (n-1)

/-- The number of valid colorings for a 3 × 10 grid -/
def N : ℕ := (a 5)^2

/-- Theorem stating that the number of valid colorings for a 3 × 10 grid
    is equal to 25636^2 -/
theorem valid_colorings_3x10 : N = 25636^2 := by
  sorry

end valid_colorings_3x10_l3230_323098


namespace cab_journey_time_l3230_323012

/-- The usual time for a cab to cover a journey -/
def usual_time : ℝ → Prop :=
  λ T => (6 / 5 * T = T + 15) ∧ (T = 75)

theorem cab_journey_time :
  ∃ T : ℝ, usual_time T :=
sorry

end cab_journey_time_l3230_323012


namespace fourth_grade_students_l3230_323059

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (new : ℕ) (final : ℕ) : 
  left = 6 → new = 42 → final = 47 → initial + new - left = final → initial = 11 := by
  sorry

end fourth_grade_students_l3230_323059


namespace square_minus_product_plus_square_l3230_323036

theorem square_minus_product_plus_square : 6^2 - 4*5 + 4^2 = 32 := by
  sorry

end square_minus_product_plus_square_l3230_323036


namespace mathborough_rainfall_2006_l3230_323067

/-- The total rainfall in Mathborough for 2006 given the average monthly rainfall in 2005 and the increase in 2006 -/
theorem mathborough_rainfall_2006 
  (avg_2005 : ℝ) 
  (increase_2006 : ℝ) 
  (h1 : avg_2005 = 40) 
  (h2 : increase_2006 = 3) : 
  (avg_2005 + increase_2006) * 12 = 516 := by
  sorry

#check mathborough_rainfall_2006

end mathborough_rainfall_2006_l3230_323067


namespace chair_cost_l3230_323060

theorem chair_cost (total_cost : ℝ) (table_cost : ℝ) (num_chairs : ℕ) 
  (h1 : total_cost = 135)
  (h2 : table_cost = 55)
  (h3 : num_chairs = 4) :
  (total_cost - table_cost) / num_chairs = 20 := by
  sorry

end chair_cost_l3230_323060


namespace rectangle_area_l3230_323057

/-- The area of a rectangle with perimeter 176 inches and length 8 inches more than its width is 1920 square inches. -/
theorem rectangle_area (w l : ℝ) (h1 : l = w + 8) (h2 : 2*l + 2*w = 176) : w * l = 1920 := by
  sorry

end rectangle_area_l3230_323057


namespace matrix_power_difference_l3230_323069

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem matrix_power_difference :
  B^20 - 3 • B^19 = !![0, 4 * 2^19; 0, -2^19] := by sorry

end matrix_power_difference_l3230_323069


namespace unique_solution_quadratic_l3230_323052

theorem unique_solution_quadratic (p : ℝ) (hp : p ≠ 0) :
  (∃! x : ℝ, p * x^2 - 8 * x + 2 = 0) ↔ p = 8 := by
  sorry

end unique_solution_quadratic_l3230_323052


namespace no_integer_solutions_l3230_323048

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^2 - 4*y^2 = 2011 := by
  sorry

end no_integer_solutions_l3230_323048


namespace line_equation_point_slope_l3230_323082

/-- Theorem: Equation of a line with given slope passing through a point -/
theorem line_equation_point_slope (k x₀ y₀ : ℝ) :
  ∀ x y : ℝ, (y - y₀ = k * (x - x₀)) ↔ (y = k * x + (y₀ - k * x₀)) :=
by sorry

end line_equation_point_slope_l3230_323082


namespace ratio_difference_theorem_l3230_323096

theorem ratio_difference_theorem (x : ℝ) (h1 : x > 0) :
  (2 * x) / (3 * x) = 2 / 3 ∧
  (2 * x + 4) / (3 * x + 4) = 5 / 7 →
  3 * x - 2 * x = 8 :=
by sorry

end ratio_difference_theorem_l3230_323096


namespace arithmetic_sequence_a5_l3230_323002

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 8 = 12 → a 5 = 6 := by
  sorry

end arithmetic_sequence_a5_l3230_323002


namespace multiples_of_six_or_nine_l3230_323056

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n - 1) / m

theorem multiples_of_six_or_nine (n : ℕ) (h : n = 201) : 
  (count_multiples n 6 + count_multiples n 9) - count_multiples n (lcm 6 9) = 33 :=
by sorry

end multiples_of_six_or_nine_l3230_323056


namespace simplify_expression_l3230_323017

theorem simplify_expression (b y : ℝ) (hb : b = 2) (hy : y = 3) :
  18 * b^4 * y^6 / (27 * b^3 * y^5) = 4 := by
  sorry

end simplify_expression_l3230_323017


namespace expression_factorization_l3230_323062

theorem expression_factorization (x : ℝ) : 
  (7 * x^6 + 36 * x^4 - 8) - (3 * x^6 - 4 * x^4 + 6) = 2 * (2 * x^6 + 20 * x^4 - 7) := by
  sorry

end expression_factorization_l3230_323062


namespace fraction_equals_zero_l3230_323054

theorem fraction_equals_zero (x : ℝ) : (2 * x - 6) / (5 * x + 10) = 0 ↔ x = 3 := by
  sorry

end fraction_equals_zero_l3230_323054


namespace pure_imaginary_complex_number_l3230_323006

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∀ z : ℂ, z = Complex.mk (a^2 + 2*a - 3) (a^2 + a - 6) → z.re = 0) → 
  a = 1 := by
sorry

end pure_imaginary_complex_number_l3230_323006


namespace continuous_cauchy_solution_is_linear_l3230_323031

/-- Cauchy's functional equation -/
def CauchyEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- The theorem stating that continuous solutions of Cauchy's equation are linear -/
theorem continuous_cauchy_solution_is_linear
  (f : ℝ → ℝ) (hf_cont : Continuous f) (hf_cauchy : CauchyEquation f) :
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x :=
sorry

end continuous_cauchy_solution_is_linear_l3230_323031


namespace decimal_value_l3230_323042

theorem decimal_value (x : ℚ) : (10^5 - 10^3) * x = 31 → x = 1 / 3168 := by
  sorry

end decimal_value_l3230_323042


namespace smaller_cube_side_length_l3230_323030

theorem smaller_cube_side_length (R : ℝ) (x : ℝ) : 
  R = Real.sqrt 3 →
  (1 + x)^2 + (x * Real.sqrt 2 / 2)^2 = R^2 →
  x = 2/3 :=
sorry

end smaller_cube_side_length_l3230_323030


namespace union_equality_iff_range_l3230_323081

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m + 1)*x + 2*m < 0}

theorem union_equality_iff_range (m : ℝ) : A ∪ B m = A ↔ -1/2 ≤ m ∧ m ≤ 1 := by
  sorry

end union_equality_iff_range_l3230_323081


namespace f_value_at_half_l3230_323023

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of f is [a-1, 2a] -/
def HasDomain (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x ≠ 0 → a - 1 ≤ x ∧ x ≤ 2 * a

/-- The function f(x) = ax² + bx + 3a + b -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + 3 * a + b

theorem f_value_at_half (a b : ℝ) :
  IsEven (f a b) → HasDomain (f a b) a → f a b (1/2) = 13/12 := by
  sorry

end f_value_at_half_l3230_323023


namespace total_curve_length_is_6pi_l3230_323086

/-- Regular tetrahedron with edge length 4 -/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (is_regular : edge_length = 4)

/-- Point on the surface of the tetrahedron -/
structure SurfacePoint (t : RegularTetrahedron) :=
  (distance_from_vertex : ℝ)
  (on_surface : distance_from_vertex = 3)

/-- Total length of curve segments -/
def total_curve_length (t : RegularTetrahedron) (p : SurfacePoint t) : ℝ := sorry

/-- Theorem: The total length of curve segments is 6π -/
theorem total_curve_length_is_6pi (t : RegularTetrahedron) (p : SurfacePoint t) :
  total_curve_length t p = 6 * Real.pi :=
sorry

end total_curve_length_is_6pi_l3230_323086


namespace remaining_payment_l3230_323001

/-- Given a 10% deposit of $55, prove that the remaining amount to be paid is $495. -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (total_cost : ℝ) : 
  deposit = 55 ∧ 
  deposit_percentage = 0.1 ∧ 
  deposit = deposit_percentage * total_cost →
  total_cost - deposit = 495 := by
sorry

end remaining_payment_l3230_323001


namespace f_s_not_multiplicative_for_other_s_l3230_323051

/-- The count of integer solutions to x_1^2 + x_2^2 + ... + x_s^2 = n -/
def r_s (s n : ℕ) : ℕ := sorry

/-- f_s(n) = r_s(n) / (2s) -/
def f_s (s n : ℕ) : ℚ := (r_s s n : ℚ) / (2 * s : ℚ)

/-- The multiplication rule for f_s -/
def multiplication_rule (s : ℕ) : Prop :=
  ∀ m n : ℕ, Nat.Coprime m n → f_s s (m * n) = f_s s m * f_s s n

theorem f_s_not_multiplicative_for_other_s :
  ∀ s : ℕ, s ≠ 1 ∧ s ≠ 2 ∧ s ≠ 4 ∧ s ≠ 8 →
    ∃ m n : ℕ, f_s s (m * n) ≠ f_s s m * f_s s n :=
by sorry

end f_s_not_multiplicative_for_other_s_l3230_323051


namespace fortieth_number_in_sampling_l3230_323008

/-- Represents the systematic sampling process in a math competition. -/
def systematicSampling (totalStudents : Nat) (sampleSize : Nat) (firstSelected : Nat) : Nat → Nat :=
  fun n => firstSelected + (totalStudents / sampleSize) * (n - 1)

/-- Theorem stating the 40th number in the systematic sampling. -/
theorem fortieth_number_in_sampling :
  systematicSampling 1000 50 15 40 = 795 := by
  sorry

end fortieth_number_in_sampling_l3230_323008


namespace fraction_problem_l3230_323061

theorem fraction_problem (p : ℚ) (f : ℚ) : 
  p = 49 →
  p = 2 * f * p + 35 →
  f = 1 / 7 := by
sorry

end fraction_problem_l3230_323061


namespace painted_cubes_multiple_of_unpainted_l3230_323093

theorem painted_cubes_multiple_of_unpainted (n : ℕ) : ∃ n, n > 0 ∧ (n + 2)^3 > 10 ∧ n^3 ∣ ((n + 2)^3 - n^3) := by
  sorry

end painted_cubes_multiple_of_unpainted_l3230_323093


namespace games_within_division_is_48_l3230_323077

/-- Represents a basketball league with specific game scheduling rules -/
structure BasketballLeague where
  N : ℕ  -- Number of games against each team in own division
  M : ℕ  -- Number of games against each team in other division
  h1 : N > 3 * M
  h2 : M > 5
  h3 : 3 * N + 4 * M = 88

/-- The number of games a team plays within its own division -/
def gamesWithinDivision (league : BasketballLeague) : ℕ := 3 * league.N

/-- Theorem stating the number of games played within a team's own division -/
theorem games_within_division_is_48 (league : BasketballLeague) :
  gamesWithinDivision league = 48 := by
  sorry

#check games_within_division_is_48

end games_within_division_is_48_l3230_323077


namespace lists_count_l3230_323018

/-- The number of distinct items to choose from -/
def n : ℕ := 15

/-- The number of times we draw an item -/
def k : ℕ := 4

/-- The number of possible lists when drawing with replacement -/
def num_lists : ℕ := n^k

theorem lists_count : num_lists = 50625 := by
  sorry

end lists_count_l3230_323018


namespace quadratic_two_distinct_roots_l3230_323046

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 2 = 0 ∧ x₂^2 + m*x₂ - 2 = 0 := by
  sorry

end quadratic_two_distinct_roots_l3230_323046


namespace probability_same_color_value_l3230_323032

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (black_cards : Nat)
  (red_cards : Nat)
  (h_total : total_cards = 52)
  (h_black : black_cards = 26)
  (h_red : red_cards = 26)
  (h_sum : black_cards + red_cards = total_cards)

/-- The probability of drawing four cards of the same color from a standard deck -/
def probability_same_color (d : Deck) : Rat :=
  2 * (d.black_cards.choose 4) / d.total_cards.choose 4

/-- Theorem stating the probability of drawing four cards of the same color -/
theorem probability_same_color_value (d : Deck) :
  probability_same_color d = 276 / 2499 := by
  sorry

end probability_same_color_value_l3230_323032


namespace tan_derivative_l3230_323053

open Real

theorem tan_derivative (x : ℝ) : deriv tan x = 1 / (cos x)^2 := by
  sorry

end tan_derivative_l3230_323053


namespace sqrt_12_minus_neg_one_power_zero_plus_abs_sqrt_3_minus_1_l3230_323050

theorem sqrt_12_minus_neg_one_power_zero_plus_abs_sqrt_3_minus_1 :
  Real.sqrt 12 - ((-1 : ℝ) ^ (0 : ℕ)) + |Real.sqrt 3 - 1| = 3 * Real.sqrt 3 - 2 := by
  sorry

end sqrt_12_minus_neg_one_power_zero_plus_abs_sqrt_3_minus_1_l3230_323050


namespace train_speed_train_speed_proof_l3230_323040

/-- The speed of two trains crossing each other -/
theorem train_speed (train_length : Real) (crossing_time : Real) : Real :=
  let relative_speed := (2 * train_length) / crossing_time
  let train_speed_ms := relative_speed / 2
  let train_speed_kmh := train_speed_ms * 3.6
  18

/-- Proof that the speed of each train is 18 km/hr -/
theorem train_speed_proof :
  train_speed 120 24 = 18 := by
  sorry

end train_speed_train_speed_proof_l3230_323040


namespace complement_of_A_in_U_l3230_323029

-- Define the universal set U
def U : Set ℤ := {-1, 0, 1}

-- Define the set A
def A : Set ℤ := {0, 1}

-- Theorem statement
theorem complement_of_A_in_U :
  {x : ℤ | x ∈ U ∧ x ∉ A} = {-1} := by sorry

end complement_of_A_in_U_l3230_323029


namespace courses_choice_theorem_l3230_323078

/-- The number of courses available -/
def total_courses : ℕ := 4

/-- The number of courses each person chooses -/
def courses_per_person : ℕ := 2

/-- The number of ways to choose courses with at least one difference -/
def ways_with_difference : ℕ := 30

/-- Theorem stating the number of ways to choose courses with at least one difference -/
theorem courses_choice_theorem : 
  (Nat.choose total_courses courses_per_person) * 
  (Nat.choose total_courses courses_per_person) - 
  (Nat.choose total_courses courses_per_person) = ways_with_difference :=
by sorry

end courses_choice_theorem_l3230_323078


namespace h_function_iff_strictly_increasing_l3230_323022

def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ StrictMono f := by sorry

end h_function_iff_strictly_increasing_l3230_323022


namespace ellipse_k_range_l3230_323039

-- Define the ellipse equation
def ellipse_equation (x y k : ℝ) : Prop := x^2 + k * y^2 = 2

-- Define the condition for foci on y-axis
def foci_on_y_axis (k : ℝ) : Prop := k > 0 ∧ k < 1

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, ellipse_equation x y k ∧ foci_on_y_axis k) ↔ 0 < k ∧ k < 1 :=
sorry

end ellipse_k_range_l3230_323039


namespace solve_equation_l3230_323016

theorem solve_equation : ∃ x : ℚ, (3 * x + 15 = (1/3) * (7 * x + 42)) ∧ (x = -3/2) := by
  sorry

end solve_equation_l3230_323016


namespace system_solutions_l3230_323025

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  y = x^3 * (3 - 2*x) ∧
  z = y^3 * (3 - 2*y) ∧
  x = z^3 * (3 - 2*z)

/-- The theorem stating the solutions of the system -/
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ 
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨
     (x = 1 ∧ y = 1 ∧ z = 1) ∨
     (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by sorry

end system_solutions_l3230_323025


namespace prob_heart_king_spade_l3230_323071

/-- Represents a standard deck of 52 cards -/
def standardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def numHearts : ℕ := 13

/-- Number of kings in a standard deck -/
def numKings : ℕ := 4

/-- Number of spades in a standard deck -/
def numSpades : ℕ := 13

/-- Probability of drawing a heart, then a king, then a spade from a standard 52-card deck without replacement -/
theorem prob_heart_king_spade : 
  (numHearts : ℚ) / standardDeck * 
  numKings / (standardDeck - 1) * 
  numSpades / (standardDeck - 2) = 13 / 2550 := by sorry

end prob_heart_king_spade_l3230_323071


namespace min_value_of_e_l3230_323070

def e (x : ℝ) (C : ℝ) : ℝ := (x - 1) * (x - 3) * (x - 4) * (x - 6) + C

theorem min_value_of_e (C : ℝ) : 
  (C = -0.5625) ↔ (∀ x : ℝ, e x C ≥ 1 ∧ ∃ x₀ : ℝ, e x₀ C = 1) :=
sorry

end min_value_of_e_l3230_323070


namespace polynomial_coefficient_bound_l3230_323063

-- Define the polynomial function
def p (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem polynomial_coefficient_bound (a b c d : ℝ) :
  (∀ x : ℝ, |x| < 1 → |p a b c d x| ≤ 1) →
  |a| + |b| + |c| + |d| ≤ 7 := by
  sorry

end polynomial_coefficient_bound_l3230_323063


namespace last_ten_shots_made_l3230_323010

/-- Represents the number of shots made in a sequence of basketball shots -/
structure BasketballShots where
  total : ℕ
  made : ℕ
  percentage : ℚ
  inv_percentage_def : percentage = made / total

/-- The problem statement -/
theorem last_ten_shots_made 
  (initial : BasketballShots)
  (final : BasketballShots)
  (h1 : initial.total = 30)
  (h2 : initial.percentage = 3/5)
  (h3 : final.total = initial.total + 10)
  (h4 : final.percentage = 29/50)
  : final.made - initial.made = 5 := by
  sorry

end last_ten_shots_made_l3230_323010


namespace lcm_gcf_ratio_280_476_l3230_323079

theorem lcm_gcf_ratio_280_476 : Nat.lcm 280 476 / Nat.gcd 280 476 = 170 := by sorry

end lcm_gcf_ratio_280_476_l3230_323079


namespace negative_three_less_than_negative_sqrt_eight_l3230_323045

theorem negative_three_less_than_negative_sqrt_eight : -3 < -Real.sqrt 8 := by
  sorry

end negative_three_less_than_negative_sqrt_eight_l3230_323045


namespace weight_of_CCl4_l3230_323087

/-- The molar mass of Carbon in g/mol -/
def molar_mass_C : ℝ := 12.01

/-- The molar mass of Chlorine in g/mol -/
def molar_mass_Cl : ℝ := 35.45

/-- The number of Carbon atoms in a CCl4 molecule -/
def num_C_atoms : ℕ := 1

/-- The number of Chlorine atoms in a CCl4 molecule -/
def num_Cl_atoms : ℕ := 4

/-- The number of moles of CCl4 -/
def num_moles : ℝ := 8

/-- Theorem: The weight of 8 moles of CCl4 is 1230.48 grams -/
theorem weight_of_CCl4 : 
  let molar_mass_CCl4 := molar_mass_C * num_C_atoms + molar_mass_Cl * num_Cl_atoms
  num_moles * molar_mass_CCl4 = 1230.48 := by
  sorry

end weight_of_CCl4_l3230_323087


namespace cubic_three_zeros_a_range_l3230_323092

/-- A function f(x) = x^3 - 3x + a has three distinct zeros -/
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 3*x + a = 0 ∧
    y^3 - 3*y + a = 0 ∧
    z^3 - 3*z + a = 0

/-- If f(x) = x^3 - 3x + a has three distinct zeros, then a is in the open interval (-2, 2) -/
theorem cubic_three_zeros_a_range :
  ∀ a : ℝ, has_three_distinct_zeros a → -2 < a ∧ a < 2 :=
by sorry

end cubic_three_zeros_a_range_l3230_323092


namespace line_equation_through_midpoint_on_hyperbola_l3230_323097

/-- Given a hyperbola and a point M, prove that a line passing through M and intersecting the hyperbola at two points with M as their midpoint has a specific equation. -/
theorem line_equation_through_midpoint_on_hyperbola (x y : ℝ → ℝ) (A B M : ℝ × ℝ) :
  (∀ t : ℝ, (x t)^2 - (y t)^2 / 2 = 1) →  -- Hyperbola equation
  M = (2, 1) →  -- Coordinates of point M
  (∃ t₁ t₂ : ℝ, A = (x t₁, y t₁) ∧ B = (x t₂, y t₂)) →  -- A and B are on the hyperbola
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  ∃ k b : ℝ, k = 4 ∧ b = -7 ∧ ∀ x y : ℝ, y = k * x + b ↔ 4 * x - y - 7 = 0 :=
by sorry

end line_equation_through_midpoint_on_hyperbola_l3230_323097


namespace exam_passing_marks_l3230_323047

theorem exam_passing_marks (T : ℝ) (P : ℝ) : 
  (0.3 * T = P - 60) →
  (0.4 * T + 10 = P) →
  (0.5 * T - 5 = P + 40) →
  P = 210 := by
  sorry

end exam_passing_marks_l3230_323047


namespace sandy_correct_sums_l3230_323076

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (marks_per_correct : ℕ) 
  (marks_per_incorrect : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 65)
  (h3 : marks_per_correct = 3)
  (h4 : marks_per_incorrect = 2) :
  ∃ (correct_sums : ℕ), 
    correct_sums * marks_per_correct - (total_sums - correct_sums) * marks_per_incorrect = total_marks ∧ 
    correct_sums = 25 :=
by sorry

end sandy_correct_sums_l3230_323076


namespace reasonable_reasoning_types_l3230_323066

/-- Represents different types of reasoning --/
inductive ReasoningType
  | Analogy
  | Inductive
  | Deductive

/-- Determines if a reasoning type is considered reasonable --/
def is_reasonable (r : ReasoningType) : Prop :=
  match r with
  | ReasoningType.Analogy => true
  | ReasoningType.Inductive => true
  | ReasoningType.Deductive => false

/-- Theorem stating which reasoning types are reasonable --/
theorem reasonable_reasoning_types :
  (is_reasonable ReasoningType.Analogy) ∧
  (is_reasonable ReasoningType.Inductive) ∧
  ¬(is_reasonable ReasoningType.Deductive) :=
by sorry


end reasonable_reasoning_types_l3230_323066


namespace cylinder_volume_l3230_323003

/-- The volume of a cylinder with diameter 8 cm and height 5 cm is 80π cubic centimeters. -/
theorem cylinder_volume (π : ℝ) (h : π > 0) : 
  let d : ℝ := 8 -- diameter in cm
  let h : ℝ := 5 -- height in cm
  let r : ℝ := d / 2 -- radius in cm
  let volume : ℝ := π * r^2 * h
  volume = 80 * π := by sorry

end cylinder_volume_l3230_323003


namespace solution_set_a_neg_one_range_of_a_l3230_323020

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |x + 3|

-- Part 1: Solution set when a = -1
theorem solution_set_a_neg_one :
  {x : ℝ | f (-1) x ≤ 1} = {x : ℝ | x ≥ -5/2} := by sorry

-- Part 2: Range of a when f(x) ≤ 4 for all x ∈ [0,3]
theorem range_of_a :
  {a : ℝ | ∀ x ∈ Set.Icc 0 3, f a x ≤ 4} = Set.Icc (-7) 7 := by sorry

end solution_set_a_neg_one_range_of_a_l3230_323020


namespace min_value_of_sum_l3230_323033

theorem min_value_of_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 4/b = 1) :
  ∃ (min : ℝ), min = 9 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 4/y = 1 → x + y ≥ min :=
by sorry

end min_value_of_sum_l3230_323033


namespace elberta_has_35_l3230_323038

/-- Amount of money Granny Smith has -/
def granny_smith : ℕ := 120

/-- Amount of money Anjou has -/
def anjou : ℕ := granny_smith / 4

/-- Amount of money Elberta has -/
def elberta : ℕ := anjou + 5

/-- Theorem stating that Elberta has $35 -/
theorem elberta_has_35 : elberta = 35 := by
  sorry

end elberta_has_35_l3230_323038


namespace power_division_result_l3230_323028

theorem power_division_result : 6^15 / 36^5 = 7776 := by
  sorry

end power_division_result_l3230_323028


namespace abs_eq_neg_iff_nonpos_l3230_323080

theorem abs_eq_neg_iff_nonpos (a : ℝ) : |a| = -a ↔ a ≤ 0 := by sorry

end abs_eq_neg_iff_nonpos_l3230_323080


namespace gcd_problem_l3230_323088

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 1171) :
  Int.gcd (3 * b^2 + 17 * b + 91) (b + 11) = 11 := by
  sorry

end gcd_problem_l3230_323088


namespace pure_imaginary_fraction_l3230_323041

theorem pure_imaginary_fraction (m : ℝ) : 
  (∃ (k : ℝ), (2 - m * Complex.I) / (1 + Complex.I) = k * Complex.I) → m = 2 := by
  sorry

end pure_imaginary_fraction_l3230_323041


namespace even_function_properties_l3230_323084

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem even_function_properties (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_incr : increasing_on f 0 7)
  (h_f7 : f 7 = 6) :
  decreasing_on f (-7) 0 ∧ ∀ x, -7 ≤ x → x ≤ 7 → f x ≤ 6 :=
sorry

end even_function_properties_l3230_323084


namespace lemming_average_distance_l3230_323037

theorem lemming_average_distance (square_side : ℝ) (diagonal_move : ℝ) (perpendicular_move : ℝ) : 
  square_side = 12 →
  diagonal_move = 7.2 →
  perpendicular_move = 3 →
  let diagonal_length := square_side * Real.sqrt 2
  let fraction := diagonal_move / diagonal_length
  let x := fraction * square_side + perpendicular_move
  let y := fraction * square_side
  let dist_left := x
  let dist_bottom := y
  let dist_right := square_side - x
  let dist_top := square_side - y
  (dist_left + dist_bottom + dist_right + dist_top) / 4 = 6 := by
sorry

end lemming_average_distance_l3230_323037


namespace smallest_multiple_of_45_and_60_not_18_l3230_323019

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_multiple_of_45_and_60_not_18 : 
  (∀ n : ℕ, n < 810 → (is_multiple n 45 ∧ is_multiple n 60) → is_multiple n 18) ∧ 
  is_multiple 810 45 ∧ 
  is_multiple 810 60 ∧ 
  ¬is_multiple 810 18 := by
  sorry

end smallest_multiple_of_45_and_60_not_18_l3230_323019
