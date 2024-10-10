import Mathlib

namespace count_positive_3_to_1400_l2070_207001

/-- Represents the operation of flipping signs in three cells -/
def flip_operation (strip : List Bool) (i j k : Nat) : List Bool :=
  sorry

/-- Checks if a number N is positive according to the problem definition -/
def is_positive (N : Nat) : Bool :=
  sorry

/-- Counts the number of positive integers in the range [3, 1400] -/
def count_positive : Nat :=
  (List.range 1398).filter (fun n => is_positive (n + 3)) |>.length

/-- The main theorem stating the count of positive numbers -/
theorem count_positive_3_to_1400 : count_positive = 1396 :=
  sorry

end count_positive_3_to_1400_l2070_207001


namespace ellipse_equation_equivalence_l2070_207055

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10) ↔
  (x^2 / 25 + y^2 / 21 = 1) :=
by sorry

end ellipse_equation_equivalence_l2070_207055


namespace oxford_high_school_classes_l2070_207085

/-- Represents the structure of Oxford High School -/
structure OxfordHighSchool where
  teachers : ℕ
  principal : ℕ
  students_per_class : ℕ
  total_people : ℕ

/-- Calculates the number of classes in Oxford High School -/
def number_of_classes (school : OxfordHighSchool) : ℕ :=
  let total_students := school.total_people - school.teachers - school.principal
  total_students / school.students_per_class

/-- Theorem stating that Oxford High School has 15 classes -/
theorem oxford_high_school_classes :
  let school : OxfordHighSchool := {
    teachers := 48,
    principal := 1,
    students_per_class := 20,
    total_people := 349
  }
  number_of_classes school = 15 := by
  sorry


end oxford_high_school_classes_l2070_207085


namespace parabola_translation_l2070_207097

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk 2 0 1
  let translated := translate original (-1) (-3)
  translated = Parabola.mk 2 4 (-2) := by sorry

end parabola_translation_l2070_207097


namespace arithmetic_sequence_solution_l2070_207038

theorem arithmetic_sequence_solution :
  let a₁ : ℚ := 2/3
  let a₂ := y - 2
  let a₃ := 4*y - 1
  (a₂ - a₁ = a₃ - a₂) → y = 11/6 :=
by
  sorry

end arithmetic_sequence_solution_l2070_207038


namespace crayons_per_unit_is_six_l2070_207062

/-- Given the total number of units, cost per crayon, and total cost,
    calculate the number of crayons in each unit. -/
def crayons_per_unit (total_units : ℕ) (cost_per_crayon : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost / cost_per_crayon) / total_units

/-- Theorem stating that under the given conditions, there are 6 crayons in each unit. -/
theorem crayons_per_unit_is_six :
  crayons_per_unit 4 2 48 = 6 := by
  sorry

#eval crayons_per_unit 4 2 48

end crayons_per_unit_is_six_l2070_207062


namespace min_diameter_bounds_l2070_207054

/-- The minimum diameter of n points on a plane where the distance between any two points is at least 1 -/
def min_diameter (n : ℕ) : ℝ :=
  sorry

/-- The distance between any two points is at least 1 -/
axiom min_distance (n : ℕ) (i j : Fin n) (points : Fin n → ℝ × ℝ) :
  i ≠ j → dist (points i) (points j) ≥ 1

theorem min_diameter_bounds :
  (∀ n : ℕ, n = 2 ∨ n = 3 → min_diameter n ≥ 1) ∧
  (min_diameter 4 ≥ Real.sqrt 2) ∧
  (min_diameter 5 ≥ (1 + Real.sqrt 5) / 2) :=
sorry

end min_diameter_bounds_l2070_207054


namespace apples_in_box_l2070_207059

/-- The number of boxes containing apples -/
def num_boxes : ℕ := 5

/-- The number of apples removed from each box -/
def apples_removed : ℕ := 60

/-- The number of apples initially in each box -/
def apples_per_box : ℕ := 100

theorem apples_in_box : 
  (num_boxes * apples_per_box) - (num_boxes * apples_removed) = 2 * apples_per_box := by
  sorry

end apples_in_box_l2070_207059


namespace tangent_parallel_at_minus_one_l2070_207074

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_parallel_at_minus_one :
  ∃ (P : ℝ × ℝ), P.1 = -1 ∧ P.2 = f P.1 ∧ f' P.1 = 4 := by
  sorry

end tangent_parallel_at_minus_one_l2070_207074


namespace participation_schemes_count_l2070_207089

/-- The number of students to choose from -/
def totalStudents : Nat := 4

/-- The number of students to be selected -/
def selectedStudents : Nat := 3

/-- The number of subjects -/
def subjects : Nat := 3

/-- Represents that student A must participate -/
def studentAMustParticipate : Prop := True

/-- The total number of different participation schemes -/
def participationSchemes : Nat := 18

theorem participation_schemes_count :
  studentAMustParticipate →
  participationSchemes = (Nat.choose (totalStudents - 1) (selectedStudents - 1)) * (Nat.factorial selectedStudents) :=
by sorry

end participation_schemes_count_l2070_207089


namespace ice_cream_theorem_l2070_207027

def ice_cream_sales (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    a + b + c + d = n ∧
    b = (n - a + 1) / 2 ∧
    c = ((n - a - b + 1) / 2 : ℕ) ∧
    d = ((n - a - b - c + 1) / 2 : ℕ) ∧
    d = 1

theorem ice_cream_theorem :
  ∀ n : ℕ, ice_cream_sales n → n = 15 := by
  sorry

end ice_cream_theorem_l2070_207027


namespace factorization_equality_l2070_207005

theorem factorization_equality (x : ℝ) : 3 * x^2 - 9 * x = 3 * x * (x - 3) := by
  sorry

end factorization_equality_l2070_207005


namespace complement_intersection_equals_set_l2070_207008

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {1, 2, 3, 4}

theorem complement_intersection_equals_set : 
  (U \ (A ∩ B)) = {4, 5} := by sorry

end complement_intersection_equals_set_l2070_207008


namespace max_product_of_three_l2070_207021

def S : Finset Int := {-9, -5, -3, 1, 4, 6, 8}

theorem max_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a * b * c ≤ 360 ∧ ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 360 :=
by sorry

end max_product_of_three_l2070_207021


namespace tank_capacity_is_900_l2070_207007

/-- Represents the capacity of a tank and its filling/draining rates. -/
structure TankSystem where
  capacity : ℕ
  fill_rate_A : ℕ
  fill_rate_B : ℕ
  drain_rate_C : ℕ

/-- Calculates the net amount of water added to the tank in one cycle. -/
def net_fill_per_cycle (t : TankSystem) : ℕ :=
  t.fill_rate_A + t.fill_rate_B - t.drain_rate_C

/-- Theorem stating that under given conditions, the tank capacity is 900 liters. -/
theorem tank_capacity_is_900 (t : TankSystem) 
  (h1 : t.fill_rate_A = 40)
  (h2 : t.fill_rate_B = 30)
  (h3 : t.drain_rate_C = 20)
  (h4 : (54 : ℕ) * (net_fill_per_cycle t) / 3 = t.capacity) :
  t.capacity = 900 := by
  sorry

#check tank_capacity_is_900

end tank_capacity_is_900_l2070_207007


namespace f_geq_2_solution_set_f_minus_abs_geq_0_t_range_l2070_207016

-- Define the function f
def f (x : ℝ) : ℝ := |x| - 2*|x + 3|

-- Theorem for the first part of the problem
theorem f_geq_2_solution_set :
  {x : ℝ | f x ≥ 2} = {x : ℝ | -4 ≤ x ∧ x ≤ -8/3} := by sorry

-- Theorem for the second part of the problem
theorem f_minus_abs_geq_0_t_range :
  {t : ℝ | ∃ x, f x - |3*t - 2| ≥ 0} = {t : ℝ | -1/3 ≤ t ∧ t ≤ 5/3} := by sorry

end f_geq_2_solution_set_f_minus_abs_geq_0_t_range_l2070_207016


namespace angle_c_in_triangle_l2070_207086

theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := by
  sorry

end angle_c_in_triangle_l2070_207086


namespace basketball_court_width_l2070_207010

theorem basketball_court_width (perimeter : ℝ) (length_diff : ℝ) : perimeter = 96 ∧ length_diff = 14 → 
  ∃ width : ℝ, width = 17 ∧ 2 * (width + length_diff) + 2 * width = perimeter := by
  sorry

end basketball_court_width_l2070_207010


namespace paint_for_similar_statues_l2070_207002

/-- The amount of paint needed for similar statues -/
theorem paint_for_similar_statues
  (original_height : ℝ)
  (original_paint : ℝ)
  (new_height : ℝ)
  (num_statues : ℕ)
  (h1 : original_height = 8)
  (h2 : original_paint = 1)
  (h3 : new_height = 2)
  (h4 : num_statues = 320) :
  (num_statues : ℝ) * original_paint * (new_height / original_height) ^ 2 = 20 :=
by sorry

end paint_for_similar_statues_l2070_207002


namespace class_size_l2070_207032

theorem class_size (total : ℕ) (girls_ratio : ℚ) (boys : ℕ) : 
  girls_ratio = 5 / 8 → boys = 60 → total = 160 := by
  sorry

end class_size_l2070_207032


namespace max_area_rectangle_l2070_207079

/-- The maximum area of a rectangle with perimeter 40 is 100 -/
theorem max_area_rectangle (p : ℝ) (h : p = 40) : 
  (∃ (a : ℝ), a > 0 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = p / 2 → x * y ≤ a) ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = p / 2 ∧ x * y = 100) :=
sorry

end max_area_rectangle_l2070_207079


namespace heart_diamond_club_probability_l2070_207077

-- Define a standard deck of cards
def standard_deck : ℕ := 52

-- Define the number of cards of each suit
def cards_per_suit : ℕ := 13

-- Define the probability of drawing a specific sequence of cards
def draw_probability (deck_size : ℕ) (hearts diamonds clubs : ℕ) : ℚ :=
  (hearts : ℚ) / deck_size *
  (diamonds : ℚ) / (deck_size - 1) *
  (clubs : ℚ) / (deck_size - 2)

-- Theorem statement
theorem heart_diamond_club_probability :
  draw_probability standard_deck cards_per_suit cards_per_suit cards_per_suit = 2197 / 132600 := by
  sorry

end heart_diamond_club_probability_l2070_207077


namespace hyperbola_equation_l2070_207013

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (b / a = Real.sqrt 3) → 
  (∃ c : ℝ, c = 4 ∧ c^2 = a^2 + b^2) → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 / 12 = 1) := by
sorry

end hyperbola_equation_l2070_207013


namespace area_ratio_preserved_under_affine_transformation_l2070_207045

-- Define a polygon type
def Polygon := Set (ℝ × ℝ)

-- Define an affine transformation type
def AffineTransformation := (ℝ × ℝ) → (ℝ × ℝ)

-- Define an area function for polygons
noncomputable def area (P : Polygon) : ℝ := sorry

-- State the theorem
theorem area_ratio_preserved_under_affine_transformation
  (M N : Polygon) (f : AffineTransformation) :
  let M' := f '' M
  let N' := f '' N
  area M / area N = area M' / area N' := by sorry

end area_ratio_preserved_under_affine_transformation_l2070_207045


namespace problem_two_l2070_207031

theorem problem_two : -2.5 / (5/16) * (-1/8) = 1 := by
  sorry

end problem_two_l2070_207031


namespace parabola_focus_distance_l2070_207044

/-- The distance from a point on the parabola y^2 = 4x to its focus -/
def distance_to_focus (x : ℝ) : ℝ :=
  x + 1

theorem parabola_focus_distance :
  let x : ℝ := 2
  let y : ℝ := 2 * Real.sqrt 2
  y^2 = 4*x → distance_to_focus x = 3 := by
  sorry

end parabola_focus_distance_l2070_207044


namespace rancher_loss_calculation_l2070_207006

/-- Represents the rancher's cattle situation and calculates the loss --/
def rancher_loss (initial_cattle : ℕ) (initial_total_price : ℕ) (dead_cattle : ℕ) (price_reduction : ℕ) : ℕ :=
  let original_price_per_head := initial_total_price / initial_cattle
  let new_price_per_head := original_price_per_head - price_reduction
  let remaining_cattle := initial_cattle - dead_cattle
  let new_total_price := new_price_per_head * remaining_cattle
  initial_total_price - new_total_price

/-- Theorem stating the rancher's loss given the problem conditions --/
theorem rancher_loss_calculation :
  rancher_loss 340 204000 172 150 = 128400 := by
  sorry

end rancher_loss_calculation_l2070_207006


namespace simplified_expression_value_l2070_207052

theorem simplified_expression_value (a b : ℝ) (h : (b - 1)^2 + |a + 3| = 0) :
  -a^2*b + (3*a*b^2 - a^2*b) - 2*(2*a*b^2 - a^2*b) = 3 := by sorry

end simplified_expression_value_l2070_207052


namespace f_at_two_l2070_207065

/-- Horner's method representation of the polynomial 2x^4 + 3x^3 + 5x - 4 --/
def f (x : ℝ) : ℝ := ((2 * x + 3) * x + 0) * x + 5 * x - 4

/-- Theorem stating that f(2) = 62 --/
theorem f_at_two : f 2 = 62 := by sorry

end f_at_two_l2070_207065


namespace sufficient_not_necessary_l2070_207098

theorem sufficient_not_necessary (a b : ℝ) : 
  (∀ a b : ℝ, b > a ∧ a > 0 → a * (b + 1) > a^2) ∧ 
  (∃ a b : ℝ, a * (b + 1) > a^2 ∧ ¬(b > a ∧ a > 0)) := by
  sorry

end sufficient_not_necessary_l2070_207098


namespace exponent_base_proof_l2070_207099

theorem exponent_base_proof (m : ℤ) (x : ℝ) : 
  ((-2)^(2*m) = x^(12-m)) → (m = 4) → (x = -2) := by
  sorry

end exponent_base_proof_l2070_207099


namespace cube_path_exists_l2070_207030

/-- Represents a cell on the chessboard --/
structure Cell :=
  (x : Fin 8)
  (y : Fin 8)

/-- Represents a face of the cube --/
inductive Face
  | Top
  | Bottom
  | North
  | South
  | East
  | West

/-- Represents the state of the cube on the board --/
structure CubeState :=
  (position : Cell)
  (topFace : Face)

/-- Represents a move of the cube --/
inductive Move
  | North
  | South
  | East
  | West

/-- Function to apply a move to a cube state --/
def applyMove (state : CubeState) (move : Move) : CubeState :=
  sorry

/-- Predicate to check if a cell has been visited --/
def hasVisited (cell : Cell) (path : List CubeState) : Prop :=
  sorry

/-- Theorem: There exists a path for the cube that visits all cells while keeping one face never touching the board --/
theorem cube_path_exists : 
  ∃ (initialState : CubeState) (path : List Move),
    (∀ cell : Cell, hasVisited cell (initialState :: (List.scanl applyMove initialState path))) ∧
    (∃ face : Face, ∀ state ∈ (initialState :: (List.scanl applyMove initialState path)), state.topFace ≠ face) :=
  sorry

end cube_path_exists_l2070_207030


namespace fraction_power_product_l2070_207096

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by sorry

end fraction_power_product_l2070_207096


namespace solve_movie_problem_l2070_207064

def movie_problem (regular_price child_discount adults_count money_given change : ℕ) : Prop :=
  let child_price := regular_price - child_discount
  let total_spent := money_given - change
  let adults_cost := adults_count * regular_price
  let children_cost := total_spent - adults_cost
  ∃ (children_count : ℕ), children_count * child_price = children_cost

theorem solve_movie_problem :
  movie_problem 9 2 2 40 1 = ∃ (children_count : ℕ), children_count = 3 := by
  sorry

end solve_movie_problem_l2070_207064


namespace roots_cubed_equation_reciprocal_squares_equation_sum_and_reciprocal_equation_quotient_roots_equation_l2070_207033

-- Define the quadratic equation and its roots
variable (p q : ℝ)
variable (x₁ x₂ : ℝ)

-- Define the original equation
def original_eq (x : ℝ) : Prop := x^2 + p*x + q = 0

-- Define that x₁ and x₂ are roots of the original equation
axiom root_x₁ : original_eq p q x₁
axiom root_x₂ : original_eq p q x₂

-- Part a
theorem roots_cubed_equation :
  ∀ y, y^2 + (p^3 - 3*p*q)*y + q^3 = 0 ↔ (y = x₁^3 ∨ y = x₂^3) :=
sorry

-- Part b
theorem reciprocal_squares_equation :
  ∀ y, q^2*y^2 + (2*q - p^2)*y + 1 = 0 ↔ (y = 1/x₁^2 ∨ y = 1/x₂^2) :=
sorry

-- Part c
theorem sum_and_reciprocal_equation :
  ∀ y, q*y^2 + p*(q + 1)*y + (q + 1)^2 = 0 ↔ (y = x₁ + 1/x₂ ∨ y = x₂ + 1/x₁) :=
sorry

-- Part d
theorem quotient_roots_equation :
  ∀ y, q*y^2 + (2*q - p^2)*y + q = 0 ↔ (y = x₂/x₁ ∨ y = x₁/x₂) :=
sorry

end roots_cubed_equation_reciprocal_squares_equation_sum_and_reciprocal_equation_quotient_roots_equation_l2070_207033


namespace similar_triangles_height_l2070_207018

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small = 5 →
  area_ratio = 25 / 9 →
  ∃ h_large : ℝ, h_large = h_small * (area_ratio.sqrt) ∧ h_large = 25 / 3 :=
by sorry

end similar_triangles_height_l2070_207018


namespace quadratic_polynomial_negative_root_l2070_207046

theorem quadratic_polynomial_negative_root
  (f : ℝ → ℝ)
  (h1 : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0)
  (h2 : ∀ (a b : ℝ), f (a^2 + b^2) ≥ f (2*a*b)) :
  ∃ (x : ℝ), x < 0 ∧ f x = 0 :=
sorry

end quadratic_polynomial_negative_root_l2070_207046


namespace unique_perfect_square_l2070_207068

theorem unique_perfect_square (x : ℕ) (y : ℕ) : ∃! x, y = x^4 + 2*x^3 + 2*x^2 + 2*x + 1 ∧ ∃ z, y = z^2 := by
  sorry

end unique_perfect_square_l2070_207068


namespace equation_roots_relation_l2070_207070

theorem equation_roots_relation (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, 3 * x₁ - 4 = a ∧ (x₂ + a) / 3 = 1 ∧ x₁ = 2 * x₂) → a = 2 := by
sorry

end equation_roots_relation_l2070_207070


namespace room_width_calculation_l2070_207036

/-- Given a rectangular room with the following properties:
  * length: 5.5 meters
  * total paving cost: 16500 Rs
  * paving rate: 800 Rs per square meter
  This theorem proves that the width of the room is 3.75 meters. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate : ℝ) :
  length = 5.5 →
  total_cost = 16500 →
  rate = 800 →
  (total_cost / rate) / length = 3.75 := by
  sorry

end room_width_calculation_l2070_207036


namespace divisor_quotient_remainder_equality_l2070_207040

theorem divisor_quotient_remainder_equality (n : ℕ) (h : n > 1) :
  let divisors := {d : ℕ | d ∣ (n + 1)}
  let quotients := {q : ℕ | ∃ d ∈ divisors, q = n / d}
  let remainders := {r : ℕ | ∃ d ∈ divisors, r = n % d}
  quotients = remainders :=
by sorry

end divisor_quotient_remainder_equality_l2070_207040


namespace cubic_monotonically_increasing_iff_l2070_207061

/-- A cubic function f(x) = ax³ + bx² + cx + d -/
def cubic_function (a b c d : ℝ) : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d

/-- A function is monotonically increasing -/
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem: For a cubic function f(x) = ax³ + bx² + cx + d with a > 0,
    f(x) is monotonically increasing on ℝ if and only if b² - 3ac ≤ 0 -/
theorem cubic_monotonically_increasing_iff {a b c d : ℝ} (ha : a > 0) :
  monotonically_increasing (cubic_function a b c d) ↔ b^2 - 3*a*c ≤ 0 :=
sorry

end cubic_monotonically_increasing_iff_l2070_207061


namespace divisibility_of_fifth_powers_l2070_207017

theorem divisibility_of_fifth_powers (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (y - z) * (z - x) * (x - y)) := by
  sorry

end divisibility_of_fifth_powers_l2070_207017


namespace parabola_intersects_line_segment_range_l2070_207041

/-- Parabola equation -/
def parabola (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x - 1

/-- Line segment AB -/
def line_segment_AB (x : ℝ) : ℝ := -x + 3

/-- Point A -/
def point_A : ℝ × ℝ := (3, 0)

/-- Point B -/
def point_B : ℝ × ℝ := (0, 3)

/-- Theorem stating the range of m for which the parabola intersects line segment AB at two distinct points -/
theorem parabola_intersects_line_segment_range :
  ∃ (m_min m_max : ℝ), m_min = 3 ∧ m_max = 10/3 ∧
  ∀ (m : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
              0 ≤ x₁ ∧ x₁ ≤ 3 ∧ 0 ≤ x₂ ∧ x₂ ≤ 3 ∧
              parabola m x₁ = line_segment_AB x₁ ∧
              parabola m x₂ = line_segment_AB x₂) ↔
             (m_min ≤ m ∧ m ≤ m_max) :=
sorry

end parabola_intersects_line_segment_range_l2070_207041


namespace removed_cone_height_l2070_207081

-- Define the frustum
structure Frustum where
  height : ℝ
  lowerBaseArea : ℝ
  upperBaseArea : ℝ

-- Define the theorem
theorem removed_cone_height (f : Frustum) (h1 : f.height = 30)
  (h2 : f.lowerBaseArea = 400 * Real.pi) (h3 : f.upperBaseArea = 100 * Real.pi) :
  ∃ (removedHeight : ℝ), removedHeight = f.height := by
  sorry

end removed_cone_height_l2070_207081


namespace train_stop_time_l2070_207048

/-- Proves that a train with given speeds stops for 18 minutes per hour -/
theorem train_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 30)
  (h2 : speed_with_stops = 21) : ℝ :=
by
  -- Define the stop time in minutes
  let stop_time : ℝ := 18
  
  -- Proof goes here
  sorry


end train_stop_time_l2070_207048


namespace max_fraction_sum_l2070_207084

theorem max_fraction_sum (a b c d : ℕ) 
  (h1 : a + c = 1000) 
  (h2 : b + d = 500) : 
  (∀ a' b' c' d' : ℕ, 
    a' + c' = 1000 → 
    b' + d' = 500 → 
    (a' : ℚ) / b' + (c' : ℚ) / d' ≤ 1 / 499 + 999) := by
  sorry

end max_fraction_sum_l2070_207084


namespace river_rectification_l2070_207000

theorem river_rectification 
  (total_length : ℝ) 
  (rate_A : ℝ) 
  (rate_B : ℝ) 
  (total_time : ℝ) 
  (h1 : total_length = 180)
  (h2 : rate_A = 8)
  (h3 : rate_B = 12)
  (h4 : total_time = 20) :
  ∃ (length_A length_B : ℝ),
    length_A + length_B = total_length ∧
    length_A / rate_A + length_B / rate_B = total_time ∧
    length_A = 120 ∧
    length_B = 60 :=
by sorry

end river_rectification_l2070_207000


namespace arithmetic_sequence_n_value_l2070_207003

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_n_value
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a4 : a 4 = 7)
  (h_a3_a6 : a 3 + a 6 = 16)
  (h_an : ∃ n : ℕ, a n = 31) :
  ∃ n : ℕ, a n = 31 ∧ n = 16 :=
sorry

end arithmetic_sequence_n_value_l2070_207003


namespace inequality_equivalence_l2070_207093

theorem inequality_equivalence (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (c - a) * (c - b) * (b - a) < 0 ↔ b*c^2 + c*a^2 + a*b^2 < b^2*c + c^2*a + a^2*b :=
by sorry

end inequality_equivalence_l2070_207093


namespace bags_sold_is_30_l2070_207047

-- Define the variables
def cost_price : ℕ := 4
def selling_price : ℕ := 8
def total_profit : ℕ := 120

-- Define the profit per bag
def profit_per_bag : ℕ := selling_price - cost_price

-- Theorem to prove
theorem bags_sold_is_30 : total_profit / profit_per_bag = 30 := by
  sorry

end bags_sold_is_30_l2070_207047


namespace inequality_transformation_l2070_207071

theorem inequality_transformation (a b : ℝ) (h : a < b) : -a/3 > -b/3 := by
  sorry

end inequality_transformation_l2070_207071


namespace square_of_one_plus_sqrt_two_l2070_207075

theorem square_of_one_plus_sqrt_two : (1 + Real.sqrt 2) ^ 2 = 3 + 2 * Real.sqrt 2 := by
  sorry

end square_of_one_plus_sqrt_two_l2070_207075


namespace games_played_l2070_207063

/-- Given that Andrew spent $9.00 for each game and $45 in total,
    prove that the number of games played is 5. -/
theorem games_played (cost_per_game : ℝ) (total_spent : ℝ) : 
  cost_per_game = 9 → total_spent = 45 → (total_spent / cost_per_game : ℝ) = 5 := by
  sorry

end games_played_l2070_207063


namespace jane_crayon_count_l2070_207022

/-- The number of crayons Jane ends up with after various events -/
def final_crayon_count (initial_count : ℕ) (eaten : ℕ) (packs_bought : ℕ) (crayons_per_pack : ℕ) (broken : ℕ) : ℕ :=
  initial_count - eaten + packs_bought * crayons_per_pack - broken

/-- Theorem stating that Jane ends up with 127 crayons given the conditions -/
theorem jane_crayon_count :
  final_crayon_count 87 7 5 10 3 = 127 := by
  sorry

end jane_crayon_count_l2070_207022


namespace complex_product_real_implies_a_equals_negative_one_l2070_207012

theorem complex_product_real_implies_a_equals_negative_one (a : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (↑(1 + a * Complex.I) * ↑(1 + Complex.I) : ℂ).im = 0 →
  a = -1 := by
  sorry

end complex_product_real_implies_a_equals_negative_one_l2070_207012


namespace cousin_distribution_l2070_207028

/-- The number of ways to distribute n indistinguishable objects into k distinct containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of cousins -/
def num_cousins : ℕ := 5

/-- The number of rooms -/
def num_rooms : ℕ := 4

theorem cousin_distribution :
  distribute num_cousins num_rooms = 52 := by sorry

end cousin_distribution_l2070_207028


namespace max_sum_reciprocal_constraint_l2070_207083

theorem max_sum_reciprocal_constraint (a b c : ℕ+) : 
  (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / c →
  Nat.gcd a.val (Nat.gcd b.val c.val) = 1 →
  a.val + b.val ≤ 2011 →
  (∀ a' b' c' : ℕ+, (1 : ℚ) / a' + (1 : ℚ) / b' = (1 : ℚ) / c' →
    Nat.gcd a'.val (Nat.gcd b'.val c'.val) = 1 →
    a'.val + b'.val ≤ 2011 →
    a'.val + b'.val ≤ a.val + b.val) →
  a.val + b.val = 1936 :=
by sorry

end max_sum_reciprocal_constraint_l2070_207083


namespace square_side_length_l2070_207095

/-- Given a square with perimeter 36 cm, prove that the side length is 9 cm -/
theorem square_side_length (perimeter : ℝ) (is_square : Bool) : 
  is_square ∧ perimeter = 36 → (perimeter / 4 : ℝ) = 9 := by
  sorry

end square_side_length_l2070_207095


namespace arctan_equation_solution_l2070_207034

theorem arctan_equation_solution (x : ℝ) :
  3 * Real.arctan (1/4) + Real.arctan (1/7) + Real.arctan (1/x) = π/4 →
  x = 34/13 := by
sorry

end arctan_equation_solution_l2070_207034


namespace cube_weight_doubling_l2070_207011

/-- Given a cube of metal weighing 7 pounds, prove that another cube of the same metal
    with sides twice as long will weigh 56 pounds. -/
theorem cube_weight_doubling (ρ : ℝ) (s : ℝ) (h1 : s > 0) (h2 : ρ * s^3 = 7) :
  ρ * (2*s)^3 = 56 := by
sorry

end cube_weight_doubling_l2070_207011


namespace max_value_of_f_l2070_207039

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 18 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ M :=
by sorry

end max_value_of_f_l2070_207039


namespace perpendicular_implies_m_eq_half_parallel_implies_m_eq_neg_one_l2070_207060

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

-- Define perpendicularity condition for two lines
def perpendicular (m : ℝ) : Prop := 1 * (m - 2) + m * 3 = 0

-- Define parallelism condition for two lines
def parallel (m : ℝ) : Prop := 1 / (m - 2) = m / 3 ∧ m ≠ 3

-- Theorem 1: If l₁ is perpendicular to l₂, then m = 1/2
theorem perpendicular_implies_m_eq_half :
  ∀ m : ℝ, perpendicular m → m = 1/2 :=
by sorry

-- Theorem 2: If l₁ is parallel to l₂, then m = -1
theorem parallel_implies_m_eq_neg_one :
  ∀ m : ℝ, parallel m → m = -1 :=
by sorry

end perpendicular_implies_m_eq_half_parallel_implies_m_eq_neg_one_l2070_207060


namespace function_properties_l2070_207019

-- Define the function f
def f (m x : ℝ) : ℝ := (m^2 - 1) * x + m^2 - 3*m + 2

-- State the theorem
theorem function_properties :
  ∀ m : ℝ,
  (∀ x y : ℝ, x < y → f m x > f m y) →  -- f is decreasing
  f m 1 = 0 →                          -- f(1) = 0
  (m = 1/2 ∧                           -- m = 1/2
   ∀ x : ℝ, f m (x+1) ≥ x^2 ↔ -3/4 ≤ x ∧ x ≤ 0)  -- range of x
  := by sorry

end function_properties_l2070_207019


namespace weight_of_four_parts_l2070_207082

theorem weight_of_four_parts (total_weight : ℚ) (num_parts : ℕ) (parts_of_interest : ℕ) : 
  total_weight = 2 →
  num_parts = 9 →
  parts_of_interest = 4 →
  (parts_of_interest : ℚ) * (total_weight / num_parts) = 8/9 :=
by sorry

end weight_of_four_parts_l2070_207082


namespace range_of_a_l2070_207088

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x > 0, x + 1/x > a
def q (a : ℝ) : Prop := ∃ x, x^2 - 2*a*x + 1 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(¬(q a))) → 
  (¬(p a ∧ q a)) → 
  ((q a) → (a ≤ -1 ∨ a ≥ 1)) →
  ((¬(p a)) → a ≥ 2) →
  a ≥ 2 := by
sorry

end range_of_a_l2070_207088


namespace cookies_with_new_ingredients_l2070_207042

/-- Represents the number of cookies that can be made with given amounts of flour and sugar. -/
def cookies_made (flour : ℚ) (sugar : ℚ) : ℚ :=
  18 * (flour / 2) -- or equivalently, 18 * (sugar / 1)

/-- Theorem stating that 27 cookies can be made with 3 cups of flour and 1.5 cups of sugar,
    given the initial ratio of ingredients to cookies. -/
theorem cookies_with_new_ingredients :
  cookies_made 3 1.5 = 27 := by
  sorry

end cookies_with_new_ingredients_l2070_207042


namespace share_face_value_l2070_207014

/-- Given a share with a 9% dividend rate and a market value of Rs. 42,
    prove that the face value is Rs. 56 if an investor wants a 12% return. -/
theorem share_face_value (dividend_rate : ℝ) (market_value : ℝ) (desired_return : ℝ) :
  dividend_rate = 0.09 →
  market_value = 42 →
  desired_return = 0.12 →
  ∃ (face_value : ℝ), face_value = 56 ∧ dividend_rate * face_value = desired_return * market_value :=
by
  sorry

#check share_face_value

end share_face_value_l2070_207014


namespace jellybean_problem_l2070_207090

theorem jellybean_problem (x : ℚ) 
  (caleb_jellybeans : x > 0)
  (sophie_jellybeans : ℚ → ℚ)
  (sophie_half : sophie_jellybeans x = x / 2)
  (total_jellybeans : 12 * x + 12 * (sophie_jellybeans x) = 54) :
  x = 3 := by
sorry

end jellybean_problem_l2070_207090


namespace sum_of_fractions_equals_one_l2070_207072

theorem sum_of_fractions_equals_one (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) : 
  1 / (1 + a + a * b) + 1 / (1 + b + b * c) + 1 / (1 + c + c * a) = 1 := by
  sorry

end sum_of_fractions_equals_one_l2070_207072


namespace union_of_sets_l2070_207076

theorem union_of_sets : 
  let A : Set ℤ := {0, 1, 2}
  let B : Set ℤ := {-1, 0}
  A ∪ B = {-1, 0, 1, 2} := by
sorry

end union_of_sets_l2070_207076


namespace car_distance_calculation_l2070_207009

/-- Given a car's speed and how a speed increase affects travel time, calculate the distance traveled. -/
theorem car_distance_calculation (V : ℝ) (D : ℝ) (h1 : V = 40) 
  (h2 : D / V - D / (V + 20) = 0.5) : D = 60 := by
  sorry

end car_distance_calculation_l2070_207009


namespace frequency_converges_to_probability_l2070_207069

/-- A random event in an experiment. -/
structure Event where
  -- Add necessary fields here
  mk :: -- Constructor

/-- The frequency of an event after a given number of trials. -/
def frequency (e : Event) (n : ℕ) : ℝ :=
  sorry

/-- The probability of an event. -/
def probability (e : Event) : ℝ :=
  sorry

/-- Statement: As the number of trials increases, the frequency of an event
    converges to its probability. -/
theorem frequency_converges_to_probability (e : Event) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |frequency e n - probability e| < ε :=
sorry

end frequency_converges_to_probability_l2070_207069


namespace square_perimeter_l2070_207023

/-- Given a square cut into four equal rectangles, where each rectangle's length is four times
    its width, and these rectangles are arranged to form a shape with perimeter 56,
    prove that the perimeter of the original square is 32. -/
theorem square_perimeter (x : ℝ) : 
  x > 0 →  -- width of each rectangle is positive
  (4 * x) > 0 →  -- length of each rectangle is positive
  (28 * x) = 56 →  -- perimeter of the P shape
  (4 * (4 * x)) = 32  -- perimeter of the original square
  := by sorry

end square_perimeter_l2070_207023


namespace simplify_expression_l2070_207029

theorem simplify_expression :
  (8 * 10^12) / (4 * 10^4) + 2 * 10^3 = 200002000 := by
  sorry

end simplify_expression_l2070_207029


namespace unknown_number_value_l2070_207092

theorem unknown_number_value : ∃ x : ℝ, 5 + 2 * (8 - x) = 15 ∧ x = 3 := by
  sorry

end unknown_number_value_l2070_207092


namespace quadratic_point_relationship_l2070_207015

/-- A quadratic function of the form y = -(x-1)² + c -/
def quadratic_function (x c : ℝ) : ℝ := -(x - 1)^2 + c

/-- Three points on the quadratic function -/
structure Points where
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ

/-- The theorem stating the relationship between y₁, y₂, and y₃ -/
theorem quadratic_point_relationship (c : ℝ) (p : Points) :
  p.y₁ = quadratic_function (-3) c →
  p.y₂ = quadratic_function (-1) c →
  p.y₃ = quadratic_function 5 c →
  p.y₂ > p.y₁ ∧ p.y₁ = p.y₃ :=
by sorry

end quadratic_point_relationship_l2070_207015


namespace inequalities_proof_l2070_207037

theorem inequalities_proof :
  (((12 : ℝ) / 11) ^ 11 > ((11 : ℝ) / 10) ^ 10) ∧
  (((12 : ℝ) / 11) ^ 12 < ((11 : ℝ) / 10) ^ 11) ∧
  (((12 : ℝ) / 11) ^ 10 > ((11 : ℝ) / 10) ^ 9) ∧
  (((11 : ℝ) / 10) ^ 12 > ((12 : ℝ) / 11) ^ 13) :=
by sorry

end inequalities_proof_l2070_207037


namespace tj_race_second_half_time_l2070_207026

/-- Represents a race with given parameters -/
structure Race where
  totalDistance : ℝ
  firstHalfTime : ℝ
  averagePace : ℝ

/-- Calculates the time for the second half of the race -/
def secondHalfTime (race : Race) : ℝ :=
  race.averagePace * race.totalDistance - race.firstHalfTime

/-- Theorem stating that for a 10K race with given conditions, 
    the second half time is 30 minutes -/
theorem tj_race_second_half_time :
  let race : Race := {
    totalDistance := 10,
    firstHalfTime := 20,
    averagePace := 5
  }
  secondHalfTime race = 30 := by
  sorry


end tj_race_second_half_time_l2070_207026


namespace parabola_vertex_y_zero_l2070_207053

/-- The parabola y = x^2 - 10x + d has its vertex at y = 0 when d = 25 -/
theorem parabola_vertex_y_zero (x y d : ℝ) : 
  y = x^2 - 10*x + d → 
  (∃ x₀, ∀ x, x^2 - 10*x + d ≥ x₀^2 - 10*x₀ + d) → 
  d = 25 ↔ x^2 - 10*x + d ≥ 0 ∧ ∃ x₁, x₁^2 - 10*x₁ + d = 0 := by
sorry


end parabola_vertex_y_zero_l2070_207053


namespace absolute_value_equation_solution_l2070_207080

theorem absolute_value_equation_solution (a : ℝ) : 
  50 - |a - 2| = |4 - a| → (a = -22 ∨ a = 28) := by
  sorry

end absolute_value_equation_solution_l2070_207080


namespace infinite_triangular_pairs_l2070_207073

theorem infinite_triangular_pairs :
  ∃ (S : Set Nat), Set.Infinite S ∧ 
    (∀ p ∈ S, Prime p ∧ Odd p ∧
      (∀ t : Nat, t > 0 →
        (∃ n : Nat, t = n * (n + 1) / 2) ↔
        (∃ m : Nat, p^2 * t + (p^2 - 1) / 8 = m * (m + 1) / 2))) := by
  sorry

end infinite_triangular_pairs_l2070_207073


namespace three_digit_number_eleven_times_sum_of_digits_l2070_207078

theorem three_digit_number_eleven_times_sum_of_digits :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = 11 * (n / 100 + (n / 10) % 10 + n % 10) :=
by
  sorry

end three_digit_number_eleven_times_sum_of_digits_l2070_207078


namespace angle_expression_equals_half_l2070_207057

theorem angle_expression_equals_half (θ : Real) (h : Real.tan θ = 3) :
  (Real.sin θ + Real.cos (π - θ)) / (Real.sin (π / 2 - θ) - Real.sin (π + θ)) = 1 / 2 := by
  sorry

end angle_expression_equals_half_l2070_207057


namespace same_color_isosceles_count_independent_l2070_207067

/-- Represents a coloring of vertices in a regular polygon -/
structure Coloring (n : ℕ) where
  red : Finset (Fin (6*n+1))
  blue : Finset (Fin (6*n+1))
  partition : red ∪ blue = Finset.univ
  disjoint : red ∩ blue = ∅

/-- Counts the number of isosceles triangles with vertices of the same color -/
def count_same_color_isosceles_triangles (n : ℕ) (c : Coloring n) : ℕ := sorry

/-- Theorem stating that the count of same-color isosceles triangles is independent of coloring -/
theorem same_color_isosceles_count_independent (n : ℕ) :
  ∀ c₁ c₂ : Coloring n, count_same_color_isosceles_triangles n c₁ = count_same_color_isosceles_triangles n c₂ :=
sorry

end same_color_isosceles_count_independent_l2070_207067


namespace prob_exactly_two_prob_at_least_one_l2070_207035

/-- The probability of exactly two out of three independent events occurring, 
    given their individual probabilities -/
theorem prob_exactly_two (p1 p2 p3 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (h3 : 0 ≤ p3 ∧ p3 ≤ 1) :
  p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3 = 
    0.398 ↔ p1 = 0.8 ∧ p2 = 0.7 ∧ p3 = 0.9 :=
sorry

/-- The probability of at least one out of three independent events occurring, 
    given their individual probabilities -/
theorem prob_at_least_one (p1 p2 p3 : ℝ) 
  (h1 : 0 ≤ p1 ∧ p1 ≤ 1) 
  (h2 : 0 ≤ p2 ∧ p2 ≤ 1) 
  (h3 : 0 ≤ p3 ∧ p3 ≤ 1) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 
    0.994 ↔ p1 = 0.8 ∧ p2 = 0.7 ∧ p3 = 0.9 :=
sorry

end prob_exactly_two_prob_at_least_one_l2070_207035


namespace perpendicular_bisector_value_l2070_207043

/-- A line that is a perpendicular bisector of a line segment passes through its midpoint -/
axiom perpendicular_bisector_passes_through_midpoint {x₁ y₁ x₂ y₂ : ℝ} (b : ℝ) :
  (∀ x y, x + y = b → (x - (x₁ + x₂) / 2)^2 + (y - (y₁ + y₂) / 2)^2 = ((x₂ - x₁)^2 + (y₂ - y₁)^2) / 4) →
  b = (x₁ + x₂) / 2 + (y₁ + y₂) / 2

/-- The value of b for the perpendicular bisector of the line segment from (2,1) to (8,7) -/
theorem perpendicular_bisector_value : 
  (∀ x y, x + y = b → (x - 5)^2 + (y - 4)^2 = 25) → b = 9 :=
by sorry

end perpendicular_bisector_value_l2070_207043


namespace red_bead_cost_l2070_207004

/-- The cost of a box of red beads -/
def red_cost : ℝ := 2.30

/-- The cost of a box of yellow beads -/
def yellow_cost : ℝ := 2.00

/-- The number of boxes of each color used -/
def boxes_per_color : ℕ := 4

/-- The total number of mixed boxes -/
def total_boxes : ℕ := 10

/-- The cost per box of mixed beads -/
def mixed_cost : ℝ := 1.72

theorem red_bead_cost :
  red_cost * boxes_per_color + yellow_cost * boxes_per_color = mixed_cost * total_boxes := by
  sorry

#check red_bead_cost

end red_bead_cost_l2070_207004


namespace coeff_x_cubed_eq_60_l2070_207049

/-- The coefficient of x^3 in the expansion of x(1+2x)^6 -/
def coeff_x_cubed : ℕ :=
  (Finset.range 7).sum (fun k => k.choose 6 * 2^k * if k = 2 then 1 else 0)

/-- Theorem stating that the coefficient of x^3 in x(1+2x)^6 is 60 -/
theorem coeff_x_cubed_eq_60 : coeff_x_cubed = 60 := by
  sorry

end coeff_x_cubed_eq_60_l2070_207049


namespace ratio_equality_l2070_207024

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 49)
  (sum_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end ratio_equality_l2070_207024


namespace gift_box_volume_l2070_207087

/-- The volume of a rectangular box -/
def box_volume (width length height : ℝ) : ℝ := width * length * height

/-- Theorem: The volume of a box with dimensions 9 cm × 4 cm × 7 cm is 252 cm³ -/
theorem gift_box_volume :
  box_volume 9 4 7 = 252 := by
  sorry

end gift_box_volume_l2070_207087


namespace triangle_circumcircle_point_length_l2070_207058

/-- Triangle PQR with sides PQ = 39, QR = 52, PR = 25 -/
structure Triangle :=
  (PQ QR PR : ℝ)
  (PQ_pos : PQ > 0)
  (QR_pos : QR > 0)
  (PR_pos : PR > 0)

/-- S is a point on the circumcircle of triangle PQR -/
structure CircumcirclePoint (t : Triangle) :=
  (S : ℝ × ℝ)

/-- S is on the perpendicular bisector of PR, not on the same side as Q -/
def onPerpendicularBisector (t : Triangle) (p : CircumcirclePoint t) : Prop := sorry

/-- The length of PS can be expressed as a√b where a and b are positive integers -/
def PSLength (t : Triangle) (p : CircumcirclePoint t) : ℕ × ℕ := sorry

/-- b is not divisible by the square of any prime -/
def notDivisibleBySquare : ℕ → Prop := sorry

theorem triangle_circumcircle_point_length 
  (t : Triangle) 
  (h1 : t.PQ = 39 ∧ t.QR = 52 ∧ t.PR = 25) 
  (p : CircumcirclePoint t) 
  (h2 : onPerpendicularBisector t p) 
  (h3 : let (a, b) := PSLength t p; notDivisibleBySquare b) : 
  let (a, b) := PSLength t p
  (a : ℕ) + Real.sqrt b = 54 := by sorry

end triangle_circumcircle_point_length_l2070_207058


namespace janet_return_time_l2070_207025

/-- Represents the walking pattern of Janet in a grid system -/
structure WalkingPattern where
  north : ℕ
  west : ℕ
  south : ℕ
  east : ℕ

/-- Calculates the time taken to return home given a walking pattern and speed -/
def timeToReturnHome (pattern : WalkingPattern) (speed : ℕ) : ℕ :=
  let net_north := pattern.south - pattern.north
  let net_west := pattern.west - pattern.east
  let total_blocks := net_north + net_west
  total_blocks / speed

/-- Janet's specific walking pattern -/
def janetsPattern : WalkingPattern :=
  { north := 3
  , west := 7 * 3
  , south := 8
  , east := 2 * 8 }

/-- Janet's walking speed in blocks per minute -/
def janetsSpeed : ℕ := 2

/-- Theorem stating that it takes Janet 5 minutes to return home -/
theorem janet_return_time :
  timeToReturnHome janetsPattern janetsSpeed = 5 := by
  sorry

end janet_return_time_l2070_207025


namespace unique_increasing_matrix_l2070_207091

/-- A 4x4 matrix with entries from 1 to 16 in increasing order -/
def IncreasingMatrix : Type := Matrix (Fin 4) (Fin 4) (Fin 16)

/-- The property that all entries in the matrix are unique and from 1 to 16 -/
def has_all_entries (M : IncreasingMatrix) : Prop :=
  ∀ i : Fin 16, ∃! (row col : Fin 4), M row col = i

/-- The property that each row is in strictly increasing order -/
def rows_increasing (M : IncreasingMatrix) : Prop :=
  ∀ row col₁ col₂, col₁ < col₂ → M row col₁ < M row col₂

/-- The property that each column is in strictly increasing order -/
def cols_increasing (M : IncreasingMatrix) : Prop :=
  ∀ row₁ row₂ col, row₁ < row₂ → M row₁ col < M row₂ col

/-- The main theorem stating there is exactly one matrix satisfying all conditions -/
theorem unique_increasing_matrix :
  ∃! M : IncreasingMatrix,
    has_all_entries M ∧
    rows_increasing M ∧
    cols_increasing M :=
  sorry

end unique_increasing_matrix_l2070_207091


namespace complex_fraction_equality_l2070_207051

theorem complex_fraction_equality : (1 - I) / (2 - I) = 3/5 - (1/5) * I := by
  sorry

end complex_fraction_equality_l2070_207051


namespace eugene_model_house_l2070_207050

theorem eugene_model_house (cards_per_deck : ℕ) (unused_cards : ℕ) (toothpicks_per_card : ℕ) (toothpicks_per_box : ℕ) :
  cards_per_deck = 52 →
  unused_cards = 16 →
  toothpicks_per_card = 75 →
  toothpicks_per_box = 450 →
  toothpicks_per_box = 450 :=
by
  sorry

end eugene_model_house_l2070_207050


namespace henry_final_book_count_l2070_207056

def initial_books : ℕ := 99
def boxes_donated : ℕ := 3
def books_per_box : ℕ := 15
def room_books : ℕ := 21
def coffee_table_books : ℕ := 4
def kitchen_books : ℕ := 18
def free_books_taken : ℕ := 12

theorem henry_final_book_count :
  initial_books - 
  (boxes_donated * books_per_box + room_books + coffee_table_books + kitchen_books) + 
  free_books_taken = 23 := by
  sorry

end henry_final_book_count_l2070_207056


namespace h_range_proof_range_sum_l2070_207094

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

theorem h_range_proof :
  ∀ x : ℝ, h x > 0 ∧
  (∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, |x| > N → h x < ε) ∧
  (∀ x : ℝ, h x ≤ h 0) →
  Set.range h = Set.Ioo 0 1 := by
sorry

theorem range_sum :
  Set.range h = Set.Ioo 0 1 →
  0 + 1 = 1 := by
sorry

end h_range_proof_range_sum_l2070_207094


namespace consecutive_numbers_with_special_properties_l2070_207020

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem consecutive_numbers_with_special_properties :
  ∃ (n : ℕ), sum_of_digits n = 8 ∧ (n + 1) % 8 = 0 :=
by
  use 71
  sorry

#eval sum_of_digits 71  -- Should output 8
#eval (72 % 8)  -- Should output 0

end consecutive_numbers_with_special_properties_l2070_207020


namespace fraction_evaluation_l2070_207066

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end fraction_evaluation_l2070_207066
