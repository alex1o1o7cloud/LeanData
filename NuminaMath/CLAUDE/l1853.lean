import Mathlib

namespace not_perfect_square_with_digit_sum_2006_l1853_185356

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem not_perfect_square_with_digit_sum_2006 (n : ℕ) 
  (h : sum_of_digits n = 2006) : 
  ¬ ∃ (m : ℕ), n = m^2 := by
  sorry

end not_perfect_square_with_digit_sum_2006_l1853_185356


namespace triangle_division_exists_l1853_185334

/-- Represents a part of the triangle -/
structure TrianglePart where
  numbers : List Nat
  sum : Nat

/-- Represents the entire triangle -/
structure Triangle where
  total_sum : Nat
  parts : List TrianglePart

/-- Checks if a triangle is valid according to the problem conditions -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.total_sum = 63 ∧
  t.parts.length = 3 ∧
  (∀ p ∈ t.parts, p.sum = p.numbers.sum) ∧
  (∀ p ∈ t.parts, p.sum = t.total_sum / 3) ∧
  (t.parts.map (·.numbers)).join.sum = t.total_sum

theorem triangle_division_exists :
  ∃ t : Triangle, is_valid_triangle t :=
sorry

end triangle_division_exists_l1853_185334


namespace circle_condition_l1853_185331

/-- The equation x^2 + y^2 - 2x + m = 0 represents a circle if and only if m < 1 -/
theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 - 2*x + m = 0 ∧ ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - 2*x + m = 0) ↔ 
  m < 1 := by sorry

end circle_condition_l1853_185331


namespace P_no_negative_roots_l1853_185341

/-- The polynomial P(x) = x^4 - 5x^3 + 3x^2 - 7x + 1 -/
def P (x : ℝ) : ℝ := x^4 - 5*x^3 + 3*x^2 - 7*x + 1

/-- Theorem: The polynomial P(x) has no negative roots -/
theorem P_no_negative_roots : ∀ x : ℝ, x < 0 → P x ≠ 0 := by
  sorry

end P_no_negative_roots_l1853_185341


namespace angle_q_approx_77_14_l1853_185372

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  /-- Angle P in degrees -/
  angleP : ℝ
  /-- Angle Q in degrees -/
  angleQ : ℝ
  /-- Angle R in degrees -/
  angleR : ℝ
  /-- The sum of all angles is 180° -/
  angle_sum : angleP + angleQ + angleR = 180
  /-- Angles Q and R are congruent -/
  qr_congruent : angleQ = angleR
  /-- Angle R is three times angle P -/
  r_triple_p : angleR = 3 * angleP

/-- The measure of angle Q in the isosceles triangle -/
def angle_q_measure (t : IsoscelesTriangle) : ℝ := t.angleQ

/-- Theorem: The measure of angle Q is approximately 77.14° -/
theorem angle_q_approx_77_14 (t : IsoscelesTriangle) :
  abs (angle_q_measure t - 540 / 7) < 0.01 := by
  sorry

#eval (540 : ℚ) / 7

end angle_q_approx_77_14_l1853_185372


namespace symmetric_points_difference_l1853_185384

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_about_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetric_points_difference (a b : ℝ) :
  symmetric_about_x_axis (1, a) (b, 2) → a - b = -3 := by
  sorry

end symmetric_points_difference_l1853_185384


namespace imaginary_part_of_complex_division_l1853_185380

theorem imaginary_part_of_complex_division :
  let i : ℂ := Complex.I
  let z₁ : ℂ := 1 + i
  let z₂ : ℂ := 1 - i
  (z₁ / z₂).im = 1 := by sorry

end imaginary_part_of_complex_division_l1853_185380


namespace teaching_arrangements_l1853_185391

-- Define the number of classes
def num_classes : ℕ := 4

-- Define the number of Chinese teachers
def num_chinese_teachers : ℕ := 2

-- Define the number of math teachers
def num_math_teachers : ℕ := 2

-- Define the number of classes each teacher teaches
def classes_per_teacher : ℕ := 2

-- Theorem statement
theorem teaching_arrangements :
  (Nat.choose num_classes classes_per_teacher) * (Nat.choose num_classes classes_per_teacher) = 36 := by
  sorry

end teaching_arrangements_l1853_185391


namespace b_2017_value_l1853_185301

/-- Given sequences a and b with the specified properties, b₂₀₁₇ equals 2016/2017 -/
theorem b_2017_value (a b : ℕ → ℚ) : 
  (b 1 = 0) →
  (∀ n : ℕ, n ≥ 1 → a n = 1 / (n * (n + 1))) →
  (∀ n : ℕ, n ≥ 2 → b n = b (n - 1) + a (n - 1)) →
  b 2017 = 2016 / 2017 := by
  sorry

end b_2017_value_l1853_185301


namespace complex_equation_solutions_l1853_185383

theorem complex_equation_solutions :
  let f : ℂ → ℂ := λ z => (z^4 - 1) / (z^3 + z^2 - 2*z)
  ∃! (S : Finset ℂ), S.card = 3 ∧ ∀ z ∈ S, f z = 0 ∧ ∀ z ∉ S, f z ≠ 0 :=
by sorry

end complex_equation_solutions_l1853_185383


namespace income_comparison_l1853_185395

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * (1 - 0.4))
  (h2 : mary = tim * (1 + 0.4)) :
  mary = juan * 0.84 := by
sorry

end income_comparison_l1853_185395


namespace no_equal_factorial_and_even_factorial_l1853_185376

theorem no_equal_factorial_and_even_factorial :
  ¬ ∃ (n m : ℕ), n.factorial = 2^m * m.factorial ∧ m ≥ 2 := by
  sorry

end no_equal_factorial_and_even_factorial_l1853_185376


namespace max_self_intersections_l1853_185346

/-- A closed six-segment broken line with vertices on a circle -/
structure BrokenLine where
  vertices : Fin 6 → ℝ × ℝ
  on_circle : ∀ i, (vertices i).1^2 + (vertices i).2^2 = 1

/-- The number of self-intersections in a broken line -/
def num_self_intersections (bl : BrokenLine) : ℕ := sorry

/-- Theorem: The maximum number of self-intersections is 7 -/
theorem max_self_intersections (bl : BrokenLine) :
  num_self_intersections bl ≤ 7 := by sorry

end max_self_intersections_l1853_185346


namespace altered_solution_detergent_volume_l1853_185339

/-- Given a cleaning solution with initial ratio of bleach:detergent:disinfectant:water as 2:40:10:100,
    and after altering the solution such that:
    1) The ratio of bleach to detergent is tripled
    2) The ratio of detergent to water is halved
    3) The ratio of disinfectant to bleach is doubled
    If the altered solution contains 300 liters of water, prove that it contains 60 liters of detergent. -/
theorem altered_solution_detergent_volume (b d f w : ℚ) : 
  b / d = 2 / 40 →
  d / w = 40 / 100 →
  f / b = 10 / 2 →
  (3 * b) / d = 3 * (2 / 40) →
  d / w = (1 / 2) * (40 / 100) →
  f / (3 * b) = 2 * (10 / 2) →
  w = 300 →
  d = 60 := by
sorry

end altered_solution_detergent_volume_l1853_185339


namespace brick_breadth_is_10cm_l1853_185373

/-- Prove that the breadth of a brick is 10 cm given the specified conditions -/
theorem brick_breadth_is_10cm 
  (courtyard_length : ℝ) 
  (courtyard_width : ℝ) 
  (brick_length : ℝ) 
  (total_bricks : ℕ) 
  (h1 : courtyard_length = 20) 
  (h2 : courtyard_width = 16) 
  (h3 : brick_length = 0.2) 
  (h4 : total_bricks = 16000) : 
  ∃ (brick_width : ℝ), brick_width = 0.1 ∧ 
    courtyard_length * courtyard_width = (brick_length * brick_width) * total_bricks :=
by sorry

end brick_breadth_is_10cm_l1853_185373


namespace broken_line_endpoint_characterization_l1853_185310

/-- A broken line from O to M -/
structure BrokenLine where
  segments : List (ℝ × ℝ)
  start_at_origin : segments.foldl (λ acc (x, y) => (acc.1 + x, acc.2 + y)) (0, 0) = (0, 0)
  unit_length : segments.foldl (λ acc (x, y) => acc + x^2 + y^2) 0 = 1

/-- Predicate to check if a broken line satisfies the intersection condition -/
def satisfies_intersection_condition (l : BrokenLine) : Prop :=
  ∀ (a b : ℝ), (∀ (x y : ℝ), (x, y) ∈ l.segments → (a * x + b * y ≠ 0 ∨ a * x + b * y ≠ 1))

theorem broken_line_endpoint_characterization (x y : ℝ) :
  (∃ (l : BrokenLine), satisfies_intersection_condition l ∧ 
   l.segments.foldl (λ acc (dx, dy) => (acc.1 + dx, acc.2 + dy)) (0, 0) = (x, y)) →
  x^2 + y^2 ≤ 1 ∧ |x| + |y| ≥ 1 := by
  sorry

end broken_line_endpoint_characterization_l1853_185310


namespace impossible_transformation_l1853_185389

/-- Represents a binary sequence -/
inductive BinarySeq
| empty : BinarySeq
| cons : Bool → BinarySeq → BinarySeq

/-- Represents the color of a digit in the sequence -/
inductive Color
| Red
| Green
| Blue

/-- Assigns colors to a binary sequence -/
def colorSequence : BinarySeq → List Color
| BinarySeq.empty => []
| BinarySeq.cons _ rest => [Color.Red, Color.Green, Color.Blue] ++ colorSequence rest

/-- Counts the number of red 1s in a colored binary sequence -/
def countRed1s : BinarySeq → Nat
| BinarySeq.empty => 0
| BinarySeq.cons true (BinarySeq.cons _ (BinarySeq.cons _ rest)) => 1 + countRed1s rest
| BinarySeq.cons false (BinarySeq.cons _ (BinarySeq.cons _ rest)) => countRed1s rest
| _ => 0

/-- Represents an operation on the binary sequence -/
inductive Operation
| Insert : BinarySeq → Operation
| Delete : BinarySeq → Operation

/-- Applies an operation to a binary sequence -/
def applyOperation : BinarySeq → Operation → BinarySeq := sorry

/-- Theorem: It's impossible to transform "10" into "01" using the allowed operations -/
theorem impossible_transformation :
  ∀ (ops : List Operation),
    let initial := BinarySeq.cons true (BinarySeq.cons false BinarySeq.empty)
    let final := BinarySeq.cons false (BinarySeq.cons true BinarySeq.empty)
    let result := ops.foldl applyOperation initial
    result ≠ final :=
sorry

end impossible_transformation_l1853_185389


namespace quadratic_shift_l1853_185302

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := 2 * x^2

-- Define the vertical shift
def vertical_shift : ℝ := 1

-- Define the shifted function
def shifted_function (x : ℝ) : ℝ := original_function x + vertical_shift

-- Theorem statement
theorem quadratic_shift :
  ∀ x : ℝ, shifted_function x = 2 * x^2 + 1 :=
by sorry

end quadratic_shift_l1853_185302


namespace aunt_angela_nephews_l1853_185345

theorem aunt_angela_nephews (total_jellybeans : ℕ) (jellybeans_per_child : ℕ) (num_nieces : ℕ) :
  total_jellybeans = 70 →
  jellybeans_per_child = 14 →
  num_nieces = 2 →
  total_jellybeans = (num_nieces + 3) * jellybeans_per_child :=
by sorry

end aunt_angela_nephews_l1853_185345


namespace z_in_fourth_quadrant_l1853_185354

/-- Given a complex number z satisfying (1+2i)z=4+3i, prove that z is located in the fourth quadrant of the complex plane -/
theorem z_in_fourth_quadrant (z : ℂ) (h : (1 + 2*Complex.I)*z = 4 + 3*Complex.I) : 
  Complex.re z > 0 ∧ Complex.im z < 0 := by
  sorry

end z_in_fourth_quadrant_l1853_185354


namespace sphere_deflation_radius_l1853_185332

theorem sphere_deflation_radius (r : ℝ) (h : r = 4) :
  let hemisphere_volume := (2/3) * Real.pi * r^3
  let original_sphere_volume := (4/3) * Real.pi * (((4 * Real.rpow 2 (1/3)) / Real.rpow 3 (1/3))^3)
  hemisphere_volume = (3/4) * original_sphere_volume :=
by sorry

end sphere_deflation_radius_l1853_185332


namespace min_socks_for_pairs_l1853_185325

/-- Represents a sock with a color -/
inductive Sock
| Blue
| Red

/-- Represents a drawer containing socks -/
structure Drawer where
  socks : List Sock
  blue_count : Nat
  red_count : Nat
  balanced : blue_count = red_count

/-- Checks if a list of socks contains a pair of the same color -/
def hasSameColorPair (socks : List Sock) : Bool :=
  sorry

/-- Checks if a list of socks contains a pair of different colors -/
def hasDifferentColorPair (socks : List Sock) : Bool :=
  sorry

/-- Theorem stating the minimum number of socks required -/
theorem min_socks_for_pairs (d : Drawer) :
  (∀ n : Nat, n < 4 → ¬(∀ subset : List Sock, subset.length = n →
    (hasSameColorPair subset ∧ hasDifferentColorPair subset))) ∧
  (∃ subset : List Sock, subset.length = 4 ∧
    (hasSameColorPair subset ∧ hasDifferentColorPair subset)) :=
  sorry

end min_socks_for_pairs_l1853_185325


namespace max_lateral_surface_area_l1853_185399

/-- A right prism ABCD-A₁B₁C₁D₁ inscribed in a sphere O -/
structure InscribedPrism where
  /-- The base edge length of the prism -/
  a : ℝ
  /-- The height of the prism -/
  h : ℝ
  /-- The radius of the sphere -/
  r : ℝ
  /-- The surface area of the sphere is 12π -/
  sphere_area : 4 * π * r^2 = 12 * π
  /-- The prism is inscribed in the sphere -/
  inscribed : 2 * a^2 + h^2 = 4 * r^2

/-- The lateral surface area of the prism -/
def lateralSurfaceArea (p : InscribedPrism) : ℝ := 4 * p.a * p.h

/-- The theorem stating the maximum lateral surface area of the inscribed prism -/
theorem max_lateral_surface_area (p : InscribedPrism) : 
  lateralSurfaceArea p ≤ 12 * Real.sqrt 2 := by
  sorry

end max_lateral_surface_area_l1853_185399


namespace cos_sixty_degrees_l1853_185317

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end cos_sixty_degrees_l1853_185317


namespace g_derivative_at_one_l1853_185382

/-- The sequence of functions gₖ(x) -/
noncomputable def g : ℕ → (ℝ → ℝ)
| 0 => λ x => x^2 / (2 - x)
| (k+1) => λ x => x * g k x / (2 - g k x)

/-- The statement to be proved -/
theorem g_derivative_at_one (k : ℕ) :
  HasDerivAt (g k) (2^(k+1) - 1) 1 :=
sorry

end g_derivative_at_one_l1853_185382


namespace cheesecake_eggs_proof_l1853_185355

/-- The number of eggs needed for each chocolate cake -/
def chocolate_cake_eggs : ℕ := 3

/-- The number of eggs needed for each cheesecake -/
def cheesecake_eggs : ℕ := 8

/-- Proof that the number of eggs for each cheesecake is 8 -/
theorem cheesecake_eggs_proof : 
  9 * cheesecake_eggs = 5 * chocolate_cake_eggs + 57 :=
by sorry

end cheesecake_eggs_proof_l1853_185355


namespace only_winning_lottery_is_random_l1853_185347

-- Define the type for events
inductive Event
  | WaterBoiling
  | WinningLottery
  | AthleteRunning
  | DrawingRedBall

-- Define the property of being a random event
def isRandomEvent (e : Event) : Prop :=
  match e with
  | Event.WaterBoiling => false
  | Event.WinningLottery => true
  | Event.AthleteRunning => false
  | Event.DrawingRedBall => false

-- Theorem statement
theorem only_winning_lottery_is_random :
  ∀ e : Event, isRandomEvent e ↔ e = Event.WinningLottery :=
sorry

end only_winning_lottery_is_random_l1853_185347


namespace stone_slab_length_l1853_185348

theorem stone_slab_length (num_slabs : ℕ) (total_area : ℝ) (slab_length : ℝ) :
  num_slabs = 50 →
  total_area = 98 →
  num_slabs * (slab_length ^ 2) = total_area →
  slab_length = 1.4 :=
by
  sorry

#check stone_slab_length

end stone_slab_length_l1853_185348


namespace sufficient_not_necessary_l1853_185394

def A (m : ℝ) : Set ℝ := {2, m^2}
def B : Set ℝ := {0, 1, 3}

theorem sufficient_not_necessary :
  (∀ m : ℝ, m = 1 → A m ∩ B = {1}) ∧
  (∃ m : ℝ, m ≠ 1 ∧ A m ∩ B = {1}) :=
by sorry

end sufficient_not_necessary_l1853_185394


namespace polygonal_chain_existence_l1853_185352

/-- A type representing a line in a plane -/
structure Line where
  -- Add necessary fields here
  mk :: -- Add constructor parameters here

/-- A type representing a point in a plane -/
structure Point where
  -- Add necessary fields here
  mk :: -- Add constructor parameters here

/-- A type representing a polygonal chain -/
structure PolygonalChain (n : ℕ) where
  vertices : Fin (n + 1) → Point
  segments : Fin n → Line

/-- Predicate to check if a polygonal chain is non-self-intersecting -/
def is_non_self_intersecting (chain : PolygonalChain n) : Prop :=
  sorry

/-- Predicate to check if each segment of a polygonal chain lies on a unique line -/
def segments_on_unique_lines (chain : PolygonalChain n) (lines : Fin n → Line) : Prop :=
  sorry

/-- Predicate to check if no two lines are parallel -/
def no_parallel_lines (lines : Fin n → Line) : Prop :=
  sorry

/-- Predicate to check if no three lines intersect at the same point -/
def no_three_lines_intersect (lines : Fin n → Line) : Prop :=
  sorry

/-- Main theorem statement -/
theorem polygonal_chain_existence (n : ℕ) (lines : Fin n → Line) 
  (h1 : no_parallel_lines lines) 
  (h2 : no_three_lines_intersect lines) : 
  ∃ (chain : PolygonalChain n), 
    is_non_self_intersecting chain ∧ 
    segments_on_unique_lines chain lines :=
  sorry

end polygonal_chain_existence_l1853_185352


namespace part_one_part_two_l1853_185363

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 3*x - 10 > 0

-- Part I
theorem part_one (x : ℝ) :
  (∃ a : ℝ, a = 1 ∧ a > 0 ∧ p x a ∧ q x) → 2 < x ∧ x < 3 :=
sorry

-- Part II
theorem part_two :
  (∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x)) →
  (∃ a : ℝ, 1 < a ∧ a ≤ 2) :=
sorry

end part_one_part_two_l1853_185363


namespace convex_polygon_in_rectangle_l1853_185316

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields and properties to define a convex polygon
  is_convex : Bool  -- Placeholder for convexity property

/-- A rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a shape -/
class HasArea (α : Type) where
  area : α → ℝ

/-- Instance for ConvexPolygon -/
instance : HasArea ConvexPolygon where
  area := sorry

/-- Instance for Rectangle -/
instance : HasArea Rectangle where
  area r := r.width * r.height

/-- A polygon is contained in a rectangle -/
def ContainedIn (p : ConvexPolygon) (r : Rectangle) : Prop :=
  sorry  -- Definition of containment

theorem convex_polygon_in_rectangle :
  ∀ (p : ConvexPolygon), HasArea.area p = 1 →
  ∃ (r : Rectangle), ContainedIn p r ∧ HasArea.area r ≤ 2 :=
sorry

end convex_polygon_in_rectangle_l1853_185316


namespace david_twice_rosy_age_l1853_185323

/-- Represents the current age of Rosy -/
def rosy_age : ℕ := 8

/-- Represents the current age difference between David and Rosy -/
def age_difference : ℕ := 12

/-- Calculates the number of years until David is twice Rosy's age -/
def years_until_double : ℕ :=
  let david_age := rosy_age + age_difference
  (david_age - 2 * rosy_age)

theorem david_twice_rosy_age : years_until_double = 4 := by
  sorry

end david_twice_rosy_age_l1853_185323


namespace davids_math_marks_l1853_185336

theorem davids_math_marks (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) 
  (h1 : english = 76)
  (h2 : physics = 82)
  (h3 : chemistry = 67)
  (h4 : biology = 85)
  (h5 : average = 75)
  (h6 : (english + physics + chemistry + biology + mathematics) / 5 = average) :
  mathematics = 65 := by
  sorry

#check davids_math_marks

end davids_math_marks_l1853_185336


namespace equation_solution_l1853_185309

theorem equation_solution : 
  {x : ℝ | x * (x - 14) = 0} = {0, 14} := by sorry

end equation_solution_l1853_185309


namespace min_c_value_l1853_185360

theorem min_c_value (a b c k : ℕ+) (h1 : b = a + k) (h2 : c = a + 2*k) 
  (h3 : a < b ∧ b < c) 
  (h4 : ∃! (x y : ℝ), 3*x + y = 3005 ∧ y = |x - a| + |x - (a + k)| + |x - (a + 2*k)|) :
  c ≥ 6005 ∧ ∃ (a₀ b₀ c₀ k₀ : ℕ+), 
    b₀ = a₀ + k₀ ∧ c₀ = a₀ + 2*k₀ ∧ a₀ < b₀ ∧ b₀ < c₀ ∧ c₀ = 6005 ∧
    ∃! (x y : ℝ), 3*x + y = 3005 ∧ y = |x - a₀| + |x - (a₀ + k₀)| + |x - (a₀ + 2*k₀)| :=
by sorry

end min_c_value_l1853_185360


namespace rectangle_perimeter_l1853_185312

/-- Given a square with perimeter 80 units divided into two congruent rectangles
    by a horizontal line, prove that the perimeter of one rectangle is 60 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 80) :
  let square_side := square_perimeter / 4
  let rectangle_width := square_side
  let rectangle_height := square_side / 2
  2 * (rectangle_width + rectangle_height) = 60 :=
by sorry

end rectangle_perimeter_l1853_185312


namespace stability_comparison_lower_variance_more_stable_shooting_competition_result_l1853_185313

/-- Represents a shooter in the competition -/
structure Shooter where
  name : String
  variance : ℝ

/-- Defines the stability of a shooter based on their variance -/
def moreStable (a b : Shooter) : Prop :=
  a.variance < b.variance

theorem stability_comparison (a b : Shooter) 
  (h : a.variance ≠ b.variance) : 
  moreStable a b ∨ moreStable b a :=
sorry

theorem lower_variance_more_stable (a b : Shooter) 
  (h : a.variance < b.variance) : 
  moreStable a b :=
sorry

theorem shooting_competition_result (a b : Shooter)
  (ha : a.name = "A" ∧ a.variance = 0.25)
  (hb : b.name = "B" ∧ b.variance = 0.12) :
  moreStable b a :=
sorry

end stability_comparison_lower_variance_more_stable_shooting_competition_result_l1853_185313


namespace scientific_notation_correct_l1853_185337

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be represented -/
def original_number : ℕ := 384000

/-- The scientific notation representation -/
def scientific_rep : ScientificNotation :=
  { coefficient := 3.84
    exponent := 5
    coeff_range := by sorry }

theorem scientific_notation_correct :
  (scientific_rep.coefficient * (10 : ℝ) ^ scientific_rep.exponent) = original_number := by
  sorry

end scientific_notation_correct_l1853_185337


namespace polynomial_equality_l1853_185375

theorem polynomial_equality (x : ℝ) (h : 2 * x^2 - x = 1) :
  4 * x^4 - 4 * x^3 + 3 * x^2 - x - 1 = 1 := by
  sorry

end polynomial_equality_l1853_185375


namespace midpoint_distance_to_y_axis_l1853_185357

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the focus
def line_through_focus (m b : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = m * p.1 + b ∧ focus.2 = m * focus.1 + b

-- Define the intersection points of the line and the parabola
def intersection_points (m b : ℝ) : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ line_through_focus m b p}

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (m b : ℝ) 
  (M N : ℝ × ℝ) 
  (h_M : M ∈ intersection_points m b) 
  (h_N : N ∈ intersection_points m b) 
  (h_distinct : M ≠ N) :
  let midpoint := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  midpoint.1 = 2 := by sorry

end midpoint_distance_to_y_axis_l1853_185357


namespace product_of_three_numbers_l1853_185338

theorem product_of_three_numbers (a b c m : ℚ) : 
  a + b + c = 200 ∧ 
  8 * a = m ∧ 
  b = m + 10 ∧ 
  c = m - 10 →
  a * b * c = 505860000 / 4913 := by
  sorry

end product_of_three_numbers_l1853_185338


namespace pet_store_cages_l1853_185362

theorem pet_store_cages (initial_puppies : Nat) (sold_puppies : Nat) (puppies_per_cage : Nat) : 
  initial_puppies = 18 → sold_puppies = 3 → puppies_per_cage = 5 → 
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := by
  sorry

end pet_store_cages_l1853_185362


namespace sum_of_roots_inequality_l1853_185342

theorem sum_of_roots_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq_one : a + b + c = 1) :
  Real.sqrt ((1 / a) - 1) * Real.sqrt ((1 / b) - 1) +
  Real.sqrt ((1 / b) - 1) * Real.sqrt ((1 / c) - 1) +
  Real.sqrt ((1 / c) - 1) * Real.sqrt ((1 / a) - 1) ≥ 6 := by
  sorry

end sum_of_roots_inequality_l1853_185342


namespace M_equals_P_l1853_185386

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x, y = x^2 - 1}
def P : Set ℝ := {a | ∃ b, a = b^2 - 1}

-- Theorem statement
theorem M_equals_P : M = P := by
  sorry

end M_equals_P_l1853_185386


namespace saline_solution_concentration_l1853_185306

/-- Proves that given a tank with 100 gallons of pure water and 66.67 gallons of saline solution
    added to create a 10% salt solution, the original saline solution must have contained 25% salt. -/
theorem saline_solution_concentration
  (pure_water : ℝ)
  (saline_added : ℝ)
  (final_concentration : ℝ)
  (h1 : pure_water = 100)
  (h2 : saline_added = 66.67)
  (h3 : final_concentration = 0.1)
  : (final_concentration * (pure_water + saline_added)) / saline_added = 0.25 := by
  sorry

end saline_solution_concentration_l1853_185306


namespace right_triangle_area_l1853_185398

/-- The area of a right-angled triangle can be expressed in terms of its hypotenuse and one of its acute angles. -/
theorem right_triangle_area (c α : ℝ) (h_c : c > 0) (h_α : 0 < α ∧ α < π / 2) :
  let t := (1 / 4) * c^2 * Real.sin (2 * α)
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 ∧ (1 / 2) * a * b = t :=
sorry

end right_triangle_area_l1853_185398


namespace domain_of_f_x_squared_l1853_185385

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-1) 3

-- State the theorem
theorem domain_of_f_x_squared 
  (h : ∀ x, x ∈ domain_f_x_plus_1 ↔ f (x + 1) ∈ Set.range f) : 
  (∀ x, f (x^2) ∈ Set.range f ↔ x ∈ Set.Icc (-2) 2) := by
  sorry

end domain_of_f_x_squared_l1853_185385


namespace absolute_value_properties_l1853_185367

theorem absolute_value_properties :
  (∀ a : ℚ, a = 5 → |a| / a = 1) ∧
  (∀ a : ℚ, a = -2 → a / |a| = -1) ∧
  (∀ a b : ℚ, a * b > 0 → a / |a| + |b| / b = 2 ∨ a / |a| + |b| / b = -2) ∧
  (∀ a b c : ℚ, a * b * c < 0 → 
    a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = 0 ∨
    a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| = -4) :=
by sorry

end absolute_value_properties_l1853_185367


namespace skew_iff_a_neq_zero_l1853_185304

def line1 (a t : ℝ) : Fin 4 → ℝ := fun i =>
  match i with
  | 0 => 1 + 2*t
  | 1 => 3 + 4*t
  | 2 => 0 + 1*t
  | 3 => a + 3*t

def line2 (u : ℝ) : Fin 4 → ℝ := fun i =>
  match i with
  | 0 => 3 + 4*u
  | 1 => 4 + 5*u
  | 2 => 1 + 2*u
  | 3 => 0 + 1*u

def are_skew (a : ℝ) : Prop :=
  ∀ t u : ℝ, line1 a t ≠ line2 u

theorem skew_iff_a_neq_zero (a : ℝ) :
  are_skew a ↔ a ≠ 0 := by sorry

end skew_iff_a_neq_zero_l1853_185304


namespace range_of_a_l1853_185364

-- Define the proposition
def proposition (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ x^2 + 2*x - a ≥ 0

-- State the theorem
theorem range_of_a (h : ∀ a : ℝ, proposition a ↔ a ∈ Set.Iic 15) :
  {a : ℝ | proposition a} = Set.Iic 15 := by sorry

end range_of_a_l1853_185364


namespace roots_sum_powers_l1853_185321

theorem roots_sum_powers (α β : ℝ) : 
  α^2 - 4*α + 1 = 0 → β^2 - 4*β + 1 = 0 → 7*α^3 + 3*β^4 = 1019 := by
sorry

end roots_sum_powers_l1853_185321


namespace problem_solution_l1853_185324

theorem problem_solution (x : ℝ) : 
  x + Real.sqrt (x^2 - 1) + 1 / (x - Real.sqrt (x^2 - 1)) = 24 →
  x^2 + Real.sqrt (x^4 - 1) + 1 / (x^2 + Real.sqrt (x^4 - 1)) = 10525 / 144 :=
by sorry

end problem_solution_l1853_185324


namespace expression_value_l1853_185328

theorem expression_value (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 27665/27 := by
  sorry

end expression_value_l1853_185328


namespace star_equality_implies_x_equals_six_l1853_185322

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a + c, b - d)

theorem star_equality_implies_x_equals_six :
  ∀ x y : ℤ, star 5 5 2 2 = star x y 1 3 → x = 6 := by
  sorry

end star_equality_implies_x_equals_six_l1853_185322


namespace multiple_without_zero_l1853_185303

/-- A function that checks if a natural number contains the digit 0 in its decimal representation -/
def containsZero (n : ℕ) : Prop :=
  ∃ (k : ℕ), n % (10^(k+1)) / (10^k) = 0

theorem multiple_without_zero (n : ℕ) (h : n % 10 ≠ 0) :
  ∃ (k : ℕ), k % n = 0 ∧ ¬containsZero k := by
  sorry

end multiple_without_zero_l1853_185303


namespace yellow_score_mixture_l1853_185330

theorem yellow_score_mixture (white black : ℕ) : 
  white * 6 = black * 7 →
  2 * (white - black) = 3 * 4 →
  white + black = 78 := by
sorry

end yellow_score_mixture_l1853_185330


namespace ratio_difference_bound_l1853_185370

theorem ratio_difference_bound (a : Fin 5 → ℝ) (h : ∀ i, 0 < a i) :
  ∃ i j k l : Fin 5, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    |a i / a j - a k / a l| < (1 : ℝ) / 2 := by
  sorry

end ratio_difference_bound_l1853_185370


namespace cleo_final_marbles_l1853_185369

def initial_marbles : ℕ := 240

def day2_fraction : ℚ := 2/3
def day2_people : ℕ := 3

def day3_fraction : ℚ := 3/5
def day3_people : ℕ := 2

def day4_cleo_fraction : ℚ := 7/8
def day4_estela_fraction : ℚ := 1/4

theorem cleo_final_marbles :
  let day2_marbles := (initial_marbles : ℚ) * day2_fraction
  let day2_per_person := ⌊day2_marbles / day2_people⌋
  let day3_remaining := initial_marbles - (day2_per_person * day2_people)
  let day3_marbles := (day3_remaining : ℚ) * day3_fraction
  let day3_cleo := ⌊day3_marbles / day3_people⌋
  let day4_cleo := ⌊(day3_cleo : ℚ) * day4_cleo_fraction⌋
  let day4_estela := ⌊(day4_cleo : ℚ) * day4_estela_fraction⌋
  day4_cleo - day4_estela = 16 := by sorry

end cleo_final_marbles_l1853_185369


namespace cylinder_lateral_surface_area_l1853_185329

/-- Given a cylinder with base area S whose lateral surface unfolds into a square,
    prove that its lateral surface area is 4πS. -/
theorem cylinder_lateral_surface_area (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / Real.pi)
  let h := 2 * Real.pi * r
  (h = 2 * r)  → 2 * Real.pi * r * h = 4 * Real.pi * S :=
by
  sorry

end cylinder_lateral_surface_area_l1853_185329


namespace riverton_soccer_team_l1853_185349

theorem riverton_soccer_team (total_players : ℕ) (math_players : ℕ) (both_players : ℕ) :
  total_players = 15 →
  math_players = 9 →
  both_players = 3 →
  math_players + (total_players - math_players) ≥ total_players →
  total_players - math_players + both_players = 9 :=
by sorry

end riverton_soccer_team_l1853_185349


namespace quadratic_inequality_l1853_185377

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x + 2 < 1 ↔ 1 < x ∧ x < 2 := by
  sorry

end quadratic_inequality_l1853_185377


namespace rectangle_image_is_curved_region_l1853_185351

-- Define the rectangle OAPB
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (0, 3)

-- Define the transformation
def u (x y : ℝ) : ℝ := x^2 - y^2
def v (x y : ℝ) : ℝ := x * y

-- Define the image of a point under the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ := (u p.1 p.2, v p.1 p.2)

-- Theorem statement
theorem rectangle_image_is_curved_region :
  ∃ (R : Set (ℝ × ℝ)), 
    (∀ p ∈ R, ∃ q ∈ Set.Icc O A ∪ Set.Icc A P ∪ Set.Icc P B ∪ Set.Icc B O, p = transform q) ∧
    (∀ q ∈ Set.Icc O A ∪ Set.Icc A P ∪ Set.Icc P B ∪ Set.Icc B O, transform q ∈ R) ∧
    (∃ f g : ℝ → ℝ, Continuous f ∧ Continuous g ∧ 
      (∀ t ∈ Set.Icc 0 1, (f t, g t) ∈ R) ∧
      (f 0, g 0) = transform O ∧ (f 1, g 1) = transform A) ∧
    (∃ f g : ℝ → ℝ, Continuous f ∧ Continuous g ∧ 
      (∀ t ∈ Set.Icc 0 1, (f t, g t) ∈ R) ∧
      (f 0, g 0) = transform A ∧ (f 1, g 1) = transform P) ∧
    (∃ f g : ℝ → ℝ, Continuous f ∧ Continuous g ∧ 
      (∀ t ∈ Set.Icc 0 1, (f t, g t) ∈ R) ∧
      (f 0, g 0) = transform P ∧ (f 1, g 1) = transform B) ∧
    (∃ f g : ℝ → ℝ, Continuous f ∧ Continuous g ∧ 
      (∀ t ∈ Set.Icc 0 1, (f t, g t) ∈ R) ∧
      (f 0, g 0) = transform B ∧ (f 1, g 1) = transform O) :=
sorry

end rectangle_image_is_curved_region_l1853_185351


namespace complex_power_series_sum_l1853_185393

def complex_power_sequence (n : ℕ) : ℂ := (2 + Complex.I) ^ n

def real_part_sequence (n : ℕ) : ℝ := (complex_power_sequence n).re
def imag_part_sequence (n : ℕ) : ℝ := (complex_power_sequence n).im

theorem complex_power_series_sum :
  (∑' n, (real_part_sequence n * imag_part_sequence n) / 7 ^ n) = 7 / 16 := by
  sorry

end complex_power_series_sum_l1853_185393


namespace sum_of_distances_constant_l1853_185300

/-- A regular tetrahedron in three-dimensional space -/
structure RegularTetrahedron where
  -- Define the properties of a regular tetrahedron here
  -- (We don't need to fully define it for this statement)

/-- A point inside a regular tetrahedron -/
structure InnerPoint (t : RegularTetrahedron) where
  -- Define the properties of an inner point here
  -- (We don't need to fully define it for this statement)

/-- The sum of distances from a point to all faces of a regular tetrahedron -/
def sum_of_distances_to_faces (t : RegularTetrahedron) (p : InnerPoint t) : ℝ :=
  sorry -- Definition would go here

/-- Theorem stating that the sum of distances from any point inside a regular tetrahedron to its faces is constant -/
theorem sum_of_distances_constant (t : RegularTetrahedron) :
  ∃ c : ℝ, ∀ p : InnerPoint t, sum_of_distances_to_faces t p = c :=
sorry

end sum_of_distances_constant_l1853_185300


namespace expression_value_l1853_185359

theorem expression_value : 
  Real.sqrt ((16^12 + 8^15) / (16^5 + 8^16)) = (3 * Real.sqrt 2) / 4 := by
sorry

end expression_value_l1853_185359


namespace joans_cake_eggs_l1853_185361

/-- The number of eggs needed for baking cakes -/
def total_eggs (vanilla_count chocolate_count carrot_count : ℕ) 
               (vanilla_eggs chocolate_eggs carrot_eggs : ℕ) : ℕ :=
  vanilla_count * vanilla_eggs + chocolate_count * chocolate_eggs + carrot_count * carrot_eggs

/-- Theorem stating the total number of eggs needed for Joan's cakes -/
theorem joans_cake_eggs : 
  total_eggs 5 4 3 8 6 10 = 94 := by
  sorry

end joans_cake_eggs_l1853_185361


namespace circle_ratio_l1853_185388

/-- For a circle with diameter 100 cm and circumference 314 cm, 
    the ratio of circumference to diameter is 3.14 -/
theorem circle_ratio : 
  ∀ (diameter circumference : ℝ), 
    diameter = 100 → 
    circumference = 314 → 
    circumference / diameter = 3.14 := by
  sorry

end circle_ratio_l1853_185388


namespace discount_ratio_proof_l1853_185371

/-- Proves that the ratio of discounts at different times is 1.833 -/
theorem discount_ratio_proof (original_bill : ℝ) (original_discount : ℝ) (longer_discount : ℝ) :
  original_bill = 110 →
  original_discount = 10 →
  longer_discount = 18.33 →
  longer_discount / original_discount = 1.833 := by
  sorry

end discount_ratio_proof_l1853_185371


namespace poster_collection_ratio_l1853_185319

theorem poster_collection_ratio : 
  let current_size : ℕ := 22
  let past_size : ℕ := 14
  let gcd := Nat.gcd current_size past_size
  (current_size / gcd) = 11 ∧ (past_size / gcd) = 7 :=
by sorry

end poster_collection_ratio_l1853_185319


namespace subtracted_number_l1853_185327

theorem subtracted_number (a b x : ℕ) : 
  (a : ℚ) / b = 6 / 5 →
  (a - x : ℚ) / (b - x) = 5 / 4 →
  a - b = 5 →
  x = 5 := by
sorry

end subtracted_number_l1853_185327


namespace art_arrangement_probability_l1853_185318

/-- The probability of arranging n items in a row, where k specific items are placed consecutively. -/
def consecutive_probability (n k : ℕ) : ℚ :=
  (Nat.factorial (n - k + 1) * Nat.factorial k) / Nat.factorial n

/-- The probability of arranging 12 items in a row, where 4 specific items are placed consecutively, is 1/55. -/
theorem art_arrangement_probability : consecutive_probability 12 4 = 1 / 55 := by
  sorry

end art_arrangement_probability_l1853_185318


namespace vector_operation_l1853_185350

/-- Given two vectors a and b in ℝ², prove that 3b - a equals the specified result. -/
theorem vector_operation (a b : ℝ × ℝ) (ha : a = (3, 2)) (hb : b = (0, -1)) :
  (3 : ℝ) • b - a = (-3, -5) := by
  sorry

end vector_operation_l1853_185350


namespace road_repair_length_l1853_185396

theorem road_repair_length : 
  ∀ (total_length : ℝ),
  (200 : ℝ) + 0.4 * total_length + 700 = total_length →
  total_length = 1500 := by
sorry

end road_repair_length_l1853_185396


namespace digit_sum_problem_l1853_185365

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number only uses specified digits -/
def uses_specified_digits (n : ℕ) : Prop := sorry

theorem digit_sum_problem (M : ℕ) 
  (h_even : Even M)
  (h_digits : uses_specified_digits M)
  (h_double : sum_of_digits (2 * M) = 35)
  (h_half : sum_of_digits (M / 2) = 29) :
  sum_of_digits M = 31 := by sorry

end digit_sum_problem_l1853_185365


namespace exact_money_for_widgets_l1853_185320

/-- If a person can buy exactly 6 items at a certain price, and exactly 8 items if the price is reduced by 10%, then the person has exactly $5 to spend. -/
theorem exact_money_for_widgets (price : ℝ) (money : ℝ) 
  (h1 : money = 6 * price) 
  (h2 : money = 8 * (0.9 * price)) : 
  money = 5 := by sorry

end exact_money_for_widgets_l1853_185320


namespace min_value_x_plus_y_l1853_185314

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2*a + 8*b - a*b = 0 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + 8*y - x*y = 0 ∧ x + y = 18 :=
by sorry

end min_value_x_plus_y_l1853_185314


namespace part_one_part_two_l1853_185397

-- Part 1
theorem part_one (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (x - 1) / (x^2 + 2*x + 1) / (1 - 2 / (x + 1)) = Real.sqrt 3 / 3 := by
  sorry

-- Part 2
theorem part_two (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 3 / 16) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = 3 / 64 := by
  sorry

end part_one_part_two_l1853_185397


namespace set_equality_implies_values_l1853_185308

/-- Given two sets A and B, prove that if they are equal and have the specified form,
    then x = 2 and y = 2 -/
theorem set_equality_implies_values (x y : ℝ) : 
  ({x, y^2, 1} : Set ℝ) = ({1, 2*x, y} : Set ℝ) → x = 2 ∧ y = 2 := by
  sorry

end set_equality_implies_values_l1853_185308


namespace sufficient_not_necessary_l1853_185392

theorem sufficient_not_necessary (x : ℝ) :
  (x > 2 → abs (x - 1) > 1) ∧ ¬(abs (x - 1) > 1 → x > 2) :=
sorry

end sufficient_not_necessary_l1853_185392


namespace unique_number_l1853_185390

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if a natural number has exactly two prime factors -/
def hasTwoPrimeFactors (n : ℕ) : Prop :=
  ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ p ≠ q ∧ n = p * q

/-- A function that checks if a number doesn't contain the digit 7 -/
def noSeven (n : ℕ) : Prop :=
  ∀ d : ℕ, d < n → (n / (10^d)) % 10 ≠ 7

theorem unique_number : 
  ∃! n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧  -- four-digit number
    noSeven n ∧             -- doesn't contain 7
    hasTwoPrimeFactors n ∧  -- product of exactly two primes
    ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ n = p * q ∧ q = p + 4 ∧  -- prime factors differ by 4
    n = 2021                -- the number is 2021
  := by sorry

end unique_number_l1853_185390


namespace defect_probability_is_22_900_l1853_185335

/-- Represents a machine in the production line -/
structure Machine where
  defectProb : ℝ
  productivityRatio : ℝ

/-- The production setup with three machines -/
def productionSetup : List Machine := [
  { defectProb := 0.02, productivityRatio := 3 },
  { defectProb := 0.03, productivityRatio := 1 },
  { defectProb := 0.04, productivityRatio := 0.5 }
]

/-- Calculates the probability of a randomly selected part being defective -/
def calculateDefectProbability (setup : List Machine) : ℝ :=
  sorry

/-- Theorem stating that the probability of a defective part is 22/900 -/
theorem defect_probability_is_22_900 :
  calculateDefectProbability productionSetup = 22 / 900 := by
  sorry

end defect_probability_is_22_900_l1853_185335


namespace complex_product_theorem_l1853_185378

theorem complex_product_theorem (z₁ z₂ : ℂ) (a b : ℝ) : 
  z₁ = (1 - Complex.I) * (3 + Complex.I) →
  a = z₁.im →
  z₂ = (1 + Complex.I) / (2 - Complex.I) →
  b = z₂.re →
  a * b = -2/5 := by
    sorry

end complex_product_theorem_l1853_185378


namespace childrens_tickets_l1853_185344

theorem childrens_tickets (adult_price child_price total_tickets total_cost : ℚ)
  (h1 : adult_price = 5.5)
  (h2 : child_price = 3.5)
  (h3 : total_tickets = 21)
  (h4 : total_cost = 83.5) :
  ∃ (adult_tickets child_tickets : ℚ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_cost ∧
    child_tickets = 16 := by
  sorry

end childrens_tickets_l1853_185344


namespace apple_cost_proof_l1853_185353

/-- The cost of apples for the first 30 kgs (in rupees per kg) -/
def l : ℝ := sorry

/-- The cost of apples for each additional kg beyond 30 kgs (in rupees per kg) -/
def q : ℝ := sorry

/-- The total cost of 33 kgs of apples (in rupees) -/
def cost_33 : ℝ := 11.67

/-- The total cost of 36 kgs of apples (in rupees) -/
def cost_36 : ℝ := 12.48

/-- The cost of the first 10 kgs of apples (in rupees) -/
def cost_10 : ℝ := 10 * l

theorem apple_cost_proof :
  (30 * l + 3 * q = cost_33) ∧
  (30 * l + 6 * q = cost_36) →
  cost_10 = 3.62 := by
  sorry

end apple_cost_proof_l1853_185353


namespace mystery_number_proof_l1853_185305

theorem mystery_number_proof : ∃ x : ℕ, x * 48 = 173 * 240 ∧ x = 865 := by
  sorry

end mystery_number_proof_l1853_185305


namespace circular_track_circumference_l1853_185387

/-- Represents a circular track with two moving points. -/
structure CircularTrack where
  /-- The circumference of the track in yards. -/
  circumference : ℝ
  /-- The constant speed of both points (assumed to be the same). -/
  speed : ℝ
  /-- The distance B travels before the first meeting. -/
  first_meeting_distance : ℝ
  /-- The remaining distance A needs to travel after the second meeting to complete a lap. -/
  second_meeting_remaining : ℝ

/-- The theorem stating the conditions and the result to be proved. -/
theorem circular_track_circumference (track : CircularTrack) 
  (h1 : track.first_meeting_distance = 100)
  (h2 : track.second_meeting_remaining = 60)
  (h3 : track.speed > 0) :
  track.circumference = 480 := by
  sorry

end circular_track_circumference_l1853_185387


namespace rational_equation_solutions_l1853_185343

theorem rational_equation_solutions (a b : ℚ) :
  (∃ x y : ℚ, a * x^2 + b * y^2 = 1) →
  (∀ n : ℕ, ∃ (x₁ y₁ : ℚ) (x₂ y₂ : ℚ), 
    (a * x₁^2 + b * y₁^2 = 1) ∧ 
    (a * x₂^2 + b * y₂^2 = 1) ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) :=
by sorry

end rational_equation_solutions_l1853_185343


namespace seating_arrangements_l1853_185333

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where two specific people sit next to each other -/
def adjacentArrangements (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * 2

/-- The number of ways to arrange n people in a row where two specific people cannot sit next to each other -/
def nonAdjacentArrangements (n : ℕ) : ℕ := totalArrangements n - adjacentArrangements n

theorem seating_arrangements :
  nonAdjacentArrangements 7 = 3600 := by
  sorry

end seating_arrangements_l1853_185333


namespace geometric_sum_remainder_l1853_185381

theorem geometric_sum_remainder (n : ℕ) :
  (7^(n+1) - 1) / 6 % 500 = 1 :=
by sorry

end geometric_sum_remainder_l1853_185381


namespace ad_arrangement_count_l1853_185358

-- Define the number of original ads
def original_ads : Nat := 5

-- Define the number of ads to be kept
def kept_ads : Nat := 2

-- Define the number of new ads to be added
def new_ads : Nat := 1

-- Define the number of PSAs to be added
def psas : Nat := 2

-- Define the function to calculate the number of arrangements
def num_arrangements (n m : Nat) : Nat :=
  (n.choose m) * (m + 1) * 2

-- Theorem statement
theorem ad_arrangement_count :
  num_arrangements original_ads kept_ads = 120 :=
by sorry

end ad_arrangement_count_l1853_185358


namespace area_covered_by_overlapping_strips_l1853_185366

/-- Represents a rectangular strip with a given length and width of 1 unit -/
structure Strip where
  length : ℝ
  width : ℝ := 1

/-- Calculates the total area of overlaps between strips -/
def totalOverlapArea (strips : List Strip) : ℝ := sorry

/-- Theorem: Area covered by overlapping strips -/
theorem area_covered_by_overlapping_strips
  (strips : List Strip)
  (h_strips : strips = [
    { length := 8 },
    { length := 10 },
    { length := 12 },
    { length := 7 },
    { length := 9 }
  ])
  (h_overlap : totalOverlapArea strips = 16) :
  (strips.map (λ s => s.length * s.width)).sum - totalOverlapArea strips = 30 := by
  sorry

end area_covered_by_overlapping_strips_l1853_185366


namespace quadratic_coefficient_l1853_185374

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- The theorem statement -/
theorem quadratic_coefficient (q : QuadraticFunction) 
  (vertex_x : q.f 2 = 5) 
  (point : q.f 1 = 6) : 
  q.a = 1 := by sorry

end quadratic_coefficient_l1853_185374


namespace only_pairD_not_opposite_l1853_185307

-- Define a structure for a pair of quantities
structure QuantityPair where
  first : String
  second : String

-- Define the function to check if a pair has opposite meanings
def hasOppositeMeanings (pair : QuantityPair) : Bool :=
  match pair with
  | ⟨"Income of 200 yuan", "expenditure of 20 yuan"⟩ => true
  | ⟨"Rise of 10 meters", "fall of 7 meters"⟩ => true
  | ⟨"Exceeding 0.05 mm", "falling short of 0.03 m"⟩ => true
  | ⟨"Increase of 2 years", "decrease of 2 liters"⟩ => false
  | _ => false

-- Define the pairs
def pairA : QuantityPair := ⟨"Income of 200 yuan", "expenditure of 20 yuan"⟩
def pairB : QuantityPair := ⟨"Rise of 10 meters", "fall of 7 meters"⟩
def pairC : QuantityPair := ⟨"Exceeding 0.05 mm", "falling short of 0.03 m"⟩
def pairD : QuantityPair := ⟨"Increase of 2 years", "decrease of 2 liters"⟩

-- Theorem statement
theorem only_pairD_not_opposite : 
  (hasOppositeMeanings pairA = true) ∧ 
  (hasOppositeMeanings pairB = true) ∧ 
  (hasOppositeMeanings pairC = true) ∧ 
  (hasOppositeMeanings pairD = false) :=
sorry

end only_pairD_not_opposite_l1853_185307


namespace bryden_quarters_value_l1853_185326

/-- The face value of a regular quarter in dollars -/
def regular_quarter_value : ℚ := 1/4

/-- The number of regular quarters Bryden has -/
def regular_quarters : ℕ := 4

/-- The number of special quarters Bryden has -/
def special_quarters : ℕ := 1

/-- The value multiplier for a special quarter compared to a regular quarter -/
def special_quarter_multiplier : ℚ := 2

/-- The percentage of face value the collector offers -/
def collector_offer_percentage : ℚ := 1500

theorem bryden_quarters_value :
  let total_face_value := regular_quarter_value * regular_quarters +
                          regular_quarter_value * special_quarter_multiplier * special_quarters
  let collector_offer_multiplier := collector_offer_percentage / 100
  collector_offer_multiplier * total_face_value = 45/2 :=
sorry

end bryden_quarters_value_l1853_185326


namespace sum_of_non_solutions_l1853_185379

/-- Given an equation with infinitely many solutions, prove the sum of non-solutions -/
theorem sum_of_non_solutions (A B C : ℝ) : 
  (∀ x : ℝ, (x + B) * (A * x + 28) = 2 * (x + C) * (x + 7)) →
  (∃ S : Finset ℝ, (∀ x ∉ S, (x + B) * (A * x + 28) = 2 * (x + C) * (x + 7)) ∧ 
    (∀ x ∈ S, (x + B) * (A * x + 28) ≠ 2 * (x + C) * (x + 7)) ∧
    (Finset.sum S id = -21)) :=
by sorry

end sum_of_non_solutions_l1853_185379


namespace smallest_k_bound_l1853_185315

def S : Set (ℝ → ℝ) :=
  {f | (∀ x ∈ Set.Icc 0 1, 0 ≤ f x) ∧
       f 1 = 1 ∧
       ∀ x y, x + y ≤ 1 → f x + f y ≤ f (x + y)}

theorem smallest_k_bound (f : ℝ → ℝ) (h : f ∈ S) :
  (∀ x ∈ Set.Icc 0 1, f x ≤ 2 * x) ∧
  ∀ k < 2, ∃ g ∈ S, ∃ x ∈ Set.Icc 0 1, g x > k * x :=
sorry

end smallest_k_bound_l1853_185315


namespace geometric_sequence_product_l1853_185340

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence condition
  (2 * a 2^2 - 7 * a 2 + 6 = 0) →  -- a_2 is a root of 2x^2 - 7x + 6 = 0
  (2 * a 8^2 - 7 * a 8 + 6 = 0) →  -- a_8 is a root of 2x^2 - 7x + 6 = 0
  (a 1 * a 3 * a 5 * a 7 * a 9 = 9 * Real.sqrt 3 ∨ 
   a 1 * a 3 * a 5 * a 7 * a 9 = -9 * Real.sqrt 3) :=
by sorry

end geometric_sequence_product_l1853_185340


namespace unique_postal_codes_exist_l1853_185311

def PostalCode := Fin 6 → Fin 7

def validDigits (code : PostalCode) : Prop :=
  ∀ i : Fin 6, code i < 7 ∧ code i ≠ 4

def distinctDigits (code : PostalCode) : Prop :=
  ∀ i j : Fin 6, i ≠ j → code i ≠ code j

def matchingPositions (code1 code2 : PostalCode) : Nat :=
  (List.range 6).filter (λ i => code1 i = code2 i) |>.length

def A : PostalCode := λ i => [3, 2, 0, 6, 5, 1][i]
def B : PostalCode := λ i => [1, 0, 5, 2, 6, 3][i]
def C : PostalCode := λ i => [6, 1, 2, 3, 0, 5][i]
def D : PostalCode := λ i => [3, 1, 6, 2, 5, 0][i]

theorem unique_postal_codes_exist : 
  ∃! (M N : PostalCode), 
    validDigits M ∧ validDigits N ∧
    distinctDigits M ∧ distinctDigits N ∧
    M ≠ N ∧
    (matchingPositions A M = 2 ∧ matchingPositions A N = 2) ∧
    (matchingPositions B M = 2 ∧ matchingPositions B N = 2) ∧
    (matchingPositions C M = 2 ∧ matchingPositions C N = 2) ∧
    (matchingPositions D M = 3 ∧ matchingPositions D N = 3) := by
  sorry

end unique_postal_codes_exist_l1853_185311


namespace c_value_for_four_roots_l1853_185368

/-- A complex number is a root of the polynomial Q(x) if Q(x) = 0 -/
def is_root (Q : ℂ → ℂ) (z : ℂ) : Prop := Q z = 0

/-- The polynomial Q(x) -/
def Q (c : ℂ) (x : ℂ) : ℂ := (x^2 - 3*x + 3) * (x^2 - c*x + 2) * (x^2 - 5*x + 5)

/-- The theorem stating the value of |c| for Q(x) with exactly 4 distinct roots -/
theorem c_value_for_four_roots :
  ∃ (c : ℂ), (∃ (z₁ z₂ z₃ z₄ : ℂ), z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    is_root (Q c) z₁ ∧ is_root (Q c) z₂ ∧ is_root (Q c) z₃ ∧ is_root (Q c) z₄ ∧
    (∀ (z : ℂ), is_root (Q c) z → z = z₁ ∨ z = z₂ ∨ z = z₃ ∨ z = z₄)) →
  Complex.abs c = Real.sqrt (18 - Real.sqrt 15 / 2) :=
sorry

end c_value_for_four_roots_l1853_185368
