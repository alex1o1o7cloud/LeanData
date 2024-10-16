import Mathlib

namespace NUMINAMATH_CALUDE_class_average_problem_l479_47983

theorem class_average_problem (N : ℝ) (h : N > 0) :
  let total_average : ℝ := 80
  let three_fourths_average : ℝ := 76
  let one_fourth_average : ℝ := (4 * total_average * N - 3 * three_fourths_average * N) / N
  one_fourth_average = 92 := by sorry

end NUMINAMATH_CALUDE_class_average_problem_l479_47983


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l479_47908

/-- The number of trailing zeros in base 12 for the product 33 * 59 -/
def trailing_zeros_base_12 : ℕ := 2

/-- The product we're working with -/
def product : ℕ := 33 * 59

/-- Conversion to base 12 -/
def to_base_12 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 12) ((m % 12) :: acc)
    aux n []

/-- Count trailing zeros in a list of digits -/
def count_trailing_zeros (digits : List ℕ) : ℕ :=
  digits.reverse.takeWhile (· = 0) |>.length

theorem product_trailing_zeros :
  count_trailing_zeros (to_base_12 product) = trailing_zeros_base_12 := by
  sorry

#eval to_base_12 product
#eval count_trailing_zeros (to_base_12 product)

end NUMINAMATH_CALUDE_product_trailing_zeros_l479_47908


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l479_47928

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l479_47928


namespace NUMINAMATH_CALUDE_weeks_to_add_fish_is_three_l479_47939

/-- Calculates the number of weeks it took to add fish to a tank -/
def weeks_to_add_fish (initial_total : ℕ) (final_koi : ℕ) (final_goldfish : ℕ) 
                      (daily_koi_addition : ℕ) (daily_goldfish_addition : ℕ) : ℚ :=
  let final_total := final_koi + final_goldfish
  let total_added := final_total - initial_total
  let daily_total_addition := daily_koi_addition + daily_goldfish_addition
  let days := total_added / daily_total_addition
  days / 7

/-- Theorem stating that the number of weeks to add fish is 3 -/
theorem weeks_to_add_fish_is_three :
  weeks_to_add_fish 280 227 200 2 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_weeks_to_add_fish_is_three_l479_47939


namespace NUMINAMATH_CALUDE_negation_of_perpendicular_plane_l479_47900

-- Define the concept of a line
variable (Line : Type)

-- Define the concept of a plane
variable (Plane : Type)

-- Define what it means for a plane to be perpendicular to a line
variable (perpendicular : Plane → Line → Prop)

-- State the theorem
theorem negation_of_perpendicular_plane :
  (¬ ∀ l : Line, ∃ α : Plane, perpendicular α l) ↔ 
  (∃ l : Line, ∀ α : Plane, ¬ perpendicular α l) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_perpendicular_plane_l479_47900


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l479_47933

-- Define the quadratic function
def f (x : ℝ) := x^2 - 3*x + 2

-- Define the solution set
def solution_set : Set ℝ := {x | f x < 0}

-- State the theorem
theorem quadratic_inequality_solution :
  solution_set = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l479_47933


namespace NUMINAMATH_CALUDE_function_zeros_theorem_l479_47920

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp x - k * x + k

theorem function_zeros_theorem (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : f k x₁ = 0) 
  (h2 : f k x₂ = 0) 
  (h3 : x₁ ≠ x₂) : 
  k > Real.exp 2 ∧ x₁ + x₂ > 4 := by
  sorry

end NUMINAMATH_CALUDE_function_zeros_theorem_l479_47920


namespace NUMINAMATH_CALUDE_volunteer_selection_count_l479_47951

/-- The number of boys in the group -/
def num_boys : ℕ := 4

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of students to be selected -/
def num_selected : ℕ := 4

/-- The total number of students -/
def total_students : ℕ := num_boys + num_girls

/-- The number of ways to select 4 students from 7 students -/
def total_selections : ℕ := Nat.choose total_students num_selected

/-- The number of ways to select 4 boys from 4 boys -/
def all_boys_selections : ℕ := Nat.choose num_boys num_selected

theorem volunteer_selection_count :
  total_selections - all_boys_selections = 34 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_count_l479_47951


namespace NUMINAMATH_CALUDE_container_emptying_possible_l479_47903

/-- Represents a container with water -/
structure Container where
  water : ℕ

/-- Represents the state of three containers -/
structure ContainerState where
  a : Container
  b : Container
  c : Container

/-- Represents a transfer of water between containers -/
inductive Transfer : ContainerState → ContainerState → Prop where
  | ab (s : ContainerState) : 
      Transfer s ⟨⟨s.a.water + s.b.water⟩, ⟨0⟩, s.c⟩
  | ac (s : ContainerState) : 
      Transfer s ⟨⟨s.a.water + s.c.water⟩, s.b, ⟨0⟩⟩
  | ba (s : ContainerState) : 
      Transfer s ⟨⟨0⟩, ⟨s.a.water + s.b.water⟩, s.c⟩
  | bc (s : ContainerState) : 
      Transfer s ⟨s.a, ⟨s.b.water + s.c.water⟩, ⟨0⟩⟩
  | ca (s : ContainerState) : 
      Transfer s ⟨⟨0⟩, s.b, ⟨s.a.water + s.c.water⟩⟩
  | cb (s : ContainerState) : 
      Transfer s ⟨s.a, ⟨0⟩, ⟨s.b.water + s.c.water⟩⟩

/-- Represents a sequence of transfers -/
def TransferSeq := List (ContainerState → ContainerState)

/-- Applies a sequence of transfers to an initial state -/
def applyTransfers (initial : ContainerState) (seq : TransferSeq) : ContainerState :=
  seq.foldl (fun state transfer => transfer state) initial

/-- Predicate to check if a container is empty -/
def isEmptyContainer (c : Container) : Prop := c.water = 0

/-- Predicate to check if any container in the state is empty -/
def hasEmptyContainer (s : ContainerState) : Prop :=
  isEmptyContainer s.a ∨ isEmptyContainer s.b ∨ isEmptyContainer s.c

/-- The main theorem to prove -/
theorem container_emptying_possible (initial : ContainerState) : 
  ∃ (seq : TransferSeq), hasEmptyContainer (applyTransfers initial seq) := by
  sorry

end NUMINAMATH_CALUDE_container_emptying_possible_l479_47903


namespace NUMINAMATH_CALUDE_definite_integral_exp_plus_2x_l479_47963

theorem definite_integral_exp_plus_2x : ∫ (x : ℝ) in (0)..(1), (Real.exp x + 2 * x) = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_exp_plus_2x_l479_47963


namespace NUMINAMATH_CALUDE_equilateral_triangle_grid_polygon_area_l479_47996

/-- Represents an equilateral triangular grid -/
structure EquilateralTriangularGrid where
  sideLength : ℕ
  totalPoints : ℕ

/-- Represents a polygon on the grid -/
structure Polygon (G : EquilateralTriangularGrid) where
  vertices : ℕ
  nonSelfIntersecting : Bool
  usesAllPoints : Bool

/-- The area of a polygon on an equilateral triangular grid -/
noncomputable def polygonArea (G : EquilateralTriangularGrid) (S : Polygon G) : ℝ :=
  sorry

theorem equilateral_triangle_grid_polygon_area 
  (G : EquilateralTriangularGrid) 
  (S : Polygon G) :
  G.sideLength = 20 ∧ 
  G.totalPoints = 210 ∧ 
  S.vertices = 210 ∧ 
  S.nonSelfIntersecting = true ∧ 
  S.usesAllPoints = true →
  polygonArea G S = 52 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_grid_polygon_area_l479_47996


namespace NUMINAMATH_CALUDE_sqrt_27_simplification_l479_47901

theorem sqrt_27_simplification : Real.sqrt 27 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_simplification_l479_47901


namespace NUMINAMATH_CALUDE_power_of_four_exponent_l479_47915

theorem power_of_four_exponent (n : ℕ) (x : ℕ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^x) 
  (h2 : n = 17) : 
  x = 18 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_exponent_l479_47915


namespace NUMINAMATH_CALUDE_action_figure_price_l479_47993

theorem action_figure_price (board_game_cost : ℝ) (num_figures : ℕ) (total_cost : ℝ) :
  board_game_cost = 2 →
  num_figures = 4 →
  total_cost = 30 →
  ∃ (figure_price : ℝ), figure_price = 7 ∧ total_cost = board_game_cost + num_figures * figure_price :=
by
  sorry

end NUMINAMATH_CALUDE_action_figure_price_l479_47993


namespace NUMINAMATH_CALUDE_abs_equation_roots_properties_l479_47944

def abs_equation (x : ℝ) : Prop := |x|^2 + 2*|x| - 8 = 0

theorem abs_equation_roots_properties :
  ∃ (root1 root2 : ℝ),
    (abs_equation root1 ∧ abs_equation root2) ∧
    (root1 = 2 ∧ root2 = -2) ∧
    (root1 + root2 = 0) ∧
    (root1 * root2 = -4) := by sorry

end NUMINAMATH_CALUDE_abs_equation_roots_properties_l479_47944


namespace NUMINAMATH_CALUDE_unit_vector_same_direction_l479_47921

def b : Fin 2 → ℝ := ![(-3), 4]

theorem unit_vector_same_direction (a : Fin 2 → ℝ) : 
  (∀ i, a i * a i = 1) →  -- a is a unit vector
  (∃ c : ℝ, c ≠ 0 ∧ ∀ i, a i = c * b i) →  -- a is in the same direction as b
  a = ![(-3/5), 4/5] := by
sorry

end NUMINAMATH_CALUDE_unit_vector_same_direction_l479_47921


namespace NUMINAMATH_CALUDE_rectangle_area_l479_47980

/-- Given a rectangle with diagonal length x and length three times its width, 
    prove that its area is (3/10)x^2 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ w : ℝ, 
  w > 0 ∧ 
  x^2 = (3*w)^2 + w^2 ∧ 
  (3*w) * w = (3/10) * x^2 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_l479_47980


namespace NUMINAMATH_CALUDE_oranges_for_juice_l479_47926

/-- Given that 18 oranges make 27 liters of orange juice, 
    prove that 6 oranges are needed to make 9 liters of orange juice. -/
theorem oranges_for_juice (oranges : ℕ) (juice : ℕ) 
  (h : 18 * juice = 27 * oranges) : 
  6 * juice = 9 * oranges :=
by sorry

end NUMINAMATH_CALUDE_oranges_for_juice_l479_47926


namespace NUMINAMATH_CALUDE_fred_card_purchase_l479_47985

/-- The number of packs of football cards Fred bought -/
def football_packs : ℕ := 2

/-- The cost of one pack of football cards -/
def football_cost : ℚ := 273/100

/-- The cost of the pack of Pokemon cards -/
def pokemon_cost : ℚ := 401/100

/-- The cost of the deck of baseball cards -/
def baseball_cost : ℚ := 895/100

/-- The total amount Fred spent on cards -/
def total_spent : ℚ := 1842/100

theorem fred_card_purchase :
  (football_packs : ℚ) * football_cost + pokemon_cost + baseball_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_fred_card_purchase_l479_47985


namespace NUMINAMATH_CALUDE_algebra_to_calculus_ratio_l479_47937

/-- Represents the number of years Devin taught each subject and in total -/
structure TeachingYears where
  calculus : ℕ
  algebra : ℕ
  statistics : ℕ
  total : ℕ

/-- Defines the conditions of Devin's teaching career -/
def devin_teaching (y : TeachingYears) : Prop :=
  y.calculus = 4 ∧
  y.statistics = 5 * y.algebra ∧
  y.total = y.calculus + y.algebra + y.statistics ∧
  y.total = 52

/-- Theorem stating the ratio of Algebra to Calculus teaching years -/
theorem algebra_to_calculus_ratio (y : TeachingYears) 
  (h : devin_teaching y) : y.algebra / y.calculus = 2 := by
  sorry

end NUMINAMATH_CALUDE_algebra_to_calculus_ratio_l479_47937


namespace NUMINAMATH_CALUDE_equation_solution_difference_l479_47972

theorem equation_solution_difference : ∃ (a b : ℝ),
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ -6 → ((5 * x - 20) / (x^2 + 3*x - 18) = x + 3 ↔ (x = a ∨ x = b))) ∧
  a > b ∧
  a - b = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l479_47972


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l479_47968

-- Define the conditions p and q
def p (x : ℝ) : Prop := x^2 - 4*x + 3 > 0
def q (x : ℝ) : Prop := x^2 < 1

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l479_47968


namespace NUMINAMATH_CALUDE_room_length_is_ten_l479_47910

/-- Proves that the length of a rectangular room is 10 meters given specific conditions. -/
theorem room_length_is_ten (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 4.75 →
  total_cost = 42750 →
  paving_rate = 900 →
  total_cost / paving_rate / width = 10 := by
  sorry


end NUMINAMATH_CALUDE_room_length_is_ten_l479_47910


namespace NUMINAMATH_CALUDE_vector_is_direction_vector_l479_47930

/-- A line in 2D space --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A vector in 2D space --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a vector is a direction vector of a line --/
def isDirectionVector (l : Line2D) (v : Vector2D) : Prop :=
  l.a * v.x + l.b * v.y = 0

/-- The given line x - 3y + 1 = 0 --/
def givenLine : Line2D :=
  { a := 1, b := -3, c := 1 }

/-- The vector (3,1) --/
def givenVector : Vector2D :=
  { x := 3, y := 1 }

/-- Theorem: (3,1) is a direction vector of the line x - 3y + 1 = 0 --/
theorem vector_is_direction_vector : isDirectionVector givenLine givenVector := by
  sorry

end NUMINAMATH_CALUDE_vector_is_direction_vector_l479_47930


namespace NUMINAMATH_CALUDE_sin_405_degrees_l479_47995

theorem sin_405_degrees (h : 405 = 360 + 45) : Real.sin (405 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_405_degrees_l479_47995


namespace NUMINAMATH_CALUDE_min_value_z_l479_47909

theorem min_value_z (x y : ℝ) (h1 : 2*x + 3*y - 3 ≤ 0) (h2 : 2*x - 3*y + 3 ≥ 0) (h3 : y + 3 ≥ 0) :
  ∀ z : ℝ, z = 2*x + y → z ≥ -3 ∧ ∃ x₀ y₀ : ℝ, 2*x₀ + 3*y₀ - 3 ≤ 0 ∧ 2*x₀ - 3*y₀ + 3 ≥ 0 ∧ y₀ + 3 ≥ 0 ∧ 2*x₀ + y₀ = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l479_47909


namespace NUMINAMATH_CALUDE_allocation_theorem_l479_47947

/-- Represents the number of students -/
def num_students : ℕ := 5

/-- Represents the number of groups -/
def num_groups : ℕ := 3

/-- Function to calculate the number of allocation methods -/
def allocation_methods (n : ℕ) (k : ℕ) (excluded_pair : Bool) : ℕ :=
  sorry

/-- Theorem stating the number of allocation methods -/
theorem allocation_theorem :
  allocation_methods num_students num_groups true = 114 :=
sorry

end NUMINAMATH_CALUDE_allocation_theorem_l479_47947


namespace NUMINAMATH_CALUDE_circle_radius_tangent_to_three_lines_l479_47964

/-- A circle with center (0, k) where k > 8 is tangent to y = x, y = -x, and y = 8. Its radius is 8√2. -/
theorem circle_radius_tangent_to_three_lines (k : ℝ) (h1 : k > 8) : 
  let center := (0, k)
  let radius := (λ p : ℝ × ℝ ↦ Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2))
  let tangent_to_line := (λ l : ℝ × ℝ → Prop ↦ ∃ p, l p ∧ radius p = radius center)
  tangent_to_line (λ p ↦ p.2 = p.1) ∧ 
  tangent_to_line (λ p ↦ p.2 = -p.1) ∧
  tangent_to_line (λ p ↦ p.2 = 8) →
  radius center = 8 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_tangent_to_three_lines_l479_47964


namespace NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l479_47965

theorem largest_n_for_sin_cos_inequality : 
  (∃ (n : ℕ), n > 0 ∧ (∀ x : ℝ, (Real.sin x)^n + (Real.cos x)^n ≥ 2/n)) ∧ 
  (∀ m : ℕ, m > 4 → ∃ x : ℝ, (Real.sin x)^m + (Real.cos x)^m < 2/m) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l479_47965


namespace NUMINAMATH_CALUDE_power_sum_equals_six_l479_47999

theorem power_sum_equals_six : 2 - 2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 + 2^10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_six_l479_47999


namespace NUMINAMATH_CALUDE_exactly_two_rectangle_coverage_l479_47986

/-- Represents a rectangle on a grid -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the overlap between two rectangles -/
structure Overlap where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the area of an overlap -/
def overlapArea (o : Overlap) : ℕ := o.width * o.height

/-- The main theorem -/
theorem exactly_two_rectangle_coverage 
  (r1 r2 r3 : Rectangle)
  (o12 o23 : Overlap)
  (h1 : r1.width = 4 ∧ r1.height = 6)
  (h2 : r2.width = 4 ∧ r2.height = 6)
  (h3 : r3.width = 4 ∧ r3.height = 6)
  (h4 : o12.width = 2 ∧ o12.height = 4)
  (h5 : o23.width = 2 ∧ o23.height = 4)
  (h6 : overlapArea o12 + overlapArea o23 = 16) :
  11 = overlapArea o12 + overlapArea o23 - 5 := by
  sorry


end NUMINAMATH_CALUDE_exactly_two_rectangle_coverage_l479_47986


namespace NUMINAMATH_CALUDE_yard_trees_l479_47984

/-- Calculates the number of trees in a yard given the yard length and distance between trees. -/
def num_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem stating that in a 325-meter yard with trees 13 meters apart, there are 26 trees. -/
theorem yard_trees : num_trees 325 13 = 26 := by
  sorry

end NUMINAMATH_CALUDE_yard_trees_l479_47984


namespace NUMINAMATH_CALUDE_janice_purchase_problem_l479_47919

theorem janice_purchase_problem :
  ∀ (a b c : ℕ),
    a + b + c = 60 →
    15 * a + 400 * b + 500 * c = 6000 →
    a = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_janice_purchase_problem_l479_47919


namespace NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l479_47994

theorem min_sum_with_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1/x + 4/y = 1) : x + y ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_reciprocal_constraint_l479_47994


namespace NUMINAMATH_CALUDE_division_problem_solution_l479_47929

theorem division_problem_solution :
  ∃! y : ℝ, y > 0 ∧ (2 * (62.5 + 5) / y) - 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_solution_l479_47929


namespace NUMINAMATH_CALUDE_green_chips_count_l479_47954

/-- Given a jar of chips where:
  * 3 blue chips represent 10% of the total
  * 50% of the chips are white
  * The remaining chips are green
  Prove that there are 12 green chips -/
theorem green_chips_count (total : ℕ) (blue white green : ℕ) : 
  blue = 3 ∧ 
  blue * 10 = total ∧ 
  2 * white = total ∧ 
  blue + white + green = total → 
  green = 12 := by
sorry

end NUMINAMATH_CALUDE_green_chips_count_l479_47954


namespace NUMINAMATH_CALUDE_impossible_flower_movement_l479_47938

/-- Represents a vase containing white and red roses -/
structure Vase where
  white : ℕ
  red : ℕ

/-- Represents the circular arrangement of vases -/
def VaseCircle := Fin 2019 → Vase

/-- A function that moves one flower from each vase to the next -/
def MoveFlowers (circle : VaseCircle) : VaseCircle := sorry

theorem impossible_flower_movement (circle : VaseCircle) :
  ¬∀ (i : Fin 2019),
    let new_circle := MoveFlowers circle
    (new_circle i).white ≠ (circle i).white ∧
    (new_circle i).red ≠ (circle i).red :=
  sorry

#check impossible_flower_movement

end NUMINAMATH_CALUDE_impossible_flower_movement_l479_47938


namespace NUMINAMATH_CALUDE_large_bucket_capacity_l479_47971

theorem large_bucket_capacity (small : ℝ) (large : ℝ) 
  (h1 : large = 2 * small + 3)
  (h2 : 2 * small + 5 * large = 63) :
  large = 11 := by
sorry

end NUMINAMATH_CALUDE_large_bucket_capacity_l479_47971


namespace NUMINAMATH_CALUDE_part_one_part_two_l479_47967

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y - 1
def B (x y : ℝ) : ℝ := x^2 - x * y

-- Part 1
theorem part_one : ∀ x y : ℝ, (x + 1)^2 + |y - 2| = 0 → A x y - 2 * B x y = -7 := by
  sorry

-- Part 2
theorem part_two : (∃ c : ℝ, ∀ x y : ℝ, A x y - 2 * B x y = c) → 
  ∃ x : ℝ, x^2 - 2*x - 1 = -1/25 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l479_47967


namespace NUMINAMATH_CALUDE_largest_integer_quadratic_negative_l479_47952

theorem largest_integer_quadratic_negative : 
  (∀ m : ℤ, m > 7 → m^2 - 11*m + 24 ≥ 0) ∧ 
  (7^2 - 11*7 + 24 < 0) := by
sorry

end NUMINAMATH_CALUDE_largest_integer_quadratic_negative_l479_47952


namespace NUMINAMATH_CALUDE_max_students_distribution_l479_47902

def number_of_pens : ℕ := 2010
def number_of_pencils : ℕ := 1050

theorem max_students_distribution (n : ℕ) :
  (n ∣ number_of_pens) ∧ 
  (n ∣ number_of_pencils) ∧ 
  (∀ m : ℕ, m > n → ¬(m ∣ number_of_pens) ∨ ¬(m ∣ number_of_pencils)) →
  n = 30 :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l479_47902


namespace NUMINAMATH_CALUDE_max_rectangles_after_removal_l479_47962

/-- Represents a grid with some squares removed -/
structure Grid :=
  (size : Nat)
  (removedSquares : List (Nat × Nat × Nat))

/-- Represents a rectangle -/
structure Rectangle :=
  (width : Nat)
  (height : Nat)

/-- The maximum number of rectangles that can be cut from a grid -/
def maxRectangles (g : Grid) (r : Rectangle) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem max_rectangles_after_removal :
  let initialGrid : Grid := { size := 8, removedSquares := [(2, 2, 3)] }
  let targetRectangle : Rectangle := { width := 1, height := 3 }
  maxRectangles initialGrid targetRectangle = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_after_removal_l479_47962


namespace NUMINAMATH_CALUDE_range_of_m_l479_47976

open Set Real

theorem range_of_m (m : ℝ) : 
  let A : Set ℝ := {x | -1 < x ∧ x < 7}
  let B : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 3 * m + 1}
  let p := A ∩ B = B
  let q := ∃! x, x^2 + 2*m*x + 2*m ≤ 0
  ¬(p ∨ q) → m ∈ Ici 2 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_m_l479_47976


namespace NUMINAMATH_CALUDE_intersection_when_a_is_four_subset_iff_a_geq_four_l479_47982

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem 1: When a = 4, A ∩ B = A
theorem intersection_when_a_is_four :
  A ∩ B 4 = A := by sorry

-- Theorem 2: A ⊆ B if and only if a ≥ 4
theorem subset_iff_a_geq_four (a : ℝ) :
  A ⊆ B a ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_four_subset_iff_a_geq_four_l479_47982


namespace NUMINAMATH_CALUDE_x_value_proof_l479_47935

theorem x_value_proof (x : Real) 
  (h1 : Real.sin (π / 2 - x) = -Real.sqrt 3 / 2)
  (h2 : π < x ∧ x < 2 * π) : 
  x = 7 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l479_47935


namespace NUMINAMATH_CALUDE_gcd_A_B_eq_one_l479_47997

def A : ℤ := 1989^1990 - 1988^1990
def B : ℤ := 1989^1989 - 1988^1989

theorem gcd_A_B_eq_one : Int.gcd A B = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_A_B_eq_one_l479_47997


namespace NUMINAMATH_CALUDE_sales_tax_rate_zero_l479_47975

theorem sales_tax_rate_zero (sale_price_with_tax : ℝ) (profit_percentage : ℝ) (cost_price : ℝ)
  (h1 : sale_price_with_tax = 616)
  (h2 : profit_percentage = 16)
  (h3 : cost_price = 531.03) :
  let profit := (profit_percentage / 100) * cost_price
  let sale_price_before_tax := cost_price + profit
  let sales_tax_rate := ((sale_price_with_tax - sale_price_before_tax) / sale_price_before_tax) * 100
  sales_tax_rate = 0 := by sorry

end NUMINAMATH_CALUDE_sales_tax_rate_zero_l479_47975


namespace NUMINAMATH_CALUDE_octagon_perimeter_l479_47960

/-- An octagon is a polygon with 8 sides -/
def Octagon : Type := Unit

/-- The length of each side of the octagon -/
def side_length : ℝ := 3

/-- The perimeter of a polygon is the sum of the lengths of its sides -/
def perimeter (p : Octagon) : ℝ := 8 * side_length

theorem octagon_perimeter : 
  ∀ (o : Octagon), perimeter o = 24 := by
  sorry

end NUMINAMATH_CALUDE_octagon_perimeter_l479_47960


namespace NUMINAMATH_CALUDE_triangle_side_not_unique_l479_47934

/-- Represents a triangle with sides a, b, and c, and area A -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ

/-- Theorem stating that the length of side 'a' in a triangle cannot be uniquely determined
    given only the lengths of two sides and the area -/
theorem triangle_side_not_unique (t : Triangle) (h1 : t.b = 19) (h2 : t.c = 5) (h3 : t.A = 47.5) :
  ¬ ∃! a : ℝ, t.a = a ∧ 0 < a ∧ a + t.b > t.c ∧ a + t.c > t.b ∧ t.b + t.c > a :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_not_unique_l479_47934


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l479_47936

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 4 + a 6 + a 13 + a 15 = 20) :
  a 10 - (1/5) * a 12 = 4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l479_47936


namespace NUMINAMATH_CALUDE_cookies_left_after_week_l479_47973

/-- The number of cookies left after a week -/
def cookiesLeftAfterWeek (initialCookies : ℕ) (cookiesTakenInFourDays : ℕ) : ℕ :=
  initialCookies - 7 * (cookiesTakenInFourDays / 4)

/-- Theorem: The number of cookies left after a week is 28 -/
theorem cookies_left_after_week :
  cookiesLeftAfterWeek 70 24 = 28 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_after_week_l479_47973


namespace NUMINAMATH_CALUDE_rotation_270_of_8_minus_4i_l479_47953

-- Define the rotation function
def rotate270 (z : ℂ) : ℂ := -z.im + z.re * Complex.I

-- State the theorem
theorem rotation_270_of_8_minus_4i :
  rotate270 (8 - 4 * Complex.I) = -4 - 8 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_rotation_270_of_8_minus_4i_l479_47953


namespace NUMINAMATH_CALUDE_log_inequality_l479_47918

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

-- State the theorem
theorem log_inequality (m : ℝ) 
  (h1 : ∀ x : ℝ, 1/m - 4 ≥ f m x)
  (h2 : m > 0) :
  Real.log (m + 2) / Real.log (m + 1) > Real.log (m + 3) / Real.log (m + 2) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_l479_47918


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l479_47950

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 16 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l479_47950


namespace NUMINAMATH_CALUDE_pen_promotion_result_l479_47987

/-- Represents the promotion event in a shop selling pens and giving away teddy bears. -/
structure PenPromotion where
  /-- Profit in yuan for selling one pen -/
  profit_per_pen : ℕ
  /-- Cost in yuan for one teddy bear -/
  cost_per_bear : ℕ
  /-- Total profit in yuan from the promotion event -/
  total_profit : ℕ

/-- Calculates the number of pens sold during the promotion event -/
def pens_sold (promo : PenPromotion) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that given the specific conditions of the promotion,
    the number of pens sold is 335 -/
theorem pen_promotion_result :
  let promo : PenPromotion := {
    profit_per_pen := 7,
    cost_per_bear := 2,
    total_profit := 2011
  }
  pens_sold promo = 335 := by
  sorry

end NUMINAMATH_CALUDE_pen_promotion_result_l479_47987


namespace NUMINAMATH_CALUDE_band_second_set_songs_l479_47904

/-- Proves the number of songs played in the second set given the band's repertoire and performance details -/
theorem band_second_set_songs 
  (total_songs : ℕ) 
  (first_set : ℕ) 
  (encore : ℕ) 
  (avg_third_fourth : ℕ) 
  (h1 : total_songs = 30)
  (h2 : first_set = 5)
  (h3 : encore = 2)
  (h4 : avg_third_fourth = 8) :
  ∃ (second_set : ℕ), 
    second_set = 7 ∧ 
    (total_songs - first_set - second_set - encore) / 2 = avg_third_fourth :=
by sorry

end NUMINAMATH_CALUDE_band_second_set_songs_l479_47904


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l479_47988

theorem partial_fraction_sum_zero (x A B C D E F : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) = 
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l479_47988


namespace NUMINAMATH_CALUDE_unique_n_squared_plus_2n_prime_l479_47905

theorem unique_n_squared_plus_2n_prime :
  ∃! (n : ℕ), n > 0 ∧ Nat.Prime (n^2 + 2*n) :=
sorry

end NUMINAMATH_CALUDE_unique_n_squared_plus_2n_prime_l479_47905


namespace NUMINAMATH_CALUDE_truck_speed_l479_47966

/-- Calculates the speed of a truck in kilometers per hour -/
theorem truck_speed (distance : ℝ) (time : ℝ) (h1 : distance = 600) (h2 : time = 10) :
  (distance / time) * 3.6 = 216 := by
  sorry

#check truck_speed

end NUMINAMATH_CALUDE_truck_speed_l479_47966


namespace NUMINAMATH_CALUDE_expression_equality_l479_47981

theorem expression_equality (x y : ℝ) (h : x^2 + y^2 = 1) :
  2*x^4 + 3*x^2*y^2 + y^4 + y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l479_47981


namespace NUMINAMATH_CALUDE_new_player_weight_l479_47912

theorem new_player_weight (n : ℕ) (old_avg new_avg new_weight : ℝ) : 
  n = 20 →
  old_avg = 180 →
  new_avg = 181.42857142857142 →
  (n * old_avg + new_weight) / (n + 1) = new_avg →
  new_weight = 210 := by
sorry

end NUMINAMATH_CALUDE_new_player_weight_l479_47912


namespace NUMINAMATH_CALUDE_long_jump_challenge_l479_47946

/-- Represents a student in the long jump challenge -/
structure Student where
  success_prob : ℚ
  deriving Repr

/-- Calculates the probability of a student achieving excellence -/
def excellence_prob (s : Student) : ℚ :=
  s.success_prob + (1 - s.success_prob) * s.success_prob

/-- Calculates the probability of a student achieving a good rating -/
def good_prob (s : Student) : ℚ :=
  1 - excellence_prob s

/-- The probability that exactly two out of three students achieve a good rating -/
def prob_two_good (s1 s2 s3 : Student) : ℚ :=
  excellence_prob s1 * good_prob s2 * good_prob s3 +
  good_prob s1 * excellence_prob s2 * good_prob s3 +
  good_prob s1 * good_prob s2 * excellence_prob s3

theorem long_jump_challenge (s1 s2 s3 : Student)
  (h1 : s1.success_prob = 3/4)
  (h2 : s2.success_prob = 1/2)
  (h3 : s3.success_prob = 1/3) :
  prob_two_good s1 s2 s3 = 77/576 := by
  sorry

#eval prob_two_good ⟨3/4⟩ ⟨1/2⟩ ⟨1/3⟩

end NUMINAMATH_CALUDE_long_jump_challenge_l479_47946


namespace NUMINAMATH_CALUDE_multiplier_problem_l479_47991

theorem multiplier_problem (n : ℝ) (h : n = 15) : 
  ∃ m : ℝ, 2 * n = (26 - n) + 19 ∧ n * m = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_multiplier_problem_l479_47991


namespace NUMINAMATH_CALUDE_arithmetic_sequence_unique_value_l479_47949

theorem arithmetic_sequence_unique_value (a : ℝ) (a_n : ℕ → ℝ) : 
  a > 0 ∧ 
  (∀ n : ℕ, a_n (n + 1) - a_n n = a_n (n + 2) - a_n (n + 1)) ∧ 
  a_n 1 = a ∧
  (∃! q : ℝ, a_n 2 - a_n 1 = q ∧ (a_n 2 + 2) - (a_n 1 + 1) = q ∧ (a_n 3 + 3) - (a_n 2 + 2) = q) →
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_unique_value_l479_47949


namespace NUMINAMATH_CALUDE_power_functions_inequality_l479_47948

theorem power_functions_inequality (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) :
  (x₁ + x₂)^2 / 4 < (x₁^2 + x₂^2) / 2 ∧
  2 / (x₁ + x₂) < (1 / x₁ + 1 / x₂) / 2 :=
by sorry

end NUMINAMATH_CALUDE_power_functions_inequality_l479_47948


namespace NUMINAMATH_CALUDE_max_candies_eaten_l479_47917

theorem max_candies_eaten (n : Nat) (h : n = 32) : 
  (n.choose 2) = 496 := by
  sorry

end NUMINAMATH_CALUDE_max_candies_eaten_l479_47917


namespace NUMINAMATH_CALUDE_jordan_rectangle_length_l479_47924

/-- Given two rectangles with equal area, where one rectangle measures 5 inches by 24 inches
    and the other has a width of 15 inches, prove that the length of the second rectangle is 8 inches. -/
theorem jordan_rectangle_length (carol_length carol_width jordan_width : ℕ) 
  (h1 : carol_length = 5)
  (h2 : carol_width = 24)
  (h3 : jordan_width = 15)
  (h4 : carol_length * carol_width = jordan_width * (carol_length * carol_width / jordan_width)) :
  carol_length * carol_width / jordan_width = 8 := by
  sorry

#check jordan_rectangle_length

end NUMINAMATH_CALUDE_jordan_rectangle_length_l479_47924


namespace NUMINAMATH_CALUDE_daps_equivalent_to_24_dips_l479_47927

-- Define the units
variable (dap dop dip : ℝ)

-- Define the relationships between units
axiom dap_to_dop : 5 * dap = 4 * dop
axiom dop_to_dip : 3 * dop = 8 * dip

-- Theorem to prove
theorem daps_equivalent_to_24_dips : 
  24 * dip = (45/4) * dap := by sorry

end NUMINAMATH_CALUDE_daps_equivalent_to_24_dips_l479_47927


namespace NUMINAMATH_CALUDE_f_properties_l479_47961

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

theorem f_properties :
  (∀ x, x < -1/3 ∨ x > 1 → f' x > 0) ∧
  (∀ x, -1/3 < x ∧ x < 1 → f' x < 0) ∧
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ≤ 2) ∧
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ≥ -10) ∧
  (∃ x, x ∈ Set.Icc (-2 : ℝ) 2 ∧ f x = 2) ∧
  (∃ x, x ∈ Set.Icc (-2 : ℝ) 2 ∧ f x = -10) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l479_47961


namespace NUMINAMATH_CALUDE_cosine_cube_sum_l479_47942

theorem cosine_cube_sum (α : ℝ) :
  (Real.cos α)^3 + (Real.cos (α + 2 * Real.pi / 3))^3 + (Real.cos (α - 2 * Real.pi / 3))^3 = 
  3/4 * Real.cos (3 * α) := by
  sorry

end NUMINAMATH_CALUDE_cosine_cube_sum_l479_47942


namespace NUMINAMATH_CALUDE_max_area_rectangle_l479_47943

/-- The maximum area of a rectangle with perimeter 40 cm is 100 square centimeters. -/
theorem max_area_rectangle (x y : ℝ) (h : x + y = 20) : 
  x * y ≤ 100 :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l479_47943


namespace NUMINAMATH_CALUDE_prime_sum_square_l479_47978

theorem prime_sum_square (p q r : ℕ) (n : ℕ+) :
  Prime p → Prime q → Prime r → p^(n:ℕ) + q^(n:ℕ) = r^2 → n = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_square_l479_47978


namespace NUMINAMATH_CALUDE_rectangle_area_l479_47989

/-- Theorem: Area of a rectangle with one side 15 and diagonal 17 is 120 -/
theorem rectangle_area (side : ℝ) (diagonal : ℝ) (area : ℝ) : 
  side = 15 → diagonal = 17 → area = side * (Real.sqrt (diagonal^2 - side^2)) → area = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l479_47989


namespace NUMINAMATH_CALUDE_product_of_h_at_roots_of_p_l479_47931

theorem product_of_h_at_roots_of_p (y₁ y₂ y₃ y₄ y₅ : ℂ) : 
  (y₁^5 - y₁^3 + 1 = 0) → 
  (y₂^5 - y₂^3 + 1 = 0) → 
  (y₃^5 - y₃^3 + 1 = 0) → 
  (y₄^5 - y₄^3 + 1 = 0) → 
  (y₅^5 - y₅^3 + 1 = 0) → 
  ((y₁^2 + 1) * (y₂^2 + 1) * (y₃^2 + 1) * (y₄^2 + 1) * (y₅^2 + 1) = Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_product_of_h_at_roots_of_p_l479_47931


namespace NUMINAMATH_CALUDE_range_of_a_l479_47958

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_a (a : ℝ) : Prop := a ≤ -2 ∨ a = 1

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, (¬(¬(p a) ∨ ¬(q a))) → range_a a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l479_47958


namespace NUMINAMATH_CALUDE_f_max_value_l479_47970

/-- The function f(x) = 9x - 4x^2 -/
def f (x : ℝ) : ℝ := 9*x - 4*x^2

/-- The maximum value of f(x) is 81/16 -/
theorem f_max_value : ∀ x : ℝ, f x ≤ 81/16 := by sorry

end NUMINAMATH_CALUDE_f_max_value_l479_47970


namespace NUMINAMATH_CALUDE_gcd_459_357_l479_47911

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l479_47911


namespace NUMINAMATH_CALUDE_five_distinct_values_of_triple_exponentiation_l479_47941

def exponentiateThree (n : ℕ) : ℕ := 3^n

theorem five_distinct_values_of_triple_exponentiation :
  ∃! (s : Finset ℕ), 
    (∀ x ∈ s, ∃ f : ℕ → ℕ → ℕ, x = f (exponentiateThree 3) (exponentiateThree (exponentiateThree 3))) ∧ 
    s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_distinct_values_of_triple_exponentiation_l479_47941


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l479_47906

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6 + c

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (a b c : ℕ) : ℕ := a * 5^2 + b * 5 + c

theorem base_conversion_subtraction :
  base6ToBase10 3 5 4 - base5ToBase10 2 3 1 = 76 := by sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l479_47906


namespace NUMINAMATH_CALUDE_min_radios_problem_l479_47979

/-- Represents the problem of finding the minimum number of radios. -/
theorem min_radios_problem (n d : ℕ) : 
  n > 0 → -- n is positive
  d > 0 → -- d is positive
  (45 : ℚ) - (d + 90 : ℚ) / n = -105 → -- profit equation
  n ≥ 2 := by
  sorry

#check min_radios_problem

end NUMINAMATH_CALUDE_min_radios_problem_l479_47979


namespace NUMINAMATH_CALUDE_quadrilateral_area_l479_47992

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the quadrilateral ABCD
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the conditions
def inscribed_in_parabola (q : Quadrilateral) : Prop :=
  parabola q.A.1 = q.A.2 ∧ parabola q.B.1 = q.B.2 ∧
  parabola q.C.1 = q.C.2 ∧ parabola q.D.1 = q.D.2

def angle_BAD_is_right (q : Quadrilateral) : Prop :=
  (q.B.1 - q.A.1) * (q.D.1 - q.A.1) + (q.B.2 - q.A.2) * (q.D.2 - q.A.2) = 0

def AC_parallel_to_x_axis (q : Quadrilateral) : Prop :=
  q.A.2 = q.C.2

def AC_bisects_BAD (q : Quadrilateral) : Prop :=
  (q.C.1 - q.A.1)^2 + (q.C.2 - q.A.2)^2 =
  (q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2

def diagonal_BD_length (q : Quadrilateral) (p : ℝ) : Prop :=
  (q.B.1 - q.D.1)^2 + (q.B.2 - q.D.2)^2 = p^2

-- The theorem
theorem quadrilateral_area (q : Quadrilateral) (p : ℝ) :
  inscribed_in_parabola q →
  angle_BAD_is_right q →
  AC_parallel_to_x_axis q →
  AC_bisects_BAD q →
  diagonal_BD_length q p →
  (q.A.1 - q.C.1) * (q.B.2 - q.D.2) / 2 = (p^2 - 4) / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l479_47992


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l479_47940

/-- Given an arithmetic sequence a with common difference d ≠ 0,
    where a₁, a₃, a₉ form a geometric sequence,
    prove that (a₁ + a₃ + a₆) / (a₂ + a₄ + a₁₀) = 5/8 -/
theorem arithmetic_geometric_sequence_ratio
  (a : ℕ → ℚ) (d : ℚ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_d_nonzero : d ≠ 0)
  (h_geometric : ∃ r, a 3 = a 1 * r ∧ a 9 = a 3 * r) :
  (a 1 + a 3 + a 6) / (a 2 + a 4 + a 10) = 5/8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l479_47940


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l479_47932

theorem product_of_sum_and_difference (x y : ℝ) (h1 : x > y) (h2 : x + y = 20) (h3 : x - y = 4) :
  (3 * x) * y = 288 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l479_47932


namespace NUMINAMATH_CALUDE_min_tries_for_given_counts_l479_47922

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  purple : Nat
  blue : Nat
  yellow : Nat
  green : Nat

/-- The minimum number of tries required to get at least two blue, two yellow, and one green ball -/
def minTriesRequired (counts : BallCounts) : Nat :=
  counts.purple + (counts.yellow - 1) + (counts.green - 1) + 2 + 2

/-- Theorem stating the minimum number of tries required for the given ball counts -/
theorem min_tries_for_given_counts :
  let counts : BallCounts := ⟨9, 7, 13, 6⟩
  minTriesRequired counts = 30 := by sorry

end NUMINAMATH_CALUDE_min_tries_for_given_counts_l479_47922


namespace NUMINAMATH_CALUDE_pizza_area_difference_l479_47974

theorem pizza_area_difference : ∃ (N : ℝ), 
  (abs (N - 96) < 1) ∧ 
  (π * 7^2 = π * 5^2 * (1 + N / 100)) := by
  sorry

end NUMINAMATH_CALUDE_pizza_area_difference_l479_47974


namespace NUMINAMATH_CALUDE_expression_evaluation_l479_47907

theorem expression_evaluation : (100 - (5000 - 500)) * (5000 - (500 - 100)) = -20240000 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l479_47907


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_l479_47913

theorem square_area_from_rectangle (circle_radius : ℝ) (rectangle_length rectangle_breadth rectangle_area square_side : ℝ) : 
  rectangle_length = (2 / 3) * circle_radius →
  circle_radius = square_side →
  rectangle_area = 598 →
  rectangle_breadth = 13 →
  rectangle_area = rectangle_length * rectangle_breadth →
  square_side ^ 2 = 4761 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_l479_47913


namespace NUMINAMATH_CALUDE_square_tiling_for_n_ge_5_l479_47916

/-- A rectangle is dominant if it is similar to a 2 × 1 rectangle -/
def DominantRectangle (r : Rectangle) : Prop := sorry

/-- A tiling of a square with n dominant rectangles -/
def SquareTiling (n : ℕ) : Prop := sorry

/-- Theorem: For all integers n ≥ 5, it is possible to tile a square with n dominant rectangles -/
theorem square_tiling_for_n_ge_5 (n : ℕ) (h : n ≥ 5) : SquareTiling n := by
  sorry

end NUMINAMATH_CALUDE_square_tiling_for_n_ge_5_l479_47916


namespace NUMINAMATH_CALUDE_davids_purchase_cost_l479_47998

/-- The minimum cost to buy a given number of bottles, given the price of individual bottles and packs --/
def min_cost (single_price : ℚ) (pack_price : ℚ) (pack_size : ℕ) (total_bottles : ℕ) : ℚ :=
  let num_packs := total_bottles / pack_size
  let remaining_bottles := total_bottles % pack_size
  num_packs * pack_price + remaining_bottles * single_price

/-- Theorem stating the minimum cost for David's purchase --/
theorem davids_purchase_cost :
  let single_price : ℚ := 280 / 100  -- $2.80
  let pack_price : ℚ := 1500 / 100   -- $15.00
  let pack_size : ℕ := 6
  let total_bottles : ℕ := 22
  min_cost single_price pack_price pack_size total_bottles = 5620 / 100 := by
  sorry


end NUMINAMATH_CALUDE_davids_purchase_cost_l479_47998


namespace NUMINAMATH_CALUDE_simplify_expression_l479_47957

theorem simplify_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l479_47957


namespace NUMINAMATH_CALUDE_negative_sqrt_eleven_squared_l479_47925

theorem negative_sqrt_eleven_squared : (-Real.sqrt 11)^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_eleven_squared_l479_47925


namespace NUMINAMATH_CALUDE_four_number_equation_solutions_l479_47969

def is_solution (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  x₁ + x₂*x₃*x₄ = 2 ∧
  x₂ + x₁*x₃*x₄ = 2 ∧
  x₃ + x₁*x₂*x₄ = 2 ∧
  x₄ + x₁*x₂*x₃ = 2

theorem four_number_equation_solutions :
  ∀ x₁ x₂ x₃ x₄ : ℝ, is_solution x₁ x₂ x₃ x₄ ↔
    ((x₁, x₂, x₃, x₄) = (1, 1, 1, 1) ∨
     (x₁, x₂, x₃, x₄) = (-1, -1, -1, 3) ∨
     (x₁, x₂, x₃, x₄) = (-1, -1, 3, -1) ∨
     (x₁, x₂, x₃, x₄) = (-1, 3, -1, -1) ∨
     (x₁, x₂, x₃, x₄) = (3, -1, -1, -1)) :=
by sorry


end NUMINAMATH_CALUDE_four_number_equation_solutions_l479_47969


namespace NUMINAMATH_CALUDE_system_of_equations_l479_47956

theorem system_of_equations (a b c d k : ℝ) 
  (h1 : a + b = 11)
  (h2 : b^2 + c^2 = k)
  (h3 : b + c = 9)
  (h4 : c + d = 3)
  (h5 : k > 0) :
  a + d = 5 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l479_47956


namespace NUMINAMATH_CALUDE_circular_track_length_l479_47990

/-- The length of the circular track in meters -/
def track_length : ℝ := 480

/-- Alex's speed in meters per unit time -/
def alex_speed : ℝ := 4

/-- Jamie's speed in meters per unit time -/
def jamie_speed : ℝ := 3

/-- Distance Alex runs to first meeting point in meters -/
def alex_first_meeting : ℝ := 150

/-- Distance Jamie runs after first meeting to second meeting point in meters -/
def jamie_second_meeting : ℝ := 180

theorem circular_track_length :
  track_length = 480 ∧
  alex_speed / jamie_speed = 4 / 3 ∧
  alex_first_meeting = 150 ∧
  jamie_second_meeting = 180 ∧
  track_length / 2 + alex_first_meeting = 
    (track_length / 2 - alex_first_meeting) + jamie_second_meeting + track_length / 2 :=
by sorry

end NUMINAMATH_CALUDE_circular_track_length_l479_47990


namespace NUMINAMATH_CALUDE_function_local_max_condition_l479_47959

/-- Given a real constant a, prove that for a function f(x) = (x-a)²(x+b)e^x 
    where b is real and x=a is a local maximum point of f(x), 
    then b must be less than -a. -/
theorem function_local_max_condition (a : ℝ) :
  ∀ b : ℝ, (∃ f : ℝ → ℝ, 
    (∀ x : ℝ, f x = (x - a)^2 * (x + b) * Real.exp x) ∧
    (IsLocalMax f a)) →
  b < -a :=
by sorry

end NUMINAMATH_CALUDE_function_local_max_condition_l479_47959


namespace NUMINAMATH_CALUDE_min_value_polynomial_l479_47923

theorem min_value_polynomial (x : ℝ) : 
  (13 - x) * (11 - x) * (13 + x) * (11 + x) + 1000 ≥ 424 :=
by sorry

end NUMINAMATH_CALUDE_min_value_polynomial_l479_47923


namespace NUMINAMATH_CALUDE_sport_to_standard_ratio_l479_47945

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard : DrinkRatio :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

/-- The sport formulation ratio -/
def sport : DrinkRatio :=
  { flavoring := (15 : ℚ) / 60,
    corn_syrup := 1,
    water := 15 }

/-- The ratio of flavoring to corn syrup for a given formulation -/
def flavoring_to_corn_syrup_ratio (d : DrinkRatio) : ℚ :=
  d.flavoring / d.corn_syrup

theorem sport_to_standard_ratio :
  flavoring_to_corn_syrup_ratio sport / flavoring_to_corn_syrup_ratio standard = 3 := by
  sorry

end NUMINAMATH_CALUDE_sport_to_standard_ratio_l479_47945


namespace NUMINAMATH_CALUDE_function_identity_l479_47914

theorem function_identity (f : ℕ → ℕ) 
  (h1 : f 1 > 0)
  (h2 : ∀ m n : ℕ, f (m^2 + n^2) = (f m)^2 + (f n)^2) :
  ∀ n : ℕ, f n = n :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l479_47914


namespace NUMINAMATH_CALUDE_john_total_spent_l479_47955

def tshirt_price : ℕ := 20
def num_tshirts : ℕ := 3
def pants_price : ℕ := 50

def total_spent : ℕ := tshirt_price * num_tshirts + pants_price

theorem john_total_spent : total_spent = 110 := by
  sorry

end NUMINAMATH_CALUDE_john_total_spent_l479_47955


namespace NUMINAMATH_CALUDE_cafeteria_pies_l479_47977

/-- Given a cafeteria with total apples, apples handed out, and apples needed per pie,
    calculate the number of pies that can be made. -/
def pies_made (total_apples handed_out apples_per_pie : ℕ) : ℕ :=
  (total_apples - handed_out) / apples_per_pie

/-- Theorem: The cafeteria can make 9 pies with the given conditions. -/
theorem cafeteria_pies :
  pies_made 525 415 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l479_47977
