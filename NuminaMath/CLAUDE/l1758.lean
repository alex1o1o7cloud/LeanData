import Mathlib

namespace NUMINAMATH_CALUDE_trigonometric_identity_l1758_175834

theorem trigonometric_identity (α : ℝ) : 
  Real.cos (4 * α) + Real.cos (3 * α) = 2 * Real.cos ((7 * α) / 2) * Real.cos (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1758_175834


namespace NUMINAMATH_CALUDE_supporting_pillars_concrete_l1758_175899

/-- The amount of concrete needed for a bridge construction --/
structure BridgeConcrete where
  roadwayDeck : ℕ
  oneAnchor : ℕ
  totalBridge : ℕ

/-- Calculates the amount of concrete needed for supporting pillars --/
def supportingPillarsAmount (b : BridgeConcrete) : ℕ :=
  b.totalBridge - (b.roadwayDeck + 2 * b.oneAnchor)

/-- Theorem stating the amount of concrete needed for supporting pillars --/
theorem supporting_pillars_concrete (b : BridgeConcrete) 
  (h1 : b.roadwayDeck = 1600)
  (h2 : b.oneAnchor = 700)
  (h3 : b.totalBridge = 4800) :
  supportingPillarsAmount b = 1800 := by
  sorry

#eval supportingPillarsAmount ⟨1600, 700, 4800⟩

end NUMINAMATH_CALUDE_supporting_pillars_concrete_l1758_175899


namespace NUMINAMATH_CALUDE_parallel_lines_k_l1758_175849

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 5x - 3 and y = (3k)x + 7 are parallel -/
theorem parallel_lines_k : ∃ k : ℝ, 
  (∀ x y : ℝ, y = 5 * x - 3 ↔ y = (3 * k) * x + 7) ↔ k = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_l1758_175849


namespace NUMINAMATH_CALUDE_johns_original_earnings_l1758_175806

/-- Given that John's weekly earnings increased by 20% to $72, prove that his original weekly earnings were $60. -/
theorem johns_original_earnings (current_earnings : ℝ) (increase_rate : ℝ) : 
  current_earnings = 72 ∧ increase_rate = 0.20 → 
  current_earnings / (1 + increase_rate) = 60 := by
  sorry

end NUMINAMATH_CALUDE_johns_original_earnings_l1758_175806


namespace NUMINAMATH_CALUDE_midpoint_segments_equal_l1758_175815

/-- A structure representing a rectangle with a circle intersection --/
structure RectangleWithCircle where
  /-- The rectangle --/
  rectangle : Set (ℝ × ℝ)
  /-- The circle --/
  circle : Set (ℝ × ℝ)
  /-- The four right triangles formed by the intersection --/
  triangles : Fin 4 → Set (ℝ × ℝ)
  /-- The midpoints of the hypotenuses of the triangles --/
  midpoints : Fin 4 → ℝ × ℝ

/-- The theorem stating that A₀C₀ = B₀D₀ --/
theorem midpoint_segments_equal (rc : RectangleWithCircle) :
  dist (rc.midpoints 0) (rc.midpoints 2) = dist (rc.midpoints 1) (rc.midpoints 3) :=
sorry

end NUMINAMATH_CALUDE_midpoint_segments_equal_l1758_175815


namespace NUMINAMATH_CALUDE_monomial_product_l1758_175832

theorem monomial_product (a b : ℤ) (x y : ℝ) (h1 : 4 * a - b = 2) (h2 : a + b = 3) :
  (-2 * x^(4*a-b) * y^3) * ((1/2) * x^2 * y^(a+b)) = -x^4 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_monomial_product_l1758_175832


namespace NUMINAMATH_CALUDE_janes_stick_length_l1758_175822

-- Define the lengths of the sticks and other quantities
def pat_stick_length : ℕ := 30
def covered_length : ℕ := 7
def feet_to_inches : ℕ := 12

-- Define the theorem
theorem janes_stick_length :
  let uncovered_length : ℕ := pat_stick_length - covered_length
  let sarahs_stick_length : ℕ := 2 * uncovered_length
  let janes_stick_length : ℕ := sarahs_stick_length - 2 * feet_to_inches
  janes_stick_length = 22 := by
sorry

end NUMINAMATH_CALUDE_janes_stick_length_l1758_175822


namespace NUMINAMATH_CALUDE_count_integers_satisfying_conditions_l1758_175878

theorem count_integers_satisfying_conditions :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ 9 ∣ n ∧ Nat.lcm (Nat.factorial 6) n = 9 * Nat.gcd (Nat.factorial 9) n) ∧
    S.card = 30 ∧
    (∀ n : ℕ, n > 0 → 9 ∣ n → Nat.lcm (Nat.factorial 6) n = 9 * Nat.gcd (Nat.factorial 9) n → n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_conditions_l1758_175878


namespace NUMINAMATH_CALUDE_triangle_side_length_l1758_175887

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1758_175887


namespace NUMINAMATH_CALUDE_heather_walk_distance_l1758_175838

/-- The total distance Heather walked at the county fair -/
def total_distance (d1 d2 d3 : ℝ) : ℝ := d1 + d2 + d3

/-- Theorem stating the total distance Heather walked -/
theorem heather_walk_distance :
  let d1 : ℝ := 0.33  -- Distance from car to entrance
  let d2 : ℝ := 0.33  -- Distance to carnival rides
  let d3 : ℝ := 0.08  -- Distance from carnival rides back to car
  total_distance d1 d2 d3 = 0.74 := by
  sorry

end NUMINAMATH_CALUDE_heather_walk_distance_l1758_175838


namespace NUMINAMATH_CALUDE_probability_four_white_balls_l1758_175829

/-- The number of white balls in the box -/
def white_balls : ℕ := 7

/-- The number of black balls in the box -/
def black_balls : ℕ := 8

/-- The total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 5

/-- The number of white balls we want to draw -/
def target_white : ℕ := 4

/-- The number of black balls we want to draw -/
def target_black : ℕ := drawn_balls - target_white

theorem probability_four_white_balls : 
  (Nat.choose white_balls target_white * Nat.choose black_balls target_black : ℚ) / 
  Nat.choose total_balls drawn_balls = 280 / 3003 := by
sorry

end NUMINAMATH_CALUDE_probability_four_white_balls_l1758_175829


namespace NUMINAMATH_CALUDE_ball_bearing_savings_ball_bearing_savings_correct_l1758_175866

/-- Calculates the savings when buying ball bearings during a sale with a bulk discount -/
theorem ball_bearing_savings
  (num_machines : ℕ)
  (bearings_per_machine : ℕ)
  (regular_price : ℚ)
  (sale_price : ℚ)
  (bulk_discount : ℚ)
  (h1 : num_machines = 10)
  (h2 : bearings_per_machine = 30)
  (h3 : regular_price = 1)
  (h4 : sale_price = 3/4)
  (h5 : bulk_discount = 1/5)
  : ℚ :=
  let total_bearings := num_machines * bearings_per_machine
  let regular_cost := total_bearings * regular_price
  let sale_cost := total_bearings * sale_price
  let discounted_cost := sale_cost * (1 - bulk_discount)
  let savings := regular_cost - discounted_cost
  120

theorem ball_bearing_savings_correct : ball_bearing_savings 10 30 1 (3/4) (1/5) rfl rfl rfl rfl rfl = 120 := by
  sorry

end NUMINAMATH_CALUDE_ball_bearing_savings_ball_bearing_savings_correct_l1758_175866


namespace NUMINAMATH_CALUDE_sum_bound_l1758_175856

theorem sum_bound (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_sum : a + b + c = 1) :
  let S := 1 / (1 + a) + 1 / (1 + b) + 1 / (1 + c)
  9 / 4 ≤ S ∧ S ≤ 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_bound_l1758_175856


namespace NUMINAMATH_CALUDE_max_ab_value_l1758_175886

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃! x, x^2 + Real.sqrt a * x - b + 1/4 = 0) → 
  ∀ c, a * b ≤ c → c ≤ 1/16 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l1758_175886


namespace NUMINAMATH_CALUDE_smart_car_competition_probability_l1758_175870

/-- The probability of selecting exactly 4 girls when randomly choosing 10 people
    from a group of 15 people (7 girls and 8 boys) -/
theorem smart_car_competition_probability :
  let total_members : ℕ := 15
  let girls : ℕ := 7
  let boys : ℕ := total_members - girls
  let selected : ℕ := 10
  let prob_four_girls := (Nat.choose girls 4 * Nat.choose boys 6 : ℚ) / Nat.choose total_members selected
  prob_four_girls = (Nat.choose girls 4 * Nat.choose boys 6 : ℚ) / Nat.choose total_members selected :=
by sorry

end NUMINAMATH_CALUDE_smart_car_competition_probability_l1758_175870


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1758_175823

/-- The polynomial in question -/
def P (x m : ℝ) : ℝ := (x-1)*(x+3)*(x-4)*(x-8) + m

/-- The polynomial is a perfect square -/
def is_perfect_square (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x, f x = (g x)^2

theorem perfect_square_condition :
  ∃! m : ℝ, is_perfect_square (P · m) ∧ m = 196 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1758_175823


namespace NUMINAMATH_CALUDE_arrow_balance_l1758_175810

/-- A polygon with arrows on its sides. -/
structure ArrowPolygon where
  n : ℕ  -- number of sides/vertices
  arrows : Fin n → Bool  -- True if arrow points clockwise, False if counterclockwise

/-- The number of vertices with two incoming arrows. -/
def incoming_two (p : ArrowPolygon) : ℕ := sorry

/-- The number of vertices with two outgoing arrows. -/
def outgoing_two (p : ArrowPolygon) : ℕ := sorry

/-- Theorem: The number of vertices with two incoming arrows equals the number of vertices with two outgoing arrows. -/
theorem arrow_balance (p : ArrowPolygon) : incoming_two p = outgoing_two p := by sorry

end NUMINAMATH_CALUDE_arrow_balance_l1758_175810


namespace NUMINAMATH_CALUDE_class_average_l1758_175833

theorem class_average (total_students : ℕ) (top_scorers : ℕ) (top_score : ℕ) (zero_scorers : ℕ) (rest_average : ℕ) :
  total_students = 25 →
  top_scorers = 3 →
  top_score = 95 →
  zero_scorers = 5 →
  rest_average = 45 →
  (top_scorers * top_score + zero_scorers * 0 + (total_students - top_scorers - zero_scorers) * rest_average) / total_students = 42 := by
  sorry

end NUMINAMATH_CALUDE_class_average_l1758_175833


namespace NUMINAMATH_CALUDE_probability_prime_or_odd_l1758_175884

/-- A function that determines if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that determines if a natural number is odd -/
def isOdd (n : ℕ) : Prop := sorry

/-- The set of balls numbered 1 through 8 -/
def ballSet : Finset ℕ := sorry

/-- The probability of selecting a ball with a number that is either prime or odd -/
def probabilityPrimeOrOdd : ℚ := sorry

/-- Theorem stating that the probability of selecting a ball with a number
    that is either prime or odd is 5/8 -/
theorem probability_prime_or_odd :
  probabilityPrimeOrOdd = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_prime_or_odd_l1758_175884


namespace NUMINAMATH_CALUDE_min_value_problem_l1758_175831

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (x^2 + y^2 + x) / (x*y) ≥ 2*Real.sqrt 2 + 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1758_175831


namespace NUMINAMATH_CALUDE_blue_balls_removed_l1758_175805

theorem blue_balls_removed (initial_total : ℕ) (initial_blue : ℕ) (final_probability : ℚ) : ℕ :=
  let removed : ℕ := 3
  have h1 : initial_total = 18 := by sorry
  have h2 : initial_blue = 6 := by sorry
  have h3 : final_probability = 1 / 5 := by sorry
  have h4 : (initial_blue - removed : ℚ) / (initial_total - removed) = final_probability := by sorry
  removed

#check blue_balls_removed

end NUMINAMATH_CALUDE_blue_balls_removed_l1758_175805


namespace NUMINAMATH_CALUDE_particular_number_proof_l1758_175877

theorem particular_number_proof (x : ℚ) : x / 4 + 3 = 5 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_proof_l1758_175877


namespace NUMINAMATH_CALUDE_quadratic_sum_theorem_l1758_175882

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex_x (f : QuadraticFunction) : ℚ := -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
def vertex_y (f : QuadraticFunction) : ℚ := f.c - f.b^2 / (4 * f.a)

/-- Theorem: For a quadratic function with integer coefficients and vertex at (2, -3),
    the sum a + b - c equals -4 -/
theorem quadratic_sum_theorem (f : QuadraticFunction) 
  (h1 : vertex_x f = 2)
  (h2 : vertex_y f = -3) :
  f.a + f.b - f.c = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_theorem_l1758_175882


namespace NUMINAMATH_CALUDE_square_root_of_nine_l1758_175888

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {-3, 3} := by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l1758_175888


namespace NUMINAMATH_CALUDE_triangle_problem_l1758_175841

open Real

theorem triangle_problem (A B C : ℝ) (a b c S : ℝ) :
  (2 * sin B - 2 * sin B ^ 2 - cos (2 * B) = sqrt 3 - 1) →
  (B = π / 3 ∨ B = 2 * π / 3) ∧
  (B = π / 3 ∧ a = 6 ∧ S = 6 * sqrt 3 → b = 2 * sqrt 7) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1758_175841


namespace NUMINAMATH_CALUDE_height_comparison_l1758_175857

theorem height_comparison (h_a h_b h_c : ℝ) 
  (h_a_def : h_a = 0.6 * h_b) 
  (h_c_def : h_c = 1.25 * h_a) : 
  (h_b - h_a) / h_a = 2/3 ∧ (h_c - h_a) / h_a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l1758_175857


namespace NUMINAMATH_CALUDE_polyhedron_20_faces_l1758_175890

/-- A polyhedron with triangular faces -/
structure Polyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ
  triangular_faces : faces * 3 = edges * 2
  euler_formula : vertices - edges + faces = 2

/-- Theorem: A polyhedron with 20 triangular faces has 12 vertices and 30 edges -/
theorem polyhedron_20_faces (P : Polyhedron) (h : P.faces = 20) : 
  P.vertices = 12 ∧ P.edges = 30 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_20_faces_l1758_175890


namespace NUMINAMATH_CALUDE_rearrangement_theorem_l1758_175824

/-- The number of ways to choose and rearrange 3 people from a group of 7 -/
def rearrangement_count : ℕ := 70

/-- The number of people in the class -/
def class_size : ℕ := 7

/-- The number of people to be rearranged -/
def rearrange_size : ℕ := 3

/-- The number of ways to derange 3 people -/
def derangement_3 : ℕ := 2

theorem rearrangement_theorem : 
  rearrangement_count = derangement_3 * (class_size.choose rearrange_size) := by
  sorry

end NUMINAMATH_CALUDE_rearrangement_theorem_l1758_175824


namespace NUMINAMATH_CALUDE_solve_for_m_l1758_175847

theorem solve_for_m : ∃ m : ℤ, 5^2 + 7 = 4^3 + m ∧ m = -32 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l1758_175847


namespace NUMINAMATH_CALUDE_transform_to_successor_l1758_175860

/-- Represents the allowed operations on natural numbers -/
inductive Operation
  | AddNine
  | EraseOne

/-- Applies a single operation to a natural number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddNine => n + 9
  | Operation.EraseOne => sorry  -- Implementation of erasing 1 is complex and not provided

/-- Applies a sequence of operations to a natural number -/
def applyOperations (n : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation n

/-- 
Theorem: For any natural number A, there exists a sequence of operations 
that transforms A into A+1
-/
theorem transform_to_successor (A : ℕ) : 
  ∃ (ops : List Operation), applyOperations A ops = A + 1 :=
sorry

end NUMINAMATH_CALUDE_transform_to_successor_l1758_175860


namespace NUMINAMATH_CALUDE_line_equation_transformation_l1758_175865

/-- Given a line l: Ax + By + C = 0 and a point (x₀, y₀) on the line,
    prove that the line equation can be transformed to A(x - x₀) + B(y - y₀) = 0 -/
theorem line_equation_transformation 
  (A B C x₀ y₀ : ℝ) 
  (h1 : A ≠ 0 ∨ B ≠ 0) 
  (h2 : A * x₀ + B * y₀ + C = 0) :
  ∀ x y, A * x + B * y + C = 0 ↔ A * (x - x₀) + B * (y - y₀) = 0 :=
sorry

end NUMINAMATH_CALUDE_line_equation_transformation_l1758_175865


namespace NUMINAMATH_CALUDE_rectangle_opposite_sides_l1758_175879

/-- A parallelogram is a quadrilateral with opposite sides parallel and equal. -/
structure Parallelogram where
  opposite_sides_parallel : Bool
  opposite_sides_equal : Bool

/-- A rectangle is a special case of a parallelogram with right angles. -/
structure Rectangle extends Parallelogram where
  right_angles : Bool

/-- Deductive reasoning is a method of logical reasoning that uses general rules to reach a specific conclusion. -/
def DeductiveReasoning : Prop := True

/-- The reasoning method used in the given statement. -/
def reasoning_method : Prop := DeductiveReasoning

theorem rectangle_opposite_sides (p : Parallelogram) (r : Rectangle) :
  p.opposite_sides_parallel ∧ p.opposite_sides_equal →
  r.opposite_sides_parallel ∧ r.opposite_sides_equal →
  reasoning_method := by sorry

end NUMINAMATH_CALUDE_rectangle_opposite_sides_l1758_175879


namespace NUMINAMATH_CALUDE_monotone_sine_range_l1758_175880

/-- The function f(x) = 2sin(ωx) is monotonically increasing on [-π/4, 2π/3] if and only if ω is in (0, 3/4] -/
theorem monotone_sine_range (ω : ℝ) (h : ω > 0) :
  StrictMonoOn (fun x => 2 * Real.sin (ω * x)) (Set.Icc (-π/4) (2*π/3)) ↔ ω ∈ Set.Ioo 0 (3/4) ∪ {3/4} := by
  sorry

end NUMINAMATH_CALUDE_monotone_sine_range_l1758_175880


namespace NUMINAMATH_CALUDE_solve_equation_l1758_175892

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1758_175892


namespace NUMINAMATH_CALUDE_green_fish_count_l1758_175830

theorem green_fish_count (T : ℕ) : ℕ := by
  -- Define the number of blue fish
  let blue : ℕ := T / 2

  -- Define the number of orange fish
  let orange : ℕ := blue - 15

  -- Define the number of green fish
  let green : ℕ := T - blue - orange

  -- Prove that green = 15
  sorry

end NUMINAMATH_CALUDE_green_fish_count_l1758_175830


namespace NUMINAMATH_CALUDE_parallelogram_revolution_surface_area_l1758_175818

/-- The surface area of a solid of revolution formed by rotating a parallelogram -/
theorem parallelogram_revolution_surface_area
  (p d : ℝ)
  (perimeter_positive : p > 0)
  (diagonal_positive : d > 0) :
  let perimeter := 2 * p
  let diagonal := d
  let surface_area := 2 * Real.pi * d * p
  surface_area = 2 * Real.pi * diagonal * (perimeter / 2) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_revolution_surface_area_l1758_175818


namespace NUMINAMATH_CALUDE_monotonic_increasing_sequence_l1758_175858

/-- A sequence {a_n} with general term a_n = n^2 + bn is monotonically increasing if and only if b > -3 -/
theorem monotonic_increasing_sequence (b : ℝ) :
  (∀ n : ℕ, (n : ℝ)^2 + b * n < ((n + 1) : ℝ)^2 + b * (n + 1)) ↔ b > -3 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_sequence_l1758_175858


namespace NUMINAMATH_CALUDE_square_perimeter_l1758_175891

/-- The perimeter of a square with side length 13 centimeters is 52 centimeters. -/
theorem square_perimeter : ∀ (s : ℝ), s = 13 → 4 * s = 52 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1758_175891


namespace NUMINAMATH_CALUDE_frog_jump_difference_l1758_175889

/-- The frog's jump distance in inches -/
def frog_jump : ℕ := 39

/-- The grasshopper's jump distance in inches -/
def grasshopper_jump : ℕ := 17

/-- Theorem: The frog jumped 22 inches farther than the grasshopper -/
theorem frog_jump_difference : frog_jump - grasshopper_jump = 22 := by
  sorry

end NUMINAMATH_CALUDE_frog_jump_difference_l1758_175889


namespace NUMINAMATH_CALUDE_special_number_prime_iff_l1758_175802

/-- Represents a natural number formed by one digit 7 and n-1 digits 1 -/
def special_number (n : ℕ) : ℕ :=
  7 * 10^(n-1) + (10^(n-1) - 1) / 9

/-- Predicate to check if a natural number is prime -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

/-- The main theorem stating that only n = 1 and n = 2 satisfy the condition -/
theorem special_number_prime_iff (n : ℕ) :
  (∀ k : ℕ, k ≤ n → is_prime (special_number k)) ↔ n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_CALUDE_special_number_prime_iff_l1758_175802


namespace NUMINAMATH_CALUDE_jays_savings_l1758_175876

def savings_sequence (n : ℕ) : ℕ := 20 + 10 * n

def total_savings (weeks : ℕ) : ℕ :=
  (List.range weeks).map savings_sequence |> List.sum

theorem jays_savings : total_savings 4 = 140 := by
  sorry

end NUMINAMATH_CALUDE_jays_savings_l1758_175876


namespace NUMINAMATH_CALUDE_no_solution_equation_l1758_175881

theorem no_solution_equation :
  ¬ ∃ x : ℝ, (x + 2) / (x - 2) - x / (x + 2) = 16 / (x^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_equation_l1758_175881


namespace NUMINAMATH_CALUDE_point_coordinates_on_terminal_side_l1758_175846

/-- Given a point P on the terminal side of -π/4 with |OP| = 2, prove its coordinates are (√2, -√2) -/
theorem point_coordinates_on_terminal_side (P : ℝ × ℝ) :
  (P.1 = Real.sqrt 2 ∧ P.2 = -Real.sqrt 2) ↔
  (∃ (r : ℝ), r > 0 ∧ P.1 = r * Real.cos (-π/4) ∧ P.2 = r * Real.sin (-π/4) ∧ r^2 = P.1^2 + P.2^2 ∧ r = 2) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_on_terminal_side_l1758_175846


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1758_175897

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) : 
  C = π / 4 →  -- 45° in radians
  c = Real.sqrt 2 → 
  a = Real.sqrt 3 → 
  (A = π / 3 ∨ A = 2 * π / 3) -- 60° or 120° in radians
  :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1758_175897


namespace NUMINAMATH_CALUDE_xyz_value_l1758_175845

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 37)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) :
  x * y * z = 26 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1758_175845


namespace NUMINAMATH_CALUDE_smallest_n_ending_same_as_n_squared_l1758_175873

theorem smallest_n_ending_same_as_n_squared : 
  ∃ (N : ℕ), 
    N > 0 ∧ 
    (N % 1000 = N^2 % 1000) ∧ 
    (N ≥ 100) ∧
    (∀ (M : ℕ), M > 0 ∧ M < N → (M % 1000 ≠ M^2 % 1000 ∨ M < 100)) ∧ 
    N = 376 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_ending_same_as_n_squared_l1758_175873


namespace NUMINAMATH_CALUDE_min_elements_in_set_l1758_175807

theorem min_elements_in_set (S : Type) [Fintype S] 
  (X : Fin 100 → Set S)
  (h_nonempty : ∀ i, Set.Nonempty (X i))
  (h_distinct : ∀ i j, i ≠ j → X i ≠ X j)
  (h_disjoint : ∀ i : Fin 99, Disjoint (X i) (X (Fin.succ i)))
  (h_not_union : ∀ i : Fin 99, (X i) ∪ (X (Fin.succ i)) ≠ Set.univ) :
  Fintype.card S ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_elements_in_set_l1758_175807


namespace NUMINAMATH_CALUDE_shirt_shoe_cost_multiple_l1758_175863

/-- The multiple of the cost of the shirt that represents the cost of the shoes -/
def multiple_of_shirt_cost (total_cost shirt_cost shoe_cost : ℚ) : ℚ :=
  (shoe_cost - 9) / shirt_cost

theorem shirt_shoe_cost_multiple :
  let total_cost : ℚ := 300
  let shirt_cost : ℚ := 97
  let shoe_cost : ℚ := total_cost - shirt_cost
  shoe_cost = multiple_of_shirt_cost total_cost shirt_cost shoe_cost * shirt_cost + 9 →
  multiple_of_shirt_cost total_cost shirt_cost shoe_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_shirt_shoe_cost_multiple_l1758_175863


namespace NUMINAMATH_CALUDE_translation_theorem_l1758_175894

def original_function (x : ℝ) : ℝ := (x - 2)^2 + 1

def translated_function (x : ℝ) : ℝ := original_function (x + 2) - 2

theorem translation_theorem :
  ∀ x : ℝ, translated_function x = x^2 - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l1758_175894


namespace NUMINAMATH_CALUDE_f_4_equals_24_l1758_175825

-- Define the function f recursively
def f : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * f n

-- State the theorem
theorem f_4_equals_24 : f 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_f_4_equals_24_l1758_175825


namespace NUMINAMATH_CALUDE_range_of_a_l1758_175874

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (|4*x - 3| ≤ 1 → x^2 - (2*a + 1)*x + a^2 + a ≤ 0) ∧
   ¬(∀ x : ℝ, x^2 - (2*a + 1)*x + a^2 + a ≤ 0 → |4*x - 3| ≤ 1)) →
  0 ≤ a ∧ a ≤ 1/2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1758_175874


namespace NUMINAMATH_CALUDE_revenue_decrease_percent_l1758_175813

/-- Calculates the decrease percent in revenue when tax is reduced and consumption is increased -/
theorem revenue_decrease_percent 
  (original_tax : ℝ) 
  (original_consumption : ℝ) 
  (tax_reduction_percent : ℝ) 
  (consumption_increase_percent : ℝ) 
  (h1 : tax_reduction_percent = 22) 
  (h2 : consumption_increase_percent = 9) 
  : (1 - (1 - tax_reduction_percent / 100) * (1 + consumption_increase_percent / 100)) * 100 = 15.02 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_percent_l1758_175813


namespace NUMINAMATH_CALUDE_abcd_sum_proof_l1758_175855

/-- Given four different digits A, B, C, and D forming a four-digit number ABCD,
    prove that if ABCD + ABCD = 7314, then ABCD = 3657 -/
theorem abcd_sum_proof (A B C D : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (h2 : 1000 ≤ A * 1000 + B * 100 + C * 10 + D ∧ A * 1000 + B * 100 + C * 10 + D < 10000)
    (h3 : (A * 1000 + B * 100 + C * 10 + D) + (A * 1000 + B * 100 + C * 10 + D) = 7314) :
  A * 1000 + B * 100 + C * 10 + D = 3657 := by
  sorry

end NUMINAMATH_CALUDE_abcd_sum_proof_l1758_175855


namespace NUMINAMATH_CALUDE_train_meeting_distance_l1758_175826

/-- Represents the distance traveled by a train given its speed and time -/
def distanceTraveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents the total initial distance between the trains -/
def totalDistance : ℝ := 350

/-- Represents the speed of Train A in miles per hour -/
def speedA : ℝ := 40

/-- Represents the speed of Train B in miles per hour -/
def speedB : ℝ := 30

/-- Theorem stating that Train A will have traveled 200 miles when the trains meet -/
theorem train_meeting_distance :
  ∃ (t : ℝ), t > 0 ∧ 
  distanceTraveled speedA t + distanceTraveled speedB t = totalDistance ∧
  distanceTraveled speedA t = 200 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_distance_l1758_175826


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l1758_175811

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x

-- State the theorem
theorem tangent_slope_at_one : 
  (deriv f) 1 = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l1758_175811


namespace NUMINAMATH_CALUDE_triangle_segment_length_l1758_175864

/-- Given a triangle ABC with point D on AC and point E on AD, prove that FC = 10.125 -/
theorem triangle_segment_length 
  (DC CB : ℝ) 
  (h_DC : DC = 9)
  (h_CB : CB = 6)
  (AD AB ED : ℝ)
  (h_AB : AB = (1/3) * AD)
  (h_ED : ED = (3/4) * AD)
  : ∃ (FC : ℝ), FC = 10.125 := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_length_l1758_175864


namespace NUMINAMATH_CALUDE_greatest_power_of_three_specific_case_l1758_175828

theorem greatest_power_of_three (n : ℕ) : ∃ (k : ℕ), (3^n : ℤ) ∣ (6^n - 3^n) ∧ ¬(3^(n+1) : ℤ) ∣ (6^n - 3^n) :=
by
  sorry

theorem specific_case : ∃ (k : ℕ), (3^1503 : ℤ) ∣ (6^1503 - 3^1503) ∧ ¬(3^1504 : ℤ) ∣ (6^1503 - 3^1503) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_specific_case_l1758_175828


namespace NUMINAMATH_CALUDE_alloy_gold_percentage_l1758_175850

-- Define the weights and percentages
def total_weight : ℝ := 12.4
def metal_weight : ℝ := 6.2
def gold_percent_1 : ℝ := 0.60
def gold_percent_2 : ℝ := 0.40

-- Theorem statement
theorem alloy_gold_percentage :
  (metal_weight * gold_percent_1 + metal_weight * gold_percent_2) / total_weight = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_alloy_gold_percentage_l1758_175850


namespace NUMINAMATH_CALUDE_inverse_g_solution_l1758_175840

noncomputable section

variables (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0)

def g (x : ℝ) : ℝ := 1 / (c * x + d)

theorem inverse_g_solution :
  let x := 1 / (-2 * c + d)
  g x = 1 / 2 := by sorry

end

end NUMINAMATH_CALUDE_inverse_g_solution_l1758_175840


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1758_175896

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1758_175896


namespace NUMINAMATH_CALUDE_series_sum_equals_20_over_3_l1758_175862

/-- The sum of the series (7n+2)/k^n from n=1 to infinity -/
noncomputable def series_sum (k : ℝ) : ℝ := ∑' n, (7 * n + 2) / k^n

/-- Theorem stating that if k > 1 and the series sum equals 20/3, then k = 2.9 -/
theorem series_sum_equals_20_over_3 (k : ℝ) (h1 : k > 1) (h2 : series_sum k = 20/3) : k = 2.9 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_20_over_3_l1758_175862


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l1758_175868

theorem trigonometric_expression_equals_one : 
  (Real.sin (15 * π / 180) * Real.cos (25 * π / 180) + Real.cos (165 * π / 180) * Real.cos (115 * π / 180)) / 
  (Real.sin (35 * π / 180) * Real.cos (5 * π / 180) + Real.cos (145 * π / 180) * Real.cos (85 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l1758_175868


namespace NUMINAMATH_CALUDE_sum_of_abs_values_l1758_175871

theorem sum_of_abs_values (a b : ℝ) : 
  (abs a = 3) → (abs b = 4) → (a < b) → (a + b = 1 ∨ a + b = 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abs_values_l1758_175871


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l1758_175804

theorem triangle_angle_relation (a b c α β γ : ℝ) : 
  b = (a + c) / Real.sqrt 2 →
  β = (α + γ) / 2 →
  c > a →
  γ = α + 90 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l1758_175804


namespace NUMINAMATH_CALUDE_spring_mass_for_length_30_l1758_175843

def spring_length (mass : ℝ) : ℝ := 18 + 2 * mass

theorem spring_mass_for_length_30 :
  ∃ (mass : ℝ), spring_length mass = 30 ∧ mass = 6 :=
by sorry

end NUMINAMATH_CALUDE_spring_mass_for_length_30_l1758_175843


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1758_175817

/-- A quadratic function satisfying certain conditions -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem statement -/
theorem quadratic_function_theorem (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, f a b c (x - 4) = f a b c (2 - x)) →
  (∀ x : ℝ, f a b c x ≥ x) →
  (∀ x ∈ Set.Ioo 0 2, f a b c x ≤ ((x + 1) / 2)^2) →
  (∃ x : ℝ, ∀ y : ℝ, f a b c x ≤ f a b c y) →
  (∃ x : ℝ, f a b c x = 0) →
  (∃ m : ℝ, m > 1 ∧ 
    (∀ m' : ℝ, m' > m → 
      ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x) ∧
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x)) ∧
  (∀ m : ℝ, (m > 1 ∧ 
    (∀ m' : ℝ, m' > m → 
      ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x) ∧
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x)) → m = 9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1758_175817


namespace NUMINAMATH_CALUDE_distance_to_point_l1758_175872

theorem distance_to_point : Real.sqrt ((-12 - 0)^2 + (16 - 0)^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_l1758_175872


namespace NUMINAMATH_CALUDE_sum_of_differences_base7_l1758_175816

/-- Converts a base 7 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 7 * acc) 0

/-- Converts a decimal number to its base 7 representation as a list of digits -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec go (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else go (m / 7) ((m % 7) :: acc)
  go n []

/-- Calculates the difference between two base 7 numbers -/
def diffBase7 (a b : List Nat) : List Nat :=
  toBase7 (toDecimal a - toDecimal b)

/-- Calculates the sum of two base 7 numbers -/
def sumBase7 (a b : List Nat) : List Nat :=
  toBase7 (toDecimal a + toDecimal b)

theorem sum_of_differences_base7 :
  let a := [5, 2, 4, 3]
  let b := [3, 1, 0, 5]
  let c := [6, 6, 6, 5]
  let d := [4, 3, 1, 2]
  let result := [4, 4, 5, 2]
  sumBase7 (diffBase7 a b) (diffBase7 c d) = result :=
by sorry

end NUMINAMATH_CALUDE_sum_of_differences_base7_l1758_175816


namespace NUMINAMATH_CALUDE_bargain_bin_books_l1758_175800

def total_books (x y : ℕ) (z : ℚ) : ℚ :=
  (x : ℚ) - (y : ℚ) + z / 100 * (x : ℚ)

theorem bargain_bin_books (x y : ℕ) (z : ℚ) :
  total_books x y z = (x : ℚ) - (y : ℚ) + z / 100 * (x : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_bargain_bin_books_l1758_175800


namespace NUMINAMATH_CALUDE_part_one_part_two_l1758_175867

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

-- Part 1
theorem part_one :
  let a := 2
  (∀ x ∈ Set.Icc (-1) 3, f a x ≥ -2) ∧
  (∃ x ∈ Set.Icc (-1) 3, f a x = -2) ∧
  (∀ x ∈ Set.Icc (-1) 3, f a x ≤ 14) ∧
  (∃ x ∈ Set.Icc (-1) 3, f a x = 14) :=
sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 2, f a x ≥ 3) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 3) ↔
  a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1758_175867


namespace NUMINAMATH_CALUDE_optimal_station_is_75km_l1758_175895

/-- Represents a petrol station with its distance from a given point --/
structure PetrolStation :=
  (distance : ℝ)

/-- Represents a car with its fuel consumption rate --/
structure Car :=
  (consumption : ℝ)  -- litres per km

/-- Represents a journey with various parameters --/
structure Journey :=
  (totalDistance : ℝ)
  (initialFuel : ℝ)
  (initialDriven : ℝ)
  (stations : List PetrolStation)
  (tankCapacity : ℝ)

def Journey.optimalStation (j : Journey) (c : Car) : Option PetrolStation :=
  sorry

theorem optimal_station_is_75km 
  (j : Journey)
  (c : Car)
  (h1 : j.totalDistance = 520)
  (h2 : j.initialFuel = 14)
  (h3 : c.consumption = 0.1)
  (h4 : j.initialDriven = 55)
  (h5 : j.stations = [
    { distance := 35 },
    { distance := 45 },
    { distance := 55 },
    { distance := 75 },
    { distance := 95 }
  ])
  (h6 : j.tankCapacity = 40) :
  (Journey.optimalStation j c).map PetrolStation.distance = some 75 := by
  sorry

end NUMINAMATH_CALUDE_optimal_station_is_75km_l1758_175895


namespace NUMINAMATH_CALUDE_worker_travel_time_l1758_175848

theorem worker_travel_time (T : ℝ) 
  (h1 : T > 0) 
  (h2 : (3/4 : ℝ) * T * (T + 12) = T * T) : 
  T = 36 := by
sorry

end NUMINAMATH_CALUDE_worker_travel_time_l1758_175848


namespace NUMINAMATH_CALUDE_product_expansion_l1758_175852

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * (8 / x^2 - 5 * x^3) = 6 / x^2 - 15 * x^3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l1758_175852


namespace NUMINAMATH_CALUDE_family_movie_night_l1758_175883

/-- Calculates the number of children in a family given ticket prices and payment information. -/
def number_of_children (regular_ticket_price : ℕ) (child_discount : ℕ) (total_payment : ℕ) (change : ℕ) (num_adults : ℕ) : ℕ :=
  let child_ticket_price := regular_ticket_price - child_discount
  let total_spent := total_payment - change
  let adult_tickets_cost := regular_ticket_price * num_adults
  let children_tickets_cost := total_spent - adult_tickets_cost
  children_tickets_cost / child_ticket_price

/-- Proves that the number of children in the family is 3 given the problem conditions. -/
theorem family_movie_night : number_of_children 9 2 40 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_family_movie_night_l1758_175883


namespace NUMINAMATH_CALUDE_quadratic_common_root_l1758_175814

theorem quadratic_common_root (p1 p2 q1 q2 : ℂ) :
  (∃ x : ℂ, x^2 + p1*x + q1 = 0 ∧ x^2 + p2*x + q2 = 0) ↔
  (q2 - q1)^2 + (p1 - p2)*(p1*q2 - q1*p2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_common_root_l1758_175814


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1758_175820

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- A point is on the parabola if its y-coordinate equals f(x) -/
def on_parabola (p : ℝ × ℝ) : Prop := p.2 = f p.1

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- A square is inscribed if its top vertices are on the parabola and bottom vertices are on the x-axis -/
def is_inscribed_square (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), a < b ∧
    s = {(a, 0), (b, 0), (a, b-a), (b, b-a)} ∧
    on_parabola (a, b-a) ∧ on_parabola (b, b-a)

/-- The area of a square with side length s -/
def square_area (s : ℝ) : ℝ := s^2

theorem inscribed_square_area :
  ∀ s : Set (ℝ × ℝ), is_inscribed_square s → ∃ a : ℝ, square_area a = (3 - Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1758_175820


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1758_175859

noncomputable def f (x : ℝ) := Real.sin x - x

theorem solution_set_of_inequality (x : ℝ) :
  f (x + 2) + f (1 - 2*x) < 0 ↔ x < 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1758_175859


namespace NUMINAMATH_CALUDE_M_intersect_P_eq_y_geq_1_l1758_175821

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def P : Set ℝ := {y | ∃ x : ℝ, y = Real.log x}

-- State the theorem
theorem M_intersect_P_eq_y_geq_1 : M ∩ P = {y : ℝ | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_P_eq_y_geq_1_l1758_175821


namespace NUMINAMATH_CALUDE_mrs_flannery_muffins_count_l1758_175842

/-- The number of muffins baked by Mrs. Brier's class -/
def mrs_brier_muffins : ℕ := 18

/-- The number of muffins baked by Mrs. MacAdams's class -/
def mrs_macadams_muffins : ℕ := 20

/-- The total number of muffins baked by all first grade classes -/
def total_muffins : ℕ := 55

/-- The number of muffins baked by Mrs. Flannery's class -/
def mrs_flannery_muffins : ℕ := total_muffins - (mrs_brier_muffins + mrs_macadams_muffins)

theorem mrs_flannery_muffins_count : mrs_flannery_muffins = 17 := by
  sorry

end NUMINAMATH_CALUDE_mrs_flannery_muffins_count_l1758_175842


namespace NUMINAMATH_CALUDE_alice_age_problem_l1758_175827

theorem alice_age_problem :
  ∃! x : ℕ+, 
    (∃ n : ℕ+, (x : ℤ) - 4 = n^2) ∧ 
    (∃ m : ℕ+, (x : ℤ) + 2 = m^3) ∧ 
    x = 58 := by
  sorry

end NUMINAMATH_CALUDE_alice_age_problem_l1758_175827


namespace NUMINAMATH_CALUDE_line_slope_l1758_175819

theorem line_slope (t : ℝ) : 
  let x := 3 - (Real.sqrt 3 / 2) * t
  let y := 1 + (1 / 2) * t
  (y - 1) / (x - 3) = -Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l1758_175819


namespace NUMINAMATH_CALUDE_split_tree_sum_lower_bound_l1758_175853

/-- Represents a tree where each node splits into two children that sum to the parent -/
inductive SplitTree : Nat → Type
  | leaf : SplitTree 1
  | node : (n : Nat) → (left right : Nat) → left + right = n → 
           SplitTree left → SplitTree right → SplitTree n

/-- The sum of all numbers in a SplitTree -/
def treeSum : {n : Nat} → SplitTree n → Nat
  | _, SplitTree.leaf => 1
  | n, SplitTree.node _ left right _ leftTree rightTree => 
      n + treeSum leftTree + treeSum rightTree

/-- Theorem: The sum of all numbers in a SplitTree starting with 2^n is at least n * 2^n -/
theorem split_tree_sum_lower_bound (n : Nat) (tree : SplitTree (2^n)) :
  treeSum tree ≥ n * 2^n := by
  sorry

end NUMINAMATH_CALUDE_split_tree_sum_lower_bound_l1758_175853


namespace NUMINAMATH_CALUDE_book_ratio_is_three_l1758_175893

/-- The number of books read last week -/
def books_last_week : ℕ := 5

/-- The number of pages in each book -/
def pages_per_book : ℕ := 300

/-- The total number of pages read this week -/
def pages_this_week : ℕ := 4500

/-- The ratio of books read this week to books read last week -/
def book_ratio : ℚ := (pages_this_week / pages_per_book) / books_last_week

theorem book_ratio_is_three : book_ratio = 3 := by
  sorry

end NUMINAMATH_CALUDE_book_ratio_is_three_l1758_175893


namespace NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_111_l1758_175812

theorem smallest_six_digit_divisible_by_111 : ∃ n : ℕ, 
  (n ≥ 100000 ∧ n < 1000000) ∧ 
  n % 111 = 0 ∧
  ∀ m : ℕ, (m ≥ 100000 ∧ m < 1000000) ∧ m % 111 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_111_l1758_175812


namespace NUMINAMATH_CALUDE_closest_fraction_l1758_175837

def medals_won : ℚ := 28 / 150

def options : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8]

theorem closest_fraction : 
  ∃ (closest : ℚ), closest ∈ options ∧ 
  ∀ (x : ℚ), x ∈ options → |medals_won - closest| ≤ |medals_won - x| ∧
  closest = 1/5 := by sorry

end NUMINAMATH_CALUDE_closest_fraction_l1758_175837


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l1758_175869

theorem minimum_value_theorem (x : ℝ) (h : x > 4) :
  (x + 11) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 15 ∧
  (∃ x₀ > 4, (x₀ + 11) / Real.sqrt (x₀ - 4) = 2 * Real.sqrt 15 ∧ x₀ = 19) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l1758_175869


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1758_175808

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (h_sum : a * b + b * c + a * c = 1) :
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ≥ (9 + 3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1758_175808


namespace NUMINAMATH_CALUDE_mothers_age_l1758_175809

/-- Proves that the mother's age is 42 given the conditions of the problem -/
theorem mothers_age (daughter_age : ℕ) (future_years : ℕ) (mother_age : ℕ) : 
  daughter_age = 8 →
  future_years = 9 →
  mother_age + future_years = 3 * (daughter_age + future_years) →
  mother_age = 42 := by
  sorry

end NUMINAMATH_CALUDE_mothers_age_l1758_175809


namespace NUMINAMATH_CALUDE_not_necessarily_equal_distances_l1758_175851

-- Define the points A_i and B_i in 3D space
variable (A B : ℕ → ℝ × ℝ × ℝ)

-- Define the radius of a circumscribed circle for a triangle
def circumradius (p q r : ℝ × ℝ × ℝ) : ℝ := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ × ℝ) : ℝ := sorry

-- A_1 is the circumcenter of a triangle
axiom A1_is_circumcenter : ∃ (p q r : ℝ × ℝ × ℝ), circumradius (A 1) p q = distance (A 1) p

-- The radii of circumscribed circles of triangles A_iA_jA_k and B_iB_jB_k are equal for any i, j, k
axiom equal_circumradii : ∀ (i j k : ℕ), circumradius (A i) (A j) (A k) = circumradius (B i) (B j) (B k)

-- The theorem to be proved
theorem not_necessarily_equal_distances :
  ¬(∀ (i j : ℕ), distance (A i) (A j) = distance (B i) (B j)) :=
sorry

end NUMINAMATH_CALUDE_not_necessarily_equal_distances_l1758_175851


namespace NUMINAMATH_CALUDE_fraction_simplification_l1758_175854

theorem fraction_simplification :
  (3/6 + 4/5) / (5/12 + 1/4) = 39/20 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1758_175854


namespace NUMINAMATH_CALUDE_smallest_label_on_1993_l1758_175839

theorem smallest_label_on_1993 (n : ℕ) (h : n > 0) :
  (n * (n + 1) / 2) % 2000 = 1021 →
  ∀ m, 0 < m ∧ m < n → (m * (m + 1) / 2) % 2000 ≠ 1021 →
  n = 118 := by
sorry

end NUMINAMATH_CALUDE_smallest_label_on_1993_l1758_175839


namespace NUMINAMATH_CALUDE_tablet_count_l1758_175801

theorem tablet_count : 
  ∀ (n : ℕ) (x y : ℕ),
  -- Lenovo (x), Samsung (x+6), and Huawei (y) make up less than a third of the total
  (2*x + y + 6 < n/3) →
  -- Apple iPads are three times as many as Huawei tablets
  (n - 2*x - y - 6 = 3*y) →
  -- If Lenovo tablets were tripled, there would be 59 Apple iPads
  (n - 3*x - (x+6) - y = 59) →
  (n = 94) := by
sorry

end NUMINAMATH_CALUDE_tablet_count_l1758_175801


namespace NUMINAMATH_CALUDE_max_guaranteed_rectangle_area_l1758_175898

/-- Represents a chessboard with some squares removed -/
structure Chessboard :=
  (size : Nat)
  (removed : Finset (Nat × Nat))

/-- Represents a rectangle on the chessboard -/
structure Rectangle :=
  (top_left : Nat × Nat)
  (width : Nat)
  (height : Nat)

/-- Check if a rectangle fits on the chessboard without overlapping removed squares -/
def Rectangle.fits (board : Chessboard) (rect : Rectangle) : Prop :=
  rect.top_left.1 + rect.width ≤ board.size ∧
  rect.top_left.2 + rect.height ≤ board.size ∧
  ∀ x y, rect.top_left.1 ≤ x ∧ x < rect.top_left.1 + rect.width ∧
         rect.top_left.2 ≤ y ∧ y < rect.top_left.2 + rect.height →
         (x, y) ∉ board.removed

/-- The main theorem -/
theorem max_guaranteed_rectangle_area (board : Chessboard) 
  (h1 : board.size = 8) 
  (h2 : board.removed.card = 8) : 
  (∀ n > 8, ∃ rect : Rectangle, rect.width * rect.height = n → ¬rect.fits board) ∧ 
  (∃ rect : Rectangle, rect.width * rect.height = 8 ∧ rect.fits board) :=
sorry

end NUMINAMATH_CALUDE_max_guaranteed_rectangle_area_l1758_175898


namespace NUMINAMATH_CALUDE_slope_condition_l1758_175875

/-- The slope of a line with y-intercept (0, 8) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def slope_intersecting_line_ellipse (m : ℝ) : Prop :=
  ∃ x y : ℝ, 
    y = m * x + 8 ∧ 
    4 * x^2 + 25 * y^2 = 100

/-- Theorem stating the condition for the slope of the intersecting line -/
theorem slope_condition : 
  ∀ m : ℝ, slope_intersecting_line_ellipse m ↔ m^2 ≥ 3/77 :=
by sorry

end NUMINAMATH_CALUDE_slope_condition_l1758_175875


namespace NUMINAMATH_CALUDE_floor_area_K_l1758_175885

/-- The number of circles in the ring -/
def n : ℕ := 7

/-- The radius of the larger circle C -/
def R : ℝ := 35

/-- The radius of each of the n congruent circles -/
noncomputable def r : ℝ := R * (Real.sqrt (2 - 2 * Real.cos (2 * Real.pi / n))) / 2

/-- The area K of the region inside circle C and outside all n circles -/
noncomputable def K : ℝ := Real.pi * (R^2 - n * r^2)

theorem floor_area_K : ⌊K⌋ = 1476 := by sorry

end NUMINAMATH_CALUDE_floor_area_K_l1758_175885


namespace NUMINAMATH_CALUDE_wholesale_cost_calculation_l1758_175844

/-- The wholesale cost of a sleeping bag -/
def wholesale_cost : ℝ := 24.56

/-- The selling price of a sleeping bag -/
def selling_price : ℝ := 28

/-- The gross profit percentage -/
def profit_percentage : ℝ := 0.14

theorem wholesale_cost_calculation :
  selling_price = wholesale_cost * (1 + profit_percentage) := by
  sorry

end NUMINAMATH_CALUDE_wholesale_cost_calculation_l1758_175844


namespace NUMINAMATH_CALUDE_divisor_of_a_l1758_175835

theorem divisor_of_a (a b c d : ℕ+) 
  (h1 : Nat.gcd a b = 24)
  (h2 : Nat.gcd b c = 36)
  (h3 : Nat.gcd c d = 54)
  (h4 : 70 < Nat.gcd d a ∧ Nat.gcd d a < 100) :
  13 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_a_l1758_175835


namespace NUMINAMATH_CALUDE_g_sum_symmetric_l1758_175861

-- Define the function g
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^8 + e * x^6 - f * x^4 + 5

-- Theorem statement
theorem g_sum_symmetric (d e f : ℝ) :
  (∃ x, g d e f x = 7) → g d e f 2 + g d e f (-2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_symmetric_l1758_175861


namespace NUMINAMATH_CALUDE_mosaic_configurations_l1758_175803

/-- Represents a tile in the mosaic --/
inductive Tile
| small : Tile  -- 1×1 tile
| large : Tile  -- 1×2 tile

/-- Represents a digit in the number 2021 --/
inductive Digit
| two : Digit
| zero : Digit
| one : Digit

/-- The number of cells used by each digit --/
def digit_cells (d : Digit) : Nat :=
  match d with
  | Digit.two => 13
  | Digit.zero => 18
  | Digit.one => 8

/-- The total number of tiles available --/
def available_tiles : Nat × Nat := (4, 24)  -- (small tiles, large tiles)

/-- A configuration of tiles for a single digit --/
def DigitConfiguration := List Tile

/-- A configuration of tiles for the entire number 2021 --/
def NumberConfiguration := List DigitConfiguration

/-- Checks if a digit configuration is valid for a given digit --/
def is_valid_digit_config (d : Digit) (config : DigitConfiguration) : Prop := sorry

/-- Checks if a number configuration is valid --/
def is_valid_number_config (config : NumberConfiguration) : Prop := sorry

/-- Counts the number of valid configurations --/
def count_valid_configs : Nat := sorry

/-- The main theorem --/
theorem mosaic_configurations :
  count_valid_configs = 6517 := sorry

end NUMINAMATH_CALUDE_mosaic_configurations_l1758_175803


namespace NUMINAMATH_CALUDE_max_stamps_problem_l1758_175836

/-- The maximum number of stamps that can be bought -/
def max_stamps (initial_money : ℕ) (bus_ticket_cost : ℕ) (stamp_price : ℕ) : ℕ :=
  ((initial_money * 100 - bus_ticket_cost) / stamp_price : ℕ)

/-- Theorem: Given $50 initial money, 180 cents bus ticket cost, and 45 cents stamp price,
    the maximum number of stamps that can be bought is 107 -/
theorem max_stamps_problem : max_stamps 50 180 45 = 107 := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_problem_l1758_175836
