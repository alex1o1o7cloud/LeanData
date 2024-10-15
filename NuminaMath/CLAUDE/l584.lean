import Mathlib

namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l584_58427

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  1 / x + 2 / y ≥ 8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ 1 / x + 2 / y = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l584_58427


namespace NUMINAMATH_CALUDE_shipping_scenario_l584_58416

/-- The number of broken artworks in a shipping scenario -/
def broken_artworks : ℕ :=
  -- We'll define this later in the theorem
  4

/-- The total number of artworks -/
def total_artworks : ℕ := 2000

/-- The shipping cost per artwork in yuan -/
def shipping_cost : ℚ := 0.2

/-- The compensation cost for a broken artwork in yuan -/
def compensation_cost : ℚ := 2.3

/-- The total profit in yuan -/
def total_profit : ℚ := 390

theorem shipping_scenario :
  shipping_cost * (total_artworks - broken_artworks : ℚ) - compensation_cost * broken_artworks = total_profit :=
by sorry

end NUMINAMATH_CALUDE_shipping_scenario_l584_58416


namespace NUMINAMATH_CALUDE_coprime_n_minus_two_and_n_squared_minus_n_minus_one_part2_solutions_part3_solution_l584_58477

-- Part 1
theorem coprime_n_minus_two_and_n_squared_minus_n_minus_one (n : ℕ) :
  Nat.gcd (n - 2) (n^2 - n - 1) = 1 :=
sorry

-- Part 2
def is_valid_solution_part2 (n m : ℕ) : Prop :=
  n^3 - 3*n^2 + n + 2 = 5^m

theorem part2_solutions :
  ∀ n m : ℕ, is_valid_solution_part2 n m ↔ (n = 3 ∧ m = 1) ∨ (n = 1 ∧ m = 0) :=
sorry

-- Part 3
def is_valid_solution_part3 (n m : ℕ) : Prop :=
  2*n^3 - n^2 + 2*n + 1 = 3^m

theorem part3_solution :
  ∀ n m : ℕ, is_valid_solution_part3 n m ↔ (n = 0 ∧ m = 0) :=
sorry

end NUMINAMATH_CALUDE_coprime_n_minus_two_and_n_squared_minus_n_minus_one_part2_solutions_part3_solution_l584_58477


namespace NUMINAMATH_CALUDE_computer_device_properties_l584_58412

theorem computer_device_properties :
  ∃ f : ℕ → ℕ → ℕ,
    (f 1 1 = 1) ∧
    (∀ m n : ℕ, f m (n + 1) = f m n + 2) ∧
    (∀ m : ℕ, f (m + 1) 1 = 2 * f m 1) ∧
    (∀ n : ℕ, f 1 n = 2 * n - 1) ∧
    (∀ m : ℕ, f m 1 = 2^(m - 1)) :=
by sorry

end NUMINAMATH_CALUDE_computer_device_properties_l584_58412


namespace NUMINAMATH_CALUDE_bisector_triangle_area_l584_58426

/-- Given a tetrahedron ABCD with face areas P (ABC) and Q (ADC), and dihedral angle α between these faces,
    the area S of the triangle formed by the plane bisecting α is (2PQ cos(α/2)) / (P + Q) -/
theorem bisector_triangle_area (P Q α : ℝ) (hP : P > 0) (hQ : Q > 0) (hα : 0 < α ∧ α < π) :
  ∃ S : ℝ, S = (2 * P * Q * Real.cos (α / 2)) / (P + Q) ∧ S > 0 :=
sorry

end NUMINAMATH_CALUDE_bisector_triangle_area_l584_58426


namespace NUMINAMATH_CALUDE_smallest_base_for_150_l584_58437

theorem smallest_base_for_150 :
  ∃ b : ℕ, b = 6 ∧ b^2 ≤ 150 ∧ 150 < b^3 ∧ ∀ n : ℕ, n < b → (n^2 > 150 ∨ 150 ≥ n^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_150_l584_58437


namespace NUMINAMATH_CALUDE_f_composition_value_l584_58440

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then (1/2)^x
  else if 0 < x ∧ x < 1 then Real.log x / Real.log 4
  else 0  -- This case is added to make the function total

-- State the theorem
theorem f_composition_value : f (f 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l584_58440


namespace NUMINAMATH_CALUDE_rectangle_cutting_l584_58484

/-- Represents a rectangle on a cartesian plane with sides parallel to coordinate axes -/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h1 : x_min < x_max
  h2 : y_min < y_max

/-- Predicate to check if a vertical line intersects a rectangle -/
def vertical_intersects (x : ℝ) (r : Rectangle) : Prop :=
  r.x_min < x ∧ x < r.x_max

/-- Predicate to check if a horizontal line intersects a rectangle -/
def horizontal_intersects (y : ℝ) (r : Rectangle) : Prop :=
  r.y_min < y ∧ y < r.y_max

/-- Any two rectangles can be cut by a vertical or a horizontal line -/
axiom rectangle_separation (r1 r2 : Rectangle) :
  (∃ x : ℝ, vertical_intersects x r1 ∧ vertical_intersects x r2) ∨
  (∃ y : ℝ, horizontal_intersects y r1 ∧ horizontal_intersects y r2)

/-- The main theorem -/
theorem rectangle_cutting (rectangles : Set Rectangle) :
  ∃ (x y : ℝ), ∀ r ∈ rectangles, vertical_intersects x r ∨ horizontal_intersects y r :=
sorry

end NUMINAMATH_CALUDE_rectangle_cutting_l584_58484


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l584_58432

/-- Given the following conditions for a company's finances over two years:
    1. In the previous year, profits were a percentage of revenues
    2. In 2009, revenues fell by 20%
    3. In 2009, profits were 20% of revenues
    4. Profits in 2009 were 160% of profits in the previous year
    
    This theorem proves that the percentage of profits to revenues in the previous year was 10%. -/
theorem profit_percentage_previous_year 
  (R : ℝ) -- Revenues in the previous year
  (P : ℝ) -- Profits in the previous year
  (h1 : P > 0) -- Ensure profits are positive
  (h2 : R > 0) -- Ensure revenues are positive
  (h3 : 0.8 * R * 0.2 = 1.6 * P) -- Condition relating 2009 profits to previous year
  : P / R = 0.1 := by
  sorry

#check profit_percentage_previous_year

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l584_58432


namespace NUMINAMATH_CALUDE_elevator_scenarios_count_l584_58428

/-- Represents the number of floors in the building -/
def num_floors : ℕ := 7

/-- Represents the number of people entering the elevator -/
def num_people : ℕ := 3

/-- Calculates the number of scenarios where exactly one person goes to the top floor
    and person A does not get off on the second floor -/
def elevator_scenarios : ℕ :=
  let a_to_top := (num_floors - 2)^(num_people - 1)
  let others_to_top := (num_people - 1) * (num_floors - 3) * (num_floors - 2)
  a_to_top + others_to_top

/-- The main theorem stating that the number of scenarios is 65 -/
theorem elevator_scenarios_count :
  elevator_scenarios = 65 := by sorry

end NUMINAMATH_CALUDE_elevator_scenarios_count_l584_58428


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l584_58487

def inequality_system (x t : ℝ) : Prop :=
  6 - (2 * x + 5) > -15 ∧ (x + 3) / 2 - t < x

theorem inequality_system_solutions :
  (∀ x : ℤ, inequality_system x 2 → x ≥ 0) ∧
  (∃ x : ℤ, inequality_system x 2 ∧ x = 0) ∧
  (∀ x : ℝ, inequality_system x 4 ↔ -5 < x ∧ x < 8) ∧
  (∃! t : ℝ, ∀ x : ℝ, inequality_system x t ↔ -5 < x ∧ x < 8) ∧
  (∀ t : ℝ, (∃! (a b c : ℤ), 
    inequality_system (a : ℝ) t ∧ 
    inequality_system (b : ℝ) t ∧ 
    inequality_system (c : ℝ) t ∧ 
    a < b ∧ b < c ∧
    (∀ x : ℤ, inequality_system (x : ℝ) t → x = a ∨ x = b ∨ x = c)) 
    ↔ -1 < t ∧ t ≤ -1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l584_58487


namespace NUMINAMATH_CALUDE_complex_magnitude_range_l584_58457

theorem complex_magnitude_range (θ : ℝ) :
  let z : ℂ := Complex.mk (Real.sqrt 3 * Real.sin θ) (Real.cos θ)
  Complex.abs z < Real.sqrt 2 ↔ ∃ k : ℤ, -π/4 + k*π < θ ∧ θ < π/4 + k*π :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_range_l584_58457


namespace NUMINAMATH_CALUDE_problem_statement_l584_58442

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 6)
  (h2 : x * y = 128)
  (h3 : x = 2 * y^2) :
  (x + y) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l584_58442


namespace NUMINAMATH_CALUDE_mango_count_proof_l584_58458

/-- Given a ratio of mangoes to apples and the number of apples, 
    calculate the number of mangoes -/
def calculate_mangoes (mango_ratio : ℕ) (apple_ratio : ℕ) (apple_count : ℕ) : ℕ :=
  (mango_ratio * apple_count) / apple_ratio

/-- Theorem: Given the ratio of mangoes to apples is 10:3 and there are 36 apples,
    prove that the number of mangoes is 120 -/
theorem mango_count_proof :
  calculate_mangoes 10 3 36 = 120 := by
  sorry

end NUMINAMATH_CALUDE_mango_count_proof_l584_58458


namespace NUMINAMATH_CALUDE_social_relationships_theorem_l584_58482

/-- Represents the relationship between two people -/
inductive Relationship
  | Knows
  | DoesNotKnow

/-- A function representing the relationship between people -/
def relationship (people : ℕ) : (Fin people → Fin people → Relationship) :=
  sorry

theorem social_relationships_theorem (n : ℕ) :
  ∃ (A B : Fin (2*n+2)), ∃ (S : Finset (Fin (2*n+2))),
    S.card ≥ n ∧
    (∀ C ∈ S, C ≠ A ∧ C ≠ B) ∧
    (∀ C ∈ S, (relationship (2*n+2) A C = relationship (2*n+2) B C)) :=
  sorry

end NUMINAMATH_CALUDE_social_relationships_theorem_l584_58482


namespace NUMINAMATH_CALUDE_middle_five_sum_l584_58495

theorem middle_five_sum (total : ℕ) (avg_all : ℚ) (avg_first_ten : ℚ) (avg_last_ten : ℚ) (avg_middle_seven : ℚ) :
  total = 21 →
  avg_all = 44 →
  avg_first_ten = 48 →
  avg_last_ten = 41 →
  avg_middle_seven = 45 →
  (total * avg_all : ℚ) = (10 * avg_first_ten + 10 * avg_last_ten + (total - 20) * avg_middle_seven : ℚ) →
  (5 : ℕ) * ((7 : ℕ) * avg_middle_seven - avg_first_ten - avg_last_ten) = 226 :=
by sorry

end NUMINAMATH_CALUDE_middle_five_sum_l584_58495


namespace NUMINAMATH_CALUDE_tenths_minus_hundredths_l584_58466

theorem tenths_minus_hundredths : (0.5 : ℝ) - (0.05 : ℝ) = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_tenths_minus_hundredths_l584_58466


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruence_l584_58404

theorem smallest_four_digit_congruence :
  ∃ (n : ℕ), 
    (n ≥ 1000 ∧ n < 10000) ∧ 
    (75 * n) % 345 = 225 ∧
    (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ (75 * m) % 345 = 225 → m ≥ n) ∧
    n = 1015 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruence_l584_58404


namespace NUMINAMATH_CALUDE_unique_element_in_S_l584_58465

-- Define the set
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.log (p.1^3 + (1/3) * p.2^3 + 1/9) = Real.log p.1 + Real.log p.2}

-- Theorem statement
theorem unique_element_in_S : ∃! p : ℝ × ℝ, p ∈ S := by sorry

end NUMINAMATH_CALUDE_unique_element_in_S_l584_58465


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l584_58469

theorem sqrt_product_equality : Real.sqrt 72 * Real.sqrt 27 * Real.sqrt 8 = 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l584_58469


namespace NUMINAMATH_CALUDE_exactly_four_sets_l584_58478

-- Define the set S1 as {-1, 0, 1}
def S1 : Set Int := {-1, 0, 1}

-- Define the set S2 as {-2, 0, 2}
def S2 : Set Int := {-2, 0, 2}

-- Define the set R as {-2, 0, 1, 2}
def R : Set Int := {-2, 0, 1, 2}

-- Define the conditions for set A
def satisfiesConditions (A : Set Int) : Prop :=
  (A ∩ S1 = {0, 1}) ∧ (A ∪ S2 = R)

-- Theorem stating that there are exactly 4 sets satisfying the conditions
theorem exactly_four_sets :
  ∃! (s : Finset (Set Int)), (∀ A ∈ s, satisfiesConditions A) ∧ s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_exactly_four_sets_l584_58478


namespace NUMINAMATH_CALUDE_distance_to_hole_is_250_l584_58453

/-- The distance from the starting tee to the hole in a golf game --/
def distance_to_hole (first_hit second_hit beyond_hole : ℕ) : ℕ :=
  first_hit + second_hit - beyond_hole

/-- Theorem stating the distance to the hole given the conditions in the problem --/
theorem distance_to_hole_is_250 :
  let first_hit := 180
  let second_hit := first_hit / 2
  let beyond_hole := 20
  distance_to_hole first_hit second_hit beyond_hole = 250 := by
  sorry

#eval distance_to_hole 180 90 20

end NUMINAMATH_CALUDE_distance_to_hole_is_250_l584_58453


namespace NUMINAMATH_CALUDE_external_tangent_same_color_l584_58447

/-- A point on a line --/
structure Point where
  x : ℝ

/-- A circle with diameter endpoints --/
structure Circle where
  p1 : Point
  p2 : Point

/-- A color represented as a natural number --/
def Color := ℕ

/-- The set of all circles formed by pairs of points --/
def allCircles (points : List Point) : List Circle :=
  sorry

/-- Checks if two circles are externally tangent --/
def areExternallyTangent (c1 c2 : Circle) : Prop :=
  sorry

/-- Assigns a color to each circle --/
def colorAssignment (circles : List Circle) (n : ℕ) : Circle → Color :=
  sorry

/-- Main theorem --/
theorem external_tangent_same_color 
  (k n : ℕ) (points : List Point) (h : k > 2^n) (h2 : points.length = k) :
  ∃ (c1 c2 : Circle), c1 ∈ allCircles points ∧ c2 ∈ allCircles points ∧ 
    c1 ≠ c2 ∧
    areExternallyTangent c1 c2 ∧
    colorAssignment (allCircles points) n c1 = colorAssignment (allCircles points) n c2 :=
  sorry

end NUMINAMATH_CALUDE_external_tangent_same_color_l584_58447


namespace NUMINAMATH_CALUDE_solution_for_E_l584_58467

/-- The function E as defined in the problem -/
def E (a b c : ℚ) : ℚ := a * b^2 + c

/-- Theorem stating that -1/10 is the solution to E(a,4,5) = E(a,6,7) -/
theorem solution_for_E : 
  ∃ a : ℚ, E a 4 5 = E a 6 7 ∧ a = -1/10 := by
  sorry

end NUMINAMATH_CALUDE_solution_for_E_l584_58467


namespace NUMINAMATH_CALUDE_min_frames_for_18x15_grid_l584_58475

/-- Represents a grid with given dimensions -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square frame with side length 1 -/
structure Frame :=
  (side_length : ℕ := 1)

/-- Calculates the minimum number of frames needed to cover the grid -/
def min_frames_needed (g : Grid) : ℕ :=
  g.rows * g.cols - ((g.rows - 2) / 2 * (g.cols - 2))

/-- The theorem stating the minimum number of frames needed for an 18x15 grid -/
theorem min_frames_for_18x15_grid :
  let g : Grid := ⟨18, 15⟩
  min_frames_needed g = 166 := by
  sorry

#eval min_frames_needed ⟨18, 15⟩

end NUMINAMATH_CALUDE_min_frames_for_18x15_grid_l584_58475


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l584_58430

theorem complex_number_magnitude (a : ℝ) (i : ℂ) (z : ℂ) : 
  a < 0 → 
  i * i = -1 → 
  z = a * i / (1 - 2 * i) → 
  Complex.abs z = Real.sqrt 5 → 
  a = -5 := by sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l584_58430


namespace NUMINAMATH_CALUDE_existence_condition_equiv_range_l584_58406

theorem existence_condition_equiv_range (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 3 ∧ |x₀^2 - a*x₀ + 4| ≤ 3*x₀) ↔ 
  (2 ≤ a ∧ a ≤ 7 + 1/3) := by
sorry

end NUMINAMATH_CALUDE_existence_condition_equiv_range_l584_58406


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l584_58461

/-- Given a quadratic function f(x) = x^2 + x + m where m is positive,
    if f(t) < 0 for some real t, then f has a root in the interval (t, t+1) -/
theorem quadratic_root_existence (m : ℝ) (t : ℝ) (h_m : m > 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 + x + m
  f t < 0 → ∃ x : ℝ, t < x ∧ x < t + 1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l584_58461


namespace NUMINAMATH_CALUDE_simplify_expression_l584_58421

theorem simplify_expression (a b : ℝ) :
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l584_58421


namespace NUMINAMATH_CALUDE_max_value_theorem_l584_58405

theorem max_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (x : ℝ), x = Real.rpow (a * b * c) (1/3) ∧
  (∀ (y : ℝ), (∃ (p q r : ℝ), 0 < p ∧ 0 < q ∧ 0 < r ∧ p + q + r = 1 ∧
    y ≤ a * p / q ∧ y ≤ b * q / r ∧ y ≤ c * r / p) → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l584_58405


namespace NUMINAMATH_CALUDE_scientific_notation_of_57277000_l584_58476

theorem scientific_notation_of_57277000 :
  (57277000 : ℝ) = 5.7277 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_57277000_l584_58476


namespace NUMINAMATH_CALUDE_distance_P_to_x_axis_l584_58414

/-- The distance from a point to the x-axis is the absolute value of its y-coordinate -/
def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

/-- Point P with coordinates (3, -5) -/
def P : ℝ × ℝ := (3, -5)

/-- Theorem stating that the distance from P to the x-axis is 5 -/
theorem distance_P_to_x_axis :
  distance_to_x_axis P = 5 := by sorry

end NUMINAMATH_CALUDE_distance_P_to_x_axis_l584_58414


namespace NUMINAMATH_CALUDE_speaking_orders_count_l584_58435

/-- The number of contestants -/
def n : ℕ := 6

/-- The number of positions where contestant A can speak -/
def a_positions : ℕ := n - 2

/-- The number of permutations for the remaining contestants -/
def remaining_permutations : ℕ := Nat.factorial (n - 1)

/-- The total number of different speaking orders -/
def total_orders : ℕ := a_positions * remaining_permutations

theorem speaking_orders_count : total_orders = 480 := by
  sorry

end NUMINAMATH_CALUDE_speaking_orders_count_l584_58435


namespace NUMINAMATH_CALUDE_grid_solution_l584_58418

/-- Represents a 3x3 grid of integers -/
structure Grid :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : Int)

/-- Checks if the middle number in each row is the sum of the numbers at its ends -/
def rowSumsValid (g : Grid) : Prop :=
  g.a12 = g.a11 + g.a13 ∧ g.a22 = g.a21 + g.a23 ∧ g.a32 = g.a31 + g.a33

/-- Checks if the sums of the numbers on both diagonals are equal -/
def diagonalSumsEqual (g : Grid) : Prop :=
  g.a11 + g.a22 + g.a33 = g.a13 + g.a22 + g.a31

/-- The theorem stating the solution to the grid problem -/
theorem grid_solution :
  ∀ (g : Grid),
    g.a11 = 4 ∧ g.a12 = 12 ∧ g.a13 = 8 ∧ g.a21 = 10 →
    rowSumsValid g →
    diagonalSumsEqual g →
    g.a22 = 3 ∧ g.a23 = 9 ∧ g.a31 = -3 ∧ g.a32 = -2 ∧ g.a33 = 1 :=
by sorry

end NUMINAMATH_CALUDE_grid_solution_l584_58418


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l584_58413

/-- An infinite geometric series with first term a and common ratio r has sum S if and only if |r| < 1 and S = a / (1 - r) -/
def is_infinite_geometric_series_sum (a : ℝ) (r : ℝ) (S : ℝ) : Prop :=
  |r| < 1 ∧ S = a / (1 - r)

/-- The positive common ratio of an infinite geometric series with first term 500 and sum 4000 is 7/8 -/
theorem infinite_geometric_series_ratio : 
  ∃ (r : ℝ), r > 0 ∧ is_infinite_geometric_series_sum 500 r 4000 ∧ r = 7/8 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l584_58413


namespace NUMINAMATH_CALUDE_gcf_of_36_60_90_l584_58443

theorem gcf_of_36_60_90 : Nat.gcd 36 (Nat.gcd 60 90) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_36_60_90_l584_58443


namespace NUMINAMATH_CALUDE_program_output_is_twenty_l584_58445

/-- The result of evaluating the arithmetic expression (3+2)*4 -/
def program_result : ℕ := (3 + 2) * 4

/-- Theorem stating that the result of the program is 20 -/
theorem program_output_is_twenty : program_result = 20 := by
  sorry

end NUMINAMATH_CALUDE_program_output_is_twenty_l584_58445


namespace NUMINAMATH_CALUDE_combinatorial_identity_l584_58429

theorem combinatorial_identity : Nat.choose 98 97 + 2 * Nat.choose 98 96 + Nat.choose 98 95 = Nat.choose 100 97 := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_identity_l584_58429


namespace NUMINAMATH_CALUDE_abc_def_ratio_l584_58451

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) :
  a * b * c / (d * e * f) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_abc_def_ratio_l584_58451


namespace NUMINAMATH_CALUDE_final_nickel_count_is_45_l584_58403

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  half_dollars : ℕ

/-- Represents a transaction of coins -/
structure CoinTransaction where
  nickels : ℤ
  dimes : ℤ
  quarters : ℤ
  half_dollars : ℤ

def initial_coins : CoinCount := {
  pennies := 45,
  nickels := 29,
  dimes := 16,
  quarters := 8,
  half_dollars := 4
}

def dad_gives : CoinTransaction := {
  nickels := 24,
  dimes := 15,
  quarters := 12,
  half_dollars := 6
}

def dad_takes : CoinTransaction := {
  nickels := -13,
  dimes := -9,
  quarters := -5,
  half_dollars := 0
}

def additional_percentage : ℚ := 20 / 100

/-- Applies a transaction to the coin count -/
def apply_transaction (coins : CoinCount) (transaction : CoinTransaction) : CoinCount :=
  { coins with
    nickels := (coins.nickels : ℤ) + transaction.nickels |>.toNat,
    dimes := (coins.dimes : ℤ) + transaction.dimes |>.toNat,
    quarters := (coins.quarters : ℤ) + transaction.quarters |>.toNat,
    half_dollars := (coins.half_dollars : ℤ) + transaction.half_dollars |>.toNat
  }

/-- Calculates the final number of nickels Sam has -/
def final_nickel_count : ℕ :=
  let after_first_transaction := apply_transaction initial_coins dad_gives
  let after_second_transaction := apply_transaction after_first_transaction dad_takes
  let additional_nickels := (dad_gives.nickels : ℚ) * additional_percentage |>.ceil.toNat
  after_second_transaction.nickels + additional_nickels

theorem final_nickel_count_is_45 : final_nickel_count = 45 := by
  sorry

end NUMINAMATH_CALUDE_final_nickel_count_is_45_l584_58403


namespace NUMINAMATH_CALUDE_concert_ticket_revenue_l584_58450

theorem concert_ticket_revenue 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (full_price_tickets : ℕ) 
  (student_price_tickets : ℕ) 
  (full_price : ℕ) 
  (h1 : total_tickets = 150)
  (h2 : total_revenue = 2450)
  (h3 : student_price_tickets = total_tickets - full_price_tickets)
  (h4 : total_revenue = full_price_tickets * full_price + student_price_tickets * (full_price / 2))
  : full_price_tickets * full_price = 1150 := by
  sorry

#check concert_ticket_revenue

end NUMINAMATH_CALUDE_concert_ticket_revenue_l584_58450


namespace NUMINAMATH_CALUDE_probability_of_two_mismatches_l584_58422

/-- Represents a set of pens and caps -/
structure PenSet :=
  (pens : Finset (Fin 3))
  (caps : Finset (Fin 3))

/-- Represents a pairing of pens and caps -/
def Pairing := Fin 3 → Fin 3

/-- The set of all possible pairings -/
def allPairings : Finset Pairing := sorry

/-- Predicate for a pairing that mismatches two pairs -/
def mismatchesTwoPairs (p : Pairing) : Prop := sorry

/-- The number of pairings that mismatch two pairs -/
def numMismatchedPairings : Nat := sorry

theorem probability_of_two_mismatches (ps : PenSet) :
  (numMismatchedPairings : ℚ) / (Finset.card allPairings : ℚ) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_of_two_mismatches_l584_58422


namespace NUMINAMATH_CALUDE_polynomial_factorization_l584_58474

theorem polynomial_factorization (a : ℝ) :
  (a^2 + 2*a)*(a^2 + 2*a + 2) + 1 = (a + 1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l584_58474


namespace NUMINAMATH_CALUDE_hamburgers_needed_proof_l584_58480

/-- Calculates the number of additional hamburgers needed to reach a target revenue -/
def additional_hamburgers_needed (target_revenue : ℕ) (price_per_hamburger : ℕ) (hamburgers_sold : ℕ) : ℕ :=
  ((target_revenue - (price_per_hamburger * hamburgers_sold)) + (price_per_hamburger - 1)) / price_per_hamburger

/-- Proves that 4 additional hamburgers are needed to reach $50 given the conditions -/
theorem hamburgers_needed_proof (target_revenue : ℕ) (price_per_hamburger : ℕ) (hamburgers_sold : ℕ)
  (h1 : target_revenue = 50)
  (h2 : price_per_hamburger = 5)
  (h3 : hamburgers_sold = 6) :
  additional_hamburgers_needed target_revenue price_per_hamburger hamburgers_sold = 4 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_needed_proof_l584_58480


namespace NUMINAMATH_CALUDE_students_opted_both_math_and_science_l584_58408

theorem students_opted_both_math_and_science 
  (total_students : ℕ) 
  (not_math : ℕ) 
  (not_science : ℕ) 
  (not_either : ℕ) 
  (h1 : total_students = 40)
  (h2 : not_math = 10)
  (h3 : not_science = 15)
  (h4 : not_either = 2) :
  total_students - (not_math + not_science - not_either) = 17 := by
  sorry

#check students_opted_both_math_and_science

end NUMINAMATH_CALUDE_students_opted_both_math_and_science_l584_58408


namespace NUMINAMATH_CALUDE_parabola_circle_tangent_radius_l584_58490

/-- The radius of a circle that is tangent to the parabola y = 1/4 * x^2 at a point where the tangent line to the parabola is also tangent to the circle. -/
theorem parabola_circle_tangent_radius : ∃ (r : ℝ) (P : ℝ × ℝ),
  r > 0 ∧
  (P.2 = (1/4) * P.1^2) ∧
  ((P.1 - 1)^2 + (P.2 - 2)^2 = r^2) ∧
  (∃ (m : ℝ), (∀ (x y : ℝ), y - P.2 = m * (x - P.1) → 
    y = (1/4) * x^2 ∨ (x - 1)^2 + (y - 2)^2 = r^2)) →
  r = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangent_radius_l584_58490


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l584_58472

-- Problem 1
theorem problem_one (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

-- Problem 2
theorem problem_two (a b c x y z : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^x = b^y) (h5 : b^y = c^z)
  (h6 : 1/x + 1/y + 1/z = 0) : a * b * c = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l584_58472


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l584_58438

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = x*y) :
  x + 2*y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = x₀*y₀ ∧ x₀ + 2*y₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l584_58438


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l584_58434

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -1)
  parallel a b → x = -3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l584_58434


namespace NUMINAMATH_CALUDE_min_marked_cells_for_unique_determination_l584_58491

/-- Represents a 9x9 board -/
def Board := Fin 9 → Fin 9 → Bool

/-- An L-shaped piece covering 3 cells -/
structure LPiece where
  x : Fin 9
  y : Fin 9
  orientation : Fin 4

/-- Checks if a given L-piece is uniquely determined by the marked cells -/
def isUniqueDetermination (board : Board) (piece : LPiece) : Bool :=
  sorry

/-- Checks if all possible L-piece placements are uniquely determined -/
def allPiecesUnique (board : Board) : Bool :=
  sorry

/-- Counts the number of marked cells on the board -/
def countMarkedCells (board : Board) : Nat :=
  sorry

/-- The main theorem: The minimum number of marked cells for unique determination is 63 -/
theorem min_marked_cells_for_unique_determination :
  ∃ (board : Board), allPiecesUnique board ∧ countMarkedCells board = 63 ∧
  ∀ (other_board : Board), allPiecesUnique other_board → countMarkedCells other_board ≥ 63 :=
sorry

end NUMINAMATH_CALUDE_min_marked_cells_for_unique_determination_l584_58491


namespace NUMINAMATH_CALUDE_negation_existence_lt_one_l584_58483

theorem negation_existence_lt_one :
  (¬ ∃ x : ℝ, x < 1) ↔ (∀ x : ℝ, x ≥ 1) := by sorry

end NUMINAMATH_CALUDE_negation_existence_lt_one_l584_58483


namespace NUMINAMATH_CALUDE_max_n_value_l584_58423

theorem max_n_value (a b c : ℝ) (n : ℕ) 
  (h1 : a > b) (h2 : b > c)
  (h3 : ∀ a b c, a > b → b > c → 1 / (a - b) + 1 / (b - c) ≥ n / (a - c)) :
  n ≤ 4 ∧ ∃ a b c, a > b ∧ b > c ∧ 1 / (a - b) + 1 / (b - c) = 4 / (a - c) :=
sorry

end NUMINAMATH_CALUDE_max_n_value_l584_58423


namespace NUMINAMATH_CALUDE_problem_solution_l584_58431

-- Define the solution set for |x-m| < |x|
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | 1 < x}

-- Define the inequality condition
def inequality_condition (a m : ℝ) (x : ℝ) : Prop :=
  (a - 5) / x < |1 + 1/x| - |1 - m/x| ∧ |1 + 1/x| - |1 - m/x| < (a + 2) / x

theorem problem_solution :
  (∀ x : ℝ, x ∈ solution_set m ↔ |x - m| < |x|) →
  m = 2 ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → inequality_condition a m x) ↔ 1 < a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l584_58431


namespace NUMINAMATH_CALUDE_paulas_aunt_money_l584_58493

/-- The amount of money Paula's aunt gave her -/
def aunt_money (shirt_price : ℕ) (num_shirts : ℕ) (pants_price : ℕ) (money_left : ℕ) : ℕ :=
  shirt_price * num_shirts + pants_price + money_left

/-- Theorem stating the total amount of money Paula's aunt gave her -/
theorem paulas_aunt_money :
  aunt_money 11 2 13 74 = 109 := by
  sorry

end NUMINAMATH_CALUDE_paulas_aunt_money_l584_58493


namespace NUMINAMATH_CALUDE_marias_trip_distance_l584_58486

theorem marias_trip_distance :
  ∀ (D : ℝ),
  (D / 2) / 4 + 180 = D / 2 →
  D = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_marias_trip_distance_l584_58486


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l584_58402

/-- Given a grocery store inventory, prove the number of diet soda bottles -/
theorem diet_soda_bottles (regular_soda : ℕ) (total_regular_and_diet : ℕ) 
  (h1 : regular_soda = 49)
  (h2 : total_regular_and_diet = 89) :
  total_regular_and_diet - regular_soda = 40 := by
  sorry

#check diet_soda_bottles

end NUMINAMATH_CALUDE_diet_soda_bottles_l584_58402


namespace NUMINAMATH_CALUDE_problem_solution_l584_58455

theorem problem_solution : 
  ∀ N : ℝ, (2 + 3 + 4) / 3 = (1990 + 1991 + 1992) / N → N = 1991 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l584_58455


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l584_58463

/-- Proves that the cost of gas per gallon is $4, given the conditions of Dan's car fuel efficiency and travel distance. -/
theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) : 
  miles_per_gallon = 32 → total_miles = 464 → total_cost = 58 → 
  (total_cost / (total_miles / miles_per_gallon)) = 4 := by
  sorry

/-- The actual cost of gas per gallon based on the given conditions. -/
def actual_gas_cost : ℝ := 4

#check gas_cost_per_gallon
#check actual_gas_cost

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l584_58463


namespace NUMINAMATH_CALUDE_square_greater_not_sufficient_nor_necessary_l584_58449

theorem square_greater_not_sufficient_nor_necessary :
  ∃ (a b : ℝ), a^2 > b^2 ∧ ¬(a > b) ∧
  ∃ (c d : ℝ), c > d ∧ ¬(c^2 > d^2) := by
  sorry

end NUMINAMATH_CALUDE_square_greater_not_sufficient_nor_necessary_l584_58449


namespace NUMINAMATH_CALUDE_initial_number_proof_l584_58409

theorem initial_number_proof : ∃ (N : ℕ), N > 0 ∧ (N - 10) % 21 = 0 ∧ ∀ (M : ℕ), M > 0 → (M - 10) % 21 = 0 → M ≥ N := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l584_58409


namespace NUMINAMATH_CALUDE_helga_work_days_l584_58454

/-- Represents Helga's work schedule and output --/
structure HelgaWork where
  articles_per_half_hour : ℕ := 5
  usual_hours_per_day : ℕ := 4
  extra_hours_thursday : ℕ := 2
  extra_hours_friday : ℕ := 3
  total_articles_week : ℕ := 250

/-- Calculates the number of days Helga usually works in a week --/
def usual_work_days (hw : HelgaWork) : ℕ :=
  let articles_per_hour := hw.articles_per_half_hour * 2
  let articles_per_day := articles_per_hour * hw.usual_hours_per_day
  let extra_articles := articles_per_hour * (hw.extra_hours_thursday + hw.extra_hours_friday)
  let usual_articles := hw.total_articles_week - extra_articles
  usual_articles / articles_per_day

theorem helga_work_days (hw : HelgaWork) : usual_work_days hw = 5 := by
  sorry

end NUMINAMATH_CALUDE_helga_work_days_l584_58454


namespace NUMINAMATH_CALUDE_wall_length_is_800_l584_58460

-- Define the dimensions of a single brick
def brick_length : ℝ := 100
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the wall dimensions
def wall_height : ℝ := 600 -- 6 m converted to cm
def wall_width : ℝ := 22.5

-- Define the number of bricks
def num_bricks : ℕ := 1600

-- Define the volume of a single brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Define the total volume of all bricks
def total_brick_volume : ℝ := brick_volume * num_bricks

-- Theorem stating the length of the wall
theorem wall_length_is_800 : 
  ∃ (wall_length : ℝ), wall_length * wall_height * wall_width = total_brick_volume ∧ wall_length = 800 := by
sorry

end NUMINAMATH_CALUDE_wall_length_is_800_l584_58460


namespace NUMINAMATH_CALUDE_circular_film_radius_l584_58444

/-- The radius of a circular film formed by a non-mixing liquid on water -/
theorem circular_film_radius 
  (volume : ℝ) 
  (thickness : ℝ) 
  (radius : ℝ) 
  (h1 : volume = 400) 
  (h2 : thickness = 0.2) 
  (h3 : π * radius^2 * thickness = volume) : 
  radius = Real.sqrt (2000 / π) := by
sorry

end NUMINAMATH_CALUDE_circular_film_radius_l584_58444


namespace NUMINAMATH_CALUDE_odd_integers_sum_169_l584_58456

/-- Sum of consecutive odd integers from 1 to n -/
def sumOddIntegers (n : ℕ) : ℕ :=
  (n * n + n) / 2

/-- The problem statement -/
theorem odd_integers_sum_169 :
  ∃ n : ℕ, n % 2 = 1 ∧ sumOddIntegers n = 169 ∧ n = 25 := by
  sorry

end NUMINAMATH_CALUDE_odd_integers_sum_169_l584_58456


namespace NUMINAMATH_CALUDE_y_value_l584_58407

theorem y_value (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -5) : y = 18 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l584_58407


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l584_58485

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l584_58485


namespace NUMINAMATH_CALUDE_river_depth_problem_l584_58497

theorem river_depth_problem (depth_may : ℝ) (depth_june : ℝ) (depth_july : ℝ) 
  (h1 : depth_june = depth_may + 10)
  (h2 : depth_july = 3 * depth_june)
  (h3 : depth_july = 45) :
  depth_may = 5 := by
sorry

end NUMINAMATH_CALUDE_river_depth_problem_l584_58497


namespace NUMINAMATH_CALUDE_right_triangle_properties_l584_58446

theorem right_triangle_properties (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 9) :
  let variance := (a^2 + b^2 + 9) / 3 - ((a + b + 3) / 3)^2
  let std_dev := Real.sqrt variance
  let min_std_dev := Real.sqrt 2 - 1
  let optimal_leg := 3 * Real.sqrt 2 / 2
  (variance < 5) ∧
  (std_dev ≥ min_std_dev) ∧
  (std_dev = min_std_dev ↔ a = optimal_leg ∧ b = optimal_leg) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_properties_l584_58446


namespace NUMINAMATH_CALUDE_hyperbola_parabola_symmetry_l584_58439

/-- Given a hyperbola and points on a parabola, prove the value of m -/
theorem hyperbola_parabola_symmetry (a b : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (m : ℝ) : 
  a > 0 → b > 0 → 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ |2*x| = 4) →  -- Condition for hyperbola
  y₁ = a*x₁^2 → y₂ = a*x₂^2 →  -- Points on parabola
  (x₁ + x₂)/2 + m = (y₁ + y₂)/2 →  -- Midpoint on symmetry line
  (y₂ - y₁)/(x₂ - x₁) = -1 →  -- Perpendicular to symmetry line
  x₁*x₂ = -1/2 → 
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_symmetry_l584_58439


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_sum_4_l584_58448

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Check if a year is after 2020 and has sum of digits equal to 4 -/
def isValidYear (year : ℕ) : Prop :=
  year > 2020 ∧ sumOfDigits year = 4

/-- 2030 is the first year after 2020 with sum of digits equal to 4 -/
theorem first_year_after_2020_with_sum_4 :
  (∀ y : ℕ, y > 2020 ∧ y < 2030 → sumOfDigits y ≠ 4) ∧
  sumOfDigits 2030 = 4 :=
sorry

end NUMINAMATH_CALUDE_first_year_after_2020_with_sum_4_l584_58448


namespace NUMINAMATH_CALUDE_range_of_a_l584_58479

def has_real_roots (m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁^2 - m*x₁ - 1 = 0 ∧ x₂^2 - m*x₂ - 1 = 0

def roots_difference_bound (a : ℝ) : Prop :=
  ∀ (m : ℝ), has_real_roots m → 
    ∃ (x₁ x₂ : ℝ), x₁^2 - m*x₁ - 1 = 0 ∧ x₂^2 - m*x₂ - 1 = 0 ∧ a^2 + 4*a - 3 ≤ |x₁ - x₂|

def quadratic_has_solution (a : ℝ) : Prop :=
  ∃ (x : ℝ), x^2 + 2*x + a < 0

def proposition_p (a : ℝ) : Prop :=
  has_real_roots 0 ∧ roots_difference_bound a

def proposition_q (a : ℝ) : Prop :=
  quadratic_has_solution a

theorem range_of_a :
  ∀ (a : ℝ), (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) →
    a = 1 ∨ a < -5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l584_58479


namespace NUMINAMATH_CALUDE_police_can_catch_gangster_police_can_reach_same_side_l584_58489

/-- Represents the setup of the police and gangster problem -/
structure PoliceGangsterSetup where
  a : ℝ  -- side length of the square
  police_speed : ℝ  -- speed of the police officer
  gangster_speed : ℝ  -- speed of the gangster
  h_positive_a : 0 < a  -- side length is positive
  h_positive_police_speed : 0 < police_speed  -- police speed is positive
  h_gangster_speed : gangster_speed = 2.9 * police_speed  -- gangster speed is 2.9 times police speed

/-- Theorem stating that the police officer can always reach a side of the square before the gangster moves more than one side length -/
theorem police_can_catch_gangster (setup : PoliceGangsterSetup) :
  setup.a / (2 * setup.police_speed) < 1.45 * setup.a := by
  sorry

/-- Corollary stating that the police officer can always end up on the same side as the gangster -/
theorem police_can_reach_same_side (setup : PoliceGangsterSetup) :
  ∃ (t : ℝ), t > 0 ∧ t * setup.police_speed ≥ setup.a / 2 ∧ t * setup.gangster_speed < setup.a := by
  sorry

end NUMINAMATH_CALUDE_police_can_catch_gangster_police_can_reach_same_side_l584_58489


namespace NUMINAMATH_CALUDE_cubic_function_unique_form_l584_58415

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b

theorem cubic_function_unique_form 
  (a b : ℝ) 
  (h_a : a > 0)
  (h_max : ∃ x₁, ∀ x, f x a b ≤ f x₁ a b ∧ f x₁ a b = 5)
  (h_min : ∃ x₂, ∀ x, f x a b ≥ f x₂ a b ∧ f x₂ a b = 1) :
  ∀ x, f x a b = x^3 + 3*x^2 + 1 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_unique_form_l584_58415


namespace NUMINAMATH_CALUDE_solution_to_inequality_system_l584_58496

theorem solution_to_inequality_system (x : ℝ) :
  (x + 3 > 2 ∧ 1 - 2*x < -3) → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_to_inequality_system_l584_58496


namespace NUMINAMATH_CALUDE_combined_solid_volume_l584_58401

/-- The volume of the combined solid with a square base and triangular prism on top -/
theorem combined_solid_volume (s : ℝ) (h : s = 8 * Real.sqrt 2) :
  let original_volume := (Real.sqrt 2 * (2 * s)^3) / 24
  let prism_volume := (s^3 * Real.sqrt 15) / 4
  original_volume + prism_volume = 2048 + 576 * Real.sqrt 30 := by
sorry

end NUMINAMATH_CALUDE_combined_solid_volume_l584_58401


namespace NUMINAMATH_CALUDE_proposition_false_iff_a_less_than_neg_thirteen_half_l584_58462

theorem proposition_false_iff_a_less_than_neg_thirteen_half :
  (∀ x ∈ Set.Icc 1 2, x^2 + a*x + 9 ≥ 0) = false ↔ a < -13/2 :=
sorry

end NUMINAMATH_CALUDE_proposition_false_iff_a_less_than_neg_thirteen_half_l584_58462


namespace NUMINAMATH_CALUDE_complex_equation_solution_l584_58459

theorem complex_equation_solution (z : ℂ) :
  z * (Complex.I - Complex.I^2) = 1 + Complex.I^3 → z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l584_58459


namespace NUMINAMATH_CALUDE_subtraction_problem_l584_58488

theorem subtraction_problem (x : ℤ) : x - 29 = 63 → x - 47 = 45 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l584_58488


namespace NUMINAMATH_CALUDE_transaction_conservation_l584_58410

/-- Represents the transaction in the restaurant problem -/
structure RestaurantTransaction where
  initial_payment : ℕ
  people : ℕ
  overcharge : ℕ
  refund_per_person : ℕ
  assistant_kept : ℕ

/-- The actual cost of the meal -/
def actual_cost (t : RestaurantTransaction) : ℕ :=
  t.initial_payment - t.overcharge

/-- The amount effectively paid by the customers -/
def effective_payment (t : RestaurantTransaction) : ℕ :=
  t.people * (t.initial_payment / t.people - t.refund_per_person)

/-- Theorem stating that the total amount involved is conserved -/
theorem transaction_conservation (t : RestaurantTransaction) 
  (h1 : t.initial_payment = 30)
  (h2 : t.people = 3)
  (h3 : t.overcharge = 5)
  (h4 : t.refund_per_person = 1)
  (h5 : t.assistant_kept = 2) :
  effective_payment t + (t.people * t.refund_per_person) + t.assistant_kept = t.initial_payment := by
  sorry

#check transaction_conservation

end NUMINAMATH_CALUDE_transaction_conservation_l584_58410


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l584_58499

-- Define the sets A, B, and C
def A : Set ℝ := {x | (x + 3) / (x - 1) ≥ 0}
def B : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def C (a : ℝ) : Set ℝ := {x | x ≥ a^2 - 2}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

-- Theorem for the range of a when B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ∈ Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l584_58499


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l584_58464

/-- Given a triangle with base 12 and altitude 8, and an inscribed rectangle with height 4,
    the area of the rectangle is 48. -/
theorem inscribed_rectangle_area (b h x : ℝ) : 
  b = 12 → h = 8 → x = h / 2 → x = 4 → 
  ∃ (w : ℝ), w * x = 48 := by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l584_58464


namespace NUMINAMATH_CALUDE_price_ratio_theorem_l584_58417

theorem price_ratio_theorem (cost_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) 
  (profit_price : ℝ) (loss_price : ℝ) :
  profit_percent = 26 →
  loss_percent = 16 →
  profit_price = cost_price * (1 + profit_percent / 100) →
  loss_price = cost_price * (1 - loss_percent / 100) →
  loss_price / profit_price = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_price_ratio_theorem_l584_58417


namespace NUMINAMATH_CALUDE_first_day_over_200_paperclips_l584_58494

def paperclip_count (n : ℕ) : ℕ :=
  if n < 2 then 3 else 3 * 2^(n - 2)

theorem first_day_over_200_paperclips :
  ∀ n : ℕ, n < 9 → paperclip_count n ≤ 200 ∧
  paperclip_count 9 > 200 :=
sorry

end NUMINAMATH_CALUDE_first_day_over_200_paperclips_l584_58494


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l584_58419

theorem sum_of_powers_of_i (i : ℂ) (h : i^2 = -1) :
  i^14761 + i^14762 + i^14763 + i^14764 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l584_58419


namespace NUMINAMATH_CALUDE_handshakes_at_gathering_l584_58473

theorem handshakes_at_gathering (n : ℕ) (h : n = 6) : 
  n * (2 * n - 1) = 60 := by
  sorry

#check handshakes_at_gathering

end NUMINAMATH_CALUDE_handshakes_at_gathering_l584_58473


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l584_58470

theorem largest_n_for_factorization : 
  let P (n : ℤ) := ∃ (A B : ℤ), 5 * X^2 + n * X + 120 = (5 * X + A) * (X + B)
  ∀ (m : ℤ), P m → m ≤ 601 ∧ P 601 := by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l584_58470


namespace NUMINAMATH_CALUDE_merchandise_profit_rate_l584_58492

/-- Given a merchandise with cost price x, prove that the profit rate is 5% -/
theorem merchandise_profit_rate (x : ℝ) (h : 1.1 * x - 10 = 210) : 
  (210 - x) / x * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_merchandise_profit_rate_l584_58492


namespace NUMINAMATH_CALUDE_elenas_bread_recipe_l584_58425

/-- Given Elena's bread recipe, prove the amount of butter needed for the original recipe -/
theorem elenas_bread_recipe (original_flour : ℝ) (scale_factor : ℝ) (new_butter : ℝ) (new_flour : ℝ) :
  original_flour = 14 →
  scale_factor = 4 →
  new_butter = 12 →
  new_flour = 56 →
  (new_butter / new_flour) * original_flour = 3 := by
  sorry

end NUMINAMATH_CALUDE_elenas_bread_recipe_l584_58425


namespace NUMINAMATH_CALUDE_max_a_value_l584_58420

theorem max_a_value (a : ℝ) : 
  a > 0 →
  (∀ x ∈ Set.Icc 1 2, ∃ y, y = x - a / x) →
  (∀ M : ℝ × ℝ, M.1 ∈ Set.Icc 1 2 → M.2 = M.1 - a / M.1 → 
    ∀ N : ℝ × ℝ, N.1 = M.1 ∧ 
    N.2 = (1 + a / 2) * (M.1 - 1) + (1 - a) → 
    (M.2 - N.2)^2 ≤ 1) →
  a ≤ 6 + 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l584_58420


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l584_58498

/-- Simple interest calculation -/
def simple_interest (principal time rate : ℝ) : ℝ := principal * time * rate

/-- Theorem: Given the conditions, prove the rate of interest is 0.06 -/
theorem interest_rate_calculation (principal time interest : ℝ) 
  (h_principal : principal = 15000)
  (h_time : time = 3)
  (h_interest : interest = 2700) :
  ∃ rate : ℝ, simple_interest principal time rate = interest ∧ rate = 0.06 := by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_interest_rate_calculation_l584_58498


namespace NUMINAMATH_CALUDE_min_value_theorem_l584_58433

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2*m + n = 1) :
  (1/m + 2/n) ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2*m₀ + n₀ = 1 ∧ 1/m₀ + 2/n₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l584_58433


namespace NUMINAMATH_CALUDE_malcolm_total_followers_l584_58468

/-- The total number of followers Malcolm has on all social media platforms --/
def total_followers (instagram facebook : ℕ) : ℕ :=
  let twitter := (instagram + facebook) / 2
  let tiktok := 3 * twitter
  let youtube := tiktok + 510
  instagram + facebook + twitter + tiktok + youtube

/-- Theorem stating that Malcolm's total followers across all platforms is 3840 --/
theorem malcolm_total_followers :
  total_followers 240 500 = 3840 := by
  sorry

#eval total_followers 240 500

end NUMINAMATH_CALUDE_malcolm_total_followers_l584_58468


namespace NUMINAMATH_CALUDE_z1_over_z2_value_l584_58441

def complex_symmetric_about_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem z1_over_z2_value (z₁ z₂ : ℂ) :
  complex_symmetric_about_imaginary_axis z₁ z₂ →
  z₁ = 3 - I →
  z₁ / z₂ = -4/5 + 3/5 * I :=
by sorry

end NUMINAMATH_CALUDE_z1_over_z2_value_l584_58441


namespace NUMINAMATH_CALUDE_profit_loss_recording_l584_58471

/-- Represents the financial record of a store. -/
inductive FinancialRecord
  | profit (amount : ℤ)
  | loss (amount : ℤ)

/-- Records a financial transaction. -/
def recordTransaction (transaction : FinancialRecord) : ℤ :=
  match transaction with
  | FinancialRecord.profit amount => amount
  | FinancialRecord.loss amount => -amount

/-- The theorem stating how profits and losses should be recorded. -/
theorem profit_loss_recording (profitAmount lossAmount : ℤ) 
  (h : profitAmount = 20 ∧ lossAmount = 10) : 
  recordTransaction (FinancialRecord.profit profitAmount) = 20 ∧
  recordTransaction (FinancialRecord.loss lossAmount) = -10 := by
  sorry

end NUMINAMATH_CALUDE_profit_loss_recording_l584_58471


namespace NUMINAMATH_CALUDE_additional_marbles_needed_l584_58411

def friends : ℕ := 12
def current_marbles : ℕ := 50

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem additional_marbles_needed : 
  sum_first_n friends - current_marbles = 28 := by
  sorry

end NUMINAMATH_CALUDE_additional_marbles_needed_l584_58411


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l584_58481

theorem six_digit_divisibility (abc : Nat) (h : abc ≥ 100 ∧ abc < 1000) :
  let abcabc := abc * 1000 + abc
  (abcabc % 11 = 0) ∧ (abcabc % 13 = 0) ∧ (abcabc % 1001 = 0) ∧
  ∃ x : Nat, x ≥ 100 ∧ x < 1000 ∧ (x * 1000 + x) % 101 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l584_58481


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l584_58452

theorem trigonometric_equation_solutions :
  ∀ x : ℝ, x ∈ Set.Icc 0 (2 * Real.pi) →
    (3 * Real.sin x = 1 + Real.cos (2 * x)) ↔ (x = Real.pi / 6 ∨ x = 5 * Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l584_58452


namespace NUMINAMATH_CALUDE_smartphone_sale_price_l584_58436

theorem smartphone_sale_price (initial_cost : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  initial_cost = 300 →
  loss_percentage = 15 →
  selling_price = initial_cost - (loss_percentage / 100) * initial_cost →
  selling_price = 255 := by
sorry

end NUMINAMATH_CALUDE_smartphone_sale_price_l584_58436


namespace NUMINAMATH_CALUDE_average_age_increase_l584_58400

theorem average_age_increase 
  (n : Nat) 
  (initial_avg : ℝ) 
  (man1_age man2_age : ℝ) 
  (women_avg : ℝ) : 
  n = 8 → 
  man1_age = 20 → 
  man2_age = 22 → 
  women_avg = 29 → 
  ((n * initial_avg - man1_age - man2_age + 2 * women_avg) / n) - initial_avg = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l584_58400


namespace NUMINAMATH_CALUDE_intersection_union_equality_l584_58424

def M : Set Nat := {0, 1, 2, 4, 5, 7}
def N : Set Nat := {1, 4, 6, 8, 9}
def P : Set Nat := {4, 7, 9}

theorem intersection_union_equality : (M ∩ N) ∪ (M ∩ P) = {1, 4, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_union_equality_l584_58424
