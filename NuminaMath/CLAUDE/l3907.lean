import Mathlib

namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l3907_390734

-- Define a perfect square trinomial
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (x + k)^2

theorem perfect_square_trinomial_m_value (m : ℝ) :
  is_perfect_square_trinomial 1 (2*m) 9 → m = 3 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l3907_390734


namespace NUMINAMATH_CALUDE_ratio_problem_l3907_390751

theorem ratio_problem (a b c : ℚ) 
  (h1 : c / b = 4)
  (h2 : b / a = 2)
  (h3 : c = 20 - 7 * b) :
  a = 10 / 11 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3907_390751


namespace NUMINAMATH_CALUDE_sum_of_coefficients_factorized_form_l3907_390739

theorem sum_of_coefficients_factorized_form (x y : ℝ) : 
  ∃ (a b c d e : ℤ), 
    27 * x^6 - 512 * y^6 = (a * x^2 + b * y^2) * (c * x^4 + d * x^2 * y^2 + e * y^4) ∧
    a + b + c + d + e = 92 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_factorized_form_l3907_390739


namespace NUMINAMATH_CALUDE_circular_course_circumference_l3907_390745

/-- The circumference of a circular course where two people walking at different speeds meet after a certain time. -/
theorem circular_course_circumference
  (speed_a speed_b : ℝ)
  (meeting_time : ℝ)
  (h1 : speed_a = 4)
  (h2 : speed_b = 5)
  (h3 : meeting_time = 115)
  (h4 : speed_b > speed_a) :
  (speed_b - speed_a) * meeting_time = 115 :=
by sorry

end NUMINAMATH_CALUDE_circular_course_circumference_l3907_390745


namespace NUMINAMATH_CALUDE_square_field_side_length_l3907_390792

theorem square_field_side_length (area : ℝ) (side : ℝ) :
  area = 256 → side ^ 2 = area → side = 16 := by sorry

end NUMINAMATH_CALUDE_square_field_side_length_l3907_390792


namespace NUMINAMATH_CALUDE_min_value_w_l3907_390775

theorem min_value_w (x y : ℝ) : 
  ∃ (w_min : ℝ), w_min = 20.25 ∧ ∀ (w : ℝ), w = 3*x^2 + 3*y^2 + 9*x - 6*y + 27 → w ≥ w_min :=
by sorry

end NUMINAMATH_CALUDE_min_value_w_l3907_390775


namespace NUMINAMATH_CALUDE_celia_savings_l3907_390749

def weekly_food_budget : ℕ := 100
def num_weeks : ℕ := 4
def monthly_rent : ℕ := 1500
def monthly_streaming : ℕ := 30
def monthly_cell_phone : ℕ := 50
def savings_rate : ℚ := 1 / 10

def total_spending : ℕ := weekly_food_budget * num_weeks + monthly_rent + monthly_streaming + monthly_cell_phone

def savings : ℚ := (total_spending : ℚ) * savings_rate

theorem celia_savings : savings = 198 := by sorry

end NUMINAMATH_CALUDE_celia_savings_l3907_390749


namespace NUMINAMATH_CALUDE_chris_age_l3907_390709

/-- The ages of four friends satisfying certain conditions -/
def FriendsAges (a b c d : ℝ) : Prop :=
  -- The average age is 12
  (a + b + c + d) / 4 = 12 ∧
  -- Five years ago, Chris was twice as old as Amy
  c - 5 = 2 * (a - 5) ∧
  -- In 2 years, Ben's age will be three-quarters of Amy's age
  b + 2 = 3/4 * (a + 2) ∧
  -- Diana is 15 years old
  d = 15

/-- Chris's age is 16 given the conditions -/
theorem chris_age (a b c d : ℝ) (h : FriendsAges a b c d) : c = 16 := by
  sorry

end NUMINAMATH_CALUDE_chris_age_l3907_390709


namespace NUMINAMATH_CALUDE_average_movie_price_l3907_390789

theorem average_movie_price (dvd_count : ℕ) (dvd_price : ℚ) (bluray_count : ℕ) (bluray_price : ℚ) : 
  dvd_count = 8 → 
  dvd_price = 12 → 
  bluray_count = 4 → 
  bluray_price = 18 → 
  (dvd_count * dvd_price + bluray_count * bluray_price) / (dvd_count + bluray_count) = 14 := by
sorry

end NUMINAMATH_CALUDE_average_movie_price_l3907_390789


namespace NUMINAMATH_CALUDE_function_range_l3907_390768

theorem function_range (m : ℝ) : 
  (∀ x : ℝ, (2 * m * x^2 - 2 * (4 - m) * x + 1 > 0) ∨ (m * x > 0)) → 
  (m > 0 ∧ m < 8) := by
sorry

end NUMINAMATH_CALUDE_function_range_l3907_390768


namespace NUMINAMATH_CALUDE_inner_quadrilateral_area_l3907_390784

/-- A square with side length 10 cm, partitioned by lines from corners to opposite midpoints -/
structure PartitionedSquare where
  side_length : ℝ
  is_ten_cm : side_length = 10

/-- The inner quadrilateral formed by the intersecting lines -/
def inner_quadrilateral (s : PartitionedSquare) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The area of the inner quadrilateral is 25 cm² -/
theorem inner_quadrilateral_area (s : PartitionedSquare) :
  area (inner_quadrilateral s) = 25 := by
  sorry

end NUMINAMATH_CALUDE_inner_quadrilateral_area_l3907_390784


namespace NUMINAMATH_CALUDE_triangle_point_distance_inequality_triangle_point_distance_equality_condition_l3907_390719

-- Define a triangle ABC
variable (A B C : ℝ × ℝ)

-- Define a point P inside or on the boundary of triangle ABC
variable (P : ℝ × ℝ)

-- Define distances from P to sides of the triangle
def da : ℝ := sorry
def db : ℝ := sorry
def dc : ℝ := sorry

-- Define distances from P to vertices of the triangle
def AP : ℝ := sorry
def BP : ℝ := sorry
def CP : ℝ := sorry

-- Theorem statement
theorem triangle_point_distance_inequality :
  (max AP (max BP CP)) ≥ Real.sqrt (da^2 + db^2 + dc^2) :=
sorry

-- Equality condition
theorem triangle_point_distance_equality_condition :
  (max AP (max BP CP)) = Real.sqrt (da^2 + db^2 + dc^2) ↔
  (A = B ∧ B = C) ∧ P = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_point_distance_inequality_triangle_point_distance_equality_condition_l3907_390719


namespace NUMINAMATH_CALUDE_quadratic_solution_l3907_390701

theorem quadratic_solution (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9 : ℝ) - 45 = 0) → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3907_390701


namespace NUMINAMATH_CALUDE_fewer_noodles_than_pirates_l3907_390733

theorem fewer_noodles_than_pirates (noodles pirates : ℕ) : 
  noodles < pirates →
  pirates = 45 →
  noodles + pirates = 83 →
  pirates - noodles = 7 := by
sorry

end NUMINAMATH_CALUDE_fewer_noodles_than_pirates_l3907_390733


namespace NUMINAMATH_CALUDE_constant_in_toll_formula_l3907_390726

/-- The toll formula for a truck crossing a bridge -/
def toll (x : ℕ) (constant : ℝ) : ℝ :=
  1.50 + 0.50 * (x - constant)

/-- The number of axles on an 18-wheel truck with 2 wheels on its front axle and 2 wheels on each of its other axles -/
def axles_18_wheel_truck : ℕ := 9

theorem constant_in_toll_formula :
  ∃ (constant : ℝ), 
    toll axles_18_wheel_truck constant = 5 ∧ 
    constant = 2 := by sorry

end NUMINAMATH_CALUDE_constant_in_toll_formula_l3907_390726


namespace NUMINAMATH_CALUDE_fraction_above_line_l3907_390762

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A line in the 2D plane defined by two points --/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the area of a square --/
def squareArea (s : Square) : ℝ :=
  let (x1, y1) := s.bottomLeft
  let (x2, y2) := s.topRight
  (x2 - x1) * (y2 - y1)

/-- Calculate the area of the part of the square above the line --/
def areaAboveLine (s : Square) (l : Line) : ℝ :=
  sorry  -- The actual calculation would go here

/-- The main theorem --/
theorem fraction_above_line (s : Square) (l : Line) : 
  s.bottomLeft = (4, 0) → 
  s.topRight = (9, 5) → 
  l.point1 = (4, 1) → 
  l.point2 = (9, 5) → 
  areaAboveLine s l / squareArea s = 9 / 10 := by
  sorry


end NUMINAMATH_CALUDE_fraction_above_line_l3907_390762


namespace NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l3907_390765

theorem smallest_angle_in_special_triangle :
  ∀ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    a + b + c = 180 →
    b = 3 * a →
    c = 5 * a →
    a = 20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_special_triangle_l3907_390765


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l3907_390717

/-- The rate per kg for grapes that Andrew purchased -/
def grape_rate : ℝ := 74

/-- The amount of grapes Andrew purchased in kg -/
def grape_amount : ℝ := 6

/-- The rate per kg for mangoes that Andrew purchased -/
def mango_rate : ℝ := 59

/-- The amount of mangoes Andrew purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount Andrew paid to the shopkeeper -/
def total_paid : ℝ := 975

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l3907_390717


namespace NUMINAMATH_CALUDE_ron_four_times_maurice_age_l3907_390727

/-- The number of years in the future when Ron will be four times as old as Maurice -/
def years_until_four_times_age : ℕ → ℕ → ℕ 
| ron_age, maurice_age => 
  let x : ℕ := (ron_age - 4 * maurice_age) / 3
  x

theorem ron_four_times_maurice_age (ron_current_age maurice_current_age : ℕ) 
  (h1 : ron_current_age = 43)
  (h2 : maurice_current_age = 7) : 
  years_until_four_times_age ron_current_age maurice_current_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_ron_four_times_maurice_age_l3907_390727


namespace NUMINAMATH_CALUDE_spider_web_paths_l3907_390740

/-- The number of paths in a grid where only right and up moves are allowed -/
def number_of_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem: In a 7x3 grid, the number of paths from bottom-left to top-right
    moving only right and up is equal to (10 choose 7) -/
theorem spider_web_paths :
  number_of_paths 7 3 = Nat.choose 10 7 := by
  sorry

end NUMINAMATH_CALUDE_spider_web_paths_l3907_390740


namespace NUMINAMATH_CALUDE_ryan_to_bill_ratio_l3907_390783

/-- Represents the number of math problems composed by each person -/
structure ProblemCounts where
  bill : ℕ
  ryan : ℕ
  frank : ℕ

/-- Represents the conditions of the problem -/
def problem_conditions (p : ProblemCounts) : Prop :=
  p.bill = 20 ∧
  p.frank = 3 * p.ryan ∧
  p.frank = 30 * 4

/-- The theorem to be proved -/
theorem ryan_to_bill_ratio 
  (p : ProblemCounts) 
  (h : problem_conditions p) : 
  p.ryan / p.bill = 2 := by
  sorry

#check ryan_to_bill_ratio

end NUMINAMATH_CALUDE_ryan_to_bill_ratio_l3907_390783


namespace NUMINAMATH_CALUDE_smallest_k_for_same_color_squares_k_is_smallest_l3907_390780

theorem smallest_k_for_same_color_squares (n : ℕ) (hn : n > 0) :
  let k := 2 * n^2 - n + 1
  ∀ (f : ℕ → ℕ → Fin n),
    ∃ (r₁ r₂ c₁ c₂ : ℕ) (h₁ : r₁ < 2*n) (h₂ : r₂ < 2*n) (h₃ : r₁ ≠ r₂)
    (h₄ : c₁ < k) (h₅ : c₂ < k) (h₆ : c₁ ≠ c₂),
    f r₁ c₁ = f r₁ c₂ ∧ f r₁ c₁ = f r₂ c₁ ∧ f r₁ c₁ = f r₂ c₂ :=
by sorry

theorem k_is_smallest (n : ℕ) (hn : n > 0) :
  ∀ k < 2 * n^2 - n + 1,
    ∃ (f : ℕ → ℕ → Fin n),
      ∀ (r₁ r₂ c₁ c₂ : ℕ) (h₁ : r₁ < 2*n) (h₂ : r₂ < 2*n) (h₃ : r₁ ≠ r₂)
      (h₄ : c₁ < k) (h₅ : c₂ < k) (h₆ : c₁ ≠ c₂),
      f r₁ c₁ ≠ f r₁ c₂ ∨ f r₁ c₁ ≠ f r₂ c₁ ∨ f r₁ c₁ ≠ f r₂ c₂ :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_same_color_squares_k_is_smallest_l3907_390780


namespace NUMINAMATH_CALUDE_artist_june_pictures_l3907_390700

/-- The number of pictures painted in June -/
def june_pictures : ℕ := sorry

/-- The number of pictures painted in July -/
def july_pictures : ℕ := june_pictures + 2

/-- The number of pictures painted in August -/
def august_pictures : ℕ := 9

/-- The total number of pictures painted over the three months -/
def total_pictures : ℕ := 13

theorem artist_june_pictures :
  june_pictures = 1 ∧
  june_pictures + july_pictures + august_pictures = total_pictures :=
sorry

end NUMINAMATH_CALUDE_artist_june_pictures_l3907_390700


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3907_390778

theorem angle_sum_is_pi_over_two (α β : Real)
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h4 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l3907_390778


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3907_390769

theorem angle_measure_proof : ∃! x : ℝ, 0 < x ∧ x < 90 ∧ x + (3 * x^2 + 10) = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3907_390769


namespace NUMINAMATH_CALUDE_initial_salt_percentage_l3907_390756

theorem initial_salt_percentage
  (initial_mass : ℝ)
  (added_salt : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_mass = 100)
  (h2 : added_salt = 38.46153846153846)
  (h3 : final_percentage = 35) :
  let final_mass := initial_mass + added_salt
  let final_salt_mass := (final_percentage / 100) * final_mass
  let initial_salt_mass := final_salt_mass - added_salt
  initial_salt_mass / initial_mass * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_initial_salt_percentage_l3907_390756


namespace NUMINAMATH_CALUDE_cube_volume_division_l3907_390742

theorem cube_volume_division (V : ℝ) (a b : ℝ) (h1 : V > 0) (h2 : b > a) (h3 : a > 0) :
  let diagonal_ratio := a / (b - a)
  let volume_ratio := a^3 / (b^3 - a^3)
  ∃ (V1 V2 : ℝ), V1 + V2 = V ∧ V1 / V2 = volume_ratio :=
sorry

end NUMINAMATH_CALUDE_cube_volume_division_l3907_390742


namespace NUMINAMATH_CALUDE_angle_subtraction_quadrant_l3907_390731

/-- An angle is in the second quadrant if it's between 90° and 180° -/
def is_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α < 180

/-- An angle is in the first quadrant if it's between 0° and 90° -/
def is_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < 90

theorem angle_subtraction_quadrant (α : ℝ) (h : is_second_quadrant α) : 
  is_first_quadrant (180 - α) := by
  sorry

end NUMINAMATH_CALUDE_angle_subtraction_quadrant_l3907_390731


namespace NUMINAMATH_CALUDE_cos_240_degrees_l3907_390761

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l3907_390761


namespace NUMINAMATH_CALUDE_min_distance_four_points_l3907_390706

/-- Given four points in a metric space with specified distances between consecutive points,
    the theorem states that the minimum possible distance between the first and last points is 3. -/
theorem min_distance_four_points (X : Type*) [MetricSpace X] (P Q R S : X) :
  dist P Q = 12 →
  dist Q R = 7 →
  dist R S = 2 →
  ∃ (configuration : X → X), dist (configuration P) (configuration S) = 3 ∧
  (∀ (P' Q' R' S' : X),
    dist P' Q' = 12 →
    dist Q' R' = 7 →
    dist R' S' = 2 →
    dist P' S' ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_four_points_l3907_390706


namespace NUMINAMATH_CALUDE_max_self_intersections_l3907_390785

/-- A polygonal chain on a graph paper -/
structure PolygonalChain where
  segments : ℕ
  closed : Bool
  on_graph_paper : Bool
  no_segments_on_same_line : Bool

/-- The number of self-intersection points of a polygonal chain -/
def self_intersection_points (chain : PolygonalChain) : ℕ := sorry

/-- Theorem: The maximum number of self-intersection points for a closed 14-segment polygonal chain 
    on a graph paper, where no two segments lie on the same line, is 17 -/
theorem max_self_intersections (chain : PolygonalChain) :
  chain.segments = 14 ∧ 
  chain.closed ∧ 
  chain.on_graph_paper ∧ 
  chain.no_segments_on_same_line →
  self_intersection_points chain ≤ 17 ∧ 
  ∃ (chain' : PolygonalChain), 
    chain'.segments = 14 ∧ 
    chain'.closed ∧ 
    chain'.on_graph_paper ∧ 
    chain'.no_segments_on_same_line ∧
    self_intersection_points chain' = 17 :=
sorry

end NUMINAMATH_CALUDE_max_self_intersections_l3907_390785


namespace NUMINAMATH_CALUDE_number_pair_problem_l3907_390752

theorem number_pair_problem (a b : ℕ) : 
  a + b = 62 → 
  (a = b + 12 ∨ b = a + 12) → 
  (a = 25 ∨ b = 25) → 
  (a = 37 ∨ b = 37) :=
by sorry

end NUMINAMATH_CALUDE_number_pair_problem_l3907_390752


namespace NUMINAMATH_CALUDE_larger_number_proof_l3907_390703

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 2342) (h3 : L = 9 * S + 23) : L = 2624 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3907_390703


namespace NUMINAMATH_CALUDE_square_field_area_l3907_390746

/-- Proves that a square field crossed diagonally in 9 seconds by a man walking at 6 km/h has an area of 112.5 square meters. -/
theorem square_field_area (speed_kmh : ℝ) (time_s : ℝ) (area : ℝ) : 
  speed_kmh = 6 → time_s = 9 → area = 112.5 → 
  let speed_ms := speed_kmh * 1000 / 3600
  let diagonal := speed_ms * time_s
  let side := (diagonal^2 / 2).sqrt
  area = side^2 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l3907_390746


namespace NUMINAMATH_CALUDE_divisible_by_27_l3907_390711

theorem divisible_by_27 (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) :
  ∃ k : ℤ, x + y + z = 27 * k := by
sorry

end NUMINAMATH_CALUDE_divisible_by_27_l3907_390711


namespace NUMINAMATH_CALUDE_union_complement_equals_set_l3907_390798

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 4}
def N : Set Nat := {2, 5}

theorem union_complement_equals_set : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equals_set_l3907_390798


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3907_390793

theorem smallest_n_congruence (n : ℕ+) : (23 * n.val ≡ 5678 [ZMOD 11]) ↔ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3907_390793


namespace NUMINAMATH_CALUDE_closure_of_A_range_of_a_l3907_390730

-- Define set A
def A : Set ℝ := {x | x < -1 ∨ x > -1/2}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}

-- Theorem for the closure of A
theorem closure_of_A : 
  closure A = {x : ℝ | -1 ≤ x ∧ x ≤ -1/2} := by sorry

-- Theorem for the range of a
theorem range_of_a : 
  (∃ a : ℝ, A ∪ B a = Set.univ) ↔ ∃ a : ℝ, -3/2 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_closure_of_A_range_of_a_l3907_390730


namespace NUMINAMATH_CALUDE_two_person_island_puzzle_l3907_390758

/-- Represents a person who can either be a liar or a truth-teller -/
inductive Person
  | Liar
  | TruthTeller

/-- The statement of a person about the number of truth-tellers -/
def statement (p : Person) (actual_truth_tellers : Nat) : Nat :=
  match p with
  | Person.Liar => actual_truth_tellers - 1  -- A liar reduces the number by one
  | Person.TruthTeller => actual_truth_tellers

/-- The main theorem -/
theorem two_person_island_puzzle (total_population : Nat) (liars truth_tellers : Nat)
    (h1 : total_population = liars + truth_tellers)
    (h2 : liars = 1000)
    (h3 : truth_tellers = 1000)
    (person1 person2 : Person)
    (h4 : statement person1 truth_tellers ≠ statement person2 truth_tellers) :
    person1 = Person.Liar ∧ person2 = Person.TruthTeller :=
  sorry


end NUMINAMATH_CALUDE_two_person_island_puzzle_l3907_390758


namespace NUMINAMATH_CALUDE_pat_calculation_error_l3907_390767

theorem pat_calculation_error (x : ℝ) : 
  (x / 7 - 20 = 13) → (7 * x + 20 > 1100) := by
  sorry

end NUMINAMATH_CALUDE_pat_calculation_error_l3907_390767


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3907_390796

theorem quadratic_roots_property (m n : ℝ) : 
  (∀ x, x^2 - 3*x + 1 = 0 ↔ x = m ∨ x = n) →
  -m - n - m*n = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3907_390796


namespace NUMINAMATH_CALUDE_no_prime_solution_l3907_390747

def base_p_to_decimal (n : ℕ) (p : ℕ) : ℕ :=
  let digits := n.digits p
  (List.range digits.length).foldl (λ acc i => acc + digits[i]! * p ^ i) 0

theorem no_prime_solution :
  ∀ p : ℕ, p.Prime → p ≠ 2 → p ≠ 3 → p ≠ 5 → p ≠ 7 →
    base_p_to_decimal 1014 p + base_p_to_decimal 309 p + base_p_to_decimal 120 p +
    base_p_to_decimal 132 p + base_p_to_decimal 7 p ≠
    base_p_to_decimal 153 p + base_p_to_decimal 276 p + base_p_to_decimal 371 p :=
by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l3907_390747


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3907_390737

theorem intersection_of_lines : 
  ∃! (x y : ℚ), 2 * y = 3 * x ∧ 3 * y + 1 = -6 * x ∧ x = -2/21 ∧ y = -1/7 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3907_390737


namespace NUMINAMATH_CALUDE_probability_three_and_zero_painted_faces_l3907_390777

/-- Represents a 5x5x5 cube with three faces sharing a common corner painted red -/
structure PaintedCube :=
  (side_length : ℕ)
  (total_cubes : ℕ)
  (painted_faces : ℕ)

/-- Counts the number of unit cubes with a specific number of painted faces -/
def count_painted_cubes (c : PaintedCube) (num_painted_faces : ℕ) : ℕ :=
  sorry

/-- Calculates the probability of selecting two specific types of unit cubes -/
def probability_of_selection (c : PaintedCube) (faces1 faces2 : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the probability of selecting one cube with three
    painted faces and one cube with no painted faces -/
theorem probability_three_and_zero_painted_faces (c : PaintedCube) :
  c.side_length = 5 ∧ c.total_cubes = 125 ∧ c.painted_faces = 3 →
  probability_of_selection c 3 0 = 44 / 3875 :=
sorry

end NUMINAMATH_CALUDE_probability_three_and_zero_painted_faces_l3907_390777


namespace NUMINAMATH_CALUDE_max_xy_value_l3907_390714

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  xy ≤ 1/8 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ x * y = 1/8 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l3907_390714


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3907_390779

theorem modulus_of_complex_number (θ : Real) (h : 2 * Real.pi < θ ∧ θ < 3 * Real.pi) :
  Complex.abs (1 - Real.cos θ + Complex.I * Real.sin θ) = -2 * Real.sin (θ / 2) := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3907_390779


namespace NUMINAMATH_CALUDE_ones_digit_of_33_power_l3907_390771

theorem ones_digit_of_33_power (n : ℕ) : 
  (33^(33*(12^12))) % 10 = 1 := by
sorry

end NUMINAMATH_CALUDE_ones_digit_of_33_power_l3907_390771


namespace NUMINAMATH_CALUDE_sequence_ratio_l3907_390797

/-- Given a sequence a with sum S of its first n terms satisfying 3S_n - 6 = 2a_n,
    prove that S_5 / a_5 = 11/16 -/
theorem sequence_ratio (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h : ∀ n, 3 * S n - 6 = 2 * a n) :
  S 5 / a 5 = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_l3907_390797


namespace NUMINAMATH_CALUDE_boat_speed_l3907_390736

theorem boat_speed (current_speed : ℝ) (downstream_distance : ℝ) (time : ℝ) :
  current_speed = 3 →
  downstream_distance = 6.75 →
  time = 0.25 →
  ∃ (boat_speed : ℝ),
    boat_speed = 24 ∧
    downstream_distance = (boat_speed + current_speed) * time :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l3907_390736


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l3907_390716

theorem integer_roots_of_cubic (x : ℤ) :
  x^3 - 4*x^2 - 11*x + 24 = 0 ↔ x = -4 ∨ x = 3 ∨ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l3907_390716


namespace NUMINAMATH_CALUDE_glass_bottles_count_l3907_390760

/-- The number of glass bottles initially weighed -/
def initial_glass_bottles : ℕ := 3

/-- The weight of a plastic bottle in grams -/
def plastic_bottle_weight : ℕ := 50

/-- The weight of a glass bottle in grams -/
def glass_bottle_weight : ℕ := plastic_bottle_weight + 150

theorem glass_bottles_count :
  (initial_glass_bottles * glass_bottle_weight = 600) ∧
  (4 * glass_bottle_weight + 5 * plastic_bottle_weight = 1050) ∧
  (glass_bottle_weight = plastic_bottle_weight + 150) →
  initial_glass_bottles = 3 :=
by sorry

end NUMINAMATH_CALUDE_glass_bottles_count_l3907_390760


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3907_390772

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Main theorem
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_product : a 2 * a 4 = 4) :
  a 1 * a 5 + a 3 = 6 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l3907_390772


namespace NUMINAMATH_CALUDE_odd_product_probability_l3907_390766

theorem odd_product_probability : 
  let n : ℕ := 25
  let odd_count : ℕ := (n + 1) / 2
  let total_combinations : ℕ := n * (n - 1) / 2
  let odd_combinations : ℕ := odd_count * (odd_count - 1) / 2
  (odd_combinations : ℚ) / total_combinations = 13 / 50 := by sorry

end NUMINAMATH_CALUDE_odd_product_probability_l3907_390766


namespace NUMINAMATH_CALUDE_negative_one_times_negative_three_l3907_390770

theorem negative_one_times_negative_three : (-1 : ℤ) * (-3 : ℤ) = (3 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_negative_one_times_negative_three_l3907_390770


namespace NUMINAMATH_CALUDE_solution_x_l3907_390744

theorem solution_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 104) : x = 34 := by
  sorry

end NUMINAMATH_CALUDE_solution_x_l3907_390744


namespace NUMINAMATH_CALUDE_min_value_a_l3907_390795

theorem min_value_a (a : ℝ) : 
  (∀ x > a, 2 * x + 2 / (x - 1) ≥ 7) → a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l3907_390795


namespace NUMINAMATH_CALUDE_johns_days_off_l3907_390759

/-- Calculates the number of days John takes off per week given his streaming schedule and earnings. -/
theorem johns_days_off (hours_per_session : ℕ) (hourly_rate : ℕ) (weekly_earnings : ℕ) (days_per_week : ℕ)
  (h1 : hours_per_session = 4)
  (h2 : hourly_rate = 10)
  (h3 : weekly_earnings = 160)
  (h4 : days_per_week = 7) :
  days_per_week - (weekly_earnings / hourly_rate / hours_per_session) = 3 :=
by sorry

end NUMINAMATH_CALUDE_johns_days_off_l3907_390759


namespace NUMINAMATH_CALUDE_element_in_intersection_complement_l3907_390718

theorem element_in_intersection_complement (S : Type) (A B : Set S) (a : S) :
  Set.Nonempty A →
  Set.Nonempty B →
  A ⊂ Set.univ →
  B ⊂ Set.univ →
  a ∈ A →
  a ∉ B →
  a ∈ A ∩ (Set.univ \ B) :=
by sorry

end NUMINAMATH_CALUDE_element_in_intersection_complement_l3907_390718


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3907_390728

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∀ x y, (x - c)^2 + y^2 = 4 * a^2) →  -- Circle equation
  (c^2 = a^2 * (1 + b^2 / a^2)) →  -- Semi-latus rectum condition
  (∃ x y, (b * x + a * y = 0) ∧ (x - c)^2 + y^2 = 4 * a^2 ∧ 
    ∃ x' y', (b * x' + a * y' = 0) ∧ (x' - c)^2 + y'^2 = 4 * a^2 ∧
    (x - x')^2 + (y - y')^2 = 4 * b^2) →  -- Asymptote intercepted by circle
  (c^2 / a^2 - 1)^(1/2) = Real.sqrt 3 :=  -- Eccentricity equals sqrt(3)
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3907_390728


namespace NUMINAMATH_CALUDE_inequalities_comparison_l3907_390721

theorem inequalities_comparison (a b : ℝ) (h : a > b) : (a - 3 > b - 3) ∧ (-4*a < -4*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_comparison_l3907_390721


namespace NUMINAMATH_CALUDE_complement_of_28_39_l3907_390708

/-- Represents an angle in degrees and minutes -/
structure Angle where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the complement of an angle -/
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  { degrees := totalMinutes / 60,
    minutes := totalMinutes % 60,
    valid := by sorry }

theorem complement_of_28_39 :
  let a : Angle := { degrees := 28, minutes := 39, valid := by sorry }
  complement a = { degrees := 61, minutes := 21, valid := by sorry } := by
  sorry

end NUMINAMATH_CALUDE_complement_of_28_39_l3907_390708


namespace NUMINAMATH_CALUDE_sampling_probabilities_equal_l3907_390774

/-- Represents the composition of a batch of components -/
structure BatchComposition where
  total : ℕ
  first_class : ℕ
  second_class : ℕ
  third_class : ℕ
  unqualified : ℕ

/-- Represents the probabilities of selecting an individual component using different sampling methods -/
structure SamplingProbabilities where
  simple_random : ℚ
  stratified : ℚ
  systematic : ℚ

/-- Theorem stating that all sampling probabilities are equal to 1/8 for the given batch composition and sample size -/
theorem sampling_probabilities_equal (batch : BatchComposition) (sample_size : ℕ) 
  (h1 : batch.total = 160)
  (h2 : batch.first_class = 48)
  (h3 : batch.second_class = 64)
  (h4 : batch.third_class = 32)
  (h5 : batch.unqualified = 16)
  (h6 : sample_size = 20)
  (h7 : batch.total = batch.first_class + batch.second_class + batch.third_class + batch.unqualified) :
  ∃ (probs : SamplingProbabilities), 
    probs.simple_random = 1/8 ∧ 
    probs.stratified = 1/8 ∧ 
    probs.systematic = 1/8 := by
  sorry


end NUMINAMATH_CALUDE_sampling_probabilities_equal_l3907_390774


namespace NUMINAMATH_CALUDE_percentage_calculation_l3907_390722

theorem percentage_calculation : 
  (0.2 * (0.75 * 800)) / 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3907_390722


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l3907_390738

theorem smallest_five_digit_divisible_by_first_five_primes :
  let first_five_primes := [2, 3, 5, 7, 11]
  let is_five_digit (n : ℕ) := 10000 ≤ n ∧ n ≤ 99999
  let divisible_by_all (n : ℕ) := ∀ p ∈ first_five_primes, n % p = 0
  ∃ (n : ℕ), is_five_digit n ∧ divisible_by_all n ∧
    ∀ m, is_five_digit m ∧ divisible_by_all m → n ≤ m ∧ n = 11550 :=
by
  sorry

#eval 11550 % 2  -- Should output 0
#eval 11550 % 3  -- Should output 0
#eval 11550 % 5  -- Should output 0
#eval 11550 % 7  -- Should output 0
#eval 11550 % 11 -- Should output 0

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_first_five_primes_l3907_390738


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3907_390773

/-- Given that the solution set of ax^2 + 5x + b > 0 is {x | 2 < x < 3},
    prove that the solution set of bx^2 - 5x + a < 0 is {x | x < -1/2 or x > -1/3} -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | a * x^2 + 5 * x + b > 0}) :
  {x : ℝ | b * x^2 - 5 * x + a < 0} = {x : ℝ | x < -1/2 ∨ x > -1/3} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3907_390773


namespace NUMINAMATH_CALUDE_expression_evaluation_l3907_390715

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  (x^3 + 1) / x * (y^3 + 1) / y + (x^3 - 1) / y * (y^3 - 1) / x = 2 * x^2 * y^2 + 2 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3907_390715


namespace NUMINAMATH_CALUDE_final_brand_z_percentage_l3907_390750

/-- Represents the state of the fuel tank -/
structure TankState where
  brandZ : ℚ  -- Amount of Brand Z gasoline
  brandX : ℚ  -- Amount of Brand X gasoline

/-- Fills the tank with Brand Z gasoline -/
def fillWithZ (s : TankState) : TankState :=
  { brandZ := s.brandZ + (1 - s.brandZ - s.brandX), brandX := s.brandX }

/-- Fills the tank with Brand X gasoline -/
def fillWithX (s : TankState) : TankState :=
  { brandZ := s.brandZ, brandX := s.brandX + (1 - s.brandZ - s.brandX) }

/-- Empties the tank by the given fraction -/
def emptyTank (s : TankState) (fraction : ℚ) : TankState :=
  { brandZ := s.brandZ * (1 - fraction), brandX := s.brandX * (1 - fraction) }

/-- The main theorem stating the final percentage of Brand Z gasoline -/
theorem final_brand_z_percentage : 
  let s0 := TankState.mk 1 0  -- Initial state: full of Brand Z
  let s1 := fillWithX (emptyTank s0 (3/4))  -- 3/4 empty, fill with X
  let s2 := fillWithZ (emptyTank s1 (1/2))  -- 1/2 empty, fill with Z
  let s3 := fillWithX (emptyTank s2 (1/2))  -- 1/2 empty, fill with X
  s3.brandZ / (s3.brandZ + s3.brandX) = 5/16 := by
  sorry

#eval (5/16 : ℚ) * 100  -- Should evaluate to 31.25

end NUMINAMATH_CALUDE_final_brand_z_percentage_l3907_390750


namespace NUMINAMATH_CALUDE_modular_congruence_unique_solution_l3907_390732

theorem modular_congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 38635 % 23 = n := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_unique_solution_l3907_390732


namespace NUMINAMATH_CALUDE_value_of_a_value_of_b_when_perpendicular_distance_when_parallel_l3907_390753

-- Define the lines
def l1 (a : ℝ) : ℝ → ℝ → Prop := λ x y => a * x + 2 * y - 1 = 0
def l2 (b : ℝ) : ℝ → ℝ → Prop := λ x y => x + b * y - 3 = 0

-- Define the angle of inclination
def angle_of_inclination (l : ℝ → ℝ → Prop) : ℝ := sorry

-- Define perpendicularity
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

-- Define parallelism
def parallel (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

-- Define distance between parallel lines
def distance_between_parallel_lines (l1 l2 : ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statements
theorem value_of_a (a : ℝ) : 
  angle_of_inclination (l1 a) = π / 4 → a = -2 := by sorry

theorem value_of_b_when_perpendicular (b : ℝ) : 
  perpendicular (l1 (-2)) (l2 b) → b = 1 := by sorry

theorem distance_when_parallel (b : ℝ) : 
  parallel (l1 (-2)) (l2 b) → 
  distance_between_parallel_lines (l1 (-2)) (l2 b) = 7 * Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_value_of_a_value_of_b_when_perpendicular_distance_when_parallel_l3907_390753


namespace NUMINAMATH_CALUDE_oliver_stickers_l3907_390702

theorem oliver_stickers (initial_stickers : ℕ) (h1 : initial_stickers = 135) :
  let remaining_after_use := initial_stickers - (initial_stickers / 3)
  let given_away := remaining_after_use * 2 / 5
  let kept := remaining_after_use - given_away
  kept = 54 := by sorry

end NUMINAMATH_CALUDE_oliver_stickers_l3907_390702


namespace NUMINAMATH_CALUDE_runners_in_picture_probability_l3907_390757

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Calculates the probability of two runners being in a picture -/
def probabilityInPicture (jenny : Runner) (jack : Runner) : ℚ :=
  -- Define the probability calculation here
  23 / 60

/-- Theorem stating the probability of Jenny and Jack being in the picture -/
theorem runners_in_picture_probability :
  let jenny : Runner := { lapTime := 75, direction := true }
  let jack : Runner := { lapTime := 70, direction := false }
  let pictureTime : ℝ := 15 * 60  -- 15 minutes in seconds
  let pictureDuration : ℝ := 60  -- 1 minute in seconds
  let pictureTrackCoverage : ℝ := 1 / 3
  probabilityInPicture jenny jack = 23 / 60 := by
  sorry

#eval probabilityInPicture { lapTime := 75, direction := true } { lapTime := 70, direction := false }

end NUMINAMATH_CALUDE_runners_in_picture_probability_l3907_390757


namespace NUMINAMATH_CALUDE_average_age_is_35_l3907_390704

-- Define the ages of John, Mary, and Tonya
def john_age : ℕ := 30
def mary_age : ℕ := 15
def tonya_age : ℕ := 60

-- State the theorem
theorem average_age_is_35 :
  (john_age = 2 * mary_age) ∧  -- John is twice as old as Mary
  (2 * john_age = tonya_age) ∧  -- John is half as old as Tonya
  (tonya_age = 60) →  -- Tonya is 60 years old
  (john_age + mary_age + tonya_age) / 3 = 35 := by
  sorry

#check average_age_is_35

end NUMINAMATH_CALUDE_average_age_is_35_l3907_390704


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l3907_390748

theorem unique_solution_sqrt_equation :
  ∀ m n : ℕ+, 
    (m : ℝ)^2 = Real.sqrt (n : ℝ) + Real.sqrt ((2 * n + 1) : ℝ) → 
    m = 13 ∧ n = 4900 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l3907_390748


namespace NUMINAMATH_CALUDE_equal_cheese_division_l3907_390794

/-- Represents an equilateral triangle cheese -/
structure EquilateralTriangleCheese where
  side_length : ℝ
  area : ℝ

/-- Represents a division of the cheese -/
structure CheeseDivision where
  num_pieces : ℕ
  piece_area : ℝ

/-- The number of people to divide the cheese among -/
def num_people : ℕ := 5

theorem equal_cheese_division 
  (cheese : EquilateralTriangleCheese) 
  (division : CheeseDivision) :
  division.num_pieces = 25 ∧
  division.piece_area * division.num_pieces = cheese.area ∧
  division.num_pieces % num_people = 0 →
  ∃ (pieces_per_person : ℕ), 
    pieces_per_person * num_people = division.num_pieces ∧
    pieces_per_person = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_cheese_division_l3907_390794


namespace NUMINAMATH_CALUDE_insurance_problem_l3907_390799

/-- Number of policyholders -/
def n : ℕ := 10000

/-- Claim payment amount in yuan -/
def claim_payment : ℕ := 10000

/-- Operational cost in yuan -/
def operational_cost : ℕ := 50000

/-- Probability of the company paying at least one claim -/
def prob_at_least_one_claim : ℝ := 1 - 0.999^n

/-- Probability of a single policyholder making a claim -/
def p : ℝ := 0.001

/-- Minimum premium that ensures non-negative expected profit -/
def min_premium : ℝ := 15

theorem insurance_problem (a : ℝ) :
  (1 - (1 - p)^n = prob_at_least_one_claim) ∧
  (a ≥ min_premium ↔ n * a - n * p * claim_payment - operational_cost ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_insurance_problem_l3907_390799


namespace NUMINAMATH_CALUDE_bills_final_money_is_411_l3907_390720

/-- Calculates the final amount of money Bill has after all transactions and expenses. -/
def bills_final_money : ℝ :=
  let merchant_a_sale := 8 * 9
  let merchant_b_sale := 15 * 11
  let merchant_c_sale := 25 * 8
  let passerby_sale := 12 * 7
  let total_income := merchant_a_sale + merchant_b_sale + merchant_c_sale + passerby_sale
  let fine := 80
  let protection_cost := 30
  let total_expenses := fine + protection_cost
  total_income - total_expenses

/-- Theorem stating that Bill's final amount of money is $411. -/
theorem bills_final_money_is_411 : bills_final_money = 411 := by
  sorry

end NUMINAMATH_CALUDE_bills_final_money_is_411_l3907_390720


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_sides_4_and_9_l3907_390755

/-- An isosceles triangle with side lengths a, b, and c, where at least two sides are equal. -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  isosceles : (a = b) ∨ (b = c) ∨ (a = c)
  triangle_inequality : a + b > c ∧ b + c > a ∧ a + c > b

/-- The theorem stating that in an isosceles triangle with two sides of lengths 4 and 9, the third side must be 9. -/
theorem isosceles_triangle_with_sides_4_and_9 :
  ∀ (t : IsoscelesTriangle), (t.a = 4 ∧ t.b = 9) ∨ (t.a = 9 ∧ t.b = 4) → t.c = 9 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_with_sides_4_and_9_l3907_390755


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_perpendicular_l3907_390712

/-- Given a hyperbola 4x^2 - y^2 = 1, the value of t for which one of its asymptotes
    is perpendicular to the line tx + y + 1 = 0 is ±1/2 -/
theorem hyperbola_asymptote_perpendicular (x y t : ℝ) : 
  (4 * x^2 - y^2 = 1) → 
  (∃ (m : ℝ), (y = m * x ∨ y = -m * x) ∧ 
              (m * (-1/t) = -1 ∨ (-m) * (-1/t) = -1)) → 
  (t = 1/2 ∨ t = -1/2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_perpendicular_l3907_390712


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3907_390790

-- Define the rectangle's dimensions
variable (l w : ℝ)

-- Define the conditions
def condition1 : Prop := (l + 3) * (w - 1) = l * w
def condition2 : Prop := (l - 1.5) * (w + 2) = l * w

-- State the theorem
theorem rectangle_area_theorem (h1 : condition1 l w) (h2 : condition2 l w) : l * w = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3907_390790


namespace NUMINAMATH_CALUDE_ab_value_l3907_390781

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧
    Real.sqrt (log a) = m ∧
    Real.sqrt (log b) = n ∧
    log (Real.sqrt a) = m^2 / 2 ∧
    log (Real.sqrt b) = n^2 / 2 ∧
    m + n + m^2 / 2 + n^2 / 2 = 100) →
  a * b = 10^164 := by
sorry

end NUMINAMATH_CALUDE_ab_value_l3907_390781


namespace NUMINAMATH_CALUDE_officer_assignment_count_l3907_390764

def group_members : Nat := 4
def officer_positions : Nat := 3

theorem officer_assignment_count : 
  group_members ^ officer_positions = 64 := by
  sorry

end NUMINAMATH_CALUDE_officer_assignment_count_l3907_390764


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l3907_390725

theorem choose_four_from_ten (n : ℕ) (k : ℕ) : n = 10 ∧ k = 4 → Nat.choose n k = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l3907_390725


namespace NUMINAMATH_CALUDE_log_range_incorrect_l3907_390763

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_range_incorrect (b : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : b > 1) 
  (h2 : y = log b x) 
  (h3 : Real.sqrt b < x) 
  (h4 : x < b) : 
  ¬ (0.5 < y ∧ y < 1.5) :=
sorry

end NUMINAMATH_CALUDE_log_range_incorrect_l3907_390763


namespace NUMINAMATH_CALUDE_product_sum_theorem_l3907_390729

theorem product_sum_theorem (p q r s t : ℤ) :
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t →
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120 →
  p + q + r + s + t = 35 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l3907_390729


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3907_390791

theorem complex_fraction_simplification :
  (1 + 3 * Complex.I) / (1 + Complex.I) = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3907_390791


namespace NUMINAMATH_CALUDE_isosceles_if_root_one_equilateral_roots_l3907_390788

/-- Triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- The quadratic equation associated with the triangle -/
def quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.b) * x^2 - 2 * t.c * x + (t.a - t.b)

theorem isosceles_if_root_one (t : Triangle) :
  quadratic t 1 = 0 → t.a = t.c :=
sorry

theorem equilateral_roots (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (quadratic t 0 = 0 ∧ quadratic t 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_isosceles_if_root_one_equilateral_roots_l3907_390788


namespace NUMINAMATH_CALUDE_jenny_lasagna_profit_l3907_390707

/-- Calculates the profit for Jenny's lasagna business -/
def lasagna_profit (cost_per_pan : ℝ) (num_pans : ℕ) (price_per_pan : ℝ) : ℝ :=
  num_pans * price_per_pan - num_pans * cost_per_pan

/-- Proves that Jenny's profit is $300.00 given the specified conditions -/
theorem jenny_lasagna_profit :
  lasagna_profit 10 20 25 = 300 :=
by sorry

end NUMINAMATH_CALUDE_jenny_lasagna_profit_l3907_390707


namespace NUMINAMATH_CALUDE_alien_eggs_conversion_l3907_390713

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- The number of eggs laid by the alien creature in base 7 -/
def alienEggsBase7 : ℕ := 215

theorem alien_eggs_conversion :
  base7ToBase10 alienEggsBase7 = 110 := by
  sorry

end NUMINAMATH_CALUDE_alien_eggs_conversion_l3907_390713


namespace NUMINAMATH_CALUDE_multiplication_of_powers_l3907_390710

theorem multiplication_of_powers (a : ℝ) : 4 * (a^2) * (a^3) = 4 * (a^5) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_powers_l3907_390710


namespace NUMINAMATH_CALUDE_total_hockey_games_l3907_390705

/-- The number of hockey games in a season -/
def hockey_games_in_season (games_per_month : ℕ) (months_in_season : ℕ) : ℕ :=
  games_per_month * months_in_season

/-- Theorem stating that there are 182 hockey games in the season -/
theorem total_hockey_games :
  hockey_games_in_season 13 14 = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_hockey_games_l3907_390705


namespace NUMINAMATH_CALUDE_eliminate_denominators_l3907_390754

theorem eliminate_denominators (x : ℝ) : 
  (2*x - 3) / 5 = 2*x / 3 - 3 ↔ 3*(2*x - 3) = 5*(2*x) - 3*15 := by
  sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l3907_390754


namespace NUMINAMATH_CALUDE_study_supplies_cost_l3907_390787

/-- The cost of study supplies -/
theorem study_supplies_cost 
  (x y z : ℚ) -- x: cost of a pencil, y: cost of an exercise book, z: cost of a ballpoint pen
  (h1 : 3*x + 7*y + z = 3.15) -- First condition
  (h2 : 4*x + 10*y + z = 4.2) -- Second condition
  : x + y + z = 1.05 := by sorry

end NUMINAMATH_CALUDE_study_supplies_cost_l3907_390787


namespace NUMINAMATH_CALUDE_x_squared_inequality_l3907_390724

theorem x_squared_inequality (x : ℝ) (h : x^2 + x < 0) : x < x^2 ∧ x^2 < -x := by
  sorry

end NUMINAMATH_CALUDE_x_squared_inequality_l3907_390724


namespace NUMINAMATH_CALUDE_crabapple_theorem_l3907_390786

/-- The number of different sequences of crabapple recipients in a week for two classes -/
def crabapple_sequences (students1 : ℕ) (meetings1 : ℕ) (students2 : ℕ) (meetings2 : ℕ) : ℕ :=
  (students1 ^ meetings1) * (students2 ^ meetings2)

/-- Theorem stating the number of crabapple recipient sequences for the given classes -/
theorem crabapple_theorem :
  crabapple_sequences 12 3 9 2 = 139968 := by
  sorry

#eval crabapple_sequences 12 3 9 2

end NUMINAMATH_CALUDE_crabapple_theorem_l3907_390786


namespace NUMINAMATH_CALUDE_quadratic_radical_combination_l3907_390782

theorem quadratic_radical_combination (a : ℝ) : 
  (∃ k : ℝ, (k * Real.sqrt 2)^2 = a + 1) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_radical_combination_l3907_390782


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l3907_390735

/-- The number of grandchildren -/
def n : ℕ := 12

/-- The probability of a child being male (or female) -/
def p : ℚ := 1/2

/-- The probability of having an unequal number of males and females
    given n independently determined genders with probability p of being male -/
def unequal_probability (n : ℕ) (p : ℚ) : ℚ :=
  1 - (n.choose (n/2) : ℚ) * p^(n/2) * (1-p)^(n/2)

/-- The theorem to be proved -/
theorem unequal_gender_probability :
  unequal_probability n p = 793/1024 := by
  sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l3907_390735


namespace NUMINAMATH_CALUDE_marissa_boxes_tied_l3907_390776

/-- The number of boxes tied with a given amount of ribbon -/
def boxes_tied (total_ribbon leftover_ribbon ribbon_per_box : ℚ) : ℚ :=
  (total_ribbon - leftover_ribbon) / ribbon_per_box

/-- Theorem: Given the conditions, Marissa tied 5 boxes -/
theorem marissa_boxes_tied :
  boxes_tied 4.5 1 0.7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_marissa_boxes_tied_l3907_390776


namespace NUMINAMATH_CALUDE_scientific_notation_508_billion_yuan_l3907_390723

theorem scientific_notation_508_billion_yuan :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧
    508 * (10 ^ 9) = a * (10 ^ n) ∧
    a = 5.08 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_508_billion_yuan_l3907_390723


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l3907_390743

theorem opposite_of_negative_three :
  -((-3 : ℤ)) = 3 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l3907_390743


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3907_390741

theorem contrapositive_equivalence (f : ℝ → ℝ) (a : ℝ) :
  (a ≥ (1/2) → ∀ x ≥ 0, f x ≥ 0) ↔
  (∃ x ≥ 0, f x < 0 → a < (1/2)) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3907_390741
