import Mathlib

namespace NUMINAMATH_CALUDE_point_outside_circle_l3595_359588

/-- A point is outside a circle if its distance from the center is greater than the radius -/
def IsOutsideCircle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  Real.sqrt ((point.1 - center.1)^2 + (point.2 - center.2)^2) > radius

/-- Given a circle with radius 3 and a point at distance 5 from the center,
    prove that the point is outside the circle -/
theorem point_outside_circle (O : ℝ × ℝ) (A : ℝ × ℝ) 
    (h1 : Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) = 5) :
    IsOutsideCircle O 3 A := by
  sorry


end NUMINAMATH_CALUDE_point_outside_circle_l3595_359588


namespace NUMINAMATH_CALUDE_zoe_leo_difference_l3595_359550

-- Define variables
variable (t : ℝ) -- Leo's driving time
variable (s : ℝ) -- Leo's speed

-- Define Leo's distance
def leo_distance (t s : ℝ) : ℝ := t * s

-- Define Maria's distance
def maria_distance (t s : ℝ) : ℝ := (t + 2) * (s + 15)

-- Define Zoe's distance
def zoe_distance (t s : ℝ) : ℝ := (t + 3) * (s + 20)

-- Theorem statement
theorem zoe_leo_difference (t s : ℝ) :
  maria_distance t s = leo_distance t s + 110 →
  zoe_distance t s - leo_distance t s = 180 := by
  sorry


end NUMINAMATH_CALUDE_zoe_leo_difference_l3595_359550


namespace NUMINAMATH_CALUDE_equation_solution_l3595_359572

theorem equation_solution : ∃ x : ℚ, (x - 75) / 3 = (8 - 3*x) / 4 ∧ x = 324 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3595_359572


namespace NUMINAMATH_CALUDE_chord_sum_squares_l3595_359551

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 100}

def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry
def E : ℝ × ℝ := sorry

-- State the theorem
theorem chord_sum_squares (h1 : A ∈ Circle) (h2 : B ∈ Circle) (h3 : C ∈ Circle) (h4 : D ∈ Circle) (h5 : E ∈ Circle)
  (h6 : A.1 = -B.1 ∧ A.2 = -B.2) -- AB is a diameter
  (h7 : (E.1 - C.1) * (B.1 - A.1) + (E.2 - C.2) * (B.2 - A.2) = 0) -- CD intersects AB at E
  (h8 : (B.1 - E.1)^2 + (B.2 - E.2)^2 = 40) -- BE = 2√10
  (h9 : (A.1 - E.1) * (C.1 - E.1) + (A.2 - E.2) * (C.2 - E.2) = 
        Real.sqrt 3 * Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2) * Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2) / 2) -- Angle AEC = 30°
  : (C.1 - E.1)^2 + (C.2 - E.2)^2 + (D.1 - E.1)^2 + (D.2 - E.2)^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_chord_sum_squares_l3595_359551


namespace NUMINAMATH_CALUDE_billy_homework_ratio_l3595_359547

/-- Represents the number of questions solved in each hour -/
structure QuestionsSolved where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents Billy's homework solving session -/
def BillyHomework (qs : QuestionsSolved) : Prop :=
  qs.third = 132 ∧
  qs.third = 2 * qs.second ∧
  qs.first + qs.second + qs.third = 242

theorem billy_homework_ratio (qs : QuestionsSolved) 
  (h : BillyHomework qs) : qs.third / qs.first = 3 := by
  sorry

end NUMINAMATH_CALUDE_billy_homework_ratio_l3595_359547


namespace NUMINAMATH_CALUDE_first_half_speed_l3595_359574

/-- Proves that given a trip of 8 hours, where the second half is traveled at 85 km/h, 
    and the total distance is 620 km, the speed during the first half of the trip is 70 km/h. -/
theorem first_half_speed 
  (total_time : ℝ) 
  (second_half_speed : ℝ) 
  (total_distance : ℝ) 
  (h1 : total_time = 8) 
  (h2 : second_half_speed = 85) 
  (h3 : total_distance = 620) : 
  (total_distance - (second_half_speed * (total_time / 2))) / (total_time / 2) = 70 := by
sorry

end NUMINAMATH_CALUDE_first_half_speed_l3595_359574


namespace NUMINAMATH_CALUDE_initial_cells_count_l3595_359532

/-- Calculates the number of cells after one hour given the initial number -/
def cellsAfterOneHour (initialCells : ℕ) : ℕ :=
  2 * (initialCells - 2)

/-- Calculates the number of cells after n hours given the initial number -/
def cellsAfterNHours (initialCells n : ℕ) : ℕ :=
  match n with
  | 0 => initialCells
  | m + 1 => cellsAfterOneHour (cellsAfterNHours initialCells m)

/-- Theorem stating that if there are 164 cells after 5 hours, the initial number of cells was 9 -/
theorem initial_cells_count (initialCells : ℕ) :
  cellsAfterNHours initialCells 5 = 164 → initialCells = 9 :=
by
  sorry

#check initial_cells_count

end NUMINAMATH_CALUDE_initial_cells_count_l3595_359532


namespace NUMINAMATH_CALUDE_horner_method_for_f_l3595_359531

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 + 3x^3 + 5x - 4 -/
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_method_for_f :
  f 2 = horner [2, 3, 0, 5, -4] 2 ∧ horner [2, 3, 0, 5, -4] 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_for_f_l3595_359531


namespace NUMINAMATH_CALUDE_line_of_intersection_canonical_equation_l3595_359579

/-- Given two planes in 3D space, this theorem states that their line of intersection 
    can be represented by a specific canonical equation. -/
theorem line_of_intersection_canonical_equation 
  (plane1 : x + 5*y + 2*z = 5) 
  (plane2 : 2*x - 5*y - z = -5) :
  ∃ (t : ℝ), x = 5*t ∧ y = 5*t + 1 ∧ z = -15*t :=
by sorry

end NUMINAMATH_CALUDE_line_of_intersection_canonical_equation_l3595_359579


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_translation_l3595_359505

theorem fixed_point_of_exponential_translation (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a^(x-3) + 3
  f 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_translation_l3595_359505


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3595_359516

theorem sum_of_fractions : (1 : ℚ) / 6 + 5 / 12 = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3595_359516


namespace NUMINAMATH_CALUDE_intersection_points_parabola_and_circle_l3595_359563

theorem intersection_points_parabola_and_circle (A : ℝ) (h : A > 0) :
  ∃! (points : Finset (ℝ × ℝ)), points.card = 4 ∧
    ∀ (x y : ℝ), (x, y) ∈ points ↔ 
      (y = A * x^2 ∧ y^2 + 5 = x^2 + 6 * y) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_parabola_and_circle_l3595_359563


namespace NUMINAMATH_CALUDE_solve_for_Y_l3595_359533

theorem solve_for_Y : ∃ Y : ℤ, 80 - (Y - (6 + 2 * (7 - 8 - 5))) = 89 ∧ Y = -15 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_Y_l3595_359533


namespace NUMINAMATH_CALUDE_car_travel_distance_l3595_359556

theorem car_travel_distance (rate : ℚ) (time : ℚ) : 
  rate = 3 / 4 → time = 2 → rate * time * 60 = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l3595_359556


namespace NUMINAMATH_CALUDE_candy_bar_cost_l3595_359568

/-- The cost of a candy bar given initial and remaining amounts -/
theorem candy_bar_cost (initial_amount : ℚ) (remaining_amount : ℚ) 
  (h1 : initial_amount = 3)
  (h2 : remaining_amount = 2) :
  initial_amount - remaining_amount = 1 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l3595_359568


namespace NUMINAMATH_CALUDE_similar_triangles_height_l3595_359580

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  let h_large := h_small * Real.sqrt area_ratio
  h_small = 5 →
  h_large = 15 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l3595_359580


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3595_359582

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  ¬(((a > 0 ∧ b > 0) → (a * b < ((a + b) / 2)^2)) ∧
    ((a * b < ((a + b) / 2)^2) → (a > 0 ∧ b > 0))) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3595_359582


namespace NUMINAMATH_CALUDE_area_of_M_figure_l3595_359591

-- Define the set of points M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ α : ℝ, (p.1 - 3 * Real.cos α)^2 + (p.2 - 3 * Real.sin α)^2 = 25}

-- Define the area of the figure formed by all points in M
noncomputable def area_of_figure : ℝ := Real.pi * ((3 + 5)^2 - (5 - 3)^2)

-- Theorem statement
theorem area_of_M_figure : area_of_figure = 60 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_M_figure_l3595_359591


namespace NUMINAMATH_CALUDE_shooting_statistics_l3595_359597

def scores : List ℕ := [7, 5, 8, 9, 6, 6, 7, 7, 8, 7]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

def variance (l : List ℕ) : ℚ := sorry

theorem shooting_statistics :
  mode scores = 7 ∧
  median scores = 7 ∧
  mean scores = 7 ∧
  variance scores = 6/5 := by sorry

end NUMINAMATH_CALUDE_shooting_statistics_l3595_359597


namespace NUMINAMATH_CALUDE_regular_octahedron_has_six_vertices_l3595_359523

/-- A regular octahedron is a Platonic solid with equilateral triangular faces. -/
structure RegularOctahedron where
  -- We don't need to define the internal structure for this problem

/-- The number of vertices in a regular octahedron -/
def num_vertices (o : RegularOctahedron) : ℕ := 6

/-- Theorem: A regular octahedron has 6 vertices -/
theorem regular_octahedron_has_six_vertices (o : RegularOctahedron) : 
  num_vertices o = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_octahedron_has_six_vertices_l3595_359523


namespace NUMINAMATH_CALUDE_orange_count_l3595_359507

theorem orange_count (initial : ℕ) (thrown_away : ℕ) (added : ℕ) : 
  initial = 5 → thrown_away = 2 → added = 28 → 
  initial - thrown_away + added = 31 :=
by sorry

end NUMINAMATH_CALUDE_orange_count_l3595_359507


namespace NUMINAMATH_CALUDE_math_club_members_l3595_359519

theorem math_club_members (total_books : ℕ) (books_per_member : ℕ) (members_per_book : ℕ) :
  total_books = 12 →
  books_per_member = 2 →
  members_per_book = 3 →
  total_books * members_per_book = books_per_member * (total_books * members_per_book / books_per_member) :=
by sorry

end NUMINAMATH_CALUDE_math_club_members_l3595_359519


namespace NUMINAMATH_CALUDE_lions_in_first_group_l3595_359564

/-- The killing rate of lions in deers per minute -/
def killing_rate (lions : ℕ) (deers : ℕ) (minutes : ℕ) : ℚ :=
  (deers : ℚ) / (lions : ℚ) / (minutes : ℚ)

/-- The number of lions in the first group -/
def first_group_lions : ℕ := 10

theorem lions_in_first_group :
  (killing_rate first_group_lions 10 10 = killing_rate 100 100 10) →
  first_group_lions = 10 := by
  sorry

end NUMINAMATH_CALUDE_lions_in_first_group_l3595_359564


namespace NUMINAMATH_CALUDE_cake_mix_distribution_l3595_359594

theorem cake_mix_distribution (first_tray second_tray total : ℕ) : 
  first_tray = second_tray + 20 →
  first_tray + second_tray = 500 →
  second_tray = 240 := by
sorry

end NUMINAMATH_CALUDE_cake_mix_distribution_l3595_359594


namespace NUMINAMATH_CALUDE_soccer_club_girls_l3595_359555

theorem soccer_club_girls (total_members : ℕ) (attended_members : ℕ) 
  (h1 : total_members = 30)
  (h2 : attended_members = 18)
  (h3 : ∃ (boys girls : ℕ), boys + girls = total_members ∧ boys + girls / 3 = attended_members) :
  ∃ (girls : ℕ), girls = 18 ∧ ∃ (boys : ℕ), boys + girls = total_members := by
sorry

end NUMINAMATH_CALUDE_soccer_club_girls_l3595_359555


namespace NUMINAMATH_CALUDE_tangent_lines_to_C_value_of_m_l3595_359593

-- Define the curve C
def curve_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define point P
def point_P : ℝ × ℝ := (3, -1)

-- Define the intersecting line
def line_L (x y : ℝ) : Prop :=
  x + 2*y + 5 = 0

-- Part 1: Tangent lines
theorem tangent_lines_to_C :
  ∃ (k : ℝ), 
    (∀ x y : ℝ, curve_C 1 x y → (x = 3 ∨ 5*x + 12*y - 3 = 0) → 
      ((x - 3)^2 + (y + 1)^2 = k^2)) ∧
    (∀ x y : ℝ, (x = 3 ∨ 5*x + 12*y - 3 = 0) → 
      ((x - 3)^2 + (y + 1)^2 ≤ k^2)) :=
sorry

-- Part 2: Value of m
theorem value_of_m :
  ∃! m : ℝ, 
    (∃ x1 y1 x2 y2 : ℝ,
      curve_C m x1 y1 ∧ curve_C m x2 y2 ∧
      line_L x1 y1 ∧ line_L x2 y2 ∧
      (x1 - x2)^2 + (y1 - y2)^2 = 20) ∧
    m = -20 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_to_C_value_of_m_l3595_359593


namespace NUMINAMATH_CALUDE_oranges_packed_l3595_359543

/-- Given boxes that hold 10 oranges each and 265 boxes used, prove that the total number of oranges packed is 2650. -/
theorem oranges_packed (oranges_per_box : ℕ) (boxes_used : ℕ) (h1 : oranges_per_box = 10) (h2 : boxes_used = 265) :
  oranges_per_box * boxes_used = 2650 := by
  sorry

end NUMINAMATH_CALUDE_oranges_packed_l3595_359543


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l3595_359562

theorem triangle_area_ratio (k : ℕ) (H : ℝ) (h : ℝ) :
  k > 0 →
  H > 0 →
  h > 0 →
  h / H = 1 / Real.sqrt k →
  (h / H)^2 = 1 / k :=
by sorry


end NUMINAMATH_CALUDE_triangle_area_ratio_l3595_359562


namespace NUMINAMATH_CALUDE_polynomial_value_l3595_359549

theorem polynomial_value (x y : ℚ) (h : x - 2 * y - 3 = -5) : 2 * y - x = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3595_359549


namespace NUMINAMATH_CALUDE_price_crossover_year_l3595_359534

def price_X (year : ℕ) : ℚ :=
  5.20 + 0.45 * (year - 2001 : ℚ)

def price_Y (year : ℕ) : ℚ :=
  7.30 + 0.20 * (year - 2001 : ℚ)

theorem price_crossover_year :
  (∀ y : ℕ, y < 2010 → price_X y ≤ price_Y y) ∧
  price_X 2010 > price_Y 2010 := by
  sorry

end NUMINAMATH_CALUDE_price_crossover_year_l3595_359534


namespace NUMINAMATH_CALUDE_height_difference_after_three_years_l3595_359544

/-- Represents the seasons of the year -/
inductive Season
  | spring
  | summer
  | fall
  | winter

/-- Calculates the growth of an object over a season given its monthly growth rate -/
def seasonalGrowth (monthlyRate : ℕ) : ℕ := 3 * monthlyRate

/-- Calculates the total growth over a year given seasonal growth rates -/
def yearlyGrowth (spring summer fall winter : ℕ) : ℕ :=
  seasonalGrowth spring + seasonalGrowth summer + seasonalGrowth fall + seasonalGrowth winter

/-- Theorem: The height difference between the tree and the boy after 3 years is 73 inches -/
theorem height_difference_after_three_years :
  let initialTreeHeight : ℕ := 16
  let initialBoyHeight : ℕ := 24
  let treeGrowth : Season → ℕ
    | Season.spring => 4
    | Season.summer => 6
    | Season.fall => 2
    | Season.winter => 1
  let boyGrowth : Season → ℕ
    | Season.spring => 2
    | Season.summer => 2
    | Season.fall => 0
    | Season.winter => 0
  let treeYearlyGrowth := yearlyGrowth (treeGrowth Season.spring) (treeGrowth Season.summer) (treeGrowth Season.fall) (treeGrowth Season.winter)
  let boyYearlyGrowth := yearlyGrowth (boyGrowth Season.spring) (boyGrowth Season.summer) (boyGrowth Season.fall) (boyGrowth Season.winter)
  let finalTreeHeight := initialTreeHeight + 3 * treeYearlyGrowth
  let finalBoyHeight := initialBoyHeight + 3 * boyYearlyGrowth
  finalTreeHeight - finalBoyHeight = 73 := by
  sorry


end NUMINAMATH_CALUDE_height_difference_after_three_years_l3595_359544


namespace NUMINAMATH_CALUDE_no_rational_solution_l3595_359576

theorem no_rational_solution :
  ∀ (a b c d : ℚ), (a + b * Real.sqrt 3)^4 + (c + d * Real.sqrt 3)^4 ≠ 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_l3595_359576


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l3595_359571

theorem complex_magnitude_proof : Complex.abs (7/4 - 3*I) = Real.sqrt 193 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l3595_359571


namespace NUMINAMATH_CALUDE_hidden_primes_average_l3595_359584

-- Define the visible numbers on the cards
def visible_numbers : List Nat := [44, 59, 38]

-- Define a function to check if a number is prime
def is_prime (n : Nat) : Prop := Nat.Prime n

-- Define the property that the sum of numbers on each card is equal
def equal_sums (x y z : Nat) : Prop :=
  44 + x = 59 + y ∧ 59 + y = 38 + z

-- The main theorem
theorem hidden_primes_average (x y z : Nat) : 
  is_prime x ∧ is_prime y ∧ is_prime z ∧ 
  equal_sums x y z → 
  (x + y + z) / 3 = 14 :=
sorry

end NUMINAMATH_CALUDE_hidden_primes_average_l3595_359584


namespace NUMINAMATH_CALUDE_prism_36_edges_14_faces_l3595_359524

/-- A prism is a polyhedron with two congruent parallel faces (bases) and lateral faces that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  2 + p.edges / 3

/-- Theorem: A prism with 36 edges has 14 faces. -/
theorem prism_36_edges_14_faces (p : Prism) (h : p.edges = 36) : num_faces p = 14 := by
  sorry


end NUMINAMATH_CALUDE_prism_36_edges_14_faces_l3595_359524


namespace NUMINAMATH_CALUDE_floor_product_equals_twelve_l3595_359596

theorem floor_product_equals_twelve (x : ℝ) : 
  ⌊x * ⌊x / 2⌋⌋ = 12 ↔ x ≥ 4.9 ∧ x < 5.1 := by sorry

end NUMINAMATH_CALUDE_floor_product_equals_twelve_l3595_359596


namespace NUMINAMATH_CALUDE_integer_root_quadratic_count_l3595_359552

theorem integer_root_quadratic_count :
  ∃! (S : Finset ℝ), 
    Finset.card S = 8 ∧ 
    (∀ a ∈ S, ∃ r s : ℤ, 
      (∀ x : ℝ, x^2 + a*x + 12*a = 0 ↔ x = r ∨ x = s)) :=
sorry

end NUMINAMATH_CALUDE_integer_root_quadratic_count_l3595_359552


namespace NUMINAMATH_CALUDE_principal_is_2000_l3595_359545

/-- Calculates the simple interest for a given principal, rate, and time. -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (principal * rate * time) / 100

/-- Proves that the principal is 2000 given the conditions of the problem. -/
theorem principal_is_2000 (rate : ℚ) (time : ℚ) (interest : ℚ) 
    (h_rate : rate = 5)
    (h_time : time = 13)
    (h_interest : interest = 1300)
    (h_simple_interest : simpleInterest principal rate time = interest) :
  principal = 2000 := by
  sorry

#check principal_is_2000

end NUMINAMATH_CALUDE_principal_is_2000_l3595_359545


namespace NUMINAMATH_CALUDE_expression_simplification_l3595_359590

theorem expression_simplification (m n : ℝ) 
  (hm : m = (400 : ℝ) ^ (1/4))
  (hn : n = (5 : ℝ) ^ (1/2)) :
  ((2 - n) / (n - 1) + 4 * (m - 1) / (m - 2)) / 
  (n^2 * (m - 1) / (n - 1) + m^2 * (2 - n) / (m - 2)) = 
  ((5 : ℝ) ^ (1/2)) / 5 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3595_359590


namespace NUMINAMATH_CALUDE_fish_count_l3595_359567

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 8

/-- The number of fish Max has -/
def max_fish : ℕ := 15

/-- The total number of fish Lilly, Rosy, and Max have -/
def total_fish : ℕ := lilly_fish + rosy_fish + max_fish

theorem fish_count : total_fish = 33 := by sorry

end NUMINAMATH_CALUDE_fish_count_l3595_359567


namespace NUMINAMATH_CALUDE_probability_red_or_white_l3595_359539

/-- Probability of selecting a red or white marble from a bag -/
theorem probability_red_or_white (total : ℕ) (blue : ℕ) (red : ℕ) :
  total = 20 →
  blue = 5 →
  red = 9 →
  (red + (total - blue - red)) / total = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_white_l3595_359539


namespace NUMINAMATH_CALUDE_d2_equals_18_l3595_359511

/-- Definition of E(m) -/
def E (m : ℕ) : ℕ :=
  sorry

/-- The polynomial r(x) -/
def r (x : ℕ) : ℕ :=
  sorry

/-- Theorem stating that d₂ = 18 in the polynomial r(x) that satisfies E(m) = r(m) -/
theorem d2_equals_18 :
  ∃ (d₄ d₃ d₂ d₁ d₀ : ℤ),
    (∀ m : ℕ, m ≥ 7 → Odd m → E m = d₄ * m^4 + d₃ * m^3 + d₂ * m^2 + d₁ * m + d₀) →
    d₂ = 18 :=
  sorry

end NUMINAMATH_CALUDE_d2_equals_18_l3595_359511


namespace NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l3595_359508

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  n_pos : n > 0

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_18gon_symmetry_sum :
  ∀ (p : RegularPolygon 18),
  (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by sorry

end NUMINAMATH_CALUDE_regular_18gon_symmetry_sum_l3595_359508


namespace NUMINAMATH_CALUDE_jake_weight_loss_l3595_359554

/-- Given that Jake and his sister together weigh 156 pounds and Jake's current weight is 108 pounds,
    this theorem proves that Jake needs to lose 12 pounds to weigh twice as much as his sister. -/
theorem jake_weight_loss (total_weight : ℕ) (jake_weight : ℕ) 
  (h1 : total_weight = 156)
  (h2 : jake_weight = 108) :
  jake_weight - (2 * (total_weight - jake_weight)) = 12 := by
  sorry

#check jake_weight_loss

end NUMINAMATH_CALUDE_jake_weight_loss_l3595_359554


namespace NUMINAMATH_CALUDE_product_sum_6936_l3595_359530

theorem product_sum_6936 : ∃ a b : ℕ, 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 6936 ∧ 
  a + b = 168 := by
sorry

end NUMINAMATH_CALUDE_product_sum_6936_l3595_359530


namespace NUMINAMATH_CALUDE_quadratic_from_means_l3595_359538

theorem quadratic_from_means (α β : ℝ) : 
  (α + β) / 2 = 8 → 
  (α * β) = 15^2 → 
  ∀ x, x^2 - 16*x + 225 = 0 ↔ (x = α ∨ x = β) := by
sorry

end NUMINAMATH_CALUDE_quadratic_from_means_l3595_359538


namespace NUMINAMATH_CALUDE_negation_equivalence_l3595_359535

theorem negation_equivalence (a b x : ℝ) : 
  ¬(x ≠ a ∧ x ≠ b → x^2 - (a+b)*x + a*b ≠ 0) ↔ 
  (x = a ∨ x = b → x^2 - (a+b)*x + a*b = 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3595_359535


namespace NUMINAMATH_CALUDE_cos_18_degrees_l3595_359520

theorem cos_18_degrees : Real.cos (18 * π / 180) = (1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_18_degrees_l3595_359520


namespace NUMINAMATH_CALUDE_triangle_property_l3595_359525

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, cos A = 1/2 and the area is (3√3)/4 -/
theorem triangle_property (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- Positive angles
  A + B + C = π →  -- Angle sum in a triangle
  c * Real.cos A + a * Real.cos C = 2 * b * Real.cos A →  -- Given condition
  a = Real.sqrt 7 →  -- Given condition
  b + c = 4 →  -- Given condition
  Real.cos A = 1 / 2 ∧  -- Conclusion 1
  (1 / 2) * b * c * Real.sqrt (1 - (Real.cos A)^2) = (3 * Real.sqrt 3) / 4  -- Conclusion 2 (area)
  := by sorry

end NUMINAMATH_CALUDE_triangle_property_l3595_359525


namespace NUMINAMATH_CALUDE_roots_of_derivative_in_triangle_l3595_359587

open Complex

-- Define the polynomial f(x) = (x-a)(x-b)(x-c)
def f (x a b c : ℂ) : ℂ := (x - a) * (x - b) * (x - c)

-- Define the derivative of f
def f_derivative (x a b c : ℂ) : ℂ := 
  (x - b) * (x - c) + (x - a) * (x - c) + (x - a) * (x - b)

-- Define a triangle in the complex plane
def triangle_contains (a b c z : ℂ) : Prop :=
  ∃ (t1 t2 t3 : ℝ), t1 ≥ 0 ∧ t2 ≥ 0 ∧ t3 ≥ 0 ∧ t1 + t2 + t3 = 1 ∧
    z = t1 • a + t2 • b + t3 • c

-- Theorem statement
theorem roots_of_derivative_in_triangle (a b c : ℂ) :
  ∀ z : ℂ, f_derivative z a b c = 0 → triangle_contains a b c z :=
sorry

end NUMINAMATH_CALUDE_roots_of_derivative_in_triangle_l3595_359587


namespace NUMINAMATH_CALUDE_excavator_transport_theorem_l3595_359583

/-- Represents the transportation problem for excavators after an earthquake. -/
structure ExcavatorTransport where
  area_a_need : ℕ := 27
  area_b_need : ℕ := 25
  province_a_donate : ℕ := 28
  province_b_donate : ℕ := 24
  cost_a_to_a : ℚ := 0.4
  cost_a_to_b : ℚ := 0.3
  cost_b_to_a : ℚ := 0.5
  cost_b_to_b : ℚ := 0.2

/-- The functional relationship between total cost y and number of excavators x
    transported from Province A to Area A. -/
def total_cost (et : ExcavatorTransport) (x : ℕ) : ℚ :=
  et.cost_a_to_a * x + et.cost_a_to_b * (et.province_a_donate - x) +
  et.cost_b_to_a * (et.area_a_need - x) + et.cost_b_to_b * (x - 3)

/-- The theorem stating the functional relationship and range of x. -/
theorem excavator_transport_theorem (et : ExcavatorTransport) :
  ∀ x : ℕ, 3 ≤ x ∧ x ≤ 27 →
    total_cost et x = -0.2 * x + 21.3 ∧
    (∀ y : ℚ, y = total_cost et x → -0.2 * x + 21.3 = y) := by
  sorry

#check excavator_transport_theorem

end NUMINAMATH_CALUDE_excavator_transport_theorem_l3595_359583


namespace NUMINAMATH_CALUDE_rhombus_longest_diagonal_l3595_359537

/-- Given a rhombus with area 144 and diagonal ratio 4:3, prove its longest diagonal is 8√6 -/
theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℝ) (long_diag : ℝ) : 
  area = 144 → ratio = 4/3 → long_diag = 8 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longest_diagonal_l3595_359537


namespace NUMINAMATH_CALUDE_ratio_equality_l3595_359548

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (norm_abc : a^2 + b^2 + c^2 = 25)
  (norm_xyz : x^2 + y^2 + z^2 = 36)
  (dot_product : a*x + b*y + c*z = 30) :
  (a + b + c) / (x + y + z) = 5/6 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l3595_359548


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l3595_359506

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0
def line2 (x y : ℝ) (b c : ℝ) : Prop := 6 * x + b * y + c = 0

-- Define the distance between lines
def distance_between_lines (b c : ℝ) : ℝ := 3

-- Define the parallelism condition
def parallel_lines (b : ℝ) : Prop := b = 8

theorem parallel_lines_distance (b c : ℝ) :
  parallel_lines b → distance_between_lines b c = 3 →
  (b + c = -12 ∨ b + c = 48) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l3595_359506


namespace NUMINAMATH_CALUDE_base_subtraction_proof_l3595_359557

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

theorem base_subtraction_proof :
  let base5_num := [4, 2, 3]  -- 324 in base 5 (least significant digit first)
  let base6_num := [3, 5, 1]  -- 153 in base 6 (least significant digit first)
  toBase10 base5_num 5 - toBase10 base6_num 6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_proof_l3595_359557


namespace NUMINAMATH_CALUDE_ball_difference_l3595_359586

def soccer_boxes : ℕ := 8
def basketball_boxes : ℕ := 5
def balls_per_box : ℕ := 12

theorem ball_difference : 
  soccer_boxes * balls_per_box - basketball_boxes * balls_per_box = 36 := by
  sorry

end NUMINAMATH_CALUDE_ball_difference_l3595_359586


namespace NUMINAMATH_CALUDE_cafeteria_vertical_stripes_l3595_359522

def cafeteria_problem (total : ℕ) (checkered : ℕ) (horizontal_multiplier : ℕ) : Prop :=
  let stripes : ℕ := total - checkered
  let horizontal : ℕ := horizontal_multiplier * checkered
  let vertical : ℕ := stripes - horizontal
  vertical = 5

theorem cafeteria_vertical_stripes :
  cafeteria_problem 40 7 4 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_vertical_stripes_l3595_359522


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l3595_359540

theorem complete_square_quadratic :
  ∀ x : ℝ, x^2 - 4*x - 6 = 0 ↔ (x - 2)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l3595_359540


namespace NUMINAMATH_CALUDE_worksheets_graded_l3595_359565

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) : 
  total_worksheets = 15 →
  problems_per_worksheet = 3 →
  problems_left = 24 →
  (total_worksheets * problems_per_worksheet - problems_left) / problems_per_worksheet = 7 := by
sorry

end NUMINAMATH_CALUDE_worksheets_graded_l3595_359565


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3595_359595

/-- An isosceles triangle with two sides of lengths 3 and 6 has a perimeter of 15. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 6 ∧ c = 6 →  -- Two sides are 6, one side is 3
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  a + b + c = 15 :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3595_359595


namespace NUMINAMATH_CALUDE_stratified_sampling_suitable_l3595_359504

/-- Represents a sampling method -/
inductive SamplingMethod
  | DrawingLots
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a school population -/
structure SchoolPopulation where
  total_students : Nat
  boys : Nat
  girls : Nat
  sample_size : Nat

/-- Determines if a sampling method is suitable for a given school population -/
def is_suitable_sampling_method (population : SchoolPopulation) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Stratified ∧
  population.total_students = population.boys + population.girls ∧
  population.sample_size < population.total_students

/-- Theorem stating that stratified sampling is suitable for the given school population -/
theorem stratified_sampling_suitable (population : SchoolPopulation) 
  (h1 : population.total_students = 1000)
  (h2 : population.boys = 520)
  (h3 : population.girls = 480)
  (h4 : population.sample_size = 100) :
  is_suitable_sampling_method population SamplingMethod.Stratified :=
by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_suitable_l3595_359504


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l3595_359585

-- Define the proposition
def P (x m : ℝ) : Prop := x / 2 + 1 / (2 * x) - 3 / 2 > m

-- Define the condition
def condition (m : ℝ) : Prop := ∀ x > 0, P x m

-- Define necessary condition
def necessary (m : ℝ) : Prop := condition m → m ≤ -1/2

-- Define sufficient condition
def sufficient (m : ℝ) : Prop := m ≤ -1/2 → condition m

-- Theorem statement
theorem necessary_not_sufficient :
  (∀ m : ℝ, necessary m) ∧ (∃ m : ℝ, ¬ sufficient m) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l3595_359585


namespace NUMINAMATH_CALUDE_genuine_purses_and_handbags_l3595_359570

theorem genuine_purses_and_handbags 
  (total_purses : ℕ) 
  (total_handbags : ℕ) 
  (fake_purses_ratio : ℚ) 
  (fake_handbags_ratio : ℚ) 
  (h1 : total_purses = 26) 
  (h2 : total_handbags = 24) 
  (h3 : fake_purses_ratio = 1/2) 
  (h4 : fake_handbags_ratio = 1/4) :
  (total_purses - total_purses * fake_purses_ratio) + 
  (total_handbags - total_handbags * fake_handbags_ratio) = 31 := by
sorry

end NUMINAMATH_CALUDE_genuine_purses_and_handbags_l3595_359570


namespace NUMINAMATH_CALUDE_students_liking_sports_l3595_359581

theorem students_liking_sports (total : ℕ) (basketball cricket soccer : ℕ)
  (basketball_cricket basketball_soccer cricket_soccer : ℕ) (all_three : ℕ)
  (h1 : total = 30)
  (h2 : basketball = 12)
  (h3 : cricket = 10)
  (h4 : soccer = 8)
  (h5 : basketball_cricket = 4)
  (h6 : basketball_soccer = 3)
  (h7 : cricket_soccer = 2)
  (h8 : all_three = 1) :
  basketball + cricket + soccer - (basketball_cricket + basketball_soccer + cricket_soccer) + all_three = 22 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_sports_l3595_359581


namespace NUMINAMATH_CALUDE_expression_equality_l3595_359513

theorem expression_equality (x : ℝ) : 
  (Real.sqrt x / Real.sqrt 0.81 + Real.sqrt 0.81 / Real.sqrt 0.49 = 2.507936507936508) → 
  x = 1.21 := by
sorry

end NUMINAMATH_CALUDE_expression_equality_l3595_359513


namespace NUMINAMATH_CALUDE_todds_initial_gum_l3595_359558

theorem todds_initial_gum (x : ℕ) : x + 16 = 54 → x = 38 := by
  sorry

end NUMINAMATH_CALUDE_todds_initial_gum_l3595_359558


namespace NUMINAMATH_CALUDE_tuna_price_is_two_l3595_359503

/-- Represents the daily catch and earnings of a fisherman -/
structure FishermanData where
  red_snappers : ℕ
  tunas : ℕ
  red_snapper_price : ℚ
  daily_earnings : ℚ

/-- Calculates the price of a Tuna given the fisherman's data -/
def tuna_price (data : FishermanData) : ℚ :=
  (data.daily_earnings - data.red_snappers * data.red_snapper_price) / data.tunas

/-- Theorem stating that the price of a Tuna is $2 given the fisherman's data -/
theorem tuna_price_is_two (data : FishermanData)
  (h1 : data.red_snappers = 8)
  (h2 : data.tunas = 14)
  (h3 : data.red_snapper_price = 3)
  (h4 : data.daily_earnings = 52) :
  tuna_price data = 2 := by
  sorry

end NUMINAMATH_CALUDE_tuna_price_is_two_l3595_359503


namespace NUMINAMATH_CALUDE_antons_offer_is_cheapest_l3595_359502

/-- Represents a shareholder in the company -/
structure Shareholder where
  name : String
  shares : Nat
  yield : Rat

/-- Represents the company and its shareholders -/
structure Company where
  totalShares : Nat
  sharePrice : Nat
  shareholders : List Shareholder

def Company.largestShareholderShares (c : Company) : Nat :=
  c.shareholders.map (·.shares) |>.maximum?.getD 0

def buySharesCost (sharePrice : Nat) (shares : Nat) (yield : Rat) : Nat :=
  Nat.ceil (sharePrice * shares * (1 + yield))

theorem antons_offer_is_cheapest (c : Company) (arina : Shareholder) : 
  c.totalShares = 300000 ∧
  c.sharePrice = 10 ∧
  arina.shares = 90001 ∧
  c.shareholders = [
    ⟨"Maxim", 104999, 1/10⟩,
    ⟨"Inga", 30000, 1/4⟩,
    ⟨"Yuri", 30000, 3/20⟩,
    ⟨"Yulia", 30000, 3/10⟩,
    ⟨"Anton", 15000, 2/5⟩
  ] →
  let requiredShares := c.largestShareholderShares - arina.shares + 1
  let antonsCost := buySharesCost c.sharePrice (c.shareholders.find? (·.name = "Anton") |>.map (·.shares) |>.getD 0) (2/5)
  ∀ s ∈ c.shareholders, s.name ≠ "Anton" →
    buySharesCost c.sharePrice s.shares s.yield ≥ antonsCost ∧
    s.shares ≥ requiredShares →
    antonsCost ≤ buySharesCost c.sharePrice s.shares s.yield :=
by sorry


end NUMINAMATH_CALUDE_antons_offer_is_cheapest_l3595_359502


namespace NUMINAMATH_CALUDE_ones_digit_of_31_power_l3595_359599

theorem ones_digit_of_31_power (n : ℕ) : (31^(15 * 7^7) : ℕ) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_31_power_l3595_359599


namespace NUMINAMATH_CALUDE_largest_angle_of_inclination_l3595_359566

-- Define the angle of inclination for a line given its slope
noncomputable def angle_of_inclination (slope : ℝ) : ℝ :=
  Real.arctan slope * (180 / Real.pi)

-- Define the lines
def line_A : ℝ → ℝ := λ x => -x + 1
def line_B : ℝ → ℝ := λ x => x + 1
def line_C : ℝ → ℝ := λ x => 2*x + 1
def line_D : ℝ → ℝ := λ _ => 1

-- Theorem statement
theorem largest_angle_of_inclination :
  let angle_A := angle_of_inclination (-1)
  let angle_B := angle_of_inclination 1
  let angle_C := angle_of_inclination 2
  let angle_D := 90
  angle_A > angle_B ∧ angle_A > angle_C ∧ angle_A > angle_D :=
by sorry


end NUMINAMATH_CALUDE_largest_angle_of_inclination_l3595_359566


namespace NUMINAMATH_CALUDE_sqrt_cube_root_problem_l3595_359527

theorem sqrt_cube_root_problem (x y : ℝ) : 
  y = Real.sqrt (x - 24) + Real.sqrt (24 - x) - 8 → 
  (x - 5 * y)^(1/3 : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_cube_root_problem_l3595_359527


namespace NUMINAMATH_CALUDE_mary_seth_age_ratio_l3595_359575

/-- Given that Mary is 9 years older than Seth and Seth is currently 3.5 years old,
    prove that the ratio of Mary's age to Seth's age in a year is 3:1. -/
theorem mary_seth_age_ratio :
  ∀ (seth_age mary_age seth_future_age mary_future_age : ℝ),
  seth_age = 3.5 →
  mary_age = seth_age + 9 →
  seth_future_age = seth_age + 1 →
  mary_future_age = mary_age + 1 →
  mary_future_age / seth_future_age = 3 := by
sorry

end NUMINAMATH_CALUDE_mary_seth_age_ratio_l3595_359575


namespace NUMINAMATH_CALUDE_even_quadratic_implies_k_equals_one_l3595_359500

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- The quadratic function f(x) = kx^2 + (k-1)x + 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 3

theorem even_quadratic_implies_k_equals_one :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_implies_k_equals_one_l3595_359500


namespace NUMINAMATH_CALUDE_total_marbles_is_172_l3595_359560

/-- Represents the number of marbles of each color in a bag -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  purple : ℕ

/-- Checks if the given MarbleBag satisfies the ratio conditions -/
def satisfiesRatios (bag : MarbleBag) : Prop :=
  7 * bag.red = 4 * bag.blue ∧ 3 * bag.blue = 2 * bag.purple

/-- Theorem: Given the conditions, the total number of marbles is 172 -/
theorem total_marbles_is_172 (bag : MarbleBag) 
  (h1 : satisfiesRatios bag) 
  (h2 : bag.red = 32) : 
  bag.red + bag.blue + bag.purple = 172 := by
  sorry

#check total_marbles_is_172

end NUMINAMATH_CALUDE_total_marbles_is_172_l3595_359560


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3595_359515

theorem perfect_square_condition (n : ℕ+) : 
  (∃ m : ℕ, n^4 - 4*n^3 + 22*n^2 - 36*n + 18 = m^2) ↔ (n = 1 ∨ n = 3) :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3595_359515


namespace NUMINAMATH_CALUDE_remaining_money_l3595_359578

def initialAmount : ℚ := 7.10
def spentOnSweets : ℚ := 1.05
def givenToFriend : ℚ := 1.00
def numberOfFriends : ℕ := 2

theorem remaining_money :
  initialAmount - (spentOnSweets + givenToFriend * numberOfFriends) = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l3595_359578


namespace NUMINAMATH_CALUDE_survey_analysis_l3595_359559

-- Define the survey data
structure SurveyData where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  male_like : ℕ
  female_like : ℕ
  male_dislike : ℕ
  female_dislike : ℕ

-- Define the theorem
theorem survey_analysis (data : SurveyData) 
  (h1 : data.total_students = 400 + data.female_like + data.male_dislike)
  (h2 : data.male_students = 280 + data.male_dislike)
  (h3 : data.female_students = 120 + data.female_like)
  (h4 : data.male_students = (4 : ℚ) / 7 * data.total_students)
  (h5 : data.female_like = (3 : ℚ) / 5 * data.female_students)
  (h6 : data.male_like = 280)
  (h7 : data.female_dislike = 120) :
  data.female_like = 180 ∧ 
  data.male_dislike = 120 ∧ 
  ((700 : ℚ) * (280 * 120 - 180 * 120)^2 / (460 * 240 * 400 * 300) < (10828 : ℚ) / 1000) :=
sorry


end NUMINAMATH_CALUDE_survey_analysis_l3595_359559


namespace NUMINAMATH_CALUDE_sum_of_cubes_equation_l3595_359561

theorem sum_of_cubes_equation (x y : ℝ) :
  x^3 + 21*x*y + y^3 = 343 → x + y = 7 ∨ x + y = -14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equation_l3595_359561


namespace NUMINAMATH_CALUDE_problem_statement_l3595_359553

theorem problem_statement (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = -7 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3595_359553


namespace NUMINAMATH_CALUDE_minimum_a_value_l3595_359589

theorem minimum_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 1 → |x| < a) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_minimum_a_value_l3595_359589


namespace NUMINAMATH_CALUDE_sector_central_angle_l3595_359569

/-- Given a sector with radius 10 cm and area 100 cm², prove that the central angle is 2 radians. -/
theorem sector_central_angle (r : ℝ) (S : ℝ) (α : ℝ) :
  r = 10 →
  S = 100 →
  S = (1 / 2) * α * r^2 →
  α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3595_359569


namespace NUMINAMATH_CALUDE_richard_david_age_difference_l3595_359509

/-- The ages of Richard, David, and Scott in a family -/
structure FamilyAges where
  R : ℕ  -- Richard's age
  D : ℕ  -- David's age
  S : ℕ  -- Scott's age

/-- The conditions of the family ages problem -/
def FamilyAgesProblem (ages : FamilyAges) : Prop :=
  ages.R > ages.D ∧                 -- Richard is older than David
  ages.D = ages.S + 8 ∧             -- David is 8 years older than Scott
  ages.R + 8 = 2 * (ages.S + 8) ∧   -- In 8 years, Richard will be twice as old as Scott
  ages.D = 14                       -- David was 9 years old 5 years ago

/-- The theorem stating that Richard is 6 years older than David -/
theorem richard_david_age_difference (ages : FamilyAges) 
  (h : FamilyAgesProblem ages) : ages.R = ages.D + 6 := by
  sorry


end NUMINAMATH_CALUDE_richard_david_age_difference_l3595_359509


namespace NUMINAMATH_CALUDE_matrix_power_vector_product_l3595_359514

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; -1, 4]
def a : Matrix (Fin 2) (Fin 1) ℝ := !![7; 4]

theorem matrix_power_vector_product :
  A^6 * a = !![435; 339] := by sorry

end NUMINAMATH_CALUDE_matrix_power_vector_product_l3595_359514


namespace NUMINAMATH_CALUDE_school_cleanup_participants_l3595_359577

/-- The expected number of participants after n years, given an initial number and annual increase rate -/
def expected_participants (initial : ℕ) (rate : ℚ) (years : ℕ) : ℚ :=
  initial * (1 + rate) ^ years

theorem school_cleanup_participants : expected_participants 1000 (60/100) 3 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_school_cleanup_participants_l3595_359577


namespace NUMINAMATH_CALUDE_symmetric_point_of_P_l3595_359518

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The symmetric point of a given point with respect to the x-axis. -/
def symmetricPointXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Theorem: The symmetric point of P(2, -5) with respect to the x-axis is (2, 5). -/
theorem symmetric_point_of_P : 
  let P : Point := { x := 2, y := -5 }
  symmetricPointXAxis P = { x := 2, y := 5 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_of_P_l3595_359518


namespace NUMINAMATH_CALUDE_inequality_addition_l3595_359512

theorem inequality_addition (a b c : ℝ) : a > b → a + c > b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l3595_359512


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3595_359526

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 36 →
  b = 48 →
  c^2 = a^2 + b^2 →
  c = 60 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3595_359526


namespace NUMINAMATH_CALUDE_solve_board_game_problem_l3595_359501

def board_game_problem (cost_per_game : ℕ) (payment : ℕ) (change_bills : ℕ) (change_denomination : ℕ) : Prop :=
  let total_change : ℕ := change_bills * change_denomination
  let spent : ℕ := payment - total_change
  spent / cost_per_game = 6 ∧ spent % cost_per_game = 0

theorem solve_board_game_problem :
  board_game_problem 15 100 2 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_board_game_problem_l3595_359501


namespace NUMINAMATH_CALUDE_liquid_film_radius_l3595_359573

/-- The radius of a circular film formed by a liquid on water -/
theorem liquid_film_radius 
  (thickness : ℝ) 
  (volume : ℝ) 
  (h1 : thickness = 0.2)
  (h2 : volume = 320) : 
  ∃ (r : ℝ), r = Real.sqrt (1600 / Real.pi) ∧ π * r^2 * thickness = volume :=
sorry

end NUMINAMATH_CALUDE_liquid_film_radius_l3595_359573


namespace NUMINAMATH_CALUDE_solution_value_l3595_359541

theorem solution_value (x y a : ℝ) : 
  x = 1 → y = -3 → a * x - y = 1 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3595_359541


namespace NUMINAMATH_CALUDE_calculate_expression_l3595_359592

theorem calculate_expression : (-8) * 3 / ((-2)^2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3595_359592


namespace NUMINAMATH_CALUDE_poster_area_is_28_l3595_359521

/-- The area of a rectangular poster -/
def poster_area (width height : ℝ) : ℝ := width * height

/-- Theorem: The area of a rectangular poster with width 4 inches and height 7 inches is 28 square inches -/
theorem poster_area_is_28 : poster_area 4 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_poster_area_is_28_l3595_359521


namespace NUMINAMATH_CALUDE_car_distance_is_360_l3595_359528

/-- The distance a car needs to cover, given initial time, time factor, and new speed. -/
def car_distance (initial_time : ℝ) (time_factor : ℝ) (new_speed : ℝ) : ℝ :=
  initial_time * time_factor * new_speed

/-- Theorem stating that the car distance is 360 kilometers under given conditions. -/
theorem car_distance_is_360 :
  car_distance 6 (3/2) 40 = 360 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_is_360_l3595_359528


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3595_359542

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 7 = -8)
  (h_second : a 2 = 2) :
  ∃ d : ℤ, d = -3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3595_359542


namespace NUMINAMATH_CALUDE_marble_probability_l3595_359517

theorem marble_probability (total : ℕ) (blue : ℕ) (red : ℕ) 
  (h_total : total = 50)
  (h_blue : blue = 12)
  (h_red : red = 18)
  (h_white : total - blue - red = 20) :
  (red + (total - blue - red)) / total = 19 / 25 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3595_359517


namespace NUMINAMATH_CALUDE_log_three_five_l3595_359546

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_three_five (a : ℝ) (h : log 5 45 = a) : log 5 3 = (a - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_three_five_l3595_359546


namespace NUMINAMATH_CALUDE_curve_intersection_and_tangent_l3595_359529

noncomputable section

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := Real.exp x * (c*x + d)

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 2*x + a

-- Define the derivative of g
def g' (c d x : ℝ) : ℝ := Real.exp x * (c*x + d + c)

-- State the theorem
theorem curve_intersection_and_tangent (a b c d : ℝ) :
  (f a b 0 = 2) →
  (g c d 0 = 2) →
  (f' a 0 = 4) →
  (g' c d 0 = 4) →
  (a = 4 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∧
  (∀ k, (∀ x, x ≥ -2 → f 4 2 x ≤ k * g 2 2 x) ↔ (1 ≤ k ∧ k ≤ Real.exp 2)) :=
by sorry

end

end NUMINAMATH_CALUDE_curve_intersection_and_tangent_l3595_359529


namespace NUMINAMATH_CALUDE_square_roots_problem_l3595_359510

theorem square_roots_problem (a : ℝ) (x : ℝ) 
  (h1 : a > 0)
  (h2 : (2*x + 6)^2 = a)
  (h3 : (x - 18)^2 = a) :
  a = 196 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l3595_359510


namespace NUMINAMATH_CALUDE_vinegar_mixture_l3595_359536

/-- Given a mixture of water and vinegar, prove the amount of vinegar used. -/
theorem vinegar_mixture (total_mixture water_fraction vinegar_fraction : ℚ) 
  (h_total : total_mixture = 27)
  (h_water : water_fraction = 3/5)
  (h_vinegar : vinegar_fraction = 5/6)
  (h_water_amount : water_fraction * 20 + vinegar_fraction * vinegar_amount = total_mixture) :
  vinegar_amount = 15 := by
  sorry

#check vinegar_mixture

end NUMINAMATH_CALUDE_vinegar_mixture_l3595_359536


namespace NUMINAMATH_CALUDE_cos_arctan_equal_x_squared_l3595_359598

theorem cos_arctan_equal_x_squared :
  ∃ (x : ℝ), x > 0 ∧ Real.cos (Real.arctan x) = x → x^2 = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_arctan_equal_x_squared_l3595_359598
