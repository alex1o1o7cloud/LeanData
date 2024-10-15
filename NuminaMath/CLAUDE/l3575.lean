import Mathlib

namespace NUMINAMATH_CALUDE_initial_capacity_proof_l3575_357559

/-- The initial capacity of a barrel in liters -/
def initial_capacity : ℝ := 220

/-- The percentage of contents remaining after the leak -/
def remaining_percentage : ℝ := 0.9

/-- The amount of liquid remaining in the barrel after the leak, in liters -/
def remaining_liquid : ℝ := 198

/-- Theorem stating that the initial capacity is correct given the conditions -/
theorem initial_capacity_proof : 
  initial_capacity * remaining_percentage = remaining_liquid :=
by sorry

end NUMINAMATH_CALUDE_initial_capacity_proof_l3575_357559


namespace NUMINAMATH_CALUDE_find_m_find_t_range_l3575_357577

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- Theorem 1: Find the value of m
theorem find_m :
  (∃ m : ℝ, ∀ x : ℝ, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-2) 2) →
  (∃ m : ℝ, m = 2 ∧ ∀ x : ℝ, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-2) 2) :=
sorry

-- Theorem 2: Find the range of t
theorem find_t_range (m : ℝ) (h : m = 2) :
  (∀ x t : ℝ, f m x ≥ -|x + 6| - t^2 + t) →
  (∀ t : ℝ, t ∈ Set.Iic (-2) ∪ Set.Ici 3) :=
sorry

end NUMINAMATH_CALUDE_find_m_find_t_range_l3575_357577


namespace NUMINAMATH_CALUDE_garden_perimeter_l3575_357560

/-- The perimeter of a rectangular garden with width 24 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is 64 meters. -/
theorem garden_perimeter : 
  ∀ (garden_length : ℝ),
  garden_length > 0 →
  24 * garden_length = 16 * 12 →
  2 * garden_length + 2 * 24 = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l3575_357560


namespace NUMINAMATH_CALUDE_point_b_value_l3575_357533

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- The distance between two points on a number line -/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_b_value (a b : Point) (h1 : a.value = -2) (h2 : distance a b = 4) : b.value = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_b_value_l3575_357533


namespace NUMINAMATH_CALUDE_equation_represents_point_l3575_357507

/-- The equation represents a point in the xy-plane -/
theorem equation_represents_point (a b x y : ℝ) :
  x^2 + y^2 + 2*a*x + 2*b*y + a^2 + b^2 = 0 ↔ (x = -a ∧ y = -b) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_point_l3575_357507


namespace NUMINAMATH_CALUDE_cars_meet_time_l3575_357545

/-- Two cars meet on a highway -/
theorem cars_meet_time (highway_length : ℝ) (speed1 speed2 : ℝ) (h1 : highway_length = 500) 
  (h2 : speed1 = 40) (h3 : speed2 = 60) : 
  (highway_length / (speed1 + speed2) = 5) := by
sorry

end NUMINAMATH_CALUDE_cars_meet_time_l3575_357545


namespace NUMINAMATH_CALUDE_total_pets_l3575_357511

def num_dogs : ℕ := 2
def num_cats : ℕ := 3
def num_fish : ℕ := 2 * (num_dogs + num_cats)

theorem total_pets : num_dogs + num_cats + num_fish = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_l3575_357511


namespace NUMINAMATH_CALUDE_equation_solution_l3575_357553

theorem equation_solution (x y : ℝ) : 
  x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ↔ x = 1/3 ∧ y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3575_357553


namespace NUMINAMATH_CALUDE_only_cooking_count_l3575_357566

/-- Given information about curriculum participation --/
structure CurriculumParticipation where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cooking_and_yoga : ℕ
  all_curriculums : ℕ
  cooking_and_weaving : ℕ

/-- Theorem stating the number of people who study only cooking --/
theorem only_cooking_count (cp : CurriculumParticipation)
  (h1 : cp.yoga = 25)
  (h2 : cp.cooking = 15)
  (h3 : cp.weaving = 8)
  (h4 : cp.cooking_and_yoga = 7)
  (h5 : cp.all_curriculums = 3)
  (h6 : cp.cooking_and_weaving = 3) :
  cp.cooking - (cp.cooking_and_yoga - cp.all_curriculums) - (cp.cooking_and_weaving - cp.all_curriculums) - cp.all_curriculums = 8 :=
by sorry

end NUMINAMATH_CALUDE_only_cooking_count_l3575_357566


namespace NUMINAMATH_CALUDE_fahrenheit_celsius_conversion_l3575_357531

theorem fahrenheit_celsius_conversion (F C : ℝ) : 
  C = (5 / 9) * (F - 32) → C = 40 → F = 104 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_celsius_conversion_l3575_357531


namespace NUMINAMATH_CALUDE_cos_20_cos_10_minus_sin_160_sin_10_l3575_357556

theorem cos_20_cos_10_minus_sin_160_sin_10 :
  Real.cos (20 * π / 180) * Real.cos (10 * π / 180) -
  Real.sin (160 * π / 180) * Real.sin (10 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_20_cos_10_minus_sin_160_sin_10_l3575_357556


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l3575_357529

theorem arithmetic_sequence_squares (k : ℤ) : k = 1612 →
  ∃ (a d : ℤ), 
    (25 + k = (a - d)^2) ∧ 
    (289 + k = a^2) ∧ 
    (529 + k = (a + d)^2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l3575_357529


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_8_l3575_357588

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → 118 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_8_l3575_357588


namespace NUMINAMATH_CALUDE_rectangle_cutting_l3575_357590

theorem rectangle_cutting (m : ℕ) (h : m > 12) : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * y > m ∧ x * (y - 1) < m :=
by sorry

end NUMINAMATH_CALUDE_rectangle_cutting_l3575_357590


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_ellipse_parabola_intersection_distance_l3575_357576

/-- The distance between intersection points of a parabola and a vertical line -/
theorem parabola_intersection_distance 
  (a : ℝ) -- Parameter of the parabola
  (x_intersect : ℝ) -- x-coordinate of the vertical line
  (h1 : a > 0) -- Ensure parabola opens to the right
  (h2 : x_intersect > 0) -- Ensure vertical line is to the right of y-axis
  : 
  let y1 := Real.sqrt (4 * a * x_intersect)
  let y2 := -Real.sqrt (4 * a * x_intersect)
  abs (y1 - y2) = 2 * Real.sqrt (4 * a * x_intersect) :=
by sorry

/-- The main theorem about the specific ellipse and parabola -/
theorem ellipse_parabola_intersection_distance :
  let ellipse := fun (x y : ℝ) => x^2 / 25 + y^2 / 16 = 1
  let parabola := fun (x y : ℝ) => y^2 = (100 / 3) * x
  let x_intersect := 25 / 3
  abs ((Real.sqrt ((100 / 3) * x_intersect)) - (-Real.sqrt ((100 / 3) * x_intersect))) = 100 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_ellipse_parabola_intersection_distance_l3575_357576


namespace NUMINAMATH_CALUDE_current_rate_calculation_l3575_357580

/-- Given a boat with speed in still water and its downstream travel details, 
    calculate the rate of the current. -/
theorem current_rate_calculation (boat_speed : ℝ) (distance : ℝ) (time : ℝ) :
  boat_speed = 15 →
  distance = 3.6 →
  time = 1/5 →
  ∃ (current_rate : ℝ), current_rate = 3 ∧ 
    distance = (boat_speed + current_rate) * time :=
by sorry

end NUMINAMATH_CALUDE_current_rate_calculation_l3575_357580


namespace NUMINAMATH_CALUDE_bouquet_stamens_l3575_357589

/-- Proves that the total number of stamens in a bouquet is 216 --/
theorem bouquet_stamens :
  ∀ (black_roses crimson_flowers : ℕ),
  (4 * black_roses + 8 * crimson_flowers) - (2 * black_roses + 3 * crimson_flowers) = 108 →
  4 * black_roses + 10 * crimson_flowers = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_bouquet_stamens_l3575_357589


namespace NUMINAMATH_CALUDE_john_bought_three_shirts_l3575_357549

/-- The number of dress shirts John bought -/
def num_shirts : ℕ := 3

/-- The cost of each dress shirt in dollars -/
def shirt_cost : ℚ := 20

/-- The tax rate as a percentage -/
def tax_rate : ℚ := 10

/-- The total amount John paid in dollars -/
def total_paid : ℚ := 66

/-- Theorem stating that the number of shirts John bought is correct -/
theorem john_bought_three_shirts :
  (shirt_cost * num_shirts) * (1 + tax_rate / 100) = total_paid := by
  sorry

end NUMINAMATH_CALUDE_john_bought_three_shirts_l3575_357549


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3575_357594

theorem point_in_fourth_quadrant (a : ℝ) (h : a > 1) :
  let P : ℝ × ℝ := (1 + a, 1 - a)
  P.1 > 0 ∧ P.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l3575_357594


namespace NUMINAMATH_CALUDE_plot_length_l3575_357569

/-- Proves that the length of a rectangular plot is 55 meters given the specified conditions -/
theorem plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) : 
  length = breadth + 10 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  total_cost = perimeter * cost_per_meter →
  length = 55 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_l3575_357569


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l3575_357572

theorem quadratic_roots_nature (x : ℝ) : 
  (x^2 - 6*x + 9 = 0) → (∃ r : ℝ, x = r ∧ x^2 - 6*x + 9 = 0) ∧ 
  (∃! r : ℝ, x^2 - 6*x + 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l3575_357572


namespace NUMINAMATH_CALUDE_inverse_power_function_at_4_l3575_357593

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- State the theorem
theorem inverse_power_function_at_4 (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) :
  Function.invFun f 4 = 16 := by
sorry

end NUMINAMATH_CALUDE_inverse_power_function_at_4_l3575_357593


namespace NUMINAMATH_CALUDE_board_covering_l3575_357510

-- Define a function to check if a board can be covered by dominoes
def can_cover_board (n m k : ℕ+) : Prop :=
  ∃ (arrangement : ℕ → ℕ → Bool), ∀ (i j : ℕ), i < n.val ∧ j < m.val →
    (arrangement i j = true ∧ arrangement (i + 1) j = true ∧ i + 1 < n.val) ∨
    (arrangement i j = true ∧ arrangement i (j + 1) = true ∧ j + 1 < m.val)

-- State the theorem
theorem board_covering (n m k : ℕ+) :
  can_cover_board n m k ↔ (k.val ∣ n.val ∨ k.val ∣ m.val) :=
sorry

end NUMINAMATH_CALUDE_board_covering_l3575_357510


namespace NUMINAMATH_CALUDE_rolling_circle_traces_hypotrochoid_l3575_357570

/-- A circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point on a 2D plane -/
def Point := ℝ × ℝ

/-- Represents a hypotrochoid curve -/
def Hypotrochoid := Point → ℝ → Point

theorem rolling_circle_traces_hypotrochoid 
  (large_circle : Circle)
  (small_circle : Circle)
  (h1 : large_circle.radius = 2 * small_circle.radius)
  (h2 : small_circle.radius > 0)
  (point : Point) 
  (h3 : ∃ (θ : ℝ), point = 
    (small_circle.center.1 + small_circle.radius * Real.cos θ, 
     small_circle.center.2 + small_circle.radius * Real.sin θ))
  : ∃ (curve : Hypotrochoid), 
    ∀ (t : ℝ), curve point t = 
      ((large_circle.radius - small_circle.radius) * Real.cos t + small_circle.radius * Real.cos ((large_circle.radius / small_circle.radius - 1) * t),
       (large_circle.radius - small_circle.radius) * Real.sin t - small_circle.radius * Real.sin ((large_circle.radius / small_circle.radius - 1) * t)) :=
by sorry

end NUMINAMATH_CALUDE_rolling_circle_traces_hypotrochoid_l3575_357570


namespace NUMINAMATH_CALUDE_product_of_sum_of_squares_l3575_357502

theorem product_of_sum_of_squares (a b c d : ℝ) :
  (a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 + (a*d - b*c)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_squares_l3575_357502


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3575_357512

theorem inscribed_circle_radius 
  (r : ℝ) 
  (α γ : ℝ) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.sin α * Real.sin γ = 1/Real.sqrt 10) : 
  ∃ ρ : ℝ, ρ = ((2 * Real.sqrt 10 - 5) / 5) * r :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3575_357512


namespace NUMINAMATH_CALUDE_complete_square_equivalence_l3575_357571

theorem complete_square_equivalence : 
  ∀ x : ℝ, x^2 - 6*x - 7 = 0 ↔ (x - 3)^2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_complete_square_equivalence_l3575_357571


namespace NUMINAMATH_CALUDE_max_value_f_in_interval_l3575_357595

/-- The function f(x) = x^3 - 3x^2 + 2 -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

/-- Theorem: The maximum value of f(x) = x^3 - 3x^2 + 2 in the interval [-1, 1] is 2 -/
theorem max_value_f_in_interval :
  (∀ x : ℝ, x ≥ -1 ∧ x ≤ 1 → f x ≤ 2) ∧
  (∃ x : ℝ, x ≥ -1 ∧ x ≤ 1 ∧ f x = 2) :=
sorry

end NUMINAMATH_CALUDE_max_value_f_in_interval_l3575_357595


namespace NUMINAMATH_CALUDE_carlos_laundry_time_l3575_357567

/-- The time it takes for Carlos to do his laundry -/
def laundry_time (num_loads : ℕ) (wash_time_per_load : ℕ) (dry_time : ℕ) : ℕ :=
  num_loads * wash_time_per_load + dry_time

/-- Theorem: Carlos's laundry takes 165 minutes -/
theorem carlos_laundry_time :
  laundry_time 2 45 75 = 165 := by
  sorry

end NUMINAMATH_CALUDE_carlos_laundry_time_l3575_357567


namespace NUMINAMATH_CALUDE_range_of_fraction_l3575_357587

theorem range_of_fraction (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a * b = 2) :
  (a^2 + b^2) / (a - b) ≤ -4 ∧ ∃ (a' b' : ℝ), b' > a' ∧ a' > 0 ∧ a' * b' = 2 ∧ (a'^2 + b'^2) / (a' - b') = -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l3575_357587


namespace NUMINAMATH_CALUDE_max_triangles_three_families_l3575_357561

/-- Represents a family of parallel lines -/
structure LineFamily :=
  (count : ℕ)

/-- Calculates the maximum number of triangles formed by three families of parallel lines -/
def max_triangles (f1 f2 f3 : LineFamily) : ℕ :=
  150

/-- Theorem stating that three families of 10 parallel lines form at most 150 triangles -/
theorem max_triangles_three_families :
  ∀ (f1 f2 f3 : LineFamily),
    f1.count = 10 → f2.count = 10 → f3.count = 10 →
    max_triangles f1 f2 f3 = 150 :=
by
  sorry

#check max_triangles_three_families

end NUMINAMATH_CALUDE_max_triangles_three_families_l3575_357561


namespace NUMINAMATH_CALUDE_max_edges_no_triangle_max_edges_no_K4_l3575_357518

/-- The Turán number T(n, r) is the maximum number of edges in a graph with n vertices that does not contain a complete subgraph of r+1 vertices. -/
def turan_number (n : ℕ) (r : ℕ) : ℕ := sorry

/-- A graph with n vertices -/
structure Graph (n : ℕ) where
  edges : Finset (Fin n × Fin n)

/-- The number of edges in a graph -/
def num_edges {n : ℕ} (G : Graph n) : ℕ := G.edges.card

/-- A graph contains a triangle if it has a complete subgraph of 3 vertices -/
def has_triangle {n : ℕ} (G : Graph n) : Prop := sorry

/-- A graph contains a K4 if it has a complete subgraph of 4 vertices -/
def has_K4 {n : ℕ} (G : Graph n) : Prop := sorry

theorem max_edges_no_triangle (G : Graph 30) :
  ¬has_triangle G → num_edges G ≤ 225 :=
sorry

theorem max_edges_no_K4 (G : Graph 30) :
  ¬has_K4 G → num_edges G ≤ 200 :=
sorry

end NUMINAMATH_CALUDE_max_edges_no_triangle_max_edges_no_K4_l3575_357518


namespace NUMINAMATH_CALUDE_paint_remaining_l3575_357543

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint = 1 → 
  let remaining_after_day1 := initial_paint - (1/4 * initial_paint)
  let remaining_after_day2 := remaining_after_day1 - (1/2 * remaining_after_day1)
  remaining_after_day2 = 3/8 * initial_paint := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_l3575_357543


namespace NUMINAMATH_CALUDE_arithmetic_sequence_61st_term_l3575_357592

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_61st_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_5 : a 5 = 33)
  (h_45 : a 45 = 153) :
  a 61 = 201 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_61st_term_l3575_357592


namespace NUMINAMATH_CALUDE_tetrahedron_distance_sum_l3575_357544

/-- Theorem about distances in a tetrahedron -/
theorem tetrahedron_distance_sum (V : ℝ) (S₁ S₂ S₃ S₄ : ℝ) (H₁ H₂ H₃ H₄ : ℝ) (k : ℝ) :
  V > 0 →
  S₁ > 0 → S₂ > 0 → S₃ > 0 → S₄ > 0 →
  H₁ > 0 → H₂ > 0 → H₃ > 0 → H₄ > 0 →
  S₁ = k → S₂ = 2*k → S₃ = 3*k → S₄ = 4*k →
  V = (1/3) * (S₁*H₁ + S₂*H₂ + S₃*H₃ + S₄*H₄) →
  H₁ + 2*H₂ + 3*H₃ + 4*H₄ = 3*V/k :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_distance_sum_l3575_357544


namespace NUMINAMATH_CALUDE_max_missed_problems_l3575_357524

theorem max_missed_problems (total_problems : ℕ) (pass_percentage : ℚ) 
  (h1 : total_problems = 50)
  (h2 : pass_percentage = 85 / 100) : 
  ∃ (max_missed : ℕ), 
    (max_missed ≤ total_problems) ∧ 
    ((total_problems - max_missed : ℚ) / total_problems ≥ pass_percentage) ∧
    ∀ (n : ℕ), n > max_missed → 
      ((total_problems - n : ℚ) / total_problems < pass_percentage) :=
by
  sorry

end NUMINAMATH_CALUDE_max_missed_problems_l3575_357524


namespace NUMINAMATH_CALUDE_sledding_problem_l3575_357591

/-- Sledding problem -/
theorem sledding_problem (mary_hill_length : ℝ) (mary_speed : ℝ) (ann_speed : ℝ) (time_difference : ℝ)
  (h1 : mary_hill_length = 630)
  (h2 : mary_speed = 90)
  (h3 : ann_speed = 40)
  (h4 : time_difference = 13) :
  let mary_time := mary_hill_length / mary_speed
  let ann_time := mary_time + time_difference
  ann_speed * ann_time = 800 :=
by sorry

end NUMINAMATH_CALUDE_sledding_problem_l3575_357591


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l3575_357585

def sumIntegers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def countEvenIntegers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  let x := sumIntegers 30 40
  let y := countEvenIntegers 30 40
  x + y = 391 := by sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l3575_357585


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3575_357573

theorem x_plus_y_value (x y : ℝ) (hx : |x| = 3) (hy : |y| = 2) (hxy : x > y) :
  x + y = 5 ∨ x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3575_357573


namespace NUMINAMATH_CALUDE_erins_launderette_machines_l3575_357555

/-- Represents the number of coins in a machine --/
structure CoinCount where
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total value of coins in dollars --/
def coinValue (c : CoinCount) : Rat :=
  (c.quarters * 25 + c.dimes * 10 + c.nickels * 5 + c.pennies) / 100

/-- Represents the launderette problem --/
structure LaunderetteProblem where
  machineCoins : CoinCount
  totalCashed : Rat
  minMachines : Nat
  maxMachines : Nat

/-- The specific launderette problem instance --/
def erinsProblem : LaunderetteProblem :=
  { machineCoins := { quarters := 80, dimes := 100, nickels := 50, pennies := 120 }
    totalCashed := 165
    minMachines := 3
    maxMachines := 5 }

theorem erins_launderette_machines (p : LaunderetteProblem) (h : p = erinsProblem) :
    ∃ n : Nat, n ≥ p.minMachines ∧ n ≤ p.maxMachines ∧ 
    n * coinValue p.machineCoins = p.totalCashed := by sorry

end NUMINAMATH_CALUDE_erins_launderette_machines_l3575_357555


namespace NUMINAMATH_CALUDE_f_increasing_and_odd_l3575_357513

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - 5^(-x) else 5^x - 1

theorem f_increasing_and_odd :
  (∀ x y, x < y → f x < f y) ∧
  (∀ x, f (-x) = -f x) :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_and_odd_l3575_357513


namespace NUMINAMATH_CALUDE_salary_remaining_l3575_357581

def salary : ℕ := 180000

def food_fraction : ℚ := 1/5
def rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5

def remaining_money : ℕ := 18000

theorem salary_remaining :
  salary - (↑salary * (food_fraction + rent_fraction + clothes_fraction)).floor = remaining_money := by
  sorry

end NUMINAMATH_CALUDE_salary_remaining_l3575_357581


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_l3575_357598

theorem quadratic_always_real_roots (m : ℝ) (hm : m ≠ 0) :
  let a := m
  let b := 1 - 5 * m
  let c := -5
  (b^2 - 4*a*c) ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_l3575_357598


namespace NUMINAMATH_CALUDE_equation_satisfied_l3575_357568

theorem equation_satisfied (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) :
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l3575_357568


namespace NUMINAMATH_CALUDE_exists_100_same_polygons_l3575_357564

/-- Represents a convex polygon --/
structure ConvexPolygon where
  vertices : ℕ

/-- Represents the state of the paper after some cuts --/
structure PaperState where
  polygons : List ConvexPolygon

/-- A function that performs a single cut --/
def cut (state : PaperState) : PaperState :=
  sorry

/-- A function that checks if there are 100 polygons with the same number of vertices --/
def has_100_same_polygons (state : PaperState) : Bool :=
  sorry

/-- The main theorem --/
theorem exists_100_same_polygons :
  ∃ (n : ℕ), ∀ (initial : PaperState),
    has_100_same_polygons (n.iterate cut initial) = true :=
  sorry

end NUMINAMATH_CALUDE_exists_100_same_polygons_l3575_357564


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_eq_five_l3575_357536

/-- The number of zeroes to the right of the decimal point and before the first non-zero digit
    in the decimal representation of 1/(2^3 * 5^6) -/
def zeros_before_first_nonzero : ℕ :=
  let denominator := 2^3 * 5^6
  let decimal_places := 6  -- log_10(denominator)
  decimal_places - 1

theorem zeros_before_first_nonzero_eq_five :
  zeros_before_first_nonzero = 5 := by
  sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_eq_five_l3575_357536


namespace NUMINAMATH_CALUDE_sum_of_roots_for_f_l3575_357526

theorem sum_of_roots_for_f (f : ℝ → ℝ) : 
  (∀ x, f (x / 4) = x^2 + 3*x + 2) →
  (∃ z₁ z₂, f (4*z₁) = 8 ∧ f (4*z₂) = 8 ∧ z₁ ≠ z₂ ∧ 
    (∀ z, f (4*z) = 8 → z = z₁ ∨ z = z₂) ∧
    z₁ + z₂ = -3/16) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_for_f_l3575_357526


namespace NUMINAMATH_CALUDE_fifth_term_of_specific_geometric_progression_l3575_357550

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fifth_term_of_specific_geometric_progression :
  let a := 2 ^ (1/4 : ℝ)
  let r := 2 ^ (1/4 : ℝ)
  geometric_progression a r 1 = 2 ^ (1/4 : ℝ) ∧
  geometric_progression a r 2 = 2 ^ (1/2 : ℝ) ∧
  geometric_progression a r 3 = 2 ^ (3/4 : ℝ) →
  geometric_progression a r 5 = 2 ^ (5/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_specific_geometric_progression_l3575_357550


namespace NUMINAMATH_CALUDE_peters_class_size_l3575_357584

/-- Represents the number of students with a specific number of hands -/
structure HandDistribution :=
  (hands : ℕ)
  (students : ℕ)

/-- Represents the class data -/
structure ClassData :=
  (total_hands : ℕ)
  (distribution : List HandDistribution)
  (unspecified_students : ℕ)

/-- Calculates the total number of students in Peter's class -/
def total_students (data : ClassData) : ℕ :=
  (data.distribution.map (λ d => d.students)).sum + data.unspecified_students + 1

/-- Theorem stating that the total number of students in Peter's class is 17 -/
theorem peters_class_size (data : ClassData) 
  (h1 : data.total_hands = 20)
  (h2 : data.distribution = [
    ⟨2, 7⟩, 
    ⟨1, 3⟩, 
    ⟨3, 1⟩, 
    ⟨0, 2⟩
  ])
  (h3 : data.unspecified_students = 3) :
  total_students data = 17 := by
  sorry

end NUMINAMATH_CALUDE_peters_class_size_l3575_357584


namespace NUMINAMATH_CALUDE_rectangle_area_is_six_l3575_357537

/-- Represents a square within the rectangle ABCD -/
structure Square where
  side_length : ℝ
  area : ℝ
  area_eq : area = side_length ^ 2

/-- The rectangle ABCD containing three squares -/
structure Rectangle where
  squares : Fin 3 → Square
  non_overlapping : ∀ i j, i ≠ j → (squares i).area + (squares j).area ≤ area
  shaded_square_area : (squares 0).area = 1
  area : ℝ

/-- The theorem stating that the area of rectangle ABCD is 6 square inches -/
theorem rectangle_area_is_six (rect : Rectangle) : rect.area = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_six_l3575_357537


namespace NUMINAMATH_CALUDE_one_pattern_cannot_fold_to_pyramid_l3575_357505

/-- Represents a pattern of identical squares -/
structure Pattern :=
  (squares : ℕ)
  (foldable : Bool)

/-- Represents a pyramid with a square base -/
structure Pyramid :=
  (base : ℕ)
  (sides : ℕ)

/-- Function to check if a pattern can be folded into a pyramid -/
def can_fold_to_pyramid (p : Pattern) (pyr : Pyramid) : Prop :=
  p.squares = pyr.base + pyr.sides ∧ p.foldable

/-- Theorem stating that exactly one pattern cannot be folded into a pyramid -/
theorem one_pattern_cannot_fold_to_pyramid 
  (A B C D : Pattern) 
  (pyr : Pyramid) 
  (h_pyr : pyr.base = 1 ∧ pyr.sides = 4) 
  (h_ABC : can_fold_to_pyramid A pyr ∧ can_fold_to_pyramid B pyr ∧ can_fold_to_pyramid C pyr) 
  (h_D : ¬can_fold_to_pyramid D pyr) : 
  ∃! p : Pattern, ¬can_fold_to_pyramid p pyr :=
sorry

end NUMINAMATH_CALUDE_one_pattern_cannot_fold_to_pyramid_l3575_357505


namespace NUMINAMATH_CALUDE_vacant_seats_l3575_357535

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) 
  (h1 : total_seats = 600) 
  (h2 : filled_percentage = 60 / 100) : 
  (1 - filled_percentage) * total_seats = 240 := by
sorry

end NUMINAMATH_CALUDE_vacant_seats_l3575_357535


namespace NUMINAMATH_CALUDE_equation_solutions_l3575_357575

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 + 2*x₁ = 0 ∧ x₂^2 + 2*x₂ = 0) ∧ x₁ = 0 ∧ x₂ = -2) ∧
  (∃ y₁ y₂ : ℝ, (2*y₁^2 - 2*y₁ = 1 ∧ 2*y₂^2 - 2*y₂ = 1) ∧ 
    y₁ = (1 + Real.sqrt 3) / 2 ∧ y₂ = (1 - Real.sqrt 3) / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3575_357575


namespace NUMINAMATH_CALUDE_pure_imaginary_square_root_l3575_357501

theorem pure_imaginary_square_root (a : ℝ) : 
  let z : ℂ := (a - Complex.I) ^ 2
  (∃ (b : ℝ), z = Complex.I * b) → (a = 1 ∨ a = -1) := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_square_root_l3575_357501


namespace NUMINAMATH_CALUDE_equal_students_after_transfer_total_students_after_transfer_l3575_357565

/-- Represents a section in Grade 4 -/
inductive Section
| Diligence
| Industry

/-- The number of students in a section before the transfer -/
def students_before (s : Section) : ℕ :=
  match s with
  | Section.Diligence => 23
  | Section.Industry => sorry  -- We don't know this value

/-- The number of students transferred from Industry to Diligence -/
def transferred_students : ℕ := 2

/-- The number of students in a section after the transfer -/
def students_after (s : Section) : ℕ :=
  match s with
  | Section.Diligence => students_before Section.Diligence + transferred_students
  | Section.Industry => students_before Section.Industry - transferred_students

/-- Theorem stating that the sections have equal students after transfer -/
theorem equal_students_after_transfer :
  students_after Section.Diligence = students_after Section.Industry := by sorry

/-- The main theorem to prove -/
theorem total_students_after_transfer :
  students_after Section.Diligence + students_after Section.Industry = 50 := by sorry

end NUMINAMATH_CALUDE_equal_students_after_transfer_total_students_after_transfer_l3575_357565


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3575_357597

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  a = (2, -1) →
  b = (0, 1) →
  ‖a + 2 • b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3575_357597


namespace NUMINAMATH_CALUDE_hen_count_l3575_357517

theorem hen_count (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) 
  (h1 : total_animals = 48)
  (h2 : total_feet = 136)
  (h3 : hen_feet = 2)
  (h4 : cow_feet = 4) : 
  ∃ (hens cows : ℕ), 
    hens + cows = total_animals ∧ 
    hen_feet * hens + cow_feet * cows = total_feet ∧
    hens = 28 := by
  sorry

end NUMINAMATH_CALUDE_hen_count_l3575_357517


namespace NUMINAMATH_CALUDE_equation_solution_l3575_357538

theorem equation_solution : ∃! r : ℚ, (r^2 - 5*r + 4)/(r^2 - 8*r + 7) = (r^2 - 2*r - 15)/(r^2 - r - 20) ∧ r = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3575_357538


namespace NUMINAMATH_CALUDE_square_side_length_l3575_357521

theorem square_side_length 
  (total_width : ℕ) 
  (total_height : ℕ) 
  (r : ℕ) 
  (s : ℕ) :
  total_width = 3300 →
  total_height = 2000 →
  2 * r + s = total_height →
  2 * r + 3 * s = total_width →
  s = 650 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3575_357521


namespace NUMINAMATH_CALUDE_high_school_math_club_payment_l3575_357520

theorem high_school_math_club_payment (A : Nat) : 
  A < 10 → (2 * 100 + A * 10 + 3) % 3 = 0 → A = 1 ∨ A = 4 := by
  sorry

end NUMINAMATH_CALUDE_high_school_math_club_payment_l3575_357520


namespace NUMINAMATH_CALUDE_diplomats_speaking_both_languages_l3575_357548

theorem diplomats_speaking_both_languages (T F H : ℕ) (p : ℚ) : 
  T = 120 →
  F = 20 →
  T - H = 32 →
  p = 20 / 100 →
  (p * T : ℚ) = 24 →
  (F + H - (F + H - T : ℤ) : ℚ) / T * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_diplomats_speaking_both_languages_l3575_357548


namespace NUMINAMATH_CALUDE_jamal_shelving_problem_l3575_357586

/-- The number of books Jamal still has to shelve after working through different sections of the library. -/
def books_to_shelve (initial : ℕ) (history : ℕ) (fiction : ℕ) (children : ℕ) (misplaced : ℕ) : ℕ :=
  initial - history - fiction - children + misplaced

/-- Theorem stating that Jamal has 16 books left to shelve given the specific numbers from the problem. -/
theorem jamal_shelving_problem :
  books_to_shelve 51 12 19 8 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_jamal_shelving_problem_l3575_357586


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l3575_357596

/-- Given a positive real number a, prove that the difference in area between
    a rectangle with length (a-2) and width 7, and a rectangle with length a
    and width 5, is equal to 2a - 14. -/
theorem rectangle_area_difference (a : ℝ) (h : a > 0) :
  (a - 2) * 7 - a * 5 = 2 * a - 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l3575_357596


namespace NUMINAMATH_CALUDE_equation_equivalence_implies_uvw_product_l3575_357557

theorem equation_equivalence_implies_uvw_product (a b x y : ℝ) (u v w : ℤ) :
  (a^10 * x * y - a^9 * y - a^8 * x = a^6 * (b^5 - 1)) →
  ((a^u * x - a^v) * (a^w * y - a^3) = a^6 * b^5) →
  u * v * w = 48 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_implies_uvw_product_l3575_357557


namespace NUMINAMATH_CALUDE_ac_plus_bd_equals_negative_26_l3575_357528

theorem ac_plus_bd_equals_negative_26
  (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = 6) :
  a * c + b * d = -26 := by
  sorry

end NUMINAMATH_CALUDE_ac_plus_bd_equals_negative_26_l3575_357528


namespace NUMINAMATH_CALUDE_prob_same_tails_value_l3575_357582

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The probability of getting tails on a single penny toss -/
def prob_tails : ℚ := 1/2

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := 2^keiko_pennies * 2^ephraim_pennies

/-- The number of favorable outcomes (same number of tails) -/
def favorable_outcomes : ℕ := 3

/-- The probability of Ephraim getting the same number of tails as Keiko -/
def prob_same_tails : ℚ := favorable_outcomes / total_outcomes

theorem prob_same_tails_value : prob_same_tails = 3/32 := by sorry

end NUMINAMATH_CALUDE_prob_same_tails_value_l3575_357582


namespace NUMINAMATH_CALUDE_set_equality_l3575_357534

open Set

def M : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def N : Set ℝ := {x : ℝ | x ≤ 3}

theorem set_equality : (Mᶜ ∩ (M ∩ N)) = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3575_357534


namespace NUMINAMATH_CALUDE_bills_speed_day2_l3575_357599

/-- Represents Bill's two-day drive --/
structure TwoDayDrive where
  speed_day2 : ℝ
  time_day2 : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The conditions of Bill's drive satisfy the problem statement --/
def satisfies_conditions (d : TwoDayDrive) : Prop :=
  d.total_distance = (d.speed_day2 + 5) * (d.time_day2 + 2) + d.speed_day2 * d.time_day2 ∧
  d.total_time = d.time_day2 + 2 + d.time_day2 ∧
  d.total_distance = 680 ∧
  d.total_time = 18

/-- Theorem stating that Bill's average speed on the second day was 35 mph --/
theorem bills_speed_day2 (d : TwoDayDrive) (h : satisfies_conditions d) :
  d.speed_day2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_bills_speed_day2_l3575_357599


namespace NUMINAMATH_CALUDE_sum_of_real_solutions_l3575_357508

theorem sum_of_real_solutions (a : ℝ) (h : a > 1) :
  ∃ x : ℝ, x > 0 ∧ Real.sqrt (a - Real.sqrt (a + x)) = x ∧
  x = (Real.sqrt (4 * a - 3) - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_real_solutions_l3575_357508


namespace NUMINAMATH_CALUDE_no_solution_equations_l3575_357503

theorem no_solution_equations :
  (∀ x : ℝ, (|2*x| + 7 ≠ 0)) ∧
  (∀ x : ℝ, (Real.sqrt (3*x) + 2 ≠ 0)) ∧
  (∃ x : ℝ, ((x - 5)^2 = 0)) ∧
  (∃ x : ℝ, (Real.cos x - 1 = 0)) ∧
  (∃ x : ℝ, (|x| - 3 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equations_l3575_357503


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_system_l3575_357532

theorem smallest_solution_congruence_system :
  ∃ (x : ℕ), x > 0 ∧ 
    (6 * x) % 31 = 17 % 31 ∧
    x % 7 = 3 % 7 ∧
    (∀ (y : ℕ), y > 0 ∧ (6 * y) % 31 = 17 % 31 ∧ y % 7 = 3 % 7 → x ≤ y) ∧
    x = 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_system_l3575_357532


namespace NUMINAMATH_CALUDE_complex_equality_sum_l3575_357579

theorem complex_equality_sum (a b : ℝ) : 
  (a + b * Complex.I : ℂ) = Complex.I ^ 2 → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_sum_l3575_357579


namespace NUMINAMATH_CALUDE_weight_order_l3575_357525

-- Define the weights as real numbers
variable (B S C K : ℝ)

-- State the given conditions
axiom suitcase_heavier : S > B
axiom satchel_backpack_heavier : C + B > S + K
axiom basket_satchel_equal_suitcase_backpack : K + C = S + B

-- Theorem to prove
theorem weight_order : C > S ∧ S > B ∧ B > K := by sorry

end NUMINAMATH_CALUDE_weight_order_l3575_357525


namespace NUMINAMATH_CALUDE_infinitely_many_m_minus_f_eq_1989_l3575_357506

/-- The number of factors of 2 in m! -/
def f (m : ℕ) : ℕ := sorry

/-- Condition that 11 · 15m is a positive integer -/
def is_valid (m : ℕ) : Prop := 0 < 11 * 15 * m

/-- The main theorem -/
theorem infinitely_many_m_minus_f_eq_1989 :
  ∀ n : ℕ, ∃ m > n, is_valid m ∧ m - f m = 1989 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_m_minus_f_eq_1989_l3575_357506


namespace NUMINAMATH_CALUDE_problem_statement_l3575_357516

theorem problem_statement (x : ℝ) (h : x = 0.5) : 9 / (1 + 4 / x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3575_357516


namespace NUMINAMATH_CALUDE_inequality_proof_l3575_357500

theorem inequality_proof (x y z : ℝ) (h : x + y + z = x*y + y*z + z*x) :
  x / Real.sqrt (x^4 + x^2 + 1) + y / Real.sqrt (y^4 + y^2 + 1) + z / Real.sqrt (z^4 + z^2 + 1) ≥ -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3575_357500


namespace NUMINAMATH_CALUDE_correct_households_using_both_l3575_357563

/-- Represents the survey results of household soap usage -/
structure SoapSurvey where
  total : Nat
  neither : Nat
  onlyA : Nat
  bothRatio : Nat
  /-- Proves that the number of households using both brands is 30 -/
  householdsUsingBoth : Nat

/-- The actual survey data -/
def actualSurvey : SoapSurvey := {
  total := 260
  neither := 80
  onlyA := 60
  bothRatio := 3
  householdsUsingBoth := 30
}

/-- Theorem stating that the number of households using both brands is correct -/
theorem correct_households_using_both (s : SoapSurvey) : 
  s.householdsUsingBoth = 30 ∧ 
  s.total = s.neither + s.onlyA + s.householdsUsingBoth + s.bothRatio * s.householdsUsingBoth :=
by sorry

end NUMINAMATH_CALUDE_correct_households_using_both_l3575_357563


namespace NUMINAMATH_CALUDE_small_room_four_painters_l3575_357504

/-- Represents the number of work-days required for a given number of painters to complete a room -/
def work_days (painters : ℕ) (room_size : ℝ) : ℝ := sorry

theorem small_room_four_painters 
  (large_room_size small_room_size : ℝ)
  (h1 : work_days 5 large_room_size = 2)
  (h2 : small_room_size = large_room_size / 2)
  : work_days 4 small_room_size = 1.25 := by sorry

end NUMINAMATH_CALUDE_small_room_four_painters_l3575_357504


namespace NUMINAMATH_CALUDE_m_greater_than_n_l3575_357547

theorem m_greater_than_n (a b : ℝ) (h1 : 0 < a) (h2 : a < 1/b) : 
  (1/(1+a) + 1/(1+b)) > (a/(1+a) + b/(1+b)) := by
sorry

end NUMINAMATH_CALUDE_m_greater_than_n_l3575_357547


namespace NUMINAMATH_CALUDE_equation_with_two_variables_degree_one_is_linear_l3575_357540

/-- Definition of a linear equation in two variables -/
def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ) (c : ℝ), ∀ (x y : ℝ), f x y = a * x + b * y + c

/-- Theorem stating that an equation with two variables and terms of degree 1 is a linear equation in two variables -/
theorem equation_with_two_variables_degree_one_is_linear 
  (f : ℝ → ℝ → ℝ) 
  (h1 : ∃ (x y : ℝ), f x y ≠ f 0 0) -- Condition: contains two variables
  (h2 : ∀ (x y : ℝ), ∃ (a b : ℝ) (c : ℝ), f x y = a * x + b * y + c) -- Condition: terms with variables are of degree 1
  : is_linear_equation_in_two_variables f :=
sorry

end NUMINAMATH_CALUDE_equation_with_two_variables_degree_one_is_linear_l3575_357540


namespace NUMINAMATH_CALUDE_find_b_squared_l3575_357527

/-- A complex function satisfying certain properties -/
def f (a b : ℝ) (z : ℂ) : ℂ := (a + b * Complex.I) * z

/-- The main theorem -/
theorem find_b_squared (a b : ℝ) :
  (∀ z : ℂ, Complex.abs (f a b z - z) = Complex.abs (f a b z)) →
  a = 2 →
  Complex.abs (a + b * Complex.I) = 10 →
  b^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_find_b_squared_l3575_357527


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3575_357583

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) → a ∈ Set.Icc (-8) 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3575_357583


namespace NUMINAMATH_CALUDE_finite_moves_l3575_357574

/-- Represents the position of a number after m minutes -/
def position (initial_pos : ℕ) (m : ℕ) : ℕ :=
  if m ∣ initial_pos then initial_pos + m - 1 else initial_pos - 1

/-- Represents whether a number at initial_pos has moved after m minutes -/
def has_moved (initial_pos : ℕ) (m : ℕ) : Prop :=
  position initial_pos m ≠ initial_pos

/-- The main theorem stating that each natural number moves only finitely many times -/
theorem finite_moves (n : ℕ) : ∃ (M : ℕ), ∀ (m : ℕ), m ≥ M → ¬(has_moved n m) := by
  sorry


end NUMINAMATH_CALUDE_finite_moves_l3575_357574


namespace NUMINAMATH_CALUDE_angle_in_first_or_third_quadrant_l3575_357530

/-- An angle is in the first quadrant if it's between 0° and 90° -/
def is_first_quadrant (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

/-- An angle is in the third quadrant if it's between 180° and 270° -/
def is_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

/-- The main theorem: for any integer k, the angle k·180° + 45° is in either the first or third quadrant -/
theorem angle_in_first_or_third_quadrant (k : ℤ) :
  let α := k * 180 + 45
  is_first_quadrant (α % 360) ∨ is_third_quadrant (α % 360) :=
sorry

end NUMINAMATH_CALUDE_angle_in_first_or_third_quadrant_l3575_357530


namespace NUMINAMATH_CALUDE_angles_on_x_axis_characterization_l3575_357514

/-- The set of angles with terminal sides on the x-axis -/
def AnglesOnXAxis : Set ℝ := {α | ∃ k : ℤ, α = k * Real.pi}

/-- Theorem: The set of angles with terminal sides on the x-axis is equal to {α | α = kπ, k ∈ ℤ} -/
theorem angles_on_x_axis_characterization :
  AnglesOnXAxis = {α : ℝ | ∃ k : ℤ, α = k * Real.pi} := by
  sorry

end NUMINAMATH_CALUDE_angles_on_x_axis_characterization_l3575_357514


namespace NUMINAMATH_CALUDE_modular_inverse_11_mod_101_l3575_357558

theorem modular_inverse_11_mod_101 :
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 100 ∧ (11 * x) % 101 = 1 :=
by
  use 46
  sorry

end NUMINAMATH_CALUDE_modular_inverse_11_mod_101_l3575_357558


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3575_357551

/-- A regular polygon with side length 7 and exterior angle 72 degrees has a perimeter of 35 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧
  side_length = 7 ∧
  exterior_angle = 72 ∧
  n * exterior_angle = 360 →
  n * side_length = 35 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3575_357551


namespace NUMINAMATH_CALUDE_remainder_seven_n_l3575_357578

theorem remainder_seven_n (n : ℤ) (h : n % 5 = 3) : (7 * n) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_n_l3575_357578


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l3575_357523

theorem smallest_m_for_integral_solutions : ∃ (m : ℕ), 
  (∀ (k : ℕ), k < m → ¬∃ (x y : ℤ), 15 * x^2 - k * x + 315 = 0 ∧ 15 * y^2 - k * y + 315 = 0 ∧ x ≠ y) ∧
  (∃ (x y : ℤ), 15 * x^2 - m * x + 315 = 0 ∧ 15 * y^2 - m * y + 315 = 0 ∧ x ≠ y) ∧
  m = 150 := by
sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l3575_357523


namespace NUMINAMATH_CALUDE_planes_perpendicular_l3575_357541

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (m l : Line) (α β : Plane) 
  (h1 : m ≠ l) 
  (h2 : α ≠ β) 
  (h3 : parallel m l) 
  (h4 : perpendicular l β) 
  (h5 : contains α m) : 
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l3575_357541


namespace NUMINAMATH_CALUDE_shaded_to_white_ratio_is_five_thirds_l3575_357515

/-- A square subdivided into smaller squares where vertices of inner squares 
    are at midpoints of sides of the next larger square -/
structure SubdividedSquare :=
  (side : ℝ)
  (is_positive : side > 0)

/-- The ratio of shaded to white area in a subdivided square -/
def shaded_to_white_ratio (s : SubdividedSquare) : ℚ :=
  5 / 3

/-- Theorem stating that the ratio of shaded to white area is 5:3 -/
theorem shaded_to_white_ratio_is_five_thirds (s : SubdividedSquare) :
  shaded_to_white_ratio s = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_white_ratio_is_five_thirds_l3575_357515


namespace NUMINAMATH_CALUDE_expected_rounds_is_correct_l3575_357522

/-- Represents the number of rounds in the ball-drawing experiment -/
inductive Round : Type
  | one : Round
  | two : Round
  | three : Round

/-- The probability distribution of the number of rounds -/
def prob (r : Round) : ℚ :=
  match r with
  | Round.one => 1/4
  | Round.two => 1/12
  | Round.three => 2/3

/-- The expected number of rounds -/
def expected_rounds : ℚ := 29/12

/-- Theorem stating that the expected number of rounds is 29/12 -/
theorem expected_rounds_is_correct :
  (prob Round.one * 1 + prob Round.two * 2 + prob Round.three * 3 : ℚ) = expected_rounds := by
  sorry


end NUMINAMATH_CALUDE_expected_rounds_is_correct_l3575_357522


namespace NUMINAMATH_CALUDE_election_combinations_l3575_357539

def number_of_students : ℕ := 6
def number_of_positions : ℕ := 3

theorem election_combinations :
  (number_of_students * (number_of_students - 1) * (number_of_students - 2) = 120) :=
by sorry

end NUMINAMATH_CALUDE_election_combinations_l3575_357539


namespace NUMINAMATH_CALUDE_emilee_earnings_l3575_357509

/-- Given the earnings of three people with specific conditions, prove Emilee's earnings. -/
theorem emilee_earnings (total : ℕ) (terrence_earnings : ℕ) (jermaine_extra : ℕ) :
  total = 90 →
  terrence_earnings = 30 →
  jermaine_extra = 5 →
  ∃ emilee_earnings : ℕ,
    emilee_earnings = total - (terrence_earnings + (terrence_earnings + jermaine_extra)) ∧
    emilee_earnings = 25 :=
by sorry

end NUMINAMATH_CALUDE_emilee_earnings_l3575_357509


namespace NUMINAMATH_CALUDE_list_number_fraction_l3575_357554

theorem list_number_fraction (S : ℝ) (n : ℝ) :
  n = 7 * (S / 50) →
  n / (S + n) = 7 / 57 := by
  sorry

end NUMINAMATH_CALUDE_list_number_fraction_l3575_357554


namespace NUMINAMATH_CALUDE_shopping_mall_goods_problem_l3575_357542

/-- Shopping mall goods problem -/
theorem shopping_mall_goods_problem 
  (total_cost_A : ℝ) 
  (total_cost_B : ℝ) 
  (cost_diff : ℝ) 
  (selling_price_A : ℝ) 
  (selling_price_B : ℝ) 
  (discount_rate : ℝ) 
  (min_profit : ℝ)
  (h1 : total_cost_A = 2000)
  (h2 : total_cost_B = 2400)
  (h3 : cost_diff = 8)
  (h4 : selling_price_A = 60)
  (h5 : selling_price_B = 88)
  (h6 : discount_rate = 0.3)
  (h7 : min_profit = 2460)
  : ∃ (cost_price_A cost_price_B : ℝ) (min_units_A : ℕ),
    cost_price_A = 40 ∧ 
    cost_price_B = 48 ∧ 
    min_units_A = 20 ∧
    (total_cost_A / cost_price_A = total_cost_B / cost_price_B) ∧
    (selling_price_A - cost_price_A) * min_units_A + 
    (selling_price_A * (1 - discount_rate) - cost_price_A) * (total_cost_A / cost_price_A - min_units_A) + 
    (selling_price_B - cost_price_B) * (total_cost_B / cost_price_B) ≥ min_profit :=
by sorry

end NUMINAMATH_CALUDE_shopping_mall_goods_problem_l3575_357542


namespace NUMINAMATH_CALUDE_least_number_with_special_property_l3575_357546

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by another number -/
def is_divisible_by (n m : ℕ) : Prop := sorry

/-- The least positive integer whose digits add to a multiple of 27 yet the number itself is not a multiple of 27 -/
theorem least_number_with_special_property : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), sum_of_digits n = 27 * k) ∧ 
  ¬(is_divisible_by n 27) ∧
  (∀ (m : ℕ), m < n → 
    ((∃ (k : ℕ), sum_of_digits m = 27 * k) → (is_divisible_by m 27))) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_special_property_l3575_357546


namespace NUMINAMATH_CALUDE_walkway_time_when_stopped_l3575_357552

/-- The time it takes to walk a moving walkway when it's stopped -/
theorem walkway_time_when_stopped 
  (length : ℝ) 
  (time_with : ℝ) 
  (time_against : ℝ) 
  (h1 : length = 60) 
  (h2 : time_with = 30) 
  (h3 : time_against = 120) : 
  (2 * length) / (length / time_with + length / time_against) = 48 := by
  sorry

end NUMINAMATH_CALUDE_walkway_time_when_stopped_l3575_357552


namespace NUMINAMATH_CALUDE_beth_sheep_count_l3575_357562

theorem beth_sheep_count (aaron_sheep : ℕ) (beth_sheep : ℕ) 
  (h1 : aaron_sheep = 7 * beth_sheep) 
  (h2 : aaron_sheep + beth_sheep = 608) : 
  beth_sheep = 76 := by
sorry

end NUMINAMATH_CALUDE_beth_sheep_count_l3575_357562


namespace NUMINAMATH_CALUDE_always_quadratic_in_x_l3575_357519

theorem always_quadratic_in_x (k : ℝ) :
  ∃ a b c : ℝ, a ≠ 0 ∧
  ∀ x : ℝ, (k^2 + 1) * x^2 - (k * x - 8) - 1 = a * x^2 + b * x + c :=
by sorry

end NUMINAMATH_CALUDE_always_quadratic_in_x_l3575_357519
