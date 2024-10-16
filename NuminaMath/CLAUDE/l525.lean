import Mathlib

namespace NUMINAMATH_CALUDE_frank_final_position_l525_52502

/-- Calculates Frank's final position after a series of dance moves --/
def frankPosition (initialBackSteps : ℤ) (firstForwardSteps : ℤ) (secondBackSteps : ℤ) : ℤ :=
  -initialBackSteps + firstForwardSteps - secondBackSteps + 2 * secondBackSteps

/-- Proves that Frank ends up 7 steps forward from his original starting point --/
theorem frank_final_position :
  frankPosition 5 10 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_frank_final_position_l525_52502


namespace NUMINAMATH_CALUDE_solution_sets_equality_l525_52586

theorem solution_sets_equality (b c : ℝ) : 
  (∀ x : ℝ, |2*x - 3| < 5 ↔ -x^2 + b*x + c > 0) → b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_equality_l525_52586


namespace NUMINAMATH_CALUDE_boston_snow_depth_l525_52531

/-- The amount of snow in Boston after five days of winter -/
theorem boston_snow_depth :
  let initial_snow : ℝ := 0.5  -- Initial snow in feet
  let second_day_snow : ℝ := 8 / 12  -- 8 inches converted to feet
  let melted_snow : ℝ := 2 / 12  -- 2 inches melted, converted to feet
  let fifth_day_snow : ℝ := 2 * initial_snow  -- Twice the initial snow
  
  initial_snow + second_day_snow - melted_snow + fifth_day_snow = 2 :=
by
  sorry

#check boston_snow_depth

end NUMINAMATH_CALUDE_boston_snow_depth_l525_52531


namespace NUMINAMATH_CALUDE_f_increasing_implies_f_one_geq_25_l525_52579

/-- A function f that is quadratic with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- Theorem stating that if f is increasing on [-2, +∞), then f(1) ≥ 25 -/
theorem f_increasing_implies_f_one_geq_25 (m : ℝ) 
  (h : ∀ x y, -2 ≤ x ∧ x < y → f m x < f m y) : 
  f m 1 ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_implies_f_one_geq_25_l525_52579


namespace NUMINAMATH_CALUDE_grocery_expense_l525_52589

def monthly_expenses (rent milk education petrol misc groceries savings : ℕ) : Prop :=
  let total_salary := savings * 10
  rent + milk + education + petrol + misc + groceries + savings = total_salary

theorem grocery_expense : 
  ∃ (groceries : ℕ), monthly_expenses 5000 1500 2500 2000 5200 groceries 2300 ∧ groceries = 4500 :=
by sorry

end NUMINAMATH_CALUDE_grocery_expense_l525_52589


namespace NUMINAMATH_CALUDE_stuffed_animals_difference_l525_52509

theorem stuffed_animals_difference (thor jake quincy : ℕ) : 
  quincy = 100 * (thor + jake) →
  jake = 2 * thor + 15 →
  quincy = 4000 →
  quincy - jake = 3969 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_difference_l525_52509


namespace NUMINAMATH_CALUDE_tv_final_price_l525_52595

/-- Calculates the final price after applying successive discounts -/
def final_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (fun price discount => price * (1 - discount)) original_price

/-- Proves that the final price of a $450 TV after 10%, 20%, and 5% discounts is $307.80 -/
theorem tv_final_price : 
  let original_price : ℝ := 450
  let discounts : List ℝ := [0.1, 0.2, 0.05]
  final_price original_price discounts = 307.80 := by
sorry

#eval final_price 450 [0.1, 0.2, 0.05]

end NUMINAMATH_CALUDE_tv_final_price_l525_52595


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l525_52562

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem intersection_points_theorem (f : ℝ → ℝ) (a : ℝ) 
  (h1 : is_even f)
  (h2 : has_period f 2)
  (h3 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2)
  (h4 : ∃! (p q : ℝ), p ≠ q ∧ f p = p + a ∧ f q = q + a) :
  ∃ k : ℤ, a = 2 * k ∨ a = 2 * k - 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l525_52562


namespace NUMINAMATH_CALUDE_f_properties_l525_52563

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x

theorem f_properties (f : ℝ → ℝ) 
  (h1 : is_odd (λ x => f (x + 1)))
  (h2 : ∀ x, f (x + 4) = f (-x)) :
  is_even f ∧ f 3 = 0 ∧ f 2023 = 0 := by sorry

end NUMINAMATH_CALUDE_f_properties_l525_52563


namespace NUMINAMATH_CALUDE_reciprocal_of_fraction_difference_l525_52566

theorem reciprocal_of_fraction_difference : (((2 : ℚ) / 3 - (3 : ℚ) / 4)⁻¹ : ℚ) = -12 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_fraction_difference_l525_52566


namespace NUMINAMATH_CALUDE_bookcase_length_in_feet_l525_52508

/-- Converts inches to feet -/
def inches_to_feet (inches : ℕ) : ℚ :=
  inches / 12

theorem bookcase_length_in_feet :
  inches_to_feet 48 = 4 :=
by sorry

end NUMINAMATH_CALUDE_bookcase_length_in_feet_l525_52508


namespace NUMINAMATH_CALUDE_tom_ate_three_fruits_l525_52561

/-- The number of fruits Tom ate -/
def fruits_eaten (initial_oranges initial_lemons remaining_fruits : ℕ) : ℕ :=
  initial_oranges + initial_lemons - remaining_fruits

/-- Proof that Tom ate 3 fruits -/
theorem tom_ate_three_fruits :
  fruits_eaten 3 6 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_ate_three_fruits_l525_52561


namespace NUMINAMATH_CALUDE_square_area_with_circles_l525_52583

/-- The area of a square containing six circles arranged in two rows and three columns, 
    where each circle has a radius of 3 units. -/
theorem square_area_with_circles (radius : ℝ) (h : radius = 3) : 
  (3 * (2 * radius))^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_circles_l525_52583


namespace NUMINAMATH_CALUDE_pool_filling_problem_l525_52558

/-- The pool filling problem -/
theorem pool_filling_problem (pool_capacity : ℝ) (both_valves_time : ℝ) (first_valve_time : ℝ) :
  pool_capacity = 12000 →
  both_valves_time = 48 →
  first_valve_time = 120 →
  let first_valve_rate := pool_capacity / first_valve_time
  let both_valves_rate := pool_capacity / both_valves_time
  let second_valve_rate := both_valves_rate - first_valve_rate
  second_valve_rate - first_valve_rate = 50 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_problem_l525_52558


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l525_52521

/-- The number of packs of red bouncy balls -/
def red_packs : ℕ := 6

/-- The number of balls in each pack of red bouncy balls -/
def red_balls_per_pack : ℕ := 12

/-- The number of packs of yellow bouncy balls -/
def yellow_packs : ℕ := 10

/-- The number of balls in each pack of yellow bouncy balls -/
def yellow_balls_per_pack : ℕ := 8

/-- The number of packs of green bouncy balls -/
def green_packs : ℕ := 4

/-- The number of balls in each pack of green bouncy balls -/
def green_balls_per_pack : ℕ := 15

/-- The number of packs of blue bouncy balls -/
def blue_packs : ℕ := 3

/-- The number of balls in each pack of blue bouncy balls -/
def blue_balls_per_pack : ℕ := 20

/-- The total number of bouncy balls Maggie bought -/
def total_bouncy_balls : ℕ := 
  red_packs * red_balls_per_pack + 
  yellow_packs * yellow_balls_per_pack + 
  green_packs * green_balls_per_pack + 
  blue_packs * blue_balls_per_pack

theorem maggie_bouncy_balls : total_bouncy_balls = 272 := by
  sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l525_52521


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l525_52571

/-- Ellipse C with foci F₁ and F₂, and point P on C -/
structure Ellipse :=
  (a b : ℝ)
  (h_ab : a > b ∧ b > 0)
  (F₁ F₂ P : ℝ × ℝ)
  (h_on_ellipse : (P.1^2 / a^2) + (P.2^2 / b^2) = 1)
  (h_perp : (P.1 - F₂.1) * (F₂.1 - F₁.1) + (P.2 - F₂.2) * (F₂.2 - F₁.2) = 0)
  (h_angle : Real.cos (30 * π / 180) = 
    ((P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2)) / 
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)))

/-- The eccentricity of an ellipse is √3/3 -/
theorem ellipse_eccentricity (C : Ellipse) : 
  Real.sqrt ((C.F₂.1 - C.F₁.1)^2 + (C.F₂.2 - C.F₁.2)^2) / (2 * C.a) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l525_52571


namespace NUMINAMATH_CALUDE_picture_placement_l525_52565

/-- Given a wall of width 30 feet, with two pictures each 4 feet wide and spaced 1 foot apart
    hung in the center, the distance from the end of the wall to the nearest edge of the first
    picture is 10.5 feet. -/
theorem picture_placement (wall_width : ℝ) (picture_width : ℝ) (picture_space : ℝ)
  (h_wall : wall_width = 30)
  (h_picture : picture_width = 4)
  (h_space : picture_space = 1) :
  let total_picture_space := 2 * picture_width + picture_space
  (wall_width - total_picture_space) / 2 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_picture_placement_l525_52565


namespace NUMINAMATH_CALUDE_expression_bounds_l525_52591

theorem expression_bounds (x y : ℝ) (h : x^2 + y^2 = 4) :
  1 ≤ 4*(x - 1/2)^2 + (y - 1)^2 + 4*x*y ∧ 
  4*(x - 1/2)^2 + (y - 1)^2 + 4*x*y ≤ 22 + 4*Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l525_52591


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l525_52560

theorem unique_solution_quadratic_inequality (m : ℝ) : 
  (∃! x : ℝ, x^2 - m*x + 1 ≤ 0) → (m = 2 ∨ m = -2) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l525_52560


namespace NUMINAMATH_CALUDE_equal_segments_exist_l525_52547

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) :=
  (vertices : Fin (2*n) → ℝ × ℝ)
  (is_regular : sorry)

/-- A pairing of vertices in a regular polygon -/
def VertexPairing (n : ℕ) := Fin n → Fin (2*n) × Fin (2*n)

/-- The distance between two vertices in a regular polygon -/
def distance (p : RegularPolygon n) (i j : Fin (2*n)) : ℝ := sorry

theorem equal_segments_exist (m : ℕ) (n : ℕ) (h : n = 4*m + 2 ∨ n = 4*m + 3) 
  (p : RegularPolygon n) (pairing : VertexPairing n) : 
  ∃ (i j k l : Fin n), i ≠ j ∧ k ≠ l ∧ i ≠ k ∧ j ≠ l ∧
    distance p (pairing i).1 (pairing i).2 = distance p (pairing k).1 (pairing k).2 :=
sorry

end NUMINAMATH_CALUDE_equal_segments_exist_l525_52547


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l525_52539

/-- Given a geometric sequence with common ratio 2 and sum of first four terms equal to 1,
    the sum of the first eight terms is 17. -/
theorem geometric_sequence_sum (a : ℝ) : 
  (∃ (S₄ S₈ : ℝ), 
    S₄ = a * (1 + 2 + 2^2 + 2^3) ∧
    S₄ = 1 ∧
    S₈ = a * (1 + 2 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7)) →
  (∃ S₈ : ℝ, S₈ = a * (1 + 2 + 2^2 + 2^3 + 2^4 + 2^5 + 2^6 + 2^7) ∧ S₈ = 17) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l525_52539


namespace NUMINAMATH_CALUDE_root_sum_fraction_l525_52513

theorem root_sum_fraction (a b c : ℝ) : 
  a^3 - 15*a^2 + 25*a - 10 = 0 →
  b^3 - 15*b^2 + 25*b - 10 = 0 →
  c^3 - 15*c^2 + 25*c - 10 = 0 →
  (a / (1/a + b*c)) + (b / (1/b + c*a)) + (c / (1/c + a*b)) = 175/11 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l525_52513


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l525_52517

theorem quadratic_solution_property (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0 → 2 * a + 4 * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l525_52517


namespace NUMINAMATH_CALUDE_surface_area_of_T_l525_52516

-- Define the cube
def cube_edge_length : ℝ := 10

-- Define points M, N, O
def point_M : ℝ × ℝ × ℝ := (3, 0, 0)
def point_N : ℝ × ℝ × ℝ := (0, 3, 0)
def point_O : ℝ × ℝ × ℝ := (0, 0, 3)

-- Define the distance from A to M, N, O
def distance_AM : ℝ := 3
def distance_AN : ℝ := 3
def distance_AO : ℝ := 3

-- Function to calculate the area of a triangle given three points in 3D space
def triangle_area (p1 p2 p3 : ℝ × ℝ × ℝ) : ℝ := sorry

-- Function to calculate the surface area of a cube given its edge length
def cube_surface_area (edge_length : ℝ) : ℝ := sorry

-- Theorem: The surface area of solid T is 600 - 27√2
theorem surface_area_of_T :
  let triangle_face_area := triangle_area point_M point_N point_O
  let cube_area := cube_surface_area cube_edge_length
  let removed_area := 3 * triangle_face_area
  cube_area - removed_area = 600 - 27 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_surface_area_of_T_l525_52516


namespace NUMINAMATH_CALUDE_smallest_cover_count_l525_52577

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a rectangular region to be covered -/
structure Region where
  width : ℕ
  height : ℕ

def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

def Region.area (r : Region) : ℕ := r.width * r.height

/-- The number of rectangles needed to cover a region -/
def coverCount (r : Rectangle) (reg : Region) : ℕ :=
  Region.area reg / Rectangle.area r

theorem smallest_cover_count (r : Rectangle) (reg : Region) :
  r.width = 3 ∧ r.height = 4 ∧ reg.width = 12 ∧ reg.height = 12 →
  coverCount r reg = 12 ∧
  ∀ (r' : Rectangle) (reg' : Region),
    r'.width * r'.height ≤ r.width * r.height →
    reg'.width = 12 →
    coverCount r' reg' ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_cover_count_l525_52577


namespace NUMINAMATH_CALUDE_sequence_increasing_iff_last_term_l525_52592

theorem sequence_increasing_iff_last_term (a : ℕ → ℝ) : 
  (∀ n : ℕ, n ≥ 1 ∧ n < 64 → |a (n + 1) - a n| = n) →
  a 1 = 2 →
  (∀ n : ℕ, n ≥ 1 ∧ n < 64 → a (n + 1) > a n) ↔ a 64 = 2018 :=
by sorry

end NUMINAMATH_CALUDE_sequence_increasing_iff_last_term_l525_52592


namespace NUMINAMATH_CALUDE_smallest_x_value_l525_52585

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = (y : ℚ) / ((250 : ℚ) + x)) :
  x ≥ 2 ∧ ∃ (y' : ℕ+), (3 : ℚ) / 4 = (y' : ℚ) / ((250 : ℚ) + 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l525_52585


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l525_52587

/-- The minimum distance between points on y = x^2 + 1 and y = √(x - 1) -/
theorem min_distance_between_curves : 
  let P : ℝ × ℝ → Prop := λ p => ∃ x : ℝ, x ≥ 0 ∧ p = (x, x^2 + 1)
  let Q : ℝ × ℝ → Prop := λ q => ∃ y : ℝ, y ≥ 1 ∧ q = (y, Real.sqrt (y - 1))
  ∀ p q : ℝ × ℝ, P p → Q q → 
    ∃ p' q' : ℝ × ℝ, P p' ∧ Q q' ∧ 
      Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) = 3 * Real.sqrt 2 / 4 ∧
      ∀ p'' q'' : ℝ × ℝ, P p'' → Q q'' → 
        Real.sqrt ((p''.1 - q''.1)^2 + (p''.2 - q''.2)^2) ≥ 3 * Real.sqrt 2 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_between_curves_l525_52587


namespace NUMINAMATH_CALUDE_sine_cosine_cube_difference_l525_52519

theorem sine_cosine_cube_difference (α : ℝ) (n : ℝ) 
  (h : Real.sin α - Real.cos α = n) : 
  Real.sin α ^ 3 - Real.cos α ^ 3 = (3 * n - n^3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_cube_difference_l525_52519


namespace NUMINAMATH_CALUDE_initial_bananas_count_l525_52507

/-- The number of bananas Raj has eaten -/
def bananas_eaten : ℕ := 70

/-- The number of bananas left on the tree after Raj cut some -/
def bananas_left_on_tree : ℕ := 100

/-- The number of bananas remaining in Raj's basket -/
def bananas_in_basket : ℕ := 2 * bananas_eaten

/-- The total number of bananas Raj cut from the tree -/
def bananas_cut : ℕ := bananas_eaten + bananas_in_basket

/-- The initial number of bananas on the tree -/
def initial_bananas : ℕ := bananas_cut + bananas_left_on_tree

theorem initial_bananas_count : initial_bananas = 310 := by
  sorry

end NUMINAMATH_CALUDE_initial_bananas_count_l525_52507


namespace NUMINAMATH_CALUDE_work_completion_time_l525_52528

theorem work_completion_time (x : ℝ) : 
  x > 0 →  -- p's completion time is positive
  (2 / x + 3 * (1 / x + 1 / 6) = 1) →  -- work equation
  x = 10 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l525_52528


namespace NUMINAMATH_CALUDE_jose_peanut_count_l525_52544

-- Define the number of peanuts Kenya has
def kenya_peanuts : ℕ := 133

-- Define the difference between Kenya's and Jose's peanuts
def peanut_difference : ℕ := 48

-- Define Jose's peanuts
def jose_peanuts : ℕ := kenya_peanuts - peanut_difference

-- Theorem statement
theorem jose_peanut_count : jose_peanuts = 85 := by sorry

end NUMINAMATH_CALUDE_jose_peanut_count_l525_52544


namespace NUMINAMATH_CALUDE_solution_sets_intersection_and_union_l525_52574

def equation1 (p : ℝ) (x : ℝ) : Prop := x^2 - p*x + 6 = 0

def equation2 (q : ℝ) (x : ℝ) : Prop := x^2 + 6*x - q = 0

def solution_set (equation : ℝ → Prop) : Set ℝ :=
  {x | equation x}

theorem solution_sets_intersection_and_union
  (p q : ℝ)
  (M : Set ℝ)
  (N : Set ℝ)
  (h1 : M = solution_set (equation1 p))
  (h2 : N = solution_set (equation2 q))
  (h3 : M ∩ N = {2}) :
  p = 5 ∧ q = 16 ∧ M ∪ N = {2, 3, -8} := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_intersection_and_union_l525_52574


namespace NUMINAMATH_CALUDE_f_properties_l525_52564

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -4 * x^2 else x^2 - x

theorem f_properties :
  (∃ a : ℝ, f a = -1/4 ∧ (a = -1/4 ∨ a = 1/2)) ∧
  (∃ b : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f x - b = 0 ∧ f y - b = 0 ∧ f z - b = 0) →
    -1/4 < b ∧ b < 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l525_52564


namespace NUMINAMATH_CALUDE_nina_spiders_count_l525_52551

/-- Proves that Nina has 3 spiders given the conditions of the problem -/
theorem nina_spiders_count :
  ∀ (spiders : ℕ),
  (∃ (total_eyes : ℕ),
    total_eyes = 124 ∧
    total_eyes = 8 * spiders + 2 * 50) →
  spiders = 3 := by
sorry

end NUMINAMATH_CALUDE_nina_spiders_count_l525_52551


namespace NUMINAMATH_CALUDE_fourteen_distinct_patterns_l525_52598

/-- Represents a pattern on a 4x4 grid --/
def Pattern := Fin 16 → Bool

/-- Checks if two patterns are equivalent under rotations and reflections --/
def equivalent (p q : Pattern) : Prop := sorry

/-- Counts the number of distinct patterns with exactly 3 shaded squares --/
def distinctPatterns : ℕ := sorry

/-- The main theorem: There are exactly 14 distinct patterns --/
theorem fourteen_distinct_patterns : distinctPatterns = 14 := by sorry

end NUMINAMATH_CALUDE_fourteen_distinct_patterns_l525_52598


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l525_52578

-- Define the lines
def line1 (x y : ℝ) : Prop := y = 3 * x + 14
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = 40

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = -68 := by sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l525_52578


namespace NUMINAMATH_CALUDE_distribute_5_3_l525_52556

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l525_52556


namespace NUMINAMATH_CALUDE_fraction_value_l525_52584

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) 
  (h4 : b ≠ 0) 
  (h5 : d ≠ 0) : 
  a * c / (b * d) = 20 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_l525_52584


namespace NUMINAMATH_CALUDE_square_side_length_l525_52594

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) : 
  ∃ s : ℝ, s * s = d * d / 2 ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l525_52594


namespace NUMINAMATH_CALUDE_identity_is_increasing_proportional_l525_52529

/-- A proportional function where y increases as x increases -/
def increasing_proportional_function (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∃ k : ℝ, ∀ x : ℝ, f x = k * x)

/-- The function f(x) = x is an increasing proportional function -/
theorem identity_is_increasing_proportional : increasing_proportional_function (λ x : ℝ => x) := by
  sorry


end NUMINAMATH_CALUDE_identity_is_increasing_proportional_l525_52529


namespace NUMINAMATH_CALUDE_tan_alpha_4_implies_fraction_9_l525_52550

theorem tan_alpha_4_implies_fraction_9 (α : Real) (h : Real.tan α = 4) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_4_implies_fraction_9_l525_52550


namespace NUMINAMATH_CALUDE_system_solution_l525_52568

theorem system_solution (x y m : ℚ) : 
  (2 * x + 3 * y = 4) → 
  (3 * x + 2 * y = 2 * m - 3) → 
  (x + y = -3/5) → 
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l525_52568


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l525_52582

theorem quadratic_equation_roots_ratio (m : ℝ) : 
  (∃ (r s : ℝ), r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 / 2 ∧ 
   r + s = 5 ∧ r * s = m ∧ 
   ∀ x : ℝ, x^2 - 5*x + m = 0 ↔ (x = r ∨ x = s)) → 
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l525_52582


namespace NUMINAMATH_CALUDE_provisions_last_five_days_l525_52532

/-- Calculates the number of days provisions will last after new groups join -/
def remaining_days (initial_men : ℕ) (initial_days : ℕ) (initial_rate : ℚ)
  (days_before_join : ℕ) (high_metabolism_men : ℕ) (high_metabolism_rate : ℚ)
  (special_diet_men : ℕ) (special_diet_rate : ℚ) : ℕ :=
  sorry

/-- Theorem stating that given the problem conditions, the provisions will last 5 more days -/
theorem provisions_last_five_days :
  remaining_days 1500 17 2 10 280 3 40 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_provisions_last_five_days_l525_52532


namespace NUMINAMATH_CALUDE_complex_real_condition_l525_52536

theorem complex_real_condition (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (2 - Complex.I)
  (∃ (x : ℝ), z = x) → a = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_real_condition_l525_52536


namespace NUMINAMATH_CALUDE_linear_function_m_value_l525_52514

/-- Linear function passing through a point -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x - 4

/-- Theorem: For a linear function y = (m-1)x - 4 passing through (2, 4), m = 5 -/
theorem linear_function_m_value :
  ∃ (m : ℝ), linear_function m 2 = 4 ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_m_value_l525_52514


namespace NUMINAMATH_CALUDE_master_bathroom_towel_price_l525_52555

/-- The price of towel sets for the master bathroom, given the following conditions:
  * 2 sets of towels for guest bathroom and 4 sets for master bathroom are bought
  * Guest bathroom towel sets cost $40.00 each
  * The store offers a 20% discount
  * The total spent on towel sets is $224
-/
theorem master_bathroom_towel_price :
  ∀ (x : ℝ),
    2 * 40 * (1 - 0.2) + 4 * x * (1 - 0.2) = 224 →
    x = 50 := by
  sorry

end NUMINAMATH_CALUDE_master_bathroom_towel_price_l525_52555


namespace NUMINAMATH_CALUDE_area_equals_perimeter_count_l525_52541

/-- A structure representing a rectangle with integer sides -/
structure Rectangle where
  a : ℕ
  b : ℕ

/-- A structure representing a right triangle with integer sides -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The area of a rectangle is equal to its perimeter -/
def Rectangle.areaEqualsPerimeter (r : Rectangle) : Prop :=
  r.a * r.b = 2 * (r.a + r.b)

/-- The area of a right triangle is equal to its perimeter -/
def RightTriangle.areaEqualsPerimeter (t : RightTriangle) : Prop :=
  t.a * t.b = 2 * (t.a + t.b + t.c)

/-- The sides of a right triangle satisfy the Pythagorean theorem -/
def RightTriangle.isPythagorean (t : RightTriangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

/-- The main theorem stating the number of rectangles and right triangles that satisfy the conditions -/
theorem area_equals_perimeter_count :
  (∃! (rs : Finset Rectangle), ∀ r ∈ rs, r.areaEqualsPerimeter ∧ rs.card = 2) ∧
  (∃! (ts : Finset RightTriangle), ∀ t ∈ ts, t.areaEqualsPerimeter ∧ t.isPythagorean ∧ ts.card = 1) := by
  sorry


end NUMINAMATH_CALUDE_area_equals_perimeter_count_l525_52541


namespace NUMINAMATH_CALUDE_age_difference_l525_52597

theorem age_difference (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (10 * a + b + 5) = 3 * (10 * b + a + 5)) :
  (10 * a + b) - (10 * b + a) = 45 :=
sorry

end NUMINAMATH_CALUDE_age_difference_l525_52597


namespace NUMINAMATH_CALUDE_tina_total_time_l525_52596

/-- The time it takes to clean one key, in minutes -/
def time_per_key : ℕ := 3

/-- The number of keys left to clean -/
def keys_to_clean : ℕ := 14

/-- The time it takes to complete the assignment, in minutes -/
def assignment_time : ℕ := 10

/-- The total time it takes for Tina to clean the remaining keys and finish her assignment -/
def total_time : ℕ := time_per_key * keys_to_clean + assignment_time

theorem tina_total_time : total_time = 52 := by
  sorry

end NUMINAMATH_CALUDE_tina_total_time_l525_52596


namespace NUMINAMATH_CALUDE_max_daily_revenue_l525_52593

def sales_price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then -t + 100
  else if 25 ≤ t ∧ t ≤ 30 then t + 20
  else 0

def daily_sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def daily_revenue (t : ℕ) : ℝ := sales_price t * daily_sales_volume t

theorem max_daily_revenue :
  (∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ daily_revenue t = 1125) ∧
  (∀ t : ℕ, 0 < t ∧ t ≤ 30 → daily_revenue t ≤ 1125) ∧
  (∀ t : ℕ, 0 < t ∧ t ≤ 30 ∧ daily_revenue t = 1125 → t = 25) :=
sorry

end NUMINAMATH_CALUDE_max_daily_revenue_l525_52593


namespace NUMINAMATH_CALUDE_smallest_angle_range_l525_52557

theorem smallest_angle_range (A B C : Real) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) 
  (h4 : A + B + C = 180) : 
  ∃ α : Real, (α = min A (min B C)) ∧ (0 < α) ∧ (α ≤ 60) := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_range_l525_52557


namespace NUMINAMATH_CALUDE_a_range_is_open_2_5_l525_52505

-- Define the sequence a_n
def a_n (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 5 then (5 - a) * n - 11 else a ^ (n - 4)

-- Theorem statement
theorem a_range_is_open_2_5 :
  ∀ a : ℝ, (∀ n : ℕ, a_n a n < a_n a (n + 1)) →
  (2 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_a_range_is_open_2_5_l525_52505


namespace NUMINAMATH_CALUDE_total_sightings_is_280_l525_52503

/-- Represents the data for a single month in the national park -/
structure MonthData where
  families : ℕ
  sightings : ℕ

/-- Calculates the total number of animal sightings over six months -/
def totalSightings (jan feb mar apr may jun : MonthData) : ℕ :=
  jan.sightings + feb.sightings + mar.sightings + apr.sightings + may.sightings + jun.sightings

/-- Theorem stating that the total number of animal sightings is 280 -/
theorem total_sightings_is_280 
  (jan : MonthData)
  (feb : MonthData)
  (mar : MonthData)
  (apr : MonthData)
  (may : MonthData)
  (jun : MonthData)
  (h1 : jan.families = 100 ∧ jan.sightings = 26)
  (h2 : feb.families = 150 ∧ feb.sightings = 78)
  (h3 : mar.families = 120 ∧ mar.sightings = 39)
  (h4 : apr.families = 204 ∧ apr.sightings = 55)
  (h5 : may.families = 204 ∧ may.sightings = 41)
  (h6 : jun.families = 265 ∧ jun.sightings = 41) :
  totalSightings jan feb mar apr may jun = 280 := by
  sorry

#check total_sightings_is_280

end NUMINAMATH_CALUDE_total_sightings_is_280_l525_52503


namespace NUMINAMATH_CALUDE_series_sum_l525_52588

def series (n : ℕ) : ℚ :=
  if n = 0 then 2
  else if n = 1 then 3 + (1/2) * (series 0)
  else (1005 - n + 1 : ℚ) + (1/2) * (series (n-1))

theorem series_sum : series 1003 = 2008 := by sorry

end NUMINAMATH_CALUDE_series_sum_l525_52588


namespace NUMINAMATH_CALUDE_probability_of_no_three_consecutive_ones_l525_52548

/-- Represents the number of valid sequences of length n -/
def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 4
| n + 3 => b (n + 2) + b (n + 1) + b n

/-- The probability of a 12-element sequence not containing three consecutive 1s -/
def probability : ℚ := b 12 / 2^12

theorem probability_of_no_three_consecutive_ones : probability = 281 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_no_three_consecutive_ones_l525_52548


namespace NUMINAMATH_CALUDE_usual_bus_time_l525_52524

/-- The usual time to catch the bus, given that walking at 4/5 of the usual speed results in missing the bus by 3 minutes, is 12 minutes. -/
theorem usual_bus_time (usual_speed : ℝ) (usual_time : ℝ) : 
  (4 / 5 * usual_speed * (usual_time + 3) = usual_speed * usual_time) → 
  usual_time = 12 := by
sorry

end NUMINAMATH_CALUDE_usual_bus_time_l525_52524


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l525_52567

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 8) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) ≤ 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l525_52567


namespace NUMINAMATH_CALUDE_intersection_implies_z_l525_52549

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the sets M and N
def M (z : ℂ) : Set ℂ := {1, 2, z * i}
def N : Set ℂ := {3, 4}

-- State the theorem
theorem intersection_implies_z (z : ℂ) : M z ∩ N = {4} → z = -4 * i := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_z_l525_52549


namespace NUMINAMATH_CALUDE_a_range_l525_52572

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x : ℝ, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (a*x^2 - x + a)

-- Define the theorem
theorem a_range (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → (a ≥ 1/2 ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_a_range_l525_52572


namespace NUMINAMATH_CALUDE_jason_newspaper_earnings_l525_52530

-- Define the initial and final amounts for Jason
def jason_initial : ℕ := 3
def jason_final : ℕ := 63

-- Define Jason's earnings
def jason_earnings : ℕ := jason_final - jason_initial

-- Theorem to prove
theorem jason_newspaper_earnings :
  jason_earnings = 60 := by
  sorry

end NUMINAMATH_CALUDE_jason_newspaper_earnings_l525_52530


namespace NUMINAMATH_CALUDE_complex_equation_sum_l525_52520

theorem complex_equation_sum (a b : ℝ) : 
  (a : ℂ) + b * Complex.I = (1 + Complex.I) * (2 - Complex.I) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l525_52520


namespace NUMINAMATH_CALUDE_original_student_count_l525_52580

/-- Prove that given the initial average weight, new student's weight, and new average weight,
    the number of original students is 29. -/
theorem original_student_count
  (initial_avg : ℝ)
  (new_student_weight : ℝ)
  (new_avg : ℝ)
  (h1 : initial_avg = 28)
  (h2 : new_student_weight = 22)
  (h3 : new_avg = 27.8)
  : ∃ n : ℕ, n = 29 ∧ 
    (n : ℝ) * initial_avg + new_student_weight = (n + 1 : ℝ) * new_avg :=
by
  sorry

end NUMINAMATH_CALUDE_original_student_count_l525_52580


namespace NUMINAMATH_CALUDE_partnership_investment_l525_52518

/-- A partnership problem where three partners invest different amounts and receive different shares. -/
theorem partnership_investment (a_investment b_investment c_investment : ℝ)
  (b_share a_share : ℝ) :
  b_investment = 11000 →
  c_investment = 18000 →
  b_share = 2200 →
  a_share = 1400 →
  (b_share / b_investment = a_share / a_investment) →
  a_investment = 7000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_l525_52518


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l525_52590

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l525_52590


namespace NUMINAMATH_CALUDE_total_interest_calculation_l525_52559

/-- Calculates the total interest earned from two bank investments -/
theorem total_interest_calculation 
  (total_investment : ℝ) 
  (bank1_rate : ℝ) 
  (bank2_rate : ℝ) 
  (bank1_investment : ℝ) 
  (h1 : total_investment = 5000)
  (h2 : bank1_rate = 0.04)
  (h3 : bank2_rate = 0.065)
  (h4 : bank1_investment = 1700) :
  let bank2_investment := total_investment - bank1_investment
  let interest1 := bank1_investment * bank1_rate
  let interest2 := bank2_investment * bank2_rate
  interest1 + interest2 = 282.50 := by
sorry

end NUMINAMATH_CALUDE_total_interest_calculation_l525_52559


namespace NUMINAMATH_CALUDE_ABC_equality_l525_52500

variables (u v w : ℝ)
variables (A B C : ℝ)

def A_def : A = u * v + u + 1 := by sorry
def B_def : B = v * w + v + 1 := by sorry
def C_def : C = w * u + w + 1 := by sorry
def uvw_condition : u * v * w = 1 := by sorry

theorem ABC_equality : A * B * C = A * B + B * C + C * A := by sorry

end NUMINAMATH_CALUDE_ABC_equality_l525_52500


namespace NUMINAMATH_CALUDE_inequality_problem_l525_52543

theorem inequality_problem (m : ℝ) (h : ∀ x : ℝ, |x - 2| + |x - 3| ≥ m) :
  (∃ k : ℝ, k = 1 ∧ (∀ m' : ℝ, (∀ x : ℝ, |x - 2| + |x - 3| ≥ m') → m' ≤ k)) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 1/a + 1/(2*b) + 1/(3*c) = 1 → a + 2*b + 3*c ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l525_52543


namespace NUMINAMATH_CALUDE_math_books_prob_theorem_l525_52540

/-- The probability of all three mathematics textbooks ending up in the same box -/
def math_books_same_box_prob (total_books n_math_books : ℕ) 
  (box_sizes : Fin 3 → ℕ) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem math_books_prob_theorem :
  let total_books : ℕ := 15
  let n_math_books : ℕ := 3
  let box_sizes : Fin 3 → ℕ := ![4, 5, 6]
  math_books_same_box_prob total_books n_math_books box_sizes = 9 / 121 :=
sorry

end NUMINAMATH_CALUDE_math_books_prob_theorem_l525_52540


namespace NUMINAMATH_CALUDE_percentage_of_hindu_boys_l525_52569

theorem percentage_of_hindu_boys (total_boys : ℕ) (muslim_percent : ℚ) (sikh_percent : ℚ) (other_boys : ℕ) : 
  total_boys = 650 →
  muslim_percent = 44 / 100 →
  sikh_percent = 10 / 100 →
  other_boys = 117 →
  (total_boys - (muslim_percent * total_boys + sikh_percent * total_boys + other_boys)) / total_boys = 28 / 100 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_hindu_boys_l525_52569


namespace NUMINAMATH_CALUDE_harry_potter_book_price_l525_52511

theorem harry_potter_book_price : 
  ∀ (wang_money li_money book_price : ℕ),
  wang_money + 6 = 2 * book_price →
  li_money + 31 = 2 * book_price →
  wang_money + li_money = 3 * book_price →
  book_price = 37 := by
sorry

end NUMINAMATH_CALUDE_harry_potter_book_price_l525_52511


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l525_52535

/-- The line passing through the intersection points of two circles -/
theorem intersection_line_of_circles (x y : ℝ) : 
  (x - 2)^2 + (y + 3)^2 = 8^2 →
  (x + 5)^2 + (y - 7)^2 = 136 →
  x + y = 4.35 :=
by
  sorry


end NUMINAMATH_CALUDE_intersection_line_of_circles_l525_52535


namespace NUMINAMATH_CALUDE_four_layer_grid_triangles_l525_52573

/-- Calculates the total number of triangles in a triangular grid with a given number of layers. -/
def triangles_in_grid (layers : ℕ) : ℕ :=
  let small_triangles := (layers * (layers + 1)) / 2
  let medium_triangles := if layers ≥ 3 then (layers - 2) * (layers - 1) / 2 else 0
  let large_triangles := 1
  small_triangles + medium_triangles + large_triangles

/-- Theorem stating that a triangular grid with 4 layers contains 21 triangles. -/
theorem four_layer_grid_triangles :
  triangles_in_grid 4 = 21 :=
by sorry

end NUMINAMATH_CALUDE_four_layer_grid_triangles_l525_52573


namespace NUMINAMATH_CALUDE_symmetric_points_l525_52542

/-- Given that point A(2, 4) is symmetric to point B(b-1, 2a) with respect to the origin, prove that a - b = -1 -/
theorem symmetric_points (a b : ℝ) : 
  (2 = -(b - 1) ∧ 4 = -2*a) → a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_l525_52542


namespace NUMINAMATH_CALUDE_ratio_problem_l525_52576

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y) = 1 / 2) : 
  x / y = 7 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l525_52576


namespace NUMINAMATH_CALUDE_f_is_quadratic_l525_52545

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: f is a quadratic equation in one variable -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l525_52545


namespace NUMINAMATH_CALUDE_strawberry_jelly_amount_l525_52533

/-- Given the total amount of jelly and the amount of blueberry jelly, 
    calculate the amount of strawberry jelly. -/
theorem strawberry_jelly_amount 
  (total_jelly : ℕ) 
  (blueberry_jelly : ℕ) 
  (h1 : total_jelly = 6310)
  (h2 : blueberry_jelly = 4518) : 
  total_jelly - blueberry_jelly = 1792 := by
sorry

end NUMINAMATH_CALUDE_strawberry_jelly_amount_l525_52533


namespace NUMINAMATH_CALUDE_smallest_integer_in_A_l525_52510

def A : Set ℝ := {x | |x - 2| ≤ 5}

theorem smallest_integer_in_A : 
  ∃ (n : ℤ), (n : ℝ) ∈ A ∧ ∀ (m : ℤ), (m : ℝ) ∈ A → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_A_l525_52510


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l525_52526

-- Part 1
theorem problem_one : (2 * Real.sqrt 3 - 1)^2 + (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = 12 - 4 * Real.sqrt 3 := by
  sorry

-- Part 2
theorem problem_two : (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - 6 * Real.sqrt (1/2) = -6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l525_52526


namespace NUMINAMATH_CALUDE_smallest_d_for_3150_square_l525_52554

/-- The smallest positive integer d such that 3150 * d is a perfect square is 14 -/
theorem smallest_d_for_3150_square : ∃ (n : ℕ), 
  (3150 * 14 = n ^ 2) ∧ 
  (∀ (d : ℕ), d > 0 ∧ d < 14 → ¬∃ (m : ℕ), 3150 * d = m ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_d_for_3150_square_l525_52554


namespace NUMINAMATH_CALUDE_groom_age_l525_52504

theorem groom_age (bride_age groom_age : ℕ) : 
  bride_age = groom_age + 19 →
  bride_age + groom_age = 185 →
  groom_age = 83 := by
sorry

end NUMINAMATH_CALUDE_groom_age_l525_52504


namespace NUMINAMATH_CALUDE_milk_parts_in_drink_A_l525_52525

/-- Represents the composition of a drink mixture -/
structure DrinkMixture where
  milk : ℕ
  fruit_juice : ℕ

/-- Converts volume in parts to liters -/
def parts_to_liters (total_parts : ℕ) (volume_liters : ℕ) (parts : ℕ) : ℕ :=
  (volume_liters * parts) / total_parts

theorem milk_parts_in_drink_A (drink_A : DrinkMixture) (drink_B : DrinkMixture) : 
  drink_A.fruit_juice = 3 →
  drink_B.milk = 3 →
  drink_B.fruit_juice = 4 →
  parts_to_liters (drink_A.milk + drink_A.fruit_juice) 21 drink_A.fruit_juice +
    7 = parts_to_liters (drink_B.milk + drink_B.fruit_juice) 28 drink_B.fruit_juice →
  drink_A.milk = 12 := by
  sorry

end NUMINAMATH_CALUDE_milk_parts_in_drink_A_l525_52525


namespace NUMINAMATH_CALUDE_square_side_length_l525_52527

theorem square_side_length (s : ℝ) : s > 0 → s^2 = 3 * (4 * s) → s = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l525_52527


namespace NUMINAMATH_CALUDE_coefficient_x6_in_expansion_l525_52522

theorem coefficient_x6_in_expansion : 
  (Finset.range 5).sum (fun k => 
    (Nat.choose 4 k : ℝ) * (1 : ℝ)^(4 - k) * (3 : ℝ)^k * 
    if k = 2 then 1 else 0) = 54 := by sorry

end NUMINAMATH_CALUDE_coefficient_x6_in_expansion_l525_52522


namespace NUMINAMATH_CALUDE_PQ_length_l525_52538

-- Define the triangles and their properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle : ℝ

-- Define the given triangles
def triangle_PQR : Triangle := {
  a := 8,   -- PR
  b := 10,  -- QR
  c := 5,   -- PQ (to be proved)
  angle := 60
}

def triangle_STU : Triangle := {
  a := 3,   -- SU
  b := 4,   -- TU (derived from similarity)
  c := 2,   -- ST
  angle := 60
}

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop :=
  t1.angle = t2.angle ∧ t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

-- Theorem statement
theorem PQ_length :
  similar triangle_PQR triangle_STU →
  triangle_PQR.c = 5 := by sorry

end NUMINAMATH_CALUDE_PQ_length_l525_52538


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l525_52581

theorem triangle_angle_problem (A B C : Real) (a b c : Real) :
  (A + B + C = π) →
  (a * Real.cos B - b * Real.cos A = c) →
  (C = π / 5) →
  (B = 3 * π / 10) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l525_52581


namespace NUMINAMATH_CALUDE_olympiad_problem_distribution_l525_52546

theorem olympiad_problem_distribution (n : ℕ) (m : ℕ) (k : ℕ) 
  (h1 : n = 30) 
  (h2 : m = 40) 
  (h3 : k = 5) 
  (h4 : ∃ (x y z q r : ℕ), 
    x + y + z + q + r = n ∧ 
    x + 2*y + 3*z + 4*q + 5*r = m ∧ 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ q > 0 ∧ r > 0) :
  ∃ (x : ℕ), x = 26 ∧ 
    ∃ (y z q r : ℕ), 
      x + y + z + q + r = n ∧ 
      x + 2*y + 3*z + 4*q + 5*r = m ∧
      y = 1 ∧ z = 1 ∧ q = 1 ∧ r = 1 := by
  sorry

end NUMINAMATH_CALUDE_olympiad_problem_distribution_l525_52546


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l525_52506

/-- Proves that it takes 2 years for a man's age to be twice his son's age -/
theorem mans_age_twice_sons (
  son_age : ℕ) 
  (man_age : ℕ) 
  (h1 : son_age = 20) 
  (h2 : man_age = son_age + 22) : 
  ∃ y : ℕ, y = 2 ∧ man_age + y = 2 * (son_age + y) :=
sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l525_52506


namespace NUMINAMATH_CALUDE_projection_scalar_multiple_l525_52570

def proj_w (v : ℝ × ℝ) (w : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_scalar_multiple (v w : ℝ × ℝ) :
  proj_w v w = (4, 3) → proj_w (7 • v) w = (28, 21) := by
  sorry

end NUMINAMATH_CALUDE_projection_scalar_multiple_l525_52570


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l525_52523

/-- Given a geometric sequence {aₙ} with a₁ = 3 and common ratio q = √2, prove that a₇ = 24 -/
theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℝ), 
    (a 1 = 3) →
    (∀ n : ℕ, a (n + 1) = a n * Real.sqrt 2) →
    (a 7 = 24) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l525_52523


namespace NUMINAMATH_CALUDE_complex_equation_solution_l525_52515

/-- Given that (1-i)z = 3+i, prove that z = 1 + 2i -/
theorem complex_equation_solution (z : ℂ) (h : (1 - Complex.I) * z = 3 + Complex.I) : 
  z = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l525_52515


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l525_52599

/-- The perimeter of a hexagon with side length 5 inches is 30 inches. -/
theorem hexagon_perimeter (side_length : ℝ) (h : side_length = 5) : 
  6 * side_length = 30 := by sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l525_52599


namespace NUMINAMATH_CALUDE_inequality_solution_l525_52575

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, x^2 - (a + 3) * x + 2 * (a + 1) ≥ 0 ↔ 
    (a ≥ 1 ∧ (x ≥ a + 1 ∨ x ≤ 2)) ∨ 
    (a < 1 ∧ (x ≥ 2 ∨ x ≤ a + 1))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l525_52575


namespace NUMINAMATH_CALUDE_box_weight_sum_l525_52534

theorem box_weight_sum (a b c d : ℝ) 
  (h1 : a + b + c = 135)
  (h2 : a + b + d = 139)
  (h3 : a + c + d = 142)
  (h4 : b + c + d = 145)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a + b + c + d = 187 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_sum_l525_52534


namespace NUMINAMATH_CALUDE_rain_probability_tel_aviv_l525_52512

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv :
  let n : ℕ := 6
  let k : ℕ := 4
  let p : ℝ := 0.5
  binomial_probability n k p = 0.234375 := by
sorry

end NUMINAMATH_CALUDE_rain_probability_tel_aviv_l525_52512


namespace NUMINAMATH_CALUDE_number_of_children_l525_52501

theorem number_of_children (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 4) (h2 : total_pencils = 32) :
  total_pencils / pencils_per_child = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l525_52501


namespace NUMINAMATH_CALUDE_winnie_balloons_l525_52553

theorem winnie_balloons (red white green chartreuse : ℕ) 
  (h1 : red = 17) 
  (h2 : white = 33) 
  (h3 : green = 65) 
  (h4 : chartreuse = 83) 
  (friends : ℕ) 
  (h5 : friends = 8) : 
  (red + white + green + chartreuse) % friends = 6 := by
sorry

end NUMINAMATH_CALUDE_winnie_balloons_l525_52553


namespace NUMINAMATH_CALUDE_better_fit_example_l525_52552

/-- Represents a regression model with its RSS (Residual Sum of Squares) -/
structure RegressionModel where
  rss : ℝ

/-- Determines if one model has a better fit than another based on RSS -/
def better_fit (model1 model2 : RegressionModel) : Prop :=
  model1.rss < model2.rss

theorem better_fit_example :
  let model1 : RegressionModel := ⟨168⟩
  let model2 : RegressionModel := ⟨197⟩
  better_fit model1 model2 := by
  sorry

end NUMINAMATH_CALUDE_better_fit_example_l525_52552


namespace NUMINAMATH_CALUDE_solve_equation_l525_52537

theorem solve_equation :
  ∃! y : ℚ, 2 * y + 3 * y = 200 - (4 * y + 10 * y / 2) ∧ y = 100 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l525_52537
