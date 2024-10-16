import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_point_quadrant_l718_71865

/-- Given that point P(m,m-n) is symmetric to point Q(2,3) with respect to the origin,
    prove that point M(m,n) is in the second quadrant. -/
theorem symmetric_point_quadrant (m n : ℝ) : 
  (m = -2 ∧ m - n = -3) → m < 0 ∧ n > 0 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_quadrant_l718_71865


namespace NUMINAMATH_CALUDE_symmetry_condition_l718_71844

/-- A curve in the xy-plane represented by the equation x^2 + y^2 + Dx + Ey + F = 0 -/
structure Curve (D E F : ℝ) where
  condition : D^2 + E^2 - 4*F > 0

/-- Predicate for a curve being symmetric about the line y = x -/
def is_symmetric_about_y_eq_x (c : Curve D E F) : Prop :=
  D = E

/-- Theorem stating the condition for symmetry about y = x -/
theorem symmetry_condition (D E F : ℝ) (c : Curve D E F) :
  is_symmetric_about_y_eq_x c ↔ D = E :=
sorry

end NUMINAMATH_CALUDE_symmetry_condition_l718_71844


namespace NUMINAMATH_CALUDE_johns_run_l718_71822

/-- Theorem: John's total distance traveled is 5 miles -/
theorem johns_run (solo_speed : ℝ) (dog_speed : ℝ) (total_time : ℝ) (dog_time : ℝ) :
  solo_speed = 4 →
  dog_speed = 6 →
  total_time = 1 →
  dog_time = 0.5 →
  dog_speed * dog_time + solo_speed * (total_time - dog_time) = 5 := by
  sorry

#check johns_run

end NUMINAMATH_CALUDE_johns_run_l718_71822


namespace NUMINAMATH_CALUDE_radio_show_ad_break_duration_l718_71803

theorem radio_show_ad_break_duration 
  (total_show_time : ℕ) 
  (talking_segment_duration : ℕ) 
  (num_talking_segments : ℕ) 
  (num_ad_breaks : ℕ) 
  (song_duration : ℕ) 
  (h1 : total_show_time = 3 * 60) 
  (h2 : talking_segment_duration = 10)
  (h3 : num_talking_segments = 3)
  (h4 : num_ad_breaks = 5)
  (h5 : song_duration = 125) : 
  (total_show_time - (num_talking_segments * talking_segment_duration) - song_duration) / num_ad_breaks = 5 := by
sorry

end NUMINAMATH_CALUDE_radio_show_ad_break_duration_l718_71803


namespace NUMINAMATH_CALUDE_petrol_consumption_reduction_l718_71896

theorem petrol_consumption_reduction (P C : ℝ) (h : P > 0) (h' : C > 0) :
  let new_price := 1.25 * P
  let new_consumption := C * (P / new_price)
  (P * C = new_price * new_consumption) → (new_consumption / C = 0.8) :=
by sorry

end NUMINAMATH_CALUDE_petrol_consumption_reduction_l718_71896


namespace NUMINAMATH_CALUDE_increasing_function_property_l718_71874

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_property (f : ℝ → ℝ) (h : increasing_function f) (a b : ℝ) :
  (a + b ≥ 0) ↔ (f a + f b ≥ f (-a) + f (-b)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_property_l718_71874


namespace NUMINAMATH_CALUDE_rectangle_width_l718_71809

/-- Given a rectangle with specific properties, prove its width is 6 meters -/
theorem rectangle_width (area perimeter length width : ℝ) 
  (h_area : area = 50)
  (h_perimeter : perimeter = 30)
  (h_ratio : length = (3/2) * width)
  (h_area_def : area = length * width)
  (h_perimeter_def : perimeter = 2 * (length + width)) :
  width = 6 := by sorry

end NUMINAMATH_CALUDE_rectangle_width_l718_71809


namespace NUMINAMATH_CALUDE_range_of_m_for_cubic_equation_l718_71859

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem range_of_m_for_cubic_equation (m : ℝ) :
  (∃ x ∈ Set.Icc 0 2, f x + m = 0) → m ∈ Set.Icc (-2) 2 := by
  sorry


end NUMINAMATH_CALUDE_range_of_m_for_cubic_equation_l718_71859


namespace NUMINAMATH_CALUDE_football_inventory_solution_l718_71872

/-- Represents the football inventory problem -/
structure FootballInventory where
  total_footballs : ℕ
  total_cost : ℕ
  football_a_purchase : ℕ
  football_a_marked : ℕ
  football_b_purchase : ℕ
  football_b_marked : ℕ
  football_a_discount : ℚ
  football_b_discount : ℚ

/-- The specific football inventory problem instance -/
def problem : FootballInventory :=
  { total_footballs := 200
  , total_cost := 14400
  , football_a_purchase := 80
  , football_a_marked := 120
  , football_b_purchase := 60
  , football_b_marked := 90
  , football_a_discount := 1/5
  , football_b_discount := 1/10
  }

/-- Theorem stating the solution to the football inventory problem -/
theorem football_inventory_solution (p : FootballInventory) 
  (h1 : p = problem) : 
  ∃ (a b profit : ℕ), 
    a + b = p.total_footballs ∧ 
    a * p.football_a_purchase + b * p.football_b_purchase = p.total_cost ∧
    a = 120 ∧ 
    b = 80 ∧
    profit = a * (p.football_a_marked * (1 - p.football_a_discount) - p.football_a_purchase) + 
             b * (p.football_b_marked * (1 - p.football_b_discount) - p.football_b_purchase) ∧
    profit = 3600 :=
by
  sorry

end NUMINAMATH_CALUDE_football_inventory_solution_l718_71872


namespace NUMINAMATH_CALUDE_cube_sphere_volume_ratio_l718_71825

/-- The ratio of the volume of a cube with edge length 8 inches to the volume of a sphere with diameter 12 inches is 16 / (9π). -/
theorem cube_sphere_volume_ratio :
  let cube_edge : ℝ := 8
  let sphere_diameter : ℝ := 12
  let cube_volume := cube_edge ^ 3
  let sphere_radius := sphere_diameter / 2
  let sphere_volume := (4 / 3) * π * sphere_radius ^ 3
  cube_volume / sphere_volume = 16 / (9 * π) := by
  sorry

end NUMINAMATH_CALUDE_cube_sphere_volume_ratio_l718_71825


namespace NUMINAMATH_CALUDE_sum_of_divisors_91_l718_71843

/-- The sum of all positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all positive divisors of 91 is 112 -/
theorem sum_of_divisors_91 : sum_of_divisors 91 = 112 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_91_l718_71843


namespace NUMINAMATH_CALUDE_specific_ellipse_sum_l718_71863

/-- Represents an ellipse in a 2D Cartesian coordinate system -/
structure Ellipse where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  a : ℝ  -- length of the semi-major axis
  b : ℝ  -- length of the semi-minor axis

/-- The sum of center coordinates and axis lengths for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

/-- Theorem: For an ellipse with center (3, -5), horizontal semi-major axis 6, and vertical semi-minor axis 2, the sum h + k + a + b equals 6 -/
theorem specific_ellipse_sum :
  ∃ (e : Ellipse), e.h = 3 ∧ e.k = -5 ∧ e.a = 6 ∧ e.b = 2 ∧ ellipse_sum e = 6 := by
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_sum_l718_71863


namespace NUMINAMATH_CALUDE_min_unboxed_balls_tennis_balls_storage_l718_71898

theorem min_unboxed_balls (total_balls : ℕ) (big_box_size small_box_size : ℕ) : ℕ :=
  let min_unboxed := total_balls % big_box_size
  let remaining_after_big := total_balls % big_box_size
  let min_unboxed_small := remaining_after_big % small_box_size
  min min_unboxed min_unboxed_small

theorem tennis_balls_storage :
  min_unboxed_balls 104 25 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_unboxed_balls_tennis_balls_storage_l718_71898


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l718_71893

/-- Given a sequence {a_n} defined by a_n = 2n + 5, prove it is an arithmetic sequence with common difference 2 -/
theorem arithmetic_sequence_proof (n : ℕ) : ∃ (d : ℝ), d = 2 ∧ ∀ k, (2 * (k + 1) + 5) - (2 * k + 5) = d := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l718_71893


namespace NUMINAMATH_CALUDE_remaining_quantities_l718_71875

theorem remaining_quantities (total : ℕ) (avg_all : ℚ) (subset : ℕ) (avg_subset : ℚ) (avg_remaining : ℚ) :
  total = 5 →
  avg_all = 12 →
  subset = 3 →
  avg_subset = 4 →
  avg_remaining = 24 →
  total - subset = 2 :=
by sorry

end NUMINAMATH_CALUDE_remaining_quantities_l718_71875


namespace NUMINAMATH_CALUDE_james_payment_l718_71892

/-- The cost of cable program for James and his roommate -/
def cable_cost (first_100_cost : ℕ) (total_channels : ℕ) : ℕ :=
  if total_channels ≤ 100 then
    first_100_cost
  else
    first_100_cost + (first_100_cost / 2)

/-- James' share of the cable cost -/
def james_share (total_cost : ℕ) : ℕ := total_cost / 2

theorem james_payment (first_100_cost : ℕ) (total_channels : ℕ) :
  first_100_cost = 100 →
  total_channels = 200 →
  james_share (cable_cost first_100_cost total_channels) = 75 := by
  sorry

#eval james_share (cable_cost 100 200)

end NUMINAMATH_CALUDE_james_payment_l718_71892


namespace NUMINAMATH_CALUDE_symmetry_probability_l718_71810

/-- Represents a point on the grid -/
structure GridPoint where
  x : Fin 11
  y : Fin 11

/-- The center point of the grid -/
def centerPoint : GridPoint := ⟨5, 5⟩

/-- Checks if a point is on a line of symmetry -/
def isOnLineOfSymmetry (p : GridPoint) : Bool :=
  p.x = 5 ∨ p.y = 5 ∨ p.x = p.y ∨ p.x + p.y = 10

/-- The total number of points in the grid -/
def totalPoints : Nat := 121

/-- The number of points on lines of symmetry, excluding the center -/
def symmetryPoints : Nat := 40

/-- Theorem stating the probability of selecting a point on a line of symmetry -/
theorem symmetry_probability :
  (symmetryPoints : ℚ) / (totalPoints - 1 : ℚ) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_symmetry_probability_l718_71810


namespace NUMINAMATH_CALUDE_power_function_through_point_power_function_at_4_l718_71845

/-- A power function that passes through the point (3, √3) -/
def f (x : ℝ) : ℝ := x^(1/2)

theorem power_function_through_point : f 3 = Real.sqrt 3 := by sorry

theorem power_function_at_4 : f 4 = 2 := by sorry

end NUMINAMATH_CALUDE_power_function_through_point_power_function_at_4_l718_71845


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l718_71823

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set of f(x) > 0
def solution_set (a : ℝ) : Set ℝ := {x | f a x > 0}

-- Define the second quadratic function
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 5 * x + a^2 - 1

-- State the theorem
theorem quadratic_inequality_problem (a : ℝ) :
  solution_set a = {x | 1/2 < x ∧ x < 2} →
  (a = -2 ∧ {x | g a x > 0} = {x | -3 < x ∧ x < 1/2}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l718_71823


namespace NUMINAMATH_CALUDE_mower_next_tangent_east_l718_71819

/-- Represents the cardinal directions --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a circular garden with a mower --/
structure CircularGarden where
  garden_radius : ℝ
  mower_radius : ℝ
  initial_direction : Direction
  roll_direction : Bool  -- true for counterclockwise, false for clockwise

/-- 
  Determines the next tangent point where the mower's marker aims north again
  given a circular garden configuration
--/
def next_north_tangent (garden : CircularGarden) : Direction :=
  sorry

/-- The main theorem to be proved --/
theorem mower_next_tangent_east :
  let garden := CircularGarden.mk 15 5 Direction.North true
  next_north_tangent garden = Direction.East :=
sorry

end NUMINAMATH_CALUDE_mower_next_tangent_east_l718_71819


namespace NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l718_71852

theorem greatest_perimeter_of_special_triangle :
  ∀ (a b c : ℕ+),
  (a = 4 * b ∨ b = 4 * a ∨ a = 4 * c ∨ c = 4 * a ∨ b = 4 * c ∨ c = 4 * b) →
  (a = 18 ∨ b = 18 ∨ c = 18) →
  (a + b > c) →
  (a + c > b) →
  (b + c > a) →
  (a + b + c ≤ 43) :=
by sorry

end NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l718_71852


namespace NUMINAMATH_CALUDE_no_real_roots_implies_k_less_than_negative_one_l718_71800

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 - 2*x - k

-- Theorem statement
theorem no_real_roots_implies_k_less_than_negative_one (k : ℝ) :
  (∀ x : ℝ, quadratic x k ≠ 0) → k < -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_implies_k_less_than_negative_one_l718_71800


namespace NUMINAMATH_CALUDE_tangent_line_equality_l718_71828

noncomputable def f (x : ℝ) : ℝ := Real.log x

def g (a x : ℝ) : ℝ := a * x^2 - a

theorem tangent_line_equality (a : ℝ) : 
  (∀ x, deriv f x = deriv (g a) x) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equality_l718_71828


namespace NUMINAMATH_CALUDE_spring_work_compression_l718_71895

/-- Given a spring that is compressed 1 cm by a 10 N force, 
    the work done to compress it by 10 cm is 5 J. -/
theorem spring_work_compression (k : ℝ) : 
  (10 : ℝ) = k * 1 → (∫ x in (0 : ℝ)..(10 : ℝ), k * x) = 5 := by
  sorry

end NUMINAMATH_CALUDE_spring_work_compression_l718_71895


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l718_71878

-- Define the sets P and Q
def P : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

-- Define the result set
def result : Set ℝ := Set.Icc 0 2

-- State the theorem
theorem intersection_complement_equality : P ∩ (Set.univ \ Q) = result := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l718_71878


namespace NUMINAMATH_CALUDE_base_number_is_two_l718_71888

theorem base_number_is_two :
  ∀ x w : ℝ,
  w = 12 →
  x^(2*w) = 8^(w-4) →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_base_number_is_two_l718_71888


namespace NUMINAMATH_CALUDE_diagonal_angle_60_implies_equilateral_triangle_or_regular_hexagon_l718_71848

structure Polygon where
  sides : ℕ
  interior_angle : ℝ
  diagonal_angle : ℝ

def is_equilateral_triangle (p : Polygon) : Prop :=
  p.sides = 3 ∧ p.interior_angle = 60

def is_regular_hexagon (p : Polygon) : Prop :=
  p.sides = 6 ∧ p.interior_angle = 120

theorem diagonal_angle_60_implies_equilateral_triangle_or_regular_hexagon (p : Polygon) :
  p.diagonal_angle = 60 → is_equilateral_triangle p ∨ is_regular_hexagon p :=
by sorry

end NUMINAMATH_CALUDE_diagonal_angle_60_implies_equilateral_triangle_or_regular_hexagon_l718_71848


namespace NUMINAMATH_CALUDE_smaller_pack_size_l718_71882

/-- Represents the number of eggs in a package -/
structure EggPackage where
  size : ℕ

/-- Represents a purchase of eggs -/
structure EggPurchase where
  totalEggs : ℕ
  largePacks : ℕ
  smallPacks : ℕ
  largePackSize : ℕ
  smallPackSize : ℕ

/-- Defines a valid egg purchase -/
def isValidPurchase (p : EggPurchase) : Prop :=
  p.totalEggs = p.largePacks * p.largePackSize + p.smallPacks * p.smallPackSize

/-- Theorem: Given the conditions, the size of the smaller pack must be 24 eggs -/
theorem smaller_pack_size (p : EggPurchase) 
    (h1 : p.totalEggs = 79)
    (h2 : p.largePacks = 5)
    (h3 : p.largePackSize = 11)
    (h4 : isValidPurchase p) :
    p.smallPackSize = 24 := by
  sorry

#check smaller_pack_size

end NUMINAMATH_CALUDE_smaller_pack_size_l718_71882


namespace NUMINAMATH_CALUDE_cookies_per_bag_l718_71804

theorem cookies_per_bag (chocolate_chip : ℕ) (oatmeal : ℕ) (baggies : ℕ) 
  (h1 : chocolate_chip = 23)
  (h2 : oatmeal = 25)
  (h3 : baggies = 8) :
  (chocolate_chip + oatmeal) / baggies = 6 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l718_71804


namespace NUMINAMATH_CALUDE_parallelogram_slant_height_l718_71853

/-- Given a rectangle and a shape composed of an isosceles triangle and a parallelogram,
    prove that the slant height of the parallelogram is approximately 8.969 inches
    when the areas are equal. -/
theorem parallelogram_slant_height (rectangle_length rectangle_width triangle_base triangle_height parallelogram_base parallelogram_height : ℝ) 
  (h_rectangle_length : rectangle_length = 5)
  (h_rectangle_width : rectangle_width = 24)
  (h_triangle_base : triangle_base = 12)
  (h_parallelogram_base : parallelogram_base = 12)
  (h_equal_heights : triangle_height = parallelogram_height)
  (h_equal_areas : rectangle_length * rectangle_width = 
    (1/2 * triangle_base * triangle_height) + (parallelogram_base * parallelogram_height)) :
  ∃ (slant_height : ℝ), abs (slant_height - 8.969) < 0.001 ∧ 
    slant_height^2 = parallelogram_height^2 + (parallelogram_base/2)^2 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_slant_height_l718_71853


namespace NUMINAMATH_CALUDE_zoo_visitors_l718_71813

theorem zoo_visitors (total_people : ℕ) (adult_price kid_price total_sales : ℚ)
  (h1 : total_people = 254)
  (h2 : adult_price = 28)
  (h3 : kid_price = 12)
  (h4 : total_sales = 3864) :
  ∃ (adults : ℕ), adults = 51 ∧
    ∃ (kids : ℕ), adults + kids = total_people ∧
      adult_price * adults + kid_price * kids = total_sales :=
by sorry

end NUMINAMATH_CALUDE_zoo_visitors_l718_71813


namespace NUMINAMATH_CALUDE_g_composition_of_three_l718_71883

def g (n : ℤ) : ℤ :=
  if n < 5 then n^2 + 2*n - 1 else 2*n + 3

theorem g_composition_of_three : g (g (g 3)) = 65 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l718_71883


namespace NUMINAMATH_CALUDE_sqrt_625_equals_5_to_m_l718_71830

theorem sqrt_625_equals_5_to_m (m : ℝ) : (625 : ℝ)^(1/2) = 5^m → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_625_equals_5_to_m_l718_71830


namespace NUMINAMATH_CALUDE_train_speed_l718_71891

-- Define the length of the train in meters
def train_length : ℝ := 130

-- Define the time taken to cross the pole in seconds
def crossing_time : ℝ := 3.249740020798336

-- Define the conversion factor from m/s to km/hr
def ms_to_kmhr : ℝ := 3.6

-- Theorem to prove the train's speed
theorem train_speed : 
  (train_length / crossing_time) * ms_to_kmhr = 144 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l718_71891


namespace NUMINAMATH_CALUDE_smallest_add_subtract_for_perfect_square_l718_71871

theorem smallest_add_subtract_for_perfect_square (n m : ℕ) : 
  (∀ k : ℕ, k < 470 → ¬∃ i : ℕ, 92555 + k = i^2) ∧
  (∃ i : ℕ, 92555 + 470 = i^2) ∧
  (∀ j : ℕ, j < 139 → ¬∃ i : ℕ, 92555 - j = i^2) ∧
  (∃ i : ℕ, 92555 - 139 = i^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_add_subtract_for_perfect_square_l718_71871


namespace NUMINAMATH_CALUDE_function_equation_solution_l718_71861

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (x - y) * (f x - f y) = f (x - f y) * f (f x - y)) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) :=
sorry

end NUMINAMATH_CALUDE_function_equation_solution_l718_71861


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_AB_l718_71881

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 4 = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Assume the circles intersect at A and B
axiom intersect_at_A : circle1 A.1 A.2 ∧ circle2 A.1 A.2
axiom intersect_at_B : circle1 B.1 B.2 ∧ circle2 B.1 B.2

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_of_AB :
  ∀ x y : ℝ, perpendicular_bisector x y ↔ 
  (x - A.1) * (B.1 - A.1) + (y - A.2) * (B.2 - A.2) = 0 ∧
  (x - (A.1 + B.1) / 2)^2 + (y - (A.2 + B.2) / 2)^2 = 
  ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_AB_l718_71881


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l718_71833

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f : ℝ → ℝ := λ x => Real.sqrt x
  let h : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  ∃ x₀ y₀ : ℝ,
    x₀ ≥ 0 ∧
    y₀ = f x₀ ∧
    h x₀ y₀ ∧
    (λ x => (f x₀ - 0) / (x₀ - (-1)) * (x - x₀) + f x₀) (-1) = 0 →
    (Real.sqrt (a^2 + b^2)) / a = (Real.sqrt 5 + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l718_71833


namespace NUMINAMATH_CALUDE_least_number_divisible_l718_71812

theorem least_number_divisible (n : ℕ) : n = 857 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 24 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 32 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 36 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 7) = 54 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 7) = 24 * k₁ ∧ (n + 7) = 32 * k₂ ∧ (n + 7) = 36 * k₃ ∧ (n + 7) = 54 * k₄) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_l718_71812


namespace NUMINAMATH_CALUDE_dog_grouping_combinations_l718_71831

def total_dogs : ℕ := 15
def group_1_size : ℕ := 6
def group_2_size : ℕ := 5
def group_3_size : ℕ := 4

theorem dog_grouping_combinations :
  (total_dogs = group_1_size + group_2_size + group_3_size) →
  (Nat.choose (total_dogs - 2) (group_1_size - 1) * Nat.choose (total_dogs - group_1_size - 1) group_2_size = 72072) := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_combinations_l718_71831


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l718_71824

/-- The slope angle of the line x - √3y + 2 = 0 is 30° -/
theorem slope_angle_of_line (x y : ℝ) : 
  x - Real.sqrt 3 * y + 2 = 0 → 
  ∃ α : ℝ, α = 30 * (π / 180) ∧ 
    Real.tan α = Real.sqrt 3 / 3 ∧ 
    0 ≤ α ∧ α < π := by
  sorry


end NUMINAMATH_CALUDE_slope_angle_of_line_l718_71824


namespace NUMINAMATH_CALUDE_cube_root_26_approximation_l718_71869

theorem cube_root_26_approximation (ε : ℝ) (h : ε > 0) : 
  ∃ (x : ℝ), |x - (3 - 1/27)| < ε ∧ x^3 = 26 :=
sorry

end NUMINAMATH_CALUDE_cube_root_26_approximation_l718_71869


namespace NUMINAMATH_CALUDE_parabola_adjoint_tangent_locus_l718_71860

/-- Given a parabola y = 2px, prove that the locus of points (x, y) where the tangents 
    to the parabola are its own adjoint lines is described by the equation y² = -p/2 * x -/
theorem parabola_adjoint_tangent_locus (p : ℝ) (x y x₁ y₁ : ℝ) 
  (h1 : y₁ = 2 * p * x₁)  -- Original parabola equation
  (h2 : x = -x₁)          -- Relation between x and x₁
  (h3 : y = y₁ / 2)       -- Relation between y and y₁
  : y^2 = -p/2 * x := by sorry

end NUMINAMATH_CALUDE_parabola_adjoint_tangent_locus_l718_71860


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l718_71817

theorem book_arrangement_proof :
  let total_books : ℕ := 11
  let geometry_books : ℕ := 5
  let number_theory_books : ℕ := 6
  Nat.choose total_books geometry_books = 462 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l718_71817


namespace NUMINAMATH_CALUDE_log_inequality_solution_set_l718_71834

-- Define the logarithm function with base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the solution set
def solution_set : Set ℝ := { x | x > -1 ∧ x ≤ 0 }

-- Theorem statement
theorem log_inequality_solution_set :
  { x : ℝ | x > -1 ∧ log10 (x + 1) ≤ 0 } = solution_set :=
sorry

end NUMINAMATH_CALUDE_log_inequality_solution_set_l718_71834


namespace NUMINAMATH_CALUDE_no_solution_exists_l718_71862

theorem no_solution_exists : ∀ n : ℤ, n^2022 - 2*n^2021 + 3*n^2019 ≠ 2020 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l718_71862


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l718_71866

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 3*x ∧ 6*s^2 = 6*x) → x = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l718_71866


namespace NUMINAMATH_CALUDE_largest_factor_of_9975_l718_71894

theorem largest_factor_of_9975 : 
  ∀ n : ℕ, n ∣ 9975 ∧ n < 10000 → n ≤ 4975 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_factor_of_9975_l718_71894


namespace NUMINAMATH_CALUDE_cupboard_cost_price_l718_71873

theorem cupboard_cost_price (selling_price selling_price_increased : ℝ) 
  (h1 : selling_price = 0.84 * 5625)
  (h2 : selling_price_increased = 1.16 * 5625)
  (h3 : selling_price_increased - selling_price = 1800) : 
  5625 = 5625 := by sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_l718_71873


namespace NUMINAMATH_CALUDE_boat_weight_problem_l718_71820

theorem boat_weight_problem (initial_average : ℝ) (new_person_weight : ℝ) (new_average : ℝ) :
  initial_average = 60 →
  new_person_weight = 45 →
  new_average = 55 →
  ∃ n : ℕ, n * initial_average + new_person_weight = (n + 1) * new_average ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_boat_weight_problem_l718_71820


namespace NUMINAMATH_CALUDE_complex_sum_equality_l718_71835

/-- Given complex numbers B, Q, R, and T, prove that their sum is equal to 1 + 9i -/
theorem complex_sum_equality (B Q R T : ℂ) 
  (hB : B = 3 + 2*I)
  (hQ : Q = -5)
  (hR : R = 2*I)
  (hT : T = 3 + 5*I) :
  B - Q + R + T = 1 + 9*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l718_71835


namespace NUMINAMATH_CALUDE_carries_remaining_money_l718_71832

/-- The amount of money Carrie has left after shopping -/
def money_left (initial_amount sweater_price tshirt_price shoes_price jeans_original_price jeans_discount : ℚ) : ℚ :=
  initial_amount - (sweater_price + tshirt_price + shoes_price + (jeans_original_price * (1 - jeans_discount)))

/-- Proof that Carrie has $27.50 left after shopping -/
theorem carries_remaining_money :
  money_left 91 24 6 11 30 (25/100) = 27.5 := by
  sorry

end NUMINAMATH_CALUDE_carries_remaining_money_l718_71832


namespace NUMINAMATH_CALUDE_larger_number_proof_l718_71847

theorem larger_number_proof (x y : ℤ) : 
  x + y = 84 → y = x + 12 → y = 48 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l718_71847


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l718_71818

def M : Set Int := {-1, 0, 1}
def N : Set Int := {-2, -1, 0, 2}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l718_71818


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l718_71807

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l718_71807


namespace NUMINAMATH_CALUDE_proportion_solution_l718_71821

theorem proportion_solution (x : ℝ) : 
  (1.25 / x = 15 / 26.5) → x = 33.125 / 15 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l718_71821


namespace NUMINAMATH_CALUDE_decrease_by_one_point_five_l718_71806

/-- Represents a linear regression equation of the form y = a + bx -/
structure LinearRegression where
  a : ℝ
  b : ℝ

/-- The change in y when x increases by one unit in a linear regression -/
def change_in_y (lr : LinearRegression) : ℝ := -lr.b

/-- Theorem: In the given linear regression, when x increases by one unit, y decreases by 1.5 units -/
theorem decrease_by_one_point_five :
  let lr : LinearRegression := { a := 2, b := -1.5 }
  change_in_y lr = -1.5 := by sorry

end NUMINAMATH_CALUDE_decrease_by_one_point_five_l718_71806


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_18_l718_71839

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  0.5 * abs ((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) -
             (p1.y * p2.x + p2.y * p3.x + p3.y * p4.x + p4.y * p1.x))

/-- Theorem: The area of the quadrilateral with vertices at (0,0), (4,0), (6,3), and (4,6) is 18 -/
theorem quadrilateral_area_is_18 :
  let p1 : Point := ⟨0, 0⟩
  let p2 : Point := ⟨4, 0⟩
  let p3 : Point := ⟨6, 3⟩
  let p4 : Point := ⟨4, 6⟩
  quadrilateralArea p1 p2 p3 p4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_18_l718_71839


namespace NUMINAMATH_CALUDE_reinforcement_calculation_l718_71811

/-- Calculates the size of reinforcement given initial garrison size, provision days, and remaining days after reinforcement --/
def calculate_reinforcement (initial_garrison : ℕ) (initial_provision_days : ℕ) (days_before_reinforcement : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_provision_days
  let provisions_left := initial_garrison * (initial_provision_days - days_before_reinforcement)
  (provisions_left / remaining_days) - initial_garrison

theorem reinforcement_calculation (initial_garrison : ℕ) (initial_provision_days : ℕ) (days_before_reinforcement : ℕ) (remaining_days : ℕ) :
  initial_garrison = 1850 →
  initial_provision_days = 28 →
  days_before_reinforcement = 12 →
  remaining_days = 10 →
  calculate_reinforcement initial_garrison initial_provision_days days_before_reinforcement remaining_days = 1110 :=
by sorry

end NUMINAMATH_CALUDE_reinforcement_calculation_l718_71811


namespace NUMINAMATH_CALUDE_alice_paid_percentage_l718_71829

def suggested_retail_price : ℝ → ℝ := id

def marked_price (P : ℝ) : ℝ := 0.6 * P

def alice_paid (P : ℝ) : ℝ := 0.4 * marked_price P

theorem alice_paid_percentage (P : ℝ) (h : P > 0) :
  alice_paid P / suggested_retail_price P = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_alice_paid_percentage_l718_71829


namespace NUMINAMATH_CALUDE_original_speed_theorem_l718_71870

def distance : ℝ := 160
def speed_increase : ℝ := 0.25
def time_saved : ℝ := 0.4

theorem original_speed_theorem (original_speed : ℝ) 
  (h1 : original_speed > 0) 
  (h2 : distance / original_speed - distance / (original_speed * (1 + speed_increase)) = time_saved) : 
  original_speed = 80 := by
sorry

end NUMINAMATH_CALUDE_original_speed_theorem_l718_71870


namespace NUMINAMATH_CALUDE_root_relation_implies_k_value_l718_71876

theorem root_relation_implies_k_value :
  ∀ (k : ℝ) (a b : ℝ),
    (a^2 + k*a + 8 = 0) ∧
    (b^2 + k*b + 8 = 0) ∧
    ((a + 3)^2 - k*(a + 3) + 8 = 0) ∧
    ((b + 3)^2 - k*(b + 3) + 8 = 0) →
    k = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_root_relation_implies_k_value_l718_71876


namespace NUMINAMATH_CALUDE_smallest_number_proof_smallest_number_is_4725_l718_71802

theorem smallest_number_proof (x : ℕ) : 
  (x + 3 = 4728) ∧ 
  (∃ k₁ : ℕ, (x + 3) = 27 * k₁) ∧ 
  (∃ k₂ : ℕ, (x + 3) = 35 * k₂) ∧ 
  (∃ k₃ : ℕ, (x + 3) = 25 * k₃) →
  x ≥ 4725 :=
by sorry

theorem smallest_number_is_4725 : 
  (4725 + 3 = 4728) ∧ 
  (∃ k₁ : ℕ, (4725 + 3) = 27 * k₁) ∧ 
  (∃ k₂ : ℕ, (4725 + 3) = 35 * k₂) ∧ 
  (∃ k₃ : ℕ, (4725 + 3) = 25 * k₃) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_smallest_number_is_4725_l718_71802


namespace NUMINAMATH_CALUDE_total_precious_stones_l718_71838

/-- The number of precious stones in agate -/
def agate_stones : ℕ := 30

/-- The number of precious stones in olivine -/
def olivine_stones : ℕ := agate_stones + 5

/-- The number of precious stones in diamond -/
def diamond_stones : ℕ := olivine_stones + 11

/-- The total number of precious stones in agate, olivine, and diamond -/
def total_stones : ℕ := agate_stones + olivine_stones + diamond_stones

theorem total_precious_stones : total_stones = 111 := by
  sorry

end NUMINAMATH_CALUDE_total_precious_stones_l718_71838


namespace NUMINAMATH_CALUDE_cubic_inequality_l718_71836

theorem cubic_inequality (x y : ℝ) (h : x > y) : x^3 > y^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l718_71836


namespace NUMINAMATH_CALUDE_max_x_value_l718_71889

theorem max_x_value (x : ℝ) : 
  ((6*x - 15)/(4*x - 5))^2 - 3*((6*x - 15)/(4*x - 5)) - 10 = 0 → x ≤ 25/14 :=
by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l718_71889


namespace NUMINAMATH_CALUDE_kylie_daisies_left_l718_71886

def daisies_problem (initial_daisies : ℕ) (received_daisies : ℕ) : ℕ :=
  let total_daisies := initial_daisies + received_daisies
  let given_to_mother := total_daisies / 2
  total_daisies - given_to_mother

theorem kylie_daisies_left : daisies_problem 5 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_kylie_daisies_left_l718_71886


namespace NUMINAMATH_CALUDE_intersection_count_l718_71815

-- Define the equations
def eq1 (x y : ℝ) : Prop := (x + 2*y - 10) * (x - 4*y + 8) = 0
def eq2 (x y : ℝ) : Prop := (2*x - y - 1) * (5*x + 3*y - 15) = 0

-- Define a function to count distinct intersection points
noncomputable def count_intersections : ℕ :=
  -- Implementation details are omitted
  sorry

-- Theorem statement
theorem intersection_count : count_intersections = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l718_71815


namespace NUMINAMATH_CALUDE_david_found_amount_l718_71887

/-- The amount of money David found on the street -/
def david_found : ℕ := sorry

/-- The amount of money Evan initially had -/
def evan_initial : ℕ := 1

/-- The cost of the watch Evan wants to buy -/
def watch_cost : ℕ := 20

/-- The amount Evan still needs after receiving money from David -/
def evan_still_needs : ℕ := 7

theorem david_found_amount :
  david_found = watch_cost - evan_still_needs - evan_initial :=
sorry

end NUMINAMATH_CALUDE_david_found_amount_l718_71887


namespace NUMINAMATH_CALUDE_rectangular_playground_area_l718_71808

theorem rectangular_playground_area : 
  ∀ (length width : ℝ),
  length > 0 ∧ width > 0 →
  2 * (length + width) = 84 →
  length = 3 * width →
  length * width = 330.75 := by
sorry

end NUMINAMATH_CALUDE_rectangular_playground_area_l718_71808


namespace NUMINAMATH_CALUDE_compare_complex_fractions_l718_71868

theorem compare_complex_fractions : 
  1 / ((123^2 - 4) * 1375) > (7 / (5 * 9150625)) - (1 / (605 * 125^2)) := by
  sorry

end NUMINAMATH_CALUDE_compare_complex_fractions_l718_71868


namespace NUMINAMATH_CALUDE_binomial_sum_equals_120_l718_71877

theorem binomial_sum_equals_120 : 
  Nat.choose 8 2 + Nat.choose 8 3 + Nat.choose 9 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_equals_120_l718_71877


namespace NUMINAMATH_CALUDE_gerald_wood_pieces_l718_71880

/-- The number of pieces of wood needed to make a table -/
def wood_per_table : ℕ := 12

/-- The number of pieces of wood needed to make a chair -/
def wood_per_chair : ℕ := 8

/-- The number of chairs Gerald can make -/
def chairs : ℕ := 48

/-- The number of tables Gerald can make -/
def tables : ℕ := 24

/-- Theorem stating the total number of wood pieces Gerald has -/
theorem gerald_wood_pieces : 
  wood_per_table * tables + wood_per_chair * chairs = 672 := by
  sorry

end NUMINAMATH_CALUDE_gerald_wood_pieces_l718_71880


namespace NUMINAMATH_CALUDE_octal_2016_to_binary_l718_71842

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ := sorry

/-- Converts a decimal number to binary --/
def decimal_to_binary (decimal : ℕ) : List ℕ := sorry

/-- Converts a list of binary digits to a natural number --/
def binary_list_to_nat (binary : List ℕ) : ℕ := sorry

theorem octal_2016_to_binary :
  let octal := 2016
  let decimal := octal_to_decimal octal
  let binary := decimal_to_binary decimal
  binary_list_to_nat binary = binary_list_to_nat [1,0,0,0,0,0,0,1,1,1,0] := by sorry

end NUMINAMATH_CALUDE_octal_2016_to_binary_l718_71842


namespace NUMINAMATH_CALUDE_largest_x_value_l718_71801

theorem largest_x_value (x : ℝ) : 
  x / 7 + 3 / (7 * x) = 1 → x ≤ (7 + Real.sqrt 37) / 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_value_l718_71801


namespace NUMINAMATH_CALUDE_soda_cost_l718_71846

theorem soda_cost (burger_cost soda_cost : ℚ) : 
  (3 * burger_cost + 2 * soda_cost = 360) →
  (4 * burger_cost + 3 * soda_cost = 490) →
  soda_cost = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l718_71846


namespace NUMINAMATH_CALUDE_tangent_line_equation_l718_71884

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the point P
def point_P : ℝ × ℝ := (1, 2)

-- Define a function to check if a line is tangent to the circle C at a point
def is_tangent_to_C (a b c : ℝ) (x y : ℝ) : Prop :=
  circle_C x y ∧ a*x + b*y + c = 0 ∧
  ∀ x' y', circle_C x' y' → (a*x' + b*y' + c)^2 ≥ (a^2 + b^2) * (x'^2 + y'^2 - 2)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    is_tangent_to_C (point_P.1 - x₁) (point_P.2 - y₁) (x₁*(x₁ - point_P.1) + y₁*(y₁ - point_P.2)) x₁ y₁ ∧
    is_tangent_to_C (point_P.1 - x₂) (point_P.2 - y₂) (x₂*(x₂ - point_P.1) + y₂*(y₂ - point_P.2)) x₂ y₂ ∧
    x₁ + 2*y₁ - 2 = 0 ∧ x₂ + 2*y₂ - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l718_71884


namespace NUMINAMATH_CALUDE_joyce_gave_three_oranges_l718_71855

/-- The number of oranges Joyce gave to Clarence -/
def oranges_from_joyce (initial_oranges final_oranges : ℕ) : ℕ :=
  final_oranges - initial_oranges

theorem joyce_gave_three_oranges :
  oranges_from_joyce 5 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_joyce_gave_three_oranges_l718_71855


namespace NUMINAMATH_CALUDE_lcm_problem_l718_71864

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 36 m = 108) (h2 : Nat.lcm m 45 = 180) : m = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l718_71864


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l718_71841

-- Define the universal set U
def U : Set ℝ := {x | x^2 - (5/2)*x + 1 ≥ 0}

-- Define set A
def A : Set ℝ := {x | |x - 1| > 1}

-- Define set B
def B : Set ℝ := {x | (x + 1)/(x - 2) ≥ 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | x ≤ -1 ∨ x > 2} := by sorry

-- Theorem for A ∪ (CᵤB)
theorem union_A_complement_B : A ∪ (U \ B) = U := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l718_71841


namespace NUMINAMATH_CALUDE_line_perp_plane_transitive_l718_71890

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_transitive 
  (m n : Line) (α : Plane) :
  para m n → perp n α → perp m α :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_transitive_l718_71890


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l718_71816

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Checks if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.population_size

theorem systematic_sample_theorem (s : SystematicSample)
    (h_pop : s.population_size = 48)
    (h_sample : s.sample_size = 4)
    (h_interval : s.interval = s.population_size / s.sample_size)
    (h5 : s.contains 5)
    (h29 : s.contains 29)
    (h41 : s.contains 41) :
    s.contains 17 := by
  sorry

#check systematic_sample_theorem

end NUMINAMATH_CALUDE_systematic_sample_theorem_l718_71816


namespace NUMINAMATH_CALUDE_point_on_line_l718_71849

/-- Given a line passing through (0,10) and (-8,0), this theorem proves that 
    the x-coordinate of a point on this line with y-coordinate -6 is -64/5 -/
theorem point_on_line (x : ℚ) : 
  (∀ t : ℚ, t * (-8) = x ∧ t * (-10) + 10 = -6) → x = -64/5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l718_71849


namespace NUMINAMATH_CALUDE_job_completion_time_l718_71805

theorem job_completion_time (p_rate q_rate : ℚ) (t : ℚ) : 
  p_rate = 1/4 →
  q_rate = 1/15 →
  t * (p_rate + q_rate) + 1/5 * p_rate = 1 →
  t = 3 :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l718_71805


namespace NUMINAMATH_CALUDE_negative_discriminant_implies_no_real_roots_l718_71814

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- Calculates the discriminant of a quadratic equation -/
def discriminant {α : Type*} [Field α] (eq : QuadraticEquation α) : α :=
  eq.b ^ 2 - 4 * eq.a * eq.c

/-- Represents the property of having real roots -/
def has_real_roots {α : Type*} [Field α] (eq : QuadraticEquation α) : Prop :=
  ∃ x : α, eq.a * x ^ 2 + eq.b * x + eq.c = 0

theorem negative_discriminant_implies_no_real_roots 
  {k : ℝ} (eq : QuadraticEquation ℝ) 
  (h_eq : eq = { a := 3, b := -4 * Real.sqrt 3, c := k }) 
  (h_discr : discriminant eq < 0) : 
  ¬ has_real_roots eq :=
sorry

end NUMINAMATH_CALUDE_negative_discriminant_implies_no_real_roots_l718_71814


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l718_71899

theorem cos_pi_third_minus_alpha (α : Real) 
  (h : Real.sin (π / 6 + α) = 1 / 3) : 
  Real.cos (π / 3 - α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l718_71899


namespace NUMINAMATH_CALUDE_product_equation_solution_l718_71840

theorem product_equation_solution :
  ∀ B : ℕ,
  B < 10 →
  (10 * B + 4) * (10 * 8 + B) = 7008 →
  B = 7 := by
sorry

end NUMINAMATH_CALUDE_product_equation_solution_l718_71840


namespace NUMINAMATH_CALUDE_distance_AB_is_750_l718_71854

/-- The distance between two points A and B -/
def distance_AB : ℝ := 750

/-- The speed of person B in meters per minute -/
def speed_B : ℝ := 50

/-- The time it takes for A to catch up with B when moving in the same direction (in minutes) -/
def time_same_direction : ℝ := 30

/-- The time it takes for A and B to meet when moving towards each other (in minutes) -/
def time_towards_each_other : ℝ := 6

/-- The theorem stating that the distance between A and B is 750 meters -/
theorem distance_AB_is_750 : distance_AB = 750 :=
  sorry

end NUMINAMATH_CALUDE_distance_AB_is_750_l718_71854


namespace NUMINAMATH_CALUDE_total_flowers_l718_71879

theorem total_flowers (num_pots : ℕ) (flowers_per_pot : ℕ) (h1 : num_pots = 141) (h2 : flowers_per_pot = 71) : 
  num_pots * flowers_per_pot = 10011 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l718_71879


namespace NUMINAMATH_CALUDE_typing_speed_ratio_l718_71837

theorem typing_speed_ratio (T M : ℝ) (h1 : T > 0) (h2 : M > 0) 
  (h3 : T + M = 12) (h4 : T + 1.25 * M = 14) : M / T = 2 := by
  sorry

end NUMINAMATH_CALUDE_typing_speed_ratio_l718_71837


namespace NUMINAMATH_CALUDE_unique_5digit_number_l718_71858

/-- A function that generates all 3-digit numbers from a list of 5 digits -/
def generate_3digit_numbers (digits : List Nat) : List Nat :=
  sorry

/-- The sum of all 3-digit numbers generated from the digits of a 5-digit number -/
def sum_3digit_numbers (n : Nat) : Nat :=
  sorry

/-- Checks if a number has 5 different non-zero digits -/
def has_5_different_nonzero_digits (n : Nat) : Prop :=
  sorry

theorem unique_5digit_number : 
  ∃! n : Nat, 
    10000 ≤ n ∧ n < 100000 ∧
    has_5_different_nonzero_digits n ∧
    n = sum_3digit_numbers n ∧
    n = 35964 :=
  sorry

end NUMINAMATH_CALUDE_unique_5digit_number_l718_71858


namespace NUMINAMATH_CALUDE_monic_quartic_specific_values_l718_71856

-- Define a monic quartic polynomial
def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem monic_quartic_specific_values (f : ℝ → ℝ) 
  (h_monic : is_monic_quartic f)
  (h1 : f (-2) = -4)
  (h2 : f 1 = -1)
  (h3 : f (-4) = -16)
  (h4 : f 5 = -25) :
  f 0 = 40 := by sorry

end NUMINAMATH_CALUDE_monic_quartic_specific_values_l718_71856


namespace NUMINAMATH_CALUDE_max_value_of_f_l718_71851

-- Define the function
def f (x : ℝ) : ℝ := -3 * x^2 + 6

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l718_71851


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l718_71885

theorem mixed_number_calculation : 
  72 * ((2 + 3/4) - (3 + 1/2)) / ((3 + 1/3) + (1 + 1/4)) = -(13 + 1/11) := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l718_71885


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l718_71850

def f (x : ℝ) := x^3 + 3*x - 3

theorem root_exists_in_interval : ∃ x ∈ Set.Icc 0 1, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l718_71850


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l718_71867

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_equal_volume (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c = 10) :
  6 * ((a * b * c) ^ (1/3 : ℝ))^2 = 6 * (350 ^ (1/3 : ℝ))^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l718_71867


namespace NUMINAMATH_CALUDE_jar_water_problem_l718_71826

theorem jar_water_problem (capacity_x : ℝ) (capacity_y : ℝ) 
  (h1 : capacity_y = (1 / 2) * capacity_x) 
  (h2 : capacity_x > 0) :
  let initial_water_x := (1 / 2) * capacity_x
  let initial_water_y := (1 / 2) * capacity_y
  let final_water_x := initial_water_x + initial_water_y
  final_water_x = (3 / 4) * capacity_x := by
sorry

end NUMINAMATH_CALUDE_jar_water_problem_l718_71826


namespace NUMINAMATH_CALUDE_f_increasing_range_l718_71827

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * (x - 1)^2 + 1 else (a + 3) * x + 4 * a

theorem f_increasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔
  a ∈ Set.Icc (-2/5 : ℝ) 0 ∧ a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_range_l718_71827


namespace NUMINAMATH_CALUDE_box_weights_sum_l718_71897

/-- Given four boxes with weights a, b, c, and d, where:
    - The weight of box d is 60 pounds
    - The combined weight of (a,b) is 132 pounds
    - The combined weight of (a,c) is 136 pounds
    - The combined weight of (b,c) is 138 pounds
    Prove that the sum of weights a, b, and c is 203 pounds. -/
theorem box_weights_sum (a b c d : ℝ) 
    (hd : d = 60)
    (hab : a + b = 132)
    (hac : a + c = 136)
    (hbc : b + c = 138) :
    a + b + c = 203 := by
  sorry

end NUMINAMATH_CALUDE_box_weights_sum_l718_71897


namespace NUMINAMATH_CALUDE_b_21_mod_12_l718_71857

/-- Definition of b_n as the integer obtained by writing all integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- Theorem stating that b_21 mod 12 = 9 -/
theorem b_21_mod_12 : b 21 % 12 = 9 := by sorry

end NUMINAMATH_CALUDE_b_21_mod_12_l718_71857
