import Mathlib

namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l665_66597

theorem trigonometric_product_equals_one : 
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1/cos30) * (1 + 1/sin60) * (1 - 1/sin30) * (1 + 1/cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l665_66597


namespace NUMINAMATH_CALUDE_expand_expression_solve_inequality_system_l665_66524

-- Problem 1
theorem expand_expression (x : ℝ) : (2*x + 1)^2 + x*(x - 4) = 5*x^2 + 1 := by
  sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) :
  (3*x - 6 > 0 ∧ (5 - x)/2 < 1) ↔ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_solve_inequality_system_l665_66524


namespace NUMINAMATH_CALUDE_min_value_of_expression_l665_66531

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1/x + x/y ≥ 3 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1/x + x/y = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l665_66531


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l665_66509

/-- Given a cube with surface area 294 square centimeters, prove its volume is 343 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 294 → s^3 = 343 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l665_66509


namespace NUMINAMATH_CALUDE_inequality_chain_l665_66567

theorem inequality_chain (a : ℝ) (h : a - 1 > 0) : -a < -1 ∧ -1 < 1 ∧ 1 < a := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l665_66567


namespace NUMINAMATH_CALUDE_initial_marbles_relationship_l665_66511

/-- Represents the marble collection problem --/
structure MarbleCollection where
  initial : ℕ  -- Initial number of marbles
  lost : ℕ     -- Number of marbles lost
  found : ℕ    -- Number of marbles found
  current : ℕ  -- Current number of marbles after losses and finds

/-- The marble collection satisfies the problem conditions --/
def validCollection (m : MarbleCollection) : Prop :=
  m.lost = 16 ∧ m.found = 8 ∧ m.lost - m.found = 8 ∧ m.current = m.initial - m.lost + m.found

/-- Theorem stating the relationship between initial and current marbles --/
theorem initial_marbles_relationship (m : MarbleCollection) 
  (h : validCollection m) : m.initial = m.current + 8 := by
  sorry

#check initial_marbles_relationship

end NUMINAMATH_CALUDE_initial_marbles_relationship_l665_66511


namespace NUMINAMATH_CALUDE_ice_cream_bill_l665_66578

/-- Calculate the final bill for ice cream sundaes with tip -/
theorem ice_cream_bill (price1 price2 price3 price4 : ℝ) :
  let total_price := price1 + price2 + price3 + price4
  let tip_percentage := 0.20
  let tip := total_price * tip_percentage
  let final_bill := total_price + tip
  final_bill = total_price * (1 + tip_percentage) :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_bill_l665_66578


namespace NUMINAMATH_CALUDE_inequality_solution_l665_66591

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≥ 4 / 3) ↔ (-2 < x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l665_66591


namespace NUMINAMATH_CALUDE_carla_restock_theorem_l665_66506

/-- Represents the food bank inventory and distribution problem -/
structure FoodBank where
  initial_stock : ℕ
  day1_people : ℕ
  day1_cans_per_person : ℕ
  day2_people : ℕ
  day2_cans_per_person : ℕ
  final_restock : ℕ
  total_given_away : ℕ

/-- Calculates the number of cans restocked after the first day -/
def cans_restocked_after_day1 (fb : FoodBank) : ℕ :=
  fb.total_given_away - (fb.initial_stock - fb.day1_people * fb.day1_cans_per_person) +
  (fb.final_restock - fb.day2_people * fb.day2_cans_per_person)

/-- Theorem stating that Carla restocked 2000 cans after the first day -/
theorem carla_restock_theorem (fb : FoodBank)
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.day1_people = 500)
  (h3 : fb.day1_cans_per_person = 1)
  (h4 : fb.day2_people = 1000)
  (h5 : fb.day2_cans_per_person = 2)
  (h6 : fb.final_restock = 3000)
  (h7 : fb.total_given_away = 2500) :
  cans_restocked_after_day1 fb = 2000 := by
  sorry

end NUMINAMATH_CALUDE_carla_restock_theorem_l665_66506


namespace NUMINAMATH_CALUDE_expression_simplification_l665_66564

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 3) / x / ((x^2 - 6*x + 9) / (x^2 - 9)) - (x + 1) / x = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l665_66564


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l665_66565

open Real

theorem triangle_abc_properties (a b c A B C : ℝ) (k : ℤ) :
  -- Conditions
  (2 * Real.sqrt 3 * a * Real.sin C * Real.sin B = a * Real.sin A + b * Real.sin B - c * Real.sin C) →
  (a * Real.cos (π / 2 - B) = b * Real.cos (2 * ↑k * π + A)) →
  (a = 2) →
  -- Conclusions
  (C = π / 6) ∧
  (1 / 2 * a * c * Real.sin B = (1 + Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l665_66565


namespace NUMINAMATH_CALUDE_f_minus_three_equals_six_l665_66587

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_minus_three_equals_six 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 4) 
  (h_sum : f 1 + f 2 + f 3 + f 4 + f 5 = 6) : 
  f (-3) = 6 := by
sorry

end NUMINAMATH_CALUDE_f_minus_three_equals_six_l665_66587


namespace NUMINAMATH_CALUDE_trapezoid_triangle_area_l665_66596

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Checks if two line segments are perpendicular -/
def perpendicular (P Q R S : Point) : Prop := sorry

/-- Checks if two line segments are parallel -/
def parallel (P Q R S : Point) : Prop := sorry

/-- Calculates the length of a line segment -/
def length (P Q : Point) : ℝ := sorry

/-- Checks if a point is on a line segment -/
def on_segment (P Q R : Point) : Prop := sorry

/-- Calculates the area of a triangle -/
def triangle_area (P Q R : Point) : ℝ := sorry

theorem trapezoid_triangle_area 
  (ABCD : Trapezoid) 
  (E : Point) 
  (h1 : perpendicular ABCD.A ABCD.D ABCD.D ABCD.C)
  (h2 : length ABCD.A ABCD.D = 4)
  (h3 : length ABCD.A ABCD.B = 4)
  (h4 : length ABCD.D ABCD.C = 10)
  (h5 : on_segment E ABCD.D ABCD.C)
  (h6 : length ABCD.D E = 7)
  (h7 : parallel ABCD.B E ABCD.A ABCD.D) :
  triangle_area ABCD.B E ABCD.C = 6 := by sorry

end NUMINAMATH_CALUDE_trapezoid_triangle_area_l665_66596


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_external_tangents_l665_66507

/-- Given two externally tangent circles, this theorem proves the radius of the circle
    tangent to their common external tangents and the line segment connecting the
    external points of tangency on the larger circle. -/
theorem inscribed_circle_radius_external_tangents
  (R : ℝ) (r : ℝ) (h_R : R = 4) (h_r : r = 3) (h_touch : R > r) :
  let d := R + r  -- Distance between circle centers
  let inscribed_radius := (R * r) / d
  inscribed_radius = 12 / 7 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_external_tangents_l665_66507


namespace NUMINAMATH_CALUDE_cube_of_negative_l665_66560

theorem cube_of_negative (x : ℝ) (h : x^3 = 32.768) : (-x)^3 = -32.768 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_l665_66560


namespace NUMINAMATH_CALUDE_opposite_of_two_l665_66533

theorem opposite_of_two : (- 2 : ℤ) = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_two_l665_66533


namespace NUMINAMATH_CALUDE_max_product_decomposition_l665_66553

theorem max_product_decomposition :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x + y = 100 → x * y ≤ 50 * 50 :=
by sorry

end NUMINAMATH_CALUDE_max_product_decomposition_l665_66553


namespace NUMINAMATH_CALUDE_cube_root_of_hundred_l665_66593

theorem cube_root_of_hundred (x : ℝ) : (Real.sqrt x)^3 = 100 → x = 10^(4/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_hundred_l665_66593


namespace NUMINAMATH_CALUDE_sum_of_cubes_squares_and_product_l665_66505

theorem sum_of_cubes_squares_and_product : (3 + 7)^3 + (3^2 + 7^2) + 3 * 7 = 1079 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_squares_and_product_l665_66505


namespace NUMINAMATH_CALUDE_divisibility_by_three_l665_66582

/-- A sequence of integers satisfying the recurrence relation -/
def SatisfiesRecurrence (a : ℕ → ℤ) (k : ℕ+) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n * n = a (n - 1) + n^(k : ℕ)

/-- The main theorem -/
theorem divisibility_by_three (k : ℕ+) (a : ℕ → ℤ) 
  (h : SatisfiesRecurrence a k) : 
  3 ∣ (k : ℤ) - 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l665_66582


namespace NUMINAMATH_CALUDE_triangle_side_length_l665_66556

/-- Given a triangle ABC with side lengths a, b, c, prove that if a = 2, b + c = 7, and cos B = -1/4, then b = 4 -/
theorem triangle_side_length (a b c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : b + c = 7) (h3 : Real.cos B = -1/4) : b = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l665_66556


namespace NUMINAMATH_CALUDE_pelicans_remaining_l665_66513

/-- Represents the number of pelicans in Shark Bite Cove -/
def original_pelicans : ℕ := 30

/-- Represents the number of sharks in Pelican Bay -/
def sharks : ℕ := 60

/-- Represents the fraction of pelicans that moved from Shark Bite Cove to Pelican Bay -/
def moved_fraction : ℚ := 1/3

/-- The theorem stating the number of pelicans remaining in Shark Bite Cove -/
theorem pelicans_remaining : 
  sharks = 2 * original_pelicans ∧ 
  (original_pelicans : ℚ) * (1 - moved_fraction) = 20 :=
sorry

end NUMINAMATH_CALUDE_pelicans_remaining_l665_66513


namespace NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l665_66522

theorem square_area_to_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ ^ 2 / s₂ ^ 2 = 25 / 36) :
  (4 * s₁) / (4 * s₂) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l665_66522


namespace NUMINAMATH_CALUDE_xyz_sum_l665_66529

theorem xyz_sum (x y z : ℕ+) 
  (h : (x + y + z : ℕ+)^3 - x^3 - y^3 - z^3 = 300) : 
  (x : ℕ) + y + z = 7 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l665_66529


namespace NUMINAMATH_CALUDE_library_books_theorem_l665_66577

variable (Library : Type)
variable (is_new_edition : Library → Prop)

theorem library_books_theorem :
  (¬ ∀ (book : Library), is_new_edition book) →
  (∃ (book : Library), ¬ is_new_edition book) ∧
  (¬ ∀ (book : Library), is_new_edition book) :=
by sorry

end NUMINAMATH_CALUDE_library_books_theorem_l665_66577


namespace NUMINAMATH_CALUDE_parallelogram_count_in_triangle_grid_l665_66536

/-- Given an equilateral triangle with sides divided into n parts, 
    calculates the number of parallelograms formed by parallel lines --/
def parallelogramCount (n : ℕ) : ℕ :=
  3 * Nat.choose (n + 2) 4

/-- Theorem stating the number of parallelograms in the grid --/
theorem parallelogram_count_in_triangle_grid (n : ℕ) :
  parallelogramCount n = 3 * Nat.choose (n + 2) 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_count_in_triangle_grid_l665_66536


namespace NUMINAMATH_CALUDE_train_crossing_time_l665_66512

/-- Time for a train to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 350 → train_speed_kmh = 144 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l665_66512


namespace NUMINAMATH_CALUDE_units_digit_of_m_cubed_plus_three_to_m_l665_66520

def m : ℕ := 2011^2 + 2^2011

theorem units_digit_of_m_cubed_plus_three_to_m (m : ℕ := 2011^2 + 2^2011) : 
  (m^3 + 3^m) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_m_cubed_plus_three_to_m_l665_66520


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l665_66527

/-- Given a very large box containing small boxes of chocolate bars, 
    calculate the total number of chocolate bars. -/
theorem chocolate_bar_count 
  (num_small_boxes : ℕ) 
  (bars_per_small_box : ℕ) 
  (h1 : num_small_boxes = 150) 
  (h2 : bars_per_small_box = 37) : 
  num_small_boxes * bars_per_small_box = 5550 := by
  sorry

#check chocolate_bar_count

end NUMINAMATH_CALUDE_chocolate_bar_count_l665_66527


namespace NUMINAMATH_CALUDE_square_perimeter_l665_66530

theorem square_perimeter (t : ℝ) (h1 : t > 0) : 
  (5 / 2 * t = 40) → (4 * t = 64) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l665_66530


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l665_66568

theorem partial_fraction_decomposition_product (N₁ N₂ : ℝ) : 
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ 2 → (42 * x - 36) / (x^2 - 3*x + 2) = N₁ / (x - 1) + N₂ / (x - 2)) →
  N₁ * N₂ = -288 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l665_66568


namespace NUMINAMATH_CALUDE_incorrect_operation_l665_66589

theorem incorrect_operation (x y : ℝ) : -2*x*(x - y) ≠ -2*x^2 - 2*x*y := by
  sorry

end NUMINAMATH_CALUDE_incorrect_operation_l665_66589


namespace NUMINAMATH_CALUDE_surface_area_of_problem_solid_l665_66517

/-- Represents an L-shaped solid formed by unit cubes -/
structure LShapedSolid where
  base_layer : ℕ
  top_layer : ℕ
  top_layer_start : ℕ

/-- Calculates the surface area of an L-shaped solid -/
def surface_area (solid : LShapedSolid) : ℕ :=
  let base_exposed := solid.base_layer - (solid.top_layer - (solid.top_layer_start - 1))
  let top_exposed := solid.top_layer
  let front_back := 2 * (solid.base_layer + solid.top_layer)
  let sides := 2 * 2
  let top_bottom := base_exposed + top_exposed + (solid.top_layer_start - 1)
  front_back + sides + top_bottom

/-- The specific L-shaped solid described in the problem -/
def problem_solid : LShapedSolid :=
  { base_layer := 8
  , top_layer := 6
  , top_layer_start := 5 }

theorem surface_area_of_problem_solid :
  surface_area problem_solid = 44 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_problem_solid_l665_66517


namespace NUMINAMATH_CALUDE_shortest_path_between_circles_l665_66548

theorem shortest_path_between_circles (center_distance : Real) 
  (radius_large : Real) (radius_small : Real) : Real :=
by
  -- Define the conditions
  have h1 : center_distance = 51 := by sorry
  have h2 : radius_large = 12 := by sorry
  have h3 : radius_small = 7 := by sorry

  -- Calculate the length of the external tangent
  let total_distance := center_distance + radius_large + radius_small
  let tangent_length := Real.sqrt (total_distance^2 - radius_large^2)

  -- Prove that the tangent length is 69 feet
  have h4 : tangent_length = 69 := by sorry

  -- Return the result
  exact tangent_length

end NUMINAMATH_CALUDE_shortest_path_between_circles_l665_66548


namespace NUMINAMATH_CALUDE_consecutive_integers_product_210_l665_66510

theorem consecutive_integers_product_210 (n : ℤ) :
  n * (n + 1) * (n + 2) = 210 → n + (n + 1) = 11 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_210_l665_66510


namespace NUMINAMATH_CALUDE_andy_rahim_age_difference_l665_66566

/-- The age difference between Andy and Rahim -/
def ageDifference (rahimAge : ℕ) (andyFutureAge : ℕ) : ℕ :=
  andyFutureAge - 5 - rahimAge

theorem andy_rahim_age_difference :
  ∀ (rahimAge : ℕ) (andyFutureAge : ℕ),
    rahimAge = 6 →
    andyFutureAge = 2 * rahimAge →
    ageDifference rahimAge andyFutureAge = 1 := by
  sorry

end NUMINAMATH_CALUDE_andy_rahim_age_difference_l665_66566


namespace NUMINAMATH_CALUDE_integer_fraction_characterization_l665_66563

def is_integer_fraction (m n : ℕ+) : Prop :=
  ∃ k : ℤ, (n.val ^ 3 + 1 : ℤ) = k * (m.val * n.val - 1)

def solution_set : Set (ℕ+ × ℕ+) :=
  {(2, 1), (3, 1), (1, 2), (2, 2), (5, 2), (1, 3), (5, 3), (3, 5)}

theorem integer_fraction_characterization :
  ∀ m n : ℕ+, is_integer_fraction m n ↔ (m, n) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_characterization_l665_66563


namespace NUMINAMATH_CALUDE_slope_product_for_30_degree_angle_l665_66579

theorem slope_product_for_30_degree_angle (m₁ m₂ : ℝ) :
  m₁ ≠ 0 →
  m₂ = 4 * m₁ →
  |((m₂ - m₁) / (1 + m₁ * m₂))| = 1 / Real.sqrt 3 →
  m₁ * m₂ = (38 - 6 * Real.sqrt 33) / 16 :=
by sorry

end NUMINAMATH_CALUDE_slope_product_for_30_degree_angle_l665_66579


namespace NUMINAMATH_CALUDE_cone_base_circumference_l665_66594

theorem cone_base_circumference (r : ℝ) (angle : ℝ) (h1 : r = 6) (h2 : angle = 120) :
  let original_circumference := 2 * π * r
  let sector_fraction := angle / 360
  let base_circumference := (1 - sector_fraction) * original_circumference
  base_circumference = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l665_66594


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l665_66502

theorem sqrt_product_simplification : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l665_66502


namespace NUMINAMATH_CALUDE_solution_range_l665_66500

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1/4)^x + (1/2)^(x-1) + a = 0) → 
  -3 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l665_66500


namespace NUMINAMATH_CALUDE_reciprocal_of_2016_l665_66569

theorem reciprocal_of_2016 : (2016⁻¹ : ℚ) = 1 / 2016 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_2016_l665_66569


namespace NUMINAMATH_CALUDE_grid_difference_theorem_l665_66549

def Grid := Fin 8 → Fin 8 → Fin 64

def adjacent (x₁ y₁ x₂ y₂ : Fin 8) : Prop :=
  (x₁ = x₂ ∧ (y₁.val + 1 = y₂.val ∨ y₂.val + 1 = y₁.val)) ∨
  (y₁ = y₂ ∧ (x₁.val + 1 = x₂.val ∨ x₂.val + 1 = x₁.val))

theorem grid_difference_theorem (g : Grid) (h : Function.Injective g) :
    ∃ (x₁ y₁ x₂ y₂ : Fin 8), adjacent x₁ y₁ x₂ y₂ ∧ 
    (g x₁ y₁).val.succ.succ.succ.succ ≤ (g x₂ y₂).val ∨ 
    (g x₂ y₂).val.succ.succ.succ.succ ≤ (g x₁ y₁).val := by
  sorry

end NUMINAMATH_CALUDE_grid_difference_theorem_l665_66549


namespace NUMINAMATH_CALUDE_jonah_profit_l665_66581

/-- Calculates the profit from selling pineapple rings given the following conditions:
  * Number of pineapples bought
  * Cost per pineapple
  * Number of rings per pineapple
  * Number of rings sold as a set
  * Price per set of rings
-/
def calculate_profit (num_pineapples : ℕ) (cost_per_pineapple : ℕ) 
                     (rings_per_pineapple : ℕ) (rings_per_set : ℕ) 
                     (price_per_set : ℕ) : ℕ :=
  let total_cost := num_pineapples * cost_per_pineapple
  let total_rings := num_pineapples * rings_per_pineapple
  let num_sets := total_rings / rings_per_set
  let total_revenue := num_sets * price_per_set
  total_revenue - total_cost

/-- Proves that Jonah's profit is $342 given the specified conditions -/
theorem jonah_profit : 
  calculate_profit 6 3 12 4 5 = 342 := by
  sorry

end NUMINAMATH_CALUDE_jonah_profit_l665_66581


namespace NUMINAMATH_CALUDE_seven_telephones_wires_l665_66572

/-- The number of wires needed to connect n telephone sets, where each pair is connected. -/
def wiresNeeded (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 7 telephone sets, the number of wires needed is 21. -/
theorem seven_telephones_wires : wiresNeeded 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_seven_telephones_wires_l665_66572


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_equals_five_l665_66537

theorem sum_of_reciprocals_equals_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 5 * x * y) : 1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_equals_five_l665_66537


namespace NUMINAMATH_CALUDE_no_five_naturals_product_equals_sum_l665_66558

theorem no_five_naturals_product_equals_sum :
  ¬ ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ d * e = a + b + c + d + e := by
sorry

end NUMINAMATH_CALUDE_no_five_naturals_product_equals_sum_l665_66558


namespace NUMINAMATH_CALUDE_runner_area_theorem_l665_66585

/-- Given a table and three runners, calculates the total area of the runners -/
def total_runner_area (table_area : ℝ) (double_layer_area : ℝ) (triple_layer_area : ℝ) : ℝ :=
  let covered_area := 0.8 * table_area
  let single_layer_area := covered_area - double_layer_area - triple_layer_area
  single_layer_area + 2 * double_layer_area + 3 * triple_layer_area

/-- Theorem stating that under the given conditions, the total area of the runners is 168 square inches -/
theorem runner_area_theorem (table_area : ℝ) (double_layer_area : ℝ) (triple_layer_area : ℝ) 
  (h1 : table_area = 175)
  (h2 : double_layer_area = 24)
  (h3 : triple_layer_area = 28) :
  total_runner_area table_area double_layer_area triple_layer_area = 168 := by
  sorry

#eval total_runner_area 175 24 28

end NUMINAMATH_CALUDE_runner_area_theorem_l665_66585


namespace NUMINAMATH_CALUDE_min_turns_10x10_grid_l665_66541

/-- Represents a city grid -/
structure CityGrid where
  parallel_streets : ℕ
  intersecting_streets : ℕ

/-- Represents a bus route in the city -/
structure BusRoute where
  turns : ℕ
  closed : Bool
  covers_all_intersections : Bool

/-- The minimum number of turns for a valid bus route -/
def min_turns (city : CityGrid) : ℕ := 2 * (city.parallel_streets + city.intersecting_streets)

/-- Theorem stating the minimum number of turns for a 10x10 grid city -/
theorem min_turns_10x10_grid :
  let city : CityGrid := ⟨10, 10⟩
  let route : BusRoute := ⟨min_turns city, true, true⟩
  route.turns = 20 ∧
  ∀ (other_route : BusRoute),
    (other_route.closed ∧ other_route.covers_all_intersections) →
    other_route.turns ≥ route.turns :=
by sorry

end NUMINAMATH_CALUDE_min_turns_10x10_grid_l665_66541


namespace NUMINAMATH_CALUDE_extra_calories_burned_l665_66503

def calories_per_hour : ℕ := 30

def calories_burned (hours : ℕ) : ℕ := hours * calories_per_hour

theorem extra_calories_burned : calories_burned 5 - calories_burned 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_extra_calories_burned_l665_66503


namespace NUMINAMATH_CALUDE_point_on_linear_graph_l665_66521

/-- For any point (a, b) on the graph of y = 2x - 1, 2a - b + 1 = 2 -/
theorem point_on_linear_graph (a b : ℝ) (h : b = 2 * a - 1) : 2 * a - b + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_linear_graph_l665_66521


namespace NUMINAMATH_CALUDE_ellen_hits_nine_l665_66588

-- Define the set of possible scores
def ScoreSet : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

-- Define a type for the players
inductive Player : Type
| Alice | Ben | Cindy | Dave | Ellen | Frank

-- Define a function that returns the total score for each player
def playerScore (p : Player) : ℕ :=
  match p with
  | Player.Alice => 27
  | Player.Ben => 14
  | Player.Cindy => 20
  | Player.Dave => 22
  | Player.Ellen => 24
  | Player.Frank => 30

-- Define a predicate that checks if a list of scores is valid for a player
def validScores (scores : List ℕ) (p : Player) : Prop :=
  scores.length = 3 ∧
  scores.toFinset.card = 3 ∧
  (∀ s ∈ scores, s ∈ ScoreSet) ∧
  scores.sum = playerScore p

theorem ellen_hits_nine :
  ∃ (scores : List ℕ), validScores scores Player.Ellen ∧ 9 ∈ scores ∧
  (∀ (p : Player), p ≠ Player.Ellen → ∀ (s : List ℕ), validScores s p → 9 ∉ s) :=
sorry

end NUMINAMATH_CALUDE_ellen_hits_nine_l665_66588


namespace NUMINAMATH_CALUDE_zeros_imply_b_and_c_b_in_interval_l665_66539

-- Define the quadratic function f(x)
def f (b c x : ℝ) : ℝ := x^2 + 2*b*x + c

-- Part 1: Prove that if -1 and 1 are zeros of f(x), then b = 0 and c = -1
theorem zeros_imply_b_and_c (b c : ℝ) :
  f b c (-1) = 0 ∧ f b c 1 = 0 → b = 0 ∧ c = -1 := by sorry

-- Part 2: Prove that given the conditions, b is in the interval (1/5, 5/7)
theorem b_in_interval (b c : ℝ) :
  f b c 1 = 0 ∧ 
  (∃ x₁ x₂, -3 < x₁ ∧ x₁ < -2 ∧ 0 < x₂ ∧ x₂ < 1 ∧ 
    f b c x₁ + x₁ + b = 0 ∧ f b c x₂ + x₂ + b = 0) →
  1/5 < b ∧ b < 5/7 := by sorry

end NUMINAMATH_CALUDE_zeros_imply_b_and_c_b_in_interval_l665_66539


namespace NUMINAMATH_CALUDE_max_d_value_l665_66519

def a (n : ℕ+) : ℕ := 101 + n.val ^ 2 + 3 * n.val

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∀ n : ℕ+, d n ≤ 4 ∧ ∃ m : ℕ+, d m = 4 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l665_66519


namespace NUMINAMATH_CALUDE_inequality_of_positive_reals_l665_66547

theorem inequality_of_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_reals_l665_66547


namespace NUMINAMATH_CALUDE_krishans_money_l665_66580

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Ram's amount,
    prove that Krishan has Rs. 3468. -/
theorem krishans_money (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 588 →
  krishan = 3468 := by
sorry

end NUMINAMATH_CALUDE_krishans_money_l665_66580


namespace NUMINAMATH_CALUDE_root_sum_theorem_l665_66583

theorem root_sum_theorem (x : ℝ) : 
  (1/x + 1/(x + 4) - 1/(x + 8) - 1/(x + 12) - 1/(x + 16) - 1/(x + 20) + 1/(x + 24) + 1/(x + 28) = 0) →
  (∃ (a b c d : ℕ), 
    (x = -a + Real.sqrt (b + c * Real.sqrt d) ∨ x = -a - Real.sqrt (b + c * Real.sqrt d) ∨
     x = -a + Real.sqrt (b - c * Real.sqrt d) ∨ x = -a - Real.sqrt (b - c * Real.sqrt d)) ∧
    a + b + c + d = 123) := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l665_66583


namespace NUMINAMATH_CALUDE_mary_total_spending_l665_66508

/-- The total amount Mary spent on clothing, given the costs of a shirt and a jacket. -/
def total_spent (shirt_cost jacket_cost : ℚ) : ℚ :=
  shirt_cost + jacket_cost

/-- Theorem stating that Mary's total spending is $25.31 -/
theorem mary_total_spending :
  total_spent 13.04 12.27 = 25.31 := by
  sorry

end NUMINAMATH_CALUDE_mary_total_spending_l665_66508


namespace NUMINAMATH_CALUDE_heather_blocks_shared_l665_66534

/-- The number of blocks Heather shared with Jose -/
def blocks_shared (initial final : ℕ) : ℕ := initial - final

/-- Theorem stating that the number of blocks shared is the difference between initial and final counts -/
theorem heather_blocks_shared : 
  blocks_shared 86 45 = 41 := by
  sorry

end NUMINAMATH_CALUDE_heather_blocks_shared_l665_66534


namespace NUMINAMATH_CALUDE_translated_sine_function_l665_66592

/-- Given a function f and its right-translated version g, prove that g has the expected form. -/
theorem translated_sine_function (f g : ℝ → ℝ) (h : ℝ → ℝ → Prop) : 
  (∀ x, f x = 2 * Real.sin (2 * x + 2 * Real.pi / 3)) →
  (∀ x, h x (g x) ↔ h (x - Real.pi / 6) (f x)) →
  (∀ x, g x = 2 * Real.sin (2 * x + Real.pi / 3)) := by
  sorry


end NUMINAMATH_CALUDE_translated_sine_function_l665_66592


namespace NUMINAMATH_CALUDE_complement_union_A_B_l665_66515

def A : Set ℝ := {x : ℝ | x ≤ 0}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

theorem complement_union_A_B : 
  (A ∪ B)ᶜ = {x : ℝ | x > 1} :=
sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l665_66515


namespace NUMINAMATH_CALUDE_outfit_count_l665_66516

/-- Represents the number of shirts available -/
def num_shirts : ℕ := 7

/-- Represents the number of pants available -/
def num_pants : ℕ := 5

/-- Represents the number of ties available -/
def num_ties : ℕ := 4

/-- Represents the total number of tie options (including the option of not wearing a tie) -/
def tie_options : ℕ := num_ties + 1

/-- Calculates the total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_pants * tie_options

/-- Theorem stating that the total number of possible outfits is 175 -/
theorem outfit_count : total_outfits = 175 := by sorry

end NUMINAMATH_CALUDE_outfit_count_l665_66516


namespace NUMINAMATH_CALUDE_utility_bill_total_l665_66518

def fifty_bill_value : ℕ := 50
def ten_bill_value : ℕ := 10
def fifty_bill_count : ℕ := 3
def ten_bill_count : ℕ := 2

theorem utility_bill_total : 
  fifty_bill_value * fifty_bill_count + ten_bill_value * ten_bill_count = 170 := by
  sorry

end NUMINAMATH_CALUDE_utility_bill_total_l665_66518


namespace NUMINAMATH_CALUDE_box_height_l665_66561

/-- Given a rectangular box with width 10 inches, length 20 inches, and height h inches,
    if the area of the triangle formed by the center points of three faces meeting at a corner
    is 40 square inches, then h = (24 * sqrt(21)) / 5 inches. -/
theorem box_height (h : ℝ) : 
  let width : ℝ := 10
  let length : ℝ := 20
  let triangle_area : ℝ := 40
  let diagonal := Real.sqrt (width ^ 2 + length ^ 2)
  let side1 := Real.sqrt (width ^ 2 + (h / 2) ^ 2)
  let side2 := Real.sqrt (length ^ 2 + (h / 2) ^ 2)
  triangle_area = Real.sqrt (
    (diagonal + side1 + side2) *
    (diagonal + side1 - side2) *
    (diagonal - side1 + side2) *
    (-diagonal + side1 + side2)
  ) / 4
  →
  h = 24 * Real.sqrt 21 / 5 := by
sorry

end NUMINAMATH_CALUDE_box_height_l665_66561


namespace NUMINAMATH_CALUDE_set_operations_l665_66599

def A : Set ℤ := {x | |x| ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

theorem set_operations :
  (A ∩ (B ∩ C) = {3}) ∧
  (A ∩ (A \ (B ∪ C)) = {-6, -5, -4, -3, -2, -1, 0}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l665_66599


namespace NUMINAMATH_CALUDE_cost_of_300_pencils_l665_66554

/-- The cost of pencils in dollars -/
def cost_in_dollars (num_pencils : ℕ) (cost_per_pencil_cents : ℕ) (cents_per_dollar : ℕ) : ℚ :=
  (num_pencils * cost_per_pencil_cents : ℚ) / cents_per_dollar

/-- Theorem: The cost of 300 pencils is 7.5 dollars -/
theorem cost_of_300_pencils :
  cost_in_dollars 300 5 200 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_300_pencils_l665_66554


namespace NUMINAMATH_CALUDE_fraction_simplification_l665_66542

theorem fraction_simplification :
  (5 : ℚ) / ((8 : ℚ) / 13 + (1 : ℚ) / 13) = 65 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l665_66542


namespace NUMINAMATH_CALUDE_rachel_total_steps_l665_66562

/-- The total number of steps Rachel took during her trip to the Eiffel Tower -/
def total_steps (steps_up steps_down : ℕ) : ℕ := steps_up + steps_down

/-- Theorem stating that Rachel took 892 steps in total -/
theorem rachel_total_steps : total_steps 567 325 = 892 := by
  sorry

end NUMINAMATH_CALUDE_rachel_total_steps_l665_66562


namespace NUMINAMATH_CALUDE_ruler_cost_l665_66570

theorem ruler_cost (total_students : ℕ) (buyers : ℕ) (rulers_per_student : ℕ) (ruler_cost : ℕ) :
  total_students = 36 →
  buyers > total_students / 2 →
  rulers_per_student > 1 →
  ruler_cost > rulers_per_student →
  buyers * rulers_per_student * ruler_cost = 1729 →
  ruler_cost = 13 :=
by sorry

end NUMINAMATH_CALUDE_ruler_cost_l665_66570


namespace NUMINAMATH_CALUDE_three_tangents_implies_a_greater_than_three_l665_66523

/-- A curve of the form y = x³ + ax² + bx -/
structure Curve where
  a : ℝ
  b : ℝ

/-- The number of tangent lines to the curve that pass through (0,-1) -/
noncomputable def numTangentLines (c : Curve) : ℕ := sorry

/-- Theorem stating that if there are exactly three tangent lines passing through (0,-1), then a > 3 -/
theorem three_tangents_implies_a_greater_than_three (c : Curve) :
  numTangentLines c = 3 → c.a > 3 := by sorry

end NUMINAMATH_CALUDE_three_tangents_implies_a_greater_than_three_l665_66523


namespace NUMINAMATH_CALUDE_rs_value_l665_66543

theorem rs_value (r s : ℝ) (hr : 0 < r) (hs : 0 < s) 
  (h1 : r^2 + s^2 = 2) (h2 : r^4 + s^4 = 9/8) : r * s = Real.sqrt 23 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rs_value_l665_66543


namespace NUMINAMATH_CALUDE_hallway_area_in_sq_yards_l665_66595

-- Define the dimensions of the hallway
def hallway_length : ℝ := 15
def hallway_width : ℝ := 4

-- Define the conversion factor from square feet to square yards
def sq_feet_per_sq_yard : ℝ := 9

-- Theorem statement
theorem hallway_area_in_sq_yards :
  (hallway_length * hallway_width) / sq_feet_per_sq_yard = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hallway_area_in_sq_yards_l665_66595


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l665_66573

theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) : 
  B = 60 * π / 180 →
  b = 7 * Real.sqrt 6 →
  a = 14 →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  A = 45 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l665_66573


namespace NUMINAMATH_CALUDE_no_partition_of_integers_l665_66540

theorem no_partition_of_integers : ¬ ∃ (A B C : Set ℤ), 
  (A ∪ B ∪ C = Set.univ) ∧ 
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
  (∀ n : ℤ, (n ∈ A ∧ (n - 50) ∈ B ∧ (n + 2011) ∈ C) ∨
            (n ∈ A ∧ (n - 50) ∈ C ∧ (n + 2011) ∈ B) ∨
            (n ∈ B ∧ (n - 50) ∈ A ∧ (n + 2011) ∈ C) ∨
            (n ∈ B ∧ (n - 50) ∈ C ∧ (n + 2011) ∈ A) ∨
            (n ∈ C ∧ (n - 50) ∈ A ∧ (n + 2011) ∈ B) ∨
            (n ∈ C ∧ (n - 50) ∈ B ∧ (n + 2011) ∈ A)) :=
by
  sorry


end NUMINAMATH_CALUDE_no_partition_of_integers_l665_66540


namespace NUMINAMATH_CALUDE_yield_increase_correct_l665_66557

/-- The percentage increase in rice yield after each harvest -/
def yield_increase_percentage : ℝ := 20

/-- The initial harvest yield in sacks of rice -/
def initial_harvest : ℝ := 20

/-- The total yield after two harvests in sacks of rice -/
def total_yield_two_harvests : ℝ := 44

/-- Theorem stating that the given yield increase percentage is correct -/
theorem yield_increase_correct : 
  initial_harvest + initial_harvest * (1 + yield_increase_percentage / 100) = total_yield_two_harvests :=
by sorry

end NUMINAMATH_CALUDE_yield_increase_correct_l665_66557


namespace NUMINAMATH_CALUDE_average_weight_problem_l665_66545

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 43 →
  (b + c) / 2 = 42 →
  b = 51 →
  (a + b) / 2 = 48 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l665_66545


namespace NUMINAMATH_CALUDE_simultaneous_sequence_probability_l665_66546

-- Define the probabilities for each coin
def coin_a_heads : ℝ := 0.3
def coin_a_tails : ℝ := 0.7
def coin_b_heads : ℝ := 0.4
def coin_b_tails : ℝ := 0.6

-- Define the number of consecutive flips
def num_flips : ℕ := 6

-- Define the probability of the desired sequence for each coin
def prob_a_sequence : ℝ := coin_a_tails * coin_a_tails * coin_a_heads
def prob_b_sequence : ℝ := coin_b_heads * coin_b_tails * coin_b_tails

-- Theorem to prove
theorem simultaneous_sequence_probability :
  prob_a_sequence * prob_b_sequence = 0.021168 :=
sorry

end NUMINAMATH_CALUDE_simultaneous_sequence_probability_l665_66546


namespace NUMINAMATH_CALUDE_total_income_scientific_notation_exponent_l665_66525

/-- Represents the average annual income from 1 acre of medicinal herbs in dollars -/
def average_income_per_acre : ℝ := 20000

/-- Represents the number of acres of medicinal herbs planted in the county -/
def acres_planted : ℝ := 8000

/-- Calculates the total annual income from medicinal herbs in the county -/
def total_income : ℝ := average_income_per_acre * acres_planted

/-- Represents the exponent in the scientific notation of the total income -/
def n : ℕ := 8

/-- Theorem stating that the exponent in the scientific notation of the total income is 8 -/
theorem total_income_scientific_notation_exponent : 
  ∃ (a : ℝ), a > 1 ∧ a < 10 ∧ total_income = a * (10 : ℝ) ^ n :=
sorry

end NUMINAMATH_CALUDE_total_income_scientific_notation_exponent_l665_66525


namespace NUMINAMATH_CALUDE_chemistry_physics_difference_l665_66551

/-- Proves that the difference between chemistry and physics scores is 10 -/
theorem chemistry_physics_difference
  (M P C : ℕ)  -- Marks in Mathematics, Physics, and Chemistry
  (h1 : M + P = 60)  -- Sum of Mathematics and Physics scores
  (h2 : (M + C) / 2 = 35)  -- Average of Mathematics and Chemistry scores
  (h3 : C > P)  -- Chemistry score is higher than Physics score
  : C - P = 10 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_physics_difference_l665_66551


namespace NUMINAMATH_CALUDE_base6_division_theorem_l665_66571

/-- Convert a number from base 6 to base 10 -/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Convert a number from base 10 to base 6 -/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Perform division in base 6 -/
def divBase6 (a b : List Nat) : List Nat × Nat :=
  let a10 := base6ToBase10 a
  let b10 := base6ToBase10 b
  let q := a10 / b10
  let r := a10 % b10
  (base10ToBase6 q, r)

theorem base6_division_theorem :
  let a := [3, 2, 1, 2]  -- 2123 in base 6
  let b := [3, 2]        -- 23 in base 6
  let (q, r) := divBase6 a b
  q = [2, 5] ∧ r = 3 := by
  sorry

end NUMINAMATH_CALUDE_base6_division_theorem_l665_66571


namespace NUMINAMATH_CALUDE_rays_fish_market_rays_fish_market_specific_l665_66575

/-- The number of customers who will not receive fish in Mr. Ray's fish market scenario -/
theorem rays_fish_market (total_customers : ℕ) (num_tuna : ℕ) (tuna_weight : ℕ) (customer_request : ℕ) : ℕ :=
  let total_fish := num_tuna * tuna_weight
  let served_customers := total_fish / customer_request
  total_customers - served_customers

/-- Proof of the specific scenario in Mr. Ray's fish market -/
theorem rays_fish_market_specific : rays_fish_market 100 10 200 25 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rays_fish_market_rays_fish_market_specific_l665_66575


namespace NUMINAMATH_CALUDE_lineup_selections_15_l665_66504

/-- The number of ways to select an ordered lineup of 5 players and 1 substitute from 15 players -/
def lineup_selections (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)

/-- Theorem stating that the number of lineup selections from 15 players is 3,276,000 -/
theorem lineup_selections_15 :
  lineup_selections 15 = 3276000 := by
  sorry

#eval lineup_selections 15

end NUMINAMATH_CALUDE_lineup_selections_15_l665_66504


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l665_66559

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 - 64*y^2 + 16*x - 32 = 0

/-- Definition of a hyperbola -/
def is_hyperbola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
  (∀ x y, eq x y ↔ ((x - c)^2 / a^2) - ((y - d)^2 / b^2) = 1) ∨
  (∀ x y, eq x y ↔ ((y - d)^2 / a^2) - ((x - c)^2 / b^2) = 1)

/-- Theorem: The given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l665_66559


namespace NUMINAMATH_CALUDE_increasing_cubic_function_condition_l665_66552

/-- The function f(x) = 2x^3 - 3mx^2 + 6x is increasing on (1, +∞) if and only if m ≤ 2 -/
theorem increasing_cubic_function_condition (m : ℝ) :
  (∀ x > 1, Monotone (fun x => 2*x^3 - 3*m*x^2 + 6*x)) ↔ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_condition_l665_66552


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_l665_66550

/-- Represents a rhombus with given area and one diagonal -/
structure Rhombus where
  area : ℝ
  diagonal1 : ℝ

/-- Theorem: In a rhombus with area 60 cm² and one diagonal 12 cm, the other diagonal is 10 cm -/
theorem rhombus_other_diagonal (r : Rhombus) (h1 : r.area = 60) (h2 : r.diagonal1 = 12) :
  ∃ (diagonal2 : ℝ), diagonal2 = 10 ∧ r.area = (r.diagonal1 * diagonal2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_l665_66550


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l665_66538

theorem smaller_solution_quadratic_equation :
  ∃ (x y : ℝ), x < y ∧ 
  x^2 - 9*x - 22 = 0 ∧ 
  y^2 - 9*y - 22 = 0 ∧
  ∀ z : ℝ, z^2 - 9*z - 22 = 0 → z = x ∨ z = y ∧
  x = -2 := by
sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_equation_l665_66538


namespace NUMINAMATH_CALUDE_sidney_kittens_l665_66555

/-- The number of kittens Sidney has -/
def num_kittens : ℕ := sorry

/-- The number of adult cats Sidney has -/
def num_adult_cats : ℕ := 3

/-- The number of cans Sidney already has -/
def initial_cans : ℕ := 7

/-- The number of additional cans Sidney needs to buy -/
def additional_cans : ℕ := 35

/-- The number of days Sidney needs to feed the cats -/
def num_days : ℕ := 7

/-- The amount of food (in cans) an adult cat eats per day -/
def adult_cat_food_per_day : ℚ := 1

/-- The amount of food (in cans) a kitten eats per day -/
def kitten_food_per_day : ℚ := 3/4

theorem sidney_kittens : 
  num_kittens = 4 ∧
  (num_kittens : ℚ) * kitten_food_per_day * num_days + 
  (num_adult_cats : ℚ) * adult_cat_food_per_day * num_days = 
  initial_cans + additional_cans :=
sorry

end NUMINAMATH_CALUDE_sidney_kittens_l665_66555


namespace NUMINAMATH_CALUDE_expression_equals_hundred_l665_66535

theorem expression_equals_hundred : (7.5 * 7.5 + 37.5 + 2.5 * 2.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_hundred_l665_66535


namespace NUMINAMATH_CALUDE_max_third_place_books_l665_66514

structure BookDistribution where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

def is_valid_distribution (d : BookDistribution) : Prop :=
  d.first > d.second ∧
  d.second > d.third ∧
  d.third > d.fourth ∧
  d.fourth > d.fifth ∧
  d.first % 100 = 0 ∧
  d.second % 100 = 0 ∧
  d.third % 100 = 0 ∧
  d.fourth % 100 = 0 ∧
  d.fifth % 100 = 0 ∧
  d.first = d.second + d.third ∧
  d.second = d.fourth + d.fifth ∧
  d.first + d.second + d.third + d.fourth + d.fifth ≤ 10000

theorem max_third_place_books :
  ∀ d : BookDistribution,
    is_valid_distribution d →
    d.third ≤ 1900 :=
by sorry

end NUMINAMATH_CALUDE_max_third_place_books_l665_66514


namespace NUMINAMATH_CALUDE_root_difference_quadratic_equation_l665_66586

theorem root_difference_quadratic_equation : 
  let a : ℝ := 2
  let b : ℝ := 5
  let c : ℝ := -12
  let discriminant := b^2 - 4*a*c
  let root1 := (-b + Real.sqrt discriminant) / (2*a)
  let root2 := (-b - Real.sqrt discriminant) / (2*a)
  abs (root1 - root2) = 5.5 := by
sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_equation_l665_66586


namespace NUMINAMATH_CALUDE_balloon_arrangements_l665_66576

def balloon_permutations : ℕ := 1260

theorem balloon_arrangements :
  (7 * 6 * 5 * 4 * 3) / 2 = balloon_permutations := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l665_66576


namespace NUMINAMATH_CALUDE_rotten_apples_percentage_l665_66598

theorem rotten_apples_percentage (total : ℕ) (good : ℕ) 
  (h1 : total = 75) (h2 : good = 66) : 
  (((total - good : ℚ) / total) * 100 : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_rotten_apples_percentage_l665_66598


namespace NUMINAMATH_CALUDE_jennifer_future_age_l665_66526

def jennifer_age_in_10_years : ℕ := 30

def jordana_current_age : ℕ := 80

theorem jennifer_future_age :
  jennifer_age_in_10_years = 30 :=
by
  have h1 : jordana_current_age + 10 = 3 * jennifer_age_in_10_years :=
    sorry
  sorry

#check jennifer_future_age

end NUMINAMATH_CALUDE_jennifer_future_age_l665_66526


namespace NUMINAMATH_CALUDE_no_opposite_midpoints_l665_66532

/-- Represents a rectangular billiard table -/
structure BilliardTable where
  length : ℝ
  width : ℝ
  corner_pockets : Bool

/-- Represents the trajectory of a ball on the billiard table -/
structure BallTrajectory where
  table : BilliardTable
  start_corner : Fin 4
  angle : ℝ

/-- Predicate to check if a point is on the midpoint of a side -/
def is_side_midpoint (table : BilliardTable) (x y : ℝ) : Prop :=
  (x = 0 ∧ y = table.width / 2) ∨
  (x = table.length ∧ y = table.width / 2) ∨
  (y = 0 ∧ x = table.length / 2) ∨
  (y = table.width ∧ x = table.length / 2)

/-- Theorem stating that a ball cannot visit midpoints of opposite sides -/
theorem no_opposite_midpoints (trajectory : BallTrajectory) 
  (h1 : trajectory.angle = π/4)
  (h2 : ∃ (x1 y1 : ℝ), is_side_midpoint trajectory.table x1 y1) :
  ¬ ∃ (x2 y2 : ℝ), 
    is_side_midpoint trajectory.table x2 y2 ∧ 
    ((x1 = 0 ∧ x2 = trajectory.table.length) ∨ 
     (x1 = trajectory.table.length ∧ x2 = 0) ∨
     (y1 = 0 ∧ y2 = trajectory.table.width) ∨
     (y1 = trajectory.table.width ∧ y2 = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_opposite_midpoints_l665_66532


namespace NUMINAMATH_CALUDE_pedestrian_cyclist_speeds_l665_66501

theorem pedestrian_cyclist_speeds
  (distance : ℝ)
  (pedestrian_start : ℝ)
  (cyclist1_start : ℝ)
  (cyclist2_start : ℝ)
  (pedestrian_speed : ℝ)
  (cyclist_speed : ℝ)
  (h1 : distance = 40)
  (h2 : cyclist1_start - pedestrian_start = 10/3)
  (h3 : cyclist2_start - pedestrian_start = 4.5)
  (h4 : pedestrian_speed * ((cyclist1_start - pedestrian_start) + (distance/2 - pedestrian_speed * (cyclist1_start - pedestrian_start)) / (cyclist_speed - pedestrian_speed)) = distance/2)
  (h5 : pedestrian_speed * ((cyclist1_start - pedestrian_start) + (distance/2 - pedestrian_speed * (cyclist1_start - pedestrian_start)) / (cyclist_speed - pedestrian_speed) + 1) + cyclist_speed * ((cyclist2_start - pedestrian_start) - ((cyclist1_start - pedestrian_start) + (distance/2 - pedestrian_speed * (cyclist1_start - pedestrian_start)) / (cyclist_speed - pedestrian_speed) + 1)) = distance)
  : pedestrian_speed = 5 ∧ cyclist_speed = 30 := by
  sorry

end NUMINAMATH_CALUDE_pedestrian_cyclist_speeds_l665_66501


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l665_66590

theorem polar_coordinates_of_point (x y : ℝ) (ρ θ : ℝ) :
  x = 1 ∧ y = -Real.sqrt 3 →
  ρ = 2 ∧ θ = 5 * Real.pi / 3 →
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l665_66590


namespace NUMINAMATH_CALUDE_swimming_pool_capacity_l665_66544

theorem swimming_pool_capacity (initial_fraction : ℚ) (added_amount : ℚ) (final_fraction : ℚ) :
  initial_fraction = 1/3 →
  added_amount = 180 →
  final_fraction = 4/5 →
  ∃ (total_capacity : ℚ), 
    total_capacity * initial_fraction + added_amount = total_capacity * final_fraction ∧
    total_capacity = 2700/7 := by
  sorry

#eval (2700 : ℚ) / 7

end NUMINAMATH_CALUDE_swimming_pool_capacity_l665_66544


namespace NUMINAMATH_CALUDE_garden_perimeter_l665_66574

/-- The perimeter of a rectangular garden with width 4 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is 104 meters. -/
theorem garden_perimeter : 
  let playground_length : ℝ := 16
  let playground_width : ℝ := 12
  let garden_width : ℝ := 4
  let playground_area := playground_length * playground_width
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * (garden_length + garden_width)
  garden_perimeter = 104 := by sorry

end NUMINAMATH_CALUDE_garden_perimeter_l665_66574


namespace NUMINAMATH_CALUDE_rosie_account_balance_l665_66584

/-- Represents the total amount in Rosie's account after m deposits -/
def account_balance (initial_amount : ℕ) (deposit_amount : ℕ) (num_deposits : ℕ) : ℕ :=
  initial_amount + deposit_amount * num_deposits

/-- Theorem stating that Rosie's account balance is correctly represented -/
theorem rosie_account_balance (m : ℕ) : 
  account_balance 120 30 m = 120 + 30 * m := by
  sorry

#check rosie_account_balance

end NUMINAMATH_CALUDE_rosie_account_balance_l665_66584


namespace NUMINAMATH_CALUDE_factorial_expression_equals_2884_l665_66528

theorem factorial_expression_equals_2884 :
  (Real.sqrt ((Nat.factorial 5 * Nat.factorial 4) + 2^2))^2 = 2884 := by sorry

end NUMINAMATH_CALUDE_factorial_expression_equals_2884_l665_66528
