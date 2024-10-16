import Mathlib

namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3709_370919

theorem fahrenheit_to_celsius (C F : ℝ) : C = (5 / 9) * (F - 32) → C = 20 → F = 68 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l3709_370919


namespace NUMINAMATH_CALUDE_same_type_quadratic_root_l3709_370959

theorem same_type_quadratic_root (a : ℝ) : 
  (∃ (k : ℝ), k^2 = 12 ∧ k^2 = 2*a - 5) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_same_type_quadratic_root_l3709_370959


namespace NUMINAMATH_CALUDE_erased_number_l3709_370999

/-- Given nine consecutive integers where the sum of eight of them is 1703, prove that the missing number is 214. -/
theorem erased_number (a : ℤ) (b : ℤ) (h1 : -4 ≤ b ∧ b ≤ 4) (h2 : 8*a - b = 1703) : a + b = 214 := by
  sorry

end NUMINAMATH_CALUDE_erased_number_l3709_370999


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l3709_370917

/-- Represents a 3x3x3 cube made up of smaller cubes -/
structure LargeCube :=
  (small_cubes : Fin 3 → Fin 3 → Fin 3 → Bool)

/-- Represents the shading pattern on a face of the large cube -/
inductive FaceShading
  | FourCorners
  | LShape

/-- Represents the shading of opposite faces -/
structure OppositeShading :=
  (face1 : FaceShading)
  (face2 : FaceShading)

/-- The shading pattern for all three pairs of opposite faces -/
def cube_shading : Fin 3 → OppositeShading :=
  λ _ => { face1 := FaceShading.FourCorners, face2 := FaceShading.LShape }

/-- Counts the number of smaller cubes with at least one face shaded -/
def count_shaded_cubes (c : LargeCube) (shading : Fin 3 → OppositeShading) : Nat :=
  sorry

theorem shaded_cubes_count :
  ∀ c : LargeCube,
  count_shaded_cubes c cube_shading = 17 :=
sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l3709_370917


namespace NUMINAMATH_CALUDE_jacket_final_price_l3709_370976

/-- Calculates the final price of an item after two discounts and a tax --/
def finalPrice (originalPrice firstDiscount secondDiscount taxRate : ℝ) : ℝ :=
  let priceAfterFirstDiscount := originalPrice * (1 - firstDiscount)
  let priceAfterSecondDiscount := priceAfterFirstDiscount * (1 - secondDiscount)
  priceAfterSecondDiscount * (1 + taxRate)

/-- Theorem stating that the final price of the jacket is approximately $77.11 --/
theorem jacket_final_price :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (finalPrice 120 0.3 0.15 0.08 - 77.11) < ε :=
sorry

end NUMINAMATH_CALUDE_jacket_final_price_l3709_370976


namespace NUMINAMATH_CALUDE_dining_table_original_price_l3709_370962

theorem dining_table_original_price (discount_percentage : ℝ) (sale_price : ℝ) (original_price : ℝ) : 
  discount_percentage = 10 →
  sale_price = 450 →
  sale_price = original_price * (1 - discount_percentage / 100) →
  original_price = 500 := by
sorry

end NUMINAMATH_CALUDE_dining_table_original_price_l3709_370962


namespace NUMINAMATH_CALUDE_second_exponent_base_l3709_370946

theorem second_exponent_base (x b : ℕ) (h1 : b > 0) (h2 : (18^6) * (x^17) = (2^6) * (3^b)) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_exponent_base_l3709_370946


namespace NUMINAMATH_CALUDE_f_properties_l3709_370905

def f (x : ℝ) : ℝ := -x - x^3

theorem f_properties (x₁ x₂ : ℝ) (h : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧ (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3709_370905


namespace NUMINAMATH_CALUDE_integer_fraction_implication_l3709_370970

theorem integer_fraction_implication (m n p q : ℕ) (h1 : m ≠ p) 
  (h2 : ∃ k : ℤ, k = (m * n + p * q) / (m - p)) : 
  ∃ l : ℤ, l = (m * q + n * p) / (m - p) := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_implication_l3709_370970


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3709_370944

theorem condition_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 0 ∧ b > 0 → a^2 + b^2 ≥ 2*a*b) ∧
  ¬(∀ a b : ℝ, a^2 + b^2 ≥ 2*a*b → a > 0 ∧ b > 0) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3709_370944


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l3709_370927

theorem recipe_flour_amount (flour_added : ℕ) (flour_needed : ℕ) : 
  flour_added = 2 → flour_needed = 6 → flour_added + flour_needed = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l3709_370927


namespace NUMINAMATH_CALUDE_ford_vehicle_count_l3709_370975

/-- Represents the number of vehicles of each brand on Louie's store parking lot -/
structure VehicleCounts where
  D : ℕ  -- Dodge
  H : ℕ  -- Hyundai
  K : ℕ  -- Kia
  Ho : ℕ -- Honda
  F : ℕ  -- Ford

/-- Conditions for the vehicle counts -/
def satisfiesConditions (v : VehicleCounts) : Prop :=
  v.D + v.H + v.K + v.Ho + v.F = 1000 ∧
  (35 : ℕ) * (v.D + v.H + v.K + v.Ho + v.F) = 100 * v.D ∧
  (10 : ℕ) * (v.D + v.H + v.K + v.Ho + v.F) = 100 * v.H ∧
  v.K = 2 * v.Ho + 50 ∧
  v.F = v.D - 200

theorem ford_vehicle_count (v : VehicleCounts) 
  (h : satisfiesConditions v) : v.F = 150 := by
  sorry

end NUMINAMATH_CALUDE_ford_vehicle_count_l3709_370975


namespace NUMINAMATH_CALUDE_unique_a_with_integer_solutions_l3709_370931

theorem unique_a_with_integer_solutions : 
  ∃! a : ℕ+, (a : ℝ) ≤ 100 ∧ 
  ∃ x y : ℤ, x ≠ y ∧ 
  (x : ℝ)^2 + (2 * (a : ℝ) - 3) * (x : ℝ) + ((a : ℝ) - 1)^2 = 0 ∧
  (y : ℝ)^2 + (2 * (a : ℝ) - 3) * (y : ℝ) + ((a : ℝ) - 1)^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_a_with_integer_solutions_l3709_370931


namespace NUMINAMATH_CALUDE_square_side_length_l3709_370910

theorem square_side_length (perimeter : ℝ) (h : perimeter = 17.8) :
  perimeter / 4 = 4.45 := by sorry

end NUMINAMATH_CALUDE_square_side_length_l3709_370910


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l3709_370980

/-- Given a line passing through points (1, 4) and (-2, -2), prove that the product of its slope and y-intercept is 4. -/
theorem line_slope_intercept_product (m b : ℝ) : 
  (4 = m * 1 + b) → 
  (-2 = m * (-2) + b) → 
  m * b = 4 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l3709_370980


namespace NUMINAMATH_CALUDE_original_people_count_l3709_370924

theorem original_people_count (x : ℚ) : 
  (2 * x / 3 + 6 - x / 6 = 15) → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_original_people_count_l3709_370924


namespace NUMINAMATH_CALUDE_weaving_time_approx_l3709_370913

/-- The time taken to weave a certain amount of cloth, given the weaving rate and total time -/
def weaving_time (rate : Real) (total_time : Real) : Real :=
  total_time

theorem weaving_time_approx :
  let rate := 1.14  -- meters per second
  let total_time := 45.6140350877193  -- seconds
  ∃ ε > 0, |weaving_time rate total_time - 45.614| < ε :=
sorry

end NUMINAMATH_CALUDE_weaving_time_approx_l3709_370913


namespace NUMINAMATH_CALUDE_four_numbers_lcm_l3709_370921

theorem four_numbers_lcm (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a + b + c + d = 2020 →
  Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 202 →
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 2424 := by
sorry

end NUMINAMATH_CALUDE_four_numbers_lcm_l3709_370921


namespace NUMINAMATH_CALUDE_number_of_bad_oranges_l3709_370978

/-- Given a basket with good and bad oranges, where the number of good oranges
    is known and the ratio of good to bad oranges is given, this theorem proves
    the number of bad oranges. -/
theorem number_of_bad_oranges
  (good_oranges : ℕ)
  (ratio_good : ℕ)
  (ratio_bad : ℕ)
  (h1 : good_oranges = 24)
  (h2 : ratio_good = 3)
  (h3 : ratio_bad = 1)
  : ∃ bad_oranges : ℕ, bad_oranges = 8 ∧ good_oranges * ratio_bad = bad_oranges * ratio_good :=
by
  sorry


end NUMINAMATH_CALUDE_number_of_bad_oranges_l3709_370978


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3709_370995

theorem sphere_surface_area (d : ℝ) (h : d = 2) : 
  4 * Real.pi * (d / 2)^2 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3709_370995


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3709_370912

theorem min_value_trig_expression :
  ∃ (x : ℝ), ∀ (y : ℝ),
    (Real.sin y)^8 + (Real.cos y)^8 + 1
    ≤ ((Real.sin x)^8 + (Real.cos x)^8 + 1) / ((Real.sin x)^6 + (Real.cos x)^6 + 1)
    ∧ ((Real.sin x)^8 + (Real.cos x)^8 + 1) / ((Real.sin x)^6 + (Real.cos x)^6 + 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3709_370912


namespace NUMINAMATH_CALUDE_angle_PQ_A1BD_l3709_370918

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D

/-- Represents a line in 3D space -/
structure Line where
  direction : Point3D

/-- Calculates the reflection of a point with respect to a plane -/
def reflect_point_plane (p : Point3D) (plane : Plane) : Point3D :=
  sorry

/-- Calculates the reflection of a point with respect to a line -/
def reflect_point_line (p : Point3D) (line : Line) : Point3D :=
  sorry

/-- Calculates the angle between a line and a plane -/
def angle_line_plane (line : Line) (plane : Plane) : ℝ :=
  sorry

theorem angle_PQ_A1BD (cube : Cube) : 
  let C1BD : Plane := sorry
  let B1D : Line := sorry
  let A1BD : Plane := sorry
  let P : Point3D := reflect_point_plane cube.A C1BD
  let Q : Point3D := reflect_point_line cube.A B1D
  let PQ : Line := { direction := sorry }
  Real.sin (angle_line_plane PQ A1BD) = 2 * Real.sqrt 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_PQ_A1BD_l3709_370918


namespace NUMINAMATH_CALUDE_min_posts_for_specific_plot_l3709_370940

/-- Calculates the number of fence posts required for a given length -/
def posts_for_length (length : ℕ) : ℕ :=
  length / 10 + 1

/-- Represents a rectangular garden plot -/
structure GardenPlot where
  width : ℕ
  length : ℕ
  wall_length : ℕ

/-- Calculates the minimum number of fence posts required for a garden plot -/
def min_posts (plot : GardenPlot) : ℕ :=
  posts_for_length plot.length + 2 * (posts_for_length plot.width - 1)

/-- Theorem stating the minimum number of posts for the specific garden plot -/
theorem min_posts_for_specific_plot :
  ∃ (plot : GardenPlot), plot.width = 30 ∧ plot.length = 50 ∧ plot.wall_length = 80 ∧ min_posts plot = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_posts_for_specific_plot_l3709_370940


namespace NUMINAMATH_CALUDE_bons_win_probability_main_theorem_l3709_370974

/-- The probability of rolling a six. -/
def prob_six : ℚ := 1/6

/-- The probability of not rolling a six. -/
def prob_not_six : ℚ := 1 - prob_six

/-- The probability that Mr. B. Bons wins the game. -/
def prob_bons_win : ℚ := 5/11

theorem bons_win_probability :
  prob_bons_win = prob_not_six * prob_six + prob_not_six * prob_not_six * prob_bons_win :=
by sorry

/-- The main theorem stating that the probability of Mr. B. Bons winning is 5/11. -/
theorem main_theorem : prob_bons_win = 5/11 :=
by sorry

end NUMINAMATH_CALUDE_bons_win_probability_main_theorem_l3709_370974


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_value_l3709_370967

theorem mean_equality_implies_x_value : ∃ x : ℝ,
  (7 + 9 + 23) / 3 = (16 + x) / 2 → x = 10 := by sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_value_l3709_370967


namespace NUMINAMATH_CALUDE_brownie_calories_l3709_370954

-- Define the parameters of the problem
def cake_slices : ℕ := 8
def calories_per_cake_slice : ℕ := 347
def brownies : ℕ := 6
def calorie_difference : ℕ := 526

-- Define the function to calculate calories per brownie
def calories_per_brownie : ℕ :=
  ((cake_slices * calories_per_cake_slice - calorie_difference) / brownies : ℕ)

-- Theorem statement
theorem brownie_calories :
  calories_per_brownie = 375 := by
  sorry

end NUMINAMATH_CALUDE_brownie_calories_l3709_370954


namespace NUMINAMATH_CALUDE_fishing_ratio_l3709_370936

theorem fishing_ratio (sara_catch melanie_catch : ℕ) 
  (h1 : sara_catch = 5)
  (h2 : melanie_catch = 10) :
  (melanie_catch : ℚ) / sara_catch = 2 := by
  sorry

end NUMINAMATH_CALUDE_fishing_ratio_l3709_370936


namespace NUMINAMATH_CALUDE_school_water_cases_l3709_370933

theorem school_water_cases : 
  ∀ (bottles_per_case : ℕ) 
    (bottles_used_first_game : ℕ) 
    (bottles_used_second_game : ℕ) 
    (bottles_left : ℕ),
  bottles_per_case = 20 →
  bottles_used_first_game = 70 →
  bottles_used_second_game = 110 →
  bottles_left = 20 →
  (bottles_used_first_game + bottles_used_second_game + bottles_left) / bottles_per_case = 10 := by
sorry

end NUMINAMATH_CALUDE_school_water_cases_l3709_370933


namespace NUMINAMATH_CALUDE_fraction_sum_constraint_l3709_370984

theorem fraction_sum_constraint (n : ℕ) (hn : n > 0) :
  (1 : ℚ) / 2 + 1 / 3 + 1 / 10 + 1 / n < 1 → n > 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_constraint_l3709_370984


namespace NUMINAMATH_CALUDE_point_c_range_l3709_370993

-- Define the parabola
def on_parabola (x y : ℝ) : Prop := y^2 = x + 4

-- Define perpendicularity
def perpendicular (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : Prop :=
  (y2 - y1) * (y4 - y3) = -(x2 - x1) * (x4 - x3)

theorem point_c_range :
  ∀ (x y : ℝ),
    on_parabola x y →
    (∃ (x1 y1 : ℝ),
      on_parabola x1 y1 ∧
      perpendicular 0 2 x1 y1 x1 y1 x y) →
    y ≤ 0 ∨ y ≥ 4 := by sorry

end NUMINAMATH_CALUDE_point_c_range_l3709_370993


namespace NUMINAMATH_CALUDE_fraction_irreducibility_l3709_370968

theorem fraction_irreducibility (n : ℕ) : 
  (Nat.gcd (2*n^2 + 11*n - 18) (n + 7) = 1) ↔ (n % 3 = 0 ∨ n % 3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_irreducibility_l3709_370968


namespace NUMINAMATH_CALUDE_cistern_fill_time_l3709_370930

theorem cistern_fill_time (empty_rate : ℝ) (combined_fill_time : ℝ) (fill_time : ℝ) : 
  empty_rate = 1 / 9 →
  combined_fill_time = 7 / 3 →
  1 / fill_time - empty_rate = 1 / combined_fill_time →
  fill_time = 63 / 34 := by
sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l3709_370930


namespace NUMINAMATH_CALUDE_even_function_property_l3709_370945

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The main theorem -/
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : IsEven f) 
  (h_neg : ∀ x < 0, f x = 3 * x - 1) : 
  ∀ x > 0, f x = -3 * x - 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_property_l3709_370945


namespace NUMINAMATH_CALUDE_right_pyramid_base_side_length_l3709_370925

/-- Represents a right pyramid with an equilateral triangular base -/
structure RightPyramid where
  base_side_length : ℝ
  slant_height : ℝ
  lateral_face_area : ℝ

/-- Theorem: If the area of one lateral face is 90 square meters and the slant height is 20 meters,
    then the side length of the base is 9 meters -/
theorem right_pyramid_base_side_length 
  (pyramid : RightPyramid) 
  (h1 : pyramid.lateral_face_area = 90) 
  (h2 : pyramid.slant_height = 20) : 
  pyramid.base_side_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_right_pyramid_base_side_length_l3709_370925


namespace NUMINAMATH_CALUDE_lucky_larry_problem_l3709_370915

theorem lucky_larry_problem (p q r s t : ℤ) 
  (hp : p = 2) (hq : q = 4) (hr : r = 6) (hs : s = 8) :
  p - (q - (r - (s - t))) = p - q - r + s - t → t = 2 := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_problem_l3709_370915


namespace NUMINAMATH_CALUDE_equation_solution_l3709_370986

theorem equation_solution : 
  ∀ x : ℝ, (Real.sqrt (5 * x - 2) + 12 / Real.sqrt (5 * x - 2) = 8) ↔ 
  (x = 38 / 5 ∨ x = 6 / 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3709_370986


namespace NUMINAMATH_CALUDE_min_value_expression_l3709_370956

theorem min_value_expression (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_one : x + y + z + w = 1) (x_eq_y : x = y) :
  ∀ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d = 1 ∧ a = b →
  (a + b + c) / (a * b * c * d) ≥ (x + y + z) / (x * y * z * w) ∧
  (x + y + z) / (x * y * z * w) ≥ 1024 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3709_370956


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l3709_370958

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_odd_product : a 1 * a 3 * a 5 * a 7 * a 9 = 2)
  (h_even_product : a 2 * a 4 * a 6 * a 8 * a 10 = 64) :
  q = 2 :=
sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l3709_370958


namespace NUMINAMATH_CALUDE_sum_interior_eighth_row_l3709_370977

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def sum_interior (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The row number where interior numbers begin in Pascal's Triangle -/
def interior_start : ℕ := 3

theorem sum_interior_eighth_row :
  sum_interior 6 = 30 →
  sum_interior 8 = 126 :=
by sorry

end NUMINAMATH_CALUDE_sum_interior_eighth_row_l3709_370977


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3709_370904

theorem geometric_sequence_problem (a b c d e : ℕ) : 
  (2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 100) →
  Nat.gcd a e = 1 →
  (∃ (r : ℚ), b = a * r ∧ c = a * r^2 ∧ d = a * r^3 ∧ e = a * r^4) →
  c = 36 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3709_370904


namespace NUMINAMATH_CALUDE_divisible_by_six_l3709_370969

theorem divisible_by_six (n : ℕ) : ∃ k : ℤ, (n : ℤ)^3 + 5*n = 6*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l3709_370969


namespace NUMINAMATH_CALUDE_paper_crane_ratio_l3709_370972

/-- Represents the number of paper cranes Alice wants in total -/
def total_cranes : ℕ := 1000

/-- Represents the number of paper cranes Alice still needs to fold -/
def remaining_cranes : ℕ := 400

/-- Represents the ratio of cranes folded by Alice's friend to remaining cranes after Alice folded half -/
def friend_to_remaining_ratio : Rat := 1 / 5

theorem paper_crane_ratio :
  let alice_folded := total_cranes / 2
  let remaining_after_alice := total_cranes - alice_folded
  let friend_folded := remaining_after_alice - remaining_cranes
  friend_folded / remaining_after_alice = friend_to_remaining_ratio := by
sorry

end NUMINAMATH_CALUDE_paper_crane_ratio_l3709_370972


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3709_370923

theorem fractional_equation_solution :
  ∃ x : ℝ, (x + 1) / x = 2 / 3 ∧ x = -3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3709_370923


namespace NUMINAMATH_CALUDE_tray_height_l3709_370964

theorem tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) : 
  side_length = 120 →
  cut_distance = Real.sqrt 20 →
  cut_angle = π / 4 →
  ∃ (height : ℝ), height = Real.sqrt 10 ∧ 
    height = (cut_distance * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_tray_height_l3709_370964


namespace NUMINAMATH_CALUDE_sports_conference_games_l3709_370966

/-- Calculates the total number of games in a sports conference season -/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  (total_teams * (intra_division_games * (teams_per_division - 1) + 
  inter_division_games * teams_per_division)) / 2

/-- Theorem stating the total number of games in the given sports conference -/
theorem sports_conference_games : 
  total_games 16 8 2 1 = 176 := by
  sorry

end NUMINAMATH_CALUDE_sports_conference_games_l3709_370966


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l3709_370981

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l3709_370981


namespace NUMINAMATH_CALUDE_angle_ABH_measure_l3709_370937

/-- A regular octagon is a polygon with 8 equal sides and 8 equal angles. -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The measure of angle ABH in a regular octagon ABCDEFGH. -/
def angle_ABH (octagon : RegularOctagon) : ℝ := sorry

/-- Theorem: The measure of angle ABH in a regular octagon is 22.5 degrees. -/
theorem angle_ABH_measure (octagon : RegularOctagon) : 
  angle_ABH octagon = 22.5 := by sorry

end NUMINAMATH_CALUDE_angle_ABH_measure_l3709_370937


namespace NUMINAMATH_CALUDE_larger_number_problem_l3709_370909

theorem larger_number_problem (x y : ℕ) 
  (h1 : y - x = 1365)
  (h2 : y = 6 * x + 15) : 
  y = 1635 := by sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3709_370909


namespace NUMINAMATH_CALUDE_parabola_and_line_intersection_l3709_370951

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = -2*p*y
def line1 (x y : ℝ) : Prop := y = (1/2)*x - 1
def line2 (k : ℝ) (x y : ℝ) : Prop := y = k*x - 3/2

-- Define the theorem
theorem parabola_and_line_intersection 
  (p : ℝ) 
  (x_M y_M x_N y_N : ℝ) 
  (h_p : p > 0)
  (h_intersect1 : line1 x_M y_M ∧ line1 x_N y_N)
  (h_parabola1 : parabola p x_M y_M ∧ parabola p x_N y_N)
  (h_condition : (x_M + 1) * (x_N + 1) = -8)
  (k : ℝ)
  (x_A y_A x_B y_B : ℝ)
  (h_k : k ≠ 0)
  (h_intersect2 : line2 k x_A y_A ∧ line2 k x_B y_B)
  (h_parabola2 : parabola p x_A y_A ∧ parabola p x_B y_B)
  (x_A' : ℝ)
  (h_symmetric : x_A' = -x_A) :
  (∀ x y, parabola p x y ↔ x^2 = -6*y) ∧
  (∃ t : ℝ, t = (y_B - y_A) / (x_B - x_A') ∧ 
            0 = t * 0 + y_A - t * x_A' ∧
            3/2 = t * 0 + y_A - t * x_A') :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_line_intersection_l3709_370951


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3709_370902

/-- Given Sam's current age s and Tim's current age t, where:
    1. s - 4 = 4(t - 4)
    2. s - 10 = 5(t - 10)
    Prove that the number of years x until their age ratio is 3:1 is 8. -/
theorem age_ratio_problem (s t : ℕ) 
  (h1 : s - 4 = 4 * (t - 4)) 
  (h2 : s - 10 = 5 * (t - 10)) : 
  ∃ x : ℕ, x = 8 ∧ (s + x) / (t + x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3709_370902


namespace NUMINAMATH_CALUDE_black_to_grey_ratio_in_square_with_circles_l3709_370947

/-- The ratio of black to grey areas in a square with four inscribed circles -/
theorem black_to_grey_ratio_in_square_with_circles (s : ℝ) (h : s > 0) :
  let r := s / 4
  let circle_area := π * r^2
  let total_square_area := s^2
  let remaining_area := total_square_area - 4 * circle_area
  let black_area := remaining_area / 4
  let grey_area := 3 * black_area
  black_area / grey_area = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_black_to_grey_ratio_in_square_with_circles_l3709_370947


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l3709_370989

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

end NUMINAMATH_CALUDE_scientific_notation_correct_l3709_370989


namespace NUMINAMATH_CALUDE_line_equation_through_points_l3709_370961

/-- Given two points P(3,2) and Q(4,7), prove that the equation 5x - y - 13 = 0
    represents the line passing through these points. -/
theorem line_equation_through_points (x y : ℝ) :
  let P : ℝ × ℝ := (3, 2)
  let Q : ℝ × ℝ := (4, 7)
  (5 * x - y - 13 = 0) ↔ 
    (∃ t : ℝ, (x, y) = ((1 - t) • P.1 + t • Q.1, (1 - t) • P.2 + t • Q.2)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l3709_370961


namespace NUMINAMATH_CALUDE_sequence_relation_l3709_370990

def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * x (n + 1) - x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * y (n + 1) - y n

theorem sequence_relation (n : ℕ) : (y n)^2 = 3 * (x n)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_relation_l3709_370990


namespace NUMINAMATH_CALUDE_decimal_addition_l3709_370987

theorem decimal_addition : 1 + 0.01 + 0.0001 = 1.0101 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l3709_370987


namespace NUMINAMATH_CALUDE_other_root_is_one_l3709_370997

/-- Given a quadratic function f(x) = x^2 + 2x - a with a root of -3, 
    prove that the other root is 1. -/
theorem other_root_is_one (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 + 2*x - a) 
    (h2 : f (-3) = 0) : 
  ∃ x, x ≠ -3 ∧ f x = 0 ∧ x = 1 := by
sorry

end NUMINAMATH_CALUDE_other_root_is_one_l3709_370997


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l3709_370920

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 1|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x < 6} = Set.Ioo (-3) 3 := by sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ (m n : ℝ), m > 0 → n > 0 → m + n = 1 → 
    ∃ x₀ : ℝ, 1/m + 1/n ≥ f a x₀} = Set.Icc (-5) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l3709_370920


namespace NUMINAMATH_CALUDE_right_triangle_sets_set_a_not_right_triangle_l3709_370926

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that set A cannot form a right triangle while others can -/
theorem right_triangle_sets :
  ¬(is_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5)) ∧
  (is_right_triangle 1 (Real.sqrt 2) (Real.sqrt 3)) ∧
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle 3 4 5) := by
  sorry

/-- The specific theorem for set A -/
theorem set_a_not_right_triangle :
  ¬(is_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sets_set_a_not_right_triangle_l3709_370926


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3709_370908

/-- Two arithmetic sequences and their sum sequences -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_ratio : 
  ∀ (a b : ℕ → ℚ) (S T : ℕ → ℚ),
  arithmetic_sequence a →
  arithmetic_sequence b →
  (∀ n : ℕ, S n = (n : ℚ) * a n - (n - 1 : ℚ) / 2 * (a n - a 1)) →
  (∀ n : ℕ, T n = (n : ℚ) * b n - (n - 1 : ℚ) / 2 * (b n - b 1)) →
  (∀ n : ℕ, n > 0 → S n / T n = (5 * n - 3 : ℚ) / (2 * n + 1)) →
  a 20 / b 7 = 64 / 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3709_370908


namespace NUMINAMATH_CALUDE_circle_parameter_range_l3709_370935

/-- Represents the equation of a potential circle -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 4*y + 5*a = 0

/-- Determines if the equation represents a valid circle -/
def is_valid_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The main theorem stating the range of 'a' for which the equation represents a circle -/
theorem circle_parameter_range :
  ∀ a : ℝ, is_valid_circle a ↔ (a > 4 ∨ a < 1) :=
sorry

end NUMINAMATH_CALUDE_circle_parameter_range_l3709_370935


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l3709_370922

theorem circle_tangent_to_line (m : ℝ) (hm : m ≥ 0) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = m}
  let line := {(x, y) : ℝ × ℝ | x + y = Real.sqrt (2 * m)}
  ∃ (p : ℝ × ℝ), p ∈ circle ∧ p ∈ line ∧
    ∀ (q : ℝ × ℝ), q ∈ circle → q ∈ line → q = p :=
by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l3709_370922


namespace NUMINAMATH_CALUDE_factory_bulb_reliability_l3709_370907

theorem factory_bulb_reliability 
  (factory_x_reliability : ℝ) 
  (factory_x_supply : ℝ) 
  (total_reliability : ℝ) 
  (h1 : factory_x_reliability = 0.59) 
  (h2 : factory_x_supply = 0.60) 
  (h3 : total_reliability = 0.62) :
  let factory_y_supply := 1 - factory_x_supply
  let factory_y_reliability := (total_reliability - factory_x_supply * factory_x_reliability) / factory_y_supply
  factory_y_reliability = 0.665 := by
sorry

end NUMINAMATH_CALUDE_factory_bulb_reliability_l3709_370907


namespace NUMINAMATH_CALUDE_f_properties_l3709_370985

noncomputable def f (a b x : ℝ) : ℝ := Real.log x - a * x + b

theorem f_properties (a b : ℝ) (ha : a > 0) 
  (hf : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a b y ≤ f a b x) ∧
  (∀ (x₁ x₂ : ℝ), f a b x₁ = 0 → f a b x₂ = 0 → x₁ * x₂ < 1 / (a^2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3709_370985


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3709_370950

/-- Given that the solution set of ax^2 + bx + 2 > 0 is {x | -1/2 < x < 1/3}, prove that a - b = -10 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a - b = -10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3709_370950


namespace NUMINAMATH_CALUDE_solution_quadratic_equation_l3709_370952

theorem solution_quadratic_equation : 
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ x = 2 ∨ x = -3 := by sorry

end NUMINAMATH_CALUDE_solution_quadratic_equation_l3709_370952


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l3709_370994

theorem expression_equals_negative_one (a y : ℝ) 
  (h1 : a ≠ 0) (h2 : a ≠ 2*y) (h3 : a ≠ -2*y) :
  (a / (a + 2*y) + y / (a - 2*y)) / (y / (a + 2*y) - a / (a - 2*y)) = -1 ↔ y = -a/3 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l3709_370994


namespace NUMINAMATH_CALUDE_power_multiplication_l3709_370991

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3709_370991


namespace NUMINAMATH_CALUDE_complex_fourth_quadrant_range_l3709_370939

theorem complex_fourth_quadrant_range (a : ℝ) : 
  let z₁ : ℂ := 3 - a * Complex.I
  let z₂ : ℂ := 1 + 2 * Complex.I
  (0 < (z₁ / z₂).re ∧ (z₁ / z₂).im < 0) → (-6 < a ∧ a < 3/2) :=
by sorry

end NUMINAMATH_CALUDE_complex_fourth_quadrant_range_l3709_370939


namespace NUMINAMATH_CALUDE_sequence_properties_l3709_370973

/-- Sequence type representing our 0-1 sequence --/
def Sequence := ℕ → Bool

/-- Generate the nth term of the sequence --/
def generateTerm (n : ℕ) : Sequence := sorry

/-- Check if a sequence is periodic --/
def isPeriodic (s : Sequence) : Prop := sorry

/-- Get the nth digit of the sequence --/
def nthDigit (s : Sequence) (n : ℕ) : Bool := sorry

/-- Get the position of the nth occurrence of a digit --/
def nthOccurrence (s : Sequence) (digit : Bool) (n : ℕ) : ℕ := sorry

theorem sequence_properties (s : Sequence) :
  (s = generateTerm 0) →
  (¬ isPeriodic s) ∧
  (nthDigit s 1000 = true) ∧
  (nthOccurrence s true 10000 = 21328) ∧
  (∀ n : ℕ, nthOccurrence s true n = ⌊(2 + Real.sqrt 2) * n⌋) ∧
  (∀ n : ℕ, nthOccurrence s false n = ⌊Real.sqrt 2 * n⌋) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3709_370973


namespace NUMINAMATH_CALUDE_certain_number_proof_l3709_370983

theorem certain_number_proof :
  ∃! x : ℝ, 0.65 * x = (4 / 5 : ℝ) * 25 + 6 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3709_370983


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l3709_370900

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a line is tangent to a circle -/
def Line.tangentTo (l : Line) (c : Circle) : Prop :=
  (l.a * c.center.1 + l.b * c.center.2 + l.c)^2 = 
    (l.a^2 + l.b^2) * c.radius^2

theorem circle_and_tangent_line 
  (c : Circle) 
  (l : Line) : 
  c.contains (0, 0) → 
  c.contains (4, 0) → 
  c.contains (0, 2) → 
  l.a = 2 → 
  l.b = -1 → 
  l.c = 2 → 
  l.tangentTo c → 
  c = { center := (2, 1), radius := Real.sqrt 5 } ∧ 
  l = { a := 2, b := -1, c := 2 } := by
  sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l3709_370900


namespace NUMINAMATH_CALUDE_paint_mixture_problem_l3709_370965

/-- Given a paint mixture with ratio 7:2:1:1 for blue, red, white, and green,
    prove that if 140 oz of blue paint is used and the total mixture should not exceed 220 oz,
    then 20 oz of white paint is required. -/
theorem paint_mixture_problem (blue red white green : ℕ) 
  (ratio : blue = 7 ∧ red = 2 ∧ white = 1 ∧ green = 1) 
  (blue_amount : ℕ) (total_limit : ℕ)
  (h_blue_amount : blue_amount = 140)
  (h_total_limit : total_limit = 220) :
  let total_parts := blue + red + white + green
  let ounces_per_part := blue_amount / blue
  let white_amount := ounces_per_part * white
  white_amount = 20 ∧ white_amount ≤ total_limit - blue_amount :=
by sorry

end NUMINAMATH_CALUDE_paint_mixture_problem_l3709_370965


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3709_370941

/-- A geometric sequence with common ratio 2 and fourth term 16 has first term equal to 2 -/
theorem geometric_sequence_first_term (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  a 4 = 16 →                    -- fourth term is 16
  a 1 = 2 :=                    -- prove that first term is 2
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3709_370941


namespace NUMINAMATH_CALUDE_mountain_climb_fraction_l3709_370903

theorem mountain_climb_fraction (mountain_height : ℕ) (num_trips : ℕ) (total_distance : ℕ)
  (h1 : mountain_height = 40000)
  (h2 : num_trips = 10)
  (h3 : total_distance = 600000) :
  (total_distance / (2 * num_trips)) / mountain_height = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_mountain_climb_fraction_l3709_370903


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3709_370916

theorem complex_equation_sum (a b : ℝ) : 
  (a : ℂ) + b * Complex.I = (1 - Complex.I) * (2 + Complex.I) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3709_370916


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l3709_370929

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis :
  let A : Point := { x := 1, y := -2 }
  reflectAcrossXAxis A = { x := 1, y := 2 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l3709_370929


namespace NUMINAMATH_CALUDE_man_downstream_speed_l3709_370955

/-- Calculates the downstream speed of a man given his upstream and still water speeds -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem: Given a man's upstream speed of 26 km/h and still water speed of 28 km/h, 
    his downstream speed is 30 km/h -/
theorem man_downstream_speed :
  downstream_speed 26 28 = 30 := by
  sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l3709_370955


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l3709_370996

theorem smallest_m_for_integral_solutions : 
  (∃ m : ℕ, m > 0 ∧ 
   (∃ x y : ℤ, 10 * x^2 - m * x + 1980 = 0 ∧ 10 * y^2 - m * y + 1980 = 0 ∧ x ≠ y) ∧
   (∀ k : ℕ, k > 0 ∧ k < m → 
     ¬∃ x y : ℤ, 10 * x^2 - k * x + 1980 = 0 ∧ 10 * y^2 - k * y + 1980 = 0 ∧ x ≠ y)) ∧
  (∀ m : ℕ, m > 0 ∧ 
   (∃ x y : ℤ, 10 * x^2 - m * x + 1980 = 0 ∧ 10 * y^2 - m * y + 1980 = 0 ∧ x ≠ y) ∧
   (∀ k : ℕ, k > 0 ∧ k < m → 
     ¬∃ x y : ℤ, 10 * x^2 - k * x + 1980 = 0 ∧ 10 * y^2 - k * y + 1980 = 0 ∧ x ≠ y) →
   m = 290) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l3709_370996


namespace NUMINAMATH_CALUDE_some_zims_not_cims_l3709_370901

-- Define the universe
variable (U : Type)

-- Define predicates for Zim, Bim, and Cim
variable (Zim Bim Cim : U → Prop)

-- Hypothesis I: All Zims are Bims
axiom h1 : ∀ x, Zim x → Bim x

-- Hypothesis II: Some Bims are not Cims
axiom h2 : ∃ x, Bim x ∧ ¬Cim x

-- Theorem to prove
theorem some_zims_not_cims : ∃ x, Zim x ∧ ¬Cim x := by
  sorry


end NUMINAMATH_CALUDE_some_zims_not_cims_l3709_370901


namespace NUMINAMATH_CALUDE_helen_cookies_yesterday_l3709_370998

def cookies_this_morning : ℕ := 270
def cookies_day_before_yesterday : ℕ := 419
def cookies_till_last_night : ℕ := 450

theorem helen_cookies_yesterday :
  cookies_day_before_yesterday + cookies_this_morning - cookies_till_last_night = 239 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_yesterday_l3709_370998


namespace NUMINAMATH_CALUDE_nine_digit_multiplier_problem_l3709_370988

theorem nine_digit_multiplier_problem : 
  ∃! (N : ℕ), 
    (100000000 ≤ N ∧ N ≤ 999999999) ∧ 
    (N * 123456789) % 1000000000 = 987654321 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_multiplier_problem_l3709_370988


namespace NUMINAMATH_CALUDE_toy_car_energy_comparison_l3709_370948

theorem toy_car_energy_comparison (m : ℝ) (h : m > 0) :
  let KE (v : ℝ) := (1/2) * m * v^2
  (KE 4 - KE 2) = 3 * (KE 2 - KE 0) :=
by
  sorry

end NUMINAMATH_CALUDE_toy_car_energy_comparison_l3709_370948


namespace NUMINAMATH_CALUDE_third_month_sale_l3709_370943

/-- Calculates the unknown sale in the third month given the sales of other months and the average --/
theorem third_month_sale
  (sale1 sale2 sale4 sale5 sale6 : ℕ)
  (avg : ℕ)
  (h1 : sale1 = 5124)
  (h2 : sale2 = 5366)
  (h4 : sale4 = 6124)
  (h6 : sale6 = 4579)
  (havg : avg = 5400)
  (h_avg : (sale1 + sale2 + sale4 + sale5 + sale6 + sale3) / 6 = avg)
  : sale3 = 11207 :=
by
  sorry

#check third_month_sale

end NUMINAMATH_CALUDE_third_month_sale_l3709_370943


namespace NUMINAMATH_CALUDE_inequality_relationship_l3709_370914

theorem inequality_relationship (a b : ℝ) (ha : a > 0) (hb : -1 < b ∧ b < 0) :
  a * b < a * b^2 ∧ a * b^2 < a := by
  sorry

end NUMINAMATH_CALUDE_inequality_relationship_l3709_370914


namespace NUMINAMATH_CALUDE_leftover_coins_value_l3709_370938

/-- The number of nickels in a complete roll -/
def nickels_per_roll : ℕ := 40

/-- The number of pennies in a complete roll -/
def pennies_per_roll : ℕ := 50

/-- The number of nickels Sarah has -/
def sarah_nickels : ℕ := 132

/-- The number of pennies Sarah has -/
def sarah_pennies : ℕ := 245

/-- The number of nickels Tom has -/
def tom_nickels : ℕ := 98

/-- The number of pennies Tom has -/
def tom_pennies : ℕ := 203

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The total value of leftover coins after combining and rolling -/
def leftover_value : ℚ :=
  (((sarah_nickels + tom_nickels) % nickels_per_roll : ℚ) * nickel_value) +
  (((sarah_pennies + tom_pennies) % pennies_per_roll : ℚ) * penny_value)

theorem leftover_coins_value :
  leftover_value = 1.98 := by sorry

end NUMINAMATH_CALUDE_leftover_coins_value_l3709_370938


namespace NUMINAMATH_CALUDE_set_equality_l3709_370971

def positive_integers : Set ℕ := {n : ℕ | n > 0}

def set_a : Set ℕ := {x ∈ positive_integers | x - 3 < 2}
def set_b : Set ℕ := {1, 2, 3, 4}

theorem set_equality : set_a = set_b := by sorry

end NUMINAMATH_CALUDE_set_equality_l3709_370971


namespace NUMINAMATH_CALUDE_square_difference_identity_l3709_370960

theorem square_difference_identity (x : ℝ) (c : ℝ) (hc : c > 0) :
  (x^2 + c)^2 - (x^2 - c)^2 = 4*x^2*c := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l3709_370960


namespace NUMINAMATH_CALUDE_probability_less_than_three_l3709_370942

/-- A bag containing 5 balls labeled 1 to 5 -/
def Bag : Finset ℕ := {1, 2, 3, 4, 5}

/-- The set of balls with numbers less than 3 -/
def LessThanThree : Finset ℕ := {1, 2}

/-- The probability of drawing a ball with a number less than 3 -/
def probability : ℚ := (LessThanThree.card : ℚ) / (Bag.card : ℚ)

theorem probability_less_than_three : probability = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_three_l3709_370942


namespace NUMINAMATH_CALUDE_opposite_of_abs_neg_pi_l3709_370906

theorem opposite_of_abs_neg_pi : -(|-π|) = -π := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_abs_neg_pi_l3709_370906


namespace NUMINAMATH_CALUDE_farm_animals_l3709_370949

theorem farm_animals (total_legs : ℕ) (total_animals : ℕ) (sheep : ℕ) :
  total_legs = 60 →
  total_animals = 20 →
  total_legs = 2 * (total_animals - sheep) + 4 * sheep →
  sheep = 10 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l3709_370949


namespace NUMINAMATH_CALUDE_triangle_altitude_l3709_370953

theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) :
  area = 960 →
  base = 48 →
  area = (1 / 2) * base * altitude →
  altitude = 40 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_l3709_370953


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3709_370957

theorem intersection_of_sets : 
  let M : Set ℕ := {0, 1, 2, 3}
  let N : Set ℕ := {1, 3, 4}
  M ∩ N = {1, 3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3709_370957


namespace NUMINAMATH_CALUDE_min_value_of_f_l3709_370979

-- Define the function f(x) = x^2 + 6x + 13
def f (x : ℝ) : ℝ := x^2 + 6*x + 13

-- Theorem: The minimum value of f(x) is 4 for all real x
theorem min_value_of_f : ∀ x : ℝ, f x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3709_370979


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_12_l3709_370928

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_12 :
  ∀ n : ℕ, is_four_digit n → (sum_of_first_n n % 12 = 0) → n ≥ 1001 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_12_l3709_370928


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l3709_370982

/-- The y-coordinate of a point on the y-axis equidistant from (5, 0) and (3, 6) is 5/3 -/
theorem equidistant_point_y_coordinate :
  let A : ℝ × ℝ := (5, 0)
  let B : ℝ × ℝ := (3, 6)
  let P : ℝ → ℝ × ℝ := fun y ↦ (0, y)
  ∃ y : ℝ, (dist (P y) A)^2 = (dist (P y) B)^2 ∧ y = 5/3 :=
by sorry


end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l3709_370982


namespace NUMINAMATH_CALUDE_candy_bar_profit_l3709_370932

/-- Calculates the profit from selling candy bars --/
def calculate_profit (
  total_bars : ℕ
  ) (buy_rate : ℚ × ℚ)
    (sell_rate : ℚ × ℚ)
    (discount_rate : ℕ × ℚ) : ℚ :=
  let cost_per_bar := buy_rate.2 / buy_rate.1
  let sell_per_bar := sell_rate.2 / sell_rate.1
  let total_cost := cost_per_bar * total_bars
  let total_revenue := sell_per_bar * total_bars
  let total_discounts := (total_bars / discount_rate.1) * discount_rate.2
  total_revenue - total_discounts - total_cost

theorem candy_bar_profit :
  calculate_profit 1200 (3, 1.5) (4, 3) (100, 2) = 276 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l3709_370932


namespace NUMINAMATH_CALUDE_xy_inequality_l3709_370911

theorem xy_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = x + y + 3) :
  (x + y ≥ 6) ∧ (x * y ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_l3709_370911


namespace NUMINAMATH_CALUDE_x_less_than_y_l3709_370992

theorem x_less_than_y (s x y : ℝ) (hs : s > 0) (hxy : x * y ≠ 0) (hsxy : s * x < s * y) : x < y := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_y_l3709_370992


namespace NUMINAMATH_CALUDE_mixture_capacity_l3709_370963

/-- Represents the capacity and alcohol percentage of a vessel -/
structure Vessel where
  capacity : ℝ
  alcoholPercentage : ℝ

/-- Represents the mixture of two vessels -/
def Mixture (v1 v2 : Vessel) : ℝ × ℝ :=
  (v1.capacity + v2.capacity, v1.capacity * v1.alcoholPercentage + v2.capacity * v2.alcoholPercentage)

theorem mixture_capacity (v1 v2 : Vessel) (newConcentration : ℝ) :
  v1.capacity = 3 →
  v1.alcoholPercentage = 0.25 →
  v2.capacity = 5 →
  v2.alcoholPercentage = 0.40 →
  (Mixture v1 v2).1 = 8 →
  newConcentration = 0.275 →
  (Mixture v1 v2).2 / newConcentration = 10 := by
  sorry

#check mixture_capacity

end NUMINAMATH_CALUDE_mixture_capacity_l3709_370963


namespace NUMINAMATH_CALUDE_abs_x_minus_two_plus_three_min_l3709_370934

theorem abs_x_minus_two_plus_three_min (x : ℝ) : 
  ∃ (min : ℝ), (∀ x, |x - 2| + 3 ≥ min) ∧ (∃ x, |x - 2| + 3 = min) := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_two_plus_three_min_l3709_370934
