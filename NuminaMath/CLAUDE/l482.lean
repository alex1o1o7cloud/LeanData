import Mathlib

namespace NUMINAMATH_CALUDE_trapezoid_larger_base_l482_48235

/-- Given a trapezoid with base ratio 1:3 and midline length 24, 
    prove the larger base is 36 -/
theorem trapezoid_larger_base 
  (shorter_base longer_base midline : ℝ) 
  (h_ratio : longer_base = 3 * shorter_base) 
  (h_midline : midline = (shorter_base + longer_base) / 2) 
  (h_midline_length : midline = 24) : 
  longer_base = 36 := by
sorry


end NUMINAMATH_CALUDE_trapezoid_larger_base_l482_48235


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_solution_set_theorem_l482_48295

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem for part (1)
theorem intersection_of_A_and_B : A ∩ B = A_intersect_B := by sorry

-- Define the solution set of x^2 + ax - b < 0
def solution_set (a b : ℝ) : Set ℝ := {x | x < -1 ∨ x > 2}

-- Theorem for part (2)
theorem solution_set_theorem (a b : ℝ) :
  ({x : ℝ | x^2 + a*x + b < 0} = A_intersect_B) →
  ({x : ℝ | x^2 + a*x - b < 0} = solution_set a b) := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_solution_set_theorem_l482_48295


namespace NUMINAMATH_CALUDE_wrench_force_calculation_l482_48299

/-- The force required to loosen a nut with a wrench -/
def force_to_loosen (handle_length : ℝ) (force : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * handle_length = k

theorem wrench_force_calculation 
  (h₁ : force_to_loosen 12 480) 
  (h₂ : force_to_loosen 18 f) : 
  f = 320 := by
  sorry

end NUMINAMATH_CALUDE_wrench_force_calculation_l482_48299


namespace NUMINAMATH_CALUDE_max_portfolios_is_six_l482_48253

/-- Represents the number of items Stacy purchases -/
structure Purchase where
  pens : ℕ
  pads : ℕ
  portfolios : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  2 * p.pens + 5 * p.pads + 15 * p.portfolios

/-- Checks if a purchase is valid according to the problem constraints -/
def isValidPurchase (p : Purchase) : Prop :=
  p.pens ≥ 1 ∧ p.pads ≥ 1 ∧ p.portfolios ≥ 1 ∧ totalCost p = 100

/-- The maximum number of portfolios that can be purchased -/
def maxPortfolios : ℕ := 6

/-- Theorem stating that 6 is the maximum number of portfolios that can be purchased -/
theorem max_portfolios_is_six :
  (∀ p : Purchase, isValidPurchase p → p.portfolios ≤ maxPortfolios) ∧
  (∃ p : Purchase, isValidPurchase p ∧ p.portfolios = maxPortfolios) := by
  sorry


end NUMINAMATH_CALUDE_max_portfolios_is_six_l482_48253


namespace NUMINAMATH_CALUDE_man_twice_son_age_l482_48236

/-- Represents the number of years until a man's age is twice his son's age. -/
def yearsUntilTwiceAge (sonAge : ℕ) (ageDifference : ℕ) : ℕ :=
  2

theorem man_twice_son_age (sonAge : ℕ) (ageDifference : ℕ) 
  (h1 : sonAge = 25) 
  (h2 : ageDifference = 27) : 
  yearsUntilTwiceAge sonAge ageDifference = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_twice_son_age_l482_48236


namespace NUMINAMATH_CALUDE_nails_per_plank_l482_48219

theorem nails_per_plank (large_planks : ℕ) (additional_nails : ℕ) (total_nails : ℕ) :
  large_planks = 13 →
  additional_nails = 8 →
  total_nails = 229 →
  ∃ (nails_per_plank : ℕ), nails_per_plank * large_planks + additional_nails = total_nails ∧ nails_per_plank = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_nails_per_plank_l482_48219


namespace NUMINAMATH_CALUDE_imaginary_sum_equals_negative_i_l482_48255

theorem imaginary_sum_equals_negative_i (i : ℂ) (hi : i^2 = -1) :
  i^11 + i^16 + i^21 + i^26 + i^31 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_equals_negative_i_l482_48255


namespace NUMINAMATH_CALUDE_crayons_per_pack_l482_48286

/-- Given that Nancy bought a total of 615 crayons in 41 packs,
    prove that there were 15 crayons in each pack. -/
theorem crayons_per_pack :
  ∀ (total_crayons : ℕ) (num_packs : ℕ),
    total_crayons = 615 →
    num_packs = 41 →
    total_crayons / num_packs = 15 :=
by sorry

end NUMINAMATH_CALUDE_crayons_per_pack_l482_48286


namespace NUMINAMATH_CALUDE_small_pump_fills_in_three_hours_l482_48260

-- Define the filling rates for the pumps
def large_pump_rate : ℝ := 4 -- 1 / (1/4)
def combined_time : ℝ := 0.23076923076923078

-- Define the time it takes for the small pump to fill the tank
def small_pump_time : ℝ := 3

-- Theorem statement
theorem small_pump_fills_in_three_hours :
  let combined_rate := 1 / combined_time
  let small_pump_rate := combined_rate - large_pump_rate
  1 / small_pump_rate = small_pump_time := by sorry

end NUMINAMATH_CALUDE_small_pump_fills_in_three_hours_l482_48260


namespace NUMINAMATH_CALUDE_symmetric_lines_l482_48205

/-- Given two lines in the xy-plane, this function returns true if they are symmetric about the line x = a -/
def are_symmetric_lines (line1 line2 : ℝ → ℝ → Prop) (a : ℝ) : Prop :=
  ∀ x y, line1 x y ↔ line2 (2*a - x) y

/-- The equation of the first line: 2x + y - 1 = 0 -/
def line1 (x y : ℝ) : Prop := 2*x + y - 1 = 0

/-- The equation of the second line: 2x - y - 3 = 0 -/
def line2 (x y : ℝ) : Prop := 2*x - y - 3 = 0

/-- The line of symmetry: x = 1 -/
def symmetry_line : ℝ := 1

theorem symmetric_lines : are_symmetric_lines line1 line2 symmetry_line := by
  sorry

end NUMINAMATH_CALUDE_symmetric_lines_l482_48205


namespace NUMINAMATH_CALUDE_felix_distance_covered_l482_48268

/-- The initial speed in miles per hour -/
def initial_speed : ℝ := 66

/-- The number of hours Felix wants to drive -/
def drive_hours : ℝ := 4

/-- The factor by which Felix wants to increase his speed -/
def speed_increase_factor : ℝ := 2

/-- Calculates the distance covered given a speed and time -/
def distance_covered (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating the distance Felix will cover -/
theorem felix_distance_covered : 
  distance_covered (initial_speed * speed_increase_factor) drive_hours = 528 := by
  sorry

end NUMINAMATH_CALUDE_felix_distance_covered_l482_48268


namespace NUMINAMATH_CALUDE_solution_set_abs_equation_l482_48231

theorem solution_set_abs_equation (x : ℝ) :
  |x - 2| + |2*x - 3| = |3*x - 5| ↔ x ≤ 3/2 ∨ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_abs_equation_l482_48231


namespace NUMINAMATH_CALUDE_cubic_arithmetic_progression_l482_48239

/-- 
A cubic equation x^3 + ax^2 + bx + c = 0 has three real roots forming an arithmetic progression 
if and only if the following conditions are satisfied:
1) ab/3 - 2a^3/27 - c = 0
2) a^3/3 - b ≥ 0
-/
theorem cubic_arithmetic_progression (a b c : ℝ) : 
  (∃ x y z : ℝ, x < y ∧ y < z ∧ 
    (∀ t : ℝ, t^3 + a*t^2 + b*t + c = 0 ↔ t = x ∨ t = y ∨ t = z) ∧
    y - x = z - y) ↔ 
  (a*b/3 - 2*a^3/27 - c = 0 ∧ a^3/3 - b ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_arithmetic_progression_l482_48239


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l482_48256

/-- Given a hyperbola C passing through the point (1,1) with asymptotes 2x+y=0 and 2x-y=0,
    its standard equation is 4x²/3 - y²/3 = 1. -/
theorem hyperbola_standard_equation (C : Set (ℝ × ℝ)) :
  (∀ x y, (x, y) ∈ C ↔ 4 * x^2 / 3 - y^2 / 3 = 1) ↔
  ((1, 1) ∈ C ∧
   (∀ x y, 2*x + y = 0 → (x, y) ∈ frontier C) ∧
   (∀ x y, 2*x - y = 0 → (x, y) ∈ frontier C)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l482_48256


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l482_48230

def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

theorem quadratic_minimum_value :
  ∀ x : ℝ, f x ≥ 1 ∧ ∃ x₀ : ℝ, f x₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l482_48230


namespace NUMINAMATH_CALUDE_stamp_arrangement_count_l482_48249

/-- Represents the number of stamps of each denomination --/
def stamp_counts : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

/-- Represents the value of each stamp denomination --/
def stamp_values : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

/-- A function to calculate the number of unique arrangements --/
def count_arrangements (counts : List Nat) (values : List Nat) (target : Nat) : Nat :=
  sorry

theorem stamp_arrangement_count :
  count_arrangements stamp_counts stamp_values 20 = 76 :=
by sorry

end NUMINAMATH_CALUDE_stamp_arrangement_count_l482_48249


namespace NUMINAMATH_CALUDE_fraction_equality_l482_48226

theorem fraction_equality (m n p q r : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5)
  (h4 : m / r = 10) :
  r / q = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l482_48226


namespace NUMINAMATH_CALUDE_bears_in_shipment_bears_shipment_proof_l482_48213

/-- The number of bears in a toy store shipment -/
theorem bears_in_shipment (initial_stock : ℕ) (shelves : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  shelves * bears_per_shelf - initial_stock

/-- Proof that the number of bears in the shipment is 7 -/
theorem bears_shipment_proof :
  bears_in_shipment 5 2 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_bears_in_shipment_bears_shipment_proof_l482_48213


namespace NUMINAMATH_CALUDE_probability_even_sum_l482_48280

def set_A : Finset ℕ := {3, 4, 5, 8}
def set_B : Finset ℕ := {6, 7, 9}

def is_sum_even (a b : ℕ) : Bool :=
  (a + b) % 2 = 0

def count_even_sums : ℕ :=
  (set_A.card * set_B.card).div 2

theorem probability_even_sum :
  (count_even_sums : ℚ) / (set_A.card * set_B.card) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_l482_48280


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l482_48262

theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_edge : ℝ) :
  l = 10 ∧ w = 5 ∧ h = 24 →
  cube_edge^3 = l * w * h →
  6 * cube_edge^2 = 6 * (1200^(2/3)) := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l482_48262


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l482_48283

theorem rectangular_plot_breadth (length breadth area : ℝ) : 
  length = 3 * breadth →
  area = length * breadth →
  area = 972 →
  breadth = 18 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l482_48283


namespace NUMINAMATH_CALUDE_max_visible_sum_l482_48267

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- The set of numbers on each cube --/
def cube_numbers : Finset ℕ := {1, 3, 6, 12, 24, 48}

/-- A stack of three cubes --/
structure CubeStack :=
  (bottom : Cube)
  (middle : Cube)
  (top : Cube)

/-- The sum of visible numbers in a cube stack --/
def visible_sum (stack : CubeStack) : ℕ := sorry

/-- Theorem stating the maximum sum of visible numbers --/
theorem max_visible_sum :
  ∃ (stack : CubeStack),
    (∀ (c : Cube) (i : Fin 6), c.faces i ∈ cube_numbers) →
    (∀ (stack' : CubeStack), visible_sum stack' ≤ visible_sum stack) →
    visible_sum stack = 267 :=
sorry

end NUMINAMATH_CALUDE_max_visible_sum_l482_48267


namespace NUMINAMATH_CALUDE_cows_bought_is_two_l482_48287

/-- The number of cows bought given the total cost, number of goats, and average prices. -/
def number_of_cows (total_cost goats goat_price cow_price : ℕ) : ℕ :=
  (total_cost - goats * goat_price) / cow_price

/-- Theorem stating that the number of cows bought is 2 under the given conditions. -/
theorem cows_bought_is_two :
  number_of_cows 1500 10 70 400 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cows_bought_is_two_l482_48287


namespace NUMINAMATH_CALUDE_range_of_m_l482_48243

theorem range_of_m (x m : ℝ) : 
  (∀ x, (2*m - 3 ≤ x ∧ x ≤ 2*m + 1) → x ≤ -5) → 
  m ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l482_48243


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l482_48291

/-- The displacement function of an object with respect to time -/
def displacement (t : ℝ) : ℝ := 4 - 2*t + t^2

/-- The velocity function of an object with respect to time -/
def velocity (t : ℝ) : ℝ := 2*t - 2

theorem instantaneous_velocity_at_4_seconds :
  velocity 4 = 6 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_seconds_l482_48291


namespace NUMINAMATH_CALUDE_derivative_of_f_l482_48258

/-- The function f(x) = 3x^2 -/
def f (x : ℝ) : ℝ := 3 * x^2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x

theorem derivative_of_f (x : ℝ) : deriv f x = f' x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l482_48258


namespace NUMINAMATH_CALUDE_equilateral_triangle_coverage_l482_48263

/-- An equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The union of a set of equilateral triangles -/
def UnionOfTriangles (triangles : Set EquilateralTriangle) : Set (ℝ × ℝ) := sorry

/-- A triangle is contained in a set of points -/
def TriangleContainedIn (t : EquilateralTriangle) (s : Set (ℝ × ℝ)) : Prop := sorry

theorem equilateral_triangle_coverage 
  (Δ : EquilateralTriangle) 
  (a b : ℝ)
  (h_a : Δ.sideLength = a)
  (h_b : b > 0)
  (h_five : ∃ (five_triangles : Finset EquilateralTriangle), 
    five_triangles.card = 5 ∧ 
    (∀ t ∈ five_triangles, t.sideLength = b) ∧
    TriangleContainedIn Δ (UnionOfTriangles five_triangles.toSet)) :
  ∃ (four_triangles : Finset EquilateralTriangle),
    four_triangles.card = 4 ∧
    (∀ t ∈ four_triangles, t.sideLength = b) ∧
    TriangleContainedIn Δ (UnionOfTriangles four_triangles.toSet) := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_coverage_l482_48263


namespace NUMINAMATH_CALUDE_smaller_side_of_rearranged_rectangle_l482_48233

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the result of dividing and rearranging a rectangle -/
structure RearrangedRectangle where
  original : Rectangle
  new : Rectangle
  is_valid : original.width * original.height = new.width * new.height

/-- The theorem to be proved -/
theorem smaller_side_of_rearranged_rectangle 
  (r : RearrangedRectangle) 
  (h1 : r.original.width = 10) 
  (h2 : r.original.height = 25) :
  min r.new.width r.new.height = 10 := by
  sorry

#check smaller_side_of_rearranged_rectangle

end NUMINAMATH_CALUDE_smaller_side_of_rearranged_rectangle_l482_48233


namespace NUMINAMATH_CALUDE_determinant_solution_l482_48208

theorem determinant_solution (a : ℝ) (h : a ≠ 0) :
  ∃ x : ℝ, Matrix.det 
    ![![x + a, x, x],
      ![x, x + a, x],
      ![x, x, x + a]] = 0 ↔ x = -a / 3 := by
  sorry

end NUMINAMATH_CALUDE_determinant_solution_l482_48208


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l482_48257

theorem imaginary_part_of_z (z : ℂ) (h : (1 : ℂ) / z = 1 / (1 + 2*I) + 1 / (1 - I)) : 
  z.im = -(1 / 5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l482_48257


namespace NUMINAMATH_CALUDE_product_of_x_values_l482_48297

theorem product_of_x_values (x₁ x₂ : ℝ) : 
  (|20 / x₁ + 4| = 3 ∧ |20 / x₂ + 4| = 3 ∧ x₁ ≠ x₂) → x₁ * x₂ = 400 / 7 :=
by sorry

end NUMINAMATH_CALUDE_product_of_x_values_l482_48297


namespace NUMINAMATH_CALUDE_store_comparison_l482_48272

/-- Represents the cost difference between Store B and Store A -/
def cost_difference (x : ℝ) : ℝ := 520 - 2.5 * x

theorem store_comparison (x : ℝ) (h : x > 40) :
  cost_difference x = 520 - 2.5 * x ∧
  cost_difference 80 > 0 :=
sorry

#check store_comparison

end NUMINAMATH_CALUDE_store_comparison_l482_48272


namespace NUMINAMATH_CALUDE_exists_m_for_all_n_l482_48234

theorem exists_m_for_all_n (n : ℕ+) : ∃ m : ℤ, (2^(2^n.val) - 1) ∣ (m^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_for_all_n_l482_48234


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l482_48296

theorem arithmetic_calculation : 4 * 10 + 5 * 11 + 12 * 4 + 4 * 9 = 179 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l482_48296


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l482_48252

theorem least_positive_integer_congruence :
  ∃! x : ℕ+, x.val + 3649 ≡ 304 [ZMOD 15] ∧
  ∀ y : ℕ+, y.val + 3649 ≡ 304 [ZMOD 15] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l482_48252


namespace NUMINAMATH_CALUDE_sinusoidal_function_properties_l482_48274

theorem sinusoidal_function_properties (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let f := fun x => a * Real.sin (b * x + c)
  (∀ x, f x ≤ 3) ∧ (f (π / 3) = 3) → a = 3 ∧ c = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_function_properties_l482_48274


namespace NUMINAMATH_CALUDE_polygon_sides_l482_48232

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1080 → ∃ n : ℕ, n = 8 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l482_48232


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l482_48271

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) :
  train_length = 320 →
  train_speed_kmh = 45 →
  time_to_pass = 36.8 →
  ∃ (bridge_length : ℝ), bridge_length = 140 ∧
    bridge_length = (train_speed_kmh * 1000 / 3600 * time_to_pass) - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l482_48271


namespace NUMINAMATH_CALUDE_number_problem_l482_48294

theorem number_problem (x : ℝ) : 0.95 * x - 12 = 178 ↔ x = 200 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l482_48294


namespace NUMINAMATH_CALUDE_badminton_players_l482_48203

/-- A sports club with members playing badminton and tennis -/
structure SportsClub where
  total_members : ℕ
  tennis_players : ℕ
  both_players : ℕ
  neither_players : ℕ

/-- Theorem stating the number of badminton players in the sports club -/
theorem badminton_players (club : SportsClub)
  (h1 : club.total_members = 30)
  (h2 : club.tennis_players = 19)
  (h3 : club.both_players = 9)
  (h4 : club.neither_players = 2) :
  club.total_members - club.tennis_players + club.both_players - club.neither_players = 18 :=
by sorry

end NUMINAMATH_CALUDE_badminton_players_l482_48203


namespace NUMINAMATH_CALUDE_no_real_solutions_l482_48289

theorem no_real_solutions (x : ℝ) :
  x ≠ -1 → (x^2 + x + 1) / (x + 1) ≠ x^2 + 5*x + 6 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l482_48289


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l482_48241

theorem algebraic_expression_value (x y : ℝ) (h : x + 2*y - 1 = 0) :
  (2*x + 4*y) / (x^2 + 4*x*y + 4*y^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l482_48241


namespace NUMINAMATH_CALUDE_inequality_solution_set_inequality_with_conditions_l482_48284

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f x + f (x + 4) ≥ 8} = {x : ℝ | x ≤ -5 ∨ x ≥ 3} := by sorry

-- Theorem for the inequality with conditions
theorem inequality_with_conditions (a b : ℝ) 
  (ha : |a| < 1) (hb : |b| < 1) (ha_neq_zero : a ≠ 0) :
  f (a * b) > |a| * f (b / a) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_inequality_with_conditions_l482_48284


namespace NUMINAMATH_CALUDE_find_b_value_l482_48285

theorem find_b_value (x b : ℝ) (h1 : 5 * x + 3 = b * x - 22) (h2 : x = 5) : b = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l482_48285


namespace NUMINAMATH_CALUDE_jake_kendra_weight_ratio_l482_48298

/-- The problem of Jake and Kendra's weight ratio -/
theorem jake_kendra_weight_ratio :
  ∀ (j k : ℝ),
  j + k = 293 →
  j - 8 = 2 * k →
  (j - 8) / k = 2 :=
by sorry

end NUMINAMATH_CALUDE_jake_kendra_weight_ratio_l482_48298


namespace NUMINAMATH_CALUDE_shopkeeper_ornaments_profit_least_possible_n_l482_48227

theorem shopkeeper_ornaments_profit (n d : ℕ) (h1 : d > 0) : 
  (3 * (d / (3 * n)) + (n - 3) * (d / n + 10) - d = 150) → n ≥ 18 :=
by
  sorry

theorem least_possible_n : 
  ∃ (n d : ℕ), d > 0 ∧ 3 * (d / (3 * n)) + (n - 3) * (d / n + 10) - d = 150 ∧ n = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_ornaments_profit_least_possible_n_l482_48227


namespace NUMINAMATH_CALUDE_emilys_small_gardens_l482_48248

/-- Given Emily's gardening scenario, prove the number of small gardens. -/
theorem emilys_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : 
  total_seeds = 41 →
  big_garden_seeds = 29 →
  seeds_per_small_garden = 4 →
  (total_seeds - big_garden_seeds) / seeds_per_small_garden = 3 := by
  sorry

end NUMINAMATH_CALUDE_emilys_small_gardens_l482_48248


namespace NUMINAMATH_CALUDE_highlighters_count_l482_48221

/-- The number of pink highlighters in the teacher's desk -/
def pink_highlighters : ℕ := 3

/-- The number of yellow highlighters in the teacher's desk -/
def yellow_highlighters : ℕ := 7

/-- The number of blue highlighters in the teacher's desk -/
def blue_highlighters : ℕ := 5

/-- The total number of highlighters in the teacher's desk -/
def total_highlighters : ℕ := pink_highlighters + yellow_highlighters + blue_highlighters

theorem highlighters_count : total_highlighters = 15 := by
  sorry

end NUMINAMATH_CALUDE_highlighters_count_l482_48221


namespace NUMINAMATH_CALUDE_total_snakes_count_l482_48288

/-- Represents the total population in the neighborhood -/
def total_population : ℕ := 200

/-- Represents the percentage of people who own only snakes -/
def only_snakes_percent : ℚ := 5 / 100

/-- Represents the percentage of people who own both cats and snakes, but no other pets -/
def cats_and_snakes_percent : ℚ := 4 / 100

/-- Represents the percentage of people who own both snakes and rabbits, but no other pets -/
def snakes_and_rabbits_percent : ℚ := 5 / 100

/-- Represents the percentage of people who own both snakes and birds, but no other pets -/
def snakes_and_birds_percent : ℚ := 3 / 100

/-- Represents the percentage of exotic pet owners who also own snakes -/
def exotic_and_snakes_percent : ℚ := 25 / 100

/-- Represents the total percentage of exotic pet owners -/
def total_exotic_percent : ℚ := 34 / 100

/-- Calculates the total percentage of snake owners in the neighborhood -/
def total_snake_owners_percent : ℚ :=
  only_snakes_percent + cats_and_snakes_percent + snakes_and_rabbits_percent + 
  snakes_and_birds_percent + (exotic_and_snakes_percent * total_exotic_percent)

/-- Theorem stating that the total number of snakes in the neighborhood is 51 -/
theorem total_snakes_count : ⌊(total_snake_owners_percent * total_population : ℚ)⌋ = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_snakes_count_l482_48288


namespace NUMINAMATH_CALUDE_steve_final_marbles_l482_48261

theorem steve_final_marbles (steve_initial sam_initial sally_initial : ℕ) 
  (h1 : sam_initial = 2 * steve_initial)
  (h2 : sally_initial = sam_initial - 5)
  (h3 : sam_initial - 6 = 8) : 
  steve_initial + 3 = 10 := by
sorry

end NUMINAMATH_CALUDE_steve_final_marbles_l482_48261


namespace NUMINAMATH_CALUDE_intersection_slope_range_l482_48222

/-- Given two points A and B, and a line l that intersects the line segment AB,
    prove that the slope k of line l is within a specific range. -/
theorem intersection_slope_range (A B : ℝ × ℝ) (k : ℝ) : 
  A = (1, 3) →
  B = (-2, -1) →
  (∃ x y : ℝ, x ∈ Set.Icc (min A.1 B.1) (max A.1 B.1) ∧ 
              y ∈ Set.Icc (min A.2 B.2) (max A.2 B.2) ∧
              y = k * (x - 2) + 1) →
  -2 ≤ k ∧ k ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_slope_range_l482_48222


namespace NUMINAMATH_CALUDE_alphabet_size_l482_48214

theorem alphabet_size :
  ∀ (dot_and_line dot_only line_only : ℕ),
    dot_and_line = 16 →
    line_only = 30 →
    dot_only = 4 →
    dot_and_line + dot_only + line_only = 50 := by
  sorry

end NUMINAMATH_CALUDE_alphabet_size_l482_48214


namespace NUMINAMATH_CALUDE_bottle_sales_revenue_l482_48211

/-- Calculate the total revenue from bottle sales -/
theorem bottle_sales_revenue : 
  let small_bottles : ℕ := 6000
  let big_bottles : ℕ := 14000
  let medium_bottles : ℕ := 9000
  let small_price : ℚ := 2
  let big_price : ℚ := 4
  let medium_price : ℚ := 3
  let small_sold_percent : ℚ := 20 / 100
  let big_sold_percent : ℚ := 23 / 100
  let medium_sold_percent : ℚ := 15 / 100
  
  let small_revenue := (small_bottles : ℚ) * small_sold_percent * small_price
  let big_revenue := (big_bottles : ℚ) * big_sold_percent * big_price
  let medium_revenue := (medium_bottles : ℚ) * medium_sold_percent * medium_price
  
  let total_revenue := small_revenue + big_revenue + medium_revenue
  
  total_revenue = 19330 := by sorry

end NUMINAMATH_CALUDE_bottle_sales_revenue_l482_48211


namespace NUMINAMATH_CALUDE_c_finishes_in_60_days_l482_48204

/-- The number of days it takes for worker c to finish the job alone, given:
  * Workers a and b together finish the job in 15 days
  * Workers a, b, and c together finish the job in 12 days
-/
def days_for_c_alone : ℚ :=
  let rate_ab : ℚ := 1 / 15  -- Combined rate of a and b
  let rate_abc : ℚ := 1 / 12 -- Combined rate of a, b, and c
  let rate_c : ℚ := rate_abc - rate_ab -- Rate of c alone
  1 / rate_c -- Days for c to finish the job

/-- Theorem stating that worker c alone can finish the job in 60 days -/
theorem c_finishes_in_60_days : days_for_c_alone = 60 := by
  sorry


end NUMINAMATH_CALUDE_c_finishes_in_60_days_l482_48204


namespace NUMINAMATH_CALUDE_yanni_money_problem_l482_48266

/-- The amount of money Yanni's mother gave him -/
def mothers_gift : ℚ := 0.40

theorem yanni_money_problem :
  let initial_money : ℚ := 0.85
  let found_money : ℚ := 0.50
  let toy_cost : ℚ := 1.60
  let final_balance : ℚ := 0.15
  initial_money + mothers_gift + found_money - toy_cost = final_balance :=
by sorry

end NUMINAMATH_CALUDE_yanni_money_problem_l482_48266


namespace NUMINAMATH_CALUDE_eleven_in_base_two_l482_48282

theorem eleven_in_base_two : 11 = 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 := by
  sorry

#eval toString (1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0)

end NUMINAMATH_CALUDE_eleven_in_base_two_l482_48282


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l482_48228

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^1023 - 1) (2^1034 - 1) = 2^11 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l482_48228


namespace NUMINAMATH_CALUDE_min_value_theorem_l482_48259

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x + 3 * y = 4) :
  (∃ (m : ℝ), m = 3/8 + Real.sqrt 2/4 ∧
    ∀ (z : ℝ), z = 1 / (2 * x + 1) + 1 / (3 * y + 2) → z ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l482_48259


namespace NUMINAMATH_CALUDE_decimal_38_to_binary_l482_48245

-- Define a function to convert decimal to binary
def decimalToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

-- Theorem statement
theorem decimal_38_to_binary :
  decimalToBinary 38 = [false, true, true, false, false, true] := by
  sorry

#eval decimalToBinary 38

end NUMINAMATH_CALUDE_decimal_38_to_binary_l482_48245


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_m_value_l482_48220

theorem intersection_nonempty_implies_m_value (m : ℤ) : 
  let P : Set ℤ := {0, m}
  let Q : Set ℤ := {x | 2 * x^2 - 5 * x < 0}
  (P ∩ Q).Nonempty → m = 1 ∨ m = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_m_value_l482_48220


namespace NUMINAMATH_CALUDE_power_of_product_with_negative_l482_48207

theorem power_of_product_with_negative (m n : ℝ) : (-2 * m^3 * n^2)^2 = 4 * m^6 * n^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_with_negative_l482_48207


namespace NUMINAMATH_CALUDE_average_age_combined_l482_48209

theorem average_age_combined (n_students : ℕ) (n_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  n_students = 33 →
  n_parents = 55 →
  avg_age_students = 11 →
  avg_age_parents = 33 →
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents : ℚ) = 24.75 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l482_48209


namespace NUMINAMATH_CALUDE_line_intersects_circle_twice_tangent_line_m_value_l482_48279

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 25

-- Define the line L
def line_L (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Define the circle D
def circle_D (R x y : ℝ) : Prop := (x + 1)^2 + (y - 5)^2 = R^2

theorem line_intersects_circle_twice (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_L m x₁ y₁ ∧ line_L m x₂ y₂ :=
sorry

theorem tangent_line_m_value :
  ∃ (R : ℝ), R > 0 ∧
    (∀ (R' : ℝ), R' > 0 →
      (∃ (x y : ℝ), circle_D R' x y ∧ line_L (-2/3) x y) →
      R' ≤ R) ∧
    (∃ (x y : ℝ), circle_D R x y ∧ line_L (-2/3) x y) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_twice_tangent_line_m_value_l482_48279


namespace NUMINAMATH_CALUDE_rectangle_least_area_l482_48218

theorem rectangle_least_area :
  ∀ l w : ℕ,
  l = 3 * w →
  2 * (l + w) = 120 →
  ∀ l' w' : ℕ,
  l' = 3 * w' →
  2 * (l' + w') = 120 →
  l * w ≤ l' * w' →
  l * w = 675 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_least_area_l482_48218


namespace NUMINAMATH_CALUDE_expression_divisibility_l482_48278

theorem expression_divisibility (n : ℕ) (x : ℝ) (hx : x ≠ 1) :
  ∃ g : ℝ → ℝ, n * x^(n+1) * (1 - 1/x) - x^n * (1 - 1/x^n) = (x - 1)^2 * g x :=
by sorry

end NUMINAMATH_CALUDE_expression_divisibility_l482_48278


namespace NUMINAMATH_CALUDE_equation_solution_l482_48276

theorem equation_solution (x : ℝ) : 
  x ≠ 3 ∧ x ≠ -3 → (4 / (x^2 - 9) - x / (3 - x) = 1 ↔ x = -13/3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l482_48276


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l482_48281

theorem subset_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {0, -a}
  let B : Set ℝ := {1, a-2, 2*a-2}
  A ⊆ B → a = 1 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l482_48281


namespace NUMINAMATH_CALUDE_square_remainder_l482_48212

theorem square_remainder (n : ℤ) : n % 5 = 3 → n^2 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_l482_48212


namespace NUMINAMATH_CALUDE_evaluate_f_l482_48215

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 9

theorem evaluate_f : 3 * f 5 + 4 * f (-2) = 217 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l482_48215


namespace NUMINAMATH_CALUDE_overlap_length_l482_48244

theorem overlap_length (total_length edge_to_edge_distance : ℝ) 
  (h1 : total_length = 98) 
  (h2 : edge_to_edge_distance = 83) 
  (h3 : ∃ x : ℝ, total_length = edge_to_edge_distance + 6 * x) :
  ∃ x : ℝ, x = 2.5 ∧ total_length = edge_to_edge_distance + 6 * x := by
sorry

end NUMINAMATH_CALUDE_overlap_length_l482_48244


namespace NUMINAMATH_CALUDE_gumball_probability_l482_48273

theorem gumball_probability (blue_prob : ℚ) (pink_prob : ℚ) : 
  blue_prob^2 = 9/49 → pink_prob = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l482_48273


namespace NUMINAMATH_CALUDE_mowing_time_calculation_l482_48217

/-- Represents the dimensions of a rectangular section of the lawn -/
structure LawnSection where
  length : ℝ
  width : ℝ

/-- Represents the mower specifications -/
structure Mower where
  swath_width : ℝ
  overlap : ℝ

/-- Calculates the time required to mow an L-shaped lawn -/
def mowing_time (section1 : LawnSection) (section2 : LawnSection) (mower : Mower) (walking_rate : ℝ) : ℝ :=
  sorry

/-- Theorem stating the time required to mow the lawn -/
theorem mowing_time_calculation :
  let section1 : LawnSection := { length := 120, width := 50 }
  let section2 : LawnSection := { length := 70, width := 50 }
  let mower : Mower := { swath_width := 35 / 12, overlap := 5 / 12 }
  let walking_rate : ℝ := 4000
  mowing_time section1 section2 mower walking_rate = 0.95 :=
by sorry

end NUMINAMATH_CALUDE_mowing_time_calculation_l482_48217


namespace NUMINAMATH_CALUDE_homework_time_reduction_l482_48202

theorem homework_time_reduction (initial_time final_time : ℝ) (x : ℝ) :
  initial_time = 100 →
  final_time = 70 →
  0 < x →
  x < 1 →
  initial_time * (1 - x)^2 = final_time :=
by
  sorry

end NUMINAMATH_CALUDE_homework_time_reduction_l482_48202


namespace NUMINAMATH_CALUDE_lenny_grocery_expense_l482_48240

/-- Proves the amount Lenny spent at the grocery store, given his initial amount, video game expense, and remaining amount. -/
theorem lenny_grocery_expense (initial : ℕ) (video_games : ℕ) (remaining : ℕ) 
  (h1 : initial = 84)
  (h2 : video_games = 24)
  (h3 : remaining = 39) :
  initial - video_games - remaining = 21 := by
  sorry

#check lenny_grocery_expense

end NUMINAMATH_CALUDE_lenny_grocery_expense_l482_48240


namespace NUMINAMATH_CALUDE_a_minus_b_value_l482_48277

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 2) (h2 : b^2 = 9) (h3 : a < b) :
  a - b = -1 ∨ a - b = -5 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l482_48277


namespace NUMINAMATH_CALUDE_identity_matrix_solution_l482_48200

def matrix_equation (N : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  N^4 - 3 • N^3 + 3 • N^2 - N = !![5, 15; 0, 5]

theorem identity_matrix_solution :
  ∃! N : Matrix (Fin 2) (Fin 2) ℝ, matrix_equation N ∧ N = 1 := by sorry

end NUMINAMATH_CALUDE_identity_matrix_solution_l482_48200


namespace NUMINAMATH_CALUDE_smaller_number_theorem_l482_48210

theorem smaller_number_theorem (x y : ℝ) : 
  x + y = 15 → x * y = 36 → min x y = 3 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_theorem_l482_48210


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l482_48254

theorem complex_sum_theorem (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -2 → b ≠ -2 → c ≠ -2 → d ≠ -2 →
  ω^4 = 1 →
  ω ≠ 1 →
  1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω^2 →
  1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) + 1 / (d + 2) = 2 := by
sorry


end NUMINAMATH_CALUDE_complex_sum_theorem_l482_48254


namespace NUMINAMATH_CALUDE_triangles_in_polygon_l482_48237

/-- The number of triangles formed by diagonals passing through one vertex of an n-sided polygon -/
def triangles_from_diagonals (n : ℕ) : ℕ :=
  n - 2

/-- Theorem stating that the number of triangles formed by diagonals passing through one vertex
    of an n-sided polygon is equal to (n-2) -/
theorem triangles_in_polygon (n : ℕ) (h : n ≥ 3) :
  triangles_from_diagonals n = n - 2 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_polygon_l482_48237


namespace NUMINAMATH_CALUDE_gary_egg_collection_l482_48229

/-- The number of chickens Gary starts with -/
def initial_chickens : ℕ := 4

/-- The factor by which the number of chickens increases after two years -/
def growth_factor : ℕ := 8

/-- The number of eggs each chicken lays per day -/
def eggs_per_chicken_per_day : ℕ := 6

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of eggs Gary collects every week after two years -/
def weekly_egg_collection : ℕ := initial_chickens * growth_factor * eggs_per_chicken_per_day * days_in_week

theorem gary_egg_collection : weekly_egg_collection = 1344 := by
  sorry

end NUMINAMATH_CALUDE_gary_egg_collection_l482_48229


namespace NUMINAMATH_CALUDE_negative_a_sign_l482_48223

theorem negative_a_sign (a : ℝ) : ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ (-a = x ∨ -a = y) :=
  sorry

end NUMINAMATH_CALUDE_negative_a_sign_l482_48223


namespace NUMINAMATH_CALUDE_new_total_weight_l482_48206

/-- Proves that the new total weight of Ram and Shyam is 13.8 times their original common weight factor -/
theorem new_total_weight (x : ℝ) (x_pos : x > 0) : 
  let ram_original := 7 * x
  let shyam_original := 5 * x
  let ram_new := ram_original * 1.1
  let shyam_new := shyam_original * 1.22
  let total_original := ram_original + shyam_original
  let total_new := ram_new + shyam_new
  total_new = total_original * 1.15 ∧ total_new = 13.8 * x :=
by sorry

end NUMINAMATH_CALUDE_new_total_weight_l482_48206


namespace NUMINAMATH_CALUDE_factorization_of_ax_squared_minus_9a_l482_48225

theorem factorization_of_ax_squared_minus_9a (a x : ℝ) : a * x^2 - 9 * a = a * (x - 3) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_ax_squared_minus_9a_l482_48225


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l482_48216

/-- Given an arithmetic sequence {a_n} with a_1 = 2 and common difference d = 3,
    prove that the fifth term a_5 equals 14. -/
theorem arithmetic_sequence_fifth_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 3) →  -- Common difference is 3
  a 1 = 2 →                    -- First term is 2
  a 5 = 14 :=                  -- Fifth term is 14
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l482_48216


namespace NUMINAMATH_CALUDE_pirate_treasure_sum_l482_48293

def base7_to_base10 (n : ℕ) : ℕ := sorry

def diamonds : ℕ := 6352
def ancient_coins : ℕ := 3206
def silver : ℕ := 156

theorem pirate_treasure_sum :
  base7_to_base10 diamonds + base7_to_base10 ancient_coins + base7_to_base10 silver = 3465 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_sum_l482_48293


namespace NUMINAMATH_CALUDE_all_composites_reachable_l482_48251

/-- A proper divisor of n is a positive integer that divides n and is not equal to 1 or n. -/
def ProperDivisor (d n : ℕ) : Prop :=
  d ∣ n ∧ d ≠ 1 ∧ d ≠ n

/-- The set of numbers that can be obtained by starting from 4 and repeatedly adding proper divisors. -/
inductive Reachable : ℕ → Prop
  | base : Reachable 4
  | step {n m : ℕ} : Reachable n → ProperDivisor m n → Reachable (n + m)

/-- A composite number is a natural number greater than 1 that is not prime. -/
def Composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- Theorem: Any composite number can be reached by starting from 4 and repeatedly adding proper divisors. -/
theorem all_composites_reachable : ∀ n : ℕ, Composite n → Reachable n := by
  sorry

end NUMINAMATH_CALUDE_all_composites_reachable_l482_48251


namespace NUMINAMATH_CALUDE_divisors_of_2_pow_48_minus_1_l482_48201

theorem divisors_of_2_pow_48_minus_1 :
  ∃! (a b : ℕ), 60 < a ∧ a < 70 ∧ 60 < b ∧ b < 70 ∧
  (2^48 - 1) % a = 0 ∧ (2^48 - 1) % b = 0 ∧
  a = 63 ∧ b = 65 := by sorry

end NUMINAMATH_CALUDE_divisors_of_2_pow_48_minus_1_l482_48201


namespace NUMINAMATH_CALUDE_min_value_theorem_l482_48238

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geo_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^(2*b))) : 
  2/a + 1/b ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l482_48238


namespace NUMINAMATH_CALUDE_distance_center_to_line_l482_48242

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the circle C
def circle_C (x y θ : ℝ) : Prop :=
  x = 2 * Real.cos θ ∧ y = 2 * Real.sin θ + 2 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Theorem statement
theorem distance_center_to_line :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x y θ : ℝ, circle_C x y θ → (x - x₀)^2 + (y - y₀)^2 ≤ 4) ∧
    (|x₀ + y₀ - 6| / Real.sqrt 2 = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_distance_center_to_line_l482_48242


namespace NUMINAMATH_CALUDE_max_weighings_for_15_coins_l482_48290

/-- Represents a coin which can be either genuine or counterfeit -/
inductive Coin
| genuine : Coin
| counterfeit : Coin

/-- Represents the result of a weighing -/
inductive WeighingResult
| left_heavier : WeighingResult
| right_heavier : WeighingResult
| equal : WeighingResult

/-- A function that simulates weighing two groups of coins -/
def weigh (left : List Coin) (right : List Coin) : WeighingResult := sorry

/-- A function that finds the counterfeit coin -/
def find_counterfeit (coins : List Coin) : Nat → Option Coin := sorry

theorem max_weighings_for_15_coins :
  ∀ (coins : List Coin),
    coins.length = 15 →
    (∃! c, c ∈ coins ∧ c = Coin.counterfeit) →
    ∃ n, n ≤ 3 ∧ (find_counterfeit coins n).isSome ∧
        ∀ m, m < n → (find_counterfeit coins m).isNone := by sorry

#check max_weighings_for_15_coins

end NUMINAMATH_CALUDE_max_weighings_for_15_coins_l482_48290


namespace NUMINAMATH_CALUDE_older_brother_stamps_l482_48265

theorem older_brother_stamps : 
  ∀ (younger older : ℕ), 
  younger + older = 25 → 
  older = 2 * younger + 1 → 
  older = 17 := by sorry

end NUMINAMATH_CALUDE_older_brother_stamps_l482_48265


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l482_48246

def complex_i : ℂ := Complex.I

def z : ℂ := complex_i + complex_i^2

def second_quadrant (c : ℂ) : Prop :=
  c.re < 0 ∧ c.im > 0

theorem z_in_second_quadrant : second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l482_48246


namespace NUMINAMATH_CALUDE_infinite_composite_values_l482_48250

theorem infinite_composite_values (m n k : ℕ) :
  (∃ f : ℕ → ℕ, ∀ k ≥ 2, f k = 4 * k^4) ∧
  (∀ k ≥ 2, ∀ m, ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ m^4 + 4 * k^4 = a * b) :=
sorry

end NUMINAMATH_CALUDE_infinite_composite_values_l482_48250


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l482_48264

theorem sphere_radius_ratio (V_large : ℝ) (V_small : ℝ) :
  V_large = 288 * Real.pi →
  V_small = 0.125 * V_large →
  ∃ (r_large r_small : ℝ),
    V_large = (4 / 3) * Real.pi * r_large^3 ∧
    V_small = (4 / 3) * Real.pi * r_small^3 ∧
    r_small / r_large = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l482_48264


namespace NUMINAMATH_CALUDE_distance_minus_two_to_three_l482_48275

-- Define the distance function between two points on a number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem distance_minus_two_to_three : distance (-2) 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_minus_two_to_three_l482_48275


namespace NUMINAMATH_CALUDE_sandwich_combinations_l482_48269

/-- The number of available toppings -/
def num_toppings : ℕ := 10

/-- The number of slice options -/
def num_slice_options : ℕ := 4

/-- The total number of sandwich combinations -/
def total_combinations : ℕ := num_slice_options * 2^num_toppings

/-- Theorem: The total number of sandwich combinations is 4096 -/
theorem sandwich_combinations :
  total_combinations = 4096 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l482_48269


namespace NUMINAMATH_CALUDE_train_journey_solution_l482_48292

/-- Represents the train journey problem -/
structure TrainJourney where
  distance : ℝ  -- Distance between stations in km
  speed : ℝ     -- Initial speed of the train in km/h

/-- Conditions of the train journey -/
def journey_conditions (j : TrainJourney) : Prop :=
  let reduced_speed := j.speed / 3
  let first_day_time := 2 + 0.5 + (j.distance - 2 * j.speed) / reduced_speed
  let second_day_time := (2 * j.speed + 14) / j.speed + 0.5 + (j.distance - (2 * j.speed + 14)) / reduced_speed
  first_day_time = j.distance / j.speed + 7/6 ∧
  second_day_time = j.distance / j.speed + 5/6

/-- The theorem to prove -/
theorem train_journey_solution :
  ∃ j : TrainJourney, journey_conditions j ∧ j.distance = 196 ∧ j.speed = 84 :=
sorry

end NUMINAMATH_CALUDE_train_journey_solution_l482_48292


namespace NUMINAMATH_CALUDE_john_jenny_meeting_point_l482_48270

/-- Represents the running scenario of John and Jenny -/
structure RunningScenario where
  total_distance : ℝ
  uphill_distance : ℝ
  downhill_distance : ℝ
  john_start_time_diff : ℝ
  john_uphill_speed : ℝ
  john_downhill_speed : ℝ
  jenny_uphill_speed : ℝ
  jenny_downhill_speed : ℝ

/-- Calculates the meeting point of John and Jenny -/
def meeting_point (scenario : RunningScenario) : ℝ :=
  sorry

/-- Theorem stating that John and Jenny meet 45/32 km from the top of the hill -/
theorem john_jenny_meeting_point :
  let scenario : RunningScenario := {
    total_distance := 12,
    uphill_distance := 6,
    downhill_distance := 6,
    john_start_time_diff := 1/4,
    john_uphill_speed := 12,
    john_downhill_speed := 18,
    jenny_uphill_speed := 14,
    jenny_downhill_speed := 21
  }
  meeting_point scenario = 45/32 := by sorry

end NUMINAMATH_CALUDE_john_jenny_meeting_point_l482_48270


namespace NUMINAMATH_CALUDE_min_balls_for_twenty_of_one_color_l482_48224

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minBallsForGuarantee (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The specific ball counts in our problem -/
def problemCounts : BallCounts :=
  { red := 35, green := 30, yellow := 25, blue := 15, white := 12, black := 10 }

theorem min_balls_for_twenty_of_one_color :
  minBallsForGuarantee problemCounts 20 = 95 := by sorry

end NUMINAMATH_CALUDE_min_balls_for_twenty_of_one_color_l482_48224


namespace NUMINAMATH_CALUDE_angle_sum_result_l482_48247

theorem angle_sum_result (a b : Real) (h1 : 0 < a ∧ a < π/2) (h2 : 0 < b ∧ b < π/2)
  (h3 : 5 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 2)
  (h4 : 5 * Real.sin (2*a) + 3 * Real.sin (2*b) = 0) :
  2*a + b = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_result_l482_48247
