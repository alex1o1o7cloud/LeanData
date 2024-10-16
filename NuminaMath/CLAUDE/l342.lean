import Mathlib

namespace NUMINAMATH_CALUDE_calculation_proofs_l342_34261

theorem calculation_proofs :
  (∃ (x : ℝ), x = Real.sqrt 12 * Real.sqrt (1/3) - Real.sqrt 18 + |Real.sqrt 2 - 2| ∧ x = 4 - 4 * Real.sqrt 2) ∧
  (∃ (y : ℝ), y = (7 + 4 * Real.sqrt 3) * (7 - 4 * Real.sqrt 3) - (Real.sqrt 3 - 1)^2 ∧ y = 2 * Real.sqrt 3 - 3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l342_34261


namespace NUMINAMATH_CALUDE_solution_set_characterization_l342_34228

open Set

def solution_set (f : ℝ → ℝ) : Set ℝ := {x | x > 0 ∧ f x ≤ Real.log x}

theorem solution_set_characterization
  (f : ℝ → ℝ) (hf : Differentiable ℝ f)
  (h1 : f 1 = 0)
  (h2 : ∀ x > 0, x * (deriv f x) > 1) :
  solution_set f = Ioc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l342_34228


namespace NUMINAMATH_CALUDE_fescue_percentage_in_y_l342_34280

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
def final_mixture (x y : SeedMixture) (x_proportion : ℝ) : SeedMixture :=
  { ryegrass := x_proportion * x.ryegrass + (1 - x_proportion) * y.ryegrass,
    bluegrass := x_proportion * x.bluegrass + (1 - x_proportion) * y.bluegrass,
    fescue := x_proportion * x.fescue + (1 - x_proportion) * y.fescue }

theorem fescue_percentage_in_y
  (x : SeedMixture)
  (y : SeedMixture)
  (h_x_ryegrass : x.ryegrass = 0.4)
  (h_x_bluegrass : x.bluegrass = 0.6)
  (h_x_fescue : x.fescue = 0)
  (h_y_ryegrass : y.ryegrass = 0.25)
  (h_final_ryegrass : (final_mixture x y 0.6667).ryegrass = 0.35)
  : y.fescue = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_fescue_percentage_in_y_l342_34280


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l342_34242

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_derivative_at_one : 
  deriv f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l342_34242


namespace NUMINAMATH_CALUDE_pebbles_count_l342_34282

/-- The width of a splash made by a pebble in meters -/
def pebble_splash : ℚ := 1/4

/-- The width of a splash made by a rock in meters -/
def rock_splash : ℚ := 1/2

/-- The width of a splash made by a boulder in meters -/
def boulder_splash : ℚ := 2

/-- The number of rocks tossed -/
def rocks_tossed : ℕ := 3

/-- The number of boulders tossed -/
def boulders_tossed : ℕ := 2

/-- The total width of all splashes in meters -/
def total_splash_width : ℚ := 7

/-- The number of pebbles tossed -/
def pebbles_tossed : ℕ := 6

theorem pebbles_count :
  pebbles_tossed * pebble_splash + 
  rocks_tossed * rock_splash + 
  boulders_tossed * boulder_splash = 
  total_splash_width := by sorry

end NUMINAMATH_CALUDE_pebbles_count_l342_34282


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l342_34279

theorem rationalize_and_simplify :
  ∃ (A B C D : ℕ), 
    (A * Real.sqrt B + C) / D = Real.sqrt 50 / (Real.sqrt 25 - Real.sqrt 5) ∧
    A * Real.sqrt B + C = 5 * Real.sqrt 2 + Real.sqrt 10 ∧
    D = 4 ∧
    A + B + C + D = 12 ∧
    ∀ (A' B' C' D' : ℕ), 
      (A' * Real.sqrt B' + C') / D' = Real.sqrt 50 / (Real.sqrt 25 - Real.sqrt 5) →
      A' + B' + C' + D' ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l342_34279


namespace NUMINAMATH_CALUDE_coloring_book_coupons_l342_34206

theorem coloring_book_coupons 
  (initial_stock : ℝ) 
  (books_sold : ℝ) 
  (coupons_per_book : ℝ) 
  (h1 : initial_stock = 40.0) 
  (h2 : books_sold = 20.0) 
  (h3 : coupons_per_book = 4.0) : 
  (initial_stock - books_sold) * coupons_per_book = 80.0 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_coupons_l342_34206


namespace NUMINAMATH_CALUDE_flower_beds_count_l342_34278

theorem flower_beds_count (total_seeds : ℕ) (seeds_per_bed : ℕ) (h1 : total_seeds = 270) (h2 : seeds_per_bed = 9) :
  total_seeds / seeds_per_bed = 30 := by
  sorry

end NUMINAMATH_CALUDE_flower_beds_count_l342_34278


namespace NUMINAMATH_CALUDE_existence_of_abc_l342_34214

theorem existence_of_abc (n k : ℕ) (h1 : n > 20) (h2 : k > 1) (h3 : k^2 ∣ n) :
  ∃ a b c : ℕ, n = a * b + b * c + c * a := by
sorry

end NUMINAMATH_CALUDE_existence_of_abc_l342_34214


namespace NUMINAMATH_CALUDE_triangle_area_change_l342_34231

theorem triangle_area_change (h : ℝ) (b₁ b₂ : ℝ) (a₁ a₂ : ℝ) :
  h = 8 ∧ b₁ = 16 ∧ b₂ = 5 ∧
  a₁ = 1/2 * b₁ * h ∧
  a₂ = 1/2 * b₂ * h →
  a₁ = 64 ∧ a₂ = 20 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_change_l342_34231


namespace NUMINAMATH_CALUDE_cube_red_faces_ratio_l342_34265

/-- Represents a cube with side length n -/
structure Cube where
  n : ℕ

/-- Calculates the number of red faces on the original cube -/
def redFaces (c : Cube) : ℕ := 6 * c.n^2

/-- Calculates the total number of faces of all small cubes -/
def totalFaces (c : Cube) : ℕ := 6 * c.n^3

/-- Theorem: The side length of the cube is 3 if and only if
    exactly one-third of the faces of the small cubes are red -/
theorem cube_red_faces_ratio (c : Cube) : 
  c.n = 3 ↔ 3 * redFaces c = totalFaces c := by
  sorry


end NUMINAMATH_CALUDE_cube_red_faces_ratio_l342_34265


namespace NUMINAMATH_CALUDE_cross_section_distance_in_pyramid_l342_34270

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the base hexagon -/
  base_side : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- Distance from the apex of the pyramid -/
  distance_from_apex : ℝ
  /-- Area of the cross section -/
  area : ℝ

/-- Theorem about the distance of a cross section in a right hexagonal pyramid -/
theorem cross_section_distance_in_pyramid 
  (pyramid : RightHexagonalPyramid)
  (section1 section2 : CrossSection) :
  section1.area = 216 * Real.sqrt 3 →
  section2.area = 486 * Real.sqrt 3 →
  |section1.distance_from_apex - section2.distance_from_apex| = 8 →
  section2.distance_from_apex = 24 :=
by sorry

end NUMINAMATH_CALUDE_cross_section_distance_in_pyramid_l342_34270


namespace NUMINAMATH_CALUDE_tan_435_degrees_l342_34252

theorem tan_435_degrees : Real.tan (435 * Real.pi / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_435_degrees_l342_34252


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l342_34262

theorem largest_n_with_unique_k : ∀ n : ℕ,
  n > 112 →
  ¬(∃! k : ℤ, (8 : ℚ)/15 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 7/13) ∧
  (∃! k : ℤ, (8 : ℚ)/15 < (112 : ℚ)/(112 + k) ∧ (112 : ℚ)/(112 + k) < 7/13) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l342_34262


namespace NUMINAMATH_CALUDE_number_division_problem_l342_34208

theorem number_division_problem (x : ℚ) : x / 5 = 70 + x / 6 → x = 2100 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l342_34208


namespace NUMINAMATH_CALUDE_solve_equation_l342_34291

theorem solve_equation (X : ℝ) : (X^3)^(1/2) = 9 * 81^(1/9) → X = 3^(44/27) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l342_34291


namespace NUMINAMATH_CALUDE_symmetric_parabola_b_eq_six_l342_34204

/-- A function f(x) = x^2 + (a+2)x + 3 with domain [a, b] that is symmetric about x = 1 -/
def symmetric_parabola (a b : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, 
    (∀ x ∈ Set.Icc a b, f x = x^2 + (a+2)*x + 3) ∧ 
    (∀ x ∈ Set.Icc a b, f (2 - x) = f x)

/-- If a parabola is symmetric about x = 1, then b = 6 -/
theorem symmetric_parabola_b_eq_six (a b : ℝ) :
  symmetric_parabola a b → b = 6 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_parabola_b_eq_six_l342_34204


namespace NUMINAMATH_CALUDE_original_selling_price_l342_34255

def cost_price : ℝ := 17500

def selling_price : ℝ := 21000

theorem original_selling_price :
  (cost_price = 17500) →
  (0.9 * selling_price = cost_price * 1.08) →
  selling_price = 21000 := by
sorry

end NUMINAMATH_CALUDE_original_selling_price_l342_34255


namespace NUMINAMATH_CALUDE_difference_is_ten_l342_34211

/-- Properties of a rectangular plot -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area_eq : area = 20 * breadth
  breadth_value : breadth = 10

/-- The area of a rectangle -/
def area (plot : RectangularPlot) : ℝ := plot.length * plot.breadth

/-- The difference between length and breadth -/
def length_breadth_difference (plot : RectangularPlot) : ℝ :=
  plot.length - plot.breadth

/-- Theorem: The difference between length and breadth is 10 meters -/
theorem difference_is_ten (plot : RectangularPlot) :
  length_breadth_difference plot = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_is_ten_l342_34211


namespace NUMINAMATH_CALUDE_smallest_divisor_for_perfect_square_l342_34256

/-- A positive integer n is a perfect square if there exists an integer m such that n = m^2 -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- The smallest positive integer n such that 2880/n is a perfect square is 10 -/
theorem smallest_divisor_for_perfect_square : 
  (∀ k : ℕ, k > 0 ∧ k < 10 → ¬ IsPerfectSquare (2880 / k)) ∧ 
  IsPerfectSquare (2880 / 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisor_for_perfect_square_l342_34256


namespace NUMINAMATH_CALUDE_max_take_home_pay_l342_34268

/-- The take-home pay function for income y (in thousand dollars) -/
def P (y : ℝ) : ℝ := -10 * (y - 5)^2 + 1000

/-- The income that yields the greatest take-home pay -/
def max_income : ℝ := 5

theorem max_take_home_pay :
  ∀ y : ℝ, y ≥ 0 → P y ≤ P max_income :=
sorry

end NUMINAMATH_CALUDE_max_take_home_pay_l342_34268


namespace NUMINAMATH_CALUDE_angle_between_vectors_l342_34249

/-- Given vectors a, b, and c in a real inner product space,
    if their norms are equal and nonzero, and a + b = √3 * c,
    then the angle between a and c is π/6. -/
theorem angle_between_vectors (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b c : V) (h1 : ‖a‖ = ‖b‖) (h2 : ‖b‖ = ‖c‖) (h3 : ‖a‖ ≠ 0)
  (h4 : a + b = Real.sqrt 3 • c) :
  Real.arccos (inner a c / (‖a‖ * ‖c‖)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l342_34249


namespace NUMINAMATH_CALUDE_dilution_proof_l342_34272

theorem dilution_proof (initial_volume : ℝ) (initial_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 12 → 
  initial_concentration = 0.6 → 
  target_concentration = 0.4 → 
  water_added = 6 → 
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration := by
  sorry

#check dilution_proof

end NUMINAMATH_CALUDE_dilution_proof_l342_34272


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l342_34263

/-- A hyperbola with center at the origin, focus on the y-axis, and eccentricity √5 -/
structure Hyperbola where
  /-- The eccentricity of the hyperbola -/
  e : ℝ
  /-- The eccentricity is √5 -/
  h_e : e = Real.sqrt 5
  /-- The center is at the origin -/
  center : ℝ × ℝ
  h_center : center = (0, 0)
  /-- The focus is on the y-axis -/
  focus : ℝ × ℝ
  h_focus : focus.1 = 0

/-- The equations of the asymptotes of the hyperbola -/
def asymptotes (h : Hyperbola) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (1/2) * p.1 ∨ p.2 = -(1/2) * p.1}

/-- Theorem: The asymptotes of the given hyperbola are y = ± (1/2)x -/
theorem hyperbola_asymptotes (h : Hyperbola) : 
  asymptotes h = {p : ℝ × ℝ | p.2 = (1/2) * p.1 ∨ p.2 = -(1/2) * p.1} := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l342_34263


namespace NUMINAMATH_CALUDE_perfect_square_and_cube_is_sixth_power_l342_34287

theorem perfect_square_and_cube_is_sixth_power (n : ℕ) :
  (∃ a : ℕ, n = a^2) ∧ (∃ b : ℕ, n = b^3) → ∃ c : ℕ, n = c^6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_and_cube_is_sixth_power_l342_34287


namespace NUMINAMATH_CALUDE_median_intersection_l342_34266

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a median
def median (t : Triangle) (v : Point) (m : Point) : Prop :=
  (v = t.A ∨ v = t.B ∨ v = t.C) ∧
  (m = ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2) ∨
   m = ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2) ∨
   m = ((t.C.1 + t.A.1) / 2, (t.C.2 + t.A.2) / 2))

-- Define the ratio of division
def divides_in_ratio (p : Point) (v : Point) (m : Point) : Prop :=
  let d1 := ((p.1 - v.1)^2 + (p.2 - v.2)^2).sqrt
  let d2 := ((m.1 - p.1)^2 + (m.2 - p.2)^2).sqrt
  d1 / d2 = 2 / 1

-- Theorem statement
theorem median_intersection (t : Triangle) : 
  ∃ (O : Point), 
    (∀ (v m : Point), median t v m → divides_in_ratio O v m) ∧
    (∀ (v1 m1 v2 m2 : Point), 
      median t v1 m1 → median t v2 m2 → 
      ∃ (k : ℝ), O = (k * v1.1 + (1 - k) * m1.1, k * v1.2 + (1 - k) * m1.2) ∧
                 O = (k * v2.1 + (1 - k) * m2.1, k * v2.2 + (1 - k) * m2.2)) :=
by
  sorry

end NUMINAMATH_CALUDE_median_intersection_l342_34266


namespace NUMINAMATH_CALUDE_unique_number_with_digit_product_l342_34251

/-- Given a natural number n, multiply_digits n returns the product of n and all its digits. -/
def multiply_digits (n : ℕ) : ℕ := sorry

/-- Given a natural number n, digits n returns the list of digits of n. -/
def digits (n : ℕ) : List ℕ := sorry

theorem unique_number_with_digit_product : ∃! n : ℕ, multiply_digits n = 1995 ∧ digits n = [5, 7] := by sorry

end NUMINAMATH_CALUDE_unique_number_with_digit_product_l342_34251


namespace NUMINAMATH_CALUDE_parabola_equation_l342_34281

/-- Given a parabola and a circle, prove the equation of the parabola -/
theorem parabola_equation (p : ℝ) (hp : p ≠ 0) :
  (∀ x y, x^2 = 2*p*y) →  -- Parabola equation
  (∀ x y, (x - 2)^2 + (y - 1)^2 = 1) →  -- Circle equation
  (∃ y, ∀ x, (x - 2)^2 + (y - 1)^2 = 1 ∧ y = -p/2) →  -- Axis of parabola is tangent to circle
  (∀ x y, x^2 = -8*y) :=  -- Conclusion: equation of the parabola
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l342_34281


namespace NUMINAMATH_CALUDE_jerry_speed_is_30_l342_34275

/-- Jerry's average speed in miles per hour -/
def jerry_speed : ℝ := 30

/-- Carla's average speed in miles per hour -/
def carla_speed : ℝ := 35

/-- Time difference between Jerry and Carla's departure in hours -/
def time_difference : ℝ := 0.5

/-- Time it takes Carla to catch up to Jerry in hours -/
def catch_up_time : ℝ := 3

/-- Theorem stating that Jerry's speed is 30 miles per hour -/
theorem jerry_speed_is_30 :
  jerry_speed = 30 ∧
  carla_speed * catch_up_time = jerry_speed * (catch_up_time + time_difference) :=
by sorry

end NUMINAMATH_CALUDE_jerry_speed_is_30_l342_34275


namespace NUMINAMATH_CALUDE_vector_collinearity_and_k_value_l342_34222

/-- Given two non-collinear vectors e₁ and e₂ in a vector space, 
    and points A, B, C, D, F such that:
    AB = 2e₁ - 8e₂
    CB = e₁ + 3e₂
    CD = 2e₁ - e₂
    BF = 3e₁ - ke₂
    Prove that:
    1. A, B, and D are collinear
    2. If B, D, and F are collinear, then k = 12 -/
theorem vector_collinearity_and_k_value 
  (V : Type*) [AddCommGroup V] [Module ℝ V]
  (e₁ e₂ : V) (hne : ¬ ∃ (r : ℝ), e₁ = r • e₂)
  (A B C D F : V)
  (hAB : B - A = 2 • e₁ - 8 • e₂)
  (hCB : B - C = e₁ + 3 • e₂)
  (hCD : D - C = 2 • e₁ - e₂)
  (hBF : F - B = 3 • e₁ - k • e₂) :
  (∃ (t : ℝ), D - A = t • (B - A)) ∧
  (∃ (s : ℝ), F - B = s • (D - B) → k = 12) := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_and_k_value_l342_34222


namespace NUMINAMATH_CALUDE_g_1993_of_4_l342_34254

def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

def g_n : ℕ → (ℚ → ℚ)
  | 0 => g
  | n + 1 => λ x => g (g_n n x)

theorem g_1993_of_4 : g_n 1993 4 = 11 / 20 := by sorry

end NUMINAMATH_CALUDE_g_1993_of_4_l342_34254


namespace NUMINAMATH_CALUDE_hcl_effects_l342_34276

-- Define the initial state of distilled water
structure DistilledWater :=
  (temp : ℝ)
  (pH : ℝ)
  (c_H : ℝ)
  (c_OH : ℝ)
  (Kw : ℝ)

-- Define the state after adding HCl
structure WaterWithHCl :=
  (temp : ℝ)
  (pH : ℝ)
  (c_H : ℝ)
  (c_OH : ℝ)
  (Kw : ℝ)
  (c_HCl : ℝ)

-- Define the theorem
theorem hcl_effects 
  (initial : DistilledWater) 
  (final : WaterWithHCl) 
  (h_temp : final.temp = initial.temp) 
  (h_HCl : final.c_HCl > 0) :
  (final.Kw = initial.Kw) ∧ 
  (final.pH < initial.pH) ∧ 
  (final.c_OH < initial.c_OH) ∧ 
  (final.c_H - final.c_HCl < initial.c_H) :=
sorry

end NUMINAMATH_CALUDE_hcl_effects_l342_34276


namespace NUMINAMATH_CALUDE_sum_of_numbers_l342_34296

theorem sum_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x * y = 16) (h2 : 1 / x = 5 / y) : x + y = 24 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l342_34296


namespace NUMINAMATH_CALUDE_sine_cosine_sum_equals_root_two_over_two_l342_34258

theorem sine_cosine_sum_equals_root_two_over_two :
  Real.sin (30 * π / 180) * Real.cos (15 * π / 180) + 
  Real.cos (30 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_equals_root_two_over_two_l342_34258


namespace NUMINAMATH_CALUDE_school_garden_flowers_l342_34239

theorem school_garden_flowers :
  let green_flowers : ℕ := 9
  let red_flowers : ℕ := 3 * green_flowers
  let yellow_flowers : ℕ := 12
  let total_flowers : ℕ := green_flowers + red_flowers + yellow_flowers + (green_flowers + red_flowers + yellow_flowers)
  total_flowers = 96 := by
  sorry

end NUMINAMATH_CALUDE_school_garden_flowers_l342_34239


namespace NUMINAMATH_CALUDE_wire_cutting_l342_34293

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) :
  total_length = 50 →
  ratio = 2 / 5 →
  shorter_piece + shorter_piece / ratio = total_length →
  shorter_piece = 100 / 7 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l342_34293


namespace NUMINAMATH_CALUDE_jills_lavender_candles_l342_34210

/-- Represents the number of candles made with each scent -/
structure CandleCounts where
  lavender : ℕ
  coconut : ℕ
  almond : ℕ
  jasmine : ℕ

/-- Represents the amount of scent required for each candle -/
structure ScentRequirements where
  lavender : ℕ
  coconut : ℕ
  almond : ℕ
  jasmine : ℕ

/-- Theorem stating the number of lavender candles Jill made -/
theorem jills_lavender_candles 
  (req : ScentRequirements)
  (counts : CandleCounts)
  (h1 : req.lavender = 10)
  (h2 : req.coconut = 8)
  (h3 : req.almond = 12)
  (h4 : req.jasmine = 14)
  (h5 : counts.lavender = 3 * counts.coconut)
  (h6 : counts.almond = 2 * counts.jasmine)
  (h7 : counts.almond = 10)
  (h8 : req.coconut * counts.coconut = (5/2) * req.almond * counts.almond)
  : counts.lavender = 111 := by
  sorry

#check jills_lavender_candles

end NUMINAMATH_CALUDE_jills_lavender_candles_l342_34210


namespace NUMINAMATH_CALUDE_alcohol_percentage_problem_l342_34285

theorem alcohol_percentage_problem (initial_volume : Real) 
  (added_alcohol : Real) (final_percentage : Real) :
  initial_volume = 6 →
  added_alcohol = 3.6 →
  final_percentage = 50 →
  let final_volume := initial_volume + added_alcohol
  let final_alcohol := final_volume * (final_percentage / 100)
  let initial_alcohol := final_alcohol - added_alcohol
  initial_alcohol / initial_volume * 100 = 20 := by
sorry


end NUMINAMATH_CALUDE_alcohol_percentage_problem_l342_34285


namespace NUMINAMATH_CALUDE_legos_given_to_sister_l342_34253

theorem legos_given_to_sister (initial : ℕ) (lost : ℕ) (current : ℕ) : 
  initial = 380 → lost = 57 → current = 299 → initial - lost - current = 24 :=
by sorry

end NUMINAMATH_CALUDE_legos_given_to_sister_l342_34253


namespace NUMINAMATH_CALUDE_cobbler_hourly_rate_l342_34218

theorem cobbler_hourly_rate 
  (mold_cost : ℝ) 
  (work_hours : ℝ) 
  (discount_rate : ℝ) 
  (total_payment : ℝ) 
  (h1 : mold_cost = 250)
  (h2 : work_hours = 8)
  (h3 : discount_rate = 0.8)
  (h4 : total_payment = 730) :
  ∃ hourly_rate : ℝ, 
    hourly_rate = 75 ∧ 
    total_payment = mold_cost + discount_rate * work_hours * hourly_rate :=
by
  sorry

end NUMINAMATH_CALUDE_cobbler_hourly_rate_l342_34218


namespace NUMINAMATH_CALUDE_line_equation_slope_intercept_l342_34229

/-- Given a line equation, prove its slope and y-intercept -/
theorem line_equation_slope_intercept :
  ∀ (x y : ℝ),
  (3 : ℝ) * (x - 2) + (-4 : ℝ) * (y - 8) = 0 →
  ∃ (m b : ℝ), m = 3/4 ∧ b = 13/2 ∧ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_line_equation_slope_intercept_l342_34229


namespace NUMINAMATH_CALUDE_unique_student_count_l342_34289

/-- Represents the number of students that can be seated in large boats -/
def large_boat_capacity (num_boats : ℕ) : ℕ := 17 * num_boats + 6

/-- Represents the number of students that can be seated in small boats -/
def small_boat_capacity (num_boats : ℕ) : ℕ := 10 * num_boats + 2

/-- Theorem stating that 142 is the only number of students satisfying all conditions -/
theorem unique_student_count : 
  ∃! n : ℕ, 
    100 < n ∧ 
    n < 200 ∧ 
    (∃ x y : ℕ, large_boat_capacity x = n ∧ small_boat_capacity y = n) :=
by
  sorry

#check unique_student_count

end NUMINAMATH_CALUDE_unique_student_count_l342_34289


namespace NUMINAMATH_CALUDE_exists_fibonacci_divisible_by_10000_l342_34244

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem exists_fibonacci_divisible_by_10000 :
  ∃ k, k ≤ 10^8 + 1 ∧ fibonacci k % 10000 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_fibonacci_divisible_by_10000_l342_34244


namespace NUMINAMATH_CALUDE_arithmetic_progression_ratio_l342_34236

/-- Given three numbers in arithmetic progression where the largest is 70
    and the difference between the smallest and largest is 40,
    prove that their ratio is 3:5:7 -/
theorem arithmetic_progression_ratio :
  ∀ (a b c : ℕ),
  c = 70 →
  c - a = 40 →
  b - a = c - b →
  ∃ (k : ℕ), k > 0 ∧ a = 3 * k ∧ b = 5 * k ∧ c = 7 * k :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_ratio_l342_34236


namespace NUMINAMATH_CALUDE_complex_powers_sum_l342_34221

theorem complex_powers_sum (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_powers_sum_l342_34221


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l342_34288

/-- A figure composed of a triangle and an adjacent quadrilateral -/
structure TriangleQuadrilateralFigure where
  /-- The three sides of the triangle -/
  triangle_side1 : ℝ
  triangle_side2 : ℝ
  triangle_side3 : ℝ
  /-- The side length of the quadrilateral (all sides equal) -/
  quad_side : ℝ

/-- The perimeter of the TriangleQuadrilateralFigure -/
def perimeter (figure : TriangleQuadrilateralFigure) : ℝ :=
  figure.triangle_side1 + figure.triangle_side2 + figure.triangle_side3 + 4 * figure.quad_side

/-- Theorem stating that the perimeter of the specific figure is 44 -/
theorem specific_figure_perimeter :
  ∃ (figure : TriangleQuadrilateralFigure),
    figure.triangle_side1 = 6 ∧
    figure.triangle_side2 = 8 ∧
    figure.triangle_side3 = 10 ∧
    figure.quad_side = 5 ∧
    perimeter figure = 44 :=
sorry

end NUMINAMATH_CALUDE_specific_figure_perimeter_l342_34288


namespace NUMINAMATH_CALUDE_bee_flight_time_l342_34205

/-- Flight time of a honey bee between flowers -/
theorem bee_flight_time (time_daisy_to_rose : ℝ) (speed_daisy_to_rose : ℝ) (speed_difference : ℝ) (distance_difference : ℝ) :
  time_daisy_to_rose = 10 →
  speed_daisy_to_rose = 2.6 →
  speed_difference = 3 →
  distance_difference = 8 →
  ∃ (time_rose_to_poppy : ℝ),
    time_rose_to_poppy > 0 ∧
    time_rose_to_poppy < 4 ∧
    (speed_daisy_to_rose * time_daisy_to_rose - distance_difference) / (speed_daisy_to_rose + speed_difference) = time_rose_to_poppy :=
by sorry

end NUMINAMATH_CALUDE_bee_flight_time_l342_34205


namespace NUMINAMATH_CALUDE_machine_chip_production_l342_34225

/-- The number of computer chips produced by a machine in a day, given the number of
    video game consoles it can supply chips for and the number of chips per console. -/
def chips_per_day (consoles_per_day : ℕ) (chips_per_console : ℕ) : ℕ :=
  consoles_per_day * chips_per_console

/-- Theorem stating that a machine supplying chips for 93 consoles per day,
    with 5 chips per console, produces 465 chips per day. -/
theorem machine_chip_production :
  chips_per_day 93 5 = 465 := by
  sorry

end NUMINAMATH_CALUDE_machine_chip_production_l342_34225


namespace NUMINAMATH_CALUDE_parabola_c_value_l342_34247

/-- Represents a parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a parabola, its vertex, and a point it passes through, prove that c = 9/2 -/
theorem parabola_c_value (p : Parabola) (h1 : p.a * (-1)^2 + p.b * (-1) + p.c = 5)
    (h2 : p.a * 1^2 + p.b * 1 + p.c = 3) : p.c = 9/2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_c_value_l342_34247


namespace NUMINAMATH_CALUDE_haley_josh_necklace_difference_haley_josh_necklace_difference_proof_l342_34223

/-- Given the number of necklaces for Haley, Jason, and Josh, prove that Haley has 15 more necklaces than Josh. -/
theorem haley_josh_necklace_difference : ℕ → ℕ → ℕ → Prop :=
  fun haley jason josh =>
    (haley = jason + 5) →
    (josh = jason / 2) →
    (haley = 25) →
    (haley - josh = 15)

/-- Proof of the theorem -/
theorem haley_josh_necklace_difference_proof :
  ∀ haley jason josh, haley_josh_necklace_difference haley jason josh :=
by
  sorry

#check haley_josh_necklace_difference
#check haley_josh_necklace_difference_proof

end NUMINAMATH_CALUDE_haley_josh_necklace_difference_haley_josh_necklace_difference_proof_l342_34223


namespace NUMINAMATH_CALUDE_china_internet_users_scientific_notation_l342_34295

/-- Represents the number of internet users in China in billions -/
def china_internet_users : ℝ := 1.067

/-- The scientific notation representation of the number of internet users -/
def scientific_notation : ℝ := 1.067 * (10 ^ 9)

theorem china_internet_users_scientific_notation :
  china_internet_users * (10 ^ 9) = scientific_notation := by sorry

end NUMINAMATH_CALUDE_china_internet_users_scientific_notation_l342_34295


namespace NUMINAMATH_CALUDE_open_box_volume_l342_34238

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_length : ℝ) 
  (h1 : sheet_length = 48) 
  (h2 : sheet_width = 36) 
  (h3 : cut_length = 8) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 5120 := by
  sorry

#check open_box_volume

end NUMINAMATH_CALUDE_open_box_volume_l342_34238


namespace NUMINAMATH_CALUDE_significant_figures_of_number_l342_34213

/-- Count the number of significant figures in a rational number represented as a string -/
def count_significant_figures (s : String) : ℕ := sorry

/-- The rational number in question -/
def number : String := "0.0050400"

/-- Theorem stating that the number of significant figures in 0.0050400 is 5 -/
theorem significant_figures_of_number : count_significant_figures number = 5 := by sorry

end NUMINAMATH_CALUDE_significant_figures_of_number_l342_34213


namespace NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l342_34257

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k > 24 ∧ ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ m : ℤ, m ≤ 24 → (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_five_consecutive_integers_l342_34257


namespace NUMINAMATH_CALUDE_min_value_inequality_l342_34207

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a + 1 / b + 2 * Real.sqrt (a * b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l342_34207


namespace NUMINAMATH_CALUDE_muffin_spending_l342_34219

theorem muffin_spending (x : ℝ) : 
  (x = 0.9 * x + 15) → (x + 0.9 * x = 285) :=
by sorry

end NUMINAMATH_CALUDE_muffin_spending_l342_34219


namespace NUMINAMATH_CALUDE_joans_games_l342_34267

theorem joans_games (football_this_year basketball_this_year total_both_years : ℕ)
  (h1 : football_this_year = 4)
  (h2 : basketball_this_year = 3)
  (h3 : total_both_years = 9) :
  total_both_years - (football_this_year + basketball_this_year) = 2 := by
sorry

end NUMINAMATH_CALUDE_joans_games_l342_34267


namespace NUMINAMATH_CALUDE_min_value_expression_l342_34297

theorem min_value_expression (x y : ℝ) 
  (h : 4 - 16 * x^2 - 8 * x * y - y^2 > 0) : 
  (13 * x^2 + 24 * x * y + 13 * y^2 - 14 * x - 16 * y + 61) / 
  (4 - 16 * x^2 - 8 * x * y - y^2)^(7/2) ≥ 7/16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l342_34297


namespace NUMINAMATH_CALUDE_gain_percentage_is_20_percent_l342_34271

def selling_price : ℝ := 90
def gain : ℝ := 15

theorem gain_percentage_is_20_percent :
  let cost_price := selling_price - gain
  (gain / cost_price) * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_gain_percentage_is_20_percent_l342_34271


namespace NUMINAMATH_CALUDE_unit_digit_of_expression_l342_34212

theorem unit_digit_of_expression : ∃ n : ℕ, n % 10 = 4 ∧ 
  n = (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_expression_l342_34212


namespace NUMINAMATH_CALUDE_vince_savings_l342_34292

/-- Calculates Vince's savings given his earnings per customer, number of customers,
    fixed expenses, and percentage allocated for recreation. -/
def calculate_savings (earnings_per_customer : ℚ) (num_customers : ℕ) 
                      (fixed_expenses : ℚ) (recreation_percent : ℚ) : ℚ :=
  let total_earnings := earnings_per_customer * num_customers
  let recreation_amount := recreation_percent * total_earnings
  let total_expenses := fixed_expenses + recreation_amount
  total_earnings - total_expenses

/-- Proves that Vince's savings are $872 given the problem conditions. -/
theorem vince_savings : 
  calculate_savings 18 80 280 (20/100) = 872 := by
  sorry

end NUMINAMATH_CALUDE_vince_savings_l342_34292


namespace NUMINAMATH_CALUDE_limit_fraction_binomial_sums_l342_34220

def a (n : ℕ+) : ℝ := (3 : ℝ) ^ n.val
def b (n : ℕ+) : ℝ := (2 : ℝ) ^ n.val

theorem limit_fraction_binomial_sums :
  ∀ ε > 0, ∃ N : ℕ+, ∀ n ≥ N,
    |((b (n + 1) - a n) / (a (n + 1) + b n)) + (1 / 3)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_fraction_binomial_sums_l342_34220


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l342_34273

/-- A natural number that ends in 5 zeros and has exactly 42 divisors -/
def SpecialNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^5 * k ∧ (Nat.divisors n).card = 42

/-- The theorem stating that there are exactly two distinct natural numbers
    that satisfy the SpecialNumber property, and their sum is 700000 -/
theorem sum_of_special_numbers :
  ∃! (a b : ℕ), a < b ∧ SpecialNumber a ∧ SpecialNumber b ∧ a + b = 700000 := by
  sorry

#check sum_of_special_numbers

end NUMINAMATH_CALUDE_sum_of_special_numbers_l342_34273


namespace NUMINAMATH_CALUDE_parabola_shift_l342_34209

-- Define the initial parabola
def initial_parabola (x y : ℝ) : Prop :=
  y = -1/3 * (x - 2)^2

-- Define the shift
def shift_right : ℝ := 1
def shift_down : ℝ := 2

-- Define the resulting parabola
def resulting_parabola (x y : ℝ) : Prop :=
  y = -1/3 * (x - 3)^2 - 2

-- Theorem statement
theorem parabola_shift :
  ∀ x y : ℝ, initial_parabola x y →
  resulting_parabola (x + shift_right) (y - shift_down) :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l342_34209


namespace NUMINAMATH_CALUDE_root_conditions_l342_34294

theorem root_conditions (a b c : ℝ) : 
  (∀ x : ℝ, x^5 + 4*x^4 + a*x^2 = b*x + 4*c ↔ x = 2 ∨ x = -2) ↔ 
  (a = -48 ∧ b = 16 ∧ c = -32) :=
sorry

end NUMINAMATH_CALUDE_root_conditions_l342_34294


namespace NUMINAMATH_CALUDE_omega_values_l342_34201

theorem omega_values (z ω : ℂ) 
  (h1 : ∃ (k : ℝ), (1 + 3*I) * z = k*I)
  (h2 : ω = z / (2 + I))
  (h3 : Complex.abs ω = 5 * Real.sqrt 2) :
  ω = 7 - I ∨ ω = -7 + I := by
  sorry

end NUMINAMATH_CALUDE_omega_values_l342_34201


namespace NUMINAMATH_CALUDE_triangle_problem_l342_34284

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  A = 2 * π / 3 →  -- 120° in radians
  c > b →
  a = Real.sqrt 21 →
  S = Real.sqrt 3 →
  (1/2) * b * c * Real.sin A = S →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  (∃ (B C : ℝ), A + B + C = π ∧ 
    a / Real.sin A = b / Real.sin B ∧
    Real.cos C = (a^2 + b^2 - c^2) / (2*a*b)) →
  (b = 1 ∧ c = 4) ∧
  Real.sin B + Real.cos C = (Real.sqrt 7 + 2 * Real.sqrt 21) / 14 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l342_34284


namespace NUMINAMATH_CALUDE_range_of_a_l342_34298

/-- Given propositions p and q, and the condition that ¬p is a sufficient but not necessary condition for ¬q, prove that the range of real number a is [-1, 2]. -/
theorem range_of_a (a : ℝ) : 
  (∀ x, (x^2 - (2*a+4)*x + a^2 + 4*a < 0) ↔ (a < x ∧ x < a+4)) →
  (∀ x, ((x-2)*(x-3) < 0) ↔ (2 < x ∧ x < 3)) →
  (∀ x, ¬(a < x ∧ x < a+4) → ¬(2 < x ∧ x < 3)) →
  (∃ x, (2 < x ∧ x < 3) ∧ ¬(a < x ∧ x < a+4)) →
  -1 ≤ a ∧ a ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_range_of_a_l342_34298


namespace NUMINAMATH_CALUDE_balanced_equation_oxygen_coefficient_l342_34260

/-- Represents a chemical element in a molecule --/
inductive Element
  | As
  | S
  | O

/-- Represents a molecule in a chemical equation --/
structure Molecule where
  elements : List (Element × Nat)

/-- Represents a side of a chemical equation --/
structure EquationSide where
  molecules : List (Molecule × Nat)

/-- Represents a chemical equation --/
structure ChemicalEquation where
  leftSide : EquationSide
  rightSide : EquationSide

/-- Checks if a chemical equation is balanced --/
def isBalanced (eq : ChemicalEquation) : Bool :=
  sorry

/-- Checks if coefficients are the smallest possible integers --/
def hasSmallestCoefficients (eq : ChemicalEquation) : Bool :=
  sorry

/-- The coefficient of O₂ in the balanced equation --/
def oxygenCoefficient (eq : ChemicalEquation) : Nat :=
  sorry

theorem balanced_equation_oxygen_coefficient :
  ∀ (eq : ChemicalEquation),
    eq.leftSide.molecules = [
      (Molecule.mk [(Element.As, 2), (Element.S, 3)], 2),
      (Molecule.mk [(Element.O, 2)], oxygenCoefficient eq)
    ] →
    eq.rightSide.molecules = [
      (Molecule.mk [(Element.As, 2), (Element.O, 3)], 4),
      (Molecule.mk [(Element.S, 1), (Element.O, 2)], 6)
    ] →
    isBalanced eq →
    hasSmallestCoefficients eq →
    oxygenCoefficient eq = 9 :=
  sorry

end NUMINAMATH_CALUDE_balanced_equation_oxygen_coefficient_l342_34260


namespace NUMINAMATH_CALUDE_spade_problem_l342_34215

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_problem : spade 5 (spade 3 9) = 1 := by sorry

end NUMINAMATH_CALUDE_spade_problem_l342_34215


namespace NUMINAMATH_CALUDE_house_selling_price_l342_34240

theorem house_selling_price (commission_rate : ℝ) (commission : ℝ) (selling_price : ℝ) :
  commission_rate = 0.06 →
  commission = 8880 →
  commission = commission_rate * selling_price →
  selling_price = 148000 := by
  sorry

end NUMINAMATH_CALUDE_house_selling_price_l342_34240


namespace NUMINAMATH_CALUDE_simplify_expression_l342_34230

theorem simplify_expression (x : ℝ) : (5 - 2*x) - (7 + 3*x) = -2 - 5*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l342_34230


namespace NUMINAMATH_CALUDE_root_difference_l342_34237

/-- The polynomial coefficients -/
def a : ℚ := 8
def b : ℚ := -22
def c : ℚ := 15
def d : ℚ := -2

/-- The polynomial function -/
def f (x : ℚ) : ℚ := a * x^3 + b * x^2 + c * x + d

/-- The roots of the polynomial are in geometric progression -/
axiom roots_in_geometric_progression : ∃ (r₁ r₂ r₃ : ℚ), 
  (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) ∧ 
  (∃ (q : ℚ), r₂ = r₁ * q ∧ r₃ = r₂ * q)

/-- The theorem to be proved -/
theorem root_difference : 
  ∃ (r₁ r₃ : ℚ), (f r₁ = 0 ∧ f r₃ = 0) ∧ 
  (∀ r, f r = 0 → r₁ ≤ r ∧ r ≤ r₃) ∧
  (r₃ - r₁ = 33 / 14) := by
sorry

end NUMINAMATH_CALUDE_root_difference_l342_34237


namespace NUMINAMATH_CALUDE_evaluate_expression_l342_34250

theorem evaluate_expression (a : ℝ) (h : a = 2) : (7 * a ^ 2 - 10 * a + 3) * (3 * a - 4) = 22 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l342_34250


namespace NUMINAMATH_CALUDE_infiniteSeriesSum_l342_34274

/-- The sum of the infinite series Σ(k/3^k) for k from 1 to ∞ -/
noncomputable def infiniteSeries : ℝ := ∑' k, k / (3 ^ k)

/-- Theorem: The sum of the infinite series Σ(k/3^k) for k from 1 to ∞ is equal to 3/4 -/
theorem infiniteSeriesSum : infiniteSeries = 3/4 := by sorry

end NUMINAMATH_CALUDE_infiniteSeriesSum_l342_34274


namespace NUMINAMATH_CALUDE_red_blocks_count_l342_34224

theorem red_blocks_count (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : yellow = red + 7)
  (h2 : blue = red + 14)
  (h3 : red + yellow + blue = 75) :
  red = 18 := by
  sorry

end NUMINAMATH_CALUDE_red_blocks_count_l342_34224


namespace NUMINAMATH_CALUDE_jacksons_running_distance_l342_34277

/-- Calculates the final daily running distance after a given number of weeks,
    starting from an initial distance and increasing by a fixed amount each week. -/
def finalRunningDistance (initialDistance : ℕ) (weeklyIncrease : ℕ) (totalWeeks : ℕ) : ℕ :=
  initialDistance + weeklyIncrease * (totalWeeks - 1)

/-- Proves that Jackson's final daily running distance is 7 miles
    after 5 weeks of training. -/
theorem jacksons_running_distance :
  finalRunningDistance 3 1 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_jacksons_running_distance_l342_34277


namespace NUMINAMATH_CALUDE_proportion_equality_l342_34246

theorem proportion_equality (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l342_34246


namespace NUMINAMATH_CALUDE_right_triangle_power_equation_l342_34234

theorem right_triangle_power_equation (a b c n : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  n > 2 →
  a^2 + b^2 = c^2 →
  (a^n + b^n + c^n)^2 = 2*(a^(2*n) + b^(2*n) + c^(2*n)) →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_power_equation_l342_34234


namespace NUMINAMATH_CALUDE_sin_cos_shift_l342_34202

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (2 * x - π / 4) = Real.cos (2 * (x - 3 * π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l342_34202


namespace NUMINAMATH_CALUDE_equation_solution_l342_34235

theorem equation_solution (n : ℤ) : n + (n + 1) + (n + 2) = 15 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l342_34235


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l342_34226

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 9*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  p + q = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l342_34226


namespace NUMINAMATH_CALUDE_collinear_vectors_k_l342_34264

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-2, 1]
def c : Fin 2 → ℝ := ![3, 2]

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, ∀ i : Fin 2, v i = t * w i

theorem collinear_vectors_k (k : ℝ) :
  collinear c (fun i ↦ k * a i + b i) → k = -1 := by
  sorry

#check collinear_vectors_k

end NUMINAMATH_CALUDE_collinear_vectors_k_l342_34264


namespace NUMINAMATH_CALUDE_fated_number_probability_l342_34243

def Digit := Fin 4

structure ThreeDigitNumber where
  hundreds : Digit
  tens : Digit
  units : Digit
  distinct : hundreds ≠ tens ∧ tens ≠ units ∧ hundreds ≠ units

def isFatedNumber (n : ThreeDigitNumber) : Prop :=
  (n.hundreds.val + n.tens.val = n.units.val) ∨
  (n.hundreds.val + n.units.val = n.tens.val) ∨
  (n.tens.val + n.units.val = n.hundreds.val)

def allThreeDigitNumbers : Finset ThreeDigitNumber :=
  sorry

def fatedNumbers : Finset ThreeDigitNumber :=
  sorry

theorem fated_number_probability :
  (Finset.card fatedNumbers : ℚ) / (Finset.card allThreeDigitNumbers) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fated_number_probability_l342_34243


namespace NUMINAMATH_CALUDE_n_ge_digit_product_eq_digit_product_iff_eq_four_l342_34259

/-- Function that returns the product of digits of a positive integer -/
def digit_product (n : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that n is greater than or equal to the product of its digits -/
theorem n_ge_digit_product (n : ℕ+) : (n : ℕ) ≥ digit_product n :=
  sorry

/-- Theorem stating that n^2 - 17n + 56 equals the product of digits of n if and only if n = 4 -/
theorem eq_digit_product_iff_eq_four (n : ℕ+) : 
  (n : ℕ)^2 - 17*(n : ℕ) + 56 = digit_product n ↔ n = 4 :=
  sorry

end NUMINAMATH_CALUDE_n_ge_digit_product_eq_digit_product_iff_eq_four_l342_34259


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l342_34217

theorem smallest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) → (n % 17 = 0) → n ≥ 102 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l342_34217


namespace NUMINAMATH_CALUDE_range_of_average_l342_34227

theorem range_of_average (α β : ℝ) (h1 : -π/2 < α) (h2 : α < β) (h3 : β < π/2) :
  -π/2 < (α + β) / 2 ∧ (α + β) / 2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_average_l342_34227


namespace NUMINAMATH_CALUDE_cubic_root_sum_l342_34248

theorem cubic_root_sum (p q r : ℕ) (h_pos_p : p > 0) (h_pos_q : q > 0) (h_pos_r : r > 0) :
  (∃ x : ℝ, 27 * x^3 - 11 * x^2 - 11 * x - 3 = 0 ∧ x = (p^(1/3) + q^(1/3) + 1) / r) →
  p + q + r = 782 := by
sorry


end NUMINAMATH_CALUDE_cubic_root_sum_l342_34248


namespace NUMINAMATH_CALUDE_sum_of_powers_half_l342_34283

theorem sum_of_powers_half : 
  (-1/2 : ℚ)^3 + (-1/2 : ℚ)^2 + (-1/2 : ℚ)^1 + (1/2 : ℚ)^1 + (1/2 : ℚ)^2 + (1/2 : ℚ)^3 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_half_l342_34283


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l342_34290

theorem power_zero_eq_one (r : ℚ) (h : r ≠ 0) : r^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l342_34290


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l342_34232

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    (4 * Real.pi * r^2 = 144 * Real.pi) →
    ((4 / 3) * Real.pi * r^3 = 288 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l342_34232


namespace NUMINAMATH_CALUDE_symmetric_line_l342_34286

/-- Given a point (x, y) on the line y = -x + 2, prove that it is symmetric to a point on the line y = x about the line x = 1 -/
theorem symmetric_line (x y : ℝ) : 
  y = -x + 2 → 
  ∃ (x' y' : ℝ), 
    (y' = x') ∧  -- Point (x', y') is on the line y = x
    ((x + x') / 2 = 1) ∧  -- Midpoint of x and x' is on the line x = 1
    (y = y') -- y-coordinates are the same
    := by sorry

end NUMINAMATH_CALUDE_symmetric_line_l342_34286


namespace NUMINAMATH_CALUDE_bird_count_proof_l342_34245

/-- The number of birds initially on a branch -/
def initial_birds (initial_parrots : ℕ) (initial_crows : ℕ) : ℕ :=
  initial_parrots + initial_crows

theorem bird_count_proof 
  (initial_parrots : ℕ) 
  (initial_crows : ℕ) 
  (remaining_parrots : ℕ) 
  (remaining_crow : ℕ) 
  (h1 : initial_parrots = 7)
  (h2 : remaining_parrots = 2)
  (h3 : remaining_crow = 1)
  (h4 : initial_parrots - remaining_parrots = initial_crows - remaining_crow) :
  initial_birds initial_parrots initial_crows = 13 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_proof_l342_34245


namespace NUMINAMATH_CALUDE_integral_cosine_over_trig_combination_l342_34200

theorem integral_cosine_over_trig_combination : 
  ∫ x in -((2 * Real.pi) / 3)..0, (Real.cos x) / (1 + Real.cos x - Real.sin x) = Real.pi / 3 - Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_cosine_over_trig_combination_l342_34200


namespace NUMINAMATH_CALUDE_orange_seller_gain_percentage_l342_34233

theorem orange_seller_gain_percentage
  (loss_rate : ℝ)
  (loss_quantity : ℝ)
  (gain_quantity : ℝ)
  (h_loss_rate : loss_rate = 0.04)
  (h_loss_quantity : loss_quantity = 16)
  (h_gain_quantity : gain_quantity = 12) :
  let cost_price := 1 / (1 - loss_rate)
  let gain_percentage := ((cost_price * gain_quantity) / (1 - loss_rate * cost_price) - 1) * 100
  gain_percentage = 28 := by
sorry

end NUMINAMATH_CALUDE_orange_seller_gain_percentage_l342_34233


namespace NUMINAMATH_CALUDE_distance_between_points_l342_34216

/-- The distance between points (3, 24) and (10, 0) is 25. -/
theorem distance_between_points : Real.sqrt ((10 - 3)^2 + (24 - 0)^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l342_34216


namespace NUMINAMATH_CALUDE_roots_relation_l342_34203

theorem roots_relation (a b c d : ℝ) : 
  (∀ x, (x - a) * (x - b) - x = 0 ↔ x = c ∨ x = d) →
  (∀ x, (x - c) * (x - d) + x = 0 ↔ x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_roots_relation_l342_34203


namespace NUMINAMATH_CALUDE_expression_equals_one_l342_34241

theorem expression_equals_one : 
  (144^2 - 12^2) / (120^2 - 18^2) * ((120-18)*(120+18)) / ((144-12)*(144+12)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l342_34241


namespace NUMINAMATH_CALUDE_notebook_cost_l342_34269

theorem notebook_cost (notebook_cost pencil_cost : ℝ) 
  (total_cost : notebook_cost + pencil_cost = 3.40)
  (price_difference : notebook_cost = pencil_cost + 2) : 
  notebook_cost = 2.70 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l342_34269


namespace NUMINAMATH_CALUDE_cubic_equation_value_l342_34299

theorem cubic_equation_value (x : ℝ) (h : 2 * x^2 - 3 * x - 2022 = 0) :
  2 * x^3 - x^2 - 2025 * x - 2020 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l342_34299
