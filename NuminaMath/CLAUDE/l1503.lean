import Mathlib

namespace NUMINAMATH_CALUDE_barry_sotter_magic_l1503_150369

/-- The increase factor for day k --/
def increase_factor (k : ℕ) : ℚ := (k + 3) / (k + 2)

/-- The overall increase factor after n days --/
def overall_increase (n : ℕ) : ℚ := (n + 3) / 3

theorem barry_sotter_magic (n : ℕ) : overall_increase n = 50 → n = 147 := by
  sorry

end NUMINAMATH_CALUDE_barry_sotter_magic_l1503_150369


namespace NUMINAMATH_CALUDE_christinas_earnings_l1503_150328

/-- The amount Christina earns for planting flowers and mowing the lawn -/
theorem christinas_earnings (flower_rate : ℚ) (mow_rate : ℚ) (flowers_planted : ℚ) (area_mowed : ℚ) :
  flower_rate = 8/3 →
  mow_rate = 5/2 →
  flowers_planted = 9/4 →
  area_mowed = 7/3 →
  flower_rate * flowers_planted + mow_rate * area_mowed = 71/6 := by
sorry

end NUMINAMATH_CALUDE_christinas_earnings_l1503_150328


namespace NUMINAMATH_CALUDE_plane_sphere_intersection_l1503_150389

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere3D where
  center : Point3D
  radius : ℝ

/-- The theorem to be proved -/
theorem plane_sphere_intersection (a b c p q r : ℝ) 
  (plane : Plane3D) 
  (sphere : Sphere3D) : 
  (plane.a = a ∧ plane.b = b ∧ plane.c = c) →  -- Plane passes through (a,b,c)
  (∃ (α β γ : ℝ), 
    plane.a * α + plane.b * 0 + plane.c * 0 + plane.d = 0 ∧  -- Plane intersects x-axis at (α,0,0)
    plane.a * 0 + plane.b * β + plane.c * 0 + plane.d = 0 ∧  -- Plane intersects y-axis at (0,β,0)
    plane.a * 0 + plane.b * 0 + plane.c * γ + plane.d = 0) →  -- Plane intersects z-axis at (0,0,γ)
  (sphere.center = Point3D.mk (p+1) (q+1) (r+1)) →  -- Sphere center is shifted by (1,1,1)
  (∃ (α β γ : ℝ), 
    sphere.radius^2 = (p+1)^2 + (q+1)^2 + (r+1)^2 ∧  -- Sphere passes through origin
    sphere.radius^2 = ((p+1) - α)^2 + (q+1)^2 + (r+1)^2 ∧  -- Sphere passes through A
    sphere.radius^2 = (p+1)^2 + ((q+1) - β)^2 + (r+1)^2 ∧  -- Sphere passes through B
    sphere.radius^2 = (p+1)^2 + (q+1)^2 + ((r+1) - γ)^2) →  -- Sphere passes through C
  a/p + b/q + c/r = 2 := by
  sorry


end NUMINAMATH_CALUDE_plane_sphere_intersection_l1503_150389


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1503_150343

/-- A rectangular solid with prime edge lengths and volume 273 has surface area 302 -/
theorem rectangular_solid_surface_area (a b c : ℕ) : 
  Prime a → Prime b → Prime c → a * b * c = 273 → 2 * (a * b + b * c + c * a) = 302 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1503_150343


namespace NUMINAMATH_CALUDE_bobby_final_paycheck_l1503_150371

/-- Represents Bobby's weekly paycheck calculation -/
def bobby_paycheck (salary : ℝ) (federal_tax_rate : ℝ) (state_tax_rate : ℝ) 
                   (health_insurance : ℝ) (life_insurance : ℝ) (parking_fee : ℝ) : ℝ :=
  salary - (federal_tax_rate * salary) - (state_tax_rate * salary) - 
  health_insurance - life_insurance - parking_fee

/-- Theorem stating that Bobby's final paycheck amount is $184 -/
theorem bobby_final_paycheck : 
  bobby_paycheck 450 (1/3) 0.08 50 20 10 = 184 := by
  sorry

end NUMINAMATH_CALUDE_bobby_final_paycheck_l1503_150371


namespace NUMINAMATH_CALUDE_lost_card_value_l1503_150360

theorem lost_card_value (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ≤ n) : 
  (n * (n + 1)) / 2 - 101 = 4 :=
by sorry

end NUMINAMATH_CALUDE_lost_card_value_l1503_150360


namespace NUMINAMATH_CALUDE_rectangle_problem_l1503_150310

theorem rectangle_problem (a b k l : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < k) (h4 : 0 < l) :
  (13 * (a + b) = a * k) →
  (13 * (a + b) = b * l) →
  (k > l) →
  (k = 182) ∧ (l = 14) := by
sorry

end NUMINAMATH_CALUDE_rectangle_problem_l1503_150310


namespace NUMINAMATH_CALUDE_special_equation_result_l1503_150333

/-- If y is a real number satisfying y + 1/y = 3, then y^13 - 5y^9 + y^5 = 0 -/
theorem special_equation_result (y : ℝ) (h : y + 1/y = 3) : y^13 - 5*y^9 + y^5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_equation_result_l1503_150333


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l1503_150377

/-- The area of a square sheet of wrapping paper required to wrap a rectangular box -/
theorem wrapping_paper_area (w h : ℝ) (hw : w > h) : 
  let l := 2 * w
  let box_diagonal := Real.sqrt (l^2 + w^2 + h^2)
  box_diagonal^2 = 5 * w^2 + h^2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l1503_150377


namespace NUMINAMATH_CALUDE_intersection_is_square_l1503_150366

/-- A rectangle with center at the origin -/
structure CenteredRectangle where
  width : ℝ
  height : ℝ
  (width_pos : 0 < width)
  (height_pos : 0 < height)

/-- The perimeter of a rectangle -/
def perimeter (r : CenteredRectangle) : ℝ := 2 * (r.width + r.height)

/-- The set of all points inside a centered rectangle -/
def rectangle_points (r : CenteredRectangle) : Set (ℝ × ℝ) :=
  {p | abs p.1 ≤ r.width / 2 ∧ abs p.2 ≤ r.height / 2}

/-- The set of all centered rectangles with a given perimeter -/
def rectangles_with_perimeter (p : ℝ) : Set CenteredRectangle :=
  {r | perimeter r = p}

/-- The intersection of all rectangle_points for rectangles with a given perimeter -/
def intersection_of_rectangles (p : ℝ) : Set (ℝ × ℝ) :=
  ⋂ r ∈ rectangles_with_perimeter p, rectangle_points r

/-- A square centered at the origin -/
structure CenteredSquare where
  side : ℝ
  (side_pos : 0 < side)

/-- The set of all points inside a centered square -/
def square_points (s : CenteredSquare) : Set (ℝ × ℝ) :=
  {p | abs p.1 ≤ s.side / 2 ∧ abs p.2 ≤ s.side / 2}

theorem intersection_is_square (p : ℝ) (h : 0 < p) :
  ∃ (s : CenteredSquare), intersection_of_rectangles p = square_points s :=
sorry

end NUMINAMATH_CALUDE_intersection_is_square_l1503_150366


namespace NUMINAMATH_CALUDE_trajectory_of_point_P_l1503_150331

/-- The trajectory of point P given vertices A and B and slope product condition -/
theorem trajectory_of_point_P (x y : ℝ) :
  let A := (0, -Real.sqrt 2)
  let B := (0, Real.sqrt 2)
  let slope_PA := (y - A.2) / (x - A.1)
  let slope_PB := (y - B.2) / (x - B.1)
  x ≠ 0 →
  slope_PA * slope_PB = -2 →
  y^2 / 2 + x^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_point_P_l1503_150331


namespace NUMINAMATH_CALUDE_skew_lines_distance_and_angle_l1503_150337

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary operations and relations
variable (distance : Line → Line → ℝ)
variable (distancePointToLine : Plane → Point → Line → ℝ)
variable (orthogonalProjection : Line → Plane → Line)
variable (angle : Line → Line → ℝ)
variable (perpendicular : Plane → Line → Prop)
variable (intersect : Plane → Line → Point → Prop)
variable (skew : Line → Line → Prop)

-- State the theorem
theorem skew_lines_distance_and_angle 
  (a b : Line) (α : Plane) (A : Point) :
  skew a b →
  perpendicular α a →
  intersect α a A →
  let b' := orthogonalProjection b α
  distance a b = distancePointToLine α A b' ∧
  angle b b' + angle a b = 90 := by
  sorry

end NUMINAMATH_CALUDE_skew_lines_distance_and_angle_l1503_150337


namespace NUMINAMATH_CALUDE_log_N_between_consecutive_integers_l1503_150313

theorem log_N_between_consecutive_integers 
  (N : ℝ) 
  (h : Real.log 2500 < Real.log N ∧ Real.log N < Real.log 10000) : 
  ∃ (m : ℤ), m + (m + 1) = 7 ∧ 
    (↑m : ℝ) < Real.log N ∧ Real.log N < (↑m + 1 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_log_N_between_consecutive_integers_l1503_150313


namespace NUMINAMATH_CALUDE_complex_triangle_problem_l1503_150303

theorem complex_triangle_problem (x y z : ℂ) 
  (eq1 : x^2 + y^2 + z^2 = x*y + y*z + z*x)
  (eq2 : Complex.abs (x + y + z) = 21)
  (eq3 : Complex.abs (x - y) = 2 * Real.sqrt 3)
  (eq4 : Complex.abs x = 3 * Real.sqrt 3) :
  Complex.abs y^2 + Complex.abs z^2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_complex_triangle_problem_l1503_150303


namespace NUMINAMATH_CALUDE_waiter_customers_l1503_150392

/-- Given a number of tables and the number of women and men at each table,
    calculate the total number of customers. -/
def total_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) : ℕ :=
  num_tables * (women_per_table + men_per_table)

/-- Theorem: The waiter has 90 customers in total. -/
theorem waiter_customers :
  total_customers 9 7 3 = 90 := by
  sorry

#eval total_customers 9 7 3

end NUMINAMATH_CALUDE_waiter_customers_l1503_150392


namespace NUMINAMATH_CALUDE_complex_power_sum_l1503_150390

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_sum : 2 * (i^13 + i^18 + i^23 + i^28 + i^33) = 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1503_150390


namespace NUMINAMATH_CALUDE_video_votes_l1503_150304

theorem video_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 140 ∧ like_percentage = 70 / 100 → 
  ∃ (total_votes : ℕ), 
    (like_percentage : ℚ) * total_votes - (1 - like_percentage) * total_votes = score ∧
    total_votes = 350 := by
  sorry

end NUMINAMATH_CALUDE_video_votes_l1503_150304


namespace NUMINAMATH_CALUDE_range_of_b_l1503_150383

theorem range_of_b (b : ℝ) : 
  Real.sqrt ((b - 2)^2) = 2 - b ↔ b ∈ Set.Iic 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_l1503_150383


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1503_150308

theorem diophantine_equation_solution (x y z : ℤ) :
  x^2 + y^2 + z^2 - x*y - y*z - z*x = 3 ↔ 
  (∃ k : ℤ, (x = k + 2 ∧ y = k + 1 ∧ z = k) ∨ (x = k - 2 ∧ y = k - 1 ∧ z = k)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1503_150308


namespace NUMINAMATH_CALUDE_stacy_berries_l1503_150367

theorem stacy_berries (total : ℕ) (stacy steve skylar : ℕ) : 
  total = 1100 →
  stacy = 4 * steve →
  steve = 2 * skylar →
  total = stacy + steve + skylar →
  stacy = 800 := by
sorry

end NUMINAMATH_CALUDE_stacy_berries_l1503_150367


namespace NUMINAMATH_CALUDE_range_of_a_l1503_150301

theorem range_of_a (a : ℝ) : 
  (∀ x θ : ℝ, θ ∈ Set.Icc 0 (Real.pi / 2) → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1503_150301


namespace NUMINAMATH_CALUDE_unique_triangle_arrangement_l1503_150326

/-- Represents the arrangement of numbers in the triangle --/
structure TriangleArrangement where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Checks if the given arrangement is valid according to the problem conditions --/
def is_valid_arrangement (arr : TriangleArrangement) : Prop :=
  -- All numbers are between 6 and 9
  (arr.A ≥ 6 ∧ arr.A ≤ 9) ∧
  (arr.B ≥ 6 ∧ arr.B ≤ 9) ∧
  (arr.C ≥ 6 ∧ arr.C ≤ 9) ∧
  (arr.D ≥ 6 ∧ arr.D ≤ 9) ∧
  -- All numbers are different
  arr.A ≠ arr.B ∧ arr.A ≠ arr.C ∧ arr.A ≠ arr.D ∧
  arr.B ≠ arr.C ∧ arr.B ≠ arr.D ∧
  arr.C ≠ arr.D ∧
  -- Sum of numbers on each side is equal
  arr.A + arr.C + 3 + 4 = 5 + arr.D + 2 + 4 ∧
  5 + 1 + arr.B + arr.A = 5 + arr.D + 2 + 4 ∧
  arr.A + arr.C + 3 + 4 = 5 + 1 + arr.B + arr.A

theorem unique_triangle_arrangement :
  ∃! arr : TriangleArrangement, is_valid_arrangement arr ∧
    arr.A = 6 ∧ arr.B = 8 ∧ arr.C = 7 ∧ arr.D = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_triangle_arrangement_l1503_150326


namespace NUMINAMATH_CALUDE_orange_juice_serving_size_l1503_150365

/-- Calculates the size of each orange juice serving given the concentrate and water ratio,
    number of servings, and number and size of concentrate cans. -/
theorem orange_juice_serving_size
  (concentrate_to_water_ratio : ℚ)
  (num_servings : ℕ)
  (num_concentrate_cans : ℕ)
  (concentrate_can_size : ℚ)
  (h1 : concentrate_to_water_ratio = 1 / 4)
  (h2 : num_servings = 320)
  (h3 : num_concentrate_cans = 40)
  (h4 : concentrate_can_size = 12) :
  (num_concentrate_cans * concentrate_can_size) / (concentrate_to_water_ratio * num_servings) = 6 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_serving_size_l1503_150365


namespace NUMINAMATH_CALUDE_choose_four_from_nine_l1503_150332

theorem choose_four_from_nine : Nat.choose 9 4 = 126 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_nine_l1503_150332


namespace NUMINAMATH_CALUDE_range_of_a_l1503_150399

theorem range_of_a (a x : ℝ) : 
  (∀ x, (a - 4 < x ∧ x < a + 4) → (x - 2) * (x - 3) > 0) →
  (a ≤ -2 ∨ a ≥ 7) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1503_150399


namespace NUMINAMATH_CALUDE_max_triangle_area_in_circle_l1503_150362

/-- Given a circle with center C and radius r, and a chord AB that intersects
    the circle at points A and B, forming a triangle ABC. The central angle
    subtended by chord AB is α. -/
theorem max_triangle_area_in_circle (r : ℝ) (α : ℝ) (h : 0 < r) :
  let area := (1/2) * r^2 * Real.sin α
  (∀ θ, 0 ≤ θ ∧ θ ≤ π → area ≥ (1/2) * r^2 * Real.sin θ) ↔ α = π/2 ∧ 
  let chord_length := 2 * r * Real.sin (α/2)
  chord_length = r * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_triangle_area_in_circle_l1503_150362


namespace NUMINAMATH_CALUDE_cubic_function_identity_l1503_150386

/-- Given a cubic function g(x) = px³ + qx² + rx + s where g(3) = 4,
    prove that 6p - 3q + r - 2s = 60p + 15q + 7r - 8 -/
theorem cubic_function_identity (p q r s : ℝ) 
  (h : 27 * p + 9 * q + 3 * r + s = 4) :
  6 * p - 3 * q + r - 2 * s = 60 * p + 15 * q + 7 * r - 8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_identity_l1503_150386


namespace NUMINAMATH_CALUDE_sequence_product_l1503_150372

/-- An arithmetic sequence with first term -9 and last term -1 -/
def arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ d : ℝ, a₁ = -9 + d ∧ a₂ = a₁ + d ∧ -1 = a₂ + d

/-- A geometric sequence with first term -9 and last term -1 -/
def geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ r : ℝ, b₁ = -9 * r ∧ b₂ = b₁ * r ∧ b₃ = b₂ * r ∧ -1 = b₃ * r

theorem sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) 
  (h₁ : arithmetic_sequence a₁ a₂)
  (h₂ : geometric_sequence b₁ b₂ b₃) :
  b₂ * (a₂ - a₁) = -8 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l1503_150372


namespace NUMINAMATH_CALUDE_jellybean_ratio_l1503_150388

/-- Proves that the ratio of Sophie's jellybeans to Caleb's jellybeans is 1:2 -/
theorem jellybean_ratio (caleb_dozens : ℕ) (total : ℕ) : 
  caleb_dozens = 3 → total = 54 → 
  (total - caleb_dozens * 12) / (caleb_dozens * 12) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_jellybean_ratio_l1503_150388


namespace NUMINAMATH_CALUDE_xy_coefficient_zero_l1503_150335

theorem xy_coefficient_zero (k : ℚ) (x y : ℚ) :
  k = 1 / 3 → -3 * k + 1 = 0 :=
by
  sorry

#check xy_coefficient_zero

end NUMINAMATH_CALUDE_xy_coefficient_zero_l1503_150335


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1503_150336

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y + 2) : 
  1 / x + 1 / y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1503_150336


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1503_150309

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 9*x*y - x^2 - 8*y^2 = 2005 ↔ 
  (x = 63 ∧ y = 58) ∨ (x = 459 ∧ y = 58) ∨ (x = -63 ∧ y = -58) ∨ (x = -459 ∧ y = -58) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1503_150309


namespace NUMINAMATH_CALUDE_man_upstream_speed_l1503_150325

/-- Calculates the upstream speed of a man given his downstream speed and the stream speed. -/
def upstream_speed (downstream_speed stream_speed : ℝ) : ℝ :=
  downstream_speed - 2 * stream_speed

/-- Theorem stating that given a downstream speed of 14 km/h and a stream speed of 3 km/h, 
    the upstream speed is 8 km/h. -/
theorem man_upstream_speed : 
  upstream_speed 14 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_man_upstream_speed_l1503_150325


namespace NUMINAMATH_CALUDE_no_valid_n_l1503_150318

theorem no_valid_n : ¬ ∃ (n : ℕ), n > 0 ∧ 
  (3*n - 3 + 2*n + 7 > 4*n + 6) ∧
  (3*n - 3 + 4*n + 6 > 2*n + 7) ∧
  (2*n + 7 + 4*n + 6 > 3*n - 3) ∧
  (2*n + 7 > 4*n + 6) ∧
  (4*n + 6 > 3*n - 3) :=
sorry

end NUMINAMATH_CALUDE_no_valid_n_l1503_150318


namespace NUMINAMATH_CALUDE_sequence_proof_l1503_150375

theorem sequence_proof : ∃ (a b c d : ℕ), 
  (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧
  (47 * a = 423) ∧
  (423 * b = 282 * 3) ∧
  (282 * c + c * 10 + d = 1448) ∧
  (282 * c + 348 = 1053) := by
  sorry

end NUMINAMATH_CALUDE_sequence_proof_l1503_150375


namespace NUMINAMATH_CALUDE_cab_driver_income_l1503_150340

theorem cab_driver_income (day1 day3 day4 day5 average : ℕ) 
  (h1 : day1 = 300)
  (h2 : day3 = 750)
  (h3 : day4 = 200)
  (h4 : day5 = 600)
  (h5 : average = 400)
  (h6 : (day1 + day3 + day4 + day5 + (5 * average - (day1 + day3 + day4 + day5))) / 5 = average) :
  5 * average - (day1 + day3 + day4 + day5) = 150 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l1503_150340


namespace NUMINAMATH_CALUDE_total_pure_acid_l1503_150306

theorem total_pure_acid (solution1_volume : Real) (solution1_concentration : Real)
                        (solution2_volume : Real) (solution2_concentration : Real)
                        (solution3_volume : Real) (solution3_concentration : Real) :
  solution1_volume = 6 →
  solution1_concentration = 0.40 →
  solution2_volume = 4 →
  solution2_concentration = 0.35 →
  solution3_volume = 3 →
  solution3_concentration = 0.55 →
  solution1_volume * solution1_concentration +
  solution2_volume * solution2_concentration +
  solution3_volume * solution3_concentration = 5.45 := by
sorry

end NUMINAMATH_CALUDE_total_pure_acid_l1503_150306


namespace NUMINAMATH_CALUDE_mike_changed_64_tires_l1503_150352

/-- The number of tires changed by Mike -/
def total_tires_changed (num_motorcycles num_cars tires_per_motorcycle tires_per_car : ℕ) : ℕ :=
  num_motorcycles * tires_per_motorcycle + num_cars * tires_per_car

/-- Theorem stating that Mike changed 64 tires in total -/
theorem mike_changed_64_tires :
  let num_motorcycles : ℕ := 12
  let num_cars : ℕ := 10
  let tires_per_motorcycle : ℕ := 2
  let tires_per_car : ℕ := 4
  total_tires_changed num_motorcycles num_cars tires_per_motorcycle tires_per_car = 64 := by
  sorry

end NUMINAMATH_CALUDE_mike_changed_64_tires_l1503_150352


namespace NUMINAMATH_CALUDE_logical_equivalence_l1503_150312

theorem logical_equivalence (S X Y : Prop) :
  (S → ¬X ∧ ¬Y) ↔ (X ∨ Y → ¬S) := by
  sorry

end NUMINAMATH_CALUDE_logical_equivalence_l1503_150312


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1503_150395

/-- Given a hyperbola and a line intersecting it, proves that the eccentricity is √2 under specific conditions -/
theorem hyperbola_eccentricity (a b k m : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (A B N : ℝ × ℝ), 
    -- The line y = kx + m intersects the hyperbola at A and B
    (A.1^2 / a^2 - A.2^2 / b^2 = 1 ∧ A.2 = k * A.1 + m) ∧
    (B.1^2 / a^2 - B.2^2 / b^2 = 1 ∧ B.2 = k * B.1 + m) ∧
    -- A and B are where the asymptotes intersect the line
    (A.2 = -b/a * A.1 ∨ A.2 = b/a * A.1) ∧
    (B.2 = -b/a * B.1 ∨ B.2 = b/a * B.1) ∧
    -- N is on both lines
    (N.2 = k * N.1 + m) ∧
    (N.2 = 1/k * N.1) ∧
    -- N is the midpoint of AB
    (N.1 = (A.1 + B.1) / 2 ∧ N.2 = (A.2 + B.2) / 2)) →
  -- The eccentricity of the hyperbola is √2
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1503_150395


namespace NUMINAMATH_CALUDE_residue_modulo_17_l1503_150344

theorem residue_modulo_17 : (101 * 15 - 7 * 9 + 5) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_residue_modulo_17_l1503_150344


namespace NUMINAMATH_CALUDE_remainder_sum_mod_eight_l1503_150307

theorem remainder_sum_mod_eight (a b c : ℕ) : 
  a < 8 → b < 8 → c < 8 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 8 = 1 →
  (7 * c) % 8 = 3 →
  (5 * b) % 8 = (4 + b) % 8 →
  (a + b + c) % 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_eight_l1503_150307


namespace NUMINAMATH_CALUDE_system_solutions_l1503_150314

/-- The system of equations has only two solutions -/
theorem system_solutions :
  ∀ x y z : ℝ,
  (x + y * z = 2 ∧ y + x * z = 2 ∧ z + x * y = 2) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -2 ∧ y = -2 ∧ z = -2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l1503_150314


namespace NUMINAMATH_CALUDE_balloon_count_sum_l1503_150338

/-- The number of yellow balloons Fred has -/
def fred_balloons : ℕ := 5

/-- The number of yellow balloons Sam has -/
def sam_balloons : ℕ := 6

/-- The number of yellow balloons Mary has -/
def mary_balloons : ℕ := 7

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 18

/-- Theorem stating that the sum of individual balloon counts equals the total -/
theorem balloon_count_sum :
  fred_balloons + sam_balloons + mary_balloons = total_balloons := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_sum_l1503_150338


namespace NUMINAMATH_CALUDE_angle_sum_is_420_l1503_150378

/-- A geometric configuration with six angles A, B, C, D, E, and F -/
structure GeometricConfig where
  A : Real
  B : Real
  C : Real
  D : Real
  E : Real
  F : Real

/-- The theorem stating that if angle E is 30 degrees, then the sum of all angles is 420 degrees -/
theorem angle_sum_is_420 (config : GeometricConfig) (h : config.E = 30) :
  config.A + config.B + config.C + config.D + config.E + config.F = 420 := by
  sorry

#check angle_sum_is_420

end NUMINAMATH_CALUDE_angle_sum_is_420_l1503_150378


namespace NUMINAMATH_CALUDE_initial_machines_count_l1503_150361

/-- The number of pens produced by a group of machines in a given time -/
structure PenProduction where
  machines : ℕ
  pens : ℕ
  minutes : ℕ

/-- The rate of pen production per minute for a given number of machines -/
def production_rate (p : PenProduction) : ℚ :=
  p.pens / (p.machines * p.minutes)

theorem initial_machines_count (total_rate : ℕ) (sample : PenProduction) :
  sample.machines * total_rate = sample.pens * production_rate sample →
  total_rate = 240 →
  sample = { machines := 5, pens := 750, minutes := 5 } →
  ∃ n : ℕ, n * (production_rate sample) = total_rate ∧ n = 8 := by
sorry

end NUMINAMATH_CALUDE_initial_machines_count_l1503_150361


namespace NUMINAMATH_CALUDE_ploughing_time_l1503_150315

/-- Proves that if R and S together can plough a field in 10 hours, and R alone requires 15 hours,
    then S alone requires 30 hours to plough the same field. -/
theorem ploughing_time (r s : ℝ) : 
  (1 / r + 1 / s = 1 / 10) →   -- R and S together take 10 hours
  (1 / r = 1 / 15) →           -- R alone takes 15 hours
  (1 / s = 1 / 30) :=          -- S alone takes 30 hours
by sorry

end NUMINAMATH_CALUDE_ploughing_time_l1503_150315


namespace NUMINAMATH_CALUDE_infinitely_many_non_square_plus_prime_numbers_l1503_150355

theorem infinitely_many_non_square_plus_prime_numbers :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, ¬∃ (m : ℤ) (p : ℕ), Nat.Prime p ∧ n = m^2 + p := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_square_plus_prime_numbers_l1503_150355


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_l1503_150368

/-- The capacity of a small bottle in milliliters -/
def small_bottle_capacity : ℝ := 35

/-- The capacity of a large bottle in milliliters -/
def large_bottle_capacity : ℝ := 500

/-- The minimum number of small bottles needed to completely fill a large bottle -/
def min_bottles : ℕ := 15

theorem min_bottles_to_fill :
  ∃ (n : ℕ), n * small_bottle_capacity ≥ large_bottle_capacity ∧
  ∀ (m : ℕ), m * small_bottle_capacity ≥ large_bottle_capacity → n ≤ m ∧
  n = min_bottles :=
by sorry

end NUMINAMATH_CALUDE_min_bottles_to_fill_l1503_150368


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1503_150300

/-- Given a quadratic function f(x) = -x^2 + 4x + a on the interval [0, 1] 
    with a minimum value of -2, prove that its maximum value is 1. -/
theorem quadratic_max_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = -x^2 + 4*x + a) →
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f x ≤ f y) →
  (∃ x ∈ Set.Icc 0 1, f x = -2) →
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f y ≤ f x) →
  (∃ x ∈ Set.Icc 0 1, f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1503_150300


namespace NUMINAMATH_CALUDE_tobias_driveways_shoveled_l1503_150373

/-- Calculates the number of driveways Tobias shoveled given his earnings and expenses. -/
theorem tobias_driveways_shoveled (shoe_cost allowance_per_month lawn_mowing_charge driveway_shoveling_charge change_after_purchase : ℕ) (months_saved lawns_mowed : ℕ) : 
  shoe_cost = 95 →
  allowance_per_month = 5 →
  months_saved = 3 →
  lawn_mowing_charge = 15 →
  driveway_shoveling_charge = 7 →
  change_after_purchase = 15 →
  lawns_mowed = 4 →
  (shoe_cost + change_after_purchase - months_saved * allowance_per_month - lawns_mowed * lawn_mowing_charge) / driveway_shoveling_charge = 5 := by
sorry

end NUMINAMATH_CALUDE_tobias_driveways_shoveled_l1503_150373


namespace NUMINAMATH_CALUDE_rabbit_area_l1503_150342

theorem rabbit_area (ear_area : ℝ) (total_area : ℝ) : 
  ear_area = 10 → ear_area = (1/8) * total_area → total_area = 80 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_area_l1503_150342


namespace NUMINAMATH_CALUDE_five_number_average_l1503_150353

theorem five_number_average (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a + b + c = 48 →
  a = 2 * b →
  (d + e) / 2 = 26 := by
sorry

end NUMINAMATH_CALUDE_five_number_average_l1503_150353


namespace NUMINAMATH_CALUDE_quilt_patch_cost_is_450_l1503_150323

/-- Calculates the total cost of patches for a quilt with given dimensions and patch pricing. -/
def quilt_patch_cost (quilt_length : ℕ) (quilt_width : ℕ) (patch_area : ℕ) 
                     (initial_patch_cost : ℕ) (initial_patch_count : ℕ) : ℕ :=
  let total_area := quilt_length * quilt_width
  let total_patches := total_area / patch_area
  let initial_cost := initial_patch_count * initial_patch_cost
  let remaining_patches := total_patches - initial_patch_count
  let remaining_cost := remaining_patches * (initial_patch_cost / 2)
  initial_cost + remaining_cost

/-- The total cost of patches for a 16-foot by 20-foot quilt with specified patch pricing is $450. -/
theorem quilt_patch_cost_is_450 : 
  quilt_patch_cost 16 20 4 10 10 = 450 := by
  sorry

end NUMINAMATH_CALUDE_quilt_patch_cost_is_450_l1503_150323


namespace NUMINAMATH_CALUDE_cecilia_B_count_l1503_150350

/-- The number of students who received a 'B' in Mrs. Cecilia's class -/
def students_with_B_cecilia (jacob_total : ℕ) (jacob_B : ℕ) (cecilia_total : ℕ) (cecilia_absent : ℕ) : ℕ :=
  let jacob_proportion : ℚ := jacob_B / jacob_total
  let cecilia_present : ℕ := cecilia_total - cecilia_absent
  ⌊(jacob_proportion * cecilia_present : ℚ)⌋₊

theorem cecilia_B_count :
  students_with_B_cecilia 20 12 30 6 = 14 :=
by sorry

end NUMINAMATH_CALUDE_cecilia_B_count_l1503_150350


namespace NUMINAMATH_CALUDE_inequality_holds_l1503_150376

-- Define an even function that is increasing on (-∞, 0]
def EvenIncreasingNegative (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

-- State the theorem
theorem inequality_holds (f : ℝ → ℝ) (h : EvenIncreasingNegative f) :
  ∀ a : ℝ, f 1 > f (a^2 + 2*a + 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_l1503_150376


namespace NUMINAMATH_CALUDE_max_profit_at_max_price_max_profit_value_mall_sale_max_profit_l1503_150320

/-- Represents the shopping mall's clothing sale scenario -/
structure ClothingSale where
  cost : ℝ
  sales_function : ℝ → ℝ
  profit_function : ℝ → ℝ
  min_price : ℝ
  max_price : ℝ

/-- The specific clothing sale scenario as described in the problem -/
def mall_sale : ClothingSale :=
  { cost := 60
  , sales_function := λ x => -x + 120
  , profit_function := λ x => (x - 60) * (-x + 120)
  , min_price := 60
  , max_price := 84
  }

/-- Theorem stating that the maximum profit is achieved at the highest allowed price -/
theorem max_profit_at_max_price (sale : ClothingSale) :
  ∀ x ∈ Set.Icc sale.min_price sale.max_price,
    sale.profit_function x ≤ sale.profit_function sale.max_price :=
sorry

/-- Theorem stating that the maximum profit is 864 dollars -/
theorem max_profit_value (sale : ClothingSale) :
  sale.profit_function sale.max_price = 864 :=
sorry

/-- Main theorem combining the above results -/
theorem mall_sale_max_profit :
  ∃ x ∈ Set.Icc mall_sale.min_price mall_sale.max_price,
    mall_sale.profit_function x = 864 ∧
    ∀ y ∈ Set.Icc mall_sale.min_price mall_sale.max_price,
      mall_sale.profit_function y ≤ 864 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_max_price_max_profit_value_mall_sale_max_profit_l1503_150320


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_k_range_l1503_150396

theorem empty_solution_set_implies_k_range (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 ≥ 0) → k ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_k_range_l1503_150396


namespace NUMINAMATH_CALUDE_surfers_count_l1503_150330

/-- The number of surfers on Santa Monica beach -/
def santa_monica_surfers : ℕ := 20

/-- The number of surfers on Malibu beach -/
def malibu_surfers : ℕ := 2 * santa_monica_surfers

/-- The total number of surfers on both beaches -/
def total_surfers : ℕ := malibu_surfers + santa_monica_surfers

theorem surfers_count : total_surfers = 60 := by
  sorry

end NUMINAMATH_CALUDE_surfers_count_l1503_150330


namespace NUMINAMATH_CALUDE_median_is_212_l1503_150374

/-- The sum of integers from 1 to n -/
def triangularSum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total count of numbers in our special list up to n -/
def cumulativeCount (n : ℕ) : ℕ := triangularSum n

/-- The total length of our special list -/
def totalLength : ℕ := triangularSum 300

/-- The position of the lower median element -/
def lowerMedianPos : ℕ := totalLength / 2

/-- The position of the upper median element -/
def upperMedianPos : ℕ := lowerMedianPos + 1

theorem median_is_212 : 
  ∃ (n : ℕ), n = 212 ∧ 
  cumulativeCount (n - 1) < lowerMedianPos ∧
  cumulativeCount n ≥ upperMedianPos :=
sorry

end NUMINAMATH_CALUDE_median_is_212_l1503_150374


namespace NUMINAMATH_CALUDE_exists_quadratic_sequence_l1503_150370

/-- A quadratic sequence is a finite sequence of integers where the absolute difference
    between consecutive terms is equal to the square of their position. -/
def IsQuadraticSequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i ≤ n → |a i - a (i - 1)| = i^2

/-- For any two integers, there exists a quadratic sequence connecting them. -/
theorem exists_quadratic_sequence (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ IsQuadraticSequence a n :=
sorry

end NUMINAMATH_CALUDE_exists_quadratic_sequence_l1503_150370


namespace NUMINAMATH_CALUDE_percent_students_in_school_l1503_150393

/-- Given that 40% of students are learning from home and the remaining students are equally divided
    into two groups with only one group attending school on any day, prove that the percent of
    students present in school is 30%. -/
theorem percent_students_in_school :
  let total_percent : ℚ := 100
  let home_percent : ℚ := 40
  let remaining_percent : ℚ := total_percent - home_percent
  let in_school_percent : ℚ := remaining_percent / 2
  in_school_percent = 30 := by sorry

end NUMINAMATH_CALUDE_percent_students_in_school_l1503_150393


namespace NUMINAMATH_CALUDE_string_length_problem_l1503_150319

theorem string_length_problem (total_strings : ℕ) (avg_length : ℝ) (other_strings : ℕ) (other_avg : ℝ) :
  total_strings = 6 →
  avg_length = 80 →
  other_strings = 4 →
  other_avg = 85 →
  let remaining_strings := total_strings - other_strings
  let total_length := avg_length * total_strings
  let other_length := other_avg * other_strings
  let remaining_length := total_length - other_length
  remaining_length / remaining_strings = 70 := by
sorry

end NUMINAMATH_CALUDE_string_length_problem_l1503_150319


namespace NUMINAMATH_CALUDE_cody_grandmother_age_l1503_150346

/-- Given that Cody is 14 years old and his grandmother is 6 times as old as he is,
    prove that Cody's grandmother is 84 years old. -/
theorem cody_grandmother_age (cody_age : ℕ) (grandmother_age_ratio : ℕ) 
  (h1 : cody_age = 14)
  (h2 : grandmother_age_ratio = 6) :
  cody_age * grandmother_age_ratio = 84 := by
  sorry

end NUMINAMATH_CALUDE_cody_grandmother_age_l1503_150346


namespace NUMINAMATH_CALUDE_average_speed_inequality_l1503_150339

theorem average_speed_inequality (a b v : ℝ) (hab : a < b) (hv : v = (2 * a * b) / (a + b)) : 
  a < v ∧ v < Real.sqrt (a * b) := by sorry

end NUMINAMATH_CALUDE_average_speed_inequality_l1503_150339


namespace NUMINAMATH_CALUDE_smallest_input_129_l1503_150359

def f (n : ℕ+) : ℕ := 9 * n.val + 120

theorem smallest_input_129 :
  ∀ m : ℕ+, f m ≥ f 129 → m ≥ 129 :=
sorry

end NUMINAMATH_CALUDE_smallest_input_129_l1503_150359


namespace NUMINAMATH_CALUDE_wade_team_score_l1503_150364

/-- Calculates the total points scored by a basketball team after a given number of games -/
def team_total_points (wade_avg : ℕ) (teammates_avg : ℕ) (games : ℕ) : ℕ :=
  (wade_avg + teammates_avg) * games

/-- Proves that Wade's team scores 300 points in 5 games -/
theorem wade_team_score : team_total_points 20 40 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_wade_team_score_l1503_150364


namespace NUMINAMATH_CALUDE_tina_brownies_l1503_150391

theorem tina_brownies (total_brownies : ℕ) (days : ℕ) (husband_daily : ℕ) (shared_guests : ℕ) (leftover : ℕ) :
  total_brownies = 24 →
  days = 5 →
  husband_daily = 1 →
  shared_guests = 4 →
  leftover = 5 →
  (total_brownies - (days * husband_daily + shared_guests + leftover)) / (days * 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tina_brownies_l1503_150391


namespace NUMINAMATH_CALUDE_second_rectangle_width_l1503_150363

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Theorem: Given two rectangles with specified properties, the width of the second rectangle is 3 inches -/
theorem second_rectangle_width (r1 r2 : Rectangle) : 
  r1.width = 4 → 
  r1.height = 5 → 
  r2.height = 6 → 
  area r1 = area r2 + 2 → 
  r2.width = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_rectangle_width_l1503_150363


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1503_150381

theorem trigonometric_identity (α : ℝ) :
  (2 * (Real.cos ((9 / 4) * π - α))^2) / (1 + Real.cos ((π / 2) + 2 * α)) -
  (Real.sin (α + (7 / 4) * π) / Real.sin (α + (π / 4))) *
  (1 / Real.tan ((3 / 4) * π - α)) =
  (4 * Real.sin (2 * α)) / (Real.cos (2 * α))^2 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1503_150381


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inverse_sqrt_x_l1503_150351

theorem sqrt_x_plus_inverse_sqrt_x (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inverse_sqrt_x_l1503_150351


namespace NUMINAMATH_CALUDE_remainder_problem_l1503_150387

theorem remainder_problem : 7 * 12^24 + 3^24 ≡ 0 [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1503_150387


namespace NUMINAMATH_CALUDE_total_coin_value_l1503_150316

-- Define the number of rolls for each coin type
def quarters_rolls : ℕ := 5
def dimes_rolls : ℕ := 4
def nickels_rolls : ℕ := 3
def pennies_rolls : ℕ := 2

-- Define the number of coins in each roll
def quarters_per_roll : ℕ := 40
def dimes_per_roll : ℕ := 50
def nickels_per_roll : ℕ := 40
def pennies_per_roll : ℕ := 50

-- Define the value of each coin in cents
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def penny_value : ℕ := 1

-- Calculate the total value in cents
def total_value : ℕ :=
  quarters_rolls * quarters_per_roll * quarter_value +
  dimes_rolls * dimes_per_roll * dime_value +
  nickels_rolls * nickels_per_roll * nickel_value +
  pennies_rolls * pennies_per_roll * penny_value

-- Theorem to prove
theorem total_coin_value : total_value = 7700 := by
  sorry

end NUMINAMATH_CALUDE_total_coin_value_l1503_150316


namespace NUMINAMATH_CALUDE_root_of_polynomial_l1503_150345

theorem root_of_polynomial : ∃ (x : ℝ), x^3 = 5 ∧ x^6 - 6*x^4 - 10*x^3 - 60*x + 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_of_polynomial_l1503_150345


namespace NUMINAMATH_CALUDE_problem_solution_l1503_150380

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |3*x - 2|

-- Define the solution set of f(x) ≤ 5
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | -4*a/5 ≤ x ∧ x ≤ 3*a/5}

-- State the theorem
theorem problem_solution (a b : ℝ) : 
  (∀ x : ℝ, f x ≤ 5 ↔ x ∈ solution_set a) →
  (a = 1 ∧ b = 2) ∧
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ (m : ℝ)^2 - 3*m) ∧
  (∃ m : ℝ, m = (3 + Real.sqrt 21) / 2 ∧
    ∀ m' : ℝ, (∀ x : ℝ, |x - 1| + |x + 2| ≥ m'^2 - 3*m') → m' ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1503_150380


namespace NUMINAMATH_CALUDE_eight_by_ten_grid_theorem_l1503_150347

/-- Represents a rectangular grid -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the number of squares not intersected by diagonals in a grid -/
def squares_not_intersected (g : Grid) : ℕ :=
  sorry

/-- Theorem: In an 8 × 10 grid, 48 squares are not intersected by either diagonal -/
theorem eight_by_ten_grid_theorem : 
  let g : Grid := { rows := 8, cols := 10 }
  squares_not_intersected g = 48 := by
  sorry

end NUMINAMATH_CALUDE_eight_by_ten_grid_theorem_l1503_150347


namespace NUMINAMATH_CALUDE_subtracted_number_l1503_150305

theorem subtracted_number (x y : ℤ) (h1 : x = 127) (h2 : 2 * x - y = 102) : y = 152 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l1503_150305


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1503_150382

theorem sqrt_inequality (x : ℝ) : 
  3 * x - 2 ≥ 0 → (|Real.sqrt (3 * x - 2) - 3| > 1 ↔ x > 6 ∨ (2/3 ≤ x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1503_150382


namespace NUMINAMATH_CALUDE_angela_unfinished_problems_l1503_150341

theorem angela_unfinished_problems (total : Nat) (martha : Nat) (jenna : Nat) (mark : Nat)
  (h1 : total = 20)
  (h2 : martha = 2)
  (h3 : jenna = 4 * martha - 2)
  (h4 : mark = jenna / 2)
  (h5 : martha + jenna + mark ≤ total) :
  total - (martha + jenna + mark) = 9 := by
sorry

end NUMINAMATH_CALUDE_angela_unfinished_problems_l1503_150341


namespace NUMINAMATH_CALUDE_quadratic_vertex_l1503_150348

/-- The quadratic function f(x) = 2(x-3)^2 + 1 has its vertex at (3, 1). -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * (x - 3)^2 + 1
  (∀ x, f x ≥ f 3) ∧ f 3 = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l1503_150348


namespace NUMINAMATH_CALUDE_julia_cakes_l1503_150322

theorem julia_cakes (x : ℕ) : 
  (x * 6 - 3 = 21) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_julia_cakes_l1503_150322


namespace NUMINAMATH_CALUDE_solution_value_l1503_150358

theorem solution_value (a : ℝ) (h : 3 * a^2 + 2 * a - 1 = 0) : 
  3 * a^2 + 2 * a - 2019 = -2018 := by
sorry

end NUMINAMATH_CALUDE_solution_value_l1503_150358


namespace NUMINAMATH_CALUDE_percent_relation_l1503_150398

theorem percent_relation (x y z : ℝ) 
  (h1 : x = 1.20 * y) 
  (h2 : y = 0.30 * z) : 
  x = 0.36 * z := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l1503_150398


namespace NUMINAMATH_CALUDE_existence_of_multiple_factorizations_l1503_150356

def V_n (n : ℕ) := {m : ℕ | ∃ k : ℕ, k ≥ 1 ∧ m = 1 + k * n}

def irreducible_in_V_n (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → p * q ≠ m

theorem existence_of_multiple_factorizations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ (factors1 factors2 : List ℕ),
      factors1 ≠ factors2 ∧
      (∀ f ∈ factors1, irreducible_in_V_n n f) ∧
      (∀ f ∈ factors2, irreducible_in_V_n n f) ∧
      r = factors1.prod ∧
      r = factors2.prod :=
  sorry

end NUMINAMATH_CALUDE_existence_of_multiple_factorizations_l1503_150356


namespace NUMINAMATH_CALUDE_event_probability_l1503_150384

theorem event_probability (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^4 = 65/81) →
  p = 1/3 := by sorry

end NUMINAMATH_CALUDE_event_probability_l1503_150384


namespace NUMINAMATH_CALUDE_fencing_required_l1503_150349

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : 
  area > 0 → uncovered_side > 0 → area = uncovered_side * (area / uncovered_side) →
  2 * (area / uncovered_side) + uncovered_side = 32 := by
  sorry

#check fencing_required 120 20

end NUMINAMATH_CALUDE_fencing_required_l1503_150349


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l1503_150394

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/3 + 1/2, 1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/9, 1/3 + 1/10]
  (∀ x ∈ sums, x ≤ 1/3 + 1/2) ∧ (1/3 + 1/2 = 5/6) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l1503_150394


namespace NUMINAMATH_CALUDE_car_city_efficiency_approx_36_l1503_150385

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  highway : ℝ  -- Miles per gallon on highway
  city : ℝ     -- Miles per gallon in city
  tank_size : ℝ -- Size of the fuel tank in gallons

/-- Conditions for the car's fuel efficiency -/
def efficiency_conditions (c : CarFuelEfficiency) : Prop :=
  c.highway * c.tank_size = 690 ∧
  c.city * c.tank_size = 420 ∧
  c.city = c.highway - 23

/-- Theorem stating that under given conditions, the car's city fuel efficiency is approximately 36 mpg -/
theorem car_city_efficiency_approx_36 (c : CarFuelEfficiency) 
  (h : efficiency_conditions c) : 
  ∃ ε > 0, |c.city - 36| < ε :=
sorry

end NUMINAMATH_CALUDE_car_city_efficiency_approx_36_l1503_150385


namespace NUMINAMATH_CALUDE_cover_properties_l1503_150321

-- Define a type for points in the plane
variable {Point : Type}

-- Define a type for sets of points
variable {Set : Type}

-- Define the cover operation
variable (cover : Set → Set)

-- Define the subset relation
variable (subset : Set → Set → Prop)

-- Define the union operation
variable (union : Set → Set → Set)

-- Axiom for the given condition
axiom cover_union_superset (X Y : Set) :
  subset (union (cover (union X Y)) Y) (union (union (cover (cover X)) (cover Y)) Y)

-- Statement to prove
theorem cover_properties (X Y : Set) :
  (subset X (cover X)) ∧ 
  (cover (cover X) = cover X) ∧
  (subset X Y → subset (cover X) (cover Y)) :=
sorry

end NUMINAMATH_CALUDE_cover_properties_l1503_150321


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1503_150302

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_ones : ones < 10

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Checks if all digits in a three-digit number are the same -/
def allDigitsSame (n : Nat) : Prop :=
  (n / 100 = (n / 10) % 10) ∧ ((n / 10) % 10 = n % 10)

theorem unique_three_digit_number :
  ∃! (n : ThreeDigitNumber),
    (n.hundreds + n.ones = 5) ∧
    (n.tens = 3) ∧
    (n.hundreds ≠ n.tens) ∧
    (n.tens ≠ n.ones) ∧
    (n.hundreds ≠ n.ones) ∧
    allDigitsSame (n.toNat + 124) ∧
    n.toNat = 431 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1503_150302


namespace NUMINAMATH_CALUDE_solution_characterization_l1503_150354

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(4/3, 4/3, -5/3), (4/3, -5/3, 4/3), (-5/3, 4/3, 4/3),
   (-4/3, -4/3, 5/3), (-4/3, 5/3, -4/3), (5/3, -4/3, -4/3)}

def satisfies_equations (x y z : ℝ) : Prop :=
  x^2 - y*z = |y - z| + 1 ∧
  y^2 - z*x = |z - x| + 1 ∧
  z^2 - x*y = |x - y| + 1

theorem solution_characterization :
  {p : ℝ × ℝ × ℝ | satisfies_equations p.1 p.2.1 p.2.2} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l1503_150354


namespace NUMINAMATH_CALUDE_square_sum_difference_l1503_150324

theorem square_sum_difference : 3^2 + 7^2 - 5^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_difference_l1503_150324


namespace NUMINAMATH_CALUDE_water_volume_in_cone_l1503_150397

/-- The volume of water remaining in a conical container after pouring from a cylindrical container -/
theorem water_volume_in_cone (base_radius : ℝ) (height : ℝ) (overflow_volume : ℝ) :
  base_radius > 0 ∧ height > 0 ∧ overflow_volume = 36.2 →
  let cone_volume := (1 / 3) * Real.pi * base_radius^2 * height
  let cylinder_volume := Real.pi * base_radius^2 * height
  overflow_volume = 2 / 3 * cylinder_volume →
  cone_volume = 18.1 := by
sorry

end NUMINAMATH_CALUDE_water_volume_in_cone_l1503_150397


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1503_150329

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fourth_term_of_geometric_sequence
  (a : ℕ → ℕ)
  (h_geometric : GeometricSequence a)
  (h_first : a 1 = 5)
  (h_fifth : a 5 = 1280) :
  a 4 = 320 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1503_150329


namespace NUMINAMATH_CALUDE_allowance_increase_l1503_150317

/-- The base amount of Kathleen's middle school allowance -/
def base_amount : ℝ := 8

/-- Kathleen's middle school allowance -/
def middle_school_allowance (x : ℝ) : ℝ := x + 2

/-- Kathleen's senior year allowance -/
def senior_year_allowance (x : ℝ) : ℝ := 5 + 2 * (x + 2)

/-- The percentage increase in Kathleen's weekly allowance -/
def percentage_increase : ℝ := 150

theorem allowance_increase (x : ℝ) :
  x = base_amount ↔
  (1 + percentage_increase / 100) * middle_school_allowance x = senior_year_allowance x :=
sorry

end NUMINAMATH_CALUDE_allowance_increase_l1503_150317


namespace NUMINAMATH_CALUDE_diamond_club_evaluation_l1503_150379

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := (3 * a + b) / (a - b)

-- Define the club operation
def club (a b : ℚ) : ℚ := 2

-- Theorem statement
theorem diamond_club_evaluation :
  club (diamond 4 6) (diamond 7 5) = 2 := by sorry

end NUMINAMATH_CALUDE_diamond_club_evaluation_l1503_150379


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l1503_150334

theorem right_triangle_inequality (a b c m : ℝ) 
  (h1 : c^2 = a^2 + b^2) 
  (h2 : a * b = c * m) 
  (h3 : m > 0) : 
  m + c > a + b := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l1503_150334


namespace NUMINAMATH_CALUDE_soda_cost_l1503_150311

theorem soda_cost (burger_cost soda_cost : ℕ) : 
  (4 * burger_cost + 3 * soda_cost = 440) →
  (3 * burger_cost + 2 * soda_cost = 310) →
  soda_cost = 80 := by
sorry

end NUMINAMATH_CALUDE_soda_cost_l1503_150311


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_through_point_parallel_to_line_proof_l1503_150357

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem line_through_point_parallel_to_line 
  (given_line : Line) 
  (given_point : Point) 
  (result_line : Line) : Prop :=
  given_line.a = 1 ∧ 
  given_line.b = -2 ∧ 
  given_line.c = 3 ∧
  given_point.x = -1 ∧
  given_point.y = 3 ∧
  result_line.a = 1 ∧
  result_line.b = -2 ∧
  result_line.c = 7 →
  given_point.liesOn result_line ∧
  result_line.isParallelTo given_line

-- The proof of the theorem
theorem line_through_point_parallel_to_line_proof 
  (given_line : Line) 
  (given_point : Point) 
  (result_line : Line) : 
  line_through_point_parallel_to_line given_line given_point result_line :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_through_point_parallel_to_line_proof_l1503_150357


namespace NUMINAMATH_CALUDE_square_root_of_2m_minus_n_is_2_l1503_150327

theorem square_root_of_2m_minus_n_is_2 
  (m n : ℝ) 
  (eq1 : m * 2 + n * 1 = 8) 
  (eq2 : n * 2 - m * 1 = 1) : 
  Real.sqrt (2 * m - n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_2m_minus_n_is_2_l1503_150327
