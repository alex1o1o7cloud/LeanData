import Mathlib

namespace NUMINAMATH_CALUDE_intersection_points_l1212_121266

/-- The number of intersection points for k lines in a plane -/
def f (k : ℕ) : ℕ := sorry

/-- No two lines are parallel and no three lines intersect at the same point -/
axiom line_properties (k : ℕ) : True

theorem intersection_points (k : ℕ) : f (k + 1) = f k + k :=
  sorry

end NUMINAMATH_CALUDE_intersection_points_l1212_121266


namespace NUMINAMATH_CALUDE_quadratic_equivalences_l1212_121282

theorem quadratic_equivalences (x : ℝ) : 
  (((x ≠ 1 ∧ x ≠ 2) → x^2 - 3*x + 2 ≠ 0) ∧
   ((x^2 - 3*x + 2 = 0) → (x = 1 ∨ x = 2)) ∧
   ((x = 1 ∨ x = 2) → x^2 - 3*x + 2 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equivalences_l1212_121282


namespace NUMINAMATH_CALUDE_gcd_twelve_digit_form_l1212_121257

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def twelve_digit_form (m : ℕ) : ℕ := 1000001 * m

theorem gcd_twelve_digit_form :
  ∃ (g : ℕ), ∀ (m : ℕ), is_six_digit m → 
    (∃ (k : ℕ), twelve_digit_form m = g * k) ∧
    (∀ (d : ℕ), (∀ (n : ℕ), is_six_digit n → ∃ (k : ℕ), twelve_digit_form n = d * k) → d ≤ g) ∧
    g = 1000001 :=
by sorry

end NUMINAMATH_CALUDE_gcd_twelve_digit_form_l1212_121257


namespace NUMINAMATH_CALUDE_two_million_times_three_million_l1212_121281

theorem two_million_times_three_million : 
  (2 * 1000000) * (3 * 1000000) = 6 * 1000000000000 := by
  sorry

end NUMINAMATH_CALUDE_two_million_times_three_million_l1212_121281


namespace NUMINAMATH_CALUDE_distance_to_other_focus_l1212_121259

/-- The distance from a point on an ellipse to the other focus -/
theorem distance_to_other_focus (x y : ℝ) :
  x^2 / 9 + y^2 / 4 = 1 →  -- P is on the ellipse
  ∃ (f₁ f₂ : ℝ × ℝ),  -- existence of two foci
    (∀ (p : ℝ × ℝ), x^2 / 9 + y^2 / 4 = 1 →
      Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
      Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 6) →  -- definition of ellipse
    Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) = 1 →  -- distance to one focus is 1
    Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2) = 5  -- distance to other focus is 5
    := by sorry

end NUMINAMATH_CALUDE_distance_to_other_focus_l1212_121259


namespace NUMINAMATH_CALUDE_sqrt_three_between_fractions_l1212_121284

theorem sqrt_three_between_fractions (n : ℕ+) :
  ((n + 3 : ℝ) / n < Real.sqrt 3 ∧ Real.sqrt 3 < (n + 4 : ℝ) / (n + 1)) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_between_fractions_l1212_121284


namespace NUMINAMATH_CALUDE_degree_of_product_l1212_121295

-- Define polynomials h and j
variable (h j : Polynomial ℝ)

-- Define the degrees of h and j
variable (deg_h : Polynomial.degree h = 3)
variable (deg_j : Polynomial.degree j = 5)

-- Theorem statement
theorem degree_of_product :
  Polynomial.degree (h.comp (Polynomial.X ^ 4) * j.comp (Polynomial.X ^ 3)) = 27 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_product_l1212_121295


namespace NUMINAMATH_CALUDE_roots_sum_sixth_power_l1212_121208

theorem roots_sum_sixth_power (u v : ℝ) : 
  u^2 - 3 * u * Real.sqrt 3 + 3 = 0 →
  v^2 - 3 * v * Real.sqrt 3 + 3 = 0 →
  u^6 + v^6 = 178767 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_sixth_power_l1212_121208


namespace NUMINAMATH_CALUDE_triangular_prism_tetrahedra_l1212_121250

/-- The number of vertices in a triangular prism -/
def triangular_prism_vertices : ℕ := 6

/-- The number of distinct tetrahedra that can be formed using the vertices of a triangular prism -/
def distinct_tetrahedra (n : ℕ) : ℕ := Nat.choose n 4 - 3

theorem triangular_prism_tetrahedra :
  distinct_tetrahedra triangular_prism_vertices = 12 := by sorry

end NUMINAMATH_CALUDE_triangular_prism_tetrahedra_l1212_121250


namespace NUMINAMATH_CALUDE_corn_growth_ratio_l1212_121211

theorem corn_growth_ratio :
  ∀ (growth_week1 growth_week2 growth_week3 total_height : ℝ),
    growth_week1 = 2 →
    growth_week2 = 2 * growth_week1 →
    total_height = 22 →
    total_height = growth_week1 + growth_week2 + growth_week3 →
    growth_week3 / growth_week2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_corn_growth_ratio_l1212_121211


namespace NUMINAMATH_CALUDE_intersection_point_l1212_121233

/-- The line equation y = 2x + 2 -/
def line_equation (x y : ℝ) : Prop := y = 2 * x + 2

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- Theorem: The point (0, 2) is the intersection of the line y = 2x + 2 with the y-axis -/
theorem intersection_point : 
  line_equation 0 2 ∧ on_y_axis 0 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l1212_121233


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_ABCD_l1212_121228

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with side length 1 -/
def UnitCube : Set Point3D :=
  {p | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1 ∧ 0 ≤ p.z ∧ p.z ≤ 1}

/-- The diagonal vertices of the cube -/
def A : Point3D := ⟨0, 0, 0⟩
def C : Point3D := ⟨1, 1, 1⟩

/-- The midpoints of two opposite edges not containing A or C -/
def B : Point3D := ⟨0.5, 0, 1⟩
def D : Point3D := ⟨0.5, 1, 0⟩

/-- The plane passing through A, B, C, and D -/
def InterceptingPlane : Set Point3D :=
  {p | ∃ (s t : ℝ), p = ⟨s, t, 1 - s - t⟩ ∧ 0 ≤ s ∧ s ≤ 1 ∧ 0 ≤ t ∧ t ≤ 1}

/-- The quadrilateral ABCD formed by the intersection of the plane and the cube -/
def QuadrilateralABCD : Set Point3D :=
  UnitCube ∩ InterceptingPlane

/-- The area of a quadrilateral given its vertices -/
def quadrilateralArea (a b c d : Point3D) : ℝ := sorry

theorem area_of_quadrilateral_ABCD :
  quadrilateralArea A B C D = Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_ABCD_l1212_121228


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l1212_121226

/-- The height of a tree after a given number of years, given that it triples its height each year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem: A tree that triples its height every year and reaches 243 feet after 5 years has a height of 9 feet after 2 years -/
theorem tree_height_after_two_years :
  ∃ (initial_height : ℝ),
    tree_height initial_height 5 = 243 ∧
    tree_height initial_height 2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l1212_121226


namespace NUMINAMATH_CALUDE_unique_prime_in_range_l1212_121274

/-- The only prime number in the range (200, 220) is 211 -/
theorem unique_prime_in_range : ∃! (n : ℕ), 200 < n ∧ n < 220 ∧ Nat.Prime n :=
  sorry

end NUMINAMATH_CALUDE_unique_prime_in_range_l1212_121274


namespace NUMINAMATH_CALUDE_benny_picked_two_l1212_121283

-- Define the total number of apples picked
def total_apples : ℕ := 11

-- Define the number of apples Dan picked
def dan_apples : ℕ := 9

-- Define Benny's apples as the difference between total and Dan's
def benny_apples : ℕ := total_apples - dan_apples

-- Theorem stating that Benny picked 2 apples
theorem benny_picked_two : benny_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_benny_picked_two_l1212_121283


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l1212_121263

/-- Given a line with slope m and y-intercept b, prove that their product mb equals -6 -/
theorem line_slope_intercept_product :
  ∀ (m b : ℝ), m = 2 → b = -3 → m * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l1212_121263


namespace NUMINAMATH_CALUDE_matrix_power_4_l1212_121229

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_power_4 : A^4 = !![(-8), 8; 0, 3] := by sorry

end NUMINAMATH_CALUDE_matrix_power_4_l1212_121229


namespace NUMINAMATH_CALUDE_square_difference_equals_360_l1212_121273

theorem square_difference_equals_360 :
  (15 + 12)^2 - (12^2 + 15^2) = 360 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_360_l1212_121273


namespace NUMINAMATH_CALUDE_inequality_proof_l1212_121258

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 / (y - 1)) + (y^2 / (x - 1)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1212_121258


namespace NUMINAMATH_CALUDE_largest_red_socks_l1212_121290

/-- The largest number of red socks in a drawer with specific conditions -/
theorem largest_red_socks (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total ≤ 1991)
  (h2 : total = red + blue)
  (h3 : (red * (red - 1) + blue * (blue - 1)) / (total * (total - 1)) = 1/2) :
  red ≤ 990 ∧ ∃ (r : ℕ), r = 990 ∧ 
    ∃ (t b : ℕ), t ≤ 1991 ∧ t = r + b ∧ 
      (r * (r - 1) + b * (b - 1)) / (t * (t - 1)) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_largest_red_socks_l1212_121290


namespace NUMINAMATH_CALUDE_same_solution_implies_b_value_l1212_121268

theorem same_solution_implies_b_value :
  ∀ (x b : ℚ),
  (3 * x + 9 = 0) ∧ (b * x + 15 = 5) →
  b = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_b_value_l1212_121268


namespace NUMINAMATH_CALUDE_trig_expression_equals_four_l1212_121251

theorem trig_expression_equals_four : 
  1 / Real.cos (80 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_four_l1212_121251


namespace NUMINAMATH_CALUDE_candy_box_count_candy_box_theorem_l1212_121287

theorem candy_box_count : ℝ → Prop :=
  fun x =>
    let day1_eaten := 0.2 * x + 16
    let day1_remaining := x - day1_eaten
    let day2_eaten := 0.3 * day1_remaining + 20
    let day2_remaining := day1_remaining - day2_eaten
    let day3_eaten := 0.75 * day2_remaining + 30
    day3_eaten = day2_remaining ∧ x = 270

theorem candy_box_theorem : ∃ x : ℝ, candy_box_count x :=
  sorry

end NUMINAMATH_CALUDE_candy_box_count_candy_box_theorem_l1212_121287


namespace NUMINAMATH_CALUDE_poultry_farm_daily_loss_l1212_121296

/-- Calculates the daily loss of guinea fowls in a poultry farm scenario --/
theorem poultry_farm_daily_loss (initial_chickens initial_turkeys initial_guinea_fowls : ℕ)
  (daily_chicken_loss daily_turkey_loss : ℕ) (total_birds_after_week : ℕ) :
  initial_chickens = 300 →
  initial_turkeys = 200 →
  initial_guinea_fowls = 80 →
  daily_chicken_loss = 20 →
  daily_turkey_loss = 8 →
  total_birds_after_week = 349 →
  ∃ (daily_guinea_fowl_loss : ℕ),
    daily_guinea_fowl_loss = 5 ∧
    total_birds_after_week = 
      initial_chickens - 7 * daily_chicken_loss +
      initial_turkeys - 7 * daily_turkey_loss +
      initial_guinea_fowls - 7 * daily_guinea_fowl_loss :=
by
  sorry


end NUMINAMATH_CALUDE_poultry_farm_daily_loss_l1212_121296


namespace NUMINAMATH_CALUDE_canoe_kayak_ratio_l1212_121243

/-- Represents the rental business scenario -/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  total_revenue : ℕ
  canoe_kayak_difference : ℕ

/-- Theorem stating the ratio of canoes to kayaks rented -/
theorem canoe_kayak_ratio (rb : RentalBusiness) 
  (h1 : rb.canoe_price = 14)
  (h2 : rb.kayak_price = 15)
  (h3 : rb.total_revenue = 288)
  (h4 : rb.canoe_kayak_difference = 4)
  (h5 : ∃ (k : ℕ), rb.canoe_price * (k + rb.canoe_kayak_difference) + rb.kayak_price * k = rb.total_revenue) :
  ∃ (c k : ℕ), c = k + rb.canoe_kayak_difference ∧ c * rb.canoe_price + k * rb.kayak_price = rb.total_revenue ∧ c * 2 = k * 3 :=
by sorry

end NUMINAMATH_CALUDE_canoe_kayak_ratio_l1212_121243


namespace NUMINAMATH_CALUDE_cos_ninety_degrees_l1212_121212

theorem cos_ninety_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_ninety_degrees_l1212_121212


namespace NUMINAMATH_CALUDE_x_equals_nine_l1212_121202

theorem x_equals_nine (u : ℤ) (x : ℚ) 
  (h1 : u = -6) 
  (h2 : x = (1 : ℚ) / 3 * (3 - 4 * u)) : 
  x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_nine_l1212_121202


namespace NUMINAMATH_CALUDE_quadratic_factor_problem_l1212_121256

theorem quadratic_factor_problem (a b : ℝ) :
  (∀ x, x^2 + 6*x + a = (x + 5)*(x + b)) → b = 1 ∧ a = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factor_problem_l1212_121256


namespace NUMINAMATH_CALUDE_well_volume_l1212_121245

/-- The volume of a circular cylinder with diameter 2 metres and height 10 metres is π × 10 m³ -/
theorem well_volume :
  let diameter : ℝ := 2
  let depth : ℝ := 10
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * depth
  volume = π * 10 := by
  sorry

end NUMINAMATH_CALUDE_well_volume_l1212_121245


namespace NUMINAMATH_CALUDE_laser_reflection_theorem_l1212_121262

/-- Regular hexagon with side length 2 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- Point G on BC where the laser beam hits -/
def G (h : RegularHexagon) : ℝ × ℝ := sorry

/-- Midpoint of DE -/
def M (h : RegularHexagon) : ℝ × ℝ := sorry

/-- Length of BG -/
def BG_length (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating that BG length is 2/5 -/
theorem laser_reflection_theorem (h : RegularHexagon) :
  let g := G h
  let m := M h
  (∃ (t : ℝ), t • (g.1 - h.A.1, g.2 - h.A.2) = (m.1 - g.1, m.2 - g.2)) →
  BG_length h = 2/5 := by sorry

end NUMINAMATH_CALUDE_laser_reflection_theorem_l1212_121262


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_bisector_x_intersection_l1212_121240

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def perpendicular_bisector_intersects_x_axis (A B P : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ), (P.2 = 0) ∧ 
  (P.1 - (A.1 + B.1)/2) = m * ((A.2 + B.2)/2) ∧
  (B.2 - A.2) * (P.1 - (A.1 + B.1)/2) = (A.1 - B.1) * ((A.2 + B.2)/2)

theorem ellipse_perpendicular_bisector_x_intersection
  (a b : ℝ) (h_ab : a > b ∧ b > 0) (A B P : ℝ × ℝ) :
  ellipse a b A.1 A.2 →
  ellipse a b B.1 B.2 →
  perpendicular_bisector_intersects_x_axis A B P →
  -((a^2 - b^2)/a) < P.1 ∧ P.1 < (a^2 - b^2)/a :=
by sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_bisector_x_intersection_l1212_121240


namespace NUMINAMATH_CALUDE_greatest_integer_x_less_than_32_l1212_121272

theorem greatest_integer_x_less_than_32 :
  (∃ x : ℕ+, x.val = 5 ∧ (∀ y : ℕ+, y.val > 5 → (y.val : ℝ)^5 / (y.val : ℝ)^3 ≥ 32)) ∧
  ((5 : ℝ)^5 / (5 : ℝ)^3 < 32) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_x_less_than_32_l1212_121272


namespace NUMINAMATH_CALUDE_is_focus_of_hyperbola_l1212_121223

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - 3*y^2 + 6*x - 12*y - 8 = 0

/-- The focus point -/
def focus : ℝ × ℝ := (-1, -2)

/-- Theorem stating that the given point is a focus of the hyperbola -/
theorem is_focus_of_hyperbola : 
  ∃ (c : ℝ), c > 0 ∧ 
  ∀ (x y : ℝ), hyperbola_equation x y → 
    (x + 1)^2 + (y + 2)^2 - ((x + 5)^2 + (y + 2)^2) = 4*c := by
  sorry

end NUMINAMATH_CALUDE_is_focus_of_hyperbola_l1212_121223


namespace NUMINAMATH_CALUDE_equation_solution_l1212_121270

theorem equation_solution : ∃ x : ℝ, 2 * ((x - 1) - (2 * x + 1)) = 6 ∧ x = -5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1212_121270


namespace NUMINAMATH_CALUDE_bookshop_inventory_l1212_121235

/-- Calculates the final number of books in a bookshop after weekend sales and a new shipment --/
theorem bookshop_inventory (initial_inventory : ℕ) (saturday_in_store : ℕ) (saturday_online : ℕ) (sunday_in_store_multiplier : ℕ) (sunday_online_increase : ℕ) (new_shipment : ℕ) : 
  initial_inventory = 743 →
  saturday_in_store = 37 →
  saturday_online = 128 →
  sunday_in_store_multiplier = 2 →
  sunday_online_increase = 34 →
  new_shipment = 160 →
  initial_inventory - 
    (saturday_in_store + saturday_online + 
     sunday_in_store_multiplier * saturday_in_store + 
     (saturday_online + sunday_online_increase)) + 
  new_shipment = 502 := by
sorry

end NUMINAMATH_CALUDE_bookshop_inventory_l1212_121235


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1212_121279

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo 2 3 : Set ℝ) = {x | (x - 2) * (x - 3) / (x^2 + 1) < 0} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1212_121279


namespace NUMINAMATH_CALUDE_det_E_l1212_121299

/-- A 3x3 matrix representing a dilation centered at the origin with scale factor 4 -/
def E : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ ↦ 4)

/-- Theorem: The determinant of E is 64 -/
theorem det_E : Matrix.det E = 64 := by
  sorry

end NUMINAMATH_CALUDE_det_E_l1212_121299


namespace NUMINAMATH_CALUDE_marble_difference_l1212_121215

theorem marble_difference (red_marbles : ℕ) (red_bags : ℕ) (blue_marbles : ℕ) (blue_bags : ℕ)
  (h1 : red_marbles = 288)
  (h2 : red_bags = 12)
  (h3 : blue_marbles = 243)
  (h4 : blue_bags = 9)
  (h5 : red_bags ≠ 0)
  (h6 : blue_bags ≠ 0) :
  blue_marbles / blue_bags - red_marbles / red_bags = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l1212_121215


namespace NUMINAMATH_CALUDE_rectangle_to_circle_area_l1212_121278

/-- Given a rectangle with area 200 square units and one side 5 units longer than twice the other side,
    the area of the largest circle that can be formed from a string equal in length to the rectangle's perimeter
    is 400/π square units. -/
theorem rectangle_to_circle_area (x : ℝ) (h1 : x > 0) (h2 : x * (2 * x + 5) = 200) : 
  let perimeter := 2 * (x + (2 * x + 5))
  (perimeter / (2 * Real.pi))^2 * Real.pi = 400 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_rectangle_to_circle_area_l1212_121278


namespace NUMINAMATH_CALUDE_set_operations_l1212_121242

def U : Set ℤ := {x | -3 ≤ x ∧ x ≤ 3}
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-1, 0, 1}
def C : Set ℤ := {-2, 0, 2}

theorem set_operations :
  (A ∪ (B ∩ C) = {0, 1, 2, 3}) ∧
  (A ∩ (U \ (B ∪ C)) = {3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1212_121242


namespace NUMINAMATH_CALUDE_quadratic_roots_and_m_value_l1212_121293

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (2-m)*x + (1-m)

-- Theorem statement
theorem quadratic_roots_and_m_value (m : ℝ) :
  (∀ x : ℝ, ∃ y z : ℝ, y ≠ z ∧ quadratic m y = 0 ∧ quadratic m z = 0) ∧
  (m < 0 → (∃ y z : ℝ, y ≠ z ∧ quadratic m y = 0 ∧ quadratic m z = 0 ∧ |y - z| = 3) → m = -3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_m_value_l1212_121293


namespace NUMINAMATH_CALUDE_series_sum_l1212_121265

theorem series_sum : 
  let a : ℕ → ℚ := fun n => (4*n + 3) / ((4*n + 1)^2 * (4*n + 5)^2)
  ∑' n, a n = 1/200 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l1212_121265


namespace NUMINAMATH_CALUDE_M_always_positive_l1212_121227

theorem M_always_positive (x y : ℝ) : 3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13 > 0 := by
  sorry

end NUMINAMATH_CALUDE_M_always_positive_l1212_121227


namespace NUMINAMATH_CALUDE_expression_simplification_l1212_121277

theorem expression_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ((x + 1) / (x^2 - 4) - 1 / (x + 2)) / (3 / (x - 2)) = 1 / (x + 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1212_121277


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1212_121289

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (2, -1) and b = (k, 5/2), then k = -5. -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, -1) →
  b = (k, 5/2) →
  (∃ (t : ℝ), t ≠ 0 ∧ b = t • a) →
  k = -5 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1212_121289


namespace NUMINAMATH_CALUDE_tenth_power_sum_l1212_121222

theorem tenth_power_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_tenth_power_sum_l1212_121222


namespace NUMINAMATH_CALUDE_star_equal_set_is_three_lines_l1212_121244

-- Define the ⋆ operation
def star (a b : ℝ) : ℝ := a^2 * b + a * b^2

-- Define the set of points (x, y) where x ⋆ y = y ⋆ x
def star_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

-- Theorem statement
theorem star_equal_set_is_three_lines :
  star_equal_set = {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 + p.2 = 0} :=
by sorry

end NUMINAMATH_CALUDE_star_equal_set_is_three_lines_l1212_121244


namespace NUMINAMATH_CALUDE_product_mod_twenty_l1212_121271

theorem product_mod_twenty :
  ∃ n : ℕ, 0 ≤ n ∧ n < 20 ∧ (77 * 88 * 99 : ℤ) ≡ n [ZMOD 20] ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_twenty_l1212_121271


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l1212_121237

theorem divisibility_equivalence (n m : ℤ) : 
  (∃ k : ℤ, 2*n + 5*m = 9*k) ↔ (∃ l : ℤ, 5*n + 8*m = 9*l) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l1212_121237


namespace NUMINAMATH_CALUDE_barbara_shopping_l1212_121207

/-- The amount spent on goods other than tuna and water in Barbara's shopping trip -/
def other_goods_cost (tuna_packs : ℕ) (tuna_price : ℚ) (water_bottles : ℕ) (water_price : ℚ) (total_cost : ℚ) : ℚ :=
  total_cost - (tuna_packs * tuna_price + water_bottles * water_price)

/-- Theorem stating that Barbara spent $40 on goods other than tuna and water -/
theorem barbara_shopping :
  other_goods_cost 5 2 4 (3/2) 56 = 40 := by
  sorry

end NUMINAMATH_CALUDE_barbara_shopping_l1212_121207


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1212_121269

theorem consecutive_integers_sum (n : ℕ) (h : n > 0) :
  (6 * n + 15 = 2013) → (n + 5 = 338) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1212_121269


namespace NUMINAMATH_CALUDE_frequency_calculation_l1212_121204

theorem frequency_calculation (sample_size : ℕ) (area_percentage : ℚ) (h1 : sample_size = 50) (h2 : area_percentage = 16/100) :
  (sample_size : ℚ) * area_percentage = 8 := by
  sorry

end NUMINAMATH_CALUDE_frequency_calculation_l1212_121204


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1212_121225

theorem fraction_subtraction : (5/6 : ℚ) + (1/4 : ℚ) - (2/3 : ℚ) = (5/12 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1212_121225


namespace NUMINAMATH_CALUDE_plane_speed_calculation_l1212_121255

/-- Two planes traveling in opposite directions -/
structure TwoPlanes where
  speed_west : ℝ
  speed_east : ℝ
  time : ℝ
  total_distance : ℝ

/-- The theorem stating the conditions and the result to be proved -/
theorem plane_speed_calculation (planes : TwoPlanes) 
  (h1 : planes.speed_west = 275)
  (h2 : planes.time = 3.5)
  (h3 : planes.total_distance = 2100)
  : planes.speed_east = 325 := by
  sorry

#check plane_speed_calculation

end NUMINAMATH_CALUDE_plane_speed_calculation_l1212_121255


namespace NUMINAMATH_CALUDE_hemisphere_base_area_l1212_121232

/-- Given a hemisphere with total surface area 9, prove its base area is 3 -/
theorem hemisphere_base_area (r : ℝ) (h : 3 * π * r^2 = 9) : π * r^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_base_area_l1212_121232


namespace NUMINAMATH_CALUDE_parabola_focus_hyperbola_vertex_asymptote_distance_l1212_121203

-- Define the parabola
def parabola (a : ℝ) (x y : ℝ) : Prop := x = a * y^2 ∧ a ≠ 0

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 12 - y^2 / 4 = 1

-- Theorem for the focus of the parabola
theorem parabola_focus (a : ℝ) :
  ∃ (x y : ℝ), parabola a x y → (x = 1 / (4 * a) ∧ y = 0) :=
sorry

-- Theorem for the distance from vertex to asymptote of the hyperbola
theorem hyperbola_vertex_asymptote_distance :
  ∃ (d : ℝ), (∀ x y : ℝ, hyperbola x y → d = Real.sqrt 30 / 5) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_hyperbola_vertex_asymptote_distance_l1212_121203


namespace NUMINAMATH_CALUDE_roger_tray_capacity_l1212_121294

/-- The number of trays Roger can carry at a time -/
def trays_per_trip : ℕ := sorry

/-- The number of trips Roger made -/
def num_trips : ℕ := 3

/-- The number of trays Roger picked up from the first table -/
def trays_table1 : ℕ := 10

/-- The number of trays Roger picked up from the second table -/
def trays_table2 : ℕ := 2

/-- The total number of trays Roger picked up -/
def total_trays : ℕ := trays_table1 + trays_table2

theorem roger_tray_capacity :
  trays_per_trip * num_trips = total_trays ∧ trays_per_trip = 4 := by
  sorry

end NUMINAMATH_CALUDE_roger_tray_capacity_l1212_121294


namespace NUMINAMATH_CALUDE_inverse_g_at_167_l1212_121230

def g (x : ℝ) : ℝ := 5 * x^5 + 7

theorem inverse_g_at_167 : g⁻¹ 167 = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_g_at_167_l1212_121230


namespace NUMINAMATH_CALUDE_total_cost_is_49_l1212_121218

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def discount_threshold : ℕ := 35
def discount_amount : ℕ := 5
def num_sandwiches : ℕ := 6
def num_sodas : ℕ := 10

def total_cost : ℕ :=
  let pre_discount := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  if pre_discount > discount_threshold then
    pre_discount - discount_amount
  else
    pre_discount

theorem total_cost_is_49 : total_cost = 49 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_49_l1212_121218


namespace NUMINAMATH_CALUDE_problem_solution_l1212_121200

theorem problem_solution (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (∀ x y z, x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
    1/a + 1/b + 1/c ≤ 1/x + 1/y + 1/z) ∧
  (1/(1-a) + 1/(1-b) + 1/(1-c) ≥ 2/(1+a) + 2/(1+b) + 2/(1+c)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1212_121200


namespace NUMINAMATH_CALUDE_u_n_eq_2n_minus_1_l1212_121246

/-- 
Given a positive integer n, u_n is the smallest positive integer such that 
for any odd integer d, the number of integers in any u_n consecutive odd integers 
that are divisible by d is at least as many as the number of integers among 
1, 3, 5, ..., 2n-1 that are divisible by d.
-/
def u_n (n : ℕ+) : ℕ := sorry

/-- The main theorem stating that u_n is equal to 2n - 1 -/
theorem u_n_eq_2n_minus_1 (n : ℕ+) : u_n n = 2 * n - 1 := by sorry

end NUMINAMATH_CALUDE_u_n_eq_2n_minus_1_l1212_121246


namespace NUMINAMATH_CALUDE_smallest_a_value_l1212_121205

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) :
  17 ≤ a ∧ ∀ a' : ℝ, (0 ≤ a' ∧ (∀ x : ℤ, Real.sin (a' * x + b) = Real.sin (17 * x))) → a' ≥ 17 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l1212_121205


namespace NUMINAMATH_CALUDE_interior_perimeter_is_155_l1212_121252

/-- Triangle PQR with parallel lines forming interior triangle --/
structure TriangleWithParallels where
  /-- Side length PQ --/
  pq : ℝ
  /-- Side length QR --/
  qr : ℝ
  /-- Side length PR --/
  pr : ℝ
  /-- Length of intersection of m_P with triangle interior --/
  m_p : ℝ
  /-- Length of intersection of m_Q with triangle interior --/
  m_q : ℝ
  /-- Length of intersection of m_R with triangle interior --/
  m_r : ℝ
  /-- m_P is parallel to QR --/
  m_p_parallel_qr : True
  /-- m_Q is parallel to RP --/
  m_q_parallel_rp : True
  /-- m_R is parallel to PQ --/
  m_r_parallel_pq : True

/-- The perimeter of the interior triangle formed by parallel lines --/
def interiorPerimeter (t : TriangleWithParallels) : ℝ :=
  t.m_p + t.m_q + t.m_r

/-- Theorem: The perimeter of the interior triangle is 155 --/
theorem interior_perimeter_is_155 (t : TriangleWithParallels) 
  (h1 : t.pq = 160) (h2 : t.qr = 300) (h3 : t.pr = 240)
  (h4 : t.m_p = 75) (h5 : t.m_q = 60) (h6 : t.m_r = 20) :
  interiorPerimeter t = 155 := by
  sorry

end NUMINAMATH_CALUDE_interior_perimeter_is_155_l1212_121252


namespace NUMINAMATH_CALUDE_second_discount_percentage_l1212_121267

theorem second_discount_percentage
  (initial_price : ℝ)
  (first_discount : ℝ)
  (final_price : ℝ)
  (h1 : initial_price = 400)
  (h2 : first_discount = 25)
  (h3 : final_price = 240)
  : ∃ (second_discount : ℝ),
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l1212_121267


namespace NUMINAMATH_CALUDE_baking_powder_difference_l1212_121249

-- Define the constants
def yesterday_supply : Real := 1.5 -- in kg
def today_supply : Real := 1.2 -- in kg (converted from 1200 grams)
def box_size : Real := 5 -- kg per box

-- Define the theorem
theorem baking_powder_difference :
  yesterday_supply - today_supply = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_baking_powder_difference_l1212_121249


namespace NUMINAMATH_CALUDE_lawrence_average_work_hours_l1212_121288

def lawrence_work_hours (full_days : ℕ) (partial_days : ℕ) (full_day_hours : ℝ) (partial_day_hours : ℝ) : ℝ :=
  (full_days : ℝ) * full_day_hours + (partial_days : ℝ) * partial_day_hours

theorem lawrence_average_work_hours :
  let total_days : ℕ := 5
  let full_days : ℕ := 3
  let partial_days : ℕ := 2
  let full_day_hours : ℝ := 8
  let partial_day_hours : ℝ := 5.5
  let total_hours := lawrence_work_hours full_days partial_days full_day_hours partial_day_hours
  total_hours / total_days = 7 := by
sorry

end NUMINAMATH_CALUDE_lawrence_average_work_hours_l1212_121288


namespace NUMINAMATH_CALUDE_square_side_length_is_twenty_l1212_121241

/-- The side length of a square that can contain specific numbers of square tiles of different sizes -/
def square_side_length : ℕ := 
  let one_by_one := 4
  let two_by_two := 8
  let three_by_three := 12
  let four_by_four := 16
  let total_area := one_by_one * 1^2 + two_by_two * 2^2 + three_by_three * 3^2 + four_by_four * 4^2
  Nat.sqrt total_area

/-- Theorem stating that the side length of the square is 20 -/
theorem square_side_length_is_twenty : square_side_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_is_twenty_l1212_121241


namespace NUMINAMATH_CALUDE_factor_expression_l1212_121292

theorem factor_expression (x : ℝ) : 75 * x^2 + 50 * x = 25 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1212_121292


namespace NUMINAMATH_CALUDE_max_sum_with_length_constraint_l1212_121213

/-- Length of an integer is the number of positive prime factors (not necessarily distinct) --/
def length (n : ℕ) : ℕ := sorry

theorem max_sum_with_length_constraint :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ length x + length y = 16 ∧
  ∀ (a b : ℕ), a > 1 → b > 1 → length a + length b = 16 → a + 3 * b ≤ x + 3 * y ∧
  x + 3 * y = 98305 := by sorry

end NUMINAMATH_CALUDE_max_sum_with_length_constraint_l1212_121213


namespace NUMINAMATH_CALUDE_value_added_to_forty_percent_l1212_121285

theorem value_added_to_forty_percent (N : ℝ) (V : ℝ) : 
  N = 100 → 0.4 * N + V = N → V = 60 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_forty_percent_l1212_121285


namespace NUMINAMATH_CALUDE_distance_between_foci_l1212_121238

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 45 + y^2 / 5 = 9

-- Theorem statement
theorem distance_between_foci :
  ∃ (a b c : ℝ), 
    (∀ x y, ellipse_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    2 * c = 12 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_distance_between_foci_l1212_121238


namespace NUMINAMATH_CALUDE_well_depth_l1212_121275

/-- The depth of a well given specific conditions -/
theorem well_depth : 
  -- Define the distance function
  let distance (t : ℝ) : ℝ := 16 * t^2
  -- Define the speed of sound
  let sound_speed : ℝ := 1120
  -- Define the total time
  let total_time : ℝ := 7.7
  -- Define the depth of the well
  let depth : ℝ := distance (total_time - depth / sound_speed)
  -- Prove that the depth is 784 feet
  depth = 784 := by sorry

end NUMINAMATH_CALUDE_well_depth_l1212_121275


namespace NUMINAMATH_CALUDE_race_start_distance_l1212_121209

theorem race_start_distance (speed_a speed_b : ℝ) (total_distance : ℝ) (start_distance : ℝ) : 
  speed_a = (5 / 3) * speed_b →
  total_distance = 200 →
  total_distance / speed_a = (total_distance - start_distance) / speed_b →
  start_distance = 80 := by
sorry

end NUMINAMATH_CALUDE_race_start_distance_l1212_121209


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1212_121210

theorem quadratic_always_positive (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 > 0) ↔ -2 < k ∧ k < 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1212_121210


namespace NUMINAMATH_CALUDE_cookies_leftover_is_four_l1212_121291

/-- The number of cookies left over when selling in packs of 10 -/
def cookies_leftover (abigail beatrice carson : ℕ) : ℕ :=
  (abigail + beatrice + carson) % 10

/-- Theorem stating that the number of cookies left over is 4 -/
theorem cookies_leftover_is_four :
  cookies_leftover 53 65 26 = 4 := by
  sorry

end NUMINAMATH_CALUDE_cookies_leftover_is_four_l1212_121291


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l1212_121298

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 2 * y = 2) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (z : ℝ), 3^x + 9^y ≥ z ∧ (∃ (a b : ℝ), a + 2 * b = 2 ∧ 3^a + 9^b = z) → m ≤ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l1212_121298


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l1212_121224

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ x : ℝ, ax < 1 ∧ x - a < 0) → a ∈ Set.Ici (-1) :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l1212_121224


namespace NUMINAMATH_CALUDE_abs_greater_than_negative_l1212_121221

theorem abs_greater_than_negative (a b : ℝ) (h : a < b ∧ b < 0) : |a| > -b := by
  sorry

end NUMINAMATH_CALUDE_abs_greater_than_negative_l1212_121221


namespace NUMINAMATH_CALUDE_largest_n_for_product_l1212_121260

/-- An arithmetic sequence with integer terms -/
def ArithmeticSequence (a₁ : ℤ) (d : ℤ) : ℕ → ℤ := fun n => a₁ + (n - 1) * d

theorem largest_n_for_product (x y : ℤ) (hxy : x < y) :
  let a := ArithmeticSequence 2 x
  let b := ArithmeticSequence 3 y
  (∃ n : ℕ, a n * b n = 1638) →
  (∀ m : ℕ, a m * b m = 1638 → m ≤ 35) ∧
  (a 35 * b 35 = 1638) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_product_l1212_121260


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l1212_121247

theorem unique_square_divisible_by_three_in_range : ∃! x : ℕ,
  (∃ n : ℕ, x = n^2) ∧
  x % 3 = 0 ∧
  60 < x ∧ x < 130 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_three_in_range_l1212_121247


namespace NUMINAMATH_CALUDE_first_player_wins_l1212_121276

/-- Represents the game state with k piles of stones -/
structure GameState where
  k : ℕ+
  n : Fin k → ℕ

/-- Defines the set of winning positions -/
def WinningPositions : Set GameState :=
  sorry

/-- Defines a valid move in the game -/
def ValidMove (s₁ s₂ : GameState) : Prop :=
  sorry

/-- Theorem stating the winning condition for the first player -/
theorem first_player_wins (s : GameState) :
  s ∈ WinningPositions ↔
    ∃ (s' : GameState), ValidMove s s' ∧ 
      ∀ (s'' : GameState), ValidMove s' s'' → s'' ∈ WinningPositions :=
by sorry

end NUMINAMATH_CALUDE_first_player_wins_l1212_121276


namespace NUMINAMATH_CALUDE_plant_arrangement_count_l1212_121231

/-- The number of ways to arrange plants in a row -/
def arrangePlants (basil tomato pepper : Nat) : Nat :=
  Nat.factorial 3 * Nat.factorial basil * Nat.factorial tomato * Nat.factorial pepper

theorem plant_arrangement_count :
  arrangePlants 4 4 3 = 20736 := by
  sorry

end NUMINAMATH_CALUDE_plant_arrangement_count_l1212_121231


namespace NUMINAMATH_CALUDE_total_fault_movement_total_movement_is_17_25_l1212_121280

/-- Represents the movement of a fault line over two years -/
structure FaultMovement where
  pastYear : Float
  yearBefore : Float

/-- Calculates the total movement of a fault line over two years -/
def totalMovement (fault : FaultMovement) : Float :=
  fault.pastYear + fault.yearBefore

/-- Theorem: The total movement of all fault lines is the sum of their individual movements -/
theorem total_fault_movement (faultA faultB faultC : FaultMovement) :
  totalMovement faultA + totalMovement faultB + totalMovement faultC =
  faultA.pastYear + faultA.yearBefore +
  faultB.pastYear + faultB.yearBefore +
  faultC.pastYear + faultC.yearBefore := by
  sorry

/-- Given fault movements -/
def faultA : FaultMovement := { pastYear := 1.25, yearBefore := 5.25 }
def faultB : FaultMovement := { pastYear := 2.5, yearBefore := 3.0 }
def faultC : FaultMovement := { pastYear := 0.75, yearBefore := 4.5 }

/-- Theorem: The total movement of the given fault lines is 17.25 inches -/
theorem total_movement_is_17_25 :
  totalMovement faultA + totalMovement faultB + totalMovement faultC = 17.25 := by
  sorry

end NUMINAMATH_CALUDE_total_fault_movement_total_movement_is_17_25_l1212_121280


namespace NUMINAMATH_CALUDE_red_jellybeans_count_l1212_121234

def total_jellybeans : ℕ := 200
def blue_jellybeans : ℕ := 14
def purple_jellybeans : ℕ := 26
def orange_jellybeans : ℕ := 40

theorem red_jellybeans_count :
  total_jellybeans - (blue_jellybeans + purple_jellybeans + orange_jellybeans) = 120 := by
  sorry

end NUMINAMATH_CALUDE_red_jellybeans_count_l1212_121234


namespace NUMINAMATH_CALUDE_triangle_area_product_l1212_121254

theorem triangle_area_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (1/2) * (12/a) * (12/b) = 12) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_product_l1212_121254


namespace NUMINAMATH_CALUDE_problem_statement_l1212_121239

theorem problem_statement (x n f : ℝ) : 
  x = (3 + Real.sqrt 8)^500 →
  n = ⌊x⌋ →
  f = x - n →
  x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1212_121239


namespace NUMINAMATH_CALUDE_quadratic_with_odd_coeff_no_rational_roots_l1212_121219

theorem quadratic_with_odd_coeff_no_rational_roots (a b c : ℤ) :
  Odd a → Odd b → Odd c → ¬ IsSquare (b^2 - 4*a*c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_with_odd_coeff_no_rational_roots_l1212_121219


namespace NUMINAMATH_CALUDE_elf_circle_arrangement_exists_l1212_121201

/-- Represents the height of an elf -/
inductive ElfHeight
| Short
| Tall

/-- Represents an elf in the circle -/
structure Elf :=
  (position : Nat)
  (height : ElfHeight)

/-- Checks if an elf is taller than both neighbors -/
def isTallerThanNeighbors (elves : List Elf) (position : Nat) : Bool :=
  sorry

/-- Checks if an elf is shorter than both neighbors -/
def isShorterThanNeighbors (elves : List Elf) (position : Nat) : Bool :=
  sorry

/-- Checks if all elves in the circle satisfy the eye-closing condition -/
def allElvesSatisfyCondition (elves : List Elf) : Bool :=
  sorry

/-- Theorem: There exists an arrangement of 100 elves that satisfies all conditions -/
theorem elf_circle_arrangement_exists : 
  ∃ (elves : List Elf), 
    elves.length = 100 ∧ 
    (∀ e ∈ elves, e.position ≤ 100) ∧
    allElvesSatisfyCondition elves :=
  sorry

end NUMINAMATH_CALUDE_elf_circle_arrangement_exists_l1212_121201


namespace NUMINAMATH_CALUDE_field_dimension_solution_l1212_121264

/-- Represents the dimensions of a rectangular field -/
structure FieldDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular field -/
def fieldArea (d : FieldDimensions) : ℝ := d.length * d.width

/-- Theorem: For a rectangular field with dimensions (3m + 4) and (m - 3),
    if the area is 80 square units, then m = 19/3 -/
theorem field_dimension_solution (m : ℝ) :
  let d := FieldDimensions.mk (3 * m + 4) (m - 3)
  fieldArea d = 80 → m = 19/3 := by
  sorry


end NUMINAMATH_CALUDE_field_dimension_solution_l1212_121264


namespace NUMINAMATH_CALUDE_sqrt_two_minus_sqrt_eight_l1212_121253

theorem sqrt_two_minus_sqrt_eight : Real.sqrt 2 - Real.sqrt 8 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_sqrt_eight_l1212_121253


namespace NUMINAMATH_CALUDE_nut_distribution_theorem_l1212_121261

/-- Represents a distribution of nuts among three piles -/
structure NutDistribution :=
  (pile1 pile2 pile3 : ℕ)

/-- Represents an operation of moving nuts between piles -/
inductive MoveOperation
  | move12 : MoveOperation  -- Move from pile 1 to pile 2
  | move13 : MoveOperation  -- Move from pile 1 to pile 3
  | move21 : MoveOperation  -- Move from pile 2 to pile 1
  | move23 : MoveOperation  -- Move from pile 2 to pile 3
  | move31 : MoveOperation  -- Move from pile 3 to pile 1
  | move32 : MoveOperation  -- Move from pile 3 to pile 2

/-- Applies a single move operation to a distribution -/
def applyMove (d : NutDistribution) (m : MoveOperation) : NutDistribution :=
  sorry

/-- Checks if a pile has an even number of nuts -/
def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Checks if a distribution has the desired property (one pile with half the nuts) -/
def hasHalfInOnePile (d : NutDistribution) : Prop :=
  let total := d.pile1 + d.pile2 + d.pile3
  d.pile1 = total / 2 ∨ d.pile2 = total / 2 ∨ d.pile3 = total / 2

/-- The main theorem statement -/
theorem nut_distribution_theorem (initial : NutDistribution) :
  isEven (initial.pile1 + initial.pile2 + initial.pile3) →
  ∃ (moves : List MoveOperation), 
    hasHalfInOnePile (moves.foldl applyMove initial) :=
by sorry

end NUMINAMATH_CALUDE_nut_distribution_theorem_l1212_121261


namespace NUMINAMATH_CALUDE_number_exceeding_fraction_l1212_121286

theorem number_exceeding_fraction : ∃ x : ℚ, x = (3/8) * x + 25 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_fraction_l1212_121286


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1212_121214

/-- Given Mike's current age m and Dan's current age d, prove that the number of years
    until their age ratio is 3:2 is 97, given the initial conditions. -/
theorem age_ratio_problem (m d : ℕ) (h1 : m - 3 = 4 * (d - 3)) (h2 : m - 8 = 5 * (d - 8)) :
  ∃ x : ℕ, x = 97 ∧ (m + x : ℚ) / (d + x) = 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l1212_121214


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1212_121206

theorem binomial_coefficient_ratio (m n : ℕ) : 
  (Nat.choose (n + 1) (m + 1) : ℚ) / (Nat.choose (n + 1) m : ℚ) = 5 / 3 →
  (Nat.choose (n + 1) m : ℚ) / (Nat.choose (n + 1) (m - 1) : ℚ) = 5 / 3 →
  m = 3 ∧ n = 6 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1212_121206


namespace NUMINAMATH_CALUDE_odd_number_pattern_l1212_121216

/-- Represents the number of odd numbers in a row of the pattern -/
def row_length (n : ℕ) : ℕ := 2 * n - 1

/-- Calculates the sum of odd numbers up to the nth row -/
def sum_to_row (n : ℕ) : ℕ := n^2

/-- Represents the nth odd number -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

/-- The problem statement -/
theorem odd_number_pattern :
  let total_previous_rows := sum_to_row 20
  let position_in_row := 6
  nth_odd (total_previous_rows + position_in_row) = 811 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_pattern_l1212_121216


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_l1212_121236

theorem right_triangle_max_ratio (k l a b c : ℝ) (hk : k > 0) (hl : l > 0) : 
  (k * a)^2 + (l * b)^2 = c^2 → (k * a + l * b) / c ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_l1212_121236


namespace NUMINAMATH_CALUDE_student_contribution_l1212_121297

theorem student_contribution
  (total_contribution : ℕ)
  (class_funds : ℕ)
  (num_students : ℕ)
  (h1 : total_contribution = 90)
  (h2 : class_funds = 14)
  (h3 : num_students = 19) :
  (total_contribution - class_funds) / num_students = 4 :=
by sorry

end NUMINAMATH_CALUDE_student_contribution_l1212_121297


namespace NUMINAMATH_CALUDE_inequality_holds_for_p_greater_than_two_l1212_121220

theorem inequality_holds_for_p_greater_than_two (p q : ℝ) 
  (hp : p > 2) (hq : q > 0) : 
  4 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 4 * p * q) / (p + q) > 3 * p^3 * q := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_for_p_greater_than_two_l1212_121220


namespace NUMINAMATH_CALUDE_opposite_sign_and_integer_part_l1212_121217

theorem opposite_sign_and_integer_part (a b c : ℝ) : 
  (∃ k : ℝ, k * (Real.sqrt (a - 4)) = -(2 - 2*b)^2 ∧ k ≠ 0) →
  c = Int.floor (Real.sqrt 10) →
  a = 4 ∧ b = 1 ∧ c = 3 := by
sorry

end NUMINAMATH_CALUDE_opposite_sign_and_integer_part_l1212_121217


namespace NUMINAMATH_CALUDE_car_push_distance_l1212_121248

/-- Proves that the total distance traveled is 10 miles given the conditions of the problem --/
theorem car_push_distance : 
  let segment1_distance : ℝ := 3
  let segment1_speed : ℝ := 6
  let segment2_distance : ℝ := 3
  let segment2_speed : ℝ := 3
  let segment3_distance : ℝ := 4
  let segment3_speed : ℝ := 8
  let total_time : ℝ := 2
  segment1_distance / segment1_speed + 
  segment2_distance / segment2_speed + 
  segment3_distance / segment3_speed = total_time →
  segment1_distance + segment2_distance + segment3_distance = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_car_push_distance_l1212_121248
