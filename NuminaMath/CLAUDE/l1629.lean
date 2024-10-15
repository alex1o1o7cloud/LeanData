import Mathlib

namespace NUMINAMATH_CALUDE_samantha_laundry_loads_l1629_162994

/-- The number of loads of laundry Samantha did in the wash -/
def laundry_loads : ℕ :=
  -- We'll define this later in the theorem
  sorry

/-- The cost of using a washer for one load -/
def washer_cost : ℚ := 4

/-- The cost of using a dryer for 10 minutes -/
def dryer_cost_per_10min : ℚ := (1 : ℚ) / 4

/-- The number of dryers Samantha uses -/
def num_dryers : ℕ := 3

/-- The number of minutes Samantha uses each dryer -/
def dryer_minutes : ℕ := 40

/-- The total amount Samantha spends -/
def total_spent : ℚ := 11

theorem samantha_laundry_loads :
  laundry_loads = 2 ∧
  laundry_loads * washer_cost +
    (num_dryers * (dryer_minutes / 10) * dryer_cost_per_10min) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_samantha_laundry_loads_l1629_162994


namespace NUMINAMATH_CALUDE_square_area_ratio_l1629_162920

theorem square_area_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := s₂ * Real.sqrt 2
  (s₁ ^ 2) / (s₂ ^ 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1629_162920


namespace NUMINAMATH_CALUDE_square25_on_top_l1629_162955

/-- Represents a position on the 5x5 grid -/
structure Position :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents the state of the grid after folding -/
structure FoldedGrid :=
  (top : Position)

/-- Fold operation 1: fold the top half over the bottom half -/
def fold1 (p : Position) : Position :=
  ⟨4 - p.row, p.col⟩

/-- Fold operation 2: fold the right half over the left half -/
def fold2 (p : Position) : Position :=
  ⟨p.row, 4 - p.col⟩

/-- Fold operation 3: fold along the diagonal from top-left to bottom-right -/
def fold3 (p : Position) : Position :=
  ⟨p.col, p.row⟩

/-- Fold operation 4: fold the bottom half over the top half -/
def fold4 (p : Position) : Position :=
  ⟨4 - p.row, p.col⟩

/-- Apply all fold operations in sequence -/
def applyAllFolds (p : Position) : Position :=
  fold4 (fold3 (fold2 (fold1 p)))

/-- The initial position of square 25 -/
def initialPos25 : Position :=
  ⟨4, 4⟩

/-- The theorem to be proved -/
theorem square25_on_top :
  applyAllFolds initialPos25 = ⟨0, 4⟩ :=
sorry


end NUMINAMATH_CALUDE_square25_on_top_l1629_162955


namespace NUMINAMATH_CALUDE_sum_in_base8_l1629_162973

/-- Converts a base-8 number represented as a list of digits to a natural number. -/
def fromBase8 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits. -/
def toBase8 (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: toBase8 (n / 8)

theorem sum_in_base8 :
  let a := fromBase8 [4, 7, 6, 5]
  let b := fromBase8 [5, 6, 3, 2]
  toBase8 (a + b) = [6, 2, 2, 0, 1] := by
  sorry

#eval fromBase8 [4, 7, 6, 5]
#eval fromBase8 [5, 6, 3, 2]
#eval toBase8 (fromBase8 [4, 7, 6, 5] + fromBase8 [5, 6, 3, 2])

end NUMINAMATH_CALUDE_sum_in_base8_l1629_162973


namespace NUMINAMATH_CALUDE_external_tangent_intersection_collinear_l1629_162956

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point type
abbrev Point := ℝ × ℝ

-- Define a function to check if three points are collinear
def collinear (p q r : Point) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

-- Define a function to get the intersection point of external tangents
def externalTangentIntersection (c1 c2 : Circle) : Point :=
  sorry  -- The actual implementation is not needed for the theorem statement

-- State the theorem
theorem external_tangent_intersection_collinear (γ₁ γ₂ γ₃ : Circle) :
  let X := externalTangentIntersection γ₁ γ₂
  let Y := externalTangentIntersection γ₂ γ₃
  let Z := externalTangentIntersection γ₃ γ₁
  collinear X Y Z :=
by sorry

end NUMINAMATH_CALUDE_external_tangent_intersection_collinear_l1629_162956


namespace NUMINAMATH_CALUDE_constant_function_theorem_l1629_162937

/-- A function f: ℝ → ℝ is twice differentiable if it has a second derivative -/
def TwiceDifferentiable (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ Differentiable ℝ (deriv f)

/-- The given inequality condition for the function f -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (deriv^[2] f x) * Real.cos (f x) ≥ (deriv f x)^2 * Real.sin (f x)

/-- Main theorem: If f is twice differentiable and satisfies the inequality,
    then f is a constant function -/
theorem constant_function_theorem (f : ℝ → ℝ) 
    (h1 : TwiceDifferentiable f) (h2 : SatisfiesInequality f) :
    ∃ k : ℝ, ∀ x : ℝ, f x = k := by
  sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l1629_162937


namespace NUMINAMATH_CALUDE_graduation_ceremony_attendance_l1629_162943

/-- Graduation ceremony attendance problem -/
theorem graduation_ceremony_attendance
  (graduates : ℕ)
  (chairs : ℕ)
  (parents_per_graduate : ℕ)
  (h_graduates : graduates = 50)
  (h_chairs : chairs = 180)
  (h_parents : parents_per_graduate = 2)
  (h_admins : administrators = teachers / 2) :
  teachers = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_graduation_ceremony_attendance_l1629_162943


namespace NUMINAMATH_CALUDE_non_union_women_percentage_is_75_l1629_162948

/-- Represents the composition of employees in a company -/
structure CompanyEmployees where
  total : ℕ
  men : ℕ
  unionized : ℕ
  unionized_men : ℕ

/-- The percentage of non-union employees who are women -/
def non_union_women_percentage (c : CompanyEmployees) : ℚ :=
  let non_union := c.total - c.unionized
  let non_union_men := c.men - c.unionized_men
  let non_union_women := non_union - non_union_men
  (non_union_women : ℚ) / non_union * 100

/-- Theorem stating the percentage of non-union women employees -/
theorem non_union_women_percentage_is_75 (c : CompanyEmployees) : 
  c.total > 0 →
  c.men = (52 * c.total) / 100 →
  c.unionized = (60 * c.total) / 100 →
  c.unionized_men = (70 * c.unionized) / 100 →
  non_union_women_percentage c = 75 := by
sorry

end NUMINAMATH_CALUDE_non_union_women_percentage_is_75_l1629_162948


namespace NUMINAMATH_CALUDE_least_sum_with_conditions_l1629_162954

theorem least_sum_with_conditions (m n : ℕ+) 
  (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m ^ m.val = k * n ^ n.val)
  (h3 : ¬ ∃ k : ℕ, m = k * n) :
  (∀ m' n' : ℕ+, 
    Nat.gcd (m' + n') 330 = 1 → 
    (∃ k : ℕ, m' ^ m'.val = k * n' ^ n'.val) → 
    (¬ ∃ k : ℕ, m' = k * n') → 
    m' + n' ≥ m + n) → 
  m + n = 429 := by
sorry

end NUMINAMATH_CALUDE_least_sum_with_conditions_l1629_162954


namespace NUMINAMATH_CALUDE_smallest_shift_is_sixty_l1629_162946

/-- A function with period 30 -/
def periodic_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (x + 30) = g x

/-- The smallest positive shift for g(2x) -/
def smallest_shift (g : ℝ → ℝ) (b : ℝ) : Prop :=
  (b > 0) ∧
  (∀ x : ℝ, g (2*x + b) = g (2*x)) ∧
  (∀ c : ℝ, c > 0 → (∀ x : ℝ, g (2*x + c) = g (2*x)) → b ≤ c)

theorem smallest_shift_is_sixty (g : ℝ → ℝ) :
  periodic_function g → smallest_shift g 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_shift_is_sixty_l1629_162946


namespace NUMINAMATH_CALUDE_nonnegative_difference_of_roots_l1629_162960

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 + 42*x + 336 = -48

-- Define the roots of the equation
def root1 : ℝ := -24
def root2 : ℝ := -16

-- Theorem statement
theorem nonnegative_difference_of_roots : 
  (quadratic_equation root1 ∧ quadratic_equation root2) → 
  |root1 - root2| = 8 := by
sorry

end NUMINAMATH_CALUDE_nonnegative_difference_of_roots_l1629_162960


namespace NUMINAMATH_CALUDE_present_age_ratio_l1629_162914

theorem present_age_ratio (R M : ℝ) (h1 : M - R = 7.5) (h2 : (R + 10) / (M + 10) = 2 / 3) 
  (h3 : R > 0) (h4 : M > 0) : R / M = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_present_age_ratio_l1629_162914


namespace NUMINAMATH_CALUDE_remainder_problem_l1629_162971

theorem remainder_problem (x : ℤ) : x % 84 = 25 → x % 14 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1629_162971


namespace NUMINAMATH_CALUDE_longest_side_length_l1629_162986

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  x + 2*y ≤ 4 ∧ 3*x + y ≥ 3 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the vertices of the quadrilateral
def Vertices : Set (ℝ × ℝ) :=
  {(0, 3), (0.4, 1.8), (4, 0), (0, 0)}

-- State the theorem
theorem longest_side_length :
  ∃ (a b : ℝ × ℝ), a ∈ Vertices ∧ b ∈ Vertices ∧
    (∀ (c d : ℝ × ℝ), c ∈ Vertices → d ∈ Vertices →
      Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2) ≤ Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)) ∧
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 5 :=
by sorry

end NUMINAMATH_CALUDE_longest_side_length_l1629_162986


namespace NUMINAMATH_CALUDE_inequality_range_l1629_162944

theorem inequality_range (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, (x^2 + a*x > 4*x + a - 3) ↔ (x > 3 ∨ x < -1) := by sorry

end NUMINAMATH_CALUDE_inequality_range_l1629_162944


namespace NUMINAMATH_CALUDE_monthly_order_is_168_l1629_162957

/-- The number of apples Chandler eats per week -/
def chandler_weekly : ℕ := 23

/-- The number of apples Lucy eats per week -/
def lucy_weekly : ℕ := 19

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- The number of apples Chandler and Lucy need to order for a month -/
def monthly_order : ℕ := (chandler_weekly + lucy_weekly) * weeks_per_month

theorem monthly_order_is_168 : monthly_order = 168 := by
  sorry

end NUMINAMATH_CALUDE_monthly_order_is_168_l1629_162957


namespace NUMINAMATH_CALUDE_complex_division_equality_l1629_162923

theorem complex_division_equality : (3 - Complex.I) / Complex.I = -1 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equality_l1629_162923


namespace NUMINAMATH_CALUDE_min_value_theorem_l1629_162966

theorem min_value_theorem (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 9 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1629_162966


namespace NUMINAMATH_CALUDE_bench_capacity_l1629_162905

theorem bench_capacity (num_benches : ℕ) (people_sitting : ℕ) (spaces_available : ℕ) 
  (h1 : num_benches = 50)
  (h2 : people_sitting = 80)
  (h3 : spaces_available = 120) :
  (num_benches * 4 = people_sitting + spaces_available) ∧ 
  (4 = (people_sitting + spaces_available) / num_benches) :=
by sorry

end NUMINAMATH_CALUDE_bench_capacity_l1629_162905


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l1629_162978

/-- Given a circle with center O and radius r, where arc PQ is half the circle,
    the perimeter of the shaded region formed by OP, OQ, and arc PQ is 2r + πr. -/
theorem shaded_region_perimeter (r : ℝ) (h : r > 0) : 
  2 * r + π * r = 2 * r + (1 / 2) * (2 * π * r) := by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l1629_162978


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1629_162990

-- Define the hyperbola
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  passing_point : ℝ × ℝ

-- Define the point M
structure PointM where
  on_right_branch : Bool
  dot_product_zero : Bool

-- Theorem statement
theorem hyperbola_properties (h : Hyperbola) (m : PointM) 
    (h_center : h.center = (0, 0))
    (h_foci : h.foci_on_x_axis = true)
    (h_eccentricity : h.eccentricity = Real.sqrt 2)
    (h_passing_point : h.passing_point = (4, -2 * Real.sqrt 2))
    (h_m_right : m.on_right_branch = true)
    (h_m_dot : m.dot_product_zero = true) :
    (∃ (x y : ℝ), x^2 - y^2 = 8) ∧ 
    (∃ (area : ℝ), area = 8) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1629_162990


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l1629_162927

theorem system_of_equations_solutions :
  -- System 1
  (∃ x y : ℝ, y = 2*x - 3 ∧ 3*x - 2*y = 8 ∧ x = -2 ∧ y = -7) ∧
  -- System 2
  (∃ x y : ℝ, 3*x + 4*y = 5 ∧ 5*x - 2*y = 30 ∧ x = 5 ∧ y = -5/2) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l1629_162927


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_P_l1629_162983

theorem cos_alpha_for_point_P (α : Real) :
  let P : ℝ × ℝ := (-3, 4)
  (∃ t : ℝ, t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_P_l1629_162983


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1629_162921

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ k ∈ Set.Ici 0 ∩ Set.Iio 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1629_162921


namespace NUMINAMATH_CALUDE_lemonade_solution_water_parts_l1629_162947

theorem lemonade_solution_water_parts (water_parts : ℝ) : 
  (7 : ℝ) / (water_parts + 7) > (1 : ℝ) / 10 ∧ 
  (7 : ℝ) / (water_parts + 7 - 2.1428571428571423 + 2.1428571428571423) = (1 : ℝ) / 10 → 
  water_parts = 63 := by
sorry

end NUMINAMATH_CALUDE_lemonade_solution_water_parts_l1629_162947


namespace NUMINAMATH_CALUDE_cos_225_degrees_l1629_162925

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l1629_162925


namespace NUMINAMATH_CALUDE_division_with_remainder_l1629_162987

theorem division_with_remainder (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = divisor * quotient + remainder →
  remainder < divisor →
  dividend = 11 →
  divisor = 3 →
  remainder = 2 →
  quotient = 3 := by
sorry

end NUMINAMATH_CALUDE_division_with_remainder_l1629_162987


namespace NUMINAMATH_CALUDE_group_size_problem_l1629_162903

theorem group_size_problem (T : ℝ) 
  (hat_wearers : ℝ → ℝ)
  (shoe_wearers : ℝ → ℝ)
  (both_wearers : ℝ → ℝ)
  (h1 : hat_wearers T = 0.40 * T + 60)
  (h2 : shoe_wearers T = 0.25 * T)
  (h3 : both_wearers T = 0.20 * T)
  (h4 : both_wearers T = hat_wearers T - shoe_wearers T) :
  T = 1200 := by
sorry

end NUMINAMATH_CALUDE_group_size_problem_l1629_162903


namespace NUMINAMATH_CALUDE_largest_lcm_with_18_l1629_162976

theorem largest_lcm_with_18 (n : Fin 6 → ℕ) (h : n = ![3, 6, 9, 12, 15, 18]) :
  (Finset.range 6).sup (λ i => Nat.lcm 18 (n i)) = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_18_l1629_162976


namespace NUMINAMATH_CALUDE_sin_cos_extrema_l1629_162915

theorem sin_cos_extrema (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  let m := Real.sin x - Real.cos y ^ 2
  (∀ a b, Real.sin a + Real.sin b = 1/3 → m ≤ Real.sin a - Real.cos b ^ 2) ∧
  (∃ x' y', Real.sin x' + Real.sin y' = 1/3 ∧ m = 4/9) ∧
  (∀ a b, Real.sin a + Real.sin b = 1/3 → Real.sin a - Real.cos b ^ 2 ≤ m) ∧
  (∃ x'' y'', Real.sin x'' + Real.sin y'' = 1/3 ∧ m = -11/16) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_extrema_l1629_162915


namespace NUMINAMATH_CALUDE_jacket_price_calculation_l1629_162929

def calculate_final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (coupon : ℝ) (tax_rate : ℝ) : ℝ :=
  let price_after_discount1 := original_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let price_after_coupon := price_after_discount2 - coupon
  let final_price := price_after_coupon * (1 + tax_rate)
  final_price

theorem jacket_price_calculation :
  calculate_final_price 150 0.25 0.10 10 0.10 = 100.38 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_calculation_l1629_162929


namespace NUMINAMATH_CALUDE_frustum_cone_height_l1629_162996

theorem frustum_cone_height (h : ℝ) (a_lower a_upper : ℝ) 
  (h_positive : h > 0)
  (a_lower_positive : a_lower > 0)
  (a_upper_positive : a_upper > 0)
  (h_value : h = 30)
  (a_lower_value : a_lower = 400 * Real.pi)
  (a_upper_value : a_upper = 100 * Real.pi) :
  let r_lower := (a_lower / Real.pi).sqrt
  let r_upper := (a_upper / Real.pi).sqrt
  let h_total := h * r_lower / (r_lower - r_upper)
  h_total / 3 = 15 := by sorry

end NUMINAMATH_CALUDE_frustum_cone_height_l1629_162996


namespace NUMINAMATH_CALUDE_abs_ratio_equality_l1629_162908

theorem abs_ratio_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 9*a*b) :
  |(a + b) / (a - b)| = Real.sqrt 77 / 7 := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_equality_l1629_162908


namespace NUMINAMATH_CALUDE_distinct_sides_not_isosceles_l1629_162909

-- Define a triangle with sides a, b, and c
structure Triangle (α : Type*) :=
  (a b c : α)

-- Define what it means for a triangle to be isosceles
def is_isosceles {α : Type*} [PartialOrder α] (t : Triangle α) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Theorem statement
theorem distinct_sides_not_isosceles {α : Type*} [LinearOrder α] 
  (t : Triangle α) (h_distinct : t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c) :
  ¬(is_isosceles t) :=
sorry

end NUMINAMATH_CALUDE_distinct_sides_not_isosceles_l1629_162909


namespace NUMINAMATH_CALUDE_equation_solution_l1629_162942

theorem equation_solution (m n : ℝ) (h : m ≠ n) :
  let f := fun x : ℝ => x^2 + (x + m)^2 - (x + n)^2 - 2*m*n
  (∀ x, f x = 0 ↔ x = -m + n + Real.sqrt (2*(n^2 - m*n + m^2)) ∨
                   x = -m + n - Real.sqrt (2*(n^2 - m*n + m^2))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1629_162942


namespace NUMINAMATH_CALUDE_line_through_point_l1629_162916

theorem line_through_point (k : ℚ) :
  (1 - k * 5 = -2 * (-4)) → k = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l1629_162916


namespace NUMINAMATH_CALUDE_cauchy_problem_solution_l1629_162907

noncomputable def y (x : ℝ) : ℝ := x^2/2 + x^3/6 + x^4/12 + x^5/20 + x + 1

theorem cauchy_problem_solution (x : ℝ) :
  (deriv^[2] y) x = 1 + x + x^2 + x^3 ∧
  y 0 = 1 ∧
  (deriv y) 0 = 1 := by sorry

end NUMINAMATH_CALUDE_cauchy_problem_solution_l1629_162907


namespace NUMINAMATH_CALUDE_spot_fraction_l1629_162922

theorem spot_fraction (rover_spots cisco_spots granger_spots total_spots : ℕ) 
  (f : ℚ) : 
  rover_spots = 46 →
  granger_spots = 5 * cisco_spots →
  granger_spots + cisco_spots = total_spots →
  total_spots = 108 →
  cisco_spots = f * 46 - 5 →
  f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_spot_fraction_l1629_162922


namespace NUMINAMATH_CALUDE_ellipse_x_intersection_l1629_162928

/-- Definition of the ellipse -/
def ellipse (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + (y - 3)^2) + Real.sqrt ((x - 4)^2 + (y - 1)^2) = 10

/-- Theorem stating the other x-axis intersection point of the ellipse -/
theorem ellipse_x_intersection :
  ∃ x : ℝ, x = 1 + Real.sqrt 40 ∧ ellipse x 0 ∧ ellipse 0 0 := by sorry

end NUMINAMATH_CALUDE_ellipse_x_intersection_l1629_162928


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_smallest_n_is_626_l1629_162985

theorem smallest_n_for_sqrt_difference (n : ℕ) : n ≥ 626 ↔ Real.sqrt n - Real.sqrt (n - 1) < 0.02 := by
  sorry

theorem smallest_n_is_626 : ∀ k : ℕ, k < 626 → Real.sqrt k - Real.sqrt (k - 1) ≥ 0.02 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_difference_smallest_n_is_626_l1629_162985


namespace NUMINAMATH_CALUDE_bus_variance_proof_l1629_162975

def bus_durations : List ℝ := [10, 11, 9, 9, 11]

theorem bus_variance_proof :
  let n : ℕ := bus_durations.length
  let mean : ℝ := (bus_durations.sum) / n
  let variance : ℝ := (bus_durations.map (fun x => (x - mean)^2)).sum / n
  (mean = 10 ∧ n = 5) → variance = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_bus_variance_proof_l1629_162975


namespace NUMINAMATH_CALUDE_two_machines_copies_l1629_162930

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  rate : ℕ  -- copies per minute

/-- Calculates the number of copies made by a machine in a given time -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.rate * minutes

/-- Theorem: Two copy machines working together make 2550 copies in 30 minutes -/
theorem two_machines_copies : 
  let machine1 : CopyMachine := ⟨30⟩
  let machine2 : CopyMachine := ⟨55⟩
  let total_time : ℕ := 30
  copies_made machine1 total_time + copies_made machine2 total_time = 2550 := by
  sorry


end NUMINAMATH_CALUDE_two_machines_copies_l1629_162930


namespace NUMINAMATH_CALUDE_f_iteration_result_l1629_162991

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^2 + 1 else -z^2 - 1

-- State the theorem
theorem f_iteration_result :
  f (f (f (f (2 + I)))) = 1042434 - 131072 * I :=
by sorry

end NUMINAMATH_CALUDE_f_iteration_result_l1629_162991


namespace NUMINAMATH_CALUDE_product_divisible_by_sum_iff_not_odd_prime_l1629_162974

theorem product_divisible_by_sum_iff_not_odd_prime (n : ℕ) : 
  (∃ k : ℕ, n.factorial = k * (n * (n + 1) / 2)) ↔ ¬(Nat.Prime (n + 1) ∧ Odd (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_product_divisible_by_sum_iff_not_odd_prime_l1629_162974


namespace NUMINAMATH_CALUDE_nesbitt_inequality_l1629_162992

theorem nesbitt_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 ∧
  (a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_nesbitt_inequality_l1629_162992


namespace NUMINAMATH_CALUDE_probability_sum_six_l1629_162968

def cards : Finset ℕ := {1, 2, 3, 4, 5, 6}

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)}

def total_outcomes : Finset (ℕ × ℕ) :=
  cards.product cards

theorem probability_sum_six :
  (favorable_outcomes.card : ℚ) / total_outcomes.card = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_six_l1629_162968


namespace NUMINAMATH_CALUDE_exists_five_digit_not_sum_of_beautiful_l1629_162952

/-- A beautiful number is a number consisting of identical digits. -/
def is_beautiful (n : ℕ) : Prop :=
  ∃ d : ℕ, d ≤ 9 ∧ ∃ k : ℕ, k > 0 ∧ n = d * (10^k - 1) / 9

/-- The sum of beautiful numbers with pairwise different lengths. -/
def sum_of_beautiful (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), 
    n = a * 11111 + b * 1111 + c * 111 + d * 11 + e * 1 ∧
    is_beautiful (a * 11111) ∧ 
    is_beautiful (b * 1111) ∧ 
    is_beautiful (c * 111) ∧ 
    is_beautiful (d * 11) ∧ 
    is_beautiful e

/-- Theorem: There exists a five-digit number that cannot be represented as a sum of beautiful numbers with pairwise different lengths. -/
theorem exists_five_digit_not_sum_of_beautiful : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ ¬(sum_of_beautiful n) := by
  sorry

end NUMINAMATH_CALUDE_exists_five_digit_not_sum_of_beautiful_l1629_162952


namespace NUMINAMATH_CALUDE_integer_solution_equation_l1629_162936

theorem integer_solution_equation (k x : ℤ) : 
  (Real.sqrt (39 - 6 * Real.sqrt 12) + Real.sqrt (k * x * (k * x + Real.sqrt 12) + 3) = 2 * k) → 
  (k = 3 ∨ k = 6) := by
sorry

end NUMINAMATH_CALUDE_integer_solution_equation_l1629_162936


namespace NUMINAMATH_CALUDE_binomial_expansion_property_l1629_162926

theorem binomial_expansion_property (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (2*x + 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_property_l1629_162926


namespace NUMINAMATH_CALUDE_intersection_when_a_zero_subset_condition_l1629_162939

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- Theorem 1: When a = 0, A ∩ B = {x | 0 < x < 1}
theorem intersection_when_a_zero :
  A 0 ∩ B = {x | 0 < x ∧ x < 1} := by sorry

-- Theorem 2: A ⊆ B if and only if 1 ≤ a ≤ 2
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ 1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_zero_subset_condition_l1629_162939


namespace NUMINAMATH_CALUDE_always_in_range_l1629_162980

theorem always_in_range (k : ℝ) : ∃ x : ℝ, x^2 + 2*k*x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_always_in_range_l1629_162980


namespace NUMINAMATH_CALUDE_students_neither_outstanding_nor_pioneer_l1629_162963

theorem students_neither_outstanding_nor_pioneer (total : ℕ) (outstanding : ℕ) (pioneers : ℕ) (both : ℕ)
  (h_total : total = 87)
  (h_outstanding : outstanding = 58)
  (h_pioneers : pioneers = 63)
  (h_both : both = 49) :
  total - outstanding - pioneers + both = 15 :=
by sorry

end NUMINAMATH_CALUDE_students_neither_outstanding_nor_pioneer_l1629_162963


namespace NUMINAMATH_CALUDE_salary_calculation_l1629_162902

theorem salary_calculation (salary : ℝ) : 
  (salary * (1/5 : ℝ) + salary * (1/10 : ℝ) + salary * (3/5 : ℝ) + 15000 = salary) → 
  salary = 150000 := by
sorry

end NUMINAMATH_CALUDE_salary_calculation_l1629_162902


namespace NUMINAMATH_CALUDE_x_neq_zero_necessary_not_sufficient_for_x_plus_abs_x_positive_l1629_162906

theorem x_neq_zero_necessary_not_sufficient_for_x_plus_abs_x_positive :
  (∀ x : ℝ, x + |x| > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ x + |x| ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_x_neq_zero_necessary_not_sufficient_for_x_plus_abs_x_positive_l1629_162906


namespace NUMINAMATH_CALUDE_legs_code_is_6189_l1629_162962

-- Define the type for our code
def Code := String

-- Define the mapping function
def digit_map (code : Code) (c : Char) : Nat :=
  match c with
  | 'N' => 0
  | 'E' => 1
  | 'W' => 2
  | 'C' => 3
  | 'H' => 4
  | 'A' => 5
  | 'L' => 6
  | 'G' => 8
  | 'S' => 9
  | _ => 0  -- Default case, should not occur in our problem

-- Define the function to convert a code word to a number
def code_to_number (code : Code) : Nat :=
  code.foldl (fun acc c => 10 * acc + digit_map code c) 0

-- The main theorem
theorem legs_code_is_6189 (code : Code) (h1 : code = "NEW CHALLENGES") :
  code_to_number "LEGS" = 6189 := by
  sorry


end NUMINAMATH_CALUDE_legs_code_is_6189_l1629_162962


namespace NUMINAMATH_CALUDE_equal_hikes_in_64_weeks_l1629_162933

/-- The number of weeks it takes for Camila to have hiked as many times as Steven -/
def weeks_to_equal_hikes : ℕ :=
  let camila_initial := 7
  let amanda_initial := 8 * camila_initial
  let steven_initial := amanda_initial + 15
  let david_initial := 2 * steven_initial
  let elizabeth_initial := david_initial - 10
  let camila_weekly := 4
  let amanda_weekly := 2
  let steven_weekly := 3
  let david_weekly := 5
  let elizabeth_weekly := 1
  64

theorem equal_hikes_in_64_weeks :
  let camila_initial := 7
  let amanda_initial := 8 * camila_initial
  let steven_initial := amanda_initial + 15
  let david_initial := 2 * steven_initial
  let elizabeth_initial := david_initial - 10
  let camila_weekly := 4
  let amanda_weekly := 2
  let steven_weekly := 3
  let david_weekly := 5
  let elizabeth_weekly := 1
  let w := weeks_to_equal_hikes
  camila_initial + camila_weekly * w = steven_initial + steven_weekly * w :=
by sorry

end NUMINAMATH_CALUDE_equal_hikes_in_64_weeks_l1629_162933


namespace NUMINAMATH_CALUDE_smallest_g_for_square_3150_l1629_162924

theorem smallest_g_for_square_3150 : 
  ∃ (g : ℕ), g > 0 ∧ 
  (∃ (n : ℕ), 3150 * g = n^2) ∧ 
  (∀ (k : ℕ), k > 0 → k < g → ¬∃ (m : ℕ), 3150 * k = m^2) ∧
  g = 14 := by
sorry

end NUMINAMATH_CALUDE_smallest_g_for_square_3150_l1629_162924


namespace NUMINAMATH_CALUDE_inequality_solution_l1629_162934

theorem inequality_solution (x : ℝ) : 
  (x^2 + 3*x + 3)^(5*x^3 - 3*x^2) ≤ (x^2 + 3*x + 3)^(3*x^3 + 5*x) ↔ 
  x ≤ -2 ∨ x = -1 ∨ (0 ≤ x ∧ x ≤ 5/2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1629_162934


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1629_162913

theorem trigonometric_identities :
  (Real.sin (30 * π / 180) + Real.cos (45 * π / 180) = (1 + Real.sqrt 2) / 2) ∧
  (Real.sin (60 * π / 180) ^ 2 + Real.cos (60 * π / 180) ^ 2 - Real.tan (45 * π / 180) = 0) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1629_162913


namespace NUMINAMATH_CALUDE_green_mandm_probability_l1629_162988

/-- Represents the count of M&Ms of each color -/
structure MandMCount where
  green : ℕ
  red : ℕ
  blue : ℕ
  orange : ℕ
  yellow : ℕ
  purple : ℕ
  brown : ℕ

/-- Calculates the total count of M&Ms -/
def totalCount (count : MandMCount) : ℕ :=
  count.green + count.red + count.blue + count.orange + count.yellow + count.purple + count.brown

/-- Represents the actions taken by Carter and others -/
def finalCount : MandMCount :=
  let initial := MandMCount.mk 35 25 10 15 0 0 0
  let afterCarter := MandMCount.mk (initial.green - 20) (initial.red - 8) initial.blue initial.orange 0 0 0
  let afterSister := MandMCount.mk afterCarter.green (afterCarter.red / 2) (afterCarter.blue - 5) afterCarter.orange 14 0 0
  let afterAlex := MandMCount.mk afterSister.green afterSister.red afterSister.blue (afterSister.orange - 7) (afterSister.yellow - 3) 8 0
  MandMCount.mk afterAlex.green afterAlex.red 0 afterAlex.orange afterAlex.yellow afterAlex.purple 10

/-- The main theorem to prove -/
theorem green_mandm_probability :
  (finalCount.green : ℚ) / (totalCount finalCount : ℚ) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_green_mandm_probability_l1629_162988


namespace NUMINAMATH_CALUDE_no_divisible_by_five_l1629_162995

def g (x : ℤ) : ℤ := x^2 + 5*x + 3

def T : Set ℤ := {x | 0 ≤ x ∧ x ≤ 30}

theorem no_divisible_by_five : 
  ∀ t ∈ T, ¬(g t % 5 = 0) := by
sorry

end NUMINAMATH_CALUDE_no_divisible_by_five_l1629_162995


namespace NUMINAMATH_CALUDE_expression_value_l1629_162951

theorem expression_value (x y : ℝ) (h : y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 1) :
  (Real.sqrt (48 * y) + Real.sqrt (8 * x)) * (4 * Real.sqrt (3 * y) - 2 * Real.sqrt (2 * x)) - x * y = 30 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1629_162951


namespace NUMINAMATH_CALUDE_geometric_sequence_S3_lower_bound_l1629_162977

/-- A geometric sequence with positive terms where the second term is 1 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (a 2 = 1) ∧ (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)

/-- The sum of the first three terms of a sequence -/
def S3 (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 3

theorem geometric_sequence_S3_lower_bound
  (a : ℕ → ℝ) (h : GeometricSequence a) : S3 a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_S3_lower_bound_l1629_162977


namespace NUMINAMATH_CALUDE_absolute_difference_l1629_162950

theorem absolute_difference (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 10) : |m - n| = 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_l1629_162950


namespace NUMINAMATH_CALUDE_joseph_card_distribution_l1629_162982

theorem joseph_card_distribution (initial_cards : ℕ) (cards_per_student : ℕ) (remaining_cards : ℕ) :
  initial_cards = 357 →
  cards_per_student = 23 →
  remaining_cards = 12 →
  ∃ (num_students : ℕ), num_students = 15 ∧ initial_cards = cards_per_student * num_students + remaining_cards :=
by sorry

end NUMINAMATH_CALUDE_joseph_card_distribution_l1629_162982


namespace NUMINAMATH_CALUDE_smallest_book_count_l1629_162949

theorem smallest_book_count (b : ℕ) : 
  (b % 6 = 5) ∧ (b % 8 = 7) ∧ (b % 9 = 2) → 
  (∀ n : ℕ, n < b → ¬((n % 6 = 5) ∧ (n % 8 = 7) ∧ (n % 9 = 2))) → 
  b = 119 := by
sorry

end NUMINAMATH_CALUDE_smallest_book_count_l1629_162949


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l1629_162961

theorem smallest_gcd_multiple (a b : ℕ+) (h : Nat.gcd a b = 18) :
  (Nat.gcd (12 * a) (20 * b) ≥ 72) ∧ ∃ (a₀ b₀ : ℕ+), Nat.gcd a₀ b₀ = 18 ∧ Nat.gcd (12 * a₀) (20 * b₀) = 72 := by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l1629_162961


namespace NUMINAMATH_CALUDE_triangle_ax_length_l1629_162999

-- Define the triangle ABC and point X
structure Triangle :=
  (A B C X : ℝ × ℝ)

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  let d := (λ p q : ℝ × ℝ => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt)
  d t.A t.B = 60 ∧ 
  d t.A t.C = 36 ∧
  -- C is on the angle bisector of ∠AXB
  (t.C.1 - t.X.1) / (t.A.1 - t.X.1) = (t.C.2 - t.X.2) / (t.A.2 - t.X.2) ∧
  (t.C.1 - t.X.1) / (t.B.1 - t.X.1) = (t.C.2 - t.X.2) / (t.B.2 - t.X.2)

-- Theorem statement
theorem triangle_ax_length (t : Triangle) (h : TriangleProperties t) : 
  let d := (λ p q : ℝ × ℝ => ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt)
  d t.A t.X = 20 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ax_length_l1629_162999


namespace NUMINAMATH_CALUDE_isabellas_haircuts_l1629_162972

/-- The total length of hair cut off in two haircuts -/
def total_hair_cut (initial_length first_cut_length second_cut_length : ℝ) : ℝ :=
  (initial_length - first_cut_length) + (first_cut_length - second_cut_length)

/-- Theorem: The total length of hair cut off in Isabella's two haircuts is 9 inches -/
theorem isabellas_haircuts :
  total_hair_cut 18 14 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_haircuts_l1629_162972


namespace NUMINAMATH_CALUDE_maker_funds_and_loan_repayment_l1629_162910

/-- Represents the remaining funds after n months -/
def remaining_funds (n : ℕ) : ℝ := sorry

/-- The initial borrowed capital -/
def initial_capital : ℝ := 100000

/-- Monthly profit rate -/
def profit_rate : ℝ := 0.2

/-- Monthly expense rate (rent and tax) -/
def expense_rate : ℝ := 0.1

/-- Monthly fixed expenses -/
def fixed_expenses : ℝ := 3000

/-- Annual interest rate for the bank loan -/
def annual_interest_rate : ℝ := 0.05

/-- Number of months in a year -/
def months_in_year : ℕ := 12

theorem maker_funds_and_loan_repayment :
  remaining_funds months_in_year = 194890 ∧
  remaining_funds months_in_year > initial_capital * (1 + annual_interest_rate) :=
sorry

end NUMINAMATH_CALUDE_maker_funds_and_loan_repayment_l1629_162910


namespace NUMINAMATH_CALUDE_train_crossing_time_l1629_162979

/-- Time for a train to cross another train moving in the same direction -/
theorem train_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 420)
  (h2 : length2 = 640)
  (h3 : speed1 = 72)
  (h4 : speed2 = 36) :
  (length1 + length2) / ((speed1 - speed2) * (1000 / 3600)) = 106 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1629_162979


namespace NUMINAMATH_CALUDE_equation_solution_l1629_162932

theorem equation_solution (x : ℝ) : 
  (8 * x^2 + 52 * x + 4) / (3 * x + 13) = 2 * x + 3 ↔ 
  x = (-17 + Real.sqrt 569) / 4 ∨ x = (-17 - Real.sqrt 569) / 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1629_162932


namespace NUMINAMATH_CALUDE_translation_of_line_segment_l1629_162918

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (t : Translation) (p : Point) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_of_line_segment :
  let A : Point := { x := 1, y := 1 }
  let B : Point := { x := -2, y := 0 }
  let A' : Point := { x := 4, y := 0 }
  let t : Translation := { dx := A'.x - A.x, dy := A'.y - A.y }
  let B' : Point := applyTranslation t B
  B'.x = 1 ∧ B'.y = -1 := by
  sorry

end NUMINAMATH_CALUDE_translation_of_line_segment_l1629_162918


namespace NUMINAMATH_CALUDE_expression_evaluation_l1629_162993

theorem expression_evaluation : (32 * 2 - 16) / (8 - (2 * 3)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1629_162993


namespace NUMINAMATH_CALUDE_die_roll_events_l1629_162953

-- Define the sample space for a six-sided die roll
def Ω : Type := Fin 6

-- Define the events A_k
def A (k : Fin 6) : Set Ω := {ω : Ω | ω.val + 1 = k.val + 1}

-- Define event A: rolling an even number of points
def event_A : Set Ω := A 1 ∪ A 3 ∪ A 5

-- Define event B: rolling an odd number of points
def event_B : Set Ω := A 0 ∪ A 2 ∪ A 4

-- Define event C: rolling a multiple of three
def event_C : Set Ω := A 2 ∪ A 5

-- Define event D: rolling a number greater than three
def event_D : Set Ω := A 3 ∪ A 4 ∪ A 5

theorem die_roll_events :
  (event_A = A 1 ∪ A 3 ∪ A 5) ∧
  (event_B = A 0 ∪ A 2 ∪ A 4) ∧
  (event_C = A 2 ∪ A 5) ∧
  (event_D = A 3 ∪ A 4 ∪ A 5) := by sorry

end NUMINAMATH_CALUDE_die_roll_events_l1629_162953


namespace NUMINAMATH_CALUDE_max_servings_is_eight_l1629_162919

/-- Represents the recipe requirements for 4 servings --/
structure Recipe :=
  (eggs : ℚ)
  (sugar : ℚ)
  (milk : ℚ)

/-- Represents Lisa's available ingredients --/
structure Available :=
  (eggs : ℚ)
  (sugar : ℚ)
  (milk : ℚ)

/-- Calculates the maximum number of servings possible for a given ingredient --/
def max_servings_for_ingredient (recipe_amount : ℚ) (available_amount : ℚ) : ℚ :=
  (available_amount / recipe_amount) * 4

/-- Finds the maximum number of servings possible given the recipe and available ingredients --/
def max_servings (recipe : Recipe) (available : Available) : ℚ :=
  min (max_servings_for_ingredient recipe.eggs available.eggs)
    (min (max_servings_for_ingredient recipe.sugar available.sugar)
      (max_servings_for_ingredient recipe.milk available.milk))

theorem max_servings_is_eight :
  let recipe := Recipe.mk 3 (1/2) 2
  let available := Available.mk 10 1 9
  max_servings recipe available = 8 := by
  sorry

#eval max_servings (Recipe.mk 3 (1/2) 2) (Available.mk 10 1 9)

end NUMINAMATH_CALUDE_max_servings_is_eight_l1629_162919


namespace NUMINAMATH_CALUDE_money_ratio_problem_l1629_162964

/-- Given the ratios of money between Ram and Gopal (7:17) and between Gopal and Krishan (7:17),
    and that Ram has Rs. 588, prove that Krishan has Rs. 12,065. -/
theorem money_ratio_problem (ram gopal krishan : ℚ) : 
  ram / gopal = 7 / 17 →
  gopal / krishan = 7 / 17 →
  ram = 588 →
  krishan = 12065 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_problem_l1629_162964


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l1629_162901

/-- An isosceles triangle with congruent sides of length 10 and perimeter 35 has a base of length 15 -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let congruentSide := (10 : ℝ)
    let perimeter := (35 : ℝ)
    (2 * congruentSide + base = perimeter) →
    (base = 15)

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l1629_162901


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_expected_result_l1629_162969

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | (1 - x) / x < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the expected result
def expected_result : Set ℝ := {x | -1 < x ∧ x < 0} ∪ {x | 1 < x ∧ x < 3}

-- Theorem statement
theorem A_intersect_B_eq_expected_result : A_intersect_B = expected_result := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_expected_result_l1629_162969


namespace NUMINAMATH_CALUDE_abcd_sum_l1629_162965

theorem abcd_sum (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 9)
  (eq3 : a + c + d = 20)
  (eq4 : b + c + d = 13) :
  a * b + c * d = 72 := by
  sorry

end NUMINAMATH_CALUDE_abcd_sum_l1629_162965


namespace NUMINAMATH_CALUDE_evening_campers_l1629_162984

theorem evening_campers (afternoon_campers : ℕ) (difference : ℕ) : 
  afternoon_campers = 34 → difference = 24 → afternoon_campers - difference = 10 := by
  sorry

end NUMINAMATH_CALUDE_evening_campers_l1629_162984


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l1629_162904

/-- A quadratic equation with roots 1 and -2 -/
def quadratic_equation (x : ℝ) : Prop :=
  x^2 + x - 2 = 0

theorem roots_of_quadratic : 
  (quadratic_equation 1) ∧ (quadratic_equation (-2)) := by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l1629_162904


namespace NUMINAMATH_CALUDE_kite_area_is_18_l1629_162941

/-- The area of a kite with width 6 units and height 7 units, where each unit is one inch. -/
def kite_area : ℝ := 18

/-- The width of the kite in units. -/
def kite_width : ℕ := 6

/-- The height of the kite in units. -/
def kite_height : ℕ := 7

/-- Theorem stating that the area of the kite is 18 square inches. -/
theorem kite_area_is_18 : kite_area = 18 := by sorry

end NUMINAMATH_CALUDE_kite_area_is_18_l1629_162941


namespace NUMINAMATH_CALUDE_reflected_tetrahedron_volume_l1629_162981

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  D : ℝ × ℝ × ℝ

/-- Calculates the volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Reflects a point with respect to another point -/
def reflect (point center : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Calculates the centroid of a triangle -/
def centroid (a b c : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Creates a new tetrahedron by reflecting each vertex of the original tetrahedron
    with respect to the centroid of the opposite face -/
def reflectedTetrahedron (t : Tetrahedron) : Tetrahedron :=
  let A' := reflect t.A (centroid t.B t.C t.D)
  let B' := reflect t.B (centroid t.A t.C t.D)
  let C' := reflect t.C (centroid t.A t.B t.D)
  let D' := reflect t.D (centroid t.A t.B t.C)
  ⟨A', B', C', D'⟩

/-- Theorem: The volume of the reflected tetrahedron is 125/27 times the volume of the original tetrahedron -/
theorem reflected_tetrahedron_volume (t : Tetrahedron) :
  volume (reflectedTetrahedron t) = (125 / 27) * volume t := by sorry

end NUMINAMATH_CALUDE_reflected_tetrahedron_volume_l1629_162981


namespace NUMINAMATH_CALUDE_ball_probability_relationship_l1629_162940

/-- Given a pocket with 7 balls, including 3 white and 4 black balls, 
    if x white balls and y black balls are added, and the probability 
    of drawing a white ball becomes 1/4, then y = 3x + 5 -/
theorem ball_probability_relationship (x y : ℤ) : 
  (((3 : ℚ) + x) / ((7 : ℚ) + x + y) = (1 : ℚ) / 4) → y = 3 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_relationship_l1629_162940


namespace NUMINAMATH_CALUDE_fraction_of_married_women_l1629_162989

theorem fraction_of_married_women (total : ℕ) (total_pos : total > 0) :
  let women := (58 : ℚ) / 100 * total
  let married := (60 : ℚ) / 100 * total
  let men := total - women
  let single_men := (2 : ℚ) / 3 * men
  let married_men := men - single_men
  let married_women := married - married_men
  (married_women / women) = 23 / 29 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_married_women_l1629_162989


namespace NUMINAMATH_CALUDE_average_of_twenty_digits_l1629_162997

theorem average_of_twenty_digits :
  let total_count : ℕ := 20
  let group1_count : ℕ := 14
  let group2_count : ℕ := 6
  let group1_average : ℝ := 390
  let group2_average : ℝ := 756.67
  let total_average : ℝ := (group1_count * group1_average + group2_count * group2_average) / total_count
  total_average = 500.001 := by sorry

end NUMINAMATH_CALUDE_average_of_twenty_digits_l1629_162997


namespace NUMINAMATH_CALUDE_only_proposition2_correct_l1629_162900

-- Define the propositions
def proposition1 : Prop := ∀ (p q : Prop), (p ∧ q) → (p ∨ q) ∧ ¬((p ∨ q) → (p ∧ q))

def proposition2 : Prop :=
  let p := ∃ x : ℝ, x^2 + 2*x ≤ 0
  let not_p := ∀ x : ℝ, x^2 + 2*x > 0
  (¬p) ↔ not_p

def proposition3 : Prop := ∀ (p q : Prop), p ∧ ¬q → (p ∧ ¬q) ∧ (¬p ∨ q)

def proposition4 : Prop := ∀ (p q : Prop), (¬p → q) ↔ (p → ¬q)

-- Theorem stating that only proposition2 is correct
theorem only_proposition2_correct :
  ¬proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ ¬proposition4 :=
sorry

end NUMINAMATH_CALUDE_only_proposition2_correct_l1629_162900


namespace NUMINAMATH_CALUDE_worker_payment_problem_l1629_162958

/-- Proves that the total number of days is 60 given the conditions of the worker payment problem. -/
theorem worker_payment_problem (daily_pay : ℕ) (daily_deduction : ℕ) (total_payment : ℕ) (idle_days : ℕ) :
  daily_pay = 20 →
  daily_deduction = 3 →
  total_payment = 280 →
  idle_days = 40 →
  ∃ (work_days : ℕ), daily_pay * work_days - daily_deduction * idle_days = total_payment ∧
                      work_days + idle_days = 60 :=
by sorry

end NUMINAMATH_CALUDE_worker_payment_problem_l1629_162958


namespace NUMINAMATH_CALUDE_cream_ratio_is_15_23_l1629_162912

/-- The ratio of cream in Joe's coffee to JoAnn's coffee -/
def cream_ratio : ℚ := sorry

/-- Initial amount of coffee for both Joe and JoAnn -/
def initial_coffee : ℚ := 20

/-- Amount of cream added by both Joe and JoAnn -/
def cream_added : ℚ := 3

/-- Amount of mixture Joe drank -/
def joe_drank : ℚ := 4

/-- Amount of coffee JoAnn drank before adding cream -/
def joann_drank : ℚ := 4

/-- Theorem stating the ratio of cream in Joe's coffee to JoAnn's coffee -/
theorem cream_ratio_is_15_23 : cream_ratio = 15 / 23 := by sorry

end NUMINAMATH_CALUDE_cream_ratio_is_15_23_l1629_162912


namespace NUMINAMATH_CALUDE_product_2022_sum_possibilities_l1629_162998

theorem product_2022_sum_possibilities (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧ 
  a * b * c * d * e = 2022 → 
  a + b + c + d + e = 342 ∨ 
  a + b + c + d + e = 338 ∨ 
  a + b + c + d + e = 336 ∨ 
  a + b + c + d + e = -332 :=
by sorry

end NUMINAMATH_CALUDE_product_2022_sum_possibilities_l1629_162998


namespace NUMINAMATH_CALUDE_five_circle_five_num_five_circle_seven_num_l1629_162959

-- Define the structure of the diagram
structure Diagram :=
  (n : ℕ)  -- number of circles
  (m : ℕ)  -- maximum number to be used

-- Define a valid filling of the diagram
def ValidFilling (d : Diagram) := Fin d.m → Fin d.n

-- Define the number of valid fillings
def NumValidFillings (d : Diagram) : ℕ := sorry

-- Theorem for the case with 5 circles and numbers 1 to 5
theorem five_circle_five_num :
  ∀ d : Diagram, d.n = 5 ∧ d.m = 5 → NumValidFillings d = 8 := by sorry

-- Theorem for the case with 5 circles and numbers 1 to 7
theorem five_circle_seven_num :
  ∀ d : Diagram, d.n = 5 ∧ d.m = 7 → NumValidFillings d = 48 := by sorry

end NUMINAMATH_CALUDE_five_circle_five_num_five_circle_seven_num_l1629_162959


namespace NUMINAMATH_CALUDE_lukes_trays_l1629_162931

/-- Given that Luke can carry 4 trays at a time, made 9 trips, and picked up 16 trays from the second table,
    prove that he picked up 20 trays from the first table. -/
theorem lukes_trays (trays_per_trip : ℕ) (total_trips : ℕ) (trays_second_table : ℕ)
    (h1 : trays_per_trip = 4)
    (h2 : total_trips = 9)
    (h3 : trays_second_table = 16) :
    trays_per_trip * total_trips - trays_second_table = 20 :=
by sorry

end NUMINAMATH_CALUDE_lukes_trays_l1629_162931


namespace NUMINAMATH_CALUDE_line_graph_shows_trends_l1629_162911

-- Define the types of statistical graphs
inductive StatGraph
  | BarGraph
  | LineGraph
  | PieChart
  | Histogram

-- Define the properties of statistical graphs
def comparesQuantities (g : StatGraph) : Prop :=
  g = StatGraph.BarGraph

def showsTrends (g : StatGraph) : Prop :=
  g = StatGraph.LineGraph

def displaysParts (g : StatGraph) : Prop :=
  g = StatGraph.PieChart

def showsDistribution (g : StatGraph) : Prop :=
  g = StatGraph.Histogram

-- Define the set of common statistical graphs
def commonGraphs : Set StatGraph :=
  {StatGraph.BarGraph, StatGraph.LineGraph, StatGraph.PieChart, StatGraph.Histogram}

-- Theorem: The line graph is the type that can display the trend of data
theorem line_graph_shows_trends :
  ∃ (g : StatGraph), g ∈ commonGraphs ∧ showsTrends g ∧
    ∀ (h : StatGraph), h ∈ commonGraphs → showsTrends h → h = g :=
  sorry

end NUMINAMATH_CALUDE_line_graph_shows_trends_l1629_162911


namespace NUMINAMATH_CALUDE_function_extrema_sum_l1629_162935

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 0

-- State the theorem
theorem function_extrema_sum (m : ℝ) :
  (∃ (max min : ℝ), 
    max ∈ Set.image (f m) interval ∧ 
    min ∈ Set.image (f m) interval ∧
    (∀ y ∈ Set.image (f m) interval, y ≤ max ∧ y ≥ min) ∧
    max + min = -14) →
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_function_extrema_sum_l1629_162935


namespace NUMINAMATH_CALUDE_value_of_s_l1629_162967

-- Define the variables as natural numbers
variable (a b c p q s : ℕ)

-- Define the conditions
axiom distinct_nonzero : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧
                         b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧
                         c ≠ p ∧ c ≠ q ∧ c ≠ s ∧
                         p ≠ q ∧ p ≠ s ∧
                         q ≠ s ∧
                         a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0

axiom eq1 : a + b = p
axiom eq2 : p + c = s
axiom eq3 : s + a = q
axiom eq4 : b + c + q = 18

-- Theorem to prove
theorem value_of_s : s = 9 :=
sorry

end NUMINAMATH_CALUDE_value_of_s_l1629_162967


namespace NUMINAMATH_CALUDE_fruit_shop_quantities_l1629_162917

/-- Represents the quantities and prices of fruits in a shop --/
structure FruitShop where
  apple_quantity : ℝ
  pear_quantity : ℝ
  apple_price : ℝ
  pear_price : ℝ
  apple_profit_rate : ℝ
  pear_price_ratio : ℝ

/-- Theorem stating the correct quantities of apples and pears purchased --/
theorem fruit_shop_quantities (shop : FruitShop) 
  (total_weight : shop.apple_quantity + shop.pear_quantity = 200)
  (apple_price : shop.apple_price = 15)
  (pear_price : shop.pear_price = 10)
  (apple_profit : shop.apple_profit_rate = 0.4)
  (pear_price_ratio : shop.pear_price_ratio = 2/3)
  (total_profit : 
    shop.apple_quantity * shop.apple_price * shop.apple_profit_rate + 
    shop.pear_quantity * (shop.apple_price * (1 + shop.apple_profit_rate) * shop.pear_price_ratio - shop.pear_price) = 1020) :
  shop.apple_quantity = 110 ∧ shop.pear_quantity = 90 := by
  sorry

end NUMINAMATH_CALUDE_fruit_shop_quantities_l1629_162917


namespace NUMINAMATH_CALUDE_rectangle_z_value_l1629_162938

-- Define the rectangle
def rectangle (z : ℝ) : Set (ℝ × ℝ) :=
  {(-2, z), (6, z), (-2, 4), (6, 4)}

-- Define the area of the rectangle
def area (z : ℝ) : ℝ :=
  (6 - (-2)) * (z - 4)

-- Theorem statement
theorem rectangle_z_value (z : ℝ) :
  z > 0 ∧ area z = 64 → z = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_z_value_l1629_162938


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l1629_162970

theorem geometric_progression_proof (y : ℝ) : 
  (90 + y)^2 = (30 + y) * (180 + y) → 
  y = 90 ∧ (90 + y) / (30 + y) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l1629_162970


namespace NUMINAMATH_CALUDE_sprint_competition_races_l1629_162945

/-- Calculates the number of races required to determine a champion in a sprint competition. -/
def races_to_champion (total_sprinters : ℕ) (lanes : ℕ) (eliminated_per_race : ℕ) (advance_interval : ℕ) : ℕ :=
  let regular_races := 32
  let special_races := 16
  regular_races + special_races

/-- Theorem stating that 48 races are required for the given sprint competition setup. -/
theorem sprint_competition_races :
  races_to_champion 300 8 6 3 = 48 := by
  sorry

end NUMINAMATH_CALUDE_sprint_competition_races_l1629_162945
