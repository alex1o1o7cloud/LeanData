import Mathlib

namespace largest_area_is_16_l489_48939

/-- Represents a polygon made of squares and right triangles -/
structure Polygon where
  num_squares : Nat
  num_triangles : Nat

/-- Calculates the area of a polygon -/
def area (p : Polygon) : ℝ :=
  4 * p.num_squares + 2 * p.num_triangles

/-- The set of all possible polygons in our problem -/
def polygon_set : Set Polygon :=
  { p | p.num_squares + p.num_triangles ≤ 4 }

theorem largest_area_is_16 :
  ∃ (p : Polygon), p ∈ polygon_set ∧ area p = 16 ∧ ∀ (q : Polygon), q ∈ polygon_set → area q ≤ 16 := by
  sorry

end largest_area_is_16_l489_48939


namespace f_neg_two_eq_three_l489_48909

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

-- State the theorem
theorem f_neg_two_eq_three 
  (a b c : ℝ) 
  (h : f a b c 2 = -1) : 
  f a b c (-2) = 3 := by
sorry

end f_neg_two_eq_three_l489_48909


namespace syrup_volume_in_tank_syrup_volume_specific_l489_48933

/-- The volume of syrup in a partially filled cylindrical tank -/
theorem syrup_volume_in_tank (tank_height : ℝ) (tank_diameter : ℝ) 
  (fill_ratio : ℝ) (syrup_ratio : ℝ) : ℝ :=
  let tank_radius : ℝ := tank_diameter / 2
  let liquid_height : ℝ := fill_ratio * tank_height
  let liquid_volume : ℝ := Real.pi * tank_radius^2 * liquid_height
  let syrup_volume : ℝ := liquid_volume * syrup_ratio / (1 + 1/syrup_ratio)
  syrup_volume

/-- The volume of syrup in the specific tank described in the problem -/
theorem syrup_volume_specific : 
  ∃ (ε : ℝ), abs (syrup_volume_in_tank 9 4 (1/3) (1/5) - 6.28) < ε ∧ ε < 0.01 := by
  sorry

end syrup_volume_in_tank_syrup_volume_specific_l489_48933


namespace quadratic_real_roots_l489_48919

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 4 * x + 2 = 0) ↔ k ≤ 2 ∧ k ≠ 0 := by
  sorry

end quadratic_real_roots_l489_48919


namespace product_of_repeating_decimals_l489_48971

/-- The first repeating decimal 0.030303... -/
def decimal1 : ℚ := 1 / 33

/-- The second repeating decimal 0.363636... -/
def decimal2 : ℚ := 4 / 11

/-- Theorem stating that the product of the two repeating decimals is 4/363 -/
theorem product_of_repeating_decimals :
  decimal1 * decimal2 = 4 / 363 := by sorry

end product_of_repeating_decimals_l489_48971


namespace number_solution_l489_48960

theorem number_solution : 
  ∀ (number : ℝ), (number * (-8) = 1600) → number = -200 := by
  sorry

end number_solution_l489_48960


namespace degree_three_iff_c_eq_neg_seven_fifteenths_l489_48954

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 7*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 5 - 2*x - 6*x^3 + 15*x^4

/-- The combined polynomial h(x, c) = f(x) + c*g(x) -/
def h (x c : ℝ) : ℝ := f x + c * g x

/-- Theorem stating that h(x, c) has degree 3 if and only if c = -7/15 -/
theorem degree_three_iff_c_eq_neg_seven_fifteenths :
  (∀ x, h x (-7/15) = 1 - 12*x + 3*x^2 - 6/5*x^3) ∧
  (∀ c, (∀ x, h x c = 1 - 12*x + 3*x^2 - 6/5*x^3) → c = -7/15) :=
by sorry

end degree_three_iff_c_eq_neg_seven_fifteenths_l489_48954


namespace sqrt_88_plus_42sqrt3_form_l489_48973

theorem sqrt_88_plus_42sqrt3_form : ∃ (a b c : ℤ), 
  (Real.sqrt (88 + 42 * Real.sqrt 3) = a + b * Real.sqrt c) ∧ 
  (∀ (k : ℕ), k > 1 → ¬(∃ (m : ℕ), c = k^2 * m)) ∧
  (a + b + c = 13) := by
  sorry

end sqrt_88_plus_42sqrt3_form_l489_48973


namespace tangent_line_problem_l489_48975

-- Define f as a real-valued function
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_problem (h : ∀ y, y = f 2 → y = 2 + 4) : 
  f 2 + deriv f 2 = 7 := by
  sorry

end tangent_line_problem_l489_48975


namespace unique_four_digit_square_l489_48922

def is_consecutive_digits (n : ℕ) : Prop :=
  ∃ (a : ℕ), a < 10 ∧ n = a * 1000 + (a + 1) * 100 + (a + 2) * 10 + (a + 3)

def swap_first_two_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let rest := n % 100
  d2 * 1000 + d1 * 100 + rest

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem unique_four_digit_square : 
  ∀ (n : ℕ), 1000 ≤ n ∧ n < 10000 →
    (is_consecutive_digits n ∧ 
     is_perfect_square (swap_first_two_digits n)) ↔ 
    n = 4356 := by sorry

end unique_four_digit_square_l489_48922


namespace eliminate_denominators_l489_48989

theorem eliminate_denominators (x : ℚ) : 
  (2*x - 1) / 2 = 1 - (3 - x) / 3 ↔ 3*(2*x - 1) = 6 - 2*(3 - x) := by
sorry

end eliminate_denominators_l489_48989


namespace percentage_multiplication_l489_48965

theorem percentage_multiplication : (10 / 100 * 10) * (20 / 100 * 20) = 4 := by
  sorry

end percentage_multiplication_l489_48965


namespace expression_simplification_l489_48967

theorem expression_simplification (a b : ℝ) : 
  3 * a - 4 * b + 2 * a^2 - (7 * a - 2 * a^2 + 3 * b - 5) = -4 * a - 7 * b + 4 * a^2 + 5 := by
sorry

end expression_simplification_l489_48967


namespace two_point_six_million_scientific_notation_l489_48908

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem two_point_six_million_scientific_notation :
  toScientificNotation 2600000 = ScientificNotation.mk 2.6 6 sorry := by
  sorry

end two_point_six_million_scientific_notation_l489_48908


namespace seven_lines_regions_l489_48972

/-- The number of regions formed by n lines in a plane, where no two are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1)) / 2

/-- The property that no two lines are parallel and no three are concurrent -/
def general_position (n : ℕ) : Prop := n > 0

theorem seven_lines_regions :
  general_position 7 → num_regions 7 = 29 := by
  sorry

end seven_lines_regions_l489_48972


namespace imaginary_part_of_z_l489_48996

theorem imaginary_part_of_z (z : ℂ) : z = Complex.I * (1 - 3 * Complex.I) → z.im = 1 := by
  sorry

end imaginary_part_of_z_l489_48996


namespace gcf_lcm_sum_15_20_30_l489_48959

/-- The sum of the greatest common factor and the least common multiple of 15, 20, and 30 is 65 -/
theorem gcf_lcm_sum_15_20_30 : 
  (Nat.gcd 15 (Nat.gcd 20 30) + Nat.lcm 15 (Nat.lcm 20 30)) = 65 := by
  sorry

end gcf_lcm_sum_15_20_30_l489_48959


namespace segment_length_l489_48950

/-- The length of a segment with endpoints (1,2) and (9,16) is 2√65 -/
theorem segment_length : Real.sqrt ((9 - 1)^2 + (16 - 2)^2) = 2 * Real.sqrt 65 := by
  sorry

end segment_length_l489_48950


namespace scientific_notation_of_ten_billion_thirty_million_l489_48943

theorem scientific_notation_of_ten_billion_thirty_million :
  (10030000000 : ℝ) = 1.003 * (10 : ℝ)^10 :=
by sorry

end scientific_notation_of_ten_billion_thirty_million_l489_48943


namespace set_equality_implies_a_equals_one_l489_48949

theorem set_equality_implies_a_equals_one (a : ℝ) :
  let A : Set ℝ := {1, -2, a^2 - 1}
  let B : Set ℝ := {1, a^2 - 3*a, 0}
  A = B → a = 1 := by
sorry

end set_equality_implies_a_equals_one_l489_48949


namespace simplify_and_evaluate_l489_48994

theorem simplify_and_evaluate (a b : ℚ) (ha : a = 2) (hb : b = 2/5) :
  (2*a + b)^2 - (3*b + 2*a) * (2*a - 3*b) = 24/5 := by
  sorry

end simplify_and_evaluate_l489_48994


namespace binomial_probability_theorem_l489_48935

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial random variable -/
def expected_value (X : BinomialRV) : ℝ := X.n * X.p

/-- The probability mass function of a binomial random variable -/
def binomial_pmf (X : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose X.n k : ℝ) * X.p^k * (1 - X.p)^(X.n - k)

/-- The main theorem -/
theorem binomial_probability_theorem (X : BinomialRV) 
  (h_p : X.p = 1/3) 
  (h_ev : expected_value X = 2) : 
  binomial_pmf X 2 = 80/243 := by
  sorry

end binomial_probability_theorem_l489_48935


namespace circle_center_polar_coordinates_l489_48977

/-- The polar coordinates of the center of the circle ρ = √2(cos θ + sin θ) are (1, π/4) -/
theorem circle_center_polar_coordinates :
  let ρ : ℝ → ℝ := λ θ => Real.sqrt 2 * (Real.cos θ + Real.sin θ)
  ∃ r θ_c, r = 1 ∧ θ_c = π/4 ∧
    ∀ θ, ρ θ = 2 * Real.cos (θ - θ_c) :=
by sorry

end circle_center_polar_coordinates_l489_48977


namespace square_b_minus_d_l489_48903

theorem square_b_minus_d (a b c d : ℤ) 
  (eq1 : a - b - c + d = 18) 
  (eq2 : a + b - c - d = 6) : 
  (b - d)^2 = 36 := by
sorry

end square_b_minus_d_l489_48903


namespace min_difference_unit_complex_l489_48905

theorem min_difference_unit_complex (z w : ℂ) 
  (hz : Complex.abs z = 1) 
  (hw : Complex.abs w = 1) 
  (h_sum : 1 ≤ Complex.abs (z + w) ∧ Complex.abs (z + w) ≤ Real.sqrt 2) : 
  Real.sqrt 2 ≤ Complex.abs (z - w) := by
  sorry

end min_difference_unit_complex_l489_48905


namespace pencil_distribution_l489_48920

/-- Given:
  - total_pencils: The total number of pencils
  - original_classes: The original number of classes
  - remaining_pencils: The number of pencils remaining after distribution
  - pencil_difference: The difference in pencils per class compared to the original plan
  This theorem proves that the actual number of classes is 11
-/
theorem pencil_distribution
  (total_pencils : ℕ)
  (original_classes : ℕ)
  (remaining_pencils : ℕ)
  (pencil_difference : ℕ)
  (h1 : total_pencils = 172)
  (h2 : original_classes = 4)
  (h3 : remaining_pencils = 7)
  (h4 : pencil_difference = 28)
  : ∃ (actual_classes : ℕ),
    actual_classes > original_classes ∧
    (total_pencils - remaining_pencils) / actual_classes + pencil_difference = total_pencils / original_classes ∧
    actual_classes = 11 :=
sorry

end pencil_distribution_l489_48920


namespace cafeteria_pies_l489_48962

/-- Given a cafeteria with initial apples, apples handed out, and apples required per pie,
    calculate the number of pies that can be made. -/
def calculate_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

/-- Theorem stating that with 47 initial apples, 27 handed out, and 4 apples per pie,
    the number of pies that can be made is 5. -/
theorem cafeteria_pies :
  calculate_pies 47 27 4 = 5 := by
  sorry

end cafeteria_pies_l489_48962


namespace jeffreys_poultry_farm_l489_48981

/-- The number of roosters for every 3 hens on Jeffrey's poultry farm -/
def roosters_per_three_hens : ℕ := by sorry

theorem jeffreys_poultry_farm :
  let total_hens : ℕ := 12
  let chicks_per_hen : ℕ := 5
  let total_chickens : ℕ := 76
  roosters_per_three_hens = 1 := by sorry

end jeffreys_poultry_farm_l489_48981


namespace garden_fence_length_l489_48927

/-- The total length of a fence surrounding a sector-shaped garden -/
def fence_length (radius : ℝ) (central_angle : ℝ) : ℝ :=
  radius * central_angle + 2 * radius

/-- Proof that the fence length for a garden with radius 30m and central angle 120° is 20π + 60m -/
theorem garden_fence_length :
  fence_length 30 (2 * Real.pi / 3) = 20 * Real.pi + 60 := by
  sorry

end garden_fence_length_l489_48927


namespace jane_well_days_l489_48999

/-- Represents Jane's performance levels --/
inductive Performance
  | Poor
  | Well
  | Excellent

/-- Returns the daily earnings based on performance --/
def dailyEarnings (p : Performance) : ℕ :=
  match p with
  | Performance.Poor => 2
  | Performance.Well => 4
  | Performance.Excellent => 6

/-- Represents Jane's work record over 15 days --/
structure WorkRecord :=
  (poorDays : ℕ)
  (wellDays : ℕ)
  (excellentDays : ℕ)
  (total_days : poorDays + wellDays + excellentDays = 15)
  (excellent_poor_relation : excellentDays = poorDays + 4)
  (total_earnings : poorDays * 2 + wellDays * 4 + excellentDays * 6 = 66)

/-- Theorem stating that Jane performed well for 11 days --/
theorem jane_well_days (record : WorkRecord) : record.wellDays = 11 := by
  sorry

end jane_well_days_l489_48999


namespace select_gloves_count_l489_48961

/-- The number of ways to select 4 gloves from 6 different pairs such that exactly two of the selected gloves are of the same color -/
def select_gloves : ℕ :=
  (Nat.choose 6 1) * (Nat.choose 10 2 - 5)

/-- Theorem stating that the number of ways to select the gloves is 240 -/
theorem select_gloves_count : select_gloves = 240 := by
  sorry

end select_gloves_count_l489_48961


namespace pascal_zero_property_l489_48948

/-- Pascal's triangle binomial coefficient -/
def pascal (n k : ℕ) : ℕ := Nat.choose n k

/-- Property that all elements except extremes are zero in a row -/
def all_zero_except_extremes (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k < n → pascal n k = 0

/-- Theorem: If s-th row has all elements zero except extremes,
    then s^k-th rows also have this property for all k ≥ 2 -/
theorem pascal_zero_property (s : ℕ) (hs : s > 1) :
  all_zero_except_extremes s →
  ∀ k, k ≥ 2 → all_zero_except_extremes (s^k) :=
by sorry

end pascal_zero_property_l489_48948


namespace equal_roots_cubic_l489_48925

theorem equal_roots_cubic (k : ℝ) :
  (∃ a b : ℝ, (3 * a^3 + 9 * a^2 - 150 * a + k = 0) ∧
              (3 * b^3 + 9 * b^2 - 150 * b + k = 0) ∧
              (a ≠ b)) ∧
  (∃ x : ℝ, (3 * x^3 + 9 * x^2 - 150 * x + k = 0) ∧
            (∃ y : ℝ, y ≠ x ∧ 3 * y^3 + 9 * y^2 - 150 * y + k = 0)) ∧
  (k > 0) →
  k = 950 / 27 :=
sorry

end equal_roots_cubic_l489_48925


namespace total_slices_eq_207_l489_48915

/-- The total number of watermelon and fruit slices at a family picnic --/
def total_slices : ℕ :=
  let danny_watermelon := 3 * 10
  let sister_watermelon := 1 * 15
  let cousin_watermelon := 2 * 8
  let cousin_apples := 5 * 4
  let aunt_watermelon := 4 * 12
  let aunt_oranges := 7 * 6
  let grandfather_watermelon := 1 * 6
  let grandfather_pineapples := 3 * 10
  danny_watermelon + sister_watermelon + cousin_watermelon + cousin_apples +
  aunt_watermelon + aunt_oranges + grandfather_watermelon + grandfather_pineapples

theorem total_slices_eq_207 : total_slices = 207 := by
  sorry

end total_slices_eq_207_l489_48915


namespace one_real_root_l489_48986

-- Define the determinant function
def det (x c b d : ℝ) : ℝ := x^3 + (c^2 + d^2) * x

-- State the theorem
theorem one_real_root
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0) :
  ∃! x : ℝ, det x c b d = 0 :=
sorry

end one_real_root_l489_48986


namespace max_mice_caught_max_mice_achievable_l489_48991

/-- Production Possibility Frontier for a male kitten -/
def male_ppf (k : ℝ) : ℝ := 80 - 4 * k

/-- Production Possibility Frontier for a female kitten -/
def female_ppf (k : ℝ) : ℝ := 16 - 0.25 * k

/-- The maximum number of mice that can be caught by any combination of two kittens -/
def max_mice : ℝ := 160

/-- Theorem stating that the maximum number of mice that can be caught by any combination
    of two kittens is 160 -/
theorem max_mice_caught :
  ∀ k₁ k₂ : ℝ, k₁ ≥ 0 → k₂ ≥ 0 →
  (male_ppf k₁ + male_ppf k₂ ≤ max_mice) ∧
  (male_ppf k₁ + female_ppf k₂ ≤ max_mice) ∧
  (female_ppf k₁ + female_ppf k₂ ≤ max_mice) :=
sorry

/-- Theorem stating that there exist values of k₁ and k₂ for which the maximum is achieved -/
theorem max_mice_achievable :
  ∃ k₁ k₂ : ℝ, k₁ ≥ 0 ∧ k₂ ≥ 0 ∧ male_ppf k₁ + male_ppf k₂ = max_mice :=
sorry

end max_mice_caught_max_mice_achievable_l489_48991


namespace specific_tetrahedron_volume_l489_48906

/-- Represents a tetrahedron with vertices P, Q, R, and S. -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths. -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 3√2. -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := 4,
    RS := (15/4) * Real.sqrt 2
  }
  tetrahedronVolume t = 3 * Real.sqrt 2 := by
  sorry

end specific_tetrahedron_volume_l489_48906


namespace rhombus_area_fraction_l489_48930

/-- Represents a point on a 2D grid -/
structure Point where
  x : Int
  y : Int

/-- Represents a rhombus on a 2D grid -/
structure Rhombus where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the area of a rhombus given its vertices -/
def rhombusArea (r : Rhombus) : ℚ :=
  1 -- placeholder for the actual calculation

/-- Calculates the area of a square grid -/
def gridArea (side : ℕ) : ℕ :=
  side * side

/-- The main theorem to prove -/
theorem rhombus_area_fraction :
  let r : Rhombus := {
    v1 := { x := 3, y := 2 },
    v2 := { x := 4, y := 3 },
    v3 := { x := 3, y := 4 },
    v4 := { x := 2, y := 3 }
  }
  let gridSide : ℕ := 6
  rhombusArea r / gridArea gridSide = 1 / 36 := by
  sorry

end rhombus_area_fraction_l489_48930


namespace lakes_country_islands_l489_48941

/-- A connected planar graph representing the lakes and canals system -/
structure LakeSystem where
  V : ℕ  -- number of vertices (lakes)
  E : ℕ  -- number of edges (canals)
  is_connected : Bool
  is_planar : Bool

/-- The number of islands in a lake system -/
def num_islands (sys : LakeSystem) : ℕ :=
  sys.V - sys.E + 2 - 1

/-- Theorem stating the number of islands in the given lake system -/
theorem lakes_country_islands (sys : LakeSystem) 
  (h1 : sys.V = 7)
  (h2 : sys.E = 10)
  (h3 : sys.is_connected = true)
  (h4 : sys.is_planar = true) :
  num_islands sys = 4 := by
  sorry

#eval num_islands ⟨7, 10, true, true⟩

end lakes_country_islands_l489_48941


namespace pick_one_book_theorem_l489_48937

/-- The number of ways to pick one book from a shelf with given numbers of math, physics, and chemistry books -/
def ways_to_pick_one_book (math_books : ℕ) (physics_books : ℕ) (chemistry_books : ℕ) : ℕ :=
  math_books + physics_books + chemistry_books

/-- Theorem stating that picking one book from a shelf with 5 math books, 4 physics books, and 5 chemistry books can be done in 14 ways -/
theorem pick_one_book_theorem : ways_to_pick_one_book 5 4 5 = 14 := by
  sorry

end pick_one_book_theorem_l489_48937


namespace area_of_ABCM_l489_48983

/-- A 12-sided polygon with specific properties -/
structure TwelveSidedPolygon where
  /-- The length of each side of the polygon -/
  side_length : ℝ
  /-- The property that each two consecutive sides form a right angle -/
  right_angles : Bool

/-- The intersection point of two diagonals in the polygon -/
def IntersectionPoint (p : TwelveSidedPolygon) := Unit

/-- A quadrilateral formed by three vertices of the polygon and the intersection point -/
def Quadrilateral (p : TwelveSidedPolygon) (m : IntersectionPoint p) := Unit

/-- The area of a quadrilateral -/
def area (q : Quadrilateral p m) : ℝ := sorry

/-- Theorem stating the area of quadrilateral ABCM in the given polygon -/
theorem area_of_ABCM (p : TwelveSidedPolygon) (m : IntersectionPoint p) 
  (q : Quadrilateral p m) (h1 : p.side_length = 4) (h2 : p.right_angles = true) : 
  area q = 88 / 5 := by sorry

end area_of_ABCM_l489_48983


namespace complex_equation_solution_l489_48952

theorem complex_equation_solution :
  ∀ z : ℂ, (z - Complex.I) * (2 - Complex.I) = 5 → z = 2 + 2 * Complex.I :=
by
  sorry

end complex_equation_solution_l489_48952


namespace total_candies_l489_48901

/-- The total number of candies in a jar, given the number of red and blue candies -/
theorem total_candies (red : ℕ) (blue : ℕ) (h1 : red = 145) (h2 : blue = 3264) : 
  red + blue = 3409 :=
by sorry

end total_candies_l489_48901


namespace x4_plus_y4_equals_135_point_5_l489_48934

theorem x4_plus_y4_equals_135_point_5 (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 14) : 
  x^4 + y^4 = 135.5 := by
sorry

end x4_plus_y4_equals_135_point_5_l489_48934


namespace box_volume_is_four_cubic_feet_l489_48976

/-- Calculates the internal volume of a box in cubic feet given its external dimensions in inches and wall thickness -/
def internal_volume (length width height wall_thickness : ℚ) : ℚ :=
  let internal_length := length - 2 * wall_thickness
  let internal_width := width - 2 * wall_thickness
  let internal_height := height - 2 * wall_thickness
  (internal_length * internal_width * internal_height) / 1728

/-- Proves that the internal volume of the specified box is 4 cubic feet -/
theorem box_volume_is_four_cubic_feet :
  internal_volume 26 26 14 1 = 4 := by
  sorry

end box_volume_is_four_cubic_feet_l489_48976


namespace books_sold_in_garage_sale_l489_48995

theorem books_sold_in_garage_sale 
  (initial_books : ℝ)
  (books_given_to_friend : ℝ)
  (final_books : ℝ)
  (h1 : initial_books = 284.5)
  (h2 : books_given_to_friend = 63.7)
  (h3 : final_books = 112.3) :
  initial_books - books_given_to_friend - final_books = 108.5 :=
by
  sorry

#eval 284.5 - 63.7 - 112.3  -- This should evaluate to 108.5

end books_sold_in_garage_sale_l489_48995


namespace broken_beads_count_l489_48964

/-- Calculates the number of necklaces with broken beads -/
def necklaces_with_broken_beads (initial_count : ℕ) (purchased : ℕ) (gifted : ℕ) (final_count : ℕ) : ℕ :=
  initial_count + purchased - gifted - final_count

theorem broken_beads_count :
  necklaces_with_broken_beads 50 5 15 37 = 3 := by
  sorry

end broken_beads_count_l489_48964


namespace modulus_of_one_minus_i_times_one_plus_i_l489_48917

theorem modulus_of_one_minus_i_times_one_plus_i : 
  Complex.abs ((1 - Complex.I) * (1 + Complex.I)) = 2 := by sorry

end modulus_of_one_minus_i_times_one_plus_i_l489_48917


namespace bank_interest_equation_l489_48970

theorem bank_interest_equation (initial_deposit : ℝ) (interest_tax_rate : ℝ) 
  (total_amount : ℝ) (annual_interest_rate : ℝ) 
  (h1 : initial_deposit = 2500)
  (h2 : interest_tax_rate = 0.2)
  (h3 : total_amount = 2650) :
  initial_deposit * (1 + annual_interest_rate * (1 - interest_tax_rate)) = total_amount :=
by sorry

end bank_interest_equation_l489_48970


namespace not_a_gt_b_l489_48921

theorem not_a_gt_b (A B : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) 
  (h : A / (1 / 5) = B * (1 / 4)) : A ≤ B := by
  sorry

end not_a_gt_b_l489_48921


namespace special_sequence_a9_l489_48923

/-- A sequence of positive integers satisfying the given property -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ p q : ℕ, a (p + q) = a p + a q)

theorem special_sequence_a9 (a : ℕ → ℕ) (h : SpecialSequence a) (h2 : a 2 = 4) : 
  a 9 = 18 := by
  sorry

end special_sequence_a9_l489_48923


namespace find_A_l489_48940

theorem find_A (A : ℕ) (B : ℕ) (h1 : 0 ≤ B ∧ B ≤ 999) 
  (h2 : 1000 * A + B = A * (A + 1) / 2) : A = 1999 := by
  sorry

end find_A_l489_48940


namespace walking_speed_problem_l489_48966

/-- 
Given two people walking in opposite directions for 12 hours, 
with one walking at 3 km/hr and the distance between them after 12 hours being 120 km, 
prove that the speed of the other person is 7 km/hr.
-/
theorem walking_speed_problem (v : ℝ) : 
  v > 0 → -- Assuming positive speed
  (v + 3) * 12 = 120 → 
  v = 7 := by
sorry

end walking_speed_problem_l489_48966


namespace rose_count_l489_48988

theorem rose_count (lilies roses tulips : ℕ) : 
  roses = lilies + 22 →
  roses = tulips - 20 →
  lilies + roses + tulips = 100 →
  roses = 34 := by
sorry

end rose_count_l489_48988


namespace noemi_roulette_loss_l489_48955

/-- Noemi's gambling problem -/
theorem noemi_roulette_loss 
  (initial_amount : ℕ) 
  (final_amount : ℕ) 
  (blackjack_loss : ℕ) 
  (h1 : initial_amount = 1700)
  (h2 : final_amount = 800)
  (h3 : blackjack_loss = 500) :
  initial_amount - final_amount - blackjack_loss = 400 := by
sorry

end noemi_roulette_loss_l489_48955


namespace hyperbola_cosh_sinh_l489_48963

theorem hyperbola_cosh_sinh (t : ℝ) : (Real.cosh t)^2 - (Real.sinh t)^2 = 1 := by
  sorry

end hyperbola_cosh_sinh_l489_48963


namespace difference_of_squares_example_l489_48953

theorem difference_of_squares_example : (25 + 15)^2 - (25 - 15)^2 = 1500 := by
  sorry

end difference_of_squares_example_l489_48953


namespace matrix_addition_result_l489_48978

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -2; -3, 5]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 0; 7, -8]

theorem matrix_addition_result : A + B = !![-2, -2; 4, -3] := by sorry

end matrix_addition_result_l489_48978


namespace bound_step_difference_is_10_l489_48918

/-- The number of steps Martha takes between consecutive lamp posts -/
def martha_steps : ℕ := 50

/-- The number of bounds Percy takes between consecutive lamp posts -/
def percy_bounds : ℕ := 15

/-- The total number of lamp posts -/
def total_posts : ℕ := 51

/-- The distance between the first and last lamp post in feet -/
def total_distance : ℕ := 10560

/-- The difference between Percy's bound length and Martha's step length in feet -/
def bound_step_difference : ℚ := 10

theorem bound_step_difference_is_10 :
  (total_distance : ℚ) / ((total_posts - 1) * percy_bounds) -
  (total_distance : ℚ) / ((total_posts - 1) * martha_steps) =
  bound_step_difference := by sorry

end bound_step_difference_is_10_l489_48918


namespace geometric_series_sum_l489_48968

theorem geometric_series_sum (x : ℝ) :
  (|x| < 1) →
  (∑' n, x^n = 16) →
  x = 15/16 := by
sorry

end geometric_series_sum_l489_48968


namespace brown_class_points_l489_48936

theorem brown_class_points (william_points mr_adams_points daniel_points : ℕ)
  (mean_points : ℚ) (total_classes : ℕ) :
  william_points = 50 →
  mr_adams_points = 57 →
  daniel_points = 57 →
  mean_points = 53.3 →
  total_classes = 4 →
  ∃ (brown_points : ℕ),
    (william_points + mr_adams_points + daniel_points + brown_points) / total_classes = mean_points ∧
    brown_points = 49 :=
by sorry

end brown_class_points_l489_48936


namespace distance_between_trees_l489_48979

theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 360 →
  num_trees = 31 →
  yard_length / (num_trees - 1) = 12 := by
  sorry

end distance_between_trees_l489_48979


namespace slope_of_line_l489_48980

theorem slope_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : 
  (y - 4) / x = -4 / 7 := by
sorry

end slope_of_line_l489_48980


namespace M_closed_under_multiplication_l489_48913

def M : Set ℕ := {n | ∃ m : ℕ, m > 0 ∧ n = m^2}

theorem M_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ M → b ∈ M → (a * b) ∈ M :=
by
  sorry

end M_closed_under_multiplication_l489_48913


namespace teachers_separation_probability_l489_48998

/-- The number of students in the group photo arrangement. -/
def num_students : ℕ := 5

/-- The number of teachers in the group photo arrangement. -/
def num_teachers : ℕ := 2

/-- The total number of people in the group photo arrangement. -/
def total_people : ℕ := num_students + num_teachers

/-- The probability of arranging the group such that the two teachers
    are not at the ends and not adjacent to each other. -/
def probability_teachers_separated : ℚ :=
  (num_students.factorial * (num_students + 1).choose 2) / total_people.factorial

theorem teachers_separation_probability :
  probability_teachers_separated = 2 / 7 := by
  sorry

end teachers_separation_probability_l489_48998


namespace min_value_expression_l489_48907

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 2) :
  (1/a + 1/(2*b) + 4*a*b) ≥ 4 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 2 ∧ (1/a + 1/(2*b) + 4*a*b) = 4 :=
sorry

end min_value_expression_l489_48907


namespace distribute_identical_items_l489_48910

theorem distribute_identical_items (n : ℕ) (k : ℕ) :
  n = 10 → k = 3 → Nat.choose (n + k - 1) k = 220 := by
  sorry

end distribute_identical_items_l489_48910


namespace convex_pentagon_probability_l489_48929

/-- Given seven points on a circle -/
def num_points : ℕ := 7

/-- Number of chords that can be formed from seven points -/
def total_chords : ℕ := num_points.choose 2

/-- Number of chords selected -/
def selected_chords : ℕ := 5

/-- The probability of forming a convex pentagon -/
def probability : ℚ := (num_points.choose selected_chords) / (total_chords.choose selected_chords)

/-- Theorem: The probability of forming a convex pentagon by randomly selecting
    five chords from seven points on a circle is 1/969 -/
theorem convex_pentagon_probability : probability = 1 / 969 := by
  sorry

end convex_pentagon_probability_l489_48929


namespace sequence_length_l489_48990

theorem sequence_length (n : ℕ) (b : ℕ → ℝ) : 
  n > 0 ∧
  b 0 = 41 ∧
  b 1 = 68 ∧
  b n = 0 ∧
  (∀ k : ℕ, 1 ≤ k ∧ k < n → b (k + 1) = b (k - 1) - 4 / b k) →
  n = 698 := by
sorry

end sequence_length_l489_48990


namespace determinant_zero_l489_48931

theorem determinant_zero (α β : Real) : 
  let M : Matrix (Fin 3) (Fin 3) Real := ![![0, Real.cos α, -Real.sin α],
                                           ![-Real.cos α, 0, Real.cos β],
                                           ![Real.sin α, -Real.cos β, 0]]
  Matrix.det M = 0 := by
sorry

end determinant_zero_l489_48931


namespace grade_assignment_count_l489_48958

theorem grade_assignment_count (num_students : ℕ) (num_grades : ℕ) :
  num_students = 10 →
  num_grades = 4 →
  (num_grades ^ num_students : ℕ) = 1048576 := by
  sorry

end grade_assignment_count_l489_48958


namespace total_snow_is_0_53_l489_48944

/-- The amount of snow on Monday in inches -/
def snow_monday : ℝ := 0.32

/-- The amount of snow on Tuesday in inches -/
def snow_tuesday : ℝ := 0.21

/-- The total amount of snow on Monday and Tuesday combined -/
def total_snow : ℝ := snow_monday + snow_tuesday

/-- Theorem stating that the total snow on Monday and Tuesday is 0.53 inches -/
theorem total_snow_is_0_53 : total_snow = 0.53 := by
  sorry

end total_snow_is_0_53_l489_48944


namespace difference_of_squares_81_49_l489_48984

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end difference_of_squares_81_49_l489_48984


namespace two_disco_cassettes_probability_l489_48938

/-- Represents the total number of cassettes -/
def total_cassettes : ℕ := 30

/-- Represents the number of disco cassettes -/
def disco_cassettes : ℕ := 12

/-- Represents the number of classical cassettes -/
def classical_cassettes : ℕ := 18

/-- Probability of selecting two disco cassettes when the first is returned -/
def prob_two_disco_with_return : ℚ := 4/25

/-- Probability of selecting two disco cassettes when the first is not returned -/
def prob_two_disco_without_return : ℚ := 22/145

/-- Theorem stating the probabilities of selecting two disco cassettes -/
theorem two_disco_cassettes_probability :
  (prob_two_disco_with_return = (disco_cassettes : ℚ) / total_cassettes * (disco_cassettes : ℚ) / total_cassettes) ∧
  (prob_two_disco_without_return = (disco_cassettes : ℚ) / total_cassettes * ((disco_cassettes - 1) : ℚ) / (total_cassettes - 1)) :=
by sorry

end two_disco_cassettes_probability_l489_48938


namespace symmetric_circle_correct_l489_48942

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + (y + 1)^2 = 1

-- Theorem stating that the symmetric circle is correct
theorem symmetric_circle_correct :
  ∀ (x y : ℝ), 
    (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧ 
      (∀ (x' y' : ℝ), symmetry_line ((x + x₀)/2) ((y + y₀)/2) → 
        (x - x')^2 + (y - y')^2 = (x₀ - x')^2 + (y₀ - y')^2)) →
    symmetric_circle x y :=
by sorry

end symmetric_circle_correct_l489_48942


namespace largest_angle_convex_hexagon_l489_48957

/-- The measure of the largest angle in a convex hexagon with specific angle measures -/
theorem largest_angle_convex_hexagon : 
  ∀ x : ℝ,
  (x + 2) + (2*x + 3) + (3*x - 1) + (4*x + 2) + (5*x - 4) + (6*x - 3) = 720 →
  max (x + 2) (max (2*x + 3) (max (3*x - 1) (max (4*x + 2) (max (5*x - 4) (6*x - 3))))) = 203 :=
by
  sorry

end largest_angle_convex_hexagon_l489_48957


namespace bushes_for_zucchinis_l489_48945

/-- Represents the number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 8

/-- Represents the number of containers traded for zucchinis -/
def containers_traded : ℕ := 5

/-- Represents the number of zucchinis received in trade -/
def zucchinis_received : ℕ := 2

/-- Represents the target number of zucchinis -/
def target_zucchinis : ℕ := 48

/-- Calculates the number of bushes needed to obtain the target number of zucchinis -/
def bushes_needed : ℕ :=
  (target_zucchinis * containers_traded) / (zucchinis_received * containers_per_bush)

theorem bushes_for_zucchinis :
  bushes_needed = 15 :=
sorry

end bushes_for_zucchinis_l489_48945


namespace license_plate_count_l489_48992

def even_digits : Nat := 5
def consonants : Nat := 20
def vowels : Nat := 6

def license_plate_combinations : Nat :=
  even_digits * consonants * vowels * consonants

theorem license_plate_count :
  license_plate_combinations = 12000 := by
  sorry

end license_plate_count_l489_48992


namespace min_value_ab_min_value_is_16_min_value_achieved_l489_48974

theorem min_value_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq : 1/a + 4/b = 1) : 
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ 1/x + 4/y = 1 → a * b ≤ x * y := by
  sorry

theorem min_value_is_16 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq : 1/a + 4/b = 1) : 
  a * b ≥ 16 := by
  sorry

theorem min_value_achieved (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_eq : 1/a + 4/b = 1) : 
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ 1/x + 4/y = 1 ∧ x * y = 16 := by
  sorry

end min_value_ab_min_value_is_16_min_value_achieved_l489_48974


namespace min_value_of_sum_of_ratios_l489_48928

theorem min_value_of_sum_of_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b) + (b / c) + (c / a) + (a / c) ≥ 4 ∧
  ((a / b) + (b / c) + (c / a) + (a / c) = 4 ↔ a = b ∧ b = c) :=
by sorry

end min_value_of_sum_of_ratios_l489_48928


namespace arc_length_from_sector_area_l489_48997

/-- Given a circle with radius 5 cm and a sector with area 13.75 cm²,
    prove that the length of the arc forming the sector is 5.5 cm. -/
theorem arc_length_from_sector_area (r : ℝ) (area : ℝ) (arc_length : ℝ) :
  r = 5 →
  area = 13.75 →
  arc_length = (2 * area) / r →
  arc_length = 5.5 :=
by
  sorry

#check arc_length_from_sector_area

end arc_length_from_sector_area_l489_48997


namespace distribute_6_3_l489_48924

/-- The number of ways to distribute n identical objects into k distinct containers,
    with each container having at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The problem statement -/
theorem distribute_6_3 : distribute 6 3 = 10 := by sorry

end distribute_6_3_l489_48924


namespace quadratic_equation_solution_l489_48904

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = 2024*x ↔ x = 0 ∨ x = 2024 := by
sorry

end quadratic_equation_solution_l489_48904


namespace power_comparison_l489_48914

theorem power_comparison : 4^15 = 8^10 ∧ 8^10 < 2^31 := by sorry

end power_comparison_l489_48914


namespace cosine_of_angle_between_vectors_l489_48969

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (3, 4)

theorem cosine_of_angle_between_vectors :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = 2 * Real.sqrt 5 / 25 := by
  sorry

end cosine_of_angle_between_vectors_l489_48969


namespace hyperbola_equation_l489_48947

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is y = √3 x and one of its foci lies on the line x = -6,
    then the equation of the hyperbola is x²/9 - y²/27 = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 6) →
  b/a = Real.sqrt 3 →
  ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2/9 - y^2/27 = 1 :=
by sorry

end hyperbola_equation_l489_48947


namespace perfect_square_two_pow_plus_three_l489_48985

theorem perfect_square_two_pow_plus_three (n : ℕ) : 
  (∃ k : ℕ, 2^n + 3 = k^2) ↔ n = 0 := by sorry

end perfect_square_two_pow_plus_three_l489_48985


namespace value_is_square_of_number_l489_48946

theorem value_is_square_of_number (n v : ℕ) : 
  n = 14 → 
  v = n^2 → 
  n + v = 210 → 
  v = 196 := by sorry

end value_is_square_of_number_l489_48946


namespace product_mod_twenty_l489_48902

theorem product_mod_twenty : (53 * 76 * 91) % 20 = 8 := by
  sorry

end product_mod_twenty_l489_48902


namespace total_population_l489_48926

/-- Represents the population of a school -/
structure SchoolPopulation where
  boys : ℕ
  girls : ℕ
  teachers : ℕ
  staff : ℕ

/-- The ratios in the school population -/
def school_ratios (p : SchoolPopulation) : Prop :=
  p.boys = 4 * p.girls ∧ 
  p.girls = 8 * p.teachers ∧ 
  p.staff = 2 * p.teachers

theorem total_population (p : SchoolPopulation) 
  (h : school_ratios p) : 
  p.boys + p.girls + p.teachers + p.staff = (43 * p.boys) / 32 := by
  sorry

end total_population_l489_48926


namespace route_length_l489_48900

/-- Given two trains traveling on a route, prove the length of the route. -/
theorem route_length : 
  ∀ (route_length : ℝ) (train_x_speed : ℝ) (train_y_speed : ℝ),
  train_x_speed > 0 →
  train_y_speed > 0 →
  route_length / train_x_speed = 5 →
  route_length / train_y_speed = 4 →
  80 / train_x_speed + (route_length - 80) / train_y_speed = route_length / train_y_speed →
  route_length = 180 := by
  sorry

end route_length_l489_48900


namespace novel_distribution_count_l489_48916

/-- The number of ways to distribute 4 novels among 5 students -/
def novel_distribution : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem stating that the number of novel distributions is 240 -/
theorem novel_distribution_count : novel_distribution = 240 := by
  sorry

end novel_distribution_count_l489_48916


namespace min_time_is_200_minutes_l489_48987

/-- Represents the travel problem between two cities -/
structure TravelProblem where
  distance : ℝ
  num_people : ℕ
  num_bicycles : ℕ
  cyclist_speed : ℝ
  pedestrian_speed : ℝ

/-- Calculates the minimum travel time for the given problem -/
def min_travel_time (problem : TravelProblem) : ℝ :=
  sorry

/-- Theorem stating that the minimum travel time for the given problem is 200 minutes -/
theorem min_time_is_200_minutes :
  let problem : TravelProblem := {
    distance := 45,
    num_people := 3,
    num_bicycles := 2,
    cyclist_speed := 15,
    pedestrian_speed := 5
  }
  min_travel_time problem = 200 / 60 := by sorry

end min_time_is_200_minutes_l489_48987


namespace lieutenant_age_l489_48956

theorem lieutenant_age : ∃ (n : ℕ) (x : ℕ),
  n * (n + 5) = x * (n + 9) ∧
  x = 24 := by
  sorry

end lieutenant_age_l489_48956


namespace angle_A_is_pi_over_three_perimeter_range_l489_48911

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Triangle inequality
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  ineq_a : a < b + c
  ineq_b : b < a + c
  ineq_c : c < a + b
  -- Angle sum is π
  angle_sum : A + B + C = π
  -- Sine rule
  sine_rule_a : a / Real.sin A = b / Real.sin B
  sine_rule_b : b / Real.sin B = c / Real.sin C
  -- Cosine rule
  cosine_rule_a : a^2 = b^2 + c^2 - 2*b*c*Real.cos A
  cosine_rule_b : b^2 = a^2 + c^2 - 2*a*c*Real.cos B
  cosine_rule_c : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem angle_A_is_pi_over_three (t : Triangle) (h : t.a * Real.cos t.C + (1/2) * t.c = t.b) :
  t.A = π/3 := by sorry

theorem perimeter_range (t : Triangle) (h1 : t.a = 1) (h2 : t.a * Real.cos t.C + (1/2) * t.c = t.b) :
  2 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 3 := by sorry

end angle_A_is_pi_over_three_perimeter_range_l489_48911


namespace total_money_after_redistribution_l489_48951

/-- Represents the money redistribution problem among three friends --/
def money_redistribution (a j t : ℕ) : Prop :=
  let a1 := a - 2*(t + j)
  let j1 := 3*j
  let t1 := 3*t
  let a2 := 2*a1
  let j2 := j1 - (a1 + t1)
  let t2 := 2*t1
  let a3 := 2*a2
  let j3 := 2*j2
  let t3 := t2 - (a2 + j2)
  (t = 48) ∧ (t3 = 48) ∧ (a3 + j3 + t3 = 528)

/-- Theorem stating the total amount of money after redistribution --/
theorem total_money_after_redistribution :
  ∃ (a j : ℕ), money_redistribution a j 48 :=
sorry

end total_money_after_redistribution_l489_48951


namespace local_extremum_cubic_l489_48982

/-- Given a cubic function f(x) = ax³ + 3x² - 6ax + b with a local extremum of 9 at x = 2,
    prove that a + 2b = -24 -/
theorem local_extremum_cubic (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + 3 * x^2 - 6 * a * x + b
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≤ f 2) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f x ≥ f 2) ∧
  f 2 = 9 →
  a + 2 * b = -24 := by
sorry

end local_extremum_cubic_l489_48982


namespace fraction_division_calculate_fraction_l489_48993

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  a / (c / d) = (a * d) / c := by
  sorry

theorem calculate_fraction :
  7 / (9 / 14) = 98 / 9 := by
  sorry

end fraction_division_calculate_fraction_l489_48993


namespace unique_functional_equation_l489_48912

theorem unique_functional_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (2 * f x + f y) = 2 * x + f y :=
by
  -- The proof goes here
  sorry

end unique_functional_equation_l489_48912


namespace welcoming_and_planning_committees_l489_48932

theorem welcoming_and_planning_committees (n : ℕ) : 
  (Nat.choose n 2 = 6) → (Nat.choose n 4 = 1) :=
by sorry

end welcoming_and_planning_committees_l489_48932
