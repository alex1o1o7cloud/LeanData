import Mathlib

namespace complex_magnitude_equation_l3488_348828

theorem complex_magnitude_equation (s : ℝ) (hs : s > 0) :
  Complex.abs (3 + s * Complex.I) = 13 → s = 4 * Real.sqrt 10 := by
  sorry

end complex_magnitude_equation_l3488_348828


namespace rectangle_ratio_golden_ratio_l3488_348815

/-- Given a unit square AEFD, prove that if the ratio of length to width of rectangle ABCD
    equals the ratio of length to width of rectangle BCFE, then the length of AB (W) is (1 + √5) / 2. -/
theorem rectangle_ratio_golden_ratio (W : ℝ) : 
  W > 0 ∧ W / 1 = 1 / (W - 1) → W = (1 + Real.sqrt 5) / 2 := by
  sorry

#check rectangle_ratio_golden_ratio

end rectangle_ratio_golden_ratio_l3488_348815


namespace negative_three_halves_less_than_negative_one_l3488_348829

theorem negative_three_halves_less_than_negative_one :
  -((3 : ℚ) / 2) < -1 := by sorry

end negative_three_halves_less_than_negative_one_l3488_348829


namespace sum_of_qp_at_points_l3488_348856

def p (x : ℝ) : ℝ := |x^2 - 4|

def q (x : ℝ) : ℝ := -|x|

def evaluation_points : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_qp_at_points :
  (evaluation_points.map (λ x => q (p x))).sum = -20 := by sorry

end sum_of_qp_at_points_l3488_348856


namespace quadratic_equation_from_condition_l3488_348895

theorem quadratic_equation_from_condition (a b : ℝ) :
  a^2 - 4*a*b + 5*b^2 - 2*b + 1 = 0 →
  ∃ (x : ℝ → ℝ), (x a = 0 ∧ x b = 0) ∧ (∀ y, x y = y^2 - 3*y + 2) :=
by sorry

end quadratic_equation_from_condition_l3488_348895


namespace area_of_region_l3488_348825

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 17 ∧ 
   A = Real.pi * (Real.sqrt ((x + 1)^2 + (y - 2)^2))^2 ∧
   x^2 + y^2 + 2*x - 4*y = 12) :=
by sorry

end area_of_region_l3488_348825


namespace hyperbola_focal_length_l3488_348892

theorem hyperbola_focal_length (m : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/m = 1) → 
  (∃ c : ℝ, c = 5) →
  m = 16 :=
sorry

end hyperbola_focal_length_l3488_348892


namespace concert_attendance_difference_l3488_348833

theorem concert_attendance_difference (first_concert : Nat) (second_concert : Nat)
  (h1 : first_concert = 65899)
  (h2 : second_concert = 66018) :
  second_concert - first_concert = 119 := by
  sorry

end concert_attendance_difference_l3488_348833


namespace isosceles_trapezoid_perimeter_l3488_348867

theorem isosceles_trapezoid_perimeter 
  (base1 : ℝ) (base2 : ℝ) (altitude : ℝ)
  (h1 : base1 = Real.log 3)
  (h2 : base2 = Real.log 192)
  (h3 : altitude = Real.log 16)
  (h4 : ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ 
    perimeter = Real.log (2^p * 3^q)) :
  ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ 
    perimeter = Real.log (2^p * 3^q) ∧ p + q = 18 :=
  sorry

end isosceles_trapezoid_perimeter_l3488_348867


namespace condition_p_necessary_not_sufficient_for_q_l3488_348839

theorem condition_p_necessary_not_sufficient_for_q :
  (∀ x y : ℝ, Real.sqrt x > Real.sqrt y → x > y) ∧
  (∃ x y : ℝ, x > y ∧ ¬(Real.sqrt x > Real.sqrt y)) := by
  sorry

end condition_p_necessary_not_sufficient_for_q_l3488_348839


namespace leftover_coin_value_l3488_348849

def quarters_per_roll : ℕ := 45
def dimes_per_roll : ℕ := 55
def james_quarters : ℕ := 95
def james_dimes : ℕ := 173
def lindsay_quarters : ℕ := 140
def lindsay_dimes : ℕ := 285
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftover_coin_value :
  let total_quarters := james_quarters + lindsay_quarters
  let total_dimes := james_dimes + lindsay_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  let leftover_value := (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value
  leftover_value = 5.30 := by sorry

end leftover_coin_value_l3488_348849


namespace basketball_team_selection_l3488_348858

/-- The number of ways to choose k elements from n elements -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players in the team -/
def total_players : ℕ := 18

/-- The number of quadruplets (who must be included in the starting lineup) -/
def quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def starters : ℕ := 7

theorem basketball_team_selection :
  binomial (total_players - quadruplets) (starters - quadruplets) = 364 := by
  sorry

end basketball_team_selection_l3488_348858


namespace symmetric_point_l3488_348882

/-- Given a line l: x + y = 1 and two points P and Q, 
    this function checks if Q is symmetric to P with respect to l --/
def is_symmetric (P Q : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  (qy - py) / (qx - px) = -1 ∧ -- Perpendicular condition
  (px + qx) / 2 + (py + qy) / 2 = 1 -- Midpoint on the line condition

/-- Theorem stating that Q(-4, -1) is symmetric to P(2, 5) with respect to the line x + y = 1 --/
theorem symmetric_point : is_symmetric (2, 5) (-4, -1) := by
  sorry

end symmetric_point_l3488_348882


namespace ellipse_and_line_equations_l3488_348898

-- Define the ellipse G
def ellipse_G (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = Real.sqrt 6 / 3

-- Define the right focus
def right_focus (x y : ℝ) : Prop := x = 2 * Real.sqrt 2 ∧ y = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := ∃ (m : ℝ), y = x + m

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_G A.1 A.2 ∧ ellipse_G B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define the isosceles triangle
def isosceles_triangle (A B : ℝ × ℝ) : Prop :=
  ∃ (P : ℝ × ℝ), P = (-3, 2) ∧
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2

-- Theorem statement
theorem ellipse_and_line_equations :
  ∀ (A B : ℝ × ℝ) (e : ℝ),
  ellipse_G A.1 A.2 ∧ ellipse_G B.1 B.2 ∧
  eccentricity e ∧
  right_focus (2 * Real.sqrt 2) 0 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
  intersection_points A B ∧
  isosceles_triangle A B →
  (∀ (x y : ℝ), ellipse_G x y ↔ x^2 / 12 + y^2 / 4 = 1) ∧
  (∀ (x y : ℝ), line_l x y ↔ x - y + 2 = 0) :=
by sorry

end ellipse_and_line_equations_l3488_348898


namespace floor_abs_sum_equality_l3488_348873

theorem floor_abs_sum_equality : ⌊|(-7.3 : ℝ)|⌋ + |⌊(-7.3 : ℝ)⌋| = 15 := by
  sorry

end floor_abs_sum_equality_l3488_348873


namespace quadratic_form_ratio_l3488_348806

theorem quadratic_form_ratio (a d : ℝ) : 
  (∀ x, x^2 + 500*x + 2500 = (x + a)^2 + d) →
  d / a = -240 := by
sorry

end quadratic_form_ratio_l3488_348806


namespace march_greatest_drop_l3488_348871

/-- Represents the months of the year --/
inductive Month
| january | february | march | april | may | june | july

/-- The price change for each month --/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.january  => -0.75
  | Month.february => 1.50
  | Month.march    => -3.00
  | Month.april    => 2.50
  | Month.may      => -1.00
  | Month.june     => 0.50
  | Month.july     => -2.50

/-- The set of months considered in the problem --/
def considered_months : List Month :=
  [Month.january, Month.february, Month.march, Month.april, Month.may, Month.june, Month.july]

/-- Predicate to check if a month has a price drop --/
def has_price_drop (m : Month) : Prop :=
  price_change m < 0

/-- The theorem stating that March had the greatest monthly drop in price --/
theorem march_greatest_drop :
  ∀ m ∈ considered_months, has_price_drop m →
    price_change Month.march ≤ price_change m :=
  sorry

end march_greatest_drop_l3488_348871


namespace evaluate_expression_l3488_348870

theorem evaluate_expression : -(18 / 3 * 8 - 70 + 5 * 7) = -13 := by
  sorry

end evaluate_expression_l3488_348870


namespace first_number_in_sequence_l3488_348877

def sequence_property (s : Fin 10 → ℕ) : Prop :=
  ∀ n : Fin 10, n.val ≥ 2 → s n = s (n - 1) * s (n - 2)

theorem first_number_in_sequence 
  (s : Fin 10 → ℕ) 
  (h_property : sequence_property s)
  (h_last_three : s 7 = 81 ∧ s 8 = 6561 ∧ s 9 = 43046721) :
  s 0 = 3486784401 :=
sorry

end first_number_in_sequence_l3488_348877


namespace shop_length_calculation_l3488_348881

/-- Given a shop with specified dimensions and rent, calculate its length -/
theorem shop_length_calculation (width : ℝ) (monthly_rent : ℝ) (annual_rent_per_sqft : ℝ) :
  width = 20 →
  monthly_rent = 3600 →
  annual_rent_per_sqft = 120 →
  (monthly_rent * 12) / (width * annual_rent_per_sqft) = 18 :=
by sorry

end shop_length_calculation_l3488_348881


namespace rectangle_area_change_l3488_348857

/-- Theorem: When the length of a rectangle is halved and its breadth is tripled, 
    the percentage change in area is a 50% increase. -/
theorem rectangle_area_change (L B : ℝ) (h1 : L > 0) (h2 : B > 0) : 
  let original_area := L * B
  let new_area := (L / 2) * (3 * B)
  let percent_change := (new_area - original_area) / original_area * 100
  percent_change = 50 := by
sorry


end rectangle_area_change_l3488_348857


namespace range_of_m_min_distance_to_origin_range_of_slope_l3488_348813

-- Define the circle C
def C (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 8*y + m = 0

-- Define point P
def P : ℝ × ℝ := (0, 4)

-- Theorem 1
theorem range_of_m (m : ℝ) :
  (∀ x y, C m x y → (P.1 - x)^2 + (P.2 - y)^2 > 0) → 16 < m ∧ m < 25 :=
sorry

-- Theorem 2
theorem min_distance_to_origin (x y : ℝ) :
  C 24 x y → x^2 + y^2 ≥ 16 :=
sorry

-- Theorem 3
theorem range_of_slope (x y : ℝ) :
  C 24 x y → x ≠ 0 → -Real.sqrt 2 / 4 ≤ (y - 4) / x ∧ (y - 4) / x ≤ Real.sqrt 2 / 4 :=
sorry

end range_of_m_min_distance_to_origin_range_of_slope_l3488_348813


namespace right_triangle_increase_sides_acute_l3488_348885

theorem right_triangle_increase_sides_acute (a b c x : ℝ) : 
  a > 0 → b > 0 → c > 0 → x > 0 → c^2 = a^2 + b^2 → 
  (a + x)^2 + (b + x)^2 > (c + x)^2 := by
sorry

end right_triangle_increase_sides_acute_l3488_348885


namespace triangle_inequality_l3488_348841

/-- Given a triangle with side lengths a, b, and c, the expression
    a^2 b(a-b) + b^2 c(b-c) + c^2 a(c-a) is non-negative,
    with equality if and only if the triangle is equilateral. -/
theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end triangle_inequality_l3488_348841


namespace unique_number_251_l3488_348827

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 251 is the unique positive integer whose product with its sum of digits is 2008 -/
theorem unique_number_251 : ∃! (n : ℕ), n > 0 ∧ n * sum_of_digits n = 2008 :=
  sorry

end unique_number_251_l3488_348827


namespace sqrt_plus_square_zero_implies_diff_five_l3488_348844

theorem sqrt_plus_square_zero_implies_diff_five (x y : ℝ) 
  (h : Real.sqrt (x - 3) + (y + 2)^2 = 0) : x - y = 5 := by
  sorry

end sqrt_plus_square_zero_implies_diff_five_l3488_348844


namespace factorization_problems_l3488_348861

theorem factorization_problems :
  (∀ x : ℝ, 4*x^2 - 16 = 4*(x+2)*(x-2)) ∧
  (∀ x y : ℝ, 2*x^3 - 12*x^2*y + 18*x*y^2 = 2*x*(x-3*y)^2) := by
sorry

end factorization_problems_l3488_348861


namespace discount_order_difference_l3488_348894

/-- Calculates the final price after applying discounts and tax -/
def final_price (initial_price : ℚ) (flat_discount : ℚ) (percent_discount : ℚ) (tax_rate : ℚ) (flat_first : Bool) : ℚ :=
  let price_after_flat := initial_price - flat_discount
  let price_after_percent := initial_price * (1 - percent_discount)
  let discounted_price := if flat_first then
    price_after_flat * (1 - percent_discount)
  else
    price_after_percent - flat_discount
  discounted_price * (1 + tax_rate)

/-- The difference in final price between two discount application orders -/
def price_difference (initial_price flat_discount percent_discount tax_rate : ℚ) : ℚ :=
  (final_price initial_price flat_discount percent_discount tax_rate true) -
  (final_price initial_price flat_discount percent_discount tax_rate false)

theorem discount_order_difference :
  price_difference 30 5 (25/100) (10/100) = 1375/1000 := by
  sorry

end discount_order_difference_l3488_348894


namespace second_project_length_l3488_348811

/-- Represents a digging project with depth, length, and breadth measurements. -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of a digging project. -/
def volume (project : DiggingProject) : ℝ :=
  project.depth * project.length * project.breadth

theorem second_project_length : 
  ∀ (project1 project2 : DiggingProject),
  project1.depth = 100 →
  project1.length = 25 →
  project1.breadth = 30 →
  project2.depth = 75 →
  project2.breadth = 50 →
  volume project1 = volume project2 →
  project2.length = 20 := by
  sorry

#check second_project_length

end second_project_length_l3488_348811


namespace even_function_implies_m_equals_one_l3488_348891

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x² + (m-1)x -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (m - 1) * x

theorem even_function_implies_m_equals_one (m : ℝ) :
  IsEven (f m) → m = 1 := by
  sorry

end even_function_implies_m_equals_one_l3488_348891


namespace base_number_proof_l3488_348835

theorem base_number_proof (x : ℝ) (k : ℕ+) 
  (h1 : x^(k : ℝ) = 4) 
  (h2 : x^(2*(k : ℝ) + 2) = 64) : 
  x = 2 := by
sorry

end base_number_proof_l3488_348835


namespace square_sum_eq_243_l3488_348868

theorem square_sum_eq_243 (x y : ℝ) (h1 : x + 3 * y = 9) (h2 : x * y = -27) :
  x^2 + 9 * y^2 = 243 := by
  sorry

end square_sum_eq_243_l3488_348868


namespace fruit_cost_prices_l3488_348880

/-- Represents the cost and selling prices of fruits -/
structure FruitPrices where
  appleCost : ℚ
  appleSell : ℚ
  orangeCost : ℚ
  orangeSell : ℚ
  bananaCost : ℚ
  bananaSell : ℚ

/-- Calculates the cost prices of fruits based on selling prices and profit/loss percentages -/
def calculateCostPrices (p : FruitPrices) : Prop :=
  p.appleSell = p.appleCost - (1/6 * p.appleCost) ∧
  p.orangeSell = p.orangeCost + (1/5 * p.orangeCost) ∧
  p.bananaSell = p.bananaCost

/-- Theorem stating the correct cost prices of fruits -/
theorem fruit_cost_prices :
  ∃ (p : FruitPrices),
    p.appleSell = 15 ∧
    p.orangeSell = 20 ∧
    p.bananaSell = 10 ∧
    calculateCostPrices p ∧
    p.appleCost = 18 ∧
    p.orangeCost = 100/6 ∧
    p.bananaCost = 10 :=
  sorry

end fruit_cost_prices_l3488_348880


namespace cylinder_height_relationship_l3488_348819

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end cylinder_height_relationship_l3488_348819


namespace max_sum_with_reciprocals_l3488_348821

theorem max_sum_with_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y = 5) : 
  x + y ≤ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b + 1/a + 1/b = 5 ∧ a + b = 4 :=
sorry

end max_sum_with_reciprocals_l3488_348821


namespace min_value_theorem_l3488_348878

/-- Two circles C₁ and C₂ with given equations -/
def C₁ (x y a : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 9 = 0
def C₂ (x y b : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The condition that the circles have only one common tangent line -/
def one_common_tangent (a b : ℝ) : Prop := sorry

theorem min_value_theorem (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : one_common_tangent a b) :
  (∀ x y, C₁ x y a → C₂ x y b → 4/a^2 + 1/b^2 ≥ 4) ∧ 
  (∃ x y, C₁ x y a ∧ C₂ x y b ∧ 4/a^2 + 1/b^2 = 4) :=
by sorry

end min_value_theorem_l3488_348878


namespace complex_magnitude_three_fifths_minus_four_sevenths_i_l3488_348896

theorem complex_magnitude_three_fifths_minus_four_sevenths_i :
  Complex.abs (3/5 - (4/7)*Complex.I) = 29/35 := by
  sorry

end complex_magnitude_three_fifths_minus_four_sevenths_i_l3488_348896


namespace absent_laborers_l3488_348865

theorem absent_laborers (W : ℝ) : 
  let L := 17.5
  let original_days := 6
  let actual_days := 10
  let absent := L * (1 - (original_days : ℝ) / (actual_days : ℝ))
  absent = 14 := by sorry

end absent_laborers_l3488_348865


namespace circle_condition_l3488_348812

/-- A circle in the xy-plane can be represented by the equation (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius. -/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ h k r, r > 0 ∧ ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The equation of the form x^2 + y^2 + dx + ey + f = 0 -/
def general_quadratic (d e f : ℝ) (x y : ℝ) : ℝ :=
  x^2 + y^2 + d*x + e*y + f

theorem circle_condition (m : ℝ) :
  is_circle (general_quadratic (-2) (-4) m) → m < 5 := by
  sorry


end circle_condition_l3488_348812


namespace train_platform_passing_time_l3488_348802

-- Define the given constants
def train_length : ℝ := 250
def pole_passing_time : ℝ := 10
def platform_length : ℝ := 1250
def speed_reduction_factor : ℝ := 0.75

-- Define the theorem
theorem train_platform_passing_time :
  let original_speed := train_length / pole_passing_time
  let incline_speed := original_speed * speed_reduction_factor
  let total_distance := train_length + platform_length
  total_distance / incline_speed = 80 := by
  sorry

end train_platform_passing_time_l3488_348802


namespace grid_diagonal_property_l3488_348822

/-- Represents a cell color in the grid -/
inductive Color
| Black
| White

/-- Represents a 100 x 100 grid -/
def Grid := Fin 100 → Fin 100 → Color

/-- A predicate that checks if a cell is on the boundary of the grid -/
def isBoundary (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- A predicate that checks if a 2x2 subgrid is monochromatic -/
def isMonochromatic (g : Grid) (i j : Fin 100) : Prop :=
  g i j = g (i+1) j ∧ g i j = g i (j+1) ∧ g i j = g (i+1) (j+1)

/-- A predicate that checks if a 2x2 subgrid has the desired diagonal property -/
def hasDiagonalProperty (g : Grid) (i j : Fin 100) : Prop :=
  (g i j = g (i+1) (j+1) ∧ g i (j+1) = g (i+1) j ∧ g i j ≠ g i (j+1))
  ∨ (g i j = g (i+1) (j+1) ∧ g i (j+1) = g (i+1) j ∧ g i (j+1) ≠ g i j)

theorem grid_diagonal_property (g : Grid) 
  (boundary_black : ∀ i j, isBoundary i j → g i j = Color.Black)
  (no_monochromatic : ∀ i j, ¬isMonochromatic g i j) :
  ∃ i j, hasDiagonalProperty g i j := by
  sorry

end grid_diagonal_property_l3488_348822


namespace solve_equation_l3488_348853

theorem solve_equation (x y : ℚ) : 
  y = 2 / (4 * x + 2) → y = 1/2 → x = 1/2 := by
  sorry

end solve_equation_l3488_348853


namespace rationalize_denominator_l3488_348862

theorem rationalize_denominator :
  let x : ℝ := Real.rpow 3 (1/3)
  (1 / (x + Real.rpow 27 (1/3) - Real.rpow 9 (1/3))) = (x^2 + 3*x + 3) / (3 * 21) := by
  sorry

end rationalize_denominator_l3488_348862


namespace fifth_figure_perimeter_l3488_348837

/-- Represents the outer perimeter of a figure in the sequence -/
def outer_perimeter (n : ℕ) : ℕ :=
  4 + 4 * (n - 1)

/-- The outer perimeter of the fifth figure in the sequence is 20 -/
theorem fifth_figure_perimeter :
  outer_perimeter 5 = 20 := by
  sorry

#check fifth_figure_perimeter

end fifth_figure_perimeter_l3488_348837


namespace min_value_on_line_l3488_348884

theorem min_value_on_line (m n : ℝ) : 
  m + 2 * n = 1 → 2^m + 4^n ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_on_line_l3488_348884


namespace root_of_equations_l3488_348826

theorem root_of_equations (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + p = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + p * m + q = 0) :
  m^5 = q / p ∧ (p = q → ∃ k : Fin 5, m = Complex.exp (2 * Real.pi * I * (k : ℝ) / 5)) :=
sorry

end root_of_equations_l3488_348826


namespace transaction_period_is_one_year_l3488_348843

/-- Represents the financial transaction described in the problem -/
structure Transaction where
  principal : ℝ
  borrow_rate : ℝ
  lend_rate : ℝ
  gain_per_year : ℝ

/-- Calculates the number of years for the transaction -/
def transaction_years (t : Transaction) : ℝ :=
  1

/-- Theorem stating that the transaction period is 1 year -/
theorem transaction_period_is_one_year (t : Transaction) 
  (h1 : t.principal = 5000)
  (h2 : t.borrow_rate = 0.04)
  (h3 : t.lend_rate = 0.08)
  (h4 : t.gain_per_year = 200) :
  transaction_years t = 1 := by
  sorry

end transaction_period_is_one_year_l3488_348843


namespace cubic_sum_l3488_348860

theorem cubic_sum (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c →
  a^3 + b^3 + c^3 = -36 := by
sorry

end cubic_sum_l3488_348860


namespace congruence_implies_prime_and_n_equals_m_minus_one_l3488_348824

theorem congruence_implies_prime_and_n_equals_m_minus_one 
  (n m : ℕ) 
  (h_n : n ≥ 2) 
  (h_m : m ≥ 2) 
  (h_cong : ∀ k : ℕ, 1 ≤ k → k ≤ n → k^n % m = 1) : 
  Nat.Prime m ∧ n = m - 1 := by
sorry

end congruence_implies_prime_and_n_equals_m_minus_one_l3488_348824


namespace negation_of_existence_negation_of_quadratic_inequality_l3488_348817

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_inequality_l3488_348817


namespace tangent_line_value_l3488_348847

/-- The line x + y = c is tangent to the circle x^2 + y^2 = 8, where c is a positive real number. -/
def is_tangent_line (c : ℝ) : Prop :=
  c > 0 ∧ ∃ (x y : ℝ), x^2 + y^2 = 8 ∧ x + y = c ∧
  ∀ (x' y' : ℝ), x' + y' = c → x'^2 + y'^2 ≥ 8

theorem tangent_line_value :
  ∀ c : ℝ, is_tangent_line c → c = 4 :=
by sorry

end tangent_line_value_l3488_348847


namespace banana_arrangement_count_l3488_348846

/-- The number of unique arrangements of the letters in BANANA -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in BANANA -/
def total_letters : ℕ := 6

/-- The number of A's in BANANA -/
def num_a : ℕ := 3

/-- The number of N's in BANANA -/
def num_n : ℕ := 2

/-- The number of B's in BANANA -/
def num_b : ℕ := 1

theorem banana_arrangement_count :
  banana_arrangements = Nat.factorial total_letters / (Nat.factorial num_a * Nat.factorial num_n) :=
sorry

end banana_arrangement_count_l3488_348846


namespace units_digit_of_large_power_l3488_348890

theorem units_digit_of_large_power (n : ℕ) : n % 10 = (7^(3^(5^2))) % 10 → n = 3 := by
  sorry

end units_digit_of_large_power_l3488_348890


namespace negation_of_proposition_l3488_348831

theorem negation_of_proposition :
  (¬ ∀ (x y : ℝ), xy = 0 → x = 0) ↔ (∃ (x y : ℝ), xy = 0 ∧ x ≠ 0) :=
by sorry

end negation_of_proposition_l3488_348831


namespace tan_sum_ratio_equals_neg_sqrt_three_over_three_l3488_348869

theorem tan_sum_ratio_equals_neg_sqrt_three_over_three : 
  (Real.tan (10 * π / 180) + Real.tan (20 * π / 180) + Real.tan (150 * π / 180)) / 
  (Real.tan (10 * π / 180) * Real.tan (20 * π / 180)) = -Real.sqrt 3 / 3 := by
  sorry

end tan_sum_ratio_equals_neg_sqrt_three_over_three_l3488_348869


namespace dividend_divisor_quotient_remainder_problem_l3488_348875

theorem dividend_divisor_quotient_remainder_problem 
  (y1 y2 z1 z2 r1 x1 x2 : ℤ)
  (hy1 : y1 = 2)
  (hy2 : y2 = 3)
  (hz1 : z1 = 3)
  (hz2 : z2 = 5)
  (hr1 : r1 = 1)
  (hx1 : x1 = 4)
  (hx2 : x2 = 6)
  (y : ℤ) (hy : y = 3*(y1 + y2) + 4)
  (z : ℤ) (hz : z = 2*z1^2 - z2)
  (r : ℤ) (hr : r = 3*r1 + 2)
  (x : ℤ) (hx : x = 2*x1*y1 - x2 + 10) :
  x = 20 ∧ y = 19 ∧ z = 13 ∧ r = 5 := by
sorry

end dividend_divisor_quotient_remainder_problem_l3488_348875


namespace company_supervisors_l3488_348801

/-- Represents the number of workers per team lead -/
def workers_per_team_lead : ℕ := 10

/-- Represents the number of team leads per supervisor -/
def team_leads_per_supervisor : ℕ := 3

/-- Represents the total number of workers in the company -/
def total_workers : ℕ := 390

/-- Calculates the number of supervisors in the company -/
def calculate_supervisors : ℕ :=
  (total_workers / workers_per_team_lead) / team_leads_per_supervisor

theorem company_supervisors :
  calculate_supervisors = 13 := by sorry

end company_supervisors_l3488_348801


namespace arithmetic_sequence_common_difference_l3488_348816

theorem arithmetic_sequence_common_difference
  (a₁ : ℚ)    -- first term
  (aₙ : ℚ)    -- last term
  (S  : ℚ)    -- sum of all terms
  (h₁ : a₁ = 3)
  (h₂ : aₙ = 34)
  (h₃ : S = 222) :
  ∃ (n : ℕ) (d : ℚ), n > 1 ∧ d = 31/11 ∧ 
    aₙ = a₁ + (n - 1) * d ∧
    S = n * (a₁ + aₙ) / 2 :=
by sorry

end arithmetic_sequence_common_difference_l3488_348816


namespace pie_eaten_after_seven_trips_l3488_348800

def eat_pie (n : ℕ) : ℚ :=
  1 - (2/3)^n

theorem pie_eaten_after_seven_trips :
  eat_pie 7 = 1093 / 2187 :=
by sorry

end pie_eaten_after_seven_trips_l3488_348800


namespace fixed_points_bisector_range_l3488_348852

noncomputable def f (a b x : ℝ) : ℝ := a * x + b + 1

theorem fixed_points_bisector_range (a b : ℝ) :
  (0 < a) → (a < 2) →
  (∃ x₀ : ℝ, f a b x₀ = x₀) →
  (∃ A B : ℝ × ℝ, 
    (f a b A.1 = A.2 ∧ f a b B.1 = B.2) ∧
    (∀ x y : ℝ, y = x + 1 / (2 * a^2 + 1) ↔ 
      ((x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
       2 * x = A.1 + B.1 ∧ 2 * y = A.2 + B.2))) →
  b ∈ Set.Icc (-Real.sqrt 2 / 4) 0 ∧ b ≠ 0 :=
by sorry

end fixed_points_bisector_range_l3488_348852


namespace soap_cost_two_years_l3488_348876

-- Define the cost of one bar of soap
def cost_per_bar : ℕ := 4

-- Define the number of months in a year
def months_per_year : ℕ := 12

-- Define the number of years
def years : ℕ := 2

-- Define the function to calculate total cost
def total_cost (cost_per_bar months_per_year years : ℕ) : ℕ :=
  cost_per_bar * months_per_year * years

-- Theorem statement
theorem soap_cost_two_years :
  total_cost cost_per_bar months_per_year years = 96 := by
  sorry

end soap_cost_two_years_l3488_348876


namespace parabola_line_intersection_l3488_348854

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define the line passing through A(0,-2) and B(t,0)
def line (t x y : ℝ) : Prop := y = (2/t)*x - 2

-- Define the condition for no intersection
def no_intersection (t : ℝ) : Prop :=
  ∀ x y : ℝ, parabola x y → ¬(line t x y)

-- Theorem statement
theorem parabola_line_intersection (t : ℝ) :
  no_intersection t ↔ t < -1 ∨ t > 1 :=
sorry

end parabola_line_intersection_l3488_348854


namespace football_tickets_problem_l3488_348883

/-- Given a ticket price and budget, calculates the maximum number of tickets that can be purchased. -/
def max_tickets (price : ℕ) (budget : ℕ) : ℕ :=
  (budget / price : ℕ)

/-- Proves that given a ticket price of 15 and a budget of 120, the maximum number of tickets that can be purchased is 8. -/
theorem football_tickets_problem :
  max_tickets 15 120 = 8 := by
  sorry

end football_tickets_problem_l3488_348883


namespace sum_equals_point_nine_six_repeating_l3488_348836

/-- Represents a repeating decimal where the digit 8 repeats infinitely -/
def repeating_eight : ℚ := 8/9

/-- Represents the decimal 0.07 -/
def seven_hundredths : ℚ := 7/100

/-- Theorem stating that the sum of 0.8̇ and 0.07 is equal to 0.96̇ -/
theorem sum_equals_point_nine_six_repeating :
  repeating_eight + seven_hundredths = 29/30 := by sorry

end sum_equals_point_nine_six_repeating_l3488_348836


namespace anya_hair_growth_l3488_348851

/-- The number of hairs Anya washes down the drain -/
def hairs_washed : ℕ := 32

/-- The number of hairs Anya brushes out -/
def hairs_brushed : ℕ := hairs_washed / 2

/-- The number of hairs Anya needs to grow back -/
def hairs_to_grow : ℕ := 49

/-- The total number of additional hairs Anya wants to have -/
def additional_hairs : ℕ := hairs_washed + hairs_brushed + hairs_to_grow

theorem anya_hair_growth :
  additional_hairs = 97 := by sorry

end anya_hair_growth_l3488_348851


namespace player_B_most_consistent_l3488_348814

/-- Represents a player in the rope skipping test -/
inductive Player : Type
  | A : Player
  | B : Player
  | C : Player
  | D : Player

/-- Returns the variance of a player's performance -/
def variance (p : Player) : ℝ :=
  match p with
  | Player.A => 0.023
  | Player.B => 0.018
  | Player.C => 0.020
  | Player.D => 0.021

/-- States that Player B has the most consistent performance -/
theorem player_B_most_consistent :
  ∀ p : Player, p ≠ Player.B → variance Player.B < variance p :=
by sorry

end player_B_most_consistent_l3488_348814


namespace river_travel_time_l3488_348818

structure RiverSystem where
  docks : Fin 3 → String
  distance : Fin 3 → Fin 3 → ℝ
  time_against_current : ℝ
  time_with_current : ℝ

def valid_river_system (rs : RiverSystem) : Prop :=
  (∀ i j, rs.distance i j = 3) ∧
  rs.time_against_current = 30 ∧
  rs.time_with_current = 18 ∧
  rs.time_against_current > rs.time_with_current

def travel_time (rs : RiverSystem) : Set ℝ :=
  {24, 72}

theorem river_travel_time (rs : RiverSystem) (h : valid_river_system rs) :
  ∀ i j, i ≠ j → (rs.distance i j / rs.time_against_current * 60 ∈ travel_time rs) ∨
                 (rs.distance i j / rs.time_with_current * 60 ∈ travel_time rs) :=
sorry

end river_travel_time_l3488_348818


namespace beth_sells_80_coins_l3488_348838

/-- Calculates the number of coins Beth sells given her initial coins and a gift -/
def coins_sold (initial : ℕ) (gift : ℕ) : ℕ :=
  (initial + gift) / 2

/-- Proves that Beth sells 80 coins given her initial 125 coins and Carl's gift of 35 coins -/
theorem beth_sells_80_coins : coins_sold 125 35 = 80 := by
  sorry

end beth_sells_80_coins_l3488_348838


namespace pie_piece_price_l3488_348888

/-- Represents the price of a single piece of pie -/
def price_per_piece : ℝ := 3.83

/-- Represents the number of pieces a single pie is divided into -/
def pieces_per_pie : ℕ := 3

/-- Represents the number of pies the bakery can make in one hour -/
def pies_per_hour : ℕ := 12

/-- Represents the cost to create one pie -/
def cost_per_pie : ℝ := 0.5

/-- Represents the total revenue from selling all pie pieces -/
def total_revenue : ℝ := 138

theorem pie_piece_price :
  price_per_piece * (pieces_per_pie * pies_per_hour) = total_revenue :=
by sorry

end pie_piece_price_l3488_348888


namespace barrels_of_pitch_day4_is_two_l3488_348820

/-- Represents the roadwork company's paving project --/
structure RoadworkProject where
  total_length : ℕ
  gravel_per_truck : ℕ
  gravel_to_pitch_ratio : ℕ
  day1_truckloads_per_mile : ℕ
  day1_miles_paved : ℕ
  day2_truckloads_per_mile : ℕ
  day2_miles_paved : ℕ
  day3_truckloads_per_mile : ℕ
  day3_miles_paved : ℕ
  day4_truckloads_per_mile : ℕ

/-- Calculates the number of barrels of pitch needed for the fourth day --/
def barrels_of_pitch_day4 (project : RoadworkProject) : ℕ :=
  let remaining_miles := project.total_length - (project.day1_miles_paved + project.day2_miles_paved + project.day3_miles_paved)
  let day4_truckloads := remaining_miles * project.day4_truckloads_per_mile
  let pitch_per_truck := project.gravel_per_truck / project.gravel_to_pitch_ratio
  let total_pitch := day4_truckloads * pitch_per_truck
  (total_pitch + 9) / 10  -- Round up to the nearest whole barrel

/-- Theorem stating that the number of barrels of pitch needed for the fourth day is 2 --/
theorem barrels_of_pitch_day4_is_two (project : RoadworkProject) 
  (h1 : project.total_length = 20)
  (h2 : project.gravel_per_truck = 2)
  (h3 : project.gravel_to_pitch_ratio = 5)
  (h4 : project.day1_truckloads_per_mile = 3)
  (h5 : project.day1_miles_paved = 4)
  (h6 : project.day2_truckloads_per_mile = 4)
  (h7 : project.day2_miles_paved = 7)
  (h8 : project.day3_truckloads_per_mile = 2)
  (h9 : project.day3_miles_paved = 5)
  (h10 : project.day4_truckloads_per_mile = 1) :
  barrels_of_pitch_day4 project = 2 := by
  sorry

end barrels_of_pitch_day4_is_two_l3488_348820


namespace sum_of_coefficients_of_fifth_power_l3488_348893

theorem sum_of_coefficients_of_fifth_power (a b : ℕ) (h : (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2) : a + b = 70 := by
  sorry

end sum_of_coefficients_of_fifth_power_l3488_348893


namespace isosceles_triangle_condition_l3488_348864

theorem isosceles_triangle_condition 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)
  (h5 : a = 2 * b * Real.cos C)
  (h6 : a > 0 ∧ b > 0 ∧ c > 0)
  : B = C := by
  sorry

end isosceles_triangle_condition_l3488_348864


namespace wilson_gained_money_l3488_348887

def watch_problem (selling_price : ℝ) (profit_percentage : ℝ) (loss_percentage : ℝ) : Prop :=
  let cost_price1 := selling_price / (1 + profit_percentage / 100)
  let cost_price2 := selling_price / (1 - loss_percentage / 100)
  let total_cost := cost_price1 + cost_price2
  let total_revenue := 2 * selling_price
  total_revenue > total_cost

theorem wilson_gained_money : watch_problem 150 25 15 := by
  sorry

end wilson_gained_money_l3488_348887


namespace max_cabbages_is_256_l3488_348879

structure Region where
  area : ℕ
  sunlight : ℕ
  water : ℕ

def is_suitable (r : Region) : Bool :=
  r.sunlight ≥ 4 ∧ r.water ≤ 16

def count_cabbages (regions : List Region) : ℕ :=
  (regions.filter is_suitable).foldl (fun acc r => acc + r.area) 0

def garden : List Region :=
  [
    ⟨30, 5, 15⟩,
    ⟨25, 6, 12⟩,
    ⟨35, 8, 18⟩,
    ⟨40, 4, 10⟩,
    ⟨20, 7, 14⟩
  ]

theorem max_cabbages_is_256 :
  count_cabbages garden + 181 = 256 :=
by sorry

end max_cabbages_is_256_l3488_348879


namespace expected_knowers_value_l3488_348863

/-- The number of scientists at the conference -/
def total_scientists : ℕ := 18

/-- The number of scientists who initially know the news -/
def initial_knowers : ℕ := 10

/-- The probability that an initially unknowing scientist learns the news during the coffee break -/
def prob_learn : ℚ := 10 / 17

/-- The expected number of scientists who know the news after the coffee break -/
def expected_knowers : ℚ := initial_knowers + (total_scientists - initial_knowers) * prob_learn

theorem expected_knowers_value : expected_knowers = 248 / 17 := by sorry

end expected_knowers_value_l3488_348863


namespace permutation_square_sum_bounds_l3488_348810

def is_permutation (a : Fin 10 → ℕ) : Prop :=
  ∀ i : Fin 10, ∃ j : Fin 10, a j = i.val + 1

theorem permutation_square_sum_bounds 
  (a b : Fin 10 → ℕ) 
  (ha : is_permutation a) 
  (hb : is_permutation b) :
  (∃ k : Fin 10, a k ^ 2 + b k ^ 2 ≥ 101) ∧
  (∃ k : Fin 10, a k ^ 2 + b k ^ 2 ≤ 61) :=
sorry

end permutation_square_sum_bounds_l3488_348810


namespace water_balloon_ratio_l3488_348842

theorem water_balloon_ratio : ∀ (anthony_balloons luke_balloons tom_balloons : ℕ),
  anthony_balloons = 44 →
  luke_balloons = anthony_balloons / 4 →
  tom_balloons = 33 →
  (tom_balloons : ℚ) / luke_balloons = 3 / 1 := by
  sorry

end water_balloon_ratio_l3488_348842


namespace age_difference_proof_l3488_348834

theorem age_difference_proof (jack_age bill_age : ℕ) : 
  jack_age = 3 * bill_age →
  (jack_age + 3) = 2 * (bill_age + 3) →
  jack_age - bill_age = 6 := by
sorry

end age_difference_proof_l3488_348834


namespace village_population_l3488_348803

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.05) * (1 - 0.15) = 2553 → P = 3162 :=
by sorry

end village_population_l3488_348803


namespace swimming_pool_width_l3488_348808

/-- Proves that the width of a rectangular swimming pool is 20 feet -/
theorem swimming_pool_width :
  ∀ (length width : ℝ) (water_removed : ℝ) (depth_lowered : ℝ),
    length = 60 →
    water_removed = 4500 →
    depth_lowered = 0.5 →
    water_removed / 7.5 = length * width * depth_lowered →
    width = 20 := by
  sorry

end swimming_pool_width_l3488_348808


namespace last_day_pages_for_specific_book_l3488_348866

/-- Calculates the number of pages read on the last day to complete a book -/
def pages_on_last_day (total_pages : ℕ) (pages_per_day : ℕ) (break_interval : ℕ) : ℕ :=
  let pages_per_cycle := pages_per_day * (break_interval - 1)
  let full_cycles := (total_pages / pages_per_cycle : ℕ)
  let pages_read_in_full_cycles := full_cycles * pages_per_cycle
  total_pages - pages_read_in_full_cycles

theorem last_day_pages_for_specific_book :
  pages_on_last_day 575 37 3 = 57 := by
  sorry

end last_day_pages_for_specific_book_l3488_348866


namespace egg_carton_problem_l3488_348872

theorem egg_carton_problem (abigail_eggs beatrice_eggs carson_eggs carton_size : ℕ) 
  (h1 : abigail_eggs = 48)
  (h2 : beatrice_eggs = 63)
  (h3 : carson_eggs = 27)
  (h4 : carton_size = 15) :
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  (total_eggs % carton_size = 3) ∧ (total_eggs / carton_size = 9) := by
  sorry

end egg_carton_problem_l3488_348872


namespace profit_maximizing_price_l3488_348830

/-- Represents the sales volume as a function of unit price -/
def sales_volume (x : ℝ) : ℝ := -2 * x + 100

/-- Represents the profit as a function of unit price -/
def profit (x : ℝ) : ℝ := (x - 20) * (sales_volume x)

/-- Theorem stating that the profit-maximizing price is 35 yuan -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), ∀ (y : ℝ), profit y ≤ profit x ∧ x = 35 := by
  sorry

end profit_maximizing_price_l3488_348830


namespace journey_speed_l3488_348823

/-- Proves the required speed for the second part of a journey given the total distance, total time, initial speed, and initial time. -/
theorem journey_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (h1 : total_distance = 24) 
  (h2 : total_time = 8) 
  (h3 : initial_speed = 4) 
  (h4 : initial_time = 4) 
  : 
  (total_distance - initial_speed * initial_time) / (total_time - initial_time) = 2 := by
  sorry

#check journey_speed

end journey_speed_l3488_348823


namespace sum_of_coefficients_l3488_348804

/-- The expansion of (1 - 1/(2x))^6 in terms of 1/x -/
def expansion (x : ℝ) (a : Fin 7 → ℝ) : Prop :=
  (1 - 1/(2*x))^6 = a 0 + a 1 * (1/x) + a 2 * (1/x)^2 + a 3 * (1/x)^3 + 
                    a 4 * (1/x)^4 + a 5 * (1/x)^5 + a 6 * (1/x)^6

/-- The sum of the coefficients a_3 and a_4 is equal to -25/16 -/
theorem sum_of_coefficients (x : ℝ) (a : Fin 7 → ℝ) 
  (h : expansion x a) : a 3 + a 4 = -25/16 := by
  sorry

end sum_of_coefficients_l3488_348804


namespace complex_fraction_simplification_l3488_348855

theorem complex_fraction_simplification :
  (3 + 3 * Complex.I) / (-4 + 5 * Complex.I) = 3 / 41 - 27 / 41 * Complex.I := by
  sorry

end complex_fraction_simplification_l3488_348855


namespace ratio_problem_l3488_348899

theorem ratio_problem (x y z : ℝ) 
  (h : y / z = z / x ∧ z / x = x / y ∧ x / y = 1 / 2) : 
  (x / (y * z)) / (y / (z * x)) = 4 := by
  sorry

end ratio_problem_l3488_348899


namespace composite_shape_sum_l3488_348807

/-- Represents a 3D geometric shape with faces, edges, and vertices -/
structure Shape where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- The initial triangular prism -/
def triangularPrism : Shape := ⟨5, 9, 6⟩

/-- Attaches a regular pentagonal prism to a quadrilateral face of the given shape -/
def attachPentagonalPrism (s : Shape) : Shape :=
  ⟨s.faces - 1 + 7, s.edges + 10, s.vertices + 5⟩

/-- Adds a pyramid to a pentagonal face of the given shape -/
def addPyramid (s : Shape) : Shape :=
  ⟨s.faces - 1 + 5, s.edges + 5, s.vertices + 1⟩

/-- Calculates the sum of faces, edges, and vertices of a shape -/
def sumFeatures (s : Shape) : ℕ :=
  s.faces + s.edges + s.vertices

/-- Theorem stating that the sum of features of the final composite shape is 51 -/
theorem composite_shape_sum :
  sumFeatures (addPyramid (attachPentagonalPrism triangularPrism)) = 51 := by
  sorry

end composite_shape_sum_l3488_348807


namespace pressure_calculation_l3488_348848

/-- Prove that given the ideal gas law and specific conditions, the pressure is 1125000 Pa -/
theorem pressure_calculation (v R T V : ℝ) (h1 : v = 30)
  (h2 : R = 8.31) (h3 : T = 300) (h4 : V = 0.06648) :
  v * R * T / V = 1125000 :=
by sorry

end pressure_calculation_l3488_348848


namespace solution_set_for_m_eq_2_range_of_m_l3488_348897

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - m|

-- Part I
theorem solution_set_for_m_eq_2 :
  {x : ℝ | f 2 x > 7 - |x - 1|} = {x : ℝ | x < -4 ∨ x > 5} := by sorry

-- Part II
theorem range_of_m :
  {m : ℝ | ∃ x : ℝ, f m x > 7 + |x - 1|} = {m : ℝ | m < -6 ∨ m > 8} := by sorry

end solution_set_for_m_eq_2_range_of_m_l3488_348897


namespace crypto_encoding_theorem_l3488_348840

/-- Represents the digits in the cryptographic encoding -/
inductive CryptoDigit
| V
| W
| X
| Y
| Z

/-- Represents a number in the cryptographic encoding -/
def CryptoNumber := List CryptoDigit

/-- Converts a CryptoNumber to its base 5 representation -/
def toBase5 : CryptoNumber → Nat := sorry

/-- Converts a base 5 number to base 10 -/
def base5ToBase10 : Nat → Nat := sorry

/-- The theorem to be proved -/
theorem crypto_encoding_theorem 
  (encode : Nat → CryptoNumber) 
  (n : Nat) :
  encode n = [CryptoDigit.V, CryptoDigit.Y, CryptoDigit.Z] ∧
  encode (n + 1) = [CryptoDigit.V, CryptoDigit.Y, CryptoDigit.X] ∧
  encode (n + 2) = [CryptoDigit.V, CryptoDigit.V, CryptoDigit.W] →
  base5ToBase10 (toBase5 [CryptoDigit.X, CryptoDigit.Y, CryptoDigit.Z]) = 108 := by
  sorry

end crypto_encoding_theorem_l3488_348840


namespace unique_representation_l3488_348859

theorem unique_representation (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
by sorry

end unique_representation_l3488_348859


namespace david_recreation_spending_l3488_348805

theorem david_recreation_spending :
  ∀ (last_week_wages : ℝ) (last_week_percent : ℝ),
    last_week_percent > 0 →
    (0.7 * last_week_wages * 0.2) = (0.7 * (last_week_percent / 100) * last_week_wages) →
    last_week_percent = 20 := by
  sorry

end david_recreation_spending_l3488_348805


namespace shaded_area_is_fifty_l3488_348889

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length and partitioning points -/
structure PartitionedSquare where
  sideLength : ℝ
  pointA : Point
  pointB : Point

/-- Calculates the area of the shaded diamond region in the partitioned square -/
def shadedAreaInPartitionedSquare (square : PartitionedSquare) : ℝ :=
  sorry

/-- The theorem stating that the shaded area in the given partitioned square is 50 square cm -/
theorem shaded_area_is_fifty (square : PartitionedSquare) 
  (h1 : square.sideLength = 10)
  (h2 : square.pointA = ⟨10/3, 10⟩)
  (h3 : square.pointB = ⟨20/3, 0⟩) : 
  shadedAreaInPartitionedSquare square = 50 := by
  sorry

end shaded_area_is_fifty_l3488_348889


namespace sum_of_abc_l3488_348809

theorem sum_of_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 24 := by
  sorry

end sum_of_abc_l3488_348809


namespace least_integer_with_8_factors_l3488_348832

/-- A function that counts the number of positive factors of a natural number -/
def count_factors (n : ℕ) : ℕ := sorry

/-- The property of being the least positive integer with exactly 8 factors -/
def is_least_with_8_factors (n : ℕ) : Prop :=
  count_factors n = 8 ∧ ∀ m : ℕ, m > 0 ∧ m < n → count_factors m ≠ 8

theorem least_integer_with_8_factors :
  is_least_with_8_factors 24 := by sorry

end least_integer_with_8_factors_l3488_348832


namespace correct_propositions_are_123_l3488_348886

-- Define the type for propositions
inductive GeometricProposition
  | frustum_def
  | frustum_edges
  | cone_def
  | hemisphere_rotation

-- Define a function to check if a proposition is correct
def is_correct_proposition (p : GeometricProposition) : Prop :=
  match p with
  | GeometricProposition.frustum_def => True
  | GeometricProposition.frustum_edges => True
  | GeometricProposition.cone_def => True
  | GeometricProposition.hemisphere_rotation => False

-- Define the set of all propositions
def all_propositions : Set GeometricProposition :=
  {GeometricProposition.frustum_def, GeometricProposition.frustum_edges, 
   GeometricProposition.cone_def, GeometricProposition.hemisphere_rotation}

-- Define the set of correct propositions
def correct_propositions : Set GeometricProposition :=
  {p ∈ all_propositions | is_correct_proposition p}

-- Theorem to prove
theorem correct_propositions_are_123 :
  correct_propositions = {GeometricProposition.frustum_def, 
                          GeometricProposition.frustum_edges, 
                          GeometricProposition.cone_def} := by
  sorry

end correct_propositions_are_123_l3488_348886


namespace binary_multiplication_division_equality_l3488_348874

def binary_to_nat (s : String) : Nat :=
  s.foldl (fun acc c => 2 * acc + c.toNat - '0'.toNat) 0

theorem binary_multiplication_division_equality : 
  (binary_to_nat "1100101" * binary_to_nat "101101" * binary_to_nat "110") / 
  binary_to_nat "100" = binary_to_nat "1111101011011011000" := by
  sorry

end binary_multiplication_division_equality_l3488_348874


namespace basketball_handshakes_l3488_348850

theorem basketball_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  team_size = 6 → num_teams = 2 → num_referees = 3 →
  (team_size * team_size) + (team_size * num_teams * num_referees) = 72 :=
by
  sorry

end basketball_handshakes_l3488_348850


namespace bowling_ball_volume_l3488_348845

theorem bowling_ball_volume :
  let sphere_diameter : ℝ := 40
  let hole1_depth : ℝ := 10
  let hole1_diameter : ℝ := 5
  let hole2_depth : ℝ := 12
  let hole2_diameter : ℝ := 4
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2)^3
  let hole1_volume := π * (hole1_diameter / 2)^2 * hole1_depth
  let hole2_volume := π * (hole2_diameter / 2)^2 * hole2_depth
  sphere_volume - hole1_volume - hole2_volume = 10556.17 * π :=
by sorry

end bowling_ball_volume_l3488_348845
