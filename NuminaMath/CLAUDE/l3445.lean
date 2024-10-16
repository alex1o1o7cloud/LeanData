import Mathlib

namespace NUMINAMATH_CALUDE_optimal_newspaper_sales_l3445_344598

/-- Represents the daily newspaper sales data --/
structure NewspaperSalesData where
  costPrice : ℝ
  sellingPrice : ℝ
  returnPrice : ℝ
  highSalesDays : ℕ
  highSalesAmount : ℕ
  lowSalesDays : ℕ
  lowSalesAmount : ℕ

/-- Calculates the monthly profit based on the number of copies purchased daily --/
def monthlyProfit (data : NewspaperSalesData) (dailyPurchase : ℕ) : ℝ :=
  let soldProfit := data.sellingPrice - data.costPrice
  let returnLoss := data.costPrice - data.returnPrice
  let totalSold := data.highSalesDays * (min dailyPurchase data.highSalesAmount) +
                   data.lowSalesDays * (min dailyPurchase data.lowSalesAmount)
  let totalReturned := (data.highSalesDays + data.lowSalesDays) * dailyPurchase - totalSold
  soldProfit * totalSold - returnLoss * totalReturned

/-- Theorem stating the optimal daily purchase and maximum monthly profit --/
theorem optimal_newspaper_sales (data : NewspaperSalesData)
  (h1 : data.costPrice = 0.12)
  (h2 : data.sellingPrice = 0.20)
  (h3 : data.returnPrice = 0.04)
  (h4 : data.highSalesDays = 20)
  (h5 : data.highSalesAmount = 400)
  (h6 : data.lowSalesDays = 10)
  (h7 : data.lowSalesAmount = 250) :
  (∀ x : ℕ, monthlyProfit data x ≤ monthlyProfit data 400) ∧
  monthlyProfit data 400 = 840 := by
  sorry


end NUMINAMATH_CALUDE_optimal_newspaper_sales_l3445_344598


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_equality_l3445_344543

theorem max_value_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) :
  2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ (Real.sqrt 2 - 1) / 2 := by
sorry

theorem max_value_equality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) :
  (2 * Real.sqrt (a * b) - 4 * a^2 - b^2 = (Real.sqrt 2 - 1) / 2) ↔ (a = 1/4 ∧ b = 1/2) := by
sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_equality_l3445_344543


namespace NUMINAMATH_CALUDE_smallest_prime_square_mod_six_is_five_l3445_344551

theorem smallest_prime_square_mod_six_is_five :
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p^2 % 6 = 1 ∧ 
    (∀ (q : ℕ), Nat.Prime q → q^2 % 6 = 1 → p ≤ q) ∧
    p = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_square_mod_six_is_five_l3445_344551


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l3445_344519

theorem fixed_point_of_line (m : ℝ) :
  (∀ m : ℝ, ∃! p : ℝ × ℝ, m * p.1 + p.2 - 1 + 2 * m = 0) →
  (∃ p : ℝ × ℝ, p = (-2, 1) ∧ ∀ m : ℝ, m * p.1 + p.2 - 1 + 2 * m = 0) :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l3445_344519


namespace NUMINAMATH_CALUDE_power_function_through_point_l3445_344503

/-- A power function passing through the point (33, 3) has exponent 3 -/
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x^α) →  -- f is a power function with exponent α
  f 33 = 3 →          -- f passes through the point (33, 3)
  α = 3 :=             -- the exponent α is equal to 3
by
  sorry


end NUMINAMATH_CALUDE_power_function_through_point_l3445_344503


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l3445_344557

theorem polar_to_rectangular :
  let r : ℝ := 3 * Real.sqrt 2
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 3 ∧ y = 3) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l3445_344557


namespace NUMINAMATH_CALUDE_line_segment_polar_equation_l3445_344555

/-- The polar equation of the line segment y = 1 - x where 0 ≤ x ≤ 1 -/
theorem line_segment_polar_equation (θ : Real) (ρ : Real) :
  (0 ≤ θ) ∧ (θ ≤ Real.pi / 2) →
  (ρ * Real.cos θ + ρ * Real.sin θ = 1) ↔
  (ρ * Real.sin θ = 1 - ρ * Real.cos θ) ∧
  (0 ≤ ρ * Real.cos θ) ∧ (ρ * Real.cos θ ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_polar_equation_l3445_344555


namespace NUMINAMATH_CALUDE_xiaoMingCarbonEmissions_l3445_344588

/-- The carbon dioxide emissions formula for household tap water usage -/
def carbonEmissions (x : ℝ) : ℝ := 0.9 * x

/-- Xiao Ming's household tap water usage in tons -/
def xiaoMingWaterUsage : ℝ := 10

theorem xiaoMingCarbonEmissions :
  carbonEmissions xiaoMingWaterUsage = 9 := by
  sorry

end NUMINAMATH_CALUDE_xiaoMingCarbonEmissions_l3445_344588


namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_primes_l3445_344561

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial) : ℕ := sorry

/-- Evaluate a polynomial at a given point -/
def eval (p : IntPolynomial) (x : ℕ) : ℤ := sorry

/-- The set of values that can be represented by a list of polynomials -/
def representableSet (polys : List IntPolynomial) : Set ℕ :=
  {n : ℕ | ∃ (p : IntPolynomial) (a : ℕ), p ∈ polys ∧ eval p a = n}

/-- The main theorem -/
theorem infinitely_many_non_representable_primes
  (n : ℕ)
  (polys : List IntPolynomial)
  (h_degree : ∀ p ∈ polys, degree p ≥ 2)
  : Set.Infinite {p : ℕ | Nat.Prime p ∧ p ∉ representableSet polys} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_primes_l3445_344561


namespace NUMINAMATH_CALUDE_expression_factorization_l3445_344564

theorem expression_factorization (a b c : ℝ) (h : c ≠ 0) :
  3 * a^3 * (b^2 - c^2) - 2 * b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2) =
  (a - b) * (b - c) * (c - a) * (3 * a^2 - 2 * b^2 - 3 * a^3 / c + c) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3445_344564


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_a_greater_than_one_l3445_344581

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∪ B = {x | 1 ≤ x < 10}
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 10} := by sorry

-- Theorem 2: (∁ₗA) ∩ B = {x | 7 ≤ x < 10}
theorem complement_A_intersect_B : (Set.compl A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a > 1
theorem a_greater_than_one (a : ℝ) (h : (A ∩ C a).Nonempty) : a > 1 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_a_greater_than_one_l3445_344581


namespace NUMINAMATH_CALUDE_simplify_fraction_l3445_344573

theorem simplify_fraction (m : ℝ) (hm : m ≠ 0) :
  ((m^2 - 3*m + 1) / m + 1) / ((m^2 - 1) / m) = (m - 1) / (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3445_344573


namespace NUMINAMATH_CALUDE_reading_time_per_page_l3445_344566

theorem reading_time_per_page 
  (planned_hours : ℝ) 
  (actual_fraction : ℝ) 
  (pages_read : ℕ) : 
  planned_hours = 3 → 
  actual_fraction = 3/4 → 
  pages_read = 9 → 
  (planned_hours * actual_fraction * 60) / pages_read = 15 := by
sorry

end NUMINAMATH_CALUDE_reading_time_per_page_l3445_344566


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l3445_344559

theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 
  let width := 2 * r
  let length := ratio * width
  width * length = 588 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l3445_344559


namespace NUMINAMATH_CALUDE_golden_ratio_bounds_l3445_344576

theorem golden_ratio_bounds : ∃ x : ℝ, x = Real.sqrt (1 + x) ∧ 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_bounds_l3445_344576


namespace NUMINAMATH_CALUDE_sum_of_xyz_is_718_l3445_344526

noncomputable def a : ℝ := -1 / Real.sqrt 3
noncomputable def b : ℝ := (3 + Real.sqrt 7) / 3

theorem sum_of_xyz_is_718 (ha : a^2 = 9/27) (hb : b^2 = (3 + Real.sqrt 7)^2 / 9)
  (ha_neg : a < 0) (hb_pos : b > 0)
  (h_expr : ∃ (x y z : ℕ+), (a + b)^3 = (x : ℝ) * Real.sqrt y / z) :
  ∃ (x y z : ℕ+), (a + b)^3 = (x : ℝ) * Real.sqrt y / z ∧ x + y + z = 718 :=
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_is_718_l3445_344526


namespace NUMINAMATH_CALUDE_water_surface_scientific_notation_correct_l3445_344596

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The area of water surface in China in km² -/
def water_surface_area : ℕ := 370000

/-- The scientific notation representation of the water surface area -/
def water_surface_scientific : ScientificNotation :=
  { coefficient := 3.7
    exponent := 5
    is_valid := by sorry }

/-- Theorem stating that the water surface area is correctly represented in scientific notation -/
theorem water_surface_scientific_notation_correct :
  (water_surface_scientific.coefficient * (10 : ℝ) ^ water_surface_scientific.exponent) = water_surface_area := by
  sorry

end NUMINAMATH_CALUDE_water_surface_scientific_notation_correct_l3445_344596


namespace NUMINAMATH_CALUDE_fraction_inequality_l3445_344520

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  b / a < (b + m) / (a + m) := by
sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3445_344520


namespace NUMINAMATH_CALUDE_frustum_smaller_cone_height_l3445_344509

-- Define the frustum
structure Frustum where
  height : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

-- Define the theorem
theorem frustum_smaller_cone_height (f : Frustum) 
  (h1 : f.height = 18)
  (h2 : f.lower_base_area = 144 * Real.pi)
  (h3 : f.upper_base_area = 16 * Real.pi) :
  ∃ (smaller_cone_height : ℝ), smaller_cone_height = 9 := by
  sorry

end NUMINAMATH_CALUDE_frustum_smaller_cone_height_l3445_344509


namespace NUMINAMATH_CALUDE_friends_weight_loss_l3445_344567

/-- The combined weight loss of two friends over different periods -/
theorem friends_weight_loss (aleesia_weekly_loss : ℝ) (aleesia_weeks : ℕ)
                             (alexei_weekly_loss : ℝ) (alexei_weeks : ℕ) :
  aleesia_weekly_loss = 1.5 ∧ 
  aleesia_weeks = 10 ∧
  alexei_weekly_loss = 2.5 ∧ 
  alexei_weeks = 8 →
  aleesia_weekly_loss * aleesia_weeks + alexei_weekly_loss * alexei_weeks = 35 := by
  sorry

end NUMINAMATH_CALUDE_friends_weight_loss_l3445_344567


namespace NUMINAMATH_CALUDE_x_value_theorem_l3445_344562

theorem x_value_theorem (x y : ℝ) (h : (x - 1) / x = (y^3 + 3*y^2 - 4) / (y^3 + 3*y^2 - 5)) :
  x = y^3 + 3*y^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_theorem_l3445_344562


namespace NUMINAMATH_CALUDE_complex_modulus_l3445_344539

theorem complex_modulus (a b : ℝ) (h : (1 + 2*a*Complex.I) * Complex.I = 1 - b*Complex.I) : 
  Complex.abs (a + b*Complex.I) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3445_344539


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3445_344541

theorem polynomial_coefficient_sum (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : 
  ((∀ x, (1 + m * x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) ∧
   a₁ - a₂ + a₃ - a₄ + a₅ - a₆ = -63) →
  m = 3 ∨ m = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3445_344541


namespace NUMINAMATH_CALUDE_sports_purchase_equation_l3445_344546

/-- Represents the cost of sports equipment purchases -/
structure SportsPurchase where
  volleyball_cost : ℝ  -- Cost of one volleyball in yuan
  shot_put_cost : ℝ    -- Cost of one shot put ball in yuan

/-- Conditions of the sports equipment purchase problem -/
def purchase_conditions (p : SportsPurchase) : Prop :=
  2 * p.volleyball_cost + 3 * p.shot_put_cost = 95 ∧
  5 * p.volleyball_cost + 7 * p.shot_put_cost = 230

/-- The theorem stating that the given system of linear equations 
    correctly represents the sports equipment purchase problem -/
theorem sports_purchase_equation (p : SportsPurchase) :
  purchase_conditions p ↔ 
  (2 * p.volleyball_cost + 3 * p.shot_put_cost = 95 ∧
   5 * p.volleyball_cost + 7 * p.shot_put_cost = 230) :=
by sorry

end NUMINAMATH_CALUDE_sports_purchase_equation_l3445_344546


namespace NUMINAMATH_CALUDE_modular_inverse_of_4_mod_21_l3445_344595

theorem modular_inverse_of_4_mod_21 : ∃ x : ℕ, x ≤ 20 ∧ (4 * x) % 21 = 1 :=
by
  use 16
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_4_mod_21_l3445_344595


namespace NUMINAMATH_CALUDE_sum_square_value_l3445_344592

theorem sum_square_value (x y : ℝ) 
  (h1 : x * (x + y) = 36) 
  (h2 : y * (x + y) = 72) : 
  (x + y)^2 = 108 := by
sorry

end NUMINAMATH_CALUDE_sum_square_value_l3445_344592


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l3445_344548

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Theorem: For a parabola x² = 2py (p > 0 and constant), if a line with slope 1
    passing through the focus intersects the parabola at points A and B,
    then the length of AB is 4p. -/
theorem parabola_intersection_length
  (p : ℝ)
  (hp : p > 0)
  (A B : ParabolaPoint)
  (h_parabola_A : A.x^2 = 2*p*A.y)
  (h_parabola_B : B.x^2 = 2*p*B.y)
  (h_line : B.y - A.y = B.x - A.x)
  (h_focus : ∃ (f : ℝ), A.y = A.x + f ∧ B.y = B.x + f ∧ f = p/2) :
  Real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2) = 4*p :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l3445_344548


namespace NUMINAMATH_CALUDE_distribute_four_balls_four_boxes_l3445_344586

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 35 ways to distribute 4 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_four_balls_four_boxes : distribute_balls 4 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_balls_four_boxes_l3445_344586


namespace NUMINAMATH_CALUDE_equal_height_locus_is_circle_l3445_344530

/-- Two flagpoles in a plane -/
structure Flagpoles where
  h : ℝ  -- height of first flagpole
  k : ℝ  -- height of second flagpole
  a : ℝ  -- half the distance between flagpoles

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The locus of points where flagpoles appear equally tall -/
def equalHeightLocus (f : Flagpoles) : Set Point :=
  {p : Point | ∃ (A B : Point), 
    -- A divides the line internally in ratio h:k
    A.x = -f.a + 2*f.a*f.h/(f.h + f.k) ∧ 
    -- B divides the line externally in ratio h:k
    B.x = -f.a - 2*f.a*f.h/(f.h - f.k) ∧ 
    -- P is on the circle with diameter AB
    (p.x - A.x)^2 + p.y^2 = (B.x - A.x)^2 / 4}

/-- The theorem statement -/
theorem equal_height_locus_is_circle (f : Flagpoles) :
  ∀ (p : Point), p ∈ equalHeightLocus f ↔ 
    (∃ (A B : Point), 
      A.x = -f.a + 2*f.a*f.h/(f.h + f.k) ∧
      B.x = -f.a - 2*f.a*f.h/(f.h - f.k) ∧
      (p.x - A.x)^2 + p.y^2 = (B.x - A.x)^2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_equal_height_locus_is_circle_l3445_344530


namespace NUMINAMATH_CALUDE_robin_hair_cut_l3445_344584

/-- Calculates the length of hair cut off given initial length, growth, and final length -/
def hair_cut_length (initial_length growth final_length : ℝ) : ℝ :=
  initial_length + growth - final_length

/-- Theorem stating that given the conditions in the problem, Robin cut off 11 inches of hair -/
theorem robin_hair_cut :
  let initial_length : ℝ := 16
  let growth : ℝ := 12
  let final_length : ℝ := 17
  hair_cut_length initial_length growth final_length = 11 := by
  sorry

end NUMINAMATH_CALUDE_robin_hair_cut_l3445_344584


namespace NUMINAMATH_CALUDE_laura_age_l3445_344577

theorem laura_age :
  ∃ (L : ℕ), 
    L > 0 ∧
    L < 100 ∧
    (L - 1) % 8 = 0 ∧
    (L + 1) % 7 = 0 ∧
    (∃ (A : ℕ), 
      A > L ∧
      A < 100 ∧
      (A - 1) % 8 = 0 ∧
      (A + 1) % 7 = 0) →
    L = 41 := by
  sorry

end NUMINAMATH_CALUDE_laura_age_l3445_344577


namespace NUMINAMATH_CALUDE_distance_to_line_not_greater_than_two_l3445_344580

/-- A structure representing a line in a plane -/
structure Line :=
  (points : Set Point)

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- The distance from a point to a line -/
def distanceToLine (p : Point) (l : Line) : ℝ := sorry

/-- Theorem: If a point P is outside a line l, and there are three points A, B, and C on l
    such that PA = 2, PB = 2.5, and PC = 3, then the distance from P to l is not greater than 2 -/
theorem distance_to_line_not_greater_than_two
  (P : Point) (l : Line) (A B C : Point)
  (h_P_outside : P ∉ l.points)
  (h_ABC_on_l : A ∈ l.points ∧ B ∈ l.points ∧ C ∈ l.points)
  (h_PA : distance P A = 2)
  (h_PB : distance P B = 2.5)
  (h_PC : distance P C = 3) :
  distanceToLine P l ≤ 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_line_not_greater_than_two_l3445_344580


namespace NUMINAMATH_CALUDE_function_properties_l3445_344545

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + 7 * Real.pi / 4) + Real.cos (x - 3 * Real.pi / 4)

theorem function_properties (α : ℝ) 
  (h1 : 0 < α) (h2 : α < 3 * Real.pi / 4) (h3 : f α = 6 / 5) :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧ 
    ∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  (∀ x : ℝ, f x ≥ -2) ∧
  (∃ x : ℝ, f x = -2) ∧
  f (2 * α) = 31 * Real.sqrt 2 / 25 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3445_344545


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l3445_344512

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_nonprime (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

def consecutive_nonprimes (start : ℕ) : Prop :=
  ∀ i : ℕ, i < 7 → is_nonprime (start + i)

theorem smallest_prime_after_seven_nonprimes :
  ∃ start : ℕ, consecutive_nonprimes start ∧ 
    is_prime 97 ∧
    (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ p > start + 6)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l3445_344512


namespace NUMINAMATH_CALUDE_charlie_metal_purchase_l3445_344593

/-- Given that Charlie needs a total amount of metal and has some in storage,
    this function calculates the additional amount he needs to buy. -/
def additional_metal_needed (total_needed : ℕ) (in_storage : ℕ) : ℕ :=
  total_needed - in_storage

/-- Theorem stating that given Charlie's specific situation, 
    he needs to buy 359 lbs of additional metal. -/
theorem charlie_metal_purchase : 
  additional_metal_needed 635 276 = 359 := by sorry

end NUMINAMATH_CALUDE_charlie_metal_purchase_l3445_344593


namespace NUMINAMATH_CALUDE_nested_sum_equals_geometric_sum_l3445_344554

def nested_sum : ℕ → ℕ
  | 0 => 5
  | n + 1 => 5 * (1 + nested_sum n)

theorem nested_sum_equals_geometric_sum : nested_sum 11 = 305175780 := by
  sorry

end NUMINAMATH_CALUDE_nested_sum_equals_geometric_sum_l3445_344554


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3445_344538

theorem polynomial_divisibility (a b : ℚ) : 
  (∀ x : ℚ, (x^2 - x - 2) ∣ (a * x^4 + b * x^2 + 1)) ↔ 
  (a = 1/4 ∧ b = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3445_344538


namespace NUMINAMATH_CALUDE_factors_of_243_times_5_l3445_344542

-- Define the number we're working with
def n : Nat := 243 * 5

-- Define a function to count the number of distinct positive factors
def countDistinctPositiveFactors (x : Nat) : Nat :=
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).card

-- State the theorem
theorem factors_of_243_times_5 : countDistinctPositiveFactors n = 12 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_243_times_5_l3445_344542


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_area_l3445_344575

theorem right_triangle_hypotenuse_and_area 
  (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_a : a = 60) (h_b : b = 80) : 
  c = 100 ∧ (1/2 * a * b) = 2400 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_area_l3445_344575


namespace NUMINAMATH_CALUDE_custom_operation_equality_l3445_344523

-- Define the custom operation
def delta (a b : ℝ) : ℝ := a^3 - 2*b

-- State the theorem
theorem custom_operation_equality :
  let x := delta 6 8
  let y := delta 2 7
  delta (5^x) (2^y) = (5^200)^3 - 1/32 := by sorry

end NUMINAMATH_CALUDE_custom_operation_equality_l3445_344523


namespace NUMINAMATH_CALUDE_teal_survey_l3445_344556

theorem teal_survey (total : ℕ) (more_green : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 150)
  (h_more_green : more_green = 90)
  (h_both : both = 40)
  (h_neither : neither = 25) :
  total - (more_green - both + both + neither) = 75 :=
sorry

end NUMINAMATH_CALUDE_teal_survey_l3445_344556


namespace NUMINAMATH_CALUDE_range_of_m_l3445_344511

def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem range_of_m : 
  (∃ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) → 
  (∀ m : ℝ, (1 < m ∧ m ≤ 2) ∨ m ≥ 3 ↔ (p m ∨ q m) ∧ ¬(p m ∧ q m)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3445_344511


namespace NUMINAMATH_CALUDE_train_speed_l3445_344527

/-- Proves that a train with given parameters has a specific speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) : 
  train_length = 300 →
  crossing_time = 9 →
  man_speed = 3 →
  ∃ (train_speed : ℝ), train_speed = 117 ∧ 
    (train_speed * 1000 / 3600 + man_speed * 1000 / 3600) * crossing_time = train_length :=
by sorry


end NUMINAMATH_CALUDE_train_speed_l3445_344527


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l3445_344590

theorem repeating_decimal_division (a b : ℚ) :
  a = 81 / 99 →
  b = 36 / 99 →
  a / b = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l3445_344590


namespace NUMINAMATH_CALUDE_ellipse_area_l3445_344505

def ellipse_equation (x y : ℝ) : Prop :=
  2 * x^2 + 8 * x + 3 * y^2 - 9 * y + 12 = 0

theorem ellipse_area : 
  ∃ (A : ℝ), A = Real.pi * Real.sqrt 6 / 6 ∧ 
  ∀ (x y : ℝ), ellipse_equation x y → A = Real.pi * Real.sqrt ((1 / 2) * (1 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_area_l3445_344505


namespace NUMINAMATH_CALUDE_circle_area_l3445_344508

/-- The area of the circle defined by the equation 3x^2 + 3y^2 + 12x - 9y - 27 = 0 is 49/4 * π -/
theorem circle_area (x y : ℝ) : 
  (3 * x^2 + 3 * y^2 + 12 * x - 9 * y - 27 = 0) → 
  (∃ (center : ℝ × ℝ) (r : ℝ), 
    ((x - center.1)^2 + (y - center.2)^2 = r^2) ∧ 
    (π * r^2 = 49/4 * π)) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l3445_344508


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3445_344518

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.abs (1 + Complex.I)) : 
  z.im = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3445_344518


namespace NUMINAMATH_CALUDE_sequence_characterization_l3445_344568

theorem sequence_characterization (a : ℕ → ℕ) :
  (∀ n : ℕ, n ≥ 1 → a (n + 2) * (a (n + 1) - 1) = a n * (a (n + 1) + 1)) →
  ∃ k : ℕ, ∀ n : ℕ, n ≥ 1 → a n = k + n :=
sorry

end NUMINAMATH_CALUDE_sequence_characterization_l3445_344568


namespace NUMINAMATH_CALUDE_f_properties_l3445_344507

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 - 2*(x + 1) + 7

-- Theorem statement
theorem f_properties :
  (f 2 = 10) ∧
  (∀ a, f a = a^2 + 6) ∧
  (∀ x, f x = x^2 + 6) ∧
  (∀ x, f (x + 1) = x^2 + 2*x + 7) ∧
  (∀ y, y ∈ Set.range (λ x => f (x + 1)) ↔ y ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3445_344507


namespace NUMINAMATH_CALUDE_quadratic_function_problem_l3445_344594

/-- Given a quadratic function f(x) = x^2 + ax + b, if f(f(x) + x) / f(x) = x^2 + 2023x + 3000,
    then a = 2021 and b = 979. -/
theorem quadratic_function_problem (a b : ℝ) : 
  (let f := fun x => x^2 + a*x + b
   (∀ x, (f (f x + x)) / (f x) = x^2 + 2023*x + 3000)) → 
  (a = 2021 ∧ b = 979) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_problem_l3445_344594


namespace NUMINAMATH_CALUDE_cylinder_area_ratio_l3445_344534

theorem cylinder_area_ratio (r : ℝ) (h : ℝ) (h_positive : h > 0) (r_positive : r > 0) (h_eq_2r : h = 2 * r) :
  (2 * Real.pi * r * h) / (2 * Real.pi * r^2 + 2 * Real.pi * r * h) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_area_ratio_l3445_344534


namespace NUMINAMATH_CALUDE_inscribed_isosceles_tangent_circle_radius_l3445_344529

/-- Given an isosceles triangle inscribed in a circle, with a second circle
    tangent to both legs of the triangle and the first circle, this theorem
    states the radius of the second circle in terms of the base and base angle
    of the isosceles triangle. -/
theorem inscribed_isosceles_tangent_circle_radius
  (a : ℝ) (α : ℝ) (h_a_pos : a > 0) (h_α_pos : α > 0) (h_α_lt_pi_2 : α < π / 2) :
  ∃ (r : ℝ),
    r > 0 ∧
    r = a / (2 * Real.sin α * (1 + Real.cos α)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_isosceles_tangent_circle_radius_l3445_344529


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l3445_344517

/-- Given the total cloth length, total selling price, and loss per metre, 
    calculate the cost price for one metre of cloth. -/
theorem cost_price_per_metre 
  (total_length : ℕ) 
  (total_selling_price : ℕ) 
  (loss_per_metre : ℕ) 
  (h1 : total_length = 200)
  (h2 : total_selling_price = 12000)
  (h3 : loss_per_metre = 12) : 
  (total_selling_price + total_length * loss_per_metre) / total_length = 72 := by
  sorry

#check cost_price_per_metre

end NUMINAMATH_CALUDE_cost_price_per_metre_l3445_344517


namespace NUMINAMATH_CALUDE_denominator_of_0_34_l3445_344574

def decimal_to_fraction (d : ℚ) : ℕ × ℕ := sorry

theorem denominator_of_0_34 :
  (decimal_to_fraction 0.34).2 = 100 := by sorry

end NUMINAMATH_CALUDE_denominator_of_0_34_l3445_344574


namespace NUMINAMATH_CALUDE_columbia_arrangements_l3445_344537

def columbia_letters : Nat := 9
def repeated_i : Nat := 2
def repeated_u : Nat := 2

theorem columbia_arrangements :
  (columbia_letters.factorial) / (repeated_i.factorial * repeated_u.factorial) = 90720 := by
  sorry

end NUMINAMATH_CALUDE_columbia_arrangements_l3445_344537


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3445_344513

theorem sufficient_not_necessary_condition :
  ∃ (q : ℝ → Prop), 
    (∀ x, q x → x^2 - x - 6 < 0) ∧ 
    (∃ x, x^2 - x - 6 < 0 ∧ ¬(q x)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3445_344513


namespace NUMINAMATH_CALUDE_problem_2_l3445_344504

def f (x a : ℝ) : ℝ := |x - a|

theorem problem_2 (a m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_solution_set : ∀ x, f x a ≤ 2 ↔ -1 ≤ x ∧ x ≤ 3)
  (h_equation : m + 2*n = 2*m*n - 3*a) : 
  m + 2*n ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_problem_2_l3445_344504


namespace NUMINAMATH_CALUDE_second_person_receives_345_l3445_344552

/-- The total amount of money distributed -/
def total_amount : ℕ := 1000

/-- The sequence of distributions -/
def distribution_sequence (n : ℕ) : ℕ := n

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The largest n such that the sum of the first n natural numbers is at most the total amount -/
def max_n : ℕ := 44

/-- The amount received by the second person (Bernardo) -/
def amount_received_by_second : ℕ := 345

/-- Theorem stating that the second person (Bernardo) receives 345 reais -/
theorem second_person_receives_345 :
  (∀ n : ℕ, n ≤ max_n → sum_first_n n ≤ total_amount) →
  (∀ k : ℕ, k ≤ 15 → distribution_sequence (3*k - 1) ≤ max_n) →
  amount_received_by_second = 345 := by
  sorry

end NUMINAMATH_CALUDE_second_person_receives_345_l3445_344552


namespace NUMINAMATH_CALUDE_workshop_salary_problem_l3445_344563

theorem workshop_salary_problem (total_workers : ℕ) (avg_salary : ℕ) 
  (num_technicians : ℕ) (avg_salary_technicians : ℕ) :
  total_workers = 21 →
  avg_salary = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 12000 →
  let remaining_workers := total_workers - num_technicians
  let total_salary := total_workers * avg_salary
  let technicians_salary := num_technicians * avg_salary_technicians
  let remaining_salary := total_salary - technicians_salary
  (remaining_salary / remaining_workers : ℚ) = 6000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_salary_problem_l3445_344563


namespace NUMINAMATH_CALUDE_unique_z_value_l3445_344516

theorem unique_z_value : ∃! z : ℝ,
  (∃ x : ℤ, x = ⌊z⌋ ∧ 3 * x^2 + 19 * x - 84 = 0) ∧
  (∃ y : ℝ, 0 ≤ y ∧ y < 1 ∧ y = z - ⌊z⌋ ∧ 4 * y^2 - 14 * y + 6 = 0) ∧
  z = -11 := by
sorry

end NUMINAMATH_CALUDE_unique_z_value_l3445_344516


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l3445_344597

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (k : ℕ), k = 15 ∧ 
  (∀ m : ℕ, m > k → ¬(m ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13))) ∧
  (k ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l3445_344597


namespace NUMINAMATH_CALUDE_consecutive_color_draws_probability_l3445_344583

def blue_chips : ℕ := 4
def green_chips : ℕ := 3
def red_chips : ℕ := 5
def total_chips : ℕ := blue_chips + green_chips + red_chips

def probability_consecutive_color_draws : ℚ :=
  (Nat.factorial 3 * Nat.factorial blue_chips * Nat.factorial green_chips * Nat.factorial red_chips) /
  Nat.factorial total_chips

theorem consecutive_color_draws_probability :
  probability_consecutive_color_draws = 1 / 4620 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_color_draws_probability_l3445_344583


namespace NUMINAMATH_CALUDE_jerome_money_theorem_l3445_344560

/-- Calculates the amount of money Jerome has left after giving money to Meg and Bianca. -/
def jerome_money_left (initial_money : ℕ) (meg_amount : ℕ) (bianca_multiplier : ℕ) : ℕ :=
  initial_money - meg_amount - (meg_amount * bianca_multiplier)

/-- Proves that Jerome has $54 left after giving money to Meg and Bianca. -/
theorem jerome_money_theorem :
  let initial_money := 43 * 2
  let meg_amount := 8
  let bianca_multiplier := 3
  jerome_money_left initial_money meg_amount bianca_multiplier = 54 := by
  sorry

end NUMINAMATH_CALUDE_jerome_money_theorem_l3445_344560


namespace NUMINAMATH_CALUDE_equation_solution_l3445_344599

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ 9 - 3 / (1 / x) + 3 = 3 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3445_344599


namespace NUMINAMATH_CALUDE_cream_ratio_is_15_23_l3445_344515

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

end NUMINAMATH_CALUDE_cream_ratio_is_15_23_l3445_344515


namespace NUMINAMATH_CALUDE_sum_of_first_n_naturals_l3445_344578

theorem sum_of_first_n_naturals (n : ℕ) : 
  (n * (n + 1)) / 2 = 3675 ↔ n = 81 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_n_naturals_l3445_344578


namespace NUMINAMATH_CALUDE_track_walking_speed_l3445_344524

theorem track_walking_speed 
  (track_width : ℝ) 
  (time_difference : ℝ) 
  (inner_length : ℝ → ℝ → ℝ) 
  (outer_length : ℝ → ℝ → ℝ) :
  track_width = 6 →
  time_difference = 48 →
  (∀ a b, inner_length a b = 2 * a + 2 * π * b) →
  (∀ a b, outer_length a b = 2 * a + 2 * π * (b + track_width)) →
  ∃ s a b, 
    outer_length a b / s = inner_length a b / s + time_difference ∧
    s = π / 4 :=
by sorry

end NUMINAMATH_CALUDE_track_walking_speed_l3445_344524


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l3445_344540

theorem square_perimeter_problem (perimeter_C : ℝ) (area_C area_D : ℝ) :
  perimeter_C = 32 →
  area_D = area_C / 8 →
  ∃ (side_D : ℝ), side_D * side_D = area_D ∧ 4 * side_D = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l3445_344540


namespace NUMINAMATH_CALUDE_trick_decks_total_spent_l3445_344528

/-- The total amount spent by Victor and his friend on trick decks -/
def total_spent (cost_per_deck : ℕ) (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  cost_per_deck * (victor_decks + friend_decks)

/-- Theorem: Victor and his friend spent 64 dollars on trick decks -/
theorem trick_decks_total_spent :
  total_spent 8 6 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_trick_decks_total_spent_l3445_344528


namespace NUMINAMATH_CALUDE_number_equals_scientific_rep_l3445_344536

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be represented -/
def number : ℕ := 1300000

/-- The scientific notation representation of the number -/
def scientific_rep : ScientificNotation :=
  { coefficient := 1.3
  , exponent := 6
  , h_coeff := by sorry }

theorem number_equals_scientific_rep :
  (number : ℝ) = scientific_rep.coefficient * (10 : ℝ) ^ scientific_rep.exponent :=
by sorry

end NUMINAMATH_CALUDE_number_equals_scientific_rep_l3445_344536


namespace NUMINAMATH_CALUDE_range_of_f_l3445_344547

def f (x : ℝ) := x^2 - 2*x + 4

theorem range_of_f :
  ∀ y ∈ Set.Icc 3 7, ∃ x ∈ Set.Icc 0 3, f x = y ∧
  ∀ x ∈ Set.Icc 0 3, ∃ y ∈ Set.Icc 3 7, f x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3445_344547


namespace NUMINAMATH_CALUDE_johnny_works_four_and_half_hours_l3445_344501

/-- Represents Johnny's dog walking business --/
structure DogWalker where
  dogs_per_walk : ℕ
  pay_30min : ℕ
  pay_60min : ℕ
  long_walks_per_day : ℕ
  work_days_per_week : ℕ
  weekly_earnings : ℕ

/-- Calculates the number of hours Johnny works per day --/
def hours_worked_per_day (dw : DogWalker) : ℚ :=
  let long_walk_earnings := dw.pay_60min * (dw.long_walks_per_day / dw.dogs_per_walk)
  let weekly_long_walk_earnings := long_walk_earnings * dw.work_days_per_week
  let weekly_short_walk_earnings := dw.weekly_earnings - weekly_long_walk_earnings
  let short_walks_per_week := weekly_short_walk_earnings / dw.pay_30min
  let short_walks_per_day := short_walks_per_week / dw.work_days_per_week
  let short_walk_sets_per_day := short_walks_per_day / dw.dogs_per_walk
  ((dw.long_walks_per_day / dw.dogs_per_walk) * 60 + short_walk_sets_per_day * 30) / 60

/-- Theorem stating that Johnny works 4.5 hours per day --/
theorem johnny_works_four_and_half_hours
  (johnny : DogWalker)
  (h1 : johnny.dogs_per_walk = 3)
  (h2 : johnny.pay_30min = 15)
  (h3 : johnny.pay_60min = 20)
  (h4 : johnny.long_walks_per_day = 6)
  (h5 : johnny.work_days_per_week = 5)
  (h6 : johnny.weekly_earnings = 1500) :
  hours_worked_per_day johnny = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_johnny_works_four_and_half_hours_l3445_344501


namespace NUMINAMATH_CALUDE_marbles_given_to_brother_l3445_344565

theorem marbles_given_to_brother 
  (total_marbles : ℕ) 
  (mario_ratio : ℕ) 
  (manny_ratio : ℕ) 
  (manny_current : ℕ) 
  (h1 : total_marbles = 36)
  (h2 : mario_ratio = 4)
  (h3 : manny_ratio = 5)
  (h4 : manny_current = 18) :
  (manny_ratio * total_marbles) / (mario_ratio + manny_ratio) - manny_current = 2 :=
sorry

end NUMINAMATH_CALUDE_marbles_given_to_brother_l3445_344565


namespace NUMINAMATH_CALUDE_factorization_equality_l3445_344502

theorem factorization_equality (x : ℝ) : 
  (3 * x^3 + 48 * x^2 - 14) - (-9 * x^3 + 2 * x^2 - 14) = 2 * x^2 * (6 * x + 23) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3445_344502


namespace NUMINAMATH_CALUDE_money_distribution_l3445_344553

theorem money_distribution (a b c : ℕ) 
  (h1 : a + b + c = 1000)
  (h2 : a + c = 700)
  (h3 : b + c = 600) :
  c = 300 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3445_344553


namespace NUMINAMATH_CALUDE_speed_from_x_to_y_l3445_344587

/-- Proves that given two towns and specific travel conditions, the speed from x to y is 60 km/hr -/
theorem speed_from_x_to_y (D : ℝ) (V : ℝ) (h : D > 0) : 
  (2 * D) / (D / V + D / 36) = 45 → V = 60 := by
  sorry

end NUMINAMATH_CALUDE_speed_from_x_to_y_l3445_344587


namespace NUMINAMATH_CALUDE_only_log23_not_computable_l3445_344579

-- Define the given logarithm values
def log27 : ℝ := 1.4314
def log32 : ℝ := 1.5052

-- Define a function to represent the computability of a logarithm
def is_computable (x : ℝ) : Prop := 
  ∃ (f : ℝ → ℝ → ℝ), f log27 log32 = Real.log x

-- State the theorem
theorem only_log23_not_computable :
  ¬(is_computable 23) ∧ 
  (is_computable (9/8)) ∧ 
  (is_computable 28) ∧ 
  (is_computable 800) ∧ 
  (is_computable 0.45) := by
  sorry

end NUMINAMATH_CALUDE_only_log23_not_computable_l3445_344579


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l3445_344514

theorem complex_modulus_theorem (z : ℂ) (h : z + 3 / z = 0) : Complex.abs z = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l3445_344514


namespace NUMINAMATH_CALUDE_basketball_average_points_l3445_344500

/-- Given a basketball player who scored 60 points in 5 games, 
    prove that their average points per game is 12. -/
theorem basketball_average_points (total_points : ℕ) (num_games : ℕ) 
  (h1 : total_points = 60) (h2 : num_games = 5) : 
  total_points / num_games = 12 := by
  sorry

end NUMINAMATH_CALUDE_basketball_average_points_l3445_344500


namespace NUMINAMATH_CALUDE_subject_selection_ways_l3445_344521

/-- The number of ways to choose 1 subject from 2 options -/
def physics_history_choices : Nat := 2

/-- The number of subjects to choose from for the remaining two subjects -/
def remaining_subject_options : Nat := 4

/-- The number of subjects to be chosen from the remaining options -/
def subjects_to_choose : Nat := 2

/-- Calculates the number of ways to choose k items from n options -/
def choose (n k : Nat) : Nat :=
  Nat.choose n k

theorem subject_selection_ways :
  physics_history_choices * choose remaining_subject_options subjects_to_choose = 12 := by
  sorry

end NUMINAMATH_CALUDE_subject_selection_ways_l3445_344521


namespace NUMINAMATH_CALUDE_quadratic_roots_triangle_range_l3445_344591

/-- Given a quadratic equation x^2 - 2x + m = 0 with two real roots a and b,
    where a, b, and 1 can form the sides of a triangle, prove that 3/4 < m ≤ 1 --/
theorem quadratic_roots_triangle_range (m : ℝ) (a b : ℝ) : 
  (∀ x, x^2 - 2*x + m = 0 ↔ x = a ∨ x = b) → 
  (a + b > 1 ∧ a > 0 ∧ b > 0 ∧ 1 > 0 ∧ a + 1 > b ∧ b + 1 > a) →
  (3/4 < m ∧ m ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_triangle_range_l3445_344591


namespace NUMINAMATH_CALUDE_triangle_area_l3445_344544

theorem triangle_area (A B C : Real) (R : Real) : 
  A = π / 7 → B = 2 * π / 7 → C = 4 * π / 7 → R = 1 →
  2 * R^2 * Real.sin A * Real.sin B * Real.sin C = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3445_344544


namespace NUMINAMATH_CALUDE_kendra_initial_money_l3445_344531

def wooden_toy_price : ℕ := 20
def hat_price : ℕ := 10
def wooden_toys_bought : ℕ := 2
def hats_bought : ℕ := 3
def change_received : ℕ := 30

theorem kendra_initial_money :
  wooden_toy_price * wooden_toys_bought + hat_price * hats_bought + change_received = 100 :=
by sorry

end NUMINAMATH_CALUDE_kendra_initial_money_l3445_344531


namespace NUMINAMATH_CALUDE_age_difference_l3445_344535

theorem age_difference (a b c d : ℤ) 
  (total_ab_cd : a + b = c + d + 20)
  (total_bd_ac : b + d = a + c + 10) :
  d = a - 5 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3445_344535


namespace NUMINAMATH_CALUDE_jacob_peter_age_difference_l3445_344506

/-- Given that Peter's age 10 years ago was one-third of Jacob's age at that time,
    and Peter is currently 16 years old, prove that Jacob's current age is 12 years
    more than Peter's current age. -/
theorem jacob_peter_age_difference :
  ∀ (peter_age_10_years_ago jacob_age_10_years_ago : ℕ),
  peter_age_10_years_ago = jacob_age_10_years_ago / 3 →
  peter_age_10_years_ago + 10 = 16 →
  jacob_age_10_years_ago + 10 - (peter_age_10_years_ago + 10) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_jacob_peter_age_difference_l3445_344506


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l3445_344572

/-- The number of students taking both Geometry and History -/
def both_geometry_history : ℕ := 15

/-- The total number of students taking Geometry -/
def total_geometry : ℕ := 30

/-- The number of students taking History only -/
def history_only : ℕ := 15

/-- The number of students taking both Geometry and Science -/
def both_geometry_science : ℕ := 8

/-- The number of students taking Science only -/
def science_only : ℕ := 10

/-- Theorem stating that the number of students taking only one subject is 32 -/
theorem students_taking_one_subject :
  (total_geometry - both_geometry_history - both_geometry_science) + history_only + science_only = 32 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l3445_344572


namespace NUMINAMATH_CALUDE_value_of_B_l3445_344570

theorem value_of_B : ∃ B : ℝ, (3 * B + 2 = 20) ∧ (B = 6) := by
  sorry

end NUMINAMATH_CALUDE_value_of_B_l3445_344570


namespace NUMINAMATH_CALUDE_new_average_after_exclusion_l3445_344510

theorem new_average_after_exclusion (total_students : ℕ) (initial_average : ℚ) 
  (excluded_students : ℕ) (excluded_average : ℚ) (new_average : ℚ) : 
  total_students = 20 →
  initial_average = 90 →
  excluded_students = 2 →
  excluded_average = 45 →
  new_average = (total_students * initial_average - excluded_students * excluded_average) / 
    (total_students - excluded_students) →
  new_average = 95 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_exclusion_l3445_344510


namespace NUMINAMATH_CALUDE_integral_proof_l3445_344571

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * log (abs (x - 2)) - 1 / (2 * (x - 1)^2)

theorem integral_proof (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  deriv f x = (2 * x^3 - 6 * x^2 + 7 * x - 4) / ((x - 2) * (x - 1)^3) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l3445_344571


namespace NUMINAMATH_CALUDE_sector_central_angle_l3445_344549

/-- Given a sector with perimeter 8 and area 4, its central angle is 2 radians -/
theorem sector_central_angle (l r : ℝ) (h1 : 2 * r + l = 8) (h2 : (1 / 2) * l * r = 4) :
  l / r = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3445_344549


namespace NUMINAMATH_CALUDE_flensburgian_iff_even_l3445_344582

/-- A set of equations is Flensburgian if there exists an i ∈ {1, 2, 3} such that
    for every solution where all variables are pairwise different, x_i > x_j for all j ≠ i -/
def IsFlensburgian (f : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ i : Fin 3, ∀ x y z : ℝ, f x y z → x ≠ y → y ≠ z → z ≠ x →
    match i with
    | 0 => x > y ∧ x > z
    | 1 => y > x ∧ y > z
    | 2 => z > x ∧ z > y

/-- The set of equations a^n + b = a and c^(n+1) + b^2 = ab -/
def EquationSet (n : ℕ) (a b c : ℝ) : Prop :=
  a^n + b = a ∧ c^(n+1) + b^2 = a * b

theorem flensburgian_iff_even (n : ℕ) (h : n ≥ 2) :
  IsFlensburgian (EquationSet n) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_flensburgian_iff_even_l3445_344582


namespace NUMINAMATH_CALUDE_choose_15_4_l3445_344550

theorem choose_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_15_4_l3445_344550


namespace NUMINAMATH_CALUDE_constant_speed_running_time_l3445_344522

/-- Given a constant running speed, if it takes 30 minutes to run 5 miles,
    then it will take 18 minutes to run 3 miles. -/
theorem constant_speed_running_time
  (speed : ℝ)
  (h1 : speed > 0)
  (h2 : 5 / speed = 30) :
  3 / speed = 18 := by
  sorry

end NUMINAMATH_CALUDE_constant_speed_running_time_l3445_344522


namespace NUMINAMATH_CALUDE_solution_range_l3445_344558

-- Define the system of inequalities
def system (x m : ℝ) : Prop :=
  (6 - 3*(x + 1) < x - 9) ∧ 
  (x - m > -1) ∧ 
  (x > 3)

-- Theorem statement
theorem solution_range (m : ℝ) : 
  (∀ x, system x m → x > 3) → m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l3445_344558


namespace NUMINAMATH_CALUDE_decreases_by_integer_factor_iff_in_valid_set_l3445_344533

def decreases_by_integer_factor (x : ℕ+) : Prop :=
  ∃ f : ℕ+, (x : ℕ) / 10 = x / f ∧ f > 1

def valid_integers : Finset ℕ+ :=
  {11, 22, 33, 44, 55, 66, 77, 88, 99, 12, 24, 36, 48, 13, 26, 39, 14, 28, 15, 16, 17, 18, 19}

theorem decreases_by_integer_factor_iff_in_valid_set (x : ℕ+) :
  decreases_by_integer_factor x ↔ x ∈ valid_integers := by
  sorry

end NUMINAMATH_CALUDE_decreases_by_integer_factor_iff_in_valid_set_l3445_344533


namespace NUMINAMATH_CALUDE_variance_invariant_under_translation_l3445_344585

def variance (data : List ℝ) : ℝ := sorry

theorem variance_invariant_under_translation (data : List ℝ) (c : ℝ) :
  variance data = variance (data.map (λ x => x - c)) := by sorry

end NUMINAMATH_CALUDE_variance_invariant_under_translation_l3445_344585


namespace NUMINAMATH_CALUDE_solve_for_c_l3445_344589

theorem solve_for_c (y : ℝ) (h1 : y > 0) : 
  ∃ c : ℝ, (7 * y) / 20 + (c * y) / 10 = 0.6499999999999999 * y ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l3445_344589


namespace NUMINAMATH_CALUDE_y_satisfies_differential_equation_l3445_344525

noncomputable def y (x : ℝ) : ℝ := x / (x - 1) + x^2

theorem y_satisfies_differential_equation (x : ℝ) :
  x * (x - 1) * (deriv y x) + y x = x^2 * (2 * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_y_satisfies_differential_equation_l3445_344525


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3445_344569

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ a ≥ 0, ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ∧
  (∃ a < 0, ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3445_344569


namespace NUMINAMATH_CALUDE_relative_errors_equal_l3445_344532

theorem relative_errors_equal (length1 length2 error1 error2 : ℝ) 
  (h1 : length1 = 20)
  (h2 : length2 = 150)
  (h3 : error1 = 0.04)
  (h4 : error2 = 0.3) :
  error1 / length1 = error2 / length2 := by
  sorry

end NUMINAMATH_CALUDE_relative_errors_equal_l3445_344532
