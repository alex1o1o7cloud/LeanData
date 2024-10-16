import Mathlib

namespace NUMINAMATH_CALUDE_wishing_pond_problem_l2468_246852

/-- The number of coins each person throws into the pond -/
structure CoinCounts where
  cindy_dimes : ℕ
  eric_quarters : ℕ
  garrick_nickels : ℕ
  ivy_pennies : ℕ

/-- The value of each coin type in cents -/
def coin_values : CoinCounts → ℕ
  | ⟨cd, eq, gn, ip⟩ => cd * 10 + eq * 25 + gn * 5 + ip * 1

/-- The problem statement -/
theorem wishing_pond_problem (coins : CoinCounts) : 
  coins.eric_quarters = 3 →
  coins.garrick_nickels = 8 →
  coins.ivy_pennies = 60 →
  coin_values coins = 200 →
  coins.cindy_dimes = 2 := by
  sorry

#eval coin_values ⟨2, 3, 8, 60⟩

end NUMINAMATH_CALUDE_wishing_pond_problem_l2468_246852


namespace NUMINAMATH_CALUDE_roots_sum_square_l2468_246829

theorem roots_sum_square (α β : ℝ) : 
  (α^2 - α - 2006 = 0) → 
  (β^2 - β - 2006 = 0) → 
  (α + β = 1) →
  α + β^2 = 2007 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_square_l2468_246829


namespace NUMINAMATH_CALUDE_product_of_powers_l2468_246856

theorem product_of_powers (y : ℝ) (hy : y ≠ 0) :
  (18 * y^3) * (4 * y^2) * (1 / (2*y)^3) = 9 * y^2 := by
sorry

end NUMINAMATH_CALUDE_product_of_powers_l2468_246856


namespace NUMINAMATH_CALUDE_pentagon_percentage_is_31_25_percent_l2468_246838

/-- Represents a tiling pattern on a plane -/
structure TilingPattern where
  largeSqCount : ℕ  -- Number of large squares
  smallSqPerLarge : ℕ  -- Number of small squares per large square
  pentagonPerLarge : ℕ  -- Number of pentagons per large square

/-- Calculate the percentage of the plane enclosed by pentagons -/
def pentagonPercentage (pattern : TilingPattern) : ℚ :=
  pattern.pentagonPerLarge / pattern.smallSqPerLarge

/-- The specific tiling pattern described in the problem -/
def problemPattern : TilingPattern :=
  { largeSqCount := 1
  , smallSqPerLarge := 16
  , pentagonPerLarge := 5 }

theorem pentagon_percentage_is_31_25_percent :
  pentagonPercentage problemPattern = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_percentage_is_31_25_percent_l2468_246838


namespace NUMINAMATH_CALUDE_two_distinct_roots_l2468_246888

/-- The equation has exactly two distinct real roots if and only if a > 0 or a = -2 -/
theorem two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧
    x^2 - 6*x + (a-2)*|x-3| + 9 - 2*a = 0 ∧
    y^2 - 6*y + (a-2)*|y-3| + 9 - 2*a = 0 ∧
    (∀ z : ℝ, z^2 - 6*z + (a-2)*|z-3| + 9 - 2*a = 0 → z = x ∨ z = y)) ↔
  (a > 0 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l2468_246888


namespace NUMINAMATH_CALUDE_A_contribution_is_500_l2468_246861

def total : ℕ := 820
def ratio_A_to_B : Rat := 5 / 2
def ratio_B_to_C : Rat := 5 / 3

theorem A_contribution_is_500 : 
  ∃ (a b c : ℕ), 
    a + b + c = total ∧ 
    (a : ℚ) / b = ratio_A_to_B ∧ 
    (b : ℚ) / c = ratio_B_to_C ∧ 
    a = 500 := by
  sorry

end NUMINAMATH_CALUDE_A_contribution_is_500_l2468_246861


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2468_246867

-- Define the present ages of Lewis and Brown
def lewis_age : ℚ := 2
def brown_age : ℚ := 4

-- Define the conditions
theorem age_ratio_proof :
  -- Condition 1: Present ages are in ratio 1:2
  lewis_age / brown_age = 1 / 2 →
  -- Condition 2: Combined present age is 6
  lewis_age + brown_age = 6 →
  -- Prove: Ratio of ages three years from now is 5:7
  (lewis_age + 3) / (brown_age + 3) = 5 / 7 := by
sorry


end NUMINAMATH_CALUDE_age_ratio_proof_l2468_246867


namespace NUMINAMATH_CALUDE_pennies_count_l2468_246808

def pennies_in_jar (nickels dimes quarters : ℕ) (ice_cream_cost leftover : ℕ) : ℕ :=
  let nickel_value := 5
  let dime_value := 10
  let quarter_value := 25
  let total_without_pennies := nickels * nickel_value + dimes * dime_value + quarters * quarter_value
  let total_in_jar := ice_cream_cost + leftover
  total_in_jar - total_without_pennies

theorem pennies_count (nickels dimes quarters : ℕ) (ice_cream_cost leftover : ℕ) :
  nickels = 85 → dimes = 35 → quarters = 26 → ice_cream_cost = 1500 → leftover = 48 →
  pennies_in_jar nickels dimes quarters ice_cream_cost leftover = 123 := by
  sorry

#eval pennies_in_jar 85 35 26 1500 48

end NUMINAMATH_CALUDE_pennies_count_l2468_246808


namespace NUMINAMATH_CALUDE_sphere_impulse_theorem_l2468_246836

/-- Represents a uniform sphere -/
structure UniformSphere where
  mass : ℝ
  radius : ℝ

/-- Represents the initial conditions and applied impulse -/
structure ImpulseConditions where
  sphere : UniformSphere
  impulse : ℝ
  beta : ℝ

/-- Theorem stating the final speed and condition for rolling without slipping -/
theorem sphere_impulse_theorem (conditions : ImpulseConditions) 
  (h1 : conditions.beta ≥ -1) 
  (h2 : conditions.beta ≤ 1) : 
  ∃ (v : ℝ), 
    v = (5 * conditions.impulse * conditions.beta) / (7 * conditions.sphere.mass) ∧
    (conditions.beta = 7/5 → 
      v * conditions.sphere.mass = conditions.impulse ∧ 
      v = conditions.sphere.radius * ((5 * conditions.impulse * conditions.beta) / 
        (7 * conditions.sphere.mass * conditions.sphere.radius))) := by
  sorry

end NUMINAMATH_CALUDE_sphere_impulse_theorem_l2468_246836


namespace NUMINAMATH_CALUDE_toothpick_grid_60_32_l2468_246878

/-- Calculates the total number of toothpicks in a rectangular grid -/
def total_toothpicks (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem: A 60x32 toothpick grid uses 3932 toothpicks -/
theorem toothpick_grid_60_32 :
  total_toothpicks 60 32 = 3932 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_grid_60_32_l2468_246878


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2468_246898

/-- Given real numbers x, y, and z forming a geometric sequence with -1 and -3,
    prove that their product equals -3√3 -/
theorem geometric_sequence_product (x y z : ℝ) 
  (h1 : ∃ (r : ℝ), x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -3 = z * r) :
  x * y * z = -3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2468_246898


namespace NUMINAMATH_CALUDE_trisomicCrossRatio_l2468_246893

-- Define the basic types
inductive Genotype
| BB
| Bb
| bb
| Bbb
| bbb

inductive Gamete
| B
| b
| Bb
| bb

-- Define the meiosis process for trisomic cells
def trisomicMeiosis (g : Genotype) : List Gamete := sorry

-- Define the fertilization process
def fertilize (female : Gamete) (male : Gamete) : Option Genotype := sorry

-- Define the phenotype (disease resistance) based on genotype
def isResistant (g : Genotype) : Bool := sorry

-- Define the cross between two plants
def cross (female : Genotype) (male : Genotype) : List Genotype := sorry

-- Define the ratio calculation function
def ratioResistantToSusceptible (offspring : List Genotype) : Rat := sorry

-- Theorem statement
theorem trisomicCrossRatio :
  let femaleParent : Genotype := Genotype.bbb
  let maleParent : Genotype := Genotype.BB
  let f1 : List Genotype := cross femaleParent maleParent
  let f1Trisomic : Genotype := Genotype.Bbb
  let susceptibleNormal : Genotype := Genotype.bb
  let f2 : List Genotype := cross f1Trisomic susceptibleNormal
  ratioResistantToSusceptible f2 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_trisomicCrossRatio_l2468_246893


namespace NUMINAMATH_CALUDE_negative_one_to_zero_power_l2468_246840

theorem negative_one_to_zero_power : (-1 : ℝ) ^ (0 : ℕ) = 1 := by sorry

end NUMINAMATH_CALUDE_negative_one_to_zero_power_l2468_246840


namespace NUMINAMATH_CALUDE_perfect_squares_difference_99_l2468_246854

theorem perfect_squares_difference_99 :
  ∃! (l : List ℕ), 
    (∀ x ∈ l, ∃ a : ℕ, x = a^2 ∧ ∃ b : ℕ, x + 99 = b^2) ∧ 
    (∀ x : ℕ, (∃ a : ℕ, x = a^2 ∧ ∃ b : ℕ, x + 99 = b^2) → x ∈ l) ∧
    l.length = 3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_difference_99_l2468_246854


namespace NUMINAMATH_CALUDE_range_of_m_l2468_246884

theorem range_of_m (x m : ℝ) : 
  (∀ x, (4 * x - m < 0 → 1 ≤ 3 - x ∧ 3 - x ≤ 4) ∧ 
  ∃ x, (1 ≤ 3 - x ∧ 3 - x ≤ 4 ∧ ¬(4 * x - m < 0))) →
  m > 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2468_246884


namespace NUMINAMATH_CALUDE_problem_solution_l2468_246800

/-- Proposition p: x² - 4ax + 3a² < 0 -/
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

/-- Proposition q: |x - 3| < 1 -/
def q (x : ℝ) : Prop := |x - 3| < 1

theorem problem_solution :
  (∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3)) ∧
  (∀ a : ℝ, (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) ↔ (4/3 ≤ a ∧ a ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2468_246800


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2468_246891

theorem exponent_multiplication (a : ℝ) : a^2 * a^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2468_246891


namespace NUMINAMATH_CALUDE_mandy_med_school_acceptances_l2468_246801

theorem mandy_med_school_acceptances
  (total_researched : ℕ)
  (applied_fraction : ℚ)
  (accepted_fraction : ℚ)
  (h1 : total_researched = 96)
  (h2 : applied_fraction = 5 / 8)
  (h3 : accepted_fraction = 3 / 5)
  : ℕ :=
by
  sorry

end NUMINAMATH_CALUDE_mandy_med_school_acceptances_l2468_246801


namespace NUMINAMATH_CALUDE_paint_cans_for_house_l2468_246886

/-- Calculates the number of paint cans needed for a house painting job. -/
def paint_cans_needed (num_bedrooms : ℕ) (paint_per_room : ℕ) 
  (color_can_size : ℕ) (white_can_size : ℕ) : ℕ :=
  let num_other_rooms := 2 * num_bedrooms
  let total_color_paint := num_bedrooms * paint_per_room
  let total_white_paint := num_other_rooms * paint_per_room
  let color_cans := (total_color_paint + color_can_size - 1) / color_can_size
  let white_cans := (total_white_paint + white_can_size - 1) / white_can_size
  color_cans + white_cans

/-- Theorem stating that the number of paint cans needed for the given conditions is 10. -/
theorem paint_cans_for_house : paint_cans_needed 3 2 1 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_for_house_l2468_246886


namespace NUMINAMATH_CALUDE_complex_number_trigonometric_form_l2468_246809

/-- Prove that the complex number z = sin 36° + i cos 54° is equal to √2 sin 36° (cos 45° + i sin 45°) -/
theorem complex_number_trigonometric_form 
  (z : ℂ) 
  (h1 : z = Complex.ofReal (Real.sin (36 * π / 180)) + Complex.I * Complex.ofReal (Real.cos (54 * π / 180)))
  (h2 : Real.cos (54 * π / 180) = Real.sin (36 * π / 180)) :
  z = Complex.ofReal (Real.sqrt 2 * Real.sin (36 * π / 180)) * 
      (Complex.ofReal (Real.cos (45 * π / 180)) + Complex.I * Complex.ofReal (Real.sin (45 * π / 180))) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_trigonometric_form_l2468_246809


namespace NUMINAMATH_CALUDE_unique_rectangle_with_half_perimeter_quarter_area_l2468_246857

theorem unique_rectangle_with_half_perimeter_quarter_area 
  (a b : ℝ) (hab : a < b) : 
  ∃! (x y : ℝ), x < b ∧ y < b ∧ 
  2 * (x + y) = a + b ∧ 
  x * y = (a * b) / 4 := by
sorry

end NUMINAMATH_CALUDE_unique_rectangle_with_half_perimeter_quarter_area_l2468_246857


namespace NUMINAMATH_CALUDE_function_simplification_l2468_246879

theorem function_simplification (x : ℝ) : 
  Real.sqrt (4 * Real.sin x ^ 4 - 2 * Real.cos (2 * x) + 3) + 
  Real.sqrt (4 * Real.cos x ^ 4 + 2 * Real.cos (2 * x) + 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_simplification_l2468_246879


namespace NUMINAMATH_CALUDE_sector_area_l2468_246845

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (area : ℝ) : 
  perimeter = 8 → central_angle = 2 → area = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2468_246845


namespace NUMINAMATH_CALUDE_joey_age_l2468_246895

theorem joey_age : 
  let ages : List ℕ := [3, 5, 7, 9, 11, 13]
  let movie_pair : ℕ × ℕ := (3, 13)
  let baseball_pair : ℕ × ℕ := (7, 9)
  let stay_home : ℕ × ℕ := (5, 11)
  (∀ (a b : ℕ), a ∈ ages ∧ b ∈ ages ∧ a + b = 16 → (a, b) = movie_pair) ∧
  (∀ (a b : ℕ), a ∈ ages ∧ b ∈ ages ∧ a < 10 ∧ b < 10 ∧ (a, b) ≠ movie_pair → (a, b) = baseball_pair) ∧
  (∀ (a : ℕ), a ∈ ages ∧ a ∉ [movie_pair.1, movie_pair.2, baseball_pair.1, baseball_pair.2, 5] → a = 11) →
  stay_home.2 = 11 :=
by sorry

end NUMINAMATH_CALUDE_joey_age_l2468_246895


namespace NUMINAMATH_CALUDE_f_fixed_points_l2468_246864

def f (x : ℝ) : ℝ := x^3 - 3*x^2

theorem f_fixed_points : 
  ∃ (x : ℝ), (f (f x) = f x) ∧ (x = 0 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_f_fixed_points_l2468_246864


namespace NUMINAMATH_CALUDE_ellipse_properties_l2468_246866

noncomputable def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def right_focus : ℝ × ℝ := (1, 0)

theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ellipse_C (-1) (Real.sqrt 2 / 2) a b →
  (∀ x y : ℝ, ellipse_C x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  (∃ Q : ℝ × ℝ, Q.1 = 5/4 ∧ Q.2 = 0 ∧
    ∀ A B : ℝ × ℝ, 
      ellipse_C A.1 A.2 a b →
      ellipse_C B.1 B.2 a b →
      (∃ t : ℝ, A.1 = t * A.2 + 1 ∧ B.1 = t * B.2 + 1) →
      ((A.1 - Q.1) * (B.1 - Q.1) + (A.2 - Q.2) * (B.2 - Q.2) = -7/16)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2468_246866


namespace NUMINAMATH_CALUDE_max_absolute_sum_l2468_246806

theorem max_absolute_sum (x y z : ℝ) :
  (|x + 2*y - 3*z| ≤ 6) →
  (|x - 2*y + 3*z| ≤ 6) →
  (|x - 2*y - 3*z| ≤ 6) →
  (|x + 2*y + 3*z| ≤ 6) →
  |x| + |y| + |z| ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_absolute_sum_l2468_246806


namespace NUMINAMATH_CALUDE_trajectory_of_complex_point_l2468_246848

theorem trajectory_of_complex_point (z : ℂ) (h : Complex.abs z ≤ 1) :
  ∃ (P : ℝ × ℝ), P.1^2 + P.2^2 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_complex_point_l2468_246848


namespace NUMINAMATH_CALUDE_price_changes_l2468_246862

theorem price_changes (original_price : ℝ) : 
  let price_after_first_increase := original_price * 1.2
  let price_after_second_increase := price_after_first_increase + 5
  let price_after_first_decrease := price_after_second_increase * 0.8
  let final_price := price_after_first_decrease - 5
  final_price = 120 → original_price = 126.04 := by
sorry

#eval (121 / 0.96 : Float)

end NUMINAMATH_CALUDE_price_changes_l2468_246862


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l2468_246873

theorem no_function_satisfies_conditions : ¬∃ (f : ℝ → ℝ), 
  (∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), -M ≤ f x ∧ f x ≤ M) ∧ 
  (f 1 = 1) ∧
  (∀ (x : ℝ), x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l2468_246873


namespace NUMINAMATH_CALUDE_john_umbrellas_in_car_l2468_246805

/-- The number of umbrellas John has in his house -/
def umbrellas_in_house : ℕ := 2

/-- The cost of each umbrella in dollars -/
def cost_per_umbrella : ℕ := 8

/-- The total amount John paid for all umbrellas in dollars -/
def total_paid : ℕ := 24

/-- The number of umbrellas John has in his car -/
def umbrellas_in_car : ℕ := total_paid / cost_per_umbrella - umbrellas_in_house

theorem john_umbrellas_in_car : umbrellas_in_car = 1 := by
  sorry

end NUMINAMATH_CALUDE_john_umbrellas_in_car_l2468_246805


namespace NUMINAMATH_CALUDE_school_students_count_l2468_246882

/-- Proves that given the specified conditions, the total number of students in the school is 387 -/
theorem school_students_count : ∃ (boys girls : ℕ), 
  boys ≥ 150 ∧ 
  boys % 6 = 0 ∧ 
  girls = boys + boys / 20 * 3 ∧ 
  boys + girls ≤ 400 ∧
  boys + girls = 387 := by
  sorry

end NUMINAMATH_CALUDE_school_students_count_l2468_246882


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l2468_246876

theorem stratified_sampling_medium_stores 
  (total_stores : ℕ) 
  (medium_stores : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_stores = 300)
  (h2 : medium_stores = 75)
  (h3 : sample_size = 20) :
  (sample_size * medium_stores) / total_stores = 5 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l2468_246876


namespace NUMINAMATH_CALUDE_limit_of_rational_function_at_four_l2468_246843

theorem limit_of_rational_function_at_four :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 4| ∧ |x - 4| < δ →
    |((x^2 - 2*x - 8) / (x - 4)) - 6| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_of_rational_function_at_four_l2468_246843


namespace NUMINAMATH_CALUDE_lcm_gcf_relation_l2468_246896

theorem lcm_gcf_relation (n : ℕ) :
  Nat.lcm n 24 = 48 ∧ Nat.gcd n 24 = 8 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_relation_l2468_246896


namespace NUMINAMATH_CALUDE_total_pencils_count_l2468_246870

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of people who have the color box -/
def total_people : ℕ := 8

/-- The number of pencils in each color box -/
def pencils_per_box : ℕ := rainbow_colors

/-- The total number of pencils -/
def total_pencils : ℕ := pencils_per_box * total_people

theorem total_pencils_count : total_pencils = 56 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_count_l2468_246870


namespace NUMINAMATH_CALUDE_midpoint_complex_plane_l2468_246885

theorem midpoint_complex_plane (A B C : ℂ) : 
  A = 6 + 5*I ∧ B = -2 + 3*I ∧ C = (A + B) / 2 → C = 2 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_midpoint_complex_plane_l2468_246885


namespace NUMINAMATH_CALUDE_shopping_cost_theorem_l2468_246859

/-- Calculates the total cost of Fabian's shopping trip --/
def calculate_shopping_cost (
  apple_price : ℝ)
  (walnut_price : ℝ)
  (orange_price : ℝ)
  (pasta_price : ℝ)
  (sugar_discount : ℝ)
  (orange_discount : ℝ)
  (sales_tax : ℝ) : ℝ :=
  let apple_cost := 5 * apple_price
  let sugar_cost := 3 * (apple_price - sugar_discount)
  let walnut_cost := 0.5 * walnut_price
  let orange_cost := 2 * orange_price * (1 - orange_discount)
  let pasta_cost := 3 * pasta_price
  let total_before_tax := apple_cost + sugar_cost + walnut_cost + orange_cost + pasta_cost
  total_before_tax * (1 + sales_tax)

/-- The theorem stating the total cost of Fabian's shopping --/
theorem shopping_cost_theorem :
  calculate_shopping_cost 2 6 3 1.5 1 0.1 0.05 = 27.20 := by sorry

end NUMINAMATH_CALUDE_shopping_cost_theorem_l2468_246859


namespace NUMINAMATH_CALUDE_unique_base_representation_l2468_246813

theorem unique_base_representation : 
  ∃! (x y z b : ℕ), 
    x * b^2 + y * b + z = 1987 ∧
    x + y + z = 25 ∧
    x ≥ 1 ∧
    y < b ∧
    z < b ∧
    b > 1 ∧
    x = 5 ∧
    y = 9 ∧
    z = 11 ∧
    b = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_representation_l2468_246813


namespace NUMINAMATH_CALUDE_product_of_greater_than_one_is_greater_than_one_l2468_246803

theorem product_of_greater_than_one_is_greater_than_one (a b : ℝ) : a > 1 → b > 1 → a * b > 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_greater_than_one_is_greater_than_one_l2468_246803


namespace NUMINAMATH_CALUDE_power_of_three_plus_five_mod_five_l2468_246839

theorem power_of_three_plus_five_mod_five : 
  (3^100 + 5) % 5 = 1 := by sorry

end NUMINAMATH_CALUDE_power_of_three_plus_five_mod_five_l2468_246839


namespace NUMINAMATH_CALUDE_bowling_balls_count_l2468_246883

/-- The number of red bowling balls -/
def red_balls : ℕ := 30

/-- The difference between green and red bowling balls -/
def green_red_difference : ℕ := 6

/-- The total number of bowling balls -/
def total_balls : ℕ := red_balls + (red_balls + green_red_difference)

theorem bowling_balls_count : total_balls = 66 := by
  sorry

end NUMINAMATH_CALUDE_bowling_balls_count_l2468_246883


namespace NUMINAMATH_CALUDE_problem_solution_l2468_246833

theorem problem_solution (x y A : ℝ) 
  (h1 : 2^x = A) 
  (h2 : 7^(2*y) = A) 
  (h3 : 1/x + 1/y = 2) : 
  A = 7 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2468_246833


namespace NUMINAMATH_CALUDE_water_bottle_consumption_l2468_246899

theorem water_bottle_consumption (total : ℕ) (first_day_fraction : ℚ) (remaining : ℕ) : 
  total = 24 → 
  first_day_fraction = 1/3 → 
  remaining = 8 → 
  (total - (first_day_fraction * total).num - remaining : ℚ) / (total - (first_day_fraction * total).num : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_consumption_l2468_246899


namespace NUMINAMATH_CALUDE_problem_solution_l2468_246863

-- Define x as the solution to the equation x = 1 + √3 / x
noncomputable def x : ℝ := Real.sqrt 3 + 1

-- State the theorem
theorem problem_solution :
  1 / ((x + 1) * (x - 3)) = (-Real.sqrt 3 - 4) / 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2468_246863


namespace NUMINAMATH_CALUDE_gordons_heavier_bag_weight_l2468_246892

theorem gordons_heavier_bag_weight (trace_bag_count : ℕ) (trace_bag_weight : ℝ)
  (gordon_bag_count : ℕ) (gordon_lighter_bag_weight : ℝ) :
  trace_bag_count = 5 →
  trace_bag_weight = 2 →
  gordon_bag_count = 2 →
  gordon_lighter_bag_weight = 3 →
  trace_bag_count * trace_bag_weight = gordon_lighter_bag_weight + gordon_heavier_bag_weight →
  gordon_heavier_bag_weight = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_gordons_heavier_bag_weight_l2468_246892


namespace NUMINAMATH_CALUDE_sum_of_seven_terms_l2468_246828

/-- An arithmetic sequence with a_4 = 7 -/
def arithmetic_seq (n : ℕ) : ℝ :=
  sorry

/-- Sum of first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ :=
  sorry

theorem sum_of_seven_terms :
  arithmetic_seq 4 = 7 → S 7 = 49 :=
sorry

end NUMINAMATH_CALUDE_sum_of_seven_terms_l2468_246828


namespace NUMINAMATH_CALUDE_line_y_coordinate_proof_l2468_246822

/-- Given a line passing through points (-1, y) and (5, 0.8) with slope 0.8,
    prove that the y-coordinate of the first point is 4. -/
theorem line_y_coordinate_proof (y : ℝ) : 
  (0.8 - y) / (5 - (-1)) = 0.8 → y = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_y_coordinate_proof_l2468_246822


namespace NUMINAMATH_CALUDE_problem_solution_l2468_246815

theorem problem_solution : 
  let M : ℚ := 2013 / 3
  let N : ℚ := M / 3
  let X : ℚ := M + N
  X = 895 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2468_246815


namespace NUMINAMATH_CALUDE_digit_sum_difference_l2468_246894

-- Define a function to calculate the sum of digits of a number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool := sorry

-- Define the sum of digits for all even numbers from 1 to 1000
def sumEvenDigits : ℕ := 
  (List.range 1000).filter isEven |>.map sumOfDigits |>.sum

-- Define the sum of digits for all odd numbers from 1 to 1000
def sumOddDigits : ℕ := 
  (List.range 1000).filter (λ n => ¬(isEven n)) |>.map sumOfDigits |>.sum

-- Theorem statement
theorem digit_sum_difference :
  sumOddDigits - sumEvenDigits = 499 := by sorry

end NUMINAMATH_CALUDE_digit_sum_difference_l2468_246894


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2468_246887

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence. -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

/-- Given an arithmetic sequence with S₈ = 30 and S₄ = 7, prove that a₄ = 13/4. -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h₁ : S seq 8 = 30)
    (h₂ : S seq 4 = 7) : 
  seq.a 4 = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2468_246887


namespace NUMINAMATH_CALUDE_smallest_z_l2468_246810

theorem smallest_z (x y z : ℤ) : 
  x < y → y < z → 
  2 * y = x + z →  -- arithmetic progression
  z * z = x * y →  -- geometric progression
  z ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_l2468_246810


namespace NUMINAMATH_CALUDE_choir_members_count_l2468_246834

theorem choir_members_count : ∃ n₁ n₂ : ℕ, 
  150 < n₁ ∧ n₁ < 250 ∧
  150 < n₂ ∧ n₂ < 250 ∧
  n₁ % 3 = 1 ∧
  n₁ % 6 = 2 ∧
  n₁ % 8 = 3 ∧
  n₂ % 3 = 1 ∧
  n₂ % 6 = 2 ∧
  n₂ % 8 = 3 ∧
  n₁ = 195 ∧
  n₂ = 219 ∧
  ∀ n : ℕ, (150 < n ∧ n < 250 ∧ n % 3 = 1 ∧ n % 6 = 2 ∧ n % 8 = 3) → (n = 195 ∨ n = 219) :=
by sorry

end NUMINAMATH_CALUDE_choir_members_count_l2468_246834


namespace NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l2468_246820

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := a^2 - abs b

-- Theorem statement
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_neg_two_neg_one_l2468_246820


namespace NUMINAMATH_CALUDE_sin_double_angle_sum_l2468_246850

open Real

theorem sin_double_angle_sum (θ : ℝ) (h : ∑' n, sin θ ^ (2 * n) = 3) : 
  sin (2 * θ) = 2 * sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_sum_l2468_246850


namespace NUMINAMATH_CALUDE_employee_payment_l2468_246811

/-- Given two employees X and Y with a total payment of 550 units,
    where X is paid 120% of Y's payment, prove that Y is paid 250 units. -/
theorem employee_payment (x y : ℝ) 
  (total : x + y = 550)
  (ratio : x = 1.2 * y) : 
  y = 250 := by
  sorry

end NUMINAMATH_CALUDE_employee_payment_l2468_246811


namespace NUMINAMATH_CALUDE_triangle_side_a_equals_one_l2468_246826

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (-Real.cos x, Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ :=
  let m_vec := m x
  let n_vec := n x
  (m_vec.1 * (0.5 * m_vec.1 - n_vec.1)) + (m_vec.2 * (0.5 * m_vec.2 - n_vec.2))

theorem triangle_side_a_equals_one (A B C : ℝ) (a b c : ℝ) :
  f (B / 2) = 1 → b = 1 → c = Real.sqrt 3 →
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_a_equals_one_l2468_246826


namespace NUMINAMATH_CALUDE_problem_statement_l2468_246868

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -3) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = -5932 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2468_246868


namespace NUMINAMATH_CALUDE_right_triangle_30_60_90_l2468_246865

theorem right_triangle_30_60_90 (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_angle : a / c = 1 / 2) (h_hypotenuse : c = 10) : 
  a = 5 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_30_60_90_l2468_246865


namespace NUMINAMATH_CALUDE_max_pens_purchased_l2468_246817

/-- Represents the prices and quantities of pens and mechanical pencils -/
structure PriceQuantity where
  pen_price : ℕ
  pencil_price : ℕ
  pen_quantity : ℕ
  pencil_quantity : ℕ

/-- Represents the pricing conditions given in the problem -/
def pricing_conditions (p : PriceQuantity) : Prop :=
  2 * p.pen_price + 5 * p.pencil_price = 75 ∧
  3 * p.pen_price + 2 * p.pencil_price = 85

/-- Represents the promotion and quantity conditions -/
def promotion_conditions (p : PriceQuantity) : Prop :=
  p.pencil_quantity = 2 * p.pen_quantity + 8 ∧
  p.pen_price * p.pen_quantity + p.pencil_price * (p.pencil_quantity - p.pen_quantity) < 670

/-- Theorem stating the maximum number of pens that can be purchased -/
theorem max_pens_purchased (p : PriceQuantity) 
  (h1 : pricing_conditions p) 
  (h2 : promotion_conditions p) : 
  p.pen_quantity ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_max_pens_purchased_l2468_246817


namespace NUMINAMATH_CALUDE_minimum_buses_needed_l2468_246881

def bus_capacity : ℕ := 48
def total_passengers : ℕ := 1230

def buses_needed (capacity : ℕ) (passengers : ℕ) : ℕ :=
  (passengers + capacity - 1) / capacity

theorem minimum_buses_needed : 
  buses_needed bus_capacity total_passengers = 26 := by
  sorry

end NUMINAMATH_CALUDE_minimum_buses_needed_l2468_246881


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l2468_246807

theorem tangent_circles_radius (r : ℝ) (R : ℝ) : 
  r > 0 → 
  (∃ (A B C : ℝ × ℝ) (O : ℝ × ℝ),
    -- Three circles with centers A, B, C and radius r are externally tangent to each other
    dist A B = 2 * r ∧ 
    dist B C = 2 * r ∧ 
    dist C A = 2 * r ∧
    -- These three circles are internally tangent to a larger circle with center O and radius R
    dist O A = R - r ∧
    dist O B = R - r ∧
    dist O C = R - r) →
  -- Then the radius of the large circle is 2(√3 + 1) when r = 2
  r = 2 → R = 2 * (Real.sqrt 3 + 1) := by
sorry


end NUMINAMATH_CALUDE_tangent_circles_radius_l2468_246807


namespace NUMINAMATH_CALUDE_range_of_t_l2468_246851

theorem range_of_t (a b t : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) 
  (ht : ∀ a b, a > 0 → b > 0 → 2 * a + b = 1 → 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1/2) : 
  t ≥ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_t_l2468_246851


namespace NUMINAMATH_CALUDE_european_passenger_fraction_l2468_246860

theorem european_passenger_fraction (total : ℕ) 
  (north_america : ℚ) (africa : ℚ) (asia : ℚ) (other : ℕ) :
  total = 108 →
  north_america = 1 / 12 →
  africa = 1 / 9 →
  asia = 1 / 6 →
  other = 42 →
  (total : ℚ) - (north_america * total + africa * total + asia * total + other) = 1 / 4 * total :=
by sorry

end NUMINAMATH_CALUDE_european_passenger_fraction_l2468_246860


namespace NUMINAMATH_CALUDE_finite_solutions_of_system_l2468_246812

theorem finite_solutions_of_system (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∃ (S : Finset (ℤ × ℤ × ℤ × ℤ)), ∀ (x y z w : ℤ),
    x * y + z * w = a ∧ x * z + y * w = b → (x, y, z, w) ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_of_system_l2468_246812


namespace NUMINAMATH_CALUDE_hexagon_largest_angle_l2468_246814

theorem hexagon_largest_angle (a b c d e f : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  b / a = 3 / 2 →
  c / a = 3 / 2 →
  d / a = 2 →
  e / a = 2 →
  f / a = 3 →
  a + b + c + d + e + f = 720 →
  f = 2160 / 11 := by
sorry

end NUMINAMATH_CALUDE_hexagon_largest_angle_l2468_246814


namespace NUMINAMATH_CALUDE_not_perfect_square_l2468_246871

theorem not_perfect_square (n : ℤ) (h : n > 4) : ¬∃ (k : ℕ), n^2 - 3*n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2468_246871


namespace NUMINAMATH_CALUDE_boat_stream_speed_l2468_246841

/-- Proves that given a boat with a speed of 36 kmph in still water,
    if it can cover 80 km downstream or 40 km upstream in the same time,
    then the speed of the stream is 12 kmph. -/
theorem boat_stream_speed 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (h1 : boat_speed = 36)
  (h2 : downstream_distance = 80)
  (h3 : upstream_distance = 40)
  (h4 : downstream_distance / (boat_speed + stream_speed) = upstream_distance / (boat_speed - stream_speed)) :
  stream_speed = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_boat_stream_speed_l2468_246841


namespace NUMINAMATH_CALUDE_solution_to_equation_l2468_246877

theorem solution_to_equation (x : ℝ) (hx : x ≠ 0) :
  (3 * x)^5 = (9 * x)^4 → x = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2468_246877


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l2468_246849

theorem sum_of_squares_and_products (a b c : ℝ) 
  (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_c : c ≥ 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 48)
  (h_sum_products : a*b + b*c + c*a = 18) :
  a + b + c = 2 * Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l2468_246849


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l2468_246875

/-- If a line x - y - 1 = 0 is tangent to a parabola y = ax², then a = 1/4 -/
theorem tangent_line_to_parabola (a : ℝ) : 
  (∃ x y : ℝ, x - y - 1 = 0 ∧ y = a * x^2 ∧ 
   ∀ x' y' : ℝ, x' - y' - 1 = 0 → y' ≥ a * x'^2) → 
  a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l2468_246875


namespace NUMINAMATH_CALUDE_diluted_vinegar_concentration_diluted_vinegar_concentration_proof_l2468_246853

/-- Calculates the concentration of a diluted vinegar solution -/
theorem diluted_vinegar_concentration 
  (original_volume : ℝ) 
  (original_concentration : ℝ) 
  (water_added : ℝ) : ℝ :=
  let vinegar_amount := original_volume * (original_concentration / 100)
  let total_volume := original_volume + water_added
  let diluted_concentration := (vinegar_amount / total_volume) * 100
  diluted_concentration

/-- Proves that the diluted vinegar concentration is approximately 7% -/
theorem diluted_vinegar_concentration_proof 
  (original_volume : ℝ) 
  (original_concentration : ℝ) 
  (water_added : ℝ) 
  (h1 : original_volume = 12) 
  (h2 : original_concentration = 36.166666666666664) 
  (h3 : water_added = 50) :
  ∃ ε > 0, |diluted_vinegar_concentration original_volume original_concentration water_added - 7| < ε :=
sorry

end NUMINAMATH_CALUDE_diluted_vinegar_concentration_diluted_vinegar_concentration_proof_l2468_246853


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_opposite_of_negative_one_over_2023_l2468_246818

theorem opposite_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  (-(1 : ℚ) / n) + (1 : ℚ) / n = 0 :=
by sorry

theorem opposite_of_negative_one_over_2023 :
  (-(1 : ℚ) / 2023) + (1 : ℚ) / 2023 = 0 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_opposite_of_negative_one_over_2023_l2468_246818


namespace NUMINAMATH_CALUDE_taxi_speed_l2468_246819

/-- Given a taxi and a bus with specific conditions, proves that the taxi's speed is 60 mph --/
theorem taxi_speed (taxi_speed bus_speed : ℝ) : 
  (taxi_speed > 0) →  -- Ensure positive speed
  (bus_speed > 0) →   -- Ensure positive speed
  (bus_speed = taxi_speed - 30) →  -- Bus is 30 mph slower
  (3 * taxi_speed = 6 * bus_speed) →  -- Taxi covers in 3 hours what bus covers in 6
  (taxi_speed = 60) :=
by
  sorry

#check taxi_speed

end NUMINAMATH_CALUDE_taxi_speed_l2468_246819


namespace NUMINAMATH_CALUDE_no_solution_to_inequalities_l2468_246802

theorem no_solution_to_inequalities :
  ¬ ∃ x : ℝ, (4 * x + 2 < (x + 3)^2) ∧ ((x + 3)^2 < 8 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequalities_l2468_246802


namespace NUMINAMATH_CALUDE_set_difference_example_l2468_246816

def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem set_difference_example :
  let A : Set ℕ := {2, 3}
  let B : Set ℕ := {1, 3, 4}
  set_difference A B = {2} := by
sorry

end NUMINAMATH_CALUDE_set_difference_example_l2468_246816


namespace NUMINAMATH_CALUDE_log_cutting_theorem_l2468_246823

/-- The number of pieces of wood after cutting a log -/
def num_pieces (initial_logs : ℕ) (num_cuts : ℕ) : ℕ :=
  initial_logs + num_cuts

/-- Theorem: Cutting a single log 10 times results in 11 pieces -/
theorem log_cutting_theorem :
  num_pieces 1 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_cutting_theorem_l2468_246823


namespace NUMINAMATH_CALUDE_no_two_perfect_scores_l2468_246804

/-- Represents the exam scores of a student -/
structure ExamScores where
  russian : ℤ
  physics : ℤ
  mathematics : ℤ

/-- Defines the initial relationship between exam scores -/
def validScores (scores : ExamScores) : Prop :=
  scores.russian = scores.physics - 3 ∧ scores.physics = scores.mathematics - 7

/-- Represents a score modification operation -/
inductive ScoreModification
  | addOne : ScoreModification
  | decreaseOneIncreaseTwo : ScoreModification

/-- Applies a score modification to the exam scores -/
def applyModification (scores : ExamScores) (mod : ScoreModification) : ExamScores :=
  match mod with
  | ScoreModification.addOne => 
      { russian := scores.russian + 1,
        physics := scores.physics + 1,
        mathematics := scores.mathematics + 1 }
  | ScoreModification.decreaseOneIncreaseTwo =>
      { russian := scores.russian - 3,
        physics := scores.physics + 1,
        mathematics := scores.mathematics + 1 }

/-- Checks if any score exceeds 100 -/
def exceedsLimit (scores : ExamScores) : Prop :=
  scores.russian > 100 ∨ scores.physics > 100 ∨ scores.mathematics > 100

/-- Checks if more than one score is equal to 100 -/
def moreThanOneHundred (scores : ExamScores) : Prop :=
  (scores.russian = 100 ∧ scores.physics = 100) ∨
  (scores.russian = 100 ∧ scores.mathematics = 100) ∨
  (scores.physics = 100 ∧ scores.mathematics = 100)

/-- The main theorem to be proved -/
theorem no_two_perfect_scores (initialScores : ExamScores) 
  (h : validScores initialScores) :
  ∀ (mods : List ScoreModification),
    let finalScores := mods.foldl applyModification initialScores
    ¬(moreThanOneHundred finalScores ∧ ¬exceedsLimit finalScores) := by
  sorry


end NUMINAMATH_CALUDE_no_two_perfect_scores_l2468_246804


namespace NUMINAMATH_CALUDE_probability_two_females_l2468_246830

theorem probability_two_females (total : Nat) (females : Nat) (chosen : Nat) :
  total = 8 →
  females = 5 →
  chosen = 2 →
  (Nat.choose females chosen : ℚ) / (Nat.choose total chosen : ℚ) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_l2468_246830


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l2468_246846

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_different (a b c d e f : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

theorem digit_sum_puzzle (a b c d e f : ℕ) :
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
  is_valid_digit d ∧ is_valid_digit e ∧ is_valid_digit f ∧
  are_different a b c d e f ∧
  100 * a + 10 * b + c +
  100 * d + 10 * e + a +
  100 * f + 10 * a + b = 1111 →
  a + b + c + d + e + f = 24 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l2468_246846


namespace NUMINAMATH_CALUDE_expression_values_l2468_246869

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let e := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  e = 5 ∨ e = 1 ∨ e = -1 ∨ e = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l2468_246869


namespace NUMINAMATH_CALUDE_unique_intersection_l2468_246821

/-- The value of m for which the line x = m intersects the parabola x = -3y^2 - 4y + 7 at exactly one point -/
def m : ℚ := 25/3

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

theorem unique_intersection :
  ∃! y, parabola y = m :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l2468_246821


namespace NUMINAMATH_CALUDE_field_trip_vans_l2468_246831

theorem field_trip_vans (total_people : ℕ) (num_buses : ℕ) (people_per_bus : ℕ) (people_per_van : ℕ) :
  total_people = 180 →
  num_buses = 8 →
  people_per_bus = 18 →
  people_per_van = 6 →
  ∃ (num_vans : ℕ), num_vans = 6 ∧ total_people = num_buses * people_per_bus + num_vans * people_per_van :=
by sorry

end NUMINAMATH_CALUDE_field_trip_vans_l2468_246831


namespace NUMINAMATH_CALUDE_min_value_implies_a_values_l2468_246824

theorem min_value_implies_a_values (a : ℝ) : 
  (∃ (m : ℝ), ∀ (x : ℝ), |x + 1| + |x + a| ≥ m ∧ (∃ (y : ℝ), |y + 1| + |y + a| = m) ∧ m = 1) →
  a = 0 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_implies_a_values_l2468_246824


namespace NUMINAMATH_CALUDE_solution_sum_l2468_246837

theorem solution_sum (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁ * y₁ - x₁ = 180 ∧ y₁ + x₁ * y₁ = 208) ∧
  (x₂ * y₂ - x₂ = 180 ∧ y₂ + x₂ * y₂ = 208) ∧
  (x₁ ≠ x₂) →
  x₁ + 10 * y₁ + x₂ + 10 * y₂ = 317 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l2468_246837


namespace NUMINAMATH_CALUDE_luka_water_needed_l2468_246897

/-- Represents the recipe ratios and amount of lemon juice used --/
structure Recipe where
  water_sugar_ratio : ℚ
  sugar_lemon_ratio : ℚ
  lemon_juice : ℚ

/-- Calculates the amount of water needed based on the recipe --/
def water_needed (r : Recipe) : ℚ :=
  r.water_sugar_ratio * r.sugar_lemon_ratio * r.lemon_juice

/-- Theorem stating that Luka needs 24 cups of water --/
theorem luka_water_needed :
  let r : Recipe := {
    water_sugar_ratio := 4,
    sugar_lemon_ratio := 2,
    lemon_juice := 3
  }
  water_needed r = 24 := by sorry

end NUMINAMATH_CALUDE_luka_water_needed_l2468_246897


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2468_246872

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y : ℝ, (y + 3) * (y - 4) = 22 ∧ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2468_246872


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2468_246844

theorem completing_square_equivalence :
  ∀ x : ℝ, 4 * x^2 - 2 * x - 1 = 0 ↔ (x - 1/4)^2 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2468_246844


namespace NUMINAMATH_CALUDE_equation_solution_exists_l2468_246890

theorem equation_solution_exists (m : ℕ+) :
  ∃ n : ℕ+, (n : ℚ) / m = ⌊(n^2 : ℚ)^(1/3)⌋ + ⌊(n : ℚ)^(1/2)⌋ + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l2468_246890


namespace NUMINAMATH_CALUDE_range_of_a_l2468_246874

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

theorem range_of_a (h : ¬(∃ a : ℝ, p a ∨ q a)) :
  ∃ a : ℝ, 1 < a ∧ a < 2 ∧ ∀ b : ℝ, (1 < b ∧ b < 2) → (¬(p b) ∧ ¬(q b)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2468_246874


namespace NUMINAMATH_CALUDE_irreducible_fractions_count_l2468_246855

/-- A rational number between 0 and 1 in irreducible fraction form -/
structure IrreducibleFraction :=
  (numerator : ℕ)
  (denominator : ℕ)
  (is_between_0_and_1 : numerator < denominator)
  (is_irreducible : Nat.gcd numerator denominator = 1)
  (product_is_20 : numerator * denominator = 20)

/-- The count of irreducible fractions between 0 and 1 with numerator-denominator product of 20 -/
def count_irreducible_fractions : ℕ := sorry

/-- The main theorem stating there are 128 such fractions -/
theorem irreducible_fractions_count :
  count_irreducible_fractions = 128 := by sorry

end NUMINAMATH_CALUDE_irreducible_fractions_count_l2468_246855


namespace NUMINAMATH_CALUDE_prob_no_standing_pairs_10_l2468_246835

/-- Represents the number of valid arrangements for n people where no two adjacent people form a standing pair -/
def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| n+2 => 3 * b (n+1) - b n

/-- The probability of no standing pairs for n people -/
def prob_no_standing_pairs (n : ℕ) : ℚ :=
  (b n : ℚ) / (2^n : ℚ)

theorem prob_no_standing_pairs_10 :
  prob_no_standing_pairs 10 = 31 / 128 := by sorry

end NUMINAMATH_CALUDE_prob_no_standing_pairs_10_l2468_246835


namespace NUMINAMATH_CALUDE_female_officers_count_l2468_246847

theorem female_officers_count (total_on_duty : ℕ) (male_percentage : ℚ) (female_on_duty_percentage : ℚ) :
  total_on_duty = 500 →
  male_percentage = 60 / 100 →
  female_on_duty_percentage = 10 / 100 →
  (female_on_duty_percentage * (total_female_officers : ℕ) : ℚ) = ((1 - male_percentage) * total_on_duty : ℚ) →
  total_female_officers = 2000 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l2468_246847


namespace NUMINAMATH_CALUDE_same_terminal_side_l2468_246842

theorem same_terminal_side (k : ℤ) : 
  let angles : List ℝ := [-5*π/3, 2*π/3, 4*π/3, 5*π/3]
  let target : ℝ := -π/3
  let same_side (α : ℝ) : Prop := ∃ n : ℤ, α = 2*π*n + target
  ∀ α ∈ angles, same_side α ↔ α = 5*π/3 :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_l2468_246842


namespace NUMINAMATH_CALUDE_jeongyeon_height_is_142_57_l2468_246889

/-- Jeongyeon's height in centimeters -/
def jeongyeon_height : ℝ := 1.06 * 134.5

/-- Theorem stating that Jeongyeon's height is 142.57 cm -/
theorem jeongyeon_height_is_142_57 : 
  jeongyeon_height = 142.57 := by sorry

end NUMINAMATH_CALUDE_jeongyeon_height_is_142_57_l2468_246889


namespace NUMINAMATH_CALUDE_regular_polygon_30_degree_central_angle_has_12_sides_l2468_246825

/-- Theorem: A regular polygon with a central angle of 30° has 12 sides. -/
theorem regular_polygon_30_degree_central_angle_has_12_sides :
  ∀ (n : ℕ) (central_angle : ℝ),
    central_angle = 30 →
    (360 : ℝ) / central_angle = n →
    n = 12 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_30_degree_central_angle_has_12_sides_l2468_246825


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l2468_246858

-- Define the curve
def f (x a : ℝ) : ℝ := x^3 + a*x + 1

-- Define the tangent line
def tangent_line (k b x : ℝ) : ℝ := k*x + b

-- Theorem statement
theorem tangent_line_b_value :
  ∀ (a k b : ℝ),
  (f 2 a = 3) →  -- The curve passes through (2, 3)
  (tangent_line k b 2 = 3) →  -- The tangent line passes through (2, 3)
  (∀ x, tangent_line k b x = k*x + b) →  -- Definition of tangent line
  (∀ x, f x a = x^3 + a*x + 1) →  -- Definition of the curve
  (k = (3*2^2 + a)) →  -- Slope of tangent is derivative at x = 2
  (b = -15) :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l2468_246858


namespace NUMINAMATH_CALUDE_max_k_for_tangent_line_l2468_246827

/-- The maximum value of k for which the line y = kx - 2 has at least one point 
    where a line tangent to the circle x^2 + y^2 = 1 can be drawn -/
theorem max_k_for_tangent_line : 
  ∃ (k : ℝ), ∀ (k' : ℝ), 
    (∃ (x y : ℝ), y = k' * x - 2 ∧ 
      ∃ (m : ℝ), (y - m * x)^2 = (1 + m^2) * (1 - x^2)) → 
    k' ≤ k ∧ 
    k = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_max_k_for_tangent_line_l2468_246827


namespace NUMINAMATH_CALUDE_isosceles_triangle_third_side_l2468_246880

/-- An isosceles triangle with side lengths a, b, and c, where a = b -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : a = b
  isPositive : 0 < a ∧ 0 < b ∧ 0 < c
  triangleInequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: In an isosceles triangle with two sides of lengths 13 and 6, the third side is 13 -/
theorem isosceles_triangle_third_side 
  (t : IsoscelesTriangle) 
  (h1 : t.a = 13 ∨ t.b = 13 ∨ t.c = 13) 
  (h2 : t.a = 6 ∨ t.b = 6 ∨ t.c = 6) : 
  t.c = 13 := by
  sorry

#check isosceles_triangle_third_side

end NUMINAMATH_CALUDE_isosceles_triangle_third_side_l2468_246880


namespace NUMINAMATH_CALUDE_f_composition_negative_eight_l2468_246832

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then -x^(1/3)
  else x + 2/x - 7

-- State the theorem
theorem f_composition_negative_eight : f (f (-8)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_eight_l2468_246832
