import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_five_integers_l2718_271851

theorem sum_of_five_integers (a b c d e : ℕ) :
  a ∈ Finset.range 20 ∧ 
  b ∈ Finset.range 20 ∧ 
  c ∈ Finset.range 20 ∧ 
  d ∈ Finset.range 20 ∧ 
  e ∈ Finset.range 20 ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e →
  15 ≤ a + b + c + d + e ∧ a + b + c + d + e ≤ 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_integers_l2718_271851


namespace NUMINAMATH_CALUDE_area_enclosed_by_cosine_curve_l2718_271855

theorem area_enclosed_by_cosine_curve : 
  let f (x : ℝ) := Real.cos x
  let area := ∫ x in (0)..(π/2), f x - ∫ x in (π/2)..(3*π/2), f x
  area = 3 := by
sorry

end NUMINAMATH_CALUDE_area_enclosed_by_cosine_curve_l2718_271855


namespace NUMINAMATH_CALUDE_leak_empty_time_correct_l2718_271802

/-- Represents a tank with a leak and an inlet pipe -/
structure Tank where
  capacity : ℝ
  inletRate : ℝ
  emptyTimeWithInlet : ℝ

/-- Calculates the time it takes for the leak alone to empty the tank -/
def leakEmptyTime (t : Tank) : ℝ :=
  -- Definition to be proved
  9

/-- Theorem stating the correct leak empty time for the given tank -/
theorem leak_empty_time_correct (t : Tank) 
  (h1 : t.capacity = 12960)
  (h2 : t.inletRate = 6 * 60)  -- 6 litres per minute converted to per hour
  (h3 : t.emptyTimeWithInlet = 12) : 
  leakEmptyTime t = 9 := by
  sorry

#check leak_empty_time_correct

end NUMINAMATH_CALUDE_leak_empty_time_correct_l2718_271802


namespace NUMINAMATH_CALUDE_no_solutions_for_cos_and_odd_multiples_of_90_l2718_271887

theorem no_solutions_for_cos_and_odd_multiples_of_90 :
  ¬ ∃ x : ℝ, 0 ≤ x ∧ x < 720 ∧ Real.cos (x * π / 180) = -0.6 ∧ ∃ n : ℕ, x = (2 * n + 1) * 90 :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_for_cos_and_odd_multiples_of_90_l2718_271887


namespace NUMINAMATH_CALUDE_product_unit_digit_is_one_l2718_271862

def unit_digit (n : ℕ) : ℕ := n % 10

def numbers : List ℕ := [7858413, 10864231, 45823797, 97833129, 51679957, 
                         38213827, 75946153, 27489543, 94837311, 37621597]

theorem product_unit_digit_is_one :
  unit_digit (numbers.prod) = 1 := by
  sorry

#check product_unit_digit_is_one

end NUMINAMATH_CALUDE_product_unit_digit_is_one_l2718_271862


namespace NUMINAMATH_CALUDE_bargain_bin_books_l2718_271852

/-- Calculate the number of books in a bargain bin after sales and additions. -/
def booksInBin (initial : ℕ) (sold : ℕ) (added : ℕ) : ℕ :=
  initial - sold + added

/-- Theorem stating that for the given values, the number of books in the bin is 11. -/
theorem bargain_bin_books : booksInBin 4 3 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bargain_bin_books_l2718_271852


namespace NUMINAMATH_CALUDE_specific_book_arrangement_l2718_271884

/-- The number of arrangements for placing math and English books on a shelf. -/
def book_arrangements (n_math : ℕ) (n_english : ℕ) (adjacent_math : ℕ) : ℕ :=
  Nat.factorial n_english * 
  (n_english - 1) * 
  Nat.choose (n_english + adjacent_math - 1) (n_math - adjacent_math)

/-- Theorem stating the number of arrangements for the specific book problem. -/
theorem specific_book_arrangement : book_arrangements 6 5 2 = 2400 := by
  sorry

#eval book_arrangements 6 5 2

end NUMINAMATH_CALUDE_specific_book_arrangement_l2718_271884


namespace NUMINAMATH_CALUDE_collinear_probability_l2718_271880

/-- The number of dots in each row or column of the grid -/
def gridSize : ℕ := 5

/-- The number of dots to be chosen -/
def dotsChosen : ℕ := 4

/-- The total number of ways to choose 4 dots from a 5x5 grid -/
def totalWays : ℕ := Nat.choose (gridSize * gridSize) dotsChosen

/-- The number of ways to choose 4 collinear dots -/
def collinearWays : ℕ := 
  gridSize * Nat.choose gridSize dotsChosen + -- Horizontal lines
  gridSize * Nat.choose gridSize dotsChosen + -- Vertical lines
  2 * Nat.choose gridSize dotsChosen +        -- Main diagonals
  4                                           -- Adjacent diagonals

/-- The probability of choosing 4 collinear dots from a 5x5 grid -/
theorem collinear_probability : 
  (collinearWays : ℚ) / totalWays = 64 / 12650 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_l2718_271880


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l2718_271885

theorem imaginary_part_of_z_is_zero (z : ℂ) (h : z / (1 + 2*I) = 1 - 2*I) : 
  z.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l2718_271885


namespace NUMINAMATH_CALUDE_log_sum_difference_equals_two_l2718_271853

theorem log_sum_difference_equals_two :
  Real.log 50 / Real.log 10 + Real.log 20 / Real.log 10 - Real.log 10 / Real.log 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_difference_equals_two_l2718_271853


namespace NUMINAMATH_CALUDE_optimal_garden_dimensions_l2718_271854

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  width : Real
  length : Real

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : Real :=
  d.width * d.length

/-- Calculates the perimeter of a rectangular garden -/
def gardenPerimeter (d : GardenDimensions) : Real :=
  2 * (d.width + d.length)

/-- Theorem: Optimal dimensions for a rectangular garden with minimum fencing -/
theorem optimal_garden_dimensions :
  ∃ (d : GardenDimensions),
    d.length = 2 * d.width ∧
    gardenArea d ≥ 500 ∧
    (∀ (d' : GardenDimensions),
      d'.length = 2 * d'.width →
      gardenArea d' ≥ 500 →
      gardenPerimeter d ≤ gardenPerimeter d') ∧
    d.width = 5 * Real.sqrt 10 ∧
    d.length = 10 * Real.sqrt 10 ∧
    gardenPerimeter d = 30 * Real.sqrt 10 :=
  sorry


end NUMINAMATH_CALUDE_optimal_garden_dimensions_l2718_271854


namespace NUMINAMATH_CALUDE_inscribed_polyhedron_volume_relation_l2718_271813

/-- A polyhedron with an inscribed sphere -/
structure InscribedPolyhedron where
  -- The volume of the polyhedron
  volume : ℝ
  -- The radius of the inscribed sphere
  sphereRadius : ℝ
  -- The surface area of the polyhedron
  surfaceArea : ℝ
  -- Assumption that the sphere is inscribed in the polyhedron
  isInscribed : Prop
  -- Assumption that the polyhedron can be decomposed into pyramids
  canDecompose : Prop
  -- Assumption that each pyramid has a face as base and sphere center as apex
  pyramidProperty : Prop

/-- Theorem stating the volume relation for a polyhedron with an inscribed sphere -/
theorem inscribed_polyhedron_volume_relation (p : InscribedPolyhedron) :
  p.volume = (1 / 3) * p.sphereRadius * p.surfaceArea := by
  sorry

end NUMINAMATH_CALUDE_inscribed_polyhedron_volume_relation_l2718_271813


namespace NUMINAMATH_CALUDE_functional_equation_zero_solution_l2718_271833

theorem functional_equation_zero_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x - f y) : 
  ∀ x : ℝ, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_zero_solution_l2718_271833


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt3_l2718_271881

theorem sin_40_tan_10_minus_sqrt3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt3_l2718_271881


namespace NUMINAMATH_CALUDE_yellow_highlighters_count_l2718_271826

theorem yellow_highlighters_count (yellow pink blue : ℕ) : 
  pink = yellow + 7 →
  blue = pink + 5 →
  yellow + pink + blue = 40 →
  yellow = 7 := by
  sorry

end NUMINAMATH_CALUDE_yellow_highlighters_count_l2718_271826


namespace NUMINAMATH_CALUDE_lcm_of_five_numbers_l2718_271863

theorem lcm_of_five_numbers :
  Nat.lcm 456 (Nat.lcm 783 (Nat.lcm 935 (Nat.lcm 1024 1297))) = 2308474368000 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_five_numbers_l2718_271863


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2718_271878

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x - 3) * (|x| + 1) < 0
def q (x : ℝ) : Prop := |1 - x| < 2

-- Theorem stating that p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient :
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2718_271878


namespace NUMINAMATH_CALUDE_initial_number_of_boys_l2718_271849

theorem initial_number_of_boys (initial_girls : ℕ) (boys_dropped : ℕ) (girls_dropped : ℕ) (remaining_total : ℕ) : 
  initial_girls = 10 →
  boys_dropped = 4 →
  girls_dropped = 3 →
  remaining_total = 17 →
  ∃ initial_boys : ℕ, 
    initial_boys - boys_dropped + (initial_girls - girls_dropped) = remaining_total ∧
    initial_boys = 14 :=
by sorry

end NUMINAMATH_CALUDE_initial_number_of_boys_l2718_271849


namespace NUMINAMATH_CALUDE_remainder_101_47_mod_100_l2718_271819

theorem remainder_101_47_mod_100 : 101^47 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_47_mod_100_l2718_271819


namespace NUMINAMATH_CALUDE_postcard_problem_l2718_271825

theorem postcard_problem (initial_postcards : ℕ) : 
  (initial_postcards / 2 + (initial_postcards / 2) * 3 = 36) → 
  initial_postcards = 18 := by
  sorry

end NUMINAMATH_CALUDE_postcard_problem_l2718_271825


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2718_271839

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x ∈ Set.Ico 1 2 → (x^2 - a ≤ 0 → a > 4)) ∧
  ¬(∀ x : ℝ, x ∈ Set.Ico 1 2 → (a > 4 → x^2 - a ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2718_271839


namespace NUMINAMATH_CALUDE_odd_product_remainder_l2718_271821

def odd_product : ℕ → ℕ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then odd_product n else (2 * n + 1) * odd_product n

theorem odd_product_remainder :
  odd_product 1002 % 1000 = 875 :=
sorry

end NUMINAMATH_CALUDE_odd_product_remainder_l2718_271821


namespace NUMINAMATH_CALUDE_subtracted_value_l2718_271859

theorem subtracted_value (N : ℝ) (x : ℝ) 
  (h1 : (N - x) / 7 = 7) 
  (h2 : (N - 14) / 10 = 4) : 
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l2718_271859


namespace NUMINAMATH_CALUDE_business_profit_l2718_271895

theorem business_profit (total_profit : ℝ) : 
  (0.25 * total_profit) + 2 * (0.25 * (0.75 * total_profit)) = 50000 →
  total_profit = 80000 := by
sorry

end NUMINAMATH_CALUDE_business_profit_l2718_271895


namespace NUMINAMATH_CALUDE_greatest_b_for_nonrange_l2718_271877

theorem greatest_b_for_nonrange (b : ℤ) : (∀ x : ℝ, x^2 + b*x + 20 ≠ 5) ↔ b ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_greatest_b_for_nonrange_l2718_271877


namespace NUMINAMATH_CALUDE_alien_invasion_characteristics_l2718_271868

-- Define the characteristics of an alien species invasion
structure AlienInvasion where
  j_shaped_growth : Bool
  unrestricted_growth : Bool
  threatens_biodiversity : Bool
  eliminated_if_unadapted : Bool

-- Define the correct characteristics of an alien invasion
def correct_invasion : AlienInvasion :=
  { j_shaped_growth := true,
    unrestricted_growth := false,
    threatens_biodiversity := true,
    eliminated_if_unadapted := true }

-- Theorem: The correct characteristics of an alien invasion are as defined
theorem alien_invasion_characteristics :
  ∃ (invasion : AlienInvasion),
    invasion.j_shaped_growth ∧
    ¬invasion.unrestricted_growth ∧
    invasion.threatens_biodiversity ∧
    invasion.eliminated_if_unadapted :=
by
  sorry


end NUMINAMATH_CALUDE_alien_invasion_characteristics_l2718_271868


namespace NUMINAMATH_CALUDE_grape_juice_percentage_l2718_271869

/-- Calculates the percentage of grape juice in a mixture after adding pure grape juice -/
theorem grape_juice_percentage
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_pure_juice : ℝ)
  (h1 : initial_volume = 40)
  (h2 : initial_concentration = 0.1)
  (h3 : added_pure_juice = 20)
  : (initial_volume * initial_concentration + added_pure_juice) / (initial_volume + added_pure_juice) = 0.4 := by
  sorry

#check grape_juice_percentage

end NUMINAMATH_CALUDE_grape_juice_percentage_l2718_271869


namespace NUMINAMATH_CALUDE_sum_of_divisors_450_has_three_prime_factors_l2718_271823

/-- The sum of positive divisors function -/
noncomputable def sigma (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors function -/
noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_has_three_prime_factors :
  let n : ℕ := 450
  let sum_of_divisors : ℕ := sigma n
  num_distinct_prime_factors sum_of_divisors = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_450_has_three_prime_factors_l2718_271823


namespace NUMINAMATH_CALUDE_max_triangle_area_l2718_271817

/-- Ellipse type -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Line type -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle type -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Function to calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Function to check if a point is on an ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop := sorry

/-- Function to check if a line intersects an ellipse at two distinct points -/
def intersectsAtTwoPoints (l : Line) (e : Ellipse) : Prop := sorry

/-- Theorem statement -/
theorem max_triangle_area 
  (e : Ellipse) 
  (h_eccentricity : e.a^2 - e.b^2 = e.a^2 / 2)
  (A : Point)
  (h_A_on_ellipse : isOnEllipse A e)
  (h_A_coords : A.x = 1 ∧ A.y = Real.sqrt 2)
  (l : Line)
  (h_l_slope : l.slope = Real.sqrt 2)
  (h_intersects : intersectsAtTwoPoints l e) :
  ∃ (B C : Point), 
    isOnEllipse B e ∧ 
    isOnEllipse C e ∧ 
    B ≠ C ∧
    ∀ (B' C' : Point), 
      isOnEllipse B' e → 
      isOnEllipse C' e → 
      B' ≠ C' →
      triangleArea ⟨A, B', C'⟩ ≤ Real.sqrt 2 ∧
      triangleArea ⟨A, B, C⟩ = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_max_triangle_area_l2718_271817


namespace NUMINAMATH_CALUDE_range_of_a_l2718_271894

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) 
  (h1 : ∀ x, p x → q x)
  (h2 : ∃ x, q x ∧ ¬(p x))
  (hp : ∀ x, p x ↔ x^2 - 2*x - 3 < 0)
  (hq : ∀ x, q x ↔ x > a) :
  a ≤ -1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2718_271894


namespace NUMINAMATH_CALUDE_triangle_quadratic_no_real_roots_l2718_271879

/-- Given a triangle with side lengths a, b, c, the quadratic equation 
    b^2 x^2 - (b^2 + c^2 - a^2)x + c^2 = 0 has no real roots. -/
theorem triangle_quadratic_no_real_roots (a b c : ℝ) 
    (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
    ∀ x : ℝ, b^2 * x^2 - (b^2 + c^2 - a^2) * x + c^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_quadratic_no_real_roots_l2718_271879


namespace NUMINAMATH_CALUDE_factor_polynomial_l2718_271834

theorem factor_polynomial (x : ℝ) : 54 * x^4 - 135 * x^8 = -27 * x^4 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2718_271834


namespace NUMINAMATH_CALUDE_m_value_l2718_271809

/-- Triangle DEF with median DG to side EF -/
structure TriangleDEF where
  DE : ℝ
  EF : ℝ
  DF : ℝ
  DG : ℝ
  is_median : DE = 5 ∧ EF = 12 ∧ DF = 13 ∧ DG * DG = 2 * (m * m)

/-- The value of m in the equation DG = m√2 for the given triangle -/
def find_m (t : TriangleDEF) : ℝ := sorry

/-- Theorem stating that m = √266 / 2 for the given triangle -/
theorem m_value (t : TriangleDEF) : find_m t = Real.sqrt 266 / 2 := by sorry

end NUMINAMATH_CALUDE_m_value_l2718_271809


namespace NUMINAMATH_CALUDE_recreation_spending_percentage_l2718_271842

theorem recreation_spending_percentage
  (last_week_wages : ℝ)
  (last_week_recreation_percent : ℝ)
  (wage_decrease_percent : ℝ)
  (this_week_recreation_increase : ℝ)
  (h1 : last_week_recreation_percent = 10)
  (h2 : wage_decrease_percent = 10)
  (h3 : this_week_recreation_increase = 360) :
  let this_week_wages := last_week_wages * (1 - wage_decrease_percent / 100)
  let last_week_recreation := last_week_wages * (last_week_recreation_percent / 100)
  let this_week_recreation := last_week_recreation * (this_week_recreation_increase / 100)
  this_week_recreation / this_week_wages * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_recreation_spending_percentage_l2718_271842


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2718_271815

theorem triangle_abc_properties (A : Real) (h : Real.sin A + Real.cos A = 1/5) :
  (Real.sin A * Real.cos A = -12/25) ∧
  (π/2 < A ∧ A < π) ∧
  (Real.tan A = -4/3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2718_271815


namespace NUMINAMATH_CALUDE_abs_increasing_on_unit_interval_l2718_271828

/-- The function f(x) = |x| is increasing on the interval (0,1) -/
theorem abs_increasing_on_unit_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → |x₁| < |x₂| := by
  sorry

end NUMINAMATH_CALUDE_abs_increasing_on_unit_interval_l2718_271828


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2718_271836

theorem quadratic_inequality_solution (x : ℝ) : 
  -9 * x^2 + 6 * x + 1 < 0 ↔ (1 - Real.sqrt 2) / 3 < x ∧ x < (1 + Real.sqrt 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2718_271836


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l2718_271801

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x + 5

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

-- Theorem statement
theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x + 3| < ε → f a (-3) ≥ f a x) →
  f' a (-3) = 0 →
  a = 5 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_5_l2718_271801


namespace NUMINAMATH_CALUDE_radhika_games_count_l2718_271864

/-- The number of video games Radhika owns now -/
def total_games (christmas_games birthday_games family_games : ℕ) : ℕ :=
  let total_gifts := christmas_games + birthday_games + family_games
  let initial_games := (2 * total_gifts) / 3
  initial_games + total_gifts

/-- Theorem stating the total number of video games Radhika owns -/
theorem radhika_games_count :
  total_games 12 8 5 = 41 := by
  sorry

#eval total_games 12 8 5

end NUMINAMATH_CALUDE_radhika_games_count_l2718_271864


namespace NUMINAMATH_CALUDE_frozen_yoghurt_cost_l2718_271840

theorem frozen_yoghurt_cost (ice_cream_quantity : ℕ) (frozen_yoghurt_quantity : ℕ) 
  (ice_cream_cost : ℕ) (ice_cream_total : ℕ) (price_difference : ℕ) :
  ice_cream_quantity = 10 →
  frozen_yoghurt_quantity = 4 →
  ice_cream_cost = 4 →
  ice_cream_total = ice_cream_quantity * ice_cream_cost →
  ice_cream_total = price_difference + (frozen_yoghurt_quantity * 1) →
  1 = (ice_cream_total - price_difference) / frozen_yoghurt_quantity :=
by
  sorry

end NUMINAMATH_CALUDE_frozen_yoghurt_cost_l2718_271840


namespace NUMINAMATH_CALUDE_ali_money_problem_l2718_271838

theorem ali_money_problem (initial_money : ℝ) : 
  (initial_money / 2 - (initial_money / 2) / 3 = 160) → initial_money = 480 := by
  sorry

end NUMINAMATH_CALUDE_ali_money_problem_l2718_271838


namespace NUMINAMATH_CALUDE_real_solution_implies_m_positive_l2718_271835

theorem real_solution_implies_m_positive (x m : ℝ) : 
  (∃ x : ℝ, 3^x - m = 0) → m > 0 := by
sorry

end NUMINAMATH_CALUDE_real_solution_implies_m_positive_l2718_271835


namespace NUMINAMATH_CALUDE_broken_cone_height_l2718_271824

/-- Theorem: New height of a broken cone -/
theorem broken_cone_height (r : ℝ) (l : ℝ) (l_new : ℝ) (H : ℝ) :
  r = 6 →
  l = 13 →
  l_new = l - 2 →
  H^2 + r^2 = l_new^2 →
  H = Real.sqrt 85 := by
  sorry

end NUMINAMATH_CALUDE_broken_cone_height_l2718_271824


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l2718_271845

-- Define the point
def point : ℝ × ℝ := (8, -15)

-- Theorem statement
theorem distance_from_origin_to_point :
  Real.sqrt ((point.1 - 0)^2 + (point.2 - 0)^2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l2718_271845


namespace NUMINAMATH_CALUDE_crayons_in_boxes_l2718_271832

/-- Given a number of crayons per box and a number of boxes, 
    calculate the total number of crayons -/
def total_crayons (crayons_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  crayons_per_box * num_boxes

/-- Theorem stating that with 8 crayons per box and 10 boxes, 
    the total number of crayons is 80 -/
theorem crayons_in_boxes : total_crayons 8 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_boxes_l2718_271832


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l2718_271891

def total_amount : ℕ := 600
def amount_a : ℕ := 200

theorem ratio_a_to_b : 
  (amount_a : ℚ) / ((total_amount - amount_a) : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l2718_271891


namespace NUMINAMATH_CALUDE_base_subtraction_l2718_271822

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The theorem statement --/
theorem base_subtraction :
  to_base_10 [3, 2, 5] 9 - to_base_10 [2, 3, 1] 6 = 175 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l2718_271822


namespace NUMINAMATH_CALUDE_negation_of_sine_inequality_l2718_271858

theorem negation_of_sine_inequality :
  (¬ ∀ x : ℝ, |Real.sin x| < 1) ↔ (∃ x : ℝ, |Real.sin x| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_sine_inequality_l2718_271858


namespace NUMINAMATH_CALUDE_angle_A_is_pi_div_6_max_area_when_a_is_2_l2718_271808

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.a * Real.cos t.C = (2 * t.b - Real.sqrt 3 * t.c) * Real.cos t.A

-- Theorem 1: Prove that angle A = π/6
theorem angle_A_is_pi_div_6 (t : Triangle) (h : condition t) : t.A = π / 6 :=
sorry

-- Theorem 2: Prove that when a = 2, the maximum area of triangle ABC is 2 + √3
theorem max_area_when_a_is_2 (t : Triangle) (h1 : condition t) (h2 : t.a = 2) :
  (t.b * t.c * Real.sin t.A / 2) ≤ 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_div_6_max_area_when_a_is_2_l2718_271808


namespace NUMINAMATH_CALUDE_cosine_power_identity_l2718_271892

theorem cosine_power_identity (θ : ℝ) (u : ℝ) (n : ℤ) :
  2 * Real.cos θ = u + 1 / u →
  2 * Real.cos (n * θ) = u^n + 1 / u^n :=
by sorry

end NUMINAMATH_CALUDE_cosine_power_identity_l2718_271892


namespace NUMINAMATH_CALUDE_f_congruence_implies_input_congruence_l2718_271857

def f (x : ℤ) : ℤ := x^3 + 7*x^2 + 9*x + 10

theorem f_congruence_implies_input_congruence :
  ∀ (a b : ℤ), f a ≡ f b [ZMOD 11] → a ≡ b [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_f_congruence_implies_input_congruence_l2718_271857


namespace NUMINAMATH_CALUDE_complement_of_46_35_l2718_271899

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- Calculates the complement of an angle -/
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  { degrees := totalMinutes / 60, minutes := totalMinutes % 60 }

/-- The theorem stating that the complement of 46°35' is 43°25' -/
theorem complement_of_46_35 :
  complement { degrees := 46, minutes := 35 } = { degrees := 43, minutes := 25 } := by
  sorry

end NUMINAMATH_CALUDE_complement_of_46_35_l2718_271899


namespace NUMINAMATH_CALUDE_sin_x_plus_y_eq_one_sixth_l2718_271876

theorem sin_x_plus_y_eq_one_sixth (x y : ℝ) 
  (h1 : 3 * Real.sin x + 4 * Real.cos y = 5) 
  (h2 : 4 * Real.sin y + 3 * Real.cos x = 2) : 
  Real.sin (x + y) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_sin_x_plus_y_eq_one_sixth_l2718_271876


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2718_271882

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x * y - x - 2 * y = 0) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a * b - a - 2 * b = 0 → x + y ≤ a + b ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀ - x₀ - 2 * y₀ = 0 ∧ x₀ + y₀ = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2718_271882


namespace NUMINAMATH_CALUDE_boiling_temperature_calculation_boiling_temperature_proof_l2718_271816

theorem boiling_temperature_calculation (initial_temp : ℝ) (temp_increase : ℝ) 
  (pasta_time : ℝ) (total_time : ℝ) : ℝ :=
  let mixing_time := pasta_time / 3
  let cooking_and_mixing_time := pasta_time + mixing_time
  let time_to_boil := total_time - cooking_and_mixing_time
  let temp_increase_total := time_to_boil * temp_increase
  initial_temp + temp_increase_total

theorem boiling_temperature_proof :
  boiling_temperature_calculation 41 3 12 73 = 212 := by
  sorry

end NUMINAMATH_CALUDE_boiling_temperature_calculation_boiling_temperature_proof_l2718_271816


namespace NUMINAMATH_CALUDE_john_payment_is_1200_l2718_271831

/-- Calculates John's payment for renting a camera -/
def johnPayment (cameraValue : ℝ) (rentalRatePerWeek : ℝ) (rentalWeeks : ℕ) (friendContributionRate : ℝ) : ℝ :=
  let totalRental := cameraValue * rentalRatePerWeek * rentalWeeks
  let friendContribution := totalRental * friendContributionRate
  totalRental - friendContribution

/-- Theorem stating that John's payment is $1200 given the problem conditions -/
theorem john_payment_is_1200 :
  johnPayment 5000 0.1 4 0.4 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_john_payment_is_1200_l2718_271831


namespace NUMINAMATH_CALUDE_initial_average_weight_l2718_271807

/-- Proves the initially calculated average weight given the conditions of the problem -/
theorem initial_average_weight (n : ℕ) (misread_weight correct_weight : ℝ) (correct_avg : ℝ) :
  n = 20 ∧ 
  misread_weight = 56 ∧
  correct_weight = 61 ∧
  correct_avg = 58.65 →
  ∃ initial_avg : ℝ, 
    initial_avg * n + (correct_weight - misread_weight) = correct_avg * n ∧
    initial_avg = 58.4 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_weight_l2718_271807


namespace NUMINAMATH_CALUDE_roots_sum_theorem_l2718_271871

theorem roots_sum_theorem (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a + b + c = 6 →
  a*b + a*c + b*c = 11 →
  a*b*c = 6 →
  (a / (b*c + 2)) + (b / (a*c + 2)) + (c / (a*b + 2)) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_theorem_l2718_271871


namespace NUMINAMATH_CALUDE_leaf_travel_11_gusts_l2718_271848

/-- The net distance traveled by a leaf after a number of wind gusts -/
def leaf_travel (gusts : ℕ) (forward : ℕ) (backward : ℕ) : ℤ :=
  (gusts * forward : ℤ) - (gusts * backward : ℤ)

/-- Theorem: The leaf travels 33 feet after 11 gusts of wind -/
theorem leaf_travel_11_gusts :
  leaf_travel 11 5 2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_leaf_travel_11_gusts_l2718_271848


namespace NUMINAMATH_CALUDE_no_real_solutions_l2718_271837

theorem no_real_solutions : ¬∃ (x y z : ℝ), (x + y = 3) ∧ (x * y - z^2 = 3) ∧ (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2718_271837


namespace NUMINAMATH_CALUDE_train_length_l2718_271805

/-- The length of a train given specific crossing times -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ) :
  bridge_length = 200 ∧ bridge_time = 10 ∧ post_time = 5 →
  ∃ train_length : ℝ, train_length = 200 ∧
    train_length / post_time = (train_length + bridge_length) / bridge_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2718_271805


namespace NUMINAMATH_CALUDE_bottle_cap_count_l2718_271860

theorem bottle_cap_count (caps_per_box : ℝ) (num_boxes : ℝ) 
  (h1 : caps_per_box = 35.0) 
  (h2 : num_boxes = 7.0) : 
  caps_per_box * num_boxes = 245.0 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_count_l2718_271860


namespace NUMINAMATH_CALUDE_inheritance_problem_l2718_271804

theorem inheritance_problem (x : ℝ) : 
  (100 + (1/10) * (x - 100) = 200 + (1/10) * (x - (100 + (1/10) * (x - 100)) - 200)) →
  x = 8100 := by
sorry

end NUMINAMATH_CALUDE_inheritance_problem_l2718_271804


namespace NUMINAMATH_CALUDE_scientific_notation_of_rural_population_l2718_271872

theorem scientific_notation_of_rural_population :
  ∃ (x : ℝ), x = 42.39 * 10^6 ∧ x = 4.239 * 10^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_rural_population_l2718_271872


namespace NUMINAMATH_CALUDE_largest_prime_factor_l2718_271874

theorem largest_prime_factor : 
  (Nat.factors (18^4 + 3*18^2 + 1 - 16^4)).maximum = some 83 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l2718_271874


namespace NUMINAMATH_CALUDE_triangle_problem_l2718_271896

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a^2 - t.b^2 = Real.sqrt 3 * t.b * t.c ∧
  Real.sin t.C = 2 * Real.sqrt 3 * Real.sin t.B

theorem triangle_problem (t : Triangle) (h : TriangleConditions t) :
  t.A = π/6 ∧ (t.b = 1 → 1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2718_271896


namespace NUMINAMATH_CALUDE_triangle_half_angle_sine_product_l2718_271800

theorem triangle_half_angle_sine_product (A B C : ℝ) (h_triangle : A + B + C = π) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_half_angle_sine_product_l2718_271800


namespace NUMINAMATH_CALUDE_hiker_first_pack_weight_l2718_271846

/-- Calculates the weight of the first pack for a hiker given specific conditions --/
theorem hiker_first_pack_weight
  (supplies_per_mile : Real)
  (hiking_rate : Real)
  (hours_per_day : Real)
  (days : Real)
  (first_resupply_ratio : Real)
  (second_resupply_ratio : Real)
  (h1 : supplies_per_mile = 0.6)
  (h2 : hiking_rate = 2.5)
  (h3 : hours_per_day = 9)
  (h4 : days = 7)
  (h5 : first_resupply_ratio = 0.3)
  (h6 : second_resupply_ratio = 0.2) :
  let total_distance := hiking_rate * hours_per_day * days
  let total_supplies := supplies_per_mile * total_distance
  let first_resupply := first_resupply_ratio * total_supplies
  let second_resupply := second_resupply_ratio * total_supplies
  let first_pack_weight := total_supplies - (first_resupply + second_resupply)
  first_pack_weight = 47.25 := by sorry

end NUMINAMATH_CALUDE_hiker_first_pack_weight_l2718_271846


namespace NUMINAMATH_CALUDE_painting_time_equation_l2718_271850

/-- Represents the time (in hours) it takes for a person to paint a room alone -/
structure PaintTime where
  hours : ℝ
  hours_positive : hours > 0

/-- Represents the painting scenario with Doug and Dave -/
structure PaintingScenario where
  doug_time : PaintTime
  dave_time : PaintTime
  doug_start_time : ℝ
  dave_join_time : ℝ
  total_time : ℝ
  doug_start_first : doug_start_time = 0
  dave_joins_later : dave_join_time > doug_start_time

/-- The main theorem stating the equation that the total painting time satisfies -/
theorem painting_time_equation (scenario : PaintingScenario) 
  (h1 : scenario.doug_time.hours = 3)
  (h2 : scenario.dave_time.hours = 4)
  (h3 : scenario.dave_join_time = 1) :
  (scenario.total_time - 1) * (7/12 : ℝ) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_painting_time_equation_l2718_271850


namespace NUMINAMATH_CALUDE_total_players_count_l2718_271861

/-- The number of players who play kabadi (including those who play both) -/
def kabadi_players : ℕ := 10

/-- The number of players who play kho kho only -/
def kho_kho_only_players : ℕ := 15

/-- The number of players who play both games -/
def both_games_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := kabadi_players + kho_kho_only_players - both_games_players

theorem total_players_count : total_players = 20 := by sorry

end NUMINAMATH_CALUDE_total_players_count_l2718_271861


namespace NUMINAMATH_CALUDE_real_imaginary_intersection_empty_l2718_271820

-- Define the universal set C (complex numbers)
variable (C : Type)

-- Define R (real numbers) and I (pure imaginary numbers) as subsets of C
variable (R I : Set C)

-- Theorem statement
theorem real_imaginary_intersection_empty : R ∩ I = ∅ := by
  sorry

end NUMINAMATH_CALUDE_real_imaginary_intersection_empty_l2718_271820


namespace NUMINAMATH_CALUDE_composition_equality_l2718_271810

theorem composition_equality (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x / 3 + 2) →
  (∀ x, g x = 5 - 2 * x) →
  f (g a) = 4 →
  a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_composition_equality_l2718_271810


namespace NUMINAMATH_CALUDE_win_loss_opposite_win_loss_opposite_meanings_l2718_271866

/-- Represents the outcome of a game -/
inductive GameOutcome
| Win
| Loss

/-- Represents a team's or individual's record -/
structure Record where
  wins : ℕ
  losses : ℕ

/-- Updates the record based on a game outcome -/
def updateRecord (r : Record) (outcome : GameOutcome) : Record :=
  match outcome with
  | GameOutcome.Win => { wins := r.wins + 1, losses := r.losses }
  | GameOutcome.Loss => { wins := r.wins, losses := r.losses + 1 }

/-- Theorem stating that winning and losing have opposite effects on a record -/
theorem win_loss_opposite (r : Record) :
  updateRecord r GameOutcome.Win ≠ updateRecord r GameOutcome.Loss :=
by
  sorry

/-- Theorem stating that winning and losing are quantities with opposite meanings -/
theorem win_loss_opposite_meanings :
  ∃ (r : Record), updateRecord r GameOutcome.Win ≠ updateRecord r GameOutcome.Loss :=
by
  sorry

end NUMINAMATH_CALUDE_win_loss_opposite_win_loss_opposite_meanings_l2718_271866


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2718_271883

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a ≤ 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2718_271883


namespace NUMINAMATH_CALUDE_owen_work_hours_l2718_271889

/-- Represents the number of hours in a day -/
def hours_in_day : ℕ := 24

/-- Represents the number of hours Owen spends on daily chores -/
def hours_on_chores : ℕ := 7

/-- Represents the number of hours Owen sleeps -/
def hours_sleeping : ℕ := 11

/-- Calculates the number of hours Owen spends at work -/
def hours_at_work : ℕ := hours_in_day - hours_on_chores - hours_sleeping

/-- Theorem stating that Owen spends 6 hours at work -/
theorem owen_work_hours : hours_at_work = 6 := by
  sorry

end NUMINAMATH_CALUDE_owen_work_hours_l2718_271889


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2718_271841

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℂ, x^2 - (2*Complex.I - 1)*x + 3*m - Complex.I = 0 ∧ x.im = 0) → m = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2718_271841


namespace NUMINAMATH_CALUDE_original_average_from_doubled_l2718_271814

theorem original_average_from_doubled (n : ℕ) (A : ℚ) (h1 : n = 10) (h2 : 2 * A = 80) : A = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_average_from_doubled_l2718_271814


namespace NUMINAMATH_CALUDE_negate_neg_sum_l2718_271856

theorem negate_neg_sum (a b : ℝ) : -(-a - b) = a + b := by
  sorry

end NUMINAMATH_CALUDE_negate_neg_sum_l2718_271856


namespace NUMINAMATH_CALUDE_four_plus_five_result_l2718_271806

/-- Define the sequence operation for two consecutive integers -/
def seqOperation (a b : ℕ) : ℕ := (a + b)^2 + 1

/-- Theorem stating that 4 + 5 results in 82 in the given sequence -/
theorem four_plus_five_result :
  seqOperation 4 5 = 82 :=
by sorry

end NUMINAMATH_CALUDE_four_plus_five_result_l2718_271806


namespace NUMINAMATH_CALUDE_journey_distance_l2718_271830

/-- Represents the journey from John's house to the conference center -/
structure Journey where
  initial_speed : ℝ             -- Initial speed in miles per hour
  initial_distance : ℝ          -- Distance covered in the first hour
  late_time : ℝ                 -- Time he would be late if continued at initial speed
  speed_increase : ℝ            -- Increase in speed for the rest of the journey
  early_time : ℝ                -- Time he arrives early after increasing speed

/-- Calculates the total distance of the journey -/
def calculate_distance (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that the distance to the conference center is 191.25 miles -/
theorem journey_distance (j : Journey) 
  (h1 : j.initial_speed = 45)
  (h2 : j.initial_distance = 45)
  (h3 : j.late_time = 0.75)
  (h4 : j.speed_increase = 20)
  (h5 : j.early_time = 0.25) :
  calculate_distance j = 191.25 :=
sorry

end NUMINAMATH_CALUDE_journey_distance_l2718_271830


namespace NUMINAMATH_CALUDE_elisas_painting_l2718_271803

theorem elisas_painting (monday : ℝ) 
  (h1 : monday > 0)
  (h2 : monday + 2 * monday + monday / 2 = 105) : 
  monday = 30 := by
  sorry

end NUMINAMATH_CALUDE_elisas_painting_l2718_271803


namespace NUMINAMATH_CALUDE_sequence_length_l2718_271873

/-- Proves that an arithmetic sequence starting at 2.5, ending at 67.5, with a common difference of 5, has 14 terms. -/
theorem sequence_length : 
  ∀ (a : ℚ) (d : ℚ) (last : ℚ) (n : ℕ),
  a = 2.5 ∧ d = 5 ∧ last = 67.5 →
  last = a + (n - 1) * d →
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_sequence_length_l2718_271873


namespace NUMINAMATH_CALUDE_triangle_equality_l2718_271897

/-- Given a triangle ABC with sides a, b, and c satisfying a^2 + b^2 + c^2 = ab + bc + ac,
    prove that the triangle is equilateral. -/
theorem triangle_equality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
    (eq : a^2 + b^2 + c^2 = a*b + b*c + a*c) : a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_l2718_271897


namespace NUMINAMATH_CALUDE_wedding_gift_cost_l2718_271886

/-- The cost of a single steak knife set -/
def steak_knife_set_cost : ℝ := 80

/-- The number of steak knife sets Elizabeth wants to buy -/
def num_steak_knife_sets : ℕ := 2

/-- The cost of the dinnerware set -/
def dinnerware_set_cost : ℝ := 200

/-- The discount rate applied to the total purchase -/
def discount_rate : ℝ := 0.1

/-- The sales tax rate applied after the discount -/
def sales_tax_rate : ℝ := 0.05

/-- The total cost Elizabeth will spend on the wedding gift -/
def total_cost : ℝ :=
  let initial_cost := steak_knife_set_cost * num_steak_knife_sets + dinnerware_set_cost
  let discounted_cost := initial_cost * (1 - discount_rate)
  discounted_cost * (1 + sales_tax_rate)

theorem wedding_gift_cost : total_cost = 340.20 := by
  sorry

end NUMINAMATH_CALUDE_wedding_gift_cost_l2718_271886


namespace NUMINAMATH_CALUDE_midpoint_product_l2718_271875

/-- Given that C = (4, 3) is the midpoint of line segment AB, where A = (2, 6) and B = (x, y), prove that xy = 0 -/
theorem midpoint_product (x y : ℝ) : 
  (4 : ℝ) = (2 + x) / 2 → 
  (3 : ℝ) = (6 + y) / 2 → 
  x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_product_l2718_271875


namespace NUMINAMATH_CALUDE_math_problem_l2718_271811

theorem math_problem :
  (8 * 40 = 320) ∧
  (5 * (1 / 6) = 5 / 6) ∧
  (6 * 500 = 3000) ∧
  (∃ n : ℕ, 3000 = n * 1000) := by
sorry

end NUMINAMATH_CALUDE_math_problem_l2718_271811


namespace NUMINAMATH_CALUDE_expression_equality_l2718_271898

theorem expression_equality (x y z : ℝ) 
  (h1 : x * y = 6)
  (h2 : x - z = 2)
  (h3 : x + y + z = 9) :
  x / y - z / x - z^2 / (x * y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2718_271898


namespace NUMINAMATH_CALUDE_shell_addition_problem_l2718_271844

/-- Calculates the final addition of shells given the initial amount, additions, and removals. -/
def final_addition (initial : ℕ) (first_addition : ℕ) (removal : ℕ) (total : ℕ) : ℕ :=
  total - (initial + first_addition - removal)

/-- Proves that the final addition of shells is 16 pounds given the problem conditions. -/
theorem shell_addition_problem :
  final_addition 5 9 2 28 = 16 := by
  sorry

end NUMINAMATH_CALUDE_shell_addition_problem_l2718_271844


namespace NUMINAMATH_CALUDE_apple_sale_loss_l2718_271893

/-- The fraction of the cost price lost by a seller when selling an item -/
def fractionLost (sellingPrice costPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice

/-- Theorem: The fraction of the cost price lost when selling an apple for 19 Rs with a cost price of 20 Rs is 1/20 -/
theorem apple_sale_loss : fractionLost 19 20 = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_apple_sale_loss_l2718_271893


namespace NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l2718_271890

theorem smallest_sum_of_a_and_b (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 - a*x + 3*b = 0) →
  (∃ x : ℝ, x^2 - 3*b*x + a = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x : ℝ, x^2 - a'*x + 3*b' = 0) →
    (∃ x : ℝ, x^2 - 3*b'*x + a' = 0) →
    a + b ≤ a' + b') →
  a + b = 32 * Real.sqrt 3 / 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l2718_271890


namespace NUMINAMATH_CALUDE_watermelon_weight_theorem_l2718_271829

/-- Represents the weight of watermelons in a basket -/
structure WatermelonBasket where
  initialWeight : ℝ  -- Initial weight of basket with watermelons
  halfRemovedWeight : ℝ  -- Weight after removing half of watermelons
  basketWeight : ℝ  -- Weight of the empty basket

/-- Calculates the total weight of watermelons in the basket -/
def totalWatermelonWeight (basket : WatermelonBasket) : ℝ :=
  basket.initialWeight - basket.basketWeight

/-- Theorem stating the total weight of watermelons in the given scenario -/
theorem watermelon_weight_theorem (basket : WatermelonBasket) 
  (h1 : basket.initialWeight = 63)
  (h2 : basket.halfRemovedWeight = 34)
  (h3 : basket.basketWeight = basket.halfRemovedWeight - (basket.initialWeight - basket.basketWeight) / 2) :
  totalWatermelonWeight basket = 58 := by
  sorry

#check watermelon_weight_theorem

end NUMINAMATH_CALUDE_watermelon_weight_theorem_l2718_271829


namespace NUMINAMATH_CALUDE_yellow_gumdrops_after_replacement_l2718_271865

/-- Represents the number of gumdrops of each color in a jar -/
structure GumdropsJar where
  blue : ℕ
  brown : ℕ
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- The total number of gumdrops in the jar -/
def GumdropsJar.total (jar : GumdropsJar) : ℕ :=
  jar.blue + jar.brown + jar.red + jar.yellow + jar.green

/-- The percentage of gumdrops of a given color -/
def GumdropsJar.percentage (jar : GumdropsJar) (color : ℕ) : ℚ :=
  color / jar.total

theorem yellow_gumdrops_after_replacement (jar : GumdropsJar) :
  jar.blue = (jar.total * 2) / 5 →
  jar.brown = (jar.total * 3) / 20 →
  jar.red = jar.total / 10 →
  jar.yellow = jar.total / 5 →
  jar.green = 50 →
  (jar.yellow + jar.red / 3 : ℕ) = 78 := by
  sorry

end NUMINAMATH_CALUDE_yellow_gumdrops_after_replacement_l2718_271865


namespace NUMINAMATH_CALUDE_parabola_intersection_midpoint_l2718_271888

/-- Parabola defined by y² = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Condition that A and B are on the parabola and |AF| + |BF| = 10 -/
def IntersectionCondition (A B : ℝ × ℝ) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧
  Real.sqrt ((A.1 - Focus.1)^2 + (A.2 - Focus.2)^2) +
  Real.sqrt ((B.1 - Focus.1)^2 + (B.2 - Focus.2)^2) = 10

/-- The theorem to be proved -/
theorem parabola_intersection_midpoint
  (A B : ℝ × ℝ) (h : IntersectionCondition A B) :
  (A.1 + B.1) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_midpoint_l2718_271888


namespace NUMINAMATH_CALUDE_builder_total_payment_l2718_271843

/-- Calculates the total amount paid for a purchase of drill bits, hammers, and a toolbox with specific taxes and discounts. -/
def total_amount_paid (drill_bit_sets : ℕ) (drill_bit_price : ℚ) (drill_bit_tax : ℚ)
                      (hammers : ℕ) (hammer_price : ℚ) (hammer_discount : ℚ)
                      (toolbox_price : ℚ) (toolbox_tax : ℚ) : ℚ :=
  let drill_bits_cost := drill_bit_sets * drill_bit_price * (1 + drill_bit_tax)
  let hammers_cost := hammers * hammer_price * (1 - hammer_discount)
  let toolbox_cost := toolbox_price * (1 + toolbox_tax)
  drill_bits_cost + hammers_cost + toolbox_cost

/-- The total amount paid by the builder is $84.55. -/
theorem builder_total_payment :
  total_amount_paid 5 6 (10/100) 3 8 (5/100) 25 (15/100) = 8455/100 := by
  sorry

end NUMINAMATH_CALUDE_builder_total_payment_l2718_271843


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l2718_271847

theorem farmer_land_ownership (total_land : ℝ) : 
  (0.9 * total_land * 0.1 = 540) →
  total_land = 6000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l2718_271847


namespace NUMINAMATH_CALUDE_keanu_fish_problem_l2718_271867

theorem keanu_fish_problem :
  ∀ (dog_fish cat_fish : ℕ),
    cat_fish = dog_fish / 2 →
    dog_fish + cat_fish = 240 / 4 →
    dog_fish = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_keanu_fish_problem_l2718_271867


namespace NUMINAMATH_CALUDE_decimal_123_to_binary_l2718_271827

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

theorem decimal_123_to_binary :
  decimal_to_binary 123 = [true, true, false, true, true, true, true] := by
  sorry

end NUMINAMATH_CALUDE_decimal_123_to_binary_l2718_271827


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2718_271812

theorem gcd_of_three_numbers : Nat.gcd 779 (Nat.gcd 209 589) = 19 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2718_271812


namespace NUMINAMATH_CALUDE_triangle_side_length_l2718_271818

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if A = 60°, b = 4, and the area is 2√3, then a = 2√3 -/
theorem triangle_side_length (a b c : ℝ) (A : Real) (S : ℝ) :
  A = π / 3 →  -- 60° in radians
  b = 4 →
  S = 2 * Real.sqrt 3 →
  S = 1 / 2 * b * c * Real.sin A →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  a = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2718_271818


namespace NUMINAMATH_CALUDE_final_bill_amount_l2718_271870

def original_bill : ℝ := 500
def late_charge_rate : ℝ := 0.02

theorem final_bill_amount :
  let first_increase := original_bill * (1 + late_charge_rate)
  let final_bill := first_increase * (1 + late_charge_rate)
  final_bill = 520.20 := by sorry

end NUMINAMATH_CALUDE_final_bill_amount_l2718_271870
