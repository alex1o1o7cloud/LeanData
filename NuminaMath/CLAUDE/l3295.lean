import Mathlib

namespace NUMINAMATH_CALUDE_complex_equation_solution_l3295_329501

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3295_329501


namespace NUMINAMATH_CALUDE_sally_grew_six_carrots_l3295_329561

/-- The number of carrots grown by Fred -/
def fred_carrots : ℕ := 4

/-- The total number of carrots grown by Sally and Fred -/
def total_carrots : ℕ := 10

/-- The number of carrots grown by Sally -/
def sally_carrots : ℕ := total_carrots - fred_carrots

theorem sally_grew_six_carrots : sally_carrots = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_grew_six_carrots_l3295_329561


namespace NUMINAMATH_CALUDE_max_sum_with_reciprocals_l3295_329551

theorem max_sum_with_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y = 5) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b + 1/a + 1/b = 5 → x + y ≥ a + b ∧ x + y ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_reciprocals_l3295_329551


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3295_329566

theorem intersection_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | |x| ≤ 4} → 
  B = {x : ℝ | 4 ≤ x ∧ x < 5} → 
  A ∩ B = {4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3295_329566


namespace NUMINAMATH_CALUDE_vegetarian_gluten_free_fraction_is_one_twentieth_l3295_329517

/-- Represents a restaurant menu -/
structure Menu :=
  (total_dishes : ℕ)
  (vegetarian_dishes : ℕ)
  (gluten_free_vegetarian_dishes : ℕ)

/-- The fraction of dishes that are both vegetarian and gluten-free -/
def vegetarian_gluten_free_fraction (menu : Menu) : ℚ :=
  menu.gluten_free_vegetarian_dishes / menu.total_dishes

theorem vegetarian_gluten_free_fraction_is_one_twentieth
  (menu : Menu)
  (h1 : menu.vegetarian_dishes = 4)
  (h2 : menu.vegetarian_dishes = menu.total_dishes / 5)
  (h3 : menu.gluten_free_vegetarian_dishes = menu.vegetarian_dishes - 3) :
  vegetarian_gluten_free_fraction menu = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_gluten_free_fraction_is_one_twentieth_l3295_329517


namespace NUMINAMATH_CALUDE_root_values_l3295_329521

theorem root_values (p q r s m : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : p * m^3 + q * m^2 + r * m + s = 0)
  (h2 : q * m^3 + r * m^2 + (s + m * m) * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_root_values_l3295_329521


namespace NUMINAMATH_CALUDE_white_surface_fraction_l3295_329502

/-- Represents a cube with its properties -/
structure Cube where
  edge_length : ℕ
  total_subcubes : ℕ
  white_subcubes : ℕ
  black_subcubes : ℕ

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℕ := 6 * c.edge_length * c.edge_length

/-- Calculates the number of exposed faces of subcubes at diagonal ends -/
def exposed_diagonal_faces (c : Cube) : ℕ := 3 * c.black_subcubes

/-- Theorem: The fraction of white surface area in the given cube configuration is 1/2 -/
theorem white_surface_fraction (c : Cube) 
  (h1 : c.edge_length = 4)
  (h2 : c.total_subcubes = 64)
  (h3 : c.white_subcubes = 48)
  (h4 : c.black_subcubes = 16)
  (h5 : exposed_diagonal_faces c = c.black_subcubes * 3) :
  (surface_area c - exposed_diagonal_faces c) / surface_area c = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_fraction_l3295_329502


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3295_329572

theorem decimal_to_fraction :
  (35 : ℚ) / 100 = 7 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3295_329572


namespace NUMINAMATH_CALUDE_ellipse_slope_l3295_329505

/-- Given an ellipse with eccentricity √3/2 and a point P on the ellipse such that
    the sum of tangents of angles formed by PA and PB with the x-axis is 1,
    prove that the slope of PA is (1 ± √2)/2. -/
theorem ellipse_slope (a b : ℝ) (x y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  let e := Real.sqrt 3 / 2
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  let A := (-a, 0)
  let B := (a, 0)
  let P := (x, y)
  ∀ (α β : ℝ),
    e = Real.sqrt (a^2 - b^2) / a →
    P ∈ C →
    (y / (x + a)) + (y / (x - a)) = 1 →
    (∃ (k : ℝ), k = y / (x + a) ∧ (k = (1 + Real.sqrt 2) / 2 ∨ k = (1 - Real.sqrt 2) / 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_slope_l3295_329505


namespace NUMINAMATH_CALUDE_goose_egg_hatching_rate_l3295_329519

theorem goose_egg_hatching_rate : 
  ∀ (total_eggs : ℕ) (hatched_eggs : ℕ),
    (hatched_eggs : ℚ) / total_eggs = 1 →
    (3 : ℚ) / 4 * ((2 : ℚ) / 5 * hatched_eggs) = 180 →
    hatched_eggs ≤ total_eggs →
    (hatched_eggs : ℚ) / total_eggs = 1 := by
  sorry

end NUMINAMATH_CALUDE_goose_egg_hatching_rate_l3295_329519


namespace NUMINAMATH_CALUDE_fraction_addition_l3295_329511

theorem fraction_addition : (168 : ℚ) / 240 + 100 / 150 = 41 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3295_329511


namespace NUMINAMATH_CALUDE_statement_equivalence_l3295_329508

-- Define the statement as a function that takes a real number y
def statementIsTrue (y : ℝ) : Prop := (1/2 * y + 5) > 0

-- Define the theorem
theorem statement_equivalence :
  ∀ y : ℝ, statementIsTrue y ↔ (1/2 * y + 5 > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_statement_equivalence_l3295_329508


namespace NUMINAMATH_CALUDE_weight_of_b_l3295_329510

theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 43 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  b = 37 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l3295_329510


namespace NUMINAMATH_CALUDE_molecular_weight_BaCl2_calculation_l3295_329590

/-- The molecular weight of 8 moles of BaCl2 -/
def molecular_weight_BaCl2 (atomic_weight_Ba : ℝ) (atomic_weight_Cl : ℝ) : ℝ :=
  8 * (atomic_weight_Ba + 2 * atomic_weight_Cl)

/-- Theorem stating the molecular weight of 8 moles of BaCl2 -/
theorem molecular_weight_BaCl2_calculation :
  molecular_weight_BaCl2 137.33 35.45 = 1665.84 := by
  sorry

#eval molecular_weight_BaCl2 137.33 35.45

end NUMINAMATH_CALUDE_molecular_weight_BaCl2_calculation_l3295_329590


namespace NUMINAMATH_CALUDE_increasing_interval_of_f_l3295_329582

noncomputable def f (x : ℝ) := x - Real.exp x

theorem increasing_interval_of_f :
  {x : ℝ | ∀ y, x < y → f x < f y} = Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_increasing_interval_of_f_l3295_329582


namespace NUMINAMATH_CALUDE_climb_8_stairs_l3295_329593

/-- The number of ways to climb n stairs, taking 1, 2, 3, or 4 steps at a time. -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | n + 4 => climbStairs n + climbStairs (n + 1) + climbStairs (n + 2) + climbStairs (n + 3)

/-- Theorem stating that there are 108 ways to climb 8 stairs -/
theorem climb_8_stairs : climbStairs 8 = 108 := by
  sorry

end NUMINAMATH_CALUDE_climb_8_stairs_l3295_329593


namespace NUMINAMATH_CALUDE_correct_calculation_l3295_329526

theorem correct_calculation (x : ℤ) (h : 23 - x = 4) : 23 * x = 437 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3295_329526


namespace NUMINAMATH_CALUDE_games_in_own_group_l3295_329580

/-- Represents a baseball league with two groups of teams. -/
structure BaseballLeague where
  n : ℕ  -- Number of games played against each team in own group
  m : ℕ  -- Number of games played against each team in other group

/-- Theorem about the number of games played within a team's own group. -/
theorem games_in_own_group (league : BaseballLeague)
  (h1 : league.n > 2 * league.m)
  (h2 : league.m > 4)
  (h3 : 3 * league.n + 4 * league.m = 76) :
  3 * league.n = 48 := by
sorry

end NUMINAMATH_CALUDE_games_in_own_group_l3295_329580


namespace NUMINAMATH_CALUDE_area_covered_by_strips_l3295_329535

/-- The area covered by overlapping rectangular strips -/
theorem area_covered_by_strips (n : ℕ) (length width overlap_length : ℝ) : 
  n = 5 → 
  length = 12 → 
  width = 1 → 
  overlap_length = 2 → 
  (n : ℝ) * length * width - (n.choose 2 : ℝ) * overlap_length * width = 40 := by
  sorry

#check area_covered_by_strips

end NUMINAMATH_CALUDE_area_covered_by_strips_l3295_329535


namespace NUMINAMATH_CALUDE_no_solutions_for_inequality_l3295_329513

theorem no_solutions_for_inequality : 
  ¬ ∃ (n : ℕ), n ≥ 1 ∧ n ≤ n! - 4^n ∧ n! - 4^n ≤ 4*n :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_for_inequality_l3295_329513


namespace NUMINAMATH_CALUDE_cookie_box_duration_l3295_329544

/-- Proves that a box of cookies lasts 9 days given the specified conditions -/
theorem cookie_box_duration (oldest_son_cookies : ℕ) (youngest_son_cookies : ℕ) (total_cookies : ℕ) : 
  oldest_son_cookies = 4 → 
  youngest_son_cookies = 2 → 
  total_cookies = 54 → 
  (total_cookies / (oldest_son_cookies + youngest_son_cookies) : ℕ) = 9 := by
  sorry

#check cookie_box_duration

end NUMINAMATH_CALUDE_cookie_box_duration_l3295_329544


namespace NUMINAMATH_CALUDE_pulley_center_distance_l3295_329507

def pulley_problem (r₁ r₂ d : ℝ) : Prop :=
  r₁ > 0 ∧ r₂ > 0 ∧ d > 0 ∧
  r₁ = 10 ∧ r₂ = 6 ∧ d = 26 →
  ∃ (center_distance : ℝ),
    center_distance = 2 * Real.sqrt 173

theorem pulley_center_distance :
  ∀ (r₁ r₂ d : ℝ), pulley_problem r₁ r₂ d :=
by sorry

end NUMINAMATH_CALUDE_pulley_center_distance_l3295_329507


namespace NUMINAMATH_CALUDE_complement_of_37_45_l3295_329541

-- Define angle in degrees and minutes
structure AngleDM where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

-- Define the complement of an angle
def complement (α : AngleDM) : AngleDM :=
  let totalMinutes := (90 * 60) - (α.degrees * 60 + α.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60, by sorry⟩

theorem complement_of_37_45 :
  let α : AngleDM := ⟨37, 45, by sorry⟩
  complement α = ⟨52, 15, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_37_45_l3295_329541


namespace NUMINAMATH_CALUDE_probability_of_specific_pair_l3295_329577

def total_items : ℕ := 4
def items_to_select : ℕ := 2
def favorable_outcomes : ℕ := 1

theorem probability_of_specific_pair :
  (favorable_outcomes : ℚ) / (total_items.choose items_to_select) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_pair_l3295_329577


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l3295_329584

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f 2 x ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f a x ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l3295_329584


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3295_329516

theorem max_value_of_expression (a b c : ℝ) 
  (ha : -1 < a ∧ a < 1) 
  (hb : -1 < b ∧ b < 1) 
  (hc : -1 < c ∧ c < 1) : 
  1/((1 - a^2)*(1 - b^2)*(1 - c^2)) + 1/((1 + a^2)*(1 + b^2)*(1 + c^2)) ≤ 2 ∧ 
  (1/((1 - 0^2)*(1 - 0^2)*(1 - 0^2)) + 1/((1 + 0^2)*(1 + 0^2)*(1 + 0^2)) = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3295_329516


namespace NUMINAMATH_CALUDE_nth_root_approximation_l3295_329560

/-- Approximation of nth root of x₀ⁿ + Δx --/
theorem nth_root_approximation
  (n : ℕ) (x₀ Δx ε : ℝ) (h_x₀_pos : x₀ > 0) (h_Δx_small : |Δx| < x₀^n) :
  ∃ (ε : ℝ), ε > 0 ∧ 
  |((x₀^n + Δx)^(1/n : ℝ) : ℝ) - (x₀ + Δx / (n * x₀^(n-1)))| < ε :=
by sorry

end NUMINAMATH_CALUDE_nth_root_approximation_l3295_329560


namespace NUMINAMATH_CALUDE_middle_school_enrollment_l3295_329578

theorem middle_school_enrollment (band_percentage : Real) (sports_percentage : Real)
  (band_count : Nat) (sports_count : Nat)
  (h1 : band_percentage = 0.20)
  (h2 : sports_percentage = 0.30)
  (h3 : band_count = 168)
  (h4 : sports_count = 252) :
  ∃ (total : Nat), (band_count : Real) / band_percentage = total ∧
                   (sports_count : Real) / sports_percentage = total ∧
                   total = 840 := by
  sorry

end NUMINAMATH_CALUDE_middle_school_enrollment_l3295_329578


namespace NUMINAMATH_CALUDE_rebecca_bought_four_tent_stakes_l3295_329564

/-- The number of tent stakes bought by Rebecca. -/
def tent_stakes : ℕ := sorry

/-- The number of packets of drink mix bought by Rebecca. -/
def drink_mix : ℕ := sorry

/-- The number of bottles of water bought by Rebecca. -/
def water_bottles : ℕ := sorry

/-- The total number of items bought by Rebecca. -/
def total_items : ℕ := 22

/-- Theorem stating that Rebecca bought 4 tent stakes. -/
theorem rebecca_bought_four_tent_stakes :
  (drink_mix = 3 * tent_stakes) ∧
  (water_bottles = tent_stakes + 2) ∧
  (tent_stakes + drink_mix + water_bottles = total_items) →
  tent_stakes = 4 := by
sorry

end NUMINAMATH_CALUDE_rebecca_bought_four_tent_stakes_l3295_329564


namespace NUMINAMATH_CALUDE_hardcover_nonfiction_count_l3295_329583

/-- Represents the number of books of each type --/
structure BookCounts where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ
  hardcoverFiction : ℕ

/-- The total number of books --/
def totalBooks : ℕ := 10000

/-- Conditions for the book counts --/
def validBookCounts (bc : BookCounts) : Prop :=
  bc.paperbackFiction + bc.paperbackNonfiction + bc.hardcoverNonfiction + bc.hardcoverFiction = totalBooks ∧
  bc.paperbackNonfiction = bc.hardcoverNonfiction + 100 ∧
  bc.paperbackFiction * 3 = bc.hardcoverFiction * 5 ∧
  bc.hardcoverFiction = totalBooks / 100 * 12 ∧
  bc.paperbackNonfiction + bc.hardcoverNonfiction = totalBooks / 100 * 30

theorem hardcover_nonfiction_count (bc : BookCounts) (h : validBookCounts bc) : 
  bc.hardcoverNonfiction = 1450 := by
  sorry

end NUMINAMATH_CALUDE_hardcover_nonfiction_count_l3295_329583


namespace NUMINAMATH_CALUDE_chocolate_lollipop_cost_equivalence_l3295_329537

/-- Proves that the cost of one pack of chocolate equals the cost of 4 lollipops -/
theorem chocolate_lollipop_cost_equivalence 
  (lollipop_count : ℕ) 
  (chocolate_pack_count : ℕ)
  (lollipop_cost : ℕ)
  (bills_given : ℕ)
  (bill_value : ℕ)
  (change_received : ℕ)
  (h1 : lollipop_count = 4)
  (h2 : chocolate_pack_count = 6)
  (h3 : lollipop_cost = 2)
  (h4 : bills_given = 6)
  (h5 : bill_value = 10)
  (h6 : change_received = 4) :
  (bills_given * bill_value - change_received - lollipop_count * lollipop_cost) / chocolate_pack_count = 4 * lollipop_cost :=
by sorry

end NUMINAMATH_CALUDE_chocolate_lollipop_cost_equivalence_l3295_329537


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l3295_329555

theorem boys_to_girls_ratio (total : ℕ) (diff : ℕ) : 
  total = 36 → 
  diff = 6 → 
  ∃ (boys girls : ℕ), 
    boys = girls + diff ∧ 
    boys + girls = total ∧ 
    boys * 5 = girls * 7 := by
sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l3295_329555


namespace NUMINAMATH_CALUDE_cos_315_degrees_l3295_329569

theorem cos_315_degrees :
  Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l3295_329569


namespace NUMINAMATH_CALUDE_sum_of_q_p_equals_negative_twenty_l3295_329594

def p (x : ℝ) : ℝ := x^2 - 4

def q (x : ℝ) : ℝ := -abs x

def xValues : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_q_p_equals_negative_twenty :
  (xValues.map (λ x => q (p x))).sum = -20 := by sorry

end NUMINAMATH_CALUDE_sum_of_q_p_equals_negative_twenty_l3295_329594


namespace NUMINAMATH_CALUDE_lcm_45_75_l3295_329545

theorem lcm_45_75 : Nat.lcm 45 75 = 225 := by sorry

end NUMINAMATH_CALUDE_lcm_45_75_l3295_329545


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l3295_329548

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, n > 0 ∧ 
  (∃ m : ℕ, n = m^2) ∧
  2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧
  (∀ k : ℕ, k > 0 → (∃ l : ℕ, k = l^2) → 2 ∣ k → 3 ∣ k → 5 ∣ k → k ≥ n) ∧
  n = 900 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l3295_329548


namespace NUMINAMATH_CALUDE_mean_equality_implies_z_value_l3295_329515

theorem mean_equality_implies_z_value : ∃ z : ℚ, 
  (8 + 15 + 27) / 3 = (18 + z) / 2 → z = 46 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_z_value_l3295_329515


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3295_329587

/-- If the terminal side of angle α passes through point (-1, 2), 
    then tan(α + π/4) = -1/3 -/
theorem tan_alpha_plus_pi_fourth (α : ℝ) :
  (∃ (t : ℝ), t > 0 ∧ t * Real.cos α = -1 ∧ t * Real.sin α = 2) →
  Real.tan (α + π/4) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l3295_329587


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_product_l3295_329528

theorem largest_divisor_of_consecutive_even_product : 
  ∃ (d : ℕ), d = 24 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (2*n) * (2*n + 2) * (2*n + 4)) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (2*m) * (2*m + 2) * (2*m + 4))) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_product_l3295_329528


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_l3295_329591

def p (x : ℝ) : ℝ := 3*x^3 + 2*x^2 + 5*x + 4
def q (x : ℝ) : ℝ := 7*x^3 + 5*x^2 + 6*x + 7

theorem coefficient_x_cubed (x : ℝ) : 
  ∃ (a b c d : ℝ), p x * q x = 38*x^3 + a*x^4 + b*x^2 + c*x + d :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_l3295_329591


namespace NUMINAMATH_CALUDE_complex_calculation_l3295_329595

theorem complex_calculation (A M S : ℂ) (P : ℝ) 
  (hA : A = 5 - 2*I) 
  (hM : M = -3 + 2*I) 
  (hS : S = 2*I) 
  (hP : P = 3) : 
  2 * (A - M + S - P) = 10 - 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_calculation_l3295_329595


namespace NUMINAMATH_CALUDE_vector_operation_l3295_329512

def vector_a : ℝ × ℝ × ℝ := (2, 0, -1)
def vector_b : ℝ × ℝ × ℝ := (0, 1, -2)

theorem vector_operation :
  (2 : ℝ) • vector_a - vector_b = (4, -1, 0) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3295_329512


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3295_329523

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3295_329523


namespace NUMINAMATH_CALUDE_balloon_distribution_l3295_329531

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 215) (h2 : num_friends = 9) :
  total_balloons % num_friends = 8 := by
sorry

end NUMINAMATH_CALUDE_balloon_distribution_l3295_329531


namespace NUMINAMATH_CALUDE_total_bones_l3295_329568

/-- The number of bones Xiao Qi has -/
def xiao_qi_bones : ℕ := sorry

/-- The number of bones Xiao Shi has -/
def xiao_shi_bones : ℕ := sorry

/-- The number of bones Xiao Ha has -/
def xiao_ha_bones : ℕ := sorry

/-- Xiao Ha has 2 more bones than twice the number of bones Xiao Shi has -/
axiom ha_shi_relation : xiao_ha_bones = 2 * xiao_shi_bones + 2

/-- Xiao Shi has 3 more bones than three times the number of bones Xiao Qi has -/
axiom shi_qi_relation : xiao_shi_bones = 3 * xiao_qi_bones + 3

/-- Xiao Ha has 5 fewer bones than seven times the number of bones Xiao Qi has -/
axiom ha_qi_relation : xiao_ha_bones = 7 * xiao_qi_bones - 5

/-- The total number of bones is 141 -/
theorem total_bones :
  xiao_qi_bones + xiao_shi_bones + xiao_ha_bones = 141 :=
sorry

end NUMINAMATH_CALUDE_total_bones_l3295_329568


namespace NUMINAMATH_CALUDE_six_by_six_untileable_large_rectangle_tileable_six_by_eight_tileable_l3295_329559

/-- A domino is a 1x2 tile -/
structure Domino :=
  (length : Nat := 2)
  (width : Nat := 1)

/-- A rectangle with dimensions m and n -/
structure Rectangle (m n : Nat) where
  mk ::

/-- A tiling of a rectangle with dominoes -/
def Tiling (m n : Nat) := List Domino

/-- A seam is a straight line not cutting through any dominoes -/
def HasSeam (t : Tiling m n) : Prop := sorry

/-- Theorem: A 6x6 square cannot be tiled with dominoes without a seam -/
theorem six_by_six_untileable : 
  ∀ (t : Tiling 6 6), HasSeam t := sorry

/-- Theorem: Any m×n rectangle where m, n > 6 and mn is even can be tiled without a seam -/
theorem large_rectangle_tileable (m n : Nat) 
  (hm : m > 6) (hn : n > 6) (h_even : Even (m * n)) : 
  ∃ (t : Tiling m n), ¬HasSeam t := sorry

/-- Theorem: A 6x8 rectangle can be tiled without a seam -/
theorem six_by_eight_tileable : 
  ∃ (t : Tiling 6 8), ¬HasSeam t := sorry

end NUMINAMATH_CALUDE_six_by_six_untileable_large_rectangle_tileable_six_by_eight_tileable_l3295_329559


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l3295_329589

/-- An equilateral triangle with perimeter 69 cm has sides of length 23 cm -/
theorem equilateral_triangle_side_length (triangle : Set ℝ) (perimeter : ℝ) :
  perimeter = 69 →
  ∃ (side_length : ℝ), side_length * 3 = perimeter ∧ side_length = 23 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l3295_329589


namespace NUMINAMATH_CALUDE_non_adjacent_book_selection_l3295_329586

/-- The number of books on the shelf -/
def total_books : ℕ := 12

/-- The number of books to be chosen -/
def books_to_choose : ℕ := 5

/-- The theorem stating that the number of ways to choose 5 books out of 12
    such that no two chosen books are adjacent is equal to C(8,5) -/
theorem non_adjacent_book_selection :
  (Nat.choose (total_books - books_to_choose + 1) books_to_choose) =
  (Nat.choose 8 5) := by sorry

end NUMINAMATH_CALUDE_non_adjacent_book_selection_l3295_329586


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l3295_329596

theorem inscribed_cube_surface_area :
  ∀ (outer_cube_surface_area : ℝ) (inner_cube_surface_area : ℝ),
    outer_cube_surface_area = 54 →
    (∃ (outer_cube_side : ℝ) (sphere_diameter : ℝ) (inner_cube_diagonal : ℝ) (inner_cube_side : ℝ),
      outer_cube_surface_area = 6 * outer_cube_side^2 ∧
      sphere_diameter = outer_cube_side ∧
      inner_cube_diagonal = sphere_diameter ∧
      inner_cube_diagonal = inner_cube_side * Real.sqrt 3 ∧
      inner_cube_surface_area = 6 * inner_cube_side^2) →
    inner_cube_surface_area = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l3295_329596


namespace NUMINAMATH_CALUDE_tangent_four_implies_expression_l3295_329579

theorem tangent_four_implies_expression (α : Real) (h : Real.tan α = 4) :
  (1 + Real.cos (2 * α) + 8 * Real.sin α ^ 2) / Real.sin (2 * α) = 65 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_four_implies_expression_l3295_329579


namespace NUMINAMATH_CALUDE_line_slope_k_l3295_329538

/-- Given a line passing through the points (-1, -4) and (4, k) with slope k, prove that k = 1 -/
theorem line_slope_k (k : ℝ) : 
  (k - (-4)) / (4 - (-1)) = k → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_k_l3295_329538


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3295_329509

def U : Finset ℕ := {1, 3, 5, 7, 9}
def A : Finset ℕ := {1, 5, 7}

theorem complement_of_A_in_U :
  U \ A = {3, 9} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3295_329509


namespace NUMINAMATH_CALUDE_rectangle_perimeter_theorem_l3295_329588

theorem rectangle_perimeter_theorem (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a * b > 2 * (a + b) → 2 * (a + b) > 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_theorem_l3295_329588


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3295_329536

theorem fraction_equivalence : ∃ n : ℚ, (4 + n) / (7 + n) = 7 / 9 :=
by
  use 13 / 2
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3295_329536


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l3295_329504

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cosine (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 8 * Real.pi →
  Real.cos (a 3 + a 7) = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l3295_329504


namespace NUMINAMATH_CALUDE_largest_value_is_B_l3295_329506

theorem largest_value_is_B (a b c e : ℚ) : 
  a = (1/2) / (3/4) →
  b = 1 / ((2/3) / 4) →
  c = ((1/2) / 3) / 4 →
  e = (1 / (2/3)) / 4 →
  b > a ∧ b > c ∧ b > e :=
by sorry

end NUMINAMATH_CALUDE_largest_value_is_B_l3295_329506


namespace NUMINAMATH_CALUDE_xy_xz_yz_bounds_l3295_329597

open Real

theorem xy_xz_yz_bounds (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  (∃ N n : ℝ, (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 → a * b + a * c + b * c ≤ N) ∧
              (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 → n ≤ a * b + a * c + b * c) ∧
              N = 75 ∧ n = 0) := by
  sorry

#check xy_xz_yz_bounds

end NUMINAMATH_CALUDE_xy_xz_yz_bounds_l3295_329597


namespace NUMINAMATH_CALUDE_f_properties_l3295_329570

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_properties :
  ∃ (m : ℝ),
    (∀ (a x₁ x₂ : ℝ), x₁ < x₂ → f a x₁ < f a x₂) ∧
    (∀ (x : ℝ), f 1 x = -f 1 (-x)) ∧
    (m = 12/5 ∧ ∀ (x : ℝ), x ∈ Set.Icc 2 3 → f 1 x ≥ m / 2^x) ∧
    (∀ (m' : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 2 3 → f 1 x ≥ m' / 2^x) → m' ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3295_329570


namespace NUMINAMATH_CALUDE_largest_n_value_largest_n_achievable_l3295_329547

/-- Represents a digit in base 5 -/
def Base5Digit := Fin 5

/-- Represents a digit in base 9 -/
def Base9Digit := Fin 9

/-- Converts a number from base 5 to base 10 -/
def fromBase5 (a b c : Base5Digit) : ℕ :=
  25 * a.val + 5 * b.val + c.val

/-- Converts a number from base 9 to base 10 -/
def fromBase9 (c b a : Base9Digit) : ℕ :=
  81 * c.val + 9 * b.val + a.val

theorem largest_n_value (n : ℕ) 
  (h1 : ∃ (a b c : Base5Digit), n = fromBase5 a b c)
  (h2 : ∃ (a b c : Base9Digit), n = fromBase9 c b a) :
  n ≤ 111 := by
  sorry

theorem largest_n_achievable : 
  ∃ (n : ℕ) (a b c : Base5Digit) (x y z : Base9Digit),
    n = fromBase5 a b c ∧ 
    n = fromBase9 z y x ∧ 
    n = 111 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_value_largest_n_achievable_l3295_329547


namespace NUMINAMATH_CALUDE_no_5_6_8_multiplier_l3295_329552

/-- Function to get the number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + num_digits (n / 10)

/-- Function to get the leading digit of a positive integer -/
def leading_digit (n : ℕ) : ℕ :=
  if n < 10 then n else leading_digit (n / 10)

/-- Function to move the leading digit to the end -/
def move_leading_digit (n : ℕ) : ℕ :=
  let d := num_digits n
  let lead := leading_digit n
  (n - lead * 10^(d-1)) * 10 + lead

/-- Theorem stating that no integer becomes 5, 6, or 8 times larger when its leading digit is moved to the end -/
theorem no_5_6_8_multiplier (n : ℕ) (h : n ≥ 10) : 
  let m := move_leading_digit n
  m ≠ 5*n ∧ m ≠ 6*n ∧ m ≠ 8*n :=
sorry

end NUMINAMATH_CALUDE_no_5_6_8_multiplier_l3295_329552


namespace NUMINAMATH_CALUDE_thirteen_in_binary_l3295_329534

theorem thirteen_in_binary : 
  (13 : ℕ).digits 2 = [1, 0, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_thirteen_in_binary_l3295_329534


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l3295_329573

theorem no_solution_fractional_equation :
  ∀ x : ℝ, (((1 - x) / (x - 2)) + 2 ≠ 1 / (2 - x)) ∨ (x = 2) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l3295_329573


namespace NUMINAMATH_CALUDE_marks_score_l3295_329557

theorem marks_score (highest_score : ℕ) (score_range : ℕ) (marks_score : ℕ) :
  highest_score = 98 →
  score_range = 75 →
  marks_score = 2 * (highest_score - score_range) →
  marks_score = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_marks_score_l3295_329557


namespace NUMINAMATH_CALUDE_cube_root_inequality_l3295_329556

theorem cube_root_inequality (x : ℝ) (h : x > 0) :
  Real.rpow x (1/3) < 3 - x ↔ x < 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l3295_329556


namespace NUMINAMATH_CALUDE_product_of_primes_in_equation_l3295_329524

theorem product_of_primes_in_equation (p q : ℕ) : 
  Prime p → Prime q → p * 1 + q = 99 → p * q = 194 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_in_equation_l3295_329524


namespace NUMINAMATH_CALUDE_power_product_equals_four_l3295_329576

theorem power_product_equals_four : 4^2020 * (1/4)^2019 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_four_l3295_329576


namespace NUMINAMATH_CALUDE_distance_to_larger_section_l3295_329514

/-- Given a right hexagonal pyramid with two parallel cross sections -/
structure HexagonalPyramid where
  /-- Ratio of areas of two parallel cross sections -/
  area_ratio : ℝ
  /-- Distance between the two parallel cross sections -/
  distance_between_sections : ℝ

/-- Theorem stating the distance from apex to larger cross section -/
theorem distance_to_larger_section (pyramid : HexagonalPyramid)
  (h_area_ratio : pyramid.area_ratio = 4 / 9)
  (h_distance : pyramid.distance_between_sections = 12) :
  ∃ (d : ℝ), d = 36 ∧ d > 0 ∧ 
  d = (pyramid.distance_between_sections * 3) / (1 - (pyramid.area_ratio)^(1/2)) :=
sorry

end NUMINAMATH_CALUDE_distance_to_larger_section_l3295_329514


namespace NUMINAMATH_CALUDE_fraction_sum_equation_l3295_329542

theorem fraction_sum_equation (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (a + 5 * b) / (b + 5 * a) = 3) : a / b = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equation_l3295_329542


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_85_l3295_329520

theorem largest_multiple_of_seven_below_negative_85 :
  ∀ n : ℤ, n % 7 = 0 ∧ n < -85 → n ≤ -91 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_below_negative_85_l3295_329520


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3295_329532

/-- The trajectory of the midpoint of a line segment with one end fixed and the other on a circle -/
theorem midpoint_trajectory (m n x y : ℝ) : 
  (m + 1)^2 + n^2 = 4 →  -- B(m, n) is on the circle (x+1)^2 + y^2 = 4
  x = (m + 4) / 2 →      -- x-coordinate of midpoint M
  y = (n - 3) / 2 →      -- y-coordinate of midpoint M
  (x - 3/2)^2 + (y + 3/2)^2 = 1 := by
sorry


end NUMINAMATH_CALUDE_midpoint_trajectory_l3295_329532


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_equals_five_l3295_329599

theorem x_squared_minus_y_squared_equals_five 
  (x y : ℝ) 
  (h1 : 23 * x + 977 * y = 2023) 
  (h2 : 977 * x + 23 * y = 2977) : 
  x^2 - y^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_equals_five_l3295_329599


namespace NUMINAMATH_CALUDE_oarsmen_count_l3295_329522

theorem oarsmen_count (weight_old : ℝ) (weight_new : ℝ) (avg_increase : ℝ) :
  weight_old = 53 →
  weight_new = 71 →
  avg_increase = 1.8 →
  ∃ n : ℕ, n > 0 ∧ n * avg_increase = weight_new - weight_old :=
by
  sorry

end NUMINAMATH_CALUDE_oarsmen_count_l3295_329522


namespace NUMINAMATH_CALUDE_science_club_membership_l3295_329558

theorem science_club_membership (total : ℕ) (chem : ℕ) (bio : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : chem = 48)
  (h3 : bio = 40)
  (h4 : both = 25) :
  total - (chem + bio - both) = 17 := by
  sorry

end NUMINAMATH_CALUDE_science_club_membership_l3295_329558


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l3295_329562

def a : ℝ × ℝ × ℝ := (3, -2, 4)
def b : ℝ → ℝ → ℝ × ℝ × ℝ := λ x y ↦ (1, x, y)

theorem parallel_vectors_sum (x y : ℝ) :
  (∃ (k : ℝ), b x y = k • a) → x + y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l3295_329562


namespace NUMINAMATH_CALUDE_last_digit_3_count_l3295_329554

/-- The number of terms in the sequence 7^1, 7^2, ..., 7^2008 that have a last digit of 3 -/
def count_last_digit_3 : ℕ := 502

/-- The length of the sequence 7^1, 7^2, ..., 7^2008 -/
def sequence_length : ℕ := 2008

theorem last_digit_3_count :
  count_last_digit_3 = sequence_length / 4 :=
sorry

end NUMINAMATH_CALUDE_last_digit_3_count_l3295_329554


namespace NUMINAMATH_CALUDE_binomial_10_choose_5_l3295_329553

theorem binomial_10_choose_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_5_l3295_329553


namespace NUMINAMATH_CALUDE_derivative_zero_in_interval_l3295_329530

theorem derivative_zero_in_interval (n : ℕ) (f : ℝ → ℝ) 
  (h_diff : ContDiff ℝ (n + 1) f)
  (h_f_zero : f 1 = 0 ∧ f 0 = 0)
  (h_derivatives_zero : ∀ k : ℕ, k ≤ n → (deriv^[k] f) 0 = 0) :
  ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ (deriv^[n + 1] f) x = 0 := by
sorry

end NUMINAMATH_CALUDE_derivative_zero_in_interval_l3295_329530


namespace NUMINAMATH_CALUDE_ABC_collinear_l3295_329574

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Three points in the plane -/
def A : Point := ⟨-1, 4⟩
def B : Point := ⟨-3, 2⟩
def C : Point := ⟨0, 5⟩

/-- Definition of collinearity for three points -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Theorem: Points A, B, and C are collinear -/
theorem ABC_collinear : collinear A B C := by
  sorry

end NUMINAMATH_CALUDE_ABC_collinear_l3295_329574


namespace NUMINAMATH_CALUDE_negation_equivalence_l3295_329575

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3295_329575


namespace NUMINAMATH_CALUDE_jogger_distance_ahead_l3295_329529

/-- Proves that a jogger is 270 meters ahead of a train given specific conditions -/
theorem jogger_distance_ahead (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (5 / 18) →  -- Convert 9 km/hr to m/s
  train_speed = 45 * (5 / 18) →  -- Convert 45 km/hr to m/s
  train_length = 120 →
  passing_time = 39 →
  (train_speed - jogger_speed) * passing_time = train_length + 270 :=
by sorry

end NUMINAMATH_CALUDE_jogger_distance_ahead_l3295_329529


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l3295_329592

theorem quadratic_inequality_no_solution :
  ∀ x : ℝ, 2 * x^2 - 3 * x + 4 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l3295_329592


namespace NUMINAMATH_CALUDE_correct_formula_l3295_329571

def f (x : ℝ) : ℝ := 5 * x^2 + x

theorem correct_formula : 
  (f 0 = 0) ∧ 
  (f 1 = 20) ∧ 
  (f 2 = 60) ∧ 
  (f 3 = 120) ∧ 
  (f 4 = 200) := by
  sorry

end NUMINAMATH_CALUDE_correct_formula_l3295_329571


namespace NUMINAMATH_CALUDE_group_arrangement_count_l3295_329539

theorem group_arrangement_count :
  let total_men : ℕ := 4
  let total_women : ℕ := 5
  let small_group_size : ℕ := 2
  let large_group_size : ℕ := 5
  let small_group_count : ℕ := 2
  let large_group_count : ℕ := 1
  let total_people : ℕ := total_men + total_women
  let total_groups : ℕ := small_group_count + large_group_count

  -- Condition: At least one man and one woman in each group
  ∀ (men_in_small_group men_in_large_group : ℕ),
    (men_in_small_group ≥ 1 ∧ men_in_small_group < small_group_size) →
    (men_in_large_group ≥ 1 ∧ men_in_large_group < large_group_size) →
    (men_in_small_group * small_group_count + men_in_large_group * large_group_count = total_men) →

  -- The number of ways to arrange the groups
  (Nat.choose total_men 2 * Nat.choose total_women 3 +
   Nat.choose total_men 3 * Nat.choose total_women 2) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_group_arrangement_count_l3295_329539


namespace NUMINAMATH_CALUDE_lukes_father_twenty_bills_l3295_329563

def mother_fifty : ℕ := 1
def mother_twenty : ℕ := 2
def mother_ten : ℕ := 3

def father_fifty : ℕ := 4
def father_ten : ℕ := 1

def school_fee : ℕ := 350

theorem lukes_father_twenty_bills :
  ∃ (father_twenty : ℕ),
    50 * mother_fifty + 20 * mother_twenty + 10 * mother_ten +
    50 * father_fifty + 20 * father_twenty + 10 * father_ten = school_fee ∧
    father_twenty = 1 :=
by sorry

end NUMINAMATH_CALUDE_lukes_father_twenty_bills_l3295_329563


namespace NUMINAMATH_CALUDE_inequality_proof_l3295_329500

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a * b + b * c + c * a = 1) :
  a / b + b / c + c / a ≥ a^2 + b^2 + c^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3295_329500


namespace NUMINAMATH_CALUDE_cubic_inequality_l3295_329598

theorem cubic_inequality (x : ℝ) :
  x ≥ 0 → (x^3 - 9*x^2 - 16*x > 0 ↔ x > 16) := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3295_329598


namespace NUMINAMATH_CALUDE_statements_with_nonzero_solutions_l3295_329527

theorem statements_with_nonzero_solutions :
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (Real.sqrt (a^2 + b^2) = 3 * (a * b) ∨ Real.sqrt (a^2 + b^2) = 2 * (a * b)) ∧
  ¬(∃ (c d : ℝ), c ≠ 0 ∧ d ≠ 0 ∧
    (Real.sqrt (c^2 + d^2) = 2 * (c + d) ∨ Real.sqrt (c^2 + d^2) = (1/2) * (c + d))) :=
by
  sorry


end NUMINAMATH_CALUDE_statements_with_nonzero_solutions_l3295_329527


namespace NUMINAMATH_CALUDE_divisibility_by_11_l3295_329550

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def is_valid_digit (d : ℕ) : Prop :=
  d ≥ 0 ∧ d ≤ 9

def five_digit_number (a b : ℕ) : ℕ :=
  40000 + a * 1000 + b * 100 + 20 + b

theorem divisibility_by_11 (a b : ℕ) 
  (h1 : is_valid_digit a) 
  (h2 : is_valid_digit b) 
  (h3 : is_divisible_by_11 (five_digit_number a b)) : 
  a = 6 ∧ is_valid_digit b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_11_l3295_329550


namespace NUMINAMATH_CALUDE_a_range_when_f_decreasing_l3295_329581

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Define the property of f being decreasing on (-∞, 6)
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → x < 6 → y < 6 → f a x > f a y

-- State the theorem
theorem a_range_when_f_decreasing (a : ℝ) :
  is_decreasing_on_interval a → a ∈ Set.Ici 6 :=
by
  sorry

#check a_range_when_f_decreasing

end NUMINAMATH_CALUDE_a_range_when_f_decreasing_l3295_329581


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l3295_329503

/-- The quadratic function f(x) = x^2 + ax + 6 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 6

theorem quadratic_inequality_solutions (a : ℝ) :
  (a = 5 → {x : ℝ | f 5 x < 0} = {x : ℝ | -3 < x ∧ x < -2}) ∧
  ({x : ℝ | f a x > 0} = Set.univ → a ∈ Set.Ioo (-2*Real.sqrt 6) (2*Real.sqrt 6)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l3295_329503


namespace NUMINAMATH_CALUDE_rhombus_area_l3295_329540

/-- The area of a rhombus with side length 25 and one diagonal 30 is 600 -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) (area : ℝ) : 
  side = 25 → 
  diagonal1 = 30 → 
  diagonal2^2 = 4 * (side^2 - (diagonal1/2)^2) → 
  area = (diagonal1 * diagonal2) / 2 → 
  area = 600 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3295_329540


namespace NUMINAMATH_CALUDE_logical_equivalence_l3295_329533

theorem logical_equivalence (P Q : Prop) :
  (¬P → Q) ↔ (¬Q → P) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l3295_329533


namespace NUMINAMATH_CALUDE_first_cat_weight_l3295_329543

theorem first_cat_weight (total_weight second_cat_weight third_cat_weight : ℕ) 
  (h1 : total_weight = 13)
  (h2 : second_cat_weight = 7)
  (h3 : third_cat_weight = 4)
  : total_weight - second_cat_weight - third_cat_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_cat_weight_l3295_329543


namespace NUMINAMATH_CALUDE_part_one_part_two_l3295_329565

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 2| - |x + 2|

-- Part 1
theorem part_one : 
  {x : ℝ | f 2 x ≤ 1} = {x : ℝ | -1/3 ≤ x ∧ x ≤ 5} :=
sorry

-- Part 2
theorem part_two : 
  (∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4) → (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3295_329565


namespace NUMINAMATH_CALUDE_cos_180_deg_l3295_329525

-- Define cosine function for angles in degrees
noncomputable def cos_deg (θ : ℝ) : ℝ := 
  Real.cos (θ * Real.pi / 180)

-- Theorem statement
theorem cos_180_deg : cos_deg 180 = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_deg_l3295_329525


namespace NUMINAMATH_CALUDE_student_tickets_sold_l3295_329567

/-- Proves that the number of student tickets sold is 9 given the specified conditions -/
theorem student_tickets_sold (adult_price : ℝ) (student_price : ℝ) (total_tickets : ℕ) (total_revenue : ℝ)
  (h1 : adult_price = 4)
  (h2 : student_price = 2.5)
  (h3 : total_tickets = 59)
  (h4 : total_revenue = 222.5) :
  ∃ (student_tickets : ℕ), 
    student_tickets = 9 ∧
    (total_tickets - student_tickets : ℝ) * adult_price + (student_tickets : ℝ) * student_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_student_tickets_sold_l3295_329567


namespace NUMINAMATH_CALUDE_correct_divisor_l3295_329518

theorem correct_divisor (X D : ℕ) (h1 : X % D = 0) (h2 : X = 49 * 12) (h3 : X = 28 * D) : D = 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l3295_329518


namespace NUMINAMATH_CALUDE_obtuse_angle_range_l3295_329546

/-- The angle between two 2D vectors is obtuse if and only if their dot product is negative -/
def is_obtuse_angle (a b : Fin 2 → ℝ) : Prop :=
  (a 0 * b 0 + a 1 * b 1) < 0

/-- The set of real numbers x for which the angle between (1, 3) and (x, -1) is obtuse -/
def obtuse_angle_set : Set ℝ :=
  {x : ℝ | is_obtuse_angle (![1, 3]) (![x, -1])}

theorem obtuse_angle_range :
  obtuse_angle_set = {x : ℝ | x < -1/3 ∨ (-1/3 < x ∧ x < 3)} := by sorry

end NUMINAMATH_CALUDE_obtuse_angle_range_l3295_329546


namespace NUMINAMATH_CALUDE_average_pieces_lost_is_13_4_l3295_329549

/-- The number of games played -/
def num_games : ℕ := 5

/-- Audrey's lost pieces in each game -/
def audrey_lost : List ℕ := [6, 8, 4, 7, 10]

/-- Thomas's lost pieces in each game -/
def thomas_lost : List ℕ := [5, 6, 3, 7, 11]

/-- The average number of pieces lost per game for both players combined -/
def average_pieces_lost : ℚ :=
  (audrey_lost.sum + thomas_lost.sum : ℚ) / num_games

theorem average_pieces_lost_is_13_4 :
  average_pieces_lost = 134/10 := by sorry

end NUMINAMATH_CALUDE_average_pieces_lost_is_13_4_l3295_329549


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3295_329585

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {y | ∃ x ∈ M, y = x^2}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3295_329585
