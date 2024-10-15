import Mathlib

namespace NUMINAMATH_CALUDE_rivertown_puzzle_l518_51855

theorem rivertown_puzzle (p h s c d : ℕ) : 
  p = 4 * h →
  s = 5 * c →
  d = 4 * p →
  ¬ ∃ (h c : ℕ), 99 = 21 * h + 6 * c :=
by sorry

end NUMINAMATH_CALUDE_rivertown_puzzle_l518_51855


namespace NUMINAMATH_CALUDE_four_digit_difference_l518_51834

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ n / 1000 = 7

def reverse_last_three_digits (n : ℕ) : ℕ :=
  let a := (n / 100) % 10
  let b := (n / 10) % 10
  let c := n % 10
  1000 * c + 100 * b + 10 * a + 7

theorem four_digit_difference (n : ℕ) : 
  is_valid_number n → n = reverse_last_three_digits n + 3546 → 
  n = 7053 ∨ n = 7163 ∨ n = 7273 ∨ n = 7383 ∨ n = 7493 :=
sorry

end NUMINAMATH_CALUDE_four_digit_difference_l518_51834


namespace NUMINAMATH_CALUDE_floor_equation_solution_l518_51805

theorem floor_equation_solution (x : ℝ) : 
  (⌊(5 + 6*x) / 8⌋ : ℝ) = (15*x - 7) / 5 ↔ x = 7/15 ∨ x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l518_51805


namespace NUMINAMATH_CALUDE_function_properties_l518_51872

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + Real.cos x + a

theorem function_properties (a : ℝ) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f a (x + T) = f a x ∧ 
   ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f a (x + T') = f a x) → T ≤ T') ∧
  (∃ (M : ℝ), M = 3 ∧ (∀ (x : ℝ), f a x ≤ M) → a = 1) ∧
  (∀ (k : ℤ), ∀ (x y : ℝ), 
    2 * k * Real.pi - 2 * Real.pi / 3 ≤ x ∧ 
    x < y ∧ 
    y ≤ 2 * k * Real.pi + Real.pi / 3 → 
    f a x < f a y) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l518_51872


namespace NUMINAMATH_CALUDE_expression_bounds_bounds_are_tight_l518_51827

theorem expression_bounds (p q r s : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) (hr : 0 ≤ r ∧ r ≤ 1) (hs : 0 ≤ s ∧ s ≤ 1) : 
  2 * Real.sqrt 2 ≤ Real.sqrt (p^2 + (1-q)^2) + Real.sqrt (q^2 + (1-r)^2) + 
    Real.sqrt (r^2 + (1-s)^2) + Real.sqrt (s^2 + (1-p)^2) ∧
  Real.sqrt (p^2 + (1-q)^2) + Real.sqrt (q^2 + (1-r)^2) + 
    Real.sqrt (r^2 + (1-s)^2) + Real.sqrt (s^2 + (1-p)^2) ≤ 4 :=
by sorry

theorem bounds_are_tight : 
  ∃ (p q r s : ℝ), (0 ≤ p ∧ p ≤ 1) ∧ (0 ≤ q ∧ q ≤ 1) ∧ (0 ≤ r ∧ r ≤ 1) ∧ (0 ≤ s ∧ s ≤ 1) ∧
    Real.sqrt (p^2 + (1-q)^2) + Real.sqrt (q^2 + (1-r)^2) + 
    Real.sqrt (r^2 + (1-s)^2) + Real.sqrt (s^2 + (1-p)^2) = 2 * Real.sqrt 2 ∧
  ∃ (p q r s : ℝ), (0 ≤ p ∧ p ≤ 1) ∧ (0 ≤ q ∧ q ≤ 1) ∧ (0 ≤ r ∧ r ≤ 1) ∧ (0 ≤ s ∧ s ≤ 1) ∧
    Real.sqrt (p^2 + (1-q)^2) + Real.sqrt (q^2 + (1-r)^2) + 
    Real.sqrt (r^2 + (1-s)^2) + Real.sqrt (s^2 + (1-p)^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_bounds_are_tight_l518_51827


namespace NUMINAMATH_CALUDE_quadratic_solution_with_nested_root_l518_51887

theorem quadratic_solution_with_nested_root (a b : ℤ) :
  (∃ x : ℝ, x^2 + a*x + b = 0 ∧ x = Real.sqrt (2010 + 2 * Real.sqrt 2009)) →
  a = 0 ∧ b = -2008 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_with_nested_root_l518_51887


namespace NUMINAMATH_CALUDE_markus_final_candies_l518_51877

theorem markus_final_candies 
  (markus_initial : ℕ) 
  (katharina_initial : ℕ) 
  (sanjiv_distribution : ℕ) 
  (h1 : markus_initial = 9)
  (h2 : katharina_initial = 5)
  (h3 : sanjiv_distribution = 10)
  (h4 : ∃ (x : ℕ), x + markus_initial + x + katharina_initial = markus_initial + katharina_initial + sanjiv_distribution) :
  ∃ (markus_final : ℕ), markus_final = 12 ∧ 2 * markus_final = markus_initial + katharina_initial + sanjiv_distribution :=
sorry

end NUMINAMATH_CALUDE_markus_final_candies_l518_51877


namespace NUMINAMATH_CALUDE_strawberry_picking_problem_l518_51821

/-- The strawberry picking problem -/
theorem strawberry_picking_problem 
  (betty_strawberries : ℕ)
  (matthew_strawberries : ℕ)
  (natalie_strawberries : ℕ)
  (strawberries_per_jar : ℕ)
  (price_per_jar : ℕ)
  (total_money_made : ℕ)
  (h1 : betty_strawberries = 16)
  (h2 : matthew_strawberries > betty_strawberries)
  (h3 : matthew_strawberries = 2 * natalie_strawberries)
  (h4 : strawberries_per_jar = 7)
  (h5 : price_per_jar = 4)
  (h6 : total_money_made = 40)
  (h7 : betty_strawberries + matthew_strawberries + natalie_strawberries = 
        (total_money_made / price_per_jar) * strawberries_per_jar) :
  matthew_strawberries - betty_strawberries = 20 := by
sorry

end NUMINAMATH_CALUDE_strawberry_picking_problem_l518_51821


namespace NUMINAMATH_CALUDE_perpendicular_tangent_line_l518_51858

/-- The curve y = x^3 -/
def f (x : ℝ) : ℝ := x^3

/-- The derivative of f -/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem perpendicular_tangent_line (a b : ℝ) :
  (∃ (x y : ℝ), a * x - b * y - 2 = 0) →  -- Given line exists
  f 1 = 1 →  -- Point (1,1) is on the curve
  (a / b) * (f' 1) = -1 →  -- Perpendicular condition
  b / a = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangent_line_l518_51858


namespace NUMINAMATH_CALUDE_fraction_value_when_a_equals_4b_l518_51818

theorem fraction_value_when_a_equals_4b (a b : ℝ) (h : a = 4 * b) :
  (a^2 + b^2) / (a * b) = 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_when_a_equals_4b_l518_51818


namespace NUMINAMATH_CALUDE_sum_equals_zero_l518_51838

theorem sum_equals_zero (a b c : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a * b + c^2 + 4 = 0) : 
  a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_zero_l518_51838


namespace NUMINAMATH_CALUDE_same_terminal_side_l518_51815

theorem same_terminal_side : ∀ (k : ℤ), 95 = -265 + k * 360 → 95 ≡ -265 [ZMOD 360] := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l518_51815


namespace NUMINAMATH_CALUDE_sin_alpha_value_l518_51812

theorem sin_alpha_value (α : Real) 
  (h1 : 2 * Real.tan α * Real.sin α = 3)
  (h2 : -π/2 < α ∧ α < 0) : 
  Real.sin α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l518_51812


namespace NUMINAMATH_CALUDE_root_sum_squares_l518_51820

theorem root_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 2*h*x = 3 ∧ y^2 + 2*h*y = 3 ∧ x^2 + y^2 = 10) → 
  |h| = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_squares_l518_51820


namespace NUMINAMATH_CALUDE_summer_program_sophomores_l518_51802

theorem summer_program_sophomores :
  ∀ (total_students : ℕ) 
    (non_soph_jun : ℕ)
    (soph_debate_ratio : ℚ)
    (jun_debate_ratio : ℚ),
  total_students = 40 →
  non_soph_jun = 5 →
  soph_debate_ratio = 1/5 →
  jun_debate_ratio = 1/4 →
  ∃ (sophomores juniors : ℚ),
    sophomores + juniors = total_students - non_soph_jun ∧
    sophomores * soph_debate_ratio = juniors * jun_debate_ratio ∧
    sophomores = 175/9 :=
by sorry

end NUMINAMATH_CALUDE_summer_program_sophomores_l518_51802


namespace NUMINAMATH_CALUDE_triangle_third_vertex_l518_51824

/-- Given a triangle with vertices at (8,5), (0,0), and (x,0) where x < 0,
    if the area of the triangle is 40 square units, then x = -16. -/
theorem triangle_third_vertex (x : ℝ) (h1 : x < 0) :
  (1/2 : ℝ) * abs (8 * 0 - x * 5) = 40 → x = -16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_vertex_l518_51824


namespace NUMINAMATH_CALUDE_sum_remainder_modulo_9_l518_51850

theorem sum_remainder_modulo_9 : 
  (8 + 77 + 666 + 5555 + 44444 + 333333 + 2222222 + 11111111) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_modulo_9_l518_51850


namespace NUMINAMATH_CALUDE_missing_number_is_33_l518_51879

def known_numbers : List ℝ := [1, 22, 24, 25, 26, 27, 2]

theorem missing_number_is_33 :
  ∃ x : ℝ, (known_numbers.sum + x) / 8 = 20 ∧ x = 33 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_is_33_l518_51879


namespace NUMINAMATH_CALUDE_sweater_selling_price_l518_51814

/-- The selling price of a sweater given the cost of materials and total gain -/
theorem sweater_selling_price 
  (balls_per_sweater : ℕ) 
  (cost_per_ball : ℕ) 
  (total_gain : ℕ) 
  (num_sweaters : ℕ) : 
  balls_per_sweater = 4 → 
  cost_per_ball = 6 → 
  total_gain = 308 → 
  num_sweaters = 28 → 
  (balls_per_sweater * cost_per_ball * num_sweaters + total_gain) / num_sweaters = 35 := by
  sorry

#check sweater_selling_price

end NUMINAMATH_CALUDE_sweater_selling_price_l518_51814


namespace NUMINAMATH_CALUDE_min_disks_is_fifteen_l518_51893

/-- Represents the storage problem with given file sizes and quantities --/
structure StorageProblem where
  total_files : Nat
  disk_capacity : Real
  files_09mb : Nat
  files_08mb : Nat
  files_05mb : Nat
  h_total : total_files = files_09mb + files_08mb + files_05mb
  h_capacity : disk_capacity = 2

/-- Calculates the minimum number of disks required for the given storage problem --/
def min_disks_required (problem : StorageProblem) : Nat :=
  sorry

/-- The main theorem stating that the minimum number of disks required is 15 --/
theorem min_disks_is_fifteen :
  ∀ (problem : StorageProblem),
    problem.total_files = 35 →
    problem.files_09mb = 4 →
    problem.files_08mb = 15 →
    problem.files_05mb = 16 →
    min_disks_required problem = 15 :=
  sorry

end NUMINAMATH_CALUDE_min_disks_is_fifteen_l518_51893


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l518_51830

theorem initial_markup_percentage 
  (C : ℝ) 
  (M : ℝ) 
  (h1 : C > 0) 
  (h2 : (C * (1 + M) * 1.25 * 0.75) = (C * 1.125)) : 
  M = 0.2 := by
sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l518_51830


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l518_51875

/-- The polynomial function we're considering -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

/-- Theorem stating that 1, -1, and 3 are the only roots of the polynomial -/
theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l518_51875


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l518_51859

theorem geometric_sequence_ratio_sum (k a₂ a₃ b₂ b₃ p r : ℝ) 
  (h1 : k ≠ 0)
  (h2 : p ≠ 1)
  (h3 : r ≠ 1)
  (h4 : p ≠ r)
  (h5 : a₂ = k * p)
  (h6 : a₃ = k * p^2)
  (h7 : b₂ = k * r)
  (h8 : b₃ = k * r^2)
  (h9 : a₃ - b₃ = 4 * (a₂ - b₂)) :
  p + r = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l518_51859


namespace NUMINAMATH_CALUDE_equation_solution_l518_51869

theorem equation_solution :
  ∃ x : ℝ, (x^2 - x + 2) / (x - 1) = x + 3 ∧ x = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l518_51869


namespace NUMINAMATH_CALUDE_kennel_cat_dog_ratio_l518_51854

theorem kennel_cat_dog_ratio :
  ∀ (num_dogs num_cats : ℕ),
    num_dogs = 32 →
    num_cats = num_dogs - 8 →
    (num_cats : ℚ) / (num_dogs : ℚ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_kennel_cat_dog_ratio_l518_51854


namespace NUMINAMATH_CALUDE_perfect_linear_correlation_l518_51866

/-- A scatter plot where all points lie on a straight line with non-zero slope -/
structure PerfectLinearScatterPlot where
  points : Set (ℝ × ℝ)
  non_zero_slope : ℝ
  line_equation : ℝ → ℝ
  all_points_on_line : ∀ (x y : ℝ), (x, y) ∈ points → y = line_equation x
  slope_non_zero : non_zero_slope ≠ 0

/-- The correlation coefficient of a scatter plot -/
def correlation_coefficient (plot : PerfectLinearScatterPlot) : ℝ :=
  sorry

/-- Theorem: The correlation coefficient of a perfect linear scatter plot is 1 -/
theorem perfect_linear_correlation (plot : PerfectLinearScatterPlot) :
  correlation_coefficient plot = 1 :=
sorry

end NUMINAMATH_CALUDE_perfect_linear_correlation_l518_51866


namespace NUMINAMATH_CALUDE_smallest_n_congruent_to_neg_2023_mod_9_l518_51829

theorem smallest_n_congruent_to_neg_2023_mod_9 : 
  ∃ n : ℕ, 
    (4 ≤ n ∧ n ≤ 12) ∧ 
    n ≡ -2023 [ZMOD 9] ∧
    (∀ m : ℕ, (4 ≤ m ∧ m ≤ 12) ∧ m ≡ -2023 [ZMOD 9] → n ≤ m) ∧
    n = 11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruent_to_neg_2023_mod_9_l518_51829


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l518_51849

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (∀ θ : ℝ, θ = 150 ∧ θ = (n - 2 : ℝ) * 180 / n) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l518_51849


namespace NUMINAMATH_CALUDE_work_equivalence_first_group_size_correct_l518_51894

/-- The number of hours it takes the first group to complete the work -/
def first_group_hours : ℕ := 20

/-- The number of men in the second group -/
def second_group_men : ℕ := 15

/-- The number of hours it takes the second group to complete the work -/
def second_group_hours : ℕ := 48

/-- The number of men in the first group -/
def first_group_men : ℕ := 36

theorem work_equivalence :
  first_group_men * first_group_hours = second_group_men * second_group_hours :=
by sorry

/-- Proves that the number of men in the first group is correct -/
theorem first_group_size_correct :
  first_group_men = (second_group_men * second_group_hours) / first_group_hours :=
by sorry

end NUMINAMATH_CALUDE_work_equivalence_first_group_size_correct_l518_51894


namespace NUMINAMATH_CALUDE_f_domain_and_range_l518_51835

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt (1 - Real.cos (2 * x) + 2 * Real.sin x) + 1 / Real.sqrt (Real.sin x ^ 2 + Real.sin x)

def domain (x : ℝ) : Prop := ∃ k : ℤ, 2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi

theorem f_domain_and_range :
  (∀ x : ℝ, f x ≠ 0 → domain x) ∧
  (∀ y : ℝ, y ≥ 2 * (2 : ℝ) ^ (1/4) → ∃ x : ℝ, f x = y) :=
sorry

end NUMINAMATH_CALUDE_f_domain_and_range_l518_51835


namespace NUMINAMATH_CALUDE_initial_water_percentage_l518_51831

theorem initial_water_percentage 
  (initial_volume : ℝ) 
  (added_water : ℝ) 
  (final_water_percentage : ℝ) :
  initial_volume = 120 →
  added_water = 8 →
  final_water_percentage = 25 →
  (initial_volume * (20 / 100) + added_water) / (initial_volume + added_water) = final_water_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l518_51831


namespace NUMINAMATH_CALUDE_tax_rate_is_ten_percent_l518_51883

/-- Calculates the tax rate given the total amount spent, sales tax, and cost of tax-free items -/
def calculate_tax_rate (total_amount : ℚ) (sales_tax : ℚ) (tax_free_cost : ℚ) : ℚ :=
  let taxable_cost := total_amount - tax_free_cost - sales_tax
  (sales_tax / taxable_cost) * 100

/-- Theorem stating that the tax rate is 10% given the problem conditions -/
theorem tax_rate_is_ten_percent 
  (total_amount : ℚ) 
  (sales_tax : ℚ) 
  (tax_free_cost : ℚ)
  (h1 : total_amount = 25)
  (h2 : sales_tax = 3/10)
  (h3 : tax_free_cost = 217/10) :
  calculate_tax_rate total_amount sales_tax tax_free_cost = 10 := by
  sorry

#eval calculate_tax_rate 25 (3/10) (217/10)

end NUMINAMATH_CALUDE_tax_rate_is_ten_percent_l518_51883


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l518_51842

/-- Given a polar coordinate equation r = 3, prove it represents a circle with radius 3 centered at the origin in Cartesian coordinates. -/
theorem polar_to_cartesian_circle (x y : ℝ) : 
  (∃ θ : ℝ, x = 3 * Real.cos θ ∧ y = 3 * Real.sin θ) ↔ x^2 + y^2 = 9 := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l518_51842


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_star_l518_51897

-- Define the * operation
def star (a b : ℝ) : ℝ := a^2 - a*b - b^2

-- State the theorem
theorem sin_cos_pi_12_star :
  star (Real.sin (π/12)) (Real.cos (π/12)) = -(1 + 2*Real.sqrt 3)/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_star_l518_51897


namespace NUMINAMATH_CALUDE_haji_mother_sales_l518_51822

/-- Calculate the total sales for Haji's mother given the following conditions:
  - Tough week sales: $800
  - Tough week sales are half of good week sales
  - Number of good weeks: 5
  - Number of tough weeks: 3
-/
theorem haji_mother_sales (tough_week_sales : ℕ) (good_weeks : ℕ) (tough_weeks : ℕ)
  (h1 : tough_week_sales = 800)
  (h2 : good_weeks = 5)
  (h3 : tough_weeks = 3) :
  tough_week_sales * 2 * good_weeks + tough_week_sales * tough_weeks = 10400 :=
by sorry

end NUMINAMATH_CALUDE_haji_mother_sales_l518_51822


namespace NUMINAMATH_CALUDE_bookstore_earnings_difference_l518_51863

/-- Represents the earnings difference between two books --/
def earnings_difference (price_top : ℕ) (price_abc : ℕ) (quantity_top : ℕ) (quantity_abc : ℕ) : ℕ :=
  (price_top * quantity_top) - (price_abc * quantity_abc)

/-- Theorem: The earnings difference between "TOP" and "ABC" books is $12 --/
theorem bookstore_earnings_difference :
  earnings_difference 8 23 13 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_earnings_difference_l518_51863


namespace NUMINAMATH_CALUDE_age_difference_proof_l518_51811

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Calculates the age of a person after a given number of years -/
def age_after (p : Person) (years : ℕ) : ℕ := p.age + years

/-- Calculates the age of a person before a given number of years -/
def age_before (p : Person) (years : ℕ) : ℕ := p.age - years

/-- The problem statement -/
theorem age_difference_proof (john james james_brother : Person) 
    (h1 : john.age = 39)
    (h2 : age_before john 3 = 2 * age_after james 6)
    (h3 : james_brother.age = 16) :
  james_brother.age - james.age = 4 := by
  sorry


end NUMINAMATH_CALUDE_age_difference_proof_l518_51811


namespace NUMINAMATH_CALUDE_promotion_theorem_l518_51884

/-- Calculates the maximum amount of goods that can be purchased given a promotion and initial spending. -/
def maxPurchaseAmount (promotionRate : Rat) (rewardRate : Rat) (initialSpend : ℕ) : ℕ :=
  sorry

/-- The promotion theorem -/
theorem promotion_theorem :
  let promotionRate : Rat := 100
  let rewardRate : Rat := 20
  let initialSpend : ℕ := 7020
  maxPurchaseAmount promotionRate rewardRate initialSpend = 8760 := by
  sorry

end NUMINAMATH_CALUDE_promotion_theorem_l518_51884


namespace NUMINAMATH_CALUDE_outfit_combinations_l518_51865

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (shoes : ℕ) :
  shirts = 4 → pants = 5 → shoes = 3 →
  shirts * pants * shoes = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l518_51865


namespace NUMINAMATH_CALUDE_at_least_two_equal_l518_51860

theorem at_least_two_equal (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2/y + y^2/z + z^2/x = x^2/z + y^2/x + z^2/y) :
  x = y ∨ y = z ∨ z = x := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_equal_l518_51860


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l518_51844

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![2, x]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 8]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, v i = k * w i)

-- Theorem statement
theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (a x) (b x) → x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l518_51844


namespace NUMINAMATH_CALUDE_smallest_candy_count_l518_51840

theorem smallest_candy_count : ∃ (n : ℕ), 
  100 ≤ n ∧ n < 1000 ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 9) % 6 = 0 ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < n ∧ (m + 7) % 9 = 0 ∧ (m - 9) % 6 = 0 → False) ∧
  n = 101 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l518_51840


namespace NUMINAMATH_CALUDE_monotone_sin_range_l518_51895

theorem monotone_sin_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, Monotone (fun x ↦ f x)) →
  (∀ x ∈ Set.Icc 0 a, f x = Real.sin (2 * x + π / 3)) →
  a > 0 →
  0 < a ∧ a ≤ π / 12 := by
  sorry

end NUMINAMATH_CALUDE_monotone_sin_range_l518_51895


namespace NUMINAMATH_CALUDE_binary_110101101_equals_429_l518_51806

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110101101_equals_429 :
  binary_to_decimal [true, false, true, true, false, true, false, true, true] = 429 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101101_equals_429_l518_51806


namespace NUMINAMATH_CALUDE_max_distance_to_c_l518_51813

/-- The maximum distance from the origin to point C in an equilateral triangle ABC, 
    where A is on the unit circle and B is at (3,0) -/
theorem max_distance_to_c (A B C : ℝ × ℝ) : 
  (A.1^2 + A.2^2 = 1) →  -- A is on the unit circle
  (B = (3, 0)) →         -- B is at (3,0)
  (dist A B = dist B C ∧ dist B C = dist C A) →  -- ABC is equilateral
  (∃ (D : ℝ × ℝ), (D.1^2 + D.2^2 = 1) ∧  -- D is another point on the unit circle
    (dist D B = dist B C ∧ dist B C = dist C D) →  -- DBC is also equilateral
    dist (0, 0) C ≤ 4) :=  -- The distance from O to C is at most 4
by sorry

end NUMINAMATH_CALUDE_max_distance_to_c_l518_51813


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l518_51843

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 50) (h2 : x * y = 25) :
  1 / x + 1 / y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l518_51843


namespace NUMINAMATH_CALUDE_f_properties_l518_51841

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) * Real.exp x - (Real.exp 1 / 3) * x^3

noncomputable def g (x : ℝ) : ℝ := f x - 2

theorem f_properties :
  (∀ M : ℝ, ∃ x : ℝ, f x > M) ∧
  (∃ x₀ : ℝ, x₀ = 1 ∧ f x₀ = (2/3) * Real.exp 1 ∧ ∀ x : ℝ, f x ≥ f x₀) ∧
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ g x₁ = 0 ∧ g x₂ = 0 ∧ ∀ x ∈ Set.Ioo x₁ x₂, g x < 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l518_51841


namespace NUMINAMATH_CALUDE_total_hotdogs_sold_l518_51876

/-- Represents the number of hotdogs sold in each size category -/
structure HotdogSales where
  small : Float
  medium : Float
  large : Float
  extra_large : Float

/-- Calculates the total number of hotdogs sold -/
def total_hotdogs (sales : HotdogSales) : Float :=
  sales.small + sales.medium + sales.large + sales.extra_large

/-- Theorem: The total number of hotdogs sold is 131.3 -/
theorem total_hotdogs_sold (sales : HotdogSales)
  (h1 : sales.small = 58.3)
  (h2 : sales.medium = 21.7)
  (h3 : sales.large = 35.9)
  (h4 : sales.extra_large = 15.4) :
  total_hotdogs sales = 131.3 := by
  sorry

#eval total_hotdogs { small := 58.3, medium := 21.7, large := 35.9, extra_large := 15.4 }

end NUMINAMATH_CALUDE_total_hotdogs_sold_l518_51876


namespace NUMINAMATH_CALUDE_system_solution_l518_51836

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^3 + y₁^3) * (x₁^2 + y₁^2) = 64 ∧
    x₁ + y₁ = 2 ∧
    x₁ = 1 + Real.sqrt (5/3) ∧
    y₁ = 1 - Real.sqrt (5/3) ∧
    (x₂^3 + y₂^3) * (x₂^2 + y₂^2) = 64 ∧
    x₂ + y₂ = 2 ∧
    x₂ = 1 - Real.sqrt (5/3) ∧
    y₂ = 1 + Real.sqrt (5/3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l518_51836


namespace NUMINAMATH_CALUDE_square_root_calculations_l518_51847

theorem square_root_calculations : 
  (Real.sqrt 3)^2 = 3 ∧ Real.sqrt 8 * Real.sqrt 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_calculations_l518_51847


namespace NUMINAMATH_CALUDE_child_age_proof_l518_51808

/-- Represents a family with its members and their ages -/
structure Family where
  members : ℕ
  total_age : ℕ

/-- Calculates the average age of a family -/
def average_age (f : Family) : ℚ :=
  f.total_age / f.members

theorem child_age_proof (initial_family : Family)
  (h1 : initial_family.members = 5)
  (h2 : average_age initial_family = 17)
  (h3 : ∃ (new_family : Family),
    new_family.members = initial_family.members + 1 ∧
    new_family.total_age = initial_family.total_age + 3 * initial_family.members + 2 ∧
    average_age new_family = average_age initial_family) :
  2 = 2 := by
  sorry

#check child_age_proof

end NUMINAMATH_CALUDE_child_age_proof_l518_51808


namespace NUMINAMATH_CALUDE_sum_of_c_values_l518_51899

theorem sum_of_c_values : ∃ (S : Finset ℤ),
  (∀ c ∈ S, c ≤ 30 ∧ 
    ∃ (x y : ℚ), x^2 - 9*x - c = 0 ∧ y^2 - 9*y - c = 0 ∧ x ≠ y) ∧
  (∀ c : ℤ, c ≤ 30 → 
    (∃ (x y : ℚ), x^2 - 9*x - c = 0 ∧ y^2 - 9*y - c = 0 ∧ x ≠ y) → 
    c ∈ S) ∧
  (S.sum id = 32) := by
sorry

end NUMINAMATH_CALUDE_sum_of_c_values_l518_51899


namespace NUMINAMATH_CALUDE_unique_modular_residue_l518_51823

theorem unique_modular_residue :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -3736 [ZMOD 6] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_residue_l518_51823


namespace NUMINAMATH_CALUDE_planet_combinations_correct_l518_51890

/-- The number of different combinations of planets that can be occupied. -/
def planetCombinations : ℕ :=
  let earthLike := 7
  let marsLike := 8
  let earthUnits := 3
  let marsUnits := 1
  let totalUnits := 21
  2941

/-- Theorem stating that the number of planet combinations is correct. -/
theorem planet_combinations_correct : planetCombinations = 2941 := by
  sorry

end NUMINAMATH_CALUDE_planet_combinations_correct_l518_51890


namespace NUMINAMATH_CALUDE_line_inclination_angle_l518_51848

theorem line_inclination_angle (x y : ℝ) :
  x + Real.sqrt 3 * y - 1 = 0 →
  ∃ θ : ℝ, θ = 5 * Real.pi / 6 ∧ Real.tan θ = -1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l518_51848


namespace NUMINAMATH_CALUDE_prime_triple_divisibility_l518_51853

theorem prime_triple_divisibility (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  p ∣ (q + r) ∧ q ∣ (r + 2*p) ∧ r ∣ (p + 3*q) →
  ((p = 5 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 2 ∧ q = 11 ∧ r = 7) ∨ 
   (p = 2 ∧ q = 3 ∧ r = 11)) :=
by sorry

#check prime_triple_divisibility

end NUMINAMATH_CALUDE_prime_triple_divisibility_l518_51853


namespace NUMINAMATH_CALUDE_min_cooking_time_is_15_l518_51819

/-- Represents the time required for each step in the noodle cooking process -/
structure CookingTimes where
  washPot : ℕ
  washVegetables : ℕ
  prepareIngredients : ℕ
  boilWater : ℕ
  cookNoodles : ℕ

/-- Calculates the minimum time to cook noodles given the cooking times -/
def minCookingTime (times : CookingTimes) : ℕ :=
  let simultaneousTime := max times.washVegetables times.prepareIngredients
  times.washPot + simultaneousTime + times.cookNoodles

/-- Theorem stating that the minimum cooking time is 15 minutes -/
theorem min_cooking_time_is_15 (times : CookingTimes) 
  (h1 : times.washPot = 2)
  (h2 : times.washVegetables = 6)
  (h3 : times.prepareIngredients = 2)
  (h4 : times.boilWater = 10)
  (h5 : times.cookNoodles = 3) :
  minCookingTime times = 15 := by
  sorry

#eval minCookingTime ⟨2, 6, 2, 10, 3⟩

end NUMINAMATH_CALUDE_min_cooking_time_is_15_l518_51819


namespace NUMINAMATH_CALUDE_parallel_transitivity_perpendicular_to_parallel_not_always_intersects_l518_51846

-- Define a 3D space
structure Space3D where
  -- Add necessary fields for 3D space

-- Define a line in 3D space
structure Line3D where
  -- Add necessary fields for a line in 3D space

-- Define parallel lines in 3D space
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

-- Define perpendicular lines in 3D space
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

-- Define line intersection in 3D space
def intersects (l1 l2 : Line3D) : Prop :=
  sorry

theorem parallel_transitivity (l1 l2 l3 : Line3D) :
  parallel l1 l2 → parallel l2 l3 → parallel l1 l3 :=
  sorry

theorem perpendicular_to_parallel (l1 l2 l3 : Line3D) :
  parallel l1 l2 → perpendicular l3 l1 → perpendicular l3 l2 :=
  sorry

theorem not_always_intersects (l1 l2 l3 : Line3D) :
  ¬(parallel l1 l2 → intersects l3 l1 → intersects l3 l2) :=
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_perpendicular_to_parallel_not_always_intersects_l518_51846


namespace NUMINAMATH_CALUDE_final_balance_l518_51896

def account_balance (initial : ℕ) (coffee_beans : ℕ) (tumbler : ℕ) (coffee_filter : ℕ) (refund : ℕ) : ℕ :=
  initial - (coffee_beans + tumbler + coffee_filter) + refund

theorem final_balance :
  account_balance 50 10 30 5 20 = 25 := by
  sorry

end NUMINAMATH_CALUDE_final_balance_l518_51896


namespace NUMINAMATH_CALUDE_prob_first_ace_equal_sum_prob_is_one_l518_51864

/-- Represents a player in the card game -/
inductive Player : Type
| one : Player
| two : Player
| three : Player
| four : Player

/-- The total number of cards in the deck -/
def totalCards : ℕ := 32

/-- The number of aces in the deck -/
def numAces : ℕ := 4

/-- The number of players in the game -/
def numPlayers : ℕ := 4

/-- Calculates the probability of a player getting the first ace -/
def probFirstAce (p : Player) : ℚ :=
  1 / 8

/-- Theorem: The probability of each player getting the first ace is 1/8 -/
theorem prob_first_ace_equal (p : Player) : 
  probFirstAce p = 1 / 8 := by
  sorry

/-- Theorem: The sum of probabilities for all players is 1 -/
theorem sum_prob_is_one : 
  (probFirstAce Player.one) + (probFirstAce Player.two) + 
  (probFirstAce Player.three) + (probFirstAce Player.four) = 1 := by
  sorry

end NUMINAMATH_CALUDE_prob_first_ace_equal_sum_prob_is_one_l518_51864


namespace NUMINAMATH_CALUDE_linear_function_property_l518_51817

-- Define a linear function g
def g (x : ℝ) : ℝ := sorry

-- State the theorem
theorem linear_function_property :
  (∀ x y a b : ℝ, g (a * x + b * y) = a * g x + b * g y) →  -- g is linear
  (∀ x : ℝ, g x = 3 * g⁻¹ x + 5) →  -- g(x) = 3g^(-1)(x) + 5
  g 0 = 3 →  -- g(0) = 3
  g (-1) = 3 - Real.sqrt 3 :=  -- g(-1) = 3 - √3
by sorry

end NUMINAMATH_CALUDE_linear_function_property_l518_51817


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l518_51878

theorem geometric_progression_ratio (a b c : ℝ) (x : ℝ) (r : ℝ) : 
  a = 30 → b = 80 → c = 160 →
  (b + x)^2 = (a + x) * (c + x) →
  x = 160 / 3 →
  r = (b + x) / (a + x) →
  r = (c + x) / (b + x) →
  r = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l518_51878


namespace NUMINAMATH_CALUDE_pecans_weight_l518_51839

def total_nuts : ℝ := 0.52
def almonds : ℝ := 0.14

theorem pecans_weight : total_nuts - almonds = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_pecans_weight_l518_51839


namespace NUMINAMATH_CALUDE_line_parameterization_l518_51825

/-- Given a line y = 3x - 11 parameterized by (x, y) = (r, 1) + t(4, k),
    prove that r = 4 and k = 12 -/
theorem line_parameterization (r k : ℝ) : 
  (∀ t : ℝ, (r + 4*t, 1 + k*t) ∈ {p : ℝ × ℝ | p.2 = 3*p.1 - 11}) ↔ 
  (r = 4 ∧ k = 12) :=
by sorry

end NUMINAMATH_CALUDE_line_parameterization_l518_51825


namespace NUMINAMATH_CALUDE_solution_composition_l518_51885

theorem solution_composition (x : ℝ) : 
  -- First solution composition
  let solution1_A := 0.20
  let solution1_B := 0.80
  -- Second solution composition
  let solution2_A := x
  let solution2_B := 0.70
  -- Mixture composition
  let mixture_solution1 := 0.80
  let mixture_solution2 := 0.20
  -- Final mixture composition of material A
  let final_mixture_A := 0.22
  -- Equation for material A in the final mixture
  solution1_A * mixture_solution1 + solution2_A * mixture_solution2 = final_mixture_A
  →
  x = 0.30 := by
sorry

end NUMINAMATH_CALUDE_solution_composition_l518_51885


namespace NUMINAMATH_CALUDE_sebastian_took_no_arabs_l518_51837

theorem sebastian_took_no_arabs (x : ℕ) (y : ℕ) (z : ℕ) : x > 0 →
  -- x is the initial number of each type of soldier
  -- y is the number of cowboys taken (equal to remaining Eskimos)
  -- z is the number of Arab soldiers taken
  y ≤ x →  -- Number of cowboys taken cannot exceed initial number
  4 * x / 3 = y + (x - y) + x / 3 + z →  -- Total soldiers taken
  z = 0 := by
sorry

end NUMINAMATH_CALUDE_sebastian_took_no_arabs_l518_51837


namespace NUMINAMATH_CALUDE_circle_rational_points_infinite_l518_51871

theorem circle_rational_points_infinite :
  ∃ (S : Set (ℚ × ℚ)), Set.Infinite S ∧ ∀ (p : ℚ × ℚ), p ∈ S → (p.1^2 + p.2^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_rational_points_infinite_l518_51871


namespace NUMINAMATH_CALUDE_division_simplification_l518_51857

theorem division_simplification : (180 : ℚ) / (12 + 15 * 3) = 180 / 57 := by sorry

end NUMINAMATH_CALUDE_division_simplification_l518_51857


namespace NUMINAMATH_CALUDE_lisas_lasagna_consumption_l518_51800

/-- The number of pieces Lisa eats from a lasagna, given the eating habits of her friends. -/
def lisas_lasagna_pieces (total_pieces manny_pieces aaron_pieces : ℚ) : ℚ :=
  let kai_pieces := 2 * manny_pieces
  let raphael_pieces := manny_pieces / 2
  total_pieces - (manny_pieces + kai_pieces + raphael_pieces + aaron_pieces)

/-- Theorem stating that Lisa will eat 2.5 pieces of lasagna given the specific conditions. -/
theorem lisas_lasagna_consumption :
  lisas_lasagna_pieces 6 1 0 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_lisas_lasagna_consumption_l518_51800


namespace NUMINAMATH_CALUDE_vector_problem_l518_51807

theorem vector_problem (a b : Fin 2 → ℝ) (x : ℝ) 
    (h1 : a + b = ![2, x])
    (h2 : a - b = ![-2, 1])
    (h3 : ‖a‖^2 - ‖b‖^2 = -1) : 
  x = 3 := by sorry

end NUMINAMATH_CALUDE_vector_problem_l518_51807


namespace NUMINAMATH_CALUDE_not_divisible_five_power_l518_51852

theorem not_divisible_five_power (n k : ℕ) : ¬ ((5^k - 1) ∣ (5^n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_five_power_l518_51852


namespace NUMINAMATH_CALUDE_sqrt_a_plus_b_equals_four_l518_51898

theorem sqrt_a_plus_b_equals_four :
  ∀ a b : ℕ,
  (a = ⌊Real.sqrt 17⌋) →
  (b - 1 = Real.sqrt 121) →
  Real.sqrt (a + b) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_plus_b_equals_four_l518_51898


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_container_l518_51881

def container_capacity : ℕ := 1125
def bottle_type1_capacity : ℕ := 45
def bottle_type2_capacity : ℕ := 75

theorem min_bottles_to_fill_container :
  ∃ (n1 n2 : ℕ),
    n1 * bottle_type1_capacity + n2 * bottle_type2_capacity = container_capacity ∧
    ∀ (m1 m2 : ℕ), 
      m1 * bottle_type1_capacity + m2 * bottle_type2_capacity = container_capacity →
      n1 + n2 ≤ m1 + m2 ∧
    n1 + n2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_bottles_to_fill_container_l518_51881


namespace NUMINAMATH_CALUDE_correct_change_l518_51888

/-- Calculates the change received when purchasing frames -/
def change_received (num_frames : ℕ) (frame_cost : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_frames * frame_cost)

/-- Proves that the change received is correct for the given problem -/
theorem correct_change : change_received 3 3 20 = 11 := by
  sorry

end NUMINAMATH_CALUDE_correct_change_l518_51888


namespace NUMINAMATH_CALUDE_lasagna_profit_proof_l518_51891

/-- Calculates the profit after expenses for selling lasagna pans -/
def profit_after_expenses (num_pans : ℕ) (cost_per_pan : ℚ) (price_per_pan : ℚ) : ℚ :=
  num_pans * (price_per_pan - cost_per_pan)

/-- Proves that the profit after expenses for selling 20 pans of lasagna is $300.00 -/
theorem lasagna_profit_proof :
  profit_after_expenses 20 10 25 = 300 := by
  sorry

end NUMINAMATH_CALUDE_lasagna_profit_proof_l518_51891


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l518_51826

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a ≤ 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l518_51826


namespace NUMINAMATH_CALUDE_parabola_point_distance_l518_51892

/-- Parabola type representing y = -ax²/6 + ax + c -/
structure Parabola where
  a : ℝ
  c : ℝ
  h_a : a < 0

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y = -p.a * x^2 / 6 + p.a * x + p.c

/-- Theorem statement -/
theorem parabola_point_distance (p : Parabola) 
  (A B C : ParabolaPoint p) 
  (h_B_vertex : B.y = 3 * p.a / 2 + p.c) 
  (h_y_order : A.y > C.y ∧ C.y > B.y) :
  |A.x - B.x| ≥ |B.x - C.x| := by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l518_51892


namespace NUMINAMATH_CALUDE_road_trip_ratio_l518_51862

theorem road_trip_ratio : 
  ∀ (x : ℝ),
  x > 0 →
  x + 2*x + 40 + 2*(x + 2*x + 40) = 560 →
  40 / x = 9 / 11 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_ratio_l518_51862


namespace NUMINAMATH_CALUDE_no_xy_term_iff_k_eq_four_l518_51851

/-- The polynomial multiplication (x+2y)(2x-ky-1) does not contain the term xy if and only if k = 4 -/
theorem no_xy_term_iff_k_eq_four (k : ℝ) : 
  (∀ x y : ℝ, (x + 2*y) * (2*x - k*y - 1) = 2*x^2 - x - 2*k*y^2 - 2*y) ↔ k = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_xy_term_iff_k_eq_four_l518_51851


namespace NUMINAMATH_CALUDE_max_reflections_l518_51880

theorem max_reflections (angle : ℝ) (h : angle = 8) : 
  ∃ (n : ℕ), n ≤ 10 ∧ n * angle < 90 ∧ ∀ m : ℕ, m > n → m * angle ≥ 90 :=
by sorry

end NUMINAMATH_CALUDE_max_reflections_l518_51880


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l518_51882

/-- A parabola passing through the points (-1, -6) and (1, 0) -/
def Parabola (x y : ℝ) : Prop :=
  ∃ (m n : ℝ), y = x^2 + m*x + n ∧ -6 = 1 - m + n ∧ 0 = 1 + m + n

/-- The intersection point of the parabola with the y-axis -/
def YAxisIntersection (x y : ℝ) : Prop :=
  Parabola x y ∧ x = 0

theorem parabola_y_axis_intersection :
  ∀ x y, YAxisIntersection x y → x = 0 ∧ y = -4 := by sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l518_51882


namespace NUMINAMATH_CALUDE_system_solutions_l518_51803

def is_solution (x y z : ℝ) : Prop :=
  x + y + z = 3 ∧
  x + 2*y - z = 2 ∧
  x + y*z + z*x = 3

theorem system_solutions :
  (∃ (x y z : ℝ), is_solution x y z) ∧
  (∀ (x y z : ℝ), is_solution x y z →
    ((x = 6 + Real.sqrt 29 ∧
      y = (-7 - 2 * Real.sqrt 29) / 3 ∧
      z = (-2 - Real.sqrt 29) / 3) ∨
     (x = 6 - Real.sqrt 29 ∧
      y = (-7 + 2 * Real.sqrt 29) / 3 ∧
      z = (-2 + Real.sqrt 29) / 3))) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l518_51803


namespace NUMINAMATH_CALUDE_candy_distribution_l518_51886

theorem candy_distribution (total_candy : ℕ) (num_students : ℕ) (candy_per_student : ℕ) : 
  total_candy = 18 → num_students = 9 → candy_per_student = total_candy / num_students → 
  candy_per_student = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l518_51886


namespace NUMINAMATH_CALUDE_intersection_M_N_l518_51861

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N in U
def complement_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := {x | x ∈ U ∧ x ∉ complement_N}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l518_51861


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l518_51870

theorem multiplication_addition_equality : 15 * 36 + 15 * 24 = 900 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l518_51870


namespace NUMINAMATH_CALUDE_expression_equals_zero_l518_51856

theorem expression_equals_zero (x y : ℝ) : 
  (5 * x^2 - 3 * x + 2) * (107 - 107) + (7 * y^2 + 4 * y - 1) * (93 - 93) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l518_51856


namespace NUMINAMATH_CALUDE_chris_birthday_money_l518_51828

theorem chris_birthday_money (x : ℕ) : 
  x + 25 + 20 + 75 = 279 → x = 159 := by
sorry

end NUMINAMATH_CALUDE_chris_birthday_money_l518_51828


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l518_51810

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- Positive terms
  (q ≠ 1) →  -- Common ratio not equal to 1
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence definition
  (a 2 - (1/2 * a 3) = (1/2 * a 3) - a 1) →  -- Arithmetic sequence condition
  ((a 4 + a 5) / (a 3 + a 4) = (1 + Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l518_51810


namespace NUMINAMATH_CALUDE_remainder_3_87_plus_5_mod_9_l518_51809

theorem remainder_3_87_plus_5_mod_9 : (3^87 + 5) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_87_plus_5_mod_9_l518_51809


namespace NUMINAMATH_CALUDE_line_equation_slope_5_through_0_2_l518_51867

/-- The equation of a line with slope 5 passing through (0, 2) -/
theorem line_equation_slope_5_through_0_2 :
  ∀ (x y : ℝ), (5 * x - y + 2 = 0) ↔ 
  (∃ (t : ℝ), x = t ∧ y = 5 * t + 2) := by sorry

end NUMINAMATH_CALUDE_line_equation_slope_5_through_0_2_l518_51867


namespace NUMINAMATH_CALUDE_min_value_xyz_l518_51873

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 18) :
  x + 3 * y + 6 * z ≥ 3 * (2 * Real.sqrt 6 + 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_xyz_l518_51873


namespace NUMINAMATH_CALUDE_expansion_properties_l518_51889

theorem expansion_properties (n : ℕ) : 
  (∃ a b : ℚ, 
    (1 : ℚ) = a ∧ 
    (n : ℚ) * (1 / 2 : ℚ) = a + b ∧ 
    (n * (n - 1) / 2 : ℚ) * (1 / 4 : ℚ) = a + 2 * b) → 
  n = 8 ∧ (2 : ℕ) ^ n = 256 :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l518_51889


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_sum_l518_51816

/-- The number of entries in the nth row of Pascal's Triangle -/
def pascalRowEntries (n : ℕ) : ℕ := n + 1

/-- The sum of entries in the first n rows of Pascal's Triangle -/
def pascalTriangleSum (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem pascal_triangle_30_rows_sum :
  pascalTriangleSum 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_sum_l518_51816


namespace NUMINAMATH_CALUDE_line_intersects_circle_l518_51832

/-- The line l defined by 2mx - y - 8m - 3 = 0 -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  2 * m * x - y - 8 * m - 3 = 0

/-- The circle C defined by (x - 3)² + (y + 6)² = 25 -/
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 6)^2 = 25

/-- The theorem stating that the line l intersects the circle C for any real m -/
theorem line_intersects_circle :
  ∀ m : ℝ, ∃ x y : ℝ, line_l m x y ∧ circle_C x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l518_51832


namespace NUMINAMATH_CALUDE_inequality_proof_l518_51868

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l518_51868


namespace NUMINAMATH_CALUDE_pencil_distribution_l518_51845

theorem pencil_distribution (P : ℕ) (h : P % 9 = 8) :
  ∃ k : ℕ, P = 9 * k + 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_distribution_l518_51845


namespace NUMINAMATH_CALUDE_intersection_of_sets_l518_51833

theorem intersection_of_sets : 
  let P : Set ℕ := {3, 5, 6, 8}
  let Q : Set ℕ := {4, 5, 7, 8}
  P ∩ Q = {5, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l518_51833


namespace NUMINAMATH_CALUDE_distance_between_foci_l518_51801

def ellipse (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y + 9)^2) = 22

def focus1 : ℝ × ℝ := (4, -5)
def focus2 : ℝ × ℝ := (-6, 9)

theorem distance_between_foci :
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = 2 * Real.sqrt 74 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_foci_l518_51801


namespace NUMINAMATH_CALUDE_min_value_on_circle_l518_51874

theorem min_value_on_circle (x y : ℝ) (h : x^2 + y^2 - 2*x - 2*y + 1 = 0) :
  ∃ (m : ℝ), m = 4/3 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 - 2*x' - 2*y' + 1 = 0 →
    (y' - 4) / (x' - 2) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l518_51874


namespace NUMINAMATH_CALUDE_proportion_solve_l518_51804

theorem proportion_solve (x : ℚ) : (3 : ℚ) / 12 = x / 16 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solve_l518_51804
