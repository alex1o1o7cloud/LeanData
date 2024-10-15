import Mathlib

namespace NUMINAMATH_CALUDE_inverse_function_sum_l3231_323109

/-- Given a function g and its inverse g⁻¹, prove that c + d = 3 * (2^(1/3)) -/
theorem inverse_function_sum (c d : ℝ) 
  (g : ℝ → ℝ) (g_inv : ℝ → ℝ)
  (hg : ∀ x, g x = c * x + d)
  (hg_inv : ∀ x, g_inv x = d * x - 2 * c)
  (h_inverse : ∀ x, g (g_inv x) = x) :
  c + d = 3 * Real.rpow 2 (1/3) := by
sorry

end NUMINAMATH_CALUDE_inverse_function_sum_l3231_323109


namespace NUMINAMATH_CALUDE_special_triangle_min_perimeter_l3231_323107

/-- Triangle ABC with integer side lengths and specific angle conditions -/
structure SpecialTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  angle_A_twice_B : angle_A = 2 * angle_B
  angle_C_obtuse : angle_C > Real.pi / 2
  angle_sum : angle_A + angle_B + angle_C = Real.pi

/-- The minimum perimeter of a SpecialTriangle is 77 -/
theorem special_triangle_min_perimeter (t : SpecialTriangle) : t.a + t.b + t.c ≥ 77 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_min_perimeter_l3231_323107


namespace NUMINAMATH_CALUDE_combined_annual_income_l3231_323121

def monthly_income_problem (A B C D : ℝ) : Prop :=
  -- Ratio condition
  A / B = 5 / 3 ∧ B / C = 3 / 2 ∧
  -- B's income is 12% more than C's
  B = 1.12 * C ∧
  -- D's income is 15% less than A's
  D = 0.85 * A ∧
  -- C's income is 17000
  C = 17000

theorem combined_annual_income 
  (A B C D : ℝ) 
  (h : monthly_income_problem A B C D) : 
  (A + B + C + D) * 12 = 1375980 := by
  sorry

#check combined_annual_income

end NUMINAMATH_CALUDE_combined_annual_income_l3231_323121


namespace NUMINAMATH_CALUDE_circle_graph_percentage_l3231_323112

theorem circle_graph_percentage (total_degrees : ℝ) (total_percentage : ℝ) 
  (manufacturing_degrees : ℝ) (manufacturing_percentage : ℝ) : 
  total_degrees = 360 →
  total_percentage = 100 →
  manufacturing_degrees = 108 →
  manufacturing_percentage / total_percentage = manufacturing_degrees / total_degrees →
  manufacturing_percentage = 30 := by
sorry

end NUMINAMATH_CALUDE_circle_graph_percentage_l3231_323112


namespace NUMINAMATH_CALUDE_mean_proportional_segment_l3231_323177

theorem mean_proportional_segment (a c : ℝ) (x : ℝ) 
  (ha : a = 9) (hc : c = 4) (hx : x^2 = a * c) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_segment_l3231_323177


namespace NUMINAMATH_CALUDE_lake_bright_population_is_16000_l3231_323173

-- Define the total population
def total_population : ℕ := 80000

-- Define Gordonia's population as a fraction of the total
def gordonia_population : ℕ := total_population / 2

-- Define Toadon's population as a percentage of Gordonia's
def toadon_population : ℕ := (gordonia_population * 60) / 100

-- Define Lake Bright's population
def lake_bright_population : ℕ := total_population - gordonia_population - toadon_population

-- Theorem statement
theorem lake_bright_population_is_16000 :
  lake_bright_population = 16000 := by sorry

end NUMINAMATH_CALUDE_lake_bright_population_is_16000_l3231_323173


namespace NUMINAMATH_CALUDE_no_primes_in_sequence_infinitely_many_x_with_no_primes_l3231_323162

/-- Definition of the sequence a_n -/
def a (x : ℕ) : ℕ → ℕ
| 0 => 1
| 1 => x + 1
| (n + 2) => x * a x (n + 1) - a x n

/-- Theorem stating that for any c ≥ 3, the sequence contains no primes when x = c² - 2 -/
theorem no_primes_in_sequence (c : ℕ) (h : c ≥ 3) :
  ∀ n : ℕ, ¬ Nat.Prime (a (c^2 - 2) n) := by
  sorry

/-- Corollary: There exist infinitely many x such that the sequence contains no primes -/
theorem infinitely_many_x_with_no_primes :
  ∃ f : ℕ → ℕ, Monotone f ∧ ∀ k : ℕ, ∀ n : ℕ, ¬ Nat.Prime (a (f k) n) := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_sequence_infinitely_many_x_with_no_primes_l3231_323162


namespace NUMINAMATH_CALUDE_arccos_negative_one_equals_pi_l3231_323185

theorem arccos_negative_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_one_equals_pi_l3231_323185


namespace NUMINAMATH_CALUDE_price_increase_percentage_l3231_323178

theorem price_increase_percentage (new_price : ℝ) (h1 : new_price - 0.8 * new_price = 4) : 
  (new_price - (0.8 * new_price)) / (0.8 * new_price) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l3231_323178


namespace NUMINAMATH_CALUDE_binomial_representation_existence_and_uniqueness_l3231_323147

theorem binomial_representation_existence_and_uniqueness 
  (t l : ℕ) : 
  ∃! (m : ℕ) (a : ℕ → ℕ), 
    m ≤ l ∧ 
    (∀ i ∈ Finset.range (l - m + 1), a (m + i) ≥ m + i) ∧
    (∀ i ∈ Finset.range (l - m), a (m + i + 1) > a (m + i)) ∧
    t = (Finset.range (l - m + 1)).sum (λ i => Nat.choose (a (m + i)) (m + i)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_representation_existence_and_uniqueness_l3231_323147


namespace NUMINAMATH_CALUDE_digit_156_is_5_l3231_323196

/-- The decimal expansion of 47/777 -/
def decimal_expansion : ℚ := 47 / 777

/-- The length of the repeating block in the decimal expansion -/
def repeating_block_length : ℕ := 6

/-- The position of the digit we're looking for -/
def target_position : ℕ := 156

/-- The function that returns the nth digit after the decimal point in the decimal expansion -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

theorem digit_156_is_5 : nth_digit (target_position - 1) = 5 := by sorry

end NUMINAMATH_CALUDE_digit_156_is_5_l3231_323196


namespace NUMINAMATH_CALUDE_average_student_headcount_theorem_l3231_323192

/-- Represents the student headcount for a specific academic year --/
structure StudentCount where
  year : String
  count : ℕ

/-- Calculates the average of a list of natural numbers --/
def average (nums : List ℕ) : ℚ :=
  (nums.sum : ℚ) / nums.length

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (q : ℚ) : ℤ :=
  (q + 1/2).floor

theorem average_student_headcount_theorem 
  (headcounts : List StudentCount)
  (error_margin : ℕ)
  (h1 : headcounts.length = 3)
  (h2 : error_margin = 50)
  (h3 : ∀ sc ∈ headcounts, sc.count ≥ 10000 ∧ sc.count ≤ 12000) :
  roundToNearest (average (headcounts.map (λ sc ↦ sc.count))) = 10833 := by
sorry

end NUMINAMATH_CALUDE_average_student_headcount_theorem_l3231_323192


namespace NUMINAMATH_CALUDE_bubble_sort_correct_l3231_323106

def bubble_sort (xs : List Int) : List Int :=
  let rec pass (ys : List Int) : List Int :=
    match ys with
    | [] => []
    | [x] => [x]
    | x :: y :: rest =>
      if x > y
      then y :: pass (x :: rest)
      else x :: pass (y :: rest)
  let rec sort (zs : List Int) (n : Nat) : List Int :=
    if n = 0 then zs
    else sort (pass zs) (n - 1)
  sort xs xs.length

theorem bubble_sort_correct (xs : List Int) :
  bubble_sort [8, 6, 3, 18, 21, 67, 54] = [3, 6, 8, 18, 21, 54, 67] := by
  sorry

end NUMINAMATH_CALUDE_bubble_sort_correct_l3231_323106


namespace NUMINAMATH_CALUDE_alpha_value_when_beta_is_36_l3231_323179

/-- Given that α² is inversely proportional to β, and α = 4 when β = 9,
    prove that α = ±2 when β = 36. -/
theorem alpha_value_when_beta_is_36
  (k : ℝ)  -- Constant of proportionality
  (h1 : ∀ α β : ℝ, α ^ 2 * β = k)  -- α² is inversely proportional to β
  (h2 : 4 ^ 2 * 9 = k)  -- α = 4 when β = 9
  : {α : ℝ | α ^ 2 * 36 = k} = {2, -2} := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_when_beta_is_36_l3231_323179


namespace NUMINAMATH_CALUDE_food_drive_total_cans_l3231_323181

theorem food_drive_total_cans 
  (mark_cans jaydon_cans rachel_cans : ℕ) 
  (h1 : mark_cans = 4 * jaydon_cans)
  (h2 : jaydon_cans = 2 * rachel_cans + 5)
  (h3 : mark_cans = 100) : 
  mark_cans + jaydon_cans + rachel_cans = 135 := by
sorry


end NUMINAMATH_CALUDE_food_drive_total_cans_l3231_323181


namespace NUMINAMATH_CALUDE_movie_count_theorem_l3231_323199

/-- The number of movies Timothy and Theresa watched in 2009 and 2010 -/
def total_movies (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ) : ℕ :=
  timothy_2009 + timothy_2010 + theresa_2009 + theresa_2010

theorem movie_count_theorem :
  ∀ (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ),
    timothy_2010 = timothy_2009 + 7 →
    timothy_2009 = 24 →
    theresa_2010 = 2 * timothy_2010 →
    theresa_2009 = timothy_2009 / 2 →
    total_movies timothy_2009 timothy_2010 theresa_2009 theresa_2010 = 129 :=
by
  sorry

end NUMINAMATH_CALUDE_movie_count_theorem_l3231_323199


namespace NUMINAMATH_CALUDE_origin_and_point_same_side_l3231_323144

def line_equation (x y : ℝ) : ℝ := 3 * x + 2 * y + 5

def same_side (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  line_equation x₁ y₁ * line_equation x₂ y₂ > 0

theorem origin_and_point_same_side : same_side 0 0 (-3) 4 := by sorry

end NUMINAMATH_CALUDE_origin_and_point_same_side_l3231_323144


namespace NUMINAMATH_CALUDE_cindy_envelope_distribution_l3231_323156

theorem cindy_envelope_distribution (initial_envelopes : ℕ) (friends : ℕ) (remaining_envelopes : ℕ) 
  (h1 : initial_envelopes = 37)
  (h2 : friends = 5)
  (h3 : remaining_envelopes = 22) :
  (initial_envelopes - remaining_envelopes) / friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_cindy_envelope_distribution_l3231_323156


namespace NUMINAMATH_CALUDE_pentagonal_field_fencing_cost_l3231_323174

/-- Calculates the cost of fencing for an irregular pentagonal field -/
def fencing_cost (side1 side2 side3 side4 side5 : ℕ) 
                 (rate1 rate2 rate3 : ℕ) : ℕ :=
  rate1 * (side1 + side2) + rate2 * side3 + rate3 * (side4 + side5)

/-- Theorem stating the total cost of fencing for the given pentagonal field -/
theorem pentagonal_field_fencing_cost :
  fencing_cost 42 37 52 65 48 7 5 10 = 1943 := by sorry

end NUMINAMATH_CALUDE_pentagonal_field_fencing_cost_l3231_323174


namespace NUMINAMATH_CALUDE_triangle_angle_B_l3231_323119

theorem triangle_angle_B (A B C : Real) (a b : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths
  a = 4 ∧ b = 5 →
  -- Given condition
  Real.cos (B + C) + 3/5 = 0 →
  -- Conclusion: Measure of angle B
  B = Real.pi - Real.arccos (3/5) := by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l3231_323119


namespace NUMINAMATH_CALUDE_rectangle_placement_l3231_323117

theorem rectangle_placement (a b c d : ℝ) 
  (h1 : a < c) (h2 : c ≤ d) (h3 : d < b) (h4 : a * b < c * d) :
  (∃ (x y : ℝ), x ≤ c ∧ y ≤ d ∧ x * y = a * b) ↔ 
  (b^2 - a^2)^2 ≤ (b*c - a*d)^2 + (b*d - a*c)^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_placement_l3231_323117


namespace NUMINAMATH_CALUDE_digit_150_is_1_l3231_323146

/-- The decimal expansion of 5/31 -/
def decimal_expansion : ℚ := 5 / 31

/-- The length of the repeating part in the decimal expansion of 5/31 -/
def repetition_length : ℕ := 15

/-- The position we're interested in -/
def target_position : ℕ := 150

/-- The function that returns the nth digit after the decimal point in the decimal expansion of 5/31 -/
noncomputable def nth_digit (n : ℕ) : ℕ := 
  sorry

theorem digit_150_is_1 : nth_digit target_position = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_is_1_l3231_323146


namespace NUMINAMATH_CALUDE_jims_estimate_l3231_323116

theorem jims_estimate (x y ε : ℝ) (hx : x > y) (hy : y > 0) (hε : ε > 0) :
  (x^2 + ε) - (y^2 - ε) > x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_jims_estimate_l3231_323116


namespace NUMINAMATH_CALUDE_flower_shop_purchase_l3231_323100

theorem flower_shop_purchase 
  (total_flowers : ℕ) 
  (total_cost : ℚ) 
  (carnation_price : ℚ) 
  (rose_price : ℚ) 
  (h1 : total_flowers = 400)
  (h2 : total_cost = 1020)
  (h3 : carnation_price = 6/5)  -- $1.2 as a rational number
  (h4 : rose_price = 3) :
  ∃ (carnations roses : ℕ),
    carnations + roses = total_flowers ∧
    carnation_price * carnations + rose_price * roses = total_cost ∧
    carnations = 100 ∧
    roses = 300 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_purchase_l3231_323100


namespace NUMINAMATH_CALUDE_luis_gum_contribution_l3231_323127

/-- Calculates the number of gum pieces Luis gave to Maria -/
def luisGumPieces (initialPieces tomsContribution totalPieces : ℕ) : ℕ :=
  totalPieces - (initialPieces + tomsContribution)

theorem luis_gum_contribution :
  luisGumPieces 25 16 61 = 20 := by
  sorry

end NUMINAMATH_CALUDE_luis_gum_contribution_l3231_323127


namespace NUMINAMATH_CALUDE_triangle_properties_l3231_323193

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (t.a^2 + t.b^2 - t.c^2) * Real.tan t.C = Real.sqrt 2 * t.a * t.b) :
  (t.C = π/4 ∨ t.C = 3*π/4) ∧ 
  (t.c = 2 ∧ t.b = 2 * Real.sqrt 2 → 
    1/2 * t.a * t.b * Real.sin t.C = 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3231_323193


namespace NUMINAMATH_CALUDE_half_square_identity_l3231_323149

theorem half_square_identity (a : ℤ) : (a + 1/2)^2 = a * (a + 1) + 1/4 := by
  sorry

end NUMINAMATH_CALUDE_half_square_identity_l3231_323149


namespace NUMINAMATH_CALUDE_solution_set_f_geq_12_range_of_a_l3231_323123

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 5| + |x + 4|

-- Theorem for the solution set of f(x) ≥ 12
theorem solution_set_f_geq_12 :
  {x : ℝ | f x ≥ 12} = {x : ℝ | x ≥ 13/2 ∨ x ≤ -11/2} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x - 2^(1 - 3*a) - 1 ≥ 0) → a ≥ -2/3 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_12_range_of_a_l3231_323123


namespace NUMINAMATH_CALUDE_attendance_difference_l3231_323198

/-- The attendance difference between this week and last week for baseball games --/
theorem attendance_difference : 
  let second_game : ℕ := 80
  let first_game : ℕ := second_game - 20
  let third_game : ℕ := second_game + 15
  let this_week_total : ℕ := first_game + second_game + third_game
  let last_week_total : ℕ := 200
  this_week_total - last_week_total = 35 := by
  sorry

end NUMINAMATH_CALUDE_attendance_difference_l3231_323198


namespace NUMINAMATH_CALUDE_rectangular_field_area_l3231_323197

/-- The area of a rectangular field with given perimeter and width -/
theorem rectangular_field_area
  (perimeter : ℝ) (width : ℝ)
  (h_perimeter : perimeter = 70)
  (h_width : width = 15) :
  width * ((perimeter / 2) - width) = 300 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l3231_323197


namespace NUMINAMATH_CALUDE_company_employees_l3231_323170

/-- Calculates the initial number of employees in a company given the following conditions:
  * Hourly wage
  * Hours worked per day
  * Days worked per week
  * Weeks worked per month
  * Number of new hires
  * Total monthly payroll after hiring
-/
def initial_employees (
  hourly_wage : ℕ
  ) (hours_per_day : ℕ
  ) (days_per_week : ℕ
  ) (weeks_per_month : ℕ
  ) (new_hires : ℕ
  ) (total_payroll : ℕ
  ) : ℕ :=
  let monthly_hours := hours_per_day * days_per_week * weeks_per_month
  let monthly_wage := hourly_wage * monthly_hours
  (total_payroll / monthly_wage) - new_hires

theorem company_employees :
  initial_employees 12 10 5 4 200 1680000 = 500 := by
  sorry

end NUMINAMATH_CALUDE_company_employees_l3231_323170


namespace NUMINAMATH_CALUDE_sign_determination_l3231_323168

theorem sign_determination (a b : ℝ) (h1 : a + b < 0) (h2 : b / a > 0) : a < 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_determination_l3231_323168


namespace NUMINAMATH_CALUDE_maisy_earns_fifteen_more_l3231_323195

/-- Represents Maisy's job options and calculates the difference in earnings -/
def maisys_job_earnings_difference : ℝ :=
  let current_hours : ℝ := 8
  let current_wage : ℝ := 10
  let new_hours : ℝ := 4
  let new_wage : ℝ := 15
  let bonus : ℝ := 35
  let current_earnings := current_hours * current_wage
  let new_earnings := new_hours * new_wage + bonus
  new_earnings - current_earnings

/-- Theorem stating that Maisy will earn $15 more per week at her new job -/
theorem maisy_earns_fifteen_more : maisys_job_earnings_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_maisy_earns_fifteen_more_l3231_323195


namespace NUMINAMATH_CALUDE_tangent_sum_difference_l3231_323165

theorem tangent_sum_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) : 
  Real.tan (α + π/4) = 3/22 := by
sorry

end NUMINAMATH_CALUDE_tangent_sum_difference_l3231_323165


namespace NUMINAMATH_CALUDE_inventory_difference_l3231_323187

/-- Inventory problem -/
theorem inventory_difference (ties belts black_shirts white_shirts : ℕ) 
  (h_ties : ties = 34)
  (h_belts : belts = 40)
  (h_black_shirts : black_shirts = 63)
  (h_white_shirts : white_shirts = 42)
  : (2 * (black_shirts + white_shirts) / 3) - ((ties + belts) / 2) = 33 := by
  sorry

end NUMINAMATH_CALUDE_inventory_difference_l3231_323187


namespace NUMINAMATH_CALUDE_count_five_digit_with_four_or_five_l3231_323175

/-- The number of five-digit positive integers. -/
def total_five_digit_integers : ℕ := 90000

/-- The number of five-digit positive integers without 4 or 5. -/
def five_digit_without_four_or_five : ℕ := 28672

/-- The number of five-digit positive integers containing either 4 or 5 at least once. -/
def five_digit_with_four_or_five : ℕ := total_five_digit_integers - five_digit_without_four_or_five

theorem count_five_digit_with_four_or_five :
  five_digit_with_four_or_five = 61328 := by
  sorry

end NUMINAMATH_CALUDE_count_five_digit_with_four_or_five_l3231_323175


namespace NUMINAMATH_CALUDE_product_148_152_l3231_323140

theorem product_148_152 : 148 * 152 = 22496 := by
  sorry

end NUMINAMATH_CALUDE_product_148_152_l3231_323140


namespace NUMINAMATH_CALUDE_train_speed_l3231_323171

/-- Proves that the current average speed of a train is 48 kmph given the specified conditions -/
theorem train_speed (distance : ℝ) : 
  (distance = (50 / 60) * 48) → 
  (distance = (40 / 60) * 60) → 
  48 = (60 * 40) / 50 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3231_323171


namespace NUMINAMATH_CALUDE_complex_roots_quadratic_l3231_323138

theorem complex_roots_quadratic (p q : ℝ) : 
  (p + 3*I : ℂ) * (p + 3*I : ℂ) - (12 + 11*I : ℂ) * (p + 3*I : ℂ) + (9 + 63*I : ℂ) = 0 ∧
  (q + 6*I : ℂ) * (q + 6*I : ℂ) - (12 + 11*I : ℂ) * (q + 6*I : ℂ) + (9 + 63*I : ℂ) = 0 →
  p = 9 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_quadratic_l3231_323138


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3231_323131

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a n > 0 ∧ a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 1 * a 9 = 16 → a 2 * a 5 * a 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3231_323131


namespace NUMINAMATH_CALUDE_initial_pens_count_l3231_323104

theorem initial_pens_count (P : ℕ) : P = 5 :=
  by
  have h1 : 2 * (P + 20) - 19 = 31 := by sorry
  sorry

end NUMINAMATH_CALUDE_initial_pens_count_l3231_323104


namespace NUMINAMATH_CALUDE_dinosaur_weight_theorem_l3231_323110

/-- The weight of a regular dinosaur in pounds -/
def regular_dino_weight : ℕ := 800

/-- The number of regular dinosaurs -/
def num_regular_dinos : ℕ := 5

/-- The additional weight of Barney compared to the combined weight of regular dinosaurs -/
def barney_extra_weight : ℕ := 1500

/-- The combined weight of Barney and the regular dinosaurs -/
def total_weight : ℕ := regular_dino_weight * num_regular_dinos + barney_extra_weight + regular_dino_weight * num_regular_dinos

theorem dinosaur_weight_theorem : total_weight = 9500 := by
  sorry

end NUMINAMATH_CALUDE_dinosaur_weight_theorem_l3231_323110


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3231_323108

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_even_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 30)
  (h_odd_sum : a 1 + a 3 + a 5 + a 7 + a 9 = 25) :
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3231_323108


namespace NUMINAMATH_CALUDE_g_property_S_sum_S_difference_l3231_323183

def g (k : ℕ+) : ℕ+ :=
  sorry

def S (n : ℕ) : ℕ :=
  sorry

theorem g_property (m : ℕ+) : g (2 * m) = g m :=
  sorry

theorem S_sum : S 1 + S 2 + S 3 = 30 :=
  sorry

theorem S_difference (n : ℕ) (h : n ≥ 2) : S n - S (n - 1) = 4^(n - 1) :=
  sorry

end NUMINAMATH_CALUDE_g_property_S_sum_S_difference_l3231_323183


namespace NUMINAMATH_CALUDE_sunzi_carriage_problem_l3231_323114

theorem sunzi_carriage_problem (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x / 3 = y + 2 ∧ x / 2 + 9 = y) ↔ (x / 3 = y - 2 ∧ (x - 9) / 2 = y) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_carriage_problem_l3231_323114


namespace NUMINAMATH_CALUDE_percentage_equals_1000_l3231_323126

theorem percentage_equals_1000 (x : ℝ) (p : ℝ) : 
  (p / 100) * x = 1000 → 
  (120 / 100) * x = 6000 → 
  p = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_equals_1000_l3231_323126


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l3231_323141

theorem largest_n_for_equation : ∃ (x y z : ℕ+), 
  9^2 = 2*x^2 + 2*y^2 + 2*z^2 + 4*x*y + 4*y*z + 4*z*x + 6*x + 6*y + 6*z - 14 ∧ 
  ∀ (n : ℕ+), n > 9 → ¬∃ (a b c : ℕ+), 
    n^2 = 2*a^2 + 2*b^2 + 2*c^2 + 4*a*b + 4*b*c + 4*c*a + 6*a + 6*b + 6*c - 14 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l3231_323141


namespace NUMINAMATH_CALUDE_pool_capacity_theorem_l3231_323122

/-- Represents the dimensions and draining parameters of a pool -/
structure Pool :=
  (width : ℝ)
  (length : ℝ)
  (depth : ℝ)
  (drainRate : ℝ)
  (drainTime : ℝ)

/-- Calculates the volume of a pool -/
def poolVolume (p : Pool) : ℝ :=
  p.width * p.length * p.depth

/-- Calculates the amount of water drained from a pool -/
def waterDrained (p : Pool) : ℝ :=
  p.drainRate * p.drainTime

/-- Theorem stating that if the water drained equals the pool volume, 
    then the pool was at 100% capacity -/
theorem pool_capacity_theorem (p : Pool) 
  (h1 : p.width = 80)
  (h2 : p.length = 150)
  (h3 : p.depth = 10)
  (h4 : p.drainRate = 60)
  (h5 : p.drainTime = 2000)
  (h6 : waterDrained p = poolVolume p) :
  poolVolume p / poolVolume p = 1 := by
  sorry


end NUMINAMATH_CALUDE_pool_capacity_theorem_l3231_323122


namespace NUMINAMATH_CALUDE_largest_replacement_l3231_323191

def original_number : ℚ := -0.3168

def replace_digit (n : ℚ) (old_digit new_digit : ℕ) : ℚ := sorry

theorem largest_replacement :
  ∀ d : ℕ, d ≠ 0 → d ≠ 3 → d ≠ 1 → d ≠ 6 → d ≠ 8 →
    replace_digit original_number 6 4 ≥ replace_digit original_number d 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_replacement_l3231_323191


namespace NUMINAMATH_CALUDE_monster_count_is_thirteen_l3231_323102

/-- Represents the state of the battlefield --/
structure Battlefield where
  ultraman_heads : Nat
  ultraman_legs : Nat
  initial_monster_heads : Nat
  initial_monster_legs : Nat
  split_monster_heads : Nat
  split_monster_legs : Nat
  total_heads : Nat
  total_legs : Nat

/-- Calculates the number of monsters on the battlefield --/
def count_monsters (b : Battlefield) : Nat :=
  let remaining_heads := b.total_heads - b.ultraman_heads
  let remaining_legs := b.total_legs - b.ultraman_legs
  let initial_monsters := remaining_heads / b.initial_monster_heads
  let extra_legs := remaining_legs - (initial_monsters * b.initial_monster_legs)
  let splits := extra_legs / (2 * b.split_monster_legs - b.initial_monster_legs)
  initial_monsters + splits

/-- The main theorem stating that the number of monsters is 13 --/
theorem monster_count_is_thirteen (b : Battlefield) 
  (h1 : b.ultraman_heads = 1)
  (h2 : b.ultraman_legs = 2)
  (h3 : b.initial_monster_heads = 2)
  (h4 : b.initial_monster_legs = 5)
  (h5 : b.split_monster_heads = 1)
  (h6 : b.split_monster_legs = 6)
  (h7 : b.total_heads = 21)
  (h8 : b.total_legs = 73) :
  count_monsters b = 13 := by
  sorry

#eval count_monsters {
  ultraman_heads := 1,
  ultraman_legs := 2,
  initial_monster_heads := 2,
  initial_monster_legs := 5,
  split_monster_heads := 1,
  split_monster_legs := 6,
  total_heads := 21,
  total_legs := 73
}

end NUMINAMATH_CALUDE_monster_count_is_thirteen_l3231_323102


namespace NUMINAMATH_CALUDE_shepherd_problem_l3231_323158

theorem shepherd_problem :
  ∃! (a b c : ℕ), 
    a + b + 10 * c = 100 ∧
    20 * a + 10 * b + 10 * c = 200 ∧
    a = 1 ∧ b = 9 ∧ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_shepherd_problem_l3231_323158


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3231_323136

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 2 = 1 →
  a 3 + a 4 = 4 →
  a 5 + a 6 + a 7 + a 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3231_323136


namespace NUMINAMATH_CALUDE_chess_pawn_loss_l3231_323152

theorem chess_pawn_loss (total_pawns_start : ℕ) (pawns_per_player : ℕ) 
  (kennedy_lost : ℕ) (pawns_left : ℕ) : 
  total_pawns_start = 2 * pawns_per_player →
  pawns_per_player = 8 →
  kennedy_lost = 4 →
  pawns_left = 11 →
  pawns_per_player - (pawns_left - (pawns_per_player - kennedy_lost)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_chess_pawn_loss_l3231_323152


namespace NUMINAMATH_CALUDE_ceiling_evaluation_l3231_323124

theorem ceiling_evaluation : ⌈(4 * (8 - 1/3 : ℚ))⌉ = 31 := by sorry

end NUMINAMATH_CALUDE_ceiling_evaluation_l3231_323124


namespace NUMINAMATH_CALUDE_difference_largest_smallest_l3231_323105

/-- The set of available digits --/
def available_digits : Finset Nat := {2, 0, 3, 5, 8}

/-- A four-digit number formed from the available digits --/
structure FourDigitNumber where
  digits : Finset Nat
  size_eq : digits.card = 4
  subset : digits ⊆ available_digits

/-- The largest four-digit number that can be formed --/
def largest_number : Nat := 8532

/-- The smallest four-digit number that can be formed --/
def smallest_number : Nat := 2035

/-- Theorem: The difference between the largest and smallest four-digit numbers is 6497 --/
theorem difference_largest_smallest :
  largest_number - smallest_number = 6497 := by sorry

end NUMINAMATH_CALUDE_difference_largest_smallest_l3231_323105


namespace NUMINAMATH_CALUDE_f_composition_equals_pi_plus_two_l3231_323125

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 2
  else if x = 0 then Real.pi
  else 0

-- State the theorem
theorem f_composition_equals_pi_plus_two :
  f (f (f (-2))) = Real.pi + 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_pi_plus_two_l3231_323125


namespace NUMINAMATH_CALUDE_work_completion_time_l3231_323115

/-- Given a piece of work that can be completed by different combinations of workers,
    this theorem proves how long it takes two workers to complete the work. -/
theorem work_completion_time
  (work : ℝ) -- The total amount of work to be done
  (rate_ab : ℝ) -- The rate at which a and b work together
  (rate_c : ℝ) -- The rate at which c works
  (h1 : rate_ab + rate_c = work) -- a, b, and c together complete the work in 1 day
  (h2 : rate_c = work / 2) -- c alone completes the work in 2 days
  : rate_ab = work / 2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3231_323115


namespace NUMINAMATH_CALUDE_kite_parabolas_sum_l3231_323133

/-- Given two parabolas that intersect the coordinate axes in four points forming a kite -/
structure KiteParabolas where
  /-- Coefficient of x^2 in the first parabola y = ax^2 - 3 -/
  a : ℝ
  /-- Coefficient of x^2 in the second parabola y = 5 - bx^2 -/
  b : ℝ
  /-- The four intersection points form a kite -/
  is_kite : Bool
  /-- The area of the kite formed by the intersection points -/
  kite_area : ℝ
  /-- The parabolas intersect the coordinate axes in exactly four points -/
  four_intersections : Bool

/-- Theorem stating that under the given conditions, a + b = 128/81 -/
theorem kite_parabolas_sum (k : KiteParabolas) 
  (h1 : k.is_kite = true) 
  (h2 : k.kite_area = 18) 
  (h3 : k.four_intersections = true) : 
  k.a + k.b = 128/81 := by
  sorry

end NUMINAMATH_CALUDE_kite_parabolas_sum_l3231_323133


namespace NUMINAMATH_CALUDE_chords_and_triangles_10_points_l3231_323167

/-- The number of chords formed by n points on a circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- The number of triangles formed by n points on a circumference -/
def num_triangles (n : ℕ) : ℕ := n.choose 3

/-- Theorem about chords and triangles formed by 10 points on a circumference -/
theorem chords_and_triangles_10_points :
  num_chords 10 = 45 ∧ num_triangles 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_chords_and_triangles_10_points_l3231_323167


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3231_323161

/-- A quadratic function with zeros at -2 and 3 -/
def f (x : ℝ) : ℝ := x^2 + a*x + b
  where
  a : ℝ := -1  -- Derived from the zeros, but not explicitly using the solution
  b : ℝ := -6  -- Derived from the zeros, but not explicitly using the solution

/-- The theorem statement -/
theorem solution_set_of_inequality (x : ℝ) :
  (f (-2*x) * (-1) > 0) ↔ (-3/2 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3231_323161


namespace NUMINAMATH_CALUDE_three_to_nine_over_nine_cubed_equals_27_l3231_323111

theorem three_to_nine_over_nine_cubed_equals_27 : (3^9) / (9^3) = 27 := by
  sorry

end NUMINAMATH_CALUDE_three_to_nine_over_nine_cubed_equals_27_l3231_323111


namespace NUMINAMATH_CALUDE_shell_collection_sum_l3231_323148

/-- The sum of an arithmetic sequence with first term 2, common difference 3, and 15 terms -/
def shell_sum : ℕ := 
  let a₁ : ℕ := 2  -- first term
  let d : ℕ := 3   -- common difference
  let n : ℕ := 15  -- number of terms
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Theorem stating that the sum of shells collected is 345 -/
theorem shell_collection_sum : shell_sum = 345 := by
  sorry

end NUMINAMATH_CALUDE_shell_collection_sum_l3231_323148


namespace NUMINAMATH_CALUDE_base3_20121_equals_178_l3231_323163

/-- Converts a base 3 number to base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base3_20121_equals_178 : 
  base3ToBase10 [2, 0, 1, 2, 1] = 178 := by
  sorry

end NUMINAMATH_CALUDE_base3_20121_equals_178_l3231_323163


namespace NUMINAMATH_CALUDE_profit_difference_is_183_50_l3231_323113

-- Define the given quantities
def cat_food_packages : ℕ := 9
def dog_food_packages : ℕ := 7
def cans_per_cat_package : ℕ := 15
def cans_per_dog_package : ℕ := 8
def cost_per_cat_package : ℚ := 14
def cost_per_dog_package : ℚ := 10
def price_per_cat_can : ℚ := 2.5
def price_per_dog_can : ℚ := 1.75

-- Define the profit calculation function
def profit_difference : ℚ :=
  let cat_revenue := (cat_food_packages * cans_per_cat_package : ℚ) * price_per_cat_can
  let dog_revenue := (dog_food_packages * cans_per_dog_package : ℚ) * price_per_dog_can
  let cat_cost := (cat_food_packages : ℚ) * cost_per_cat_package
  let dog_cost := (dog_food_packages : ℚ) * cost_per_dog_package
  (cat_revenue - cat_cost) - (dog_revenue - dog_cost)

-- Theorem statement
theorem profit_difference_is_183_50 : profit_difference = 183.5 := by sorry

end NUMINAMATH_CALUDE_profit_difference_is_183_50_l3231_323113


namespace NUMINAMATH_CALUDE_hollow_sphere_weight_double_radius_l3231_323132

/-- The weight of a hollow sphere given its radius -/
noncomputable def sphereWeight (r : ℝ) : ℝ :=
  4 * Real.pi * r^2

theorem hollow_sphere_weight_double_radius (r : ℝ) (h : r > 0) :
  sphereWeight r = 8 → sphereWeight (2 * r) = 32 := by
  sorry

end NUMINAMATH_CALUDE_hollow_sphere_weight_double_radius_l3231_323132


namespace NUMINAMATH_CALUDE_problem_statement_l3231_323139

theorem problem_statement (a b c : ℝ) 
  (h1 : a^2 + b^2 - 4*a ≤ 1)
  (h2 : b^2 + c^2 - 8*b ≤ -3)
  (h3 : c^2 + a^2 - 12*c ≤ -26) :
  (a + b)^c = 27 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3231_323139


namespace NUMINAMATH_CALUDE_spatial_relationships_l3231_323153

structure Space3D where
  Point : Type
  Line : Type
  Plane : Type
  intersects : Line → Plane → Prop
  parallel : Line → Line → Prop
  parallel_line_plane : Line → Plane → Prop
  perpendicular : Line → Line → Prop
  in_plane : Line → Plane → Prop

variable (S : Space3D)

theorem spatial_relationships :
  (∀ (a : S.Line) (α : S.Plane), S.intersects a α → ¬∃ (l : S.Line), S.in_plane l α ∧ S.parallel l a) ∧
  (∃ (a b : S.Line) (α : S.Plane), S.parallel_line_plane b α ∧ S.perpendicular a b ∧ S.parallel_line_plane a α) ∧
  (∃ (a b : S.Line) (α : S.Plane), S.parallel a b ∧ S.in_plane b α ∧ ¬S.parallel_line_plane a α) :=
by sorry

end NUMINAMATH_CALUDE_spatial_relationships_l3231_323153


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l3231_323189

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ∧
    C = 2 / Real.sqrt 3 ∧
    ∀ (D : ℝ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 1 ≥ D * (x + y + z)) → D ≤ C :=
by
  sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l3231_323189


namespace NUMINAMATH_CALUDE_triangle_inequality_with_120_degree_angle_l3231_323166

/-- Given a triangle with sides a, b, and c, where an angle of 120 degrees lies opposite to side c,
    prove that a, c, and a + b satisfy the triangle inequality theorem. -/
theorem triangle_inequality_with_120_degree_angle 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (triangle_exists : a + b > c ∧ b + c > a ∧ c + a > b) 
  (angle_120 : a^2 = b^2 + c^2 - b*c) : 
  a + c > a + b ∧ a + (a + b) > c ∧ c + (a + b) > a :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_120_degree_angle_l3231_323166


namespace NUMINAMATH_CALUDE_house_coloring_l3231_323164

/-- A type representing the colors of houses -/
inductive Color
| Blue
| Green
| Red

/-- A function representing the move of residents between houses -/
def move (n : ℕ) : ℕ → ℕ :=
  sorry

/-- A function representing the coloring of houses -/
def color (n : ℕ) : ℕ → Color :=
  sorry

/-- The main theorem -/
theorem house_coloring (n : ℕ) (h_pos : 0 < n) :
  ∃ (move : ℕ → ℕ) (color : ℕ → Color),
    (∀ i : ℕ, i < n → move i < n) ∧  -- Each person moves to a valid house
    (∀ i j : ℕ, i < n → j < n → i ≠ j → move i ≠ move j) ∧  -- No two people move to the same house
    (∀ i : ℕ, i < n → move (move i) ≠ i) ∧  -- No person returns to their original house
    (∀ i : ℕ, i < n → color i ≠ color (move i)) :=  -- No person's new house has the same color as their old house
  sorry

#check house_coloring 1000

end NUMINAMATH_CALUDE_house_coloring_l3231_323164


namespace NUMINAMATH_CALUDE_min_guests_football_banquet_l3231_323186

theorem min_guests_football_banquet (total_food : ℕ) (max_per_guest : ℕ) 
  (h1 : total_food = 325)
  (h2 : max_per_guest = 2) :
  (total_food + max_per_guest - 1) / max_per_guest = 163 := by
  sorry

end NUMINAMATH_CALUDE_min_guests_football_banquet_l3231_323186


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3231_323157

theorem inequality_solution_set (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 →
  (∀ x, x^2 - a*x - 6*a^2 < 0 ↔ x₁ < x ∧ x < x₂) →
  x₂ - x₁ = 10 →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3231_323157


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_one_range_of_m_for_solution_l3231_323143

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - |x - 1|

-- Theorem for the solution set of f(x) > 1
theorem solution_set_f_greater_than_one :
  {x : ℝ | f x > 1} = {x : ℝ | x > 0} := by sorry

-- Theorem for the range of m
theorem range_of_m_for_solution (m : ℝ) :
  (∃ x : ℝ, f x + 4 ≥ |1 - 2*m|) ↔ m ∈ Set.Icc (-3) 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_one_range_of_m_for_solution_l3231_323143


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_2021_l3231_323128

theorem tens_digit_of_13_pow_2021 : ∃ n : ℕ, 13^2021 ≡ 10 * n + 1 [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_2021_l3231_323128


namespace NUMINAMATH_CALUDE_function_equation_solution_l3231_323134

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = -x - 1) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l3231_323134


namespace NUMINAMATH_CALUDE_coefficient_is_200_l3231_323120

/-- The coefficient of x^4 in the expansion of (1+x^3)(1-x)^10 -/
def coefficientOfX4 : ℕ :=
  (Nat.choose 10 4) - (Nat.choose 10 1)

/-- Theorem stating that the coefficient of x^4 in the expansion of (1+x^3)(1-x)^10 is 200 -/
theorem coefficient_is_200 : coefficientOfX4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_is_200_l3231_323120


namespace NUMINAMATH_CALUDE_bookshop_unsold_percentage_l3231_323103

/-- The percentage of unsold books in a bookshop -/
def unsold_percentage (initial_stock : ℕ) (mon_sales tues_sales wed_sales thurs_sales fri_sales : ℕ) : ℚ :=
  (initial_stock - (mon_sales + tues_sales + wed_sales + thurs_sales + fri_sales)) / initial_stock * 100

/-- Theorem stating the percentage of unsold books for the given scenario -/
theorem bookshop_unsold_percentage :
  unsold_percentage 1300 75 50 64 78 135 = 69.15384615384615 := by
  sorry

end NUMINAMATH_CALUDE_bookshop_unsold_percentage_l3231_323103


namespace NUMINAMATH_CALUDE_jerry_added_eleven_action_figures_l3231_323118

/-- The number of action figures Jerry added to his shelf -/
def action_figures_added (initial : ℕ) (removed : ℕ) (final : ℕ) : ℤ :=
  final - initial + removed

/-- Proof that Jerry added 11 action figures to his shelf -/
theorem jerry_added_eleven_action_figures :
  action_figures_added 7 10 8 = 11 := by
  sorry

end NUMINAMATH_CALUDE_jerry_added_eleven_action_figures_l3231_323118


namespace NUMINAMATH_CALUDE_inverse_statement_l3231_323190

theorem inverse_statement : 
  (∀ x : ℝ, x > 1 → x^2 - 2*x + 3 > 0) ↔ 
  (∀ x : ℝ, x^2 - 2*x + 3 > 0 → x > 1) :=
by sorry

end NUMINAMATH_CALUDE_inverse_statement_l3231_323190


namespace NUMINAMATH_CALUDE_eliot_account_balance_l3231_323135

/-- Proves that Eliot's account balance is $200 given the problem conditions --/
theorem eliot_account_balance :
  ∀ (A E : ℝ),
    A > E →  -- Al has more money than Eliot
    A - E = (1/12) * (A + E) →  -- Difference is 1/12 of sum
    1.1 * A = 1.2 * E + 20 →  -- After increase, Al has $20 more
    E = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_eliot_account_balance_l3231_323135


namespace NUMINAMATH_CALUDE_coefficient_of_x_l3231_323155

/-- Given that for some natural number n:
    1) M = 4^n is the sum of coefficients in (5x - 1/√x)^n
    2) N = 2^n is the sum of binomial coefficients
    3) M - N = 240
    Then the coefficient of x in the expansion of (5x - 1/√x)^n is 150 -/
theorem coefficient_of_x (n : ℕ) (M N : ℝ) 
  (hM : M = 4^n)
  (hN : N = 2^n)
  (hDiff : M - N = 240) :
  ∃ (coeff : ℝ), coeff = 150 ∧ 
  coeff = (-1)^2 * (n.choose 2) * 5^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l3231_323155


namespace NUMINAMATH_CALUDE_shaded_area_between_tangent_circles_l3231_323160

theorem shaded_area_between_tangent_circles 
  (r₁ : ℝ) (r₂ : ℝ) (d : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) (h₃ : d = 4) :
  let area_shaded := π * r₂^2 - π * r₁^2
  area_shaded = 48 * π := by
sorry

end NUMINAMATH_CALUDE_shaded_area_between_tangent_circles_l3231_323160


namespace NUMINAMATH_CALUDE_max_volume_box_l3231_323172

/-- The maximum volume of a box created from a rectangular metal sheet --/
theorem max_volume_box (sheet_length sheet_width : ℝ) (h_length : sheet_length = 16)
  (h_width : sheet_width = 12) :
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < sheet_length / 2 ∧ 
    x < sheet_width / 2 ∧
    ∀ (y : ℝ), 
      0 < y ∧ 
      y < sheet_length / 2 ∧ 
      y < sheet_width / 2 → 
      y * (sheet_length - 2*y) * (sheet_width - 2*y) ≤ 128 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_box_l3231_323172


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3231_323188

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 6 * x + c = 0) →  -- exactly one solution
  (a + c = 12) →                     -- sum condition
  (a < c) →                          -- order condition
  (a = 6 - 3 * Real.sqrt 3 ∧ c = 6 + 3 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l3231_323188


namespace NUMINAMATH_CALUDE_expression_value_l3231_323130

theorem expression_value (a b : ℝ) (h : a * b > 0) :
  (a / abs a) + (b / abs b) + ((a * b) / abs (a * b)) = 3 ∨
  (a / abs a) + (b / abs b) + ((a * b) / abs (a * b)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3231_323130


namespace NUMINAMATH_CALUDE_cab_journey_time_l3231_323142

/-- Given a cab walking at 5/6 of its usual speed and arriving 8 minutes late,
    prove that its usual time to cover the journey is 40 minutes. -/
theorem cab_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (5 / 6 * usual_speed) * (usual_time + 8) = usual_speed * usual_time → 
  usual_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_cab_journey_time_l3231_323142


namespace NUMINAMATH_CALUDE_impossible_coloring_l3231_323169

theorem impossible_coloring : ¬∃(color : ℕ → Bool),
  (∀ n : ℕ, color n ≠ color (n + 5)) ∧
  (∀ n : ℕ, color n ≠ color (2 * n)) :=
by sorry

end NUMINAMATH_CALUDE_impossible_coloring_l3231_323169


namespace NUMINAMATH_CALUDE_minimum_time_for_assessment_l3231_323129

/-- Represents the minimum time needed to assess students -/
def minimum_assessment_time (
  teacher1_problem_solving_time : ℕ)
  (teacher1_theory_time : ℕ)
  (teacher2_problem_solving_time : ℕ)
  (teacher2_theory_time : ℕ)
  (total_students : ℕ) : ℕ :=
  110

/-- Theorem stating the minimum time needed to assess 25 students
    given the specified conditions -/
theorem minimum_time_for_assessment :
  minimum_assessment_time 5 7 3 4 25 = 110 := by
  sorry

end NUMINAMATH_CALUDE_minimum_time_for_assessment_l3231_323129


namespace NUMINAMATH_CALUDE_radiator_problem_l3231_323180

/-- Represents the fraction of original substance remaining after multiple replacements -/
def fractionRemaining (totalVolume : ℚ) (replacementVolume : ℚ) (numberOfReplacements : ℕ) : ℚ :=
  (1 - replacementVolume / totalVolume) ^ numberOfReplacements

/-- The radiator problem -/
theorem radiator_problem :
  let totalVolume : ℚ := 25
  let replacementVolume : ℚ := 5
  let numberOfReplacements : ℕ := 3
  fractionRemaining totalVolume replacementVolume numberOfReplacements = 64 / 125 := by
  sorry

end NUMINAMATH_CALUDE_radiator_problem_l3231_323180


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3231_323150

theorem quadratic_roots_property (α β : ℝ) : 
  (α^2 - 4*α - 3 = 0) → 
  (β^2 - 4*β - 3 = 0) → 
  (α - 3) * (β - 3) = -6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3231_323150


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3231_323101

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 6) % 12 = 0 ∧
  (n - 6) % 16 = 0 ∧
  (n - 6) % 18 = 0 ∧
  (n - 6) % 21 = 0 ∧
  (n - 6) % 28 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 1014 ∧
  ∀ m : ℕ, m < 1014 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3231_323101


namespace NUMINAMATH_CALUDE_complex_number_magnitude_squared_l3231_323137

theorem complex_number_magnitude_squared (z : ℂ) : z + Complex.abs z = 2 + 8*I → Complex.abs z ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_squared_l3231_323137


namespace NUMINAMATH_CALUDE_team_selection_problem_l3231_323159

def num_players : ℕ := 6
def team_size : ℕ := 3

def ways_to_select (n k : ℕ) : ℕ := (n.factorial) / ((n - k).factorial)

theorem team_selection_problem :
  ways_to_select num_players team_size - ways_to_select (num_players - 1) (team_size - 1) = 100 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_problem_l3231_323159


namespace NUMINAMATH_CALUDE_mani_pedi_regular_price_l3231_323151

/-- The regular price of a mani/pedi, given a 25% discount, 5 purchases, and $150 total spent. -/
theorem mani_pedi_regular_price :
  ∀ (regular_price : ℝ),
  (regular_price * 0.75 * 5 = 150) →
  regular_price = 40 := by
sorry

end NUMINAMATH_CALUDE_mani_pedi_regular_price_l3231_323151


namespace NUMINAMATH_CALUDE_second_number_problem_l3231_323194

theorem second_number_problem (A B : ℝ) : 
  A = 580 → 0.20 * A = 0.30 * B + 80 → B = 120 := by
sorry

end NUMINAMATH_CALUDE_second_number_problem_l3231_323194


namespace NUMINAMATH_CALUDE_test_modes_l3231_323176

/-- Represents the frequency of each score in the test --/
def score_frequency : List (Nat × Nat) := [
  (65, 2), (73, 1), (82, 1), (88, 1),
  (91, 1), (96, 4), (102, 1), (104, 4), (110, 3)
]

/-- Finds the modes of a list of score frequencies --/
def find_modes (frequencies : List (Nat × Nat)) : List Nat :=
  sorry

/-- States that 96 and 104 are the modes of the given score frequencies --/
theorem test_modes : find_modes score_frequency = [96, 104] := by
  sorry

end NUMINAMATH_CALUDE_test_modes_l3231_323176


namespace NUMINAMATH_CALUDE_union_A_complement_B_equals_result_l3231_323154

-- Define the set I
def I : Set ℤ := {x | |x| < 3}

-- Define set A
def A : Set ℤ := {1, 2}

-- Define set B
def B : Set ℤ := {-2, -1, 2}

-- Theorem statement
theorem union_A_complement_B_equals_result : A ∪ (I \ B) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_A_complement_B_equals_result_l3231_323154


namespace NUMINAMATH_CALUDE_line_l_properties_l3231_323145

/-- A line that passes through (3,2) and has equal intercepts on both axes -/
def line_l (x y : ℝ) : Prop :=
  y = -x + 5

theorem line_l_properties :
  (∃ a : ℝ, line_l a 2 ∧ a = 3) ∧
  (∃ b : ℝ, line_l b 0 ∧ line_l 0 b ∧ b > 0) :=
by sorry

end NUMINAMATH_CALUDE_line_l_properties_l3231_323145


namespace NUMINAMATH_CALUDE_percentage_four_leaf_clovers_l3231_323182

/-- Proves that 20% of clovers have four leaves given the conditions -/
theorem percentage_four_leaf_clovers 
  (total_clovers : ℕ) 
  (purple_four_leaf : ℕ) 
  (h1 : total_clovers = 500)
  (h2 : purple_four_leaf = 25)
  (h3 : (4 : ℚ) * purple_four_leaf = total_clovers * (percentage_four_leaf / 100)) :
  percentage_four_leaf = 20 := by
  sorry

#check percentage_four_leaf_clovers

end NUMINAMATH_CALUDE_percentage_four_leaf_clovers_l3231_323182


namespace NUMINAMATH_CALUDE_cantor_set_removal_operations_l3231_323184

theorem cantor_set_removal_operations (n : ℕ) : 
  (((2 : ℝ) / 3) ^ (n - 1) * (1 / 3) ≥ 1 / 60) ↔ n ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_cantor_set_removal_operations_l3231_323184
