import Mathlib

namespace NUMINAMATH_CALUDE_work_completion_time_l352_35252

/-- Given a work that B can complete in 10 days, and when A and B work together,
    B's share of the total 5000 Rs wages is 3333 Rs, prove that A alone can do the work in 20 days. -/
theorem work_completion_time
  (b_time : ℝ)
  (total_wages : ℝ)
  (b_wages : ℝ)
  (h1 : b_time = 10)
  (h2 : total_wages = 5000)
  (h3 : b_wages = 3333)
  : ∃ (a_time : ℝ), a_time = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l352_35252


namespace NUMINAMATH_CALUDE_product_correction_l352_35223

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

theorem product_correction (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (reverse_digits a * b = 182) →  -- reversed a multiplied by b is 182
  (a * b = 533) :=  -- the correct product is 533
by sorry

end NUMINAMATH_CALUDE_product_correction_l352_35223


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l352_35217

theorem subtraction_preserves_inequality (a b : ℝ) : a > b → a - 1 > b - 1 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l352_35217


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l352_35269

theorem least_subtraction_for_divisibility :
  ∃ (x : ℕ), x = 1 ∧ 
  (5026 - x) % 5 = 0 ∧ 
  ∀ (y : ℕ), y < x → (5026 - y) % 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l352_35269


namespace NUMINAMATH_CALUDE_complex_equation_solution_l352_35267

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l352_35267


namespace NUMINAMATH_CALUDE_eleven_days_sufficiency_l352_35247

/-- Represents the amount of cat food in a package -/
structure CatFood where
  days : ℝ
  nonneg : days ≥ 0

/-- The amount of food in a large package -/
def large_package : CatFood := sorry

/-- The amount of food in a small package -/
def small_package : CatFood := sorry

/-- One large package and four small packages last for 14 days -/
axiom package_combination : large_package.days + 4 * small_package.days = 14

theorem eleven_days_sufficiency :
  large_package.days + 3 * small_package.days ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_days_sufficiency_l352_35247


namespace NUMINAMATH_CALUDE_polynomial_remainder_l352_35253

theorem polynomial_remainder (x : ℝ) : 
  (x^4 - 4*x^2 + 7) % (x - 1) = 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l352_35253


namespace NUMINAMATH_CALUDE_expression_evaluation_l352_35278

theorem expression_evaluation :
  let a : ℤ := -1
  (2 - a)^2 - (1 + a)*(a - 1) - a*(a - 3) = 5 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l352_35278


namespace NUMINAMATH_CALUDE_division_of_fractions_l352_35251

theorem division_of_fractions : 
  (5 / 6 : ℚ) / (7 / 9 : ℚ) / (11 / 13 : ℚ) = 195 / 154 := by sorry

end NUMINAMATH_CALUDE_division_of_fractions_l352_35251


namespace NUMINAMATH_CALUDE_initial_alloy_weight_l352_35297

/-- Represents the composition of an alloy --/
structure Alloy where
  zinc : ℝ
  copper : ℝ

/-- The initial ratio of zinc to copper in the alloy --/
def initial_ratio : ℚ := 5 / 3

/-- The final ratio of zinc to copper after adding zinc --/
def final_ratio : ℚ := 3 / 1

/-- The amount of zinc added to the alloy --/
def added_zinc : ℝ := 8

/-- Theorem stating the initial weight of the alloy --/
theorem initial_alloy_weight (a : Alloy) :
  (a.zinc / a.copper = initial_ratio) →
  ((a.zinc + added_zinc) / a.copper = final_ratio) →
  (a.zinc + a.copper = 16) :=
by sorry

end NUMINAMATH_CALUDE_initial_alloy_weight_l352_35297


namespace NUMINAMATH_CALUDE_solve_system_l352_35233

theorem solve_system (x y : ℝ) 
  (eq1 : 2 * x - y = 5) 
  (eq2 : x + 2 * y = 5) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l352_35233


namespace NUMINAMATH_CALUDE_absolute_value_theorem_l352_35280

theorem absolute_value_theorem (a : ℝ) (h : a = -1) : |a + 3| = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_theorem_l352_35280


namespace NUMINAMATH_CALUDE_complement_to_set_l352_35249

def U : Set ℤ := {-1, 0, 1, 2, 4}

theorem complement_to_set (M : Set ℤ) (h : {x : ℤ | x ∈ U ∧ x ∉ M} = {-1, 1}) : 
  M = {0, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_to_set_l352_35249


namespace NUMINAMATH_CALUDE_farm_animal_count_l352_35237

/-- Represents the distribution of animals on a farm --/
structure FarmDistribution where
  chicken_coops : List Nat
  duck_coops : List Nat
  geese_coop : Nat
  quail_coop : Nat
  turkey_coops : List Nat
  cow_sheds : List Nat
  pig_sections : List Nat

/-- Calculates the total number of animals on the farm --/
def total_animals (farm : FarmDistribution) : Nat :=
  (farm.chicken_coops.sum + farm.duck_coops.sum + farm.geese_coop + 
   farm.quail_coop + farm.turkey_coops.sum + farm.cow_sheds.sum + 
   farm.pig_sections.sum)

/-- Theorem stating that the total number of animals on the farm is 431 --/
theorem farm_animal_count (farm : FarmDistribution) 
  (h1 : farm.chicken_coops = [60, 45, 55])
  (h2 : farm.duck_coops = [40, 35])
  (h3 : farm.geese_coop = 20)
  (h4 : farm.quail_coop = 50)
  (h5 : farm.turkey_coops = [10, 10])
  (h6 : farm.cow_sheds = [20, 10, 6])
  (h7 : farm.pig_sections = [15, 25, 30, 0]) :
  total_animals farm = 431 := by
  sorry

#eval total_animals {
  chicken_coops := [60, 45, 55],
  duck_coops := [40, 35],
  geese_coop := 20,
  quail_coop := 50,
  turkey_coops := [10, 10],
  cow_sheds := [20, 10, 6],
  pig_sections := [15, 25, 30, 0]
}

end NUMINAMATH_CALUDE_farm_animal_count_l352_35237


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l352_35285

/-- The eccentricity of a hyperbola passing through the focus of a specific parabola -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) : 
  let parabola := fun x y : ℝ => y^2 = 8*x
  let hyperbola := fun x y : ℝ => x^2/a^2 - y^2 = 1
  let focus : ℝ × ℝ := (2, 0)
  (∀ x y, parabola x y → (x, y) = focus) →
  hyperbola (focus.1) (focus.2) →
  let e := Real.sqrt ((a^2 + a^2) / a^2)
  e = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l352_35285


namespace NUMINAMATH_CALUDE_sears_tower_height_calculation_l352_35222

/-- The height of Burj Khalifa in meters -/
def burj_khalifa_height : ℕ := 830

/-- The difference in height between Burj Khalifa and Sears Tower in meters -/
def height_difference : ℕ := 303

/-- The height of Sears Tower in meters -/
def sears_tower_height : ℕ := burj_khalifa_height - height_difference

theorem sears_tower_height_calculation :
  sears_tower_height = 527 :=
by sorry

end NUMINAMATH_CALUDE_sears_tower_height_calculation_l352_35222


namespace NUMINAMATH_CALUDE_distance_to_cemetery_l352_35213

/-- The distance from the school to the Martyrs' Cemetery in kilometers. -/
def distance : ℝ := 216

/-- The original scheduled time for the journey in minutes. -/
def scheduled_time : ℝ := 180

/-- The time saved in minutes when the bus increases speed by 1/5 after 1 hour. -/
def time_saved_1 : ℝ := 20

/-- The time saved in minutes when the bus increases speed by 1/3 after 72 km. -/
def time_saved_2 : ℝ := 30

/-- The distance traveled at original speed before increasing speed by 1/3. -/
def initial_distance : ℝ := 72

theorem distance_to_cemetery :
  (1 + 1/5) * (scheduled_time - 60 - time_saved_1) = scheduled_time - 60 ∧
  (1 + 1/3) * (scheduled_time - time_saved_2) = scheduled_time ∧
  distance = initial_distance / (1 - 2/3) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_cemetery_l352_35213


namespace NUMINAMATH_CALUDE_triangle_angle_B_l352_35220

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_B (t : Triangle) :
  t.a = 2 ∧ t.b = 3 ∧ t.A = π/4 → t.B = π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l352_35220


namespace NUMINAMATH_CALUDE_katie_earnings_l352_35284

/-- The number of bead necklaces sold -/
def bead_necklaces : ℕ := 4

/-- The number of gem stone necklaces sold -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 3

/-- The total money earned by Katie -/
def total_money : ℕ := (bead_necklaces + gem_necklaces) * necklace_cost

theorem katie_earnings : total_money = 21 := by
  sorry

end NUMINAMATH_CALUDE_katie_earnings_l352_35284


namespace NUMINAMATH_CALUDE_money_left_after_sale_l352_35241

/-- Represents the total revenue from selling items in a store's inventory. -/
def total_revenue (
  category_a_items : ℕ)
  (category_b_items : ℕ)
  (category_c_items : ℕ)
  (category_a_price : ℚ)
  (category_b_price : ℚ)
  (category_c_price : ℚ)
  (category_a_discount : ℚ)
  (category_b_discount : ℚ)
  (category_c_discount : ℚ)
  (category_a_sold_percent : ℚ)
  (category_b_sold_percent : ℚ)
  (category_c_sold_percent : ℚ) : ℚ :=
  (category_a_items : ℚ) * category_a_price * (1 - category_a_discount) * category_a_sold_percent +
  (category_b_items : ℚ) * category_b_price * (1 - category_b_discount) * category_b_sold_percent +
  (category_c_items : ℚ) * category_c_price * (1 - category_c_discount) * category_c_sold_percent

/-- Theorem stating the amount of money left after the sale and paying creditors. -/
theorem money_left_after_sale : 
  total_revenue 1000 700 300 50 75 100 0.8 0.7 0.6 0.85 0.75 0.9 - 15000 = 16112.5 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_sale_l352_35241


namespace NUMINAMATH_CALUDE_one_intersection_condition_tangent_lines_at_point_l352_35298

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem for the range of m
theorem one_intersection_condition (m : ℝ) :
  (∃! x, f x = m) ↔ (m < -2 ∨ m > 2) :=
sorry

-- Theorem for the tangent lines
theorem tangent_lines_at_point :
  let P : ℝ × ℝ := (2, -6)
  ∃ (l₁ l₂ : ℝ → ℝ),
    (∀ x, l₁ x = -3*x) ∧
    (∀ x, l₂ x = 24*x - 54) ∧
    (∀ t, ∃ x, (x, f x) = (t, l₁ t) ∨ (x, f x) = (t, l₂ t)) ∧
    (l₁ 2 = -6) ∧ (l₂ 2 = -6) :=
sorry

end NUMINAMATH_CALUDE_one_intersection_condition_tangent_lines_at_point_l352_35298


namespace NUMINAMATH_CALUDE_partner_investment_period_l352_35259

/-- Given two partners p and q with investment ratio 7:5 and profit ratio 7:10,
    where q invests for 14 months, prove that p invests for 7 months. -/
theorem partner_investment_period
  (investment_ratio : ℚ) -- Ratio of p's investment to q's investment
  (profit_ratio : ℚ) -- Ratio of p's profit to q's profit
  (q_period : ℕ) -- Investment period of partner q in months
  (h1 : investment_ratio = 7 / 5)
  (h2 : profit_ratio = 7 / 10)
  (h3 : q_period = 14) :
  ∃ (p_period : ℕ), p_period = 7 ∧ 
    (investment_ratio * p_period) / (q_period : ℚ) = profit_ratio :=
by sorry

end NUMINAMATH_CALUDE_partner_investment_period_l352_35259


namespace NUMINAMATH_CALUDE_simplify_expression_l352_35286

theorem simplify_expression (x y : ℝ) (h : y ≠ 0) :
  ((x + y)^2 - (x + y)*(x - y)) / (2*y) = y + x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l352_35286


namespace NUMINAMATH_CALUDE_polynomial_simplification_l352_35221

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 8 * x - 5) - (2 * x^3 + x^2 + 6 * x - 7) = x^3 + 3 * x^2 + 2 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l352_35221


namespace NUMINAMATH_CALUDE_direct_inverse_variation_l352_35215

theorem direct_inverse_variation (k : ℝ) (R X Y : ℝ → ℝ) :
  (∀ t, R t = k * X t / Y t) →  -- R varies directly as X and inversely as Y
  R 0 = 10 ∧ X 0 = 2 ∧ Y 0 = 4 →  -- Initial condition
  R 1 = 8 ∧ Y 1 = 5 →  -- New condition
  X 1 = 2 :=  -- Conclusion
by sorry

end NUMINAMATH_CALUDE_direct_inverse_variation_l352_35215


namespace NUMINAMATH_CALUDE_soap_box_length_proof_l352_35228

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (box : BoxDimensions) : ℝ :=
  box.length * box.width * box.height

theorem soap_box_length_proof 
  (carton : BoxDimensions) 
  (soap_box : BoxDimensions) 
  (max_boxes : ℕ) :
  carton.length = 25 ∧ 
  carton.width = 42 ∧ 
  carton.height = 60 ∧
  soap_box.width = 6 ∧ 
  soap_box.height = 10 ∧
  max_boxes = 150 ∧
  (max_boxes : ℝ) * boxVolume soap_box = boxVolume carton →
  soap_box.length = 7 := by
sorry

end NUMINAMATH_CALUDE_soap_box_length_proof_l352_35228


namespace NUMINAMATH_CALUDE_task_completion_theorem_l352_35243

/-- Represents the number of workers and days to complete a task. -/
structure WorkerDays where
  workers : ℕ
  days : ℕ

/-- Represents the conditions of the problem. -/
structure TaskConditions where
  original : WorkerDays
  reduced : WorkerDays
  increased : WorkerDays

/-- The theorem to prove based on the given conditions. -/
theorem task_completion_theorem (conditions : TaskConditions) : 
  conditions.original.workers = 60 ∧ conditions.original.days = 10 :=
by
  have h1 : conditions.reduced.workers = conditions.original.workers - 20 := by sorry
  have h2 : conditions.reduced.days = conditions.original.days + 5 := by sorry
  have h3 : conditions.increased.workers = conditions.original.workers + 15 := by sorry
  have h4 : conditions.increased.days = conditions.original.days - 2 := by sorry
  have h5 : conditions.original.workers * conditions.original.days = 
            conditions.reduced.workers * conditions.reduced.days := by sorry
  have h6 : conditions.original.workers * conditions.original.days = 
            conditions.increased.workers * conditions.increased.days := by sorry
  sorry

end NUMINAMATH_CALUDE_task_completion_theorem_l352_35243


namespace NUMINAMATH_CALUDE_product_sum_theorem_l352_35216

theorem product_sum_theorem : ∃ (a b c : ℕ), 
  a ∈ ({1, 2, 4, 8, 16, 20} : Set ℕ) ∧ 
  b ∈ ({1, 2, 4, 8, 16, 20} : Set ℕ) ∧ 
  c ∈ ({1, 2, 4, 8, 16, 20} : Set ℕ) ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a * b * c = 80 ∧
  a + b + c = 25 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l352_35216


namespace NUMINAMATH_CALUDE_shorts_weight_l352_35277

/-- The maximum allowed weight for washing clothes -/
def max_weight : ℕ := 50

/-- The weight of a pair of socks in ounces -/
def sock_weight : ℕ := 2

/-- The weight of a pair of underwear in ounces -/
def underwear_weight : ℕ := 4

/-- The weight of a shirt in ounces -/
def shirt_weight : ℕ := 5

/-- The weight of a pair of pants in ounces -/
def pants_weight : ℕ := 10

/-- The number of pairs of pants Tony is washing -/
def num_pants : ℕ := 1

/-- The number of shirts Tony is washing -/
def num_shirts : ℕ := 2

/-- The number of pairs of socks Tony is washing -/
def num_socks : ℕ := 3

/-- The number of additional pairs of underwear Tony can add -/
def additional_underwear : ℕ := 4

/-- Theorem stating that the weight of a pair of shorts is 8 ounces -/
theorem shorts_weight :
  ∃ (shorts_weight : ℕ),
    shorts_weight = max_weight -
      (num_pants * pants_weight +
       num_shirts * shirt_weight +
       num_socks * sock_weight +
       additional_underwear * underwear_weight) :=
by sorry

end NUMINAMATH_CALUDE_shorts_weight_l352_35277


namespace NUMINAMATH_CALUDE_simple_interest_principal_l352_35246

/-- Simple interest calculation -/
theorem simple_interest_principal (rate : ℚ) (time : ℚ) (interest : ℚ) (principal : ℚ) : 
  rate = 6/100 →
  time = 4 →
  interest = 192 →
  principal * rate * time = interest →
  principal = 800 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l352_35246


namespace NUMINAMATH_CALUDE_correct_average_calculation_l352_35242

def total_numbers : ℕ := 20
def initial_average : ℚ := 35
def incorrect_numbers : List (ℚ × ℚ) := [(90, 45), (73, 36), (85, 42), (-45, -27), (64, 35)]

theorem correct_average_calculation :
  let incorrect_sum := initial_average * total_numbers
  let adjustment := (incorrect_numbers.map (λ (x : ℚ × ℚ) => x.1 - x.2)).sum
  let correct_sum := incorrect_sum + adjustment
  correct_sum / total_numbers = 41.8 := by sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l352_35242


namespace NUMINAMATH_CALUDE_frustum_slant_height_is_9_l352_35236

/-- Represents a cone cut by a plane parallel to its base, forming a frustum -/
structure ConeFrustum where
  -- Ratio of top to bottom surface areas
  area_ratio : ℝ
  -- Slant height of the removed cone
  removed_slant_height : ℝ

/-- Calculates the slant height of the frustum -/
def slant_height_frustum (cf : ConeFrustum) : ℝ :=
  sorry

/-- Theorem stating the slant height of the frustum is 9 given the conditions -/
theorem frustum_slant_height_is_9 (cf : ConeFrustum) 
  (h1 : cf.area_ratio = 1 / 16)
  (h2 : cf.removed_slant_height = 3) : 
  slant_height_frustum cf = 9 :=
sorry

end NUMINAMATH_CALUDE_frustum_slant_height_is_9_l352_35236


namespace NUMINAMATH_CALUDE_fifty_roses_cost_l352_35290

/-- The cost of a bouquet of roses, given the number of roses -/
def bouquetCost (roses : ℕ) : ℚ :=
  let baseCost := 24 * roses / 12
  if roses ≥ 45 then baseCost * (1 - 1/10) else baseCost

theorem fifty_roses_cost :
  bouquetCost 50 = 90 := by sorry

end NUMINAMATH_CALUDE_fifty_roses_cost_l352_35290


namespace NUMINAMATH_CALUDE_marta_worked_19_hours_l352_35202

/-- Calculates the number of hours worked given total money collected, hourly wage, and total tips collected. -/
def hours_worked (total_money : ℕ) (hourly_wage : ℕ) (total_tips : ℕ) : ℕ :=
  (total_money - total_tips) / hourly_wage

theorem marta_worked_19_hours :
  hours_worked 240 10 50 = 19 := by
sorry

end NUMINAMATH_CALUDE_marta_worked_19_hours_l352_35202


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l352_35234

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width +  -- bottom area
  2 * length * depth +  -- longer sides area
  2 * width * depth  -- shorter sides area

/-- Theorem: The wet surface area of a 7m x 5m cistern with 1.40m water depth is 68.6 m² -/
theorem cistern_wet_surface_area :
  wetSurfaceArea 7 5 1.40 = 68.6 := by
  sorry

#eval wetSurfaceArea 7 5 1.40

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l352_35234


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l352_35248

-- Define the universal set U
def U : Set ℝ := {x : ℝ | -3 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + x - 2 < 0}

-- Theorem statement
theorem complement_of_A_in_U : 
  (U \ A) = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x < 3)} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l352_35248


namespace NUMINAMATH_CALUDE_clothing_selection_probability_l352_35203

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 3

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 8

/-- The total number of articles of clothing in the drawer -/
def total_items : ℕ := num_shirts + num_shorts + num_socks

/-- The number of articles of clothing to be randomly selected -/
def num_selected : ℕ := 4

/-- The probability of selecting at least one shirt, exactly one pair of shorts, and one pair of socks -/
theorem clothing_selection_probability : 
  (Nat.choose num_shorts 1 * Nat.choose num_socks 1 * 
   (Nat.choose num_shirts 2 + Nat.choose num_shirts 1)) / 
  Nat.choose total_items num_selected = 84 / 397 := by sorry

end NUMINAMATH_CALUDE_clothing_selection_probability_l352_35203


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l352_35276

theorem arithmetic_evaluation : (300 + 5 * 8) / (2^3 : ℝ) = 42.5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l352_35276


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_20_l352_35265

/-- A four-digit number is a natural number between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

/-- Theorem stating that 9920 is the largest four-digit number whose digits sum to 20 -/
theorem largest_four_digit_sum_20 :
  FourDigitNumber 9920 ∧
  sumOfDigits 9920 = 20 ∧
  ∀ n : ℕ, FourDigitNumber n → sumOfDigits n = 20 → n ≤ 9920 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_20_l352_35265


namespace NUMINAMATH_CALUDE_triangle_side_length_20_l352_35270

theorem triangle_side_length_20 :
  ∃ (T S : ℕ), 
    T = 20 ∧ 
    3 * T = 4 * S :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_20_l352_35270


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_extreme_angles_l352_35224

/-- A cyclic quadrilateral with a specific angle ratio -/
structure CyclicQuadrilateral where
  -- Three consecutive angles with ratio 5:6:4
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  angle_ratio : a = 5 * (b / 6) ∧ c = 4 * (b / 6)
  -- Sum of opposite angles is 180°
  opposite_sum : a + d = 180 ∧ b + c = 180

/-- The largest and smallest angles in the cyclic quadrilateral -/
def extreme_angles (q : CyclicQuadrilateral) : ℝ × ℝ :=
  (108, 72)

theorem cyclic_quadrilateral_extreme_angles (q : CyclicQuadrilateral) :
  extreme_angles q = (108, 72) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_extreme_angles_l352_35224


namespace NUMINAMATH_CALUDE_quadratic_has_real_roots_l352_35262

/-- The quadratic equation x^2 - 4x + 2a = 0 has real roots when a = 1 -/
theorem quadratic_has_real_roots : ∃ (x : ℝ), x^2 - 4*x + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_real_roots_l352_35262


namespace NUMINAMATH_CALUDE_tangent_sum_problem_l352_35225

theorem tangent_sum_problem (p q : ℝ) 
  (h1 : (Real.sin p / Real.cos q) + (Real.sin q / Real.cos p) = 2)
  (h2 : (Real.cos p / Real.sin q) + (Real.cos q / Real.sin p) = 3) :
  (Real.tan p / Real.tan q) + (Real.tan q / Real.tan p) = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_problem_l352_35225


namespace NUMINAMATH_CALUDE_factorization_sum_l352_35291

theorem factorization_sum (a b : ℤ) :
  (∀ x : ℚ, 25 * x^2 - 85 * x - 150 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -24 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l352_35291


namespace NUMINAMATH_CALUDE_emmanuel_regular_plan_cost_l352_35244

/-- Calculates the regular plan cost given the stay duration, international data cost per day, and total charges. -/
def regular_plan_cost (stay_duration : ℕ) (intl_data_cost_per_day : ℚ) (total_charges : ℚ) : ℚ :=
  total_charges - (stay_duration : ℚ) * intl_data_cost_per_day

/-- Proves that Emmanuel's regular plan cost is $175 given the problem conditions. -/
theorem emmanuel_regular_plan_cost :
  regular_plan_cost 10 (350/100) 210 = 175 := by
  sorry

end NUMINAMATH_CALUDE_emmanuel_regular_plan_cost_l352_35244


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l352_35204

theorem arithmetic_calculations : 
  ((1 : ℝ) + 4 - (-7) + (-8) = 3) ∧ 
  (-8.9 - (-4.7) + 7.5 = 3.3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l352_35204


namespace NUMINAMATH_CALUDE_go_games_theorem_l352_35261

/-- The number of complete Go games that can be played simultaneously -/
def maxSimultaneousGames (totalBalls : ℕ) (ballsPerGame : ℕ) : ℕ :=
  totalBalls / ballsPerGame

theorem go_games_theorem :
  maxSimultaneousGames 901 53 = 17 := by
  sorry

end NUMINAMATH_CALUDE_go_games_theorem_l352_35261


namespace NUMINAMATH_CALUDE_hanks_fruit_purchase_cost_hanks_fruit_purchase_cost_is_1704_l352_35239

/-- Calculates the total cost of Hank's fruit purchase at Clark's Food Store --/
theorem hanks_fruit_purchase_cost : ℝ :=
  let apple_price_per_dozen : ℝ := 40
  let pear_price_per_dozen : ℝ := 50
  let orange_price_per_dozen : ℝ := 30
  let apple_dozens_bought : ℝ := 14
  let pear_dozens_bought : ℝ := 18
  let orange_dozens_bought : ℝ := 10
  let apple_discount_rate : ℝ := 0.1

  let apple_cost : ℝ := apple_price_per_dozen * apple_dozens_bought
  let discounted_apple_cost : ℝ := apple_cost * (1 - apple_discount_rate)
  let pear_cost : ℝ := pear_price_per_dozen * pear_dozens_bought
  let orange_cost : ℝ := orange_price_per_dozen * orange_dozens_bought

  let total_cost : ℝ := discounted_apple_cost + pear_cost + orange_cost

  1704

/-- Proves that Hank's total fruit purchase cost is 1704 dollars --/
theorem hanks_fruit_purchase_cost_is_1704 : hanks_fruit_purchase_cost = 1704 := by
  sorry

end NUMINAMATH_CALUDE_hanks_fruit_purchase_cost_hanks_fruit_purchase_cost_is_1704_l352_35239


namespace NUMINAMATH_CALUDE_right_triangle_identification_l352_35232

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_identification :
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5) ∧
  is_right_triangle 3 4 5 ∧
  ¬ is_right_triangle 7 8 9 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l352_35232


namespace NUMINAMATH_CALUDE_rectangle_width_l352_35245

theorem rectangle_width (length area : ℚ) (h1 : length = 3/5) (h2 : area = 1/3) :
  area / length = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l352_35245


namespace NUMINAMATH_CALUDE_max_items_for_alex_washing_l352_35292

/-- Represents a washing machine with its characteristics and items to wash -/
structure WashingMachine where
  total_items : ℕ
  cycle_duration : ℕ  -- in minutes
  total_wash_time : ℕ  -- in minutes

/-- Calculates the maximum number of items that can be washed per cycle -/
def max_items_per_cycle (wm : WashingMachine) : ℕ :=
  wm.total_items / (wm.total_wash_time / wm.cycle_duration)

/-- Theorem stating the maximum number of items per cycle for the given washing machine -/
theorem max_items_for_alex_washing (wm : WashingMachine) 
  (h1 : wm.total_items = 60)
  (h2 : wm.cycle_duration = 45)
  (h3 : wm.total_wash_time = 180) :
  max_items_per_cycle wm = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_items_for_alex_washing_l352_35292


namespace NUMINAMATH_CALUDE_find_multiple_l352_35255

theorem find_multiple (n m : ℝ) (h1 : n + n + m * n + 4 * n = 104) (h2 : n = 13) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_multiple_l352_35255


namespace NUMINAMATH_CALUDE_promotion_savings_difference_l352_35295

/-- Represents the cost of a pair of shoes in dollars -/
def shoe_cost : ℝ := 50

/-- Calculates the cost of two pairs of shoes using Promotion X -/
def cost_promotion_x : ℝ :=
  shoe_cost + (shoe_cost * (1 - 0.4))

/-- Calculates the cost of two pairs of shoes using Promotion Y -/
def cost_promotion_y : ℝ :=
  shoe_cost + (shoe_cost - 15)

/-- Theorem: The difference in cost between Promotion Y and Promotion X is $5 -/
theorem promotion_savings_difference :
  cost_promotion_y - cost_promotion_x = 5 := by
  sorry

end NUMINAMATH_CALUDE_promotion_savings_difference_l352_35295


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l352_35207

theorem simplify_trig_expression :
  2 * Real.sqrt (1 + Real.sin 4) + Real.sqrt (2 + 2 * Real.cos 4) = 2 * Real.sin 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l352_35207


namespace NUMINAMATH_CALUDE_smallest_integer_theorem_l352_35266

def is_divisible (n m : ℕ) : Prop := m ∣ n

def smallest_integer_with_divisors (excluded : List ℕ) : ℕ :=
  let divisors := (List.range 31).filter (λ x => x ∉ excluded)
  divisors.foldl Nat.lcm 1

theorem smallest_integer_theorem :
  let n := smallest_integer_with_divisors [17, 19]
  (∀ k ∈ List.range 31, k ≠ 17 → k ≠ 19 → is_divisible n k) ∧
  (∀ m < n, ∃ k ∈ List.range 31, k ≠ 17 ∧ k ≠ 19 ∧ ¬is_divisible m k) ∧
  n = 122522400 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_theorem_l352_35266


namespace NUMINAMATH_CALUDE_total_apples_is_75_l352_35254

/-- The number of apples Benny picked from each tree -/
def benny_apples_per_tree : ℕ := 2

/-- The number of trees Benny picked from -/
def benny_trees : ℕ := 4

/-- The number of apples Dan picked from each tree -/
def dan_apples_per_tree : ℕ := 9

/-- The number of trees Dan picked from -/
def dan_trees : ℕ := 5

/-- Calculate the total number of apples picked by Benny -/
def benny_total : ℕ := benny_apples_per_tree * benny_trees

/-- Calculate the total number of apples picked by Dan -/
def dan_total : ℕ := dan_apples_per_tree * dan_trees

/-- Calculate the number of apples picked by Sarah (half of Dan's total, rounded down) -/
def sarah_total : ℕ := dan_total / 2

/-- The total number of apples picked by all three people -/
def total_apples : ℕ := benny_total + dan_total + sarah_total

theorem total_apples_is_75 : total_apples = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_75_l352_35254


namespace NUMINAMATH_CALUDE_bowlingPrizeOrders_l352_35206

/-- Represents the number of bowlers in the tournament -/
def numBowlers : ℕ := 7

/-- Represents the number of playoff matches -/
def numMatches : ℕ := 6

/-- The number of possible outcomes for each match -/
def outcomesPerMatch : ℕ := 2

/-- Calculates the total number of possible prize orders -/
def totalPossibleOrders : ℕ := outcomesPerMatch ^ numMatches

/-- Proves that the number of different possible prize orders is 64 -/
theorem bowlingPrizeOrders : totalPossibleOrders = 64 := by
  sorry

end NUMINAMATH_CALUDE_bowlingPrizeOrders_l352_35206


namespace NUMINAMATH_CALUDE_pie_division_l352_35235

theorem pie_division (initial_pie : ℚ) (scrooge_share : ℚ) (num_friends : ℕ) : 
  initial_pie = 4/5 → scrooge_share = 1/5 → num_friends = 3 →
  (initial_pie - scrooge_share * initial_pie) / num_friends = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_pie_division_l352_35235


namespace NUMINAMATH_CALUDE_smallest_square_coverage_l352_35205

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℕ := s.side * s.side

/-- Checks if a square can be exactly covered by a given number of rectangles -/
def canCoverSquare (s : Square) (r : Rectangle) (n : ℕ) : Prop :=
  squareArea s = n * rectangleArea r

/-- The theorem to be proved -/
theorem smallest_square_coverage :
  ∃ (s : Square) (n : ℕ),
    let r : Rectangle := ⟨2, 3⟩
    canCoverSquare s r n ∧
    (∀ (s' : Square) (n' : ℕ), canCoverSquare s' r n' → squareArea s ≤ squareArea s') ∧
    n = 6 :=
  sorry

end NUMINAMATH_CALUDE_smallest_square_coverage_l352_35205


namespace NUMINAMATH_CALUDE_fraction_equality_l352_35289

theorem fraction_equality (a b : ℝ) (h : a/b = 2) : a/(a-b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l352_35289


namespace NUMINAMATH_CALUDE_divisor_proof_l352_35263

theorem divisor_proof (original : Nat) (added : Nat) (sum : Nat) (h1 : original = 859622) (h2 : added = 859560) (h3 : sum = original + added) :
  sum % added = 0 ∧ added ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_divisor_proof_l352_35263


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l352_35275

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l352_35275


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l352_35229

theorem arithmetic_sequence_middle_term (a : ℕ → ℝ) :
  (a 0 = 3^2) →
  (a 2 = 3^4) →
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →
  (a 1 = 45) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l352_35229


namespace NUMINAMATH_CALUDE_largest_c_for_4_in_range_l352_35279

/-- The quadratic function f(x) = x^2 + 5x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 5*x + c

/-- Theorem: The largest value of c such that 4 is in the range of f(x) = x^2 + 5x + c is 10.25 -/
theorem largest_c_for_4_in_range : 
  (∃ (x : ℝ), f 10.25 x = 4) ∧ 
  (∀ (c : ℝ), c > 10.25 → ¬∃ (x : ℝ), f c x = 4) := by
  sorry


end NUMINAMATH_CALUDE_largest_c_for_4_in_range_l352_35279


namespace NUMINAMATH_CALUDE_last_two_pieces_l352_35260

def pieces : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_odd (n : Nat) : Bool := n % 2 = 1

def product_is_24 (s : Finset Nat) : Bool :=
  s.prod id = 24

def removal_process (s : Finset Nat) : Finset Nat :=
  let after_odd_removal := s.filter (fun n => ¬is_odd n)
  let after_product_removal := after_odd_removal.filter (fun n => ¬product_is_24 {n})
  after_product_removal

theorem last_two_pieces (s : Finset Nat) :
  s = pieces →
  (removal_process s = {2, 8} ∨ removal_process s = {6, 8}) :=
sorry

end NUMINAMATH_CALUDE_last_two_pieces_l352_35260


namespace NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l352_35293

/-- The total surface area of a shape consisting of a hemisphere attached to a cylindrical segment -/
theorem hemisphere_cylinder_surface_area (r : ℝ) (h : r = 10) :
  let hemisphere_area := 2 * π * r^2
  let cylinder_base_area := π * r^2
  let cylinder_lateral_area := 2 * π * r * (r / 2)
  hemisphere_area + cylinder_base_area + cylinder_lateral_area = 40 * π * r^2 :=
by sorry

end NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l352_35293


namespace NUMINAMATH_CALUDE_circle_tangency_and_chord_properties_l352_35282

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define point P
def point_P (t : ℝ) : ℝ × ℝ := (-1, t)

-- Define circle N
def circle_N (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 1

-- Define the external tangency condition
def external_tangent (a b : ℝ) : Prop := (a - 2)^2 + b^2 = 9

-- Define the chord length condition
def chord_length (t : ℝ) : Prop := ∃ (k : ℝ), 8*k^2 + 6*t*k + t^2 - 1 = 0

-- Define the ST distance condition
def ST_distance (t : ℝ) : Prop := (t^2 + 8) / 16 = 9/16

theorem circle_tangency_and_chord_properties :
  ∀ t : ℝ,
  (∃ a b : ℝ, circle_N (-1) 1 a b ∧ external_tangent a b) →
  (chord_length t ∧ ST_distance t) →
  ((circle_N x y (-1) 0 ∨ circle_N x y (-2/5) (9/5)) ∧ (t = 1 ∨ t = -1)) :=
sorry

end NUMINAMATH_CALUDE_circle_tangency_and_chord_properties_l352_35282


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l352_35209

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.00000000005 = a * 10^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l352_35209


namespace NUMINAMATH_CALUDE_minimum_shoeing_time_l352_35201

/-- The minimum time required for blacksmiths to shoe horses -/
theorem minimum_shoeing_time 
  (num_blacksmiths : ℕ) 
  (num_horses : ℕ) 
  (time_per_horseshoe : ℕ) 
  (horseshoes_per_horse : ℕ) : 
  num_blacksmiths = 48 → 
  num_horses = 60 → 
  time_per_horseshoe = 5 → 
  horseshoes_per_horse = 4 → 
  (num_horses * horseshoes_per_horse * time_per_horseshoe) / num_blacksmiths = 25 := by
  sorry

end NUMINAMATH_CALUDE_minimum_shoeing_time_l352_35201


namespace NUMINAMATH_CALUDE_slope_range_for_inclination_angle_l352_35299

theorem slope_range_for_inclination_angle (α : Real) :
  π / 4 ≤ α ∧ α ≤ 3 * π / 4 →
  ∃ k : Real, (k < -1 ∨ k = -1 ∨ k = 1 ∨ k > 1) ∧ k = Real.tan α :=
sorry

end NUMINAMATH_CALUDE_slope_range_for_inclination_angle_l352_35299


namespace NUMINAMATH_CALUDE_mandy_gets_fifteen_l352_35287

def chocolate_bar : ℕ := 60

def michael_share (total : ℕ) : ℕ := total / 2

def paige_share (remaining : ℕ) : ℕ := remaining / 2

def mandy_share (total : ℕ) : ℕ :=
  let after_michael := total - michael_share total
  after_michael - paige_share after_michael

theorem mandy_gets_fifteen :
  mandy_share chocolate_bar = 15 := by
  sorry

end NUMINAMATH_CALUDE_mandy_gets_fifteen_l352_35287


namespace NUMINAMATH_CALUDE_rectangle_ratio_l352_35258

/-- A configuration of squares and rectangles -/
structure SquareRectConfig where
  /-- Side length of the inner square -/
  s : ℝ
  /-- Shorter side of each rectangle -/
  y : ℝ
  /-- Longer side of each rectangle -/
  x : ℝ
  /-- The shorter side of each rectangle is half the side of the inner square -/
  short_side_half : y = s / 2
  /-- The area of the outer square is 9 times that of the inner square -/
  area_ratio : (s + 2 * y)^2 = 9 * s^2
  /-- The longer side of the rectangle forms the side of the outer square with the inner square -/
  outer_square_side : x + s / 2 = 3 * s

/-- The ratio of the longer side to the shorter side of each rectangle is 5 -/
theorem rectangle_ratio (config : SquareRectConfig) : config.x / config.y = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l352_35258


namespace NUMINAMATH_CALUDE_toothpick_pattern_15th_stage_l352_35227

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem toothpick_pattern_15th_stage :
  arithmetic_sequence 5 3 15 = 47 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_pattern_15th_stage_l352_35227


namespace NUMINAMATH_CALUDE_trapezium_height_l352_35294

theorem trapezium_height (a b h : ℝ) : 
  a > 0 → b > 0 → h > 0 →
  a = 20 → b = 18 → 
  (1/2) * (a + b) * h = 190 →
  h = 10 := by
sorry

end NUMINAMATH_CALUDE_trapezium_height_l352_35294


namespace NUMINAMATH_CALUDE_reciprocal_sum_one_triples_l352_35219

def reciprocal_sum_one (a b c : ℕ+) : Prop :=
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = 1

def valid_triples : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(2, 3, 6), (2, 6, 3), (3, 2, 6), (3, 6, 2), (6, 2, 3), (6, 3, 2),
   (2, 4, 4), (4, 2, 4), (4, 4, 2), (3, 3, 3)}

theorem reciprocal_sum_one_triples :
  ∀ (a b c : ℕ+), reciprocal_sum_one a b c ↔ (a, b, c) ∈ valid_triples := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_one_triples_l352_35219


namespace NUMINAMATH_CALUDE_triangle_properties_l352_35281

noncomputable section

/-- Triangle ABC with given properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle given two sides and the included angle -/
def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 2)
  (h2 : t.C = π/3) :
  (area t = Real.sqrt 3 → t.a = 2 ∧ t.b = 2) ∧
  (Real.sin t.B = 2 * Real.sin t.A → area t = 4 * Real.sqrt 3 / 3) := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l352_35281


namespace NUMINAMATH_CALUDE_basketball_team_subjects_l352_35268

theorem basketball_team_subjects (P C B : Finset Nat) : 
  (P ∪ C ∪ B).card = 18 →
  P.card = 10 →
  B.card = 7 →
  C.card = 5 →
  (P ∩ B).card = 3 →
  (B ∩ C).card = 2 →
  (P ∩ C).card = 1 →
  (P ∩ C ∩ B).card = 2 := by
sorry

end NUMINAMATH_CALUDE_basketball_team_subjects_l352_35268


namespace NUMINAMATH_CALUDE_stadium_problem_l352_35230

theorem stadium_problem (total_start : ℕ) (total_end : ℕ) 
  (h1 : total_start = 600)
  (h2 : total_end = 480) :
  ∃ (boys girls : ℕ),
    boys + girls = total_start ∧
    boys - boys / 4 + girls - girls / 8 = total_end ∧
    girls = 240 := by
  sorry

end NUMINAMATH_CALUDE_stadium_problem_l352_35230


namespace NUMINAMATH_CALUDE_alcohol_percentage_solution_x_l352_35250

/-- Proves that the percentage of alcohol by volume in solution x is 10% -/
theorem alcohol_percentage_solution_x :
  ∀ (x y : ℝ),
  y = 0.30 →
  450 * y + 300 * x = 0.22 * (450 + 300) →
  x = 0.10 := by
sorry

end NUMINAMATH_CALUDE_alcohol_percentage_solution_x_l352_35250


namespace NUMINAMATH_CALUDE_g_of_2_l352_35257

def g (x : ℝ) : ℝ := x^2 + 3*x - 1

theorem g_of_2 : g 2 = 9 := by sorry

end NUMINAMATH_CALUDE_g_of_2_l352_35257


namespace NUMINAMATH_CALUDE_sugar_flour_difference_l352_35288

-- Define constants based on the problem conditions
def flour_recipe : Real := 2.25  -- kg
def sugar_recipe : Real := 5.5   -- lb
def flour_added : Real := 1      -- kg
def kg_to_lb : Real := 2.205     -- 1 kg = 2.205 lb
def kg_to_g : Real := 1000       -- 1 kg = 1000 g

-- Theorem statement
theorem sugar_flour_difference :
  let flour_remaining := (flour_recipe - flour_added) * kg_to_g
  let sugar_needed := (sugar_recipe / kg_to_lb) * kg_to_g
  ∃ ε > 0, abs (sugar_needed - flour_remaining - 1244.8) < ε :=
by sorry

end NUMINAMATH_CALUDE_sugar_flour_difference_l352_35288


namespace NUMINAMATH_CALUDE_factorization_proof_l352_35272

theorem factorization_proof (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l352_35272


namespace NUMINAMATH_CALUDE_inequality_proof_l352_35238

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 3) : 
  1 / (Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1)) + 
  1 / (Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1)) + 
  1 / (Real.sqrt (3 * z + 1) + Real.sqrt (3 * x + 1)) ≥ 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l352_35238


namespace NUMINAMATH_CALUDE_fraction_subtraction_l352_35214

theorem fraction_subtraction : 7 - (2 / 5)^3 = 867 / 125 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l352_35214


namespace NUMINAMATH_CALUDE_incompatible_inequalities_l352_35226

theorem incompatible_inequalities :
  ¬∃ (a b c d : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
    a + b < c + d ∧
    (a + b) * (c + d) < a * b + c * d ∧
    (a + b) * c * d < a * b * (c + d) := by
  sorry

end NUMINAMATH_CALUDE_incompatible_inequalities_l352_35226


namespace NUMINAMATH_CALUDE_unknown_number_problem_l352_35273

theorem unknown_number_problem : ∃ x : ℝ, 0.5 * 56 = 0.3 * x + 13 ∧ x = 50 := by sorry

end NUMINAMATH_CALUDE_unknown_number_problem_l352_35273


namespace NUMINAMATH_CALUDE_line_through_point_l352_35211

/-- 
Given a line equation 2 - kx = -4y that passes through the point (3, -2),
prove that k = -2.
-/
theorem line_through_point (k : ℝ) : 
  (2 - k * 3 = -4 * (-2)) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l352_35211


namespace NUMINAMATH_CALUDE_irregular_quadrilateral_tiles_plane_l352_35212

-- Define an irregular quadrilateral
structure IrregularQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ

-- Define a tiling of the plane
def PlaneTiling (Q : Type) := ℝ × ℝ → Q

-- Define the property of being a valid tiling (no gaps or overlaps)
def IsValidTiling (Q : Type) (tiling : PlaneTiling Q) : Prop := sorry

-- Theorem statement
theorem irregular_quadrilateral_tiles_plane (q : IrregularQuadrilateral) :
  ∃ (tiling : PlaneTiling IrregularQuadrilateral), IsValidTiling IrregularQuadrilateral tiling :=
sorry

end NUMINAMATH_CALUDE_irregular_quadrilateral_tiles_plane_l352_35212


namespace NUMINAMATH_CALUDE_hyperbola_iff_mn_neg_l352_35208

/-- Defines whether an equation represents a hyperbola -/
def is_hyperbola (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / m + y^2 / n = 1 ∧ 
  ¬∃ (a b : ℝ), ∀ (x y : ℝ), x^2 / m + y^2 / n = 1 ↔ (x - a)^2 + (y - b)^2 = 1

/-- Proves that mn < 0 is necessary and sufficient for the equation to represent a hyperbola -/
theorem hyperbola_iff_mn_neg (m n : ℝ) :
  is_hyperbola m n ↔ m * n < 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_iff_mn_neg_l352_35208


namespace NUMINAMATH_CALUDE_blue_section_probability_l352_35283

def bernoulli_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem blue_section_probability : 
  bernoulli_probability 7 7 (2/7) = 128/823543 := by
  sorry

end NUMINAMATH_CALUDE_blue_section_probability_l352_35283


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l352_35296

theorem lcm_gcd_problem (x y : ℕ+) : 
  Nat.lcm x y = 5940 → 
  Nat.gcd x y = 22 → 
  x = 220 → 
  y = 594 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l352_35296


namespace NUMINAMATH_CALUDE_power_equality_implies_x_equals_two_l352_35200

theorem power_equality_implies_x_equals_two :
  ∀ x : ℝ, (2 : ℝ)^10 = 32^x → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_implies_x_equals_two_l352_35200


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l352_35218

theorem sum_of_A_and_B : ∀ (A B : ℚ), 3/7 = 6/A ∧ 6/A = B/21 → A + B = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l352_35218


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l352_35256

theorem rectangle_perimeter (square_perimeter : ℝ) (h : square_perimeter = 100) :
  let square_side := square_perimeter / 4
  let rectangle_length := square_side
  let rectangle_width := square_side / 2
  2 * (rectangle_length + rectangle_width) = 75 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l352_35256


namespace NUMINAMATH_CALUDE_no_entangled_numbers_l352_35210

/-- A two-digit positive integer is entangled if it equals twice the sum of its nonzero tens digit and the cube of its units digit -/
def is_entangled (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ (a b : ℕ), a > 0 ∧ b < 10 ∧ n = 10 * a + b ∧ n = 2 * (a + b^3)

/-- There are no entangled two-digit positive integers -/
theorem no_entangled_numbers : ¬∃ (n : ℕ), is_entangled n := by
  sorry

end NUMINAMATH_CALUDE_no_entangled_numbers_l352_35210


namespace NUMINAMATH_CALUDE_stationery_store_profit_l352_35271

/-- Profit data for a week at a stationery store -/
structure WeekProfit :=
  (mon tue wed thu fri sat sun : ℝ)
  (total : ℝ)
  (sum_condition : mon + tue + wed + thu + fri + sat + sun = total)

/-- Theorem stating the properties of the profit data -/
theorem stationery_store_profit 
  (w : WeekProfit)
  (h1 : w.mon = -27.8)
  (h2 : w.tue = -70.3)
  (h3 : w.wed = 200)
  (h4 : w.thu = 138.1)
  (h5 : w.sun = 188)
  (h6 : w.total = 458) :
  (w.fri = -8 → w.sat = 38) ∧
  (w.sat = w.fri + 10 → w.sat = 20) ∧
  (w.fri < 0 → w.sat > 0 → w.sat > 30) :=
by sorry

end NUMINAMATH_CALUDE_stationery_store_profit_l352_35271


namespace NUMINAMATH_CALUDE_min_value_problem_l352_35240

theorem min_value_problem (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 4)
  (h5 : y^2 = x^2 + 2) (h6 : z^2 = y^2 + 2) :
  ∃ (min_val : ℝ), min_val = 4 - 2 * Real.sqrt 3 ∧ 
  ∀ (x' y' z' : ℝ), 0 ≤ x' ∧ x' ≤ y' ∧ y' ≤ z' ∧ z' ≤ 4 ∧ 
  y'^2 = x'^2 + 2 ∧ z'^2 = y'^2 + 2 → z' - x' ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l352_35240


namespace NUMINAMATH_CALUDE_six_students_three_competitions_l352_35264

/-- The number of ways to assign students to competitions -/
def registration_methods (num_students : ℕ) (num_competitions : ℕ) : ℕ :=
  num_competitions ^ num_students

/-- Theorem: The number of ways to assign 6 students to 3 competitions is 729 -/
theorem six_students_three_competitions :
  registration_methods 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_six_students_three_competitions_l352_35264


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l352_35274

/-- Represents a repeating decimal with a single digit repeating -/
def SingleDigitRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + repeating / 9

/-- Represents a repeating decimal with two digits repeating -/
def TwoDigitRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + repeating / 99

theorem sum_of_repeating_decimals :
  (SingleDigitRepeatingDecimal 0 2) + (TwoDigitRepeatingDecimal 0 4) = 26 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l352_35274


namespace NUMINAMATH_CALUDE_m_eq_2_sufficient_not_necessary_l352_35231

def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

theorem m_eq_2_sufficient_not_necessary :
  (∃ m : ℝ, (A m) ∩ B = {4} ∧ m ≠ 2) ∧
  (∀ m : ℝ, m = 2 → (A m) ∩ B = {4}) :=
sorry

end NUMINAMATH_CALUDE_m_eq_2_sufficient_not_necessary_l352_35231
