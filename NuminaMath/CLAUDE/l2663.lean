import Mathlib

namespace NUMINAMATH_CALUDE_chocolate_cost_is_75_cents_l2663_266391

/-- The cost of a candy bar in cents -/
def candy_bar_cost : ℕ := 25

/-- The cost of a pack of juice in cents -/
def juice_cost : ℕ := 50

/-- The total cost in cents for 3 candy bars, 2 pieces of chocolate, and 1 pack of juice -/
def total_cost : ℕ := 275

/-- The number of candy bars purchased -/
def num_candy_bars : ℕ := 3

/-- The number of chocolate pieces purchased -/
def num_chocolates : ℕ := 2

/-- The number of juice packs purchased -/
def num_juice_packs : ℕ := 1

theorem chocolate_cost_is_75_cents :
  ∃ (chocolate_cost : ℕ),
    chocolate_cost * num_chocolates + 
    candy_bar_cost * num_candy_bars + 
    juice_cost * num_juice_packs = total_cost ∧
    chocolate_cost = 75 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_is_75_cents_l2663_266391


namespace NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l2663_266360

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The problem statement --/
theorem base_10_to_base_7_conversion :
  base7ToBase10 [5, 0, 2, 2] = 789 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l2663_266360


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2663_266317

/-- Represents a chicken farm with its population -/
structure Farm where
  population : ℕ

/-- Calculates the sample size for a farm given the total population and total sample size -/
def sampleSize (farm : Farm) (totalPopulation : ℕ) (totalSample : ℕ) : ℕ :=
  (farm.population * totalSample) / totalPopulation

theorem stratified_sampling_theorem (farmA farmB farmC : Farm) 
    (h1 : farmA.population = 12000)
    (h2 : farmB.population = 8000)
    (h3 : farmC.population = 4000)
    (totalSample : ℕ)
    (h4 : totalSample = 120) :
  let totalPopulation := farmA.population + farmB.population + farmC.population
  (sampleSize farmA totalPopulation totalSample = 60) ∧
  (sampleSize farmB totalPopulation totalSample = 40) ∧
  (sampleSize farmC totalPopulation totalSample = 20) := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2663_266317


namespace NUMINAMATH_CALUDE_multiples_of_three_never_reach_one_l2663_266378

def operation (n : ℕ) : ℕ :=
  (n + 3 * (5 - n % 5) % 5) / 5

theorem multiples_of_three_never_reach_one (k : ℕ) :
  ∀ n : ℕ, (∃ m : ℕ, n = operation^[m] (3 * k)) → n ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_multiples_of_three_never_reach_one_l2663_266378


namespace NUMINAMATH_CALUDE_continuity_at_6_delta_formula_l2663_266373

def f (x : ℝ) : ℝ := 3 * x^2 + 7

theorem continuity_at_6 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 6| < δ → |f x - f 6| < ε :=
by
  sorry

theorem delta_formula (ε : ℝ) (h : ε > 0) : 
  ∃ δ > 0, δ = ε / 36 ∧ ∀ x, |x - 6| < δ → |f x - f 6| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_continuity_at_6_delta_formula_l2663_266373


namespace NUMINAMATH_CALUDE_field_division_l2663_266399

theorem field_division (total_area smaller_area larger_area : ℝ) : 
  total_area = 700 ∧ 
  smaller_area + larger_area = total_area ∧ 
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area = 315 := by
sorry

end NUMINAMATH_CALUDE_field_division_l2663_266399


namespace NUMINAMATH_CALUDE_gcf_of_48_and_14_l2663_266376

theorem gcf_of_48_and_14 :
  let n : ℕ := 48
  let m : ℕ := 14
  let lcm_nm : ℕ := 56
  Nat.lcm n m = lcm_nm →
  Nat.gcd n m = 12 := by
sorry

end NUMINAMATH_CALUDE_gcf_of_48_and_14_l2663_266376


namespace NUMINAMATH_CALUDE_brians_books_l2663_266397

/-- The number of chapters in the first book Brian read -/
def book1_chapters : ℕ := 20

/-- The total number of chapters Brian read -/
def total_chapters : ℕ := 75

/-- The number of identical books Brian read -/
def identical_books : ℕ := 2

theorem brians_books (x : ℕ) : 
  book1_chapters + identical_books * x + (book1_chapters + identical_books * x) / 2 = total_chapters → 
  x = 15 :=
by sorry

end NUMINAMATH_CALUDE_brians_books_l2663_266397


namespace NUMINAMATH_CALUDE_correct_sum_is_45250_l2663_266334

/-- Represents the sum with errors --/
def incorrect_sum : ℕ := 52000

/-- Represents the error in the first number's tens place --/
def tens_error : ℤ := 50

/-- Represents the error in the first number's hundreds place --/
def hundreds_error : ℤ := -300

/-- Represents the error in the second number's thousands place --/
def thousands_error : ℤ := 7000

/-- The total error introduced by the mistakes --/
def total_error : ℤ := tens_error + hundreds_error + thousands_error

/-- The correct sum after adjusting for errors --/
def correct_sum : ℕ := incorrect_sum - total_error.toNat

theorem correct_sum_is_45250 : correct_sum = 45250 := by
  sorry

end NUMINAMATH_CALUDE_correct_sum_is_45250_l2663_266334


namespace NUMINAMATH_CALUDE_polygon_diagonals_l2663_266388

/-- The number of diagonals in a polygon with exterior angles of 10 degrees each -/
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem polygon_diagonals :
  ∃ n : ℕ,
    n > 0 ∧
    n * 10 = 360 ∧
    num_diagonals n = 594 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l2663_266388


namespace NUMINAMATH_CALUDE_quadratic_completion_sum_l2663_266352

/-- For the quadratic x^2 - 24x + 50, when written as (x+b)^2 + c, b+c equals -106 -/
theorem quadratic_completion_sum (b c : ℝ) : 
  (∀ x, x^2 - 24*x + 50 = (x+b)^2 + c) → b + c = -106 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_sum_l2663_266352


namespace NUMINAMATH_CALUDE_dad_steps_l2663_266316

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between steps taken by Dad and Masha -/
def dad_masha_ratio (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha

/-- Defines the relationship between steps taken by Masha and Yasha -/
def masha_yasha_ratio (s : Steps) : Prop :=
  5 * s.masha = 3 * s.yasha

/-- States that Masha and Yasha together took 400 steps -/
def total_masha_yasha (s : Steps) : Prop :=
  s.masha + s.yasha = 400

/-- Theorem stating that given the conditions, Dad took 90 steps -/
theorem dad_steps :
  ∀ s : Steps,
  dad_masha_ratio s →
  masha_yasha_ratio s →
  total_masha_yasha s →
  s.dad = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_dad_steps_l2663_266316


namespace NUMINAMATH_CALUDE_grain_demand_formula_l2663_266342

/-- World grain supply and demand model -/
structure GrainModel where
  S : ℝ  -- World grain supply
  D : ℝ  -- World grain demand
  F : ℝ  -- Production fluctuations
  P : ℝ  -- Population growth
  S0 : ℝ  -- Base supply value
  D0 : ℝ  -- Initial demand value

/-- Conditions for the grain model -/
def GrainModelConditions (m : GrainModel) : Prop :=
  m.S = 0.75 * m.D ∧
  m.S = m.S0 * (1 + m.F) ∧
  m.D = m.D0 * (1 + m.P) ∧
  m.S0 = 1800000

/-- Theorem: Given the conditions, the world grain demand D can be expressed as D = (1,800,000 * (1 + F)) / 0.75 -/
theorem grain_demand_formula (m : GrainModel) (h : GrainModelConditions m) :
  m.D = (1800000 * (1 + m.F)) / 0.75 := by
  sorry


end NUMINAMATH_CALUDE_grain_demand_formula_l2663_266342


namespace NUMINAMATH_CALUDE_conic_section_classification_l2663_266309

/-- Given an interior angle θ of a triangle, vectors m and n, and their dot product,
    prove that the equation x²sin θ - y²cos θ = 1 represents an ellipse with foci on the y-axis. -/
theorem conic_section_classification (θ : ℝ) (m n : Fin 2 → ℝ) :
  (∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ θ = Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))) →  -- θ is an interior angle of a triangle
  (m 0 = Real.sin θ ∧ m 1 = Real.cos θ) →  -- m⃗ = (sin θ, cos θ)
  (n 0 = 1 ∧ n 1 = 1) →  -- n⃗ = (1, 1)
  (m 0 * n 0 + m 1 * n 1 = 1/3) →  -- m⃗ · n⃗ = 1/3
  (∃ (x y : ℝ), x^2 * Real.sin θ - y^2 * Real.cos θ = 1) →  -- Equation: x²sin θ - y²cos θ = 1
  (∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a ≠ b ∧ 
    ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b) :=  -- The equation represents an ellipse with foci on the y-axis
by sorry

end NUMINAMATH_CALUDE_conic_section_classification_l2663_266309


namespace NUMINAMATH_CALUDE_true_propositions_count_l2663_266336

/-- Represents the four propositions about geometric solids -/
inductive GeometricProposition
| RegularPyramidLateralEdges
| RightPrismLateralFaces
| CylinderGeneratrix
| ConeSectionIsoscelesTriangles

/-- Determines if a given geometric proposition is true -/
def isTrue (prop : GeometricProposition) : Bool :=
  match prop with
  | .RegularPyramidLateralEdges => true
  | .RightPrismLateralFaces => false
  | .CylinderGeneratrix => true
  | .ConeSectionIsoscelesTriangles => true

/-- The list of all geometric propositions -/
def allPropositions : List GeometricProposition :=
  [.RegularPyramidLateralEdges, .RightPrismLateralFaces, .CylinderGeneratrix, .ConeSectionIsoscelesTriangles]

/-- Counts the number of true propositions -/
def countTruePropositions (props : List GeometricProposition) : Nat :=
  props.filter isTrue |>.length

/-- Theorem stating that the number of true propositions is 3 -/
theorem true_propositions_count :
  countTruePropositions allPropositions = 3 := by
  sorry


end NUMINAMATH_CALUDE_true_propositions_count_l2663_266336


namespace NUMINAMATH_CALUDE_multiply_95_105_l2663_266363

theorem multiply_95_105 : 95 * 105 = 9975 := by
  sorry

end NUMINAMATH_CALUDE_multiply_95_105_l2663_266363


namespace NUMINAMATH_CALUDE_clothing_business_profit_l2663_266307

/-- Represents the daily profit function for a clothing business -/
def daily_profit (x : ℝ) : ℝ :=
  (40 - x) * (20 + 2 * x)

theorem clothing_business_profit :
  (∃ x : ℝ, x ≥ 0 ∧ daily_profit x = 1200) ∧
  (∀ y : ℝ, y ≥ 0 → daily_profit y ≠ 1800) := by
  sorry

end NUMINAMATH_CALUDE_clothing_business_profit_l2663_266307


namespace NUMINAMATH_CALUDE_vector_collinearity_problem_l2663_266323

/-- Given two 2D vectors are collinear if the cross product of their coordinates is zero -/
def collinear (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

/-- The problem statement -/
theorem vector_collinearity_problem (m : ℝ) :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-1, 2)
  collinear (m * a.1 + 4 * b.1, m * a.2 + 4 * b.2) (a.1 - 2 * b.1, a.2 - 2 * b.2) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_problem_l2663_266323


namespace NUMINAMATH_CALUDE_total_cost_proof_l2663_266361

def bow_cost : ℕ := 5
def vinegar_cost : ℕ := 2
def baking_soda_cost : ℕ := 1
def num_students : ℕ := 23

def total_cost_per_student : ℕ := bow_cost + vinegar_cost + baking_soda_cost

theorem total_cost_proof : 
  total_cost_per_student * num_students = 184 :=
by sorry

end NUMINAMATH_CALUDE_total_cost_proof_l2663_266361


namespace NUMINAMATH_CALUDE_fox_jeans_price_l2663_266395

/-- The regular price of Fox jeans -/
def F : ℝ := 15

/-- The regular price of Pony jeans -/
def pony_price : ℝ := 18

/-- The discount rate on Pony jeans -/
def pony_discount : ℝ := 0.1

/-- The sum of the two discount rates -/
def total_discount : ℝ := 0.22

/-- The total savings when purchasing 5 pairs of jeans -/
def total_savings : ℝ := 9

/-- The number of Fox jeans purchased -/
def fox_count : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_count : ℕ := 2

theorem fox_jeans_price :
  F = 15 ∧
  pony_price = 18 ∧
  pony_discount = 0.1 ∧
  total_discount = 0.22 ∧
  total_savings = 9 ∧
  fox_count = 3 ∧
  pony_count = 2 →
  F = 15 := by sorry

end NUMINAMATH_CALUDE_fox_jeans_price_l2663_266395


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l2663_266329

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 40 → area = (perimeter / 4) ^ 2 → area = 100 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l2663_266329


namespace NUMINAMATH_CALUDE_no_solutions_to_sqrt_equation_l2663_266304

theorem no_solutions_to_sqrt_equation :
  ∀ x : ℝ, x ≥ 4 →
  ¬∃ y : ℝ, y = Real.sqrt (x + 5 - 6 * Real.sqrt (x - 4)) + Real.sqrt (x + 18 - 8 * Real.sqrt (x - 4)) ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_to_sqrt_equation_l2663_266304


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l2663_266367

/-- Surface area of a cuboid -/
def surface_area (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: The surface area of a cuboid with length 8 cm, breadth 10 cm, and height 12 cm is 592 square centimeters -/
theorem cuboid_surface_area :
  surface_area 8 10 12 = 592 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l2663_266367


namespace NUMINAMATH_CALUDE_toms_age_ratio_l2663_266396

theorem toms_age_ratio (T N : ℝ) (h1 : T > 0) (h2 : N > 0) 
  (h3 : T - N = 2 * (T - 3 * N)) : T / N = 5 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l2663_266396


namespace NUMINAMATH_CALUDE_triangle_angle_b_value_l2663_266318

theorem triangle_angle_b_value 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : A + B + C = π) 
  (h3 : 0 < A ∧ A < π) 
  (h4 : 0 < B ∧ B < π) 
  (h5 : 0 < C ∧ C < π) 
  (h6 : (c - b) / (Real.sqrt 2 * c - a) = Real.sin A / (Real.sin B + Real.sin C)) : 
  B = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_b_value_l2663_266318


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2663_266302

/-- Represents a rectangular field -/
structure RectangularField where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangular field -/
def perimeter (field : RectangularField) : ℝ :=
  2 * (field.width + field.length)

/-- Calculates the area of a rectangular field -/
def area (field : RectangularField) : ℝ :=
  field.width * field.length

/-- Theorem: The area of a rectangular field with perimeter 120 meters and width 20 meters is 800 square meters -/
theorem rectangular_field_area :
  ∀ (field : RectangularField),
    field.width = 20 →
    perimeter field = 120 →
    area field = 800 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2663_266302


namespace NUMINAMATH_CALUDE_total_towels_folded_per_hour_l2663_266348

/-- Represents the number of towels a person can fold in one hour -/
def towels_per_hour (towels : ℕ) (minutes : ℕ) : ℕ :=
  (60 / minutes) * towels

/-- Proves that Jane, Kyla, and Anthony can fold 87 towels together in one hour -/
theorem total_towels_folded_per_hour :
  let jane_rate := towels_per_hour 3 5
  let kyla_rate := towels_per_hour 5 10
  let anthony_rate := towels_per_hour 7 20
  jane_rate + kyla_rate + anthony_rate = 87 := by
  sorry

#eval towels_per_hour 3 5 + towels_per_hour 5 10 + towels_per_hour 7 20

end NUMINAMATH_CALUDE_total_towels_folded_per_hour_l2663_266348


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2663_266362

theorem min_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  2 * x^2 + 8 * x * y + 32 * y^2 + 16 * y * z + 8 * z^2 ≥ 72 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    2 * x₀^2 + 8 * x₀ * y₀ + 32 * y₀^2 + 16 * y₀ * z₀ + 8 * z₀^2 = 72 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2663_266362


namespace NUMINAMATH_CALUDE_sum_of_integers_l2663_266383

theorem sum_of_integers (x y : ℕ+) (h1 : x.val^2 + y.val^2 = 250) (h2 : x.val * y.val = 108) :
  x.val + y.val = Real.sqrt 466 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2663_266383


namespace NUMINAMATH_CALUDE_wall_length_calculation_l2663_266310

theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 18 →
  wall_width = 32 →
  (mirror_side * mirror_side) * 2 = wall_width * (20.25 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l2663_266310


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_with_negation_greater_than_neg_80_l2663_266369

theorem largest_multiple_of_8_with_negation_greater_than_neg_80 : 
  ∀ n : ℤ, (∃ k : ℤ, n = 8 * k) → -n > -80 → n ≤ 72 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_with_negation_greater_than_neg_80_l2663_266369


namespace NUMINAMATH_CALUDE_min_value_expression_l2663_266371

theorem min_value_expression (m n : ℝ) (h : m - n^2 = 8) :
  58 ≤ m^2 - 3*n^2 + m - 14 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2663_266371


namespace NUMINAMATH_CALUDE_sqrt_equation_l2663_266327

theorem sqrt_equation (n : ℝ) : Real.sqrt (5 + n) = 7 → n = 44 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_l2663_266327


namespace NUMINAMATH_CALUDE_square_equation_implies_m_equals_negative_one_l2663_266356

theorem square_equation_implies_m_equals_negative_one :
  (∀ a : ℝ, a^2 + m * a + 1/4 = (a - 1/2)^2) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_implies_m_equals_negative_one_l2663_266356


namespace NUMINAMATH_CALUDE_syrup_cost_is_fifty_cents_l2663_266354

/-- The cost of a Build Your Own Hot Brownie dessert --/
def dessert_cost (brownie_cost ice_cream_cost nuts_cost syrup_cost : ℚ) : ℚ :=
  brownie_cost + 2 * ice_cream_cost + nuts_cost + 2 * syrup_cost

/-- Theorem: The syrup cost is $0.50 per serving --/
theorem syrup_cost_is_fifty_cents :
  ∃ (syrup_cost : ℚ),
    dessert_cost 2.5 1 1.5 syrup_cost = 7 ∧
    syrup_cost = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_syrup_cost_is_fifty_cents_l2663_266354


namespace NUMINAMATH_CALUDE_practice_for_five_months_l2663_266306

/-- Calculates the total piano practice hours over a given number of months -/
def total_practice_hours (weekly_hours : ℕ) (months : ℕ) : ℕ :=
  weekly_hours * 4 * months

/-- Theorem stating that practicing 4 hours weekly for 5 months results in 80 total hours -/
theorem practice_for_five_months : 
  total_practice_hours 4 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_practice_for_five_months_l2663_266306


namespace NUMINAMATH_CALUDE_protest_jail_time_l2663_266353

/-- Calculates the total combined weeks of jail time given protest conditions --/
theorem protest_jail_time 
  (days_of_protest : ℕ) 
  (num_cities : ℕ) 
  (arrests_per_day_per_city : ℕ) 
  (days_in_jail_before_trial : ℕ) 
  (sentence_weeks : ℕ) 
  (h1 : days_of_protest = 30)
  (h2 : num_cities = 21)
  (h3 : arrests_per_day_per_city = 10)
  (h4 : days_in_jail_before_trial = 4)
  (h5 : sentence_weeks = 2) :
  (days_of_protest * num_cities * arrests_per_day_per_city * days_in_jail_before_trial) / 7 +
  (days_of_protest * num_cities * arrests_per_day_per_city * sentence_weeks) / 2 = 9900 := by
  sorry


end NUMINAMATH_CALUDE_protest_jail_time_l2663_266353


namespace NUMINAMATH_CALUDE_unique_solution_for_all_z_l2663_266370

theorem unique_solution_for_all_z (x : ℚ) : 
  (∀ z : ℚ, 10 * x * z - 15 * z + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_all_z_l2663_266370


namespace NUMINAMATH_CALUDE_arctan_sum_equation_l2663_266359

theorem arctan_sum_equation (n : ℕ) : 
  (n > 0) → 
  (Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/n : ℝ) = π/2) → 
  n = 46 := by
sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_l2663_266359


namespace NUMINAMATH_CALUDE_white_marbles_count_l2663_266387

theorem white_marbles_count (total : ℕ) (blue : ℕ) (red : ℕ) (prob_red_or_white : ℚ) 
  (h1 : total = 30)
  (h2 : blue = 5)
  (h3 : red = 9)
  (h4 : prob_red_or_white = 25/30) :
  total - (blue + red) = 16 := by
  sorry

end NUMINAMATH_CALUDE_white_marbles_count_l2663_266387


namespace NUMINAMATH_CALUDE_initial_hours_were_eight_l2663_266349

/-- Represents the highway construction scenario -/
structure HighwayConstruction where
  initial_workforce : ℕ
  total_length : ℕ
  initial_duration : ℕ
  partial_duration : ℕ
  partial_completion : ℚ
  additional_workforce : ℕ
  new_daily_hours : ℕ

/-- Calculates the initial daily working hours -/
def calculate_initial_hours (scenario : HighwayConstruction) : ℚ :=
  (scenario.new_daily_hours * (scenario.initial_workforce + scenario.additional_workforce) * scenario.partial_duration * (1 - scenario.partial_completion)) /
  (scenario.initial_workforce * scenario.partial_duration * scenario.partial_completion)

/-- Theorem stating that the initial daily working hours were 8 -/
theorem initial_hours_were_eight (scenario : HighwayConstruction) 
  (h1 : scenario.initial_workforce = 100)
  (h2 : scenario.total_length = 2)
  (h3 : scenario.initial_duration = 50)
  (h4 : scenario.partial_duration = 25)
  (h5 : scenario.partial_completion = 1/3)
  (h6 : scenario.additional_workforce = 60)
  (h7 : scenario.new_daily_hours = 10) :
  calculate_initial_hours scenario = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_hours_were_eight_l2663_266349


namespace NUMINAMATH_CALUDE_complementary_angles_are_acute_l2663_266385

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- An angle is acute if it is less than 90 degrees -/
def acute (a : ℝ) : Prop := a < 90

/-- For any two complementary angles, both angles are always acute -/
theorem complementary_angles_are_acute (a b : ℝ) (h : complementary a b) : 
  acute a ∧ acute b := by sorry

end NUMINAMATH_CALUDE_complementary_angles_are_acute_l2663_266385


namespace NUMINAMATH_CALUDE_remainder_of_98765432101_mod_240_l2663_266365

theorem remainder_of_98765432101_mod_240 :
  98765432101 % 240 = 61 := by sorry

end NUMINAMATH_CALUDE_remainder_of_98765432101_mod_240_l2663_266365


namespace NUMINAMATH_CALUDE_bike_wheel_radius_increase_l2663_266308

theorem bike_wheel_radius_increase (initial_circumference final_circumference : ℝ) 
  (h1 : initial_circumference = 30)
  (h2 : final_circumference = 40) :
  (final_circumference / (2 * Real.pi)) - (initial_circumference / (2 * Real.pi)) = 5 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_bike_wheel_radius_increase_l2663_266308


namespace NUMINAMATH_CALUDE_inequality_solution_l2663_266337

theorem inequality_solution : 
  ∀ x : ℝ, (|x - 1| + |x + 2| + |x| < 7) ↔ (-2 < x ∧ x < 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2663_266337


namespace NUMINAMATH_CALUDE_picture_placement_l2663_266345

/-- Given a wall and a picture with specified widths and offset, calculate the distance from the nearest end of the wall to the nearest edge of the picture. -/
theorem picture_placement (wall_width picture_width offset : ℝ) 
  (hw : wall_width = 25)
  (hp : picture_width = 5)
  (ho : offset = 2) :
  let center := (wall_width - picture_width) / 2
  let distance_to_nearest_edge := center - offset
  distance_to_nearest_edge = 8 := by sorry

end NUMINAMATH_CALUDE_picture_placement_l2663_266345


namespace NUMINAMATH_CALUDE_left_handed_jazz_no_glasses_l2663_266372

/-- Represents a club with members having various characteristics -/
structure Club where
  total : Nat
  leftHanded : Nat
  jazzLovers : Nat
  rightHandedJazzDislikers : Nat
  glassesWearers : Nat

/-- The main theorem to be proved -/
theorem left_handed_jazz_no_glasses (c : Club)
  (h_total : c.total = 50)
  (h_left : c.leftHanded = 22)
  (h_jazz : c.jazzLovers = 35)
  (h_right_no_jazz : c.rightHandedJazzDislikers = 5)
  (h_glasses : c.glassesWearers = 10)
  (h_hand_exclusive : c.leftHanded + (c.total - c.leftHanded) = c.total)
  (h_glasses_independent : True) :
  ∃ x : Nat, x = 4 ∧ 
    x = c.leftHanded + c.jazzLovers - c.total + c.rightHandedJazzDislikers - c.glassesWearers :=
sorry


end NUMINAMATH_CALUDE_left_handed_jazz_no_glasses_l2663_266372


namespace NUMINAMATH_CALUDE_sin_minus_abs_sin_range_l2663_266301

theorem sin_minus_abs_sin_range :
  ∀ y : ℝ, (∃ x : ℝ, y = Real.sin x - |Real.sin x|) ↔ -2 ≤ y ∧ y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_abs_sin_range_l2663_266301


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l2663_266347

theorem different_color_chips_probability : 
  let total_chips := 20
  let blue_chips := 4
  let red_chips := 3
  let yellow_chips := 2
  let green_chips := 5
  let orange_chips := 6
  let prob_diff_color := 
    (blue_chips / total_chips) * ((total_chips - blue_chips) / total_chips) +
    (red_chips / total_chips) * ((total_chips - red_chips) / total_chips) +
    (yellow_chips / total_chips) * ((total_chips - yellow_chips) / total_chips) +
    (green_chips / total_chips) * ((total_chips - green_chips) / total_chips) +
    (orange_chips / total_chips) * ((total_chips - orange_chips) / total_chips)
  prob_diff_color = 31 / 40 := by
  sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l2663_266347


namespace NUMINAMATH_CALUDE_inequality_proof_l2663_266333

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2663_266333


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l2663_266325

theorem partial_fraction_sum (A B C D E : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) →
  A + B + C + D + E = 1 / 30 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l2663_266325


namespace NUMINAMATH_CALUDE_line_through_point_l2663_266398

theorem line_through_point (a : ℚ) : 
  (3 * a * 2 + (2 * a + 3) * (-5) = 4 * a + 6) → a = -21 / 8 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2663_266398


namespace NUMINAMATH_CALUDE_equation_solution_l2663_266394

theorem equation_solution :
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2663_266394


namespace NUMINAMATH_CALUDE_men_sent_to_project_l2663_266389

/-- Represents the number of men sent to another project -/
def men_sent : ℕ := 33

/-- Represents the original number of men -/
def original_men : ℕ := 50

/-- Represents the original number of days to complete the work -/
def original_days : ℕ := 10

/-- Represents the new number of days to complete the work -/
def new_days : ℕ := 30

/-- Theorem stating that given the original conditions and the new completion time,
    the number of men sent to another project is 33 -/
theorem men_sent_to_project :
  (original_men * original_days = (original_men - men_sent) * new_days) →
  men_sent = 33 := by
  sorry


end NUMINAMATH_CALUDE_men_sent_to_project_l2663_266389


namespace NUMINAMATH_CALUDE_isosceles_triangle_circle_properties_l2663_266351

/-- An isosceles triangle with base 48 and side length 30 -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  isIsosceles : base = 48 ∧ side = 30

/-- Properties of the inscribed and circumscribed circles of the isosceles triangle -/
def CircleProperties (t : IsoscelesTriangle) : Prop :=
  ∃ (r R d : ℝ),
    r = 8 ∧  -- radius of inscribed circle
    R = 25 ∧  -- radius of circumscribed circle
    d = 15 ∧  -- distance between centers
    r > 0 ∧ R > 0 ∧ d > 0

/-- Theorem stating the properties of the inscribed and circumscribed circles -/
theorem isosceles_triangle_circle_properties (t : IsoscelesTriangle) :
  CircleProperties t :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_circle_properties_l2663_266351


namespace NUMINAMATH_CALUDE_no_real_roots_l2663_266335

theorem no_real_roots : ∀ x : ℝ, x ≠ 2 → 
  (3 * x^2) / (x - 2) - (3 * x + 8) / 2 + (5 - 9 * x) / (x - 2) + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2663_266335


namespace NUMINAMATH_CALUDE_six_throws_total_skips_l2663_266330

def stone_skips (n : ℕ) : ℕ := n^2 + n

def total_skips (num_throws : ℕ) : ℕ :=
  (List.range num_throws).map stone_skips |>.sum

theorem six_throws_total_skips :
  total_skips 5 + 2 * stone_skips 6 = 154 := by
  sorry

end NUMINAMATH_CALUDE_six_throws_total_skips_l2663_266330


namespace NUMINAMATH_CALUDE_xy_system_solution_l2663_266328

theorem xy_system_solution (x y : ℝ) 
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) : 
  x^2 + y^2 = 5104 / 81 := by
sorry

end NUMINAMATH_CALUDE_xy_system_solution_l2663_266328


namespace NUMINAMATH_CALUDE_marys_garbage_bill_is_164_l2663_266386

/-- Calculates Mary's garbage bill based on the given conditions --/
def calculate_garbage_bill : ℝ :=
  let weeks_in_month : ℕ := 4
  let trash_bin_charge : ℝ := 10
  let recycling_bin_charge : ℝ := 5
  let green_waste_bin_charge : ℝ := 3
  let trash_bins : ℕ := 2
  let recycling_bins : ℕ := 1
  let green_waste_bins : ℕ := 1
  let flat_service_fee : ℝ := 15
  let trash_discount : ℝ := 0.18
  let recycling_discount : ℝ := 0.12
  let green_waste_discount : ℝ := 0.10
  let recycling_fine : ℝ := 20
  let overfilling_fine : ℝ := 15
  let unsorted_green_waste_fine : ℝ := 10
  let late_payment_fee : ℝ := 10

  let weekly_cost : ℝ := trash_bin_charge * trash_bins + recycling_bin_charge * recycling_bins + green_waste_bin_charge * green_waste_bins
  let monthly_cost : ℝ := weekly_cost * weeks_in_month
  let weekly_discount : ℝ := trash_bin_charge * trash_bins * trash_discount + recycling_bin_charge * recycling_bins * recycling_discount + green_waste_bin_charge * green_waste_bins * green_waste_discount
  let monthly_discount : ℝ := weekly_discount * weeks_in_month
  let adjusted_monthly_cost : ℝ := monthly_cost - monthly_discount + flat_service_fee
  let total_fines : ℝ := recycling_fine + overfilling_fine + unsorted_green_waste_fine + late_payment_fee

  adjusted_monthly_cost + total_fines

/-- Theorem stating that Mary's garbage bill is equal to $164 --/
theorem marys_garbage_bill_is_164 : calculate_garbage_bill = 164 := by
  sorry

end NUMINAMATH_CALUDE_marys_garbage_bill_is_164_l2663_266386


namespace NUMINAMATH_CALUDE_function_identity_l2663_266339

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_identity :
  (∀ x : ℝ, f (x + 1) = x^2 - 2*x) →
  (∀ x : ℝ, f x = x^2 - 4*x + 3) :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l2663_266339


namespace NUMINAMATH_CALUDE_h_in_terms_of_f_l2663_266382

-- Define the domain of f
def I : Set ℝ := Set.Icc (-3 : ℝ) 3

-- Define f as a function on the interval I
variable (f : I → ℝ)

-- Define h as a function derived from f
def h (x : ℝ) : ℝ := -(f ⟨x + 6, sorry⟩)

-- Theorem statement
theorem h_in_terms_of_f (x : ℝ) : h f x = -f ⟨x + 6, sorry⟩ := by sorry

end NUMINAMATH_CALUDE_h_in_terms_of_f_l2663_266382


namespace NUMINAMATH_CALUDE_xyz_problem_l2663_266364

theorem xyz_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 132)
  (h2 : y * (z + x) = 152)
  (h3 : x * y * z = 160) :
  z * (x + y) = 131.92 := by
  sorry

end NUMINAMATH_CALUDE_xyz_problem_l2663_266364


namespace NUMINAMATH_CALUDE_pool_tiles_l2663_266346

theorem pool_tiles (total_needed : ℕ) (blue_tiles : ℕ) (additional_needed : ℕ) 
  (h1 : total_needed = 100)
  (h2 : blue_tiles = 48)
  (h3 : additional_needed = 20) :
  total_needed - additional_needed - blue_tiles = 32 := by
  sorry

#check pool_tiles

end NUMINAMATH_CALUDE_pool_tiles_l2663_266346


namespace NUMINAMATH_CALUDE_min_value_of_ab_l2663_266303

theorem min_value_of_ab (a b : ℝ) (h : (1 / a) + (1 / b) = Real.sqrt (a * b)) : 
  2 ≤ a * b ∧ ∃ (x y : ℝ), (1 / x) + (1 / y) = Real.sqrt (x * y) ∧ x * y = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_ab_l2663_266303


namespace NUMINAMATH_CALUDE_school_referendum_l2663_266332

theorem school_referendum (total_students : ℕ) (first_issue : ℕ) (second_issue : ℕ) (against_both : ℕ)
  (h1 : total_students = 150)
  (h2 : first_issue = 110)
  (h3 : second_issue = 95)
  (h4 : against_both = 15) :
  first_issue + second_issue - (total_students - against_both) = 70 := by
  sorry

end NUMINAMATH_CALUDE_school_referendum_l2663_266332


namespace NUMINAMATH_CALUDE_tank_capacity_l2663_266374

theorem tank_capacity (x : ℝ) 
  (h1 : x / 8 + 120 = x / 2) : x = 320 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l2663_266374


namespace NUMINAMATH_CALUDE_smallest_n_for_342_fraction_l2663_266326

/-- Checks if two numbers are relatively prime -/
def are_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Checks if the decimal representation of m/n contains 342 consecutively -/
def contains_342 (m n : ℕ) : Prop :=
  ∃ k : ℕ, 342 * n ≤ 1000 * k * m ∧ 1000 * k * m < 343 * n

theorem smallest_n_for_342_fraction :
  (∃ n : ℕ, n > 0 ∧
    (∃ m : ℕ, m > 0 ∧ m < n ∧
      are_relatively_prime m n ∧
      contains_342 m n)) ∧
  (∀ n : ℕ, n > 0 →
    (∃ m : ℕ, m > 0 ∧ m < n ∧
      are_relatively_prime m n ∧
      contains_342 m n) →
    n ≥ 331) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_342_fraction_l2663_266326


namespace NUMINAMATH_CALUDE_megan_current_seashells_l2663_266355

-- Define the number of seashells Megan currently has
def current_seashells : ℕ := 19

-- Define the number of additional seashells Megan needs
def additional_seashells : ℕ := 6

-- Define the total number of seashells Megan will have after adding more
def total_seashells : ℕ := 25

-- Theorem stating that Megan currently has 19 seashells
theorem megan_current_seashells : 
  current_seashells = total_seashells - additional_seashells :=
by
  sorry

end NUMINAMATH_CALUDE_megan_current_seashells_l2663_266355


namespace NUMINAMATH_CALUDE_smallest_absolute_value_l2663_266338

theorem smallest_absolute_value : ∀ (a b c : ℤ), 
  a = -3 → b = -2 → c = 1 → 
  |0| < |a| ∧ |0| < |b| ∧ |0| < |c| :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_l2663_266338


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2663_266319

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_equality : (3 : ℂ) / (1 - i)^2 = (3/2 : ℂ) * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2663_266319


namespace NUMINAMATH_CALUDE_two_sided_iced_subcubes_count_l2663_266341

/-- Represents a cube with icing on all sides -/
structure IcedCube where
  size : Nat
  deriving Repr

/-- Counts the number of subcubes with icing on exactly two sides -/
def count_two_sided_iced_subcubes (cube : IcedCube) : Nat :=
  sorry

/-- Theorem stating that a 5×5×5 iced cube has 40 subcubes with icing on exactly two sides -/
theorem two_sided_iced_subcubes_count (cube : IcedCube) (h : cube.size = 5) : 
  count_two_sided_iced_subcubes cube = 40 := by
  sorry

end NUMINAMATH_CALUDE_two_sided_iced_subcubes_count_l2663_266341


namespace NUMINAMATH_CALUDE_business_investment_l2663_266390

/-- Prove that the total investment is 90000 given the conditions of the business problem -/
theorem business_investment (a b c : ℕ) (total_profit a_share : ℕ) : 
  a = b + 6000 →
  c = b + 3000 →
  total_profit = 8640 →
  a_share = 3168 →
  a_share * (a + b + c) = a * total_profit →
  a + b + c = 90000 :=
by sorry

end NUMINAMATH_CALUDE_business_investment_l2663_266390


namespace NUMINAMATH_CALUDE_birthday_stickers_l2663_266357

/-- Represents the number of stickers Luke has at different stages --/
structure StickerCount where
  initial : ℕ
  bought : ℕ
  birthday : ℕ
  givenAway : ℕ
  used : ℕ
  final : ℕ

/-- Theorem stating the number of stickers Luke got for his birthday --/
theorem birthday_stickers (s : StickerCount) 
  (h1 : s.initial = 20)
  (h2 : s.bought = 12)
  (h3 : s.givenAway = 5)
  (h4 : s.used = 8)
  (h5 : s.final = 39)
  (h6 : s.final = s.initial + s.bought + s.birthday - s.givenAway - s.used) :
  s.birthday = 20 := by
  sorry

end NUMINAMATH_CALUDE_birthday_stickers_l2663_266357


namespace NUMINAMATH_CALUDE_savings_account_balance_l2663_266381

theorem savings_account_balance 
  (total : ℕ) 
  (checking : ℕ) 
  (h1 : total = 9844)
  (h2 : checking = 6359) :
  total - checking = 3485 :=
by sorry

end NUMINAMATH_CALUDE_savings_account_balance_l2663_266381


namespace NUMINAMATH_CALUDE_complex_expansion_l2663_266315

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_expansion : i * (1 + i)^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expansion_l2663_266315


namespace NUMINAMATH_CALUDE_brownie_distribution_l2663_266358

theorem brownie_distribution (columns rows people : ℕ) : 
  columns = 6 → rows = 3 → people = 6 → (columns * rows) / people = 3 := by
  sorry

end NUMINAMATH_CALUDE_brownie_distribution_l2663_266358


namespace NUMINAMATH_CALUDE_carrot_distribution_l2663_266320

theorem carrot_distribution (total : ℕ) (leftover : ℕ) (people : ℕ) : 
  total = 74 → 
  leftover = 2 → 
  people > 1 → 
  people < 72 → 
  (total - leftover) % people = 0 → 
  72 % people = 0 := by
sorry

end NUMINAMATH_CALUDE_carrot_distribution_l2663_266320


namespace NUMINAMATH_CALUDE_m_range_condition_l2663_266393

def A : Set ℝ := Set.Ioo (-2) 2
def B (m : ℝ) : Set ℝ := Set.Ici (m - 1)

theorem m_range_condition (m : ℝ) : A ⊆ B m ↔ m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_m_range_condition_l2663_266393


namespace NUMINAMATH_CALUDE_cosine_equation_solution_l2663_266384

theorem cosine_equation_solution (x : ℝ) : 
  (Real.cos x + 2 * Real.cos (6 * x))^2 = 9 + (Real.sin (3 * x))^2 ↔ 
  ∃ k : ℤ, x = 2 * k * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cosine_equation_solution_l2663_266384


namespace NUMINAMATH_CALUDE_f_is_even_l2663_266321

-- Define a real-valued function g
variable (g : ℝ → ℝ)

-- Define the property of g being an odd function
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Define function f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^4)|

-- Define the property of f being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : IsOdd g) : IsEven (f g) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l2663_266321


namespace NUMINAMATH_CALUDE_parabola_f_value_l2663_266300

/-- A parabola with equation y = dx^2 + ex + f -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.d * x^2 + p.e * x + p.f

theorem parabola_f_value (p : Parabola) :
  p.y_coord (-1) = -2 →  -- vertex condition
  p.y_coord 0 = -1.5 →   -- point condition
  p.f = -1.5 := by
sorry

end NUMINAMATH_CALUDE_parabola_f_value_l2663_266300


namespace NUMINAMATH_CALUDE_race_time_difference_per_hurdle_l2663_266380

/-- Given a race with the following parameters:
  * Total distance: 120 meters
  * Hurdles placed every 20 meters
  * Runner A's total time: 36 seconds
  * Runner B's total time: 45 seconds
Prove that the time difference between the runners at each hurdle is 1.5 seconds. -/
theorem race_time_difference_per_hurdle 
  (total_distance : ℝ) 
  (hurdle_interval : ℝ)
  (runner_a_time : ℝ)
  (runner_b_time : ℝ)
  (h1 : total_distance = 120)
  (h2 : hurdle_interval = 20)
  (h3 : runner_a_time = 36)
  (h4 : runner_b_time = 45) :
  (runner_b_time - runner_a_time) / (total_distance / hurdle_interval) = 1.5 := by
sorry

end NUMINAMATH_CALUDE_race_time_difference_per_hurdle_l2663_266380


namespace NUMINAMATH_CALUDE_permutations_of_four_objects_l2663_266366

theorem permutations_of_four_objects : Nat.factorial 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_four_objects_l2663_266366


namespace NUMINAMATH_CALUDE_no_real_solutions_l2663_266379

theorem no_real_solutions : ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 12*y + 36 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2663_266379


namespace NUMINAMATH_CALUDE_combined_tax_rate_l2663_266343

/-- Given Mork's and Mindy's tax rates and relative incomes, compute their combined tax rate -/
theorem combined_tax_rate (mork_rate mindy_rate : ℚ) (income_ratio : ℕ) :
  mork_rate = 45/100 →
  mindy_rate = 25/100 →
  income_ratio = 4 →
  (mork_rate + income_ratio * mindy_rate) / (1 + income_ratio) = 29/100 := by
sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l2663_266343


namespace NUMINAMATH_CALUDE_smaller_number_of_product_323_and_difference_2_l2663_266344

theorem smaller_number_of_product_323_and_difference_2 :
  ∀ x y : ℕ+,
  (x : ℕ) * y = 323 →
  (x : ℕ) - y = 2 →
  y = 17 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_of_product_323_and_difference_2_l2663_266344


namespace NUMINAMATH_CALUDE_article_pricing_gain_l2663_266324

/-- Proves that if selling an article at 2/3 of its original price results in a 10% loss,
    then selling it at the original price results in a 35% gain. -/
theorem article_pricing_gain (P : ℝ) (P_pos : P > 0) :
  (2 / 3 : ℝ) * P = (9 / 10 : ℝ) * ((20 / 27 : ℝ) * P) →
  ((P - (20 / 27 : ℝ) * P) / ((20 / 27 : ℝ) * P)) * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_article_pricing_gain_l2663_266324


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l2663_266313

theorem restaurant_bill_proof (total_friends : Nat) (paying_friends : Nat) (extra_payment : ℚ) : 
  total_friends = 10 →
  paying_friends = 8 →
  extra_payment = 3 →
  ∃ (total_bill : ℚ), total_bill = 120 ∧ 
    paying_friends * (total_bill / total_friends + extra_payment) = total_bill :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l2663_266313


namespace NUMINAMATH_CALUDE_simplify_expression_l2663_266314

theorem simplify_expression (a b : ℝ) :
  (15 * a + 45 * b) + (12 * a + 35 * b) - (7 * a + 30 * b) - (3 * a + 15 * b) = 17 * a + 35 * b :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2663_266314


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l2663_266392

/-- The surface area of a sphere containing all vertices of a rectangular solid -/
theorem sphere_surface_area_rectangular_solid (l w h : ℝ) (r : ℝ) : 
  l = 2 → w = 1 → h = 2 → 
  r^2 = (l^2 + w^2 + h^2) / 4 →
  4 * Real.pi * r^2 = 9 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l2663_266392


namespace NUMINAMATH_CALUDE_minimum_detectors_l2663_266322

/-- Represents a position on the board -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a detector on the board -/
structure Detector :=
  (pos : Position)

/-- Represents a ship on the board -/
structure Ship :=
  (pos : Position)
  (size : Nat)

def boardSize : Nat := 2015
def shipSize : Nat := 1500

/-- Checks if a detector can detect a ship at a given position -/
def canDetect (d : Detector) (s : Ship) : Prop :=
  d.pos.x ≥ s.pos.x ∧ d.pos.x < s.pos.x + s.size ∧
  d.pos.y ≥ s.pos.y ∧ d.pos.y < s.pos.y + s.size

/-- Checks if a set of detectors can determine the position of any ship -/
def canDetermineShipPosition (detectors : List Detector) : Prop :=
  ∀ (s1 s2 : Ship),
    (∀ (d : Detector), d ∈ detectors → (canDetect d s1 ↔ canDetect d s2)) →
    s1 = s2

theorem minimum_detectors :
  ∃ (detectors : List Detector),
    detectors.length = 1030 ∧
    canDetermineShipPosition detectors ∧
    ∀ (d : List Detector),
      d.length < 1030 →
      ¬ canDetermineShipPosition d :=
sorry

end NUMINAMATH_CALUDE_minimum_detectors_l2663_266322


namespace NUMINAMATH_CALUDE_buses_met_count_l2663_266331

/-- Represents the schedule of buses between Moscow and Voronezh -/
structure BusSchedule where
  moscow_departure_minute : Nat
  voronezh_departure_minute : Nat
  travel_time_hours : Nat

/-- Calculates the number of buses from Voronezh that a bus from Moscow will meet -/
def buses_met (schedule : BusSchedule) : Nat :=
  2 * schedule.travel_time_hours

/-- Theorem stating that a bus from Moscow will meet 16 buses from Voronezh -/
theorem buses_met_count (schedule : BusSchedule) 
  (h1 : schedule.moscow_departure_minute = 0)
  (h2 : schedule.voronezh_departure_minute = 30)
  (h3 : schedule.travel_time_hours = 8) : 
  buses_met schedule = 16 := by
  sorry

#eval buses_met { moscow_departure_minute := 0, voronezh_departure_minute := 30, travel_time_hours := 8 }

end NUMINAMATH_CALUDE_buses_met_count_l2663_266331


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2663_266377

theorem arithmetic_sequence_sum (d : ℤ) : ∃ (S : ℕ → ℤ) (a : ℕ → ℤ), 
  (∀ n, a (n + 1) = a n + d) ∧  -- Arithmetic sequence definition
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) ∧  -- Sum formula
  a 1 = 190 ∧  -- First term
  S 20 > 0 ∧  -- S₂₀ > 0
  S 24 < 0 ∧  -- S₂₄ < 0
  d = -17  -- One possible value for d
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2663_266377


namespace NUMINAMATH_CALUDE_music_school_enrollment_cost_l2663_266305

/-- Calculates the total cost for music school enrollment for four siblings --/
theorem music_school_enrollment_cost :
  let regular_tuition : ℕ := 45
  let early_bird_discount : ℕ := 15
  let first_sibling_discount : ℕ := 15
  let additional_sibling_discount : ℕ := 10
  let weekend_class_extra : ℕ := 20
  let multi_instrument_discount : ℕ := 10

  let ali_cost : ℕ := regular_tuition - early_bird_discount
  let matt_cost : ℕ := regular_tuition - first_sibling_discount
  let jane_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra - multi_instrument_discount
  let sarah_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra - multi_instrument_discount

  ali_cost + matt_cost + jane_cost + sarah_cost = 150 :=
by
  sorry


end NUMINAMATH_CALUDE_music_school_enrollment_cost_l2663_266305


namespace NUMINAMATH_CALUDE_smallest_y_in_geometric_sequence_125_l2663_266350

/-- A geometric sequence of three positive integers with product 125 -/
structure GeometricSequence125 where
  x : ℕ+
  y : ℕ+
  z : ℕ+
  geometric : ∃ (r : ℚ), y = x * r ∧ z = y * r
  product : x * y * z = 125

/-- The smallest possible value of y in a geometric sequence of three positive integers with product 125 -/
theorem smallest_y_in_geometric_sequence_125 : 
  ∀ (seq : GeometricSequence125), seq.y ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_y_in_geometric_sequence_125_l2663_266350


namespace NUMINAMATH_CALUDE_initial_average_production_l2663_266340

theorem initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) 
  (h1 : n = 14)
  (h2 : today_production = 90)
  (h3 : new_average = 62) :
  (n * (n + 1) * new_average - n * today_production) / n = 60 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_production_l2663_266340


namespace NUMINAMATH_CALUDE_isabella_exchange_l2663_266368

/-- Exchange rate from U.S. dollars to Euros -/
def exchange_rate : ℚ := 5 / 8

/-- The amount of Euros Isabella spent -/
def euros_spent : ℕ := 80

theorem isabella_exchange (d : ℕ) : 
  (exchange_rate * d : ℚ) - euros_spent = 2 * d → d = 58 := by
  sorry

end NUMINAMATH_CALUDE_isabella_exchange_l2663_266368


namespace NUMINAMATH_CALUDE_infinite_sum_equality_l2663_266312

/-- Given a positive real number t satisfying t^3 + 3/7*t - 1 = 0,
    the infinite sum t^3 + 2t^6 + 3t^9 + 4t^12 + ... equals (49/9)*t -/
theorem infinite_sum_equality (t : ℝ) (ht : t > 0) (heq : t^3 + 3/7*t - 1 = 0) :
  ∑' n, (n : ℝ) * t^(3*n) = 49/9 * t := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equality_l2663_266312


namespace NUMINAMATH_CALUDE_set_of_multiples_of_six_l2663_266375

def is_closed_under_addition_subtraction (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S ∧ (x - y) ∈ S

def smallest_positive (S : Set ℝ) (a : ℝ) : Prop :=
  a ∈ S ∧ a > 0 ∧ ∀ x ∈ S, x > 0 → x ≥ a

theorem set_of_multiples_of_six (S : Set ℝ) :
  S.Nonempty →
  is_closed_under_addition_subtraction S →
  smallest_positive S 6 →
  S = {x : ℝ | ∃ n : ℤ, x = 6 * n} :=
by sorry

end NUMINAMATH_CALUDE_set_of_multiples_of_six_l2663_266375


namespace NUMINAMATH_CALUDE_polynomial_multiplication_correction_l2663_266311

theorem polynomial_multiplication_correction (x a b : ℚ) : 
  (2*x-a)*(3*x+b) = 6*x^2 + 11*x - 10 →
  (2*x+a)*(x+b) = 2*x^2 - 9*x + 10 →
  (2*x+a)*(3*x+b) = 6*x^2 - 19*x + 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_correction_l2663_266311
