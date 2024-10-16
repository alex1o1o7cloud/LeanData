import Mathlib

namespace NUMINAMATH_CALUDE_license_plate_count_l2631_263194

/-- The number of consonants in the English alphabet -/
def num_consonants : ℕ := 20

/-- The number of even digits -/
def num_even_digits : ℕ := 5

/-- The number of possible license plates meeting the specified criteria -/
def num_license_plates : ℕ := num_consonants * 1 * num_consonants * num_even_digits

/-- Theorem stating that the number of license plates meeting the criteria is 2000 -/
theorem license_plate_count : num_license_plates = 2000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2631_263194


namespace NUMINAMATH_CALUDE_set_operation_result_arithmetic_expression_result_l2631_263164

-- Define the sets A, B, and C
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -2 < x ∧ x < 2}
def C : Set ℝ := {x | -3 < x ∧ x < 5}

-- Theorem 1: Set operation result
theorem set_operation_result : (A ∪ B) ∩ C = {x : ℝ | -2 < x ∧ x < 5} := by sorry

-- Theorem 2: Arithmetic expression result
theorem arithmetic_expression_result :
  (2 + 1/4)^(1/2) - (-9.6)^0 - (3 + 3/8)^(-2/3) + (1.5)^(-2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_set_operation_result_arithmetic_expression_result_l2631_263164


namespace NUMINAMATH_CALUDE_least_distinct_values_l2631_263118

theorem least_distinct_values (n : ℕ) (mode_freq : ℕ) (list_size : ℕ) 
  (h1 : n > 0)
  (h2 : mode_freq = 13)
  (h3 : list_size = 2023) :
  (∃ (list : List ℕ),
    list.length = list_size ∧
    (∃ (mode : ℕ), list.count mode = mode_freq ∧
      ∀ x : ℕ, x ≠ mode → list.count x < mode_freq) ∧
    (∀ m : ℕ, m < n → ¬∃ (list' : List ℕ),
      list'.length = list_size ∧
      (∃ (mode' : ℕ), list'.count mode' = mode_freq ∧
        ∀ x : ℕ, x ≠ mode' → list'.count x < mode_freq) ∧
      list'.toFinset.card = m)) →
  n = 169 := by
sorry

end NUMINAMATH_CALUDE_least_distinct_values_l2631_263118


namespace NUMINAMATH_CALUDE_function_properties_l2631_263124

noncomputable def f (x : ℝ) : ℝ := 2 * x / (x + 1)

theorem function_properties :
  (∀ x : ℝ, x ≠ -1 → f x = 2 * x / (x + 1)) ∧
  f 1 = 1 ∧
  f (-2) = 4 ∧
  (∃ c : ℝ, ∀ x : ℝ, x ≠ -1 → f x + f (c - x) = 4) ∧
  (∀ x m : ℝ, x ∈ Set.Icc 1 2 → 2 < m → m ≤ 4 → f x ≤ 2 * m / ((x + 1) * |x - m|)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2631_263124


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l2631_263103

theorem chicken_wings_distribution (num_friends : ℕ) (pre_cooked : ℕ) (additional_cooked : ℕ) :
  num_friends = 3 →
  pre_cooked = 8 →
  additional_cooked = 10 →
  (pre_cooked + additional_cooked) / num_friends = 6 := by
  sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l2631_263103


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2631_263154

theorem perpendicular_vectors (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![-2, x]
  (∀ i, i < 2 → a i * b i = 0) → x = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2631_263154


namespace NUMINAMATH_CALUDE_money_distribution_l2631_263114

/-- Given a total amount of money and the fraction one person has relative to the others,
    calculate how much money that person has. -/
theorem money_distribution (total : ℕ) (fraction : ℚ) (person_amount : ℕ) : 
  total = 7000 →
  fraction = 2 / 3 →
  person_amount = total * (fraction / (1 + fraction)) →
  person_amount = 2800 := by
  sorry

#check money_distribution

end NUMINAMATH_CALUDE_money_distribution_l2631_263114


namespace NUMINAMATH_CALUDE_boat_speed_l2631_263130

/-- Given a boat that travels 11 km/h downstream and 5 km/h upstream, 
    its speed in still water is 8 km/h. -/
theorem boat_speed (downstream upstream : ℝ) 
  (h1 : downstream = 11) 
  (h2 : upstream = 5) : 
  (downstream + upstream) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l2631_263130


namespace NUMINAMATH_CALUDE_find_B_value_l2631_263134

theorem find_B_value (A B : ℕ) (h1 : A < 10) (h2 : B < 10) 
  (h3 : 600 + 10 * A + 5 + 100 * B + 3 = 748) : B = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_B_value_l2631_263134


namespace NUMINAMATH_CALUDE_range_of_a_l2631_263133

theorem range_of_a (x y a : ℝ) (h1 : x < y) (h2 : (a - 3) * x > (a - 3) * y) : a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2631_263133


namespace NUMINAMATH_CALUDE_zero_lite_soda_bottles_l2631_263146

/-- The number of bottles of lite soda in a grocery store -/
def lite_soda_bottles (regular_soda diet_soda total_regular_and_diet : ℕ) : ℕ :=
  total_regular_and_diet - (regular_soda + diet_soda)

/-- Theorem: The number of lite soda bottles is 0 -/
theorem zero_lite_soda_bottles :
  lite_soda_bottles 49 40 89 = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_lite_soda_bottles_l2631_263146


namespace NUMINAMATH_CALUDE_product_remainder_is_one_l2631_263151

def sequence1 : List Nat := List.range 10 |>.map (fun n => 3 + 10 * n)
def sequence2 : List Nat := List.range 10 |>.map (fun n => 7 + 10 * n)

theorem product_remainder_is_one :
  (sequence1.prod * sequence2.prod) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_is_one_l2631_263151


namespace NUMINAMATH_CALUDE_johns_age_doubles_l2631_263110

/-- Represents John's current age -/
def current_age : ℕ := 18

/-- Represents the number of years ago when John's age was half of a future age -/
def years_ago : ℕ := 5

/-- Represents the number of years until John's age is twice his age from five years ago -/
def years_until_double : ℕ := 8

/-- Theorem stating that in 8 years, John's age will be twice his age from five years ago -/
theorem johns_age_doubles : 
  2 * (current_age - years_ago) = current_age + years_until_double := by
  sorry

end NUMINAMATH_CALUDE_johns_age_doubles_l2631_263110


namespace NUMINAMATH_CALUDE_total_lines_eq_88_l2631_263139

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- The number of triangles drawn -/
def num_triangles : ℕ := 12

/-- The number of squares drawn -/
def num_squares : ℕ := 8

/-- The number of pentagons drawn -/
def num_pentagons : ℕ := 4

/-- The total number of lines drawn -/
def total_lines : ℕ := num_triangles * triangle_sides + num_squares * square_sides + num_pentagons * pentagon_sides

theorem total_lines_eq_88 : total_lines = 88 := by
  sorry

end NUMINAMATH_CALUDE_total_lines_eq_88_l2631_263139


namespace NUMINAMATH_CALUDE_linear_function_solution_l2631_263116

/-- A linear function passing through (0,2) with negative slope -/
def linearFunction (k : ℝ) (x : ℝ) : ℝ := k * x + 2

theorem linear_function_solution :
  ∀ k : ℝ, k < 0 → linearFunction (-1) = linearFunction k := by sorry

end NUMINAMATH_CALUDE_linear_function_solution_l2631_263116


namespace NUMINAMATH_CALUDE_darnells_average_yards_is_11_l2631_263189

/-- Calculates Darnell's average yards rushed per game given the total yards and other players' yards. -/
def darnells_average_yards (total_yards : ℕ) (malik_yards_per_game : ℕ) (josiah_yards_per_game : ℕ) (num_games : ℕ) : ℕ := 
  (total_yards - (malik_yards_per_game * num_games + josiah_yards_per_game * num_games)) / num_games

/-- Proves that Darnell's average yards rushed per game is 11 yards given the problem conditions. -/
theorem darnells_average_yards_is_11 : 
  darnells_average_yards 204 18 22 4 = 11 := by
  sorry

#eval darnells_average_yards 204 18 22 4

end NUMINAMATH_CALUDE_darnells_average_yards_is_11_l2631_263189


namespace NUMINAMATH_CALUDE_absolute_value_expression_l2631_263147

theorem absolute_value_expression : 
  |(-2)| * (|(-Real.sqrt 25)| - |Real.sin (5 * Real.pi / 2)|) = 8 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l2631_263147


namespace NUMINAMATH_CALUDE_john_horses_count_l2631_263131

/-- Represents the number of horses John has -/
def num_horses : ℕ := 25

/-- Represents the number of feedings per day for each horse -/
def feedings_per_day : ℕ := 2

/-- Represents the amount of food in pounds per feeding -/
def food_per_feeding : ℕ := 20

/-- Represents the weight of a bag of food in pounds -/
def bag_weight : ℕ := 1000

/-- Represents the number of days -/
def num_days : ℕ := 60

/-- Represents the number of bags needed for the given number of days -/
def num_bags : ℕ := 60

theorem john_horses_count :
  num_horses * feedings_per_day * food_per_feeding * num_days = num_bags * bag_weight := by
  sorry


end NUMINAMATH_CALUDE_john_horses_count_l2631_263131


namespace NUMINAMATH_CALUDE_no_real_roots_l2631_263167

theorem no_real_roots :
  ¬∃ x : ℝ, x^2 = 2*x - 3 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_l2631_263167


namespace NUMINAMATH_CALUDE_money_distribution_l2631_263178

/-- Given three people A, B, and C with money, prove that A and C together have 200 units of money -/
theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 →  -- Total money between A, B, and C
  b + c = 360 →      -- Money B and C have together
  c = 60 →           -- Money C has
  a + c = 200 :=     -- Prove A and C have 200 together
by
  sorry


end NUMINAMATH_CALUDE_money_distribution_l2631_263178


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l2631_263161

/-- Proves the volume of fuel A in a partially filled tank -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 214 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (volume_a : ℝ), 
    volume_a + (tank_capacity - volume_a) = tank_capacity ∧
    ethanol_a * volume_a + ethanol_b * (tank_capacity - volume_a) = total_ethanol ∧
    volume_a = 106 := by
  sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l2631_263161


namespace NUMINAMATH_CALUDE_no_bounded_function_satisfying_inequality_l2631_263129

theorem no_bounded_function_satisfying_inequality :
  ¬ ∃ f : ℝ → ℝ, (∀ x : ℝ, ∃ M : ℝ, |f x| ≤ M) ∧ 
    (f 1 > 0) ∧ 
    (∀ x y : ℝ, (f (x + y))^2 ≥ (f x)^2 + 2 * f (x * y) + (f y)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_no_bounded_function_satisfying_inequality_l2631_263129


namespace NUMINAMATH_CALUDE_light_intensity_reduction_l2631_263104

/-- Given light with original intensity a passing through n pieces of glass,
    each reducing intensity by 10%, calculate the final intensity -/
def final_intensity (a : ℝ) (n : ℕ) : ℝ :=
  a * (0.9 ^ n)

/-- Theorem: Light with original intensity a passing through 3 pieces of glass,
    each reducing intensity by 10%, results in a final intensity of 0.729a -/
theorem light_intensity_reduction (a : ℝ) :
  final_intensity a 3 = 0.729 * a := by
  sorry

end NUMINAMATH_CALUDE_light_intensity_reduction_l2631_263104


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_200_l2631_263128

theorem greatest_number_with_odd_factors_under_200 : 
  ∃ n : ℕ, n = 196 ∧ 
  n < 200 ∧ 
  (∃ k : ℕ, n = k^2) ∧ 
  (∀ m : ℕ, m < 200 → (∃ j : ℕ, m = j^2) → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_200_l2631_263128


namespace NUMINAMATH_CALUDE_perimeter_gt_three_times_diameter_l2631_263135

/-- A convex polyhedron. -/
class ConvexPolyhedron (M : Type*) where
  -- Add necessary axioms for convex polyhedron

/-- The perimeter of a convex polyhedron. -/
def perimeter (M : Type*) [ConvexPolyhedron M] : ℝ := sorry

/-- The diameter of a convex polyhedron. -/
def diameter (M : Type*) [ConvexPolyhedron M] : ℝ := sorry

/-- Theorem: The perimeter of a convex polyhedron is greater than three times its diameter. -/
theorem perimeter_gt_three_times_diameter (M : Type*) [ConvexPolyhedron M] :
  perimeter M > 3 * diameter M := by sorry

end NUMINAMATH_CALUDE_perimeter_gt_three_times_diameter_l2631_263135


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2631_263119

theorem hyperbola_focal_length : 
  let a : ℝ := Real.sqrt 10
  let b : ℝ := Real.sqrt 2
  let c : ℝ := Real.sqrt (a^2 + b^2)
  2 * c = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2631_263119


namespace NUMINAMATH_CALUDE_complete_square_transformation_l2631_263125

theorem complete_square_transformation (x : ℝ) : 
  x^2 - 2*x = 9 ↔ (x - 1)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_transformation_l2631_263125


namespace NUMINAMATH_CALUDE_product_evaluation_l2631_263184

theorem product_evaluation :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) * (3^8 + 1^8) = 21523360 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2631_263184


namespace NUMINAMATH_CALUDE_min_pencils_per_box_l2631_263199

/-- Represents a configuration of pencils in boxes -/
structure PencilConfiguration where
  num_boxes : Nat
  num_colors : Nat
  pencils_per_box : Nat

/-- Checks if a configuration satisfies the color requirement -/
def satisfies_color_requirement (config : PencilConfiguration) : Prop :=
  ∀ (subset : Finset (Fin config.num_boxes)), 
    subset.card = 4 → (subset.card * config.pencils_per_box ≥ config.num_colors)

/-- The main theorem stating the minimum number of pencils required -/
theorem min_pencils_per_box : 
  ∀ (config : PencilConfiguration),
    config.num_boxes = 6 ∧ 
    config.num_colors = 26 ∧ 
    satisfies_color_requirement config →
    config.pencils_per_box ≥ 13 := by
  sorry

end NUMINAMATH_CALUDE_min_pencils_per_box_l2631_263199


namespace NUMINAMATH_CALUDE_ellipse_transformation_l2631_263136

/-- Given an ellipse with equation x²/6 + y² = 1, prove that compressing
    the x-coordinates to 1/2 of their original value and stretching the
    y-coordinates to twice their original value results in a curve with
    equation 2x²/3 + y²/4 = 1. -/
theorem ellipse_transformation (x y : ℝ) :
  (x^2 / 6 + y^2 = 1) →
  (∃ x' y' : ℝ, x' = x / 2 ∧ y' = 2 * y ∧ 2 * x'^2 / 3 + y'^2 / 4 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_transformation_l2631_263136


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2631_263140

/-- Given a curve C in polar coordinates with equation ρ = 6 * cos(θ),
    prove that its equivalent Cartesian equation is x² + y² = 6x -/
theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 6 * Real.cos θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  x^2 + y^2 = 6*x := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l2631_263140


namespace NUMINAMATH_CALUDE_survey_is_simple_random_sampling_l2631_263179

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | ComplexRandom

/-- Represents a population of students --/
structure Population where
  size : Nat
  year : Nat

/-- Represents a sample from a population --/
structure Sample where
  size : Nat
  method : SamplingMethod

/-- Defines the conditions of the survey --/
def survey_conditions (pop : Population) (samp : Sample) : Prop :=
  pop.size = 200 ∧ pop.year = 1 ∧ samp.size = 20

/-- Theorem stating that the sampling method used is Simple Random Sampling --/
theorem survey_is_simple_random_sampling 
  (pop : Population) (samp : Sample) 
  (h : survey_conditions pop samp) : 
  samp.method = SamplingMethod.SimpleRandom := by
  sorry


end NUMINAMATH_CALUDE_survey_is_simple_random_sampling_l2631_263179


namespace NUMINAMATH_CALUDE_cosine_amplitude_l2631_263152

/-- Given a cosine function y = a * cos(b * x + c) + d with positive constants a, b, c, and d,
    if the maximum value of y is 5 and the minimum value is -3, then a = 4. -/
theorem cosine_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x, a * Real.cos (b * x + c) + d ≤ 5) ∧
  (∀ x, a * Real.cos (b * x + c) + d ≥ -3) ∧
  (∃ x, a * Real.cos (b * x + c) + d = 5) ∧
  (∃ x, a * Real.cos (b * x + c) + d = -3) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l2631_263152


namespace NUMINAMATH_CALUDE_sector_arc_length_l2631_263144

/-- Given a circular sector with perimeter 12 and central angle 4 radians,
    the length of its arc is 8. -/
theorem sector_arc_length (p : ℝ) (θ : ℝ) (l : ℝ) (r : ℝ) :
  p = 12 →  -- perimeter of the sector
  θ = 4 →   -- central angle in radians
  p = l + 2 * r →  -- perimeter formula for a sector
  l = θ * r →  -- arc length formula
  l = 8 :=
by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2631_263144


namespace NUMINAMATH_CALUDE_triangle_problem_l2631_263112

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (2 * cos C * (a * cos B + b * cos A) = c) →
  (c = Real.sqrt 7) →
  (1/2 * a * b * sin C = 3 * Real.sqrt 3 / 2) →
  (C = π/3 ∧ a + b + c = 5 + Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2631_263112


namespace NUMINAMATH_CALUDE_shell_collection_ratio_l2631_263173

theorem shell_collection_ratio :
  ∀ (ben_shells laurie_shells alan_shells : ℕ),
    alan_shells = 4 * ben_shells →
    laurie_shells = 36 →
    alan_shells = 48 →
    ben_shells.gcd laurie_shells = ben_shells →
    (ben_shells : ℚ) / laurie_shells = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shell_collection_ratio_l2631_263173


namespace NUMINAMATH_CALUDE_jerome_solution_l2631_263186

def jerome_problem (initial_money : ℕ) (remaining_money : ℕ) (meg_amount : ℕ) : Prop :=
  initial_money = 2 * 43 ∧
  remaining_money = 54 ∧
  initial_money = remaining_money + meg_amount + 3 * meg_amount ∧
  meg_amount = 8

theorem jerome_solution : ∃ initial_money remaining_money meg_amount, jerome_problem initial_money remaining_money meg_amount :=
  sorry

end NUMINAMATH_CALUDE_jerome_solution_l2631_263186


namespace NUMINAMATH_CALUDE_square_pyramid_components_l2631_263102

/-- The number of rows in the square pyramid -/
def num_rows : ℕ := 10

/-- The number of unit rods in the first row -/
def first_row_rods : ℕ := 4

/-- The number of additional rods in each subsequent row -/
def additional_rods_per_row : ℕ := 4

/-- Calculate the total number of unit rods in the pyramid -/
def total_rods (n : ℕ) : ℕ :=
  first_row_rods * n * (n + 1) / 2

/-- Calculate the number of internal connectors -/
def internal_connectors (n : ℕ) : ℕ :=
  4 * (n * (n - 1) / 2)

/-- Calculate the number of vertical connectors -/
def vertical_connectors (n : ℕ) : ℕ :=
  4 * (n - 1)

/-- The total number of connectors -/
def total_connectors (n : ℕ) : ℕ :=
  internal_connectors n + vertical_connectors n

/-- The main theorem: proving the total number of unit rods and connectors -/
theorem square_pyramid_components :
  total_rods num_rows + total_connectors num_rows = 436 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_components_l2631_263102


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_and_exclusion_l2631_263141

theorem systematic_sampling_interval_and_exclusion 
  (total_stores : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_stores = 92) 
  (h2 : sample_size = 30) :
  ∃ (interval : ℕ) (excluded : ℕ),
    interval * sample_size + excluded = total_stores ∧ 
    interval = 3 ∧ 
    excluded = 2 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_and_exclusion_l2631_263141


namespace NUMINAMATH_CALUDE_f_1_eq_0_f_decreasing_f_abs_lt_neg_2_iff_l2631_263172

noncomputable section

variable (f : ℝ → ℝ)

axiom f_domain : ∀ x, x > 0 → f x ≠ 0

axiom f_functional_equation : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y

axiom f_negative_when_gt_one : ∀ x, x > 1 → f x < 0

axiom f_3_eq_neg_1 : f 3 = -1

theorem f_1_eq_0 : f 1 = 0 := by sorry

theorem f_decreasing : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ > x₂ → f x₁ < f x₂ := by sorry

theorem f_abs_lt_neg_2_iff : ∀ x, f (|x|) < -2 ↔ x < -9 ∨ x > 9 := by sorry

end

end NUMINAMATH_CALUDE_f_1_eq_0_f_decreasing_f_abs_lt_neg_2_iff_l2631_263172


namespace NUMINAMATH_CALUDE_equidistant_point_on_z_axis_l2631_263174

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the squared distance between two points -/
def squaredDistance (p1 p2 : Point3D) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

/-- The theorem stating that C(0, 0, 1) is equidistant from A(1, 0, 2) and B(1, 1, 1) -/
theorem equidistant_point_on_z_axis : 
  let A : Point3D := ⟨1, 0, 2⟩
  let B : Point3D := ⟨1, 1, 1⟩
  let C : Point3D := ⟨0, 0, 1⟩
  squaredDistance A C = squaredDistance B C := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_z_axis_l2631_263174


namespace NUMINAMATH_CALUDE_chicken_multiple_l2631_263101

theorem chicken_multiple (total chickens : ℕ) (colten_chickens : ℕ) (m : ℕ) : 
  total = 383 →
  colten_chickens = 37 →
  (∃ (quentin skylar : ℕ), 
    quentin + skylar + colten_chickens = total ∧
    quentin = 2 * skylar + 25 ∧
    skylar = m * colten_chickens - 4) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_chicken_multiple_l2631_263101


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2631_263107

/-- An arithmetic sequence with first term a₁ and common ratio q -/
structure ArithmeticSequence (α : Type*) [Semiring α] where
  a₁ : α
  q : α

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm {α : Type*} [Semiring α] (seq : ArithmeticSequence α) (n : ℕ) : α :=
  seq.a₁ * seq.q ^ (n - 1)

/-- Theorem: The general term of an arithmetic sequence -/
theorem arithmetic_sequence_general_term {α : Type*} [Semiring α] (seq : ArithmeticSequence α) (n : ℕ) :
  seq.nthTerm n = seq.a₁ * seq.q ^ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2631_263107


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2631_263132

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - z) :
  Complex.im z = -1/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2631_263132


namespace NUMINAMATH_CALUDE_equation_solutions_l2631_263155

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 25 = 0 ↔ x = 5 ∨ x = -5) ∧
  (∀ x : ℝ, 8 * (x - 1)^3 = 27 ↔ x = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2631_263155


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2631_263145

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m : ℝ) : Prop :=
  2 * (m + 1) * (m - 3) + 2 * (m - 3) = 0

/-- m = 3 is a sufficient condition for the lines to be perpendicular -/
theorem sufficient_condition : perpendicular 3 := by sorry

/-- m = 3 is not a necessary condition for the lines to be perpendicular -/
theorem not_necessary_condition : ∃ m ≠ 3, perpendicular m := by sorry

/-- m = 3 is a sufficient but not necessary condition for the lines to be perpendicular -/
theorem sufficient_but_not_necessary :
  (perpendicular 3) ∧ (∃ m ≠ 3, perpendicular m) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2631_263145


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l2631_263111

theorem value_of_a_minus_b (a b c : ℚ) 
  (eq1 : 2011 * a + 2015 * b + c = 2021)
  (eq2 : 2013 * a + 2017 * b + c = 2023)
  (eq3 : 2012 * a + 2016 * b + 2 * c = 2026) :
  a - b = -2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l2631_263111


namespace NUMINAMATH_CALUDE_inverse_g_sum_l2631_263183

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else x^2 - 4*x + 5

theorem inverse_g_sum : ∃ y₁ y₂ y₃ : ℝ,
  g y₁ = -1 ∧ g y₂ = 1 ∧ g y₃ = 4 ∧ y₁ + y₂ + y₃ = 4 :=
sorry

end NUMINAMATH_CALUDE_inverse_g_sum_l2631_263183


namespace NUMINAMATH_CALUDE_remaining_movie_time_l2631_263109

def movie_length : ℕ := 120
def session1 : ℕ := 35
def session2 : ℕ := 20
def session3 : ℕ := 15

theorem remaining_movie_time :
  movie_length - (session1 + session2 + session3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_remaining_movie_time_l2631_263109


namespace NUMINAMATH_CALUDE_remainder_theorem_l2631_263105

theorem remainder_theorem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k < 39) 
  (h3 : k % 5 = 2) 
  (h4 : k % 6 = 5) : 
  k % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2631_263105


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2631_263187

/-- Given a circle with radius 5 and an isosceles trapezoid circumscribed around it,
    where the distance between the points of tangency of its lateral sides is 8,
    prove that the area of the trapezoid is 125. -/
theorem isosceles_trapezoid_area (r : ℝ) (d : ℝ) (A : ℝ) :
  r = 5 →
  d = 8 →
  A = (5 * d) * 2.5 →
  A = 125 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2631_263187


namespace NUMINAMATH_CALUDE_complex_moduli_sum_l2631_263117

theorem complex_moduli_sum : 
  let z1 : ℂ := 3 - 5*I
  let z2 : ℂ := 3 + 5*I
  Complex.abs z1 + Complex.abs z2 = 2 * Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_complex_moduli_sum_l2631_263117


namespace NUMINAMATH_CALUDE_marble_distribution_l2631_263120

theorem marble_distribution (capacity_second : ℝ) : 
  capacity_second > 0 →
  capacity_second + (3/4 * capacity_second) = 1050 →
  capacity_second = 600 := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l2631_263120


namespace NUMINAMATH_CALUDE_third_month_sale_l2631_263168

/-- Calculates the missing sale amount given the other sales and the required average -/
def missing_sale (sale1 sale2 sale4 sale5 sale6 required_average : ℕ) : ℕ :=
  6 * required_average - (sale1 + sale2 + sale4 + sale5 + sale6)

/-- Proves that the missing sale in the third month is 10555 -/
theorem third_month_sale : missing_sale 2500 6500 7230 7000 11915 7500 = 10555 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l2631_263168


namespace NUMINAMATH_CALUDE_train_length_l2631_263165

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed : Real) (time : Real) :
  speed = 108 →
  time = 1.4998800095992322 →
  ∃ (length : Real), abs (length - (speed * 1000 / 3600 * time)) < 0.001 ∧ abs (length - 44.996) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2631_263165


namespace NUMINAMATH_CALUDE_gcf_68_92_l2631_263150

theorem gcf_68_92 : Nat.gcd 68 92 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcf_68_92_l2631_263150


namespace NUMINAMATH_CALUDE_square_sum_given_linear_and_product_l2631_263197

theorem square_sum_given_linear_and_product (x y : ℝ) 
  (h1 : x + 2*y = 6) (h2 : x*y = -12) : x^2 + 4*y^2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_linear_and_product_l2631_263197


namespace NUMINAMATH_CALUDE_triangle_proof_l2631_263108

theorem triangle_proof 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : A < π / 2) 
  (h2 : Real.sin (A - π / 4) = Real.sqrt 2 / 10) 
  (h3 : (1 / 2) * b * c * Real.sin A = 24) 
  (h4 : b = 10) : 
  Real.sin A = 4 / 5 ∧ a = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_proof_l2631_263108


namespace NUMINAMATH_CALUDE_base_number_proof_l2631_263191

theorem base_number_proof (x : ℝ) (k : ℕ+) 
  (h1 : x^(k : ℝ) = 4) 
  (h2 : x^(2*(k : ℝ) + 2) = 64) : 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_base_number_proof_l2631_263191


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l2631_263158

/-- Represents a 2D point -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in parametric form -/
structure ParametricLine where
  p : Point  -- Point on the line
  v : Point  -- Direction vector

/-- The first line -/
def line1 : ParametricLine :=
  { p := { x := 2, y := 2 },
    v := { x := 3, y := -4 } }

/-- The second line -/
def line2 : ParametricLine :=
  { p := { x := 7, y := -6 },
    v := { x := 5, y := 3 } }

/-- The claimed intersection point -/
def intersectionPoint : Point :=
  { x := 11, y := -886/87 }

/-- Theorem stating that the given point is the unique intersection of the two lines -/
theorem intersection_point_is_unique :
  ∃! t u : ℚ,
    line1.p.x + t * line1.v.x = intersectionPoint.x ∧
    line1.p.y + t * line1.v.y = intersectionPoint.y ∧
    line2.p.x + u * line2.v.x = intersectionPoint.x ∧
    line2.p.y + u * line2.v.y = intersectionPoint.y :=
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l2631_263158


namespace NUMINAMATH_CALUDE_white_balls_count_l2631_263175

/-- Given a bag with 10 balls where the probability of drawing a white ball is 30%,
    prove that the number of white balls in the bag is 3. -/
theorem white_balls_count (total_balls : ℕ) (prob_white : ℚ) (white_balls : ℕ) :
  total_balls = 10 →
  prob_white = 3/10 →
  white_balls = (total_balls : ℚ) * prob_white →
  white_balls = 3 :=
by sorry

end NUMINAMATH_CALUDE_white_balls_count_l2631_263175


namespace NUMINAMATH_CALUDE_max_diff_correct_l2631_263180

/-- A convex N-gon divided into triangles by non-intersecting diagonals -/
structure ConvexNGon (N : ℕ) where
  triangles : ℕ
  diagonals : ℕ
  triangles_eq : triangles = N - 2
  diagonals_eq : diagonals = N - 3

/-- Coloring of triangles in black and white -/
structure Coloring (N : ℕ) where
  ngon : ConvexNGon N
  white : ℕ
  black : ℕ
  sum_eq : white + black = ngon.triangles
  adjacent_diff : white ≠ black → white > black

/-- Maximum difference between white and black triangles -/
def max_diff (N : ℕ) : ℕ :=
  if N % 3 = 1 then N / 3 - 1 else N / 3

theorem max_diff_correct (N : ℕ) (c : Coloring N) :
  c.white - c.black ≤ max_diff N :=
sorry

end NUMINAMATH_CALUDE_max_diff_correct_l2631_263180


namespace NUMINAMATH_CALUDE_females_watch_count_l2631_263195

/-- The number of people who watch WXLT -/
def total_watch : ℕ := 160

/-- The number of males who watch WXLT -/
def males_watch : ℕ := 85

/-- The number of females who don't watch WXLT -/
def females_dont_watch : ℕ := 120

/-- The total number of people who don't watch WXLT -/
def total_dont_watch : ℕ := 180

/-- The number of females who watch WXLT -/
def females_watch : ℕ := total_watch - males_watch

theorem females_watch_count : females_watch = 75 := by
  sorry

end NUMINAMATH_CALUDE_females_watch_count_l2631_263195


namespace NUMINAMATH_CALUDE_fourth_pentagon_dots_l2631_263159

/-- Represents the number of dots in a pentagon at a given position in the sequence -/
def dots_in_pentagon (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else dots_in_pentagon (n - 1) + 5 * (n - 1)

/-- The main theorem stating that the fourth pentagon contains 31 dots -/
theorem fourth_pentagon_dots :
  dots_in_pentagon 4 = 31 := by
  sorry

#eval dots_in_pentagon 4

end NUMINAMATH_CALUDE_fourth_pentagon_dots_l2631_263159


namespace NUMINAMATH_CALUDE_roller_coaster_problem_l2631_263166

def roller_coaster_rides (people_in_line : ℕ) (cars : ℕ) (people_per_car : ℕ) : ℕ :=
  (people_in_line + cars * people_per_car - 1) / (cars * people_per_car)

theorem roller_coaster_problem :
  roller_coaster_rides 84 7 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roller_coaster_problem_l2631_263166


namespace NUMINAMATH_CALUDE_ryanne_hezekiah_age_difference_l2631_263181

/-- Given that Ryanne and Hezekiah's combined age is 15 and Hezekiah is 4 years old,
    prove that Ryanne is 7 years older than Hezekiah. -/
theorem ryanne_hezekiah_age_difference :
  ∀ (ryanne_age hezekiah_age : ℕ),
    ryanne_age + hezekiah_age = 15 →
    hezekiah_age = 4 →
    ryanne_age - hezekiah_age = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ryanne_hezekiah_age_difference_l2631_263181


namespace NUMINAMATH_CALUDE_cylinder_cone_sphere_volume_l2631_263160

/-- Given a cylinder with volume 54π cm³ and height three times its radius,
    prove that the total volume of a cone and a sphere both having the same radius
    as the cylinder is 42π cm³ -/
theorem cylinder_cone_sphere_volume (r : ℝ) (h : ℝ) : 
  (π * r^2 * h = 54 * π) →
  (h = 3 * r) →
  (π * r^2 * r / 3 + 4 * π * r^3 / 3 = 42 * π) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cone_sphere_volume_l2631_263160


namespace NUMINAMATH_CALUDE_f_increasing_and_odd_l2631_263123

def f (x : ℝ) : ℝ := x + x^3

theorem f_increasing_and_odd :
  (∀ x y, x < y → f x < f y) ∧ 
  (∀ x, f (-x) = -f x) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_and_odd_l2631_263123


namespace NUMINAMATH_CALUDE_divisibility_condition_l2631_263121

/-- A pair of positive integers (m, n) satisfies the divisibility condition if and only if
    it is of the form (k^2 + 1, k) or (k, k^2 + 1) for some positive integer k. -/
theorem divisibility_condition (m n : ℕ+) : 
  (∃ d : ℕ+, d * (m * n - 1) = (n^2 - n + 1)^2) ↔ 
  (∃ k : ℕ+, (m = k^2 + 1 ∧ n = k) ∨ (m = k ∧ n = k^2 + 1)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2631_263121


namespace NUMINAMATH_CALUDE_bacteriophage_and_transformation_principle_correct_biological_experiment_description_l2631_263170

/-- Represents a biological experiment --/
structure BiologicalExperiment where
  name : String
  description : String

/-- Represents the principle behind an experiment --/
inductive ExperimentPrinciple
  | GeneticContinuity
  | Other

/-- Function to determine the principle of an experiment --/
def experimentPrinciple (exp : BiologicalExperiment) : ExperimentPrinciple :=
  if exp.name = "Bacteriophage Infection" || exp.name = "Bacterial Transformation" then
    ExperimentPrinciple.GeneticContinuity
  else
    ExperimentPrinciple.Other

/-- Theorem stating that bacteriophage infection and bacterial transformation 
    experiments are based on the same principle of genetic continuity --/
theorem bacteriophage_and_transformation_principle :
  ∀ (exp1 exp2 : BiologicalExperiment),
    exp1.name = "Bacteriophage Infection" →
    exp2.name = "Bacterial Transformation" →
    experimentPrinciple exp1 = experimentPrinciple exp2 :=
by
  sorry

/-- Main theorem proving the correctness of the statement --/
theorem correct_biological_experiment_description :
  ∃ (exp1 exp2 : BiologicalExperiment),
    exp1.name = "Bacteriophage Infection" ∧
    exp2.name = "Bacterial Transformation" ∧
    experimentPrinciple exp1 = ExperimentPrinciple.GeneticContinuity ∧
    experimentPrinciple exp2 = ExperimentPrinciple.GeneticContinuity :=
by
  sorry

end NUMINAMATH_CALUDE_bacteriophage_and_transformation_principle_correct_biological_experiment_description_l2631_263170


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2631_263106

theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) : 
  diagonal = 12 * Real.sqrt 2 → area = 144 → 
  diagonal^2 / 2 = area := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2631_263106


namespace NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l2631_263198

/-- Given a mixture of water and alcohol, calculate the new alcohol percentage after adding water. -/
theorem alcohol_percentage_after_dilution
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 15)
  (h2 : initial_percentage = 20)
  (h3 : added_water = 5)
  : (initial_volume * initial_percentage / 100) / (initial_volume + added_water) * 100 = 15 := by
  sorry

#check alcohol_percentage_after_dilution

end NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l2631_263198


namespace NUMINAMATH_CALUDE_monomial_degree_equality_l2631_263193

-- Define the degree of a monomial
def degree (x y z : ℕ) (m : ℕ) : ℕ := x + y

-- Define the theorem
theorem monomial_degree_equality (m : ℕ) :
  degree 2 4 0 0 = degree 0 1 (m + 2) m →
  3 * m - 2 = 7 := by sorry

end NUMINAMATH_CALUDE_monomial_degree_equality_l2631_263193


namespace NUMINAMATH_CALUDE_erased_number_proof_l2631_263115

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  x ≤ n ∧ x ≥ 1 →
  (n * (n + 1) / 2 - x) / (n - 1) = 614 / 17 →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l2631_263115


namespace NUMINAMATH_CALUDE_geometric_sequence_arithmetic_means_l2631_263192

theorem geometric_sequence_arithmetic_means (a b c m n : ℝ) 
  (h1 : b^2 = a*c)  -- geometric sequence condition
  (h2 : m = (a + b) / 2)  -- arithmetic mean of a and b
  (h3 : n = (b + c) / 2)  -- arithmetic mean of b and c
  : a / m + c / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_arithmetic_means_l2631_263192


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l2631_263196

theorem reciprocal_of_negative_two :
  ∃ x : ℚ, x * (-2) = 1 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l2631_263196


namespace NUMINAMATH_CALUDE_f_difference_nonnegative_l2631_263169

def f (x : ℝ) : ℝ := x^2 - 6*x + 5

theorem f_difference_nonnegative (x y : ℝ) :
  f x - f y ≥ 0 ↔ (x ≥ y ∧ x + y ≥ 6) ∨ (x ≤ y ∧ x + y ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_f_difference_nonnegative_l2631_263169


namespace NUMINAMATH_CALUDE_quadratic_intersection_theorem_l2631_263100

def line_l (x y : ℝ) : Prop := y = 4

def quadratic_function (x a : ℝ) : ℝ :=
  (x - a)^2 + (x - 2*a)^2 + (x - 3*a)^2 - 2*a^2 + a

def has_two_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    line_l x₁ (quadratic_function x₁ a) ∧
    line_l x₂ (quadratic_function x₂ a)

def axis_of_symmetry (a : ℝ) : ℝ := 2 * a

theorem quadratic_intersection_theorem (a : ℝ) :
  has_two_intersections a ∧ axis_of_symmetry a > 0 → 0 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_theorem_l2631_263100


namespace NUMINAMATH_CALUDE_conditional_statement_b_is_content_when_met_l2631_263148

/-- Represents the structure of a conditional statement -/
structure ConditionalStatement where
  condition : Prop
  contentWhenMet : Prop
  contentWhenNotMet : Prop

/-- Theorem stating that B in a conditional statement represents the content executed when the condition is met -/
theorem conditional_statement_b_is_content_when_met (stmt : ConditionalStatement) :
  stmt.contentWhenMet = stmt.contentWhenMet := by sorry

end NUMINAMATH_CALUDE_conditional_statement_b_is_content_when_met_l2631_263148


namespace NUMINAMATH_CALUDE_min_values_a_b_l2631_263153

theorem min_values_a_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = 2 * a + b + 2) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y + 2 → a * b ≤ x * y) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y + 2 → a + 2 * b ≤ x + 2 * y) ∧
  a * b = 6 + 4 * Real.sqrt 2 ∧
  a + 2 * b = 4 * Real.sqrt 2 + 5 :=
by sorry

end NUMINAMATH_CALUDE_min_values_a_b_l2631_263153


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2631_263171

-- Define the universal set U
def U : Set Int := {-1, 0, 1, 2}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {0, 2}

-- Theorem statement
theorem complement_A_intersect_B : (U \ A) ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2631_263171


namespace NUMINAMATH_CALUDE_min_xy_value_l2631_263149

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) :
  xy ≥ 64 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2/x + 8/y = 1 ∧ x*y = 64 :=
sorry

end NUMINAMATH_CALUDE_min_xy_value_l2631_263149


namespace NUMINAMATH_CALUDE_water_needed_to_fill_tanks_l2631_263176

/-- Proves that the total amount of water needed to fill three tanks with equal capacity is 1593 liters, 
    given the specified conditions. -/
theorem water_needed_to_fill_tanks (capacity : ℝ) 
  (h1 : capacity * 0.45 = 450)
  (h2 : capacity > 0) : 
  (capacity - 300) + (capacity - 450) + (capacity - (capacity * 0.657)) = 1593 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_to_fill_tanks_l2631_263176


namespace NUMINAMATH_CALUDE_segments_between_five_points_segments_between_five_points_proof_l2631_263157

/-- Given 5 points where no three are collinear, the number of segments needed to connect each pair of points is 10. -/
theorem segments_between_five_points : ℕ → Prop :=
  fun n => n = 5 → (∀ p1 p2 p3 : ℝ × ℝ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬Collinear p1 p2 p3) →
    (Nat.choose n 2 = 10)
  where
    Collinear (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) (p3 : ℝ × ℝ) : Prop :=
      ∃ t : ℝ, p3 = p1 + t • (p2 - p1)

/-- Proof of the theorem -/
theorem segments_between_five_points_proof : segments_between_five_points 5 := by
  sorry

end NUMINAMATH_CALUDE_segments_between_five_points_segments_between_five_points_proof_l2631_263157


namespace NUMINAMATH_CALUDE_xy_divides_x_squared_plus_2y_minus_1_l2631_263126

theorem xy_divides_x_squared_plus_2y_minus_1 (x y : ℕ+) :
  (x * y) ∣ (x^2 + 2*y - 1) ↔ 
  ((x = 3 ∧ y = 8) ∨ 
   (x = 5 ∧ y = 8) ∨ 
   (x = 1) ∨ 
   (∃ n : ℕ+, x = 2*n - 1 ∧ y = n)) := by
sorry

end NUMINAMATH_CALUDE_xy_divides_x_squared_plus_2y_minus_1_l2631_263126


namespace NUMINAMATH_CALUDE_difference_of_two_numbers_l2631_263190

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) :
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_difference_of_two_numbers_l2631_263190


namespace NUMINAMATH_CALUDE_calculation_proof_l2631_263137

theorem calculation_proof : (-7)^3 / 7^2 - 2^5 + 4^3 - 8 = 81 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2631_263137


namespace NUMINAMATH_CALUDE_june_1_2014_is_sunday_l2631_263185

def is_leap_year (year : ℕ) : Bool :=
  year % 4 = 0 && (year % 100 ≠ 0 || year % 400 = 0)

def days_in_month (year : ℕ) (month : ℕ) : ℕ :=
  if month = 2 then
    if is_leap_year year then 29 else 28
  else if month ∈ [4, 6, 9, 11] then 30
  else 31

def days_between (start_year start_month start_day : ℕ) (end_year end_month end_day : ℕ) : ℕ :=
  sorry

theorem june_1_2014_is_sunday :
  let start_date := (2013, 12, 31)
  let end_date := (2014, 6, 1)
  let start_day_of_week := 2  -- Tuesday
  let days_passed := days_between start_date.1 start_date.2.1 start_date.2.2 end_date.1 end_date.2.1 end_date.2.2
  (start_day_of_week + days_passed) % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_june_1_2014_is_sunday_l2631_263185


namespace NUMINAMATH_CALUDE_harmonic_mean_pairs_l2631_263182

theorem harmonic_mean_pairs : 
  let count := Finset.filter (fun p : ℕ × ℕ => 
    p.1 < p.2 ∧ 
    (2 * p.1 * p.2 : ℚ) / (p.1 + p.2) = 4^30
  ) (Finset.range (2^61 + 1) ×ˢ Finset.range (2^61 + 1))
  
  count.card = 61 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_pairs_l2631_263182


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2631_263122

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ := sorry

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List (Fin 4) := sorry

/-- The binary representation of the number 10110010 -/
def binary_number : List Bool := [true, false, true, true, false, false, true, false]

/-- The quaternary representation of the number 2302 -/
def quaternary_number : List (Fin 4) := [2, 3, 0, 2]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_number) = quaternary_number := by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l2631_263122


namespace NUMINAMATH_CALUDE_area_of_specific_triangle_l2631_263188

/-- Configuration of hexagons with a central hexagon of side length 2,
    surrounded by hexagons of side length 2 and 1 -/
structure HexagonConfiguration where
  centralHexagonSide : ℝ
  firstLevelSide : ℝ
  secondLevelSide : ℝ

/-- The triangle formed by connecting centers of three specific hexagons
    at the second surrounding level -/
def TriangleAtSecondLevel (config : HexagonConfiguration) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a triangle -/
def triangleArea (triangle : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem area_of_specific_triangle (config : HexagonConfiguration) 
  (h1 : config.centralHexagonSide = 2)
  (h2 : config.firstLevelSide = 2)
  (h3 : config.secondLevelSide = 1) :
  triangleArea (TriangleAtSecondLevel config) = 48 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_area_of_specific_triangle_l2631_263188


namespace NUMINAMATH_CALUDE_vacation_towel_problem_l2631_263138

theorem vacation_towel_problem (families : ℕ) (days : ℕ) (towels_per_person_per_day : ℕ) 
  (towels_per_load : ℕ) (total_loads : ℕ) :
  families = 3 →
  days = 7 →
  towels_per_person_per_day = 1 →
  towels_per_load = 14 →
  total_loads = 6 →
  (total_loads * towels_per_load) / (days * families) = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_vacation_towel_problem_l2631_263138


namespace NUMINAMATH_CALUDE_complement_union_problem_l2631_263113

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 2}
def B : Set Int := {-2, 2}

theorem complement_union_problem : (U \ A) ∪ B = {-2, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_problem_l2631_263113


namespace NUMINAMATH_CALUDE_least_value_f_1998_l2631_263156

/-- A function from positive integers to positive integers satisfying the given property -/
def FunctionF :=
  {f : ℕ+ → ℕ+ | ∀ m n : ℕ+, f (n^2 * f m) = m * (f n)^2}

/-- The theorem stating the least possible value of f(1998) -/
theorem least_value_f_1998 :
  (∃ f ∈ FunctionF, f 1998 = 120) ∧
  (∀ f ∈ FunctionF, f 1998 ≥ 120) :=
sorry

end NUMINAMATH_CALUDE_least_value_f_1998_l2631_263156


namespace NUMINAMATH_CALUDE_simplify_expression_l2631_263142

theorem simplify_expression : 
  (((Real.sqrt 2 - 1) ^ (-(Real.sqrt 3) + Real.sqrt 5)) / 
   ((Real.sqrt 2 + 1) ^ (Real.sqrt 5 - Real.sqrt 3))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2631_263142


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_l2631_263127

theorem smallest_multiple_of_seven (x y : ℤ) 
  (hx : (x - 2) % 7 = 0) 
  (hy : (y + 2) % 7 = 0) : 
  (∃ n : ℕ+, (x^2 + x*y + y^2 + n) % 7 = 0 ∧ 
    ∀ m : ℕ+, m < n → (x^2 + x*y + y^2 + m) % 7 ≠ 0) → 
  (∃ n : ℕ+, n = 3 ∧ (x^2 + x*y + y^2 + n) % 7 = 0 ∧ 
    ∀ m : ℕ+, m < n → (x^2 + x*y + y^2 + m) % 7 ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_l2631_263127


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l2631_263163

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_nonzero : 1 ≤ tens ∧ tens ≤ 9
  units_bound : units ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- The reverse of a two-digit number -/
def TwoDigitNumber.reverse (n : TwoDigitNumber) : TwoDigitNumber where
  tens := n.units
  units := n.tens
  tens_nonzero := by sorry
  units_bound := n.tens_nonzero.2

theorem two_digit_number_sum (n : TwoDigitNumber) :
  (n.value - n.reverse.value = 7 * (n.tens + n.units)) →
  (n.value + n.reverse.value = 99) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l2631_263163


namespace NUMINAMATH_CALUDE_ab_greater_than_a_plus_b_l2631_263177

theorem ab_greater_than_a_plus_b (a b : ℝ) (ha : a > 2) (hb : b > 2) :
  a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_a_plus_b_l2631_263177


namespace NUMINAMATH_CALUDE_piglet_straws_l2631_263143

theorem piglet_straws (total_straws : ℕ) (adult_pig_fraction : ℚ) (piglet_fraction : ℚ) (num_piglets : ℕ) :
  total_straws = 300 →
  adult_pig_fraction = 7 / 15 →
  piglet_fraction = 2 / 5 →
  num_piglets = 20 →
  (piglet_fraction * total_straws) / num_piglets = 6 := by
  sorry

end NUMINAMATH_CALUDE_piglet_straws_l2631_263143


namespace NUMINAMATH_CALUDE_even_polynomial_iff_composition_l2631_263162

open Polynomial

theorem even_polynomial_iff_composition (P : Polynomial ℝ) :
  (∀ x, P.eval (-x) = P.eval x) ↔ 
  ∃ Q : Polynomial ℝ, P = Q.comp (X ^ 2) :=
sorry

end NUMINAMATH_CALUDE_even_polynomial_iff_composition_l2631_263162
