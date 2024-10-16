import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_inequality_l592_59256

theorem solution_set_inequality (x : ℝ) : 
  (2 * x) / (x + 1) ≤ 1 ↔ x ∈ Set.Ioc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l592_59256


namespace NUMINAMATH_CALUDE_import_tax_calculation_l592_59236

/-- Given an item with a total value V, subject to a 7% import tax on the portion
    exceeding $1,000, prove that if the tax paid is $109.90, then V = $2,567. -/
theorem import_tax_calculation (V : ℝ) : 
  (0.07 * (V - 1000) = 109.90) → V = 2567 := by
  sorry

end NUMINAMATH_CALUDE_import_tax_calculation_l592_59236


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l592_59238

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- {a_n} is a geometric sequence with common ratio q
  (a 6 + a 8 - a 5 = a 7 - (a 6 + a 8)) →  -- a_5, a_6 + a_8, a_7 form an arithmetic sequence
  q = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l592_59238


namespace NUMINAMATH_CALUDE_boatman_distance_along_current_l592_59263

/-- Represents the speed of a boat in different conditions -/
structure BoatSpeed where
  stationary : ℝ
  against_current : ℝ
  current : ℝ
  along_current : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (speed time : ℝ) : ℝ := speed * time

/-- Theorem: The boatman travels 1 km along the current -/
theorem boatman_distance_along_current 
  (speed : BoatSpeed)
  (h1 : distance speed.against_current 4 = 4) -- 4 km against current in 4 hours
  (h2 : distance speed.stationary 3 = 6)      -- 6 km in stationary water in 3 hours
  (h3 : speed.current = speed.stationary - speed.against_current)
  (h4 : speed.along_current = speed.stationary + speed.current)
  : distance speed.along_current (1/3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_boatman_distance_along_current_l592_59263


namespace NUMINAMATH_CALUDE_nested_radical_equation_l592_59284

theorem nested_radical_equation (x : ℝ) : 
  x = 34 → 
  Real.sqrt (9 + Real.sqrt (18 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_nested_radical_equation_l592_59284


namespace NUMINAMATH_CALUDE_drug_storage_temperature_range_l592_59253

def central_temp : ℝ := 20
def variation : ℝ := 2

def lower_limit : ℝ := central_temp - variation
def upper_limit : ℝ := central_temp + variation

theorem drug_storage_temperature_range : 
  (lower_limit = 18 ∧ upper_limit = 22) := by sorry

end NUMINAMATH_CALUDE_drug_storage_temperature_range_l592_59253


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l592_59245

theorem power_of_power_at_three :
  (3^3)^(3^3) = 27^27 := by sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l592_59245


namespace NUMINAMATH_CALUDE_company_y_installation_charge_l592_59212

-- Define the given constants
def company_x_price : ℝ := 575
def company_x_surcharge_rate : ℝ := 0.04
def company_x_installation : ℝ := 82.50
def company_y_price : ℝ := 530
def company_y_surcharge_rate : ℝ := 0.03
def total_charge_difference : ℝ := 41.60

-- Define the function to calculate total cost
def total_cost (price surcharge_rate installation : ℝ) : ℝ :=
  price + price * surcharge_rate + installation

-- State the theorem
theorem company_y_installation_charge :
  ∃ (company_y_installation : ℝ),
    company_y_installation = 93 ∧
    total_cost company_x_price company_x_surcharge_rate company_x_installation -
    total_cost company_y_price company_y_surcharge_rate company_y_installation =
    total_charge_difference :=
by
  sorry

end NUMINAMATH_CALUDE_company_y_installation_charge_l592_59212


namespace NUMINAMATH_CALUDE_factor_expression_l592_59265

theorem factor_expression (c : ℝ) : 210 * c^3 + 35 * c^2 = 35 * c^2 * (6 * c + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l592_59265


namespace NUMINAMATH_CALUDE_min_value_theorem_l592_59241

theorem min_value_theorem (m n : ℝ) (h1 : m * n > 0) (h2 : 2 * m + n = 1) :
  (1 / m + 2 / n) ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l592_59241


namespace NUMINAMATH_CALUDE_unfolded_paper_has_symmetric_holes_l592_59274

/-- Represents a rectangular piece of paper -/
structure Paper :=
  (width : ℝ)
  (height : ℝ)
  (is_rectangular : width > 0 ∧ height > 0)

/-- Represents a hole on the paper -/
structure Hole :=
  (x : ℝ)
  (y : ℝ)

/-- Represents the state of the paper after folding and punching -/
structure FoldedPaper :=
  (original : Paper)
  (hole : Hole)
  (is_folded_left_right : Bool)
  (is_folded_diagonally : Bool)
  (is_hole_near_center : Bool)

/-- Represents the state of the paper after unfolding -/
structure UnfoldedPaper :=
  (original : Paper)
  (holes : List Hole)

/-- Function to unfold the paper -/
def unfold (fp : FoldedPaper) : UnfoldedPaper :=
  sorry

/-- Predicate to check if holes are symmetrically placed -/
def are_holes_symmetric (up : UnfoldedPaper) : Prop :=
  sorry

/-- Main theorem: Unfolding a properly folded and punched paper results in four symmetrically placed holes -/
theorem unfolded_paper_has_symmetric_holes (fp : FoldedPaper) 
  (h1 : fp.is_folded_left_right = true)
  (h2 : fp.is_folded_diagonally = true)
  (h3 : fp.is_hole_near_center = true) :
  let up := unfold fp
  (up.holes.length = 4) ∧ (are_holes_symmetric up) :=
  sorry

end NUMINAMATH_CALUDE_unfolded_paper_has_symmetric_holes_l592_59274


namespace NUMINAMATH_CALUDE_geometric_mean_problem_l592_59294

theorem geometric_mean_problem : 
  let a := 7 + 3 * Real.sqrt 5
  let b := 7 - 3 * Real.sqrt 5
  ∃ x : ℝ, x^2 = a * b ∧ (x = 2 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_problem_l592_59294


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l592_59223

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem derivative_f_at_one : 
  deriv f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l592_59223


namespace NUMINAMATH_CALUDE_line_through_P_intersecting_C_l592_59288

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 8*y - 5 = 0

-- Define the point P
def point_P : ℝ × ℝ := (5, 0)

-- Define the chord length
def chord_length : ℝ := 8

-- Define the two possible line equations
def line_eq1 (x : ℝ) : Prop := x = 5
def line_eq2 (x y : ℝ) : Prop := 7*x + 24*y - 35 = 0

-- Theorem statement
theorem line_through_P_intersecting_C :
  ∃ (l : ℝ → ℝ → Prop),
    (∀ x y, l x y → (x = point_P.1 ∧ y = point_P.2)) ∧
    (∃ x1 y1 x2 y2, l x1 y1 ∧ l x2 y2 ∧ circle_C x1 y1 ∧ circle_C x2 y2 ∧
      (x2 - x1)^2 + (y2 - y1)^2 = chord_length^2) ∧
    ((∀ x y, l x y ↔ line_eq1 x) ∨ (∀ x y, l x y ↔ line_eq2 x y)) :=
sorry

end NUMINAMATH_CALUDE_line_through_P_intersecting_C_l592_59288


namespace NUMINAMATH_CALUDE_newcomer_weight_l592_59287

/-- Represents the weight of a group of people -/
structure GroupWeight where
  initial : ℝ
  new : ℝ

/-- The problem setup -/
def weightProblem (g : GroupWeight) : Prop :=
  -- Initial weight is between 400 kg and 420 kg
  400 ≤ g.initial ∧ g.initial ≤ 420 ∧
  -- The average weight increase is 3.5 kg
  g.new = g.initial - 47 + 68 ∧
  -- The average weight increases by 3.5 kg
  (g.new / 6) - (g.initial / 6) = 3.5

/-- The theorem to prove -/
theorem newcomer_weight (g : GroupWeight) : 
  weightProblem g → 68 = g.new - g.initial + 47 := by
  sorry


end NUMINAMATH_CALUDE_newcomer_weight_l592_59287


namespace NUMINAMATH_CALUDE_second_candidate_votes_l592_59276

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℚ) : 
  total_votes = 800 → 
  first_candidate_percentage = 70 / 100 →
  (total_votes : ℚ) * (1 - first_candidate_percentage) = 240 := by
  sorry

#check second_candidate_votes

end NUMINAMATH_CALUDE_second_candidate_votes_l592_59276


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l592_59231

theorem quadratic_equation_roots (a b c : ℝ) (h : a = 2 ∧ b = -3 ∧ c = 1) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l592_59231


namespace NUMINAMATH_CALUDE_three_digit_to_four_digit_l592_59270

theorem three_digit_to_four_digit (a : ℕ) (h : 100 ≤ a ∧ a ≤ 999) :
  (10 * a + 1 : ℕ) = 1000 + (a - 100) * 10 + 1 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_to_four_digit_l592_59270


namespace NUMINAMATH_CALUDE_product_properties_l592_59218

-- Define the range of two-digit numbers
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define the range of three-digit numbers
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Define the number of digits in a natural number
def NumDigits (n : ℕ) : ℕ := (Nat.log 10 n).succ

-- Define approximate equality
def ApproxEqual (x y : ℕ) (ε : ℕ) : Prop := (x : ℤ) - (y : ℤ) ≤ ε ∧ (y : ℤ) - (x : ℤ) ≤ ε

theorem product_properties :
  (NumDigits (52 * 403) = 5) ∧
  (ApproxEqual (52 * 403) 20000 1000) ∧
  (∀ a b, ThreeDigitNumber a → TwoDigitNumber b →
    (NumDigits (a * b) = 4 ∨ NumDigits (a * b) = 5)) :=
by sorry

end NUMINAMATH_CALUDE_product_properties_l592_59218


namespace NUMINAMATH_CALUDE_product_xyz_equals_two_l592_59282

theorem product_xyz_equals_two
  (x y z : ℝ)
  (h1 : x + 1 / y = 2)
  (h2 : y + 1 / z = 2)
  (h3 : x + 1 / z = 3) :
  x * y * z = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_xyz_equals_two_l592_59282


namespace NUMINAMATH_CALUDE_closest_fraction_l592_59271

def medals_won : ℚ := 25 / 160

def fractions : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (f : ℚ), f ∈ fractions ∧ 
  ∀ (g : ℚ), g ∈ fractions → |medals_won - f| ≤ |medals_won - g| ∧
  f = 1/8 :=
sorry

end NUMINAMATH_CALUDE_closest_fraction_l592_59271


namespace NUMINAMATH_CALUDE_garden_area_l592_59297

/-- Represents a rectangular garden with given properties -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  length_walk : length * 20 = 1000
  perimeter_walk : (length + width) * 2 * 8 = 1000

/-- The area of a rectangular garden with the given properties is 625 square meters -/
theorem garden_area (g : RectangularGarden) : g.length * g.width = 625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l592_59297


namespace NUMINAMATH_CALUDE_evaluate_expression_l592_59230

theorem evaluate_expression : 
  45 * ((4 + 1/3) - (5 + 1/4)) / ((3 + 1/2) + (2 + 1/5)) = -(7 + 9/38) := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l592_59230


namespace NUMINAMATH_CALUDE_functional_equation_implies_additive_l592_59200

/-- A function satisfying the given functional equation is additive. -/
theorem functional_equation_implies_additive (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y + x * y) = f x + f y + f (x * y)) :
  ∀ x y : ℝ, f (x + y) = f x + f y := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_additive_l592_59200


namespace NUMINAMATH_CALUDE_maria_savings_l592_59228

/-- The amount of money Maria will have left after buying sweaters and scarves -/
def money_left (sweater_price scarf_price num_sweaters num_scarves savings : ℕ) : ℕ :=
  savings - (sweater_price * num_sweaters + scarf_price * num_scarves)

/-- Theorem stating that Maria will have $200 left after her purchases -/
theorem maria_savings : money_left 30 20 6 6 500 = 200 := by
  sorry

end NUMINAMATH_CALUDE_maria_savings_l592_59228


namespace NUMINAMATH_CALUDE_fraction_equation_sum_l592_59204

theorem fraction_equation_sum (A B : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 5 →
    (B * x - 17) / (x^2 - 9*x + 20) = A / (x - 4) + 5 / (x - 5)) →
  A + B = 9/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_sum_l592_59204


namespace NUMINAMATH_CALUDE_power_inequality_l592_59220

theorem power_inequality (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) (hm : m > 0) (hn : n > 0) (hmn : m > n) :
  a^m + a^(-m) > a^n + a^(-n) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l592_59220


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l592_59291

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem states that for a geometric sequence satisfying given conditions, 
    the sum of the 5th and 6th terms equals 48. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a)
  (h_sum1 : a 1 + a 2 = 3)
  (h_sum2 : a 3 + a 4 = 12) :
  a 5 + a 6 = 48 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l592_59291


namespace NUMINAMATH_CALUDE_shopping_tax_calculation_l592_59232

theorem shopping_tax_calculation (total : ℝ) (clothing_percent : ℝ) (food_percent : ℝ) 
  (other_percent : ℝ) (clothing_tax : ℝ) (food_tax : ℝ) (total_tax_percent : ℝ) 
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.1)
  (h3 : other_percent = 0.4)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax = 0.04)
  (h6 : food_tax = 0)
  (h7 : total_tax_percent = 0.052)
  : ∃ other_tax : ℝ, 
    clothing_tax * clothing_percent * total + 
    food_tax * food_percent * total + 
    other_tax * other_percent * total = 
    total_tax_percent * total ∧ 
    other_tax = 0.08 := by
sorry

end NUMINAMATH_CALUDE_shopping_tax_calculation_l592_59232


namespace NUMINAMATH_CALUDE_production_average_l592_59214

theorem production_average (n : ℕ) 
  (h1 : (n * 50 + 60) / (n + 1) = 55) : n = 1 := by
  sorry

end NUMINAMATH_CALUDE_production_average_l592_59214


namespace NUMINAMATH_CALUDE_exam_question_count_l592_59252

theorem exam_question_count :
  ∀ (num_type_a num_type_b : ℕ) (time_per_a time_per_b : ℚ),
    num_type_a = 100 →
    time_per_a = 2 * time_per_b →
    num_type_a * time_per_a = 120 →
    num_type_a * time_per_a + num_type_b * time_per_b = 180 →
    num_type_a + num_type_b = 200 := by
  sorry

end NUMINAMATH_CALUDE_exam_question_count_l592_59252


namespace NUMINAMATH_CALUDE_stage_25_l592_59221

/-- Represents the number of toothpicks in a stage of the triangle pattern. -/
def toothpicks (n : ℕ) : ℕ := 3 * n

/-- The triangle pattern starts with 1 toothpick per side in Stage 1. -/
axiom stage_one : toothpicks 1 = 3

/-- Each stage adds one toothpick to each side of the triangle. -/
axiom stage_increase (n : ℕ) : toothpicks (n + 1) = toothpicks n + 3

/-- The number of toothpicks in the 25th stage is 75. -/
theorem stage_25 : toothpicks 25 = 75 := by sorry

end NUMINAMATH_CALUDE_stage_25_l592_59221


namespace NUMINAMATH_CALUDE_jimin_calculation_l592_59222

theorem jimin_calculation (x : ℤ) (h : 20 - x = 60) : 34 * x = -1360 := by
  sorry

end NUMINAMATH_CALUDE_jimin_calculation_l592_59222


namespace NUMINAMATH_CALUDE_orange_gumdrops_after_replacement_l592_59206

/-- Represents the number of gumdrops of each color in a jar --/
structure GumdropsJar where
  purple : ℕ
  orange : ℕ
  violet : ℕ
  yellow : ℕ
  white : ℕ
  green : ℕ

/-- Calculates the total number of gumdrops in the jar --/
def total_gumdrops (jar : GumdropsJar) : ℕ :=
  jar.purple + jar.orange + jar.violet + jar.yellow + jar.white + jar.green

/-- Theorem stating the number of orange gumdrops after replacement --/
theorem orange_gumdrops_after_replacement (jar : GumdropsJar) :
  jar.white = 40 ∧
  total_gumdrops jar = 160 ∧
  jar.purple = 40 ∧
  jar.orange = 24 ∧
  jar.violet = 32 ∧
  jar.yellow = 24 →
  jar.orange + (jar.purple / 3) = 37 := by
  sorry

#check orange_gumdrops_after_replacement

end NUMINAMATH_CALUDE_orange_gumdrops_after_replacement_l592_59206


namespace NUMINAMATH_CALUDE_john_climbs_70_feet_l592_59281

/-- Calculates the total height climbed by John given the number of flights, height per flight, and additional ladder length. -/
def totalHeightClimbed (numFlights : ℕ) (flightHeight : ℕ) (additionalLadderLength : ℕ) : ℕ :=
  let stairsHeight := numFlights * flightHeight
  let ropeHeight := stairsHeight / 2
  let ladderHeight := ropeHeight + additionalLadderLength
  stairsHeight + ropeHeight + ladderHeight

/-- Theorem stating that under the given conditions, John climbs a total of 70 feet. -/
theorem john_climbs_70_feet :
  totalHeightClimbed 3 10 10 = 70 := by
  sorry

end NUMINAMATH_CALUDE_john_climbs_70_feet_l592_59281


namespace NUMINAMATH_CALUDE_max_removable_edges_in_complete_graph_l592_59229

theorem max_removable_edges_in_complete_graph :
  ∀ (n : ℕ), n = 30 →
  ∃ (k : ℕ), k = 406 ∧
  (((n * (n - 1)) / 2) - k = n - 1) ∧
  k = ((n * (n - 1)) / 2) - (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_max_removable_edges_in_complete_graph_l592_59229


namespace NUMINAMATH_CALUDE_binomial_coefficient_inequality_l592_59233

theorem binomial_coefficient_inequality (n k : ℕ) (h1 : n > k) (h2 : k > 0) : 
  (1 : ℝ) / (n + 1 : ℝ) * (n^n : ℝ) / ((k^k * (n-k)^(n-k)) : ℝ) < 
  (n.factorial : ℝ) / ((k.factorial * (n-k).factorial) : ℝ) ∧
  (n.factorial : ℝ) / ((k.factorial * (n-k).factorial) : ℝ) < 
  (n^n : ℝ) / ((k^k * (n-k)^(n-k)) : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_inequality_l592_59233


namespace NUMINAMATH_CALUDE_odd_function_value_l592_59225

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x - 3*x + k else -(2^(-x) - 3*(-x) + k)
  where k : ℝ := -1 -- We define k here to make the function complete

-- State the theorem
theorem odd_function_value : f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l592_59225


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l592_59279

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • (B^14) = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l592_59279


namespace NUMINAMATH_CALUDE_food_preference_count_l592_59264

/-- The number of students who like the food -/
def students_like : ℕ := 383

/-- The total number of students who participated in the discussion -/
def total_students : ℕ := 814

/-- The number of students who didn't like the food -/
def students_dislike : ℕ := total_students - students_like

theorem food_preference_count : students_dislike = 431 := by
  sorry

end NUMINAMATH_CALUDE_food_preference_count_l592_59264


namespace NUMINAMATH_CALUDE_percent_of_a_is_4b_l592_59266

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.8 * b) : 
  (4 * b) / a * 100 = 222.22 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_is_4b_l592_59266


namespace NUMINAMATH_CALUDE_abc_sum_problem_l592_59209

theorem abc_sum_problem (a b c d : ℝ) 
  (eq1 : a + b + c = 6)
  (eq2 : a + b + d = 2)
  (eq3 : a + c + d = 3)
  (eq4 : b + c + d = -3) :
  a * b + c * d = -37 / 9 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_problem_l592_59209


namespace NUMINAMATH_CALUDE_rice_purchase_difference_l592_59226

/-- Represents the price and quantity of rice from a supplier -/
structure RiceSupply where
  quantity : ℝ
  price : ℝ

/-- Calculates the total cost of rice supplies -/
def totalCost (supplies : List RiceSupply) : ℝ :=
  supplies.foldl (fun acc supply => acc + supply.quantity * supply.price) 0

/-- Represents the rice purchase scenario -/
structure RicePurchase where
  supplies : List RiceSupply
  keptRatio : ℝ
  conversionRate : ℝ

theorem rice_purchase_difference (purchase : RicePurchase) 
  (h1 : purchase.supplies = [
    ⟨15, 1.2⟩, ⟨10, 1.4⟩, ⟨12, 1.6⟩, ⟨8, 1.9⟩, ⟨5, 2.3⟩
  ])
  (h2 : purchase.keptRatio = 7/10)
  (h3 : purchase.conversionRate = 1.15) :
  let totalCostEuros := totalCost purchase.supplies
  let keptCostDollars := totalCostEuros * purchase.keptRatio * purchase.conversionRate
  let givenCostDollars := totalCostEuros * (1 - purchase.keptRatio) * purchase.conversionRate
  keptCostDollars - givenCostDollars = 35.88 := by
  sorry

end NUMINAMATH_CALUDE_rice_purchase_difference_l592_59226


namespace NUMINAMATH_CALUDE_alices_preferred_numbers_l592_59289

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

def preferred_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 150 ∧ 
  n % 7 = 0 ∧
  ¬(n % 3 = 0) ∧
  is_prime (digit_sum n)

theorem alices_preferred_numbers :
  {n : ℕ | preferred_number n} = {119, 133, 140} := by sorry

end NUMINAMATH_CALUDE_alices_preferred_numbers_l592_59289


namespace NUMINAMATH_CALUDE_principal_is_250_l592_59208

/-- Proves that the principal is 250 given the conditions of the problem -/
theorem principal_is_250 (P : ℝ) (I : ℝ) : 
  I = P * 0.04 * 8 →  -- Simple interest formula for 4% per annum over 8 years
  I = P - 170 →       -- Interest is 170 less than the principal
  P = 250 := by
sorry

end NUMINAMATH_CALUDE_principal_is_250_l592_59208


namespace NUMINAMATH_CALUDE_books_total_is_54_l592_59290

/-- The total number of books Darla, Katie, and Gary have -/
def total_books (darla_books katie_books gary_books : ℕ) : ℕ :=
  darla_books + katie_books + gary_books

/-- Theorem stating the total number of books is 54 -/
theorem books_total_is_54 :
  ∀ (darla_books katie_books gary_books : ℕ),
    darla_books = 6 →
    katie_books = darla_books / 2 →
    gary_books = 5 * (darla_books + katie_books) →
    total_books darla_books katie_books gary_books = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_books_total_is_54_l592_59290


namespace NUMINAMATH_CALUDE_real_part_of_reciprocal_l592_59298

theorem real_part_of_reciprocal (x y : ℝ) (z : ℂ) (h1 : z = x + y * I) (h2 : z ≠ x) (h3 : Complex.abs z = 1) :
  (1 / (2 - z)).re = (2 - x) / (5 - 4 * x) := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_reciprocal_l592_59298


namespace NUMINAMATH_CALUDE_regular_polygon_135_degrees_has_8_sides_l592_59202

/-- A regular polygon with interior angles of 135 degrees has 8 sides -/
theorem regular_polygon_135_degrees_has_8_sides :
  ∀ n : ℕ, 
  n > 2 →
  (180 * (n - 2) : ℝ) = 135 * n →
  n = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_regular_polygon_135_degrees_has_8_sides_l592_59202


namespace NUMINAMATH_CALUDE_share_difference_l592_59250

def money_distribution (total : ℕ) (faruk vasim ranjith : ℕ) : Prop :=
  faruk + vasim + ranjith = total ∧ 3 * ranjith = 7 * faruk ∧ faruk = vasim

theorem share_difference (total : ℕ) (faruk vasim ranjith : ℕ) :
  money_distribution total faruk vasim ranjith → vasim = 1500 → ranjith - faruk = 2000 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_l592_59250


namespace NUMINAMATH_CALUDE_division_problem_l592_59295

theorem division_problem : (144 : ℚ) / ((12 : ℚ) / 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l592_59295


namespace NUMINAMATH_CALUDE_h_solutions_l592_59292

noncomputable def h (x : ℝ) : ℝ :=
  if x < 2 then 4 * x + 10 else 3 * x - 12

theorem h_solutions :
  ∀ x : ℝ, h x = 6 ↔ x = -1 ∨ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_h_solutions_l592_59292


namespace NUMINAMATH_CALUDE_tank_filling_time_l592_59224

theorem tank_filling_time (fill_rate : ℝ) (leak_rate : ℝ) (fill_time_no_leak : ℝ) (empty_time_leak : ℝ) :
  fill_rate = 1 / fill_time_no_leak →
  leak_rate = 1 / empty_time_leak →
  fill_time_no_leak = 8 →
  empty_time_leak = 72 →
  (1 : ℝ) / (fill_rate - leak_rate) = 9 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l592_59224


namespace NUMINAMATH_CALUDE_regression_correction_l592_59205

/-- Represents a data point with x and y coordinates -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Represents a linear regression equation -/
structure RegressionEquation where
  slope : ℝ
  intercept : ℝ

/-- Represents the center of sample points -/
structure SampleCenter where
  x : ℝ
  y : ℝ

theorem regression_correction (data : List DataPoint) 
  (initial_eq : RegressionEquation) 
  (initial_center : SampleCenter)
  (incorrect_point1 incorrect_point2 correct_point1 correct_point2 : DataPoint)
  (corrected_slope : ℝ)
  (h1 : data.length = 8)
  (h2 : initial_eq.slope = 2 ∧ initial_eq.intercept = 5)
  (h3 : initial_center.x = 2)
  (h4 : incorrect_point1 = ⟨7, 3⟩ ∧ correct_point1 = ⟨3, 7⟩)
  (h5 : incorrect_point2 = ⟨4, -6⟩ ∧ correct_point2 = ⟨4, 6⟩)
  (h6 : corrected_slope = 13/3) :
  ∃ k : ℝ, k = 9/2 ∧ 
    ∀ x y : ℝ, y = corrected_slope * x + k → 
      ∃ center : SampleCenter, center.x = 3/2 ∧ center.y = 11 ∧
        y = corrected_slope * center.x + k := by
  sorry

end NUMINAMATH_CALUDE_regression_correction_l592_59205


namespace NUMINAMATH_CALUDE_max_value_of_f_l592_59234

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 * Real.exp x

theorem max_value_of_f (h : a ≠ 0) :
  (a > 0 → ∃ M, M = 4 * a * Real.exp (-2) ∧ ∀ x, f a x ≤ M) ∧
  (a < 0 → ∃ M, M = 0 ∧ ∀ x, f a x ≤ M) :=
sorry

end

end NUMINAMATH_CALUDE_max_value_of_f_l592_59234


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l592_59283

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c for which the lines y = 6x + 3 and y = 3cx + 1 are parallel -/
theorem parallel_lines_c_value : 
  (∀ x y : ℝ, y = 6 * x + 3 ↔ y = 3 * c * x + 1) → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l592_59283


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l592_59215

theorem orange_juice_fraction (pitcher1_capacity pitcher2_capacity : ℚ)
  (pitcher1_juice_fraction pitcher2_juice_fraction : ℚ) :
  pitcher1_capacity = 500 →
  pitcher2_capacity = 800 →
  pitcher1_juice_fraction = 1/4 →
  pitcher2_juice_fraction = 1/2 →
  let total_juice := pitcher1_capacity * pitcher1_juice_fraction + pitcher2_capacity * pitcher2_juice_fraction
  let total_volume := pitcher1_capacity + pitcher2_capacity
  (total_juice / total_volume) = 21/52 := by
sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_l592_59215


namespace NUMINAMATH_CALUDE_CD_length_theorem_l592_59201

-- Define the line segment CD
def CD : Set (ℝ × ℝ × ℝ) := sorry

-- Define the region within 4 units of CD
def region (CD : Set (ℝ × ℝ × ℝ)) : Set (ℝ × ℝ × ℝ) := sorry

-- Define the volume of a set in 3D space
def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define the length of a line segment
def length (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem CD_length_theorem (CD : Set (ℝ × ℝ × ℝ)) :
  volume (region CD) = 448 * Real.pi → length CD = 68 / 3 := by
  sorry

end NUMINAMATH_CALUDE_CD_length_theorem_l592_59201


namespace NUMINAMATH_CALUDE_f_is_quadratic_l592_59239

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = 5x^2 - 1 -/
def f (x : ℝ) : ℝ := 5 * x^2 - 1

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l592_59239


namespace NUMINAMATH_CALUDE_sequence_property_l592_59244

def a (n : ℕ) (x : ℝ) : ℝ := 1 + x^(n+1) + x^(n+2)

theorem sequence_property (x : ℝ) :
  (a 2 x)^2 = (a 1 x) * (a 3 x) →
  ∀ n ≥ 3, (a n x)^2 = a n x :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l592_59244


namespace NUMINAMATH_CALUDE_area_r_is_twelve_point_five_percent_l592_59269

/-- Represents a circular spinner with specific properties -/
structure CircularSpinner where
  /-- Diameter PQ passes through the center -/
  has_diameter_through_center : Bool
  /-- Areas R and S are equal -/
  r_equals_s : Bool
  /-- R and S together form a quadrant -/
  r_plus_s_is_quadrant : Bool

/-- Calculates the percentage of the total area occupied by region R -/
def area_percentage_r (spinner : CircularSpinner) : ℝ :=
  sorry

/-- Theorem stating that the area of region R is 12.5% of the total circle area -/
theorem area_r_is_twelve_point_five_percent (spinner : CircularSpinner) 
  (h1 : spinner.has_diameter_through_center = true)
  (h2 : spinner.r_equals_s = true)
  (h3 : spinner.r_plus_s_is_quadrant = true) : 
  area_percentage_r spinner = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_area_r_is_twelve_point_five_percent_l592_59269


namespace NUMINAMATH_CALUDE_chocolate_bars_left_chocolate_problem_l592_59255

theorem chocolate_bars_left (initial_bars : ℕ) 
  (thomas_and_friends : ℕ) (piper_reduction : ℕ) 
  (friend_return : ℕ) : ℕ :=
  let thomas_take := initial_bars / 4
  let friend_take := thomas_take / thomas_and_friends
  let total_taken := thomas_take - friend_return
  let piper_take := total_taken - piper_reduction
  initial_bars - total_taken - piper_take

theorem chocolate_problem : 
  chocolate_bars_left 200 5 5 5 = 115 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_left_chocolate_problem_l592_59255


namespace NUMINAMATH_CALUDE_freddy_age_l592_59285

/-- Represents the ages of three children --/
structure ChildrenAges where
  matthew : ℕ
  rebecca : ℕ
  freddy : ℕ

/-- The conditions of the problem --/
def problem_conditions (ages : ChildrenAges) : Prop :=
  ages.matthew + ages.rebecca + ages.freddy = 35 ∧
  ages.matthew = ages.rebecca + 2 ∧
  ages.freddy = ages.matthew + 4

/-- The theorem stating that under the given conditions, Freddy is 15 years old --/
theorem freddy_age (ages : ChildrenAges) : 
  problem_conditions ages → ages.freddy = 15 := by
  sorry


end NUMINAMATH_CALUDE_freddy_age_l592_59285


namespace NUMINAMATH_CALUDE_equations_different_graphs_l592_59237

-- Define the three equations
def eq1 (x y : ℝ) : Prop := y = 2 * x - 3
def eq2 (x y : ℝ) : Prop := y = (2 * x^2 - 18) / (x + 3)
def eq3 (x y : ℝ) : Prop := (x + 3) * y = 2 * x^2 - 18

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, eq1 x y ↔ eq2 x y

-- Theorem statement
theorem equations_different_graphs :
  ¬(same_graph eq1 eq2) ∧ ¬(same_graph eq1 eq3) ∧ ¬(same_graph eq2 eq3) :=
sorry

end NUMINAMATH_CALUDE_equations_different_graphs_l592_59237


namespace NUMINAMATH_CALUDE_ferris_wheel_rides_count_l592_59248

/-- Represents the number of ferris wheel rides -/
def ferris_wheel_rides : ℕ := sorry

/-- Represents the number of bumper car rides -/
def bumper_car_rides : ℕ := 4

/-- Represents the cost of each ride in tickets -/
def cost_per_ride : ℕ := 7

/-- Represents the total number of tickets used -/
def total_tickets : ℕ := 63

/-- Theorem stating that the number of ferris wheel rides is 5 -/
theorem ferris_wheel_rides_count : ferris_wheel_rides = 5 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_rides_count_l592_59248


namespace NUMINAMATH_CALUDE_probability_of_b_in_rabbit_l592_59203

def word : String := "rabbit"

def count_letter (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem probability_of_b_in_rabbit :
  (count_letter word 'b' : ℚ) / word.length = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_b_in_rabbit_l592_59203


namespace NUMINAMATH_CALUDE_variance_implies_stability_l592_59213

-- Define a structure for a data set
structure DataSet where
  variance : ℝ
  stability : ℝ

-- Define a relation for comparing stability
def more_stable (a b : DataSet) : Prop :=
  a.stability > b.stability

-- Theorem statement
theorem variance_implies_stability (a b : DataSet) 
  (h : a.variance < b.variance) : more_stable a b :=
sorry

end NUMINAMATH_CALUDE_variance_implies_stability_l592_59213


namespace NUMINAMATH_CALUDE_unique_solution_for_special_integers_l592_59258

theorem unique_solution_for_special_integers (a b : ℕ+) : 
  a ≠ b → 
  (∃ p : ℕ, Prime p ∧ ∃ k : ℕ, a + b^2 = p^k) → 
  (a + b^2 ∣ a^2 + b) → 
  a = 5 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_special_integers_l592_59258


namespace NUMINAMATH_CALUDE_even_increasing_inequality_l592_59246

-- Define an even function that is increasing on [0,+∞)
def is_even_and_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x → x < y → f x < f y)

-- State the theorem
theorem even_increasing_inequality (f : ℝ → ℝ) 
  (h : is_even_and_increasing_on_nonneg f) : 
  f π > f (-3) ∧ f (-3) > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_inequality_l592_59246


namespace NUMINAMATH_CALUDE_only_B_is_difference_of_squares_l592_59247

-- Define the difference of squares formula
def difference_of_squares (a b : ℝ) : ℝ := a^2 - b^2

-- Define the expressions
def expr_A (x : ℝ) : ℝ := (x - 2) * (x + 1)
def expr_B (x y : ℝ) : ℝ := (x + 2*y) * (x - 2*y)
def expr_C (x y : ℝ) : ℝ := (x + y) * (-x - y)
def expr_D (x : ℝ) : ℝ := (-x + 1) * (x - 1)

-- Theorem stating that only expr_B fits the difference of squares formula
theorem only_B_is_difference_of_squares :
  (∃ (a b : ℝ), expr_B x y = difference_of_squares a b) ∧
  (∀ (a b : ℝ), expr_A x ≠ difference_of_squares a b) ∧
  (∀ (a b : ℝ), expr_C x y ≠ difference_of_squares a b) ∧
  (∀ (a b : ℝ), expr_D x ≠ difference_of_squares a b) :=
by sorry

end NUMINAMATH_CALUDE_only_B_is_difference_of_squares_l592_59247


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l592_59249

/-- Given an arithmetic sequence {a_n} where a_2 = -5 and the common difference d = 3,
    prove that the first term a_1 is equal to -8. -/
theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) 
    (h_arithmetic : ∀ n, a (n + 1) = a n + d)
    (h_a2 : a 2 = -5)
    (h_d : d = 3) : 
  a 1 = -8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l592_59249


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l592_59296

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2 * y + 3 * z)) :
  ∃ C : ℝ, ∀ x : ℝ, f x = C :=
sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l592_59296


namespace NUMINAMATH_CALUDE_perpendicular_tangents_theorem_l592_59235

noncomputable def f (x : ℝ) : ℝ := abs x / Real.exp x

def is_perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

theorem perpendicular_tangents_theorem (x₀ : ℝ) (m : ℤ) :
  x₀ > 0 ∧
  x₀ ∈ Set.Ioo (m / 4 : ℝ) ((m + 1) / 4 : ℝ) ∧
  is_perpendicular ((deriv f) (-1)) ((deriv f) x₀) →
  m = 2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_theorem_l592_59235


namespace NUMINAMATH_CALUDE_hotel_guests_count_l592_59242

/-- The number of guests attending the Oates reunion -/
def oates_attendees : ℕ := 50

/-- The number of guests attending the Hall reunion -/
def hall_attendees : ℕ := 62

/-- The number of guests attending both reunions -/
def both_attendees : ℕ := 12

/-- The total number of guests at the hotel -/
def total_guests : ℕ := (oates_attendees - both_attendees) + (hall_attendees - both_attendees) + both_attendees

theorem hotel_guests_count :
  total_guests = 100 := by sorry

end NUMINAMATH_CALUDE_hotel_guests_count_l592_59242


namespace NUMINAMATH_CALUDE_ceiling_sqrt_sum_l592_59280

theorem ceiling_sqrt_sum : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 27⌉ * 2 + ⌈Real.sqrt 243⌉ = 30 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_sum_l592_59280


namespace NUMINAMATH_CALUDE_stickers_per_sheet_l592_59293

theorem stickers_per_sheet 
  (initial_stickers : ℕ) 
  (shared_stickers : ℕ) 
  (remaining_sheets : ℕ) 
  (h1 : initial_stickers = 150)
  (h2 : shared_stickers = 100)
  (h3 : remaining_sheets = 5)
  (h4 : initial_stickers ≥ shared_stickers)
  (h5 : remaining_sheets > 0) :
  (initial_stickers - shared_stickers) / remaining_sheets = 10 :=
by sorry

end NUMINAMATH_CALUDE_stickers_per_sheet_l592_59293


namespace NUMINAMATH_CALUDE_hydrogen_atom_count_l592_59211

/-- Represents the number of atoms of each element in the compound -/
structure AtomCount where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

/-- Calculates the molecular weight of a compound -/
def molecularWeight (count : AtomCount) (weights : AtomicWeights) : ℝ :=
  count.carbon * weights.carbon + count.hydrogen * weights.hydrogen + count.oxygen * weights.oxygen

/-- The main theorem stating the number of hydrogen atoms in the compound -/
theorem hydrogen_atom_count (weights : AtomicWeights) 
    (h_carbon : weights.carbon = 12)
    (h_hydrogen : weights.hydrogen = 1)
    (h_oxygen : weights.oxygen = 16) : 
  ∃ (count : AtomCount), 
    count.carbon = 3 ∧ 
    count.oxygen = 1 ∧ 
    molecularWeight count weights = 58 ∧ 
    count.hydrogen = 6 := by
  sorry

end NUMINAMATH_CALUDE_hydrogen_atom_count_l592_59211


namespace NUMINAMATH_CALUDE_parabola_properties_l592_59219

/-- Definition of the parabola C: x² = 4y -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Definition of the line y = x + 1 -/
def line (x y : ℝ) : Prop := y = x + 1

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, 1)

/-- The length of the chord AB -/
def chord_length : ℝ := 8

/-- Theorem stating the properties of the parabola and its intersection with the line -/
theorem parabola_properties :
  (∀ x y, parabola x y → (x, y) ≠ focus → (x - focus.1)^2 + (y - focus.2)^2 > 0) ∧
  (∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = chord_length) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l592_59219


namespace NUMINAMATH_CALUDE_square_root_of_four_l592_59240

theorem square_root_of_four :
  {x : ℝ | x^2 = 4} = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l592_59240


namespace NUMINAMATH_CALUDE_irrational_condition_l592_59259

-- Define the set A(x)
def A (x : ℝ) : Set ℤ := {n : ℤ | ∃ m : ℕ, n = ⌊m * x⌋}

-- State the theorem
theorem irrational_condition (α : ℝ) (h_irr : Irrational α) (h_gt_two : α > 2) :
  ∀ β : ℝ, β > 0 → (A α ⊃ A β) → ∃ n : ℤ, β = n * α :=
by sorry

end NUMINAMATH_CALUDE_irrational_condition_l592_59259


namespace NUMINAMATH_CALUDE_polygon_interior_less_than_exterior_has_three_sides_l592_59273

theorem polygon_interior_less_than_exterior_has_three_sides
  (n : ℕ) -- number of sides of the polygon
  (h_polygon : n ≥ 3) -- n is at least 3 for a polygon
  (interior_sum : ℝ) -- sum of interior angles
  (exterior_sum : ℝ) -- sum of exterior angles
  (h_interior : interior_sum = (n - 2) * 180) -- formula for interior angle sum
  (h_exterior : exterior_sum = 360) -- exterior angle sum is always 360°
  (h_less : interior_sum < exterior_sum) -- given condition
  : n = 3 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_less_than_exterior_has_three_sides_l592_59273


namespace NUMINAMATH_CALUDE_letter_150_is_B_l592_59262

def repeating_pattern : ℕ → Char
  | n => match n % 4 with
    | 0 => 'D'
    | 1 => 'A'
    | 2 => 'B'
    | _ => 'C'

theorem letter_150_is_B : repeating_pattern 150 = 'B' := by
  sorry

end NUMINAMATH_CALUDE_letter_150_is_B_l592_59262


namespace NUMINAMATH_CALUDE_count_3digit_even_no_repeat_is_360_l592_59268

/-- A function that counts the number of 3-digit even numbers with no repeated digits -/
def count_3digit_even_no_repeat : ℕ :=
  let first_digit_options := 9  -- 1 to 9
  let second_digit_options := 8  -- Any digit except the first
  let last_digit_zero := first_digit_options * second_digit_options
  let last_digit_even_not_zero := first_digit_options * second_digit_options * 4
  last_digit_zero + last_digit_even_not_zero

/-- Theorem stating that the count of 3-digit even numbers with no repeated digits is 360 -/
theorem count_3digit_even_no_repeat_is_360 :
  count_3digit_even_no_repeat = 360 := by
  sorry

end NUMINAMATH_CALUDE_count_3digit_even_no_repeat_is_360_l592_59268


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l592_59210

theorem complex_fraction_equality (c d : ℂ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : c^2 - c*d + d^2 = 0) : 
  (c^12 + d^12) / (c + d)^12 = 2 / 81 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l592_59210


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l592_59254

theorem triangle_angle_calculation (a c : ℝ) (C : ℝ) (hA : a = 1) (hC : c = Real.sqrt 3) (hAngle : C = 2 * Real.pi / 3) :
  ∃ (A : ℝ), A = Real.pi / 6 ∧ 0 < A ∧ A < Real.pi ∧ 
  Real.sin A = a * Real.sin C / c ∧
  A + C < Real.pi :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l592_59254


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l592_59272

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 294 → volume = (((surface_area / 6) ^ (1/2 : ℝ)) ^ 3) → volume = 343 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l592_59272


namespace NUMINAMATH_CALUDE_game_score_theorem_l592_59207

theorem game_score_theorem (a b : ℕ) (h1 : 0 < b) (h2 : b < a) (h3 : a < 1986)
  (h4 : ∀ x : ℕ, x ≥ 1986 → ∃ (m n : ℕ), x = m * a + n * b)
  (h5 : ¬∃ (m n : ℕ), 1985 = m * a + n * b)
  (h6 : ¬∃ (m n : ℕ), 663 = m * a + n * b) :
  a = 332 ∧ b = 7 := by
sorry

end NUMINAMATH_CALUDE_game_score_theorem_l592_59207


namespace NUMINAMATH_CALUDE_candy_bar_cost_l592_59261

/-- The cost of a candy bar given initial and remaining amounts -/
theorem candy_bar_cost (initial_amount remaining_amount : ℕ) :
  initial_amount = 5 ∧ remaining_amount = 3 →
  initial_amount - remaining_amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l592_59261


namespace NUMINAMATH_CALUDE_number_of_women_is_six_l592_59277

/-- The number of women in a group that can color 360 meters of cloth in 3 days,
    given that 5 women can color 100 meters of cloth in 1 day. -/
def number_of_women : ℕ :=
  let meters_per_day := 360 / 3
  let meters_per_woman_per_day := 100 / 5
  meters_per_day / meters_per_woman_per_day

theorem number_of_women_is_six : number_of_women = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_women_is_six_l592_59277


namespace NUMINAMATH_CALUDE_quadratic_equation_a_range_l592_59278

/-- The range of values for a in the quadratic equation (a-1)x^2 + √(a+1)x + 2 = 0 -/
theorem quadratic_equation_a_range :
  ∀ a : ℝ, (∃ x : ℝ, (a - 1) * x^2 + Real.sqrt (a + 1) * x + 2 = 0) →
  (a ≥ -1 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_a_range_l592_59278


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l592_59216

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome :
  ∃ (n : ℕ), n > 6 ∧ 
    isPalindrome n 2 ∧ 
    isPalindrome n 4 ∧ 
    (∀ m : ℕ, m > 6 ∧ m < n → ¬(isPalindrome m 2 ∧ isPalindrome m 4)) ∧
    n = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l592_59216


namespace NUMINAMATH_CALUDE_football_players_count_l592_59217

theorem football_players_count (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 36)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 7) :
  total - neither - (tennis - both) = 26 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l592_59217


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l592_59227

def digits : List Nat := [1, 1, 5, 5]

def is_multiple_of_five (n : Nat) : Prop :=
  n % 5 = 0

def is_four_digit (n : Nat) : Prop :=
  n ≥ 1000 ∧ n < 10000

def count_valid_arrangements (ds : List Nat) : Nat :=
  sorry

theorem valid_arrangements_count :
  count_valid_arrangements digits = 3 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l592_59227


namespace NUMINAMATH_CALUDE_sum_longest_altitudes_is_21_l592_59267

/-- A triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 9
  hb : b = 12
  hc : c = 15
  right_angle : a^2 + b^2 = c^2

/-- The sum of the lengths of the two longest altitudes in the right triangle -/
def sum_longest_altitudes (t : RightTriangle) : ℝ := t.a + t.b

/-- Theorem stating that the sum of the two longest altitudes is 21 -/
theorem sum_longest_altitudes_is_21 (t : RightTriangle) :
  sum_longest_altitudes t = 21 := by sorry

end NUMINAMATH_CALUDE_sum_longest_altitudes_is_21_l592_59267


namespace NUMINAMATH_CALUDE_odd_times_abs_even_is_odd_l592_59275

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem odd_times_abs_even_is_odd (f g : ℝ → ℝ) 
  (h_f_odd : IsOdd f) (h_g_even : IsEven g) : 
  IsOdd (fun x ↦ f x * |g x|) := by
  sorry

end NUMINAMATH_CALUDE_odd_times_abs_even_is_odd_l592_59275


namespace NUMINAMATH_CALUDE_tommy_house_price_l592_59257

/-- The original price of Tommy's first house -/
def original_price : ℝ := 100000

/-- The increased value of Tommy's first house -/
def increased_value : ℝ := original_price * 1.25

/-- The cost of Tommy's new house -/
def new_house_cost : ℝ := 500000

/-- The percentage Tommy paid for the new house from his own funds -/
def own_funds_percentage : ℝ := 0.25

theorem tommy_house_price :
  original_price = 100000 ∧
  increased_value = original_price * 1.25 ∧
  new_house_cost = 500000 ∧
  own_funds_percentage = 0.25 ∧
  new_house_cost * own_funds_percentage = increased_value - original_price :=
by sorry

end NUMINAMATH_CALUDE_tommy_house_price_l592_59257


namespace NUMINAMATH_CALUDE_square_area_percent_difference_l592_59286

theorem square_area_percent_difference (A B : ℝ) (h : A > B) :
  (A^2 - B^2) / B^2 * 100 = 100 * (A^2 - B^2) / B^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_percent_difference_l592_59286


namespace NUMINAMATH_CALUDE_children_playing_both_sports_l592_59299

theorem children_playing_both_sports 
  (total : ℕ) 
  (tennis : ℕ) 
  (squash : ℕ) 
  (neither : ℕ) 
  (h1 : total = 38) 
  (h2 : tennis = 19) 
  (h3 : squash = 21) 
  (h4 : neither = 10) : 
  tennis + squash - (total - neither) = 12 := by
sorry

end NUMINAMATH_CALUDE_children_playing_both_sports_l592_59299


namespace NUMINAMATH_CALUDE_apartment_cost_increase_is_40_percent_l592_59260

/-- The percentage increase in cost of a new apartment compared to an old apartment. -/
def apartment_cost_increase (old_cost monthly_savings : ℚ) : ℚ := by
  -- Define John's share of the new apartment cost
  let johns_share := old_cost - monthly_savings
  -- Calculate the total new apartment cost (3 times John's share)
  let new_cost := 3 * johns_share
  -- Calculate the percentage increase
  exact ((new_cost - old_cost) / old_cost) * 100

/-- Theorem stating the percentage increase in apartment cost -/
theorem apartment_cost_increase_is_40_percent : 
  apartment_cost_increase 1200 (7680 / 12) = 40 := by
  sorry


end NUMINAMATH_CALUDE_apartment_cost_increase_is_40_percent_l592_59260


namespace NUMINAMATH_CALUDE_ice_cream_distribution_l592_59243

theorem ice_cream_distribution (total_sandwiches : ℕ) (num_nieces : ℕ) 
  (h1 : total_sandwiches = 1857)
  (h2 : num_nieces = 37) :
  (total_sandwiches / num_nieces : ℕ) = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_distribution_l592_59243


namespace NUMINAMATH_CALUDE_square_sum_from_conditions_l592_59251

theorem square_sum_from_conditions (x y : ℝ) 
  (h1 : x + 2 * y = 6) 
  (h2 : x * y = -6) : 
  x^2 + 4 * y^2 = 60 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_conditions_l592_59251
