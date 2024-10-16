import Mathlib

namespace NUMINAMATH_CALUDE_mn_value_l1642_164242

theorem mn_value (m n : ℤ) : 
  (∀ x : ℤ, (x + 5) * (x + n) = x^2 + m*x - 5) → m * n = -4 := by
  sorry

end NUMINAMATH_CALUDE_mn_value_l1642_164242


namespace NUMINAMATH_CALUDE_austin_picked_24_bags_l1642_164247

/-- The number of bags of apples Dallas picked -/
def dallas_apples : ℕ := 14

/-- The number of bags of pears Dallas picked -/
def dallas_pears : ℕ := 9

/-- The number of additional bags of apples Austin picked compared to Dallas -/
def austin_extra_apples : ℕ := 6

/-- The number of fewer bags of pears Austin picked compared to Dallas -/
def austin_fewer_pears : ℕ := 5

/-- The total number of bags of fruit Austin picked -/
def austin_total : ℕ := (dallas_apples + austin_extra_apples) + (dallas_pears - austin_fewer_pears)

theorem austin_picked_24_bags :
  austin_total = 24 := by sorry

end NUMINAMATH_CALUDE_austin_picked_24_bags_l1642_164247


namespace NUMINAMATH_CALUDE_five_skill_players_wait_l1642_164235

/-- Represents the water cooler scenario for a football team -/
structure WaterCooler where
  totalWater : ℕ
  numLinemen : ℕ
  numSkillPlayers : ℕ
  linemenWater : ℕ
  skillPlayerWater : ℕ

/-- Calculates the number of skill position players who must wait for water -/
def skillPlayersWaiting (wc : WaterCooler) : ℕ :=
  let linemenTotalWater := wc.numLinemen * wc.linemenWater
  let remainingWater := wc.totalWater - linemenTotalWater
  let skillPlayersServed := remainingWater / wc.skillPlayerWater
  wc.numSkillPlayers - skillPlayersServed

/-- Theorem stating that 5 skill position players must wait for water in the given scenario -/
theorem five_skill_players_wait (wc : WaterCooler) 
  (h1 : wc.totalWater = 126)
  (h2 : wc.numLinemen = 12)
  (h3 : wc.numSkillPlayers = 10)
  (h4 : wc.linemenWater = 8)
  (h5 : wc.skillPlayerWater = 6) :
  skillPlayersWaiting wc = 5 := by
  sorry

#eval skillPlayersWaiting { totalWater := 126, numLinemen := 12, numSkillPlayers := 10, linemenWater := 8, skillPlayerWater := 6 }

end NUMINAMATH_CALUDE_five_skill_players_wait_l1642_164235


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1642_164210

/-- A geometric sequence with first term a₁ = -1 and a₂ + a₃ = -2 has common ratio q = -2 or q = 1 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a1 : a 1 = -1) 
  (h_sum : a 2 + a 3 = -2) :
  q = -2 ∨ q = 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1642_164210


namespace NUMINAMATH_CALUDE_same_color_combination_probability_l1642_164224

def total_candies : ℕ := 12 + 8 + 5

theorem same_color_combination_probability :
  let red : ℕ := 12
  let blue : ℕ := 8
  let green : ℕ := 5
  let total : ℕ := total_candies
  
  -- Probability of picking two red candies
  let p_red : ℚ := (red * (red - 1)) / (total * (total - 1)) *
                   ((red - 2) * (red - 3)) / ((total - 2) * (total - 3))
  
  -- Probability of picking two blue candies
  let p_blue : ℚ := (blue * (blue - 1)) / (total * (total - 1)) *
                    ((blue - 2) * (blue - 3)) / ((total - 2) * (total - 3))
  
  -- Probability of picking two green candies
  let p_green : ℚ := (green * (green - 1)) / (total * (total - 1)) *
                     ((green - 2) * (green - 3)) / ((total - 2) * (total - 3))
  
  -- Total probability of picking the same color combination
  p_red + p_blue + p_green = 11 / 77 :=
by sorry

end NUMINAMATH_CALUDE_same_color_combination_probability_l1642_164224


namespace NUMINAMATH_CALUDE_prime_fraction_sum_l1642_164294

theorem prime_fraction_sum (p q x y : ℕ) : 
  Prime p → Prime q → x > 0 → y > 0 → x < p → y < q → 
  (∃ k : ℤ, (p : ℚ) / x + (q : ℚ) / y = k) → x = y := by
sorry

end NUMINAMATH_CALUDE_prime_fraction_sum_l1642_164294


namespace NUMINAMATH_CALUDE_painted_cubes_count_l1642_164248

/-- Represents a rectangular solid with given dimensions -/
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of cubes with one face painted in a rectangular solid -/
def cubes_with_one_face_painted (solid : RectangularSolid) : ℕ :=
  2 * ((solid.length - 2) * (solid.width - 2) +
       (solid.length - 2) * (solid.height - 2) +
       (solid.width - 2) * (solid.height - 2))

/-- Theorem: In a 9x10x11 rectangular solid, 382 cubes have exactly one face painted -/
theorem painted_cubes_count :
  let solid : RectangularSolid := ⟨9, 10, 11⟩
  cubes_with_one_face_painted solid = 382 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l1642_164248


namespace NUMINAMATH_CALUDE_gcd_2352_1560_l1642_164262

theorem gcd_2352_1560 : Nat.gcd 2352 1560 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2352_1560_l1642_164262


namespace NUMINAMATH_CALUDE_simplify_expression_l1642_164282

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) (hx2 : x ≠ 2) :
  (x - 2) / (x^2) / (1 - 2/x) = 1/x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1642_164282


namespace NUMINAMATH_CALUDE_sheila_attend_probability_l1642_164263

/-- The probability of rain -/
def prob_rain : ℝ := 0.3

/-- The probability of cloudy weather -/
def prob_cloudy : ℝ := 0.4

/-- The probability of sunshine -/
def prob_sunny : ℝ := 0.3

/-- The probability Sheila attends if it rains -/
def prob_attend_rain : ℝ := 0.25

/-- The probability Sheila attends if it's cloudy -/
def prob_attend_cloudy : ℝ := 0.5

/-- The probability Sheila attends if it's sunny -/
def prob_attend_sunny : ℝ := 0.75

/-- The theorem stating the probability of Sheila attending the picnic -/
theorem sheila_attend_probability : 
  prob_rain * prob_attend_rain + prob_cloudy * prob_attend_cloudy + prob_sunny * prob_attend_sunny = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_sheila_attend_probability_l1642_164263


namespace NUMINAMATH_CALUDE_price_determined_by_digits_price_for_one_price_for_twelve_price_for_five_hundred_twelve_l1642_164285

/-- Calculates the number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + num_digits (n / 10)

/-- Calculates the price based on the number of digits -/
def price (quantity : ℕ) : ℕ := 1000 * num_digits quantity

/-- Theorem stating that the price is determined by the number of digits -/
theorem price_determined_by_digits (quantity : ℕ) :
  price quantity = 1000 * num_digits quantity :=
by sorry

/-- Theorem verifying the price for one unit -/
theorem price_for_one : price 1 = 1000 :=
by sorry

/-- Theorem verifying the price for twelve units -/
theorem price_for_twelve : price 12 = 2000 :=
by sorry

/-- Theorem verifying the price for five hundred twelve units -/
theorem price_for_five_hundred_twelve : price 512 = 3000 :=
by sorry

end NUMINAMATH_CALUDE_price_determined_by_digits_price_for_one_price_for_twelve_price_for_five_hundred_twelve_l1642_164285


namespace NUMINAMATH_CALUDE_supplement_statement_is_proposition_l1642_164299

-- Define what a proposition is
def isPropositon (s : String) : Prop := ∃ (b : Bool), (b = true ∨ b = false)

-- Define the statement
def supplementStatement : String := "The supplements of the same angle are equal"

-- Theorem to prove
theorem supplement_statement_is_proposition : isPropositon supplementStatement := by
  sorry

end NUMINAMATH_CALUDE_supplement_statement_is_proposition_l1642_164299


namespace NUMINAMATH_CALUDE_doghouse_area_l1642_164221

/-- The area outside a regular hexagon that can be reached by a tethered point -/
theorem doghouse_area (side_length : ℝ) (rope_length : ℝ) (area : ℝ) :
  side_length = 2 →
  rope_length = 4 →
  area = 12 * Real.pi →
  area = (rope_length^2 * Real.pi * (2/3) + 2 * (side_length^2 * Real.pi * (1/6))) :=
by sorry

end NUMINAMATH_CALUDE_doghouse_area_l1642_164221


namespace NUMINAMATH_CALUDE_survey_respondents_count_l1642_164234

theorem survey_respondents_count :
  ∀ (x y : ℕ),
    x = 200 →
    4 * y = x →
    x + y = 250 :=
by sorry

end NUMINAMATH_CALUDE_survey_respondents_count_l1642_164234


namespace NUMINAMATH_CALUDE_triangle_unique_solution_l1642_164252

open Real

theorem triangle_unique_solution (a b : ℝ) (A : ℝ) (ha : a = 30) (hb : b = 25) (hA : A = 150 * π / 180) :
  ∃! B : ℝ, 0 < B ∧ B < π ∧ sin B = (b / a) * sin A :=
sorry

end NUMINAMATH_CALUDE_triangle_unique_solution_l1642_164252


namespace NUMINAMATH_CALUDE_parabola_focus_l1642_164295

/-- The parabola equation --/
def parabola (x y : ℝ) : Prop := y = (1/8) * x^2

/-- The focus of a parabola --/
def focus (p q : ℝ) : Prop := p = 0 ∧ q = 2

theorem parabola_focus :
  ∀ x y : ℝ, parabola x y → ∃ p q : ℝ, focus p q := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l1642_164295


namespace NUMINAMATH_CALUDE_retail_price_calculation_l1642_164273

theorem retail_price_calculation (total_cost : ℕ) (price_difference : ℕ) (additional_books : ℕ) :
  total_cost = 48 ∧ price_difference = 2 ∧ additional_books = 4 →
  ∃ (n : ℕ), n > 0 ∧ total_cost / n = 6 ∧ 
  (total_cost / n - price_difference) * (n + additional_books) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l1642_164273


namespace NUMINAMATH_CALUDE_composite_function_equation_l1642_164278

def δ (x : ℝ) : ℝ := 5 * x + 6
def φ (x : ℝ) : ℝ := 6 * x + 5

theorem composite_function_equation (x : ℝ) :
  δ (φ x) = -1 → x = -16/15 := by sorry

end NUMINAMATH_CALUDE_composite_function_equation_l1642_164278


namespace NUMINAMATH_CALUDE_inequality_solutions_l1642_164267

theorem inequality_solutions :
  let S : Set ℤ := {x | (x : ℚ) ≥ 0 ∧ (x - 2) / 2 ≤ (7 - x) / 3}
  S = {0, 1, 2, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_inequality_solutions_l1642_164267


namespace NUMINAMATH_CALUDE_mary_max_earnings_l1642_164260

/-- Calculates the maximum weekly earnings for a worker with given parameters. -/
def maxWeeklyEarnings (maxHours : ℕ) (regularHours : ℕ) (regularRate : ℚ) (overtimeRateIncrease : ℚ) : ℚ :=
  let overtimeRate := regularRate * (1 + overtimeRateIncrease)
  let regularEarnings := (regularHours.min maxHours : ℚ) * regularRate
  let overtimeHours := maxHours - regularHours
  let overtimeEarnings := (overtimeHours.max 0 : ℚ) * overtimeRate
  regularEarnings + overtimeEarnings

/-- Theorem stating Mary's maximum weekly earnings -/
theorem mary_max_earnings :
  maxWeeklyEarnings 80 20 8 (1/4) = 760 := by
  sorry

end NUMINAMATH_CALUDE_mary_max_earnings_l1642_164260


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1642_164201

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + y = 9) 
  (eq2 : x + 3 * y = 10) : 
  10 * x^2 + 19 * x * y + 10 * y^2 = 181 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1642_164201


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l1642_164205

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.log x / Real.log (1/2))^2 - 2 * (Real.log x / Real.log (1/2)) + 1

theorem f_monotone_increasing :
  ∀ x y, x ≥ Real.sqrt 2 / 2 → y ≥ Real.sqrt 2 / 2 → x < y → f x < f y :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l1642_164205


namespace NUMINAMATH_CALUDE_max_faces_convex_polyhedron_l1642_164226

/-- A convex polyhedron with n congruent triangular faces, each having angles 36°, 72°, and 72° -/
structure ConvexPolyhedron where
  n : ℕ  -- number of faces
  convex : Bool
  congruentFaces : Bool
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- The maximum number of faces for the given polyhedron is 36 -/
theorem max_faces_convex_polyhedron (p : ConvexPolyhedron) 
  (h1 : p.convex = true)
  (h2 : p.congruentFaces = true)
  (h3 : p.angleA = 36)
  (h4 : p.angleB = 72)
  (h5 : p.angleC = 72) :
  p.n ≤ 36 :=
sorry

end NUMINAMATH_CALUDE_max_faces_convex_polyhedron_l1642_164226


namespace NUMINAMATH_CALUDE_reaching_penglai_sufficient_for_immortal_l1642_164200

/-- Reaching Penglai implies becoming an immortal -/
def reaching_penglai_implies_immortal (reaching_penglai becoming_immortal : Prop) : Prop :=
  reaching_penglai → becoming_immortal

/-- Not reaching Penglai implies not becoming an immortal -/
axiom not_reaching_penglai_implies_not_immortal {reaching_penglai becoming_immortal : Prop} :
  ¬reaching_penglai → ¬becoming_immortal

/-- Prove that reaching Penglai is a sufficient condition for becoming an immortal -/
theorem reaching_penglai_sufficient_for_immortal
  {reaching_penglai becoming_immortal : Prop}
  (h : ¬reaching_penglai → ¬becoming_immortal) :
  reaching_penglai_implies_immortal reaching_penglai becoming_immortal :=
by sorry

end NUMINAMATH_CALUDE_reaching_penglai_sufficient_for_immortal_l1642_164200


namespace NUMINAMATH_CALUDE_remainder_1493829_div_7_l1642_164233

theorem remainder_1493829_div_7 : 1493829 % 7 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_1493829_div_7_l1642_164233


namespace NUMINAMATH_CALUDE_total_repair_cost_is_50_95_l1642_164241

def tire_repair_cost (num_tires : ℕ) (cost_per_tire : ℚ) (sales_tax : ℚ) 
                     (discount_rate : ℚ) (discount_valid : Bool) (city_fee : ℚ) : ℚ :=
  let base_cost := num_tires * cost_per_tire
  let tax_cost := num_tires * sales_tax
  let fee_cost := num_tires * city_fee
  let discount := if discount_valid then discount_rate * base_cost else 0
  base_cost + tax_cost + fee_cost - discount

theorem total_repair_cost_is_50_95 :
  let car_a_cost := tire_repair_cost 3 7 0.5 0.05 true 2.5
  let car_b_cost := tire_repair_cost 2 8.5 0 0.1 false 2.5
  car_a_cost + car_b_cost = 50.95 := by
sorry

#eval tire_repair_cost 3 7 0.5 0.05 true 2.5 + tire_repair_cost 2 8.5 0 0.1 false 2.5

end NUMINAMATH_CALUDE_total_repair_cost_is_50_95_l1642_164241


namespace NUMINAMATH_CALUDE_quadratic_root_square_relation_l1642_164202

theorem quadratic_root_square_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = x^2) →
  b^2 = 3 * a * c + c^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_square_relation_l1642_164202


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1642_164232

theorem rectangle_diagonal (l w : ℝ) (h1 : l + w = 23) (h2 : l * w = 120) :
  Real.sqrt (l^2 + w^2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1642_164232


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l1642_164215

theorem square_minus_product_plus_square : 6^2 - 5*6 + 4^2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l1642_164215


namespace NUMINAMATH_CALUDE_imaginary_power_sum_zero_l1642_164274

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum_zero : 
  i^14762 + i^14763 + i^14764 + i^14765 = 0 :=
by
  sorry

-- Define the property of i
axiom i_squared : i^2 = -1

end NUMINAMATH_CALUDE_imaginary_power_sum_zero_l1642_164274


namespace NUMINAMATH_CALUDE_eccentricity_ratio_for_common_point_l1642_164217

/-- The eccentricity of an ellipse -/
def eccentricity_ellipse (a b : ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity_hyperbola (a b : ℝ) : ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem eccentricity_ratio_for_common_point 
  (F₁ F₂ P : ℝ × ℝ) 
  (e₁ : ℝ) 
  (e₂ : ℝ) 
  (h_ellipse : e₁ = eccentricity_ellipse (distance F₁ P) (distance F₂ P))
  (h_hyperbola : e₂ = eccentricity_hyperbola (distance F₁ P) (distance F₂ P))
  (h_common_point : distance P F₁ + distance P F₂ = distance F₁ F₂) :
  (e₁ * e₂) / Real.sqrt (e₁^2 + e₂^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_eccentricity_ratio_for_common_point_l1642_164217


namespace NUMINAMATH_CALUDE_inequality_proof_l1642_164288

theorem inequality_proof (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) 
  (hab : a + b = 1) : 
  (a*x + b*y) * (b*x + a*y) ≥ x*y := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1642_164288


namespace NUMINAMATH_CALUDE_equation_solution_l1642_164218

theorem equation_solution (a b x : ℝ) :
  (a ≠ b ∧ a ≠ -b ∧ b ≠ 0 → x = a^2 - b^2 → 
    1 / (a + b) + (a - b) / x = 1 / (a - b) + (a - b) / x) ∧
  (b = 0 ∧ a ≠ 0 ∧ x ≠ 0 → 
    1 / (a + b) + (a - b) / x = 1 / (a - b) + (a - b) / x) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1642_164218


namespace NUMINAMATH_CALUDE_correct_expansion_of_expression_l1642_164254

theorem correct_expansion_of_expression (a : ℝ) : 
  5 + a - 2 * (3 * a - 5) = 5 + a - 6 * a + 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_expansion_of_expression_l1642_164254


namespace NUMINAMATH_CALUDE_probability_two_male_finalists_l1642_164284

/-- The probability of selecting two male finalists from a group of 7 finalists (3 male, 4 female) -/
theorem probability_two_male_finalists (total : ℕ) (males : ℕ) (females : ℕ) 
  (h_total : total = 7)
  (h_males : males = 3)
  (h_females : females = 4)
  (h_sum : males + females = total) :
  (males.choose 2 : ℚ) / (total.choose 2) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_male_finalists_l1642_164284


namespace NUMINAMATH_CALUDE_a_value_m_minimum_l1642_164259

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- Theorem 1: Prove that a = 1
theorem a_value : 
  ∀ x ∈ solution_set 1, f 1 x ≤ 6 ∧
  ∀ a : ℝ, (∀ x ∈ solution_set a, f a x ≤ 6) → a = 1 :=
sorry

-- Define the function g (which is f with a = 1)
def g (x : ℝ) : ℝ := |2*x - 1| + 1

-- Theorem 2: Prove that the minimum value of m is 3.5
theorem m_minimum :
  (∃ m : ℝ, ∀ t : ℝ, g (t/2) ≤ m - g (-t)) ∧
  (∀ m : ℝ, (∀ t : ℝ, g (t/2) ≤ m - g (-t)) → m ≥ 3.5) :=
sorry

end NUMINAMATH_CALUDE_a_value_m_minimum_l1642_164259


namespace NUMINAMATH_CALUDE_max_t_value_l1642_164264

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * x^2 - 2*x + Real.log (x + 1)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x - Real.log (x + 1) + x^3

theorem max_t_value (m : ℝ) (t : ℝ) :
  m ∈ Set.Icc (-4) (-1) →
  (∀ x ∈ Set.Icc 1 t, g m x ≤ g m 1) →
  t ≤ (1 + Real.sqrt 13) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_t_value_l1642_164264


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l1642_164275

theorem absolute_value_equation_solutions :
  ∃! (s : Set ℝ), s = {x : ℝ | |x + 1| = |x - 2| + |x - 5| + |x - 6|} ∧ s = {4, 7} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l1642_164275


namespace NUMINAMATH_CALUDE_complex_number_location_l1642_164279

theorem complex_number_location :
  let z : ℂ := Complex.I * (1 + Complex.I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l1642_164279


namespace NUMINAMATH_CALUDE_medical_team_selection_l1642_164229

theorem medical_team_selection (nurses : ℕ) (doctors : ℕ) : 
  nurses = 3 → doctors = 6 → 
  (Nat.choose (nurses + doctors) 5 - Nat.choose doctors 5) = 120 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_selection_l1642_164229


namespace NUMINAMATH_CALUDE_distance_to_origin_l1642_164214

def A : ℝ × ℝ := (-1, -2)

theorem distance_to_origin : Real.sqrt ((A.1 - 0)^2 + (A.2 - 0)^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1642_164214


namespace NUMINAMATH_CALUDE_south_movement_representation_l1642_164255

/-- Represents the direction of movement -/
inductive Direction
  | North
  | South

/-- Represents a movement with a distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Converts a movement to its signed representation -/
def Movement.toSigned (m : Movement) : ℝ :=
  match m.direction with
  | Direction.North => m.distance
  | Direction.South => -m.distance

/-- The problem statement -/
theorem south_movement_representation :
  let north20 : Movement := ⟨20, Direction.North⟩
  let south120 : Movement := ⟨120, Direction.South⟩
  north20.toSigned = 20 →
  south120.toSigned = -120 := by
  sorry

end NUMINAMATH_CALUDE_south_movement_representation_l1642_164255


namespace NUMINAMATH_CALUDE_exponential_multiplication_specific_exponential_multiplication_l1642_164298

theorem exponential_multiplication (n : ℕ) : (10 : ℝ) ^ n * (10 : ℝ) ^ n = (10 : ℝ) ^ (2 * n) := by
  sorry

-- The specific case for n = 1000
theorem specific_exponential_multiplication : (10 : ℝ) ^ 1000 * (10 : ℝ) ^ 1000 = (10 : ℝ) ^ 2000 := by
  sorry

end NUMINAMATH_CALUDE_exponential_multiplication_specific_exponential_multiplication_l1642_164298


namespace NUMINAMATH_CALUDE_solve_equation_l1642_164268

theorem solve_equation (x : ℝ) : 3 * x = (26 - x) + 14 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1642_164268


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1642_164225

-- Define the solution set type
def SolutionSet := Set ℝ

-- Define the inequalities
def Inequality1 (k a b c : ℝ) (x : ℝ) : Prop :=
  k / (x + a) + (x + b) / (x + c) < 0

def Inequality2 (k a b c : ℝ) (x : ℝ) : Prop :=
  (k * x) / (a * x + 1) + (b * x + 1) / (c * x + 1) < 0

-- State the theorem
theorem solution_set_equivalence 
  (k a b c : ℝ) 
  (S1 : SolutionSet) 
  (h1 : S1 = {x | x ∈ (Set.Ioo (-3) (-1)) ∪ (Set.Ioo 1 2) ∧ Inequality1 k a b c x}) :
  {x | Inequality2 k a b c x} = 
    {x | x ∈ (Set.Ioo (-1) (-1/3)) ∪ (Set.Ioo (1/2) 1)} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1642_164225


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l1642_164286

def not_in_second_quadrant (m n : ℝ) : Prop :=
  (m / n > 0) ∧ (1 / n < 0)

theorem mn_positive_necessary_not_sufficient (m n : ℝ) :
  (not_in_second_quadrant m n → m * n > 0) ∧
  ¬(m * n > 0 → not_in_second_quadrant m n) :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l1642_164286


namespace NUMINAMATH_CALUDE_roots_sum_powers_l1642_164236

theorem roots_sum_powers (a b : ℝ) : 
  a^2 - 3*a + 2 = 0 → b^2 - 3*b + 2 = 0 → a^3 + a^4*b^2 + a^2*b^4 + b^3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l1642_164236


namespace NUMINAMATH_CALUDE_range_of_g_range_of_g_complete_l1642_164256

def f (x : ℝ) : ℝ := 5 * x + 3

def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ y ∈ Set.range g, -157 ≤ y ∧ y ≤ 1093 :=
sorry

theorem range_of_g_complete :
  ∀ y, -157 ≤ y ∧ y ≤ 1093 → ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_g_range_of_g_complete_l1642_164256


namespace NUMINAMATH_CALUDE_greatest_c_value_l1642_164209

theorem greatest_c_value (c : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 20 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 9*5 - 20 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_c_value_l1642_164209


namespace NUMINAMATH_CALUDE_solution_implies_c_value_l1642_164269

-- Define the function f
def f (x b : ℝ) : ℝ := x^2 + x + b

-- State the theorem
theorem solution_implies_c_value
  (b : ℝ)  -- b is a real number
  (h1 : ∀ x, f x b ≥ 0)  -- Value range of f is [0, +∞)
  (h2 : ∃ m, ∀ x, f x b < 16 ↔ x < m + 8)  -- Solution to f(x) < c is m + 8
  : 16 = 16 :=  -- We want to prove c = 16
by
  sorry

end NUMINAMATH_CALUDE_solution_implies_c_value_l1642_164269


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1642_164213

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 + x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1642_164213


namespace NUMINAMATH_CALUDE_first_train_crossing_time_l1642_164246

/-- Two trains running in opposite directions with equal speeds -/
structure TwoTrains where
  v₁ : ℝ  -- Speed of the first train
  v₂ : ℝ  -- Speed of the second train
  L₁ : ℝ  -- Length of the first train
  L₂ : ℝ  -- Length of the second train
  t₂ : ℝ  -- Time taken by the second train to cross the man
  cross_time : ℝ  -- Time taken for the trains to cross each other

/-- The conditions given in the problem -/
def problem_conditions (trains : TwoTrains) : Prop :=
  trains.v₁ > 0 ∧ 
  trains.v₂ > 0 ∧ 
  trains.L₁ > 0 ∧ 
  trains.L₂ > 0 ∧ 
  trains.v₁ = trains.v₂ ∧  -- Ratio of speeds is 1
  trains.t₂ = 17 ∧  -- Second train crosses the man in 17 seconds
  trains.cross_time = 22 ∧  -- Trains cross each other in 22 seconds
  (trains.L₁ + trains.L₂) / (trains.v₁ + trains.v₂) = trains.cross_time

/-- The theorem to be proved -/
theorem first_train_crossing_time (trains : TwoTrains) 
  (h : problem_conditions trains) : 
  trains.L₁ / trains.v₁ = 27 := by
  sorry


end NUMINAMATH_CALUDE_first_train_crossing_time_l1642_164246


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l1642_164290

theorem cos_pi_third_minus_alpha (α : ℝ) (h : Real.sin (π / 6 + α) = 1 / 3) :
  Real.cos (π / 3 - α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l1642_164290


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l1642_164243

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Define the half-open interval [1, 2)
def interval : Set ℝ := {x | 1 ≤ x ∧ x < 2}

-- Theorem statement
theorem intersection_equals_interval : M ∩ N = interval := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l1642_164243


namespace NUMINAMATH_CALUDE_positive_abc_l1642_164206

theorem positive_abc (a b c : ℝ) 
  (sum_pos : a + b + c > 0)
  (sum_prod_pos : a * b + b * c + c * a > 0)
  (prod_pos : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
sorry

end NUMINAMATH_CALUDE_positive_abc_l1642_164206


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_eighty_is_half_more_than_forty_l1642_164280

theorem twenty_five_percent_less_than_eighty_is_half_more_than_forty : 
  ∃ x : ℝ, (80 - 0.25 * 80 = x + 0.5 * x) ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_eighty_is_half_more_than_forty_l1642_164280


namespace NUMINAMATH_CALUDE_red_ball_probability_l1642_164265

/-- The probability of drawing a red ball from a bag with 1 red ball and 4 white balls is 0.2 -/
theorem red_ball_probability (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 1 →
  white_balls = 4 →
  (red_balls : ℚ) / total_balls = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_red_ball_probability_l1642_164265


namespace NUMINAMATH_CALUDE_find_number_l1642_164270

theorem find_number : ∃ x : ℝ, 3 * x + 3 * 14 + 3 * 18 + 11 = 152 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1642_164270


namespace NUMINAMATH_CALUDE_expression_equals_zero_l1642_164212

theorem expression_equals_zero (x y z : ℝ) (h : x*y + y*z + z*x = 0) :
  3*x*y*z + x^2*(y + z) + y^2*(z + x) + z^2*(x + y) = 0 := by sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l1642_164212


namespace NUMINAMATH_CALUDE_closest_beetle_positions_l1642_164253

structure Table where
  sugar_position : ℝ × ℝ
  ant_radius : ℝ
  beetle_radius : ℝ
  ant_initial_position : ℝ × ℝ
  beetle_initial_position : ℝ × ℝ

def closest_positions (t : Table) : Set (ℝ × ℝ) :=
  {(2, 2 * Real.sqrt 3), (-4, 0), (2, -2 * Real.sqrt 3)}

theorem closest_beetle_positions (t : Table) 
  (h1 : t.sugar_position = (0, 0))
  (h2 : t.ant_radius = 2)
  (h3 : t.beetle_radius = 4)
  (h4 : t.ant_initial_position = (-1, Real.sqrt 3))
  (h5 : t.beetle_initial_position = (2 * Real.sqrt 3, 2)) :
  closest_positions t = {(2, 2 * Real.sqrt 3), (-4, 0), (2, -2 * Real.sqrt 3)} := by
  sorry

end NUMINAMATH_CALUDE_closest_beetle_positions_l1642_164253


namespace NUMINAMATH_CALUDE_special_triangle_side_length_l1642_164237

/-- An equilateral triangle with a special interior point -/
structure SpecialTriangle where
  -- The side length of the equilateral triangle
  s : ℝ
  -- The coordinates of the interior point P
  P : ℝ × ℝ
  -- Condition that the triangle is equilateral
  equilateral : s > 0
  -- Conditions for distances from P to vertices
  dist_AP : Real.sqrt ((P.1 - s/2)^2 + (P.2 - Real.sqrt 3 * s/2)^2) = Real.sqrt 2
  dist_BP : Real.sqrt ((P.1 - s)^2 + P.2^2) = 2
  dist_CP : Real.sqrt P.1^2 + P.2^2 = 1

/-- The side length of a special triangle is 5 -/
theorem special_triangle_side_length (t : SpecialTriangle) : t.s = 5 := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_side_length_l1642_164237


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1642_164251

/-- Given an arithmetic sequence where:
    - S_n is the sum of the first n terms
    - S_{2n} is the sum of the first 2n terms
    - S_{3n} is the sum of the first 3n terms
    This theorem proves that if S_n = 45 and S_{2n} = 60, then S_{3n} = 65. -/
theorem arithmetic_sequence_sum (n : ℕ) (S_n S_2n S_3n : ℝ) 
  (h1 : S_n = 45)
  (h2 : S_2n = 60) :
  S_3n = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1642_164251


namespace NUMINAMATH_CALUDE_blue_balls_count_l1642_164287

theorem blue_balls_count (B : ℕ) : 
  (5 : ℚ) * 4 / (2 * ((7 + B : ℚ) * (6 + B))) = 0.1282051282051282 → B = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_count_l1642_164287


namespace NUMINAMATH_CALUDE_difference_in_combined_area_l1642_164276

/-- The difference in combined area (front and back) between two rectangular sheets of paper -/
theorem difference_in_combined_area : 
  let sheet1_length : ℝ := 11
  let sheet1_width : ℝ := 13
  let sheet2_length : ℝ := 6.5
  let sheet2_width : ℝ := 11
  let combined_area (l w : ℝ) := 2 * l * w
  combined_area sheet1_length sheet1_width - combined_area sheet2_length sheet2_width = 143
  := by sorry

end NUMINAMATH_CALUDE_difference_in_combined_area_l1642_164276


namespace NUMINAMATH_CALUDE_complex_fraction_problem_l1642_164203

theorem complex_fraction_problem (m : ℝ) : 
  let z₁ : ℂ := m + Complex.I
  let z₂ : ℂ := 1 - 2 * Complex.I
  (z₁ / z₂ = -1/2) → m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_problem_l1642_164203


namespace NUMINAMATH_CALUDE_imaginary_part_of_fraction_l1642_164239

theorem imaginary_part_of_fraction (i : ℂ) : i * i = -1 → Complex.im ((1 - i) / (1 + i)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_fraction_l1642_164239


namespace NUMINAMATH_CALUDE_range_of_a_l1642_164281

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, |x + a| ≤ 2) → a ∈ Set.Icc (-3) 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1642_164281


namespace NUMINAMATH_CALUDE_prob_four_same_face_five_coins_l1642_164293

/-- The number of coins being flipped -/
def num_coins : ℕ := 5

/-- The probability of getting at least 'num_same' coins showing the same face when flipping 'num_coins' fair coins -/
def prob_same_face (num_same : ℕ) : ℚ :=
  let total_outcomes := 2^num_coins
  let favorable_outcomes := 2 * (Nat.choose num_coins (num_coins - num_same + 1))
  favorable_outcomes / total_outcomes

/-- The probability of getting at least 4 coins showing the same face when flipping 5 fair coins is 3/8 -/
theorem prob_four_same_face_five_coins : prob_same_face 4 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_same_face_five_coins_l1642_164293


namespace NUMINAMATH_CALUDE_expression_simplification_l1642_164245

theorem expression_simplification (x : ℝ) : 
  x - 3*(1+x) + 4*(1-x)^2 - 5*(1+3*x) = 4*x^2 - 25*x - 4 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l1642_164245


namespace NUMINAMATH_CALUDE_disk_division_l1642_164204

/-- Represents the number of non-overlapping areas in a circular disk divided by radii and a tangent line -/
def num_areas (n : ℕ) : ℕ := 3 * n

/-- Theorem stating that the number of non-overlapping areas in a circular disk
    divided by 3n equally spaced radii and one tangent line is equal to 3n -/
theorem disk_division (n : ℕ) :
  num_areas n = 3 * n :=
by sorry

end NUMINAMATH_CALUDE_disk_division_l1642_164204


namespace NUMINAMATH_CALUDE_inequality_proof_l1642_164244

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_eq : 1 + a + b + c = 2 * a * b * c) : 
  (a * b) / (1 + a + b) + (b * c) / (1 + b + c) + (c * a) / (1 + c + a) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1642_164244


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1642_164207

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 1) :
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1642_164207


namespace NUMINAMATH_CALUDE_reading_time_difference_l1642_164228

/-- The difference in reading time between two people with different reading speeds -/
theorem reading_time_difference 
  (tristan_speed : ℝ) 
  (ella_speed : ℝ) 
  (book_pages : ℝ) 
  (h1 : tristan_speed = 120)
  (h2 : ella_speed = 40)
  (h3 : book_pages = 360) :
  (book_pages / ella_speed - book_pages / tristan_speed) * 60 = 360 :=
by sorry

end NUMINAMATH_CALUDE_reading_time_difference_l1642_164228


namespace NUMINAMATH_CALUDE_probability_two_non_defective_10_2_l1642_164208

/-- Given a box of pens, calculates the probability of selecting two non-defective pens. -/
def probability_two_non_defective (total_pens : ℕ) (defective_pens : ℕ) : ℚ :=
  let non_defective := total_pens - defective_pens
  (non_defective : ℚ) / total_pens * (non_defective - 1) / (total_pens - 1)

/-- Theorem stating that the probability of selecting two non-defective pens
    from a box of 10 pens with 2 defective pens is 28/45. -/
theorem probability_two_non_defective_10_2 :
  probability_two_non_defective 10 2 = 28 / 45 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_non_defective_10_2_l1642_164208


namespace NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l1642_164211

/-- Given that dried grapes contain 20% water by weight and 40 kg of fresh grapes
    produce 5 kg of dried grapes, prove that the percentage of water in fresh grapes is 90%. -/
theorem water_percentage_in_fresh_grapes :
  ∀ (fresh_weight dried_weight : ℝ) (dried_water_percentage : ℝ),
    fresh_weight = 40 →
    dried_weight = 5 →
    dried_water_percentage = 20 →
    (fresh_weight - dried_weight * (1 - dried_water_percentage / 100)) / fresh_weight * 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l1642_164211


namespace NUMINAMATH_CALUDE_second_year_percentage_correct_l1642_164250

/-- The number of second-year students studying numeric methods -/
def numeric_methods : ℕ := 250

/-- The number of second-year students studying automatic control of airborne vehicles -/
def automatic_control : ℕ := 423

/-- The number of second-year students studying both subjects -/
def both_subjects : ℕ := 134

/-- The total number of students in the faculty -/
def total_students : ℕ := 673

/-- The percentage of second-year students in the faculty -/
def second_year_percentage : ℚ :=
  (numeric_methods + automatic_control - both_subjects : ℚ) / total_students * 100

theorem second_year_percentage_correct :
  second_year_percentage = (250 + 423 - 134 : ℚ) / 673 * 100 :=
by sorry

end NUMINAMATH_CALUDE_second_year_percentage_correct_l1642_164250


namespace NUMINAMATH_CALUDE_bubble_sort_probability_main_result_l1642_164296

def n : ℕ := 50

/-- The probability that r₂₅ ends up in the 35th position after one bubble pass -/
def probability : ℚ := 1 / 1190

theorem bubble_sort_probability (r : Fin n → ℕ) (h : Function.Injective r) :
  probability = (Nat.factorial 33) / (Nat.factorial 35) :=
sorry

theorem main_result : probability.num + probability.den = 1191 :=
sorry

end NUMINAMATH_CALUDE_bubble_sort_probability_main_result_l1642_164296


namespace NUMINAMATH_CALUDE_digit_150_of_75_over_625_l1642_164231

theorem digit_150_of_75_over_625 : ∃ (d : ℕ), d = 2 ∧ 
  (∃ (a b : ℕ), (75 : ℚ) / 625 = ↑a + (↑b / 100) ∧ 
  (∀ n : ℕ, (75 * 10^(n+2)) % 625 = (75 * 10^(n+150)) % 625) ∧
  d = ((75 * 10^150) / 625) % 10) :=
sorry

end NUMINAMATH_CALUDE_digit_150_of_75_over_625_l1642_164231


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_one_l1642_164271

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line l₁: x + ay + 3 = 0 -/
def l1 (a : ℝ) : Line :=
  { a := 1, b := a, c := 3 }

/-- The second line l₂: (a-2)x + 3y + a = 0 -/
def l2 (a : ℝ) : Line :=
  { a := a - 2, b := 3, c := a }

/-- Theorem: The lines l₁ and l₂ are parallel if and only if a = -1 -/
theorem lines_parallel_iff_a_eq_neg_one :
  ∀ a : ℝ, are_parallel (l1 a) (l2 a) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_one_l1642_164271


namespace NUMINAMATH_CALUDE_sequence_a_property_sequence_a_formula_l1642_164289

def sequence_a : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n + 2) => sorry

theorem sequence_a_property (n : ℕ) (h : n ≥ 1) :
  n * (n + 1) * sequence_a (n + 1) = n * (n - 1) * sequence_a n - (n - 2) * sequence_a (n - 1) := by sorry

theorem sequence_a_formula (n : ℕ) (h : n ≥ 2) : sequence_a n = 1 / n.factorial := by sorry

end NUMINAMATH_CALUDE_sequence_a_property_sequence_a_formula_l1642_164289


namespace NUMINAMATH_CALUDE_chicken_chick_difference_l1642_164222

theorem chicken_chick_difference (total : ℕ) (chicks : ℕ) : 
  total = 821 → chicks = 267 → total - chicks - chicks = 287 := by
  sorry

end NUMINAMATH_CALUDE_chicken_chick_difference_l1642_164222


namespace NUMINAMATH_CALUDE_cotton_planting_rate_l1642_164227

/-- Calculates the required acres per tractor per day to plant cotton --/
theorem cotton_planting_rate (total_acres : ℕ) (total_days : ℕ) 
  (tractors_first_period : ℕ) (days_first_period : ℕ)
  (tractors_second_period : ℕ) (days_second_period : ℕ) :
  total_acres = 1700 →
  total_days = 5 →
  tractors_first_period = 2 →
  days_first_period = 2 →
  tractors_second_period = 7 →
  days_second_period = 3 →
  (total_acres : ℚ) / ((tractors_first_period * days_first_period + 
    tractors_second_period * days_second_period) : ℚ) = 68 := by
  sorry

#eval (1700 : ℚ) / 25  -- Should output 68

end NUMINAMATH_CALUDE_cotton_planting_rate_l1642_164227


namespace NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l1642_164277

-- Problem 1
theorem simplify_fraction_1 (a x : ℝ) (ha : a > 0) (hx : x > 0) :
  (a * Real.sqrt x - x * Real.sqrt a) / (Real.sqrt a - Real.sqrt x) = Real.sqrt (a * x) := by
  sorry

-- Problem 2
theorem simplify_fraction_2 (a b : ℝ) (ha : a > 0) (hb : b^2 < a^2) :
  (Real.sqrt (a + b) - Real.sqrt (a - b)) / (a + b - Real.sqrt (a^2 - b^2)) = 1 / Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l1642_164277


namespace NUMINAMATH_CALUDE_pass_rate_two_procedures_l1642_164249

theorem pass_rate_two_procedures (a b : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  let pass_rate := (1 - a) * (1 - b)
  0 ≤ pass_rate ∧ pass_rate ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_pass_rate_two_procedures_l1642_164249


namespace NUMINAMATH_CALUDE_lineup_calculation_1_lineup_calculation_2_l1642_164297

/-- Represents a basketball team -/
structure BasketballTeam where
  veterans : Nat
  newPlayers : Nat

/-- Represents the conditions for lineup selection -/
structure LineupConditions where
  specificVeteranMustPlay : Bool
  specificNewPlayersCannotPlay : Nat
  forwardPlayers : Nat
  guardPlayers : Nat
  versatilePlayers : Nat

/-- Calculates the number of different lineups under given conditions -/
def calculateLineups (team : BasketballTeam) (conditions : LineupConditions) : Nat :=
  sorry

/-- Theorem for the first lineup calculation -/
theorem lineup_calculation_1 (team : BasketballTeam) (conditions : LineupConditions) :
  team.veterans = 7 ∧ team.newPlayers = 5 ∧
  conditions.specificVeteranMustPlay = true ∧
  conditions.specificNewPlayersCannotPlay = 2 →
  calculateLineups team conditions = 126 :=
sorry

/-- Theorem for the second lineup calculation -/
theorem lineup_calculation_2 (team : BasketballTeam) (conditions : LineupConditions) :
  team.veterans + team.newPlayers = 12 ∧
  conditions.forwardPlayers = 6 ∧
  conditions.guardPlayers = 4 ∧
  conditions.versatilePlayers = 2 →
  calculateLineups team conditions = 636 :=
sorry

end NUMINAMATH_CALUDE_lineup_calculation_1_lineup_calculation_2_l1642_164297


namespace NUMINAMATH_CALUDE_fraction_sum_equals_2315_over_1200_l1642_164257

theorem fraction_sum_equals_2315_over_1200 :
  (1/2 : ℚ) * (3/4) + (5/6) * (7/8) + (9/10) * (11/12) = 2315/1200 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_2315_over_1200_l1642_164257


namespace NUMINAMATH_CALUDE_sum_of_divisors_24_l1642_164230

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_24 : sum_of_divisors 24 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_24_l1642_164230


namespace NUMINAMATH_CALUDE_at_least_one_less_than_or_equal_to_one_l1642_164261

theorem at_least_one_less_than_or_equal_to_one 
  (x y z : ℝ) 
  (pos_x : 0 < x) 
  (pos_y : 0 < y) 
  (pos_z : 0 < z) 
  (sum_eq_three : x + y + z = 3) :
  (x * (x + y - z) ≤ 1) ∨ (y * (y + z - x) ≤ 1) ∨ (z * (z + x - y) ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_or_equal_to_one_l1642_164261


namespace NUMINAMATH_CALUDE_inequality_proof_l1642_164266

theorem inequality_proof (x : ℝ) (h : x > 0) : x^8 - x^5 - 1/x + 1/x^4 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1642_164266


namespace NUMINAMATH_CALUDE_problem_solution_l1642_164216

theorem problem_solution (p q : ℚ) 
  (h1 : 5 * p + 7 * q = 19)
  (h2 : 7 * p + 5 * q = 26) : 
  p = 29 / 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1642_164216


namespace NUMINAMATH_CALUDE_prob_king_queen_heart_l1642_164240

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Number of Hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of King of Hearts in a standard deck -/
def NumKingOfHearts : ℕ := 1

/-- Probability of drawing a King, then a Queen, then a Heart from a standard 52-card deck -/
theorem prob_king_queen_heart : 
  (NumKings * (NumQueens - 1) * NumHearts + 
   NumKingOfHearts * NumQueens * (NumHearts - 1) + 
   NumKingOfHearts * (NumQueens - 1) * (NumHearts - 1)) / 
  (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2)) = 67 / 44200 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_queen_heart_l1642_164240


namespace NUMINAMATH_CALUDE_grape_purchases_l1642_164258

theorem grape_purchases (lena_shown ira_shown combined_shown : ℝ)
  (h1 : lena_shown = 2)
  (h2 : ira_shown = 3)
  (h3 : combined_shown = 4.5) :
  ∃ (lena_actual ira_actual offset : ℝ),
    lena_actual + offset = lena_shown ∧
    ira_actual + offset = ira_shown ∧
    lena_actual + ira_actual = combined_shown ∧
    lena_actual = 1.5 ∧
    ira_actual = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_grape_purchases_l1642_164258


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1642_164220

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1642_164220


namespace NUMINAMATH_CALUDE_fish_tank_problem_l1642_164283

theorem fish_tank_problem (initial_fish : ℕ) : 
  (initial_fish - 4 = 8) → (initial_fish + 8 = 20) := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l1642_164283


namespace NUMINAMATH_CALUDE_biancas_books_l1642_164238

/-- The number of coloring books Bianca has after giving some away and buying more -/
def final_book_count (initial : ℕ) (given_away : ℕ) (bought : ℕ) : ℕ :=
  initial - given_away + bought

/-- Theorem stating that Bianca's final book count is 59 -/
theorem biancas_books : final_book_count 45 6 20 = 59 := by
  sorry

end NUMINAMATH_CALUDE_biancas_books_l1642_164238


namespace NUMINAMATH_CALUDE_square_rearrangement_theorem_l1642_164219

-- Define a type for square sheets of paper
def Square : Type := Unit

-- Define a function that represents the possibility of cutting and rearranging squares
def can_cut_and_rearrange (n : ℕ) : Prop :=
  ∀ (squares : Fin n → Square), ∃ (new_square : Square), True

-- State the theorem
theorem square_rearrangement_theorem (n : ℕ) (h : n > 1) :
  can_cut_and_rearrange n :=
sorry

end NUMINAMATH_CALUDE_square_rearrangement_theorem_l1642_164219


namespace NUMINAMATH_CALUDE_x_power_y_value_l1642_164292

theorem x_power_y_value (x y : ℝ) (h : |x + 1/2| + (y - 3)^2 = 0) : x^y = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_x_power_y_value_l1642_164292


namespace NUMINAMATH_CALUDE_lantern_tower_top_count_l1642_164223

/-- Represents a tower with geometric progression of lanterns -/
structure LanternTower where
  levels : ℕ
  ratio : ℕ
  total : ℕ
  top : ℕ

/-- Calculates the sum of a geometric sequence -/
def geometricSum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

/-- Theorem: In a 7-level tower where the number of lanterns doubles at each level
    from top to bottom, and the total number of lanterns is 381,
    the number of lanterns at the top level is 3. -/
theorem lantern_tower_top_count (tower : LanternTower)
    (h1 : tower.levels = 7)
    (h2 : tower.ratio = 2)
    (h3 : tower.total = 381)
    : tower.top = 3 := by
  sorry

#check lantern_tower_top_count

end NUMINAMATH_CALUDE_lantern_tower_top_count_l1642_164223


namespace NUMINAMATH_CALUDE_power_of_three_equality_l1642_164291

theorem power_of_three_equality : (3^5)^6 = 3^12 * 3^18 := by sorry

end NUMINAMATH_CALUDE_power_of_three_equality_l1642_164291


namespace NUMINAMATH_CALUDE_westward_movement_negative_l1642_164272

/-- Represents the direction of movement --/
inductive Direction
| East
| West

/-- Represents a movement with magnitude and direction --/
structure Movement where
  magnitude : ℝ
  direction : Direction

/-- Converts a movement to a signed real number --/
def Movement.toSignedReal (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.magnitude
  | Direction.West => -m.magnitude

theorem westward_movement_negative 
  (east_convention : Movement.toSignedReal { magnitude := 2, direction := Direction.East } = 2) :
  Movement.toSignedReal { magnitude := 3, direction := Direction.West } = -3 := by
  sorry

end NUMINAMATH_CALUDE_westward_movement_negative_l1642_164272
