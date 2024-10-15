import Mathlib

namespace NUMINAMATH_CALUDE_ratio_problem_l3634_363410

theorem ratio_problem (y : ℚ) : (1 : ℚ) / 3 = y / 5 → y = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3634_363410


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3634_363457

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- Condition that the asymptote slope is positive -/
  asymptote_slope_pos : asymptote_slope > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem: The eccentricity of a hyperbola with asymptote slope √2/2 is √6/2 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = Real.sqrt 2 / 2) : 
    eccentricity h = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3634_363457


namespace NUMINAMATH_CALUDE_bank_exceeds_500_on_day_9_l3634_363414

def deposit_amount (day : ℕ) : ℕ :=
  if day ≤ 1 then 3
  else if day % 2 = 0 then 3 * deposit_amount (day - 2)
  else deposit_amount (day - 1)

def total_amount (day : ℕ) : ℕ :=
  List.sum (List.map deposit_amount (List.range (day + 1)))

theorem bank_exceeds_500_on_day_9 :
  total_amount 8 ≤ 500 ∧ total_amount 9 > 500 :=
sorry

end NUMINAMATH_CALUDE_bank_exceeds_500_on_day_9_l3634_363414


namespace NUMINAMATH_CALUDE_two_numbers_with_specific_means_l3634_363472

theorem two_numbers_with_specific_means :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  Real.sqrt (a * b) = 2 * Real.sqrt 3 ∧
  (a + b) / 2 = 6 ∧
  2 / (1 / a + 1 / b) = 2 ∧
  ((a = 6 - 2 * Real.sqrt 6 ∧ b = 6 + 2 * Real.sqrt 6) ∨
   (a = 6 + 2 * Real.sqrt 6 ∧ b = 6 - 2 * Real.sqrt 6)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_specific_means_l3634_363472


namespace NUMINAMATH_CALUDE_complex_calculation_result_l3634_363490

theorem complex_calculation_result : (13.672 * 125 + 136.72 * 12.25 - 1367.2 * 1.875) / 17.09 = 60.5 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_result_l3634_363490


namespace NUMINAMATH_CALUDE_matthew_owns_26_cheap_shares_l3634_363412

/-- Calculates the number of shares of the less valuable stock Matthew owns --/
def calculate_less_valuable_shares (total_assets : ℕ) (expensive_share_price : ℕ) (expensive_shares : ℕ) : ℕ :=
  let cheap_share_price := expensive_share_price / 2
  let expensive_stock_value := expensive_share_price * expensive_shares
  let cheap_stock_value := total_assets - expensive_stock_value
  cheap_stock_value / cheap_share_price

/-- Proves that Matthew owns 26 shares of the less valuable stock --/
theorem matthew_owns_26_cheap_shares :
  calculate_less_valuable_shares 2106 78 14 = 26 := by
  sorry

end NUMINAMATH_CALUDE_matthew_owns_26_cheap_shares_l3634_363412


namespace NUMINAMATH_CALUDE_f_min_value_l3634_363440

/-- The function f as defined in the problem -/
def f (x y : ℝ) : ℝ := x^3 + y^3 + x^2*y + x*y^2 - 3*(x^2 + y^2 + x*y) + 3*(x + y)

/-- Theorem stating that f(x,y) ≥ 1 for all x,y ≥ 1/2 -/
theorem f_min_value (x y : ℝ) (hx : x ≥ 1/2) (hy : y ≥ 1/2) : f x y ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l3634_363440


namespace NUMINAMATH_CALUDE_cube_root_of_27_l3634_363420

theorem cube_root_of_27 (x : ℝ) (h : (Real.sqrt x) ^ 3 = 27) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_27_l3634_363420


namespace NUMINAMATH_CALUDE_triangle_determinant_l3634_363452

theorem triangle_determinant (A B C : Real) (h1 : A = 45 * π / 180)
    (h2 : B = 75 * π / 180) (h3 : C = 60 * π / 180) :
  let M : Matrix (Fin 3) (Fin 3) Real :=
    ![![Real.tan A, 1, 1],
      ![1, Real.tan B, 1],
      ![1, 1, Real.tan C]]
  Matrix.det M = -1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_determinant_l3634_363452


namespace NUMINAMATH_CALUDE_complex_calculation_l3634_363409

theorem complex_calculation : 
  (2/3 * Real.sqrt 180) * (0.4 * 300)^3 - (0.4 * 180 - 1/3 * (0.4 * 180)) = 15454377.6 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l3634_363409


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_fractions_l3634_363463

theorem min_value_of_sum_of_fractions (n : ℕ) (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
  (1 / (1 + a ^ n) + 1 / (1 + b ^ n)) ≥ 1 ∧ 
  (1 / (1 + 1 ^ n) + 1 / (1 + 1 ^ n) = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_fractions_l3634_363463


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3634_363419

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 7*x < 8 ↔ -8 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3634_363419


namespace NUMINAMATH_CALUDE_inequality_solution_l3634_363491

theorem inequality_solution (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  (2 * x) / (x + 1) + (x - 3) / (3 * x) ≤ 4 ↔ x < -1 ∨ x > -1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3634_363491


namespace NUMINAMATH_CALUDE_find_other_number_l3634_363481

theorem find_other_number (x y : ℤ) (h1 : 4 * x + 3 * y = 154) (h2 : x = 14 ∨ y = 14) : x = 28 ∨ y = 28 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l3634_363481


namespace NUMINAMATH_CALUDE_cinema_seating_l3634_363433

/-- The number of people sitting between the far right and far left audience members -/
def people_between : ℕ := 30

/-- The total number of people sitting in the chairs -/
def total_people : ℕ := people_between + 2

theorem cinema_seating : total_people = 32 := by
  sorry

end NUMINAMATH_CALUDE_cinema_seating_l3634_363433


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3634_363435

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the arithmetic sequence condition
def arithmeticSequenceCondition (a : ℕ → ℝ) : Prop :=
  a 2 + a 1 = a 3

-- Main theorem
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geometric : isGeometricSequence a q)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arithmetic : arithmeticSequenceCondition a) :
  q = (Real.sqrt 5 + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3634_363435


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3634_363483

theorem division_remainder_problem (L S R : ℕ) : 
  L - S = 1365 →
  L = 1636 →
  L = 6 * S + R →
  R < S →
  R = 10 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3634_363483


namespace NUMINAMATH_CALUDE_triangle_problem_l3634_363429

theorem triangle_problem (A B C a b c p : ℝ) :
  -- Triangle ABC with angles A, B, C corresponding to sides a, b, c
  (0 < A ∧ A < π) → (0 < B ∧ B < π) → (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  -- Given conditions
  (Real.sin A + Real.sin C = p * Real.sin B) →
  (a * c = (1/4) * b^2) →
  -- Part I
  (p = 5/4 ∧ b = 1) →
  ((a = 1 ∧ c = 1/4) ∨ (a = 1/4 ∧ c = 1)) ∧
  -- Part II
  (0 < B ∧ B < π/2) →
  (Real.sqrt 6 / 2 < p ∧ p < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3634_363429


namespace NUMINAMATH_CALUDE_solution_of_system_l3634_363407

variable (a b c x y z : ℝ)

theorem solution_of_system :
  (a * x + b * y - c * z = 2 * a * b) →
  (a * x - b * y + c * z = 2 * a * c) →
  (-a * x + b * y - c * z = 2 * b * c) →
  (x = b + c ∧ y = a + c ∧ z = a + b) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l3634_363407


namespace NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l3634_363450

/-- Given an arithmetic progression with first term 7, if adding 3 to the second term and 25 to the third term
    results in a geometric progression, then the smallest possible value for the third term of the geometric
    progression is -0.62. -/
theorem smallest_third_term_of_geometric_progression (d : ℝ) :
  let a₁ := 7
  let a₂ := a₁ + d
  let a₃ := a₁ + 2*d
  let g₁ := a₁
  let g₂ := a₂ + 3
  let g₃ := a₃ + 25
  (g₂^2 = g₁ * g₃) →
  ∃ (d' : ℝ), g₃ ≥ -0.62 ∧ (∀ (d'' : ℝ),
    let g₁' := 7
    let g₂' := (7 + d'') + 3
    let g₃' := (7 + 2*d'') + 25
    (g₂'^2 = g₁' * g₃') → g₃' ≥ g₃) :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l3634_363450


namespace NUMINAMATH_CALUDE_soap_brands_survey_l3634_363462

/-- The number of households that use both brands of soap -/
def households_both_brands : ℕ := 30

/-- The total number of households surveyed -/
def total_households : ℕ := 260

/-- The number of households that use neither brand A nor brand B -/
def households_neither_brand : ℕ := 80

/-- The number of households that use only brand A -/
def households_only_A : ℕ := 60

theorem soap_brands_survey :
  households_both_brands = 30 ∧
  total_households = households_neither_brand + households_only_A + households_both_brands + 3 * households_both_brands :=
by sorry

end NUMINAMATH_CALUDE_soap_brands_survey_l3634_363462


namespace NUMINAMATH_CALUDE_min_difference_l3634_363402

/-- Represents a 4-digit positive integer ABCD -/
def FourDigitNum (a b c d : Nat) : Nat :=
  1000 * a + 100 * b + 10 * c + d

/-- Represents a 2-digit positive integer -/
def TwoDigitNum (x y : Nat) : Nat :=
  10 * x + y

/-- The difference between a 4-digit number and the product of its two 2-digit parts -/
def Difference (a b c d : Nat) : Nat :=
  FourDigitNum a b c d - TwoDigitNum a b * TwoDigitNum c d

theorem min_difference :
  ∀ (a b c d : Nat),
    a ≠ 0 → c ≠ 0 →
    a < 10 → b < 10 → c < 10 → d < 10 →
    Difference a b c d ≥ 109 :=
by sorry

end NUMINAMATH_CALUDE_min_difference_l3634_363402


namespace NUMINAMATH_CALUDE_journey_distance_l3634_363477

/-- A journey with two parts -/
structure Journey where
  total_time : ℝ
  speed1 : ℝ
  time1 : ℝ
  speed2 : ℝ

/-- Calculate the total distance of a journey -/
def total_distance (j : Journey) : ℝ :=
  j.speed1 * j.time1 + j.speed2 * (j.total_time - j.time1)

/-- Theorem: The total distance of the given journey is 240 km -/
theorem journey_distance :
  ∃ (j : Journey),
    j.total_time = 5 ∧
    j.speed1 = 40 ∧
    j.time1 = 3 ∧
    j.speed2 = 60 ∧
    total_distance j = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l3634_363477


namespace NUMINAMATH_CALUDE_correct_stratified_sample_size_l3634_363453

/-- Represents the student population in each year -/
structure StudentPopulation where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the stratified sample size for each year -/
def stratifiedSampleSize (population : StudentPopulation) (total_sample : ℕ) : ℕ × ℕ × ℕ :=
  sorry

/-- Theorem stating that the calculated stratified sample sizes are correct -/
theorem correct_stratified_sample_size 
  (population : StudentPopulation)
  (h1 : population.first_year = 540)
  (h2 : population.second_year = 440)
  (h3 : population.third_year = 420)
  (total_sample : ℕ)
  (h4 : total_sample = 70) :
  stratifiedSampleSize population total_sample = (27, 22, 21) :=
sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_size_l3634_363453


namespace NUMINAMATH_CALUDE_combined_savings_equal_separate_savings_l3634_363432

/-- Represents the store's window offer -/
structure WindowOffer where
  price : ℕ  -- Price per window
  buy : ℕ    -- Number of windows to buy
  free : ℕ   -- Number of free windows

/-- Calculates the cost for a given number of windows under the offer -/
def calculateCost (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let groups := windowsNeeded / (offer.buy + offer.free)
  let remainder := windowsNeeded % (offer.buy + offer.free)
  (groups * offer.buy + min remainder offer.buy) * offer.price

/-- Calculates the savings for a given number of windows under the offer -/
def calculateSavings (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  windowsNeeded * offer.price - calculateCost offer windowsNeeded

theorem combined_savings_equal_separate_savings 
  (offer : WindowOffer)
  (davesWindows : ℕ)
  (dougsWindows : ℕ)
  (h1 : offer.price = 150)
  (h2 : offer.buy = 8)
  (h3 : offer.free = 2)
  (h4 : davesWindows = 10)
  (h5 : dougsWindows = 16) :
  calculateSavings offer (davesWindows + dougsWindows) = 
  calculateSavings offer davesWindows + calculateSavings offer dougsWindows :=
by sorry

end NUMINAMATH_CALUDE_combined_savings_equal_separate_savings_l3634_363432


namespace NUMINAMATH_CALUDE_accumulate_small_steps_necessary_not_sufficient_l3634_363492

-- Define the concept of "reaching a thousand miles"
def reach_thousand_miles : Prop := sorry

-- Define the concept of "accumulating small steps"
def accumulate_small_steps : Prop := sorry

-- Xunzi's saying as an axiom
axiom xunzi_saying : ¬accumulate_small_steps → ¬reach_thousand_miles

-- Define what it means to be a necessary condition
def is_necessary_condition (condition goal : Prop) : Prop :=
  ¬condition → ¬goal

-- Define what it means to be a sufficient condition
def is_sufficient_condition (condition goal : Prop) : Prop :=
  condition → goal

-- Theorem to prove
theorem accumulate_small_steps_necessary_not_sufficient :
  (is_necessary_condition accumulate_small_steps reach_thousand_miles) ∧
  ¬(is_sufficient_condition accumulate_small_steps reach_thousand_miles) := by
  sorry

end NUMINAMATH_CALUDE_accumulate_small_steps_necessary_not_sufficient_l3634_363492


namespace NUMINAMATH_CALUDE_min_sum_of_slopes_l3634_363438

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop :=
  x^2 - 4*(x+y) + y^2 = 2*x*y + 8

/-- Tangent line to the parabola at point (a, b) -/
def tangent_line (a b x y : ℝ) : Prop :=
  y - b = ((b - a + 2) / (b - a - 2)) * (x - a)

/-- Intersection point of the tangent lines -/
def intersection_point (p q : ℝ) : Prop :=
  p + q = -32

theorem min_sum_of_slopes :
  ∃ (a b p q : ℝ),
    parabola a b ∧
    parabola b a ∧
    intersection_point p q ∧
    tangent_line a b p q ∧
    tangent_line b a p q ∧
    ((b - a + 2) / (b - a - 2) + (a - b + 2) / (a - b - 2) ≥ 62 / 29) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_slopes_l3634_363438


namespace NUMINAMATH_CALUDE_lauren_mail_count_l3634_363471

/-- The number of pieces of mail Lauren sent on Monday -/
def monday : ℕ := 65

/-- The number of pieces of mail Lauren sent on Tuesday -/
def tuesday : ℕ := monday + 10

/-- The number of pieces of mail Lauren sent on Wednesday -/
def wednesday : ℕ := tuesday - 5

/-- The number of pieces of mail Lauren sent on Thursday -/
def thursday : ℕ := wednesday + 15

/-- The total number of pieces of mail Lauren sent over four days -/
def total_mail : ℕ := monday + tuesday + wednesday + thursday

theorem lauren_mail_count : total_mail = 295 := by sorry

end NUMINAMATH_CALUDE_lauren_mail_count_l3634_363471


namespace NUMINAMATH_CALUDE_marble_probability_l3634_363480

theorem marble_probability (box1 box2 : Nat) : 
  box1 + box2 = 36 →
  (box1 * box2 : Rat) = 36 →
  (∃ black1 black2 : Nat, 
    black1 ≤ box1 ∧ 
    black2 ≤ box2 ∧ 
    (black1 * black2 : Rat) / (box1 * box2) = 25 / 36) →
  (∃ white1 white2 : Nat,
    white1 = box1 - black1 ∧
    white2 = box2 - black2 ∧
    (white1 * white2 : Rat) / (box1 * box2) = 169 / 324) :=
by sorry

end NUMINAMATH_CALUDE_marble_probability_l3634_363480


namespace NUMINAMATH_CALUDE_volleyballs_count_l3634_363406

/-- The number of volleyballs in Reynald's purchase --/
def volleyballs : ℕ :=
  let total_balls : ℕ := 145
  let soccer_balls : ℕ := 20
  let basketballs : ℕ := soccer_balls + 5
  let tennis_balls : ℕ := 2 * soccer_balls
  let baseballs : ℕ := soccer_balls + 10
  total_balls - (soccer_balls + basketballs + tennis_balls + baseballs)

/-- Theorem stating that the number of volleyballs is 30 --/
theorem volleyballs_count : volleyballs = 30 := by
  sorry

end NUMINAMATH_CALUDE_volleyballs_count_l3634_363406


namespace NUMINAMATH_CALUDE_currency_conversion_weight_conversion_gram_to_kg_weight_conversion_kg_to_ton_length_conversion_l3634_363455

-- Define conversion rates
def yuan_to_jiao : ℚ := 10
def yuan_to_fen : ℚ := 100
def kg_to_gram : ℚ := 1000
def ton_to_kg : ℚ := 1000
def meter_to_cm : ℚ := 100

-- Define the conversion functions
def jiao_to_yuan (j : ℚ) : ℚ := j / yuan_to_jiao
def fen_to_yuan (f : ℚ) : ℚ := f / yuan_to_fen
def gram_to_kg (g : ℚ) : ℚ := g / kg_to_gram
def kg_to_ton (k : ℚ) : ℚ := k / ton_to_kg
def cm_to_meter (c : ℚ) : ℚ := c / meter_to_cm

-- Theorem statements
theorem currency_conversion :
  5 + jiao_to_yuan 4 + fen_to_yuan 8 = 5.48 := by sorry

theorem weight_conversion_gram_to_kg :
  gram_to_kg 80 = 0.08 := by sorry

theorem weight_conversion_kg_to_ton :
  kg_to_ton 73 = 0.073 := by sorry

theorem length_conversion :
  1 + cm_to_meter 5 = 1.05 := by sorry

end NUMINAMATH_CALUDE_currency_conversion_weight_conversion_gram_to_kg_weight_conversion_kg_to_ton_length_conversion_l3634_363455


namespace NUMINAMATH_CALUDE_find_a_value_l3634_363400

/-- The problem statement translated to Lean 4 --/
theorem find_a_value (a : ℝ) :
  (∀ x y : ℝ, 2*x - y + a ≥ 0 ∧ 3*x + y - 3 ≤ 0 →
    4*x + 3*y ≤ 8) ∧
  (∃ x y : ℝ, 2*x - y + a ≥ 0 ∧ 3*x + y - 3 ≤ 0 ∧
    4*x + 3*y = 8) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l3634_363400


namespace NUMINAMATH_CALUDE_expression_evaluation_l3634_363488

theorem expression_evaluation :
  let x : ℝ := -1
  let y : ℝ := 2
  2 * x^2 - y^2 + (2 * y^2 - 3 * x^2) - (2 * y^2 + x^2) = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3634_363488


namespace NUMINAMATH_CALUDE_certification_cost_certification_cost_proof_l3634_363448

/-- The cost of certification for John's seeing-eye dog --/
theorem certification_cost : ℝ → Prop :=
  fun c =>
    let adoption_fee : ℝ := 150
    let training_cost_per_week : ℝ := 250
    let training_weeks : ℝ := 12
    let insurance_coverage_percent : ℝ := 90
    let total_out_of_pocket : ℝ := 3450
    let total_cost_before_certification : ℝ := adoption_fee + training_cost_per_week * training_weeks
    let out_of_pocket_certification : ℝ := c * (100 - insurance_coverage_percent) / 100
    total_out_of_pocket = total_cost_before_certification + out_of_pocket_certification →
    c = 3000

/-- Proof of the certification cost --/
theorem certification_cost_proof : certification_cost 3000 := by
  sorry

end NUMINAMATH_CALUDE_certification_cost_certification_cost_proof_l3634_363448


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l3634_363484

theorem quadratic_perfect_square (x : ℝ) : 
  (∃ a : ℝ, x^2 + 10*x + 25 = (x + a)^2) ∧ 
  (∀ c : ℝ, c ≠ 25 → ¬∃ a : ℝ, x^2 + 10*x + c = (x + a)^2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l3634_363484


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3634_363469

theorem matrix_equation_solution : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 9, 4]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -43/14, 53/14]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3634_363469


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l3634_363445

/-- A hyperbola with equation mx² + y² = 1 where the length of its imaginary axis 
    is twice the length of its real axis -/
structure Hyperbola where
  m : ℝ
  eq : ∀ x y : ℝ, m * x^2 + y^2 = 1
  axis_ratio : (imaginary_axis_length : ℝ) = 2 * (real_axis_length : ℝ)

/-- The value of m for a hyperbola with the given properties is -1/4 -/
theorem hyperbola_m_value (h : Hyperbola) : h.m = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l3634_363445


namespace NUMINAMATH_CALUDE_circular_arrangement_students_l3634_363422

/-- Given a circular arrangement of students, if the 8th student is directly opposite the 33rd student, then the total number of students is 52. -/
theorem circular_arrangement_students (n : ℕ) : 
  (∃ (a b : ℕ), a = 8 ∧ b = 33 ∧ a < b ∧ b - a = n - (b - a)) → n = 52 := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_students_l3634_363422


namespace NUMINAMATH_CALUDE_abc_sum_bound_l3634_363437

theorem abc_sum_bound (a b c : ℝ) (h : a + b + c = 1) :
  ∃ (x : ℝ), x ≤ 1/2 ∧ ∃ (a' b' c' : ℝ), a' + b' + c' = 1 ∧ a'*b' + a'*c' + b'*c' = x :=
sorry

end NUMINAMATH_CALUDE_abc_sum_bound_l3634_363437


namespace NUMINAMATH_CALUDE_problem_solution_l3634_363495

theorem problem_solution (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3634_363495


namespace NUMINAMATH_CALUDE_grammar_club_committee_probability_l3634_363401

/-- The number of boys in the Grammar club -/
def num_boys : ℕ := 15

/-- The number of girls in the Grammar club -/
def num_girls : ℕ := 15

/-- The size of the committee to be formed -/
def committee_size : ℕ := 5

/-- The minimum number of boys required in the committee -/
def min_boys : ℕ := 2

/-- The probability of forming a committee with at least 2 boys and at least 1 girl -/
def committee_probability : ℚ := 515 / 581

/-- Theorem stating the probability of forming a committee with the given conditions -/
theorem grammar_club_committee_probability :
  let total_members := num_boys + num_girls
  let valid_committees := (Finset.range (committee_size + 1)).filter (λ k => k ≥ min_boys ∧ k < committee_size)
    |>.sum (λ k => (Nat.choose num_boys k) * (Nat.choose num_girls (committee_size - k)))
  let total_committees := Nat.choose total_members committee_size
  (valid_committees : ℚ) / total_committees = committee_probability := by
  sorry

#check grammar_club_committee_probability

end NUMINAMATH_CALUDE_grammar_club_committee_probability_l3634_363401


namespace NUMINAMATH_CALUDE_mult_41_equivalence_l3634_363498

theorem mult_41_equivalence (x y : ℤ) :
  (25 * x + 31 * y) % 41 = 0 ↔ (3 * x + 7 * y) % 41 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mult_41_equivalence_l3634_363498


namespace NUMINAMATH_CALUDE_mrs_wonderful_class_size_l3634_363443

theorem mrs_wonderful_class_size :
  ∀ (girls : ℕ) (boys : ℕ),
  boys = girls + 3 →
  girls * girls + boys * boys + 10 + 8 = 450 →
  girls + boys = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_wonderful_class_size_l3634_363443


namespace NUMINAMATH_CALUDE_parent_current_age_l3634_363444

-- Define the son's age next year
def sons_age_next_year : ℕ := 8

-- Define the relation between parent's and son's age
def parent_age_relation (parent_age son_age : ℕ) : Prop :=
  parent_age = 5 * son_age

-- Theorem to prove
theorem parent_current_age : 
  ∃ (parent_age : ℕ), parent_age_relation parent_age (sons_age_next_year - 1) ∧ parent_age = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_parent_current_age_l3634_363444


namespace NUMINAMATH_CALUDE_jason_newspaper_earnings_l3634_363494

/-- Proves that Jason's earnings from delivering newspapers equals $1.875 --/
theorem jason_newspaper_earnings 
  (fred_initial : ℝ) 
  (jason_initial : ℝ) 
  (emily_initial : ℝ) 
  (fred_increase : ℝ) 
  (jason_increase : ℝ) 
  (emily_increase : ℝ) 
  (h1 : fred_initial = 49) 
  (h2 : jason_initial = 3) 
  (h3 : emily_initial = 25) 
  (h4 : fred_increase = 1.5) 
  (h5 : jason_increase = 1.625) 
  (h6 : emily_increase = 1.4) :
  jason_initial * (jason_increase - 1) = 1.875 := by
  sorry

end NUMINAMATH_CALUDE_jason_newspaper_earnings_l3634_363494


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3634_363403

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 7 ∧ 
  ∀ (m : ℤ), |n - (5^3 + 7^3)^(1/3)| ≤ |m - (5^3 + 7^3)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3634_363403


namespace NUMINAMATH_CALUDE_change_in_responses_l3634_363468

theorem change_in_responses (initial_yes initial_no final_yes final_no : ℚ) 
  (h1 : initial_yes = 50 / 100)
  (h2 : initial_no = 50 / 100)
  (h3 : final_yes = 70 / 100)
  (h4 : final_no = 30 / 100)
  (h5 : initial_yes + initial_no = 1)
  (h6 : final_yes + final_no = 1) :
  ∃ (min_change max_change : ℚ),
    min_change ≥ 0 ∧
    max_change ≤ 1 ∧
    min_change ≤ max_change ∧
    max_change - min_change = 30 / 100 :=
by sorry

end NUMINAMATH_CALUDE_change_in_responses_l3634_363468


namespace NUMINAMATH_CALUDE_six_integers_mean_twice_mode_l3634_363442

theorem six_integers_mean_twice_mode (x y : ℕ) : 
  x > 0 ∧ y > 0 ∧ 
  x ≤ 100 ∧ y ≤ 100 ∧
  y > x ∧
  (21 + 45 + 77 + 2 * x + y) / 6 = 2 * x →
  x = 16 := by
sorry

end NUMINAMATH_CALUDE_six_integers_mean_twice_mode_l3634_363442


namespace NUMINAMATH_CALUDE_readers_all_three_genres_l3634_363436

/-- Represents the number of readers for each genre and their intersections --/
structure ReaderCounts where
  total : ℕ
  sf : ℕ
  lw : ℕ
  hf : ℕ
  sf_lw : ℕ
  sf_hf : ℕ
  lw_hf : ℕ

/-- The principle of inclusion-exclusion for three sets --/
def inclusionExclusion (r : ReaderCounts) (x : ℕ) : Prop :=
  r.total = r.sf + r.lw + r.hf - r.sf_lw - r.sf_hf - r.lw_hf + x

/-- The theorem stating the number of readers who read all three genres --/
theorem readers_all_three_genres (r : ReaderCounts) 
  (h_total : r.total = 800)
  (h_sf : r.sf = 300)
  (h_lw : r.lw = 600)
  (h_hf : r.hf = 400)
  (h_sf_lw : r.sf_lw = 175)
  (h_sf_hf : r.sf_hf = 150)
  (h_lw_hf : r.lw_hf = 250) :
  ∃ x, inclusionExclusion r x ∧ x = 75 := by
  sorry

end NUMINAMATH_CALUDE_readers_all_three_genres_l3634_363436


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3634_363447

theorem product_of_sums_equals_difference_of_powers : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l3634_363447


namespace NUMINAMATH_CALUDE_coronavirus_spread_rate_l3634_363479

/-- The number of people infected after two rounds of novel coronavirus spread -/
def total_infected : ℕ := 121

/-- The number of people initially infected -/
def initial_infected : ℕ := 1

/-- The average number of people infected by one person in each round -/
def m : ℕ := 10

/-- Theorem stating that m = 10 given the conditions of the coronavirus spread -/
theorem coronavirus_spread_rate :
  (initial_infected + m)^2 = total_infected :=
sorry

end NUMINAMATH_CALUDE_coronavirus_spread_rate_l3634_363479


namespace NUMINAMATH_CALUDE_unique_b_for_three_integer_solutions_l3634_363431

theorem unique_b_for_three_integer_solutions :
  ∃! b : ℤ, ∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x : ℤ, x ∈ s ↔ x^2 + b*x + 5 ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_unique_b_for_three_integer_solutions_l3634_363431


namespace NUMINAMATH_CALUDE_sum_of_intercepts_l3634_363404

-- Define the parabola
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 5

-- Define the x-intercept
def x_intercept : ℝ := parabola 0

-- Define the y-intercepts
def y_intercepts : Set ℝ := {y | parabola y = 0}

-- Theorem statement
theorem sum_of_intercepts :
  ∃ (b c : ℝ), b ∈ y_intercepts ∧ c ∈ y_intercepts ∧ b ≠ c ∧ x_intercept + b + c = 8 :=
sorry

end NUMINAMATH_CALUDE_sum_of_intercepts_l3634_363404


namespace NUMINAMATH_CALUDE_chris_bluray_purchase_l3634_363473

/-- The number of Blu-ray movies Chris bought -/
def num_bluray : ℕ := sorry

/-- The number of DVD movies Chris bought -/
def num_dvd : ℕ := 8

/-- The price of each DVD movie -/
def price_dvd : ℚ := 12

/-- The price of each Blu-ray movie -/
def price_bluray : ℚ := 18

/-- The average price per movie -/
def avg_price : ℚ := 14

theorem chris_bluray_purchase :
  (num_dvd * price_dvd + num_bluray * price_bluray) / (num_dvd + num_bluray) = avg_price ∧
  num_bluray = 4 := by sorry

end NUMINAMATH_CALUDE_chris_bluray_purchase_l3634_363473


namespace NUMINAMATH_CALUDE_fib_100_mod_5_l3634_363416

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Theorem statement
theorem fib_100_mod_5 : fib 99 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_100_mod_5_l3634_363416


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l3634_363474

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l3634_363474


namespace NUMINAMATH_CALUDE_twelfth_term_is_twelve_l3634_363497

/-- An arithmetic sequence with a₂ = -8 and common difference d = 2 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  -8 + 2 * (n - 2)

/-- Theorem: The 12th term of the arithmetic sequence is 12 -/
theorem twelfth_term_is_twelve : arithmetic_sequence 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_is_twelve_l3634_363497


namespace NUMINAMATH_CALUDE_optimal_game_result_exact_distinct_rows_l3634_363405

/-- Represents the game board -/
def GameBoard := Fin (2^100) → Fin 100 → Bool

/-- Player A's strategy -/
def StrategyA := GameBoard → Fin 100 → Fin 100

/-- Player B's strategy -/
def StrategyB := GameBoard → Fin 100 → Fin 100

/-- Counts the number of distinct rows in a game board -/
def countDistinctRows (board : GameBoard) : ℕ := sorry

/-- Simulates the game with given strategies -/
def playGame (stratA : StrategyA) (stratB : StrategyB) : GameBoard := sorry

/-- The main theorem stating the result of the game -/
theorem optimal_game_result :
  ∀ (stratA : StrategyA) (stratB : StrategyB),
  ∃ (optimalA : StrategyA) (optimalB : StrategyB),
  countDistinctRows (playGame optimalA stratB) ≥ 2^50 ∧
  countDistinctRows (playGame stratA optimalB) ≤ 2^50 := by sorry

/-- The final theorem stating the exact number of distinct rows -/
theorem exact_distinct_rows :
  ∃ (optimalA : StrategyA) (optimalB : StrategyB),
  countDistinctRows (playGame optimalA optimalB) = 2^50 := by sorry

end NUMINAMATH_CALUDE_optimal_game_result_exact_distinct_rows_l3634_363405


namespace NUMINAMATH_CALUDE_fraction_simplification_l3634_363430

theorem fraction_simplification 
  (b c d x y z : ℝ) :
  (c * x * (b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3) + 
   d * z * (b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3)) / 
  (c * x + d * z) = 
  b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3634_363430


namespace NUMINAMATH_CALUDE_power_boat_travel_time_l3634_363451

/-- Represents the scenario of a power boat and raft traveling on a river --/
structure RiverJourney where
  r : ℝ  -- Speed of the river current (km/h)
  p : ℝ  -- Speed of the power boat relative to the river (km/h)
  t : ℝ  -- Time for power boat to travel from A to B (hours)
  s : ℝ  -- Stopping time at dock B (hours)

/-- The theorem stating that the time for the power boat to travel from A to B is 5 hours --/
theorem power_boat_travel_time 
  (journey : RiverJourney) 
  (h1 : journey.r > 0)  -- River speed is positive
  (h2 : journey.p > journey.r)  -- Power boat is faster than river current
  (h3 : journey.s = 1)  -- Stopping time is 1 hour
  (h4 : (journey.p + journey.r) * journey.t + (journey.p - journey.r) * (12 - journey.t - journey.s) = 12 * journey.r)  -- Distance equation
  : journey.t = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_boat_travel_time_l3634_363451


namespace NUMINAMATH_CALUDE_C_power_100_l3634_363424

def C : Matrix (Fin 2) (Fin 2) ℝ := !![5, -1; 12, 3]

theorem C_power_100 : 
  C^100 = (3^99 : ℝ) • !![1, 100; 6000, -200] := by sorry

end NUMINAMATH_CALUDE_C_power_100_l3634_363424


namespace NUMINAMATH_CALUDE_ball_returns_to_bella_l3634_363417

/-- Represents the number of girls in the circle -/
def n : ℕ := 13

/-- Represents the number of positions to move in each throw -/
def k : ℕ := 6

/-- Represents the position after a certain number of throws -/
def position (throws : ℕ) : ℕ :=
  (1 + throws * k) % n

/-- Theorem: The ball returns to Bella after exactly 13 throws -/
theorem ball_returns_to_bella :
  position 13 = 1 ∧ ∀ m : ℕ, m < 13 → position m ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_ball_returns_to_bella_l3634_363417


namespace NUMINAMATH_CALUDE_factorization_problem1_l3634_363470

theorem factorization_problem1 (m a : ℝ) : m * (a - 3) + 2 * (3 - a) = (a - 3) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem1_l3634_363470


namespace NUMINAMATH_CALUDE_cylinder_ellipse_eccentricity_l3634_363489

/-- The eccentricity of an ellipse formed by intersecting a cylinder with a plane -/
theorem cylinder_ellipse_eccentricity (d : ℝ) (θ : ℝ) (h_d : d = 12) (h_θ : θ = π / 6) :
  let r := d / 2
  let b := r
  let a := r / Real.cos θ
  let c := Real.sqrt (a^2 - b^2)
  c / a = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cylinder_ellipse_eccentricity_l3634_363489


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3634_363413

theorem divisibility_by_five (n : ℕ) : (76 * n^5 + 115 * n^4 + 19 * n) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l3634_363413


namespace NUMINAMATH_CALUDE_locker_labeling_cost_l3634_363467

/-- Calculates the cost of labeling lockers given the number of lockers and cost per digit -/
def labelingCost (numLockers : ℕ) (costPerDigit : ℚ) : ℚ :=
  let singleDigitCost := (min numLockers 9 : ℕ) * costPerDigit
  let doubleDigitCost := (min (numLockers - 9) 90 : ℕ) * 2 * costPerDigit
  let tripleDigitCost := (min (numLockers - 99) 900 : ℕ) * 3 * costPerDigit
  let quadrupleDigitCost := (max (numLockers - 999) 0 : ℕ) * 4 * costPerDigit
  singleDigitCost + doubleDigitCost + tripleDigitCost + quadrupleDigitCost

theorem locker_labeling_cost :
  labelingCost 2999 (3 / 100) = 32667 / 100 :=
by sorry

end NUMINAMATH_CALUDE_locker_labeling_cost_l3634_363467


namespace NUMINAMATH_CALUDE_photo_framing_yards_l3634_363423

/-- Calculates the minimum number of linear yards of framing needed for an enlarged photo with border. -/
def min_framing_yards (original_width : ℕ) (original_height : ℕ) (enlarge_factor : ℕ) (border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (framed_width + framed_height)
  let yards_needed := (perimeter_inches + 35) / 36  -- Ceiling division
  yards_needed

/-- Theorem stating that for a 5x7 inch photo enlarged 4 times with a 3-inch border,
    the minimum number of linear yards of framing needed is 4. -/
theorem photo_framing_yards :
  min_framing_yards 5 7 4 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_photo_framing_yards_l3634_363423


namespace NUMINAMATH_CALUDE_abs_neg_seven_l3634_363446

theorem abs_neg_seven : |(-7 : ℤ)| = 7 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_seven_l3634_363446


namespace NUMINAMATH_CALUDE_inequality_range_l3634_363466

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) ↔ (a ≤ -1 ∨ a ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3634_363466


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3634_363486

theorem absolute_value_inequality (x : ℝ) :
  |2 * x + 1| < 3 ↔ -2 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3634_363486


namespace NUMINAMATH_CALUDE_base_height_proof_l3634_363415

/-- Represents the height of an object in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ
  h_valid_inches : inches < 12

/-- Converts a Height to feet -/
def heightToFeet (h : Height) : ℚ :=
  h.feet + h.inches / 12

theorem base_height_proof (sculpture_height : Height) 
    (h_sculpture_height : sculpture_height = ⟨2, 10, by norm_num⟩) 
    (combined_height : ℚ) 
    (h_combined_height : combined_height = 3) :
  let base_height := combined_height - heightToFeet sculpture_height
  base_height * 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_height_proof_l3634_363415


namespace NUMINAMATH_CALUDE_unmarked_trees_l3634_363434

def total_trees : ℕ := 200
def mark_interval_out : ℕ := 5
def mark_interval_back : ℕ := 8

theorem unmarked_trees :
  let marked_out := total_trees / mark_interval_out
  let marked_back := total_trees / mark_interval_back
  let overlap := total_trees / (mark_interval_out * mark_interval_back)
  let total_marked := marked_out + marked_back - overlap
  total_trees - total_marked = 140 := by
  sorry

end NUMINAMATH_CALUDE_unmarked_trees_l3634_363434


namespace NUMINAMATH_CALUDE_zero_discriminant_implies_geometric_progression_l3634_363449

-- Define the quadratic equation ax^2 - 6bx + 9c = 0
def quadratic_equation (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 - 6 * b * x + 9 * c = 0

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℝ) : ℝ :=
  36 * b^2 - 36 * a * c

-- Define a geometric progression
def is_geometric_progression (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Theorem statement
theorem zero_discriminant_implies_geometric_progression
  (a b c : ℝ) (h : discriminant a b c = 0) :
  is_geometric_progression a b c := by
sorry

end NUMINAMATH_CALUDE_zero_discriminant_implies_geometric_progression_l3634_363449


namespace NUMINAMATH_CALUDE_correct_value_proof_l3634_363465

theorem correct_value_proof (n : ℕ) (initial_mean correct_mean wrong_value : ℚ) 
  (h1 : n = 30)
  (h2 : initial_mean = 250)
  (h3 : correct_mean = 251)
  (h4 : wrong_value = 135) :
  ∃ (correct_value : ℚ),
    correct_value = 165 ∧
    n * correct_mean = n * initial_mean - wrong_value + correct_value :=
by
  sorry

end NUMINAMATH_CALUDE_correct_value_proof_l3634_363465


namespace NUMINAMATH_CALUDE_headphones_savings_visits_l3634_363499

/-- The cost of the headphones in rubles -/
def headphones_cost : ℕ := 275

/-- The cost of a combined pool and sauna visit in rubles -/
def combined_cost : ℕ := 250

/-- The difference between pool-only cost and sauna-only cost in rubles -/
def pool_sauna_diff : ℕ := 200

/-- Calculates the cost of a pool-only visit -/
def pool_only_cost : ℕ := combined_cost - (combined_cost - pool_sauna_diff) / 2

/-- Calculates the savings per visit when choosing pool-only instead of combined -/
def savings_per_visit : ℕ := combined_cost - pool_only_cost

/-- The number of pool-only visits needed to save enough for the headphones -/
def visits_needed : ℕ := (headphones_cost + savings_per_visit - 1) / savings_per_visit

theorem headphones_savings_visits : visits_needed = 11 := by
  sorry

#eval visits_needed

end NUMINAMATH_CALUDE_headphones_savings_visits_l3634_363499


namespace NUMINAMATH_CALUDE_sandy_change_is_13_5_l3634_363439

/-- Represents the prices and quantities of drinks in Sandy's order -/
structure DrinkOrder where
  cappuccino_price : ℝ
  iced_tea_price : ℝ
  cafe_latte_price : ℝ
  espresso_price : ℝ
  mocha_price : ℝ
  hot_chocolate_price : ℝ
  cappuccino_qty : ℕ
  iced_tea_qty : ℕ
  cafe_latte_qty : ℕ
  espresso_qty : ℕ
  mocha_qty : ℕ
  hot_chocolate_qty : ℕ

/-- Calculates the total cost of the drink order -/
def total_cost (order : DrinkOrder) : ℝ :=
  order.cappuccino_price * order.cappuccino_qty +
  order.iced_tea_price * order.iced_tea_qty +
  order.cafe_latte_price * order.cafe_latte_qty +
  order.espresso_price * order.espresso_qty +
  order.mocha_price * order.mocha_qty +
  order.hot_chocolate_price * order.hot_chocolate_qty

/-- Calculates the change from a given payment amount -/
def calculate_change (payment : ℝ) (order : DrinkOrder) : ℝ :=
  payment - total_cost order

/-- Theorem stating that Sandy's change is $13.5 -/
theorem sandy_change_is_13_5 :
  let sandy_order : DrinkOrder := {
    cappuccino_price := 2,
    iced_tea_price := 3,
    cafe_latte_price := 1.5,
    espresso_price := 1,
    mocha_price := 2.5,
    hot_chocolate_price := 2,
    cappuccino_qty := 4,
    iced_tea_qty := 3,
    cafe_latte_qty := 5,
    espresso_qty := 3,
    mocha_qty := 2,
    hot_chocolate_qty := 2
  }
  calculate_change 50 sandy_order = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_sandy_change_is_13_5_l3634_363439


namespace NUMINAMATH_CALUDE_meet_once_l3634_363493

/-- Represents the movement of Michael and the garbage truck --/
structure Movement where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def number_of_meetings (m : Movement) : ℕ :=
  sorry

/-- The specific movement scenario described in the problem --/
def problem_scenario : Movement where
  michael_speed := 6
  truck_speed := 12
  pail_distance := 300
  truck_stop_time := 20
  initial_distance := 300

/-- Theorem stating that Michael and the truck meet exactly once --/
theorem meet_once : number_of_meetings problem_scenario = 1 := by
  sorry

end NUMINAMATH_CALUDE_meet_once_l3634_363493


namespace NUMINAMATH_CALUDE_fifth_term_is_five_l3634_363456

def fibonacci_like_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci_like_sequence n + fibonacci_like_sequence (n + 1)

theorem fifth_term_is_five :
  fibonacci_like_sequence 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_five_l3634_363456


namespace NUMINAMATH_CALUDE_sum_of_sequences_is_300_l3634_363478

def sequence1 : List ℕ := [2, 13, 24, 35, 46]
def sequence2 : List ℕ := [4, 15, 26, 37, 48]

theorem sum_of_sequences_is_300 : 
  (sequence1.sum + sequence2.sum) = 300 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_is_300_l3634_363478


namespace NUMINAMATH_CALUDE_translation_theorem_l3634_363461

/-- The original function -/
def f (x : ℝ) : ℝ := -(x - 1)^2 + 4

/-- The translated function -/
def g (x : ℝ) : ℝ := -(x + 1)^2 + 1

/-- Translation parameters -/
def left_shift : ℝ := 2
def down_shift : ℝ := 3

theorem translation_theorem :
  ∀ x : ℝ, g x = f (x + left_shift) - down_shift := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l3634_363461


namespace NUMINAMATH_CALUDE_additive_implies_linear_l3634_363487

/-- A function satisfying the given additive property -/
def AdditiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

/-- A linear function with zero intercept -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x, f x = k * x

/-- If a function satisfies the additive property, then it is a linear function with zero intercept -/
theorem additive_implies_linear (f : ℝ → ℝ) (h : AdditiveFunction f) : LinearFunction f := by
  sorry

end NUMINAMATH_CALUDE_additive_implies_linear_l3634_363487


namespace NUMINAMATH_CALUDE_composite_has_at_least_three_factors_l3634_363418

/-- A positive integer is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ k ∣ n

/-- The number of factors of a natural number. -/
def numFactors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- Theorem: A composite number has at least three factors. -/
theorem composite_has_at_least_three_factors (n : ℕ) (h : IsComposite n) :
    3 ≤ numFactors n := by
  sorry

end NUMINAMATH_CALUDE_composite_has_at_least_three_factors_l3634_363418


namespace NUMINAMATH_CALUDE_papayas_needed_l3634_363441

/-- The number of papayas Jake can eat in a week -/
def jake_weekly : ℕ := 3

/-- The number of papayas Jake's brother can eat in a week -/
def brother_weekly : ℕ := 5

/-- The number of papayas Jake's father can eat in a week -/
def father_weekly : ℕ := 4

/-- The number of weeks to account for -/
def num_weeks : ℕ := 4

/-- The total number of papayas needed for the given number of weeks -/
def total_papayas : ℕ := (jake_weekly + brother_weekly + father_weekly) * num_weeks

theorem papayas_needed : total_papayas = 48 := by
  sorry

end NUMINAMATH_CALUDE_papayas_needed_l3634_363441


namespace NUMINAMATH_CALUDE_line_equation_l3634_363458

/-- Given a line L with slope -3 and y-intercept 7, its equation is y = -3x + 7 -/
theorem line_equation (L : Set (ℝ × ℝ)) (slope : ℝ) (y_intercept : ℝ)
  (h1 : slope = -3)
  (h2 : y_intercept = 7)
  (h3 : ∀ (x y : ℝ), (x, y) ∈ L ↔ y = slope * x + y_intercept) :
  ∀ (x y : ℝ), (x, y) ∈ L ↔ y = -3 * x + 7 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l3634_363458


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3634_363485

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 0 ∧ d ≤ 9 ∧ d % 3 ≠ 0 ∧ d % 7 ≠ 0

def has_valid_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_valid_digit d

theorem no_integer_solutions (p : ℕ) (hp : Prime p) (hp_gt : p > 5) (hp_digits : has_valid_digits p) :
  ¬∃ (x y : ℤ), x^4 + p = 3 * y^4 :=
sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3634_363485


namespace NUMINAMATH_CALUDE_total_fuel_consumption_l3634_363427

-- Define fuel consumption rates
def highway_consumption_60 : ℝ := 3
def highway_consumption_70 : ℝ := 3.5
def city_consumption_30 : ℝ := 5
def city_consumption_15 : ℝ := 4.5

-- Define driving durations and speeds
def day1_highway_60 : ℝ := 2
def day1_highway_70 : ℝ := 1
def day1_city_30 : ℝ := 4

def day2_highway_70 : ℝ := 3
def day2_city_15 : ℝ := 3
def day2_city_30 : ℝ := 1

def day3_highway_60 : ℝ := 1.5
def day3_city_30 : ℝ := 3
def day3_city_15 : ℝ := 1

-- Theorem statement
theorem total_fuel_consumption :
  let day1 := day1_highway_60 * 60 * highway_consumption_60 +
              day1_highway_70 * 70 * highway_consumption_70 +
              day1_city_30 * 30 * city_consumption_30
  let day2 := day2_highway_70 * 70 * highway_consumption_70 +
              day2_city_15 * 15 * city_consumption_15 +
              day2_city_30 * 30 * city_consumption_30
  let day3 := day3_highway_60 * 60 * highway_consumption_60 +
              day3_city_30 * 30 * city_consumption_30 +
              day3_city_15 * 15 * city_consumption_15
  day1 + day2 + day3 = 3080 := by
  sorry

end NUMINAMATH_CALUDE_total_fuel_consumption_l3634_363427


namespace NUMINAMATH_CALUDE_not_both_odd_l3634_363425

theorem not_both_odd (m n : ℕ) (h : (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 2020) :
  Even m ∨ Even n :=
sorry

end NUMINAMATH_CALUDE_not_both_odd_l3634_363425


namespace NUMINAMATH_CALUDE_solution_for_F_l3634_363496

/-- Definition of function F --/
def F (a b c : ℝ) : ℝ := a * b^2 - c

/-- Theorem stating that 1/6 is the solution to F(a,5,10) = F(a,7,14) --/
theorem solution_for_F : ∃ a : ℝ, F a 5 10 = F a 7 14 ∧ a = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_solution_for_F_l3634_363496


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l3634_363475

/-- Represents the probability of drawing specific colored balls from an urn --/
def draw_probability (red white green total : ℕ) : ℚ :=
  (red : ℚ) / total * (white : ℚ) / (total - 1) * (green : ℚ) / (total - 2)

/-- Represents the probability of drawing specific colored balls in any order --/
def draw_probability_any_order (red white green total : ℕ) : ℚ :=
  6 * draw_probability red white green total

theorem urn_probability_theorem (red white green : ℕ) 
  (h_red : red = 15) (h_white : white = 9) (h_green : green = 4) :
  let total := red + white + green
  draw_probability red white green total = 5 / 182 ∧
  draw_probability_any_order red white green total = 15 / 91 := by
  sorry


end NUMINAMATH_CALUDE_urn_probability_theorem_l3634_363475


namespace NUMINAMATH_CALUDE_shooter_probability_l3634_363428

theorem shooter_probability (p10 p9 p8 : ℝ) (h1 : p10 = 0.24) (h2 : p9 = 0.28) (h3 : p8 = 0.19) :
  1 - p10 - p9 = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l3634_363428


namespace NUMINAMATH_CALUDE_min_value_of_function_max_sum_with_constraint_l3634_363421

-- Part 1
theorem min_value_of_function (x : ℝ) (h : x > -1) :
  ∃ (min_y : ℝ), min_y = 9 ∧ ∀ y, y = (x^2 + 7*x + 10) / (x + 1) → y ≥ min_y :=
sorry

-- Part 2
theorem max_sum_with_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  x + y ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_max_sum_with_constraint_l3634_363421


namespace NUMINAMATH_CALUDE_factorize_expression_1_l3634_363411

theorem factorize_expression_1 (x y : ℝ) :
  x^2*y - 4*x*y + 4*y = y*(x-2)^2 := by sorry

end NUMINAMATH_CALUDE_factorize_expression_1_l3634_363411


namespace NUMINAMATH_CALUDE_school_home_time_ratio_l3634_363460

/-- Represents the road segments in Xiaoming's journey --/
inductive RoadSegment
| Flat
| Uphill
| Downhill

/-- Represents the direction of Xiaoming's journey --/
inductive Direction
| ToSchool
| ToHome

/-- Calculates the time taken for a segment of the journey --/
def segmentTime (segment : RoadSegment) (direction : Direction) : ℚ :=
  match segment, direction with
  | RoadSegment.Flat, _ => 1 / 3
  | RoadSegment.Uphill, Direction.ToSchool => 1
  | RoadSegment.Uphill, Direction.ToHome => 1
  | RoadSegment.Downhill, Direction.ToSchool => 1 / 4
  | RoadSegment.Downhill, Direction.ToHome => 1 / 2

/-- Calculates the total time for a journey in a given direction --/
def journeyTime (direction : Direction) : ℚ :=
  segmentTime RoadSegment.Flat direction +
  2 * segmentTime RoadSegment.Uphill direction +
  segmentTime RoadSegment.Downhill direction

/-- Main theorem: The ratio of time to school vs time to home is 19:16 --/
theorem school_home_time_ratio :
  (journeyTime Direction.ToSchool) / (journeyTime Direction.ToHome) = 19 / 16 := by
  sorry


end NUMINAMATH_CALUDE_school_home_time_ratio_l3634_363460


namespace NUMINAMATH_CALUDE_fifth_term_constant_binomial_l3634_363426

theorem fifth_term_constant_binomial (n : ℕ) : 
  (∃ k : ℝ, k ≠ 0 ∧ (Nat.choose n 4) * (-2)^4 * k = (Nat.choose n 4) * (-2)^4) → 
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_constant_binomial_l3634_363426


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3634_363464

theorem repeating_decimal_to_fraction :
  ∀ (a b : ℕ) (x : ℚ),
    (x = 0.4 + (31 : ℚ) / (990 : ℚ)) →
    (x = (427 : ℚ) / (990 : ℚ)) :=
by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3634_363464


namespace NUMINAMATH_CALUDE_tangent_slope_at_2_10_l3634_363459

/-- The slope of the tangent line to y = x^2 + 3x at (2, 10) is 7 -/
theorem tangent_slope_at_2_10 : 
  let f (x : ℝ) := x^2 + 3*x
  let A : ℝ × ℝ := (2, 10)
  let slope := (deriv f) A.1
  slope = 7 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_2_10_l3634_363459


namespace NUMINAMATH_CALUDE_sum_of_complex_roots_of_unity_l3634_363476

open Complex

theorem sum_of_complex_roots_of_unity : 
  let ω : ℂ := exp (Complex.I * Real.pi / 11)
  (ω + ω^3 + ω^5 + ω^7 + ω^9 + ω^11 + ω^13 + ω^15 + ω^17 + ω^19 + ω^21) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_roots_of_unity_l3634_363476


namespace NUMINAMATH_CALUDE_car_travel_distance_l3634_363408

def initial_distance : ℝ := 192
def initial_gallons : ℝ := 6
def efficiency_increase : ℝ := 0.1
def new_gallons : ℝ := 8

theorem car_travel_distance : 
  let initial_mpg := initial_distance / initial_gallons
  let new_mpg := initial_mpg * (1 + efficiency_increase)
  new_mpg * new_gallons = 281.6 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_distance_l3634_363408


namespace NUMINAMATH_CALUDE_distribute_negative_three_l3634_363454

theorem distribute_negative_three (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_three_l3634_363454


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3634_363482

/-- The equation of the tangent line to y = xe^(x-1) at (1, 1) is y = 2x - 1 -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = x * Real.exp (x - 1)) → -- Curve equation
  (1 = 1 * Real.exp (1 - 1)) → -- Point (1, 1) satisfies the curve equation
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ∧ 
    (y - 1 = m * (x - 1)) ∧   -- Point-slope form of tangent line
    (m = (1 + 1) * Real.exp (1 - 1)) ∧ -- Slope at x = 1
    (y = 2 * x - 1)) -- Equation of the tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3634_363482
