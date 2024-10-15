import Mathlib

namespace NUMINAMATH_CALUDE_cos_sum_equals_one_l2872_287247

theorem cos_sum_equals_one (α β : Real) 
  (h : (Real.cos α * Real.cos (β/2)) / Real.cos (α - β/2) + 
       (Real.cos β * Real.cos (α/2)) / Real.cos (β - α/2) = 1) : 
  Real.cos α + Real.cos β = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_equals_one_l2872_287247


namespace NUMINAMATH_CALUDE_glasses_cost_l2872_287244

theorem glasses_cost (frame_cost : ℝ) (coupon : ℝ) (insurance_coverage : ℝ) (total_cost : ℝ) :
  frame_cost = 200 →
  coupon = 50 →
  insurance_coverage = 0.8 →
  total_cost = 250 →
  ∃ (lens_cost : ℝ), lens_cost = 500 ∧
    total_cost = (frame_cost - coupon) + (1 - insurance_coverage) * lens_cost :=
by sorry

end NUMINAMATH_CALUDE_glasses_cost_l2872_287244


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2872_287210

/-- Given a geometric sequence with first term √3 and second term 3√3, 
    the seventh term is 729√3 -/
theorem seventh_term_of_geometric_sequence 
  (a₁ : ℝ) 
  (a₂ : ℝ) 
  (h₁ : a₁ = Real.sqrt 3)
  (h₂ : a₂ = 3 * Real.sqrt 3) :
  (a₁ * (a₂ / a₁)^6 : ℝ) = 729 * Real.sqrt 3 := by
  sorry

#check seventh_term_of_geometric_sequence

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2872_287210


namespace NUMINAMATH_CALUDE_marias_number_problem_l2872_287256

theorem marias_number_problem (x : ℝ) : 
  (((x - 3) * 3 + 3) / 3 = 10) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_marias_number_problem_l2872_287256


namespace NUMINAMATH_CALUDE_yard_length_with_11_trees_l2872_287252

/-- The length of a yard with equally spaced trees -/
def yardLength (numTrees : ℕ) (distanceBetweenTrees : ℝ) : ℝ :=
  (numTrees - 1) * distanceBetweenTrees

/-- Theorem: The length of a yard with 11 equally spaced trees, 
    with 15 meters between consecutive trees, is 150 meters -/
theorem yard_length_with_11_trees : 
  yardLength 11 15 = 150 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_with_11_trees_l2872_287252


namespace NUMINAMATH_CALUDE_prob_king_then_ten_l2872_287209

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def KingsInDeck : ℕ := 4

/-- Number of 10s in a standard deck -/
def TensInDeck : ℕ := 4

/-- Probability of drawing a King first and then a 10 from a standard deck -/
theorem prob_king_then_ten (deck : ℕ) (kings : ℕ) (tens : ℕ) :
  deck = StandardDeck → kings = KingsInDeck → tens = TensInDeck →
  (kings : ℚ) / deck * tens / (deck - 1) = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_then_ten_l2872_287209


namespace NUMINAMATH_CALUDE_min_value_theorem_l2872_287254

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) :
  (1 / x + 1 / (3 * y)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2872_287254


namespace NUMINAMATH_CALUDE_z3_magnitude_range_l2872_287279

/-- Given complex numbers satisfying certain conditions, prove the range of the magnitude of z₃ -/
theorem z3_magnitude_range (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ = Real.sqrt 2)
  (h2 : Complex.abs z₂ = Real.sqrt 2)
  (h3 : (z₁.re * z₂.re + z₁.im * z₂.im) = 0)
  (h4 : Complex.abs (z₁ + z₂ - z₃) = 2) :
  0 ≤ Complex.abs z₃ ∧ Complex.abs z₃ ≤ 4 := by sorry

end NUMINAMATH_CALUDE_z3_magnitude_range_l2872_287279


namespace NUMINAMATH_CALUDE_waiter_customers_l2872_287229

/-- Calculates the final number of customers for a waiter given the initial number,
    the number who left, and the number of new customers. -/
def final_customers (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem stating that for the given scenario, the final number of customers is 28. -/
theorem waiter_customers : final_customers 33 31 26 = 28 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l2872_287229


namespace NUMINAMATH_CALUDE_probability_different_classes_l2872_287246

/-- The probability of selecting two students from different language classes -/
theorem probability_different_classes (total : ℕ) (german : ℕ) (chinese : ℕ) 
  (h1 : total = 30)
  (h2 : german = 22)
  (h3 : chinese = 19)
  (h4 : german + chinese - total ≥ 0) : 
  (Nat.choose total 2 - (Nat.choose (german + chinese - total) 2 + 
   Nat.choose (german - (german + chinese - total)) 2 + 
   Nat.choose (chinese - (german + chinese - total)) 2)) / Nat.choose total 2 = 352 / 435 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_classes_l2872_287246


namespace NUMINAMATH_CALUDE_remaining_area_is_27_l2872_287221

/-- Represents the square grid --/
def Grid := Fin 6 → Fin 6 → Bool

/-- The area of a single cell in square centimeters --/
def cellArea : ℝ := 1

/-- The total area of the square in square centimeters --/
def totalArea : ℝ := 36

/-- The area of the dark grey triangles in square centimeters --/
def darkGreyArea : ℝ := 3

/-- The area of the light grey triangles in square centimeters --/
def lightGreyArea : ℝ := 6

/-- The total area of removed triangles in square centimeters --/
def removedArea : ℝ := darkGreyArea + lightGreyArea

/-- Theorem: The area of the remaining shape after cutting out triangles is 27 square cm --/
theorem remaining_area_is_27 : totalArea - removedArea = 27 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_is_27_l2872_287221


namespace NUMINAMATH_CALUDE_alice_baked_five_more_l2872_287243

/-- The number of additional chocolate chip cookies Alice baked after the accident -/
def additional_cookies (alice_initial bob_initial thrown_away bob_additional final_count : ℕ) : ℕ :=
  final_count - (alice_initial + bob_initial - thrown_away + bob_additional)

/-- Theorem stating that Alice baked 5 more chocolate chip cookies after the accident -/
theorem alice_baked_five_more : additional_cookies 74 7 29 36 93 = 5 := by
  sorry

end NUMINAMATH_CALUDE_alice_baked_five_more_l2872_287243


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2872_287204

theorem complex_modulus_problem (x y : ℝ) (h : Complex.I * Complex.mk x y = Complex.mk 3 4) :
  Complex.abs (Complex.mk x y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2872_287204


namespace NUMINAMATH_CALUDE_system_solution_l2872_287207

theorem system_solution (x y : ℝ) (hx : x = 4) (hy : y = -1) : x - 2*y = 6 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2872_287207


namespace NUMINAMATH_CALUDE_min_draws_for_twenty_l2872_287266

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to guarantee at least n of a single color -/
def minDraws (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The actual ball counts in the problem -/
def problemCounts : BallCounts :=
  { red := 35, green := 25, yellow := 22, blue := 15, white := 12, black := 10 }

/-- The theorem to be proved -/
theorem min_draws_for_twenty (counts : BallCounts) :
  counts = problemCounts → minDraws counts 20 = 95 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_twenty_l2872_287266


namespace NUMINAMATH_CALUDE_f_one_geq_25_l2872_287257

def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem f_one_geq_25 (m : ℝ) (h : ∀ x ≥ -2, Monotone (f m)) :
  f m 1 ≥ 25 := by sorry

end NUMINAMATH_CALUDE_f_one_geq_25_l2872_287257


namespace NUMINAMATH_CALUDE_factorial_equation_l2872_287201

theorem factorial_equation : (Nat.factorial 6 - Nat.factorial 4) / Nat.factorial 5 = 29/5 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_l2872_287201


namespace NUMINAMATH_CALUDE_ramanujan_number_l2872_287291

theorem ramanujan_number (r h : ℂ) : 
  r * h = 50 - 14 * I →
  h = 7 + 2 * I →
  r = 6 - (198 / 53) * I := by
sorry

end NUMINAMATH_CALUDE_ramanujan_number_l2872_287291


namespace NUMINAMATH_CALUDE_property_square_footage_l2872_287271

/-- Given a property worth $333,200 and a price of $98 per square foot,
    prove that the total square footage is 3400 square feet. -/
theorem property_square_footage :
  let property_value : ℕ := 333200
  let price_per_sqft : ℕ := 98
  let total_sqft : ℕ := property_value / price_per_sqft
  total_sqft = 3400 := by
  sorry

end NUMINAMATH_CALUDE_property_square_footage_l2872_287271


namespace NUMINAMATH_CALUDE_min_a_value_l2872_287231

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l2872_287231


namespace NUMINAMATH_CALUDE_square_difference_formula_l2872_287269

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8 / 15) (h2 : x - y = 1 / 45) : 
  x^2 - y^2 = 8 / 675 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l2872_287269


namespace NUMINAMATH_CALUDE_principal_is_800_l2872_287224

/-- Calculates the principal amount given the simple interest rate, final amount, and time period. -/
def calculate_principal (rate : ℚ) (final_amount : ℚ) (time : ℕ) : ℚ :=
  (final_amount * 100) / (rate * time)

/-- Theorem stating that the principal amount is 800 given the specified conditions. -/
theorem principal_is_800 (rate : ℚ) (final_amount : ℚ) (time : ℕ) 
  (h_rate : rate = 25/400)  -- 6.25% as a rational number
  (h_final_amount : final_amount = 200)
  (h_time : time = 4) :
  calculate_principal rate final_amount time = 800 := by
  sorry

#eval calculate_principal (25/400) 200 4  -- This should evaluate to 800

end NUMINAMATH_CALUDE_principal_is_800_l2872_287224


namespace NUMINAMATH_CALUDE_original_number_proof_l2872_287248

theorem original_number_proof (x : ℝ) : 
  (x * 1.125 - x * 0.75 = 30) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2872_287248


namespace NUMINAMATH_CALUDE_min_correct_answers_is_17_l2872_287282

/-- AMC 12 scoring system and Sarah's strategy -/
structure AMC12 where
  total_questions : Nat
  attempted_questions : Nat
  points_correct : Nat
  points_incorrect : Nat
  points_unanswered : Nat
  min_score : Nat

/-- Calculate the minimum number of correct answers needed -/
def min_correct_answers (amc : AMC12) : Nat :=
  let unanswered := amc.total_questions - amc.attempted_questions
  let points_from_unanswered := unanswered * amc.points_unanswered
  let required_points := amc.min_score - points_from_unanswered
  (required_points + amc.points_correct - 1) / amc.points_correct

/-- Theorem stating the minimum number of correct answers needed -/
theorem min_correct_answers_is_17 (amc : AMC12) 
  (h1 : amc.total_questions = 30)
  (h2 : amc.attempted_questions = 24)
  (h3 : amc.points_correct = 7)
  (h4 : amc.points_incorrect = 0)
  (h5 : amc.points_unanswered = 2)
  (h6 : amc.min_score = 130) : 
  min_correct_answers amc = 17 := by
  sorry

#eval min_correct_answers {
  total_questions := 30,
  attempted_questions := 24,
  points_correct := 7,
  points_incorrect := 0,
  points_unanswered := 2,
  min_score := 130
}

end NUMINAMATH_CALUDE_min_correct_answers_is_17_l2872_287282


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2872_287214

-- Define the function f(x) = -x^2 + bx + c
def f (b c x : ℝ) : ℝ := -x^2 + b*x + c

-- Theorem statement
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  (f b c 0 = -3) →
  (f b c (-6) = -3) →
  (b = -6 ∧ c = -3) ∧
  (∀ x : ℝ, -4 ≤ x ∧ x ≤ 0 → f b c x ≤ 6) ∧
  (∃ x : ℝ, -4 ≤ x ∧ x ≤ 0 ∧ f b c x = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2872_287214


namespace NUMINAMATH_CALUDE_apartment_fraction_sum_l2872_287215

theorem apartment_fraction_sum : 
  let one_bedroom : ℝ := 0.12
  let two_bedroom : ℝ := 0.26
  let three_bedroom : ℝ := 0.38
  let four_bedroom : ℝ := 0.24
  one_bedroom + two_bedroom + three_bedroom = 0.76 :=
by sorry

end NUMINAMATH_CALUDE_apartment_fraction_sum_l2872_287215


namespace NUMINAMATH_CALUDE_average_equation_solution_l2872_287203

theorem average_equation_solution (x : ℝ) : 
  ((x + 3) + (4 * x + 1) + (3 * x + 6)) / 3 = 3 * x - 8 → x = 34 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l2872_287203


namespace NUMINAMATH_CALUDE_cubic_root_sum_square_l2872_287211

theorem cubic_root_sum_square (p q r t : ℝ) : 
  (p^3 - 6*p^2 + 8*p - 1 = 0) →
  (q^3 - 6*q^2 + 8*q - 1 = 0) →
  (r^3 - 6*r^2 + 8*r - 1 = 0) →
  (t = Real.sqrt p + Real.sqrt q + Real.sqrt r) →
  (t^4 - 12*t^2 - 8*t = -4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_square_l2872_287211


namespace NUMINAMATH_CALUDE_onion_bag_weight_l2872_287225

/-- Proves that the weight of each bag of onions is 50 kgs given the specified conditions -/
theorem onion_bag_weight 
  (bags_per_trip : ℕ) 
  (num_trips : ℕ) 
  (total_weight : ℕ) 
  (h1 : bags_per_trip = 10)
  (h2 : num_trips = 20)
  (h3 : total_weight = 10000) :
  total_weight / (bags_per_trip * num_trips) = 50 := by
  sorry

end NUMINAMATH_CALUDE_onion_bag_weight_l2872_287225


namespace NUMINAMATH_CALUDE_smallest_value_of_quadratic_l2872_287208

theorem smallest_value_of_quadratic :
  (∀ x : ℝ, x^2 + 6*x + 9 ≥ 0) ∧ (∃ x : ℝ, x^2 + 6*x + 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_of_quadratic_l2872_287208


namespace NUMINAMATH_CALUDE_sum_of_roots_l2872_287276

theorem sum_of_roots (a b : ℝ) : 
  a * (a - 4) = 5 → b * (b - 4) = 5 → a ≠ b → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2872_287276


namespace NUMINAMATH_CALUDE_completing_square_sum_l2872_287292

theorem completing_square_sum (a b c : ℤ) : 
  (∀ x, 36 * x^2 - 60 * x + 25 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 26 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_sum_l2872_287292


namespace NUMINAMATH_CALUDE_function_period_l2872_287227

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_period (f : ℝ → ℝ) (h : ∀ x, f (x + 3) = -f x) :
  is_periodic f 6 :=
sorry

end NUMINAMATH_CALUDE_function_period_l2872_287227


namespace NUMINAMATH_CALUDE_min_value_at_six_l2872_287236

/-- The quadratic function f(x) = x^2 - 12x + 36 -/
def f (x : ℝ) : ℝ := x^2 - 12*x + 36

/-- Theorem stating that f(x) has a minimum value when x = 6 -/
theorem min_value_at_six :
  ∀ x : ℝ, f x ≥ f 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_at_six_l2872_287236


namespace NUMINAMATH_CALUDE_y_minimum_value_and_interval_l2872_287245

def y (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem y_minimum_value_and_interval :
  (∃ (m : ℝ), ∀ (x : ℝ), y x ≥ m ∧ (∃ (x₀ : ℝ), y x₀ = m)) ∧
  (∀ (x : ℝ), y x = 2 ↔ -1 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_y_minimum_value_and_interval_l2872_287245


namespace NUMINAMATH_CALUDE_polygon_interior_exterior_angle_relation_l2872_287268

theorem polygon_interior_exterior_angle_relation (n : ℕ) : 
  (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_exterior_angle_relation_l2872_287268


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_4_solution_existence_condition_l2872_287240

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a|

-- Theorem for the first part of the problem
theorem solution_set_for_a_equals_4 :
  {x : ℝ | |2*x - 4| < 8 - |x - 1|} = Set.Ioo (-1) (13/3) := by sorry

-- Theorem for the second part of the problem
theorem solution_existence_condition (a : ℝ) :
  (∃ x, f a x > 8 + |2*x - 1|) ↔ (a > 9 ∨ a < -7) := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_4_solution_existence_condition_l2872_287240


namespace NUMINAMATH_CALUDE_income_distribution_l2872_287286

theorem income_distribution (total_income : ℝ) 
  (h_total : total_income = 100) 
  (food_percent : ℝ) (h_food : food_percent = 35)
  (education_percent : ℝ) (h_education : education_percent = 25)
  (rent_percent : ℝ) (h_rent : rent_percent = 80) : 
  (total_income - (food_percent + education_percent) * total_income / 100 - 
   rent_percent * (total_income - (food_percent + education_percent) * total_income / 100) / 100) / 
  total_income * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_income_distribution_l2872_287286


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l2872_287226

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  full_days : Nat        -- Number of days working 8 hours
  partial_days : Nat     -- Number of days working 6 hours
  weekly_earnings : Nat  -- Total earnings per week in dollars

/-- Calculate Sheila's hourly wage -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (8 * schedule.full_days + 6 * schedule.partial_days)

/-- Theorem: Sheila's hourly wage is $6 -/
theorem sheila_hourly_wage :
  let schedule : WorkSchedule := {
    full_days := 3,
    partial_days := 2,
    weekly_earnings := 216
  }
  hourly_wage schedule = 6 := by sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l2872_287226


namespace NUMINAMATH_CALUDE_harmonic_mean_4_5_10_l2872_287255

def harmonic_mean (a b c : ℚ) : ℚ := 3 / (1/a + 1/b + 1/c)

theorem harmonic_mean_4_5_10 :
  harmonic_mean 4 5 10 = 60 / 11 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_4_5_10_l2872_287255


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2872_287222

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a + b + c = 23) : 
  a*b + b*c + a*c = 131 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2872_287222


namespace NUMINAMATH_CALUDE_no_perfect_squares_in_range_l2872_287251

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

def base_n_value (n : ℕ) : ℕ :=
  n^3 + 2*n^2 + 3*n + 4

theorem no_perfect_squares_in_range : 
  ¬ ∃ n : ℕ, 5 ≤ n ∧ n ≤ 20 ∧ is_perfect_square (base_n_value n) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_in_range_l2872_287251


namespace NUMINAMATH_CALUDE_clairaut_general_solution_l2872_287278

/-- Clairaut's equation -/
def clairaut_equation (y : ℝ → ℝ) (x : ℝ) : Prop :=
  y x = x * (deriv y x) + 1 / (2 * deriv y x)

/-- General solution of Clairaut's equation -/
def is_general_solution (y : ℝ → ℝ) : Prop :=
  ∃ C : ℝ, (∀ x, y x = C * x + 1 / (2 * C)) ∨ (∀ x, (y x)^2 = 2 * x)

/-- Theorem: The general solution satisfies Clairaut's equation -/
theorem clairaut_general_solution :
  ∀ y : ℝ → ℝ, is_general_solution y → ∀ x : ℝ, clairaut_equation y x :=
sorry

end NUMINAMATH_CALUDE_clairaut_general_solution_l2872_287278


namespace NUMINAMATH_CALUDE_small_cube_edge_length_small_cube_edge_length_proof_l2872_287283

/-- Given a cube made of 8 smaller cubes with a total volume of 1000 cm³,
    the length of one edge of a smaller cube is 5 cm. -/
theorem small_cube_edge_length : ℝ :=
  let total_volume : ℝ := 1000
  let num_small_cubes : ℕ := 8
  let edge_ratio : ℝ := 2  -- ratio of large cube edge to small cube edge
  
  -- Define the volume of the large cube in terms of the small cube's edge length
  let large_cube_volume (small_edge : ℝ) : ℝ := (edge_ratio * small_edge) ^ 3
  
  -- Define the equation: large cube volume equals total volume
  let volume_equation (small_edge : ℝ) : Prop := large_cube_volume small_edge = total_volume
  
  -- The length of one edge of the smaller cube
  5

/-- Proof of the theorem -/
theorem small_cube_edge_length_proof : small_cube_edge_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_small_cube_edge_length_small_cube_edge_length_proof_l2872_287283


namespace NUMINAMATH_CALUDE_number_problem_l2872_287253

theorem number_problem (x : ℝ) : 0.65 * x = 0.8 * x - 21 → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2872_287253


namespace NUMINAMATH_CALUDE_intersection_distance_l2872_287212

/-- The distance between intersections of x = y³ and x + y² = 1 -/
theorem intersection_distance (a : ℝ) : 
  (a^4 + a^3 + a^2 - 1 = 0) →
  ∃ (u v p : ℝ), 
    (2 * Real.sqrt (a^6 + a^2) = Real.sqrt (u + v * Real.sqrt p)) ∧
    ((a^3)^2 + a^2)^2 = (((-a)^3)^2 + (-a)^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l2872_287212


namespace NUMINAMATH_CALUDE_jenny_spent_fraction_l2872_287267

theorem jenny_spent_fraction (initial_amount : ℚ) : 
  (initial_amount / 2 = 21) →
  (initial_amount - 24 > 0) →
  ((initial_amount - 24) / initial_amount = 3/7) := by
  sorry

end NUMINAMATH_CALUDE_jenny_spent_fraction_l2872_287267


namespace NUMINAMATH_CALUDE_max_a_for_monotonic_f_l2872_287241

/-- Given that f(x) = x^3 - ax is monotonically increasing on [1, +∞), 
    the maximum value of a is 3. -/
theorem max_a_for_monotonic_f (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → (x₁^3 - a*x₁) < (x₂^3 - a*x₂)) →
  a ≤ 3 ∧ ∀ b > a, ∃ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ (x₁^3 - b*x₁) ≥ (x₂^3 - b*x₂) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_monotonic_f_l2872_287241


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slope_range_l2872_287233

/-- The slope range for a line intersecting an ellipse --/
theorem line_ellipse_intersection_slope_range :
  ∀ m : ℝ,
  (∃ x y : ℝ, y = m * x + 7 ∧ 4 * x^2 + 25 * y^2 = 100) →
  -Real.sqrt (9/5) ≤ m ∧ m ≤ Real.sqrt (9/5) := by
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slope_range_l2872_287233


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2872_287275

theorem sqrt_meaningful_range (x : ℝ) : 3 * x - 6 ≥ 0 ↔ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2872_287275


namespace NUMINAMATH_CALUDE_min_value_at_angle_l2872_287219

def minimizing_angle (k : ℤ) : ℝ := 660 + 720 * k

theorem min_value_at_angle (A : ℝ) :
  (∃ k : ℤ, A = minimizing_angle k) ↔
  ∀ B : ℝ, Real.sin (A / 2) - Real.sqrt 3 * Real.cos (A / 2) ≤ 
           Real.sin (B / 2) - Real.sqrt 3 * Real.cos (B / 2) :=
by sorry

#check min_value_at_angle

end NUMINAMATH_CALUDE_min_value_at_angle_l2872_287219


namespace NUMINAMATH_CALUDE_solution_set_equivalence_a_range_l2872_287274

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x + x^2 + 1

-- Define the solution set of f(x) ≤ 0
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ 0}

-- Define the condition that g has two distinct zeros in (1,2)
def has_two_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ x₁ ≠ x₂

-- Theorem 1
theorem solution_set_equivalence (a : ℝ) :
  solution_set a = Set.Icc 1 2 →
  {x : ℝ | f a x ≥ 1 - x^2} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 1} := by sorry

-- Theorem 2
theorem a_range (a : ℝ) :
  has_two_zeros a → -5 < a ∧ a < -2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_a_range_l2872_287274


namespace NUMINAMATH_CALUDE_margaret_mean_score_l2872_287230

def scores : List ℕ := [84, 86, 90, 92, 93, 95, 97, 96, 99]

def cyprian_count : ℕ := 5
def margaret_count : ℕ := 4
def cyprian_mean : ℕ := 92

theorem margaret_mean_score :
  let total_sum := scores.sum
  let cyprian_sum := cyprian_count * cyprian_mean
  let margaret_sum := total_sum - cyprian_sum
  (margaret_sum : ℚ) / margaret_count = 93 := by sorry

end NUMINAMATH_CALUDE_margaret_mean_score_l2872_287230


namespace NUMINAMATH_CALUDE_berry_ratio_l2872_287270

theorem berry_ratio (total : ℕ) (blueberries : ℕ) : 
  total = 42 →
  blueberries = 7 →
  (total / 2 : ℚ) = (total - blueberries - (total / 2) : ℚ) →
  (total - blueberries - (total / 2) : ℚ) / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_berry_ratio_l2872_287270


namespace NUMINAMATH_CALUDE_freshmen_in_liberal_arts_l2872_287213

theorem freshmen_in_liberal_arts (total_students : ℝ) 
  (freshmen_percent : ℝ) 
  (psych_majors_percent : ℝ) 
  (freshmen_psych_liberal_arts_percent : ℝ) :
  freshmen_percent = 0.6 →
  psych_majors_percent = 0.2 →
  freshmen_psych_liberal_arts_percent = 0.048 →
  (freshmen_psych_liberal_arts_percent * total_students) / 
  (psych_majors_percent * freshmen_percent * total_students) = 0.4 := by
sorry

end NUMINAMATH_CALUDE_freshmen_in_liberal_arts_l2872_287213


namespace NUMINAMATH_CALUDE_smallest_b_for_divisibility_l2872_287263

def is_single_digit (n : ℕ) : Prop := n < 10

def is_divisible_by_13 (n : ℕ) : Prop := n % 13 = 0

theorem smallest_b_for_divisibility : 
  ∃ (B : ℕ), is_single_digit B ∧ 
             is_divisible_by_13 (200 + 10 * B + 5) ∧ 
             (∀ (k : ℕ), k < B → ¬(is_divisible_by_13 (200 + 10 * k + 5))) ∧
             B = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_for_divisibility_l2872_287263


namespace NUMINAMATH_CALUDE_negation_existence_statement_l2872_287298

theorem negation_existence_statement (A : Set ℝ) :
  (¬ ∃ x ∈ A, x^2 - 2*x - 3 > 0) ↔ (∀ x ∈ A, x^2 - 2*x - 3 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_existence_statement_l2872_287298


namespace NUMINAMATH_CALUDE_apples_to_eat_raw_l2872_287293

theorem apples_to_eat_raw (total : ℕ) (wormy : ℕ) (bruised : ℕ) : 
  total = 85 → 
  wormy = total / 5 →
  bruised = wormy + 9 →
  total - wormy - bruised = 42 := by
sorry

end NUMINAMATH_CALUDE_apples_to_eat_raw_l2872_287293


namespace NUMINAMATH_CALUDE_min_steps_parallel_line_l2872_287294

/-- A line in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A circle in a plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a construction step using either a line or a circle -/
inductive ConstructionStep
  | line : Line → ConstructionStep
  | circle : Circle → ConstructionStep

/-- Checks if a point is on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Checks if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- The main theorem stating that the minimum number of construction steps to create a parallel line is 3 -/
theorem min_steps_parallel_line 
  (a : Line) (O : Point) (h : ¬ O.onLine a) :
  ∃ (steps : List ConstructionStep) (l : Line),
    steps.length = 3 ∧
    O.onLine l ∧
    l.parallel a ∧
    (∀ (steps' : List ConstructionStep) (l' : Line),
      steps'.length < 3 →
      ¬(O.onLine l' ∧ l'.parallel a)) :=
sorry

end NUMINAMATH_CALUDE_min_steps_parallel_line_l2872_287294


namespace NUMINAMATH_CALUDE_zero_point_not_implies_product_negative_l2872_287299

-- Define a continuous function on a closed interval
def ContinuousOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ContinuousOn f (Set.Icc a b)

-- Define the existence of a zero point in an open interval
def HasZeroInOpenInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ f x = 0

-- Theorem statement
theorem zero_point_not_implies_product_negative
  (f : ℝ → ℝ) (a b : ℝ) (h1 : a < b) (h2 : ContinuousOnInterval f a b) :
  HasZeroInOpenInterval f a b → (f a) * (f b) < 0 → False :=
sorry

end NUMINAMATH_CALUDE_zero_point_not_implies_product_negative_l2872_287299


namespace NUMINAMATH_CALUDE_additional_toothpicks_for_extension_l2872_287234

/-- The number of toothpicks required for a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ :=
  if n ≤ 2 then 2 * n + 2 else 2 * n + (n - 1) * (n - 2)

theorem additional_toothpicks_for_extension :
  toothpicks 4 = 26 →
  toothpicks 6 - toothpicks 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_additional_toothpicks_for_extension_l2872_287234


namespace NUMINAMATH_CALUDE_cubic_equation_implies_square_l2872_287288

theorem cubic_equation_implies_square (y : ℝ) : 
  2 * y^3 + 3 * y^2 - 2 * y - 8 = 0 → (5 * y - 2)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_implies_square_l2872_287288


namespace NUMINAMATH_CALUDE_three_over_x_equals_one_l2872_287289

theorem three_over_x_equals_one (x : ℝ) (h : 1 - 6/x + 9/x^2 = 0) : 3/x = 1 := by
  sorry

end NUMINAMATH_CALUDE_three_over_x_equals_one_l2872_287289


namespace NUMINAMATH_CALUDE_edward_money_theorem_l2872_287259

/-- Represents the amount of money Edward had before spending --/
def initial_amount : ℝ := 22

/-- Represents the amount Edward spent on books --/
def spent_amount : ℝ := 16

/-- Represents the amount Edward has left --/
def remaining_amount : ℝ := 6

/-- Represents the number of books Edward bought --/
def number_of_books : ℕ := 92

/-- Theorem stating that the initial amount equals the sum of spent and remaining amounts --/
theorem edward_money_theorem :
  initial_amount = spent_amount + remaining_amount := by sorry

end NUMINAMATH_CALUDE_edward_money_theorem_l2872_287259


namespace NUMINAMATH_CALUDE_trigonometric_product_bounds_l2872_287265

theorem trigonometric_product_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2 + Real.sqrt 3) / 8 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_bounds_l2872_287265


namespace NUMINAMATH_CALUDE_tangent_line_inclination_l2872_287272

theorem tangent_line_inclination (a : ℝ) : 
  (∀ x : ℝ, (fun x => a * x^3 - 2) x = a * x^3 - 2) →
  (slope_at_neg_one : ℝ) →
  slope_at_neg_one = Real.tan (π / 4) →
  slope_at_neg_one = (fun x => 3 * a * x^2) (-1) →
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_inclination_l2872_287272


namespace NUMINAMATH_CALUDE_journey_time_approx_24_hours_l2872_287280

/-- Represents a segment of the journey --/
structure Segment where
  distance : Float
  speed : Float
  stay : Float

/-- Calculates the time taken for a segment --/
def segmentTime (s : Segment) : Float :=
  s.distance / s.speed + s.stay

/-- Represents Manex's journey --/
def manexJourney : List Segment := [
  { distance := 70, speed := 60, stay := 1 },
  { distance := 50, speed := 35, stay := 3 },
  { distance := 20, speed := 60, stay := 0 },
  { distance := 20, speed := 30, stay := 2 },
  { distance := 30, speed := 40, stay := 0 },
  { distance := 60, speed := 70, stay := 2.5 },
  { distance := 60, speed := 35, stay := 0.75 }
]

/-- Calculates the total outbound distance --/
def outboundDistance : Float :=
  (manexJourney.map (·.distance)).sum

/-- Represents the return journey --/
def returnJourney : Segment :=
  { distance := outboundDistance + 100, speed := 55, stay := 0 }

/-- Calculates the total journey time --/
def totalJourneyTime : Float :=
  (manexJourney.map segmentTime).sum + segmentTime returnJourney

/-- Theorem stating that the total journey time is approximately 24 hours --/
theorem journey_time_approx_24_hours :
  (totalJourneyTime).round = 24 := by
  sorry


end NUMINAMATH_CALUDE_journey_time_approx_24_hours_l2872_287280


namespace NUMINAMATH_CALUDE_parallel_line_equation_l2872_287220

/-- A line passing through (1,1) and parallel to x+2y+2016=0 has equation x+2y-3=0 -/
theorem parallel_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∃ c : ℝ, l = {(x, y) | x + 2*y + c = 0}) →
  ((1, 1) ∈ l) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (x + 2*y + 2016 = 0 → False)) →
  l = {(x, y) | x + 2*y - 3 = 0} :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l2872_287220


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2872_287238

/-- Given a geometric sequence {a_n} where a_2 = 2 and a_5 = 1/4,
    prove that the sum a_1*a_2 + a_2*a_3 + ... + a_5*a_6 equals 341/32. -/
theorem geometric_sequence_sum (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 2) * a n = (a (n + 1))^2) →  -- geometric sequence property
  a 2 = 2 →
  a 5 = 1/4 →
  (a 1 * a 2 + a 2 * a 3 + a 3 * a 4 + a 4 * a 5 + a 5 * a 6 : ℚ) = 341/32 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2872_287238


namespace NUMINAMATH_CALUDE_negative_one_point_five_less_than_negative_one_and_one_fifth_l2872_287290

theorem negative_one_point_five_less_than_negative_one_and_one_fifth : -1.5 < -(1 + 1/5) := by
  sorry

end NUMINAMATH_CALUDE_negative_one_point_five_less_than_negative_one_and_one_fifth_l2872_287290


namespace NUMINAMATH_CALUDE_expression_evaluation_l2872_287277

theorem expression_evaluation (x : ℝ) (h : x = -3) :
  (5 + 2*x*(x+2) - 4^2) / (x - 4 + x^2) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2872_287277


namespace NUMINAMATH_CALUDE_volunteer_arrangements_l2872_287237

/-- The number of ways to arrange n people among k exits, with each exit having at least one person. -/
def arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of permutations of r items chosen from n items. -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

theorem volunteer_arrangements :
  arrangements 5 4 = choose 5 2 * permutations 3 3 ∧ 
  arrangements 5 4 = 240 := by sorry

end NUMINAMATH_CALUDE_volunteer_arrangements_l2872_287237


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2872_287242

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (x, 1)
  are_parallel a b → x = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2872_287242


namespace NUMINAMATH_CALUDE_arithmetic_sum_special_case_l2872_287258

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ d n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sum_special_case (k : ℕ) :
  arithmetic_sum (k^2 - k + 1) 1 (2*k + 1) = (2*k + 1) * (k^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_special_case_l2872_287258


namespace NUMINAMATH_CALUDE_min_probability_bound_l2872_287295

open Real

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- The probability function P(k) -/
noncomputable def P (k : ℕ) : ℝ :=
  let count := Finset.filter (fun n : ℕ => 
    floor (n / k) + floor ((200 - n) / k) = floor (200 / k)) 
    (Finset.range 199)
  (count.card : ℝ) / 199

theorem min_probability_bound :
  ∀ k : ℕ, k % 2 = 1 → 1 ≤ k → k ≤ 199 → P k ≥ 50 / 101 := by sorry

end NUMINAMATH_CALUDE_min_probability_bound_l2872_287295


namespace NUMINAMATH_CALUDE_swimming_pool_volume_l2872_287264

/-- Calculate the volume of a trapezoidal prism-shaped swimming pool -/
theorem swimming_pool_volume 
  (width : ℝ) 
  (length : ℝ) 
  (shallow_depth : ℝ) 
  (deep_depth : ℝ) 
  (h_width : width = 9) 
  (h_length : length = 12) 
  (h_shallow : shallow_depth = 1) 
  (h_deep : deep_depth = 4) : 
  (1 / 2) * (shallow_depth + deep_depth) * width * length = 270 := by
sorry

end NUMINAMATH_CALUDE_swimming_pool_volume_l2872_287264


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_property_l2872_287205

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a straight line -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- Checks if a point lies on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Checks if a point lies on a line -/
def on_line (l : Line) (p : Point) : Prop :=
  p.y = l.m * p.x + l.c

/-- Checks if a point lies on an asymptote of the hyperbola -/
def on_asymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y = (h.b / h.a) * p.x ∨ p.y = -(h.b / h.a) * p.x

/-- The main theorem -/
theorem hyperbola_line_intersection_property
  (h : Hyperbola) (l : Line)
  (p q p' q' : Point)
  (hp : on_hyperbola h p)
  (hq : on_hyperbola h q)
  (hp' : on_asymptote h p')
  (hq' : on_asymptote h q')
  (hlp : on_line l p)
  (hlq : on_line l q)
  (hlp' : on_line l p')
  (hlq' : on_line l q') :
  |p.x - p'.x| = |q.x - q'.x| ∧ |p.y - p'.y| = |q.y - q'.y| :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_property_l2872_287205


namespace NUMINAMATH_CALUDE_comparison_of_trigonometric_expressions_l2872_287223

theorem comparison_of_trigonometric_expressions :
  let a := (1/2) * Real.cos (4 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (4 * π / 180)
  let b := Real.cos (13 * π / 180)^2 - Real.sin (13 * π / 180)^2
  let c := (2 * Real.tan (23 * π / 180)) / (1 - Real.tan (23 * π / 180)^2)
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_comparison_of_trigonometric_expressions_l2872_287223


namespace NUMINAMATH_CALUDE_exactly_one_mean_value_point_l2872_287262

-- Define the function f(x) = x³ + 2x
def f (x : ℝ) : ℝ := x^3 + 2*x

-- Define the mean value point condition
def is_mean_value_point (f : ℝ → ℝ) (x₀ : ℝ) (a b : ℝ) : Prop :=
  x₀ ∈ Set.Icc a b ∧ f x₀ = (∫ (x : ℝ) in a..b, f x) / (b - a)

-- Theorem statement
theorem exactly_one_mean_value_point :
  ∃! x₀ : ℝ, is_mean_value_point f x₀ (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_mean_value_point_l2872_287262


namespace NUMINAMATH_CALUDE_find_number_l2872_287273

theorem find_number : ∃ (x : ℝ), 5 + x * (8 - 3) = 15 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2872_287273


namespace NUMINAMATH_CALUDE_money_difference_l2872_287261

-- Define the amounts for each day
def tuesday_amount : ℝ := 8.5
def wednesday_amount : ℝ := 5.5 * tuesday_amount
def thursday_amount : ℝ := wednesday_amount * 1.1
def friday_amount : ℝ := thursday_amount * 0.75

-- Define the difference
def difference : ℝ := friday_amount - tuesday_amount

-- Theorem statement
theorem money_difference : difference = 30.06875 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l2872_287261


namespace NUMINAMATH_CALUDE_james_comics_count_l2872_287297

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- The number of years James writes comics -/
def numYears : ℕ := 4

/-- The frequency of James writing comics (every other day) -/
def comicFrequency : ℕ := 2

/-- The total number of comics James writes in 4 non-leap years -/
def totalComics : ℕ := (daysInYear * numYears) / comicFrequency

theorem james_comics_count : totalComics = 730 := by
  sorry

end NUMINAMATH_CALUDE_james_comics_count_l2872_287297


namespace NUMINAMATH_CALUDE_largest_lcm_with_15_l2872_287285

theorem largest_lcm_with_15 : 
  (Finset.image (fun x => Nat.lcm 15 x) {3, 5, 9, 12, 10, 15}).max = some 60 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_15_l2872_287285


namespace NUMINAMATH_CALUDE_max_beads_is_27_l2872_287284

/-- Represents the maximum number of weighings allowed -/
def max_weighings : ℕ := 3

/-- Represents the number of groups in each weighing -/
def groups_per_weighing : ℕ := 3

/-- Calculates the maximum number of beads that can be in the pile -/
def max_beads : ℕ := groups_per_weighing ^ max_weighings

/-- Theorem stating that the maximum number of beads is 27 -/
theorem max_beads_is_27 : max_beads = 27 := by
  sorry

#eval max_beads -- Should output 27

end NUMINAMATH_CALUDE_max_beads_is_27_l2872_287284


namespace NUMINAMATH_CALUDE_min_value_3x_plus_4y_l2872_287239

theorem min_value_3x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_4y_l2872_287239


namespace NUMINAMATH_CALUDE_expression_evaluation_l2872_287235

theorem expression_evaluation :
  let a : ℚ := -2
  let b : ℚ := 1/2
  (2*a + b) * (2*a - b) + (3*a - b)^2 - ((12*a*b^2 - 16*a^2*b + 4*b) / (2*b)) = 104 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2872_287235


namespace NUMINAMATH_CALUDE_turtle_theorem_l2872_287287

def turtle_problem (initial : ℕ) : ℕ :=
  let additional := 3 * initial - 2
  let total := initial + additional
  total / 2

theorem turtle_theorem : turtle_problem 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_turtle_theorem_l2872_287287


namespace NUMINAMATH_CALUDE_product_of_fractions_l2872_287296

theorem product_of_fractions : 
  (10 : ℚ) / 6 * 4 / 20 * 20 / 12 * 16 / 32 * 40 / 24 * 8 / 40 * 60 / 36 * 32 / 64 = 25 / 324 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2872_287296


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2872_287249

theorem min_reciprocal_sum : ∀ a b : ℕ+, 
  4 * a + b = 6 → 
  (1 : ℝ) / 1 + (1 : ℝ) / 2 ≤ (1 : ℝ) / a + (1 : ℝ) / b :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2872_287249


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l2872_287232

theorem rectangular_garden_width (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 432 →
  width = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l2872_287232


namespace NUMINAMATH_CALUDE_regression_and_probability_theorem_l2872_287218

/-- Data point representing year and sales volume -/
structure DataPoint where
  year : ℕ
  sales : ℕ

/-- Linear regression coefficients -/
structure RegressionCoefficients where
  b : ℚ
  a : ℚ

def data : List DataPoint := [
  ⟨1, 5⟩, ⟨2, 5⟩, ⟨3, 6⟩, ⟨4, 7⟩, ⟨5, 7⟩
]

def calculateRegressionCoefficients (data : List DataPoint) : RegressionCoefficients :=
  sorry

def probabilityConsecutiveYears (data : List DataPoint) : ℚ :=
  sorry

theorem regression_and_probability_theorem :
  let coeffs := calculateRegressionCoefficients data
  coeffs.b = 3/5 ∧ coeffs.a = 21/5 ∧ probabilityConsecutiveYears data = 2/5 := by
  sorry

#check regression_and_probability_theorem

end NUMINAMATH_CALUDE_regression_and_probability_theorem_l2872_287218


namespace NUMINAMATH_CALUDE_sandys_hourly_wage_l2872_287217

theorem sandys_hourly_wage (hours_friday hours_saturday hours_sunday : ℕ) 
  (total_earnings : ℕ) (hourly_wage : ℚ) :
  hours_friday = 10 →
  hours_saturday = 6 →
  hours_sunday = 14 →
  total_earnings = 450 →
  hourly_wage * (hours_friday + hours_saturday + hours_sunday) = total_earnings →
  hourly_wage = 15 := by
sorry

end NUMINAMATH_CALUDE_sandys_hourly_wage_l2872_287217


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l2872_287260

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ

/-- Represents the result of removing smaller cubes from the corners of a larger cube -/
structure ModifiedCube where
  originalCube : Cube
  removedCube : Cube

/-- Calculates the number of edges in the modified cube structure -/
def edgeCount (mc : ModifiedCube) : ℕ :=
  12 + 8 * 6

/-- Theorem stating that removing cubes of side length 5 from each corner of a cube 
    with side length 10 results in a solid with 60 edges -/
theorem modified_cube_edge_count :
  let largeCube := Cube.mk 10
  let smallCube := Cube.mk 5
  let modifiedCube := ModifiedCube.mk largeCube smallCube
  edgeCount modifiedCube = 60 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l2872_287260


namespace NUMINAMATH_CALUDE_max_ab_for_tangent_circle_l2872_287216

/-- Given a line l: x + 2y = 0 tangent to a circle C: (x-a)² + (y-b)² = 5,
    where the center (a,b) of C is above l, the maximum value of ab is 25/8 -/
theorem max_ab_for_tangent_circle (a b : ℝ) : 
  (∀ x y : ℝ, (x + 2*y = 0) → ((x - a)^2 + (y - b)^2 = 5)) →  -- tangency condition
  (a + 2*b > 0) →  -- center above the line
  (∀ a' b' : ℝ, (∀ x y : ℝ, (x + 2*y = 0) → ((x - a')^2 + (y - b')^2 = 5)) → 
                (a' + 2*b' > 0) → 
                a * b ≤ a' * b') →
  a * b = 25/8 := by sorry

end NUMINAMATH_CALUDE_max_ab_for_tangent_circle_l2872_287216


namespace NUMINAMATH_CALUDE_ride_cost_is_factor_of_remaining_tickets_l2872_287228

def total_tickets : ℕ := 40
def spent_tickets : ℕ := 28
def remaining_tickets : ℕ := total_tickets - spent_tickets

def is_factor (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

theorem ride_cost_is_factor_of_remaining_tickets :
  ∀ (num_rides cost_per_ride : ℕ),
    num_rides > 0 →
    cost_per_ride > 0 →
    num_rides * cost_per_ride = remaining_tickets →
    is_factor remaining_tickets cost_per_ride :=
by sorry

end NUMINAMATH_CALUDE_ride_cost_is_factor_of_remaining_tickets_l2872_287228


namespace NUMINAMATH_CALUDE_zeros_sum_inequality_l2872_287206

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * log x

theorem zeros_sum_inequality (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ →
  f (1 / exp 2) x₁ = 0 → f (1 / exp 2) x₂ = 0 →
  log (x₁ + x₂) > log 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_zeros_sum_inequality_l2872_287206


namespace NUMINAMATH_CALUDE_largest_class_size_l2872_287250

theorem largest_class_size (n : ℕ) (total : ℕ) : 
  n = 5 → 
  total = 115 → 
  ∃ x : ℕ, x > 0 ∧ 
    (x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = total) ∧ 
    x = 27 := by
  sorry

end NUMINAMATH_CALUDE_largest_class_size_l2872_287250


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2872_287202

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 40)
  (h2 : badminton = 20)
  (h3 : tennis = 18)
  (h4 : neither = 5)
  (h5 : badminton + tennis - (badminton + tennis - total + neither) = total - neither) :
  badminton + tennis - total + neither = 3 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2872_287202


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l2872_287200

theorem quadratic_roots_problem (α β b : ℝ) : 
  (∀ x, x^2 + b*x - 1 = 0 ↔ x = α ∨ x = β) →
  α * β - 2*α - 2*β = -11 →
  b = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l2872_287200


namespace NUMINAMATH_CALUDE_set_equality_implies_a_values_l2872_287281

theorem set_equality_implies_a_values (a : ℝ) : 
  ({0, -1, 2*a} : Set ℝ) = {a-1, -|a|, a+1} ↔ a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_values_l2872_287281
