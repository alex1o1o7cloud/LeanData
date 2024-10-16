import Mathlib

namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l466_46696

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-18/17, 46/17)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: 2y = -7x - 2 -/
def line2 (x y : ℚ) : Prop := 2 * y = -7 * x - 2

theorem intersection_point_is_unique :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = intersection_point :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l466_46696


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l466_46611

/-- Given a line with equation 3x - y + 2 = 0, its symmetric line with respect to the y-axis has the equation 3x + y - 2 = 0 -/
theorem symmetric_line_equation (x y : ℝ) :
  (3 * x - y + 2 = 0) → 
  ∃ (x' y' : ℝ), (3 * x' + y' - 2 = 0 ∧ x' = -x ∧ y' = y) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l466_46611


namespace NUMINAMATH_CALUDE_cookie_ratio_l466_46685

/-- Prove that the ratio of Chris's cookies to Kenny's cookies is 1:2 -/
theorem cookie_ratio (total : ℕ) (glenn : ℕ) (kenny : ℕ) (chris : ℕ)
  (h1 : total = 33)
  (h2 : glenn = 24)
  (h3 : glenn = 4 * kenny)
  (h4 : total = chris + kenny + glenn) :
  chris = kenny / 2 := by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_l466_46685


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_l466_46638

theorem product_of_five_consecutive_integers (n : ℕ) : 
  n = 3 → (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_l466_46638


namespace NUMINAMATH_CALUDE_new_savings_is_200_l466_46629

/-- Calculates the new monthly savings after an increase in expenses -/
def new_monthly_savings (salary : ℚ) (initial_savings_rate : ℚ) (expense_increase_rate : ℚ) : ℚ :=
  let initial_expenses := salary * (1 - initial_savings_rate)
  let new_expenses := initial_expenses * (1 + expense_increase_rate)
  salary - new_expenses

/-- Proves that the new monthly savings is 200 given the specified conditions -/
theorem new_savings_is_200 :
  new_monthly_savings 5000 (20 / 100) (20 / 100) = 200 := by
  sorry

end NUMINAMATH_CALUDE_new_savings_is_200_l466_46629


namespace NUMINAMATH_CALUDE_real_solutions_exist_l466_46600

theorem real_solutions_exist : ∃ x : ℝ, x^4 - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_real_solutions_exist_l466_46600


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l466_46643

theorem absolute_value_inequality (a b : ℝ) (h : a > b) : |a| > b := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l466_46643


namespace NUMINAMATH_CALUDE_rose_rice_problem_l466_46619

theorem rose_rice_problem (x : ℚ) : 
  (10000 * (1 - x) * (3/4) = 750) → x = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_rose_rice_problem_l466_46619


namespace NUMINAMATH_CALUDE_daily_shoppers_l466_46681

theorem daily_shoppers (tax_free_percentage : ℝ) (weekly_tax_payers : ℕ) : 
  tax_free_percentage = 0.06 →
  weekly_tax_payers = 6580 →
  ∃ (daily_shoppers : ℕ), daily_shoppers = 1000 ∧ 
    (1 - tax_free_percentage) * (daily_shoppers : ℝ) * 7 = weekly_tax_payers := by
  sorry

end NUMINAMATH_CALUDE_daily_shoppers_l466_46681


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l466_46632

theorem similar_triangles_leg_length (a b c d : ℝ) :
  a > 0 → b > 0 → c > 0 → d > 0 →
  a = 12 → b = 9 → c = 7.5 →
  a / c = b / d →
  d = 5.625 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l466_46632


namespace NUMINAMATH_CALUDE_tree_spacing_l466_46635

/-- Proves that the distance between consecutive trees is 18 meters
    given a yard of 414 meters with 24 equally spaced trees. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 414)
  (h2 : num_trees = 24)
  (h3 : num_trees ≥ 2) :
  yard_length / (num_trees - 1) = 18 := by
sorry

end NUMINAMATH_CALUDE_tree_spacing_l466_46635


namespace NUMINAMATH_CALUDE_min_value_of_function_l466_46626

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  (x^2 + 4) / x ≥ 4 ∧ ∃ y > 0, (y^2 + 4) / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l466_46626


namespace NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l466_46651

theorem arithmetic_mean_sqrt2 : 
  (Real.sqrt 2 + 1 + (Real.sqrt 2 - 1)) / 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l466_46651


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l466_46655

theorem boys_to_girls_ratio : 
  ∀ (boys girls : ℕ), 
    boys = 80 →
    girls = boys + 128 →
    ∃ (a b : ℕ), a = 5 ∧ b = 13 ∧ a * girls = b * boys :=
by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l466_46655


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_two_l466_46654

theorem sqrt_sum_equals_two (x y θ : ℝ) 
  (h1 : x + y = 3 - Real.cos (4 * θ)) 
  (h2 : x - y = 4 * Real.sin (2 * θ)) : 
  Real.sqrt x + Real.sqrt y = 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_two_l466_46654


namespace NUMINAMATH_CALUDE_apple_distribution_l466_46698

theorem apple_distribution (t x : ℕ) (h1 : t = 4) (h2 : (9 * t * x) / 10 - 6 = 48) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l466_46698


namespace NUMINAMATH_CALUDE_perpendicular_tangents_a_value_l466_46674

/-- The value of 'a' for which the curves y = ax³ - 6x² + 12x and y = exp(x)
    have perpendicular tangents at x = 1 -/
theorem perpendicular_tangents_a_value :
  ∀ a : ℝ,
  (∀ x : ℝ, deriv (fun x => a * x^3 - 6 * x^2 + 12 * x) 1 *
             deriv (fun x => Real.exp x) 1 = -1) →
  a = -1 / (3 * Real.exp 1) := by
sorry


end NUMINAMATH_CALUDE_perpendicular_tangents_a_value_l466_46674


namespace NUMINAMATH_CALUDE_double_sum_of_factors_17_l466_46606

/-- The sum of positive factors of a natural number -/
def sum_of_factors (n : ℕ) : ℕ := sorry

/-- The boxed notation representing the sum of positive factors -/
notation "⌈" n "⌉" => sum_of_factors n

/-- Theorem stating that the double application of sum_of_factors to 17 equals 39 -/
theorem double_sum_of_factors_17 : ⌈⌈17⌉⌉ = 39 := by sorry

end NUMINAMATH_CALUDE_double_sum_of_factors_17_l466_46606


namespace NUMINAMATH_CALUDE_quadratic_sum_l466_46683

/-- Given a quadratic function g(x) = 2x^2 + Bx + C, 
    if g(1) = 3 and g(2) = 0, then 2 + B + C + 2C = 23 -/
theorem quadratic_sum (B C : ℝ) : 
  (2 * 1^2 + B * 1 + C = 3) → 
  (2 * 2^2 + B * 2 + C = 0) → 
  (2 + B + C + 2 * C = 23) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l466_46683


namespace NUMINAMATH_CALUDE_translation_proof_l466_46656

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point in 2D -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

theorem translation_proof :
  let P : Point := { x := -3, y := 2 }
  let translated_P := translate (translate P 2 0) 0 (-2)
  translated_P = { x := -1, y := 0 } := by sorry

end NUMINAMATH_CALUDE_translation_proof_l466_46656


namespace NUMINAMATH_CALUDE_local_extremum_and_minimum_l466_46686

-- Define the function f
def f (a b x : ℝ) : ℝ := a^2 * x^3 + 3 * a * x^2 - b * x - 1

-- State the theorem
theorem local_extremum_and_minimum (a b : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≥ f a b 1) ∧
  (f a b 1 = 0) ∧
  (∀ x ≥ 0, f a b x ≥ -1) →
  a = -1/2 ∧ b = -9/4 ∧ ∀ x ≥ 0, f a b x ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_local_extremum_and_minimum_l466_46686


namespace NUMINAMATH_CALUDE_product_correction_l466_46616

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem product_correction (a b : ℕ) :
  a ≥ 10 ∧ a < 100 →  -- a is a two-digit number
  a > 0 ∧ b > 0 →  -- a and b are positive
  (reverse_digits a) * b = 143 →
  a * b = 341 := by
sorry

end NUMINAMATH_CALUDE_product_correction_l466_46616


namespace NUMINAMATH_CALUDE_strawberry_harvest_l466_46605

theorem strawberry_harvest (garden_length : ℝ) (garden_width : ℝ) 
  (plantable_percentage : ℝ) (plants_per_sqft : ℝ) (strawberries_per_plant : ℝ) : ℝ :=
  by
  have garden_length_eq : garden_length = 10 := by sorry
  have garden_width_eq : garden_width = 12 := by sorry
  have plantable_percentage_eq : plantable_percentage = 0.9 := by sorry
  have plants_per_sqft_eq : plants_per_sqft = 4 := by sorry
  have strawberries_per_plant_eq : strawberries_per_plant = 8 := by sorry
  
  have total_area : ℝ := garden_length * garden_width
  have plantable_area : ℝ := total_area * plantable_percentage
  have total_plants : ℝ := plantable_area * plants_per_sqft
  have total_strawberries : ℝ := total_plants * strawberries_per_plant
  
  exact total_strawberries

end NUMINAMATH_CALUDE_strawberry_harvest_l466_46605


namespace NUMINAMATH_CALUDE_calculation_proofs_l466_46633

theorem calculation_proofs :
  (1) -2^2 * (1/4) + 4 / (4/9) + (-1)^2023 = 7 ∧
  (2) -1^4 + |2 - (-3)^2| + (1/2) / (-3/2) = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l466_46633


namespace NUMINAMATH_CALUDE_inequality_solution_set_l466_46690

theorem inequality_solution_set (x : ℝ) :
  (x / (x - 1) + (x + 1) / (2 * x) ≥ 5 / 2) ↔ (x ≥ 1 / 2 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l466_46690


namespace NUMINAMATH_CALUDE_ten_mile_taxi_cost_l466_46615

/-- Calculates the cost of a taxi ride given the base fare, per-mile rate, and distance traveled. -/
def taxiRideCost (baseFare : ℝ) (perMileRate : ℝ) (distance : ℝ) : ℝ :=
  baseFare + perMileRate * distance

/-- Theorem stating that a 10-mile taxi ride costs $5.00 given the specified base fare and per-mile rate. -/
theorem ten_mile_taxi_cost :
  let baseFare : ℝ := 2.00
  let perMileRate : ℝ := 0.30
  let distance : ℝ := 10
  taxiRideCost baseFare perMileRate distance = 5.00 := by
  sorry


end NUMINAMATH_CALUDE_ten_mile_taxi_cost_l466_46615


namespace NUMINAMATH_CALUDE_range_of_linear_function_l466_46692

def g (c d x : ℝ) : ℝ := c * x + d

theorem range_of_linear_function (c d : ℝ) (hc : c > 0) :
  Set.range (fun x => g c d x) = Set.Icc (-c + d) (2*c + d) :=
sorry

end NUMINAMATH_CALUDE_range_of_linear_function_l466_46692


namespace NUMINAMATH_CALUDE_quiz_probability_l466_46639

theorem quiz_probability (n : ℕ) (k : ℕ) (p : ℚ) : 
  n = 6 → k = 4 → p = 1 / k → 
  1 - (1 - p)^n = 3367 / 4096 := by
sorry

end NUMINAMATH_CALUDE_quiz_probability_l466_46639


namespace NUMINAMATH_CALUDE_simplify_cube_roots_l466_46608

theorem simplify_cube_roots : (64 : ℝ) ^ (1/3) - (216 : ℝ) ^ (1/3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_l466_46608


namespace NUMINAMATH_CALUDE_inverse_sum_property_l466_46640

-- Define a function f with domain ℝ
variable (f : ℝ → ℝ)

-- Define the inverse function g of f
variable (g : ℝ → ℝ)

-- Define the symmetry condition for f
def symmetric_about_neg_one_zero (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f ((-1) - x) = f ((-1) + x)

-- Define the inverse relationship between f and g
def inverse_functions (f g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (f x) = x ∧ f (g x) = x

-- Theorem statement
theorem inverse_sum_property
  (h_sym : symmetric_about_neg_one_zero f)
  (h_inv : inverse_functions f g)
  (x₁ x₂ : ℝ)
  (h_sum : x₁ + x₂ = 0) :
  g x₁ + g x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_property_l466_46640


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_l466_46668

theorem dormitory_to_city_distance : ∃ (D : ℝ), 
  (1/5 : ℝ) * D + (2/3 : ℝ) * D + 14 = D ∧ D = 105 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_l466_46668


namespace NUMINAMATH_CALUDE_cube_sum_implies_sum_l466_46610

theorem cube_sum_implies_sum (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_implies_sum_l466_46610


namespace NUMINAMATH_CALUDE_stating_initial_amount_is_200_l466_46620

/-- Represents the exchange rate from U.S. dollars to Canadian dollars -/
def exchange_rate : ℚ := 6 / 5

/-- Represents the amount spent in Canadian dollars -/
def amount_spent : ℚ := 80

/-- 
Given an initial amount of U.S. dollars, calculates the remaining amount 
of Canadian dollars after exchanging and spending
-/
def remaining_amount (d : ℚ) : ℚ := (4 / 5) * d

/-- 
Theorem stating that given the exchange rate and spending conditions, 
the initial amount of U.S. dollars is 200
-/
theorem initial_amount_is_200 : 
  ∃ d : ℚ, d = 200 ∧ 
  exchange_rate * d - amount_spent = remaining_amount d :=
sorry

end NUMINAMATH_CALUDE_stating_initial_amount_is_200_l466_46620


namespace NUMINAMATH_CALUDE_sector_area_l466_46663

/-- Given a circular sector with perimeter 6 and central angle 1 radian, its area is 2. -/
theorem sector_area (r : ℝ) (h1 : r + 2 * r = 6) (h2 : 1 = 1) : r * r / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l466_46663


namespace NUMINAMATH_CALUDE_existence_of_N_l466_46697

theorem existence_of_N : ∃ N : ℝ, (0.47 * N - 0.36 * 1412) + 63 = 3 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_N_l466_46697


namespace NUMINAMATH_CALUDE_bernard_luke_age_problem_l466_46689

/-- Given that in 8 years, Mr. Bernard will be 3 times as old as Luke is now,
    prove that 10 years less than their average current age is 2 * L - 14,
    where L is Luke's current age. -/
theorem bernard_luke_age_problem (L : ℕ) : 
  (L + ((3 * L) - 8)) / 2 - 10 = 2 * L - 14 := by
  sorry

end NUMINAMATH_CALUDE_bernard_luke_age_problem_l466_46689


namespace NUMINAMATH_CALUDE_kenny_basketball_time_l466_46666

/-- 
Given that:
- Kenny played basketball last week
- He ran for twice as long as he played basketball
- He practiced on the trumpet for twice as long as he ran
- He practiced on the trumpet for 40 hours

Prove that Kenny played basketball for 10 hours last week.
-/
theorem kenny_basketball_time (trumpet_time : ℕ) (h1 : trumpet_time = 40) :
  let run_time := trumpet_time / 2
  let basketball_time := run_time / 2
  basketball_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_kenny_basketball_time_l466_46666


namespace NUMINAMATH_CALUDE_curve_is_circle_and_line_l466_46695

/-- The polar equation of the curve -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 * Real.cos θ - 3 * ρ * Real.cos θ + ρ - 3 = 0

/-- Definition of a circle in polar coordinates -/
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b r : ℝ, ∀ ρ θ : ℝ, f ρ θ ↔ (ρ * Real.cos θ - a)^2 + (ρ * Real.sin θ - b)^2 = r^2

/-- Definition of a line in polar coordinates -/
def is_line (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, ∀ ρ θ : ℝ, f ρ θ ↔ ρ * (a * Real.cos θ + b * Real.sin θ) = 1

/-- The theorem stating that the curve consists of a circle and a line -/
theorem curve_is_circle_and_line :
  (∃ f g : ℝ → ℝ → Prop, 
    (∀ ρ θ : ℝ, polar_equation ρ θ ↔ (f ρ θ ∨ g ρ θ)) ∧
    is_circle f ∧ is_line g) :=
sorry

end NUMINAMATH_CALUDE_curve_is_circle_and_line_l466_46695


namespace NUMINAMATH_CALUDE_michaels_number_l466_46621

theorem michaels_number (m : ℕ) :
  m % 75 = 0 ∧ m % 40 = 0 ∧ 1000 ≤ m ∧ m ≤ 3000 →
  m = 1800 ∨ m = 2400 ∨ m = 3000 := by
sorry

end NUMINAMATH_CALUDE_michaels_number_l466_46621


namespace NUMINAMATH_CALUDE_remaining_weight_calculation_l466_46601

/-- Calculates the total remaining weight of groceries after an accident --/
theorem remaining_weight_calculation (green_beans_weight : ℝ) : 
  green_beans_weight = 60 →
  let rice_weight := green_beans_weight - 30
  let sugar_weight := green_beans_weight - 10
  let rice_remaining := rice_weight * (2/3)
  let sugar_remaining := sugar_weight * (4/5)
  rice_remaining + sugar_remaining + green_beans_weight = 120 :=
by
  sorry


end NUMINAMATH_CALUDE_remaining_weight_calculation_l466_46601


namespace NUMINAMATH_CALUDE_coefficient_f_nonzero_l466_46614

-- Define the polynomial Q(x)
def Q (a b c d f : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + f

-- Define the theorem
theorem coefficient_f_nonzero 
  (a b c d f : ℝ) 
  (h1 : ∃ p q r s : ℝ, p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧ 
                       Q a b c d f p = 0 ∧ Q a b c d f q = 0 ∧ Q a b c d f r = 0 ∧ Q a b c d f s = 0)
  (h2 : Q a b c d f 1 = 0) : 
  f ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_f_nonzero_l466_46614


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_16_cube_root_between_9_and_9_1_l466_46684

theorem unique_integer_divisible_by_16_cube_root_between_9_and_9_1 :
  ∃! n : ℕ+, 
    (∃ k : ℕ, n = 16 * k) ∧ 
    9 < (n : ℝ) ^ (1/3) ∧ 
    (n : ℝ) ^ (1/3) < 9.1 ∧
    n = 736 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_16_cube_root_between_9_and_9_1_l466_46684


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l466_46642

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem tenth_term_of_sequence (a₁ r : ℚ) (h₁ : a₁ = 12) (h₂ : r = 1/2) :
  geometric_sequence a₁ r 10 = 3/128 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l466_46642


namespace NUMINAMATH_CALUDE_divisibility_by_five_l466_46631

theorem divisibility_by_five (d : Nat) : 
  d ≤ 9 → (41830 + d) % 5 = 0 ↔ d = 0 ∨ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l466_46631


namespace NUMINAMATH_CALUDE_negation_existential_geq_zero_l466_46667

theorem negation_existential_geq_zero :
  ¬(∃ x : ℝ, x + 1 ≥ 0) ↔ ∀ x : ℝ, x + 1 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_existential_geq_zero_l466_46667


namespace NUMINAMATH_CALUDE_product_expansion_evaluation_l466_46612

theorem product_expansion_evaluation :
  ∀ (a b c d : ℝ),
  (∀ x : ℝ, (4 * x^2 - 3 * x + 6) * (9 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = 48 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_evaluation_l466_46612


namespace NUMINAMATH_CALUDE_boys_average_age_l466_46649

/-- Proves that the average age of boys is 12 years given the school statistics -/
theorem boys_average_age (total_students : ℕ) (girls : ℕ) (girls_avg_age : ℝ) (school_avg_age : ℝ) :
  total_students = 652 →
  girls = 163 →
  girls_avg_age = 11 →
  school_avg_age = 11.75 →
  let boys := total_students - girls
  let boys_total_age := school_avg_age * total_students - girls_avg_age * girls
  boys_total_age / boys = 12 := by
sorry


end NUMINAMATH_CALUDE_boys_average_age_l466_46649


namespace NUMINAMATH_CALUDE_anna_guessing_ratio_l466_46628

theorem anna_guessing_ratio (c d : ℝ) 
  (h1 : c > 0 ∧ d > 0)  -- Ensure c and d are positive
  (h2 : 0.9 * c + 0.05 * d = 0.1 * c + 0.95 * d)  -- Equal number of cat and dog images
  (h3 : 0.95 * d = d - 0.05 * d)  -- 95% correct when guessing dog
  (h4 : 0.9 * c = c - 0.1 * c)  -- 90% correct when guessing cat
  : d / c = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_anna_guessing_ratio_l466_46628


namespace NUMINAMATH_CALUDE_existence_and_not_forall_l466_46670

theorem existence_and_not_forall :
  (∃ x₀ : ℝ, x₀ - 2 > 0) ∧ ¬(∀ x : ℝ, 2^x > x^2) := by
  sorry

end NUMINAMATH_CALUDE_existence_and_not_forall_l466_46670


namespace NUMINAMATH_CALUDE_g_at_minus_one_l466_46660

/-- The function g(x) = -2x^2 + 5x - 7 --/
def g (x : ℝ) : ℝ := -2 * x^2 + 5 * x - 7

/-- Theorem: g(-1) = -14 --/
theorem g_at_minus_one : g (-1) = -14 := by
  sorry

end NUMINAMATH_CALUDE_g_at_minus_one_l466_46660


namespace NUMINAMATH_CALUDE_f_max_min_l466_46661

-- Define the function
def f (x : ℝ) : ℝ := |-(x)| - |x - 3|

-- State the theorem
theorem f_max_min :
  (∀ x : ℝ, f x ≤ 3) ∧
  (∃ x : ℝ, f x = 3) ∧
  (∀ x : ℝ, f x ≥ -3) ∧
  (∃ x : ℝ, f x = -3) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_l466_46661


namespace NUMINAMATH_CALUDE_probability_score_le_6_is_13_35_l466_46627

structure Bag where
  red_balls : ℕ
  black_balls : ℕ

def score (red : ℕ) (black : ℕ) : ℕ :=
  red + 3 * black

def probability_score_le_6 (b : Bag) : ℚ :=
  let total_balls := b.red_balls + b.black_balls
  let drawn_balls := 4
  (Nat.choose b.red_balls 4 * Nat.choose b.black_balls 0 +
   Nat.choose b.red_balls 3 * Nat.choose b.black_balls 1) /
  Nat.choose total_balls drawn_balls

theorem probability_score_le_6_is_13_35 (b : Bag) :
  b.red_balls = 4 → b.black_balls = 3 → probability_score_le_6 b = 13 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_score_le_6_is_13_35_l466_46627


namespace NUMINAMATH_CALUDE_slightly_used_crayons_l466_46617

theorem slightly_used_crayons (total : ℕ) (new : ℕ) (broken : ℕ) (slightly_used : ℕ) : 
  total = 120 →
  new = total / 3 →
  broken = total / 5 →
  slightly_used = total - new - broken →
  slightly_used = 56 := by
sorry

end NUMINAMATH_CALUDE_slightly_used_crayons_l466_46617


namespace NUMINAMATH_CALUDE_role_assignment_count_l466_46672

def number_of_role_assignments (men : ℕ) (women : ℕ) : ℕ :=
  let male_role_assignments := men
  let female_role_assignments := women
  let remaining_actors := men + women - 2
  let either_gender_role_assignments := Nat.choose remaining_actors 4 * Nat.factorial 4
  male_role_assignments * female_role_assignments * either_gender_role_assignments

theorem role_assignment_count :
  number_of_role_assignments 6 7 = 33120 :=
sorry

end NUMINAMATH_CALUDE_role_assignment_count_l466_46672


namespace NUMINAMATH_CALUDE_fence_perimeter_is_106_l466_46665

/-- Given a square field enclosed by posts, calculates the outer perimeter of the fence. -/
def fence_perimeter (num_posts : ℕ) (post_width : ℝ) (gap : ℝ) : ℝ :=
  let posts_per_side : ℕ := (num_posts - 4) / 4 + 2
  let gaps_per_side : ℕ := posts_per_side - 1
  let side_length : ℝ := gaps_per_side * gap + posts_per_side * post_width
  4 * side_length

/-- Theorem stating that the fence with given specifications has a perimeter of 106 feet. -/
theorem fence_perimeter_is_106 :
  fence_perimeter 16 0.5 6 = 106 := by
  sorry

#eval fence_perimeter 16 0.5 6

end NUMINAMATH_CALUDE_fence_perimeter_is_106_l466_46665


namespace NUMINAMATH_CALUDE_base3_to_base10_equality_l466_46657

/-- Converts a base-3 number to base-10 --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

/-- The base-3 representation of the number --/
def base3Number : List Nat := [1, 2, 0, 1, 2]

/-- Theorem stating that the base-3 number 12012 is equal to 140 in base-10 --/
theorem base3_to_base10_equality : base3ToBase10 base3Number = 140 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_equality_l466_46657


namespace NUMINAMATH_CALUDE_max_abc_value_l466_46637

theorem max_abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + c = (a + c) * (b + c)) :
  a * b * c ≤ 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_max_abc_value_l466_46637


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l466_46648

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (3 * x - 2 * y = 3) ∧ (x + 4 * y = 1) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l466_46648


namespace NUMINAMATH_CALUDE_class_average_problem_l466_46646

theorem class_average_problem (x : ℝ) : 
  0.15 * x + 0.50 * 78 + 0.35 * 63 = 76.05 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l466_46646


namespace NUMINAMATH_CALUDE_parabola_tangents_and_triangle_l466_46679

/-- Parabola equation: 8y = x^2 + 16 -/
def parabola (x y : ℝ) : Prop := 8 * y = x^2 + 16

/-- Point M coordinates -/
def M : ℝ × ℝ := (3, 0)

/-- Tangent line equation -/
def tangent_line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

/-- Points of tangency A and B -/
def A : ℝ × ℝ := (-2, 2.5)
def B : ℝ × ℝ := (8, 10)

/-- Main theorem -/
theorem parabola_tangents_and_triangle :
  ∃ (m₁ b₁ m₂ b₂ : ℝ),
    /- Tangent equations -/
    (∀ x y, tangent_line m₁ b₁ x y ↔ y = -1/2 * x + 1.5) ∧
    (∀ x y, tangent_line m₂ b₂ x y ↔ y = 2 * x - 6) ∧
    /- Angle between tangents -/
    (Real.arctan ((m₂ - m₁) / (1 + m₁ * m₂)) = Real.pi / 2) ∧
    /- Area of triangle ABM -/
    (1/2 * Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) *
     Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2) = 125/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangents_and_triangle_l466_46679


namespace NUMINAMATH_CALUDE_second_account_interest_rate_l466_46622

/-- Proves that the interest rate of the second account is 5% given the problem conditions -/
theorem second_account_interest_rate 
  (total_investment : ℝ) 
  (first_account_investment : ℝ) 
  (first_account_rate : ℝ) 
  (total_interest : ℝ) 
  (h1 : total_investment = 8000)
  (h2 : first_account_investment = 3000)
  (h3 : first_account_rate = 0.08)
  (h4 : total_interest = 490) :
  let second_account_investment := total_investment - first_account_investment
  let first_account_interest := first_account_investment * first_account_rate
  let second_account_interest := total_interest - first_account_interest
  let second_account_rate := second_account_interest / second_account_investment
  second_account_rate = 0.05 := by
sorry


end NUMINAMATH_CALUDE_second_account_interest_rate_l466_46622


namespace NUMINAMATH_CALUDE_two_digit_quadratic_equation_l466_46659

theorem two_digit_quadratic_equation :
  ∃ (P : ℕ), 
    (P ≥ 10 ∧ P < 100) ∧ 
    (∀ x : ℝ, x^2 + P*x + 2001 = (x + 29) * (x + 69)) :=
sorry

end NUMINAMATH_CALUDE_two_digit_quadratic_equation_l466_46659


namespace NUMINAMATH_CALUDE_rachel_total_score_l466_46653

/-- Rachel's video game scoring system -/
def video_game_score (points_per_treasure : ℕ) (treasures_level1 : ℕ) (treasures_level2 : ℕ) : ℕ :=
  points_per_treasure * (treasures_level1 + treasures_level2)

/-- Theorem: Rachel's total score is 63 points -/
theorem rachel_total_score :
  video_game_score 9 5 2 = 63 :=
by sorry

end NUMINAMATH_CALUDE_rachel_total_score_l466_46653


namespace NUMINAMATH_CALUDE_cubic_factorization_l466_46645

theorem cubic_factorization (y : ℝ) : y^3 - 4*y^2 + 4*y = y*(y-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l466_46645


namespace NUMINAMATH_CALUDE_population_less_than_15_percent_in_fifth_year_l466_46658

def population_decrease_rate : ℝ := 0.35
def target_population_ratio : ℝ := 0.15

def population_after_n_years (n : ℕ) : ℝ :=
  (1 - population_decrease_rate) ^ n

theorem population_less_than_15_percent_in_fifth_year :
  (∀ k < 5, population_after_n_years k > target_population_ratio) ∧
  population_after_n_years 5 < target_population_ratio :=
sorry

end NUMINAMATH_CALUDE_population_less_than_15_percent_in_fifth_year_l466_46658


namespace NUMINAMATH_CALUDE_jersey_shoe_cost_ratio_l466_46678

/-- Given the information about Jeff's purchase of shoes and jerseys,
    prove that the ratio of the cost of one jersey to one pair of shoes is 1:4 -/
theorem jersey_shoe_cost_ratio :
  ∀ (total_cost shoe_cost : ℕ) (shoe_pairs jersey_count : ℕ),
    total_cost = 560 →
    shoe_cost = 480 →
    shoe_pairs = 6 →
    jersey_count = 4 →
    (total_cost - shoe_cost) / jersey_count / (shoe_cost / shoe_pairs) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_jersey_shoe_cost_ratio_l466_46678


namespace NUMINAMATH_CALUDE_first_grade_sample_size_l466_46613

/-- Given a total sample size and ratios for three groups, 
    calculate the number of samples for the first group -/
def stratifiedSampleSize (totalSample : ℕ) (ratio1 ratio2 ratio3 : ℕ) : ℕ :=
  (ratio1 * totalSample) / (ratio1 + ratio2 + ratio3)

/-- Theorem: For a total sample of 80 and ratios 4:3:3, 
    the first group's sample size is 32 -/
theorem first_grade_sample_size :
  stratifiedSampleSize 80 4 3 3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_first_grade_sample_size_l466_46613


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l466_46669

/-- An ellipse with foci at (15, 30) and (15, 90) that is tangent to the y-axis has a major axis of length 30√5 -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)) (F₁ F₂ Y : ℝ × ℝ),
  F₁ = (15, 30) →
  F₂ = (15, 90) →
  Y.1 = 0 →
  (∀ p ∈ E, dist p F₁ + dist p F₂ = dist Y F₁ + dist Y F₂) →
  (∀ q : ℝ × ℝ, q.1 = 0 → dist q F₁ + dist q F₂ ≥ dist Y F₁ + dist Y F₂) →
  dist Y F₁ + dist Y F₂ = 30 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l466_46669


namespace NUMINAMATH_CALUDE_series_sum_equals_inverse_sqrt5_minus_1_l466_46625

/-- The sum of the series $\sum_{k=0}^{\infty} \frac{5^{2^k}}{25^{2^k} - 1}$ is equal to $\frac{1}{\sqrt{5}-1}$ -/
theorem series_sum_equals_inverse_sqrt5_minus_1 :
  let series_term (k : ℕ) := (5 ^ (2 ^ k)) / ((25 ^ (2 ^ k)) - 1)
  ∑' (k : ℕ), series_term k = 1 / (Real.sqrt 5 - 1) := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_inverse_sqrt5_minus_1_l466_46625


namespace NUMINAMATH_CALUDE_right_triangle_area_l466_46676

/-- A right-angled triangle with an altitude from the right angle -/
structure RightTriangleWithAltitude where
  /-- The length of one leg of the triangle -/
  a : ℝ
  /-- The length of the other leg of the triangle -/
  b : ℝ
  /-- The radius of the inscribed circle in one of the smaller triangles -/
  r₁ : ℝ
  /-- The radius of the inscribed circle in the other smaller triangle -/
  r₂ : ℝ
  /-- Ensure the radii are positive -/
  h_positive_r₁ : r₁ > 0
  h_positive_r₂ : r₂ > 0
  /-- The ratio of the legs is equal to the ratio of the radii -/
  h_ratio : a / b = r₁ / r₂

/-- The theorem stating the area of the right-angled triangle -/
theorem right_triangle_area (t : RightTriangleWithAltitude) (h_r₁ : t.r₁ = 3) (h_r₂ : t.r₂ = 4) : 
  (1/2) * t.a * t.b = 150 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_area_l466_46676


namespace NUMINAMATH_CALUDE_max_value_tan_l466_46682

/-- Given a function f(x) = 3sin(x) + 2cos(x), when f(x) reaches its maximum value, tan(x) = 3/2 -/
theorem max_value_tan (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 3 * Real.sin x + 2 * Real.cos x
  ∃ (x_max : ℝ), (∀ y, f y ≤ f x_max) → Real.tan x_max = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_tan_l466_46682


namespace NUMINAMATH_CALUDE_no_uniform_L_partition_l466_46604

/-- An 'L' shape on a grid -/
structure LShape (n : ℕ) where
  cells : Finset (Fin n × Fin n)
  size : cells.card = 3
  adjacent : ∀ (c1 c2 : Fin n × Fin n), c1 ∈ cells → c2 ∈ cells → 
    (c1.1 = c2.1 ∧ c1.2.succ = c2.2) ∨ 
    (c1.1 = c2.1 ∧ c1.2 = c2.2.succ) ∨ 
    (c1.1.succ = c2.1 ∧ c1.2 = c2.2) ∨ 
    (c1.1 = c2.1.succ ∧ c1.2 = c2.2)

/-- A partition of a grid into 'L' shapes -/
def LPartition (n : ℕ) := 
  {partition : Finset (LShape n) // 
    (∀ (i j : Fin n), ∃! (L : LShape n), L ∈ partition ∧ (i, j) ∈ L.cells)}

/-- The number of 'L' shapes intersecting a row -/
def rowIntersections (n : ℕ) (partition : LPartition n) (row : Fin n) : ℕ :=
  (partition.val.filter (λ L => ∃ j, (row, j) ∈ L.cells)).card

/-- The number of 'L' shapes intersecting a column -/
def colIntersections (n : ℕ) (partition : LPartition n) (col : Fin n) : ℕ :=
  (partition.val.filter (λ L => ∃ i, (i, col) ∈ L.cells)).card

/-- The main theorem -/
theorem no_uniform_L_partition :
  ¬ ∃ (partition : LPartition 12),
    (∀ row : Fin 12, rowIntersections 12 partition row = rowIntersections 12 partition 0) ∧
    (∀ col : Fin 12, colIntersections 12 partition col = colIntersections 12 partition 0) :=
sorry

end NUMINAMATH_CALUDE_no_uniform_L_partition_l466_46604


namespace NUMINAMATH_CALUDE_max_min_product_l466_46641

theorem max_min_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (sum_eq : a + b + c = 10) (prod_sum_eq : a * b + b * c + c * a = 25) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 25 / 9 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' + b' + c' = 10 ∧ a' * b' + b' * c' + c' * a' = 25 ∧
    min (a' * b') (min (b' * c') (c' * a')) = 25 / 9 :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l466_46641


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l466_46673

def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

theorem roots_of_polynomial :
  (p 1 = 0) ∧ (p 2 = 0) ∧ (p 4 = 0) ∧
  (∀ x : ℝ, p x = 0 → x = 1 ∨ x = 2 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l466_46673


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l466_46644

/-- A parabola y = ax^2 + bx + 2 is tangent to the line y = 2x + 3 if and only if a = -1 and b = 4 -/
theorem parabola_tangent_to_line (a b : ℝ) : 
  (∃ x : ℝ, ax^2 + bx + 2 = 2*x + 3 ∧ 
   ∀ y : ℝ, y ≠ x → ax^2 + bx + 2 ≠ 2*y + 3) ↔ 
  (a = -1 ∧ b = 4) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l466_46644


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l466_46609

/-- The area of the shaded region in a grid with given dimensions and an unshaded triangle -/
theorem shaded_area_calculation (grid_width grid_height triangle_base triangle_height : ℝ) 
  (hw : grid_width = 15)
  (hh : grid_height = 5)
  (hb : triangle_base = grid_width)
  (ht : triangle_height = 3) :
  grid_width * grid_height - (1/2 * triangle_base * triangle_height) = 52.5 := by
  sorry

#check shaded_area_calculation

end NUMINAMATH_CALUDE_shaded_area_calculation_l466_46609


namespace NUMINAMATH_CALUDE_f_ln_2_value_l466_46624

/-- A function f is monotonically decreasing on (0, +∞) if for any a, b ∈ (0, +∞) with a < b, f(a) ≥ f(b) -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ a b, 0 < a → a < b → f a ≥ f b

/-- The main theorem -/
theorem f_ln_2_value (f : ℝ → ℝ) 
  (h_mono : MonoDecreasing f)
  (h_domain : ∀ x, x > 0 → f x ≠ 0)
  (h_eq : ∀ x, x > 0 → f (f x - 1 / Real.exp x) = 1 / Real.exp 1 + 1) :
  f (Real.log 2) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_f_ln_2_value_l466_46624


namespace NUMINAMATH_CALUDE_machine_a_production_rate_l466_46652

/-- The number of sprockets produced by each machine -/
def total_sprockets : ℕ := 880

/-- The additional time taken by Machine P compared to Machine Q -/
def time_difference : ℕ := 10

/-- The production rate of Machine Q relative to Machine A -/
def q_rate_relative_to_a : ℚ := 11/10

/-- The production rate of Machine A in sprockets per hour -/
def machine_a_rate : ℚ := 8

/-- The production rate of Machine Q in sprockets per hour -/
def machine_q_rate : ℚ := q_rate_relative_to_a * machine_a_rate

/-- The time taken by Machine Q to produce the total sprockets -/
def machine_q_time : ℚ := total_sprockets / machine_q_rate

/-- The time taken by Machine P to produce the total sprockets -/
def machine_p_time : ℚ := machine_q_time + time_difference

theorem machine_a_production_rate :
  (total_sprockets : ℚ) = machine_a_rate * machine_p_time ∧
  (total_sprockets : ℚ) = machine_q_rate * machine_q_time ∧
  machine_a_rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_machine_a_production_rate_l466_46652


namespace NUMINAMATH_CALUDE_eighth_odd_multiple_of_5_l466_46694

/-- A function that generates the nth positive odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- The property of being both odd and a multiple of 5 -/
def isOddMultipleOf5 (k : ℕ) : Prop := k % 2 = 1 ∧ k % 5 = 0

theorem eighth_odd_multiple_of_5 :
  nthOddMultipleOf5 8 = 75 ∧ 
  isOddMultipleOf5 (nthOddMultipleOf5 8) ∧
  (∀ m < 8, ∃ k < nthOddMultipleOf5 8, k > 0 ∧ isOddMultipleOf5 k) :=
sorry

end NUMINAMATH_CALUDE_eighth_odd_multiple_of_5_l466_46694


namespace NUMINAMATH_CALUDE_complement_of_sqrt_range_l466_46602

-- Define the universal set U as ℝ
def U := ℝ

-- Define the set A as the range of y = x^(1/2)
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt x}

-- State the theorem
theorem complement_of_sqrt_range :
  Set.compl A = Set.Iio (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_complement_of_sqrt_range_l466_46602


namespace NUMINAMATH_CALUDE_min_value_of_w_l466_46623

def w (x y : ℝ) : ℝ := 2 * x^2 + 3 * y^2 + 8 * x - 6 * y + 30

theorem min_value_of_w :
  ∀ x y : ℝ, w x y ≥ 19 ∧ ∃ x₀ y₀ : ℝ, w x₀ y₀ = 19 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_w_l466_46623


namespace NUMINAMATH_CALUDE_equation_solution_l466_46607

theorem equation_solution : 
  ∃ (x : ℤ), 45 - (28 - (37 - (15 - x))) = 56 ∧ x = 122 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l466_46607


namespace NUMINAMATH_CALUDE_square_property_l466_46688

theorem square_property (n : ℕ+) : ∃ k : ℤ, (n + 1 : ℤ) * (n + 2) * (n^2 + 3*n) + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_property_l466_46688


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l466_46618

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that the athlete's heart beats 19500 times during the race -/
theorem athlete_heartbeats : 
  total_heartbeats 150 26 5 = 19500 := by
  sorry

end NUMINAMATH_CALUDE_athlete_heartbeats_l466_46618


namespace NUMINAMATH_CALUDE_initial_distance_proof_l466_46687

/-- The initial distance between two cars on a main road --/
def initial_distance : ℝ := 165

/-- The total distance traveled by the first car --/
def car1_distance : ℝ := 65

/-- The distance traveled by the second car --/
def car2_distance : ℝ := 62

/-- The final distance between the two cars --/
def final_distance : ℝ := 38

/-- Theorem stating that the initial distance is correct given the problem conditions --/
theorem initial_distance_proof :
  initial_distance = car1_distance + car2_distance + final_distance :=
by sorry

end NUMINAMATH_CALUDE_initial_distance_proof_l466_46687


namespace NUMINAMATH_CALUDE_largest_positive_root_cubic_bounded_coeff_l466_46647

/-- The largest positive real root of a cubic equation with bounded coefficients -/
theorem largest_positive_root_cubic_bounded_coeff :
  ∀ (b₂ b₁ b₀ : ℝ),
  (|b₂| ≤ 1) → (|b₁| ≤ 1) → (|b₀| ≤ 1) →
  ∃ (r : ℝ),
    (∀ (x : ℝ), x^3 + b₂*x^2 + b₁*x + b₀ = 0 → x ≤ r) ∧
    (r^3 + b₂*r^2 + b₁*r + b₀ = 0) ∧
    (1.5 < r) ∧ (r < 2) :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_root_cubic_bounded_coeff_l466_46647


namespace NUMINAMATH_CALUDE_tank_capacity_after_adding_gas_l466_46671

/-- Given a tank with a capacity of 54 gallons, initially filled to 3/4 of its capacity,
    prove that after adding 9 gallons of gasoline, the tank will be filled to 23/25 of its capacity. -/
theorem tank_capacity_after_adding_gas (tank_capacity : ℚ) (initial_fill : ℚ) (added_gas : ℚ) :
  tank_capacity = 54 →
  initial_fill = 3 / 4 →
  added_gas = 9 →
  (initial_fill * tank_capacity + added_gas) / tank_capacity = 23 / 25 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_after_adding_gas_l466_46671


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l466_46636

theorem solution_set_implies_sum (a b : ℝ) : 
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l466_46636


namespace NUMINAMATH_CALUDE_polynomial_symmetry_representation_l466_46675

theorem polynomial_symmetry_representation
  (p : ℝ → ℝ) (a : ℝ)
  (h_symmetry : ∀ x, p x = p (a - x)) :
  ∃ h : ℝ → ℝ, ∀ x, p x = h ((x - a / 2) ^ 2) :=
sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_representation_l466_46675


namespace NUMINAMATH_CALUDE_initial_bananas_per_child_l466_46650

theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) :
  total_children = 640 →
  absent_children = 320 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ),
    total_children * initial_bananas = (total_children - absent_children) * (initial_bananas + extra_bananas) ∧
    initial_bananas = 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_bananas_per_child_l466_46650


namespace NUMINAMATH_CALUDE_not_surjective_product_of_injective_exists_surjective_factors_l466_46691

-- Part a
theorem not_surjective_product_of_injective (f g : ℤ → ℤ) 
  (hf : Function.Injective f) (hg : Function.Injective g) :
  ¬ Function.Surjective (fun x ↦ f x * g x) := by
  sorry

-- Part b
theorem exists_surjective_factors (f : ℤ → ℤ) (hf : Function.Surjective f) :
  ∃ g h : ℤ → ℤ, Function.Surjective g ∧ Function.Surjective h ∧
    ∀ x, f x = g x * h x := by
  sorry

end NUMINAMATH_CALUDE_not_surjective_product_of_injective_exists_surjective_factors_l466_46691


namespace NUMINAMATH_CALUDE_paperclip_excess_day_l466_46630

def paperclip_sequence (k : ℕ) : ℕ := 4 * 3^k

theorem paperclip_excess_day :
  (∀ j : ℕ, j < 6 → paperclip_sequence j ≤ 2000) ∧
  paperclip_sequence 6 > 2000 :=
sorry

end NUMINAMATH_CALUDE_paperclip_excess_day_l466_46630


namespace NUMINAMATH_CALUDE_quadratic_function_transformation_l466_46603

-- Define the original function
def original_function (x : ℝ) : ℝ := (x + 1)^2 + 3

-- Define the transformation
def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 2) - 1

-- Define the expected result function
def result_function (x : ℝ) : ℝ := (x - 1)^2 + 2

-- Theorem statement
theorem quadratic_function_transformation :
  ∀ x : ℝ, transform original_function x = result_function x :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_transformation_l466_46603


namespace NUMINAMATH_CALUDE_custom_op_theorem_l466_46693

def customOp (M N : Set ℕ) : Set ℕ := {x | x ∈ M ∨ x ∈ N ∧ x ∉ M ∩ N}

def M : Set ℕ := {0, 2, 4, 6, 8, 10}
def N : Set ℕ := {0, 3, 6, 9, 12, 15}

theorem custom_op_theorem :
  (customOp (customOp M N) M) = {2, 4, 8, 10, 3, 9, 12, 15} := by sorry

end NUMINAMATH_CALUDE_custom_op_theorem_l466_46693


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l466_46677

theorem arithmetic_simplification : 2537 + 240 * 3 / 60 - 347 = 2202 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l466_46677


namespace NUMINAMATH_CALUDE_average_marks_of_failed_candidates_l466_46699

theorem average_marks_of_failed_candidates
  (total_candidates : ℕ)
  (overall_average : ℚ)
  (passed_average : ℚ)
  (passed_candidates : ℕ)
  (h1 : total_candidates = 120)
  (h2 : overall_average = 35)
  (h3 : passed_average = 39)
  (h4 : passed_candidates = 100) :
  (total_candidates * overall_average - passed_candidates * passed_average) / (total_candidates - passed_candidates) = 15 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_of_failed_candidates_l466_46699


namespace NUMINAMATH_CALUDE_smallest_perimeter_of_rectangle_l466_46664

theorem smallest_perimeter_of_rectangle (a b : ℕ) : 
  a * b = 1000 → 
  2 * (a + b) ≥ 130 ∧ 
  ∃ (x y : ℕ), x * y = 1000 ∧ 2 * (x + y) = 130 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_of_rectangle_l466_46664


namespace NUMINAMATH_CALUDE_translate_sin_function_l466_46662

/-- Translates the given trigonometric function and proves the result -/
theorem translate_sin_function :
  let f (x : ℝ) := Real.sin (2 * x + π / 6)
  let g (x : ℝ) := f (x + π / 6) + 1
  ∀ x, g x = 2 * (Real.cos x) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_translate_sin_function_l466_46662


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l466_46634

theorem two_digit_number_puzzle : ∃ (n : ℕ) (x y : ℕ),
  0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
  n = 10 * x + y ∧
  x^2 + y^2 = 10 * x + y + 11 ∧
  2 * x * y = 10 * x + y - 5 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l466_46634


namespace NUMINAMATH_CALUDE_zero_in_interval_l466_46680

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 - 8 + 2 * x

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 3 4, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l466_46680
