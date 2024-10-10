import Mathlib

namespace investment_percentage_proof_l3133_313369

theorem investment_percentage_proof (total_sum P1 P2 x : ℝ) : 
  total_sum = 1600 →
  P1 + P2 = total_sum →
  P2 = 1100 →
  (P1 * x / 100) + (P2 * 5 / 100) = 85 →
  x = 6 := by
sorry

end investment_percentage_proof_l3133_313369


namespace quadratic_coefficients_from_absolute_value_l3133_313345

theorem quadratic_coefficients_from_absolute_value (x : ℝ) :
  (|x - 3| = 4 ↔ x = 7 ∨ x = -1) →
  ∃ d e : ℝ, (∀ x : ℝ, x^2 + d*x + e = 0 ↔ x = 7 ∨ x = -1) ∧ d = -6 ∧ e = -7 :=
by sorry

end quadratic_coefficients_from_absolute_value_l3133_313345


namespace sequence_problem_l3133_313315

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, b (n + 1) = b n + d

theorem sequence_problem (a b : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 2 →
  a 4 = 16 →
  arithmetic_sequence b →
  a 3 = b 3 →
  a 5 = b 5 →
  (∀ n : ℕ, a n = 2^n) ∧
  (∀ n : ℕ, b n = 12*n - 28) ∧
  (∀ n : ℕ, S n = (3*n - 10) * 2^(n+3) - 80) :=
by sorry

end sequence_problem_l3133_313315


namespace total_apples_picked_l3133_313368

/-- The total number of apples picked by Mike, Nancy, and Keith is 16. -/
theorem total_apples_picked (mike_apples nancy_apples keith_apples : ℕ)
  (h1 : mike_apples = 7)
  (h2 : nancy_apples = 3)
  (h3 : keith_apples = 6) :
  mike_apples + nancy_apples + keith_apples = 16 := by
  sorry

end total_apples_picked_l3133_313368


namespace fixed_point_on_line_l3133_313362

-- Define the line equation
def line_equation (m x y : ℝ) : Prop :=
  m * x - y + 2 * m + 1 = 0

-- Theorem statement
theorem fixed_point_on_line :
  ∀ m : ℝ, line_equation m (-2) 1 := by
  sorry

end fixed_point_on_line_l3133_313362


namespace garden_perimeter_l3133_313354

/-- A rectangular garden with given diagonal and area has a specific perimeter -/
theorem garden_perimeter (x y : ℝ) (h_rectangle : x > 0 ∧ y > 0) 
  (h_diagonal : x^2 + y^2 = 34^2) (h_area : x * y = 240) : 
  2 * (x + y) = 80 := by
  sorry

end garden_perimeter_l3133_313354


namespace only_valid_root_l3133_313309

def original_equation (x : ℝ) : Prop :=
  (2 * x^2) / (x - 1) - (2 * x + 7) / 3 + (4 - 6 * x) / (x - 1) + 1 = 0

def transformed_equation (x : ℝ) : Prop :=
  x^2 - 5 * x + 4 = 0

theorem only_valid_root :
  (∀ x : ℝ, transformed_equation x ↔ (x = 4 ∨ x = 1)) →
  (∀ x : ℝ, original_equation x ↔ x = 4) :=
sorry

end only_valid_root_l3133_313309


namespace fraction_product_simplification_l3133_313347

theorem fraction_product_simplification :
  (240 : ℚ) / 18 * 7 / 210 * 9 / 4 = 1 := by sorry

end fraction_product_simplification_l3133_313347


namespace complex_number_properties_l3133_313312

theorem complex_number_properties (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) :
  let z := x + Complex.I * y
  (0 < z.re ∧ 0 < z.im) ∧ Complex.abs z = Real.sqrt 2 ∧ z.re = 1 := by
  sorry

end complex_number_properties_l3133_313312


namespace cora_reading_schedule_l3133_313301

/-- The number of pages Cora needs to read on Thursday to finish her book -/
def pages_to_read_thursday (total_pages : ℕ) (pages_monday : ℕ) (pages_tuesday : ℕ) (pages_wednesday : ℕ) : ℕ :=
  let pages_thursday := (total_pages - pages_monday - pages_tuesday - pages_wednesday) / 3
  pages_thursday

theorem cora_reading_schedule :
  pages_to_read_thursday 158 23 38 61 = 12 := by
  sorry

#eval pages_to_read_thursday 158 23 38 61

end cora_reading_schedule_l3133_313301


namespace jacks_purchase_cost_l3133_313341

/-- The cost of Jack's purchase of a squat rack and barbell -/
theorem jacks_purchase_cost (squat_rack_cost : ℝ) (barbell_cost : ℝ) : 
  squat_rack_cost = 2500 →
  barbell_cost = squat_rack_cost / 10 →
  squat_rack_cost + barbell_cost = 2750 := by
sorry

end jacks_purchase_cost_l3133_313341


namespace min_value_of_y_l3133_313311

theorem min_value_of_y (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by sorry

end min_value_of_y_l3133_313311


namespace prism_volume_problem_l3133_313343

/-- 
Given a rectangular prism with dimensions 15 cm × 5 cm × 4 cm and a smaller prism
with dimensions y cm × 5 cm × x cm removed, if the remaining volume is 120 cm³,
then x + y = 15, where x and y are integers.
-/
theorem prism_volume_problem (x y : ℤ) : 
  (15 * 5 * 4 - y * 5 * x = 120) → (x + y = 15) := by sorry

end prism_volume_problem_l3133_313343


namespace morgan_hula_hoop_time_l3133_313344

/-- Given information about hula hooping times for Nancy, Casey, and Morgan,
    prove that Morgan can hula hoop for 21 minutes. -/
theorem morgan_hula_hoop_time :
  ∀ (nancy casey morgan : ℕ),
    nancy = 10 →
    casey = nancy - 3 →
    morgan = 3 * casey →
    morgan = 21 := by
  sorry

end morgan_hula_hoop_time_l3133_313344


namespace square_root_of_x_plus_y_l3133_313376

theorem square_root_of_x_plus_y (x y : ℝ) 
  (h1 : 2*x + 7*y + 1 = 6^2) 
  (h2 : 8*x + 3*y = 5^3) : 
  (x + y).sqrt = 4 ∨ (x + y).sqrt = -4 := by
  sorry

end square_root_of_x_plus_y_l3133_313376


namespace newspaper_collection_ratio_l3133_313302

def chris_newspapers : ℕ := 42
def lily_extra_newspapers : ℕ := 23

def lily_newspapers : ℕ := chris_newspapers + lily_extra_newspapers

theorem newspaper_collection_ratio :
  (chris_newspapers : ℚ) / (lily_newspapers : ℚ) = 42 / 65 :=
by sorry

end newspaper_collection_ratio_l3133_313302


namespace sin_630_degrees_l3133_313356

theorem sin_630_degrees : Real.sin (630 * π / 180) = -1 := by sorry

end sin_630_degrees_l3133_313356


namespace joans_sandwiches_l3133_313359

/-- Given the conditions for Joan's sandwich making, prove the number of grilled cheese sandwiches. -/
theorem joans_sandwiches (total_cheese : ℕ) (ham_sandwiches : ℕ) (cheese_per_ham : ℕ) (cheese_per_grilled : ℕ)
  (h_total : total_cheese = 50)
  (h_ham : ham_sandwiches = 10)
  (h_cheese_ham : cheese_per_ham = 2)
  (h_cheese_grilled : cheese_per_grilled = 3) :
  (total_cheese - ham_sandwiches * cheese_per_ham) / cheese_per_grilled = 10 := by
  sorry

#eval (50 - 10 * 2) / 3  -- Expected output: 10

end joans_sandwiches_l3133_313359


namespace nilpotent_matrix_square_zero_l3133_313392

theorem nilpotent_matrix_square_zero 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : 
  A ^ 2 = 0 := by
  sorry

end nilpotent_matrix_square_zero_l3133_313392


namespace fresh_mushroom_mass_calculation_l3133_313393

/-- The mass of fresh mushrooms in kg that, when dried, become 15 kg lighter
    and have a moisture content of 60%, given that fresh mushrooms contain 90% water. -/
def fresh_mushroom_mass : ℝ := 20

/-- The water content of fresh mushrooms as a percentage. -/
def fresh_water_content : ℝ := 90

/-- The water content of dried mushrooms as a percentage. -/
def dried_water_content : ℝ := 60

/-- The mass reduction after drying in kg. -/
def mass_reduction : ℝ := 15

theorem fresh_mushroom_mass_calculation :
  fresh_mushroom_mass * (1 - fresh_water_content / 100) =
  (fresh_mushroom_mass - mass_reduction) * (1 - dried_water_content / 100) :=
by sorry

end fresh_mushroom_mass_calculation_l3133_313393


namespace gcd_1029_1437_5649_l3133_313379

theorem gcd_1029_1437_5649 : Nat.gcd 1029 (Nat.gcd 1437 5649) = 3 := by
  sorry

end gcd_1029_1437_5649_l3133_313379


namespace equilateral_triangle_area_l3133_313313

/-- The area of an equilateral triangle with altitude 2√3 is 4√3 -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = 2 * Real.sqrt 3) :
  let side : ℝ := 2 * h / Real.sqrt 3
  let area : ℝ := 1/2 * side * h
  area = 4 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_area_l3133_313313


namespace binary_111_equals_7_l3133_313371

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number we want to convert -/
def binary_111 : List Nat := [1, 1, 1]

/-- Theorem stating that the binary number 111 is equal to the decimal number 7 -/
theorem binary_111_equals_7 : binary_to_decimal binary_111 = 7 := by
  sorry

end binary_111_equals_7_l3133_313371


namespace imaginary_complex_magnitude_l3133_313363

theorem imaginary_complex_magnitude (a : ℝ) : 
  (∃ (b : ℝ), (Complex.I : ℂ) * b = (a + 2 * Complex.I) / (1 + Complex.I)) → 
  Complex.abs (a + Complex.I) = Real.sqrt 5 := by
sorry

end imaginary_complex_magnitude_l3133_313363


namespace cost_of_corn_seeds_l3133_313395

/-- The cost of corn seeds for a farmer's harvest -/
theorem cost_of_corn_seeds
  (fertilizer_pesticide_cost : ℕ)
  (labor_cost : ℕ)
  (bags_of_corn : ℕ)
  (profit_percentage : ℚ)
  (price_per_bag : ℕ)
  (h1 : fertilizer_pesticide_cost = 35)
  (h2 : labor_cost = 15)
  (h3 : bags_of_corn = 10)
  (h4 : profit_percentage = 1/10)
  (h5 : price_per_bag = 11) :
  ∃ (corn_seed_cost : ℕ),
    corn_seed_cost = 49 ∧
    (corn_seed_cost : ℚ) + fertilizer_pesticide_cost + labor_cost +
      (profit_percentage * (bags_of_corn * price_per_bag)) =
    bags_of_corn * price_per_bag :=
by sorry

end cost_of_corn_seeds_l3133_313395


namespace pears_left_l3133_313367

theorem pears_left (jason_pears keith_pears mike_ate : ℕ) 
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_ate = 12) :
  jason_pears + keith_pears - mike_ate = 81 := by
  sorry

end pears_left_l3133_313367


namespace aisha_age_l3133_313384

/-- Given the ages of Ali, Yusaf, and Umar, prove Aisha's age --/
theorem aisha_age (ali_age : ℕ) (yusaf_age : ℕ) (umar_age : ℕ) 
  (h1 : ali_age = 8)
  (h2 : ali_age = yusaf_age + 3)
  (h3 : umar_age = 2 * yusaf_age)
  (h4 : ∃ (aisha_age : ℕ), aisha_age = (ali_age + umar_age) / 2) :
  ∃ (aisha_age : ℕ), aisha_age = 9 := by
  sorry

end aisha_age_l3133_313384


namespace ellipse_line_and_fixed_circle_l3133_313329

/-- Given an ellipse C and points P, Q, and conditions for line l, prove the equation of l and that point S lies on a fixed circle. -/
theorem ellipse_line_and_fixed_circle 
  (x₀ y₀ : ℝ) 
  (hy₀ : y₀ ≠ 0)
  (hP : x₀^2/4 + y₀^2/3 = 1) 
  (Q : ℝ × ℝ)
  (hQ : Q = (x₀/4, y₀/3))
  (l : Set (ℝ × ℝ))
  (hl : ∀ M ∈ l, (M.1 - x₀) * (x₀/4) + (M.2 - y₀) * (y₀/3) = 0)
  (F : ℝ × ℝ)
  (hF : F.1 > 0 ∧ F.1^2 = 1 + F.2^2/3)  -- Condition for right focus
  (S : ℝ × ℝ)
  (hS : ∃ k, S = (4 + k * (4*y₀)/(3*x₀), k) ∧ 
             S.2 = (y₀/(x₀-1)) * (S.1 - 1)) :
  (∀ x y, (x, y) ∈ l ↔ x₀*x/4 + y₀*y/3 = 1) ∧ 
  ((S.1 - 1)^2 + S.2^2 = 36) := by
  sorry


end ellipse_line_and_fixed_circle_l3133_313329


namespace selection_probabilities_l3133_313365

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The number of people to be selected -/
def num_selected : ℕ := 3

/-- The total number of ways to select 3 people from 5 people -/
def total_combinations : ℕ := Nat.choose (num_boys + num_girls) num_selected

/-- The probability of selecting all boys -/
def prob_all_boys : ℚ := (Nat.choose num_boys num_selected : ℚ) / total_combinations

/-- The probability of selecting exactly one girl -/
def prob_one_girl : ℚ := (Nat.choose num_boys (num_selected - 1) * Nat.choose num_girls 1 : ℚ) / total_combinations

/-- The probability of selecting at least one girl -/
def prob_at_least_one_girl : ℚ := 1 - prob_all_boys

theorem selection_probabilities :
  prob_all_boys = 1/10 ∧
  prob_one_girl = 6/10 ∧
  prob_at_least_one_girl = 9/10 := by
  sorry

end selection_probabilities_l3133_313365


namespace floor_length_is_sqrt_150_l3133_313306

/-- Represents a rectangular floor with specific properties -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  total_paint_cost : ℝ
  paint_rate_per_sqm : ℝ

/-- The length is 200% more than the breadth -/
def length_breadth_relation (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth

/-- The total paint cost divided by the rate per sqm gives the area -/
def area_from_paint_cost (floor : RectangularFloor) : Prop :=
  floor.total_paint_cost / floor.paint_rate_per_sqm = floor.length * floor.breadth

/-- Theorem stating the length of the floor -/
theorem floor_length_is_sqrt_150 (floor : RectangularFloor) 
  (h1 : length_breadth_relation floor)
  (h2 : area_from_paint_cost floor)
  (h3 : floor.total_paint_cost = 100)
  (h4 : floor.paint_rate_per_sqm = 2) : 
  floor.length = Real.sqrt 150 := by
  sorry

end floor_length_is_sqrt_150_l3133_313306


namespace odd_periodic_function_sum_l3133_313338

-- Define the properties of function f
def is_odd_periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + 6) = f x) ∧
  (f 1 = 1)

-- State the theorem
theorem odd_periodic_function_sum (f : ℝ → ℝ) 
  (h : is_odd_periodic_function f) : 
  f 2015 + f 2016 = -1 := by
  sorry

end odd_periodic_function_sum_l3133_313338


namespace sugar_profit_percentage_l3133_313323

/-- Proves that given 1000 kg of sugar, with 400 kg sold at 8% profit and 600 kg sold at x% profit,
    if the overall profit is 14%, then x = 18. -/
theorem sugar_profit_percentage 
  (total_sugar : ℝ) 
  (sugar_at_8_percent : ℝ) 
  (sugar_at_x_percent : ℝ) 
  (x : ℝ) :
  total_sugar = 1000 →
  sugar_at_8_percent = 400 →
  sugar_at_x_percent = 600 →
  sugar_at_8_percent * 0.08 + sugar_at_x_percent * (x / 100) = total_sugar * 0.14 →
  x = 18 := by
  sorry

end sugar_profit_percentage_l3133_313323


namespace bisection_method_theorem_l3133_313322

/-- The bisection method theorem -/
theorem bisection_method_theorem 
  (f : ℝ → ℝ) (a b : ℝ) 
  (h_continuous : Continuous f) 
  (h_unique_zero : ∃! x, x ∈ Set.Ioo a b ∧ f x = 0) 
  (h_interval : b - a = 0.1) :
  ∃ n : ℕ, n ≤ 10 ∧ (0.1 / 2^n : ℝ) < 0.0001 := by
sorry

end bisection_method_theorem_l3133_313322


namespace tom_clothing_count_l3133_313353

/-- The total number of pieces of clothing Tom had -/
def total_clothing : ℕ := 36

/-- The number of pieces in the first load -/
def first_load : ℕ := 18

/-- The number of pieces in each of the two equal loads -/
def equal_load : ℕ := 9

/-- The number of equal loads -/
def num_equal_loads : ℕ := 2

theorem tom_clothing_count :
  total_clothing = first_load + num_equal_loads * equal_load :=
by sorry

end tom_clothing_count_l3133_313353


namespace punch_difference_l3133_313374

def orange_punch : ℝ := 4.5
def cherry_punch : ℝ := 2 * orange_punch
def total_punch : ℝ := 21

def apple_juice : ℝ := total_punch - orange_punch - cherry_punch

theorem punch_difference : cherry_punch - apple_juice = 1.5 := by
  sorry

end punch_difference_l3133_313374


namespace function_symmetry_periodicity_l3133_313328

theorem function_symmetry_periodicity 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, f x = -f (2 - x)) : 
  ∀ x, f (x + 4) = f x := by
sorry

end function_symmetry_periodicity_l3133_313328


namespace inequality_solution_subset_l3133_313310

theorem inequality_solution_subset (a : ℝ) :
  (∀ x : ℝ, x^2 < |x - 1| + a → -3 < x ∧ x < 3) →
  a ≤ 5 :=
by sorry

end inequality_solution_subset_l3133_313310


namespace workers_wage_increase_l3133_313389

theorem workers_wage_increase (original_wage : ℝ) (increase_percentage : ℝ) (new_wage : ℝ) : 
  increase_percentage = 40 →
  new_wage = 35 →
  new_wage = original_wage * (1 + increase_percentage / 100) →
  original_wage = 25 := by
sorry

end workers_wage_increase_l3133_313389


namespace sum_of_digits_9ab_l3133_313399

/-- The number of digits in the sequence -/
def n : ℕ := 2023

/-- Integer a consisting of n nines in base 10 -/
def a : ℕ := 10^n - 1

/-- Integer b consisting of n sixes in base 10 -/
def b : ℕ := 2 * (10^n - 1) / 3

/-- The product 9ab -/
def prod : ℕ := 9 * a * b

/-- Sum of digits function -/
def sum_of_digits (m : ℕ) : ℕ := sorry

theorem sum_of_digits_9ab : sum_of_digits prod = 20235 := by sorry

end sum_of_digits_9ab_l3133_313399


namespace triangle_conditions_l3133_313364

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define what it means for a triangle to be right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 ∨ t.b^2 = t.a^2 + t.c^2 ∨ t.c^2 = t.a^2 + t.b^2

-- Define the conditions
def conditionA (t : Triangle) : Prop := t.A = t.B - t.C
def conditionB (t : Triangle) : Prop := t.A = t.B ∧ t.C = 2 * t.A
def conditionC (t : Triangle) : Prop := t.b^2 = t.a^2 - t.c^2
def conditionD (t : Triangle) : Prop := ∃ (k : ℝ), t.a = 2*k ∧ t.b = 3*k ∧ t.c = 4*k

-- The theorem to prove
theorem triangle_conditions (t : Triangle) :
  (conditionA t → isRightTriangle t) ∧
  (conditionB t → isRightTriangle t) ∧
  (conditionC t → isRightTriangle t) ∧
  (conditionD t → ¬isRightTriangle t) := by
  sorry

end triangle_conditions_l3133_313364


namespace age_sum_in_five_years_l3133_313337

/-- Given a person (Mike) who is 30 years younger than his mom, and the sum of their ages is 70 years,
    the sum of their ages in 5 years will be 80 years. -/
theorem age_sum_in_five_years (mike_age mom_age : ℕ) : 
  mike_age = mom_age - 30 → 
  mike_age + mom_age = 70 → 
  (mike_age + 5) + (mom_age + 5) = 80 := by
sorry

end age_sum_in_five_years_l3133_313337


namespace rectangular_toilet_area_l3133_313336

theorem rectangular_toilet_area :
  let length : ℝ := 5
  let width : ℝ := 17 / 20
  let area := length * width
  area = 4.25 := by sorry

end rectangular_toilet_area_l3133_313336


namespace infimum_attained_by_uniform_distribution_l3133_313321

-- Define the set of Borel functions
def BorelFunction (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being an increasing function
def Increasing (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → f x ≤ f y

-- Define a random variable
def RandomVariable (X : ℝ → ℝ) : Prop := sorry

-- Define the property of density not exceeding 1/2
def DensityNotExceedingHalf (X : ℝ → ℝ) : Prop := sorry

-- Define uniform distribution on [-1, 1]
def UniformDistributionOnUnitInterval (U : ℝ → ℝ) : Prop := sorry

-- Define expected value
def ExpectedValue (f : ℝ → ℝ) (X : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem infimum_attained_by_uniform_distribution
  (f : ℝ → ℝ) (X U : ℝ → ℝ) :
  BorelFunction f →
  Increasing f →
  RandomVariable X →
  DensityNotExceedingHalf X →
  UniformDistributionOnUnitInterval U →
  ExpectedValue (fun x => f (abs x)) X ≥ ExpectedValue (fun x => f (abs x)) U :=
sorry

end infimum_attained_by_uniform_distribution_l3133_313321


namespace cryptarithmetic_solution_l3133_313377

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The cryptarithmetic equation ABAC + BAC = KCKDC -/
def CryptarithmeticEquation (A B C D K : Digit) : Prop :=
  1000 * A.val + 100 * B.val + 10 * A.val + C.val +
  100 * B.val + 10 * A.val + C.val =
  10000 * K.val + 1000 * C.val + 100 * K.val + 10 * D.val + C.val

/-- All digits are different -/
def AllDifferent (A B C D K : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ K ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ K ∧
  C ≠ D ∧ C ≠ K ∧
  D ≠ K

theorem cryptarithmetic_solution :
  ∃! (A B C D K : Digit),
    CryptarithmeticEquation A B C D K ∧
    AllDifferent A B C D K ∧
    A.val = 9 ∧ B.val = 5 ∧ C.val = 0 ∧ D.val = 8 ∧ K.val = 1 :=
sorry

end cryptarithmetic_solution_l3133_313377


namespace quadratic_equation_coefficients_l3133_313331

theorem quadratic_equation_coefficients :
  ∀ (a b c d e f : ℝ),
  (∀ x, a * x^2 + b * x + c = d * x^2 + e * x + f) →
  (a - d = 4) →
  (b - e = -2 ∧ c - f = 0) :=
by sorry

end quadratic_equation_coefficients_l3133_313331


namespace range_of_a_l3133_313394

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, x^2 - 2*a*x + 2 ≥ 0) → 
  a ∈ Set.Iic (Real.sqrt 2) :=
sorry

end range_of_a_l3133_313394


namespace complex_exponential_sum_l3133_313317

open Complex

theorem complex_exponential_sum (α β γ : ℝ) :
  exp (I * α) + exp (I * β) + exp (I * γ) = 1 + I →
  exp (-I * α) + exp (-I * β) + exp (-I * γ) = 1 - I :=
by sorry

end complex_exponential_sum_l3133_313317


namespace fixed_points_of_f_l3133_313349

theorem fixed_points_of_f (f : ℝ → ℝ) (hf : ∀ x, f x = 4 * x - x^2) :
  ∃ a b : ℝ, a ≠ b ∧ f a = b ∧ f b = a ∧
    ((a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2) ∨
     (a = (5 - Real.sqrt 5) / 2 ∧ b = (5 + Real.sqrt 5) / 2)) := by
  sorry

end fixed_points_of_f_l3133_313349


namespace twelve_star_x_multiple_of_144_l3133_313386

def star (a b : ℤ) : ℤ := a^2 * b

theorem twelve_star_x_multiple_of_144 (x : ℤ) : ∃ k : ℤ, star 12 x = 144 * k := by
  sorry

end twelve_star_x_multiple_of_144_l3133_313386


namespace power_of_two_greater_than_n_and_factorial_greater_than_power_of_two_l3133_313342

theorem power_of_two_greater_than_n_and_factorial_greater_than_power_of_two :
  (∀ n : ℕ, 2^n > n) ∧
  (∀ n : ℕ, n ≥ 4 → n.factorial > 2^n) := by
  sorry

end power_of_two_greater_than_n_and_factorial_greater_than_power_of_two_l3133_313342


namespace scientific_notation_of_1340000000_l3133_313314

theorem scientific_notation_of_1340000000 :
  ∃ (a : ℝ) (n : ℤ), 1340000000 = a * (10 : ℝ)^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 1.34 ∧ n = 9 :=
by sorry

end scientific_notation_of_1340000000_l3133_313314


namespace min_tiles_for_l_shape_min_tiles_for_specific_l_shape_l3133_313333

/-- Represents the dimensions of a rectangle in inches -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle in square inches -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the number of tiles needed to cover a rectangle -/
def tilesNeeded (r : Rectangle) (tileArea : ℕ) : ℕ := 
  (area r + tileArea - 1) / tileArea

theorem min_tiles_for_l_shape (tile : Rectangle) 
  (large : Rectangle) (small : Rectangle) : ℕ :=
  let tileArea := area tile
  let largeRect := Rectangle.mk (feetToInches large.length) (feetToInches large.width)
  let smallRect := Rectangle.mk (feetToInches small.length) (feetToInches small.width)
  tilesNeeded largeRect tileArea + tilesNeeded smallRect tileArea

theorem min_tiles_for_specific_l_shape : 
  min_tiles_for_l_shape (Rectangle.mk 2 6) (Rectangle.mk 3 4) (Rectangle.mk 2 1) = 168 := by
  sorry

end min_tiles_for_l_shape_min_tiles_for_specific_l_shape_l3133_313333


namespace combinations_of_three_from_seven_l3133_313330

theorem combinations_of_three_from_seven (n k : ℕ) : n = 7 ∧ k = 3 → Nat.choose n k = 35 := by
  sorry

end combinations_of_three_from_seven_l3133_313330


namespace katie_game_difference_l3133_313304

theorem katie_game_difference (katie_games friends_games : ℕ) 
  (h1 : katie_games = 81) (h2 : friends_games = 59) : 
  katie_games - friends_games = 22 := by
sorry

end katie_game_difference_l3133_313304


namespace tangent_line_to_circle_l3133_313320

/-- The line x + 2y - 6 = 0 is tangent to the circle (x-1)^2 + y^2 = 5 at the point (2, 2) -/
theorem tangent_line_to_circle : 
  let circle : ℝ × ℝ → Prop := λ (x, y) ↦ (x - 1)^2 + y^2 = 5
  let line : ℝ × ℝ → Prop := λ (x, y) ↦ x + 2*y - 6 = 0
  let P : ℝ × ℝ := (2, 2)
  (circle P) ∧ (line P) ∧ 
  (∀ Q : ℝ × ℝ, Q ≠ P → (circle Q ∧ line Q → False)) :=
by sorry

end tangent_line_to_circle_l3133_313320


namespace belize_homes_count_belize_homes_count_proof_l3133_313397

theorem belize_homes_count : ℕ → Prop :=
  fun total_homes =>
    let white_homes := total_homes / 4
    let non_white_homes := total_homes - white_homes
    let non_white_homes_with_fireplace := non_white_homes / 5
    let non_white_homes_without_fireplace := non_white_homes - non_white_homes_with_fireplace
    non_white_homes_without_fireplace = 240 → total_homes = 400

-- Proof
theorem belize_homes_count_proof : ∃ (total_homes : ℕ), belize_homes_count total_homes :=
  sorry

end belize_homes_count_belize_homes_count_proof_l3133_313397


namespace arithmetic_sequence_problem_l3133_313352

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 5 = 0.3 →
  arithmetic_sequence a₁ d 12 = 3.1 →
  a₁ = -1.3 ∧ d = 0.4 ∧
  (arithmetic_sequence a₁ d 18 +
   arithmetic_sequence a₁ d 19 +
   arithmetic_sequence a₁ d 20 +
   arithmetic_sequence a₁ d 21 +
   arithmetic_sequence a₁ d 22) = 31.5 := by
  sorry

end arithmetic_sequence_problem_l3133_313352


namespace calculate_not_less_than_50_l3133_313339

/-- Represents the frequency of teachers in different age groups -/
structure TeacherAgeFrequency where
  less_than_30 : ℝ
  between_30_and_50 : ℝ
  not_less_than_50 : ℝ

/-- The sum of all frequencies in a probability distribution is 1 -/
axiom sum_of_frequencies (f : TeacherAgeFrequency) : 
  f.less_than_30 + f.between_30_and_50 + f.not_less_than_50 = 1

/-- Theorem: Given the frequencies for two age groups, we can calculate the third -/
theorem calculate_not_less_than_50 (f : TeacherAgeFrequency) 
    (h1 : f.less_than_30 = 0.3) 
    (h2 : f.between_30_and_50 = 0.5) : 
  f.not_less_than_50 = 0.2 := by
  sorry


end calculate_not_less_than_50_l3133_313339


namespace simplify_2M_minus_N_value_at_specific_points_independence_condition_l3133_313319

-- Define the polynomials M and N
def M (x y : ℝ) : ℝ := x^2 + x*y + 2*y - 2
def N (x y : ℝ) : ℝ := 2*x^2 - 2*x*y + x - 4

-- Theorem 1: Simplification of 2M - N
theorem simplify_2M_minus_N (x y : ℝ) :
  2 * M x y - N x y = 4*x*y + 4*y - x :=
sorry

-- Theorem 2: Value of 2M - N when x = -2 and y = -4
theorem value_at_specific_points :
  2 * M (-2) (-4) - N (-2) (-4) = 18 :=
sorry

-- Theorem 3: Condition for 2M - N to be independent of x
theorem independence_condition (y : ℝ) :
  (∀ x : ℝ, ∃ c : ℝ, 2 * M x y - N x y = c) ↔ y = 1/4 :=
sorry

end simplify_2M_minus_N_value_at_specific_points_independence_condition_l3133_313319


namespace olivia_cookies_l3133_313388

/-- The number of chocolate chip cookies Olivia has -/
def chocolate_chip_cookies (cookies_per_bag : ℕ) (oatmeal_cookies : ℕ) (baggies : ℕ) : ℕ :=
  cookies_per_bag * baggies - oatmeal_cookies

/-- Proof that Olivia has 13 chocolate chip cookies -/
theorem olivia_cookies : chocolate_chip_cookies 9 41 6 = 13 := by
  sorry

end olivia_cookies_l3133_313388


namespace gargamel_tire_purchase_l3133_313396

def sale_price : ℕ := 75
def total_savings : ℕ := 36
def original_price : ℕ := 84

theorem gargamel_tire_purchase :
  (total_savings / (original_price - sale_price) : ℕ) = 4 := by
  sorry

end gargamel_tire_purchase_l3133_313396


namespace rectangle_dimensions_l3133_313326

theorem rectangle_dimensions :
  ∀ (w l : ℝ),
  w > 0 →
  l = 2 * w →
  2 * (l + w) = 3 * (l * w) →
  w = 1 ∧ l = 2 :=
by sorry

end rectangle_dimensions_l3133_313326


namespace net_profit_calculation_l3133_313350

/-- Calculates the net profit given the purchase price, markup, and overhead percentage. -/
def calculate_net_profit (purchase_price markup overhead_percent : ℚ) : ℚ :=
  let overhead := purchase_price * overhead_percent
  markup - overhead

/-- Theorem stating that given the specific values in the problem, the net profit is $40.60. -/
theorem net_profit_calculation :
  let purchase_price : ℚ := 48
  let markup : ℚ := 55
  let overhead_percent : ℚ := 0.30
  calculate_net_profit purchase_price markup overhead_percent = 40.60 := by
  sorry

#eval calculate_net_profit 48 55 0.30

end net_profit_calculation_l3133_313350


namespace min_travel_time_less_than_3_9_l3133_313360

/-- Represents the problem of three people traveling with a motorcycle --/
structure TravelProblem where
  distance : ℝ
  walkSpeed : ℝ
  motorSpeed : ℝ
  motorCapacity : ℕ

/-- Calculates the minimum time for all three people to reach the destination --/
def minTravelTime (p : TravelProblem) : ℝ :=
  sorry

/-- The main theorem stating that the minimum travel time is less than 3.9 hours --/
theorem min_travel_time_less_than_3_9 :
  let p : TravelProblem := {
    distance := 135,
    walkSpeed := 6,
    motorSpeed := 90,
    motorCapacity := 2
  }
  minTravelTime p < 3.9 := by
  sorry

end min_travel_time_less_than_3_9_l3133_313360


namespace relationship_abc_l3133_313355

theorem relationship_abc (x : ℝ) (a b c : ℝ) 
  (h1 : x > Real.exp (-1)) 
  (h2 : x < 1) 
  (h3 : a = Real.log x) 
  (h4 : b = (1/2) ^ (Real.log x)) 
  (h5 : c = Real.exp (Real.log x)) : 
  b > c ∧ c > a := by
  sorry

end relationship_abc_l3133_313355


namespace equal_population_after_14_years_second_village_initial_population_is_correct_l3133_313391

/-- The initial population of Village X -/
def village_x_initial : ℕ := 70000

/-- The yearly decrease in population of Village X -/
def village_x_decrease : ℕ := 1200

/-- The yearly increase in population of the second village -/
def village_2_increase : ℕ := 800

/-- The number of years after which the populations will be equal -/
def years_until_equal : ℕ := 14

/-- The initial population of the second village -/
def village_2_initial : ℕ := 42000

theorem equal_population_after_14_years :
  village_x_initial - village_x_decrease * years_until_equal = 
  village_2_initial + village_2_increase * years_until_equal :=
by sorry

/-- The theorem stating that the calculated initial population of the second village is correct -/
theorem second_village_initial_population_is_correct : village_2_initial = 42000 :=
by sorry

end equal_population_after_14_years_second_village_initial_population_is_correct_l3133_313391


namespace muffin_cost_is_two_l3133_313327

/-- The cost of a muffin given the conditions of Francis and Kiera's breakfast -/
def muffin_cost : ℝ :=
  let fruit_cup_cost : ℝ := 3
  let francis_muffins : ℕ := 2
  let francis_fruit_cups : ℕ := 2
  let kiera_muffins : ℕ := 2
  let kiera_fruit_cups : ℕ := 1
  let total_cost : ℝ := 17
  2

theorem muffin_cost_is_two :
  let fruit_cup_cost : ℝ := 3
  let francis_muffins : ℕ := 2
  let francis_fruit_cups : ℕ := 2
  let kiera_muffins : ℕ := 2
  let kiera_fruit_cups : ℕ := 1
  let total_cost : ℝ := 17
  muffin_cost = 2 ∧
  (francis_muffins + kiera_muffins : ℝ) * muffin_cost +
    (francis_fruit_cups + kiera_fruit_cups : ℝ) * fruit_cup_cost = total_cost :=
by
  sorry

end muffin_cost_is_two_l3133_313327


namespace expression_evaluation_l3133_313316

theorem expression_evaluation (b x : ℝ) (h : x = b + 4) :
  2*x - b + 5 = b + 13 := by
  sorry

end expression_evaluation_l3133_313316


namespace gcd_count_for_product_360_l3133_313351

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ), (∀ x ∈ S, ∃ c d : ℕ+, (Nat.gcd c d * Nat.lcm c d = 360 ∧ Nat.gcd c d = x)) ∧ 
                      (∀ y : ℕ, (∃ e f : ℕ+, (Nat.gcd e f * Nat.lcm e f = 360 ∧ Nat.gcd e f = y)) → y ∈ S) ∧
                      S.card = 12) := by
  sorry

end gcd_count_for_product_360_l3133_313351


namespace flea_treatment_ratio_l3133_313373

theorem flea_treatment_ratio (F : ℕ) (p : ℚ) : 
  F - 14 = 210 → 
  F * (1 - p)^4 = 14 → 
  ∃ (n : ℕ), n * (F * p) = F ∧ n = 448 := by
sorry

end flea_treatment_ratio_l3133_313373


namespace spring_fills_sixty_barrels_per_day_l3133_313335

/-- A spring fills barrels of water -/
structure Spring where
  fill_time : ℕ  -- Time to fill one barrel in minutes

/-- A day has a certain number of hours and minutes per hour -/
structure Day where
  hours : ℕ
  minutes_per_hour : ℕ

def barrels_filled_per_day (s : Spring) (d : Day) : ℕ :=
  (d.hours * d.minutes_per_hour) / s.fill_time

/-- Theorem: A spring that fills a barrel in 24 minutes will fill 60 barrels in a day -/
theorem spring_fills_sixty_barrels_per_day (s : Spring) (d : Day) :
  s.fill_time = 24 → d.hours = 24 → d.minutes_per_hour = 60 →
  barrels_filled_per_day s d = 60 := by
  sorry

end spring_fills_sixty_barrels_per_day_l3133_313335


namespace max_value_theorem_l3133_313366

theorem max_value_theorem (x y : ℝ) (h : x + y = 5) :
  ∃ (max : ℝ), max = 1175 / 16 ∧
  ∀ (a b : ℝ), a + b = 5 → a^3 * b + a^2 * b + a * b + a * b^2 ≤ max :=
by sorry

end max_value_theorem_l3133_313366


namespace joan_has_sixteen_seashells_l3133_313308

/-- The number of seashells Joan has after giving some to Mike -/
def joans_remaining_seashells (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

/-- Theorem: Joan has 16 seashells after giving Mike 63 of her initial 79 seashells -/
theorem joan_has_sixteen_seashells :
  joans_remaining_seashells 79 63 = 16 := by
  sorry

end joan_has_sixteen_seashells_l3133_313308


namespace arithmetic_sequence_ratio_l3133_313346

/-- Arithmetic sequence sum -/
def arithmetic_sum (a : ℕ → ℚ) (n : ℕ) : ℚ := (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n, S n = arithmetic_sum a n) →
  (∀ n, T n = arithmetic_sum b n) →
  (∀ n, S n / T n = (7 * n + 1) / (4 * n + 27)) →
  a 11 / b 11 = 4 / 3 := by
  sorry

end arithmetic_sequence_ratio_l3133_313346


namespace expression_evaluation_l3133_313378

theorem expression_evaluation : 
  (123 - (45 * (9 - 6) - 78)) + (0 / 1994) = 66 := by
  sorry

end expression_evaluation_l3133_313378


namespace system_solution_l3133_313305

/-- The system of equations:
    1. 3x² - xy = 1
    2. 9xy + y² = 22
    has exactly four solutions: (1,2), (-1,-2), (-1/6, 5.5), and (1/6, -5.5) -/
theorem system_solution :
  let f (x y : ℝ) := 3 * x^2 - x * y - 1
  let g (x y : ℝ) := 9 * x * y + y^2 - 22
  ∀ x y : ℝ, f x y = 0 ∧ g x y = 0 ↔
    (x = 1 ∧ y = 2) ∨
    (x = -1 ∧ y = -2) ∨
    (x = -1/6 ∧ y = 11/2) ∨
    (x = 1/6 ∧ y = -11/2) :=
by sorry

end system_solution_l3133_313305


namespace degree_of_polynomial_power_l3133_313398

/-- The degree of the polynomial (5x^3 + 7x + 2)^10 is 30. -/
theorem degree_of_polynomial_power (x : ℝ) : 
  Polynomial.degree ((5 * X^3 + 7 * X + 2 : Polynomial ℝ)^10) = 30 := by
  sorry

end degree_of_polynomial_power_l3133_313398


namespace point_on_extension_line_l3133_313380

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the points
variable (O A B P : V)

-- Define the conditions
variable (h_not_collinear : ¬Collinear ℝ {O, A, B})
variable (h_vector_equation : (2 : ℝ) • (P - O) = (2 : ℝ) • (A - O) + (2 : ℝ) • (B - O))

-- Theorem statement
theorem point_on_extension_line :
  ∃ (t : ℝ), t < 0 ∧ P = A + t • (B - A) :=
sorry

end point_on_extension_line_l3133_313380


namespace power_and_multiplication_l3133_313387

theorem power_and_multiplication : 2 * (3^2)^4 = 13122 := by
  sorry

end power_and_multiplication_l3133_313387


namespace equilateral_triangle_count_l3133_313340

/-- Represents a line in the coordinate plane --/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Generates horizontal lines y = k for k ∈ [-15, 15] --/
def horizontal_lines : List Line :=
  sorry

/-- Generates sloped lines y = √2x + 3k and y = -√2x + 3k for k ∈ [-15, 15] --/
def sloped_lines : List Line :=
  sorry

/-- All lines in the problem --/
def all_lines : List Line :=
  horizontal_lines ++ sloped_lines

/-- Predicate for an equilateral triangle with side length √2 --/
def is_unit_triangle (p q r : ℝ × ℝ) : Prop :=
  sorry

/-- Count of equilateral triangles formed by the intersection of lines --/
def triangle_count : ℕ :=
  sorry

/-- Main theorem stating the number of equilateral triangles formed --/
theorem equilateral_triangle_count :
  triangle_count = 12336 :=
sorry

end equilateral_triangle_count_l3133_313340


namespace friend_reading_time_l3133_313382

/-- Given a person who reads at half the speed of their friend and takes 4 hours to read a book,
    prove that their friend will take 120 minutes to read the same book. -/
theorem friend_reading_time (my_speed friend_speed : ℝ) (my_time friend_time : ℝ) :
  my_speed = (1/2) * friend_speed →
  my_time = 4 →
  friend_time = 2 →
  friend_time * 60 = 120 := by
  sorry

end friend_reading_time_l3133_313382


namespace bertha_family_females_without_daughters_l3133_313307

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  total_descendants : ℕ
  each_daughter_has_equal_children : Bool
  no_great_grandchildren : Bool

/-- Calculates the number of females with no daughters in Bertha's family -/
def females_without_daughters (family : BerthaFamily) : ℕ :=
  family.total_descendants - family.daughters

/-- Theorem stating that the number of females with no daughters in Bertha's family is 32 -/
theorem bertha_family_females_without_daughters :
  ∀ (family : BerthaFamily),
    family.daughters = 8 ∧
    family.total_descendants = 40 ∧
    family.each_daughter_has_equal_children = true ∧
    family.no_great_grandchildren = true →
    females_without_daughters family = 32 := by
  sorry

end bertha_family_females_without_daughters_l3133_313307


namespace white_squares_42nd_row_l3133_313385

/-- Represents the number of squares in a row of the stair-step figure -/
def squares_in_row (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of white squares in a row of the stair-step figure -/
def white_squares_in_row (n : ℕ) : ℕ := (squares_in_row n + 1) / 2

/-- Theorem stating the number of white squares in the 42nd row -/
theorem white_squares_42nd_row :
  white_squares_in_row 42 = 42 := by
  sorry

end white_squares_42nd_row_l3133_313385


namespace intersection_M_N_l3133_313324

def M : Set ℝ := {x : ℝ | |x + 1| ≤ 1}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0} := by
  sorry

end intersection_M_N_l3133_313324


namespace total_herd_count_l3133_313300

/-- Represents the number of animals in a shepherd's herd -/
structure Herd where
  count : ℕ

/-- Represents a shepherd with their herd -/
structure Shepherd where
  name : String
  herd : Herd

/-- The conditions of the problem -/
def exchange_conditions (jack jim dan : Shepherd) : Prop :=
  (jim.herd.count + 6 = 2 * (jack.herd.count - 1)) ∧
  (jack.herd.count + 14 = 3 * (dan.herd.count - 1)) ∧
  (dan.herd.count + 4 = 6 * (jim.herd.count - 1))

/-- The theorem to be proved -/
theorem total_herd_count (jack jim dan : Shepherd) :
  exchange_conditions jack jim dan →
  jack.herd.count + jim.herd.count + dan.herd.count = 39 := by
  sorry


end total_herd_count_l3133_313300


namespace floss_leftover_and_cost_l3133_313348

/-- Represents the floss requirements for a class -/
structure ClassFlossRequirement where
  students : ℕ
  flossPerStudent : ℚ

/-- Represents the floss sale conditions -/
structure FlossSaleConditions where
  metersPerPacket : ℚ
  pricePerPacket : ℚ
  discountRate : ℚ
  discountThreshold : ℕ

def yardToMeter : ℚ := 0.9144

def classes : List ClassFlossRequirement := [
  ⟨20, 1.5⟩,
  ⟨25, 1.75⟩,
  ⟨30, 2⟩
]

def saleConditions : FlossSaleConditions := {
  metersPerPacket := 50,
  pricePerPacket := 5,
  discountRate := 0.1,
  discountThreshold := 2
}

theorem floss_leftover_and_cost 
  (classes : List ClassFlossRequirement) 
  (saleConditions : FlossSaleConditions) 
  (yardToMeter : ℚ) : 
  ∃ (cost leftover : ℚ), cost = 14.5 ∧ leftover = 27.737 := by
  sorry

end floss_leftover_and_cost_l3133_313348


namespace f_properties_l3133_313381

def f (a x : ℝ) : ℝ := |x - 2*a| + |x + a|

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f 1 x ≥ 3) ∧ 
  (∃ x : ℝ, f 1 x = 3) ∧
  (∀ x : ℝ, a < 0 → f a x ≥ 5*a) ∧
  (∀ x : ℝ, a > 0 → (f a x ≥ 5*a ↔ (x ≤ -2*a ∨ x ≥ 3*a))) :=
sorry

end f_properties_l3133_313381


namespace concert_tickets_sold_l3133_313370

theorem concert_tickets_sold (T : ℕ) : 
  (3 / 4 : ℚ) * T + (5 / 9 : ℚ) * (1 / 4 : ℚ) * T + 80 + 20 = T → T = 900 := by
  sorry

end concert_tickets_sold_l3133_313370


namespace average_salary_after_bonuses_and_taxes_l3133_313361

def employee_salary (name : Char) : ℕ :=
  match name with
  | 'A' => 8000
  | 'B' => 5000
  | 'C' => 11000
  | 'D' => 7000
  | 'E' => 9000
  | 'F' => 6000
  | 'G' => 10000
  | _ => 0

def apply_bonus_or_tax (salary : ℕ) (rate : ℚ) (is_bonus : Bool) : ℚ :=
  if is_bonus then
    salary + salary * rate
  else
    salary - salary * rate

def final_salary (name : Char) : ℚ :=
  match name with
  | 'A' => apply_bonus_or_tax (employee_salary 'A') (1/10) true
  | 'B' => apply_bonus_or_tax (employee_salary 'B') (1/20) false
  | 'C' => employee_salary 'C'
  | 'D' => apply_bonus_or_tax (employee_salary 'D') (1/20) false
  | 'E' => apply_bonus_or_tax (employee_salary 'E') (3/100) false
  | 'F' => apply_bonus_or_tax (employee_salary 'F') (1/20) false
  | 'G' => apply_bonus_or_tax (employee_salary 'G') (3/40) true
  | _ => 0

def total_final_salaries : ℚ :=
  (final_salary 'A') + (final_salary 'B') + (final_salary 'C') +
  (final_salary 'D') + (final_salary 'E') + (final_salary 'F') +
  (final_salary 'G')

def number_of_employees : ℕ := 7

theorem average_salary_after_bonuses_and_taxes :
  (total_final_salaries / number_of_employees) = 8054.29 := by
  sorry

end average_salary_after_bonuses_and_taxes_l3133_313361


namespace remainder_3_pow_1999_mod_13_l3133_313332

theorem remainder_3_pow_1999_mod_13 : 3^1999 % 13 = 3 := by sorry

end remainder_3_pow_1999_mod_13_l3133_313332


namespace fraction_simplification_l3133_313334

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 200 + 3 * Real.sqrt 50 + 5) = (5 * Real.sqrt 2 - 1) / 49 := by
  sorry

end fraction_simplification_l3133_313334


namespace line_through_points_l3133_313358

/-- Given a line passing through points (-3, 1) and (1, 5) with equation y = mx + b, prove that m + b = 5 -/
theorem line_through_points (m b : ℝ) : 
  (1 = m * (-3) + b) → (5 = m * 1 + b) → m + b = 5 := by
  sorry

end line_through_points_l3133_313358


namespace parabola_vertex_l3133_313303

/-- A parabola defined by y = x^2 - 2ax + b passing through (1, 1) and intersecting the x-axis at only one point -/
structure Parabola where
  a : ℝ
  b : ℝ
  point_condition : 1 = 1^2 - 2*a*1 + b
  single_intersection : ∃! x, x^2 - 2*a*x + b = 0

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := (p.a, p.a^2 - p.b)

theorem parabola_vertex (p : Parabola) : vertex p = (0, 0) ∨ vertex p = (2, 0) := by
  sorry


end parabola_vertex_l3133_313303


namespace sum_of_coefficients_is_thirteen_thirds_l3133_313375

/-- Given two functions f and g, where f is linear and g(f(x)) = 4x + 2,
    prove that the sum of coefficients of f is 13/3 -/
theorem sum_of_coefficients_is_thirteen_thirds
  (f g : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x, f x = a * x + b)
  (h2 : ∀ x, g x = 3 * x - 7)
  (h3 : ∀ x, g (f x) = 4 * x + 2) :
  a + b = 13 / 3 := by
  sorry

end sum_of_coefficients_is_thirteen_thirds_l3133_313375


namespace scaling_transformation_result_l3133_313390

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y = 3 * Real.sin (2 * x)

-- Define the scaling transformation
def scaling_transformation (x y x' y' : ℝ) : Prop := x' = 2 * x ∧ y' = 3 * y

-- State the theorem
theorem scaling_transformation_result :
  ∀ (x y x' y' : ℝ),
  original_curve x y →
  scaling_transformation x y x' y' →
  y' = 9 * Real.sin x' := by sorry

end scaling_transformation_result_l3133_313390


namespace drink_composition_l3133_313357

theorem drink_composition (coke sprite mountain_dew : ℕ) 
  (h1 : coke = 2)
  (h2 : sprite = 1)
  (h3 : mountain_dew = 3)
  (h4 : (6 : ℚ) / (coke / (coke + sprite + mountain_dew)) = 18) :
  (6 : ℚ) / ((coke : ℚ) / (coke + sprite + mountain_dew)) = 18 := by
  sorry

end drink_composition_l3133_313357


namespace knights_count_l3133_313372

/-- Represents an islander, who can be either a knight or a liar -/
inductive Islander
| Knight
| Liar

/-- The total number of islanders -/
def total_islanders : Nat := 6

/-- Determines if an islander's statement is true based on the actual number of liars -/
def statement_is_true (actual_liars : Nat) : Prop :=
  actual_liars = 4

/-- Determines if an islander's behavior is consistent with their type and statement -/
def is_consistent (islander : Islander) (actual_liars : Nat) : Prop :=
  match islander with
  | Islander.Knight => statement_is_true actual_liars
  | Islander.Liar => ¬statement_is_true actual_liars

/-- The main theorem to prove -/
theorem knights_count :
  ∀ (knights : Nat),
    (knights ≤ total_islanders) →
    (∀ i : Fin total_islanders,
      is_consistent
        (if i.val < knights then Islander.Knight else Islander.Liar)
        (total_islanders - knights - 1)) →
    (knights = 0 ∨ knights = 2) :=
by sorry


end knights_count_l3133_313372


namespace square_difference_601_597_l3133_313325

theorem square_difference_601_597 : 601^2 - 597^2 = 4792 := by
  sorry

end square_difference_601_597_l3133_313325


namespace intersection_A_B_union_B_complement_A_l3133_313383

-- Define the universal set U
def U : Set ℝ := {x | -5 ≤ x ∧ x ≤ 5}

-- Define set A
def A : Set ℝ := {x ∈ U | 0 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x ∈ U | -2 ≤ x ∧ x ≤ 1}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x ∈ U | 0 < x ∧ x ≤ 1} := by sorry

-- Theorem for B ∪ (ᶜA)
theorem union_B_complement_A : B ∪ (U \ A) = {x ∈ U | x ≤ 1 ∨ 3 < x} := by sorry

end intersection_A_B_union_B_complement_A_l3133_313383


namespace shortest_path_on_parallelepiped_l3133_313318

/-- The shortest path on the surface of a rectangular parallelepiped -/
theorem shortest_path_on_parallelepiped (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  let surface_paths := [
    Real.sqrt ((a + c + a)^2 + b^2),
    Real.sqrt ((a + b + a)^2 + c^2),
    Real.sqrt ((b + a + b)^2 + c^2)
  ]
  ∃ (path : ℝ), path ∈ surface_paths ∧ path = Real.sqrt 125 ∧ ∀ x ∈ surface_paths, path ≤ x :=
by sorry

end shortest_path_on_parallelepiped_l3133_313318
