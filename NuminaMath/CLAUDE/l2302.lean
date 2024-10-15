import Mathlib

namespace NUMINAMATH_CALUDE_cost_difference_l2302_230273

/-- The price difference between two types of candy in kopecks per kilogram -/
def price_difference : ℕ := 80

/-- The total amount of candy bought by each person in grams -/
def total_amount : ℕ := 150

/-- The cost of Andrey's purchase in kopecks -/
def andrey_cost (x : ℕ) : ℚ :=
  (150 * x + 8000 : ℚ) / 1000

/-- The cost of Yura's purchase in kopecks -/
def yura_cost (x : ℕ) : ℚ :=
  (150 * x + 6000 : ℚ) / 1000

/-- The theorem stating the difference in cost between Andrey's and Yura's purchases -/
theorem cost_difference (x : ℕ) :
  andrey_cost x - yura_cost x = 2 / 1000 := by sorry

end NUMINAMATH_CALUDE_cost_difference_l2302_230273


namespace NUMINAMATH_CALUDE_angle_range_in_scalene_triangle_l2302_230296

-- Define a scalene triangle
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_triangle : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the theorem
theorem angle_range_in_scalene_triangle (t : ScaleneTriangle) 
  (h_longest : t.a ≥ t.b ∧ t.a ≥ t.c) 
  (h_inequality : t.a^2 < t.b^2 + t.c^2) :
  let A := Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))
  60 * π / 180 < A ∧ A < 90 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_range_in_scalene_triangle_l2302_230296


namespace NUMINAMATH_CALUDE_smallest_n_theorem_l2302_230272

/-- The smallest positive integer n for which the equation 15x^2 - nx + 630 = 0 has integral solutions -/
def smallest_n : ℕ := 195

/-- The equation 15x^2 - nx + 630 = 0 has integral solutions -/
def has_integral_solutions (n : ℕ) : Prop :=
  ∃ x : ℤ, 15 * x^2 - n * x + 630 = 0

theorem smallest_n_theorem :
  (has_integral_solutions smallest_n) ∧
  (∀ m : ℕ, m < smallest_n → ¬(has_integral_solutions m)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_theorem_l2302_230272


namespace NUMINAMATH_CALUDE_number_of_arrangements_l2302_230258

/-- Represents a person in the group photo --/
inductive Person
  | StudentA
  | StudentB
  | StudentC
  | StudentD
  | StudentE
  | TeacherX
  | TeacherY

/-- Represents a valid arrangement of people in the group photo --/
def ValidArrangement : Type := List Person

/-- Checks if students A, B, and C are standing together in the arrangement --/
def studentsABCTogether (arrangement : ValidArrangement) : Prop := sorry

/-- Checks if teachers X and Y are not standing next to each other in the arrangement --/
def teachersNotAdjacent (arrangement : ValidArrangement) : Prop := sorry

/-- The set of all valid arrangements satisfying the given conditions --/
def validArrangements : Set ValidArrangement :=
  {arrangement | studentsABCTogether arrangement ∧ teachersNotAdjacent arrangement}

/-- The main theorem stating that the number of valid arrangements is 504 --/
theorem number_of_arrangements (h : Fintype validArrangements) :
  Fintype.card validArrangements = 504 := by sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l2302_230258


namespace NUMINAMATH_CALUDE_aquarium_length_l2302_230232

theorem aquarium_length (L : ℝ) : 
  L > 0 → 
  3 * (1/4 * L * 6 * 3) = 54 → 
  L = 4 := by
sorry

end NUMINAMATH_CALUDE_aquarium_length_l2302_230232


namespace NUMINAMATH_CALUDE_awards_distribution_l2302_230278

/-- The number of ways to distribute n distinct awards to k students,
    where each student receives at least one award. -/
def distribute_awards (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n distinct items. -/
def choose (n r : ℕ) : ℕ := sorry

theorem awards_distribution :
  distribute_awards 5 4 = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_awards_distribution_l2302_230278


namespace NUMINAMATH_CALUDE_patrick_current_age_l2302_230208

/-- Patrick's age is half of Robert's age -/
def patrick_age_relation (patrick_age robert_age : ℕ) : Prop :=
  patrick_age = robert_age / 2

/-- Robert will be 30 years old in 2 years -/
def robert_future_age (robert_age : ℕ) : Prop :=
  robert_age + 2 = 30

/-- The theorem stating Patrick's current age -/
theorem patrick_current_age :
  ∃ (patrick_age robert_age : ℕ),
    patrick_age_relation patrick_age robert_age ∧
    robert_future_age robert_age ∧
    patrick_age = 14 := by
  sorry

end NUMINAMATH_CALUDE_patrick_current_age_l2302_230208


namespace NUMINAMATH_CALUDE_candied_yams_order_l2302_230293

theorem candied_yams_order (total_shoppers : ℕ) (buying_frequency : ℕ) (packages_per_box : ℕ) : 
  total_shoppers = 375 →
  buying_frequency = 3 →
  packages_per_box = 25 →
  (total_shoppers / buying_frequency) / packages_per_box = 5 := by
  sorry

end NUMINAMATH_CALUDE_candied_yams_order_l2302_230293


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_3_dividing_27_factorial_l2302_230225

/-- The largest power of 3 that divides 27! -/
def largest_power_of_3 : ℕ := 13

/-- The ones digit of 3^n -/
def ones_digit_of_3_power (n : ℕ) : ℕ :=
  (3^n) % 10

theorem ones_digit_of_largest_power_of_3_dividing_27_factorial :
  ones_digit_of_3_power largest_power_of_3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_3_dividing_27_factorial_l2302_230225


namespace NUMINAMATH_CALUDE_second_trial_point_theorem_l2302_230209

/-- Represents the fractional method for optimization experiments -/
structure FractionalMethod where
  range_start : ℝ
  range_end : ℝ
  rounds : ℕ

/-- Calculates the possible second trial points for the fractional method -/
def second_trial_points (fm : FractionalMethod) : Set ℝ :=
  let interval_length := fm.range_end - fm.range_start
  let num_divisions := 2^fm.rounds
  let step := interval_length / num_divisions
  {fm.range_start + 3 * step, fm.range_end - 3 * step}

/-- Theorem stating that for the given experimental setup, 
    the second trial point is either 40 or 60 -/
theorem second_trial_point_theorem (fm : FractionalMethod) 
  (h1 : fm.range_start = 10) 
  (h2 : fm.range_end = 90) 
  (h3 : fm.rounds = 4) : 
  second_trial_points fm = {40, 60} := by
  sorry

end NUMINAMATH_CALUDE_second_trial_point_theorem_l2302_230209


namespace NUMINAMATH_CALUDE_solve_equation_l2302_230221

theorem solve_equation (x : ℝ) (h : 0.009 / x = 0.01) : x = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2302_230221


namespace NUMINAMATH_CALUDE_museum_visitors_theorem_l2302_230242

/-- Represents the inverse proportional relationship between visitors and ticket price -/
def inverse_proportion (v t k : ℝ) : Prop := v * t = k

/-- Given conditions of the problem -/
def museum_conditions (v₁ v₂ t₁ t₂ k : ℝ) : Prop :=
  t₁ = 20 ∧ v₁ = 150 ∧ t₂ = 30 ∧
  inverse_proportion v₁ t₁ k ∧
  inverse_proportion v₂ t₂ k

/-- Theorem statement -/
theorem museum_visitors_theorem (v₁ v₂ t₁ t₂ k : ℝ) :
  museum_conditions v₁ v₂ t₁ t₂ k → v₂ = 100 := by
  sorry

end NUMINAMATH_CALUDE_museum_visitors_theorem_l2302_230242


namespace NUMINAMATH_CALUDE_harry_sister_stamp_ratio_l2302_230248

/-- Proves the ratio of Harry's stamps to his sister's stamps -/
theorem harry_sister_stamp_ratio :
  let total_stamps : ℕ := 240
  let sister_stamps : ℕ := 60
  let harry_stamps : ℕ := total_stamps - sister_stamps
  (harry_stamps : ℚ) / sister_stamps = 3 := by
  sorry

end NUMINAMATH_CALUDE_harry_sister_stamp_ratio_l2302_230248


namespace NUMINAMATH_CALUDE_number_of_digits_c_l2302_230243

theorem number_of_digits_c (a b c : ℕ) : 
  a < b → b < c → 
  (b + a) % (b - a) = 0 → 
  (c + b) % (c - b) = 0 → 
  a ≥ 10^2010 → a < 10^2011 →
  b ≥ 10^2011 → b < 10^2012 →
  c ≥ 10^4 ∧ c < 10^5 := by sorry

end NUMINAMATH_CALUDE_number_of_digits_c_l2302_230243


namespace NUMINAMATH_CALUDE_max_value_of_five_numbers_l2302_230260

theorem max_value_of_five_numbers (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- distinct and ordered
  (a + b + c + d + e) / 5 = 12 →  -- average is 12
  c = 17 →  -- median is 17
  e ≤ 24 :=  -- maximum possible value is 24
by sorry

end NUMINAMATH_CALUDE_max_value_of_five_numbers_l2302_230260


namespace NUMINAMATH_CALUDE_division_problem_l2302_230247

theorem division_problem : ∃ (dividend : Nat) (divisor : Nat),
  dividend = 10004678 ∧ 
  divisor = 142 ∧ 
  100 ≤ divisor ∧ 
  divisor < 1000 ∧
  10000000 ≤ dividend ∧ 
  dividend < 100000000 ∧
  dividend / divisor = 70709 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2302_230247


namespace NUMINAMATH_CALUDE_second_set_amount_l2302_230203

def total_spent : ℝ := 900
def first_set : ℝ := 325
def last_set : ℝ := 315

theorem second_set_amount :
  total_spent - first_set - last_set = 260 := by sorry

end NUMINAMATH_CALUDE_second_set_amount_l2302_230203


namespace NUMINAMATH_CALUDE_fraction_conforms_to_standard_notation_l2302_230222

/-- Rules for standard algebraic notation -/
structure AlgebraicNotationRules where
  no_multiplication_sign : Bool
  mixed_numbers_as_fractions : Bool
  division_as_fraction : Bool

/-- An algebraic expression -/
inductive AlgebraicExpression
  | Multiply : ℕ → Char → AlgebraicExpression
  | MixedNumber : ℕ → ℚ → Char → AlgebraicExpression
  | Fraction : Char → Char → ℕ → AlgebraicExpression
  | Divide : Char → ℕ → Char → AlgebraicExpression

/-- Function to check if an expression conforms to standard algebraic notation -/
def conforms_to_standard_notation (rules : AlgebraicNotationRules) (expr : AlgebraicExpression) : Prop :=
  match expr with
  | AlgebraicExpression.Fraction _ _ _ => true
  | _ => false

/-- Theorem stating that -b/a² conforms to standard algebraic notation -/
theorem fraction_conforms_to_standard_notation (rules : AlgebraicNotationRules) :
  conforms_to_standard_notation rules (AlgebraicExpression.Fraction 'b' 'a' 2) :=
sorry

end NUMINAMATH_CALUDE_fraction_conforms_to_standard_notation_l2302_230222


namespace NUMINAMATH_CALUDE_solve_installment_problem_l2302_230252

def installment_problem (cash_price : ℕ) (down_payment : ℕ) (first_four_payment : ℕ) (next_four_payment : ℕ) (total_months : ℕ) (installment_markup : ℕ) : Prop :=
  let total_installment_price := cash_price + installment_markup
  let paid_so_far := down_payment + 4 * first_four_payment + 4 * next_four_payment
  let remaining_amount := total_installment_price - paid_so_far
  let last_four_months := total_months - 8
  remaining_amount / last_four_months = 30

theorem solve_installment_problem :
  installment_problem 450 100 40 35 12 70 :=
sorry

end NUMINAMATH_CALUDE_solve_installment_problem_l2302_230252


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l2302_230230

theorem maintenance_check_increase (original_time new_time : ℝ) 
  (h1 : original_time = 25)
  (h2 : new_time = 30) :
  (new_time - original_time) / original_time * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l2302_230230


namespace NUMINAMATH_CALUDE_track_width_l2302_230292

theorem track_width (r₁ r₂ : ℝ) (h : r₁ > r₂) :
  2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi →
  r₁ - r₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_track_width_l2302_230292


namespace NUMINAMATH_CALUDE_unique_n_with_special_divisors_l2302_230256

def isDivisor (d n : ℕ) : Prop := d ∣ n

def divisors (n : ℕ) : Set ℕ := {d : ℕ | isDivisor d n}

theorem unique_n_with_special_divisors :
  ∃! n : ℕ, n > 0 ∧
  ∃ (d₂ d₃ : ℕ), d₂ ∈ divisors n ∧ d₃ ∈ divisors n ∧
  1 < d₂ ∧ d₂ < d₃ ∧
  n = d₂^2 + d₃^3 ∧
  ∀ d ∈ divisors n, d = 1 ∨ d ≥ d₂ :=
by
  sorry

end NUMINAMATH_CALUDE_unique_n_with_special_divisors_l2302_230256


namespace NUMINAMATH_CALUDE_seohyeon_distance_longer_l2302_230205

/-- Proves that Seohyeon's distance to school is longer than Kunwoo's. -/
theorem seohyeon_distance_longer (kunwoo_distance : ℝ) (seohyeon_distance : ℝ) 
  (h1 : kunwoo_distance = 3.97) 
  (h2 : seohyeon_distance = 4028) : 
  seohyeon_distance > kunwoo_distance * 1000 :=
by
  sorry

#check seohyeon_distance_longer

end NUMINAMATH_CALUDE_seohyeon_distance_longer_l2302_230205


namespace NUMINAMATH_CALUDE_average_and_product_problem_l2302_230213

theorem average_and_product_problem (x y : ℝ) : 
  (10 + 25 + x + y) / 4 = 20 →
  x * y = 156 →
  ((x = 12 ∧ y = 33) ∨ (x = 33 ∧ y = 12)) :=
by sorry

end NUMINAMATH_CALUDE_average_and_product_problem_l2302_230213


namespace NUMINAMATH_CALUDE_dish_temperature_l2302_230216

/-- Calculates the final temperature of a dish in an oven -/
def final_temperature (start_temp : ℝ) (heating_rate : ℝ) (cooking_time : ℝ) : ℝ :=
  start_temp + heating_rate * cooking_time

/-- Proves that the dish reaches 100 degrees given the specified conditions -/
theorem dish_temperature : final_temperature 20 5 16 = 100 := by
  sorry

end NUMINAMATH_CALUDE_dish_temperature_l2302_230216


namespace NUMINAMATH_CALUDE_binary_11011_equals_27_l2302_230214

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 11011₂ -/
def binary_11011 : List Bool := [true, true, false, true, true]

theorem binary_11011_equals_27 :
  binary_to_decimal binary_11011 = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_equals_27_l2302_230214


namespace NUMINAMATH_CALUDE_general_equation_proof_l2302_230295

theorem general_equation_proof (n : ℝ) (h1 : n ≠ 4) (h2 : n ≠ 8) :
  n / (n - 4) + (8 - n) / ((8 - n) - 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_general_equation_proof_l2302_230295


namespace NUMINAMATH_CALUDE_work_time_ratio_l2302_230262

/-- Proves that the ratio of Celeste's work time to Bianca's work time is 2:1 given the specified conditions. -/
theorem work_time_ratio (bianca_time : ℝ) (celeste_multiplier : ℝ) :
  bianca_time = 12.5 →
  bianca_time * celeste_multiplier + (bianca_time * celeste_multiplier - 8.5) + bianca_time = 54 →
  celeste_multiplier = 2 := by
  sorry

#check work_time_ratio

end NUMINAMATH_CALUDE_work_time_ratio_l2302_230262


namespace NUMINAMATH_CALUDE_fourth_root_equivalence_l2302_230224

theorem fourth_root_equivalence (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 * y^(1/3))^(1/4) = x^(1/2) * y^(1/12) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equivalence_l2302_230224


namespace NUMINAMATH_CALUDE_girls_tryout_count_l2302_230264

theorem girls_tryout_count (boys : ℕ) (called_back : ℕ) (didnt_make_cut : ℕ) :
  boys = 4 →
  called_back = 26 →
  didnt_make_cut = 17 →
  ∃ girls : ℕ, girls + boys = called_back + didnt_make_cut ∧ girls = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_tryout_count_l2302_230264


namespace NUMINAMATH_CALUDE_greatest_common_divisor_with_same_remainder_l2302_230239

theorem greatest_common_divisor_with_same_remainder (a b c : ℕ) (ha : a = 54) (hb : b = 87) (hc : c = 172) :
  ∃ (d : ℕ), d > 0 ∧ 
  (∃ (r : ℕ), a % d = r ∧ b % d = r ∧ c % d = r) ∧
  (∀ (k : ℕ), k > d → ¬(∃ (s : ℕ), a % k = s ∧ b % k = s ∧ c % k = s)) →
  d = 1 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_with_same_remainder_l2302_230239


namespace NUMINAMATH_CALUDE_problem_solution_l2302_230277

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - x - a = 0}
def B : Set ℝ := {2, -5}

-- Define the theorem
theorem problem_solution :
  ∃ (a : ℝ),
    (2 ∈ A a) ∧
    (a = 2) ∧
    (A a = {-1, 2}) ∧
    (let U := A a ∪ B;
     U = {-5, -1, 2} ∧
     (U \ A a) ∪ (U \ B) = {-5, -1}) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l2302_230277


namespace NUMINAMATH_CALUDE_power_mod_seven_l2302_230204

theorem power_mod_seven : 3^255 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_power_mod_seven_l2302_230204


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2302_230234

theorem expand_and_simplify (y : ℝ) (h : y ≠ 0) :
  (3 / 4) * ((8 / y) - 6 * y^2 + 3 * y) = 6 / y - (9 * y^2) / 2 + (9 * y) / 4 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2302_230234


namespace NUMINAMATH_CALUDE_polygon_similarity_nesting_l2302_230249

-- Define polygons
variable (Polygon : Type)

-- Define similarity relation between polygons
variable (similar : Polygon → Polygon → Prop)

-- Define nesting relation between polygons
variable (nesting : Polygon → Polygon → Prop)

-- Main theorem
theorem polygon_similarity_nesting 
  (p q : Polygon) : 
  (¬ similar p q) ↔ 
  (∃ r : Polygon, similar r q ∧ ¬ nesting r p) :=
sorry

end NUMINAMATH_CALUDE_polygon_similarity_nesting_l2302_230249


namespace NUMINAMATH_CALUDE_product_of_sums_l2302_230270

theorem product_of_sums (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  a * b + a + b = 99 ∧ b * c + b + c = 99 ∧ c * a + c + a = 99 →
  (a + 1) * (b + 1) * (c + 1) = 1000 := by
sorry

end NUMINAMATH_CALUDE_product_of_sums_l2302_230270


namespace NUMINAMATH_CALUDE_student_weight_l2302_230228

/-- Given two people, a student and his sister, prove that the student's weight is 60 kg
    under the following conditions:
    1. If the student loses 5 kg, he will weigh 25% more than his sister.
    2. Together, they now weigh 104 kg. -/
theorem student_weight (student_weight sister_weight : ℝ) : 
  (student_weight - 5 = 1.25 * sister_weight) →
  (student_weight + sister_weight = 104) →
  student_weight = 60 := by
  sorry

#check student_weight

end NUMINAMATH_CALUDE_student_weight_l2302_230228


namespace NUMINAMATH_CALUDE_milk_consumption_l2302_230255

/-- The amount of regular milk consumed by Mitch's family in 1 week -/
def regular_milk : ℝ := 0.5

/-- The amount of soy milk consumed by Mitch's family in 1 week -/
def soy_milk : ℝ := 0.1

/-- The total amount of milk consumed by Mitch's family in 1 week -/
def total_milk : ℝ := regular_milk + soy_milk

theorem milk_consumption :
  total_milk = 0.6 := by sorry

end NUMINAMATH_CALUDE_milk_consumption_l2302_230255


namespace NUMINAMATH_CALUDE_raisin_mixture_l2302_230206

theorem raisin_mixture (raisin_pounds : ℝ) (nut_pounds : ℝ) (raisin_cost : ℝ) (nut_cost : ℝ) :
  nut_pounds = 4 ∧
  nut_cost = 3 * raisin_cost ∧
  raisin_pounds * raisin_cost = 0.25 * (raisin_pounds * raisin_cost + nut_pounds * nut_cost) →
  raisin_pounds = 4 := by
sorry

end NUMINAMATH_CALUDE_raisin_mixture_l2302_230206


namespace NUMINAMATH_CALUDE_triangle_probability_l2302_230281

theorem triangle_probability (total_figures : ℕ) (triangle_count : ℕ) 
  (h1 : total_figures = 8) (h2 : triangle_count = 3) :
  (triangle_count : ℚ) / total_figures = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_probability_l2302_230281


namespace NUMINAMATH_CALUDE_painter_paintings_l2302_230282

/-- Given a painter who makes a certain number of paintings per day and already has some paintings,
    calculate the total number of paintings after a given number of days. -/
def total_paintings (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  paintings_per_day * days + initial_paintings

/-- Theorem: A painter who makes 2 paintings per day and already has 20 paintings
    will have 80 paintings in total after 30 days. -/
theorem painter_paintings : total_paintings 2 20 30 = 80 := by
  sorry

end NUMINAMATH_CALUDE_painter_paintings_l2302_230282


namespace NUMINAMATH_CALUDE_basketball_purchase_theorem_l2302_230257

/-- Represents the prices and quantities of basketballs --/
structure BasketballPurchase where
  priceA : ℕ  -- Price of brand A basketball
  priceB : ℕ  -- Price of brand B basketball
  quantityA : ℕ  -- Quantity of brand A basketballs
  quantityB : ℕ  -- Quantity of brand B basketballs

/-- Represents the conditions of the basketball purchase problem --/
def BasketballProblem (p : BasketballPurchase) : Prop :=
  p.priceB = p.priceA + 40 ∧
  4800 / p.priceA = (3/2) * (4000 / p.priceB) ∧
  p.quantityA + p.quantityB = 90 ∧
  p.quantityB ≥ 2 * p.quantityA ∧
  p.priceA * p.quantityA + p.priceB * p.quantityB ≤ 17200

/-- The theorem to be proved --/
theorem basketball_purchase_theorem (p : BasketballPurchase) 
  (h : BasketballProblem p) : 
  p.priceA = 160 ∧ 
  p.priceB = 200 ∧ 
  (∃ n : ℕ, n = 11 ∧ 
    ∀ m : ℕ, (20 ≤ m ∧ m ≤ 30) ↔ 
      BasketballProblem ⟨p.priceA, p.priceB, m, 90 - m⟩) ∧
  (∀ a : ℕ, 30 < a ∧ a < 50 → 
    (a < 40 → p.quantityA = 30) ∧ 
    (a > 40 → p.quantityA = 20)) :=
sorry


end NUMINAMATH_CALUDE_basketball_purchase_theorem_l2302_230257


namespace NUMINAMATH_CALUDE_odd_function_properties_l2302_230212

noncomputable def f (a b x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

theorem odd_function_properties (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  (a = 2 ∧ b = 1) ∧
  (∀ x y : ℝ, x < y → f 2 1 x > f 2 1 y) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ≥ 1 → f 2 1 (k * 3^x) + f 2 1 (3^x - 9^x + 2) > 0) ↔ k < 4/3) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l2302_230212


namespace NUMINAMATH_CALUDE_intersection_dot_product_l2302_230246

-- Define the line l: 4x + 3y - 5 = 0
def line_l (x y : ℝ) : Prop := 4 * x + 3 * y - 5 = 0

-- Define the circle C: x² + y² - 4 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0

-- Define the intersection points A and B
def is_intersection (x y : ℝ) : Prop := line_l x y ∧ circle_C x y

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem intersection_dot_product :
  ∃ (A B : ℝ × ℝ),
    is_intersection A.1 A.2 ∧
    is_intersection B.1 B.2 ∧
    A ≠ B ∧
    (A.1 * B.1 + A.2 * B.2 = -2) :=
sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l2302_230246


namespace NUMINAMATH_CALUDE_luncheon_cost_theorem_l2302_230240

/-- The cost of items in a luncheon -/
structure LuncheonCost where
  sandwich : ℝ
  coffee : ℝ
  pie : ℝ
  cookie : ℝ

/-- Given conditions of the luncheon costs -/
def luncheon_conditions (cost : LuncheonCost) : Prop :=
  5 * cost.sandwich + 9 * cost.coffee + 2 * cost.pie + 3 * cost.cookie = 5.85 ∧
  6 * cost.sandwich + 12 * cost.coffee + 2 * cost.pie + 4 * cost.cookie = 7.20

/-- Theorem stating the cost of one of each item -/
theorem luncheon_cost_theorem (cost : LuncheonCost) :
  luncheon_conditions cost →
  cost.sandwich + cost.coffee + cost.pie + cost.cookie = 1.35 :=
by sorry

end NUMINAMATH_CALUDE_luncheon_cost_theorem_l2302_230240


namespace NUMINAMATH_CALUDE_house_size_problem_l2302_230285

theorem house_size_problem (sara_house nada_house : ℝ) : 
  sara_house = 1000 ∧ 
  sara_house = 2 * nada_house + 100 → 
  nada_house = 450 := by
sorry

end NUMINAMATH_CALUDE_house_size_problem_l2302_230285


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2302_230253

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let roots := {x : ℝ | f x = 0}
  (∃ x y : ℝ, x ∈ roots ∧ y ∈ roots ∧ x ≠ y) →
  (∀ z : ℝ, z ∈ roots → z = x ∨ z = y) →
  x + y = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 - 15 * x + 20
  let roots := {x : ℝ | f x = 0}
  ∃ C D : ℝ, C ∈ roots ∧ D ∈ roots ∧ C ≠ D ∧
    (∀ z : ℝ, z ∈ roots → z = C ∨ z = D) ∧
    C + D = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2302_230253


namespace NUMINAMATH_CALUDE_population_average_age_l2302_230265

/-- Proves that given a population with a specific ratio of women to men and their respective average ages, the average age of the entire population can be calculated. -/
theorem population_average_age 
  (total_population : ℕ) 
  (women_ratio : ℚ) 
  (men_ratio : ℚ) 
  (women_avg_age : ℚ) 
  (men_avg_age : ℚ) 
  (h1 : women_ratio + men_ratio = 1) 
  (h2 : women_ratio = 11 / 21) 
  (h3 : men_ratio = 10 / 21) 
  (h4 : women_avg_age = 34) 
  (h5 : men_avg_age = 32) : 
  (women_ratio * women_avg_age + men_ratio * men_avg_age : ℚ) = 33 + 1 / 21 := by
  sorry

#check population_average_age

end NUMINAMATH_CALUDE_population_average_age_l2302_230265


namespace NUMINAMATH_CALUDE_boys_clay_maple_basketball_l2302_230259

/-- Represents a school in the sports camp -/
inductive School
| Jonas
| Clay
| Maple

/-- Represents an activity in the sports camp -/
inductive Activity
| Basketball
| Swimming

/-- Represents the gender of a student -/
inductive Gender
| Boy
| Girl

/-- Data about the sports camp -/
structure SportsData where
  total_students : ℕ
  total_boys : ℕ
  total_girls : ℕ
  jonas_students : ℕ
  clay_students : ℕ
  maple_students : ℕ
  jonas_boys : ℕ
  swimming_girls : ℕ
  clay_swimming_boys : ℕ

/-- Theorem stating the number of boys from Clay and Maple who attended basketball -/
theorem boys_clay_maple_basketball (data : SportsData)
  (h1 : data.total_students = 120)
  (h2 : data.total_boys = 70)
  (h3 : data.total_girls = 50)
  (h4 : data.jonas_students = 50)
  (h5 : data.clay_students = 40)
  (h6 : data.maple_students = 30)
  (h7 : data.jonas_boys = 28)
  (h8 : data.swimming_girls = 16)
  (h9 : data.clay_swimming_boys = 10) :
  (data.total_boys - data.jonas_boys - data.clay_swimming_boys) = 30 := by
  sorry

end NUMINAMATH_CALUDE_boys_clay_maple_basketball_l2302_230259


namespace NUMINAMATH_CALUDE_smallest_positive_shift_l2302_230251

-- Define a function f with period 20
def f : ℝ → ℝ := sorry

-- Define the periodicity property
axiom f_periodic : ∀ x : ℝ, f (x - 20) = f x

-- Define the property for the scaled and shifted function
def scaled_shifted_property (a : ℝ) : Prop :=
  ∀ x : ℝ, f ((x - a) / 4) = f (x / 4)

-- Theorem statement
theorem smallest_positive_shift :
  ∃ a : ℝ, a > 0 ∧ scaled_shifted_property a ∧
  ∀ b : ℝ, b > 0 ∧ scaled_shifted_property b → a ≤ b :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_shift_l2302_230251


namespace NUMINAMATH_CALUDE_total_rainfall_proof_l2302_230299

/-- Given rainfall data for three days, proves the total rainfall. -/
theorem total_rainfall_proof (sunday_rain : ℝ) (monday_rain : ℝ) (tuesday_rain : ℝ)
  (h1 : sunday_rain = 4)
  (h2 : monday_rain = sunday_rain + 3)
  (h3 : tuesday_rain = 2 * monday_rain) :
  sunday_rain + monday_rain + tuesday_rain = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_proof_l2302_230299


namespace NUMINAMATH_CALUDE_initial_bureaus_correct_l2302_230250

/-- The number of offices -/
def num_offices : ℕ := 14

/-- The additional bureaus needed for equal distribution -/
def additional_bureaus : ℕ := 10

/-- The initial number of bureaus -/
def initial_bureaus : ℕ := 8

/-- Theorem stating that the initial number of bureaus is correct -/
theorem initial_bureaus_correct :
  ∃ (x : ℕ), (initial_bureaus + additional_bureaus = num_offices * x) ∧
             (∀ y : ℕ, initial_bureaus ≠ num_offices * y) :=
by sorry

end NUMINAMATH_CALUDE_initial_bureaus_correct_l2302_230250


namespace NUMINAMATH_CALUDE_max_time_digit_sum_l2302_230288

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given time -/
def timeDigitSum (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The theorem stating the maximum sum of digits in a 24-hour time display -/
theorem max_time_digit_sum :
  (∃ (t : Time24), ∀ (t' : Time24), timeDigitSum t' ≤ timeDigitSum t) ∧
  (∀ (t : Time24), timeDigitSum t ≤ 24) :=
sorry

end NUMINAMATH_CALUDE_max_time_digit_sum_l2302_230288


namespace NUMINAMATH_CALUDE_clerical_staff_percentage_l2302_230235

def total_employees : ℕ := 3600
def initial_clerical_ratio : ℚ := 1 / 6
def clerical_reduction_ratio : ℚ := 1 / 4

theorem clerical_staff_percentage : 
  let initial_clerical := (initial_clerical_ratio * total_employees : ℚ)
  let reduced_clerical := initial_clerical - (clerical_reduction_ratio * initial_clerical)
  let remaining_employees := total_employees - (initial_clerical - reduced_clerical)
  (reduced_clerical / remaining_employees) * 100 = 450 / 3450 * 100 := by
  sorry

end NUMINAMATH_CALUDE_clerical_staff_percentage_l2302_230235


namespace NUMINAMATH_CALUDE_factorization_equality_l2302_230269

theorem factorization_equality (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2302_230269


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2302_230226

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |2*x - 5| = 3*x - 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2302_230226


namespace NUMINAMATH_CALUDE_cube_sum_minus_product_2003_l2302_230286

theorem cube_sum_minus_product_2003 :
  {(x, y, z) : ℤ × ℤ × ℤ | x^3 + y^3 + z^3 - 3*x*y*z = 2003} =
  {(668, 668, 667), (668, 667, 668), (667, 668, 668)} := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_minus_product_2003_l2302_230286


namespace NUMINAMATH_CALUDE_original_speed_B_l2302_230283

/-- Two people traveling towards each other -/
structure TravelScenario where
  speed_A : ℝ
  speed_B : ℝ

/-- The condition that the meeting point remains the same after speed changes -/
def meeting_point_unchanged (s : TravelScenario) : Prop :=
  s.speed_A / s.speed_B = (5/4 * s.speed_A) / (s.speed_B + 10)

/-- The theorem stating that if the meeting point is unchanged, B's original speed is 40 km/h -/
theorem original_speed_B (s : TravelScenario) :
  meeting_point_unchanged s → s.speed_B = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_speed_B_l2302_230283


namespace NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l2302_230241

theorem range_of_a_for_quadratic_inequality :
  {a : ℝ | ∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0} = {a : ℝ | -8 ≤ a ∧ a ≤ 0} := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l2302_230241


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2302_230236

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if its focus (c, 0) is symmetric about the asymptote y = (b/a)x and 
    the symmetric point of the focus lies on the other asymptote y = -(b/a)x,
    then its eccentricity is 2. -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = (b/a) * x ∧ (x - c)^2 + y^2 = (x + c)^2 + y^2) →
  (∃ x y : ℝ, y = -(b/a) * x ∧ (x + c)^2 + y^2 = 0) →
  c / a = 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2302_230236


namespace NUMINAMATH_CALUDE_calculate_units_produced_l2302_230211

/-- Given fixed cost, marginal cost, and total cost, calculate the number of units produced. -/
theorem calculate_units_produced 
  (fixed_cost : ℝ) 
  (marginal_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : fixed_cost = 12000)
  (h2 : marginal_cost = 200)
  (h3 : total_cost = 16000) :
  (total_cost - fixed_cost) / marginal_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_calculate_units_produced_l2302_230211


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l2302_230254

theorem cubic_polynomials_common_roots (c d : ℝ) : 
  (∃ u v : ℝ, u ≠ v ∧ 
    u^3 + c*u^2 + 10*u + 4 = 0 ∧ 
    u^3 + d*u^2 + 13*u + 5 = 0 ∧
    v^3 + c*v^2 + 10*v + 4 = 0 ∧ 
    v^3 + d*v^2 + 13*v + 5 = 0) → 
  c = 7 ∧ d = 8 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l2302_230254


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l2302_230275

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 14

-- Theorem statement
theorem conic_is_ellipse :
  ∃ (a b c d e f : ℝ), 
    (∀ x y : ℝ, conic_equation x y ↔ a*x^2 + b*y^2 + c*x*y + d*x + e*y + f = 0) ∧
    b^2 * c^2 - 4 * a * b * (a * e^2 + b * d^2 - c * d * e + f * (c^2 - 4 * a * b)) > 0 ∧
    a + b ≠ 0 ∧ a * b > 0 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l2302_230275


namespace NUMINAMATH_CALUDE_square_perimeter_l2302_230245

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 450 → 
  side^2 = area → 
  perimeter = 4 * side → 
  perimeter = 60 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l2302_230245


namespace NUMINAMATH_CALUDE_constant_curvature_curves_l2302_230217

/-- A plane curve is a continuous function from ℝ to ℝ² --/
def PlaneCurve := ℝ → ℝ × ℝ

/-- The curvature of a plane curve at a point --/
noncomputable def curvature (γ : PlaneCurve) (t : ℝ) : ℝ := sorry

/-- A curve has constant curvature if its curvature is the same at all points --/
def has_constant_curvature (γ : PlaneCurve) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, curvature γ t = k

/-- A straight line --/
def is_straight_line (γ : PlaneCurve) : Prop :=
  ∃ a b : ℝ × ℝ, ∀ t : ℝ, γ t = a + t • (b - a)

/-- A circle --/
def is_circle (γ : PlaneCurve) : Prop :=
  ∃ c : ℝ × ℝ, ∃ r : ℝ, r > 0 ∧ ∀ t : ℝ, ‖γ t - c‖ = r

/-- Theorem: The only plane curves with constant curvature are straight lines and circles --/
theorem constant_curvature_curves (γ : PlaneCurve) :
  has_constant_curvature γ ↔ is_straight_line γ ∨ is_circle γ :=
sorry

end NUMINAMATH_CALUDE_constant_curvature_curves_l2302_230217


namespace NUMINAMATH_CALUDE_fish_catch_problem_l2302_230220

theorem fish_catch_problem (total_fish : ℕ) (tagged_fish : ℕ) (tagged_caught : ℕ) (second_catch : ℕ) :
  total_fish = 250 →
  tagged_fish = 50 →
  tagged_caught = 10 →
  (tagged_caught : ℚ) / second_catch = tagged_fish / total_fish →
  second_catch = 50 := by
sorry

end NUMINAMATH_CALUDE_fish_catch_problem_l2302_230220


namespace NUMINAMATH_CALUDE_matts_plantation_length_l2302_230280

/-- Represents Matt's peanut plantation and its production process -/
structure PeanutPlantation where
  width : ℝ
  length : ℝ
  peanuts_per_sqft : ℝ
  peanuts_to_butter_ratio : ℝ
  butter_price_per_kg : ℝ
  total_revenue : ℝ

/-- Calculates the length of one side of Matt's plantation -/
def calculate_plantation_length (p : PeanutPlantation) : ℝ :=
  p.width

/-- Theorem stating that given the conditions, the length of Matt's plantation is 500 feet -/
theorem matts_plantation_length (p : PeanutPlantation) 
  (h1 : p.length = 500)
  (h2 : p.peanuts_per_sqft = 50)
  (h3 : p.peanuts_to_butter_ratio = 20 / 5)
  (h4 : p.butter_price_per_kg = 10)
  (h5 : p.total_revenue = 31250) :
  calculate_plantation_length p = 500 := by
  sorry

end NUMINAMATH_CALUDE_matts_plantation_length_l2302_230280


namespace NUMINAMATH_CALUDE_pen_refill_purchase_comparison_l2302_230291

theorem pen_refill_purchase_comparison (p₁ p₂ : ℝ) (hp₁ : p₁ > 0) (hp₂ : p₂ > 0) :
  (2 * p₁ * p₂) / (p₁ + p₂) ≤ (p₁ + p₂) / 2 ∧
  (2 * p₁ * p₂) / (p₁ + p₂) = (p₁ + p₂) / 2 ↔ p₁ = p₂ := by
  sorry

end NUMINAMATH_CALUDE_pen_refill_purchase_comparison_l2302_230291


namespace NUMINAMATH_CALUDE_final_deleted_files_l2302_230207

def deleted_pictures : ℕ := 5
def deleted_songs : ℕ := 12
def deleted_text_files : ℕ := 10
def deleted_video_files : ℕ := 6
def restored_pictures : ℕ := 3
def restored_video_files : ℕ := 4

theorem final_deleted_files :
  deleted_pictures + deleted_songs + deleted_text_files + deleted_video_files
  - (restored_pictures + restored_video_files) = 26 := by
  sorry

end NUMINAMATH_CALUDE_final_deleted_files_l2302_230207


namespace NUMINAMATH_CALUDE_place_letters_in_mailboxes_l2302_230297

theorem place_letters_in_mailboxes :
  let n_letters : ℕ := 3
  let n_mailboxes : ℕ := 5
  (n_letters > 0) → (n_mailboxes > 0) →
  (number_of_ways : ℕ := n_mailboxes ^ n_letters) →
  number_of_ways = 125 := by
  sorry

end NUMINAMATH_CALUDE_place_letters_in_mailboxes_l2302_230297


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l2302_230231

/-- Given an initial price and a discount amount, the discounted price is the difference between the initial price and the discount. -/
theorem discounted_price_calculation (initial_price discount : ℝ) :
  initial_price = 475 →
  discount = 276 →
  initial_price - discount = 199 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l2302_230231


namespace NUMINAMATH_CALUDE_min_words_for_passing_score_l2302_230294

/-- Represents the German vocabulary exam parameters and conditions -/
structure GermanExam where
  total_words : ℕ := 800
  correct_points : ℚ := 1
  incorrect_penalty : ℚ := 1/4
  target_score_percentage : ℚ := 90/100

/-- Calculates the exam score based on the number of words learned -/
def examScore (exam : GermanExam) (words_learned : ℕ) : ℚ :=
  (words_learned : ℚ) * exam.correct_points - 
  ((exam.total_words - words_learned) : ℚ) * exam.incorrect_penalty

/-- Theorem stating that learning at least 736 words ensures a score of at least 90% -/
theorem min_words_for_passing_score (exam : GermanExam) :
  ∀ words_learned : ℕ, words_learned ≥ 736 →
  examScore exam words_learned ≥ (exam.target_score_percentage * exam.total_words) :=
by sorry

#check min_words_for_passing_score

end NUMINAMATH_CALUDE_min_words_for_passing_score_l2302_230294


namespace NUMINAMATH_CALUDE_spider_human_leg_relationship_l2302_230219

/-- The number of legs a spider has -/
def spider_legs : ℕ := 8

/-- The number of legs a human has -/
def human_legs : ℕ := sorry

/-- The relationship between spider legs and human legs -/
def leg_relationship : ℕ := sorry

theorem spider_human_leg_relationship :
  spider_legs = leg_relationship * human_legs :=
by sorry

end NUMINAMATH_CALUDE_spider_human_leg_relationship_l2302_230219


namespace NUMINAMATH_CALUDE_computers_produced_per_month_l2302_230261

/-- Represents the number of computers produced in a month -/
def computers_per_month (computers_per_interval : ℝ) (days_per_month : ℕ) : ℝ :=
  computers_per_interval * (days_per_month * 24 * 2)

/-- Theorem stating that 4200 computers are produced per month -/
theorem computers_produced_per_month :
  computers_per_month 3.125 28 = 4200 := by
  sorry

end NUMINAMATH_CALUDE_computers_produced_per_month_l2302_230261


namespace NUMINAMATH_CALUDE_library_checkout_false_implication_l2302_230266

-- Define the universe of books in the library
variable (Book : Type)

-- Define a predicate for books available for checkout
variable (available_for_checkout : Book → Prop)

-- The main theorem
theorem library_checkout_false_implication 
  (h : ¬ ∀ (b : Book), available_for_checkout b) :
  (∃ (b : Book), ¬ available_for_checkout b) ∧ 
  (¬ ∀ (b : Book), available_for_checkout b) := by
  sorry

end NUMINAMATH_CALUDE_library_checkout_false_implication_l2302_230266


namespace NUMINAMATH_CALUDE_incorrect_number_value_l2302_230233

theorem incorrect_number_value (n : ℕ) (initial_avg correct_avg incorrect_value : ℚ) 
  (h1 : n = 10)
  (h2 : initial_avg = 20)
  (h3 : incorrect_value = 26)
  (h4 : correct_avg = 26) :
  ∃ (actual_value : ℚ),
    n * correct_avg - (n * initial_avg - incorrect_value + actual_value) = 0 ∧
    actual_value = 86 := by
sorry

end NUMINAMATH_CALUDE_incorrect_number_value_l2302_230233


namespace NUMINAMATH_CALUDE_johns_average_speed_l2302_230263

/-- John's cycling and walking trip -/
def johns_trip (uphill_distance : ℝ) (uphill_time : ℝ) (downhill_time : ℝ) (walk_distance : ℝ) (walk_time : ℝ) : Prop :=
  let total_distance := 2 * uphill_distance + walk_distance
  let total_time := uphill_time + downhill_time + walk_time
  (total_distance / (total_time / 60)) = 6

theorem johns_average_speed :
  johns_trip 3 45 15 2 20 := by
  sorry

end NUMINAMATH_CALUDE_johns_average_speed_l2302_230263


namespace NUMINAMATH_CALUDE_sum_of_first_10_common_elements_l2302_230298

/-- Arithmetic progression with first term 4 and common difference 3 -/
def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

/-- Geometric progression with first term 20 and common ratio 2 -/
def geometric_progression (k : ℕ) : ℕ := 20 * 2^k

/-- Common elements between the arithmetic and geometric progressions -/
def common_elements (n : ℕ) : Prop :=
  ∃ k : ℕ, arithmetic_progression n = geometric_progression k

/-- The sum of the first 10 common elements -/
def sum_of_common_elements : ℕ := 13981000

theorem sum_of_first_10_common_elements :
  sum_of_common_elements = 13981000 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_10_common_elements_l2302_230298


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l2302_230290

theorem function_inequality_implies_parameter_bound 
  (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 4, f x ∈ Set.Icc 1 2) →
  (∀ x ∈ Set.Icc 0 4, f x ^ 2 - a * f x + 2 < 0) →
  a > 3 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l2302_230290


namespace NUMINAMATH_CALUDE_magnitude_squared_of_complex_l2302_230201

theorem magnitude_squared_of_complex (z : ℂ) : z = 3 + 4*I → Complex.abs z ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_squared_of_complex_l2302_230201


namespace NUMINAMATH_CALUDE_comic_book_stacking_permutations_l2302_230274

theorem comic_book_stacking_permutations :
  let spiderman_books : ℕ := 7
  let archie_books : ℕ := 4
  let garfield_books : ℕ := 5
  let batman_books : ℕ := 3
  let total_books : ℕ := spiderman_books + archie_books + garfield_books + batman_books
  let non_batman_types : ℕ := 3  -- Spiderman, Archie, and Garfield

  (spiderman_books.factorial * archie_books.factorial * garfield_books.factorial * batman_books.factorial) *
  non_batman_types.factorial = 55085760 :=
by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacking_permutations_l2302_230274


namespace NUMINAMATH_CALUDE_stating_triangle_field_q_formula_l2302_230289

/-- Represents a right-angled triangle field with two people walking along its edges -/
structure TriangleField where
  /-- Length of LM -/
  r : ℝ
  /-- Length of LX -/
  p : ℝ
  /-- Length of XN -/
  q : ℝ
  /-- r is positive -/
  r_pos : r > 0
  /-- p is positive -/
  p_pos : p > 0
  /-- q is positive -/
  q_pos : q > 0
  /-- LMN is a right-angled triangle -/
  right_angle : (p + q)^2 + r^2 = (p + r - q)^2

/-- 
Theorem stating that in a TriangleField, the length q can be expressed as pr / (2p + r)
-/
theorem triangle_field_q_formula (tf : TriangleField) : tf.q = (tf.p * tf.r) / (2 * tf.p + tf.r) := by
  sorry

end NUMINAMATH_CALUDE_stating_triangle_field_q_formula_l2302_230289


namespace NUMINAMATH_CALUDE_ball_probability_theorem_l2302_230227

/-- The probability of drawing exactly one white ball from a bag -/
def prob_one_white (red : ℕ) (white : ℕ) : ℚ :=
  white / (red + white)

/-- The probability of drawing exactly one red ball from a bag -/
def prob_one_red (red : ℕ) (white : ℕ) : ℚ :=
  red / (red + white)

theorem ball_probability_theorem (n : ℕ) :
  prob_one_white 5 3 = 3/8 ∧
  (prob_one_red 5 (3 + n) = 1/2 → n = 2) := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_theorem_l2302_230227


namespace NUMINAMATH_CALUDE_no_real_solutions_l2302_230223

theorem no_real_solutions : ∀ x : ℝ, ¬(Real.sqrt (9 - 3*x) = x * Real.sqrt (9 - 9*x)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2302_230223


namespace NUMINAMATH_CALUDE_remainder_sum_l2302_230238

theorem remainder_sum (a b : ℤ) 
  (ha : a % 80 = 75) 
  (hb : b % 90 = 85) : 
  (a + b) % 40 = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l2302_230238


namespace NUMINAMATH_CALUDE_pentagon_triangles_l2302_230244

/-- The number of triangles formed in a pentagon when drawing diagonals from one vertex --/
def triangles_in_pentagon : ℕ := 3

/-- The number of vertices in a pentagon --/
def pentagon_vertices : ℕ := 5

/-- The number of diagonals drawn from one vertex in a pentagon --/
def diagonals_from_vertex : ℕ := 2

theorem pentagon_triangles :
  triangles_in_pentagon = diagonals_from_vertex + 1 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_triangles_l2302_230244


namespace NUMINAMATH_CALUDE_painting_price_increase_l2302_230271

theorem painting_price_increase (X : ℝ) : 
  (1 + X / 100) * (1 - 0.15) = 1.02 → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_painting_price_increase_l2302_230271


namespace NUMINAMATH_CALUDE_opposite_of_three_l2302_230276

theorem opposite_of_three (a : ℝ) : (2 * a + 3) + 3 = 0 → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l2302_230276


namespace NUMINAMATH_CALUDE_sqrt_neg_six_squared_minus_one_l2302_230218

theorem sqrt_neg_six_squared_minus_one : Real.sqrt ((-6)^2) - 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_six_squared_minus_one_l2302_230218


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2302_230287

theorem arithmetic_calculation : 4 * 6 * 8 + 18 / 3^2 = 194 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2302_230287


namespace NUMINAMATH_CALUDE_cycle_price_proof_l2302_230268

/-- Proves that given a cycle sold at a 10% loss for Rs. 1620, its original price was Rs. 1800. -/
theorem cycle_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1620)
  (h2 : loss_percentage = 10) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_cycle_price_proof_l2302_230268


namespace NUMINAMATH_CALUDE_andrena_debelyn_difference_l2302_230267

/-- The number of dolls each person has after the gift exchange --/
structure DollCounts where
  debelyn : ℕ
  christel : ℕ
  andrena : ℕ

/-- Calculate the final doll counts based on initial counts and gifts --/
def finalCounts (debelynInitial christelInitial : ℕ) : DollCounts :=
  let debelynFinal := debelynInitial - 2
  let christelFinal := christelInitial - 5
  let andrenaFinal := christelFinal + 2
  { debelyn := debelynFinal
  , christel := christelFinal
  , andrena := andrenaFinal }

/-- Theorem stating the difference in doll counts between Andrena and Debelyn --/
theorem andrena_debelyn_difference : 
  let counts := finalCounts 20 24
  counts.andrena - counts.debelyn = 3 := by sorry

end NUMINAMATH_CALUDE_andrena_debelyn_difference_l2302_230267


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2302_230210

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {1, 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2302_230210


namespace NUMINAMATH_CALUDE_prob_king_jack_queen_value_l2302_230237

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Jacks in a standard deck -/
def NumJacks : ℕ := 4

/-- Number of Queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Probability of drawing a King, then a Jack, then a Queen from a standard deck without replacement -/
def prob_king_jack_queen : ℚ :=
  (NumKings : ℚ) / StandardDeck *
  (NumJacks : ℚ) / (StandardDeck - 1) *
  (NumQueens : ℚ) / (StandardDeck - 2)

theorem prob_king_jack_queen_value :
  prob_king_jack_queen = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_jack_queen_value_l2302_230237


namespace NUMINAMATH_CALUDE_vertical_asymptote_at_five_l2302_230202

/-- The function f(x) = (x^2 - 3x + 10) / (x - 5) has a vertical asymptote at x = 5 -/
theorem vertical_asymptote_at_five :
  let f : ℝ → ℝ := λ x => (x^2 - 3*x + 10) / (x - 5)
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), 0 < δ ∧ δ < ε →
    (∀ x : ℝ, 0 < |x - 5| ∧ |x - 5| < δ → |f x| > 1/δ) :=
by sorry

end NUMINAMATH_CALUDE_vertical_asymptote_at_five_l2302_230202


namespace NUMINAMATH_CALUDE_impossibility_of_2023_linked_triangles_l2302_230229

-- Define the space and points
def Space := Type
def Point : Type := Unit

-- Define the colors of points
inductive Color
| Yellow
| Red

-- Define the properties of the space
structure SpaceProperties (s : Space) :=
  (total_points : Nat)
  (yellow_points : Nat)
  (red_points : Nat)
  (no_four_coplanar : Prop)
  (total_points_eq : total_points = yellow_points + red_points)

-- Define a triangle
structure Triangle (s : Space) :=
  (vertices : Fin 3 → Point)

-- Define the linking relation between triangles
def isLinked (s : Space) (yellow : Triangle s) (red : Triangle s) : Prop := sorry

-- Define the count of linked triangles
def linkedTrianglesCount (s : Space) (props : SpaceProperties s) : Nat := sorry

-- The main theorem
theorem impossibility_of_2023_linked_triangles (s : Space) 
  (props : SpaceProperties s) 
  (h1 : props.total_points = 43)
  (h2 : props.yellow_points = 3)
  (h3 : props.red_points = 40)
  (h4 : props.no_four_coplanar) :
  linkedTrianglesCount s props ≠ 2023 := by sorry

end NUMINAMATH_CALUDE_impossibility_of_2023_linked_triangles_l2302_230229


namespace NUMINAMATH_CALUDE_arthur_muffins_l2302_230284

theorem arthur_muffins (initial_muffins : ℕ) (multiplier : ℚ) : 
  initial_muffins = 80 →
  multiplier = 5/2 →
  (multiplier * initial_muffins : ℚ) - initial_muffins = 120 :=
by sorry

end NUMINAMATH_CALUDE_arthur_muffins_l2302_230284


namespace NUMINAMATH_CALUDE_candice_spending_l2302_230279

def total_money : ℕ := 100
def mildred_spent : ℕ := 25
def money_left : ℕ := 40

theorem candice_spending : 
  total_money - mildred_spent - money_left = 35 := by
sorry

end NUMINAMATH_CALUDE_candice_spending_l2302_230279


namespace NUMINAMATH_CALUDE_missing_number_proof_l2302_230215

theorem missing_number_proof (x : ℝ) (y : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (128 + 255 + y + 1023 + x) / 5 = 398.2 →
  y = 511 := by
sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2302_230215


namespace NUMINAMATH_CALUDE_suitcase_weight_problem_l2302_230200

/-- Proves that given the initial ratio of books to clothes to electronics as 7:4:3, 
    and the fact that removing 6 pounds of clothing doubles the ratio of books to clothes, 
    the weight of electronics is 9 pounds. -/
theorem suitcase_weight_problem (B C E : ℝ) : 
  (B / C = 7 / 4) →  -- Initial ratio of books to clothes
  (C / E = 4 / 3) →  -- Initial ratio of clothes to electronics
  (B / (C - 6) = 7 / 2) →  -- New ratio after removing 6 pounds of clothes
  E = 9 := by
sorry

end NUMINAMATH_CALUDE_suitcase_weight_problem_l2302_230200
