import Mathlib

namespace NUMINAMATH_CALUDE_financial_equation_solution_l871_87150

theorem financial_equation_solution (g t p : ℂ) : 
  3 * g * p - t = 9000 ∧ g = 3 ∧ t = 3 + 75 * Complex.I → 
  p = 1000 + 1/3 + 8 * Complex.I + 1/3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_financial_equation_solution_l871_87150


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l871_87190

theorem function_inequality_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → Real.log (1 + x) ≥ a * x / (1 + x)) →
  a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l871_87190


namespace NUMINAMATH_CALUDE_unique_solution_l871_87133

theorem unique_solution : ∃! x : ℝ, x + x^2 + 15 = 96 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l871_87133


namespace NUMINAMATH_CALUDE_train_cars_count_l871_87130

/-- The number of cars counted in the first 15 seconds -/
def initial_cars : ℕ := 9

/-- The time in seconds for the initial count -/
def initial_time : ℕ := 15

/-- The total time in seconds for the train to clear the crossing -/
def total_time : ℕ := 210

/-- The number of cars in the train -/
def train_cars : ℕ := (initial_cars * total_time) / initial_time

theorem train_cars_count : train_cars = 126 := by
  sorry

end NUMINAMATH_CALUDE_train_cars_count_l871_87130


namespace NUMINAMATH_CALUDE_function_decomposition_l871_87118

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (g h : ℝ → ℝ),
    (∀ x, f x = g x + h x) ∧
    (∀ x, g x = g (-x)) ∧
    (∀ x, h (1 + x) = h (1 - x)) := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l871_87118


namespace NUMINAMATH_CALUDE_gina_tip_percentage_is_five_percent_l871_87131

/-- The bill amount in dollars -/
def bill_amount : ℝ := 26

/-- The minimum tip percentage for good tippers -/
def good_tipper_percentage : ℝ := 20

/-- The additional amount in cents Gina needs to tip to be a good tipper -/
def additional_tip_cents : ℝ := 390

/-- Gina's tip percentage -/
def gina_tip_percentage : ℝ := 5

/-- Theorem stating that Gina's tip percentage is 5% given the conditions -/
theorem gina_tip_percentage_is_five_percent :
  (gina_tip_percentage / 100) * bill_amount + (additional_tip_cents / 100) =
  (good_tipper_percentage / 100) * bill_amount :=
by sorry

end NUMINAMATH_CALUDE_gina_tip_percentage_is_five_percent_l871_87131


namespace NUMINAMATH_CALUDE_original_price_calculation_l871_87125

theorem original_price_calculation (initial_price : ℚ) : 
  (initial_price * (1 + 10/100) * (1 - 20/100) = 2) → initial_price = 25/11 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l871_87125


namespace NUMINAMATH_CALUDE_fraction_equality_l871_87197

theorem fraction_equality (y : ℝ) (h : y > 0) : (9 * y) / 20 + (3 * y) / 10 = 0.75 * y := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l871_87197


namespace NUMINAMATH_CALUDE_roxanne_change_l871_87183

/-- Represents the purchase and payment scenario for Roxanne --/
structure Purchase where
  lemonade_count : ℕ
  lemonade_price : ℚ
  sandwich_count : ℕ
  sandwich_price : ℚ
  paid_amount : ℚ

/-- Calculates the change Roxanne should receive --/
def calculate_change (p : Purchase) : ℚ :=
  p.paid_amount - (p.lemonade_count * p.lemonade_price + p.sandwich_count * p.sandwich_price)

/-- Theorem stating that Roxanne's change should be $11 --/
theorem roxanne_change :
  let p : Purchase := {
    lemonade_count := 2,
    lemonade_price := 2,
    sandwich_count := 2,
    sandwich_price := 2.5,
    paid_amount := 20
  }
  calculate_change p = 11 := by sorry

end NUMINAMATH_CALUDE_roxanne_change_l871_87183


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l871_87108

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (4 + Real.sqrt x) = 4 → x = 144 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l871_87108


namespace NUMINAMATH_CALUDE_gingers_garden_water_usage_l871_87180

/-- Represents the problem of calculating water usage in Ginger's garden --/
theorem gingers_garden_water_usage 
  (hours_worked : ℕ) 
  (bottle_capacity : ℕ) 
  (total_water_used : ℕ) 
  (h1 : hours_worked = 8)
  (h2 : bottle_capacity = 2)
  (h3 : total_water_used = 26) :
  (total_water_used - hours_worked * bottle_capacity) / bottle_capacity = 5 := by
  sorry

#check gingers_garden_water_usage

end NUMINAMATH_CALUDE_gingers_garden_water_usage_l871_87180


namespace NUMINAMATH_CALUDE_remainder_sum_l871_87145

theorem remainder_sum (D : ℕ) (h1 : D > 0) (h2 : 242 % D = 11) (h3 : 698 % D = 18) :
  940 % D = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l871_87145


namespace NUMINAMATH_CALUDE_certain_number_value_l871_87121

theorem certain_number_value (x y z : ℝ) 
  (h1 : y = 1.10 * z) 
  (h2 : x = 0.90 * y) 
  (h3 : x = 123.75) : 
  z = 125 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l871_87121


namespace NUMINAMATH_CALUDE_inequality_solution_l871_87184

theorem inequality_solution (a x : ℝ) :
  (a * x) / (x - 1) < (a - 1) / (x - 1) ↔
  (a > 0 ∧ (a - 1) / a < x ∧ x < 1) ∨
  (a = 0 ∧ x < 1) ∨
  (a < 0 ∧ (x > (a - 1) / a ∨ x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l871_87184


namespace NUMINAMATH_CALUDE_unique_prime_factors_count_l871_87199

theorem unique_prime_factors_count (n : ℕ+) (h : Nat.card (Nat.divisors n) = 12320) :
  Finset.card (Nat.factors n).toFinset = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_factors_count_l871_87199


namespace NUMINAMATH_CALUDE_theater_tickets_proof_l871_87191

theorem theater_tickets_proof (reduced_first_week : ℕ) 
  (h1 : reduced_first_week > 0)
  (h2 : 5 * reduced_first_week = 16500)
  (h3 : reduced_first_week + 16500 = 25200) : 
  reduced_first_week = 8700 := by
  sorry

end NUMINAMATH_CALUDE_theater_tickets_proof_l871_87191


namespace NUMINAMATH_CALUDE_square_and_sqrt_identities_l871_87159

theorem square_and_sqrt_identities :
  (1001 : ℕ)^2 = 1002001 ∧
  (1001001 : ℕ)^2 = 1002003002001 ∧
  (1002003004005004003002001 : ℕ).sqrt = 1001001001001 := by
  sorry

end NUMINAMATH_CALUDE_square_and_sqrt_identities_l871_87159


namespace NUMINAMATH_CALUDE_count_parallelograms_l871_87101

/-- The number of parallelograms formed in a grid created by intersecting a parallelogram
    with two sets of m lines each (parallel to the parallelogram's sides) -/
def num_parallelograms (m : ℕ) : ℕ :=
  ((m + 1) * (m + 2) / 2) ^ 2

/-- Theorem stating that num_parallelograms correctly calculates the number of parallelograms -/
theorem count_parallelograms (m : ℕ) :
  num_parallelograms m = ((m + 1) * (m + 2) / 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_count_parallelograms_l871_87101


namespace NUMINAMATH_CALUDE_cancellable_fractions_characterization_l871_87134

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def cancellable_fraction (n d : ℕ) : Prop :=
  is_two_digit n ∧ is_two_digit d ∧
  ∃ (a b c : ℕ), 0 < a ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 < c ∧
    n = 10 * a + b ∧ d = 10 * b + c ∧ n * c = a * d

def valid_fractions : Set (ℕ × ℕ) :=
  {(19, 95), (16, 64), (11, 11), (26, 65), (22, 22), (33, 33),
   (49, 98), (44, 44), (55, 55), (66, 66), (77, 77), (88, 88), (99, 99)}

theorem cancellable_fractions_characterization :
  {p : ℕ × ℕ | cancellable_fraction p.1 p.2} = valid_fractions := by sorry

end NUMINAMATH_CALUDE_cancellable_fractions_characterization_l871_87134


namespace NUMINAMATH_CALUDE_cake_recipe_difference_l871_87187

theorem cake_recipe_difference (flour_required sugar_required sugar_added : ℕ) :
  flour_required = 9 →
  sugar_required = 6 →
  sugar_added = 4 →
  flour_required - (sugar_required - sugar_added) = 7 := by
sorry

end NUMINAMATH_CALUDE_cake_recipe_difference_l871_87187


namespace NUMINAMATH_CALUDE_binary_sum_equals_852_l871_87114

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def num1 : List Bool := [true, true, true, true, true, true, true, true, true]
def num2 : List Bool := [true, false, true, false, true, false, true, false, true]

theorem binary_sum_equals_852 : 
  binary_to_decimal num1 + binary_to_decimal num2 = 852 := by
sorry

end NUMINAMATH_CALUDE_binary_sum_equals_852_l871_87114


namespace NUMINAMATH_CALUDE_average_age_of_students_average_age_proof_l871_87105

theorem average_age_of_students (total_students : Nat) 
  (group1_count : Nat) (group1_avg : Nat) 
  (group2_count : Nat) (group2_avg : Nat)
  (last_student_age : Nat) : Nat :=
  let total_age := group1_count * group1_avg + group2_count * group2_avg + last_student_age
  total_age / total_students

theorem average_age_proof :
  average_age_of_students 15 8 14 6 16 17 = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_students_average_age_proof_l871_87105


namespace NUMINAMATH_CALUDE_complex_equation_solution_l871_87173

theorem complex_equation_solution (x y : ℝ) :
  (Complex.I * (x + Complex.I) + y = 1 + 2 * Complex.I) →
  x - y = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l871_87173


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l871_87172

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def is_scalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem smallest_prime_perimeter_triangle :
  ∀ a b c : ℕ,
    a = 5 →
    is_prime a →
    is_prime b →
    is_prime c →
    is_prime (a + b + c) →
    triangle_inequality a b c →
    is_scalene a b c →
    ∀ p q r : ℕ,
      p = 5 →
      is_prime p →
      is_prime q →
      is_prime r →
      is_prime (p + q + r) →
      triangle_inequality p q r →
      is_scalene p q r →
      a + b + c ≤ p + q + r →
    a + b + c = 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l871_87172


namespace NUMINAMATH_CALUDE_basketball_three_pointers_l871_87179

/-- Represents the number of 3-point shots in a basketball game -/
def three_point_shots (total_points total_shots : ℕ) : ℕ :=
  sorry

/-- The number of 3-point shots is 4 when the total points is 26 and total shots is 11 -/
theorem basketball_three_pointers :
  three_point_shots 26 11 = 4 :=
sorry

end NUMINAMATH_CALUDE_basketball_three_pointers_l871_87179


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_plus_self_l871_87153

theorem complex_magnitude_squared_plus_self (z : ℂ) (h : z = 1 + I) :
  Complex.abs (z^2 + z) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_plus_self_l871_87153


namespace NUMINAMATH_CALUDE_payroll_after_layoffs_l871_87158

/-- Represents the company's employee structure and payroll --/
structure Company where
  total_employees : Nat
  employees_2000 : Nat
  employees_2500 : Nat
  employees_3000 : Nat
  bonus_2000 : Nat
  health_benefit_2500 : Nat
  retirement_benefit_3000 : Nat

/-- Calculates the remaining employees after a layoff --/
def layoff (employees : Nat) (percentage : Nat) : Nat :=
  employees - (employees * percentage / 100)

/-- Applies the first round of layoffs and benefit changes --/
def first_round (c : Company) : Company :=
  { c with
    employees_2000 := layoff c.employees_2000 20,
    employees_2500 := layoff c.employees_2500 25,
    employees_3000 := layoff c.employees_3000 15,
    bonus_2000 := 400,
    health_benefit_2500 := 300 }

/-- Applies the second round of layoffs and benefit changes --/
def second_round (c : Company) : Company :=
  { c with
    employees_2000 := layoff c.employees_2000 10,
    employees_2500 := layoff c.employees_2500 15,
    employees_3000 := layoff c.employees_3000 5,
    retirement_benefit_3000 := 480 }

/-- Calculates the total payroll after both rounds of layoffs --/
def total_payroll (c : Company) : Nat :=
  c.employees_2000 * (2000 + c.bonus_2000) +
  c.employees_2500 * (2500 + c.health_benefit_2500) +
  c.employees_3000 * (3000 + c.retirement_benefit_3000)

/-- The initial company state --/
def initial_company : Company :=
  { total_employees := 450,
    employees_2000 := 150,
    employees_2500 := 200,
    employees_3000 := 100,
    bonus_2000 := 500,
    health_benefit_2500 := 400,
    retirement_benefit_3000 := 600 }

theorem payroll_after_layoffs :
  total_payroll (second_round (first_round initial_company)) = 893200 := by
  sorry

end NUMINAMATH_CALUDE_payroll_after_layoffs_l871_87158


namespace NUMINAMATH_CALUDE_double_root_condition_l871_87148

/-- For a polynomial of the form A x^(n+1) + B x^n + 1, where n is a natural number,
    x = 1 is a root with multiplicity at least 2 if and only if A = n and B = -(n+1). -/
theorem double_root_condition (n : ℕ) (A B : ℝ) :
  (∀ x : ℝ, A * x^(n+1) + B * x^n + 1 = 0 ∧ 
   (A * (n+1) * x^n + B * n * x^(n-1) = 0)) ↔ 
  (A = n ∧ B = -(n+1)) :=
sorry

end NUMINAMATH_CALUDE_double_root_condition_l871_87148


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l871_87151

/-- Given a 40-liter solution of alcohol and water, prove that the initial percentage
    of alcohol is 5% if adding 4.5 liters of alcohol and 5.5 liters of water
    results in a 50-liter solution that is 13% alcohol. -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (added_water : ℝ)
  (final_volume : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 40)
  (h2 : added_alcohol = 4.5)
  (h3 : added_water = 5.5)
  (h4 : final_volume = initial_volume + added_alcohol + added_water)
  (h5 : final_percentage = 13)
  (h6 : final_percentage / 100 * final_volume = 
        initial_volume * (initial_percentage / 100) + added_alcohol) :
  initial_percentage = 5 :=
by sorry

end NUMINAMATH_CALUDE_initial_alcohol_percentage_l871_87151


namespace NUMINAMATH_CALUDE_intersection_point_determines_k_l871_87176

/-- Line with slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem intersection_point_determines_k 
  (m n : Line)
  (p : Point)
  (k : ℝ)
  (h1 : m.slope = 4)
  (h2 : m.intercept = 2)
  (h3 : n.slope = k)
  (h4 : n.intercept = 3)
  (h5 : p.x = 1)
  (h6 : p.y = 6)
  (h7 : p.on_line m)
  (h8 : p.on_line n)
  : k = 3 := by
  sorry

#check intersection_point_determines_k

end NUMINAMATH_CALUDE_intersection_point_determines_k_l871_87176


namespace NUMINAMATH_CALUDE_organization_growth_l871_87169

/-- Represents the number of people in the organization at year k -/
def people_count (k : ℕ) : ℕ :=
  if k = 0 then 30
  else 3 * people_count (k - 1) - 20

/-- The number of leaders in the organization each year -/
def num_leaders : ℕ := 10

/-- The initial number of people in the organization -/
def initial_people : ℕ := 30

theorem organization_growth :
  people_count 10 = 1180990 :=
sorry

end NUMINAMATH_CALUDE_organization_growth_l871_87169


namespace NUMINAMATH_CALUDE_omega_is_abc_l871_87138

theorem omega_is_abc (ω a b c x y z : ℝ) 
  (distinct : ω ≠ a ∧ ω ≠ b ∧ ω ≠ c ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (eq1 : x + y + z = 1)
  (eq2 : a^2 * x + b^2 * y + c^2 * z = ω^2)
  (eq3 : a^3 * x + b^3 * y + c^3 * z = ω^3)
  (eq4 : a^4 * x + b^4 * y + c^4 * z = ω^4) :
  ω = a ∨ ω = b ∨ ω = c :=
sorry

end NUMINAMATH_CALUDE_omega_is_abc_l871_87138


namespace NUMINAMATH_CALUDE_fraction_is_positive_l871_87100

theorem fraction_is_positive
  (a b c d : ℝ)
  (ha : a < 0) (hb : b < 0) (hc : c < 0) (hd : d < 0)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h : |x₁ - a| + |x₂ + b| + |x₃ - c| + |x₄ + d| = 0) :
  (x₁ * x₂) / (x₃ * x₄) > 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_is_positive_l871_87100


namespace NUMINAMATH_CALUDE_cars_remaining_l871_87141

theorem cars_remaining (initial : Nat) (first_group : Nat) (second_group : Nat)
  (h1 : initial = 24)
  (h2 : first_group = 8)
  (h3 : second_group = 6) :
  initial - first_group - second_group = 10 := by
  sorry

end NUMINAMATH_CALUDE_cars_remaining_l871_87141


namespace NUMINAMATH_CALUDE_probability_of_sum_17_l871_87126

/-- The number of faces on each die -/
def numFaces : ℕ := 8

/-- The target sum we're aiming for -/
def targetSum : ℕ := 17

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The probability of rolling a specific number on a single die -/
def singleDieProbability : ℚ := 1 / numFaces

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (ways to get a sum of 17) -/
def favorableOutcomes : ℕ := 27

/-- The theorem stating the probability of rolling a sum of 17 with three 8-faced dice -/
theorem probability_of_sum_17 : 
  (favorableOutcomes : ℚ) / totalOutcomes = 27 / 512 :=
sorry

end NUMINAMATH_CALUDE_probability_of_sum_17_l871_87126


namespace NUMINAMATH_CALUDE_train_travel_times_l871_87139

theorem train_travel_times
  (usual_speed_A : ℝ)
  (usual_speed_B : ℝ)
  (distance_XM : ℝ)
  (h1 : usual_speed_A > 0)
  (h2 : usual_speed_B > 0)
  (h3 : distance_XM > 0)
  (h4 : usual_speed_B * 2 = usual_speed_A * 3) :
  let t : ℝ := 180
  let current_speed_A : ℝ := (6 / 7) * usual_speed_A
  let time_XM_reduced : ℝ := distance_XM / current_speed_A
  let time_XM_usual : ℝ := distance_XM / usual_speed_A
  let time_XY_A : ℝ := 3 * time_XM_usual
  let time_XY_B : ℝ := 810
  (time_XM_reduced = time_XM_usual + 30) ∧
  (time_XM_usual = t) ∧
  (time_XY_B = 1.5 * time_XY_A) := by
  sorry

end NUMINAMATH_CALUDE_train_travel_times_l871_87139


namespace NUMINAMATH_CALUDE_range_of_a_l871_87136

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + 2*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), ¬ Monotone (g a))
  ∧ (∀ x ∈ Set.Icc 1 (Real.exp 1), g a x ≤ g a (Real.exp 1))
  ∧ (∀ x ∈ Set.Icc 1 (Real.exp 1), x ≠ Real.exp 1 → g a x < g a (Real.exp 1))
  → 3 < a ∧ a < (Real.exp 1)^2 / 2 + 2 * Real.exp 1 - 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l871_87136


namespace NUMINAMATH_CALUDE_croissant_cost_calculation_l871_87175

/-- Calculates the cost of croissants for a committee luncheon --/
theorem croissant_cost_calculation 
  (people : ℕ) 
  (sandwiches_per_person : ℕ) 
  (croissants_per_dozen : ℕ) 
  (cost_per_dozen : ℚ) : 
  people = 24 → 
  sandwiches_per_person = 2 → 
  croissants_per_dozen = 12 → 
  cost_per_dozen = 8 → 
  (people * sandwiches_per_person / croissants_per_dozen : ℚ) * cost_per_dozen = 32 :=
by sorry

#check croissant_cost_calculation

end NUMINAMATH_CALUDE_croissant_cost_calculation_l871_87175


namespace NUMINAMATH_CALUDE_sequence_properties_l871_87123

def S (n : ℕ) (a : ℕ → ℝ) : ℝ := a n + n^2 - 1

def b_relation (n : ℕ) (a b : ℕ → ℝ) : Prop :=
  3^n * b (n+1) = (n+1) * a (n+1) - n * a n

theorem sequence_properties (a b : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n, S n a = a n + n^2 - 1) →
  (∀ n, b_relation n a b) →
  b 1 = 3 →
  (∀ n, a n = 2*n + 1) ∧
  (∀ n, b n = (4*n - 1) / 3^(n-1)) ∧
  (∀ n, T n = 15/2 - (4*n + 5) / (2 * 3^(n-1))) ∧
  (∀ n > 3, T n ≥ 7) ∧
  (T 3 < 7) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l871_87123


namespace NUMINAMATH_CALUDE_semicircle_circumference_from_rectangle_square_l871_87171

/-- Given a rectangle and a square with equal perimeters, prove the circumference of a semicircle
    whose diameter is equal to the side of the square. -/
theorem semicircle_circumference_from_rectangle_square 
  (rect_length : ℝ) (rect_breadth : ℝ) (square_side : ℝ) :
  rect_length = 8 →
  rect_breadth = 6 →
  2 * (rect_length + rect_breadth) = 4 * square_side →
  ∃ (semicircle_circumference : ℝ), 
    semicircle_circumference = Real.pi * square_side / 2 + square_side :=
by sorry

end NUMINAMATH_CALUDE_semicircle_circumference_from_rectangle_square_l871_87171


namespace NUMINAMATH_CALUDE_cleaning_time_calculation_l871_87106

/-- Represents the cleaning schedule for a person -/
structure CleaningSchedule where
  vacuuming : Nat × Nat  -- (minutes per day, days per week)
  dusting : Nat × Nat
  sweeping : Nat × Nat
  deep_cleaning : Nat × Nat

/-- Calculates the total cleaning time in minutes per week -/
def totalCleaningTime (schedule : CleaningSchedule) : Nat :=
  schedule.vacuuming.1 * schedule.vacuuming.2 +
  schedule.dusting.1 * schedule.dusting.2 +
  schedule.sweeping.1 * schedule.sweeping.2 +
  schedule.deep_cleaning.1 * schedule.deep_cleaning.2

/-- Converts minutes to hours and minutes -/
def minutesToHoursAndMinutes (minutes : Nat) : Nat × Nat :=
  (minutes / 60, minutes % 60)

/-- Aron's cleaning schedule -/
def aronSchedule : CleaningSchedule :=
  { vacuuming := (30, 3)
    dusting := (20, 2)
    sweeping := (15, 4)
    deep_cleaning := (45, 1) }

/-- Ben's cleaning schedule -/
def benSchedule : CleaningSchedule :=
  { vacuuming := (40, 2)
    dusting := (25, 3)
    sweeping := (20, 5)
    deep_cleaning := (60, 1) }

theorem cleaning_time_calculation :
  let aronTime := totalCleaningTime aronSchedule
  let benTime := totalCleaningTime benSchedule
  let aronHoursMinutes := minutesToHoursAndMinutes aronTime
  let benHoursMinutes := minutesToHoursAndMinutes benTime
  let timeDifference := benTime - aronTime
  let timeDifferenceHoursMinutes := minutesToHoursAndMinutes timeDifference
  aronHoursMinutes = (3, 55) ∧
  benHoursMinutes = (5, 15) ∧
  timeDifferenceHoursMinutes = (1, 20) := by
  sorry

end NUMINAMATH_CALUDE_cleaning_time_calculation_l871_87106


namespace NUMINAMATH_CALUDE_no_solution_for_system_l871_87107

theorem no_solution_for_system :
  ¬ ∃ (x y : ℝ), (2 * x - 3 * y = 6) ∧ (4 * x - 6 * y = 8) ∧ (5 * x - 5 * y = 15) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_system_l871_87107


namespace NUMINAMATH_CALUDE_smallest_block_with_399_hidden_cubes_l871_87192

/-- A rectangular block made of identical cubes -/
structure RectangularBlock where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of cubes in a rectangular block -/
def RectangularBlock.volume (b : RectangularBlock) : ℕ :=
  b.length * b.width * b.height

/-- The number of hidden cubes when three faces are visible -/
def RectangularBlock.hiddenCubes (b : RectangularBlock) : ℕ :=
  (b.length - 1) * (b.width - 1) * (b.height - 1)

/-- The theorem stating the smallest possible value of N -/
theorem smallest_block_with_399_hidden_cubes :
  ∀ b : RectangularBlock,
    b.hiddenCubes = 399 →
    b.volume ≥ 640 ∧
    ∃ b' : RectangularBlock, b'.hiddenCubes = 399 ∧ b'.volume = 640 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_with_399_hidden_cubes_l871_87192


namespace NUMINAMATH_CALUDE_positive_difference_of_average_l871_87155

theorem positive_difference_of_average (y : ℝ) : 
  (50 + y) / 2 = 35 → |50 - y| = 30 := by
  sorry

end NUMINAMATH_CALUDE_positive_difference_of_average_l871_87155


namespace NUMINAMATH_CALUDE_equal_roots_condition_l871_87156

theorem equal_roots_condition (x m : ℝ) : 
  (x * (x - 2) - (m + 2)) / ((x - 2) * (m - 2)) = x / m → 
  (∃ (a : ℝ), ∀ (x : ℝ), x * (x - 2) - (m + 2) = (x - 2) * (m - 2) * (x / m) → x = a) →
  m = -3/2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l871_87156


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l871_87178

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 30 →
  initial_mean = 150 →
  incorrect_value = 135 →
  correct_value = 165 →
  let total_sum := n * initial_mean
  let corrected_sum := total_sum - incorrect_value + correct_value
  corrected_sum / n = 151 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l871_87178


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_necessary_not_sufficient_l871_87132

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the specific lines and planes
variable (l m : Line) (α β : Plane)

-- State the theorem
theorem perpendicular_implies_parallel_necessary_not_sufficient 
  (h1 : perp_plane l α) 
  (h2 : subset m β) :
  (∀ α β, parallel α β → perp l m) ∧ 
  (∃ α β, perp l m ∧ ¬ parallel α β) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_necessary_not_sufficient_l871_87132


namespace NUMINAMATH_CALUDE_early_movie_savings_l871_87195

/-- Calculates the savings for going to an earlier movie given ticket and food combo prices and discounts --/
theorem early_movie_savings 
  (evening_ticket_price : ℚ)
  (evening_combo_price : ℚ)
  (ticket_discount_percent : ℚ)
  (combo_discount_percent : ℚ)
  (h1 : evening_ticket_price = 10)
  (h2 : evening_combo_price = 10)
  (h3 : ticket_discount_percent = 20 / 100)
  (h4 : combo_discount_percent = 50 / 100)
  : evening_ticket_price * ticket_discount_percent + 
    evening_combo_price * combo_discount_percent = 7 := by
  sorry

end NUMINAMATH_CALUDE_early_movie_savings_l871_87195


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l871_87189

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I) = 2) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l871_87189


namespace NUMINAMATH_CALUDE_tangent_line_equation_l871_87146

/-- The function f(x) = x³ - 3x² + x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 1

theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m*x + b ↔ 2*x + y - 1 = 0) ∧
    (m = f' 1) ∧
    (f 1 = m*1 + b) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l871_87146


namespace NUMINAMATH_CALUDE_max_quartets_correct_max_quartets_5x5_l871_87122

def max_quartets (m n : ℕ) : ℕ :=
  if m % 2 = 0 ∧ n % 2 = 0 then
    m * n / 4
  else if (m % 2 = 0 ∧ n % 2 = 1) ∨ (m % 2 = 1 ∧ n % 2 = 0) then
    m * (n - 1) / 4
  else
    (m * (n - 1) - 2) / 4

theorem max_quartets_correct (m n : ℕ) :
  max_quartets m n = 
    if m % 2 = 0 ∧ n % 2 = 0 then
      m * n / 4
    else if (m % 2 = 0 ∧ n % 2 = 1) ∨ (m % 2 = 1 ∧ n % 2 = 0) then
      m * (n - 1) / 4
    else
      (m * (n - 1) - 2) / 4 :=
by sorry

theorem max_quartets_5x5 : max_quartets 5 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_quartets_correct_max_quartets_5x5_l871_87122


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_5_pow_7_plus_10_pow_6_l871_87115

theorem greatest_prime_factor_of_5_pow_7_plus_10_pow_6 :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (5^7 + 10^6) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (5^7 + 10^6) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_5_pow_7_plus_10_pow_6_l871_87115


namespace NUMINAMATH_CALUDE_sequence_range_l871_87113

theorem sequence_range (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, 12 * S n = 4 * a (n + 1) + 5^n - 13) →
  (∀ n : ℕ, S n ≤ S 4) →
  (∀ n : ℕ, S (n + 1) = S n + a (n + 1)) →
  13/48 ≤ a 1 ∧ a 1 ≤ 59/64 := by
  sorry

end NUMINAMATH_CALUDE_sequence_range_l871_87113


namespace NUMINAMATH_CALUDE_monomial_properties_l871_87166

/-- Represents a monomial in variables a and b -/
structure Monomial where
  coeff : ℤ
  a_exp : ℕ
  b_exp : ℕ

/-- The coefficient of a monomial -/
def coefficient (m : Monomial) : ℤ := m.coeff

/-- The degree of a monomial -/
def degree (m : Monomial) : ℕ := m.a_exp + m.b_exp

/-- The monomial 2a²b -/
def m : Monomial := { coeff := 2, a_exp := 2, b_exp := 1 }

theorem monomial_properties :
  coefficient m = 2 ∧ degree m = 3 := by sorry

end NUMINAMATH_CALUDE_monomial_properties_l871_87166


namespace NUMINAMATH_CALUDE_compound_interest_rate_l871_87127

theorem compound_interest_rate (ci_2 ci_3 : ℚ) 
  (h1 : ci_2 = 1200)
  (h2 : ci_3 = 1260) : 
  ∃ r : ℚ, r = 0.05 ∧ r * ci_2 = ci_3 - ci_2 :=
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l871_87127


namespace NUMINAMATH_CALUDE_sale_price_lower_than_original_l871_87198

theorem sale_price_lower_than_original : ∀ x : ℝ, x > 0 → 0.75 * (1.3 * x) < x := by
  sorry

end NUMINAMATH_CALUDE_sale_price_lower_than_original_l871_87198


namespace NUMINAMATH_CALUDE_sally_boxes_proof_l871_87161

/-- The number of boxes Sally sold on Saturday -/
def saturday_boxes : ℕ := 65

/-- The number of boxes Sally sold on Sunday -/
def sunday_boxes : ℕ := (3 * saturday_boxes) / 2

/-- The number of boxes Sally sold on Monday -/
def monday_boxes : ℕ := (13 * sunday_boxes) / 10

theorem sally_boxes_proof :
  saturday_boxes + sunday_boxes + monday_boxes = 290 :=
sorry

end NUMINAMATH_CALUDE_sally_boxes_proof_l871_87161


namespace NUMINAMATH_CALUDE_salt_solution_weight_l871_87157

/-- 
Given a salt solution with initial concentration of 10% and final concentration of 30%,
this theorem proves that if 28.571428571428573 kg of pure salt is added,
the initial weight of the solution was 100 kg.
-/
theorem salt_solution_weight 
  (initial_concentration : Real) 
  (final_concentration : Real)
  (added_salt : Real) 
  (initial_weight : Real) :
  initial_concentration = 0.10 →
  final_concentration = 0.30 →
  added_salt = 28.571428571428573 →
  initial_concentration * initial_weight + added_salt = 
    final_concentration * (initial_weight + added_salt) →
  initial_weight = 100 := by
  sorry

#check salt_solution_weight

end NUMINAMATH_CALUDE_salt_solution_weight_l871_87157


namespace NUMINAMATH_CALUDE_average_marks_combined_l871_87174

theorem average_marks_combined (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ) 
  (h₁ : n₁ = 12) (h₂ : n₂ = 28) (h₃ : avg₁ = 40) (h₄ : avg₂ = 60) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℚ) = 54 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_l871_87174


namespace NUMINAMATH_CALUDE_segment_ratio_l871_87177

/-- Given a line segment GH with points E and F lying on it, 
    where GE is 3 times EH and GF is 7 times FH, 
    prove that EF is 1/8 of GH. -/
theorem segment_ratio (G E F H : Real) : 
  (E - G) = 3 * (H - E) →
  (F - G) = 7 * (H - F) →
  (F - E) = (1/8) * (H - G) := by
  sorry

end NUMINAMATH_CALUDE_segment_ratio_l871_87177


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l871_87196

theorem sqrt_equation_solution :
  let f : ℝ → ℝ := λ x => Real.sqrt (x + 9) - 2 * Real.sqrt (x - 2) + 3
  ∃ x₁ x₂ : ℝ, x₁ = 8 + 4 * Real.sqrt 2 ∧ x₂ = 8 - 4 * Real.sqrt 2 ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l871_87196


namespace NUMINAMATH_CALUDE_james_quiz_bowl_points_l871_87110

/-- Calculates the total points earned by a student in a quiz bowl game. -/
def quiz_bowl_points (total_rounds : ℕ) (questions_per_round : ℕ) (points_per_correct : ℕ) 
  (bonus_points : ℕ) (questions_missed : ℕ) : ℕ :=
  let total_questions := total_rounds * questions_per_round
  let correct_answers := total_questions - questions_missed
  let base_points := correct_answers * points_per_correct
  let full_rounds := total_rounds - (questions_missed + questions_per_round - 1) / questions_per_round
  let bonus_total := full_rounds * bonus_points
  base_points + bonus_total

/-- Theorem stating that James earned 64 points in the quiz bowl game. -/
theorem james_quiz_bowl_points : 
  quiz_bowl_points 5 5 2 4 1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_james_quiz_bowl_points_l871_87110


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l871_87182

/-- Represents a batsman's statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored) / (b.innings + 1)

/-- Theorem: A batsman's average after 12 innings is 58, given the conditions -/
theorem batsman_average_after_12th_innings 
  (b : Batsman)
  (h1 : b.innings = 11)
  (h2 : newAverage b 80 = b.average + 2)
  : newAverage b 80 = 58 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l871_87182


namespace NUMINAMATH_CALUDE_x_value_l871_87149

theorem x_value (x : ℝ) : x + Real.sqrt 81 = 25 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l871_87149


namespace NUMINAMATH_CALUDE_square_field_area_l871_87143

theorem square_field_area (side_length : ℝ) (h : side_length = 5) :
  side_length * side_length = 25 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l871_87143


namespace NUMINAMATH_CALUDE_omi_age_l871_87140

/-- Given the ages of Kimiko, Arlette, and Omi, prove Omi's age -/
theorem omi_age (kimiko_age : ℕ) (arlette_age : ℕ) (omi_age : ℕ) : 
  kimiko_age = 28 →
  arlette_age = (3 * kimiko_age) / 4 →
  (kimiko_age + arlette_age + omi_age) / 3 = 35 →
  omi_age = 56 := by
sorry

end NUMINAMATH_CALUDE_omi_age_l871_87140


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l871_87170

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  3 * x^2 + m * x + 2 = 0

-- Part I
theorem part_one (m : ℝ) : quadratic_equation m 2 → m = -7 := by
  sorry

-- Part II
theorem part_two :
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2/3 ∧ quadratic_equation (-5) x₁ ∧ quadratic_equation (-5) x₂) := by
  sorry

-- Part III
theorem part_three (m : ℝ) :
  m ≥ 5 →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l871_87170


namespace NUMINAMATH_CALUDE_vikkis_take_home_pay_is_correct_l871_87124

/-- Calculates Vikki's take-home pay after all deductions --/
def vikkis_take_home_pay (hours_worked : ℕ) (hourly_rate : ℚ) 
  (federal_tax_rate_low : ℚ) (federal_tax_rate_high : ℚ) (federal_tax_threshold : ℚ)
  (state_tax_rate : ℚ) (retirement_rate : ℚ) (insurance_rate : ℚ) (union_dues : ℚ) : ℚ :=
  let gross_earnings := hours_worked * hourly_rate
  let federal_tax_low := min federal_tax_threshold gross_earnings * federal_tax_rate_low
  let federal_tax_high := max 0 (gross_earnings - federal_tax_threshold) * federal_tax_rate_high
  let state_tax := gross_earnings * state_tax_rate
  let retirement := gross_earnings * retirement_rate
  let insurance := gross_earnings * insurance_rate
  let total_deductions := federal_tax_low + federal_tax_high + state_tax + retirement + insurance + union_dues
  gross_earnings - total_deductions

/-- Theorem stating that Vikki's take-home pay is $328.48 --/
theorem vikkis_take_home_pay_is_correct : 
  vikkis_take_home_pay 42 12 (15/100) (22/100) 300 (7/100) (6/100) (3/100) 5 = 328.48 := by
  sorry

end NUMINAMATH_CALUDE_vikkis_take_home_pay_is_correct_l871_87124


namespace NUMINAMATH_CALUDE_area_of_union_S_l871_87137

/-- A disc D in the 2D plane -/
structure Disc where
  center : ℝ × ℝ
  radius : ℝ

/-- The set S of discs D -/
def S : Set Disc :=
  {D : Disc | D.center.2 = D.center.1^2 - 3/4 ∧ 
              ∀ (x y : ℝ), (x - D.center.1)^2 + (y - D.center.2)^2 < D.radius^2 → y < 0}

/-- The area of the union of all discs in S -/
def unionArea (S : Set Disc) : ℝ := sorry

/-- Theorem stating the area of the union of discs in S -/
theorem area_of_union_S : unionArea S = (2 * Real.pi / 3) + (Real.sqrt 3 / 4) := by sorry

end NUMINAMATH_CALUDE_area_of_union_S_l871_87137


namespace NUMINAMATH_CALUDE_ratio_problem_l871_87152

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 4) : 
  (a + 2*b) / (b + 2*c) = 7/27 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l871_87152


namespace NUMINAMATH_CALUDE_fish_price_calculation_l871_87163

theorem fish_price_calculation (discount_rate : ℝ) (discounted_price : ℝ) (package_weight : ℝ) : 
  discount_rate = 0.6 →
  discounted_price = 4.5 →
  package_weight = 0.75 →
  (discounted_price / package_weight) / (1 - discount_rate) = 15 := by
sorry

end NUMINAMATH_CALUDE_fish_price_calculation_l871_87163


namespace NUMINAMATH_CALUDE_johns_share_ratio_l871_87119

theorem johns_share_ratio (total : ℕ) (johns_share : ℕ) 
  (h1 : total = 4800) (h2 : johns_share = 1600) : 
  johns_share / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_johns_share_ratio_l871_87119


namespace NUMINAMATH_CALUDE_homothety_composition_l871_87120

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a homothety in 3D space
structure Homothety3D where
  center : Point3D
  ratio : ℝ

-- Define a line in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Function to compose two homotheties
def compose_homotheties (h1 h2 : Homothety3D) : Homothety3D :=
  sorry

-- Function to check if a point lies on a line
def point_on_line (p : Point3D) (l : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem homothety_composition 
  (h1 h2 : Homothety3D) 
  (l : Line3D) :
  let h3 := compose_homotheties h1 h2
  point_on_line h3.center l ∧ 
  h3.ratio = h1.ratio * h2.ratio ∧
  point_on_line h1.center l ∧
  point_on_line h2.center l :=
sorry

end NUMINAMATH_CALUDE_homothety_composition_l871_87120


namespace NUMINAMATH_CALUDE_calculation_proof_l871_87111

theorem calculation_proof : 121 * (13 / 25) + 12 * (21 / 25) = 73 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l871_87111


namespace NUMINAMATH_CALUDE_parsley_sprigs_theorem_l871_87147

/-- Calculates the number of parsley sprigs left after decorating plates -/
def sprigs_left (initial_sprigs : ℕ) (whole_sprig_plates : ℕ) (half_sprig_plates : ℕ) : ℕ :=
  initial_sprigs - (whole_sprig_plates + (half_sprig_plates / 2))

/-- Proves that given the specific conditions, 11 sprigs are left -/
theorem parsley_sprigs_theorem :
  sprigs_left 25 8 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_parsley_sprigs_theorem_l871_87147


namespace NUMINAMATH_CALUDE_no_intersection_l871_87194

theorem no_intersection : ¬ ∃ x : ℝ, |3 * x + 6| = -|2 * x - 4| := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_l871_87194


namespace NUMINAMATH_CALUDE_find_number_l871_87116

theorem find_number (x y N : ℝ) (h1 : x / (2 * y) = N / 2) (h2 : (7 * x + 2 * y) / (x - 2 * y) = 23) : N = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l871_87116


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l871_87167

theorem cubic_sum_theorem (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = 3) 
  (h3 : a * b * c = 1) : 
  a^3 + b^3 + c^3 = 5 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l871_87167


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l871_87104

/-- Proves that in a rhombus with one diagonal of 62 meters and an area of 2480 square meters,
    the length of the other diagonal is 80 meters. -/
theorem rhombus_diagonal_length (d1 : ℝ) (area : ℝ) (d2 : ℝ) 
    (h1 : d1 = 62) 
    (h2 : area = 2480) 
    (h3 : area = (d1 * d2) / 2) : d2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l871_87104


namespace NUMINAMATH_CALUDE_expression_simplification_l871_87144

theorem expression_simplification (a : ℝ) :
  (a - (2*a - 1) / a) / ((1 - a^2) / (a^2 + a)) = a + 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l871_87144


namespace NUMINAMATH_CALUDE_equation_one_solution_system_of_equations_solution_l871_87165

-- Equation (1)
theorem equation_one_solution :
  ∃! x : ℚ, (2*x + 5) / 6 - (3*x - 2) / 8 = 1 :=
by sorry

-- System of equations (2)
theorem system_of_equations_solution :
  ∃! (x y : ℚ), (2*x - 1) / 5 + (3*y - 2) / 4 = 2 ∧
                (3*x + 1) / 5 - (3*y + 2) / 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_one_solution_system_of_equations_solution_l871_87165


namespace NUMINAMATH_CALUDE_family_probability_l871_87109

theorem family_probability :
  let p_boy : ℝ := 1/2
  let p_girl : ℝ := 1/2
  let num_children : ℕ := 4
  p_boy + p_girl = 1 →
  (1 : ℝ) - (p_boy ^ num_children + p_girl ^ num_children) = 7/8 :=
by sorry

end NUMINAMATH_CALUDE_family_probability_l871_87109


namespace NUMINAMATH_CALUDE_folded_paper_perimeter_ratio_l871_87186

/-- Represents the dimensions of a rectangular piece of paper --/
structure PaperDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular piece of paper --/
def perimeter (p : PaperDimensions) : ℝ :=
  2 * (p.length + p.width)

/-- Represents the paper after folding and cutting --/
structure FoldedPaper where
  original : PaperDimensions
  flap : PaperDimensions
  largest_rectangle : PaperDimensions

/-- The theorem to be proved --/
theorem folded_paper_perimeter_ratio 
  (paper : FoldedPaper) 
  (h1 : paper.original.length = 6 ∧ paper.original.width = 6)
  (h2 : paper.flap.length = 3 ∧ paper.flap.width = 3)
  (h3 : paper.largest_rectangle.length = 6 ∧ paper.largest_rectangle.width = 4.5) :
  perimeter paper.flap / perimeter paper.largest_rectangle = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_perimeter_ratio_l871_87186


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_ten_l871_87154

theorem sum_of_x_and_y_is_ten (x y : ℝ) (h1 : x = 25 / y) (h2 : x^2 + y^2 = 50) : x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_ten_l871_87154


namespace NUMINAMATH_CALUDE_inclined_plane_angle_theorem_l871_87112

/-- 
Given a system with two blocks connected by a cord over a frictionless pulley,
where one block of mass m is on a frictionless inclined plane and the other block
of mass M is hanging vertically, this theorem proves that if M = 1.5 * m and the
acceleration of the system is g/3, then the angle θ of the inclined plane
satisfies sin θ = 2/3.
-/
theorem inclined_plane_angle_theorem 
  (m : ℝ) 
  (M : ℝ) 
  (g : ℝ) 
  (θ : ℝ) 
  (h_mass : M = 1.5 * m) 
  (h_accel : m * g / 3 = m * g * (1 - Real.sin θ)) : 
  Real.sin θ = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inclined_plane_angle_theorem_l871_87112


namespace NUMINAMATH_CALUDE_binomial_12_3_l871_87160

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by sorry

end NUMINAMATH_CALUDE_binomial_12_3_l871_87160


namespace NUMINAMATH_CALUDE_valid_3x3_grid_exists_l871_87102

/-- Represents a county with a diagonal road -/
inductive County
  | NorthEast
  | SouthWest

/-- Represents a 3x3 grid of counties -/
def Grid := Fin 3 → Fin 3 → County

/-- Checks if two adjacent counties have compatible road directions -/
def compatible (c1 c2 : County) : Bool :=
  match c1, c2 with
  | County.NorthEast, County.SouthWest => true
  | County.SouthWest, County.NorthEast => true
  | _, _ => false

/-- Checks if the grid forms a valid closed path -/
def isValidPath (g : Grid) : Bool :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that a valid 3x3 grid configuration exists -/
theorem valid_3x3_grid_exists : ∃ g : Grid, isValidPath g := by
  sorry

end NUMINAMATH_CALUDE_valid_3x3_grid_exists_l871_87102


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l871_87162

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) = Real.sqrt 35 / 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l871_87162


namespace NUMINAMATH_CALUDE_integer_between_sqrt_2n_and_sqrt_5n_l871_87135

theorem integer_between_sqrt_2n_and_sqrt_5n (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℤ, Real.sqrt (2 * n) < k ∧ k < Real.sqrt (5 * n) := by
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt_2n_and_sqrt_5n_l871_87135


namespace NUMINAMATH_CALUDE_sandy_tokens_l871_87129

theorem sandy_tokens (total_tokens : ℕ) (num_siblings : ℕ) : 
  total_tokens = 1000000 →
  num_siblings = 4 →
  let sandy_share := total_tokens / 2
  let remaining_tokens := total_tokens - sandy_share
  let sibling_share := remaining_tokens / num_siblings
  sandy_share - sibling_share = 375000 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_tokens_l871_87129


namespace NUMINAMATH_CALUDE_race_head_start_l871_87142

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (22 / 19) * Vb) :
  ∃ H : ℝ, H / L = 3 / 22 ∧ L / Va = (L - H) / Vb :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l871_87142


namespace NUMINAMATH_CALUDE_cone_height_calculation_l871_87193

theorem cone_height_calculation (cylinder_base_area cone_base_area cylinder_height : ℝ) 
  (h1 : cylinder_base_area * cylinder_height = (1/3) * cone_base_area * cone_height)
  (h2 : cylinder_base_area / cone_base_area = 3/5)
  (h3 : cylinder_height = 8) : 
  cone_height = 14.4 := by
  sorry

#check cone_height_calculation

end NUMINAMATH_CALUDE_cone_height_calculation_l871_87193


namespace NUMINAMATH_CALUDE_yogurt_combinations_l871_87128

theorem yogurt_combinations (flavors : Nat) (toppings : Nat) : 
  flavors = 5 → toppings = 7 → 
  flavors * (1 + toppings.choose 1 + toppings.choose 2) = 145 := by
sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l871_87128


namespace NUMINAMATH_CALUDE_t_range_max_radius_equation_l871_87185

-- Define the circle equation
def circle_equation (x y t : ℝ) : Prop := x^2 + y^2 - 2*x + t^2 = 0

-- Theorem for the range of t
theorem t_range : ∀ x y t : ℝ, circle_equation x y t → -1 < t ∧ t < 1 := by sorry

-- Define the maximum radius
def max_radius (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Theorem for the circle equation when radius is maximum
theorem max_radius_equation : 
  (∃ t : ℝ, ∀ x y : ℝ, circle_equation x y t ∧ 
    (∀ t' : ℝ, circle_equation x y t' → 
      (x - 1)^2 + y^2 ≥ (x - 1)^2 + y^2)) → 
  ∀ x y : ℝ, max_radius x y := by sorry

end NUMINAMATH_CALUDE_t_range_max_radius_equation_l871_87185


namespace NUMINAMATH_CALUDE_multiply_fractions_l871_87188

theorem multiply_fractions : 12 * (1 / 15) * 30 = 24 := by
  sorry

end NUMINAMATH_CALUDE_multiply_fractions_l871_87188


namespace NUMINAMATH_CALUDE_lighting_effect_improves_l871_87168

theorem lighting_effect_improves (a b m : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m) : 
  a / b < (a + m) / (b + m) := by
  sorry

end NUMINAMATH_CALUDE_lighting_effect_improves_l871_87168


namespace NUMINAMATH_CALUDE_choose_four_from_nine_l871_87117

theorem choose_four_from_nine (n : ℕ) (k : ℕ) : n = 9 → k = 4 → Nat.choose n k = 126 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_nine_l871_87117


namespace NUMINAMATH_CALUDE_correct_sum_l871_87164

theorem correct_sum (x y : ℕ+) (h1 : x - y = 4) (h2 : x * y = 132) : x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_correct_sum_l871_87164


namespace NUMINAMATH_CALUDE_sin_thirty_degrees_l871_87103

/-- Given a point Q on the unit circle 30° counterclockwise from (1,0),
    and E as the foot of the altitude from Q to the x-axis,
    prove that sin(30°) = 1/2 -/
theorem sin_thirty_degrees (Q : ℝ × ℝ) (E : ℝ × ℝ) :
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (Q.1 = Real.cos (30 * π / 180)) →  -- Q is 30° counterclockwise from (1,0)
  (Q.2 = Real.sin (30 * π / 180)) →
  (E.1 = Q.1 ∧ E.2 = 0) →  -- E is the foot of the altitude from Q to the x-axis
  Real.sin (30 * π / 180) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_thirty_degrees_l871_87103


namespace NUMINAMATH_CALUDE_negation_of_exists_lt_is_forall_ge_l871_87181

theorem negation_of_exists_lt_is_forall_ge :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_lt_is_forall_ge_l871_87181
