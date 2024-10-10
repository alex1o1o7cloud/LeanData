import Mathlib

namespace sequence_general_term_l3021_302103

/-- Given a sequence {a_n} where S_n is the sum of the first n terms 
    and S_n = (1/2)(1 - a_n), prove that a_n = (1/3)^n -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = (1/2) * (1 - a n)) :
  ∀ n, a n = (1/3)^n := by
  sorry

end sequence_general_term_l3021_302103


namespace not_divides_power_diff_l3021_302102

theorem not_divides_power_diff (m n : ℕ) : 
  m ≥ 3 → n ≥ 3 → Odd m → Odd n → ¬(2^m - 1 ∣ 3^n - 1) := by
  sorry

end not_divides_power_diff_l3021_302102


namespace midpoint_fraction_l3021_302144

theorem midpoint_fraction : 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 7
  let midpoint := (a + b) / 2
  midpoint = (41 : ℚ) / 56 := by
sorry

end midpoint_fraction_l3021_302144


namespace smallest_part_of_proportional_division_l3021_302179

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ (x : ℕ), x > 0 ∧
    (a * x).gcd (b * x) = 1 ∧
    (a * x).gcd (c * x) = 1 ∧
    (b * x).gcd (c * x) = 1 ∧
    a * x + b * x + c * x = total ∧
    min (a * x) (min (b * x) (c * x)) = 24 :=
by sorry

end smallest_part_of_proportional_division_l3021_302179


namespace number_of_choices_l3021_302189

-- Define the total number of subjects
def total_subjects : ℕ := 6

-- Define the number of science subjects
def science_subjects : ℕ := 3

-- Define the number of humanities subjects
def humanities_subjects : ℕ := 3

-- Define the number of subjects to be chosen
def subjects_to_choose : ℕ := 3

-- Define the minimum number of science subjects to be chosen
def min_science_subjects : ℕ := 2

-- Theorem statement
theorem number_of_choices :
  (Nat.choose science_subjects 2 * Nat.choose humanities_subjects 1) +
  (Nat.choose science_subjects 3) = 10 := by
  sorry

end number_of_choices_l3021_302189


namespace square_difference_l3021_302139

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end square_difference_l3021_302139


namespace f_inequality_range_l3021_302116

/-- The function f(x) = |x-a| + |x+2| -/
def f (a x : ℝ) : ℝ := |x - a| + |x + 2|

/-- The theorem stating the range of a values for which ∃x₀ ∈ ℝ such that f(x₀) ≤ |2a+1| -/
theorem f_inequality_range (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ ≤ |2*a + 1|) ↔ a ≤ -1 ∨ a ≥ 1 := by
  sorry

end f_inequality_range_l3021_302116


namespace theater_ticket_sales_l3021_302149

/-- Calculates the total ticket sales for a theater performance --/
theorem theater_ticket_sales 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (total_attendance : ℕ) 
  (child_attendance : ℕ) : 
  adult_price = 8 → 
  child_price = 1 → 
  total_attendance = 22 → 
  child_attendance = 18 → 
  (total_attendance - child_attendance) * adult_price + child_attendance * child_price = 50 := by
  sorry

end theater_ticket_sales_l3021_302149


namespace sum_of_coordinates_A_l3021_302112

/-- Given three points A, B, and C in a plane satisfying certain conditions,
    prove that the sum of coordinates of A is 22. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (C.2 - A.2) / (B.2 - A.2) = 1/3 →
  B = (2, 8) →
  C = (5, 11) →
  A.1 + A.2 = 22 := by
sorry

end sum_of_coordinates_A_l3021_302112


namespace solve_linear_equation_l3021_302151

theorem solve_linear_equation (x y a : ℚ) : 
  x = 2 → y = a → 2 * x - 3 * y = 5 → a = -1/3 := by sorry

end solve_linear_equation_l3021_302151


namespace equality_multiplication_negative_two_l3021_302111

theorem equality_multiplication_negative_two (m n : ℝ) : m = n → -2 * m = -2 * n := by
  sorry

end equality_multiplication_negative_two_l3021_302111


namespace output_for_three_l3021_302133

/-- Represents the output of the program based on the input x -/
def program_output (x : ℤ) : ℤ :=
  if x < 0 then -1
  else if x = 0 then 0
  else 1

/-- Theorem stating that when x = 3, the program outputs 1 -/
theorem output_for_three : program_output 3 = 1 := by
  sorry

end output_for_three_l3021_302133


namespace gcd_lcm_product_300_l3021_302180

theorem gcd_lcm_product_300 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 300) :
  ∃! (s : Finset ℕ), s.card = 8 ∧ ∀ d, d ∈ s ↔ ∃ (x y : ℕ+), x * y = 300 ∧ Nat.gcd x y = d :=
sorry

end gcd_lcm_product_300_l3021_302180


namespace rectangle_area_equals_perimeter_l3021_302182

theorem rectangle_area_equals_perimeter (x : ℝ) : 
  let length : ℝ := 4 * x
  let width : ℝ := 2 * x + 6
  length > 0 ∧ width > 0 →
  length * width = 2 * (length + width) →
  x = (-3 + Real.sqrt 33) / 4 :=
by sorry

end rectangle_area_equals_perimeter_l3021_302182


namespace problem_statement_l3021_302143

theorem problem_statement (a b : ℝ) (h1 : a * b = 2) (h2 : a + b = 3) :
  a^2 * b + a * b^2 = 6 := by
  sorry

end problem_statement_l3021_302143


namespace negation_of_exists_negation_of_quadratic_equation_l3021_302174

theorem negation_of_exists (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 - 3*x + 2 = 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0) :=
by sorry

end negation_of_exists_negation_of_quadratic_equation_l3021_302174


namespace output_for_input_12_l3021_302160

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 35 then
    step1 + 10
  else
    step1 - 7

theorem output_for_input_12 :
  function_machine 12 = 29 := by sorry

end output_for_input_12_l3021_302160


namespace increase_in_average_weight_l3021_302137

/-- Theorem: Increase in average weight when replacing a person in a group -/
theorem increase_in_average_weight 
  (group_size : ℕ) 
  (old_weight new_weight : ℝ) 
  (h1 : group_size = 8)
  (h2 : old_weight = 55)
  (h3 : new_weight = 87) :
  (new_weight - old_weight) / group_size = 4 := by
  sorry

end increase_in_average_weight_l3021_302137


namespace root_implies_constant_value_l3021_302118

theorem root_implies_constant_value (c : ℝ) : 
  ((-5 : ℝ)^2 = c^2) → (c = 5 ∨ c = -5) := by
  sorry

end root_implies_constant_value_l3021_302118


namespace equation_solutions_no_other_solutions_l3021_302107

/-- Definition of the factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The main theorem stating the solutions to the equation -/
theorem equation_solutions :
  ∀ x y z : ℕ, 2^x + 3^y + 7 = factorial z ↔ (x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5) :=
by sorry

/-- Auxiliary theorem: There are no other solutions -/
theorem no_other_solutions :
  ∀ x y z : ℕ, 2^x + 3^y + 7 = factorial z →
  (x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5) :=
by sorry

end equation_solutions_no_other_solutions_l3021_302107


namespace matts_bike_ride_l3021_302145

/-- Given Matt's bike ride scenario, prove the remaining distance after the second sign. -/
theorem matts_bike_ride (total_distance : ℕ) (distance_to_first_sign : ℕ) (distance_between_signs : ℕ)
  (h1 : total_distance = 1000)
  (h2 : distance_to_first_sign = 350)
  (h3 : distance_between_signs = 375) :
  total_distance - (distance_to_first_sign + distance_between_signs) = 275 := by
  sorry

end matts_bike_ride_l3021_302145


namespace determinant_of_cubic_roots_l3021_302172

theorem determinant_of_cubic_roots (p q : ℝ) (a b c : ℂ) : 
  (a^3 + p*a + q = 0) → 
  (b^3 + p*b + q = 0) → 
  (c^3 + p*c + q = 0) → 
  (Complex.abs a ≠ 0) →
  (Complex.abs b ≠ 0) →
  (Complex.abs c ≠ 0) →
  let matrix := !![2 + a^2, 1, 1; 1, 2 + b^2, 1; 1, 1, 2 + c^2]
  Matrix.det matrix = (2*p^2 : ℂ) - 4*q + q^2 := by
sorry

end determinant_of_cubic_roots_l3021_302172


namespace factorization_of_four_a_squared_minus_one_l3021_302119

theorem factorization_of_four_a_squared_minus_one (a : ℝ) : 4 * a^2 - 1 = (2*a - 1) * (2*a + 1) := by
  sorry

end factorization_of_four_a_squared_minus_one_l3021_302119


namespace jelly_bean_probability_l3021_302138

/-- The probability of selecting either a blue or purple jelly bean from a bag -/
theorem jelly_bean_probability : 
  let red : ℕ := 4
  let green : ℕ := 5
  let yellow : ℕ := 9
  let blue : ℕ := 7
  let purple : ℕ := 10
  let total : ℕ := red + green + yellow + blue + purple
  (blue + purple : ℚ) / total = 17 / 35 := by sorry

end jelly_bean_probability_l3021_302138


namespace birthday_on_sunday_l3021_302176

/-- Represents days of the week -/
inductive Day : Type
  | sunday : Day
  | monday : Day
  | tuesday : Day
  | wednesday : Day
  | thursday : Day
  | friday : Day
  | saturday : Day

/-- Returns the day before the given day -/
def dayBefore (d : Day) : Day :=
  match d with
  | Day.sunday => Day.saturday
  | Day.monday => Day.sunday
  | Day.tuesday => Day.monday
  | Day.wednesday => Day.tuesday
  | Day.thursday => Day.wednesday
  | Day.friday => Day.thursday
  | Day.saturday => Day.friday

/-- Returns the day that is n days after the given day -/
def daysAfter (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => daysAfter (match d with
    | Day.sunday => Day.monday
    | Day.monday => Day.tuesday
    | Day.tuesday => Day.wednesday
    | Day.wednesday => Day.thursday
    | Day.thursday => Day.friday
    | Day.friday => Day.saturday
    | Day.saturday => Day.sunday) m

theorem birthday_on_sunday (today : Day) (birthday : Day) : 
  today = Day.thursday → 
  daysAfter (dayBefore birthday) 2 = dayBefore (dayBefore (dayBefore today)) → 
  birthday = Day.sunday := by
  sorry

end birthday_on_sunday_l3021_302176


namespace problem_solution_l3021_302136

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2*a| + |x - a|

theorem problem_solution :
  ∀ a : ℝ, a ≠ 0 →
  (∀ x : ℝ, f a x ≥ |x - 2*a| + |x - a|) →
  (∀ x : ℝ, (f 1 x > 2 ↔ x < 1/2 ∨ x > 5/2)) ∧
  (∀ b : ℝ, b ≠ 0 → f a b ≥ f a a) ∧
  (∀ b : ℝ, b ≠ 0 → (f a b = f a a ↔ (2*a - b)*(b - a) ≥ 0 ∨ 2*a - b = 0 ∨ b - a = 0)) :=
by sorry

end problem_solution_l3021_302136


namespace unique_solution_quadratic_l3021_302198

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = m + 3 * x) ↔ m = 6 := by
  sorry

end unique_solution_quadratic_l3021_302198


namespace matrix_equality_l3021_302101

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]

def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]

theorem matrix_equality (x y z w : ℝ) (h1 : A * B x y z w = B x y z w * A) (h2 : 4 * y ≠ z) :
  (x - w) / (z - 4 * y) = 1 / 2 := by
  sorry

end matrix_equality_l3021_302101


namespace adult_ticket_price_l3021_302117

theorem adult_ticket_price 
  (child_price : ℕ)
  (total_attendance : ℕ)
  (total_collection : ℕ)
  (children_attendance : ℕ) :
  child_price = 25 →
  total_attendance = 280 →
  total_collection = 140 * 100 →
  children_attendance = 80 →
  (total_attendance - children_attendance) * 60 + children_attendance * child_price = total_collection :=
by sorry

end adult_ticket_price_l3021_302117


namespace arithmetic_problem_l3021_302141

theorem arithmetic_problem : 300 + 5 * 8 = 340 := by
  sorry

end arithmetic_problem_l3021_302141


namespace anita_apples_l3021_302131

/-- The number of apples Anita has, given the number of students and apples per student -/
def total_apples (num_students : ℕ) (apples_per_student : ℕ) : ℕ :=
  num_students * apples_per_student

/-- Theorem: Anita has 360 apples -/
theorem anita_apples : total_apples 60 6 = 360 := by
  sorry

end anita_apples_l3021_302131


namespace g_of_three_l3021_302187

theorem g_of_three (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3*x - 2) = 4*x + 1) : g 3 = 23/3 := by
  sorry

end g_of_three_l3021_302187


namespace existence_of_special_sequence_l3021_302155

theorem existence_of_special_sequence (n : ℕ) : 
  ∃ (a : ℕ → ℕ), 
    (∀ i j, i < j → j ≤ n → a i > a j) ∧ 
    (∀ i, i < n → a i ∣ (a (i + 1))^2) ∧
    (∀ i j, i ≠ j → i ≤ n → j ≤ n → ¬(a i ∣ a j)) :=
by sorry

end existence_of_special_sequence_l3021_302155


namespace stella_unpaid_leave_l3021_302163

/-- Calculates the number of months of unpaid leave taken by an employee given their monthly income and actual annual income. -/
def unpaid_leave_months (monthly_income : ℕ) (actual_annual_income : ℕ) : ℕ :=
  12 - actual_annual_income / monthly_income

/-- Proves that given Stella's monthly income of 4919 dollars and her actual annual income of 49190 dollars, the number of months of unpaid leave she took is 2. -/
theorem stella_unpaid_leave :
  unpaid_leave_months 4919 49190 = 2 := by
  sorry

end stella_unpaid_leave_l3021_302163


namespace no_roots_lost_l3021_302161

theorem no_roots_lost (x : ℝ) : 
  (x^4 + x^3 + x^2 + x + 1 = 0) ↔ (x^2 + x + 1 + 1/x + 1/x^2 = 0) :=
by sorry

end no_roots_lost_l3021_302161


namespace isosceles_triangle_legs_l3021_302192

/-- An isosceles triangle with integer side lengths and perimeter 12 -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ
  perimeter_eq : leg + leg + base = 12
  triangle_inequality : base < leg + leg ∧ leg < base + leg

/-- The possible leg lengths of an isosceles triangle with perimeter 12 -/
def possibleLegLengths : Set ℕ :=
  {n : ℕ | ∃ (t : IsoscelesTriangle), t.leg = n}

theorem isosceles_triangle_legs :
  possibleLegLengths = {4, 5} := by sorry

end isosceles_triangle_legs_l3021_302192


namespace fourth_root_equation_solutions_l3021_302162

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x ^ (1/4) = 15 / (8 - x ^ (1/4))) ↔ (x = 625 ∨ x = 81) :=
by sorry

end fourth_root_equation_solutions_l3021_302162


namespace max_value_m_l3021_302169

theorem max_value_m (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, m / (3 * a + b) - 3 / a - 1 / b ≤ 0) →
  (∃ m : ℝ, m = 16 ∧ ∀ m' : ℝ, m' / (3 * a + b) - 3 / a - 1 / b ≤ 0 → m' ≤ m) :=
by sorry

end max_value_m_l3021_302169


namespace john_smith_payment_l3021_302147

-- Define the number of cakes
def num_cakes : ℕ := 3

-- Define the cost per cake in cents (to avoid floating-point numbers)
def cost_per_cake : ℕ := 1200

-- Define the number of people sharing the cost
def num_people : ℕ := 2

-- Theorem to prove
theorem john_smith_payment :
  (num_cakes * cost_per_cake) / num_people = 1800 := by
  sorry

end john_smith_payment_l3021_302147


namespace equation_holds_for_negative_eight_l3021_302115

theorem equation_holds_for_negative_eight :
  let t : ℝ := -8
  let f (x : ℝ) : ℝ := (2 / (x + 3)) + (x / (x + 3)) - (4 / (x + 3))
  f t = 2 := by
sorry

end equation_holds_for_negative_eight_l3021_302115


namespace flour_requirement_undetermined_l3021_302171

/-- Represents the recipe requirements and current state of baking --/
structure BakingScenario where
  sugar_required : ℕ
  sugar_added : ℕ
  flour_added : ℕ

/-- Represents the unknown total flour required by the recipe --/
def total_flour_required : ℕ → Prop := fun _ => True

/-- Theorem stating that the total flour required cannot be determined --/
theorem flour_requirement_undetermined (scenario : BakingScenario) 
  (h1 : scenario.sugar_required = 11)
  (h2 : scenario.sugar_added = 10)
  (h3 : scenario.flour_added = 12) :
  ∀ n : ℕ, total_flour_required n :=
by sorry

end flour_requirement_undetermined_l3021_302171


namespace prob_less_than_four_at_least_six_of_seven_l3021_302104

/-- The probability of rolling a number less than four on a fair die. -/
def p_less_than_four : ℚ := 1/2

/-- The number of times the die is rolled. -/
def num_rolls : ℕ := 7

/-- The minimum number of successful rolls (less than four) we're interested in. -/
def min_successes : ℕ := 6

/-- The probability of rolling a number less than four at least 'min_successes' times in 'num_rolls' rolls. -/
def probability_at_least_min_successes : ℚ :=
  (Finset.range (num_rolls - min_successes + 1)).sum fun k =>
    (Nat.choose num_rolls (num_rolls - k)) *
    (p_less_than_four ^ (num_rolls - k)) *
    ((1 - p_less_than_four) ^ k)

/-- The main theorem: The probability of rolling a number less than four at least six times in seven rolls of a fair die is 15/128. -/
theorem prob_less_than_four_at_least_six_of_seven :
  probability_at_least_min_successes = 15/128 := by
  sorry

end prob_less_than_four_at_least_six_of_seven_l3021_302104


namespace x_greater_than_ln_one_plus_x_l3021_302170

theorem x_greater_than_ln_one_plus_x {x : ℝ} (h : x > 0) : x > Real.log (1 + x) := by
  sorry

end x_greater_than_ln_one_plus_x_l3021_302170


namespace quadratic_inequality_solution_l3021_302185

theorem quadratic_inequality_solution (a : ℝ) (ha : a > 0) :
  (∀ x : ℝ, ax^2 - (a + 1)*x + 1 < 0 ↔ 
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1/a) ∨ 
     (a > 1 ∧ 1/a < x ∧ x < 1))) ∧
  (a = 1 → ∀ x : ℝ, ¬(x^2 - 2*x + 1 < 0)) :=
by sorry

end quadratic_inequality_solution_l3021_302185


namespace negation_of_forall_inequality_negation_of_specific_inequality_l3021_302153

theorem negation_of_forall_inequality (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, x ≤ 1 → p x) ↔ (∃ x : ℝ, x ≤ 1 ∧ ¬(p x)) := by sorry

theorem negation_of_specific_inequality :
  (¬ ∀ x : ℝ, x ≤ 1 → x^2 - 2*x + 1 ≥ 0) ↔ (∃ x : ℝ, x ≤ 1 ∧ x^2 - 2*x + 1 < 0) := by sorry

end negation_of_forall_inequality_negation_of_specific_inequality_l3021_302153


namespace maximize_f_l3021_302129

-- Define the function f
def f (a b c x y z : ℝ) : ℝ := a * x + b * y + c * z

-- State the theorem
theorem maximize_f (a b c : ℝ) :
  (∀ x y z : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ -5 ≤ y ∧ y ≤ 5 ∧ -5 ≤ z ∧ z ≤ 5) →
  f a b c 3 1 1 > f a b c 2 1 1 →
  f a b c 2 2 3 > f a b c 2 3 4 →
  f a b c 3 3 4 > f a b c 3 3 3 →
  ∀ x y z : ℝ, -5 ≤ x ∧ x ≤ 5 ∧ -5 ≤ y ∧ y ≤ 5 ∧ -5 ≤ z ∧ z ≤ 5 →
  f a b c x y z ≤ f a b c 5 (-5) 5 :=
by sorry

end maximize_f_l3021_302129


namespace area_implies_m_value_existence_implies_a_range_l3021_302175

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Theorem for part (1)
theorem area_implies_m_value (m : ℝ) (h1 : m > 3) 
  (h2 : (1/2) * ((m - 1)/2 - (-(m + 1)/2) + 3) * (m - 3) = 7/2) : 
  m = 4 := by sorry

-- Theorem for part (2)
theorem existence_implies_a_range (a : ℝ) 
  (h : ∃ x ∈ Set.Icc 0 2, f x ≥ |a - 3|) :
  -2 ≤ a ∧ a ≤ 8 := by sorry

end area_implies_m_value_existence_implies_a_range_l3021_302175


namespace f_extrema_on_interval_1_f_extrema_on_interval_2_l3021_302154

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Theorem for the interval [-2, 0]
theorem f_extrema_on_interval_1 :
  (∀ x ∈ Set.Icc (-2) 0, f x ≤ 2) ∧
  (∀ x ∈ Set.Icc (-2) 0, f x ≥ 10) ∧
  (∃ x ∈ Set.Icc (-2) 0, f x = 2) ∧
  (∃ x ∈ Set.Icc (-2) 0, f x = 10) :=
sorry

-- Theorem for the interval [2, 3]
theorem f_extrema_on_interval_2 :
  (∀ x ∈ Set.Icc 2 3, f x ≤ 5) ∧
  (∀ x ∈ Set.Icc 2 3, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 2 3, f x = 5) ∧
  (∃ x ∈ Set.Icc 2 3, f x = 2) :=
sorry

end f_extrema_on_interval_1_f_extrema_on_interval_2_l3021_302154


namespace existence_of_large_n_with_same_digit_occurrences_l3021_302195

open Nat

-- Define a function to check if two numbers have the same digit occurrences
def sameDigitOccurrences (a b : ℕ) : Prop := sorry

-- Define the theorem
theorem existence_of_large_n_with_same_digit_occurrences :
  ∃ n : ℕ, n > 10^100 ∧
    sameDigitOccurrences (n^2) ((n+1)^2) := by
  sorry

end existence_of_large_n_with_same_digit_occurrences_l3021_302195


namespace part1_part2_l3021_302148

-- Part 1
theorem part1 (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0)
  (hab : a + b = 10) (hxy : a / x + b / y = 1) (hmin : ∀ x' y', x' > 0 → y' > 0 → a / x' + b / y' = 1 → x' + y' ≥ 18) :
  (a = 2 ∧ b = 8) ∨ (a = 8 ∧ b = 2) := by sorry

-- Part 2
theorem part2 :
  ∃ a : ℝ, a > 0 ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * Real.sqrt (2 * x * y) ≤ a * (x + y)) ∧
  (∀ a' : ℝ, a' > 0 → (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * Real.sqrt (2 * x * y) ≤ a' * (x + y)) → a ≤ a') ∧
  a = 2 := by sorry

end part1_part2_l3021_302148


namespace unique_power_of_two_product_l3021_302164

theorem unique_power_of_two_product (a b : ℕ) :
  (∃ k : ℕ, (a + b^2) * (b + a^2) = 2^k) ↔ (a = 1 ∧ b = 1) :=
by sorry

end unique_power_of_two_product_l3021_302164


namespace hyperbola_equation_l3021_302110

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if a circle with diameter equal to the distance between its foci
    intersects one of its asymptotes at the point (3,4),
    then the equation of the hyperbola is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ 3^2 + 4^2 = c^2) →
  (∃ (k : ℝ), k = b/a ∧ 4/3 = k) →
  a^2 = 9 ∧ b^2 = 16 :=
by sorry

end hyperbola_equation_l3021_302110


namespace prob_all_white_value_l3021_302159

/-- The number of small cubes forming the larger cube -/
def num_cubes : ℕ := 8

/-- The probability of a single small cube showing a white face after flipping -/
def prob_white_face : ℚ := 5/6

/-- The probability of all surfaces of the larger cube becoming white after flipping -/
def prob_all_white : ℚ := (prob_white_face ^ num_cubes).num / (prob_white_face ^ num_cubes).den

theorem prob_all_white_value : prob_all_white = 390625/1679616 := by sorry

end prob_all_white_value_l3021_302159


namespace function_min_value_l3021_302186

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

-- State the theorem
theorem function_min_value 
  (h_max : ∃ (m : ℝ), ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3 ∧ ∃ y ∈ Set.Icc (-2 : ℝ) 2, f y m = 3) :
  ∃ (m : ℝ), ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≥ -37 ∧ ∃ y ∈ Set.Icc (-2 : ℝ) 2, f y m = -37 :=
sorry

end function_min_value_l3021_302186


namespace equation_solution_complex_l3021_302121

theorem equation_solution_complex (a b : ℂ) : 
  a ≠ 0 → 
  a + b ≠ 0 → 
  (a + b) / a = 3 * b / (a + b) → 
  (¬(a.im = 0) ∧ b.im = 0) ∨ (a.im = 0 ∧ ¬(b.im = 0)) ∨ (¬(a.im = 0) ∧ ¬(b.im = 0)) :=
sorry

end equation_solution_complex_l3021_302121


namespace simplify_tan_product_l3021_302140

theorem simplify_tan_product : (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end simplify_tan_product_l3021_302140


namespace circumscribed_sphere_surface_area_l3021_302124

/-- The surface area of a sphere circumscribed around a rectangular solid -/
theorem circumscribed_sphere_surface_area 
  (length width height : ℝ) 
  (h_length : length = 2)
  (h_width : width = 2)
  (h_height : height = 2 * Real.sqrt 2) : 
  4 * Real.pi * ((length^2 + width^2 + height^2) / 4) = 16 * Real.pi := by
  sorry

end circumscribed_sphere_surface_area_l3021_302124


namespace product_of_successive_numbers_l3021_302158

theorem product_of_successive_numbers : 
  let n : Real := 88.49858755935034
  let product := n * (n + 1)
  ∃ ε > 0, |product - 7913| < ε :=
by
  sorry

end product_of_successive_numbers_l3021_302158


namespace largest_remaining_number_l3021_302156

/-- Represents the original number sequence as a list of digits -/
def originalSequence : List Nat := sorry

/-- Represents the result after removing 100 digits -/
def resultSequence : List Nat := sorry

/-- The number of digits to remove -/
def digitsToRemove : Nat := 100

/-- Checks if a sequence is a valid subsequence of another sequence -/
def isValidSubsequence (sub seq : List Nat) : Prop := sorry

/-- Checks if a number represented as a list of digits is greater than another -/
def isGreaterThan (a b : List Nat) : Prop := sorry

theorem largest_remaining_number :
  isValidSubsequence resultSequence originalSequence ∧
  resultSequence.length = originalSequence.length - digitsToRemove ∧
  (∀ (other : List Nat), 
    isValidSubsequence other originalSequence → 
    other.length = originalSequence.length - digitsToRemove →
    isGreaterThan resultSequence other ∨ resultSequence = other) :=
sorry

end largest_remaining_number_l3021_302156


namespace average_of_data_set_l3021_302157

def data_set : List ℤ := [7, 5, -2, 5, 10]

theorem average_of_data_set :
  (data_set.sum : ℚ) / data_set.length = 5 := by
  sorry

end average_of_data_set_l3021_302157


namespace fraction_equivalence_and_decimal_l3021_302178

theorem fraction_equivalence_and_decimal : 
  let original : ℚ := 2 / 4
  let equiv1 : ℚ := 6 / 12
  let equiv2 : ℚ := 20 / 40
  let decimal : ℝ := 0.5
  (original = equiv1) ∧ (original = equiv2) ∧ (original = decimal) := by
  sorry

end fraction_equivalence_and_decimal_l3021_302178


namespace robie_cards_l3021_302167

/-- The number of cards in each box -/
def cards_per_box : ℕ := 10

/-- The number of cards not placed in a box -/
def loose_cards : ℕ := 5

/-- The number of boxes Robie gave away -/
def boxes_given_away : ℕ := 2

/-- The number of boxes Robie has left -/
def boxes_left : ℕ := 5

/-- The total number of cards Robie had in the beginning -/
def total_cards : ℕ := (boxes_given_away + boxes_left) * cards_per_box + loose_cards

theorem robie_cards : total_cards = 75 := by
  sorry

end robie_cards_l3021_302167


namespace max_y_value_l3021_302109

theorem max_y_value (x y : ℝ) :
  (Real.log (x + y) / Real.log (x^2 + y^2) ≥ 1) →
  y ≤ 1/2 + Real.sqrt 2 / 2 :=
by sorry

end max_y_value_l3021_302109


namespace polynomial_factorization_l3021_302106

theorem polynomial_factorization (m : ℝ) : 
  -4 * m^3 + 4 * m^2 - m = -m * (2*m - 1)^2 := by sorry

end polynomial_factorization_l3021_302106


namespace complex_modulus_problem_l3021_302134

theorem complex_modulus_problem (z : ℂ) : z = (1 - 3*I) / (1 + 2*I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l3021_302134


namespace sqrt_difference_equality_l3021_302128

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (36 - 9) = Real.sqrt 170 - 3 * Real.sqrt 3 := by
  sorry

end sqrt_difference_equality_l3021_302128


namespace friday_temperature_l3021_302152

def monday_temp : ℝ := 40

theorem friday_temperature 
  (h1 : (monday_temp + tuesday_temp + wednesday_temp + thursday_temp) / 4 = 48)
  (h2 : (tuesday_temp + wednesday_temp + thursday_temp + friday_temp) / 4 = 46) :
  friday_temp = 32 := by
sorry

end friday_temperature_l3021_302152


namespace parabola_focus_focus_of_specific_parabola_l3021_302188

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ y - a * x^2
  ∃! p : ℝ × ℝ, f p = 0 ∧ p.1 = 0 ∧ p.2 = 1 / (4 * a) :=
sorry

/-- The focus of the parabola y = 4x^2 is at (0, 1/16) -/
theorem focus_of_specific_parabola :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ y - 4 * x^2
  ∃! p : ℝ × ℝ, f p = 0 ∧ p.1 = 0 ∧ p.2 = 1 / 16 :=
sorry

end parabola_focus_focus_of_specific_parabola_l3021_302188


namespace problem_1_l3021_302184

theorem problem_1 : (-8) + 10 - 2 + (-1) = -1 := by sorry

end problem_1_l3021_302184


namespace poster_wall_width_l3021_302105

/-- Calculates the minimum wall width required to attach a given number of posters with specified width and overlap. -/
def minimumWallWidth (posterWidth : ℕ) (overlap : ℕ) (numPosters : ℕ) : ℕ :=
  posterWidth + (numPosters - 1) * (posterWidth - overlap)

/-- Theorem stating that 15 posters of width 30 cm, overlapping by 2 cm, require a wall width of 422 cm. -/
theorem poster_wall_width :
  minimumWallWidth 30 2 15 = 422 := by
  sorry

end poster_wall_width_l3021_302105


namespace car_selling_problem_l3021_302173

/-- Represents the selling price and profit information for two types of cars -/
structure CarInfo where
  price_a : ℕ  -- Selling price of type A car in yuan
  price_b : ℕ  -- Selling price of type B car in yuan
  profit_a : ℕ  -- Profit from selling one type A car in yuan
  profit_b : ℕ  -- Profit from selling one type B car in yuan

/-- Represents a purchasing plan -/
structure PurchasePlan where
  count_a : ℕ  -- Number of type A cars purchased
  count_b : ℕ  -- Number of type B cars purchased

/-- Theorem stating the properties of the car selling problem -/
theorem car_selling_problem (info : CarInfo) 
  (h1 : 2 * info.price_a + 3 * info.price_b = 800000)
  (h2 : 3 * info.price_a + 2 * info.price_b = 950000)
  (h3 : info.profit_a = 8000)
  (h4 : info.profit_b = 5000) :
  info.price_a = 250000 ∧ 
  info.price_b = 100000 ∧ 
  (∃ (plans : Finset PurchasePlan), 
    (∀ plan ∈ plans, plan.count_a * info.price_a + plan.count_b * info.price_b = 2000000) ∧
    plans.card = 3 ∧
    (∀ plan ∈ plans, ∀ other_plan : PurchasePlan, 
      other_plan.count_a * info.price_a + other_plan.count_b * info.price_b = 2000000 →
      other_plan ∈ plans)) ∧
  (∃ (max_profit : ℕ), 
    max_profit = 91000 ∧
    ∀ plan : PurchasePlan, 
      plan.count_a * info.price_a + plan.count_b * info.price_b = 2000000 →
      plan.count_a * info.profit_a + plan.count_b * info.profit_b ≤ max_profit) := by
  sorry


end car_selling_problem_l3021_302173


namespace max_different_digits_is_eight_l3021_302146

/-- A natural number satisfying the divisibility condition -/
def DivisibleNumber (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ Finset.range 10 → d ≠ 0 → (n.digits 10).contains d → n % d = 0

/-- The maximum number of different digits in a DivisibleNumber -/
def MaxDifferentDigits : ℕ := 8

/-- Theorem stating the maximum number of different digits in a DivisibleNumber -/
theorem max_different_digits_is_eight :
  ∃ n : ℕ, DivisibleNumber n ∧ (n.digits 10).card = MaxDifferentDigits ∧
  ∀ m : ℕ, DivisibleNumber m → (m.digits 10).card ≤ MaxDifferentDigits :=
sorry

end max_different_digits_is_eight_l3021_302146


namespace percentage_literate_inhabitants_l3021_302191

theorem percentage_literate_inhabitants (total_inhabitants : ℕ) 
  (male_percentage : ℚ) (literate_male_percentage : ℚ) (literate_female_percentage : ℚ)
  (h1 : total_inhabitants = 1000)
  (h2 : male_percentage = 60 / 100)
  (h3 : literate_male_percentage = 20 / 100)
  (h4 : literate_female_percentage = 325 / 1000) : 
  (↑(total_inhabitants * (male_percentage * literate_male_percentage * total_inhabitants + 
    (1 - male_percentage) * literate_female_percentage * total_inhabitants)) / 
    (↑total_inhabitants * 1000) : ℚ) = 25 / 100 := by
  sorry

end percentage_literate_inhabitants_l3021_302191


namespace coefficient_x_squared_in_expansion_l3021_302183

theorem coefficient_x_squared_in_expansion :
  let expansion := (fun x => (x - 2/x)^8)
  let coefficient_x_squared (f : ℝ → ℝ) := 
    (1/2) * (deriv (deriv f) 0)
  coefficient_x_squared expansion = -Nat.choose 8 3 * 2^3 := by
  sorry

end coefficient_x_squared_in_expansion_l3021_302183


namespace functional_equation_solution_l3021_302165

open Function Real

theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, x + y + z = 0 → f (x^3) + f y^3 + f z^3 = 3*x*y*z) : 
  ∀ x : ℝ, f x = x := by
sorry

end functional_equation_solution_l3021_302165


namespace termite_ridden_not_collapsing_l3021_302132

/-- The fraction of homes on Gotham Street that are termite-ridden -/
def termite_ridden_fraction : ℚ := 1/3

/-- The fraction of termite-ridden homes that are collapsing -/
def collapsing_fraction : ℚ := 7/10

/-- Theorem: The fraction of homes that are termite-ridden but not collapsing is 1/10 -/
theorem termite_ridden_not_collapsing : 
  termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction) = 1/10 := by
  sorry

end termite_ridden_not_collapsing_l3021_302132


namespace laundry_cost_theorem_l3021_302194

/-- Represents the cost per load of laundry in EUR cents -/
def cost_per_load (loads_per_bottle : ℕ) (regular_price : ℚ) (sale_price : ℚ) 
  (tax_rate : ℚ) (coupon : ℚ) (conversion_rate : ℚ) : ℚ :=
  let total_loads := 2 * loads_per_bottle
  let pre_tax_cost := 2 * sale_price - coupon
  let total_cost := pre_tax_cost * (1 + tax_rate)
  let cost_in_eur := total_cost * conversion_rate
  (cost_in_eur * 100) / total_loads

theorem laundry_cost_theorem (loads_per_bottle : ℕ) (regular_price : ℚ) 
  (sale_price : ℚ) (tax_rate : ℚ) (coupon : ℚ) (conversion_rate : ℚ) :
  loads_per_bottle = 80 →
  regular_price = 25 →
  sale_price = 20 →
  tax_rate = 0.05 →
  coupon = 5 →
  conversion_rate = 0.85 →
  ∃ (n : ℕ), n ≤ 20 ∧ 20 < n + 1 ∧ 
    cost_per_load loads_per_bottle regular_price sale_price tax_rate coupon conversion_rate < n + 1 ∧
    n < cost_per_load loads_per_bottle regular_price sale_price tax_rate coupon conversion_rate := by
  sorry

end laundry_cost_theorem_l3021_302194


namespace imaginary_part_of_one_over_one_plus_i_l3021_302166

theorem imaginary_part_of_one_over_one_plus_i :
  let z : ℂ := 1 / (1 + Complex.I)
  Complex.im z = -1/2 := by sorry

end imaginary_part_of_one_over_one_plus_i_l3021_302166


namespace continuous_piecewise_function_l3021_302108

-- Define the piecewise function g(x)
noncomputable def g (x a b : ℝ) : ℝ :=
  if x > 1 then b * x + 1
  else if -3 ≤ x ∧ x ≤ 1 then 2 * x - 4
  else 3 * x - a

-- State the theorem
theorem continuous_piecewise_function (a b : ℝ) :
  Continuous g → a + b = -2 :=
by sorry

end continuous_piecewise_function_l3021_302108


namespace limit_at_one_l3021_302130

def f (x : ℝ) : ℝ := 2 * x^3

theorem limit_at_one (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f (1 + Δx) - f 1) / Δx) - 6| < ε := by
  sorry

end limit_at_one_l3021_302130


namespace student_score_problem_l3021_302123

theorem student_score_problem (total_questions : ℕ) (score : ℤ) 
  (h1 : total_questions = 100)
  (h2 : score = 73) :
  ∃ (correct incorrect : ℕ),
    correct + incorrect = total_questions ∧
    (correct : ℤ) - 2 * (incorrect : ℤ) = score ∧
    correct = 91 := by
  sorry

end student_score_problem_l3021_302123


namespace final_ratio_is_two_to_one_l3021_302114

/-- Represents the ratio of milk to water in a mixture -/
structure Ratio where
  milk : ℕ
  water : ℕ

/-- Represents a can containing a mixture of milk and water -/
structure Can where
  capacity : ℕ
  current_volume : ℕ
  mixture : Ratio

def add_milk (can : Can) (amount : ℕ) : Can :=
  { can with
    current_volume := can.current_volume + amount
    mixture := Ratio.mk (can.mixture.milk + amount) can.mixture.water
  }

theorem final_ratio_is_two_to_one
  (initial_can : Can)
  (h1 : initial_can.mixture = Ratio.mk 4 3)
  (h2 : initial_can.capacity = 36)
  (h3 : (add_milk initial_can 8).current_volume = initial_can.capacity) :
  (add_milk initial_can 8).mixture = Ratio.mk 2 1 := by
  sorry

end final_ratio_is_two_to_one_l3021_302114


namespace circle_on_y_axis_through_point_one_two_l3021_302135

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_on_y_axis_through_point_one_two :
  ∃ (c : Circle),
    c.center.1 = 0 ∧
    c.radius = 1 ∧
    circle_equation c 1 2 ∧
    ∀ (x y : ℝ), circle_equation c x y ↔ x^2 + (y - 2)^2 = 1 :=
sorry

end circle_on_y_axis_through_point_one_two_l3021_302135


namespace check_cashing_error_l3021_302120

theorem check_cashing_error (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  x > y →
  100 * y + x - (100 * x + y) = 2187 →
  x - y = 22 := by
sorry

end check_cashing_error_l3021_302120


namespace ambika_candles_l3021_302177

theorem ambika_candles (ambika : ℕ) (aniyah : ℕ) : 
  aniyah = 6 * ambika →
  (ambika + aniyah) / 2 = 14 →
  ambika = 4 := by
sorry

end ambika_candles_l3021_302177


namespace five_divides_cube_iff_five_divides_l3021_302193

theorem five_divides_cube_iff_five_divides (a : ℤ) : 
  (5 : ℤ) ∣ a^3 ↔ (5 : ℤ) ∣ a := by
  sorry

end five_divides_cube_iff_five_divides_l3021_302193


namespace divisibility_by_three_l3021_302126

/-- A sequence of integers satisfying the recurrence relation -/
def SatisfiesRecurrence (a : ℕ → ℤ) (k : ℕ+) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n * n = a (n - 1) + n^(k : ℕ)

/-- The main theorem -/
theorem divisibility_by_three (k : ℕ+) (a : ℕ → ℤ) 
  (h : SatisfiesRecurrence a k) : 
  3 ∣ (k : ℤ) - 2 := by
  sorry

end divisibility_by_three_l3021_302126


namespace hall_of_mirrors_glass_area_l3021_302196

/-- Calculates the total area of glass needed for James' hall of mirrors --/
theorem hall_of_mirrors_glass_area :
  let wall1_length : ℝ := 30
  let wall1_width : ℝ := 12
  let wall2_length : ℝ := 30
  let wall2_width : ℝ := 12
  let wall3_length : ℝ := 20
  let wall3_width : ℝ := 12
  let wall1_area := wall1_length * wall1_width
  let wall2_area := wall2_length * wall2_width
  let wall3_area := wall3_length * wall3_width
  let total_area := wall1_area + wall2_area + wall3_area
  total_area = 960 := by sorry

end hall_of_mirrors_glass_area_l3021_302196


namespace money_distribution_l3021_302142

/-- The problem of distributing money among boys -/
theorem money_distribution (total_amount : ℕ) (extra_per_boy : ℕ) : 
  (total_amount = 5040) →
  (extra_per_boy = 80) →
  ∃ (x : ℕ), 
    (x * (total_amount / 18 + extra_per_boy) = total_amount) ∧
    (x = 14) := by
  sorry

end money_distribution_l3021_302142


namespace train_length_calculation_l3021_302100

/-- Calculates the length of a train given its speed and time to cross a pole. -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 240 → time_s = 21 → 
  ∃ (length_m : ℝ), abs (length_m - 1400.07) < 0.01 ∧ length_m = speed_kmh * (1000 / 3600) * time_s := by
  sorry

end train_length_calculation_l3021_302100


namespace line_length_after_erasing_l3021_302190

/-- Proves that erasing 33 cm from a 1 m line results in a 67 cm line -/
theorem line_length_after_erasing :
  ∀ (initial_length : ℝ) (erased_length : ℝ),
  initial_length = 1 →
  erased_length = 33 / 100 →
  (initial_length - erased_length) * 100 = 67 := by
sorry

end line_length_after_erasing_l3021_302190


namespace root_sum_theorem_l3021_302127

theorem root_sum_theorem (x : ℝ) : 
  (1/x + 1/(x + 4) - 1/(x + 8) - 1/(x + 12) - 1/(x + 16) - 1/(x + 20) + 1/(x + 24) + 1/(x + 28) = 0) →
  (∃ (a b c d : ℕ), 
    (x = -a + Real.sqrt (b + c * Real.sqrt d) ∨ x = -a - Real.sqrt (b + c * Real.sqrt d) ∨
     x = -a + Real.sqrt (b - c * Real.sqrt d) ∨ x = -a - Real.sqrt (b - c * Real.sqrt d)) ∧
    a + b + c + d = 123) := by
  sorry

end root_sum_theorem_l3021_302127


namespace integral_3x_plus_sin_x_l3021_302150

theorem integral_3x_plus_sin_x (x : Real) : 
  ∫ x in (0)..(π/2), (3*x + Real.sin x) = 3*π^2/8 + 1 := by
  sorry

end integral_3x_plus_sin_x_l3021_302150


namespace a_4_value_l3021_302113

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_4_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2 = 2 ∨ a 2 = 32) →
  (a 6 = 2 ∨ a 6 = 32) →
  a 2 * a 6 = 64 →
  a 2 + a 6 = 34 →
  a 4 = 8 := by
  sorry

end a_4_value_l3021_302113


namespace circle_and_line_intersection_l3021_302125

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem circle_and_line_intersection :
  ∀ m : ℝ,
  (∀ x y : ℝ, circle_equation x y m → x^2 + y^2 = (x - 1)^2 + (y - 2)^2 + (5 - m)) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ m ∧ 
    circle_equation x₂ y₂ m ∧ 
    line_equation x₁ y₁ ∧ 
    line_equation x₂ y₂ ∧ 
    perpendicular x₁ y₁ x₂ y₂) →
  (m < 5 ∧ m = 8/5 ∧ 
   ∀ x y : ℝ, x^2 + y^2 - (8/5)*x - (16/5)*y = 0 ↔ 
   ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
   x = (1-t)*x₁ + t*x₂ ∧ 
   y = (1-t)*y₁ + t*y₂) :=
sorry

end circle_and_line_intersection_l3021_302125


namespace anya_initial_seat_l3021_302199

def Friend := Fin 5

structure SeatingArrangement where
  seats : Friend → Fin 5
  bijective : Function.Bijective seats

def move_right (n : Nat) (s : Fin 5) : Fin 5 :=
  ⟨(s.val + n) % 5, by sorry⟩

def move_left (n : Nat) (s : Fin 5) : Fin 5 :=
  ⟨(s.val + 5 - n % 5) % 5, by sorry⟩

def swap (s1 s2 : Fin 5) (s : Fin 5) : Fin 5 :=
  if s = s1 then s2
  else if s = s2 then s1
  else s

theorem anya_initial_seat (initial final : SeatingArrangement) 
  (anya varya galya diana ellya : Friend) :
  initial.seats anya ≠ 1 →
  initial.seats anya ≠ 5 →
  final.seats anya = 1 ∨ final.seats anya = 5 →
  final.seats varya = move_right 1 (initial.seats varya) →
  final.seats galya = move_left 3 (initial.seats galya) →
  final.seats diana = initial.seats ellya →
  final.seats ellya = initial.seats diana →
  initial.seats anya = 3 := by sorry

end anya_initial_seat_l3021_302199


namespace median_and_mode_of_scores_l3021_302122

def student_scores : List Nat := [3, 6, 4, 6, 4, 3, 6, 5, 7]

def median (l : List Nat) : Nat := sorry

def mode (l : List Nat) : Nat := sorry

theorem median_and_mode_of_scores : 
  median student_scores = 5 ∧ mode student_scores = 6 := by sorry

end median_and_mode_of_scores_l3021_302122


namespace part_one_part_two_l3021_302197

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | x^2 - 3*x ≤ 10}

-- Part 1
theorem part_one : (Set.compl (P 3) ∩ Q) = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two : ∀ a : ℝ, (P a ∪ Q = Q) ↔ a ∈ Set.Iic 2 := by sorry

end part_one_part_two_l3021_302197


namespace unique_grid_placement_l3021_302168

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two positions are adjacent in the grid --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- The sum of adjacent numbers is less than 12 --/
def valid_sum (g : Grid) : Prop :=
  ∀ p1 p2 : Fin 3 × Fin 3, adjacent p1 p2 → g p1.1 p1.2 + g p2.1 p2.2 < 12

/-- The grid contains all numbers from 1 to 9 --/
def contains_all_numbers (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n.val + 1

/-- The given positions of odd numbers --/
def given_positions (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7 ∧ g 0 2 = 9

/-- The theorem to be proved --/
theorem unique_grid_placement :
  ∀ g : Grid,
    valid_sum g →
    contains_all_numbers g →
    given_positions g →
    g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by
  sorry

end unique_grid_placement_l3021_302168


namespace all_points_on_line_l3021_302181

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The set of n points on the plane -/
def PointSet (n : ℕ) := Fin n → Point

/-- The property that any line through two points contains at least one more point -/
def ThreePointProperty (points : PointSet n) : Prop :=
  ∀ (i j k : Fin n), i ≠ j →
    ∃ (l : Line), pointOnLine (points i) l ∧ pointOnLine (points j) l →
      ∃ (m : Fin n), m ≠ i ∧ m ≠ j ∧ pointOnLine (points m) l

/-- The theorem statement -/
theorem all_points_on_line (n : ℕ) (points : PointSet n) 
  (h : ThreePointProperty points) : 
  ∃ (l : Line), ∀ (i : Fin n), pointOnLine (points i) l :=
sorry

end all_points_on_line_l3021_302181
