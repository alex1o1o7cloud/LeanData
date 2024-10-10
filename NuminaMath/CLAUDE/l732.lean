import Mathlib

namespace max_tickets_proof_l732_73225

/-- Represents the maximum number of tickets Jane can buy given the following conditions:
  * Each ticket costs $15
  * Jane has a budget of $180
  * If more than 10 tickets are bought, there's a discount of $2 per ticket
-/
def max_tickets : ℕ := 13

/-- The cost of a ticket without discount -/
def ticket_cost : ℕ := 15

/-- Jane's budget -/
def budget : ℕ := 180

/-- The discount per ticket when buying more than 10 tickets -/
def discount : ℕ := 2

/-- The threshold for applying the discount -/
def discount_threshold : ℕ := 10

theorem max_tickets_proof :
  (∀ n : ℕ, n ≤ discount_threshold → n * ticket_cost ≤ budget) ∧
  (∀ n : ℕ, n > discount_threshold → n * (ticket_cost - discount) ≤ budget) ∧
  (∀ n : ℕ, n > max_tickets → 
    (if n ≤ discount_threshold then n * ticket_cost > budget
     else n * (ticket_cost - discount) > budget)) :=
sorry

end max_tickets_proof_l732_73225


namespace quadratic_inequality_range_l732_73262

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a-1)*x + 1 < 0) → (a > 3 ∨ a < -1) := by
  sorry

end quadratic_inequality_range_l732_73262


namespace apple_packing_difference_is_500_l732_73251

/-- Represents the apple packing scenario over two weeks -/
structure ApplePacking where
  apples_per_box : ℕ
  boxes_per_day : ℕ
  days_per_week : ℕ
  total_apples_two_weeks : ℕ

/-- Calculates the difference in daily apple packing between the first and second week -/
def daily_packing_difference (ap : ApplePacking) : ℕ :=
  let normal_daily_packing := ap.apples_per_box * ap.boxes_per_day
  let first_week_total := normal_daily_packing * ap.days_per_week
  let second_week_total := ap.total_apples_two_weeks - first_week_total
  let second_week_daily_average := second_week_total / ap.days_per_week
  normal_daily_packing - second_week_daily_average

/-- Theorem stating the difference in daily apple packing is 500 -/
theorem apple_packing_difference_is_500 :
  ∀ (ap : ApplePacking),
    ap.apples_per_box = 40 ∧
    ap.boxes_per_day = 50 ∧
    ap.days_per_week = 7 ∧
    ap.total_apples_two_weeks = 24500 →
    daily_packing_difference ap = 500 := by
  sorry

end apple_packing_difference_is_500_l732_73251


namespace rational_roots_quadratic_l732_73250

theorem rational_roots_quadratic (r : ℚ) : 
  (∃ (n : ℤ), 2 * r = n) →
  (∃ (x : ℚ), (r^2 + r) * x^2 + 4 - r^2 = 0) →
  r = 2 ∨ r = -2 ∨ r = -4 := by
sorry

end rational_roots_quadratic_l732_73250


namespace stratified_sampling_result_l732_73284

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  elderly_population : ℕ
  young_population : ℕ
  young_sample : ℕ
  (elderly_population_le_total : elderly_population ≤ total_population)
  (young_population_le_total : young_population ≤ total_population)
  (young_sample_le_young_population : young_sample ≤ young_population)

/-- Calculates the number of elderly in the sample based on stratified sampling -/
def elderly_in_sample (s : StratifiedSample) : ℚ :=
  s.elderly_population * (s.young_sample : ℚ) / s.young_population

/-- Theorem stating the result of the stratified sampling problem -/
theorem stratified_sampling_result (s : StratifiedSample) 
  (h_total : s.total_population = 430)
  (h_elderly : s.elderly_population = 90)
  (h_young : s.young_population = 160)
  (h_young_sample : s.young_sample = 32) :
  elderly_in_sample s = 18 := by
  sorry

end stratified_sampling_result_l732_73284


namespace correct_popularity_order_l732_73264

/-- Represents the activities available for the sports day --/
inductive Activity
| dodgeball
| chessTournament
| track
| swimming

/-- Returns the fraction of students preferring a given activity --/
def preference (a : Activity) : Rat :=
  match a with
  | Activity.dodgeball => 3/8
  | Activity.chessTournament => 9/24
  | Activity.track => 5/16
  | Activity.swimming => 1/3

/-- Compares two activities based on their popularity --/
def morePopularThan (a b : Activity) : Prop :=
  preference a > preference b

/-- States that the given order of activities is correct based on popularity --/
theorem correct_popularity_order :
  morePopularThan Activity.swimming Activity.dodgeball ∧
  morePopularThan Activity.dodgeball Activity.chessTournament ∧
  morePopularThan Activity.chessTournament Activity.track :=
by sorry

end correct_popularity_order_l732_73264


namespace milburg_adults_l732_73292

theorem milburg_adults (total_population children : ℕ) 
  (h1 : total_population = 5256)
  (h2 : children = 2987) :
  total_population - children = 2269 := by
  sorry

end milburg_adults_l732_73292


namespace complement_intersection_theorem_l732_73239

open Set

theorem complement_intersection_theorem (M N : Set ℝ) :
  M = {x | x > 1} →
  N = {x | |x| ≤ 2} →
  (𝓤 \ M) ∩ N = Icc (-2) 1 := by
  sorry

end complement_intersection_theorem_l732_73239


namespace remainder_sum_l732_73277

theorem remainder_sum (n : ℤ) (h : n % 12 = 5) : (n % 3 + n % 4 = 3) := by
  sorry

end remainder_sum_l732_73277


namespace max_value_p_l732_73266

theorem max_value_p (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b * c + a + c = b) : 
  let p := 2 / (1 + a^2) - 2 / (1 + b^2) + 3 / (1 + c^2)
  ∃ (max_p : ℝ), max_p = 10/3 ∧ p ≤ max_p := by
  sorry

end max_value_p_l732_73266


namespace science_problem_time_l732_73296

/-- Calculates the time taken for each science problem given the number of problems and time constraints. -/
theorem science_problem_time 
  (math_problems : ℕ) 
  (social_studies_problems : ℕ) 
  (science_problems : ℕ) 
  (math_time_per_problem : ℚ) 
  (social_studies_time_per_problem : ℚ) 
  (total_time : ℚ) 
  (h1 : math_problems = 15)
  (h2 : social_studies_problems = 6)
  (h3 : science_problems = 10)
  (h4 : math_time_per_problem = 2)
  (h5 : social_studies_time_per_problem = 1/2)
  (h6 : total_time = 48) :
  (total_time - (math_problems * math_time_per_problem + social_studies_problems * social_studies_time_per_problem)) / science_problems = 3/2 := by
  sorry

end science_problem_time_l732_73296


namespace root_in_smaller_interval_l732_73289

-- Define the function
def f (x : ℝ) := x^3 - 6*x^2 + 4

-- State the theorem
theorem root_in_smaller_interval :
  (∃ x ∈ Set.Ioo 0 1, f x = 0) →
  (∃ x ∈ Set.Ioo (1/2) 1, f x = 0) :=
by sorry

end root_in_smaller_interval_l732_73289


namespace test_probabilities_l732_73212

theorem test_probabilities (p_first : ℝ) (p_second : ℝ) (p_both : ℝ) 
  (h1 : p_first = 0.7)
  (h2 : p_second = 0.55)
  (h3 : p_both = 0.45) :
  1 - (p_first + p_second - p_both) = 0.2 :=
by sorry

end test_probabilities_l732_73212


namespace expenditure_ratio_proof_l732_73242

/-- Given the income ratio and savings of Uma and Bala, prove their expenditure ratio -/
theorem expenditure_ratio_proof (uma_income bala_income uma_expenditure bala_expenditure : ℚ) 
  (h1 : uma_income = (8 : ℚ) / 7 * bala_income)
  (h2 : uma_income = 16000)
  (h3 : uma_income - uma_expenditure = 2000)
  (h4 : bala_income - bala_expenditure = 2000) :
  uma_expenditure / bala_expenditure = 7 / 6 := by
  sorry

end expenditure_ratio_proof_l732_73242


namespace fraction_equality_l732_73243

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) :
  4 / 7 + (12/5) / (2 * q + p) = 1 := by sorry

end fraction_equality_l732_73243


namespace triangle_properties_l732_73256

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (A > 0 ∧ B > 0 ∧ C > 0) →
  (A + B + C = π) →
  (c * Real.sin B = b * Real.cos (C - π/6) ∨
   Real.cos B = (2*a - b) / (2*c) ∨
   (a^2 + b^2 - c^2) * Real.tan C = Real.sqrt 3 * a * b) →
  (Real.sin C = Real.sqrt 3 / 2 ∧
   (c = Real.sqrt 3 ∧ a = 3*b → 
    1/2 * a * b * Real.sin C = 9 * Real.sqrt 3 / 28)) := by
  sorry

#check triangle_properties

end triangle_properties_l732_73256


namespace farm_animal_count_l732_73220

/-- Represents the count of animals on a farm -/
structure FarmCount where
  chickens : ℕ
  ducks : ℕ
  geese : ℕ
  quails : ℕ
  turkeys : ℕ
  cow_sheds : ℕ
  cows_per_shed : ℕ
  pigs : ℕ

/-- Calculates the total number of animals on the farm -/
def total_animals (farm : FarmCount) : ℕ :=
  farm.chickens + farm.ducks + farm.geese + farm.quails + farm.turkeys +
  (farm.cow_sheds * farm.cows_per_shed) + farm.pigs

/-- Theorem stating that the total number of animals on the given farm is 219 -/
theorem farm_animal_count :
  let farm := FarmCount.mk 60 40 20 50 10 3 8 15
  total_animals farm = 219 := by
  sorry

#eval total_animals (FarmCount.mk 60 40 20 50 10 3 8 15)

end farm_animal_count_l732_73220


namespace mary_money_left_l732_73244

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let drink_cost := p
  let medium_pizza_cost := 2 * p
  let large_pizza_cost := 3 * p
  let total_cost := 3 * drink_cost + medium_pizza_cost + large_pizza_cost
  30 - total_cost

/-- Theorem stating that the amount of money Mary has left is 30 - 8p -/
theorem mary_money_left (p : ℝ) : money_left p = 30 - 8 * p := by
  sorry

end mary_money_left_l732_73244


namespace francie_remaining_money_l732_73261

/-- Calculates the remaining money after Francie's savings and purchases -/
def remaining_money (initial_allowance : ℕ) (initial_weeks : ℕ) 
  (raised_allowance : ℕ) (raised_weeks : ℕ) (video_game_cost : ℕ) : ℕ :=
  let total_savings := initial_allowance * initial_weeks + raised_allowance * raised_weeks
  let after_clothes := total_savings / 2
  after_clothes - video_game_cost

/-- Theorem stating that Francie's remaining money is $3 -/
theorem francie_remaining_money :
  remaining_money 5 8 6 6 35 = 3 := by
  sorry

#eval remaining_money 5 8 6 6 35

end francie_remaining_money_l732_73261


namespace sector_angle_l732_73269

/-- 
Given a circular sector where:
- r is the radius of the sector
- α is the central angle of the sector in radians
- l is the arc length of the sector
- C is the circumference of the sector

Prove that if C = 4r, then α = 2.
-/
theorem sector_angle (r : ℝ) (α : ℝ) (l : ℝ) (C : ℝ) 
  (h1 : C = 4 * r)  -- Circumference is four times the radius
  (h2 : C = 2 * r + l)  -- Circumference formula for a sector
  (h3 : l = α * r)  -- Arc length formula
  : α = 2 := by
  sorry

end sector_angle_l732_73269


namespace parallelogram_angles_l732_73252

/-- A parallelogram with angles measured in degrees -/
structure Parallelogram where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_360 : A + B + C + D = 360
  opposite_equal_AC : A = C
  opposite_equal_BD : B = D

/-- Theorem: In a parallelogram ABCD where angle A measures 125°, 
    the measures of angles B, C, and D are 55°, 125°, and 55° respectively. -/
theorem parallelogram_angles (p : Parallelogram) (h : p.A = 125) : 
  p.B = 55 ∧ p.C = 125 ∧ p.D = 55 := by
  sorry


end parallelogram_angles_l732_73252


namespace bike_five_times_a_week_l732_73200

/-- Given Onur's daily biking distance, Hanil's additional distance, and their total weekly distance,
    calculate the number of days they bike per week. -/
def biking_days_per_week (onur_daily : ℕ) (hanil_additional : ℕ) (total_weekly : ℕ) : ℕ :=
  total_weekly / (onur_daily + (onur_daily + hanil_additional))

/-- Theorem stating that under the given conditions, Onur and Hanil bike 5 times a week. -/
theorem bike_five_times_a_week :
  biking_days_per_week 250 40 2700 = 5 := by
  sorry

end bike_five_times_a_week_l732_73200


namespace quadratic_equation_solution_l732_73204

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = -2 + Real.sqrt 2 ∧ x₂ = -2 - Real.sqrt 2) ∧ 
  (∀ x : ℝ, x^2 + 4*x + 2 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end quadratic_equation_solution_l732_73204


namespace smallest_dual_base_palindrome_l732_73201

/-- A function that checks if a natural number is a palindrome in a given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- A function that converts a natural number from one base to another. -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- A function that returns the number of digits of a natural number in a given base. -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∀ n : ℕ,
  (isPalindrome n 2 ∧ numDigits n 2 = 5) →
  (∃ b : ℕ, b > 2 ∧ isPalindrome (baseConvert n 2 b) b ∧ numDigits (baseConvert n 2 b) b = 3) →
  n ≥ 17 := by
  sorry

end smallest_dual_base_palindrome_l732_73201


namespace completing_square_solution_l732_73241

theorem completing_square_solution (x : ℝ) :
  (x^2 - 4*x + 3 = 0) ↔ ((x - 2)^2 = 1) :=
by sorry

end completing_square_solution_l732_73241


namespace nina_running_distance_l732_73240

theorem nina_running_distance :
  let d1 : ℚ := 0.08333333333333333
  let d2 : ℚ := 0.08333333333333333
  let d3 : ℚ := 0.6666666666666666
  d1 + d2 + d3 = 0.8333333333333333 := by sorry

end nina_running_distance_l732_73240


namespace incorrect_number_calculation_l732_73291

theorem incorrect_number_calculation (n : ℕ) (correct_num incorrect_num : ℝ) 
  (incorrect_avg correct_avg : ℝ) :
  n = 10 ∧ 
  correct_num = 75 ∧
  n * incorrect_avg = n * correct_avg - (correct_num - incorrect_num) →
  incorrect_num = 25 :=
by sorry

end incorrect_number_calculation_l732_73291


namespace power_division_equality_l732_73267

theorem power_division_equality : 8^15 / 64^6 = 512 :=
by
  sorry

end power_division_equality_l732_73267


namespace bernardo_always_wins_l732_73299

def even_set : Finset ℕ := {2, 4, 6, 8, 10}
def odd_set : Finset ℕ := {1, 3, 5, 7, 9}

def form_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem bernardo_always_wins :
  ∀ (a b c : ℕ) (d e f : ℕ),
    a ∈ even_set → b ∈ even_set → c ∈ even_set →
    d ∈ odd_set → e ∈ odd_set → f ∈ odd_set →
    a ≠ b → b ≠ c → a ≠ c →
    d ≠ e → e ≠ f → d ≠ f →
    a > b → b > c →
    d > e → e > f →
    form_number a b c > form_number d e f :=
by sorry

end bernardo_always_wins_l732_73299


namespace pencils_per_row_l732_73287

/-- Given 12 pencils distributed equally among 3 rows, prove that there are 4 pencils in each row. -/
theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : total_pencils = 12) 
  (h2 : num_rows = 3) 
  (h3 : total_pencils = num_rows * pencils_per_row) : 
  pencils_per_row = 4 := by
  sorry

end pencils_per_row_l732_73287


namespace infinite_binary_decimal_divisible_by_2019_l732_73276

/-- A number composed only of 0 and 1 in decimal form -/
def BinaryDecimal (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The set of numbers composed only of 0 and 1 in decimal form that are divisible by 2019 -/
def BinaryDecimalDivisibleBy2019 : Set ℕ :=
  {n : ℕ | BinaryDecimal n ∧ 2019 ∣ n}

/-- The set of numbers composed only of 0 and 1 in decimal form that are divisible by 2019 is infinite -/
theorem infinite_binary_decimal_divisible_by_2019 :
    Set.Infinite BinaryDecimalDivisibleBy2019 :=
  sorry

end infinite_binary_decimal_divisible_by_2019_l732_73276


namespace octal_253_equals_171_l732_73208

/-- Converts an octal digit to its decimal representation -/
def octal_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- The octal representation of the number -/
def octal_number : List Nat := [2, 5, 3]

/-- Converts an octal number to its decimal representation -/
def octal_to_decimal_conversion (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + octal_to_decimal digit * (8 ^ i)) 0

theorem octal_253_equals_171 :
  octal_to_decimal_conversion octal_number = 171 := by
  sorry

end octal_253_equals_171_l732_73208


namespace quadratic_form_nonnegative_l732_73258

theorem quadratic_form_nonnegative (a b c : ℝ) :
  (∀ (f g : ℝ × ℝ), a * (f.1^2 + f.2^2) + b * (f.1 * g.1 + f.2 * g.2) + c * (g.1^2 + g.2^2) ≥ 0) ↔
  (a ≥ 0 ∧ c ≥ 0 ∧ 4 * a * c ≥ b^2) :=
by sorry

end quadratic_form_nonnegative_l732_73258


namespace equation_has_real_roots_l732_73221

theorem equation_has_real_roots (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  ∃ x : ℝ, x ≠ 1 ∧ a^2 / x + b^2 / (x - 1) = 1 :=
by sorry

end equation_has_real_roots_l732_73221


namespace largest_integral_x_l732_73222

theorem largest_integral_x : ∃ (x : ℤ), 
  (1/4 : ℚ) < (x : ℚ)/6 ∧ (x : ℚ)/6 < 2/3 ∧ 
  x < 10 ∧
  x = 3 ∧
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 2/3 ∧ y < 10) → y ≤ x :=
by sorry

end largest_integral_x_l732_73222


namespace wage_difference_l732_73215

/-- Proves the wage difference between a manager and a chef given specific wage relationships -/
theorem wage_difference (manager_wage : ℝ) 
  (h1 : manager_wage = 6.50)
  (h2 : ∃ dishwasher_wage : ℝ, dishwasher_wage = manager_wage / 2)
  (h3 : ∃ chef_wage : ℝ, chef_wage = (manager_wage / 2) * 1.20) :
  manager_wage - ((manager_wage / 2) * 1.20) = 2.60 := by
  sorry

end wage_difference_l732_73215


namespace men_per_table_l732_73285

theorem men_per_table (num_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) :
  num_tables = 9 →
  women_per_table = 7 →
  total_customers = 90 →
  ∃ (men_per_table : ℕ), 
    men_per_table * num_tables + women_per_table * num_tables = total_customers ∧
    men_per_table = 3 :=
by sorry

end men_per_table_l732_73285


namespace students_liking_both_soda_and_coke_l732_73231

/-- Given a school with the following conditions:
  - Total number of students: 500
  - Students who like soda: 337
  - Students who like coke: 289
  - Students who neither like soda nor coke: 56
  Prove that the number of students who like both soda and coke is 182. -/
theorem students_liking_both_soda_and_coke 
  (total : ℕ) (soda : ℕ) (coke : ℕ) (neither : ℕ) 
  (h_total : total = 500)
  (h_soda : soda = 337)
  (h_coke : coke = 289)
  (h_neither : neither = 56) :
  soda + coke - total + neither = 182 := by
  sorry

end students_liking_both_soda_and_coke_l732_73231


namespace simplify_expression_l732_73246

theorem simplify_expression : 18 * (8 / 16) * (3 / 27) = 1 := by
  sorry

end simplify_expression_l732_73246


namespace eraser_price_correct_l732_73297

/-- The price of an eraser given the following conditions:
  1. 3 erasers and 5 pencils cost 10.6 yuan
  2. 4 erasers and 4 pencils cost 12 yuan -/
def eraser_price : ℝ := 2.2

/-- The price of a pencil (to be determined) -/
def pencil_price : ℝ := sorry

theorem eraser_price_correct :
  3 * eraser_price + 5 * pencil_price = 10.6 ∧
  4 * eraser_price + 4 * pencil_price = 12 :=
by sorry

end eraser_price_correct_l732_73297


namespace operation_2012_equals_55_l732_73209

def operation_sequence (n : ℕ) : ℕ :=
  match n % 3 with
  | 1 => 133
  | 2 => 55
  | 0 => 250
  | _ => 0  -- This case is unreachable, but needed for completeness

theorem operation_2012_equals_55 : operation_sequence 2012 = 55 := by
  sorry

end operation_2012_equals_55_l732_73209


namespace quadrilateral_area_l732_73213

/-- The area of the quadrilateral formed by three coplanar squares -/
theorem quadrilateral_area (s₁ s₂ s₃ : ℝ) (hs₁ : s₁ = 3) (hs₂ : s₂ = 5) (hs₃ : s₃ = 7) : 
  let h₁ := s₁ * (s₃ / (s₁ + s₂ + s₃))
  let h₂ := (s₁ + s₂) * (s₃ / (s₁ + s₂ + s₃))
  (h₁ + h₂) * s₂ / 2 = 12.825 := by
sorry

end quadrilateral_area_l732_73213


namespace amount_fraction_is_one_third_l732_73210

/-- Represents the amounts received by A, B, and C in dollars -/
structure Amounts where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (x : Amounts) (total : ℝ) (fraction : ℝ) : Prop :=
  x.a + x.b + x.c = total ∧
  x.a = fraction * (x.b + x.c) ∧
  x.b = (2 / 7) * (x.a + x.c) ∧
  x.a = x.b + 10

theorem amount_fraction_is_one_third :
  ∃ (x : Amounts) (fraction : ℝ),
    satisfies_conditions x 360 fraction ∧ fraction = 1 / 3 := by sorry

end amount_fraction_is_one_third_l732_73210


namespace puppies_weight_difference_l732_73268

/-- The weight difference between two dogs after a year, given their initial weights and weight gain percentage -/
def weight_difference (labrador_initial : ℝ) (dachshund_initial : ℝ) (weight_gain_percentage : ℝ) : ℝ :=
  (labrador_initial * (1 + weight_gain_percentage)) - (dachshund_initial * (1 + weight_gain_percentage))

/-- Theorem stating that the weight difference between the labrador and dachshund puppies after a year is 35 pounds -/
theorem puppies_weight_difference :
  weight_difference 40 12 0.25 = 35 := by
  sorry

end puppies_weight_difference_l732_73268


namespace tax_assignment_correct_l732_73217

/-- Represents the different types of budget levels in the Russian tax system -/
inductive BudgetLevel
  | Federal
  | Regional

/-- Represents the different types of taxes in the Russian tax system -/
inductive TaxType
  | PropertyTax
  | FederalTax
  | ProfitTax
  | RegionalTax
  | TransportFee

/-- Assigns a tax to a budget level -/
def assignTax (tax : TaxType) : BudgetLevel :=
  match tax with
  | TaxType.PropertyTax => BudgetLevel.Regional
  | TaxType.FederalTax => BudgetLevel.Federal
  | TaxType.ProfitTax => BudgetLevel.Regional
  | TaxType.RegionalTax => BudgetLevel.Regional
  | TaxType.TransportFee => BudgetLevel.Regional

/-- Theorem stating the correct assignment of taxes to budget levels -/
theorem tax_assignment_correct :
  (assignTax TaxType.PropertyTax = BudgetLevel.Regional) ∧
  (assignTax TaxType.FederalTax = BudgetLevel.Federal) ∧
  (assignTax TaxType.ProfitTax = BudgetLevel.Regional) ∧
  (assignTax TaxType.RegionalTax = BudgetLevel.Regional) ∧
  (assignTax TaxType.TransportFee = BudgetLevel.Regional) :=
by sorry

end tax_assignment_correct_l732_73217


namespace rectangle_to_rhombus_l732_73294

-- Define a rectangle
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)
  (width_pos : width > 0)
  (height_pos : height > 0)

-- Define a rhombus
structure Rhombus :=
  (side : ℝ)
  (side_pos : side > 0)

-- Define the theorem
theorem rectangle_to_rhombus (r : Rectangle) : 
  ∃ (rh : Rhombus), r.width * r.height = 4 * rh.side^2 := by
  sorry

end rectangle_to_rhombus_l732_73294


namespace diophantine_equation_solution_l732_73281

theorem diophantine_equation_solution : 
  {(x, y) : ℕ+ × ℕ+ | x.val - y.val - (x.val / y.val) - (x.val^3 / y.val^3) + (x.val^4 / y.val^4) = 2017} = 
  {(⟨2949, by norm_num⟩, ⟨983, by norm_num⟩), (⟨4022, by norm_num⟩, ⟨2011, by norm_num⟩)} :=
by sorry

end diophantine_equation_solution_l732_73281


namespace circle_area_ratio_l732_73272

theorem circle_area_ratio : 
  let d1 : ℝ := 2  -- diameter of smallest circle
  let d2 : ℝ := 6  -- diameter of middle circle
  let d3 : ℝ := 10 -- diameter of largest circle
  let r1 : ℝ := d1 / 2  -- radius of smallest circle
  let r2 : ℝ := d2 / 2  -- radius of middle circle
  let r3 : ℝ := d3 / 2  -- radius of largest circle
  let area_smallest : ℝ := π * r1^2
  let area_middle : ℝ := π * r2^2
  let area_largest : ℝ := π * r3^2
  let area_green : ℝ := area_largest - area_middle
  let area_red : ℝ := area_smallest
  (area_green / area_red : ℝ) = 16
  := by sorry


end circle_area_ratio_l732_73272


namespace probability_three_fourths_radius_l732_73238

/-- A circle concentric with and outside a square --/
structure ConcentricCircleSquare where
  squareSideLength : ℝ
  circleRadius : ℝ
  squareSideLength_pos : 0 < squareSideLength
  circleRadius_gt_squareSideLength : squareSideLength < circleRadius

/-- The probability of seeing two sides of the square from a random point on the circle --/
def probabilityTwoSides (c : ConcentricCircleSquare) : ℝ := sorry

theorem probability_three_fourths_radius (c : ConcentricCircleSquare) 
  (h : c.squareSideLength = 4) 
  (prob : probabilityTwoSides c = 3/4) : 
  c.circleRadius = 8 * Real.sqrt 3 / 3 := by sorry

end probability_three_fourths_radius_l732_73238


namespace expression_evaluation_l732_73249

theorem expression_evaluation :
  let x : ℝ := 16
  let expr := (2 + x * (2 + Real.sqrt x) - 4^2) / (Real.sqrt x - 4 + x^2)
  expr = 41 / 128 := by
  sorry

end expression_evaluation_l732_73249


namespace soda_sales_difference_l732_73271

/-- Calculates the difference between evening and morning sales for Remy and Nick's soda business -/
theorem soda_sales_difference (remy_morning : ℕ) (nick_difference : ℕ) (price : ℚ) (evening_sales : ℚ) : 
  remy_morning = 55 →
  nick_difference = 6 →
  price = 1/2 →
  evening_sales = 55 →
  evening_sales - (price * (remy_morning + (remy_morning - nick_difference))) = 3 := by
  sorry

end soda_sales_difference_l732_73271


namespace f_monotonicity_and_zeros_l732_73259

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem f_monotonicity_and_zeros (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ∨
  (a > 0 ∧
    (∀ x y, x < y ∧ y < Real.log a → f a x > f a y) ∧
    (∀ x y, Real.log a < x ∧ x < y → f a x < f a y)) ∧
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ (∀ z, z ≠ x ∧ z ≠ y → f a z ≠ 0) ↔
    (0 < a ∧ a < 1) ∨ (a > 1)) :=
sorry

end f_monotonicity_and_zeros_l732_73259


namespace mathborough_rainfall_2004_l732_73207

theorem mathborough_rainfall_2004 (rainfall_2003 rainfall_2004 : ℕ) 
  (h1 : rainfall_2003 = 45)
  (h2 : rainfall_2004 = rainfall_2003 + 3)
  (h3 : ∃ (high_months low_months : ℕ), 
    high_months = 8 ∧ 
    low_months = 12 - high_months ∧
    (high_months * (rainfall_2004 + 5) + low_months * rainfall_2004 = 616)) : 
  rainfall_2004 * 12 + 8 * 5 = 616 := by
  sorry

end mathborough_rainfall_2004_l732_73207


namespace world_cup_2006_group_stage_matches_l732_73286

/-- The number of matches in a single round-robin tournament with n teams -/
def matches_in_group (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total number of matches in a tournament with groups -/
def total_matches (total_teams : ℕ) (num_groups : ℕ) : ℕ :=
  let teams_per_group := total_teams / num_groups
  num_groups * matches_in_group teams_per_group

theorem world_cup_2006_group_stage_matches :
  total_matches 32 8 = 48 := by
  sorry

end world_cup_2006_group_stage_matches_l732_73286


namespace circle_graph_proportions_l732_73205

theorem circle_graph_proportions :
  ∀ (total : ℝ) (blue : ℝ),
    blue > 0 →
    total = blue + 3 * blue + 0.5 * blue →
    (3 * blue / total = 2 / 3) ∧
    (blue / total = 1 / 4.5) ∧
    (0.5 * blue / total = 1 / 9) := by
  sorry

end circle_graph_proportions_l732_73205


namespace volunteer_selection_schemes_l732_73248

def num_candidates : ℕ := 5
def num_volunteers : ℕ := 4
def num_jobs : ℕ := 4

def driver_only_volunteer : ℕ := 1
def versatile_volunteers : ℕ := num_candidates - driver_only_volunteer

theorem volunteer_selection_schemes :
  (versatile_volunteers.factorial / (versatile_volunteers - (num_jobs - 1)).factorial) +
  (versatile_volunteers.factorial / (versatile_volunteers - num_jobs).factorial) = 48 :=
by sorry

end volunteer_selection_schemes_l732_73248


namespace fraction_zero_implies_x_equals_one_l732_73278

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  (x^2 - 1) / ((x - 2) * (x + 1)) = 0 → x = 1 :=
by sorry

end fraction_zero_implies_x_equals_one_l732_73278


namespace correct_factorization_l732_73224

theorem correct_factorization (x : ℝ) : 4 * x^2 - 4 * x + 1 = (2 * x - 1)^2 := by
  sorry

end correct_factorization_l732_73224


namespace shaded_area_percentage_l732_73255

theorem shaded_area_percentage (total_squares : Nat) (shaded_squares : Nat) :
  total_squares = 36 →
  shaded_squares = 16 →
  (shaded_squares : ℚ) / total_squares * 100 = 44.44 := by
  sorry

end shaded_area_percentage_l732_73255


namespace distance_traveled_downstream_l732_73290

/-- The distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: The distance traveled downstream is 68 km -/
theorem distance_traveled_downstream : 
  distance_downstream 13 4 4 = 68 := by
  sorry

end distance_traveled_downstream_l732_73290


namespace paint_together_l732_73232

/-- The amount of wall Heidi and Tom can paint together in a given time -/
def wall_painted (heidi_time tom_time paint_time : ℚ) : ℚ :=
  paint_time * (1 / heidi_time + 1 / tom_time)

/-- Theorem: Heidi and Tom can paint 5/12 of the wall in 15 minutes -/
theorem paint_together : wall_painted 60 90 15 = 5/12 := by
  sorry

end paint_together_l732_73232


namespace mean_equality_implies_y_equals_10_l732_73203

theorem mean_equality_implies_y_equals_10 : ∀ y : ℝ, 
  (6 + 9 + 18) / 3 = (12 + y) / 2 → y = 10 := by
  sorry

end mean_equality_implies_y_equals_10_l732_73203


namespace g_value_at_4_l732_73265

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the properties of g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧  -- g is a cubic polynomial
  (g 0 = -2) ∧  -- g(0) = -2
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, g s = 0 ∧ s = r^2)  -- roots of g are squares of roots of f

-- Theorem statement
theorem g_value_at_4 (g : ℝ → ℝ) (h : g_properties g) : g 4 = 4 := by
  sorry

end g_value_at_4_l732_73265


namespace unique_residue_mod_11_l732_73295

theorem unique_residue_mod_11 :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [ZMOD 11] := by
  sorry

end unique_residue_mod_11_l732_73295


namespace max_sin_squared_sum_l732_73206

theorem max_sin_squared_sum (A B C : Real) (a b c : Real) :
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = Real.pi) →
  (a > 0) → (b > 0) → (c > 0) →
  (a / (Real.sin A) = b / (Real.sin B)) →
  (b / (Real.sin B) = c / (Real.sin C)) →
  ((2 * Real.sin A - Real.sin C) / Real.sin C = (a^2 + b^2 - c^2) / (a^2 + c^2 - b^2)) →
  (∃ (x : Real), x = Real.sin A^2 + Real.sin C^2 ∧ ∀ (y : Real), y = Real.sin A^2 + Real.sin C^2 → y ≤ x) →
  (Real.sin A^2 + Real.sin C^2 ≤ 3/2) :=
by sorry

end max_sin_squared_sum_l732_73206


namespace maria_candy_l732_73227

/-- Calculates the remaining candy pieces after eating some. -/
def remaining_candy (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem: Maria has 3 pieces of candy left -/
theorem maria_candy : remaining_candy 67 64 = 3 := by
  sorry

end maria_candy_l732_73227


namespace quadratic_one_root_l732_73260

/-- A quadratic equation with coefficients m and n has exactly one real root if and only if m > 0 and n = 9m^2 -/
theorem quadratic_one_root (m n : ℝ) : 
  (∃! x : ℝ, x^2 + 6*m*x + n = 0) ∧ (m > 0) ∧ (n > 0) ↔ (m > 0) ∧ (n = 9*m^2) := by
sorry

end quadratic_one_root_l732_73260


namespace smallest_common_multiple_of_6_and_15_l732_73270

theorem smallest_common_multiple_of_6_and_15 :
  ∃ b : ℕ, b > 0 ∧ 6 ∣ b ∧ 15 ∣ b ∧ ∀ x : ℕ, x > 0 → 6 ∣ x → 15 ∣ x → b ≤ x :=
by
  -- The proof goes here
  sorry

end smallest_common_multiple_of_6_and_15_l732_73270


namespace smallest_with_eight_factors_l732_73229

def factorCount (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_with_eight_factors :
  ∀ n : ℕ, n > 0 → factorCount n = 8 → n ≥ 24 :=
by sorry

end smallest_with_eight_factors_l732_73229


namespace largest_inclination_angle_l732_73273

-- Define the inclination angle function
noncomputable def inclinationAngle (m : ℝ) : ℝ := Real.arctan m

-- Define the lines
def line1 (x : ℝ) : ℝ := -x + 1
def line2 (x : ℝ) : ℝ := x + 1
def line3 (x : ℝ) : ℝ := 2*x + 1
def line4 : ℝ → Prop := λ x => x = 1

-- Theorem statement
theorem largest_inclination_angle :
  ∀ (θ1 θ2 θ3 θ4 : ℝ),
    θ1 = inclinationAngle (-1) →
    θ2 = inclinationAngle 1 →
    θ3 = inclinationAngle 2 →
    θ4 = Real.pi / 2 →
    θ1 > θ2 ∧ θ1 > θ3 ∧ θ1 > θ4 :=
sorry

end largest_inclination_angle_l732_73273


namespace specific_prism_surface_area_l732_73228

/-- A right prism with an isosceles trapezoid base -/
structure RightPrism where
  AB : ℝ
  BC : ℝ
  AD : ℝ
  diagonal_cross_section_area : ℝ

/-- The total surface area of the right prism -/
def total_surface_area (p : RightPrism) : ℝ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the total surface area of the specific prism -/
theorem specific_prism_surface_area :
  ∃ (p : RightPrism),
    p.AB = 13 ∧
    p.BC = 11 ∧
    p.AD = 21 ∧
    p.diagonal_cross_section_area = 180 ∧
    total_surface_area p = 906 := by
  sorry

end specific_prism_surface_area_l732_73228


namespace store_profit_calculation_l732_73211

theorem store_profit_calculation (C : ℝ) : 
  let initial_markup := 0.20
  let new_year_markup := 0.25
  let february_discount := 0.18
  let initial_price := C * (1 + initial_markup)
  let new_year_price := initial_price * (1 + new_year_markup)
  let final_price := new_year_price * (1 - february_discount)
  let profit := final_price - C
  profit = 0.23 * C := by sorry

end store_profit_calculation_l732_73211


namespace february_first_is_monday_l732_73216

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a day in February -/
structure FebruaryDay where
  day : Nat
  weekday : Weekday

/-- Defines the properties of February in the given year -/
structure FebruaryProperties where
  days : List FebruaryDay
  monday_count : Nat
  thursday_count : Nat
  first_day : Weekday

/-- Theorem stating that if February has exactly four Mondays and four Thursdays,
    then February 1 must be a Monday -/
theorem february_first_is_monday
  (feb : FebruaryProperties)
  (h1 : feb.monday_count = 4)
  (h2 : feb.thursday_count = 4)
  : feb.first_day = Weekday.Monday := by
  sorry

end february_first_is_monday_l732_73216


namespace rectangle_area_l732_73233

/-- The area of a rectangle with sides 5.9 cm and 3 cm is 17.7 square centimeters. -/
theorem rectangle_area : 
  let side1 : ℝ := 5.9
  let side2 : ℝ := 3
  side1 * side2 = 17.7 := by
  sorry

end rectangle_area_l732_73233


namespace sum_even_divisors_1000_l732_73283

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem sum_even_divisors_1000 : sum_even_divisors 1000 = 2184 := by sorry

end sum_even_divisors_1000_l732_73283


namespace susan_spending_ratio_l732_73279

/-- Proves that the ratio of the amount spent on books to the amount left after buying clothes is 1:2 --/
theorem susan_spending_ratio (
  total_earned : ℝ)
  (spent_on_clothes : ℝ)
  (left_after_books : ℝ)
  (h1 : total_earned = 600)
  (h2 : spent_on_clothes = total_earned / 2)
  (h3 : left_after_books = 150) :
  (total_earned - spent_on_clothes - left_after_books) / (total_earned - spent_on_clothes) = 1 / 2 := by
  sorry

end susan_spending_ratio_l732_73279


namespace midpoint_between_fractions_l732_73280

theorem midpoint_between_fractions :
  (1 / 12 + 1 / 15) / 2 = 3 / 40 := by
  sorry

end midpoint_between_fractions_l732_73280


namespace max_abs_z5_l732_73219

open Complex

theorem max_abs_z5 (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h1 : abs z₁ ≤ 1) (h2 : abs z₂ ≤ 1)
  (h3 : abs (2 * z₃ - (z₁ + z₂)) ≤ abs (z₁ - z₂))
  (h4 : abs (2 * z₄ - (z₁ + z₂)) ≤ abs (z₁ - z₂))
  (h5 : abs (2 * z₅ - (z₃ + z₄)) ≤ abs (z₃ - z₄)) :
  abs z₅ ≤ Real.sqrt 3 ∧ ∃ z₁ z₂ z₃ z₄ z₅ : ℂ, abs z₅ = Real.sqrt 3 := by
  sorry

end max_abs_z5_l732_73219


namespace parabola_point_shift_l732_73237

/-- Given a point P(m,n) on the parabola y = ax^2 (a ≠ 0), 
    prove that (m-1, n) lies on y = a(x+1)^2 -/
theorem parabola_point_shift (a m n : ℝ) (h1 : a ≠ 0) (h2 : n = a * m^2) :
  n = a * ((m - 1) + 1)^2 := by
  sorry

end parabola_point_shift_l732_73237


namespace rectangle_length_equals_two_l732_73223

theorem rectangle_length_equals_two
  (square_side : ℝ)
  (rectangle_width : ℝ)
  (h1 : square_side = 4)
  (h2 : rectangle_width = 8)
  (h3 : square_side ^ 2 = rectangle_width * rectangle_length) :
  rectangle_length = 2 :=
by
  sorry

#check rectangle_length_equals_two

end rectangle_length_equals_two_l732_73223


namespace alan_tickets_l732_73230

theorem alan_tickets (total : ℕ) (alan : ℕ) (marcy : ℕ) 
  (h1 : total = 150)
  (h2 : alan + marcy = total)
  (h3 : marcy = 5 * alan - 6) :
  alan = 26 := by
sorry

end alan_tickets_l732_73230


namespace difference_ones_zeros_is_six_l732_73263

-- Define the number in base 10
def base_10_num : Nat := 253

-- Define a function to convert a number to its binary representation
def to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

-- Define functions to count zeros and ones in a binary representation
def count_zeros (binary : List Bool) : Nat :=
  binary.filter (· = false) |>.length

def count_ones (binary : List Bool) : Nat :=
  binary.filter (· = true) |>.length

-- Theorem statement
theorem difference_ones_zeros_is_six :
  let binary := to_binary base_10_num
  let y := count_ones binary
  let x := count_zeros binary
  y - x = 6 := by sorry

end difference_ones_zeros_is_six_l732_73263


namespace sum_largest_triangles_geq_twice_area_l732_73214

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields and properties to define a convex polygon
  -- This is a simplified representation
  vertices : Set ℝ × ℝ
  is_convex : sorry

/-- The area of a polygon -/
def area (P : ConvexPolygon) : ℝ := sorry

/-- The largest triangle area for a given side of the polygon -/
def largest_triangle_area (P : ConvexPolygon) (side : ℝ × ℝ × ℝ × ℝ) : ℝ := sorry

/-- The sum of largest triangle areas for all sides of the polygon -/
def sum_largest_triangle_areas (P : ConvexPolygon) : ℝ := sorry

/-- Theorem: The sum of the areas of the largest triangles formed within P, 
    each having one side coinciding with a side of P, 
    is at least twice the area of P -/
theorem sum_largest_triangles_geq_twice_area (P : ConvexPolygon) :
  sum_largest_triangle_areas P ≥ 2 * area P := by sorry

end sum_largest_triangles_geq_twice_area_l732_73214


namespace first_term_of_constant_ratio_l732_73274

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a + (n - 1 : ℚ) * d) / 2

theorem first_term_of_constant_ratio (d : ℚ) (h : d = 5) :
  (∃ k : ℚ, ∀ n : ℕ, n > 0 → 
    arithmetic_sum a d (4*n) / arithmetic_sum a d n = k) →
  a = -5/2 :=
sorry

end first_term_of_constant_ratio_l732_73274


namespace greatest_among_given_numbers_l732_73293

theorem greatest_among_given_numbers :
  let a := (42 : ℚ) * (7 / 11) / 100
  let b := 17 / 23
  let c := (7391 : ℚ) / 10000
  let d := 29 / 47
  b ≥ a ∧ b ≥ c ∧ b ≥ d := by sorry

end greatest_among_given_numbers_l732_73293


namespace triangle_property_l732_73298

theorem triangle_property (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  A < π/2 ∧ B < π/2 ∧ C < π/2 →
  S > 0 →
  S = (1/2) * b * c * Real.sin A →
  (b - c) * Real.sin B = b * Real.sin (A - C) →
  A = π/3 ∧ 
  4 * Real.sqrt 3 ≤ (a^2 + b^2 + c^2) / S ∧ 
  (a^2 + b^2 + c^2) / S < 16 * Real.sqrt 3 / 3 := by
sorry

end triangle_property_l732_73298


namespace family_theorem_l732_73254

structure Family where
  teresa_age : ℕ
  morio_age : ℕ
  morio_age_at_michiko_birth : ℕ
  kenji_michiko_age_diff : ℕ
  yuki_kenji_age_diff : ℕ
  years_after_yuki_adoption_to_anniversary : ℕ
  anniversary_years : ℕ

def years_into_marriage_at_michiko_birth (f : Family) : ℕ := sorry

def teresa_age_at_michiko_birth (f : Family) : ℕ := sorry

theorem family_theorem (f : Family)
  (h1 : f.teresa_age = 59)
  (h2 : f.morio_age = 71)
  (h3 : f.morio_age_at_michiko_birth = 38)
  (h4 : f.kenji_michiko_age_diff = 4)
  (h5 : f.yuki_kenji_age_diff = 3)
  (h6 : f.years_after_yuki_adoption_to_anniversary = 3)
  (h7 : f.anniversary_years = 25) :
  years_into_marriage_at_michiko_birth f = 8 ∧ teresa_age_at_michiko_birth f = 26 := by
  sorry


end family_theorem_l732_73254


namespace reciprocal_location_l732_73218

theorem reciprocal_location (a b : ℝ) (h1 : a < 0) (h2 : b < 0) (h3 : a^2 + b^2 < 1) :
  let F := Complex.mk a b
  let recip := F⁻¹
  (Complex.re recip > 0) ∧ (Complex.im recip > 0) ∧ (Complex.abs recip > 1) := by
  sorry

end reciprocal_location_l732_73218


namespace probability_one_good_one_inferior_l732_73226

/-- The probability of drawing one good quality bulb and one inferior quality bulb -/
theorem probability_one_good_one_inferior (total : ℕ) (good : ℕ) (inferior : ℕ) :
  total = 6 →
  good = 4 →
  inferior = 2 →
  (good + inferior : ℚ) / total * inferior / total + inferior / total * good / total = 4 / 9 := by
  sorry

end probability_one_good_one_inferior_l732_73226


namespace inequality_solution_set_l732_73234

theorem inequality_solution_set (x : ℝ) : 
  (1 / (x^2 + 4) > 5/x + 21/10) ↔ (-2 < x ∧ x < 0) :=
sorry

end inequality_solution_set_l732_73234


namespace relay_race_arrangements_l732_73253

/-- The number of team members in the relay race -/
def team_size : ℕ := 5

/-- The position in which Jordan (the fastest runner) must run -/
def jordan_position : ℕ := 5

/-- The number of runners that need to be arranged -/
def runners_to_arrange : ℕ := team_size - 1

/-- Calculates the number of permutations of n elements -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

theorem relay_race_arrangements :
  permutations runners_to_arrange = 24 := by
  sorry

end relay_race_arrangements_l732_73253


namespace equation_solution_range_l732_73235

-- Define the set of real numbers greater than 0 and not equal to 1
def A : Set ℝ := {a | a > 0 ∧ a ≠ 1}

-- Define the function representing the equation
def f (a : ℝ) (k : ℝ) (x : ℝ) : Prop :=
  Real.log (x - a * k) / Real.log (Real.sqrt a) = Real.log (x^2 - a^2) / Real.log a

-- Define the set of k values that satisfy the equation for some x
def K (a : ℝ) : Set ℝ := {k | ∃ x, f a k x}

-- Theorem statement
theorem equation_solution_range (a : A) :
  K a = {k | k < -1 ∨ (k > 0 ∧ k < 1)} :=
by sorry

end equation_solution_range_l732_73235


namespace line_points_relation_l732_73202

/-- Given a line x = 5y + 5 passing through points (a, n) and (a + 2, n + 0.4),
    prove that a = 5n + 5 -/
theorem line_points_relation (a n : ℝ) : 
  (a = 5 * n + 5 ∧ (a + 2) = 5 * (n + 0.4) + 5) → a = 5 * n + 5 := by
  sorry

end line_points_relation_l732_73202


namespace greatest_power_of_three_dividing_30_factorial_l732_73245

-- Define v as 30!
def v : ℕ := Nat.factorial 30

-- Define the property that 3^k is a factor of v
def is_factor_of_v (k : ℕ) : Prop := ∃ m : ℕ, v = m * (3^k)

-- Theorem statement
theorem greatest_power_of_three_dividing_30_factorial :
  (∃ k : ℕ, is_factor_of_v k ∧ ∀ j : ℕ, j > k → ¬is_factor_of_v j) ∧
  (∀ k : ℕ, (∃ j : ℕ, j > k ∧ is_factor_of_v j) → k ≤ 14) ∧
  is_factor_of_v 14 := by
  sorry

end greatest_power_of_three_dividing_30_factorial_l732_73245


namespace quadratic_roots_l732_73236

theorem quadratic_roots (m : ℝ) : 
  (∃! x : ℝ, (m + 2) * x^2 - 2 * (m + 1) * x + m = 0) →
  (∃ x : ℝ, (m + 1) * x^2 - 2 * m * x + (m - 2) = 0 ∧
             ∀ y : ℝ, (m + 1) * y^2 - 2 * m * y + (m - 2) = 0 → y = x) :=
by sorry

end quadratic_roots_l732_73236


namespace infinitely_many_pairs_l732_73282

theorem infinitely_many_pairs : 
  Set.Infinite {p : ℕ × ℕ | 2019 < (2 : ℝ)^p.1 / (3 : ℝ)^p.2 ∧ (2 : ℝ)^p.1 / (3 : ℝ)^p.2 < 2020} :=
by sorry

end infinitely_many_pairs_l732_73282


namespace prime_congruence_l732_73257

theorem prime_congruence (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  (∃ x : ℤ, (x^4 + x^3 + x^2 + x + 1) % p = 0) →
  p % 5 = 1 := by
  sorry

end prime_congruence_l732_73257


namespace min_value_x_plus_four_over_x_l732_73288

theorem min_value_x_plus_four_over_x (x : ℝ) (hx : x > 0) :
  x + 4 / x ≥ 4 ∧ ∃ y > 0, y + 4 / y = 4 :=
sorry

end min_value_x_plus_four_over_x_l732_73288


namespace square_root_problem_l732_73247

theorem square_root_problem (x : ℝ) :
  (Real.sqrt 1.21 / Real.sqrt 0.64) + (Real.sqrt x / Real.sqrt 0.49) = 3.0892857142857144 →
  x = 1.44 := by
  sorry

end square_root_problem_l732_73247


namespace bianca_extra_flowers_l732_73275

/-- The number of extra flowers Bianca picked -/
def extra_flowers (tulips roses used : ℕ) : ℕ :=
  tulips + roses - used

/-- Theorem stating that Bianca picked 7 extra flowers -/
theorem bianca_extra_flowers :
  extra_flowers 39 49 81 = 7 := by
  sorry

end bianca_extra_flowers_l732_73275
