import Mathlib

namespace repeating_decimal_sum_equality_l318_31812

/-- Represents a repeating decimal with a repeating part and a period length. -/
structure RepeatingDecimal where
  repeating_part : ℕ
  period_length : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def to_rational (d : RepeatingDecimal) : ℚ :=
  d.repeating_part / (10^d.period_length - 1)

/-- The sum of the three given repeating decimals equals 10099098/29970003. -/
theorem repeating_decimal_sum_equality : 
  let d1 := RepeatingDecimal.mk 3 1
  let d2 := RepeatingDecimal.mk 4 3
  let d3 := RepeatingDecimal.mk 5 4
  to_rational d1 + to_rational d2 + to_rational d3 = 10099098 / 29970003 := by
  sorry

#eval (10099098 : ℚ) / 29970003

end repeating_decimal_sum_equality_l318_31812


namespace mikes_hourly_rate_l318_31838

/-- Given Mike's weekly earnings information, calculate his hourly rate for the second job. -/
theorem mikes_hourly_rate (total_wage : ℚ) (first_job_wage : ℚ) (second_job_hours : ℚ) 
  (h1 : total_wage = 160)
  (h2 : first_job_wage = 52)
  (h3 : second_job_hours = 12) :
  (total_wage - first_job_wage) / second_job_hours = 9 := by
sorry

#eval (160 : ℚ) - (52 : ℚ) / (12 : ℚ)

end mikes_hourly_rate_l318_31838


namespace min_value_theorem_l318_31841

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min_val : ℝ), min_val = 6 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 1 →
    1/(x-1) + 9/(y-1) ≥ min_val := by
  sorry

end min_value_theorem_l318_31841


namespace old_supervisor_salary_is_870_l318_31801

/-- Calculates the old supervisor's salary given the conditions of the problem -/
def old_supervisor_salary (num_workers : ℕ) (initial_avg : ℚ) (new_avg : ℚ) (new_supervisor_salary : ℚ) : ℚ :=
  (num_workers + 1) * initial_avg - (num_workers * new_avg + new_supervisor_salary)

/-- Theorem stating that the old supervisor's salary is $870 given the problem conditions -/
theorem old_supervisor_salary_is_870 :
  old_supervisor_salary 8 430 410 690 = 870 := by
  sorry

#eval old_supervisor_salary 8 430 410 690

end old_supervisor_salary_is_870_l318_31801


namespace number_divisibility_l318_31894

theorem number_divisibility (N : ℕ) : 
  N % 5 = 0 ∧ N % 4 = 2 → N / 5 = 2 := by
  sorry

end number_divisibility_l318_31894


namespace expected_value_of_three_from_seven_l318_31882

/-- The number of marbles in the bag -/
def n : ℕ := 7

/-- The number of marbles drawn -/
def k : ℕ := 3

/-- The sum of numbers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The average value of a set of k elements from 1 to n -/
def avg_value (n k : ℕ) : ℚ := (sum_to_n n : ℚ) / n * k

/-- The expected value of the sum of k randomly chosen marbles from n marbles -/
def expected_value (n k : ℕ) : ℚ := avg_value n k

theorem expected_value_of_three_from_seven :
  expected_value n k = 12 := by sorry

end expected_value_of_three_from_seven_l318_31882


namespace sum_of_squares_l318_31897

theorem sum_of_squares (a b : ℝ) (h1 : a - b = 6) (h2 : a * b = 7) :
  a^2 + b^2 = 50 := by
sorry

end sum_of_squares_l318_31897


namespace performance_orders_count_l318_31864

/-- The number of ways to select 4 programs from 8 options -/
def total_options : ℕ := 8

/-- The number of programs to be selected -/
def selected_programs : ℕ := 4

/-- The number of special programs (A and B) -/
def special_programs : ℕ := 2

/-- The number of non-special programs -/
def other_programs : ℕ := total_options - special_programs

/-- Calculates the number of performance orders with only one special program -/
def orders_with_one_special : ℕ :=
  special_programs * (Nat.choose other_programs (selected_programs - 1)) * (Nat.factorial selected_programs)

/-- Calculates the number of performance orders with both special programs -/
def orders_with_both_special : ℕ :=
  (Nat.choose other_programs (selected_programs - 2)) * (Nat.factorial 2) * (Nat.factorial (selected_programs - 2))

/-- The total number of valid performance orders -/
def total_orders : ℕ := orders_with_one_special + orders_with_both_special

theorem performance_orders_count :
  total_orders = 2860 :=
sorry

end performance_orders_count_l318_31864


namespace product_of_squares_minus_seven_squares_l318_31803

theorem product_of_squares_minus_seven_squares 
  (a b c d : ℤ) : (a^2 - 7*b^2) * (c^2 - 7*d^2) = (a*c + 7*b*d)^2 - 7*(a*d + b*c)^2 :=
by sorry

end product_of_squares_minus_seven_squares_l318_31803


namespace range_of_a_l318_31880

theorem range_of_a (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3) 
  (sum_sq : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l318_31880


namespace descending_order_abc_l318_31802

theorem descending_order_abc : 3^34 > 2^51 ∧ 2^51 > 4^25 := by
  sorry

end descending_order_abc_l318_31802


namespace gcd_4004_10010_l318_31890

theorem gcd_4004_10010 : Nat.gcd 4004 10010 = 2002 := by
  sorry

end gcd_4004_10010_l318_31890


namespace milk_production_theorem_l318_31833

/-- Represents the milk production scenario with varying cow efficiencies -/
structure MilkProduction where
  a : ℕ  -- number of cows in original group
  b : ℝ  -- gallons of milk produced by original group
  c : ℕ  -- number of days for original group
  d : ℕ  -- number of cows in new group
  e : ℕ  -- number of days for new group
  h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0  -- ensure positive values

/-- The theorem stating the milk production for the new group -/
theorem milk_production_theorem (mp : MilkProduction) :
  let avg_rate := mp.b / (mp.a * mp.c)
  let efficient_rate := 2 * avg_rate
  let inefficient_rate := avg_rate / 2
  let new_production := mp.d * (efficient_rate * mp.a / 2 + inefficient_rate * mp.a / 2) / mp.a * mp.e
  new_production = mp.d * mp.b * mp.e / (mp.a * mp.c) := by
  sorry

#check milk_production_theorem

end milk_production_theorem_l318_31833


namespace divisibility_problem_solutions_l318_31835

/-- The set of solutions for the divisibility problem -/
def SolutionSet : Set (ℕ × ℕ) := {(1, 1), (1, 5), (5, 1)}

/-- The divisibility condition -/
def DivisibilityCondition (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ (m * n) ∣ ((2^(2^n) + 1) * (2^(2^m) + 1))

/-- Theorem stating that the SolutionSet contains all and only the pairs satisfying the divisibility condition -/
theorem divisibility_problem_solutions :
  ∀ m n : ℕ, DivisibilityCondition m n ↔ (m, n) ∈ SolutionSet := by
  sorry


end divisibility_problem_solutions_l318_31835


namespace tennis_tournament_matches_l318_31836

theorem tennis_tournament_matches (total_players : ℕ) (advanced_players : ℕ) 
  (h1 : total_players = 128)
  (h2 : advanced_players = 20)
  (h3 : total_players > advanced_players) :
  (total_players - 1 : ℕ) = 127 := by
sorry

end tennis_tournament_matches_l318_31836


namespace cos_theta_value_l318_31842

theorem cos_theta_value (x y : ℝ) (θ : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (hθ : θ ∈ Set.Ioo (π/4) (π/2))
  (h1 : y / Real.sin θ = x / Real.cos θ)
  (h2 : 10 / (x^2 + y^2) = 3 / (x * y)) :
  Real.cos θ = Real.sqrt 10 / 10 := by
sorry

end cos_theta_value_l318_31842


namespace yoongi_has_fewest_apples_l318_31859

def jungkook_apples : ℕ := 6 * 3
def yoongi_apples : ℕ := 4
def yuna_apples : ℕ := 5

theorem yoongi_has_fewest_apples :
  yoongi_apples ≤ jungkook_apples ∧ yoongi_apples ≤ yuna_apples := by
  sorry

end yoongi_has_fewest_apples_l318_31859


namespace furniture_purchase_cost_l318_31834

/-- Calculate the final cost of furniture purchase --/
theorem furniture_purchase_cost :
  let table_cost : ℚ := 140
  let chair_cost : ℚ := table_cost / 7
  let sofa_cost : ℚ := 2 * table_cost
  let num_chairs : ℕ := 4
  let table_discount_rate : ℚ := 1 / 10
  let sales_tax_rate : ℚ := 7 / 100
  let exchange_rate : ℚ := 12 / 10

  let total_chair_cost : ℚ := num_chairs * chair_cost
  let discounted_table_cost : ℚ := table_cost * (1 - table_discount_rate)
  let subtotal : ℚ := discounted_table_cost + total_chair_cost + sofa_cost
  let sales_tax : ℚ := subtotal * sales_tax_rate
  let final_cost : ℚ := subtotal + sales_tax

  final_cost = 52002 / 100 := by sorry

end furniture_purchase_cost_l318_31834


namespace sequence_product_l318_31869

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The line l passing through the origin with normal vector (3,1) -/
def Line (x y : ℝ) : Prop := 3 * x + y = 0

/-- The sequence {a_n} satisfies the condition that (a_{n+1}, a_n) lies on the line for all n -/
def SequenceOnLine (a : Sequence) : Prop := ∀ n : ℕ, Line (a (n + 1)) (a n)

theorem sequence_product (a : Sequence) (h1 : SequenceOnLine a) (h2 : a 2 = 6) :
  a 1 * a 2 * a 3 * a 4 * a 5 = -32 := by
  sorry

end sequence_product_l318_31869


namespace angle_in_second_quadrant_l318_31830

theorem angle_in_second_quadrant : ∃ θ : Real, 
  θ = -10 * Real.pi / 3 ∧ 
  π / 2 < θ % (2 * π) ∧ 
  θ % (2 * π) < π :=
sorry

end angle_in_second_quadrant_l318_31830


namespace smallest_five_digit_multiple_correct_l318_31818

/-- The smallest positive five-digit number divisible by 2, 3, 5, 7, and 11 -/
def smallest_five_digit_multiple : ℕ := 11550

/-- The five smallest prime numbers -/
def smallest_primes : List ℕ := [2, 3, 5, 7, 11]

theorem smallest_five_digit_multiple_correct :
  (∀ p ∈ smallest_primes, smallest_five_digit_multiple % p = 0) ∧
  smallest_five_digit_multiple ≥ 10000 ∧
  smallest_five_digit_multiple < 100000 ∧
  (∀ n : ℕ, n < smallest_five_digit_multiple →
    n < 10000 ∨ (∃ p ∈ smallest_primes, n % p ≠ 0)) :=
by sorry

end smallest_five_digit_multiple_correct_l318_31818


namespace class_average_mark_l318_31862

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 10 →
  excluded_students = 5 →
  excluded_avg = 50 →
  remaining_avg = 90 →
  (total_students * (total_students * excluded_avg + (total_students - excluded_students) * remaining_avg) / total_students) / total_students = 70 := by
  sorry

end class_average_mark_l318_31862


namespace pet_food_discount_l318_31866

/-- Proves that the regular discount is 30% given the conditions of the problem -/
theorem pet_food_discount (msrp : ℝ) (sale_price : ℝ) (additional_discount : ℝ) :
  msrp = 45 →
  sale_price = 25.2 →
  additional_discount = 20 →
  ∃ (regular_discount : ℝ),
    sale_price = msrp * (1 - regular_discount / 100) * (1 - additional_discount / 100) ∧
    regular_discount = 30 := by
  sorry

#check pet_food_discount

end pet_food_discount_l318_31866


namespace unique_solution_when_p_equals_two_l318_31881

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^(1/3) + (2 - x)^(1/3)

-- State the theorem
theorem unique_solution_when_p_equals_two :
  ∃! p : ℝ, ∃! x : ℝ, f x = p :=
by
  -- The proof would go here
  sorry

end unique_solution_when_p_equals_two_l318_31881


namespace biker_distance_difference_l318_31892

/-- The difference in distance traveled between two bikers with different speeds over a fixed time -/
theorem biker_distance_difference (alberto_speed bjorn_speed : ℝ) (race_duration : ℝ) 
  (h1 : alberto_speed = 18)
  (h2 : bjorn_speed = 15)
  (h3 : race_duration = 6) :
  alberto_speed * race_duration - bjorn_speed * race_duration = 18 :=
by sorry

end biker_distance_difference_l318_31892


namespace simplify_quadratic_expression_l318_31805

theorem simplify_quadratic_expression (a : ℝ) : -2 * a^2 + 4 * a^2 = 2 * a^2 := by
  sorry

end simplify_quadratic_expression_l318_31805


namespace sum_factorial_units_digit_2023_l318_31851

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def factorialUnitsDigit (n : ℕ) : ℕ :=
  if n > 4 then 0 else unitsDigit (factorial n)

def sumFactorialUnitsDigits (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => (acc + factorialUnitsDigit (i + 1)) % 10) 0

theorem sum_factorial_units_digit_2023 :
  sumFactorialUnitsDigits 2023 = 3 := by sorry

end sum_factorial_units_digit_2023_l318_31851


namespace chi_square_relationship_confidence_l318_31844

/-- The critical value for 99% confidence level in this χ² test -/
def critical_value : ℝ := 6.635

/-- The observed χ² value -/
def observed_chi_square : ℝ := 8.654

/-- The confidence level as a percentage -/
def confidence_level : ℝ := 99

theorem chi_square_relationship_confidence :
  observed_chi_square > critical_value →
  confidence_level = 99 := by
sorry

end chi_square_relationship_confidence_l318_31844


namespace binomial_coefficient_equality_l318_31807

theorem binomial_coefficient_equality (p k : ℕ) (hp : Prime p) :
  ∃ n : ℕ, (n.choose p) = ((n + k).choose p) := by
  sorry

end binomial_coefficient_equality_l318_31807


namespace max_weight_for_john_and_mike_l318_31829

/-- The maximum weight the bench can support -/
def bench_max_weight : ℝ := 1000

/-- The safety margin for one person -/
def safety_margin_one : ℝ := 0.2

/-- The safety margin for two people -/
def safety_margin_two : ℝ := 0.3

/-- John's weight -/
def john_weight : ℝ := 250

/-- Mike's weight -/
def mike_weight : ℝ := 180

/-- Theorem: The maximum weight John and Mike can put on the bar when using the bench together is 270 pounds -/
theorem max_weight_for_john_and_mike : 
  bench_max_weight * (1 - safety_margin_two) - (john_weight + mike_weight) = 270 := by
  sorry

end max_weight_for_john_and_mike_l318_31829


namespace rice_grains_difference_l318_31884

def grains_on_square (k : ℕ) : ℕ := 3^k

def sum_of_grains (n : ℕ) : ℕ := 
  3 * (3^n - 1) / 2

theorem rice_grains_difference : 
  grains_on_square 11 - sum_of_grains 9 = 147624 := by
  sorry

end rice_grains_difference_l318_31884


namespace tiffany_homework_l318_31857

theorem tiffany_homework (math_pages : ℕ) (problems_per_page : ℕ) (total_problems : ℕ) 
  (h1 : math_pages = 6)
  (h2 : problems_per_page = 3)
  (h3 : total_problems = 30) :
  (total_problems - math_pages * problems_per_page) / problems_per_page = 4 := by
  sorry

end tiffany_homework_l318_31857


namespace length_of_diagonal_l318_31885

/-- Given a quadrilateral ABCD with specific side lengths, prove the length of AC -/
theorem length_of_diagonal (AB DC AD : ℝ) (h1 : AB = 15) (h2 : DC = 24) (h3 : AD = 7) :
  ∃ AC : ℝ, abs (AC - Real.sqrt (417 + 112 * Real.sqrt 6)) < 0.05 :=
sorry

end length_of_diagonal_l318_31885


namespace smallest_positive_integer_congruence_l318_31808

theorem smallest_positive_integer_congruence :
  ∃ (y : ℕ), y > 0 ∧ (56 * y + 8) % 26 = 6 % 26 ∧
  ∀ (z : ℕ), z > 0 ∧ (56 * z + 8) % 26 = 6 % 26 → y ≤ z :=
by
  -- The proof goes here
  sorry

end smallest_positive_integer_congruence_l318_31808


namespace range_of_a_for_increasing_f_l318_31886

/-- A function f defined piecewise on the real numbers. -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^x + 1 else -x^2 + a*x + 1

/-- The theorem stating the range of 'a' for which f is increasing on ℝ. -/
theorem range_of_a_for_increasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 3 :=
sorry

end range_of_a_for_increasing_f_l318_31886


namespace x_squared_minus_y_squared_l318_31850

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 9/16) 
  (h2 : x - y = 5/16) : 
  x^2 - y^2 = 45/256 := by
sorry

end x_squared_minus_y_squared_l318_31850


namespace last_letter_of_93rd_perm_l318_31853

def word := "BRAVE"

/-- Represents a permutation of the word "BRAVE" -/
def Permutation := Fin 5 → Char

/-- The set of all permutations of "BRAVE" -/
def all_permutations : Finset Permutation :=
  sorry

/-- Dictionary order for permutations -/
def dict_order (p q : Permutation) : Prop :=
  sorry

/-- The 93rd permutation in dictionary order -/
def perm_93 : Permutation :=
  sorry

theorem last_letter_of_93rd_perm :
  (perm_93 4) = 'R' :=
sorry

end last_letter_of_93rd_perm_l318_31853


namespace constant_term_expansion_l318_31828

theorem constant_term_expansion (n : ℕ) : 
  (∃ (k : ℕ), (Nat.choose n (2*n/3 : ℕ)) = 15 ∧ 2*n/3 = k) → n = 6 :=
by sorry

end constant_term_expansion_l318_31828


namespace jack_waiting_time_l318_31876

/-- The total waiting time in hours for Jack's travel to Canada -/
def total_waiting_time (customs_hours : ℕ) (quarantine_days : ℕ) : ℕ :=
  customs_hours + 24 * quarantine_days

/-- Theorem stating that Jack's total waiting time is 356 hours -/
theorem jack_waiting_time :
  total_waiting_time 20 14 = 356 := by
  sorry

end jack_waiting_time_l318_31876


namespace parallelogram_area_is_3sqrt14_l318_31860

-- Define the complex equations
def equation1 (z : ℂ) : Prop := z^2 = 9 + 9 * Complex.I * Real.sqrt 7
def equation2 (z : ℂ) : Prop := z^2 = 3 + 3 * Complex.I * Real.sqrt 2

-- Define the solutions
def solutions : Set ℂ := {z : ℂ | equation1 z ∨ equation2 z}

-- Define the parallelogram area function
noncomputable def parallelogramArea (vertices : Set ℂ) : ℝ :=
  sorry -- Actual implementation would go here

-- Theorem statement
theorem parallelogram_area_is_3sqrt14 :
  parallelogramArea solutions = 3 * Real.sqrt 14 :=
sorry

end parallelogram_area_is_3sqrt14_l318_31860


namespace commission_calculation_l318_31877

/-- Calculates the commission amount given a commission rate and total sales -/
def calculate_commission (rate : ℚ) (sales : ℚ) : ℚ :=
  rate * sales

theorem commission_calculation :
  let rate : ℚ := 25 / 1000  -- 2.5% expressed as a rational number
  let sales : ℚ := 600
  calculate_commission rate sales = 15 := by
  sorry

end commission_calculation_l318_31877


namespace kenneth_earnings_l318_31815

def earnings_problem (E : ℝ) : Prop :=
  let joystick := 0.10 * E
  let accessories := 0.15 * E
  let phone_bill := 0.05 * E
  let snacks := 0.20 * E - 25
  let utility := 0.25 * E - 15
  let remaining := 405
  E = joystick + accessories + phone_bill + snacks + utility + remaining

theorem kenneth_earnings : 
  ∃ E : ℝ, earnings_problem E ∧ E = 1460 :=
sorry

end kenneth_earnings_l318_31815


namespace sqrt_inequality_l318_31895

theorem sqrt_inequality (a b : ℝ) : Real.sqrt a < Real.sqrt b → a < b := by
  sorry

end sqrt_inequality_l318_31895


namespace same_color_plate_probability_l318_31875

theorem same_color_plate_probability (total : ℕ) (yellow : ℕ) (green : ℕ) 
  (h1 : total = yellow + green)
  (h2 : yellow = 7)
  (h3 : green = 5) :
  (Nat.choose yellow 2 + Nat.choose green 2) / Nat.choose total 2 = 31 / 66 := by
  sorry

end same_color_plate_probability_l318_31875


namespace outfit_choices_l318_31858

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 6

/-- The number of shirts available -/
def num_shirts : ℕ := num_colors

/-- The number of pants available -/
def num_pants : ℕ := num_colors

/-- The number of hats available -/
def num_hats : ℕ := num_colors

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- The number of outfits where all items are the same color -/
def same_color_outfits : ℕ := num_colors

/-- Theorem: The number of outfit choices where not all items are the same color -/
theorem outfit_choices : 
  total_combinations - same_color_outfits = 210 :=
sorry

end outfit_choices_l318_31858


namespace divisor_cube_eq_four_n_l318_31854

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The set of solutions to the equation d(n)^3 = 4n -/
def solution_set : Set ℕ := {2, 128, 2000}

/-- Theorem stating that n is a solution if and only if it's in the solution set -/
theorem divisor_cube_eq_four_n (n : ℕ) : 
  (num_divisors n)^3 = 4 * n ↔ n ∈ solution_set := by sorry

end divisor_cube_eq_four_n_l318_31854


namespace sum_distances_geq_6r_sum_squared_distances_geq_12r_squared_l318_31896

-- Define a triangle in a plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point in the plane
def Point : Type := ℝ × ℝ

-- Define the distance function
def distance (p1 p2 : Point) : ℝ := sorry

-- Define the radius of the inscribed circle
def inRadius (t : Triangle) : ℝ := sorry

-- Define Ra, Rb, Rc
def Ra (t : Triangle) (M : Point) : ℝ := distance M t.A
def Rb (t : Triangle) (M : Point) : ℝ := distance M t.B
def Rc (t : Triangle) (M : Point) : ℝ := distance M t.C

-- Theorem 1
theorem sum_distances_geq_6r (t : Triangle) (M : Point) :
  Ra t M + Rb t M + Rc t M ≥ 6 * inRadius t := sorry

-- Theorem 2
theorem sum_squared_distances_geq_12r_squared (t : Triangle) (M : Point) :
  Ra t M ^ 2 + Rb t M ^ 2 + Rc t M ^ 2 ≥ 12 * (inRadius t) ^ 2 := sorry

end sum_distances_geq_6r_sum_squared_distances_geq_12r_squared_l318_31896


namespace integer_sum_problem_l318_31865

theorem integer_sum_problem (x y : ℕ+) 
  (h1 : x.val - y.val = 15)
  (h2 : x.val * y.val = 56) :
  x.val + y.val = Real.sqrt 449 := by
  sorry

end integer_sum_problem_l318_31865


namespace sqrt_square_eq_abs_l318_31840

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by sorry

end sqrt_square_eq_abs_l318_31840


namespace shortest_distance_proof_l318_31837

/-- Given a body moving on a horizontal plane, prove that with displacements of 4 meters
    along the x-axis and 3 meters along the y-axis, the shortest distance between
    the initial and final points is 5 meters. -/
theorem shortest_distance_proof (x y : ℝ) (hx : x = 4) (hy : y = 3) :
  Real.sqrt (x^2 + y^2) = 5 := by
  sorry

end shortest_distance_proof_l318_31837


namespace average_of_eleven_numbers_l318_31883

theorem average_of_eleven_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (middle : ℝ) :
  first_six_avg = 10.5 →
  last_six_avg = 11.4 →
  middle = 22.5 →
  (6 * first_six_avg + 6 * last_six_avg - middle) / 11 = 9.9 := by
sorry

end average_of_eleven_numbers_l318_31883


namespace equation_solution_l318_31887

theorem equation_solution :
  ∃ x : ℚ, (1 / (x + 2) + 3 * x / (x + 2) + 4 / (x + 2) = 1) ∧ (x = -3 / 2) := by
  sorry

end equation_solution_l318_31887


namespace weight_of_three_moles_l318_31806

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.01

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in C6H8O6 -/
def carbon_count : ℕ := 6

/-- The number of Hydrogen atoms in C6H8O6 -/
def hydrogen_count : ℕ := 8

/-- The number of Oxygen atoms in C6H8O6 -/
def oxygen_count : ℕ := 6

/-- The number of moles of C6H8O6 -/
def mole_count : ℝ := 3

/-- The molecular weight of C6H8O6 in g/mol -/
def molecular_weight : ℝ := 
  carbon_count * carbon_weight + 
  hydrogen_count * hydrogen_weight + 
  oxygen_count * oxygen_weight

/-- The total weight of 3 moles of C6H8O6 in grams -/
theorem weight_of_three_moles : 
  mole_count * molecular_weight = 528.42 := by sorry

end weight_of_three_moles_l318_31806


namespace area_conversion_time_conversion_l318_31863

-- Define the conversion factors
def square_meters_per_hectare : ℝ := 10000
def minutes_per_hour : ℝ := 60

-- Define the input values
def area_in_square_meters : ℝ := 123000
def time_in_hours : ℝ := 4.25

-- Theorem for area conversion
theorem area_conversion :
  area_in_square_meters / square_meters_per_hectare = 12.3 := by sorry

-- Theorem for time conversion
theorem time_conversion :
  ∃ (whole_hours minutes : ℕ),
    whole_hours = 4 ∧
    minutes = 15 ∧
    time_in_hours = whole_hours + (minutes : ℝ) / minutes_per_hour := by sorry

end area_conversion_time_conversion_l318_31863


namespace y_decreases_as_x_increases_l318_31899

/-- A linear function y = -2x - 3 -/
def f (x : ℝ) : ℝ := -2 * x - 3

/-- Theorem: For any two points on the graph of f, if the x-coordinate of the first point
    is less than the x-coordinate of the second point, then the y-coordinate of the first point
    is greater than the y-coordinate of the second point. -/
theorem y_decreases_as_x_increases (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f x₂ = y₂) 
  (h3 : x₁ < x₂) : 
  y₁ > y₂ := by
  sorry

end y_decreases_as_x_increases_l318_31899


namespace solve_walnuts_problem_l318_31821

def walnuts_problem (initial_walnuts boy_gathered girl_gathered girl_ate final_walnuts : ℕ) : Prop :=
  ∃ (dropped : ℕ),
    initial_walnuts + boy_gathered - dropped + girl_gathered - girl_ate = final_walnuts

theorem solve_walnuts_problem :
  walnuts_problem 12 6 5 2 20 → 
  ∃ (dropped : ℕ), dropped = 1 := by
  sorry

end solve_walnuts_problem_l318_31821


namespace chess_tournament_games_l318_31873

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) : 
  n = 5 → total_games = 20 → (n * (n - 1)) / 2 = total_games → n - 1 = 4 := by
  sorry

end chess_tournament_games_l318_31873


namespace intersection_implies_a_value_l318_31871

theorem intersection_implies_a_value (A B : Set ℝ) (a : ℝ) : 
  A = {-1, 1, 3} →
  B = {a + 2, a^2 + 4} →
  A ∩ B = {1} →
  a = 1 :=
by sorry

end intersection_implies_a_value_l318_31871


namespace parabola_directrix_l318_31800

/-- Given a parabola with equation x^2 = (1/2)y, its directrix is y = -1/8 -/
theorem parabola_directrix (x y : ℝ) : 
  (x^2 = (1/2) * y) → (∃ p : ℝ, p = 1/4 ∧ y = -p/2) :=
by sorry

end parabola_directrix_l318_31800


namespace angle_with_same_terminal_side_l318_31879

/-- Given an angle α and another angle θ, proves that if α = 1560°, 
    θ has the same terminal side as α, and -360° < θ < 360°, 
    then θ = 120° or θ = -240°. -/
theorem angle_with_same_terminal_side 
  (α θ : ℝ) 
  (h1 : α = 1560)
  (h2 : ∃ (k : ℤ), θ = 360 * k + 120)
  (h3 : -360 < θ ∧ θ < 360) :
  θ = 120 ∨ θ = -240 :=
sorry

end angle_with_same_terminal_side_l318_31879


namespace problem_statement_l318_31872

theorem problem_statement (n : ℕ) (x m : ℝ) :
  let p := x^2 - 2*x - 8 ≤ 0
  let q := |x - 2| ≤ m
  (∀ k : ℕ, k ≤ n → ((-1:ℝ)^k * (n.choose k) = (-1)^n * (n.choose (n-k)))) →
  (
    (m = 3 ∧ p ∧ q) → -1 ≤ x ∧ x ≤ 4
  ) ∧
  (
    (∀ y : ℝ, (y^2 - 2*y - 8 ≤ 0) → |y - 2| ≤ m) → m ≥ 4
  ) :=
by sorry

end problem_statement_l318_31872


namespace students_playing_sports_l318_31856

theorem students_playing_sports (B C : Finset Nat) : 
  (B.card = 7) → 
  (C.card = 8) → 
  ((B ∩ C).card = 5) → 
  ((B ∪ C).card = 10) := by
sorry

end students_playing_sports_l318_31856


namespace circle_symmetry_sum_l318_31826

/-- A circle in the xy-plane -/
structure Circle where
  D : ℝ
  E : ℝ
  F : ℝ

/-- A line in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a circle is symmetric with respect to a line -/
def isSymmetric (circle : Circle) (line : Line) : Prop :=
  sorry

/-- The main theorem -/
theorem circle_symmetry_sum (circle : Circle) 
    (l₁ : Line) (l₂ : Line) :
    l₁ = Line.mk 1 (-1) 4 →
    l₂ = Line.mk 1 3 0 →
    isSymmetric circle l₁ →
    isSymmetric circle l₂ →
    circle.D + circle.E = 4 := by
  sorry

end circle_symmetry_sum_l318_31826


namespace triangle_side_expression_l318_31822

/-- Given a triangle with sides a, b, and c, prove that |a-b+c|-|c-a-b| = 2c-2b -/
theorem triangle_side_expression (a b c : ℝ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  |a - b + c| - |c - a - b| = 2 * c - 2 * b := by
  sorry

end triangle_side_expression_l318_31822


namespace isosceles_triangle_line_equation_l318_31846

/-- An isosceles triangle AOB with given properties -/
structure IsoscelesTriangle where
  /-- Point O is at the origin -/
  O : ℝ × ℝ := (0, 0)
  /-- Point A coordinates -/
  A : ℝ × ℝ := (1, 3)
  /-- Point B is on the positive x-axis -/
  B : ℝ × ℝ
  /-- B's y-coordinate is 0 -/
  h_B_on_x_axis : B.2 = 0
  /-- B's x-coordinate is positive -/
  h_B_positive_x : B.1 > 0
  /-- AO = AB (isosceles property) -/
  h_isosceles : (A.1 - O.1)^2 + (A.2 - O.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2

/-- The equation of line AB in an isosceles triangle AOB is y-3 = -3(x-1) -/
theorem isosceles_triangle_line_equation (t : IsoscelesTriangle) :
  ∀ x y : ℝ, (y - 3 = -3 * (x - 1)) ↔ (∃ k : ℝ, x = t.A.1 + k * (t.B.1 - t.A.1) ∧ y = t.A.2 + k * (t.B.2 - t.A.2)) :=
sorry

end isosceles_triangle_line_equation_l318_31846


namespace delivery_tip_cost_is_eight_l318_31878

/-- Calculates the delivery and tip cost given grocery order details --/
def delivery_and_tip_cost (original_order : ℝ) 
                          (tomatoes_old : ℝ) (tomatoes_new : ℝ)
                          (lettuce_old : ℝ) (lettuce_new : ℝ)
                          (celery_old : ℝ) (celery_new : ℝ)
                          (total_bill : ℝ) : ℝ :=
  let price_increase := (tomatoes_new - tomatoes_old) + 
                        (lettuce_new - lettuce_old) + 
                        (celery_new - celery_old)
  let new_grocery_cost := original_order + price_increase
  total_bill - new_grocery_cost

/-- Theorem stating that the delivery and tip cost is $8.00 --/
theorem delivery_tip_cost_is_eight :
  delivery_and_tip_cost 25 0.99 2.20 1.00 1.75 1.96 2.00 35 = 8 :=
by sorry


end delivery_tip_cost_is_eight_l318_31878


namespace female_students_count_l318_31811

theorem female_students_count (total_average : ℚ) (male_count : ℕ) (male_average : ℚ) (female_average : ℚ) :
  total_average = 90 →
  male_count = 8 →
  male_average = 82 →
  female_average = 92 →
  ∃ (female_count : ℕ),
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 32 :=
by sorry

end female_students_count_l318_31811


namespace grinder_price_is_15000_l318_31816

/-- The price of a grinder and a mobile phone transaction --/
def GrinderMobileTransaction (grinder_price : ℝ) : Prop :=
  let mobile_price : ℝ := 8000
  let grinder_sell_price : ℝ := grinder_price * 0.96
  let mobile_sell_price : ℝ := mobile_price * 1.10
  let total_buy_price : ℝ := grinder_price + mobile_price
  let total_sell_price : ℝ := grinder_sell_price + mobile_sell_price
  total_sell_price - total_buy_price = 200

/-- The grinder price is 15000 given the transaction conditions --/
theorem grinder_price_is_15000 : 
  ∃ (price : ℝ), GrinderMobileTransaction price ∧ price = 15000 := by
  sorry

end grinder_price_is_15000_l318_31816


namespace square_sum_nonzero_iff_not_both_zero_l318_31867

theorem square_sum_nonzero_iff_not_both_zero (a b : ℝ) :
  a^2 + b^2 ≠ 0 ↔ ¬(a = 0 ∧ b = 0) := by
  sorry

end square_sum_nonzero_iff_not_both_zero_l318_31867


namespace sum_of_three_numbers_l318_31839

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 20 := by sorry

end sum_of_three_numbers_l318_31839


namespace sin_symmetry_condition_l318_31843

def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem sin_symmetry_condition (φ : ℝ) :
  (φ = π / 2 → is_symmetric_about_y_axis (fun x ↦ Real.sin (x + φ))) ∧
  ¬(is_symmetric_about_y_axis (fun x ↦ Real.sin (x + φ)) → φ = π / 2) :=
by sorry

end sin_symmetry_condition_l318_31843


namespace problem_statement_l318_31831

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : (12 : ℝ) ^ x = (18 : ℝ) ^ y) (h2 : (12 : ℝ) ^ x = 6 ^ (x * y)) :
  x + y = 2 := by
  sorry

end problem_statement_l318_31831


namespace unique_solution_l318_31868

theorem unique_solution (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (h : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 42) :
  x = 11 ∧ y = 9 ∧ z = 7 := by
  sorry

end unique_solution_l318_31868


namespace product_first_three_is_960_l318_31845

/-- An arithmetic sequence with seventh term 20 and common difference 2 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  8 + 2 * (n - 1)

/-- The product of the first three terms of the arithmetic sequence -/
def product_first_three : ℚ :=
  (arithmetic_sequence 1) * (arithmetic_sequence 2) * (arithmetic_sequence 3)

theorem product_first_three_is_960 :
  product_first_three = 960 :=
by sorry

end product_first_three_is_960_l318_31845


namespace sarahs_bowling_score_l318_31814

theorem sarahs_bowling_score :
  ∀ (sarah_score greg_score : ℕ),
    sarah_score = greg_score + 50 →
    (sarah_score + greg_score) / 2 = 110 →
    sarah_score = 135 :=
by
  sorry

end sarahs_bowling_score_l318_31814


namespace circles_intersect_implies_equilateral_l318_31861

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- Predicate that checks if any two circles intersect -/
def circlesIntersect (t : Triangle) : Prop :=
  t.c/2 ≤ t.a/4 + t.b/4 ∧ t.a/2 ≤ t.b/4 + t.c/4 ∧ t.b/2 ≤ t.c/4 + t.a/4

/-- Predicate that checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- Theorem stating that if circles drawn around midpoints of a triangle's sides
    with radii 1/4 of the side lengths intersect, then the triangle is equilateral -/
theorem circles_intersect_implies_equilateral (t : Triangle) :
  circlesIntersect t → isEquilateral t :=
by
  sorry


end circles_intersect_implies_equilateral_l318_31861


namespace existence_of_larger_prime_factor_l318_31820

theorem existence_of_larger_prime_factor (p : ℕ) (hp : Prime p) (hp_ge_3 : p ≥ 3) :
  ∃ N : ℕ, ∀ x ≥ N, ∃ i ∈ Finset.range ((p + 3) / 2), ∃ q : ℕ, Prime q ∧ q > p ∧ q ∣ (x + i + 1) :=
by sorry

end existence_of_larger_prime_factor_l318_31820


namespace percentage_of_mathematicians_in_it_l318_31855

theorem percentage_of_mathematicians_in_it (total : ℝ) (mathematicians : ℝ) 
  (h1 : mathematicians > 0) 
  (h2 : total > mathematicians) 
  (h3 : 0.7 * mathematicians = 0.07 * total) : 
  mathematicians / total = 0.1 := by
sorry

end percentage_of_mathematicians_in_it_l318_31855


namespace smallest_n_satisfying_conditions_l318_31810

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem smallest_n_satisfying_conditions : 
  ∃ (n : ℕ), is_three_digit n ∧ 
             (9 ∣ (n + 6)) ∧ 
             (6 ∣ (n - 4)) ∧ 
             (∀ m, is_three_digit m → (9 ∣ (m + 6)) → (6 ∣ (m - 4)) → n ≤ m) ∧
             n = 112 := by
  sorry

end smallest_n_satisfying_conditions_l318_31810


namespace coronavirus_diameter_scientific_notation_l318_31825

theorem coronavirus_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000125 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.25 ∧ n = -7 :=
by sorry

end coronavirus_diameter_scientific_notation_l318_31825


namespace fuel_price_increase_l318_31804

/-- Calculates the percentage increase in fuel prices given the original cost for one tank,
    and the new cost for double the capacity. -/
theorem fuel_price_increase (original_cost new_cost : ℝ) : 
  original_cost > 0 →
  new_cost > original_cost * 2 →
  (new_cost - original_cost * 2) / (original_cost * 2) * 100 = 20 :=
by
  sorry

#check fuel_price_increase

end fuel_price_increase_l318_31804


namespace smallest_two_three_digit_multiples_sum_l318_31832

/-- The smallest positive two-digit number -/
def smallest_two_digit : ℕ := 10

/-- The smallest positive three-digit number -/
def smallest_three_digit : ℕ := 100

/-- The smallest positive two-digit multiple of 5 -/
def c : ℕ := smallest_two_digit

/-- The smallest positive three-digit multiple of 7 -/
def d : ℕ := 
  (smallest_three_digit + 7 - 1) / 7 * 7

theorem smallest_two_three_digit_multiples_sum :
  c + d = 115 := by sorry

end smallest_two_three_digit_multiples_sum_l318_31832


namespace bee_count_l318_31824

theorem bee_count (initial_bees new_bees : ℕ) : 
  initial_bees = 16 → new_bees = 7 → initial_bees + new_bees = 23 := by
  sorry

end bee_count_l318_31824


namespace circle_line_intersection_l318_31891

/-- The x-coordinates of the intersection points between a circle and a line -/
theorem circle_line_intersection
  (x1 y1 x2 y2 : ℝ)  -- Endpoints of the circle's diameter
  (m b : ℝ)  -- Line equation coefficients (y = mx + b)
  (h_distinct : (x1, y1) ≠ (x2, y2))  -- Ensure distinct endpoints
  (h_line : m = -1/2 ∧ b = 5)  -- Specific line equation
  (h_endpoints : x1 = 2 ∧ y1 = 4 ∧ x2 = 10 ∧ y2 = 8)  -- Specific endpoint coordinates
  : ∃ (x_left x_right : ℝ),
    x_left = 4.4 - 2.088 ∧
    x_right = 4.4 + 2.088 ∧
    (∀ (x y : ℝ),
      (x - (x1 + x2)/2)^2 + (y - (y1 + y2)/2)^2 = ((x2 - x1)^2 + (y2 - y1)^2)/4 ∧
      y = m * x + b →
      x = x_left ∨ x = x_right) :=
sorry

end circle_line_intersection_l318_31891


namespace sam_placed_twelve_crayons_l318_31888

/-- The number of crayons Sam placed in the drawer -/
def crayons_placed (initial_crayons final_crayons : ℕ) : ℕ :=
  final_crayons - initial_crayons

/-- Theorem: Sam placed 12 crayons in the drawer -/
theorem sam_placed_twelve_crayons :
  crayons_placed 41 53 = 12 := by
  sorry

end sam_placed_twelve_crayons_l318_31888


namespace saras_quarters_l318_31852

theorem saras_quarters (initial_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 21)
  (h2 : final_quarters = 70) : 
  final_quarters - initial_quarters = 49 := by
  sorry

end saras_quarters_l318_31852


namespace frog_path_count_l318_31827

-- Define the octagon and frog movement
def Octagon := Fin 8
def adjacent (v : Octagon) : Set Octagon := {w | (v.val + 1) % 8 = w.val ∨ (v.val + 7) % 8 = w.val}

-- Define the path count function
noncomputable def a (n : ℕ) : ℝ :=
  if n % 2 = 1 then 0
  else ((2 + Real.sqrt 2) ^ ((n / 2) - 1) - (2 - Real.sqrt 2) ^ ((n / 2) - 1)) / Real.sqrt 2

-- State the theorem
theorem frog_path_count :
  ∀ n : ℕ, a n = (if n % 2 = 1 then 0
              else ((2 + Real.sqrt 2) ^ ((n / 2) - 1) - (2 - Real.sqrt 2) ^ ((n / 2) - 1)) / Real.sqrt 2) :=
by sorry

end frog_path_count_l318_31827


namespace rectangle_x_coordinate_l318_31848

/-- A rectangle with vertices (1, 0), (x, 0), (1, 2), and (x, 2) is divided into two identical
    quadrilaterals by a line passing through the origin with slope 0.2.
    This theorem proves that the x-coordinate of the second and fourth vertices is 9. -/
theorem rectangle_x_coordinate (x : ℝ) :
  (∃ (l : Set (ℝ × ℝ)),
    -- Line l passes through the origin
    (0, 0) ∈ l ∧
    -- Line l has slope 0.2
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = 0.2) ∧
    -- Line l divides the rectangle into two identical quadrilaterals
    (∃ (m n : ℝ × ℝ), m ∈ l ∧ n ∈ l ∧
      m.1 = (1 + x) / 2 ∧ m.2 = 1 ∧
      n.1 = (1 + x) / 2 ∧ n.2 = 1)) →
  x = 9 := by
sorry

end rectangle_x_coordinate_l318_31848


namespace tangent_lines_max_area_and_slope_l318_31819

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y - 3 = 0

-- Define point M
def point_M : ℝ × ℝ := (-6, -5)

-- Define point N
def point_N : ℝ × ℝ := (1, 3)

-- Theorem for tangent lines
theorem tangent_lines :
  ∃ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, l₁ x y ↔ x = -6) ∧
    (∀ x y, l₂ x y ↔ 3*x - 4*y - 2 = 0) ∧
    (∀ l, (∀ x y, l x y → circle_C x y) →
          (l (point_M.1) (point_M.2)) →
          (∃ x₀ y₀, circle_C x₀ y₀ ∧ l x₀ y₀ ∧
            ∀ x y, circle_C x y ∧ l x y → (x, y) = (x₀, y₀)) →
          (l = l₁ ∨ l = l₂)) :=
sorry

-- Theorem for maximum area and slope
theorem max_area_and_slope :
  ∃ (max_area : ℝ) (slope₁ slope₂ : ℝ),
    max_area = 8 ∧
    slope₁ = 2 * Real.sqrt 2 ∧
    slope₂ = -2 * Real.sqrt 2 ∧
    (∀ l : ℝ → ℝ → Prop,
      (l (point_N.1) (point_N.2)) →
      (∃ A B : ℝ × ℝ,
        circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
        l A.1 A.2 ∧ l B.1 B.2 ∧ A ≠ B) →
      (∃ C : ℝ × ℝ, C = point_N) →
      (∃ area : ℝ, area ≤ max_area) ∧
      (∃ k : ℝ, (k = slope₁ ∨ k = slope₂) →
        ∀ x y, l x y ↔ y - point_N.2 = k * (x - point_N.1))) :=
sorry

end tangent_lines_max_area_and_slope_l318_31819


namespace f_properties_l318_31823

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

theorem f_properties :
  ∀ a : ℝ,
  (a = -1/2 → 
    (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), ∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x ≥ f a y) ∧
    (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), ∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x ≤ f a y) ∧
    (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x = 1/2 + (Real.exp 1)^2/4) ∧
    (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x = 5/4)) ∧
  ((a ≤ -1 → ∀ x y : ℝ, 0 < x → 0 < y → x < y → f a x > f a y) ∧
   (a ≥ 0 → ∀ x y : ℝ, 0 < x → 0 < y → x < y → f a x < f a y) ∧
   (-1 < a → a < 0 → 
     ∃ z : ℝ, 0 < z ∧ 
     (∀ x y : ℝ, 0 < x → x < y → y < z → f a x > f a y) ∧
     (∀ x y : ℝ, z ≤ x → x < y → f a x < f a y))) ∧
  (-1 < a → a < 0 → 
    (∀ x : ℝ, 0 < x → f a x > 1 + a / 2 * Real.log (-a)) ↔ 1/Real.exp 1 - 1 < a) :=
by sorry

end f_properties_l318_31823


namespace foreign_language_score_l318_31817

theorem foreign_language_score (chinese_score math_score foreign_score : ℕ) : 
  (chinese_score + math_score + foreign_score) / 3 = 95 →
  (chinese_score + math_score) / 2 = 93 →
  foreign_score = 99 := by
sorry

end foreign_language_score_l318_31817


namespace double_discount_reduction_l318_31893

theorem double_discount_reduction (original_price : ℝ) (discount : ℝ) : 
  discount = 0.4 → 
  (1 - (1 - discount) * (1 - discount)) * 100 = 64 := by
sorry

end double_discount_reduction_l318_31893


namespace num_valid_distributions_is_180_l318_31847

/-- Represents a club -/
inductive Club
| ChunhuiLiteratureSociety
| DancersRollerSkatingClub
| BasketballHome
| GoGarden

/-- Represents a student -/
inductive Student
| A
| B
| C
| D
| E

/-- A valid distribution of students to clubs -/
def ValidDistribution := Student → Club

/-- Checks if a distribution is valid according to the problem conditions -/
def isValidDistribution (d : ValidDistribution) : Prop :=
  (∀ c : Club, ∃ s : Student, d s = c) ∧ 
  (d Student.A ≠ Club.GoGarden)

/-- The number of valid distributions -/
def numValidDistributions : ℕ := sorry

/-- The main theorem stating that the number of valid distributions is 180 -/
theorem num_valid_distributions_is_180 : numValidDistributions = 180 := by sorry

end num_valid_distributions_is_180_l318_31847


namespace triangle_medians_area_relationship_l318_31898

/-- Represents a triangle with three medians -/
structure Triangle where
  median1 : ℝ
  median2 : ℝ
  median3 : ℝ
  area : ℝ

/-- The theorem stating the relationship between the medians and area of the triangle -/
theorem triangle_medians_area_relationship (t : Triangle) 
  (h1 : t.median1 = 5)
  (h2 : t.median2 = 7)
  (h3 : t.area = 10 * Real.sqrt 3) :
  t.median3 = 4 * Real.sqrt 3 := by
  sorry

#check triangle_medians_area_relationship

end triangle_medians_area_relationship_l318_31898


namespace tims_change_l318_31849

/-- Tim's change calculation -/
theorem tims_change (initial_amount : ℕ) (spent_amount : ℕ) (change : ℕ) : 
  initial_amount = 50 → spent_amount = 45 → change = initial_amount - spent_amount → change = 5 := by
  sorry

end tims_change_l318_31849


namespace arithmetic_sequence_sum_property_l318_31809

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- Theorem: If S_20 = S_40 for an arithmetic sequence, then S_60 = 0 -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
  (h : seq.S 20 = seq.S 40) : seq.S 60 = 0 := by
  sorry

end arithmetic_sequence_sum_property_l318_31809


namespace system_solution_ratio_l318_31889

theorem system_solution_ratio (x y z a b : ℝ) 
  (eq1 : 4 * x - 3 * y + z = a)
  (eq2 : 6 * y - 8 * x - 2 * z = b)
  (b_nonzero : b ≠ 0)
  (has_solution : ∃ (x y z : ℝ), 4 * x - 3 * y + z = a ∧ 6 * y - 8 * x - 2 * z = b) :
  a / b = -1 / 2 := by
sorry

end system_solution_ratio_l318_31889


namespace rectangular_array_sum_ratio_l318_31870

theorem rectangular_array_sum_ratio (a : Fin 50 → Fin 40 → ℝ) :
  let row_sum : Fin 50 → ℝ := λ i => (Finset.univ.sum (λ j => a i j))
  let col_sum : Fin 40 → ℝ := λ j => (Finset.univ.sum (λ i => a i j))
  let C : ℝ := (Finset.univ.sum row_sum) / 50
  let D : ℝ := (Finset.univ.sum col_sum) / 40
  C / D = 4 / 5 := by
sorry

end rectangular_array_sum_ratio_l318_31870


namespace coefficient_x3y3_l318_31813

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion of (2x-y)^5
def expansion_term (r : ℕ) : ℤ := 
  (2^(5-r)) * ((-1)^r : ℤ) * (binomial 5 r)

-- Define the coefficient of x^3y^3 in (x+y)(2x-y)^5
def coefficient : ℤ := 
  expansion_term 3 + 2 * expansion_term 2

-- Theorem statement
theorem coefficient_x3y3 : coefficient = 40 := by sorry

end coefficient_x3y3_l318_31813


namespace prob_A_hit_given_target_hit_l318_31874

/-- The probability of A hitting the target -/
def prob_A_hit : ℚ := 3/5

/-- The probability of B hitting the target -/
def prob_B_hit : ℚ := 4/5

/-- The probability of the target being hit by either A or B -/
def prob_target_hit : ℚ := 1 - (1 - prob_A_hit) * (1 - prob_B_hit)

/-- The probability of A hitting the target (regardless of B) -/
def prob_A_hit_total : ℚ := prob_A_hit * (1 - prob_B_hit) + prob_A_hit * prob_B_hit

theorem prob_A_hit_given_target_hit :
  prob_A_hit_total / prob_target_hit = 15/23 :=
sorry

end prob_A_hit_given_target_hit_l318_31874
