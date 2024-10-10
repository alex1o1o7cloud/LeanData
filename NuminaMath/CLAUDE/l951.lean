import Mathlib

namespace romney_value_l951_95107

theorem romney_value (N O : ℕ) (a b c d e f : ℕ) :
  (0 < N) → (N < O) →  -- N/O is a proper fraction
  (N = 4) → (O = 7) →  -- N/O = 4/7
  (0 ≤ a) → (a ≤ 9) → (0 ≤ b) → (b ≤ 9) → (0 ≤ c) → (c ≤ 9) →
  (0 ≤ d) → (d ≤ 9) → (0 ≤ e) → (e ≤ 9) → (0 ≤ f) → (f ≤ 9) →  -- Each letter is a digit
  (a ≠ b) → (a ≠ c) → (a ≠ d) → (a ≠ e) → (a ≠ f) →
  (b ≠ c) → (b ≠ d) → (b ≠ e) → (b ≠ f) →
  (c ≠ d) → (c ≠ e) → (c ≠ f) →
  (d ≠ e) → (d ≠ f) →
  (e ≠ f) →  -- All letters are distinct
  (N : ℚ) / (O : ℚ) = 
    (a : ℚ) / 10 + (b : ℚ) / 100 + (c : ℚ) / 1000 + (d : ℚ) / 10000 + (e : ℚ) / 100000 + (f : ℚ) / 1000000 +
    (a : ℚ) / 1000000 + (b : ℚ) / 10000000 + (c : ℚ) / 100000000 + (d : ℚ) / 1000000000 + (e : ℚ) / 10000000000 + (f : ℚ) / 100000000000 +
    (a : ℚ) / 100000000000 + (b : ℚ) / 1000000000000 + (c : ℚ) / 10000000000000 + (d : ℚ) / 100000000000000 + (e : ℚ) / 1000000000000000 + (f : ℚ) / 10000000000000000 +
    (a : ℚ) / 10000000000000000 + (b : ℚ) / 100000000000000000 + (c : ℚ) / 1000000000000000000 + (d : ℚ) / 10000000000000000000 + (e : ℚ) / 100000000000000000000 + (f : ℚ) / 1000000000000000000000 →  -- Decimal representation
  a = 5 ∧ b = 7 ∧ c = 1 ∧ d = 4 ∧ e = 2 ∧ f = 8 := by
  sorry

end romney_value_l951_95107


namespace necessary_not_sufficient_l951_95102

theorem necessary_not_sufficient (a b h : ℝ) (h_pos : h > 0) :
  (∀ a b : ℝ, |a - 1| < h ∧ |b - 1| < h → |a - b| < 2 * h) ∧
  (∃ a b : ℝ, |a - b| < 2 * h ∧ ¬(|a - 1| < h ∧ |b - 1| < h)) :=
by sorry

end necessary_not_sufficient_l951_95102


namespace exactly_one_double_root_l951_95103

/-- The function f(x) representing the left side of the equation -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 2)^2 * (x + 7)^2 + a

/-- The theorem stating the condition for exactly one double-root -/
theorem exactly_one_double_root (a : ℝ) : 
  (∃! x : ℝ, f a x = 0 ∧ (∀ y : ℝ, y ≠ x → f a y > 0)) ↔ a = -39.0625 := by sorry

end exactly_one_double_root_l951_95103


namespace average_of_remaining_numbers_l951_95179

theorem average_of_remaining_numbers
  (n : ℕ)
  (total : ℕ)
  (subset_sum : ℕ)
  (h1 : n = 5)
  (h2 : total = n * 20)
  (h3 : subset_sum = 48)
  (h4 : subset_sum < total) :
  (total - subset_sum) / 2 = 26 := by
sorry

end average_of_remaining_numbers_l951_95179


namespace f_even_and_increasing_l951_95197

-- Define the function
def f (x : ℝ) : ℝ := x^(2/3)

-- State the theorem
theorem f_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by
  sorry

end f_even_and_increasing_l951_95197


namespace solution_set_correct_range_of_b_l951_95193

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + 2

-- Define the solution set for f(x) > 0
def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 then Set.Ioo (1/a) 2
  else if a = 0 then Set.Iio 2
  else if 0 < a ∧ a < 1/2 then Set.Iio 2 ∪ Set.Ioi (1/a)
  else if a = 1/2 then Set.Iio 2 ∪ Set.Ioi 2
  else Set.Iio (1/a) ∪ Set.Ioi 2

-- State the theorem for the solution set
theorem solution_set_correct (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ f a x > 0 :=
sorry

-- State the theorem for the range of b
theorem range_of_b :
  ∀ x ∈ Set.Icc (1/3) 1,
  ∀ m ∈ Set.Icc 1 4,
  f 1 (1/x) + (3 - 2*m)/x ≤ b^2 - 2*b - 2 →
  b ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
sorry

end solution_set_correct_range_of_b_l951_95193


namespace sams_tuna_discount_l951_95112

/-- Calculates the discount per coupon for a tuna purchase. -/
def discount_per_coupon (num_cans : ℕ) (num_coupons : ℕ) (paid : ℕ) (change : ℕ) (cost_per_can : ℕ) : ℕ :=
  let total_paid := paid - change
  let total_cost := num_cans * cost_per_can
  let total_discount := total_cost - total_paid
  total_discount / num_coupons

/-- Proves that the discount per coupon is 25 cents for Sam's tuna purchase. -/
theorem sams_tuna_discount :
  discount_per_coupon 9 5 2000 550 175 = 25 := by
  sorry

end sams_tuna_discount_l951_95112


namespace vacation_fund_adjustment_l951_95174

/-- Calculates the required weekly hours to meet a financial goal after losing one week of work --/
def required_hours (original_weeks : ℕ) (original_hours_per_week : ℕ) (total_earnings : ℕ) : ℚ :=
  let remaining_weeks := original_weeks - 1
  let hourly_rate := (total_earnings : ℚ) / (original_weeks * original_hours_per_week)
  let weekly_earnings_needed := (total_earnings : ℚ) / remaining_weeks
  weekly_earnings_needed / hourly_rate

theorem vacation_fund_adjustment (original_weeks : ℕ) (original_hours_per_week : ℕ) (total_earnings : ℕ) 
    (h1 : original_weeks = 10)
    (h2 : original_hours_per_week = 25)
    (h3 : total_earnings = 2500) :
  ∃ (n : ℕ), n ≤ required_hours original_weeks original_hours_per_week total_earnings ∧ 
             required_hours original_weeks original_hours_per_week total_earnings < n + 1 ∧
             n = 28 :=
  sorry

end vacation_fund_adjustment_l951_95174


namespace base_ten_to_four_156_base_four_to_ten_2130_l951_95195

/-- Converts a natural number from base 10 to base 4 --/
def toBaseFour (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Converts a list of digits in base 4 to a natural number in base 10 --/
def fromBaseFour (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ (digits.length - 1 - i))) 0

theorem base_ten_to_four_156 : toBaseFour 156 = [2, 1, 3, 0] := by sorry

theorem base_four_to_ten_2130 : fromBaseFour [2, 1, 3, 0] = 156 := by sorry

end base_ten_to_four_156_base_four_to_ten_2130_l951_95195


namespace exchange_result_l951_95122

/-- The exchange rate from Canadian dollars (CAD) to Japanese yen (JPY) -/
def exchange_rate : ℝ := 85

/-- The amount of Canadian dollars to be exchanged -/
def cad_amount : ℝ := 5

/-- Theorem stating that exchanging 5 CAD results in 425 JPY -/
theorem exchange_result : cad_amount * exchange_rate = 425 := by
  sorry

end exchange_result_l951_95122


namespace ladder_problem_l951_95169

theorem ladder_problem (ladder_length height base : ℝ) :
  ladder_length = 15 ∧ height = 12 ∧ ladder_length ^ 2 = height ^ 2 + base ^ 2 →
  base = 9 := by
sorry

end ladder_problem_l951_95169


namespace range_of_a_l951_95134

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) : 
  (∀ x, p x ↔ 2*x^2 - 3*x + 1 ≤ 0) →
  (∀ x, q x ↔ (x - a)*(x - a - 1) ≤ 0) →
  (∀ x, p x → (1/2 : ℝ) ≤ x ∧ x ≤ 1) →
  (∀ x, q x → a ≤ x ∧ x ≤ a + 1) →
  (∀ x, ¬(p x) → ¬(q x)) →
  (∃ x, ¬(p x) ∧ q x) →
  0 ≤ a ∧ a ≤ (1/2 : ℝ) :=
by sorry

end range_of_a_l951_95134


namespace permutations_congruence_l951_95182

/-- The number of ways to arrange n elements, choosing k of them -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid permutations of "AAAABBBBCCCCDDDD" -/
def N : ℕ :=
  (choose 5 0 * choose 4 4 * choose 3 3 * choose 4 0) +
  (choose 5 1 * choose 4 3 * choose 3 2 * choose 4 1) +
  (choose 5 2 * choose 4 2 * choose 3 1 * choose 4 2) +
  (choose 5 3 * choose 4 1 * choose 3 0 * choose 4 3)

theorem permutations_congruence :
  N ≡ 581 [MOD 1000] := by sorry

end permutations_congruence_l951_95182


namespace common_remainder_exists_l951_95145

theorem common_remainder_exists : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 312837 % n = 310650 % n ∧ 312837 % n = 96 := by
  sorry

end common_remainder_exists_l951_95145


namespace negative_roots_range_l951_95148

theorem negative_roots_range (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ (1/2)^x = 3*a + 2) → a > -1/3 :=
by sorry

end negative_roots_range_l951_95148


namespace rhombus_longer_diagonal_l951_95136

/-- 
Theorem: In a rhombus with side length 65 units and shorter diagonal 56 units, 
the longer diagonal is 118 units.
-/
theorem rhombus_longer_diagonal 
  (side_length : ℝ) 
  (shorter_diagonal : ℝ) 
  (h1 : side_length = 65) 
  (h2 : shorter_diagonal = 56) : ℝ :=
by
  -- Define the longer diagonal
  let longer_diagonal : ℝ := 118
  
  -- The proof would go here
  sorry

#check rhombus_longer_diagonal

end rhombus_longer_diagonal_l951_95136


namespace sum_of_roots_l951_95131

theorem sum_of_roots (p q r : ℕ+) : 
  4 * (7^(1/4) - 6^(1/4) : ℝ) = p^(1/4) + q^(1/4) - r^(1/4) → p + q + r = 122 := by
  sorry

end sum_of_roots_l951_95131


namespace quadratic_minimum_value_l951_95163

/-- The minimum value of a quadratic function f(x) = ax^2 + (b + 5)x + c where a > 0 -/
theorem quadratic_minimum_value (a b c : ℝ) (ha : a > 0) :
  let f := fun x => a * x^2 + (b + 5) * x + c
  ∃ m, ∀ x, f x ≥ m ∧ ∃ x₀, f x₀ = m :=
by
  sorry

end quadratic_minimum_value_l951_95163


namespace A_mod_126_l951_95132

/-- A function that generates the number A by concatenating all three-digit numbers from 100 to 799 -/
def generate_A : ℕ := sorry

/-- Theorem stating that the number A is congruent to 91 modulo 126 -/
theorem A_mod_126 : generate_A % 126 = 91 := by sorry

end A_mod_126_l951_95132


namespace circular_seating_l951_95170

theorem circular_seating (total_people : Nat) (seated_people : Nat) (arrangements : Nat) :
  total_people = 6 →
  seated_people ≤ total_people →
  arrangements = 144 →
  arrangements = Nat.factorial (seated_people - 1) →
  seated_people = 5 := by
  sorry

end circular_seating_l951_95170


namespace equation_solutions_l951_95180

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (-1 + Real.sqrt 5) / 2 ∧ x₂ = (-1 - Real.sqrt 5) / 2 ∧
    x₁^2 + x₁ - 1 = 0 ∧ x₂^2 + x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = 2/3 ∧
    2*(x₁ - 3) = 3*x₁*(x₁ - 3) ∧ 2*(x₂ - 3) = 3*x₂*(x₂ - 3)) :=
by sorry

end equation_solutions_l951_95180


namespace nine_point_zero_one_closest_l951_95183

def options : List ℝ := [10.01, 9.998, 9.9, 9.01]

def closest_to_nine (x : ℝ) : Prop :=
  ∀ y ∈ options, |x - 9| ≤ |y - 9|

theorem nine_point_zero_one_closest :
  closest_to_nine 9.01 := by sorry

end nine_point_zero_one_closest_l951_95183


namespace rectangle_perimeter_l951_95162

theorem rectangle_perimeter (l w : ℝ) (h1 : l > 0) (h2 : w > 0) : 
  l * w = 360 ∧ (l + 10) * (w - 6) = 360 → 2 * (l + w) = 76 := by
  sorry

end rectangle_perimeter_l951_95162


namespace smallest_h_divisible_by_primes_l951_95196

theorem smallest_h_divisible_by_primes : ∃ (h : ℕ), h > 0 ∧ 
  (∀ (h' : ℕ), h' < h → ¬∃ (k : ℤ), (13 ∣ (h' + k)) ∧ (17 ∣ (h' + k)) ∧ (29 ∣ (h' + k))) ∧
  ∃ (k : ℤ), (13 ∣ (h + k)) ∧ (17 ∣ (h + k)) ∧ (29 ∣ (h + k)) :=
by sorry

#check smallest_h_divisible_by_primes

end smallest_h_divisible_by_primes_l951_95196


namespace hostel_expenditure_equation_l951_95178

/-- Represents the average expenditure calculation for a student hostel with varying group costs. -/
theorem hostel_expenditure_equation 
  (A B C : ℕ) -- Original number of students in each group
  (a b c : ℕ) -- New students in each group
  (X Y Z : ℝ) -- Average expenditure for each group
  (h1 : A + B + C = 35) -- Total original students
  (h2 : a + b + c = 7)  -- Total new students
  : (A * X + B * Y + C * Z) / 35 - 1 = 
    ((A + a) * X + (B + b) * Y + (C + c) * Z + 42) / 42 := by
  sorry

#check hostel_expenditure_equation

end hostel_expenditure_equation_l951_95178


namespace polynomial_remainder_theorem_l951_95175

theorem polynomial_remainder_theorem (c d : ℚ) : 
  let g : ℚ → ℚ := λ x ↦ c * x^3 - 4 * x^2 + d * x - 7
  (g 2 = -7) ∧ (g (-1) = -20) → c = -1/3 ∧ d = 28/3 := by
  sorry

end polynomial_remainder_theorem_l951_95175


namespace three_digit_numbers_from_4_and_5_l951_95176

def is_valid_digit (d : ℕ) : Prop := d = 4 ∨ d = 5

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_formed_from_4_and_5 (n : ℕ) : Prop :=
  is_three_digit_number n ∧
  is_valid_digit (n / 100) ∧
  is_valid_digit ((n / 10) % 10) ∧
  is_valid_digit (n % 10)

def valid_numbers : Finset ℕ :=
  {444, 445, 454, 455, 544, 545, 554, 555}

theorem three_digit_numbers_from_4_and_5 :
  ∀ n : ℕ, is_formed_from_4_and_5 n ↔ n ∈ valid_numbers :=
by sorry

end three_digit_numbers_from_4_and_5_l951_95176


namespace sector_perimeter_l951_95198

/-- Given a circular sector with central angle 2/3π and area 3π, its perimeter is 6 + 2π. -/
theorem sector_perimeter (θ : Real) (S : Real) (R : Real) (l : Real) :
  θ = (2/3) * Real.pi →
  S = 3 * Real.pi →
  S = (1/2) * θ * R^2 →
  l = θ * R →
  (l + 2 * R) = 6 + 2 * Real.pi :=
by sorry

end sector_perimeter_l951_95198


namespace log_equation_l951_95120

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation : (log10 5)^2 + log10 2 * log10 50 = 1 := by
  sorry

end log_equation_l951_95120


namespace inequality_and_sum_theorem_l951_95106

def f (x : ℝ) : ℝ := |3*x - 1|

theorem inequality_and_sum_theorem :
  (∀ x : ℝ, f x - f (2 - x) > x ↔ x ∈ Set.Ioo (6/5) 4) ∧
  (∀ a b : ℝ, a + b = 2 → f (a^2) + f (b^2) ≥ 4) := by sorry

end inequality_and_sum_theorem_l951_95106


namespace additional_cost_for_new_requirements_l951_95151

def initial_bales : ℕ := 15
def initial_cost_per_bale : ℕ := 20
def new_cost_per_bale : ℕ := 27

theorem additional_cost_for_new_requirements :
  (initial_bales * 3 * new_cost_per_bale) - (initial_bales * initial_cost_per_bale) = 915 := by
  sorry

end additional_cost_for_new_requirements_l951_95151


namespace frog_jump_distance_l951_95119

/-- The jumping contest between a grasshopper and a frog -/
theorem frog_jump_distance (grasshopper_jump : ℕ) (extra_distance : ℕ) : 
  grasshopper_jump = 17 → extra_distance = 22 → grasshopper_jump + extra_distance = 39 :=
by
  sorry

#check frog_jump_distance

end frog_jump_distance_l951_95119


namespace unique_third_rectangle_dimensions_l951_95156

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Given three rectangles that form a larger rectangle without gaps and overlapping,
    where two of the rectangles are 3 cm × 8 cm and 2 cm × 5 cm,
    prove that there is only one possible set of dimensions for the third rectangle -/
theorem unique_third_rectangle_dimensions (r1 r2 r3 : Rectangle)
  (h1 : r1.width = 3 ∧ r1.height = 8)
  (h2 : r2.width = 2 ∧ r2.height = 5)
  (h_total_area : r1.area + r2.area + r3.area = (r1.width + r2.width + r3.width) * (r1.height + r2.height + r3.height)) :
  r3.width = 4 ∧ r3.height = 1 ∨ r3.width = 1 ∧ r3.height = 4 := by
  sorry

#check unique_third_rectangle_dimensions

end unique_third_rectangle_dimensions_l951_95156


namespace number_of_bags_l951_95133

theorem number_of_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 52) (h2 : cookies_per_bag = 2) :
  total_cookies / cookies_per_bag = 26 := by
  sorry

end number_of_bags_l951_95133


namespace polygon_sides_theorem_l951_95155

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The measure of one interior angle in a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ := 180 * (n - 2) / n

/-- Predicate to check if a pair of numbers satisfies the polygon conditions -/
def satisfies_conditions (x y : ℕ) : Prop :=
  y = x + 10 ∧
  num_diagonals y - num_diagonals x = interior_angle x - 15

theorem polygon_sides_theorem :
  ∀ x y : ℕ, satisfies_conditions x y → (x = 5 ∧ y = 15) ∨ (x = 8 ∧ y = 18) :=
sorry

#check polygon_sides_theorem

end polygon_sides_theorem_l951_95155


namespace number_problem_l951_95124

theorem number_problem : ∃ n : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  n / sum = 2 * diff ∧ n % sum = 50 ∧ n = 220050 := by
  sorry

end number_problem_l951_95124


namespace sum_of_factors_l951_95186

theorem sum_of_factors (a b c : ℤ) : 
  (∀ x, x^2 + 20*x + 96 = (x + a) * (x + b)) →
  (∀ x, x^2 + 18*x + 81 = (x - b) * (x + c)) →
  a + b + c = 30 := by
sorry

end sum_of_factors_l951_95186


namespace jessica_milk_problem_l951_95115

theorem jessica_milk_problem (initial_milk : ℚ) (given_away : ℚ) (remaining : ℚ) :
  initial_milk = 5 →
  given_away = 16 / 3 →
  remaining = initial_milk - given_away →
  remaining = -1 / 3 := by
sorry

end jessica_milk_problem_l951_95115


namespace sum_of_reciprocal_equations_l951_95188

theorem sum_of_reciprocal_equations (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : 1/x - 1/y = -1) : 
  x + y = 5/6 := by
sorry

end sum_of_reciprocal_equations_l951_95188


namespace loss_percent_calculation_l951_95171

theorem loss_percent_calculation (cost_price selling_price : ℝ) : 
  cost_price = 600 → 
  selling_price = 550 → 
  (cost_price - selling_price) / cost_price * 100 = 8.33 := by
sorry

end loss_percent_calculation_l951_95171


namespace countMultiplesIs943_l951_95117

/-- The number of integers between 1 and 3000 (inclusive) that are multiples of 5 or 7 but not multiples of 35 -/
def countMultiples : ℕ := sorry

theorem countMultiplesIs943 : countMultiples = 943 := by sorry

end countMultiplesIs943_l951_95117


namespace second_ruler_alignment_l951_95113

/-- Represents a small ruler in relation to the large ruler -/
structure SmallRuler where
  large_units : ℚ  -- Number of units on the large ruler
  small_units : ℚ  -- Number of units on the small ruler

/-- Represents the set square system with two small rulers and a large ruler -/
structure SetSquare where
  first_ruler : SmallRuler
  second_ruler : SmallRuler
  point_b : ℚ  -- Position of point B on the large ruler

/-- Main theorem statement -/
theorem second_ruler_alignment (s : SetSquare) : 
  s.first_ruler = SmallRuler.mk 11 10 →   -- First ruler divides 11 units into 10
  s.second_ruler = SmallRuler.mk 9 10 →   -- Second ruler divides 9 units into 10
  18 < s.point_b ∧ s.point_b < 19 →       -- Point B is between 18 and 19
  (s.point_b + 3 * s.first_ruler.large_units / s.first_ruler.small_units).floor = 
    (s.point_b + 3 * s.first_ruler.large_units / s.first_ruler.small_units) →  
    -- 3rd unit of first ruler coincides with an integer
  ∃ k : ℕ, (s.point_b + 7 * s.second_ruler.large_units / s.second_ruler.small_units) = ↑k :=
by sorry

end second_ruler_alignment_l951_95113


namespace arithmetic_geometric_sequence_proposition_l951_95111

theorem arithmetic_geometric_sequence_proposition :
  let p : Prop := ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) - a n = a 1 - a 0) → a 1 - a 0 ≠ 0
  let q : Prop := ∀ (g : ℕ → ℝ), (∀ n, g (n + 1) / g n = g 1 / g 0) → g 1 / g 0 ≠ 1
  ¬p ∧ ¬q → (¬p ∨ ¬q) :=
by
  sorry

end arithmetic_geometric_sequence_proposition_l951_95111


namespace thirteen_pow_seven_mod_eight_l951_95118

theorem thirteen_pow_seven_mod_eight : 13^7 % 8 = 5 := by
  sorry

end thirteen_pow_seven_mod_eight_l951_95118


namespace journey_time_proof_l951_95159

theorem journey_time_proof (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 240)
  (h2 : total_time = 5)
  (h3 : speed1 = 40)
  (h4 : speed2 = 60) :
  ∃ (t1 : ℝ), t1 = 3 ∧ 
  ∃ (t2 : ℝ), t1 + t2 = total_time ∧ 
  speed1 * t1 + speed2 * t2 = total_distance :=
by sorry

end journey_time_proof_l951_95159


namespace max_protesters_l951_95100

theorem max_protesters (population : ℕ) (reforms : ℕ) (dislike_per_reform : ℕ) :
  population = 96 →
  reforms = 5 →
  dislike_per_reform = population / 2 →
  (∀ r : ℕ, r ≤ reforms → dislike_per_reform = population / 2) →
  (∃ max_protesters : ℕ,
    max_protesters ≤ population ∧
    max_protesters * (reforms / 2 + 1) ≤ reforms * dislike_per_reform ∧
    ∀ n : ℕ, n ≤ population →
      n * (reforms / 2 + 1) ≤ reforms * dislike_per_reform →
      n ≤ max_protesters) →
  (∃ max_protesters : ℕ, max_protesters = 80) :=
by sorry

end max_protesters_l951_95100


namespace distance_to_fountain_is_30_l951_95126

/-- The distance from Mrs. Hilt's desk to the water fountain -/
def distance_to_fountain : ℕ := sorry

/-- The total distance Mrs. Hilt walks for all trips to the fountain -/
def total_distance : ℕ := 120

/-- The number of times Mrs. Hilt goes to the water fountain -/
def number_of_trips : ℕ := 4

/-- Theorem stating that the distance to the fountain is 30 feet -/
theorem distance_to_fountain_is_30 : 
  distance_to_fountain = total_distance / number_of_trips :=
sorry

end distance_to_fountain_is_30_l951_95126


namespace cos_unique_identifier_l951_95130

open Real

theorem cos_unique_identifier (x : ℝ) (h1 : π / 2 < x) (h2 : x < π) :
  (sin x > 0 ∧ cos x < 0 ∧ cot x < 0) ∧
  (∀ f : ℝ → ℝ, f = sin ∨ f = cos ∨ f = cot →
    (f x < 0 → f = cos)) :=
by sorry

end cos_unique_identifier_l951_95130


namespace asymptote_equation_correct_l951_95121

/-- Represents a hyperbola with equation x^2 - y^2/b^2 = 1 and one focus at (2, 0) -/
structure Hyperbola where
  b : ℝ
  h_b_pos : b > 0

/-- The equation of the asymptotes of the hyperbola -/
def asymptote_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (Real.sqrt 3 * x = y) ∨ (Real.sqrt 3 * x = -y)

/-- Theorem stating that the equation of the asymptotes is correct -/
theorem asymptote_equation_correct (h : Hyperbola) :
  asymptote_equation h = λ x y => (Real.sqrt 3 * x = y) ∨ (Real.sqrt 3 * x = -y) :=
by sorry

end asymptote_equation_correct_l951_95121


namespace harold_marble_sharing_l951_95199

theorem harold_marble_sharing (total_marbles : ℕ) (kept_marbles : ℕ) (marbles_per_friend : ℕ) 
  (h1 : total_marbles = 100)
  (h2 : kept_marbles = 20)
  (h3 : marbles_per_friend = 16)
  : (total_marbles - kept_marbles) / marbles_per_friend = 5 := by
  sorry

end harold_marble_sharing_l951_95199


namespace multiple_remainder_l951_95138

theorem multiple_remainder (x : ℕ) (h : x % 9 = 5) :
  ∃ k : ℕ, k > 0 ∧ (k * x) % 9 = 8 ∧ k = 7 := by
  sorry

end multiple_remainder_l951_95138


namespace chess_tournament_ties_l951_95137

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  num_players : Nat
  points_win : Rat
  points_loss : Rat
  points_tie : Rat
  total_games : Nat
  total_points : Rat
  best_three_points : Rat
  last_nine_points : Rat

/-- The main theorem to be proved -/
theorem chess_tournament_ties (t : ChessTournament) : 
  t.num_players = 14 ∧ 
  t.points_win = 1 ∧ 
  t.points_loss = 0 ∧ 
  t.points_tie = 1/2 ∧ 
  t.total_games = 91 ∧ 
  t.total_points = 91 ∧
  t.best_three_points = t.last_nine_points ∧
  t.best_three_points = 36 →
  ∃ (num_ties : Nat), num_ties = 29 ∧ 
    (∀ (other_num_ties : Nat), other_num_ties > num_ties → 
      ¬(∃ (valid_tournament : ChessTournament), 
        valid_tournament.num_players = 14 ∧
        valid_tournament.points_win = 1 ∧
        valid_tournament.points_loss = 0 ∧
        valid_tournament.points_tie = 1/2 ∧
        valid_tournament.total_games = 91 ∧
        valid_tournament.total_points = 91 ∧
        valid_tournament.best_three_points = valid_tournament.last_nine_points ∧
        valid_tournament.best_three_points = 36)) :=
by
  sorry


end chess_tournament_ties_l951_95137


namespace hyperbola_properties_l951_95181

-- Define the hyperbolas
def C₁ (x y : ℝ) : Prop := x^2/4 - y^2/3 = 1
def C₂ (x y : ℝ) : Prop := x^2/4 - y^2/3 = -1

-- Define focal length
def focal_length (C : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Define foci
def foci (C : (ℝ → ℝ → Prop)) : Set (ℝ × ℝ) := sorry

-- Define asymptotic lines
def asymptotic_lines (C : (ℝ → ℝ → Prop)) : Set (ℝ → ℝ → Prop) := sorry

-- Define eccentricity
def eccentricity (C : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem hyperbola_properties :
  (focal_length C₁ = focal_length C₂) ∧
  (∃ (r : ℝ), ∀ (p : ℝ × ℝ), p ∈ foci C₁ ∪ foci C₂ → p.1^2 + p.2^2 = r^2) ∧
  (asymptotic_lines C₁ = asymptotic_lines C₂) ∧
  (eccentricity C₁ ≠ eccentricity C₂) := by sorry

end hyperbola_properties_l951_95181


namespace bullet_speed_difference_l951_95167

/-- The speed of the horse in feet per second -/
def horse_speed : ℝ := 20

/-- The speed of the bullet in feet per second -/
def bullet_speed : ℝ := 400

/-- The difference in bullet speed when fired in the same direction as the horse's movement
    versus the opposite direction -/
def speed_difference : ℝ := (bullet_speed + horse_speed) - (bullet_speed - horse_speed)

theorem bullet_speed_difference :
  speed_difference = 40 := by sorry

end bullet_speed_difference_l951_95167


namespace f_monotone_implies_a_range_l951_95152

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

-- Define the property of being monotonically increasing
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem f_monotone_implies_a_range (a : ℝ) :
  MonotonicallyIncreasing (f a) → 2 < a ∧ a ≤ 3 :=
by sorry

end f_monotone_implies_a_range_l951_95152


namespace quadratic_factorization_sum_l951_95141

theorem quadratic_factorization_sum : ∃ (a b c d : ℝ),
  (∀ x, x^2 + 23*x + 132 = (x + a) * (x + b)) ∧
  (∀ x, x^2 - 25*x + 168 = (x - c) * (x - d)) ∧
  (a + c + d = 42) := by
  sorry

end quadratic_factorization_sum_l951_95141


namespace quadratic_function_properties_l951_95189

-- Define the quadratic function f(x)
def f (x : ℝ) := 2 * x^2 - 10 * x

-- State the theorem
theorem quadratic_function_properties :
  (∀ x ∈ Set.Icc (-1 : ℝ) 4, f x ≤ 12) ∧  -- maximum value is 12 on [-1,4]
  (∀ x : ℝ, f x < 0 ↔ x ∈ Set.Ioo 0 5) ∧  -- solution set of f(x) < 0 is (0,5)
  (∀ x m : ℝ, m < -5 ∨ m > 1 → f (2 - 2 * Real.cos x) < f (1 - Real.cos x - m)) :=
by sorry

end quadratic_function_properties_l951_95189


namespace compound_oxygen_count_l951_95128

/-- Represents the number of atoms of a particular element in a compound -/
structure AtomCount where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given its atom counts and atomic weights -/
def molecularWeight (atoms : AtomCount) (carbonWeight hydrogenWeight oxygenWeight : ℝ) : ℝ :=
  atoms.carbon * carbonWeight + atoms.hydrogen * hydrogenWeight + atoms.oxygen * oxygenWeight

/-- Theorem stating that a compound with formula C3H6 and molecular weight 58 g/mol contains 1 oxygen atom -/
theorem compound_oxygen_count : 
  ∀ (atoms : AtomCount),
    atoms.carbon = 3 →
    atoms.hydrogen = 6 →
    molecularWeight atoms 12.01 1.008 16.00 = 58 →
    atoms.oxygen = 1 := by
  sorry

end compound_oxygen_count_l951_95128


namespace tshirt_price_is_8_l951_95187

-- Define the prices and quantities
def sweater_price : ℝ := 18
def jacket_original_price : ℝ := 80
def jacket_discount : ℝ := 0.1
def sales_tax : ℝ := 0.05
def num_tshirts : ℕ := 6
def num_sweaters : ℕ := 4
def num_jackets : ℕ := 5
def total_cost : ℝ := 504

-- Define the function to calculate the total cost
def calculate_total_cost (tshirt_price : ℝ) : ℝ :=
  let jacket_price := jacket_original_price * (1 - jacket_discount)
  let subtotal := num_tshirts * tshirt_price + num_sweaters * sweater_price + num_jackets * jacket_price
  subtotal * (1 + sales_tax)

-- Theorem to prove
theorem tshirt_price_is_8 :
  ∃ (tshirt_price : ℝ), calculate_total_cost tshirt_price = total_cost ∧ tshirt_price = 8 :=
sorry

end tshirt_price_is_8_l951_95187


namespace addition_problem_l951_95143

theorem addition_problem (x y : ℕ) :
  (x + y = x + 2000) ∧ (x + y = y + 6) →
  (x = 6 ∧ y = 2000 ∧ x + y = 2006) :=
by sorry

end addition_problem_l951_95143


namespace sin_690_degrees_l951_95165

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end sin_690_degrees_l951_95165


namespace no_three_digit_perfect_square_sum_l951_95105

theorem no_three_digit_perfect_square_sum : 
  ∀ (a b c : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 1 ≤ a → 
  ¬∃ (m : ℕ), m^2 = 111 * (a + b + c) := by
sorry

end no_three_digit_perfect_square_sum_l951_95105


namespace root_sum_product_l951_95125

theorem root_sum_product (p q : ℝ) : 
  (Complex.I * 2 - 1)^2 + p * (Complex.I * 2 - 1) + q = 0 → p + q = 7 := by
  sorry

end root_sum_product_l951_95125


namespace rhombus_perimeter_l951_95123

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 8 * Real.sqrt 41 := by
  sorry

end rhombus_perimeter_l951_95123


namespace consecutive_odd_integers_sum_quadratic_equation_constant_geometric_sequence_ratio_exponential_expression_l951_95140

-- Problem 1
theorem consecutive_odd_integers_sum (k : ℤ) : 
  k + (k + 2) + (k + 4) = 51 → k = 15 := by sorry

-- Problem 2
theorem quadratic_equation_constant (x k a C : ℝ) :
  x^2 + 6*x + k = (x + a)^2 + C → C = 6 := by sorry

-- Problem 3
theorem geometric_sequence_ratio (p q r s R : ℝ) :
  p/q = 2 ∧ q/r = 2 ∧ r/s = 2 ∧ R = p/s → R = 8 := by sorry

-- Problem 4
theorem exponential_expression (n : ℕ) (A : ℝ) :
  A = (3^n * 9^(n+1)) / 27^(n-1) → A = 729 := by sorry

end consecutive_odd_integers_sum_quadratic_equation_constant_geometric_sequence_ratio_exponential_expression_l951_95140


namespace binomial_505_505_equals_1_l951_95168

theorem binomial_505_505_equals_1 : Nat.choose 505 505 = 1 := by
  sorry

end binomial_505_505_equals_1_l951_95168


namespace prism_with_nine_faces_has_fourteen_vertices_l951_95177

/-- A prism is a polyhedron with two congruent polygon bases and rectangular lateral faces. -/
structure Prism where
  num_faces : ℕ
  num_base_sides : ℕ
  num_vertices : ℕ

/-- The number of faces in a prism is related to the number of sides in its base. -/
axiom prism_faces (p : Prism) : p.num_faces = p.num_base_sides + 2

/-- The number of vertices in a prism is twice the number of sides in its base. -/
axiom prism_vertices (p : Prism) : p.num_vertices = 2 * p.num_base_sides

/-- Theorem: A prism with 9 faces has 14 vertices. -/
theorem prism_with_nine_faces_has_fourteen_vertices :
  ∃ (p : Prism), p.num_faces = 9 ∧ p.num_vertices = 14 := by
  sorry


end prism_with_nine_faces_has_fourteen_vertices_l951_95177


namespace line_plane_intersection_l951_95142

/-- The point of intersection between a line and a plane in 3D space. -/
theorem line_plane_intersection
  (A1 A2 A3 A4 : ℝ × ℝ × ℝ)
  (h1 : A1 = (1, 2, -3))
  (h2 : A2 = (1, 0, 1))
  (h3 : A3 = (-2, -1, 6))
  (h4 : A4 = (0, -5, -4)) :
  ∃ P : ℝ × ℝ × ℝ,
    (∃ t : ℝ, P = A4 + t • (A4 - A1)) ∧
    (∃ u v : ℝ, P = A1 + u • (A2 - A1) + v • (A3 - A1)) :=
by sorry


end line_plane_intersection_l951_95142


namespace minimum_tip_percentage_l951_95192

theorem minimum_tip_percentage
  (meal_cost : ℝ)
  (total_paid : ℝ)
  (h_meal_cost : meal_cost = 35.50)
  (h_total_paid : total_paid = 37.275)
  (h_tip_less_than_8 : (total_paid - meal_cost) / meal_cost < 0.08) :
  (total_paid - meal_cost) / meal_cost = 0.05 :=
by sorry

end minimum_tip_percentage_l951_95192


namespace factorization_sum_l951_95161

theorem factorization_sum (A B C D E F G H J K : ℤ) (x y : ℚ) : 
  (125 * x^8 - 2401 * y^8 = (A * x + B * y) * (C * x^4 + D * x * y + E * y^4) * 
                            (F * x + G * y) * (H * x^4 + J * x * y + K * y^4)) →
  A + B + C + D + E + F + G + H + J + K = 102 := by
sorry

end factorization_sum_l951_95161


namespace dice_roll_probability_prob_first_less_than_second_l951_95110

/-- The probability that when rolling two fair six-sided dice, the first roll is less than the second roll -/
theorem dice_roll_probability : ℚ :=
  5/12

/-- A fair six-sided die -/
def fair_die : Finset ℕ := Finset.range 6

/-- The sample space of rolling two dice -/
def two_dice_rolls : Finset (ℕ × ℕ) :=
  fair_die.product fair_die

/-- The event where the first roll is less than the second roll -/
def first_less_than_second : Set (ℕ × ℕ) :=
  {p | p.1 < p.2}

/-- The probability of the event where the first roll is less than the second roll -/
theorem prob_first_less_than_second :
  (two_dice_rolls.filter (λ p => p.1 < p.2)).card / two_dice_rolls.card = dice_roll_probability := by
  sorry

end dice_roll_probability_prob_first_less_than_second_l951_95110


namespace train_journey_duration_l951_95191

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the difference in minutes between two times -/
def timeDifferenceInMinutes (t1 t2 : Time) : Nat :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- Represents the state of clock hands -/
inductive ClockHandState
  | Symmetrical
  | NotSymmetrical

theorem train_journey_duration (stationArrival : Time)
                               (trainDeparture : Time)
                               (destinationArrival : Time)
                               (stationDeparture : Time)
                               (boardingState : ClockHandState)
                               (alightingState : ClockHandState) :
  stationArrival = ⟨8, 0⟩ →
  trainDeparture = ⟨8, 35⟩ →
  destinationArrival = ⟨14, 15⟩ →
  stationDeparture = ⟨15, 0⟩ →
  boardingState = ClockHandState.Symmetrical →
  alightingState = ClockHandState.Symmetrical →
  timeDifferenceInMinutes trainDeparture stationDeparture = 385 :=
by sorry

end train_journey_duration_l951_95191


namespace tree_distance_l951_95172

/-- Given 10 equally spaced trees along a road, with 100 feet between the 1st and 5th tree,
    the distance between the 1st and 10th tree is 225 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 10) (h2 : d = 100) :
  let space := d / 4
  (n - 1) * space = 225 :=
by sorry

end tree_distance_l951_95172


namespace inverse_function_point_sum_l951_95154

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverses of each other
axiom inverse_relation : ∀ x, f_inv (f x) = x

-- Given condition: (2, 4) is on the graph of y = f(x)/3
axiom point_on_f : f 2 = 12

-- Theorem to prove
theorem inverse_function_point_sum :
  ∃ a b : ℝ, f_inv a = 3 * b ∧ a + b = 38 / 3 :=
sorry

end inverse_function_point_sum_l951_95154


namespace books_checked_out_after_returns_l951_95173

-- Define the initial state
def initial_books : ℕ := 15
def initial_movies : ℕ := 6

-- Define the number of books returned
def books_returned : ℕ := 8

-- Define the fraction of movies returned
def movie_return_fraction : ℚ := 1 / 3

-- Define the final total of items
def final_total : ℕ := 20

-- Theorem to prove
theorem books_checked_out_after_returns (checked_out : ℕ) : 
  checked_out = 9 :=
by
  sorry


end books_checked_out_after_returns_l951_95173


namespace number_with_specific_remainders_l951_95144

theorem number_with_specific_remainders : ∃ (N : ℕ), N % 13 = 11 ∧ N % 17 = 9 ∧ N = 141 := by
  sorry

end number_with_specific_remainders_l951_95144


namespace quadratic_function_proof_l951_95127

/-- Quadratic function passing through specific points with given minimum --/
theorem quadratic_function_proof (a h k : ℝ) :
  a ≠ 0 →
  a * (1 - h)^2 + k = 3 →
  a * (3 - h)^2 + k = 3 →
  k = -1 →
  a = 4 ∧ h = 2 := by
  sorry

end quadratic_function_proof_l951_95127


namespace military_unit_march_speeds_l951_95190

/-- Proves that given the conditions of the military unit's march, the average speeds on the first and second days are 12 km/h and 10 km/h respectively. -/
theorem military_unit_march_speeds :
  ∀ (speed_day1 speed_day2 : ℝ),
    4 * speed_day1 + 5 * speed_day2 = 98 →
    4 * speed_day1 = 5 * speed_day2 - 2 →
    speed_day1 = 12 ∧ speed_day2 = 10 := by
  sorry

end military_unit_march_speeds_l951_95190


namespace total_swordfish_catch_l951_95184

/-- The number of swordfish Shelly catches per trip -/
def shelly_catch : ℕ := 5 - 2

/-- The number of swordfish Sam catches per trip -/
def sam_catch : ℕ := shelly_catch - 1

/-- The number of fishing trips -/
def num_trips : ℕ := 5

/-- The total number of swordfish caught by Shelly and Sam -/
def total_catch : ℕ := (shelly_catch + sam_catch) * num_trips

theorem total_swordfish_catch : total_catch = 25 := by
  sorry

end total_swordfish_catch_l951_95184


namespace equilateral_iff_complex_equation_l951_95158

/-- A primitive cube root of unity -/
noncomputable def w : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 3))

/-- Definition of an equilateral triangle in the complex plane -/
def is_equilateral (z₁ z₂ z₃ : ℂ) : Prop :=
  Complex.abs (z₂ - z₁) = Complex.abs (z₃ - z₂) ∧
  Complex.abs (z₃ - z₂) = Complex.abs (z₁ - z₃)

/-- Definition of counterclockwise orientation -/
def is_counterclockwise (z₁ z₂ z₃ : ℂ) : Prop :=
  (z₂ - z₁).arg < (z₃ - z₁).arg ∧ (z₃ - z₁).arg < (z₂ - z₁).arg + Real.pi

/-- Theorem: A triangle is equilateral iff it satisfies the given complex equation -/
theorem equilateral_iff_complex_equation (z₁ z₂ z₃ : ℂ) :
  is_counterclockwise z₁ z₂ z₃ →
  is_equilateral z₁ z₂ z₃ ↔ z₁ + w * z₂ + w^2 * z₃ = 0 :=
by sorry

end equilateral_iff_complex_equation_l951_95158


namespace sandy_average_book_price_l951_95160

/-- The average price of books Sandy bought given the conditions -/
def average_price_per_book (books_shop1 books_shop2 : ℕ) (price_shop1 price_shop2 : ℚ) : ℚ :=
  (price_shop1 + price_shop2) / (books_shop1 + books_shop2)

/-- Theorem stating that the average price Sandy paid per book is $18 -/
theorem sandy_average_book_price :
  let books_shop1 : ℕ := 65
  let books_shop2 : ℕ := 55
  let price_shop1 : ℚ := 1280
  let price_shop2 : ℚ := 880
  average_price_per_book books_shop1 books_shop2 price_shop1 price_shop2 = 18 := by
  sorry


end sandy_average_book_price_l951_95160


namespace ladder_distance_l951_95114

theorem ladder_distance (angle : Real) (length : Real) (distance : Real) : 
  angle = 60 * π / 180 →
  length = 19 →
  distance = length * Real.cos angle →
  distance = 9.5 := by
  sorry

end ladder_distance_l951_95114


namespace total_dots_is_89_l951_95150

/-- The number of ladybugs caught on Monday -/
def monday_ladybugs : ℕ := 8

/-- The number of dots on each ladybug caught on Monday -/
def monday_dots_per_ladybug : ℕ := 6

/-- The number of ladybugs caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The number of ladybugs caught on Wednesday -/
def wednesday_ladybugs : ℕ := 4

/-- The number of dots on each ladybug caught on Tuesday -/
def tuesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 1

/-- The number of dots on each ladybug caught on Wednesday -/
def wednesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 2

/-- The total number of dots on all ladybugs caught over three days -/
def total_dots : ℕ :=
  monday_ladybugs * monday_dots_per_ladybug +
  tuesday_ladybugs * tuesday_dots_per_ladybug +
  wednesday_ladybugs * wednesday_dots_per_ladybug

theorem total_dots_is_89 : total_dots = 89 := by
  sorry

end total_dots_is_89_l951_95150


namespace real_roots_range_l951_95185

theorem real_roots_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + 4*a*x - 4*a + 3 = 0) ∨ 
  (∃ x : ℝ, x^2 + (a-1)*x + a^2 = 0) ∨ 
  (∃ x : ℝ, x^2 + 2*a*x - 2*a = 0) → 
  a ≤ -3/2 ∨ a ≥ -1 := by
sorry

end real_roots_range_l951_95185


namespace intersection_sum_l951_95129

theorem intersection_sum (c d : ℝ) : 
  (∀ x y : ℝ, x = (1/3) * y + c ∧ y = (1/3) * x + d → (x = 3 ∧ y = 3)) → 
  c + d = 4 := by
sorry

end intersection_sum_l951_95129


namespace characterize_function_l951_95166

theorem characterize_function (f : ℤ → ℤ) :
  (∀ x y z : ℤ, x + y + z = 0 → f x + f y + f z = x * y * z) →
  ∃ c : ℤ, ∀ x : ℤ, f x = (x^3 - x) / 3 + c * x :=
sorry

end characterize_function_l951_95166


namespace distracted_scientist_waiting_time_l951_95116

/-- The average waiting time for the first bite given the conditions of the distracted scientist problem -/
theorem distracted_scientist_waiting_time 
  (first_rod_bites : ℝ) 
  (second_rod_bites : ℝ) 
  (total_bites : ℝ) 
  (time_interval : ℝ) 
  (h1 : first_rod_bites = 3) 
  (h2 : second_rod_bites = 2) 
  (h3 : total_bites = first_rod_bites + second_rod_bites) 
  (h4 : time_interval = 6) : 
  (time_interval / total_bites) = 1.2 := by
  sorry

end distracted_scientist_waiting_time_l951_95116


namespace largest_element_of_A_l951_95147

def A : Set ℝ := {x | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 2020 ∧ x = n ^ (1 / n : ℝ)}

theorem largest_element_of_A : ∀ x ∈ A, x ≤ 3 ^ (1 / 3 : ℝ) :=
sorry

end largest_element_of_A_l951_95147


namespace min_participants_is_61_l951_95104

/-- Represents the number of participants in the race. -/
def n : ℕ := 61

/-- Represents the number of people who finished before Andrei. -/
def x : ℕ := 20

/-- Represents the number of people who finished before Dima. -/
def y : ℕ := 15

/-- Represents the number of people who finished before Lenya. -/
def z : ℕ := 12

/-- Theorem stating that 61 is the minimum number of participants satisfying the given conditions. -/
theorem min_participants_is_61 :
  (x + 1 + 2 * x = n) ∧
  (y + 1 + 3 * y = n) ∧
  (z + 1 + 4 * z = n) ∧
  (∀ m : ℕ, m < n → ¬((m - 1) % 3 = 0 ∧ (m - 1) % 4 = 0 ∧ (m - 1) % 5 = 0)) :=
by sorry

#check min_participants_is_61

end min_participants_is_61_l951_95104


namespace rectangle_width_length_ratio_l951_95149

theorem rectangle_width_length_ratio 
  (w : ℝ) 
  (h1 : w > 0) 
  (h2 : 2 * (w + 10) = 30) : 
  w / 10 = 1 / 2 := by
sorry

end rectangle_width_length_ratio_l951_95149


namespace min_K_is_two_l951_95157

-- Define the function f
def f (x : ℝ) : ℝ := 2 - x - x^2

-- Define the property that f_K(x) = f(x) for all x ≥ 0
def f_K_equals_f (K : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x ≤ K

-- Theorem statement
theorem min_K_is_two :
  ∃ K : ℝ, (f_K_equals_f K ∧ ∀ K' : ℝ, K' < K → ¬f_K_equals_f K') ∧ K = 2 :=
sorry

end min_K_is_two_l951_95157


namespace point_p_coordinates_l951_95101

/-- Given points A(2, 3) and B(4, -3), if a point P satisfies |AP| = 3/2 |PB|, 
    then P has coordinates (16/5, 0). -/
theorem point_p_coordinates (A B P : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (4, -3) → 
  ‖A - P‖ = (3/2) * ‖P - B‖ → 
  P = (16/5, 0) := by
  sorry

end point_p_coordinates_l951_95101


namespace x_powers_sum_l951_95153

theorem x_powers_sum (x : ℝ) (h : x + 1/x = 10) : 
  x^2 + 1/x^2 = 98 ∧ x^3 + 1/x^3 = 970 := by
  sorry

end x_powers_sum_l951_95153


namespace math_majors_consecutive_probability_l951_95146

/-- The number of people sitting around the table -/
def total_people : ℕ := 12

/-- The number of math majors -/
def math_majors : ℕ := 5

/-- The number of physics majors -/
def physics_majors : ℕ := 4

/-- The number of biology majors -/
def biology_majors : ℕ := 3

/-- The probability of all math majors sitting in consecutive seats -/
def prob_consecutive_math : ℚ := 2 / 55

theorem math_majors_consecutive_probability :
  (total_people = math_majors + physics_majors + biology_majors) →
  (prob_consecutive_math = 2 / 55) := by
  sorry

end math_majors_consecutive_probability_l951_95146


namespace solve_jewelry_store_problem_l951_95108

/-- Represents the jewelry store inventory problem --/
def jewelry_store_problem (necklace_capacity ring_capacity bracelet_capacity : ℕ)
  (current_rings current_bracelets : ℕ)
  (price_necklace price_ring price_bracelet : ℕ)
  (total_cost : ℕ) : Prop :=
  let rings_needed := ring_capacity - current_rings
  let bracelets_needed := bracelet_capacity - current_bracelets
  let necklaces_on_stand := necklace_capacity - 
    ((total_cost - price_ring * rings_needed - price_bracelet * bracelets_needed) / price_necklace)
  necklaces_on_stand = 5

/-- The main theorem stating the solution to the jewelry store problem --/
theorem solve_jewelry_store_problem :
  jewelry_store_problem 12 30 15 18 8 4 10 5 183 := by
  sorry

end solve_jewelry_store_problem_l951_95108


namespace cryptarithm_no_solution_l951_95194

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a mapping from characters to digits -/
def DigitAssignment := Char → Digit

/-- Checks if all characters in a string are mapped to unique digits -/
def all_unique (s : String) (assignment : DigitAssignment) : Prop :=
  ∀ c₁ c₂, c₁ ∈ s.data → c₂ ∈ s.data → c₁ ≠ c₂ → assignment c₁ ≠ assignment c₂

/-- Converts a string to a number using the given digit assignment -/
def to_number (s : String) (assignment : DigitAssignment) : ℕ :=
  s.foldl (fun acc c => 10 * acc + (assignment c).val) 0

/-- The main theorem stating that the cryptarithm has no solution -/
theorem cryptarithm_no_solution :
  ¬ ∃ (assignment : DigitAssignment),
    all_unique "DONAKLENVG" assignment ∧
    to_number "DON" assignment + to_number "OKA" assignment +
    to_number "LENA" assignment + to_number "VOLGA" assignment =
    to_number "ANGARA" assignment :=
by sorry


end cryptarithm_no_solution_l951_95194


namespace candidate_votes_proof_l951_95109

theorem candidate_votes_proof (total_votes : ℕ) (invalid_percentage : ℚ) (candidate_percentage : ℚ) :
  total_votes = 560000 →
  invalid_percentage = 15 / 100 →
  candidate_percentage = 80 / 100 →
  ⌊(1 - invalid_percentage) * candidate_percentage * total_votes⌋ = 380800 := by
  sorry

end candidate_votes_proof_l951_95109


namespace diff_eq_linear_solution_l951_95139

/-- The differential equation y'' = 0 has a general solution of the form y = C₁x + C₂,
    where C₁ and C₂ are arbitrary constants. -/
theorem diff_eq_linear_solution (x : ℝ) :
  ∃ (y : ℝ → ℝ) (C₁ C₂ : ℝ), (∀ x, (deriv^[2] y) x = 0) ∧ (∀ x, y x = C₁ * x + C₂) := by
  sorry

end diff_eq_linear_solution_l951_95139


namespace segments_in_proportion_l951_95164

/-- Four line segments are in proportion if the product of the outer segments
    equals the product of the inner segments -/
def are_in_proportion (a b c d : ℝ) : Prop :=
  a * d = b * c

/-- The set of line segments (4, 8, 5, 10) -/
def segment_set : Vector ℝ 4 := ⟨[4, 8, 5, 10], rfl⟩

/-- Theorem: The set of line segments (4, 8, 5, 10) is in proportion -/
theorem segments_in_proportion :
  are_in_proportion (segment_set.get 0) (segment_set.get 1) (segment_set.get 2) (segment_set.get 3) :=
by
  sorry

end segments_in_proportion_l951_95164


namespace consecutive_integers_around_sqrt_17_l951_95135

theorem consecutive_integers_around_sqrt_17 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 17) → (Real.sqrt 17 < b) → (a + b = 9) := by
  sorry

end consecutive_integers_around_sqrt_17_l951_95135
