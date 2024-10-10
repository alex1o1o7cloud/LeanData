import Mathlib

namespace committee_count_l1642_164253

/-- The number of ways to choose k elements from a set of n elements -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of students in the group -/
def total_students : ℕ := 8

/-- The number of students in each committee -/
def committee_size : ℕ := 3

/-- The number of different committees that can be formed -/
def num_committees : ℕ := binomial total_students committee_size

theorem committee_count : num_committees = 56 := by
  sorry

end committee_count_l1642_164253


namespace circle_diameter_from_area_l1642_164229

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) :
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end circle_diameter_from_area_l1642_164229


namespace unemployment_rate_calculation_l1642_164263

theorem unemployment_rate_calculation (previous_employment_rate previous_unemployment_rate : ℝ)
  (h1 : previous_employment_rate + previous_unemployment_rate = 100)
  (h2 : previous_employment_rate > 0)
  (h3 : previous_unemployment_rate > 0) :
  let new_employment_rate := 0.85 * previous_employment_rate
  let new_unemployment_rate := 1.1 * previous_unemployment_rate
  new_unemployment_rate = 66 :=
by
  sorry

#check unemployment_rate_calculation

end unemployment_rate_calculation_l1642_164263


namespace ingrids_tax_rate_l1642_164239

theorem ingrids_tax_rate 
  (john_tax_rate : ℝ)
  (john_income : ℝ)
  (ingrid_income : ℝ)
  (combined_tax_rate : ℝ)
  (h1 : john_tax_rate = 0.30)
  (h2 : john_income = 58000)
  (h3 : ingrid_income = 72000)
  (h4 : combined_tax_rate = 0.3554)
  : (combined_tax_rate * (john_income + ingrid_income) - john_tax_rate * john_income) / ingrid_income = 0.40 := by
  sorry

end ingrids_tax_rate_l1642_164239


namespace bryans_bookshelves_l1642_164216

/-- Given that Bryan has 56 books in each bookshelf and 504 books in total,
    prove that he has 9 bookshelves. -/
theorem bryans_bookshelves (books_per_shelf : ℕ) (total_books : ℕ) 
    (h1 : books_per_shelf = 56) (h2 : total_books = 504) :
    total_books / books_per_shelf = 9 := by
  sorry

end bryans_bookshelves_l1642_164216


namespace bike_cost_l1642_164214

/-- The cost of Jenn's bike given her savings in quarters and leftover money -/
theorem bike_cost (num_jars : ℕ) (quarters_per_jar : ℕ) (quarter_value : ℚ) (leftover : ℕ) : 
  num_jars = 5 →
  quarters_per_jar = 160 →
  quarter_value = 1/4 →
  leftover = 20 →
  (num_jars * quarters_per_jar : ℕ) * quarter_value - leftover = 200 := by
  sorry

end bike_cost_l1642_164214


namespace certain_event_draw_two_white_l1642_164252

/-- A box containing only white balls -/
structure WhiteBallBox where
  num_balls : ℕ

/-- The probability of drawing two white balls from a box -/
def prob_draw_two_white (box : WhiteBallBox) : ℚ :=
  if box.num_balls ≥ 2 then 1 else 0

/-- Theorem: Drawing 2 white balls from a box with 5 white balls is a certain event -/
theorem certain_event_draw_two_white :
  prob_draw_two_white ⟨5⟩ = 1 := by sorry

end certain_event_draw_two_white_l1642_164252


namespace events_B_C_complementary_l1642_164248

-- Define the sample space (faces of the die)
def Die : Type := Fin 6

-- Define event B
def eventB (x : Die) : Prop := x.val + 1 ≤ 3

-- Define event C
def eventC (x : Die) : Prop := x.val + 1 ≥ 4

-- Theorem statement
theorem events_B_C_complementary :
  ∀ (x : Die), (eventB x ∧ ¬eventC x) ∨ (¬eventB x ∧ eventC x) :=
by sorry

end events_B_C_complementary_l1642_164248


namespace common_sum_is_negative_fifteen_l1642_164282

def is_valid_arrangement (arr : Matrix (Fin 5) (Fin 5) ℤ) : Prop :=
  ∀ i j, -15 ≤ arr i j ∧ arr i j ≤ 9

def row_sum (arr : Matrix (Fin 5) (Fin 5) ℤ) (i : Fin 5) : ℤ :=
  (Finset.range 5).sum (λ j => arr i j)

def col_sum (arr : Matrix (Fin 5) (Fin 5) ℤ) (j : Fin 5) : ℤ :=
  (Finset.range 5).sum (λ i => arr i j)

def main_diagonal_sum (arr : Matrix (Fin 5) (Fin 5) ℤ) : ℤ :=
  (Finset.range 5).sum (λ i => arr i i)

def anti_diagonal_sum (arr : Matrix (Fin 5) (Fin 5) ℤ) : ℤ :=
  (Finset.range 5).sum (λ i => arr i (4 - i))

def all_sums_equal (arr : Matrix (Fin 5) (Fin 5) ℤ) : Prop :=
  ∃ s, (∀ i, row_sum arr i = s) ∧
       (∀ j, col_sum arr j = s) ∧
       (main_diagonal_sum arr = s) ∧
       (anti_diagonal_sum arr = s)

theorem common_sum_is_negative_fifteen
  (arr : Matrix (Fin 5) (Fin 5) ℤ)
  (h1 : is_valid_arrangement arr)
  (h2 : all_sums_equal arr) :
  ∃ s, s = -15 ∧ all_sums_equal arr ∧ (∀ i j, row_sum arr i = s ∧ col_sum arr j = s) :=
sorry

end common_sum_is_negative_fifteen_l1642_164282


namespace arithmetic_sequence_common_difference_l1642_164210

theorem arithmetic_sequence_common_difference :
  ∀ (a d : ℚ) (n : ℕ),
    a = 2 →
    a + (n - 1) * d = 20 →
    n * (a + (a + (n - 1) * d)) / 2 = 132 →
    d = 18 / 11 := by
  sorry

end arithmetic_sequence_common_difference_l1642_164210


namespace equation_solutions_l1642_164205

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁^4 + (3 - x₁)^4 = 82) ∧ 
    (x₂^4 + (3 - x₂)^4 = 82) ∧ 
    (x₁ = 1.5 + Real.sqrt 1.375) ∧ 
    (x₂ = 1.5 - Real.sqrt 1.375) ∧
    (∀ x : ℝ, x^4 + (3 - x)^4 = 82 → x = x₁ ∨ x = x₂) :=
by sorry

end equation_solutions_l1642_164205


namespace k_range_l1642_164295

def p (k : ℝ) : Prop := ∀ x y : ℝ, x < y → k * x + 1 < k * y + 1

def q (k : ℝ) : Prop := ∃ x : ℝ, x^2 + (2*k - 3)*x + 1 = 0

theorem k_range (k : ℝ) : 
  (¬(p k ∧ q k) ∧ (p k ∨ q k)) → 
  (k ≤ 0 ∨ (1/2 < k ∧ k < 5/2)) :=
sorry

end k_range_l1642_164295


namespace complex_simplification_l1642_164209

theorem complex_simplification :
  (7 - 4 * Complex.I) - (2 + 6 * Complex.I) + (3 - 3 * Complex.I) = 8 - 13 * Complex.I :=
by sorry

end complex_simplification_l1642_164209


namespace solution_set_f_greater_than_x_range_of_x_for_inequality_l1642_164247

-- Define the function f
def f (x : ℝ) := |2*x - 1| - |x + 1|

-- Theorem for part I
theorem solution_set_f_greater_than_x :
  {x : ℝ | f x > x} = {x : ℝ | x < 0} := by sorry

-- Theorem for part II
theorem range_of_x_for_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x : ℝ, (1/a + 4/b) ≥ f x) →
  (∀ x : ℝ, f x ≤ 9) →
  (∀ x : ℝ, -7 ≤ x ∧ x ≤ 11) := by sorry

end solution_set_f_greater_than_x_range_of_x_for_inequality_l1642_164247


namespace cos_two_alpha_value_l1642_164207

theorem cos_two_alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 2/3) :
  Real.cos (2 * α) = 2 * Real.sqrt 14 / 9 ∨ Real.cos (2 * α) = -2 * Real.sqrt 14 / 9 := by
  sorry

end cos_two_alpha_value_l1642_164207


namespace trigonometric_expression_equality_l1642_164212

theorem trigonometric_expression_equality (θ : Real) (h : Real.tan θ = 3) :
  (1 + Real.cos θ) / Real.sin θ + Real.sin θ / (1 - Real.cos θ) = 17 * (Real.sqrt 10 + 1) / 24 := by
  sorry

end trigonometric_expression_equality_l1642_164212


namespace stevens_weight_l1642_164284

/-- Given that Danny weighs 40 kg and Steven weighs 20% more than Danny, 
    prove that Steven's weight is 48 kg. -/
theorem stevens_weight (danny_weight : ℝ) (steven_weight : ℝ) 
    (h1 : danny_weight = 40)
    (h2 : steven_weight = danny_weight * 1.2) : 
  steven_weight = 48 := by
  sorry

end stevens_weight_l1642_164284


namespace dime_difference_is_243_l1642_164258

/-- Represents the types of coins in the piggy bank --/
inductive Coin
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- The value of each coin type in cents --/
def coinValue : Coin → Nat
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- A configuration of coins in the piggy bank --/
structure CoinConfiguration where
  nickels : Nat
  dimes : Nat
  quarters : Nat
  halfDollars : Nat

/-- The total number of coins in a configuration --/
def totalCoins (c : CoinConfiguration) : Nat :=
  c.nickels + c.dimes + c.quarters + c.halfDollars

/-- The total value of coins in a configuration in cents --/
def totalValue (c : CoinConfiguration) : Nat :=
  c.nickels * coinValue Coin.Nickel +
  c.dimes * coinValue Coin.Dime +
  c.quarters * coinValue Coin.Quarter +
  c.halfDollars * coinValue Coin.HalfDollar

/-- Predicate to check if a configuration is valid --/
def isValidConfiguration (c : CoinConfiguration) : Prop :=
  totalCoins c = 150 ∧ totalValue c = 2000

/-- The maximum number of dimes possible in a valid configuration --/
def maxDimes : Nat :=
  250

/-- The minimum number of dimes possible in a valid configuration --/
def minDimes : Nat :=
  7

theorem dime_difference_is_243 :
  ∃ (cMax cMin : CoinConfiguration),
    isValidConfiguration cMax ∧
    isValidConfiguration cMin ∧
    cMax.dimes = maxDimes ∧
    cMin.dimes = minDimes ∧
    maxDimes - minDimes = 243 :=
  sorry

end dime_difference_is_243_l1642_164258


namespace min_value_of_f_l1642_164261

def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by sorry

end min_value_of_f_l1642_164261


namespace max_value_operation_l1642_164299

theorem max_value_operation (n : ℕ) : 
  (10 ≤ n ∧ n ≤ 99) → 4 * (300 - n) ≤ 1160 :=
by sorry

end max_value_operation_l1642_164299


namespace square_difference_l1642_164293

theorem square_difference (x y z : ℝ) 
  (sum_xy : x + y = 10)
  (diff_xy : x - y = 8)
  (sum_yz : y + z = 15) :
  x^2 - z^2 = -115 := by
sorry

end square_difference_l1642_164293


namespace burger_lovers_l1642_164265

theorem burger_lovers (total : ℕ) (pizza_lovers : ℕ) (both_lovers : ℕ) 
    (h1 : total = 200)
    (h2 : pizza_lovers = 125)
    (h3 : both_lovers = 40)
    (h4 : both_lovers ≤ pizza_lovers)
    (h5 : pizza_lovers ≤ total) :
  total - (pizza_lovers - both_lovers) - both_lovers = 115 := by
  sorry

end burger_lovers_l1642_164265


namespace negation_of_universal_proposition_l1642_164291

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1 → x^2 ≥ 3) ↔ (∃ x : ℝ, x > 1 ∧ x^2 < 3) := by sorry

end negation_of_universal_proposition_l1642_164291


namespace amount_after_two_years_l1642_164264

-- Define the initial amount
def initial_amount : ℚ := 64000

-- Define the annual increase rate
def annual_rate : ℚ := 1 / 8

-- Define the time period in years
def years : ℕ := 2

-- Define the function to calculate the amount after n years
def amount_after_years (initial : ℚ) (rate : ℚ) (n : ℕ) : ℚ :=
  initial * (1 + rate) ^ n

-- Theorem statement
theorem amount_after_two_years :
  amount_after_years initial_amount annual_rate years = 81000 := by
  sorry

end amount_after_two_years_l1642_164264


namespace sum_of_reciprocals_l1642_164227

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) : 
  1 / x + 1 / y = 8 / 75 := by
sorry

end sum_of_reciprocals_l1642_164227


namespace cartesian_to_polar_equivalence_l1642_164269

theorem cartesian_to_polar_equivalence :
  ∀ (x y ρ θ : ℝ),
  x = 2 ∧ y = -2 →
  ρ = Real.sqrt (x^2 + y^2) →
  θ = Real.arctan (y / x) →
  (ρ = 2 * Real.sqrt 2 ∧ θ = -π/4) := by
  sorry

end cartesian_to_polar_equivalence_l1642_164269


namespace students_not_eating_lunch_proof_l1642_164202

def students_not_eating_lunch (total_students cafeteria_students : ℕ) : ℕ :=
  total_students - (cafeteria_students + 3 * cafeteria_students)

theorem students_not_eating_lunch_proof :
  students_not_eating_lunch 60 10 = 20 := by
  sorry

end students_not_eating_lunch_proof_l1642_164202


namespace number_problem_l1642_164259

theorem number_problem (N : ℝ) :
  (4 / 5) * N = (N / (4 / 5)) - 27 → N = 60 := by
  sorry

end number_problem_l1642_164259


namespace system_solution_exists_l1642_164208

theorem system_solution_exists : ∃ (x y : ℝ), 
  0 ≤ x ∧ x ≤ 6 ∧ 
  0 ≤ y ∧ y ≤ 4 ∧ 
  x + 2 * Real.sqrt y = 6 ∧ 
  Real.sqrt x + y = 4 ∧ 
  abs (x - 2.985) < 0.001 ∧ 
  abs (y - 2.272) < 0.001 := by
  sorry

end system_solution_exists_l1642_164208


namespace max_students_distribution_l1642_164296

theorem max_students_distribution (pens pencils : ℕ) 
  (h_pens : pens = 1340) (h_pencils : pencils = 1280) : 
  Nat.gcd pens pencils = 20 := by
sorry

end max_students_distribution_l1642_164296


namespace inequality_proof_l1642_164219

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a / b + b / c + c / a)^2 ≥ 3 * (a / c + c / b + b / a) := by
  sorry

end inequality_proof_l1642_164219


namespace association_member_condition_l1642_164238

/-- Represents a member of the association -/
structure Member where
  number : Nat
  country : Fin 6

/-- The set of all members in the association -/
def Association := Fin 1978 → Member

/-- Predicate to check if a member's number satisfies the condition -/
def SatisfiesCondition (assoc : Association) (m : Member) : Prop :=
  ∃ (a b : Member),
    a.country = m.country ∧ b.country = m.country ∧
    ((a.number + b.number = m.number) ∨ (2 * a.number = m.number))

/-- Main theorem -/
theorem association_member_condition (assoc : Association) :
  ∃ (m : Member), m ∈ Set.range assoc ∧ SatisfiesCondition assoc m := by
  sorry


end association_member_condition_l1642_164238


namespace absolute_value_inequality_l1642_164273

theorem absolute_value_inequality (m : ℝ) :
  (∀ x : ℝ, |x - 3| - |x - 1| > m) → m < -2 := by
  sorry

end absolute_value_inequality_l1642_164273


namespace quadratic_root_transformation_l1642_164223

theorem quadratic_root_transformation (r s : ℝ) : 
  (3 * r^2 + 4 * r + 2 = 0) →
  (3 * s^2 + 4 * s + 2 = 0) →
  (r + s = -4/3) →
  (r * s = 2/3) →
  ∃ q : ℝ, r^3 + s^3 = -16/27 ∧ r^3 * s^3 = q ∧ 
    ∀ x : ℝ, x^2 + (16/27) * x + q = 0 ↔ (x = r^3 ∨ x = s^3) :=
sorry

end quadratic_root_transformation_l1642_164223


namespace car_hire_cost_for_b_l1642_164213

theorem car_hire_cost_for_b (total_cost : ℚ) (time_a time_b time_c : ℚ) : 
  total_cost = 520 →
  time_a = 7 →
  time_b = 8 →
  time_c = 11 →
  time_b / (time_a + time_b + time_c) * total_cost = 160 := by
  sorry

end car_hire_cost_for_b_l1642_164213


namespace savings_difference_is_75_cents_l1642_164249

/-- The price of the book in dollars -/
def book_price : ℚ := 30

/-- The fixed discount amount in dollars -/
def fixed_discount : ℚ := 5

/-- The percentage discount as a decimal -/
def percent_discount : ℚ := 0.15

/-- The cost after applying the fixed discount first, then the percentage discount -/
def cost_fixed_first : ℚ := (book_price - fixed_discount) * (1 - percent_discount)

/-- The cost after applying the percentage discount first, then the fixed discount -/
def cost_percent_first : ℚ := book_price * (1 - percent_discount) - fixed_discount

/-- The difference in savings between the two discount sequences, in cents -/
def savings_difference : ℚ := (cost_fixed_first - cost_percent_first) * 100

theorem savings_difference_is_75_cents : savings_difference = 75 := by
  sorry

end savings_difference_is_75_cents_l1642_164249


namespace project_distribution_count_l1642_164240

/-- The number of ways to distribute 8 distinct projects among 4 companies -/
def distribute_projects : ℕ :=
  Nat.choose 8 3 * Nat.choose 5 1 * Nat.choose 4 2 * Nat.choose 2 2

/-- Theorem stating that the number of ways to distribute the projects is 1680 -/
theorem project_distribution_count : distribute_projects = 1680 := by
  sorry

end project_distribution_count_l1642_164240


namespace coin_collection_l1642_164224

theorem coin_collection (nickels dimes quarters : ℕ) (total_value : ℕ) : 
  nickels = dimes →
  quarters = 2 * nickels →
  total_value = 1950 →
  5 * nickels + 10 * dimes + 25 * quarters = total_value →
  nickels = 30 := by
sorry

end coin_collection_l1642_164224


namespace equation_solution_l1642_164262

theorem equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ - 3) * (x₁ + 1) = 5 ∧ 
  (x₂ - 3) * (x₂ + 1) = 5 ∧ 
  x₁ = 4 ∧ 
  x₂ = -2 := by
sorry

end equation_solution_l1642_164262


namespace consecutive_squares_remainder_l1642_164255

theorem consecutive_squares_remainder (n : ℕ) : 
  (n - 1)^2 + n^2 + (n + 1)^2 ≡ 2 [MOD 3] :=
by sorry

end consecutive_squares_remainder_l1642_164255


namespace no_prime_10101_base_n_l1642_164267

theorem no_prime_10101_base_n : ¬ ∃ (n : ℕ), n ≥ 2 ∧ Nat.Prime (n^4 + n^2 + 1) := by
  sorry

end no_prime_10101_base_n_l1642_164267


namespace quinns_reading_challenge_l1642_164274

/-- Proves the number of weeks Quinn needs to participate in the reading challenge -/
theorem quinns_reading_challenge
  (books_per_donut : ℕ)
  (books_per_week : ℕ)
  (target_donuts : ℕ)
  (h1 : books_per_donut = 5)
  (h2 : books_per_week = 2)
  (h3 : target_donuts = 4) :
  (target_donuts * books_per_donut) / books_per_week = 10 :=
by sorry

end quinns_reading_challenge_l1642_164274


namespace train_length_l1642_164235

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  speed = 72 * 1000 / 3600 → 
  platform_length = 300 → 
  crossing_time = 26 → 
  speed * crossing_time - platform_length = 220 :=
by sorry

end train_length_l1642_164235


namespace smallest_perfect_square_divisible_by_2_and_5_l1642_164231

theorem smallest_perfect_square_divisible_by_2_and_5 : ∃ n : ℕ, 
  n > 0 ∧ 
  (∃ m : ℕ, n = m ^ 2) ∧ 
  n % 2 = 0 ∧ 
  n % 5 = 0 ∧
  (∀ k : ℕ, k > 0 → (∃ m : ℕ, k = m ^ 2) → k % 2 = 0 → k % 5 = 0 → k ≥ n) ∧
  n = 100 := by
sorry

end smallest_perfect_square_divisible_by_2_and_5_l1642_164231


namespace shortest_dividing_line_l1642_164257

-- Define a circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a broken line
def BrokenLine := List (ℝ × ℝ)

-- Function to calculate the length of a broken line
def length (bl : BrokenLine) : ℝ := sorry

-- Function to check if a broken line divides the circle into two equal parts
def divides_equally (bl : BrokenLine) (c : Circle) : Prop := sorry

-- Define the diameter of a circle
def diameter (c : Circle) : ℝ := 2

-- Theorem statement
theorem shortest_dividing_line (c : Circle) (bl : BrokenLine) :
  divides_equally bl c → length bl ≥ diameter c ∧
  (length bl = diameter c ↔ ∃ a b : ℝ × ℝ, bl = [a, b] ∧ a.1^2 + a.2^2 = 1 ∧ b.1^2 + b.2^2 = 1 ∧ (a.1 + b.1 = 0 ∧ a.2 + b.2 = 0)) :=
sorry

end shortest_dividing_line_l1642_164257


namespace rectangular_solid_surface_area_l1642_164281

/-- The total surface area of a rectangular solid -/
def total_surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: The total surface area of a rectangular solid with length 5 meters, width 4 meters, and depth 1 meter is 58 square meters -/
theorem rectangular_solid_surface_area :
  total_surface_area 5 4 1 = 58 := by
  sorry

end rectangular_solid_surface_area_l1642_164281


namespace parabola_properties_l1642_164200

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Theorem about the slope of line AB and the equation of the parabola -/
theorem parabola_properties (para : Parabola) (F H A B : Point) :
  -- Conditions
  (A.y^2 = 2 * para.p * A.x) →  -- A is on the parabola
  (B.y^2 = 2 * para.p * B.x) →  -- B is on the parabola
  (H.x = -para.p/2 ∧ H.y = 0) →  -- H is on the x-axis at (-p/2, 0)
  (F.x = para.p/2 ∧ F.y = 0) →  -- F is the focus at (p/2, 0)
  ((B.x - F.x)^2 + (B.y - F.y)^2 = 4 * ((A.x - F.x)^2 + (A.y - F.y)^2)) →  -- |BF| = 2|AF|
  -- Conclusions
  let slope := (B.y - A.y) / (B.x - A.x)
  (slope = 2*Real.sqrt 2/3 ∨ slope = -2*Real.sqrt 2/3) ∧
  (((B.x - A.x) * (B.y + A.y) / 2 = Real.sqrt 2) → para.p = 2) :=
by sorry

end parabola_properties_l1642_164200


namespace probability_of_black_ball_l1642_164201

theorem probability_of_black_ball (p_red p_white p_black : ℝ) : 
  p_red = 0.38 →
  p_white = 0.34 →
  p_red + p_white + p_black = 1 →
  p_black = 0.28 := by
sorry

end probability_of_black_ball_l1642_164201


namespace no_simultaneous_extrema_l1642_164268

/-- A partition of rational numbers -/
structure RationalPartition where
  M : Set ℚ
  N : Set ℚ
  M_nonempty : M.Nonempty
  N_nonempty : N.Nonempty
  union_eq_rat : M ∪ N = Set.univ
  intersection_empty : M ∩ N = ∅
  M_lt_N : ∀ m ∈ M, ∀ n ∈ N, m < n

/-- Theorem stating that in a partition of rationals, M cannot have a maximum and N cannot have a minimum simultaneously -/
theorem no_simultaneous_extrema (p : RationalPartition) :
  ¬(∃ (max_M : ℚ), max_M ∈ p.M ∧ ∀ m ∈ p.M, m ≤ max_M) ∨
  ¬(∃ (min_N : ℚ), min_N ∈ p.N ∧ ∀ n ∈ p.N, min_N ≤ n) :=
sorry

end no_simultaneous_extrema_l1642_164268


namespace initial_pencils_l1642_164286

/-- Given that a person:
  - starts with an initial number of pencils
  - gives away 18 pencils
  - buys 22 more pencils
  - ends up with 43 pencils
  This theorem proves that the initial number of pencils was 39. -/
theorem initial_pencils (initial : ℕ) : 
  initial - 18 + 22 = 43 → initial = 39 := by
  sorry

end initial_pencils_l1642_164286


namespace crayon_selection_theorem_l1642_164206

def total_crayons : ℕ := 15
def red_crayons : ℕ := 2
def selection_size : ℕ := 6

def select_crayons_with_red : ℕ := Nat.choose total_crayons selection_size - Nat.choose (total_crayons - red_crayons) selection_size

theorem crayon_selection_theorem : select_crayons_with_red = 2860 := by
  sorry

end crayon_selection_theorem_l1642_164206


namespace midpoint_coordinate_sum_l1642_164225

/-- The sum of the coordinates of the midpoint of a segment with endpoints (10, -2) and (-4, 8) is 6. -/
theorem midpoint_coordinate_sum : 
  let p1 : ℝ × ℝ := (10, -2)
  let p2 : ℝ × ℝ := (-4, 8)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 + midpoint.2 = 6 := by
  sorry

end midpoint_coordinate_sum_l1642_164225


namespace net_amount_calculation_l1642_164270

/-- Calculates the net amount received after selling a stock and deducting brokerage -/
def net_amount_after_brokerage (sale_amount : ℚ) (brokerage_rate : ℚ) : ℚ :=
  sale_amount - (sale_amount * brokerage_rate)

/-- Theorem stating that the net amount received after selling a stock for Rs. 108.25 
    with a 1/4% brokerage rate is Rs. 107.98 -/
theorem net_amount_calculation :
  let sale_amount : ℚ := 108.25
  let brokerage_rate : ℚ := 1 / 400  -- 1/4% expressed as a fraction
  net_amount_after_brokerage sale_amount brokerage_rate = 107.98 := by
  sorry

#eval net_amount_after_brokerage 108.25 (1 / 400)

end net_amount_calculation_l1642_164270


namespace purchase_price_calculation_l1642_164233

/-- Given a markup that includes overhead and net profit, calculate the purchase price. -/
theorem purchase_price_calculation (markup : ℝ) (overhead_rate : ℝ) (net_profit : ℝ) : 
  markup = 35 ∧ overhead_rate = 0.1 ∧ net_profit = 12 →
  ∃ (price : ℝ), price = 230 ∧ markup = overhead_rate * price + net_profit :=
by sorry

end purchase_price_calculation_l1642_164233


namespace investment_rate_correct_l1642_164292

/-- Represents the annual interest rate as a real number between 0 and 1 -/
def annual_interest_rate : ℝ := sorry

/-- The initial investment amount in yuan -/
def initial_investment : ℝ := 20000

/-- The amount withdrawn after the first year in yuan -/
def withdrawal : ℝ := 10000

/-- The final amount received after two years in yuan -/
def final_amount : ℝ := 13200

/-- Theorem stating that the annual interest rate satisfies the investment conditions -/
theorem investment_rate_correct : 
  (initial_investment * (1 + annual_interest_rate) - withdrawal) * (1 + annual_interest_rate) = final_amount ∧ 
  annual_interest_rate = 0.1 := by sorry

end investment_rate_correct_l1642_164292


namespace angle_quadrant_from_point_l1642_164204

def point_in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

def angle_in_fourth_quadrant (α : ℝ) : Prop := 
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

theorem angle_quadrant_from_point (α : ℝ) :
  point_in_third_quadrant (Real.sin α) (Real.tan α) →
  angle_in_fourth_quadrant α := by
  sorry

end angle_quadrant_from_point_l1642_164204


namespace magnitude_of_one_minus_i_l1642_164278

theorem magnitude_of_one_minus_i :
  let z : ℂ := 1 - Complex.I
  Complex.abs z = Real.sqrt 2 := by sorry

end magnitude_of_one_minus_i_l1642_164278


namespace max_salary_soccer_team_l1642_164250

/-- Represents the maximum salary problem for a soccer team -/
theorem max_salary_soccer_team 
  (num_players : ℕ) 
  (min_salary : ℕ) 
  (max_total_salary : ℕ) 
  (h1 : num_players = 25)
  (h2 : min_salary = 20000)
  (h3 : max_total_salary = 900000) :
  ∃ (max_single_salary : ℕ),
    max_single_salary = 420000 ∧
    max_single_salary + (num_players - 1) * min_salary ≤ max_total_salary ∧
    ∀ (salary : ℕ), 
      salary > max_single_salary → 
      salary + (num_players - 1) * min_salary > max_total_salary :=
by sorry

end max_salary_soccer_team_l1642_164250


namespace complement_A_intersect_B_l1642_164290

def U : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}
def A : Set ℕ := {1, 2, 3, 5, 8}
def B : Set ℕ := {1, 3, 5, 7, 9}

theorem complement_A_intersect_B : (Aᶜ ∩ B) = {7, 9} := by
  sorry

end complement_A_intersect_B_l1642_164290


namespace balanced_allocation_exists_l1642_164254

/-- Represents the daily production capacity of a worker for each part type -/
structure ProductionRate where
  typeA : ℕ
  typeB : ℕ

/-- Represents the composition of a set in terms of part types -/
structure SetComposition where
  typeA : ℕ
  typeB : ℕ

/-- Represents the allocation of workers to different part types -/
structure WorkerAllocation where
  typeA : ℕ
  typeB : ℕ

/-- Checks if the worker allocation is valid and balanced -/
def isBalancedAllocation (totalWorkers : ℕ) (rate : ProductionRate) (composition : SetComposition) (allocation : WorkerAllocation) : Prop :=
  allocation.typeA + allocation.typeB = totalWorkers ∧
  rate.typeA * allocation.typeA * composition.typeB = rate.typeB * allocation.typeB * composition.typeA

theorem balanced_allocation_exists (totalWorkers : ℕ) (rate : ProductionRate) (composition : SetComposition) 
    (h_total : totalWorkers = 85)
    (h_rate : rate = { typeA := 10, typeB := 16 })
    (h_composition : composition = { typeA := 3, typeB := 2 }) :
  ∃ (allocation : WorkerAllocation), isBalancedAllocation totalWorkers rate composition allocation ∧ 
    allocation.typeA = 60 ∧ allocation.typeB = 25 := by
  sorry

end balanced_allocation_exists_l1642_164254


namespace calculation_result_l1642_164243

theorem calculation_result : 1 + 0.1 - 0.1 + 1 = 2 := by
  sorry

end calculation_result_l1642_164243


namespace unique_solution_floor_product_l1642_164289

theorem unique_solution_floor_product : 
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 45 ∧ x = 7.5 := by sorry

end unique_solution_floor_product_l1642_164289


namespace quadratic_equation_root_zero_l1642_164272

theorem quadratic_equation_root_zero (k : ℝ) : 
  (k + 3 ≠ 0) →
  (∀ x, (k + 3) * x^2 + 5 * x + k^2 + 2 * k - 3 = 0 ↔ x = 0 ∨ x ≠ 0) →
  k = 1 := by
  sorry

end quadratic_equation_root_zero_l1642_164272


namespace fox_distribution_l1642_164236

/-- The fox distribution problem -/
theorem fox_distribution
  (m : ℕ) (a : ℝ) (x y : ℝ)
  (h_positive : m > 1 ∧ a > 0)
  (h_distribution : ∀ (n : ℕ), n > 0 → n * a + (x - (n - 1) * y - n * a) / m = y) :
  x = (m - 1)^2 * a ∧ y = (m - 1) * a ∧ (m - 1 : ℝ) = x / y :=
by sorry

end fox_distribution_l1642_164236


namespace inequality_proof_l1642_164283

theorem inequality_proof (a b c : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ 1) (h3 : 1 ≥ c) (h4 : c ≥ 0) 
  (h5 : a + b + c = 3) : 
  2 ≤ a * b + b * c + c * a ∧ 
  a * b + b * c + c * a ≤ 3 ∧
  24 / (a^3 + b^3 + c^3) + 25 / (a * b + b * c + c * a) ≥ 14 := by
  sorry

#check inequality_proof

end inequality_proof_l1642_164283


namespace fixed_point_on_circle_l1642_164242

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*m*y + 6*m - 2 = 0

-- Theorem statement
theorem fixed_point_on_circle :
  ∀ m : ℝ, circle_equation 1 1 m ∨ circle_equation (1/5) (7/5) m :=
by sorry

end fixed_point_on_circle_l1642_164242


namespace line_slope_product_l1642_164222

/-- Given two lines L₁ and L₂ with equations y = 3mx and y = nx respectively,
    where L₁ makes three times the angle with the horizontal as L₂,
    L₁ has 3 times the slope of L₂, and L₁ is not vertical,
    prove that the product mn equals 9/4. -/
theorem line_slope_product (m n : ℝ) : 
  (∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ 
               3 * m = Real.tan θ₁ ∧ 
               n = Real.tan θ₂ ∧ 
               m = 3 * n ∧ 
               m ≠ 0) →
  m * n = 9 / 4 := by
sorry

end line_slope_product_l1642_164222


namespace prob_no_shaded_correct_l1642_164221

/-- Represents a rectangle in the 2 by 2005 grid -/
structure Rectangle where
  left : Fin 2006
  right : Fin 2006
  top : Fin 3
  bottom : Fin 3
  h_valid : left < right

/-- The total number of rectangles in the grid -/
def total_rectangles : ℕ := 3 * (1003 * 2005)

/-- The number of rectangles containing a shaded square -/
def shaded_rectangles : ℕ := 3 * (1003 * 1003)

/-- Predicate for whether a rectangle contains a shaded square -/
def contains_shaded (r : Rectangle) : Prop :=
  (r.left ≤ 1003 ∧ r.right > 1003) ∨ (r.top = 0 ∧ r.bottom = 1) ∨ (r.top = 1 ∧ r.bottom = 2)

/-- The probability of choosing a rectangle that does not contain a shaded square -/
def prob_no_shaded : ℚ := 1002 / 2005

theorem prob_no_shaded_correct :
  (total_rectangles - shaded_rectangles : ℚ) / total_rectangles = prob_no_shaded := by
  sorry

end prob_no_shaded_correct_l1642_164221


namespace largest_multiple_of_18_with_9_and_0_digits_l1642_164211

def is_multiple_of_18 (n : ℕ) : Prop := ∃ k : ℕ, n = 18 * k

def digits_are_9_or_0 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 9 ∨ d = 0

theorem largest_multiple_of_18_with_9_and_0_digits :
  ∃ m : ℕ,
    is_multiple_of_18 m ∧
    digits_are_9_or_0 m ∧
    (∀ n : ℕ, is_multiple_of_18 n → digits_are_9_or_0 n → n ≤ m) ∧
    m = 900 ∧
    m / 18 = 50 := by sorry

end largest_multiple_of_18_with_9_and_0_digits_l1642_164211


namespace excess_value_proof_l1642_164276

def two_digit_number : ℕ := 57

def tens_digit (n : ℕ) : ℕ := n / 10
def ones_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ := tens_digit n + ones_digit n

def reversed_number (n : ℕ) : ℕ := 10 * (ones_digit n) + tens_digit n

theorem excess_value_proof :
  ∃ (v : ℕ), two_digit_number = 4 * (sum_of_digits two_digit_number) + v ∧
  two_digit_number + 18 = reversed_number two_digit_number ∧
  v = 9 := by
  sorry

end excess_value_proof_l1642_164276


namespace bingo_prize_distribution_l1642_164217

theorem bingo_prize_distribution (total_prize : ℕ) (first_winner_fraction : ℚ) 
  (remaining_winners : ℕ) (remaining_fraction : ℚ) :
  total_prize = 2400 →
  first_winner_fraction = 1 / 3 →
  remaining_winners = 10 →
  remaining_fraction = 1 / 10 →
  (total_prize - (first_winner_fraction * total_prize).num) / remaining_winners = 160 := by
  sorry

end bingo_prize_distribution_l1642_164217


namespace f_max_value_l1642_164245

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 2 := by
  sorry

end f_max_value_l1642_164245


namespace binomial_n_value_l1642_164241

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_n_value (ξ : BinomialRV) 
  (h_exp : expectation ξ = 6)
  (h_var : variance ξ = 3) : 
  ξ.n = 12 := by
  sorry

end binomial_n_value_l1642_164241


namespace video_game_collection_cost_l1642_164234

theorem video_game_collection_cost (total_games : ℕ) 
  (games_at_12 : ℕ) (price_12 : ℕ) (price_7 : ℕ) (price_3 : ℕ) :
  total_games = 346 →
  games_at_12 = 80 →
  price_12 = 12 →
  price_7 = 7 →
  price_3 = 3 →
  (games_at_12 * price_12 + 
   ((total_games - games_at_12) / 2) * price_7 + 
   ((total_games - games_at_12) - ((total_games - games_at_12) / 2)) * price_3) = 2290 := by
sorry

#eval 80 * 12 + ((346 - 80) / 2) * 7 + ((346 - 80) - ((346 - 80) / 2)) * 3

end video_game_collection_cost_l1642_164234


namespace sqrt_of_four_l1642_164275

theorem sqrt_of_four : Real.sqrt 4 = 2 := by
  sorry

end sqrt_of_four_l1642_164275


namespace light_flash_duration_l1642_164237

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The interval between flashes in seconds -/
def flash_interval : ℕ := 20

/-- The number of flashes -/
def num_flashes : ℕ := 180

/-- Theorem: The time it takes for 180 flashes of a light that flashes every 20 seconds is equal to 1 hour -/
theorem light_flash_duration : 
  (flash_interval * num_flashes) = seconds_per_hour := by sorry

end light_flash_duration_l1642_164237


namespace divisiblity_condition_l1642_164244

def recursive_sequence : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => (recursive_sequence (n + 1))^2 + recursive_sequence (n + 1) + 1 / recursive_sequence n

theorem divisiblity_condition (a b : ℕ) :
  a > 0 ∧ b > 0 →
  a ∣ (b^2 + b + 1) →
  b ∣ (a^2 + a + 1) →
  ((a = 1 ∧ b = 3) ∨ 
   (∃ n : ℕ, a = recursive_sequence n ∧ b = recursive_sequence (n + 1))) :=
by sorry

end divisiblity_condition_l1642_164244


namespace expression_equality_l1642_164298

theorem expression_equality : -2^2 + (1 / (Real.sqrt 2 - 1))^0 - abs (2 * Real.sqrt 2 - 3) + Real.cos (π / 3) = -5 + 2 * Real.sqrt 2 := by
  sorry

end expression_equality_l1642_164298


namespace value_of_y_l1642_164280

theorem value_of_y : ∃ y : ℝ, (3 * y - 9) / 3 = 18 ∧ y = 21 := by
  sorry

end value_of_y_l1642_164280


namespace matrix_operation_result_l1642_164226

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 6, 1]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-7, 8; 3, -5]

theorem matrix_operation_result : 
  2 • A + B = !![1, 2; 15, -3] := by sorry

end matrix_operation_result_l1642_164226


namespace apple_orange_ratio_l1642_164215

/-- Given a basket of fruit with apples and oranges, prove the ratio of apples to oranges --/
theorem apple_orange_ratio (total_fruit : ℕ) (oranges : ℕ) : 
  total_fruit = 40 → oranges = 10 → (total_fruit - oranges) / oranges = 3 := by
  sorry

#check apple_orange_ratio

end apple_orange_ratio_l1642_164215


namespace gcd_of_specific_numbers_l1642_164294

theorem gcd_of_specific_numbers : Nat.gcd 333333 888888888 = 3 := by sorry

end gcd_of_specific_numbers_l1642_164294


namespace marble_drawing_probability_l1642_164232

/-- Represents the total number of marbles in the bag. -/
def total_marbles : ℕ := 800

/-- Represents the number of different colors of marbles. -/
def num_colors : ℕ := 100

/-- Represents the number of marbles of each color. -/
def marbles_per_color : ℕ := 8

/-- Represents the number of marbles drawn so far. -/
def marbles_drawn : ℕ := 699

/-- Represents the target number of marbles of the same color to stop drawing. -/
def target_same_color : ℕ := 8

/-- Represents the probability of stopping after drawing the 700th marble. -/
def stop_probability : ℚ := 99 / 101

theorem marble_drawing_probability :
  total_marbles = num_colors * marbles_per_color ∧
  marbles_drawn < total_marbles ∧
  marbles_drawn ≥ (num_colors - 1) * (target_same_color - 1) + (target_same_color - 2) →
  stop_probability = 99 / 101 :=
by sorry

end marble_drawing_probability_l1642_164232


namespace probability_of_two_in_three_elevenths_l1642_164297

theorem probability_of_two_in_three_elevenths : 
  let decimal_rep := (3 : ℚ) / 11
  let period := 2
  let count_of_two := 1
  (count_of_two : ℚ) / period = 1 / 2 := by sorry

end probability_of_two_in_three_elevenths_l1642_164297


namespace juice_bar_spending_l1642_164251

theorem juice_bar_spending (mango_price pineapple_price pineapple_total group_size : ℕ) 
  (h1 : mango_price = 5)
  (h2 : pineapple_price = 6)
  (h3 : pineapple_total = 54)
  (h4 : group_size = 17) :
  ∃ (mango_glasses pineapple_glasses : ℕ),
    mango_glasses + pineapple_glasses = group_size ∧
    mango_glasses * mango_price + pineapple_glasses * pineapple_price = 94 :=
by
  sorry

end juice_bar_spending_l1642_164251


namespace hyperbola_real_axis_length_l1642_164271

/-- Represents a hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a line with equation x - √3y + 2 = 0 -/
def special_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 2 = 0

theorem hyperbola_real_axis_length
  (h : Hyperbola)
  (focus_on_line : ∃ (x y : ℝ), special_line x y ∧ x^2 / h.a^2 - y^2 / h.b^2 = 1)
  (perpendicular_to_asymptote : h.b / h.a = Real.sqrt 3) :
  2 * h.a = 2 := by sorry

end hyperbola_real_axis_length_l1642_164271


namespace expression_equals_eight_l1642_164266

theorem expression_equals_eight (a : ℝ) (h : a = 2) : 
  (a^3 + (3*a)^3) / (a^2 - a*(3*a) + (3*a)^2) = 8 := by
  sorry

end expression_equals_eight_l1642_164266


namespace can_repair_propeller_l1642_164277

/-- The cost of a blade in tugriks -/
def blade_cost : ℕ := 120

/-- The cost of a screw in tugriks -/
def screw_cost : ℕ := 9

/-- The discount rate applied after spending 250 tugriks -/
def discount_rate : ℚ := 0.2

/-- The threshold for applying the discount -/
def discount_threshold : ℕ := 250

/-- Karlsson's budget in tugriks -/
def budget : ℕ := 360

/-- The number of blades needed -/
def blades_needed : ℕ := 3

/-- The number of screws needed -/
def screws_needed : ℕ := 1

/-- Function to calculate the total cost with discount -/
def total_cost_with_discount (blade_cost screw_cost : ℕ) (discount_rate : ℚ) 
  (discount_threshold blades_needed screws_needed : ℕ) : ℚ :=
  let initial_purchase := 2 * blade_cost + 2 * screw_cost
  let remaining_purchase := blade_cost
  if initial_purchase ≥ discount_threshold
  then initial_purchase + remaining_purchase * (1 - discount_rate)
  else initial_purchase + remaining_purchase

/-- Theorem stating that Karlsson can afford to repair his propeller -/
theorem can_repair_propeller : 
  total_cost_with_discount blade_cost screw_cost discount_rate 
    discount_threshold blades_needed screws_needed ≤ budget := by
  sorry

end can_repair_propeller_l1642_164277


namespace no_k_exists_product_minus_one_is_power_l1642_164260

/-- The nth odd prime number -/
def nthOddPrime (n : ℕ) : ℕ := sorry

/-- The product of the first k odd prime numbers -/
def productFirstKOddPrimes (k : ℕ) : ℕ := sorry

/-- Theorem: There does not exist a natural number k such that the product of the first k odd prime numbers minus 1 is an exact power of a natural number greater than one -/
theorem no_k_exists_product_minus_one_is_power :
  ¬ ∃ (k : ℕ), ∃ (a n : ℕ), n > 1 ∧ productFirstKOddPrimes k - 1 = a^n :=
sorry

end no_k_exists_product_minus_one_is_power_l1642_164260


namespace work_remaining_fraction_l1642_164288

theorem work_remaining_fraction 
  (days_a : ℝ) (days_b : ℝ) (days_c : ℝ) (work_days : ℝ) 
  (h1 : days_a = 10) 
  (h2 : days_b = 20) 
  (h3 : days_c = 30) 
  (h4 : work_days = 5) : 
  1 - work_days * (1 / days_a + 1 / days_b + 1 / days_c) = 5 / 60 := by
  sorry

end work_remaining_fraction_l1642_164288


namespace nine_cakes_l1642_164246

/-- Represents the arrangement of cakes on a round table. -/
def CakeArrangement (n : ℕ) := Fin n

/-- Represents the action of eating every third cake. -/
def eatEveryThird (n : ℕ) (i : Fin n) : Fin n :=
  ⟨(i + 3) % n, by sorry⟩

/-- Represents the number of laps needed to eat all cakes. -/
def lapsToEatAll (n : ℕ) : ℕ := 7

/-- The last cake eaten is the same as the first one encountered. -/
def lastIsFirst (n : ℕ) : Prop :=
  ∃ (i : Fin n), (lapsToEatAll n).iterate (eatEveryThird n) i = i

/-- The main theorem stating that there are 9 cakes on the table. -/
theorem nine_cakes :
  ∃ (n : ℕ), n = 9 ∧ 
  lapsToEatAll n = 7 ∧
  lastIsFirst n :=
sorry

end nine_cakes_l1642_164246


namespace parametric_to_equation_l1642_164285

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parametric equation of a plane -/
structure ParametricPlane where
  constant : Point3D
  direction1 : Point3D
  direction2 : Point3D

/-- Represents the equation of a plane in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if the given coefficients satisfy the required conditions -/
def validCoefficients (eq : PlaneEquation) : Prop :=
  eq.A > 0 ∧ Nat.gcd (Int.natAbs eq.A) (Int.natAbs eq.B) = 1 ∧
  Nat.gcd (Int.natAbs eq.A) (Int.natAbs eq.C) = 1 ∧
  Nat.gcd (Int.natAbs eq.A) (Int.natAbs eq.D) = 1

/-- Check if a point satisfies the plane equation -/
def satisfiesEquation (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- The main theorem to prove -/
theorem parametric_to_equation (plane : ParametricPlane) :
  ∃ (eq : PlaneEquation),
    validCoefficients eq ∧
    (∀ (s t : ℝ),
      let p : Point3D := {
        x := plane.constant.x + s * plane.direction1.x + t * plane.direction2.x,
        y := plane.constant.y + s * plane.direction1.y + t * plane.direction2.y,
        z := plane.constant.z + s * plane.direction1.z + t * plane.direction2.z
      }
      satisfiesEquation p eq) ∧
    eq.A = 2 ∧ eq.B = -5 ∧ eq.C = 2 ∧ eq.D = -7 := by
  sorry

end parametric_to_equation_l1642_164285


namespace marbles_in_bag_l1642_164287

theorem marbles_in_bag (total_marbles : ℕ) (red_marbles : ℕ) : 
  red_marbles = 12 →
  ((total_marbles - red_marbles : ℚ) / total_marbles) ^ 2 = 9 / 16 →
  total_marbles = 48 := by
sorry

end marbles_in_bag_l1642_164287


namespace sunglasses_wearers_l1642_164279

theorem sunglasses_wearers (total_adults : ℕ) (women_percentage : ℚ) (men_percentage : ℚ) : 
  total_adults = 1800 → 
  women_percentage = 25 / 100 →
  men_percentage = 10 / 100 →
  (total_adults / 2 * women_percentage + total_adults / 2 * men_percentage : ℚ) = 315 := by
  sorry

end sunglasses_wearers_l1642_164279


namespace g_of_four_value_l1642_164228

/-- A function from positive reals to positive reals satisfying certain conditions -/
def G := {g : ℝ → ℝ // ∀ x > 0, g x > 0 ∧ g 1 = 1 ∧ g (x^2 * g x) = x * g (x^2) + g x}

theorem g_of_four_value (g : G) : g.val 4 = 36/23 := by
  sorry

end g_of_four_value_l1642_164228


namespace otimes_example_l1642_164220

/-- Custom operation ⊗ defined as a ⊗ b = a² - ab -/
def otimes (a b : ℤ) : ℤ := a^2 - a * b

/-- Theorem stating that 4 ⊗ [2 ⊗ (-5)] = -40 -/
theorem otimes_example : otimes 4 (otimes 2 (-5)) = -40 := by
  sorry

end otimes_example_l1642_164220


namespace planes_parallel_to_same_plane_not_axiom_l1642_164218

-- Define the type for geometric propositions
inductive GeometricProposition
  | PlanesParallelToSamePlaneAreParallel
  | ThreePointsDetermineUniquePlane
  | LineInPlaneImpliesAllPointsInPlane
  | TwoPlanesWithCommonPointHaveCommonLine

-- Define the set of axioms
def geometryAxioms : Set GeometricProposition :=
  { GeometricProposition.ThreePointsDetermineUniquePlane,
    GeometricProposition.LineInPlaneImpliesAllPointsInPlane,
    GeometricProposition.TwoPlanesWithCommonPointHaveCommonLine }

-- Theorem statement
theorem planes_parallel_to_same_plane_not_axiom :
  GeometricProposition.PlanesParallelToSamePlaneAreParallel ∉ geometryAxioms :=
by sorry

end planes_parallel_to_same_plane_not_axiom_l1642_164218


namespace expression_equality_l1642_164256

theorem expression_equality (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10) 
  (sum_squares_condition : p^2 + q^2 + r^2 + s^2 = 26) :
  6 * (p^4 + q^4 + r^4 + s^4) - (p^3 + q^3 + r^3 + s^3) = 
  6 * ((p-1)^4 + (q-1)^4 + (r-1)^4 + (s-1)^4) - ((p-1)^3 + (q-1)^3 + (r-1)^3 + (s-1)^3) := by
  sorry

end expression_equality_l1642_164256


namespace range_of_m_l1642_164230

theorem range_of_m (p q : Prop) (m : ℝ) 
  (h1 : p ∨ q) 
  (h2 : ¬(p ∧ q)) 
  (h3 : p ↔ m < 0) 
  (h4 : q ↔ m < 2) : 
  0 ≤ m ∧ m < 2 := by
sorry

end range_of_m_l1642_164230


namespace negation_of_existence_real_root_l1642_164203

theorem negation_of_existence_real_root : 
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by
  sorry

end negation_of_existence_real_root_l1642_164203
