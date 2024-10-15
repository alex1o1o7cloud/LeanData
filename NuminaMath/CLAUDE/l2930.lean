import Mathlib

namespace NUMINAMATH_CALUDE_power_of_three_mod_thousand_l2930_293011

theorem power_of_three_mod_thousand :
  ∃ n : ℕ, n < 1000 ∧ 3^5000 ≡ n [MOD 1000] :=
sorry

end NUMINAMATH_CALUDE_power_of_three_mod_thousand_l2930_293011


namespace NUMINAMATH_CALUDE_max_correct_answers_l2930_293004

/-- Represents an exam score. -/
structure ExamScore where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ
  totalQuestions : ℕ
  score : ℤ

/-- Checks if the exam score is valid according to the rules. -/
def ExamScore.isValid (e : ExamScore) : Prop :=
  e.correct + e.incorrect + e.unanswered = e.totalQuestions ∧
  6 * e.correct - 3 * e.incorrect = e.score

/-- Theorem: The maximum number of correct answers for the given exam conditions is 14. -/
theorem max_correct_answers :
  ∀ e : ExamScore,
    e.totalQuestions = 25 →
    e.score = 57 →
    e.isValid →
    e.correct ≤ 14 ∧
    ∃ e' : ExamScore, e'.totalQuestions = 25 ∧ e'.score = 57 ∧ e'.isValid ∧ e'.correct = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l2930_293004


namespace NUMINAMATH_CALUDE_farmers_children_count_l2930_293031

/-- Represents the problem of determining the number of farmer's children based on apple collection and consumption. -/
theorem farmers_children_count :
  ∀ (n : ℕ),
  (n * 15 - 8 - 7 = 60) →
  n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_farmers_children_count_l2930_293031


namespace NUMINAMATH_CALUDE_sue_dogs_walked_l2930_293081

def perfume_cost : ℕ := 50
def christian_initial_savings : ℕ := 5
def sue_initial_savings : ℕ := 7
def yards_mowed : ℕ := 4
def yard_mowing_rate : ℕ := 5
def dog_walking_rate : ℕ := 2
def additional_needed : ℕ := 6

theorem sue_dogs_walked :
  ∃ (dogs_walked : ℕ),
    perfume_cost =
      christian_initial_savings + sue_initial_savings +
      yards_mowed * yard_mowing_rate +
      dogs_walked * dog_walking_rate +
      additional_needed ∧
    dogs_walked = 6 := by
  sorry

end NUMINAMATH_CALUDE_sue_dogs_walked_l2930_293081


namespace NUMINAMATH_CALUDE_part_1_part_2_l2930_293034

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.exp x - 1) - a * x^2

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x + x * Real.exp x - 1 - 2 * a * x

-- Theorem for part 1
theorem part_1 (a : ℝ) : f' a 1 = 2 * Real.exp 1 - 2 → a = 1/2 := by sorry

-- Define the specific function f with a = 1/2
def f_half (x : ℝ) : ℝ := x * (Real.exp x - 1) - (1/2) * x^2

-- Define the derivative of f_half
def f_half' (x : ℝ) : ℝ := (x + 1) * (Real.exp x - 1)

-- Theorem for part 2
theorem part_2 (m : ℝ) : 
  (∀ x ∈ Set.Ioo (2*m - 3) (3*m - 2), f_half' x > 0) ↔ 
  (m ∈ Set.Ioc (-1) (1/3) ∪ Set.Ici (3/2)) := by sorry

end

end NUMINAMATH_CALUDE_part_1_part_2_l2930_293034


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l2930_293051

/-- The number of basic ice cream flavors -/
def num_flavors : ℕ := 4

/-- The number of scoops used to create a new flavor -/
def num_scoops : ℕ := 4

/-- 
The number of ways to distribute indistinguishable objects into distinguishable categories
n: number of objects
k: number of categories
-/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The total number of distinct ice cream flavors that can be created -/
def total_flavors : ℕ := stars_and_bars num_scoops num_flavors

theorem ice_cream_flavors_count : total_flavors = 35 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l2930_293051


namespace NUMINAMATH_CALUDE_max_two_wins_l2930_293019

/-- Represents a single-elimination tournament --/
structure Tournament :=
  (participants : ℕ)

/-- Represents the number of participants who won exactly two matches --/
def exactlyTwoWins (t : Tournament) : ℕ := sorry

/-- The theorem stating the maximum number of participants who can win exactly two matches --/
theorem max_two_wins (t : Tournament) (h : t.participants = 100) : 
  exactlyTwoWins t ≤ 49 ∧ ∃ (strategy : Unit), exactlyTwoWins t = 49 := by sorry

end NUMINAMATH_CALUDE_max_two_wins_l2930_293019


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l2930_293099

theorem rectangle_shorter_side (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Positive dimensions
  2 * (a + b) = 52 ∧  -- Perimeter condition
  a * b = 168 ∧  -- Area condition
  a ≥ b  -- a is the longer side
  → b = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l2930_293099


namespace NUMINAMATH_CALUDE_total_money_division_l2930_293088

theorem total_money_division (b c : ℕ) (total : ℕ) : 
  (b : ℚ) / c = 4 / 16 →
  c * 100 = 1600 →
  total = b * 100 + c * 100 →
  total = 2000 := by
sorry

end NUMINAMATH_CALUDE_total_money_division_l2930_293088


namespace NUMINAMATH_CALUDE_neg_half_pow_4_mul_6_three_squared_mul_neg_three_cubed_two_cubed_sum_four_times_find_p_in_equation_l2930_293089

-- Operation rule of multiplication of powers with the same base
axiom pow_mul_rule {α : Type*} [Monoid α] (a : α) (m n : ℕ) : a^m * a^n = a^(m+n)

-- Statement 1
theorem neg_half_pow_4_mul_6 : (-1/2 : ℚ)^4 * (-1/2 : ℚ)^6 = (-1/2 : ℚ)^10 := by sorry

-- Statement 2
theorem three_squared_mul_neg_three_cubed : (3 : ℤ)^2 * (-3 : ℤ)^3 = -243 := by sorry

-- Statement 3
theorem two_cubed_sum_four_times : (2 : ℕ)^3 + (2 : ℕ)^3 + (2 : ℕ)^3 + (2 : ℕ)^3 = (2 : ℕ)^5 := by sorry

-- Statement 4
theorem find_p_in_equation (x y : ℝ) :
  ∃ p : ℕ, (x - y)^2 * (x - y)^p * (x - y)^5 = (x - y)^2023 ∧ p = 2016 := by sorry

end NUMINAMATH_CALUDE_neg_half_pow_4_mul_6_three_squared_mul_neg_three_cubed_two_cubed_sum_four_times_find_p_in_equation_l2930_293089


namespace NUMINAMATH_CALUDE_seating_theorem_l2930_293092

/-- The number of ways to seat n people around a round table. -/
def circular_permutations (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to seat 6 people around a round table,
    with two specific people always sitting next to each other. -/
def seating_arrangements : ℕ :=
  2 * circular_permutations 5

theorem seating_theorem : seating_arrangements = 48 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l2930_293092


namespace NUMINAMATH_CALUDE_velocity_dividing_trapezoid_area_l2930_293075

/-- 
Given a trapezoidal velocity-time graph with bases V and U, 
this theorem proves that the velocity W that divides the area 
under the curve in the ratio 1:k is given by W = √((V^2 + kU^2) / (k + 1)).
-/
theorem velocity_dividing_trapezoid_area 
  (V U : ℝ) (k : ℝ) (hk : k > 0) :
  let W := Real.sqrt ((V^2 + k * U^2) / (k + 1))
  ∃ (h : ℝ), 
    h * (V - W) = (1 / (k + 1)) * ((1 / 2) * h * (V + U)) ∧
    h * (W - U) = (k / (k + 1)) * ((1 / 2) * h * (V + U)) :=
by sorry

end NUMINAMATH_CALUDE_velocity_dividing_trapezoid_area_l2930_293075


namespace NUMINAMATH_CALUDE_range_of_a_l2930_293062

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → a * Real.exp (a * Real.exp x + a) ≥ Real.log (Real.exp x + 1)) → 
  a ≥ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2930_293062


namespace NUMINAMATH_CALUDE_yellow_to_red_ratio_l2930_293086

/-- Represents the number of chairs of each color in Susan's house. -/
structure ChairCounts where
  red : ℕ
  yellow : ℕ
  blue : ℕ

/-- The conditions of the chair problem in Susan's house. -/
def susansChairs : ChairCounts → Prop := fun c =>
  c.red = 5 ∧
  c.blue = c.yellow - 2 ∧
  c.red + c.yellow + c.blue = 43

/-- The theorem stating the ratio of yellow to red chairs. -/
theorem yellow_to_red_ratio (c : ChairCounts) (h : susansChairs c) :
  c.yellow / c.red = 4 := by
  sorry

#check yellow_to_red_ratio

end NUMINAMATH_CALUDE_yellow_to_red_ratio_l2930_293086


namespace NUMINAMATH_CALUDE_square_of_sum_l2930_293039

theorem square_of_sum (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l2930_293039


namespace NUMINAMATH_CALUDE_anthony_jim_difference_l2930_293035

/-- The number of pairs of shoes Scott has -/
def scott_shoes : ℕ := 7

/-- The number of pairs of shoes Anthony has -/
def anthony_shoes : ℕ := 3 * scott_shoes

/-- The number of pairs of shoes Jim has -/
def jim_shoes : ℕ := anthony_shoes - 2

/-- Theorem: Anthony has 2 more pairs of shoes than Jim -/
theorem anthony_jim_difference : anthony_shoes - jim_shoes = 2 := by
  sorry

end NUMINAMATH_CALUDE_anthony_jim_difference_l2930_293035


namespace NUMINAMATH_CALUDE_sarah_and_tom_ages_l2930_293052

/-- Given the age relationship between Sarah and Tom, prove their current ages sum to 33 -/
theorem sarah_and_tom_ages : ∃ (s t : ℕ),
  (s = t + 7) ∧                   -- Sarah is seven years older than Tom
  (s + 10 = 3 * (t - 3)) ∧        -- Ten years from now, Sarah will be three times as old as Tom was three years ago
  (s + t = 33)                    -- The sum of their current ages is 33
:= by sorry

end NUMINAMATH_CALUDE_sarah_and_tom_ages_l2930_293052


namespace NUMINAMATH_CALUDE_production_system_l2930_293021

/-- Represents the profit functions and properties of a production system with two products. -/
theorem production_system (total_workers : ℕ) 
  (prod_rate_A prod_rate_B : ℕ) 
  (profit_per_A profit_per_B cost_increase_B : ℚ) : 
  total_workers = 65 → 
  prod_rate_A = 2 →
  prod_rate_B = 1 →
  profit_per_A = 15 →
  profit_per_B = 120 →
  cost_increase_B = 2 →
  ∃ (profit_A profit_B : ℚ → ℚ) (x : ℚ),
    (∀ x, profit_A x = 1950 - 30 * x) ∧
    (∀ x, profit_B x = 120 * x - 2 * x^2) ∧
    (profit_A x - profit_B x = 1250 → x = 5) ∧
    (∃ (total_profit : ℚ → ℚ),
      (∀ x, total_profit x = profit_A x + profit_B x) ∧
      (∀ y, total_profit y ≤ 2962) ∧
      (total_profit 22 = 2962 ∨ total_profit 23 = 2962)) :=
by sorry

end NUMINAMATH_CALUDE_production_system_l2930_293021


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_1997_l2930_293097

theorem tens_digit_of_13_pow_1997 :
  13^1997 % 100 = 53 := by sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_1997_l2930_293097


namespace NUMINAMATH_CALUDE_zach_remaining_amount_l2930_293026

/-- Represents the financial situation for Zach's bike purchase --/
structure BikeSavings where
  bike_cost : ℕ
  weekly_allowance : ℕ
  lawn_mowing_pay : ℕ
  babysitting_rate : ℕ
  current_savings : ℕ
  babysitting_hours : ℕ

/-- Calculates the remaining amount needed to buy the bike --/
def remaining_amount (s : BikeSavings) : ℕ :=
  s.bike_cost - (s.current_savings + s.weekly_allowance + s.lawn_mowing_pay + s.babysitting_rate * s.babysitting_hours)

/-- Theorem stating the remaining amount Zach needs to earn --/
theorem zach_remaining_amount :
  let s : BikeSavings := {
    bike_cost := 100,
    weekly_allowance := 5,
    lawn_mowing_pay := 10,
    babysitting_rate := 7,
    current_savings := 65,
    babysitting_hours := 2
  }
  remaining_amount s = 6 := by sorry

end NUMINAMATH_CALUDE_zach_remaining_amount_l2930_293026


namespace NUMINAMATH_CALUDE_factorization_3mx_minus_9my_l2930_293096

theorem factorization_3mx_minus_9my (m x y : ℝ) :
  3 * m * x - 9 * m * y = 3 * m * (x - 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3mx_minus_9my_l2930_293096


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2930_293022

theorem y_intercept_of_line (x y : ℝ) :
  2 * x - 3 * y = 6 → x = 0 → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2930_293022


namespace NUMINAMATH_CALUDE_equation_solution_l2930_293013

theorem equation_solution :
  ∃ (x : ℝ), x ≠ 2 ∧ x ≠ -2 ∧ (x / (x - 2) + 2 / (x^2 - 4) = 1) ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2930_293013


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2930_293042

theorem fraction_evaluation : (4 * 3) / (2 + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2930_293042


namespace NUMINAMATH_CALUDE_notebook_cost_l2930_293049

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (s n c : Nat),
  -- Total number of students
  total_students = 42 ∧
  -- Majority of students bought notebooks
  s > total_students / 2 ∧
  -- Number of notebooks per student is greater than 2
  n > 2 ∧
  -- Cost in cents is greater than number of notebooks
  c > n ∧
  -- Total cost equation
  s * n * c = total_cost ∧
  -- Given total cost
  total_cost = 2773 →
  -- Conclusion: cost of a notebook is 103 cents
  c = 103 :=
sorry

end NUMINAMATH_CALUDE_notebook_cost_l2930_293049


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2930_293038

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) (h : ∀ x ≤ 1, f (1 - Real.sqrt x) = x) :
  ∀ x ≤ 1, f x = x^2 - 2*x + 1 := by sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) 
  (h1 : ∃ a b : ℝ, ∀ x, f x = a * x + b) 
  (h2 : ∀ x, f (f x) = 4 * x + 3) :
  (∀ x, f x = 2 * x + 1) ∨ (∀ x, f x = -2 * x - 3) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2930_293038


namespace NUMINAMATH_CALUDE_percentage_of_fair_haired_women_l2930_293055

theorem percentage_of_fair_haired_women (total : ℝ) 
  (h1 : total > 0) 
  (fair_haired_ratio : ℝ) 
  (h2 : fair_haired_ratio = 0.75)
  (women_ratio_among_fair_haired : ℝ) 
  (h3 : women_ratio_among_fair_haired = 0.40) : 
  (fair_haired_ratio * women_ratio_among_fair_haired) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_fair_haired_women_l2930_293055


namespace NUMINAMATH_CALUDE_circle_equation_l2930_293074

-- Define the line L1: x + y + 2 = 0
def L1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 + 2 = 0}

-- Define the circle C1: x² + y² = 4
def C1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define the line L2: 2x - y - 3 = 0
def L2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2*p.1 - p.2 - 3 = 0}

-- Define the circle C we're looking for
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 - 6*p.2 - 16 = 0}

theorem circle_equation :
  (∀ p ∈ L1 ∩ C1, p ∈ C) ∧
  (∃ center ∈ L2, ∀ p ∈ C, (p.1 - center.1)^2 + (p.2 - center.2)^2 = (6^2 + 6^2) / 4) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l2930_293074


namespace NUMINAMATH_CALUDE_exists_p_with_conditions_l2930_293048

theorem exists_p_with_conditions : ∃ p : ℕ+, 
  ∃ q r s : ℕ+,
  (Nat.gcd p q = 40) ∧
  (Nat.gcd q r = 45) ∧
  (Nat.gcd r s = 60) ∧
  (∃ k : ℕ+, Nat.gcd s p = 10 * k ∧ k ≥ 10 ∧ k < 100) ∧
  (∃ m : ℕ+, p = 7 * m) := by
sorry

end NUMINAMATH_CALUDE_exists_p_with_conditions_l2930_293048


namespace NUMINAMATH_CALUDE_tomato_price_is_five_l2930_293060

/-- Represents the price per pound of tomatoes -/
def tomato_price : ℝ := sorry

/-- The number of pounds of tomatoes bought -/
def tomato_pounds : ℝ := 2

/-- The number of pounds of apples bought -/
def apple_pounds : ℝ := 5

/-- The price per pound of apples -/
def apple_price : ℝ := 6

/-- The total amount spent -/
def total_spent : ℝ := 40

/-- Theorem stating that the price per pound of tomatoes is $5 -/
theorem tomato_price_is_five :
  tomato_price * tomato_pounds + apple_price * apple_pounds = total_spent →
  tomato_price = 5 := by sorry

end NUMINAMATH_CALUDE_tomato_price_is_five_l2930_293060


namespace NUMINAMATH_CALUDE_range_of_a_l2930_293047

/-- The range of a satisfying the given conditions -/
theorem range_of_a : ∀ a : ℝ, 
  ((∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 2*x - 8 > 0) ∧ 
   (∃ x : ℝ, x^2 + 2*x - 8 > 0 ∧ x^2 - 4*a*x + 3*a^2 ≥ 0)) ↔ 
  (a ≤ -4 ∨ a ≥ 2 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2930_293047


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l2930_293043

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l2930_293043


namespace NUMINAMATH_CALUDE_problem_solution_l2930_293054

theorem problem_solution :
  let x : ℝ := -39660 - 17280 * Real.sqrt 2
  (x + 720 * Real.sqrt 1152) / Real.rpow 15625 (1/3) = 7932 / (3^2 - Real.sqrt 196) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2930_293054


namespace NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_75_sin_15_eq_zero_l2930_293008

theorem cos_75_cos_15_minus_sin_75_sin_15_eq_zero :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) - 
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_75_sin_15_eq_zero_l2930_293008


namespace NUMINAMATH_CALUDE_archer_probability_l2930_293098

theorem archer_probability (p_a p_b : ℝ) (h_p_a : p_a = 1/3) (h_p_b : p_b = 1/2) :
  1 - p_a * p_b = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_archer_probability_l2930_293098


namespace NUMINAMATH_CALUDE_randy_blocks_theorem_l2930_293001

/-- The number of blocks Randy used to build a tower -/
def blocks_used : ℕ := 25

/-- The number of blocks Randy has left -/
def blocks_left : ℕ := 72

/-- The initial number of blocks Randy had -/
def initial_blocks : ℕ := blocks_used + blocks_left

theorem randy_blocks_theorem : initial_blocks = 97 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_theorem_l2930_293001


namespace NUMINAMATH_CALUDE_cyclic_fraction_product_l2930_293044

theorem cyclic_fraction_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : (x + y) / z = (y + z) / x) (h2 : (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 := by
sorry

end NUMINAMATH_CALUDE_cyclic_fraction_product_l2930_293044


namespace NUMINAMATH_CALUDE_M_congruent_to_1_mod_47_l2930_293080

def M : ℕ := sorry -- Definition of M as the 81-digit number

theorem M_congruent_to_1_mod_47 :
  M % 47 = 1 := by sorry

end NUMINAMATH_CALUDE_M_congruent_to_1_mod_47_l2930_293080


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2930_293010

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- State the theorem
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), parabola x y ∧ hyperbola x y a b) →
  (∀ (x y : ℝ), asymptotes x y) →
  (∀ (x y : ℝ), hyperbola x y 1 (Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2930_293010


namespace NUMINAMATH_CALUDE_pentagonal_field_fencing_cost_l2930_293032

/-- Represents the cost of fencing for a pentagonal field -/
def fencing_cost (side1 side2 side3 side4 side5 rate_a rate_b rate_c : ℝ) : ℝ × ℝ × ℝ :=
  let perimeter := side1 + side2 + side3 + side4 + side5
  (perimeter * rate_a, perimeter * rate_b, perimeter * rate_c)

/-- Theorem stating the correct fencing costs for the given pentagonal field -/
theorem pentagonal_field_fencing_cost :
  fencing_cost 25 35 40 45 50 3.5 2.25 1.5 = (682.5, 438.75, 292.5) := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_field_fencing_cost_l2930_293032


namespace NUMINAMATH_CALUDE_range_of_m_l2930_293061

open Set

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x | -7 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 7}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 3*m - 2}

-- Theorem statement
theorem range_of_m (m : ℝ) : A ∩ B m = B m → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2930_293061


namespace NUMINAMATH_CALUDE_cost_of_500_candies_l2930_293063

def candies_per_box : ℕ := 20
def cost_per_box : ℚ := 8
def discount_percentage : ℚ := 0.1
def discount_threshold : ℕ := 400
def order_size : ℕ := 500

theorem cost_of_500_candies : 
  let boxes_needed : ℕ := order_size / candies_per_box
  let total_cost : ℚ := boxes_needed * cost_per_box
  let discount : ℚ := if order_size > discount_threshold then discount_percentage * total_cost else 0
  let final_cost : ℚ := total_cost - discount
  final_cost = 180 := by sorry

end NUMINAMATH_CALUDE_cost_of_500_candies_l2930_293063


namespace NUMINAMATH_CALUDE_sonika_deposit_l2930_293018

theorem sonika_deposit (P R : ℝ) : 
  (P + (P * R * 3) / 100 = 11200) → 
  (P + (P * (R + 2) * 3) / 100 = 11680) → 
  P = 8000 := by
  sorry

end NUMINAMATH_CALUDE_sonika_deposit_l2930_293018


namespace NUMINAMATH_CALUDE_f_non_monotonic_l2930_293091

/-- A piecewise function f defined on ℝ with a parameter a and a split point t -/
noncomputable def f (a t : ℝ) (x : ℝ) : ℝ :=
  if x ≤ t then (2*a - 1)*x + 3*a - 4 else x^3 - x

/-- The theorem stating the condition for non-monotonicity of f -/
theorem f_non_monotonic (a : ℝ) :
  (∀ t : ℝ, ¬ Monotone (f a t)) ↔ a ≤ (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_f_non_monotonic_l2930_293091


namespace NUMINAMATH_CALUDE_percent_to_decimal_l2930_293082

theorem percent_to_decimal (p : ℚ) : p / 100 = p / 100 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_l2930_293082


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2930_293050

variable (p : ℝ)

theorem polynomial_simplification :
  (6 * p^4 + 2 * p^3 - 8 * p + 9) + (-3 * p^3 + 7 * p^2 - 5 * p - 1) =
  6 * p^4 - p^3 + 7 * p^2 - 13 * p + 8 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2930_293050


namespace NUMINAMATH_CALUDE_shirt_price_proof_l2930_293069

theorem shirt_price_proof (shirt_price pants_price : ℝ) 
  (h1 : shirt_price ≠ pants_price)
  (h2 : 2 * shirt_price + 3 * pants_price = 120)
  (h3 : 3 * pants_price = 0.25 * 120) : 
  shirt_price = 45 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_proof_l2930_293069


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2930_293066

-- Problem 1
theorem problem_1 : -4.7 + 0.9 = -3.8 := by sorry

-- Problem 2
theorem problem_2 : -1/2 - (-1/3) = -1/6 := by sorry

-- Problem 3
theorem problem_3 : (-1 - 1/9) * (-0.6) = 2/3 := by sorry

-- Problem 4
theorem problem_4 : 0 * (-5) = 0 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2930_293066


namespace NUMINAMATH_CALUDE_product_xyz_is_zero_l2930_293094

theorem product_xyz_is_zero 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 1) 
  (h3 : y ≠ 0) 
  (h4 : z ≠ 0) : 
  x * y * z = 0 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_is_zero_l2930_293094


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2930_293030

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_product (n : ℕ) : ℕ :=
  (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def valid_number (n : ℕ) : Prop :=
  is_four_digit n ∧ digit_product n = 18

theorem count_valid_numbers : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, valid_number n) ∧ S.card = 24 ∧ 
  (∀ m : ℕ, valid_number m → m ∈ S) :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2930_293030


namespace NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l2930_293093

theorem recurring_decimal_fraction_sum (a b : ℕ+) :
  (a : ℚ) / (b : ℚ) = 36 / 99 →
  Nat.gcd a b = 1 →
  a + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_fraction_sum_l2930_293093


namespace NUMINAMATH_CALUDE_billboard_average_is_twenty_l2930_293067

/-- Calculates the average number of billboards seen per hour given the counts for three consecutive hours. -/
def average_billboards (hour1 hour2 hour3 : ℕ) : ℚ :=
  (hour1 + hour2 + hour3 : ℚ) / 3

/-- Theorem stating that the average number of billboards seen per hour is 20 given the specific counts. -/
theorem billboard_average_is_twenty :
  average_billboards 17 20 23 = 20 := by
  sorry

end NUMINAMATH_CALUDE_billboard_average_is_twenty_l2930_293067


namespace NUMINAMATH_CALUDE_expression_value_approximation_l2930_293005

theorem expression_value_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |((85 : ℝ) + Real.sqrt 32 / 113) * 113^2 - 10246| < ε :=
sorry

end NUMINAMATH_CALUDE_expression_value_approximation_l2930_293005


namespace NUMINAMATH_CALUDE_sequence_difference_l2930_293027

def arithmetic_sum (a₁ aₙ : ℤ) (n : ℕ) : ℤ := n * (a₁ + aₙ) / 2

def sequence_1_sum : ℤ := arithmetic_sum 2 2021 674
def sequence_2_sum : ℤ := arithmetic_sum 3 2022 674

theorem sequence_difference : sequence_1_sum - sequence_2_sum = -544 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l2930_293027


namespace NUMINAMATH_CALUDE_expression_equality_l2930_293059

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem expression_equality : (-8)^(1/3) + π^0 + lg 4 + lg 25 = 1 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l2930_293059


namespace NUMINAMATH_CALUDE_probability_two_heads_in_four_tosses_l2930_293045

-- Define the number of coin tosses
def n : ℕ := 4

-- Define the number of heads we're looking for
def k : ℕ := 2

-- Define the probability of getting heads on a single toss
def p : ℚ := 1/2

-- Define the probability of getting tails on a single toss
def q : ℚ := 1/2

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of getting exactly k heads in n tosses
def probability_k_heads (n k : ℕ) (p q : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * q^(n-k)

-- Theorem statement
theorem probability_two_heads_in_four_tosses :
  probability_k_heads n k p q = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_heads_in_four_tosses_l2930_293045


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2930_293046

theorem quadratic_equation_result (x : ℝ) : 
  7 * x^2 - 2 * x - 4 = 4 * x + 11 → (5 * x - 7)^2 = 570 / 49 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2930_293046


namespace NUMINAMATH_CALUDE_homework_assignment_question_distribution_l2930_293056

theorem homework_assignment_question_distribution :
  ∃! (x y z : ℕ),
    x + y + z = 100 ∧
    (0.5 : ℝ) * x + 3 * y + 10 * z = 100 ∧
    x = 80 ∧ y = 20 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_homework_assignment_question_distribution_l2930_293056


namespace NUMINAMATH_CALUDE_xyz_product_l2930_293068

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 3 * y = -9)
  (eq2 : y * z + 3 * z = -9)
  (eq3 : z * x + 3 * x = -9) :
  x * y * z = 27 := by sorry

end NUMINAMATH_CALUDE_xyz_product_l2930_293068


namespace NUMINAMATH_CALUDE_parabola_directrix_l2930_293002

/-- Given a parabola with equation y = -3x^2 + 6x - 5, its directrix is y = -23/12 -/
theorem parabola_directrix (x y : ℝ) : 
  y = -3 * x^2 + 6 * x - 5 →
  ∃ (k : ℝ), k = -23/12 ∧ k = y - (1/(4 * -3)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2930_293002


namespace NUMINAMATH_CALUDE_largest_quantity_l2930_293053

def A : ℚ := 2010 / 2009 + 2010 / 2011
def B : ℚ := 2010 / 2011 + 2012 / 2011
def C : ℚ := 2011 / 2010 + 2011 / 2012

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l2930_293053


namespace NUMINAMATH_CALUDE_valid_distributions_count_l2930_293065

def number_of_valid_distributions : ℕ :=
  (Finset.filter (fun d => d ≠ 1 ∧ d ≠ 360) (Nat.divisors 360)).card

theorem valid_distributions_count : number_of_valid_distributions = 22 := by
  sorry

end NUMINAMATH_CALUDE_valid_distributions_count_l2930_293065


namespace NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l2930_293017

theorem circle_radius_from_longest_chord (c : Real) (h : c > 0) : 
  ∃ (r : Real), r > 0 ∧ r = c / 2 := by sorry

end NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l2930_293017


namespace NUMINAMATH_CALUDE_last_two_digits_1976_power_100_l2930_293071

theorem last_two_digits_1976_power_100 : 
  1976^100 % 100 = 76 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_1976_power_100_l2930_293071


namespace NUMINAMATH_CALUDE_complex_power_2017_l2930_293057

theorem complex_power_2017 : ((1 - Complex.I) / (1 + Complex.I)) ^ 2017 = -Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_2017_l2930_293057


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2930_293078

theorem smallest_number_divisible (n : ℕ) : n = 257 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ : ℕ, 
    m + 7 = 8 * k₁ ∧ 
    m + 7 = 11 * k₂ ∧ 
    m + 7 = 24 * k₃)) ∧ 
  (∃ k₁ k₂ k₃ : ℕ, 
    n + 7 = 8 * k₁ ∧ 
    n + 7 = 11 * k₂ ∧ 
    n + 7 = 24 * k₃) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2930_293078


namespace NUMINAMATH_CALUDE_impossible_arrangement_l2930_293024

-- Define a 3x3 grid as a function from (Fin 3 × Fin 3) to ℕ
def Grid := Fin 3 → Fin 3 → ℕ

-- Define a predicate to check if a number is between 1 and 9
def InRange (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

-- Define a predicate to check if a grid contains all numbers from 1 to 9
def ContainsAllNumbers (g : Grid) : Prop :=
  ∀ n, InRange n → ∃ i j, g i j = n

-- Define a predicate to check if the product of numbers in a row is a multiple of 4
def RowProductMultipleOf4 (g : Grid) : Prop :=
  ∀ i, (g i 0) * (g i 1) * (g i 2) % 4 = 0

-- Define a predicate to check if the product of numbers in a column is a multiple of 4
def ColProductMultipleOf4 (g : Grid) : Prop :=
  ∀ j, (g 0 j) * (g 1 j) * (g 2 j) % 4 = 0

-- The main theorem
theorem impossible_arrangement : ¬∃ (g : Grid),
  ContainsAllNumbers g ∧ 
  RowProductMultipleOf4 g ∧ 
  ColProductMultipleOf4 g :=
sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l2930_293024


namespace NUMINAMATH_CALUDE_carpet_width_l2930_293083

/-- Calculates the width of a carpet given room dimensions and carpeting costs -/
theorem carpet_width
  (room_length : ℝ)
  (room_breadth : ℝ)
  (carpet_cost_paisa : ℝ)
  (total_cost_rupees : ℝ)
  (h1 : room_length = 15)
  (h2 : room_breadth = 6)
  (h3 : carpet_cost_paisa = 30)
  (h4 : total_cost_rupees = 36)
  : ∃ (carpet_width : ℝ), carpet_width = 75 := by
  sorry

end NUMINAMATH_CALUDE_carpet_width_l2930_293083


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2930_293079

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - 11 * i) / (3 - 4 * i) = 2 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2930_293079


namespace NUMINAMATH_CALUDE_exists_transformation_458_to_14_l2930_293041

-- Define the operations
def double (n : ℕ) : ℕ := 2 * n

def erase_last_digit (n : ℕ) : ℕ :=
  if n < 10 then n else n / 10

-- Define a single step transformation
inductive Step
| Double : Step
| EraseLastDigit : Step

def apply_step (n : ℕ) (s : Step) : ℕ :=
  match s with
  | Step.Double => double n
  | Step.EraseLastDigit => erase_last_digit n

-- Define a sequence of steps
def apply_steps (n : ℕ) (steps : List Step) : ℕ :=
  steps.foldl apply_step n

-- Theorem statement
theorem exists_transformation_458_to_14 :
  ∃ (steps : List Step), apply_steps 458 steps = 14 := by
  sorry

end NUMINAMATH_CALUDE_exists_transformation_458_to_14_l2930_293041


namespace NUMINAMATH_CALUDE_expand_expression_l2930_293036

theorem expand_expression (x : ℝ) : (5 * x^2 + 3) * 4 * x^3 = 20 * x^5 + 12 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2930_293036


namespace NUMINAMATH_CALUDE_ellipse_second_focus_x_coordinate_l2930_293006

-- Define the ellipse properties
structure Ellipse where
  inFirstQuadrant : Bool
  tangentToXAxis : Bool
  tangentToYAxis : Bool
  focus1 : ℝ × ℝ
  tangentToY1 : Bool

-- Define the theorem
theorem ellipse_second_focus_x_coordinate
  (e : Ellipse)
  (h1 : e.inFirstQuadrant = true)
  (h2 : e.tangentToXAxis = true)
  (h3 : e.tangentToYAxis = true)
  (h4 : e.focus1 = (4, 9))
  (h5 : e.tangentToY1 = true) :
  ∃ d : ℝ, d = 16 ∧ (∃ y : ℝ, (d, y) = e.focus1 ∨ (d, 9) ≠ e.focus1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_second_focus_x_coordinate_l2930_293006


namespace NUMINAMATH_CALUDE_molecular_weight_15_C2H5Cl_12_O2_l2930_293014

/-- Calculates the molecular weight of a given number of moles of C2H5Cl and O2 -/
def molecularWeight (moles_C2H5Cl : ℝ) (moles_O2 : ℝ) : ℝ :=
  let atomic_weight_C := 12.01
  let atomic_weight_H := 1.01
  let atomic_weight_Cl := 35.45
  let atomic_weight_O := 16.00
  let mw_C2H5Cl := 2 * atomic_weight_C + 5 * atomic_weight_H + atomic_weight_Cl
  let mw_O2 := 2 * atomic_weight_O
  moles_C2H5Cl * mw_C2H5Cl + moles_O2 * mw_O2

theorem molecular_weight_15_C2H5Cl_12_O2 :
  molecularWeight 15 12 = 1351.8 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_15_C2H5Cl_12_O2_l2930_293014


namespace NUMINAMATH_CALUDE_mo_tea_consumption_l2930_293029

/-- Represents Mo's drinking habits and weather conditions for a week -/
structure MoDrinkingHabits where
  n : ℕ  -- number of hot chocolate cups on rainy mornings
  t : ℕ  -- number of tea cups on non-rainy mornings
  rainyDays : ℕ
  nonRainyDays : ℕ

/-- Theorem stating Mo's tea consumption on non-rainy mornings -/
theorem mo_tea_consumption (habits : MoDrinkingHabits) : habits.t = 4 :=
  by
  have h1 : habits.rainyDays = 2 := by sorry
  have h2 : habits.nonRainyDays = 7 - habits.rainyDays := by sorry
  have h3 : habits.n * habits.rainyDays + habits.t * habits.nonRainyDays = 26 := by sorry
  have h4 : habits.t * habits.nonRainyDays = habits.n * habits.rainyDays + 14 := by sorry
  sorry

#check mo_tea_consumption

end NUMINAMATH_CALUDE_mo_tea_consumption_l2930_293029


namespace NUMINAMATH_CALUDE_eight_possible_rankings_l2930_293084

/-- Represents a player in the tournament -/
inductive Player : Type
| X : Player
| Y : Player
| Z : Player
| W : Player

/-- Represents a match between two players -/
structure Match :=
  (player1 : Player)
  (player2 : Player)

/-- Represents the tournament structure -/
structure Tournament :=
  (day1_match1 : Match)
  (day1_match2 : Match)
  (no_draws : Bool)

/-- Represents a final ranking of players -/
def Ranking := List Player

/-- Function to generate all possible rankings given a tournament structure -/
def generateRankings (t : Tournament) : List Ranking :=
  sorry

/-- Theorem stating that there are exactly 8 possible ranking sequences -/
theorem eight_possible_rankings (t : Tournament) 
  (h1 : t.day1_match1 = ⟨Player.X, Player.Y⟩)
  (h2 : t.day1_match2 = ⟨Player.Z, Player.W⟩)
  (h3 : t.no_draws = true)
  (h4 : (generateRankings t).length > 0)
  (h5 : [Player.X, Player.Z, Player.Y, Player.W] ∈ generateRankings t) :
  (generateRankings t).length = 8 :=
sorry

end NUMINAMATH_CALUDE_eight_possible_rankings_l2930_293084


namespace NUMINAMATH_CALUDE_total_washing_time_l2930_293076

/-- The time William spends washing a normal car -/
def normal_car_time : ℕ := 4 + 7 + 4 + 9

/-- The number of normal cars William washed -/
def normal_cars : ℕ := 2

/-- The number of SUVs William washed -/
def suvs : ℕ := 1

/-- The time multiplier for washing an SUV compared to a normal car -/
def suv_time_multiplier : ℕ := 2

/-- Theorem: William spent 96 minutes washing all vehicles -/
theorem total_washing_time : 
  normal_car_time * normal_cars + normal_car_time * suv_time_multiplier * suvs = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_washing_time_l2930_293076


namespace NUMINAMATH_CALUDE_find_divisor_l2930_293070

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 16698) (h2 : quotient = 89) (h3 : remainder = 14) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 187 :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l2930_293070


namespace NUMINAMATH_CALUDE_set_union_problem_l2930_293000

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {1, 2^a}
def B (a b : ℝ) : Set ℝ := {a, b}

-- State the theorem
theorem set_union_problem (a b : ℝ) : 
  A a ∩ B a b = {1/2} → A a ∪ B a b = {-1, 1/2, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l2930_293000


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2930_293028

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem x_eq_2_sufficient_not_necessary :
  let a : ℝ → ℝ × ℝ := λ x ↦ (1, x)
  let b : ℝ → ℝ × ℝ := λ x ↦ (x, 4)
  (∀ x, x = 2 → parallel (a x) (b x)) ∧
  ¬(∀ x, parallel (a x) (b x) → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2930_293028


namespace NUMINAMATH_CALUDE_charlyn_visible_area_l2930_293015

/-- The length of one side of the square in kilometers -/
def square_side : ℝ := 5

/-- The visibility range in kilometers -/
def visibility_range : ℝ := 1

/-- The area of the region Charlyn can see during her walk -/
noncomputable def visible_area : ℝ :=
  (square_side + 2 * visibility_range) ^ 2 - (square_side - 2 * visibility_range) ^ 2 + Real.pi * visibility_range ^ 2

theorem charlyn_visible_area :
  ‖visible_area - 43.14‖ < 0.01 :=
sorry

end NUMINAMATH_CALUDE_charlyn_visible_area_l2930_293015


namespace NUMINAMATH_CALUDE_max_value_of_f_l2930_293020

noncomputable def f (t : ℝ) : ℝ := (3^t - 4*t)*t / 9^t

theorem max_value_of_f :
  ∃ (max : ℝ), max = 1/16 ∧ ∀ (t : ℝ), f t ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2930_293020


namespace NUMINAMATH_CALUDE_complex_arithmetic_simplification_l2930_293037

theorem complex_arithmetic_simplification :
  (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) = 1 - 15 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_simplification_l2930_293037


namespace NUMINAMATH_CALUDE_fraction_exponent_equality_l2930_293040

theorem fraction_exponent_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / y)^(-3/4 : ℝ) = 4 * (y / x)^3 := by sorry

end NUMINAMATH_CALUDE_fraction_exponent_equality_l2930_293040


namespace NUMINAMATH_CALUDE_largest_pile_size_l2930_293023

theorem largest_pile_size (total : ℕ) (small medium large : ℕ) : 
  total = small + medium + large →
  medium = 2 * small →
  large = 3 * small →
  total = 240 →
  large = 120 := by
sorry

end NUMINAMATH_CALUDE_largest_pile_size_l2930_293023


namespace NUMINAMATH_CALUDE_bears_per_shelf_l2930_293077

theorem bears_per_shelf (initial_stock : ℕ) (new_shipment : ℕ) (num_shelves : ℕ) :
  initial_stock = 4 →
  new_shipment = 10 →
  num_shelves = 2 →
  (initial_stock + new_shipment) / num_shelves = 7 :=
by sorry

end NUMINAMATH_CALUDE_bears_per_shelf_l2930_293077


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l2930_293012

theorem arctan_equation_solution :
  ∀ x : ℝ, 3 * Real.arctan (1/4) + Real.arctan (1/5) + Real.arctan (1/x) = π/4 → x = -250/37 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l2930_293012


namespace NUMINAMATH_CALUDE_box_sum_equals_sixteen_l2930_293058

def box (a b c : ℤ) : ℚ := (a ^ b : ℚ) + (b ^ c : ℚ) - (c ^ a : ℚ)

theorem box_sum_equals_sixteen : box 2 3 (-1) + box (-1) 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_box_sum_equals_sixteen_l2930_293058


namespace NUMINAMATH_CALUDE_university_theater_sales_l2930_293025

/-- The total money made from ticket sales at University Theater --/
def total_money_made (total_tickets : ℕ) (adult_price senior_price : ℕ) (senior_tickets : ℕ) : ℕ :=
  let adult_tickets := total_tickets - senior_tickets
  adult_tickets * adult_price + senior_tickets * senior_price

/-- Theorem: The University Theater made $8748 from ticket sales --/
theorem university_theater_sales : total_money_made 510 21 15 327 = 8748 := by
  sorry

end NUMINAMATH_CALUDE_university_theater_sales_l2930_293025


namespace NUMINAMATH_CALUDE_mileage_pay_is_104_l2930_293016

/-- Calculates the mileage pay for a delivery driver given the distances for three packages and the pay rate per mile. -/
def calculate_mileage_pay (first_package : ℝ) (second_package : ℝ) (third_package : ℝ) (pay_rate : ℝ) : ℝ :=
  (first_package + second_package + third_package) * pay_rate

/-- Theorem stating that given specific package distances and pay rate, the mileage pay is $104. -/
theorem mileage_pay_is_104 :
  let first_package : ℝ := 10
  let second_package : ℝ := 28
  let third_package : ℝ := second_package / 2
  let pay_rate : ℝ := 2
  calculate_mileage_pay first_package second_package third_package pay_rate = 104 := by
  sorry

#check mileage_pay_is_104

end NUMINAMATH_CALUDE_mileage_pay_is_104_l2930_293016


namespace NUMINAMATH_CALUDE_compounded_growth_rate_l2930_293087

/-- Given an initial investment P that grows by k% in the first year and m% in the second year,
    the compounded rate of growth R after two years is equal to k + m + (km/100). -/
theorem compounded_growth_rate (P k m : ℝ) (hP : P > 0) (hk : k ≥ 0) (hm : m ≥ 0) :
  let R := k + m + (k * m) / 100
  let growth_factor := (1 + k / 100) * (1 + m / 100)
  R = (growth_factor - 1) * 100 :=
by sorry

end NUMINAMATH_CALUDE_compounded_growth_rate_l2930_293087


namespace NUMINAMATH_CALUDE_equation_solution_l2930_293095

theorem equation_solution (a c : ℝ) :
  let x := (c^2 - a^3) / (3*a^2 - 1)
  x^2 + c^2 = (a - x)^3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2930_293095


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2930_293085

-- Define the inequality system
def inequality_system (x k : ℝ) : Prop :=
  (2 * x + 9 > 6 * x + 1) ∧ (x - k < 1)

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < 2

-- Theorem statement
theorem inequality_system_solution (k : ℝ) :
  (∀ x, inequality_system x k ↔ solution_set x) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2930_293085


namespace NUMINAMATH_CALUDE_bennys_books_l2930_293090

/-- Given the number of books Sandy, Tim, and the total, find Benny's books --/
theorem bennys_books (sandy_books tim_books total_books : ℕ) 
  (h1 : sandy_books = 10)
  (h2 : tim_books = 33)
  (h3 : total_books = 67)
  (h4 : total_books = sandy_books + tim_books + benny_books) :
  benny_books = 24 := by
  sorry

end NUMINAMATH_CALUDE_bennys_books_l2930_293090


namespace NUMINAMATH_CALUDE_sports_club_overlap_l2930_293007

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h_total : total = 150)
  (h_badminton : badminton = 75)
  (h_tennis : tennis = 60)
  (h_neither : neither = 25) :
  badminton + tennis - (total - neither) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l2930_293007


namespace NUMINAMATH_CALUDE_triangle_angle_A_l2930_293009

theorem triangle_angle_A (b c S_ABC : ℝ) (h1 : b = 8) (h2 : c = 8 * Real.sqrt 3)
  (h3 : S_ABC = 16 * Real.sqrt 3) :
  let A := Real.arcsin (1 / 2)
  A = π / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l2930_293009


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2930_293072

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2930_293072


namespace NUMINAMATH_CALUDE_initial_people_count_l2930_293003

/-- The number of people initially on the train -/
def initial_people : ℕ := sorry

/-- The number of people left on the train after the first stop -/
def people_left : ℕ := 31

/-- The number of people who got off at the first stop -/
def people_off : ℕ := 17

/-- Theorem stating that the initial number of people on the train was 48 -/
theorem initial_people_count : initial_people = people_left + people_off :=
by sorry

end NUMINAMATH_CALUDE_initial_people_count_l2930_293003


namespace NUMINAMATH_CALUDE_date_statistics_order_l2930_293073

def date_counts : List (Nat × Nat) := 
  (List.range 30).map (fun n => (n + 1, 12)) ++ [(31, 7)]

def total_count : Nat := date_counts.foldl (fun acc (_, count) => acc + count) 0

def sum_of_values : Nat := date_counts.foldl (fun acc (date, count) => acc + date * count) 0

def mean : ℚ := sum_of_values / total_count

def median : Nat := 16

def median_of_modes : ℚ := 15.5

theorem date_statistics_order : median_of_modes < mean ∧ mean < median := by sorry

end NUMINAMATH_CALUDE_date_statistics_order_l2930_293073


namespace NUMINAMATH_CALUDE_tax_percentage_calculation_l2930_293064

theorem tax_percentage_calculation (original_cost total_paid : ℝ) 
  (h1 : original_cost = 200)
  (h2 : total_paid = 230) :
  (total_paid - original_cost) / original_cost * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_tax_percentage_calculation_l2930_293064


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2930_293033

theorem min_value_of_expression (x : ℝ) : (x^2 + 8) / Real.sqrt (x^2 + 4) ≥ 4 ∧
  ∃ x₀ : ℝ, (x₀^2 + 8) / Real.sqrt (x₀^2 + 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2930_293033
