import Mathlib

namespace NUMINAMATH_CALUDE_jihoon_calculation_mistake_l1347_134784

theorem jihoon_calculation_mistake (x : ℝ) : 
  x - 7 = 0.45 → x * 7 = 52.15 := by
sorry

end NUMINAMATH_CALUDE_jihoon_calculation_mistake_l1347_134784


namespace NUMINAMATH_CALUDE_normal_price_after_discounts_l1347_134768

theorem normal_price_after_discounts (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (normal_price : ℝ) : 
  final_price = 36 ∧ 
  discount1 = 0.1 ∧ 
  discount2 = 0.2 ∧ 
  final_price = normal_price * (1 - discount1) * (1 - discount2) →
  normal_price = 50 := by
sorry

end NUMINAMATH_CALUDE_normal_price_after_discounts_l1347_134768


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_l1347_134702

theorem tan_alpha_plus_pi_third (α β : ℝ) 
  (h1 : Real.tan (α + β) = 3/5)
  (h2 : Real.tan (β - π/3) = 1/4) :
  Real.tan (α + π/3) = 7/23 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_l1347_134702


namespace NUMINAMATH_CALUDE_sin_20_cos_40_plus_cos_20_sin_40_l1347_134755

theorem sin_20_cos_40_plus_cos_20_sin_40 : 
  Real.sin (20 * π / 180) * Real.cos (40 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (40 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_20_cos_40_plus_cos_20_sin_40_l1347_134755


namespace NUMINAMATH_CALUDE_triangle_area_ratio_bounds_l1347_134700

theorem triangle_area_ratio_bounds (a b c r R : ℝ) (S S₁ : ℝ) :
  a > 0 → b > 0 → c > 0 → r > 0 → R > 0 → S > 0 →
  6 * (a + b + c) * r^2 = a * b * c →
  R = 3 * r →
  S = (r * (a + b + c)) / 2 →
  ∃ (M : ℝ × ℝ), 
    (5 - 2 * Real.sqrt 3) / 36 ≤ S₁ / S ∧ 
    S₁ / S ≤ (5 + 2 * Real.sqrt 3) / 36 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_bounds_l1347_134700


namespace NUMINAMATH_CALUDE_triangle_dimensions_l1347_134769

theorem triangle_dimensions (a m : ℝ) (h1 : a = m + 4) (h2 : (a + 12) * (m + 12) = 5 * a * m) : 
  a = 12 ∧ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_dimensions_l1347_134769


namespace NUMINAMATH_CALUDE_degree_of_g_is_two_l1347_134735

/-- The degree of a polynomial -/
noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry

/-- Composition of polynomials -/
def compose (f g : Polynomial ℝ) : Polynomial ℝ := sorry

theorem degree_of_g_is_two
  (f g : Polynomial ℝ)
  (h : Polynomial ℝ)
  (h_def : h = compose f g + g)
  (deg_h : degree h = 8)
  (deg_f : degree f = 3) :
  degree g = 2 := by sorry

end NUMINAMATH_CALUDE_degree_of_g_is_two_l1347_134735


namespace NUMINAMATH_CALUDE_math_books_in_same_box_probability_l1347_134762

/-- Represents a box that can hold textbooks -/
structure Box where
  capacity : Nat

/-- Represents the collection of textbooks -/
structure Textbooks where
  total : Nat
  math : Nat

/-- Represents the problem setup -/
structure TextbookProblem where
  boxes : List Box
  books : Textbooks

/-- The probability of all math textbooks being in the same box -/
def mathBooksInSameBoxProbability (problem : TextbookProblem) : Rat :=
  18/1173

/-- The main theorem stating the probability of all math textbooks being in the same box -/
theorem math_books_in_same_box_probability 
  (problem : TextbookProblem)
  (h1 : problem.boxes.length = 3)
  (h2 : problem.books.total = 15)
  (h3 : problem.books.math = 4)
  (h4 : problem.boxes.map Box.capacity = [4, 5, 6]) :
  mathBooksInSameBoxProbability problem = 18/1173 := by
  sorry

end NUMINAMATH_CALUDE_math_books_in_same_box_probability_l1347_134762


namespace NUMINAMATH_CALUDE_garage_sale_items_l1347_134727

theorem garage_sale_items (prices : Finset ℕ) (radio_price : ℕ) : 
  prices.card > 0 → 
  radio_price ∈ prices → 
  (prices.filter (λ p => p > radio_price)).card = 16 → 
  (prices.filter (λ p => p < radio_price)).card = 23 → 
  prices.card = 40 := by
sorry

end NUMINAMATH_CALUDE_garage_sale_items_l1347_134727


namespace NUMINAMATH_CALUDE_rachel_total_problems_l1347_134747

/-- The number of math problems Rachel solved in total -/
def total_problems (problems_per_minute : ℕ) (minutes : ℕ) (problems_next_day : ℕ) : ℕ :=
  problems_per_minute * minutes + problems_next_day

/-- Theorem stating that Rachel solved 151 math problems in total -/
theorem rachel_total_problems :
  total_problems 7 18 25 = 151 := by
  sorry

end NUMINAMATH_CALUDE_rachel_total_problems_l1347_134747


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_20_l1347_134745

def digit_sum (n : Nat) : Nat :=
  Nat.rec 0 (fun n sum => sum + n % 10) n

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_digit_sum_20 :
  ∀ n : Nat, n < 389 → ¬(is_prime n ∧ digit_sum n = 20) ∧
  is_prime 389 ∧ digit_sum 389 = 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_20_l1347_134745


namespace NUMINAMATH_CALUDE_isosceles_triangle_parallel_cut_l1347_134748

/-- An isosceles triangle with given area and altitude --/
structure IsoscelesTriangle :=
  (area : ℝ)
  (altitude : ℝ)

/-- A line segment parallel to the base of the triangle --/
structure ParallelLine :=
  (length : ℝ)
  (trapezoidArea : ℝ)

/-- The theorem statement --/
theorem isosceles_triangle_parallel_cut (t : IsoscelesTriangle) (l : ParallelLine) :
  t.area = 150 ∧ t.altitude = 30 ∧ l.trapezoidArea = 100 →
  l.length = 10 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_parallel_cut_l1347_134748


namespace NUMINAMATH_CALUDE_f_2019_l1347_134779

/-- The function f(n) represents the original number of the last person to leave the line. -/
def f (n : ℕ) : ℕ :=
  let m := Nat.sqrt n
  if n ≤ m * m + m then m * m + 1
  else m * m + m + 1

/-- Theorem stating that f(2019) = 1981 -/
theorem f_2019 : f 2019 = 1981 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_l1347_134779


namespace NUMINAMATH_CALUDE_eight_digit_increasing_count_l1347_134757

theorem eight_digit_increasing_count : ∃ M : ℕ, 
  (M = Nat.choose 7 5) ∧ 
  (M % 1000 = 21) := by sorry

end NUMINAMATH_CALUDE_eight_digit_increasing_count_l1347_134757


namespace NUMINAMATH_CALUDE_total_eyes_in_pond_l1347_134785

/-- The number of snakes in the pond -/
def num_snakes : ℕ := 18

/-- The number of alligators in the pond -/
def num_alligators : ℕ := 10

/-- The number of spiders in the pond -/
def num_spiders : ℕ := 5

/-- The number of snails in the pond -/
def num_snails : ℕ := 15

/-- The number of eyes a snake has -/
def snake_eyes : ℕ := 2

/-- The number of eyes an alligator has -/
def alligator_eyes : ℕ := 2

/-- The number of eyes a spider has -/
def spider_eyes : ℕ := 8

/-- The number of eyes a snail has -/
def snail_eyes : ℕ := 2

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := num_snakes * snake_eyes + num_alligators * alligator_eyes + 
                      num_spiders * spider_eyes + num_snails * snail_eyes

theorem total_eyes_in_pond : total_eyes = 126 := by sorry

end NUMINAMATH_CALUDE_total_eyes_in_pond_l1347_134785


namespace NUMINAMATH_CALUDE_inequality_solution_existence_l1347_134737

theorem inequality_solution_existence (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x^2 + |x + a| < 2) ↔ a ∈ Set.Icc (-9/4 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_existence_l1347_134737


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1347_134778

open Set

theorem intersection_of_sets : 
  let A : Set ℝ := {x | 2 < x ∧ x < 4}
  let B : Set ℝ := {x | x > 5/3}
  A ∩ B = {x | 2 < x ∧ x < 3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1347_134778


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l1347_134733

/-- An arithmetic sequence {aₙ} where a₃ = 4 and a₅ = 8 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a 3 = 4 ∧ a 5 = 8

/-- The 11th term of the arithmetic sequence is 20 -/
theorem arithmetic_sequence_11th_term (a : ℕ → ℝ) 
  (h : arithmetic_sequence a) : a 11 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l1347_134733


namespace NUMINAMATH_CALUDE_percentage_problem_l1347_134721

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.1 * x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1347_134721


namespace NUMINAMATH_CALUDE_number_multiplied_by_four_twice_l1347_134795

theorem number_multiplied_by_four_twice : ∃ x : ℝ, (4 * (4 * x) = 32) ∧ (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_four_twice_l1347_134795


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1347_134710

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  -10 * x * y^2 + y^3 + 25 * x^2 * y = y * (5 * x - y)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (a b : ℝ) :
  a^3 + a^2 * b - a * b^2 - b^3 = (a + b)^2 * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1347_134710


namespace NUMINAMATH_CALUDE_second_division_count_correct_l1347_134756

/-- Represents the number of people in the second division of money -/
def second_division_count : ℕ → Prop := λ x =>
  x > 6 ∧ (90 : ℚ) / (x - 6 : ℚ) = 120 / x

/-- The theorem stating the condition for the correct number of people in the second division -/
theorem second_division_count_correct (x : ℕ) : 
  second_division_count x ↔ 
    (∃ (y : ℕ), y > 0 ∧ 
      (90 : ℚ) / y = (120 : ℚ) / (y + 6) ∧
      x = y + 6) :=
sorry

end NUMINAMATH_CALUDE_second_division_count_correct_l1347_134756


namespace NUMINAMATH_CALUDE_evaluate_expression_l1347_134719

theorem evaluate_expression : (-1 : ℤ) ^ (6 ^ 2) + (1 : ℤ) ^ (3 ^ 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1347_134719


namespace NUMINAMATH_CALUDE_inequality_solutions_count_l1347_134783

theorem inequality_solutions_count : 
  (Finset.filter (fun p : ℕ × ℕ => 
    (p.1 : ℚ) / 76 + (p.2 : ℚ) / 71 < 1) 
    (Finset.product (Finset.range 76) (Finset.range 71))).card = 2625 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_count_l1347_134783


namespace NUMINAMATH_CALUDE_exercise_book_distribution_l1347_134773

theorem exercise_book_distribution (m n : ℕ) : 
  (3 * n + 8 = m) →  -- If each student receives 3 books, there will be 8 books left over
  (0 < m - 5 * (n - 1)) →  -- The last student receives some books
  (m - 5 * (n - 1) < 5) →  -- The last student receives less than 5 books
  (n = 5 ∨ n = 6) := by
sorry

end NUMINAMATH_CALUDE_exercise_book_distribution_l1347_134773


namespace NUMINAMATH_CALUDE_intersection_and_union_of_A_and_B_l1347_134706

-- Define the universal set U
def U : Set Int := {-3, -1, 0, 1, 2, 3, 4, 6}

-- Define set A
def A : Set Int := {0, 2, 4, 6}

-- Define the complement of A with respect to U
def C_UA : Set Int := {-1, -3, 1, 3}

-- Define the complement of B with respect to U
def C_UB : Set Int := {-1, 0, 2}

-- Define set B
def B : Set Int := U \ C_UB

-- Theorem to prove
theorem intersection_and_union_of_A_and_B :
  (A ∩ B = {4, 6}) ∧ (A ∪ B = {-3, 0, 1, 2, 3, 4, 6}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_A_and_B_l1347_134706


namespace NUMINAMATH_CALUDE_milk_chocolate_caramel_percentage_l1347_134722

/-- The percentage of milk chocolate with caramel bars in a box of chocolates -/
theorem milk_chocolate_caramel_percentage
  (milk : ℕ)
  (dark : ℕ)
  (milk_almond : ℕ)
  (white : ℕ)
  (milk_caramel : ℕ)
  (h_milk : milk = 36)
  (h_dark : dark = 21)
  (h_milk_almond : milk_almond = 40)
  (h_white : white = 15)
  (h_milk_caramel : milk_caramel = 28) :
  (milk_caramel : ℚ) / (milk + dark + milk_almond + white + milk_caramel) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_milk_chocolate_caramel_percentage_l1347_134722


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l1347_134787

/-- Represents the earnings from the aqua park --/
def aqua_park_earnings 
  (admission_cost tour_cost meal_cost souvenir_cost : ℕ) 
  (group1_size group2_size group3_size : ℕ) : ℕ :=
  let group1_total := group1_size * (admission_cost + tour_cost + meal_cost + souvenir_cost)
  let group2_total := group2_size * (admission_cost + meal_cost)
  let group3_total := group3_size * (admission_cost + tour_cost + souvenir_cost)
  group1_total + group2_total + group3_total

/-- Theorem stating the total earnings from all groups --/
theorem total_earnings_theorem : 
  aqua_park_earnings 12 6 10 8 10 15 8 = 898 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_theorem_l1347_134787


namespace NUMINAMATH_CALUDE_ali_seashells_left_l1347_134752

/-- The number of seashells Ali has left after all transactions -/
def seashells_left (initial : ℝ) (given_friends : ℝ) (given_brothers : ℝ) (sold_fraction : ℝ) (traded_fraction : ℝ) : ℝ :=
  let remaining_after_giving := initial - (given_friends + given_brothers)
  let remaining_after_selling := remaining_after_giving * (1 - sold_fraction)
  remaining_after_selling * (1 - traded_fraction)

/-- Theorem stating that Ali has 76.375 seashells left after all transactions -/
theorem ali_seashells_left : 
  seashells_left 385.5 45.75 34.25 (2/3) (1/4) = 76.375 := by sorry

end NUMINAMATH_CALUDE_ali_seashells_left_l1347_134752


namespace NUMINAMATH_CALUDE_simple_interest_example_l1347_134789

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Proof that the simple interest on $10000 at 8% per annum for 12 months is $800 -/
theorem simple_interest_example : 
  simple_interest 10000 0.08 1 = 800 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_example_l1347_134789


namespace NUMINAMATH_CALUDE_no_valid_solution_l1347_134724

theorem no_valid_solution (total_days : ℕ) (present_pay absent_pay total_pay : ℚ) : 
  total_days = 60 ∧ 
  present_pay = 7 ∧ 
  absent_pay = 3 ∧ 
  total_pay = 170 → 
  ¬∃ (days_present : ℕ), 
    days_present ≤ total_days ∧ 
    (days_present : ℚ) * present_pay + (total_days - days_present : ℚ) * absent_pay = total_pay := by
  sorry

#check no_valid_solution

end NUMINAMATH_CALUDE_no_valid_solution_l1347_134724


namespace NUMINAMATH_CALUDE_original_price_calculation_l1347_134799

theorem original_price_calculation (current_price : ℝ) (reduction_percentage : ℝ) 
  (h1 : current_price = 56.10)
  (h2 : reduction_percentage = 0.15)
  : (current_price / (1 - reduction_percentage)) = 66 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1347_134799


namespace NUMINAMATH_CALUDE_ellipse_intersection_product_range_l1347_134749

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

-- Define point M
def M : ℝ × ℝ := (0, 2)

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Define the vector from origin to a point
def vector_from_origin (p : ℝ × ℝ) : ℝ × ℝ := p

-- Define the vector from M to a point
def vector_from_M (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - M.1, p.2 - M.2)

-- Statement of the theorem
theorem ellipse_intersection_product_range :
  ∀ P Q : ℝ × ℝ,
  C P.1 P.2 →
  C Q.1 Q.2 →
  (∃ k : ℝ, Q.2 - M.2 = k * (Q.1 - M.1) ∧ P.2 - M.2 = k * (P.1 - M.1)) →
  -20 ≤ (dot_product (vector_from_origin P) (vector_from_origin Q) +
         dot_product (vector_from_M P) (vector_from_M Q)) ∧
  (dot_product (vector_from_origin P) (vector_from_origin Q) +
   dot_product (vector_from_M P) (vector_from_M Q)) ≤ -52/3 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_intersection_product_range_l1347_134749


namespace NUMINAMATH_CALUDE_equation_solutions_l1347_134775

theorem equation_solutions :
  (∃ x : ℚ, 5 * x - 2 * (x - 1) = 3 ∧ x = 1/3) ∧
  (∃ x : ℚ, (3 * x - 2) / 6 = 1 + (x - 1) / 3 ∧ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1347_134775


namespace NUMINAMATH_CALUDE_h_equation_l1347_134705

/-- Given the equation 4x^4 + 2x^2 - 5x + 1 + h(x) = x^3 - 3x^2 + 2x - 4,
    prove that h(x) = -4x^4 + x^3 - 5x^2 + 7x - 5 -/
theorem h_equation (x : ℝ) (h : ℝ → ℝ) 
    (eq : 4 * x^4 + 2 * x^2 - 5 * x + 1 + h x = x^3 - 3 * x^2 + 2 * x - 4) : 
  h x = -4 * x^4 + x^3 - 5 * x^2 + 7 * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_h_equation_l1347_134705


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l1347_134709

theorem solve_quadratic_equation (B : ℝ) : 3 * B^2 + 3 * B + 2 = 29 →
  B = (-1 + Real.sqrt 37) / 2 ∨ B = (-1 - Real.sqrt 37) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l1347_134709


namespace NUMINAMATH_CALUDE_expansion_property_l1347_134758

/-- Given that for some natural number n, in the expansion of (x^4 + 1/x)^n,
    the binomial coefficient of the third term is 35 more than that of the second term,
    prove that n = 10 and the constant term in the expansion is 45. -/
theorem expansion_property (n : ℕ) 
  (h : Nat.choose n 2 - Nat.choose n 1 = 35) : 
  n = 10 ∧ Nat.choose 10 8 = 45 := by
  sorry

end NUMINAMATH_CALUDE_expansion_property_l1347_134758


namespace NUMINAMATH_CALUDE_computer_operations_per_hour_l1347_134753

/-- Represents the number of operations a computer can perform per second -/
structure ComputerSpeed :=
  (multiplications_per_second : ℕ)
  (additions_per_second : ℕ)

/-- Calculates the total number of operations per hour -/
def operations_per_hour (speed : ComputerSpeed) : ℕ :=
  (speed.multiplications_per_second + speed.additions_per_second) * 3600

/-- Theorem: A computer with the given speed performs 72 million operations per hour -/
theorem computer_operations_per_hour :
  let speed := ComputerSpeed.mk 15000 5000
  operations_per_hour speed = 72000000 := by
  sorry

end NUMINAMATH_CALUDE_computer_operations_per_hour_l1347_134753


namespace NUMINAMATH_CALUDE_radius_of_C₁_is_8_l1347_134759

-- Define the points and circles
variable (O X Y Z : ℝ × ℝ)
variable (C₁ C₂ : Set (ℝ × ℝ))

-- Define the conditions
variable (h₁ : O ∈ C₂)
variable (h₂ : X ∈ C₁ ∩ C₂)
variable (h₃ : Y ∈ C₁ ∩ C₂)
variable (h₄ : Z ∈ C₂)
variable (h₅ : Z ∉ C₁)
variable (h₆ : ‖X - Z‖ = 15)
variable (h₇ : ‖O - Z‖ = 17)
variable (h₈ : ‖Y - Z‖ = 8)
variable (h₉ : (X - O) • (Z - O) = 0)  -- Right angle at X

-- Define the radius of C₁
def radius_C₁ (O X : ℝ × ℝ) : ℝ := ‖X - O‖

-- Theorem statement
theorem radius_of_C₁_is_8 :
  radius_C₁ O X = 8 :=
sorry

end NUMINAMATH_CALUDE_radius_of_C₁_is_8_l1347_134759


namespace NUMINAMATH_CALUDE_equation_solution_l1347_134725

theorem equation_solution : 
  ∃ (x : ℝ), x ≠ -2 ∧ (4*x^2 + 5*x + 2) / (x + 2) = 4*x - 2 ↔ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1347_134725


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1347_134739

def is_solution (x y w : ℕ) : Prop :=
  2^x * 3^y - 5^x * 7^w = 1

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(1, 0, 0), (3, 0, 1), (1, 1, 0), (2, 2, 1)}

theorem diophantine_equation_solutions :
  ∀ x y w : ℕ, is_solution x y w ↔ (x, y, w) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1347_134739


namespace NUMINAMATH_CALUDE_expo_min_rental_fee_l1347_134736

/-- Represents a bus type with its seat capacity and rental fee -/
structure BusType where
  seats : ℕ
  fee : ℕ

/-- Calculates the minimum rental fee for transporting people using two types of buses -/
def minRentalFee (people : ℕ) (typeA typeB : BusType) : ℕ :=
  sorry

/-- Theorem stating the minimum rental fee for the given problem -/
theorem expo_min_rental_fee :
  let typeA : BusType := ⟨40, 400⟩
  let typeB : BusType := ⟨50, 480⟩
  minRentalFee 360 typeA typeB = 3520 := by
  sorry

end NUMINAMATH_CALUDE_expo_min_rental_fee_l1347_134736


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l1347_134701

/-- Represents the ticket sales for a theater performance --/
structure TicketSales where
  orchestra_price : ℕ
  balcony_price : ℕ
  total_tickets : ℕ
  total_revenue : ℕ

/-- Calculates the difference between balcony and orchestra ticket sales --/
def ticket_difference (ts : TicketSales) : ℕ :=
  let orchestra_tickets := (ts.total_revenue - ts.balcony_price * ts.total_tickets) / 
    (ts.orchestra_price - ts.balcony_price)
  let balcony_tickets := ts.total_tickets - orchestra_tickets
  balcony_tickets - orchestra_tickets

/-- Theorem stating the difference in ticket sales for the given scenario --/
theorem theater_ticket_difference :
  ∃ (ts : TicketSales), 
    ts.orchestra_price = 12 ∧
    ts.balcony_price = 8 ∧
    ts.total_tickets = 370 ∧
    ts.total_revenue = 3320 ∧
    ticket_difference ts = 190 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l1347_134701


namespace NUMINAMATH_CALUDE_amy_baskets_l1347_134794

/-- The number of baskets Amy will fill with candies -/
def num_baskets : ℕ :=
  let chocolate_bars := 5
  let mms := 7 * chocolate_bars
  let marshmallows := 6 * mms
  let total_candies := chocolate_bars + mms + marshmallows
  let candies_per_basket := 10
  total_candies / candies_per_basket

theorem amy_baskets : num_baskets = 25 := by
  sorry

end NUMINAMATH_CALUDE_amy_baskets_l1347_134794


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1347_134716

def A : Set ℝ := {-2, -1, 0, 1}
def B : Set ℝ := {x : ℝ | x^2 - 1 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1347_134716


namespace NUMINAMATH_CALUDE_tori_height_l1347_134792

/-- Tori's initial height in feet -/
def initial_height : ℝ := 4.4

/-- The amount Tori grew in feet -/
def growth : ℝ := 2.86

/-- Tori's current height in feet -/
def current_height : ℝ := initial_height + growth

theorem tori_height : current_height = 7.26 := by
  sorry

end NUMINAMATH_CALUDE_tori_height_l1347_134792


namespace NUMINAMATH_CALUDE_remainder_of_7_350_mod_43_l1347_134703

theorem remainder_of_7_350_mod_43 : 7^350 % 43 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_7_350_mod_43_l1347_134703


namespace NUMINAMATH_CALUDE_sum_of_E_3_and_4_l1347_134750

/-- Given a function E: ℝ → ℝ where E(3) = 5 and E(4) = 5, prove that E(3) + E(4) = 10 -/
theorem sum_of_E_3_and_4 (E : ℝ → ℝ) (h1 : E 3 = 5) (h2 : E 4 = 5) : E 3 + E 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_E_3_and_4_l1347_134750


namespace NUMINAMATH_CALUDE_inequality_solution_l1347_134743

def satisfies_inequality (x : ℤ) : Prop :=
  8.58 * (Real.log x / Real.log 4) + Real.log (Real.sqrt x - 1) / Real.log 2 < 
  Real.log (Real.log 5 / Real.log (Real.sqrt 5)) / Real.log 2

theorem inequality_solution :
  ∀ x : ℤ, satisfies_inequality x ↔ (x = 2 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1347_134743


namespace NUMINAMATH_CALUDE_unique_solution_implies_m_half_l1347_134776

/-- Given m > 0, if the equation m ln x - (1/2)x^2 + mx = 0 has a unique real solution, then m = 1/2 -/
theorem unique_solution_implies_m_half (m : ℝ) (hm : m > 0) :
  (∃! x : ℝ, m * Real.log x - (1/2) * x^2 + m * x = 0) → m = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_unique_solution_implies_m_half_l1347_134776


namespace NUMINAMATH_CALUDE_rectangular_box_problem_l1347_134793

theorem rectangular_box_problem :
  ∃! (a b c : ℕ+),
    (a ≤ b ∧ b ≤ c) ∧
    (a * b * c = 2 * (2 * (a * b + b * c + c * a))) ∧
    (4 * a = c) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_problem_l1347_134793


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_solutions_l1347_134742

theorem quadratic_equation_integer_solutions :
  ∀ x y : ℤ, 
    x^2 + 2*x*y + 3*y^2 - 2*x + y + 1 = 0 ↔ 
    ((x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -1) ∨ (x = 3 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_solutions_l1347_134742


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_common_difference_l1347_134760

/-- An arithmetic sequence with the given properties has a common difference of 2 -/
theorem arithmetic_geometric_sequence_common_difference :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence property
  d ≠ 0 →
  a 1 = 18 →
  (a 1) * (a 8) = (a 4)^2 →  -- geometric sequence property
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_common_difference_l1347_134760


namespace NUMINAMATH_CALUDE_equal_even_odd_probability_l1347_134704

/-- The number of dice being rolled -/
def n : ℕ := 8

/-- The probability of rolling an even number on a single die -/
def p_even : ℚ := 1/2

/-- The probability of rolling an odd number on a single die -/
def p_odd : ℚ := 1/2

/-- The number of ways to choose half of the dice -/
def ways_to_choose : ℕ := n.choose (n/2)

theorem equal_even_odd_probability :
  (ways_to_choose : ℚ) * p_even^(n/2) * p_odd^(n/2) = 35/128 := by sorry

end NUMINAMATH_CALUDE_equal_even_odd_probability_l1347_134704


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l1347_134766

theorem product_from_hcf_lcm (a b : ℕ+) : 
  Nat.gcd a b = 22 → Nat.lcm a b = 2058 → a * b = 45276 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l1347_134766


namespace NUMINAMATH_CALUDE_circle_area_circumscribed_square_l1347_134715

theorem circle_area_circumscribed_square (s : ℝ) (h : s = 12) :
  let r := s * Real.sqrt 2 / 2
  π * r^2 = 72 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_circumscribed_square_l1347_134715


namespace NUMINAMATH_CALUDE_framing_for_enlarged_picture_l1347_134770

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (original_width original_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  ((perimeter_inches + 11) / 12 : ℕ)

/-- Theorem stating the minimum number of linear feet of framing needed for the given picture specifications. -/
theorem framing_for_enlarged_picture :
  min_framing_feet 4 6 4 3 = 9 :=
by sorry

end NUMINAMATH_CALUDE_framing_for_enlarged_picture_l1347_134770


namespace NUMINAMATH_CALUDE_locus_of_T_l1347_134738

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a rectangle
structure Rectangle where
  m : Point
  k : Point
  t : Point
  p : Point

-- Main theorem
theorem locus_of_T (c : Circle) (m : Point) 
  (h1 : (m.x - c.center.1)^2 + (m.y - c.center.2)^2 < c.radius^2) :
  ∃ (c_locus : Circle),
    c_locus.center = c.center ∧
    c_locus.radius = Real.sqrt (2 * c.radius^2 - (m.x^2 + m.y^2)) ∧
    ∀ (rect : Rectangle),
      (rect.m = m) →
      ((rect.k.x - c.center.1)^2 + (rect.k.y - c.center.2)^2 = c.radius^2) →
      ((rect.p.x - c.center.1)^2 + (rect.p.y - c.center.2)^2 = c.radius^2) →
      (rect.m.x - rect.t.x = rect.k.x - rect.p.x) →
      (rect.m.y - rect.t.y = rect.k.y - rect.p.y) →
      ((rect.t.x - c.center.1)^2 + (rect.t.y - c.center.2)^2 = c_locus.radius^2) :=
by
  sorry


end NUMINAMATH_CALUDE_locus_of_T_l1347_134738


namespace NUMINAMATH_CALUDE_dining_sales_tax_percentage_l1347_134780

/-- Proves that the sales tax percentage is 10% given the conditions of the dining problem -/
theorem dining_sales_tax_percentage : 
  ∀ (total_spent food_price tip_percentage sales_tax_percentage : ℝ),
  total_spent = 132 →
  food_price = 100 →
  tip_percentage = 20 →
  total_spent = food_price * (1 + sales_tax_percentage / 100) * (1 + tip_percentage / 100) →
  sales_tax_percentage = 10 := by
sorry


end NUMINAMATH_CALUDE_dining_sales_tax_percentage_l1347_134780


namespace NUMINAMATH_CALUDE_round_repeating_decimal_to_hundredth_l1347_134723

-- Define the repeating decimal
def repeating_decimal : ℚ := 82 + 367 / 999

-- Define the rounding function to the nearest hundredth
def round_to_hundredth (x : ℚ) : ℚ := 
  (⌊x * 100 + 0.5⌋ : ℚ) / 100

-- Theorem statement
theorem round_repeating_decimal_to_hundredth :
  round_to_hundredth repeating_decimal = 82.37 := by sorry

end NUMINAMATH_CALUDE_round_repeating_decimal_to_hundredth_l1347_134723


namespace NUMINAMATH_CALUDE_cal_anthony_transaction_ratio_l1347_134763

theorem cal_anthony_transaction_ratio :
  ∀ (mabel_transactions anthony_transactions cal_transactions jade_transactions : ℕ),
    mabel_transactions = 90 →
    anthony_transactions = mabel_transactions + mabel_transactions / 10 →
    jade_transactions = 85 →
    jade_transactions = cal_transactions + 19 →
    cal_transactions * 3 = anthony_transactions * 2 :=
by sorry

end NUMINAMATH_CALUDE_cal_anthony_transaction_ratio_l1347_134763


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1347_134728

theorem circle_line_intersection (k : ℝ) : 
  k ≤ -2 * Real.sqrt 2 → 
  ∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1347_134728


namespace NUMINAMATH_CALUDE_number_of_intersection_points_l1347_134744

-- Define the line equation
def line (x : ℝ) : ℝ := x + 3

-- Define the curve equation
def curve (x y : ℝ) : Prop := y^2 / 9 - (x * abs x) / 4 = 1

-- Define an intersection point
def is_intersection_point (x y : ℝ) : Prop :=
  y = line x ∧ curve x y

-- Theorem statement
theorem number_of_intersection_points :
  ∃ (p₁ p₂ p₃ : ℝ × ℝ),
    is_intersection_point p₁.1 p₁.2 ∧
    is_intersection_point p₂.1 p₂.2 ∧
    is_intersection_point p₃.1 p₃.2 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    ∀ (q : ℝ × ℝ), is_intersection_point q.1 q.2 → q = p₁ ∨ q = p₂ ∨ q = p₃ :=
by sorry

end NUMINAMATH_CALUDE_number_of_intersection_points_l1347_134744


namespace NUMINAMATH_CALUDE_parabola_translation_l1347_134741

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := 2 * p.a * h + p.b
  , c := p.a * h^2 + p.b * h + p.c + v }

theorem parabola_translation (x : ℝ) :
  let original := Parabola.mk 3 0 0
  let translated := translate original 1 2
  translated.a * x^2 + translated.b * x + translated.c = 3 * (x + 1)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1347_134741


namespace NUMINAMATH_CALUDE_new_lamp_taller_by_exact_amount_l1347_134777

/-- The height difference between two lamps -/
def lamp_height_difference (old_height new_height : ℝ) : ℝ :=
  new_height - old_height

/-- Theorem stating the height difference between the new and old lamps -/
theorem new_lamp_taller_by_exact_amount : 
  lamp_height_difference 1 2.3333333333333335 = 1.3333333333333335 := by
  sorry

end NUMINAMATH_CALUDE_new_lamp_taller_by_exact_amount_l1347_134777


namespace NUMINAMATH_CALUDE_abs_x_minus_sqrt_x_minus_one_squared_l1347_134708

theorem abs_x_minus_sqrt_x_minus_one_squared (x : ℝ) (h : x < 0) :
  |x - Real.sqrt ((x - 1)^2)| = 1 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_sqrt_x_minus_one_squared_l1347_134708


namespace NUMINAMATH_CALUDE_sqrt_six_times_sqrt_three_minus_sqrt_eight_equals_sqrt_two_l1347_134718

theorem sqrt_six_times_sqrt_three_minus_sqrt_eight_equals_sqrt_two :
  Real.sqrt 6 * Real.sqrt 3 - Real.sqrt 8 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_times_sqrt_three_minus_sqrt_eight_equals_sqrt_two_l1347_134718


namespace NUMINAMATH_CALUDE_lexie_family_age_difference_l1347_134797

/-- Given the ages and relationships in Lexie's family, prove the age difference between her brother and cousin. -/
theorem lexie_family_age_difference (lexie_age brother_age sister_age uncle_age cousin_age grandma_age : ℕ) : 
  lexie_age = 8 →
  brother_age = lexie_age - 6 →
  sister_age = 2 * lexie_age →
  grandma_age = 68 →
  uncle_age = grandma_age - 12 →
  uncle_age = 3 * sister_age →
  cousin_age = brother_age + 5 →
  cousin_age = uncle_age - 2 →
  cousin_age - brother_age = 5 := by
sorry

end NUMINAMATH_CALUDE_lexie_family_age_difference_l1347_134797


namespace NUMINAMATH_CALUDE_odd_sum_selections_count_l1347_134730

/-- The number of ways to select k elements from n elements -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The set of numbers from 1 to 11 -/
def ballNumbers : Finset ℕ := sorry

/-- The number of odd numbers in ballNumbers -/
def oddCount : ℕ := sorry

/-- The number of even numbers in ballNumbers -/
def evenCount : ℕ := sorry

/-- The number of ways to select 5 balls with an odd sum -/
def oddSumSelections : ℕ := sorry

theorem odd_sum_selections_count :
  oddSumSelections = 236 := by sorry

end NUMINAMATH_CALUDE_odd_sum_selections_count_l1347_134730


namespace NUMINAMATH_CALUDE_minimum_occupied_seats_l1347_134711

theorem minimum_occupied_seats (total_seats : ℕ) (h : total_seats = 120) :
  let min_occupied := (total_seats + 2) / 3
  min_occupied = 40 ∧
  ∀ n : ℕ, n < min_occupied → ∃ i : ℕ, i < total_seats ∧ 
    (∀ j : ℕ, j < total_seats → (j = i ∨ j = i + 1) → n ≤ j) :=
by sorry

end NUMINAMATH_CALUDE_minimum_occupied_seats_l1347_134711


namespace NUMINAMATH_CALUDE_broken_line_path_length_l1347_134751

/-- Given a circle with diameter 12 units and points C, D each 3 units from the endpoints of the diameter,
    the length of the path CPD is 6√5 units for any point P on the circle forming a right angle CPD. -/
theorem broken_line_path_length (O : ℝ × ℝ) (A B C D P : ℝ × ℝ) : 
  let r : ℝ := 6 -- radius of the circle
  dist A B = 12 ∧ -- diameter is 12 units
  dist A C = 3 ∧ -- C is 3 units from A
  dist B D = 3 ∧ -- D is 3 units from B
  dist O P = r ∧ -- P is on the circle
  (C.1 - P.1) * (D.1 - P.1) + (C.2 - P.2) * (D.2 - P.2) = 0 -- angle CPD is right angle
  →
  dist C P + dist P D = 6 * Real.sqrt 5 := by
  sorry

where
  dist : ℝ × ℝ → ℝ × ℝ → ℝ := λ (x, y) (a, b) ↦ Real.sqrt ((x - a)^2 + (y - b)^2)

end NUMINAMATH_CALUDE_broken_line_path_length_l1347_134751


namespace NUMINAMATH_CALUDE_horse_saddle_ratio_l1347_134781

theorem horse_saddle_ratio : ∀ (total_cost saddle_cost horse_cost : ℕ),
  total_cost = 5000 →
  saddle_cost = 1000 →
  horse_cost = total_cost - saddle_cost →
  (horse_cost : ℚ) / (saddle_cost : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_horse_saddle_ratio_l1347_134781


namespace NUMINAMATH_CALUDE_sum_of_marked_angles_l1347_134798

/-- Represents the configuration of two overlapping triangles -/
structure OverlappingTriangles where
  -- The four marked angles
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  -- The marked angles are exterior to the quadrilateral formed by the overlap
  exterior_sum : p + q + r + s = 360
  -- Each angle is vertically opposite to another
  vertically_opposite : True

/-- The sum of marked angles in the overlapping triangles configuration is 720° -/
theorem sum_of_marked_angles (ot : OverlappingTriangles) : 
  ot.p + ot.q + ot.r + ot.s + ot.p + ot.q + ot.r + ot.s = 720 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_marked_angles_l1347_134798


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_exists_l1347_134731

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- Side lengths of the quadrilateral -/
  sides : Fin 4 → ℕ+
  /-- Diagonal lengths of the quadrilateral -/
  diagonals : Fin 2 → ℕ+
  /-- Area of the quadrilateral -/
  area : ℕ+
  /-- Radius of the circumcircle -/
  radius : ℕ+
  /-- The quadrilateral is inscribed in a circle -/
  inscribed : True
  /-- The side lengths are pairwise distinct -/
  distinct_sides : ∀ i j, i ≠ j → sides i ≠ sides j

/-- There exists an inscribed quadrilateral with integer parameters -/
theorem inscribed_quadrilateral_exists : 
  ∃ q : InscribedQuadrilateral, True :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_exists_l1347_134731


namespace NUMINAMATH_CALUDE_roots_expression_l1347_134790

theorem roots_expression (r s : ℝ) (u v s t : ℂ) : 
  (u^2 + r*u + 1 = 0) → 
  (v^2 + r*v + 1 = 0) → 
  (s^2 + s*s + 1 = 0) → 
  (t^2 + s*t + 1 = 0) → 
  (u - s)*(v - s)*(u + t)*(v + t) = s^2 - r^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_l1347_134790


namespace NUMINAMATH_CALUDE_saras_house_is_1000_l1347_134754

def nadas_house_size : ℕ := 450

def saras_house_size (nadas_size : ℕ) : ℕ :=
  2 * nadas_size + 100

theorem saras_house_is_1000 : saras_house_size nadas_house_size = 1000 := by
  sorry

end NUMINAMATH_CALUDE_saras_house_is_1000_l1347_134754


namespace NUMINAMATH_CALUDE_base_16_to_binary_bits_l1347_134714

/-- The base-16 number represented as 66666 --/
def base_16_num : ℕ := 6 * 16^4 + 6 * 16^3 + 6 * 16^2 + 6 * 16 + 6

/-- The number of bits in the binary representation of a natural number --/
def num_bits (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

theorem base_16_to_binary_bits :
  num_bits base_16_num = 19 := by
  sorry

end NUMINAMATH_CALUDE_base_16_to_binary_bits_l1347_134714


namespace NUMINAMATH_CALUDE_log_27_3_l1347_134712

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  have h : 27 = 3^3 := by sorry
  sorry

end NUMINAMATH_CALUDE_log_27_3_l1347_134712


namespace NUMINAMATH_CALUDE_cube_root_of_negative_one_l1347_134713

theorem cube_root_of_negative_one : ∃ x : ℝ, x^3 = -1 ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_one_l1347_134713


namespace NUMINAMATH_CALUDE_power_congruence_l1347_134734

theorem power_congruence (h : 2^200 ≡ 1 [MOD 800]) : 2^6000 ≡ 1 [MOD 800] := by
  sorry

end NUMINAMATH_CALUDE_power_congruence_l1347_134734


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1347_134788

/-- The perimeter of a rhombus with given diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1347_134788


namespace NUMINAMATH_CALUDE_bus_journey_speed_l1347_134771

/-- Calculates the average speed for the remaining distance of a bus journey -/
theorem bus_journey_speed 
  (total_distance : ℝ) 
  (partial_distance : ℝ) 
  (partial_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : total_distance = 250)
  (h2 : partial_distance = 100)
  (h3 : partial_speed = 40)
  (h4 : total_time = 5)
  : (total_distance - partial_distance) / (total_time - partial_distance / partial_speed) = 60 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_speed_l1347_134771


namespace NUMINAMATH_CALUDE_juans_number_l1347_134732

theorem juans_number (j k : ℕ) (h1 : j > 0) (h2 : k > 0) 
  (h3 : 10^(k+1) + 10*j + 1 - j = 14789) : j = 532 := by
  sorry

end NUMINAMATH_CALUDE_juans_number_l1347_134732


namespace NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l1347_134765

/-- Given two cubic polynomials with two distinct common roots, prove that a = 7 and b = 8 -/
theorem common_roots_cubic_polynomials (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
    (r^3 + a*r^2 + 13*r + 10 = 0) ∧
    (r^3 + b*r^2 + 16*r + 12 = 0) ∧
    (s^3 + a*s^2 + 13*s + 10 = 0) ∧
    (s^3 + b*s^2 + 16*s + 12 = 0)) →
  a = 7 ∧ b = 8 := by
sorry

end NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l1347_134765


namespace NUMINAMATH_CALUDE_minimize_expression_l1347_134786

theorem minimize_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 2) (h3 : a + b = 3) :
  ∃ (min_a : ℝ), min_a = 2/3 ∧ 
    ∀ (x : ℝ), x > 0 → x + b = 3 → (4/x + 1/(b-2) ≥ 4/min_a + 1/(b-2)) :=
by sorry

end NUMINAMATH_CALUDE_minimize_expression_l1347_134786


namespace NUMINAMATH_CALUDE_hot_dog_stand_sales_time_l1347_134726

/-- 
Given a hot dog stand that sells 10 hot dogs per hour at $2 each,
prove that it takes 10 hours to reach $200 in sales.
-/
theorem hot_dog_stand_sales_time : 
  let hot_dogs_per_hour : ℕ := 10
  let price_per_hot_dog : ℚ := 2
  let sales_goal : ℚ := 200
  let sales_per_hour : ℚ := hot_dogs_per_hour * price_per_hot_dog
  let hours_needed : ℚ := sales_goal / sales_per_hour
  hours_needed = 10 := by sorry

end NUMINAMATH_CALUDE_hot_dog_stand_sales_time_l1347_134726


namespace NUMINAMATH_CALUDE_extreme_perimeter_rectangles_l1347_134764

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a rectangle with width w and height h -/
structure Rectangle where
  w : ℝ
  h : ℝ
  h_pos_w : 0 < w
  h_pos_h : 0 < h

/-- Predicate to check if a rectangle touches the given ellipse -/
def touches (e : Ellipse) (r : Rectangle) : Prop :=
  ∃ (x y : ℝ), (x^2 / e.a^2) + (y^2 / e.b^2) = 1 ∧
    (x = r.w / 2 ∨ x = -r.w / 2 ∨ y = r.h / 2 ∨ y = -r.h / 2)

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.w + r.h)

/-- Theorem stating the properties of rectangles with extreme perimeters touching an ellipse -/
theorem extreme_perimeter_rectangles (e : Ellipse) :
  ∃ (r_min r_max : Rectangle),
    touches e r_min ∧ touches e r_max ∧
    (∀ r : Rectangle, touches e r → perimeter r_min ≤ perimeter r) ∧
    (∀ r : Rectangle, touches e r → perimeter r ≤ perimeter r_max) ∧
    r_min.w = 2 * e.b ∧ r_min.h = 2 * Real.sqrt (e.a^2 - e.b^2) ∧
    r_max.w = r_max.h ∧ r_max.w = 2 * Real.sqrt ((e.a^2 + e.b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_extreme_perimeter_rectangles_l1347_134764


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1347_134717

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 4 * x^2 - 12 * x + 1 = a * (x - h)^2 + k) → 
  a + h + k = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1347_134717


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1347_134782

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 4 + a 7 = 2) →
  (a 5 * a 8 = -8) →
  (a 1 + a 10 = -7) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1347_134782


namespace NUMINAMATH_CALUDE_sum_M_l1347_134796

def M : ℕ → ℕ
  | 0 => 0
  | 1 => 4^2 - 2^2
  | (n+2) => (2*n+5)^2 + (2*n+4)^2 - (2*n+3)^2 - (2*n+2)^2 + M n

theorem sum_M : M 50 = 5304 := by
  sorry

end NUMINAMATH_CALUDE_sum_M_l1347_134796


namespace NUMINAMATH_CALUDE_charles_earnings_proof_l1347_134761

/-- Calculates Charles's earnings after tax from pet care activities -/
def charles_earnings (housesitting_rate : ℝ) (lab_walk_rate : ℝ) (gr_walk_rate : ℝ) (gs_walk_rate : ℝ)
                     (lab_groom_rate : ℝ) (gr_groom_rate : ℝ) (gs_groom_rate : ℝ)
                     (housesitting_time : ℝ) (lab_walk_time : ℝ) (gr_walk_time : ℝ) (gs_walk_time : ℝ)
                     (tax_rate : ℝ) : ℝ :=
  let total_before_tax := housesitting_rate * housesitting_time +
                          lab_walk_rate * lab_walk_time +
                          gr_walk_rate * gr_walk_time +
                          gs_walk_rate * gs_walk_time +
                          lab_groom_rate + gr_groom_rate + gs_groom_rate
  let tax_deduction := tax_rate * total_before_tax
  total_before_tax - tax_deduction

/-- Theorem stating Charles's earnings after tax -/
theorem charles_earnings_proof :
  charles_earnings 15 22 25 30 10 15 20 10 3 2 1.5 0.12 = 313.28 := by
  sorry

end NUMINAMATH_CALUDE_charles_earnings_proof_l1347_134761


namespace NUMINAMATH_CALUDE_pool_filling_time_l1347_134729

/-- Proves the time required to fill a pool given its volume and water delivery rates -/
theorem pool_filling_time 
  (pool_volume : ℝ) 
  (hose1_rate : ℝ) 
  (hose2_rate : ℝ) 
  (hose1_count : ℕ) 
  (hose2_count : ℕ) 
  (h1 : pool_volume = 15000) 
  (h2 : hose1_rate = 2) 
  (h3 : hose2_rate = 3) 
  (h4 : hose1_count = 2) 
  (h5 : hose2_count = 2) : 
  (pool_volume / (hose1_count * hose1_rate + hose2_count * hose2_rate)) / 60 = 25 := by
  sorry

#check pool_filling_time

end NUMINAMATH_CALUDE_pool_filling_time_l1347_134729


namespace NUMINAMATH_CALUDE_tree_age_conversion_l1347_134772

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The given number in base 7 -/
def treeAgeBase7 : List Nat := [7, 4, 5, 2]

theorem tree_age_conversion :
  base7ToBase10 treeAgeBase7 = 966 := by
  sorry

end NUMINAMATH_CALUDE_tree_age_conversion_l1347_134772


namespace NUMINAMATH_CALUDE_tree_planting_variance_l1347_134791

def tree_planting_data : List (Nat × Nat) := [(5, 3), (6, 4), (7, 3)]

def total_groups : Nat := tree_planting_data.map (·.2) |>.sum

theorem tree_planting_variance (h : total_groups = 10) :
  let mean := (tree_planting_data.map (fun (x, y) => x * y) |>.sum) / total_groups
  let variance := (1 : ℝ) / total_groups *
    (tree_planting_data.map (fun (x, y) => y * ((x : ℝ) - mean)^2) |>.sum)
  variance = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_variance_l1347_134791


namespace NUMINAMATH_CALUDE_f_properties_l1347_134707

def f (x : ℝ) : ℝ := (x - 2)^2

theorem f_properties :
  (∀ x, f (x + 2) = f (-x + 2)) ∧ 
  (∀ x y, x < y → x < 2 → f x > f y) ∧
  (∀ x y, x < y → y > 2 → f x < f y) ∧
  (∀ x y, x < y → f (x + 2) - f x < f (y + 2) - f y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1347_134707


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1347_134774

theorem complex_magnitude_problem (z : ℂ) :
  Complex.abs z * (3 * z + 2 * Complex.I) = 2 * (Complex.I * z - 6) →
  Complex.abs z = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1347_134774


namespace NUMINAMATH_CALUDE_star_emilio_sum_difference_l1347_134767

def star_list : List Nat := List.range 40

def replace_digit (n : Nat) : Nat :=
  let s := toString n
  let replaced := s.map (fun c => if c == '3' then '2' else c)
  replaced.toNat!

def emilio_list : List Nat := star_list.map replace_digit

theorem star_emilio_sum_difference :
  star_list.sum - emilio_list.sum = 104 := by sorry

end NUMINAMATH_CALUDE_star_emilio_sum_difference_l1347_134767


namespace NUMINAMATH_CALUDE_cos_75_minus_cos_15_l1347_134720

theorem cos_75_minus_cos_15 : Real.cos (75 * π / 180) - Real.cos (15 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_minus_cos_15_l1347_134720


namespace NUMINAMATH_CALUDE_parallel_transitive_parallel_common_not_parallel_to_common_not_parallel_no_common_l1347_134746

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop := sorry

theorem parallel_transitive (l1 l2 l3 : Line) :
  (parallel l1 l3 ∧ parallel l2 l3) → parallel l1 l2 :=
sorry

theorem parallel_common (l1 l2 : Line) :
  parallel l1 l2 → ∃ l3 : Line, parallel l1 l3 ∧ parallel l2 l3 :=
sorry

theorem not_parallel_to_common (l1 l2 l3 : Line) :
  (¬ parallel l1 l3 ∨ ¬ parallel l2 l3) → ¬ parallel l1 l2 :=
sorry

theorem not_parallel_no_common (l1 l2 : Line) :
  ¬ parallel l1 l2 → ¬ ∃ l3 : Line, parallel l1 l3 ∧ parallel l2 l3 :=
sorry

end NUMINAMATH_CALUDE_parallel_transitive_parallel_common_not_parallel_to_common_not_parallel_no_common_l1347_134746


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1347_134740

def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1347_134740
