import Mathlib

namespace NUMINAMATH_CALUDE_complex_cube_equation_positive_integer_components_l2501_250190

theorem complex_cube_equation : ∃ (d : ℤ), (1 + 3*I : ℂ)^3 = -26 + d*I := by sorry

theorem positive_integer_components : (1 : ℤ) > 0 ∧ (3 : ℤ) > 0 := by sorry

end NUMINAMATH_CALUDE_complex_cube_equation_positive_integer_components_l2501_250190


namespace NUMINAMATH_CALUDE_last_locker_opened_l2501_250149

/-- Represents the locker opening pattern described in the problem -/
def lockerOpeningPattern (n : ℕ) : Prop :=
  ∃ (lastLocker : ℕ),
    -- There are 2048 lockers
    n = 2048 ∧
    -- The last locker opened is 2041
    lastLocker = 2041 ∧
    -- The pattern follows the described rules
    (∀ k : ℕ, k ≤ n → ∃ (trip : ℕ),
      -- Each trip opens lockers based on the trip number
      (k % trip = 0 → k ≠ lastLocker) ∧
      -- The last locker is only opened in the final trip
      (k = lastLocker → ∀ j < trip, k % j ≠ 0))

/-- Theorem stating that the last locker opened is 2041 -/
theorem last_locker_opened (n : ℕ) (h : lockerOpeningPattern n) :
  ∃ (lastLocker : ℕ), lastLocker = 2041 ∧ 
  ∀ k : ℕ, k ≤ n → k ≠ lastLocker → ∃ (trip : ℕ), k % trip = 0 :=
  sorry

end NUMINAMATH_CALUDE_last_locker_opened_l2501_250149


namespace NUMINAMATH_CALUDE_point_on_hyperbola_l2501_250100

theorem point_on_hyperbola : 
  let x : ℝ := 2
  let y : ℝ := 3
  y = 6 / x := by sorry

end NUMINAMATH_CALUDE_point_on_hyperbola_l2501_250100


namespace NUMINAMATH_CALUDE_condition_analysis_l2501_250140

theorem condition_analysis (a : ℝ) : 
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l2501_250140


namespace NUMINAMATH_CALUDE_optimal_game_outcome_l2501_250151

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents a strategy for a player -/
def Strategy := List ℤ → ℤ

/-- The game state, including the current sum and remaining numbers -/
structure GameState :=
  (sum : ℤ)
  (remaining : List ℤ)

/-- The result of playing the game with given strategies -/
def playGame (firstStrategy : Strategy) (secondStrategy : Strategy) : ℤ :=
  sorry

/-- An optimal strategy for the first player -/
def optimalFirstStrategy : Strategy :=
  sorry

/-- An optimal strategy for the second player -/
def optimalSecondStrategy : Strategy :=
  sorry

/-- The theorem stating the optimal outcome of the game -/
theorem optimal_game_outcome :
  playGame optimalFirstStrategy optimalSecondStrategy = 30 :=
sorry

end NUMINAMATH_CALUDE_optimal_game_outcome_l2501_250151


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2501_250130

theorem inscribed_circle_radius (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 6) (h₃ : c = 7) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 2 * Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2501_250130


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l2501_250179

theorem max_digits_product_5_4 : 
  ∃ (a b : ℕ), 
    10000 ≤ a ∧ a < 100000 ∧
    1000 ≤ b ∧ b < 10000 ∧
    (∀ (x y : ℕ), 
      10000 ≤ x ∧ x < 100000 ∧ 
      1000 ≤ y ∧ y < 10000 → 
      x * y < 1000000000) ∧
    999999999 < a * b :=
by sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l2501_250179


namespace NUMINAMATH_CALUDE_coefficient_of_x_six_in_expansion_l2501_250101

theorem coefficient_of_x_six_in_expansion (x : ℝ) : 
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ), 
    (2*x^2 + 1)^5 = a₀ + a₁*x^2 + a₂*x^4 + a₃*x^6 + a₄*x^8 + a₅*x^10 ∧ 
    a₃ = 80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_six_in_expansion_l2501_250101


namespace NUMINAMATH_CALUDE_max_consecutive_interesting_l2501_250167

def is_interesting (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p * q

theorem max_consecutive_interesting :
  (∃ a : ℕ, ∀ k : ℕ, k < 3 → is_interesting (a + k)) ∧
  (∀ a : ℕ, ∃ k : ℕ, k < 4 → ¬is_interesting (a + k)) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_interesting_l2501_250167


namespace NUMINAMATH_CALUDE_right_triangle_area_l2501_250104

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 12 →
  angle = 30 * π / 180 →
  let shortest_side := hypotenuse / 2
  let longest_side := hypotenuse / 2 * Real.sqrt 3
  let area := shortest_side * longest_side / 2
  area = 18 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2501_250104


namespace NUMINAMATH_CALUDE_regions_count_l2501_250165

-- Define the points
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (-2, 2)
def C : ℝ × ℝ := (-2, -2)
def D : ℝ × ℝ := (2, -2)
def E : ℝ × ℝ := (1, 0)
def F : ℝ × ℝ := (0, 1)
def G : ℝ × ℝ := (-1, 0)
def H : ℝ × ℝ := (0, -1)

-- Define the set of all points
def points : Set (ℝ × ℝ) := {A, B, C, D, E, F, G, H}

-- Define the square ABCD
def squareABCD : Set (ℝ × ℝ) := {(x, y) | -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2}

-- Define a function to count regions formed by line segments
def countRegions (pts : Set (ℝ × ℝ)) (square : Set (ℝ × ℝ)) : ℕ := sorry

-- Theorem statement
theorem regions_count : countRegions points squareABCD = 60 := by sorry

end NUMINAMATH_CALUDE_regions_count_l2501_250165


namespace NUMINAMATH_CALUDE_quadratic_real_root_l2501_250148

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root if and only if b ∈ (-∞, -10] ∪ [10, ∞) -/
theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l2501_250148


namespace NUMINAMATH_CALUDE_largest_awesome_prime_l2501_250113

def is_awesome_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∀ q : ℕ, 0 < q → q < p → Nat.Prime (p + 2 * q)

theorem largest_awesome_prime : 
  (∃ p : ℕ, is_awesome_prime p) ∧ 
  (∀ p : ℕ, is_awesome_prime p → p ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_largest_awesome_prime_l2501_250113


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2501_250182

theorem matrix_equation_solution :
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 4]
  N^4 - 5 • N^3 + 9 • N^2 - 5 • N = !![6, 12; 3, 6] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2501_250182


namespace NUMINAMATH_CALUDE_total_payment_l2501_250199

/-- The cost of potatoes in yuan per kilogram -/
def potato_cost : ℝ := 1

/-- The cost of celery in yuan per kilogram -/
def celery_cost : ℝ := 0.7

/-- The total cost of buying potatoes and celery -/
def total_cost (a b : ℝ) : ℝ := a * potato_cost + b * celery_cost

theorem total_payment (a b : ℝ) : total_cost a b = a + 0.7 * b := by
  sorry

end NUMINAMATH_CALUDE_total_payment_l2501_250199


namespace NUMINAMATH_CALUDE_perfect_square_k_l2501_250197

theorem perfect_square_k (K : ℕ) (h1 : K > 1) (h2 : 1000 < K^4) (h3 : K^4 < 5000) :
  ∃ (n : ℕ), K^4 = n^2 ↔ K = 6 ∨ K = 7 ∨ K = 8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_k_l2501_250197


namespace NUMINAMATH_CALUDE_complement_of_union_l2501_250194

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union : (U \ (M ∪ N)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l2501_250194


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l2501_250183

-- Part 1
theorem part_one : Real.sqrt 16 + (1 - Real.sqrt 3) ^ 0 - 2⁻¹ = 4.5 := by sorry

-- Part 2
def system_solution (x : ℝ) : Prop :=
  -2 * x + 6 ≥ 4 ∧ (4 * x + 1) / 3 > x - 1

theorem part_two : ∀ x : ℝ, system_solution x ↔ -4 < x ∧ x ≤ 1 := by sorry

-- Part 3
theorem part_three : {x : ℕ | system_solution x} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l2501_250183


namespace NUMINAMATH_CALUDE_olympic_mascots_arrangement_l2501_250185

/-- The number of possible arrangements of 5 items with specific constraints -/
def num_arrangements : ℕ := 16

/-- The number of ways to choose 1 item from 2 -/
def choose_one_from_two : ℕ := 2

/-- The number of ways to arrange 2 items -/
def arrange_two : ℕ := 2

theorem olympic_mascots_arrangement :
  num_arrangements = 2 * choose_one_from_two * choose_one_from_two * arrange_two :=
sorry

end NUMINAMATH_CALUDE_olympic_mascots_arrangement_l2501_250185


namespace NUMINAMATH_CALUDE_emily_has_ten_employees_l2501_250186

/-- Calculates the number of employees Emily has based on salary information. -/
def calculate_employees (emily_original_salary : ℕ) (emily_new_salary : ℕ) 
                        (employee_original_salary : ℕ) (employee_new_salary : ℕ) : ℕ :=
  (emily_original_salary - emily_new_salary) / (employee_new_salary - employee_original_salary)

/-- Proves that Emily has 10 employees given the salary information. -/
theorem emily_has_ten_employees :
  calculate_employees 1000000 850000 20000 35000 = 10 := by
  sorry

end NUMINAMATH_CALUDE_emily_has_ten_employees_l2501_250186


namespace NUMINAMATH_CALUDE_fermat_5_divisible_by_641_fermat_numbers_coprime_l2501_250145

-- Define Fermat numbers
def F (n : ℕ) : ℕ := 2^(2^n) + 1

-- Theorem 1: F_5 is divisible by 641
theorem fermat_5_divisible_by_641 : 
  641 ∣ F 5 := by sorry

-- Theorem 2: F_k and F_n are relatively prime for k ≠ n
theorem fermat_numbers_coprime {k n : ℕ} (h : k ≠ n) : 
  Nat.gcd (F k) (F n) = 1 := by sorry

end NUMINAMATH_CALUDE_fermat_5_divisible_by_641_fermat_numbers_coprime_l2501_250145


namespace NUMINAMATH_CALUDE_may_to_june_increase_l2501_250178

-- Define the percentage changes
def march_to_april_increase : ℝ := 0.10
def april_to_may_decrease : ℝ := 0.20
def overall_increase : ℝ := 0.3200000000000003

-- Define the function to calculate the final value after percentage changes
def final_value (initial : ℝ) (increase1 : ℝ) (decrease : ℝ) (increase2 : ℝ) : ℝ :=
  initial * (1 + increase1) * (1 - decrease) * (1 + increase2)

-- Theorem to prove
theorem may_to_june_increase (initial : ℝ) (initial_pos : initial > 0) :
  ∃ (may_to_june : ℝ), 
    final_value initial march_to_april_increase april_to_may_decrease may_to_june = 
    initial * (1 + overall_increase) ∧ 
    may_to_june = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_may_to_june_increase_l2501_250178


namespace NUMINAMATH_CALUDE_second_class_males_count_l2501_250161

/-- Represents the number of students in each class by gender -/
structure ClassComposition where
  males : ℕ
  females : ℕ

/-- Represents the composition of the three classes -/
structure SquareDancingClasses where
  class1 : ClassComposition
  class2 : ClassComposition
  class3 : ClassComposition

def total_males (classes : SquareDancingClasses) : ℕ :=
  classes.class1.males + classes.class2.males + classes.class3.males

def total_females (classes : SquareDancingClasses) : ℕ :=
  classes.class1.females + classes.class2.females + classes.class3.females

theorem second_class_males_count 
  (classes : SquareDancingClasses)
  (h1 : classes.class1 = ⟨17, 13⟩)
  (h2 : classes.class2.females = 18)
  (h3 : classes.class3 = ⟨15, 17⟩)
  (h4 : total_males classes - total_females classes = 2) :
  classes.class2.males = 18 :=
sorry

end NUMINAMATH_CALUDE_second_class_males_count_l2501_250161


namespace NUMINAMATH_CALUDE_terminating_decimal_fractions_l2501_250139

theorem terminating_decimal_fractions (n : ℕ) : n > 1 → (∃ (a b c d : ℕ), 1 / n = a / (2^b * 5^c) ∧ 1 / (n + 1) = d / (2^b * 5^c)) ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_terminating_decimal_fractions_l2501_250139


namespace NUMINAMATH_CALUDE_area_triangle_parallel_lines_circle_l2501_250150

/-- Given two parallel lines with distance x between them, where one line is tangent 
    to a circle of radius R at point A and the other line intersects the circle at 
    points B and C, the area S of triangle ABC is equal to x √(2Rx - x²). -/
theorem area_triangle_parallel_lines_circle (R x : ℝ) (h : 0 < R ∧ 0 < x ∧ x < 2*R) :
  ∃ (S : ℝ), S = x * Real.sqrt (2 * R * x - x^2) := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_parallel_lines_circle_l2501_250150


namespace NUMINAMATH_CALUDE_range_of_f_l2501_250132

-- Define the function f
def f (x : ℝ) := x^2 - 6*x - 9

-- State the theorem
theorem range_of_f :
  ∃ (a b : ℝ), a < b ∧
  (Set.Icc a b) = {y | ∃ x ∈ Set.Ioo 1 4, f x = y} ∧
  a = -18 ∧ b = -14 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l2501_250132


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2501_250173

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2501_250173


namespace NUMINAMATH_CALUDE_nonreal_roots_product_l2501_250115

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 6*x^3 + 15*x^2 - 20*x = 396) →
  (∃ z w : ℂ, z ≠ w ∧ z.im ≠ 0 ∧ w.im ≠ 0 ∧ 
   (x^4 - 6*x^3 + 15*x^2 - 20*x - 396 = 0 → x = z ∨ x = w) ∧
   z * w = 4 + Real.sqrt 412) :=
by sorry

end NUMINAMATH_CALUDE_nonreal_roots_product_l2501_250115


namespace NUMINAMATH_CALUDE_smallest_cube_with_divisor_l2501_250128

theorem smallest_cube_with_divisor (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r →
  (∀ m : ℕ, m < (p * q * r^2)^3 → ¬(∃ k : ℕ, m = k^3 ∧ p^2 * q^3 * r^5 ∣ m)) →
  (p * q * r^2)^3 = (p * q * r^2)^3 ∧ p^2 * q^3 * r^5 ∣ (p * q * r^2)^3 := by
  sorry

#check smallest_cube_with_divisor

end NUMINAMATH_CALUDE_smallest_cube_with_divisor_l2501_250128


namespace NUMINAMATH_CALUDE_f_value_at_log_half_24_l2501_250133

/-- An odd function with period 2 and specific definition on (0,1) -/
def f (x : ℝ) : ℝ :=
  sorry

theorem f_value_at_log_half_24 :
  ∀ (f : ℝ → ℝ),
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, f (x + 1) = f (x - 1)) →  -- f has period 2
  (∀ x ∈ Set.Ioo 0 1, f x = 2^x - 2) →  -- definition on (0,1)
  f (Real.log 24 / Real.log (1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_log_half_24_l2501_250133


namespace NUMINAMATH_CALUDE_class_duty_assignment_l2501_250107

theorem class_duty_assignment (num_boys num_girls : ℕ) 
  (h1 : num_boys = 16) 
  (h2 : num_girls = 14) : 
  num_boys * num_girls = 224 := by
  sorry

end NUMINAMATH_CALUDE_class_duty_assignment_l2501_250107


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l2501_250136

-- Define the polynomials
def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def j (x : ℝ) : ℝ := x^2 - x - 3

-- State the theorem
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x + j x = -3 * x^2 + 11 * x - 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l2501_250136


namespace NUMINAMATH_CALUDE_local_max_range_l2501_250143

-- Define the function f and its derivative
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * (x + 1) * (x - a)

-- State the theorem
theorem local_max_range (a : ℝ) :
  (∀ x, HasDerivAt f (f_deriv a x) x) →  -- f' is the derivative of f
  (∃ δ > 0, ∀ x, x ≠ a → |x - a| < δ → f x ≤ f a) →  -- local maximum at x = a
  -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_local_max_range_l2501_250143


namespace NUMINAMATH_CALUDE_orthogonal_circles_product_l2501_250195

theorem orthogonal_circles_product (x y u v : ℝ) 
  (h1 : x^2 + y^2 = 1)
  (h2 : u^2 + v^2 = 1)
  (h3 : x*u + y*v = 0) :
  x*y + u*v = 0 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_circles_product_l2501_250195


namespace NUMINAMATH_CALUDE_total_money_end_is_3933_33_l2501_250134

/-- Calculates the total money after splitting and increasing the remainder -/
def totalMoneyAtEnd (cecilMoney : ℚ) : ℚ :=
  let catherineMoney := 2 * cecilMoney - 250
  let carmelaMoney := 2 * cecilMoney + 50
  let averageMoney := (cecilMoney + catherineMoney + carmelaMoney) / 3
  let carlosMoney := averageMoney + 200
  let totalMoney := cecilMoney + catherineMoney + carmelaMoney + carlosMoney
  let splitAmount := totalMoney / 7
  let remainingAmount := totalMoney - (splitAmount * 7)
  let increase := remainingAmount * (5 / 100)
  totalMoney + increase

/-- Theorem stating that the total money at the end is $3933.33 -/
theorem total_money_end_is_3933_33 :
  totalMoneyAtEnd 600 = 3933.33 := by sorry

end NUMINAMATH_CALUDE_total_money_end_is_3933_33_l2501_250134


namespace NUMINAMATH_CALUDE_rides_second_day_l2501_250116

def rides_first_day : ℕ := 4
def total_rides : ℕ := 7

theorem rides_second_day : total_rides - rides_first_day = 3 := by
  sorry

end NUMINAMATH_CALUDE_rides_second_day_l2501_250116


namespace NUMINAMATH_CALUDE_conor_eggplants_per_day_l2501_250118

/-- The number of eggplants Conor can chop in a day -/
def eggplants_per_day : ℕ := sorry

/-- The number of carrots Conor can chop in a day -/
def carrots_per_day : ℕ := 9

/-- The number of potatoes Conor can chop in a day -/
def potatoes_per_day : ℕ := 8

/-- The number of days Conor works per week -/
def work_days_per_week : ℕ := 4

/-- The total number of vegetables Conor chops in a week -/
def total_vegetables_per_week : ℕ := 116

theorem conor_eggplants_per_day :
  eggplants_per_day = 12 :=
by sorry

end NUMINAMATH_CALUDE_conor_eggplants_per_day_l2501_250118


namespace NUMINAMATH_CALUDE_smallest_meeting_time_l2501_250108

/-- The number of horses -/
def num_horses : ℕ := 8

/-- The time taken by horse k to complete one lap -/
def lap_time (k : ℕ) : ℕ := k^2

/-- Predicate to check if a time t is when at least 4 horses are at the starting point -/
def at_least_four_horses_meet (t : ℕ) : Prop :=
  ∃ (h1 h2 h3 h4 : ℕ), 
    h1 ≤ num_horses ∧ h2 ≤ num_horses ∧ h3 ≤ num_horses ∧ h4 ≤ num_horses ∧
    h1 ≠ h2 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h3 ∧ h2 ≠ h4 ∧ h3 ≠ h4 ∧
    t % (lap_time h1) = 0 ∧ t % (lap_time h2) = 0 ∧ t % (lap_time h3) = 0 ∧ t % (lap_time h4) = 0

/-- The smallest positive time when at least 4 horses meet at the starting point -/
def S : ℕ := 144

theorem smallest_meeting_time : 
  (S > 0) ∧ 
  at_least_four_horses_meet S ∧ 
  ∀ t, 0 < t ∧ t < S → ¬(at_least_four_horses_meet t) :=
by sorry

end NUMINAMATH_CALUDE_smallest_meeting_time_l2501_250108


namespace NUMINAMATH_CALUDE_timothy_movie_count_l2501_250189

theorem timothy_movie_count (timothy_prev : ℕ) 
  (h1 : timothy_prev + (timothy_prev + 7) + 2 * (timothy_prev + 7) + timothy_prev / 2 = 129) : 
  timothy_prev = 24 := by
  sorry

end NUMINAMATH_CALUDE_timothy_movie_count_l2501_250189


namespace NUMINAMATH_CALUDE_sandbag_weight_l2501_250110

theorem sandbag_weight (bag_capacity : ℝ) (fill_percentage : ℝ) (weight_increase : ℝ) : 
  bag_capacity = 250 →
  fill_percentage = 0.8 →
  weight_increase = 0.4 →
  (bag_capacity * fill_percentage * (1 + weight_increase)) = 280 := by
  sorry

end NUMINAMATH_CALUDE_sandbag_weight_l2501_250110


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2501_250111

theorem complex_expression_equality : 
  let a : ℂ := 3 - 2*I
  let b : ℂ := -2 + 3*I
  3*a + 4*b = 1 + 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2501_250111


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l2501_250176

theorem digit_sum_puzzle (c o u n t s : ℕ) : 
  c ≠ 0 → o ≠ 0 → u ≠ 0 → n ≠ 0 → t ≠ 0 → s ≠ 0 →
  c + o = u →
  u + n = t →
  t + c = s →
  o + n + s = 18 →
  t = 9 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l2501_250176


namespace NUMINAMATH_CALUDE_initial_apples_count_l2501_250141

/-- Represents the number of apple trees Rachel has -/
def total_trees : ℕ := 52

/-- Represents the number of apples picked from one tree -/
def apples_picked : ℕ := 2

/-- Represents the number of apples remaining on the tree after picking -/
def apples_remaining : ℕ := 7

/-- Theorem stating that the initial number of apples on the tree is equal to
    the sum of apples remaining and apples picked -/
theorem initial_apples_count : 
  ∃ (initial_apples : ℕ), initial_apples = apples_remaining + apples_picked :=
by sorry

end NUMINAMATH_CALUDE_initial_apples_count_l2501_250141


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l2501_250172

/-- Represents the profit distribution in a partnership --/
structure ProfitDistribution where
  investment_A : ℕ
  investment_B : ℕ
  investment_C : ℕ
  profit_share_C : ℕ

/-- Calculates the total profit given a ProfitDistribution --/
def calculate_total_profit (pd : ProfitDistribution) : ℕ :=
  let total_investment := pd.investment_A + pd.investment_B + pd.investment_C
  let profit_per_unit := pd.profit_share_C * total_investment / pd.investment_C
  profit_per_unit

/-- Theorem stating that given the specific investments and C's profit share, the total profit is 86400 --/
theorem partnership_profit_calculation (pd : ProfitDistribution) 
  (h1 : pd.investment_A = 12000)
  (h2 : pd.investment_B = 16000)
  (h3 : pd.investment_C = 20000)
  (h4 : pd.profit_share_C = 36000) :
  calculate_total_profit pd = 86400 := by
  sorry


end NUMINAMATH_CALUDE_partnership_profit_calculation_l2501_250172


namespace NUMINAMATH_CALUDE_vector_dot_product_and_trigonometry_l2501_250163

/-- Given vectors a and b, and a function f, prove the following statements. -/
theorem vector_dot_product_and_trigonometry 
  (a : ℝ × ℝ) 
  (b : ℝ → ℝ × ℝ) 
  (f : ℝ → ℝ) 
  (h_a : a = (Real.sqrt 3, 1))
  (h_b : ∀ x, b x = (Real.cos x, Real.sin x))
  (h_f : ∀ x, f x = a.1 * (b x).1 + a.2 * (b x).2)
  (h_x : ∀ x, 0 < x ∧ x < Real.pi)
  (α : ℝ)
  (h_α : f α = 2 * Real.sqrt 2 / 3) :
  (∃ x, a.1 * (b x).1 + a.2 * (b x).2 = 0 → x = 2 * Real.pi / 3) ∧ 
  Real.sin (2 * α + Real.pi / 6) = -5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_and_trigonometry_l2501_250163


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l2501_250120

theorem value_of_a_minus_b (a b c : ℝ) 
  (h1 : a - (b - 2*c) = 19) 
  (h2 : a - b - 2*c = 7) : 
  a - b = 13 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l2501_250120


namespace NUMINAMATH_CALUDE_storks_on_fence_l2501_250155

/-- The number of storks initially on the fence -/
def initial_storks : ℕ := 4

/-- The number of birds initially on the fence -/
def initial_birds : ℕ := 3

/-- The number of additional storks that joined -/
def additional_storks : ℕ := 6

/-- The total number of birds and storks after additional storks joined -/
def total_after : ℕ := 13

theorem storks_on_fence :
  initial_birds + initial_storks + additional_storks = total_after :=
by sorry

end NUMINAMATH_CALUDE_storks_on_fence_l2501_250155


namespace NUMINAMATH_CALUDE_expression_evaluation_l2501_250135

theorem expression_evaluation (x : ℤ) 
  (h1 : 1 - x > (-1 - x) / 2) 
  (h2 : x + 1 > 0) 
  (h3 : x ≠ 1) 
  (h4 : x ≠ 0) : 
  (1 + (3*x - 1) / (x + 1)) / (x / (x^2 - 1)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2501_250135


namespace NUMINAMATH_CALUDE_gcd_9157_2695_l2501_250112

theorem gcd_9157_2695 : Nat.gcd 9157 2695 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9157_2695_l2501_250112


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2501_250170

theorem divisibility_implies_equality (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : a * b ∣ (a^2 + b^2)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2501_250170


namespace NUMINAMATH_CALUDE_rotating_squares_intersection_area_l2501_250119

/-- The area of intersection of two rotating unit squares after 5 minutes -/
theorem rotating_squares_intersection_area : 
  let revolution_rate : ℝ := 2 * Real.pi / 60 -- radians per minute
  let rotation_time : ℝ := 5 -- minutes
  let rotation_angle : ℝ := revolution_rate * rotation_time
  let intersection_area : ℝ := (1 - Real.cos rotation_angle) * (1 - Real.sin rotation_angle)
  intersection_area = (2 - Real.sqrt 3) / 4 := by
sorry


end NUMINAMATH_CALUDE_rotating_squares_intersection_area_l2501_250119


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2501_250198

/-- The area of the shaded region in a square with side length 6 and inscribed circles
    of radius 2√3 at each corner is equal to 36 - 12√3 - 4π. -/
theorem shaded_area_square_with_circles (square_side : ℝ) (circle_radius : ℝ)
  (h_side : square_side = 6)
  (h_radius : circle_radius = 2 * Real.sqrt 3) :
  let total_area := square_side ^ 2
  let triangle_area := 8 * (1 / 2 * (square_side / 2) * circle_radius)
  let sector_area := 4 * (1 / 12 * π * circle_radius ^ 2)
  total_area - triangle_area - sector_area = 36 - 12 * Real.sqrt 3 - 4 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2501_250198


namespace NUMINAMATH_CALUDE_family_eye_count_l2501_250160

-- Define the family members and their eye counts
def mother_eyes : ℕ := 1
def father_eyes : ℕ := 3
def num_children : ℕ := 3
def eyes_per_child : ℕ := 4

-- Theorem statement
theorem family_eye_count :
  mother_eyes + father_eyes + num_children * eyes_per_child = 16 :=
by sorry

end NUMINAMATH_CALUDE_family_eye_count_l2501_250160


namespace NUMINAMATH_CALUDE_ellipse_sum_l2501_250166

/-- Theorem: For an ellipse with center (-3, 1), horizontal semi-major axis length 4,
    and vertical semi-minor axis length 2, the sum of h, k, a, and c is equal to 4. -/
theorem ellipse_sum (h k a c : ℝ) : 
  h = -3 ∧ k = 1 ∧ a = 4 ∧ c = 2 → h + k + a + c = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l2501_250166


namespace NUMINAMATH_CALUDE_joe_caramel_probability_l2501_250162

/-- Represents the set of candies in Joe's pocket -/
structure CandySet :=
  (lemon : ℕ)
  (caramel : ℕ)

/-- Calculates the probability of selecting a caramel-flavored candy -/
def probability_caramel (cs : CandySet) : ℚ :=
  cs.caramel / (cs.lemon + cs.caramel)

/-- Theorem stating that the probability of selecting a caramel-flavored candy
    from Joe's set is 3/7 -/
theorem joe_caramel_probability :
  let joe_candies : CandySet := { lemon := 4, caramel := 3 }
  probability_caramel joe_candies = 3 / 7 := by
  sorry


end NUMINAMATH_CALUDE_joe_caramel_probability_l2501_250162


namespace NUMINAMATH_CALUDE_decimal_expansion_of_three_sevenths_l2501_250187

/-- The length of the smallest repeating block in the decimal expansion of 3/7 -/
def repeatingBlockLength : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 3/7

theorem decimal_expansion_of_three_sevenths :
  ∃ (d : ℕ → ℕ) (n : ℕ),
    (∀ k, d k < 10) ∧
    (∀ k, d (k + n) = d k) ∧
    (∀ m, m < n → ∃ k, d (k + m) ≠ d k) ∧
    fraction = ∑' k, (d k : ℚ) / 10^(k + 1) ∧
    n = repeatingBlockLength :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_of_three_sevenths_l2501_250187


namespace NUMINAMATH_CALUDE_unique_triplet_solution_l2501_250196

theorem unique_triplet_solution :
  ∀ (x y ℓ : ℕ), x^3 + y^3 - 53 = 7^ℓ ↔ x = 3 ∧ y = 3 ∧ ℓ = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_solution_l2501_250196


namespace NUMINAMATH_CALUDE_smallest_multiple_three_is_solution_three_is_smallest_l2501_250157

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 675 ∣ (450 * x) → x ≥ 3 :=
sorry

theorem three_is_solution : 675 ∣ (450 * 3) :=
sorry

theorem three_is_smallest : ∀ y : ℕ, y > 0 ∧ y < 3 → ¬(675 ∣ (450 * y)) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_three_is_solution_three_is_smallest_l2501_250157


namespace NUMINAMATH_CALUDE_population_growth_l2501_250144

theorem population_growth (initial_population : ℝ) (final_population : ℝ) (second_year_increase : ℝ) :
  initial_population = 1000 →
  final_population = 1320 →
  second_year_increase = 0.20 →
  ∃ first_year_increase : ℝ,
    first_year_increase = 0.10 ∧
    final_population = initial_population * (1 + first_year_increase) * (1 + second_year_increase) :=
by sorry

end NUMINAMATH_CALUDE_population_growth_l2501_250144


namespace NUMINAMATH_CALUDE_append_nine_to_two_digit_number_l2501_250125

/-- Given a two-digit number with tens digit t and units digit u,
    appending 9 to the right results in 100t + 10u + 9 -/
theorem append_nine_to_two_digit_number (t u : ℕ) 
  (h1 : t ≥ 1 ∧ t ≤ 9) (h2 : u ≥ 0 ∧ u ≤ 9) :
  (10 * t + u) * 10 + 9 = 100 * t + 10 * u + 9 := by
  sorry


end NUMINAMATH_CALUDE_append_nine_to_two_digit_number_l2501_250125


namespace NUMINAMATH_CALUDE_chess_match_average_time_l2501_250127

/-- Proves that in a chess match with given conditions, one player's average move time is 28 seconds -/
theorem chess_match_average_time (total_moves : ℕ) (opponent_avg_time : ℕ) (match_duration : ℕ) :
  total_moves = 30 →
  opponent_avg_time = 40 →
  match_duration = 17 * 60 →
  ∃ (player_avg_time : ℕ), player_avg_time = 28 ∧ 
    (total_moves / 2) * (player_avg_time + opponent_avg_time) = match_duration := by
  sorry

end NUMINAMATH_CALUDE_chess_match_average_time_l2501_250127


namespace NUMINAMATH_CALUDE_triangle_properties_l2501_250174

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  sum_angles : A + B + C = π

/-- The vector m in the problem -/
def m (t : Triangle) : ℝ × ℝ := (t.a + t.c, t.b - t.a)

/-- The vector n in the problem -/
def n (t : Triangle) : ℝ × ℝ := (t.a - t.c, t.b)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem triangle_properties (t : Triangle) 
  (h_perp : dot_product (m t) (n t) = 0)
  (h_sin : 2 * Real.sin (t.A / 2) ^ 2 + 2 * Real.sin (t.B / 2) ^ 2 = 1) :
  t.C = π / 3 ∧ t.A = π / 3 ∧ t.B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2501_250174


namespace NUMINAMATH_CALUDE_arithmetic_sequences_bound_l2501_250114

theorem arithmetic_sequences_bound (n k b : ℕ) (d₁ d₂ : ℤ) :
  0 < b → b < n →
  (∀ i j, i ≠ j → i ≤ n → j ≤ n → ∃ (x y : ℤ), x ≠ y ∧ 
    (∃ (a r : ℤ), x = a + r * (if i ≤ b then d₁ else d₂) ∧
                  y = a + r * (if i ≤ b then d₁ else d₂)) ∧
    (∃ (a r : ℤ), x = a + r * (if j ≤ b then d₁ else d₂) ∧
                  y = a + r * (if j ≤ b then d₁ else d₂))) →
  b ≤ 2 * (k - d₂ / Int.gcd d₁ d₂) - 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_bound_l2501_250114


namespace NUMINAMATH_CALUDE_m_condition_necessary_not_sufficient_l2501_250184

-- Define the condition for m
def m_condition (m : ℝ) : Prop := 2 < m ∧ m < 6

-- Define the condition for the equation to represent an ellipse
def is_ellipse (m : ℝ) : Prop := m > 2 ∧ m < 6 ∧ m ≠ 4

-- Theorem stating that m_condition is necessary but not sufficient for is_ellipse
theorem m_condition_necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → m_condition m) ∧
  ¬(∀ m : ℝ, m_condition m → is_ellipse m) := by sorry

end NUMINAMATH_CALUDE_m_condition_necessary_not_sufficient_l2501_250184


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2501_250146

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | ∃ y, y = Real.log (4 - x^2)}

theorem union_of_A_and_B : A ∪ B = Ioc (-2) 4 := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2501_250146


namespace NUMINAMATH_CALUDE_fourth_quadrant_properties_l2501_250129

open Real

-- Define the fourth quadrant
def fourth_quadrant (α : ℝ) : Prop := 3 * π / 2 < α ∧ α < 2 * π

theorem fourth_quadrant_properties (α : ℝ) (h : fourth_quadrant α) :
  (∃ α, fourth_quadrant α ∧ cos (2 * α) > 0) ∧
  (∀ α, fourth_quadrant α → sin (2 * α) < 0) ∧
  (¬ ∃ α, fourth_quadrant α ∧ tan (α / 2) < 0) ∧
  (∃ α, fourth_quadrant α ∧ cos (α / 2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_fourth_quadrant_properties_l2501_250129


namespace NUMINAMATH_CALUDE_movie_attendance_l2501_250193

theorem movie_attendance (total_cost concession_cost child_ticket adult_ticket : ℕ)
  (num_children : ℕ) (h1 : total_cost = 76) (h2 : concession_cost = 12)
  (h3 : child_ticket = 7) (h4 : adult_ticket = 10) (h5 : num_children = 2) :
  (total_cost - concession_cost - num_children * child_ticket) / adult_ticket = 5 := by
  sorry

end NUMINAMATH_CALUDE_movie_attendance_l2501_250193


namespace NUMINAMATH_CALUDE_m_range_characterization_l2501_250123

theorem m_range_characterization (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - (3 - m) * x + 1 > 0 ∨ m * x > 0) ↔ 1/9 < m ∧ m < 1 := by
sorry

end NUMINAMATH_CALUDE_m_range_characterization_l2501_250123


namespace NUMINAMATH_CALUDE_direct_proportion_b_value_l2501_250158

/-- A function f is a direct proportion function if there exists a constant k such that f x = k * x for all x. -/
def IsDirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function we're considering -/
def f (b : ℝ) (x : ℝ) : ℝ := x + b - 2

theorem direct_proportion_b_value :
  (IsDirectProportion (f b)) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_b_value_l2501_250158


namespace NUMINAMATH_CALUDE_no_convex_polygon_with_1974_diagonals_l2501_250121

theorem no_convex_polygon_with_1974_diagonals :
  ¬ ∃ (N : ℕ), N > 0 ∧ N * (N - 3) / 2 = 1974 := by
  sorry

end NUMINAMATH_CALUDE_no_convex_polygon_with_1974_diagonals_l2501_250121


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2501_250105

/-- Theorem: A cube with surface area approximately 600 square cc has a volume of 1000 cubic cc. -/
theorem cube_volume_from_surface_area :
  ∃ (s : ℝ), s > 0 ∧ 6 * s^2 = 599.9999999999998 → s^3 = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2501_250105


namespace NUMINAMATH_CALUDE_sum_of_odd_periodic_function_l2501_250180

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_3 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 3) = f x

theorem sum_of_odd_periodic_function 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_periodic : is_periodic_3 f) 
  (h_value : f (-1) = 1) : 
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_odd_periodic_function_l2501_250180


namespace NUMINAMATH_CALUDE_calf_cost_l2501_250175

/-- Given a cow and a calf where the total cost is $990 and the cow costs 8 times as much as the calf, 
    the cost of the calf is $110. -/
theorem calf_cost (total_cost : ℕ) (cow_calf_ratio : ℕ) (calf_cost : ℕ) : 
  total_cost = 990 → 
  cow_calf_ratio = 8 → 
  calf_cost + cow_calf_ratio * calf_cost = total_cost → 
  calf_cost = 110 := by
  sorry

end NUMINAMATH_CALUDE_calf_cost_l2501_250175


namespace NUMINAMATH_CALUDE_opposite_of_2023_l2501_250122

theorem opposite_of_2023 : 
  ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l2501_250122


namespace NUMINAMATH_CALUDE_total_spent_on_flowers_l2501_250169

def roses_quantity : ℕ := 5
def roses_price : ℕ := 6
def daisies_quantity : ℕ := 3
def daisies_price : ℕ := 4
def tulips_quantity : ℕ := 2
def tulips_price : ℕ := 5

theorem total_spent_on_flowers :
  roses_quantity * roses_price +
  daisies_quantity * daisies_price +
  tulips_quantity * tulips_price = 52 := by
sorry

end NUMINAMATH_CALUDE_total_spent_on_flowers_l2501_250169


namespace NUMINAMATH_CALUDE_regions_bound_l2501_250159

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields to define a plane

/-- The number of regions formed by three planes in 3D space -/
def num_regions (p1 p2 p3 : Plane3D) : ℕ :=
  sorry

theorem regions_bound (p1 p2 p3 : Plane3D) :
  4 ≤ num_regions p1 p2 p3 ∧ num_regions p1 p2 p3 ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_regions_bound_l2501_250159


namespace NUMINAMATH_CALUDE_cloth_selling_price_l2501_250103

/-- Calculates the total selling price of cloth given the length, profit per meter, and cost price per meter. -/
def total_selling_price (length : ℕ) (profit_per_meter : ℕ) (cost_per_meter : ℕ) : ℕ :=
  length * (profit_per_meter + cost_per_meter)

/-- Theorem stating that the total selling price of 85 meters of cloth with a profit of Rs. 5 per meter and a cost price of Rs. 100 per meter is Rs. 8925. -/
theorem cloth_selling_price :
  total_selling_price 85 5 100 = 8925 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l2501_250103


namespace NUMINAMATH_CALUDE_tank_capacity_l2501_250181

theorem tank_capacity (initial_fill : Real) (added_amount : Real) (final_fill : Real) :
  initial_fill = 3/4 →
  added_amount = 4 →
  final_fill = 9/10 →
  ∃ (capacity : Real), capacity = 80/3 ∧
    initial_fill * capacity + added_amount = final_fill * capacity :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_l2501_250181


namespace NUMINAMATH_CALUDE_divergent_series_with_convergent_min_series_l2501_250156

theorem divergent_series_with_convergent_min_series :
  ∃ (x : ℕ → ℝ), 
    (∀ n, x n > 0) ∧ 
    (∀ n, x (n + 1) < x n) ∧ 
    (¬ Summable x) ∧
    (Summable (fun n => min (x (n + 1)) (1 / ((n + 1 : ℝ) * Real.log (n + 1))))) := by
  sorry

end NUMINAMATH_CALUDE_divergent_series_with_convergent_min_series_l2501_250156


namespace NUMINAMATH_CALUDE_square_perimeter_l2501_250137

theorem square_perimeter (side_length : ℝ) (h : side_length = 40) :
  4 * side_length = 160 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_l2501_250137


namespace NUMINAMATH_CALUDE_intersection_point_correct_l2501_250126

/-- The line equation y = x + 3 -/
def line_equation (x y : ℝ) : Prop := y = x + 3

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

/-- The intersection point of the line y = x + 3 and the y-axis -/
def intersection_point : ℝ × ℝ := (0, 3)

theorem intersection_point_correct :
  let (x, y) := intersection_point
  line_equation x y ∧ on_y_axis x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l2501_250126


namespace NUMINAMATH_CALUDE_largest_integer_less_than_150_with_remainder_2_mod_9_l2501_250106

theorem largest_integer_less_than_150_with_remainder_2_mod_9 : ∃ n : ℕ, n < 150 ∧ n % 9 = 2 ∧ ∀ m : ℕ, m < 150 ∧ m % 9 = 2 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_150_with_remainder_2_mod_9_l2501_250106


namespace NUMINAMATH_CALUDE_window_side_length_l2501_250147

/-- Represents the dimensions of a window pane -/
structure Pane where
  width : ℝ
  height : ℝ
  ratio : height = 3 * width

/-- Represents the dimensions of a square window -/
structure SquareWindow where
  pane : Pane
  borderWidth : ℝ
  sideLength : ℝ
  paneCount : ℕ
  isSquare : sideLength = 3 * pane.width + 4 * borderWidth
  hasPanes : paneCount = 9

/-- Theorem: The side length of the square window is 20 inches -/
theorem window_side_length (w : SquareWindow) (h : w.borderWidth = 2) : w.sideLength = 20 :=
by sorry

end NUMINAMATH_CALUDE_window_side_length_l2501_250147


namespace NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l2501_250177

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  radius : ℝ
  side_positive : 0 < side
  base_positive : 0 < base
  radius_positive : 0 < radius

-- Theorem statement
theorem isosceles_triangle_not_unique (r : ℝ) (hr : 0 < r) :
  ∃ (t1 t2 : IsoscelesTriangle), t1.radius = r ∧ t2.radius = r ∧ t1 ≠ t2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l2501_250177


namespace NUMINAMATH_CALUDE_dice_sum_symmetry_l2501_250102

/-- The number of dice being rolled -/
def num_dice : ℕ := 9

/-- The minimum value on each die -/
def min_value : ℕ := 1

/-- The maximum value on each die -/
def max_value : ℕ := 6

/-- The sum we're comparing to -/
def comparison_sum : ℕ := 15

/-- The function to calculate the symmetric sum -/
def symmetric_sum (s : ℕ) : ℕ :=
  2 * ((num_dice * min_value + num_dice * max_value) / 2) - s

theorem dice_sum_symmetry :
  symmetric_sum comparison_sum = 48 :=
sorry

end NUMINAMATH_CALUDE_dice_sum_symmetry_l2501_250102


namespace NUMINAMATH_CALUDE_black_ball_probability_l2501_250142

/-- Given a bag of 100 balls with 45 red balls and a probability of 0.23 for drawing a white ball,
    the probability of drawing a black ball is 0.32. -/
theorem black_ball_probability
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_prob : ℝ)
  (h_total : total_balls = 100)
  (h_red : red_balls = 45)
  (h_white_prob : white_prob = 0.23)
  : (total_balls - red_balls - (white_prob * total_balls)) / total_balls = 0.32 := by
  sorry


end NUMINAMATH_CALUDE_black_ball_probability_l2501_250142


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2501_250117

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2501_250117


namespace NUMINAMATH_CALUDE_stool_height_is_75cm_l2501_250152

/-- Represents the problem setup for Alice's light bulb replacement task -/
structure LightBulbProblem where
  ceiling_height : ℝ
  bulb_below_ceiling : ℝ
  alice_height : ℝ
  alice_reach : ℝ
  decorative_item_below_ceiling : ℝ

/-- Calculates the required stool height for Alice to reach the light bulb -/
def calculate_stool_height (p : LightBulbProblem) : ℝ :=
  p.ceiling_height - p.bulb_below_ceiling - (p.alice_height + p.alice_reach)

/-- Theorem stating that the stool height Alice needs is 75 cm -/
theorem stool_height_is_75cm (p : LightBulbProblem) 
    (h1 : p.ceiling_height = 300)
    (h2 : p.bulb_below_ceiling = 15)
    (h3 : p.alice_height = 160)
    (h4 : p.alice_reach = 50)
    (h5 : p.decorative_item_below_ceiling = 20) :
    calculate_stool_height p = 75 := by
  sorry

#eval calculate_stool_height {
  ceiling_height := 300,
  bulb_below_ceiling := 15,
  alice_height := 160,
  alice_reach := 50,
  decorative_item_below_ceiling := 20
}

end NUMINAMATH_CALUDE_stool_height_is_75cm_l2501_250152


namespace NUMINAMATH_CALUDE_no_special_quadrilateral_l2501_250192

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def side_lengths_different (q : Quadrilateral) : Prop := sorry

def angles_different (q : Quadrilateral) : Prop := sorry

-- Define functions to get side lengths and angles
def side_length (q : Quadrilateral) (side : Fin 4) : ℝ := sorry

def angle (q : Quadrilateral) (vertex : Fin 4) : ℝ := sorry

-- Define predicates for greatest and smallest
def is_greatest_side (q : Quadrilateral) (side : Fin 4) : Prop :=
  ∀ other : Fin 4, other ≠ side → side_length q side > side_length q other

def is_smallest_side (q : Quadrilateral) (side : Fin 4) : Prop :=
  ∀ other : Fin 4, other ≠ side → side_length q side < side_length q other

def is_greatest_angle (q : Quadrilateral) (vertex : Fin 4) : Prop :=
  ∀ other : Fin 4, other ≠ vertex → angle q vertex > angle q other

def is_smallest_angle (q : Quadrilateral) (vertex : Fin 4) : Prop :=
  ∀ other : Fin 4, other ≠ vertex → angle q vertex < angle q other

-- Define the main theorem
theorem no_special_quadrilateral :
  ¬ ∃ (q : Quadrilateral) (s g a : Fin 4),
    is_convex q ∧
    side_lengths_different q ∧
    angles_different q ∧
    is_smallest_side q s ∧
    is_greatest_side q g ∧
    is_greatest_angle q a ∧
    is_smallest_angle q ((a + 2) % 4) ∧
    (a + 1) % 4 ≠ s ∧
    (a + 3) % 4 ≠ s ∧
    ((a + 2) % 4 + 1) % 4 ≠ g ∧
    ((a + 2) % 4 + 3) % 4 ≠ g :=
sorry

end NUMINAMATH_CALUDE_no_special_quadrilateral_l2501_250192


namespace NUMINAMATH_CALUDE_mari_made_64_buttons_l2501_250124

/-- Given the number of buttons Sue made -/
def sue_buttons : ℕ := 6

/-- Kendra's buttons in terms of Sue's -/
def kendra_buttons : ℕ := 2 * sue_buttons

/-- Mari's buttons in terms of Kendra's -/
def mari_buttons : ℕ := 4 + 5 * kendra_buttons

/-- Theorem stating that Mari made 64 buttons -/
theorem mari_made_64_buttons : mari_buttons = 64 := by
  sorry

end NUMINAMATH_CALUDE_mari_made_64_buttons_l2501_250124


namespace NUMINAMATH_CALUDE_cookie_ratio_l2501_250153

/-- Proves that the ratio of cookies baked by Jake to Clementine is 2:1 given the problem conditions -/
theorem cookie_ratio (clementine jake tory : ℕ) (total_revenue : ℕ) : 
  clementine = 72 →
  tory = (jake + clementine) / 2 →
  total_revenue = 648 →
  2 * (clementine + jake + tory) = total_revenue →
  jake = 2 * clementine :=
by sorry

end NUMINAMATH_CALUDE_cookie_ratio_l2501_250153


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2501_250171

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 20) :
  1 / x + 1 / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2501_250171


namespace NUMINAMATH_CALUDE_octal_563_equals_base12_261_l2501_250191

-- Define a function to convert from octal to decimal
def octal_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Define a function to convert from decimal to base 12
def decimal_to_base12 (n : ℕ) : ℕ :=
  (n / 144) * 100 + ((n / 12) % 12) * 10 + (n % 12)

-- Theorem statement
theorem octal_563_equals_base12_261 :
  decimal_to_base12 (octal_to_decimal 563) = 261 :=
sorry

end NUMINAMATH_CALUDE_octal_563_equals_base12_261_l2501_250191


namespace NUMINAMATH_CALUDE_round_trip_speed_l2501_250164

/-- Proves that given specific conditions for a round trip, the outward speed is 3 km/hr -/
theorem round_trip_speed (return_speed : ℝ) (total_time : ℝ) (one_way_distance : ℝ)
  (h1 : return_speed = 2)
  (h2 : total_time = 5)
  (h3 : one_way_distance = 6) :
  (one_way_distance / (total_time - one_way_distance / return_speed) = 3) :=
by sorry

end NUMINAMATH_CALUDE_round_trip_speed_l2501_250164


namespace NUMINAMATH_CALUDE_kabadi_players_l2501_250131

theorem kabadi_players (kho_kho_only : ℕ) (both : ℕ) (total : ℕ) :
  kho_kho_only = 35 →
  both = 5 →
  total = 45 →
  ∃ kabadi : ℕ, kabadi = 15 ∧ total = kabadi + kho_kho_only - both :=
by sorry

end NUMINAMATH_CALUDE_kabadi_players_l2501_250131


namespace NUMINAMATH_CALUDE_cubic_inequality_l2501_250138

theorem cubic_inequality (a b : ℝ) (h : a < b) :
  a^3 - 3*a ≤ b^3 - 3*b + 4 ∧
  (a^3 - 3*a = b^3 - 3*b + 4 ↔ a = -1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2501_250138


namespace NUMINAMATH_CALUDE_total_envelopes_l2501_250188

/-- The number of stamps needed for an envelope weighing more than 5 pounds -/
def heavy_envelope_stamps : ℕ := 5

/-- The number of stamps needed for an envelope weighing less than 5 pounds -/
def light_envelope_stamps : ℕ := 2

/-- The total number of stamps Micah bought -/
def total_stamps : ℕ := 52

/-- The number of envelopes weighing less than 5 pounds -/
def light_envelopes : ℕ := 6

/-- Theorem stating the total number of envelopes Micah bought -/
theorem total_envelopes : 
  ∃ (heavy_envelopes : ℕ), 
    light_envelopes * light_envelope_stamps + 
    heavy_envelopes * heavy_envelope_stamps = total_stamps ∧
    light_envelopes + heavy_envelopes = 14 := by
  sorry

end NUMINAMATH_CALUDE_total_envelopes_l2501_250188


namespace NUMINAMATH_CALUDE_grid_sum_l2501_250154

theorem grid_sum (p q r s : ℕ+) 
  (h_pq : p * q = 6)
  (h_rs : r * s = 8)
  (h_pr : p * r = 4)
  (h_qs : q * s = 12)
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
  p + q + r + s = 13 := by
  sorry

end NUMINAMATH_CALUDE_grid_sum_l2501_250154


namespace NUMINAMATH_CALUDE_alice_probability_after_two_turns_l2501_250109

-- Define the game parameters
def alice_toss_prob : ℚ := 1/2
def alice_keep_prob : ℚ := 1/2
def bob_toss_prob : ℚ := 2/5
def bob_keep_prob : ℚ := 3/5

-- Define the probability that Alice has the ball after two turns
def alice_has_ball_after_two_turns : ℚ := 
  alice_toss_prob * bob_toss_prob + alice_keep_prob * alice_keep_prob

-- Theorem statement
theorem alice_probability_after_two_turns : 
  alice_has_ball_after_two_turns = 9/20 := by sorry

end NUMINAMATH_CALUDE_alice_probability_after_two_turns_l2501_250109


namespace NUMINAMATH_CALUDE_households_with_appliances_l2501_250168

theorem households_with_appliances (total : ℕ) (tv : ℕ) (fridge : ℕ) (both : ℕ) :
  total = 100 →
  tv = 65 →
  fridge = 84 →
  both = 53 →
  tv + fridge - both = 96 := by
  sorry

end NUMINAMATH_CALUDE_households_with_appliances_l2501_250168
