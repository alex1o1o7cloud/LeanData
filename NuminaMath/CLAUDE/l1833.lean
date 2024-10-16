import Mathlib

namespace NUMINAMATH_CALUDE_usual_time_is_36_l1833_183324

-- Define the usual time T as a positive real number
variable (T : ℝ) (hT : T > 0)

-- Define the relationship between normal speed and reduced speed
def reduced_speed_time : ℝ := T + 12

-- Theorem stating that the usual time T is 36 minutes
theorem usual_time_is_36 : T = 36 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_is_36_l1833_183324


namespace NUMINAMATH_CALUDE_total_weight_theorem_l1833_183329

/-- The weight of the orange ring in ounces -/
def orange_ring_oz : ℚ := 1 / 12

/-- The weight of the purple ring in ounces -/
def purple_ring_oz : ℚ := 1 / 3

/-- The weight of the white ring in ounces -/
def white_ring_oz : ℚ := 5 / 12

/-- The weight of the blue ring in ounces -/
def blue_ring_oz : ℚ := 1 / 4

/-- The weight of the green ring in ounces -/
def green_ring_oz : ℚ := 1 / 6

/-- The weight of the red ring in ounces -/
def red_ring_oz : ℚ := 1 / 10

/-- The conversion factor from ounces to grams -/
def oz_to_g : ℚ := 28.3495

/-- The total weight of all rings in grams -/
def total_weight_g : ℚ :=
  (orange_ring_oz + purple_ring_oz + white_ring_oz + blue_ring_oz + green_ring_oz + red_ring_oz) * oz_to_g

theorem total_weight_theorem :
  total_weight_g = 38.271825 := by sorry

end NUMINAMATH_CALUDE_total_weight_theorem_l1833_183329


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l1833_183308

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  m ≠ n →
  α ≠ β →
  perpendicular m α →
  parallel_lines m n →
  parallel_line_plane n β →
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l1833_183308


namespace NUMINAMATH_CALUDE_function_minimum_at_three_l1833_183363

/-- The function f(x) = x(x - c)^2 has a minimum value at x = 3 if and only if c = 3 -/
theorem function_minimum_at_three (c : ℝ) : 
  (∀ x, x * (x - c)^2 ≥ 3 * (3 - c)^2) ↔ c = 3 := by sorry

end NUMINAMATH_CALUDE_function_minimum_at_three_l1833_183363


namespace NUMINAMATH_CALUDE_ab_max_and_reciprocal_sum_min_l1833_183331

theorem ab_max_and_reciprocal_sum_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 10 * b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 10 * y = 1 ∧ a * b ≤ x * y) ∧
  (a * b ≤ 1 / 40) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 10 * y = 1 ∧ 1 / x + 1 / y ≥ 1 / a + 1 / b) ∧
  (1 / a + 1 / b ≥ 11 + 2 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_ab_max_and_reciprocal_sum_min_l1833_183331


namespace NUMINAMATH_CALUDE_total_students_olympiad_l1833_183336

/-- Represents a mathematics teacher at Archimedes Academy -/
inductive Teacher
| Euler
| Fibonacci
| Gauss
| Noether

/-- Returns the number of students taking the Math Olympiad for a given teacher -/
def students_in_class (t : Teacher) : Nat :=
  match t with
  | Teacher.Euler => 15
  | Teacher.Fibonacci => 10
  | Teacher.Gauss => 12
  | Teacher.Noether => 7

/-- The list of all teachers at Archimedes Academy -/
def all_teachers : List Teacher :=
  [Teacher.Euler, Teacher.Fibonacci, Teacher.Gauss, Teacher.Noether]

/-- Theorem stating that the total number of students taking the Math Olympiad is 44 -/
theorem total_students_olympiad :
  (all_teachers.map students_in_class).sum = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_students_olympiad_l1833_183336


namespace NUMINAMATH_CALUDE_smallest_ellipse_area_l1833_183387

theorem smallest_ellipse_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → 
  ((x - 1/2)^2 + y^2 ≥ 1/4 ∧ (x + 1/2)^2 + y^2 ≥ 1/4)) :
  ∃ k : ℝ, k = 4 ∧ π * a * b ≥ k * π := by
  sorry

end NUMINAMATH_CALUDE_smallest_ellipse_area_l1833_183387


namespace NUMINAMATH_CALUDE_teacher_worksheets_l1833_183356

theorem teacher_worksheets :
  ∀ (total_worksheets : ℕ) 
    (problems_per_worksheet : ℕ) 
    (graded_worksheets : ℕ) 
    (remaining_problems : ℕ),
  problems_per_worksheet = 7 →
  graded_worksheets = 8 →
  remaining_problems = 63 →
  problems_per_worksheet * (total_worksheets - graded_worksheets) = remaining_problems →
  total_worksheets = 17 := by
sorry

end NUMINAMATH_CALUDE_teacher_worksheets_l1833_183356


namespace NUMINAMATH_CALUDE_unique_solution_l1833_183311

theorem unique_solution (p q n : ℕ+) (h1 : Nat.gcd p.val q.val = 1)
  (h2 : p + q^2 = (n^2 + 1) * p^2 + q) :
  p = n + 1 ∧ q = n^2 + n + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1833_183311


namespace NUMINAMATH_CALUDE_gold_silver_ratio_l1833_183321

/-- Proves that the ratio of gold to silver bought is 2:1 given the specified conditions --/
theorem gold_silver_ratio :
  let silver_amount : ℝ := 1.5
  let silver_price_per_ounce : ℝ := 20
  let gold_price_multiplier : ℝ := 50
  let total_spent : ℝ := 3030
  let gold_price_per_ounce := silver_price_per_ounce * gold_price_multiplier
  let silver_cost := silver_amount * silver_price_per_ounce
  let gold_cost := total_spent - silver_cost
  let gold_amount := gold_cost / gold_price_per_ounce
  gold_amount / silver_amount = 2 := by
sorry


end NUMINAMATH_CALUDE_gold_silver_ratio_l1833_183321


namespace NUMINAMATH_CALUDE_subcommittees_with_experts_count_l1833_183342

def committee_size : ℕ := 12
def expert_count : ℕ := 5
def subcommittee_size : ℕ := 5

theorem subcommittees_with_experts_count : 
  (Nat.choose committee_size subcommittee_size) - 
  (Nat.choose (committee_size - expert_count) subcommittee_size) = 771 := by
  sorry

end NUMINAMATH_CALUDE_subcommittees_with_experts_count_l1833_183342


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l1833_183303

theorem correct_quadratic_equation :
  ∀ (a b c : ℝ),
  (∃ (a' : ℝ), a' ≠ a ∧ (a' * 4 * 4 + b * 4 + c = 0) ∧ (a' * (-3) * (-3) + b * (-3) + c = 0)) →
  (∃ (c' : ℝ), c' ≠ c ∧ (a * 7 * 7 + b * 7 + c' = 0) ∧ (a * 3 * 3 + b * 3 + c' = 0)) →
  (a = 1 ∧ b = 10 ∧ c = 21) :=
by sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l1833_183303


namespace NUMINAMATH_CALUDE_f_extrema_l1833_183305

-- Define the function f(x)
def f (p q x : ℝ) : ℝ := x^3 - p*x^2 - q*x

-- State the theorem
theorem f_extrema (p q : ℝ) :
  (f p q 1 = 0) →
  (∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f p q x ≤ f p q x₀) ∧
  (f p q x₀ = 4/27) ∧
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f p q x ≥ f p q x₁) ∧
  (f p q x₁ = -4) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l1833_183305


namespace NUMINAMATH_CALUDE_pages_revised_once_is_35_l1833_183388

/-- Represents the manuscript typing problem -/
structure ManuscriptTyping where
  total_pages : ℕ
  pages_revised_twice : ℕ
  first_typing_cost : ℕ
  revision_cost : ℕ
  total_cost : ℕ

/-- Calculates the number of pages revised once -/
def pages_revised_once (m : ManuscriptTyping) : ℕ :=
  ((m.total_cost - m.first_typing_cost * m.total_pages - 
    m.revision_cost * m.pages_revised_twice * 2) / m.revision_cost)

/-- Theorem stating that the number of pages revised once is 35 -/
theorem pages_revised_once_is_35 (m : ManuscriptTyping) 
  (h1 : m.total_pages = 100)
  (h2 : m.pages_revised_twice = 15)
  (h3 : m.first_typing_cost = 6)
  (h4 : m.revision_cost = 4)
  (h5 : m.total_cost = 860) :
  pages_revised_once m = 35 := by
  sorry

#eval pages_revised_once ⟨100, 15, 6, 4, 860⟩

end NUMINAMATH_CALUDE_pages_revised_once_is_35_l1833_183388


namespace NUMINAMATH_CALUDE_binomial_expansion_m_value_l1833_183389

/-- Given a binomial expansion (mx+1)^n where the 5th term has the largest
    coefficient and the coefficient of x^3 is 448, prove that m = 2 -/
theorem binomial_expansion_m_value (m : ℝ) (n : ℕ) : 
  (∃ k : ℕ, k = 5 ∧ 
    ∀ j : ℕ, j ≤ n + 1 → Nat.choose n (j - 1) * m^(j - 1) ≤ Nat.choose n (k - 1) * m^(k - 1)) ∧
  Nat.choose n 3 * m^3 = 448 →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_m_value_l1833_183389


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l1833_183322

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l1833_183322


namespace NUMINAMATH_CALUDE_sqrt_one_minus_two_sin_two_cos_two_l1833_183362

theorem sqrt_one_minus_two_sin_two_cos_two (h : π / 2 < 2 ∧ 2 < π) :
  Real.sqrt (1 - 2 * Real.sin 2 * Real.cos 2) = Real.sin 2 - Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_minus_two_sin_two_cos_two_l1833_183362


namespace NUMINAMATH_CALUDE_first_discount_calculation_l1833_183319

theorem first_discount_calculation (original_price final_price second_discount : ℝ) 
  (h1 : original_price = 150)
  (h2 : final_price = 105)
  (h3 : second_discount = 12.5)
  : ∃ first_discount : ℝ, 
    first_discount = 20 ∧ 
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_calculation_l1833_183319


namespace NUMINAMATH_CALUDE_initial_lions_l1833_183306

/-- Proves that the initial number of lions is 100 given the conditions of the problem -/
theorem initial_lions (net_increase_per_month : ℕ) (total_increase : ℕ) (final_count : ℕ) : 
  net_increase_per_month = 4 → 
  total_increase = 48 → 
  final_count = 148 → 
  final_count - total_increase = 100 := by
sorry

end NUMINAMATH_CALUDE_initial_lions_l1833_183306


namespace NUMINAMATH_CALUDE_shawn_red_pebbles_l1833_183366

/-- The number of red pebbles in Shawn's collection -/
def red_pebbles (total blue yellow : ℕ) : ℕ :=
  total - (blue + 3 * yellow)

/-- Theorem stating the number of red pebbles Shawn painted -/
theorem shawn_red_pebbles :
  ∃ (yellow : ℕ),
    red_pebbles 40 13 yellow = 9 ∧
    13 - yellow = 7 :=
by sorry

end NUMINAMATH_CALUDE_shawn_red_pebbles_l1833_183366


namespace NUMINAMATH_CALUDE_prime_power_sum_existence_l1833_183346

theorem prime_power_sum_existence (p : Finset Nat) (h_prime : ∀ q ∈ p, Nat.Prime q) :
  ∃ x : Nat,
    (∃ a b m n : Nat, (m ∈ p) ∧ (n ∈ p) ∧ (x = a^m + b^n)) ∧
    (∀ q : Nat, Nat.Prime q → (∃ c d : Nat, x = c^q + d^q) → q ∈ p) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_existence_l1833_183346


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1833_183343

theorem quadratic_roots_property (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 5 = 0 → 
  x₂^2 - 4*x₂ - 5 = 0 → 
  (x₁ - 1) * (x₂ - 1) = -8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1833_183343


namespace NUMINAMATH_CALUDE_solve_m_l1833_183368

def g (n : Int) : Int :=
  if n % 2 = 1 then n + 5 else n / 2

theorem solve_m (m : Int) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 15) : m = 55 := by
  sorry

end NUMINAMATH_CALUDE_solve_m_l1833_183368


namespace NUMINAMATH_CALUDE_portfolio_worth_calculation_l1833_183328

/-- Calculates the final portfolio worth after two years given the initial investment,
    growth rates, and transactions. -/
def calculate_portfolio_worth (initial_investment : ℝ) 
                              (year1_growth_rate : ℝ) 
                              (year1_addition : ℝ) 
                              (year1_withdrawal : ℝ)
                              (year2_growth_rate1 : ℝ)
                              (year2_decline_rate : ℝ) : ℝ :=
  sorry

/-- Theorem stating that given the specified conditions, 
    the final portfolio worth is approximately $115.59 -/
theorem portfolio_worth_calculation :
  let initial_investment : ℝ := 80
  let year1_growth_rate : ℝ := 0.15
  let year1_addition : ℝ := 28
  let year1_withdrawal : ℝ := 10
  let year2_growth_rate1 : ℝ := 0.10
  let year2_decline_rate : ℝ := 0.04
  
  abs (calculate_portfolio_worth initial_investment 
                                 year1_growth_rate
                                 year1_addition
                                 year1_withdrawal
                                 year2_growth_rate1
                                 year2_decline_rate - 115.59) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_portfolio_worth_calculation_l1833_183328


namespace NUMINAMATH_CALUDE_tshirt_sales_optimization_l1833_183360

/-- Represents the profit function for T-shirt sales -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 3000

/-- Represents the sales volume function based on price increase -/
def sales_volume (x : ℝ) : ℝ := 300 - 10 * x

theorem tshirt_sales_optimization :
  let initial_price : ℝ := 40
  let purchase_price : ℝ := 30
  let target_profit : ℝ := 3360
  let optimal_increase : ℝ := 2
  let max_profit_price : ℝ := 50
  let max_profit : ℝ := 4000
  
  -- Part 1: Prove that increasing the price by 2 yuan yields the target profit
  (∃ x : ℝ, x ≥ 0 ∧ profit_function x = target_profit ∧
    ∀ y : ℝ, y ≥ 0 ∧ profit_function y = target_profit → x ≤ y) ∧
  profit_function optimal_increase = target_profit ∧
  
  -- Part 2: Prove that setting the price to 50 yuan maximizes profit
  (∀ x : ℝ, profit_function x ≤ max_profit) ∧
  profit_function (max_profit_price - initial_price) = max_profit := by
  sorry

end NUMINAMATH_CALUDE_tshirt_sales_optimization_l1833_183360


namespace NUMINAMATH_CALUDE_student_group_assignments_l1833_183393

theorem student_group_assignments (n : ℕ) (k : ℕ) :
  n = 5 → k = 2 → (2 : ℕ) ^ n = 32 := by
  sorry

end NUMINAMATH_CALUDE_student_group_assignments_l1833_183393


namespace NUMINAMATH_CALUDE_negation_equivalence_l1833_183367

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1833_183367


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_8_l1833_183348

/-- The area of a circle with diameter 8 meters is 16π square meters -/
theorem circle_area_with_diameter_8 (π : ℝ) :
  let diameter : ℝ := 8
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_8_l1833_183348


namespace NUMINAMATH_CALUDE_delegate_seating_probability_l1833_183309

/-- Represents the number of delegates -/
def total_delegates : ℕ := 12

/-- Represents the number of countries -/
def num_countries : ℕ := 3

/-- Represents the number of delegates per country -/
def delegates_per_country : ℕ := 4

/-- Calculates the probability of each delegate sitting next to at least one delegate from another country -/
def seating_probability : ℚ :=
  221 / 231

/-- Theorem stating that the probability of each delegate sitting next to at least one delegate 
    from another country is 221/231 -/
theorem delegate_seating_probability :
  seating_probability = 221 / 231 := by sorry

end NUMINAMATH_CALUDE_delegate_seating_probability_l1833_183309


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l1833_183340

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

def theorem_smallest_fourth_number (fourth : ℕ) : Prop :=
  is_two_digit fourth ∧
  (sum_of_digits 24 + sum_of_digits 58 + sum_of_digits 63 + sum_of_digits fourth) * 4 =
  (24 + 58 + 63 + fourth)

theorem smallest_fourth_number :
  ∃ (fourth : ℕ), theorem_smallest_fourth_number fourth ∧
  (∀ (n : ℕ), theorem_smallest_fourth_number n → fourth ≤ n) ∧
  fourth = 35 :=
sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l1833_183340


namespace NUMINAMATH_CALUDE_stick_triangle_area_l1833_183376

/-- Given three sticks of length 24, one of which is broken into two parts,
    if these parts form a right triangle with the other two sticks,
    then the area of this triangle is 216 square centimeters. -/
theorem stick_triangle_area : ∀ a : ℝ,
  0 < a →
  a < 24 →
  a^2 + 24^2 = (48 - a)^2 →
  (1/2) * a * 24 = 216 := by
  sorry

end NUMINAMATH_CALUDE_stick_triangle_area_l1833_183376


namespace NUMINAMATH_CALUDE_difference_3003_l1833_183385

/-- The number of terms in each sequence -/
def n : ℕ := 3003

/-- The sum of the first n odd numbers -/
def sum_odd (n : ℕ) : ℕ := n * n

/-- The sum of the first n even numbers starting from 2 -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even numbers (starting from 2) 
    and the sum of the first n odd numbers -/
def difference (n : ℕ) : ℤ := sum_even n - sum_odd n

theorem difference_3003 : difference n = 7999 := by sorry

end NUMINAMATH_CALUDE_difference_3003_l1833_183385


namespace NUMINAMATH_CALUDE_min_hits_in_square_l1833_183378

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A configuration of points in a square -/
def SquareConfiguration := List Point

/-- Function to determine if a point is hit -/
def isHit (config : SquareConfiguration) (p : Point) : Bool := sorry

/-- Function to count the number of hits in a configuration -/
def countHits (config : SquareConfiguration) : Nat :=
  (config.filter (isHit config)).length

/-- Theorem stating the existence of a configuration with minimum 10 hits -/
theorem min_hits_in_square (n : Nat) (h : n = 50) :
  ∃ (config : SquareConfiguration),
    config.length = n ∧
    countHits config = 10 ∧
    ∀ (other_config : SquareConfiguration),
      other_config.length = n →
      countHits other_config ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_min_hits_in_square_l1833_183378


namespace NUMINAMATH_CALUDE_prime_square_plus_twelve_mod_twelve_l1833_183304

theorem prime_square_plus_twelve_mod_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) :
  (p^2 + 12) % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_plus_twelve_mod_twelve_l1833_183304


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_range_of_a_l1833_183370

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 0 < x ∧ x < 4}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 3} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | -1 ≤ x ∧ x < 4} := by sorry

-- Theorem for the range of a when B is a subset of C
theorem range_of_a (h : B ⊆ C a) : a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_range_of_a_l1833_183370


namespace NUMINAMATH_CALUDE_base_10_to_7_395_l1833_183339

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

theorem base_10_to_7_395 :
  toBase7 395 = [1, 1, 0, 3] :=
sorry

end NUMINAMATH_CALUDE_base_10_to_7_395_l1833_183339


namespace NUMINAMATH_CALUDE_perimeter_area_bisector_coincide_l1833_183398

/-- An isosceles triangle with side lengths 5, 5, and 6 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : a = b ∧ a = 5 ∧ c = 6

/-- A line bisecting the perimeter of the triangle -/
def perimeterBisector (t : IsoscelesTriangle) : Set (ℝ × ℝ) :=
  sorry

/-- A line bisecting the area of the triangle -/
def areaBisector (t : IsoscelesTriangle) : Set (ℝ × ℝ) :=
  sorry

/-- Theorem stating that the perimeter bisector coincides with the area bisector -/
theorem perimeter_area_bisector_coincide (t : IsoscelesTriangle) :
  perimeterBisector t = areaBisector t :=
sorry

end NUMINAMATH_CALUDE_perimeter_area_bisector_coincide_l1833_183398


namespace NUMINAMATH_CALUDE_exactly_one_even_l1833_183345

theorem exactly_one_even (a b c : ℕ) : 
  (a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0) ∨ 
  (a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0) ∨ 
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0) :=
by
  sorry

#check exactly_one_even

end NUMINAMATH_CALUDE_exactly_one_even_l1833_183345


namespace NUMINAMATH_CALUDE_b_minus_a_value_l1833_183381

theorem b_minus_a_value (a b : ℝ) (h1 : |a| = 1) (h2 : |b| = 3) (h3 : a < b) :
  b - a = 2 ∨ b - a = 4 := by
  sorry

end NUMINAMATH_CALUDE_b_minus_a_value_l1833_183381


namespace NUMINAMATH_CALUDE_problem_statements_l1833_183312

theorem problem_statements :
  (∀ x : ℝ, (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (x^2 - 3*x + 2 = 0 → x = 1)) ∧
  (¬(∀ x : ℝ, x^2 + x + 1 ≠ 0) ↔ (∃ x : ℝ, x^2 + x + 1 = 0)) ∧
  (∀ p q : Prop, (p ∧ q) → (p ∧ q)) ∧
  ((∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧ (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ ¬(x > 2))) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l1833_183312


namespace NUMINAMATH_CALUDE_abs_z_equals_one_l1833_183395

theorem abs_z_equals_one (r : ℝ) (z : ℂ) (h1 : |r| < Real.sqrt 8) (h2 : z + 1/z = r) : 
  Complex.abs z = 1 := by
sorry

end NUMINAMATH_CALUDE_abs_z_equals_one_l1833_183395


namespace NUMINAMATH_CALUDE_class_average_score_l1833_183313

theorem class_average_score
  (num_boys : ℕ)
  (num_girls : ℕ)
  (avg_score_boys : ℚ)
  (avg_score_girls : ℚ)
  (h1 : num_boys = 12)
  (h2 : num_girls = 4)
  (h3 : avg_score_boys = 84)
  (h4 : avg_score_girls = 92) :
  (num_boys * avg_score_boys + num_girls * avg_score_girls) / (num_boys + num_girls) = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_class_average_score_l1833_183313


namespace NUMINAMATH_CALUDE_z_squared_in_first_quadrant_l1833_183301

theorem z_squared_in_first_quadrant (z : ℂ) (h : (z - I) / (1 + I) = 2 - 2*I) :
  (z^2).re > 0 ∧ (z^2).im > 0 :=
by sorry

end NUMINAMATH_CALUDE_z_squared_in_first_quadrant_l1833_183301


namespace NUMINAMATH_CALUDE_office_paper_duration_l1833_183357

/-- The number of days printer paper will last given the number of packs, sheets per pack, and daily usage. -/
def printer_paper_duration (packs : ℕ) (sheets_per_pack : ℕ) (daily_usage : ℕ) : ℕ :=
  (packs * sheets_per_pack) / daily_usage

/-- Theorem stating that two packs of 240-sheet paper will last 6 days when using 80 sheets per day. -/
theorem office_paper_duration :
  printer_paper_duration 2 240 80 = 6 := by
  sorry

end NUMINAMATH_CALUDE_office_paper_duration_l1833_183357


namespace NUMINAMATH_CALUDE_theater_ticket_cost_l1833_183325

/-- The cost of theater tickets for a group -/
def theater_cost (adult_price : ℚ) : ℚ :=
  let child_price := adult_price / 2
  let total_price := 10 * adult_price + 8 * child_price
  total_price * (1 - 1/10)  -- 10% discount applied

theorem theater_ticket_cost :
  ∃ (adult_price : ℚ),
    8 * adult_price + 7 * (adult_price / 2) = 42 ∧
    theater_cost adult_price = 46 :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_cost_l1833_183325


namespace NUMINAMATH_CALUDE_count_integer_lengths_problem_triangle_l1833_183397

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  leg1 : ℕ
  leg2 : ℕ

/-- Counts the number of distinct integer lengths of line segments 
    from a vertex to points on the hypotenuse -/
def countIntegerLengths (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle in the problem -/
def problemTriangle : RightTriangle :=
  { leg1 := 15, leg2 := 36 }

/-- The theorem stating that the number of distinct integer lengths 
    for the given triangle is 24 -/
theorem count_integer_lengths_problem_triangle : 
  countIntegerLengths problemTriangle = 24 :=
sorry

end NUMINAMATH_CALUDE_count_integer_lengths_problem_triangle_l1833_183397


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_minus_three_l1833_183350

theorem one_third_of_seven_times_nine_minus_three (x : ℚ) : 
  x = (1 / 3 : ℚ) * (7 * 9) - 3 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_minus_three_l1833_183350


namespace NUMINAMATH_CALUDE_parity_of_p_and_q_l1833_183399

theorem parity_of_p_and_q (m n p q : ℤ) :
  Odd m →
  Even n →
  p - 1998 * q = n →
  1999 * p + 3 * q = m →
  (Even p ∧ Odd q) :=
by sorry

end NUMINAMATH_CALUDE_parity_of_p_and_q_l1833_183399


namespace NUMINAMATH_CALUDE_max_distance_sum_l1833_183374

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the point M
def M : ℝ × ℝ := (6, 4)

-- Statement of the theorem
theorem max_distance_sum (a b : ℝ) (F₁ : ℝ × ℝ) :
  ∃ (max : ℝ), ∀ (P : ℝ × ℝ),
    ellipse a b P.1 P.2 →
    dist P M + dist P F₁ ≤ max ∧
    (∃ (Q : ℝ × ℝ), ellipse a b Q.1 Q.2 ∧ dist Q M + dist Q F₁ = max) ∧
    max = 15 :=
sorry

end NUMINAMATH_CALUDE_max_distance_sum_l1833_183374


namespace NUMINAMATH_CALUDE_triple_345_is_right_triangle_l1833_183334

/-- A triple of natural numbers representing the sides of a triangle -/
structure TripleNat where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if a triple of natural numbers satisfies the Pythagorean theorem -/
def is_right_triangle (t : TripleNat) : Prop :=
  t.a ^ 2 + t.b ^ 2 = t.c ^ 2

/-- The specific triple (3, 4, 5) -/
def triple_345 : TripleNat :=
  { a := 3, b := 4, c := 5 }

/-- Theorem stating that (3, 4, 5) forms a right triangle -/
theorem triple_345_is_right_triangle : is_right_triangle triple_345 := by
  sorry

end NUMINAMATH_CALUDE_triple_345_is_right_triangle_l1833_183334


namespace NUMINAMATH_CALUDE_probability_of_six_or_less_l1833_183353

def num_red_balls : ℕ := 4
def num_black_balls : ℕ := 3
def total_balls : ℕ := num_red_balls + num_black_balls
def num_drawn : ℕ := 4
def red_points : ℕ := 1
def black_points : ℕ := 3

def score (red_drawn : ℕ) : ℕ :=
  red_drawn * red_points + (num_drawn - red_drawn) * black_points

def probability_of_score (s : ℕ) : ℚ :=
  (Nat.choose num_red_balls s * Nat.choose num_black_balls (num_drawn - s)) /
  Nat.choose total_balls num_drawn

theorem probability_of_six_or_less :
  probability_of_score 4 + probability_of_score 3 = 13 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_six_or_less_l1833_183353


namespace NUMINAMATH_CALUDE_problem_solution_l1833_183349

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 15) :
  a / (a + b) + b / (b + c) + c / (c + a) = -12 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1833_183349


namespace NUMINAMATH_CALUDE_unique_solution_iff_c_equals_three_l1833_183351

theorem unique_solution_iff_c_equals_three :
  ∀ c : ℝ, (∃! (x y : ℝ), (2 * |x + 7| + |y - 4| = c) ∧ (|x + 4| + 2 * |y - 7| = c)) ↔ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_c_equals_three_l1833_183351


namespace NUMINAMATH_CALUDE_final_milk_amount_l1833_183341

-- Define the initial amount of milk
def initial_milk : ℚ := 5

-- Define the amount given away
def given_away : ℚ := 18/4

-- Define the amount received back
def received_back : ℚ := 7/4

-- Theorem statement
theorem final_milk_amount :
  initial_milk - given_away + received_back = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_final_milk_amount_l1833_183341


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l1833_183355

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 + Nat.factorial 5 = 5160 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l1833_183355


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_45_l1833_183323

/-- The sum of n consecutive positive integers starting from k -/
def consecutiveSum (n k : ℕ) : ℕ := n * (2 * k + n - 1) / 2

/-- The largest number of positive consecutive integers that sum to 45 -/
theorem largest_consecutive_sum_45 :
  (∃ (k : ℕ), k > 0 ∧ consecutiveSum 9 k = 45) ∧
  (∀ (n k : ℕ), n > 9 → k > 0 → consecutiveSum n k ≠ 45) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_45_l1833_183323


namespace NUMINAMATH_CALUDE_johnson_farm_cost_l1833_183316

/-- Represents the farm and cultivation costs -/
structure Farm :=
  (total_land : ℕ)
  (corn_cost : ℕ)
  (wheat_cost : ℕ)

/-- Calculates the total cultivation cost given the acres of wheat planted -/
def total_cost (f : Farm) (wheat_acres : ℕ) : ℕ :=
  let corn_acres := f.total_land - wheat_acres
  corn_acres * f.corn_cost + wheat_acres * f.wheat_cost

/-- Theorem stating the total cost for the given farm and wheat acreage -/
theorem johnson_farm_cost :
  let f := Farm.mk 500 42 30
  total_cost f 200 = 18600 := by
  sorry

end NUMINAMATH_CALUDE_johnson_farm_cost_l1833_183316


namespace NUMINAMATH_CALUDE_sallys_earnings_l1833_183318

theorem sallys_earnings (first_month_earnings : ℝ) : 
  first_month_earnings + (first_month_earnings * 1.1) = 2100 → 
  first_month_earnings = 1000 := by
sorry

end NUMINAMATH_CALUDE_sallys_earnings_l1833_183318


namespace NUMINAMATH_CALUDE_two_trees_remain_l1833_183338

/-- The number of walnut trees remaining after removal -/
def remaining_trees (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem stating that 2 trees remain after removing 4 from 6 -/
theorem two_trees_remain :
  remaining_trees 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_trees_remain_l1833_183338


namespace NUMINAMATH_CALUDE_distance_between_points_l1833_183344

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (8, -5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1833_183344


namespace NUMINAMATH_CALUDE_two_books_selection_l1833_183315

/-- Represents the number of books in each genre -/
def num_books_per_genre : ℕ := 4

/-- Calculates the number of ways to select two books from different genres,
    with one being a fantasy novel -/
def select_two_books : ℕ :=
  num_books_per_genre * (num_books_per_genre + num_books_per_genre)

theorem two_books_selection :
  select_two_books = 32 := by
  sorry

#eval select_two_books

end NUMINAMATH_CALUDE_two_books_selection_l1833_183315


namespace NUMINAMATH_CALUDE_probability_of_ANN9_l1833_183364

/-- Represents the set of possible symbols for each position in the license plate --/
structure LicensePlateSymbols where
  vowels : Finset Char
  nonVowels : Finset Char
  digits : Finset Char

/-- Represents the rules for forming a license plate in Algebrica --/
structure LicensePlateRules where
  symbols : LicensePlateSymbols
  firstIsVowel : Char → Prop
  secondThirdAreIdenticalNonVowels : Char → Prop
  fourthIsDigit : Char → Prop

/-- Calculates the total number of possible license plates --/
def totalLicensePlates (rules : LicensePlateRules) : ℕ :=
  (rules.symbols.vowels.card) * (rules.symbols.nonVowels.card) * (rules.symbols.digits.card)

/-- Represents a specific license plate --/
structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Char

/-- Checks if a license plate is valid according to the rules --/
def isValidLicensePlate (plate : LicensePlate) (rules : LicensePlateRules) : Prop :=
  rules.firstIsVowel plate.first ∧
  rules.secondThirdAreIdenticalNonVowels plate.second ∧
  plate.second = plate.third ∧
  rules.fourthIsDigit plate.fourth

/-- The main theorem to prove --/
theorem probability_of_ANN9 (rules : LicensePlateRules)
  (h_vowels : rules.symbols.vowels.card = 5)
  (h_nonVowels : rules.symbols.nonVowels.card = 21)
  (h_digits : rules.symbols.digits.card = 10)
  (plate : LicensePlate)
  (h_plate : plate = ⟨'A', 'N', 'N', '9'⟩)
  (h_valid : isValidLicensePlate plate rules) :
  (1 : ℚ) / (totalLicensePlates rules : ℚ) = 1 / 1050 :=
sorry

end NUMINAMATH_CALUDE_probability_of_ANN9_l1833_183364


namespace NUMINAMATH_CALUDE_remainder_of_product_l1833_183310

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def product_of_list (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

theorem remainder_of_product (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 3 ∧ d = 10 ∧ n = 20 →
  (product_of_list (arithmetic_sequence a₁ d n)) % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_l1833_183310


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l1833_183379

/-- Given a circle and a point on its tangent, prove the x-coordinate of the point. -/
theorem tangent_point_x_coordinate 
  (a : ℝ) -- x-coordinate of point P
  (h1 : (a + 2)^2 + 16 = ((2 : ℝ) * Real.sqrt 3)^2 + 4) -- P is on the tangent and tangent length is 2√3
  : a = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l1833_183379


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1833_183384

theorem polynomial_division_remainder (x : ℝ) : 
  ∃ (Q : ℝ → ℝ), x^150 = (x^2 - 4*x + 3) * Q x + ((3^150 - 1)*x + (4 - 3^150)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1833_183384


namespace NUMINAMATH_CALUDE_students_called_back_l1833_183320

theorem students_called_back (girls : ℕ) (boys : ℕ) (didnt_make_cut : ℕ) 
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : didnt_make_cut = 17) :
  girls + boys - didnt_make_cut = 26 := by
  sorry

end NUMINAMATH_CALUDE_students_called_back_l1833_183320


namespace NUMINAMATH_CALUDE_decimal_51_to_binary_l1833_183337

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Checks if a list of booleans represents the given binary number -/
def isBinaryRepresentation (bits : List Bool) (binaryStr : String) : Prop :=
  bits.reverse.map (fun b => if b then '1' else '0') = binaryStr.toList

theorem decimal_51_to_binary :
  isBinaryRepresentation (toBinary 51) "110011" := by
  sorry

#eval toBinary 51

end NUMINAMATH_CALUDE_decimal_51_to_binary_l1833_183337


namespace NUMINAMATH_CALUDE_union_of_sets_l1833_183326

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 4}
  A ∪ B = {1, 2, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1833_183326


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l1833_183361

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (m : Line) (n : Line) (α β γ : Plane) :
  parallel m α → perpendicular m β → perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l1833_183361


namespace NUMINAMATH_CALUDE_total_time_in_work_week_l1833_183358

/-- Represents the days of the work week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Commute time for each day of the week -/
def commute_time (day : Weekday) : ℕ :=
  match day with
  | Weekday.Monday => 35
  | Weekday.Tuesday => 45
  | Weekday.Wednesday => 25
  | Weekday.Thursday => 40
  | Weekday.Friday => 30

/-- Additional delay for each day of the week -/
def additional_delay (day : Weekday) : ℕ :=
  match day with
  | Weekday.Monday => 5
  | Weekday.Wednesday => 10
  | Weekday.Friday => 8
  | _ => 0

/-- Security check time for each day of the week -/
def security_check_time (day : Weekday) : ℕ :=
  match day with
  | Weekday.Tuesday => 30
  | Weekday.Thursday => 10
  | _ => 15

/-- Constant time for parking and walking -/
def parking_and_walking_time : ℕ := 8

/-- Total time spent on a given day -/
def daily_total_time (day : Weekday) : ℕ :=
  commute_time day + additional_delay day + security_check_time day + parking_and_walking_time

/-- List of all work days in a week -/
def work_week : List Weekday := [Weekday.Monday, Weekday.Tuesday, Weekday.Wednesday, Weekday.Thursday, Weekday.Friday]

/-- Theorem stating the total time spent in a work week -/
theorem total_time_in_work_week : (work_week.map daily_total_time).sum = 323 := by
  sorry

end NUMINAMATH_CALUDE_total_time_in_work_week_l1833_183358


namespace NUMINAMATH_CALUDE_prob_at_least_two_same_dice_l1833_183382

-- Define the number of dice and sides
def num_dice : ℕ := 5
def num_sides : ℕ := 6

-- Define the total number of outcomes
def total_outcomes : ℕ := num_sides ^ num_dice

-- Define the number of outcomes with all different numbers
def all_different_outcomes : ℕ := num_sides * (num_sides - 1) * (num_sides - 2) * (num_sides - 3) * (num_sides - 4)

-- Define the probability of at least two dice showing the same number
def prob_at_least_two_same : ℚ := 1 - (all_different_outcomes : ℚ) / total_outcomes

-- Theorem statement
theorem prob_at_least_two_same_dice :
  prob_at_least_two_same = 7056 / 7776 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_two_same_dice_l1833_183382


namespace NUMINAMATH_CALUDE_megan_homework_problems_l1833_183359

/-- The total number of homework problems Megan had -/
def total_problems (finished : ℕ) (pages_left : ℕ) (problems_per_page : ℕ) : ℕ :=
  finished + pages_left * problems_per_page

/-- Proof that Megan had 40 homework problems in total -/
theorem megan_homework_problems :
  total_problems 26 2 7 = 40 := by
  sorry

end NUMINAMATH_CALUDE_megan_homework_problems_l1833_183359


namespace NUMINAMATH_CALUDE_ellipse_min_major_axis_l1833_183386

/-- Given an ellipse where the maximum area of a triangle formed by a point on the ellipse and its two foci is 1, 
    the minimum length of the major axis is 2√2. -/
theorem ellipse_min_major_axis (a b c : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) → 
  (b * c = 1) →  -- maximum triangle area condition
  (a^2 = b^2 + c^2) →  -- ellipse equation
  (2 * a ≥ 2 * Real.sqrt 2) ∧ 
  (∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ b₀ * c₀ = 1 ∧ a₀^2 = b₀^2 + c₀^2 ∧ 2 * a₀ = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_min_major_axis_l1833_183386


namespace NUMINAMATH_CALUDE_polynomial_equality_l1833_183373

theorem polynomial_equality (x : ℝ) : 
  let k : ℝ := -9
  let a : ℝ := 15
  let b : ℝ := 72
  (3 * x^2 - 4 * x + 5) * (5 * x^2 + k * x + 8) = 
    15 * x^4 - 47 * x^3 + a * x^2 - b * x + 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1833_183373


namespace NUMINAMATH_CALUDE_fish_tank_leakage_rate_l1833_183314

/-- Proves that the rate of leakage is 1.5 ounces per hour given the problem conditions -/
theorem fish_tank_leakage_rate 
  (bucket_capacity : ℝ) 
  (leakage_duration : ℝ) 
  (h1 : bucket_capacity = 36) 
  (h2 : leakage_duration = 12) 
  (h3 : bucket_capacity = 2 * (leakage_duration * leakage_rate)) : 
  leakage_rate = 1.5 := by
  sorry

#check fish_tank_leakage_rate

end NUMINAMATH_CALUDE_fish_tank_leakage_rate_l1833_183314


namespace NUMINAMATH_CALUDE_inverse_inequality_implies_reverse_l1833_183307

theorem inverse_inequality_implies_reverse (a b : ℝ) :
  (1 / a < 1 / b) ∧ (1 / b < 0) → a > b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_implies_reverse_l1833_183307


namespace NUMINAMATH_CALUDE_tangent_problem_l1833_183371

theorem tangent_problem (α : Real) 
  (h : Real.tan (π/4 + α) = 1/2) : 
  (Real.tan α = -1/3) ∧ 
  ((Real.sin (2*α) - Real.cos α ^ 2) / (2 + Real.cos (2*α)) = (2 * Real.tan α - 1) / (3 + Real.tan α ^ 2)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l1833_183371


namespace NUMINAMATH_CALUDE_base5_conversion_and_modulo_l1833_183383

/-- Converts a base 5 number to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Computes the modulo of a number -/
def modulo (n m : Nat) : Nat :=
  n % m

theorem base5_conversion_and_modulo :
  let base5Num : List Nat := [4, 1, 0, 1, 2]  -- 21014 in base 5, least significant digit first
  let base10Num : Nat := base5ToBase10 base5Num
  base10Num = 1384 ∧ modulo base10Num 7 = 6 := by
  sorry

#eval base5ToBase10 [4, 1, 0, 1, 2]  -- Should output 1384
#eval modulo 1384 7  -- Should output 6

end NUMINAMATH_CALUDE_base5_conversion_and_modulo_l1833_183383


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l1833_183377

theorem cubic_polynomial_property (p q r : ℝ) : 
  let Q : ℝ → ℝ := λ x => x^3 + p*x^2 + q*x + r
  let mean_zeros := -p / 3
  let product_zeros := -r
  let sum_coefficients := 1 + p + q + r
  (mean_zeros = product_zeros ∧ product_zeros = sum_coefficients ∧ r = 3) →
  q = -16 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l1833_183377


namespace NUMINAMATH_CALUDE_age_difference_is_nine_l1833_183347

/-- Represents a year in the 19th or 20th century -/
structure Year where
  century : Nat
  tens : Nat
  ones : Nat
  h : century ∈ [18, 19] ∧ tens < 10 ∧ ones < 10

/-- The age of a person at a given meeting year -/
def age (birth : Year) (meetingYear : Nat) : Nat :=
  meetingYear - (birth.century * 100 + birth.tens * 10 + birth.ones)

theorem age_difference_is_nine :
  ∀ (peterBirth : Year) (paulBirth : Year) (meetingYear : Nat),
    peterBirth.century = 18 →
    paulBirth.century = 19 →
    age peterBirth meetingYear = peterBirth.century + peterBirth.tens + peterBirth.ones + 9 →
    age paulBirth meetingYear = paulBirth.century + paulBirth.tens + paulBirth.ones + 10 →
    age peterBirth meetingYear - age paulBirth meetingYear = 9 := by
  sorry

#check age_difference_is_nine

end NUMINAMATH_CALUDE_age_difference_is_nine_l1833_183347


namespace NUMINAMATH_CALUDE_stating_max_perpendicular_diagonals_correct_l1833_183375

/-- 
Given a regular n-gon with n ≥ 3, this function returns the maximum number of diagonals
that can be drawn such that any intersecting pair is perpendicular.
-/
def maxPerpendicularDiagonals (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 2 else n - 3

/-- 
Theorem stating that maxPerpendicularDiagonals correctly computes the maximum number
of diagonals in a regular n-gon (n ≥ 3) such that any intersecting pair is perpendicular.
-/
theorem max_perpendicular_diagonals_correct (n : ℕ) (h : n ≥ 3) :
  maxPerpendicularDiagonals n = 
    if n % 2 = 0 then n - 2 else n - 3 :=
by sorry

end NUMINAMATH_CALUDE_stating_max_perpendicular_diagonals_correct_l1833_183375


namespace NUMINAMATH_CALUDE_chastity_gummy_packs_l1833_183396

/-- Given Chastity's candy purchase scenario, prove the number of gummy packs bought. -/
theorem chastity_gummy_packs :
  ∀ (initial_money : ℚ) 
    (remaining_money : ℚ) 
    (lollipop_count : ℕ) 
    (lollipop_price : ℚ) 
    (gummy_pack_price : ℚ),
  initial_money = 15 →
  remaining_money = 5 →
  lollipop_count = 4 →
  lollipop_price = 3/2 →
  gummy_pack_price = 2 →
  ∃ (gummy_pack_count : ℕ),
    gummy_pack_count = 2 ∧
    initial_money - remaining_money = 
      (lollipop_count : ℚ) * lollipop_price + (gummy_pack_count : ℚ) * gummy_pack_price :=
by sorry

end NUMINAMATH_CALUDE_chastity_gummy_packs_l1833_183396


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1833_183300

def repeating_decimal_123 : ℚ := 123 / 999
def repeating_decimal_0123 : ℚ := 123 / 9999
def repeating_decimal_000123 : ℚ := 123 / 999999

theorem sum_of_repeating_decimals :
  repeating_decimal_123 + repeating_decimal_0123 + repeating_decimal_000123 =
  (123 * 1000900) / (999 * 9999 * 100001) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1833_183300


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1833_183372

theorem polynomial_simplification (x : ℝ) :
  (3 * x^6 + 2 * x^5 + x^4 + x - 5) - (x^6 + 3 * x^5 + 2 * x^3 + 6) =
  2 * x^6 - x^5 + x^4 - 2 * x^3 + x + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1833_183372


namespace NUMINAMATH_CALUDE_swimming_pool_length_l1833_183390

theorem swimming_pool_length 
  (width : ℝ) 
  (water_removed : ℝ) 
  (water_level_lowered : ℝ) 
  (cubic_foot_to_gallon : ℝ) :
  width = 20 →
  water_removed = 4500 →
  water_level_lowered = 0.5 →
  cubic_foot_to_gallon = 7.5 →
  ∃ (length : ℝ), length = 60 ∧ 
    water_removed / cubic_foot_to_gallon = length * width * water_level_lowered :=
by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_length_l1833_183390


namespace NUMINAMATH_CALUDE_number_of_americans_l1833_183333

theorem number_of_americans (total : ℕ) (chinese : ℕ) (australians : ℕ) 
  (h1 : total = 49)
  (h2 : chinese = 22)
  (h3 : australians = 11) :
  total - chinese - australians = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_americans_l1833_183333


namespace NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l1833_183302

/-- The area of a rectangle inscribed in a square, given other inscribed shapes -/
theorem area_of_inscribed_rectangle (s : ℝ) (r1_length r1_width : ℝ) (sq_side : ℝ) :
  s = 4 →
  r1_length = 2 →
  r1_width = 4 →
  sq_side = 1 →
  ∃ (r2_length r2_width : ℝ),
    r2_length * r2_width = s^2 - (r1_length * r1_width + sq_side^2) :=
by sorry

end NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l1833_183302


namespace NUMINAMATH_CALUDE_quartic_real_root_l1833_183392

theorem quartic_real_root 
  (A B C D E : ℝ) 
  (h : ∃ t : ℝ, t > 1 ∧ A * t^2 + (C - B) * t + (E - D) = 0) :
  ∃ x : ℝ, A * x^4 + B * x^3 + C * x^2 + D * x + E = 0 :=
sorry

end NUMINAMATH_CALUDE_quartic_real_root_l1833_183392


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1833_183354

/-- A function satisfying the given functional equation and differentiability condition -/
class FunctionalEquationSolution (f : ℝ → ℝ) : Prop where
  equation : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
  smooth : ContDiff ℝ ⊤ f

/-- The main theorem stating the form of the solution -/
theorem functional_equation_solution (f : ℝ → ℝ) [FunctionalEquationSolution f] :
  ∃ a : ℝ, ∀ x : ℝ, f x = x^2 + a * x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1833_183354


namespace NUMINAMATH_CALUDE_swimming_pool_payment_analysis_l1833_183327

/-- Represents the swimming pool payment methods -/
structure SwimmingPoolPayment where
  membershipCost : ℕ
  memberSwimCost : ℕ
  nonMemberSwimCost : ℕ

/-- Calculates the cost for a given number of swims using Method 1 -/
def method1Cost (p : SwimmingPoolPayment) (swims : ℕ) : ℕ :=
  p.membershipCost + p.memberSwimCost * swims

/-- Calculates the cost for a given number of swims using Method 2 -/
def method2Cost (p : SwimmingPoolPayment) (swims : ℕ) : ℕ :=
  p.nonMemberSwimCost * swims

/-- Calculates the maximum number of swims possible with a given budget using Method 1 -/
def maxSwimMethod1 (p : SwimmingPoolPayment) (budget : ℕ) : ℕ :=
  (budget - p.membershipCost) / p.memberSwimCost

/-- Calculates the maximum number of swims possible with a given budget using Method 2 -/
def maxSwimMethod2 (p : SwimmingPoolPayment) (budget : ℕ) : ℕ :=
  budget / p.nonMemberSwimCost

theorem swimming_pool_payment_analysis 
  (p : SwimmingPoolPayment) 
  (h1 : p.membershipCost = 200)
  (h2 : p.memberSwimCost = 10)
  (h3 : p.nonMemberSwimCost = 30) :
  (method1Cost p 3 = 230) ∧
  (method2Cost p 9 < method1Cost p 9) ∧
  (maxSwimMethod1 p 600 > maxSwimMethod2 p 600) := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_payment_analysis_l1833_183327


namespace NUMINAMATH_CALUDE_parallel_planes_and_line_l1833_183317

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- Define the "not contained in" relation for lines and planes
variable (line_not_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem parallel_planes_and_line 
  (α β : Plane) (a : Line) 
  (h1 : plane_parallel α β)
  (h2 : line_not_in_plane a α)
  (h3 : line_not_in_plane a β)
  (h4 : line_parallel_plane a α) :
  line_parallel_plane a β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_and_line_l1833_183317


namespace NUMINAMATH_CALUDE_lunks_needed_for_two_dozen_oranges_l1833_183391

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks_rate : ℚ := 4 / 2

/-- Exchange rate between kunks and oranges -/
def kunks_to_oranges_rate : ℚ := 3 / 6

/-- Number of oranges in two dozen -/
def two_dozen : ℕ := 24

/-- The number of lunks required to purchase two dozen oranges -/
def lunks_for_two_dozen : ℕ := 24

theorem lunks_needed_for_two_dozen_oranges :
  (two_dozen : ℚ) / kunks_to_oranges_rate * lunks_to_kunks_rate = lunks_for_two_dozen := by
  sorry

end NUMINAMATH_CALUDE_lunks_needed_for_two_dozen_oranges_l1833_183391


namespace NUMINAMATH_CALUDE_solve_baseball_card_problem_l1833_183369

def baseball_card_problem (patricia_money : ℕ) (card_price : ℕ) : Prop :=
  let lisa_money := 5 * patricia_money
  let charlotte_money := lisa_money / 2
  let james_money := 10 + charlotte_money + lisa_money
  let total_money := patricia_money + lisa_money + charlotte_money + james_money
  card_price - total_money = 144

theorem solve_baseball_card_problem :
  baseball_card_problem 6 250 := by
  sorry

end NUMINAMATH_CALUDE_solve_baseball_card_problem_l1833_183369


namespace NUMINAMATH_CALUDE_intersection_point_of_line_with_x_axis_l1833_183365

/-- The intersection point of the line y = 2x - 4 with the x-axis is (2, 0). -/
theorem intersection_point_of_line_with_x_axis :
  let f : ℝ → ℝ := λ x ↦ 2 * x - 4
  ∃! x : ℝ, f x = 0 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_point_of_line_with_x_axis_l1833_183365


namespace NUMINAMATH_CALUDE_train_length_problem_l1833_183332

/-- Given two trains running in opposite directions, calculate the length of the second train. -/
theorem train_length_problem (length_A : ℝ) (speed_A speed_B : ℝ) (crossing_time : ℝ) :
  length_A = 230 →
  speed_A = 120 * 1000 / 3600 →
  speed_B = 80 * 1000 / 3600 →
  crossing_time = 9 →
  ∃ length_B : ℝ, abs (length_B - 269.95) < 0.01 ∧
    length_A + length_B = (speed_A + speed_B) * crossing_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_problem_l1833_183332


namespace NUMINAMATH_CALUDE_equation_system_solution_nature_l1833_183330

/-- Given a system of equations:
    x - y + z - w = 2
    x^2 - y^2 + z^2 - w^2 = 6
    x^3 - y^3 + z^3 - w^3 = 20
    x^4 - y^4 + z^4 - w^4 = 66
    Prove that this system either has no solutions or infinitely many solutions. -/
theorem equation_system_solution_nature :
  let s₁ : ℝ := 2
  let s₂ : ℝ := 6
  let s₃ : ℝ := 20
  let s₄ : ℝ := 66
  let b₁ : ℝ := s₁
  let b₂ : ℝ := (s₁^2 - s₂) / 2
  let b₃ : ℝ := (s₁^3 - 3*s₁*s₂ + 2*s₃) / 6
  let b₄ : ℝ := (s₁^4 - 6*s₁^2*s₂ + 3*s₂^2 + 8*s₁*s₃ - 6*s₄) / 24
  b₂^2 - b₁*b₃ = 0 →
  (∀ x y z w : ℝ, 
    x - y + z - w = s₁ ∧
    x^2 - y^2 + z^2 - w^2 = s₂ ∧
    x^3 - y^3 + z^3 - w^3 = s₃ ∧
    x^4 - y^4 + z^4 - w^4 = s₄ →
    (∀ ε > 0, ∃ x' y' z' w' : ℝ,
      x' - y' + z' - w' = s₁ ∧
      x'^2 - y'^2 + z'^2 - w'^2 = s₂ ∧
      x'^3 - y'^3 + z'^3 - w'^3 = s₃ ∧
      x'^4 - y'^4 + z'^4 - w'^4 = s₄ ∧
      ((x' - x)^2 + (y' - y)^2 + (z' - z)^2 + (w' - w)^2 < ε^2) ∧
      (x' ≠ x ∨ y' ≠ y ∨ z' ≠ z ∨ w' ≠ w))) ∨
  (¬∃ x y z w : ℝ,
    x - y + z - w = s₁ ∧
    x^2 - y^2 + z^2 - w^2 = s₂ ∧
    x^3 - y^3 + z^3 - w^3 = s₃ ∧
    x^4 - y^4 + z^4 - w^4 = s₄) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_system_solution_nature_l1833_183330


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1833_183380

theorem other_root_of_quadratic (m : ℚ) :
  (3 : ℚ) ∈ {x : ℚ | 3 * x^2 + m * x = 5} →
  (-5/9 : ℚ) ∈ {x : ℚ | 3 * x^2 + m * x = 5} :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1833_183380


namespace NUMINAMATH_CALUDE_two_fours_equal_64_l1833_183335

theorem two_fours_equal_64 : 
  Real.sqrt ((Real.sqrt (Real.sqrt 4)) ^ (4 * 3 * 2 * 1)) = 64 := by sorry

end NUMINAMATH_CALUDE_two_fours_equal_64_l1833_183335


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_l1833_183352

theorem complex_arithmetic_expression : 
  10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.3 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_l1833_183352


namespace NUMINAMATH_CALUDE_f_2_equals_5_l1833_183394

def f (x : ℝ) : ℝ := 2 * (x - 1) + 3

theorem f_2_equals_5 : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_2_equals_5_l1833_183394
