import Mathlib

namespace NUMINAMATH_CALUDE_solution_count_l3415_341549

/-- The number of solutions to the system of equations y = (x+1)^3 and xy + y = 1 -/
def num_solutions : ℕ := 4

/-- The number of real solutions to the system of equations y = (x+1)^3 and xy + y = 1 -/
def num_real_solutions : ℕ := 2

/-- The number of complex solutions to the system of equations y = (x+1)^3 and xy + y = 1 -/
def num_complex_solutions : ℕ := 2

/-- Definition of the first equation: y = (x+1)^3 -/
def equation1 (x y : ℂ) : Prop := y = (x + 1)^3

/-- Definition of the second equation: xy + y = 1 -/
def equation2 (x y : ℂ) : Prop := x * y + y = 1

/-- A solution is a pair (x, y) that satisfies both equations -/
def is_solution (x y : ℂ) : Prop := equation1 x y ∧ equation2 x y

/-- The main theorem stating the number and nature of solutions -/
theorem solution_count :
  (∃ (s : Finset (ℂ × ℂ)), s.card = num_solutions ∧
    (∀ (p : ℂ × ℂ), p ∈ s ↔ is_solution p.1 p.2) ∧
    (∃ (sr : Finset (ℝ × ℝ)), sr.card = num_real_solutions ∧
      (∀ (p : ℝ × ℝ), p ∈ sr ↔ is_solution p.1 p.2)) ∧
    (∃ (sc : Finset (ℂ × ℂ)), sc.card = num_complex_solutions ∧
      (∀ (p : ℂ × ℂ), p ∈ sc ↔ (is_solution p.1 p.2 ∧ ¬(p.1.im = 0 ∧ p.2.im = 0))))) :=
sorry

end NUMINAMATH_CALUDE_solution_count_l3415_341549


namespace NUMINAMATH_CALUDE_island_with_2008_roads_sum_of_roads_formula_l3415_341530

def number_of_roads (n : ℕ) : ℕ := 55 + n.choose 2

def sum_of_roads (n : ℕ) : ℕ := 55 * n + (n + 1).choose 3

theorem island_with_2008_roads : ∃ n : ℕ, n > 0 ∧ number_of_roads n = 2008 := by sorry

theorem sum_of_roads_formula (n : ℕ) (h : n > 0) : 
  (Finset.range n).sum (λ k => number_of_roads (k + 1)) = sum_of_roads n := by sorry

end NUMINAMATH_CALUDE_island_with_2008_roads_sum_of_roads_formula_l3415_341530


namespace NUMINAMATH_CALUDE_base4_arithmetic_theorem_l3415_341540

/-- Converts a number from base 4 to base 10 --/
def base4To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 --/
def base10To4 (n : ℕ) : ℕ := sorry

/-- Performs arithmetic operations in base 4 --/
def base4Arithmetic (a b c d : ℕ) : ℕ :=
  let a10 := base4To10 a
  let b10 := base4To10 b
  let c10 := base4To10 c
  let d10 := base4To10 d
  let result := a10 * b10 / c10 * d10
  base10To4 result

theorem base4_arithmetic_theorem :
  base4Arithmetic 231 21 3 2 = 10232 := by sorry

end NUMINAMATH_CALUDE_base4_arithmetic_theorem_l3415_341540


namespace NUMINAMATH_CALUDE_cubic_root_h_value_l3415_341593

theorem cubic_root_h_value : ∀ h : ℚ, 
  (3 : ℚ)^3 + h * 3 - 20 = 0 → h = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_h_value_l3415_341593


namespace NUMINAMATH_CALUDE_alpha_beta_sum_l3415_341526

theorem alpha_beta_sum (α β : ℝ) : 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 72*x + 1233) / (x^2 + 81*x - 3969)) →
  α + β = 143 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_sum_l3415_341526


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3415_341579

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (1, 6)
  parallel a b → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3415_341579


namespace NUMINAMATH_CALUDE_clarinet_cost_calculation_l3415_341553

/-- The cost of items purchased at a music store -/
structure MusicStorePurchase where
  total_spent : ℝ
  songbook_cost : ℝ
  clarinet_cost : ℝ

/-- Theorem stating the cost of the clarinet given the total spent and songbook cost -/
theorem clarinet_cost_calculation (purchase : MusicStorePurchase) 
  (h1 : purchase.total_spent = 141.54)
  (h2 : purchase.songbook_cost = 11.24)
  : purchase.clarinet_cost = 130.30 := by
  sorry

end NUMINAMATH_CALUDE_clarinet_cost_calculation_l3415_341553


namespace NUMINAMATH_CALUDE_function_value_at_two_l3415_341556

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 0, prove that f(2) = -16 -/
theorem function_value_at_two (a b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^5 + a*x^3 + b*x - 8
  f (-2) = 0 → f 2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l3415_341556


namespace NUMINAMATH_CALUDE_line_solution_l3415_341594

/-- Given a line y = ax + b (a ≠ 0) passing through points (0,4) and (-3,0),
    the solution to ax + b = 0 is x = -3. -/
theorem line_solution (a b : ℝ) (ha : a ≠ 0) :
  (4 = b) →                        -- Line passes through (0,4)
  (0 = -3*a + b) →                 -- Line passes through (-3,0)
  (∀ x, a*x + b = 0 ↔ x = -3) :=   -- Solution to ax + b = 0 is x = -3
by
  sorry

end NUMINAMATH_CALUDE_line_solution_l3415_341594


namespace NUMINAMATH_CALUDE_mower_team_size_l3415_341567

/-- Represents the mowing rate of one mower per day -/
def mower_rate : ℝ := 1

/-- Represents the area of the smaller meadow -/
def small_meadow : ℝ := 2 * mower_rate

/-- Represents the area of the larger meadow -/
def large_meadow : ℝ := 2 * small_meadow

/-- Represents the number of mowers in the team -/
def team_size : ℕ := 8

theorem mower_team_size :
  (team_size : ℝ) * mower_rate / 2 + (team_size : ℝ) * mower_rate / 2 = large_meadow ∧
  (team_size : ℝ) * mower_rate / 4 + mower_rate = small_meadow :=
by sorry

#check mower_team_size

end NUMINAMATH_CALUDE_mower_team_size_l3415_341567


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3415_341570

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →  -- a1, a3, and a4 form a geometric sequence
  a 2 = -6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3415_341570


namespace NUMINAMATH_CALUDE_trig_simplification_l3415_341565

theorem trig_simplification :
  let x : Real := 40 * π / 180
  let y : Real := 50 * π / 180
  (Real.sqrt (1 - 2 * Real.sin x * Real.cos x)) / (Real.cos x - Real.sqrt (1 - Real.sin y ^ 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3415_341565


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3415_341546

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  is_positive : ∀ n, a n > 0

/-- Theorem: In an arithmetic sequence with positive terms, if 2a₆ + 2a₈ = a₇², then a₇ = 4 -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
    (h : 2 * seq.a 6 + 2 * seq.a 8 = (seq.a 7) ^ 2) : 
    seq.a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3415_341546


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3415_341518

/-- Given a line l: x + 2y + p = 0 (p ∈ ℝ), prove that 2x - y - 1 = 0 is the equation of the line
    passing through the point P(2,3) and perpendicular to l. -/
theorem perpendicular_line_equation (p : ℝ) :
  let l : ℝ → ℝ → Prop := fun x y ↦ x + 2 * y + p = 0
  let perpendicular_line : ℝ → ℝ → Prop := fun x y ↦ 2 * x - y - 1 = 0
  (∀ x y, l x y → (perpendicular_line x y → False) → False) ∧
  perpendicular_line 2 3 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3415_341518


namespace NUMINAMATH_CALUDE_jeremy_payment_l3415_341539

/-- The total amount owed to Jeremy for cleaning rooms and washing windows -/
theorem jeremy_payment (room_rate : ℚ) (window_rate : ℚ) (rooms_cleaned : ℚ) (windows_washed : ℚ)
  (h1 : room_rate = 13 / 3)
  (h2 : window_rate = 5 / 2)
  (h3 : rooms_cleaned = 8 / 5)
  (h4 : windows_washed = 11 / 4) :
  room_rate * rooms_cleaned + window_rate * windows_washed = 553 / 40 :=
by sorry

end NUMINAMATH_CALUDE_jeremy_payment_l3415_341539


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3415_341571

/-- The distance between the vertices of the hyperbola x^2/48 - y^2/16 = 1 is 8√3 -/
theorem hyperbola_vertices_distance :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ x^2 / 48 - y^2 / 16
  ∃ (a b : ℝ), a ≠ b ∧ f (a, 0) = 1 ∧ f (b, 0) = 1 ∧ |a - b| = 8 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3415_341571


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3415_341555

theorem triangle_angle_problem (a b : ℝ) (B : ℝ) (A : ℝ) :
  a = Real.sqrt 3 →
  b = 1 →
  B = 30 * π / 180 →
  0 < A →
  A < π →
  (A = π / 3 ∨ A = 2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3415_341555


namespace NUMINAMATH_CALUDE_area_of_larger_rectangle_l3415_341507

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The area of a larger rectangle formed by six identical smaller rectangles -/
def largerRectangleArea (smallRect : Rectangle) : ℝ :=
  (3 * smallRect.width) * (2 * smallRect.length)

theorem area_of_larger_rectangle :
  ∀ (smallRect : Rectangle),
    smallRect.length = 2 * smallRect.width →
    smallRect.length + smallRect.width = 21 →
    largerRectangleArea smallRect = 588 := by
  sorry

end NUMINAMATH_CALUDE_area_of_larger_rectangle_l3415_341507


namespace NUMINAMATH_CALUDE_max_sum_of_four_numbers_l3415_341595

theorem max_sum_of_four_numbers (a b c d : ℕ) : 
  a < b → b < c → c < d → 
  (b + d) + (c + d) + (a + b + c) + (a + b + d) = 2017 →
  a + b + c + d ≤ 1006 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_four_numbers_l3415_341595


namespace NUMINAMATH_CALUDE_prime_factor_sum_l3415_341577

theorem prime_factor_sum (w x y z k : ℕ) :
  2^w * 3^x * 5^y * 7^z * 11^k = 2520 →
  2*w + 3*x + 5*y + 7*z + 11*k = 24 := by
  sorry

end NUMINAMATH_CALUDE_prime_factor_sum_l3415_341577


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3415_341597

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) * Real.sqrt (x + 3) ≥ 0

-- Define the solution set
def solution_set : Set ℝ := {-3} ∪ Set.Ici 2

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3415_341597


namespace NUMINAMATH_CALUDE_min_a_for_polynomial_with_two_zeros_in_unit_interval_l3415_341541

theorem min_a_for_polynomial_with_two_zeros_in_unit_interval : 
  ∃ (a b c : ℤ), 
    (∀ (a' : ℤ), a' > 0 ∧ a' < a →
      ¬∃ (b' c' : ℤ), ∃ (x y : ℝ), 
        0 < x ∧ x < y ∧ y < 1 ∧
        a' * x^2 - b' * x + c' = 0 ∧
        a' * y^2 - b' * y + c' = 0) ∧
    (∃ (x y : ℝ), 
      0 < x ∧ x < y ∧ y < 1 ∧
      a * x^2 - b * x + c = 0 ∧
      a * y^2 - b * y + c = 0) ∧
    a = 5 := by sorry

end NUMINAMATH_CALUDE_min_a_for_polynomial_with_two_zeros_in_unit_interval_l3415_341541


namespace NUMINAMATH_CALUDE_marathon_average_time_l3415_341599

-- Define the marathon distance in miles
def marathonDistance : ℕ := 24

-- Define the total time in minutes (3 hours and 36 minutes = 216 minutes)
def totalTimeMinutes : ℕ := 3 * 60 + 36

-- Define the average time per mile
def averageTimePerMile : ℚ := totalTimeMinutes / marathonDistance

-- Theorem statement
theorem marathon_average_time :
  averageTimePerMile = 9 := by sorry

end NUMINAMATH_CALUDE_marathon_average_time_l3415_341599


namespace NUMINAMATH_CALUDE_gcd_upper_bound_l3415_341574

theorem gcd_upper_bound (a b c : ℕ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  Nat.gcd (a * b + 1) (Nat.gcd (a * c + 1) (b * c + 1)) ≤ (a + b + c) / 3 :=
sorry

end NUMINAMATH_CALUDE_gcd_upper_bound_l3415_341574


namespace NUMINAMATH_CALUDE_book_purchase_total_price_l3415_341551

theorem book_purchase_total_price
  (total_books : ℕ)
  (math_books : ℕ)
  (math_book_price : ℕ)
  (history_book_price : ℕ)
  (h1 : total_books = 80)
  (h2 : math_books = 27)
  (h3 : math_book_price = 4)
  (h4 : history_book_price = 5) :
  let history_books := total_books - math_books
  let total_price := math_books * math_book_price + history_books * history_book_price
  total_price = 373 := by
sorry

end NUMINAMATH_CALUDE_book_purchase_total_price_l3415_341551


namespace NUMINAMATH_CALUDE_correct_calculation_l3415_341536

theorem correct_calculation (x : ℚ) : x * 9 = 153 → x * 6 = 102 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3415_341536


namespace NUMINAMATH_CALUDE_permutation_count_modulo_l3415_341562

/-- The number of characters in the string -/
def string_length : ℕ := 15

/-- The number of A's in the string -/
def num_A : ℕ := 4

/-- The number of B's in the string -/
def num_B : ℕ := 5

/-- The number of C's in the string -/
def num_C : ℕ := 5

/-- The number of D's in the string -/
def num_D : ℕ := 2

/-- The length of the first segment where A's are not allowed -/
def first_segment : ℕ := 4

/-- The length of the second segment where B's are not allowed -/
def second_segment : ℕ := 5

/-- The length of the third segment where C's and D's are not allowed -/
def third_segment : ℕ := 6

/-- The function to calculate the number of valid permutations -/
def num_permutations : ℕ := sorry

theorem permutation_count_modulo :
  num_permutations ≡ 715 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_permutation_count_modulo_l3415_341562


namespace NUMINAMATH_CALUDE_deans_vacation_cost_l3415_341596

/-- The total cost of a group vacation given the number of people and individual costs -/
def vacation_cost (num_people : ℕ) (rent transport food activities : ℚ) : ℚ :=
  num_people * (rent + transport + food + activities)

/-- Theorem stating the total cost for Dean's group vacation -/
theorem deans_vacation_cost :
  vacation_cost 7 70 25 55 40 = 1330 := by
  sorry

end NUMINAMATH_CALUDE_deans_vacation_cost_l3415_341596


namespace NUMINAMATH_CALUDE_specific_sequence_common_difference_l3415_341572

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℝ
  last_term : ℝ
  sum : ℝ
  is_arithmetic : ℝ → ℝ → ℝ → Prop

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ := 
  sorry

/-- Theorem stating the common difference of the specific sequence -/
theorem specific_sequence_common_difference :
  ∃ (seq : ArithmeticSequence), 
    seq.first_term = 5 ∧ 
    seq.last_term = 50 ∧ 
    seq.sum = 495 ∧ 
    common_difference seq = 45 / 17 := by
  sorry

end NUMINAMATH_CALUDE_specific_sequence_common_difference_l3415_341572


namespace NUMINAMATH_CALUDE_distance_case1_distance_case2_distance_formula_l3415_341521

-- Define a function to calculate the distance between two points on a number line
def distance (x1 x2 : ℝ) : ℝ := |x2 - x1|

-- Theorem for Case 1
theorem distance_case1 : distance 2 3 = 1 := by sorry

-- Theorem for Case 2
theorem distance_case2 : distance (-4) (-8) = 4 := by sorry

-- General theorem
theorem distance_formula (x1 x2 : ℝ) : 
  distance x1 x2 = |x2 - x1| := by sorry

end NUMINAMATH_CALUDE_distance_case1_distance_case2_distance_formula_l3415_341521


namespace NUMINAMATH_CALUDE_jacqueline_boxes_l3415_341531

/-- The number of erasers per box -/
def erasers_per_box : ℕ := 10

/-- The total number of erasers Jacqueline has -/
def total_erasers : ℕ := 40

/-- The number of boxes Jacqueline has -/
def num_boxes : ℕ := total_erasers / erasers_per_box

theorem jacqueline_boxes : num_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_jacqueline_boxes_l3415_341531


namespace NUMINAMATH_CALUDE_basketball_surface_area_l3415_341527

/-- The surface area of a sphere with diameter 24 centimeters is 576π square centimeters. -/
theorem basketball_surface_area : 
  let diameter : ℝ := 24
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 576 * Real.pi := by sorry

end NUMINAMATH_CALUDE_basketball_surface_area_l3415_341527


namespace NUMINAMATH_CALUDE_equation_solutions_l3415_341508

noncomputable def solution_equation (a b c d x : ℝ) : Prop :=
  (a*x + b) / (a + b*x) + (c*x + d) / (c + d*x) = 
  (a*x - b) / (a - b*x) + (c*x - d) / (c - d*x)

theorem equation_solutions 
  (a b c d : ℝ) 
  (h1 : a*d + b*c ≠ 0) 
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) 
  (h5 : d ≠ 0) :
  (∀ x : ℝ, x ≠ a/b ∧ x ≠ -a/b ∧ x ≠ c/d ∧ x ≠ -c/d →
    (x = 1 ∨ x = -1 ∨ x = Real.sqrt (a*c/(b*d)) ∨ x = -Real.sqrt (a*c/(b*d))) ↔ 
    solution_equation a b c d x) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3415_341508


namespace NUMINAMATH_CALUDE_max_value_product_sum_l3415_341586

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ A' M' C' : ℕ, A' + M' + C' = 15 →
    A' * M' * C' + A' * M' + M' * C' + C' * A' ≤ A * M * C + A * M + M * C + C * A) →
  A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l3415_341586


namespace NUMINAMATH_CALUDE_complex_magnitude_l3415_341590

theorem complex_magnitude (z : ℂ) (h : (1 + 2*I)*z = -3 + 4*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3415_341590


namespace NUMINAMATH_CALUDE_yellow_hard_hats_l3415_341545

def initial_pink : ℕ := 26
def initial_green : ℕ := 15
def carl_takes_pink : ℕ := 4
def john_takes_pink : ℕ := 6
def john_takes_green : ℕ := 2 * john_takes_pink
def remaining_total : ℕ := 43

theorem yellow_hard_hats (initial_yellow : ℕ) :
  initial_pink - carl_takes_pink - john_takes_pink +
  initial_green - john_takes_green +
  initial_yellow = remaining_total →
  initial_yellow = 24 :=
by sorry

end NUMINAMATH_CALUDE_yellow_hard_hats_l3415_341545


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l3415_341500

/-- The lateral area of a cylinder with volume π and base radius 1 is 2π -/
theorem cylinder_lateral_area (V : ℝ) (r : ℝ) (h : ℝ) : 
  V = π → r = 1 → V = π * r^2 * h → 2 * π * r * h = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l3415_341500


namespace NUMINAMATH_CALUDE_combination_sum_equals_4950_l3415_341575

theorem combination_sum_equals_4950 : Nat.choose 99 98 + Nat.choose 99 97 = 4950 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equals_4950_l3415_341575


namespace NUMINAMATH_CALUDE_lee_annual_salary_l3415_341528

/-- Lee's annual salary calculation --/
theorem lee_annual_salary (monthly_savings : ℕ) (saving_months : ℕ) : 
  monthly_savings = 1000 →
  saving_months = 10 →
  (monthly_savings * saving_months : ℕ) = (2 * (60000 / 12) : ℕ) →
  60000 = (monthly_savings * saving_months * 6 : ℕ) := by
  sorry

#check lee_annual_salary

end NUMINAMATH_CALUDE_lee_annual_salary_l3415_341528


namespace NUMINAMATH_CALUDE_flour_difference_l3415_341502

theorem flour_difference : (7 : ℚ) / 8 - (5 : ℚ) / 6 = (1 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_flour_difference_l3415_341502


namespace NUMINAMATH_CALUDE_amelias_dinner_l3415_341520

/-- Amelia's dinner problem -/
theorem amelias_dinner (first_course second_course dessert remaining_money : ℝ) 
  (h1 : first_course = 15)
  (h2 : second_course = first_course + 5)
  (h3 : dessert = 0.25 * second_course)
  (h4 : remaining_money = 20) : 
  first_course + second_course + dessert + remaining_money = 60 := by
  sorry

end NUMINAMATH_CALUDE_amelias_dinner_l3415_341520


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l3415_341506

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Performs one operation on the urn state -/
def performOperation (state : UrnState) : UrnState :=
  sorry

/-- Calculates the probability of drawing a specific color -/
def drawProbability (state : UrnState) (color : Color) : ℚ :=
  sorry

/-- Calculates the probability of a specific sequence of draws -/
def sequenceProbability (sequence : List Color) : ℚ :=
  sorry

/-- Counts the number of valid sequences resulting in 3 red and 3 blue balls -/
def countValidSequences : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability_theorem :
  let initialState : UrnState := ⟨1, 1⟩
  let finalState : UrnState := ⟨3, 3⟩
  let numOperations : ℕ := 5
  (countValidSequences * sequenceProbability [Color.Red, Color.Red, Color.Red, Color.Blue, Color.Blue]) = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_theorem_l3415_341506


namespace NUMINAMATH_CALUDE_solve_system_l3415_341578

theorem solve_system (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : x * y = 2 * (x + y))
  (eq2 : y * z = 4 * (y + z))
  (eq3 : x * z = 8 * (x + z)) :
  x = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3415_341578


namespace NUMINAMATH_CALUDE_incorrect_expression_l3415_341588

theorem incorrect_expression (x y : ℝ) (h : x / y = 3 / 4) : 
  (x - y) / y = -1 / 4 ∧ (x - y) / y ≠ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l3415_341588


namespace NUMINAMATH_CALUDE_cosine_identities_l3415_341513

theorem cosine_identities :
  (Real.cos (36 * π / 180) - Real.cos (72 * π / 180) = 1/2) ∧
  (Real.cos (π/7) - Real.cos (2*π/7) + Real.cos (3*π/7) = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_cosine_identities_l3415_341513


namespace NUMINAMATH_CALUDE_min_packs_for_120_cans_l3415_341564

/-- Represents the available pack sizes for soda cans -/
inductive PackSize
  | small : PackSize  -- 9 cans
  | medium : PackSize -- 18 cans
  | large : PackSize  -- 30 cans

/-- Calculates the number of cans in a given pack -/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | .small => 9
  | .medium => 18
  | .large => 30

/-- Represents a combination of packs -/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination -/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Checks if a combination qualifies for the promotion -/
def qualifiesForPromotion (c : PackCombination) : Bool :=
  c.large ≥ 2

/-- Represents the store's promotion rule -/
def applyPromotion (c : PackCombination) : PackCombination :=
  if qualifiesForPromotion c then
    { c with small := c.small + 1 }
  else
    c

/-- Calculates the total number of packs in a combination -/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- The main theorem to prove -/
theorem min_packs_for_120_cans :
  ∃ (c : PackCombination),
    totalCans (applyPromotion c) = 120 ∧
    totalPacks c = 4 ∧
    (∀ (c' : PackCombination),
      totalCans (applyPromotion c') = 120 →
      totalPacks c' ≥ totalPacks c) :=
  sorry


end NUMINAMATH_CALUDE_min_packs_for_120_cans_l3415_341564


namespace NUMINAMATH_CALUDE_multiplication_mistake_l3415_341514

theorem multiplication_mistake (x : ℝ) : 973 * x - 739 * x = 110305 → x = 471.4 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_l3415_341514


namespace NUMINAMATH_CALUDE_translation_of_complex_plane_l3415_341558

theorem translation_of_complex_plane (t : ℂ → ℂ) :
  (t (1 + 3*I) = 4 - 2*I) →
  (∃ w : ℂ, ∀ z : ℂ, t z = z + w) →
  (t (2 - I) = 5 - 6*I) := by
sorry

end NUMINAMATH_CALUDE_translation_of_complex_plane_l3415_341558


namespace NUMINAMATH_CALUDE_knowledge_competition_probabilities_l3415_341515

/-- Represents the types of questions available in the competition -/
inductive QuestionType
  | Easy1
  | Easy2
  | Medium
  | Hard

/-- The point value associated with each question type -/
def pointValue : QuestionType → ℕ
  | QuestionType.Easy1 => 10
  | QuestionType.Easy2 => 10
  | QuestionType.Medium => 20
  | QuestionType.Hard => 40

/-- The probability of selecting each question type -/
def selectionProbability : QuestionType → ℚ
  | _ => 1/4

theorem knowledge_competition_probabilities :
  let differentValueProb := 1 - (2/4 * 2/4 + 1/4 * 1/4 + 1/4 * 1/4)
  let greaterValueProb := 1/4 * 2/4 + 1/4 * 3/4
  differentValueProb = 5/8 ∧ greaterValueProb = 5/16 := by
  sorry

#check knowledge_competition_probabilities

end NUMINAMATH_CALUDE_knowledge_competition_probabilities_l3415_341515


namespace NUMINAMATH_CALUDE_judy_hits_percentage_l3415_341537

theorem judy_hits_percentage (total_hits : ℕ) (home_runs : ℕ) (triples : ℕ) (doubles : ℕ)
  (h_total : total_hits = 35)
  (h_home : home_runs = 1)
  (h_triple : triples = 1)
  (h_double : doubles = 5) :
  (total_hits - (home_runs + triples + doubles)) / total_hits = 4/5 := by
sorry

end NUMINAMATH_CALUDE_judy_hits_percentage_l3415_341537


namespace NUMINAMATH_CALUDE_inequality_proof_l3415_341519

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3415_341519


namespace NUMINAMATH_CALUDE_cat_bowl_refill_days_l3415_341583

theorem cat_bowl_refill_days (empty_bowl_weight : ℝ) (daily_food : ℝ) (weight_after_eating : ℝ) (eaten_amount : ℝ) :
  empty_bowl_weight = 420 →
  daily_food = 60 →
  weight_after_eating = 586 →
  eaten_amount = 14 →
  (weight_after_eating + eaten_amount - empty_bowl_weight) / daily_food = 3 := by
  sorry

end NUMINAMATH_CALUDE_cat_bowl_refill_days_l3415_341583


namespace NUMINAMATH_CALUDE_characterization_of_m_l3415_341512

-- Define a good integer
def is_good (n : ℤ) : Prop :=
  ¬ ∃ k : ℤ, n.natAbs = k^2

-- Define the property for m
def has_property (m : ℤ) : Prop :=
  ∀ N : ℕ, ∃ a b c : ℤ,
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (is_good a ∧ is_good b ∧ is_good c) ∧
    (a + b + c = m) ∧
    (∃ k : ℤ, a * b * c = (2*k + 1)^2) ∧
    (N < a.natAbs ∧ N < b.natAbs ∧ N < c.natAbs)

-- The main theorem
theorem characterization_of_m (m : ℤ) :
  has_property m ↔ m % 4 = 3 :=
sorry

end NUMINAMATH_CALUDE_characterization_of_m_l3415_341512


namespace NUMINAMATH_CALUDE_six_distinct_areas_l3415_341505

/-- Represents a point in a one-dimensional space -/
structure Point1D where
  x : ℝ

/-- Represents a line in a two-dimensional space -/
structure Line2D where
  points : List Point1D
  y : ℝ

/-- The configuration of points as described in the problem -/
structure PointConfiguration where
  line1 : Line2D
  line2 : Line2D
  w : Point1D
  x : Point1D
  y : Point1D
  z : Point1D
  p : Point1D
  q : Point1D

/-- Checks if the configuration satisfies the given conditions -/
def validConfiguration (config : PointConfiguration) : Prop :=
  config.w.x < config.x.x ∧ config.x.x < config.y.x ∧ config.y.x < config.z.x ∧
  config.x.x - config.w.x = 1 ∧
  config.y.x - config.x.x = 2 ∧
  config.z.x - config.y.x = 3 ∧
  config.q.x - config.p.x = 4 ∧
  config.line1.y ≠ config.line2.y ∧
  config.line1.points = [config.w, config.x, config.y, config.z] ∧
  config.line2.points = [config.p, config.q]

/-- Calculates the number of possible distinct triangle areas -/
def distinctTriangleAreas (config : PointConfiguration) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly 6 possible distinct triangle areas -/
theorem six_distinct_areas (config : PointConfiguration) 
  (h : validConfiguration config) : distinctTriangleAreas config = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_distinct_areas_l3415_341505


namespace NUMINAMATH_CALUDE_vector_magnitude_l3415_341534

/-- Given two planar vectors a and b, prove that |2a - b| = 2√3 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (a.1 = 3/5 ∧ a.2 = -4/5) →  -- Vector a = (3/5, -4/5)
  (Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = 1) →  -- |a| = 1
  (Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 2) →  -- |b| = 2
  (a.1 * b.1 + a.2 * b.2 = -1) →  -- a · b = -1 (dot product for 120° angle)
  Real.sqrt (((2 * a.1 - b.1) ^ 2) + ((2 * a.2 - b.2) ^ 2)) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3415_341534


namespace NUMINAMATH_CALUDE_intersection_range_l3415_341503

theorem intersection_range (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : b ≠ 0) :
  (∃ x y : ℝ, a * x + b * y = 2 ∧ x^2 / 6 + y^2 / 2 = 1) →
  a / b ∈ Set.Iic (-1) ∪ Set.Ici 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_range_l3415_341503


namespace NUMINAMATH_CALUDE_probability_to_reach_target_is_correct_l3415_341511

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a possible move direction -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of each direction -/
def directionProbability : ℚ := 1 / 4

/-- The number of steps allowed -/
def numberOfSteps : ℕ := 6

/-- The starting point -/
def startPoint : Point := ⟨0, 0⟩

/-- The target point -/
def targetPoint : Point := ⟨3, 1⟩

/-- Function to calculate the probability of reaching the target point -/
def probabilityToReachTarget (start : Point) (target : Point) (steps : ℕ) : ℚ :=
  sorry

theorem probability_to_reach_target_is_correct :
  probabilityToReachTarget startPoint targetPoint numberOfSteps = 45 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_to_reach_target_is_correct_l3415_341511


namespace NUMINAMATH_CALUDE_cube_volume_problem_l3415_341557

theorem cube_volume_problem (x : ℝ) (h : x > 0) :
  (x - 2) * x * (x + 2) = x^3 - 10 → x^3 = 15.625 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l3415_341557


namespace NUMINAMATH_CALUDE_parentheses_removal_l3415_341569

theorem parentheses_removal (a b c : ℝ) : a - (b - c) = a - b + c := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l3415_341569


namespace NUMINAMATH_CALUDE_cos_sin_identity_l3415_341576

theorem cos_sin_identity :
  Real.cos (40 * π / 180) * Real.cos (160 * π / 180) + Real.sin (40 * π / 180) * Real.sin (20 * π / 180) = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_identity_l3415_341576


namespace NUMINAMATH_CALUDE_system_solutions_l3415_341504

/-- The system of equations -/
def satisfies_system (a b c : ℝ) : Prop :=
  a^5 = 5*b^3 - 4*c ∧ b^5 = 5*c^3 - 4*a ∧ c^5 = 5*a^3 - 4*b

/-- The set of solutions -/
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (1, 1, 1), (-1, -1, -1), (2, 2, 2), (-2, -2, -2)}

/-- The main theorem -/
theorem system_solutions :
  ∀ (a b c : ℝ), satisfies_system a b c ↔ (a, b, c) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l3415_341504


namespace NUMINAMATH_CALUDE_brownie_cost_l3415_341509

/-- The cost of each brownie at Tamara's bake sale -/
theorem brownie_cost (total_revenue : ℚ) (num_pans : ℕ) (pieces_per_pan : ℕ) 
  (h1 : total_revenue = 32)
  (h2 : num_pans = 2)
  (h3 : pieces_per_pan = 8) :
  total_revenue / (num_pans * pieces_per_pan) = 2 := by
  sorry

end NUMINAMATH_CALUDE_brownie_cost_l3415_341509


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_leq_neg_three_l3415_341573

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem decreasing_f_implies_a_leq_neg_three :
  ∀ a : ℝ, (∀ x y : ℝ, x < y ∧ y ≤ 4 → f a x > f a y) → a ≤ -3 := by sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_leq_neg_three_l3415_341573


namespace NUMINAMATH_CALUDE_log_function_fixed_point_l3415_341533

-- Define the set of valid 'a' values
def ValidA := {a : ℝ | a ∈ Set.Ioo 0 1 ∪ Set.Ioi 1}

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a + 2

-- Theorem statement
theorem log_function_fixed_point (a : ℝ) (ha : a ∈ ValidA) :
  f a 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_log_function_fixed_point_l3415_341533


namespace NUMINAMATH_CALUDE_subway_distance_difference_l3415_341525

def distance (s : ℝ) : ℝ := 0.5 * s^3 + s^2

theorem subway_distance_difference : 
  distance 7 - distance 4 = 172.5 := by sorry

end NUMINAMATH_CALUDE_subway_distance_difference_l3415_341525


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l3415_341535

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * Real.log x

theorem derivative_f_at_one :
  deriv f 1 = Real.exp 1 + 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l3415_341535


namespace NUMINAMATH_CALUDE_cat_whiskers_relationship_l3415_341581

theorem cat_whiskers_relationship (princess_puff_whiskers catman_do_whiskers : ℕ) 
  (h1 : princess_puff_whiskers = 14) 
  (h2 : catman_do_whiskers = 22) : 
  (catman_do_whiskers - princess_puff_whiskers = 8) ∧ 
  (catman_do_whiskers : ℚ) / (princess_puff_whiskers : ℚ) = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cat_whiskers_relationship_l3415_341581


namespace NUMINAMATH_CALUDE_iron_conducts_electricity_l3415_341554

-- Define the universe of discourse
variable (Object : Type)

-- Define predicates
variable (is_metal : Object → Prop)
variable (conducts_electricity : Object → Prop)

-- Define iron as a constant
variable (iron : Object)

-- Theorem statement
theorem iron_conducts_electricity 
  (all_metals_conduct : ∀ x, is_metal x → conducts_electricity x) 
  (iron_is_metal : is_metal iron) : 
  conducts_electricity iron := by
  sorry

end NUMINAMATH_CALUDE_iron_conducts_electricity_l3415_341554


namespace NUMINAMATH_CALUDE_apple_count_l3415_341587

theorem apple_count (initial_oranges : ℕ) (removed_oranges : ℕ) (apples : ℕ) : 
  initial_oranges = 23 →
  removed_oranges = 13 →
  apples = (initial_oranges - removed_oranges) →
  apples = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_count_l3415_341587


namespace NUMINAMATH_CALUDE_cn_relation_sqrt_c_equals_c8_l3415_341547

-- Define cn as a function that returns a natural number with n ones
def cn (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

-- Define the relation between cn and cn+1
theorem cn_relation (n : ℕ) : cn (n + 1) = 10 * cn n + 1 := by sorry

-- Define c
def c : ℕ := 123456787654321

-- Theorem to prove
theorem sqrt_c_equals_c8 : ∃ (x : ℕ), x * x = c ∧ x = cn 8 := by sorry

end NUMINAMATH_CALUDE_cn_relation_sqrt_c_equals_c8_l3415_341547


namespace NUMINAMATH_CALUDE_triangular_square_triangular_l3415_341589

/-- Definition of triangular number -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Statement: 1 and 6 are the only triangular numbers whose squares are also triangular numbers -/
theorem triangular_square_triangular :
  ∀ n : ℕ, (∃ m : ℕ, (triangular n)^2 = triangular m) ↔ n = 1 ∨ n = 3 := by
sorry

end NUMINAMATH_CALUDE_triangular_square_triangular_l3415_341589


namespace NUMINAMATH_CALUDE_dad_age_is_36_l3415_341591

-- Define the current ages
def talia_age : ℕ := 13
def mom_age : ℕ := 39
def dad_age : ℕ := 36
def grandpa_age : ℕ := 18

-- Define the theorem
theorem dad_age_is_36 :
  (talia_age + 7 = 20) ∧
  (mom_age = 3 * talia_age) ∧
  (dad_age + 2 = grandpa_age + 2 + 5) ∧
  (dad_age + 3 = mom_age) ∧
  (grandpa_age + 3 = (mom_age + 3) / 2) →
  dad_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_dad_age_is_36_l3415_341591


namespace NUMINAMATH_CALUDE_james_total_earnings_l3415_341550

def january_earnings : ℕ := 4000

def february_earnings : ℕ := 2 * january_earnings

def march_earnings : ℕ := february_earnings - 2000

def total_earnings : ℕ := january_earnings + february_earnings + march_earnings

theorem james_total_earnings : total_earnings = 18000 := by
  sorry

end NUMINAMATH_CALUDE_james_total_earnings_l3415_341550


namespace NUMINAMATH_CALUDE_maria_paper_count_l3415_341538

/-- Represents the number of sheets of paper -/
structure PaperCount where
  whole : ℕ
  half : ℕ

/-- Calculates the remaining papers after giving away and folding -/
def remaining_papers (desk : ℕ) (backpack : ℕ) (given_away : ℕ) (folded : ℕ) : PaperCount :=
  { whole := desk + backpack - given_away - folded,
    half := folded }

theorem maria_paper_count : 
  ∀ (x y : ℕ), x ≤ 91 → y ≤ 91 - x → 
  remaining_papers 50 41 x y = { whole := 91 - x - y, half := y } := by
sorry

end NUMINAMATH_CALUDE_maria_paper_count_l3415_341538


namespace NUMINAMATH_CALUDE_father_age_l3415_341561

/-- Represents the ages of a family -/
structure FamilyAges where
  yy : ℕ
  cousin : ℕ
  mother : ℕ
  father : ℕ

/-- Defines the conditions of the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.yy = ages.cousin + 3 ∧
  ages.father = ages.mother + 4 ∧
  ages.yy + ages.cousin + ages.mother + ages.father = 95 ∧
  (ages.yy - 8) + (ages.cousin - 8) + (ages.mother - 8) + (ages.father - 8) = 65

/-- The theorem to be proved -/
theorem father_age (ages : FamilyAges) :
  problem_conditions ages → ages.father = 42 := by
  sorry

end NUMINAMATH_CALUDE_father_age_l3415_341561


namespace NUMINAMATH_CALUDE_f_neg_two_f_is_even_l3415_341516

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2 * abs x + 2

-- Define the domain
def domain : Set ℝ := {x : ℝ | -5 ≤ x ∧ x ≤ 5}

-- Theorem 1: f(-2) = 10
theorem f_neg_two : f (-2) = 10 := by sorry

-- Theorem 2: f is an even function on the domain
theorem f_is_even : ∀ x ∈ domain, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_f_neg_two_f_is_even_l3415_341516


namespace NUMINAMATH_CALUDE_common_root_pairs_l3415_341585

theorem common_root_pairs (n : ℕ) (hn : n > 1) :
  ∀ (a b : ℤ), (∃ (x : ℝ), x^n + a*x - 2008 = 0 ∧ x^n + b*x - 2009 = 0) ↔
    ((a = 2007 ∧ b = 2008) ∨ (a = (-1)^(n-1) - 2008 ∧ b = (-1)^(n-1) - 2009)) :=
by sorry

end NUMINAMATH_CALUDE_common_root_pairs_l3415_341585


namespace NUMINAMATH_CALUDE_simplified_expression_terms_l3415_341522

/-- The number of terms in the simplified expression of (x+y+z)^2006 + (x-y-z)^2006 -/
def num_terms : ℕ := 1008016

/-- The exponent used in the expression -/
def exponent : ℕ := 2006

theorem simplified_expression_terms :
  num_terms = (exponent / 2 + 1)^2 :=
sorry

end NUMINAMATH_CALUDE_simplified_expression_terms_l3415_341522


namespace NUMINAMATH_CALUDE_city_population_problem_l3415_341523

theorem city_population_problem (p : ℝ) : 
  (0.85 * (p + 1500) = p + 50) → p = 1500 := by
  sorry

end NUMINAMATH_CALUDE_city_population_problem_l3415_341523


namespace NUMINAMATH_CALUDE_f_neg_five_eq_twelve_l3415_341592

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem f_neg_five_eq_twelve : f (-5) = 12 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_five_eq_twelve_l3415_341592


namespace NUMINAMATH_CALUDE_min_value_on_circle_l3415_341568

theorem min_value_on_circle (x y : ℝ) : 
  (x - 1)^2 + (y - 2)^2 = 9 → y ≥ 2 → x + Real.sqrt 3 * y ≥ 2 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l3415_341568


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_for_given_triangle_l3415_341580

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- Distance from vertex A to side BC -/
  h_a : ℝ
  /-- Sum of distances from B to AC and from C to AB -/
  h_b_plus_h_c : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The radius satisfies the relationship with heights -/
  radius_height_relation : 1 / r = 1 / h_a + 2 / h_b_plus_h_c

/-- The theorem stating the radius of the inscribed circle for the given triangle -/
theorem inscribed_circle_radius_for_given_triangle :
  ∀ (t : TriangleWithInscribedCircle),
    t.h_a = 100 ∧ t.h_b_plus_h_c = 300 →
    t.r = 300 / 7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_for_given_triangle_l3415_341580


namespace NUMINAMATH_CALUDE_min_value_expression_l3415_341559

/-- Given positive real numbers m and n, vectors a and b, where a is parallel to b,
    prove that the minimum value of 1/m + 2/n is 3 + 2√2 -/
theorem min_value_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (a b : Fin 2 → ℝ) 
  (ha : a = ![m, 1]) 
  (hb : b = ![1-n, 1]) 
  (h_parallel : ∃ (k : ℝ), a = k • b) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 2/y ≥ 1/m + 2/n) → 
  1/m + 2/n = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3415_341559


namespace NUMINAMATH_CALUDE_subtract_like_terms_l3415_341560

theorem subtract_like_terms (x y : ℝ) : 5 * x * y - 4 * x * y = x * y := by
  sorry

end NUMINAMATH_CALUDE_subtract_like_terms_l3415_341560


namespace NUMINAMATH_CALUDE_special_sequence_eventually_periodic_l3415_341529

/-- A sequence of positive integers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (a 1 < a 2) ∧
  (∀ n ≥ 3,
    (a n > a (n-1)) ∧
    (∃! (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ n-1 ∧ a n = a i + a j) ∧
    (∀ m < n, (∃ (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ m-1 ∧ a m = a i + a j) → a n > a m))

/-- The set of even numbers in the sequence is finite -/
def FinitelyManyEven (a : ℕ → ℕ) : Prop :=
  ∃ (S : Finset ℕ), ∀ n, Even (a n) → n ∈ S

/-- The sequence of differences is eventually periodic -/
def EventuallyPeriodic (s : ℕ → ℕ) : Prop :=
  ∃ (k p : ℕ), p > 0 ∧ ∀ n ≥ k, s (n + p) = s n

/-- The main theorem -/
theorem special_sequence_eventually_periodic (a : ℕ → ℕ)
  (h1 : SpecialSequence a) (h2 : FinitelyManyEven a) :
  EventuallyPeriodic (fun n => a (n+1) - a n) :=
sorry

end NUMINAMATH_CALUDE_special_sequence_eventually_periodic_l3415_341529


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3415_341552

theorem ellipse_eccentricity (k : ℝ) : 
  (∃ (x y : ℝ), x^2 / (k + 8) + y^2 / 9 = 1) →  -- Ellipse equation
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ (a^2 - b^2) / a^2 = 1/4) →  -- Eccentricity condition
  (k = 4 ∨ k = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3415_341552


namespace NUMINAMATH_CALUDE_power_value_theorem_l3415_341501

theorem power_value_theorem (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : 
  a^(2*m - n) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_power_value_theorem_l3415_341501


namespace NUMINAMATH_CALUDE_barn_paint_area_l3415_341544

/-- Calculates the total area to be painted for a rectangular barn -/
def total_paint_area (width length height : ℝ) : ℝ :=
  let wall_area1 := 2 * width * height
  let wall_area2 := 2 * length * height
  let ceiling_area := width * length
  2 * (wall_area1 + wall_area2) + 2 * ceiling_area

/-- Theorem stating the total area to be painted for the given barn dimensions -/
theorem barn_paint_area :
  total_paint_area 12 15 6 = 1008 := by sorry

end NUMINAMATH_CALUDE_barn_paint_area_l3415_341544


namespace NUMINAMATH_CALUDE_combined_work_time_l3415_341584

/-- The time taken for three people to complete a task together, given their individual rates --/
theorem combined_work_time (rate_shawn rate_karen rate_alex : ℚ) 
  (h_shawn : rate_shawn = 1 / 18)
  (h_karen : rate_karen = 1 / 12)
  (h_alex : rate_alex = 1 / 15) :
  1 / (rate_shawn + rate_karen + rate_alex) = 180 / 37 :=
by sorry

end NUMINAMATH_CALUDE_combined_work_time_l3415_341584


namespace NUMINAMATH_CALUDE_distribution_plans_for_given_conditions_l3415_341563

/-- The number of ways to distribute employees between two departments --/
def distribution_plans (total_employees : ℕ) (translators : ℕ) (programmers : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of distribution plans for the given conditions --/
theorem distribution_plans_for_given_conditions :
  distribution_plans 8 2 3 = 36 :=
sorry

end NUMINAMATH_CALUDE_distribution_plans_for_given_conditions_l3415_341563


namespace NUMINAMATH_CALUDE_waiter_tips_l3415_341598

/-- Calculates the total tips earned by a waiter given the number of customers, 
    number of non-tipping customers, and the tip amount from each tipping customer. -/
def calculate_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Theorem stating that under the given conditions, the waiter earns $32 in tips. -/
theorem waiter_tips : 
  calculate_tips 9 5 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_l3415_341598


namespace NUMINAMATH_CALUDE_flower_pot_cost_l3415_341532

/-- The cost of the largest pot in a set of 6 pots -/
def largest_pot_cost (total_cost : ℚ) (num_pots : ℕ) (price_diff : ℚ) : ℚ :=
  let smallest_pot_cost := (total_cost - (price_diff * (num_pots - 1) * num_pots / 2)) / num_pots
  smallest_pot_cost + price_diff * (num_pots - 1)

/-- Theorem stating the cost of the largest pot given the problem conditions -/
theorem flower_pot_cost :
  largest_pot_cost 8.25 6 0.1 = 1.625 := by
  sorry

end NUMINAMATH_CALUDE_flower_pot_cost_l3415_341532


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l3415_341548

theorem system_of_equations_solutions :
  -- System (1)
  let x₁ := -1
  let y₁ := 1
  -- System (2)
  let x₂ := 5 / 2
  let y₂ := -2
  -- Proof statements
  (x₁ = y₁ - 2 ∧ 3 * x₁ + 2 * y₁ = -1) ∧
  (2 * x₂ - 3 * y₂ = 11 ∧ 4 * x₂ + 5 * y₂ = 0) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l3415_341548


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l3415_341524

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 10) 
  (h2 : sum_first_two = 7) : 
  ∃ a r : ℝ, 
    S = a / (1 - r) ∧ 
    sum_first_two = a + a * r ∧ 
    a = 10 * (1 + Real.sqrt (3 / 10)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l3415_341524


namespace NUMINAMATH_CALUDE_heptagon_foldable_to_quadrilateral_l3415_341543

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A polygon represented by its vertices -/
structure Polygon where
  vertices : List Point2D

/-- A function to check if a polygon is convex -/
def isConvex (p : Polygon) : Prop := sorry

/-- A function to check if a polygon can be folded into a two-layered quadrilateral -/
def canFoldToTwoLayeredQuadrilateral (p : Polygon) : Prop := sorry

/-- Theorem: There exists a convex heptagon that can be folded into a two-layered quadrilateral -/
theorem heptagon_foldable_to_quadrilateral :
  ∃ (h : Polygon), h.vertices.length = 7 ∧ isConvex h ∧ canFoldToTwoLayeredQuadrilateral h := by
  sorry

end NUMINAMATH_CALUDE_heptagon_foldable_to_quadrilateral_l3415_341543


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l3415_341566

theorem circle_equation_k_value (x y k : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 + 4*y - k = 0 ↔ (x + 4)^2 + (y + 2)^2 = 64) → 
  k = 44 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l3415_341566


namespace NUMINAMATH_CALUDE_rigged_coin_probability_l3415_341582

theorem rigged_coin_probability (p : ℝ) (h1 : p < 1/2) 
  (h2 : 20 * p^3 * (1-p)^3 = 1/12) : p = (1 - Real.sqrt 0.86) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rigged_coin_probability_l3415_341582


namespace NUMINAMATH_CALUDE_sum_of_series_equals_half_l3415_341510

theorem sum_of_series_equals_half :
  let series := λ k : ℕ => (3 ^ (2 ^ k)) / ((9 ^ (2 ^ k)) - 1)
  ∑' k, series k = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_half_l3415_341510


namespace NUMINAMATH_CALUDE_earl_stuffing_rate_l3415_341517

-- Define Earl's stuffing rate (envelopes per minute)
def earl_rate : ℝ := sorry

-- Define Ellen's stuffing rate (envelopes per minute)
def ellen_rate : ℝ := sorry

-- Condition 1: Ellen's rate is 2/3 of Earl's rate
axiom rate_relation : ellen_rate = (2/3) * earl_rate

-- Condition 2: Together they stuff 360 envelopes in 6 minutes
axiom combined_rate : earl_rate + ellen_rate = 360 / 6

-- Theorem to prove
theorem earl_stuffing_rate : earl_rate = 36 := by sorry

end NUMINAMATH_CALUDE_earl_stuffing_rate_l3415_341517


namespace NUMINAMATH_CALUDE_star_problem_l3415_341542

-- Define the star operation
def star (a b : ℕ) : ℕ := 3 + b^(a + 1)

-- State the theorem
theorem star_problem : star (star 2 3) 2 = 3 + 2^31 := by
  sorry

end NUMINAMATH_CALUDE_star_problem_l3415_341542
